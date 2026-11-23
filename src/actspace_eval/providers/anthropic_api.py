"""Anthropic adapter: Messages REST API via requests, without the SDK (like models.OpenAIClient; the actspace environment, shared with
the open-weight evaluation and left unchanged, does not have anthropic installed). Interface, retries and cost handling match the
OpenAI adapter and the content is identical word for word; only the wire format differs:

- The system prompt goes into system; the conversation turns are the same; images are the same JPEG bytes (base64 image blocks).
- Thinking: thinking = {type: adaptive, display: summarized}, output_config.effort = target effort (registry reasoning, subject to
  approval). temperature is not sent.
- Assistant messages in the history carry back only text, not thinking blocks: the same condition as for OpenAI (Chat Completions
  has no way to pass reasoning back). The K = 8 window rewrites earlier messages at every step, and passing thinking blocks back
  would trigger the preserved-thinking prefix check (a direct 400 on new accounts).
- Empty assistant messages (previous reply truncated) are not sent: the Messages API does not accept empty text blocks; consecutive
  user messages are merged by the server, without adding any text.
- Prompt caching: a single cache_control (default 5 minutes) only on the START image block of Active turn 0; its prefix = system
  prompt + questions + START image, which never changes. One-shot answering settings get none: they call only once, so nothing is
  ever read and the write price is higher than the input price.
- JSON: structured outputs are used (registry json.mode: schema, output_config.format, the permissive REPLY_SCHEMA shared by all
  settings); the prompts are unchanged word for word, and the server adds format instructions as officially documented. With none,
  only the format requirements already in the prompts apply.
- Keys: the job reads them from the key file, and keys.KeyPool decides which number each call uses; on exhausted quota it switches
  to the next and resends the same call (providers/keys.py).
- No fallbacks (server-side model switching): replies must come from the model under evaluation. Refusals (stop_reason = refusal)
  are recorded as model replies, with stop_details written to the log.
- Streaming requests (registry stream: true): the connection does not go idle during long thinking; time to first token is
  recorded; thinking summaries and usage are merged from the events.
- Usage: input_tokens covers only the part after the last breakpoint, so normalized input = input_tokens + cache_read +
  cache_creation; thinking tokens are in output_tokens_details.thinking_tokens and already included in output_tokens.
"""
from __future__ import annotations

import base64
import json
import re
import time

from ..data import MOVE_ACTIONS
from ..models import ModelClient, ModelError, encode_jpeg
from ..models_vllm import ModelLimit
from .base import ApiClientMixin, lost_reply, read_json, redact
from .keys import KeyQuota, active_pool, quota_error

_TOO_LONG = re.compile(r"prompt is too long|exceed.{0,40}context|input length", re.I)
_TOO_MANY_IMAGES = re.compile(r"too many images|at most \d+ images", re.I)
MAX_QUESTIONS = 5    # merged_v1.0 has at most 5 questions per group
# Reply format for structured outputs: constrains only the fields, not the answer values. additionalProperties must be false (official limit), so answers lists Q1...Qn instead of an object with arbitrary keys.
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["action"], "properties": {
    "action": {"type": "string", "enum": [*MOVE_ACTIONS, "answer"]},
    "answers": {"type": "object", "additionalProperties": False,
                "properties": {f"Q{i}": {"anyOf": [{"type": "string"}, {"type": "number"}]} for i in range(1, MAX_QUESTIONS + 1)}},
    "notes": {"type": "string"}}}


def anthropic_usage(u: dict) -> dict:
    read, write = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
    return {"input_tokens": (u.get("input_tokens") or 0) + read + write, "cached_input_tokens": read, "cache_write_tokens": write,
            "cache_write_12h_tokens": (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0,
            "output_tokens": u.get("output_tokens") or 0,
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0}


def merge_events(events, t0, fake=False) -> tuple[dict, float | None]:
    """Stream events -> (message with the same structure as non-streaming, time to first token). events: [(arrival time, event)]; with fake=True, an event list injected by tests."""
    msg, blocks, ttft, stopped = None, {}, None, False
    for item in events:
        ts, ev = (t0, item) if fake else item
        typ = ev.get("type")
        if typ == "error":
            err = ev.get("error") or {}
            raise lost_reply(f"server error in stream: {err.get('type')}: {str(err.get('message'))[:300]}")
        if typ == "message_start":
            msg = {**ev["message"], "content": [], "usage": dict(ev["message"].get("usage") or {})}
        elif typ == "content_block_start":
            blocks[ev["index"]] = dict(ev.get("content_block") or {})
        elif typ == "content_block_delta":
            d, b = ev.get("delta") or {}, blocks.setdefault(ev["index"], {})
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif d.get("type") == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            if ttft is None and d.get("type") in ("text_delta", "thinking_delta"):
                ttft = ts - t0
        elif typ == "message_delta" and msg is not None:
            msg.update({k: v for k, v in (ev.get("delta") or {}).items()})
            msg["usage"].update(ev.get("usage") or {})
        elif typ == "message_stop":
            stopped = True
    if msg is None or not stopped:
        raise lost_reply("stream ended before message_stop")
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg, ttft


class AnthropicClient(ApiClientMixin, ModelClient):
    provider = "anthropic"

    def __init__(self, mcfg, llm_cfg, cost=None, log_path=None, sleep=time.sleep, transport=None, smoke=False):
        super().__init__(mcfg, llm_cfg, cost, log_path, sleep)
        self.force_cache = False     # probe: set the cache mark on direct calls too
        self._key_no = self._key_value = None
        self._setup(smoke, transport)

    def params(self) -> dict:
        m = self.mcfg
        p = {"model": m["model"], "max_tokens": m["max_output_tokens"]}
        if m["reasoning"].get("enabled"):
            p["thinking"] = {"type": "adaptive"}
            if m["reasoning"].get("display"):
                p["thinking"]["display"] = m["reasoning"]["display"]
            p["output_config"] = {"effort": m["reasoning"]["effort"]}
        if m.get("json_schema"):
            p.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": m.get("reply_schema") or REPLY_SCHEMA}   # reply_schema: cogmap probe of the ablations (providers/ablations.py)
        if m.get("stream"):
            p["stream"] = True
        return p

    def _cache_start(self) -> bool:
        return self.mcfg.get("prompt_cache_mode") == "explicit" and (self.force_cache or (self._tag or {}).get("mode") == "active")

    def to_messages(self, system, turns):
        """-> ((system blocks, messages), {photo ID: jpeg bytes})."""
        jpegs, msgs, mark = {}, [], self._cache_start()
        for t in turns:
            if "parts" not in t:
                if t["text"].strip():
                    msgs.append({"role": t["role"], "content": t["text"]})
                continue
            content = []
            for p in t["parts"]:
                if p[0] == "text":
                    content.append({"type": "text", "text": p[1]})
                    continue
                jpegs.setdefault(p[1], encode_jpeg(p[2]))
                block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpegs[p[1]]).decode()}}
                if mark:     # first image = START image
                    block["cache_control"] = {"type": "ephemeral"}
                    mark = False
                content.append(block)
            msgs.append({"role": t["role"], "content": content})
        return ([{"type": "text", "text": system}], msgs), jpegs

    def _key(self) -> str:
        return self._key_value or super()._key()

    def _secret(self):
        return self._key_value or super()._secret()

    def _raise_http(self, status, text, headers=None):
        q = quota_error(status, text) if active_pool() else None
        if q:
            raise KeyQuota({**q, "message": redact(q["message"], self._secret())})
        if status in (400, 413) and _TOO_LONG.search(text):
            raise ModelLimit(f"context: {redact(text)[:500]}", "context", status=status)
        if status in (400, 413) and _TOO_MANY_IMAGES.search(text):
            raise ModelLimit(f"images: {redact(text)[:500]}", "images", status=status)
        raise self._http_error(status, text, headers)

    def _post(self, body: dict):
        """-> (message, time to first token)."""
        import requests
        t0 = time.time()
        stream = bool(body.get("stream"))
        try:
            r = requests.post(self.mcfg["base_url"], json=body, stream=stream, timeout=(10, self.llm_cfg["request_timeout_s"]),
                              headers={"x-api-key": self._key(), "anthropic-version": self.mcfg["api_version"], "content-type": "application/json"})
        except requests.exceptions.ConnectTimeout as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        except requests.exceptions.RequestException as e:
            raise lost_reply(f"{type(e).__name__}: {e}") from None
        if r.status_code != 200:
            self._raise_http(r.status_code, r.text, r.headers)
        if not stream:
            return read_json(r), None
        events = []
        try:
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    events.append((time.time(), json.loads(line[5:].strip())))
        except (requests.exceptions.RequestException, ValueError) as e:
            raise lost_reply(f"stream {type(e).__name__}: {e}") from None
        return merge_events(events, t0)

    def _send(self, body):
        if self.transport is None:
            return self._post(body)
        t0 = time.time()
        if hasattr(self.transport, "key_no"):       # tests: the fake transport simulates exhausted quota per current key
            self.transport.key_no, self.transport.api_key = self._key_no, self._key_value
        out = self.transport(body)
        if isinstance(out, dict) and "_http_status" in out:    # tests: simulated error returned by the server
            self._raise_http(out["_http_status"], out["_text"], out.get("_headers"))
        return merge_events(out, t0, fake=True) if body.get("stream") else (out, None)

    def _exchange(self, params, wire):
        system, msgs = wire
        body, pool = {**params, "system": system, "messages": msgs}, active_pool()
        while True:
            if pool:
                self._key_no, self._key_value = pool.current()
            try:
                m, ttft = self._send(body)
                break
            except KeyQuota as q:       # this attempt was not billed: switch to the next key and resend the same call
                pool.exhaust(self._key_no, q.info)
        u = m.get("usage")
        if not u or m.get("stop_reason") is None:
            raise lost_reply(f"incomplete response: stop_reason {m.get('stop_reason')}, usage {u}")
        if pool and self.cost is not None:
            pool.charge(self._key_no, self.cost.cost_of(anthropic_usage(u)))
        blocks = m.get("content") or []
        return {"text": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
                "thinking": "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking"),
                "finish": m["stop_reason"], "usage": anthropic_usage(u), "usage_raw": u, "refusal": m.get("stop_details"),
                "request_id": m.get("id"), "model": m.get("model"), "ttft_s": None if ttft is None else round(ttft, 3), "logprobs": [], "key_no": self._key_no}
