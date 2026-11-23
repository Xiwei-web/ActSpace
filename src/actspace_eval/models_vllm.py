"""Adapter for local open models: vLLM's OpenAI-compatible server (`vllm serve`, /v1/chat/completions).

- Reuses OpenAIClient: the message structure is identical to the API models (system prompt, multi-turn conversation, images as JPEG base64 data URLs; the "detail" field is still sent and ignored by vLLM);
  only base_url (given by the runner, pointing to the vllm serve in the same job) and parameters change. No OPENAI_API_KEY needed; not billed (price 0), only token usage is recorded.
- Parameters: max_tokens (default 16384, same as API models; when it exceeds the context, narrowed to the remaining space using the input token count returned by the server, and the actual value is recorded); sampling parameters use the model's
  generation_config.json recommendations (otherwise temperature 0.7, top_p 0.9; see sampling and sampling_source in configs/eval.yaml);
  seed is derived from the run seed (0 / 1 / 2), the question group and the call index (sample_seed); response_format = json_object (JSON mode);
  models supporting thinking enable it via chat_template_kwargs (e.g. enable_thinking: true), and the server uses --reasoning-parser to split the reasoning text into
  reasoning_content; logprobs = true, top_logprobs = 5.
- Streaming requests (stream = true, stream_options.include_usage): record time to first token (first chunk with generated content) and total time; servers without
  streaming support (the Cambrian-S wrapper) set stream: false in the config, and time to first token comes from x_timing.ttft_s returned by the server.
- Every call is written to calls.jsonl (ModelClient.complete): the full message structure (raw text; for each image the photo ID, raw frame ID, size and full JPEG SHA-256,
  not the image itself), the full raw reply (content) and reasoning text (reasoning, separate field), finish_reason, input / output / reasoning token counts, time to first token,
  actual max_tokens and seed; turn 0 of each episode also records the fully rendered prompt text (/tokenize + /detokenize, runs of repeated image placeholder tokens folded into
  "<tok>×n"). logprobs go to logprobs.jsonl.gz in the same directory (matched by call_id; token, logprob and top-5 for every generated token, marked as reasoning or reply).
- Context too small / too many images per request for the server: raise a non-retryable ModelLimit (kind = context / images); the runner marks the setting N/A with the reason.
  For multi-image requests the length is checked before images are expanded, and vLLM does not error but squeezes the output into the remaining space: effective_max_tokens records the actual limit; a reply truncated
  with less than min_output_tokens of remaining space therefore also counts as context too small.
- Per-call wall-clock limit (model config call_timeout_s, a number or auto). auto: limit = 3 × 32768 ÷ the model's normal decoding speed,
  computed and updated automatically per model (call_limit.CallLimit); only stops stuck or runaway requests. A timeout counts as truncation (finish_reason = length, recorded
  timeout_truncated = true), scored like hitting max_tokens, not retried. The Cambrian-S wrapper times on the server (the request carries x_time_limit_s, counted from when the server starts
  processing the request, excluding serial queueing); vLLM times on the client (from sending the request), closes the stream on timeout, and vLLM then aborts the request.
- Server errors: HTTP errors, error chunks in the stream, a stream ending without finish_reason, no generated tokens, broken connections all raise
  a retryable ModelError (exponential backoff in models.ModelClient); when retries are exhausted the runner marks stopped_server_error, the episode is not written to the trajectory and does not count as an invalid
  reply of the model. Replies with finish_reason length or stop but empty content come from the model (reasoning filled the limit, or no answer after reasoning) and still count as no answer.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
import uuid

from .models import Completion, ModelError, OpenAIClient, encode_jpeg

_LP_LOCK = threading.Lock()


class ModelLimit(ModelError):
    """Model capability insufficient (context, images per request): not retryable; the setting is marked N/A."""

    def __init__(self, msg, kind, status=400):
        super().__init__(msg, retryable=False, status=status)
        self.kind = kind


def sample_seed(run_seed: int, group_id: str | None, t: int | None) -> int:
    """Sampling seed of one call: derived from the run seed (0 / 1 / 2), the question group and the call index; stable across processes, 31 bits."""
    h = hashlib.sha256(f"sample:{run_seed}:{group_id}:{t}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


_CTX_RE = re.compile(r"maximum (?:context|model) length is (\d+).*?(\d+) (?:input|prompt) tokens", re.S | re.I)
_CTX_RE3 = re.compile(r"Input length \((\d+)\) exceeds model's maximum context length \((\d+)\)", re.I)   # multimodal: check after image expansion
_CTX_RE2 = re.compile(r"(?:prompt|input)[^0-9]{0,40}?(\d+) tokens?.{0,80}?(?:max(?:imum)?[_ ](?:model|context)[_ ]len(?:gth)?)[^0-9]{0,20}(\d+)", re.S | re.I)


def parse_context_error(msg: str):
    """vLLM context-overflow error → (max_model_len, prompt_tokens); None for other errors."""
    m = _CTX_RE.search(msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _CTX_RE3.search(msg)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = _CTX_RE2.search(msg)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None


def is_image_limit_error(msg: str) -> bool:
    return bool(re.search(r"(at most|limit|maximum).{0,60}image|image.{0,60}(at most|limit|exceed)", msg, re.I))


def fold_repeats(tokens: list[str], min_run=4) -> str:
    """Fold runs of the same token repeated ≥ min_run times into "<tok>×n" (image placeholder tokens often number in the hundreds or thousands)."""
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        out.append(f"{tokens[i]}×{j - i}" if j - i >= min_run else "".join(tokens[i:j]))
        i = j
    return "".join(out)


class VLLMClient(OpenAIClient):
    provider = "vllm"

    def __init__(self, mcfg, llm_cfg, cost=None, log_path=None, sleep=time.sleep, transport=None, smoke=False,
                 base_url=None, lp_path=None, http=None, call_limit=None):
        # skip the real-call switch checks of OpenAIClient.__init__: local models need no key and cost nothing
        super(OpenAIClient, self).__init__(mcfg, llm_cfg, cost, log_path, sleep)
        self.transport = transport              # injected by tests: callable(path, body) → dict (non-streaming) or list of chunks (streaming)
        self.http = http                        # injected by tests: callable(path, body) → dict (/tokenize, /detokenize)
        self.cache_key = None
        self.base_url = (base_url or mcfg.get("base_url") or "").rstrip("/")
        if transport is None and not self.base_url:
            raise ValueError("the vLLM adapter needs base_url (runner --base-url or config base_url)")
        self.lp_path = lp_path or (self.log_path.parent / "logprobs.jsonl.gz" if self.log_path else None)
        self.run_seed = 0
        self.frame_of = None                    # photo ID → raw frame ID (set per episode by the runner; only logged, never sent to the model)
        self.group_id = None
        self.call_limit = call_limit            # call_limit.CallLimit (given by the runner when call_timeout_s: auto)

    # ------------------------------------------------------------------ parameters
    def params(self, max_tokens=None, seed=None) -> dict:
        m = self.mcfg
        p = {"model": m["model"]}
        if max_tokens != "auto":      # "auto": no max_tokens, vLLM uses max_model_len − input length
            p["max_tokens"] = int(max_tokens or m["max_output_tokens"])
        p.update({k: v for k, v in (m.get("sampling") or {}).items() if v is not None})
        if seed is not None:
            p["seed"] = seed
        if m.get("json_mode"):
            p["response_format"] = {"type": "json_object"}
        if m.get("chat_template_kwargs"):
            p["chat_template_kwargs"] = dict(m["chat_template_kwargs"])
        if m.get("logprobs_top"):
            p["logprobs"], p["top_logprobs"] = True, int(m["logprobs_top"])
        if m.get("stream", True):
            p["stream"], p["stream_options"] = True, {"include_usage": True}
        if m.get("server") == "cambrian" and self.limit_s():
            p["x_time_limit_s"] = float(self.limit_s())     # timed on the server
        return p

    def limit_s(self):
        """Wall-clock limit of this call (seconds), None if there is no limit."""
        v = self.mcfg.get("call_timeout_s")
        if v == "auto":
            return self.call_limit.limit_s() if self.call_limit is not None else None
        return float(v) if v else None

    # ------------------------------------------------------------------ transport
    def _url(self, path):
        return self.base_url + path

    def _http_json(self, path, body, timeout=600):
        if self.http is not None:
            return self.http(path, body)
        import requests
        try:
            r = requests.post(self._url(path), json=body, timeout=(10, timeout))
        except requests.exceptions.RequestException as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        if r.status_code != 200:
            raise ModelError(f"HTTP {r.status_code}: {r.text[:500]}", retryable=True, status=r.status_code)
        return r.json()

    def _send(self, body: dict):
        """→ (raw result dict, time to first token in s or None). Streaming requests merge chunks into the same structure as non-streaming ones."""
        t0 = time.time()
        if self.transport is not None:
            out = self.transport("/v1/chat/completions", body)
            if isinstance(out, dict) and "_http_status" in out:   # tests: simulated server error
                self._raise_http(out["_http_status"], out["_text"])
            if body.get("stream"):
                return merge_chunks(out, t0, fake=True)
            return out, (out.get("x_timing") or {}).get("ttft_s")
        import requests
        limit = self.limit_s() if body.get("stream") else None    # vLLM: timed on the client
        read_to = self.mcfg.get("read_timeout_s", 900)
        if limit:
            read_to = min(read_to, float(limit))
        try:
            r = requests.post(self._url("/v1/chat/completions"), json=body, stream=bool(body.get("stream")),
                              timeout=(10, read_to))
        except requests.exceptions.ReadTimeout as e:
            if limit:
                return timed_out([], t0)
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        except requests.exceptions.RequestException as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        if r.status_code != 200:
            self._raise_http(r.status_code, r.text)
        if not body.get("stream"):
            out = r.json()
            return out, (out.get("x_timing") or {}).get("ttft_s")
        chunks = []
        try:
            for line in r.iter_lines(decode_unicode=True):
                if limit and time.time() - t0 >= float(limit):
                    r.close()                       # close the connection; vLLM aborts the request
                    return timed_out(chunks, t0)
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunks.append((time.time(), json.loads(data)))
        except requests.exceptions.ReadTimeout as e:
            if limit:
                return timed_out(chunks, t0)
            raise ModelError(f"stream {type(e).__name__}: {e}", retryable=True) from None
        except (requests.exceptions.RequestException, ValueError) as e:
            raise ModelError(f"stream {type(e).__name__}: {e}", retryable=True) from None
        return merge_chunks(chunks, t0)

    def _raise_http(self, status, text):
        msg = text[:1500]
        try:
            msg = json.loads(text).get("error", {}).get("message", msg) if text.strip().startswith("{") else msg
        except (ValueError, AttributeError):
            pass
        ctx = parse_context_error(str(msg))
        if status == 400 and ctx:
            raise ModelLimit(f"context: {msg}", "context") from None
        if status == 400 and is_image_limit_error(str(msg)):
            raise ModelLimit(f"images: {msg}", "images") from None
        raise ModelError(f"HTTP {status}: {msg}", retryable=True, status=status)   # server errors are always retried; stopped_server_error once exhausted

    # ------------------------------------------------------------------ request
    def to_messages_logged(self, system, turns):
        """(messages sent to the server, message structure for the log): for each image the log records photo ID, raw frame ID, size, JPEG byte count and full SHA-256."""
        detail = self.mcfg.get("image_detail", "auto")
        import base64
        msgs, logged = [{"role": "system", "content": system}], [{"role": "system", "content": system}]
        for t in turns:
            if "parts" not in t:
                msgs.append({"role": t["role"], "content": t["text"]})
                logged.append({"role": t["role"], "content": t["text"]})
                continue
            content, lc = [], []
            for p in t["parts"]:
                if p[0] == "text":
                    content.append({"type": "text", "text": p[1]})
                    lc.append({"type": "text", "text": p[1]})
                    continue
                jpg = encode_jpeg(p[2])
                url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
                content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                h, w = p[2].shape[:2]
                lc.append({"type": "image", "photo_id": p[1], "frame_id": self.frame_of(p[1]) if self.frame_of else None,
                           "w": w, "h": h, "jpeg_bytes": len(jpg), "sha256": hashlib.sha256(jpg).hexdigest()})
            msgs.append({"role": t["role"], "content": content})
            logged.append({"role": t["role"], "content": lc})
        return msgs, logged

    def rendered_prompt(self, msgs) -> dict:
        """Fully rendered prompt (chat template applied, image placeholders expanded): /tokenize gives token ids (same preprocessing as generation),
        runs of the same token repeated ≥ 4 times (image placeholders) are folded into "<tok>×n", and the other segments are turned back into text with /detokenize."""
        body = {"model": self.mcfg["model"], "messages": msgs, "add_generation_prompt": True, "return_token_strs": True}
        if self.mcfg.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = dict(self.mcfg["chat_template_kwargs"])
        try:
            tok = self._http_json("/tokenize", body)
            ids, strs = tok["tokens"], tok.get("token_strs")
            parts, i = [], 0
            while i < len(ids):
                j = i
                while j < len(ids) and ids[j] == ids[i]:
                    j += 1
                if j - i >= 4:
                    name = strs[i] if strs else self._http_json("/detokenize", {"model": self.mcfg["model"], "tokens": [ids[i]]})["prompt"]
                    parts.append(f"{name}×{j - i}")
                    i = j
                    continue
                k = i     # text segment: up to the next token repeated ≥ 4 times
                while k < len(ids):
                    r = k
                    while r < len(ids) and ids[r] == ids[k]:
                        r += 1
                    if r - k >= 4:
                        break
                    k = r
                parts.append(self._http_json("/detokenize", {"model": self.mcfg["model"], "tokens": ids[i:k]})["prompt"])
                i = k
        except (ModelError, KeyError) as e:
            return {"error": str(e)[:500]}
        return {"count": tok.get("count", len(ids)), "max_model_len": tok.get("max_model_len"), "text": "".join(parts)}

    def _request(self, system, turns):
        tag = getattr(self, "_tag", None) or {}
        seed = sample_seed(self.run_seed, tag.get("episode"), tag.get("t"))
        msgs, logged = self.to_messages_logged(system, turns)
        max_tokens = int(self.mcfg["max_output_tokens"])
        clamped = None
        ctx_len = (self.mcfg.get("vllm") or {}).get("max_model_len")
        if ctx_len and max_tokens >= ctx_len:
            # when max_tokens is not smaller than the context, vLLM computes an input space of 0 and rejects every request: omit max_tokens and let the server use the remaining space
            clamped = {"from": max_tokens, "to": "auto", "max_model_len": ctx_len}
            max_tokens = "auto"
        for _ in range(2):
            params = self.params(max_tokens, seed)
            if tag.get("t") == 0 and not params.get("stream"):
                params["return_prompt_text"] = True     # the server wrapping the official code has no /tokenize; the rendered prompt comes back with the reply
            try:
                limit_used = self.limit_s()
                t_send = time.time()
                raw, ttft = self._send({**params, "messages": msgs})
                t_total = time.time() - t_send
                break
            except ModelLimit as e:
                ctx = parse_context_error(str(e))
                if e.kind != "context" or not ctx or clamped is not None:
                    raise
                max_len, n_in = ctx
                min_out = int(self.mcfg.get("min_output_tokens", 256))
                if "at least" in str(e) or n_in == 0:     # n_in == 0: "upper bound for 0 input tokens", not an input length
                    # input was counted after truncation, true length unknown: omit max_tokens and let vLLM use the remaining space (max_model_len − input)
                    clamped = {"from": max_tokens, "to": "auto", "prompt_tokens_at_least": n_in, "max_model_len": max_len}
                    max_tokens = "auto"
                    continue
                room = max_len - n_in
                if room < min_out:
                    raise ModelLimit(f"context: prompt {n_in} tokens leaves {room} of max_model_len {max_len} "
                                     f"(< min_output_tokens {min_out})", "context") from None
                clamped = {"from": max_tokens, "to": room, "prompt_tokens": n_in, "max_model_len": max_len}
                max_tokens = room
        ch = raw["choices"][0]
        msg = ch.get("message") or {}
        u = raw.get("usage") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        timeout = bool(raw.get("x_timeout_truncated"))
        if timeout:
            ch["finish_reason"] = "length"      # a timeout counts as truncation
        if ch.get("finish_reason") is None:
            raise ModelError(f"server error: reply ended without finish_reason (usage {u})", retryable=True)
        if not timeout and not content and not reasoning and not u.get("completion_tokens"):
            raise ModelError(f"server error: no tokens generated (finish_reason {ch.get('finish_reason')}, usage {u})", retryable=True)
        lp = (ch.get("logprobs") or {}).get("content") or []
        n_reason_lp = sum(1 for x in lp if x.get("_reasoning"))
        ctd = u.get("completion_tokens_details") or {}
        reasoning_tokens = ctd.get("reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = n_reason_lp if n_reason_lp else (None if reasoning else 0)
        usage = {"input_tokens": u.get("prompt_tokens", 0), "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                 "output_tokens": u.get("completion_tokens", 0), "reasoning_tokens": reasoning_tokens or 0}
        call_id = uuid.uuid4().hex
        if lp and self.lp_path is not None:
            self._write_logprobs(call_id, tag, lp)
        if max_tokens == "auto" and clamped:
            clamped["prompt_tokens"] = usage["input_tokens"]
            max_tokens = clamped["max_model_len"] - usage["input_tokens"]
        max_len = (self.mcfg.get("vllm") or {}).get("max_model_len") or self.mcfg.get("max_model_len")
        effective = min(max_tokens, max_len - usage["input_tokens"]) if max_len else max_tokens
        min_out = int(self.mcfg.get("min_output_tokens", 256))
        if not timeout and ch.get("finish_reason") == "length" and effective < min_out:
            raise ModelLimit(f"context: prompt {usage['input_tokens']} tokens leaves {effective} of max_model_len {max_len} "
                             f"(< min_output_tokens {min_out}) and the reply was cut off", "context")
        extra = {"call_id": call_id, "reasoning": reasoning, "reasoning_chars": len(reasoning), "reasoning_tokens": reasoning_tokens,
                 "ttft_s": None if ttft is None else round(ttft, 3), "max_tokens": max_tokens, "effective_max_tokens": effective, "max_tokens_clamped": clamped,
                 "x_image_tokens": raw.get("x_image_tokens"), "x_image_mode": raw.get("x_image_mode"), "seed": seed, "run_seed": self.run_seed, "logprobs_tokens": len(lp), "logprobs_reasoning_tokens": n_reason_lp,
                 "x_json_mode": raw.get("x_json_mode"), "group_id": self.group_id, "timeout_truncated": timeout, "call_timeout_s": limit_used}
        if self.call_limit is not None and not timeout:
            from .call_limit import call_speed
            self.call_limit.record(call_speed(usage["output_tokens"], t_total, ttft, self.call_limit.min_out))
        if tag.get("t") == 0:
            extra["rendered_prompt"] = raw.get("x_prompt_text") and {"text": raw["x_prompt_text"]} or self.rendered_prompt(msgs)
        comp = Completion(text=content, usage=usage, finish_reason=ch.get("finish_reason"),
                          raw={"usage": u, "id": raw.get("id"), "model": raw.get("model")})
        comp.extra = extra
        summary = {"params": {k: v for k, v in params.items() if k != "stream_options"}, "messages": logged}
        return comp, summary

    def _write_logprobs(self, call_id, tag, lp):
        rec = {"call_id": call_id, "tag": tag,
               "token": [x.get("token") for x in lp],
               "logprob": [None if x.get("logprob") is None else round(float(x["logprob"]), 5) for x in lp],
               "top": [[[y.get("token"), None if y.get("logprob") is None else round(float(y["logprob"]), 5)] for y in (x.get("top_logprobs") or [])]
                       for x in lp],
               "reasoning": [bool(x.get("_reasoning")) for x in lp]}
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode()
        with _LP_LOCK:
            with gzip.open(self.lp_path, "ab") as f:   # multi-member gzip, can be appended directly
                f.write(line)


def timed_out(chunks, t0):
    """Client-side timeout: chunks already received are merged as usual, finish_reason is recorded as length, flagged x_timeout_truncated."""
    raw, ttft = merge_chunks(chunks, t0)
    raw["choices"][0]["finish_reason"] = "length"
    raw["x_timeout_truncated"] = True
    return raw, ttft


def merge_chunks(chunks, t0, fake=False):
    """Streaming chunks → (result with the same structure as non-streaming, time to first token). chunks: [(arrival time, chunk dict)]; fake=True for chunk lists injected by tests (no times).
    Reasoning chunks (delta.reasoning_content / delta.reasoning) carry logprobs flagged _reasoning=True."""
    content, reasoning, lps = [], [], []
    finish, usage, rid, model, ttft = None, None, None, None, None
    for item in chunks:
        ts, c = (t0, item) if fake else item
        if c.get("error") or c.get("object") == "error":
            raise ModelError(f"server error in stream: {str(c.get('error') or c.get('message'))[:500]}", retryable=True)
        rid, model = c.get("id", rid), c.get("model", model)
        if c.get("usage"):
            usage = c["usage"]
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            r_part = d.get("reasoning_content") or d.get("reasoning")
            c_part = d.get("content")
            if (r_part or c_part) and ttft is None:
                ttft = ts - t0
            if r_part:
                reasoning.append(r_part)
            if c_part:
                content.append(c_part)
            for x in ((ch.get("logprobs") or {}).get("content") or []):
                lps.append({**x, "_reasoning": bool(r_part) and not c_part})
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    return ({"id": rid, "model": model, "usage": usage or {},
             "choices": [{"index": 0, "message": msg, "finish_reason": finish, "logprobs": {"content": lps}}]}, ttft)
