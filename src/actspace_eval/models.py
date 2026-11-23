"""Model adapters (docs/evaluation_protocol.md).

- ModelClient: unified interface complete(system, turns, tag) → Completion. Handles retries on failure (exponential backoff), the call log and cost accounting.
  Every call (including every retry) writes one line to calls.jsonl: request summary (images recorded only by ID, size, byte count and SHA-256, no base64),
  raw reply, token usage, cost, latency, retry history. usage_raw is the raw usage object returned by the API (used to check
  billing items not covered by the configured prices, such as cache writes).
- OpenAIClient: Chat Completions REST API (via requests, no openai SDK dependency). The key is read only from the environment variable OPENAI_API_KEY;
  model name, temperature, max output length, reasoning mode etc. all come from the config. Before real calls, three switches are checked: llm.allow_real_calls,
  llm.prompts_approved (turned on only after the prompts were reviewed) and the model's prices_confirmed. Smoke tests (smoke=True) may bypass the first two,
  but may only run on llm.smoke_groups (checked by llm_run); confirmed prices and the key are still required.
  prompt_cache_key: one key per episode (run name + question group), so calls of the same episode share a prefix and hit the cache more often.
  Cache retention: requests do not set prompt_cache_options / prompt_cache_retention and use the default 30-minute tier.
  Cache writes: usage.prompt_tokens_details.cache_write_tokens (field name observed by probe; included in prompt_tokens, billed at the write price
  rather than as uncached input); cache_write_tokens_details.ttl_12h_tokens, if present, is billed at the 12-hour tier price.
- VLLMClient (models_vllm.py): local open models (vLLM's OpenAI-compatible server).
  Subclasses can put extra fields in Completion.extra (reasoning text, time to first token, actual max_tokens, seed, etc.), merged into that call's log entry;
  parse_hook (set by the agent) gives the parse result and failure reason of the reply, also written to the log. Concurrent threads writing the same calls.jsonl take a lock.
- MockClient (models_mock.py): for tests.
- Models in the API model registry configs/models_api.yaml (providers/) go through providers.make_api_client. ModelError may carry retry_after
  (wait in seconds requested by the service) and billed (failed but billed attempts: usage, cost_usd, etc., merged into the error record); behaviour is unchanged without these attributes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


class ModelError(RuntimeError):
    """Call failed. Retried with exponential backoff when retryable is True."""

    def __init__(self, msg, retryable=False, status=None):
        super().__init__(msg)
        self.retryable, self.status = retryable, status


class CallTimeout(ModelError):
    """A single call exceeded the wall-clock limit: not retried, treated as truncated."""

    def __init__(self, msg):
        super().__init__(msg, retryable=False)


class RealCallsDisabled(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    usage: dict            # input_tokens, cached_input_tokens, output_tokens, reasoning_tokens
    cost_usd: float = 0.0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)
    retries: int = 0
    extra: dict = field(default_factory=dict)   # extra log fields (VLLMClient)


_LOG_LOCK = threading.Lock()


def encode_jpeg(img: np.ndarray, quality=90) -> bytes:
    """RGB uint8 → JPEG bytes. Images in frames/ mode (data.JpegImage) return the bytes from the export directly (the cluster's quality-90 encoding)."""
    j = getattr(img, "jpeg", b"")
    if j and quality == 90:
        return j
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def image_summary(photo_id, img, jpeg: bytes | None) -> str:
    if img is None:
        return f"<image {photo_id}: not rendered>"
    h, w = img.shape[:2]
    if jpeg is None:
        return f"<image {photo_id} {w}x{h}>"
    return f"<image {photo_id} {w}x{h} jpeg {len(jpeg)} B sha256:{hashlib.sha256(jpeg).hexdigest()[:16]}>"


def request_summary(system: str, turns: list[dict], params: dict, jpegs: dict | None = None) -> dict:
    """Request summary: full text, image placeholders (no base64) and parameters."""
    msgs = [{"role": "system", "content": system}]
    for t in turns:
        if "parts" in t:
            msgs.append({"role": t["role"], "content": [p[1] if p[0] == "text" else image_summary(p[1], p[2], (jpegs or {}).get(p[1]))
                                                        for p in t["parts"]]})
        else:
            msgs.append({"role": t["role"], "content": t["text"]})
    return {"params": params, "messages": msgs}


def count_images(turns) -> int:
    return sum(p[0] == "image" for t in turns for p in t.get("parts", []))


def estimate_text_tokens(text: str) -> int:
    """Rough estimate: about 4 characters per token for English (tiktoken is not installed; only for cost estimates and mock usage)."""
    return max(1, (len(text) + 3) // 4)


def estimate_input_tokens(system: str, turns: list[dict], tokens_per_image: int) -> int:
    text = system + "".join(p[1] for t in turns for p in t.get("parts", []) if p[0] == "text") + \
        "".join(t.get("text", "") for t in turns)
    return estimate_text_tokens(text) + 4 * (len(turns) + 1) + tokens_per_image * count_images(turns)


def _turn_key(t):
    if "parts" in t:
        return (t["role"], tuple((p[0], p[1]) for p in t["parts"]))
    return (t["role"], t["text"])


def cached_prefix_tokens(prev, cur, tokens_per_image, min_tokens=1024, block=128) -> int:
    """Estimate cache-hit tokens with OpenAI's prefix caching rule: the message prefix identical item by item to the previous request (system prompt + first few messages);
    no hit below min_tokens, otherwise rounded down to a multiple of block. prev / cur are (system, turns); used for estimates and the mock."""
    if prev is None or prev[0] != cur[0]:
        return 0
    n = 0
    for a, b in zip(prev[1], cur[1]):
        if _turn_key(a) != _turn_key(b):
            break
        n += 1
    tok = estimate_input_tokens(cur[0], cur[1][:n], tokens_per_image) - 4 if n else estimate_input_tokens(cur[0], [], 0)
    return 0 if tok < min_tokens else tok // block * block


def openai_usage(u: dict) -> dict:
    """Raw Chat Completions usage → unified usage (input includes cache hits and writes; output includes reasoning)."""
    ptd = u.get("prompt_tokens_details") or {}
    return {"input_tokens": u.get("prompt_tokens", 0),
            "cached_input_tokens": ptd.get("cached_tokens", 0),
            "cache_write_tokens": ptd.get("cache_write_tokens") or 0,
            "cache_write_12h_tokens": (ptd.get("cache_write_tokens_details") or {}).get("ttl_12h_tokens") or 0,
            "output_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}


class ModelClient:
    """Shared retry, logging and cost logic. Subclasses implement _request(system, turns) → Completion (cost not yet computed)."""

    provider = "base"

    def __init__(self, mcfg: dict, llm_cfg: dict, cost=None, log_path: Path | None = None, sleep=time.sleep):
        self.mcfg, self.llm_cfg, self.cost = mcfg, llm_cfg, cost
        self.model_name = mcfg["model"]
        self.log_path = Path(log_path) if log_path else None
        self.sleep = sleep
        self.retry = llm_cfg["retry"]
        self._rng = random.Random(0)
        self.parse_hook = None      # set by the agent: reply text → parse result (written to the call log)
        self._tag = None

    def params(self) -> dict:
        m = self.mcfg
        p = {"model": m["model"], m.get("max_tokens_param", "max_completion_tokens"): m["max_output_tokens"]}
        if m.get("reasoning", {}).get("enabled"):
            p["reasoning_effort"] = m["reasoning"]["effort"]
        else:
            p["temperature"] = m["temperature"]
        if m.get("json_mode"):
            p["response_format"] = {"type": "json_object"}
        return p

    def _log(self, rec: dict):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            with _LOG_LOCK, open(self.log_path, "a") as f:
                f.write(line)

    def complete(self, system: str, turns: list[dict], tag: dict | None = None) -> Completion:
        if self.cost is not None:  # before the call: check the cost cap against the estimate
            self.cost.check_before(estimate_input_tokens(system, turns, self.mcfg["tokens_per_image"]), self.mcfg["max_output_tokens"])
        r = self.retry
        attempt, retries = 0, []
        self._tag = tag
        while True:
            t0 = time.time()
            try:
                comp, summary = self._request(system, turns)
            except ModelError as e:
                wait = min(r["max_delay_s"], r["base_delay_s"] * 2 ** attempt) * (1 + r["jitter"] * self._rng.random())
                wait = max(wait, getattr(e, "retry_after", None) or 0)   # wait requested by the service (retry-after response header)
                rec = {"type": "error", "tag": tag, "attempt": attempt, "error": str(e)[:1500], "status": e.status,
                       "retryable": e.retryable, "latency_s": round(time.time() - t0, 3), **(getattr(e, "billed", None) or {})}
                if getattr(e, "kind", None):
                    rec["limit_kind"] = e.kind
                if not e.retryable or attempt >= r["max_retries"]:
                    self._log({**rec, "gave_up": True})
                    raise
                self._log({**rec, "wait_s": round(wait, 2)})
                retries.append(rec)
                self.sleep(wait)
                attempt += 1
                continue
            comp.retries = attempt
            if self.cost is not None:
                comp.cost_usd = self.cost.add(comp.usage)
            rec = {"type": "call", "tag": tag, "model": self.model_name, "request": summary, "response": comp.text,
                   "finish_reason": comp.finish_reason, "usage": comp.usage, "usage_raw": comp.raw.get("usage"), "cost_usd": comp.cost_usd,
                   "spent_usd": self.cost.spent if self.cost else None, "retries": attempt,
                   "latency_s": round(time.time() - t0, 3), **comp.extra}
            if self.parse_hook is not None:
                rec["parse"] = self.parse_hook(comp.text)
            self._log(rec)
            if self.cost is not None:
                self.cost.check_after()
            return comp

    def _request(self, system, turns):
        raise NotImplementedError


class OpenAIClient(ModelClient):
    provider = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, mcfg, llm_cfg, cost=None, log_path=None, sleep=time.sleep, transport=None, smoke=False):
        super().__init__(mcfg, llm_cfg, cost, log_path, sleep)
        self.transport = transport  # injected by tests; None means real calls via requests
        self.cache_key = None       # set per episode by the runner
        if transport is None:
            if not smoke and not llm_cfg.get("allow_real_calls"):
                raise RealCallsDisabled("llm.allow_real_calls is false: real API calls are not allowed")
            if not smoke and not llm_cfg.get("prompts_approved"):
                raise RealCallsDisabled("llm.prompts_approved is false: prompts have not been reviewed and cannot be used for real calls")
            if not mcfg.get("prices_confirmed"):
                raise RealCallsDisabled(f"prices_confirmed of {mcfg['model']} is false: prices are unconfirmed, cannot be used for real calls")
            if not os.environ.get("OPENAI_API_KEY"):
                raise RealCallsDisabled("environment variable OPENAI_API_KEY is not set")

    def to_messages(self, system, turns):
        """Convert to Chat Completions messages; images become JPEG base64 data URLs. Returns (messages, {photo ID: jpeg bytes})."""
        detail = self.mcfg.get("image_detail", "auto")
        jpegs, msgs = {}, [{"role": "system", "content": system}]
        for t in turns:
            if "parts" not in t:
                msgs.append({"role": t["role"], "content": t["text"]})
                continue
            content = []
            for p in t["parts"]:
                if p[0] == "text":
                    content.append({"type": "text", "text": p[1]})
                else:
                    jpegs.setdefault(p[1], encode_jpeg(p[2]))
                    url = "data:image/jpeg;base64," + base64.b64encode(jpegs[p[1]]).decode()
                    content.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
            msgs.append({"role": t["role"], "content": content})
        return msgs, jpegs

    def _post(self, body: dict) -> dict:
        import requests
        key = os.environ["OPENAI_API_KEY"]  # read only from the environment variable, never logged
        limit = self.call_timeout_s()
        timeout = (10, float(limit)) if limit else self.llm_cfg["request_timeout_s"]
        try:
            resp = requests.post(self.mcfg.get("base_url", self.URL), json=body, timeout=timeout,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except requests.exceptions.ReadTimeout as e:
            if limit:       # exceeding the per-call limit counts as truncation, no retry
                raise CallTimeout(f"timeout_truncated: no reply within {limit} s") from None
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        except requests.exceptions.RequestException as e:  # timeouts, connection errors: retryable
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        if resp.status_code != 200:
            retryable = resp.status_code in (408, 409, 429) or resp.status_code >= 500
            raise ModelError(f"HTTP {resp.status_code}: {resp.text[:300]}", retryable=retryable, status=resp.status_code)
        return resp.json()

    def params(self) -> dict:
        p = super().params()
        if self.mcfg.get("prompt_cache_key") and self.cache_key:
            p["prompt_cache_key"] = self.cache_key
        return p

    def call_timeout_s(self):
        """Per-call wall-clock limit: the model's call_timeout_s, or llm.call_timeout_s if absent."""
        return self.mcfg.get("call_timeout_s", self.llm_cfg.get("call_timeout_s"))

    def _request(self, system, turns):
        params = self.params()
        msgs, jpegs = self.to_messages(system, turns)
        try:
            raw = (self.transport or self._post)({**params, "messages": msgs})
        except CallTimeout:
            comp = Completion(text="", usage={"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
                              finish_reason="length", raw={})
            comp.extra = {"timeout_truncated": True}
            return comp, request_summary(system, turns, params, jpegs)
        ch = raw["choices"][0]
        usage = openai_usage(raw.get("usage") or {})
        comp = Completion(text=ch["message"].get("content") or "", usage=usage, finish_reason=ch.get("finish_reason"),
                          raw={k: v for k, v in raw.items() if k != "choices"})
        return comp, request_summary(system, turns, params, jpegs)


def make_client(mcfg: dict, llm_cfg: dict, cost=None, log_path=None, **kw) -> ModelClient:
    prov = mcfg["provider"]
    if mcfg.get("api_registry"):    # API model registry configs/models_api.yaml (providers/)
        from .providers import make_api_client
        return make_api_client(mcfg, llm_cfg, cost, log_path, **kw)
    if prov == "openai":
        return OpenAIClient(mcfg, llm_cfg, cost, log_path, **kw)
    if prov == "vllm":
        from .models_vllm import VLLMClient
        return VLLMClient(mcfg, llm_cfg, cost, log_path, **kw)
    if prov == "mock":
        from .models_mock import MockClient
        return MockClient(mcfg, llm_cfg, cost, log_path, **kw)
    raise ValueError(f"unknown provider {prov}")
