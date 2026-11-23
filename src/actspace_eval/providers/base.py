"""Shared part of the closed-model adapters.

- Real-call switch: llm.allow_real_calls, llm.prompts_approved, prices_confirmed, reasoning.approved and a key (keys.KeyPool for
  Claude, an env var otherwise) are all required (checked at construction; not checked when a transport is injected). For the key,
  only its presence is checked; it is read into the request header only when a request is sent.
- Call records (same format as for open-weight models): the original text; for each image the photo ID, original frame ID, size,
  JPEG byte count and full SHA-256 (images are not stored); the full raw reply; thinking content or summary (if any); parse result;
  finish_reason; raw usage; cost; latency; logprobs (stored when the provider supports them and they are turned on).
- Server errors: HTTP errors, rate limits, dropped connections, incomplete reply structures and empty replies all raise ModelError and
  are retried with exponential backoff by ModelClient.complete; once retries are exhausted the runner marks stopped_server_error, the
  episode is not written to the trajectories and does not count as an invalid model reply. Exception (same decision as for open-weight
  models): when finish_reason says max_tokens was filled (thinking hit the limit) or the model refused, the reply comes from the model
  and counts as no answer.
- Fatal errors: HTTP 400 / 401 / 403 and other non-retryable errors (402, 404, 413, 429 for exhausted quota) indicate a problem with
  the request or the account, so retrying is useless: no retry, the whole run stops immediately (ApiCostTracker.abort: the next call
  of every other concurrent worker is not sent), status = stopped_fatal_error, with an exit code different from the cost cap.
  Insufficient model capability (ModelLimit) also stops immediately but is still recorded as N/A. Keys are removed from error
  messages before they are logged (redact).
- Single job: CostPool is the cost cap and the episode slots shared by all runs in a job; RunCost is the account of one of those runs.
  Slots are granted in run order: earlier runs use them first, and once a run has issued all its episodes its slots go to the next run
  immediately. On the cost cap, a fatal error, or exhausted server-error retries it aborts: no further calls are sent and no new
  episode is started.
- Billed failed attempts: an empty reply carries usage, so its cost is added to the total and written into the error record (billed);
  on timeouts or broken streams the server may have billed with unknown usage, recorded as possibly_billed.
- Cost reservation (ApiCostTracker): under concurrency each attempt first reserves the cost of "all input at the cache-write price +
  max_tokens fully used"; if spent + reserved would exceed the cap it waits for other calls to finish, and stops if it still does not
  fit with no call in flight. This keeps concurrent runs under the cap.
"""
from __future__ import annotations

import contextlib
import hashlib
import heapq
import itertools
import json
import os
import re
import threading
import uuid

from ..cost import TRUNCATED_FINISH, CostCapExceeded, CostTracker
from ..models import Completion, ModelError, RealCallsDisabled, estimate_input_tokens
from .keys import active_pool

REFUSED = ("refusal", "content_filter")
FATAL_STATUS = "stopped_fatal_error"
_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_*.\-]{4,}")
INPUT_MARGIN = 1.5     # inflate the input estimate when reserving: text is estimated at 4 characters per token, and tokenizers differ by provider


class ApiCostTracker(CostTracker):
    def __init__(self, prices, cap_usd, spent=0.0):
        super().__init__(prices, cap_usd, spent)
        self._cv = threading.Condition()
        self._held = 0.0
        self._mine = threading.local()
        self.aborted = None         # fatal error: no further calls are sent

    def abort(self, e: Exception):
        """No further calls are sent. On insufficient model capability only the affected run is recorded as N/A; the other runs stop as on a fatal error."""
        if getattr(e, "kind", None):
            other = ModelError(f"stopped because another run hit a model limit: {e}", retryable=False, status=e.status)
            other.stop_status = FATAL_STATUS
            e = other
        with self._cv:
            self.aborted = self.aborted or e
            self._cv.notify_all()

    def reserve(self, est_input_tokens: int, max_output_tokens: int):
        n = int(est_input_tokens * INPUT_MARGIN)
        est = self.cost_of({"input_tokens": n, "cache_write_tokens": n, "output_tokens": max_output_tokens})
        with self._cv:
            while True:
                if self.aborted is not None:
                    raise self.aborted
                if self.spent + self._held + est <= self.cap:
                    break
                if self._held <= 0:
                    raise CostCapExceeded(f"spent ${self.spent:.4f} + at most ${est:.4f} for this call would exceed the cap ${self.cap:.2f}; no further calls")
                self._cv.wait(timeout=5)
            self._held += est
            self._mine.held = est

    def release(self):
        with self._cv:
            self._held -= getattr(self._mine, "held", 0.0)
            self._mine.held = 0.0
            self._cv.notify_all()

    def add(self, usage: dict) -> float:
        with self._cv:
            c = super().add(usage)
            self._held -= getattr(self._mine, "held", 0.0)
            self._mine.held = 0.0
            self._cv.notify_all()
        return c


class CostPool(ApiCostTracker):
    """Shared by all runs in a job: cost cap, reservations, stop signal and `slots` episode slots."""

    def __init__(self, prices, cap_usd, spent=0.0, slots=1):
        super().__init__(prices, cap_usd, spent)
        self.free = slots
        self.in_use = 0
        self._waiting = []
        self._seq = itertools.count()
        self._not_arrived = set()   # runs that have not reached the slot queue yet: threads start at different times, so later runs wait for them to arrive

    def expect(self, orders):
        with self._cv:
            self._not_arrived = set(orders)

    def arrived(self, order: int):
        """Run `order` has arrived (first slot request, or it finished with no episode to run)."""
        with self._cv:
            self._not_arrived.discard(order)
            self._cv.notify_all()

    def view(self, spent=0.0) -> "RunCost":
        return RunCost(self, spent)

    def slot(self, order: int):
        """Episode slot for run `order`: calling the result with no arguments returns a context manager. Smaller orders get slots first; after a stop, the stop reason is raised."""
        @contextlib.contextmanager
        def enter():
            ticket = (order, next(self._seq))
            with self._cv:
                heapq.heappush(self._waiting, ticket)
                self._not_arrived.discard(order)
                self._cv.notify_all()
                try:
                    while self.aborted is None and not (self.free > 0 and self._waiting[0] == ticket
                                                        and min(self._not_arrived, default=order) >= order):
                        self._cv.wait(timeout=5)
                finally:
                    self._waiting.remove(ticket)
                    heapq.heapify(self._waiting)
                    self._cv.notify_all()
                if self.aborted is not None:
                    raise self.aborted
                self.free -= 1
                self.in_use += 1
            try:
                yield
            finally:
                with self._cv:
                    self.free += 1
                    self.in_use -= 1
                    self._cv.notify_all()
        return enter


class RunCost(ApiCostTracker):
    """Account of one run within a CostPool: spent is this run's cost (written to status.json); cap, reservations and stops all go through the pool."""

    def __init__(self, pool: CostPool, spent=0.0):
        CostTracker.__init__(self, pool.prices, pool.cap, spent)
        self.pool = pool

    def reserve(self, est_input_tokens, max_output_tokens):
        self.pool.reserve(est_input_tokens, max_output_tokens)

    def release(self):
        self.pool.release()

    def abort(self, e):
        self.pool.abort(e)

    def add(self, usage):
        c = self.pool.add(usage)
        with self.pool._cv:
            self.spent += c
        return c

    def check_before(self, est_input_tokens, max_output_tokens):
        self.pool.check_before(est_input_tokens, max_output_tokens)

    def check_after(self):
        self.pool.check_after()


def repair_tail(path) -> bool:
    """When a job is killed, the last line of the jsonl may be half-written: drop it so that a resume can read the file. Returns whether the file was changed."""
    if not os.path.exists(path):
        return False
    with open(path, "rb+") as f:
        data = f.read()
        keep = data.rfind(b"\n") + 1
        if keep == len(data):
            return False
        f.truncate(keep)
    return True


def check_real_call_gates(mcfg: dict, llm_cfg: dict):
    if not llm_cfg.get("allow_real_calls"):
        raise RealCallsDisabled("llm.allow_real_calls is false: real API calls are not allowed")
    if not llm_cfg.get("prompts_approved"):
        raise RealCallsDisabled("llm.prompts_approved is false: the prompts have not been reviewed and cannot be used for real calls")
    if not mcfg.get("prices_confirmed"):
        raise RealCallsDisabled(f"prices_confirmed of {mcfg['short']} is false: prices are unconfirmed, cannot be used for real calls")
    if not mcfg["reasoning"].get("approved"):
        raise RealCallsDisabled(f"reasoning.approved of {mcfg['short']} is false: the reasoning-effort setting is not approved, cannot be used for real calls")
    if mcfg["provider"] == "anthropic" and active_pool():
        return
    if not mcfg.get("key_env") or not os.environ.get(mcfg["key_env"]):
        raise RealCallsDisabled(f"environment variable {mcfg.get('key_env')} is not set")


def redact(text: str, secret: str | None = None) -> str:
    """Removes keys from error messages: the server may echo them (OpenAI's 401 echoes part of the key)."""
    if secret:
        text = text.replace(secret, "<redacted>")
    return _KEY_LIKE.sub("<redacted>", text)


def http_error(status: int, text: str, headers=None, secret=None) -> ModelError:
    """HTTP error -> ModelError. 408 / 409 / 429 / 5xx (including 529) are retryable; the rest (400, 401, 403, ...) and 429 for exhausted quota are fatal: no retry, the whole run stops."""
    text = redact(text, secret)
    quota = status == 429 and any(s in text for s in ("enforced_spend_limit_reached", "insufficient_quota"))
    e = ModelError(f"HTTP {status}: {text[:500]}", retryable=(status in (408, 409, 429) or status >= 500) and not quota, status=status)
    if not e.retryable:
        e.stop_status = FATAL_STATUS
    ra = (headers or {}).get("retry-after")
    try:
        e.retry_after = float(ra) if ra is not None else None
    except ValueError:
        e.retry_after = None
    return e


def lost_reply(msg: str) -> ModelError:
    """The request was sent but no complete reply arrived (timeout, dropped connection, broken stream, missing usage): retryable; the server may have billed, usage unknown."""
    e = ModelError(msg, retryable=True)
    e.billed = {"possibly_billed": True}
    return e


class ApiClientMixin:
    """Subclasses implement to_messages(system, turns) -> (wire content, {photo ID: jpeg}) and _exchange(params, wire content) -> dict with
    text, thinking, finish, usage (normalized usage), usage_raw, and optionally ttft_s, request_id, refusal, logprobs, model."""

    def _setup(self, smoke=False, transport=None):
        self.transport = transport
        self.cache_key = None
        self.frame_of = None        # photo ID -> original frame ID (set per episode by the runner; logged only, never sent to the model)
        self.group_id = None
        self.run_seed = 0
        self.lp_path = self.log_path.parent / "logprobs.jsonl.gz" if self.log_path else None
        self.llm_cfg = {**self.llm_cfg, "request_timeout_s": self.mcfg["request_timeout_s"]}
        if transport is None:
            check_real_call_gates(self.mcfg, self.llm_cfg)

    def _key(self) -> str:
        return os.environ[self.mcfg["key_env"]]   # goes into the request header only

    def _secret(self) -> str | None:
        return os.environ.get(self.mcfg.get("key_env") or "")

    def _http_error(self, status, text, headers=None) -> ModelError:
        return http_error(status, text, headers, secret=self._secret())

    def _logged(self, system, turns, jpegs) -> list:
        out = [{"role": "system", "content": system}]
        for t in turns:
            if "parts" not in t:
                out.append({"role": t["role"], "content": t["text"]})
                continue
            content = []
            for p in t["parts"]:
                if p[0] == "text":
                    content.append({"type": "text", "text": p[1]})
                    continue
                jpg, (h, w) = jpegs[p[1]], p[2].shape[:2]
                content.append({"type": "image", "photo_id": p[1], "frame_id": self.frame_of(p[1]) if self.frame_of else None,
                                "w": w, "h": h, "jpeg_bytes": len(jpg), "sha256": hashlib.sha256(jpg).hexdigest()})
            out.append({"role": t["role"], "content": content})
        return out

    def _reserve(self, system, turns):
        if isinstance(self.cost, ApiCostTracker):
            self.cost.reserve(estimate_input_tokens(system, turns, self.mcfg["tokens_per_image"]), self.mcfg["max_output_tokens"])

    def _release(self):
        if isinstance(self.cost, ApiCostTracker):
            self.cost.release()

    def complete(self, system, turns, tag=None):
        try:
            return super().complete(system, turns, tag)
        except (ModelError, CostCapExceeded) as e:      # fatal error, retries exhausted, cost cap: other workers stop calling too
            if isinstance(self.cost, ApiCostTracker):
                self.cost.abort(e)
            raise

    def _request(self, system, turns):
        params = self.params()
        wire, jpegs = self.to_messages(system, turns)
        logged = self._logged(system, turns, jpegs)
        self._reserve(system, turns)
        try:
            r = self._exchange(params, wire)
        except BaseException:
            self._release()
            raise
        if not r["text"].strip() and r["finish"] not in TRUNCATED_FINISH + REFUSED:
            e = ModelError(f"server error: empty reply (finish_reason {r['finish']}, usage {r['usage_raw']})", retryable=True)
            e.billed = {"usage": r["usage"], "usage_raw": r["usage_raw"], "finish_reason": r["finish"], "response": r["text"],
                        "reasoning": r.get("thinking") or "", "cost_usd": self.cost.add(r["usage"]) if self.cost is not None else 0.0}
            raise e
        call_id = uuid.uuid4().hex
        if r.get("logprobs") and self.lp_path is not None:
            from ..models_vllm import VLLMClient
            VLLMClient._write_logprobs(self, call_id, self._tag, r["logprobs"])
        comp = Completion(text=r["text"], usage=r["usage"], finish_reason=r["finish"],
                          raw={"usage": r["usage_raw"], "id": r.get("request_id"), "model": r.get("model")})
        comp.extra = {"call_id": call_id, "provider": self.provider, "request_id": r.get("request_id"), "model_returned": r.get("model"),
                      "reasoning": r.get("thinking") or "", "reasoning_chars": len(r.get("thinking") or ""),
                      "reasoning_tokens": r["usage"].get("reasoning_tokens"), "refusal": r.get("refusal"), "ttft_s": r.get("ttft_s"),
                      "max_tokens": self.mcfg["max_output_tokens"], "logprobs_tokens": len(r.get("logprobs") or []),
                      "run_seed": self.run_seed, "group_id": self.group_id, "key_no": r.get("key_no")}
        return comp, {"params": {k: v for k, v in params.items() if k != "stream"}, "messages": logged}


def read_json(resp) -> dict:
    try:
        return resp.json()
    except (ValueError, json.JSONDecodeError) as e:
        raise lost_reply(f"malformed response body: {e}") from None
