"""OpenAI adapter: reuses the message assembly and request parameters of models.OpenAIClient (Chat Completions, reasoning_effort,
max_completion_tokens, JSON mode, prompt_cache_key, image detail) and adds logging, retries and cost reservation from
base.ApiClientMixin. The name of the key's env var comes from the registry."""
from __future__ import annotations

import time

from ..models import ModelClient, ModelError, OpenAIClient, openai_usage
from .base import ApiClientMixin, lost_reply, read_json


class OpenAIAPIClient(ApiClientMixin, OpenAIClient):
    provider = "openai"

    def __init__(self, mcfg, llm_cfg, cost=None, log_path=None, sleep=time.sleep, transport=None, smoke=False):
        ModelClient.__init__(self, mcfg, llm_cfg, cost, log_path, sleep)   # the switch is checked in _setup (the key's env var name comes from the registry)
        self._setup(smoke, transport)

    def params(self) -> dict:
        p = super().params()
        if self.mcfg.get("logprobs_top"):
            p["logprobs"], p["top_logprobs"] = True, int(self.mcfg["logprobs_top"])
        return p

    def _post(self, body: dict) -> dict:
        import requests
        try:
            resp = requests.post(self.mcfg["base_url"], json=body, timeout=(10, self.llm_cfg["request_timeout_s"]),
                                 headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"})
        except requests.exceptions.ConnectTimeout as e:
            raise ModelError(f"{type(e).__name__}: {e}", retryable=True) from None
        except requests.exceptions.RequestException as e:   # the request may have reached the server
            raise lost_reply(f"{type(e).__name__}: {e}") from None
        if resp.status_code != 200:
            raise self._http_error(resp.status_code, resp.text, resp.headers)
        return read_json(resp)

    def _exchange(self, params, wire):
        t0 = time.time()
        raw = (self.transport or self._post)({**params, "messages": wire})
        if "_http_status" in raw:     # tests: simulated error returned by the server
            raise self._http_error(raw["_http_status"], raw["_text"], raw.get("_headers"))
        try:
            ch = raw["choices"][0]
            msg, u = ch["message"], raw["usage"]
        except (KeyError, IndexError, TypeError):
            raise lost_reply(f"malformed response: {str(raw)[:300]}") from None
        if ch.get("finish_reason") is None or not u:
            raise lost_reply(f"incomplete response: finish_reason {ch.get('finish_reason')}, usage {u}")
        return {"text": msg.get("content") or "", "thinking": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "finish": ch["finish_reason"], "usage": openai_usage(u), "usage_raw": u, "refusal": msg.get("refusal"),
                "request_id": raw.get("id"), "model": raw.get("model"), "ttft_s": None, "latency": time.time() - t0,
                "logprobs": (ch.get("logprobs") or {}).get("content") or []}
