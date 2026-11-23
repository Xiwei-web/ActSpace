"""Mock model (docs/evaluation_protocol.md): calls no API; returns well-formed valid / invalid replies according to policy, to test parsing, retries, cost caps and resuming.

policy:
- valid: Active first takes mock_steps steps (cycling through move actions in a fixed order), then answers; static settings answer directly. Answers are well-formed (letters for multiple choice, numbers for numeric questions).
- invalid: always returns unparseable text.
- mixed: odd-numbered calls return an invalid reply, the rest as valid.
- flaky: each request first raises fail_times retryable errors (simulating 429), then returns as valid.
- fatal: raises a non-retryable error (simulating 400).
Usage is computed with the estimation rule (input: about 4 characters per token for text + tokens_per_image per photo; output: reply length), and billed at the configured prices;
cache hits are simulated via cached_prefix_tokens (message prefix identical to the previous request).
"""
from __future__ import annotations

import json
import re

from .data import MOVE_ACTIONS
from .models import (Completion, ModelClient, ModelError, cached_prefix_tokens, estimate_input_tokens, estimate_text_tokens,
                     request_summary)


class MockClient(ModelClient):
    provider = "mock"

    def __init__(self, mcfg, llm_cfg, cost=None, log_path=None, sleep=None, transport=None, smoke=False):
        super().__init__(mcfg, llm_cfg, cost, log_path, sleep or (lambda s: None))
        self.n_calls = 0
        self._fails_left = None
        self._prev = None           # previous request (system, turns): simulates the prefix cache
        self.cache_key = None

    def _questions(self, turns):
        """Read question IDs and types from the first user message (the mock only looks at text)."""
        text = "".join(p[1] for p in turns[0]["parts"] if p[0] == "text")
        return re.findall(r"^(Q\d+) \((multiple choice|numeric, meters)\)", text, re.M)

    def _reply(self, system, turns):
        policy = self.mcfg["policy"]
        k = self.n_calls
        if policy == "invalid" or (policy == "mixed" and k % 2 == 1):
            return "I would like to look around a bit more before deciding."
        active = system.startswith("You are exploring")
        n_moves = sum(t["role"] == "assistant" for t in turns)
        if active and n_moves < self.mcfg["mock_steps"] and "No steps remain" not in str(turns[-1].get("parts", "")):
            return json.dumps({"action": MOVE_ACTIONS[n_moves % len(MOVE_ACTIONS)]})
        ans = {qid: (1.5 if kind.startswith("numeric") else "A") for qid, kind in self._questions(turns)}
        return json.dumps({"action": "answer", "answers": ans})

    def _request(self, system, turns):
        policy = self.mcfg["policy"]
        if policy == "fatal":
            raise ModelError("HTTP 400: mock bad request", retryable=False, status=400)
        if policy == "flaky":
            if self._fails_left is None:
                self._fails_left = self.mcfg["fail_times"]
            if self._fails_left > 0:
                self._fails_left -= 1
                raise ModelError("HTTP 429: mock rate limit", retryable=True, status=429)
            self._fails_left = None  # restart the count on the next request
        text = self._reply(system, turns)
        self.n_calls += 1
        tpi = self.mcfg["tokens_per_image"]
        cached = cached_prefix_tokens(self._prev, (system, turns), tpi)
        self._prev = (system, turns)
        usage = {"input_tokens": estimate_input_tokens(system, turns, tpi),
                 "cached_input_tokens": cached, "output_tokens": estimate_text_tokens(text), "reasoning_tokens": 0}
        return Completion(text=text, usage=usage, finish_reason="stop"), request_summary(system, turns, self.params())
