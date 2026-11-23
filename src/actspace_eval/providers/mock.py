"""Fake transport layer: replaces the HTTP step; the rest of the real adapter code runs. Opens no network connections and needs no key.

Requests are checked against the rules of the real APIs (on violation it returns HTTP 400 and the run stops): images must be
640x480 or 480x640 JPEGs; Anthropic does not accept temperature, empty text blocks, or requests whose last message is not a user
message. Usage is returned in each provider's field format, and caching is simulated by each provider's rules.

policy: normal; invalid (unparsable reply); rate_limit (each request first gets fail_times 429s); server_error (fail_times 5xx
first); empty (fail_times billed empty replies first); stream_cut (fail_times incomplete replies first); dead (always 500);
truncated (thinking fills the limit, empty text); refusal (model refuses, empty text); fatal (HTTP fatal_status, default 400; on 401
it echoes part of the key, as OpenAI does); tool_markup (without output_config.format it replies with tool-call-style text, rotating
among plain, limit-filled and multiple-action variants; with format it replies with normal JSON).
quota (Anthropic, independent of policy): {"<key number>": {"after": n, "kind": credit | spend_cap | billing | spend_limit}}; this
key reports exhausted quota after n successful calls.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from collections import Counter

import cv2
import numpy as np

from ..data import MOVE_ACTIONS
from ..models import estimate_text_tokens

ALLOWED_SIZES = {(640, 480), (480, 640)}
_AUDIT_LOCK = threading.Lock()
KEY_CALLS = Counter()       # successful calls per key, shared across workers; cleared at the start of each test
QUOTA_REPLIES = {"credit": (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.", None),
                 "spend_limit": (400, "invalid_request_error", "You have reached your specified workspace spend limit.", None),
                 "spend_cap": (429, "rate_limit_error", "You have reached your API usage limits: your organization has crossed its monthly API usage threshold.",
                               {"error_code": "enforced_spend_limit_reached"}),
                 "billing": (402, "billing_error", "There is an issue with your billing information.", None)}


def image_size(b64: str) -> tuple[int, int]:
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return int(img.shape[1]), int(img.shape[0])


def schema_error(s: dict) -> str | None:
    """Official structured-output limits (structured-outputs page): objects must have additionalProperties false; numeric and string constraints are unsupported; optional parameters and union types are capped."""
    optional = unions = 0

    def walk(x):
        nonlocal optional, unions
        if any(k in x for k in ("minimum", "maximum", "multipleOf", "minLength", "maxLength")):
            return "unsupported constraint"
        if "anyOf" in x or isinstance(x.get("type"), list):
            unions += 1
        if x.get("type") == "object":
            if x.get("additionalProperties") is not False:
                return "additionalProperties must be false for objects"
            optional += len(set(x.get("properties", {})) - set(x.get("required", [])))
        return next(filter(None, map(walk, [*x.get("properties", {}).values(), *x.get("anyOf", [])])), None)
    err = walk(s)
    return err or ("Schema is too complex for compilation" if optional > 24 or unions > 16 else None)


def bad(msg: str) -> dict:
    return {"_http_status": 400, "_text": json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": msg}})}


class WireMock:
    def __init__(self, mcfg: dict):
        self.mcfg, self.cfg = mcfg, mcfg["mock"]
        self.provider = mcfg["provider"]
        self.policy, self.steps, self.fail_times = self.cfg["policy"], self.cfg["steps"], self.cfg["fail_times"]
        self.tpi = mcfg["tokens_per_image"]
        self.requests = []          # received requests (checked by tests)
        self.key_no = self.api_key = None       # set by the adapter before sending
        self.fails_left = None
        self.cached = set()
        self.prev = {}
        self.n = 0

    # ---- request content
    def _blocks(self, body):
        """-> [(role, [(type, content)])], normalizing both providers' formats."""
        out = []
        for m in body["messages"]:
            c = m["content"]
            if isinstance(c, str):
                out.append((m["role"], [("text", c)]))
                continue
            items = []
            for b in c:
                if b["type"] == "text":
                    items.append(("text", b["text"]))
                elif b["type"] == "image":
                    items.append(("image", b["source"]["data"]))
                else:
                    items.append(("image", b["image_url"]["url"].split(",", 1)[1]))
            out.append((m["role"], items))
        return out

    def _system(self, body):
        if self.provider == "anthropic":
            return body["system"][0]["text"]
        return body["messages"][0]["content"]

    def _check(self, body):
        msgs = self._blocks(body)
        if self.provider == "anthropic":
            for k in ("temperature", "top_p", "top_k"):
                if k in body:
                    return bad(f"{k}: Extra inputs are not permitted")
            if msgs[-1][0] != "user" or msgs[0][0] != "user":
                return bad("conversation must start and end with a user message")
            if body["thinking"] not in ({"type": "adaptive", "display": "summarized"}, {"type": "adaptive"}) or body["output_config"]["effort"] != (self.mcfg.get("reasoning") or {}).get("effort", "high"):
                return bad("unexpected thinking configuration")
            if any(k in body for k in ("tools", "tool_choice")):
                return bad("tools are not part of this evaluation")
            fmt = body["output_config"].get("format")
            if fmt is not None:
                err = "output_config.format.type must be json_schema" if fmt.get("type") != "json_schema" else schema_error(fmt["schema"])
                if err:
                    return bad(err)
            if sum("cache_control" in b for m in body["messages"] if isinstance(m["content"], list) for b in m["content"]) > 1:
                return bad("more than one cache marker")
            for role, items in msgs:
                for kind, v in items:
                    if kind == "text" and not v.strip():
                        return bad("text content blocks must be non-empty")
        n = 0
        for _, items in msgs:
            for kind, v in items:
                if kind == "image":
                    n += 1
                    if image_size(v) not in ALLOWED_SIZES:
                        return bad(f"image size {image_size(v)} not allowed by the protocol")
        if n > 41:
            return bad(f"{n} images in one request")
        return None

    def _audit(self, body):
        path = os.environ.get("ACTSPACE_MOCK_AUDIT")
        if not path:
            return
        sizes = [image_size(v) for _, items in self._blocks(body) for k, v in items if k == "image"]
        line = json.dumps({"provider": self.provider, "model": body["model"], "images": sizes,
                           "system_sha256": hashlib.sha256(self._system(body).encode()).hexdigest()}) + "\n"
        with _AUDIT_LOCK, open(path, "a") as f:
            f.write(line)

    # ---- reply content
    def _reply(self, body):
        msgs = [m for m in self._blocks(body) if m[0] != "system"]
        if self.policy == "invalid":
            return "I would like to look around a bit more before deciding."
        if self.policy == "tool_markup" and "format" not in body["output_config"]:
            return ('\n<parameter name="action">forward</parameter>', '\n<invoke name="p">\n</invoke>' + '\n\n<invoke name="forward">\n</invoke>' * 3,
                    '\n<invoke name="x">\n</invoke>' + '\n\n{"action": "turn_left"}' * 3)[self.n % 3]
        system = self._system(body)
        n_moves = sum(r == "assistant" for r, _ in msgs)
        last = "".join(v for k, v in msgs[-1][1] if k == "text")
        if '"objects"' in last and "\nObjects:\n" in last:      # cogmap probe (providers/ablations.py): objects at odd (0-based) indices are answered unknown
            self.probes = getattr(self, "probes", 0) + 1
            if self.probes <= self.cfg.get("probe_ignore", 0):      # ignore the probe for the first probe_ignore times and answer as usual
                return json.dumps({"action": "answer", "answers": {"Q1": "cannot_determine"}})
            names = re.findall(r"^- (.+)$", last.split("\nObjects:\n", 1)[1].split("\n\n", 1)[0], re.M)
            return json.dumps({"objects": [{"name": n, "known": i % 2 == 0, "forward_m": 1.0 + i if i % 2 == 0 else None,
                                            "left_m": -0.5 if i % 2 == 0 else None} for i, n in enumerate(names)]})
        if system.startswith("You are exploring") and n_moves < self.steps and "No steps remain" not in last:
            return json.dumps({"action": MOVE_ACTIONS[n_moves % len(MOVE_ACTIONS)]})
        first = "".join(v for k, v in msgs[0][1] if k == "text")
        qs = re.findall(r"^(Q\d+) \((multiple choice|numeric, meters)\)", first, re.M)
        return json.dumps({"action": "answer", "answers": {q: (1.5 if k.startswith("numeric") else "A") for q, k in qs}})

    def _tokens(self, body):
        text = self._system(body)
        images = 0
        for _, items in self._blocks(body):
            for k, v in items:
                if k == "text":
                    text += v
                else:
                    images += 1
        injected = 100 if "format" in (body.get("output_config") or {}) else 0    # extra system prompt added server-side for structured outputs
        return estimate_text_tokens(text) + images * self.tpi + injected

    # ---- per-provider reply formats
    def _openai(self, body, text, finish):
        total = self._tokens(body)
        key = body.get("prompt_cache_key")
        cached = self.prev.get(key, 0) // 128 * 128 if self.prev.get(key, 0) >= 1024 else 0
        self.prev[key] = total
        write = total - cached if total >= 1024 else 0
        out = estimate_text_tokens(text) + 30
        usage = {"prompt_tokens": total, "completion_tokens": out, "total_tokens": total + out,
                 "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": write},
                 "completion_tokens_details": {"reasoning_tokens": 30}}
        return {"id": f"chatcmpl-mock{self.n}", "model": body["model"], "usage": usage,
                "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]}

    def _anthropic_usage(self, body, text):
        total, prefix = self._tokens(body), 0
        marked = None
        for i, b in enumerate(body["messages"][0]["content"]):
            if isinstance(b, dict) and "cache_control" in b:
                marked = i
        if marked is not None:
            blocks = body["messages"][0]["content"][:marked + 1]
            prefix = estimate_text_tokens(self._system(body) + "".join(b.get("text", "") for b in blocks)) + self.tpi
        read = write = 0
        if prefix >= 512:
            key = hashlib.sha256(json.dumps([body["system"], body["messages"][0]["content"][:marked + 1]], sort_keys=True).encode()).hexdigest()
            read, write = (prefix, 0) if key in self.cached else (0, prefix)
            self.cached.add(key)
        return {"input_tokens": total - read - write, "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
                "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0},
                "output_tokens": estimate_text_tokens(text) + 30, "output_tokens_details": {"thinking_tokens": 30}}

    def _anthropic(self, body, text, finish, stop_details=None):
        u = self._anthropic_usage(body, text)
        summary = "mock thinking summary"
        if not body.get("stream"):
            content = [{"type": "thinking", "thinking": summary, "signature": "mock"}]
            if text:
                content.append({"type": "text", "text": text})
            return {"id": f"msg_mock{self.n}", "type": "message", "role": "assistant", "model": body["model"], "content": content,
                    "stop_reason": finish, "stop_details": stop_details, "usage": u}
        first = {k: v for k, v in u.items() if k not in ("output_tokens_details",)}
        ev = [{"type": "message_start", "message": {"id": f"msg_mock{self.n}", "type": "message", "role": "assistant", "model": body["model"],
                                                    "content": [], "stop_reason": None, "usage": {**first, "output_tokens": 1}}},
              {"type": "ping"},
              {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
              {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": summary}},
              {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "mock"}},
              {"type": "content_block_stop", "index": 0}]
        if text:
            half = len(text) // 2
            ev += [{"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[:half]}},
                   {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[half:]}},
                   {"type": "content_block_stop", "index": 1}]
        ev += [{"type": "message_delta", "delta": {"stop_reason": finish, "stop_sequence": None, "stop_details": stop_details},
                "usage": {"output_tokens": u["output_tokens"], "output_tokens_details": u["output_tokens_details"]}},
               {"type": "message_stop"}]
        return ev

    def _format(self, body, text, finish=None, stop_details=None):
        if self.provider == "anthropic":
            return self._anthropic(body, text, finish or "end_turn", stop_details)
        return self._openai(body, text, finish or "stop")

    def __call__(self, body):
        self.requests.append(body)
        err = self._check(body)
        if err:
            return err
        self._audit(body)
        quota = (self.cfg.get("quota") or {}).get(str(self.key_no))
        if quota and KEY_CALLS[self.key_no] >= quota["after"]:
            status, typ, msg, details = QUOTA_REPLIES[quota["kind"]]
            err = {"type": typ, "message": f"{msg} (key {self.api_key})", **({"details": details} if details else {})}
            return {"_http_status": status, "_text": json.dumps({"type": "error", "error": err})}
        KEY_CALLS[self.key_no] += 1
        self.n += 1
        p = self.policy
        if p == "fatal":
            status = self.cfg.get("fatal_status", 400)
            if status == 400:
                return bad("mock bad request")
            return {"_http_status": status, "_text": "Incorrect API key provided: sk-abc123***wxyz"}
        if p == "dead":
            return {"_http_status": 500, "_text": "mock internal error"}
        if p == "truncated":
            return self._format(body, "", "max_tokens" if self.provider == "anthropic" else "length")
        if p == "refusal":
            return self._format(body, "", "refusal" if self.provider == "anthropic" else "content_filter",
                                {"type": "refusal", "category": None, "explanation": "mock"})
        if p in ("rate_limit", "server_error", "empty", "stream_cut"):
            if self.fails_left is None:
                self.fails_left = self.fail_times
            if self.fails_left > 0:
                self.fails_left -= 1
                if p == "rate_limit":
                    return {"_http_status": 429, "_text": "mock rate limit", "_headers": {"retry-after": "0"}}
                if p == "server_error":
                    return {"_http_status": 529 if self.provider == "anthropic" else 503, "_text": "mock overloaded"}
                if p == "empty":
                    return self._format(body, "")
                if self.provider == "anthropic":
                    return self._format(body, "x")[:-2] if body.get("stream") else {"id": "x", "content": []}
                return {"id": "x", "choices": []}
            self.fails_left = None
        text = self._reply(body)
        full = p == "tool_markup" and text.count("<invoke") > 2      # repeated tool calls fill the limit
        return self._format(body, text, "max_tokens" if full else None)


def make_transport(mcfg: dict) -> WireMock:
    return WireMock(mcfg)
