"""Closed-model API evaluation: adapters of both providers, records, retries, cost cap and resume, and the guards of the one-click script.
Uses only the fake transport (providers/mock.py) and a replaced requests.post; no network, no real keys; where a key is needed, a
test-only variable name is used."""
import base64
import copy
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from actspace_eval import prompts as P
from actspace_eval.cost import CostCapExceeded, CostTracker
from actspace_eval.data import MOVE_ACTIONS, FrozenData, load_eval_config
from actspace_eval.llm_run import env_factory, run_llm
from actspace_eval.models import ModelError, RealCallsDisabled, make_client
from actspace_eval.providers import api_eval as E
from actspace_eval.providers import format_diag as FD
from actspace_eval.providers import keys as K
from actspace_eval.providers import registry as R
from actspace_eval.providers.anthropic_api import MAX_QUESTIONS, REPLY_SCHEMA, AnthropicClient, anthropic_usage, merge_events
from actspace_eval.providers.api_probe import probe_model
from actspace_eval.providers import smoke_report as SR
from actspace_eval.providers.base import ApiCostTracker, CostPool, check_real_call_gates, http_error, redact, repair_tail
from actspace_eval.providers.mock import KEY_CALLS, WireMock, image_size, schema_error
from actspace_eval.providers.openai_api import OpenAIAPIClient
from actspace_eval.report_eval import build as report_build
from actspace_eval.score import RepBoot
from actspace_eval.trajectory import complete_episodes, replay

ROOT = Path(__file__).resolve().parents[2]
TWO = ["arkit_g_000000", "scannetpp_g_000009"]
MOCKS = ["mock-openai", "mock-anthropic"]
FAKE_VAR = "ACTSPACE_TEST_FAKE_KEY"
PNG = np.zeros((480, 640, 3), np.uint8)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The registry switches are on, so tests must never send real requests. Removes the real key env vars; tests that need to replace requests.post do so themselves."""
    import requests

    def refuse(*a, **k):
        raise AssertionError("tests must not send real requests")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(requests, "post", refuse)


@pytest.fixture(autouse=True)
def clean_key_pool():
    KEY_CALLS.clear()
    yield
    K.activate(None)


def fake_key_file(tmp_path, numbers=(1, 2), name="fake.env"):
    f = tmp_path / name
    f.write_text("".join(f"ANTHROPIC_API_KEY_{n}={fake_key(n)}\n" for n in numbers) + "ANTHROPIC_API_KEY_9=\nOPENAI_API_KEY=sk-FAKEOPENAIabcdefghij\n")
    f.chmod(0o600)
    return f


def fake_key(n):
    return f"sk-ant-FAKE{n}-abcdefghijklmnop"


@pytest.fixture()
def data():
    d = FrozenData(verify=False)
    R.merge(d)
    return d


def setup(data, model, **mock):
    """Changes this model's fake transport config (policy, steps, fail_times)."""
    d = copy.deepcopy(data.ecfg["models"][model])
    d["mock"].update(mock)
    data.ecfg["models"][model] = d


def run(data, model, tmp_path, setting="active_v2", budget=40, gids=TWO, name="r", cap=100.0, workers=1, render=True, **kw):
    return run_llm(data, model, setting, budget, gids, tmp_path / name, render=render, cost_cap=cap, print_estimate=False, workers=workers,
                   tracker_cls=ApiCostTracker, **kw)


def calls(path):
    return [json.loads(x) for x in open(path / "calls.jsonl")]


def turns_with_image(text="Questions\nQ1 (numeric, meters) How far?\n\nYou are at your starting viewpoint, photo_1 (START).\n"):
    return [{"role": "user", "parts": [("text", text), ("image", "photo_1", PNG), ("text", "\nSteps used: 0 of 40; 40 remaining. Choose your next action.")]}]


def client(data, model, **kw):
    mcfg = data.ecfg["models"][model]
    return make_client(mcfg, data.ecfg["llm"], kw.pop("cost", None), kw.pop("log", None), sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- registry
def test_registry_defaults_and_fields():
    reg = R.load_registry()
    assert set(reg["models"]) == {"sol", "terra", "luna", "astra", "fable-5.1", "opus-5.5", "sonnet-5.5", "haiku-5.5"}
    ids = {v["model_id"] for v in reg["models"].values()}
    assert {"gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna", "gpt-6-astra", "claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} == ids
    for k, v in reg["models"].items():
        assert v["prices_confirmed"] is True and v["reasoning"]["approved"] is True, k     # switched on after review
        assert v["logprobs_top"] is None
        assert v["max_output_tokens"] == 32768 and v["reasoning"]["target"] == "high"
        assert v["key_env"] in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        assert {"input", "cache_read", "cache_write", "output", "source"} <= set(v["prices"])
        assert v["prices"]["source"].startswith("https://")
        assert v["image"]["max_images_per_request"] >= 41 and v["concurrency"]["episodes"] >= 1
    assert "sk-" not in (ROOT / "configs" / "models_api.yaml").read_text()


def test_eval_yaml_models_untouched_by_merge(data):
    ec = load_eval_config()
    assert data.ecfg["models"]["gpt-6.1-sol"] == ec["models"]["gpt-6.1-sol"]
    assert "sol" not in ec["models"] and data.ecfg["models"]["sol"]["model"] == "gpt-6.1-sol"


def test_protocol_guard(data):
    R.assert_protocol(data.ecfg, data.version)
    bad = copy.deepcopy(data.ecfg)
    bad["env"]["budget"] = 50
    with pytest.raises(RuntimeError):
        R.assert_protocol(bad, data.version)
    with pytest.raises(RuntimeError):
        R.assert_protocol(data.ecfg, "arkit_only_v0.11")


@pytest.mark.parametrize("model", ["sol", "opus-5.5"])
def test_real_call_gates(data, model, monkeypatch):
    """Every switch must be on; no request is sent except through an injected transport."""
    m = {**data.ecfg["models"][model], "key_env": FAKE_VAR, "prices_confirmed": False}
    m["reasoning"] = {**m["reasoning"], "approved": False}
    llm = data.ecfg["llm"]
    monkeypatch.delenv(FAKE_VAR, raising=False)
    with pytest.raises(RealCallsDisabled, match="prices_confirmed"):
        make_client(m, llm)
    m["prices_confirmed"] = True
    with pytest.raises(RealCallsDisabled, match="reasoning.approved"):
        make_client(m, llm)
    m["reasoning"] = {**m["reasoning"], "approved": True}
    with pytest.raises(RealCallsDisabled, match=FAKE_VAR):
        make_client(m, llm)
    monkeypatch.setenv(FAKE_VAR, "fake-for-test")
    check_real_call_gates(m, llm)
    with pytest.raises(RealCallsDisabled, match="allow_real_calls"):
        make_client(m, {**llm, "allow_real_calls": False})
    with pytest.raises(RealCallsDisabled, match="prompts_approved"):
        make_client(m, {**llm, "prompts_approved": False})


# ---------------------------------------------------------------- request content
def text_blocks(c):
    return [b["text"] for b in c if b["type"] == "text"] if isinstance(c, list) else [c]


@pytest.mark.parametrize("model", MOCKS)
@pytest.mark.parametrize("setting,budget", [("blind", 40), ("passive", 8), ("passive", 40), ("random_walk", 40), ("oracle_min", 40), ("oracle_route", 40), ("active_v2", 40)])
def test_all_settings_records_and_replay(data, model, setting, budget, tmp_path):
    setup(data, model, steps=3)
    st = run(data, model, tmp_path, setting, budget)
    assert st["status"] == "complete" and st["completed"] == 2
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0
    cs = [c for c in calls(tmp_path / "r") if c["type"] == "call"]
    for c in cs:
        assert c["provider"] == model.split("-")[1] and c["parse"]["json"] and c["usage_raw"] and c["finish_reason"] and c["latency_s"] >= 0
        assert c["cost_usd"] > 0 and c["group_id"] in TWO
        for m in c["request"]["messages"][1:]:
            for x in m["content"] if isinstance(m["content"], list) else []:
                if x["type"] == "image":
                    assert (x["w"], x["h"]) in ((640, 480), (480, 640)) and len(x["sha256"]) == 64 and x["frame_id"] is not None
    assert "base64," not in (tmp_path / "r" / "calls.jsonl").read_text()
    if model == "mock-anthropic":
        assert all(c["reasoning"] for c in cs) and all(c["ttft_s"] is not None for c in cs)
    eps = complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    s = eps[0][0]
    assert s["dataset"] in ("arkitscenes", "scannetpp") and s["budget_infeasible"] is False and s["thinking"] is True and s["max_output_tokens"] == 32768
    q = json.loads((tmp_path / "r" / "summary.json").read_text())["per_question"][0]
    assert {"dataset", "budget_infeasible", "gold", "answer", "score", "evidence_hit", "cannot_determine"} <= set(q)
    if setting == "active_v2":
        step = next(r for r in eps[0] if r["type"] == "step")
        assert {"action", "cannot_move", "invalid", "pose", "to", "odometry"} <= set(step)


@pytest.mark.parametrize("model", MOCKS)
@pytest.mark.parametrize("setting,budget", [("active_v2", 40), ("passive", 8), ("oracle_route", 40), ("blind", 40)])
def test_requests_match_prompts_verbatim_and_sizes(data, model, setting, budget, tmp_path):
    """Sent requests: the text is identical word for word to what prompts.py assembles, with no extra text; all images are 640x480 or 480x640."""
    setup(data, model, steps=3)
    seen = []
    orig = (OpenAIAPIClient if model == "mock-openai" else AnthropicClient).complete

    def spy(self, system, turns, tag=None):
        comp = orig(self, system, turns, tag)
        seen.append((system, turns, self.transport.requests[-1]))
        return comp

    cls = OpenAIAPIClient if model == "mock-openai" else AnthropicClient
    cls.complete = spy
    try:
        run(data, model, tmp_path, setting, budget, workers=1)
    finally:
        cls.complete = orig
    assert seen
    for system, turns, body in seen:
        msgs = body["messages"]
        if model == "mock-openai":
            assert msgs[0] == {"role": "system", "content": system}
            msgs = msgs[1:]
            assert "reasoning_effort" in body and body["reasoning_effort"] == "high" and body["response_format"] == {"type": "json_object"}
            assert body["max_completion_tokens"] == 32768 and "temperature" not in body
        else:
            assert body["system"] == [{"type": "text", "text": system}]
            assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": REPLY_SCHEMA}}
            assert body["max_tokens"] == 32768 and "response_format" not in body and "fallbacks" not in body
        assert len(msgs) == len(turns)
        for m, t in zip(msgs, turns):
            assert m["role"] == t["role"]
            if "parts" not in t:
                assert m["content"] == t["text"]
                continue
            assert text_blocks(m["content"]) == [p[1] for p in t["parts"] if p[0] == "text"]
            imgs = [b for b in m["content"] if b["type"] != "text"]
            assert len(imgs) == sum(p[0] == "image" for p in t["parts"])
            for b in imgs:
                d = b["source"]["data"] if model == "mock-anthropic" else b["image_url"]["url"].split(",", 1)[1]
                assert image_size(d) in ((640, 480), (480, 640))
    if setting == "active_v2":
        assert seen[0][0].startswith(P.SYSTEM_ACTIVE_V2.split("\n")[0]) and "A move covers about 0.3 m." in seen[0][0]


def test_same_image_bytes_across_providers(data, tmp_path):
    bodies = {}
    for model in MOCKS:
        c = client(data, model)
        c.complete("sys", turns_with_image(), tag={"mode": "active", "t": 0, "episode": "g"})
        b = c.transport.requests[-1]["messages"]
        bodies[model] = b
    a = bodies["mock-anthropic"][0]["content"][1]["source"]["data"]
    o = bodies["mock-openai"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]
    assert a == o and bodies["mock-anthropic"][0]["content"][1]["source"]["media_type"] == "image/jpeg"


def test_anthropic_cache_marker_only_on_start_image(data, tmp_path):
    setup(data, "mock-anthropic", steps=12)
    reqs = []
    orig = WireMock.__call__
    WireMock.__call__ = lambda self, body: (reqs.append(copy.deepcopy(body)), orig(self, body))[1]
    try:
        run(data, "mock-anthropic", tmp_path, gids=TWO[:1])
    finally:
        WireMock.__call__ = orig
    assert len(reqs) == 13
    first = json.dumps([reqs[0]["system"], reqs[0]["messages"][0]], sort_keys=True)
    for b in reqs:
        marks = [(i, j) for i, m in enumerate(b["messages"]) if isinstance(m["content"], list) for j, x in enumerate(m["content"]) if "cache_control" in x]
        assert marks == [(0, 1)]                                    # one mark, on the first image of turn 0 (START)
        assert b["messages"][0]["content"][1]["type"] == "image"
        assert b["messages"][0]["content"][1]["cache_control"] == {"type": "ephemeral"}
        assert json.dumps([b["system"], b["messages"][0]], sort_keys=True) == first   # the fixed prefix does not change
        assert sum(x["type"] == "image" for m in b["messages"] if isinstance(m["content"], list) for x in m["content"]) <= 9
        assert "cache_control" not in b["system"][0]
    us = [c["usage"] for c in calls(tmp_path / "r") if c["type"] == "call"]
    assert us[0]["cache_write_tokens"] > 0 and us[0]["cached_input_tokens"] == 0
    assert all(u["cached_input_tokens"] == us[0]["cache_write_tokens"] and u["cache_write_tokens"] == 0 for u in us[1:])


@pytest.mark.parametrize("setting,budget", [("blind", 40), ("passive", 8), ("random_walk", 40)])
def test_anthropic_no_cache_marker_on_single_call_settings(data, tmp_path, setting, budget):
    run(data, "mock-anthropic", tmp_path, setting, budget, gids=TWO[:1])
    us = [c for c in calls(tmp_path / "r") if c["type"] == "call"]
    assert all(c["usage"]["cache_write_tokens"] == 0 and c["usage"]["cached_input_tokens"] == 0 for c in us)


def test_anthropic_history_is_text_only_and_skips_empty(data):
    c = client(data, "mock-anthropic")
    (sysb, msgs), _ = c.to_messages("sys", [{"role": "user", "parts": [("text", "a")]}, {"role": "assistant", "text": ""},
                                            {"role": "user", "parts": [("text", "b")]}, {"role": "assistant", "text": '{"action": "left"}'},
                                            {"role": "user", "parts": [("text", "c")]}])
    assert [m["role"] for m in msgs] == ["user", "user", "assistant", "user"]
    assert msgs[2] == {"role": "assistant", "content": '{"action": "left"}'}


# ---------------------------------------------------------------- usage and cost
def test_anthropic_usage_and_cost_from_raw():
    raw = {"input_tokens": 1000, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 300,
           "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 100},
           "output_tokens": 500, "output_tokens_details": {"thinking_tokens": 400}}
    u = anthropic_usage(raw)
    assert u == {"input_tokens": 3300, "cached_input_tokens": 2000, "cache_write_tokens": 300, "cache_write_12h_tokens": 100,
                 "output_tokens": 500, "reasoning_tokens": 400}
    p = R.to_mcfg("opus-5.5", R.load_registry()["models"]["opus-5.5"])["prices_usd_per_mtok"]
    assert p == {"input": 4.0, "cached_input": 0.2, "cache_write": 5.0, "cache_write_12h": 8.0, "output": 20.0}
    assert CostTracker(p, 1).cost_of(u) == pytest.approx((1000 * 4 + 2000 * 0.2 + 200 * 5 + 100 * 8 + 500 * 20) / 1e6)


def test_openai_cost_uses_cache_read_and_write(data, tmp_path):
    c = client(data, "mock-openai", cost=ApiCostTracker(data.ecfg["models"]["mock-openai"]["prices_usd_per_mtok"], 10), log=tmp_path / "c.jsonl")
    comp = c.complete("s" * 8000, [{"role": "user", "parts": [("text", "x")]}], tag={"mode": "blind", "t": 0, "episode": "g"})
    r = calls_from(tmp_path / "c.jsonl")[0]
    u = r["usage_raw"]["prompt_tokens_details"]
    assert comp.usage["cache_write_tokens"] == u["cache_write_tokens"] > 0
    assert comp.cost_usd == pytest.approx(u["cache_write_tokens"] * 2.5 / 1e6 + (r["usage"]["input_tokens"] - u["cache_write_tokens"]) * 2 / 1e6 + r["usage"]["output_tokens"] * 10 / 1e6)


def calls_from(p):
    return [json.loads(x) for x in open(p)]


# ---------------------------------------------------------------- replies and errors
@pytest.mark.parametrize("model", MOCKS)
def test_invalid_replies_counted_as_model(data, model, tmp_path):
    setup(data, model, policy="invalid")
    st = run(data, model, tmp_path)
    assert st["status"] == "complete"
    eps = complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    assert all(e[-1]["forced"] == "invalid_replies" for e in eps)
    assert all(r["invalid"] for e in eps for r in e if r["type"] == "step")


@pytest.mark.parametrize("model", MOCKS)
@pytest.mark.parametrize("policy", ["rate_limit", "server_error", "stream_cut"])
def test_service_errors_retried_not_counted_invalid(data, model, policy, tmp_path):
    setup(data, model, policy=policy, steps=2, fail_times=2)
    st = run(data, model, tmp_path)
    assert st["status"] == "complete" and st["completed"] == 2
    rs = calls(tmp_path / "r")
    errs = [c for c in rs if c["type"] == "error"]
    assert errs and all(c["retryable"] for c in errs)
    assert len(errs) == 2 * len([c for c in rs if c["type"] == "call"])
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["invalid_steps"] == 0 and s["parse_failures"] == 0
    if policy == "stream_cut":
        assert all(c.get("possibly_billed") for c in errs) and st["usage"]["possibly_billed_errors"] == len(errs)


@pytest.mark.parametrize("model", MOCKS)
def test_empty_reply_retried_and_billed(data, model, tmp_path):
    setup(data, model, policy="empty", steps=2, fail_times=1)
    st = run(data, model, tmp_path)
    assert st["status"] == "complete"
    rs = calls(tmp_path / "r")
    errs = [c for c in rs if c["type"] == "error"]
    assert errs and all(c["cost_usd"] > 0 and c["usage"]["output_tokens"] > 0 and "empty reply" in c["error"] for c in errs)
    total = sum(c.get("cost_usd") or 0 for c in rs)
    assert st["spent_usd"] == pytest.approx(total, abs=1e-5) and CostTracker.spent_in_log(tmp_path / "r" / "calls.jsonl") == pytest.approx(total)
    assert st["usage"]["output_tokens"] == sum(c["usage"]["output_tokens"] for c in rs)
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["invalid_steps"] == 0


@pytest.mark.parametrize("model", MOCKS)
def test_exhausted_service_errors_stop_and_resume(data, model, tmp_path):
    setup(data, model, policy="dead")
    st = run(data, model, tmp_path)
    assert st["status"] == "stopped_server_error" and st["completed"] == 0
    assert not complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    assert sum(c["type"] == "error" for c in calls(tmp_path / "r")) == data.ecfg["llm"]["retry"]["max_retries"] + 1
    setup(data, model, policy="normal", steps=2)
    st = run(data, model, tmp_path, resume=True)
    assert st["status"] == "complete" and st["completed"] == 2


@pytest.mark.parametrize("model", MOCKS)
def test_fatal_not_retried(data, model, tmp_path):
    setup(data, model, policy="fatal")
    st = run(data, model, tmp_path)
    assert st["status"] == "stopped_fatal_error" and st["completed"] == 0
    errs = [c for c in calls(tmp_path / "r") if c["type"] == "error"]
    assert len(errs) == 1 and errs[0]["status"] == 400 and not errs[0]["retryable"]


@pytest.mark.parametrize("model", MOCKS)
@pytest.mark.parametrize("status", [400, 401, 403])
def test_fatal_http_stops_without_retry_and_hides_key(data, model, status, tmp_path):
    setup(data, model, policy="fatal", fatal_status=status)
    st = run(data, model, tmp_path)
    assert st["status"] == "stopped_fatal_error" and f"HTTP {status}" in st["error"]
    rs = calls(tmp_path / "r")
    assert len(rs) == 1 and rs[0]["gave_up"] and rs[0]["status"] == status
    assert "abc123" not in (tmp_path / "r" / "calls.jsonl").read_text() + (tmp_path / "r" / "status.json").read_text()


class FailsAfter(WireMock):
    """Shared by all workers: returns 401 from the n-th request on."""

    def __init__(self, mcfg, n):
        super().__init__(mcfg)
        self.limit, self.lock, self.count = n, threading.Lock(), 0

    def __call__(self, body):
        with self.lock:
            self.count += 1
            k = self.count
        if k >= self.limit:
            return {"_http_status": 401, "_text": "invalid x-api-key"}
        return super().__call__(body)


@pytest.mark.parametrize("model", MOCKS)
def test_fatal_stops_other_workers(data, model, tmp_path):
    """Under concurrency, when one worker hits a fatal error, the next call of every other worker is not sent and no new episode starts."""
    setup(data, model, steps=30)
    t = FailsAfter(data.ecfg["models"][model], 5)
    gids = ["arkit_g_000000", "arkit_g_000003", "arkit_g_000005", "scannetpp_g_000009"]
    st = run(data, model, tmp_path, gids=gids, workers=3, client_kwargs={"transport": t})
    assert st["status"] == "stopped_fatal_error" and st["completed"] == 0
    assert t.count <= 5 + 2                       # after the fatal error, at most 2 requests still in flight
    assert not complete_episodes(tmp_path / "r" / "trajectories.jsonl")


def test_abort_wakes_waiting_workers():
    t = ApiCostTracker({"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 1e6}, cap_usd=4)
    t.reserve(0, 3)
    got = []

    def waiter():
        try:
            t.reserve(0, 3)                       # does not fit; waits for the call in flight
        except ModelError as e:
            got.append(e)

    th = threading.Thread(target=waiter)
    th.start()
    e = http_error(401, "x")
    t.abort(e)
    th.join(timeout=10)
    assert got == [e] and e.stop_status == "stopped_fatal_error"


def test_redact():
    assert "sk-abc" not in redact("Incorrect API key provided: sk-abc123***wxyz.")
    assert "my-secret" not in redact("key my-secret-value", "my-secret-value")
    assert "sk-abc" not in str(http_error(401, "sk-abc123***wxyz"))


@pytest.mark.parametrize("model", MOCKS)
@pytest.mark.parametrize("policy", ["truncated", "refusal"])
def test_truncation_and_refusal_are_model_replies(data, model, policy, tmp_path):
    """Limit filled or refusal: the reply comes from the model, no retry, counts as no answer (same decision as for open-weight models)."""
    setup(data, model, policy=policy)
    st = run(data, model, tmp_path, "passive", 8)
    assert st["status"] == "complete"
    rs = calls(tmp_path / "r")
    assert all(c["type"] == "call" for c in rs)
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["parse_failures"] == 2


def test_anthropic_empty_assistant_after_truncation_keeps_running(data, tmp_path):
    setup(data, "mock-anthropic", policy="truncated")
    st = run(data, "mock-anthropic", tmp_path, gids=TWO[:1])
    assert st["status"] == "complete"
    assert complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0][-1]["forced"] == "invalid_replies"


def test_stream_events():
    good = WireMock({"provider": "anthropic", "mock": {"policy": "normal", "steps": 0, "fail_times": 0}, "tokens_per_image": 414})
    body = {"model": "m", "stream": True, "system": [{"type": "text", "text": "s"}], "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "Q1 (numeric, meters) x"}]}]}
    ev = good(body)
    m, ttft = merge_events(ev, 0.0, fake=True)
    assert m["stop_reason"] == "end_turn" and m["usage"]["output_tokens"] > 1 and m["usage"]["output_tokens_details"]["thinking_tokens"] == 30
    assert [b["type"] for b in m["content"]] == ["thinking", "text"] and m["content"][0]["thinking"] and ttft is not None
    with pytest.raises(ModelError) as e:
        merge_events(ev[:-1], 0.0, fake=True)
    assert e.value.retryable and e.value.billed["possibly_billed"]
    with pytest.raises(ModelError) as e:
        merge_events(ev[:3] + [{"type": "error", "error": {"type": "overloaded_error", "message": "x"}}], 0.0, fake=True)
    assert e.value.retryable


def test_http_error_classes():
    for s in (408, 409, 429, 500, 503, 504, 529):
        assert http_error(s, "x").retryable
    for s in (400, 401, 402, 403, 404, 413):
        assert not http_error(s, "x").retryable
    assert http_error(429, "x", {"retry-after": "7"}).retry_after == 7
    assert not http_error(429, '{"error": {"details": {"error_code": "enforced_spend_limit_reached"}}}').retryable
    assert not http_error(429, '{"error": {"code": "insufficient_quota"}}').retryable


class FakeResp:
    def __init__(self, status, body=None, headers=None, lines=None):
        self.status_code, self._body, self.headers, self._lines = status, body or {}, headers or {}, lines or []
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


@pytest.mark.parametrize("model", ["sol", "opus-5.5"])
def test_key_only_in_header_never_logged(data, model, tmp_path, monkeypatch):
    """Goes through the real _post / stream parsing (requests.post replaced, no network): the key is read only from the registered env var, goes only into the request header, never into the log."""
    import requests
    fake = "fake-for-test-1234567890"
    monkeypatch.setenv(FAKE_VAR, fake)
    m = {**data.ecfg["models"][model], "key_env": FAKE_VAR, "prices_confirmed": True}
    m["reasoning"] = {**m["reasoning"], "approved": True}
    wm = WireMock({**m, "mock": {"policy": "normal", "steps": 1, "fail_times": 0}})
    seen = []

    def post(url, json=None, headers=None, timeout=None, stream=False):
        seen.append((url, headers, timeout))
        out = wm(json)
        if not stream:
            return FakeResp(200, out)
        return FakeResp(200, lines=[f"data: {__import__('json').dumps(e)}" for e in out])

    monkeypatch.setattr(requests, "post", post)
    log = tmp_path / "c.jsonl"
    c = make_client(m, data.ecfg["llm"], ApiCostTracker(m["prices_usd_per_mtok"], 5), log)
    comp = c.complete("You are exploring", turns_with_image(), tag={"mode": "active", "t": 0, "episode": "g"})
    assert json.loads(comp.text)["action"] == "forward"
    url, headers, timeout = seen[0]
    assert fake in json.dumps(headers) and timeout == (10, 1200 if model == "sol" else 600) and url == m["base_url"]
    if model == "opus-5.5":
        assert headers["x-api-key"] == fake and headers["anthropic-version"] == "2026-10-21"
    assert fake not in log.read_text()


def test_http_429_with_retry_after_waits(data, tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv(FAKE_VAR, "fake-for-test")
    m = {**data.ecfg["models"]["sol"], "key_env": FAKE_VAR, "prices_confirmed": True}
    m["reasoning"] = {**m["reasoning"], "approved": True}
    wm = WireMock({**m, "mock": {"policy": "normal", "steps": 0, "fail_times": 0}})
    n = [0]

    def post(url, json=None, headers=None, timeout=None, stream=False):
        n[0] += 1
        return FakeResp(429, {"error": "slow"}, {"retry-after": "90"}) if n[0] == 1 else FakeResp(200, wm(json))

    monkeypatch.setattr(requests, "post", post)
    waits = []
    c = make_client(m, data.ecfg["llm"], None, tmp_path / "c.jsonl", sleep=waits.append)
    c.complete("sys", [{"role": "user", "parts": [("text", "Q1 (numeric, meters) x")]}])
    assert waits == [90.0]


def test_timeout_is_retried_and_marked_possibly_billed(data, tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv(FAKE_VAR, "fake-for-test")
    m = {**data.ecfg["models"]["sol"], "key_env": FAKE_VAR, "prices_confirmed": True}
    m["reasoning"] = {**m["reasoning"], "approved": True}
    wm = WireMock({**m, "mock": {"policy": "normal", "steps": 0, "fail_times": 0}})
    n = [0]

    def post(url, json=None, headers=None, timeout=None, stream=False):
        n[0] += 1
        if n[0] == 1:
            raise requests.exceptions.ReadTimeout("mock")
        return FakeResp(200, wm(json))

    monkeypatch.setattr(requests, "post", post)
    log = tmp_path / "c.jsonl"
    c = make_client(m, data.ecfg["llm"], None, log, sleep=lambda s: None)
    c.complete("sys", [{"role": "user", "parts": [("text", "Q1 (numeric, meters) x")]}])
    assert CostTracker.usage_in_log(log)["possibly_billed_errors"] == 1


# ---------------------------------------------------------------- cost cap and resume
@pytest.mark.parametrize("model", MOCKS)
def test_cost_cap_stop_and_resume(data, model, tmp_path):
    setup(data, model, steps=3)
    m = data.ecfg["models"][model]
    data.ecfg["models"][model] = {**m, "max_output_tokens": 200}   # make the cap fall in the middle of the run (reservations use the maximum output)
    full = run(data, model, tmp_path, name="full", gids=TWO)["spent_usd"]
    cap = full * 0.6
    st = run(data, model, tmp_path, cap=cap)
    assert st["status"] == "stopped_cost_cap" and st["spent_usd"] <= cap and st["completed"] < 2
    st2 = run(data, model, tmp_path, resume=True, cap=cap)      # cap not raised: stops immediately
    assert st2["status"] == "stopped_cost_cap" and st2["spent_usd"] <= cap
    st3 = run(data, model, tmp_path, resume=True, cap=full * 3)
    assert st3["status"] == "complete" and st3["completed"] == 2
    eps = complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    assert sorted(e[0]["group_id"] for e in eps) == sorted(TWO)
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0


def test_reservation_keeps_concurrent_calls_under_cap():
    prices = {"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 1e6}   # $1 per output token
    t = ApiCostTracker(prices, cap_usd=10)
    done, stopped, peak = [], [], [0.0]

    def worker():
        while True:
            try:
                t.reserve(0, 3)
            except CostCapExceeded:
                stopped.append(1)
                return
            peak[0] = max(peak[0], t.spent + t._held)
            t.add({"output_tokens": 3})
            done.append(1)

    ths = [threading.Thread(target=worker) for _ in range(8)]
    for x in ths:
        x.start()
    for x in ths:
        x.join(timeout=30)
    assert len(stopped) == 8 and t.spent == 9 == len(done) * 3 and peak[0] <= 10 and t._held == pytest.approx(0)


@pytest.mark.parametrize("model", MOCKS)
def test_concurrent_episodes(data, model, tmp_path):
    setup(data, model, steps=2)
    gids = ["arkit_g_000000", "arkit_g_000003", "arkit_g_000005", "scannetpp_g_000009"]
    st = run(data, model, tmp_path, gids=gids, workers=3)
    assert st["status"] == "complete" and st["completed"] == 4 and st["workers"] == 3
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0
    assert CostTracker.spent_in_log(tmp_path / "r" / "calls.jsonl") == pytest.approx(st["spent_usd"], abs=1e-5)


def test_estimate_follows_repeats(data):
    one = E.run_estimate(data, "sol", TWO, ["active_v2", "random_walk_40", "blind"])
    three = E.run_estimate(data, "sol", TWO, ["active_v2", "random_walk_40", "blind"], n=3)
    get = lambda rows, s: next(r for r in rows if r["setting"] == s)
    assert get(one, "active_v2")["reps"] == 1 and get(three, "active_v2")["reps"] == 3
    assert get(three, "active_v2")["cost_usd"] == pytest.approx(3 * get(one, "active_v2")["cost_usd"], rel=0.05)
    assert get(three, "blind")["cost_usd"] == get(one, "blind")["cost_usd"]


def test_shared_cap_scope_across_runs_and_probe(tmp_path):
    root = tmp_path
    a = root / "m" / "blind" / "tiny_r0"
    b = root / "m" / "active_v2" / "tiny_r1"
    c = root / "m" / "active_v2" / "other_r0"
    p = root / "m" / "_probe" / "tiny"
    for d, v in ((a, 1.0), (b, 2.0), (c, 4.0)):
        d.mkdir(parents=True)
        (d / "calls.jsonl").write_text(json.dumps({"type": "call", "cost_usd": v}) + "\n")
    p.mkdir(parents=True)
    (p / "m.jsonl").write_text(json.dumps({"type": "call", "cost_usd": 0.5}) + "\n")
    assert E.scope_spent(root, "m", "tiny") == pytest.approx(3.5)


def test_plan_repeats():
    r = E.plan_runs("all", E.SETTINGS)
    assert len(r) == 8 and {x["name"] for x in r} == {"all_r0"}                 # registry default: 1 run
    r = E.plan_runs("all", E.SETTINGS, 3)
    assert len(r) == 12 and sum(x["setting"] == "active_v2" for x in r) == 3 and sum(x["setting"] == "random_walk_40" for x in r) == 3
    assert sum(x["setting"] == "blind" for x in r) == 1
    assert [x["seed"] for x in r if x["setting"] == "active_v2"] == [0, 1, 2]
    assert R.load_registry()["repeats"] == {"active_v2": 1, "random_walk_40": 1}


def test_env_json_and_seal(data, tmp_path):
    reg = R.load_registry()
    e = R.entries(reg)["mock-anthropic"]
    setup(data, "mock-anthropic", steps=1)
    run(data, "mock-anthropic", tmp_path, gids=TWO[:1], env_info=R.env_info("mock-anthropic", e), seal=True)
    d = tmp_path / "r"
    env = json.loads((d / "env.json").read_text())
    assert {"provider", "model_id", "reasoning", "max_tokens", "prices_usd_per_mtok", "data_version", "freeze_manifest_sha256", "code", "started_at", "ended_at"} <= set(env)
    assert env["provider"] == "anthropic" and env["data_version"] == "merged_v1.0" and env["key_env"] is None
    assert (d / "SHA256SUMS").exists() and not os.access(d / "calls.jsonl", os.W_OK)
    for p in d.rglob("*"):
        p.chmod(p.stat().st_mode | 0o200)
    d.chmod(d.stat().st_mode | 0o200)


# ---------------------------------------------------------------- probe and estimate
@pytest.mark.parametrize("model", MOCKS)
def test_probe_passes(data, model, tmp_path):
    r = probe_model(data, model, tmp_path, cap=5.0)
    assert r["passed"] and r["checks"] == {"images": True, "thinking": True, "json": True, "usage": True}
    assert r["tokens_per_image"] == data.ecfg["models"][model]["tokens_per_image"]
    assert len(r["calls"]) == 3 and "logprobs" not in json.dumps(r["params"])
    if model == "mock-anthropic":
        assert r["cache_write_on_first"] > 0 and r["cache_read_on_repeat"] == r["cache_write_on_first"]


@pytest.mark.parametrize("policy,check", [("invalid", "json"), ("fatal", "thinking")])
def test_probe_fails_on_bad_replies(data, policy, check, tmp_path):
    setup(data, "mock-anthropic", policy=policy)
    r = probe_model(data, "mock-anthropic", tmp_path, cap=5.0)
    assert not r["passed"] and not r["checks"][check]


def test_probe_stops_at_first_fatal(data, tmp_path):
    setup(data, "mock-openai", policy="fatal", fatal_status=401)
    r = probe_model(data, "mock-openai", tmp_path, cap=5.0)
    assert r["fatal"] and len(r["calls"]) == 1 and not r["passed"]


def test_probe_refused_without_key(data, tmp_path):
    r = probe_model(data, "opus-5.5", tmp_path, cap=5.0)
    assert not r["passed"] and "ANTHROPIC_API_KEY" in r["error"] and not r["calls"]


def test_probe_cap_too_small(data, tmp_path):
    r = probe_model(data, "mock-anthropic", tmp_path, cap=0.001)
    assert not r["passed"] and not any(c["accepted"] for c in r["calls"])


def test_estimate_uses_provider_cache_rule(data):
    gids = TWO
    a = E.run_estimate(data, "opus-5.5", gids, ["active_v2", "blind"], n=3)
    o = E.run_estimate(data, "sol", gids, ["active_v2", "blind"], n=3)
    act = next(r for r in a if r["setting"] == "active_v2" and r["steps"] == 40)
    assert act["reps"] == 3 and act["calls"] == 41 * 2 * 3
    eps = 6
    assert abs(act["cached_input_tokens"] - act["cache_write_tokens"] * 40) < 40 * eps    # each episode writes once and reads 40 times
    assert all(r["cached_input_tokens"] == 0 and r["cache_write_tokens"] == 0 for r in a if r["setting"] == "blind")
    assert next(r for r in o if r["setting"] == "active_v2" and r["steps"] == 40)["cached_input_tokens"] > 0
    assert act["cost_usd"] > 0


def test_probe_overrides_estimate(data):
    a = E.run_estimate(data, "opus-5.5", TWO, ["passive_8"])
    b = E.run_estimate(data, "opus-5.5", TWO, ["passive_8"], probe={"tokens_per_image": 828, "output_tokens": [40, 40, 40]})
    assert b[0]["input_tokens"] > a[0]["input_tokens"]


def test_verify_calls_detects_changes(data, tmp_path):
    setup(data, "mock-anthropic", steps=2)
    run(data, "mock-anthropic", tmp_path, gids=TWO)
    sys_ok, first_ok = E.design_system_prompts(), E.first_turn_regexes()
    p = tmp_path / "r" / "calls.jsonl"
    r = E.verify_calls(p, sys_ok, first_ok)
    assert r["calls"] > 0 and r["bad_system"] == r["bad_first"] == r["bad_step"] == r["bad_size"] == r["base64"] == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text(p.read_text().replace("You are exploring an indoor scene", "You explore an indoor scene"))
    assert E.verify_calls(bad, sys_ok, first_ok)["bad_system"] > 0


# ---------------------------------------------------------------- guards of the one-click script (no network)
def sh(*args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_API_KEY", "ANTHROPIC_API_KEY")) and k != "SLURM_JOB_ID"}
    e["ACTSPACE_PYTHON"] = sys.executable
    e.update(env or {})
    return subprocess.run(["bash", str(ROOT / "scripts" / "run_api_eval.sh"), *args], capture_output=True, text=True, env=e, timeout=120)


@pytest.mark.parametrize("args", [("opus-5.5", "tiny"), ("opus-5.5", "tiny", "--cap"), ("opus-5.5", "tiny", "--cap", "abc"), ("sol", "smoke", "--settings", "blind")])
def test_script_refuses_without_cap(args):
    r = sh(*args)
    assert r.returncode == 2 and "--cap" in r.stdout


def test_script_stops_without_key(tmp_path):
    r = sh("opus-5.5", "smoke", "--cap", "1", "--key-file", str(tmp_path / "none.env"), env={"API_OUT_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "does not exist" in r.stdout and "freeze verify" not in r.stdout
    f = fake_key_file(tmp_path)
    f.chmod(0o644)
    r = sh("opus-5.5", "smoke", "--cap", "1", "--key-file", str(f), env={"API_OUT_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "must be 600" in r.stdout and "freeze verify" not in r.stdout


def test_tests_can_never_read_the_real_key_file(tmp_path):
    """Test subprocesses inherit PYTEST_CURRENT_TEST, so the default key file is never read. Without a key file, keys are read from env vars; a missing key stops before verify."""
    with pytest.raises(K.KeyFileError, match="tests must not read"):
        K.read_key_file(K.DEFAULT_KEY_FILE)
    r = sh("opus-5.5", "smoke", "--cap", "1", env={"API_OUT_ROOT": str(tmp_path), "ACTSPACE_KEY_FILE": str(tmp_path / "no_such_file.env")})
    assert r.returncode == 2 and "environment" in r.stdout and "freeze verify" not in r.stdout


def test_keys_from_environment_variables(tmp_path, monkeypatch):
    """Without a key file, keys come from env vars: OpenAI uses key_env, Anthropic ANTHROPIC_API_KEY_1..n or a single ANTHROPIC_API_KEY; only numbers are returned."""
    monkeypatch.setattr(K, "DEFAULT_KEY_FILE", tmp_path / "no_such_file.env")
    for k in [x for x in os.environ if x.startswith(("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-for-test")
    info = K.setup_keys(tmp_path, "sol", {"provider": "openai", "key_env": "OPENAI_API_KEY"})
    assert info["note"] == "OPENAI_API_KEY: set" and "fake-openai" not in json.dumps(info)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-for-test")
    info = K.setup_keys(tmp_path, "opus-5.5", {"provider": "anthropic", "key_env": "ANTHROPIC_API_KEY"})
    assert info["keys"] == [1] and "fake-anthropic" not in json.dumps(info)
    K.activate(None)


@pytest.mark.parametrize("field", ["prices", "reasoning"])
def test_check_stops_on_closed_switch(field, monkeypatch, capsys, tmp_path):
    reg = R.load_registry()
    e = copy.deepcopy(reg["models"]["sol"])
    if field == "prices":
        e["prices_confirmed"] = False
    else:
        e["reasoning"]["approved"] = False
    e["key_env"] = FAKE_VAR
    reg["models"]["sol"] = e
    monkeypatch.setattr(R, "load_registry", lambda: reg)
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    f = tmp_path / "k.env"
    f.write_text(f"{FAKE_VAR}=fake-for-test\n")
    f.chmod(0o600)
    with pytest.raises(SystemExit) as x:
        E.cmd_check(SimpleNamespace(model="sol", key_file=str(f)))
    out = capsys.readouterr().out
    assert x.value.code == 2 and f"{FAKE_VAR}: set" in out and "stopped" in out and "fake-for-test" not in out


def test_job_scripts_never_expose_the_key():
    """The key is read from the key file by python inside the job: the script never handles the key, does not use --export=KEY=value, and does not print the environment."""
    for name in ("run_api_eval.sh", "run_api_job.sbatch"):
        code = "\n".join(x for x in (ROOT / "scripts" / name).read_text().splitlines() if not x.startswith("#") or x.startswith("#SBATCH"))
        for bad in ("--export", "set -x", "set -o xtrace", "printenv", "API_KEY"):
            assert bad not in code, (name, bad)
        assert not re.search(r"(^|[;|&\s])env(\s|$)", code), name


def test_script_local_refused_outside_slurm(tmp_path):
    r = sh("mock-openai", "smoke", "--cap", "1", "--local", env={"API_OUT_ROOT": str(tmp_path)})
    assert r.returncode == 2 and "Slurm" in r.stdout


def shims(tmp_path):
    """Stand-in for sbatch: only records the arguments, submits no job."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_ / "sbatch").write_text(f"""#!/bin/bash
echo "$@" >> {log}
echo 777
""")
    (bin_ / "sbatch").chmod(0o755)
    return {"PATH": f"{bin_}:{os.environ['PATH']}", "API_OUT_ROOT": str(tmp_path / "out")}, log


def test_script_submits_one_job(tmp_path, synth):   # synth: the script runs freeze verify
    env, log = shims(tmp_path)
    r = sh("mock-anthropic", "all", "--cap", "3", "--repeats", "3", "--resume", "--time", "2-00:00:00", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    jobs = log.read_text().splitlines()
    assert len(jobs) == 1 and "--dependency" not in jobs[0] and "--wait" not in jobs[0]
    j = jobs[0]
    assert "-t 2-00:00:00" in j and "scripts/run_api_job.sbatch" in j and "--repeats 3" in j and "--resume" in j and "--cap 3" in j
    assert "submitted job 777" in r.stdout and "tail -f" in r.stdout


def test_script_smoke_waits_and_reports_truncation(tmp_path, synth):   # synth: the script runs freeze verify
    env, log = shims(tmp_path)
    r = sh("mock-openai", "smoke", "--cap", "3", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    j = log.read_text().splitlines()
    assert len(j) == 1 and "--wait" in j[0] and " -t " not in j[0] and "== truncation rate" in r.stdout    # without --time, the partition limit applies


@pytest.mark.parametrize("args", [("--repeats", "0"), ("--repeats", "x"), ("--time",)])
def test_script_rejects_bad_options(args):
    assert sh("mock-openai", "smoke", "--cap", "1", *args).returncode == 2


# ---------------------------------------------------------------- truncation rate and smoke report
def test_truncation_in_status(data, tmp_path):
    setup(data, "mock-anthropic", policy="truncated")
    st = run(data, "mock-anthropic", tmp_path, "passive", 8)
    assert st["truncation_rate"] == 1.0 and st["usage"]["truncated_calls"] == 2
    assert json.loads((tmp_path / "r" / "status.json").read_text())["truncation_rate"] == 1.0
    setup(data, "mock-anthropic", policy="normal")
    assert run(data, "mock-anthropic", tmp_path, "passive", 8, name="r2")["truncation_rate"] == 0


def write_status(root, short, setting, name, calls_n, trunc, status="complete"):  # noqa: D103
    d = root / short / setting / name
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": status, "usage": {"calls": calls_n, "truncated_calls": trunc},
                                               "truncation_rate": trunc / calls_n}))


@pytest.mark.parametrize("trunc,warn", [(1, False), (2, True), (5, True)])
def test_truncation_warning_threshold(trunc, warn, tmp_path, monkeypatch, capsys):
    """Over all calls: 2 out of 100 calls is 2%, exactly the threshold."""
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    write_status(tmp_path, "m", "active_v2", "smoke_r0", 97, trunc)
    write_status(tmp_path, "m", "blind", "smoke_r0", 3, 0)
    E.cmd_truncation(SimpleNamespace(model="m", prefix="smoke", settings="blind,active_v2"))
    out = capsys.readouterr().out
    assert ("WARNING" in out) is warn and f"{trunc} / 100" in out


def test_truncation_handles_missing_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    E.cmd_truncation(SimpleNamespace(model="m", prefix="smoke", settings=None))
    assert "WARNING" not in capsys.readouterr().out


def test_truncation_table_in_report(tmp_path):
    write_status(tmp_path, "m", "blind", "smoke_r0", 3, 1)
    md = "\n".join(E.truncation_md(E.truncation_rows(tmp_path, "m", "smoke", ["blind"])))
    assert "## Truncation rate" in md and "33.33%" in md


def test_truncation_table_states_the_limit_each_run_used(tmp_path):
    write_status(tmp_path, "m", "blind", "smoke_r0", 3, 1)
    write_status(tmp_path, "m", "passive_8", "smoke_r0", 3, 0)
    (tmp_path / "m" / "blind" / "smoke_r0" / "env.json").write_text(json.dumps({"max_tokens": 16384}))
    assert "(16384, thinking included)" in "\n".join(E.truncation_md(E.truncation_rows(tmp_path, "m", "smoke", ["blind", "passive_8"])))
    (tmp_path / "m" / "passive_8" / "smoke_r0" / "env.json").write_text(json.dumps({"max_tokens": 32768}))
    assert "(16384, 32768, thinking included)" in "\n".join(E.truncation_md(E.truncation_rows(tmp_path, "m", "smoke", ["blind", "passive_8"])))


def smoke_tree(data, model, tmp_path, monkeypatch):
    """Runs all smoke settings once through the fake transport, with the same directory layout as the main runs."""
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    setup(data, model, steps=4)
    gids = data.ecfg["llm"]["smoke_groups"]
    for r in E.plan_runs("smoke", E.SETTINGS):
        out = E.run_dir(tmp_path, model, r)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_llm(data, model, r["mode"], r["budget"], gids, out, cost_cap=10.0, print_estimate=False, tracker_cls=ApiCostTracker)
    p = tmp_path / model / "_probe" / "smoke"
    r = probe_model(data, model, p, cap=5.0)
    (p / "probe.json").write_text(json.dumps(r))


@pytest.mark.parametrize("model", MOCKS)
def test_smoke_report_contents(data, model, tmp_path, monkeypatch):
    smoke_tree(data, model, tmp_path, monkeypatch)
    reg = R.load_registry()
    md = "\n".join(SR.build(data, reg, model, "smoke", targets={"tiny": data.ecfg["llm"]["tiny_groups"][:3], "all": data.ecfg["llm"]["tiny_groups"][:6]}))
    for s in ("## 1. probe", "| images | pass |", f"probe returned: ['{model}']", "Measured input tokens per image", "Active from step 2 on, cache_read > 0", "active_v2 (answer)", "| Truncated (max_tokens filled) | 0 (0.00%) |",
              "| Refusals | 0 |", "| Invalid replies | 0 |", "possibly_billed", "Total (excluding probe)", "## 7. tiny and all recomputed from measurements"):
        assert s in md, s
    tpi = R.entries(reg)[model]["image"]["tokens_per_image"]
    assert f"Measured input tokens per image: {tpi}; registered value {tpi} (difference +0, +0.0%)" in md
    if model == "mock-anthropic":
        assert "cache_read > 0: 12 / 12 calls" in md


def test_smoke_recost_scales_with_groups_and_repeats(data, tmp_path, monkeypatch):
    smoke_tree(data, "mock-openai", tmp_path, monkeypatch)
    runs = SR.collect(tmp_path, "mock-openai", "smoke")
    g = data.ecfg["llm"]["smoke_groups"]
    rec = SR.recost(data, "mock-openai", runs, {"tiny": list(g), "smoke": list(g)})
    assert rec["tiny"]["blind"][0] == pytest.approx(rec["smoke"]["blind"][0])
    assert rec["tiny"]["active_v2"][1] > rec["tiny"]["active_v2"][0]      # using all 40 steps costs more
    for s in ("active_v2", "random_walk_40"):                              # same 3 groups, 1 run per registry: cost equals the actual smoke cost
        assert rec["tiny"][s][0] == pytest.approx(sum(r["spent"] for r in runs[s]), rel=0.02)
    assert rec["smoke"]["blind"][0] == pytest.approx(sum(r["spent"] for r in runs["blind"]), rel=0.02)


def test_smoke_counts_errors_and_billed_retries(data, tmp_path):
    setup(data, "mock-anthropic", policy="empty", steps=1, fail_times=1)
    run(data, "mock-anthropic", tmp_path, gids=TWO[:1])
    c = SR.counts({"active_v2": [SR.read_run(tmp_path / "r")]})
    assert c["billed_retries"] == c["service_errors"] > 0 and c["invalid"] == 0 and c["truncated"] == 0
    m = SR.per_episode([SR.read_run(tmp_path / "r")])
    assert m["episodes"] == 1 and m["output_tokens"] == sum(x["usage"]["output_tokens"] for x in calls(tmp_path / "r"))


# ---------------------------------------------------------------- single job
def job_args(model="mock-anthropic", prefix="s", settings=None, cap=20.0, repeats=None, resume=False, key_file=None):
    return SimpleNamespace(model=model, set="smoke", prefix=prefix, settings=settings, cap=cap, repeats=repeats, resume=resume, key_file=key_file)


@pytest.fixture()
def job(data, tmp_path, monkeypatch):
    """Runs cmd_job in-process: fake transport, results written to a temporary directory. Returns (run function, root directory)."""
    monkeypatch.setattr(E, "load", lambda verify=True: (R.load_registry(), data))
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))

    def go(**kw):
        try:
            E.cmd_job(job_args(**kw))
            return 0
        except SystemExit as e:
            return e.code
    return go, tmp_path


def run_paths(root, model="mock-anthropic", prefix="s"):
    return sorted(p.parent for p in (root / model).glob(f"*/{prefix}_r*/status.json"))


def st_of(d):
    return json.loads((d / "status.json").read_text())


@pytest.mark.parametrize("model", MOCKS)
def test_job_repeats_1_full_flow(data, job, model):
    go, root = job
    setup(data, model, steps=3)
    assert go(model=model) == 0
    runs = run_paths(root, model)
    assert len(runs) == 8 and {p.name for p in runs} == {"s_r0"}
    assert all(st_of(p)["status"] == "complete" and (p / "SHA256SUMS").exists() for p in runs)
    for p in runs:
        assert replay(data, p / "trajectories.jsonl")["mismatched_episodes"] == 0
        assert st_of(p)["spent_usd"] == pytest.approx(CostTracker.spent_in_log(p / "calls.jsonl"), abs=1e-5)   # each run's own cost
    md = (root / model / "_reports" / "s.md").read_text()
    assert "excludes run-to-run variation" in md and "## Truncation rate" in md and "| Active v2 | 1 |" in md and "| random_walk_40 | 1 |" in md
    prog = (root / model / "_progress" / "s" / "active_v2_r0.log").read_text().splitlines()
    assert "3/3" in prog[-2] and "job total" in prog[-2] and "complete" in prog[-1]


def test_job_repeats_3_resume_only_adds_missing_seeds(data, job):
    go, root = job
    setup(data, "mock-openai", steps=2)
    assert go(model="mock-openai") == 0
    before = {p: (p / "SHA256SUMS").read_text() for p in run_paths(root, "mock-openai")}
    spent0 = E.scope_spent(root, "mock-openai", "s")
    n0 = sum(len(calls(p)) for p in before)
    assert go(model="mock-openai", repeats=3, resume=True) == 0
    after = run_paths(root, "mock-openai")
    assert len(after) == 12 and {p: (p / "SHA256SUMS").read_text() for p in before} == before     # completed runs were not rerun
    new = [p for p in after if p not in before]
    assert sorted((p.parent.name, p.name) for p in new) == sorted((s, f"s_r{k}") for s in ("active_v2", "random_walk_40") for k in (1, 2))
    assert sum(len(calls(p)) for p in before) == n0
    assert E.scope_spent(root, "mock-openai", "s") == pytest.approx(spent0 + sum(st_of(p)["spent_usd"] for p in new), abs=1e-4)
    md = (root / "mock-openai" / "_reports" / "s.md").read_text()
    assert "| Active v2 | 3 |" in md and "| random_walk_40 | 3 |" in md and "| Blind | 1 |" in md
    assert "Settings with a single run (Blind, Passive-8" in md and "Active v2)" not in md.split("excludes run-to-run variation")[0]
    assert go(model="mock-openai", repeats=3, resume=True) == 0                                     # run again: does nothing
    assert sum(len(calls(p)) for p in after) == sum(len(calls(p)) for p in run_paths(root, "mock-openai"))


def test_job_refuses_existing_output_without_resume(data, job):
    """Incomplete artifacts exist without --resume: exits without spending anything (no probe either)."""
    go, root = job
    d = root / "mock-anthropic" / "blind" / "s_r0"
    d.mkdir(parents=True)
    (d / "trajectories.jsonl").write_text("")
    assert go(settings="blind") == 2
    assert not (root / "mock-anthropic" / "_probe").exists() and not (d / "calls.jsonl").exists()


def test_job_concurrency_and_cap_across_settings(data, job, monkeypatch):
    """Several settings run at once: requests in flight never exceed concurrency; cost never exceeds the shared cap; the whole job stops at the cap; after raising the cap, a resume completes it."""
    import time as _t
    go, root = job
    m = data.ecfg["models"]["mock-anthropic"]
    data.ecfg["models"]["mock-anthropic"] = {**m, "max_output_tokens": 300}
    setup(data, "mock-anthropic", steps=4)
    active, peak, lock = [0], [0], threading.Lock()
    orig = WireMock.__call__

    def slow(self, body):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _t.sleep(0.01)
        try:
            return orig(self, body)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(WireMock, "__call__", slow)
    assert go(cap=0.12) == 3
    assert peak[0] <= 3                                     # concurrency of mock-anthropic
    assert E.scope_spent(root, "mock-anthropic", "s") <= 0.12
    stopped = [p for p in run_paths(root) if not (p / "SHA256SUMS").exists()]
    assert stopped and all(st_of(p)["status"] == "stopped_cost_cap" for p in stopped)
    assert not (root / "mock-anthropic" / "_reports" / "s.md").exists()
    assert go(cap=5.0, resume=True) == 0
    assert all((p / "SHA256SUMS").exists() for p in run_paths(root)) and len(run_paths(root)) == 8
    for p in run_paths(root):
        eps = complete_episodes(p / "trajectories.jsonl")
        assert sorted(e[0]["group_id"] for e in eps) == sorted(data.ecfg["llm"]["smoke_groups"])
    assert peak[0] <= 3


class FatalAfter(WireMock):
    count, limit, lock, status = 0, 10 ** 9, threading.Lock(), 401

    def __call__(self, body):
        with FatalAfter.lock:
            FatalAfter.count += 1
            k = FatalAfter.count
        if k >= FatalAfter.limit:
            return {"_http_status": FatalAfter.status, "_text": "invalid key"}
        return super().__call__(body)


@pytest.mark.parametrize("status,code,word", [(401, 4, "stopped_fatal_error"), (500, 5, "stopped_server_error")])
def test_job_stops_everything_on_fatal_and_exhausted_retries(data, job, monkeypatch, status, code, word):
    """Fatal error and exhausted retries: all runs stop, no new episode starts and no more requests are sent."""
    import actspace_eval.providers.mock as MK
    go, root = job
    setup(data, "mock-anthropic", steps=3)
    FatalAfter.count, FatalAfter.limit, FatalAfter.status = 0, 12, status   # after the 3 probe calls
    monkeypatch.setattr(MK, "WireMock", FatalAfter)
    assert go() == code
    n = FatalAfter.count
    runs = run_paths(root)
    bad = [st_of(p)["status"] for p in runs if not (p / "SHA256SUMS").exists()]
    assert bad and set(bad) == {word}
    assert n <= 12 + 2 + 3 * (data.ecfg["llm"]["retry"]["max_retries"] + 1)
    FatalAfter.limit = 10 ** 9
    assert go(resume=True) == 0
    assert len(run_paths(root)) == 8 and all(st_of(p)["status"] == "complete" for p in run_paths(root))


def test_job_probe_failure_stops_before_any_run(data, job):
    go, root = job
    setup(data, "mock-openai", policy="fatal", fatal_status=403)
    assert go(model="mock-openai") == 4
    assert not run_paths(root, "mock-openai")
    setup(data, "mock-openai", policy="invalid")
    assert go(model="mock-openai", prefix="s2") == 1


def test_job_killed_then_resume(data, job):
    """Job killed: status.json not written, the last jsonl line half-written, some episodes unfinished. --resume continues, with each group exactly once."""
    go, root = job
    setup(data, "mock-anthropic", steps=3)
    assert go(settings="active_v2,passive_8") == 0
    d = root / "mock-anthropic" / "active_v2" / "s_r0"
    for p in d.rglob("*"):
        p.chmod(p.stat().st_mode | 0o200)
    d.chmod(d.stat().st_mode | 0o200)
    lines = (d / "trajectories.jsonl").read_text().splitlines(True)
    cut = max(i for i, x in enumerate(lines) if '"episode_start"' in x)    # keep only half of the last episode
    (d / "trajectories.jsonl").write_text("".join(lines[:cut + 1]) + lines[cut + 1][:40])
    with open(d / "calls.jsonl", "a") as f:
        f.write('{"type": "call", "cost_us')
    for n in ("status.json", "SHA256SUMS", "summary.json"):
        (d / n).unlink()
    spent = sum(x.get("cost_usd") or 0 for x in map(json.loads, (d / "calls.jsonl").read_text().splitlines()[:-1]))
    assert go(settings="active_v2,passive_8", resume=True) == 0
    eps = complete_episodes(d / "trajectories.jsonl")
    assert sorted(e[0]["group_id"] for e in eps) == sorted(data.ecfg["llm"]["smoke_groups"])
    assert replay(data, d / "trajectories.jsonl")["mismatched_episodes"] == 0
    assert st_of(d)["spent_usd"] > spent and (d / "SHA256SUMS").exists()   # money spent is kept; the interrupted episode is rerun


def test_repair_tail(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b'{"a": 1}\n{"b"')
    assert repair_tail(p) and p.read_bytes() == b'{"a": 1}\n'
    assert not repair_tail(p) and not repair_tail(tmp_path / "missing")


def test_pool_gives_slots_in_run_order():
    pool = CostPool({"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 0.0}, 10, slots=1)
    got, ev = [], threading.Event()

    def hold():
        with pool.slot(0)():
            ev.wait(10)

    def worker(order):
        with pool.slot(order)():
            got.append(order)

    h = threading.Thread(target=hold)
    h.start()
    while pool.in_use == 0:
        pass
    ws = [threading.Thread(target=worker, args=(k,)) for k in (3, 1, 2)]
    for w in ws:
        w.start()
    while len(pool._waiting) < 3:
        pass
    ev.set()
    for w in ws + [h]:
        w.join(timeout=10)
    assert got == [1, 2, 3] and pool.free == 1


def test_pool_waits_for_earlier_runs_to_arrive():
    """Threads start at different times: a later run must not jump ahead of an earlier run that has not arrived yet."""
    pool = CostPool({"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 0.0}, 10, slots=1)
    pool.expect([0, 1, 2])
    got = []

    def worker(order):
        with pool.slot(order)():
            got.append(order)

    late = threading.Thread(target=worker, args=(2,))
    late.start()
    while not pool._waiting:
        pass
    pool.arrived(1)                                         # run 1 has no episode to run
    assert not got
    first = threading.Thread(target=worker, args=(0,))
    first.start()
    first.join(timeout=10)
    late.join(timeout=10)
    assert got == [0, 2]


def test_job_cheap_settings_finish_first_when_cap_stops(data, job):
    """When the cost cap is reached early, the earlier settings have finished and the run stops in a later setting."""
    go, root = job
    m = data.ecfg["models"]["mock-anthropic"]
    data.ecfg["models"]["mock-anthropic"] = {**m, "max_output_tokens": 300}
    setup(data, "mock-anthropic", steps=4)
    assert go(cap=0.12) == 3
    done = {p.parent.name for p in run_paths(root) if (p / "SHA256SUMS").exists()}
    order = [s for s in E.SETTINGS]
    k = len(done)
    assert 1 <= k < 8 and done == set(order[:k])


def test_pool_abort_releases_waiters():
    pool = CostPool({"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 0.0}, 10, slots=0)
    out = []

    def w():
        try:
            with pool.slot(0)():
                out.append("ran")
        except ModelError as e:
            out.append(e)

    t = threading.Thread(target=w)
    t.start()
    e = http_error(403, "x")
    pool.abort(e)
    t.join(timeout=10)
    assert out == [e]


def test_run_cost_tracks_each_run_and_the_pool():
    pool = CostPool({"input": 0.0, "cached_input": 0.0, "cache_write": 0.0, "output": 1e6}, 10, spent=2.0, slots=2)
    a, b = pool.view(1.0), pool.view(1.0)
    a.reserve(0, 1)
    a.add({"output_tokens": 1})
    b.reserve(0, 1)
    b.add({"output_tokens": 2})
    assert (a.spent, b.spent, pool.spent) == (2.0, 3.0, 5.0)
    pool.cap = 5.5
    with pytest.raises(CostCapExceeded):                    # the cap is shared by the job: a spent only 2, but the job has spent 5, so 1 more does not fit
        a.reserve(0, 1)


def test_interval_single_run_is_scene_only():
    """1 run: the interval covers only scene resampling, regardless of the setting name; 3 different runs give a wider interval; one table may mix both."""
    scenes = {"arkitscenes": [f"s{i}" for i in range(12)]}
    b = RepBoot(scenes, n=500, scope=None)
    rng = np.random.default_rng(0)
    one = [(f"s{i}", float(v)) for i in range(12) for v in rng.random(5)]
    c1, c2 = b.mean_ci([one], "a"), b.mean_ci([one], "b")
    assert c1 == c2
    shifted = [[(s, v + d) for s, v in one] for d in (-0.3, 0.0, 0.3)]
    c3 = b.mean_ci(shifted, "c")
    assert c3["mean"] == pytest.approx(c1["mean"]) and c3["hi"] - c3["lo"] > c1["hi"] - c1["lo"]
    d = b.diff_ci(shifted, "c", [one], "a")
    assert d["mean"] == pytest.approx(0) and d["lo"] < 0 < d["hi"]


def test_leaderboard_mixes_1_and_3_runs(data, job):
    go, root = job
    setup(data, "mock-openai", steps=1)
    setup(data, "mock-anthropic", steps=2)
    assert go(model="mock-openai", settings="random_walk_40,active_v2", repeats=3) == 0
    assert go(model="mock-anthropic", settings="random_walk_40,active_v2") == 0
    md = "\n".join(E.leaderboard(root, ["mock-openai", "mock-anthropic"], "s"))
    assert "| mock-openai | json_object | 3 | 3 |" in md and "| mock-anthropic | schema | 1 | 1 |" in md
    assert "Models with a single run (mock-anthropic)" in md and md.count("\n## ") == 4
    assert "## Thinking tokens (by call type)" in md and "**mock-anthropic**" in md and "| Active answer step |" in md
    v = report_build(root / "mock-openai", E.complete_spec(root, "mock-openai", "s"))
    assert len(v["runs"]["active_v2"]) == 3


# ---------------------------------------------------------------- reply format: request structure, structured outputs and the diagnostics tool
def anthropic_bodies(data, tmp_path, model="mock-anthropic", gids=TWO[:1], **kw):
    bodies = []
    orig = WireMock.__call__
    WireMock.__call__ = lambda self, body: (bodies.append(copy.deepcopy(body)), orig(self, body))[1]
    try:
        st = run(data, model, tmp_path, gids=gids, **kw)
    finally:
        WireMock.__call__ = orig
    return bodies, st


def test_anthropic_request_has_top_level_system_and_no_tools(data, tmp_path):
    setup(data, "mock-anthropic", steps=3)
    bodies, _ = anthropic_bodies(data, tmp_path)
    assert bodies
    for b in bodies:
        assert set(b) == {"model", "max_tokens", "thinking", "output_config", "stream", "system", "messages"}
        assert set(b["output_config"]) == {"effort", "format"}
        assert b["system"][0]["type"] == "text" and b["system"][0]["text"].startswith("You are exploring")
        assert all(m["role"] in ("user", "assistant") for m in b["messages"])
        blocks = [x for m in b["messages"] if isinstance(m["content"], list) for x in m["content"]]
        assert {x["type"] for x in blocks} <= {"text", "image"}
        text = json.dumps(b["system"]) + json.dumps([x.get("text", "") for x in blocks])
        assert not re.search(r"<invoke|<parameter|tool_use|tool_choice|function_call", text)


def test_reply_schema_follows_official_limits(data):
    assert schema_error(REPLY_SCHEMA) is None
    assert REPLY_SCHEMA["properties"]["action"]["enum"] == [*MOVE_ACTIONS, "answer"] and REPLY_SCHEMA["required"] == ["action"]
    assert max(len(g["questions"]) for g in data.groups.values()) <= MAX_QUESTIONS
    assert schema_error({"type": "object", "properties": {"a": {"type": "object"}}, "additionalProperties": False})
    assert schema_error({"type": "object", "properties": {"a": {"type": "string", "maxLength": 3}}, "additionalProperties": False})
    assert schema_error({"type": "object", "additionalProperties": False, "properties": {f"p{i}": {"type": "string"} for i in range(25)}})


def test_all_claude_models_use_the_schema_and_none_stays_available(data):
    reg = R.load_registry()
    claude = {k: v for k, v in reg["models"].items() if v["provider"] == "anthropic"}
    assert len(claude) == 4 and all(v["json"]["mode"] == "schema" and v["key_budget_usd"] == 190 for v in claude.values())
    assert all(v["json"]["mode"] == "json_object" for v in reg["models"].values() if v["provider"] == "openai")
    e = R.entries(reg)["mock-anthropic"]
    assert R.to_mcfg("x", {**e, "json": {"mode": "none"}})["json_schema"] is False
    on = R.to_mcfg("x", e)
    assert on["json_schema"] is True and on["json_mode"] is False
    fmt = AnthropicClient(on, data.ecfg["llm"], transport=WireMock(on)).params()["output_config"]
    assert fmt == {"effort": "high", "format": {"type": "json_schema", "schema": REPLY_SCHEMA}}
    with pytest.raises(ValueError):
        R.to_mcfg("x", {**R.entries(reg)["mock-openai"], "json": {"mode": "schema"}})


def test_schema_mode_runs_every_step_with_the_same_format(data, tmp_path):
    setup(data, "mock-anthropic", steps=3)
    data.ecfg["models"]["mock-anthropic"]["json_schema"] = True
    bodies, st = anthropic_bodies(data, tmp_path)
    assert st["status"] == "complete" and len(bodies) == 4
    assert all(b["output_config"]["format"]["schema"] == REPLY_SCHEMA for b in bodies)
    assert all(c["request"]["params"]["output_config"]["format"]["type"] == "json_schema" for c in calls(tmp_path / "r"))
    assert [c["usage"]["cached_input_tokens"] > 0 for c in calls(tmp_path / "r")] == [False, True, True, True]


def test_thinking_display_is_left_out_when_not_set(data):
    m = FD.variant_cfg(data.ecfg["models"]["mock-anthropic"], "v2")
    assert AnthropicClient(m, data.ecfg["llm"], transport=WireMock(m)).params()["thinking"] == {"type": "adaptive"}


def test_variants_ignore_the_registry_mode(data):
    m = {**data.ecfg["models"]["mock-anthropic"], "json_schema": True}
    assert FD.variant_cfg(m, "v0")["json_schema"] is False and FD.variant_cfg(m, "v0")["reasoning"]["display"] == "summarized"
    assert FD.variant_cfg(m, "v1")["json_schema"] is True and FD.variant_cfg(m, "v2")["reasoning"]["display"] is None


@pytest.mark.parametrize("text,one,valid,markup,multi", [
    ('{"action": "forward"}', True, True, False, False),
    ('\n{"action": "answer", "answers": {"Q1": "A"}}\n', True, True, False, False),
    ('\n{"action": "turn_left"}\n\n{"action": "turn_left"}', False, True, False, True),
    ('<invoke name="x">\n</invoke>\n\n{"action": "turn_left"}', False, True, True, False),
    ('<invoke name="forward">\n</invoke>\n<invoke name="forward">\n</invoke>', False, False, True, True),
    ('<parameter name="action">forward</parameter>', False, False, True, False),
    ('I would like to look around.', False, False, False, False),
    ('[1, 2]', False, False, False, False),
    ('', False, False, False, False),
])
def test_format_classify(text, one, valid, markup, multi):
    assert FD.classify(text) == {"one_json": one, "valid": valid, "markup": markup, "multi_action": multi}


@pytest.fixture()
def diag_source(data, tmp_path):
    setup(data, "mock-anthropic", steps=7)
    run(data, "mock-anthropic", tmp_path, name="src")
    setup(data, "mock-anthropic", policy="tool_markup")
    return tmp_path / "src" / "calls.jsonl"


def test_mid_states_rebuild_the_logged_request(data, diag_source):
    states = FD.mid_states(data, diag_source)
    assert [s["id"] for s in states] == [f"{g}_t5" for g in TWO]
    logged = {json.loads(x)["tag"]["episode"]: json.loads(x) for x in open(diag_source) if '"t": 5,' in x and '"type": "call"' in x}
    m = data.ecfg["models"]["mock-anthropic"]
    c = AnthropicClient(m, data.ecfg["llm"], transport=WireMock(m))
    for s, g in zip(states, TWO):
        assert s["system"] == logged[g]["request"]["messages"][0]["content"]
        (_, msgs), jpegs = c.to_messages(s["system"], s["turns"])
        assert c._logged(s["system"], s["turns"], jpegs)[1:] == [{k: v for k, v in x.items()} if isinstance(x["content"], str) else
                                                                 {**x, "content": [{**b, "frame_id": None} if b["type"] == "image" else b for b in x["content"]]}
                                                                 for x in logged[g]["request"]["messages"][1:]]


def test_mid_states_refuse_a_changed_image(data, diag_source, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(diag_source.read_text().replace('"sha256": "', '"sha256": "0'))
    with pytest.raises(RuntimeError, match="sha256"):
        FD.mid_states(data, bad)


def test_format_diag_separates_variants(data, diag_source, tmp_path):
    res = FD.run_diag(data, "mock-anthropic", ["v0", "v1", "v2"], 3, 50.0, tmp_path / "d", diag_source)
    assert res["status"] == "complete" and len(res["calls"]) == 18
    assert (tmp_path / "d" / "calls_v1.jsonl").exists()
    by = {v: FD.stats([c for c in res["calls"] if c["variant"] == v]) for v in res["variants"]}
    assert by["v1"]["one_json"] == 6 and by["v1"]["markup"] == 0 and by["v1"]["truncated"] == 0
    assert by["v0"]["one_json"] == 0 and by["v0"]["markup"] == 6 and by["v0"]["truncated"] > 0 and by["v0"]["multi_action"] > 0
    assert by["v2"]["markup"] == 6
    assert FD.injected(res, "v1") == 100
    assert {c["state"] for c in res["calls"]} == set(res["states"]) and len(res["states"]) == 3
    assert "output_config" in res["params"]["v1"] and "format" in res["params"]["v1"]["output_config"]
    assert "display" not in res["params"]["v2"]["thinking"]
    text = FD.render(res, "20261104_120000", "abc")
    assert "| v1 structured output | total | 6 | 6 / 6 |" in text and "<invoke" in text and "Extra input tokens of v1 over v0 (same state, median): 100" in text


def test_format_diag_interleaves_and_stops_at_cap(data, diag_source, tmp_path):
    m = data.ecfg["models"]["mock-anthropic"]
    one = CostTracker(m["prices_usd_per_mtok"], 0).cost_of({"input_tokens": 4000, "cache_write_tokens": 4000, "output_tokens": m["max_output_tokens"]})
    res = FD.run_diag(data, "mock-anthropic", ["v0", "v1"], 5, one * 1.2, tmp_path / "d", diag_source)
    assert res["status"] == "stopped_cost_cap" and "no further calls" in res["error"] and res["spent_usd"] <= one * 1.2
    assert [c["variant"] for c in res["calls"]][:2] == ["v0", "v1"]
    order = FD.schedule(["v0", "v1"], 2, [{"id": "s"}, {"id": "a"}, {"id": "b"}])
    assert [(i, v, k, s["id"]) for i, v, k, s in order] == [(0, "v0", "start", "s"), (0, "v1", "start", "s"), (0, "v0", "mid", "a"), (0, "v1", "mid", "a"),
                                                            (1, "v0", "start", "s"), (1, "v1", "start", "s"), (1, "v0", "mid", "b"), (1, "v1", "mid", "b")]


def test_format_diag_rejected_variant_does_not_stop_the_others(data, diag_source, tmp_path, monkeypatch):
    monkeypatch.setitem(REPLY_SCHEMA["properties"]["answers"], "additionalProperties", True)
    res = FD.run_diag(data, "mock-anthropic", ["v0", "v1"], 2, 50.0, tmp_path / "d", diag_source)
    assert res["status"] == "complete" and "additionalProperties" in res["rejected"]["v1"]
    assert {c["variant"] for c in res["calls"]} == {"v0"} and len(res["calls"]) == 4
    assert "v1 rejected" in FD.render(res, "t", "c")


def test_format_diag_stops_on_other_fatal_errors(data, diag_source, tmp_path):
    setup(data, "mock-anthropic", policy="fatal", fatal_status=401)
    res = FD.run_diag(data, "mock-anthropic", ["v0", "v1"], 2, 50.0, tmp_path / "d", diag_source)
    assert res["status"] == "stopped_fatal_error" and res["calls"] == [] and "sk-abc123" not in json.dumps(res)


def test_format_diag_command(data, diag_source, tmp_path):
    env = {**os.environ, "PYTHONPATH": "src", "API_OUT_ROOT": str(tmp_path / "root")}
    env.pop("SLURM_JOB_ID", None)
    base = [sys.executable, "-m", "actspace_eval.providers.format_diag"]
    rep = tmp_path / "report.md"
    r = subprocess.run(base + ["mock-anthropic", "--n", "1", "--cap", "50", "--source", str(diag_source), "--report", str(rep)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = next((tmp_path / "root" / "_format_diag" / "mock-anthropic").iterdir())
    assert {"results.json", "summary.md", "calls_v0.jsonl", "calls_v1.jsonl", "calls_v2.jsonl"} <= {p.name for p in out.iterdir()}
    assert rep.read_text() == (out / "summary.md").read_text()
    assert subprocess.run(base + ["sonnet-5.5", "--n", "1", "--cap", "1"], cwd=ROOT, env=env, capture_output=True, text=True).returncode == 2
    assert subprocess.run(base + ["mock-openai", "--n", "1", "--cap", "1"], cwd=ROOT, env=env, capture_output=True, text=True).returncode == 2
    assert subprocess.run(base + ["mock-anthropic", "--variants", "v9", "--n", "1", "--cap", "1"], cwd=ROOT, env=env, capture_output=True, text=True).returncode == 2


def test_failed_probe_reruns_without_resume_and_keeps_its_cost(data, tmp_path, monkeypatch):
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    reg = R.load_registry()
    a = SimpleNamespace(model="mock-anthropic", prefix="smoke", cap=5.0, resume=False)
    setup(data, "mock-anthropic", policy="invalid")
    with pytest.raises(SystemExit) as e:
        E.step_probe(data, reg, a)
    assert e.value.code == 1
    spent = E.scope_spent(tmp_path, "mock-anthropic", "smoke")
    assert spent > 0
    plan = E.plan_runs("smoke", ["blind"], None)
    assert E.to_run(a, tmp_path, plan)          # only probe artifacts: not an existing run, no --resume needed
    setup(data, "mock-anthropic", policy="normal", steps=0)
    E.step_probe(data, reg, a)
    assert json.loads((tmp_path / "mock-anthropic" / "_probe" / "smoke" / "probe.json").read_text())["passed"]
    assert E.scope_spent(tmp_path, "mock-anthropic", "smoke") > spent


# ---------------------------------------------------------------- key file and Claude key rotation
def tree_text(root):
    return "".join(p.read_text(errors="replace") for p in Path(root).rglob("*") if p.is_file() and p.name != "fake.env" and p.name != "more.env")


def test_key_file_formats_and_refusals(tmp_path):
    f = tmp_path / "k.env"
    f.write_text("# comment ANTHROPIC_API_KEY_7=no\nexport ANTHROPIC_API_KEY_2='b b'\nANTHROPIC_API_KEY_1=a  # note\nANTHROPIC_API_KEY_3=\nANTHROPIC_API_KEY_10=\"c\"\nnot a line\n")
    f.chmod(0o600)
    assert K.anthropic_keys(K.read_key_file(f)) == {1: "a", 2: "b b", 10: "c"}
    f.chmod(0o640)
    with pytest.raises(K.KeyFileError, match="must be 600") as e:
        K.read_key_file(f)
    assert "ANTHROPIC" not in str(e.value)
    with pytest.raises(K.KeyFileError, match="does not exist"):
        K.read_key_file(tmp_path / "none")


def test_setup_keys_only_reports_numbers(tmp_path, capsys):
    e = R.entries(R.load_registry())["mock-anthropic"]
    note = K.setup_keys(tmp_path, "mock-anthropic", e, fake_key_file(tmp_path, (1, 3)))
    assert note["keys"] == [1, 3] and note["note"] == "anthropic keys: #1 #3 (2 non-empty)"
    assert K.active_pool().budget == 190
    empty = tmp_path / "e.env"
    empty.write_text("ANTHROPIC_API_KEY_1=\n")
    empty.chmod(0o600)
    with pytest.raises(K.KeyFileError, match="contains no non-empty"):
        K.setup_keys(tmp_path, "mock-anthropic", e, empty)


@pytest.mark.parametrize("status,body,kind", [
    (400, {"error": {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API."}}, "credit_balance"),
    (400, {"error": {"type": "invalid_request_error", "message": "You have reached your specified workspace spend limit."}}, "spend_limit"),
    (429, {"error": {"type": "rate_limit_error", "message": "You have reached your API usage limits.", "details": {"error_code": "enforced_spend_limit_reached"}}}, "spend_cap"),
    (402, {"error": {"type": "billing_error", "message": "billing"}}, "billing_error"),
    (400, {"error": {"type": "invalid_request_error", "message": "messages: field required"}}, None),
    (429, {"error": {"type": "rate_limit_error", "message": "This request would exceed your organization's rate limit of 50 requests per minute."}}, None),
    (401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}, None),
    (403, {"error": {"type": "permission_error", "message": "not allowed"}}, None),
])
def test_quota_error_rules(status, body, kind):
    q = K.quota_error(status, json.dumps(body))
    assert (q and q["kind"]) == kind
    assert K.quota_error(status, "not json") is None or status == 402


def test_ledger_is_shared_and_keeps_only_numbers_and_money(tmp_path):
    a, b = K.KeyPool({1: "x", 2: "y"}, tmp_path, 100), K.KeyPool({1: "x", 2: "y"}, tmp_path, 100)
    ts = [threading.Thread(target=lambda p=p: [p.charge(1, 0.01) for _ in range(50)]) for p in (a, a, b, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert K.read_ledger(tmp_path)["keys"]["1"]["spent_usd"] == pytest.approx(2.0)
    assert set(K.read_ledger(tmp_path)["keys"]["1"]) == {"spent_usd", "exhausted"}


def test_key_switches_before_the_budget_is_used_up(data, tmp_path):
    K.activate(K.KeyPool({1: fake_key(1), 2: fake_key(2)}, tmp_path, budget=0.002, model="mock-anthropic"))
    setup(data, "mock-anthropic", steps=6)
    st = run(data, "mock-anthropic", tmp_path, gids=TWO)
    cs = [c for c in calls(tmp_path / "r") if c["type"] == "call"]
    used = [c["key_no"] for c in sorted(cs, key=lambda c: c["spent_usd"])]
    assert st["status"] == K.QUOTA_STATUS or st["status"] == "complete"
    assert used[0] == 1 and 2 in used and used == sorted(used)
    led = K.read_ledger(tmp_path)["keys"]
    assert led["1"]["exhausted"] == "budget"
    assert sum(r["spent_usd"] for r in led.values()) == pytest.approx(sum(c["cost_usd"] for c in cs), abs=1e-5)


@pytest.mark.parametrize("kind", ["credit", "spend_limit", "spend_cap", "billing"])
def test_quota_error_switches_key_and_resends_the_same_call(data, tmp_path, kind, capsys):
    K.activate(K.KeyPool({1: fake_key(1), 2: fake_key(2)}, tmp_path, model="mock-anthropic"))
    setup(data, "mock-anthropic", steps=5, quota={"1": {"after": 3, "kind": kind}})
    bodies, st = anthropic_bodies(data, tmp_path)
    assert st["status"] == "complete" and st["completed"] == 1
    cs = calls(tmp_path / "r")
    assert [c["key_no"] for c in cs] == [1, 1, 1, 2, 2, 2]
    assert bodies[3] == bodies[4]                # the same call resent unchanged
    assert not [c for c in cs if c["type"] != "call"]
    led = K.read_ledger(tmp_path)["keys"]
    assert led["1"]["exhausted"] and led["2"]["exhausted"] is None
    assert led["1"]["spent_usd"] + led["2"]["spent_usd"] == pytest.approx(sum(c["cost_usd"] for c in cs), abs=1e-5)
    ev = [json.loads(x) for x in open(tmp_path / "_keys" / "anthropic_events.jsonl")]
    ex = next(e for e in ev if e["event"] == "exhausted")
    assert ex["key"] == 1 and ex["type"] and ex["message"] and "key #2" in capsys.readouterr().out
    text = tree_text(tmp_path)
    assert "FAKE1" not in text and "FAKE2" not in text


def test_unrecognised_errors_with_keys_are_still_fatal(data, tmp_path):
    K.activate(K.KeyPool({1: fake_key(1), 2: fake_key(2)}, tmp_path, model="mock-anthropic"))
    setup(data, "mock-anthropic", policy="fatal", fatal_status=401)
    st = run(data, "mock-anthropic", tmp_path, gids=TWO[:1])
    assert st["status"] == "stopped_fatal_error" and "exhausted" not in json.dumps(K.read_ledger(tmp_path))
    assert "FAKE" not in tree_text(tmp_path)


def test_all_keys_used_up_exit_6_then_resume_with_a_new_key(data, job, tmp_path):
    go, root = job
    setup(data, "mock-anthropic", steps=2, quota={"1": {"after": 4, "kind": "credit"}, "2": {"after": 6, "kind": "spend_cap"}})
    f = fake_key_file(tmp_path)
    settings = "blind,active_v2"
    assert go(settings=settings, key_file=str(f)) == 6
    active = root / "mock-anthropic" / "active_v2" / "s_r0"
    assert json.loads((active / "status.json").read_text())["status"] == K.QUOTA_STATUS
    done = len(complete_episodes(active / "trajectories.jsonl"))
    assert done < 3
    led = K.read_ledger(root)["keys"]
    assert led["1"]["exhausted"] == "credit_balance" and led["2"]["exhausted"] == "spend_cap"
    spent = E.scope_spent(root, "mock-anthropic", "s")
    more = fake_key_file(tmp_path, (1, 2, 3), name="more.env")
    assert go(settings=settings, resume=True, key_file=str(more)) == 0
    assert len(complete_episodes(active / "trajectories.jsonl")) == 3
    assert json.loads((active / "status.json").read_text())["status"] == "complete"
    led = K.read_ledger(root)["keys"]
    assert led["3"]["spent_usd"] > 0 and {c.get("key_no") for c in calls(active)} >= {3}
    assert sum(r["spent_usd"] for r in led.values()) == pytest.approx(E.scope_spent(root, "mock-anthropic", "s"), abs=1e-4) and E.scope_spent(root, "mock-anthropic", "s") > spent
    assert "FAKE" not in tree_text(root)


def test_probe_with_no_usable_key_exits_6(data, job, tmp_path):
    go, root = job
    setup(data, "mock-anthropic", steps=1, quota={"1": {"after": 0, "kind": "credit"}})
    assert go(settings="blind", key_file=str(fake_key_file(tmp_path, (1,)))) == 6


def test_reset_key_clears_a_replaced_number(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    p = K.KeyPool({1: "x"}, tmp_path, 1.0)
    p.charge(1, 2.0)
    with pytest.raises(K.KeysExhausted):
        K.KeyPool({1: "x"}, tmp_path, 1.0).current()
    E.cmd_keys(SimpleNamespace(reset_key=[1]))
    assert "#1: total reset to zero" in capsys.readouterr().out
    assert K.KeyPool({1: "x"}, tmp_path, 1.0).current()[0] == 1
    assert "reset" in [json.loads(x)["event"] for x in open(tmp_path / "_keys" / "anthropic_events.jsonl")]


def test_keyscan_reports_only_yes_or_no(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    d = tmp_path / "mock-anthropic" / "blind" / "s_r0"
    d.mkdir(parents=True)
    (d / "calls.jsonl").write_text("the desk-mounted-lamp and a task-oriented-agent\n")
    a = SimpleNamespace(model="mock-anthropic", prefix="s", key_file=None)
    with pytest.raises(SystemExit) as e:
        E.cmd_keyscan(a)
    assert e.value.code == 0 and "suspected keys: none" in capsys.readouterr().out
    secret = "sk-ant-api03-secretsecretsecret"
    (d / "calls.jsonl").write_text(f"oops {secret}\n")
    with pytest.raises(SystemExit) as e:
        E.cmd_keyscan(a)
    out = capsys.readouterr().out
    assert e.value.code == 1 and "suspected keys: yes" in out and "secret" not in out
    f = fake_key_file(tmp_path)
    (d / "calls.jsonl").write_text("plain words\n")
    (d / "x.log").write_text(fake_key(2)[3:] + "\n")     # a fragment without the sk- prefix does not count; only the full verbatim key does
    assert K.scan_for_keys([d / "x.log"], [fake_key(2)]) == 0
    (d / "x.log").write_text("a" + fake_key(2) + "\n")
    assert K.scan_for_keys([d / "x.log"], K.read_key_file(f).values()) == 1


def test_thinking_tokens_by_call_type_in_reports(data, job, monkeypatch):
    go, root = job
    setup(data, "mock-anthropic", steps=2)
    assert go(settings="blind,active_v2") == 0
    v = E.thinking_tokens(root, "mock-anthropic", "s")
    assert len(v["Move step"]) == 6 and len(v["Active answer step"]) == 3 and len(v["One-shot answer"]) == 3
    assert E.thinking_row([0, 0, 10, 20]) == {"n": 4, "median": 5.0, "q1": 0.0, "q3": 12.5, "zero": 0.5}
    md = (root / "mock-anthropic" / "_reports" / "s.md").read_text()
    assert "## Thinking tokens (by call type)" in md and "| Active answer step | 3 | 30 | 30–30 | 0.0% |" in md
    assert "## Run conditions" in md and "| active_v2 | s_r0 | schema | 32768 |" in md


def test_smoke_report_has_thinking_and_conditions(data, tmp_path, monkeypatch):
    smoke_tree(data, "mock-anthropic", tmp_path, monkeypatch)
    text = "\n".join(SR.build(data, R.load_registry(), "mock-anthropic", "smoke", {"all": TWO}))
    assert "## 8. Thinking tokens (by call type)" in text and "## Run conditions" in text


def test_script_passes_key_file_and_reset_to_job_not_values(tmp_path, synth):   # synth: the script runs freeze verify
    env, log = shims(tmp_path)
    f = fake_key_file(tmp_path)
    r = sh("mock-anthropic", "tiny", "--cap", "1", "--key-file", str(f), "--reset-key", "2", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "anthropic keys: #1 #2 (2 non-empty)" in r.stdout and "#2: total reset to zero" in r.stdout
    j = log.read_text()
    assert f"--key-file {f}" in j and "FAKE" not in j and "FAKE" not in r.stdout + r.stderr


def test_format_diag_uses_the_key_pool(data, diag_source, tmp_path):
    K.activate(K.KeyPool({1: fake_key(1)}, tmp_path, model="mock-anthropic"))
    setup(data, "mock-anthropic", quota={"1": {"after": 2, "kind": "credit"}})
    res = FD.run_diag(data, "mock-anthropic", ["v1"], 3, 50.0, tmp_path / "d", diag_source)
    assert res["status"] == K.QUOTA_STATUS and len(res["calls"]) >= 1
    assert K.read_ledger(tmp_path)["keys"]["1"]["exhausted"] == "credit_balance"
