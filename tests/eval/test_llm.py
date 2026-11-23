"""Unit tests for the model runner: only the mock and injected fake transports, no real API calls."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from actspace_eval import prompts as P
from actspace_eval.cost import CostTracker, estimate
from actspace_eval.data import FrozenData
from actspace_eval.env import ActSpaceEnv
from actspace_eval.llm_agent import INVALID, parse_reply
from actspace_eval.llm_run import env_factory, run_llm
from actspace_eval.models import ModelError, OpenAIClient, RealCallsDisabled, make_client
from actspace_eval.models_vllm import VLLMClient
from actspace_eval.trajectory import complete_episodes, replay

ROOT = Path(__file__).resolve().parents[2]
# merged_v1.0: two datasets; arkit_g_000015 is the ARKit group with the longest route (36 steps) and has a masked question, scannetpp_g_000009 is a ScanNet++ masked group (direction_inverse, T5)
GIDS = ["arkit_g_000000", "scannetpp_g_000009", "arkit_g_000015"]


@pytest.fixture()
def data():
    return FrozenData(verify=False)


def mock_cfg(data, **kw):
    d = copy.deepcopy(data.ecfg)
    d["models"]["mock"].update(kw)
    data.ecfg = d
    return data


def run(data, tmp_path, mode, gids=GIDS, budget=40, name="r", **kw):
    return run_llm(data, "mock", mode, budget, gids, tmp_path / name, render=False, print_estimate=False, **kw)


# ---------------------------------------------------------------- prompt

def test_prompt_templates_in_protocol_doc():
    """docs/evaluation_protocol.md reproduces every template of prompts.py verbatim (the published text is the text in use)."""
    doc = (ROOT / "docs" / "evaluation_protocol.md").read_text()
    for name, text in P.ALL_TEMPLATES.items():
        assert text in doc, f"template {name} is not reproduced verbatim in docs/evaluation_protocol.md"


def test_prompts_do_not_leak():
    sys_texts = [P.system_prompt("active", budget=40, history={"mode": "last_k", "k": 8, "pin_start": True}),
                 P.system_prompt("passive", n_photos=41), P.system_prompt("blind")]
    for s in sys_texts:
        low = s.lower().replace("notes are not scored", "")   # the prompt must state that notes are not scored
        for w in ("oracle", "score", "mra", "evidence", "mask", "unanswerable", "visible", "frame", "arkit", "view graph", "threshold"):
            assert w not in low, (w, s[:80])
    assert P.system_prompt("passive", n_photos=11) == P.system_prompt("oracle", n_photos=11)  # Passive and Oracle have the same text
    blind = P.system_prompt("blind")
    assert "best guess" in blind and '"cannot_determine" is not accepted' in blind and '"Q3": "cannot_determine"' not in blind
    for s in sys_texts:
        assert '"notes"' in s and "300 characters" in s
    assert "3 invalid replies in a row" in sys_texts[0] and "START photograph is always shown" in sys_texts[0]
    assert "less than 90 degrees" in P.CONVENTIONS


def test_parse_reply():
    assert parse_reply('{"action": "forward"}') == ("forward", None)
    assert parse_reply('Sure.\n```json\n{"action": "Turn_Left"}\n```') == ("turn_left", None)
    assert parse_reply('{"action": "answer", "answers": {"Q1": "B"}}') == ("answer", {"Q1": "B"})
    assert parse_reply('{"Q1": "B", "Q2": 2.0}') == ("answer", {"Q1": "B", "Q2": 2.0})
    assert parse_reply("I will go forward") == (INVALID, None)
    assert parse_reply('{"action": "jump"}') == ("jump", None)
    assert parse_reply("") == (INVALID, None)


def test_history_last_k_and_full():
    img = np.zeros((4, 4, 3), np.uint8)
    turns = [{"role": "user", "parts": [("text", "start"), ("image", "photo_000000", img)]}]
    for t in range(1, 12):
        turns += [{"role": "assistant", "text": '{"action": "forward"}'},
                  {"role": "user", "parts": [("text", f"Step {t}"), ("image", f"photo_{t:06d}", img), ("text", "status")]}]
    shown = P.apply_history(turns, {"mode": "last_k", "k": 8})
    imgs = [p[1] for t in shown for p in t.get("parts", []) if p[0] == "image"]
    assert imgs == [f"photo_{t:06d}" for t in range(4, 12)]
    omitted = [p[1] for t in shown for p in t.get("parts", []) if p[0] == "text" and "no longer shown" in p[1]]
    assert omitted == [P.IMAGE_OMITTED.format(photo_id=f"photo_{t:06d}") for t in range(0, 4)]
    texts = lambda ts: [p[1] for t in ts for p in t.get("parts", []) if p[0] == "text" and "no longer" not in p[1]]
    assert texts(shown) == texts(turns)                                  # the text record is kept in full
    assert P.apply_history(turns, {"mode": "full"}) == turns


# ---------------------------------------------------------------- mock: four settings

@pytest.mark.parametrize("mode", ["blind", "passive", "oracle", "active"])
def test_mock_valid_four_modes(data, tmp_path, mode):
    st = run(mock_cfg(data, policy="valid"), tmp_path, mode)
    assert st["status"] == "complete" and st["completed"] == 3 and st["spent_usd"] > 0
    eps = complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    assert [e[0]["group_id"] for e in eps] == GIDS
    for ep in eps:
        assert ep[-1]["parse_ok"] and not ep[-1]["parse_errors"]
        if mode == "active":
            assert ep[-1]["steps_used"] == 5 and ep[-1]["forced"] is None
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0
    calls = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    assert all(c["type"] == "call" for c in calls)
    assert len(calls) == (3 * 6 if mode == "active" else 3)
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["episodes"] == 3 and s["mode"] == mode


def test_mock_invalid_replies(data, tmp_path):
    st = run(mock_cfg(data, policy="invalid"), tmp_path, "active", gids=GIDS[:1], budget=10)
    ep = complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0]
    steps = [r for r in ep if r["type"] == "step"]
    # an invalid reply costs 1 step; after 3 consecutive invalid replies the episode ends and is scored as unanswered
    assert len(steps) == 3 and all(r["action"] == INVALID and not r["moved"] for r in steps)
    assert ep[-1]["forced"] == "invalid_replies" and set(ep[-1]["scores"].values()) == {0.0}
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0
    st = run(mock_cfg(data, policy="invalid"), tmp_path, "passive", gids=GIDS[:1], budget=10, name="p")
    ep = complete_episodes(tmp_path / "p" / "trajectories.jsonl")[0]
    assert not ep[-1]["parse_ok"] and ep[-1]["raw_answer"].startswith("I would like")   # raw reply kept


def test_mock_mixed_counts_invalid_steps(data, tmp_path):
    run(mock_cfg(data, policy="mixed"), tmp_path, "active", gids=GIDS[:1], budget=40)
    ep = complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0]
    acts = [r["action"] for r in ep if r["type"] == "step"]
    assert INVALID in acts and any(a != INVALID for a in acts) and ep[-1]["parse_ok"]


# ---------------------------------------------------------------- retries

def test_retry_with_backoff_logged(data, tmp_path):
    waits = []
    st = run(mock_cfg(data, policy="flaky", fail_times=2), tmp_path, "oracle", gids=GIDS[:2], client_kwargs={"sleep": waits.append})
    assert st["status"] == "complete"
    recs = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    errs = [r for r in recs if r["type"] == "error"]
    assert len(errs) == 4 and all(r["retryable"] and r["status"] == 429 for r in errs)   # each request fails twice first
    r = data.ecfg["llm"]["retry"]
    for w, attempt in zip(waits, [0, 1, 0, 1]):                                         # exponential backoff: 2 s, 4 s (with jitter)
        base = r["base_delay_s"] * 2 ** attempt
        assert base <= w <= base * (1 + r["jitter"])
    assert [c["retries"] for c in recs if c["type"] == "call"] == [2, 2]


def test_retry_exhausted_stops_and_resumes(data, tmp_path):
    d = mock_cfg(data, policy="flaky", fail_times=99)
    d.ecfg["llm"]["retry"]["max_retries"] = 3
    st = run(d, tmp_path, "blind", client_kwargs={"sleep": lambda s: None})
    assert st["status"] == "stopped_model_error" and st["completed"] == 0
    recs = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    assert len(recs) == 4 and recs[-1].get("gave_up")                                   # first attempt + 3 retries, the last gives up
    st = run(mock_cfg(data, policy="valid"), tmp_path, "blind", resume=True)
    assert st["status"] == "complete" and st["completed"] == 3


def test_fatal_error_not_retried(data, tmp_path):
    waits = []
    st = run(mock_cfg(data, policy="fatal"), tmp_path, "blind", client_kwargs={"sleep": waits.append})
    recs = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    assert st["status"] == "stopped_model_error" and len(recs) == 1 and not recs[0]["retryable"] and not waits


# ---------------------------------------------------------------- cost cap and resume

def test_cost_cap_stops_and_resume_completes(data, tmp_path):
    d = mock_cfg(data, policy="valid")
    full = run(d, tmp_path, "active", name="full")["spent_usd"]
    cap = full * 0.5
    st = run(d, tmp_path, "active", cost_cap=cap)
    assert st["status"] == "stopped_cost_cap" and 0 < st["completed"] < 3
    assert st["spent_usd"] <= cap                                                      # pre-call check against the estimate never exceeds the cap
    spent_log = CostTracker.spent_in_log(tmp_path / "r" / "calls.jsonl")
    assert spent_log == pytest.approx(st["spent_usd"], abs=1e-6)
    done1 = [e[0]["group_id"] for e in complete_episodes(tmp_path / "r" / "trajectories.jsonl")]
    # on resume the cap still applies to the cumulative cost: stop immediately unless the cap is raised
    st2 = run(d, tmp_path, "active", cost_cap=cap, resume=True)
    assert st2["status"] == "stopped_cost_cap" and st2["completed"] == len(done1)
    st3 = run(d, tmp_path, "active", cost_cap=5.0, resume=True)
    assert st3["status"] == "complete" and st3["completed"] == 3
    eps = complete_episodes(tmp_path / "r" / "trajectories.jsonl")
    assert sorted(e[0]["group_id"] for e in eps) == sorted(GIDS)                       # each group exactly once
    assert st3["spent_usd"] == pytest.approx(CostTracker.spent_in_log(tmp_path / "r" / "calls.jsonl"), abs=1e-6)
    assert replay(data, tmp_path / "r" / "trajectories.jsonl")["mismatched_episodes"] == 0


def test_resume_drops_incomplete_episode_and_refuses_overwrite(data, tmp_path):
    d = mock_cfg(data, policy="valid")
    run(d, tmp_path, "oracle", gids=GIDS[:2])
    p = tmp_path / "r" / "trajectories.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")                                         # simulate an interruption: the last episode is incomplete
    with pytest.raises(FileExistsError):
        run(d, tmp_path, "oracle", gids=GIDS[:2])
    st = run(d, tmp_path, "oracle", gids=GIDS[:2], resume=True)
    assert st["completed"] == 2 and len(complete_episodes(p)) == 2 and len(p.read_text().splitlines()) == len(lines)


def test_cost_tracker_prices():
    t = CostTracker({"input": 2.5, "cached_input": 1.25, "output": 10.0}, cap_usd=1.0)
    c = t.add({"input_tokens": 1_000_000, "cached_input_tokens": 400_000, "output_tokens": 10_000})
    assert c == pytest.approx(0.6 * 2.5 + 0.4 * 1.25 + 0.01 * 10)


# ---------------------------------------------------------------- OpenAI adapter (fake transport)

def openai_cfg(data, **kw):
    m = copy.deepcopy(data.ecfg["models"]["openai_example"])
    m.update(model="test-model", **kw)
    return m


def test_openai_message_format_and_usage(data, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    sent = []

    def transport(body):
        sent.append(body)
        return {"id": "x", "choices": [{"message": {"content": '{"action": "forward"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 1000},
                          "completion_tokens_details": {"reasoning_tokens": 0}}}
    cost = CostTracker(data.ecfg["models"]["openai_example"]["prices_usd_per_mtok"], 10.0)
    c = OpenAIClient(openai_cfg(data), data.ecfg["llm"], cost, tmp_path / "calls.jsonl", transport=transport)
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    comp = c.complete("SYS", [{"role": "user", "parts": [("text", "hello"), ("image", "photo_000001", img)]}], tag={"t": 0})
    body = sent[0]
    assert body["model"] == "test-model" and body["temperature"] == 0.0 and "reasoning_effort" not in body
    assert body["max_completion_tokens"] == 1024 and body["response_format"] == {"type": "json_object"}
    part = body["messages"][1]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,") and part["image_url"]["detail"] == "high"
    assert comp.text == '{"action": "forward"}' and comp.usage["cached_input_tokens"] == 1000
    assert comp.cost_usd == pytest.approx((200 * 2.5 + 1000 * 1.25 + 8 * 10) / 1e6)
    log = (tmp_path / "calls.jsonl").read_text()
    assert "base64" not in log and "sk-test" not in log and "<image photo_000001 640x480 jpeg" in log
    # reasoning mode: send reasoning_effort, not temperature
    c2 = OpenAIClient(openai_cfg(data, reasoning={"enabled": True, "effort": "high"}), data.ecfg["llm"], transport=transport)
    assert "temperature" not in c2.params() and c2.params()["reasoning_effort"] == "high"


def test_openai_real_calls_guarded(data, monkeypatch):
    llm = copy.deepcopy(data.ecfg["llm"])
    llm.update(allow_real_calls=False, prompts_approved=False)   # order of the switch checks does not depend on current config values
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RealCallsDisabled, match="allow_real_calls"):
        OpenAIClient(openai_cfg(data), llm)
    llm["allow_real_calls"] = True
    with pytest.raises(RealCallsDisabled, match="prompts_approved"):
        OpenAIClient(openai_cfg(data), llm)
    llm["prompts_approved"] = True
    with pytest.raises(RealCallsDisabled, match="prices_confirmed"):
        OpenAIClient(openai_cfg(data), llm)
    with pytest.raises(RealCallsDisabled, match="OPENAI_API_KEY"):
        OpenAIClient(openai_cfg(data, prices_confirmed=True), llm)


def test_openai_http_errors_classified(data, monkeypatch):
    """No network requests: replace requests.post and check status code → retry or not, and that the key appears only in the request headers."""
    import requests
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    llm = dict(data.ecfg["llm"], allow_real_calls=True, prompts_approved=True)
    c = OpenAIClient(openai_cfg(data, prices_confirmed=True), llm)
    seen = []

    class R:
        def __init__(self, code):
            self.status_code, self.text = code, "err"

    for code, retryable in ((429, True), (500, True), (503, True), (400, False), (401, False)):
        monkeypatch.setattr(requests, "post", lambda url, json, timeout, headers, code=code: seen.append(headers) or R(code))
        with pytest.raises(ModelError) as e:
            c._post({"model": "x"})
        assert e.value.retryable == retryable and e.value.status == code
    assert seen[0]["Authorization"] == "Bearer sk-test-not-a-real-key-000000000000"

    def boom(*a, **k):
        raise requests.exceptions.ReadTimeout("timeout")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(ModelError) as e:     # exceeding the per-call limit counts as truncation, no retry
        c._post({"model": "x"})
    assert not e.value.retryable and "timeout_truncated" in str(e.value)
    c.mcfg = {**c.mcfg, "call_timeout_s": None}   # without a limit, read timeouts are retried as before
    with pytest.raises(ModelError) as e:
        c._post({"model": "x"})
    assert e.value.retryable


def test_vllm_client_needs_base_url(data):
    """Open models: no key needed, but the vllm serve address must be given."""
    with pytest.raises(ValueError):
        make_client(data.ecfg["models"]["Qwen3.5-9B"], data.ecfg["llm"])
    assert isinstance(make_client(data.ecfg["models"]["Qwen3.5-9B"], data.ecfg["llm"], base_url="http://127.0.0.1:1"), VLLMClient)


# ---------------------------------------------------------------- estimation

def test_estimate_counts(data):
    mcfg = data.ecfg["models"]["openai_example"]
    fac = env_factory(data)
    rows = estimate(fac, GIDS, [("blind", 40, 0), ("passive", 40, 0), ("oracle", 40, 0), ("active", 40, 40)], mcfg,
                    {"mode": "last_k", "k": 8})
    blind, passive, oracle, active = rows
    assert blind["images"] == 0 and blind["calls"] == 3
    shown = [len(ActSpaceEnv(data, mode="passive", budget=40, render=False).reset(g, seed=0)["photos"]) for g in GIDS]
    assert passive["images"] == sum(shown)
    assert active["calls"] == 3 * 41 and active["images"] == 3 * (sum(min(t + 1, 8) for t in range(41)))
    full = estimate(fac, GIDS, [("active", 40, 40)], mcfg, {"mode": "full"})[0]
    assert full["images"] == 3 * sum(t + 1 for t in range(41)) and full["cost_usd"] > active["cost_usd"]


# ---------------------------------------------------------------- changes made before the pilot run

def test_history_pins_start_image():
    turns = [{"role": "user", "parts": [("text", "Q"), ("image", "p0", None), ("text", "s")]}]
    for i in range(1, 12):
        turns += [{"role": "assistant", "text": "{}"}, {"role": "user", "parts": [("text", f"step {i}"), ("image", f"p{i}", None)]}]
    shown = P.apply_history(turns, {"mode": "last_k", "k": 8, "pin_start": True})
    imgs = [p[1] for t in shown for p in t.get("parts", []) if p[0] == "image"]
    assert imgs == ["p0"] + [f"p{i}" for i in range(4, 12)]                  # START pinned as the first image + latest 8
    assert shown[0] == turns[0]                                               # turn 0 unchanged (cache prefix)
    plain = P.apply_history(turns, {"mode": "last_k", "k": 8})
    assert [p[1] for t in plain for p in t.get("parts", []) if p[0] == "image"] == [f"p{i}" for i in range(4, 12)]


def test_static_photo_order(data):
    for mode in ("passive", "oracle"):
        env = ActSpaceEnv(data, mode=mode, budget=40, render=False)
        ph = env.reset("arkit_g_000000", seed=0)["photos"]
        assert ph[0]["start"] and not any(p["start"] for p in ph[1:])
    from actspace_eval.agents import oracle_targets
    env = ActSpaceEnv(data, mode="oracle", budget=40, render=False)
    env.reset("arkit_g_000015", seed=0)
    rest = [n for n in oracle_targets(data.groups["arkit_g_000015"]) if n != env.shown[0]]
    assert sorted(env.shown[1:]) == sorted(rest) and env.shown[1:] != rest     # Oracle shuffles the remaining photos


def test_invalid_streak_resets_on_valid_move(data):
    env = ActSpaceEnv(data, budget=40, render=False)
    env.reset("arkit_g_000000", seed=0)
    for a in ("bad", "bad", "forward", "bad", "bad"):
        obs, fb, done = env.step(a)
        assert not done
    _, _, done = env.step("bad")
    assert done and env.records[-1]["forced"] == "invalid_replies" and env.records[-1]["steps_used"] == 6


def test_notes_truncated_and_recorded(data, tmp_path):
    from actspace_eval.llm_agent import history_text, reply_notes
    long = "x" * 500
    assert reply_notes({"action": "forward", "notes": long}) == "x" * 300
    assert json.loads(history_text(json.dumps({"action": "forward", "notes": long})))["notes"] == "x" * 300
    assert history_text('{"action": "forward", "notes": "ok"}') == '{"action": "forward", "notes": "ok"}'
    env = ActSpaceEnv(data, budget=40, render=False)
    env.reset("arkit_g_000000", seed=0)
    env.step({"action": "forward", "notes": long})
    env.step({"action": "answer", "content": {"Q1": 1.0}, "notes": "done"})
    assert env.records[1]["notes"] == "x" * 300 and env.records[-1]["notes"] == "done"
    p = tmp_path / "t.jsonl"
    from actspace_eval.trajectory import TrajectoryWriter
    w = TrajectoryWriter(p)
    w.write_episode(env.records)
    w.close()
    assert replay(data, p)["mismatched_episodes"] == 0


def test_blind_excludes_unanswerable(data, tmp_path):
    run(mock_cfg(data, policy="valid"), tmp_path, "blind")
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    n_un = sum(q["evidence_label"] == "unanswerable" for g in GIDS for q in data.groups[g]["questions"])
    n_all = sum(len(data.groups[g]["questions"]) for g in GIDS)
    assert n_un > 0 and s["questions"] == n_all - n_un and s["excluded_unanswerable"] == n_un
    assert all(x["answerable"] for x in s["per_question"])


def test_mock_prefix_cache_and_usage_totals(data, tmp_path):
    st = run(mock_cfg(data, policy="valid", mock_steps=12), tmp_path, "active")
    u = st["usage"]
    assert u["calls"] == 3 * 13 and 0 < u["cached_input_tokens"] < u["input_tokens"]
    calls = [json.loads(l) for l in (tmp_path / "r" / "calls.jsonl").read_text().splitlines()]
    assert all(c["usage"]["cached_input_tokens"] == 0 for c in calls if c["tag"]["t"] == 0)   # the first call of each episode never hits the cache
    ep = complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0]
    assert ep[0]["model_name"] == "mock-1" and "reasoning_effort" in ep[0]


def test_smoke_gating_and_cache_key(data, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = copy.deepcopy(data.ecfg)
    d["llm"].update(allow_real_calls=False, prompts_approved=False)
    with pytest.raises(ValueError, match="smoke_groups"):
        run_llm(data, "gpt-6.1-sol", "blind", 40, ["arkit_g_000003"], tmp_path / "s", render=False, print_estimate=False, smoke=True)
    with pytest.raises(RealCallsDisabled, match="OPENAI_API_KEY"):   # smoke tests bypass the first two switches, but the key is still required
        OpenAIClient(d["models"]["gpt-6.1-sol"], d["llm"], smoke=True)
    with pytest.raises(RealCallsDisabled, match="allow_real_calls"):
        OpenAIClient(d["models"]["gpt-6.1-sol"], d["llm"])
    c = OpenAIClient(d["models"]["gpt-6.1-sol"], d["llm"], transport=lambda b: None)
    c.cache_key = "actspace-x-arkit_g_000000"
    p = c.params()
    assert p["prompt_cache_key"] == "actspace-x-arkit_g_000000" and p["reasoning_effort"] == "high" and "temperature" not in p
    assert p["model"] == "gpt-6.1-sol"


def test_estimate_with_pin_and_cache(data):
    mcfg = data.ecfg["models"]["gpt-6.1-sol"]
    row = estimate(env_factory(data), GIDS, [("active", 40, 40)], mcfg, {"mode": "last_k", "k": 8, "pin_start": True})[0]
    assert row["images"] == 3 * sum(min(t + 1, 9) for t in range(41))
    assert 0 < row["cached_input_tokens"] < row["input_tokens"] and row["cost_usd"] < row["cost_no_cache_usd"]
