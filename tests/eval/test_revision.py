"""Setting revision: active_v2, random_walk, oracle_route, passive_k, oracle_min, derived quantities. Mock only."""
import copy
import json

import numpy as np
import pytest

from actspace_eval import prompts as P
from actspace_eval.agents import oracle_targets, plan_route
from actspace_eval.cost import estimate
from actspace_eval.data import MOVE_ACTIONS, FreezeError, FrozenData
from actspace_eval.env import ActSpaceEnv
from actspace_eval.llm_run import env_factory, estimate_plan, run_llm, with_effort
from actspace_eval.meta import group_meta
from actspace_eval.odometry import relative_motion
from actspace_eval.report_revision import build, render
from actspace_eval.trajectory import complete_episodes, replay

GIDS = ["arkit_g_000000", "scannetpp_g_000009", "arkit_g_000015"]
LEAK = ("oracle", "score", "mra", "evidence", "mask", "unanswerable", "visible", "frame", "arkit", "view graph", "threshold")


@pytest.fixture()
def data():
    return FrozenData(verify=False)


def run(data, tmp_path, setting, name, budget=40, **kw):
    return run_llm(data, "mock", setting, budget, GIDS, tmp_path / name, render=False, print_estimate=False, **kw)


# ---------------------------------------------------------------- prompts

def test_active_v2_prompt():
    h = {"mode": "last_k", "k": 8, "pin_start": True}
    v1 = P.system_prompt("active", budget=40, history=h)
    v2 = P.system_prompt("active", budget=40, history=h, version="v2", scale={"move_m": 0.3, "turn_deg": 45})
    assert v1 == P.system_prompt("active", budget=40, history=h, version="v1")          # v1 unchanged
    assert "Pose relative" not in v1 and "pose relative to START" not in v1
    assert "A move covers about 0.3 m." in v2 and "by about 45 degrees" in v2 and "Turns can fail" in v2
    assert "pose relative to START" in v2 and "positive = turned to the left" in v2 and "negative = to the right" in v2
    assert v2.startswith("You are exploring")                                            # the mock recognises Active by this prefix
    for s in (v2, P.system_prompt("random_walk"), P.system_prompt("oracle_route")):
        low = s.lower().replace("notes are not scored", "")
        for w in LEAK:
            assert w not in low, (w, s[:80])
    # v2 differs from v1 only in the Actions / Budget / Odometry sections
    strip = lambda s: s.split("Photographs\n", 1)[1]
    assert strip(v1) == strip(v2)


def test_route_prompts_shared():
    a, b = P.system_prompt("random_walk"), P.system_prompt("oracle_route")
    assert a == b                                                                       # both route baselines have the same text
    assert P.CONVENTIONS in a and P.ANSWER_RULES in a                                    # shares the Conventions / Answers sections with Passive / Oracle
    assert "in the order in which the viewpoints were first visited" in a
    assert P.system_prompt("oracle_min", n_photos=9) == P.system_prompt("passive", n_photos=9) == P.system_prompt("oracle", n_photos=9)


# ---------------------------------------------------------------- cumulative pose

def test_cumulative_pose_matches_geometry(data):
    """Cumulative pose = displacement and heading of the true pose relative to START (rounded); composing the unrounded per-step motions step by step must give the same result (consistent coordinate frame)."""
    env = ActSpaceEnv(data, render=False, mode="active", cumulative_pose=True)
    rng = np.random.default_rng(0)
    for gid in GIDS:
        obs = env.reset(gid)
        assert obs["pose"] == {"forward_m": 0.0, "left_m": 0.0, "heading_deg": 0}
        start = env.scene.pose[env.node]
        x = y = 0.0
        h = 0.0
        for _ in range(40):
            prev = env.node
            obs, _, done = env.step(MOVE_ACTIONS[int(rng.integers(6))])
            if obs["moved"]:
                f, l, t = relative_motion(env.scene.pose[prev], env.scene.pose[env.node])
                r = np.radians(h)
                x, y, h = x + f * np.cos(r) - l * np.sin(r), y + f * np.sin(r) + l * np.cos(r), h + t
            F, Lf, T = relative_motion(start, env.scene.pose[env.node])
            assert abs(F - x) < 1e-6 and abs(Lf - y) < 1e-6 and abs(((T - h) + 180) % 360 - 180) < 1e-6
            p = obs["pose"]
            assert abs(p["forward_m"] - F) <= 0.05 + 1e-9 and abs(p["left_m"] - Lf) <= 0.05 + 1e-9
            assert abs(((p["heading_deg"] - T) + 180) % 360 - 180) <= 2.5 + 1e-9 and -180 < p["heading_deg"] <= 180
            assert env.records[-1]["pose"] == p
            if done or obs["must_answer"]:
                break


def test_v1_records_unchanged(data):
    env = ActSpaceEnv(data, render=False, mode="active")
    obs = env.reset(GIDS[0])
    assert "pose" not in obs and "cumulative_pose" not in env.records[0]
    obs, _, _ = env.step("turn_left")
    assert "pose" not in obs and "pose" not in env.records[-1]


# ---------------------------------------------------------------- route settings

def test_random_walk(data):
    walks = []
    for seed in (0, 1, 2, 0):
        env = ActSpaceEnv(data, render=False, mode="random_walk", run_seed=seed)
        obs = env.reset("arkit_g_000015")                                                       # group with masked nodes
        route = env.records[0]["route"]
        assert len(route) == 40 == len(obs["route"])
        cur = env.node
        for a, n in route:
            assert env.scene.edges[cur][a] == n and n not in env.masked                   # every step succeeds
            cur = n
        nodes = [env.node] + [n for _, n in route]
        assert env.shown == list(dict.fromkeys(nodes))                                     # photos in order of first visit
        assert [p["photo_id"] for p in obs["photos"]] == env.records[0]["shown_photo_ids"]
        assert obs["photos"][0]["start"] and obs["start_pose"] == {"forward_m": 0.0, "left_m": 0.0, "heading_deg": 0}
        assert [r["pose"] for r in obs["route"]] == [env.pose_rel_start(n) for _, n in route]
        walks.append(route)
    assert walks[0] == walks[3] and walks[0] != walks[1] and walks[1] != walks[2]


def test_oracle_route(data):
    meta = group_meta(data)
    env = ActSpaceEnv(data, render=False, mode="oracle_route")
    for gid in GIDS:
        obs = env.reset(gid)
        g = data.groups[gid]
        route, _ = plan_route(env.scene, env.masked, g["init_frame"], oracle_targets(g))
        assert env.records[0]["route"] == [list(x) for x in route] and len(route) == meta[gid]["group_route_steps"]
        assert set(oracle_targets(g)) <= set(env.shown)
        text = "".join(p[1] for p in P.user_route(obs) if p[0] == "text")
        n_img = sum(p[0] == "image" for p in P.user_route(obs))
        assert n_img == len(obs["photos"]) and text.count("Move ") == len(route)
        assert "Pose relative to START: forward 0.0 m, left 0.0 m, heading 0 degrees." in text


def test_passive_k_nested_seeds(data):
    """passive_k uses the same sampling as Passive-40 with a fixed seed (determined by the question group and k)."""
    for k in (8, 20):
        a = ActSpaceEnv(data, budget=k, render=False, mode="passive")
        a.reset(GIDS[1])
        b = ActSpaceEnv(data, budget=k, render=False, mode="passive")
        b.reset(GIDS[1])
        assert a.shown == b.shown and len(a.shown) == k + 1


# ---------------------------------------------------------------- mock end to end

@pytest.mark.parametrize("setting,budget,seed", [("active_v2", 40, 0), ("active_v1", 40, 0), ("random_walk", 40, 1),
                                                 ("oracle_route", 40, 0), ("oracle_min", 40, 0), ("passive", 8, 0), ("passive", 20, 0)])
def test_mock_new_settings(data, tmp_path, setting, budget, seed):
    st = run(data, tmp_path, setting, "r", budget=budget, run_seed=seed)
    assert st["status"] == "complete" and st["completed"] == 3 and st["setting"] == setting
    path = tmp_path / "r" / "trajectories.jsonl"
    assert replay(data, path)["mismatched_episodes"] == 0
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["setting"] == setting and s["run_seed"] == seed and s["episodes"] == 3
    calls = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    user0 = "".join(p for p in calls[0]["request"]["messages"][1]["content"] if not p.startswith("<image"))
    if setting == "active_v2":
        assert all(ep[0]["cumulative_pose"] and all("pose" in r for r in ep if r["type"] == "step") for ep in complete_episodes(path))
        later = "".join(p for p in calls[1]["request"]["messages"][-1]["content"] if not p.startswith("<image"))
        assert "Pose relative to START: forward 0.0 m, left 0.0 m, heading 0 degrees." in user0 and "Pose relative to START" in later
        assert "A move covers about 0.3 m." in calls[0]["request"]["messages"][0]["content"]
    if setting in ("random_walk", "oracle_route"):
        assert "Path record (" in user0 and len(calls) == 3
        assert all(e["route_moves"] > 0 for e in s["per_episode"])


def test_effort_override(data, tmp_path):
    from actspace_eval.llm_run import default_effort
    m = data.ecfg["models"]["gpt-6.1-sol"]
    assert with_effort(m, "medium")["reasoning"]["effort"] == "medium" and m["reasoning"]["effort"] == "high"
    # the official evaluation always uses high (llm.reasoning_effort); --effort is only for comparisons; not set for models without reasoning
    assert data.ecfg["llm"]["reasoning_effort"] == "high"
    assert default_effort(data.ecfg, "gpt-6-luna", None) == "high" and default_effort(data.ecfg, "gpt-6-luna", "low") == "low"
    assert default_effort(data.ecfg, "mock", None) is None
    with pytest.raises(ValueError):
        with_effort(data.ecfg["models"]["mock"], "high")


def test_estimate_new_plan(data):
    rows = estimate(env_factory(data), GIDS, estimate_plan(40), data.ecfg["models"]["mock"], data.ecfg["llm"]["history"],
                    data.ecfg["env"]["step_scale"])
    by = {(r["mode"], r["budget"], r["active_steps"]): r for r in rows}
    assert by[("random_walk", 40, None)]["calls"] == 9 and by[("oracle_route", 40, None)]["calls"] == 3
    assert by[("passive", 8, None)]["images"] == 3 * 9 and by[("active_v2", 40, 40)]["calls"] == 3 * 41
    assert by[("active_v2", 40, 40)]["cached_input_tokens"] > 0


# ---------------------------------------------------------------- derived quantities

def test_derived_quantities(data, tmp_path):
    root = tmp_path
    for setting, name, b, seed in [("active_v2", "a2", 40, 0), ("passive", "p40", 40, 0), ("passive", "p20", 20, 0),
                                   ("random_walk", "rw0", 40, 0), ("random_walk", "rw1", 40, 1), ("random_walk", "rw2", 40, 2),
                                   ("oracle_min", "om", 40, 0), ("oracle_route", "orr", 40, 0)]:
        assert run(data, tmp_path, setting, name, budget=b, run_seed=seed)["status"] == "complete"
    spec = {"active_v2": ["a2"], "passive_40": ["p40"], "passive_20": ["p20"], "random_walk_40": ["rw0", "rw1", "rw2"],
            "oracle_min": ["om"], "oracle_route": ["orr"]}
    R = build(data, root, spec)
    assert len(R["runs"]["random_walk_40"]["per_question"]) == 3 * len(R["runs"]["active_v2"]["per_question"])
    d = R["derived"]["active_v2"]
    assert set(d) == {"Active − Passive-40", "Active − Passive-20", "Active − random_walk_40", "oracle_min − Active", "oracle_route − Active"}
    a = np.mean([x["score"] for x in R["runs"]["active_v2"]["per_question"]])
    rw = np.mean([x["score"] for x in R["runs"]["random_walk_40"]["per_question"]])
    assert abs(d["Active − random_walk_40"]["all"]["mean"] - (a - rw)) < 1e-9
    text = "\n".join(render(R))
    assert "strictly visible" in text and "oracle_route − Active" in text


# ---------------------------------------------------------------- excluded questions

@pytest.fixture()
def data_v011(data):
    """Older single-dataset version arkit_only_v0.11 (the exclusion list and the Q28 rule target it; in the merged version these 12 questions were removed by the generation rules).
    The migration bundle only carries merged_v1.0 used for the official evaluation; skip when the question files of this old version are missing."""
    try:
        return FrozenData({**data.ecfg, "version": "arkit_only_v0.11"}, verify=False)
    except (FileNotFoundError, FreezeError) as e:
        pytest.skip(f"question files of the old version arkit_only_v0.11 are missing (the migration bundle only carries merged_v1.0): {e}")


def test_exclusions(data_v011, tmp_path):
    from actspace_eval.score import consistency, exclude_questions, load_exclusions
    data = data_v011
    ex = load_exclusions()
    assert len(ex) == 12 and all(ex.values())
    qids = {q["qid"] for g in data.groups.values() for q in g["questions"]}
    assert set(ex) <= qids
    assert run_llm(data, "mock", "oracle_min", 40, ["g_000000", "g_000064", "g_000015"], tmp_path / "om", render=False,
                   print_estimate=False, run_seed=0)["status"] == "complete"
    from actspace_eval.score import load_run
    r = load_run(tmp_path, "om")
    r = {**r, "per_episode": r["per_episode"] + [{"group_id": "g_000032"}]}   # bring the constraints of g_000032 into the check
    r["answers_by_qid"] = {**r["answers_by_qid"], "g_000032_q1": "A", "g_000032_q2": "A"}
    n0 = len(consistency(data, r))
    rx = exclude_questions(r, ["g_000032_q1", "g_000032_q2"])
    assert len(consistency(data, rx)) == n0 - 1 and rx["excluded_qids"] == ["g_000032_q1", "g_000032_q2"]
    assert len(exclude_questions(r, ex)["per_question"]) == len([x for x in r["per_question"] if x["qid"] not in ex])


def test_results_report_parts(data, tmp_path):
    """All parts of the results report (report_stage3r) can be generated from mock runs; the main table excludes degenerate questions."""
    from actspace_eval import report_stage3r as RS
    from actspace_eval.score import load_exclusions
    for setting, name, seed in [("active_v2", "a2", 0), ("random_walk", "rw0", 0), ("random_walk", "rw1", 1), ("oracle_route", "orr", 0)]:
        assert run(data, tmp_path, setting, name, run_seed=seed)["status"] == "complete"
    spec = {"active_v2": ["a2"], "random_walk_40": ["rw0", "rw1"], "oracle_route": ["orr"]}
    R = build(data, tmp_path, spec, exclude=load_exclusions())
    B = {"active_v2": RS.behavior(data, tmp_path, ["a2"], R["runs"]["active_v2"])}
    assert B["active_v2"]["episodes"] == 3 and B["active_v2"]["n_actions"] == 15
    text = "\n".join(RS.score_tables(R, "## t") + RS.diff_tables(R) + RS.behavior_table(B) + RS.per_group_table(R)
                     + RS.cost_table(tmp_path, spec)[0])
    assert "seed mean" in text and "Active − random_walk_40" in text and "Revisit rate" in text and "Total (this round)" in text


def test_exclusions_match_rule(data_v011, data):
    """The exclusion list and the generation rule (min_horizontal_sep_m = 0.5, standing point only) agree exactly on v0.11; the merged version has no violating questions."""
    from actspace_eval.score import horizontal_sep_violations, load_exclusions
    assert horizontal_sep_violations(data) == {}
    v = horizontal_sep_violations(data_v011)
    assert set(v) == set(load_exclusions())
    assert all(d < 0.5 for d, _ in v.values())



# ---------------------------------------------------------------- answer rate, model comparison preparation

def test_answer_metrics():
    from actspace_eval.score import Boot, answer_metrics
    q = lambda s, a, ans, sc: {"scene_id": s, "group_id": "g", "answerable": a, "answer": ans, "score": sc}
    run = {"per_question": [q("s1", True, "A", 1.0), q("s1", True, "cannot_determine", 0.0), q("s2", True, None, 0.0),
                            q("s2", True, 3.0, 0.5), q("s1", False, "cannot_determine", 1.0), q("s2", False, "B", 0.0)]}
    m = answer_metrics(run, Boot({"s1", "s2"}, n=50))
    assert m["answer rate (answerable)"]["mean"] == pytest.approx(0.5) and m["answer rate (answerable)"]["n"] == 4
    assert m["selective score (answered answerable)"]["mean"] == pytest.approx(0.75) and m["selective score (answered answerable)"]["n"] == 2
    assert m["correct abstention (unanswerable)"]["mean"] == pytest.approx(0.5)


def test_new_model_configs(data, monkeypatch):
    from actspace_eval.models import OpenAIClient
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    for key, prices in (("gpt-6.1-terra", (1.00, 0.05, 5.00)), ("gpt-6-luna", (0.10, 0.01, 0.50))):
        m = data.ecfg["models"][key]
        assert (m["prices_usd_per_mtok"]["input"], m["prices_usd_per_mtok"]["cached_input"], m["prices_usd_per_mtok"]["output"]) == prices
        p = OpenAIClient(with_effort(m, "high"), data.ecfg["llm"], transport=lambda b: None).params()
        sol = OpenAIClient(with_effort(data.ecfg["models"]["gpt-6.1-sol"], "high"), data.ecfg["llm"], transport=lambda b: None).params()
        assert p == {**sol, "model": key}      # differs from gpt-6.1-sol only in the model name
    assert data.ecfg["cost_cap_usd"] == 15.0


def test_probe_fake_transport(data, tmp_path, monkeypatch):
    from actspace_eval.models import ModelError
    from actspace_eval.probe import probe_messages, probe_model
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    msgs = probe_messages(data)
    assert sum(p[0] == "image" for p in msgs[2][0]["parts"]) == 1 and all(p[0] == "text" for p in msgs[1][0]["parts"])
    seen = []

    def ok(body):
        seen.append(body)
        n_img = sum(c.get("type") == "image_url" for m in body["messages"] if isinstance(m["content"], list) for c in m["content"])
        return {"choices": [{"message": {"content": '{"action": "turn_left"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1500 + 400 * n_img, "completion_tokens": 30,
                          "prompt_tokens_details": {"cached_tokens": 1280 if len(seen) == 3 else 0, "cache_write_tokens": 7},
                          "completion_tokens_details": {"reasoning_tokens": 12}}}
    r = probe_model(data, "gpt-6-luna", "high", tmp_path, msgs, {"transport": ok})
    assert r["all_accepted"] and r["reasoning_used"] and r["valid_actions"] == 3
    assert r["tokens_per_image"] == 400 and r["cached_on_repeat"] == 1280 and r["params"]["reasoning_effort"] == "high"
    assert seen[0]["prompt_cache_key"] == "actspace-probe-gpt-6-luna" and seen[0]["model"] == "gpt-6-luna"
    log = [json.loads(l) for l in open(tmp_path / "gpt-6-luna.jsonl")]
    assert log[0]["usage_raw"]["prompt_tokens_details"]["cache_write_tokens"] == 7     # raw usage written to the call log
    assert "sk-test" not in (tmp_path / "gpt-6-luna.jsonl").read_text()

    def reject(body):
        raise ModelError("HTTP 400: reasoning_effort not supported", retryable=False, status=400)
    r = probe_model(data, "gpt-6.1-terra", "high", tmp_path, msgs, {"transport": reject})
    assert not r["all_accepted"] and all(c["status"] == 400 for c in r["calls"]) and "tokens_per_image" not in r


# ---------------------------------------------------------------- cache write billing

def test_cache_write_cost(data):
    from actspace_eval.cost import CostTracker
    p = data.ecfg["models"]["gpt-6.1-terra"]["prices_usd_per_mtok"]
    assert (p["cache_write"], p["cache_write_12h"]) == (1.25, 2.50)
    t = CostTracker(p, 1.0)
    u = {"input_tokens": 2000, "cached_input_tokens": 500, "cache_write_tokens": 1200, "output_tokens": 100}
    assert t.cost_of(u) == pytest.approx((300 * 1.00 + 500 * 0.05 + 1200 * 1.25 + 100 * 5.00) / 1e6)
    assert t.cost_of({**u, "cache_write_12h_tokens": 200}) == pytest.approx((300 * 1.00 + 500 * 0.05 + 1000 * 1.25 + 200 * 2.50 + 100 * 5.00) / 1e6)
    old = CostTracker({"input": 2.5, "cached_input": 1.25, "output": 10.0}, 1.0)          # no write price: same as before
    assert old.cost_of(u) == pytest.approx((1500 * 2.5 + 500 * 1.25 + 100 * 10.0) / 1e6)
    # pre-call check: all input estimated as writes
    tight = CostTracker(p, (1000 * 1.25 + 10 * 5.00) / 1e6 - 1e-9)
    with pytest.raises(Exception):
        tight.check_before(1000, 10)


def test_cache_write_parsed_and_logged(data, tmp_path, monkeypatch):
    from actspace_eval.cost import CostTracker, call_usage, recost_log
    from actspace_eval.models import OpenAIClient
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-000000000000")
    bodies = []

    def transport(body):
        bodies.append(body)
        return {"choices": [{"message": {"content": '{"action": "forward"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1564, "completion_tokens": 20,
                          "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1200},
                          "completion_tokens_details": {"reasoning_tokens": 5}}}
    for key in ("gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"):
        m = with_effort(data.ecfg["models"][key], "high")
        cost = CostTracker(m["prices_usd_per_mtok"], 1.0)
        c = OpenAIClient(m, data.ecfg["llm"], cost, tmp_path / f"{key}.jsonl", transport=transport)
        c.cache_key = "k"
        comp = c.complete("SYS", [{"role": "user", "text": "hi"}])
        assert comp.usage["cache_write_tokens"] == 1200 and comp.usage["cache_write_12h_tokens"] == 0
        p = m["prices_usd_per_mtok"]
        assert comp.cost_usd == pytest.approx((364 * p["input"] + 1200 * p["cache_write"] + 20 * p["output"]) / 1e6)
        # default 30-minute tier: no extended-retention parameters sent
        assert not {"prompt_cache_retention", "prompt_cache_options"} & set(bodies[-1]) and bodies[-1]["prompt_cache_key"] == "k"
        assert CostTracker.usage_in_log(tmp_path / f"{key}.jsonl")["cache_write_tokens"] == 1200
        r = recost_log(tmp_path / f"{key}.jsonl", p)
        assert r["cost"] == pytest.approx(comp.cost_usd) and r["logged"] == pytest.approx(comp.cost_usd) and r["raw_calls"] == 1
    # old records: usage without writes, filled in from usage_raw; without usage_raw the upper bound counts all uncached input as writes
    rec = {"type": "call", "usage": {"input_tokens": 2000, "cached_input_tokens": 500, "output_tokens": 0},
           "usage_raw": {"prompt_tokens_details": {"cache_write_tokens": 700}}, "cost_usd": 0.0}
    assert call_usage(rec)["cache_write_tokens"] == 700
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({**rec, "usage_raw": None}) + "\n" + json.dumps(rec) + "\n")
    p = data.ecfg["models"]["gpt-6.1-sol"]["prices_usd_per_mtok"]
    r = recost_log(path, p)
    assert r["calls"] == 2 and r["raw_calls"] == 1
    assert r["cost"] == pytest.approx((1500 * 2.0 + 500 * 0.1 + 800 * 2.0 + 500 * 0.1 + 700 * 2.5) / 1e6)
    assert r["upper"] == pytest.approx((1500 * 2.5 + 500 * 0.1 + 800 * 2.0 + 500 * 0.1 + 700 * 2.5) / 1e6)
    assert r["worst"] == pytest.approx(2 * (1500 * 2.5 + 500 * 0.1) / 1e6)


def test_estimate_includes_writes(data):
    m = data.ecfg["models"]["gpt-6.1-terra"]
    rows = estimate(env_factory(data), GIDS, [("passive", 40, 0), ("active_v2", 40, 10)], m, data.ecfg["llm"]["history"],
                    data.ecfg["env"]["step_scale"])
    pas, act = rows
    assert pas["cache_write_tokens"] == pas["input_tokens"]          # static settings: every call has a new prefix
    assert 0 < act["cache_write_tokens"] <= act["input_tokens"] - act["cached_input_tokens"]
    assert act["cost_no_cache_usd"] > act["cost_usd"]


def test_models_report_parts(data, tmp_path):
    """All parts of the model comparison report (report_models) can be generated from mock runs; differences between models are paired and the matrix marks monotonicity."""
    from actspace_eval import report_models as RM_
    from actspace_eval.report_stage3r import behavior
    from actspace_eval.score import load_exclusions
    for setting, name, seed, budget in [("blind", "b", 0, 40), ("passive", "p", 0, 40), ("random_walk", "rw0", 0, 40),
                                        ("random_walk", "rw1", 1, 40), ("active_v2", "a", 0, 40)]:
        assert run(data, tmp_path, setting, name, budget=budget, run_seed=seed)["status"] == "complete"
    spec = {"blind": ["b"], "passive_40": ["p"], "random_walk_40": ["rw0", "rw1"], "active_v2_high": ["a"]}
    RM = {m: build(data, tmp_path, spec, exclude=load_exclusions()) for m in RM_.MODELS}
    text = "\n".join(RM_.score_table(RM, "## t") + RM_.matrix(RM) + RM_.gain_table(RM) + RM_.model_diff_table(RM)[0])
    assert "luna < terra < sol" in text and "Active − random_walk_40" in text and "terra − luna" in text
    d = RM_.model_diffs(RM)
    assert all(v["mean"] == 0 and v["lo"] == 0 and v["hi"] == 0 for row in d.values() for v in row.values())   # same runs: difference is 0
    B = {m: behavior(data, tmp_path, ["a"], RM[m]["runs"]["active_v2_high"]) for m in RM_.MODELS}
    assert "Revisit rate" in "\n".join(RM_.behavior_table(B, RM_.MODELS, RM_.SHORT))
    rows = RM_.cost_rows(tmp_path, {"gpt-6-luna": spec}, {"gpt-6-luna": data.ecfg["models"]["gpt-6-luna"]["prices_usd_per_mtok"]})
    assert len(rows) == 4 and rows[2]["runs"] == 2 and "luna total" in "\n".join(RM_.cost_table(rows)[0])
