"""Unit tests for the evaluation environment (docs/evaluation_protocol.md). Uses the real frozen data (read-only)."""
import json
import re

import numpy as np
import pytest

from actspace_eval import answers as A
from actspace_eval.agents import RandomAgent, ScriptedOracleAgent, shortest_path
from actspace_eval.data import MOVE_ACTIONS, FreezeError, FrozenData
from actspace_eval.env import CANNOT_MOVE, ActSpaceEnv
from actspace_eval.odometry import describe, odometry, relative_motion
from actspace_eval.run import run_episode
from actspace_eval.trajectory import TrajectoryWriter, read_episodes, replay, replay_episode


@pytest.fixture(scope="module")
def data():
    return FrozenData(verify=False)


def masked_groups(data):
    return [gid for gid in data.group_ids if data.masked_nodes(gid)]


# ---------------------------------------------------------------- frozen data verification

def test_freeze_manifest_verified(data, monkeypatch):
    FrozenData(verify=True, source="frozen")  # matches the manifest (merged_v1.0: merge_build.verify_version)
    from actspace_build import merge_build
    real = merge_build.hashes

    def tampered(*a, **k):
        h = real(*a, **k)
        h["files"]["test_groups.jsonl"] = {"sha256": "0" * 64, "bytes": 0}
        return h
    monkeypatch.setattr(merge_build, "hashes", tampered)
    with pytest.raises(FreezeError):
        FrozenData(verify=True, source="frozen")


# ---------------------------------------------------------------- action semantics

@pytest.mark.parametrize("which", [0, 1])
def test_actions_follow_view_graph(data, which):
    """For every node and action: an edge (with an unmasked target) → reach the edge's target node; otherwise "cannot move" and the position is unchanged; each costs 1 step."""
    gid = [data.group_ids[0], masked_groups(data)[0]][which]
    env = ActSpaceEnv(data, budget=10 ** 6, render=False)
    env.reset(gid, seed=1)
    scene, masked = env.scene, env.masked
    for n in scene.nodes:
        if n in masked:
            continue
        for a in MOVE_ACTIONS:
            env.node = n
            before = env.steps_used
            obs, fb, done = env.step(a)
            tgt = scene.edges[n][a]
            if tgt is None or tgt in masked:
                assert env.node == n and not obs["moved"] and fb == CANNOT_MOVE
            else:
                assert env.node == tgt and obs["moved"]
            assert env.steps_used == before + 1 and not done


def test_invalid_action_counts_one_step(data):
    env = ActSpaceEnv(data, budget=5, render=False)
    obs = env.reset(data.group_ids[0], seed=1)
    obs, fb, done = env.step("jump")
    assert fb.startswith("Invalid action") and obs["steps_used"] == 1 and not obs["moved"] and not done


# ---------------------------------------------------------------- masked nodes

def test_masked_nodes_never_returned(data):
    """All masked groups (merged_v1.0: 65 groups across both datasets; the masked question q4 of arkit_g_000073 was removed by the standing-point spacing rule but the group's mask is kept —
    the oracles of the other questions in the group are computed on the masked view graph): during long random walks, reached nodes are never masked; edges to masked nodes always give "cannot move"."""
    gids = masked_groups(data)
    assert len(gids) == sum(bool(g.get("view_graph_mask")) for g in data.groups.values()) == 65
    assert {g.split("_", 1)[0] for g in gids} == {"arkit", "scannetpp"}
    for gid in gids:
        env = ActSpaceEnv(data, budget=400, render=False)
        obs = env.reset(gid, seed=7)
        agent = RandomAgent()
        agent.reset(obs, seed=7)
        done = False
        while not done:
            obs, _, done = env.step(agent.act(obs))
        steps = [r for r in env.records if r["type"] == "step"]
        assert not {r["to"] for r in steps} & env.masked
        for r in steps:
            tgt = env.scene.edges[r["from"]].get(r["action"])
            if tgt in env.masked:
                assert not r["moved"] and r["to"] == r["from"]


# ---------------------------------------------------------------- budget and forced answers

def test_budget_and_forced_answer(data):
    gid = data.group_ids[0]
    env = ActSpaceEnv(data, budget=10, render=False)
    obs = env.reset(gid, seed=3)
    assert obs["steps_left"] == 10 and not obs["must_answer"]
    for i in range(10):
        obs, _, done = env.step(MOVE_ACTIONS[i % 6])
        assert not done
    assert obs["must_answer"] and obs["steps_left"] == 0 and env.steps_used == 10
    # move submitted after the budget is used up: not executed, the episode ends with an empty answer, the whole group scores 0
    node = env.node
    obs, fb, done = env.step("forward")
    assert done and env.node == node and env.steps_used == 10
    end = env.records[-1]
    assert end["forced"] == "move_after_budget" and not end["parse_ok"] and set(end["scores"].values()) == {0.0}
    # normal answer after the budget is used up: recorded as a forced answer (budget_exhausted), scored as usual
    obs = env.reset(gid, seed=3)
    for i in range(10):
        obs, _, _ = env.step("turn_left")
    gold = {qp["id"]: A.gold(q) for qp, q in zip(obs.get("questions") or env.q_public, data.groups[gid]["questions"])}
    _, _, done = env.step({"action": "answer", "content": json.dumps(gold)})
    assert done and env.records[-1]["forced"] == "budget_exhausted" and set(env.records[-1]["scores"].values()) == {1.0}
    with pytest.raises(RuntimeError):
        env.step("forward")


@pytest.mark.parametrize("budget", [10, 20, 40])
def test_supported_budgets(data, budget):
    env = ActSpaceEnv(data, budget=budget, render=False)
    obs = env.reset(data.group_ids[5], seed=0)
    agent = RandomAgent()
    agent.reset(obs, seed=0)
    done, n = False, 0
    while not done:
        obs, _, done = env.step(agent.act(obs))
        n += 1
    assert env.steps_used == budget and n == budget + 1


# ---------------------------------------------------------------- photo IDs and leak prevention

def test_photo_ids_randomized_and_no_leak(data):
    gid = data.group_ids[0]
    g = data.groups[gid]
    env = ActSpaceEnv(data, render=False)
    ids = {env.reset(gid, seed=s)["photo_id"] for s in range(20)}
    assert len(ids) == 20                                     # different episodes get different IDs
    assert env.reset(gid, seed=5)["photo_id"] == env.reset(gid, seed=5)["photo_id"]  # same seed is reproducible
    obs = env.reset(gid, seed=5)
    start_id = obs["photo_id"]
    assert re.fullmatch(r"photo_\d{6}", start_id)
    # walk away and back to the start: the same node keeps its ID within one episode
    a = next(a for a in ("forward", "backward", "left", "right") if env.scene.edges[g["init_frame"]][a])
    env.step(a)
    path = shortest_path(env.scene, env.masked, env.node, g["init_frame"])
    for act, _ in path:
        obs, _, _ = env.step(act)
    assert obs["photo_id"] == start_id
    # public observations contain no frame IDs, file names, group IDs, qids or gold-answer fields
    start = env.reset(gid, seed=5)
    text = json.dumps({k: v for k, v in start.items() if k != "image"})
    assert not re.search(r"f_\d{6}|\.png|\.jpg|g_\d{6}|_q\d", text)
    for q in start["questions"]:
        assert set(q) == {"id", "text", "answer_type", "unit", "options"}
    assert set(start) >= {"questions", "photo_id", "image", "steps_used", "steps_left", "must_answer"}
    assert not set(start) & {"node", "frame", "answer", "oracle_nodes", "evidence_objs", "visibility"}


def test_image_is_upright_original(data):
    env = ActSpaceEnv(data, render=True)
    obs = env.reset(data.group_ids[0], seed=0)
    img = obs["image"]
    assert img.dtype == np.uint8 and img.ndim == 3 and sorted(img.shape[:2]) == [480, 640]


# ---------------------------------------------------------------- odometry

def test_odometry_math():
    a = (np.array([1.0, 2.0, 0.5]), 90.0)                   # facing +y
    b = (np.array([0.0, 2.5, 0.7]), 130.0)                  # 0.5 m forward, 1.0 m left, 40° left turn
    f, l, t = relative_motion(a, b)
    assert np.allclose([f, l, t], [0.5, 1.0, 40.0])
    od = odometry(a, b)
    assert od == {"forward_m": 0.5, "left_m": 1.0, "turn_deg": 40}
    assert describe(od) == "Moved forward about 0.5 m and left about 1.0 m; turned left about 40 degrees."
    # rounding: 0.25 → 0.3, 37.5° → 40°, turns crossing ±180°
    od = odometry((np.zeros(3), 180.0), (np.array([-0.25, 0.0, 0.0]), -142.5))
    assert od == {"forward_m": 0.3, "left_m": 0.0, "turn_deg": 40}
    assert describe({"forward_m": -0.3, "left_m": -0.2, "turn_deg": -15}) == \
        "Moved backward about 0.3 m and right about 0.2 m; turned right about 15 degrees."
    assert describe({"forward_m": 0.0, "left_m": 0.0, "turn_deg": 45}) == "Turned left about 45 degrees."


def test_env_odometry_matches_poses_and_switch(data):
    gid = data.group_ids[3]
    env = ActSpaceEnv(data, budget=60, render=False)
    obs = env.reset(gid, seed=11)
    agent = RandomAgent()
    agent.reset(obs, seed=11)
    while env.steps_used < 60:
        obs, fb, _ = env.step(agent.act(obs))
        r = env.records[-1]
        if r["moved"]:
            assert r["odometry"] == odometry(env.scene.pose[r["from"]], env.scene.pose[r["to"]]) and fb == describe(r["odometry"])
        else:
            assert r["odometry"] is None
    off = ActSpaceEnv(data, budget=60, odometry_on=False, render=False)
    obs = off.reset(gid, seed=11)
    for r in [x for x in env.records if x["type"] == "step"]:
        obs, fb, _ = off.step(r["action"])
        assert obs["odometry"] is None and fb in ("Moved.", CANNOT_MOVE)


# ---------------------------------------------------------------- answer parsing and scoring

def test_answer_parsing_and_scoring(data):
    g = next(g for g in data.groups.values() if {q["answer_type"] for q in g["questions"]} >= {"numeric", "multiple_choice"})
    qp = A.public_questions(g)
    num = next(q for q in qp if q["answer_type"] == "numeric")
    mc = next(q for q in qp if q["answer_type"] == "multiple_choice")
    letter = mc["options"][1].split(".")[0]
    raw = f'Here is my answer:\n```json\n{{"{num["id"].lower()}": "2.5 m", "{mc["id"]}": "{letter}. whatever"}}\n```'
    p = A.parse_answers(raw, qp)
    assert p["ok"] and p["answers"][num["id"]] == 2.5 and p["answers"][mc["id"]] == letter
    assert A.parse_answers("no json here", qp)["ok"] is False
    p = A.parse_answers({num["id"]: -1, mc["id"]: "Z"}, qp)
    assert p["answers"][num["id"]] is None and p["answers"][mc["id"]] is None and len(p["errors"]) == len(qp)
    assert A.parse_answers({mc["id"]: "Cannot_Determine"}, qp)["answers"][mc["id"]] == A.CANNOT
    # scoring: T1 MRA same as GATE D; unanswerable scores only for cannot_determine; answerable answered cannot_determine scores 0
    q1 = {"evidence_label": "explore", "answer_type": "numeric", "answer_raw": 2.0}
    assert A.score_question(q1, 2.0) == 1.0 and A.score_question(q1, 2.5) == 0.5 and A.score_question(q1, A.CANNOT) == 0.0
    qu = {"evidence_label": "unanswerable", "answer_type": "multiple_choice", "answer": "B"}
    assert A.score_question(qu, A.CANNOT) == 1.0 and A.score_question(qu, "B") == 0.0
    qm = {"evidence_label": "explore", "answer_type": "multiple_choice", "answer": "B"}
    assert A.score_question(qm, "B") == 1.0 and A.score_question(qm, None) == 0.0


# ---------------------------------------------------------------- trajectories, replay, fake agents

def test_trajectory_replay(data, tmp_path):
    env = ActSpaceEnv(data, budget=20, render=False, run_seed=4)
    path = tmp_path / "t.jsonl"
    w = TrajectoryWriter(path)
    agent = RandomAgent()
    for gid in data.group_ids[:3] + masked_groups(data)[:2]:
        records, seed = run_episode(env, agent, gid, run_seed=4)
        w.write_episode(records, run="test", agent=agent.name, agent_seed=seed)
    w.close()
    res = replay(data, path)
    assert res["episodes"] == 5 and res["mismatched_episodes"] == 0
    eps = read_episodes(path)
    eps[0][3]["to"] = "f_999999"                              # tamper with one record → replay detects it
    assert replay_episode(data, eps[0])


def test_scripted_oracle_full_score(data):
    env = ActSpaceEnv(data, budget=30, render=False)
    agent = ScriptedOracleAgent(data)
    for gid in data.group_ids[:4] + masked_groups(data)[:2]:
        records, _ = run_episode(env, agent, gid)
        steps = [(r["action"], r["to"]) for r in records if r["type"] == "step"]
        assert steps == [tuple(x) for x in agent.route[:len(steps)]]   # every step reaches the planned node
        if len(agent.route) <= 30:
            assert len(steps) == len(agent.route)
        assert set(records[-1]["scores"].values()) == {1.0}


def test_plan_route_exact_vs_greedy():
    """The exact route is optimal while greedy takes a detour: a is 2 steps left of S; b is 1 step ahead of S, and c 2 more steps ahead.
    Greedy S→b→c→(back to S)→a = 1 + 2 + 5 = 8 steps; optimal S→a→(back to S)→b→c = 2 + 3 + 2 = 7 steps."""
    from types import SimpleNamespace
    from actspace_eval.agents import _greedy_route, plan_route
    star = {n: {k: None for k in MOVE_ACTIONS} for n in ["S", "a1", "a", "b", "c1", "c"]}

    def link(x, y, f, bk):
        star[x][f], star[y][bk] = y, x
    link("S", "a1", "left", "right")
    link("a1", "a", "left", "right")
    link("S", "b", "forward", "backward")
    link("b", "c1", "forward", "backward")
    link("c1", "c", "forward", "backward")
    scene = SimpleNamespace(edges=star)
    r, u = plan_route(scene, set(), "S", ["a", "b", "c"])
    g, _ = _greedy_route(scene, set(), "S", ["a", "b", "c"], [])
    assert not u and len(r) == 7 and len(g) == 8
    assert [n for _, n in r][-1] == "c"
    r, u = plan_route(scene, {"a1"}, "S", ["a", "b"])   # masked node blocks the way → a unreachable
    assert u == ["a"] and [n for _, n in r] == ["b"]
