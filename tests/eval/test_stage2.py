"""Unit tests for evaluation scoring: four settings, group-level route metadata, scoring (consistency, efficiency, bootstrap)."""
import json

import numpy as np
import pytest

from actspace_eval import answers as A
from actspace_eval.agents import RandomAgent, oracle_targets
from actspace_eval.data import FrozenData
from actspace_eval.env import ActSpaceEnv
from actspace_eval.meta import compute_group_meta, efficiency, feasible_groups
from actspace_eval.score import Boot, constraint_holds


@pytest.fixture(scope="module")
def data():
    return FrozenData(verify=False)


def masked_gid(data):
    return next(g for g in data.group_ids if data.masked_nodes(g))


# ---------------------------------------------------------------- four settings

def test_blind_has_no_photos(data):
    env = ActSpaceEnv(data, mode="blind", render=True)
    obs = env.reset(data.group_ids[0], seed=1)
    assert obs["photos"] == [] and obs["must_answer"] and "image" not in obs and obs["actions"] == ["answer"]


@pytest.mark.parametrize("k", [10, 20, 40])
def test_passive_k_photos(data, k):
    """Start + k photos drawn uniformly at random from unmasked nodes; the seed depends only on the question group and k (not on the episode seed or the agent)."""
    for gid in (data.group_ids[0], masked_gid(data)):
        env = ActSpaceEnv(data, mode="passive", budget=k, render=False)
        a = env.reset(gid, seed=1)
        nodes_a = list(env.shown)
        b = env.reset(gid, seed=999)
        assert env.shown == nodes_a                                    # the drawn nodes are fixed
        assert [p["photo_id"] for p in a["photos"]] != [p["photo_id"] for p in b["photos"]]  # photo IDs are still random per episode
        pool = [n for n in env.scene.nodes if n not in env.masked and n != data.groups[gid]["init_frame"]]
        assert nodes_a[0] == data.groups[gid]["init_frame"] and a["photos"][0]["start"]
        assert len(nodes_a) == 1 + min(k, len(pool)) and len(set(nodes_a)) == len(nodes_a)
        assert not set(nodes_a) & env.masked


def test_passive_sampling_is_uniform(data):
    """Repeat the draw with 400 config seeds: always exactly k distinct photos, and every node in the pool is drawn at least once (rough uniformity check)."""
    gid = data.group_ids[0]
    env = ActSpaceEnv(data, mode="passive", budget=10, render=False)
    pool = [n for n in env.data.scene(data.groups[gid]["scene_id"]).nodes if n != data.groups[gid]["init_frame"]]
    counts = dict.fromkeys(pool, 0)
    for s in range(400):  # different config seeds simulate independent draws
        env.data.ecfg = {**env.data.ecfg, "seed": s}
        env.reset(gid, seed=0)
        assert len(env.shown) == 11 and len(set(env.shown)) == 11
        for n in env.shown[1:]:
            counts[n] += 1
    env.data.ecfg = {**env.data.ecfg, "seed": FrozenData(verify=False).ecfg["seed"]}
    freq = np.array(list(counts.values())) / 400
    assert abs(freq.mean() - 10 / len(pool)) < 1e-9 and freq.min() > 0


def test_oracle_mode_photos_are_route_targets(data):
    for gid in data.group_ids[:5] + [masked_gid(data)]:
        g = data.groups[gid]
        env = ActSpaceEnv(data, mode="oracle", render=False)
        obs = env.reset(gid, seed=0)
        assert env.shown[0] == g["init_frame"]
        assert set(env.shown) == {g["init_frame"]} | set(oracle_targets(g))
        assert not set(env.shown) & env.masked and len(obs["photos"]) == len(env.shown)


def test_static_modes_accept_only_answer(data):
    env = ActSpaceEnv(data, mode="passive", budget=10, render=False)
    env.reset(data.group_ids[0], seed=0)
    _, _, done = env.step("forward")
    assert done and env.records[-1]["forced"] == "move_in_static_mode" and set(env.records[-1]["scores"].values()) == {0.0}
    env.reset(data.group_ids[0], seed=0)
    g = data.groups[data.group_ids[0]]
    _, _, done = env.step({"action": "answer", "content": json.dumps({qp["id"]: A.gold(q) for qp, q in zip(env.q_public, g["questions"])})})
    assert done and set(env.records[-1]["scores"].values()) == {1.0} and env.records[-1]["steps_used"] == 0


# ---------------------------------------------------------------- group-level route and efficiency

def test_group_route_steps(data):
    m = compute_group_meta(data, "arkit_g_000015")
    assert m["group_route_steps"] == 36 and m["route_exact"]
    meta = {gid: compute_group_meta(data, gid) for gid in data.group_ids[:10]}
    assert all(v["group_route_steps"] >= v["oracle_steps_max"] for v in meta.values())
    assert feasible_groups({"a": {"group_route_steps": 10}, "b": {"group_route_steps": 11}}, 10) == {"a"}


def test_efficiency():
    assert efficiency(14, 14) == 1.0 and efficiency(14, 28) == 0.5 and efficiency(14, 7) == 1.0 and efficiency(5, 0) == 1.0


# ---------------------------------------------------------------- consistency

def test_constraint_holds_on_answers(data):
    g_nc = next(g for g in data.groups.values() if any(c["type"] == "numeric_vs_comparison" for c in g["constraints"]))
    con = next(c for c in g_nc["constraints"] if c["type"] == "numeric_vs_comparison")
    qby = {q["qid"]: q for q in g_nc["questions"]}
    gold = {q["qid"]: A.gold(q) for q in g_nc["questions"]}
    assert constraint_holds(con, qby, gold) is True                    # gold answers satisfy the constraint
    q1, q2, q3 = (qby[i] for i in con["qids"])
    flipped = dict(gold, **{q1["qid"]: gold[q2["qid"]], q2["qid"]: gold[q1["qid"]]})  # swap the two distances → the comparison answer is no longer consistent
    assert constraint_holds(con, qby, flipped) is False
    assert constraint_holds(con, qby, dict(gold, **{q1["qid"]: A.CANNOT})) is None
    assert constraint_holds(con, qby, dict(gold, **{q2["qid"]: gold[q1["qid"]]})) is None  # equal numbers: undeterminable
    g_di = next(g for g in data.groups.values() if any(c["type"] == "direction_inverse" for c in g["constraints"]))
    con = next(c for c in g_di["constraints"] if c["type"] == "direction_inverse")
    qby = {q["qid"]: q for q in g_di["questions"]}
    gold = {q["qid"]: A.gold(q) for q in g_di["questions"]}
    assert constraint_holds(con, qby, gold) is True
    qa, qb = (qby[i] for i in con["qids"])
    wrong = next(L for L in "ABCD" if L != gold[qb["qid"]])
    assert constraint_holds(con, qby, dict(gold, **{qb["qid"]: wrong})) is False


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_by_scene():
    scenes = [f"s{i}" for i in range(20)]
    boot = Boot(scenes, n=500, seed=0)
    const = [(s, 0.7) for s in scenes for _ in range(3)]
    r = boot.mean_ci(const)
    assert r["mean"] == pytest.approx(0.7) and r["lo"] == pytest.approx(0.7) and r["hi"] == pytest.approx(0.7) and r["n"] == 60
    rng = np.random.default_rng(1)
    items = [(s, float(rng.random() < (0.2 if i < 10 else 0.8))) for i, s in enumerate(scenes) for _ in range(5)]
    r = boot.mean_ci(items)
    assert r["lo"] < r["mean"] < r["hi"] and r["hi"] - r["lo"] > 0.1      # large between-scene variance → wide interval
    d = boot.diff_ci(items, items)
    assert d["mean"] == 0 and d["lo"] == 0 and d["hi"] == 0               # paired: the difference on identical data is always 0
    shifted = [(s, v + 0.1) for s, v in items]
    d = boot.diff_ci(shifted, items)
    assert d["mean"] == pytest.approx(0.1) and d["lo"] == pytest.approx(0.1) and d["hi"] == pytest.approx(0.1)


def test_random_agent_static_answers_immediately(data):
    env = ActSpaceEnv(data, mode="oracle", render=False)
    obs = env.reset(data.group_ids[0], seed=0)
    agent = RandomAgent()
    agent.reset(obs, seed=0)
    _, _, done = env.step(agent.act(obs))
    assert done and env.records[-1]["parse_ok"]
