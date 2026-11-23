"""Definition tests for the Section 5.3 behaviour statistics (sec53_behavior.py): read only the frozen data, no run results."""
import numpy as np
import pytest

from actspace_eval import sec53_behavior as B
from actspace_eval.data import FrozenData
from actspace_eval.env import ActSpaceEnv
from actspace_eval.providers.ablations import vis_level


@pytest.fixture(scope="module")
def data():
    return FrozenData(verify=False)


def test_move_stats_counts():
    seq = ["a", "b", "b", "a", "a", "c"]
    acts = ["forward", "left", "turn_left", "jump", "turn_right"]
    moved = [True, False, True, False, True]
    s = B.move_stats(seq, acts, moved)
    assert s == {"revisits": 1, "arrivals": 3, "turns": 2, "actions": 5, "cannot_move": 1, "invalid": 1}
    assert B.at([1, 2, 3], 1) == 2 and B.at([1, 2, 3], 9) == 3


def test_prefix_coverage_matches_whole_route(data):
    bad = set(data.ecfg["env"]["budget_infeasible"])
    env = ActSpaceEnv(data, budget=40, render=False, mode="random_walk", run_seed=0)
    n = 0
    for gid in [g for g in data.ecfg["llm"]["tiny_groups"] if g not in bad][:12]:
        env.reset(gid)
        s = env.records[0]
        seq = [s["start_node"]] + [x for _, x in s["route"]]
        g = data.groups[gid]
        sc = data.scene(g["scene_id"])
        ev = B.evidence_objs(g)
        assert ev == sorted({o for q in g["questions"] if q["evidence_label"] != "unanswerable" for o in q["evidence_objs"]})
        allp, allv = B.prefix_coverage(sc, ev, seq)
        assert len(allp) == len(seq) == 41
        for k in (0, 7, 20, 40):      # same as the whole-route algorithm (highest level over the first k + 1 viewpoints)
            lv = [vis_level(sc, o, seq[:k + 1]) for o in ev]
            assert bool(allp[k]) == all(x >= 1 for x in lv) and bool(allv[k]) == all(x == 2 for x in lv)
        assert all(not a or b for a, b in zip(allp[:-1], allp[1:])) and all(not a or b for a, b in zip(allv, allp))      # monotone; strictly visible ⊂ at least partial
        views, fars = B.prefix_views_far(sc, seq)
        assert views[0] == 1 and fars[0] == 0 and views[-1] == len(set(seq)) and fars == sorted(fars)
        assert abs(fars[-1] - max(float(np.linalg.norm(np.asarray(sc.pose[x][0][:2]) - np.asarray(sc.pose[seq[0]][0][:2]))) for x in seq)) < 1e-9
        n += 1
    assert n == 12


def test_aggregate_pools_counts():
    xs = [dict(steps=40, forced=1, views=10, far_m=2.0, revisits=1, arrivals=4, turns=3, actions=6, cannot_move=2, invalid=0, all_partial=1, all_visible=0,
               rw_same_steps_all_partial=1, rw_same_steps_all_visible=1, dataset="arkitscenes"),
          dict(steps=10, forced=0, views=4, far_m=1.0, revisits=3, arrivals=6, turns=1, actions=4, cannot_move=0, invalid=1, all_partial=0, all_visible=0,
               rw_same_steps_all_partial=0, rw_same_steps_all_visible=0, dataset="scannetpp")]
    a = B.aggregate(xs)
    assert a["n_episodes"] == 2 and a["revisit_rate"] == 0.4 and a["turn_share"] == 0.4 and a["frac_full_40"] == 0.5 and a["steps_median"] == 25.0
    assert a["cannot_move_mean"] == 1.0 and a["frac_all_partial"] == 0.5
    t = B.summary_table({"m": xs}, ["m"])
    assert t["ARKitScenes"]["m"]["n_episodes"] == 1 and t["Merged"]["m"]["n_episodes"] == 2
    assert B.summary_table({"m": xs}, ["m"], keep=lambda x: x["steps"] < 40)["Merged"]["m"]["n_episodes"] == 1
