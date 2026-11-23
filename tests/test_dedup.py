"""Within-scene deduplication and quadrant rebalancing."""
from collections import Counter

import numpy as np

from actspace_build.step56_questions import SceneUsed, dedup_key, rebalance_quadrants


def q(t, slots, **kw):
    return {"template": t, "slots": slots, **kw}


def test_dedup_keys():
    assert dedup_key(q("T1", {"OBJ1": 1, "OBJ2": 2})) == dedup_key(q("T1", {"OBJ1": 2, "OBJ2": 1}))      # unordered object pair
    a = q("T2", {"OBJ1": 1}, variant="closest", option_objs=[2, 3, 4])
    b = q("T2", {"OBJ1": 1}, variant="closest", option_objs=[4, 3, 2])
    c = q("T2", {"OBJ1": 1}, variant="farthest", option_objs=[2, 3, 4])
    assert dedup_key(a) == dedup_key(b) and dedup_key(a) != dedup_key(c)                               # different direction counts as different
    assert dedup_key(q("T3", {"OBJ1": 1, "OBJ2": 2, "OBJ3": 3})) != dedup_key(q("T3", {"OBJ1": 2, "OBJ2": 1, "OBJ3": 3}))
    assert dedup_key(q("T5", {"OBJ1": 1, "OBJ2": 2}, option_objs=[3, 4])) == dedup_key(q("T5", {"OBJ1": 1, "OBJ2": 2}, option_objs=[5, 6]))


def test_scene_used_limits():
    u = SceneUsed(max_t4=2)
    t1 = q("T1", {"OBJ1": 1, "OBJ2": 2})
    assert u.ok(t1)
    u.add(t1)
    assert not u.ok(q("T1", {"OBJ1": 2, "OBJ2": 1}))
    t4 = q("T4", {"OBJ1": 7})
    u.add(t4)
    assert u.ok(t4)
    u.add(t4)
    assert not u.ok(t4)                      # T4 on the same object at most 2 times
    assert not u.ok_all([q("T1", {"OBJ1": 3, "OBJ2": 4}), q("T1", {"OBJ1": 4, "OBJ2": 3})])  # duplicate within the same batch


class _Log:
    def __init__(self):
        self.rows = []

    def add(self, *a, **k):
        self.rows.append(a)


def test_rebalance_removes_minimum_from_overrepresented():
    quads = ["front-left"] * 2 + ["front-right"] * 3 + ["back-left"] * 3 + ["back-right"] * 3
    groups = []
    for k, v in enumerate(quads):
        groups.append({"scene_id": "s", "constraint": {"n": 1},
                       "questions": [q("T1", {"OBJ1": 0, "OBJ2": k}), q("T3", {"OBJ1": 0, "OBJ2": 1, "OBJ3": k}, answer_value=["front-left", "front-right", "back-left", "back-right"][k % 4]),
                                     q("T4", {"OBJ1": k}, answer_value=v), q("T2", {"OBJ1": k}, variant="closest", option_objs=[1, 2, 3])]})
    # T3 roughly uniform; in T4 front-left is under-represented (2 / 11)
    log = _Log()
    removed = rebalance_quadrants(groups, {"questions_per_group": [3, 5]}, 0.05, {"s": log}, np.random.default_rng(0))
    c = Counter(x["answer_value"] for g in groups for x in g["questions"] if x["template"] == "T4")
    n = sum(c.values())
    assert all(abs(c[k] / n - 0.25) <= 0.05 for k in c)
    assert all(len(g["questions"]) >= 3 for g in groups)
    assert "T4:front-right" in removed or "T4:back-left" in removed or "T4:back-right" in removed


def test_rebalance_skips_infeasible():
    groups = [{"scene_id": "s", "constraint": {"n": 0},
               "questions": [q("T1", {"OBJ1": 0, "OBJ2": k}), q("T2", {"OBJ1": k}, variant="closest", option_objs=[1, 2, 3]),
                             q("T3", {"OBJ1": 0, "OBJ2": 1, "OBJ3": k}, answer_value="back-left"), q("T5", {"OBJ1": k, "OBJ2": 9})]}
              for k in range(6)]
    removed = rebalance_quadrants(groups, {"questions_per_group": [3, 5]}, 0.05, {"s": _Log()}, np.random.default_rng(0))
    assert removed == {"T3:infeasible_empty_quadrant": 1} and all(len(g["questions"]) == 4 for g in groups)


def test_size_reference_requires_covisible_node():
    """The two instances of larger / smaller must be visible together in at least one node; otherwise not referable."""
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V
    from actspace_build.step56_questions import covisible
    state = np.array([[V, I, V], [I, V, P], [V, V, V]], np.uint8)   # 0/1 never visible together (partial does not count); 0 and 2 visible together
    assert not covisible(state, [0, 1]) and covisible(state, [0, 2]) and covisible(state, [1, 2])


def test_evidence_linked():
    """Evidence objects must be connected with "visible together at the same node" as edges; linking through intermediate objects is allowed."""
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V
    from actspace_build.step56_questions import evidence_linked
    state = np.array([[V, I, I],    # A
                      [V, V, I],    # B: visible together with A at node 0 and with C at node 1
                      [I, V, I],    # C
                      [I, I, V],    # D: never visible together with others
                      [I, P, V]], np.uint8)  # E: only partial at node 1
    vv = dict(mode="visible_visible")
    assert evidence_linked(state, [0, 1, 2], **vv)            # A–B–C linked
    assert not evidence_linked(state, [0, 2], **vv)           # A and C have no direct edge and no intermediate object
    assert not evidence_linked(state, [0, 1, 3], **vv)        # D isolated
    assert evidence_linked(state, [3, 4], **vv)               # visible together at node 2
    assert not evidence_linked(state, [1, 2], keep=np.array([True, False, True]), **vv)  # disconnected after masking node 1


def test_evidence_link_modes():
    """Relaxed modes: visible_partial = an edge if one is visible and the other at least partial; partial_partial = both at least partial."""
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V
    from actspace_build.step56_questions import evidence_linked
    state = np.array([[V, P, I],    # A
                      [P, P, I],    # B: partial at node 0 (A visible), partial at node 1 (A partial)
                      [I, P, I]], np.uint8)  # C: partial only at node 1
    assert not evidence_linked(state, [0, 1], mode="visible_visible")
    assert evidence_linked(state, [0, 1], mode="visible_partial")       # node 0: A visible + B partial
    assert not evidence_linked(state, [1, 2], mode="visible_partial")   # at node 1 both are only partial
    assert evidence_linked(state, [1, 2], mode="partial_partial")
    assert not evidence_linked(state, [0, 1], keep=np.array([False, True, True]), mode="visible_partial")


def test_greedy_link_views():
    """Link views: greedily pick the fewest co-visibility nodes that connect the evidence graph; ties prefer oracle / initial views."""
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V
    from actspace_build.step56_questions import greedy_link_views
    state = np.array([[V, V, I, V],    # A
                      [P, I, V, V],    # B
                      [I, P, V, V]], np.uint8)  # C
    # Node 3 connects all three at once and is minimal
    assert greedy_link_views(state, [0, 1, 2], mode="visible_visible") == [3]
    # Without node 3: node 0 links A–B, node 1 links A–C (or node 2 links B–C), two nodes
    keep = np.array([True, True, True, False])
    lv = greedy_link_views(state, [0, 1, 2], keep, mode="visible_partial")
    assert len(lv) == 2 and 3 not in lv
    # Ties prefer nodes in prefer
    assert greedy_link_views(state, [1, 2], mode="visible_visible") == [2]            # nodes 2 and 3 tie; take the smaller index
    assert greedy_link_views(state, [1, 2], mode="visible_visible", prefer=[3]) == [3]
    assert greedy_link_views(state, [0], mode="visible_partial") == []
    assert greedy_link_views(state, [0, 1, 2], np.array([True, False, False, False]), mode="visible_visible") is None


def test_step8_fold_fallback():
    """With too few scenes, 5 folds fall back to 3 folds or leave-one-scene-out."""
    from actspace_build.step8_debias import effective_folds, fold_splits
    assert effective_folds(7, 5) == 5 and effective_folds(4, 5) == 3 and effective_folds(2, 5) == 2
    groups = np.array(["s1", "s1", "s2", "s3", "s4", "s4"])
    sp = fold_splits(groups, 5)
    assert len(sp) == 3 and all(not set(groups[tr]) & set(groups[te]) for tr, te in sp)


def test_remove_questions_constraint_loss():
    """Default: a deleted question in a constraint → constraint invalid → whole group dropped; with keep_group_on_constraint_loss only the constraint becomes invalid.
    Groups generated without constraints are not dropped for "having no constraint"."""
    from collections import defaultdict
    from actspace_build.step8_debias import remove_questions

    class Log:
        def add(self, *a, **k):
            pass

    def grp(con):
        qs = [{"qid": f"g_q{i}", "template": t} for i, t in enumerate(["T1", "T2", "T1", "T3"])]
        return {"group_id": "g", "scene_id": "s", "questions": qs,
                "constraints": [{"type": "numeric_vs_comparison", "qids": ["g_q0", "g_q1"]}] if con else []}
    gc = {"questions_per_group": [3, 5]}
    drops = defaultdict(Log)
    for con, keep, survives in [(True, False, False), (True, True, True), (False, False, True)]:
        groups = [grp(con)]
        remove_questions(groups, {(0, 0)}, {(0, 0): "BIAS_PRUNED"}, drops, gc, keep)
        assert (len(groups) == 1) == survives, (con, keep)
        if survives:
            assert len(groups[0]["questions"]) == 3 and groups[0]["constraints"] == []


def test_covisible_modes():
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V
    from actspace_build.step56_questions import covisible
    state = np.array([[V, P], [P, P]], np.uint8)
    assert not covisible(state, [0, 1]) and covisible(state, [0, 1], "visible_partial") and covisible(state, [0, 1], "partial_partial")
    assert not covisible(np.array([[P], [P]], np.uint8), [0, 1], "visible_partial")


def test_path_view_indices():
    from actspace_build.step56_questions import path_view_indices
    assert path_view_indices(0) == [] and path_view_indices(3) == [0, 1, 2]
    assert path_view_indices(10) == [0, 3, 6, 9] and path_view_indices(5)[-1] == 4 and len(path_view_indices(5)) == 4


def test_t2_dedup_by_answer_option():
    """ScanNet++: T2 can be deduplicated by reference + direction + correct answer; the default is still the option set (ARKit)."""
    a = q("T2", {"OBJ1": 1}, option_objs=[2, 3, 4], variant="farthest", answer_obj=2)
    b = q("T2", {"OBJ1": 1}, option_objs=[2, 5, 6], variant="farthest", answer_obj=2)
    assert dedup_key(a) != dedup_key(b) and dedup_key(a, True) == dedup_key(b, True)
    u = SceneUsed(max_t4=2, t2_by_answer=True)
    u.add(a)
    assert not u.ok(b) and not u.copy().ok(b)
    assert SceneUsed(max_t4=2).ok(b)
