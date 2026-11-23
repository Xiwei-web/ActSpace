"""oracle_views / oracle_steps and evidence labels."""
from types import SimpleNamespace

import numpy as np

from actspace_build.step56_questions import evidence_label, oracle
from actspace_build.step4_graph_visibility import STATE_INVISIBLE as I, STATE_PARTIAL as P, STATE_VISIBLE as V


def line_scene(state):
    """Bidirectional chain of 5 nodes 0-1-2-3-4; state[object, node]."""
    state = np.asarray(state, dtype=np.uint8)
    n = state.shape[1]
    adj = [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]
    return SimpleNamespace(nodes=[f"f{i}" for i in range(n)], adj=adj, state=state, row={10: 0, 11: 1, 12: 2},
                           st=lambda o, v: int(state[{10: 0, 11: 1, 12: 2}[o], v]),
                           visible_anywhere=lambda o: bool((state[{10: 0, 11: 1, 12: 2}[o]] == V).any()))


def test_min_views_take_priority_over_steps():
    # Object 10 visible at nodes 2, 4; object 11 visible at nodes 1, 4. Observing {1,2} needs only 2 steps but 2 observations;
    # observing only node 4 needs 4 steps but 1 observation → oracle_views=1, oracle_steps=4.
    sc = line_scene([[I, I, V, I, V], [I, V, I, I, V], [I, I, I, I, I]])
    k, s, observed, path = oracle(sc, [10, 11], 0)
    assert (k, s, observed) == (1, 4, [4])
    assert path == [0, 1, 2, 3, 4]


def test_objects_visible_at_p0_need_no_view():
    sc = line_scene([[V, I, I, I, I], [I, I, I, V, I], [I, I, I, I, I]])
    assert oracle(sc, [10, 11], 0)[:3] == (1, 3, [3])
    assert oracle(sc, [10], 0)[:2] == (0, 0)


def test_evidence_labels():
    sc = line_scene([[V, I, I, I, I], [I, P, I, V, I], [I, P, I, I, I]])
    assert evidence_label(sc, [10], 0) == "init"
    assert evidence_label(sc, [10, 11], 0) == "explore"
    assert evidence_label(sc, [10, 12], 0) is None        # object 12 is only partial → AMBIGUOUS_EVIDENCE
    sc2 = line_scene([[V, I, I, I, I], [I, I, I, V, I], [I, I, I, I, I]])
    assert evidence_label(sc2, [10, 12], 0) is None                                   # ARKit: never seen → AMBIGUOUS
    assert evidence_label(sc2, [10, 12], 0, never_visible_is_ambiguous=False) == "unanswerable"  # original rule


def test_oracle_on_masked_graph():
    """Masking: the whole group shares the masked view graph; the oracle may observe and move only through kept nodes."""
    # Object 10 visible at nodes 1, 4; after masking node 1, it can only be observed at node 4
    sc = line_scene([[I, V, I, I, V], [I, I, I, I, I], [I, I, I, I, I]])
    assert oracle(sc, [10], 0)[:3] == (1, 1, [1])
    keep = np.array([True, False, True, True, True])
    # Chain 0-1-2 is cut, 0 cannot reach 4 → no reachable path
    assert oracle(sc, [10], 0, keep)[1] is None
    sc2 = line_scene([[I, I, V, I, V], [I, I, I, I, I], [I, I, I, I, I]])
    keep2 = np.array([True, True, False, True, True])
    assert oracle(sc2, [10], 0)[:3] == (1, 2, [2])
    assert oracle(sc2, [10], 0, keep2)[1] is None      # masking node 2 also breaks the chain
    keep3 = np.array([True, True, True, True, False])
    assert oracle(sc2, [10], 0, keep3)[:3] == (1, 2, [2])  # masking 4 has no effect
