"""Unit tests for the merge dry run (actspace_build.merge_dryrun): standing-point horizontal separation, id prefixes, statistics
and the simulated quadrant balancing."""
import numpy as np

from actspace_build import merge_dryrun as M


def _q(t, slots, qid="g_000000_q1", options=None, option_objs=None, answer="A", label="explore", answer_type="multiple_choice"):
    q = {"template": t, "slots": slots, "qid": qid, "evidence_label": label, "answer_type": answer_type, "answer": answer}
    if options:
        q["options"] = options
    if option_objs is not None:
        q["option_objs"] = option_objs
    return q


def test_separation_standing_point_only():
    centers = {1: np.array([0.0, 0.0]), 2: np.array([0.3, 0.0]), 3: np.array([3.0, 0.0]), 4: np.array([0.0, 0.45]), 5: np.array([0.31, 0.0])}
    # T3: standing 1, facing 3, queried 2 -> standing point to queried object 0.3 m
    d, pair = M.separation(_q("T3", {"OBJ1": 1, "OBJ2": 3, "OBJ3": 2}), centers, None)
    assert abs(d - 0.3) < 1e-9 and pair == "1–2"
    # pairwise distances not involving the standing point are ignored (2 and 5 are 0.01 m apart, but both >= 0.3 from the standing point)
    d, _ = M.separation(_q("T3", {"OBJ1": 3, "OBJ2": 2, "OBJ3": 5}), centers, None)
    assert d > 2.6
    # T5: standing 1, facing 3, options include 4 (0.45 m)
    d, pair = M.separation(_q("T5", {"OBJ1": 1, "OBJ2": 3}, option_objs=[3, 4]), centers, None)
    assert abs(d - 0.45) < 1e-9 and pair == "1–4"
    # T4: object vs. start camera
    d, pair = M.separation(_q("T4", {"OBJ1": 3}), centers, np.array([2.6, 0.0]))
    assert abs(d - 0.4) < 1e-9 and pair.startswith("START camera")
    assert M.separation(_q("T1", {"OBJ1": 1, "OBJ2": 2}), centers, None) is None
    assert M.separation(_q("T2", {"OBJ1": 1}), centers, None) is None


def test_tag_groups_prefixes_ids_and_constraints():
    g = {"group_id": "g_000003", "scene_id": "scannetpp_abc", "questions": [_q("T1", {"OBJ1": 1, "OBJ2": 2}, qid="g_000003_q1")],
         "constraints": [{"type": "numeric_vs_comparison", "qids": ["g_000003_q1"]}]}
    (t,) = M.tag_groups([g], "scannetpp")
    assert t["group_id"] == "scannetpp_g_000003" and t["source_group_id"] == "g_000003" and t["source_dataset"] == "scannetpp"
    assert t["questions"][0]["qid"] == "scannetpp_g_000003_q1" and t["questions"][0]["source_qid"] == "g_000003_q1"
    assert t["constraints"][0]["qids"] == ["scannetpp_g_000003_q1"]
    assert g["group_id"] == "g_000003"   # input is not modified


def _quad(qid, ans_text, t="T3", label="explore", bias=0.0):
    opts = ["front-left", "front-right", "back-left", "back-right"]
    q = _q(t, {"OBJ1": 1}, qid=qid, options=[f"{L}. {o}" for L, o in zip("ABCD", opts)], answer="ABCD"[opts.index(ans_text)], label=label)
    q["option_values"] = opts
    q["bias_score"] = bias
    return q


def test_bias_stats_and_quadrant_simulation():
    # one scene, two groups: T3 front-left is over-represented (6 / 12); T2 only satisfies the group conditions
    qs1 = [_quad(f"arkit_g_000001_q{i}", "front-left", bias=i / 10) for i in range(6)]
    qs2 = [_quad(f"arkit_g_000002_q{i}", a) for i, a in enumerate(["front-right", "front-right", "back-left", "back-left", "back-right", "back-right"])]
    t2 = lambda qid: _q("T2", {"OBJ1": 1}, qid=qid, options=["A. x", "B. y"], answer="A")
    groups = [{"group_id": "arkit_g_000001", "scene_id": "arkit_1", "questions": qs1[:5] + [t2("arkit_g_000001_q9")], "constraints": []},
              {"group_id": "arkit_g_000002", "scene_id": "arkit_1", "questions": qs2[:4] + [t2("arkit_g_000002_q9")], "constraints": []}]
    qb = M.quadrant_bias(groups)
    assert qb["T3"]["n"] == 9 and qb["T3"]["max_dev_pp"] > 5
    lb = M.letter_bias(groups)
    assert lb["T2|2"]["counts"] == {"A": 2} and lb["T2|2"]["max_dev_pp"] == 50.0
    cfg = {"questions": {"quadrant_balance_tol": 0.05}, "debias": {"t4_front_back_tol": 0.05},
           "groups": {"questions_per_group": [3, 5]}}
    sim = M.simulate_quadrant_rebalance(groups, cfg)
    assert sim["status"] == "ok" and sim["n_removed"] > 0
    assert all(k.startswith("arkitscenes|T3") or "GROUP_INCOMPLETE" in k for k in sim["removed_by"])
    # simulation only: input unchanged
    assert len(groups[0]["questions"]) == 6


def test_rebalance_letters_by_dataset_strict_both_levels():
    """merge_build.rebalance_letters_by_dataset: within each dataset and over the merged set, letter counts per (template, number of
    options) differ by <= 1; only the option order changes."""
    from collections import Counter
    from actspace_build.merge_build import rebalance_letters_by_dataset
    groups = []
    for ds, sid, n in (("arkitscenes", "arkit_1", 5), ("scannetpp", "scannetpp_a", 7)):
        qs = []
        for i in range(n):
            q = _q("T5", {"OBJ1": 1, "OBJ2": 2}, qid=f"{ds}_q{i}", options=["A. x", "B. y", "C. z", "D. w"], option_objs=[10, 11, 12, 13], answer="A")
            qs.append(q)
        groups.append({"group_id": f"{ds}_g", "scene_id": sid, "source_dataset": ds, "questions": qs, "constraints": []})
    rebalance_letters_by_dataset(groups, 7)
    tot = Counter()
    for g in groups:
        c = Counter(q["answer"] for q in g["questions"])
        assert max(c.get(L, 0) for L in "ABCD") - min(c.get(L, 0) for L in "ABCD") <= 1
        tot.update(c)
        for q in g["questions"]:   # correct answer content unchanged (originally always x / 10)
            k = "ABCD".index(q["answer"])
            assert q["option_objs"][k] == 10 and q["options"][k].endswith(". x") and sorted(q["option_objs"]) == [10, 11, 12, 13]
    assert max(tot.values()) - min(tot.get(L, 0) for L in "ABCD") <= 1


def test_rebuild_frames_rotation_matches_pipeline():
    """Upright rotation in scripts/rebuild_frames.py matches the build / evaluation code (geometry.rotate_image_cw); ScanNet++ frame index parsing."""
    import importlib.util
    from pathlib import Path
    from actspace_build.geometry import rotate_image_cw
    from actspace_build.merge_release import video_frame_index
    spec = importlib.util.spec_from_file_location("rebuild_frames", Path(__file__).resolve().parents[1] / "scripts" / "rebuild_frames.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    for q in range(5):
        assert np.array_equal(rf.rotate_cw(img, q), rotate_image_cw(img, q))
    assert rf.rotate_cw(img, 1).shape == (6, 4, 3)
    assert video_frame_index("frame_009834") == 9834 and video_frame_index("41069021_305.361.png") is None
