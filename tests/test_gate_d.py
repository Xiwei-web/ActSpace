"""GATE D annotation collection (two-phase criteria)."""
import json

from actspace_build.gate_d import CANNOT, load_records, summarize

HC = {"min_review_correct_frac": 0.95, "max_ambiguous_frac": 0.05, "annotators_per_group": 2}
GROUPS = {"g1": {"group_id": "g1", "questions": [
    {"qid": "g1_q1", "template": "T1", "answer_type": "numeric", "answer": 2.0, "answer_raw": 2.0, "evidence_label": "explore", "text": "t1"},
    {"qid": "g1_q2", "template": "T3", "answer_type": "multiple_choice", "answer": "B", "evidence_label": "explore", "text": "t3"},
    {"qid": "g1_q3", "template": "T4", "answer_type": "multiple_choice", "answer": "A", "evidence_label": "unanswerable", "text": "t4"}]}}


def rec(ann, a1, a2, a3, verdicts, amb=False, phase="review", saved="2026-11-01T00:00:00"):
    blind = {"g1_q1": {"answer": a1}, "g1_q2": {"answer": a2, "ambiguous": amb}, "g1_q3": {"answer": a3}}
    review = {q: {"verdict": v} for q, v in zip(["g1_q1", "g1_q2", "g1_q3"], verdicts)} if phase == "review" else None
    return {"annotator": ann, "group_id": "g1", "phase": phase, "blind": blind, "review": review, "saved_at": saved}


def test_summary_metrics():
    recs = [rec("a", 2.0, "B", CANNOT, ["correct", "correct", "correct"]),
            rec("b", 3.0, "C", "A", ["correct", "incorrect", "unsure"], amb=True)]
    m = summarize(GROUPS, recs, HC)
    assert m["n_judged"] == 4 and abs(m["review_correct_frac"] - 0.75) < 1e-9 and not m["pass_correct"]  # answerable questions only (q1, q2)
    assert abs(m["ambiguous_or_unannotated_frac"] - 0.25) < 1e-9 and not m["pass"]
    # numeric MRA: 2.0 equals the ground truth -> 1.0; 3.0 has relative error 0.5, which must be < 1 - t (t = 0.50...0.95,
    # i.e. < 0.50...0.05) -> no threshold met -> 0
    assert m["perf"][("T1", "answerable")][0] == 0.5
    assert m["perf"][("T4", "unanswerable")][0] == 0.5          # only "cannot be determined" is correct
    assert m["perf"][("T3", "answerable")][0] == 0.5


def test_incomplete_and_dedup(tmp_path):
    old = rec("a", 2.0, "B", CANNOT, [None, None, None], phase="blind", saved="2026-11-01T00:00:00")
    new = rec("a", 2.0, "B", CANNOT, ["correct", "correct", "correct"], saved="2026-11-02T00:00:00")
    (tmp_path / "x1.json").write_text(json.dumps(old))
    (tmp_path / "x2.json").write_text(json.dumps(new))
    recs = load_records([str(tmp_path / "x1.json"), str(tmp_path / "x2.json")])
    assert len(recs) == 1 and recs[0]["phase"] == "review"   # same (annotator, group): the last saved record wins
    m = summarize(GROUPS, recs, HC)
    assert m["pass"] and not m["incomplete_records"]
    m2 = summarize(GROUPS, [old], HC)
    assert m2["incomplete_records"] == [("a", "g1")] and not m2["pass"]


def test_label_wrong_not_counted_as_correct():
    """First criterion: "reviewed as correct and not flagged label wrong"."""
    r = rec("a", 2.0, "B", CANNOT, ["correct", "correct", "correct"])
    r["blind"]["g1_q2"]["label_wrong"] = True
    m = summarize(GROUPS, [r], HC)
    assert abs(m["review_correct_frac"] - 0.5) < 1e-9 and not m["pass_correct"]


def test_alloc_quota():
    from actspace_build.gate_d import alloc_quota
    a = alloc_quota({"x": 10, "y": 10}, 6, {"x": 2.0, "y": 1.0})
    assert sum(a.values()) == 6 and a["x"] == 4 and a["y"] == 2
    a = alloc_quota({"x": 1, "y": 30}, 10, {"x": 2.0, "y": 1.0})
    assert a == {"x": 1, "y": 9}                    # a small stratum is capped at its size
    assert sum(alloc_quota({"x": 3, "y": 2}, 10).values()) == 5   # fewer than n available: all included
