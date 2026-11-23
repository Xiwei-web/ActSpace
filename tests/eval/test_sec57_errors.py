"""Definition tests for Section 5.7 (sec57_errors.py): no run results are read."""
import numpy as np

from actspace_eval import sec57_errors as S


def test_quadrant_error():
    assert S.quadrant_error("front-left", "front-left") == "correct"
    assert S.quadrant_error("front-right", "front-left") == "mirror_lr" and S.quadrant_error("back-left", "front-left") == "flip_fb"
    assert S.quadrant_error("back-right", "front-left") == "diagonal" and S.quadrant_error(None, "front-left") is None and S.quadrant_error("", "front-left") is None


def test_question_fields(monkeypatch):
    t1 = {"template": "T1", "answer_raw": 2.0, "answer": 2.0}
    assert S.question_fields(None, t1, 3.0)["ratio"] == 1.5 and S.question_fields(None, t1, "cannot_determine")["ratio"] == "" and S.question_fields(None, t1, None)["pred_distance_m"] == ""
    t4 = {"template": "T4", "answer": "C", "options": ["A. x", "B. x", "C. x", "D. x"], "option_values": ["back-right", "back-left", "front-left", "front-right"], "answer_value": "front-left"}
    assert S.question_fields(None, t4, "D")["error_type"] == "mirror_lr" and S.question_fields(None, t4, "B")["error_type"] == "flip_fb"
    assert S.question_fields(None, t4, "A")["error_type"] == "diagonal" and S.question_fields(None, t4, "C")["error_type"] == "correct"
    assert S.question_fields(None, t4, "cannot_determine")["error_type"] == "" and S.question_fields(None, t4, "D")["pred_quadrant"] == "front-right"
    t5 = {"template": "T5", "answer": "B", "options": ["A. a", "B. b", "C. c"], "option_objs": [2, 3, 1], "answer_obj": 3, "margins": {"theta_deg": {"1": -20.5, "2": 100.0, "3": 21.3}}}
    assert S.question_fields(None, t5, "B")["error_type"] == "correct" and S.question_fields(None, t5, "C")["error_type"] == "mirror_lr"
    assert S.question_fields(None, t5, "A")["error_type"] == "flip_fb" and S.question_fields(None, t5, "A")["true_quadrant"] == "front-left"
    t5b = {**t5, "margins": {"theta_deg": {"1": -20.5, "2": 40.0, "3": 21.3}}}
    assert S.question_fields(None, t5b, "A")["error_type"] == "same_quadrant"
    monkeypatch.setattr(S, "scene_objects", lambda scene: {1: {"center": [0, 0, 0]}, 2: {"center": [1, 0, 0]}, 3: {"center": [3, 0, 0]}})
    t2 = {"template": "T2", "variant": "pair", "answer": "A", "options": ["A. a", "B. b"], "option_objs": [2, 3], "answer_obj": 2, "slots": {"OBJ1": 1}}
    assert S.question_fields(None, t2, "B")["t2_distance_gap_m"] == 2.0 and S.question_fields(None, t2, "A")["t2_distance_gap_m"] == 0.0


def test_scene_boot_quantiles():
    b = S.SceneBoot({"arkitscenes": {"s1", "s2", "s3"}, "scannetpp": {"s4"}}, n=200)
    items = [("s1", 1.0), ("s1", 2.0), ("s2", 3.0), ("s3", 4.0), ("s4", 5.0)]
    q = b.quantile_ci(items)
    assert q[50][0] == 3.0 and q[50][1] <= 3.0 <= q[50][2] and q[25][0] == float(np.percentile([1, 2, 3, 4, 5], 25)) and b.quantile_ci([]) is None


def test_pick_cases_rules():
    base = dict(setting="active_v2", answerable=1, run="all_v1_r0", scene_id="s", dataset="arkitscenes", cannot_determine=0, answered=1, score=0.0, error_type="", template="T4")
    rows = [dict(base, model="opus-5.5", group_id="g1", qid="g1_q1", evidence_state="unseen"),
            dict(base, model="opus-5.5", group_id="g2", qid="g2_q1", evidence_state="unseen"),                       # too many steps, not "early"
            dict(base, model="opus-5.5", group_id="g3", qid="g3_q1", evidence_state="seen_lost", cannot_determine=1, answered=0),
            dict(base, model="opus-5.5", group_id="g4", qid="g4_q1", evidence_state="in_ctx", error_type="mirror_lr"),
            dict(base, model="opus-5.5", group_id="g5", qid="g5_q1", evidence_state="in_ctx", error_type="diagonal")]  # diagonal does not count
    eps = {("opus-5.5", "all_v1_r0", f"g{i}"): {"steps": 30 if i == 2 else 10, "forced": None, "n_ev": {f"g{i}_q1": 1}} for i in range(1, 6)}
    got = {(c["situation"], c["group_id"]) for c in S.pick_cases(rows, eps)}
    assert got == {("early_answer", "g1"), ("evidence_left_context", "g3"), ("wrong_with_evidence", "g4")}
