"""Definition tests for Section 5.5 (sec55_abstention.py): no run results are read."""
import re

from actspace_eval import sec55_abstention as S


def test_guess_regex():
    yes = ["Q1 is a guess.", "Sofa not located; guessing its position.", "The shelf was never found, so Q2 is a guess.", "Fridge not seen; D is a guess.",
           "Chair and TV were never seen", "I couldn't find the lamp", "I haven't seen the window", "no view of the sink", "the oven is unseen", "did not see the table",
           "The TV wasn't visible from any viewpoint", "unable to locate the bed"]
    no = ["Table seen close, chair behind it against wall.", "Bookcase seen front-left; fireplace at heading -120.", "", "All objects located; distances estimated from poses."]
    assert all(S.GUESS_RE.search(t) for t in yes), [t for t in yes if not S.GUESS_RE.search(t)]
    assert not any(S.GUESS_RE.search(t) for t in no), [t for t in no if S.GUESS_RE.search(t)]


def test_pick_question_prefers_named_and_not_in_context():
    qs = [{"q": "Q1", "state": "in_ctx"}, {"q": "Q2", "state": "unseen"}, {"q": "Q3", "state": "seen_lost"}]
    assert S.pick_question({"notes": "Q3 is a guess", "reasoning_snippet": "", "questions": qs})[0]["q"] == "Q3"
    assert S.pick_question({"notes": "sofa never seen", "reasoning_snippet": "", "questions": qs})[0]["q"] == "Q2"
    assert S.pick_question({"notes": "Q1 is a guess", "reasoning_snippet": "", "questions": qs[:1]})[0]["q"] == "Q1"


def test_runs_and_answered():
    assert S.runs_of("haiku-5.5", "active_v2") == ["active_v2/all_v1_r0", "active_v2/all_v1_r1", "active_v2/all_v1_r2"]
    assert S.runs_of("haiku-5.5", "random_walk_40") == ["random_walk_40/all_v1_r0"] and S.runs_of("sol", "oracle_route") == ["oracle_route/all_r0"]
    assert S.runs_of("Qwen3.8-27B", "random_walk_40") == ["random_walk_40/full_s0"] and S.runs_of("Qwen3.8-27B", "oracle_route") == ["oracle_route/full_r0"]
    assert S.answered("A") and S.answered(2.5) and not S.answered(None) and not S.answered("cannot_determine")
    assert re.compile(S.GUESS_RE.pattern)


def test_consistency_summary_counts():
    base = dict(model="sol", setting="active_v2", run="all_r0", dataset="arkitscenes", type="direction_inverse", has_unanswerable=False,
                all_correct=False, all_correct_lenient=False)
    rows = [dict(base, scene_id="s1", all_answered=True, determinable=True, violated=True),
            dict(base, scene_id="s1", all_answered=True, determinable=True, violated=False, all_correct=True, all_correct_lenient=True),
            dict(base, scene_id="s2", all_answered=False, determinable=False, violated=False),
            dict(base, scene_id="s2", type="numeric_vs_comparison", all_answered=True, determinable=False, violated=False)]
    out = S.consistency_summary(rows, {"arkitscenes": {"s1", "s2"}, "scannetpp": {"s3"}})
    r = next(x for x in out if x["scope"] == "Merged" and x["model"] == "sol" and x["setting"] == "active_v2" and x["constraint_type"] == "all")
    assert (r["n_constraints"], r["n_all_answered"], r["n_tie"], r["n_determinable"], r["n_violated"], r["n_all_correct"]) == (4, 3, 1, 2, 1, 1)
    assert r["violation_rate"] == 0.5 and 0 <= r["violation_lo"] <= r["violation_hi"] <= 1 and r["n_all_correct_violated"] == 0
    e = next(x for x in out if x["scope"] == "ScanNet++" and x["model"] == "sol" and x["setting"] == "active_v2" and x["constraint_type"] == "all")
    assert e["n_constraints"] == 0 and e["violation_rate"] is None
