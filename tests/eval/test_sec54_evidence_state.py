"""Definition tests for the Section 5.4 evidence-state grouping (sec54_evidence_state.py): no run results are read."""
import json

from actspace_eval import sec54_evidence_state as S


def test_summarize_shares_and_selective():
    q = lambda st, score, ans: {"state": st, "score": score, "cannot_determine": int(ans == "cannot_determine"), "missing": int(ans is None)}   # noqa: E731
    xs = [q("in_ctx", 1.0, "A"), q("in_ctx", 0.0, "cannot_determine"), q("in_ctx", 0.0, None), q("seen_lost", 0.5, 2.0), q("unseen", 0.0, "cannot_determine")]
    s = S.summarize(xs)
    assert [s[k]["n"] for k, _ in S.STATES] == [3, 1, 1] and abs(sum(s[k]["share"] for k, _ in S.STATES) - 1) < 1e-6
    a = s["in_ctx"]
    assert a["share"] == 0.6 and abs(a["score"] - 1 / 3) < 1e-6 and abs(a["cd_rate"] - 1 / 3) < 1e-6
    assert a["selective"] == 0.5 and a["selective_answered"] == 1.0 and a["n_answered"] == 1 and abs(a["missing_rate"] - 1 / 3) < 1e-6   # original definition: denominator includes empty answers
    assert s["unseen"]["selective"] is None and s["unseen"]["cd_rate"] == 1.0


def test_final_context_takes_last_call(tmp_path):
    img = lambda f: {"type": "image", "photo_id": "p", "frame_id": f}   # noqa: E731
    call = lambda g, t, frames: {"type": "call", "tag": {"episode": g, "t": t}, "request": {"messages": [   # noqa: E731
        {"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "x"}, *map(img, frames)]}, {"role": "assistant", "content": "{}"}]}}
    lines = [call("g1", 0, ["a"]), call("g1", 1, ["a", "b"]), {"type": "error", "tag": {"episode": "g1", "t": 2}}, call("g2", 0, ["c"]),
             call("g1", 0, ["z"]), call("g1", 1, ["a", "d"])]      # episode rerun after resuming: on a tag.t tie the later one in the file wins
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert S.final_context(tmp_path) == {"g1": ["a", "d"], "g2": ["c"]}


def test_state_of(monkeypatch):
    vis = {("o1", "n1"): 2, ("o1", "n2"): 0, ("o2", "n1"): 0, ("o2", "n2"): 1, ("o2", "n3"): 0, ("o1", "n3"): 0}
    monkeypatch.setattr(S, "vis_level", lambda scene, o, nodes: max([vis[(o, n)] for n in nodes], default=0))
    assert S.state_of(None, {"o1", "o2"}, ["n1", "n2", "n3"], ["n1", "n2"]) == "in_ctx"
    assert S.state_of(None, {"o1", "o2"}, ["n1", "n2", "n3"], ["n2", "n3"]) == "seen_lost"
    assert S.state_of(None, {"o1", "o2"}, ["n1", "n3"], ["n1", "n3"]) == "unseen"
    assert S.state_of(None, {"o1"}, ["n1"], []) == "seen_lost"
