"""Section 5.5: within-group consistency, answering despite knowingly not having seen the evidence, and intervals for abstention rates.
Reads only existing logs, per-question records, the sec54 per-question CSV and the frozen data; no model is called.

    python -m actspace_eval.sec55_abstention --out results/_reports/sec55

A. Within-group consistency: Active, random_walk_40 (seed 0) and oracle_route of the 17 models, 227 main-table groups. Constraints are the
   frozen data's per-group constraints
   (numeric_vs_comparison, direction_inverse), evaluated with score.constraint_holds (same semantics as the integrity check), not reimplemented.
   - Determinable: every question of the constraint was answered (not cannot_determine, not empty) and constraint_holds returns True / False.
     When the two numeric answers of a numeric_vs_comparison constraint are equal,
     constraint_holds returns None (original implementation); such "all answered but undecidable" constraints are counted separately (n_tie).
   - Violation rate = violations / determinable constraints; interval = score.RepBoot scene bootstrap (2000 resamples, stratified by dataset
     for the merged scope; haiku-5.5 has 3 Active runs, which are resampled as well).
   - All correct: every question of the constraint scores 1 (for numeric questions MRA = 1, relative error < 5%); a lenient version is also
     given: multiple-choice correct, numeric MRA > 0 (relative error < 50%).
B. Answering despite knowingly not having seen the evidence: Active of the 6 closed-source models (all main runs of each model). Keywords
   (GUESS_RE) are searched in the notes and reasoning summary of the answer step.
   A hit only means the text contains these words; episodes that "hit and gave an answer to an answerable question whose evidence was not
   fully seen, or to an unanswerable question" are counted separately (some episodes say the object was not seen and then answer cannot_determine).
   The 3 OpenAI models (sol, terra, luna) have no notes at the answer step and the API returns no reasoning, so they cannot be judged.
C. Abstention-rate intervals: cd_rate in the three sec54 evidence states, and the correct abstention rate on unanswerable questions, with the
   same scene bootstrap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import time
from pathlib import Path

from .data import FrozenData, dataset_of_scene
from .score import N_BOOT, RepBoot, constraint_holds, load_run
from .sec53_behavior import CLOSED, MODELS, OPEN, ROOT, RUNS, SCOPES
from .sec54_evidence_state import STATES

CANNOT = "cannot_determine"
TYPES = ("numeric_vs_comparison", "direction_inverse")
SETTINGS = ("active_v2", "random_walk_40", "oracle_route")
SEC54 = ROOT / "_reports" / "sec54" / "evidence_state_per_question.csv"
GUESS_RE = re.compile(r"\bguess(?:es|ed|ing|work)?\b|\b(?:not|never|wasn'?t|weren'?t|isn'?t|aren'?t) (?:been )?(?:seen|visible|found|located|spotted|observed|in view)\b"
                      r"|\b(?:didn'?t|did not|don'?t|do not|haven'?t|have not|hasn'?t|has not|never|couldn'?t|could not|can'?t|cannot|unable to) (?:\w+ )?(?:see|saw|seen|find|found|locate|located|spot|spotted)\b"
                      r"|\bno (?:clear |direct )?(?:view|sighting)\b|\bunseen\b|\bnot sighted\b|\bwithout (?:seeing|having seen)\b", re.I)


def runs_of(model: str, setting: str) -> list[str]:
    """Run names (relative to the model directory). Active uses all main runs; random_walk_40 uses seed 0; oracle_route has a single run."""
    base, act, rw = RUNS[model]
    if setting == "active_v2":
        return [f"active_v2/{r}" for r in act]
    if setting == "random_walk_40":
        return [f"random_walk_40/{rw[0]}"]
    return [f"oracle_route/{'full_r0' if model in OPEN else act[0]}"]


def answered(a) -> bool:
    return a is not None and a != CANNOT


# ---------------------------------------------------------------- A
def constraint_rows(data, bad, model: str, setting: str) -> list[dict]:
    """One row per constraint per run."""
    out = []
    for name in runs_of(model, setting):
        run = load_run(RUNS[model][0], name)
        ans, score = run["answers_by_qid"], {x["qid"]: x["score"] for x in run["per_question"]}
        for gid in {e["group_id"] for e in run["per_episode"]} - bad:
            g = data.groups[gid]
            qby = {q["qid"]: q for q in g["questions"]}
            for i, con in enumerate(g["constraints"]):
                qs = [qby[q] for q in con["qids"]]
                all_ans = all(answered(ans.get(q["qid"])) for q in qs)
                h = constraint_holds(con, qby, ans) if all_ans else None
                sc = [score.get(q["qid"], 0.0) for q in qs]
                out.append({"model": model, "setting": setting, "run": name.split("/")[1], "group_id": gid, "scene_id": g["scene_id"],
                            "dataset": dataset_of_scene(g["scene_id"]), "constraint": i, "type": con["type"],
                            "has_unanswerable": any(q["evidence_label"] == "unanswerable" for q in qs),
                            "all_answered": all_ans, "determinable": h is not None, "violated": h is False,
                            "all_correct": all(s == 1.0 for s in sc),
                            "all_correct_lenient": all((s > 0) if q["answer_type"] == "numeric" else (s == 1.0) for q, s in zip(qs, sc))})
    return out


def consistency_summary(rows: list, scenes: dict) -> list[dict]:
    out = []
    for scope, lab in SCOPES:
        b = RepBoot(scenes, scope=scope)
        for m in MODELS:
            for s in SETTINGS:
                for typ in ("all", *TYPES):
                    xs = [r for r in rows if r["model"] == m and r["setting"] == s and (scope is None or r["dataset"] == scope) and (typ == "all" or r["type"] == typ)]
                    det = [r for r in xs if r["determinable"]]
                    runs = list(dict.fromkeys(r["run"] for r in xs))
                    ci = b.mean_ci([[(r["scene_id"], float(r["violated"])) for r in det if r["run"] == run] for run in runs], f"{m}|{s}|{typ}") if det else None
                    out.append({"scope": lab, "model": m, "kind": "closed" if m in CLOSED else "open", "setting": s, "constraint_type": typ, "n_runs": len(runs),
                                "n_constraints": len(xs), "n_with_unanswerable_question": sum(r["has_unanswerable"] for r in xs),
                                "n_all_answered": sum(r["all_answered"] for r in xs), "n_tie": sum(r["all_answered"] and not r["determinable"] for r in xs),
                                "n_determinable": len(det), "n_violated": sum(r["violated"] for r in det),
                                "violation_rate": round(ci["mean"], 6) if ci else None, "violation_lo": round(ci["lo"], 6) if ci else None,
                                "violation_hi": round(ci["hi"], 6) if ci else None,
                                "n_all_correct": sum(r["all_correct"] for r in xs), "n_all_correct_violated": sum(r["all_correct"] and r["violated"] for r in xs),
                                "n_all_correct_lenient": sum(r["all_correct_lenient"] for r in xs),
                                "n_all_correct_lenient_violated": sum(r["all_correct_lenient"] and r["violated"] for r in xs),
                                "n_violated_not_all_correct": sum(r["violated"] and not r["all_correct"] for r in det)})
    return out


# ---------------------------------------------------------------- B
def answer_calls(run_dir: Path) -> dict:
    """{group_id: last call of the episode} (largest tag.t; ties go to the later one)."""
    last = {}
    with open(run_dir / "calls.jsonl") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            x = json.loads(line)
            if x.get("type") == "call" and x["tag"]["t"] >= last.get(x["tag"]["episode"], (-1,))[0]:
                last[x["tag"]["episode"]] = (x["tag"]["t"], {"response": x.get("response") or "", "reasoning": x.get("reasoning") or "",
                                                             "action": (x.get("parse") or {}).get("action")})
    return {g: v[1] for g, v in last.items()}


def snippet(text: str, m: re.Match, width: int = 160) -> str:
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("…" if a else "") + " ".join(text[a:b].split()) + ("…" if b < len(text) else "")


def guess_scan(data, bad, state_of: dict) -> tuple[list, list]:
    """→ (per-model statistics, hit episodes). state_of: {(model, run, qid): sec54 evidence state}."""
    stats, hits = [], []
    for m in CLOSED:
        base, runs, _ = RUNS[m]
        n = n_notes = n_reason = n_hit = n_hit_notes = n_hit_reason = n_hit_ans = n_ans = 0
        for run in runs:
            d = base / "active_v2" / run
            calls = answer_calls(d)
            summ = json.loads((d / "summary.json").read_text())
            pq = collections.defaultdict(dict)
            for x in summ["per_question"]:
                pq[x["group_id"]][x["qid"]] = x
            for l in open(d / "trajectories.jsonl"):
                if '"episode_start"' in l:
                    s = json.loads(l)
                elif '"episode_end"' in l:
                    end = json.loads(l)
                    gid = s["group_id"]
                    if gid in bad:
                        continue
                    c = calls.get(gid, {"response": "", "reasoning": ""})
                    notes = end.get("notes") or ""
                    n += 1
                    n_notes += bool(notes)
                    n_reason += bool(c["reasoning"])
                    mn, mr = GUESS_RE.search(notes), GUESS_RE.search(c["reasoning"])
                    n_hit_notes += bool(mn)
                    n_hit_reason += bool(mr)
                    g = data.groups[gid]
                    # Questions of this episode that are "answerable with evidence not fully seen" or "unanswerable" and received an answer (not cannot_determine, not empty)
                    blind = [q["qid"] for q in g["questions"] if q["qid"] in pq[gid] and answered(pq[gid][q["qid"]]["answer"]) and
                             (q["evidence_label"] == "unanswerable" or state_of.get((m, run, q["qid"])) == "unseen")]
                    n_ans += bool(blind)
                    if not (mn or mr):
                        continue
                    n_hit += 1
                    n_hit_ans += bool(blind)
                    pub = {v: k for k, v in s["question_ids"].items()}
                    qs = []
                    for q in g["questions"]:
                        x = pq[gid].get(q["qid"])
                        if x is None:
                            continue
                        qs.append({"q": pub[q["qid"]], "qid": q["qid"], "template": q["template"], "text": q["text"],
                                   "state": state_of.get((m, run, q["qid"]), "unanswerable" if q["evidence_label"] == "unanswerable" else "?"),
                                   "answer": x["answer"], "score": x["score"], "gold": x.get("gold")})
                    hits.append({"model": m, "run": run, "group_id": gid, "dataset": dataset_of_scene(g["scene_id"]), "steps": end["steps_used"],
                                 "notes": notes, "notes_hit": mn.group(0) if mn else "", "reasoning_snippet": snippet(c["reasoning"], mr) if mr else "",
                                 "reasoning_hit": mr.group(0) if mr else "", "questions": qs, "n_answered_without_evidence": len(blind)})
        stats.append({"model": m, "episodes": n, "with_notes": n_notes, "with_reasoning": n_reason, "hits": n_hit, "hits_notes": n_hit_notes,
                      "hits_reasoning": n_hit_reason, "hit_rate": round(n_hit / n, 6) if n else None, "has_text": bool(n_notes or n_reason),
                      "episodes_answered_without_evidence": n_ans, "hits_answered_without_evidence": n_hit_ans,
                      "hit_and_answered_rate": round(n_hit_ans / n, 6) if n else None})
    return stats, hits


def pick_question(hit: dict) -> tuple[dict, str]:
    """Which question to quote: a question id (Q1…) named in the notes / reasoning snippet, preferring ones not "in context"; if none is named, the first question whose evidence is not in context."""
    text = hit["notes"] + " " + hit["reasoning_snippet"]
    named = [q for q in hit["questions"] if re.search(rf"\b{q['q']}\b", text)]
    for pool, how in ((named, "question id named in the text"), (hit["questions"], "no question id named; first question whose evidence is not in context")):
        for q in pool:
            if q["state"] != "in_ctx":
                return q, how
    return (named or hit["questions"])[0], "question id named in the text" if named else "no question id named; first question"


STATE_LAB = {**dict(STATES), "unanswerable": "Unanswerable question", "?": "?"}


def write_samples(path: Path, stats: list, hits: list, k: int = 5):
    L = ["# Answering despite knowingly not having seen the evidence: quoted keyword hits", "",
         "Generated by `python -m actspace_eval.sec55_abstention`. Active of the 6 closed-source models, 227 main-table groups; keywords are searched in the `notes` and reasoning summary of the answer step.",
         "For each model, 5 hit episodes are sampled with a fixed seed (20261114), preferring episodes whose `notes` hit. A hit only means the text contains these words; it is not a human judgement.", "",
         f"Keywords (regex): `{GUESS_RE.pattern}`", ""]
    rng = random.Random(20261114)
    for st in stats:
        m = st["model"]
        L += [f"## {m}", ""]
        if not st["has_text"]:
            L += ["No `notes` at the answer step and the API returns no reasoning; cannot be judged.", ""]
            continue
        L += [f"Hits: {st['hits']} / {st['episodes']} episodes ({st['hit_rate']:.1%}); notes hits {st['hits_notes']} episodes, reasoning-summary hits {st['hits_reasoning']} episodes. "
              f"Hit and gave an answer to a question whose evidence was not fully seen (or an unanswerable question): {st['hits_answered_without_evidence']} episodes ({st['hit_and_answered_rate']:.1%}).", ""]
        mine = [h for h in hits if h["model"] == m]
        pool = [h for h in mine if h["notes_hit"]] or mine
        for h in rng.sample(pool, min(k, len(pool))):
            q, how = pick_question(h)
            ok = "correct" if q["score"] == 1 else ("wrong" if q["score"] == 0 else f"partial score {q['score']:.2f}")
            L += [f"### `{h['group_id']}` ({h['run']}, {h['steps']} steps)", "",
                  f"- **notes:** {h['notes'] or '(empty)'}"]
            if h["reasoning_snippet"]:
                L.append(f"- **Reasoning-summary snippet:** {h['reasoning_snippet']}")
            L += [f"- **Question:** {q['q']} ({q['template']}) {q['text']}",
                  f"- **Evidence state:** {STATE_LAB[q['state']]}; **model answer:** `{q['answer']}`; **gold answer:** `{q['gold']}`; **{ok}** ({how})", ""]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- C
def abstention_ci(data, bad, scenes: dict, q54: list) -> list[dict]:
    out = []
    unans = collections.defaultdict(list)       # model → [(run, scene, dataset, abstained)]
    for m in MODELS:
        for name in runs_of(m, "active_v2"):
            for x in load_run(RUNS[m][0], name)["per_question"]:
                if x["group_id"] not in bad and not x["answerable"]:
                    unans[m].append((name, x["scene_id"], dataset_of_scene(x["scene_id"]), float(x["answer"] == CANNOT)))
    for scope, lab in SCOPES:
        b = RepBoot(scenes, scope=scope)
        for m in MODELS:
            mine = [x for x in q54 if x["model"] == m and (scope is None or x["dataset"] == scope)]
            runs = list(dict.fromkeys(x["run"] for x in mine))
            for st, name in (*STATES, ("all_answerable", "All answerable questions")):
                xs = [x for x in mine if st == "all_answerable" or x["state"] == st]
                ci = b.mean_ci([[(x["scene_id"], float(x["cannot_determine"])) for x in xs if x["run"] == r] for r in runs], f"{m}|{st}") if xs else None
                out.append({"scope": lab, "model": m, "kind": "closed" if m in CLOSED else "open", "group": st, "group_label": name, "metric": "cd_rate",
                            "n_questions": len(xs), "n_cannot_determine": sum(int(x["cannot_determine"]) for x in xs),
                            "rate": round(ci["mean"], 6) if ci else None, "lo": round(ci["lo"], 6) if ci else None, "hi": round(ci["hi"], 6) if ci else None})
            u = [x for x in unans[m] if scope is None or x[2] == scope]
            runs_u = list(dict.fromkeys(x[0] for x in u))
            ci = b.mean_ci([[(x[1], x[3]) for x in u if x[0] == r] for r in runs_u], f"{m}|unanswerable") if u else None
            out.append({"scope": lab, "model": m, "kind": "closed" if m in CLOSED else "open", "group": "unanswerable", "group_label": "Unanswerable questions", "metric": "correct_refusal_rate",
                        "n_questions": len(u), "n_cannot_determine": int(sum(x[3] for x in u)),
                        "rate": round(ci["mean"], 6) if ci else None, "lo": round(ci["lo"], 6) if ci else None, "hi": round(ci["hi"], 6) if ci else None})
    return out


def write_csv(path: Path, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    assert N_BOOT == 2000
    data = FrozenData(verify=True)
    bad = set(data.ecfg["env"].get("budget_infeasible") or [])
    scenes = collections.defaultdict(set)
    n_con = collections.Counter()
    for gid, g in data.groups.items():
        if gid not in bad:
            scenes[dataset_of_scene(g["scene_id"])].add(g["scene_id"])
            for c in g["constraints"]:
                n_con[c["type"]] += 1
    unknown = set(n_con) - set(TYPES)
    print(f"main-table constraints: {dict(n_con)}; other types: {sorted(unknown)}", flush=True)
    # A
    rows = []
    for m in MODELS:
        for s in SETTINGS:
            rows += constraint_rows(data, bad, m, s)
        print(f"A {m}: {time.time() - t0:.0f} s", flush=True)
    summary = consistency_summary(rows, scenes)
    write_csv(out / "consistency_summary.csv", summary)
    write_csv(out / "consistency_per_constraint.csv", rows)
    # B
    q54 = list(csv.DictReader(open(SEC54)))
    for x in q54:
        x["cannot_determine"] = int(x["cannot_determine"])
    stats, hits = guess_scan(data, bad, {(x["model"], x["run"], x["qid"]): x["state"] for x in q54})
    write_samples(out / "guess_notes_samples.md", stats, hits)
    write_csv(out / "guess_hits_per_episode.csv", [{k: v for k, v in h.items() if k != "questions"} for h in hits])
    # Questions in each evidence state within hit episodes: answer rate and score
    hit_keys = {(h["model"], h["run"], h["group_id"]) for h in hits}
    by = []
    for m in CLOSED:
        for flag, lab in ((True, "hit episodes"), (False, "non-hit episodes")):
            for st, name in STATES:
                xs = [x for x in q54 if x["model"] == m and x["state"] == st and (((m, x["run"], x["group_id"]) in hit_keys) == flag)]
                by.append({"model": m, "episodes": lab, "state": st, "state_label": name, "n": len(xs),
                           "cd_rate": round(sum(x["cannot_determine"] for x in xs) / len(xs), 6) if xs else None,
                           "score": round(sum(float(x["score"]) for x in xs) / len(xs), 6) if xs else None})
    # C
    ab = abstention_ci(data, bad, scenes, q54)
    write_csv(out / "abstention_ci.csv", ab)
    info = {"data_version": data.version, "constraints_main": dict(n_con), "unknown_constraint_types": sorted(unknown), "consistency": summary,
            "guess_stats": stats, "guess_by_state": by, "abstention": ab, "seconds": round(time.time() - t0, 1),
            "constraints_with_unanswerable_question": sum(r["has_unanswerable"] for r in rows if r["model"] == MODELS[0] and r["setting"] == "oracle_route")}
    (out / "sec55_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    for st in stats:
        print("B", st, flush=True)
    print(f"wrote {out}: consistency_summary.csv ({len(summary)} rows), consistency_per_constraint.csv ({len(rows)} rows), guess_notes_samples.md, abstention_ci.csv ({len(ab)} rows), sec55_summary.json; {info['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
