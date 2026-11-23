"""Section 5.4: where the evidence is at answer time. Reads only existing trajectories, call logs, per-question records and the frozen data;
no model is called.

    python -m actspace_eval.sec54_evidence_state --out results/_reports/sec54

Definition ("is evidence that was seen still retained?"):
- Scope: Active of the 17 models, 227 main-table groups, answerable questions. The evidence of a question = its own evidence_objs.
- Route = the start plus the viewpoint reached at every step; context = the frames actually sent as images in the request of the episode's
  last call (the one with the largest tag.t).
- States: in_ctx = all evidence is at least partial in the context images; seen_lost = not the former, but all evidence is at least partial
  on the route (at least one piece is no longer in context);
  unseen = the evidence was not fully seen on the route.
- Per state: number of questions, share (of all answerable questions), score (mean per-question score), rate of answering cannot_determine,
  selective score.
  Selective score (original definition): mean score of questions not answered cannot_determine; questions with no answer (empty) stay in the
  denominator with score 0. Closed-source models have almost no empty answers;
  open-source models have many force-terminated episodes, so selective_answered (only questions that received an answer, same denominator as
  the selective score of the score tables) and missing_rate are reported as well.
- Truncation or context overflow: the broad definition of sec53_behavior (any call in the episode hit the output limit, or
  episode_end.forced = context_overflow).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

from .data import FrozenData, dataset_of_scene
from .providers.ablations import vis_level
from .sec53_behavior import ROOT, CLOSED, MODELS, RUNS, SCOPES, call_flags
from .trajectory import read_episodes

STATES = (("in_ctx", "Evidence still in context"), ("seen_lost", "Seen but no longer in context"), ("unseen", "Evidence not fully seen"))
CANNOT = "cannot_determine"
# Reference values (share %, score) for the self-check, from an earlier analysis of the same closed-source runs
EXPECT = {"sol": ((58.8, 0.877), (24.9, 0.542), (16.2, 0.377)), "terra": ((54.3, 0.776), (28.1, 0.258), (17.6, 0.207)),
          "luna": ((35.4, 0.644), (22.6, 0.302), (42.0, 0.248)), "opus-5.5": ((57.3, 0.872), (16.0, 0.832), (26.7, 0.595)),
          "haiku-5.5": ((42.3, 0.666), (6.4, 0.352), (51.2, 0.281)), "sonnet-5.5": ((55.5, 0.809), (4.6, 0.634), (39.9, 0.414))}


def final_context(run_dir: Path) -> dict:
    """{group_id: list of frame_ids actually sent as images in the last call} (the call with the largest tag.t; ties go to the later one in the file)."""
    last = {}
    with open(run_dir / "calls.jsonl") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            x = json.loads(line)
            if x.get("type") != "call":
                continue
            g, t = x["tag"]["episode"], x["tag"]["t"]
            if t >= last.get(g, (-1,))[0]:
                last[g] = (t, [p["frame_id"] for m in x["request"]["messages"] if isinstance(m.get("content"), list)
                               for p in m["content"] if isinstance(p, dict) and p.get("type") == "image"])
    return {g: v[1] for g, v in last.items()}


def state_of(scene, objs, route, ctx) -> str:
    if all(vis_level(scene, o, ctx) >= 1 for o in objs):
        return "in_ctx"
    return "seen_lost" if all(vis_level(scene, o, route) >= 1 for o in objs) else "unseen"


def model_rows(data, bad, model: str) -> tuple[list, dict]:
    base, runs, _ = RUNS[model]
    rows, chk = [], collections.Counter()
    for run in runs:
        d = base / "active_v2" / run
        ctx_of, flags = final_context(d), call_flags(d)
        pq = collections.defaultdict(dict)
        for x in json.loads((d / "summary.json").read_text())["per_question"]:
            pq[x["group_id"]][x["qid"]] = x
        for ep in read_episodes(d / "trajectories.jsonl"):
            s, end = ep[0], ep[-1]
            gid = s["group_id"]
            if gid in bad or end["type"] != "episode_end":
                continue
            g = data.groups[gid]
            sc = data.scene(g["scene_id"])
            route = [s["start_node"]] + [r["to"] for r in ep if r["type"] == "step"]
            if gid not in ctx_of:
                chk["episodes_without_call"] += 1
            ctx = ctx_of.get(gid, [])
            chk["ctx_not_subset_of_route"] += not set(ctx) <= set(route)
            fl = flags.get(gid, {"trunc": 0})
            overflow = end.get("forced") == "context_overflow"
            for q in g["questions"]:
                if q["evidence_label"] == "unanswerable":
                    continue
                x = pq[gid][q["qid"]]
                objs = {int(o) for o in q["evidence_objs"]}
                rows.append({"model": model, "kind": "closed" if model in CLOSED else "open", "run": run, "group_id": gid, "scene_id": g["scene_id"],
                             "dataset": dataset_of_scene(g["scene_id"]), "qid": q["qid"], "template": q["template"],
                             "state": state_of(sc, objs, route, ctx), "score": x["score"], "answer": "" if x["answer"] is None else x["answer"],
                             "cannot_determine": int(x["answer"] == CANNOT), "missing": int(x["answer"] is None),
                             "steps": end["steps_used"], "forced_reason": end.get("forced") or "", "views_route": len(set(route)), "views_ctx": len(set(ctx)),
                             "trunc_or_overflow": int(fl["trunc"] > 0 or overflow)})
    return rows, dict(chk)


def summarize(xs: list) -> dict:
    """{state: metrics}. The share denominator is all answerable questions in xs."""
    out, n_all = {}, len(xs)
    for st, _ in STATES:
        v = [x for x in xs if x["state"] == st]
        notcd = [x["score"] for x in v if not x["cannot_determine"]]
        ans = [x["score"] for x in v if not x["cannot_determine"] and not x["missing"]]
        out[st] = {"n": len(v), "n_all": n_all, "share": round(len(v) / n_all, 6) if n_all else None,
                   "score": round(float(np.mean([x["score"] for x in v])), 6) if v else None,
                   "cd_rate": round(float(np.mean([x["cannot_determine"] for x in v])), 6) if v else None,
                   "selective": round(float(np.mean(notcd)), 6) if notcd else None,
                   "selective_answered": round(float(np.mean(ans)), 6) if ans else None, "n_answered": len(ans),
                   "missing_rate": round(float(np.mean([x["missing"] for x in v])), 6) if v else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = FrozenData(verify=True)
    bad = set(data.ecfg["env"].get("budget_infeasible") or [])
    rows, checks = [], {}
    for m in MODELS:
        r, c = model_rows(data, bad, m)
        rows += r
        checks[m] = c
        print(f"{m}: {len(r)} questions, {c}, {time.time() - t0:.0f} s", flush=True)
    with open(out / "evidence_state_per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary, srows = {}, []
    for subset, keep in (("all", lambda x: True), ("no_trunc", lambda x: not x["trunc_or_overflow"])):
        for scope, lab in SCOPES:
            for m in MODELS:
                xs_all = [x for x in rows if x["model"] == m and (scope is None or x["dataset"] == scope)]
                xs = [x for x in xs_all if keep(x)]
                s = summarize(xs)
                removed_q = len(xs_all) - len(xs)
                removed_ep = len({(x["run"], x["group_id"]) for x in xs_all if not keep(x)})
                summary[f"{subset}|{lab}|{m}"] = {"states": s, "questions_removed": removed_q, "episodes_removed": removed_ep,
                                                  "episodes": len({(x["run"], x["group_id"]) for x in xs})}
                for st, name in STATES:
                    srows.append({"subset": subset, "scope": lab, "model": m, "state": st, "state_label": name, **s[st],
                                  "episodes_kept": len({(x["run"], x["group_id"]) for x in xs}), "episodes_removed": removed_ep, "questions_removed": removed_q})
    with open(out / "evidence_state_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    # Self-check: compare share / score with the reference values (reported with 1 and 3 decimals)
    sc = {}
    for m, exp in EXPECT.items():
        got = summary[f"all|Merged|{m}"]["states"]
        cells = []
        for (st, _), (share, score) in zip(STATES, exp):
            g = got[st]
            cells.append({"state": st, "expected": [share, score], "got": [round(100 * g["share"], 1), round(g["score"], 3)],
                          "ok": round(100 * g["share"], 1) == share and round(g["score"], 3) == score})
        sc[m] = {"cells": cells, "pass": all(c["ok"] for c in cells)}
        print("self-check", m, "passed" if sc[m]["pass"] else "FAILED", [(c["state"], c["expected"], c["got"]) for c in cells], flush=True)
    # Cross-check the truncation field against the sec53 per-episode CSV
    p = ROOT / "_reports" / "sec53" / "active_behavior_per_episode.csv"
    flag_check = None
    if p.exists():
        ref = {(r["model"], r["run"], r["group_id"]): int(r["trunc_or_overflow"]) for r in csv.DictReader(open(p))}
        mine = {(x["model"], x["run"], x["group_id"]): x["trunc_or_overflow"] for x in rows}
        flag_check = {"episodes": len(mine), "agree": sum(ref.get(k) == v for k, v in mine.items()), "in_sec53": len(ref)}
        print("truncation field vs. sec53 per-episode CSV:", flag_check, flush=True)
    info = {"data_version": data.version, "models": MODELS, "questions": len(rows), "summary": summary, "selfcheck": sc,
            "selfcheck_pass": all(v["pass"] for v in sc.values()), "checks": checks, "flag_check": flag_check, "seconds": round(time.time() - t0, 1),
            "answerable_per_run": len({x["qid"] for x in rows})}
    (out / "sec54_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"wrote {out}: evidence_state_per_question.csv ({len(rows)} rows), evidence_state_summary.csv ({len(srows)} rows), sec54_summary.json; all self-checks passed: {info['selfcheck_pass']}; {info['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
