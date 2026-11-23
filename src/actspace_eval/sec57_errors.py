"""Section 5.7: per-question predictions and gold answers, error types, case material. Reads only existing logs, per-question records,
the sec54 per-question CSV and the frozen data; no model is called.

    python -m actspace_eval.sec57_errors --out results/_reports/sec57

A. One row per question (17 models, Active and oracle_route, 227 main-table groups, including unanswerable questions):
   - Evidence state: Active uses the three sec54 states (in_ctx / seen_lost / unseen); oracle_route is recorded as in_ctx; unanswerable
     questions are recorded as unanswerable.
   - T1: predicted distance, true distance (answer_raw), ratio = predicted / true.
   - T3 / T4: predicted quadrant = quadrant of the chosen option, true quadrant = answer_value. Error types: correct / left-right mirror
     (only left/right differs) / front-back flip (only front/back differs) / diagonal (both differ).
   - T5 (standing at A facing B, which object is nearest on the left): predicted and true option; for the direction part the per-option
     azimuths stored with the question (margins.theta_deg) give the quadrants of the chosen and the correct object,
     same error types as above, plus "same-quadrant wrong choice" (chosen and correct objects are in the same quadrant; the error is in the
     distance ranking).
   - T2: predicted and true option; distance gap = |anchor-to-chosen-object distance − anchor-to-correct-object distance| (3D distance between
     object centres, same algorithm as the T1 gold answer), 0 when correct.
   Summary: for each model and setting, the median and quartiles of the T1 ratio, the shares of overestimates (ratio > 1) and underestimates
   (< 1); for direction questions the share of each error type among errors. Intervals are 95% percentile intervals from 2000 scene-bootstrap resamples
   (shares use score.RepBoot; median and quartiles use the same stratified scene resampling, pooling the episodes of multiple runs).
B. Case material: 3 candidate episodes for each of three situations, chosen by fixed rules (see pick_cases); trajectories, images and text
   are exported. Selection is automatic; no images were inspected by hand.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from actspace_build import geometry as G

from .answers import CANNOT, gold
from .data import FrozenData, dataset_of_scene
from .models import encode_jpeg
from .providers.ablations import scene_objects, vis_level
from .score import N_BOOT, RepBoot, _choice
from .sec53_behavior import CLOSED, MODELS, OPEN, ROOT, RUNS, SCOPES
from .sec54_evidence_state import final_context
from .trajectory import read_episodes

SEC54 = ROOT / "_reports" / "sec54" / "evidence_state_per_question.csv"
SETTINGS = ("active_v2", "oracle_route")
ERR_LAB = {"correct": "correct", "mirror_lr": "left-right mirror", "flip_fb": "front-back flip", "diagonal": "diagonal", "same_quadrant": "same-quadrant wrong choice"}
STATE_LAB = {"in_ctx": "Evidence still in context", "seen_lost": "Seen but no longer in context", "unseen": "Evidence not fully seen", "unanswerable": "Unanswerable question"}
SITUATIONS = (("early_answer", "Answering too early"), ("evidence_left_context", "Evidence left the context"), ("wrong_with_evidence", "Wrong despite the evidence in view"))
PREFER = ("opus-5.5", "haiku-5.5", "terra", "sonnet-5.5", "sol", "luna")


def quadrant_error(pred: str | None, true: str | None) -> str | None:
    """Relation between two quadrants (front-left etc.)."""
    if not pred or not true:
        return None
    (pf, pl), (tf, tl) = pred.split("-"), true.split("-")
    if (pf, pl) == (tf, tl):
        return "correct"
    if pf == tf:
        return "mirror_lr"
    return "flip_fb" if pl == tl else "diagonal"


def runs_of(model: str, setting: str) -> list[str]:
    base, act, _ = RUNS[model]
    return list(act) if setting == "active_v2" else ["full_r0" if model in OPEN else act[0]]


def question_fields(scene, q: dict, ans) -> dict:
    """Template-specific fields. ans: parsed answer (letter, number, cannot_determine or None)."""
    out = {"pred_distance_m": "", "true_distance_m": "", "ratio": "", "pred_option": "", "true_option": "", "pred_quadrant": "", "true_quadrant": "",
           "error_type": "", "t2_variant": "", "t2_distance_gap_m": ""}
    answered = ans is not None and ans != CANNOT
    t = q["template"]
    if t == "T1":
        out["true_distance_m"] = q["answer_raw"]
        if answered and isinstance(ans, (int, float)):
            out["pred_distance_m"] = float(ans)
            out["ratio"] = round(float(ans) / float(q["answer_raw"]), 6)
        return out
    out["true_option"] = q["answer"]
    if answered:
        out["pred_option"] = ans
    c = _choice(q, ans) if answered else None
    if t in ("T3", "T4"):
        out["true_quadrant"] = q["answer_value"]
        if c is not None:
            out["pred_quadrant"] = c
            out["error_type"] = quadrant_error(c, q["answer_value"])
    elif t == "T5":
        th = q["margins"]["theta_deg"]
        out["true_quadrant"] = G.quadrant(th[str(q["answer_obj"])]) or ""
        if c is not None:
            out["pred_quadrant"] = G.quadrant(th[str(c)]) or ""
            if c == q["answer_obj"]:
                out["error_type"] = "correct"
            else:
                e = quadrant_error(out["pred_quadrant"], out["true_quadrant"])
                out["error_type"] = "same_quadrant" if e == "correct" else (e or "")
    elif t == "T2":
        out["t2_variant"] = q.get("variant", "")
        if c is not None:
            objs = scene_objects(scene)
            ctr = lambda o: np.asarray(objs[int(o)]["center"], float)   # noqa: E731
            a = q["slots"]["OBJ1"]
            out["t2_distance_gap_m"] = round(abs(float(np.linalg.norm(ctr(a) - ctr(c))) - float(np.linalg.norm(ctr(a) - ctr(q["answer_obj"])))), 4)
    return out


def answer_rows(data, bad, state54: dict) -> list[dict]:
    rows = []
    for m in MODELS:
        for setting in SETTINGS:
            for run in runs_of(m, setting):
                d = RUNS[m][0] / setting / run
                summ = {x["qid"]: x for x in json.loads((d / "summary.json").read_text())["per_question"]}
                for qid, x in summ.items():
                    gid = x["group_id"]
                    if gid in bad:
                        continue
                    g = data.groups[gid]
                    q = next(z for z in g["questions"] if z["qid"] == qid)
                    ans = x["answer"]
                    if not x["answerable"]:
                        state = "unanswerable"
                    elif setting == "oracle_route":
                        state = "in_ctx"
                    else:
                        state = state54[(m, run, qid)]
                    rows.append({"model": m, "kind": "closed" if m in CLOSED else "open", "setting": setting, "run": run, "group_id": gid, "scene_id": g["scene_id"],
                                 "dataset": dataset_of_scene(g["scene_id"]), "qid": qid, "template": q["template"], "answerable": int(x["answerable"]),
                                 "evidence_state": state, "answered": int(ans is not None and ans != CANNOT), "cannot_determine": int(ans == CANNOT),
                                 "answer": "" if ans is None else ans, "gold": gold(q), "score": x["score"], **question_fields(data.scene(g["scene_id"]), q, ans)})
    return rows


# ---------------------------------------------------------------- summary and intervals
class SceneBoot:
    """Scene resampling with replacement (stratified by dataset for the merged scope), for medians and quartiles."""

    def __init__(self, scenes: dict, scope=None, n=N_BOOT, seed=20261114):
        ds_list = [scope] if scope else sorted(scenes)
        self.groups = [sorted(scenes[ds]) for ds in ds_list]
        self.n, self.rng = n, np.random.default_rng(seed)
        self.draws = [[self.rng.integers(len(g), size=len(g)) for g in self.groups] for _ in range(n)]

    def quantile_ci(self, items, qs=(25, 50, 75)) -> dict | None:
        """items: [(scene, value)]. → {q: (point estimate, lo, hi)}."""
        if not items:
            return None
        by = collections.defaultdict(list)
        for s, v in items:
            by[s].append(v)
        point = np.percentile([v for _, v in items], qs)
        bs = []
        for draw in self.draws:
            vals = [v for g, idx in zip(self.groups, draw) for i in idx for v in by.get(g[i], ())]
            if vals:
                bs.append(np.percentile(vals, qs))
        bs = np.array(bs)
        return {q: (float(point[k]), float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5))) for k, q in enumerate(qs)}


def summarize(rows: list, scenes: dict) -> list[dict]:
    out = []

    def rec(scope, m, s, block, metric, n, v, lo, hi, extra=""):
        out.append({"scope": scope, "model": m, "kind": "closed" if m in CLOSED else "open", "setting": s, "block": block, "metric": metric, "n": n,
                    "value": None if v is None else round(v, 6), "lo": None if lo is None else round(lo, 6), "hi": None if hi is None else round(hi, 6), "note": extra})
    for scope, lab in SCOPES:
        rb, sb = RepBoot(scenes, scope=scope), SceneBoot(scenes, scope=scope)
        for m in MODELS:
            for s in SETTINGS:
                xs = [r for r in rows if r["model"] == m and r["setting"] == s and r["answerable"] and r["answered"] and (scope is None or r["dataset"] == scope)]
                runs = list(dict.fromkeys(r["run"] for r in xs))
                t1 = [r for r in xs if r["template"] == "T1" and r["ratio"] != ""]
                qc = sb.quantile_ci([(r["scene_id"], r["ratio"]) for r in t1])
                for q, name in ((50, "ratio_median"), (25, "ratio_q1"), (75, "ratio_q3")):
                    rec(lab, m, s, "T1", name, len(t1), *(qc[q] if qc else (None, None, None)))
                for name, f in (("overestimate_share", lambda r: r["ratio"] > 1), ("underestimate_share", lambda r: r["ratio"] < 1),
                                ("within_25pct_share", lambda r: 0.75 <= r["ratio"] <= 1.25)):
                    ci = rb.mean_ci([[(r["scene_id"], float(f(r))) for r in t1 if r["run"] == run] for run in runs], f"{m}|{s}|{name}") if t1 else None
                    rec(lab, m, s, "T1", name, len(t1), *((ci["mean"], ci["lo"], ci["hi"]) if ci else (None, None, None)))
                for block, tmpl, types in (("T3", ("T3",), ("mirror_lr", "flip_fb", "diagonal")), ("T4", ("T4",), ("mirror_lr", "flip_fb", "diagonal")),
                                           ("T3+T4", ("T3", "T4"), ("mirror_lr", "flip_fb", "diagonal")),
                                           ("T5", ("T5",), ("same_quadrant", "mirror_lr", "flip_fb", "diagonal"))):
                    d = [r for r in xs if r["template"] in tmpl and r["error_type"]]
                    wrong = [r for r in d if r["error_type"] != "correct"]
                    ci = rb.mean_ci([[(r["scene_id"], float(r["error_type"] != "correct")) for r in d if r["run"] == run] for run in runs], f"{m}|{s}|{block}|err") if d else None
                    rec(lab, m, s, block, "error_rate_among_answered", len(d), *((ci["mean"], ci["lo"], ci["hi"]) if ci else (None, None, None)), f"wrong {len(wrong)}")
                    for typ in types:
                        ci = rb.mean_ci([[(r["scene_id"], float(r["error_type"] == typ)) for r in wrong if r["run"] == run] for run in runs], f"{m}|{s}|{block}|{typ}") if wrong else None
                        rec(lab, m, s, block, f"share_of_errors_{typ}", len(wrong), *((ci["mean"], ci["lo"], ci["hi"]) if ci else (None, None, None)),
                            f"{sum(r['error_type'] == typ for r in wrong)} / {len(wrong)}")
    return out


# ---------------------------------------------------------------- B. cases
TEMPLATE_ORDER = {"T4": 0, "T3": 1, "T2": 2, "T5": 3, "T1": 4}


def pick_cases(rows: list, episodes: dict) -> list[dict]:
    """3 candidate episodes per situation. Active only; haiku-5.5 uses only all_v1_r0. Within a situation, prefer 3 different models (in PREFER order), no repeated groups;
    within a model, sort by template (T4, T3, T2, T5, T1) → fewer evidence objects → group_id and take the first."""
    act = [r for r in rows if r["setting"] == "active_v2" and r["answerable"] and r["run"] == RUNS[r["model"]][1][0]]
    ep = lambda r: episodes[(r["model"], r["run"], r["group_id"])]   # noqa: E731
    wrong = lambda r: r["answered"] and r["score"] == 0   # noqa: E731
    rules = {"early_answer": lambda r: r["evidence_state"] == "unseen" and wrong(r) and ep(r)["steps"] <= 20 and not ep(r)["forced"],
             "evidence_left_context": lambda r: r["evidence_state"] == "seen_lost" and (r["cannot_determine"] or wrong(r)),
             "wrong_with_evidence": lambda r: r["evidence_state"] == "in_ctx" and r["template"] in ("T3", "T4") and r["error_type"] in ("mirror_lr", "flip_fb")}
    picked, used = [], set()
    for sit, _ in SITUATIONS:
        cand = sorted((r for r in act if rules[sit](r)), key=lambda r: (TEMPLATE_ORDER[r["template"]], ep(r)["n_ev"][r["qid"]], r["group_id"]))
        chosen = []
        for rnd in range(3):        # first one per model, then fill up from the start if not enough
            for m in PREFER:
                if len(chosen) == 3:
                    break
                if rnd == 0 and any(c["model"] == m for c in chosen):
                    continue
                r = next((r for r in cand if r["model"] == m and r["group_id"] not in used), None)
                if r is not None:
                    chosen.append(r)
                    used.add(r["group_id"])
        picked += [{"situation": sit, "n_candidates": len(cand), "n_candidate_episodes": len({(r["model"], r["group_id"]) for r in cand}), **r} for r in chosen]
    return picked


def best_frames(scene, objs, nodes, k=2) -> list[str]:
    """Up to k frames in nodes where the evidence is clearest: first by the number of evidence objects at least partially visible, then strictly visible, then the sum of their image-area fractions."""
    v = np.load(scene.wd / "visibility.npz")
    row = {int(o): i for i, o in enumerate(v["obj_ids"])}
    col = {str(n): i for i, n in enumerate(v["frame_ids"])}
    st, area = v["state"], v["area_frac"]
    sc = []
    for n in dict.fromkeys(nodes):
        if n not in col:
            continue
        s = [int(st[row[o], col[n]]) for o in objs if o in row]
        a = sum(float(area[row[o], col[n]]) for o in objs if o in row and st[row[o], col[n]] > 0)
        if any(s):
            sc.append(((sum(x >= 1 for x in s), sum(x == 2 for x in s), a), n))
    sc.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2], x[1]))
    return [n for _, n in sc[:k]]


def export_case(data, out: Path, case: dict, rows: list, ep: list, ctx: list):
    m, run, gid, sit = case["model"], case["run"], case["group_id"], case["situation"]
    g = data.groups[gid]
    scene = data.scene(g["scene_id"])
    d = out / "cases" / f"{sit}_{gid}"
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    s, end = ep[0], ep[-1]
    steps = [r for r in ep if r["type"] == "step"]
    route = [s["start_node"]] + [r["to"] for r in steps]
    masked = data.masked_nodes(gid)
    objs = scene_objects(scene)
    fq = next(q for q in g["questions"] if q["qid"] == case["qid"])
    ev = [int(o) for o in fq["evidence_objs"]]
    pose = lambda n: {"node": n, "x": round(float(scene.pose[n][0][0]), 4), "y": round(float(scene.pose[n][0][1]), 4), "yaw_deg": round(float(scene.pose[n][1]), 2)}   # noqa: E731
    obj = lambda o: {"obj_id": o, "name": objs[o]["ref_name"], "category": objs[o].get("category"),   # noqa: E731
                     "center": [round(float(c), 4) for c in objs[o]["center"]],
                     "seen_on_route": vis_level(scene, o, route), "seen_in_context": vis_level(scene, o, ctx)}
    all_objs = sorted({int(o) for q in g["questions"] for o in q["evidence_objs"]})
    if sit == "early_answer":       # not fully seen: find the clearest evidence frames in the whole (masked) view graph, and mark whether they were visited
        pool = [n for n in scene.nodes if n not in masked]
    elif sit == "evidence_left_context":
        pool = [n for n in route if n not in set(ctx)] or route
    else:
        pool = ctx
    best = best_frames(scene, ev, pool)
    images = [("start", s["start_node"])] + [(f"evidence{i}", n) for i, n in enumerate(best, 1)] + [("last", route[-1])]
    files = []
    for name, n in images:
        img = scene.image(n)
        f = f"{name}_{n}.jpg"
        (d / "images" / f).write_bytes(encode_jpeg(img))
        files.append({"file": f"images/{f}", "role": name, "node": n, "width": int(img.shape[1]), "height": int(img.shape[0]), "on_route": n in route, "in_context": n in ctx,
                      "evidence_levels": {objs[o]["ref_name"]: vis_level(scene, o, [n]) for o in ev}})
    traj = {"situation": sit, "model": m, "run": run, "group_id": gid, "scene_id": g["scene_id"], "dataset": dataset_of_scene(g["scene_id"]), "budget": s["budget"],
            "steps_used": end["steps_used"], "forced": end.get("forced"), "start": pose(s["start_node"]),
            "steps": [{"t": r["t"], "action": r["action"], "moved": bool(r["moved"]), "from": r["from"], **pose(r["to"])} for r in steps],
            "context_nodes_at_answer": ctx, "masked_nodes": sorted(masked), "focus_qid": case["qid"],
            "focus_evidence_objects": [obj(o) for o in ev], "group_objects": [obj(o) for o in all_objs], "images": files,
            "coordinates": "x, y are scene world coordinates (meters); yaw_deg is the camera's horizontal heading, heading vector = (cos yaw, sin yaw)"}
    (d / "trajectory.json").write_text(json.dumps(traj, ensure_ascii=False, indent=1))
    mine = {r["qid"]: r for r in rows if r["model"] == m and r["run"] == run and r["group_id"] == gid and r["setting"] == "active_v2"}
    pub = {v: k for k, v in s["question_ids"].items()}
    L = [f"# {dict(SITUATIONS)[sit]}: {m}, `{gid}`", "",
         f"- Situation: {dict(SITUATIONS)[sit]} (`{sit}`); model {m}, run `{run}`; dataset {traj['dataset']}, scene `{g['scene_id']}`.",
         f"- Answered after {end['steps_used']} / {s['budget']} steps; {'force-terminated: ' + str(end.get('forced')) if end.get('forced') else 'the model chose to answer'}; visited {len(set(route))} distinct viewpoints; {len(ctx)} images in context at answer time.",
         f"- Focus question: {pub[case['qid']]} ({fq['template']}), evidence state: {STATE_LAB[case['evidence_state']]}.",
         f"- {case['n_candidates']} questions in {case['n_candidate_episodes']} episodes meet the criteria for this situation; this episode was chosen automatically by fixed rules, without inspecting images.", "",
         "## notes (verbatim)", "", end.get("notes") or "(empty)", "", "## All questions of the group", ""]
    for q in g["questions"]:
        r = mine.get(q["qid"])
        L.append(f"**{pub[q['qid']]} ({q['template']})** {q['text']}")
        if q.get("options"):
            L += [f"  - {o}" for o in q["options"]]
        if r is None:
            L += ["- No per-question record", ""]
            continue
        ok = "correct" if r["score"] == 1 else ("wrong" if r["score"] == 0 else f"partial score {r['score']:.2f}")
        extra = ""
        if r["error_type"] and r["error_type"] != "correct":
            extra = f"; error type: {ERR_LAB[r['error_type']]} (answered {r['pred_quadrant']}, should be {r['true_quadrant']})"
        elif r["ratio"] != "":
            extra = f"; predicted / true = {r['ratio']:.2f}"
        L += [f"- Gold answer: `{r['gold']}`; model answer: `{r['answer'] if r['answer'] != '' else '(empty)'}`; {ok}{extra}",
              f"- Evidence state: {STATE_LAB[r['evidence_state']]}; evidence objects: {', '.join(objs[int(o)]['ref_name'] for o in q['evidence_objs'])}", ""]
    L += ["## Images", ""]
    role = {"start": "START", "last": "last frame before answering"}
    for f in files:
        lv = ", ".join(f"{k} {('invisible', 'partial', 'visible')[v]}" for k, v in f["evidence_levels"].items())
        L.append(f"- `{f['file']}`: {role.get(f['role'], 'clearest evidence frame')}, node `{f['node']}`, {f['width']}×{f['height']}; "
                 f"{'visited' if f['on_route'] else 'not visited'}, {'in context at answer time' if f['in_context'] else 'not in context at answer time'}; evidence of the focus question: {lv}")
    L += ["", "Trajectory and object positions: see `trajectory.json`.", ""]
    (d / "case.md").write_text("\n".join(L))
    return {"dir": str(d), "images": files, "notes": end.get("notes") or ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = FrozenData(verify=True)
    bad = set(data.ecfg["env"].get("budget_infeasible") or [])
    state54 = {(x["model"], x["run"], x["qid"]): x["state"] for x in csv.DictReader(open(SEC54))}
    rows = answer_rows(data, bad, state54)
    with open(out / "answers_per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"A: {len(rows)} rows, {time.time() - t0:.0f} s", flush=True)
    scenes = collections.defaultdict(set)
    for r in rows:
        scenes[r["dataset"]].add(r["scene_id"])
    summary = summarize(rows, scenes)
    with open(out / "error_type_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"summary: {len(summary)} rows, {time.time() - t0:.0f} s", flush=True)
    # B
    episodes, eps_raw, ctxs = {}, {}, {}
    for m in PREFER:
        run = RUNS[m][1][0]
        d = RUNS[m][0] / "active_v2" / run
        ctxs[m] = final_context(d)
        for ep in read_episodes(d / "trajectories.jsonl"):
            gid = ep[0]["group_id"]
            if gid in bad or ep[-1]["type"] != "episode_end":
                continue
            eps_raw[(m, run, gid)] = ep
            episodes[(m, run, gid)] = {"steps": ep[-1]["steps_used"], "forced": ep[-1].get("forced"),
                                       "n_ev": {q["qid"]: len(q["evidence_objs"]) for q in data.groups[gid]["questions"]}}
    cases = pick_cases([r for r in rows if r["model"] in PREFER], episodes)
    exported = []
    for c in cases:
        e = export_case(data, out, c, rows, eps_raw[(c["model"], c["run"], c["group_id"])], ctxs[c["model"]].get(c["group_id"], []))
        exported.append({**{k: c[k] for k in ("situation", "model", "run", "group_id", "qid", "template", "evidence_state", "answer", "gold", "score", "error_type",
                                              "pred_quadrant", "true_quadrant", "n_candidates", "n_candidate_episodes")},
                         "steps": episodes[(c["model"], c["run"], c["group_id"])]["steps"], **e})
        print("B", c["situation"], c["model"], c["group_id"], c["qid"], flush=True)
    counts = {f"{m}|{s}": sum(1 for r in rows if r["model"] == m and r["setting"] == s) for m in MODELS for s in SETTINGS}
    info = {"data_version": data.version, "rows": len(rows), "rows_by_model_setting": counts, "summary": summary, "cases": exported, "seconds": round(time.time() - t0, 1),
            "t5_quadrant_missing": sum(1 for r in rows if r["template"] == "T5" and r["answered"] and r["answerable"] and not r["pred_quadrant"])}
    (out / "sec57_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"wrote {out}: answers_per_question.csv, error_type_summary.csv, cases/ ({len(exported)}), sec57_summary.json; {info['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
