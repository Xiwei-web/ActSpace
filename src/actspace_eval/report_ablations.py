"""Ablations and cognitive-map probe (paper Section 5.6), API models. Reads only finished runs and the benchmark data; calls no model.

    python -m actspace_eval.report_ablations [--root <results>/api] [--out <root>/_reports] [--models terra,haiku-5.5]
                                             [--mem-models sol,terra,opus-5.5,sonnet-5.5,haiku-5.5,luna] [--source-prefix P]

The runs are produced by `python -m actspace_eval.providers.ablations job` (prefix abl, run abl_r0) next to the model's main runs
(source prefix: providers.ablations.SOURCE_PREFIX, or --source-prefix). Main-table definition: the 7 budget-infeasible groups are
excluded (227 groups, 962 questions). Intervals: score.RepBoot scene bootstrap (2000 resamples, stratified by dataset for the merged
scope, runs resampled when a setting has several), 95% percentile intervals; differences are paired on the same questions.

Blocks (CSV column `block`):
- A        rw40_last8_v2 vs random_walk_40 (same route, seed 0): the cost of the 8-photo memory window.
- A:<loc>  the same, split by where the evidence of an answerable question is: in the last 8 photos / only in earlier photos /
           nowhere on the route.
- A_old    rw40_last8_v2 vs rw40_last8 (wording check of the revisit lines).
- B_*      start_only vs Blind (answerable questions) and vs Passive-8.
- C_*      active_v2_medium vs active_v2 (seed 0, and all Active runs); C_behav: steps, cost.
- D:*      cognitive-map probe on the answer-time context of active_v2 and oracle_route: share of objects given a position, position
           error, quadrant accuracy, pairwise left/right and front/back relations, and the relation to T3/T4 scores.
- MEM      memory-matched exploration gain ΔRW_mem = Active − rw40_last8_v2 (and ΔRW = Active − random_walk_40 for comparison),
           for every model with a rw40_last8_v2 run.

Outputs in --out: ablations_api.json (everything), ablations_api.csv (model, scope, block, metric, series, value, lo, hi, n; values
in percent except position errors in meters and step/cost values), cogmap_api.csv (one row per probed object), ablations_api.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from actspace_build import geometry as G

from . import report_eval as RE
from .data import FrozenData, dataset_of_scene
from .providers import ablations as A
from .providers import api_eval as E
from .score import RepBoot, load_run
from .trajectory import read_episodes

SCOPES = ((None, "Merged"), ("arkitscenes", "ARKitScenes"), ("scannetpp", "ScanNet++"))
METRICS = ("total", "answer_rate", "selective", "refusal")
MET_NAMES = {"total": "total", "answer_rate": "answer rate", "selective": "selective score", "refusal": "correct abstention"}
LOC = ("evidence in the last 8 photos", "evidence only in earlier photos", "no evidence on the route")
OBJ_GROUPS = ("seen", "seen, in window", "seen, out of window", "not seen", "masked", "all (excluding masked)")
REL = ("cogmap implies the correct answer", "cogmap implies a wrong answer", "some object unknown")
MOVE, ANSWER = "move step", "answer step"
DEFAULT_MODELS = ("terra", "haiku-5.5")
DEFAULT_MEM_MODELS = ("sol", "terra", "opus-5.5", "sonnet-5.5", "haiku-5.5", "luna")


def c2(c):
    return None if c is None else {k: (float(v) if k != "n" else int(v)) for k, v in c.items()}


class Analysis:
    def __init__(self, root: Path, data: FrozenData, source_prefix: str | None = None):
        self.root, self.data, self.source_prefix = root, data, source_prefix
        self.bad = set(data.ecfg["env"]["budget_infeasible"])
        self.qof = {q["qid"]: (g, q) for g in data.groups.values() for q in g["questions"]}

    def prefix(self, m):
        return self.source_prefix or A.SOURCE_PREFIX[m]

    def run(self, m, name):
        r = load_run(self.root / m, name)
        return {**r, "per_question": [x for x in r["per_question"] if x["group_id"] not in self.bad]}

    def runs(self, m, setting):
        return [self.run(m, f"{setting}/{d.name}") for d in E.existing_runs(self.root, m, self.prefix(m), setting)
                if E.read_status(d).get("status") == "complete"]

    def done(self, m, setting):
        return (self.root / m / setting / A.RUN / "SHA256SUMS").exists()

    @staticmethod
    def scenes_of(r):
        s = collections.defaultdict(set)
        for x in r["per_question"]:
            s[dataset_of_scene(x["scene_id"])].add(x["scene_id"])
        return s

    @staticmethod
    def pair(b, ra, ka, rb, kb, sel=lambda x: True, metrics=METRICS):
        out = {}
        for k in metrics:
            ia = [RE.items({"per_question": [x for x in r["per_question"] if sel(x)]}, k) for r in ra]
            ib = [RE.items({"per_question": [x for x in r["per_question"] if sel(x)]}, k) for r in rb]
            out[k] = {"a": c2(b.mean_ci(ia, f"{ka}|{k}")), "b": c2(b.mean_ci(ib, f"{kb}|{k}")), "diff": c2(b.diff_ci(ia, f"{ka}|{k}", ib, f"{kb}|{k}"))}
        return out

    def ev_level(self, q, g, nodes):
        sc = self.data.scene(g["scene_id"])
        return min(A.vis_level(sc, int(o), nodes) for o in q["evidence_objs"])

    def steps_of(self, m, name):
        return {ep[0]["group_id"]: ep[-1]["steps_used"] for ep in read_episodes(self.root / m / name / "trajectories.jsonl")
                if ep[0]["group_id"] not in self.bad and ep[-1]["type"] == "episode_end"}

    def think(self, m, name):
        """({call kind: reasoning-token statistics}, total cost) over the main-table groups."""
        out, cost = collections.defaultdict(list), 0.0
        for line in open(self.root / m / name / "calls.jsonl"):
            x = json.loads(line)
            cost += x.get("cost_usd") or 0
            if x["type"] != "call" or x["tag"]["episode"] in self.bad:
                continue
            out[ANSWER if (x.get("parse") or {}).get("action") == "answer" else MOVE].append(x["usage"].get("reasoning_tokens") or 0)
        return {k: {"n": len(v), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
                    "zero": float(np.mean([t == 0 for t in v]))} for k, v in out.items()}, cost

    # ---------------------------------------------------------------- D: cognitive-map probe
    def cogmap(self, m, source, q_scores):
        quad = lambda f, l: G.quadrant(float(np.degrees(np.arctan2(l, f))))       # noqa: E731
        path = self.root / m / f"cogmap_{source}" / A.RUN / "probes.jsonl"
        allr = [json.loads(x) for x in open(path)]
        recs = [r for r in allr if not r["budget_infeasible"]]
        rows = []
        for r in recs:
            for o in r["objects"]:
                known = o["status"] == "known"
                err = float(np.hypot(o["forward_m"] - o["truth_forward_m"], o["left_m"] - o["truth_left_m"])) if known else None
                grp = "masked" if o["masked"] else ("not seen" if o["seen_route"] == 0 else ("seen, in window" if o["seen_window"] >= 1 else "seen, out of window"))
                rows.append({"model": m, "source": source, "group_id": r["group_id"], "scene_id": r["scene_id"], "dataset": r["dataset"],
                             "obj_id": o["obj_id"], "name": o["name"], "masked": o["masked"], "seen_route": o["seen_route"],
                             "seen_window": o["seen_window"], "group": grp, "status": o["status"], "truth_forward_m": o["truth_forward_m"],
                             "truth_left_m": o["truth_left_m"], "forward_m": o["forward_m"], "left_m": o["left_m"], "err_m": err,
                             "quadrant_ok": (quad(o["forward_m"], o["left_m"]) == quad(o["truth_forward_m"], o["truth_left_m"])) if known else None})
        by_g = collections.defaultdict(list)
        for x in rows:
            by_g[x["group_id"]].append(x)
        pairs = []
        for xs in by_g.values():
            ks = [x for x in xs if x["status"] == "known"]
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a, b = ks[i], ks[j]
                    kind = "both_seen" if a["seen_route"] and b["seen_route"] and not a["masked"] and not b["masked"] else "other"
                    pairs.append({"scene_id": a["scene_id"], "dataset": a["dataset"], "kind": kind,
                                  "lr": float(np.sign(a["left_m"] - b["left_m"]) == np.sign(a["truth_left_m"] - b["truth_left_m"])),
                                  "fb": float(np.sign(a["forward_m"] - b["forward_m"]) == np.sign(a["truth_forward_m"] - b["truth_forward_m"]))})
        scenes = collections.defaultdict(set)
        for x in rows:
            scenes[x["dataset"]].add(x["scene_id"])

        def sel(x, g):
            if g == "seen":
                return x["group"].startswith("seen")
            if g == "all (excluding masked)":
                return not x["masked"]
            return x["group"] == g
        met = {}
        for scope, lab in SCOPES:
            b = RepBoot(scenes, scope=scope)
            for g in OBJ_GROUPS:
                xs = [x for x in rows if sel(x, g) and (scope is None or x["dataset"] == scope)]
                ks = [x for x in xs if x["status"] == "known"]
                e = [x["err_m"] for x in ks]
                met[f"{lab}|{g}"] = {
                    "n": len(xs), "n_known": len(ks), "status": dict(collections.Counter(x["status"] for x in xs)),
                    "known_rate": c2(b.mean_ci([[(x["scene_id"], float(x["status"] == "known")) for x in xs]], "k")) if xs else None,
                    "n_unparsed": sum(x["status"] == "unparsed" for x in xs),
                    "err_median": float(np.median(e)) if e else None, "err_q1": float(np.percentile(e, 25)) if e else None,
                    "err_q3": float(np.percentile(e, 75)) if e else None,
                    "within_1m": c2(b.mean_ci([[(x["scene_id"], float(x["err_m"] <= 1.0)) for x in ks]], "w")) if ks else None,
                    "quadrant": c2(b.mean_ci([[(x["scene_id"], float(x["quadrant_ok"])) for x in ks]], "q")) if ks else None}
            for kind, lab2 in (("both_seen", "both seen"), (None, "all pairs of known objects")):
                ps = [p for p in pairs if (kind is None or p["kind"] == kind) and (scope is None or p["dataset"] == scope)]
                met[f"{lab}|pairs|{lab2}"] = {"n": len(ps), "lr": c2(b.mean_ci([[(p["scene_id"], p["lr"]) for p in ps]], "lr")) if ps else None,
                                              "fb": c2(b.mean_ci([[(p["scene_id"], p["fb"]) for p in ps]], "fb")) if ps else None}
        pos = {(x["group_id"], x["obj_id"]): x for x in rows}
        rel = {}
        for t in ("T3", "T4"):
            cells = collections.defaultdict(list)
            for qid, (g, q) in self.qof.items():
                if q["template"] != t or q["evidence_label"] == "unanswerable" or g["group_id"] in self.bad or qid not in q_scores:
                    continue
                xs = [pos.get((g["group_id"], q["slots"][k])) for k in sorted(q["slots"])]
                if any(x is None for x in xs):
                    continue
                if all(x["status"] == "known" for x in xs):
                    if t == "T4":
                        implied = quad(xs[0]["forward_m"], xs[0]["left_m"])
                    else:   # standing at OBJ1, facing OBJ2: quadrant of OBJ3
                        p = [np.array([x["forward_m"], x["left_m"]]) for x in xs]
                        implied = G.quadrant(G.direction_theta_facing(p[0], p[1], p[2])) if np.linalg.norm(p[1] - p[0]) > 0 else None
                    key = REL[0] if implied == q["answer_value"] else REL[1]
                else:
                    key = REL[2]
                s = q_scores[qid]
                cells[key].append((g["scene_id"], s["score"], s["answer"] == "cannot_determine"))
            b = RepBoot(scenes)
            rel[t] = {k: {"n": len(v), "score": c2(b.mean_ci([[(s, x) for s, x, _ in v]], "s")), "cd_rate": float(np.mean([c for _, _, c in v]))}
                      for k, v in cells.items()}
            if REL[0] in cells and REL[1] in cells:
                rel[t]["diff_right_minus_wrong"] = c2(b.diff_ci([[(s, x) for s, x, _ in cells[REL[0]]]], "a",
                                                                [[(s, x) for s, x, _ in cells[REL[1]]]], "b"))
        retry = {"all_groups": len(allr), "all_first_failed": sum(not r.get("first_parse_ok", r["parse_ok"]) for r in allr),
                 "all_final_failed": sum(not r["parse_ok"] for r in allr),
                 "all_retried_ok": sum(r["parse_ok"] and not r.get("first_parse_ok", True) for r in allr),
                 "main_first_failed": sum(not r.get("first_parse_ok", r["parse_ok"]) for r in recs),
                 "main_final_failed": sum(not r["parse_ok"] for r in recs), "failed_groups": [r["group_id"] for r in allr if not r["parse_ok"]]}
        return rows, {"groups": len(recs), "unparsed": retry["main_final_failed"], "retry": retry, "metrics": met, "relation": rel}

    # ---------------------------------------------------------------- A–D for one model
    def model(self, m):
        sp, res, obj_rows = self.prefix(m), {}, []
        rw, blind = self.run(m, f"random_walk_40/{sp}_r0"), self.run(m, f"blind/{sp}_r0")
        p8, orc = self.run(m, f"passive_8/{sp}_r0"), self.run(m, f"oracle_route/{sp}_r0")
        act = self.runs(m, "active_v2")
        boots = {lab: RepBoot(self.scenes_of(rw), scope=scope) for scope, lab in SCOPES}
        if self.done(m, "rw40_last8_v2"):
            l8 = self.run(m, f"rw40_last8_v2/{A.RUN}")
            res["A"] = {lab: self.pair(b, [l8], "l8", [rw], "rw") for lab, b in boots.items()}
            eps = {ep[0]["group_id"]: ep[0] for ep in read_episodes(self.root / m / "rw40_last8_v2" / A.RUN / "trajectories.jsonl")}
            loc = {}
            for x in l8["per_question"]:
                if not x["answerable"]:
                    continue
                g, q = self.qof[x["qid"]]
                s = eps[x["group_id"]]
                loc[x["qid"]] = LOC[0] if self.ev_level(q, g, s["shown_nodes"]) >= 1 else (LOC[1] if self.ev_level(q, g, s["route_nodes"]) >= 1 else LOC[2])
            res["A_groups"] = {}
            for scope, lab in SCOPES:
                for k in LOC:
                    sel = lambda x, k=k, scope=scope: loc.get(x["qid"]) == k and (scope is None or x["dataset"] == scope)    # noqa: E731
                    d = self.pair(boots[lab], [l8], "l8", [rw], "rw", sel)
                    d["n"] = sum(1 for x in l8["per_question"] if sel(x))
                    d["n_all"] = sum(1 for x in l8["per_question"] if x["answerable"] and (scope is None or x["dataset"] == scope))
                    res["A_groups"][f"{lab}|{k}"] = d
            if self.done(m, "rw40_last8"):
                old = self.run(m, f"rw40_last8/{A.RUN}")
                res["A_old"] = {lab: self.pair(b, [l8], "l8", [old], "l8old") for lab, b in boots.items()}
            res["A_photos"] = float(np.mean([len(s["shown_nodes"]) for g, s in eps.items() if g not in self.bad]))
            res["A_photos_rw"] = float(np.mean([len(s["route_nodes"]) for g, s in eps.items() if g not in self.bad]))
        if self.done(m, "start_only"):
            so = self.run(m, f"start_only/{A.RUN}")
            ans = lambda x: x["answerable"]       # noqa: E731
            res["B"] = {lab: {"vs_blind_answerable": self.pair(b, [so], "so", [blind], "blind", ans), "vs_passive8": self.pair(b, [so], "so", [p8], "p8"),
                              "vs_passive8_answerable": self.pair(b, [so], "so", [p8], "p8", ans)} for lab, b in boots.items()}
        if self.done(m, "active_v2_medium") and act:
            med = self.run(m, f"active_v2_medium/{A.RUN}")
            res["C"] = {lab: {"vs_seed0": self.pair(b, [med], "med", act[:1], "act0"), "vs_all": self.pair(b, [med], "med", act, "act")}
                        for lab, b in boots.items()}
            sm, sh = self.steps_of(m, f"active_v2_medium/{A.RUN}"), self.steps_of(m, f"active_v2/{sp}_r0")
            tm, cm = self.think(m, f"active_v2_medium/{A.RUN}")
            th, ch = self.think(m, f"active_v2/{sp}_r0")
            st = lambda d: {"median": float(np.median(list(d.values()))), "mean": float(np.mean(list(d.values()))),   # noqa: E731
                            "full": float(np.mean([v >= 40 for v in d.values()]))}
            res["C_behav"] = {"steps_medium": st(sm), "steps_high": st(sh), "steps_diff_mean": float(np.mean([sm[g] - sh[g] for g in sm])),
                              "think_medium": tm, "think_high": th, "cost_medium": cm, "cost_high": ch, "active_runs": len(act)}
        res["D"] = {}
        for source, src_run in (("active_v2", act[0] if act else None), ("oracle_route", orc)):
            if src_run is not None and self.done(m, f"cogmap_{source}"):
                rows, d = self.cogmap(m, source, {x["qid"]: x for x in src_run["per_question"]})
                obj_rows += rows
                res["D"][source] = d
        res["spent"] = E.scope_spent(self.root, m, A.PREFIX)
        res["runs"] = {}
        for s in A.ORDER:
            stt = E.read_status(self.root / m / s / A.RUN)
            res["runs"][s] = {k: stt.get(k) for k in ("status", "completed", "spent_usd", "truncation_rate", "seconds", "unparsed")} | \
                             {"calls": (stt.get("usage") or {}).get("calls"), "errors": (stt.get("usage") or {}).get("errors")}
        return res, obj_rows

    # ---------------------------------------------------------------- MEM: memory-matched exploration gain
    def mem(self, m):
        sp = self.prefix(m)
        l8, rw0 = self.run(m, f"rw40_last8_v2/{A.RUN}"), self.run(m, f"random_walk_40/{sp}_r0")
        act, rws = self.runs(m, "active_v2"), self.runs(m, "random_walk_40")
        sc = self.scenes_of(l8)
        res = {"active_runs": len(act), "rw_runs": len(rws)}
        it = lambda rs, k: [RE.items(r, k) for r in rs]          # noqa: E731
        for scope, lab in SCOPES:
            b = RepBoot(sc, scope=scope)
            res[lab] = {k: {"active0": c2(b.mean_ci(it(act[:1], k), f"a0|{k}")), "active_all": c2(b.mean_ci(it(act, k), f"a|{k}")),
                            "l8": c2(b.mean_ci(it([l8], k), f"l8|{k}")), "rw0": c2(b.mean_ci(it([rw0], k), f"rw0|{k}")),
                            "rw_all": c2(b.mean_ci(it(rws, k), f"rw|{k}")),
                            "d_mem0": c2(b.diff_ci(it(act[:1], k), f"a0|{k}", it([l8], k), f"l8|{k}")),
                            "d_mem_all": c2(b.diff_ci(it(act, k), f"a|{k}", it([l8], k), f"l8|{k}")),
                            "d_rw0": c2(b.diff_ci(it(act[:1], k), f"a0|{k}", it([rw0], k), f"rw0|{k}")),
                            "d_rw_all": c2(b.diff_ci(it(act, k), f"a|{k}", it(rws, k), f"rw|{k}")),
                            "l8_minus_rw0": c2(b.diff_ci(it([l8], k), f"l8|{k}", it([rw0], k), f"rw0|{k}"))}
                        for k in ("total", "answer_rate", "selective")}
        return res


# ---------------------------------------------------------------- CSV and Markdown

MEM_SERIES = (("d_mem0", "ΔRW_mem (seed 0)"), ("d_rw0", "ΔRW (seed 0)"), ("d_mem_all", "ΔRW_mem (all Active runs)"),
              ("d_rw_all", "ΔRW (all runs)"), ("l8", "rw40_last8_v2"), ("active0", "Active (seed 0)"))


def csv_rows(out, mem) -> list[list]:
    rows = []

    def rec(model, scope, block, metric, a_name, b_name, v):
        for k, nm in (("a", a_name), ("b", b_name), ("diff", f"{a_name} − {b_name}")):
            c = v.get(k)
            if c:
                rows.append([model, scope, block, metric, nm, round(100 * c["mean"], 2), round(100 * c["lo"], 2), round(100 * c["hi"], 2), c.get("n")])
    for m, d in out.items():
        for _, sc in SCOPES:
            if "A" in d:
                for k in METRICS:
                    if d["A"][sc][k]["a"] is not None:
                        rec(m, sc, "A", k, "rw40_last8_v2", "random_walk_40", d["A"][sc][k])
                for g in LOC:
                    v = d["A_groups"][f"{sc}|{g}"]
                    for k in METRICS[:3]:
                        if v[k]["a"] is not None:
                            rec(m, sc, f"A:{g}", k, "rw40_last8_v2", "random_walk_40", v[k])
                if sc == "Merged" and "A_old" in d:
                    for k in METRICS:
                        if d["A_old"][sc][k]["a"] is not None:
                            rec(m, sc, "A_old", k, "rw40_last8_v2", "rw40_last8 (old wording)", d["A_old"][sc][k])
            if "B" in d:
                for blk, key, mets, a, b in (("B_vs_blind_answerable", "vs_blind_answerable", METRICS[:3], "start_only", "Blind"),
                                             ("B_vs_passive8", "vs_passive8", METRICS, "start_only", "Passive-8")):
                    for k in mets:
                        if d["B"][sc][key][k]["a"] is not None:
                            rec(m, sc, blk, k, a, b, d["B"][sc][key][k])
            if "C" in d:
                for k in METRICS:
                    if d["C"][sc]["vs_seed0"][k]["a"] is not None:
                        rec(m, sc, "C_vs_seed0", k, "medium", "high (seed 0)", d["C"][sc]["vs_seed0"][k])
                    if sc == "Merged" and d["C"][sc]["vs_all"][k]["a"] is not None:
                        rec(m, sc, "C_vs_3runs", k, "medium", "high (all Active runs)", d["C"][sc]["vs_all"][k])
            for src, lab in (("active_v2", "Active"), ("oracle_route", "oracle_route")):
                if src not in d["D"]:
                    continue
                D = d["D"][src]
                for g in OBJ_GROUPS[:5]:
                    v = D["metrics"][f"{sc}|{g}"]
                    if not v["n"]:
                        continue
                    for k in ("known_rate", "within_1m", "quadrant"):
                        if v[k]:
                            rows.append([m, sc, f"D:{src}:{g}", k, lab, round(100 * v[k]["mean"], 2), round(100 * v[k]["lo"], 2), round(100 * v[k]["hi"], 2), v[k]["n"]])
                    if v["err_median"] is not None:
                        rows.append([m, sc, f"D:{src}:{g}", "err_median_m", lab, round(v["err_median"], 3), round(v["err_q1"], 3), round(v["err_q3"], 3), v["n_known"]])
                for kind in ("both seen", "all pairs of known objects"):
                    if sc != "Merged" and kind != "both seen":
                        continue
                    v = D["metrics"][f"{sc}|pairs|{kind}"]
                    for k in ("lr", "fb"):
                        if v[k]:
                            rows.append([m, sc, f"D:{src}:pairs:{kind}", k, lab, round(100 * v[k]["mean"], 2), round(100 * v[k]["lo"], 2), round(100 * v[k]["hi"], 2), v["n"]])
                if sc == "Merged":
                    for t in ("T3", "T4"):
                        r = D["relation"][t]
                        for k in REL:
                            if k in r:
                                rows.append([m, sc, f"D:{src}:relation:{t}", "score", k, round(100 * r[k]["score"]["mean"], 2),
                                             round(100 * r[k]["score"]["lo"], 2), round(100 * r[k]["score"]["hi"], 2), r[k]["n"]])
        if "C_behav" in d:
            for lev in ("medium", "high"):
                rows.append([m, "Merged", "C_behav", f"steps_mean_{lev}", lev, round(d["C_behav"][f"steps_{lev}"]["mean"], 3), "", "", 227])
                rows.append([m, "Merged", "C_behav", f"cost_usd_{lev}", lev, round(d["C_behav"][f"cost_{lev}"], 4), "", "", ""])
    for m, d in mem.items():
        for _, sc in SCOPES:
            for k in ("total", "answer_rate", "selective"):
                for key, nm in MEM_SERIES:
                    c = d[sc][k][key]
                    if c:
                        rows.append([m, sc, "MEM", k, nm, round(100 * c["mean"], 2), round(100 * c["lo"], 2), round(100 * c["hi"], 2), c["n"]])
    return rows


def fmt(c, sign=False):
    if c is None:
        return "—"
    f = "+.1f" if sign else ".1f"
    return f"{100 * c['mean']:{f}} [{100 * c['lo']:{f}}, {100 * c['hi']:{f}}]".replace("-", "−")


def markdown(out, mem) -> str:
    L = ["# Ablations and cognitive-map probe (API models)", "",
         "Main-table definition (227 groups, 962 questions). Percentages; brackets are 95% scene-bootstrap intervals; differences are paired. "
         "All values are also in `ablations_api.csv`; per-object probe results in `cogmap_api.csv`.", ""]

    def pair_table(title, get, a, b, scopes=("Merged",), metrics=METRICS):
        rows = [(m, sc, get(d, sc)) for m, d in out.items() for sc in scopes if get(d, sc) is not None]
        if not rows:
            return
        L.extend([f"## {title}", "", f"| Model | Scope | Metric | {a} | {b} | Difference |", "|---|---|---|---|---|---|"])
        for m, sc, v in rows:
            for k in metrics:
                if v[k]["a"] is not None:
                    L.append(f"| {m} | {sc} | {MET_NAMES[k]} | {fmt(v[k]['a'])} | {fmt(v[k]['b'])} | {fmt(v[k]['diff'], True)} |")
        L.append("")
    pair_table("(A) Memory window: rw40_last8_v2 vs random_walk_40", lambda d, sc: d.get("A", {}).get(sc), "rw40_last8_v2", "random_walk_40",
               scopes=tuple(lab for _, lab in SCOPES))
    if any("A_groups" in d for d in out.values()):
        L += ["## (A) By location of the evidence (answerable questions, merged)", "",
              "| Model | Evidence | Questions | Metric | rw40_last8_v2 | random_walk_40 | Difference |", "|---|---|---|---|---|---|---|"]
        for m, d in out.items():
            for g in LOC if "A_groups" in d else ():
                v = d["A_groups"][f"Merged|{g}"]
                for k in METRICS[:3]:
                    if v[k]["a"] is not None:
                        L.append(f"| {m} | {g} | {v['n']} ({100 * v['n'] / max(v['n_all'], 1):.1f}%) | {MET_NAMES[k]} | {fmt(v[k]['a'])} | {fmt(v[k]['b'])} | {fmt(v[k]['diff'], True)} |")
        L.append("")
    pair_table("(B) START photo only vs Blind (answerable questions)", lambda d, sc: d.get("B", {}).get(sc, {}).get("vs_blind_answerable"),
               "start_only", "Blind", metrics=METRICS[:3])
    pair_table("(B) START photo only vs Passive-8", lambda d, sc: d.get("B", {}).get(sc, {}).get("vs_passive8"), "start_only", "Passive-8")
    pair_table("(C) Reasoning effort medium vs high (Active v2, seed 0)", lambda d, sc: d.get("C", {}).get(sc, {}).get("vs_seed0"), "medium", "high")
    for src, lab in (("active_v2", "Active"), ("oracle_route", "oracle_route")):
        if not any(src in d["D"] for d in out.values()):
            continue
        L += [f"## (D) Cognitive-map probe on the {lab} context (merged)", "",
              "| Model | Objects | n | Position given | Median error (IQR), m | Error ≤ 1 m | Quadrant correct |", "|---|---|---|---|---|---|---|"]
        for m, d in out.items():
            if src not in d["D"]:
                continue
            for g in OBJ_GROUPS[:5]:
                v = d["D"][src]["metrics"][f"Merged|{g}"]
                if v["n"]:
                    e = "—" if v["err_median"] is None else f"{v['err_median']:.2f} ({v['err_q1']:.2f}–{v['err_q3']:.2f})"
                    L.append(f"| {m} | {g} | {v['n']} | {fmt(v['known_rate'])} | {e} | {fmt(v['within_1m'])} | {fmt(v['quadrant'])} |")
        L.append("")
    if mem:
        L += ["## Memory-matched exploration gain ΔRW_mem = Active − rw40_last8_v2 (merged, total score, seed 0)", "",
              "| Model | Active | rw40_last8_v2 | random_walk_40 | ΔRW | ΔRW_mem |", "|---|---|---|---|---|---|"]
        for m, d in mem.items():
            t = d["Merged"]["total"]
            L.append(f"| {m} | {fmt(t['active0'])} | {fmt(t['l8'])} | {fmt(t['rw0'])} | {fmt(t['d_rw0'], True)} | {fmt(t['d_mem0'], True)} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    from .providers import registry as R
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="API results root (default: <results>/api)")
    ap.add_argument("--out", help="output directory (default: <root>/_reports)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="models with ablation runs (A–D)")
    ap.add_argument("--mem-models", default=",".join(DEFAULT_MEM_MODELS), help="models for ΔRW_mem (need a rw40_last8_v2 run)")
    ap.add_argument("--source-prefix", help="prefix of the main runs (default: providers.ablations.SOURCE_PREFIX)")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else R.results_root(R.load_registry())
    outd = Path(a.out) if a.out else root / "_reports"
    an = Analysis(root, FrozenData(verify=False), a.source_prefix)
    out, obj_rows, mem = {}, [], {}
    for m in [x for x in a.models.split(",") if x]:
        out[m], rows = an.model(m)
        obj_rows += rows
    for m in [x for x in a.mem_models.split(",") if x]:
        if an.done(m, "rw40_last8_v2"):
            mem[m] = an.mem(m)
        else:
            print(f"{m}: no sealed rw40_last8_v2 run, skipped for ΔRW_mem")
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "ablations_api.json").write_text(json.dumps({"ablations": out, "memory_matched": mem}, ensure_ascii=False, indent=0))
    with open(outd / "ablations_api.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scope", "block", "metric", "series", "value", "lo", "hi", "n"])
        w.writerows(csv_rows(out, mem))
    if obj_rows:
        with open(outd / "cogmap_api.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(obj_rows[0]))
            w.writeheader()
            for x in obj_rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})
    (outd / "ablations_api.md").write_text(markdown(out, mem))
    print(f"wrote {outd}: ablations_api.json, ablations_api.csv, {'cogmap_api.csv, ' if obj_rows else ''}ablations_api.md")


if __name__ == "__main__":
    main()
