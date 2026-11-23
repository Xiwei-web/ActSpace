"""The two behaviour statistics of Section 5.3. Reads only existing trajectories, call logs and the frozen data; no model is called.

    python -m actspace_eval.sec53_behavior --out results/_reports/sec53

A. Random-walk evidence-coverage curve: random_walk_40 routes are taken from each model's existing runs (episode_start.route; for masked
   groups the route is already on the masked view graph).
   For every group and every k = 0…40, only viewpoints reached within the first k steps count.
B. Active behaviour of every model: one row per episode, plus a summary.

Definitions (visibility levels use the repository implementation providers.ablations.vis_level):
- Evidence of an episode = union of evidence_objs over all answerable questions of the group; "at least partial" = the object is not invisible
  at some visited viewpoint; "strictly visible" = it is visible at some viewpoint
  (same definition as the evidence hit of the score tables; this script checks it episode by episode against evidence_hit_all in summary.json).
- Steps = episode_end.steps_used (including failed moves and invalid replies); revisit rate = share of successful moves that arrive at an
  already visited viewpoint; turn share = share of turn_left / turn_right among all actions (including invalid ones);
  failed move = a move action that did not move; invalid reply = not a legal move action; farthest from start = maximum horizontal distance
  between a visited viewpoint and the start.
- Main-table definition: excludes the 7 groups in env.budget_infeasible, 227 groups.
- Truncated: some call in the episode hit the output limit (cost.TRUNCATED_FINISH); context overflow: episode_end.forced = context_overflow.
  "Last call truncated" is recorded as well (a narrower truncation definition).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np

from .cost import TRUNCATED_FINISH
from .data import MOVE_ACTIONS, FrozenData, dataset_of_scene, results_root
from .providers.ablations import vis_level
from .trajectory import read_episodes

ROOT = results_root()        # results root (results in configs/paths.yaml)
OPEN = ["VST-7B-RL", "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B", "InternVL3_5-8B", "Cambrian-S-7B", "Qwen3.5-4B", "Qwen3.5-9B",
        "Qwen3.6-27B", "Qwen3.6-35B-A3B", "Qwen3.8-27B", "gemma-4-31B-it"]
# model → (results directory, Active runs, random_walk_40 runs (by seed))
RUNS = {m: (ROOT / m, ["full_r0"], ["full_s0"]) for m in OPEN}
RUNS.update({"sol": (ROOT / "api" / "sol", ["all_r0"], ["all_r0"]), "terra": (ROOT / "api" / "terra", ["all_r0"], ["all_r0"]),
             "luna": (ROOT / "api" / "luna", ["all_r0"], ["all_r0"]), "opus-5.5": (ROOT / "api" / "opus-5.5", ["all_v1_r0"], ["all_v1_r0"]),
             "sonnet-5.5": (ROOT / "api" / "sonnet-5.5", ["all_v1_r0"], ["all_v1_r0"]),
             "haiku-5.5": (ROOT / "api" / "haiku-5.5", ["all_v1_r0", "all_v1_r1", "all_v1_r2"], ["all_v1_r0", "all_v1_r1", "all_v1_r2"])})
CLOSED = ["opus-5.5", "sol", "sonnet-5.5", "terra", "haiku-5.5", "luna"]
MODELS = CLOSED + OPEN
EXTRA_RW = {"haiku-5.5(all)": (ROOT / "api" / "haiku-5.5", ["all_r0", "all_r1", "all_r2"])}     # only used to cross-check routes
SCOPES = ((None, "Merged"), ("arkitscenes", "ARKitScenes"), ("scannetpp", "ScanNet++"))
TURNS = ("turn_left", "turn_right")
BUDGET = 40
RW = "random_walk_40"


# ---------------------------------------------------------------- quantities on a viewpoint sequence
def evidence_objs(group: dict) -> list:
    """Evidence of an episode: union of evidence_objs over all answerable questions of the group."""
    return sorted({int(o) for q in group["questions"] if q["evidence_label"] != "unanswerable" for o in q["evidence_objs"]})


def prefix_coverage(scene, ev, seq) -> tuple[np.ndarray, np.ndarray]:
    """→ (allp, allv), each len(seq) booleans: using only seq[:k + 1], whether all evidence is at least partial / strictly visible. True when there are no evidence objects (all([]))."""
    if not ev:
        return np.ones(len(seq), bool), np.ones(len(seq), bool)
    lv = np.array([[vis_level(scene, o, [n]) for o in ev] for n in seq])
    cum = np.maximum.accumulate(lv, axis=0)
    return (cum >= 1).all(axis=1), (cum == 2).all(axis=1)


def prefix_views_far(scene, seq) -> tuple[list, list]:
    """Number of distinct viewpoints and farthest horizontal distance from the start using only seq[:k + 1]."""
    p0 = np.asarray(scene.pose[seq[0]][0][:2], float)
    seen, far, views, fars = set(), 0.0, [], []
    for n in seq:
        seen.add(n)
        far = max(far, float(np.linalg.norm(np.asarray(scene.pose[n][0][:2], float) - p0)))
        views.append(len(seen))
        fars.append(far)
    return views, fars


def move_stats(seq, acts, moved) -> dict:
    """Counts of revisits (successful moves arriving at an already visited viewpoint), turns, failed moves and invalid replies. seq[0] is the start; then one arrival node per action."""
    visited, rev, arr = {seq[0]}, 0, 0
    for n, m in zip(seq[1:], moved):
        if m:
            arr += 1
            rev += n in visited
            visited.add(n)
    return {"revisits": rev, "arrivals": arr, "turns": sum(a in TURNS for a in acts), "actions": len(acts),
            "cannot_move": sum((not m) and a in MOVE_ACTIONS for a, m in zip(acts, moved)), "invalid": sum(a not in MOVE_ACTIONS for a in acts)}


def at(arr, k):
    """Value after the first k steps; if the route has fewer than k steps, the final value."""
    return arr[min(k, len(arr) - 1)]


# ---------------------------------------------------------------- reading
def rw_routes(run_dir: Path) -> dict:
    """{group_id: [start, node reached at step 1, …]} and actions."""
    out = {}
    for ep in read_episodes(run_dir / "trajectories.jsonl"):
        s = ep[0]
        out[s["group_id"]] = ([s["start_node"]] + [n for _, n in s["route"]], [a for a, _ in s["route"]])
    return out


def call_flags(run_dir: Path) -> dict:
    """{group_id: {"calls", "trunc" (number of calls that hit the output limit), "last_trunc" (whether the last call hit the limit)}}."""
    out = collections.defaultdict(lambda: {"calls": 0, "trunc": 0, "last_trunc": False})
    with open(run_dir / "calls.jsonl") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            g = (r.get("tag") or {}).get("episode") or r.get("group_id")
            t = r.get("finish_reason") in TRUNCATED_FINISH
            d = out[g]
            d["calls"] += 1
            d["trunc"] += t
            d["last_trunc"] = t
    return out


def q3(xs, nd=2) -> list:
    return [round(float(x), nd) for x in np.percentile(xs, [25, 50, 75])] if len(xs) else [None, None, None]


# ---------------------------------------------------------------- A
def coverage_curves(data, bad, routes_by_seed: dict) -> tuple[list, dict]:
    """routes_by_seed: {seed: {group_id: (seq, acts)}}. → (CSV rows, {(seed, scope): {k: row}}). Seed "pooled" combines all seeds."""
    per = {}     # (seed, gid) → dict(ds, allp, allv, views, far, n)
    for seed, routes in routes_by_seed.items():
        for gid, (seq, _) in routes.items():
            if gid in bad:
                continue
            g = data.groups[gid]
            sc = data.scene(g["scene_id"])
            allp, allv = prefix_coverage(sc, evidence_objs(g), seq)
            views, fars = prefix_views_far(sc, seq)
            per[(seed, gid)] = {"ds": dataset_of_scene(g["scene_id"]), "allp": allp, "allv": allv, "views": views, "far": fars, "moves": len(seq) - 1}
    rows, index = [], {}
    seeds = list(routes_by_seed) + (["pooled"] if len(routes_by_seed) > 1 else [])
    for seed in seeds:
        for scope, lab in SCOPES:
            eps = [v for (s, _), v in per.items() if (seed == "pooled" or s == seed) and (scope is None or v["ds"] == scope)]
            for k in range(BUDGET + 1):
                v, f = q3([at(e["views"], k) for e in eps]), q3([at(e["far"], k) for e in eps], 3)
                row = {"seed": seed, "scope": lab, "k": k, "n_episodes": len(eps),
                       "frac_all_partial": round(float(np.mean([at(e["allp"], k) for e in eps])), 6),
                       "frac_all_visible": round(float(np.mean([at(e["allv"], k) for e in eps])), 6),
                       "views_q1": v[0], "views_median": v[1], "views_q3": v[2], "far_q1_m": f[0], "far_median_m": f[1], "far_q3_m": f[2],
                       "n_routes_shorter_than_k": sum(e["moves"] < k for e in eps)}
                rows.append(row)
                index.setdefault((str(seed), lab), {})[k] = row
    return rows, index


# ---------------------------------------------------------------- B
def active_rows(data, bad, model: str, kind: str, base: Path, runs, rw0: dict) -> tuple[list, dict]:
    """Active of one model: one row per episode (227 main-table groups). rw0: seed-0 random_walk_40 routes of the same groups. Also returns check counts against summary.json."""
    rows, chk = [], collections.Counter()
    for run in runs:
        d = base / "active_v2" / run
        flags = call_flags(d)
        summ = json.loads((d / "summary.json").read_text())
        pe = {x["group_id"]: x for x in summ["per_episode"]}
        for ep in read_episodes(d / "trajectories.jsonl"):
            s, end = ep[0], ep[-1]
            gid = s["group_id"]
            if gid in bad or end["type"] != "episode_end":
                continue
            g = data.groups[gid]
            sc = data.scene(g["scene_id"])
            steps = [r for r in ep if r["type"] == "step"]
            seq = [s["start_node"]] + [r["to"] for r in steps]
            acts, moved = [r["action"] for r in steps], [bool(r["moved"]) for r in steps]
            ev = evidence_objs(g)
            allp, allv = prefix_coverage(sc, ev, seq)
            views, fars = prefix_views_far(sc, seq)
            ms = move_stats(seq, acts, moved)
            k = int(end["steps_used"])
            rseq = rw0[gid][0]
            rp, rv = prefix_coverage(sc, ev, rseq[:k + 1])
            fl = flags.get(gid, {"calls": 0, "trunc": 0, "last_trunc": False})
            overflow = end.get("forced") == "context_overflow"
            row = {"model": model, "kind": kind, "run": run, "group_id": gid, "dataset": dataset_of_scene(g["scene_id"]), "steps": k,
                   "forced": int(end.get("forced") is not None), "forced_reason": end.get("forced") or "",
                   "views": views[-1], "far_m": round(fars[-1], 4),
                   "revisit_rate": round(ms["revisits"] / ms["arrivals"], 6) if ms["arrivals"] else "", "revisits": ms["revisits"], "arrivals": ms["arrivals"],
                   "turn_share": round(ms["turns"] / ms["actions"], 6) if ms["actions"] else "", "turns": ms["turns"], "actions": ms["actions"],
                   "cannot_move": ms["cannot_move"], "invalid": ms["invalid"],
                   "all_partial": int(allp[-1]), "all_visible": int(allv[-1]),
                   "rw_same_steps_all_partial": int(rp[-1]), "rw_same_steps_all_visible": int(rv[-1]),
                   "n_calls": fl["calls"], "n_truncated_calls": fl["trunc"], "last_call_truncated": int(fl["last_trunc"]), "context_overflow": int(overflow),
                   "trunc_or_overflow": int(fl["trunc"] > 0 or overflow)}
            rows.append(row)
            x = pe.get(gid)
            if x is not None:
                chk["episodes"] += 1
                chk["hit_all_agree"] += bool(x.get("evidence_hit_all")) == bool(allv[-1])
                chk["cannot_move_agree"] += x.get("cannot_move") == ms["cannot_move"]
                chk["invalid_agree"] += x.get("invalid_steps") == ms["invalid"]
                chk["steps_agree"] += x.get("steps_used") == k
            chk["actions_ne_steps"] += len(acts) != k
    return rows, dict(chk)


def rw_rows(data, bad, rw0: dict) -> list:
    """Random walk (seed 0, full length) with the same fields, as one column of the summary table."""
    rows = []
    for gid, (seq, acts) in rw0.items():
        if gid in bad:
            continue
        g = data.groups[gid]
        sc = data.scene(g["scene_id"])
        allp, allv = prefix_coverage(sc, evidence_objs(g), seq)
        views, fars = prefix_views_far(sc, seq)
        ms = move_stats(seq, acts, [True] * len(acts))
        rows.append({"model": RW, "kind": "baseline", "run": "seed0", "group_id": gid, "dataset": dataset_of_scene(g["scene_id"]), "steps": len(acts), "forced": 0,
                     "views": views[-1], "far_m": fars[-1], "revisits": ms["revisits"], "arrivals": ms["arrivals"], "turns": ms["turns"], "actions": ms["actions"],
                     "cannot_move": 0, "invalid": 0, "all_partial": int(allp[-1]), "all_visible": int(allv[-1]),
                     "rw_same_steps_all_partial": int(allp[-1]), "rw_same_steps_all_visible": int(allv[-1])})
    return rows


METRICS = ("n_episodes", "steps_q1", "steps_median", "steps_q3", "steps_mean", "frac_full_40", "frac_forced", "views_q1", "views_median", "views_q3",
           "far_q1_m", "far_median_m", "far_q3_m", "revisit_rate", "turn_share", "cannot_move_mean", "invalid_mean", "frac_all_partial", "frac_all_visible",
           "rw_same_steps_frac_all_partial", "rw_same_steps_frac_all_visible")


def aggregate(xs: list) -> dict:
    """Behaviour metrics of the Active summary table. Revisit rate and turn share are computed from counts pooled over all episodes."""
    if not xs:
        return {m: None for m in METRICS} | {"n_episodes": 0}
    st, vw, fr = q3([x["steps"] for x in xs]), q3([x["views"] for x in xs]), q3([x["far_m"] for x in xs])
    mean = lambda k: round(float(np.mean([x[k] for x in xs])), 6)   # noqa: E731
    arr, act = sum(x["arrivals"] for x in xs), sum(x["actions"] for x in xs)
    return {"n_episodes": len(xs), "steps_q1": st[0], "steps_median": st[1], "steps_q3": st[2], "steps_mean": mean("steps"),
            "frac_full_40": round(float(np.mean([x["steps"] >= BUDGET for x in xs])), 6), "frac_forced": mean("forced"),
            "views_q1": vw[0], "views_median": vw[1], "views_q3": vw[2], "far_q1_m": fr[0], "far_median_m": fr[1], "far_q3_m": fr[2],
            "revisit_rate": round(sum(x["revisits"] for x in xs) / arr, 6) if arr else None,
            "turn_share": round(sum(x["turns"] for x in xs) / act, 6) if act else None,
            "cannot_move_mean": mean("cannot_move"), "invalid_mean": mean("invalid"),
            "frac_all_partial": mean("all_partial"), "frac_all_visible": mean("all_visible"),
            "rw_same_steps_frac_all_partial": mean("rw_same_steps_all_partial"), "rw_same_steps_frac_all_visible": mean("rw_same_steps_all_visible")}


def summary_table(rows_by_model: dict, models, keep=lambda x: True) -> dict:
    """{scope: {model: metrics}}."""
    ds = {lab: scope for scope, lab in SCOPES}
    return {lab: {m: aggregate([x for x in rows_by_model[m] if keep(x) and (ds[lab] is None or x["dataset"] == ds[lab])]) for m in models} for lab in ds}


def write_summary_csv(path: Path, table: dict, models, extra_rows=()):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "metric", *models])
        for lab, by in table.items():
            for metric in METRICS:
                w.writerow([lab, metric, *[by[m][metric] for m in models]])
            for name, vals in extra_rows:
                w.writerow([lab, name, *[vals.get((lab, m), "") for m in models]])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = FrozenData(verify=True)
    bad = set(data.ecfg["env"].get("budget_infeasible") or [])
    main_groups = [g for g in data.group_ids if g not in bad]
    info = {"data_version": data.version, "groups_main": len(main_groups), "budget_infeasible": sorted(bad), "models": MODELS, "runs": {}}
    print(f"data {data.version}: main table {len(main_groups)} groups", flush=True)

    # ---- A.1 are the routes identical for all models?
    routes = {}      # (model, seed) → {gid: (seq, acts)}
    for m in MODELS:
        base, _, rws = RUNS[m]
        for seed, run in enumerate(rws):
            routes[(m, seed)] = rw_routes(base / RW / run)
    for m, (base, rws) in EXTRA_RW.items():
        for seed, run in enumerate(rws):
            routes[(m, seed)] = rw_routes(base / RW / run)
    ref = {s: routes[("haiku-5.5", s)] for s in (0, 1, 2)}
    same = {}
    for (m, seed), r in sorted(routes.items(), key=lambda x: (x[0][1], x[0][0])):
        diff = [g for g in main_groups if r.get(g) != ref[seed].get(g)]
        same[f"{m}|seed{seed}"] = {"groups": len([g for g in main_groups if g in r]), "differs_from_haiku_all_v1": len(diff), "example": diff[:3]}
    seed0_identical = all(v["differs_from_haiku_all_v1"] == 0 and v["groups"] == len(main_groups) for k, v in same.items() if k.endswith("seed0"))
    info["routes"] = {"compare": same, "seed0_identical_across_models": seed0_identical, "models_with_seed0": len([k for k in same if k.endswith("seed0")]),
                      "seeds_1_2_source": "haiku-5.5 all_v1_r1 / all_v1_r2 (checked against haiku-5.5 all_r1 / all_r2)",
                      "route_lengths_seed0": dict(collections.Counter(len(ref[0][g][0]) - 1 for g in main_groups))}
    print(f"random_walk_40 seed-0 routes identical across {info['routes']['models_with_seed0']} runs: {seed0_identical}", flush=True)

    # ---- A.2–3 curves
    curve_rows, index = [], {}
    if seed0_identical:
        rows, index = coverage_curves(data, bad, ref)
        curve_rows = [{"route_set": "all_models", **r} for r in rows]
    else:        # compute per model (seed 0 only), plus the 3 haiku seeds
        rows, index = coverage_curves(data, bad, ref)
        curve_rows = [{"route_set": "haiku-5.5", **r} for r in rows]
        for m in MODELS:
            if m != "haiku-5.5":
                rows, _ = coverage_curves(data, bad, {0: routes[(m, 0)]})
                curve_rows += [{"route_set": m, **r} for r in rows]
    with open(out / "rw_coverage_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        w.writeheader()
        w.writerows(curve_rows)
    info["curve"] = {f"{seed}|{lab}": {str(k): index[(seed, lab)][k] for k in (0, 5, 10, 15, 20, 25, 30, 35, 40)} for (seed, lab) in index}
    p40 = index[("pooled", "Merged")][40]
    info["selfcheck_A"] = {"expected": {"frac_all_partial": 0.562, "frac_all_visible": 0.134, "n_episodes": 681},
                           "got": {k: p40[k] for k in ("frac_all_partial", "frac_all_visible", "n_episodes")},
                           "by_dataset_expected": {"ARKitScenes": [0.537, 0.129], "ScanNet++": [0.583, 0.137]},
                           "by_dataset_got": {lab: [index[("pooled", lab)][40]["frac_all_partial"], index[("pooled", lab)][40]["frac_all_visible"]] for lab in ("ARKitScenes", "ScanNet++")}}
    info["selfcheck_A"]["pass"] = abs(p40["frac_all_partial"] - 0.562) < 0.0005 and abs(p40["frac_all_visible"] - 0.134) < 0.0005 and p40["n_episodes"] == 681
    print("self-check A:", json.dumps(info["selfcheck_A"], ensure_ascii=False), flush=True)

    # ---- B
    by_model, checks = {}, {}
    for m in MODELS:
        base, runs, _ = RUNS[m]
        rw0 = routes[(m, 0)]
        rows, chk = active_rows(data, bad, m, "closed" if m in CLOSED else "open", base, runs, rw0)
        by_model[m], checks[m] = rows, chk
        info["runs"][m] = {"active": [str(base / "active_v2" / r) for r in runs], "random_walk_seed0": str(base / RW / RUNS[m][2][0])}
        print(f"{m}: {len(rows)} episodes, checks {chk}, {time.time() - t0:.0f} s", flush=True)
    by_model[RW] = rw_rows(data, bad, ref[0])
    fields = list(by_model[MODELS[0]][0])
    with open(out / "active_behavior_per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in MODELS:
            w.writerows(by_model[m])
    cols = MODELS + [RW]
    table = summary_table(by_model, cols)
    write_summary_csv(out / "active_behavior_summary.csv", table, cols)
    keep = lambda x: not x.get("trunc_or_overflow")      # noqa: E731
    table_nt = summary_table(by_model, OPEN, keep)
    ds = {lab: scope for scope, lab in SCOPES}
    removed = {}
    for name, f in (("n_removed", lambda x: x["trunc_or_overflow"]), ("n_with_truncated_call", lambda x: x["n_truncated_calls"] > 0),
                    ("n_context_overflow", lambda x: x["context_overflow"]), ("n_last_call_truncated", lambda x: x["last_call_truncated"])):
        removed[name] = {(lab, m): sum(1 for x in by_model[m] if f(x) and (ds[lab] is None or x["dataset"] == ds[lab])) for lab in ds for m in MODELS}
    write_summary_csv(out / "active_behavior_summary_no_trunc.csv", table_nt, OPEN, extra_rows=list(removed.items()))
    info["summary"], info["summary_no_trunc"] = table, table_nt
    info["removed"] = {name: {f"{lab}|{m}": v for (lab, m), v in d.items()} for name, d in removed.items()}
    info["summary_json_checks"] = checks
    sm = table["Merged"]
    info["selfcheck_B"] = {"opus-5.5": {"expected": {"steps_median": 17, "frac_all_partial": 0.608}, "got": {k: sm["opus-5.5"][k] for k in ("steps_median", "frac_all_partial")}},
                           "sol": {"expected": {"steps_median": 40, "frac_all_partial": 0.771}, "got": {k: sm["sol"][k] for k in ("steps_median", "frac_all_partial")}}}
    info["selfcheck_B"]["pass"] = all(v["got"]["steps_median"] == v["expected"]["steps_median"] and abs(v["got"]["frac_all_partial"] - v["expected"]["frac_all_partial"]) < 0.0005
                                      for v in info["selfcheck_B"].values())
    print("self-check B:", json.dumps(info["selfcheck_B"], ensure_ascii=False), flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    (out / "sec53_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
    print(f"wrote {out}: rw_coverage_curve.csv ({len(curve_rows)} rows), active_behavior_per_episode.csv ({sum(len(by_model[m]) for m in MODELS)} rows), "
          f"active_behavior_summary.csv, active_behavior_summary_no_trunc.csv, sec53_summary.json; {info['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
