"""tiny subset.

merged_v1.0 (select_merged): 32 groups, 16 each from ARKit and ScanNet++; the result is fixed in llm.tiny_groups of configs/eval.yaml.
  Within each dataset, stratify by (contains masked questions, has constraints), allocate slots proportionally to stratum size (largest remainder), and draw randomly by seed within a stratum, preferring different scenes;
  template stratification: among the seeds meeting the coverage requirements, take the one whose per-dataset template mix (share of answerable T1–T5) is closest to all questions of that dataset (smallest sum of L1 distances).
  Coverage requirements (per dataset): answerable T1–T5 all present (T5 ≥ 3), unanswerable T1 and T4 both present, a masked group present, groups with and without constraints present,
  at least 1 group with group_route_steps > 20. Seed [20261103, s], s in 0–199.

arkit_only_v0.11 (select): stratified draw of 24 of 104 groups, used for pilot runs and prompt tuning (kept to reproduce earlier results).

Strata: (masked or not, constraint type or none); slots allocated proportionally to stratum size (largest remainder), random draw by seed within a stratum, preferring different scenes;
coverage check after drawing: every template (T5 at least 6, from 6 scenes only), every difficulty, a masked group, with / without constraints, T5, answerable and unanswerable T4 must all appear, and at least 1 group with
group_route_steps > 20 (to test long routes); otherwise try the next seed.

    python -m actspace_eval.tiny          # print the selection and coverage statistics
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from actspace_build.gate_d import alloc_quota

from .data import FrozenData
from .meta import group_meta

N_TINY = 24
N_TINY_MERGED = {"arkitscenes": 16, "scannetpp": 16}
SEED_MERGED = 20261103
TEMPLATES = ("T1", "T2", "T3", "T4", "T5")


def stratum(g):
    return (bool(g.get("view_graph_mask")), g["constraints"][0]["type"] if g["constraints"] else "none")


def coverage(data, meta, gids):
    qs = [q for gid in gids for q in data.groups[gid]["questions"]]
    gs = [data.groups[gid] for gid in gids]
    return {"groups": len(gids), "scenes": len({g["scene_id"] for g in gs}), "questions": len(qs),
            "templates": dict(sorted(Counter(q["template"] for q in qs).items())),
            "answerable_by_template": dict(sorted(Counter(q["template"] for q in qs if q["evidence_label"] != "unanswerable").items())),
            "unanswerable_by_template": dict(sorted(Counter(q["template"] for q in qs if q["evidence_label"] == "unanswerable").items())),
            "difficulty": dict(sorted(Counter(q.get("difficulty") or "unanswerable" for q in qs).items())),
            "masked_groups": sum(bool(g.get("view_graph_mask")) for g in gs),
            "constraints": dict(sorted(Counter(stratum(g)[1] for g in gs).items())),
            "group_route_steps": sorted(meta[gid]["group_route_steps"] for gid in gids)}


def ok(cov):
    return (set(cov["answerable_by_template"]) == {"T1", "T2", "T3", "T4", "T5"} and cov["answerable_by_template"]["T5"] >= 6 and {"T1", "T4"} <= set(cov["unanswerable_by_template"])
            and {"easy", "medium", "hard"} <= set(cov["difficulty"]) and cov["masked_groups"] > 0
            and {"none", "numeric_vs_comparison", "direction_inverse"} <= set(cov["constraints"]) and max(cov["group_route_steps"]) > 20)


def select(data, meta, seed0=20261102):
    by = defaultdict(list)
    for gid in data.group_ids:
        by[stratum(data.groups[gid])].append(gid)
    alloc = alloc_quota({k: len(v) for k, v in by.items()}, N_TINY)
    for s in range(100):
        rng = np.random.default_rng([seed0, s])
        picked, scenes = [], Counter()
        for k in sorted(by, key=str):
            cand = list(rng.permutation(sorted(by[k])))
            cand.sort(key=lambda gid: scenes[data.groups[gid]["scene_id"]])  # stable sort: prefer scenes not yet used
            for gid in cand[:alloc[k]]:
                picked.append(gid)
                scenes[data.groups[gid]["scene_id"]] += 1
        picked = sorted(picked)
        if ok(coverage(data, meta, picked)):
            return picked, s, alloc
    raise RuntimeError("none of the 100 seeds meets the coverage requirements")


def stratum_merged(g):
    return (bool(g.get("view_graph_mask")), bool(g["constraints"]))


def template_mix(data, gids):
    c = Counter(q["template"] for gid in gids for q in data.groups[gid]["questions"] if q["evidence_label"] != "unanswerable")
    n = sum(c.values())
    return {t: c[t] / n for t in TEMPLATES}


def ok_merged(data, meta, gids):
    cov = coverage(data, meta, gids)
    return (set(cov["answerable_by_template"]) == set(TEMPLATES) and cov["answerable_by_template"]["T5"] >= 3
            and {"T1", "T4"} <= set(cov["unanswerable_by_template"]) and cov["masked_groups"] > 0
            and "none" in cov["constraints"] and len(cov["constraints"]) >= 2 and max(cov["group_route_steps"]) > 20)


def select_merged(data, meta, seed0=SEED_MERGED, n_seeds=200):
    """tiny set of merged_v1.0: returns (groups, seed index, allocation, template-mix distance)."""
    from .data import dataset_of_scene
    best = None
    pools = {ds: [gid for gid in data.group_ids if dataset_of_scene(data.groups[gid]["scene_id"]) == ds] for ds in N_TINY_MERGED}
    full_mix = {ds: template_mix(data, g) for ds, g in pools.items()}
    allocs = {}
    for ds, pool in pools.items():
        by = defaultdict(list)
        for gid in pool:
            by[stratum_merged(data.groups[gid])].append(gid)
        allocs[ds] = (by, alloc_quota({k: len(v) for k, v in by.items()}, N_TINY_MERGED[ds]))
    for s in range(n_seeds):
        rng = np.random.default_rng([seed0, s])
        picked, dist, good = [], 0.0, True
        for ds, (by, alloc) in allocs.items():
            sub, scenes = [], Counter()
            for k in sorted(by, key=str):
                cand = list(rng.permutation(sorted(by[k])))
                cand.sort(key=lambda gid: scenes[data.groups[gid]["scene_id"]])
                for gid in cand[:alloc[k]]:
                    sub.append(gid)
                    scenes[data.groups[gid]["scene_id"]] += 1
            good &= ok_merged(data, meta, sub)
            mix = template_mix(data, sub)
            dist += sum(abs(mix[t] - full_mix[ds][t]) for t in TEMPLATES)
            picked += sub
        if good and (best is None or dist < best[3] - 1e-12):
            best = (sorted(picked), s, {ds: {str(k): v for k, v in a[1].items()} for ds, a in allocs.items()}, dist)
    if best is None:
        raise RuntimeError(f"none of the {n_seeds} seeds meets the coverage requirements")
    return best


if __name__ == "__main__":
    import json
    from .data import dataset_of_scene
    data = FrozenData(verify=False)
    meta = group_meta(data)
    if data.merged:
        picked, s, alloc, dist = select_merged(data, meta)
        out = {"seed": [SEED_MERGED, s], "alloc": alloc, "template_mix_l1": round(dist, 4), "groups": picked}
        for ds in N_TINY_MERGED:
            sub = [g for g in picked if dataset_of_scene(data.groups[g]["scene_id"]) == ds]
            pool = [g for g in data.group_ids if dataset_of_scene(data.groups[g]["scene_id"]) == ds]
            out[ds] = {"coverage": coverage(data, meta, sub), "template_mix": {k: round(v, 3) for k, v in template_mix(data, sub).items()},
                       "template_mix_full": {k: round(v, 3) for k, v in template_mix(data, pool).items()}}
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        picked, s, alloc = select(data, meta)
        print(json.dumps({"seed_offset": s, "alloc": {str(k): v for k, v in alloc.items()}, "groups": picked,
                          "coverage": coverage(data, meta, picked)}, ensure_ascii=False, indent=1))
