"""ScanNet++ Steps 5-6: object clusters, initial views, template questions (reuses the shared step56_questions.run; same rules as ARKit).

- Clusters are drawn only from referable, non-blacklisted objects that are visible at >= 1 node (groups.cluster_require_visible_anywhere);
  scenes with <= 6 referable objects use the small-scene rule and sample 3-object clusters.
- Evidence-linking mode: the main result follows the current ARKit mode (when scannetpp.evidence_link_mode is null, the shared
  questions.evidence_link_mode is used, currently partial_partial); each mode in scannetpp.evidence_link_compare_modes is also run for
  comparison, written to _step6_<mode>/, and is not an official result.
- Option-letter and quadrant balancing are done over the 5 pilot scenes (run processes all scenes at once). The pilot is small: when balance
  cannot be reached, report it as is rather than dropping many questions to force balance.
  Option-letter balancing shuffles order and drops no questions; T3/T4 quadrant balancing drops questions (shared rule
  questions.quadrant_balance_tol). The main pilot result uses scannetpp.quadrant_balance_tol (null = drop nothing, report the deviation only);
  the full run also skips drop-based balancing, which is deferred to the Step 8 rerun after merging with ARKit.
  When scannetpp.compare_shared_quadrant_rule is true, the shared rule is also run for comparison.
- T2 deduplication: within a scene, by reference object + comparison direction + correct answer (scannetpp.t2_dedup_by_answer).

Usage: python -m actspace_build.scannetpp_step56 [--scenes ...]
Outputs: work/scannetpp/_step6/test_groups.jsonl, step6_stats.json; per scene groups_raw.json (groups_raw_<mode>.json for comparison modes)
"""
from __future__ import annotations

import argparse
import copy

from .common import load_config, path_of, write_json
from .scannetpp import sp_cfg
from .step56_questions import run

DATASET = "scannetpp"


def primary_mode(cfg: dict) -> str:
    return sp_cfg(cfg)["evidence_link_mode"] or cfg["questions"]["evidence_link_mode"]


def config_for_mode(cfg: dict, mode: str, quadrant_tol="pilot") -> dict:
    """quadrant_tol="pilot": use scannetpp.quadrant_balance_tol (null -> 1.0, i.e. never drop questions for quadrant balance); "shared": use the shared value."""
    c = copy.deepcopy(cfg)
    c["questions"]["evidence_link_mode"] = mode
    if quadrant_tol == "pilot":
        t = sp_cfg(cfg).get("quadrant_balance_tol")
        c["questions"]["quadrant_balance_tol"] = 1.0 if t is None else t
    # Within a scene, deduplicate T2 by reference object + comparison direction + correct answer
    c["questions"]["t2_dedup_by_answer"] = sp_cfg(cfg).get("t2_dedup_by_answer", False)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    scenes = sorted(args.scenes or sp_cfg(cfg)["pilot_scenes"])
    prim = primary_mode(cfg)
    # (name, evidence-linking mode, quadrant tolerance): comparisons first, main result last (step56 records in each scene's dropped.jsonl reflect the main result)
    variants = [(m, m, "pilot") for m in sp_cfg(cfg)["evidence_link_compare_modes"] if m != prim]
    if sp_cfg(cfg).get("compare_shared_quadrant_rule"):
        variants.append((f"{prim}_shared_quadrant_rule", prim, "shared"))
    variants.append((prim, prim, "pilot"))
    summary = {}
    for name, mode, qtol in variants:
        is_prim = name == prim
        out_dir = path_of(cfg, "work") / DATASET / ("_step6" if is_prim else f"_step6_{name}")
        c = config_for_mode(cfg, mode, qtol)
        st = run(c, DATASET, scenes, out_dir=out_dir, raw_name="groups_raw.json" if is_prim else f"groups_raw_{name}.json")
        summary[name] = {"primary": is_prim, "evidence_link_mode": mode, "quadrant_balance_tol": c["questions"]["quadrant_balance_tol"],
                         "num_groups": st["num_groups"], "num_questions": st["num_questions"], "per_template": st["per_template"],
                         "quadrant_rebalance_removed": st["quadrant_rebalance_removed"]}
        print(name, summary[name], flush=True)
    write_json(path_of(cfg, "work") / DATASET / "_step6" / "evidence_link_modes.json", summary)


if __name__ == "__main__":
    main()
