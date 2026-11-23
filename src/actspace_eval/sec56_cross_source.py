"""Section 5.6 supplement: consistency of model rankings between the two datasets. Reads only each run's existing per-question scores
(summary.json); no model is called.

    python -m actspace_eval.sec56_cross_source [--out results/_reports/sec56]

For the 17 models of the main table, take on ARKitScenes and on ScanNet++ separately the Active total score, the random_walk_40 total score,
the oracle_route total score and ΔRW = Active − random_walk_40,
and compute Kendall τ (τ-b) and Spearman ρ between the two datasets. Total score = main-table definition (excluding the 7 budget_infeasible
groups), mean of per-question scores; with several runs each question counts once per run
(same point estimate as the score tables and paper_table1_api); closed-source values are checked cell by cell against paper_table1_api.csv.
Runs: closed-source models use their main runs (haiku-5.5 has 3 Active and 3 random_walk_40 runs); open-source models use active_v2/full_r0,
random_walk_40/full_s0 and oracle_route/full_r0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .data import dataset_of_scene, load_eval_config
from .sec53_behavior import CLOSED, MODELS, OPEN, ROOT, RUNS

DATASETS = (("arkitscenes", "ARKitScenes"), ("scannetpp", "ScanNet++"))
METRICS = (("active", "Active total"), ("rw", "random_walk_40 total"), ("oracle_route", "oracle_route total"), ("d_rw", "ΔRW"))
AFFECTED = ("Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-31B-it")


def run_names(model: str, setting: str) -> list[str]:
    base, act, rw = RUNS[model]
    if setting == "active_v2":
        return [f"active_v2/{r}" for r in act]
    if setting == "random_walk_40":
        return [f"random_walk_40/{r}" for r in rw]
    return [f"oracle_route/{'full_r0' if model in OPEN else act[0]}"]


def mean_score(model: str, setting: str, bad: set) -> dict:
    """{dataset: mean per-question score}."""
    v = {ds: [] for ds, _ in DATASETS}
    for name in run_names(model, setting):
        for x in json.loads((RUNS[model][0] / name / "summary.json").read_text())["per_question"]:
            if x["group_id"] not in bad:
                v[dataset_of_scene(x["scene_id"])].append(x["score"])
    return {ds: float(np.mean(s)) for ds, s in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_reports" / "sec56"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = set(load_eval_config()["env"].get("budget_infeasible") or [])
    val = {}
    for m in MODELS:
        act, rw, orc = (mean_score(m, s, bad) for s in ("active_v2", "random_walk_40", "oracle_route"))
        val[m] = {ds: {"active": act[ds], "rw": rw[ds], "oracle_route": orc[ds], "d_rw": act[ds] - rw[ds]} for ds, _ in DATASETS}
    # Check against the paper's main table (closed-source models, percentages with 1 decimal)
    ref, lab = {}, dict(DATASETS)
    p = ROOT / "api" / "_reports" / "paper_table1_api.csv"
    for r in csv.DictReader(open(p)):
        ref[(r["scope"], r["model"])] = r
    mism = [(m, ds, k) for m in CLOSED for ds, _ in DATASETS for k in ("active", "rw", "oracle_route", "d_rw")
            if f"{100 * val[m][ds][k]:.1f}" != ref[(lab[ds], m)][k]]
    rows = []
    for subset, models in (("all 17 models", MODELS), ("excluding the 3 models affected by truncation or overflow", [m for m in MODELS if m not in AFFECTED])):
        for k, name in METRICS:
            x = [val[m]["arkitscenes"][k] for m in models]
            y = [val[m]["scannetpp"][k] for m in models]
            kt, sp = kendalltau(x, y), spearmanr(x, y)
            rows.append({"row": "correlation", "subset": subset, "n_models": len(models), "metric": k, "metric_label": name,
                         "kendall_tau": round(float(kt.statistic), 4), "kendall_p": float(f"{kt.pvalue:.3g}"),
                         "spearman_rho": round(float(sp.statistic), 4), "spearman_p": float(f"{sp.pvalue:.3g}"),
                         "model": "", "arkitscenes": "", "scannetpp": "", "rank_arkitscenes": "", "rank_scannetpp": ""})
    for k, name in METRICS:
        ra = {m: i for i, m in enumerate(sorted(MODELS, key=lambda m: -val[m]["arkitscenes"][k]), 1)}
        rs = {m: i for i, m in enumerate(sorted(MODELS, key=lambda m: -val[m]["scannetpp"][k]), 1)}
        for m in MODELS:
            rows.append({"row": "model", "subset": "", "n_models": "", "metric": k, "metric_label": name, "kendall_tau": "", "kendall_p": "", "spearman_rho": "", "spearman_p": "",
                         "model": m, "arkitscenes": round(100 * val[m]["arkitscenes"][k], 3), "scannetpp": round(100 * val[m]["scannetpp"][k], 3),
                         "rank_arkitscenes": ra[m], "rank_scannetpp": rs[m]})
    with open(out / "cross_source_rank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'cross_source_rank.csv'} ({len(rows)} rows); cells disagreeing with paper_table1_api.csv: {len(mism)} {mism[:5]}")
    for r in rows:
        if r["row"] == "correlation":
            print(r["subset"], r["metric"], r["n_models"], r["kendall_tau"], r["kendall_p"], r["spearman_rho"], r["spearman_p"])


if __name__ == "__main__":
    main()
