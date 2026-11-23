"""Cost estimate for the official evaluation (merged_v1.0). Reads run logs only, calls no API.

    python -m actspace_eval.cost_plan --out <json>

Basis: measured usage of each model on tiny (arkit_only_v0.11, 24 groups) (including reasoning tokens; cache hits and writes from the logs),
scaled to the new tiny set of merged_v1.0 (32 groups) and the full 234 groups.
- Input tokens: within a setting all models receive the same messages (static settings have byte-identical measured inputs), so the setting's measured value is used; Active step counts depend on the model, so the model's own measurement is used.
  Scaling from the old tiny set to a new group set multiplies by the ratio of "estimated input" (cost.estimate_episode estimates from the real prompts and photo counts; per group for static settings, per call for Active).
- Output tokens: measured for that model and setting (per group; Active includes every step and the answer). Combinations without measurements:
  * sol has only medium runs for static settings: reasoning tokens are multiplied by sol's own high / medium reasoning ratio on the Active answer step;
  * Passive-8, Passive-20, oracle_min, oracle_route of terra / luna: use the per-call output of that model's Passive-40;
  * astra has no runs: assume the same token counts as sol (high), billed at astra's prices.
  Output is scaled by the ratio of questions per group (answer length varies with the number of questions).
- Cache: hit ratio measured per setting (static settings barely hit; Active uses the model's measurement, astra uses sol's); cache writes use the "writes / uncached input" ratio measured from the
  raw usage in the luna / terra logs (static and Active separately; sol / astra use the mean of luna and terra).
- Two columns: standard price (with the hits and writes above); upper bound "all as cache writes" (all input at the 30-minute write price, no hit discount).
- Plans: A = every model runs every setting, Active and random_walk 3 times each; B = astra's Active runs only once, otherwise as A.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cost import CostTracker, call_usage
from .data import FrozenData
from .llm_run import env_factory
from .cost import estimate_episode

MODELS = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra")
SETTINGS = (("blind", "blind", 40, 1), ("passive_8", "passive", 8, 1), ("passive_20", "passive", 20, 1), ("passive_40", "passive", 40, 1),
            ("random_walk_40", "random_walk", 40, 3), ("oracle_min", "oracle_min", 40, 1), ("oracle_route", "oracle_route", 40, 1),
            ("active_v2", "active_v2", 40, 3))
STATIC = {"blind", "passive_8", "passive_20", "passive_40", "random_walk_40", "oracle_min", "oracle_route"}
# measured runs (sol's static settings are medium, Active is high)
RUNS = {
    "gpt-6-luna": {"blind": ["s3m_luna_blind"], "passive_40": ["s3m_luna_passive_b40"],
                   "random_walk_40": [f"s3m_luna_random_walk_40_s{k}" for k in range(3)], "active_v2": ["s3m_luna_active_v2_b40"]},
    "gpt-6.1-terra": {"blind": ["s3m_terra_blind"], "passive_40": ["s3m_terra_passive_b40"],
                      "random_walk_40": [f"s3m_terra_random_walk_40_s{k}" for k in range(3)], "active_v2": ["s3m_terra_active_v2_b40"]},
    "gpt-6.1-sol": {"blind": ["s3_sol_tiny_blind"], "passive_8": ["s3r_sol_passive_b8"], "passive_20": ["s3r_sol_passive_b20"],
                    "passive_40": ["s3_sol_tiny_passive_b40"], "random_walk_40": [f"s3r_sol_random_walk_40_s{k}" for k in range(3)],
                    "oracle_min": ["s3_sol_tiny_oracle"], "oracle_route": ["s3r_sol_oracle_route"], "active_v2": ["s3r_sol_high_active_v2_b40"]},
}
SOL_MEDIUM_STATIC = {"blind", "passive_8", "passive_20", "passive_40", "random_walk_40", "oracle_min", "oracle_route"}
SOL_ACTIVE_MEDIUM = "s3r_sol_active_v2_b40"


def log_usage(root: Path, names) -> dict:
    """Total usage and per-group mean of the run logs (including cache writes; writes in old logs without raw usage are unknown)."""
    tot = defaultdict(float)
    eps = set()
    for n in names:
        for line in open(root / n / "calls.jsonl"):
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            u = call_usage(r)
            raw = r.get("usage_raw") is not None
            tot["calls"] += 1
            for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
                tot[k] += u.get(k) or 0
            if raw:
                tot["raw_input"] += u["input_tokens"] - u["cached_input_tokens"]
                tot["raw_write"] += u.get("cache_write_tokens") or 0
            eps.add((n, r["tag"].get("episode")))
    n_eps = len(eps)
    return {**tot, "episodes": n_eps, "per_group": {k: tot[k] / n_eps for k in ("calls", "input_tokens", "cached_input_tokens",
                                                                                 "output_tokens", "reasoning_tokens")}}


def answer_reasoning(root: Path, name: str) -> float:
    """Mean reasoning tokens of the answer step (last call of each episode) in Active runs."""
    last = {}
    for line in open(root / name / "calls.jsonl"):
        r = json.loads(line)
        if r.get("type") == "call":
            ep, t = r["tag"]["episode"], r["tag"]["t"]
            if ep not in last or t > last[ep][0]:
                last[ep] = (t, r["usage"]["reasoning_tokens"])
    return float(np.mean([v[1] for v in last.values()]))


def est_per_group(data, gids, setting, mode, budget, seeds, mcfg, history, scale, per_call=False):
    fac = env_factory(data)
    vals = []
    for s in seeds:
        for g in gids:
            e = estimate_episode(fac, g, mode, budget, mcfg, history, budget, s, scale)
            vals.append(e["input_tokens"] / e["calls"] if per_call else e["input_tokens"])
    return float(np.mean(vals))


def plan(cfg_root=None) -> dict:
    new = FrozenData(verify=False)
    old = FrozenData({**new.ecfg, "version": "arkit_only_v0.11"}, verify=False)
    ec = new.ecfg
    root = Path(ec["paths"]["eval_runs"])
    hist = ec["llm"]["history"]
    mref = ec["models"]["gpt-6.1-sol"]   # estimated input is only used for ratios; tokens_per_image is 361 for every model
    old_tiny = [e["group_id"] for e in json.loads((root / "s3_sol_tiny_blind" / "summary.json").read_text())["per_episode"]]
    sets = {"tiny32": list(ec["llm"]["tiny_groups"]), "all234": list(new.group_ids)}
    q_per_group = {"old": np.mean([len(old.groups[g]["questions"]) for g in old_tiny]),
                   **{k: np.mean([len(new.groups[g]["questions"]) for g in v]) for k, v in sets.items()}}
    # ratio of estimated input (new group set / old tiny)
    ratio = {}
    for name, mode, budget, reps in SETTINGS:
        seeds = (0, 1, 2) if mode == "random_walk" else (0,)
        per_call = mode == "active_v2"
        o = est_per_group(old, old_tiny, name, mode, budget, seeds, mref, hist if per_call else None, ec["env"]["step_scale"], per_call)
        for k, gids in sets.items():
            ratio[(name, k)] = est_per_group(new, gids, name, mode, budget, seeds, mref, hist if per_call else None,
                                             ec["env"]["step_scale"], per_call) / o
    # measured
    meas = {m: {s: log_usage(root, names) for s, names in d.items()} for m, d in RUNS.items()}
    r_high = answer_reasoning(root, "s3r_sol_high_active_v2_b40") / answer_reasoning(root, SOL_ACTIVE_MEDIUM)
    wf = {}
    for kind in ("static", "active"):
        fr = []
        for m in ("gpt-6-luna", "gpt-6.1-terra"):
            ss = [s for s in meas[m] if (s in STATIC) == (kind == "static") and s != "blind"]
            ri = sum(meas[m][s]["raw_input"] for s in ss)
            rw = sum(meas[m][s]["raw_write"] for s in ss)
            wf[(m, kind)] = rw / ri if ri else 0.0
            fr.append(wf[(m, kind)])
        wf[("gpt-6.1-sol", kind)] = wf[("gpt-6-astra", kind)] = float(np.mean(fr))
    wf_blind = {m: 0.0 for m in MODELS}   # Blind requests are below 1024 tokens and write no cache (measured 0 for luna / terra)
    rows = []
    for m in MODELS:
        src = "gpt-6.1-sol" if m == "gpt-6-astra" else m
        prices = ec["models"][m]["prices_usd_per_mtok"]
        for name, mode, budget, reps in SETTINGS:
            # per-group (input, cached, output) on the old tiny set
            if name in meas[src]:
                u = meas[src][name]["per_group"]
                inp, cached, out, reas = u["input_tokens"], u["cached_input_tokens"], u["output_tokens"], u["reasoning_tokens"]
                note = "measured" if m == src else "usage of sol (high)"
                if src == "gpt-6.1-sol" and name in SOL_MEDIUM_STATIC:
                    out = out - reas + reas * r_high
                    note += f"; sol static settings are medium, reasoning ×{r_high:.2f} scaled to high"
            else:   # static settings terra / luna never ran: input from sol's measurement, output from this model's per-call Passive-40 output
                u = meas["gpt-6.1-sol"][name]["per_group"]
                inp, cached = u["input_tokens"], u["cached_input_tokens"]
                p40 = meas[src]["passive_40"]["per_group"]
                out = p40["output_tokens"] / p40["calls"] * u["calls"]
                note = "input from sol's measurement; output from this model's Passive-40"
            for k in sets:
                f_in = ratio[(name, k)]
                f_out = q_per_group[k] / q_per_group["old"]
                i_g = inp * f_in   # Active: f_in is the per-call ratio of estimated input; the call count follows the model's measurement
                c_g = cached * f_in
                o_g = out * f_out
                kind = "active" if mode == "active_v2" else "static"
                w = 0.0 if name == "blind" else wf[(m, kind)]
                writes = w * (i_g - c_g)
                n = len(sets[k])
                std = CostTracker(prices, float("inf")).cost_of({"input_tokens": i_g, "cached_input_tokens": c_g,
                                                                  "cache_write_tokens": writes, "output_tokens": o_g}) * n
                upper = (i_g * prices["cache_write"] + o_g * prices["output"]) / 1e6 * n
                rows.append({"model": m, "setting": name, "set": k, "groups": n, "repeats": reps, "note": note,
                             "per_run": {"input": i_g * n, "cached": c_g * n, "writes": writes * n, "output": o_g * n,
                                         "standard_usd": std, "upper_usd": upper}})
    return {"rows": rows, "ratios": {f"{a}|{b}": v for (a, b), v in ratio.items()}, "write_frac": {f"{a}|{b}": v for (a, b), v in wf.items()},
            "sol_answer_reasoning_high_over_medium": r_high, "questions_per_group": q_per_group,
            "old_tiny_groups": len(old_tiny)}


def tables(P) -> dict:
    """{(set, plan): {model: {setting: (standard, upper, runs)}}} and totals."""
    out = {}
    for k in ("tiny32", "all234"):
        for pl in ("A", "B"):
            t = defaultdict(dict)
            for r in P["rows"]:
                if r["set"] != k:
                    continue
                reps = r["repeats"]
                if pl == "B" and r["model"] == "gpt-6-astra" and r["setting"] == "active_v2":
                    reps = 1
                t[r["model"]][r["setting"]] = (r["per_run"]["standard_usd"] * reps, r["per_run"]["upper_usd"] * reps, reps)
            out[f"{k}|{pl}"] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = plan()
    T = tables(P)
    P["tables"] = {k: {m: {s: list(v) for s, v in d.items()} for m, d in t.items()} for k, t in T.items()}
    Path(args.out).write_text(json.dumps(P, ensure_ascii=False, indent=1, default=float))
    for key, t in T.items():
        tot = {m: (sum(v[0] for v in d.values()), sum(v[1] for v in d.values())) for m, d in t.items()}
        print(key, {m: (round(a, 2), round(b, 2)) for m, (a, b) in tot.items()}, "sum", round(sum(a for a, _ in tot.values()), 2),
              round(sum(b for _, b in tot.values()), 2))


if __name__ == "__main__":
    main()
