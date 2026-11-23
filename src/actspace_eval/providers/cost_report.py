"""Cost estimates for the registered models on tiny and all. Reads only run logs and the frozen data; calls no API.

    python -m actspace_eval.providers.cost_report --out <json> --md <md>

- OpenAI models: cost_plan: sol, terra and luna are scaled from their own measured usage on v0.11 tiny; astra has no runs and uses
  the usage of sol (high).
- None of the Anthropic models has been run; each is extrapolated from the measured usage of sol and of luna (luna reasons the most
  per step and serves as the reasoning-heavy case):
  * number of Active calls and the output tokens of each setting (reasoning included) are taken from the reference model's scaled
    measurements;
  * input tokens are recomputed from the real prompts: images at the registered 414 tokens each (official formula, not measured);
    text estimated at 4 characters per token and multiplied by TEXT_FACTOR (official pricing page: from Claude 4.7 on, the tokenizer
    produces about 30% more tokens for the same text; the ratio to the OpenAI tokenizer has not been measured);
  * caching: a single breakpoint on the Active START image, written on the first call of each episode and read afterwards; all
    other input counts as uncached;
  * "no cache" column: all input at the uncached price.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..cost import CostTracker, estimate
from ..cost_plan import MODELS, RUNS, log_usage, plan, tables
from ..data import FrozenData
from ..llm_agent import setting_env
from ..llm_run import env_factory
from . import registry as R
from .api_eval import RUN_OF, SETTINGS, repeats, start_prefix_tokens
from .smoke_report import collect, per_episode, recost

TEXT_FACTOR = 1.3
REFS = {"gpt-6.1-sol": "sol", "gpt-6-luna": "luna"}
OPENAI_KEY = {"luna": "gpt-6-luna", "terra": "gpt-6.1-terra", "sol": "gpt-6.1-sol", "astra": "gpt-6-astra"}
SETS = {"tiny32": "tiny", "all234": "all"}


def claude_run_cost(data, mcfg, gids, setting, steps, out_tokens):
    """(cost, cost without caching, details) of one run (for random_walk, the average over three seeds)."""
    ec = data.ecfg
    mode, budget = RUN_OF[setting]
    fac = env_factory(data)
    seeds = (0, 1, 2) if setting == "random_walk_40" else (0,)
    r = estimate(fac, gids, [(mode, budget, steps if setting_env(mode)[0] == "active" else budget, seeds)], mcfg, ec["llm"]["history"], ec["env"]["step_scale"])[0]
    tpi, n = mcfg["tokens_per_image"], len(seeds)
    text = r["input_tokens"] - r["images"] * tpi
    inp = (text * TEXT_FACTOR + r["images"] * tpi) / n
    read = write = 0.0
    if setting_env(mode)[0] == "active":
        env = fac(mode, budget, 0)
        pre = sum((start_prefix_tokens(env, g, tpi, budget, ec["llm"]["history"], data.step_scale(g)) - tpi) * TEXT_FACTOR + tpi for g in gids)
        write, read = pre, pre * (r["calls"] / len(gids) - 1)
    t = CostTracker(mcfg["prices_usd_per_mtok"], float("inf"))
    use = {"input_tokens": inp, "cached_input_tokens": read, "cache_write_tokens": write, "output_tokens": out_tokens}
    return t.cost_of(use), t.cost_of({"input_tokens": inp, "output_tokens": out_tokens}), {**use, "calls": r["calls"] / n, "images": r["images"] / n}


def build() -> dict:
    Pl = plan()
    T = tables(Pl)
    data = FrozenData(verify=False)
    reg = R.load_registry()
    R.merge(data, reg)
    root = Path(data.ecfg["paths"]["eval_runs"])
    steps = {ref: round(log_usage(root, RUNS[ref]["active_v2"])["per_group"]["calls"]) - 1 for ref in REFS}
    outs = {(r["model"], r["setting"], r["set"]): r["per_run"]["output"] for r in Pl["rows"]}
    sets = {"tiny32": list(data.ecfg["llm"]["tiny_groups"]), "all234": list(data.group_ids)}
    res = {"active_steps_assumed": steps, "text_factor": TEXT_FACTOR, "openai": {}, "anthropic": {}}
    for short, key in OPENAI_KEY.items():
        # tables assume 3 repeats: convert to per run, then multiply by the actual repeats
        res["openai"][short] = {k: {s: {"standard": v[0] / v[2] * repeats(s), "upper": v[1] / v[2] * repeats(s), "runs": repeats(s)}
                                    for s, v in T[f"{k}|A"][key].items()} for k in sets}
    for short, e in reg["models"].items():
        if e["provider"] != "anthropic":
            continue
        mcfg = data.ecfg["models"][short]
        res["anthropic"][short] = {}
        for k, gids in sets.items():
            cell = {}
            for s in SETTINGS:
                reps = repeats(s)
                c = {"runs": reps}
                for ref, name in REFS.items():
                    cost, nocache, use = claude_run_cost(data, mcfg, gids, s, steps[ref], outs[(ref, s, k)])
                    c[name] = cost * reps
                    c[f"{name}_no_cache"] = nocache * reps
                    c[f"{name}_detail"] = use
                cell[s] = c
            res["anthropic"][short][k] = cell
    return res


def markdown(res) -> list[str]:
    L = []
    for k, lab in (("tiny32", "tiny, 32 groups"), ("all234", "all, 234 groups")):
        L += [f"**{lab}, OpenAI** (USD; each cell standard / upper bound; Active x{repeats('active_v2')}, random_walk x{repeats('random_walk_40')}, already multiplied in)", "",
              "| Setting | " + " | ".join(res["openai"]) + " |", "|---|" + "---|" * len(res["openai"])]
        for s in SETTINGS:
            L.append(f"| {s} | " + " | ".join(f"{res['openai'][m][k][s]['standard']:,.2f} / {res['openai'][m][k][s]['upper']:,.2f}" for m in res["openai"]) + " |")
        L.append("| **Total** | " + " | ".join(f"**{sum(v['standard'] for v in res['openai'][m][k].values()):,.2f} / {sum(v['upper'] for v in res['openai'][m][k].values()):,.2f}**" for m in res["openai"]) + " |")
        L += ["", f"**{lab}, Anthropic** (USD; each cell with sol usage / with luna usage; Active x{repeats('active_v2')}, random_walk x{repeats('random_walk_40')}, already multiplied in)", "",
              "| Setting | " + " | ".join(res["anthropic"]) + " |", "|---|" + "---|" * len(res["anthropic"])]
        for s in SETTINGS:
            L.append(f"| {s} | " + " | ".join(f"{res['anthropic'][m][k][s]['sol']:,.2f} / {res['anthropic'][m][k][s]['luna']:,.2f}" for m in res["anthropic"]) + " |")
        tot = lambda m, f: sum(v[f] for v in res["anthropic"][m][k].values())
        L.append("| **Total** | " + " | ".join(f"**{tot(m, 'sol'):,.2f} / {tot(m, 'luna'):,.2f}**" for m in res["anthropic"]) + " |")
        L.append("| Total (no cache) | " + " | ".join(f"{tot(m, 'sol_no_cache'):,.2f} / {tot(m, 'luna_no_cache'):,.2f}" for m in res["anthropic"]) + " |")
        L.append("")
    return L


def from_smoke(data, reg, ref_of: dict, prefix="smoke") -> list[dict]:
    """Estimates all from the reference model's smoke measurements: tokens and steps from the reference model, prices from this model;
    time = total latency of all calls / concurrency within one job. Active is computed both with the measured steps and with all
    40 steps used. ref_of: {short name: reference model short name}."""
    root = R.results_root(reg)
    gids = list(data.group_ids)
    measured = {ref: collect(root, ref, prefix) for ref in set(ref_of.values())}
    rows = []
    for short, ref in ref_of.items():
        runs = measured[ref]
        rec = recost(data, short, runs, {"all": gids})["all"]
        conc = R.entries(reg)[short]["concurrency"]["episodes"]
        work = full = 0.0
        for s in SETTINGS:
            m = per_episode(runs[s])
            lat = float(np.mean([c["latency_s"] for r in runs[s] for c in r["calls"]]))
            eps = len(gids) * repeats(s)
            work += eps * m["calls"] * lat
            full += eps * (41 if s == "active_v2" else m["calls"]) * lat
        rows.append({"model": short, "ref": ref, "concurrency": conc, "cost": sum(v[0] for v in rec.values()), "cost_40": sum(v[1] for v in rec.values()),
                     "hours": work / conc / 3600, "hours_40": full / conc / 3600})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--from-smoke", action="store_true", help="estimate all for the other models from the smoke measurements of haiku-5.5 and luna")
    a = ap.parse_args()
    if a.from_smoke:
        data = FrozenData(verify=False)
        reg = R.load_registry()
        R.merge(data, reg)
        ref = {k: ("haiku-5.5" if v["provider"] == "anthropic" else "luna") for k, v in reg["models"].items()}
        rows = from_smoke(data, reg, ref)
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        L = ["| Model | Reference | Concurrency | Cost (measured steps) | Cost (Active uses all 40 steps) | Time h (measured steps) | Time h (40 steps) |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r['model']} | {r['ref']} | {r['concurrency']} | ${r['cost']:,.2f} | ${r['cost_40']:,.2f} | {r['hours']:.1f} | {r['hours_40']:.1f} |" for r in rows]
        Path(a.md).write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    res = build()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    Path(a.md).write_text("\n".join(markdown(res)) + "\n")
    print("\n".join(markdown(res)))


if __name__ == "__main__":
    main()
