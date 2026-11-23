"""Smoke analysis report. Reads only run artifacts and the frozen data; calls no API. It reads a lot of data, so run it on a compute node:

    python -m actspace_eval.providers.smoke_report <short name> [--prefix smoke] [--out <md>]

Writes reports/eval/api_smoke_<short name>.md: the probe checks; the model field actually returned; measured input tokens per
image vs. the registered value; whether cache_read > 0 from Active step 2 on; median and maximum input / cache-read / cache-write /
output / thinking tokens per call; truncation rate, refusals, invalid replies, server errors and possibly_billed counts; actual vs.
expected smoke cost; tiny and all costs recomputed from the measurements.

Recomputation: for each setting, take the measured per-group usage in smoke (failed but billed attempts included). Input is scaled
by the "ratio of estimated input" (estimated per-group input of the target group set vs. the smoke groups, with Active using the
measured number of steps); for one-shot answering settings, output is scaled by the ratio of questions per group; Active output is
not scaled. Active additionally gets "uses all 40 steps": scaled up by the ratio of 41 calls to the measured number of calls.
Smoke has only 3 groups, so each setting's sample is very small; the tiny / all recomputation is only closer to reality than an
extrapolation without measurements, not an exact value.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..cost import TRUNCATED_FINISH, CostTracker, estimate
from ..data import FrozenData
from ..llm_agent import setting_env
from ..llm_run import env_factory, group_set
from . import registry as R
from .api_eval import RUN_OF, SETTINGS, conditions_md, existing_runs, repeats, run_estimate, scope_spent, thinking_md, thinking_tokens
from .base import REFUSED

FIELDS = (("input_tokens", "Input"), ("cached_input_tokens", "Cache read"), ("cache_write_tokens", "Cache write"), ("output_tokens", "Output"), ("reasoning_tokens", "Thinking"))


def read_run(path: Path) -> dict:
    """Call records of one run -> lists used for statistics."""
    calls, errors = [], []
    p = path / "calls.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                (calls if r.get("type") == "call" else errors).append(r)
    st = json.loads((path / "status.json").read_text()) if (path / "status.json").exists() else {}
    return {"calls": calls, "errors": errors, "status": st.get("status"), "spent": CostTracker.spent_in_log(p)}


def collect(root: Path, short: str, prefix: str) -> dict:
    return {s: [read_run(d) for d in existing_runs(root, short, prefix, s)] for s in SETTINGS}


def stat_rows(runs: dict) -> list[tuple[str, list]]:
    """(row name, call records): one row per setting, with Active split into moves and answers."""
    out = []
    for s, rs in runs.items():
        cs = [c for r in rs for c in r["calls"]]
        if s == "active_v2":
            out.append(("active_v2 (move)", [c for c in cs if (c.get("parse") or {}).get("action") != "answer"]))
            out.append(("active_v2 (answer)", [c for c in cs if (c.get("parse") or {}).get("action") == "answer"]))
        else:
            out.append((s, cs))
    return out + [("All", [c for rs in runs.values() for r in rs for c in r["calls"]])]


def med_max(cs, field) -> str:
    v = [c["usage"].get(field) or 0 for c in cs]
    return f"{int(np.median(v)):,} / {max(v):,}" if v else "—"


def counts(runs: dict) -> dict:
    cs = [c for rs in runs.values() for r in rs for c in r["calls"]]
    es = [e for rs in runs.values() for r in rs for e in r["errors"]]
    trunc = sum(c.get("finish_reason") in TRUNCATED_FINISH for c in cs)
    return {"calls": len(cs), "truncated": trunc, "truncation_rate": trunc / len(cs) if cs else None,
            "refused": sum(c.get("finish_reason") in REFUSED for c in cs),
            "invalid": sum(bool((c.get("parse") or {}).get("error")) for c in cs),
            "service_errors": len(es), "error_status": Counter(str(e.get("status")) for e in es),
            "possibly_billed": sum(bool(e.get("possibly_billed")) for e in es), "billed_retries": sum(bool(e.get("usage")) for e in es),
            "finish": Counter(str(c.get("finish_reason")) for c in cs), "models": sorted({str(c.get("model_returned")) for c in cs})}


def per_episode(rs) -> dict | None:
    """Average per-group usage of this setting (averaged over all runs; failed but billed attempts included)."""
    tot, eps, calls = defaultdict(float), set(), 0
    for i, r in enumerate(rs):
        for c in r["calls"] + [e for e in r["errors"] if e.get("usage")]:
            for k, _ in FIELDS:
                tot[k] += c["usage"].get(k) or 0
            if c.get("type") == "call":
                calls += 1
                eps.add((i, (c.get("tag") or {}).get("episode")))
    if not eps:
        return None
    return {**{k: tot[k] / len(eps) for k, _ in FIELDS}, "calls": calls / len(eps), "episodes": len(eps)}


def est_input(data, mcfg, gids, setting, steps) -> float:
    ec = data.ecfg
    mode, budget = RUN_OF[setting]
    seeds = (0, 1, 2) if setting == "random_walk_40" else (0,)
    r = estimate(env_factory(data), gids, [(mode, budget, steps, seeds)], mcfg, ec["llm"]["history"], ec["env"]["step_scale"])[0]
    return r["input_tokens"] / (len(gids) * len(seeds))


def recost(data, short: str, runs: dict, targets: dict) -> dict:
    """{group set name: {setting: (cost, cost with Active using all 40 steps)}}, multiplied by the number of repeats."""
    mcfg = data.ecfg["models"][short]
    t = CostTracker(mcfg["prices_usd_per_mtok"], float("inf"))
    smoke = group_set(data, "smoke")
    qn = lambda gs: float(np.mean([len(data.groups[g]["questions"]) for g in gs]))
    out = {}
    for name, gids in targets.items():
        cell = {}
        for s in SETTINGS:
            m = per_episode(runs[s])
            if m is None:
                continue
            active = setting_env(RUN_OF[s][0])[0] == "active"
            steps = max(0, round(m["calls"]) - 1) if active else RUN_OF[s][1]
            f = est_input(data, mcfg, gids, s, steps) / est_input(data, mcfg, smoke, s, steps)
            use = {k: m[k] * f for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens")}
            use["output_tokens"] = m["output_tokens"] * (1.0 if active else qn(gids) / qn(smoke))
            c = t.cost_of(use) * len(gids) * repeats(s)
            cell[s] = (c, c * 41 / m["calls"] if active else c)
        out[name] = cell
    return out


def render(short, prefix, entry, probe, runs, est, spent, rec) -> list[str]:
    c = counts(runs)
    img = entry["image"]["tokens_per_image"]
    L = [f"# {short} smoke analysis ({entry['model_id']}, prefix {prefix})", "",
         "Generated by `python -m actspace_eval.providers.smoke_report`. Smoke has 3 groups; the sample is very small.", "", "## 1. probe", ""]
    if probe is None:
        L += ["No probe.json.", ""]
    else:
        L += ["| Check | Result |", "|---|---|"] + [f"| {k} | {'pass' if v else '**fail**'} |" for k, v in (probe.get("checks") or {}).items()]
        L += [f"| Overall | {'pass' if probe.get('passed') else '**fail**'} |", "",
              f"- Request parameters: `{json.dumps(probe.get('params'), ensure_ascii=False)}`",
              f"- Thinking tokens (A / B / C): {probe.get('reasoning_tokens')}; thinking summary characters: {probe.get('reasoning_chars')}",
              f"- Cache: B wrote {probe.get('cache_write_on_first')}, C read {probe.get('cache_read_on_repeat')}", ""]
    tpi = probe.get("tokens_per_image") if probe else None
    L += ["## 2. Model and images", "",
          f"- Registered model ID: `{entry['model_id']}`; probe returned: {sorted(set(probe.get('model_returned') or [])) if probe else '—'}; returned during runs: {c['models']}",
          f"- Measured input tokens per image: {tpi}; registered value {img}" + (f" (difference {tpi - img:+d}, {(tpi - img) / img:+.1%})" if tpi is not None else ""), ""]
    act = [x for r in runs["active_v2"] for x in r["calls"]]
    later = [x for x in act if ((x.get("tag") or {}).get("t") or 0) >= 1]
    hit = sum((x["usage"].get("cached_input_tokens") or 0) > 0 for x in later)
    L += ["## 3. Cache", "", f"- Active from step 2 on, cache_read > 0: {hit} / {len(later)} calls" + ("" if not later else f" ({hit / len(later):.1%})"),
          f"- Input over all Active calls {sum(x['usage'].get('input_tokens') or 0 for x in act):,}, of which cache read {sum(x['usage'].get('cached_input_tokens') or 0 for x in act):,}, "
          f"cache write {sum(x['usage'].get('cache_write_tokens') or 0 for x in act):,}", "",
          "## 4. Tokens per call (median / maximum)", "",
          "| Setting | Calls | " + " | ".join(n for _, n in FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
    for name, cs in stat_rows(runs):
        L.append(f"| {name} | {len(cs)} | " + " | ".join(med_max(cs, k) for k, _ in FIELDS) + " |")
    rate = "—" if c["truncation_rate"] is None else f"{c['truncation_rate']:.2%}"
    L += ["", "## 5. Replies and errors", "", "| Item | Count |", "|---|---|", f"| Calls | {c['calls']} |",
          f"| Truncated (max_tokens filled) | {c['truncated']} ({rate}) |", f"| Refusals | {c['refused']} |", f"| Invalid replies | {c['invalid']} |",
          f"| Server errors (incl. retries) | {c['service_errors']} {dict(c['error_status']) or ''} |", f"| Of which billed empty replies | {c['billed_retries']} |",
          f"| possibly_billed | {c['possibly_billed']} |", "", f"- finish / stop reasons: {dict(c['finish'])}",
          "- Run status: " + ", ".join(f"{s} {[r['status'] for r in rs]}" for s, rs in runs.items()), ""]
    run_spent = sum(r["spent"] for rs in runs.values() for r in rs)
    L += ["## 6. Cost", "", "| Setting | Actual | Expected (all 40 steps) |", "|---|---|---|"]
    for s, rs in runs.items():
        e = next((x["cost_usd"] for x in est if x["setting"] == s and x["steps"] in (None, 40)), None)
        L.append(f"| {s} | ${sum(r['spent'] for r in rs):.4f} | {'—' if e is None else f'${e:.4f}'} |")
    full = sum(x["cost_usd"] for x in est if x["steps"] in (None, 40))
    half = sum(x["cost_usd"] for x in est if x["steps"] in (None, 20))
    L += [f"| **Total (excluding probe)** | **${run_spent:.4f}** | **${full:.4f}** |", "",
          f"- Actual cost including probe: ${spent:.4f}. Expected total with Active at 20 steps: ${half:.4f}.",
          "- Actual cost comes from the raw usage in the call logs; possibly_billed attempts have unknown usage and are not included, so check against the bill.", ""]
    L += ["## 7. tiny and all recomputed from measurements", "", f"Active runs {repeats('active_v2')} time(s), random_walk {repeats('random_walk_40')} time(s) (registry repeats), already multiplied in.", "", "| Setting | tiny | tiny (Active uses all 40 steps) | all | all (Active uses all 40 steps) |", "|---|---|---|---|---|"]
    names = list(rec)
    for s in SETTINGS:
        if s in rec[names[0]]:
            L.append(f"| {s} | " + " | ".join(f"${rec[n][s][i]:,.2f}" for n in names for i in (0, 1)) + " |")
    L.append("| **Total** | " + " | ".join(f"**${sum(v[i] for v in rec[n].values()):,.2f}**" for n in names for i in (0, 1)) + " |")
    return L + ["", "See the module docstring of `providers/smoke_report.py` for the method. Route lengths and question-type mix of the smoke groups differ from the full set, so this is a rough value.", ""]


def build(data, reg, short: str, prefix: str, targets: dict | None = None) -> list[str]:
    root = R.results_root(reg)
    pj = root / short / "_probe" / prefix / "probe.json"
    probe = json.loads(pj.read_text()) if pj.exists() else None
    runs = collect(root, short, prefix)
    done = probe if probe and probe.get("tokens_per_image") and probe.get("output_tokens") else None
    est = run_estimate(data, short, group_set(data, "smoke"), SETTINGS, done)
    targets = targets or {"tiny": group_set(data, "tiny"), "all": group_set(data, "all")}
    return render(short, prefix, R.entries(reg)[short], probe, runs, est, scope_spent(root, short, prefix), recost(data, short, runs, targets)) + \
        thinking_md(thinking_tokens(root, short, prefix), "## 8. Thinking tokens (by call type)") + conditions_md(root, short, prefix, SETTINGS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prefix", default="smoke")
    ap.add_argument("--out", help="default reports/eval/api_smoke_<short name>.md")
    a = ap.parse_args()
    reg = R.load_registry()
    data = FrozenData(verify=False)
    R.merge(data, reg)
    out = Path(a.out) if a.out else R.ROOT / "reports" / "eval" / f"api_smoke_{a.model}.md"
    out.write_text("\n".join(build(data, reg, a.model, a.prefix)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
