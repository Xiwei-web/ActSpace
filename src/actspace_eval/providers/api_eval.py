"""Commands for the closed-model evaluation. scripts/run_api_eval.sh calls check and verify on the login node and job inside the job.

    python -m actspace_eval.providers.api_eval check --model <short name> [--key-file <file>]   # key file (prints numbers only), price confirmation, reasoning approval
    python -m actspace_eval.providers.api_eval keys [--reset-key <number> ...]   # accumulated spend per key (reads only the ledger, not the key file); resets replaced numbers
    python -m actspace_eval.providers.api_eval keyscan --model <short name> --prefix <prefix> [--key-file <file>]   # whether logs and results contain anything resembling a key; reports yes / no only
    python -m actspace_eval.providers.api_eval verify                           # freeze verify
    python -m actspace_eval.providers.api_eval job --model <short name> --set <group set> --prefix <prefix> --cap <USD> [--settings ...] [--repeats N] [--resume] [--key-file <file>]
    python -m actspace_eval.providers.api_eval estimate --model <short name> --set <group set> --prefix <prefix> [--settings ...] [--repeats N]
    python -m actspace_eval.providers.api_eval report --model <short name> --set <group set> --prefix <prefix> [--settings ...]
    python -m actspace_eval.providers.api_eval truncation --model <short name> --prefix <prefix> [--settings ...]   # truncation rate; warns above the threshold
    python -m actspace_eval.providers.api_eval leaderboard --models a,b,c --prefix <prefix>                 # Active scores of several models; may mix 1 and 3 runs
    python -m actspace_eval.providers.api_eval verify-wire --model <short name> --prefix <prefix>           # checks the recorded requests (for tests)

- job: one job runs, in order, the probe, the cost estimate, all settings and the score table. All runs start at the same time and
  share one cost cap and `concurrency` episode slots, granted in run order (CostPool): cheap settings finish first, Active comes
  last, and once a run has issued all its episodes its slots go to the next run immediately.
- Run directory: <results root>/<short name>/<setting>/<prefix>_r<seed>/, with the same artifacts as for open-weight models
  (trajectories.jsonl, calls.jsonl, summary.json, status.json, env.json, SHA256SUMS).
- Repeats: active_v2 and random_walk_40 run --repeats times, by default the registry's repeats (1 in the first round); seeds
  0..N-1; the other settings run once. --repeats 3 --resume runs only the missing seeds: sealed runs are skipped and not billed
  again; the score table is regenerated from all completed runs in the directory.
- Cost cap: --cap is the total cap shared by all runs and the probe of this prefix (same model), including money already spent;
  when running more later, --cap must include what has already been spent.
- Progress: <results root>/<short name>/_progress/<prefix>/<setting>_r<seed>.log, one line per finished episode (done / total, cost
  of this run, accumulated cost of the job).
- Resume: after a job is killed, --resume continues from the finished episodes; interrupted episodes are discarded and rerun (money
  already spent stays in the call log and counts toward the cap).
- Exit codes: 0 complete; 1 probe failed; 2 arguments or existing artifacts; 3 cost cap; 4 fatal error (HTTP 400 / 401 / 403 etc.);
  5 server-error retries exhausted; 6 all Claude keys exhausted (add keys, then --resume). 3, 4 and 5 can all be resumed.
- Truncation rate: share of calls that filled max_tokens, written to each run's status.json (truncation_rate) and at the end of the
  score table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .. import prompts as P
from ..cost import CostTracker, estimate
from ..data import FrozenData
from ..llm_agent import setting_env
from ..llm_run import env_factory, group_set, run_llm, setting_dir
from ..meta import group_meta
from ..models import estimate_input_tokens
from .. import report_eval as RE
from . import registry as R
from .api_probe import probe_model, report as probe_report
from . import keys as K
from .base import CostPool, repair_tail

SETTINGS = ("blind", "passive_8", "passive_20", "passive_40", "random_walk_40", "oracle_min", "oracle_route", "active_v2")
RUN_OF = {"blind": ("blind", 40), "passive_8": ("passive", 8), "passive_20": ("passive", 20), "passive_40": ("passive", 40),
          "random_walk_40": ("random_walk", 40), "oracle_min": ("oracle_min", 40), "oracle_route": ("oracle_route", 40), "active_v2": ("active_v2", 40)}
REPEATED = ("random_walk_40", "active_v2")
EXIT_CODES = {"stopped_cost_cap": 3, "stopped_fatal_error": 4, "stopped_server_error": 5, K.QUOTA_STATUS: 6}
EXIT_FATAL = EXIT_CODES["stopped_fatal_error"]


def repeats(setting: str, n: int | None = None) -> int:
    """active_v2 and random_walk_40 run n times (default: the registry's repeats); the other settings run once."""
    if setting not in REPEATED:
        return 1
    return n if n is not None else R.load_registry()["repeats"][setting]


def plan_runs(prefix: str, settings, n: int | None = None) -> list[dict]:
    out = []
    for s in settings:
        mode, budget = RUN_OF[s]
        for k in range(repeats(s, n)):
            out.append({"setting": s, "mode": mode, "budget": budget, "seed": k, "name": f"{prefix}_r{k}"})
    return out


def existing_runs(root: Path, short: str, prefix: str, setting: str) -> list[Path]:
    """Runs already in the directory (complete or not), sorted by seed."""
    mode, budget = RUN_OF[setting]
    d = root / short / setting_dir(mode, budget)
    return sorted((p for p in d.glob(f"{prefix}_r*") if re.fullmatch(rf"{re.escape(prefix)}_r\d+", p.name)), key=lambda p: int(p.name.rsplit("_r", 1)[1]))


def read_status(d: Path) -> dict:
    return json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}


def run_dir(root: Path, short: str, run: dict) -> Path:
    return root / short / setting_dir(run["mode"], run["budget"]) / run["name"]


def scope_spent(root: Path, short: str, prefix: str) -> float:
    """Money already spent under this prefix: calls.jsonl of all runs plus the probe call logs."""
    total = sum(CostTracker.spent_in_log(p) for p in (root / short / "_probe" / prefix).glob("*.jsonl"))
    for d in (root / short).glob(f"*/{prefix}_r*"):
        if d.is_dir():
            total += CostTracker.spent_in_log(d / "calls.jsonl")
    return total


def load(verify=True):
    reg = R.load_registry()
    data = FrozenData(verify=verify)
    R.merge(data, reg)
    R.assert_protocol(data.ecfg, data.version)
    return reg, data


def settings_arg(s) -> list[str]:
    out = list(SETTINGS) if not s or s == "all" else s.split(",")
    unknown = [x for x in out if x not in RUN_OF]
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; choices {list(SETTINGS)}")
    return out


# ---------------------------------------------------------------- steps 1 and 2
def cmd_check(a):
    reg = R.load_registry()
    e = R.entries(reg)[a.model]
    print(load_keys(reg, a.model, a.key_file)["note"])
    print(f"prices_confirmed: {str(e['prices_confirmed']).lower()}")
    print(f"reasoning.approved: {str(e['reasoning']['approved']).lower()}")
    if not (e["prices_confirmed"] and e["reasoning"]["approved"]):
        print(f"stopped: prices of {a.model} are unconfirmed or its reasoning-effort setting is not approved (configs/models_api.yaml)")
        sys.exit(2)


def cmd_verify(a):
    FrozenData(verify=True)


def load_keys(reg, short: str, key_file) -> dict:
    """Reads the key file; refuses to run on wrong permissions or when no usable key exists."""
    try:
        return K.setup_keys(R.results_root(reg), short, R.entries(reg)[short], key_file)
    except K.KeyFileError as e:
        sys.exit(print(f"stopped: {e}") or 2)


def key_lines(root: Path) -> list[str]:
    rows = K.read_ledger(root)["keys"]
    return [f"anthropic key #{n}: ${r['spent_usd']:.4f}" + (f", exhausted ({r['exhausted']})" if r.get("exhausted") else "") for n, r in sorted(rows.items(), key=lambda x: int(x[0]))]


def cmd_keys(a):
    root = R.results_root(R.load_registry())
    for no in a.reset_key or []:
        K.reset_key(root, no)
        print(f"anthropic key #{no}: total reset to zero")
    print("\n".join(key_lines(root)) or "the ledger has no records yet")


def run_files(root: Path, short: str, prefix: str) -> list[Path]:
    """All logs and results of this model and prefix, plus the key ledger."""
    dirs = [d for d in (root / short).glob(f"*/{prefix}_r*") if d.is_dir()]
    dirs += [root / short / "_probe" / prefix, root / short / "_progress" / prefix, root / "_keys"]
    files = [p for d in dirs for p in d.rglob("*") if p.is_file()]
    return files + sorted((root / "_jobs").glob(f"api_{short}_{prefix}_*.out")) + sorted((root / short / "_reports").glob(f"{prefix}*"))


def cmd_keyscan(a):
    """Reports yes / no only. The key file is read by this command itself, to compare against the verbatim keys."""
    reg = R.load_registry()
    e = R.entries(reg)[a.model]
    values = [] if e["key_env"] is None and a.key_file is None else list(K.read_key_file(a.key_file or K.DEFAULT_KEY_FILE).values())
    files = run_files(R.results_root(reg), a.model, a.prefix)
    hits = K.scan_for_keys(files, values)
    print(f"suspected keys: {'yes' if hits else 'none'} ({len(files)} files checked)")
    sys.exit(1 if hits else 0)


# ---------------------------------------------------------------- probe
def step_probe(data, reg, a):
    root = R.results_root(reg)
    out = root / a.model / "_probe" / a.prefix
    prev = out / "probe.json"
    if a.resume and prev.exists() and json.loads(prev.read_text()).get("passed"):
        print(f"probe already passed: {prev}")
        return
    mcfg = data.ecfg["models"][a.model]
    one_call = CostTracker(mcfg["prices_usd_per_mtok"], 0).cost_of({"input_tokens": 12000, "cache_write_tokens": 12000, "output_tokens": mcfg["max_output_tokens"]})
    cap = min(a.cap - scope_spent(root, a.model, a.prefix), 5 * one_call)   # enough for 4 calls that fill the limit; never above the total cap
    r = probe_model(data, a.model, out, cap)
    prev.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(probe_report(r))
    if not r["passed"]:
        print("probe failed, stopping")
        sys.exit(EXIT_CODES.get(r.get("stop_status"), EXIT_FATAL) if r.get("fatal") else 1)


# ---------------------------------------------------------------- cost estimate
def start_prefix_tokens(env, gid, tpi, budget, history, scale, version="v2") -> int:
    """Estimated tokens of the fixed prefix (system prompt + questions + START image)."""
    obs = env.reset(gid)
    system = P.system_prompt("active", budget=budget, history=history, version=version, scale=scale)
    parts = P.user_active_start(obs, version)
    n = next(i for i, p in enumerate(parts) if p[0] == "image")
    return estimate_input_tokens(system, [{"role": "user", "parts": parts[:n + 1]}], tpi)


def apply_anthropic_cache(rows, data, mcfg, gids):
    """cost.estimate estimates caching with OpenAI's prefix rules; Anthropic has a single breakpoint on the START image: written on the first call of each episode and read afterwards; one-shot answering settings have no caching."""
    t = CostTracker(mcfg["prices_usd_per_mtok"], float("inf"))
    ec = data.ecfg
    fac = env_factory(data)
    for r in rows:
        r["cached_input_tokens"] = r["cache_write_tokens"] = 0
        if setting_env(r["mode"])[0] == "active":
            env = fac(r["mode"], r["budget"], 0)
            pre = sum(start_prefix_tokens(env, g, mcfg["tokens_per_image"], r["budget"], ec["llm"]["history"], data.step_scale(g)) for g in gids) * r["seeds"]
            eps = len(gids) * r["seeds"]
            r["cache_write_tokens"], r["cached_input_tokens"] = pre, pre * (r["calls"] - eps) // eps
        r["cost_usd"] = t.cost_of(r)
        r["cost_no_cache_usd"] = t.cost_of({"input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
    return rows


def run_estimate(data, short, gids, settings, probe=None, n=None):
    """Estimated cost per setting. Active is estimated both with all 40 steps and with 20 steps. Tokens per image and output per step measured by the probe override the registered values."""
    ec = data.ecfg
    mcfg = dict(ec["models"][short])
    eo = dict(mcfg["est_output_tokens"])
    if probe:
        mcfg["tokens_per_image"] = probe["tokens_per_image"]
        eo["action"] = max(probe["output_tokens"])
    mcfg["est_output_tokens"] = eo
    hist, fac = ec["llm"]["history"], env_factory(data)
    rows = []
    for s in settings:
        mode, budget = RUN_OF[s]
        reps = repeats(s, n)
        act = setting_env(mode)[0] == "active"
        for steps in ((budget, 20) if act else (budget,)):
            r = estimate(fac, gids, [(mode, budget, steps, tuple(range(reps)))], mcfg, hist, ec["env"]["step_scale"])
            if mcfg["provider"] == "anthropic":
                apply_anthropic_cache(r, data, mcfg, gids)
            rows.append({"setting": s, "steps": steps if act else None, "reps": reps, **r[0]})
    return rows


def step_estimate(data, reg, a):
    gids = group_set(data, a.set)
    prev = R.results_root(reg) / a.model / "_probe" / a.prefix / "probe.json"
    probe = json.loads(prev.read_text()) if prev.exists() else None
    rows = run_estimate(data, a.model, gids, settings_arg(a.settings), probe, a.repeats)
    print(f"Cost estimate: {a.model}, {a.set} ({len(gids)} groups), {probe['tokens_per_image'] if probe else data.ecfg['models'][a.model]['tokens_per_image']} tokens per image"
          f" ({'measured by probe' if probe else 'registered value'}), {max(probe['output_tokens']) if probe else data.ecfg['models'][a.model]['est_output_tokens']['action']} output tokens per step")
    full = half = 0.0
    for r in rows:
        tag = f"{r['setting']} ×{r['reps']}" + (f" ({r['steps']} steps)" if r["steps"] else "")
        print(f"  {tag:<32} calls {r['calls']:>7,}  input {r['input_tokens']:>12,}  of which cache read {r['cached_input_tokens']:>11,}  output {r['output_tokens']:>10,}  ${r['cost_usd']:.2f}")
        if r["steps"] in (None, 40):
            full += r["cost_usd"]
        if r["steps"] in (None, 20):
            half += r["cost_usd"]
    print(f"Total: Active using all 40 steps ${full:.2f}; Active at 20 steps ${half:.2f}. Thinking tokens beyond the estimate are not included.")
    if a.cap is not None and full > a.cap:
        print(f"Note: the estimate ${full:.2f} exceeds the cap ${a.cap:.2f}; the run will stop when the cap is reached, and can continue with --resume after raising the cap")


def cmd_estimate(a):
    reg, data = load(verify=False)
    step_estimate(data, reg, a)


# ---------------------------------------------------------------- single job
def to_run(a, root, runs) -> list[dict]:
    """Runs to execute: skips sealed and N/A runs; refuses when artifacts exist without --resume (exits without spending anything)."""
    todo = []
    for r in runs:
        out = run_dir(root, a.model, r)
        if (out / "SHA256SUMS").exists() or read_status(out).get("status") == "not_applicable":
            print(f"already complete, skipping: {out}")
            continue
        r = {**r, "out": out, "resume": (out / "trajectories.jsonl").exists() or (out / "calls.jsonl").exists()}
        if r["resume"] and not a.resume:
            sys.exit(print(f"{out} already has run artifacts; add --resume to continue, or use another --prefix") or 2)
        todo.append(r)
    return todo


def run_all(data, reg, a, todo) -> dict:
    """Starts all runs at the same time, sharing the cost cap and episode slots. Returns {run directory: status or exception}."""
    root = R.results_root(reg)
    e = R.entries(reg)[a.model]
    gids = group_set(data, a.set)
    for r in todo:
        if r["resume"]:
            repair_tail(r["out"] / "calls.jsonl")
            repair_tail(r["out"] / "trajectories.jsonl")
    pool = CostPool(data.ecfg["models"][a.model]["prices_usd_per_mtok"], a.cap, scope_spent(root, a.model, a.prefix), slots=e["concurrency"]["episodes"])
    print(f"total cap ${a.cap:.2f}, already spent ${pool.spent:.4f}, concurrent episodes {e['concurrency']['episodes']}, {len(todo)} runs")
    prog_dir = root / a.model / "_progress" / a.prefix
    prog_dir.mkdir(parents=True, exist_ok=True)
    group_meta(data)
    pool.expect(range(len(todo)))
    results = {}

    def one(order, r):
        tag = f"{r['setting']}_r{r['seed']}"
        with open(prog_dir / f"{tag}.log", "a", buffering=1) as prog:
            def tick(done, total, spent):
                prog.write(f"{time.strftime('%F %T')} {done}/{total} run ${spent:.4f} job total ${pool.spent:.4f}\n")
            try:
                st = run_llm(data, a.model, r["mode"], r["budget"], gids, r["out"], resume=r["resume"], cost_cap=a.cap, run_seed=r["seed"],
                             workers=e["concurrency"]["episodes"], env_info=R.env_info(a.model, e), seal=True, print_estimate=False,
                             tracker_cls=lambda prices, cap, spent: pool.view(spent), gate=pool.slot(order), on_episode=tick)
            except Exception as x:  # noqa: BLE001  code error: stop the job and log the original error
                pool.abort(x)
                results[r["out"]] = x
                prog.write(f"{time.strftime('%F %T')} error: {x}\n")
                print(f"=== {tag} error: {x}", flush=True)
                return
            finally:
                pool.arrived(order)
            prog.write(f"{time.strftime('%F %T')} {st['status']} {st['completed']}/{len(gids)} run ${st['spent_usd']:.4f} truncation rate {st['truncation_rate']}\n")
        results[r["out"]] = st
        print(f"=== {tag} {st['status']} {st['completed']}/{len(gids)} ${st['spent_usd']:.4f} truncation {st['truncation_rate']} {st['seconds']} s" +
              (f" error: {st['error']}" if st["error"] else ""), flush=True)

    ths = [threading.Thread(target=one, args=(i, r)) for i, r in enumerate(todo)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"job total ${pool.spent:.4f}", flush=True)
    return results


def job_exit_code(results: dict) -> int:
    """0 if everything is complete or N/A; otherwise by stop reason (a job has a single stop reason; the other runs stop with it)."""
    if any(isinstance(v, Exception) for v in results.values()):
        return 1
    seen = {v["status"] for v in results.values()}
    for s in (K.QUOTA_STATUS, "stopped_fatal_error", "stopped_server_error", "stopped_cost_cap"):
        if s in seen:
            return EXIT_CODES[s]
    return 0


def cmd_job(a):
    reg, data = load()
    root = R.results_root(reg)
    settings = settings_arg(a.settings)
    print(f"== job {os.environ.get('SLURM_JOB_ID', 'local')} on {os.uname().nodename}: {a.model} {a.set} prefix {a.prefix} settings {settings} repeats {a.repeats or 'registry'}", flush=True)
    todo = to_run(a, root, plan_runs(a.prefix, settings, a.repeats))
    if todo:
        print(load_keys(reg, a.model, a.key_file)["note"], flush=True)
        print("== probe", flush=True)
        step_probe(data, reg, a)
        print("== expected cost", flush=True)
        step_estimate(data, reg, a)
        print("== runs", flush=True)
        code = job_exit_code(run_all(data, reg, a, todo))
        if code:
            print(f"stopped, exit code {code} (3 cost cap; 4 fatal error; 5 server-error retries exhausted; 6 all keys exhausted; 1 code error); details in status.json and calls.jsonl; continue with --resume", flush=True)
            sys.exit(code)
    print("== score table", flush=True)
    step_report(reg, a)
    cmd_truncation(a)


# ---------------------------------------------------------------- score table
def truncation_rows(root: Path, short: str, prefix: str, settings) -> list[dict]:
    """Status and truncation of every run in the directory; reads only status.json and env.json."""
    rows = []
    for s in settings:
        for d in existing_runs(root, short, prefix, s):
            st = read_status(d)
            u = st.get("usage") or {}
            env = json.loads((d / "env.json").read_text()) if (d / "env.json").exists() else {}
            rows.append({"setting": s, "run": d.name, "status": st.get("status"), "calls": u.get("calls", 0),
                         "truncated": u.get("truncated_calls", 0), "rate": st.get("truncation_rate"), "max_tokens": env.get("max_tokens")})
    return rows


def run_condition(d: Path) -> dict:
    """json.mode and max_tokens of a run (env.json)."""
    env = json.loads((d / "env.json").read_text()) if (d / "env.json").exists() else {}
    return {"json_mode": (env.get("json") or {}).get("mode"), "max_tokens": env.get("max_tokens")}


def conditions_md(root: Path, short: str, prefix: str, settings) -> list[str]:
    L = ["## Run conditions", "", "| Setting | Run | json.mode | max_tokens |", "|---|---|---|---|"]
    for s in settings:
        for d in existing_runs(root, short, prefix, s):
            c = run_condition(d)
            L.append(f"| {s} | {d.name} | {c['json_mode']} | {c['max_tokens']} |")
    return L + [""]


CALL_KINDS = ("Move step", "Active answer step", "One-shot answer")


def thinking_tokens(root: Path, short: str, prefix: str, settings=SETTINGS) -> dict:
    """{call type: [thinking tokens per call]}. Whether a call is an answer is decided by its parse result; invalid replies count as move steps."""
    out = {k: [] for k in CALL_KINDS}
    for s in settings:
        for d in existing_runs(root, short, prefix, s):
            if not (d / "calls.jsonl").exists():
                continue
            with open(d / "calls.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("type") != "call":
                        continue
                    kind = CALL_KINDS[2] if s != "active_v2" else CALL_KINDS[1] if (r.get("parse") or {}).get("action") == "answer" else CALL_KINDS[0]
                    out[kind].append(r["usage"].get("reasoning_tokens") or 0)
    return out


def thinking_row(v: list) -> dict | None:
    if not v:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": len(v), "median": float(med), "q1": float(q1), "q3": float(q3), "zero": sum(x == 0 for x in v) / len(v)}


def thinking_md(values: dict, heading="## Thinking tokens (by call type)") -> list[str]:
    L = [heading, "", "| Call type | Calls | Median | Quartiles | Share with zero thinking |", "|---|---|---|---|---|"]
    for k in CALL_KINDS:
        s = thinking_row(values[k])
        L.append(f"| {k} | 0 | — | — | — |" if s is None else f"| {k} | {s['n']:,} | {s['median']:,.0f} | {s['q1']:,.0f}–{s['q3']:,.0f} | {s['zero']:.1%} |")
    return L + [""]


def truncation_total(rows) -> tuple[int, int, float | None]:
    calls, trunc = sum(r["calls"] for r in rows), sum(r["truncated"] for r in rows)
    return calls, trunc, (trunc / calls if calls else None)


def truncation_md(rows) -> list[str]:
    calls, trunc, rate = truncation_total(rows)
    limits = ", ".join(str(x) for x in sorted({r["max_tokens"] for r in rows if r.get("max_tokens")}))   # the limit each run actually used
    L = ["## Truncation rate", "", f"Share of calls that filled max_tokens ({limits + ', ' if limits else ''}thinking included). Truncated replies count as no answer.", "",
         "| Setting | Run | Calls | Truncated | Truncation rate |", "|---|---|---|---|---|"]
    L += [f"| {r['setting']} | {r['run']} | {r['calls']} | {r['truncated']} | {'—' if r['rate'] is None else format(r['rate'], '.2%')} |" for r in rows]
    return L + [f"| **Total** | | **{calls}** | **{trunc}** | **{'—' if rate is None else format(rate, '.2%')}** |", ""]


def cmd_truncation(a):
    reg = R.load_registry()
    rows = truncation_rows(R.results_root(reg), a.model, a.prefix, settings_arg(a.settings))
    calls, trunc, rate = truncation_total(rows)
    for r in rows:
        print(f"  {r['setting']:<16} {r['run']:<16} {str(r['status']):<22} calls {r['calls']:>6}  truncated {r['truncated']:>4}")
    print(f"truncation rate: {a.model} {trunc} / {calls}" + ("" if rate is None else f" = {rate:.2%}"))
    limit = reg["truncation_warn_rate"]
    if rate is not None and rate >= limit:
        bar = "!" * 72
        print(f"{bar}\n!! WARNING: truncation rate of {a.model} is {rate:.2%} >= {limit:.0%} ({trunc} / {calls} calls filled max_tokens)\n"
              f"!! Truncated replies count as no answer and lower the scores. Check the smoke report before scaling up to tiny / all.\n{bar}")


def step_report(reg, a):
    """Score table: all completed runs in the directory; regenerated after additional runs."""
    root = R.results_root(reg) / a.model
    settings = settings_arg(a.settings)
    spec = {}
    for s in settings:
        for d in existing_runs(R.results_root(reg), a.model, a.prefix, s):
            st = read_status(d).get("status")
            if st == "complete":
                spec.setdefault(s, []).append(f"{d.parent.name}/{d.name}")
            else:
                print(f"incomplete, left out of the score table: {d} ({st})")
    if not spec:
        sys.exit("no completed runs")
    out = root / "_reports" / f"{a.prefix}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = truncation_rows(R.results_root(reg), a.model, a.prefix, settings)
    lines = RE.render(RE.build(root, spec), f"{a.model} scores ({a.set}, {a.prefix})")
    tail = conditions_md(R.results_root(reg), a.model, a.prefix, settings) + truncation_md(rows) + thinking_md(thinking_tokens(R.results_root(reg), a.model, a.prefix, settings))
    out.write_text("\n".join(lines) + "\n\n" + "\n".join(tail) + "\n")
    print(f"wrote {out}")
    print(f"total spent ${scope_spent(R.results_root(reg), a.model, a.prefix):.4f} (including probe)")


def cmd_report(a):
    step_report(R.load_registry(), a)


def complete_spec(root: Path, short: str, prefix: str) -> dict:
    spec = {}
    for s in SETTINGS:
        for d in existing_runs(root, short, prefix, s):
            if read_status(d).get("status") == "complete":
                spec.setdefault(s, []).append(f"{d.parent.name}/{d.name}")
    return spec


def leaderboard(root: Path, models, prefix: str) -> list[str]:
    """Active scores of several models, three tables. Each model's interval uses its own number of runs, so one table may mix 1 and 3 runs."""
    views = {m: RE.build(root / m, complete_spec(root, m, prefix)) for m in models}
    L = [f"# Model comparison (prefix {prefix})", ""]
    for scope, lab in RE.SCOPES:
        L += [f"## {lab}", "", "| Model | json.mode | Active runs | random_walk runs | Total score | Answer rate | Selective score | Correct-refusal rate | Active − random_walk_40 |", "|---|---|---|---|---|---|---|---|---|"]
        single = []
        for m, V in views.items():
            act, rw = V["runs"].get("active_v2"), V["runs"].get("random_walk_40")
            if not act or scope not in V["boots"]:
                continue
            b = V["boots"][scope]
            cells = [RE.ci(b.mean_ci([RE.items(r, k) for r in act], f"active_v2|{k}")) for k in ("total", "answer_rate", "selective", "refusal")]
            gain = RE.ci(b.diff_ci([RE.items(r, "total") for r in act], "active_v2|total", [RE.items(r, "total") for r in rw], "random_walk_40|total"), sign=True) if rw else "—"
            modes = sorted({str(run_condition(d)["json_mode"]) for s in SETTINGS for d in existing_runs(root, m, prefix, s)})
            L.append(f"| {m} | {', '.join(modes)} | {len(act)} | {len(rw or [])} | " + " | ".join(cells) + f" | {gain} |")
            if len(act) == 1 or (rw and len(rw) == 1):
                single.append(m)
        L.append("")
        if single:
            L += [f"Models with a single run ({', '.join(single)}): the interval uses the scene bootstrap only and excludes run-to-run variation, so it is too narrow when compared with 3-run models.", ""]
    L += ["## Thinking tokens (by call type)", ""]
    for m in models:
        L += thinking_md(thinking_tokens(root, m, prefix), f"**{m}**")
    return L


def cmd_leaderboard(a):
    root = R.results_root(R.load_registry())
    out = root / "_reports" / f"leaderboard_{a.prefix}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(leaderboard(root, a.models.split(","), a.prefix)) + "\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------- tests
def design_system_prompts() -> list[str]:
    """System prompts quoted verbatim in Appendix A of docs/evaluation_protocol.md."""
    doc = (R.ROOT / "docs" / "evaluation_protocol.md").read_text()
    sec = doc.split("## Appendix A", 1)[1].split("## Appendix B", 1)[0]
    return [b.strip("\n") for b in re.findall(r"```text\n(.*?)```", sec, re.S)]


def template_regex(t: str) -> str:
    out = re.sub(r"\\\{\\\{image\\\}\\\}", "<image>", re.escape(t))
    return re.sub(r"\\\{[a-z_]+(?::\\\.\df)?\\\}", "(?s:.*?)", out)


def first_turn_regexes() -> list:
    return [re.compile(template_regex(t)) for t in (P.USER_START_ACTIVE_V2, P.USER_STATIC, P.USER_ROUTE, P.USER_BLIND)]


def render_turn(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(x["text"] if x["type"] == "text" else "<image>" for x in content)


def verify_calls(path: Path, sys_ok: list, first_ok: list) -> dict:
    """One calls.jsonl: system prompts verbatim as in docs/evaluation_protocol.md; the first user message matches a first-turn template; later user messages are step feedback; all images are 640x480 or 480x640; no base64."""
    res = {"calls": 0, "images": 0, "bad_system": 0, "bad_first": 0, "bad_step": 0, "bad_size": 0, "base64": 0}
    with open(path) as f:
        for line in f:
            res["base64"] += "base64," in line or '"data":' in line
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            res["calls"] += 1
            msgs = r["request"]["messages"]
            s = re.sub(r"You are given \d+ photographs", "You are given 41 photographs", msgs[0]["content"])
            res["bad_system"] += s not in sys_ok
            res["bad_first"] += not any(x.fullmatch(render_turn(msgs[1]["content"])) for x in first_ok)
            for m in msgs[2:]:
                if m["role"] == "user":
                    t = render_turn(m["content"])
                    res["bad_step"] += not (t.startswith("Step ") and (t.endswith(P.STEP_CHOOSE) or t.endswith(P.STEP_MUST_ANSWER)))
            for m in msgs[1:]:
                for x in m["content"] if isinstance(m["content"], list) else []:
                    if x["type"] == "image":
                        res["images"] += 1
                        res["bad_size"] += (x["w"], x["h"]) not in ((640, 480), (480, 640)) or len(x["sha256"]) != 64 or x["frame_id"] is None
    return res


def cmd_verify_wire(a):
    root = R.results_root(R.load_registry()) / a.model
    sys_ok, first_ok = design_system_prompts(), first_turn_regexes()
    tot = {}
    for p in sorted(root.glob(f"*/{a.prefix}_r*/calls.jsonl")):
        r = verify_calls(p, sys_ok, first_ok)
        print(p.parent.relative_to(root), r)
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + v
    print("total", tot)
    if not tot.get("calls") or any(v for k, v in tot.items() if k not in ("calls", "images")):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("verify", cmd_verify), ("job", cmd_job), ("estimate", cmd_estimate), ("report", cmd_report),
                     ("truncation", cmd_truncation), ("leaderboard", cmd_leaderboard), ("verify-wire", cmd_verify_wire), ("keys", cmd_keys), ("keyscan", cmd_keyscan)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name == "leaderboard":
            p.add_argument("--models", required=True)
        elif name not in ("verify", "keys"):
            p.add_argument("--model", required=True)
        if name not in ("check", "verify", "keys"):
            p.add_argument("--prefix", required=True)
        if name in ("check", "job", "keyscan"):
            p.add_argument("--key-file")
        if name == "keys":
            p.add_argument("--reset-key", type=int, action="append")
        if name in ("job", "estimate", "report"):
            p.add_argument("--set", required=True, choices=["smoke", "tiny", "all"])
        if name in ("job", "estimate", "report", "truncation"):
            p.add_argument("--settings")
        if name in ("job", "estimate"):
            p.add_argument("--repeats", type=int)
        if name == "job":
            p.add_argument("--cap", type=float, required=True)
            p.add_argument("--resume", action="store_true")
        if name == "estimate":
            p.add_argument("--cap", type=float)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
