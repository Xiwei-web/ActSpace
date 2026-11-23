"""Claude reply-format diagnostics: compares whether the model replies with a single JSON object as required under several request
variants. Sends real requests and needs a key.

    python -m actspace_eval.providers.format_diag <short name> --variants v0,v1,v2 --n <repeats per variant> --cap <USD>

- Variants: v0 baseline (json.mode none, thinking.display summarized); v1 structured outputs (anthropic_api.REPLY_SCHEMA); v2 baseline
  without thinking.display. None of the three looks at the registry's current json.mode; each sets it explicitly, so v0 stays the
  original baseline after the registry is switched.
- Each variant sends n start calls (probe B: START image + questions) and n mid calls (real mid-episode states). mid uses 2 states at
  step 5 of haiku-5.5 active_v2 all_r0 (one ARKit, one ScanNet++, each with only single-JSON replies before it), alternating over the
  n calls; messages are rebuilt verbatim from the call log, images are checked against the recorded sha256, and it refuses to run
  on a mismatch.
- Calls are interleaved by round (round i: every variant for start, then every variant for mid), so sample sizes stay close when
  --cap stops the run midway.
- A single shared --cap: each call reserves its maximum output beforehand, so the cap is never exceeded. HTTP 400 stops only that
  variant (e.g. schema rejected); other fatal errors, exhausted retries and the cost cap stop everything.
- Output: <results root>/_format_diag/<short name>/<timestamp>/ (calls_<variant>.jsonl, results.json, summary.md); the summary table
  is written to reports/eval/api_format_check_<short name>.md (change with --report).
- Exit codes: 0 complete; 2 arguments or switches; 3 cost cap; 4 fatal error; 5 server-error retries exhausted; 6 all keys exhausted.
  Real models do not run on the login node: use srun or sbatch.
The key is read by this command from --key-file (default ~/.actspace/api_keys.env); which key is used and how much it spent go through
the keys.KeyPool ledger and are not written to any result file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ..cost import CostCapExceeded
from ..data import FrozenData, dataset_of_scene
from ..llm_agent import parse_detail
from ..models import ModelError, RealCallsDisabled, encode_jpeg, make_client
from ..probe import probe_messages
from . import keys as K
from . import registry as R
from .base import ApiCostTracker

VARIANTS = {"v0": "baseline", "v1": "structured output", "v2": "baseline without thinking.display"}
SOURCE = ("haiku-5.5", "active_v2", "all_r0")
MID_STEP = 5
EXIT_CAP, EXIT_FATAL, EXIT_SERVER, EXIT_QUOTA = 3, 4, 5, 6
PREVIEW_CHARS = 200


def variant_cfg(mcfg: dict, v: str) -> dict:
    display = "summarized" if v in ("v0", "v1") else None
    return {**mcfg, "json_schema": v == "v1", "reasoning": {**mcfg["reasoning"], "display": display}}


def classify(text: str) -> dict:
    """one_json: the whole reply is exactly one JSON object; valid: the evaluation parser extracted a valid action (extra text before or after still counts)."""
    t = (text or "").strip()
    try:
        one = isinstance(json.loads(t), dict)
    except ValueError:
        one = False
    n_actions = max(len(re.findall(r'"action"\s*:', t)), len(re.findall(r"<invoke\b", t)))
    return {"one_json": one, "valid": parse_detail(text)["error"] is None, "markup": "<invoke" in t or "<parameter" in t, "multi_action": n_actions >= 2}


def rebuild(data, gid: str, msgs: list) -> tuple[str, list]:
    """request.messages of the call log -> (system prompt, conversation). Images are re-read and their sha256 must match the record."""
    scene = data.scene(data.groups[gid]["scene_id"])
    turns = []
    for m in msgs[1:]:
        if isinstance(m["content"], str):
            turns.append({"role": m["role"], "text": m["content"]})
            continue
        parts = []
        for b in m["content"]:
            if b["type"] == "text":
                parts.append(("text", b["text"]))
                continue
            img = scene.image(b["frame_id"])
            if hashlib.sha256(encode_jpeg(img)).hexdigest() != b["sha256"]:
                raise RuntimeError(f"{gid} {b['photo_id']}: the rebuilt image does not match the recorded sha256")
            parts.append(("image", b["photo_id"], img))
        turns.append({"role": m["role"], "parts": parts})
    return msgs[0]["content"], turns


def mid_states(data, path: Path, step: int = MID_STEP) -> list[dict]:
    """2 states at step `step` in the call log: for ARKit and ScanNet++, the first episode whose earlier replies are all single JSON objects."""
    found = {}
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            if x.get("type") != "call" or x["tag"].get("t") != step:
                continue
            gid, msgs = x["tag"]["episode"], x["request"]["messages"]
            ds = dataset_of_scene(data.groups[gid]["scene_id"])
            if ds not in found and all(classify(m["content"])["one_json"] for m in msgs if m["role"] == "assistant"):
                system, turns = rebuild(data, gid, msgs)
                found[ds] = {"id": f"{gid}_t{step}", "system": system, "turns": turns}
    if len(found) < 2:
        raise RuntimeError(f"{path}: states at step {step} cover fewer than two datasets: {sorted(found)}")
    return [found[k] for k in sorted(found)]


def schedule(variants: list, n: int, states: list) -> list:
    """[(round, variant, start or mid, state)]: interleaved by round."""
    start, mids = states[0], states[1:]
    return [(i, v, kind, st) for i in range(n) for kind, st in (("start", start), ("mid", mids[i % len(mids)])) for v in variants]


def run_diag(data, short: str, variants: list, n: int, cap: float, out: Path, source: Path) -> dict:
    ec = data.ecfg
    mcfg = ec["models"][short]
    system, _, image_turns = probe_messages(data)
    mids = mid_states(data, source)
    start = {"id": "probe_B", "system": system, "turns": image_turns}
    tracker = ApiCostTracker(mcfg["prices_usd_per_mtok"], cap)
    clients = {}
    for v in variants:
        c = make_client(variant_cfg(mcfg, v), ec["llm"], tracker, out / f"calls_{v}.jsonl")
        c.force_cache, c.parse_hook = True, parse_detail
        clients[v] = c
    res = {"model": short, "model_id": mcfg["model"], "variants": variants, "n": n, "cap_usd": cap, "states": [start["id"]] + [s["id"] for s in mids],
           "status": "complete", "error": None, "rejected": {}, "calls": [], "params": {v: c.params() for v, c in clients.items()}}
    for i, v, kind, st in schedule(variants, n, [start, *mids]):
        if v in res["rejected"]:
            continue
        t0 = time.time()
        try:
            comp = clients[v].complete(st["system"], st["turns"], tag={"diag": kind, "state": st["id"], "variant": v, "i": i})
        except CostCapExceeded as e:
            res.update(status="stopped_cost_cap", error=str(e))
            break
        except ModelError as e:
            if e.status == 400:      # this variant's request was rejected; the other variants are unaffected
                res["rejected"][v] = str(e)[:500]
                tracker.aborted = None
                continue
            res.update(status=getattr(e, "stop_status", None) if isinstance(e, K.KeysExhausted) else "stopped_server_error" if e.retryable else "stopped_fatal_error", error=str(e)[:500])
            break
        res["calls"].append({"variant": v, "kind": kind, "state": st["id"], "i": i, "finish": comp.finish_reason, **classify(comp.text),
                             "truncated": comp.finish_reason == "max_tokens", "refusal": comp.finish_reason == "refusal", "usage": comp.usage,
                             "cost_usd": comp.cost_usd, "latency_s": round(time.time() - t0, 3), "reply": comp.text[:2000]})
    res["spent_usd"] = round(tracker.spent, 6)
    return res


def _median(xs):
    return statistics.median(xs) if xs else None


def stats(calls: list) -> dict:
    n = len(calls)
    return {"n": n, **{k: sum(c[k] for c in calls) for k in ("one_json", "valid", "markup", "multi_action", "truncated", "refusal")},
            "output": _median([c["usage"]["output_tokens"] for c in calls]), "thinking": _median([c["usage"]["reasoning_tokens"] for c in calls]),
            "input": _median([c["usage"]["input_tokens"] for c in calls]), "cost": sum(c["cost_usd"] for c in calls)}


def injected(res: dict, v: str) -> float | None:
    """Extra input tokens of v over v0: the difference of medians per state, then the median. For structured outputs this approximates the system prompt added server-side."""
    diffs = []
    for s in res["states"]:
        a = [c["usage"]["input_tokens"] for c in res["calls"] if c["state"] == s and c["variant"] == v]
        b = [c["usage"]["input_tokens"] for c in res["calls"] if c["state"] == s and c["variant"] == "v0"]
        if a and b:
            diffs.append(_median(a) - _median(b))
    return _median(diffs)


def render(res: dict, stamp: str, commit: str) -> str:
    L = [f"# Claude reply-format diagnostics: {res['model']}", "",
         f"- Model `{res['model_id']}`; run time {stamp}; code `{commit}`; status `{res['status']}`.",
         f"- Per variant: start x{res['n']}, mid x{res['n']}; spent ${res['spent_usd']:.4f}, cap ${res['cap_usd']:.2f}.",
         f"- States: {', '.join(f'`{s}`' for s in res['states'])}."]
    if res["error"]:
        L.append(f"- Stop reason: {res['error']}")
    L += ["", "| Variant | Request | Calls | Exactly one JSON | Valid action parsed | Contains `<invoke` or `<parameter` | Multiple actions at once | Limit filled | Refusals | Median output tokens | Median thinking tokens | Median input tokens | Cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else f"{x:,.0f}"
    for v in res["variants"]:
        for kind in ("start", "mid", "total"):
            s = stats([c for c in res["calls"] if c["variant"] == v and kind in (c["kind"], "total")])
            cell = lambda k: f"{s[k]} / {s['n']}"
            L.append(f"| {v} {VARIANTS[v]} | {kind} | {s['n']} | {cell('one_json')} | {cell('valid')} | {cell('markup')} | {cell('multi_action')} | {cell('truncated')} | "
                     f"{cell('refusal')} | {fmt(s['output'])} | {fmt(s['thinking'])} | {fmt(s['input'])} | ${s['cost']:.4f} |")
    L.append("")
    for v in res["variants"]:
        if v != "v0" and injected(res, v) is not None:
            L.append(f"- Extra input tokens of {v} over v0 (same state, median): {injected(res, v):,.0f}")
    for v, e in res["rejected"].items():
        L.append(f"- **{v} rejected**: {e}")
    L += ["", "**Replies that are not a single JSON object (up to 3 per variant, first 200 characters)**", ""]
    for v in res["variants"]:
        for c in [c for c in res["calls"] if c["variant"] == v and not c["one_json"]][:3]:
            L.append(f"- {v} {c['kind']} `{c['finish']}`: `{json.dumps(c['reply'][:PREVIEW_CHARS], ensure_ascii=False)}`")
    return "\n".join(L) + "\n"


def code_commit() -> str:
    return subprocess.run(["git", "-C", str(R.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--variants", default="v0,v1,v2")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--source", help="calls.jsonl to take the mid-episode states from (default haiku-5.5 active_v2 all_r0)")
    ap.add_argument("--report", help="path of the summary table (default reports/eval/api_format_check_<short name>.md)")
    ap.add_argument("--key-file", help=f"key file (default {K.DEFAULT_KEY_FILE})")
    a = ap.parse_args()
    variants = a.variants.split(",")
    if set(variants) - set(VARIANTS):
        sys.exit(print(f"unknown variants {sorted(set(variants) - set(VARIANTS))}; choices {list(VARIANTS)}") or 2)
    if a.n < 1 or a.cap <= 0:
        sys.exit(print("--n must be a positive integer and --cap must be greater than 0") or 2)
    data = FrozenData(verify=False)
    reg = R.merge(data)
    e = R.entries(reg).get(a.model)
    if e is None or e["provider"] != "anthropic":
        sys.exit(print(f"{a.model} is not an anthropic model in the registry") or 2)
    if not e.get("mock") and not os.environ.get("SLURM_JOB_ID"):
        sys.exit(print("real requests do not run on the login node: submit to a compute node with srun or sbatch") or 2)
    root = R.results_root(reg)
    try:
        print(K.setup_keys(root, a.model, e, a.key_file)["note"], flush=True)
    except K.KeyFileError as err:
        sys.exit(print(f"stopped: {err}") or 2)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root / "_format_diag" / a.model / stamp
    out.mkdir(parents=True)
    source = Path(a.source) if a.source else root.joinpath(SOURCE[0], SOURCE[1], SOURCE[2], "calls.jsonl")
    try:
        res = run_diag(data, a.model, variants, a.n, a.cap, out, source)
    except RealCallsDisabled as err:
        sys.exit(print(err) or 2)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    text = render(res, stamp, code_commit())
    (out / "summary.md").write_text(text)
    report = Path(a.report) if a.report else R.ROOT / "reports" / "eval" / f"api_format_check_{a.model}.md"
    report.write_text(text)
    print(text)
    print(f"details: {out}\nsummary table: {report}")
    sys.exit({"stopped_cost_cap": EXIT_CAP, "stopped_fatal_error": EXIT_FATAL, "stopped_server_error": EXIT_SERVER, K.QUOTA_STATUS: EXIT_QUOTA}.get(res["status"], 0))


if __name__ == "__main__":
    main()
