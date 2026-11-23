"""Summary of the open-model smoke tests: reads only the results directory and job logs; starts no server and uses no GPU.

    python -m actspace_eval.report_open_smoke --root results --prefix smoke --jobs <job id…> --out <draft.json>

- Model × setting: whether it ran through (status, N/A reason), completed episodes, number of calls, JSON parse failure rate,
  invalid replies (not a legal action or no JSON), truncations (finish_reason = length), reasoning length (tokens and characters),
  mean latency and time to first token per call, output throughput per call.
- Per model, one raw Active reply (with the first 300 characters of reasoning) and one answer reply.
- Server throughput: the "Avg prompt throughput / Avg generation throughput / Running" log lines vLLM prints every few seconds.
- GPU-hour estimate (estimate_gpu_hours): extrapolates the measured per-episode input / output tokens and call counts of the smoke
  test to tiny (32 groups) and the full set (234 groups, Active and random_walk 3 runs each). The maximum of three lower bounds:
  (1) throughput: total output tokens ÷ the server's generation throughput under high concurrency + total input tokens (the part
  not hitting the prefix cache) ÷ prefill throughput (the highest window in the log: smoke requests are sparse, so the window mean
  only reflects the arrival rate, not the prefill capacity); (2) serial: the calls of an Active episode are sequential, so
  wall clock ≥ episodes × time per episode ÷ concurrency; (3) linear: the slowest smoke run scaled by the number of batches
  (one batch = the smoke groups, same concurrency as the smoke test).
  Serial servers without vLLM logs (Cambrian-S): number of calls × mean wall clock per call in the smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import FrozenData, load_eval_config, results_root
from .open_serve import throughput

SETTINGS = ("blind", "passive_8", "passive_20", "passive_40", "random_walk_40", "oracle_min", "oracle_route", "active_v2")
_THR = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
                  r"Waiting: (\d+) reqs(?:, Preemptions: \d+)?(?:, GPU KV cache usage: ([\d.]+)%)?(?:, Prefix cache hit rate: ([\d.]+)%)?")


TRUNCATED = ("length", "max_tokens")    # output limit reached: length for OpenAI / vLLM, max_tokens for Anthropic (also counted for closed models in the paper tables)
LOOP_THRESHOLD = 0.5    # truncated replies with repeat_ratio >= this count as "repetition loop", otherwise as "unfinished reasoning"


def repeat_ratio(text: str, n=6, tail_words=2000) -> float:
    """Share of repeated n-grams at the end of the text: 1 − distinct n-grams / all n-grams. Digits are normalised ("Image 1083"
    and "Image 1084" count as the same phrase); only the last tail_words words are used."""
    w = re.sub(r"\d+", "#", text.lower()).split()[-tail_words:]
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


DIGEST_FIELDS = ("type", "group_id", "tag", "finish_reason", "timeout_truncated", "retries", "usage")


def calls_file(run_dir: Path) -> Path:
    """Call log: calls.jsonl; a results snapshot (results_snapshot) does not contain it, so read calls_digest.jsonl instead
    (one line per call: DIGEST_FIELDS, parse.json, plus repeat_ratio for truncated calls; produced by digest_call)."""
    p = run_dir / "calls.jsonl"
    return p if p.exists() or not (run_dir / "calls_digest.jsonl").exists() else run_dir / "calls_digest.jsonl"


def digest_call(r: dict) -> dict:
    """One line of calls.jsonl -> one line of calls_digest.jsonl (keeps only the fields needed for truncation, overflow,
    timeout and input-length statistics)."""
    d = {k: r[k] for k in DIGEST_FIELDS if k in r}
    d["parse"] = {"json": bool((r.get("parse") or {}).get("json"))}
    if isinstance(r.get("request"), dict) and "messages" in r["request"]:      # hash of the request content (text + SHA-256 of each image), to check after a migration that the inputs are identical
        d["request_sha256"] = hashlib.sha256(json.dumps(r["request"]["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if r.get("type") == "call" and r.get("finish_reason") in TRUNCATED:
        d["repeat_ratio"] = repeat_ratio((r.get("reasoning") or "") + " " + (r.get("response") or ""))
    return d


def truncation(run_dir: Path, only=None) -> dict:
    """Truncation statistics of one run: calls with finish_reason = length, how many of them are repetition loops vs. unfinished
    reasoning, and the repeat ratios. only: restrict to these groups."""
    p = calls_file(run_dir)
    out = {"calls": 0, "truncated": 0, "loop": 0, "unfinished": 0, "ratios": []}
    if not p.exists():
        return out
    for line in open(p):
        if not line.endswith("\n"):    # while the run is still going (tiny draft) the last line may be half-written
            break
        r = json.loads(line)
        if r.get("type") != "call" or (only is not None and r.get("group_id") not in only):
            continue
        out["calls"] += 1
        if r.get("finish_reason") in TRUNCATED:
            x = r["repeat_ratio"] if "repeat_ratio" in r else repeat_ratio((r.get("reasoning") or "") + " " + (r.get("response") or ""))
            out["truncated"] += 1
            out["ratios"].append(round(x, 3))
            out["loop" if x >= LOOP_THRESHOLD else "unfinished"] += 1
    return out


def interleave_ok(run_dir: Path) -> dict:
    """Whether images stay in place in the rendered turn-0 prompt: every photo_ ID is followed (before the next ID) by an image
    placeholder. Turn 0 of Active has a single image; Blind has none and is not checked."""
    p = run_dir / "calls.jsonl"
    out = {"checked": 0, "ok": 0}
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call" or r["tag"].get("t") != 0:
            continue
        msgs = r["request"]["messages"]
        ids = [q["photo_id"] for m in msgs if isinstance(m["content"], list) for q in m["content"] if q["type"] == "image"]
        text = (r.get("rendered_prompt") or {}).get("text") or ""
        if not ids or not text:
            continue
        out["checked"] += 1
        good, pos = True, 0
        for i, pid in enumerate(ids):
            a = text.find(pid, pos)     # search forward in order: when random_walk revisits a position the ID appears again, but no image is sent
            b = text.find(ids[i + 1], a + len(pid)) if i + 1 < len(ids) else len(text)
            seg = text[a:b if b > 0 else len(text)]
            good &= a >= 0 and bool(re.search(r"×\d+|<image>", seg))
            pos = max(a, 0) + len(pid)
        out["ok"] += int(good)
    return out


def server_throughput(log: Path) -> dict:
    """vLLM server log -> generation / prefill throughput (all windows and windows with Running >= 8), maximum concurrency,
    peak KV cache usage, prefix cache hit rate."""
    rows = []
    if not log.exists():
        return {}
    for line in open(log, errors="replace"):
        m = _THR.search(line)
        if m:
            rows.append([float(x) if x is not None else np.nan for x in m.groups()])
    if not rows:
        return {"samples": 0}
    a = np.array(rows)
    busy = a[a[:, 2] >= 8]
    out = {"samples": len(a), "max_running": int(a[:, 2].max()), "max_waiting": int(a[:, 3].max()),
           "gen_tok_s_median_active": float(np.median(a[a[:, 1] > 0, 1])) if (a[:, 1] > 0).any() else None,
           "gen_tok_s_p90": float(np.percentile(a[:, 1], 90)), "prompt_tok_s_p90": float(np.percentile(a[:, 0], 90)),
           "prompt_tok_s_max": float(a[:, 0].max()),
           "kv_usage_max_pct": float(np.nanmax(a[:, 4])) if not np.isnan(a[:, 4]).all() else None,
           "prefix_hit_last_pct": float(a[~np.isnan(a[:, 5]), 5][-1]) if (~np.isnan(a[:, 5])).any() else None}
    if len(busy):
        out |= {"busy_samples": len(busy), "gen_tok_s_busy_median": float(np.median(busy[:, 1])),
                "prompt_tok_s_busy_median": float(np.median(busy[:, 0]))}
    return out


def load_memory(log: Path) -> dict:
    """GPU memory from the vLLM startup log: model weights ("Model loading took X GiB"), available KV cache
    ("Available KV cache memory: X GiB"), estimated maximum concurrency ("Maximum concurrency for N tokens per request: Xx"), CUDA graph."""
    out = {}
    if not log.exists():
        return out
    pats = {"weights_gib": r"Model loading took ([\d.]+) GiB", "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
            "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens", "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
            "cuda_graph_gib": r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB", "max_model_len": r"max_model_len=(\d+)"}
    text = open(log, errors="replace").read()
    for k, p in pats.items():
        v = re.findall(p, text)
        if v:
            out[k] = [float(x.replace(",", "")) for x in v] if len(v) > 1 else float(v[0].replace(",", ""))
    return out


def per_episode_usage(run_dir: Path) -> dict:
    """{group_id: {calls, input_tokens, output_tokens, reasoning_tokens, seconds (sum of call latencies)}}."""
    eps = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "seconds": 0.0})
    p = run_dir / "calls.jsonl"
    if not p.exists():
        return {}
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call":
            continue
        e = eps[r["tag"]["episode"]]
        e["calls"] += 1
        e["input_tokens"] += r["usage"].get("input_tokens", 0)
        e["output_tokens"] += r["usage"].get("output_tokens", 0)
        e["reasoning_tokens"] += r["usage"].get("reasoning_tokens", 0) or 0
        e["seconds"] += r.get("latency_s", 0.0)
    return dict(eps)


def examples(run_dir: Path, n_think=300) -> dict:
    """One Active move reply (with the first n_think characters of reasoning) and one answer reply."""
    act, ans = None, None
    p = run_dir / "calls.jsonl"
    if not p.exists():
        return {}
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call":
            continue
        a = (r.get("parse") or {}).get("action")
        rec = {"episode": r["tag"]["episode"], "t": r["tag"]["t"], "response": r["response"], "reasoning_head": (r.get("reasoning") or "")[:n_think],
               "reasoning_chars": r.get("reasoning_chars"), "reasoning_tokens": r.get("reasoning_tokens"),
               "finish_reason": r.get("finish_reason"), "output_tokens": r["usage"].get("output_tokens")}
        if act is None and a not in (None, "answer") and r["tag"]["t"] >= 1:
            act = rec
        if ans is None and a == "answer":
            ans = rec
        if act and ans:
            break
    return {"action": act, "answer": ans}


def run_dirs(root: Path, model: str, prefix: str) -> dict:
    out = {}
    for s in SETTINGS:
        d = root / model / s / f"{prefix}_r0"
        if d.exists():
            out[s] = d
    return out


def image_tokens(run_dir: Path) -> dict:
    """Counts of the per-image placeholder token numbers (n in the folded "<tok>×n") in rendered prompts; for Cambrian-S the
    x_image_tokens returned by the server."""
    c = defaultdict(int)
    p = run_dir / "calls.jsonl"
    if not p.exists():
        return {}
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call":
            continue
        if r.get("x_image_tokens"):
            for n in r["x_image_tokens"]:
                c[int(n)] += 1
        elif (r.get("rendered_prompt") or {}).get("text"):
            for n in re.findall(r"×(\d+)", r["rendered_prompt"]["text"]):
                c[int(n)] += 1
    return dict(sorted(c.items()))


def estimate_gpu_hours(ec, data, model, dirs, srv, concurrency=32):
    """Extrapolated GPU hours: tiny (32 groups) and the full set (234 groups) × all settings, Active and random_walk 3 runs each.
    See the module docstring."""
    tp = (ec["models"][model].get("vllm") or {}).get("tp", 1)
    gen = srv.get("gen_tok_s_busy_median") or srv.get("gen_tok_s_p90")
    pre = srv.get("prompt_tok_s_max") or srv.get("prompt_tok_s_p90")
    hit = (srv.get("prefix_hit_last_pct") or 0) / 100
    runs = {s: throughput(d) for s, d in dirs.items()}
    smoke_wall = max((r.get("wall_s") or 0 for r in runs.values()), default=0)
    smoke_calls = sum(r["calls"] for r in runs.values())
    if not gen and not (smoke_wall and smoke_calls):
        return None
    sets = {"tiny32": len(ec["llm"]["tiny_groups"]), "all234": len(data.group_ids)}
    out = {}
    for name, n in sets.items():
        tot_out = tot_in = serial = linear = calls = 0.0
        for s, d in dirs.items():
            u = per_episode_usage(d)
            if not u:
                continue
            reps = 3 if s in ("active_v2", "random_walk_40") else 1
            k = n * reps
            o = np.mean([e["output_tokens"] for e in u.values()])
            i = np.mean([e["input_tokens"] for e in u.values()])
            tot_out += k * o
            tot_in += k * i
            calls += k * runs[s]["calls"] / len(u)
            linear = max(linear, (runs[s].get("wall_s") or 0) * n / len(u))
            if s == "active_v2":
                serial = k * np.mean([e["seconds"] for e in u.values()]) / concurrency
        thr = tot_out / gen + (tot_in * (1 - hit)) / max(pre or 1, 1) if gen else calls * smoke_wall / smoke_calls
        wall = max(thr, serial, linear) if gen else thr
        out[name] = {"output_tokens": tot_out, "input_tokens": tot_in, "calls": calls, "throughput_bound_h": thr / 3600, "serial_bound_h": serial / 3600,
                     "linear_bound_h": linear / 3600, "wall_h": wall / 3600, "gpus": tp, "gpu_hours": wall / 3600 * tp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(results_root()))
    ap.add_argument("--prefix", default="smoke", help="default run name prefix (run name <prefix>_r0)")
    ap.add_argument("--model-prefix", nargs="*", default=[], help="per-model prefix: <model>=<prefix> (for re-run models)")
    ap.add_argument("--compare-prefix", default="smoke_mt32k", help="prefix of the max_tokens comparison runs (reference only)")
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ec = load_eval_config()
    data = FrozenData(verify=False)
    root = Path(args.root)
    models = [k for k, m in ec["models"].items() if m["provider"] == "vllm"]
    res = {}
    prefix = dict(x.split("=", 1) for x in args.model_prefix)
    for m in models:
        dirs = run_dirs(root, m, prefix.get(m, args.prefix))
        if not dirs:
            continue
        logs = [root / "_jobs" / j / f"server_{m}.log" for j in args.jobs if (root / "_jobs" / j / f"server_{m}.log").exists()]
        srv = server_throughput(logs[-1]) if logs else {}
        mem = load_memory(logs[-1]) if logs else {}
        sinfo = [root / "_jobs" / j / f"server_{m}.json" for j in args.jobs if (root / "_jobs" / j / f"server_{m}.json").exists()]
        cmp_dirs = run_dirs(root, m, args.compare_prefix)
        res[m] = {"prefix": prefix.get(m, args.prefix), "runs": {s: throughput(d) for s, d in dirs.items()},
                  "truncation": {s: truncation(d) for s, d in dirs.items()},
                  "interleave": {s: interleave_ok(d) for s, d in dirs.items()},
                  "image_tokens": {s: image_tokens(d) for s, d in dirs.items()},
                  "compare": {s: {"run": throughput(d), "truncation": truncation(d),
                                  "score": json.loads((d / "summary.json").read_text()).get("mean_score_all_questions"),
                                  "score_16k": json.loads((dirs[s] / "summary.json").read_text()).get("mean_score_all_questions") if s in dirs else None}
                              for s, d in cmp_dirs.items() if (d / "summary.json").exists()},
                  "summary": {s: {k: v for k, v in json.loads((d / "summary.json").read_text()).items()
                                  if k in ("episodes", "questions", "mean_steps", "mean_score_all_questions", "forced_answers", "parse_failures",
                                           "invalid_steps", "evidence_hit_frac_answerable_questions")}
                              for s, d in dirs.items() if (d / "summary.json").exists()},
                  "examples": examples(dirs["active_v2"]) if "active_v2" in dirs else {},
                  "server": srv, "memory": mem, "server_info": json.loads(sinfo[-1].read_text()) if sinfo else None,
                  "gpu_hours": estimate_gpu_hours(ec, data, m, dirs, srv)}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
