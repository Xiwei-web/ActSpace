"""Score reports of the open-model evaluation runs: reads only run directories; starts no server and uses no GPU.

    python -m actspace_eval.report_open_full tiny  --model <model> --prefix full --report-dir reports/eval     # open/<model>_tiny.md
    python -m actspace_eval.report_open_full model --model <model> --prefix full --report-dir reports/eval     # open/<model>.md
    python -m actspace_eval.report_open_full tiny_all --prefix full --report-dir reports/eval                 # open_models_tiny.md
    python -m actspace_eval.report_open_full final --prefix full --report-dir reports/eval                    # open_models_full.md

- The three tables (ARKitScenes, ScanNet++, merged) come from report_eval.render. Active and random_walk_40 use all completed
  <prefix>_r* / <prefix>_s* runs in the directory, so later repeats 1 and 2 are merged automatically; with a single run the report
  notes that intervals do not include model sampling randomness.
- Truncation rates per model × setting, separating repetition loops from unfinished reasoning (report_open_smoke.truncation).
  Calls in calls_dropped.jsonl are not counted.
- Smoke results with max_tokens 16384 are reference only and do not enter these tables.
- Per model × setting the report also lists timeout truncations (timeout_truncated: a single call exceeded the time limit),
  context overflows (context_overflow: the episode ended early), the loop-truncation rate, and the selective score without
  truncated groups (truncated group = the group's last call was truncated, including timeouts; scored as usual, analysis only).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import content_order
from .data import load_eval_config, results_root
from .report_eval import LABELS, SCOPES, build, items, render
from .report_open_smoke import TRUNCATED, calls_file, truncation

# setting -> (llm_run --mode, budget, repeated runs)
SETTINGS = {"blind": ("blind", 40, False), "passive_8": ("passive", 8, False), "passive_20": ("passive", 20, False), "passive_40": ("passive", 40, False),
            "random_walk_40": ("random_walk", 40, True), "oracle_min": ("oracle_min", 40, False), "oracle_route": ("oracle_route", 40, False),
            "active_v2": ("active_v2", 40, True)}


TIMEOUT_NOTE = ("Timeout truncation: a single call exceeded the time limit and is scored as truncated. The limit is per model: for vLLM "
                "models 3 × 32768 ÷ the model's normal decoding speed (see the top of the report); 600 s for closed APIs and Cambrian-S. ")


def native_context(model_path) -> int | None:
    """Native context of a model: max_position_embeddings in config.json (for multimodal models the language part,
    text_config / llm_config)."""
    p = Path(model_path) / "config.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    for k in ("text_config", "llm_config"):
        if isinstance(c.get(k), dict) and c[k].get("max_position_embeddings"):
            return int(c[k]["max_position_embeddings"])
    return int(c["max_position_embeddings"]) if c.get("max_position_embeddings") else None


def served_context(ec, model) -> int | None:
    """Effective max_model_len: for vLLM models min(config, open_models.max_model_len_cap) (as in open_serve.vllm_args);
    otherwise the model config's max_model_len."""
    m = ec["models"][model]
    v = m.get("vllm")
    if v:
        return min(int(v["max_model_len"]), int(ec["open_models"]["max_model_len_cap"]))
    return m.get("max_model_len")


def max_input(root: Path, model: str, spec: dict) -> int:
    """Maximum input tokens over all calls of the evaluation runs (successful calls only)."""
    pat = re.compile(r'"input_tokens": (\d+)')
    best = 0
    for names in spec.values():
        for n in names:
            p = calls_file(root / model / n)
            if p.exists():
                with open(p) as f:
                    for line in f:
                        for x in pat.findall(line):
                            best = max(best, int(x))
    return best


def context_rows(root: Path, models, prefix: str, overflow=None) -> list[str]:
    """Per model: native context, effective max_model_len, and the maximum input of the evaluation runs."""
    ec = load_eval_config()
    L = ["| Model | Native context | Effective max_model_len | Smaller? | Max input (evaluation runs) | Context overflows (episodes) |", "|---|---|---|---|---|---|"]
    for m in models:
        if m not in ec["models"]:
            continue
        nat, srv = native_context(ec["models"][m].get("path") or ""), served_context(ec, m)
        spec = model_spec(root, m, prefix)
        cmp = "—" if nat is None or srv is None else ("smaller" if srv < nat else ("equal" if srv == nat else "larger"))
        f = lambda v: "—" if v is None else f"{v:,}"
        L.append(f"| {m} | {f(nat)} | {f(srv)} | {cmp} | {f(max_input(root, m, spec)) if spec else '—'} | {f((overflow or {}).get(m))} |")
    return L


def run_name(prefix: str, setting: str, seed: int) -> str:
    """Run names follow _r0 / _s0 (random_walk_40 uses _s<seed>, all others _r<seed>)."""
    return f"{prefix}_{'s' if setting == 'random_walk_40' else 'r'}{seed}"


def runs_of(root: Path, model: str, setting: str, prefix: str, tiny=False, suffix="") -> list[str]:
    """All runs (setting/run name) of one model and setting. Final tables use sealed runs only; tiny drafts use runs that have
    a summary.json. suffix: appendix comparison runs (e.g. full_r0_nothink); by default only <prefix>_[rs]<seed> matches,
    so appendix runs stay out of the main tables."""
    d = root / model / setting
    if not d.is_dir():
        return []
    pat = re.compile(rf"{re.escape(prefix)}_[rs]\d+{re.escape(suffix)}$")
    need = "summary.json" if tiny else "SHA256SUMS"
    return [f"{setting}/{p.name}" for p in sorted(d.iterdir()) if pat.match(p.name) and (p / need).exists()]


def model_spec(root: Path, model: str, prefix: str, tiny=False, suffix="") -> dict:
    spec = {s: runs_of(root, model, s, prefix, tiny, suffix) for s in SETTINGS}
    return {k: v for k, v in spec.items() if v}


APPENDIX = {"_nothink": "thinking disabled (chat_template_kwargs.enable_thinking: false, all other parameters unchanged)"}


def appendix_lines(root: Path, model: str, prefix: str, level="##") -> list[str]:
    """Appendix comparisons (run names with a suffix; not in the main or paper tables): one section per suffix with the
    score tables and truncation rates. Returns [] when there is no sealed appendix run."""
    L = []
    for suf, lab in APPENDIX.items():
        spec = model_spec(root, model, prefix, suffix=suf)
        if not spec:
            continue
        R = build(root / model, spec)
        body = render(R, "x")[1:]
        L += ["", f"{level} Appendix comparison: {lab}", "", f"Run names end in `{suf}`; 1 run per setting; appendix only, not in the main or paper tables."]
        L += [("#" * len(level) + x) if x.startswith("#") else x for x in body]
        L += ["", f"{level}# Truncation rates of the appendix comparison", ""] + trunc_table(trunc_rows(root, model, spec))
        L += ["", f"Runs: {', '.join(sum(spec.values(), []))}."]
    return L


def image_runs(messages) -> list[list[dict]]:
    """Adjacent images in the request (no non-empty text in between) form one run, in order of appearance."""
    runs, cur = [], []
    for m in messages:
        for q in (m["content"] if isinstance(m["content"], list) else []):
            if q["type"] == "image":
                cur.append(q)
            elif q["type"] == "text" and q["text"].strip() and cur:
                runs.append(cur)
                cur = []
    return runs + ([cur] if cur else [])


def merged_runs(messages, imgs, ns) -> list | None:
    """When there are fewer placeholder segments than images, assign the images to the segments in order:
    [(images of this segment, tokens of this segment)…]; None if they cannot be split.
    First try the runs of adjacent images in the request (same number of segments, each divisible); otherwise (images were moved
    together during rendering) assume equal tokens per image: T = total / images, and every segment must be a multiple of T.
    Only segments whose images all have the same size count towards tokens per image, so a landscape/portrait mismatch
    cannot be averaged away."""
    runs = image_runs(messages)
    if len(runs) == len(ns) and all(n % len(g) == 0 for g, n in zip(runs, ns)):
        return list(zip(runs, ns))
    total = sum(ns)
    if total % len(imgs):
        return None
    t = total // len(imgs)
    if t == 0 or any(n % t for n in ns):
        return None
    out, i = [], 0
    for n in ns:
        out.append((imgs[i:i + n // t], n))
        i += n // t
    return out


def tokens_by_orientation(run_dir: Path) -> dict:
    """Tokens per image in turn 0, keyed by "dataset W×H": {key: [token counts…]}. When the number of placeholders does not
    match the number of images, the case is recorded under "unpaired"."""
    out = defaultdict(set)
    p = run_dir / "calls.jsonl"
    if not p.exists():
        return {}
    for line in open(p):
        r = json.loads(line)
        if r.get("type") != "call" or r["tag"].get("t") != 0:
            continue
        imgs = [q for m in r["request"]["messages"] if isinstance(m["content"], list) for q in m["content"] if q["type"] == "image"]
        text = (r.get("rendered_prompt") or {}).get("text") or ""
        ns = r.get("x_image_tokens") or [int(n) for n in re.findall(r"×(\d+)", text)]
        if not imgs:
            continue
        if len(ns) != len(imgs) and ns and len(ns) % len(imgs) == 0:
            # one image split into several placeholder segments (MiniCPM-V: original + slices, 64 each): sum adjacent segments per image
            k = len(ns) // len(imgs)
            ns = [sum(ns[i * k:(i + 1) * k]) for i in range(len(imgs))]
        ds = str(r.get("group_id") or r["tag"].get("episode")).split("_")[0]
        if len(ns) != len(imgs) and ns:
            # placeholders of several adjacent images (no text in between) merged into one segment (LLaVA-OneVision: with the
            # official template putting all images first, 9 images show up as <image>×6570): group by adjacent images in the request
            # and, if the segment counts match, divide each segment by its image count; only segments whose images share one size
            # are counted, mixed-size segments only need to be divisible
            runs = merged_runs(r["request"]["messages"], imgs, [int(n) for n in ns])
            if runs:
                for g, n in runs:
                    if len({(q["w"], q["h"]) for q in g}) == 1:
                        out[f"{ds} {g[0]['w']}×{g[0]['h']}"].add(n // len(g))
                continue
        if len(ns) != len(imgs):
            out["unpaired"].add(f"{len(ns)}/{len(imgs)}")
            continue
        for q, n in zip(imgs, ns):
            out[f"{ds} {q['w']}×{q['h']}"].add(int(n))
    return {k: sorted(v, key=str) for k, v in sorted(out.items())}


# The smoke gate only detects pipeline problems (server errors, inconsistent image tokens, incomplete records). The parse rate is
# model behaviour: it is recorded but is not a reason to pause (no parse-rate threshold, for any model); truncation rate, loop rate
# and context overflows are listed in the score reports as usual.


def smoke_checks(root: Path, model: str, run: str, settings, n_groups=None) -> dict:
    """Further smoke-gate checks: complete records (every run sealed, episodes = number of smoke groups), server errors
    (number of retried calls; a task that exhausts its retries never completes, so its record is incomplete), maximum input length
    per setting; the parse rate is recorded only."""
    calls = parsed = retried = 0
    incomplete, max_in = [], {}
    for st in settings:
        d = root / model / st / run
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        if not (d / "SHA256SUMS").exists() or (n_groups is not None and summ.get("episodes") != n_groups):
            incomplete.append(st)
        p = calls_file(d)
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("type") != "call":
                continue
            calls += 1
            parsed += bool((r.get("parse") or {}).get("json"))
            retried += bool(r.get("retries"))
            max_in[st] = max(max_in.get(st, 0), int((r.get("usage") or {}).get("input_tokens") or 0))
    return {"calls": calls, "parse_rate": round(parsed / calls, 3) if calls else 0.0, "retried_calls": retried, "incomplete": incomplete,
            "max_input": max_in}


def gate_result(root: Path, model: str, run: str, settings, n_groups=None, extra=False) -> dict:
    """Smoke gate: tokens per image are present for both ARKit portrait and ScanNet++ landscape images, and all equal.
    extra: also check complete records (smoke_checks; the parse rate is recorded only)."""
    tok = defaultdict(set)
    for s in settings:
        for k, v in tokens_by_orientation(root / model / s / run).items():
            tok[k] |= set(v)
    vals = {x for k, v in tok.items() if k != "unpaired" for x in v}
    sizes = {k.split()[1] for k in tok if k != "unpaired"}
    ok = "unpaired" not in tok and len(vals) == 1 and len(sizes) >= 2
    out = {"pass": ok, "tokens": {k: sorted(v, key=str) for k, v in sorted(tok.items())}}
    if extra:
        c = smoke_checks(root, model, run, settings, n_groups)
        why = ([] if ok else ["tokens per image differ between landscape and portrait, or cannot be paired"]) + ([f"incomplete records: {', '.join(c['incomplete'])}"] if c["incomplete"] else [])
        # images and text must keep the interleaving order of the request (content_order)
        order = {s: dict(content_order.check_run(root / model / s / run)) for s in settings}
        bad = [s for s, v in order.items() if not content_order.order_ok(Counter(v))]
        if bad:
            why.append(f"wrong image/text interleaving order: {', '.join(bad)}")
        c["order"] = order
        out.update(checks=c, why=why)
        out["pass"] = not why
    return out


def run_extra(run_dir: Path, only=None) -> dict:
    """Number of timeout truncations, number of context overflows, and truncated groups (the group's last call hit the output
    limit: finish_reason = length / max_tokens, including timeouts)."""
    out = {"timeout": 0, "overflow": 0, "trunc_groups": set()}
    last = {}
    p = calls_file(run_dir)
    if p.exists():
        for line in open(p):
            if not line.endswith("\n"):
                break
            r = json.loads(line)
            if r.get("type") != "call" or (only is not None and r.get("group_id") not in only):
                continue
            out["timeout"] += bool(r.get("timeout_truncated"))
            last[r.get("group_id")] = r.get("finish_reason")
    out["trunc_groups"] = {g for g, f in last.items() if f in TRUNCATED}
    p = run_dir / "trajectories.jsonl"
    if p.exists():
        for line in open(p):
            if line.endswith("\n") and '"episode_end"' in line and '"context_overflow"' in line:
                r = json.loads(line)
                if r.get("forced") == "context_overflow":
                    out["overflow"] += 1
    return out


def trunc_rows(root: Path, model: str, spec: dict, only=None) -> list[dict]:
    rows = []
    for s, names in spec.items():
        t = {"calls": 0, "truncated": 0, "loop": 0, "unfinished": 0, "timeout": 0, "overflow": 0}
        tg = {}
        for n in names:
            x = truncation(root / model / n, only)
            e = run_extra(root / model / n, only)
            for k in ("calls", "truncated", "loop", "unfinished"):
                t[k] += x[k]
            t["timeout"] += e["timeout"]
            t["overflow"] += e["overflow"]
            tg[n] = e["trunc_groups"]
        rows.append({"setting": s, **t, "trunc_groups": tg})
    return rows


def trunc_table(rows) -> list[str]:
    L = ["| Setting | Calls | Truncated | Truncation rate | Repetition loops | Loop truncation rate | Unfinished reasoning | Timeout truncations | Context overflows (episodes) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rate = r["truncated"] / r["calls"] if r["calls"] else 0
        lr = r["loop"] / r["calls"] if r["calls"] else 0
        L.append(f"| {LABELS[r['setting']]} | {r['calls']:,} | {r['truncated']:,} | {rate:.2%} | {r['loop']:,} | {lr:.2%} | {r['unfinished']:,} | "
                 f"{r.get('timeout', 0):,} | {r.get('overflow', 0):,} |")
    return L


def selective(R, rows) -> dict:
    """setting -> (groups, truncated groups, mean over all questions, mean without truncated groups). Means are simple
    per-question averages (an analysis view outside the main tables)."""
    out = {}
    for r in rows:
        s = r["setting"]
        if s not in R["runs"]:
            continue
        allq, keep, groups, cut = [], [], 0, 0
        for name, run in zip(R["spec"][s], R["runs"][s]):
            tg = r["trunc_groups"].get(name, set())
            gs = {x["group_id"] for x in run["per_question"]}
            groups += len(gs)
            cut += len(gs & tg)
            for x in run["per_question"]:
                allq.append(x["score"])
                if x["group_id"] not in tg:
                    keep.append(x["score"])
        out[s] = (groups, cut, sum(allq) / len(allq) if allq else None, sum(keep) / len(keep) if keep else None)
    return out


def selective_table(R, rows) -> list[str]:
    L = ["| Setting | Groups | Truncated groups | Mean (all questions) | Mean without truncated groups |", "|---|---|---|---|---|"]
    for s, (g, c, a, k) in selective(R, rows).items():
        f = lambda v: "—" if v is None else f"{v:.3f}"
        L.append(f"| {LABELS[s]} | {g:,} | {c:,} | {f(a)} | {f(k)} |")
    return L


def short_line(R, rows) -> str:
    """One line for the progress file: merged Active and random_walk_40 total scores and truncation counts."""
    boot = R["boots"].get(None)
    bits = []
    for k in ("active_v2", "random_walk_40"):
        if boot is not None and k in R["runs"]:
            c = boot.mean_ci([items(r, "total") for r in R["runs"][k]], f"{k}|total")
            if c:
                bits.append(f"{LABELS[k]} merged total {c['mean']:.3f}")
    calls, tr = sum(r["calls"] for r in rows), sum(r["truncated"] for r in rows)
    loop = sum(r["loop"] for r in rows)
    bits.append(f"truncated {tr:,}/{calls:,} (loops {loop:,})")
    to, ov = sum(r.get("timeout", 0) for r in rows), sum(r.get("overflow", 0) for r in rows)
    if to or ov:
        bits.append(f"timeout truncations {to:,}, context overflows {ov:,}")
    return ", ".join(bits)


def cmd_model(root: Path, model: str, prefix: str, report_dir: Path, tiny=False):
    ec = load_eval_config()
    spec = model_spec(root, model, prefix, tiny)
    if not spec:
        print("no usable runs")
        return
    R = build(root / model, spec, only_groups=ec["llm"]["tiny_groups"] if tiny else None)
    title = f"{model}: {'tiny 32-group draft' if tiny else 'full-set scores'}"
    L = render(R, title)
    note = ec["models"].get(model, {}).get("report_note")
    if note:
        L[1:1] = ["", f"**Note**: {note}"]
    limit = root / model / "call_timeout.json"
    if limit.exists():
        c = json.loads(limit.read_text())
        L[1:1] = ["", f"Per-call time limit: {c.get('limit_s')} s ({c.get('source')}, decoding speed {c.get('speed_tok_s')} tokens/s, {c.get('n')} calls; "
                      f"rule: 3 × 32768 ÷ speed)."]
    rows = trunc_rows(root, model, spec, set(ec["llm"]["tiny_groups"]) if tiny else None)
    L += ["", "## Truncation rates", ""]
    L += trunc_table(rows)
    L += ["", TIMEOUT_NOTE + "Context overflow: the model's own earlier replies push the next request beyond the context; the episode ends and unanswered questions score 0.",
          "", "## Selective score without truncated groups (analysis only, does not replace the main tables)", "",
          "Truncated group = the group's last call was truncated (hit max_tokens or timed out). The main tables score these groups as unanswered, as usual; "
          "here are the per-question means without them, to separate \"cannot solve\" from \"did not finish\".", ""]
    L += selective_table(R, rows)
    L += ["", f"Runs: {', '.join(sum(spec.values(), []))}. max_tokens 32768 (including reasoning); truncated replies are scored as unanswered."]
    if not tiny:
        L += appendix_lines(root, model, prefix)
    out = report_dir / "open" / f"{model}{'_tiny' if tiny else ''}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"{short_line(R, rows)}. Report `open/{out.name}`")


def cmd_final(root: Path, prefix: str, report_dir: Path, tiny=False, queues=("_queue",), name=None, title=None):
    plans = [json.loads((root / q / "plan.json").read_text()) for q in queues]
    models = list(dict.fromkeys(m for p in plans for m in p["models"]))
    jobs = [j for p in plans for j in p.get("jobs", [])]
    built, truncs = {}, {}
    only = load_eval_config()["llm"]["tiny_groups"] if tiny else None
    for m in models:
        spec = model_spec(root, m, prefix, tiny)
        if spec:
            built[m] = build(root / m, spec, only_groups=only)
            truncs[m] = trunc_rows(root, m, spec, set(only) if tiny else None)
    single = all(len(R["runs"][k]) == 1 for R in built.values() for k in ("active_v2", "random_walk_40") if k in R["runs"])
    rows = list({tuple(c[:3]): c for c in (x.split("\t") for q in queues for x in (root / q / "tasks.tsv").read_text().splitlines()[1:])}.values())   # a task re-run in a later queue: the later one wins
    bad = [r for r in rows if r[3] != "done"]
    secs = sum(j.get("seconds", 0) * j.get("gpus", 0) for j in jobs)
    L = [f"# {title or 'Open models'} — {'tiny 32-group draft' if tiny else 'evaluation results'}", "",
         f"- {len(rows)} tasks, {len(rows) - len(bad)} done. Used {len(jobs)} jobs, {secs / 3600:.0f} allocated GPU hours (including idle GPUs).",
         f"- Data merged_v1.0, {'tiny 32 groups (draft)' if tiny else '234 groups'}; max_tokens 32768 (including reasoning), truncated replies are scored as unanswered.",
         "- The main tables exclude the 7 budget_infeasible groups; they are in the appendix table of each model's report "
         "(even the ideal route cannot be completed within 40 steps).", ""]
    if single:
        L += ["**Single run so far: intervals do not include model sampling randomness.**", ""]
    ecm = load_eval_config()["models"]
    notes = [f"- {m}: {ecm[m]['report_note']}" for m in models if ecm.get(m, {}).get("report_note")]
    if notes:
        L += ["## Model notes", ""] + notes + [""]
    if bad:
        L += ["## Unfinished tasks", "", "| Model | Setting | Run | State | Reason |", "|---|---|---|---|---|"]
        L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[8]} |" for r in bad]
        L.append("")
    L += ["## Total score", "", "Brackets: 95% scene-bootstrap interval. Detailed tables in `open/<model>.md`.", ""]
    cols = [s for s in SETTINGS if any(s in R["runs"] for R in built.values())]
    for scope, lab in SCOPES:
        L += [f"### {lab}", "", "| Model | " + " | ".join(LABELS[s] for s in cols) + " | Active − random_walk_40 |", "|---|" + "---|" * (len(cols) + 1)]
        for m, R in built.items():
            boot = R["boots"].get(scope)
            if boot is None:
                continue
            cells = []
            for s in cols:
                c = boot.mean_ci([items(r, "total") for r in R["runs"][s]], f"{s}|total") if s in R["runs"] else None
                cells.append("—" if c is None else f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]")
            d = None
            if "active_v2" in R["runs"] and "random_walk_40" in R["runs"]:
                d = boot.diff_ci([items(r, "total") for r in R["runs"]["active_v2"]], "active_v2|total",
                                 [items(r, "total") for r in R["runs"]["random_walk_40"]], "random_walk_40|total")
            cells.append("—" if d is None else f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Truncation rates", "", "Each cell: truncated / calls (repetition loops, unfinished reasoning, timeout truncations).", "", "| Model | " + " | ".join(LABELS[s] for s in cols) + " |", "|---|" + "---|" * len(cols)]
    for m, tr in truncs.items():
        by = {r["setting"]: r for r in tr}
        L.append(f"| {m} | " + " | ".join(f"{by[s]['truncated']:,}/{by[s]['calls']:,} ({by[s]['loop']:,}, {by[s]['unfinished']:,}, {by[s].get('timeout', 0):,})" if s in by else "—" for s in cols) + " |")
    L += ["", "## Context overflows (episode ended early)", "", "| Model | " + " | ".join(LABELS[s] for s in cols) + " |", "|---|" + "---|" * len(cols)]
    for m, tr in truncs.items():
        by = {r["setting"]: r for r in tr}
        L.append(f"| {m} | " + " | ".join(f"{by[s].get('overflow', 0):,}" if s in by else "—" for s in cols) + " |")
    L += ["", "## Context settings", "",
          "All open models use max_model_len = min(native context of the model, 65536). Rationale: the largest input of any normal conversation is 43,150 tokens "
          "(below 65,536), and 4 × A100-40GB cannot hold a larger KV cache. The native context is max_position_embeddings in the model's config.json "
          "(for multimodal models the language part).", ""]
    L += context_rows(root, models, prefix, {m: sum(r.get("overflow", 0) for r in tr) for m, tr in truncs.items()})
    L += ["", "The maximum input counts successful calls only; overflowing requests are rejected by the server and are not included "
          "(for gemma-4-31B-it the overflowing requests had 65,628–65,911 tokens)."]
    L += ["", "## Loop truncation rate and scores without truncated groups (analysis only)", "",
          "Each cell: loop truncation rate / per-question mean without truncated groups (mean over all questions).", "",
          "| Model | " + " | ".join(LABELS[s] for s in cols) + " |", "|---|" + "---|" * len(cols)]
    for m, tr in truncs.items():
        by = {r["setting"]: r for r in tr}
        sel = selective(built[m], tr)
        cell = lambda s: "—" if s not in by or s not in sel or not by[s]["calls"] else \
            f"{by[s]['loop'] / by[s]['calls']:.1%} / {'—' if sel[s][3] is None else f'{sel[s][3]:.3f}'} ({'—' if sel[s][2] is None else f'{sel[s][2]:.3f}'})"
        L.append(f"| {m} | " + " | ".join(cell(s) for s in cols) + " |")
    L += ["", "## Tokens per image for the InternVL family", "", "The tile limit is 1 for all of them; taken from the turn-0 requests of each model's Passive-40 run.", "",
          "| Model | Tokens per image |", "|---|---|"]
    for m in models:
        if "InternVL" in m:
            names = runs_of(root, m, "passive_40", prefix, tiny)
            tok = tokens_by_orientation(root / m / names[0]) if names else {}
            L.append(f"| {m} | " + "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in tok.items()) + " |")
    if not tiny:
        for m in models:
            sub = appendix_lines(root, m, prefix, level="###")
            if sub:
                L += ["", f"## Appendix comparison: {m}"] + sub
    name = name or ("open_models_tiny.md" if tiny else "open_models_full.md")
    (report_dir / name).write_text("\n".join(L) + "\n")
    print(f"Report `{name}`, {len(bad)} unfinished tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tiny", "model", "final", "tiny_all"])
    ap.add_argument("--root", default=str(results_root()))
    ap.add_argument("--prefix", default="full")
    ap.add_argument("--model")
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--queues", nargs="+", default=["_queue"], help="queue directories included in the summary report")
    ap.add_argument("--name", help="file name of the summary report (default open_models_full.md / open_models_tiny.md)")
    ap.add_argument("--title", help="title prefix of the summary report (default \"Open models\")")
    a = ap.parse_args()
    root, rd = Path(a.root), Path(a.report_dir)
    if a.what in ("final", "tiny_all"):
        cmd_final(root, a.prefix, rd, tiny=a.what == "tiny_all", queues=a.queues, name=a.name, title=a.title)
    else:
        cmd_model(root, a.model, a.prefix, rd, tiny=a.what == "tiny")


if __name__ == "__main__":
    main()
