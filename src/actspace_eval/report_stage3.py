"""Data part of the stage-3 pilot report.

    python -m actspace_eval.report_stage3 --prefix s3_sol_tiny --smoke-prefix s3_sol_smoke --out <draft.md> [--json <data.json>]

Reads only run outputs (trajectories.jsonl, calls.jsonl, summary.json, status.json) and the frozen data; calls no API.
- Scoring: score.run_metrics (scene-level bootstrap over the scenes in this run). Blind contains answerable questions only.
- Active behaviour: steps, action distribution, "cannot move", revisits of viewpoints, coverage of oracle views (oracle_nodes + link_views + T4 path_views,
  i.e. agents.oracle_targets), relation between evidence hit and correctness, whether evidence was still in the image window when answering, reasoning tokens by step.
- Cost: token usage and cost per setting, comparison with the estimate, extrapolation to all 104 groups (four models, standard and batch prices),
  cost of the K = 8 + START sliding window vs keeping all images (both request variants rebuilt from the actual trajectories, tokens calibrated on measurements).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .agents import evidence_seen, oracle_targets
from .cost import CostTracker
from .data import MOVE_ACTIONS, FrozenData
from .meta import group_meta
from .models import estimate_text_tokens
from .score import Boot, efficiency_stats, load_run, question_items, run_metrics
from .trajectory import read_episodes

MODES = (("blind", "_blind", "Blind"), ("passive", "_passive_b40", "Passive-40"),
         ("oracle", "_oracle", "Oracle"), ("active", "_active_b40", "Active-40"))

# prices per 1M tokens: input / cached input / output
PRICES = {
    "gpt-6-astra": {"standard": (10.00, 1.00, 50.00), "batch": (5.00, 0.50, 25.00)},
    "gpt-6.1-sol": {"standard": (2.00, 0.10, 10.00), "batch": (1.00, 0.05, 5.00)},
    "gpt-6.1-terra": {"standard": (1.00, 0.05, 5.00), "batch": (0.50, 0.025, 2.50)},
    "gpt-6-luna": {"standard": (0.10, 0.01, 0.50), "batch": (0.05, 0.005, 0.25)},
}
IMAGE_OMITTED_PREFIX = "["
N_FULL_GROUPS = 104


def price_cost(usage, p):
    pin, pc, pout = p
    cached = usage.get("cached_input_tokens", 0)
    return ((usage["input_tokens"] - cached) * pin + cached * pc + usage["output_tokens"] * pout) / 1e6


def ci(c, k=3):
    if c is None:
        return "—"
    return f"{c['mean']:.{k}f} [{c['lo']:.{k}f}, {c['hi']:.{k}f}] (n={c['n']})"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def load_calls(path):
    return [json.loads(l) for l in open(path)]


def episodes_by_gid(path):
    return {ep[0]["group_id"]: ep for ep in read_episodes(path)}


# ---------------------------------------------------------------- request rebuilding (sliding window vs keep all)

def _msg_tokens(m, tpi):
    c = m["content"]
    if isinstance(c, str):
        return estimate_text_tokens(c), 0
    imgs = sum(p.startswith("<image ") for p in c)
    return estimate_text_tokens("".join(p for p in c if not p.startswith("<image "))), imgs


def request_tokens(msgs, tpi):
    t = i = 0
    for m in msgs:
        a, b = _msg_tokens(m, tpi)
        t, i = t + a, i + b
    return t + 4 * len(msgs) + tpi * i


def ideal_cached(prev, cur, tpi, min_tokens=1024, block=128):
    """Message prefix identical item by item to the previous request (OpenAI prefix caching rule)."""
    if prev is None:
        return 0
    n = 0
    for a, b in zip(prev, cur):
        if a != b:
            break
        n += 1
    tok = request_tokens(cur[:n], tpi) if n else 0
    return 0 if tok < min_tokens else tok // block * block


def to_full(msgs):
    """K = 8 request → keep all: replace the text placeholders "[photo_x: image no longer shown]" with images again."""
    out = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, list):
            c = [f"<image {p[1:].split(':')[0]}>" if p.startswith("[photo_") and p.endswith("image no longer shown]") else p for p in c]
        out.append({"role": m["role"], "content": c})
    return out


def window_compare(calls, tpi, prices):
    """Rebuild both history settings from the actual Active requests: returns the token totals per setting (predicted / ideal cache) and the actual usage."""
    by_ep = defaultdict(list)
    for c in calls:
        if c["type"] == "call":
            by_ep[c["tag"]["episode"]].append(c)
    agg = {k: {"pred_in": 0, "ideal_cached": 0} for k in ("k8", "full")}
    actual = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for gid, cs in by_ep.items():
        prev = {"k8": None, "full": None}
        for c in sorted(cs, key=lambda x: x["tag"]["t"]):
            k8 = c["request"]["messages"]
            for key, msgs in (("k8", k8), ("full", to_full(k8))):
                agg[key]["pred_in"] += request_tokens(msgs, tpi)
                agg[key]["ideal_cached"] += ideal_cached(prev[key], msgs, tpi)
                prev[key] = msgs
            for k in actual:
                actual[k] += c["usage"][k]
    f = actual["input_tokens"] / agg["k8"]["pred_in"]                  # calibration factor for the text / image estimate
    r = actual["cached_input_tokens"] / max(1, f * agg["k8"]["ideal_cached"])  # share of the ideal cache actually realised
    rows = {}
    for key in ("k8", "full"):
        inp = f * agg[key]["pred_in"]
        ideal = f * agg[key]["ideal_cached"]
        rows[key] = {"episodes": len(by_ep), "input_tokens": inp, "cached_ideal": ideal, "cached_realized": min(ideal * r, inp),
                     "output_tokens": actual["output_tokens"]}
        for lab, cached in (("cost_ideal", ideal), ("cost_realized", min(ideal * r, inp)), ("cost_nocache", 0.0)):
            rows[key][lab] = price_cost({"input_tokens": inp, "cached_input_tokens": cached, "output_tokens": actual["output_tokens"]}, prices)
    return {"calib_factor": f, "cache_realization": r, "actual": actual, "rows": rows}


# ---------------------------------------------------------------- Active behaviour

def active_behavior(data, meta, ep, calls_ep):
    s, end = ep[0], ep[-1]
    g = data.groups[s["group_id"]]
    scene = data.scene(g["scene_id"])
    steps = [r for r in ep if r["type"] == "step"]
    targets = set(oracle_targets(g))
    visited, revisits, moved = {s["start_node"]}, 0, 0
    image_nodes = [s["start_node"]]
    for r in steps:
        if r["moved"]:
            moved += 1
            revisits += r["to"] in visited
            visited.add(r["to"])
            image_nodes.append(r["to"])
    window = {s["start_node"]} | set(image_nodes[1:][-8:])           # viewpoints still given as images when answering (START pinned + latest 8)
    qid2pub = {v: k for k, v in s["question_ids"].items()}
    qrows = []
    for q in g["questions"]:
        if q["evidence_label"] == "unanswerable":
            continue
        qrows.append({"template": q["template"], "score": end["scores"][qid2pub[q["qid"]]],
                      "seen": evidence_seen(scene, q, visited), "in_window": evidence_seen(scene, q, window),
                      "mc": q["template"] != "T1"})
    reasoning = [c["usage"]["reasoning_tokens"] for c in sorted(calls_ep, key=lambda x: x["tag"]["t"])]
    return {"group_id": s["group_id"], "scene_id": g["scene_id"], "steps": end["steps_used"], "forced": end["forced"],
            "actions": Counter(r["action"] for r in steps),
            "cannot_move": sum((not r["moved"]) and r["action"] in MOVE_ACTIONS for r in steps),
            "invalid": sum(r["action"] not in MOVE_ACTIONS for r in steps), "moved": moved, "revisits": revisits,
            "unique_nodes": len(visited), "targets": len(targets), "targets_seen": len(targets & visited),
            "targets_in_window": len(targets & window), "route_steps": meta[s["group_id"]]["group_route_steps"],
            "mean_score": float(np.mean([end["scores"][qid2pub[q["qid"]]] for q in g["questions"]])),
            "qrows": qrows, "reasoning": reasoning, "notes": sum("notes" in r for r in ep)}


def trajectory_text(data, ep, calls_ep):
    s, end = ep[0], ep[-1]
    g = data.groups[s["group_id"]]
    targets = set(oracle_targets(g))
    by_t = {c["tag"]["t"]: c for c in calls_ep}
    L = [f"- Start {s['start_photo_id']}{' (oracle view)' if s['start_node'] in targets else ''}; "
         f"{len(targets)} oracle views; group-level shortest route {group_meta(data)[s['group_id']]['group_route_steps']} steps.", "",
         "| Step | Action | Result | Photo | Oracle view hit | Reasoning tokens of this call | notes |", "|---|---|---|---|---|---|---|"]
    seen = {s["start_node"]}
    for r in [x for x in ep if x["type"] == "step"]:
        c = by_t.get(r["t"] - 1)
        res = "moved" if r["moved"] else ("cannot move" if r["action"] in MOVE_ACTIONS else "invalid")
        rev = " (revisit)" if r["moved"] and r["to"] in seen else ""
        seen.add(r["to"])
        hit = "yes" if r["moved"] and r["to"] in targets else ""
        L.append(f"| {r['t']} | {r['action']} | {res}{rev} | {r['photo_id'] if r['moved'] else '—'} | {hit} | "
                 f"{c['usage']['reasoning_tokens'] if c else '—'} | {r.get('notes', '')} |")
    c = by_t.get(max(by_t)) if by_t else None
    L += ["", f"- Answer ({end['forced'] or 'voluntary'}): `{json.dumps(end['answers'], ensure_ascii=False)}`; "
              f"scores {json.dumps(end['scores'])}; reasoning tokens of the answer call {c['usage']['reasoning_tokens'] if c else '—'}."]
    if end.get("notes"):
        L.append(f"- Answer notes: {end['notes']}")
    return L


# ---------------------------------------------------------------- main

def build(prefix, smoke_prefix):
    data = FrozenData()
    meta = group_meta(data)
    root = Path(data.ecfg["paths"]["eval_runs"])
    runs, status, calls = {}, {}, {}
    for mode, suf, _ in MODES:
        name = prefix + suf
        runs[mode] = load_run(root, name)
        status[mode] = json.loads((root / name / "status.json").read_text())
        calls[mode] = load_calls(root / name / "calls.jsonl")
    gids = sorted({e["group_id"] for e in runs["active"]["per_episode"]})
    boot = Boot({data.groups[g]["scene_id"] for g in gids})
    metrics = {m: run_metrics(data, runs[m], boot) for m, _, _ in MODES}
    derived = {"exploration_gain": boot.diff_ci(question_items(runs["active"]), question_items(runs["passive"])),
               "evidence_gap": boot.diff_ci(question_items(runs["oracle"]), question_items(runs["active"])),
               "exploration_gain_answerable": boot.diff_ci(question_items(runs["active"], lambda x: x["answerable"]),
                                                            question_items(runs["passive"], lambda x: x["answerable"])),
               "evidence_gap_answerable": boot.diff_ci(question_items(runs["oracle"], lambda x: x["answerable"]),
                                                        question_items(runs["active"], lambda x: x["answerable"]))}
    eps = episodes_by_gid(root / (prefix + "_active_b40") / "trajectories.jsonl")
    cby = defaultdict(list)
    for c in calls["active"]:
        if c["type"] == "call":
            cby[c["tag"]["episode"]].append(c)
    beh = {g: active_behavior(data, meta, eps[g], cby[g]) for g in gids if g in eps}
    per_mode_ep = {m: {e["group_id"]: e for e in runs[m]["per_episode"]} for m, _, _ in MODES}
    tpi = data.ecfg["models"][runs["active"]["model"]]["tokens_per_image"]
    sol = PRICES["gpt-6.1-sol"]["standard"]
    wc = {"main": window_compare(calls["active"], tpi, sol)}
    if smoke_prefix and (root / (smoke_prefix + "_active_b40") / "calls.jsonl").exists():
        wc["smoke"] = window_compare(load_calls(root / (smoke_prefix + "_active_b40") / "calls.jsonl"), tpi, sol)
    return {"data": data, "meta": meta, "runs": runs, "status": status, "calls": calls, "gids": gids, "metrics": metrics,
            "derived": derived, "beh": beh, "eps": eps, "cby": cby, "per_mode_ep": per_mode_ep, "wc": wc, "boot": boot}


def render(R) -> list[str]:
    data, runs, st, M, beh, gids = R["data"], R["runs"], R["status"], R["metrics"], R["beh"], R["gids"]
    L = []
    # ---- 1 scores
    L += ["## 1. Scores per setting", "",
          f"Brackets are scene-level bootstrap 95% intervals ({len(R['boot'].scenes)} scenes, 2000 draws). Blind counts answerable questions only.", "",
          "| Setting | All questions | MC accuracy (T2–T5, answerable) | T1 MRA (answerable) | Unanswerable |", "|---|---|---|---|---|"]
    for m, _, lab in MODES:
        x = M[m]
        L.append(f"| {lab} | {ci(x['all questions'])} | {ci(x['MC accuracy (T2–T5, answerable)'])} | {ci(x['T1 MRA (answerable)'])} | "
                 f"{ci(x['unanswerable accuracy']) if m != 'blind' else 'n/a (Blind)'} |")
    L += ["", "**By template** (answerable questions; MRA for T1, accuracy otherwise)", "", "| Template | " + " | ".join(l for _, _, l in MODES) + " |",
          "|---|" + "---|" * len(MODES)]
    for t in ("T1", "T2", "T3", "T4", "T5"):
        L.append(f"| {t} answerable | " + " | ".join(ci(M[m]["breakdown"].get(f"{t} answerable"), 2) for m, _, _ in MODES) + " |")
    for t in ("T1", "T4"):
        L.append(f"| {t} unanswerable | " + " | ".join("n/a" if m == "blind" else ci(M[m]["breakdown"].get(f"{t} unanswerable"), 2)
                                              for m, _, _ in MODES) + " |")
    L += ["", "**Constraint consistency** (groups with constraints: whether the constraint holds on the model's answers)", "", "| Setting | Determinable / constraints | Share holding |", "|---|---|---|"]
    for m, _, lab in MODES:
        c = M[m]["consistency"]
        L.append(f"| {lab} | {c['determinable']} / {c['constraints']} | {pct(c['holds_frac'])} |")
    D = R["derived"]
    eff = M["active"]["efficiency"]
    L += ["", "## 2. Derived quantities, efficiency and evidence hit", "",
          "| Quantity | All questions | Answerable only |", "|---|---|---|",
          f"| Exploration gain = Active − Passive-40 | {ci(D['exploration_gain'])} | {ci(D['exploration_gain_answerable'])} |",
          f"| Evidence gap = Oracle − Active | {ci(D['evidence_gap'])} | {ci(D['evidence_gap_answerable'])} |", "",
          f"- Efficiency (group_route_steps / steps used, capped at 1; only groups with all evidence seen): {ci(eff['efficiency'])}; "
          f"groups with all evidence seen {eff['groups_all_evidence_seen']} / {eff['groups']}.",
          "- Evidence hit (share of answerable questions whose evidence objects were all visible in viewed viewpoints): " +
          "; ".join(f"{lab} {pct(M[m]['evidence']['evidence_hit_questions'])}" for m, _, lab in MODES if m != "blind") + ".", ""]
    # ---- 3 steps and parsing
    steps = [b["steps"] for b in beh.values()]
    hist = Counter(steps)
    L += ["## 3. Active steps, parsing and invalid replies", "",
          f"- Mean steps {np.mean(steps):.1f}, median {np.median(steps):.0f}, min {min(steps)}, max {max(steps)}.",
          f"- Used the full budget (40 steps): {sum(s >= 40 for s in steps)} / {len(steps)} groups ({pct(np.mean([s >= 40 for s in steps]))}).",
          "- Step distribution: " + ", ".join(f"{a}–{a + 4} steps {sum(a <= s <= a + 4 for s in steps)} groups" for a in range(0, 40, 5)
                                if any(a <= s <= a + 4 for s in steps)) + f", 40 steps {sum(s >= 40 for s in steps)} groups.", "",
          "| Setting | Calls | Episodes with parse failure (no JSON at all) | Per-question parse failures | Invalid replies (steps) | Episodes ended by 3 consecutive invalid replies | End reasons |",
          "|---|---|---|---|---|---|---|"]
    for m, _, lab in MODES:
        s = runs[m]
        perq_err = sum(1 for x in s["per_question"] if x.get("parse_error") and x["parse_error"] != "missing")
        forced = Counter(str(e["forced"]) for e in s["per_episode"])
        L.append(f"| {lab} | {st[m]['usage']['calls']} | {s['parse_failures']} / {s['episodes']} | {perq_err} | "
                 f"{sum(e.get('invalid_steps', 0) for e in s['per_episode'])} | {forced.get('invalid_replies', 0)} | "
                 f"{', '.join(f'{k} {v}' for k, v in sorted(forced.items()))} |")
    L += ["", f"- Call errors and retries: " + "; ".join(f"{lab} {st[m]['usage']['errors']}" for m, _, lab in MODES) + ".", ""]
    # ---- 4 tokens and cost
    L += ["## 4. Token usage and cost (gpt-6.1-sol standard price)", "",
          "| Setting | Calls | Input | Cached | Output | Reasoning | Cache hit rate | Actual cost | Per group |", "|---|---|---|---|---|---|---|---|---|"]
    tot = Counter()
    for m, _, lab in MODES:
        u = st[m]["usage"]
        tot.update({k: u[k] for k in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")})
        tot["spent"] += st[m]["spent_usd"]
        L.append(f"| {lab} | {u['calls']} | {u['input_tokens']:,} | {u['cached_input_tokens']:,} | {u['output_tokens']:,} | "
                 f"{u['reasoning_tokens']:,} | {pct(u['cached_input_tokens'] / max(1, u['input_tokens']))} | ${st[m]['spent_usd']:.2f} | "
                 f"${st[m]['spent_usd'] / st[m]['completed']:.3f} |")
    L.append(f"| **Total** | {tot['calls']} | {tot['input_tokens']:,} | {tot['cached_input_tokens']:,} | {tot['output_tokens']:,} | "
             f"{tot['reasoning_tokens']:,} | {pct(tot['cached_input_tokens'] / tot['input_tokens'])} | **${tot['spent']:.2f}** | |")
    L.append("")
    # ---- 5 extrapolation
    L += ["## 5. Extrapolation: all 104 groups × four settings", "",
          "- Measured per-group token usage of this run (input, cached input, output) × 104 / number of groups in this run.",
          "- Assumes every model uses the same tokens as gpt-6.1-sol. In reality reasoning tokens, tokens per image and Active steps differ between models, so this is only an order of magnitude.",
          "- **Batch prices apply only to Blind / Passive / Oracle; Active is multi-turn interaction and can only use standard prices.** The Active part of the \"batch\" column is still at standard price.", "",
          "| Model | Blind | Passive-40 | Oracle | Active-40 | Total (standard) | Total (batch for static) |", "|---|---|---|---|---|---|---|"]
    scale = N_FULL_GROUPS / len(gids)
    for model, pr in PRICES.items():
        cs = {m: price_cost(st[m]["usage"], pr["standard"]) * scale for m, _, _ in MODES}
        cb = {m: price_cost(st[m]["usage"], pr["batch"]) * scale for m in ("blind", "passive", "oracle")}
        L.append(f"| {model} | ${cs['blind']:.2f} / ${cb['blind']:.2f} | ${cs['passive']:.2f} / ${cb['passive']:.2f} | "
                 f"${cs['oracle']:.2f} / ${cb['oracle']:.2f} | ${cs['active']:.2f} | ${sum(cs.values()):.2f} | "
                 f"${sum(cb.values()) + cs['active']:.2f} |")
    L += ["", "Cells of the three static settings read \"standard / batch\".", ""]
    # ---- 6 sliding window vs keep all
    L += ["## 6. Cost of the K = 8 + START sliding window vs keeping all images (gpt-6.1-sol standard price)", "",
          "- **Method**: rebuild the same sequence of requests under both history settings from the actual Active requests (request summaries in `calls.jsonl`). Keeping all simply replaces the text placeholders with images; trajectory and replies stay unchanged.",
          "- **Tokens**: computed with the estimation rule, then multiplied by a calibration factor = actual input tokens ÷ estimate for the sliding-window setting.",
          "- **Cache**: \"ideal\" follows the prefix caching rule (prefix identical item by item to the previous request, ≥ 1024 tokens, multiple of 128); \"realised as measured\" further multiplies by actual cache ÷ ideal cache under the sliding window.",
          "- **Limitation**: with all images kept the model sees more images and may behave differently; this only compares the cost of the same trajectory.", ""]
    for lab, w in R["wc"].items():
        n = w["rows"]["k8"]["episodes"]
        L += [f"**{'Smoke' if lab == 'smoke' else 'This run'}, {n} groups** (calibration factor {w['calib_factor']:.3f}, share of ideal cache realised {w['cache_realization']:.2f}; "
              f"actual: input {w['actual']['input_tokens']:,}, cached {w['actual']['cached_input_tokens']:,})", "",
              "| Setting | Input tokens | Cached (ideal) | Cost: ideal cache | Cost: realised as measured | Cost: no cache | Extrapolated to 104 groups (realised as measured) |",
              "|---|---|---|---|---|---|---|"]
        for key, name in (("k8", "K = 8 + START"), ("full", "keep all")):
            r = w["rows"][key]
            L.append(f"| {name} | {r['input_tokens']:,.0f} | {r['cached_ideal']:,.0f} | ${r['cost_ideal']:.2f} | ${r['cost_realized']:.2f} | "
                     f"${r['cost_nocache']:.2f} | ${r['cost_realized'] * N_FULL_GROUPS / n:.2f} |")
        L.append("")
    # ---- 7 Active behaviour
    pm = R["per_mode_ep"]
    L += ["## 7. Active behaviour", "", "### 7.1 Per group", "",
          "- Oracle views = `oracle_nodes` + `link_views` + `path_views` of T4 (target viewpoints on the scripted_oracle route).",
          "- \"Still in window when answering\" is the number of oracle views among START plus the last 8 photos.", "",
          "| Group | Questions | Group-level shortest route | Steps | Full budget | Cannot move | Revisits | Distinct viewpoints | Oracle views seen / total | Still in window when answering | Active | Passive | Oracle | A−P | O−A |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in gids:
        b = beh[g]
        a, p, o = pm["active"][g]["mean_score"], pm["passive"][g]["mean_score"], pm["oracle"][g]["mean_score"]
        L.append(f"| {g} | {len(data.groups[g]['questions'])} | {b['route_steps']} | {b['steps']} | {'yes' if b['steps'] >= 40 else ''} | "
                 f"{b['cannot_move']} | {b['revisits']} | {b['unique_nodes']} | {b['targets_seen']} / {b['targets']} | {b['targets_in_window']} | "
                 f"{a:.2f} | {p:.2f} | {o:.2f} | {a - p:+.2f} | {o - a:+.2f} |")
    acts = Counter()
    for b in beh.values():
        acts.update(b["actions"])
    nmove = sum(acts[a] for a in MOVE_ACTIONS)
    cm_by = Counter()
    for g in gids:
        for r in R["eps"][g]:
            if r["type"] == "step" and r["action"] in MOVE_ACTIONS and not r["moved"]:
                cm_by[r["action"]] += 1
    moved = sum(b["moved"] for b in beh.values())
    L += ["", "### 7.2 Action distribution", "", "| Action | Count | Share | Of which \"cannot move\" | Failure rate of this action |", "|---|---|---|---|---|"]
    for a in MOVE_ACTIONS:
        L.append(f"| {a} | {acts[a]} | {pct(acts[a] / max(1, nmove))} | {cm_by[a]} | {pct(cm_by[a] / max(1, acts[a]))} |")
    L += ["", f"- {nmove} move actions: {moved} succeeded, \"cannot move\" {sum(cm_by.values())} ({pct(sum(cm_by.values()) / max(1, nmove))}). "
              f"{sum(b['invalid'] for b in beh.values())} invalid replies.",
          f"- Successful moves returning to a visited viewpoint: {sum(b['revisits'] for b in beh.values())} ({pct(sum(b['revisits'] for b in beh.values()) / max(1, moved))}).",
          f"- Steps using notes: {sum(b['notes'] for b in beh.values())}.", ""]
    q = [x for b in beh.values() for x in b["qrows"]]

    def acc(xs):
        return "—" if not xs else f"{np.mean([x['score'] for x in xs]):.3f} (n={len(xs)})"
    L += ["### 7.3 Evidence hit and correctness", "",
          f"- Oracle view coverage: seen {sum(b['targets_seen'] for b in beh.values())} / {sum(b['targets'] for b in beh.values())}"
          f" ({pct(sum(b['targets_seen'] for b in beh.values()) / sum(b['targets'] for b in beh.values()))}), "
          f"still in the image window when answering {sum(b['targets_in_window'] for b in beh.values())}.", "",
          "| Answerable questions | MC accuracy | T1 MRA |", "|---|---|---|"]
    for lab, sel in (("evidence seen", lambda x: x["seen"]), ("evidence not seen", lambda x: not x["seen"]),
                     ("seen, and still in window when answering", lambda x: x["seen"] and x["in_window"]),
                     ("seen, but out of the window when answering", lambda x: x["seen"] and not x["in_window"])):
        L.append(f"| {lab} | {acc([x for x in q if sel(x) and x['mc']])} | {acc([x for x in q if sel(x) and not x['mc']])} |")
    pq = {m: {(x["group_id"], x["qid"]): x for x in runs[m]["per_question"]} for m in ("passive", "oracle")}
    L += ["", "Evidence hit and scores of the same answerable questions in the static settings (reference):", "", "| Setting | Evidence seen: MC / T1 | Not seen: MC / T1 |", "|---|---|---|"]
    for m, lab in (("passive", "Passive-40"), ("oracle", "Oracle")):
        xs = [x for x in pq[m].values() if x["answerable"]]
        f = lambda s, mc: [x for x in xs if bool(x["evidence_hit"]) == s and (x["template"] != "T1") == mc]
        L.append(f"| {lab} | {acc(f(True, True))} / {acc(f(True, False))} | {acc(f(False, True))} / {acc(f(False, False))} |")
    # reasoning tokens by step
    bins = [(0, 0), (1, 9), (10, 19), (20, 29), (30, 40)]
    L += ["", "### 7.4 Reasoning tokens by step", "",
          "- Call index t: the t-th call decides step t + 1.",
          "- Answer calls are listed separately (voluntary answer, or asked to answer after the budget ran out).", "",
          "| Call index | Calls | Mean reasoning tokens | Median | Max | Share with zero reasoning |", "|---|---|---|---|---|---|"]
    move_calls, ans_calls = defaultdict(list), []
    for g in gids:
        cs = sorted(R["cby"][g], key=lambda c: c["tag"]["t"])
        for c in cs[:-1]:
            move_calls[c["tag"]["t"]].append(c["usage"]["reasoning_tokens"])
        if cs:
            ans_calls.append(cs[-1]["usage"]["reasoning_tokens"])
    for lo, hi in bins:
        xs = [v for t in range(lo, hi + 1) for v in move_calls.get(t, [])]
        if xs:
            L.append(f"| {lo}–{hi} | {len(xs)} | {np.mean(xs):.0f} | {np.median(xs):.0f} | {max(xs)} | {pct(np.mean([x == 0 for x in xs]))} |")
    L.append(f"| Answer | {len(ans_calls)} | {np.mean(ans_calls):.0f} | {np.median(ans_calls):.0f} | {max(ans_calls)} | "
             f"{pct(np.mean([x == 0 for x in ans_calls]))} |")
    L.append("")
    return L


def pick_trajectories(R):
    """1 good (budget not used up, highest score), 1 bad (budget not used up, lowest score), 1 that used the full budget (lowest score); relaxed if there are not enough."""
    beh = R["beh"]
    order = sorted(beh.values(), key=lambda b: (b["mean_score"], -b["steps"]))
    not_full = [b for b in order if b["steps"] < 40]
    full = [b for b in order if b["steps"] >= 40]
    good = (not_full or order)[-1]
    bad = next((b for b in not_full if b is not good), next(b for b in order if b is not good))
    exh = next((b for b in full if b not in (good, bad)), None)
    return [("good", good), ("bad", bad)] + ([("full budget", exh)] if exh else [])


def render_trajectories(R):
    L = ["## 8. Three Active trajectories", ""]
    for i, (lab, b) in enumerate(pick_trajectories(R), 1):
        g = b["group_id"]
        L += [f"### 8.{i} {lab}: {g} ({b['scene_id']}; Active {b['mean_score']:.2f}, "
              f"Passive {R['per_mode_ep']['passive'][g]['mean_score']:.2f}, Oracle {R['per_mode_ep']['oracle'][g]['mean_score']:.2f}; {b['steps']} steps)", ""]
        L += ["Questions:", ""] + [f"- {q['id']}: {q['text']}" + (f" ({' / '.join(q['options'])})" if q["options"] else "")
                               for q in __import__("actspace_eval.answers", fromlist=["x"]).public_questions(R["data"].groups[g])] + [""]
        L += trajectory_text(R["data"], R["eps"][g], R["cby"][g]) + [""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--smoke-prefix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = build(args.prefix, args.smoke_prefix)
    L = render(R) + render_trajectories(R)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}: {len(L)} lines")


if __name__ == "__main__":
    main()
