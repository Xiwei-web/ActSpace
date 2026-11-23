"""Data part of the tiny weak-vs-strong model comparison report.

    python -m actspace_eval.report_models --out <draft.md> [--set <model>:<setting>=<run>[,<run>…] …]

Reads only run artifacts and the frozen data; calls no API.
- Three models (weak -> strong: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol) × blind, passive_40, random_walk_40 (3 seeds),
  active_v2 (effort = high). sol reuses earlier runs: blind, passive_40 and random_walk_40 at medium, active_v2 at high
  (s3r_sol_high_active_v2_b40).
- Score table: total score, answer rate, selective score, correct-refusal rate on unanswerable questions (score.answer_metrics),
  with 95% scene-bootstrap intervals; the main table drops the degenerate questions of configs/eval_exclude.yaml, the appendix
  table keeps them.
- Differences: Active − random_walk_40 and Active − Passive-40 per model; between models (terra − luna, sol − terra, sol − luna)
  the total-score difference per setting and the difference of the two differences above. All models share the same scene
  resamples (fixed Boot seed, same scenes), so differences are paired.
- Active behaviour comparison: report_stage3r.behavior.
- Cost: recomputed with cost.recost_log including cache writes (older runs without raw usage get an upper bound) and compared
  with the pre-run estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cost import CostTracker, recost_log
from .data import FrozenData
from .report_revision import ANSWER_NOTE, STRICT_NOTE, build, ci
from .report_stage3r import behavior, behavior_table
from .score import load_exclusions, question_items

MODELS = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol")   # weak -> strong
SHORT = {"gpt-6-luna": "luna", "gpt-6.1-terra": "terra", "gpt-6.1-sol": "sol"}
SETTINGS = ("blind", "passive_40", "random_walk_40", "active_v2_high")
SLABEL = {"blind": "Blind", "passive_40": "Passive-40", "random_walk_40": "random_walk_40", "active_v2_high": "Active v2 (high)"}


def _new_runs(p):
    return {"blind": [f"{p}_blind"], "passive_40": [f"{p}_passive_b40"], "random_walk_40": [f"{p}_random_walk_40_s{k}" for k in range(3)],
            "active_v2_high": [f"{p}_active_v2_b40"]}


RUNS = {"gpt-6-luna": _new_runs("s3m_luna"), "gpt-6.1-terra": _new_runs("s3m_terra"),
        "gpt-6.1-sol": {"blind": ["s3_sol_tiny_blind"], "passive_40": ["s3_sol_tiny_passive_b40"],
                        "random_walk_40": [f"s3r_sol_random_walk_40_s{k}" for k in range(3)],
                        "active_v2_high": ["s3r_sol_high_active_v2_b40"]}}
# pre-run cost estimates (including cache writes): (estimate, upper bound)
ESTIMATES = {"gpt-6.1-terra": {"blind": (0.05, 0.12), "passive_40": (0.65, 1.62), "random_walk_40": (1.45, 3.22), "active_v2_high": (4.96, 12.29)},
             "gpt-6-luna": {"blind": (0.005, 0.01), "passive_40": (0.07, 0.16), "random_walk_40": (0.15, 0.32), "active_v2_high": (0.51, 1.23)}}
PAIRS = (("gpt-6.1-terra", "gpt-6-luna"), ("gpt-6.1-sol", "gpt-6.1-terra"), ("gpt-6.1-sol", "gpt-6-luna"))
GAIN_REFS = (("random_walk_40", "Active − random_walk_40"), ("passive_40", "Active − Passive-40"))


# ---------------------------------------------------------------- bootstrap (across models, paired)

def series(boot, items):
    """(point estimate, value per resample); same resamples as Boot.mean_ci."""
    s, c = boot.agg(items)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(s.sum() / c.sum()), (boot.w @ s) / (boot.w @ c)


def interval(mean, bs, n=None):
    bs = bs[np.isfinite(bs)]
    return {"mean": mean, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": n}


def setting_series(R, setting, sel=lambda x: True):
    return series(R["boot"], question_items(R["runs"][setting], sel))


def gain_series(R, ref, sel=lambda x: True):
    a, b = setting_series(R, "active_v2_high", sel), setting_series(R, ref, sel)
    return a[0] - b[0], a[1] - b[1]


def model_diffs(RM):
    """{row name: {pair: ci}}: total-score difference per setting, and the difference of the two Active differences."""
    rows = {}
    for st in SETTINGS:
        rows[f"{SLABEL[st]} total"] = {}
        for x, y in PAIRS:
            if x in RM and y in RM and st in RM[x]["runs"] and st in RM[y]["runs"]:
                (ma, ba), (mb, bb) = setting_series(RM[x], st), setting_series(RM[y], st)
                rows[f"{SLABEL[st]} total"][(x, y)] = interval(ma - mb, ba - bb)
    for ref, lab in GAIN_REFS:
        rows[lab] = {}
        for x, y in PAIRS:
            if all(k in RM and "active_v2_high" in RM[k]["runs"] and ref in RM[k]["runs"] for k in (x, y)):
                (ma, ba), (mb, bb) = gain_series(RM[x], ref), gain_series(RM[y], ref)
                rows[lab][(x, y)] = interval(ma - mb, ba - bb)
    return rows


# ---------------------------------------------------------------- rendering

def score_table(RM, title):
    any_R = next(iter(RM.values()))
    L = [title, "", f"Brackets: 95% scene-bootstrap interval ({len(any_R['boot'].scenes)} scenes, 2000 resamples). Blind counts answerable questions only; "
         "its prompt requires an answer (cannot_determine is not accepted), so its answer rate is 1 and no refusal rate is computed.", "",
         "| Setting | Model | Total score | Answer rate (answerable) | Selective score (answered answerable) | Unanswerable: correct-refusal rate | Evidence hit (strictly visible) |",
         "|---|---|---|---|---|---|---|"]
    for st in SETTINGS:
        for m in MODELS:
            if m not in RM or st not in RM[m]["metrics"]:
                continue
            mm, R = RM[m]["metrics"][st], RM[m]
            allq = ci(mm["all questions"])
            if len(R["seed_runs"][st]) > 1:
                seeds = [np.mean([x["score"] for x in r["per_question"]]) for r in R["seed_runs"][st]]
                allq += f"; seeds {min(seeds):.3f}–{max(seeds):.3f}"
            ev = mm["evidence"]["evidence_hit_questions"]
            L.append(f"| {SLABEL[st]} | {SHORT[m]} | {allq} | {ci(mm['answer rate (answerable)'])} | {ci(mm['selective score (answered answerable)'])} | "
                     f"{'n/a' if st == 'blind' else ci(mm['correct abstention (unanswerable)'])} | "
                     f"{'—' if st == 'blind' or ev is None else f'{100 * ev:.1f}%'} |")
    L += ["", f"- {ANSWER_NOTE}", f"- {STRICT_NOTE}",
          "- sol's Blind, Passive-40 and random_walk_40 are earlier runs with reasoning_effort = medium; terra and luna use high throughout; "
          "Active is high for all three models.", ""]
    return L


def matrix(RM, key="all questions"):
    """Total-score matrix: settings as rows, models as columns (weak -> strong); the last column marks monotonicity."""
    ms = [m for m in MODELS if m in RM]
    L = ["| Setting | " + " | ".join(SHORT[m] for m in ms) + " | luna < terra < sol |", "|---|" + "---|" * (len(ms) + 1)]
    for st in SETTINGS:
        v = [RM[m]["metrics"][st][key]["mean"] if st in RM[m]["metrics"] else None for m in ms]
        mono = "—" if None in v or len(v) < 3 else ("yes" if v[0] < v[1] < v[2] else "no")
        L.append(f"| {SLABEL[st]} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in v) + f" | {mono} |")
    for ref, lab in GAIN_REFS:
        v = [gain_series(RM[m], ref)[0] if all(s in RM[m]["runs"] for s in ("active_v2_high", ref)) else None for m in ms]
        mono = "—" if None in v or len(v) < 3 else ("yes" if v[0] < v[1] < v[2] else "no")
        L.append(f"| {lab} | " + " | ".join("—" if x is None else f"{x:+.3f}" for x in v) + f" | {mono} |")
    return L


def gain_table(RM):
    L = ["| Model | Difference | All questions | Answerable only |", "|---|---|---|---|"]
    for m in MODELS:
        if m not in RM:
            continue
        for ref, lab in GAIN_REFS:
            v = RM[m]["derived"].get("active_v2_high", {}).get(lab)
            if v:
                L.append(f"| {SHORT[m]} | {lab} | {ci(v['all'], sign=True)} | {ci(v['answerable'], sign=True)} |")
    return L


def model_diff_table(RM):
    rows = model_diffs(RM)
    L = ["| Quantity | " + " | ".join(f"{SHORT[x]} − {SHORT[y]}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    for name, d in rows.items():
        L.append(f"| {name} | " + " | ".join(ci(d[p], sign=True).split(" (n=")[0] if p in d else "—" for p in PAIRS) + " |")
    return L, rows


def cost_rows(root, spec_by_model, prices_by_model):
    out = []
    for m in MODELS:
        if m not in spec_by_model:
            continue
        for st in SETTINGS:
            names = spec_by_model[m].get(st)
            if not names:
                continue
            u, r = {}, {"logged": 0.0, "cost": 0.0, "upper": 0.0, "worst": 0.0, "calls": 0, "raw_calls": 0}
            secs = 0.0
            for n in names:
                for k, v in CostTracker.usage_in_log(root / n / "calls.jsonl").items():
                    u[k] = u.get(k, 0) + v
                for k, v in recost_log(root / n / "calls.jsonl", prices_by_model[m]).items():
                    r[k] += v
                stp = root / n / "status.json"
                secs += (json.loads(stp.read_text()).get("seconds") or 0) if stp.exists() else 0
            out.append({"model": m, "setting": st, "runs": len(names), "usage": u, **r, "seconds": secs,
                        "estimate": ESTIMATES.get(m, {}).get(st)})
    return out


def cost_table(rows):
    L = ["| Model | Setting | Runs | Calls | Input | of which cache hits | of which cache writes | Output (of which reasoning) | Logged cost | Cost incl. writes | Upper bound (all as writes) | Pre-run estimate (upper) | Time |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {}
    for r in rows:
        u = r["usage"]
        raw = r["raw_calls"] == r["calls"]
        cost = f"${r['cost']:.3f}" if raw else f"≤ ${r['upper']:.3f} (no raw usage, upper bound)"
        est = "—" if r["estimate"] is None else f"${r['estimate'][0]:.2f} (${r['estimate'][1]:.2f})"
        L.append(f"| {SHORT[r['model']]} | {SLABEL[r['setting']]} | {r['runs']} | {u.get('calls', 0):,} | {u.get('input_tokens', 0):,} | "
                 f"{u.get('cached_input_tokens', 0):,} | {u.get('cache_write_tokens', 0):,}{'' if raw else ' (not recorded)'} | "
                 f"{u.get('output_tokens', 0):,} ({u.get('reasoning_tokens', 0):,}) | ${r['logged']:.3f} | {cost} | ${r['worst']:.3f} | {est} | {r['seconds'] / 60:.0f} min |")
        t = tot.setdefault(r["model"], {"logged": 0.0, "cost": 0.0, "upper": 0.0, "worst": 0.0, "est": 0.0, "est_hi": 0.0, "raw": True})
        t["logged"] += r["logged"]
        t["cost"] += r["cost"]
        t["upper"] += r["upper"]
        t["worst"] += r["worst"]
        t["raw"] &= raw
        if r["estimate"]:
            t["est"] += r["estimate"][0]
            t["est_hi"] += r["estimate"][1]
    for m, t in tot.items():
        cost = f"**${t['cost']:.2f}**" if t["raw"] else f"≤ ${t['upper']:.2f}"
        est = f"${t['est']:.2f} (${t['est_hi']:.2f})" if t["est"] else "—"
        L.append(f"| **{SHORT[m]} total** | | | | | | | | ${t['logged']:.2f} | {cost} | ${t['worst']:.2f} | {est} | |")
    return L, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="override RUNS: <model>:<setting>=<run>[,<run>…]")
    args = ap.parse_args()
    spec = {m: dict(v) for m, v in RUNS.items()}
    for x in args.set:
        k, v = x.split("=", 1)
        m, st = k.split(":")
        spec.setdefault(m, {})[st] = v.split(",")
    data = FrozenData()
    root = Path(data.ecfg["paths"]["eval_runs"])
    spec = {m: {st: ns for st, ns in d.items() if all((root / n / "summary.json").exists() for n in ns)} for m, d in spec.items()}
    spec = {m: d for m, d in spec.items() if d}
    ex = load_exclusions()
    RM = {m: build(data, root, d, exclude=ex) for m, d in spec.items()}
    RF = {m: build(data, root, d) for m, d in spec.items()}
    B = {m: behavior(data, root, d["active_v2_high"], RM[m]["runs"]["active_v2_high"]) for m, d in spec.items() if "active_v2_high" in d}
    L = [f"<!-- settings and runs: {json.dumps(spec, ensure_ascii=False)} -->", ""]
    L += score_table(RM, "## 1. Main table: three models × four settings (degenerate questions removed)")
    L += ["**Total-score matrix** (point estimates; the last column shows whether the score rises monotonically with model strength)", ""] + matrix(RM) + [""]
    L += ["## 2. Active differences per model (main-table definition)", ""] + gain_table(RM) + [""]
    t, _ = model_diff_table(RM)
    L += ["## 3. Differences between models (main-table definition, 95% scene-paired bootstrap intervals)", "",
          "Total-score difference per setting, and the between-model difference of Active − random_walk_40 and Active − Passive-40.", ""] + t + [""]
    L += ["## 4. Active behaviour comparison (active_v2, effort = high)", ""] + behavior_table(B, MODELS, SHORT) + [""]
    for m, b in B.items():
        if b["notes_ex"]:
            L.append(f"- {SHORT[m]} notes examples: " + "; ".join(f"{g} step {tt} \"{n[:120]}\"" for g, tt, n in b["notes_ex"]))
    prices = {m: data.ecfg["models"][m]["prices_usd_per_mtok"] for m in spec}
    ct, _ = cost_table(cost_rows(root, spec, prices))
    L += ["", "## 5. Cost (including cache writes)", ""] + ct + [""]
    L += score_table(RF, "## Appendix A: scores without removing degenerate questions")
    L += ["**Total-score matrix (nothing removed)**", ""] + matrix(RF) + [""]
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
