"""Data part of the results report after the setting revision.

    python -m actspace_eval.report_stage3r --out <draft.md> [--set setting=run[,run…] …]

Reads only run outputs and the frozen data; calls no API.
- Besides the total score, the score table gives the answer rate, selective score and correct abstention rate on unanswerable questions (score.answer_metrics).
- The main table excludes the degenerate questions in configs/eval_exclude.yaml; an appendix table gives unfiltered results (score.load_exclusions).
- Settings and run names default to RUNS (the s3_sol_tiny_* runs reused from the tiny pilot and the s3r_sol_* runs of this round); --set overrides them (tests use mock runs).
- random_walk_40: the 3 seeds are scored together (one item per question per seed); the mean and range per seed are also given.
- Behaviour comparison (active_v1 / active_v2 / active_v2_high): action distribution, farthest horizontal distance from the start, distinct viewpoints, revisit rate, "cannot move" rate,
  oracle view coverage, evidence hit (strictly visible), cannot_determine count, distribution of answer steps, reasoning tokens per step, notes usage.
- Cost: token usage and actual cost per run, compared with the pre-run estimate.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .agents import oracle_targets
from .data import MOVE_ACTIONS, FrozenData
from .report_revision import ANSWER_NOTE, LABELS, STRICT_NOTE, build, ci
from .score import load_exclusions, question_items
from .trajectory import read_episodes

RUNS = {"blind": ["s3_sol_tiny_blind"], "passive_8": ["s3r_sol_passive_b8"], "passive_20": ["s3r_sol_passive_b20"],
        "passive_40": ["s3_sol_tiny_passive_b40"], "random_walk_40": [f"s3r_sol_random_walk_40_s{k}" for k in range(3)],
        "active_v1": ["s3_sol_tiny_active_b40"], "active_v2": ["s3r_sol_active_v2_b40"], "active_v2_high": ["s3r_sol_high_active_v2_b40"],
        "oracle_min": ["s3_sol_tiny_oracle"], "oracle_route": ["s3r_sol_oracle_route"]}
# pre-run estimates (gpt-6.1-sol standard price): (estimate, cap); high has two cases, reasoning ×3 / ×8
ESTIMATES = {"passive_8": (0.36, None), "passive_20": (0.56, None), "oracle_route": (0.48, None), "random_walk_40": (1.92, None),
             "active_v2": (7.94, 8.66), "active_v2_high": ((9.04, 11.80), (9.81, 12.71))}
PHASES = ((0, 0), (1, 9), (10, 19), (20, 29), (30, 40))
PHASE_LABELS = [f"{a}–{b}" if a != b else str(a) for a, b in PHASES]
ACTIVE_ORDER = ("active_v1", "active_v2", "active_v2_high")
SHORT = {"passive_40": "P-40", "random_walk_40": "RW-40 (mean of 3 seeds)", "active_v1": "A v1", "active_v2": "A v2", "active_v2_high": "A v2-high",
         "oracle_min": "oracle_min", "oracle_route": "oracle_route"}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def c2(c):
    return ci(c, 2).split(" (n=")[0] if c else "—"


# ---------------------------------------------------------------- behaviour

def behavior(data, root, names, run):
    """Behaviour statistics of one Active setting (may span several runs; here only used for a single run). run is the merged result with degenerate questions excluded (used for evidence hit and cannot_determine)."""
    eps = [ep for n in names for ep in read_episodes(root / n / "trajectories.jsonl")]
    calls = [json.loads(l) for n in names for l in open(root / n / "calls.jsonl")]
    calls = [c for c in calls if c["type"] == "call"]
    acts, moved, failed, revisit, invalid = Counter(), 0, 0, 0, 0
    uniq, dmax, steps, forced, tseen, ttot, notes_steps, notes_eps, notes_ex = [], [], [], Counter(), 0, 0, 0, 0, []
    for ep in eps:
        s, end = ep[0], ep[-1]
        g = data.groups[s["group_id"]]
        sc = data.scene(g["scene_id"])
        p0 = sc.pose[s["start_node"]][0][:2]
        visited, nodes = {s["start_node"]}, [s["start_node"]]
        n_notes = 0
        for r in ep:
            if r["type"] != "step":
                continue
            acts[r["action"]] += 1
            if r["action"] not in MOVE_ACTIONS:
                invalid += 1
            elif r["moved"]:
                moved += 1
                revisit += r["to"] in visited
                visited.add(r["to"])
                nodes.append(r["to"])
            else:
                failed += 1
            if r.get("notes"):
                n_notes += 1
                if len(notes_ex) < 4:
                    notes_ex.append((s["group_id"], r["t"], r["notes"]))
        notes_steps += n_notes
        notes_eps += n_notes > 0 or bool(end.get("notes"))
        uniq.append(len(visited))
        dmax.append(max(float(np.linalg.norm(sc.pose[n][0][:2] - p0)) for n in nodes))
        steps.append(end["steps_used"])
        forced[str(end["forced"])] += 1
        tg = set(oracle_targets(g))
        tseen += len(tg & visited)
        ttot += len(tg)
    moves = moved + failed
    ans = [x for x in run["per_question"] if x["answerable"]]
    reas = defaultdict(list)
    per_ep_calls = defaultdict(list)
    for c in calls:
        per_ep_calls[c["tag"]["episode"]].append(c)
    for cs in per_ep_calls.values():
        cs = sorted(cs, key=lambda c: c["tag"]["t"])
        for c in cs[:-1]:
            t = c["tag"]["t"]
            reas[next(lab for (a, b), lab in zip(PHASES, PHASE_LABELS) if a <= t <= b)].append(c["usage"]["reasoning_tokens"])
        reas["answer"].append(cs[-1]["usage"]["reasoning_tokens"])
    step_r = [c["usage"]["reasoning_tokens"] for cs in per_ep_calls.values() for c in sorted(cs, key=lambda c: c["tag"]["t"])[:-1]]
    return {"episodes": len(eps), "actions": acts, "n_actions": sum(acts.values()), "moves": moves, "failed": failed, "moved": moved,
            "revisit": revisit, "invalid": invalid, "uniq": uniq, "dmax": dmax, "steps": steps, "forced": forced,
            "targets_seen": tseen, "targets": ttot,
            "evidence_hit": float(np.mean([x["evidence_hit"] for x in ans])) if ans else None,
            "cd_answerable": sum(x["answer"] == "cannot_determine" for x in ans), "n_answerable": len(ans),
            "reasoning_phase": {k: (float(np.mean(v)), float(np.median(v)), len(v)) for k, v in reas.items()},
            "reasoning_step": (float(np.mean(step_r)), float(np.median(step_r))) if step_r else (None, None),
            "notes_steps": notes_steps, "notes_eps": notes_eps, "notes_ex": notes_ex}


def usage(root, names):
    tot = Counter()
    for n in names:
        st = json.loads((root / n / "status.json").read_text())
        tot.update({k: st["usage"][k] for k in ("calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "errors")})
        tot["spent"] += st["spent_usd"]
        tot["seconds"] += st.get("seconds") or 0
        tot["complete"] += st["status"] == "complete"
        tot["runs"] += 1
    return tot


# ---------------------------------------------------------------- rendering

def score_tables(R, title):
    M, order = R["metrics"], [k for k in LABELS if k in R["metrics"]]
    L = [title, "", f"Brackets are scene-level bootstrap 95% intervals ({len(R['boot'].scenes)} scenes, 2000 draws). Blind counts answerable questions only.", "",
         "| Setting | All questions | Answer rate (answerable) | Selective score (answered answerable) | MC accuracy (T2–T5, answerable) | T1 MRA (answerable) | "
         "Unanswerable: correct abstention rate | Evidence hit (strictly visible) |", "|---|---|---|---|---|---|---|---|"]
    for k in order:
        m = M[k]
        ev = m["evidence"]["evidence_hit_questions"]
        allq = ci(m["all questions"])
        if len(R["seed_runs"][k]) > 1:
            seeds = [np.mean([x["score"] for x in r["per_question"]]) for r in R["seed_runs"][k]]
            allq += f"; seed mean {np.mean(seeds):.3f}, range {min(seeds):.3f}–{max(seeds):.3f}"
        L.append(f"| {LABELS[k]} | {allq} | {ci(m['answer rate (answerable)'])} | {ci(m['selective score (answered answerable)'])} | "
                 f"{ci(m['MC accuracy (T2–T5, answerable)'])} | {ci(m['T1 MRA (answerable)'])} | "
                 f"{ci(m['correct abstention (unanswerable)']) if k != 'blind' else 'n/a'} | {'—' if k == 'blind' or ev is None else pct(ev)} |")
    keys = [f"{t} answerable" for t in ("T1", "T2", "T3", "T4", "T5")] + ["T1 unanswerable", "T4 unanswerable"]
    L += ["", "**By template** (MRA for T1, accuracy otherwise; n is the number of questions, summed over the 3 seeds for random_walk_40)", "",
          "| Setting | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for k in order:
        b = M[k]["breakdown"]
        L.append(f"| {LABELS[k]} | " + " | ".join("n/a" if (k == "blind" and "unans" in key) else
                                                  (f"{c2(b.get(key))} ({b[key]['n']})" if b.get(key) else "—") for key in keys) + " |")
    L += ["", f"- {ANSWER_NOTE}", f"- {STRICT_NOTE}", ""]
    return L


def diff_tables(R):
    L = []
    for a in [k for k in ACTIVE_ORDER if k in R["derived"]]:
        rows = R["derived"][a]
        L += [f"**{LABELS[a]}**", "", "| Difference | All questions | Answerable only |", "|---|---|---|"]
        L += [f"| {lab} | {ci(v['all'], sign=True)} | {ci(v['answerable'], sign=True)} |" for lab, v in rows.items()]
        L.append("")
    boot, runs = R["boot"], R["runs"]
    extra = [(x, y) for x, y in (("active_v2", "active_v1"), ("active_v2_high", "active_v2"), ("active_v2_high", "active_v1"))
             if x in runs and y in runs]
    if extra:
        L += ["**Between Active settings**", "", "| Difference | All questions | Answerable only |", "|---|---|---|"]
        ans = lambda x: x["answerable"]
        for x, y in extra:
            L.append(f"| {LABELS[x]} − {LABELS[y]} | {ci(boot.diff_ci(question_items(runs[x]), question_items(runs[y])), sign=True)} | "
                     f"{ci(boot.diff_ci(question_items(runs[x], ans), question_items(runs[y], ans)), sign=True)} |")
        L.append("")
    return L


def _steps_hist(steps):
    return ", ".join(f"{a}–{a + 9}: {sum(a <= x <= a + 9 for x in steps)}" for a in (0, 10, 20, 30)) + f", 40: {sum(x >= 40 for x in steps)}"


def _reason(v):
    return "—" if v is None else f"{v[0]:.0f} (median {v[1]:.0f}, n={v[2]})"


def behavior_table(B, keys=None, labels=None):
    """keys / labels: column order and names (default ACTIVE_ORDER and LABELS; report_models uses model names)."""
    ks = [k for k in (keys or ACTIVE_ORDER) if k in B]
    labels = labels or LABELS
    rows = [("Episodes", lambda b: str(b["episodes"]))]
    rows += [(f"Action {a}", lambda b, a=a: pct(b["actions"][a] / max(1, b["n_actions"]))) for a in MOVE_ACTIONS]
    rows += [
        ("Total actions (incl. invalid)", lambda b: str(b["n_actions"])),
        ("Farthest horizontal distance from start: median / mean", lambda b: f"{np.median(b['dmax']):.2f} / {np.mean(b['dmax']):.2f} m"),
        ("Distinct viewpoints (incl. start): mean", lambda b: f"{np.mean(b['uniq']):.1f}"),
        ("Revisit rate (successful moves returning to a visited viewpoint)", lambda b: pct(b["revisit"] / max(1, b["moved"]))),
        ("\"Cannot move\" rate (among move actions)", lambda b: pct(b["failed"] / max(1, b["moves"]))),
        ("Invalid replies", lambda b: str(b["invalid"])),
        ("Oracle view coverage", lambda b: f"{b['targets_seen']} / {b['targets']} ({pct(b['targets_seen'] / max(1, b['targets']))})"),
        ("Evidence hit (strictly visible, answerable)", lambda b: pct(b["evidence_hit"])),
        ("Answerable questions answered cannot_determine", lambda b: f"{b['cd_answerable']} / {b['n_answerable']}"),
        ("Mean / median steps", lambda b: f"{np.mean(b['steps']):.1f} / {np.median(b['steps']):.0f}"),
        ("Used all 40 steps", lambda b: f"{sum(x >= 40 for x in b['steps'])} / {len(b['steps'])}"),
        ("Distribution of answer steps", lambda b: _steps_hist(b["steps"])),
        ("Reasoning tokens per step: mean / median", lambda b: "—" if b["reasoning_step"][0] is None else f"{b['reasoning_step'][0]:.0f} / {b['reasoning_step'][1]:.0f}"),
    ]
    rows += [(f"Mean reasoning tokens (calls {ph})", lambda b, ph=ph: _reason(b["reasoning_phase"].get(ph)))
             for ph in PHASE_LABELS + ["answer"]]
    rows.append(("notes: steps using / episodes using", lambda b: f"{b['notes_steps']} / {b['notes_eps']}"))
    L = ["| Quantity | " + " | ".join(labels[k] for k in ks) + " |", "|---|" + "---|" * len(ks)]
    L += [f"| {name} | " + " | ".join(f(B[k]) for k in ks) + " |" for name, f in rows]
    return L


def cost_table(root, spec):
    L = ["| Setting | Runs | Complete | Calls | Input | Cached | Output | Reasoning | Actual cost | Pre-run estimate | Time |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = 0.0
    for k in [k for k in LABELS if k in spec and k in ESTIMATES]:
        u = usage(root, spec[k])
        tot += u["spent"]
        e, ub = ESTIMATES[k]
        est = (f"${e[0]:.2f} / ${e[1]:.2f} (reasoning ×3 / ×8)" if isinstance(e, tuple) else f"${e:.2f}") + \
              ("" if ub is None else f"; cap {'$%.2f / $%.2f' % ub if isinstance(ub, tuple) else '$%.2f' % ub}")
        L.append(f"| {LABELS[k]} | {u['runs']} | {u['complete']} | {u['calls']} | {u['input_tokens']:,} | {u['cached_input_tokens']:,} | "
                 f"{u['output_tokens']:,} | {u['reasoning_tokens']:,} | ${u['spent']:.2f} | {est} | {u['seconds'] / 60:.0f} min |")
    L.append(f"| **Total (this round)** | | | | | | | | **${tot:.2f}** | $20.3–23.1 (cap $24.7) | |")
    return L, tot


def per_group_table(R):
    runs = R["runs"]
    ks = [k for k in ("passive_40", "random_walk_40", "active_v1", "active_v2", "active_v2_high", "oracle_min", "oracle_route") if k in runs]
    sc = {k: defaultdict(list) for k in ks}
    for k in ks:
        for x in runs[k]["per_question"]:
            sc[k][x["group_id"]].append(x["score"])
    gids = sorted({g for k in ks for g in sc[k]})
    L = ["| Group | " + " | ".join(SHORT[k] for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for g in gids:
        L.append(f"| {g} | " + " | ".join(f"{np.mean(sc[k][g]):.2f}" if sc[k][g] else "—" for k in ks) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="override RUNS: setting=run[,run…]; with --only, use only these")
    ap.add_argument("--only", action="store_true")
    args = ap.parse_args()
    spec = {} if args.only else dict(RUNS)
    for x in args.set:
        k, v = x.split("=", 1)
        spec[k] = v.split(",")
    data = FrozenData()
    root = Path(data.ecfg["paths"]["eval_runs"])
    spec = {k: v for k, v in spec.items() if all((root / n / "summary.json").exists() for n in v)}
    ex = load_exclusions()
    main_R, full_R = build(data, root, spec, exclude=ex), build(data, root, spec)
    B = {k: behavior(data, root, spec[k], main_R["runs"][k]) for k in ACTIVE_ORDER if k in spec}
    L = [f"<!-- settings and runs: {json.dumps(spec, ensure_ascii=False)} -->", ""]
    L += score_tables(main_R, f"## 1. Main table: scores (excluding the "
                              f"{sum(1 for x in full_R['runs'][next(iter(spec))]['per_question'] if x['qid'] in ex)} of the {len(ex)} degenerate questions present in this set)")
    L += ["## 2. Differences (main-table definition, degenerate questions excluded)", ""] + diff_tables(main_R)
    L += ["## 3. Per-group scores (main-table definition)", ""] + per_group_table(main_R) + [""]
    L += ["## 4. Active behaviour comparison", ""] + behavior_table(B) + [""]
    for k, b in B.items():
        if b["notes_ex"]:
            L.append(f"- notes examples of {LABELS[k]}: " + "; ".join(f"{g} step {t} \"{n[:120]}\"" for g, t, n in b["notes_ex"]))
    L += ["", "## 5. Cost"]
    ct, _ = cost_table(root, spec)
    L += [""] + ct + [""]
    L += score_tables(full_R, "## Appendix table A: scores without excluding degenerate questions")
    L += ["## Appendix table B: differences without excluding degenerate questions", ""] + diff_tables(full_R)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
