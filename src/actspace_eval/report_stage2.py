"""Stage-2 report: reads the result JSON of score.stage2 and writes Markdown tables.

    python -m actspace_eval.report_stage2 --scores <eval_runs>/s2_scores.json --out stage2_tables.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AGENTS = {"random": "random", "scripted_oracle": "scripted_oracle", "seen_oracle": "seen_oracle (auxiliary)"}


def ci(c):
    if c is None:
        return "—"
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def tables(r: dict) -> list[str]:
    M, D, B = r["metrics"], r["derived"], r["budgets"]
    main = max(B)
    L = [f"## A. Main budget {main} steps: four settings", "",
         "Brackets are scene-level bootstrap 95% intervals (41 scenes, 2000 draws). \"All questions\" is the mean per-question score over 425 questions.", "",
         "| Agent | Setting | All questions | MC accuracy (T2–T5, answerable) | T1 MRA | unanswerable | Evidence hit (answerable) | Constraints holding |",
         "|---|---|---|---|---|---|---|---|"]
    modes = [("blind", "None", "Blind"), ("passive", str(main), f"Passive-{main}"), ("oracle", "None", "Oracle"), ("active", str(main), f"Active-{main}")]
    for a in r["agents"]:
        for m, b, label in modes:
            x = M[f"{a}|{m}|{b}"]["all"]
            c = x["consistency"]
            L.append(f"| {AGENTS[a]} | {label} | {ci(x['all questions'])} | {f3(x['MC accuracy (T2–T5, answerable)']['mean'])} | {f3(x['T1 MRA (answerable)']['mean'])} | "
                     f"{f3(x['unanswerable accuracy']['mean'])} | {x['evidence']['evidence_hit_questions']:.3f} | "
                     f"{sum(v['holds'] for v in c['by_type'].values())} / {c['determinable']} |")
    L += ["", "- Evidence hit: share of answerable questions where every evidence object is visible at least once in the given or visited photos. Blind gets no photos, so it is 0.",
          "- Constraints holding: among groups with constraints, number of constraints that hold on the agent's answers / number determinable. A missing answer, `cannot_determine` or two equal distances is undeterminable.", "",
          "## B. Budget curve (10 / 20 / 40) and feasible groups", "",
          "\"Feasible groups\" are groups with `group_route_steps` ≤ budget. The two right-hand columns count only questions in these groups.", "",
          "| Agent | Budget | Feasible groups | Passive-k all groups | Active all groups | Passive-k feasible groups | Active feasible groups |",
          "|---|---|---|---|---|---|---|"]
    for a in r["agents"]:
        for b in B:
            p, ac = M[f"{a}|passive|{b}"], M[f"{a}|active|{b}"]
            L.append(f"| {AGENTS[a]} | {b} | {r['feasible'][str(b)]} / 104 | {ci(p['all']['all questions'])} | {ci(ac['all']['all questions'])} | "
                     f"{ci(p['feasible'][str(b)]['all questions'])} ({p['feasible'][str(b)]['all questions']['n']} questions) | {ci(ac['feasible'][str(b)]['all questions'])} |")
    L += ["", "## C. Derived quantities (same agent, same budget, scene-paired bootstrap)", "",
          "- Exploration gain = Active − Passive-k; evidence gap = Oracle − Active.", "",
          "| Agent | Budget | Exploration gain (all groups) | Evidence gap (all groups) | Exploration gain (feasible groups) | Evidence gap (feasible groups) |",
          "|---|---|---|---|---|---|"]
    for a in r["agents"]:
        for b in B:
            d = D[f"{a}|{b}"]
            L.append(f"| {AGENTS[a]} | {b} | {ci(d['all']['exploration_gain'])} | {ci(d['all']['evidence_gap'])} | "
                     f"{ci(d['feasible']['exploration_gain'])} | {ci(d['feasible']['evidence_gap'])} |")
    L += ["", "## D. Active efficiency", "",
          "Efficiency = `group_route_steps` / steps used (capped at 1), computed only for groups where all evidence was seen. Secondary reference: steps used / max `oracle_steps` in the group (median).", "",
          "| Agent | Budget | Mean steps | Groups with all evidence seen (evidence hit rate) | Efficiency | Steps / max oracle_steps |", "|---|---|---|---|---|---|"]
    for a in r["agents"]:
        for b in B:
            e = M[f"{a}|active|{b}"]["all"]["efficiency"]
            eff = "—" if e["efficiency"] is None else f"{ci(e['efficiency'])} ({e['efficiency']['n']} groups)"
            L.append(f"| {AGENTS[a]} | {b} | {e['mean_steps']:.1f} | {e['groups_all_evidence_seen']} / {e['groups']} ({ci(e['group_evidence_hit_rate'])}) | "
                     f"{eff} | {e['steps_over_oracle_steps_max']:.2f} |")
    L += ["", "## E. Evidence hit and correctness (diagnostic, answerable questions)", "",
          "Multiple choice (answerable T2–T5, 305 questions): share of all multiple-choice questions \"correct with evidence seen\" and \"correct without evidence seen\", and accuracy with / without evidence seen. T1 (94 questions) gives MRA split by evidence seen.", "",
          "| Agent | Setting | Evidence hit | Correct, seen | Correct, not seen | Accuracy given seen | Accuracy given not seen | T1 MRA given seen | T1 MRA given not seen |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in r["agents"]:
        for m, b, label in modes:
            e = M[f"{a}|{m}|{b}"]["all"]["evidence"]
            L.append(f"| {AGENTS[a]} | {label} | {e['evidence_hit_questions']:.3f} | {f3(e['mc_correct_and_seen'])} | {f3(e['mc_correct_not_seen'])} | "
                     f"{f3(e['mc_acc_given_seen'])} | {f3(e['mc_acc_given_not_seen'])} | {f3(e['t1_mra_seen'])} | {f3(e['t1_mra_not_seen'])} |")
    L += ["", f"## F. Breakdowns (seen_oracle, main budget {main} steps)", "",
          "Breakdowns for the other agents are in the result JSON: random is near chance in every group, scripted_oracle is 1 in every group.", ""]
    keys = list(M[f"seen_oracle|blind|None"]["all"]["breakdown"])
    L += ["| Group | " + " | ".join(lb for _, _, lb in modes) + " |", "|---|" + "---|" * len(modes)]
    for k in keys:
        cells = []
        for m, b, _ in modes:
            c = M[f"seen_oracle|{m}|{b}"]["all"]["breakdown"][k]
            cells.append("—" if c is None else f"{c['mean']:.3f} ({c['n']})")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = json.loads(Path(args.scores).read_text())
    Path(args.out).write_text("\n".join(tables(r)) + "\n")


if __name__ == "__main__":
    main()
