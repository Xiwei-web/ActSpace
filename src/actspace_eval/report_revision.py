"""Scores and derived quantities after the setting revision.

    python -m actspace_eval.report_revision --run active_v2=s3r_sol_active_v2_b40 --run passive_40=s3_sol_tiny_passive_b40 ... [--out <draft.md>]

Reads only run outputs (summary.json, trajectories.jsonl) and the frozen data; calls no API.
- Setting name → one or more run names; the 3 seeds of random_walk_40 are merged into one setting (per-question scores averaged over seeds: one item per question per seed).
- Derived quantities (scene-level bootstrap, paired):
  exploration gain = Active − Passive-40, Active − Passive-20, Active − random_walk_40;
  evidence gap = oracle_min − Active, oracle_route − Active. Given for all questions and for answerable questions only.
  Active is active_v2 (main setting); other Active settings given as well (active_v2_high, active_v1) get their own copy.
- The score table also gives the answer rate, selective score and correct abstention rate (score.answer_metrics).
- Evidence hit uses the "strictly visible" definition: every evidence object of an answerable question is visible at least once among the viewed viewpoints (partial does not count).
- exclude: questions excluded from the main report tables (configs/eval_exclude.yaml); None = no exclusion (appendix table).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import FrozenData
from .score import Boot, exclude_questions, load_run, question_items, run_metrics

LABELS = {"blind": "Blind", "passive_8": "Passive-8", "passive_20": "Passive-20", "passive_40": "Passive-40",
          "random_walk_40": "random_walk_40 (3 seeds)", "oracle_min": "oracle_min", "oracle_route": "oracle_route",
          "active_v1": "Active v1", "active_v2": "Active v2", "active_v2_high": "Active v2 (effort = high)"}
ACTIVES = ("active_v2", "active_v2_high", "active_v1")
GAINS = (("passive_40", "Active − Passive-40"), ("passive_20", "Active − Passive-20"), ("random_walk_40", "Active − random_walk_40"))
GAPS = (("oracle_min", "oracle_min − Active"), ("oracle_route", "oracle_route − Active"))
ANSWER_NOTE = ("Answer rate: share of answerable questions given an answer (not cannot_determine); selective score: answered answerable questions only (multiple choice by correctness, T1 by MRA); "
               "correct abstention rate: share of unanswerable questions answered cannot_determine, i.e. the score on unanswerable questions.")
STRICT_NOTE = "Evidence hit uses the \"strictly visible\" definition: every evidence object of an answerable question is visible at least once among the viewed viewpoints; partial does not count."


def merge_runs(runs: list[dict]) -> dict:
    """Merge several runs of one setting (random_walk seeds): concatenate per-question / per-group records; consistency uses only the first run's answers."""
    if len(runs) == 1:
        return runs[0]
    out = dict(runs[0])
    out["per_question"] = [x for r in runs for x in r["per_question"]]
    out["per_episode"] = [x for r in runs for x in r["per_episode"]]
    out["episodes"] = sum(r["episodes"] for r in runs)
    out["merged_runs"] = [r["run"] for r in runs]
    return out


def derived(boot: Boot, runs: dict) -> dict:
    """{Active setting: {derived quantity: {"all": ci, "answerable": ci}}}, only for items with runs on both sides."""
    ans = lambda x: x["answerable"]
    out = {}
    for a in ACTIVES:
        if a not in runs:
            continue
        rows = {}
        for ref, lab in GAINS:
            if ref in runs:
                rows[lab] = {"all": boot.diff_ci(question_items(runs[a]), question_items(runs[ref])),
                             "answerable": boot.diff_ci(question_items(runs[a], ans), question_items(runs[ref], ans))}
        for ref, lab in GAPS:
            if ref in runs:
                rows[lab] = {"all": boot.diff_ci(question_items(runs[ref]), question_items(runs[a])),
                             "answerable": boot.diff_ci(question_items(runs[ref], ans), question_items(runs[a], ans))}
        out[a] = rows
    return out


def build(data, root: Path, spec: dict[str, list[str]], exclude=None) -> dict:
    seed_runs = {k: [load_run(root, n) for n in names] for k, names in spec.items()}
    if exclude:
        seed_runs = {k: [exclude_questions(r, exclude) for r in rs] for k, rs in seed_runs.items()}
    runs = {k: merge_runs(rs) for k, rs in seed_runs.items()}
    scenes = {x["scene_id"] for r in runs.values() for x in r["per_question"]}
    boot = Boot(scenes)
    return {"runs": runs, "seed_runs": seed_runs, "boot": boot, "metrics": {k: run_metrics(data, r, boot) for k, r in runs.items()},
            "derived": derived(boot, runs), "spec": spec, "excluded": sorted(exclude or ())}


def ci(c, k=3, sign=False):
    if c is None:
        return "—"
    f = f"+.{k}f" if sign else f".{k}f"
    return f"{c['mean']:{f}} [{c['lo']:{f}}, {c['hi']:{f}}] (n={c['n']})"


def render(R) -> list[str]:
    M, D = R["metrics"], R["derived"]
    order = [k for k in LABELS if k in M]
    L = ["## Scores", "", f"Brackets are scene-level bootstrap 95% intervals ({len(R['boot'].scenes)} scenes, 2000 draws). Blind counts answerable questions only.", "",
         "| Setting | Runs | All questions | Answer rate (answerable) | Selective score (answered answerable) | MC accuracy (T2–T5, answerable) | T1 MRA (answerable) | "
         "Unanswerable: correct abstention rate | Evidence hit (strictly visible) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        m = M[k]
        ev = m["evidence"]["evidence_hit_questions"]
        L.append(f"| {LABELS[k]} | {', '.join(R['spec'][k])} | {ci(m['all questions'])} | {ci(m['answer rate (answerable)'])} | "
                 f"{ci(m['selective score (answered answerable)'])} | {ci(m['MC accuracy (T2–T5, answerable)'])} | "
                 f"{ci(m['T1 MRA (answerable)'])} | {ci(m['correct abstention (unanswerable)']) if k != 'blind' else 'n/a'} | "
                 f"{'—' if k == 'blind' or ev is None else f'{100 * ev:.1f}%'} |")
    L += ["", f"- {ANSWER_NOTE}", f"- {STRICT_NOTE}", ""]
    for a, rows in D.items():
        L += [f"## Derived quantities: {LABELS[a]}", "", "| Quantity | All questions | Answerable only |", "|---|---|---|"]
        L += [f"| {lab} | {ci(v['all'], sign=True)} | {ci(v['answerable'], sign=True)} |" for lab, v in rows.items()]
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="setting=run[,run…]; settings as in LABELS")
    ap.add_argument("--out")
    args = ap.parse_args()
    spec = {}
    for x in args.run:
        k, v = x.split("=", 1)
        if k not in LABELS:
            ap.error(f"unknown setting {k}; available: {', '.join(LABELS)}")
        spec[k] = v.split(",")
    data = FrozenData()
    R = build(data, Path(data.ecfg["paths"]["eval_runs"]), spec)
    text = "\n".join(render(R)) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
