"""Score tables of the main evaluation (merged_v1.0).

    python -m actspace_eval.report_eval --out <draft.md> --set <setting>=<run>[,<run>…] [--set …] [--title <title>] [--root <dir>] [--tiny]

--tiny: only llm.tiny_groups (tiny draft while runs are still in progress).

--root: root of the run directories (default paths.eval_runs). Open-model results live in <results root>/<model>/<setting>/<run>/;
use --root <results root>/<model> and write run names as "<setting>/<run>"
(e.g. --set active_v2=active_v2/tiny_r0,active_v2/tiny_r1,active_v2/tiny_r2).

Reads only run artifacts (summary.json, trajectories.jsonl); calls no API.
- Every score table shows side by side: total score, answer rate (share of answerable questions not answered cannot_determine),
  selective score (answered answerable questions only), plus the correct-refusal rate on unanswerable questions
  (as in score.answer_metrics).
- Scores are given per dataset (ARKitScenes, ScanNet++) with a separate merged score; metrics that depend on the visibility
  definition, such as evidence hit, are reported per dataset only and never merged (the two datasets use different visibility
  thresholds).
- Intervals: score.RepBoot, combining "scene bootstrap + repeats" (with 3 runs each of Active and random_walk, the repeats are
  resampled too); the merged score resamples scenes stratified by dataset.
- Differences: Active − random_walk_40 (primary metric), Active − Passive-40 (secondary metric); scenes paired, repeats independent.
- Per-template tables (T1–T5) always use percentages with 1 decimal (e.g. 29.4), intervals likewise.
- The runs column gives the number of repeats per setting; for settings with a single run the interval uses the scene bootstrap
  only, and a table note says it excludes run-to-run variation.
- Groups whose shortest evidence route exceeds 40 steps (env.budget_infeasible, 7 ScanNet++ groups) are run as usual but excluded
  from the main tables (all tables above); an appendix table lists their scores, noting that even the ideal route cannot be
  completed within 40 steps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .data import dataset_of_scene, load_eval_config
from .score import RepBoot, answered, load_run

CANNOT = "cannot_determine"
LABELS = {"blind": "Blind", "passive_8": "Passive-8", "passive_20": "Passive-20", "passive_40": "Passive-40",
          "random_walk_40": "random_walk_40", "oracle_min": "oracle_min", "oracle_route": "oracle_route",
          "active_v1": "Active v1", "active_v2": "Active v2"}
SCOPES = (("arkitscenes", "ARKitScenes"), ("scannetpp", "ScanNet++"), (None, "Merged"))
TEMPLATES = ("T1", "T2", "T3", "T4", "T5")
REPEATED = ("active_v2", "random_walk_40")    # these two settings are repeated; the report notes when only 1 run exists
GAINS = (("random_walk_40", "Active − random_walk_40 (primary metric)"), ("passive_40", "Active − Passive-40 (secondary metric)"))


def items(run, metric, template=None):
    """[(scene, value)] of one run."""
    out = []
    for x in run["per_question"]:
        if template and x["template"] != template:
            continue
        a = x["answerable"]
        if metric == "total":
            out.append((x["scene_id"], x["score"]))
        elif metric == "answer_rate" and a:
            out.append((x["scene_id"], float(answered(x))))
        elif metric == "selective" and a and answered(x):
            out.append((x["scene_id"], x["score"]))
        elif metric == "refusal" and not a:
            out.append((x["scene_id"], float(x["answer"] == CANNOT)))
        elif metric == "evidence" and a:
            out.append((x["scene_id"], float(bool(x["evidence_hit"]))))
    return out


def ci(c, k=3, sign=False):
    if c is None:
        return "—"
    f = f"+.{k}f" if sign else f".{k}f"
    return f"{c['mean']:{f}} [{c['lo']:{f}}, {c['hi']:{f}}]"


def pct(c):
    """Percentage with 1 decimal: 29.4 [22.0, 36.0] (the format of all per-template tables)."""
    if c is None:
        return "—"
    return f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def _subset(run, keep):
    return {**run, "per_question": [x for x in run["per_question"] if keep(x["group_id"])]}


def _view(runs):
    scenes = defaultdict(set)
    for rs in runs.values():
        for r in rs:
            for x in r["per_question"]:
                scenes[dataset_of_scene(x["scene_id"])].add(x["scene_id"])
    boots = {scope: RepBoot(scenes, scope=scope) for scope, _ in SCOPES if scope is None or scope in scenes}
    return {"runs": runs, "boots": boots, "scenes": {k: len(v) for k, v in scenes.items()},
            "groups": len({x["group_id"] for rs in runs.values() for r in rs for x in r["per_question"]})}


def build(root: Path, spec: dict, infeasible=None, only_groups=None) -> dict:
    """The main tables use only groups whose ideal route fits in the budget; budget_infeasible groups go to an appendix table
    (R["infeasible"]). only_groups: restrict to these groups."""
    if infeasible is None:
        infeasible = load_eval_config()["env"].get("budget_infeasible") or []
    bad = set(infeasible)
    full = {k: [load_run(root, n) for n in names] for k, names in spec.items()}
    if only_groups is not None:
        keep = set(only_groups)
        full = {k: [_subset(r, lambda g: g in keep) for r in rs] for k, rs in full.items()}
    R = _view({k: [_subset(r, lambda g: g not in bad) for r in rs] for k, rs in full.items()})
    inf = {k: [_subset(r, lambda g: g in bad) for r in rs] for k, rs in full.items()}
    R["infeasible"] = _view(inf) if any(r["per_question"] for rs in inf.values() for r in rs) else None
    R["infeasible_ids"] = sorted(bad)
    R["spec"] = spec
    return R


def single_run_note(labels) -> str:
    return " Settings with a single run (" + ", ".join(labels) + "): the interval uses the scene bootstrap only and excludes run-to-run variation." if labels else ""


def score_table(R, scope, title):
    boot = R["boots"][scope]
    per_ds = scope is not None
    single = [LABELS[k] for k in LABELS if k in R["runs"] and len(R["runs"][k]) == 1]
    L = [title, "",
         f"Brackets: 95% interval from \"scene bootstrap + repeats\" (2000 resamples). The runs column gives the number of runs per setting (> 1: repeats are resampled)."
         + ("" if per_ds else " The merged score resamples scenes stratified by dataset; evidence hit depends on each dataset's visibility definition and is not merged.") + single_run_note(single), "",
         "| Setting | Runs | Questions | Total | Answer rate (answerable) | Selective score (answered answerable) | Unanswerable: correct abstention |" + (" Evidence hit (strictly visible) |" if per_ds else ""),
         "|---|---|---|---|---|---|---|" + ("---|" if per_ds else "")]
    for k in [k for k in LABELS if k in R["runs"]]:
        rs = R["runs"][k]
        cells = []
        n = None
        for m in ("total", "answer_rate", "selective", "refusal") + (("evidence",) if per_ds else ()):
            c = boot.mean_ci([items(r, m) for r in rs], f"{k}|{m}")
            if m == "total":
                n = c["n"] if c else 0
            cells.append("n/a" if (k == "blind" and m in ("refusal",)) else ("—" if k == "blind" and m == "evidence" else ci(c)))
        L.append(f"| {LABELS[k]} | {len(rs)} | {n} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def gain_table(R, scope):
    boot = R["boots"][scope]
    L = ["| Difference | All questions | Answer rate | Selective score |", "|---|---|---|---|"]
    if "active_v2" not in R["runs"]:
        return []
    a = R["runs"]["active_v2"]
    for ref, lab in GAINS:
        if ref not in R["runs"]:
            continue
        b = R["runs"][ref]
        cells = [ci(boot.diff_ci([items(r, m) for r in a], f"active_v2|{m}", [items(r, m) for r in b], f"{ref}|{m}"), sign=True)
                 for m in ("total", "answer_rate", "selective")]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def template_table(R, scope):
    boot = R["boots"][scope]
    L = ["| Setting | " + " | ".join(TEMPLATES) + " |", "|---|" + "---|" * len(TEMPLATES)]
    for k in [k for k in LABELS if k in R["runs"]]:
        rs = R["runs"][k]
        L.append(f"| {LABELS[k]} | " + " | ".join(pct(boot.mean_ci([items(r, "total", t) for r in rs], f"{k}|total|{t}")) for t in TEMPLATES) + " |")
    return L + ["", "In %, 1 decimal; brackets: 95% interval.", ""]


def render(R, title="Evaluation scores") -> list[str]:
    L = [f"# {title}", "", f"Scenes: " + ", ".join(f"{lab} {R['scenes'].get(s, 0)}" for s, lab in SCOPES if s) + ".", ""]
    if R.get("infeasible_ids"):
        L += [f"The main tables (sections 1–3) exclude the {len(R['infeasible_ids'])} groups whose shortest evidence route exceeds 40 steps "
              f"({', '.join(R['infeasible_ids'])}); their scores are in the appendix table at the end.", ""]
    if all(len(R["runs"][k]) == 1 for k in REPEATED if k in R["runs"]) and any(k in R["runs"] for k in REPEATED):
        L += ["**Single run so far: intervals do not include model sampling randomness.**", ""]
    n = 1
    for scope, lab in SCOPES:
        if scope not in R["boots"]:
            continue
        L += score_table(R, scope, f"## {n}. {lab}")
        g = gain_table(R, scope)
        if g:
            L += [f"**{lab}: Active differences**", ""] + g
        L += [f"**{lab}: total score by template**", ""] + template_table(R, scope)
        n += 1
    I = R.get("infeasible")
    if I is not None:
        L += [f"## Appendix. Groups not completable within the budget ({I['groups']} groups)", "",
              "Even the ideal route of these groups cannot be completed within 40 steps (shortest evidence route 41–50 steps); they are excluded "
              "from the main tables. Only ScanNet++ scores are listed (all these groups come from ScanNet++).", ""]
        for scope, lab in SCOPES:
            if scope is None or scope not in I["boots"]:
                continue
            L += score_table(I, scope, f"**{lab} (non-completable groups)**")
    L += ["- Answer rate: share of answerable questions given an answer (not cannot_determine); selective score: answered answerable "
          "questions only (multiple choice by correctness, T1 by MRA); correct-refusal rate: share of unanswerable questions answered "
          "cannot_determine. The Blind prompt requires an answer, so no refusal rate is computed for it.",
          "- Evidence hit uses the \"strictly visible\" definition (each dataset's own visibility rule) and is reported per dataset only."]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True, help="<setting>=<run>[,<run>…]; settings: see LABELS")
    ap.add_argument("--title", default="Evaluation scores")
    ap.add_argument("--root", help="root of the run directories (default paths.eval_runs)")
    ap.add_argument("--tiny", action="store_true", help="only llm.tiny_groups")
    args = ap.parse_args()
    spec = {}
    for x in args.set:
        k, v = x.split("=", 1)
        if k not in LABELS:
            ap.error(f"unknown setting {k}")
        spec[k] = v.split(",")
    ec = load_eval_config()
    root = Path(args.root or ec["paths"]["eval_runs"])
    Path(args.out).write_text("\n".join(render(build(root, spec, only_groups=ec["llm"]["tiny_groups"] if args.tiny else None), args.title)) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
