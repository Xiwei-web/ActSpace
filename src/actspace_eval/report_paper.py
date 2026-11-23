"""Paper tables: main table, per-dataset tables, appendix tables, a self-check, and a comparison with hand-entered paper numbers,
for the closed API models and the open models. Reads only run directories; calls no model.

    python -m actspace_eval.report_paper --out-dir reports/eval/paper_table [--paper <hand-entered numbers>.csv]

- Definition: merged_v1.0; the main tables exclude the 7 budget_infeasible groups (227 groups, 962 questions, 899 answerable).
  Three versions: merged, ARKitScenes, ScanNet++.
- Only sealed evaluation runs are used: open models <results root>/<model>/<setting>/full_r* / full_s*, closed models
  <results root>/api/<model>/<setting>/<prefix>_r* (prefixes in CLOSED).
- Values: one item per question and run, then averaged (runs pooled, same as the point estimate of the score tables);
  percentages with 1 decimal. Scores are taken from the runs' summary.json; the self-check re-scores every question from the
  answers in the trajectories and the frozen data and compares.
- Differences: ΔRW = Active − random_walk_40 and ΔO = oracle_route − Active are the differences of the two DISPLAYED values
  (each rounded to 0.1 first), so a reader can recompute them from the table; the self-check verifies this exactly. Whether a
  difference is significant is judged from intervals computed on unrounded data (report_eval / report_open_full score tables);
  the unrounded differences are kept in paper_main.csv as d_rw_unrounded / d_o_unrounded.
- Marking: within each model group (closed API, open), the best value of each column is bold and the second best underlined.
  Ranked by unrounded values (ΔRW and ΔO by their displayed values); lower is better for ΔO, higher for all others; Chance is
  not ranked.
- Tiny-subset models (TINY_ONLY: astra) were evaluated on the 32 tiny groups only (Blind, random_walk_40, Active v2). They appear in
  the proprietary group with a dagger and a footnote ("evaluated on the 32-group tiny subset only"), are never marked, and the
  self-check verifies their group coverage and flags separately; the CSV column `subset` is tiny32 for them.
- Also writes paper_main.tex: the three main tables as booktabs tabulars with the same rows, values and marks.
- Chance row: multiple choice 1 / number of options; T1 the expected MRA of a guess uniform in [min, max] of all answerable T1
  ground-truth values of the main table; unanswerable questions score 0 (same as paper_table1_api).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .answers import DEFAULT_THRESH, gold, score_question
from .data import FrozenData, dataset_of_scene, load_eval_config, results_root
from .report_eval import TEMPLATES, build, items
from .report_open_full import model_spec, selective, trunc_rows

ROOT = results_root()        # results root (results in configs/paths.yaml)
OPEN = ["VST-7B-RL", "InternVL3_5-8B", "SenseNova-SI-1.2-InternVL3-8B", "SenseNova-SI-1.3-InternVL3-8B", "Qwen3.5-4B", "Qwen3.5-9B",
        "Cambrian-S-7B", "Qwen3.6-27B", "Qwen3.8-27B", "Qwen3.6-35B-A3B", "gemma-4-31B-it",
        # models without sealed runs are left out of the tables
        "LLaVA-OneVision-7B", "RoboBrain2.0-7B", "SenseNova-SI-1.1-InternVL3-2B", "MiniCPM-V-4_5", "Qwen3.5-2B", "GLM-4.6V-Flash"]
# closed models with complete full-set runs and their run prefixes (same as paper_table1_api)
CLOSED = [("sol", "all"), ("terra", "all"), ("opus-5.5", "all_v1"), ("sonnet-5.5", "all_v1"), ("haiku-5.5", "all_v1"), ("luna", "all")]
# closed models evaluated on the 32-group tiny subset only (llm.tiny_groups; Blind, random_walk_40 and Active v2): shown in the main table
# with a dagger and a footnote, never marked, and checked separately in the self-check
TINY_ONLY = [("astra", "tiny_core")]
TINY_NOTE = ("evaluated on the 32-group tiny subset only (Blind, random_walk_40 and Active v2); not comparable with the 227-group rows "
             "and excluded from the bold/underline marking")
GROUPS = ("Proprietary API", "Open-source")
SCOPES = ((None, "Merged", "merged"), ("arkitscenes", "ARKitScenes", "arkit"), ("scannetpp", "ScanNet++", "scannetpp"))
MAIN_COLS = [("blind", "Blind"), ("passive40", "Passive-40"), ("rw", "random_walk_40"), ("oracle_route", "oracle_route"), ("active", "Active total"),
             ("answer_rate", "Answer rate"), ("selective", "Selective score"), ("refusal", "Correct refusal"), ("d_rw", "ΔRW"), ("d_o", "ΔO")] + [(t, t) for t in TEMPLATES]
APPX_COLS = [("oracle_min", "oracle_min"), ("passive8", "Passive-8"), ("passive20", "Passive-20")]
SETTING_OF = {"blind": "blind", "passive40": "passive_40", "rw": "random_walk_40", "oracle_route": "oracle_route", "active": "active_v2",
              "oracle_min": "oracle_min", "passive8": "passive_8", "passive20": "passive_20"}
LOWER_BETTER = {"d_o"}
SIGNED = {"d_rw", "d_o"}
DASH = "–"
N_GROUPS, N_Q, N_ANS = 227, 962, 899


# ---------------------------------------------------------------- data

def pool(rs, metric, scope, t=None):
    """Per-item mean with runs pooled (unrounded, 0–1); None when there are no items."""
    v = [x for r in rs for s, x in items(r, metric, t) if scope is None or dataset_of_scene(s) == scope]
    return float(np.mean(v)) if v else None


def model_entries(root: Path) -> list[dict]:
    out = [{"group": GROUPS[0], "model": m, "root": root / "api", "prefix": p, "subset": "full"} for m, p in CLOSED]
    out += [{"group": GROUPS[0], "model": m, "root": root / "api", "prefix": p, "subset": "tiny"} for m, p in TINY_ONLY]
    out += [{"group": GROUPS[1], "model": m, "root": root, "prefix": "full", "subset": "full"} for m in OPEN]
    return out


def condition(e, spec) -> str:
    """Condition column: run prefix; number of Active and RW runs; JSON mode; max_tokens (from env.json of the first Active run)."""
    first = (spec.get("active_v2") or next(iter(spec.values()), [None]))[0]
    env = json.loads((e["root"] / e["model"] / first / "env.json").read_text()) if first else {}
    fmt = "json_object" if env.get("json_mode") else "schema"
    return (f"{e['prefix']}; Active ×{len(spec.get('active_v2', []))}, RW ×{len(spec.get('random_walk_40', []))}; {fmt}; "
            f"{env.get('max_output_tokens', DASH)}")


def table_rows(entries) -> tuple[list[dict], dict]:
    """Main-table and appendix scores: one row per model × scope. Returns (rows, {model: build result})."""
    rows, built = [], {}
    for e in entries:
        spec = model_spec(e["root"], e["model"], e["prefix"])
        if not spec:
            continue
        R = build(e["root"] / e["model"], spec)
        built[e["model"]] = (e, spec, R)
        runs = R["runs"]
        for scope, lab, _ in SCOPES:
            tot = lambda k: pool(runs[k], "total", scope) if k in runs else None
            d = {"group": e["group"], "model": e["model"], "scope": lab, "condition": condition(e, spec), "subset": e.get("subset", "full")}
            for c, s in SETTING_OF.items():
                d[c] = tot(s)
            act = runs.get("active_v2", [])
            for m in ("answer_rate", "selective", "refusal"):
                d[m] = pool(act, m, scope) if act else None
            d["d_rw"], d["d_rw_unrounded"] = display_diff(d["active"], d["rw"])
            d["d_o"], d["d_o_unrounded"] = display_diff(d["oracle_route"], d["active"])
            for t in TEMPLATES:
                d[t] = pool(act, "total", scope, t) if act else None
            rows.append(d)
    return rows, built


def chance_rows() -> list[dict]:
    data = FrozenData(verify=False)
    bad = set(data.ecfg["env"].get("budget_infeasible") or [])
    Q = [(dataset_of_scene(g["scene_id"]), q) for gid, g in data.groups.items() if gid not in bad for q in g["questions"]]
    t1 = [gold(q) for _, q in Q if q["template"] == "T1" and q["evidence_label"] != "unanswerable"]
    a, b = min(t1), max(t1)

    def e_mra(y):
        return sum(max(0.0, min(y + y * (1 - t), b) - max(y - y * (1 - t), a)) / (b - a) for t in DEFAULT_THRESH) / len(DEFAULT_THRESH)

    rows = []
    for scope, lab, _ in SCOPES:
        per, allq, ans = defaultdict(list), [], []
        for ds, q in Q:
            if scope and ds != scope:
                continue
            un = q["evidence_label"] == "unanswerable"
            v = 0.0 if un else (e_mra(gold(q)) if q["template"] == "T1" else 1 / len(q["options"]))
            per[q["template"]].append(v)
            allq.append(v)
            if not un:
                ans.append(v)
        c_all, c_ans = float(np.mean(allq)), float(np.mean(ans))
        d = {"group": "Chance", "model": "Chance", "scope": lab, "condition": "", "blind": c_ans, "passive40": c_all, "rw": c_all,
             "oracle_route": c_all, "active": c_all, "oracle_min": c_all, "passive8": c_all, "passive20": c_all,
             "answer_rate": None, "selective": c_ans, "refusal": None, "d_rw": None, "d_o": None, **{t: float(np.mean(per[t])) for t in TEMPLATES}}
        rows.append(d)
    return rows, (a, b, len(t1))


def shown(v) -> float:
    """The value as displayed in the table (percent, rounded to 0.1)."""
    return float(f"{100 * v:.1f}")


def display_diff(a, b):
    """(a − b of the displayed values, as a fraction; a − b unrounded). None when either side is missing."""
    if a is None or b is None:
        return None, None
    return round(shown(a) - shown(b), 1) / 100, a - b


def mark(rows, cols):
    """Within each scope and model group, per column: best -> best, second best -> second (unrounded values; lower ΔO is better)."""
    for r in rows:
        r.setdefault("best", set())
        r.setdefault("second", set())
    for _, lab, _ in SCOPES:
        for g in GROUPS:
            mr = [r for r in rows if r["scope"] == lab and r["group"] == g and r.get("subset", "full") == "full"]
            for c in cols:
                key = (lambda r: r[c]) if c in LOWER_BETTER else (lambda r: -r[c])
                vals = sorted({round(key(r), 10) for r in mr if r[c] is not None})
                for r in mr:
                    if r[c] is None:
                        continue
                    k = round(key(r), 10)
                    if k == vals[0]:
                        r["best"].add(c)
                    elif len(vals) > 1 and k == vals[1]:
                        r["second"].add(c)


# ---------------------------------------------------------------- formatting

def num(v, c=None) -> str:
    if v is None:
        return DASH
    return f"{100 * v:+.1f}".replace("-", "−") if c in SIGNED else f"{100 * v:.1f}"


def cell(r, c) -> str:
    s = num(r[c], c)
    if c in r.get("best", ()):
        return f"**{s}**"
    if c in r.get("second", ()):
        return f"<u>{s}</u>"
    return s


def label(r) -> str:
    return f"{r['model']} †" if r.get("subset") == "tiny" else r["model"]


def md_table(rows, cols, lab, with_cond=True) -> list[str]:
    head = "| Group | Model | " + ("Condition | " if with_cond else "") + " | ".join(n for _, n in cols) + " |"
    L = [head, "|---|---|" + ("---|" if with_cond else "") + "---|" * len(cols)]
    for r in [r for r in rows if r["scope"] == lab]:
        L.append(f"| {r['group'] if r['group'] != 'Chance' else ''} | {label(r)} | " + (f"{r['condition']} | " if with_cond else "")
                 + " | ".join(cell(r, c) if r["group"] != "Chance" else num(r[c], c) if r[c] is not None else "" for c, _ in cols) + " |")
    return L


TEX_HEAD = {"blind": "Blind", "passive40": "Passive-40", "rw": "RW-40", "oracle_route": "Oracle route", "active": "Active",
            "answer_rate": "Ans.\\ rate", "selective": "Sel.\\ score", "refusal": "Corr.\\ abst.", "d_rw": "$\\Delta_{\\mathrm{RW}}$",
            "d_o": "$\\Delta_{\\mathrm{O}}$", **{t: t for t in TEMPLATES}}


def tex_cell(r, c) -> str:
    if r[c] is None:
        return "--"
    v = num(r[c], c).replace("−", "$-$").replace("+", "$+$")
    if r["group"] != "Chance" and c in r.get("best", ()):
        return f"\\textbf{{{v}}}"
    if r["group"] != "Chance" and c in r.get("second", ()):
        return f"\\underline{{{v}}}"
    return v


def latex_tables(rows) -> str:
    """The three main tables (merged, ARKitScenes, ScanNet++) as booktabs tabulars with the same rows, values and marks as
    paper_main.md; tiny-subset models carry a dagger and the footnote."""
    cols = [c for c, _ in MAIN_COLS]
    esc = lambda x: x.replace("_", "\\_")                                           # noqa: E731
    out = ["% Generated by `python -m actspace_eval.report_paper`; requires \\usepackage{booktabs}.",
           "% Percentages with one decimal; Delta = difference of the two displayed values; bold = best, underline = second best",
           "% within each model group (unrounded values); Chance and tiny-subset models are not marked.", ""]
    tiny = any(r.get("subset") == "tiny" for r in rows)
    for scope, lab, key in SCOPES:
        out += [f"% ---- Table: {lab}", f"\\begin{{tabular}}{{l{'r' * len(cols)}}}", "\\toprule",
                "Model & " + " & ".join(TEX_HEAD[c] for c in cols) + " \\\\", "\\midrule"]
        for gi, g in enumerate(GROUPS + ("Chance",)):
            rs = [r for r in rows if r["scope"] == lab and r["group"] == g]
            if not rs:
                continue
            if gi:
                out.append("\\midrule")
            if g != "Chance":
                out.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{g}}}}} \\\\")
            for r in rs:
                name = esc(r["model"]) + ("$^\\dagger$" if r.get("subset") == "tiny" else "")
                out.append(name + " & " + " & ".join(tex_cell(r, c) for c in cols) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        if tiny:
            out.append(f"\\par\\footnotesize $^\\dagger$ {esc(TINY_NOTE[0].upper() + TINY_NOTE[1:])}.")
        out.append("")
    return "\n".join(out) + "\n"


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        raw = ("d_rw_unrounded", "d_o_unrounded")
        w.writerow(["scope", "group", "model", "subset", "condition"] + [c for c, _ in cols] + ["best", "second"] + list(raw))
        for r in rows:
            w.writerow([r["scope"], r["group"], r["model"], {"tiny": "tiny32"}.get(r.get("subset"), "full" if r["group"] != "Chance" else ""), r["condition"]] + [num(r[c], c).replace("−", "-").lstrip("+") if r[c] is not None else ""
                                                                                for c, _ in cols]
                       + ["+".join(c for c, _ in cols if c in r.get("best", ())), "+".join(c for c, _ in cols if c in r.get("second", ()))]
                       + [f"{100 * r[k]:.4f}" if r.get(k) is not None else "" for k in raw])


# ---------------------------------------------------------------- appendix: truncation, without truncated groups, the 7 non-completable groups

def appendix_extra(built) -> dict:
    trunc, sel, inf = [], [], []
    for m, (e, spec, R) in built.items():
        tr = trunc_rows(e["root"], m, spec)
        by = {r["setting"]: r for r in tr}
        trunc.append({"group": e["group"], "model": m, "by": by})
        s = selective(R, tr)
        sel.append({"group": e["group"], "model": m, "sel": s})
        I = R.get("infeasible")
        row = {"group": e["group"], "model": m}
        for c, st in SETTING_OF.items():
            row[c] = pool(I["runs"][st], "total", None) if I and st in I["runs"] else None
        row["n_groups"] = I["groups"] if I else 0
        inf.append(row)
    return {"trunc": trunc, "sel": sel, "inf": inf}


# ---------------------------------------------------------------- self-check

def self_check(rows, built, chance_range) -> tuple[list[str], int]:
    L, fails = [], 0

    def check(ok, text):
        nonlocal fails
        fails += not ok
        L.append(f"- {'PASS' if ok else '**FAIL**'}: {text}")

    data = FrozenData(verify=False)
    qby = {q["qid"]: q for g in data.groups.values() for q in g["questions"]}
    thr = tuple(data.ecfg["scoring"]["mra_thresholds"])
    tiny = list(data.ecfg["llm"]["tiny_groups"])
    t_q = sum(len(data.groups[g]["questions"]) for g in tiny)
    t_ans = sum(q["evidence_label"] != "unanswerable" for g in tiny for q in data.groups[g]["questions"])
    bad_groups, bad_n, bad_tiny, rescored, mism, boot_diff = [], [], [], 0, [], 0.0
    for m, (e, spec, R) in built.items():
        for s, rs in R["runs"].items():
            for name, r in zip(spec[s], rs):
                gs = {x["group_id"] for x in r["per_question"]}
                n = len(r["per_question"])
                if e.get("subset") == "tiny":
                    if gs != set(tiny) or n != (t_ans if s == "blind" else t_q):
                        bad_tiny.append(f"{m} {name} {len(gs)} groups, {n} questions")
                    continue
                if len(gs) != N_GROUPS:
                    bad_groups.append(f"{m} {name} {len(gs)}")
                if n != (N_ANS if s == "blind" else N_Q):
                    bad_n.append(f"{m} {name} {n}")
                for x in r["per_question"]:
                    rescored += 1
                    v = score_question(qby[x["qid"]], r["answers_by_qid"].get(x["qid"]), thr)
                    if abs(v - x["score"]) > 1e-9:
                        mism.append(f"{m} {name} {x['qid']} summary {x['score']} recomputed {v}")
            for scope, _, _ in SCOPES:
                if scope not in R["boots"]:
                    continue
                c = R["boots"][scope].mean_ci([items(r, "total") for r in rs], f"{s}|total")
                p = pool(rs, "total", scope)
                if c is not None and p is not None:
                    boot_diff = max(boot_diff, abs(c["mean"] - p))
    check(not bad_groups, f"every run has {N_GROUPS} main-table groups" + (f" (mismatches: {'; '.join(bad_groups[:10])})" if bad_groups else ""))
    check(not bad_n, f"questions per run: Blind {N_ANS} (answerable only), all others {N_Q}" + (f" (mismatches: {'; '.join(bad_n[:10])})" if bad_n else ""))
    tiny_models = sorted({m for m, (e, _, _) in built.items() if e.get("subset") == "tiny"})
    tiny_rows = [r for r in rows if r.get("subset") == "tiny"]
    flag_bad = [f"{r['scope']} {r['model']}" for r in tiny_rows if r.get("best") or r.get("second") or label(r) == r["model"]]
    flag_bad += [f"{r['scope']} {r['model']} not flagged" for r in rows if r["model"] in tiny_models and r.get("subset") != "tiny"]
    check(not bad_tiny and not flag_bad,
          f"tiny-subset models ({', '.join(tiny_models) or 'none'}): every run covers exactly the {len(tiny)} tiny groups ({t_q} questions, Blind {t_ans}); "
          "their rows are flagged with † and the footnote, and are never marked" + (f" (problems: {'; '.join(bad_tiny + flag_bad)})" if bad_tiny or flag_bad else ""))
    check(not mism, f"re-scored from raw results: {rescored:,} items (answers in the trajectories + ground truth from the frozen data, answers.score_question); "
          "per-question scores in summary.json: "
          + ("all identical" if not mism else f"{len(mism)} items differ: {'; '.join(mism[:10])}"))
    check(boot_diff < 1e-9, f"per-item means in the tables equal the score tables (report_eval, scene-bootstrap point estimate); max difference {boot_diff:.2e}")
    # merged = both datasets weighted by number of questions
    worst = 0.0
    for m, (e, spec, R) in built.items():
        for s, rs in R["runs"].items():
            parts = [(pool(rs, "total", ds), sum(1 for r in rs for sc, _ in items(r, "total") if dataset_of_scene(sc) == ds)) for ds in ("arkitscenes", "scannetpp")]
            w = sum(n for _, n in parts)
            if w:
                worst = max(worst, abs(sum((v or 0) * n for v, n in parts) / w - pool(rs, "total", None)))
    check(worst < 1e-9, f"merged score equals ARKitScenes and ScanNet++ weighted by number of questions; max difference {worst:.2e}")
    # differences and rounding
    off = [f"{r['scope']} {r['model']} {c}: {num(r[c], c)} vs {num(r[a])} − {num(r[b])}" for r in rows if r["group"] != "Chance"
           for c, a, b in (("d_rw", "active", "rw"), ("d_o", "oracle_route", "active"))
           if r[c] is not None and abs(float(num(r[c], c).replace("−", "-")) - (float(num(r[a])) - float(num(r[b])))) > 1e-9]
    check(not off, "ΔRW and ΔO equal the difference of the two displayed values exactly" + (f" (violations: {'; '.join(off)})" if off else ""))
    # marking
    dup = []
    for _, lab, _ in SCOPES:
        for g in GROUPS:
            mr = [r for r in rows if r["scope"] == lab and r["group"] == g and r.get("subset", "full") == "full"]
            for c, _ in MAIN_COLS + APPX_COLS:
                vals = [r[c] for r in mr if r[c] is not None]
                if len(vals) >= 2 and not any(c in r["best"] for r in mr):
                    dup.append(f"{lab} {g} {c} has no best value")
                best = [r for r in mr if c in r["best"]]
                if best and len({round(r[c], 10) for r in best}) != 1:
                    dup.append(f"{lab} {g} {c} best value not unique")
    check(not dup, "every scope, group and column has exactly one best value (ties are all bold)" + (f" (problems: {'; '.join(dup)})" if dup else ""))
    a, b, n = chance_range
    L.append(f"- Note: the Chance T1 range spans the minimum to maximum of the {n} answerable T1 ground-truth values of the main table, [{a:.2f}, {b:.2f}] m.")
    return L, fails


# ---------------------------------------------------------------- comparison with hand-entered paper numbers / earlier tables

COL_ALIASES = {"passive40": "passive40", "rw": "rw", "oracle_route": "oracle_route", "active": "active", "blind": "blind"}


def compare(rows, ref_rows, source) -> list[tuple]:
    """ref_rows: [(scope, model, column, value string)]. Returns mismatches: (source, scope, model, column, paper, this run).
    Missing rows and columns are listed too."""
    idx = {(r["scope"], r["model"]): r for r in rows}
    out, n = [], 0
    for scope, model, col, val in ref_rows:
        val = (val or "").strip().replace("−", "-").lstrip("+").replace("**", "").replace("<u>", "").replace("</u>", "")
        if val in ("", DASH, "—"):
            continue
        n += 1
        r = idx.get((scope, model))
        if r is None or col not in r:
            out.append((source, scope, model, col, val, "(no such cell in this run)"))
            continue
        mine = num(r[col], col).replace("−", "-").lstrip("+")
        if abs(float(val) - float(mine)) > 1e-9:
            out.append((source, scope, model, col, val, mine))
    return out, n


def ref_api_csv(path: Path) -> list[tuple]:
    out = []
    for r in csv.DictReader(open(path)):
        if r["model"] == "Chance":
            continue
        for c, _ in MAIN_COLS:
            if c in r:
                out.append((r["scope"], r["model"], c, r[c]))
    return out


def ref_by_task_csv(path: Path) -> list[tuple]:
    """Earlier export of the open-model T1–T5 (active_by_task.csv, merged): the Active v2 rows correspond to T1–T5 of the main table."""
    out = []
    for r in csv.DictReader(open(path)):
        if r["setting"] == "Active v2":
            out += [("Merged", r["model"], t, r[t]) for t in TEMPLATES]
    return out


def ref_long_csv(path: Path) -> list[tuple]:
    """Hand-entered paper numbers: CSV with columns scope, model, column, value (column uses this table's CSV column names,
    e.g. active, d_rw, T1)."""
    return [(r["scope"], r["model"], r["column"], r["value"]) for r in csv.DictReader(open(path))]


# ---------------------------------------------------------------- output

def write_all(out: Path, rows, chance, built, extra, checks, cmp_rows, cmp_sources, notes):
    out.mkdir(parents=True, exist_ok=True)
    allrows = [r for r in rows] + chance
    write_csv(out / "paper_main.csv", allrows, MAIN_COLS + APPX_COLS)
    head = ["- Definition: merged_v1.0; the main tables exclude the 7 budget_infeasible groups: 227 groups, 962 questions (899 answerable). Percentages with 1 decimal.",
            "- With several runs, each question and run is one item and the items are averaged (same as the point estimate of the score tables). Blind counts answerable questions only; T1 and T4 include unanswerable questions.",
            "- ΔRW = Active − random_walk_40, ΔO = oracle_route − Active: the difference of the two displayed (rounded) values. Significance is judged from intervals on unrounded data (score tables); unrounded differences are in `paper_main.csv`.",
            "- **Marking:** within each model group (proprietary API, open-source), the best value of each column is **bold** and the second best <u>underlined</u>; by unrounded values (ΔRW and ΔO by their displayed values), lower is better for ΔO and higher for all others; Chance is not ranked.",
            "- Condition column: run prefix; number of Active and RW runs; JSON mode (json_object / schema); max_tokens. Proprietary models use each provider's high reasoning effort, open models their default thinking setting.",
            "- The same values are in `paper_main.csv` (with best and second columns). Self-check in `self_check.md`. LaTeX version: `paper_main.tex`."]
    if any(r.get("subset") == "tiny" for r in rows):
        head.append(f"- † {TINY_NOTE}.")
    L = ["# Paper main table", ""] + head + [""]
    if notes:
        L += ["## Table notes", ""] + notes + [""]
    for i, (scope, lab, _) in enumerate(SCOPES, 1):
        L += [f"## Table {i}: {lab}", ""] + md_table(allrows, MAIN_COLS, lab) + [""]
    (out / "paper_main.md").write_text("\n".join(L) + "\n")
    (out / "paper_main.tex").write_text(latex_tables(allrows))
    for scope, lab, key in SCOPES[1:]:
        (out / f"paper_main_{key}.md").write_text("\n".join([f"# Paper main table: {lab}", ""] + head + [""] + md_table(allrows, MAIN_COLS, lab)) + "\n")
    # appendix
    A = ["# Paper appendix tables", "", "Definition and marking as in the main table (compared within each model group).", ""]
    A += ["## A1. oracle_min, Passive-8, Passive-20", ""]
    for scope, lab, _ in SCOPES:
        A += [f"### {lab}", ""] + md_table(allrows, APPX_COLS, lab, with_cond=False) + [""]
    labels = {"blind": "Blind", "passive_8": "Passive-8", "passive_20": "Passive-20", "passive_40": "Passive-40", "random_walk_40": "random_walk_40",
              "oracle_min": "oracle_min", "oracle_route": "oracle_route", "active_v2": "Active v2"}
    sts = list(labels)
    A += ["## A2. Truncation rates", "", "Each cell: truncated / calls = truncation rate (of which repetition loops). Truncated = hit max_tokens (length for vLLM and OpenAI, max_tokens for Anthropic) or timed out. Calls cover all 234 groups (including the 7 budget_infeasible groups).", "",
          "| Group | Model | " + " | ".join(labels[s] for s in sts) + " |", "|---|---|" + "---|" * len(sts)]
    for t in extra["trunc"]:
        f = lambda s: DASH if s not in t["by"] or not t["by"][s]["calls"] else \
            f"{t['by'][s]['truncated']:,}/{t['by'][s]['calls']:,} = {100 * t['by'][s]['truncated'] / t['by'][s]['calls']:.1f}% ({t['by'][s]['loop']:,})"
        A.append(f"| {t['group']} | {t['model']} | " + " | ".join(f(s) for s in sts) + " |")
    A += ["", "## A3. Scores without truncated groups (analysis only, does not replace the main table)", "",
          "Truncated group = the group's last call was truncated. Each cell: per-question mean without truncated groups (mean over all questions; number of truncated groups). Cells without truncated groups show a single number. Merged, the 227 main-table groups.", "",
          "| Group | Model | " + " | ".join(labels[s] for s in sts) + " |", "|---|---|" + "---|" * len(sts)]
    for t in extra["sel"]:
        def f(s):
            if s not in t["sel"]:
                return DASH
            g, c, a, k = t["sel"][s]
            return f"{100 * a:.1f}" if not c else f"{100 * k:.1f} ({100 * a:.1f}; {c})"
        A.append(f"| {t['group']} | {t['model']} | " + " | ".join(f(s) for s in sts) + " |")
    inf_cols = [("blind", "Blind"), ("passive40", "Passive-40"), ("rw", "random_walk_40"), ("oracle_route", "oracle_route"), ("active", "Active total")]
    for r in extra["inf"]:
        r["scope"] = "Merged"
    mark_inf = [dict(r) for r in extra["inf"]]
    mark(mark_inf, [c for c, _ in inf_cols])
    A += ["", "## A4. Scores on the 7 non-completable groups (budget_infeasible: even the ideal route cannot be completed within 40 steps; all ScanNet++)", "",
          "These 7 groups are not in the main table. Each cell is the mean over all questions of these 7 groups; marking as in the main table.", "",
          "| Group | Model | " + " | ".join(n for _, n in inf_cols) + " |", "|---|---|" + "---|" * len(inf_cols)]
    for r in mark_inf:
        A.append(f"| {r['group']} | {r['model']} | " + " | ".join(cell(r, c) for c, _ in inf_cols) + " |")
    (out / "paper_appendix.md").write_text("\n".join(A) + "\n")
    # self-check
    (out / "self_check.md").write_text("\n".join(["# Paper table self-check", ""] + checks[0] + [""]) + "\n")
    # comparison
    C = ["# Comparison with hand-entered paper numbers", ""]
    for src, n, note in cmp_sources:
        C.append(f"- {src}: {n:,} cells compared. {note}")
    C += ["", f"{len(cmp_rows)} cells differ." if cmp_rows else "", ""]
    if cmp_rows:
        C += ["| Source | Scope | Model | Column | Paper / earlier table | This run |", "|---|---|---|---|---|---|"]
        C += [f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in cmp_rows]
    (out / "check_against_paper.md").write_text("\n".join(C) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-dir", default="reports/eval/paper_table")
    ap.add_argument("--paper", help="hand-entered paper numbers (CSV: scope, model, column, value)")
    ap.add_argument("--api-csv", default=str(ROOT / "api" / "_reports" / "paper_table1_api.csv"), help="earlier closed-model table (paper_table1_api.csv)")
    ap.add_argument("--by-task-csv", default="reports/eval/paper_table/active_by_task.csv", help="earlier export of the open-model T1–T5")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out_dir)
    ec = load_eval_config()
    entries = model_entries(root)
    rows, built = table_rows(entries)
    chance, rng = chance_rows()
    mark(rows, [c for c, _ in MAIN_COLS + APPX_COLS])
    extra = appendix_extra(built)
    checks = self_check(rows, built, rng)
    notes = [f"- {m}: {ec['models'][m]['report_note']}" for m in OPEN if m in built and ec["models"].get(m, {}).get("report_note")]
    # note which models use a format-adapted chat template (listed from the vllm.chat_template config)
    adapted = [m for m in OPEN if m in built and (ec["models"].get(m, {}).get("vllm") or {}).get("chat_template")]
    if adapted:
        notes.insert(0, f"- Chat templates: {', '.join(adapted)} use format-adapted chat templates (`configs/chat_templates/`). Only the transport format changes: "
                        "vLLM passes image/text content as a list, which the official templates either reject or render with all images first; all models "
                        "receive the same content in the same image/text interleaving order (checked per episode with `actspace_eval.content_order`).")
    sel = {t["model"]: t["sel"].get("active_v2") for t in extra["sel"]}
    if sel.get("gemma-4-31B-it"):
        notes.append(f"- gemma-4-31B-it Active score without truncated groups: {100 * sel['gemma-4-31B-it'][3]:.1f} ({sel['gemma-4-31B-it'][1]} truncated groups; appendix A3, analysis only, does not replace the main table).")
    cmp_rows, sources = [], []
    if a.paper and Path(a.paper).exists():
        x, n = compare(rows + chance, ref_long_csv(Path(a.paper)), "paper (hand-entered)")
        cmp_rows += x
        sources.append(("Paper, hand-entered (" + a.paper + ")", n, ""))
    else:
        sources.append(("Paper, hand-entered", 0, "**No file with hand-entered numbers found** (pass it with `--paper`, CSV columns scope, model, column, value); comparing with the earlier exported tables below."))
    if Path(a.api_csv).exists():
        x, n = compare(rows, ref_api_csv(Path(a.api_csv)), "earlier closed-model table paper_table1_api.csv")
        cmp_rows += x
        sources.append(("Earlier closed-model table `paper_table1_api.csv`", n, ""))
    if Path(a.by_task_csv).exists():
        x, n = compare(rows, ref_by_task_csv(Path(a.by_task_csv)), "earlier open-model table active_by_task.csv")
        cmp_rows += x
        sources.append(("Earlier open-model table `active_by_task.csv` (Active v2 T1–T5, merged)", n, ""))
    write_all(out, rows, chance, built, extra, checks, cmp_rows, sources, notes)
    print(f"wrote {out}: paper_main.md (+ _arkit, _scannetpp), paper_main.csv, paper_appendix.md, self_check.md, check_against_paper.md; "
          f"self-check failures: {checks[1]}; comparison mismatches: {len(cmp_rows)}")


if __name__ == "__main__":
    main()
