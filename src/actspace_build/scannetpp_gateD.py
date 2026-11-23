"""ScanNet++ GATE D human-verification package (Step 9). Reuses the ARKit two-phase single-page app (gate_d_app), the annotation
guide template (gate_d.GUIDE) and the collection/summary code (gate_d.load_records / summarize) instead of a separate implementation.

  sample   stratify scannetpp_only_v0.1 by "constraint type × contains unanswerable × difficulty" (gate_d.stratum) and sample
           ceil(15%) per stratum with a new seed; prefer scenes not yet covered within a stratum; at least MIN_MASKED masked
           groups (all of them if fewer exist)
  pages    reports/gateD/scannetpp/pages/index.html (phase 1: blind answering on raw images; phase 2: boxes, names, top-down
           view, standing point and heading, T4 start, masking hints)
  package  reports/gateD/scannetpp_gateD_pages_v0.1.zip (includes the annotation guide plus ScanNet++-specific hints)
  summary  collect: python -m actspace_build.scannetpp_gateD summary <exported jsonl / per-group json ...>
           → reports/gateD/scannetpp/gateD_report_draft.md (two-phase verdicts, human performance, flag rates per template,
           questions to follow up)
  baseline recompute the human baseline excluding questions that involve blacklisted objects
  selftest simulate an "all correct" annotation from the ground truth and run summary end to end (self-test on the cluster)

Usage: python -m actspace_build.scannetpp_gateD sample|pages|package|selftest|summary|baseline [files ...]
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict

import numpy as np

from . import gate_d, gate_d_app
from .common import load_config, path_of, write_json, read_json
from .scannetpp import sp_work

DATASET = "scannetpp"
VERSION = "scannetpp_only_v0.1"
VERSION_GATED = VERSION  # version verified at GATE D (v0.1); the baseline recomputation also uses the v0.1 questions and annotations
SEED_TAG = 91  # new seed: [seed, 9, 91]
MIN_MASKED = 6

SP_HINTS = """<h2>Appendix: ScanNet++-specific hints</h2>
<ul><li><strong>Magenta patches</strong>: some images contain solid magenta patches. They are text, paper or faces masked by the dataset for anonymisation, not objects. Treat them as "that area cannot be seen" when answering.</li>
<li><strong>Built-in appliances, labels that do not match the appearance</strong>: if the object named in a question looks like something else in the image (e.g. a "refrigerator" or "dishwasher" that is really a cabinet-style door, a "bookshelf" that is really a low cupboard for dishes),
tick "object reference or annotation is wrong" (label wrong) and describe in the note what you see.</li>
<li>Objects in the questions are always named by their coarse category (e.g. "the shelf" may be a bookshelf, a storage rack or a wall shelf; "the chair" may be an office chair or a stool); only monitor and copier use fine-grained names.</li>
<li>Images are original iPhone captures (1920×1440); some have slight motion blur. Click to enlarge.</li></ul>
"""


def load_groups(cfg):
    return [json.loads(l) for l in open(sp_work(cfg) / "_step8" / VERSION / "test_groups.jsonl")]


def out_dir(cfg):
    d = path_of(cfg, "reports") / "gateD" / "scannetpp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_masked(grp):
    return bool(grp.get("view_graph_mask"))


def sample(cfg):
    groups = load_groups(cfg)
    hc = cfg["human_check"]
    rng = np.random.default_rng([cfg["seed"], 9, SEED_TAG])
    by = defaultdict(list)
    for grp in groups:
        by[gate_d.stratum(grp)].append(grp)
    keys = sorted(by, key=str)
    covered, picked, rows = set(), [], []
    for k in keys:
        n = math.ceil(hc["sample_frac"] * len(by[k]))
        pool = [by[k][i] for i in rng.permutation(len(by[k]))]
        sel = []
        while len(sel) < n:  # within a stratum, prefer scenes not yet covered
            rest = [g_ for g_ in pool if g_ not in sel]
            new = [g_ for g_ in rest if g_["scene_id"] not in covered]
            g_ = (new or rest)[0]
            sel.append(g_)
            covered.add(g_["scene_id"])
        picked += sel
        rows.append((k, len(by[k]), n))
    n_mask = sum(is_masked(g_) for g_ in picked)
    extra = []
    if n_mask < MIN_MASKED:
        cand = [g_ for g_ in groups if is_masked(g_) and g_ not in picked]
        cand = [cand[i] for i in rng.permutation(len(cand))]
        cand.sort(key=lambda g_: g_["scene_id"] in covered)
        extra = cand[:MIN_MASKED - n_mask]
        picked += extra
    ids = sorted(g_["group_id"] for g_ in picked)
    write_json(out_dir(cfg) / "sample_list.json",
               {"version": VERSION, "seed": [cfg["seed"], 9, SEED_TAG], "pool": len(groups), "groups": ids,
                "strata": [{"stratum": list(map(str, k)), "n": n_, "sampled": m} for k, n_, m in rows],
                "extra_masked_groups": sorted(g_["group_id"] for g_ in extra),
                "scenes": len({g_["scene_id"] for g_ in picked}), "scenes_in_pool": len({g_["scene_id"] for g_ in groups}),
                "masked_groups": sum(is_masked(g_) for g_ in picked), "questions": sum(len(g_["questions"]) for g_ in picked)})
    print(f"sampled {len(ids)} / {len(groups)} groups ({len(ids) / len(groups):.0%}), {len({g_['scene_id'] for g_ in picked})} scenes, "
          f"{sum(is_masked(g_) for g_ in picked)} masked groups, {sum(len(g_['questions']) for g_ in picked)} questions")


def pages(cfg):
    groups = {g_["group_id"]: g_ for g_ in load_groups(cfg)}
    sl = read_json(out_dir(cfg) / "sample_list.json")
    d = out_dir(cfg) / "pages"
    gate_d_app.build(VERSION, [(cfg, [groups[gid] for gid in sl["groups"]], None)], d, dataset=DATASET, draw_structure=True)
    print(f"wrote {len(sl['groups'])} groups to {d / 'index.html'}")


def package(cfg):
    sl = read_json(out_dir(cfg) / "sample_list.json")
    guide = (gate_d.GUIDE.replace("CSS_HERE", gate_d.CSS).replace("VERSION_HERE", VERSION).replace("N_HERE", str(len(sl["groups"])))
             .replace("PARTS_HERE", SP_HINTS).replace("{{", "{").replace("}}", "}")
             .replace("The images come from the ARKitScenes dataset", "The images come from the ScanNet++ dataset"))
    assert "ARKitScenes" not in guide
    (out_dir(cfg) / "annotation_guide.html").write_text(guide)
    out = path_of(cfg, "reports") / "gateD" / "scannetpp_gateD_pages_v0.1.zip"
    root = "gateD_pages_scannetpp_v0.1"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir(cfg) / "annotation_guide.html", f"{root}/annotation_guide.html")
        for f in sorted((out_dir(cfg) / "pages").rglob("*")):
            if f.is_file():
                z.write(f, f"{root}/pages/{f.relative_to(out_dir(cfg) / 'pages')}")
    print(f"packed {out} ({out.stat().st_size / 1e6:.1f} MB)")


def summary(cfg, files, out_name="gateD_report_draft.md"):
    hc = cfg["human_check"]
    records = gate_d.load_records(files)
    groups = {g_["group_id"]: g_ for g_ in load_groups(cfg)}
    sl = read_json(out_dir(cfg) / "sample_list.json")
    m = gate_d.summarize(groups, records, hc)
    per = Counter(r["group_id"] for r in records)
    L = ["# ScanNet++ GATE D human verification report (draft)", "",
         f"> Generated automatically (`python -m actspace_build.scannetpp_gateD summary`, reusing `gate_d.summarize`); pending review. Data version {VERSION}; two-phase verdict criteria as for ARKit.", "",
         f"- {len(records)} annotation records; {len(sl['groups'])} groups sampled, {sum(per[g_] > 0 for g_ in sl['groups'])} annotated; {len(m['incomplete_records'])} records with incomplete review.", "",
         "| Criterion | Actual | Required | Result |", "|---|---|---|---|",
         f"| Share reviewed as \"correct\" without label wrong | {m['review_correct_frac']:.1%} ({m['n_judged']} rows) | ≥ {hc['min_review_correct_frac']:.0%} | {'pass' if m['pass_correct'] else '**fail**'} |",
         f"| Share flagged ambiguous or \"unannotated object of the same category\" | {m['ambiguous_or_unannotated_frac']:.1%} ({m['n_answerable_rows']} rows) | ≤ {hc['max_ambiguous_frac']:.0%} | {'pass' if m['pass_ambiguous'] else '**fail**'} |", "",
         f"**Conclusion (draft): {'GATE D passed' if m['pass'] else 'GATE D failed'}**.", "",
         "## Human performance baseline (blind phase)", "", "| Template | Question category | Metric | Human score | Rows |", "|---|---|---|---|---|",
         *[f"| {t} | {k} | {metric} | {v:.3f} | {n} |" for (t, k), (v, n, metric) in sorted(m["perf"].items())], "",
         "## Flag rates per template", "", "| Template | Rows | ambiguous | label wrong | unannotated same-category object |", "|---|---|---|---|---|",
         *[f"| {t} | {f['n']} | {f['ambiguous']:.1%} | {f['label_wrong']:.1%} | {f['unannotated']:.1%} |" for t, f in m["flags"].items()], "",
         "## Questions to follow up", "", "| Question | Template | Review | Flags | Blind answer / ground truth | Notes |", "|---|---|---|---|---|---|"]
    for x in m["rows"]:
        fl = [k for k in ("ambiguous", "label_wrong", "unannotated") if x[k]]
        if x["verdict"] in ("incorrect", "unsure") or fl:
            note = "; ".join(n for n in (x["note"], x["review_note"]) if n).replace("|", "/")
            L.append(f"| {x['qid']} | {x['template']} | {x['verdict'] or '—'} | {', '.join(fl) or '—'} | {x['answer']} / {x['gt']} | {note} |")
    (out_dir(cfg) / out_name).write_text("\n".join(L) + "\n")
    print("wrote", out_dir(cfg) / out_name, {k: m[k] for k in ("n_records", "n_judged", "review_correct_frac", "pass")})
    return m


def baseline_excluding_blacklist(cfg, files, out_name="baseline_v0.2.md"):
    """Recompute the GATE D human baseline excluding questions that involve objects on the ScanNet++ blacklist (all current entries of
    configs/object_blacklist_scannetpp.yaml) as slot, option or evidence objects."""
    from .scannetpp_step3 import load_blacklist
    records = gate_d.load_records(files)
    groups = {g_["group_id"]: g_ for g_ in load_groups(cfg)}
    hc = cfg["human_check"]
    kept, dropped = {}, []
    for gid, g_ in groups.items():
        black = set(load_blacklist(cfg, g_["scene_id"].split("_", 1)[1]))
        qs = []
        for q in g_["questions"]:
            objs = set(q["slots"].values()) | set(q.get("option_objs") or []) | set(q["evidence_objs"])
            (dropped.append(q["qid"]) if objs & black else qs.append(q))
        kept[gid] = {**g_, "questions": qs}
    full, sub = gate_d.summarize(groups, records, hc), gate_d.summarize(kept, records, hc)
    n_all = sum(len(groups[r["group_id"]]["questions"]) for r in records)
    keys = sorted(set(full["perf"]) | set(sub["perf"]))
    L = ["# ScanNet++ GATE D human baseline (excluding questions on blacklisted objects)", "",
         f"> Generated by `python -m actspace_build.scannetpp_gateD baseline <annotation files>`. Annotations are from GATE D ({VERSION_GATED}, ann01). "
         f"Excluded questions involving current ScanNet++ blacklisted objects (as slot, option or evidence objects): {len([q for q in dropped if q.rsplit('_', 1)[0] in {r['group_id'] for r in records}])} / {n_all}.", "",
         "| Template | Question category | Metric | All questions | Rows | After exclusion | Rows |", "|---|---|---|---|---|---|---|",
         *[f"| {t} | {k} | {(full['perf'].get((t, k)) or sub['perf'].get((t, k)))[2]} | "
           + (f"{full['perf'][(t, k)][0]:.3f} | {full['perf'][(t, k)][1]}" if (t, k) in full["perf"] else "— | 0") + " | "
           + (f"{sub['perf'][(t, k)][0]:.3f} | {sub['perf'][(t, k)][1]}" if (t, k) in sub["perf"] else "— | 0") + " |" for t, k in keys], "",
         f"After exclusion, answerable questions reviewed as \"correct without label wrong\": {sub['review_correct_frac']:.1%} ({sub['n_judged']} rows; all questions: {full['review_correct_frac']:.1%}, {full['n_judged']} rows)."]
    (out_dir(cfg) / out_name).write_text("\n".join(L) + "\n")
    print("wrote", out_dir(cfg) / out_name, len(dropped), sub["review_correct_frac"])
    return full, sub, dropped


def selftest(cfg):
    """Simulate an annotator export that answers everything correctly and reviews everything as correct (same format as the page
    export) from the ground truth, run summary end to end, then check that the page data match the questions."""
    groups = {g_["group_id"]: g_ for g_ in load_groups(cfg)}
    sl = read_json(out_dir(cfg) / "sample_list.json")
    recs = []
    for gid in sl["groups"]:
        grp = groups[gid]
        blind = {q["qid"]: {"answer": gate_d.gt_of(q)} for q in grp["questions"]}
        review = {q["qid"]: {"verdict": "correct"} for q in grp["questions"]}
        recs.append({"group_id": gid, "scene_id": grp["scene_id"], "version": VERSION, "annotator": "selftest", "phase": "review",
                     "blind": blind, "review": review})
    p = out_dir(cfg) / "selftest_annotations.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
    m = summary(cfg, [str(p)], out_name="selftest_report.md")
    assert m["pass"] and m["review_correct_frac"] == 1.0 and not m["incomplete_records"], m
    html = (out_dir(cfg) / "pages" / "index.html").read_text()
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0]) if "const DATA = " in html else None
    n_q = sum(len(groups[g]["questions"]) for g in sl["groups"])
    checks = {"export-all button present": "Export all annotations" in html, "version string present": VERSION in html,
              "page groups = sampled groups": data is not None and len(data["groups"]) == len(sl["groups"]),
              "page questions = sampled questions": data is not None and sum(len(g_["questions"]) for g_ in data["groups"]) == n_q,
              "all image files exist": data is not None and all((out_dir(cfg) / "pages" / v["file"]).exists() for g_ in data["groups"] for v in g_["views"])
              and all((out_dir(cfg) / "pages" / f).exists() for g_ in data["groups"] for q in g_["questions"] for f in q["overlays"] + [q["topdown"]]),
              "masked nodes absent from pages": all(not set((groups[g_["group_id"]].get("view_graph_mask") or {}).get("masked_nodes", []))
                                     & {v["file"].rsplit("/", 1)[1][:-4] for v in g_["views"]} for g_ in data["groups"]) if data else False,
              # the ground truth is embedded only as base64 (decoded and shown after the blind answer is submitted); page data contain no answer fields
              "no answer fields in page data": data is not None and not any(k in q for g_ in data["groups"] for q in g_["questions"] for k in ("answer", "answer_raw", "gt")),
              "numeric ground truth not in plain text": not any(f'"{gate_d.gt_text(q)}"' in html for g in sl["groups"] for q in groups[g]["questions"] if q["answer_type"] == "numeric")}
    write_json(out_dir(cfg) / "selftest.json", checks)
    print(checks)
    assert all(checks.values()), checks
    p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "pages", "package", "selftest", "summary", "baseline"])
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "summary":
        summary(cfg, args.files)
    elif args.cmd == "baseline":
        baseline_excluding_blacklist(cfg, args.files)
    else:
        {"sample": sample, "pages": pages, "package": package, "selftest": selftest}[args.cmd](cfg)


if __name__ == "__main__":
    main()
