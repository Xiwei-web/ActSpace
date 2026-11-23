"""Step 9: human verification package (GATE D). The code only prepares tools and materials; verification is done by annotators.

  sample   Stratified sampling of question groups (strata: constraint type, contains unanswerable, group difficulty). Round 1
           (arkit_only_v0.9): >= 15% per stratum; round 2 (arkit_only_v0.10): new seed, 30 groups in total, no overlap with
           round 1, output in reports/gateD/round2/ (--version selects the version)
  pages    Build the single-page verification app pages/index.html for the sampled groups (--demo N builds only N sample groups
           for checking the interface)
  merge    Merge the per-group JSON files downloaded by an annotator into reports/gateD/annotations_<annotator>.jsonl
  summary  Collect annotations: takes several annotation files (per-group downloads or merged jsonl) and writes the GATE D
           report draft reports/gateD/gateD_report_draft.md (two-phase criteria)

Page (gate_d_app): the blind phase shows the initial view and the raw oracle views of all questions in the group (upright, no
overlays), plus questions and options. The annotator picks an answer (or "cannot be determined"), may flag ambiguous / label
wrong / an unannotated object of the same category in the scene, and may add a note. After the blind answers are submitted the
page shows the ground truth and review materials (views with overlaid boxes, top-down map, masked-question hint). Answers are
kept in the browser's localStorage; "Export all annotations" on the list page produces one jsonl file. Masked nodes of
mask-method unanswerable questions never appear on the page.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .common import load_config, path_of, read_json, work_dir, write_json

DATASET = "arkitscenes"
CANNOT = "cannot_determine"
# Verification directory of each version (under reports/gateD/): round 1 (v0.9) at the root, round 2 (v0.10) in round2/
ROUND_DIRS = {"arkit_only_v0.9": "", "arkit_only_v0.10": "round2", "arkit_only_v0.11": "round3"}


def load_v09(cfg, version=None):
    """Load the question groups of a debiased output version; version defaults to debias.output_version in the config."""
    p = path_of(cfg, "work") / DATASET / "_step8" / (version or cfg["debias"]["output_version"]) / "test_groups.jsonl"
    return [json.loads(l) for l in open(p)]


def gd_dir(cfg) -> Path:
    d = path_of(cfg, "reports") / "gateD"
    d.mkdir(parents=True, exist_ok=True)
    return d


def round_dir(cfg, version) -> Path:
    d = gd_dir(cfg) / ROUND_DIRS[version]
    d.mkdir(parents=True, exist_ok=True)
    return d


def group_keys(grp):
    """Identify "the same question group" across versions (group ids change on reruns): same scene and same initial view,
    or same scene and same object cluster, both count as overlap."""
    return {("init", grp["scene_id"], grp["init_frame"]), ("cluster", grp["scene_id"], tuple(sorted(grp["cluster_obj_ids"])))}


def stratum(grp):
    diffs = Counter(q.get("difficulty") for q in grp["questions"] if q.get("difficulty"))
    return (grp["constraints"][0]["type"] if grp["constraints"] else "none", any(q["evidence_label"] == "unanswerable" for q in grp["questions"]),
            diffs.most_common(1)[0][0] if diffs else "none")


def sample(cfg, version):
    """Round 1 (v0.9): sample ceil(sample_frac x stratum size) per stratum.
    From round 2 on: new seed, human_check.round_groups groups in total, allocated proportionally to stratum size (largest
    remainder method), with no overlap with groups sampled in earlier rounds (judged by group_keys); if fewer than round_groups
    groups are available, all are included."""
    groups = load_v09(cfg, version)
    hc = cfg["human_check"]
    rnd = list(ROUND_DIRS).index(version) + 1
    seen = set()
    for v in list(ROUND_DIRS)[:rnd - 1]:
        prev = {g_["group_id"]: g_ for g_ in load_v09(cfg, v)}
        for gid in read_json(round_dir(cfg, v) / "sample_list.json")["groups"]:
            seen |= group_keys(prev[gid])
    pool = [g_ for g_ in groups if not group_keys(g_) & seen]
    rng = np.random.default_rng([cfg["seed"], 9] if rnd == 1 else [cfg["seed"], 9, rnd])
    by = defaultdict(list)
    for grp in pool:
        by[stratum(grp)].append(grp["group_id"])
    keys = sorted(by, key=str)
    if rnd == 1:
        alloc = {k: math.ceil(hc["sample_frac"] * len(by[k])) for k in keys}
    elif len(pool) <= hc["round_groups"]:  # fewer than round_groups groups available: include all
        alloc = {k: len(by[k]) for k in keys}
    else:
        n = hc["round_groups"]
        quota = {k: n * len(by[k]) / len(pool) for k in keys}
        alloc = {k: int(quota[k]) for k in keys}
        for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:n - sum(alloc.values())]:
            alloc[k] += 1
    picked, rows = [], []
    for k in keys:
        sel = sorted(rng.choice(sorted(by[k]), alloc[k], replace=False).tolist())
        picked += sel
        rows.append((k, len(by[k]), alloc[k]))
    write_json(round_dir(cfg, version) / "sample_list.json",
               {"version": version, "round": rnd, "pool": len(pool), "excluded_overlapping_previous_rounds": len(groups) - len(pool),
                "groups": picked, "strata": [{"stratum": list(map(str, k)), "n": n_, "sampled": m} for k, n_, m in rows]})
    print(f"round {rnd}: sampled {len(picked)} / {len(pool)} groups (excluded {len(groups) - len(pool)} overlapping earlier rounds)")
    return picked


# ---------------------------------------------------------------- Round 3: part A + part B

SIZE_RE = r"\b(larger|smaller)\b"


def has_size_ref(q):
    import re
    return bool(re.search(SIZE_RE, q["text"]))


def prefixed(grp, prefix):
    """Prefix group and question ids of part B so they do not clash with part A (pages, annotation records and the report
    all use the prefixed ids)."""
    import copy
    g2 = copy.deepcopy(grp)
    g2["source_group_id"] = grp["group_id"]
    g2["group_id"] = prefix + grp["group_id"]
    for q in g2["questions"]:
        q["qid"] = prefix + q["qid"]
    for c in g2["constraints"]:
        c["qids"] = [prefix + x for x in c["qids"]]
    return g2


def b_config(cfg):
    return load_config(cfg["human_check"]["round3"]["b_config"])


def b_cancelled(cfg):
    """Size references were dropped permanently: part B is only summarized, does not count towards the verdict, and the
    outputs of the experimental branch have been deleted."""
    return bool(cfg["human_check"]["round3"].get("b_cancelled"))


def load_b_groups(cfg, version):
    """Sampled part-B groups (prefixed), stored in <round dir>/b_groups.jsonl (deleted after part B was cancelled)."""
    p = round_dir(cfg, version) / "b_groups.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def alloc_quota(sizes, n, weights=None):
    """Allocate n slots proportionally to (weighted) stratum size by the largest remainder method; a stratum's allocation is
    capped at its size and the remaining slots are redistributed among the other strata."""
    weights = weights or {k: 1.0 for k in sizes}
    keys = sorted(sizes, key=str)
    n = min(n, sum(sizes.values()))
    alloc = {k: 0 for k in keys}
    while sum(alloc.values()) < n:
        open_ = [k for k in keys if alloc[k] < sizes[k]]
        left = n - sum(alloc.values())
        tot = sum(sizes[k] * weights[k] for k in open_)
        quota = {k: left * sizes[k] * weights[k] / tot for k in open_}
        add = {k: int(quota[k]) for k in open_}
        for k in sorted(open_, key=lambda k: (-(quota[k] - add[k]), str(k)))[:left - sum(add.values())]:
            add[k] += 1
        for k in open_:
            alloc[k] += min(add[k], sizes[k] - alloc[k])
    return alloc


def sample_round3(cfg, version):
    """A: groups new in v0.11 (gateD_carryover status "new") that do not overlap groups sampled in the first two rounds,
    stratified by (constraint type or none, contains T4), with strata containing T4 or without constraints up-weighted; a_groups
    groups are sampled. B: groups of the size-reference experimental branch that contain larger / smaller, with no overlap with
    the first two rounds or part A, stratified by (constraint type or none); b_groups groups are sampled and their ids prefixed."""
    r3 = cfg["human_check"]["round3"]
    if b_cancelled(cfg):
        raise SystemExit("Round 3 has been annotated; part B was cancelled and the experimental branch outputs were deleted, so it is not resampled")
    seen = set()
    for v in list(ROUND_DIRS)[:list(ROUND_DIRS).index(version)]:
        prev = {g_["group_id"]: g_ for g_ in load_v09(cfg, v)}
        for gid in read_json(round_dir(cfg, v) / "sample_list.json")["groups"]:
            seen |= group_keys(prev[gid])
    co = read_json(path_of(cfg, "work") / DATASET / "_step8" / version / "gateD_carryover.json")["groups"]
    groups = load_v09(cfg, version)
    pool = [g_ for g_ in groups if co[g_["group_id"]]["status"] == "new" and not group_keys(g_) & seen]
    strat = lambda g_: (g_["constraints"][0]["type"] if g_["constraints"] else "none", any(q["template"] == "T4" for q in g_["questions"]))
    by = defaultdict(list)
    for g_ in pool:
        by[strat(g_)].append(g_["group_id"])
    w = {k: r3["a_priority_weight"] if (k[0] == "none" or k[1]) else 1.0 for k in by}
    alloc = alloc_quota({k: len(v) for k, v in by.items()}, r3["a_groups"], w)
    rng = np.random.default_rng([cfg["seed"], 9, 3])
    a = []
    for k in sorted(by, key=str):
        a += sorted(rng.choice(sorted(by[k]), alloc[k], replace=False).tolist())
    a_keys = set().union(*(group_keys(g_) for g_ in groups if g_["group_id"] in a)) if a else set()
    cb = b_config(cfg)
    bg = [g_ for g_ in load_v09(cb, r3["b_version"]) if any(has_size_ref(q) for q in g_["questions"])
          and not group_keys(g_) & (seen | a_keys)]
    byb = defaultdict(list)
    for g_ in bg:
        byb[g_["constraints"][0]["type"] if g_["constraints"] else "none"].append(g_["group_id"])
    allocb = alloc_quota({k: len(v) for k, v in byb.items()}, r3["b_groups"])
    rngb = np.random.default_rng([cfg["seed"], 9, 3, 2])
    b = []
    for k in sorted(byb, key=str):
        b += sorted(rngb.choice(sorted(byb[k]), allocb[k], replace=False).tolist())
    rd = round_dir(cfg, version)
    bmap = {g_["group_id"]: g_ for g_ in bg}
    with open(rd / "b_groups.jsonl", "w") as f:
        for gid in b:
            f.write(json.dumps(prefixed(bmap[gid], r3["b_prefix"]), ensure_ascii=False) + "\n")
    write_json(rd / "sample_list.json", {
        "version": version, "round": list(ROUND_DIRS).index(version) + 1,
        "groups": a + [r3["b_prefix"] + x for x in b],
        "A": {"pool": len(pool), "groups": a, "strata": [{"stratum": list(map(str, k)), "n": len(by[k]), "weight": w[k], "sampled": alloc[k]} for k in sorted(by, key=str)]},
        "B": {"version": r3["b_version"], "pool": len(bg), "groups": [r3["b_prefix"] + x for x in b],
              "strata": [{"stratum": k, "n": len(byb[k]), "sampled": allocb[k]} for k in sorted(byb)]}})
    print(f"round 3: A {len(a)} / {len(pool)} new non-overlapping groups; B {len(b)} / {len(bg)} size-reference groups")
    return a, b


# ---------------------------------------------------------------- Pages

CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e3e2dd; --accent:#2c6ecb; --warn:#b4541f; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --fg:#ecebe7; --muted:#a3a29d; --card:#242422; --line:#3a3936; --accent:#7aa7e8; --warn:#e39a6d; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, "PingFang SC", "Segoe UI", sans-serif; }
main { max-width:1180px; margin:0 auto; padding:20px 16px 60px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 10px; }
.meta, .hint { color:var(--muted); font-size:13px; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
.bar input { padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--fg); }
button { padding:7px 14px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; font-size:14px; }
button.secondary { background:transparent; color:var(--accent); }
.views { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:10px; }
figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
figure img { width:100%; display:block; cursor:zoom-in; }
figcaption { padding:6px 8px; font-size:13px; color:var(--muted); }
figure.start { border:2px solid var(--accent); } figure.start figcaption { color:var(--accent); font-weight:600; }
.q { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:12px 0; }
.q .text { font-size:16px; margin:2px 0 10px; }
.q label { display:block; margin:4px 0; cursor:pointer; }
.q .flags { margin-top:10px; padding-top:8px; border-top:1px dashed var(--line); }
.q .flags label { color:var(--warn); }
.q textarea { width:100%; min-height:44px; margin-top:6px; padding:6px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); font:inherit; }
.q input[type=number] { width:120px; padding:5px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); }
.done { color:var(--accent); font-size:13px; }
.review { margin-top:12px; padding:10px 12px; border-radius:6px; background:var(--bg); border:1px solid var(--accent); }
.review .gt { margin-bottom:6px; } label.inline { display:inline-block; margin:0 14px 0 6px; }
input:disabled + * , input:disabled { cursor:not-allowed; }
#zoom { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:10; align-items:center; justify-content:center; }
#zoom img { max-width:96vw; max-height:92vh; }
"""

def gt_text(q):
    """Ground truth shown in the review phase: "letter. option" for multiple choice, the value (meters) for numeric questions,
    "cannot be determined" for unanswerable ones."""
    if q["evidence_label"] == "unanswerable":
        return "Cannot be determined"
    if q["answer_type"] == "numeric":
        return f"{q['answer']} m"
    return next(o for o in q["options"] if o.startswith(q["answer"] + ". "))


def pages(cfg, version, demo=None):
    """Single-page verification app: <round dir>/pages/index.html plus per-group images, see gate_d_app."""
    from . import gate_d_app
    groups = {g_["group_id"]: g_ for g_ in load_v09(cfg, version)}
    sl = read_json(round_dir(cfg, version) / "sample_list.json")
    out = round_dir(cfg, version) / "pages"
    if "B" in sl and b_cancelled(cfg):  # part B cancelled and experimental branch outputs deleted: pages keep part A only
        gate_d_app.build(version, [(cfg, [groups[gid] for gid in sl["A"]["groups"]], None)], out)
        print(f"wrote {len(sl['A']['groups'])} groups (part A only) to {out / 'index.html'}")
        return
    if "B" in sl:  # round 3: part A + part B (B reads objects and visibility with the experimental branch's sandbox config)
        r3 = cfg["human_check"]["round3"]
        parts = [(cfg, [groups[gid] for gid in sl["A"]["groups"]], "Part A"),
                 (b_config(cfg), load_b_groups(cfg, version), r3["b_label"])]
        gate_d_app.build(version, parts, out)
        print(f"wrote {sum(len(p[1]) for p in parts)} groups to {out / 'index.html'}")
        return
    picked = sl["groups"]
    if demo:
        picked = demo_selection(groups, picked, demo)
    gate_d_app.build(version, [(cfg, [groups[gid] for gid in picked], None)], out)
    print(f"wrote {len(picked)} groups to {out / 'index.html'}")


GUIDE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ActSpace human verification: annotation guide</title><style>CSS_HERE body {{ }} main {{ max-width:860px; }} li {{ margin:4px 0; }}</style></head><body><main>
<h1>ActSpace human verification (GATE D): annotation guide</h1>
<p class="meta">Data version VERSION_HERE · N_HERE question groups · completed question by question by one annotator</p>
<h2>1. Opening the page</h2>
<ol><li>Extract the whole zip file you received into any folder on your computer (do not extract only part of it; the images are in the per-group folders).</li>
<li>Open <code>pages/index.html</code> in the extracted folder with Chrome, Edge, Firefox or Safari. All question groups are on this one page (list → click an id to open a group); no internet connection is needed.</li>
<li>Always use the same browser, the same computer and the same extraction folder. Answers are saved in the browser automatically; do not clear browser data or use a private window, otherwise answers that have not been exported will be lost.</li></ol>
<h2>2. Answering (two phases per question group)</h2>
<ol><li>Enter your annotator ID at the top of the page (e.g. ann01) and use the same ID for all question groups.</li>
<li><strong>Phase 1 · blind answers</strong>: answer only from the raw images on the page. You start from the image marked "Initial view" and can see all listed views.
  The listed views include the views needed to answer the questions, plus "linking views": images in which two objects of the question are visible together, used to relate the positions of several objects.
  Questions "relative to the start" (T4) also have "path views", labelled "Q i path 1/4, 2/4…": in order, they show what is seen while walking from the start to a point where the target object is visible, and help infer where the target is relative to the start.
  <ul><li>Distance questions ask for the straight-line distance between the centers of two objects (meters). <strong>Estimates are fine; just enter one number</strong>.</li>
  <li>Direction questions: <strong>imagine standing at the position and facing the direction described in the question</strong>, then judge front / back / left / right; questions "relative to the start" use the position and heading of the initial view.</li>
  <li>If the views are not enough to answer, choose "Cannot be determined from these views". <strong>If you cannot tell the distance, also choose "Cannot be determined"; do not enter 0 in the number box</strong> (the object of a masked question is not visible in any view, so "Cannot be determined" is the right choice).</li>
  <li>If you notice a problem, tick: the question is ambiguous; the object reference or label is wrong (label wrong, e.g. the object named in the question is something else in the image);
  the scene contains an unannotated object of the same category (e.g. the question says "the refrigerator" but the images show two refrigerators, so it is unclear which one is meant). You can explain in the note.</li></ul></li>
<li>When all questions are answered, click <strong>"Submit blind answers"</strong>. After submission, the blind answers and ticked flags can no longer be changed.</li>
<li><strong>Phase 2 · review</strong>: each question then shows the ground truth and review materials:
  <ul><li>each view with boxes and names of the objects in the question overlaid (green = visible in that view, orange = partially visible, red = not visible), including linking views; for T4 questions the path views come first, in order;</li>
  <li>a top-down map: object centers and names; direction questions show the standing point, heading arrow and four quadrants; questions "relative to the start" show the start position and heading.</li>
  <li>masked questions are marked explicitly: the object in the question was deliberately excluded from the given views, so it cannot be seen, and the ground truth is "Cannot be determined".</li></ul>
  For each question, judge whether the ground truth is correct: Correct / Incorrect / Unsure; if you choose Incorrect, give the reason in the note.</li></ol>
PARTS_HERE<h2>3. Exporting and returning the annotations</h2>
<ol><li>When all question groups are done, go back to the list page and click <strong>"Export all annotations"</strong> at the top right. The browser downloads one file,
  <code>annotations_YOUR-ID_VERSION_HERE.jsonl</code>; send this file to the project coordinator. Do not change the file name or contents.</li>
<li>You can also click "Download this group's annotations" on a group page to get <code>YOUR-ID__GROUP-ID.json</code> as a backup.</li></ol>
<h2>4. Notes</h2>
<ul><li>During phase 1, do not view the page source or obtain the ground truth in any other way; the blind answers serve as the human performance baseline.</li>
<li>The images come from the ARKitScenes dataset and may be used only for this annotation task; do not copy, forward or publish them.</li></ul>
</main></body></html>"""


def package(cfg, version):
    """Pack the pages and the annotation guide into reports/gateD/<round dir>/gateD_pages_<version suffix>.zip
    (e.g. gateD_pages_v0.10.zip)."""
    import zipfile
    rd = round_dir(cfg, version)
    sl = read_json(rd / "sample_list.json")
    n = len(sl["A"]["groups"] if "B" in sl and b_cancelled(cfg) else sl["groups"])
    parts = "" if "B" not in sl or b_cancelled(cfg) else (
        f"<h2>Appendix: part A and part B</h2><p>This round has {len(sl['A']['groups'])} part-A question groups and {len(sl['B']['groups'])} part-B question groups (ids start with "
        f"{cfg['human_check']['round3']['b_prefix']} and are marked \"{cfg['human_check']['round3']['b_label']}\" in the list and on the pages). "
        "Both parts are answered in exactly the same way. Part-B questions use \"the larger X / the smaller X\" to refer to the larger or smaller of two objects of the same category; "
        "please pay particular attention to whether you can identify the referred object in the images. If you cannot, or think the reference does not hold, tick \"object reference or label is wrong\" and explain in the note.</p>")
    guide = GUIDE.replace("CSS_HERE", CSS).replace("VERSION_HERE", version).replace("N_HERE", str(n)).replace("PARTS_HERE", parts).replace("{{", "{").replace("}}", "}")
    (rd / "annotation_guide.html").write_text(guide)
    short = version.rsplit("_", 1)[-1]
    out = rd / f"gateD_pages_{short}.zip"
    root = f"gateD_pages_{short}"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(rd / "annotation_guide.html", f"{root}/annotation_guide.html")
        for f in sorted((rd / "pages").rglob("*")):
            if f.is_file():
                z.write(f, f"{root}/pages/{f.relative_to(rd / 'pages')}")
    print(f"packed {out} ({out.stat().st_size / 1e6:.1f} MB)")


def demo_selection(groups, picked, n):
    """Demo: pick n groups from the sample list covering as much as possible (mask-method unanswerable, T5, T4, T2 with
    multiple options, size references)."""
    want = [lambda g_: any(q["template"] == "T1" for q in g_["questions"]),
            lambda g_: any(q["evidence_label"] == "unanswerable" for q in g_["questions"]),
            lambda g_: any(q["template"] == "T5" for q in g_["questions"]),
            lambda g_: any(q["template"] == "T2" and q.get("variant") in ("closest", "farthest") for q in g_["questions"]),
            lambda g_: any(w in q["text"] for q in g_["questions"] for w in ("larger", "smaller", "largest", "smallest", "armchair")),
            lambda g_: any(q["template"] == "T4" for q in g_["questions"]) and g_["constraints"][0]["type"] == "direction_inverse"]
    sel = []
    for f in want:
        for gid in picked:
            if gid not in sel and f(groups[gid]):
                sel.append(gid)
                break
    for gid in picked:
        if len(sel) >= n:
            break
        if gid not in sel:
            sel.append(gid)
    return sel[:n]


# ---------------------------------------------------------------- Merge and summary (used after annotation)

def merge(cfg, annotator, files):
    out = gd_dir(cfg) / f"annotations_{annotator}.jsonl"
    recs = [json.loads(Path(f).read_text()) for f in files]
    with open(out, "w") as fo:
        for r in sorted(recs, key=lambda r: r["group_id"]):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(recs)} groups -> {out}")


THRESH = np.arange(0.50, 0.951, 0.05)


def load_records(files):
    """Read annotations: per-group JSON (downloaded files) or jsonl (merge output). If the same (annotator, group) appears
    several times, the most recently saved record wins."""
    recs = {}
    for f in files:
        text = Path(f).read_text().strip()
        items = [json.loads(l) for l in text.splitlines() if l.strip()] if f.endswith(".jsonl") else [json.loads(text)]
        for r in items:
            k = (r["annotator"], r["group_id"])
            if k not in recs or r.get("saved_at", "") >= recs[k].get("saved_at", ""):
                recs[k] = r
    return list(recs.values())


def gt_of(q):
    """Ground truth: unanswerable -> cannot_determine; numeric -> answer_raw; multiple choice -> letter."""
    if q["evidence_label"] == "unanswerable":
        return CANNOT
    return float(q["answer_raw"]) if q["answer_type"] == "numeric" else q["answer"]


def summarize(groups, records, hc):
    """Two-phase summary. groups: {group_id: question group}; records: list of annotation records; hc: human_check config.
    Returns a dict of metrics."""
    from sklearn.metrics import cohen_kappa_score
    rows = []
    for r in records:
        grp = groups.get(r["group_id"])
        if grp is None:
            continue
        for q in grp["questions"]:
            b = (r.get("blind") or {}).get(q["qid"], {})
            v = (r.get("review") or {}).get(q["qid"], {}) if r.get("review") else {}
            gt, ans = gt_of(q), b.get("answer")
            if q["answer_type"] == "numeric" and gt != CANNOT:
                score = float(np.mean([abs(ans - gt) / gt < 1 - t for t in THRESH])) if isinstance(ans, (int, float)) else 0.0
            else:
                score = float(ans == gt)
            rows.append({"annotator": r["annotator"], "group_id": r["group_id"], "qid": q["qid"], "template": q["template"],
                         "answerable": q["evidence_label"] != "unanswerable", "numeric": q["answer_type"] == "numeric" and gt != CANNOT,
                         "answer": ans, "gt": gt, "score": score, "verdict": v.get("verdict"), "review_note": v.get("note", ""),
                         "ambiguous": bool(b.get("ambiguous")), "label_wrong": bool(b.get("label_wrong")),
                         "unannotated": bool(b.get("unannotated_same_category_object")), "note": b.get("note", ""),
                         "text": q["text"], "phase": r.get("phase")})
    ans_rows = [x for x in rows if x["answerable"]]
    judged = [x for x in ans_rows if x["verdict"]]
    # first criterion: "reviewed as correct and not flagged label wrong"
    correct_frac = sum(x["verdict"] == "correct" and not x["label_wrong"] for x in judged) / max(len(judged), 1)
    amb_frac = sum(x["ambiguous"] or x["unannotated"] for x in ans_rows) / max(len(ans_rows), 1)
    out = {"n_records": len(records), "n_rows": len(rows), "n_answerable_rows": len(ans_rows), "n_judged": len(judged),
           "review_correct_frac": correct_frac, "ambiguous_or_unannotated_frac": amb_frac,
           "pass_correct": correct_frac >= hc["min_review_correct_frac"], "pass_ambiguous": amb_frac <= hc["max_ambiguous_frac"],
           "incomplete_records": [(r["annotator"], r["group_id"]) for r in records if r.get("phase") != "review"
                                  or any(v.get("verdict") is None for v in (r.get("review") or {}).values())],
           "rows": rows}
    out["pass"] = out["pass_correct"] and out["pass_ambiguous"] and len(judged) > 0
    # human performance in the blind phase (per template; accuracy for multiple choice, MRA for numeric; unanswerable separately)
    perf = {}
    for t in sorted({x["template"] for x in rows}):
        for kind, sel in (("answerable", [x for x in rows if x["template"] == t and x["answerable"]]),
                          ("unanswerable", [x for x in rows if x["template"] == t and not x["answerable"]])):
            if sel:
                perf[(t, kind)] = (float(np.mean([x["score"] for x in sel])), len(sel), "MRA" if sel[0]["numeric"] else "accuracy")
    out["perf"] = perf
    # inter-annotator agreement: for questions with 2 annotators, Cohen's kappa of blind answers (multiple choice, including
    # "cannot be determined") and of review verdicts
    by_q = defaultdict(dict)
    for x in rows:
        by_q[x["qid"]][x["annotator"]] = x
    pairs_mc, pairs_rv, num_diff = [], [], []
    for qid, d in by_q.items():
        if len(d) < 2:
            continue
        a, b = list(d.values())[:2]
        if a["numeric"]:
            if isinstance(a["answer"], (int, float)) and isinstance(b["answer"], (int, float)):
                num_diff.append(abs(a["answer"] - b["answer"]))
        else:
            pairs_mc.append((str(a["answer"]), str(b["answer"])))
        if a["verdict"] and b["verdict"]:
            pairs_rv.append((a["verdict"], b["verdict"]))

    def kappa(pairs):
        if len(pairs) < 2 or len({p for pr in pairs for p in pr}) < 2:
            return None
        return float(cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs]))
    out["kappa_blind_mc"], out["n_pairs_mc"] = kappa(pairs_mc), len(pairs_mc)
    out["kappa_review"], out["n_pairs_review"] = kappa(pairs_rv), len(pairs_rv)
    out["numeric_pair_mean_abs_diff"] = float(np.mean(num_diff)) if num_diff else None
    # flag rates per template
    flags = {}
    for t in sorted({x["template"] for x in rows}):
        sel = [x for x in rows if x["template"] == t]
        flags[t] = {k: sum(x[k] for x in sel) / len(sel) for k in ("ambiguous", "label_wrong", "unannotated")} | {"n": len(sel)}
    out["flags"] = flags
    return out


SIZE_WORDS = ("larger", "smaller", "largest", "smallest", "armchair")


def failure_section(cfg, version, groups, m):
    """Failure reasons for questions that did not pass (answerable questions whose review verdict is not "correct"). The
    category table failure_categories.json is compiled by hand from the annotators' notes (per question group); this function only
    presents it, plus a data-related statistic (whether a size reference is used)."""
    import re
    p = round_dir(cfg, version) / "failure_categories.json"
    fails = [x for x in m["rows"] if x["answerable"] and x["verdict"] != "correct"]
    size = lambda x: bool(re.search(r"\b(" + "|".join(SIZE_WORDS) + r")\b", x["text"]))
    ans = [x for x in m["rows"] if x["answerable"]]
    L = ["", "## 7. Failure reasons", "",
         f"Answerable questions whose review verdict is not \"correct\": {len(fails)} (unsure {sum(x['verdict'] == 'unsure' for x in fails)}, "
         f"incorrect {sum(x['verdict'] == 'incorrect' for x in fails)}).", "",
         "By whether the question text uses a size reference (larger / smaller / largest / smallest / armchair):", "",
         "| Question text | Reviewed answerable questions | Not \"correct\" | Fraction |", "|---|---|---|---|"]
    for name, sel in (("uses size reference", [x for x in ans if size(x)]), ("no size reference", [x for x in ans if not size(x)])):
        bad = sum(x["verdict"] != "correct" for x in sel)
        L.append(f"| {name} | {len(sel)} | {bad} | {bad / max(len(sel), 1):.1%} |")
    if not fails:
        return L + ["", "No question failed in this round; nothing to categorize."]
    if not p.exists():
        return L + ["", "(No failure_categories.json yet; not categorized.)"]
    fc = json.loads(p.read_text())
    cat = {gid: v for gid, v in fc["groups"].items()}
    cnt = Counter(cat.get(x["group_id"], {}).get("category", "uncategorized") for x in fails)
    L += ["", f"Categorization: {fc['note']}. Categorized per question group (notes within a group usually refer to the whole group).", "",
          "| Category | Description | Questions |", "|---|---|---|",
          *[f"| {k} | {v.split(' ', 1)[1]} | {cnt.get(k, 0)} |" for k, v in fc["categories"].items()],
          *([f"| uncategorized | — | {cnt['uncategorized']} |"] if cnt.get("uncategorized") else []), "",
          "Per-question list (notes are the annotator's notes from the blind and review phases; basis is the reason for the category):", "",
          "| Question | Template | Blind answer | Ground truth | Review | Category | Size reference | Notes | Basis |", "|---|---|---|---|---|---|---|---|---|"]
    for x in fails:
        c = cat.get(x["group_id"], {})
        note = "; ".join(n for n in (x["note"], x["review_note"]) if n).replace("|", "/").replace("\n", " ")
        L.append(f"| {x['qid']} | {x['template']} | {x['answer']} | {x['gt']} | {x['verdict']} | {c.get('category', 'uncategorized')} | "
                 f"{'yes' if size(x) else 'no'} | {note} | {c.get('basis', '')} |")
    return L


def _correct(rows):
    """Answerable, reviewed questions: count reviewed as correct and not flagged label wrong (first GATE D criterion)."""
    sel = [x for x in rows if x["answerable"] and x["verdict"]]
    return sum(x["verdict"] == "correct" and not x["label_wrong"] for x in sel), len(sel)


def compare_section(cfg, version, groups, m):
    """Comparison with the previous round: correct rate overall, per template, by size reference and by whether linking views
    are needed; human performance in the blind phase."""
    import re
    rounds = list(ROUND_DIRS)
    k = rounds.index(version)
    if k == 0:
        return []
    pv = rounds[k - 1]
    pfiles = sorted(str(f) for f in (round_dir(cfg, pv) / "annotations").glob("*.json"))
    pgroups = {g_["group_id"]: g_ for g_ in load_v09(cfg, pv)}
    pm = summarize(pgroups, load_records(pfiles), cfg["human_check"])
    qmap = {q["qid"]: q for g_ in groups.values() for q in g_["questions"]} | {q["qid"] + "@prev": q for g_ in pgroups.values() for q in g_["questions"]}
    size = lambda x: bool(re.search(r"\b(" + "|".join(SIZE_WORDS) + r")\b", x["text"]))
    fmt = lambda c, n: f"{c / n:.1%} ({c} / {n})" if n else "—"
    cur, prev = m["rows"], pm["rows"]
    L = ["", f"## 8. Comparison with the previous round ({pv})", "",
         f"Metric: answerable, reviewed questions, fraction \"reviewed as correct and not flagged label wrong\" (first GATE D criterion). The previous round is recomputed with the same metric (the 72.7% in the previous round's report is the fraction reviewed as correct, without excluding label wrong).", "",
         "| Subset | Previous round | This round |", "|---|---|---|",
         f"| **Overall** | {fmt(*_correct(prev))} | {fmt(*_correct(cur))} |"]
    for t in sorted({x["template"] for x in cur + prev}):
        L.append(f"| {t} | {fmt(*_correct([x for x in prev if x['template'] == t]))} | {fmt(*_correct([x for x in cur if x['template'] == t]))} |")
    for name, f in (("uses size reference (larger / smaller; previous round also largest / smallest / armchair)", size), ("no size reference", lambda x: not size(x))):
        L.append(f"| {name} | {fmt(*_correct([x for x in prev if f(x)]))} | {fmt(*_correct([x for x in cur if f(x)]))} |")
    lv = lambda x: bool(qmap[x["qid"]].get("link_views"))
    L += [f"| needs linking views (link_views non-empty) | — | {fmt(*_correct([x for x in cur if lv(x)]))} |",
          f"| no linking views needed | — | {fmt(*_correct([x for x in cur if not lv(x)]))} |", "",
          "The previous version has no link_views field (the evidence-linking rule was added after round 1), shown as \"—\". In this round, non-empty link_views is equivalent to "
          "\"answerable T1 / T2 / T3 / T5 questions\" (all have >= 2 evidence objects); T4 and unanswerable questions have empty link_views.", "",
          "Other comparisons:", "",
          "| Metric | Previous round | This round |", "|---|---|---|",
          f"| Question groups / reviewed answerable questions | {len({x['group_id'] for x in prev})} / {_correct(prev)[1]} | {len({x['group_id'] for x in cur})} / {_correct(cur)[1]} |",
          f"| Ambiguous or unannotated same-category object (answerable) | {pm['ambiguous_or_unannotated_frac']:.1%} | {m['ambiguous_or_unannotated_frac']:.1%} |",
          f"| Flagged label wrong (answerable) | {sum(x['label_wrong'] for x in prev if x['answerable']) / max(sum(x['answerable'] for x in prev), 1):.1%} | "
          f"{sum(x['label_wrong'] for x in cur if x['answerable']) / max(sum(x['answerable'] for x in cur), 1):.1%} |", "",
          "## 9. Human performance in the blind phase: comparison with the previous round", "",
          "Multiple choice: accuracy (for unanswerable questions \"cannot be determined\" is correct); numeric (answerable T1): MRA.", "",
          "| Template | Question type | Metric | Previous round | This round |", "|---|---|---|---|---|"]
    for key in sorted(set(pm["perf"]) | set(m["perf"])):
        a, b = pm["perf"].get(key), m["perf"].get(key)
        metric = (a or b)[2]
        L.append(f"| {key[0]} | {key[1]} | {metric} | {f'{a[0]:.3f} ({a[1]})' if a else '—'} | {f'{b[0]:.3f} ({b[1]})' if b else '—'} |")
    return L


def flagged_objects(cfg, groups, records, retracted=()):
    """Evidence objects of questions flagged label wrong or "unannotated same-category object":
    {(video_id, obj_id): (name, [sources])}."""
    objs, src = {}, defaultdict(list)
    for r in records:
        grp = groups[r["group_id"]]
        vid = grp["scene_id"].split("_", 1)[1]
        names = {o["obj_id"]: o.get("ref_name") or o["category"] for o in read_json(work_dir(cfg, DATASET, vid) / "objects.json")}
        for q in grp["questions"]:
            b = r["blind"][q["qid"]]
            if not (b.get("label_wrong") or b.get("unannotated_same_category_object")) or q["qid"] in retracted:
                continue
            flag = "label_wrong" if b.get("label_wrong") else "unannotated_same_category_object"
            for o in q["evidence_objs"]:
                objs[(vid, o)] = names[o]
                src[(vid, o)].append(f"{q['qid']} [{flag}]")
    return {k: (objs[k], src[k]) for k in sorted(objs)}


def blacklist_suggestion_section(cfg, version, groups, records):
    """Objects suggested for the blacklist (list only; the blacklist and data are not modified)."""
    import yaml
    from .common import ROOT
    fo = flagged_objects(cfg, groups, records)
    bl = yaml.safe_load(open(ROOT / cfg["objects"]["blacklist_file"]))
    have = {(str(o["video_id"]), int(o["obj_id"])) for o in bl.get("objects") or []}
    L = ["", "## 10. Objects suggested for the blacklist (blacklist and data not modified)", "",
         "Listed by the round-1 rule (all evidence objects of questions flagged label wrong or \"unannotated same-category object\" go on the blacklist); "
         "the \"Already blacklisted\" column is checked against `configs/object_blacklist.yaml`.", "",
         "| Scene | obj_id | Name | Already blacklisted | Source (question [flag]) |", "|---|---|---|---|---|"]
    for (vid, o), (name, src) in fo.items():
        L.append(f"| {vid} | {o} | {name} | {'yes' if (vid, o) in have else 'no'} | {'; '.join(src)} |")
    if not fo:
        L.append("| — | — | — | — | no flagged questions in this round |")
    return L


def b_section(cfg, version, groups, recs):
    """Round 3 part B (size-reference check): fraction of size-reference questions "correct and not flagged label wrong";
    only at >= b_min_size_correct_frac would size references be considered for the release version. The two GATE D metrics over
    all answerable part-B questions are given for reference."""
    r3, hc = cfg["human_check"]["round3"], cfg["human_check"]
    m = summarize(groups, recs, hc)
    rows = [x for x in m["rows"] if x["answerable"]]
    sz = [x for x in rows if has_size_ref({"text": x["text"]})]
    c, n = _correct(sz)
    frac = c / n if n else 0.0
    L = ["", f"## {r3['b_label']} (experimental branch {r3['b_version']}, not part of {version})", "",
         f"- {len(recs)} annotation records; {len(rows)} answerable questions, {len(sz)} of which contain larger / smaller.", "",
         "| Criterion | Actual | Required | Result |", "|---|---|---|---|",
         f"| Size-reference questions \"correct and not flagged label wrong\" | {frac:.1%} ({c} / {n}) | >= {r3['b_min_size_correct_frac']:.0%} | "
         f"{'met' if n and frac >= r3['b_min_size_correct_frac'] else '**not met**'} |", "",
         f"For reference (all answerable part-B questions): \"correct and not flagged label wrong\" {m['review_correct_frac']:.1%}, "
         f"ambiguous or unannotated same-category object {m['ambiguous_or_unannotated_frac']:.1%}.", "",
         "Per template (size-reference questions):", "", "| Template | Correct and not flagged label wrong |", "|---|---|"]
    for t in sorted({x["template"] for x in sz}):
        ct, nt = _correct([x for x in sz if x["template"] == t])
        L.append(f"| {t} | {ct / nt:.1%} ({ct} / {nt}) |")
    L += ["", "Human performance in the blind phase (part B):", "", "| Template | Question type | Metric | Human score | Count |", "|---|---|---|---|---|",
          *[f"| {t} | {k} | {metric} | {v:.3f} | {nn} |" for (t, k), (v, nn, metric) in sorted(m["perf"].items())], "",
          "Part-B questions to follow up (review verdict not \"correct\" or any flag ticked):", "",
          "| Question | Template | Size reference | Blind answer | Ground truth | Review | Flags | Notes |", "|---|---|---|---|---|---|---|---|"]
    for x in m["rows"]:
        fl = [k for k in ("ambiguous", "label_wrong", "unannotated") if x[k]]
        if x["verdict"] in ("incorrect", "unsure") or fl:
            note = "; ".join(t for t in (x["note"], x["review_note"]) if t).replace("|", "/")
            L.append(f"| {x['qid']} | {x['template']} | {'yes' if has_size_ref({'text': x['text']}) else 'no'} | {x['answer']} | {x['gt']} | "
                     f"{x['verdict'] or '—'} | {', '.join(fl) or '—'} | {note} |")
    return L


def b_cancelled_section(cfg, version, recs):
    """Size references were dropped permanently: part B is only summarized, marked as cancelled, and does not count towards
    the GATE D verdict. The experimental branch outputs (including b_groups.jsonl) were deleted; the statistics come from
    <round dir>/b_part_cancelled.json, saved before deletion."""
    r3 = cfg["human_check"]["round3"]
    st = read_json(round_dir(cfg, version) / "b_part_cancelled.json")
    L = ["", f"## {r3['b_label']}: cancelled (does not count towards the verdict)", "",
         "**Size references were dropped permanently**: human verification of part B showed that larger / smaller references are highly ambiguous. "
         f"Part B does not count towards the GATE D verdict; the outputs of experimental branch {st['version']} were deleted and are not merged into any release version. The following is statistics only.", "",
         f"- {st['n_groups']} groups sampled from {st['n_scenes']} scenes, {st['n_questions']} questions in total ({st['n_unanswerable']} unanswerable), "
         f"{st['n_size_ref_questions']} of which contain larger / smaller.",
         f"- Returned part-B annotation records: {len(recs)}" + ("" if recs else " (only part-A annotations were uploaded in this round)") + ".", "",
         "| Template | Questions | With size reference |", "|---|---|---|",
         *[f"| {t} | {v['questions']} | {v['size_ref']} |" for t, v in st["by_template"].items()],
         f"| Total | {st['n_questions']} | {st['n_size_ref_questions']} |"]
    if recs:
        L += ["", "(The part-B group data were deleted; no per-question summary.)"]
    return L


def summary(cfg, files, out=None):
    hc = cfg["human_check"]
    records = load_records(files)
    version = Counter(r.get("version") for r in records).most_common(1)[0][0] if records else cfg["debias"]["output_version"]
    groups = {g_["group_id"]: g_ for g_ in load_v09(cfg, version)}
    sl = read_json(round_dir(cfg, version) / "sample_list.json")
    b_ids = set(sl["B"]["groups"]) if "B" in sl else set()
    b_groups = {g_["group_id"]: g_ for g_ in load_b_groups(cfg, version)}
    b_records = [r for r in records if r["group_id"] in b_ids]
    records = [r for r in records if r["group_id"] not in b_ids]  # the sections below cover part A only (or rounds without parts)
    sample = [g_ for g_ in sl["groups"] if g_ not in b_ids]
    m = summarize(groups, records, hc)
    per_group = Counter(r["group_id"] for r in records)
    annotators = sorted({r["annotator"] for r in records})
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    L = ["# GATE D human verification report (draft)" + (": part A first, then part B" if b_ids else ""), "",
         f"> Generated automatically, pending review. Data version {version}; verdict uses the two-phase criteria"
         " (replacing the earlier \"annotator agreement with ground truth >= 90%\").", "",
         "## 1. Collected annotations", "",
         f"- {len(annotators)} annotators: {', '.join(annotators) or 'none'}; {len(records)} annotation records (annotator x question group).",
         f"- {len(sample)} sampled question groups; {sum(per_group[g_] >= hc['annotators_per_group'] for g_ in sample)} with >= {hc['annotators_per_group']} annotators; "
         f"{sum(per_group[g_] == 0 for g_ in sample)} not annotated.",
         f"- Records that have not completed the review phase: {len(m['incomplete_records'])}" + (f": {m['incomplete_records'][:20]}" if m["incomplete_records"] else "") + ".", "",
         "## 2. GATE D verdict (review phase, answerable questions)", "",
         "| Criterion | Actual | Required | Result |", "|---|---|---|---|",
         f"| Reviewed as \"correct\" and not flagged label wrong | {m['review_correct_frac']:.1%} ({m['n_judged']} reviews) | >= {hc['min_review_correct_frac']:.0%} | {'pass' if m['pass_correct'] else '**fail**'} |",
         f"| Flagged ambiguous or \"unannotated same-category object\" | {m['ambiguous_or_unannotated_frac']:.1%} ({m['n_answerable_rows']} questions) | <= {hc['max_ambiguous_frac']:.0%} | {'pass' if m['pass_ambiguous'] else '**fail**'} |", "",
         f"**Conclusion (draft): {'GATE D passed' if m['pass'] else 'GATE D failed'}**.", "",
         "## 3. Human performance baseline (blind phase)", "",
         "Multiple choice: accuracy (for unanswerable questions \"cannot be determined\" is correct); numeric: MRA (thresholds 0.50–0.95, non-numeric answers score 0).", "",
         "| Template | Question type | Metric | Human score | Count |", "|---|---|---|---|---|",
         *[f"| {t} | {k} | {metric} | {v:.3f} | {n} |" for (t, k), (v, n, metric) in sorted(m["perf"].items())], "",
         "## 4. Inter-annotator agreement", "",
         f"- Blind answers (multiple choice, including \"cannot be determined\"): Cohen's kappa = {fmt(m['kappa_blind_mc'])} ({m['n_pairs_mc']} pairs)",
         f"- Review verdicts: Cohen's kappa = {fmt(m['kappa_review'])} ({m['n_pairs_review']} pairs)",
         f"- Mean absolute difference between two annotators' numeric estimates = {fmt(m['numeric_pair_mean_abs_diff'])} m", "",
         "## 5. Flag rates per template (ticked in the blind phase)", "",
         "| Template | Count | ambiguous | label wrong | Unannotated same-category object |", "|---|---|---|---|---|",
         *[f"| {t} | {f['n']} | {f['ambiguous']:.1%} | {f['label_wrong']:.1%} | {f['unannotated']:.1%} |" for t, f in m["flags"].items()], "",
         "## 6. Questions to follow up", "",
         "Questions reviewed as \"incorrect\" or \"unsure\", or with any flag ticked:", "",
         "| Question | Template | Annotator | Review | Flags | Blind answer / ground truth | Notes |", "|---|---|---|---|---|---|---|"]
    for x in m["rows"]:
        fl = [k for k in ("ambiguous", "label_wrong", "unannotated") if x[k]]
        if x["verdict"] in ("incorrect", "unsure") or fl:
            note = "; ".join(n for n in (x["note"], x["review_note"]) if n).replace("|", "/")
            L.append(f"| {x['qid']} | {x['template']} | {x['annotator']} | {x['verdict'] or '—'} | {', '.join(fl) or '—'} | "
                     f"{x['answer']} / {x['gt']} | {note} |")
    L += failure_section(cfg, version, groups, m)
    L += compare_section(cfg, version, groups, m)
    L += blacklist_suggestion_section(cfg, version, groups, records)
    if b_ids and b_cancelled(cfg):
        L += b_cancelled_section(cfg, version, b_records)
    elif b_groups:
        L += b_section(cfg, version, b_groups, b_records)
        L += [x.replace("## 10. Objects suggested for the blacklist", "### Part B: objects suggested for the blacklist") for x in
              blacklist_suggestion_section(b_config(cfg), version, b_groups, b_records)]
    app = round_dir(cfg, version) / "appendix.md"  # hand-written supplement (re-check conclusions, what-if analyses), appended verbatim
    if app.exists():
        L += ["", app.read_text().rstrip()]
    Path(out or round_dir(cfg, version) / "gateD_report_draft.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:24]))


def question_key(grp, q):
    """Identify "the same question" across versions (group ids, option letter order and paraphrases may change): scene,
    initial view, template, variant, slots, option set, answer content."""
    ans = q["answer_raw"] if q["answer_type"] == "numeric" else next(o.split(". ", 1)[1] for o in q["options"] if o.startswith(q["answer"] + ". "))
    opts = tuple(sorted(map(str, q.get("option_objs") or q.get("option_values") or [])))
    return (grp["scene_id"], grp["init_frame"], q["template"], q.get("variant"), tuple(sorted(q["slots"].items())), opts,
            str(ans), q["evidence_label"] == "unanswerable")


def carryover(cfg, version):
    """Questions already verified in GATE D that are still in the new version keep their verification results; also records
    which groups are new. Writes work/arkitscenes/_step8/<version>/gateD_carryover.json (test_groups.jsonl is not modified).
    Group status: verified = all questions were verified in some earlier round; partial = some were; new = none were."""
    from .common import assert_not_frozen
    assert_not_frozen(path_of(cfg, "work") / DATASET / "_step8" / version)
    verified = {}
    for v in ROUND_DIRS:
        if v == version:
            break
        files = sorted(str(f) for f in (round_dir(cfg, v) / "annotations").glob("*.json"))
        if not files:
            continue
        gs = {g_["group_id"]: g_ for g_ in load_v09(cfg, v)}
        for r in load_records(files):
            grp = gs[r["group_id"]]
            for q in grp["questions"]:
                b, rv = r["blind"].get(q["qid"], {}), (r.get("review") or {}).get(q["qid"], {})
                verified[question_key(grp, q)] = {"version": v, "qid": q["qid"], "verdict": rv.get("verdict"),
                                                  "label_wrong": bool(b.get("label_wrong")), "ambiguous": bool(b.get("ambiguous")),
                                                  "unannotated_same_category_object": bool(b.get("unannotated_same_category_object"))}
    out = {"version": version, "rule": "questions verified in an earlier round keep their verification result if they are still in this version (matched by question_key)", "groups": {}}
    for grp in load_v09(cfg, version):
        qs = {q["qid"]: verified.get(question_key(grp, q)) for q in grp["questions"]}
        n = sum(v is not None for v in qs.values())
        out["groups"][grp["group_id"]] = {"status": "verified" if n == len(qs) else "partial" if n else "new",
                                          "questions": {k: v for k, v in qs.items() if v}}
    c = Counter(g_["status"] for g_ in out["groups"].values())
    out["counts"] = dict(c)
    write_json(path_of(cfg, "work") / DATASET / "_step8" / version / "gateD_carryover.json", out)
    print(f"carryover {version}: {dict(c)}; questions carried {sum(len(g_['questions']) for g_ in out['groups'].values())}")
    return out


# Flags explicitly retracted by the annotator (the note says they can be ignored); not used for the blacklist.
# The value is written verbatim into configs/object_blacklist.yaml (a hashed file), so it is kept unchanged.
RETRACTED_FLAGS = {"g_000110_q3": "标注员备注：前面关于“视角中不存在 TV”的反馈可以忽略"}
EXCLUDED_GROUPS = ("g_000171", "g_000222")  # the scenes of these two groups are excluded entirely


def build_blacklist(cfg, files):
    """Evidence objects of questions flagged label wrong or "unannotated same-category object" in round 1 go on the
    blacklist; the scenes of question groups recommended for removal are excluded entirely. Writes configs/object_blacklist.yaml.
    The decided_by / rule / reason values are kept as originally written because they are stored in that hashed file."""
    import yaml
    from .common import ROOT
    groups = {g_["group_id"]: g_ for g_ in load_v09(cfg, "arkit_only_v0.9")}  # round 1 verified v0.9
    objs, src = {}, defaultdict(list)
    for r in load_records(files):
        grp = groups[r["group_id"]]
        vid = grp["scene_id"].split("_", 1)[1]
        names = {o["obj_id"]: o.get("ref_name") or o["category"] for o in read_json(work_dir(cfg, DATASET, vid) / "objects.json")}
        for q in grp["questions"]:
            b = r["blind"][q["qid"]]
            if not (b.get("label_wrong") or b.get("unannotated_same_category_object")) or q["qid"] in RETRACTED_FLAGS:
                continue
            flag = "label_wrong" if b.get("label_wrong") else "unannotated_same_category_object"
            for o in q["evidence_objs"]:
                objs[(vid, o)] = names[o]
                src[(vid, o)].append(f"{q['qid']} [{flag}] {(b.get('note') or '').strip()[:60]}")
    scenes = sorted({groups[g_]["scene_id"].split("_", 1)[1] for g_ in EXCLUDED_GROUPS})
    out = {"decided_by": "负责人 2026-11-01（GATE D 第一轮之后）",
           "rule": "本轮被勾选 label wrong 或“场景中存在同类但未被标注的物体”的题，其证据物体不再用于出题；"
                   "负责人建议舍弃的题目组（g_000171、g_000222）所在场景整场景剔除。",
           "retracted_flags_not_used": RETRACTED_FLAGS,
           "scenes": [{"video_id": v, "reason": "负责人建议整场景舍弃（GATE D 第一轮 " + ", ".join(
               g_ for g_ in EXCLUDED_GROUPS if groups[g_]["scene_id"].endswith(v)) + "）"} for v in scenes],
           "objects": [{"video_id": v, "obj_id": int(o), "name": objs[(v, o)], "flagged_in": src[(v, o)]}
                       for (v, o) in sorted(objs)]}
    path = ROOT / "configs" / "object_blacklist.yaml"
    path.write_text("# Generated by `python -m actspace_build.gate_d blacklist reports/gateD/annotations/*.json`; entries may be added or removed by hand.\n"
                    + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200))
    print(f"blacklist: {len(out['objects'])} objects in {len({v for v, _ in objs})} scenes; {len(scenes)} scenes excluded -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "pages", "package", "merge", "summary", "blacklist", "carryover"])
    ap.add_argument("--demo", type=int, default=None)
    ap.add_argument("--version", help="debiased output version (default: debias.output_version)")
    ap.add_argument("--annotator")
    ap.add_argument("--out", help="output path of summary (default: reports/gateD/gateD_report_draft.md)")
    ap.add_argument("files", nargs="*")
    args = ap.parse_intermixed_args()  # allows: summary --out X files...
    cfg = load_config()
    version = args.version or cfg["debias"]["output_version"]
    if args.cmd == "sample":
        sample_round3(cfg, version) if version == "arkit_only_v0.11" else sample(cfg, version)
    elif args.cmd == "pages":
        pages(cfg, version, args.demo)
    elif args.cmd == "package":
        package(cfg, version)
    elif args.cmd == "carryover":
        carryover(cfg, version)
    elif args.cmd == "blacklist":
        build_blacklist(cfg, args.files)
    elif args.cmd == "merge":
        merge(cfg, args.annotator, args.files)
    else:
        summary(cfg, args.files, args.out)


if __name__ == "__main__":
    main()
