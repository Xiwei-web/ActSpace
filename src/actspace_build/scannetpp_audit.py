"""ScanNet++ audit of all question objects (a full census instead of further re-seeded spot checks).

Scope: every referable object actually used in work/scannetpp/_step6/test_groups.jsonl (slot, option or evidence objects).
One tile per object: among the oracle / link views of the questions that use it, take the frame where the object is visible with
the largest projected area; otherwise the visible frame with the largest area among all nodes (otherwise the partial frame with
the largest area). The full frame is resized to width 480 with the object's box drawn, captioned with its name, scene id and
obj_id. Sorted by scene, 16 tiles per page (4 × 4).

  tiles   write reports/scannetpp_object_audit/page_XX.jpg and work/scannetpp/_full/audit_objects.csv (page, tile key)
  report  read the per-tile judgments reports/scannetpp_object_audit_judgments.tsv (key, recognizable, name_match, unsure, note)
          and write reports/scannetpp_object_audit.md (with an appendix listing objects that became non-referable after
          look-alike categories were merged)

Usage: python -m actspace_build.scannetpp_audit tiles|report
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from .common import load_config, path_of, read_json
from .scannetpp import sp_work
from .scannetpp_step4_report import base_image, box_of, put
from .step1_arkit import draw_obbs
from .step4_graph_visibility import STATE_PARTIAL, STATE_VISIBLE

TILE_W, COLS, PER_PAGE = 480, 4, 16


def used_objects(cfg):
    """{(scene, obj_id): {"views": set of oracle/link nodes, "n_questions": number of questions}}."""
    out = defaultdict(lambda: {"views": set(), "n_questions": 0})
    for l in open(sp_work(cfg) / "_step6" / "test_groups.jsonl"):
        g_ = json.loads(l)
        sid = g_["scene_id"].split("_", 1)[1]
        for q in g_["questions"]:
            objs = set(q["slots"].values()) | set(q.get("option_objs") or []) | set(q["evidence_objs"])
            views = set(q.get("oracle_nodes") or []) | set(q.get("link_views") or [])
            for o in objs:
                out[(sid, o)]["views"] |= views
                out[(sid, o)]["n_questions"] += 1
    return out


def best_frame(v, nodes, row, views):
    """Return (node, source)."""
    idx = [nodes.index(n) for n in views if n in nodes]
    for pool, src in ((idx, "oracle/link"), (list(range(len(nodes))), "all nodes")):
        vis = [c for c in pool if v["state"][row, c] == STATE_VISIBLE]
        if vis:
            return nodes[max(vis, key=lambda c: float(v["area_frac"][row, c]))], src
    part = [c for c in range(len(nodes)) if v["state"][row, c] == STATE_PARTIAL]
    return (nodes[max(part, key=lambda c: float(v["area_frac"][row, c]))], "partial") if part else (None, "none")


def tiles(cfg):
    used = used_objects(cfg)
    out_dir = path_of(cfg, "reports") / "scannetpp_object_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    rows, imgs = [], []
    for sid in sorted({s for s, _ in used}):
        wd = sp_work(cfg) / sid
        objs = {o["obj_id"]: o for o in read_json(wd / "objects.json")}
        v = np.load(wd / "visibility.npz")
        nodes = [str(x) for x in v["frame_ids"]]
        rowi = {int(o): k for k, o in enumerate(v["obj_ids"])}
        frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
        for oid in sorted(o for s, o in used if s == sid):
            o = objs[oid]
            if not o["referable"]:
                continue  # non-referable object in the evidence (should not happen; the integrity check covers it separately)
            fid, src = best_frame(v, nodes, rowi[oid], used[(sid, oid)]["views"])
            if fid is None:
                img = np.full((360, TILE_W, 3), 255, np.uint8)
            else:
                img, K, T = base_image(cfg, frames[fid], TILE_W)
                draw_obbs(img, [box_of(o, "")], K, T, np.eye(3), color=(0, 230, 0))
            k = len(rows)
            page, cell = k // PER_PAGE + 1, k % PER_PAGE + 1
            put(img, f"P{page}-{cell:02d}  {o['ref_name']}", (6, 20), 0.55)
            put(img, f"{sid} #{oid}  ({o['source_label']})", (6, img.shape[0] - 10), 0.45)
            imgs.append(img)
            rows.append({"key": f"P{page}-{cell:02d}", "page": page, "cell": cell, "scene_id": sid, "obj_id": oid, "name": o["ref_name"],
                         "source_label": o["source_label"], "frame_id": fid, "frame_source": src, "n_questions": used[(sid, oid)]["n_questions"]})
    H = max(i.shape[0] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, H - i.shape[0] + 4, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs]
    for p in range(0, len(imgs), PER_PAGE):
        chunk = imgs[p:p + PER_PAGE]
        while len(chunk) % COLS:
            chunk.append(np.full_like(imgs[0], 255))
        sheet = np.concatenate([np.concatenate(chunk[r:r + COLS], 1) for r in range(0, len(chunk), COLS)], 0)
        cv2.imwrite(str(out_dir / f"page_{p // PER_PAGE + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    pd.DataFrame(rows).to_csv(sp_work(cfg) / "_full" / "audit_objects.csv", index=False)
    print(f"{len(rows)} objects, {len(rows) // PER_PAGE + (len(rows) % PER_PAGE > 0)} pages ->", out_dir)


def s29_changes(cfg):
    """Objects whose referable status changed with the small_sofa / look-alike merge rule update (earlier snapshot: referable_before_s28_s29.json)."""
    before = read_json(sp_work(cfg) / "_full" / "referable_before_s28_s29.json")
    lost = []
    for sid, objs in before.items():
        now = {o["obj_id"]: o for o in read_json(sp_work(cfg) / sid / "objects.json")}
        for oid, (lab, name, merged, ref) in objs.items():
            o = now[int(oid)]
            if ref and not o["referable"]:
                lost.append((sid, int(oid), lab, name, o["category_merged"], o["drop_reasons"]))
    return lost


def report(cfg):
    a = pd.read_csv(sp_work(cfg) / "_full" / "audit_objects.csv")
    j = pd.read_csv(path_of(cfg, "reports") / "scannetpp_object_audit_judgments.tsv", sep="\t", keep_default_na=False)
    df = a.merge(j, on="key", how="left")
    assert df.recognizable.notna().all() and (df.recognizable != "").all(), "some tiles have no judgment"
    ok = (df.recognizable == "yes") & (df.name_match == "yes")
    fail, unsure = df[~ok], df[ok & (df.unsure == "yes")]
    tab = lambda d: ["| Tile | Scene | obj_id | Name (source label) | (1) Recognizable | (2) Name matches | Questions | Note |", "|---|---|---|---|---|---|---|---|",
                     *[f"| {r.key} | `{r.scene_id}` | {r.obj_id} | {r.name} ({r.source_label}) | {r.recognizable} | {r.name_match} | {r.n_questions} | {r.note} |"
                       for r in d.itertuples()]]
    L = ["# ScanNet++ audit of all question objects", "",
         "> Branch `scannetpp`. A full census of all question objects, replacing further re-seeded spot checks. Generated by `python -m actspace_build.scannetpp_audit tiles|report`. "
         "Scope: all referable objects used as slot, option or evidence objects in `work/scannetpp/_step6/test_groups.jsonl` (rerun after the look-alike merge rule).", "",
         f"- One tile per object (full frame at width 480 + green box + name / scene / obj_id): the frame where the object is visible with the largest projected area among the oracle / link views of the questions that use it; "
         f"otherwise the best frame among all nodes ({int((df.frame_source != 'oracle/link').sum())} objects). Sorted by scene, 16 per page, "
         f"`reports/scannetpp_object_audit/page_XX.jpg` ({df.page.max()} pages, not tracked in git); tile key P<page>-<cell>.",
         "- Each tile is judged on two items: (1) can the object be recognised and roughly located; (2) does the name match the object. Borderline cases count as \"no\". **No objects are blacklisted in this step**; decisions are made after all pages have been reviewed.", "",
         f"**Result: {len(df)} objects, {int(ok.sum())} pass both items ({ok.mean():.1%}); {len(fail)} fail; {len(unsure)} pass but uncertain.** "
         f"((1) recognizable {int((df.recognizable == 'yes').sum())} / {len(df)}; (2) name matches {int((df.name_match == 'yes').sum())} / {len(df)}.)", "",
         "## 1. Failed", "", *tab(fail), "", "## 2. Passed but uncertain (review these first)", "", *(tab(unsure) if len(unsure) else ["None."]), ""]
    review = sp_work(cfg) / "_full" / "audit_review.md"
    if review.exists():
        L += [review.read_text().rstrip("\n"), ""]
    lost = s29_changes(cfg)
    by = defaultdict(list)
    for x in lost:
        by[x[0]].append(x)
    L += ["## Appendix: objects that became non-referable after the look-alike merge rule", "",
          f"Compared with the previous Step 3 result, {len(lost)} objects became non-referable (including small_sofa and blacklist changes):", "",
          "| Scene | obj_id | Source label | Previous name | Current coarse category | Reasons |", "|---|---|---|---|---|---|",
          *[f"| `{s}` | {o} | {lab} | {name} | {m} | {', '.join(r)} |" for s in sorted(by) for (_, o, lab, name, m, r) in by[s]], ""]
    (path_of(cfg, "reports") / "scannetpp_object_audit.md").write_text("\n".join(L) + "\n")
    print("wrote scannetpp_object_audit.md", len(df), int(ok.sum()), len(fail), len(unsure))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tiles", "report"])
    args = ap.parse_args()
    cfg = load_config()
    {"tiles": tiles, "report": report}[args.cmd](cfg)


if __name__ == "__main__":
    main()
