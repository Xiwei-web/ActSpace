"""Section 5.7: replacing the S2 and S3 success cases. Reads only existing logs, the frozen data and the raw datasets; no model is called.

One job in two stages:
  A. Write round-2 previews: all S2 candidates; for S3 the top 6 after re-ranking by "common category with a distinctive appearance" and
     "target complete, centred and large in the evidence frame";
     also compute the per-model S1 funnel. Written to success_candidates/round2/; then wait for round2/choice.json ({"S2": group id, "S3": group id}).
  B. Move the old S2 and S3 exports and archives to success_assets/_superseded/ (not deleted), export the two newly chosen cases, add a note to
     the S4 README,
     repackage S2, S3, S4, rewrite SHA256SUMS and build a comparison contact sheet.
"""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from actspace_build import geometry as G
from actspace_build.common import read_json

from .data import FrozenData
from .sec57_case_assets import Case, dir_listing, export_all, jdump
from .sec57_success_assets import (CAND_OUT, CLEAR_INFRAME, ORDER, S1_MIN_SHORTEST, S1_RATIO, STRONG, SUCC_OUT, TYPES, cand, candidates, export_raw, load_eps,
                                   package, preview, vis_of)

R2 = CAND_OUT / "round2"
COMMON = ("bed", "sofa", "couch", "tv", "television", "toilet", "sink", "bathtub", "refrigerator", "oven", "stove", "washing machine", "fireplace", "piano")
N_S3 = 6
S4_NOTE = ("\n## Comparison note\n\nThe focus question of the comparison is the question C1 failed on, whose evidence is strictly visible; the evidence of the fireplace question is not strictly visible, so it is not emphasised in figures.\n")


@lru_cache(maxsize=None)
def frames_of(wd: Path):
    return {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}


def frame_score(e, o: int):
    """Best frame among strictly visible frames seen by the model where the target is "complete, centred and large": (area fraction × centredness, step); requires at least CLEAR_INFRAME of the object inside the image."""
    row, col, st, area, infr = vis_of(e.scene.wd)
    fr = frames_of(e.scene.wd)
    ob = next(x for x in read_json(e.scene.wd / "objects.json") if int(x["obj_id"]) == o)
    best = (0.0, None)
    for t, n in e.seen:
        if st[row[o], col[n]] != 2 or infr[row[o], col[n]] < CLEAR_INFRAME:
            continue
        f = fr[n]
        uv, z = G.project(np.asarray([ob["obb"]["center"]], float), np.asarray(f["K"], float), np.asarray(f["T_wc"], float))
        W, H = f["image_size"]
        cen = max(0.0, 1.0 - 2.0 * max(abs(uv[0, 0] / W - 0.5), abs(uv[0, 1] / H - 0.5))) if z[0] > 0 else 0.0
        sc = float(area[row[o], col[n]]) * cen
        if sc > best[0]:
            best = (sc, t)
    return best


def s3_rerank(data, eps) -> list:
    names = {}
    best = {}
    for (m, run), d in eps.items():
        if m not in STRONG:
            continue
        for e in d.values():
            if e.sid not in names:
                names[e.sid] = {int(x["obj_id"]): (x.get("ref_name") or x.get("category") or "") for x in read_json(e.scene.wd / "objects.json")}
            ans_ok = all(q["ok"] and q["seen_all"] for q in e.answerable)
            rest = all(x["ok"] for x in e.qs) and ans_ok
            for q in e.answerable:
                if not (q["template"] == "T4" and q["ok"] and q["seen_all"] and all(v == 0 for v in q["start_state"].values()) and (q["evidence_step"] or 0) >= 1):
                    continue
                o = q["evidence_objs"][0]
                nm = names[e.sid][o].lower()
                sc, t = frame_score(e, o)
                if t is None:
                    continue
                common = any(nm == c or nm.endswith(" " + c) for c in COMMON)
                key = (-common, -rest, -round(sc, 4), ORDER.index(m))
                c = cand(e, "S3", key, q, target_quadrant=q["answer_value"], rest_ok=rest, target_name=nm, target_obj=o, common_category=common,
                         frame_score=round(sc, 4), best_frame_step=t)
                if e.gid not in best or key < best[e.gid]["key"]:
                    best[e.gid] = c
    return sorted(best.values(), key=lambda c: (c["key"], c["group_id"]))


def funnel(eps) -> dict:
    out = {}
    for m in STRONG:
        f = collections.Counter()
        for (mm, run), d in eps.items():
            if mm != m:
                continue
            for e in d.values():
                if e.unans or not e.answerable:
                    continue
                f["unmasked_episodes"] += 1
                if not all(q["ok"] for q in e.answerable):
                    continue
                f["all_correct"] += 1
                if not all(q["seen_all"] for q in e.answerable):
                    continue
                f["all_evidence_strictly_visible"] += 1
                if e.shortest >= S1_MIN_SHORTEST and e.steps_used <= S1_RATIO * e.shortest:
                    f["steps_within_1.5x_shortest"] += 1
        out[m] = {k: f[k] for k in ("unmasked_episodes", "all_correct", "all_evidence_strictly_visible", "steps_within_1.5x_shortest")}
    out["total"] = {k: sum(out[m][k] for m in STRONG) for k in out[STRONG[0]]}
    return out


def stack(paths: list[Path], labels: list[str], out: Path):
    ims = []
    for p, lab in zip(paths, labels):
        im = cv2.imread(str(p))
        bar = np.full((40, im.shape[1], 3), 235, np.uint8)
        cv2.putText(bar, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 160), 2, cv2.LINE_AA)
        ims.append(np.vstack([bar, im]))
    W = max(i.shape[1] for i in ims)
    ims = [np.hstack([i, np.full((i.shape[0], W - i.shape[1], 3), 255, np.uint8)]) for i in ims]
    cv2.imwrite(str(out), np.vstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    data = FrozenData()
    eps = load_eps(data)
    pool = candidates(data, eps)
    old = json.load(open(CAND_OUT / "candidates.json"))["selected"]
    if R2.exists() or (SUCC_OUT / "_superseded").exists():
        print("round2/ or _superseded/ already exists; refusing to overwrite")
        sys.exit(2)
    R2.mkdir()
    s2 = pool["S2"]
    s3 = s3_rerank(data, eps)
    lists = {"S2": s2, "S3": s3[:N_S3]}
    spec = lambda c: {"model": c["model"], "run": c["run"], "group_id": c["group_id"], "focus_qid": c["focus_qid"], "situation": TYPES[c["cat"]]}     # noqa: E731
    for k, L in lists.items():
        for r, c in enumerate(L, 1):
            c["rank"] = r
            cs = Case(data, k, f"{k}_{TYPES[k]}_{c['group_id']}", spec(c))
            c["preview"] = str(R2 / f"{k}_r2rank{r}_{c['group_id']}_{c['model']}.jpg")
            preview(cs, c, Path(c["preview"]), first=[(c["best_frame_step"], [c["target_obj"]])] if k == "S3" else None)
            print(f"   {k} round 2 #{r} {c['group_id']} {c['model']} steps {c['steps_used']} / shortest {c['shortest_steps']} visited {c['n_nodes_visited']} viewpoints key {c['key']}"
                  + (f" target {c['target_name']} at step {c['best_frame_step']}" if k == "S3" else ""), flush=True)
    jdump(R2 / "round2.json", {"common_categories": COMMON, "s3_pool_size": len(s3), "s3_common_in_pool": sum(c["common_category"] for c in s3),
                               "lists": lists, "s1_funnel_by_model": funnel(eps)})
    print(f"== round-2 previews written to {R2}; waiting for choice.json", flush=True)
    t0 = time.time()
    while not (R2 / "choice.json").exists():
        if time.time() - t0 > 5 * 3600:
            print("no choice.json after 5 hours; exiting")
            sys.exit(4)
        time.sleep(20)
    time.sleep(3)
    choice = json.load(open(R2 / "choice.json"))
    new = {k: next(c for c in lists[k] if c["group_id"] == choice[k]) for k in ("S2", "S3")}
    # move old exports to _superseded/
    sup = SUCC_OUT / "_superseded"
    (sup / "_packages").mkdir(parents=True)
    summary = json.load(open(SUCC_OUT / "success_assets_summary.json"))
    shutil.copy2(SUCC_OUT / "success_assets_summary.json", sup / "success_assets_summary_round1.json")
    shutil.copy2(SUCC_OUT / "_packages" / "SHA256SUMS", sup / "_packages" / "SHA256SUMS_round1")
    for k in ("S2", "S3"):
        on = Path(summary["success"][k]["dir"]).name
        shutil.move(str(SUCC_OUT / on), str(sup / on))
        shutil.move(str(SUCC_OUT / "_packages" / f"{on}.tar.gz"), str(sup / "_packages" / f"{on}.tar.gz"))
        print(f"== old {k} ({on}) moved to {sup}", flush=True)
    names = {}
    for k, c in new.items():
        name = f"{k}_{TYPES[k]}_{c['group_id']}"
        cs = Case(data, k, name, spec(c))
        d = SUCC_OUT / name
        d.mkdir()
        info = export_all(cs, d)
        info["raw"] = export_raw(cs, d)
        info.update(dir_listing(d))
        info["candidate"] = {x: c[x] for x in c if x not in ("questions", "notes")}
        summary["success"][k] = info
        names[k] = name
        print(f"== new {k}: {name}, {info['n_files']} files, {info['total_bytes'] / (1 << 20):.1f} MB; raw missing {info['raw']['missing']}", flush=True)
    s4 = Path(summary["success"]["S4"]["dir"])
    with open(s4 / "README.md", "a") as f:
        f.write(S4_NOTE)
    summary["success"]["S4"].update(dir_listing(s4))
    s1 = Path(summary["success"]["S1"]["dir"]).name
    pk = package(SUCC_OUT, [names["S2"], names["S3"], s4.name])
    keep = [p for p in summary["packages"]["success"] if Path(p["file"]).name == f"{s1}.tar.gz"]
    h = hashlib.sha256((SUCC_OUT / "_packages" / f"{s1}.tar.gz").read_bytes()).hexdigest()
    assert h == keep[0]["sha256"]
    allp = sorted(keep + pk, key=lambda p: p["file"])
    (SUCC_OUT / "_packages" / "SHA256SUMS").write_text("".join(f"{o['sha256']}  {Path(o['file']).name}\n" for o in allp))
    summary["packages"]["success"] = allp
    summary["round2"] = {"choice": choice, "superseded_dir": str(sup)}
    jdump(SUCC_OUT / "success_assets_summary.json", summary)
    # comparison contact sheet: the superseded top candidate next to the newly chosen one
    paths, labels = [], []
    for k in ("S2", "S3"):
        o = old[k][0]
        paths += [Path(o["preview"]), Path(new[k]["preview"])]
        labels += [f"{k} OLD #1 (superseded): {o['group_id']} {o['model']}", f"{k} NEW: {new[k]['group_id']} {new[k]['model']}"]
    stack(paths, labels, CAND_OUT / "compare_S2_S3_old_vs_new.jpg")
    print("== all done", flush=True)


if __name__ == "__main__":
    main()
