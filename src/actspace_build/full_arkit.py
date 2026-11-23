"""Full ARKitScenes build (run after GATE C1 passed): batch driver and summary report for Steps 1-6.

Called by scripts/run_arkit_full.sbatch on a CPU compute node, not on the login node. Subcommands:
  gate_a        GATE A (pose convention, per-pose check, orientation) for all deduplicated selected videos
  objects       Step 3 for the scenes that passed GATE A
  download      download vga_wide + vga_wide_intrinsics for the scenes that passed Step 3 (no cap)
  scenes        Step 1 frame table, Step 2 and Step 4 for each scene (process pool)
  check_graph   stop condition: if the fraction of scenes dropped for GRAPH_DISCONNECTED exceeds the threshold,
                write a stop report and exit non-zero
  questions     Steps 5-6 (balanced across scenes, single process)
  summary       summary report reports/arkit_full_summary.md and overview sheets with one card per scene
Scene lists and per-stage results are written to work/arkitscenes/_full/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, path_of, read_json, work_dir, write_json

DATASET = "arkitscenes"


def full_dir(cfg) -> Path:
    d = path_of(cfg, "work") / DATASET / "_full"
    d.mkdir(parents=True, exist_ok=True)
    return d


def all_videos(cfg) -> list[str]:
    sel = pd.read_csv(path_of(cfg, "work") / DATASET / "_step0" / "selected_videos.csv", dtype={"video_id": str})
    return sorted(sel.video_id)


def write_list(cfg, name, vids):
    (full_dir(cfg) / f"{name}.txt").write_text("\n".join(vids) + "\n")
    print(f"{name}: {len(vids)} scenes")


def read_list(cfg, name) -> list[str]:
    return [l for l in (full_dir(cfg) / f"{name}.txt").read_text().split() if l]


# ---------------------------------------------------------------- stages

def stage_gate_a(cfg, workers):
    from multiprocessing import Pool
    from .step1_arkit import gate_a_scene, plot_pose_timeline
    vids = all_videos(cfg)
    with Pool(workers) as pool:
        res = pool.map(gate_a_scene, [(v, False) for v in vids])
    out_dir = path_of(cfg, "reports") / "gateA" / "pose_timeline"
    for r in res:
        if not (out_dir / f"{r['video_id']}.png").exists():
            plot_pose_timeline(r["video_id"], out_dir / f"{r['video_id']}.png")
    ok = [r["video_id"] for r in res if r["pose_valid"] and r["orientation_ok"]]
    write_list(cfg, "after_gate_a", ok)


def _objects_one(vid, force=False):
    from .step3_objects import objects_scene
    objects_scene(vid, force=force)
    return vid, read_json(work_dir(load_config(), DATASET, vid) / "step3_summary.json")["scene_ok"]


def stage_objects(cfg, workers, force=False):
    vids = read_list(cfg, "after_gate_a")
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_objects_one, vids, [force] * len(vids)))
    write_list(cfg, "after_step3", [v for v, ok in res if ok])


def vga_complete(cfg, vid) -> bool:
    d = path_of(cfg, "arkit_raw_root") / vid
    a, b = d / "vga_wide", d / "vga_wide_intrinsics"
    return a.is_dir() and b.is_dir() and len(list(a.glob("*.png"))) > 0 and \
        len(list(a.glob("*.png"))) == len(list(b.glob("*.pincam")))


def _download_one(cfg, vid):
    if vga_complete(cfg, vid):
        return vid, True, "exists"
    repo = Path(cfg["paths"]["arkit_repo"])
    log = full_dir(cfg) / "download_logs" / f"{vid}.log"
    log.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "download_data.py", "raw", "--split", cfg["paths"]["arkit_split_name"], "--video_id", vid,
           "--download_dir", str(repo / "data"), "--raw_dataset_assets", *cfg["arkitscenes"]["vga_assets"]]
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)
    return vid, rc == 0 and vga_complete(cfg, vid), f"rc={rc}"


def stage_download(cfg, workers):
    vids = read_list(cfg, "after_step3")
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda v: _download_one(cfg, v), vids))
    failed = [(v, why) for v, ok, why in res if not ok]
    write_json(full_dir(cfg) / "download_failed.json", failed)
    write_list(cfg, "after_download", [v for v, ok, _ in res if ok])


def _scene_one(vid):
    """Step 1 frame table + Step 2 + Step 4 for one scene. If a step fails, record the exception and skip the remaining steps."""
    from .step1_arkit import build_frames, visualize_frames
    from .step2_keyframes import keyframes_scene
    from .step4_graph_visibility import graph_visibility_scene
    cfg = load_config()
    try:
        meta = build_frames(vid, cfg["arkitscenes"]["display_frames"])
        vis_dir = path_of(cfg, "reports") / "gateA" / "vga_wide"
        if not list(vis_dir.glob(f"{vid}_vga_wide_*.jpg")):
            visualize_frames(vid, "vga_wide", cfg["gate_a"]["vis_frames_per_scene"], vis_dir)
        if meta["status"].get("vga_image_orientation_ok") is False:
            return vid, "vga_orientation_conflict"
        data_root = path_of(cfg, "arkit_3dod_root").resolve().parents[1]
        kf = keyframes_scene(DATASET, vid, data_root)
        if not kf["scene_ok"]:
            return vid, "too_few_keyframes"
        s4 = graph_visibility_scene(DATASET, vid)
        return vid, "ok" if s4["ok"] else "graph_disconnected"
    except Exception as e:  # noqa: BLE001  record and continue with the other scenes
        return vid, f"error: {type(e).__name__}: {e}"


def stage_scenes(cfg, workers):
    vids = read_list(cfg, "after_download")
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_scene_one, vids))
    write_json(full_dir(cfg) / "scene_status.json", res)
    print(Counter(v if not v.startswith("error") else "error" for v in res.values()))
    write_list(cfg, "after_step4", [v for v, s in res.items() if s == "ok"])


def stage_check_graph(cfg):
    st = read_json(full_dir(cfg) / "scene_status.json")
    reached = [v for v, s in st.items() if s in ("ok", "graph_disconnected")]
    bad = [v for v, s in st.items() if s == "graph_disconnected"]
    frac = len(bad) / max(len(reached), 1)
    thr = cfg["view_graph"]["stop_if_disconnected_frac"]
    print(f"graph_disconnected {len(bad)}/{len(reached)} = {frac:.1%} (stop if > {thr:.0%})")
    if frac > thr:
        msg = f"STOP: {len(bad)}/{len(reached)} scenes ({frac:.0%}) dropped for GRAPH_DISCONNECTED, more than {thr:.0%}: {bad}"
        (path_of(cfg, "reports") / "STOP_graph_disconnected.md").write_text(msg + "\n")
        raise SystemExit(msg)


# ---------------------------------------------------------------- summary report

def _drop_counts(cfg, vids):
    c = Counter()
    for vid in vids:
        p = work_dir(cfg, DATASET, vid) / "dropped.jsonl"
        if p.exists():
            for l in open(p):
                d = json.loads(l)
                c[(d["step"], d["reason_code"])] += 1
    return c


def overview_sheets(cfg, groups, per_sheet=12, cols=3, tile_w=860):
    """For each scene with question groups, the card of one random group, downscaled and tiled into overview sheets (per_sheet cards per page)."""
    import cv2
    from .common import scene_rng
    from .reports import _card_image
    from .step56_questions import Scene
    out = path_of(cfg, "reports") / "arkit_full_overview"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    by_scene = {}
    for grp in groups:
        by_scene.setdefault(grp["scene_id"], []).append(grp)
    tiles = []
    tmp = full_dir(cfg) / "card_tmp.jpg"
    for sid in sorted(by_scene):
        vid = sid.split("_", 1)[1]
        rng = scene_rng(cfg, sid, "full_overview_card")
        grp = by_scene[sid][int(rng.integers(len(by_scene[sid])))]
        _card_image(cfg, Scene(cfg, DATASET, vid), grp, tmp)
        img = cv2.imread(str(tmp))
        img = cv2.resize(img, None, fx=tile_w / img.shape[1], fy=tile_w / img.shape[1], interpolation=cv2.INTER_AREA)
        tiles.append(img)
    pages = []
    for k in range(0, len(tiles), per_sheet):
        chunk = tiles[k:k + per_sheet]
        rows = []
        for r in range(0, len(chunk), cols):
            row = chunk[r:r + cols]
            h = max(t.shape[0] for t in row)
            row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in row]
            while len(row) < cols:
                row.append(np.full_like(row[0], 255))
            rows.append(np.concatenate(row, axis=1))
        sheet = np.concatenate([cv2.copyMakeBorder(r, 0, 6, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows], axis=0)
        name = f"overview_{k // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        pages.append(name)
    tmp.unlink(missing_ok=True)
    return pages


def _notes(fd):
    """Manually written findings and open items (_full/summary_notes.md), kept when the report is regenerated."""
    p = fd / "summary_notes.md"
    return [p.read_text(), ""] if p.exists() else []


def _comparison_section(fd):
    """Round-by-round comparison of the question-generation rule changes."""
    p = fd / "q20_comparison.json"
    if not p.exists():
        return []
    comp = json.loads(p.read_text())
    L = ["### Round-by-round comparison", "", "| Round | Rule | Scenes with questions | Groups | Questions | unanswerable | Templates |", "|---|---|---|---|---|---|---|"]
    for k, v in comp.items():
        L.append(f"| {k} | {v['label']} | {v['scenes_with_groups']} | {v['groups']} | {v['questions']} | "
                 f"{v['unanswerable']} ({v['unanswerable'] / v['questions']:.1%}) | "
                 + ", ".join(f"{t} {n}" for t, n in sorted(v["per_template"].items())) + " |")
    pre = fd / "pre_dedup_stats.json"
    if pre.exists():
        d = json.loads(pre.read_text())
        L += ["", "### Before and after within-scene deduplication", "",
              f"Before deduplication (second-round data): {d['groups']} groups, {d['questions']} questions; {d['extra_identical_text']} redundant questions "
              f"with identical text within a scene; {d['extra_by_key_total']} redundant by content key (T1 unordered object pair, T2 reference + direction "
              f"+ option set, T3 triple, T5 standing + facing, T4 same object > 2 times): "
              + ", ".join(f"{k} {v}" for k, v in sorted(d["extra_by_key"].items())) + ".",
              "The numbers after deduplication are given at the start of this section; deduplication happens during question generation "
              "(a per-scene set of used questions), not by deleting questions afterwards."]
    return L


def stage_summary(cfg):
    fd = full_dir(cfg)
    vids_all = all_videos(cfg)
    lists = {n: read_list(cfg, n) for n in ("after_gate_a", "after_step3", "after_download", "after_step4")}
    status = read_json(fd / "scene_status.json")
    groups = [json.loads(l) for l in open(path_of(cfg, "work") / DATASET / "_step6" / "test_groups.jsonl")]
    st = read_json(path_of(cfg, "work") / DATASET / "_step6" / "step6_stats.json")
    qs = [q for g in groups for q in g["questions"]]
    scenes_with_groups = sorted({g["scene_id"] for g in groups})
    gate = {v: read_json(work_dir(cfg, DATASET, v) / "gate_a.json") for v in vids_all}
    n_pose_bad = sum(not r["pose_valid"] for r in gate.values())
    n_orient = sum(not r["orientation_ok"] for r in gate.values())
    dl_failed = read_json(fd / "download_failed.json")
    sizes = sum(f.stat().st_size for v in lists["after_download"] for f in (path_of(cfg, "arkit_raw_root") / v).rglob("*") if f.is_file())
    drops = _drop_counts(cfg, lists["after_gate_a"])
    c = lambda key: Counter(key(q) for q in qs)
    fmt = lambda d: ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0])))
    ua = sum(q["evidence_label"] == "unanswerable" for q in qs)
    per_scene_q = Counter(g["scene_id"] for g in groups for _ in g["questions"])
    pages = overview_sheets(cfg, groups)
    scene_status = Counter(v if not v.startswith("error") else "error" for v in status.values())
    L = ["# ARKitScenes full build summary (Steps 1-6)", "",
         "> Run after GATE C1 passed; computed on CPU. Step 7 onwards follows once ScanNet++ passes GATE C2.", "",
         "## 1. Scale (scene funnel)", "",
         "| Stage | Scenes | Notes |", "|---|---|---|",
         f"| Deduplicated selected videos | {len(vids_all)} | Step 0 |",
         f"| Passed GATE A | {len(lists['after_gate_a'])} | POSE_INVALID {n_pose_bad}, ORIENTATION_CONFLICT {n_orient} (incl. unconfirmed / agreement too low) |",
         f"| Referable objects >= {cfg['objects']['min_referable_objects_per_scene']} (Step 3) | {len(lists['after_step3'])} | rest TOO_FEW_OBJECTS |",
         f"| vga_wide download complete | {len(lists['after_download'])} | {len(dl_failed)} downloads failed; {sizes / 1e9:.1f} GB downloaded |",
         f"| Passed Step 1 frame table / Step 2 / Step 4 | {len(lists['after_step4'])} | " + fmt(scene_status) + " |",
         f"| Scenes with question groups | {len(scenes_with_groups)} | |", "",
         f"**{st['num_groups']}** question groups, **{st['num_questions']}** questions; median questions per scene with questions: {int(np.median(list(per_scene_q.values()))) if per_scene_q else 0}.", "",
         *_comparison_section(fd), "",
         "## 2. Question distribution", "",
         f"- Templates: {fmt(st['per_template'])}",
         f"- Variants: {fmt(st['per_variant'])}; two-option comparison questions: {sum(q.get('variant') == 'pair' for q in qs) / max(len(qs), 1):.1%} (cap {cfg['questions']['pair_max_frac']:.0%})",
         f"- Evidence labels: {fmt(st['evidence_label'])}",
         f"- **unanswerable fraction: {ua} / {len(qs)} = {ua / max(len(qs), 1):.1%}** (target {cfg['questions']['target_unanswerable_frac'][0]:.0%}-{cfg['questions']['target_unanswerable_frac'][1]:.0%}; all from the masking method, {sum(1 for q in qs if q.get('replaced_template'))} of them replacement questions)",
         f"- init fraction: {sum(q['evidence_label'] == 'init' for q in qs) / max(len(qs), 1):.1%} (cap {cfg['questions']['target_init_frac_max']:.0%})",
         f"- Difficulty (explore questions by oracle_steps tertiles, cut points {[round(x, 2) for x in st['difficulty_cuts_oracle_steps']]}): {fmt(st['difficulty'])}",
         f"- Constraints: {fmt(st['constraints'])}",
         f"- T3 quadrants: {fmt(st['quadrant_per_template']['T3'])}; T4 quadrants: {fmt(st['quadrant_per_template']['T4'])}",
         "- Correct option letters: " + "; ".join(f"{t} {fmt(d)}" for t, d in st["answer_letter_per_template"].items() if d),
         f"- oracle_views: {fmt(st['oracle_views'])}", "",
         "## 3. Drop reasons (dropped.jsonl, scenes that passed GATE A, by step)", "",
         "| Step | reason_code | Count |", "|---|---|---|",
         *[f"| {s} | {r} | {n} |" for (s, r), n in sorted(drops.items())], "",
         "Scene-level status (Step 1 frame table / 2 / 4): " + fmt(scene_status) + ".",
         *([f"Scenes with errors: {', '.join(f'{v} ({s})' for v, s in status.items() if s.startswith('error'))}"] if scene_status.get("error") else []),
         *([f"Failed downloads: {dl_failed}"] if dl_failed else []), "",
         *_notes(fd),
         "## 4. Overview sheets (one card per scene)", "",
         f"{len(scenes_with_groups)} scenes; for each scene one question group is drawn at random with the scene-level seed, 12 cards per page:", "",
         *[f"![](arkit_full_overview/{p})" for p in pages]]
    (path_of(cfg, "reports") / "arkit_full_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate_a", "objects", "download", "scenes", "check_graph", "questions", "summary"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="objects: ignore completion markers and rerun Step 3")
    args = ap.parse_args()
    cfg = load_config()
    if args.stage == "gate_a":
        stage_gate_a(cfg, args.workers)
    elif args.stage == "objects":
        stage_objects(cfg, args.workers, args.force)
    elif args.stage == "download":
        stage_download(cfg, args.workers)
    elif args.stage == "scenes":
        stage_scenes(cfg, args.workers)
    elif args.stage == "check_graph":
        stage_check_graph(cfg)
    elif args.stage == "questions":
        # After a Step 3 rerun (blacklist / excluded scenes), scenes in after_step4 that no longer pass Step 3 get no questions
        ok3 = set(read_list(cfg, "after_step3"))
        vids = [v for v in read_list(cfg, "after_step4") if v in ok3]
        sys.argv = ["step56_questions", "--dataset", DATASET, "--scenes", *vids]
        from .step56_questions import main as q_main
        q_main()
    else:
        stage_summary(cfg)


if __name__ == "__main__":
    main()
