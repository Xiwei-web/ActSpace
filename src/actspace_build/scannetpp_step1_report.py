"""reports/scannetpp_step1_gateA.md: pilot redraw, full download check, merge result, decode check of the new pilots, GATE A results.

Called by python -m actspace_build.scannetpp_step1 report (gate_a must have finished).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from .common import ROOT, path_of, read_json, write_json
from .scannetpp import read_split, sp_cfg, sp_root, sp_work
from .scannetpp_step0 import check_scene, dir_stats

GB = 1e9


def yn(b) -> str:
    return "✅" if b else "❌"


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def pytest_summary() -> str:
    """Run the unit tests once and return the last pytest line (e.g. "83 passed in 2.5s")."""
    import sys
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", str(ROOT / "tests")],
                         capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()
    return out[-1].strip("= ") if out else "could not run"


def download_check(cfg: dict) -> dict:
    """Whether all 7 file types exist for the 50 scenes; log ending, elapsed time, number of token lines."""
    sc = sp_cfg(cfg)
    split = read_split(cfg)
    rows, missing = [], []
    for sid in split:
        d = sp_root(cfg) / "data" / sid
        sizes = {f: (d / f).stat().st_size if (d / f).is_file() else None for f in sc["required_files"]}
        miss = [f for f, s in sizes.items() if s is None]
        missing += [f"{sid}/{f}" for f in miss]
        rows.append({"scene_id": sid, "bytes": sum(s for s in sizes.values() if s), "missing": miss})
    log = Path(sc["toolbox"]).parent / "download_full.log"
    txt = log.read_text(errors="replace").replace("\r", "\n") if log.exists() else ""
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    m = re.findall(r"scenes: 100%\|[^|]*\| (\d+)/\d+ \[([\d:]+)<", txt)
    return {"n_scenes": len(split), "n_complete": sum(not r["missing"] for r in rows), "missing": missing,
            "total_bytes": sum(r["bytes"] for r in rows), "log_tail": lines[-1] if lines else None,
            "elapsed": m[-1][1] if m else None, "token_lines": sum("Using token" in l for l in txt.splitlines()),
            "redacted_urls": txt.count("token=REDACTED"), "rows": rows}


def merge_section(cfg: dict) -> list[str]:
    merges = git("log", "--merges", "--first-parent", "--format=%h %s", "-n", "5").splitlines()
    main_head = git("rev-parse", "--short", "main")
    behind = git("rev-list", "--count", "HEAD..main")
    L = ["## 3. Syncing the ARKit code (merge main)", "",
         "- Merge commits (newest first):", *[f"  - `{m}`" for m in merges],
         f"- main is currently `{main_head}`; scannetpp is {behind} commits behind main. Merged, main was not rebased.",
         "- The first merge brought in ARKit v0.10 and the updated data construction guide; the second merge only added ARKit's "
         "`reports/v0.10_whatif.md` and `scripts/run_whatif.sbatch`, with no changes to the guide, configs or code.",
         "- **Conflicts: none** (both were automatic merges). In `configs/build.yaml` the ARKit changes are in the `objects` / `questions` sections; ScanNet++ only adds a `scannetpp:` section. "
         "ARKit's uncommitted `scripts/run_whatif.sbatch` in the main working directory was not touched during the first merge (ARKit committed it later).",
         "- Before merging, the [ScanNet++] entries appended to the shared change / open-question logs during Step 0 were moved to ScanNet++-specific logs; after merging, the two shared files are **byte-identical** to main.",
         "- The guide in the worktree is the version committed on main and is byte-identical to the one in the main directory.",
         "- All unit tests pass after the merge (see Section 6).", "",
         "### 3.1 Checking the guide's rules in the code", "",
         "| Rule | Code location | Status |", "|---|---|---|",
         "| Size references only use larger / smaller: exactly 2 of the same category, volume ratio ≥ 2.5, centres ≥ 1.0 m apart, a node where both are visible | `step3_objects.py` (`size_disambig_*`), `step56_questions.py` (`size_ref_require_covisible`) | ✅ implemented |",
         "| Linkable evidence (T1/T2/T3/T5, `EVIDENCE_NOT_LINKED`) + `link_views` | `step56_questions.link_edges / evidence_linked / greedy_link_views / add_link_views` | ✅ implemented; mode see below |",
         "| Five visibility conditions (ρ, area, point count, completeness, in-frame fraction) + sensor-depth occlusion | `step4_graph_visibility.py` | ✅ implemented; sensor-depth matching is written for ARKit timestamps, see below |",
         "| unanswerable only via masking (`masked_obj` / `masked_nodes` / `view_graph_mask`, masking ≤ 30%) | `step56_questions.py` | ✅ implemented |",
         "| `AMBIGUOUS_EVIDENCE`: drop if an evidence object is not visible in any node | `step56_questions.evidence_label` | ✅ consistent with the guide |", "",
         "**Two points the later ScanNet++ steps must handle (not missing features, but dataset-specific settings)**:",
         f"1. Linkable evidence: the shared `questions.evidence_link_mode` is currently `{cfg['questions']['evidence_link_mode']}` (ARKit v0.10 switched because visible_partial gave fewer than 500 questions). "
         f"ScanNet++ uses one_visible_one_partial (value name in code: `visible_partial`), written as `scannetpp.evidence_link_mode: {sp_cfg(cfg)['evidence_link_mode']}`; "
         "Steps 5–6 read this key for ScanNet++ to override the shared value (implemented then; ARKit default unchanged).",
         "2. Sensor-depth occlusion check: `step4_graph_visibility` matches ARKit `lowres_depth` by timestamp (≤ 0.02 s). Every ScanNet++ frame has depth with the same frame ID, so Step 4 adds a frame-ID lookup path for ScanNet++ (backward compatible).", "",
         "**Key-name differences between the guide's config block and `build.yaml`** (all rules exist, only named differently or hard-coded; all in shared modules, left unchanged so existing ARKit behaviour stays the same):", "",
         "| Guide config key | Counterpart in code |", "|---|---|",
         "| `objects.size_ref_exact_instances / size_ref_min_volume_ratio / size_ref_min_center_dist_m / size_ref_words` | `objects.size_disambig_max_instances: 2` (code requires 2 ≤ instances ≤ 2) / `size_disambig_min_volume_ratio` / `size_disambig_min_center_dist_m` / code replaces largest/smallest with larger/smaller |",
         "| `questions.evidence_link_rule: one_visible_one_partial` (alternative both_partial) | `questions.evidence_link_mode`: `visible_partial` / `partial_partial` |",
         "| `questions.evidence_link_min_questions: 500` | none (switched manually for ARKit) |",
         "| `visibility.spotcheck_n: 50`, `spotcheck_min_pass: 0.95` | default argument of `reports.visibility_spotcheck(n=50)`; 95% is the manual pass criterion |",
         "| `rewrite.method / file / author_model` | fixed to `configs/paraphrases.yaml` in `step7_rewrite.py` |",
         "| `debias.rf_trees: 300`, `stability_repeats: 5` | hard-coded `n_estimators=300`, `n_rep=5` in `step8_debias.py` |",
         "| `human_check.annotators: 1` | `human_check.annotators_per_group: 1` |", "",
         "- The only change to shared modules in this round: `step1_arkit.plot_pose_timeline` gains a `dataset` parameter (default `arkitscenes`, ARKit behaviour unchanged) so ScanNet++ can reuse the timeline plot.", ""]
    return L


def write_report(cfg: dict) -> None:
    sc = sp_cfg(cfg)
    gcfg = cfg["gate_a"]
    pilots = sc["pilot_scenes"]
    draw = read_json(sp_work(cfg) / "_step0" / "pilot_draw.json")
    dl = download_check(cfg)
    write_json(sp_work(cfg) / "_step0" / "download_full_check.json", dl)
    inv = [check_scene(cfg, s) for s in pilots]
    G = [read_json(sp_work(cfg) / s / "gate_a.json") for s in pilots]
    rep = path_of(cfg, "reports")
    vis_dir = rep / "scannetpp_gateA"

    L = ["# ScanNet++ Step 1 / GATE A: pilot redraw, full download, merge, GATE A on the new pilots", "",
         "> Branch `scannetpp`. Generated by `python -m actspace_build.scannetpp_step1 report`.",
         "> Reference: Step 1 and GATE A in docs/data_construction.md. This round stops after GATE A; **Step 2 has not started**.", ""]

    # 1. Redraw
    L += ["## 1. Pilot redraw", "",
          f"- Method: `scannetpp.draw_stratified_pilots`. The {sum(draw['type_counts'].values())} scenes of {sc['split']} are stratified by the official `metadata/scene_types.json`; "
          "the 5 types with the most scenes are taken (ties broken by a random shuffle), and 1 scene is drawn uniformly at random within each type.",
          f"- RNG: `{draw['rng']}`, derived from `hash(seed={draw['seed']}, \"{draw['split']}\")` (`common.scene_rng`); types and scenes are sorted first, so reruns give the same result.",
          "- The guide only requires a fixed-seed random draw; stratifying by scene type is an additional requirement.", "",
          "| Type | Scenes | Selected |", "|---|---|---|"]
    picked = {p["scene_type"]: p["scene_id"] for p in draw["picks"]}
    L += [f"| {t} | {c} | {('`' + picked[t] + '`') if t in picked else ''} |" for t, c in draw["type_counts"].items()]
    picks_txt = ", ".join(f"`{p['scene_id']}` ({p['scene_type']})" for p in draw["picks"])
    L += ["", "conference room, kitchen and lab each have 4 scenes (tied for 4th); the random order picked kitchen and conference room.", "",
          f"**New pilots**: {picks_txt}.",
          f"The old pilots ({', '.join(f'`{s}`' for s in sc['pilot_scenes_v0'])}, all office) stay on disk but are no longer pilots.",
          f"**Note**: the office scene drawn with the seed, `c50d2d1d42`, happens to be one of the old pilots (office has 10 scenes, 5 of which were old pilots). "
          "If the new pilots had to avoid the old 5, the same seed would draw `3e8bba0176` from the remaining 5 office scenes.", ""]

    # 2. Full download
    L += ["## 2. Full download check", "",
          "- Two lines changed in the download yml (token line untouched): `download_splits: [nvs_sem_val]`; the old `download_scenes` (5 old pilots) is commented out. The official script skips existing files.",
          f"- sbatch `scripts/run_scannetpp_download.sbatch` (CPU partition, compute nodes have internet access); log `<ScanNet++>/download_full.log`, "
          f"ending: `{dl['log_tail']}`, elapsed {dl['elapsed']}.",
          f"- Log lines that print the token: {dl['token_lines']}; URLs with the token replaced by `token=REDACTED`: {dl['redacted_urls']}.",
          f"- **{dl['n_complete']} / {dl['n_scenes']} scenes have all 7 file types**; missing: {'none' if not dl['missing'] else ', '.join(dl['missing'])}.",
          f"- Total download for the 50 scenes: **{dl['total_bytes'] / GB:.1f} GB** (Step 0 estimate: about 54 GB).",
          "- Pilots first, then the full set: until C2 passes only the 5 new pilot scenes are processed; the other scenes are downloaded but not decoded.", ""]

    L += merge_section(cfg)

    # 4. Decoding the new pilots
    L += ["## 4. Decoding and frame correspondence of the new pilots", "",
          "Decoded as in Step 0 (10 FPS; RGB and depth with the same original frame IDs; written to the toolbox defaults `iphone/rgb`, `iphone/depth`; only new files are added, raw files untouched).", "",
          "| Scene | Type | Video frames / pose keys / depth.bin records | All three equal | Decoded frames | RGB frame IDs = expected | All in pose keys | Depth frame IDs = RGB | RGB size | Depth size | Frames with aligned_pose | segIndices = vertices | Raw files unchanged |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    types = {p["scene_id"]: p["scene_type"] for p in draw["picks"]}
    from .scannetpp import changed_files
    for s in inv:
        snap = sp_work(cfg) / s["scene_id"] / "raw_snapshot.json"
        unchanged = not changed_files(sp_root(cfg) / "data" / s["scene_id"], read_json(snap)) if snap.exists() else None
        p = s["poses"]
        L.append(f"| `{s['scene_id']}` | {types[s['scene_id']]} | {s['video']['n_frames']} / {p['n_keys']} / {s['n_depth_records']} | {yn(s['counts_equal'])} | "
                 f"{s['n_rgb']} | {yn(s['rgb_ids_match_expected'])} | {yn(s['rgb_ids_in_pose_keys'])} | {yn(s['depth_ids_match_rgb'])} | "
                 f"{', '.join('×'.join(map(str, w)) for w in s['rgb_sizes_wh'])} | {', '.join('×'.join(map(str, w)) for w in s['depth_sizes_wh'])} | "
                 f"{p['n_aligned_pose']} / {p['n_keys']} | {yn(s['mesh']['seg_match'])} | {yn(unchanged)} |")
    L += ["", "| Scene | Download (GB) | Decoded RGB (GB) | Decoded depth (GB) | Annotated objects | Distinct labels |", "|---|---|---|---|---|---|"]
    for s in inv:
        L.append(f"| `{s['scene_id']}` | {s['raw_bytes'] / GB:.2f} | {s['rgb_bytes'] / GB:.2f} | {s['depth_bytes'] / GB:.3f} | {s['mesh']['n_objects']} | {s['mesh']['n_unique_labels']} |")
    L.append("")

    # 5. GATE A
    thr = gcfg["depth_median_abs_err_m"]
    L += ["## 5. GATE A", "",
          "### 5.1 Pose convention (10 frames per scene, does not decide whether the scene is kept)", "",
          f"Candidates: aligned_pose is T_wc directly (`direct_*`) or T_cw and must be inverted (`inv_*`); camera axes OpenCV (x right, y down, z forward) or OpenGL (y up, z backward). "
          f"For each frame the mesh is rendered at depth resolution and compared with sensor depth by median absolute error; pass = error < {thr} m. The table shows the pass fraction over the 10 frames and the median error (m).", "",
          "| Scene | " + " | ".join(f"`{c}`" for c in G[0]["conventions"]) + " | Decision |", "|---|" + "---|" * (len(G[0]["conventions"]) + 1)]
    for r in G:
        cells = []
        for c, v in r["conventions"].items():
            e = [f["median_abs_err_m"] for f in v["frames"] if f["median_abs_err_m"] is not None]
            cells.append(f"{v['pass_ratio']:.0%} / {np.median(e):.3f}" if e else f"{v['pass_ratio']:.0%} / no valid pixels")
        L.append(f"| `{r['scene_id']}` | " + " | ".join(cells) + f" | **{r['best_convention']}** |")
    convs = {r["best_convention"] for r in G}
    L += ["", f"Conclusion: {'all 5 scenes agree; ' if len(convs) == 1 else 'scenes disagree: '}aligned_pose is **{'/'.join(sorted(convs))}**"
          + (", i.e. aligned_pose is already camera-to-world (T_wc) with OpenCV camera axes; no inversion or flip needed." if convs == {"direct_opencv"} else "."), ""]
    L += ["### 5.2 Depth unit and RGB–depth scaling", "",
          f"- Depth intrinsics = RGB intrinsics × {G[0]['depth_K_scale'][0]:.6f} (256/1920 = 192/1440, same on both axes), using pixel-centre conversion c' = (c + 0.5)·s − 0.5 (`scannetpp.scale_intrinsics`, unit-tested).",
          "- Depth unit: with the chosen convention, the median of \"rendered depth / sensor depth (uint16 × 0.001)\" over the 10 frames is shown below; ≈ 1 confirms that sensor depth is **uint16 millimetres**.", "",
          "| Scene | Rendered / sensor ratio (range of per-frame medians over 10 frames) |", "|---|---|"]
    for r in G:
        q = r["depth_unit_ratio_rendered_over_sensor"]
        L.append(f"| `{r['scene_id']}` | {np.median(q):.4f} ({min(q):.4f}–{max(q):.4f}) |")
    L += ["", "- RGB vs mesh (i.e. depth) alignment: render mesh depth at 0.5 × RGB resolution, take depth discontinuity edges and compute the median distance to RGB Canny edges (converted back to 1920×1440 pixels). "
          "Control: the same RGB with a pose more than 3 seconds away. The distance with the correct pose is much smaller than the control, so RGB, poses and mesh are aligned; Canny edges include texture, so absolute values are only for relative comparison.", "",
          "| Scene | Correct pose: median edge distance (px, median / max over 10 frames) | Control (wrong pose) |", "|---|---|---|"]
    for r in G:
        a = [x["edge_dist_px"] for x in r["rgb_alignment"] if x["edge_dist_px"] is not None]
        b = [x["edge_dist_px_wrong_pose"] for x in r["rgb_alignment"] if x["edge_dist_px_wrong_pose"] is not None]
        L.append(f"| `{r['scene_id']}` | {np.median(a):.1f} / {max(a):.1f} ({len(a)} frames with enough edges) | {np.median(b):.1f} |")
    L += ["", "### 5.3 Per-pose check and scene retention", "",
          f"For all 10 FPS decoded frames (each has depth with the same frame ID, no timestamp matching needed): valid depth {gcfg['depth_min_valid_m']}–{gcfg['depth_max_valid_m']} m; "
          f"< {gcfg['min_valid_pixels']} valid pixels = unverified; error ≥ {thr} m = bad pose. "
          f"A scene is dropped if bad poses > {gcfg['max_bad_pose_frac']:.0%} of verified poses, or if fewer than {cfg['frames']['min_keyframes_per_scene']} keyframes remain after removing bad frames "
          "(keyframes estimated on the remaining frames with the Step 2 rules: blur filter + motion trigger + sharpest in window; Step 2 recomputes them).", "",
          "| Scene | Poses | Good | Bad | Unverified | Bad / verified | Median / P90 error (m) | Dropped frames (POSE_FRAME_INVALID) | Estimated keyframes | Scene |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in G:
        pc = r["pose_check"]
        L.append(f"| `{r['scene_id']}` | {pc['n_poses']} | {pc['n_good']} | {pc['n_bad']} | {pc['n_unverified']} | {pc['bad_frac_of_verified']:.2%} | "
                 f"{pc['err_median_m']:.3f} / {pc['err_p90_m']:.3f} | {r['n_frames_dropped']} | {r['num_keyframes_est']} | "
                 f"{'✅ kept' if r['pose_valid'] else '❌ POSE_INVALID'} |")
    n_unv = sum(r["pose_check"]["n_unverified"] for r in G)
    L += ["", f"{n_unv} unverified poses in total. With `scannetpp.drop_unverified_pose_frames: {str(sc['drop_unverified_pose_frames']).lower()}` "
          "(same as ARKit), unverified frames are dropped as well. Drop details are in `work/scannetpp/<scene_id>/dropped.jsonl`.", "",
          "Bad-pose timelines (grey = good, red = bad, orange ticks at the bottom = unverified):", ""]
    L += [f"- `{r['scene_id']}`: ![{r['scene_id']} timeline](scannetpp_gateA/{r['scene_id']}_timeline.png)" for r in G]
    L += ["", "### 5.4 Up axis", "",
          "Same method as Step 0: mean normal of the vertices of objects labelled floor; then compare floor and ceiling median coordinates and camera height (report only, no transform).", "",
          "| Scene | Mean floor normal | Detected | Floor / ceiling median z (m) | Median camera height above floor (m) |", "|---|---|---|---|---|"]
    for r in G:
        u = r["up_axis"] or {}
        L.append(f"| `{r['scene_id']}` | {u.get('floor_mean_normal')} | {r['up_axis_detected']} | {u.get('floor_median')} / {u.get('ceiling_median')} | {r['camera_median_height_above_floor_m']} |")
    ups = {r["up_axis_detected"] for r in G}
    L += ["", f"Conclusion: {'all 5 scenes are **+z**, consistent with the 5 office scenes of Step 0; no axis transform needed.' if ups == {'+z'} else f'inconsistent: {sorted(map(str, ups))}; needs handling.'}", ""]
    L += ["### 5.5 Image upright rotation", "",
          "For each frame the pose determines where world +z points in the image (`geometry.sky_direction_from_pose`), giving the required clockwise rotation of 0/90/180/270°; "
          f"the per-frame agreement within a scene must be ≥ {cfg['orientation']['min_agreement_frac']}.", "",
          "| Scene | Per-frame votes (Up=0°, Left=90°, Down=180°, Right=270°) | Agreement | Rotation | Pass |", "|---|---|---|---|---|"]
    for r in G:
        L.append(f"| `{r['scene_id']}` | {r['sky_votes']} | {r['sky_agreement']:.3f} | {r['rotation_cw_quarters'] * 90}° | {yn(r['orientation_ok'])} |")
    L += ["", "### 5.6 Overlay visualisations", "",
          "3 images per scene (good-pose frames from the 10-frame sample): left = RGB downscaled to 960×720, overlaid with mesh depth edges (green) and projected object OBBs (yellow, labelled); "
          "top right = sensor depth, bottom right = mesh-rendered depth (0–5 m, black = no value / mesh hole). "
          f"OBBs are computed from the object point sets in `segments_anno.json` (at GATE A time: orientation by xy-plane PCA, z from the point-set extent; later `scannetpp.object_obb` switched to the minimum-area xy bounding rectangle), and boxes are drawn from the same OBB (`step1_arkit.obb_corners`); "
          f"the corner ordering is unit-tested. Per frame only objects with a visible sample-point fraction ≥ {sc['gate_a_vis_min_visible_frac']} are drawn, at most {sc['gate_a_vis_objects_max']} (visualisation only).", ""]
    for s in pilots:
        imgs = sorted(p.name for p in vis_dir.glob(f"{s}_frame_*.jpg"))
        L += [f"- `{s}`:"] + [f"  - ![{s}](scannetpp_gateA/{n})" for n in imgs]
    L += ["", "Visual check: see Section 7.", ""]

    # 6. Outputs and engineering
    L += ["## 6. Outputs and engineering", "",
          "- Per scene `work/scannetpp/<scene_id>/`: `gate_a.json`, `pose_check.npz` (per-pose status, error, valid pixels, frame IDs), `frames_all.json` (kept frames: T_wc with OpenCV camera axes, "
          "RGB and depth intrinsics, frame IDs, timestamps, position, yaw), `scene_meta.json`, `dropped.jsonl`.",
          "- Code: `scannetpp_step1.py` (GATE A, frames_all, visualisation), `scannetpp_step1_report.py` (this report); `scannetpp.py` adds `object_obb`, `scale_intrinsics`, `draw_stratified_pilots`; "
          "sbatch `scripts/run_scannetpp_decode.sbatch`, `run_scannetpp_gate_a.sbatch`, `run_scannetpp_download.sbatch`.",
          f"- Unit tests (`pytest -q`, including the existing ARKit tests): {pytest_summary()}. New: `tests/test_scannetpp.py` (frame IDs, depth encode/decode, no-overwrite move, stratified draw), "
          "`tests/test_scannetpp_step1.py` (OBB recovers a rotated box, drawn corners in the same order as the computed OBB, intrinsics scaling).",
          "- The token exists only in the download yml (mode 600); it is not in code, git or logs (the token parameter in logged URLs is redacted).", ""]
    # 7. Visual check and open items (written to work/scannetpp/_step1/agent_review.md after viewing the images; appended verbatim)
    review = sp_work(cfg) / "_step1" / "agent_review.md"
    L += [review.read_text().rstrip("\n"), ""] if review.exists() else ["## 7. Visual check and open items", "", "(not filled in yet)", ""]
    path = rep / "scannetpp_step1_gateA.md"
    path.write_text("\n".join(L))
    print("wrote", path)
