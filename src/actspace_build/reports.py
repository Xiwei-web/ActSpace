"""Stage reports: GATE A (lowres pose convention + orientation), Step 3 / GATE B (objects and vocabulary), and the
pilot reports for Steps 2, 4 and 5-6 and GATE C1.

Usage: python -m actspace_build.reports gate_a | step3 | step2 | step4 | step6 | gate_c | orientation_review | q10 | spotcheck_images | spotcheck
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .common import load_config, path_of, read_json, work_dir


def gate_a_report(cfg):
    scenes = cfg["arkitscenes"]["pilot_scenes"] + cfg["arkitscenes"].get("pilot_scenes_dropped", [])
    rows, conv_rows, pose_rows = [], [], []
    for vid in scenes:
        r = read_json(work_dir(cfg, "arkitscenes", vid) / "gate_a.json")
        best = r["conventions"][r["best_convention"]]
        errs = [f["median_abs_err_m"] for f in best["frames"] if f["median_abs_err_m"] is not None]
        pc = r["pose_check"]
        rows.append(f"| {vid} | {r['best_convention']} | {best['pass_ratio']:.0%} | {np.median(errs):.3f} | {max(errs):.3f} | "
                    f"{r['up_axis_detected']} | {r['sky_meta']} | {r['sky_auto']} ({r['sky_agreement']:.0%}) | "
                    f"{'pass' if r['pose_valid'] and r['orientation_ok'] else ('ORIENTATION_CONFLICT' if not r['orientation_ok'] else 'POSE_INVALID')} |")
        pose_rows.append(f"| {vid} | {pc['n_poses']} | {pc['n_good']} | {pc['n_bad']} | {pc['n_unverified']} | {pc['bad_frac_of_verified']:.1%} | "
                         f"{'kept' if r['pose_valid'] else 'POSE_INVALID'} | ![](gateA/pose_timeline/{vid}.png) |")
        conv_rows.append(f"| {vid} | " + " | ".join(f"{r['conventions'][c]['pass_ratio']:.0%}" for c in
                                                    ("inv_opencv", "inv_opengl", "direct_opencv", "direct_opengl")) + " |")
    diag = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "orientation_diagnostic.csv", dtype={"video_id": str})
    ct = pd.crosstab(diag.sky_direction, diag.sky_auto)
    g = cfg["gate_a"]
    L = ["# GATE A: pose and coordinate convention check (ARKitScenes)", "",
         "## 1. Summary", "",
         "- **traj pose convention: the matrix M built from each line of `lowres_wide.traj` (axis-angle + translation) is world-to-camera (T_cw), T_wc = M⁻¹, and the camera axes follow the OpenCV convention.**"
         " Of the 4 candidate conventions tested frame by frame, only `inv_opencv` passes; the others reach <= 20%.",
         "- **World up direction: the 3DOD meshes and `obbAligned` annotations are already z-up** (182 of the 183 selected scenes are auto-detected as +z; the remaining one is an anomalous video with only 48% within-scene agreement), "
         "so the y-up -> z-up transform is the identity on this data; the code keeps the detection and the transform (`Y_UP_TO_Z_UP`) and applies them per scene.",
         f"- **7 pilot scenes** (5 valid + 2 drawn in the first round but removed for an orientation conflict).",
         "- **Resolution**: lowres_wide is 256×192 and is used only for this check; questions and display use vga_wide (640×480, 480×640 for Left scenes after uprighting).",
         f"- **Pose check**: the {g['frames_per_scene']}-frame spot check only determines the pose convention; whether a scene is kept is decided by a per-pose check: "
         f"each traj pose is compared with the lowres_depth frame within {g['pose_check_depth_tol_s']}s; a median absolute error >= {g['depth_median_abs_err_m']}m marks a bad pose, "
         f"and a scene is removed only if bad poses exceed {g['max_bad_pose_frac']:.0%} of the verified poses; a vga frame whose interpolation uses a bad or unverified pose is dropped (`POSE_FRAME_INVALID`). "
         "All 7 pilot scenes are kept (see Section 3).", "",
         "## 2. Pose convention (10-frame spot check)", "",
         f"Per scene, {g['frames_per_scene']} frames are drawn with the scene-level seed; the mesh is rendered to z-depth and compared with lowres_depth on valid pixels "
         f"({g['depth_min_valid_m']}-{g['depth_max_valid_m']}m, hit by the render).", "",
         "The \"pass rate\" below only determines the pose convention; the \"result\" follows the per-pose rule of Section 3 and the orientation check.", "",
         "| video_id | Convention | Spot-check pass rate | Median error (m) | Worst-frame error (m) | Up axis | sky_direction (metadata) | Pose-based estimate (agreement) | Result |",
         "|---|---|---|---|---|---|---|---|---|", *rows, "",
         "Pass rate of the four candidate conventions:", "",
         "| video_id | inv_opencv | inv_opengl | direct_opencv | direct_opengl |", "|---|---|---|---|---|", *conv_rows, "",
         "Material for manual inspection: `reports/gateA/<video_id>_<ts>.jpg`, 3 per scene; left: uprighted image with mesh depth edges (green) and OBB projections (yellow); middle: sensor depth; right: rendered mesh depth.", "",
         "## 3. Per-pose check (distribution of bad poses over time)", "",
         f"\"Unverified\" = no lowres_depth within {g['pose_check_depth_tol_s']}s, or fewer than {g['min_valid_pixels']} valid pixels. In the plots, grey dots = good poses, red dots = bad poses, orange ticks at the bottom = unverified; the y axis is clipped at 0.3m.", "",
         "| video_id | traj poses | Good | Bad | Unverified | Bad / verified | Result | Timeline |", "|---|---|---|---|---|---|---|---|", *pose_rows, "",
         "- The bad poses of 47331971 lie in two segments (7-30s and 140-175s), 5.7% in total, below 20%; the scene is kept under the per-pose rule and serves as the Left scene in C1 that needs a 90° clockwise rotation.",
         "", "## 4. vga_wide interpolated pose check", "", *_vga_section(cfg), "",
         "## 5. Image orientation (all 183 selected scenes, read-only diagnostic)", "",
         "The position of the world up direction in the camera frame is computed from the poses; the majority result over the frames of a scene (`sky_auto`) is cross-checked against `sky_direction` in metadata.csv:", "",
         ct.to_markdown(), "",
         f"- Agree: {int((diag.sky_auto == diag.sky_direction).sum())} / {len(diag)}; disagree: {int((diag.sky_auto != diag.sky_direction).sum())}, "
         "almost all recorded as Down / Left in metadata and estimated as Up from the poses.",
         "- For the 2 conflicting pilots drawn in the first round (42899680, 42899714) the unrotated images were inspected manually: they are upright (vertical door frames, chairs and floor look normal), "
         "so the pose-based estimate is correct and the metadata is wrong. Under the default rule they are still marked `ORIENTATION_CONFLICT` and removed unless confirmed (see `orientation.confirmed_conflict_scenes`).",
         "- K and T_wc are updated together with the uprighting, covered by the unit tests `test_rotation_projection_matches_image_rotation` and `test_sky_direction_and_upright_after_rotation`; "
         "in the overlays of the Down scene 48018375 (rotated 180°) and the Left scene 47331971 (rotated 90° clockwise) the OBBs align with the mesh edges.", "",
         "## 6. Additions to the method", "",
         "- The vga interpolated pose check only compares depth rendered from \"pose + intrinsics\" and **does not check the stored orientation of the vga images themselves**; if the vga and lowres images were stored with different orientations, the depth check would still pass. "
         f"A scene-level image content check is therefore added: after resizing the vga frame to the lowres size, the median normalised cross-correlation with the nearest lowres_wide frame must be >= {g['vga_image_min_ncc']}, and every frame must score higher than its 180°-rotated version.",
         "- Up-axis detection: in most scenes the OBBs are exactly axis-aligned in the horizontal plane, so the three axes tie in the vote; ties are broken by the axis with the highest per-frame image orientation agreement (along the true gravity axis, the orientation of a handheld video is nearly constant)."]
    (path_of(cfg, "reports") / "step1_gateA.md").write_text("\n".join(L) + "\n")


def _vga_section(cfg):
    """Section 4: depth + image content check of the interpolated vga_wide frame poses of the pilot scenes (Step 1)."""
    g, a = cfg["gate_a"], cfg["arkitscenes"]
    rows = []
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        p = work_dir(cfg, "arkitscenes", vid) / "scene_meta.json"
        m = read_json(p) if p.exists() else None
        if m is None or m.get("vga_check") is None:
            return ["To be filled in once vga_wide is downloaded (method: Step 1 + the added image content check, see Section 6)."]
        fr = m["vga_check"]["frames"]
        errs = [f["median_abs_err_m"] for f in fr if f["median_abs_err_m"] is not None]
        rows.append(f"| {vid} | {m['image_size'][0]}×{m['image_size'][1]} | {m['num_frames']} / {m['num_frames_raw']} | "
                    f"{len(fr)} | {m['vga_check']['pass_ratio']:.0%} | {np.median(errs):.3f} | {max(errs):.3f} | "
                    f"{m['vga_check']['median_ncc']:.2f} / {min(f['ncc'] for f in fr):.2f} | {max(f['ncc_rot180'] for f in fr):.2f} | {'pass' if m['vga_check']['image_orientation_ok'] else 'ORIENTATION_CONFLICT'} |")
    return [f"For each vga frame: if the nearest traj pose is within {a['vga_pose_nearest_tol_s']}s it is used directly; otherwise, if the surrounding poses are at most {a['vga_pose_interp_max_gap_s']}s apart, the pose is interpolated "
            "(linear translation, SLERP rotation); otherwise the frame is dropped (`POSE_TS_MISMATCH`); a frame whose poses include a bad or unverified pose is dropped (`POSE_FRAME_INVALID`); the intrinsics come from the nearest vga pincam.",
            f"Check: per scene, {g['frames_per_scene']} frames are drawn; mesh depth is rendered with the interpolated pose + vga intrinsics, resized to 256×192 and compared with the nearest (<= {a['vga_pose_nearest_tol_s']}s) lowres_depth; "
            "the sampled frames come from the frames kept after removing `POSE_FRAME_INVALID`; the depth pass rate is diagnostic only and does not decide whether a scene is kept. "
            f"The scene-level stored-orientation check decides: the median NCC between the resized vga image and the nearest lowres image must be >= {g['vga_image_min_ncc']}, and every frame must score higher than the 180°-rotated NCC.", "",
            "| video_id | Size after uprighting | Kept frames / all | Sampled frames | Depth pass rate (diagnostic) | Median error (m) | Worst-frame error (m) | NCC median / min | Max NCC (rotated 180°) | Orientation check |",
            "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
            "OBB projections overlaid on the uprighted vga images: `reports/gateA/vga_wide/<video_id>_vga_wide_<ts>.jpg`, 3 per scene."]


def step3_report(cfg):
    freq = pd.read_csv(path_of(cfg, "vocab") / "category_frequency_arkitscenes.tsv", sep="\t")
    rows, reasons = [], Counter()
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        objs = read_json(work_dir(cfg, "arkitscenes", vid) / "objects.json")
        for o in objs:
            reasons.update(o["drop_reasons"])
        rows.append(f"| {vid} | {len(objs)} | {sum(o['referable'] for o in objs)} | "
                    f"{', '.join(o['category'] for o in objs if o['referable'])} | "
                    f"{min(o['num_points'] for o in objs if o['referable'])} |")
    thr = cfg["objects"]["min_referable_objects_per_scene"]
    L = ["# Step 3 / GATE B: object curation and category vocabulary (ARKitScenes)", "",
         "## 1. Category vocabulary (for review)", "",
         "- `vocab/categories.tsv`: all 17 ARKitScenes categories are mapped and allowed in questions; `tv_monitor` -> `TV`, `washer` -> `washing machine` (to avoid ambiguity). "
         "The forbidden categories wall/floor/ceiling/object/stuff/unknown/misc are listed for the ScanNet++ mapping.",
         "- `vocab/label_mapping_arkitscenes.tsv`: raw label -> unified category.", "",
         "## 2. Category frequency (183 deduplicated selected scenes, annotations only)", "",
         "`scenes_unique` = number of scenes with exactly 1 instance of the category (uniquely referable).", "",
         freq.to_markdown(index=False), "",
         "## 3. Referable objects and scene size (183 deduplicated selected scenes, statistics only)", "",
         f"Counted after all Step 3 filters (uniquely referable by name with an allowed category, points >= {cfg['objects']['min_object_points']}, longest edge >= {cfg['objects']['min_longest_edge_m']}m); "
         "data in `work/arkitscenes/_step0/referability_stats.csv`.", ""]
    oc = cfg["objects"]
    rs = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "referability_stats.csv", dtype={"video_id": str})
    rs["with_size"] = rs.referable + rs.extra_size_disambig
    confirmed = {str(v) for v in cfg["orientation"]["confirmed_conflict_scenes"]}
    ok = rs[~rs.orientation_conflict | rs.video_id.isin(confirmed)]
    dist = Counter(rs.with_size.tolist())
    kinds = Counter(x.split(" ", 1)[0] if " " in x else x for e in rs.extra_examples.dropna() for x in e.split("; "))
    L += ["Distribution of referable objects per scene under the current rules:", "",
          "| Referable objects | " + " | ".join(str(k) for k in sorted(dist)) + " |",
          "|---|" + "---|" * len(dist), "| Scenes | " + " | ".join(str(dist[k]) for k in sorted(dist)) + " |", "",
          f"| Threshold | Unique category only (all {len(rs)}) | Current rules (all {len(rs)}) | Current rules (after orientation removals, {len(ok)}) |", "|---|---|---|---|",
          *[f"| >= {k}{' (current threshold)' if k == thr else ''} | {int((rs.referable >= k).sum())} | {int((rs.with_size >= k).sum())} | {int((ok.with_size >= k).sum())} |"
            for k in (3, 4, 5)], "",
          "Current rules:",
          f"- a category with exactly 1 instance in the scene -> referred to by its category name;",
          f"- if exactly one sofa instance has a longest edge < {oc['armchair_max_longest_edge_m']}m, it is renamed \"the armchair\" (a separate category; if a single sofa remains, it also becomes unique);",
          *(["- **Size-based references are disabled (`size_ref_enabled: false`)**: the next rule does not apply, and objects are referred to only by a category name that is unique in the scene."]
            if not oc.get("size_ref_enabled", False) else []),
          f"- with 2-{oc['size_disambig_max_instances']} instances of a category whose centres are pairwise >= {oc['size_disambig_min_center_dist_m']}m apart, "
          f"an instance whose volume ratio to the next largest (or next smallest) is >= {oc['size_disambig_min_volume_ratio']} can be referred to by size: \"the larger / smaller X\" for 2 instances, \"the largest / smallest X\" for 3; "
          "all instances of that category are added to evidence_objs.", "",
          f"- Compared with unique categories only, each scene gains {rs.extra_size_disambig.mean():.1f} referable objects on average; by reference type: "
          + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) + " (`sofa` = the single sofa left after the armchair is split off).",
          f"- At threshold {thr}, {int(((rs.with_size >= thr) & (rs.referable < thr)).sum())} more scenes qualify.",
          "- Note: scenes with only 3 referable objects cannot produce T2 (reference + at least 3 options need 4 objects); T5 also needs >= 4.", "",
          "## 4. Object curation in the pilot scenes", "",
          f"Each OBB is inflated by {cfg['objects']['arkit_obb_inflate_m']}m horizontally and upwards only, vertices with z < box bottom + {cfg['objects']['floor_cut_m']}m are removed, and the mesh vertices inside form the object point set; points < {cfg['objects']['min_object_points']} -> UNRELIABLE_GEOMETRY; "
          f"longest edge < {cfg['objects']['min_longest_edge_m']}m -> TOO_SMALL; non-unique or forbidden category -> NOT_REFERABLE.", "",
          "| video_id | Annotated objects | Referable | Referable categories | Min points of a referable object |", "|---|---|---|---|---|", *rows, "",
          f"Drop reasons (an object can have several): {dict(reasons)}", "",
          "Note: floor points are removed from the object point sets (no downward inflation, vertices within 3cm of the box bottom removed), which reduces the point count of the pilot objects by 12.5% on average."]
    (path_of(cfg, "reports") / "step3_objects.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- GATE C1: Step 2 / 4 / 6 reports and visual cards

DATASET = "arkitscenes"


def _drops(cfg, vid):
    import json
    p = work_dir(cfg, DATASET, vid) / "dropped.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _hist(values, bins):
    h, e = np.histogram(values, bins=bins)
    return " / ".join(f"{e[i]:.0f}–{e[i + 1]:.0f}: {n}" for i, n in enumerate(h))


def keyframe_montage(cfg, out_path, n=30, cols=6, tile_w=320):
    """Draw n random (seeded) keyframes from all pilot scenes (uprighted) and tile them into one image to check sharpness."""
    import cv2
    from . import geometry as g
    from .common import scene_rng
    from .step2_keyframes import laplacian_variance
    root = path_of(cfg, "arkit_3dod_root").resolve().parents[1]
    pool = []
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        wd = work_dir(cfg, DATASET, vid)
        frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
        pool += [(vid, frames[k]) for k in read_json(wd / "keyframes.json")["keyframe_ids"]]
    rng = scene_rng(cfg, "pilot_keyframe_montage", "montage")
    picks = sorted(rng.choice(len(pool), min(n, len(pool)), replace=False).tolist())
    tiles = []
    for i in picks:
        vid, f = pool[i]
        img = g.rotate_image_cw(cv2.imread(str(root / f["rel_path"])), f["rotation_cw_quarters"])
        sc = tile_w / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        canvas = np.full((tile_w, tile_w, 3), 255, np.uint8)
        y0, x0 = (tile_w - img.shape[0]) // 2, (tile_w - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        txt = f"{vid} {f['frame_id']} lapvar={laplacian_variance(root / f['rel_path']):.0f}"
        cv2.rectangle(canvas, (0, 0), (tile_w, 18), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(canvas, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.concatenate([np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def step2_report(cfg):
    rows, few = [], []
    fc = cfg["frames"]
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        wd = work_dir(cfg, DATASET, vid)
        k = read_json(wd / "keyframes.json")
        n_pose_bad = sum(d["reason_code"] == "POSE_FRAME_INVALID" for d in _drops(cfg, vid))
        meta = read_json(wd / "scene_meta.json")
        rows.append(f"| {vid} | {meta['num_frames_raw']} | {n_pose_bad} | {k['num_frames']} | {k['num_blurry']} ({k['num_blurry'] / k['num_frames']:.0%}) | "
                    f"{k['blur_threshold_percentile']:.1f} / {k['blur_threshold_rel_median']:.1f} | {k['num_blurry_rel_only']} | "
                    f"{k['frame_lapvar_median']:.0f} → {k['keyframe_lapvar_median']:.0f} | {k['num_keyframes']} | {'pass' if k['scene_ok'] else 'TOO_FEW_KEYFRAMES'} |")
        if not k["scene_ok"]:
            few.append(f"- {vid}: {k['num_keyframes']} keyframes (< {fc['min_keyframes_per_scene']}), marked `TOO_FEW_KEYFRAMES`; thresholds not relaxed.")
    keyframe_montage(cfg, path_of(cfg, "reports") / "step2_keyframe_montage.jpg")
    L = ["# Step 2: frame extraction and keyframe selection (ARKitScenes vga_wide, 5 pilot scenes)", "",
         "Rules:",
         f"- Blur filter: frames whose Laplacian variance is below the scene's {fc['blur_drop_percentile']}th percentile **or** below the scene median × {fc['blur_min_rel_to_median']} are dropped (`BLURRY`); the stricter of the two thresholds applies.",
         f"- Keyframes: in temporal order, the first frame with translation >= {fc['keyframe_min_translation_m']}m or |Δyaw| >= {fc['keyframe_min_rotation_deg']}° relative to the previous keyframe is the trigger frame; "
         f"within ±{fc['keyframe_sharpest_window_s']}s of it (and after the previous keyframe) the frame with the largest Laplacian variance becomes the keyframe; "
         f"scenes with fewer than {fc['min_keyframes_per_scene']} keyframes are marked `TOO_FEW_KEYFRAMES`.",
         "- Frame source: vga_wide (uprighted), with `POSE_TS_MISMATCH` and `POSE_FRAME_INVALID` frames removed.", "",
         "| video_id | vga frames | POSE_FRAME_INVALID | Frames with valid pose | Blurry dropped | Threshold: percentile / median ratio | Dropped by median ratio only | Median Laplacian variance: all frames -> keyframes | Keyframes | Result |",
         "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
         "Scenes with fewer than 40 keyframes: " + ("none." if not few else ""), *few, "",
         "## Sharpness spot check", "",
         "30 keyframes drawn at random (seeded) from all keyframes of the 5 pilot scenes (uprighted): `reports/step2_keyframe_montage.jpg`; each tile is labelled with scene, frame id and Laplacian variance.", "",
         "![](step2_keyframe_montage.jpg)"]
    (path_of(cfg, "reports") / "step2_keyframes.md").write_text("\n".join(L) + "\n")


def step4_report(cfg):
    from .step4_graph_visibility import ACTIONS
    rows, obj_rows = [], []
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        s = read_json(work_dir(cfg, DATASET, vid) / "step4_summary.json")
        rows.append(f"| {vid} | {s['num_keyframes']} | {s['num_graph_nodes']} | {s['scc_ratio']:.0%} | {'yes' if s['relaxed'] else 'no'} | "
                    + " | ".join(str(s["edges_per_action"][a]) for a in ACTIONS) + f" | {'pass' if s['ok'] else 'GRAPH_DISCONNECTED'} |")
        ref = [o for o in s["objects"] if o["referable"]]
        vis = [o["num_visible_nodes"] for o in ref]
        obj_rows.append(f"| {vid} | " + ", ".join(f"{o['category']} {o['num_visible_nodes']}/{o['num_partial_nodes']}" for o in ref)
                        + f" | {min(vis)} | {int(np.median(vis))} |")
    vc, vg = cfg["visibility"], cfg["view_graph"]
    L = ["# Step 4: view graph and per-frame visibility (ARKitScenes vga_wide, 5 pilot scenes)", "",
         "## 1. View graph", "",
         f"Nodes = keyframes; edges for the 6 actions follow the action table (move {vg['move_min_m']}-{vg['move_max_m']}m, turn {vg['turn_min_deg']}-{vg['turn_max_deg']}°); "
         f"only the largest strongly connected component is kept, and if it holds < {vg['min_scc_ratio']:.0%} of the nodes the move limit is relaxed to {vg['move_max_m_relaxed']}m and retried once.", "",
         "| video_id | Keyframes | Graph nodes | Kept fraction | Relaxed retry | " + " | ".join(ACTIONS) + " | Result |",
         "|---|---|---|---|---|" + "---|" * len(ACTIONS) + "---|", *rows, "",
         "## 2. Visible nodes of referable objects", "",
         f"Rule: visible = unoccluded ratio ρ >= {vc['visible_min_ratio']} and projected area fraction >= {vc['visible_min_area_frac']} and unoccluded points >= {vc['visible_min_points']}"
         f" and completeness (unoccluded points / all sampled points of the object) >= {vc['visible_min_completeness']}; "
         f"invisible = outside the frustum or ρ < {vc['invisible_max_ratio']}; everything else is partial. Mesh occlusion tolerance {vc['occlusion_tolerance_m']}m. "
         f"At nodes with a lowres_depth frame within {vc['sensor_depth_tol_s']}s, a point must also be unoccluded according to the sensor depth (tolerance {vc['sensor_occlusion_tolerance_m']}m). "
         "Object point sets exclude floor points (inflated horizontally and upwards only, vertices within 3cm of the box bottom removed).", "",
         "Effect of the completeness and sensor-depth rules on visible decisions (number of (object, node) pairs):", "",
         "| video_id | Graph nodes | Nodes with sensor depth | Visible (base rule) | + completeness | + sensor depth | Removed by completeness | Removed by sensor depth |", "|---|---|---|---|---|---|---|---|",
         *[f"| {v} | {x['num_graph_nodes']} | {x['nodes_with_sensor_depth']} | {x['visible_pairs']['legacy']} | {x['visible_pairs']['with_completeness']} | "
           f"{x['visible_pairs']['final_with_sensor_depth']} | {x['visible_lost_by_completeness']} | {x['visible_lost_by_sensor_depth']} |"
           for v, x in ((v, read_json(work_dir(cfg, DATASET, v) / "step4_summary.json")) for v in cfg["arkitscenes"]["pilot_scenes"])], "",
         "Spot-check results: `visibility_spotcheck.md`.", "",
         "Each entry is \"category visible nodes / partial nodes\".", "",
         "| video_id | Referable objects | Min visible nodes | Median |", "|---|---|---|---|", *obj_rows, "",
         "## 3. Visibility overlays", "",
         f"`reports/step4/{DATASET}/<video_id>_<frame_id>.jpg`, 5 per scene (evenly spaced over the graph nodes). Sampled points of each object inside the frustum: bright = unoccluded, dark = occluded; "
         "green = visible, orange = partial, red = invisible; labels show category#id, decision, ρ and area fraction a."]
    (path_of(cfg, "reports") / "step4_graph_visibility.md").write_text("\n".join(L) + "\n")


def _load_groups(cfg):
    import json
    p = path_of(cfg, "work") / DATASET / "_step6" / "test_groups.jsonl"
    return [json.loads(l) for l in open(p)]


def _special_cell(r):
    a = r.get("added")
    if a:
        return f"generated: {a['template']} ({a.get('object') or ', '.join(a.get('objects', []))})"
    return f"not generated: {r.get('reason', '')}"


def _ua_cell(r):
    if r.get("added"):
        return "; ".join(f"generated {a['template']} (masked {a['object']}, {a['masked_nodes']} nodes)" for a in r["added"])
    return "not generated: " + ", ".join(f"{k} {v}" for k, v in r.get("fail_counts", {}).items())


def step6_report(cfg):
    st = read_json(path_of(cfg, "work") / DATASET / "_step6" / "step6_stats.json")
    groups = _load_groups(cfg)
    qs = [q for grp in groups for q in grp["questions"]]
    reasons = Counter()
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        reasons.update(d["reason_code"] + (f" ({d['detail']})" if "detail" in d else "") for d in _drops(cfg, vid) if d["step"] == "step56")
    fmt = lambda d: ", ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda kv: (0, int(kv[0]), "") if str(kv[0]).isdigit() else (1, 0, str(kv[0]))))
    t1 = [q["answer"] for q in qs if q["template"] == "T1"]
    L = ["# Steps 5-6: object clusters, initial views and template questions (ARKitScenes vga_wide, 5 pilot scenes)", "",
         f"- **{st['num_groups']}** question groups, **{st['num_questions']}** questions.",
         f"- Per template: {fmt(st['per_template'])}; per variant: {fmt(st['per_variant'])}.",
         f"- Evidence labels: {fmt(st['evidence_label'])}.",
         f"- Constraint types: {fmt(st['constraints'])}.",
         f"- Difficulty (explore questions split into terciles of oracle_steps, cut points {[round(c, 2) for c in st['difficulty_cuts_oracle_steps']]}, provisional during the pilot): {fmt(st['difficulty'])}.", "",
         "## 0. Rule changes in this round", "",
         "- T2 and T5 option objects are drawn from all referable objects of the scene (the reference and standing/facing objects still come from the cluster); option objects are added to evidence_objs and count towards the evidence label and the oracle; "
         f"supplementary T2 questions have at least {cfg['questions']['t2_options'][0]} options, and two-way comparisons (`T2:pair`) appear only in constraints.",
         "- During the pilot each scene attempts at least 1 init and 1 unanswerable question (see Section 4); ratio balancing is left to the full run. T3/T4 quadrant balancing is not applied in the pilot.", "",
         "## 1. Answer distribution", "",
         "| Template | Correct option letter |", "|---|---|",
         *[f"| {t} | {fmt(d)} |" for t, d in st["answer_letter_per_template"].items() if d], "",
         "| Template | Quadrant |", "|---|---|", *[f"| {t} | {fmt(d)} |" for t, d in st["quadrant_per_template"].items()], "",
         f"T1 numeric answers (m): {_hist(t1, [0, 1, 2, 3, 4, 5, 6, 8]) if t1 else 'none'}.", "",
         "## 2. Oracle statistics", "",
         f"- `oracle_views` (minimum number of views that must be seen): {fmt(st['oracle_views'])}",
         f"- `oracle_steps` (minimum number of actions from p0): {fmt(st['oracle_steps'])}", "",
         "## 3. Filter and drop reasons (Steps 5-6, by reason_code)", "",
         "| reason_code | Count |", "|---|---|", *[f"| {k} | {v} |" for k, v in reasons.most_common()], "",
         "## 4. init / unanswerable pipeline check (1 attempt each per scene)", "",
         "Unanswerable questions use the masking method, see `q10_unanswerable.md`; init questions are generated when possible and not forced.", "",
         "| video_id | unanswerable (masking) | init |", "|---|---|---|",
         *[f"| {vid} | {_ua_cell(sp['unanswerable'])} | {_special_cell(sp['init'])} |" for vid, sp in st.get("special_attempts", {}).items()], "",
         "Visibility thresholds and connectivity requirements were not relaxed to reach the targets.", "",
         "## 5. Group-level drops", "",
         "A group that cannot reach 3-5 questions, 2 templates or 1 constraint is dropped as a whole and recorded as `GROUP_INCOMPLETE`, with the reason in detail.", "",
         "Candidate filter counts (over enumerated object combinations within a cluster; the same combination may be enumerated once for constraints and once for supplementary questions, so this is not the number of dropped questions):", "",
         "| Template:filter reason | Count |", "|---|---|", *[f"| {k} | {v} |" for k, v in st.get("candidate_filter_counts", {}).items()]]
    (path_of(cfg, "reports") / "step6_questions.md").write_text("\n".join(L) + "\n")


def _draw_tile(cfg, sc, fid, objs, title, title_color=(0, 0, 0), image_root=None):
    """Uprighted frame + OBBs of the given objects (coloured by visibility at that node: green=visible, orange=partial, red=invisible).
    image_root: root directory of the frame rel_path; defaults to the ARKit data directory (callers pass it for other datasets)."""
    import cv2
    from . import geometry as g
    from .step1_arkit import _BOX_EDGES, obb_corners
    from .step4_graph_visibility import _STATE_COLOR
    root = image_root or path_of(cfg, "arkit_3dod_root").resolve().parents[1]
    f = sc.frames[sc.node_index[fid]]
    img = g.rotate_image_cw(cv2.imread(str(root / f["rel_path"])), f["rotation_cw_quarters"])
    K, T = np.asarray(f["K"]), np.asarray(f["T_wc"])
    H, W = img.shape[:2]
    k = max(1.0, W / 640)  # line width and font size scale with image width; ARKit vga frames (width <= 640) have k = 1
    lw, fs = max(2, int(round(2 * k))), 0.6 * k
    for oid in objs:
        o = sc.objs[sc.row[oid]]["obb"]
        col = _STATE_COLOR[sc.st(oid, sc.node_index[fid])]
        C = obb_corners({k: np.asarray(v) for k, v in o.items() if k != "yaw_deg"})
        for a, b in _BOX_EDGES:
            uv, z = g.project(np.linspace(C[a], C[b], 40), K, T)
            ok = g.in_image(uv, z, W, H)
            for p, q, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
                if oa and ob:
                    cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), col, lw, cv2.LINE_AA)
        uv, z = g.project(sc.center[oid][None], K, T)
        if g.in_image(uv, z, W, H)[0]:
            p = tuple(np.round(uv[0]).astype(int))
            cv2.putText(img, sc.cat[oid], p, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2 * lw, cv2.LINE_AA)
            cv2.putText(img, sc.cat[oid], p, cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, int(26 * k)), title_color, -1)
    cv2.putText(img, title, (int(6 * k), int(19 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(round(k))), cv2.LINE_AA)
    return img


def _compose(tiles, lines):
    """Concatenate tiles horizontally and add a text panel below."""
    import cv2
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    top = np.concatenate(tiles, axis=1)
    panel = np.full((22 * len(lines) + 16, top.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([top, panel], axis=0)


def _card_image(cfg, sc, grp, out_path):
    """Card: initial view + evidence views (OBBs of cluster and evidence objects, green=visible, orange=partial, red=invisible) + questions and answers."""
    import textwrap
    import cv2
    cluster = sorted(set(grp["cluster_obj_ids"]) | {o for q in grp["questions"] for o in q["evidence_objs"]})
    count = Counter(n for q in grp["questions"] for n in q["oracle_nodes"] or [])
    ev_nodes = [n for n, _ in count.most_common(cfg["report"]["card_max_evidence_views"])]

    def tile(fid, title):
        return _draw_tile(cfg, sc, fid, cluster, title)

    tiles = [tile(grp["init_frame"], f"p0 (initial view) {grp['init_frame']}")]
    tiles += [tile(n, f"evidence view {n}") for n in ev_nodes]
    lines = [f"{grp['group_id']}  {grp['scene_id']}  cluster: " + ", ".join(sc.cat[o] for o in grp["cluster_obj_ids"])
             + "  | other evidence objects: " + (", ".join(sc.cat[o] for o in cluster if o not in grp["cluster_obj_ids"]) or "none")
             + f"  | constraint: {grp['constraints'][0]['type']}"]
    for q in grp["questions"]:
        ans = q["answer"] if q["answer_type"] == "numeric" else f"{q['answer']} ({q.get('answer_value') or sc.cat[q['answer_obj']]})"
        head = f"[{q['template']}] {q['text']}"
        lines += textwrap.wrap(head, 150)
        if q["options"]:
            lines.append("      " + "   ".join(q["options"]))
        lines.append(f"      answer: {ans}   evidence={q['evidence_label']}  oracle_views={q['oracle_views']}  oracle_steps={q['oracle_steps']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), _compose(tiles, lines))


def _unanswerable_card(cfg, sc, grp, q, out_path, variant_note=None):
    """Card for a masked unanswerable question: p0 + 2 masked views where the masked object is visible (plus 1 view masked only because the remaining graph was disconnected) + notes."""
    import textwrap
    import cv2
    x = q["masked_obj"]
    objs = sorted(set(q["evidence_objs"]))
    masked = set(q["masked_nodes"])
    # nodes masked because x is visible or partial there; visible ones are shown first
    by_obj = sorted([n for n in q["masked_nodes"] if sc.st(x, sc.node_index[n]) != 0], key=lambda n: -sc.st(x, sc.node_index[n]))[:2]
    extra = [n for n in q["masked_nodes"] if sc.st(x, sc.node_index[n]) == 0][:1]  # additionally masked because the remaining graph was disconnected
    tiles = [_draw_tile(cfg, sc, grp["init_frame"], objs, f"p0 (initial view) {grp['init_frame']}")]
    tiles += [_draw_tile(cfg, sc, n, objs, f"MASKED {n}: {sc.cat[x]} {['', 'partial', 'visible'][sc.st(x, sc.node_index[n])]} here", (40, 40, 170))
              for n in by_obj]
    tiles += [_draw_tile(cfg, sc, n, objs, f"MASKED {n}: outside p0's SCC", (40, 110, 170)) for n in extra]
    lines = [f"{grp.get('group_id', '')}  {grp['scene_id']}  " + (variant_note or "masking rule: (b)+(c), whole group shares the masked view graph"),
             *textwrap.wrap(f"[{q['template']}] {q['text']}", 150),
             f"      answer: cannot be determined (evidence=unanswerable); masked object: {sc.cat[x]}",
             f"      masked nodes: {len(masked)} / {len(sc.nodes)} ({q['masked_frac']:.1%}); "
             f"{q['masked_by_object']} because {sc.cat[x]} is visible/partial there"
             + (f", {len(masked) - q['masked_by_object']} more to keep the remaining graph strongly connected" if len(masked) > q["masked_by_object"] else ""),
             f"      other questions in this group (oracle recomputed on the masked graph): "
             + "; ".join(f"[{oq['template']}] views={oq['oracle_views']} steps={oq['oracle_steps']}" for oq in grp["questions"] if oq is not q)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), _compose(tiles, lines))


def q10_unanswerable_report(cfg):
    """Pilot cards for masked unanswerable questions. The released data uses only the strict rule."""
    import cv2
    from .common import scene_rng
    from .step56_questions import Scene
    groups = _load_groups(cfg)
    st = read_json(path_of(cfg, "work") / DATASET / "_step6" / "step6_stats.json")
    out_dir = path_of(cfg, "reports") / "q10_unanswerable_cards"
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    rows, cards = [], []
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        sg = [grp for grp in groups if grp["scene_id"] == f"arkit_{vid}"]
        sc = Scene(cfg, DATASET, vid) if sg else None
        ua = [(grp, q) for grp in sg for q in grp["questions"] if q.get("unanswerable_method") == "mask"]
        sp = st["special_attempts"][vid]["unanswerable"]
        rows.append(f"| {vid} | {len(ua)} | {sp.get('n_candidates', 0)} | "
                    + (", ".join(f"{k}: {v}" for k, v in sp.get("fail_counts", {}).items()) or sp.get("reason", "")) + " |")
        for grp, q in ua:
            path = out_dir / f"{vid}_{q['qid']}.jpg"
            _unanswerable_card(cfg, sc, grp, q, path)
            cards.append((vid, grp, q, path, False))
    qc = cfg["questions"]
    L = ["# Pilot cards for masked unanswerable questions", "",
         "Rules:", "",
         "- Remove from the view graph every node where the masked object is visible or partial; the object is invisible at all remaining nodes, and p0 is never masked;",
         "- if the remaining graph is disconnected, also mask the nodes outside p0's strongly connected component (the masked object is already invisible there);",
         f"- candidates masking > {qc['unanswerable_max_masked_frac']:.0%} of the nodes are discarded;",
         "- the other questions of the group must remain answerable after masking, and **the whole group shares the masked view graph**: the group records `view_graph_mask`, and the oracle of the other questions is recomputed on the masked graph;",
         "- the question records `masked_obj`, `masked_nodes`, `masked_frac` and `masked_by_object` (the number of nodes masked because the object is visible there).", "",
         f"Target per scene: {qc['pilot_unanswerable_per_scene'][0]}-{qc['pilot_unanswerable_per_scene'][1]} questions, preferring candidates that mask the fewest nodes; candidate questions are tried in the order T4(X), T1(X, cluster object); at most 1 per group.", "",
         "## 1. Results per scene", "",
         "| video_id | Generated | Valid candidates | Failure counts (group × object) |", "|---|---|---|---|", *rows, "",
         "Failure reasons: `group_full` the group already has 5 questions; `p0_masked` the object is visible/partial at p0; `masked_frac_too_large` the masked fraction exceeds the limit; "
         "`group_incompatible` other questions of the group become unanswerable after masking; `margin_fail` the T4 angular margin / T1 distance range is not met; `evidence_too_large` too many evidence objects.", "",
         "## 2. Cards", "",
         "Red title = masked because the masked object is visible/partial at that node; orange title = additionally masked because it is disconnected from p0. OBB colours show the original visibility at that node (green visible / orange partial / red invisible).", ""]
    for vid, grp, q, path, alt in cards:
        L += [f"### {vid} · {q['qid']}", "", f"![]({path.relative_to(path_of(cfg, 'reports'))})", "",
              f"- Question: {q['text']}",
              f"- Masked object: {sc_name(cfg, vid, q['masked_obj'])}; {len(q['masked_nodes'])} masked nodes ({q['masked_frac']:.1%}), {q['masked_by_object']} because the object is visible, {len(q['masked_nodes']) - q['masked_by_object']} because of disconnection",
              f"- Masked nodes: {', '.join(q['masked_nodes'][:30])}{' …' if len(q['masked_nodes']) > 30 else ''}", ""]
    (path_of(cfg, "reports") / "q10_unanswerable.md").write_text("\n".join(L) + "\n")


def sc_name(cfg, vid, oid):
    o = read_json(work_dir(cfg, DATASET, vid) / "objects.json")[oid]
    return o.get("ref_name") or o["category"]


def gate_c_report(cfg):
    from .common import scene_rng
    from .step56_questions import Scene
    groups = _load_groups(cfg)
    st = read_json(path_of(cfg, "work") / DATASET / "_step6" / "step6_stats.json")
    rows, drop_rows, cards = [], [], []
    card_dir = path_of(cfg, "reports") / "gateC_cards"
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        wd = work_dir(cfg, DATASET, vid)
        meta, kf = read_json(wd / "scene_meta.json"), read_json(wd / "keyframes.json")
        s4, s3 = read_json(wd / "step4_summary.json"), read_json(wd / "step3_summary.json")
        raw_p = wd / "groups_raw.json"
        n_clusters = len(read_json(raw_p)["clusters"]) if raw_p.exists() else "— (scene removed)"
        sg = [grp for grp in groups if grp["scene_id"] == f"arkit_{vid}"]
        rows.append(f"| {vid} | {meta['sky_direction_auto']} | {meta['num_frames']} / {meta['num_frames_raw']} | "
                    f"{read_json(wd / 'gate_a.json')['pose_check']['bad_frac_of_verified']:.1%} | {kf['num_keyframes']} | {s4['num_graph_nodes']} ({s4['scc_ratio']:.0%}) | "
                    f"{sum(o['referable'] for o in s4['objects'])} | {n_clusters} | {len(sg)} | {sum(len(x['questions']) for x in sg)} |")
        c = Counter(d["reason_code"] for d in _drops(cfg, vid))
        drop_rows.append(f"| {vid} | " + ", ".join(f"{k}: {v}" for k, v in c.most_common()) + " |")
        rng = scene_rng(cfg, f"arkit_{vid}", "gate_c_cards")
        # groups with supplementary init / unanswerable questions first, the rest drawn at random (seeded)
        n_cards = min(cfg["report"]["cards_per_scene"], len(sg))
        special = [i for i, grp in enumerate(sg) if any(q.get("supplement") for q in grp["questions"])][:n_cards]
        rest = [i for i in range(len(sg)) if i not in special]
        pick = sorted(special + [rest[int(j)] for j in rng.permutation(len(rest))[:n_cards - len(special)]])
        sc = Scene(cfg, DATASET, vid) if sg else None
        for i in pick:
            grp = sg[i]
            path = card_dir / f"{vid}_{grp['group_id']}.jpg"
            _card_image(cfg, sc, grp, path)
            cards.append((vid, grp, path))
    L = ["# GATE C1: ARKitScenes pilot review (5 scenes, Steps 1-6)", "",
         "> Frame source: vga_wide (640×480, uprighted); poses interpolated from lowres_wide.traj. ScanNet++ (C2) is reviewed separately.", "",
         "## 1. Scene sizes", "",
         "| video_id | Orientation | vga frames with valid pose / all | Bad poses / verified | Keyframes | Graph nodes (kept fraction) | Referable objects | Clusters | Groups | Questions |",
         "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
         f"Total: {st['num_groups']} question groups, {st['num_questions']} questions; templates {st['per_template']}; evidence labels {st['evidence_label']}. "
         "Detailed statistics in `step6_questions.md`.", "",
         "## 2. Drop reasons (dropped.jsonl, Steps 1-6)", "",
         "| video_id | reason_code: count |", "|---|---|", *drop_rows, "",
         "## 3. Points for review", "",
         f"- Evidence labels: {st['evidence_label']}; per-scene init / unanswerable attempts are listed in Section 4 of `step6_questions.md`.",
         f"- Variants: {st['per_variant']} (with T2 and T5 options drawn from all referable objects of the scene).",
         "- Poses: per-pose check; vga frames that depend on bad poses are recorded as `POSE_FRAME_INVALID` (timeline plots in Section 3 of step1_gateA.md); 47331971 is kept under the per-pose rule.",
         "- Sharpness: blur filtering and keyframe selection were rerun with the current rules; a montage of 30 keyframes is in `step2_keyframe_montage.jpg`.",
         "- T3/T4 quadrant distributions are not balanced in the pilot (left to the full run).", "",
         "## 4. Visual cards", "",
         f"{cfg['report']['cards_per_scene']} question groups per scene: groups with supplementary init / unanswerable questions first, the rest drawn with the scene-level seed. "
         "The cards show OBBs of the cluster objects and of all evidence objects of each question: green = visible in that view, orange = partial, red = invisible. "
         "The first image is the initial view p0, followed by evidence views (the views occurring most often among the questions' oracle views). Question texts are the template sentences before the Step 7 rewrite.", ""]
    for vid, grp, path in cards:
        L += [f"### {vid} · {grp['group_id']}", "", f"![]({path.relative_to(path_of(cfg, 'reports'))})", "",
              "| Template | Question | Options | Answer | Evidence label | oracle_views / steps |", "|---|---|---|---|---|---|"]
        for q in grp["questions"]:
            ans = q["answer"] if q["answer_type"] == "numeric" else f"{q['answer']}"
            L.append(f"| {q['template']}{':' + q['variant'] if q.get('variant') else ''}{' (supplementary ' + q['supplement'] + ')' if q.get('supplement') else ''} | {q['text']} | "
                     f"{'<br>'.join(q['options']) if q['options'] else '—'} | {ans} | {q['evidence_label']} | {q['oracle_views']} / {q['oracle_steps']} |")
        L += ["", f"Constraint: `{grp['constraints'][0]['type']}` -> {grp['constraints'][0]['expected']}", ""]
    (path_of(cfg, "reports") / "gateC_pilot.md").write_text("\n".join(L) + "\n")


def orientation_review(cfg, per_sheet=6, frames_per_scene=3, scale=1.25):
    """Export uprighted thumbnails for manual confirmation of scenes whose metadata orientation conflicts with the pose-based
    estimate and whose within-scene agreement meets the threshold.
    One scene per row: the first 3 tiles are uprighted with the pose-based estimate, the last one with the metadata (for contrast)."""
    import cv2
    from . import arkit
    from . import geometry as g
    from .common import scene_rng
    diag = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "orientation_diagnostic.csv", dtype={"video_id": str})
    thr = cfg["orientation"]["min_agreement_frac"]
    conf = diag[diag.sky_direction != diag.sky_auto].sort_values("video_id")
    cand, low = conf[conf.agree >= thr], conf[conf.agree < thr]
    root = path_of(cfg, "arkit_3dod_root")
    ref = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "referability_stats.csv", dtype={"video_id": str}).set_index("video_id")
    out_dir = path_of(cfg, "reports") / "orientation_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_img, sheets, table = [], [], []
    for _, r in cand.iterrows():
        frames = arkit.list_frames(arkit.scene_files_3dod(root, r.video_id)["lowres_wide"], ".png")
        rng = scene_rng(cfg, f"arkit_{r.video_id}", "orientation_review")
        picks = sorted(rng.choice(len(frames), frames_per_scene + 1, replace=False).tolist())
        tiles = []
        for k, i in enumerate(picks):
            sky = r.sky_auto if k < frames_per_scene else r.sky_direction
            img = g.rotate_image_cw(cv2.imread(str(arkit.scene_files_3dod(root, r.video_id)["lowres_wide"] / frames[i][1])), g.SKY_TO_CW_QUARTERS[sky])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            side = int(256 * scale)
            canvas = np.full((side, side, 3), 255, np.uint8)
            y0, x0 = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
            canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            label = f"pose: {sky}" if k < frames_per_scene else f"metadata: {sky} (for contrast)"
            cv2.rectangle(canvas, (0, 0), (side, 18), (0, 0, 0) if k < frames_per_scene else (60, 60, 160), -1)
            cv2.putText(canvas, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(canvas, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        head = np.full((tiles[0].shape[0], 190, 3), 255, np.uint8)
        for j, t in enumerate([r.video_id, f"metadata: {r.sky_direction}", f"pose: {r.sky_auto}", f"agreement: {r.agree:.0%}",
                               f"referable: {int(ref.loc[r.video_id, 'referable'])}"]):
            cv2.putText(head, t, (6, 40 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if j == 0 else 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        rows_img.append(np.concatenate([head, *tiles], axis=1))
        sheet_no = len(rows_img) // per_sheet + (1 if len(rows_img) % per_sheet else 0)
        table.append(f"| {r.video_id} | {r.sky_direction} | {r.sky_auto} | {r.agree:.0%} | {int(ref.loc[r.video_id, 'referable'])} | sheet_{sheet_no:02d} | ☐ |")
        if len(rows_img) % per_sheet == 0:
            sheets.append(np.concatenate(rows_img[-per_sheet:], axis=0))
    if len(rows_img) % per_sheet:
        sheets.append(np.concatenate(rows_img[-(len(rows_img) % per_sheet):], axis=0))
    for k, sh in enumerate(sheets, 1):
        cv2.imwrite(str(out_dir / f"sheet_{k:02d}.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 88])
    L = ["# Uprighted thumbnails of orientation-conflict scenes (for manual confirmation)", "",
         f"Rule: when the metadata `sky_direction` disagrees with the pose-based estimate, the pose-based estimate is used if the per-frame agreement within the scene is >= {thr:.0%} "
         "and a reviewer has confirmed the uprighted thumbnails. Confirmed video_ids go into `orientation.confirmed_conflict_scenes` in `configs/build.yaml`; "
         "unconfirmed scenes are still removed as `ORIENTATION_CONFLICT`.", "",
         "How to read: one scene per row; the first 3 tiles are uprighted with the pose-based estimate (they should be upright: floor at the bottom, vertical door frames), the last one (red label) is rotated with the metadata, for contrast only.", "",
         f"{len(cand)} scenes to confirm; another {len(low)} have agreement < {thr:.0%} and are removed directly without confirmation ("
         + ", ".join(f"{v} ({a:.0%})" for v, a in zip(low.video_id, low.agree)) + ").", "",
         "| video_id | metadata | Pose-based estimate | Agreement | Referable objects | Sheet | Confirmed upright |", "|---|---|---|---|---|---|---|", *table, "",
         *[f"![](orientation_review/sheet_{k:02d}.jpg)" for k in range(1, len(sheets) + 1)]]
    (path_of(cfg, "reports") / "orientation_review.md").write_text("\n".join(L) + "\n")


def visibility_spotcheck(cfg, n=50, per_sheet=6, sheet_cols=3, tile_w=560):
    """Draw n random (object, node) pairs judged visible; one image per pair (frame + the object's box), each judged manually:
    "can the object be recognised in the image and its position roughly told?". Judgments go into reports/visibility_spotcheck_judgments.tsv."""
    import cv2
    from .common import scene_rng
    from .step56_questions import Scene
    out_dir = path_of(cfg, "reports") / "visibility_spotcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    # round 3: new seed; referable objects only; half of the pairs come from oracle evidence views used by questions
    groups = _load_groups(cfg)
    pool, oracle_keys = [], set()
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        sc = Scene(cfg, DATASET, vid)
        vis = np.load(work_dir(cfg, DATASET, vid) / "visibility.npz")
        ref = set(sc.referable)
        for grp in groups:
            if grp["scene_id"] != sc.scene_id:
                continue
            for q in grp["questions"]:
                for nid in q["oracle_nodes"] or []:
                    for o in q["evidence_objs"]:
                        if o in ref and sc.st(o, sc.node_index[nid]) == 2:
                            oracle_keys.add((vid, o, sc.node_index[nid]))
        for r, c in zip(*np.where(sc.state == 2)):
            oid = int(vis["obj_ids"][r])
            if oid in ref:
                pool.append((vid, oid, int(c), float(vis["rho"][r, c]), float(vis["completeness"][r, c]),
                             float(vis["area_frac"][r, c]), bool(vis["has_sensor_depth"][c])))
    rng = scene_rng(cfg, "pilot_visibility_spotcheck", "spotcheck_round3")
    from_oracle = [p for p in pool if p[:3] in oracle_keys]
    others = [p for p in pool if p[:3] not in oracle_keys]
    n_or = min(len(from_oracle), (n + 1) // 2)
    picks = [(p, "oracle") for p in (from_oracle[i] for i in rng.choice(len(from_oracle), n_or, replace=False))]
    picks += [(p, "random") for p in (others[i] for i in rng.choice(len(others), min(n - n_or, len(others)), replace=False))]
    picks = [picks[i] for i in rng.permutation(len(picks))]
    scenes, rows, tiles = {}, [], []
    for k, ((vid, oid, node, rho, comp, area, has_d), source) in enumerate(picks, 1):
        sc = scenes.setdefault(vid, Scene(cfg, DATASET, vid))
        fid = sc.nodes[node]
        img = _draw_tile(cfg, sc, fid, [oid], f"#{k:02d} {vid} {fid}  {sc.cat[oid]}")
        cv2.putText(img, f"rho={rho:.2f} completeness={comp:.2f} area={area:.3f} sensor_depth={'yes' if has_d else 'no'}",
                    (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"rho={rho:.2f} completeness={comp:.2f} area={area:.3f} sensor_depth={'yes' if has_d else 'no'}",
                    (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"pair_{k:02d}.jpg"), img)
        sc_ = tile_w / max(img.shape[:2])
        t = cv2.resize(img, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA)
        canvas = np.full((tile_w, tile_w, 3), 255, np.uint8)
        canvas[(tile_w - t.shape[0]) // 2:(tile_w - t.shape[0]) // 2 + t.shape[0], (tile_w - t.shape[1]) // 2:(tile_w - t.shape[1]) // 2 + t.shape[1]] = t
        tiles.append(cv2.copyMakeBorder(canvas, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append({"pair": f"{k:02d}", "video_id": vid, "frame_id": fid, "obj_id": oid, "name": sc.cat[oid],
                     "rho": round(rho, 2), "completeness": round(comp, 2), "area_frac": round(area, 4), "sensor_depth": has_d,
                     "source": source})
    for j in range(0, len(tiles), per_sheet):
        chunk = tiles[j:j + per_sheet]
        while len(chunk) % sheet_cols:
            chunk.append(np.full_like(tiles[0], 255))
        sheet = np.concatenate([np.concatenate(chunk[r:r + sheet_cols], axis=1) for r in range(0, len(chunk), sheet_cols)], axis=0)
        cv2.imwrite(str(out_dir / f"sheet_{j // per_sheet + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    pd.DataFrame(rows).to_csv(path_of(cfg, "work") / DATASET / "_step6" / "visibility_spotcheck_pairs.csv", index=False)
    print(f"pool: {len(pool)} referable visible pairs, {len(from_oracle)} in oracle evidence views; picked {n_or} oracle + {len(picks) - n_or} random")
    return rows, len(pool)


def visibility_spotcheck_report(cfg):
    """Read the manual judgment table (pair, ok, failure_type, note) and write reports/visibility_spotcheck.md."""
    pairs = pd.read_csv(path_of(cfg, "work") / DATASET / "_step6" / "visibility_spotcheck_pairs.csv", dtype={"video_id": str, "pair": str})
    j = pd.read_csv(path_of(cfg, "reports") / "visibility_spotcheck_judgments.tsv", sep="\t", dtype={"pair": str}, keep_default_na=False)
    df = pairs.merge(j, on="pair")
    n_ok = int((df.ok == "yes").sum())
    fails = df[df.ok != "yes"]
    L = ["# Visibility precision spot check", "",
         f"Round 3: new seed, referable objects only; {int((df.source == 'oracle').sum())} pairs from oracle evidence views used by questions and "
         f"{int((df.source == 'random').sum())} drawn at random from the remaining visible pairs, {len(df)} in total. "
         "Rules: completeness >= 0.30, area >= 0.02, >= 60% of the OBB projection inside the image, sensor-depth occlusion, floor points removed, overlapping vertices handled as nested/adjacent, keyframes only from frames with matching depth. "
         "Same criterion as before, **borderline cases count as \"no\"**. The two earlier rounds scored 45/50 and 47/50. "
         "One image per pair: the uprighted frame + the object's OBB (green). Each is judged on \"can a person recognise this object in the image and roughly tell where it is?\".", "",
         f"**Result: {n_ok} / {len(df)} ({n_ok / len(df):.0%}) judged \"yes\"**, target >= 95%: {'met' if n_ok / len(df) >= 0.95 else '**not met**'}. Thresholds were not tuned on this result.", ""]
    if len(fails):
        L += ["## Failure cases", "", "| # | Scene | Frame | Object | ρ | Completeness | Sensor depth | Type | Note |", "|---|---|---|---|---|---|---|---|---|",
              *[f"| {r.pair} | {r.video_id} | {r.frame_id} | {r.name} | {r.rho} | {r.completeness} | {'yes' if r.sensor_depth else 'no'} | {r.failure_type} | {r.note} |"
                for r in fails.itertuples()], "",
              "Failure types: " + ", ".join(f"{k} {v}" for k, v in Counter(fails.failure_type).most_common()) + ".", "",
              "Metrics of failed vs. passed cases (for reference only; thresholds were not adjusted on this basis):", "",
              "| | Count | Median completeness | Completeness range | Median area fraction | Area fraction range |", "|---|---|---|---|---|---|",
              *[f"| {name} | {len(d)} | {d.completeness.median():.2f} | {d.completeness.min():.2f}–{d.completeness.max():.2f} | "
                f"{d.area_frac.median():.3f} | {d.area_frac.min():.3f}–{d.area_frac.max():.3f} |"
                for name, d in (("failed", fails), ("passed", df[df.ok == "yes"]))], ""]
    L += ["## All judgments", "", "| # | Source | Scene | Frame | Object | ρ | Completeness | Area | Sensor depth | Recognisable | Note |", "|---|---|---|---|---|---|---|---|---|---|---|",
          *[f"| {r.pair} | {r.source} | {r.video_id} | {r.frame_id} | {r.name} | {r.rho} | {r.completeness} | {r.area_frac} | {'yes' if r.sensor_depth else 'no'} | "
            f"{'yes' if r.ok == 'yes' else '**no**'} | {r.note} |" for r in df.itertuples()], "",
          "Single images: `reports/visibility_spotcheck/pair_XX.jpg`; sheets of 6: `reports/visibility_spotcheck/sheet_XX.jpg`."]
    (path_of(cfg, "reports") / "visibility_spotcheck.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["gate_a", "step3", "step2", "step4", "step6", "gate_c", "orientation_review", "q10", "spotcheck_images", "spotcheck"])
    args = ap.parse_args()
    cfg = load_config()
    {"gate_a": gate_a_report, "step3": step3_report, "step2": step2_report, "step4": step4_report,
     "step6": step6_report, "gate_c": gate_c_report, "orientation_review": orientation_review, "q10": q10_unanswerable_report,
     "spotcheck_images": visibility_spotcheck, "spotcheck": visibility_spotcheck_report}[args.which](cfg)


if __name__ == "__main__":
    main()
