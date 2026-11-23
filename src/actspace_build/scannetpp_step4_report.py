"""ScanNet++ Step 4 report and visibility spot check.

- report: reports/scannetpp_step4_graph_visibility.md (view graph, visibility distribution, magenta patches (S10), mirrors (S8),
  notes on merging main)
- spotcheck_images: with a new seed, sample 50 visible (referable object, node) pairs, evenly over scenes; at least 3 pairs near the
  mirror in 31a2c91c43 and 3 near the magenta patches in 5ee7c22ba0. One image per pair (original image + object box + name), written
  to reports/scannetpp_visibility_spotcheck/; the sample table goes to work/scannetpp/_step4/spotcheck_pairs.csv
- spotcheck: reads the per-image judgments reports/scannetpp_visibility_spotcheck_judgments.tsv (pair, ok, failure_type, note) and
  writes reports/scannetpp_visibility_spotcheck.md

Usage: python -m actspace_build.scannetpp_step4_report report|spotcheck_images|spotcheck
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from . import geometry as g
from .common import load_config, path_of, read_json, scene_rng
from .scannetpp import scale_intrinsics, sp_cfg, sp_root, sp_work
from .scannetpp_step4 import SceneCtx, apply_s12, face_corners
from .step1_arkit import draw_obbs
from .step4_graph_visibility import ACTIONS, STATE_VISIBLE

SPOT_N = 50
SPOT_SEED = ("scannetpp_pilot_visibility_spotcheck", "spotcheck_scannetpp_round2")  # round 2 (new seed); round 1 used spotcheck_scannetpp_round1
ROUND1_TARGETS = {"shape": ["21", "22"], "sharp": ["14", "39"]}  # round-1 failure pairs the two new conditions are meant to remove
SPOT_MIN_NEAR = 3
MIRROR_SCENE, MAGENTA_SCENE = "31a2c91c43", "5ee7c22ba0"
GREEN, RED, YELLOW = (0, 220, 0), (0, 0, 255), (0, 220, 255)


def load_scene(cfg, sid):
    wd = sp_work(cfg) / sid
    return (read_json(wd / "objects.json"), dict(np.load(wd / "visibility.npz")), read_json(wd / "step4_summary.json"),
            read_json(wd / "view_graph.json"), {f["frame_id"]: f for f in read_json(wd / "frames_all.json")})


def base_image(cfg, f, out_w):
    img = cv2.imread(str(sp_root(cfg) / f["rel_path"]))
    W0, H0 = f["image_size"]
    img = cv2.resize(img, (out_w, int(round(H0 * out_w / W0))), interpolation=cv2.INTER_AREA)
    K = scale_intrinsics(np.asarray(f["K"]), (W0, H0), (img.shape[1], img.shape[0]))
    return img, K, np.asarray(f["T_wc"])


def put(img, txt, org, scale=0.6, color=(255, 255, 255)):
    """Write text once on a black strip (outlined text drawn twice with different thicknesses has different widths and misaligns)."""
    (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 5), (x + w + 4, y + base + 3), (0, 0, 0), -1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def box_of(o, label):
    b = o["obb"]
    return {"center": np.asarray(b["center"]), "axes": np.asarray(b["axes"]), "size": np.asarray(b["size"]), "label": label}


# ---------------------------------------------------------------- top-down view-graph plot
def graph_plot(cfg, sid, graph, frames, kf_ids, out_path, px=900):
    P = np.array([frames[i]["position"][:2] for i in kf_ids])
    lo, hi = P.min(0) - 0.5, P.max(0) + 0.5
    s = (px - 40) / float(max(hi - lo))
    W, H = int((hi - lo)[0] * s) + 40, int((hi - lo)[1] * s) + 40
    to = lambda p: (int(20 + (p[0] - lo[0]) * s), int(H - 20 - (p[1] - lo[1]) * s))
    img = np.full((H, W, 3), 255, np.uint8)
    keep = set(graph["nodes"])
    pos = {i: frames[i]["position"][:2] for i in kf_ids}
    col = {"forward": (200, 120, 0), "backward": (200, 120, 0), "left": (0, 150, 0), "right": (0, 150, 0),
           "turn_left": (0, 0, 220), "turn_right": (0, 0, 220)}
    for a in ACTIONS:
        for n, e in graph["edges"].items():
            if e[a]:
                cv2.line(img, to(pos[n]), to(pos[e[a]]), col[a], 1, cv2.LINE_AA)
    for i in kf_ids:
        p = to(pos[i])
        y = np.radians(frames[i]["yaw_deg"])
        cv2.arrowedLine(img, p, (int(p[0] + 12 * np.cos(y)), int(p[1] - 12 * np.sin(y))), (60, 60, 60) if i in keep else (0, 0, 255), 1, tipLength=0.4)
        cv2.circle(img, p, 3, (0, 0, 0) if i in keep else (0, 0, 255), -1)
    put(img, f"{sid}: {len(keep)}/{len(kf_ids)} nodes kept (red = dropped). edges: blue fwd/back, green left/right, red turns", (8, 18), 0.45, (0, 0, 0))
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- S8 / S10 example images
def explain_image(cfg, ctx, fid, k, title, out_path, out_w=1280):
    """Per-point image of one (object, node) pair: green = finally visible, red = on a magenta patch, yellow = occluded by a mirror,
    grey = occluded by mesh / sensor depth; magenta regions are darkened and mirrors drawn as blue boxes."""
    f = ctx.frames[fid]
    r = ctx.node(fid, only=[k], keep_points=True)[0]
    img, K, T = base_image(cfg, f, out_w)
    s = img.shape[1] / f["image_size"][0]
    m = cv2.resize(r["mag"].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[m] = (0.4 * img[m] + [60, 0, 60]).astype(np.uint8)
    for fc in ctx.mirrors:
        uv, z = g.project(face_corners(fc), K, T)
        if (z > 0.05).all():
            cv2.polylines(img, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (255, 120, 0), 2, cv2.LINE_AA)
    for (u, v), vb, mg, mr in zip(r["uv_in"] * s, r["vis_base"], r["on_mag"], r["occ_mir"]):
        c = (150, 150, 150) if not vb else RED if mg else YELLOW if mr else GREEN
        cv2.circle(img, (int(round(u)), int(round(v))), 2, c, -1)
    name = ctx.objs[k]["category"]
    st = lambda x: ["invisible", "partial", "visible"][x[0]]
    put(img, title, (8, 24))
    put(img, f"{name} #{ctx.objs[k]['obj_id']}: base {st(r['base'])} -> final {st(r['final'])}; "
             f"pts on magenta {r['n_magenta']}, behind mirror {r['n_mirror']}, in frustum {r['n_in']}", (8, img.shape[0] - 12), 0.55)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- report
def round1_metrics(cfg, S) -> pd.DataFrame:
    """Values of the two S12 metrics for the 50 round-1 spot-check pairs (references computed from state_pre_s12, the state before
    the conditions), with the round-1 judgments."""
    d = sp_work(cfg) / "_step4"
    pairs = pd.read_csv(d / "spotcheck_pairs_round1.csv", dtype={"pair": str})
    j = pd.read_csv(d / "spotcheck_judgments_round1.tsv", sep="\t", dtype={"pair": str}, keep_default_na=False)
    df = pairs.merge(j[["pair", "ok", "failure_type"]], on="pair")
    sp = sp_cfg(cfg)
    ratios = {}
    for sid in df.scene_id.unique():
        vis = S[sid][1]
        for ref in ("object_nodes", "scene_frames"):
            _, r, _, _ = apply_s12(vis["state_pre_s12"], vis["min_rect_short_px"], vis["region_lapvar"], vis["frame_lapvar"],
                                   {**sp, "vis_sharpness_ref": ref, "vis_min_short_side_px": None, "vis_min_sharpness_ratio": None})
            ratios[(sid, ref)] = r
    rows = []
    for r in df.itertuples():
        vis = S[r.scene_id][1]
        o = int(np.where(vis["obj_ids"] == r.obj_id)[0][0])
        c = int(np.where(vis["frame_ids"] == r.frame_id)[0][0])
        assert vis["state_pre_s12"][o, c] == STATE_VISIBLE, (r.pair, r.scene_id, r.obj_id, r.frame_id)
        rows.append({"pair": r.pair, "scene_id": r.scene_id, "name": r.name, "obj_id": r.obj_id, "frame_id": r.frame_id, "ok": r.ok,
                     "short_px": float(vis["min_rect_short_px"][o, c]), "region_lapvar": float(vis["region_lapvar"][o, c]),
                     "ratio_object": float(ratios[(r.scene_id, "object_nodes")][o, c]), "ratio_scene": float(ratios[(r.scene_id, "scene_frames")][o, c]),
                     "frame_ratio": float(vis["frame_lapvar"][c] / np.median(vis["frame_lapvar"]))})
    return pd.DataFrame(rows)


def s12_section(cfg, S) -> list[str]:
    sp = sp_cfg(cfg)
    pilots = sp["pilot_scenes"]
    ts, tr, ref = sp["vis_min_short_side_px"], sp["vis_min_sharpness_ratio"], sp["vis_sharpness_ref"]
    m = round1_metrics(cfg, S)
    m.to_csv(sp_work(cfg) / "_step4" / "s12_round1_metrics.csv", index=False)
    rcol = "ratio_object" if ref == "object_nodes" else "ratio_scene"
    L = ["### 2.2 Extra visibility conditions (S12, ScanNet++ only)", "",
         "After the standard decision (including S8 / S10), visible entries that fail either of the following are downgraded to partial:",
         f"1. **Shape**: short side of the minimum-area bounding rectangle of the convex hull of projected visible points (original 1920x1440 pixels) >= `vis_min_short_side_px` = **{ts}**;",
         f"2. **Regional sharpness**: Laplacian variance inside that convex hull (magenta pixels excluded) / reference >= `vis_min_sharpness_ratio` = **{tr}**. "
         f"Reference `vis_sharpness_ref` = **{ref}** ({'median regional Laplacian variance of the object over its visible nodes' if ref == 'object_nodes' else 'median whole-image Laplacian variance over all nodes of the scene'}).", "",
         "#### Threshold selection (50 round-1 spot-check pairs)", "",
         "Values of the two metrics on the 50 round-1 pairs (computed from the state before the conditions). Two sharpness references are given: \"object ref\" = median regional value over the object's visible nodes; "
         "\"scene ref\" = median whole-image value over all nodes of the scene; whole-image sharpness / scene median is listed for comparison only (not a regional metric). "
         f"Bold = the 4 pairs to remove (shape: {', '.join('#' + p for p in ROUND1_TARGETS['shape'])}; sharpness: {', '.join('#' + p for p in ROUND1_TARGETS['sharp'])}).", "",
         "| # | Scene | Object | Round-1 judgment | Short side (px) | Regional Laplacian variance | Sharpness / object ref | Sharpness / scene ref | Whole image / scene median | With chosen thresholds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    targets = set(ROUND1_TARGETS["shape"] + ROUND1_TARGETS["sharp"])
    for r in m.sort_values("short_px").itertuples():
        hit = []
        if ts is not None and r.short_px < ts:
            hit.append("shape downgrade")
        if tr is not None and getattr(r, rcol) < tr:
            hit.append("sharpness downgrade")
        b = "**" if r.pair in targets else ""
        L.append(f"| {b}{r.pair}{b} | `{r.scene_id}` | {r.name} #{r.obj_id} | {'yes' if r.ok == 'yes' else 'no'} | {r.short_px:.0f} | {r.region_lapvar:.1f} | "
                 f"{r.ratio_object:.2f} | {r.ratio_scene:.2f} | {r.frame_ratio:.2f} | {', '.join(hit) or '—'} |")
    shp_t = m[m.pair.isin(ROUND1_TARGETS["shape"])]
    shp_t_max = shp_t.short_px.max()
    ok = m[m.ok == "yes"]
    L += ["", "Selection (as conservative as possible = the smallest threshold that just removes the target pairs, then check which \"yes\" pairs would also be removed):", "",
          f"- Shape: the short sides of #21 and #22 are {', '.join(f'{v:.0f}' for v in shp_t.short_px)} px; the smallest short side among \"yes\" pairs is "
          f"{ok.short_px.min():.0f} px (#{ok.loc[ok.short_px.idxmin(), 'pair']} {ok.loc[ok.short_px.idxmin(), 'name']}). "
          f"Chosen: {ts} px (> {shp_t_max:.0f})."]
    for col, label in (("ratio_object", "object ref"), ("ratio_scene", "scene ref")):
        t = m[m.pair.isin(ROUND1_TARGETS["sharp"])][col]
        thr = float(t.max())
        fp = ok[ok[col] <= thr]
        L.append(f"- Sharpness ({label}): #14 and #39 are {', '.join(f'{v:.2f}' for v in t)}; removing both needs a threshold > {thr:.2f}, "
                 f"which would also remove {len(fp)} \"yes\" pairs" + (f" ({', '.join('#' + p + ' ' + n for p, n in zip(fp.pair, fp.name))})" if len(fp) else "") + ".")
    fp_shape = ok[ok.short_px < ts] if ts is not None else ok.iloc[:0]
    fp_sharp = ok[ok[rcol] < tr] if tr is not None else ok.iloc[:0]
    gone = [p for p in targets if (ts is not None and float(m[m.pair == p].short_px.iloc[0]) < ts) or (tr is not None and float(m[m.pair == p][rcol].iloc[0]) < tr)]
    L += [f"- With the chosen thresholds, {len(gone)} of the 4 round-1 target pairs are removed ({', '.join('#' + p for p in sorted(gone))}); \"yes\" pairs also removed: shape {len(fp_shape)}"
          + (f" ({', '.join('#' + p + ' ' + n for p, n in zip(fp_shape.pair, fp_shape.name))})" if len(fp_shape) else "")
          + f", sharpness {len(fp_sharp)}" + (f" ({', '.join('#' + p + ' ' + n for p, n in zip(fp_sharp.pair, fp_sharp.name))})" if len(fp_sharp) else "") + ".",
          "- The thresholds were fitted on the round-1 sample, so 50 new pairs were drawn with a new seed for validation (`scannetpp_visibility_spotcheck.md`); round 1 does not count towards the pass rate.", ""]
    # effect
    L += ["#### Effect on the 5 scenes", "",
          "| Scene | Visible pairs (before) | Shape downgrades | Sharpness downgrades | Failing both | Visible pairs (after) | Of which referable objects: before -> after |", "|---|---|---|---|---|---|---|"]
    for sid in pilots:
        objs, vis, sm = S[sid][0], S[sid][1], S[sid][2]
        ref_ = np.array([o["referable"] for o in objs])
        s12 = sm["s12"]
        L.append(f"| `{sid}` | {s12['visible_before']} | {s12['down_shape']} | {s12['down_sharp']} | {s12['down_both']} | {s12['visible_after']} | "
                 f"{int((vis['state_pre_s12'] == STATE_VISIBLE)[ref_].sum())} -> {int((vis['state'] == STATE_VISIBLE)[ref_].sum())} |")
    became0 = [(sid, o) for sid in pilots for o in S[sid][2]["objects"] if o["referable"] and o["num_visible_nodes_pre_s12"] > 0 and o["num_visible_nodes"] == 0]
    L += ["", "Per-object changes of the visible-node count are in the table of section 2.1 (\"before -> after\"). **Referable objects that become visible at no node** because of the two new conditions: "
          + (", ".join(f"`{sid}` {o['category']} #{o['obj_id']}" for sid, o in became0) if became0 else "**none**") + ".", ""]
    # objects usable for Step 5 cluster sampling
    L += ["### 2.3 Objects usable for Step 5 cluster sampling", "",
          "Per scene, the number of referable objects that are visible at one or more nodes (after the two new conditions). Whether to add more scene sources is decided after the C2 pilot question generation.", "",
          "| Scene | Referable objects (Step 3, blacklist removed) | Visible at >= 1 node (before S12) | Visible at >= 1 node (after S12, usable in Step 5) |", "|---|---|---|---|"]
    tot = [0, 0, 0]
    for sid in pilots:
        ob = [o for o in S[sid][2]["objects"] if o["referable"]]
        a, b, c = len(ob), sum(o["num_visible_nodes_pre_s12"] > 0 for o in ob), sum(o["num_visible_nodes"] > 0 for o in ob)
        tot = [tot[0] + a, tot[1] + b, tot[2] + c]
        L.append(f"| `{sid}` | {a} | {b} | **{c}** |")
    L += [f"| **Total** | {tot[0]} | {tot[1]} | **{tot[2]}** |", ""]
    return L


def write_report(cfg) -> None:
    sp = sp_cfg(cfg)
    vc, vg = cfg["visibility"], cfg["view_graph"]
    pilots = sp["pilot_scenes"]
    rep = path_of(cfg, "reports")
    img_dir = rep / "scannetpp_step4"
    img_dir.mkdir(parents=True, exist_ok=True)
    types = __import__("json").load(open(sp_root(cfg) / "metadata" / "scene_types.json"))
    S = {sid: load_scene(cfg, sid) for sid in pilots}
    L = ["# ScanNet++ Step 4: view graph and per-frame visibility (5 pilot scenes)", "",
         "> Branch `scannetpp`. Generated by `python -m actspace_build.scannetpp_step4_report report`. See docs/data_construction.md. "
         "This round stops after Step 4 and the visibility spot check; **Step 5 has not started**. Spot-check results are in `scannetpp_visibility_spotcheck.md`.", "",
         "**Round 2 (rerun)**: two objects blacklisted (`31a2c91c43` window #25, `5ee7c22ba0` whiteboard #33); "
         "window #25 is treated as a mirror, and a list of suspected mirrors was added (section 4.1); two extra conditions (shape, regional sharpness; section 2.2) were added; "
         "objects usable in Step 5 are in section 2.3. The round-1 spot check (43 / 50) was used only to choose thresholds; round 2 redraws with a new seed.", ""]

    # ---- 0. merging main
    L += ["## 0. Merging main before Step 4", "",
          "After the GATE B changes were committed, the latest committed main was merged into `scannetpp`; **no conflicts**, and all 99 unit tests passed after the merge.",
          "Changes brought in (15 commits on main, all on the ARKit side):", "",
          "| Change | Files | Effect on ScanNet++ Steps 4-6 |", "|---|---|---|",
          "| Size references permanently disabled: `objects.size_ref_enabled: false` (applies to both ARKit and ScanNet++); `step3_objects.assign_references` returns early when the switch is off | "
          "`configs/build.yaml`, `step3_objects.py`, `reports.py` | No effect on Step 4. ScanNet++ already has its own `scannetpp.size_reference_enabled: false`, consistent with it; "
          "ScanNet++ Step 3 does not call the shared `assign_references`. |",
          "| Steps 5-6: when loading a `Scene`, objects in old outputs whose `ref_kind` is not unique are treated as non-referable; answerable T4 questions get `path_views` (up to 4 intermediate nodes sampled evenly along the shortest path) | "
          "`step56_questions.py` | Used in Steps 5-6 (shared code). ScanNet++ `objects.json` only has unique references, so the first part changes nothing; `path_views` is produced by the shared code. |",
          "| Optional group constraint: `groups.require_constraint: false`; no collective drop when a constraint is lost during debiasing: `debias.keep_group_on_constraint_loss: true`; `debias.output_version: arkit_only_v0.11` | "
          "`configs/build.yaml`, `step8_debias.py` | The group generation rules of Steps 5-6 change accordingly (shared config) and apply to ScanNet++; to be described in the Step 5-6 report. |",
          "| Frozen-version protection (`common.assert_not_frozen`, `freeze.py`), GATE D round 2-3 pages and reports, integrity check (T4 path_views), ARKit blacklist | "
          "`common.py`, `freeze.py`, `gate_d*.py`, `integrity.py`, `configs/object_blacklist.yaml` | No effect (new functions / ARKit only). |",
          "| Shared `step4_graph_visibility.py`, `geometry.py`, `step2_keyframes.py`, construction guide | — | **Unchanged**; ScanNet++ Step 4 reuses their functions directly. |", ""]

    # ---- 1. view graph
    L += ["## 1. View graph", "",
          f"- Nodes = Step 2 keyframes (camera centre p, yaw). Action semantics as specified: forward / backward / left / right require {vg['move_min_m']} m <= |d| <= {vg['move_max_m']} m, "
          f"|Δyaw| <= {vg['move_max_dyaw_deg']}° (bearing boundaries ±{vg['front_half_angle_deg']}° / ±{vg['back_min_angle_deg']}°), taking the smallest |d|; turn_left / turn_right require |d| <= {vg['turn_max_translation_m']} m, "
          f"{vg['turn_min_deg']}° <= |Δyaw| <= {vg['turn_max_deg']}°, taking the one closest to {vg['turn_target_deg']}°.",
          f"- Only the largest strongly connected component is kept; if the kept fraction is < {vg['min_scc_ratio']:.0%}, the distance limit is relaxed to {vg['move_max_m_relaxed']} m and retried once; if still insufficient the scene is dropped (`GRAPH_DISCONNECTED`).",
          "- Implementation: reuses the shared `step4_graph_visibility.build_view_graph` (unchanged). yaw is the horizontal bearing of the camera optical axis (OpenCV camera +z), written to `frames_all.json` by Step 1.", "",
          "| Scene | Type | Keyframes | Nodes (largest SCC) | Kept fraction | Relaxed | forward | backward | left | right | turn_left | turn_right | Result |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid in pilots:
        sm = S[sid][2]
        e = sm["edges_per_action"]
        L.append(f"| `{sid}` | {types.get(sid)} | {sm['num_keyframes']} | **{sm['num_graph_nodes']}** | {sm['scc_ratio']:.0%} | {'yes' if sm['relaxed'] else 'no'} | "
                 + " | ".join(str(e[a]) for a in ACTIONS) + f" | {'pass' if sm['ok'] else 'GRAPH_DISCONNECTED'} |")
    n_bad = sum(not S[s][2]["ok"] for s in pilots)
    L += ["", f"Scenes dropped for a disconnected view graph: {n_bad} / {len(pilots)}.", "",
          "Top-down view graphs (black dots = kept nodes, red dots = keyframes dropped as outside the largest SCC, short arrows = heading; blue lines = forward/backward moves, green lines = left/right moves, red lines = in-place turns):", ""]
    for sid in pilots:
        objs, vis, sm, graph, frames = S[sid]
        kf_ids = read_json(sp_work(cfg) / sid / "keyframes.json")["keyframe_ids"]
        graph_plot(cfg, sid, graph, frames, kf_ids, img_dir / f"{sid}_graph.jpg")
        L += [f"![{sid}](scannetpp_step4/{sid}_graph.jpg)", ""]

    # ---- 2. visibility
    L += ["## 2. Per-frame visibility", "",
          f"- Standard rules: object point sets are downsampled to at most {vc['max_points_per_object']} points; in-frustum points are tested for occlusion: mesh ray hit distance >= point distance - {vc['occlusion_tolerance_m']} m, "
          f"and point depth <= sensor depth + {vc['sensor_occlusion_tolerance_m']} m (pixels with invalid sensor depth use the mesh only); a point is visible only if both say it is unoccluded.",
          f"- visible: ρ >= {vc['visible_min_ratio']}, area >= {vc['visible_min_area_frac']}, visible points >= {vc['visible_min_points']}, completeness >= {vc['visible_min_completeness']}, "
          f"in-frame fraction >= {vc['visible_min_in_frame_frac']}; invisible: ρ < {vc['invisible_max_ratio']} or no in-frustum points; otherwise partial. Reuses the shared `visibility_for_node` / `classify` / `obb_in_frame_frac` (unchanged).",
          "- **Sensor depth**: every ScanNet++ frame has a depth image with the same frame number (256x192 uint16 millimetre png decoded from depth.bin in Step 1), read directly by frame number without time matching. "
          "Depth and RGB share the field of view and intrinsics scale with resolution (verified at GATE A); the depth is nearest-neighbour upsampled to RGB resolution (1920x1440) for lookup, equivalent to scaling the intrinsics. "
          f"Valid range as in GATE A: {cfg['gate_a']['depth_min_valid_m']}-{cfg['gate_a']['depth_max_valid_m']} m. The ARKit reader and its behaviour are unchanged.",
          "- **Magenta patches (S10, option b)**: pure magenta RGB pixels (R > 200, B > 200, G < 60, same definition as the Step 2 statistics) dilated by "
          f"{sp['magenta_dilate_px']} pixels form a mask; object points projecting onto the mask are always invisible. `rgb_mask.mkv` was not downloaded; only colour detection is used.",
          "- **Mirrors (S8)**: for objects whose label contains mirror (excluding mirror lamp), the thinnest box face facing the room (the side of the mean keyframe camera position) is an opaque occluder; "
          f"points whose line of sight crosses that face and that lie more than {vc['occlusion_tolerance_m']} m beyond it are always occluded (whatever the sensor depth says). The mirror's own points are unaffected. See section 4.", ""]
    L += ["| Scene | Nodes | Objects | Nodes with sensor depth | Fraction of in-frustum points without valid depth | Visible pairs (without S8/S10) | Visible pairs (final) | Referable objects |",
          "|---|---|---|---|---|---|---|---|"]
    for sid in pilots:
        objs, vis, sm, _, _ = S[sid]
        L.append(f"| `{sid}` | {sm['num_graph_nodes']} | {len(objs)} | {sm['nodes_with_sensor_depth']} | {sm['infrustum_points_without_valid_depth_frac']:.1%} | "
                 f"{sm['visible_pairs']['base']} | {sm['visible_pairs']['final']} | {sum(o['referable'] for o in objs)} |")
    L += ["", "### 2.1 Visible-node counts of referable objects", "",
          "At how many nodes each referable object is visible / partial (total node counts in the table above).", "",
          "| Scene | Object | Visible nodes (before -> after the S12 conditions) | Partial nodes | Invisible nodes | Visible fraction of nodes |", "|---|---|---|---|---|---|"]
    allv = []
    for sid in pilots:
        objs, vis, sm, _, _ = S[sid]
        N = sm["num_graph_nodes"]
        for o in sm["objects"]:
            if o["referable"]:
                allv.append(o["num_visible_nodes"])
                pre = o["num_visible_nodes_pre_s12"]
                arrow = f"{pre} -> **{o['num_visible_nodes']}**" if pre != o["num_visible_nodes"] else str(pre)
                if pre > 0 and o["num_visible_nodes"] == 0:
                    arrow += " (**dropped to 0**)"
                L.append(f"| `{sid}` | {o['category']} #{o['obj_id']} | {arrow} | {o['num_partial_nodes']} | {o['num_invisible_nodes']} | "
                         f"{o['num_visible_nodes'] / N:.0%} |")
    a = np.array(allv)
    L += ["", f"Distribution ({len(a)} referable objects): min {a.min()}, P25 {np.percentile(a, 25):.0f}, median {np.median(a):.0f}, P75 {np.percentile(a, 75):.0f}, max {a.max()}; "
          f"{int((a == 0).sum())} with 0 visible nodes, {int((a < 3).sum())} with < 3.", ""]
    allo = np.array([o["num_visible_nodes"] for sid in pilots for o in S[sid][2]["objects"]])
    L += [f"All {len(allo)} annotated objects (including non-referable): median visible-node count {np.median(allo):.0f}; {int((allo == 0).sum())} with 0 ({(allo == 0).mean():.0%}, mostly small objects, "
          "occluded objects or objects seen only in non-keyframes).", ""]
    L += s12_section(cfg, S)

    # ---- 3. S10
    L += ["## 3. Magenta patches (S10)", "",
          "\"Affected\" = at least 1 originally visible point of the (object, node) pair falls on the magenta mask; \"downgraded\" = visible without S10 and not visible with it.", "",
          "| Scene | Nodes | Nodes with patches | Affected (object, node) pairs | Of which referable objects | Downgraded from visible | Of which referable objects |", "|---|---|---|---|---|---|---|"]
    mag_examples = []
    for sid in pilots:
        objs, vis, sm, _, _ = S[sid]
        ref = np.array([o["referable"] for o in objs])
        aff = vis["n_magenta"] > 0
        down = (vis["state_base"] == STATE_VISIBLE) & (vis["state_no_mirror"] != STATE_VISIBLE)
        nodes_m = int((vis["magenta_near_px"] > 0).any(0).sum())
        L.append(f"| `{sid}` | {sm['num_graph_nodes']} | {nodes_m} | {int(aff.sum())} | {int(aff[ref].sum())} | {int(down.sum())} | {int(down[ref].sum())} |")
        for r, c in zip(*np.where(down)):
            mag_examples.append((sid, int(r), int(c), bool(ref[r]), int(vis["n_magenta"][r, c])))
    L += ["", "Downgraded pairs:", ""]
    if mag_examples:
        L += ["| Scene | Object | Referable | Node | Points on patches | With S10 |", "|---|---|---|---|---|---|"]
        for sid, r, c, rf, nm in mag_examples:
            objs, vis = S[sid][0], S[sid][1]
            L.append(f"| `{sid}` | {objs[r]['category']} ({objs[r]['source_label']}) #{objs[r]['obj_id']} | {'yes' if rf else 'no'} | {vis['frame_ids'][c]} | {nm} | "
                     f"{['invisible', 'partial', 'visible'][int(vis['state'][r, c])]} |")
    else:
        L.append("None.")
    # 3 examples: downgraded pairs first (referable first), then the most affected pairs
    ctxs = {}
    picks = sorted(mag_examples, key=lambda t: (not t[3], -t[4]))
    seen_obj = set()
    chosen = []
    for t in picks:
        if (t[0], t[1]) not in seen_obj:
            chosen.append(t)
            seen_obj.add((t[0], t[1]))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        vis = S[MAGENTA_SCENE][1]
        for r, c in sorted(zip(*np.where(vis["n_magenta"] > 0)), key=lambda rc: -vis["n_magenta"][rc]):
            if (MAGENTA_SCENE, int(r)) not in seen_obj:
                chosen.append((MAGENTA_SCENE, int(r), int(c), S[MAGENTA_SCENE][0][r]["referable"], int(vis["n_magenta"][r, c])))
                seen_obj.add((MAGENTA_SCENE, int(r)))
            if len(chosen) == 3:
                break
    L += ["", "Examples (green dots = finally visible, red dots = invisible because on a magenta patch, grey dots = occluded by mesh / sensor depth; magenta regions darkened):", ""]
    for j, (sid, r, c, rf, nm) in enumerate(chosen, 1):
        ctx = ctxs.setdefault(sid, SceneCtx(cfg, sid))
        fid = str(S[sid][1]["frame_ids"][c])
        fn = f"s10_{j}_{sid}_{r}_{fid}.jpg"
        explain_image(cfg, ctx, fid, r, f"S10 example {j}: {sid} {fid}", img_dir / fn)
        L += [f"![S10 {j}](scannetpp_step4/{fn})", ""]

    # ---- 4. S8
    L += ["## 4. Mirrors (S8)", ""]
    labelled = {sid: [o for o in S[sid][0] if "mirror" in o["source_label"].lower() and o["source_label"].lower() not in sp["mirror_label_exclude"]] for sid in pilots}
    L += ["- **Finding mirrors by label: no object in the 5 pilot scenes has a label containing mirror** (5 objects labelled mirror in the 50 scenes, none in the pilot; plus 1 mirror lamp, which is a lamp and not treated as a mirror).",
          f"- The mirror in `{MIRROR_SCENE}` (bathroom) is labelled **`window` #25**: the box shows the reflection of the door on the opposite wall and the jeans hanging on it (compare with the best view of door #49 below); "
          "the sensor depth inside the box measures the 1.7-2.2 m \"virtual image\" distance; the mesh has a hole at the mirror and the annotated points are only on the frame. "
          "It is listed in `scannetpp.mirror_object_overrides` and treated as a mirror.",
          "- `window` #25 is a label error and is blacklisted (`configs/object_blacklist_scannetpp.yaml`, recorded as `BLACKLISTED`), so it is no longer referable.", ""]
    objs, vis, sm, _, frames = S[MIRROR_SCENE]
    ms = sm["mirror"]
    ref = np.array([o["referable"] for o in objs])
    ch = vis["state_no_mirror"] != vis["state"]
    near = vis["mirror_near"] > 0
    L += [f"Objects treated as mirrors in `{MIRROR_SCENE}`: {', '.join(f'{l} #{i}' for i, l in zip(ms['mirror_obj_ids'], ms['mirror_labels']))}.", "",
          "| Metric | All objects | Referable objects |", "|---|---|---|",
          f"| (object, node) pairs with points occluded by a mirror | {int((vis['n_mirror'] > 0).sum())} | {int((vis['n_mirror'] > 0)[ref].sum())} |",
          f"| Pairs whose state changes as a result | {int(ch.sum())} | {int(ch[ref].sum())} |",
          f"| Of which downgraded from visible | {ms['visible_downgraded']} | {int(((vis['state_no_mirror'] == STATE_VISIBLE) & (vis['state'] != STATE_VISIBLE))[ref].sum())} |",
          f"| (object, node) pairs with a mirror in the same image (\"near mirror\", for spot checks) | {int(near.sum())} | {int(near[ref].sum())} |", ""]
    vb = vis["state_base"] == STATE_VISIBLE
    L += ["Note: the mesh has a hole at the mirror and there are no annotated object points behind it, so there are 0 object points \"whose line of sight crosses the mirror and that lie beyond it\"; "
          "sensor depth looking into the mirror (too far) can only make a point \"not occluded\", never turn visible into invisible. The actual effect of mirrors on visibility is therefore small. "
          "Mirrors cause another problem outside the visibility decision: reflections make objects that are actually behind the camera \"appear\" in the image (e.g. the door and the jeans), "
          "so objects judged invisible may be seen in a mirror in the image. See section 5. "
          "No pair changes state in the pilot; the two examples below show objects judged visible with a mirror in the same image (referable objects first, then others).", ""]
    # 2 examples: pairs whose state changed first; otherwise referable objects near the mirror
    ctx = ctxs.setdefault(MIRROR_SCENE, SceneCtx(cfg, MIRROR_SCENE))
    cand = [(int(r), int(c)) for r, c in zip(*np.where(ch))]
    cand += [(int(r), int(c)) for r, c in sorted(zip(*np.where(vis["n_mirror"] > 0)), key=lambda rc: -vis["n_mirror"][rc]) if (int(r), int(c)) not in cand]
    not_mirror = ~np.isin(np.arange(len(objs)), ms["mirror_obj_ids"])[:, None]
    cand += [(int(r), int(c)) for r, c in zip(*np.where(near & ref[:, None] & not_mirror & vb))]  # referable objects near the mirror
    cand += [(int(r), int(c)) for r, c in zip(*np.where(near & not_mirror & vb))]  # if still short, other objects near the mirror
    used, ex = set(), []
    for r, c in cand:
        if r not in used:
            ex.append((r, c))
            used.add(r)
        if len(ex) == 2:
            break
    L += ["Examples (blue box = mirror occluder face, yellow dots = occluded by the mirror, green dots = finally visible, grey dots = occluded by mesh / sensor depth):", ""]
    for j, (r, c) in enumerate(ex, 1):
        fid = str(vis["frame_ids"][c])
        fn = f"s8_{j}_{MIRROR_SCENE}_{r}_{fid}.jpg"
        explain_image(cfg, ctx, fid, r, f"S8 example {j}: {MIRROR_SCENE} {fid}", img_dir / fn)
        L += [f"![S8 {j}](scannetpp_step4/{fn})", ""]
    L += ["### 4.1 Suspected mirrors (to be confirmed manually for the full run)", "",
          f"For flat objects (thinnest box edge <= {sp['mirror_scan_max_thickness_m']} m, middle edge >= {sp['mirror_scan_min_face_m']} m; walls / floors / ceilings / door frames excluded), within the room-facing box face, "
          f"the fraction of face pixels (not hidden by objects in front) whose sensor depth is more than {sp['mirror_scan_far_m']} m beyond the face; only nodes where the face covers >= {sp['mirror_scan_min_px']} valid depth pixels count. "
          f"Objects with >= {sp['mirror_scan_min_nodes']} counted nodes and a fraction >= {sp['mirror_scan_min_far_frac']:.0%} are listed as suspected (mirrors, or transparent glass: windows, glass doors / partitions are listed too). Report only; states are not changed.", "",
          "| Scene | Object | Box size (m, ascending) | Counted nodes | \"Farther\" pixel fraction | Suspected |", "|---|---|---|---|---|---|"]
    n_sus = 0
    for sid in pilots:
        for r in S[sid][2]["suspected_mirrors"]:
            if r["far_frac"] >= 0.2 or r["suspected"]:
                n_sus += r["suspected"]
                L.append(f"| `{sid}` | {r['source_label']} #{r['obj_id']} | {' × '.join(map(str, r['size']))} | {r['nodes']} | {r['far_frac']:.0%} | {'**yes**' if r['suspected'] else 'no'} |")
    L += ["", f"Only objects with a fraction >= 20% or listed as suspected are shown; results for all flat objects are in `suspected_mirrors` of each scene's `step4_summary.json`. {n_sus} suspected in total.", ""]
    # mirror vs door
    vw = {x["obj_id"]: x for x in read_json(sp_work(cfg) / MIRROR_SCENE / "step3_views.json")}
    tiles = []
    for k, fid in ((25, vw[25]["frame_id"]), (49, vw[49]["frame_id"])):
        img, K, T = base_image(cfg, frames[fid], 640)
        draw_obbs(img, [box_of(objs[k], f"{objs[k]['source_label']} #{k}")], K, T, np.eye(3), color=GREEN)
        put(img, f"{MIRROR_SCENE} {fid}: {objs[k]['source_label']} #{k}", (8, 22))
        tiles.append(img)
    cv2.imwrite(str(img_dir / "s8_window_is_mirror.jpg"), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    L += ["\"window\" #25 (left) and door #49 (right): the box on the left shows the reflection of the door and jeans on the right.", "", "![window is mirror](scannetpp_step4/s8_window_is_mirror.jpg)", ""]

    # ---- 5. observations and outputs
    review = sp_work(cfg) / "_step4" / "agent_review.md"
    L += [review.read_text().rstrip("\n"), ""] if review.exists() else []
    L += ["## Outputs", "",
          "- Per scene `work/scannetpp/<scene_id>/`: `view_graph.json` (standard format), `visibility.npz`, `step4_summary.json`.",
          "- `visibility.npz` fields: the standard `obj_ids`, `frame_ids`, `state`, `rho`, `completeness`, `area_frac`, `in_frame_frac`, `n_infrustum`, `n_visible`; "
          "plus `state_base` (without S8 / S10), `state_no_mirror` (S10 only), `state_no_magenta` (S8 only), `n_magenta`, `n_mirror`, `magenta_near_px`, `mirror_near`, "
          "`has_sensor_depth`, `n_no_depth`, `mirror_obj_ids`.",
          "- Code: `scannetpp_step4.py` (reuses the shared `step4_graph_visibility` functions, adds S8 / S10), `scannetpp_step4_report.py` (this report and the spot check); "
          "unit tests `tests/test_scannetpp_step4.py` (mirror occluder face, magenta mask). Config: `magenta_*`, `mirror_*` in the `scannetpp` section.",
          "- Figures: `reports/scannetpp_step4/` (not in git)."]
    (rep / "scannetpp_step4_graph_visibility.md").write_text("\n".join(L) + "\n")
    print("wrote", rep / "scannetpp_step4_graph_visibility.md")


# ---------------------------------------------------------------- spot check
def spotcheck_images(cfg) -> None:
    sp = sp_cfg(cfg)
    pilots = sp["pilot_scenes"]
    out_dir = path_of(cfg, "reports") / "scannetpp_visibility_spotcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    # new seed (purpose string differs from all ARKit rounds)
    rng = scene_rng(cfg, *SPOT_SEED)
    S = {sid: load_scene(cfg, sid) for sid in pilots}
    pools = {}
    for sid in pilots:
        objs, vis = S[sid][0], S[sid][1]
        pools[sid] = [(sid, int(r), int(c)) for r, c in zip(*np.where(vis["state"] == STATE_VISIBLE)) if objs[r]["referable"]]
    quota = {sid: SPOT_N // len(pilots) for sid in pilots}
    for sid in pilots:  # if a scene is short, give its quota to the other scenes
        if len(pools[sid]) < quota[sid]:
            extra = quota[sid] - len(pools[sid])
            quota[sid] = len(pools[sid])
            for s2 in pilots:
                if s2 != sid and extra:
                    quota[s2] += 1
                    extra -= 1
    picks = []
    for sid in pilots:
        pool = pools[sid]
        objs, vis = S[sid][0], S[sid][1]
        near = []
        if sid == MIRROR_SCENE:
            near = [p for p in pool if vis["mirror_near"][p[1], p[2]] and p[1] not in vis["mirror_obj_ids"]]
        elif sid == MAGENTA_SCENE:
            near = [p for p in pool if vis["magenta_near_px"][p[1], p[2]] > 0]
        chosen = []
        if near:
            for i in rng.choice(len(near), min(SPOT_MIN_NEAR, len(near)), replace=False):
                chosen.append((near[i], "mirror_near" if sid == MIRROR_SCENE else "magenta_near"))
        rest = [p for p in pool if p not in {c[0] for c in chosen}]
        for i in rng.choice(len(rest), min(quota[sid] - len(chosen), len(rest)), replace=False):
            p = rest[i]
            tag = "random"
            if sid == MIRROR_SCENE and vis["mirror_near"][p[1], p[2]]:
                tag = "random(mirror_near)"
            if sid == MAGENTA_SCENE and vis["magenta_near_px"][p[1], p[2]] > 0:
                tag = "random(magenta_near)"
            chosen.append((p, tag))
        picks += chosen
    picks = [picks[i] for i in rng.permutation(len(picks))]
    rows = []
    for k, ((sid, r, c), tag) in enumerate(picks, 1):
        objs, vis, _, _, frames = S[sid]
        o = objs[r]
        fid = str(vis["frame_ids"][c])
        img, K, T = base_image(cfg, frames[fid], 1280)
        draw_obbs(img, [box_of(o, o["ref_name"])], K, T, np.eye(3), color=GREEN)
        put(img, f"#{k:02d} {sid} {fid}   object: {o['ref_name']}", (10, 30), 0.8)
        info = (f"rho={float(vis['rho'][r, c]):.2f} completeness={float(vis['completeness'][r, c]):.2f} area={float(vis['area_frac'][r, c]):.3f} "
                f"in_frame={float(vis['in_frame_frac'][r, c]):.2f}")
        put(img, info, (10, img.shape[0] - 14), 0.6)
        cv2.imwrite(str(out_dir / f"pair_{k:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({"pair": f"{k:02d}", "scene_id": sid, "frame_id": fid, "obj_id": o["obj_id"], "name": o["ref_name"], "source_label": o["source_label"],
                     "rho": round(float(vis["rho"][r, c]), 2), "completeness": round(float(vis["completeness"][r, c]), 2),
                     "area_frac": round(float(vis["area_frac"][r, c]), 4), "in_frame_frac": round(float(vis["in_frame_frac"][r, c]), 2),
                     "magenta_near_px": int(vis["magenta_near_px"][r, c]), "mirror_near": bool(vis["mirror_near"][r, c]), "source": tag})
    d = sp_work(cfg) / "_step4"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "spotcheck_pairs.csv", index=False)
    print({sid: len(pools[sid]) for sid in pilots}, Counter(r["scene_id"] for r in rows), Counter(r["source"] for r in rows))


def spotcheck_report(cfg) -> None:
    sp = sp_cfg(cfg)
    pairs = pd.read_csv(sp_work(cfg) / "_step4" / "spotcheck_pairs.csv", dtype={"pair": str, "scene_id": str})
    j = pd.read_csv(path_of(cfg, "reports") / "scannetpp_visibility_spotcheck_judgments.tsv", sep="\t", dtype={"pair": str}, keep_default_na=False)
    df = pairs.merge(j, on="pair")
    assert len(df) == len(pairs)
    n_ok = int((df.ok == "yes").sum())
    fails = df[df.ok != "yes"]
    passed = n_ok / len(df) >= 0.95
    L = ["# ScanNet++ visibility spot check (5 pilot scenes)", "",
         "> Branch `scannetpp`. Samples and images are generated by `python -m actspace_build.scannetpp_step4_report spotcheck_images`, per-image judgments are in "
         "`reports/scannetpp_visibility_spotcheck_judgments.tsv`, and this report is generated by `... spotcheck`. Implements the mandatory Step 4 visibility spot check (see docs/data_construction.md).", "",
         f"**Round 2**: Step 4 rerun with the two extra S12 conditions and the two blacklisted objects; 50 new pairs drawn with a new seed (`scene_rng{SPOT_SEED}`). "
         "Round 1 (43 / 50, failed) was used only to choose the S12 thresholds and does not count towards the pass rate; its report is `scannetpp_visibility_spotcheck_round1.md`.", "",
         "Sampling: visible (object, node) pairs drawn only from objects that are referable under the new vocabulary (`merged_unique_fine_name`) and not blacklisted, 50 pairs in total. Evidence views only exist from Step 6 on, so this round samples evenly over scenes (10 pairs per scene); "
         f"3 pairs are drawn first near the mirror in `{MIRROR_SCENE}` (mirror in the same image, mirror itself excluded) and 3 near the patches in `{MAGENTA_SCENE}` (magenta pixels within the object projection padded by 25%); the rest are random.", "",
         "One image per pair: original image (1280 wide, uncropped) + the object's box (green) + its name in questions. Each image is judged on \"can a person recognise the object in this image and roughly tell where it is\"; **barely recognisable counts as \"no\"**.", "",
         f"**Result: {n_ok} / {len(df)} ({n_ok / len(df):.0%}) judged \"yes\"**, required >= 95%: {'**pass**' if passed else '**fail**'}. "
         + ("" if passed else "Stopped as required, thresholds not tuned; failure types below."), "",
         "| Scene | Pairs | Yes | Of which near mirror / patches |", "|---|---|---|---|"]
    for sid in sp["pilot_scenes"]:
        d = df[df.scene_id == sid]
        nr = d[d.source.str.contains("near")]
        near_txt = f"{len(nr)} pairs, yes {int((nr.ok == 'yes').sum())}" if len(nr) else "—"
        L.append(f"| `{sid}` | {len(d)} | {int((d.ok == 'yes').sum())} | {near_txt} |")
    if len(fails):
        L += ["", "## Failures", "", "| # | Scene | Frame | Object | ρ | Completeness | Area | In frame | Type | Note |", "|---|---|---|---|---|---|---|---|---|---|",
              *[f"| {r.pair} | `{r.scene_id}` | {r.frame_id} | {r.name} #{r.obj_id} | {r.rho} | {r.completeness} | {r.area_frac} | {r.in_frame_frac} | {r.failure_type} | {r.note} |"
                for r in fails.itertuples()], "",
              "Failure types: " + ", ".join(f"{k} {v}" for k, v in Counter(fails.failure_type).most_common()) + "."]
    L += ["", "## All judgments", "", "| # | Source | Scene | Frame | Object (raw label) | ρ | Completeness | Area | In frame | Recognisable | Note |", "|---|---|---|---|---|---|---|---|---|---|---|",
          *[f"| {r.pair} | {r.source} | `{r.scene_id}` | {r.frame_id} | {r.name} ({r.source_label}) #{r.obj_id} | {r.rho} | {r.completeness} | {r.area_frac} | "
            f"{r.in_frame_frac} | {'yes' if r.ok == 'yes' else '**no**'} | {r.note} |" for r in df.itertuples()], "",
          ]
    review = sp_work(cfg) / "_step4" / "spotcheck_review.md"
    if review.exists():
        L += ["", review.read_text().rstrip("\n"), ""]
    L += ["## Spot-check images", ""]
    for r in df.itertuples():
        L += [f"**#{r.pair}** `{r.scene_id}` {r.frame_id} {r.name}: {'yes' if r.ok == 'yes' else '**no**'}{(', ' + r.note) if r.note else ''}  ",
              f"![{r.pair}](scannetpp_visibility_spotcheck/pair_{r.pair}.jpg)", ""]
    L += ["", "Individual images: `reports/scannetpp_visibility_spotcheck/pair_XX.jpg` (not in git)."]
    (path_of(cfg, "reports") / "scannetpp_visibility_spotcheck.md").write_text("\n".join(L) + "\n")
    print("wrote", path_of(cfg, "reports") / "scannetpp_visibility_spotcheck.md", n_ok, len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "spotcheck_images", "spotcheck"])
    args = ap.parse_args()
    cfg = load_config()
    {"report": write_report, "spotcheck_images": spotcheck_images, "spotcheck": spotcheck_report}[args.cmd](cfg)


if __name__ == "__main__":
    main()
