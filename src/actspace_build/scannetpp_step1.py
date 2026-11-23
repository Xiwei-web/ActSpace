"""Step 1 / GATE A (ScanNet++): aligned_pose convention, per-pose depth check, up axis, RGB-depth alignment, image orientation, frames_all.json.

Step 1 and GATE A (see docs/data_construction.md):
  1. Sample about 10 frames per scene and compare mesh-rendered depth with sensor depth to decide whether aligned_pose is
     T_wc or T_cw and whether the camera axes are OpenCV or OpenGL; these 10 frames do not decide whether the scene is kept.
  2. Depth check for every pose (10 FPS decoded frames, same frame ID as depth): intrinsics scaled by 256/1920 to depth
     resolution; valid depth 0.1-10 m; < 500 valid pixels = unverified; median absolute error >= 0.10 m = bad pose, and the
     frame is dropped (POSE_FRAME_INVALID).
  3. Drop the whole scene (POSE_INVALID) if bad poses > 20% of verified poses, or if fewer than 40 keyframes remain after
     removing bad frames. The keyframe count is estimated on the remaining frames with the Step 2 rules (blur filter +
     keyframe selection); Step 2 recomputes it.
Reuses the ARKit pose convention candidates, depth error, box drawing and timeline plot (step1_arkit) and keyframe
selection (step2_keyframes).

Usage: python -m actspace_build.scannetpp_step1 gate_a [--scenes ID ...] [--workers N] [--force]
       python -m actspace_build.scannetpp_step1 report
Outputs: work/scannetpp/<scene_id>/{gate_a.json, pose_check.npz, frames_all.json, scene_meta.json, dropped.jsonl},
         reports/scannetpp_gateA/*, reports/scannetpp_step1_gateA.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import open3d as o3d

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, scene_rng, write_json
from .scannetpp import object_obb, scale_intrinsics, scene_paths, sp_cfg, sp_work
from .scannetpp_step0 import frame_ids_in, up_axis_from_floor
from .step1_arkit import POSE_BAD, POSE_CONVENTIONS, POSE_GOOD, POSE_UNVERIFIED, colorize, depth_edges, depth_error, draw_obbs
from .step2_keyframes import laplacian_variance, select_keyframes

DATASET = "scannetpp"
STATUS_NAME = {POSE_GOOD: "good", POSE_BAD: "bad", POSE_UNVERIFIED: "unverified"}


# ---------------------------------------------------------------- loading
def load_scene(cfg: dict, scene_id: str) -> dict:
    sc = sp_cfg(cfg)
    scene = scene_paths(cfg, scene_id)
    poses = read_json(scene.iphone_pose_intrinsic_imu_path)
    ids = frame_ids_in(scene.iphone_rgb_dir, ".jpg")
    if ids != frame_ids_in(scene.iphone_depth_dir, ".png"):
        raise RuntimeError(f"{scene_id}: decoded rgb and depth frame ids differ")
    mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
    mesh.compute_vertex_normals()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    keys = [f"frame_{i:06d}" for i in ids]
    return {"scene": scene, "ids": ids, "keys": keys,
            "M": np.stack([np.asarray(poses[k]["aligned_pose"], float) for k in keys]),
            "K_rgb": np.stack([np.asarray(poses[k]["intrinsic"], float) for k in keys]),
            "ts": np.array([poses[k]["timestamp"] for k in keys], float),
            "poses": poses, "V": V, "F": F, "normals": np.asarray(mesh.vertex_normals),
            "raycast": arkit.raycasting_scene(V, F), "rgb_wh": tuple(sc["rgb_size"]), "depth_wh": tuple(sc["depth_size"])}


def depth_m(s: dict, k: int, scale: float) -> np.ndarray:
    return arkit.read_depth_m(s["scene"].iphone_depth_dir / f"{s['keys'][k]}.png", scale)


def load_objects(cfg: dict, s: dict) -> list[dict]:
    """Objects in segments_anno.json → mesh vertices via segments → OBB (same representation as the ARKit annotations, used by draw_obbs)."""
    seg = np.asarray(read_json(s["scene"].scan_mesh_segs_path)["segIndices"])
    by_seg: dict[int, list[int]] = {}
    for vi, sid in enumerate(seg):
        by_seg.setdefault(int(sid), []).append(vi)
    banned = set(cfg["objects"]["banned_categories"])
    objs = []
    for gobj in read_json(s["scene"].scan_anno_json_path)["segGroups"]:
        vids = np.array(sorted(v for sg in gobj["segments"] for v in by_seg.get(int(sg), [])), dtype=np.int64)
        if gobj["label"] in banned or len(vids) < 4:
            continue
        objs.append({"label": gobj["label"], "object_id": gobj["objectId"], "vertex_ids": vids,
                     **object_obb(s["V"][vids])})
    return objs


def floor_ceiling_masks(cfg: dict, s: dict):
    seg = np.asarray(read_json(s["scene"].scan_mesh_segs_path)["segIndices"])
    anno = read_json(s["scene"].scan_anno_json_path)["segGroups"]
    mask = lambda lab: np.isin(seg, [sg for gobj in anno if gobj["label"] == lab for sg in gobj["segments"]])
    return mask("floor"), mask("ceiling")


# ---------------------------------------------------------------- checks
def edge_alignment_px(s: dict, k: int, T_wc: np.ndarray, scale: float) -> float | None:
    """Discontinuity edges of mesh-rendered depth → median distance to RGB Canny edges (in pixels at the original RGB resolution)."""
    W, H = int(round(s["rgb_wh"][0] * scale)), int(round(s["rgb_wh"][1] * scale))
    K = scale_intrinsics(s["K_rgb"][k], s["rgb_wh"], (W, H))
    edges = depth_edges(g.render_depth(s["raycast"], K, T_wc, W, H))
    if edges.sum() < 50:
        return None
    gray = cv2.resize(cv2.imread(str(s["scene"].iphone_rgb_dir / f"{s['keys'][k]}.jpg"), cv2.IMREAD_GRAYSCALE), (W, H),
                      interpolation=cv2.INTER_AREA)
    dist = cv2.distanceTransform((cv2.Canny(gray, 50, 150) == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.median(dist[edges]) / scale)


def gate_a_scene(args) -> dict:
    scene_id, force = args
    cfg = load_config()
    sc, gcfg, fc = sp_cfg(cfg), cfg["gate_a"], cfg["frames"]
    wd = sp_work(cfg) / scene_id
    wd.mkdir(parents=True, exist_ok=True)
    if is_done(wd, "gate_a") and not force:
        return read_json(wd / "gate_a.json")
    drops = DropLog(wd, "step1_gate_a")
    s = load_scene(cfg, scene_id)
    n = len(s["ids"])
    dscale = sc["depth_scale"]
    K_d = [scale_intrinsics(K, s["rgb_wh"], s["depth_wh"]) for K in s["K_rgb"]]
    Wd, Hd = s["depth_wh"]

    # 1. Determine the pose convention from ~10 frames (does not decide whether the scene is kept)
    rng = scene_rng(cfg, f"scannetpp_{scene_id}", "gate_a")
    sample = sorted(rng.choice(n, size=min(gcfg["frames_per_scene"], n), replace=False).tolist())
    per_conv = {}
    for name, fn in POSE_CONVENTIONS.items():
        errs = []
        for k in sample:
            e, nv, _ = depth_error(s["raycast"], K_d[k], fn(s["M"][k]), Wd, Hd, depth_m(s, k, dscale), gcfg)
            errs.append({"frame": s["keys"][k], "median_abs_err_m": e, "valid_px": nv})
        ok = [e["median_abs_err_m"] is not None and e["median_abs_err_m"] < gcfg["depth_median_abs_err_m"] for e in errs]
        per_conv[name] = {"frames": errs, "pass_ratio": float(np.mean(ok)), "passed": float(np.mean(ok)) >= gcfg["pass_ratio"]}
    med = lambda c: np.nanmedian([e["median_abs_err_m"] if e["median_abs_err_m"] is not None else np.nan for e in per_conv[c]["frames"]])
    best = max(per_conv, key=lambda c: (per_conv[c]["pass_ratio"], -med(c)))
    conv = POSE_CONVENTIONS[best]
    T_all = np.stack([conv(M) for M in s["M"]])

    # Empirical depth unit: median of rendered depth / sensor depth (uint16 × depth_scale) should be about 1
    ratios = []
    for k in sample:
        dsen = depth_m(s, k, dscale)
        drend = g.render_depth(s["raycast"], K_d[k], T_all[k], Wd, Hd)
        v = (dsen > gcfg["depth_min_valid_m"]) & (dsen < gcfg["depth_max_valid_m"]) & np.isfinite(drend)
        if v.sum() >= gcfg["min_valid_pixels"]:
            ratios.append(float(np.median(drend[v] / dsen[v])))

    # 2. Per-pose check (all 10 FPS decoded frames)
    status = np.full(n, POSE_UNVERIFIED, np.uint8)
    err = np.full(n, np.nan, np.float32)
    nvalid = np.zeros(n, np.int32)
    for k in range(n):
        e, nv, _ = depth_error(s["raycast"], K_d[k], T_all[k], Wd, Hd, depth_m(s, k, dscale), gcfg)
        nvalid[k] = nv
        if e is not None:
            err[k] = e
            status[k] = POSE_BAD if e >= gcfg["depth_median_abs_err_m"] else POSE_GOOD
    np.savez_compressed(wd / "pose_check.npz", traj_ts=s["ts"], status=status, err=err, valid_px=nvalid,
                        frame_ids=np.array(s["ids"]))
    n_good, n_bad, n_unv = int((status == POSE_GOOD).sum()), int((status == POSE_BAD).sum()), int((status == POSE_UNVERIFIED).sum())
    bad_frac = n_bad / (n_good + n_bad) if n_good + n_bad else 1.0

    # Up axis: mean floor-vertex normal + camera above floor (report only; stop if it is not +z, no transform)
    floor, ceiling = floor_ceiling_masks(cfg, s)
    up = up_axis_from_floor(s["V"], s["normals"], floor, ceiling) if floor.any() else None
    up_label = (up["normal_sign"] + up["axis"]) if up else None
    cam_above_floor = None
    if up and up_label == "+z":
        # up_axis_from_floor only provides floor_median when a ceiling is annotated; without a ceiling use the median height of floor vertices
        floor_z = up["floor_median"] if "floor_median" in up else float(np.median(s["V"][floor, 2]))
        cam_above_floor = round(float(np.median(T_all[:, 2, 3]) - floor_z), 3)

    # Image orientation: direction of world +z in the image → number of clockwise 90° rotations needed
    sky = Counter(g.sky_direction_from_pose(T) for T in T_all)
    sky_auto, n_top = sky.most_common(1)[0]
    agreement = n_top / n
    quarters = g.SKY_TO_CW_QUARTERS[sky_auto]
    orient_ok = agreement >= cfg["orientation"]["min_agreement_frac"]
    if not orient_ok:
        drops.add("scene", scene_id, "ORIENTATION_CONFLICT", sky_auto=sky_auto, agreement=agreement, detail="low_agreement")

    # Drop bad frames (unverified frames according to scannetpp.drop_unverified_pose_frames)
    drop_states = {POSE_BAD} | ({POSE_UNVERIFIED} if sc["drop_unverified_pose_frames"] else set())
    keep = np.array([st not in drop_states for st in status])
    for k in np.flatnonzero(~keep):
        drops.add("frame", s["keys"][k], "POSE_FRAME_INVALID", pose_status=STATUS_NAME[int(status[k])],
                  median_abs_err_m=None if np.isnan(err[k]) else float(err[k]), valid_px=int(nvalid[k]))

    # Estimate the keyframe count with the Step 2 rules after removing bad frames
    kept = np.flatnonzero(keep)
    with ThreadPoolExecutor(16) as ex:
        lap = list(ex.map(lambda k: laplacian_variance(s["scene"].iphone_rgb_dir / f"{s['keys'][k]}.jpg"), kept))
    thr = max(float(np.percentile(lap, fc["blur_drop_percentile"])), fc["blur_min_rel_to_median"] * float(np.median(lap))) if lap else 0.0
    sharp = [{"timestamp": s["ts"][k], "position": T_all[k][:3, 3], "yaw_deg": g.yaw_deg_from_Twc(T_all[k]), "blur_lapvar": b}
             for k, b in zip(kept, lap) if b >= thr]
    n_kf = len(select_keyframes(sharp, fc["keyframe_min_translation_m"], fc["keyframe_min_rotation_deg"],
                                fc["keyframe_sharpest_window_s"]))
    pose_valid = bad_frac <= gcfg["max_bad_pose_frac"] and n_kf >= fc["min_keyframes_per_scene"]
    if not pose_valid:
        drops.add("scene", scene_id, "POSE_INVALID",
                  rule="bad_pose_frac" if bad_frac > gcfg["max_bad_pose_frac"] else "too_few_keyframes_after_bad_frames",
                  bad_frac_of_verified=bad_frac, num_keyframes_est=n_kf)

    # RGB vs mesh (i.e. depth) alignment: distance from mesh edges to RGB edges; control = same RGB with a pose more than 3 s away
    shift = int(round(3.0 * cfg["frames"]["scannetpp_decode_fps"]))
    align = []
    for k in sample:
        kk = k + shift if k + shift < n else k - shift
        align.append({"frame": s["keys"][k],
                      "edge_dist_px": edge_alignment_px(s, k, T_all[k], sc["gate_a_edge_check_scale"]),
                      "edge_dist_px_wrong_pose": edge_alignment_px(s, k, T_all[kk], sc["gate_a_edge_check_scale"])})

    res = {"scene_id": scene_id, "n_decoded_frames": n, "sampled_frames": [s["keys"][k] for k in sample],
           "conventions": per_conv, "best_convention": best,
           "depth_unit_ratio_rendered_over_sensor": ratios,
           "depth_K_scale": [s["depth_wh"][0] / s["rgb_wh"][0], s["depth_wh"][1] / s["rgb_wh"][1]],
           "pose_check": {"n_poses": n, "n_good": n_good, "n_bad": n_bad, "n_unverified": n_unv,
                          "bad_frac_of_verified": bad_frac, "err_median_m": float(np.nanmedian(err)) if n_good + n_bad else None,
                          "err_p90_m": float(np.nanpercentile(err, 90)) if n_good + n_bad else None},
           "n_frames_dropped": int((~keep).sum()), "num_keyframes_est": n_kf, "blur_threshold": thr,
           "pose_valid": pose_valid, "up_axis": up, "up_axis_detected": up_label, "camera_median_height_above_floor_m": cam_above_floor,
           "sky_votes": dict(sky), "sky_auto": sky_auto, "sky_agreement": agreement, "rotation_cw_quarters": quarters,
           "orientation_ok": orient_ok, "rgb_alignment": align}
    write_json(wd / "gate_a.json", res)
    write_frames(cfg, s, T_all, K_d, keep, res)
    mark_done(wd, "gate_a")
    return res


def write_frames(cfg: dict, s: dict, T_all, K_d, keep, res) -> None:
    """Step 1 outputs: frames_all.json (kept frames only; T_wc with OpenCV camera axes, upright rotation applied) and scene_meta.json."""
    sid = s["scene"].scene_id
    q = res["rotation_cw_quarters"]
    W, H = s["rgb_wh"]
    out = []
    for k in np.flatnonzero(keep):
        K2, T2, W2, H2 = g.rotate_camera_cw(s["K_rgb"][k], T_all[k], W, H, q)
        Kd2, _, Wd2, Hd2 = g.rotate_camera_cw(K_d[k], T_all[k], *s["depth_wh"], q)
        key = s["keys"][k]
        out.append({"frame_id": f"f_{len(out):06d}", "source_frame": key,
                    "rel_path": f"data/{sid}/iphone/rgb/{key}.jpg", "depth_rel_path": f"data/{sid}/iphone/depth/{key}.png",
                    "timestamp": float(s["ts"][k]), "T_wc": T2, "K": K2, "image_size": [W2, H2],
                    "depth_K": Kd2, "depth_size": [Wd2, Hd2], "rotation_cw_quarters": q,
                    "position": T2[:3, 3], "yaw_deg": g.yaw_deg_from_Twc(T2)})
    wd = sp_work(cfg) / sid
    write_json(wd / "frames_all.json", out)
    write_json(wd / "scene_meta.json", {
        "scene_id": f"scannetpp_{sid}", "dataset": DATASET, "source_id": sid, "split": sp_cfg(cfg)["split"],
        "up_axis": "z", "units": "meters", "pose_convention": "T_wc_opencv", "image_size": out[0]["image_size"] if out else None,
        "frame_source": "iphone_rgb_10fps", "num_frames": len(out), "num_frames_raw": len(s["ids"]),
        "traj_convention": res["best_convention"], "up_axis_detected_raw": res["up_axis_detected"],
        "sky_direction_auto": res["sky_auto"], "rotation_cw_quarters": q,
        "status": {"pose_valid": res["pose_valid"], "orientation_ok": res["orientation_ok"],
                   "up_axis_ok": res["up_axis_detected"] == "+z",
                   "resolution_ok": max(W, H) >= cfg["frames"]["min_rgb_long_side_px"]}})


# ---------------------------------------------------------------- visualisation
def visualize_scene(scene_id: str, out_dir) -> list[str]:
    """3 images: left = RGB (downscaled to 960×720) overlaid with mesh edges (green) and projected object OBBs (yellow, labelled); right = sensor depth / rendered depth."""
    cfg = load_config()
    sc = sp_cfg(cfg)
    res = read_json(sp_work(cfg) / scene_id / "gate_a.json")
    st = np.load(sp_work(cfg) / scene_id / "pose_check.npz")
    s = load_scene(cfg, scene_id)
    objs = load_objects(cfg, s)
    conv = POSE_CONVENTIONS[res["best_convention"]]
    good = [s["keys"].index(f) for f in res["sampled_frames"] if st["status"][s["keys"].index(f)] == POSE_GOOD]
    picks = [good[i] for i in np.linspace(0, len(good) - 1, min(3, len(good))).round().astype(int)] if good else []
    scale = sc["gate_a_edge_check_scale"]
    W, H = int(s["rgb_wh"][0] * scale), int(s["rgb_wh"][1] * scale)
    Wd, Hd = s["depth_wh"]
    rng = np.random.default_rng(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in picks:
        T = conv(s["M"][k])
        K = scale_intrinsics(s["K_rgb"][k], s["rgb_wh"], (W, H))
        Kd = scale_intrinsics(s["K_rgb"][k], s["rgb_wh"], s["depth_wh"])
        img = cv2.resize(cv2.imread(str(s["scene"].iphone_rgb_dir / f"{s['keys'][k]}.jpg")), (W, H), interpolation=cv2.INTER_AREA)
        drend_hi = g.render_depth(s["raycast"], K, T, W, H)
        over = img.copy()
        over[depth_edges(drend_hi)] = (0, 255, 0)
        shown = []
        for o in objs:      # only draw objects with a high enough visible fraction of sample points (visualisation only)
            vids = o["vertex_ids"] if len(o["vertex_ids"]) <= 300 else rng.choice(o["vertex_ids"], 300, replace=False)
            uv, z = g.project(s["V"][vids], K, T)
            inside = g.in_image(uv, z, W, H)
            if not inside.any():
                continue
            ui = np.clip(np.round(uv[inside]).astype(int), 0, [W - 1, H - 1])
            vis = z[inside] <= drend_hi[ui[:, 1], ui[:, 0]] + cfg["visibility"]["occlusion_tolerance_m"]
            if vis.sum() / len(vids) >= sc["gate_a_vis_min_visible_frac"]:
                shown.append((int(vis.sum()), o))
        shown = [o for _, o in sorted(shown, key=lambda t: -t[0])[:sc["gate_a_vis_objects_max"]]]
        draw_obbs(over, shown, K, T, np.eye(3))
        dsen = depth_m(s, k, sc["depth_scale"])
        drend = g.render_depth(s["raycast"], Kd, T, Wd, Hd)
        right = np.concatenate([cv2.resize(colorize(dsen), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST),
                                cv2.resize(colorize(drend), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)], axis=0)
        panel = np.concatenate([over, right], axis=1)
        e = st["err"][k]
        cv2.putText(panel, f"{scene_id} {s['keys'][k]} conv={res['best_convention']} up={res['up_axis_detected']} "
                           f"rot={res['rotation_cw_quarters'] * 90}cw med|err|={e:.3f}m boxes={len(shown)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "sensor depth (0-5 m)", (W + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "mesh-rendered depth", (W + 8, H // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        p = out_dir / f"{scene_id}_{s['keys'][k]}.jpg"
        cv2.imwrite(str(p), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p.name)
    return paths


def run_scene(args) -> dict:
    scene_id, force = args
    from .step1_arkit import plot_pose_timeline
    cfg = load_config()
    try:
        res = gate_a_scene((scene_id, force))
    except Exception as e:  # an error in one scene does not stop the others; that scene is recorded as NOT_RUN in the GATE A summary
        import traceback
        traceback.print_exc()
        return {"scene_id": scene_id, "error": f"{type(e).__name__}: {e}"}
    out = path_of(cfg, "reports") / "scannetpp_gateA"
    plot_pose_timeline(scene_id, out / f"{scene_id}_timeline.png", dataset=DATASET)
    res["vis"] = visualize_scene(scene_id, out)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gate_a")
    a.add_argument("--scenes", nargs="*")
    a.add_argument("--workers", type=int, default=5)
    a.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "gate_a":
        scenes = args.scenes or sp_cfg(cfg)["pilot_scenes"]
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(run_scene, [(s, args.force) for s in scenes]):
                if "error" in r:
                    print(r["scene_id"], "ERROR", r["error"], flush=True)
                    continue
                pc = r["pose_check"]
                print(r["scene_id"], r["best_convention"], f"good={pc['n_good']} bad={pc['n_bad']} unv={pc['n_unverified']}",
                      f"kf={r['num_keyframes_est']} valid={r['pose_valid']} up={r['up_axis_detected']} rot={r['rotation_cw_quarters']}",
                      f"agree={r['sky_agreement']:.3f}", flush=True)
    else:
        from .scannetpp_step1_report import write_report
        write_report(cfg)


if __name__ == "__main__":
    main()
