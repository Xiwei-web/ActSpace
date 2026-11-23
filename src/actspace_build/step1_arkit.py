"""Step 1 (ARKitScenes): GATE A pose-convention validation, up-axis detection, image orientation.

Subcommands
  gate_a  measure the traj pose convention on lowres_wide + lowres_depth (no direction assumed),
          detect the world up axis, infer image rotation from the poses and cross-check it against sky_direction.
Usage: python -m actspace_build.step1_arkit gate_a --scenes 41069021 ... [--force]
Outputs: work/arkitscenes/<vid>/gate_a.json, reports/gateA/*.jpg
"""
from __future__ import annotations

import argparse
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, scene_rng, work_dir, write_json

_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera axes (y up, z back) <-> OpenCV

# candidate pose conventions: traj matrix M -> T_wc (OpenCV camera axes)
POSE_CONVENTIONS = {
    "inv_opencv": lambda M: np.linalg.inv(M),
    "inv_opengl": lambda M: np.linalg.inv(M) @ _FLIP,
    "direct_opencv": lambda M: M,
    "direct_opengl": lambda M: M @ _FLIP,
}


# ---------------------------------------------------------------- up axis

def detect_up_rotation(objs: list[dict], T_wc: np.ndarray):
    """Determine the world up axis by a vote over the gravity axes of the annotated OBBs; the sign comes from cameras being above object bottoms.

    Returns (R_up, up_label): R_up rotates the detected up direction to +z.
    """
    # Each object votes for every world axis that one of its box axes is parallel to; the gravity axis is the one shared by all boxes.
    # Boxes whose horizontal directions happen to align with the coordinate axes vote for all three axes; ties are broken by the consistency of
    # image orientation: a handheld video keeps a roughly fixed orientation, so with the true gravity axis as "up" the per-frame sky direction is highly consistent, while with a horizontal axis it changes as the camera turns.
    cam_centers = T_wc[:, :3, 3]
    votes = Counter()
    for o in objs:
        votes.update({int(np.argmax(np.abs(row))) for row in o["axes"] if np.max(np.abs(row)) > 0.99})
    top = max(votes.values())
    tied = [k for k in range(3) if votes.get(k, 0) == top]

    def sky_consistency(k):
        up = np.eye(3)[k]
        return Counter(g.sky_direction_from_pose(T, up) for T in T_wc).most_common(1)[0][1] / len(T_wc)

    k = max(tied, key=sky_consistency)
    bottoms = []
    for o in objs:
        r = int(np.argmax(np.abs(o["axes"][:, k])))
        bottoms.append(o["center"][k] - 0.5 * o["size"][r])
    sign = 1.0 if np.median(cam_centers[:, k]) > np.median(bottoms) else -1.0
    up = np.zeros(3)
    up[k] = sign
    label = ("+" if sign > 0 else "-") + "xyz"[k]
    if label == "+z":
        return np.eye(3), label
    if label == "+y":
        return g.Y_UP_TO_Z_UP.copy(), label
    R, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])
    return R.as_matrix(), label


# ---------------------------------------------------------------- GATE A

def depth_error(raycast, K, T_wc, W, H, depth_sensor, gcfg):
    rendered = g.render_depth(raycast, K, T_wc, W, H)
    valid = (depth_sensor > gcfg["depth_min_valid_m"]) & (depth_sensor < gcfg["depth_max_valid_m"]) & np.isfinite(rendered)
    n = int(valid.sum())
    if n < gcfg["min_valid_pixels"]:
        return None, n, rendered
    return float(np.median(np.abs(rendered[valid] - depth_sensor[valid]))), n, rendered


def match_frames(frame_list, traj_ts, tol):
    """Nearest-neighbour match of frames to traj timestamps. Returns [(ts, fname, traj_index)] and the unmatched frames."""
    matched, dropped = [], []
    for ts, fname in frame_list:
        i = int(np.argmin(np.abs(traj_ts - ts)))
        if abs(traj_ts[i] - ts) <= tol:
            matched.append((ts, fname, i))
        else:
            dropped.append((ts, fname, float(abs(traj_ts[i] - ts))))
    return matched, dropped


POSE_GOOD, POSE_BAD, POSE_UNVERIFIED = 0, 1, 2


def check_all_poses(cfg, files, raycast, traj_ts, M, conv):
    """Per-pose depth check: for each traj pose find a lowres_depth within the time tolerance,
    render mesh depth and compare the median absolute error; >= threshold is a bad pose. Poses without a depth frame or with too few valid pixels are unverified."""
    gcfg = cfg["gate_a"]
    depth = arkit.list_frames(files["lowres_depth"], ".png")
    d_ts = np.array([t for t, _ in depth])
    status = np.full(len(traj_ts), POSE_UNVERIFIED, np.uint8)
    err = np.full(len(traj_ts), np.nan, np.float32)
    for i, t in enumerate(traj_ts):
        j = int(np.argmin(np.abs(d_ts - t)))
        if abs(d_ts[j] - t) > gcfg["pose_check_depth_tol_s"]:
            continue
        name = depth[j][1]
        K, W, H = arkit.load_pincam(files["lowres_wide_intrinsics"] / name.replace(".png", ".pincam"))
        dsen = arkit.read_depth_m(files["lowres_depth"] / name, cfg["arkitscenes"]["depth_scale"])
        e, _, _ = depth_error(raycast, K, POSE_CONVENTIONS[conv](M[i]), W, H, dsen, gcfg)
        if e is None:
            continue
        err[i] = e
        status[i] = POSE_BAD if e >= gcfg["depth_median_abs_err_m"] else POSE_GOOD
    return status, err


def gate_a_scene(args):
    vid, force = args
    cfg = load_config()
    gcfg = cfg["gate_a"]
    wd = work_dir(cfg, "arkitscenes", vid)
    if is_done(wd, "gate_a") and not force:
        return read_json(wd / "gate_a.json")
    root = path_of(cfg, "arkit_3dod_root")
    files = arkit.scene_files_3dod(root, vid)
    drops = DropLog(wd, "step1_gate_a")

    traj_ts, M = arkit.load_traj(files["lowres_wide.traj"])
    frames = arkit.list_frames(files["lowres_wide"], ".png")
    matched, unmatched = match_frames(frames, traj_ts, cfg["frames"]["pose_timestamp_tolerance_s"])
    for ts, fname, dt in unmatched:
        drops.add("frame", fname, "POSE_TS_MISMATCH", source="lowres_wide", dt_s=dt)

    mesh = arkit.load_mesh_legacy(files["mesh"])
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    raycast = arkit.raycasting_scene(V, F)

    rng = scene_rng(cfg, f"arkit_{vid}", "gate_a")
    sample = sorted(rng.choice(len(matched), size=min(gcfg["frames_per_scene"], len(matched)), replace=False).tolist())

    per_conv = {}
    for name, fn in POSE_CONVENTIONS.items():
        errs = []
        for si in sample:
            ts, fname, ti = matched[si]
            K, W, H = arkit.load_pincam(files["lowres_wide_intrinsics"] / fname.replace(".png", ".pincam"))
            dsen = arkit.read_depth_m(files["lowres_depth"] / fname, cfg["arkitscenes"]["depth_scale"])
            err, n, _ = depth_error(raycast, K, fn(M[ti]), W, H, dsen, gcfg)
            errs.append({"frame": fname, "median_abs_err_m": err, "valid_px": n})
        ok = [e["median_abs_err_m"] is not None and e["median_abs_err_m"] < gcfg["depth_median_abs_err_m"] for e in errs]
        per_conv[name] = {"frames": errs, "pass_ratio": float(np.mean(ok)), "passed": float(np.mean(ok)) >= gcfg["pass_ratio"]}

    best = max(per_conv, key=lambda c: (per_conv[c]["pass_ratio"],
                                        -np.nanmedian([e["median_abs_err_m"] if e["median_abs_err_m"] is not None else np.nan
                                                       for e in per_conv[c]["frames"]])))
    res = {"video_id": vid, "n_frames": len(frames), "n_matched": len(matched), "n_ts_mismatch": len(unmatched),
           "sampled_frames": [matched[i][1] for i in sample], "conventions": per_conv, "best_convention": best,
           "image_size": [W, H]}

    # up axis and orientation (using the best convention)
    T_all = np.stack([POSE_CONVENTIONS[best](M[i]) for _, _, i in matched])
    objs = arkit.load_annotation(files["annotation"])
    R_up, up_label = detect_up_rotation(objs, T_all)
    T_up = np.einsum("ij,njk->nik", g.world_transform(R_up), T_all)
    sky = Counter(g.sky_direction_from_pose(T) for T in T_up)
    sky_auto, n_top = sky.most_common(1)[0]
    # Cross-check against sky_direction in metadata.csv. Scenes with insufficient within-scene agreement are dropped; when the pose disagrees with metadata the pose decides,
    # but only after the upright thumbnails have been checked manually and the scene added to confirmed_conflict_scenes
    ocfg = cfg["orientation"]
    meta = pd.read_csv(path_of(cfg, "arkit_metadata_csv"), dtype={"video_id": str})
    sky_meta = meta.loc[meta.video_id == vid, "sky_direction"].iat[0]
    agree_ok = n_top / len(T_up) >= ocfg["min_agreement_frac"]
    meta_ok = sky_auto == sky_meta or vid in [str(v) for v in ocfg["confirmed_conflict_scenes"]]
    orient_ok = agree_ok and meta_ok
    if not orient_ok:
        drops.add("scene", vid, "ORIENTATION_CONFLICT", sky_auto=sky_auto, sky_meta=sky_meta, agreement=n_top / len(T_up),
                  detail="low_agreement" if not agree_ok else "metadata_mismatch_pending_manual_confirm")
    # the 10-frame spot check only determines the pose convention; whether a scene is kept is decided by the per-pose check
    status, err = check_all_poses(cfg, files, raycast, traj_ts, M, best)
    np.savez_compressed(wd / "pose_check.npz", traj_ts=traj_ts, status=status, err=err)
    n_good, n_bad = int((status == POSE_GOOD).sum()), int((status == POSE_BAD).sum())
    bad_frac = n_bad / (n_good + n_bad) if n_good + n_bad else 1.0
    pose_valid = bad_frac <= gcfg["max_bad_pose_frac"]
    res.update({"up_axis_detected": up_label, "R_up": R_up, "sky_votes": dict(sky),
                "sky_auto": sky_auto, "sky_agreement": n_top / len(T_up), "sky_meta": sky_meta, "orientation_ok": orient_ok,
                "pose_check": {"n_poses": len(status), "n_good": n_good, "n_bad": n_bad,
                               "n_unverified": int((status == POSE_UNVERIFIED).sum()), "bad_frac_of_verified": bad_frac},
                "pose_valid": pose_valid})
    if not pose_valid:
        drops.add("scene", vid, "POSE_INVALID", rule="bad_pose_frac", bad_frac_of_verified=bad_frac)
    write_json(wd / "gate_a.json", res)
    mark_done(wd, "gate_a")
    return res


# ---------------------------------------------------------------- visualisation

_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def obb_corners(o, R_up=np.eye(3)):
    s = 0.5 * o["size"]
    signs = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
    c = o["center"] + (signs * s) @ o["axes"]
    return c @ R_up.T


def draw_obbs(img, objs, K, T_wc, R_up, color=(0, 255, 255)):
    H, W = img.shape[:2]
    for o in objs:
        C = obb_corners(o, R_up)
        for a, b in _BOX_EDGES:
            pts = np.linspace(C[a], C[b], 40)
            uv, z = g.project(pts, K, T_wc)
            ok = g.in_image(uv, z, W, H)
            for p, q, oka, okb in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
                if oka and okb:
                    cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, 1, cv2.LINE_AA)
        uv, z = g.project((o["center"] @ R_up.T)[None], K, T_wc)
        if g.in_image(uv, z, W, H)[0]:
            cv2.putText(img, o["label"], tuple(np.round(uv[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def depth_edges(depth, jump_m=0.1):
    d = np.where(np.isfinite(depth), depth, 1e3)
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    return (gx > jump_m) | (gy > jump_m)


def colorize(depth, vmax=5.0):
    d = np.clip(np.nan_to_num(depth, posinf=0.0) / vmax, 0, 1)
    c = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~np.isfinite(depth) | (depth <= 0)] = 0
    return c


def visualize_scene(vid: str, n: int, out_dir):
    """Export n images of "mesh edges + OBB projections overlaid on the uprighted original | sensor depth | rendered depth"."""
    cfg = load_config()
    wd = work_dir(cfg, "arkitscenes", vid)
    res = read_json(wd / "gate_a.json")
    files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    traj_ts, M = arkit.load_traj(files["lowres_wide.traj"])
    mesh = arkit.load_mesh_legacy(files["mesh"])
    R_up = np.asarray(res["R_up"])
    V = np.asarray(mesh.vertices) @ R_up.T
    raycast = arkit.raycasting_scene(V, np.asarray(mesh.triangles))
    objs = arkit.load_annotation(files["annotation"])
    q = g.SKY_TO_CW_QUARTERS[res["sky_auto"]]
    conv = POSE_CONVENTIONS[res["best_convention"]]
    frames = res["sampled_frames"]
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in picks:
        ts = arkit.frame_timestamp(fname)
        ti = int(np.argmin(np.abs(traj_ts - ts)))
        K, W, H = arkit.load_pincam(files["lowres_wide_intrinsics"] / fname.replace(".png", ".pincam"))
        T = g.world_transform(R_up) @ conv(M[ti])
        K2, T2, W2, H2 = g.rotate_camera_cw(K, T, W, H, q)
        img = g.rotate_image_cw(cv2.imread(str(files["lowres_wide"] / fname)), q)
        dsen = g.rotate_image_cw(arkit.read_depth_m(files["lowres_depth"] / fname, cfg["arkitscenes"]["depth_scale"]), q)
        drend = g.render_depth(raycast, K2, T2, W2, H2)
        over = img.copy()
        over[depth_edges(drend)] = (0, 255, 0)
        draw_obbs(over, objs, K2, T2, R_up)
        valid = (dsen > 0) & np.isfinite(drend)
        err = np.median(np.abs(drend[valid] - dsen[valid])) if valid.any() else float("nan")
        panel = np.concatenate([over, colorize(dsen), colorize(drend)], axis=1)
        panel = cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, f"{vid} {fname}  conv={res['best_convention']} up={res['up_axis_detected']} rot={q * 90}cw  med|err|={err:.3f}m",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{vid}_{ts:.3f}.jpg"), panel)


# Colours of the bad-pose timeline plot: status colours (not used as series colours); text in neutral ink
_INK, _MUTED, _GRID = (25, 25, 25), (110, 110, 110), (225, 225, 225)
_GOOD, _BAD, _UNVER = (150, 150, 150), (59, 59, 208), (25, 178, 250)  # BGR: neutral grey / critical #d03b3b / warning #fab219


def plot_pose_timeline(vid: str, out_path, err_axis_max_m: float = 0.3, dataset: str = "arkitscenes"):
    """Per-traj-pose median depth error over time: grey = good, red = bad (>= threshold), orange ticks at the bottom = unverified.
    dataset selects work/<dataset>/<vid>/pose_check.npz (ScanNet++ reuses this function; default ARKit)."""
    cfg = load_config()
    d = np.load(work_dir(cfg, dataset, vid) / "pose_check.npz")
    t = d["traj_ts"] - d["traj_ts"][0]
    st, err = d["status"], d["err"]
    thr = cfg["gate_a"]["depth_median_abs_err_m"]
    W, H, L, R, T, B = 1200, 300, 70, 20, 50, 70
    img = np.full((H, W, 3), 255, np.uint8)
    tmax = max(float(t[-1]), 1e-6)
    x = lambda v: int(L + (W - L - R) * v / tmax)
    y = lambda e: int(H - B - (H - B - T) * min(e, err_axis_max_m) / err_axis_max_m)
    for e in np.arange(0, err_axis_max_m + 1e-9, 0.1):
        cv2.line(img, (L, y(e)), (W - R, y(e)), _GRID, 1)
        cv2.putText(img, f"{e:.1f}", (L - 40, y(e) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MUTED, 1, cv2.LINE_AA)
    step = 30 if tmax > 120 else 10
    for s_ in np.arange(0, tmax + 1e-9, step):
        cv2.putText(img, f"{s_:.0f}", (x(s_) - 8, H - B + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (L, y(thr)), (W - R, y(thr)), _BAD, 1, cv2.LINE_AA)
    cv2.putText(img, f"threshold {thr:.2f} m", (W - R - 130, y(thr) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _INK, 1, cv2.LINE_AA)
    for ti, si, ei in zip(t, st, err):
        if si == POSE_UNVERIFIED:
            cv2.line(img, (x(ti), H - B + 2), (x(ti), H - B + 8), _UNVER, 1)
        else:
            cv2.circle(img, (x(ti), y(ei)), 2 if si == POSE_GOOD else 3, _GOOD if si == POSE_GOOD else _BAD, -1, cv2.LINE_AA)
    n_good, n_bad, n_unv = int((st == POSE_GOOD).sum()), int((st == POSE_BAD).sum()), int((st == POSE_UNVERIFIED).sum())
    frac = n_bad / max(n_good + n_bad, 1)
    cv2.putText(img, f"{vid}  bad poses along the video: bad {n_bad} / verified {n_good + n_bad} ({frac:.1%}), unverified {n_unv}",
                (L, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _INK, 1, cv2.LINE_AA)
    lx = L
    for col, label in ((_GOOD, "good pose"), (_BAD, "bad pose (median |err| >= threshold)"), (_UNVER, "unverified (tick at bottom)")):
        cv2.circle(img, (lx + 5, H - 22), 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (lx + 14, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _INK, 1, cv2.LINE_AA)
        lx += 20 + 8 * len(label)
    cv2.putText(img, "time since video start (s)", (W - R - 200, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"depth err (m), clipped at {err_axis_max_m:.1f}", (4, T - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _MUTED, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------- unified frame table (frames_all.json)

def dataset_tag(source: str) -> str:
    """The official pipeline uses vga_wide; lowres_wide is only for debugging later steps before vga is available, with outputs stored separately."""
    return "arkitscenes" if source == "vga_wide" else "arkitscenes_lowres_debug"


def frame_source_dirs(cfg, vid, source):
    if source == "vga_wide":
        r = arkit.scene_files_raw(path_of(cfg, "arkit_raw_root"), vid)
        return r["vga_wide"], r["vga_wide_intrinsics"], f"raw/{cfg['paths']['arkit_split_name']}/{vid}"
    f = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    return f["lowres_wide"], f["lowres_wide_intrinsics"], f"3dod/{cfg['paths']['arkit_split_name']}/{vid}/{vid}_frames"


def build_frames(vid: str, source: str, force: bool = False):
    cfg = load_config()
    acfg = cfg["arkitscenes"]
    gate_wd = work_dir(cfg, "arkitscenes", vid)
    wd = work_dir(cfg, dataset_tag(source), vid)
    if is_done(wd, "frames") and not force:
        return read_json(wd / "scene_meta.json")
    gate = read_json(gate_wd / "gate_a.json")
    drops = DropLog(wd, "step1_frames")
    conv = gate["best_convention"]
    # POSE_INVALID / ORIENTATION_CONFLICT were already logged at GATE A; only the status is inherited here
    status = {"pose_valid": gate["pose_valid"], "orientation_ok": gate["orientation_ok"]}
    sky_auto, sky_meta = gate["sky_auto"], gate["sky_meta"]
    quarters = g.SKY_TO_CW_QUARTERS[sky_auto]

    files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    traj_ts, M = arkit.load_traj(files["lowres_wide.traj"])
    R4 = g.world_transform(np.asarray(gate["R_up"]))
    T_all = np.stack([R4 @ POSE_CONVENTIONS[conv](m) for m in M])

    img_dir, intr_dir, rel_base = frame_source_dirs(cfg, vid, source)
    frames = arkit.list_frames(img_dir, ".png")
    intr = arkit.list_frames(intr_dir, ".pincam")
    intr_ts = np.array([t for t, _ in intr])
    if source == "vga_wide":
        near_tol, gap = acfg["vga_pose_nearest_tol_s"], acfg["vga_pose_interp_max_gap_s"]
    else:
        near_tol, gap = cfg["frames"]["pose_timestamp_tolerance_s"], 0.0  # lowres uses nearest-neighbour matching only

    pose_status = np.load(gate_wd / "pose_check.npz")["status"]
    out = []
    for ts, fname in frames:
        r = g.pose_indices_at_timestamp(ts, traj_ts, near_tol, gap)
        if r is None:
            drops.add("frame", fname, "POSE_TS_MISMATCH", source=source)
            continue
        idx, alpha = r
        if any(pose_status[j] != POSE_GOOD for j in idx):
            drops.add("frame", fname, "POSE_FRAME_INVALID", source=source,
                      pose_status=["good", "bad", "unverified"][max(int(pose_status[j]) for j in idx)])
            continue
        T = T_all[idx[0]] if alpha is None else g.interpolate_pose(T_all[idx[0]], T_all[idx[1]], alpha)
        K, W, H = arkit.load_pincam(intr_dir / intr[int(np.argmin(np.abs(intr_ts - ts)))][1])
        K2, T2, W2, H2 = g.rotate_camera_cw(K, T, W, H, quarters)
        out.append({
            "frame_id": f"f_{len(out):06d}", "source_frame": fname, "rel_path": f"{rel_base}/{source}/{fname}",
            "timestamp": ts, "T_wc": T2, "K": K2, "image_size": [W2, H2], "rotation_cw_quarters": quarters,
            "position": T2[:3, 3], "yaw_deg": g.yaw_deg_from_Twc(T2),
        })
    write_json(wd / "frames_all.json", out)

    long_side = max(out[0]["image_size"]) if out else 0
    status["resolution_ok"] = long_side >= cfg["frames"]["min_rgb_long_side_px"]
    vga_check = verify_against_lowres_depth(cfg, vid, out, files, img_dir) if source == "vga_wide" else None
    if vga_check is not None:
        # the 10-frame depth spot check is diagnostic only and does not decide whether a scene is kept; the scene-level image orientation check still applies
        status["vga_image_orientation_ok"] = vga_check["image_orientation_ok"]
        if not vga_check["image_orientation_ok"]:
            drops.add("scene", vid, "ORIENTATION_CONFLICT", check="vga_vs_lowres_image", median_ncc=vga_check["median_ncc"])
    scene_meta = {
        "scene_id": f"arkit_{vid}", "dataset": "arkitscenes", "source_id": vid, "split": "validation",
        "up_axis": "z", "units": "meters", "pose_convention": "T_wc_opencv",
        "image_size": out[0]["image_size"] if out else None, "frame_source": source,
        "num_frames": len(out), "num_frames_raw": len(frames),
        "traj_convention": conv, "up_axis_detected_raw": gate["up_axis_detected"], "R_up": gate["R_up"],
        "sky_direction_auto": sky_auto, "sky_direction_meta": sky_meta, "rotation_cw_quarters": quarters,
        "status": status, "vga_check": vga_check,
    }
    write_json(wd / "scene_meta.json", scene_meta)
    mark_done(wd, "frames")
    return scene_meta


def verify_against_lowres_depth(cfg, vid, frames, files, img_dir):
    """Step 1 vga pose validation: interpolated pose + vga intrinsics render mesh depth -> resize to lowres size -> compare with the nearest lowres_depth."""
    gcfg = cfg["gate_a"]
    tol = cfg["arkitscenes"]["vga_pose_nearest_tol_s"]
    low = arkit.list_frames(files["lowres_depth"], ".png")
    low_ts = np.array([t for t, _ in low])
    cand = []
    for i, f in enumerate(frames):
        j = int(np.argmin(np.abs(low_ts - f["timestamp"])))
        if abs(low_ts[j] - f["timestamp"]) <= tol:
            cand.append((i, low[j][1]))
    if not cand:
        return {"depth_pass_ratio_diagnostic": False, "pass_ratio": 0.0, "image_orientation_ok": False, "median_ncc": None,
                "frames": [], "note": "no lowres_depth within tolerance"}
    gate = read_json(work_dir(cfg, "arkitscenes", vid) / "gate_a.json")
    mesh = arkit.load_mesh_legacy(files["mesh"])
    raycast = arkit.raycasting_scene(np.asarray(mesh.vertices) @ np.asarray(gate["R_up"]).T, np.asarray(mesh.triangles))
    rng = scene_rng(cfg, f"arkit_{vid}", "vga_check")
    pick = sorted(rng.choice(len(cand), size=min(gcfg["frames_per_scene"], len(cand)), replace=False).tolist())
    q = frames[0]["rotation_cw_quarters"]
    res = []
    for k in pick:
        i, low_name = cand[k]
        f = frames[i]
        K, T, (W, H) = np.asarray(f["K"]), np.asarray(f["T_wc"]), f["image_size"]
        dsen = g.rotate_image_cw(arkit.read_depth_m(files["lowres_depth"] / low_name, cfg["arkitscenes"]["depth_scale"]), q)
        drend = cv2.resize(g.render_depth(raycast, K, T, W, H), (dsen.shape[1], dsen.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = (dsen > gcfg["depth_min_valid_m"]) & (dsen < gcfg["depth_max_valid_m"]) & np.isfinite(drend)
        err = float(np.median(np.abs(drend[valid] - dsen[valid]))) if valid.sum() >= gcfg["min_valid_pixels"] else None
        # The depth comparison only validates pose and intrinsics; image content is compared as well, to confirm vga and lowres originals are stored with the same orientation
        ncc_same, ncc_rot = image_ncc(img_dir / f["source_frame"], files["lowres_wide"] / low_name)
        res.append({"frame": f["source_frame"], "lowres_depth": low_name, "dt_s": abs(arkit.frame_timestamp(low_name) - f["timestamp"]),
                    "median_abs_err_m": err, "valid_px": int(valid.sum()), "ncc": ncc_same, "ncc_rot180": ncc_rot})
    # Per-frame pass criterion as in GATE A (depth only); the image comparison is a scene-level storage-orientation check. vga and lowres frames can be ~16 ms apart,
    # so single-frame NCC drops during fast motion; hence the median is used, and every frame must exceed its 180-degree-rotated control value
    ok = [r["median_abs_err_m"] is not None and r["median_abs_err_m"] < gcfg["depth_median_abs_err_m"] for r in res]
    ncc = np.array([r["ncc"] for r in res])
    orient_ok = bool(np.median(ncc) >= gcfg["vga_image_min_ncc"] and all(r["ncc"] > r["ncc_rot180"] for r in res))
    pass_ratio = float(np.mean(ok))
    return {"depth_pass_ratio_diagnostic": pass_ratio >= gcfg["pass_ratio"], "pass_ratio": pass_ratio,
            "image_orientation_ok": orient_ok, "median_ncc": float(np.median(ncc)), "frames": res}


def image_ncc(vga_path, low_path):
    """Normalised cross-correlation between the vga original resized to lowres size and the lowres original; also returns the value after a 180-degree rotation as a control."""
    low = cv2.imread(str(low_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    vga = cv2.resize(cv2.imread(str(vga_path), cv2.IMREAD_GRAYSCALE), (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)

    def ncc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    return ncc(vga, low), ncc(vga[::-1, ::-1], low)


def visualize_frames(vid: str, source: str, n: int, out_dir):
    """Uprighted display frame + OBB projections (using the updated K, T_wc from frames_all.json)."""
    cfg = load_config()
    wd = work_dir(cfg, dataset_tag(source), vid)
    frames = read_json(wd / "frames_all.json")
    meta = read_json(wd / "scene_meta.json")
    R_up = np.asarray(meta["R_up"])
    objs = arkit.load_annotation(arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)["annotation"])
    root = path_of(cfg, "arkit_3dod_root").resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in np.linspace(0, len(frames) - 1, n + 2).round().astype(int)[1:-1]:
        f = frames[i]
        img = g.rotate_image_cw(cv2.imread(str(root / f["rel_path"])), f["rotation_cw_quarters"])
        draw_obbs(img, objs, np.asarray(f["K"]), np.asarray(f["T_wc"]), R_up)
        cv2.imwrite(str(out_dir / f"{vid}_{source}_{f['timestamp']:.3f}.jpg"), img)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate_a", "frames"])
    ap.add_argument("--scenes", nargs="*", help="list of video_ids; default: pilot_scenes from the config")
    ap.add_argument("--source", default=None, help="frame source for frames; default: arkitscenes.display_frames")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    scenes = [str(s) for s in (args.scenes or cfg["arkitscenes"]["pilot_scenes"])]
    if args.cmd == "gate_a":
        with Pool(args.workers) as pool:
            results = pool.map(gate_a_scene, [(v, args.force) for v in scenes])
        vis_dir = path_of(cfg, "reports") / "gateA"
        for r in results:
            visualize_scene(r["video_id"], cfg["gate_a"]["vis_frames_per_scene"], vis_dir)
            plot_pose_timeline(r["video_id"], vis_dir / "pose_timeline" / f"{r['video_id']}.png")
            print(r["video_id"], r["best_convention"], {c: round(v["pass_ratio"], 2) for c, v in r["conventions"].items()},
                  "up", r["up_axis_detected"], "sky_auto", r["sky_auto"], round(r["sky_agreement"], 2),
                  "pose_check", r["pose_check"], "pose_valid", r["pose_valid"])
    else:
        source = args.source or cfg["arkitscenes"]["display_frames"]
        for vid in scenes:
            m = build_frames(vid, source, args.force)
            visualize_frames(vid, source, cfg["gate_a"]["vis_frames_per_scene"], path_of(cfg, "reports") / "gateA" / source)
            print(vid, m["num_frames"], "/", m["num_frames_raw"], m["image_size"], m["status"],
                  None if m["vga_check"] is None else round(m["vga_check"]["pass_ratio"], 2))


if __name__ == "__main__":
    main()
