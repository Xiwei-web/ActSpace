"""ScanNet++ Step 4: view graph and per-frame visibility (see docs/data_construction.md), plus the ScanNet++-specific mirror (S8)
and magenta-patch (S10) rules.

- View graph: reuses step4_graph_visibility.build_view_graph (action semantics, keep only the largest strongly connected component,
  relax once if the kept fraction is < 70%).
- Visibility: reuses step4_graph_visibility.visibility_for_node / obb_in_frame_frac / classify (five conditions + mesh occlusion +
  sensor-depth occlusion). Sensor depth: the depth png with the same frame number (uint16 millimetres, 256x192), matched to RGB as
  verified at GATE A (RGB and depth share the field of view; intrinsics scale with resolution; here the depth is nearest-neighbour
  upsampled to RGB resolution, which is equivalent to scaling the intrinsics).
- Magenta patches (S10, option b): object points on pure magenta RGB pixels (dilated by magenta_dilate_px) are always invisible.
- Mirrors (S8): for objects whose label contains "mirror" (minus mirror_label_exclude) and objects listed manually in
  mirror_object_overrides, the thinnest box face facing the room is an opaque occluder; points whose projection passes through that
  face and that lie farther than it (by more than occlusion_tolerance_m) are always occluded, whatever the sensor depth says. The
  mirror's own points are unaffected.
- To report the effect of each rule, the states without S8 / S10 (state_base), with S10 only (state_no_mirror) and with S8 only
  (state_no_magenta) are stored as well.
- Two extra conditions (S12; ScanNet++ only; disabled when the threshold is null) downgrade visible to partial after the above:
  shape: short side of the minimum-area bounding rectangle of the convex hull of projected visible points (in original 1920x1440
  pixels) < vis_min_short_side_px;
  regional sharpness: Laplacian variance inside that convex hull (magenta pixels excluded) / reference < vis_min_sharpness_ratio, with
  the reference chosen by vis_sharpness_ref: scene_frames = median whole-image Laplacian variance over all nodes of the scene;
  object_nodes = median regional Laplacian variance of the object over its visible nodes (before these conditions).
  The state before these conditions is stored as state_pre_s12.
- Suspected mirrors (supplement to S8, report only): fraction of pixels inside the box face of flat objects where the sensor depth is
  clearly farther than the face; see suspected_mirrors.

Usage: python -m actspace_build.scannetpp_step4 run [--scenes ...] [--force]
Outputs (work/scannetpp/<sid>/): view_graph.json, visibility.npz (fields as specified in docs/data_construction.md, plus the
comparison fields above), step4_summary.json
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, read_json, scene_rng, write_json
from .scannetpp import sp_cfg, sp_root, sp_work
from .scannetpp_motion import camera_speed
from .scannetpp_step3 import load_mesh
from .step4_graph_visibility import (ACTIONS, STATE_INVISIBLE, STATE_PARTIAL, STATE_VISIBLE, build_view_graph, classify, hull_area,
                                     obb_in_frame_frac, visibility_for_node)

DATASET = "scannetpp"


# ---------------------------------------------------------------- sensor depth, magenta patches
def sensor_depth(cfg: dict, f: dict) -> np.ndarray | None:
    """Sensor depth of the same frame number (metres, 0 = invalid), nearest-neighbour upsampled to RGB resolution."""
    p = sp_root(cfg) / f["depth_rel_path"]
    if not p.exists():
        return None
    gc = cfg["gate_a"]
    d = arkit.read_depth_m(p, sp_cfg(cfg)["depth_scale"])
    d = np.where((d > gc["depth_min_valid_m"]) & (d < gc["depth_max_valid_m"]), d, 0.0)
    W, H = f["image_size"]
    return cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def magenta_mask(img: np.ndarray, sp: dict) -> np.ndarray:
    """Pure magenta pixels (same definition as the Step 2 statistics), dilated by magenta_dilate_px to cover JPEG compression fringes."""
    b, g_, r = (img[..., k].astype(np.int16) for k in range(3))
    m = ((r > sp["magenta_r_min"]) & (b > sp["magenta_b_min"]) & (g_ < sp["magenta_g_max"])).astype(np.uint8)
    k = 2 * sp["magenta_dilate_px"] + 1
    return cv2.dilate(m, np.ones((k, k), np.uint8)).astype(bool) if m.any() else m.astype(bool)


# ---------------------------------------------------------------- mirrors
def mirror_object_ids(sp: dict, sid: str, objs: list[dict]) -> list[int]:
    ex = {x.lower() for x in sp["mirror_label_exclude"]}
    ids = [o["obj_id"] for o in objs if "mirror" in o["source_label"].lower() and o["source_label"].lower() not in ex]
    return sorted(set(ids) | set(sp["mirror_object_overrides"].get(sid, [])))


def mirror_face(o: dict, toward: np.ndarray) -> dict:
    """Of the two box faces along the thinnest axis, the one facing `toward` (mean keyframe camera position): centre, outward normal,
    the two edge directions and half-lengths."""
    c, A, s = np.asarray(o["obb"]["center"]), np.asarray(o["obb"]["axes"]), np.asarray(o["obb"]["size"])
    k = int(np.argmin(s))
    n = A[k] * (1.0 if (toward - c) @ A[k] >= 0 else -1.0)
    u, v = [j for j in range(3) if j != k]
    return {"obj_id": o["obj_id"], "center": c + n * s[k] / 2, "normal": n, "u": A[u], "hu": s[u] / 2, "v": A[v], "hv": s[v] / 2}


def mirror_occluded(P: np.ndarray, C: np.ndarray, face: dict, tol: float) -> np.ndarray:
    """True if the segment from camera C to point P crosses the mirror rectangle and P lies more than tol beyond the intersection.
    Not applied when the camera is behind the mirror."""
    n, F0 = face["normal"], face["center"]
    if len(P) == 0 or (C - F0) @ n <= 0:
        return np.zeros(len(P), bool)
    d = P - C
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((F0 - C) @ n) / denom
    L = np.linalg.norm(d, axis=1)
    ok = (denom < 0) & (s > 0) & (s * L < L - tol)
    X = C + s[:, None] * d
    ok &= (np.abs((X - F0) @ face["u"]) <= face["hu"]) & (np.abs((X - F0) @ face["v"]) <= face["hv"])
    return ok


def face_corners(face: dict) -> np.ndarray:
    c, u, v = face["center"], face["u"] * face["hu"], face["v"] * face["hv"]
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


# ---------------------------------------------------------------- scene context and single node
class SceneCtx:
    def __init__(self, cfg: dict, sid: str):
        self.cfg, self.sid, self.sp = cfg, sid, sp_cfg(cfg)
        # ScanNet++ may override visible_min_ratio (null = use the shared value); ARKit is unaffected
        self.vcfg = {**cfg["visibility"], **({"visible_min_ratio": self.sp["visible_min_ratio"]} if self.sp.get("visible_min_ratio") is not None else {})}
        self.wd = sp_work(cfg) / sid
        self.frames = {f["frame_id"]: f for f in read_json(self.wd / "frames_all.json")}
        self.kf_ids = read_json(self.wd / "keyframes.json")["keyframe_ids"]
        self.objs = read_json(self.wd / "objects.json")
        idx = np.load(self.wd / "object_points.npz")
        _, V, F = load_mesh(cfg, sid)
        self.raycast = arkit.raycasting_scene(V, F)
        rng = scene_rng(cfg, f"scannetpp_{sid}", "visibility_points")
        self.pts = []
        for o in self.objs:
            i = idx[f"obj_{o['obj_id']}"]
            if len(i) > self.vcfg["max_points_per_object"]:
                i = np.sort(rng.choice(i, self.vcfg["max_points_per_object"], replace=False))
            self.pts.append(V[i] if len(i) else np.zeros((0, 3)))
        self.obbs = [{k: np.asarray(o["obb"][k]) for k in ("center", "axes", "size")} for o in self.objs]
        cam_mean = np.mean([self.frames[i]["position"] for i in self.kf_ids], axis=0)
        self.mirror_ids = mirror_object_ids(self.sp, sid, self.objs)
        self.mirrors = [mirror_face(self.objs[m], cam_mean) for m in self.mirror_ids]

    def node(self, fid: str, only: list[int] | None = None, keep_points: bool = False) -> list[dict]:
        """Per-object statistics at one node. If `only` is given, only those objects are computed (for report figures). Each returned item
        also carries frame_lapvar (whole-image Laplacian variance)."""
        f = self.frames[fid]
        K, T, (W, H) = np.asarray(f["K"]), np.asarray(f["T_wc"]), f["image_size"]
        C = T[:3, 3]
        ks = list(range(len(self.objs))) if only is None else list(only)
        sd = sensor_depth(self.cfg, f)
        res = visibility_for_node(self.raycast, K, T, W, H, [self.pts[k] for k in ks], self.vcfg, sd)
        img = cv2.imread(str(sp_root(self.cfg) / f["rel_path"]))
        lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        frame_lapvar = float(lap.var())
        mag = magenta_mask(img, self.sp)
        integ = cv2.integral(mag.astype(np.uint8)) if mag.any() else None
        mir_rects = []
        for fc in self.mirrors:
            uv, z = g.project(face_corners(fc), K, T)
            mir_rects.append(uv[z > 0.05])
        out = []
        for k, r in zip(ks, res):
            P = self.pts[k]
            uv, z = g.project(P, K, T)
            inf = g.in_image(uv, z, W, H)
            Pin, uv_in, vis_base = P[inf], r["uv_in"], r["vis_in"]
            px = np.clip(np.round(uv_in).astype(int), 0, [W - 1, H - 1]) if len(uv_in) else np.zeros((0, 2), int)
            on_mag = mag[px[:, 1], px[:, 0]] if len(px) else np.zeros(0, bool)
            occ_mir = np.zeros(len(Pin), bool)
            for fc in self.mirrors:
                if fc["obj_id"] != self.objs[k]["obj_id"]:
                    occ_mir |= mirror_occluded(Pin, C, fc, self.vcfg["occlusion_tolerance_m"])
            in_frame = obb_in_frame_frac(self.obbs[k]["center"], self.obbs[k]["axes"], self.obbs[k]["size"], K, T, W, H) if r["n_in"] else 0.0

            def st(mask):
                nv = int(mask.sum())
                a = hull_area(uv_in[mask]) / (W * H)
                return classify(r["n_in"], nv, a, self.vcfg, r["n_total"], in_frame), nv, a

            s_base = st(vis_base)
            s_mag = st(vis_base & ~on_mag)
            s_mir = st(vis_base & ~occ_mir)
            vis_fin = vis_base & ~on_mag & ~occ_mir
            s_fin = st(vis_fin)
            short_px, region_lapvar = region_metrics(uv_in[vis_fin], lap, mag)
            # "nearby" (for spot checks): magenta pixels inside the 2D bounding box (padded by 25%) of the object's in-frustum points;
            # whether a mirror appears in the same image (mirror projection intersects the image)
            mag_near = 0
            mir_near = any(len(R) and fc_overlap(R, (0, 0, W, H), W, H) for R in mir_rects)
            if len(uv_in):
                lo, hi = uv_in.min(0), uv_in.max(0)
                pad = 0.25 * (hi - lo) + 10
                x0, y0 = np.clip(np.floor(lo - pad), 0, [W, H]).astype(int)
                x1, y1 = np.clip(np.ceil(hi + pad), 0, [W, H]).astype(int)
                if integ is not None:
                    mag_near = int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
            d = {"n_total": r["n_total"], "n_in": r["n_in"], "in_frame": in_frame, "n_no_depth": r["n_no_depth"],
                 "base": s_base, "no_mirror": s_mag, "no_magenta": s_mir, "final": s_fin,
                 "n_magenta": int((vis_base & on_mag).sum()), "n_mirror": int((vis_base & occ_mir).sum()),
                 "magenta_near_px": mag_near, "mirror_near": mir_near, "has_depth": sd is not None,
                 "short_px": short_px, "region_lapvar": region_lapvar, "frame_lapvar": frame_lapvar,
                 "n_in_magenta": int(on_mag.sum())}
            if keep_points:
                d.update(uv_in=uv_in, vis_base=vis_base, on_mag=on_mag, occ_mir=occ_mir, mag=mag, mir_rects=mir_rects)
            out.append(d)
        return out


def region_metrics(uv: np.ndarray, lap: np.ndarray, mag: np.ndarray) -> tuple[float, float]:
    """Short side (pixels) of the minimum-area bounding rectangle of the convex hull of projected visible points, and the Laplacian
    variance inside the hull (magenta pixels excluded). Returns (0, 0) for fewer than 3 points."""
    if len(uv) < 3:
        return 0.0, 0.0
    hull = cv2.convexHull(uv.astype(np.float32))
    short = float(min(cv2.minAreaRect(hull)[1]))
    H, W = lap.shape
    x0, y0 = np.clip(np.floor(uv.min(0)), 0, [W - 1, H - 1]).astype(int)
    x1, y1 = np.clip(np.ceil(uv.max(0)) + 1, 1, [W, H]).astype(int)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.round(hull.reshape(-1, 2) - [x0, y0]).astype(np.int32), 1)
    m = m.astype(bool) & ~mag[y0:y1, x0:x1]
    return short, float(lap[y0:y1, x0:x1][m].var()) if m.sum() >= 2 else 0.0


def apply_s12(state: np.ndarray, short_px: np.ndarray, region_lapvar: np.ndarray, frame_lapvar: np.ndarray, sp: dict):
    """The two extra S12 conditions: visible entries failing the shape or regional-sharpness test are downgraded to partial.
    Returns (new state, sharpness ratio, shape downgrades, sharpness downgrades)."""
    vis = state == STATE_VISIBLE
    if sp["vis_sharpness_ref"] == "scene_frames":
        ref = np.full(state.shape, np.median(frame_lapvar) if len(frame_lapvar) else 1.0)
    else:
        ref = np.ones(state.shape)
        for o in range(state.shape[0]):
            if vis[o].any():
                ref[o] = np.median(region_lapvar[o, vis[o]])
    ratio = np.where(ref > 0, region_lapvar / np.maximum(ref, 1e-9), 0.0)
    thr_s, thr_r = sp["vis_min_short_side_px"], sp["vis_min_sharpness_ratio"]
    d_shape = vis & (short_px < thr_s) if thr_s is not None else np.zeros_like(vis)
    d_sharp = vis & (ratio < thr_r) if thr_r is not None else np.zeros_like(vis)
    out = state.copy()
    out[d_shape | d_sharp] = STATE_PARTIAL
    return out, ratio, d_shape, d_sharp


def motion_fast_nodes(ang: np.ndarray, lin: np.ndarray, max_ang, max_lin) -> np.ndarray:
    """Nodes with angular speed >= max_ang or linear speed >= max_lin (a test is disabled when its threshold is None)."""
    fast = np.zeros(len(ang), bool)
    if max_ang is not None:
        fast |= ang >= max_ang
    if max_lin is not None:
        fast |= lin >= max_lin
    return fast


def node_pose_err(wd, frames: dict, nodes: list[str]) -> np.ndarray:
    """Median depth error of each node in the GATE A per-pose check (metres; NaN if unverified)."""
    pc = np.load(wd / "pose_check.npz")
    err = dict(zip(pc["frame_ids"].tolist(), pc["err"].tolist()))
    return np.array([err.get(int(frames[n]["source_frame"].split("_")[1]), np.nan) for n in nodes], np.float32)


def pose_suspect_nodes(err: np.ndarray, lin: np.ndarray, ratio, min_lin) -> np.ndarray:
    """Nodes with error >= (median node error of the scene) x ratio and linear speed >= min_lin (disabled if either threshold is None)."""
    if ratio is None or min_lin is None or not np.isfinite(err).any():
        return np.zeros(len(err), bool)
    med = np.nanmedian(err)
    return (np.nan_to_num(err, nan=0.0) >= ratio * med) & (lin >= min_lin)


def apply_magenta_frac(state, n_mag, n_in, thr):
    """Visible entries whose fraction of in-frustum points on magenta pixels is > thr are downgraded to partial (disabled if thr is None).
    Returns (new state, downgrade mask)."""
    if thr is None:
        return state, np.zeros(state.shape, bool)
    frac = np.where(n_in > 0, n_mag / np.maximum(n_in, 1), 0.0)
    d = (state == STATE_VISIBLE) & (frac > thr)
    out = state.copy()
    out[d] = STATE_PARTIAL
    return out, d


def suspected_mirrors(ctx: "SceneCtx", nodes: list[str], sp: dict) -> list[dict]:
    """For flat objects (thinnest box edge <= mirror_scan_max_thickness_m, middle edge >= mirror_scan_min_face_m; walls / floors /
    ceilings / door frames excluded): fraction of the room-facing box-face pixels (not hidden by objects in front) whose sensor depth is
    more than mirror_scan_far_m beyond the face. Computed at depth resolution; report only, does not change any state."""
    skip = {"wall", "floor", "ceiling", "doorframe"}
    cam_mean = np.mean([ctx.frames[i]["position"] for i in ctx.kf_ids], axis=0)
    out = []
    for o in ctx.objs:
        s = np.sort(np.asarray(o["obb"]["size"]))
        if o["category_fine"] in skip or s[0] > sp["mirror_scan_max_thickness_m"] or s[1] < sp["mirror_scan_min_face_m"]:
            continue
        fc = mirror_face(o, cam_mean)
        far = tot = n_nodes = 0
        for fid in nodes:
            f = ctx.frames[fid]
            Kd, T, (Wd, Hd) = np.asarray(f["depth_K"]), np.asarray(f["T_wc"]), f["depth_size"]
            C, R = T[:3, 3], T[:3, :3]
            if (C - fc["center"]) @ fc["normal"] <= 0:
                continue
            uv, z = g.project(face_corners(fc), Kd, T)
            if (z <= 0.05).any():
                continue
            m = np.zeros((Hd, Wd), np.uint8)
            cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
            vv, uu = np.nonzero(m)
            if len(vv) < sp["mirror_scan_min_px"]:
                continue
            dirs = np.stack([(uu - Kd[0, 2]) / Kd[0, 0], (vv - Kd[1, 2]) / Kd[1, 1], np.ones(len(uu))], 1) @ R.T
            zf = ((fc["center"] - C) @ fc["normal"]) / (dirs @ fc["normal"])  # box-face depth (camera z)
            dpath = sp_root(ctx.cfg) / f["depth_rel_path"]
            d = arkit.read_depth_m(dpath, sp["depth_scale"])[vv, uu]
            ok = (d > 0) & (d >= zf - 0.10)  # drop pixels hidden by objects in front
            if ok.sum() < sp["mirror_scan_min_px"]:
                continue
            far += int((d[ok] > zf[ok] + sp["mirror_scan_far_m"]).sum())
            tot += int(ok.sum())
            n_nodes += 1
        if tot:
            out.append({"obj_id": o["obj_id"], "source_label": o["source_label"], "size": np.round(s, 2).tolist(), "nodes": n_nodes,
                        "far_frac": far / tot, "suspected": n_nodes >= sp["mirror_scan_min_nodes"] and far / tot >= sp["mirror_scan_min_far_frac"]})
    return sorted(out, key=lambda r: -r["far_frac"])


def fc_overlap(R: np.ndarray, box, W: int, H: int) -> bool:
    """Whether the bounding box of the projected mirror points (clipped to the image) intersects box."""
    x0, y0, x1, y1 = box
    lo, hi = np.clip(R.min(0), 0, [W, H]), np.clip(R.max(0), 0, [W, H])
    return bool(hi[0] > lo[0] and hi[1] > lo[1] and lo[0] < x1 and hi[0] > x0 and lo[1] < y1 and hi[1] > y0)


# ---------------------------------------------------------------- scene
def graph_visibility_scene(sid: str, force: bool = False) -> dict:
    cfg = load_config()
    wd = sp_work(cfg) / sid
    if is_done(wd, "graph_visibility") and not force:
        return read_json(wd / "step4_summary.json")
    ctx = SceneCtx(cfg, sid)
    drops = DropLog(wd, "step4")
    kfs = [ctx.frames[i] for i in ctx.kf_ids]
    graph, ginfo = build_view_graph(kfs, cfg["view_graph"])
    if not ginfo["ok"]:
        drops.add("scene", sid, "GRAPH_DISCONNECTED", **ginfo)
    write_json(wd / "view_graph.json", graph)

    nodes = graph["nodes"]
    O, N = len(ctx.objs), len(nodes)
    z = lambda dt: np.zeros((O, N), dt)
    A = {k: z(np.uint8) for k in ("state", "state_base", "state_no_mirror", "state_no_magenta")}
    A.update({k: z(np.float32) for k in ("rho", "completeness", "area_frac", "in_frame_frac")})
    A.update({k: z(np.int32) for k in ("n_infrustum", "n_visible", "n_no_depth", "n_magenta", "n_mirror", "magenta_near_px")})
    A["mirror_near"] = z(np.uint8)
    A["min_rect_short_px"], A["region_lapvar"] = z(np.float32), z(np.float32)
    A["n_infrustum_magenta"] = z(np.int32)
    has_depth, frame_lapvar = np.zeros(N, bool), np.zeros(N, np.float32)
    for v, fid in enumerate(nodes):
        res = ctx.node(fid)
        has_depth[v] = res[0]["has_depth"] if res else False
        frame_lapvar[v] = res[0]["frame_lapvar"] if res else 0.0
        for o, r in enumerate(res):
            st, nv, a = r["final"]
            A["state"][o, v], A["n_visible"][o, v], A["area_frac"][o, v] = st, nv, a
            A["state_base"][o, v], A["state_no_mirror"][o, v], A["state_no_magenta"][o, v] = r["base"][0], r["no_mirror"][0], r["no_magenta"][0]
            A["n_infrustum"][o, v], A["in_frame_frac"][o, v], A["n_no_depth"][o, v] = r["n_in"], r["in_frame"], r["n_no_depth"]
            A["rho"][o, v] = nv / r["n_in"] if r["n_in"] else 0.0
            A["completeness"][o, v] = nv / r["n_total"] if r["n_total"] else 0.0
            A["n_magenta"][o, v], A["n_mirror"][o, v] = r["n_magenta"], r["n_mirror"]
            A["magenta_near_px"][o, v], A["mirror_near"][o, v] = r["magenta_near_px"], r["mirror_near"]
            A["min_rect_short_px"][o, v], A["region_lapvar"][o, v] = r["short_px"], r["region_lapvar"]
            A["n_infrustum_magenta"][o, v] = r["n_in_magenta"]
    # extra S12 conditions (only downgrade visible; raw values such as rho / completeness are unchanged)
    A["state_pre_s12"] = A["state"]
    A["state"], A["sharpness_ratio"], d_shape, d_sharp = apply_s12(A["state"], A["min_rect_short_px"], A["region_lapvar"], frame_lapvar, ctx.sp)
    A["s12_down_shape"], A["s12_down_sharp"] = d_shape.astype(np.uint8), d_sharp.astype(np.uint8)
    # S10 supplement: visible -> partial when the fraction of in-frustum points on magenta pixels is > vis_max_magenta_frac
    A["state"], d_magfrac = apply_magenta_frac(A["state"], A["n_infrustum_magenta"], A["n_infrustum"], ctx.sp.get("vis_max_magenta_frac"))
    A["s10_down_frac"] = d_magfrac.astype(np.uint8)
    # motion blur: at nodes whose camera angular / linear speed exceeds the threshold, all visible objects become partial (the node stays in the view graph)
    spd = camera_speed(cfg, sid)
    ang = np.array([spd[n]["ang_deg_s"] if spd[n]["ang_deg_s"] is not None else 0.0 for n in nodes], np.float32)
    lin = np.array([spd[n]["lin_m_s"] if spd[n]["lin_m_s"] is not None else 0.0 for n in nodes], np.float32)
    fast = motion_fast_nodes(ang, lin, ctx.sp.get("motion_max_ang_deg_s"), ctx.sp.get("motion_max_lin_m_s"))
    d_motion = (A["state"] == STATE_VISIBLE) & fast[None, :]
    A["state"][d_motion] = STATE_PARTIAL
    A["motion_down"] = d_motion.astype(np.uint8)
    # lateral pose drift: at nodes whose per-pose depth error >= (median node error of the scene) x pose_err_ratio_max and whose linear
    # speed >= pose_err_min_lin_m_s, all visible objects become partial
    perr = node_pose_err(wd, ctx.frames, nodes)
    suspect = pose_suspect_nodes(perr, lin, ctx.sp.get("pose_err_ratio_max"), ctx.sp.get("pose_err_min_lin_m_s"))
    d_pose = (A["state"] == STATE_VISIBLE) & suspect[None, :]
    A["state"][d_pose] = STATE_PARTIAL
    A["pose_suspect_down"] = d_pose.astype(np.uint8)
    f16 = ("rho", "completeness", "area_frac", "in_frame_frac", "sharpness_ratio")
    np.savez_compressed(wd / "visibility.npz", obj_ids=np.array([o["obj_id"] for o in ctx.objs]), frame_ids=np.array(nodes),
                        **{k: (v.astype(np.float16) if k in f16 else v) for k, v in A.items()}, has_sensor_depth=has_depth, frame_lapvar=frame_lapvar,
                        cam_ang_deg_s=ang, cam_lin_m_s=lin, motion_fast_node=fast, pose_err_m=perr, pose_suspect_node=suspect,
                        mirror_obj_ids=np.array(ctx.mirror_ids, dtype=np.int64))
    S, B, NM, NG, P12 = A["state"], A["state_base"], A["state_no_mirror"], A["state_no_magenta"], A["state_pre_s12"]
    vis = S == STATE_VISIBLE
    summary = {"scene_id": sid, "num_keyframes": len(kfs), "num_graph_nodes": N, **ginfo,
               "edges_per_action": {a: sum(graph["edges"][n][a] is not None for n in nodes) for a in ACTIONS},
               "nodes_with_sensor_depth": int(has_depth.sum()),
               "infrustum_points_without_valid_depth_frac": float(A["n_no_depth"].sum() / max(int(A["n_infrustum"].sum()), 1)),
               "visible_pairs": {"base": int((B == STATE_VISIBLE).sum()), "final": int(vis.sum())},
               "magenta": {"nodes_with_mask": int(sum((A["magenta_near_px"][:, v] > 0).any() for v in range(N))),
                           "pairs_affected": int((A["n_magenta"] > 0).sum()),
                           "visible_downgraded": int(((B == STATE_VISIBLE) & (NM != STATE_VISIBLE)).sum()),
                           "state_changed": int((B != NM).sum())},
               "mirror": {"mirror_obj_ids": ctx.mirror_ids,
                          "mirror_labels": [ctx.objs[m]["source_label"] for m in ctx.mirror_ids],
                          "pairs_affected": int((A["n_mirror"] > 0).sum()),
                          "state_changed": int((NM != P12).sum()),
                          "visible_downgraded": int(((NM == STATE_VISIBLE) & (P12 != STATE_VISIBLE)).sum())},
               "s12": {"thresholds": {k: ctx.sp[k] for k in ("vis_min_short_side_px", "vis_min_sharpness_ratio", "vis_sharpness_ref")},
                       "visible_before": int((P12 == STATE_VISIBLE).sum()), "visible_after": int(vis.sum()),
                       "down_shape": int(d_shape.sum()), "down_sharp": int(d_sharp.sum()), "down_both": int((d_shape & d_sharp).sum())},
               "magenta_frac": {"threshold": ctx.sp.get("vis_max_magenta_frac"), "visible_downgraded": int(d_magfrac.sum())},
               "visible_min_ratio": ctx.vcfg["visible_min_ratio"],
               "motion": {"max_ang_deg_s": ctx.sp.get("motion_max_ang_deg_s"), "max_lin_m_s": ctx.sp.get("motion_max_lin_m_s"),
                          "fast_nodes": int(fast.sum()), "visible_downgraded": int(d_motion.sum())},
               "pose_suspect": {"err_ratio_max": ctx.sp.get("pose_err_ratio_max"), "min_lin_m_s": ctx.sp.get("pose_err_min_lin_m_s"),
                                "scene_median_err_m": float(np.nanmedian(perr)) if np.isfinite(perr).any() else None,
                                "nodes": int(suspect.sum()), "visible_downgraded": int(d_pose.sum())},
               "suspected_mirrors": suspected_mirrors(ctx, nodes, ctx.sp),
               "objects": [{"obj_id": o["obj_id"], "category": o["category"], "referable": o["referable"],
                            "num_visible_nodes": int(vis[k].sum()), "num_visible_nodes_pre_s12": int((P12[k] == STATE_VISIBLE).sum()),
                            "num_partial_nodes": int((S[k] == STATE_PARTIAL).sum()),
                            "num_invisible_nodes": int((S[k] == STATE_INVISIBLE).sum())} for k, o in enumerate(ctx.objs)]}
    write_json(wd / "step4_summary.json", summary)
    mark_done(wd, "graph_visibility")
    return summary


def _run(a):
    sid, force = a
    return graph_visibility_scene(sid, force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    scenes = args.scenes or sp_cfg(cfg)["pilot_scenes"]
    from multiprocessing import Pool
    disconnected = []
    with Pool(min(args.workers, len(scenes))) as pool:
        for s in pool.imap(_run, [(sid, args.force) for sid in scenes]):
            if not s["ok"]:
                disconnected.append(s["scene_id"])
            print({k: v for k, v in s.items() if k != "objects"},
                  "referable visible-node counts:", [o["num_visible_nodes"] for o in s["objects"] if o["referable"]], flush=True)
    frac = len(disconnected) / len(scenes) if scenes else 0.0
    if frac > cfg["view_graph"]["stop_if_disconnected_frac"]:
        print(f"STOP: {len(disconnected)}/{len(scenes)} scenes ({frac:.0%}) dropped as GRAPH_DISCONNECTED: {disconnected}", flush=True)


if __name__ == "__main__":
    main()
