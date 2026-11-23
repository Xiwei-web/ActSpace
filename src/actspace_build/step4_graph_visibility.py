"""Step 4: view graph construction and per-frame visibility.

Usage: python -m actspace_build.step4_graph_visibility --dataset arkitscenes --scenes ... [--force]
Input: frames_all.json, keyframes.json, objects.json, object_points.npz, scene_meta.json
Outputs: view_graph.json, visibility.npz, step4_summary.json; reports/step4/*.jpg
"""
from __future__ import annotations

import argparse

import cv2
import networkx as nx
import numpy as np

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, scene_rng, work_dir, write_json

ACTIONS = ("forward", "backward", "left", "right", "turn_left", "turn_right")
STATE_INVISIBLE, STATE_PARTIAL, STATE_VISIBLE = 0, 1, 2


# ---------------------------------------------------------------- view graph

def action_edges(pos: np.ndarray, yaw: np.ndarray, vc: dict, move_max: float) -> list[dict]:
    """For each node, pick at most one target node (index) per action according to the action semantics table; None if there is no candidate."""
    n = len(pos)
    edges = []
    idx = np.arange(n)
    for i in range(n):
        d = pos[:, :2] - pos[i, :2]
        dist = np.linalg.norm(d, axis=1)
        phi = g.wrap_deg(np.degrees(np.arctan2(d[:, 1], d[:, 0])) - yaw[i])
        dyaw = g.wrap_deg(yaw - yaw[i])
        other = idx != i
        move = other & (dist >= vc["move_min_m"]) & (dist <= move_max) & (np.abs(dyaw) <= vc["move_max_dyaw_deg"])
        fh, bh = vc["front_half_angle_deg"], vc["back_min_angle_deg"]
        turn = other & (dist <= vc["turn_max_translation_m"])
        cands = {
            "forward": (move & (np.abs(phi) <= fh), dist),
            "backward": (move & (np.abs(phi) >= bh), dist),
            "left": (move & (phi > fh) & (phi < bh), dist),
            "right": (move & (phi < -fh) & (phi > -bh), dist),
            "turn_left": (turn & (dyaw >= vc["turn_min_deg"]) & (dyaw <= vc["turn_max_deg"]), np.abs(dyaw - vc["turn_target_deg"])),
            "turn_right": (turn & (dyaw <= -vc["turn_min_deg"]) & (dyaw >= -vc["turn_max_deg"]), np.abs(dyaw + vc["turn_target_deg"])),
        }
        e = {}
        for a in ACTIONS:
            mask, key = cands[a]
            js = idx[mask]
            e[a] = int(js[np.lexsort((js, key[js]))[0]]) if len(js) else None  # ties go to the smaller index
        edges.append(e)
    return edges


def largest_scc(edges: list[dict]) -> list[int]:
    G = nx.DiGraph()
    G.add_nodes_from(range(len(edges)))
    G.add_edges_from((i, j) for i, e in enumerate(edges) for j in e.values() if j is not None)
    comps = sorted(nx.strongly_connected_components(G), key=lambda c: (-len(c), min(c)))
    return sorted(comps[0])


def build_view_graph(kfs: list[dict], vc: dict):
    pos = np.array([f["position"] for f in kfs])
    yaw = np.array([f["yaw_deg"] for f in kfs])
    for move_max, relaxed in ((vc["move_max_m"], False), (vc["move_max_m_relaxed"], True)):
        edges = action_edges(pos, yaw, vc, move_max)
        scc = largest_scc(edges)
        ratio = len(scc) / len(kfs)
        if ratio >= vc["min_scc_ratio"]:
            break
    keep = set(scc)
    ids = [f["frame_id"] for f in kfs]
    graph = {"nodes": [ids[i] for i in scc],
             "edges": {ids[i]: {a: (ids[j] if j is not None and j in keep else None) for a, j in edges[i].items()} for i in scc}}
    return graph, {"scc_ratio": ratio, "relaxed": relaxed, "move_max_m": move_max, "ok": ratio >= vc["min_scc_ratio"]}


# ---------------------------------------------------------------- visibility

def hull_area(uv: np.ndarray) -> float:
    if len(uv) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32))))


def visibility_for_node(raycast, K, T_wc, W, H, pts_list, vcfg, sensor_depth=None):
    """Per-object visibility statistics. A point is "not occluded" if the mesh ray is not occluded; if sensor_depth is given (sensor z depth in metres at the
    frame's resolution and orientation, 0 = invalid), sensor depth must also consider it not occluded.
    Returns a dict per object: n_total, n_in, n_vis, area, n_vis_mesh, area_mesh, n_no_depth, uv_in, vis_in."""
    import open3d as o3d
    allp = np.concatenate(pts_list)
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts_list)])
    uv, z = g.project(allp, K, T_wc)
    inf = g.in_image(uv, z, W, H)
    vis_mesh = np.zeros(len(allp), dtype=bool)
    no_depth = np.zeros(len(allp), dtype=bool)
    if inf.any():
        c = T_wc[:3, 3]
        d = allp[inf] - c
        dist = np.linalg.norm(d, axis=1)
        rays = np.concatenate([np.broadcast_to(c, d.shape), d / dist[:, None]], axis=1).astype(np.float32)
        t = raycast.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis_mesh[np.where(inf)[0]] = t >= dist - vcfg["occlusion_tolerance_m"]
    vis = vis_mesh.copy()
    if sensor_depth is not None and inf.any():
        idx = np.where(inf)[0]
        px = np.clip(np.round(uv[idx]).astype(int), 0, [W - 1, H - 1])
        ds = sensor_depth[px[:, 1], px[:, 0]]
        valid = ds > 0
        # pixels with invalid sensor depth cannot show occlusion; the mesh alone decides
        sensor_ok = ~valid | (ds >= z[idx] - vcfg["sensor_occlusion_tolerance_m"])
        vis[idx] &= sensor_ok
        no_depth[idx] = ~valid
    out = []
    for k in range(len(pts_list)):
        m = owner == k
        out.append({"n_total": int(m.sum()), "n_in": int(inf[m].sum()), "n_vis": int(vis[m].sum()),
                    "area": hull_area(uv[m & vis]) / (W * H), "n_vis_mesh": int(vis_mesh[m].sum()),
                    "area_mesh": hull_area(uv[m & vis_mesh]) / (W * H), "n_no_depth": int(no_depth[m].sum()),
                    "uv_in": uv[m & inf], "vis_in": vis[m & inf]})
    return out


_FACE_GRID = np.linspace(-1.0, 1.0, 9)


def obb_in_frame_frac(center, axes, size, K, T_wc, W, H, near=0.05) -> float:
    """Fraction of the projected OBB area of an object that lies inside the image. Grid points are taken on the 6 OBB faces,
    points in front of the camera are projected, their convex hull approximates the projected OBB region, and it is intersected with the image rectangle. Returns 0 if the box is entirely behind the camera."""
    u, v = np.meshgrid(_FACE_GRID, _FACE_GRID)
    u, v = u.ravel(), v.ravel()
    faces = []
    for k in range(3):
        for sgn in (-1.0, 1.0):
            f = np.zeros((len(u), 3))
            f[:, k] = sgn
            f[:, (k + 1) % 3], f[:, (k + 2) % 3] = u, v
            faces.append(f)
    local = np.concatenate(faces) * (0.5 * np.asarray(size))
    P = np.asarray(center) + local @ np.asarray(axes)
    uv, z = g.project(P, K, T_wc)
    uv = uv[z > near]
    if len(uv) < 3:
        return 0.0
    hull = cv2.convexHull(uv.astype(np.float32))
    total = float(cv2.contourArea(hull))
    if total <= 0:
        return 0.0
    rect = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    inter, _ = cv2.intersectConvexConvex(hull, rect)
    return float(inter) / total


def classify(n_in, n_vis, area, vcfg, n_total=None, in_frame=None) -> int:
    """visible: rho = visible points / in-frustum points >= visible_min_ratio, area and point count above threshold, completeness = visible points / all sample points
    of the object >= visible_min_completeness (so an object showing only an edge or corner is not visible), and the fraction of the projected object OBB area inside the image
    in_frame >= visible_min_in_frame_frac (excludes objects cut off by the image border). n_total / in_frame are not checked when missing."""
    if n_in == 0:
        return STATE_INVISIBLE
    rho = n_vis / n_in
    complete = n_total is None or n_vis / n_total >= vcfg["visible_min_completeness"]
    framed = in_frame is None or in_frame >= vcfg["visible_min_in_frame_frac"]
    if rho >= vcfg["visible_min_ratio"] and area >= vcfg["visible_min_area_frac"] and n_vis >= vcfg["visible_min_points"] and complete and framed:
        return STATE_VISIBLE
    if rho < vcfg["invisible_max_ratio"]:
        return STATE_INVISIBLE
    return STATE_PARTIAL


def sensor_depth_for_frame(cfg, f, depth_list, depth_ts, depth_dir):
    """The lowres_depth within the time tolerance of the frame, uprighted and resized to the frame resolution (nearest neighbour); None if there is no match."""
    j = int(np.argmin(np.abs(depth_ts - f["timestamp"])))
    if abs(depth_ts[j] - f["timestamp"]) > cfg["visibility"]["sensor_depth_tol_s"]:
        return None
    gc = cfg["gate_a"]
    d = arkit.read_depth_m(depth_dir / depth_list[j][1], cfg["arkitscenes"]["depth_scale"])
    d = np.where((d > gc["depth_min_valid_m"]) & (d < gc["depth_max_valid_m"]), d, 0.0)
    d = g.rotate_image_cw(d, f["rotation_cw_quarters"])
    W, H = f["image_size"]
    return cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def graph_visibility_scene(dataset: str, vid: str, force: bool = False, n_vis_images: int = 5):
    cfg = load_config()
    wd = work_dir(cfg, dataset, vid)
    obj_wd = work_dir(cfg, "arkitscenes", vid)
    if is_done(wd, "graph_visibility") and not force:
        return read_json(wd / "step4_summary.json")
    drops = DropLog(wd, "step4")
    frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
    kf = read_json(wd / "keyframes.json")
    kfs = [frames[i] for i in kf["keyframe_ids"]]
    graph, ginfo = build_view_graph(kfs, cfg["view_graph"])
    if not ginfo["ok"]:
        drops.add("scene", vid, "GRAPH_DISCONNECTED", **ginfo)
    write_json(wd / "view_graph.json", graph)

    meta = read_json(wd / "scene_meta.json")
    objs = read_json(obj_wd / "objects.json")
    pts_idx = np.load(obj_wd / "object_points.npz")
    files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    mesh = arkit.load_mesh_legacy(files["mesh"])
    V = np.asarray(mesh.vertices) @ np.asarray(meta["R_up"]).T
    raycast = arkit.raycasting_scene(V, np.asarray(mesh.triangles))
    vcfg = cfg["visibility"]
    rng = scene_rng(cfg, f"arkit_{vid}", "visibility_points")
    pts_list = []
    for o in objs:
        idx = pts_idx[f"obj_{o['obj_id']}"]
        if len(idx) > vcfg["max_points_per_object"]:
            idx = np.sort(rng.choice(idx, vcfg["max_points_per_object"], replace=False))
        pts_list.append(V[idx] if len(idx) else np.zeros((0, 3)))

    nodes = graph["nodes"]
    O, N = len(objs), len(nodes)
    state = np.zeros((O, N), np.uint8)
    rho = np.zeros((O, N), np.float32)
    area = np.zeros((O, N), np.float32)
    n_in_all = np.zeros((O, N), np.int32)
    n_vis_all = np.zeros((O, N), np.int32)
    complete = np.zeros((O, N), np.float32)
    # to report how many decisions each of the two newer rules changes: legacy = no completeness, mesh only; mesh_only = with completeness, mesh only
    state_legacy = np.zeros((O, N), np.uint8)
    state_mesh_only = np.zeros((O, N), np.uint8)
    has_depth = np.zeros(N, bool)
    in_frame = np.zeros((O, N), np.float32)
    obbs = [{k: np.asarray(o["obb"][k]) for k in ("center", "axes", "size")} for o in objs]
    n_no_depth = np.zeros((O, N), np.int32)
    files3 = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    depth_list = arkit.list_frames(files3["lowres_depth"], ".png")
    depth_ts = np.array([t for t, _ in depth_list])
    vis_pick = set(np.linspace(0, N - 1, n_vis_images).round().astype(int).tolist()) if N else set()
    vis_dir = path_of(cfg, "reports") / "step4" / dataset
    data_root = path_of(cfg, "arkit_3dod_root").resolve().parents[1]
    for v, fid in enumerate(nodes):
        f = frames[fid]
        K, T, (W, H) = np.asarray(f["K"]), np.asarray(f["T_wc"]), f["image_size"]
        sd = sensor_depth_for_frame(cfg, f, depth_list, depth_ts, files3["lowres_depth"])
        has_depth[v] = sd is not None
        res = visibility_for_node(raycast, K, T, W, H, pts_list, vcfg, sd)
        for o, r in enumerate(res):
            if r["n_in"]:
                in_frame[o, v] = obb_in_frame_frac(obbs[o]["center"], obbs[o]["axes"], obbs[o]["size"], K, T, W, H)
            n_in_all[o, v], n_vis_all[o, v] = r["n_in"], r["n_vis"]
            rho[o, v] = r["n_vis"] / r["n_in"] if r["n_in"] else 0.0
            area[o, v] = r["area"]
            complete[o, v] = r["n_vis"] / r["n_total"] if r["n_total"] else 0.0
            n_no_depth[o, v] = r["n_no_depth"]
            state[o, v] = classify(r["n_in"], r["n_vis"], r["area"], vcfg, r["n_total"], in_frame[o, v])
            state_mesh_only[o, v] = classify(r["n_in"], r["n_vis_mesh"], r["area_mesh"], vcfg, r["n_total"], in_frame[o, v])
            state_legacy[o, v] = classify(r["n_in"], r["n_vis_mesh"], r["area_mesh"], vcfg)
        if v in vis_pick:
            draw_visibility(data_root, f, objs, res, state[:, v], rho[:, v], area[:, v], vis_dir / f"{vid}_{fid}.jpg")

    np.savez_compressed(wd / "visibility.npz", obj_ids=np.array([o["obj_id"] for o in objs]), frame_ids=np.array(nodes),
                        state=state, rho=rho.astype(np.float16), area_frac=area.astype(np.float16),
                        n_infrustum=n_in_all, n_visible=n_vis_all, completeness=complete.astype(np.float16),
                        in_frame_frac=in_frame.astype(np.float16),
                        state_legacy=state_legacy, state_mesh_only=state_mesh_only, has_sensor_depth=has_depth, n_no_depth=n_no_depth)
    num_visible_nodes = (state == STATE_VISIBLE).sum(1)
    summary = {"num_keyframes": len(kfs), "num_graph_nodes": N, **ginfo,
               "edges_per_action": {a: sum(graph["edges"][n][a] is not None for n in nodes) for a in ACTIONS},
               # how many (object, node) visible decisions each newer rule changes
               "visible_pairs": {"legacy": int((state_legacy == STATE_VISIBLE).sum()),
                                 "with_completeness": int((state_mesh_only == STATE_VISIBLE).sum()),
                                 "final_with_sensor_depth": int((state == STATE_VISIBLE).sum())},
               "visible_lost_by_completeness": int(((state_legacy == STATE_VISIBLE) & (state_mesh_only != STATE_VISIBLE)).sum()),
               "visible_lost_by_sensor_depth": int(((state_mesh_only == STATE_VISIBLE) & (state != STATE_VISIBLE)).sum()),
               "visible_gained_by_sensor_depth": int(((state_mesh_only != STATE_VISIBLE) & (state == STATE_VISIBLE)).sum()),
               "nodes_with_sensor_depth": int(has_depth.sum()),
               "infrustum_points_without_valid_depth_frac": float(n_no_depth.sum() / max(int(n_in_all[:, has_depth].sum()), 1)),
               "objects": [{"obj_id": o["obj_id"], "category": o["category"], "referable": o["referable"],
                            "num_visible_nodes": int(num_visible_nodes[k]), "num_partial_nodes": int((state[k] == STATE_PARTIAL).sum())}
                           for k, o in enumerate(objs)]}
    write_json(wd / "step4_summary.json", summary)
    mark_done(wd, "graph_visibility")
    return summary


_STATE_COLOR = {STATE_VISIBLE: (0, 255, 0), STATE_PARTIAL: (0, 200, 255), STATE_INVISIBLE: (0, 0, 255)}


def draw_visibility(data_root, f, objs, res, st, rh, ar, path):
    """Visibility overlay: each object's in-frustum points (bright = not occluded, dark = occluded); the label shows the decision, rho and a."""
    img = g.rotate_image_cw(cv2.imread(str(data_root / f["rel_path"])), f["rotation_cw_quarters"])
    img = cv2.resize(img, None, fx=2, fy=2) if max(img.shape) < 640 else img
    s = img.shape[1] / f["image_size"][0]
    for k, r in enumerate(res):
        n_in, uv, vmask = r["n_in"], r["uv_in"], r["vis_in"]
        if n_in == 0:
            continue
        col = _STATE_COLOR[int(st[k])]
        dark = tuple(int(c * 0.35) for c in col)
        for (u, v), ok in zip(uv * s, vmask):
            cv2.circle(img, (int(round(u)), int(round(v))), 1, col if ok else dark, -1)
        c = (uv[vmask].mean(0) if vmask.any() else uv.mean(0)) * s
        txt = f"{objs[k]['category']}#{objs[k]['obj_id']} {['INV', 'PART', 'VIS'][int(st[k])]} r={rh[k]:.2f} a={ar[k]:.4f}"
        cv2.putText(img, txt, (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arkitscenes")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    scenes = [str(s) for s in (args.scenes or cfg["arkitscenes"]["pilot_scenes"])]
    disconnected = []
    for vid in scenes:
        s = graph_visibility_scene(args.dataset, vid, args.force)
        if not s["ok"]:
            disconnected.append(vid)
        print(vid, {k: v for k, v in s.items() if k != "objects"},
              "referable visible-node counts:", [o["num_visible_nodes"] for o in s["objects"] if o["referable"]])
    # Stop and report when the share of scenes dropped for view-graph disconnection exceeds the threshold
    frac = len(disconnected) / len(scenes) if scenes else 0.0
    if frac > cfg["view_graph"]["stop_if_disconnected_frac"]:
        msg = (f"STOP: {len(disconnected)}/{len(scenes)} scenes ({frac:.0%}) dropped for GRAPH_DISCONNECTED, "
               f"exceeding {cfg['view_graph']['stop_if_disconnected_frac']:.0%}: {disconnected}")
        (path_of(cfg, "reports") / "STOP_graph_disconnected.md").write_text(msg + "\n")
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
