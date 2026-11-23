"""Export of the raw material for the Section 5.7 failure cases. Reads only existing logs, the frozen data and the raw datasets; no model is
called, no finished figures are drawn,
and no annotation or visibility definition is changed.

One directory per case, <output>/<tag>_<case directory name>/:
  geometry/     reconstruction mesh (original file copied as is) + z-up mesh_zup.ply (same coordinate frame as annotations and poses, meters)
  annotations/  objects.json (all annotated objects), view_graph.json (all nodes and action edges), visibility.npz (as is) and a visibility
                table of group objects × all nodes
  images/nodes/ upright RGB of every graph node (identical to what the model sees: scene.image + encode_jpeg)
  images/route/ the image the model saw at every step (step_00_START.jpg, step_01.jpg…, one copy per repeated visit) and the list of images
                still in context at answer time
  images/evidence/ the 3 frames with the best visibility for each evidence object of the group, plus a copy with that object's OBB projected
  depth/        sensor depth (where available) and mesh-rendered depth for the route and evidence frames above
  episode/      full group record, shortest evidence route, model trajectory, answers and notes
  topdown/      orthographic top-down rendering of the mesh (no annotations overlaid) and the pixel ↔ world conversion
Sizes are estimated first: if any case exceeds --max-case-gb or the total exceeds --max-total-gb, nothing is copied and the exit code is 3.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from actspace_build import arkit, geometry as G
from actspace_build.common import path_of, read_json
from actspace_build.obb_convention import draw_box

from . import answers as A
from .agents import oracle_targets, plan_route
from .data import FrozenData, dataset_of_scene
from .models import encode_jpeg
from .sec53_behavior import ROOT, RUNS
from .sec54_evidence_state import final_context
from .trajectory import read_episodes

CASES = (("C1", "early_answer_arkit_g_000004"), ("C4", "evidence_left_context_arkit_g_000016"), ("C9", "wrong_with_evidence_scannetpp_g_000015"))
CASE_ROOT = ROOT / "_reports" / "sec57" / "cases"
OUT = ROOT / "_reports" / "sec57" / "case_assets"
TOP_K = 3                 # number of best-visibility frames per evidence object
TOPDOWN_RES = 0.005       # top-down resolution (meters / pixel)
TOPDOWN_MARGIN = 0.2      # margin around the mesh xy bounding box in the top-down view (meters)
CUT_ABOVE_CAMERA = 0.5    # cut version: keep only surfaces with z ≤ median graph-node camera height + this value (a free choice of this script)
BOX_COLOR = (0, 255, 0)   # line colour of the projected OBB (BGR)
STATE_NAME = {0: "invisible", 1: "partial", 2: "visible"}


def jdump(p: Path, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1))


def size_of(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


@lru_cache(maxsize=None)
def _final_context(run_dir: Path) -> dict:
    return final_context(run_dir)      # calls.jsonl is large; read each run only once


class Case:
    def __init__(self, data: FrozenData, tag: str, name: str, spec: dict | None = None):
        """Without spec, model and run come from the case already exported by sec57_errors (read only); otherwise from spec's model, run, group_id, focus_qid, situation."""
        self.tag, self.name = tag, name
        self.case_md = None if spec else CASE_ROOT / name / "case.md"
        self.src = spec or read_json(CASE_ROOT / name / "trajectory.json")
        self.model, self.run, self.gid = self.src["model"], self.src["run"], self.src["group_id"]
        self.g = data.groups[self.gid]
        self.sid = self.g["scene_id"]
        self.ds = dataset_of_scene(self.sid)
        self.rid = self.sid.split("_", 1)[1]
        self.scene = data.scene(self.sid)
        self.wd = self.scene.wd
        self.cfg = data.cfg
        self.image_root = data.image_roots[self.ds]
        self.max_long = data.max_long_side
        self.frames = {f["frame_id"]: f for f in read_json(self.wd / "frames_all.json")}
        self.meta = read_json(self.wd / "scene_meta.json")
        self.objects = read_json(self.wd / "objects.json")
        self.masked = data.masked_nodes(self.gid)
        self.run_dir = RUNS[self.model][0] / "active_v2" / self.run
        ep = next(e for e in read_episodes(self.run_dir / "trajectories.jsonl") if e[0]["group_id"] == self.gid)
        self.ep = ep
        self.steps = [r for r in ep if r["type"] == "step"]
        self.route = [ep[0]["start_node"]] + [r["to"] for r in self.steps]
        self.ctx = _final_context(self.run_dir)[self.gid]
        self.ev_objs = sorted({int(o) for q in self.g["questions"] for o in q["evidence_objs"]})
        v = np.load(self.wd / "visibility.npz")
        self.vis = {k: v[k] for k in ("obj_ids", "frame_ids", "state", "area_frac", "rho", "completeness", "in_frame_frac", "has_sensor_depth")}
        self.row = {int(o): i for i, o in enumerate(self.vis["obj_ids"])}
        self.col = {str(n): i for i, n in enumerate(self.vis["frame_ids"])}
        if self.ds == "arkitscenes":
            f3 = arkit.scene_files_3dod(path_of(self.cfg, "arkit_3dod_root"), self.rid)
            self.mesh_path, self.raw_anno = f3["mesh"], [f3["annotation"]]
            self.depth_dir = f3["lowres_depth"]
            self.depth_list = arkit.list_frames(self.depth_dir, ".png")
            self.depth_ts = np.array([t for t, _ in self.depth_list])
        else:
            from actspace_build.scannetpp import scene_paths
            sc = scene_paths(self.cfg, self.rid)
            self.mesh_path, self.raw_anno = Path(sc.scan_mesh_path), [Path(sc.scan_anno_json_path), Path(sc.scan_mesh_segs_path)]

    # ---- camera: K, T_wc, width and height of the image the model sees (upright, scaled to long side ≤ max_long)
    def camera(self, n: str):
        f = self.frames[n]
        W, H = f["image_size"]
        K, T = np.asarray(f["K"], float), np.asarray(f["T_wc"], float)     # K, T_wc and image_size in frames_all.json are already upright (rotated together at build time)
        w, h = W, H
        if self.max_long and max(H, W) > self.max_long:
            k = self.max_long / max(H, W)
            w, h = round(W * k), round(H * k)
        K = K.copy()
        K[0] *= w / W
        K[1] *= h / H
        return K, T, w, h

    def state(self, o: int, n: str) -> int:
        return int(self.vis["state"][self.row[o], self.col[n]])

    def best_frames(self, o: int) -> list[str]:
        """Up to TOP_K frames among all graph nodes with the best visibility of the object: first by visibility level (visible > partial), then by the object's image-area fraction, then by frame id."""
        sc = [(-self.state(o, n), -float(self.vis["area_frac"][self.row[o], self.col[n]]), n) for n in self.scene.nodes if self.state(o, n) > 0]
        return [n for _, _, n in sorted(sc)[:TOP_K]]

    def sensor_depth_raw(self, n: str):
        """(raw depth uint16 millimeters, upright, at native depth resolution; source file; time offset to the frame), or None. Timestamp matching as at build time."""
        f = self.frames[n]
        if self.ds == "arkitscenes":
            j = int(np.argmin(np.abs(self.depth_ts - f["timestamp"])))
            dt = float(abs(self.depth_ts[j] - f["timestamp"]))
            if dt > self.cfg["visibility"]["sensor_depth_tol_s"]:
                return None
            p = self.depth_dir / self.depth_list[j][1]
        else:
            from actspace_build.scannetpp import sp_root
            p, dt = sp_root(self.cfg) / f["depth_rel_path"], 0.0
            if not p.exists():
                return None
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return G.rotate_image_cw(d, f["rotation_cw_quarters"]), p, dt

    def estimate(self) -> dict:
        """Size estimate before copying (bytes): mesh = original file × 2 (as is + z-up), images = upper bound from the original image sizes, depth = original files plus an uncompressed 16-bit upper bound."""
        n_route, n_ev = len(self.route), sum(len(self.best_frames(o)) for o in self.ev_objs)
        img = lambda n: (self.image_root / self.frames[n]["rel_path"]).stat().st_size     # noqa: E731
        mesh = self.mesh_path.stat().st_size
        _, _, w, h = self.camera(self.route[0])
        return {"geometry": 2 * mesh,
                "annotations": sum(p.stat().st_size for p in self.raw_anno) + 4 * size_of(self.wd / "objects.json") + 2 * size_of(self.wd / "frames_all.json")
                + 4 * size_of(self.wd / "visibility.npz"),
                "images_nodes": sum(img(n) for n in self.scene.nodes), "images_route": sum(img(n) for n in self.route),
                "images_evidence": 2 * sum(img(n) for o in self.ev_objs for n in self.best_frames(o)),
                "depth": (n_route + n_ev) * (w * h * 2 + 256 * 192 * 2), "episode": 2 * 1024 * 1024,
                "topdown": 4 * 3 * int(np.prod(self.topdown_shape()))}

    def topdown_shape(self):
        import open3d as o3d
        if not hasattr(self, "_mesh"):
            m = o3d.io.read_triangle_mesh(str(self.mesh_path))
            V = np.asarray(m.vertices)
            if self.ds == "arkitscenes":
                V = V @ np.asarray(self.meta["R_up"], float).T          # as at build time: mesh vertices into the z-up scene frame
            self._mesh = (V, np.asarray(m.triangles), np.asarray(m.vertex_colors) if m.has_vertex_colors() else None)
        V = self._mesh[0]
        lo, hi = V[:, :2].min(0) - TOPDOWN_MARGIN, V[:, :2].max(0) + TOPDOWN_MARGIN
        self._td = (lo, hi)
        return int(np.ceil((hi[1] - lo[1]) / TOPDOWN_RES)), int(np.ceil((hi[0] - lo[0]) / TOPDOWN_RES))


# ---------------------------------------------------------------- individual exports
def export_geometry(c: Case, d: Path) -> dict:
    import open3d as o3d
    d.mkdir(parents=True)
    shutil.copy2(c.mesh_path, d / c.mesh_path.name)
    c.topdown_shape()
    V, F, col = c._mesh
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    if col is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_triangle_mesh(str(d / "mesh_zup.ply"), m, write_ascii=False)
    R = np.asarray(c.meta["R_up"], float) if c.ds == "arkitscenes" else np.eye(3)
    info = {"original": c.mesh_path.name, "original_source": str(c.mesh_path), "zup": "mesh_zup.ply", "units": "meters", "up_axis": "+z",
            "R_up": R.tolist(), "zup_equals_original_vertices": bool(np.allclose(R, np.eye(3))),
            "note": "vertices of mesh_zup.ply = original vertices @ R_up.T; same world frame as the poses in objects.json and view_graph.json (meters, z up)",
            "n_vertices": int(len(V)), "n_triangles": int(len(F)), "has_vertex_colors": col is not None,
            "bbox_min": V.min(0).round(4).tolist(), "bbox_max": V.max(0).round(4).tolist()}
    jdump(d / "geometry_info.json", info)
    return info


def export_annotations(c: Case, d: Path):
    d.mkdir(parents=True)
    cluster = {int(o) for o in c.g.get("cluster_obj_ids", [])}
    used = {}
    for q in c.g["questions"]:
        for o in q["evidence_objs"]:
            used.setdefault(int(o), []).append(q["qid"])
    mobj = (c.g.get("view_graph_mask") or {}).get("masked_obj")
    objs = []
    for o in c.objects:
        i = int(o["obj_id"])
        objs.append({"obj_id": i, "source_label": o.get("source_label"), "category": o.get("category"),
                     **{k: o[k] for k in ("category_fine", "category_merged", "kind", "source_uid", "source_object_id", "blacklisted") if k in o},
                     "ref_name": o.get("ref_name"), "ref_kind": o.get("ref_kind"), "referable": bool(o.get("referable")), "drop_reasons": o.get("drop_reasons"),
                     "obb": {"center": o["obb"]["center"], "size": o["obb"]["size"], "axes": o["obb"]["axes"], "yaw_deg": o["obb"]["yaw_deg"]},
                     "num_points": o.get("num_points"), "in_group_cluster": i in cluster, "evidence_of_questions": used.get(i, []),
                     "in_group": i in cluster or i in used, "is_masked_obj_of_group": mobj is not None and i == int(mobj)})
    jdump(d / "objects.json", {"scene_id": c.sid, "group_id": c.gid, "units": "meters", "world": "z up; same frame as T_wc in view_graph.json",
                               "fields": {"source_label": "original dataset label", "category": "unified category name from the build (the vocabulary used in questions)", "ref_name": "name used for the object in questions (may be empty if not referable)",
                                          "obb.axes": "3×3, each row is the unit vector of one edge direction (world frame); obb.size is the edge length along these three directions (meters); obb.yaw_deg is the heading of the first axis in the xy plane",
                                          "in_group_cluster": "belongs to the group's cluster_obj_ids", "evidence_of_questions": "question ids that use it as an evidence object", "in_group": "either of the two above"},
                               "objects": objs})
    shutil.copy2(c.wd / "objects.json", d / "objects_source.json")
    for p in c.raw_anno:
        shutil.copy2(p, d / f"raw_{p.name}")
    # view graph
    nodes = []
    for n in c.scene.nodes:
        f = c.frames[n]
        K, T, w, h = c.camera(n)
        nodes.append({"frame_id": n, "image": f"images/nodes/{n}.jpg", "source_frame": f["source_frame"], "source_rel_path": f["rel_path"], "timestamp": f["timestamp"],
                      "position": f["position"], "yaw_deg": f["yaw_deg"], "T_wc": f["T_wc"], "K": f["K"], "image_size": f["image_size"],
                      "rotation_cw_quarters": f["rotation_cw_quarters"], "upright_rotation_deg_cw": 90 * int(f["rotation_cw_quarters"]),
                      "upright": {"K": K.tolist(), "T_wc": T.tolist(), "image_size": [w, h]},
                      "masked_for_group": n in c.masked, "has_sensor_depth": bool(c.vis["has_sensor_depth"][c.col[n]])})
    edges = [{"from": n, "action": a, "to": t, "masked_for_group": n in c.masked or t in c.masked}
             for n in c.scene.nodes for a, t in c.scene.edges[n].items() if t is not None]
    jdump(d / "view_graph.json", {
        "scene_id": c.sid, "group_id": c.gid, "units": "meters", "pose_convention": "T_wc is camera-to-world (OpenCV camera frame: x right, y down, z forward)",
        "fields": {"position": "camera centre (world frame)", "yaw_deg": "camera horizontal heading, heading vector = (cos yaw, sin yaw)", "K / T_wc / image_size": "as in frames_all.json: intrinsics, pose and [width, height] of the upright, unscaled image (rotated together with the image at build time)",
                   "rotation_cw_quarters": "number of 90° clockwise rotations that make the raw image on disk upright (already included in K and T_wc)", "upright": "K, T_wc and [width, height] of the image the model sees (upright and scaled to long side ≤ 640); differs from the above only by scaling",
                   "masked_for_group": "this group masks the node (or an endpoint of this edge) out of the view graph; the model cannot reach it", "edges": "directed action edges: executing action at from arrives at to"},
        "actions": list(next(iter(c.scene.edges.values()))), "n_nodes": len(nodes), "n_edges": len(edges), "masked_nodes": sorted(c.masked), "nodes": nodes, "edges": edges})
    shutil.copy2(c.wd / "view_graph.json", d / "view_graph_source.json")
    shutil.copy2(c.wd / "scene_meta.json", d / "scene_meta.json")
    # visibility
    shutil.copy2(c.wd / "visibility.npz", d / "visibility.npz")
    gobjs = sorted(set(cluster) | set(used))
    name = {int(o["obj_id"]): (o.get("ref_name") or o.get("category")) for o in c.objects}
    with open(d / "visibility_group_objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "name"] + list(c.scene.nodes))
        for o in gobjs:
            w.writerow([o, name[o]] + [c.state(o, n) for n in c.scene.nodes])
    with open(d / "visibility_group_objects_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "name", "frame_id", "state", "state_name", "area_frac", "rho", "completeness", "in_frame_frac", "node_masked_for_group", "on_route", "in_context_at_answer"])
        for o in gobjs:
            for n in c.scene.nodes:
                i, j = c.row[o], c.col[n]
                s = c.state(o, n)
                w.writerow([o, name[o], n, s, STATE_NAME[s]] + [round(float(c.vis[k][i, j]), 4) for k in ("area_frac", "rho", "completeness", "in_frame_frac")]
                           + [int(n in c.masked), int(n in c.route), int(n in c.ctx)])
    return gobjs


def write_image(c: Case, n: str, p: Path):
    img = c.scene.image(n)
    p.write_bytes(encode_jpeg(img))
    return img


def export_images(c: Case, d: Path) -> dict:
    for s in ("nodes", "route", "evidence"):
        (d / s).mkdir(parents=True)
    name = {int(o["obj_id"]): o for o in c.objects}
    idx = []
    for n in c.scene.nodes:
        img = write_image(c, n, d / "nodes" / f"{n}.jpg")
        idx.append([n, f"images/nodes/{n}.jpg", img.shape[1], img.shape[0], c.frames[n]["source_frame"], c.frames[n]["rel_path"], c.frames[n]["rotation_cw_quarters"],
                    int(n in c.masked), int(n in c.route), int(n in c.ctx)])
    with open(d / "nodes" / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "file", "width", "height", "source_frame", "source_rel_path", "rotation_cw_quarters", "masked_for_group", "on_route", "in_context_at_answer"])
        w.writerows(idx)
    # route: the image seen at every step
    route = [{"step": 0, "file": "images/route/step_00_START.jpg", "frame_id": c.route[0], "action": None, "moved": None}]
    write_image(c, c.route[0], d / "route" / "step_00_START.jpg")
    for r in c.steps:
        fn = f"step_{r['t']:02d}.jpg"
        write_image(c, r["to"], d / "route" / fn)
        route.append({"step": r["t"], "file": f"images/route/{fn}", "frame_id": r["to"], "action": r["action"], "moved": bool(r["moved"])})
    # images still in context at answer time: frame_ids actually sent as images in the last call; mapped to route step numbers (START is step 0, the rest aligned from the end backwards)
    steps_of = {}
    for x in route:
        steps_of.setdefault(x["frame_id"], []).append(x["step"])
    moved = [x["step"] for x in route if x["moved"]]      # failed actions produce no new image; the context only holds images of successful moves
    tail, aligned = [], True
    for n in reversed(c.ctx[1:] if c.ctx and c.ctx[0] == c.route[0] else c.ctx):
        if moved and route[moved[-1]]["frame_id"] == n:
            tail.append(moved.pop())
        else:
            aligned = False
            break
    ctx = []
    body = c.ctx[1:] if c.ctx and c.ctx[0] == c.route[0] else c.ctx
    if c.ctx and c.ctx[0] == c.route[0]:
        ctx.append({"order": 0, "frame_id": c.ctx[0], "step": 0, "file": "images/route/step_00_START.jpg", "all_steps_at_this_frame": steps_of[c.ctx[0]]})
    for i, n in enumerate(body):
        st = tail[::-1][i] if aligned else None
        ctx.append({"order": len(ctx), "frame_id": n, "step": st, "file": route[st]["file"] if st is not None else None, "all_steps_at_this_frame": steps_of.get(n, [])})
    jdump(d / "route" / "route_index.json", {"n_steps": len(c.steps), "note": "step 0 is the start; then one image per step. After a failed action (moved=false) the agent stays in place and receives no new image; the image stored here is that of the node it was at (same as the previous step)", "steps": route})
    jdump(d / "route" / "context_at_answer.json", {
        "source": "images actually sent in the episode's last call in calls.jsonl (sec54_evidence_state.final_context)", "history": c.ep[0].get("history"),
        "n_images": len(c.ctx), "step_alignment_ok": aligned,
        "note": "order is the image order in the request; step is the route step this image corresponds to (context = start + the most recent successful moves, aligned image by image from the end backwards; failed actions produce no new image)", "images": ctx})
    # evidence frames
    ev = []
    for o in c.ev_objs:
        ob = name[o]
        nm = (ob.get("ref_name") or ob.get("category") or "obj").replace(" ", "_").replace("/", "_")
        for rank, n in enumerate(c.best_frames(o), 1):
            base = f"obj{o:03d}_{nm}_rank{rank}_{n}"
            img = write_image(c, n, d / "evidence" / f"{base}.jpg")
            K, T, w, h = c.camera(n)
            assert (w, h) == (img.shape[1], img.shape[0]), (n, w, h, img.shape)
            bgr = np.ascontiguousarray(img[:, :, ::-1])
            draw_box(bgr, np.asarray(ob["obb"]["center"], float), np.asarray(ob["obb"]["axes"], float), np.asarray(ob["obb"]["size"], float), K, T, BOX_COLOR)
            (d / "evidence" / f"{base}_obb.jpg").write_bytes(encode_jpeg(np.ascontiguousarray(bgr[:, :, ::-1])))
            i, j = c.row[o], c.col[n]
            ev.append({"obj_id": o, "name": ob.get("ref_name") or ob.get("category"), "rank": rank, "frame_id": n, "file": f"images/evidence/{base}.jpg",
                       "file_obb": f"images/evidence/{base}_obb.jpg", "state": c.state(o, n), "state_name": STATE_NAME[c.state(o, n)],
                       "area_frac": round(float(c.vis["area_frac"][i, j]), 4), "completeness": round(float(c.vis["completeness"][i, j]), 4),
                       "node_masked_for_group": n in c.masked, "on_route": n in c.route, "route_steps": steps_of.get(n, []), "in_context_at_answer": n in c.ctx})
    none = [o for o in c.ev_objs if not c.best_frames(o)]
    jdump(d / "evidence" / "evidence_index.json", {
        "rule": f"among all graph nodes, sorted by visibility.npz state (visible > partial), then area_frac, then frame id; first {TOP_K} frames; frames with state = 0 are not used",
        "obb_drawing": "actspace_build.obb_convention.draw_box: the 12 OBB edges projected onto the image the model sees, green, line width 2; only parts inside the image and in front of the camera are drawn",
        "objects_without_any_visible_frame": none, "frames": ev})
    return {"route": route, "evidence": ev, "aligned": aligned}


def export_depth(c: Case, d: Path, im: dict) -> dict:
    (d / "sensor").mkdir(parents=True)
    (d / "mesh_render").mkdir()
    V, F, _ = c._mesh
    rs = arkit.raycasting_scene(V, F)
    need = {}
    for x in im["route"]:
        need.setdefault(x["frame_id"], []).append(x["file"])
    for x in im["evidence"]:
        need.setdefault(x["frame_id"], []).append(x["file"])
    rows, miss = [], []
    for n in sorted(need):
        K, T, w, h = c.camera(n)
        z = G.render_depth(rs, K, T, w, h)
        mm = np.where(np.isfinite(z) & (z > 0) & (z < 65.0), np.round(z * 1000.0), 0).astype(np.uint16)
        cv2.imwrite(str(d / "mesh_render" / f"{n}.png"), mm)
        sd = c.sensor_depth_raw(n)
        row = {"frame_id": n, "used_by": need[n], "mesh_render": f"depth/mesh_render/{n}.png", "mesh_render_size": [w, h], "mesh_render_K": K.tolist(), "sensor": None}
        if sd is None:
            miss.append(n)
        else:
            raw, p, dt = sd
            cv2.imwrite(str(d / "sensor" / f"{n}.png"), raw)
            hh, ww = raw.shape[:2]
            Ks = K.copy()
            Ks[0] *= ww / w
            Ks[1] *= hh / h
            row.update({"sensor": f"depth/sensor/{n}.png", "sensor_size": [ww, hh], "sensor_K": Ks.tolist(), "sensor_source": str(p), "sensor_dt_s": round(dt, 4)})
        rows.append(row)
    jdump(d / "depth_index.json", {
        "format": "16-bit single-channel PNG, millimeters, 0 = invalid (sensor) or no mesh hit (render); all upright, same orientation as the images in images/",
        "mesh_render": "z depth (along the optical axis) rendered from mesh_zup with the upright K and T_wc in view_graph.json, at the resolution of the image the model sees",
        "sensor": ("raw ARKitScenes lowres_depth values (256×192), matched by timestamp: available only if the time offset to the frame ≤ visibility.sensor_depth_tol_s (build-time definition); only made upright, not scaled, not filtered by valid range"
                   if c.ds == "arkitscenes" else "raw ScanNet++ iphone/depth values of the same frame index (256×192), not scaled, not filtered by valid range"),
        "sensor_K": "upright K scaled to the depth-map size (at build time depth is upsampled nearest-neighbour to the RGB resolution, which is equivalent to this intrinsic matrix)",
        "frames_without_sensor_depth": miss, "frames": rows})
    return {"n": len(rows), "missing_sensor": miss}


def export_episode(c: Case, d: Path):
    d.mkdir(parents=True)
    g, s, end = c.g, c.ep[0], c.ep[-1]
    jdump(d / "group.json", g)
    targets = oracle_targets(g)
    acts, unreachable = plan_route(c.scene, c.masked, g["init_frame"], targets)
    jdump(d / "shortest_evidence_route.json", {
        "source": "agents.plan_route (group-level shortest evidence route used by oracle_route): from the start, visit all oracle_nodes, link_views and path_views of all questions; masked nodes are not traversable",
        "start": g["init_frame"], "targets": targets, "unreachable_targets": unreachable, "n_steps": len(acts),
        "steps": [{"t": i, "action": a, "node": n, "x": round(float(c.scene.pose[n][0][0]), 4), "y": round(float(c.scene.pose[n][0][1]), 4),
                   "yaw_deg": round(float(c.scene.pose[n][1]), 2), "is_target": n in targets} for i, (a, n) in enumerate(acts, 1)]})
    jdump(d / "trajectory.json", trajectory_record(c))
    jdump(d / "answers.json", answers_record(c))
    (d / "notes.txt").write_text((end.get("notes") or "") + "\n")
    if c.case_md is not None:
        shutil.copy2(c.case_md, d / "case_errors.md")


def trajectory_record(c: Case, images: bool = True) -> dict:
    s = c.ep[0]
    pose = lambda n: {"x": round(float(c.scene.pose[n][0][0]), 4), "y": round(float(c.scene.pose[n][0][1]), 4), "z": round(float(c.scene.pose[n][0][2]), 4),   # noqa: E731
                      "yaw_deg": round(float(c.scene.pose[n][1]), 2)}
    img = (lambda f: {"image": f}) if images else (lambda f: {})
    return {
        "model": c.model, "run": c.run, "run_dir": str(c.run_dir), "group_id": c.gid, "scene_id": c.sid, "setting": "active_v2",
        "fields": {"moved": "whether this step actually moved", "feedback": "text returned to the model by the environment", "odometry": "odometry feedback of this step (as is)", "pose": "cumulative pose given to the model by the environment (relative to the start, as is)",
                   "world": "ground-truth pose of the reached node (world frame, not visible to the model)", "image": "image seen by the model at this step"},
        "episode_start": s, "start": {"node": s["start_node"], "world": pose(s["start_node"]), **img("images/route/step_00_START.jpg")},
        "steps": [{**r, "world": pose(r["to"]), **img(f"images/route/step_{r['t']:02d}.jpg")} for r in c.steps],
        "other_records": [r for r in c.ep[1:-1] if r["type"] != "step"], "context_nodes_at_answer": c.ctx}


def answers_record(c: Case) -> dict:
    g, s, end = c.g, c.ep[0], c.ep[-1]
    pub = {v: k for k, v in s["question_ids"].items()}
    qs = []
    for q in g["questions"]:
        k = pub[q["qid"]]
        qs.append({"id": k, "qid": q["qid"], "template": q["template"], "text": q["text"], "options": q.get("options"), "evidence_label": q["evidence_label"],
                   "gold": A.gold(q), "answer_value": q.get("answer_value"), "model_answer": end["answers"].get(k), "model_raw_answer": (end.get("raw_answer") or {}).get(k)
                   if isinstance(end.get("raw_answer"), dict) else None, "score": end["scores"].get(k), "evidence_objs": q["evidence_objs"],
                   "is_focus_question": q["qid"] == c.src["focus_qid"]})
    return {"model": c.model, "run": c.run, "group_id": c.gid, "steps_used": end["steps_used"], "budget": s["budget"], "forced": end.get("forced"),
            "parse_ok": end.get("parse_ok"), "raw_answer": end.get("raw_answer"), "notes": end.get("notes"), "focus_qid": c.src["focus_qid"],
            "situation": c.src["situation"], "questions": qs, "episode_end": end}


def export_topdown(c: Case, d: Path) -> dict:
    import open3d as o3d
    d.mkdir(parents=True)
    H, W = c.topdown_shape()
    lo, hi = c._td
    V, F, col = c._mesh
    rs = arkit.raycasting_scene(V, F)
    cam_z = float(np.median([c.scene.pose[n][0][2] for n in c.scene.nodes]))
    z_cut = cam_z + CUT_ABOVE_CAMERA
    z_top = float(V[:, 2].max()) + 1.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x = lo[0] + (u + 0.5) * TOPDOWN_RES
    y = hi[1] - (v + 0.5) * TOPDOWN_RES
    out = {}
    for name, z0 in (("full", z_top), ("cut", z_cut)):
        rays = np.stack([x, y, np.full_like(x, z0), np.zeros_like(x), np.zeros_like(x), -np.ones_like(x)], -1).reshape(-1, 6).astype(np.float32)
        r = rs.cast_rays(o3d.core.Tensor(rays))
        t, pid, uv = r["t_hit"].numpy(), r["primitive_ids"].numpy(), r["primitive_uvs"].numpy()
        hit = np.isfinite(t)
        img = np.full((H * W, 3), 255, np.uint8)
        if col is not None:
            tri = F[pid[hit]]
            b1, b2 = uv[hit, 0:1], uv[hit, 1:2]
            rgb = (1 - b1 - b2) * col[tri[:, 0]] + b1 * col[tri[:, 1]] + b2 * col[tri[:, 2]]
            img[hit] = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        else:
            img[hit] = 128
        img = img.reshape(H, W, 3)
        cv2.imwrite(str(d / f"topdown_{name}.png"), img[:, :, ::-1])
        hmm = np.zeros(H * W, np.uint16)
        zz = z0 - t[hit]
        hmm[hit] = np.clip(np.round((zz - float(V[:, 2].min())) * 1000.0) + 1, 1, 65535).astype(np.uint16)
        cv2.imwrite(str(d / f"topdown_{name}_height.png"), hmm.reshape(H, W))
        # version with a scale bar: a white strip below the image holds a 1 m scale bar, without covering the rendering
        strip = np.full((70, W, 3), 255, np.uint8)
        L = int(round(1.0 / TOPDOWN_RES))
        cv2.rectangle(strip, (20, 20), (20 + L, 32), (0, 0, 0), -1)
        cv2.putText(strip, "1 m", (30 + L, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(d / f"topdown_{name}_scalebar.png"), np.vstack([img, strip])[:, :, ::-1])
        out[name] = {"file": f"topdown_{name}.png", "file_scalebar": f"topdown_{name}_scalebar.png", "file_height": f"topdown_{name}_height.png",
                     "ray_start_z": round(z0, 4), "hit_fraction": round(float(hit.mean()), 4)}
    meta = {"projection": "orthographic top-down: for each pixel a ray is cast from above along -z and takes the vertex colour (barycentric interpolation) of the first mesh surface hit; white if nothing is hit. No annotations overlaid",
            "meters_per_pixel": TOPDOWN_RES, "image_size": [W, H], "x_min": round(float(lo[0]), 6), "x_max": round(float(lo[0] + W * TOPDOWN_RES), 6),
            "y_max": round(float(hi[1]), 6), "y_min": round(float(hi[1] - H * TOPDOWN_RES), 6),
            "world_to_pixel": "u = (x - x_min) / meters_per_pixel, v = (y_max - y) / meters_per_pixel (u to the right, v downwards; the centre of pixel (i, j) is at u = i + 0.5, v = j + 0.5)",
            "pixel_to_world": "x = x_min + u * meters_per_pixel, y = y_max - v * meters_per_pixel",
            "orientation": "image up is world +y, image right is world +x; the yaw_deg heading vector (cos yaw, sin yaw) is the pixel direction (cos yaw, -sin yaw) in the image",
            "scalebar": "*_scalebar.png has an extra 70-pixel white strip below the image containing a 1 m (= 1 / meters_per_pixel pixels) scale bar; pixel coordinates of the image area are unchanged",
            "height": "*_height.png: 16-bit PNG, 0 = no hit; otherwise value = (surface z - z_min) × 1000 + 1 (millimeters)", "z_min": round(float(V[:, 2].min()), 4),
            "variants": out, "cut_rule": f"rays of the cut version start at z = median graph-node camera height ({cam_z:.3f}) + {CUT_ABOVE_CAMERA} m, so higher surfaces (ceiling, tops of tall cabinets, etc.) are removed; the full version starts above the highest mesh point",
            "camera_z_median": round(cam_z, 4), "z_cut": round(z_cut, 4)}
    jdump(d / "topdown_meta.json", meta)
    return meta


def write_readme(c: Case, d: Path, geo: dict, gobjs: list, im: dict, dep: dict, td: dict):
    end = c.ep[-1]
    name = {int(o["obj_id"]): (o.get("ref_name") or o.get("category")) for o in c.objects}
    _, _, w, h = c.camera(c.route[0])
    L = [f"# {c.tag}: {c.name}", "",
         f"- Model {c.model}, run `{c.run}` (`{c.run_dir}`), setting active_v2; group `{c.gid}`, scene `{c.sid}` ({c.ds}).",
         f"- The model used {end['steps_used']} / {c.ep[0]['budget']} steps; the view graph has {len(c.scene.nodes)} nodes, {len(c.masked)} of them masked for this group; {len(c.ctx)} images in context at answer time.",
         f"- Group objects: {', '.join(f'{name[o]} (obj {o})' for o in gobjs)}; evidence objects: {', '.join(f'{name[o]} (obj {o})' for o in c.ev_objs)}.",
         "- All files were exported by `actspace_eval.sec57_case_assets` from existing logs, the frozen data and the raw datasets; no annotation or visibility definition was changed and no model was called.", "",
         "## Coordinate conventions", "",
         "- World frame: meters, z up; the mesh (`geometry/mesh_zup.ply`), object boxes and camera poses are all in this frame.",
         f"- The original mesh `geometry/{geo['original']}` is the dataset file; vertices of `mesh_zup.ply` = original vertices @ R_up.T"
         + (" (R_up is the identity for this scene, so the vertices are identical)." if geo["zup_equals_original_vertices"] else "."),
         "- Camera pose `T_wc`: 4×4, camera-to-world, OpenCV camera frame (x right, y down, z forward). `position` is the camera centre; `yaw_deg` is the horizontal heading, heading vector = (cos yaw, sin yaw).",
         f"- Images: the raw image on disk rotated {90 * int(c.frames[c.route[0]]['rotation_cw_quarters'])}° clockwise to upright is {c.frames[c.route[0]]['image_size'][0]}×{c.frames[c.route[0]]['image_size'][1]}, "
         f"then scaled to long side ≤ 640; the model sees {w}×{h}. `K` and `T_wc` in `view_graph.json` belong to the upright, unscaled image (rotated with the image at build time); `K` and `T_wc` under `upright` belong to the image the model sees (differ only by scaling).",
         "- OBB: `center`, `size` (three edge lengths), `axes` (3×3, one edge direction per row); 8 corners = center + (±size/2) @ axes.",
         "- Visibility levels: 0 invisible, 1 partial, 2 visible (`state` in `visibility.npz`; rows are `obj_ids`, columns are `frame_ids`).",
         "- Top-down pixel ↔ world conversion: see `topdown/topdown_meta.json`.", "",
         "## Frame ids and file names", "",
         "- Frame id `f_XXXXXX` is the view-graph node id. `images/nodes/<frame id>.jpg` is the node's image; `images/nodes/index.csv` lists frame id, original file, whether masked, whether on the route, whether in context at answer time.",
         "- `images/route/step_00_START.jpg` is the start, `step_NN.jpg` is the image seen after step NN; step ↔ frame id in `images/route/route_index.json`. After a failed action the agent stays in place and the image equals the previous step's.",
         "- `images/route/context_at_answer.json`: images still in context at answer time (frame ids and corresponding step numbers).",
         "- `images/evidence/obj<object id>_<name>_rank<rank>_<frame id>.jpg`; the same name with `_obb` has the object's OBB projected; list in `evidence_index.json`.",
         "- `depth/sensor/<frame id>.png`, `depth/mesh_render/<frame id>.png`; list and intrinsics in `depth/depth_index.json`.", "",
         "## Files", "",
         "| Path | Content |", "|---|---|",
         f"| `geometry/{geo['original']}` | reconstruction mesh of the dataset, copied as is |",
         "| `geometry/mesh_zup.ply` | z-up mesh (binary PLY with vertex colours) |",
         "| `geometry/geometry_info.json` | coordinate frame, units, R_up, bounding box, vertex and face counts |",
         "| `annotations/objects.json` | all annotated objects: original label, unified category, OBB, referability, membership in this group |",
         "| `annotations/objects_source.json` | build output objects.json, as is |",
         "| `annotations/raw_*` | original annotation files of the dataset, as is |",
         "| `annotations/view_graph.json` | all nodes (frame id, position, heading, T_wc, K, upright rotation) and all action edges |",
         "| `annotations/view_graph_source.json`, `scene_meta.json` | build outputs, as is |",
         "| `annotations/visibility.npz` | scene visibility matrix, as is |",
         "| `annotations/visibility_group_objects.csv` | visibility levels of group objects × all nodes (matrix) |",
         "| `annotations/visibility_group_objects_long.csv` | the same as a long table, with area fraction, completeness, etc. |",
         f"| `images/nodes/` | upright images of all {len(c.scene.nodes)} nodes |",
         f"| `images/route/` | images of the start + {len(c.steps)} steps, step list, list of the context at answer time |",
         f"| `images/evidence/` | up to {TOP_K} frames for each of the {len(c.ev_objs)} evidence objects, {len(im['evidence'])} frames in total, each with an OBB version |",
         f"| `depth/sensor/` | sensor depth, available for {dep['n'] - len(dep['missing_sensor'])} / {dep['n']} frames |",
         f"| `depth/mesh_render/` | mesh-rendered depth, {dep['n']} frames |",
         "| `episode/group.json` | full group record (questions, options, gold answers, evidence objects, oracle nodes) |",
         "| `episode/shortest_evidence_route.json` | shortest evidence route |",
         "| `episode/trajectory.json` | full model trajectory (action per step, success, odometry feedback, cumulative pose, ground-truth pose) |",
         "| `episode/answers.json`, `notes.txt` | model answers, scores and verbatim notes |",
         *(["| `episode/case_errors.md` | case description written by sec57_errors, as is |"] if c.case_md is not None else []),
         "| `topdown/topdown_full*.png` | top-down rendering (looking down from above the highest mesh point) |",
         f"| `topdown/topdown_cut*.png` | top-down rendering with surfaces at z > {td['z_cut']} m removed |",
         "| `topdown/topdown_meta.json` | resolution, extent, pixel ↔ world conversion |", "",
         "## Missing items", ""]
    miss = []
    if dep["missing_sensor"]:
        miss.append(f"- Sensor depth missing for {len(dep['missing_sensor'])} frames (no lowres_depth within the time tolerance): {', '.join(dep['missing_sensor'])}.")
    none = [o for o in c.ev_objs if not c.best_frames(o)]
    if none:
        miss.append(f"- Evidence objects {', '.join(f'{name[o]} (obj {o})' for o in none)} are not partial or better at any node; no evidence frames.")
    short = [o for o in c.ev_objs if 0 < len(c.best_frames(o)) < TOP_K]
    if short:
        miss.append(f"- Evidence objects {', '.join(f'{name[o]} (obj {o})' for o in short)} are visible at fewer than {TOP_K} nodes; fewer than {TOP_K} evidence frames.")
    if not im["aligned"]:
        miss.append("- The context images could not be aligned one by one to step numbers (step is empty in `context_at_answer.json`); only frame ids and all steps at which each frame occurred are given.")
    L += miss or ["- None."]
    (d / "README.md").write_text("\n".join(L) + "\n")


def dir_listing(d: Path) -> dict:
    files = sorted((str(p.relative_to(d)), p.stat().st_size) for p in d.rglob("*") if p.is_file())
    sub = {}
    for f, s in files:
        k = "/".join(f.split("/")[:2]) if f.startswith("images/") else f.split("/")[0]
        sub.setdefault(k, [0, 0])
        sub[k][0] += 1
        sub[k][1] += s
    return {"total_bytes": sum(s for _, s in files), "n_files": len(files), "by_dir": sub, "files": files}


def export_all(c: Case, d: Path) -> dict:
    """All material of one case (d must be an existing empty directory)."""
    geo = export_geometry(c, d / "geometry")
    gobjs = export_annotations(c, d / "annotations")
    im = export_images(c, d / "images")
    dep = export_depth(c, d / "depth", im)
    export_episode(c, d / "episode")
    td = export_topdown(c, d / "topdown")
    write_readme(c, d, geo, gobjs, im, dep, td)
    ls = dir_listing(d)
    print(f"== {c.tag} exported: {ls['n_files']} files, {ls['total_bytes'] / (1 << 20):.1f} MB; sensor depth missing {len(dep['missing_sensor'])} / {dep['n']} frames; "
          f"context aligned {im['aligned']}", flush=True)
    return {"dir": str(d), "name": c.name, "model": c.model, "run": c.run, "group_id": c.gid, "scene_id": c.sid, "dataset": c.ds,
            "n_nodes": len(c.scene.nodes), "n_masked": len(c.masked), "n_steps": len(c.steps), "n_context": len(c.ctx), "context_aligned": im["aligned"],
            "evidence_objs": c.ev_objs, "n_evidence_frames": len(im["evidence"]), "evidence_frames_per_obj": {str(o): len(c.best_frames(o)) for o in c.ev_objs},
            "depth_frames": dep["n"], "missing_sensor_depth": dep["missing_sensor"], "geometry": geo, "topdown": td, **ls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-case-gb", type=float, default=3.0)
    ap.add_argument("--max-total-gb", type=float, default=8.0)
    ap.add_argument("--estimate-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    data = FrozenData()
    cases = [Case(data, t, n) for t, n in CASES]
    est = {c.tag: c.estimate() for c in cases}
    GB = 1 << 30
    for c in cases:
        e = est[c.tag]
        print(f"== size estimate {c.tag} {c.name}: total {sum(e.values()) / GB:.3f} GB; " + ", ".join(f"{k} {v / (1 << 20):.1f} MB" for k, v in e.items()), flush=True)
    tot = sum(sum(e.values()) for e in est.values())
    print(f"== size estimate, all three cases: {tot / GB:.3f} GB (limits: {a.max_case_gb} GB per case, {a.max_total_gb} GB in total)", flush=True)
    if any(sum(e.values()) > a.max_case_gb * GB for e in est.values()) or tot > a.max_total_gb * GB:
        print("over the limit; nothing copied")
        sys.exit(3)
    if a.estimate_only:
        return
    summary = {"estimate_bytes": est, "cases": {}}
    for c in cases:
        d = out / f"{c.tag}_{c.name}"
        if d.exists():
            print(f"{d} already exists; refusing to overwrite")
            sys.exit(2)
        d.mkdir(parents=True)
        summary["cases"][c.tag] = export_all(c, d)
    jdump(out / "case_assets_summary.json", summary)
    print("== all done, total %.1f MB" % (sum(v["total_bytes"] for v in summary["cases"].values()) / (1 << 20)))


if __name__ == "__main__":
    main()
