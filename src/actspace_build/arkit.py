"""Readers for raw ARKitScenes files (read-only; data_raw is never modified)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import geometry as g


def scene_dir_3dod(root: Path, vid: str) -> Path:
    return Path(root) / vid


def frames_dir(root: Path, vid: str) -> Path:
    return Path(root) / vid / f"{vid}_frames"


def frame_timestamp(fname: str) -> float:
    """'<vid>_<ts>.png' / '<vid>_<ts>.pincam' -> ts."""
    return float(os.path.splitext(fname)[0].split("_", 1)[1])


def list_frames(folder: Path, ext: str) -> list[tuple[float, str]]:
    """Return [(ts, file name)] sorted by timestamp."""
    if not Path(folder).is_dir():
        return []
    out = [(frame_timestamp(f), f) for f in os.listdir(folder) if f.endswith(ext)]
    return sorted(out)


def load_traj(path: Path):
    """lowres_wide.traj -> (ts[N], M[N,4,4]). M is the matrix built directly from the file's (axis-angle, translation);
    its meaning (T_wc or T_cw) is determined empirically at GATE A, and no assumption is made here."""
    a = np.loadtxt(path, ndmin=2)
    ts = a[:, 0]
    M = np.stack([g.make_T(g.axis_angle_to_matrix(r), t) for r, t in zip(a[:, 1:4], a[:, 4:7])])
    order = np.argsort(ts, kind="stable")
    return ts[order], M[order]


def load_pincam(path: Path):
    """`.pincam`: w h fx fy cx cy -> (K, W, H)."""
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, int(w), int(h)


def load_annotation(path: Path) -> list[dict]:
    """3D box annotations. obbAligned.normalizedAxes is reshaped row-wise to 3x3; row i is the world direction of box axis i (verified
    empirically, see obb_convention.py), and axesLengths[i] is the edge length along that axis (consistent with the official box_utils.compute_box_3d)."""
    with open(path) as f:
        data = json.load(f)["data"]
    objs = []
    for i, d in enumerate(data):
        ob = d["segments"]["obbAligned"]
        objs.append({
            "src_index": i,
            "uid": d.get("uid"),
            "label": d["label"],
            "center": np.asarray(ob["centroid"], dtype=float),
            "axes": np.asarray(ob["normalizedAxes"], dtype=float).reshape(3, 3),
            "size": np.asarray(ob["axesLengths"], dtype=float),
        })
    return objs


def load_mesh_legacy(path: Path):
    import open3d as o3d
    return o3d.io.read_triangle_mesh(str(path))


def raycasting_scene(vertices: np.ndarray, triangles: np.ndarray):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(triangles.astype(np.uint32)))
    return scene


def read_depth_m(path: Path, scale: float) -> np.ndarray:
    import cv2
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) * scale


def scene_files_3dod(root: Path, vid: str) -> dict[str, Path]:
    s = scene_dir_3dod(root, vid)
    f = frames_dir(root, vid)
    return {
        "lowres_wide": f / "lowres_wide",
        "lowres_wide_intrinsics": f / "lowres_wide_intrinsics",
        "lowres_wide.traj": f / "lowres_wide.traj",
        "lowres_depth": f / "lowres_depth",
        "mesh": s / f"{vid}_3dod_mesh.ply",
        "annotation": s / f"{vid}_3dod_annotation.json",
    }


def scene_files_raw(root: Path, vid: str) -> dict[str, Path]:
    s = Path(root) / vid
    return {"vga_wide": s / "vga_wide", "vga_wide_intrinsics": s / "vga_wide_intrinsics"}
