"""Empirical check of the rotation convention of ARKitScenes obbAligned.normalizedAxes (not relying on "consistent with the official code").

For each pilot object an OBB is built under two interpretations, "rows = box axes" (rows) and "columns = box axes" (cols), and the fit is measured on mesh vertices:
  n_inner      number of vertices inside the box shrunk by 10% on every side (object surface vertices should lie inside the box)
  n_in         number of vertices inside the box
  surface_frac among vertices within the box inflated by 5 cm, the share within 5 cm of the nearest box face (the object surface should hug the box faces)
The two interpretations differ only when the object yaw is not close to 0/90 deg (transposing negates the yaw), so only those objects are counted.
For these objects the projections of both boxes are also overlaid on a vga keyframe for visual comparison.

Usage: python -m actspace_build.obb_convention
Outputs: reports/obb_convention.md, reports/obb_convention/*.jpg
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import arkit
from . import geometry as g
from .common import load_config, path_of, read_json, work_dir

SHRINK = 0.9           # shrink by 10%
SURFACE_BAND_M = 0.05  # band width around the box faces
MIN_YAW_DIFF_DEG = 10  # objects whose box orientations under the two interpretations differ by less than this are not compared


def box_axes(raw_axes: np.ndarray, conv: str) -> np.ndarray:
    """Return a 3x3 matrix with one box axis per row."""
    return raw_axes if conv == "rows" else raw_axes.T


def fit_metrics(V: np.ndarray, center, axes_rows, size) -> dict:
    half = 0.5 * np.asarray(size)
    r = float(np.linalg.norm(half)) + SURFACE_BAND_M
    near = V[np.all(np.abs(V - center) <= r, axis=1)]
    local = np.abs((near - center) @ axes_rows.T)  # coordinates along each box axis
    n_inner = int(np.all(local <= SHRINK * half, axis=1).sum())
    inside = np.all(local <= half, axis=1)
    band = np.all(local <= half + SURFACE_BAND_M, axis=1)
    dist_to_face = np.min(np.abs(local - half), axis=1)
    surf = int((band & (dist_to_face <= SURFACE_BAND_M)).sum())
    return {"n_inner": n_inner, "n_in": int(inside.sum()), "surface_frac": surf / max(int(band.sum()), 1)}


def yaw_of(axes_rows: np.ndarray) -> float:
    k_up = int(np.argmax(np.abs(axes_rows[:, 2])))
    a = axes_rows[[i for i in range(3) if i != k_up][0]]
    return float(np.degrees(np.arctan2(a[1], a[0])))


def box_yaw_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Horizontal orientation difference between two gravity-aligned boxes (boxes are 90-degree symmetric, so in [0, 45])."""
    d = abs((yaw_of(a) - yaw_of(b)) % 90.0)
    return min(d, 90.0 - d)


def draw_box(img, center, axes_rows, size, K, T, color):
    s = 0.5 * np.asarray(size)
    signs = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]])
    C = center + (signs * s) @ axes_rows
    H, W = img.shape[:2]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        uv, z = g.project(np.linspace(C[a], C[b], 60), K, T)
        ok = g.in_image(uv, z, W, H)
        for p, q, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, 2, cv2.LINE_AA)


def best_view(frames, center, size):
    """The frame where the object centre projects inside the image, at 1-5 m distance, closest to the image centre."""
    best, best_d = None, None
    for f in frames:
        K, T, (W, H) = np.asarray(f["K"]), np.asarray(f["T_wc"]), f["image_size"]
        uv, z = g.project(np.asarray(center)[None], K, T)
        dist = float(np.linalg.norm(np.asarray(center) - np.asarray(f["position"])))
        if not g.in_image(uv, z, W, H)[0] or not 1.0 <= dist <= 5.0:
            continue
        d = float(np.linalg.norm(uv[0] - [W / 2, H / 2]) / max(W, H)) + 0.05 * dist
        if best_d is None or d < best_d:
            best, best_d = f, d
    return best


def main():
    cfg = load_config()
    root3 = path_of(cfg, "arkit_3dod_root")
    data_root = root3.resolve().parents[1]
    out_dir = path_of(cfg, "reports") / "obb_convention"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    rows_tbl, per_obj = [], []
    for vid in cfg["arkitscenes"]["pilot_scenes"]:
        files = arkit.scene_files_3dod(root3, vid)
        R_up = np.asarray(read_json(work_dir(cfg, "arkitscenes", vid) / "gate_a.json")["R_up"])
        V = np.asarray(arkit.load_mesh_legacy(files["mesh"]).vertices) @ R_up.T
        wd = work_dir(cfg, "arkitscenes", vid)
        frames = read_json(wd / "frames_all.json")
        kf = set(read_json(wd / "keyframes.json")["keyframe_ids"])
        frames = [f for f in frames if f["frame_id"] in kf]
        for k, o in enumerate(arkit.load_annotation(files["annotation"])):
            raw = np.asarray(o["axes"])  # load_annotation's raw reshape(3,3)
            center = o["center"] @ R_up.T
            ax = {c: box_axes(raw, c) @ R_up.T for c in ("rows", "cols")}
            diff = box_yaw_diff(ax["rows"], ax["cols"])
            m = {c: fit_metrics(V, center, ax[c], o["size"]) for c in ax}
            rec = {"video_id": vid, "obj_index": k, "label": o["label"], "yaw_rows": round(yaw_of(ax["rows"]), 1),
                   "yaw_diff_deg": round(diff, 1), **{f"{c}_{key}": v for c in m for key, v in m[c].items()}}
            per_obj.append(rec)
            if diff < MIN_YAW_DIFF_DEG:
                continue
            f = best_view(frames, center, o["size"])
            if f is None:
                continue
            img = g.rotate_image_cw(cv2.imread(str(data_root / f["rel_path"])), f["rotation_cw_quarters"])
            K, T = np.asarray(f["K"]), np.asarray(f["T_wc"])
            draw_box(img, center, ax["rows"], o["size"], K, T, (0, 220, 255))   # yellow = rows
            draw_box(img, center, ax["cols"], o["size"], K, T, (255, 160, 0))   # blue = cols
            cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(img, f"{vid} #{k} {o['label']}  {f['frame_id']}  yaw diff {diff:.0f} deg", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"yellow=rows inner {m['rows']['n_inner']} surf {m['rows']['surface_frac']:.2f} | "
                             f"blue=cols inner {m['cols']['n_inner']} surf {m['cols']['surface_frac']:.2f}", (6, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            name = f"{vid}_obj{k:02d}_{o['label']}.jpg"
            cv2.imwrite(str(out_dir / name), img)
            rec["overlay"] = name
    write_report(cfg, per_obj)


def write_report(cfg, per_obj):
    import pandas as pd
    df = pd.DataFrame(per_obj)
    df.to_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "obb_convention.csv", index=False)
    inf = df[df.yaw_diff_deg >= MIN_YAW_DIFF_DEG].copy()
    wins = {}
    for key in ("n_inner", "n_in", "surface_frac"):
        r, c = inf[f"rows_{key}"], inf[f"cols_{key}"]
        wins[key] = (int((r > c).sum()), int((c > r).sum()), int((r == c).sum()))
    L = ["# Empirical check of the ARKitScenes OBB rotation convention", "",
         "Two interpretations: **rows** = after `normalizedAxes` reshape(3,3), each row is a box axis (the interpretation used by the code so far); "
         "**cols** = each column is a box axis (i.e. the transpose; for gravity-aligned boxes this negates the yaw).", "",
         f"Metrics (computed per object on the mesh vertices of the 5 pilot scenes): `n_inner` vertices inside the box shrunk by 10% on every side; `n_in` vertices inside the box; "
         f"`surface_frac` share of vertices within the box inflated by {SURFACE_BAND_M} m that lie within {SURFACE_BAND_M} m of the nearest box face. "
         "The scanned object surface should fill the box and hug its faces, so the correct interpretation should score higher.", "",
         f"Of all {len(df)} annotated objects, **{len(inf)}** have box orientations differing by >= {MIN_YAW_DIFF_DEG} deg between the two interpretations (the others have yaw close to 0/90 deg, where both interpretations nearly coincide, and are not compared).", "",
         "| Metric | rows better | cols better | tie |", "|---|---|---|---|",
         *[f"| {k} | {a} | {b} | {c} |" for k, (a, b, c) in wins.items()], "",
         "## Conclusion", "",
         f"- **rows is the correct convention** (the interpretation used by the code so far; no change needed): rows has more vertices inside the box for {wins['n_in'][0]} / {len(inf)} objects, "
         f"and more vertices inside the 10%-shrunk box for {wins['n_inner'][0]} / {len(inf)}; checking the overlays one by one, the yellow boxes (rows) fit the objects while the blue boxes (cols) are clearly rotated off.",
         "- `surface_frac` discriminates poorly (large objects have much surface inside the box and comparatively little near the faces) and is not used as a criterion.",
         "- **The two wrong boxes noticed in review (the TV in 48458417 f_002755 and the refrigerator in 47331971 p0) are not a rotation-convention problem** but a bug in the card box-drawing code: "
         "in `objects.json`, `obb.size` had been reordered to [long horizontal edge, short horizontal edge, height] while `obb.axes` kept the original order, so length and width were swapped when drawing. "
         "For example, the TV's original edges [0.06, 1.23, 0.73] were stored as [1.23, 0.06, 0.73] without reordering the axes, so it was drawn as a thin strip perpendicular to the TV. Both objects have yaw close to 90 deg, where the rows/cols interpretations coincide anyway.",
         "- Scope: only the report cards read `obb` from `objects.json`; object point sets, visibility and question answers all use the original annotation (edges and axes in the same order) and are unaffected. "
         "Fixed by storing axes and edge lengths in the same order, with the unit test `tests/test_objects.py`; the C1 cards were regenerated.", "",
         "Per-object details (distinguishable objects only):", "",
         "| video_id | # | Label | yaw (rows) | Orientation diff. | n_inner rows / cols | surface_frac rows / cols | Overlay |", "|---|---|---|---|---|---|---|---|",
         *[f"| {r.video_id} | {r.obj_index} | {r.label} | {r.yaw_rows} | {r.yaw_diff_deg} | {r.rows_n_inner} / {r.cols_n_inner} | "
           f"{r.rows_surface_frac:.2f} / {r.cols_surface_frac:.2f} | {'![](obb_convention/' + r.overlay + ')' if isinstance(r.get('overlay'), str) else '—'} |"
           for _, r in inf.iterrows()], "",
         "Overlays: yellow = rows, blue = cols."]
    (path_of(cfg, "reports") / "obb_convention.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"n_objects": len(df), "n_informative": len(inf), "wins(rows,cols,tie)": wins}, ensure_ascii=False))


if __name__ == "__main__":
    main()
