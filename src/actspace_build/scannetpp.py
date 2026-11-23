"""ScanNet++ specifics: scene paths and iPhone RGB / depth decoding (Step 0 / Step 2, item 1).

The path layout follows the official toolbox common/scene_release.py (ScannetppScene_Release). The toolbox's
iphone/prepare_iphone_data.py can only extract all frames (or a subset that depends on iphone/colmap) and does not
support subsampling by FPS, so the same procedure is reimplemented here: RGB via ffmpeg (same -q:v as the toolbox),
depth decompressed from the toolbox's per-frame lz4 / zlib format and written as uint16 millimetre png. Only the
frames sampled at frames.scannetpp_decode_fps are decoded; file names frame_%06d keep the original frame index and
map one-to-one to the keys of pose_intrinsic_imu.json.

Raw files are read-only: decoded output is first written to work/scannetpp/<scene_id>/_decode_tmp and then moved to
the toolbox's default iphone/rgb and iphone/depth; existing target files are skipped, never overwritten.

Usage: python -m actspace_build.scannetpp decode [--scenes ID ...] [--workers N] [--force]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, is_done, load_config, mark_done, path_of, write_json

DATASET = "scannetpp"


# ---------------------------------------------------------------- paths
def sp_cfg(cfg: dict) -> dict:
    return cfg["scannetpp"]


def sp_root(cfg: dict) -> Path:
    p = Path(sp_cfg(cfg)["root"])
    return p if p.is_absolute() else ROOT / p


def sp_work(cfg: dict) -> Path:
    return path_of(cfg, "work") / sp_cfg(cfg)["work_subdir"]


def read_split(cfg: dict, split: str | None = None) -> list[str]:
    split = split or sp_cfg(cfg)["split"]
    with open(sp_root(cfg) / "splits" / f"{split}.txt") as f:
        return [l.strip() for l in f if l.strip()]


def _scene_release_cls(cfg: dict):
    """Load the toolbox's ScannetppScene_Release from its file (without adding the toolbox to sys.path, to avoid name clashes with this package's modules)."""
    path = Path(sp_cfg(cfg)["toolbox"]) / "common" / "scene_release.py"
    spec = importlib.util.spec_from_file_location("scannetpp_toolbox_scene_release", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScannetppScene_Release


def scene_paths(cfg: dict, scene_id: str):
    return _scene_release_cls(cfg)(scene_id, data_root=sp_root(cfg) / "data")


# ---------------------------------------------------------------- pilot selection
def draw_stratified_pilots(scene_types: dict[str, str], n: int, rng: np.random.Generator) -> dict:
    """Draw n pilot scenes stratified by scene type: take the n types with the most scenes (ties broken by an
    rng permutation) and draw 1 scene uniformly within each type. Types and scenes are sorted first so the same
    seed gives the same result."""
    by_type: dict[str, list[str]] = {}
    for sid, t in sorted(scene_types.items()):
        by_type.setdefault(t, []).append(sid)
    types = sorted(by_type)
    tie_order = {t: k for k, t in enumerate(rng.permutation(types))}
    ranked = sorted(types, key=lambda t: (-len(by_type[t]), tie_order[t]))
    if len(ranked) < n:
        raise ValueError(f"only {len(ranked)} scene types, cannot draw {n} from distinct types")
    picks = [(t, by_type[t][int(rng.integers(len(by_type[t])))]) for t in ranked[:n]]
    return {"type_counts": {t: len(by_type[t]) for t in ranked}, "type_order": ranked,
            "picks": [{"scene_id": s, "scene_type": t, "type_count": len(by_type[t])} for t, s in picks]}


# ---------------------------------------------------------------- geometry
def _obb_from_xy_dir(p: np.ndarray, e1: np.ndarray) -> dict:
    """Gravity-aligned OBB for a given box-axis direction e1 in the xy plane. Row i of axes is the i-th box axis (unit
    vector in world frame) and size[i] is the extent along axes[i]; axes is right-handed (row 0 = e1, row 1 = e1 rotated
    90 deg counter-clockwise, row 2 = +z). Same representation as the ARKit annotations, usable by step1_arkit.obb_corners."""
    e1 = np.asarray(e1, float) / np.linalg.norm(e1)
    if e1[0] < -1e-12 or (abs(e1[0]) <= 1e-12 and e1[1] < 0):   # fix the sign (non-negative x component) for reproducibility
        e1 = -e1
    e2 = np.array([-e1[1], e1[0]])
    axes = np.array([[e1[0], e1[1], 0.0], [e2[0], e2[1], 0.0], [0.0, 0.0, 1.0]])
    local = p @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    return {"center": ((lo + hi) / 2) @ axes, "axes": axes, "size": hi - lo}


def _pca_dir(xy: np.ndarray) -> np.ndarray:
    xy = xy - xy.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(xy.T) if len(xy) > 1 else np.eye(2))
    return v[:, int(np.argmax(w))]


def object_obb_pca(points: np.ndarray) -> dict:
    """Original method: orientation from PCA in the xy plane, z from the point-set bounds. Replaced by object_obb;
    kept only for the old-vs-new box comparison in reports."""
    p = np.asarray(points, dtype=float)
    return _obb_from_xy_dir(p, _pca_dir(p[:, :2]))


def min_area_rect_dir(xy: np.ndarray) -> np.ndarray:
    """Direction of one side of the minimum-area bounding rectangle of an xy point set (rotating calipers: the optimal
    rectangle has a side collinear with some convex-hull edge, so enumerate edges). Returns the direction of the longer
    side; for equal sides, the one with larger |x| component (reproducible). Falls back to the PCA direction for < 3
    points or collinear points."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.asarray(xy, float)
    try:
        hull = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):
        return _pca_dir(xy)
    edges = np.roll(hull, -1, axis=0) - hull
    edges = edges[np.linalg.norm(edges, axis=1) > 1e-12]
    u = edges / np.linalg.norm(edges, axis=1, keepdims=True)                 # direction of each hull edge
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)
    pu, pv = hull @ u.T, hull @ v.T                                          # (hull point, edge)
    ext_u, ext_v = pu.max(0) - pu.min(0), pv.max(0) - pv.min(0)
    k = int(np.argmin(ext_u * ext_v))
    if np.isclose(ext_u[k], ext_v[k], rtol=1e-9, atol=1e-12):
        return u[k] if abs(u[k, 0]) >= abs(v[k, 0]) else v[k]
    return u[k] if ext_u[k] > ext_v[k] else v[k]


def object_obb(points: np.ndarray) -> dict:
    """Gravity-aligned OBB of an object point set: orientation from the minimum-area bounding rectangle in the xy plane,
    z from the point-set bounds (deliberately not xy PCA: for near-square objects or unevenly distributed points PCA
    rotates onto the diagonal and inflates the box). Row 0 of axes is the direction of the rectangle's longer side;
    size is in the same order as axes (size[i] along axes[i])."""
    p = np.asarray(points, dtype=float)
    return _obb_from_xy_dir(p, min_area_rect_dir(p[:, :2]))


def scale_intrinsics(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    """Scale intrinsics from src resolution to dst resolution (pixel centres at integer coordinates: u' = (u + 0.5)*s - 0.5)."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    K = np.asarray(K, dtype=float)
    return np.array([[K[0, 0] * sx, 0.0, (K[0, 2] + 0.5) * sx - 0.5],
                     [0.0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------- frame sampling
def decode_frame_ids(n_frames: int, src_fps: float, target_fps: float) -> list[int]:
    """Evenly spaced original frame indices starting at 0 for the target fps; the source fps must be an integer multiple of the target fps."""
    ratio = src_fps / target_fps
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(f"source fps {src_fps} is not an integer multiple of target fps {target_fps}")
    return list(range(0, n_frames, stride))


def video_info(path: Path) -> dict:
    """Count packets with ffprobe (the frame count in the container header is not trusted)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den), "n_frames": int(s["nb_read_packets"])}


def snapshot_files(scene_dir: Path) -> dict:
    """(size, mtime) of every file in the scene directory, recorded before decoding to verify that raw files were not modified."""
    return {str(p.relative_to(scene_dir)): [p.stat().st_size, p.stat().st_mtime]
            for p in sorted(scene_dir.rglob("*")) if p.is_file()}


def changed_files(scene_dir: Path, snapshot: dict) -> list[str]:
    out = []
    for rel, (size, mtime) in snapshot.items():
        p = scene_dir / rel
        if not p.is_file() or p.stat().st_size != size or abs(p.stat().st_mtime - mtime) > 1e-3:
            out.append(rel)
    return out


def move_no_overwrite(src: Path, dst: Path) -> bool:
    """Move a file; if the target already exists, do not overwrite and return False."""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


# ---------------------------------------------------------------- depth
def iter_depth_records(path: Path, height: int, width: int):
    """Yield (frame index, uint16 depth in mm) in order. Same format as the toolbox's extract_depth:
    per frame [uint32 length][lz4 uint16 mm or zlib float32 m]; whole-file zlib float32 m as the fallback."""
    import lz4.block

    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        size0 = int.from_bytes(head, "little") if len(head) == 4 else 0
        per_frame = 0 < size0 <= height * width * 4 + 1024
        if not per_frame:
            depth = np.frombuffer(zlib.decompress(f.read(), wbits=-zlib.MAX_WBITS), dtype=np.float32)
            for i, d in enumerate(depth.reshape(-1, height, width)):
                yield i, (d * 1000).astype(np.uint16)
            return
        i = 0
        while True:
            h = f.read(4)
            if len(h) == 0:
                break
            data = f.read(int.from_bytes(h, "little"))
            try:
                d = np.frombuffer(lz4.block.decompress(data, uncompressed_size=height * width * 2),
                                  dtype=np.uint16).reshape(height, width)
            except Exception:
                d = np.frombuffer(zlib.decompress(data, wbits=-zlib.MAX_WBITS), dtype=np.float32)
                d = (d.reshape(height, width) * 1000).astype(np.uint16)
            yield i, d
            i += 1


def count_depth_records(path: Path, height: int, width: int) -> int:
    """Count per-frame records by reading only the length fields (no decompression); for the whole-file zlib format, decompress and divide by the frame size."""
    with open(path, "rb") as f:
        head = f.read(4)
        size0 = int.from_bytes(head, "little") if len(head) == 4 else 0
        if not 0 < size0 <= height * width * 4 + 1024:
            f.seek(0)
            return len(zlib.decompress(f.read(), wbits=-zlib.MAX_WBITS)) // (height * width * 4)
        f.seek(0)
        n = 0
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            f.seek(int.from_bytes(h, "little"), 1)
            n += 1
        return n


# ---------------------------------------------------------------- decoding
def decode_rgb(scene, ids: list[int], stride: int, tmp: Path, qscale: int) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(scene.iphone_video_path),
           "-vf", f"select=not(mod(n\\,{stride}))", "-vsync", "vfr", "-q:v", str(qscale),
           "-start_number", "0", str(tmp / "%06d.jpg")]
    subprocess.run(cmd, check=True)
    outs = sorted(tmp.glob("*.jpg"))
    if len(outs) != len(ids):
        raise RuntimeError(f"{scene.scene_id}: ffmpeg wrote {len(outs)} rgb frames, expected {len(ids)}")
    moved = skipped = 0
    for k, src in enumerate(outs):      # k-th output = original frame ids[k]
        if move_no_overwrite(src, scene.iphone_rgb_dir / f"frame_{ids[k]:06d}.jpg"):
            moved += 1
        else:
            skipped += 1
            src.unlink()
    return {"written": moved, "skipped_existing": skipped}


def decode_depth(scene, ids: list[int], tmp: Path, height: int, width: int) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    want = set(ids)
    moved = skipped = n_records = 0
    for i, d in iter_depth_records(scene.iphone_depth_path, height, width):
        n_records += 1
        if i not in want:
            continue
        src = tmp / f"frame_{i:06d}.png"
        if not cv2.imwrite(str(src), d):
            raise RuntimeError(f"{scene.scene_id}: failed to write depth frame {i}")
        if move_no_overwrite(src, scene.iphone_depth_dir / src.name):
            moved += 1
        else:
            skipped += 1
            src.unlink()
    missing = sorted(want - set(range(n_records)))
    if missing:
        raise RuntimeError(f"{scene.scene_id}: depth.bin has {n_records} records, missing frames {missing[:5]}...")
    return {"written": moved, "skipped_existing": skipped, "n_records": n_records}


def decode_scene(scene_id: str, force: bool = False) -> dict:
    cfg = load_config()
    sc = sp_cfg(cfg)
    wd = sp_work(cfg) / scene_id
    wd.mkdir(parents=True, exist_ok=True)
    if is_done(wd, "decode") and not force:
        return {"scene_id": scene_id, "status": "cached"}
    t0 = time.time()
    scene = scene_paths(cfg, scene_id)
    snap = wd / "raw_snapshot.json"
    if not snap.exists():
        write_json(snap, snapshot_files(scene.scene_root_dir))
    info = video_info(scene.iphone_video_path)
    ids = decode_frame_ids(info["n_frames"], info["fps"], cfg["frames"]["scannetpp_decode_fps"])
    stride = ids[1] - ids[0] if len(ids) > 1 else 1
    w, h = sc["depth_size"]
    tmp = wd / "_decode_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    rgb = decode_rgb(scene, ids, stride, tmp / "rgb", sc["rgb_jpeg_qscale"])
    depth = decode_depth(scene, ids, tmp / "depth", h, w)
    shutil.rmtree(tmp, ignore_errors=True)
    rec = {"scene_id": scene_id, "status": "decoded", "video": info, "stride": stride,
           "decode_fps": cfg["frames"]["scannetpp_decode_fps"], "n_decoded": len(ids),
           "frame_ids_first_last": [ids[0], ids[-1]], "rgb": rgb, "depth": depth,
           "rgb_dir": str(scene.iphone_rgb_dir), "depth_dir": str(scene.iphone_depth_dir),
           "seconds": round(time.time() - t0, 1)}
    write_json(wd / "decode_info.json", rec)
    mark_done(wd, "decode")
    return rec


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode", help="decode rgb.mkv and depth.bin (default: pilot scenes)")
    d.add_argument("--scenes", nargs="*", help="scene IDs; default scannetpp.pilot_scenes")
    d.add_argument("--workers", type=int, help="number of worker processes; default scannetpp.decode_workers")
    d.add_argument("--force", action="store_true", help="rerun ignoring the done marker (existing frames are still never overwritten)")
    args = ap.parse_args()
    cfg = load_config()
    scenes = args.scenes or sp_cfg(cfg)["pilot_scenes"]
    workers = args.workers or sp_cfg(cfg)["decode_workers"]
    print(f"decode {len(scenes)} scenes with {workers} workers, ffmpeg={shutil.which('ffmpeg')}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(decode_scene, scenes, [args.force] * len(scenes)):
            print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
