#!/usr/bin/env python3
"""ActSpace-Bench: rebuild frame images from the original data (the release contains no images). Works for ARKitScenes and ScanNet++.

Usage:
  python scripts/rebuild_frames.py --release <annotation package or release dir> --out <output dir> \
      [--arkit-root <ARKitScenes data root>] [--scannetpp-root <ScanNet++ data root>] \
      [--scenes <scene_id> ...] [--frames graph|keyframes] [--scannetpp-size 640x480] [--ffmpeg ffmpeg] [--scannetpp-from-video]

  --arkit-root      directory containing raw/Validation/<video_id>/vga_wide/ (rel_path in frames.json is relative to it)
  --scannetpp-root  directory containing data/<scene_id>/iphone/rgb.mkv (source_video in frames.json is relative to it)
  --frames          graph (default): rebuild only viewpoint-graph nodes (in_graph); keyframes: rebuild all keyframes in frames.json
  --scannetpp-size  also save a resized copy of the ScanNet++ frames (640×480 for evaluation and display, see build_info.json);
                    by default only the original resolution is written

Writes <out>/<scene_id>/<frame_id>.png (ARKit) or .jpg (ScanNet++), rotated upright according to rotation_cw_quarters (matching
K, T_wc and image_size in frames.json). ScanNet++ frames are decoded from rgb.mkv by frame index with the same command as during the
build (ffmpeg -q:v 1, as in the official toolbox); if iphone/rgb/frame_XXXXXX.jpg decoded by the official toolbox already exist under
<scannetpp-root>, they are used directly. Depends only on numpy, opencv-python and ffmpeg.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def rotate_cw(img, quarters):
    """Rotate clockwise by quarters × 90° (k of np.rot90 counts counter-clockwise turns)."""
    return np.ascontiguousarray(np.rot90(img, k=-(int(quarters) % 4)))


def rebuild_arkit(frames, root: Path, out: Path):
    n = 0
    for f in frames:
        src = root / f["rel_path"]
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(src)
        cv2.imwrite(str(out / f"{f['frame_id']}.png"), rotate_cw(img, f["rotation_cw_quarters"]))
        n += 1
    return n


def decode_video_frames(video: Path, indices, tmp: Path, ffmpeg: str, chunk=150):
    """Decode jpgs by frame index (starting at 0): frame_index -> file path."""
    got = {}
    idx = sorted(set(indices))
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        expr = "+".join(f"eq(n\\,{k})" for k in part)
        pat = tmp / f"c{s:06d}_%06d.jpg"
        cmd = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video), "-vf", f"select={expr}", "-vsync", "vfr", "-q:v", "1", str(pat)]
        subprocess.run(cmd, check=True)
        outs = sorted(tmp.glob(f"c{s:06d}_*.jpg"))
        if len(outs) != len(part):
            raise RuntimeError(f"{video}: ffmpeg wrote {len(outs)} frames, expected {len(part)}")
        got.update(zip(part, outs))
    return got


def rebuild_scannetpp(frames, root: Path, out: Path, ffmpeg: str, size=None, from_video=False):
    have = {} if from_video else {f["frame_id"]: root / f["rel_path"] for f in frames if (root / f["rel_path"]).exists()}
    todo = [f for f in frames if f["frame_id"] not in have]
    tmpdir = None
    if todo:
        video = root / todo[0]["source_video"]
        if not video.exists():
            raise FileNotFoundError(video)
        tmpdir = Path(tempfile.mkdtemp(prefix="actspace_rebuild_"))
        dec = decode_video_frames(video, [f["video_frame_index"] for f in todo], tmpdir, ffmpeg)
        have.update({f["frame_id"]: dec[f["video_frame_index"]] for f in todo})
    n = 0
    for f in frames:
        q = int(f["rotation_cw_quarters"]) % 4
        dst = out / f"{f['frame_id']}.jpg"
        if q == 0:
            shutil.copyfile(have[f["frame_id"]], dst)   # no re-encoding
            img = None
        else:
            img = rotate_cw(cv2.imread(str(have[f["frame_id"]])), q)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if size:
            img = img if img is not None else cv2.imread(str(dst))
            small = out.parent.parent / f"{out.parent.name}_{size[0]}x{size[1]}" / out.name
            small.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(small / dst.name), cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    if tmpdir:
        shutil.rmtree(tmpdir)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arkit-root")
    ap.add_argument("--scannetpp-root")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--frames", choices=["graph", "keyframes"], default="graph")
    ap.add_argument("--scannetpp-size", help="e.g. 640x480")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--scannetpp-from-video", action="store_true", help="always decode ScanNet++ from rgb.mkv (ignore already decoded jpgs)")
    args = ap.parse_args()
    rel = Path(args.release)
    listing = rel / "scene_list.txt" if (rel / "scene_list.txt").exists() else rel / "metadata" / "scene_list.txt"   # release dir / annotation package
    scenes = args.scenes or listing.read_text().split()
    size = tuple(int(x) for x in args.scannetpp_size.lower().split("x")) if args.scannetpp_size else None
    total = 0
    for sid in scenes:
        meta = json.loads((rel / "scenes" / sid / "scene_meta.json").read_text())
        frames = json.loads((rel / "scenes" / sid / "frames.json").read_text())
        if args.frames == "graph":
            frames = [f for f in frames if f["in_graph"]]
        out = Path(args.out) / sid
        out.mkdir(parents=True, exist_ok=True)
        if meta["dataset"] == "arkitscenes":
            if not args.arkit_root:
                sys.exit("--arkit-root is required for ARKitScenes scenes")
            n = rebuild_arkit(frames, Path(args.arkit_root), out)
        elif meta["dataset"] == "scannetpp":
            if not args.scannetpp_root:
                sys.exit("--scannetpp-root is required for ScanNet++ scenes")
            n = rebuild_scannetpp(frames, Path(args.scannetpp_root), out, args.ffmpeg, size, args.scannetpp_from_video)
        else:
            sys.exit(f"unknown dataset {meta['dataset']}")
        print(f"{sid}: {n} frames")
        total += n
    print(f"done: {len(scenes)} scenes, {total} frames -> {args.out}")


if __name__ == "__main__":
    main()
