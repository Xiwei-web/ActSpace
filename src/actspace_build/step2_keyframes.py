"""Step 2: blur filtering (within-scene percentile) and keyframe selection.

Usage: python -m actspace_build.step2_keyframes --dataset arkitscenes --scenes ... [--force]
Input: work/<dataset>/<vid>/frames_all.json; output: keyframes.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, work_dir, write_json


def laplacian_variance(path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_keyframes(frames: list[dict], min_trans: float, min_rot: float, sharp_window_s: float, eligible=None) -> list[int]:
    """In time order, the first frame whose translation >= min_trans or |dyaw| >= min_rot relative to the previous keyframe is the trigger frame;
    the keyframe is the frame with the largest Laplacian variance within +-sharp_window_s of the trigger (and after the previous keyframe), and scanning resumes from it.
    If eligible (boolean sequence) is given, only eligible frames are chosen (only frames with a time-matched sensor depth);
    if the window has no eligible frame, nothing is chosen this time and scanning continues. frames must carry blur_lapvar and be sorted by time."""
    ts = np.array([f["timestamp"] for f in frames])
    ok = np.ones(len(frames), bool) if eligible is None else np.asarray(eligible, bool)
    first = np.flatnonzero(ok)
    if not len(first):
        return []
    keep = [int(first[0])]
    i = keep[0] + 1
    while i < len(frames):
        last = frames[keep[-1]]
        f = frames[i]
        dt = np.linalg.norm(np.asarray(f["position"]) - np.asarray(last["position"]))
        dyaw = abs(g.wrap_deg(f["yaw_deg"] - last["yaw_deg"]))
        if dt >= min_trans or dyaw >= min_rot:
            lo = max(int(np.searchsorted(ts, ts[i] - sharp_window_s, "left")), keep[-1] + 1)
            hi = int(np.searchsorted(ts, ts[i] + sharp_window_s, "right"))
            cand = [k for k in range(lo, hi) if ok[k]]
            if not cand:
                i += 1
                continue
            best = max(cand, key=lambda k: frames[k]["blur_lapvar"])
            keep.append(best)
            i = best + 1
        else:
            i += 1
    return keep


def keyframes_scene(dataset: str, vid: str, data_root, force: bool = False):
    cfg = load_config()
    fc = cfg["frames"]
    wd = work_dir(cfg, dataset, vid)
    if is_done(wd, "keyframes") and not force:
        return read_json(wd / "keyframes.json")
    frames = read_json(wd / "frames_all.json")
    drops = DropLog(wd, "step2_keyframes")
    with ThreadPoolExecutor(16) as ex:
        blur = list(ex.map(lambda f: laplacian_variance(data_root / f["rel_path"]), frames))
    # take the stricter of two blur thresholds: the within-scene percentile and a fixed fraction of the scene median
    thr_pct = float(np.percentile(blur, fc["blur_drop_percentile"]))
    thr_rel = float(fc["blur_min_rel_to_median"] * np.median(blur))
    thr = max(thr_pct, thr_rel)
    sharp = []
    for f, b in zip(frames, blur):
        f["blur_lapvar"] = b
        if b < thr:
            drops.add("frame", f["frame_id"], "BLURRY", lapvar=b, threshold=thr,
                      rule="percentile" if b < thr_pct else "rel_to_median")
        else:
            sharp.append(f)
    eligible = None
    if fc["keyframe_require_depth_match"] and dataset.startswith("arkitscenes"):
        files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
        d_ts = np.array([t for t, _ in arkit.list_frames(files["lowres_depth"], ".png")])
        tol = cfg["visibility"]["sensor_depth_tol_s"]
        eligible = [float(np.min(np.abs(d_ts - f["timestamp"]))) <= tol for f in sharp]
    kf_idx = select_keyframes(sharp, fc["keyframe_min_translation_m"], fc["keyframe_min_rotation_deg"], fc["keyframe_sharpest_window_s"], eligible)
    kfs = [sharp[i] for i in kf_idx]
    ok = len(kfs) >= fc["min_keyframes_per_scene"]
    if not ok:
        drops.add("scene", vid, "TOO_FEW_KEYFRAMES", num_keyframes=len(kfs))
    res = {"num_frames": len(frames), "num_blurry": len(frames) - len(sharp), "blur_threshold": thr,
           "blur_threshold_percentile": thr_pct, "blur_threshold_rel_median": thr_rel,
           "num_blurry_rel_only": int(sum(thr_pct <= b < thr_rel for b in blur)),
           "keyframe_lapvar_median": float(np.median([f["blur_lapvar"] for f in kfs])) if kfs else None,
           "frame_lapvar_median": float(np.median(blur)),
           "sharp_frames_with_depth_match": None if eligible is None else int(np.sum(eligible)),
           "num_keyframes": len(kfs), "scene_ok": ok, "keyframe_ids": [f["frame_id"] for f in kfs]}
    write_json(wd / "keyframes.json", res)
    mark_done(wd, "keyframes")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arkitscenes")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    data_root = path_of(cfg, "arkit_3dod_root").resolve().parents[1]  # ARKitScenes/data
    scenes = [str(s) for s in (args.scenes or cfg["arkitscenes"]["pilot_scenes"])]
    for vid in scenes:
        r = keyframes_scene(args.dataset, vid, data_root, args.force)
        print(vid, {k: v for k, v in r.items() if k != "keyframe_ids"})


if __name__ == "__main__":
    main()
