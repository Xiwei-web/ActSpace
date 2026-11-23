"""ScanNet++: estimate camera motion speed at keyframes from the 60 FPS poses (motion-blur condition).

For frame number k, take the aligned_pose (T_wc) of frames k - d and k + d (d = scannetpp.motion_half_window_frames, default 2 frames
~ +-1/30 s); angular speed = relative rotation angle / time difference (deg/s), linear speed = camera-centre displacement / time
difference (m/s). Results are cached in work/scannetpp/<sid>/camera_speed.json.
Usage: python -m actspace_build.scannetpp_motion [--scenes ...]
"""
from __future__ import annotations

import argparse

import numpy as np

from .common import load_config, read_json, write_json
from .scannetpp import scene_paths, sp_cfg, sp_work


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def speed_at(poses: dict, k: int, d: int) -> tuple[float, float] | None:
    a, b = poses.get(f"frame_{k - d:06d}"), poses.get(f"frame_{k + d:06d}")
    if a is None or b is None:
        return None
    Ta, Tb = np.asarray(a["aligned_pose"], float), np.asarray(b["aligned_pose"], float)
    dt = float(b["timestamp"]) - float(a["timestamp"])
    if dt <= 0:
        return None
    return rot_angle_deg(Ta[:3, :3], Tb[:3, :3]) / dt, float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) / dt


def camera_speed(cfg: dict, sid: str, force: bool = False) -> dict:
    """frame_id -> {"ang_deg_s", "lin_m_s"} for every frame in frames_all.json (None when a pose is missing)."""
    wd = sp_work(cfg) / sid
    cache = wd / "camera_speed.json"
    d = sp_cfg(cfg).get("motion_half_window_frames", 2)
    if cache.exists() and not force:
        c = read_json(cache)
        if c.get("half_window_frames") == d:
            return c["frames"]
    poses = read_json(scene_paths(cfg, sid).iphone_pose_intrinsic_imu_path)
    out = {}
    for f in read_json(wd / "frames_all.json"):
        r = speed_at(poses, int(f["source_frame"].split("_")[1]), d)
        out[f["frame_id"]] = {"ang_deg_s": r[0], "lin_m_s": r[1]} if r else {"ang_deg_s": None, "lin_m_s": None}
    write_json(cache, {"half_window_frames": d, "frames": out})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    scenes = args.scenes or [l.strip() for l in open(sp_work(cfg) / "_full" / "passed_scenes.txt") if l.strip()]
    for sid in scenes:
        camera_speed(cfg, sid, args.force)
        print(sid, flush=True)


if __name__ == "__main__":
    main()
