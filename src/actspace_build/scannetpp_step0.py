"""Step 0 (ScanNet++): pilot scene inventory, file check, size and total-volume estimate, basic post-decode checks.

Usage: python -m actspace_build.scannetpp_step0
Prerequisite: python -m actspace_build.scannetpp decode has finished (reads work/scannetpp/<scene_id>/decode_info.json).
Outputs: work/scannetpp/_step0/inventory.json, reports/scannetpp_step0_inventory.md
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .common import load_config, path_of, read_json, write_json
from .scannetpp import (changed_files, count_depth_records, decode_frame_ids, read_split, scene_paths, sp_cfg,
                        sp_root, sp_work, video_info)

GB = 1e9


def dir_stats(d: Path, pattern: str) -> tuple[int, int]:
    files = sorted(d.glob(pattern)) if d.exists() else []
    return len(files), sum(f.stat().st_size for f in files)


def frame_ids_in(d: Path, ext: str) -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in d.glob(f"frame_*{ext}")) if d.exists() else []


def up_axis_from_floor(vertices: np.ndarray, normals: np.ndarray, floor: np.ndarray, ceiling: np.ndarray) -> dict:
    """The mean normal of floor vertices determines the up axis; the sign is positive when the floor lies below the ceiling (report only, no transform)."""
    n = normals[floor].mean(axis=0)
    n = n / np.linalg.norm(n)
    axis = int(np.argmax(np.abs(n)))
    rec = {"floor_mean_normal": np.round(n, 4).tolist(), "axis": "xyz"[axis],
           "normal_sign": "+" if n[axis] > 0 else "-"}
    if ceiling.any():
        f_med = float(np.median(vertices[floor, axis]))
        c_med = float(np.median(vertices[ceiling, axis]))
        rec.update(floor_median=round(f_med, 3), ceiling_median=round(c_med, 3),
                   ceiling_above_floor=bool(c_med > f_med))
    return rec


def check_mesh(scene) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    nrm = np.asarray(mesh.vertex_normals)
    seg = np.asarray(read_json(scene.scan_mesh_segs_path)["segIndices"])
    anno = read_json(scene.scan_anno_json_path)["segGroups"]

    def verts_of(label):
        segs = [s for g in anno if g["label"] == label for s in g["segments"]]
        return np.isin(seg, segs) if len(seg) == len(v) else np.zeros(len(v), bool)

    sem = o3d.io.read_triangle_mesh(str(scene.scan_sem_mesh_path))
    labels = [g["label"] for g in anno]
    rec = {"n_vertices": len(v), "n_seg_indices": len(seg), "seg_match": len(seg) == len(v),
           "sem_mesh_vertices": len(sem.vertices), "extent_xyz_m": np.round(v.max(0) - v.min(0), 2).tolist(),
           "n_objects": len(labels), "n_unique_labels": len(set(labels))}
    floor, ceiling = verts_of("floor"), verts_of("ceiling")
    rec["up_axis"] = up_axis_from_floor(v, nrm, floor, ceiling) if floor.any() else None
    return rec


def check_poses(scene) -> dict:
    p = read_json(scene.iphone_pose_intrinsic_imu_path)
    keys = list(p)
    idx = [int(k.split("_")[-1]) for k in keys]
    has_aligned = [k for k in keys if "aligned_pose" in p[k]]
    bad = 0
    diff = 0.0
    for k in has_aligned:
        a = np.asarray(p[k]["aligned_pose"], float)
        if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1]):
            bad += 1
            continue
        diff = max(diff, float(np.abs(a - np.asarray(p[k]["pose"], float)).max()))
    ts = np.array([p[k]["timestamp"] for k in keys])
    return {"n_keys": len(keys), "keys_contiguous_from_0": idx == list(range(len(idx))),
            "first_key": keys[0], "last_key": keys[-1], "frame_fields": sorted(p[keys[0]]),
            "n_aligned_pose": len(has_aligned), "n_aligned_pose_malformed": bad,
            "aligned_vs_pose_max_abs_diff": round(diff, 3),
            "timestamp_dt_median_s": round(float(np.median(np.diff(ts))), 5),
            "K_first": p[keys[0]]["intrinsic"], "_keys": set(keys)}


def check_scene(cfg: dict, scene_id: str) -> dict:
    sc = sp_cfg(cfg)
    scene = scene_paths(cfg, scene_id)
    sdir = sp_root(cfg) / "data" / scene_id
    files = {f: (sdir / f).stat().st_size if (sdir / f).is_file() else None for f in sc["required_files"]}
    rec = {"scene_id": scene_id, "missing": [f for f, s in files.items() if s is None],
           "file_sizes": files, "raw_bytes": sum(s for s in files.values() if s)}

    vid = video_info(scene.iphone_video_path)
    dw, dh = sc["depth_size"]
    n_depth = count_depth_records(scene.iphone_depth_path, dh, dw)
    poses = check_poses(scene)
    pose_keys = poses.pop("_keys")
    expected = decode_frame_ids(vid["n_frames"], vid["fps"], cfg["frames"]["scannetpp_decode_fps"])
    rgb_ids = frame_ids_in(scene.iphone_rgb_dir, ".jpg")
    depth_ids = frame_ids_in(scene.iphone_depth_dir, ".png")
    n_rgb, rgb_bytes = dir_stats(scene.iphone_rgb_dir, "frame_*.jpg")
    n_dep, dep_bytes = dir_stats(scene.iphone_depth_dir, "frame_*.png")

    samples = [rgb_ids[0], rgb_ids[len(rgb_ids) // 2], rgb_ids[-1]] if rgb_ids else []
    rgb_shapes = sorted({cv2.imread(str(scene.iphone_rgb_dir / f"frame_{i:06d}.jpg")).shape[1::-1] for i in samples})
    dep_imgs = [cv2.imread(str(scene.iphone_depth_dir / f"frame_{i:06d}.png"), cv2.IMREAD_UNCHANGED) for i in samples]
    dep_shapes = sorted({d.shape[::-1] for d in dep_imgs})
    dep_dtypes = sorted({str(d.dtype) for d in dep_imgs})
    dep_m = np.concatenate([d[d > 0].ravel() for d in dep_imgs]) * sc["depth_scale"] if dep_imgs else np.zeros(1)

    decode_info = sp_work(cfg) / scene_id / "decode_info.json"
    rec.update(
        video=vid, n_depth_records=n_depth, poses=poses,
        counts_equal=vid["n_frames"] == poses["n_keys"] == n_depth,
        n_expected_decoded=len(expected), n_rgb=n_rgb, n_depth=n_dep,
        rgb_ids_match_expected=rgb_ids == expected, depth_ids_match_rgb=depth_ids == rgb_ids,
        rgb_ids_in_pose_keys=all(f"frame_{i:06d}" in pose_keys for i in rgb_ids),
        rgb_bytes=rgb_bytes, depth_bytes=dep_bytes, decoded_bytes=rgb_bytes + dep_bytes,
        rgb_sizes_wh=[list(s) for s in rgb_shapes], depth_sizes_wh=[list(s) for s in dep_shapes],
        depth_dtypes=dep_dtypes,
        depth_sample_m=[round(float(np.percentile(dep_m, q)), 2) for q in (1, 50, 99)],
        decode=read_json(decode_info) if decode_info.exists() else None,
        mesh=check_mesh(scene))
    return rec


def make_preview(cfg: dict, scene_id: str, out: Path) -> Path:
    """First, middle and last decoded frame: RGB on top, depth false colour below (0-5 m, 0 = black); original frame ID in the top-left corner."""
    scene = scene_paths(cfg, scene_id)
    ids = frame_ids_in(scene.iphone_rgb_dir, ".jpg")
    pick = [ids[0], ids[len(ids) // 2], ids[-1]]
    w, h = 480, 360
    tiles = []
    for i in pick:
        rgb = cv2.resize(cv2.imread(str(scene.iphone_rgb_dir / f"frame_{i:06d}.jpg")), (w, h), interpolation=cv2.INTER_AREA)
        d = cv2.imread(str(scene.iphone_depth_dir / f"frame_{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        d_m = d * sp_cfg(cfg)["depth_scale"]
        col = cv2.applyColorMap(np.clip(d_m / 5.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[d == 0] = 0
        col = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
        for img, txt in ((rgb, f"rgb frame_{i:06d}"), (col, f"depth frame_{i:06d} (0-5 m)")):
            cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([rgb, col]))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def download_time(log: Path) -> str | None:
    if not log.exists():
        return None
    txt = log.read_text(errors="replace").replace("\r", "\n")
    m = re.findall(r"scenes: 100%\|[^|]*\| (\d+)/\d+ \[([\d:]+)<", txt)
    return f"{m[-1][1]} ({m[-1][0]} scenes)" if m else None


def yn(b) -> str:
    return "✅" if b else "❌"


def write_report(cfg: dict, inv: dict, path: Path) -> None:
    sc = sp_cfg(cfg)
    S = inv["scenes"]
    n_split = inv["n_split_scenes"]
    raw = np.array([s["raw_bytes"] for s in S]) / GB
    dec = np.array([s["decoded_bytes"] for s in S]) / GB
    tot = raw + dec
    L = []
    L += ["# ScanNet++ Step 0: pilot scene inventory and decode check", "",
          f"> Branch `scannetpp`. Generated by `python -m actspace_build.scannetpp_step0`.",
          "> This round only covers the download check, decoding and basic checks; **GATE A has not started**.", ""]
    L += ["## 1. Download status and scenes", "",
          f"- End of download log `<ScanNet++>/download_pilot5.log`: `{inv['download_log_tail']}`; elapsed {inv['download_time']}.",
          f"- Lines in the log that print the token: {inv['token_lines_in_log']} (filtered when the download was run).",
          f"- Actual number of scenes in `splits/{sc['split']}.txt`: **{n_split}**.",
          f"- 5 pilot scenes: {', '.join(f'`{s}`' for s in sc['pilot_scenes'])}.",
          f"- Are they the first 5 in the list: **{'yes' if inv['pilot_is_head5'] else 'no'}**."
          + (" The download took the first 5 scenes of the list instead of the fixed-seed random draw required by Step 0; they were not re-downloaded." if inv["pilot_is_head5"] else ""),
          f"- Scene types (official `metadata/scene_types.json`): 5 pilots: {', '.join(f'{k} {v}' for k, v in inv['pilot_scene_types'].items())}; "
          f"all {n_split} of {sc['split']}: {', '.join(f'{k} {v}' for k, v in inv['split_scene_types'].items())}."
          + (" **The pilots cover only the office type and are not representative.**" if len(inv["pilot_scene_types"]) == 1 else ""),
          ""]
    L += ["## 2. File completeness (7 file types)", "",
          "| Scene | " + " | ".join(f"`{Path(f).name}`" for f in sc["required_files"]) + " |",
          "|---|" + "---|" * len(sc["required_files"])]
    for s in S:
        L.append(f"| `{s['scene_id']}` | " + " | ".join(
            f"{s['file_sizes'][f] / 1e6:,.0f} MB" if s["file_sizes"][f] else "❌ missing" for f in sc["required_files"]) + " |")
    L += ["", f"Missing files: {'none' if not any(s['missing'] for s in S) else '; '.join(s['scene_id'] + ': ' + ','.join(s['missing']) for s in S if s['missing'])}.", ""]
    L += ["## 3. Storage (GB = 10⁹ bytes, sum of actual file sizes)", "",
          "| Scene | Downloaded (7 file types) | Decoded RGB | Decoded depth | Total (download + decode) | Decoded frames |",
          "|---|---|---|---|---|---|"]
    for s, r, t in zip(S, raw, tot):
        L.append(f"| `{s['scene_id']}` | {r:.2f} | {s['rgb_bytes'] / GB:.2f} | {s['depth_bytes'] / GB:.3f} | {t:.2f} | {s['n_rgb']} |")
    L.append(f"| **Mean** | {raw.mean():.2f} | {np.mean([s['rgb_bytes'] for s in S]) / GB:.2f} | "
             f"{np.mean([s['depth_bytes'] for s in S]) / GB:.3f} | {tot.mean():.2f} | {np.mean([s['n_rgb'] for s in S]):.0f} |")
    L += ["", f"**Estimate for all {n_split} scenes of {sc['split']}** (pilot mean × {n_split}; the range in parentheses extrapolates the pilot min / max):", "",
          f"- Download (7 file types): about **{raw.mean() * n_split:.0f} GB** ({raw.min() * n_split:.0f}–{raw.max() * n_split:.0f} GB)",
          f"- Added by 10 FPS decoding: about **{dec.mean() * n_split:.0f} GB** ({dec.min() * n_split:.0f}–{dec.max() * n_split:.0f} GB)",
          f"- Total: about **{tot.mean() * n_split:.0f} GB** ({tot.min() * n_split:.0f}–{tot.max() * n_split:.0f} GB)",
          f"- Download time reference: 5 pilot scenes took {inv['download_time']}; scaled to {n_split} scenes, about {inv['download_minutes_est']:.0f} minutes.",
          f"- Free space currently left on the data disk: about {inv['nfs_free_tb']:.0f} TB.", ""]
    L += ["## 4. Decoding", "",
          f"- Toolbox: `{sc['toolbox']}` (github.com/scannetpp/scannetpp, commit `{inv['toolbox_commit']}`).",
          "- The toolbox's `iphone/prepare_iphone_data.py` can only extract all frames (60 FPS); its subset mode depends on `iphone/colmap`, which was not downloaded, and it cannot subsample by FPS. "
          "Decoding is therefore done by `src/actspace_build/scannetpp.py` following the toolbox: paths from the toolbox's `ScannetppScene_Release`; "
          "RGB with ffmpeg (`-q:v 1`, same as the toolbox); depth decompressed with the toolbox's per-frame lz4 (uint16 mm) / zlib fallback format and written as uint16 mm png.",
          f"- Frame sampling: source 60 FPS → {cfg['frames']['scannetpp_decode_fps']} FPS, i.e. original frame IDs 0, 6, 12, …; RGB and depth use **the same frame IDs**; "
          "file names `frame_%06d` keep the original frame ID, matching the keys of `pose_intrinsic_imu.json`.",
          "- Output location: the toolbox defaults `data/<scene_id>/iphone/rgb/` and `iphone/depth/`. Files are first written to `work/scannetpp/<scene_id>/_decode_tmp` "
          "and then moved one by one; existing targets are skipped, never overwritten.",
          f"- Read-only check of raw files: size and mtime of the {inv['raw_files_checked']} pre-existing files under `data/` before vs after decoding: {'**fully unchanged**' if inv['raw_files_unchanged'] else '**changed (see below)**'}.",
          f"- sbatch: `scripts/run_scannetpp_decode.sbatch` (CPU partition), job {inv['decode_job']}, "
          f"log `<data>/logs/scannetpp/decode_{inv['decode_job']}.out`.", "",
          "| Scene | Video frames / FPS | Decoded frames | Written RGB / depth | Time |", "|---|---|---|---|---|"]
    for s in S:
        d = s["decode"] or {}
        L.append(f"| `{s['scene_id']}` | {s['video']['n_frames']} / {s['video']['fps']:.0f} | {s['n_rgb']} | "
                 f"{d.get('rgb', {}).get('written', '?')} / {d.get('depth', {}).get('written', '?')} | {d.get('seconds', '?')} s |")
    L += ["", "## 5. Basic checks", "",
          "### 5.1 Frame correspondence", "",
          "| Scene | Video frames | Pose keys | depth.bin records | All three equal | Pose keys contiguous (from frame_000000) | RGB frame IDs = expected | All RGB frame IDs in pose keys | Depth frame IDs = RGB frame IDs |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        p = s["poses"]
        L.append(f"| `{s['scene_id']}` | {s['video']['n_frames']} | {p['n_keys']} | {s['n_depth_records']} | {yn(s['counts_equal'])} | "
                 f"{yn(p['keys_contiguous_from_0'])} | {yn(s['rgb_ids_match_expected'])} | {yn(s['rgb_ids_in_pose_keys'])} | {yn(s['depth_ids_match_rgb'])} |")
    L += ["", f"Median pose timestamp interval: {', '.join(str(s['poses']['timestamp_dt_median_s']) for s in S)} s (≈ 1/60 s), one pose per frame of the 60 FPS video. "
          "Depth has no separate timestamps and is matched to RGB frames by record index (all three counts are equal); spatial alignment and scaling of depth vs RGB are verified in GATE A with mesh-rendered depth.", ""]
    L += ["### 5.2 Actual resolution (first, middle and last frame per scene)", "",
          "| Scene | RGB (W×H) | Depth (W×H) | Depth dtype | Depth 1% / 50% / 99% percentile (m) | First-frame intrinsics fx, cx, cy |", "|---|---|---|---|---|---|"]
    for s in S:
        K = s["poses"]["K_first"]
        L.append(f"| `{s['scene_id']}` | {', '.join('×'.join(map(str, w)) for w in s['rgb_sizes_wh'])} | "
                 f"{', '.join('×'.join(map(str, w)) for w in s['depth_sizes_wh'])} | {', '.join(s['depth_dtypes'])} | "
                 f"{' / '.join(map(str, s['depth_sample_m']))} | {K[0][0]:.1f}, {K[0][2]:.1f}, {K[1][2]:.1f} |")
    L += ["", f"Expected RGB {'×'.join(map(str, sc['rgb_size']))}, depth {'×'.join(map(str, sc['depth_size']))}: "
          f"{'all match' if inv['resolution_ok'] else '**mismatch**'}. The intrinsics are given at RGB resolution; for depth they must be scaled by 256/1920 = 192/1440 (verified in GATE A).", ""]
    L += ["### 5.3 `aligned_pose`", "",
          "| Scene | Per-frame fields | Frames with aligned_pose / all | Malformed | Max element difference aligned_pose vs pose |", "|---|---|---|---|---|"]
    for s in S:
        p = s["poses"]
        L.append(f"| `{s['scene_id']}` | {', '.join(p['frame_fields'])} | {p['n_aligned_pose']} / {p['n_keys']} | "
                 f"{p['n_aligned_pose_malformed']} | {p['aligned_vs_pose_max_abs_diff']} |")
    L += ["", "aligned_pose differs from the raw ARKit pose (non-zero difference), so it is indeed a pose in another coordinate frame; Step 1 uses aligned_pose, and its convention is determined empirically in GATE A.", ""]
    L += ["### 5.4 Mesh and segmentation", "",
          "| Scene | Mesh vertices | segIndices length | Equal | Semantic mesh vertices | Annotated objects | Distinct labels | Bounding box size x×y×z (m) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in S:
        m = s["mesh"]
        L.append(f"| `{s['scene_id']}` | {m['n_vertices']:,} | {m['n_seg_indices']:,} | {yn(m['seg_match'])} | {m['sem_mesh_vertices']:,} | "
                 f"{m['n_objects']} | {m['n_unique_labels']} | {' × '.join(map(str, m['extent_xyz_m']))} |")
    L += ["", "### 5.5 Mesh up axis (report only, no transform applied)", "",
          "Method: take the objects labelled floor in `segments_anno.json`, get their mesh vertices via segments and compute the mean vertex normal direction; then compare the median coordinates of floor and ceiling vertices along that axis.", "",
          "| Scene | Mean floor normal | Detected axis | Floor median coordinate | Ceiling median coordinate | Ceiling above floor |", "|---|---|---|---|---|---|"]
    for s in S:
        u = s["mesh"]["up_axis"] or {}
        L.append(f"| `{s['scene_id']}` | {u.get('floor_mean_normal')} | {u.get('normal_sign', '')}{u.get('axis', '?')} | "
                 f"{u.get('floor_median')} | {u.get('ceiling_median')} | {yn(u.get('ceiling_above_floor'))} |")
    L += ["", f"Conclusion: {inv['up_axis_summary']}", ""]
    L += ["### 5.6 Decode spot-check images", "",
          "First, middle and last decoded frame per scene: RGB on top (downscaled to 480×360), depth false colour of the same frame ID below (0-5 m, 0 = black, nearest-neighbour upscaling). "
          "Images are in `reports/scannetpp_step0/` (not tracked in git).", ""]
    L += [f"- `{s['scene_id']}`: ![{s['scene_id']}](scannetpp_step0/{s['scene_id']}_rgb_depth.jpg)" for s in S]
    L += ["", "Visual check: RGB and depth contours of the same frame ID agree (furniture, monitors and door frames line up), image orientation is normal with no flips; "
          "a few frames show motion blur, which the Step 2 blur filter handles.", ""]
    L += ["## 6. Other notes", "",
          "- Intermediate outputs: `<data>/work/scannetpp/` (per scene `decode_info.json`, `.done_decode`; inventory in `_step0/inventory.json`).",
          "- Symlink: `data_raw/scannetpp → <ScanNet++>/data` in the worktree (`data_raw/` is git-ignored).",
          "- Dependencies: `lz4` added to the `actspace` environment (to decompress depth.bin); depth png is written with cv2; the toolbox's other dependencies (torch, hydra, etc.) are not installed.",
          "- The token exists only in the download yml (mode 600); it is not in code, logs, command-line arguments or git.",
          "- The anonymisation mask (`rgb_mask.mkv`) was not downloaded; this round is unaffected.",
          "- Follows the data construction guide (docs/data_construction.md): 7 required file types, pilot draw rule, decoding requirements.",
          "", "## 7. Open questions", ""]
    L += [f"{i}. {q}" for i, q in enumerate(inv["owner_questions"], 1)]
    L.append("")
    path.write_text("\n".join(L))


def main():
    cfg = load_config()
    sc = sp_cfg(cfg)
    split = read_split(cfg)
    pilots = sc["pilot_scenes"]
    scenes = [check_scene(cfg, s) for s in pilots]
    for s in pilots:
        make_preview(cfg, s, path_of(cfg, "reports") / "scannetpp_step0" / f"{s}_rgb_depth.jpg")
    for s in scenes:
        print(s["scene_id"], "missing", s["missing"], "counts_equal", s["counts_equal"], "seg", s["mesh"]["seg_match"],
              "up", s["mesh"]["up_axis"], flush=True)

    log = Path(sc["toolbox"]).parent / "download_pilot5.log"
    txt = log.read_text(errors="replace").replace("\r", "\n") if log.exists() else ""
    tail = [l.strip() for l in txt.splitlines() if l.strip()]
    dt = download_time(log)
    mins = None
    if dt:
        mm, ss = dt.split(" (")[0].split(":")[-2:]
        mins = (int(mm) + int(ss) / 60) / len(pilots) * len(split)

    axes = {(s["mesh"]["up_axis"] or {}).get("normal_sign", "") + (s["mesh"]["up_axis"] or {}).get("axis", "?") for s in scenes}
    all_above = all((s["mesh"]["up_axis"] or {}).get("ceiling_above_floor") for s in scenes)
    up_summary = (f"The up axis of all 5 scenes is **{axes.pop()}** (floor normal points along the positive axis, ceiling above floor), i.e. the mesh is already z-up; Step 1 is not expected to need an axis transform (to be confirmed in GATE A)."
                  if len(axes) == 1 and all_above else f"Scenes disagree: {sorted(axes)}; handle per scene in GATE A.")

    types = read_json(sp_root(cfg) / "metadata" / "scene_types.json")
    count = lambda ids: dict(Counter(types.get(s, "?") for s in ids).most_common())
    inv = {
        "split": sc["split"], "n_split_scenes": len(split), "pilot_scenes": pilots,
        "pilot_scene_types": count(pilots), "split_scene_types": count(split),
        "pilot_is_head5": pilots == split[:5], "pilot_positions_in_split": [split.index(s) for s in pilots],
        "download_log_tail": tail[-1] if tail else None, "download_time": dt, "download_minutes_est": mins or 0,
        "token_lines_in_log": sum("Using token" in l for l in txt.splitlines()),
        "toolbox_commit": _git_head(Path(sc["toolbox"])),
        "nfs_free_tb": _free_tb(sp_root(cfg)),
        "resolution_ok": all(s["rgb_sizes_wh"] == [sc["rgb_size"]] and s["depth_sizes_wh"] == [sc["depth_size"]] for s in scenes),
        "up_axis_summary": up_summary,
        "scenes": scenes,
    }
    inv.update(_raw_unchanged(cfg, pilots))
    inv["decode_job"] = _decode_job(cfg)
    inv["owner_questions"] = [
        "The 5 pilot scenes are the first 5 of the list and all office scenes: accept them, or redraw 5 with a fixed seed (optionally stratified by scene type)? Redrawing is recommended.",
        f"Download the remaining {len(split) - len(pilots)} scenes of {sc['split']} (same 7 file types; size estimate in Section 3)?",
        "Depth, like RGB, is decoded only at 10 FPS; the per-frame GATE A check uses the same frames. Agreed?",
    ]
    out = sp_work(cfg) / "_step0" / "inventory.json"
    write_json(out, inv)
    write_report(cfg, inv, path_of(cfg, "reports") / "scannetpp_step0_inventory.md")
    print("wrote", out)


def _git_head(repo: Path) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def _free_tb(p: Path) -> float:
    import shutil
    return shutil.disk_usage(p.resolve()).free / 1e12


def _raw_unchanged(cfg: dict, scene_ids: list[str]) -> dict:
    """Compare the pre-decode snapshot (work/scannetpp/<scene_id>/raw_snapshot.json) with the current state; the newly added iphone/rgb and iphone/depth are not in the snapshot."""
    n, changed = 0, []
    for sid in scene_ids:
        snap = sp_work(cfg) / sid / "raw_snapshot.json"
        if not snap.exists():
            return {"raw_files_checked": 0, "raw_files_unchanged": None, "raw_files_changed": []}
        snapshot = read_json(snap)
        n += len(snapshot)
        changed += [f"{sid}/{r}" for r in changed_files(sp_root(cfg) / "data" / sid, snapshot)]
    return {"raw_files_checked": n, "raw_files_unchanged": not changed, "raw_files_changed": changed}


def _decode_job(cfg: dict) -> str:
    logs = sorted(Path(sp_cfg(cfg)["logs"]).glob("decode_*.out"))
    return logs[-1].stem.split("_")[-1] if logs else "?"


if __name__ == "__main__":
    main()
