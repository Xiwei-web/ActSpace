"""Step 10: packaging of the merged release (release format described in docs/data_construction.md).

  release  build <merge.release_root>/<version>/ from the frozen merged_v1.0:
           build_info.json, scene_list.txt, SHA256SUMS, scenes/<scene_id>/{scene_meta,objects,frames,view_graph}.json + visibility.npz,
           questions/test_groups.jsonl, scripts/rebuild_frames.py
  check    release checks: no raw images / videos / depth, tokens or personal paths; files match SHA256SUMS; questions identical
           to the frozen version
  rebuild-test  rebuild a few scenes in a clean directory with scripts/rebuild_frames.py (ARKit including upright rotation,
           ScanNet++ decoded from video) and compare pixel by pixel with the images used by the pipeline

Before packaging, all freeze verifications are run (merged_v1.0 and the two source versions); packaging is refused on any mismatch.
Usage: python -m actspace_build.merge_release release|check|rebuild-test
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from .common import ROOT, path_of, read_json, write_json
from .merge_build import CONFIG_FILES, git, vdir, verify_all
from .merge_dryrun import TAGS, dataset_of_scene, ds_of, load_merge_config, read_groups, real_work, sandbox, sha256, version_dir

VIS_KEYS = ("obj_ids", "frame_ids", "state", "rho", "completeness", "area_frac", "in_frame_frac", "n_infrustum", "n_visible")
RAW_ROOT_NAME = {"arkitscenes": "ARKitScenes data root (contains raw/Validation/<video_id>/vga_wide/)",
                 "scannetpp": "ScanNet++ data root (contains data/<scene_id>/iphone/rgb.mkv)"}
# content that must not appear in the release (personal paths, credentials)
FORBIDDEN = [r"/home/", r"/Users/", r"/mnt/", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_", r"sk-[A-Za-z0-9_-]{20,}",
             r"OPENAI_API_KEY", r"token="]
FORBIDDEN += [x for x in os.environ.get("ACTSPACE_FORBIDDEN_PATTERNS", "").split("|") if x]   # machine-specific paths or user names (optional)
FORBIDDEN_SUFFIX = (".jpg", ".jpeg", ".png", ".mkv", ".mp4", ".bin", ".ply", ".zip")


def release_dir(cfg) -> Path:
    return Path(cfg["merge"]["release_root"]) / cfg["merge"]["version"]


def scene_src(cfg, sid) -> Path:
    return real_work(cfg) / dataset_of_scene(sid) / sid.split("_", 1)[1]


def video_frame_index(source_frame: str) -> int | None:
    m = re.fullmatch(r"frame_(\d+)", source_frame)
    return int(m.group(1)) if m else None


def export_scene(cfg, sid: str, out: Path) -> dict:
    ds = dataset_of_scene(sid)
    src = scene_src(cfg, sid)
    meta = read_json(src / "scene_meta.json")
    objs = read_json(src / "objects.json")
    graph = read_json(src / "view_graph.json")
    kf = read_json(src / "keyframes.json")["keyframe_ids"]
    z = np.load(src / "visibility.npz")
    nodes = set(graph["nodes"])
    col = {str(n): j for j, n in enumerate(z["frame_ids"])}
    row = {int(o): i for i, o in enumerate(z["obj_ids"])}
    out.mkdir(parents=True, exist_ok=True)
    frames_all = {f["frame_id"]: f for f in read_json(src / "frames_all.json")}
    frames = []
    for fid in kf:
        f = frames_all[fid]
        r = {"frame_id": fid, "source_frame": f["source_frame"], "rel_path": f["rel_path"], "timestamp": f["timestamp"],
             "rotation_cw_quarters": f["rotation_cw_quarters"], "T_wc": f["T_wc"], "K": f["K"], "image_size": f["image_size"],
             "position": f["position"], "yaw_deg": f["yaw_deg"],
             "has_sensor_depth": bool(z["has_sensor_depth"][col[fid]]) if fid in col else (True if ds == "arkitscenes" else "depth_rel_path" in f),
             "in_graph": fid in nodes}
        if ds == "scannetpp":   # ScanNet++ frames come from iphone/rgb.mkv: rel_path is the decoded jpg named as by the toolbox; the video and frame index are also given so frames can be rebuilt directly from the video
            r["source_video"] = f"data/{sid.split('_', 1)[1]}/iphone/rgb.mkv"
            r["video_frame_index"] = video_frame_index(f["source_frame"])
        frames.append(r)
    write_json(out / "frames.json", frames)
    o_out = []
    for o in objs:
        i = row.get(int(o["obj_id"]))
        black = bool(o.get("blacklisted")) or "BLACKLISTED" in (o.get("drop_reasons") or [])
        o_out.append({"obj_id": o["obj_id"], "category": o.get("category_merged") or o["category"], "source_label": o["source_label"],
                      "referable": o["referable"], "ref_kind": o.get("ref_kind"), "ref_name": o.get("ref_name"), "blacklisted": black,
                      "center": o["center"], "obb": {k: o["obb"][k] for k in ("center", "axes", "size", "yaw_deg")},
                      "num_points": o["num_points"],
                      "num_visible_nodes": int((z["state"][i] == 2).sum()) if i is not None else 0})
    write_json(out / "objects.json", o_out)
    write_json(out / "view_graph.json", {"nodes": graph["nodes"], "edges": graph["edges"]})
    np.savez_compressed(out / "visibility.npz", **{k: z[k] for k in VIS_KEYS})
    sm = {"scene_id": sid, "dataset": ds, "source_id": meta["source_id"], "split": meta["split"], "up_axis": meta["up_axis"],
          "units": meta["units"], "pose_convention": meta["pose_convention"], "image_size": meta["image_size"],
          "frame_source": meta.get("frame_source"), "rotation_cw_quarters": meta["rotation_cw_quarters"],
          "num_keyframes": len(kf), "num_graph_nodes": len(graph["nodes"]),
          "num_referable_objects": sum(bool(o["referable"]) for o in objs)}
    write_json(out / "scene_meta.json", sm)
    return sm


def release(cfg) -> dict:
    if not verify_all(cfg):
        raise SystemExit("freeze verify failed; not packaging")
    V = cfg["merge"]["version"]
    vd = vdir(cfg)
    fm = read_json(vd / "freeze_manifest.json")
    man = read_json(vd / "manifest.json")
    rd = release_dir(cfg)
    if rd.exists():
        shutil.rmtree(rd)
    (rd / "questions").mkdir(parents=True)
    shutil.copyfile(vd / "test_groups.jsonl", rd / "questions" / "test_groups.jsonl")
    groups = read_groups(vd / "test_groups.jsonl")
    sids = sorted({g["scene_id"] for g in groups})
    metas = {sid: export_scene(cfg, sid, rd / "scenes" / sid) for sid in sids}
    (rd / "scene_list.txt").write_text("\n".join(sids) + "\n")
    (rd / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts" / "rebuild_frames.py", rd / "scripts" / "rebuild_frames.py")
    qs = [(g, q) for g in groups for q in g["questions"]]
    st = man["stages"]
    by = lambda rs: dict(sorted(Counter(f"{r['qid'].split('_', 1)[0]}|{r['template']}|{r['reason']}" for r in rs).items()))
    bi = {"name": "ActSpace-Bench", "version": V, "label": "release candidate (pending final acceptance, GATE E)", "created": date.today().isoformat(),
          "code": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "commit": git("rev-parse", "HEAD"), "commit_at_freeze": fm["git"]["head_at_freeze"]},
          "freeze_manifest_sha256": sha256(vd / "freeze_manifest.json"),
          "sources": {ds: {"version": cfg["merge"]["sources"][ds]["version"],
                           "freeze_manifest_sha256": sha256(version_dir(cfg, ds) / "freeze_manifest.json")} for ds in TAGS},
          "configs_sha256": {n: sha256(ROOT / n) for n in CONFIG_FILES},
          "rewrite": {k: v for k, v in read_json(ROOT / "build_info.json")["step7_rewrite"].items()
                      if k in ("method", "author_model", "external_api", "paraphrase_file", "n_templates", "paraphrases_per_template", "selection")}
                     | {"paraphrase_file_sha256": sha256(ROOT / "configs" / "paraphrases.yaml"), "shared_by": "shared by ARKitScenes and ScanNet++"},
          "datasets": {"arkitscenes": {"raw_root": RAW_ROOT_NAME["arkitscenes"], "frames": "vga_wide 640×480 (upright)",
                                       "visibility": {"visible_min_ratio": cfg["visibility"]["visible_min_ratio"]}},
                       "scannetpp": {"raw_root": RAW_ROOT_NAME["scannetpp"], "split": cfg["scannetpp"]["split"],
                                     "frames": "iphone rgb.mkv sampled at 10 FPS, 1920×1440; downscaled to 640×480 for evaluation and display; visibility computed at the original resolution",
                                     "visibility": {k: cfg["scannetpp"][k] for k in ("visible_min_ratio", "vis_min_short_side_px",
                                                    "vis_min_sharpness_ratio", "vis_max_magenta_frac", "motion_max_ang_deg_s",
                                                    "pose_err_ratio_max", "pose_err_min_lin_m_s")}},
                       "shared_visibility": {k: cfg["visibility"][k] for k in ("visible_min_area_frac", "visible_min_points",
                                             "visible_min_completeness", "visible_min_in_frame_frac", "invisible_max_ratio")}},
          "steps": {"premerge": st["premerge"],
                    "standing_point_separation": {"min_horizontal_sep_m": man["min_horizontal_sep_m"], "removed": by(st["separation"])},
                    "step8": {"version": st["step8"]["version"], "status": st["step8"]["status"], "rounds": st["step8"]["rounds"],
                              "removed": by(st["step8"]["removed"])},
                    "quadrant_rebalance": {"removed": by(st["quadrant"])},
                    "blind_after": man["blind_after"], "stability_after": man["stability_after"]},
          "counts": {"scenes": dict(Counter(m["dataset"] for m in metas.values())), "groups": dict(Counter(ds_of(g) for g in groups)),
                     "questions": len(qs),
                     "by_dataset_template_label": dict(sorted(Counter(f"{ds_of(g)}|{q['template']}|{q['evidence_label']}" for g, q in qs).items()))},
          "frames": "no images; rebuild them from the raw data with scripts/rebuild_frames.py using scenes/<scene_id>/frames.json (including the upright rotation)"}
    write_json(rd / "build_info.json", bi)
    sums = [f"{sha256(p)}  {p.relative_to(rd)}" for p in sorted(rd.rglob("*")) if p.is_file()]
    (rd / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"release {rd}: {len(sids)} scenes, {len(groups)} groups, {len(qs)} questions, {len(sums)} files")
    return {"release_dir": str(rd), "scenes": len(sids), "files": len(sums), "counts": bi["counts"]}


def check(cfg) -> dict:
    rd = release_dir(cfg)
    files = [p for p in sorted(rd.rglob("*")) if p.is_file()]
    res = {}
    res["no_image_video_depth_mesh_archive_files"] = [str(p.relative_to(rd)) for p in files if p.suffix.lower() in FORBIDDEN_SUFFIX]
    hits = []
    for p in files:
        data = p.read_bytes()
        if p.suffix == ".npz":   # array file: unpack and check the string arrays
            z = np.load(p)
            data = " ".join(" ".join(map(str, z[k].tolist())) for k in z.files if z[k].dtype.kind in "US").encode()
        txt = data.decode("utf-8", errors="ignore")
        for pat in FORBIDDEN:
            if re.search(pat, txt, re.I):
                hits.append(f"{p.relative_to(rd)}: {pat}")
    res["no_personal_paths_or_tokens"] = hits
    sums = dict(reversed(l.split("  ", 1)) for l in (rd / "SHA256SUMS").read_text().splitlines())
    res["sha256sums_mismatch"] = [n for n, h in sums.items() if sha256(rd / n) != h]
    res["files_not_in_sha256sums"] = [str(p.relative_to(rd)) for p in files if str(p.relative_to(rd)) not in sums and p.name != "SHA256SUMS"]
    res["questions_identical_to_frozen"] = (rd / "questions" / "test_groups.jsonl").read_bytes() == (vdir(cfg) / "test_groups.jsonl").read_bytes()
    groups = read_groups(rd / "questions" / "test_groups.jsonl")
    sids = (rd / "scene_list.txt").read_text().split()
    res["scene_list_matches_questions"] = sorted({g["scene_id"] for g in groups}) == sids
    missing = []
    for g in groups:   # every node referenced by a question is in the released frames.json (and is a view-graph node)
        fr = {f["frame_id"]: f for f in read_json(rd / "scenes" / g["scene_id"] / "frames.json")}
        need = {g["init_frame"]} | {n for q in g["questions"] for k in ("oracle_nodes", "link_views", "path_views", "masked_nodes")
                                    for n in (q.get(k) or [])}
        missing += [f"{g['group_id']}:{n}" for n in need if n not in fr or not fr[n]["in_graph"]]
    res["referenced_nodes_missing_from_frames"] = missing
    res["all_pass"] = (not res["no_image_video_depth_mesh_archive_files"] and not hits and not res["sha256sums_mismatch"]
                       and not res["files_not_in_sha256sums"] and res["questions_identical_to_frozen"]
                       and res["scene_list_matches_questions"] and not missing)
    res["n_files"] = len(files)
    res["bytes"] = sum(p.stat().st_size for p in files)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def rebuild_test(cfg, scenes: list[str] | None = None) -> dict:
    """Rebuild a few scenes in a clean directory and compare pixel by pixel with the images used by the pipeline. ARKit: raw png
    rotated upright (same as the evaluation's data.SceneData.image); ScanNet++: decoded from rgb.mkv (same decoding command as the
    pipeline) and compared with the jpgs decoded by the pipeline."""
    import cv2
    from .geometry import rotate_image_cw
    rd = release_dir(cfg)
    sids = (rd / "scene_list.txt").read_text().split()
    if not scenes:
        rot = {s: read_json(rd / "scenes" / s / "scene_meta.json")["rotation_cw_quarters"] for s in sids}
        ark = [s for s in sids if s.startswith("arkit_")]
        scenes = [next(s for s in ark if rot[s] == 0), next(s for s in ark if rot[s] == 1), next(s for s in ark if rot[s] == 2)] + \
                 [s for s in sids if s.startswith("scannetpp_")][:2]
    out = sandbox(cfg) / "rebuild_test"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    arkit_root = (ROOT / cfg["paths"]["arkit_raw_root"]).resolve().parent.parent   # rel_path looks like raw/Validation/<video_id>/vga_wide/...
    sp_root = (ROOT / cfg["scannetpp"]["root"]).resolve()
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [sys.executable, str(rd / "scripts" / "rebuild_frames.py"), "--release", str(rd), "--out", str(out / "frames"),
           "--arkit-root", str(arkit_root), "--scannetpp-root", str(sp_root), "--ffmpeg", ffmpeg, "--scannetpp-from-video", "--scannetpp-size", "640x480",
           "--scenes", *scenes]
    r = subprocess.run(cmd, capture_output=True, text=True)
    res = {"scenes": scenes, "returncode": r.returncode, "stdout_tail": r.stdout[-1500:], "stderr_tail": r.stderr[-1500:], "per_scene": {}}
    for sid in scenes:
        fr = [f for f in read_json(rd / "scenes" / sid / "frames.json") if f["in_graph"]]
        diffs, missing, shape_bad = [], 0, 0
        for f in fr:
            p = next(iter((out / "frames" / sid).glob(f"{f['frame_id']}.*")), None)
            if p is None:
                missing += 1
                continue
            got = cv2.imread(str(p))
            if sid.startswith("arkit_"):
                ref = rotate_image_cw(cv2.imread(str(arkit_root / f["rel_path"])), f["rotation_cw_quarters"])
            else:
                ref = rotate_image_cw(cv2.imread(str(sp_root / f["rel_path"])), f["rotation_cw_quarters"])
            if got.shape != ref.shape or list(got.shape[1::-1]) != list(f["image_size"]):
                shape_bad += 1
                continue
            diffs.append(int(np.abs(got.astype(int) - ref.astype(int)).max()))
        res["per_scene"][sid] = {"graph_frames": len(fr), "rebuilt": len(fr) - missing, "missing": missing, "shape_mismatch": shape_bad,
                                 "max_abs_pixel_diff": max(diffs) if diffs else None,
                                 "identical_frames": sum(d == 0 for d in diffs),
                                 "rotation_cw_quarters": read_json(rd / "scenes" / sid / "scene_meta.json")["rotation_cw_quarters"]}
    res["all_pass"] = r.returncode == 0 and all(v["missing"] == 0 and v["shape_mismatch"] == 0 and v["max_abs_pixel_diff"] == 0
                                               for v in res["per_scene"].values())
    print(json.dumps({k: v for k, v in res.items() if k not in ("stdout_tail", "stderr_tail")}, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["release", "check", "rebuild-test"])
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    cfg = load_merge_config()
    r = {"release": release, "check": check}.get(args.cmd, lambda c: rebuild_test(c, args.scenes))(cfg)
    write_json(sandbox(cfg) / "results" / f"release_{args.cmd.replace('-', '_')}.json", r)
    if args.cmd != "release" and not r["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
