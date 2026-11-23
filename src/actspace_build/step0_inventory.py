"""Step 0: ARKitScenes scene inventory, visit_id deduplication, vga_wide download size estimate.

Usage: python -m actspace_build.step0_inventory [--no-estimate]
Outputs: work/arkitscenes/_step0/inventory.json, selected_videos.csv, reports/step0_inventory.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import arkit
from .common import load_config, path_of, write_json, read_json

REQUIRED_3DOD = ["lowres_wide", "lowres_wide_intrinsics", "lowres_wide.traj", "lowres_depth", "mesh", "annotation"]


def inspect_video(root, vid: str) -> dict:
    files = arkit.scene_files_3dod(root, vid)
    missing = [k for k in REQUIRED_3DOD if not files[k].exists()]
    n_frames = len(arkit.list_frames(files["lowres_wide"], ".png")) if "lowres_wide" not in missing else 0
    n_objects = 0
    if "annotation" not in missing:
        with open(files["annotation"]) as f:
            n_objects = len(json.load(f)["data"])
    return {"video_id": vid, "missing": missing, "n_frames": n_frames, "n_objects": n_objects}


def remote_size(url: str) -> int | None:
    """Get Content-Length (bytes) with a HEAD request."""
    try:
        out = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    size = None
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":", 1)[1])
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-estimate", action="store_true", help="skip the remote vga_wide size query")
    args = ap.parse_args()
    cfg = load_config()
    root = path_of(cfg, "arkit_3dod_root")
    split = cfg["paths"]["arkit_split_name"]
    out_dir = path_of(cfg, "work") / "arkitscenes" / "_step0"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_of(cfg, "arkit_split_csv"))
    df = df[df["fold"] == split].copy()
    df["video_id"] = df["video_id"].astype(str)
    meta = pd.read_csv(path_of(cfg, "arkit_metadata_csv"))
    meta["video_id"] = meta["video_id"].astype(str)
    df = df.merge(meta[["video_id", "sky_direction"]], on="video_id", how="left")

    with ThreadPoolExecutor(16) as ex:
        info = list(ex.map(lambda v: inspect_video(root, v), df["video_id"]))
    inv = df.merge(pd.DataFrame(info), on="video_id")
    inv["complete"] = inv["missing"].map(len) == 0

    # Deduplication: per visit_id keep the video with the most frames, then the most annotated objects; remaining ties go to the smallest video_id
    cand = inv[inv["complete"]].copy()
    cand["vid_int"] = cand["video_id"].astype(int)
    cand = cand.sort_values(["visit_id", "n_frames", "n_objects", "vid_int"], ascending=[True, False, False, True])
    selected = cand.drop_duplicates("visit_id", keep="first").sort_values("vid_int")
    visits_all = inv["visit_id"].nunique()
    visits_lost = sorted(set(inv["visit_id"]) - set(selected["visit_id"]))

    # vga_wide download size estimate
    sizes = {}
    est_path = out_dir / "vga_size_estimate.json"
    if not args.no_estimate:
        base = cfg["paths"]["arkit_raw_url"]
        jobs = [(v, a) for v in selected["video_id"] for a in cfg["arkitscenes"]["vga_assets"]]
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(lambda va: remote_size(f"{base}/{split}/{va[0]}/{va[1]}.zip"), jobs))
        for (v, a), s in zip(jobs, res):
            sizes.setdefault(v, {})[a] = s
        write_json(est_path, sizes)
    elif est_path.exists():
        sizes = read_json(est_path)

    selected[["video_id", "visit_id", "sky_direction", "n_frames", "n_objects"]].to_csv(out_dir / "selected_videos.csv", index=False)
    write_json(out_dir / "inventory.json", {
        "split": split, "n_videos": len(inv), "n_complete": int(inv["complete"].sum()),
        "n_visits": int(visits_all), "n_selected": len(selected), "visits_without_complete_video": visits_lost,
        "videos": inv.drop(columns=[]).to_dict(orient="records"),
    })
    write_report(cfg, inv, selected, visits_all, visits_lost, sizes)


def write_report(cfg, inv, selected, visits_all, visits_lost, sizes):
    gb = 1024 ** 3
    lines = ["# Step 0 inventory report (ARKitScenes)", "",
             f"- Split: `{cfg['paths']['arkit_split_name']}` (3DOD subset)",
             f"- Total videos: {len(inv)}; with all required files: {int(inv['complete'].sum())}",
             f"- Capture sites (visit_id): {visits_all}; videos selected after deduplication: {len(selected)}",
             "- Deduplication rule: within a visit_id, take the first by lowres_wide frame count descending, annotated object count descending, video_id ascending", ""]
    miss = inv[~inv["complete"]]
    lines += ["## Missing files", ""]
    if len(miss):
        lines += ["| video_id | Missing |", "|---|---|"] + [f"| {r.video_id} | {', '.join(r.missing)} |" for r in miss.itertuples()]
    else:
        lines += ["None."]
    if visits_lost:
        lines += ["", f"visit_ids without a complete video ({len(visits_lost)}): {visits_lost}"]
    lines += ["", "## Selected video statistics", "",
              f"- Frames: median {selected['n_frames'].median():.0f}, range {selected['n_frames'].min()}–{selected['n_frames'].max()}",
              f"- Annotated objects: median {selected['n_objects'].median():.0f}, range {selected['n_objects'].min()}–{selected['n_objects'].max()}",
              f"- sky_direction distribution: {selected['sky_direction'].value_counts().to_dict()}", ""]
    lines += ["## vga_wide download size estimate", ""]
    if sizes:
        per = {a: sum((s.get(a) or 0) for s in sizes.values()) for a in cfg["arkitscenes"]["vga_assets"]}
        unknown = [v for v, s in sizes.items() if any(x is None for x in s.values())]
        total = sum(per.values())
        vw = [s.get("vga_wide") or 0 for s in sizes.values()]
        lines += [f"- Videos: {len(sizes)} (Content-Length queried with HEAD requests)",
                  *[f"- `{a}.zip`: {per[a] / gb:.1f} GB" for a in per],
                  f"- **Total: {total / gb:.1f} GB** (per-video vga_wide median {pd.Series(vw).median() / gb:.2f} GB, max {max(vw) / gb:.2f} GB)",
                  f"- Peak extraction space: each video's zip is extracted right after download and then deleted, so the extra peak is about the largest single zip",
                  f"- Failed queries: {unknown if unknown else 'none'}",
                  f"- Cap `vga_download_cap_gb`: {cfg['arkitscenes']['vga_download_cap_gb']}"]
        cap = cfg["arkitscenes"]["vga_download_cap_gb"]
        if cap is None:
            lines += ["- **Status: no cap set yet; download not started.**"]
        elif total / gb > cap:
            lines += ["- **Status: cap exceeded; stopped for review.**"]
        else:
            lines += ["- Status: within the cap; download can proceed."]
    else:
        lines += ["Not estimated."]
    path_of(cfg, "reports").mkdir(exist_ok=True)
    (path_of(cfg, "reports") / "step0_inventory.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
