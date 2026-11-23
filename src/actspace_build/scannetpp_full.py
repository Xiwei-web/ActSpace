"""ScanNet++ full run (all nvs_sem_val scenes): scene-level GATE A summary and the 50% stop rule. Reports and spot checks: scannetpp_full_report.

Usage: python -m actspace_build.scannetpp_full gate_a_pass     # writes work/scannetpp/_full/gate_a.json and passed_scenes.txt; exit code 3 if < 50% pass
"""
from __future__ import annotations

import argparse
import sys

from .common import load_config, read_json, write_json
from .scannetpp import read_split, sp_cfg, sp_work

STOP_IF_PASS_FRAC_BELOW = 0.5  # stop and report if fewer than 50% of scenes pass GATE A


def gate_a_result(cfg: dict, sid: str) -> dict:
    """Scene-level GATE A: pose / orientation / up axis / resolution in scene_meta all pass. The pose item uses the per-pose rule
    (bad poses <= 20% of verified poses and >= 40 keyframes).
    Per GATE A ("these 10 frames do not decide whether a scene is kept"), the pose-convention sample only determines the convention
    and does not decide scene retention (reported only)."""
    wd = sp_work(cfg) / sid
    if not (wd / "gate_a.json").exists():
        return {"scene_id": sid, "passed": False, "reasons": ["NOT_RUN"]}
    ga = read_json(wd / "gate_a.json")
    st = read_json(wd / "scene_meta.json")["status"]
    reasons = [k for k, v in st.items() if v is False]
    pc = ga["pose_check"]
    return {"scene_id": sid, "passed": not reasons, "reasons": reasons, "best_convention": ga["best_convention"],
            "convention_pass_ratio": ga["conventions"][ga["best_convention"]]["pass_ratio"],
            "n_decoded_frames": ga["n_decoded_frames"], "n_frames_dropped": ga["n_frames_dropped"], "num_keyframes_est": ga["num_keyframes_est"],
            "n_good": pc["n_good"], "n_bad": pc["n_bad"], "n_unverified": pc["n_unverified"], "bad_frac_of_verified": pc["bad_frac_of_verified"],
            "up_axis": ga["up_axis_detected"], "sky_agreement": ga["sky_agreement"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate_a_pass"])
    args = ap.parse_args()
    cfg = load_config()
    scenes = read_split(cfg)
    res = [gate_a_result(cfg, s) for s in scenes]
    passed = [r["scene_id"] for r in res if r["passed"]]
    out = sp_work(cfg) / "_full"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "gate_a.json", {"scenes": res, "num_scenes": len(scenes), "num_passed": len(passed)})
    (out / "passed_scenes.txt").write_text("\n".join(passed) + "\n")
    print(f"GATE A: {len(passed)} / {len(scenes)} passed", [(r["scene_id"], r["reasons"]) for r in res if not r["passed"]], flush=True)
    if len(passed) < STOP_IF_PASS_FRAC_BELOW * len(scenes):
        print(f"STOP: scenes passing GATE A {len(passed)} / {len(scenes)} < {STOP_IF_PASS_FRAC_BELOW:.0%}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
