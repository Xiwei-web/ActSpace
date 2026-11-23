"""Section 5.3 supplement: paired difference between Active and "a random walk on the same group truncated to the same number of steps",
with scene-bootstrap intervals. Reads only the per-episode CSV written by sec53_behavior.

    python -m actspace_eval.sec53_matched_ci [--dir results/_reports/sec53]

Per-episode difference = whether Active saw all evidence (0 / 1) − whether the same group's random walk (seed 0), truncated to the same
number of steps, saw all evidence (0 / 1); computed for both criteria (at least partial, strictly visible).
Interval: score.RepBoot, 2000 scene resamples with replacement (stratified by dataset for the merged scope), 95% percentile interval;
for models with several Active runs (haiku-5.5, 3 runs) runs are resampled as well.
All episodes are used; truncated or context-overflow episodes are not removed. Scene ids come from the frozen data's test_groups (group → scene).
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from .data import FrozenData, results_root
from .score import N_BOOT, RepBoot

SCOPES = ((None, "Merged"), ("arkitscenes", "ARKitScenes"), ("scannetpp", "ScanNet++"))
KINDS = (("partial", "all_partial", "rw_same_steps_all_partial"), ("visible", "all_visible", "rw_same_steps_all_visible"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(results_root() / "_reports" / "sec53"))
    a = ap.parse_args()
    d = Path(a.dir)
    assert N_BOOT == 2000
    scene_of = {g: x["scene_id"] for g, x in FrozenData(verify=False).groups.items()}
    rows = list(csv.DictReader(open(d / "active_behavior_per_episode.csv")))
    models = list(dict.fromkeys(r["model"] for r in rows))
    scenes = collections.defaultdict(set)
    for r in rows:
        scenes[r["dataset"]].add(scene_of[r["group_id"]])
    out = []
    for scope, lab in SCOPES:
        b = RepBoot(scenes, scope=scope)
        for m in models:
            xs = [r for r in rows if r["model"] == m and (scope is None or r["dataset"] == scope)]
            runs = list(dict.fromkeys(r["run"] for r in xs))
            for kind, ka, kr in KINDS:
                it = lambda key, f: [[(scene_of[r["group_id"]], f(r)) for r in xs if r["run"] == run] for run in runs]     # noqa: E731
                act = b.mean_ci(it("a", lambda r: float(r[ka])), f"{m}|{kind}")
                rw = b.mean_ci(it("r", lambda r: float(r[kr])), f"{m}|{kind}")
                diff = b.mean_ci(it("d", lambda r: float(r[ka]) - float(r[kr])), f"{m}|{kind}")
                out.append({"scope": lab, "model": m, "kind": xs[0]["kind"], "evidence": kind, "n_episodes": len(xs), "n_runs": len(runs),
                            "n_scenes": len({scene_of[r["group_id"]] for r in xs}),
                            "active": round(act["mean"], 6), "rw_same_steps": round(rw["mean"], 6), "diff": round(diff["mean"], 6),
                            "diff_lo": round(diff["lo"], 6), "diff_hi": round(diff["hi"], 6), "excludes_zero": int(diff["lo"] > 0 or diff["hi"] < 0),
                            "n_active_only": sum(float(r[ka]) > float(r[kr]) for r in xs), "n_rw_only": sum(float(r[ka]) < float(r[kr]) for r in xs)})
    with open(d / "active_vs_rw_matched_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {d / 'active_vs_rw_matched_ci.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
