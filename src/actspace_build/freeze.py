"""Freeze a data version (after GATE D round 3 part A passed, arkit_only_v0.11 was marked as the ARKit final candidate and frozen).

  freeze   write freeze_manifest.json in work/arkitscenes/_step8/<version>/: group list, question counts, SHA-256 of every file in that
           directory and of every per-scene dependency, and the git commits; then make the directory and its files read-only. Step 8 and
           gate_d carryover refuse to write to a frozen version. Also writes reports/<version>_freeze.md (with the suggested blacklist
           additions; the blacklist and data are not modified).
  verify   recompute the SHA-256 hashes and compare them one by one with freeze_manifest.json (run before Step 10 packaging).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

from .common import ROOT, load_config, path_of, read_json, work_dir, write_json

DATASET = "arkitscenes"
# Per-scene intermediate outputs used by the questions, the review pages and Step 10 packaging (log files such as dropped.jsonl, groups_raw.json are not included)
SCENE_FILES = ("gate_a.json", "scene_meta.json", "frames_all.json", "keyframes.json", "objects.json", "object_points.npz",
               "view_graph.json", "visibility.npz")
CONFIG_FILES = ("configs/build.yaml", "configs/object_blacklist.yaml", "configs/paraphrases.yaml")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip("\n")


def version_dir(cfg, version) -> Path:
    return path_of(cfg, "work") / DATASET / "_step8" / version


def hashes(cfg, version, groups):
    vd = version_dir(cfg, version)
    files = {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in sorted(vd.iterdir())
             if f.is_file() and f.name != "freeze_manifest.json"}
    scenes = {}
    for sid in sorted({g_["scene_id"] for g_ in groups}):
        wd = work_dir(cfg, DATASET, sid.split("_", 1)[1])
        scenes[sid] = {n: sha256(wd / n) for n in SCENE_FILES if (wd / n).exists()}
    configs = {n: sha256(ROOT / n) for n in CONFIG_FILES}
    return files, scenes, configs


def gate_d_status(cfg, version, groups):
    """Human review status per group: results carried over from earlier rounds by question matching (gateD_carryover.json) + review rounds run directly on this version."""
    from . import gate_d
    co = read_json(version_dir(cfg, version) / "gateD_carryover.json")["groups"]
    direct = {}
    if version in gate_d.ROUND_DIRS:
        rd = gate_d.round_dir(cfg, version)
        for f in sorted((rd / "annotations").glob("*.json")):
            r = read_json(f)
            direct[r["group_id"]] = {qid: v.get("verdict") for qid, v in (r.get("review") or {}).items()}
    out = {}
    for g_ in groups:
        gid, qids = g_["group_id"], [q["qid"] for q in g_["questions"]]
        prev = co[gid]["questions"]
        now = direct.get(gid, {})
        n = sum(q in prev or q in now for q in qids)
        out[gid] = {"status": "verified" if n == len(qids) else "partial" if n else "unverified", "n_verified": n,
                    "carried_over": len(prev), "verified_in_this_version": len(now)}
    return out


def freeze(cfg, version, label, build_commits):
    vd = version_dir(cfg, version)
    if (vd / "freeze_manifest.json").exists():
        raise SystemExit(f"{version} is already frozen")
    groups = [json.loads(l) for l in open(vd / "test_groups.jsonl")]
    qs = [q for g_ in groups for q in g_["questions"]]
    files, scenes, configs = hashes(cfg, version, groups)
    gd = gate_d_status(cfg, version, groups)
    dirty = [l[3:] for l in git("status", "--porcelain").splitlines() if l]
    m = {"version": version, "label": label, "frozen": True, "frozen_on": date.today().isoformat(),
         "decided_by": "after GATE D (human verification) passed",
         "rule": "the frozen data is never modified; later changes (e.g. blacklist suggestions) are applied when Step 8 is rerun on the merged data, producing a new version",
         "git": {"build_commits": build_commits, "head_at_freeze": git("rev-parse", "HEAD"),
                 "uncommitted_at_freeze": dirty,
                 "remote": None},
         "counts": {"scenes": len({g_["scene_id"] for g_ in groups}), "groups": len(groups), "questions": len(qs),
                    "by_template": dict(sorted(Counter(q["template"] for q in qs).items())),
                    "unanswerable": sum(q["evidence_label"] == "unanswerable" for q in qs),
                    "unanswerable_by_template": dict(sorted(Counter(q["template"] for q in qs if q["evidence_label"] == "unanswerable").items())),
                    "groups_without_constraint": sum(not g_["constraints"] for g_ in groups),
                    "gate_d": dict(Counter(v["status"] for v in gd.values())),
                    "questions_human_verified": sum(v["n_verified"] for v in gd.values())},
         "files": files, "configs_sha256": configs, "scene_files_sha256": scenes,
         "groups": [{"group_id": g_["group_id"], "scene_id": g_["scene_id"], "init_frame": g_["init_frame"],
                     "n_questions": len(g_["questions"]), "templates": [q["template"] for q in g_["questions"]],
                     "qids": [q["qid"] for q in g_["questions"]],
                     "constraints": [c["type"] for c in g_["constraints"]], "gate_d": gd[g_["group_id"]]} for g_ in groups]}
    write_json(vd / "freeze_manifest.json", m)
    ro = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for f in vd.iterdir():
        os.chmod(f, f.stat().st_mode & ro)
    os.chmod(vd, vd.stat().st_mode & ro)
    print(f"froze {version}: {len(groups)} groups, {len(qs)} questions; manifest sha256 {sha256(vd / 'freeze_manifest.json')}")
    return m


def verify(cfg, version):
    vd = version_dir(cfg, version)
    m = read_json(vd / "freeze_manifest.json")
    groups = [json.loads(l) for l in open(vd / "test_groups.jsonl")]
    files, scenes, configs = hashes(cfg, version, groups)
    bad = [f"file {k}" for k in set(files) | set(m["files"]) if files.get(k, {}).get("sha256") != m["files"].get(k, {}).get("sha256")]
    bad += [f"scene {s} {n}" for s in m["scene_files_sha256"] for n, h in m["scene_files_sha256"][s].items() if scenes.get(s, {}).get(n) != h]
    changed_cfg = [k for k, h in m["configs_sha256"].items() if configs.get(k) != h]
    ro = not any(os.access(p, os.W_OK) for p in [vd, *vd.iterdir()])
    print(f"verify {version}: data files {len(m['files'])}, scene files {sum(map(len, m['scene_files_sha256'].values()))}; "
          f"mismatches {len(bad)}; read-only {ro}; configs changed since freeze: {changed_cfg or 'none'}")
    for b in bad[:50]:
        print("  MISMATCH", b)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "verify"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--label", default="ARKitScenes release candidate")
    ap.add_argument("--build-commit", action="append", default=[], help="code commit that produced this data; may be given several times (format <commit>:<note>)")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "freeze":
        commits = [dict(zip(("commit", "note"), c.split(":", 1))) for c in args.build_commit]
        for c in commits:
            c["commit"] = git("rev-parse", c["commit"])
        freeze(cfg, args.version, args.label, commits)
    else:
        raise SystemExit(0 if verify(cfg, args.version) else 1)


if __name__ == "__main__":
    main()
