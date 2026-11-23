"""Freeze scannetpp_only_v0.2 (v0.2 accepted, ScanNet++ Steps 1–9 complete). Modelled on the ARKit freeze.py, without changing shared code.

  freeze   in work/scannetpp/_step8/scannetpp_only_v0.2/:
           - copy the pre-pruning question sets into pre_prune/ (Step 6 test_groups.jsonl / step6_stats.json, Step 7
             test_groups.jsonl / build_info.json), so rerunning Steps 6–7 later does not touch the frozen copies;
           - write freeze_manifest.json: question counts and the SHA-256 of every file in the version directory, the per-scene
             artifacts (scene_meta / frames / keyframes / objects / object_points / view_graph / visibility, etc.), configs and
             vocabularies, GATE D annotations and code files, plus the git commit;
           - make the version directory and its files read-only. The shared Step 8 refuses to write to a frozen version
             (common.assert_not_frozen).
           Also writes reports/scannetpp_only_v0.2_freeze.md.
  verify   recompute all SHA-256 hashes, compare them one by one with the manifest, and check that the version directory is
           still read-only; exit code 1 on any mismatch.

Usage: python -m actspace_build.scannetpp_freeze freeze|verify
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from .common import ROOT, config_matches_frozen, load_config, path_of, read_json, write_json
from .freeze import git, sha256
from .scannetpp import sp_work

VERSION = "scannetpp_only_v0.2"
SCENE_FILES = ("gate_a.json", "pose_check.npz", "scene_meta.json", "frames_all.json", "keyframes.json", "objects.json", "object_points.npz",
               "step3_summary.json", "camera_speed.json", "view_graph.json", "visibility.npz", "step4_summary.json")
CONFIG_FILES = ("configs/build.yaml", "configs/object_blacklist_scannetpp.yaml", "configs/paraphrases.yaml",
                "vocab/label_mapping_scannetpp.tsv", "vocab/categories_scannetpp_draft.tsv", "vocab/category_frequency_scannetpp.tsv")
CODE_GLOBS = ("src/actspace_build/*.py",)
PRE_PRUNE = {"step6_test_groups.jsonl": "_step6/test_groups.jsonl", "step6_stats.json": "_step6/step6_stats.json",
             "step7_test_groups.jsonl": "_step7/test_groups.jsonl", "step7_build_info.json": "_step7/build_info.json"}


def vdir(cfg) -> Path:
    return sp_work(cfg) / "_step8" / VERSION


def gate_d_annotations_dir(cfg) -> Path:
    """GATE D annotations of v0.1. They are archived in <parent of paths.work>/human_annotations/scannetpp/
    round1_scannetpp_only_v0.1/annotations/ (read-only); after the merge into main, reports/gateD/annotations/ holds the ARKit
    round-1 annotations (same file names), so the archive is preferred and the original location in the scannetpp worktree is
    used only when the archive does not exist."""
    arch = path_of(cfg, "work").parent / "human_annotations" / "scannetpp" / "round1_scannetpp_only_v0.1" / "annotations"
    return arch if arch.is_dir() else path_of(cfg, "reports") / "gateD" / "annotations"


def collect(cfg) -> dict:
    vd = vdir(cfg)
    files = {str(f.relative_to(vd)): {"sha256": sha256(f), "bytes": f.stat().st_size}
             for f in sorted(vd.rglob("*")) if f.is_file() and f.name != "freeze_manifest.json"}
    pre = [json.loads(l) for l in open(vd / "pre_prune" / "step6_test_groups.jsonl")]
    scenes = {}
    for sid in sorted({g_["scene_id"].split("_", 1)[1] for g_ in pre}):
        wd = sp_work(cfg) / sid
        scenes[sid] = {n: sha256(wd / n) for n in SCENE_FILES if (wd / n).exists()}
    configs = {n: sha256(ROOT / n) for n in CONFIG_FILES}
    ann = sorted(gate_d_annotations_dir(cfg).glob("*.json"))
    gate_d = {f.name: sha256(f) for f in ann}
    code = {str(p.relative_to(ROOT)): sha256(p) for g in CODE_GLOBS for p in sorted(ROOT.glob(g))}
    return {"files": files, "scene_files_sha256": scenes, "configs_sha256": configs, "gate_d_annotations_sha256": gate_d, "code_sha256": code}


def freeze(cfg) -> dict:
    vd = vdir(cfg)
    if (vd / "freeze_manifest.json").exists():
        raise SystemExit(f"{VERSION} is already frozen")
    pp = vd / "pre_prune"
    pp.mkdir(exist_ok=True)
    for dst, src in PRE_PRUNE.items():
        shutil.copy2(sp_work(cfg) / src, pp / dst)
    groups = [json.loads(l) for l in open(vd / "test_groups.jsonl")]
    qs = [q for g_ in groups for q in g_["questions"]]
    pre = [json.loads(l) for l in open(pp / "step7_test_groups.jsonl")]
    h = collect(cfg)
    dirty = [l[3:] for l in git("status", "--porcelain").splitlines() if l]
    m = {"version": VERSION, "frozen": True, "frozen_on": date.today().isoformat(),
         "decided_by": "after GATE D (human verification) and acceptance of v0.2 (ScanNet++ Steps 1-9 complete)",
         "rule": "nothing that affects v0.2 is modified after freezing; the merge with ARKitScenes reruns Step 8 from pre_prune and produces a new version",
         "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head_at_freeze": git("rev-parse", "HEAD"), "uncommitted_at_freeze": dirty},
         "counts": {"scenes": len({g_["scene_id"] for g_ in groups}), "groups": len(groups), "questions": len(qs),
                    "by_template": dict(sorted(Counter(q["template"] for q in qs).items())),
                    "by_evidence_label": dict(sorted(Counter(q["evidence_label"] for q in qs).items())),
                    "pre_prune_groups": len(pre), "pre_prune_questions": sum(len(g_["questions"]) for g_ in pre),
                    "scenes_with_files": len(h["scene_files_sha256"])},
         **h,
         "groups": [{"group_id": g_["group_id"], "scene_id": g_["scene_id"], "init_frame": g_["init_frame"],
                     "qids": [q["qid"] for q in g_["questions"]], "templates": [q["template"] for q in g_["questions"]]} for g_ in groups]}
    write_json(vd / "freeze_manifest.json", m)
    ro = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for f in sorted(vd.rglob("*"), key=lambda p: -len(p.parts)):
        os.chmod(f, f.stat().st_mode & ro)
    os.chmod(vd, vd.stat().st_mode & ro)
    print(f"froze {VERSION}: {len(groups)} groups, {len(qs)} questions; manifest sha256 {sha256(vd / 'freeze_manifest.json')}")
    return m


def verify(cfg) -> bool:
    vd = vdir(cfg)
    m = read_json(vd / "freeze_manifest.json")
    now = collect(cfg)
    bad = []
    for key in ("files",):
        for k in set(m[key]) | set(now[key]):
            if (m[key].get(k) or {}).get("sha256") != (now[key].get(k) or {}).get("sha256"):
                bad.append(f"{key} {k}")
    for s, d in m["scene_files_sha256"].items():
        for n, hsh in d.items():
            if now["scene_files_sha256"].get(s, {}).get(n) != hsh:
                bad.append(f"scene {s} {n}")
    for key in ("configs_sha256", "gate_d_annotations_sha256"):
        for k, hsh in m[key].items():
            if now[key].get(k) != hsh and not (key == "configs_sha256" and config_matches_frozen(k, hsh)):   # path-only difference
                bad.append(f"{key} {k}")
    code_changed = [k for k, hsh in m["code_sha256"].items() if now["code_sha256"].get(k) != hsh]
    code_added = sorted(set(now["code_sha256"]) - set(m["code_sha256"]))
    ro = not any(os.access(p, os.W_OK) for p in [vd, *vd.rglob("*")])
    n = len(m["files"]) + sum(map(len, m["scene_files_sha256"].values())) + len(m["configs_sha256"]) + len(m["gate_d_annotations_sha256"])
    print(f"verify {VERSION}: {n} hashed files (version dir {len(m['files'])}, scene files {sum(map(len, m['scene_files_sha256'].values()))}, "
          f"configs {len(m['configs_sha256'])}, GATE D annotations {len(m['gate_d_annotations_sha256'])}); mismatches {len(bad)}; read-only {ro}; "
          f"code files changed since freeze: {code_changed or 'none'}" + (f"; added: {code_added}" if code_added else ""))
    for b in bad[:50]:
        print("  MISMATCH", b)
    return not bad and ro


def write_report(cfg, m) -> None:
    vd = vdir(cfg)
    c = m["counts"]
    mh = sha256(vd / "freeze_manifest.json")
    L = [f"# {VERSION} freeze", "",
         "## Conclusion", "",
         f"- Final spot check 48 / 50 (#15, #29 and #45 re-reviewed as recognizable); `{VERSION}` accepted, ScanNet++ Steps 1–9 complete, **frozen**. Nothing on the `scannetpp` branch that affects v0.2 is modified afterwards; "
         "the merge with ARKit is handled separately.",
         f"- Freeze manifest `work/scannetpp/_step8/{VERSION}/freeze_manifest.json` (not in the repository), SHA-256 `{mh}`.",
         f"- The version directory and its files (including the pre-pruning question-set copy `pre_prune/`) are read-only; the shared Step 8 refuses to write to a frozen version (`common.assert_not_frozen`).", "",
         "## Size", "",
         "| Scenes | Groups | Questions | T1 | T2 | T3 | T4 | T5 | explore | unanswerable | init | Pre-pruning (groups / questions) |", "|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| {c['scenes']} | {c['groups']} | {c['questions']} | " + " | ".join(str(c["by_template"].get(t, 0)) for t in ("T1", "T2", "T3", "T4", "T5"))
         + f" | {c['by_evidence_label'].get('explore', 0)} | {c['by_evidence_label'].get('unanswerable', 0)} | {c['by_evidence_label'].get('init', 0)} | "
         f"{c['pre_prune_groups']} / {c['pre_prune_questions']} |", "",
         "## 1. v0.2 question set and pre-pruning question sets", "",
         f"Directory `work/scannetpp/_step8/{VERSION}/`:", "", "| File | SHA-256 | Bytes |", "|---|---|---|",
         *[f"| `{k}` | `{v['sha256']}` | {v['bytes']} |" for k, v in m["files"].items()],
         f"| `freeze_manifest.json` | `{mh}` | — |", "",
         "- `test_groups.jsonl` is the v0.2 question set; `manifest.json` is the Step 8 pruning record.",
         "- `pre_prune/` holds the pre-pruning question sets copied from `work/scannetpp/_step6/` and `_step7/` at freeze time (the Step 7 `test_groups.jsonl` is the Step 8 input; after the merge, Step 8 is rerun from here).", "",
         "## 2. Per-scene artifacts", "",
         f"For the {c['scenes_with_files']} scenes in the pre-pruning question set, the SHA-256 of {len(SCENE_FILES)} files per scene is recorded ({sum(map(len, m['scene_files_sha256'].values()))} in total): "
         + ", ".join(f"`{n}`" for n in SCENE_FILES) + ". Paths are `work/scannetpp/<scene_id>/<file>`. Full hashes are in the freeze manifest; the table shows the file count per scene and the first 16 hex digits of the `visibility.npz` and `objects.json` hashes.", "",
         "| Scene | Files | objects.json | visibility.npz |", "|---|---|---|---|",
         *[f"| `{s}` | {len(d)} | `{d.get('objects.json', '')[:16]}` | `{d.get('visibility.npz', '')[:16]}` |" for s, d in m["scene_files_sha256"].items()], "",
         "These files are outside the version directory and are not read-only (same as for ARKit); `verify` reports each one that is changed later.", "",
         "## 3. Configs and vocabularies", "", "| File | SHA-256 |", "|---|---|", *[f"| `{k}` | `{v}` |" for k, v in m["configs_sha256"].items()], "",
         "## 4. GATE D annotations", "",
         f"The SHA-256 of the {len(m['gate_d_annotations_sha256'])} files in `reports/gateD/annotations/` (ann01, GATE D verification of v0.1) is recorded in the freeze manifest.", "",
         "## 5. Code and git commit", "",
         f"- Branch at freeze `{m['git']['branch']}`, HEAD `{m['git']['head_at_freeze']}`; uncommitted changes at freeze: {m['git']['uncommitted_at_freeze'] or 'none'}.",
         f"- The SHA-256 of all {len(m['code_sha256'])} files in `src/actspace_build/*.py` is also recorded in the freeze manifest (`verify` only reports code changes and does not count them as data mismatches).",
         "- Commit that produced v0.2: `a11c811` (Steps 7–8, integrity check, baseline); Step 3–6 rule commits: `540a4fc` and the later rule change in `a11c811`.", "",
         "## 6. Verification", "",
         "```", "cd src && python -m actspace_build.scannetpp_freeze verify", "```", "",
         "Recomputes all SHA-256 hashes of the version directory, per-scene artifacts, configs and vocabularies, and GATE D annotations, compares them one by one with the manifest, and checks that the version directory is still read-only; exit code 1 on any mismatch or if the directory is writable."]
    (path_of(cfg, "reports") / f"{VERSION}_freeze.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "verify"])
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "freeze":
        write_report(cfg, freeze(cfg))
    else:
        sys.exit(0 if verify(cfg) else 1)


if __name__ == "__main__":
    main()
