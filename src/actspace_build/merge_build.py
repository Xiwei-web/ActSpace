"""Official merged version merged_v1.0; same pipeline as the dry run merged_v1.0_dryrun, written to the real work directory.

  build      merged pre-pruning set (pre-pruning question sets of the two frozen versions, ids prefixed with the dataset)
             -> standing-point horizontal separation filter (DEGENERATE_SEPARATION)
             -> option-letter balancing (separately within each dataset, rebalance_letters_by_dataset)
             -> Step 8 (shared step8_debias, without quadrant balancing by removal) -> quadrant balancing by removal (QUADRANT_REBALANCE, ±5%)
             -> option-letter rebalancing -> text-only blind-test recheck (two controls + 5 stability runs, all must pass)
             -> <paths.work>/merged/_step8/<version>/
  integrity  integrity checks: shared integrity.check (DATASET = merged) + merged-version extras (per-dataset visibility criteria,
             ScanNet++ blacklist, standing-point horizontal separation, ids, group conditions, T2 pair share, T4 front/back,
             consistency with the source questions) -> reports/<version>_integrity_check.md
  freeze     write freeze_manifest.json (SHA-256 of the version directory, per-scene dependency files, configs incl. configs/merge.yaml,
             source freeze manifests and code) and make it read-only
  verify     recompute and compare; also runs verify for the two source versions
  compare    item-by-item comparison with the dry run (sandbox) -> <sandbox>/results/compare_formal.json

<paths.work>/merged/<scene>/ holds symlinks to the official scene artifacts of both datasets (used read-only); dropped.jsonl is
written there (step = merge_sep / step8 / merge_quadrant).
Usage: python -m actspace_build.merge_build build|integrity|freeze|verify|compare
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from . import freeze as arkit_freeze
from . import integrity, scannetpp_freeze, step8_debias
from .common import ROOT, DropLog, assert_not_frozen, config_matches_frozen, path_of, read_json, write_json
from .merge_dryrun import (DATASET_OF_TAG, TAGS, dataset_as, dataset_of_scene, describe, ds_of, letter_bias, load_merge_config,
                           merged_pre_prune, quadrant_rebalance, read_groups, real_work, sandbox, scene_geometry, separation, sha256,
                           source_path, source_probe, version_dir)

DATASET = "merged"
SCENE_FILES = {"arkitscenes": tuple(dict.fromkeys(arkit_freeze.SCENE_FILES + ("pose_check.npz",))),
               "scannetpp": scannetpp_freeze.SCENE_FILES}
CONFIG_FILES = ("configs/build.yaml", "configs/merge.yaml", "configs/paraphrases.yaml", "configs/object_blacklist.yaml",
                "configs/object_blacklist_scannetpp.yaml", "vocab/categories.tsv", "vocab/label_mapping_scannetpp.tsv",
                "vocab/categories_scannetpp_draft.tsv")


def mdir(cfg) -> Path:
    return real_work(cfg) / DATASET


def vdir(cfg) -> Path:
    return mdir(cfg) / "_step8" / cfg["merge"]["version"]


def write_groups(p: Path, groups: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


def link_scenes(cfg, sids) -> None:
    """<work>/merged/<vid>/<file> -> <work>/<dataset>/<vid>/<file> (symlinks, used read-only)."""
    for sid in sorted(sids):
        ds, vid = dataset_of_scene(sid), sid.split("_", 1)[1]
        src, dst = real_work(cfg) / ds / vid, mdir(cfg) / vid
        dst.mkdir(parents=True, exist_ok=True)
        for n in SCENE_FILES[ds]:
            if (src / n).exists() and not (dst / n).exists():
                (dst / n).symlink_to(src / n)


def rebalance_letters_by_dataset(groups: list[dict], seed: int) -> None:
    """Distribute option letters per (template, number of options) strictly uniformly within each dataset (letter counts differ by
    <= 1), and strictly uniformly over the merged set as well: each dataset's remainder questions go to the letters with the lowest
    merged count so far (ties broken at random). Options are reordered as in step8_debias.rebalance_letters (the correct option is
    moved to the assigned position, the others are shuffled; only the order changes, not the content).
    Balancing only over the merged set leaves biases of 6-7 pp within a dataset (ARKit T4 6.9 pp in the dry run): a text-only blind
    model can recognise the dataset from the object names (the two vocabularies differ) and then exploit that dataset's letter bias;
    since results are also reported per dataset, letters are balanced per dataset."""
    rng = np.random.default_rng([seed, 89])
    buckets = defaultdict(lambda: defaultdict(list))
    for g in groups:
        for q in g["questions"]:
            if q["answer_type"] != "numeric":
                buckets[(q["template"], len(q["options"]))][ds_of(g)].append(q)
    L = step8_debias.LETTERS
    for (t, n), by_ds in sorted(buckets.items()):
        total = np.zeros(n, int)
        for ds in TAGS:
            qs = by_ds.get(ds, [])
            if not qs:
                continue
            base, r = divmod(len(qs), n)
            tie = rng.permutation(n)
            extra = sorted(range(n), key=lambda k: (total[k], tie[k]))[:r]
            pos = np.array([k for k in range(n) for _ in range(base)] + extra)
            rng.shuffle(pos)
            for q, pp in zip(qs, pos):
                vals = q.get("option_objs") or q["option_values"]
                texts = step8_debias.option_texts(q)
                a = L.index(q["answer"])
                others = [k for k in range(n) if k != a]
                rng.shuffle(others)
                order = others[:pp] + [a] + others[pp:]
                key = "option_objs" if "option_objs" in q else "option_values"
                q[key] = [vals[k] for k in order]
                q["options"] = [f"{L[i]}. {texts[k]}" for i, k in enumerate(order)]
                q["answer"] = L[int(pp)]
            total += np.bincount(pos, minlength=n)


def step8_cfg(cfg, version) -> dict:
    c = copy.deepcopy(cfg)
    c["paths"]["reports"] = str(mdir(cfg) / "_reports")
    (mdir(cfg) / "_reports").mkdir(parents=True, exist_ok=True)
    c["debias"]["output_version"] = version
    c["debias"]["keep_group_on_constraint_loss"] = True
    c["questions"]["quadrant_balance_tol"] = 1.0   # quadrant balancing by removal is done separately after Step 8
    c["debias"]["t4_front_back_tol"] = 1.0
    return c


# ---------------------------------------------------------------- build

def build(cfg) -> dict:
    V = cfg["merge"]["version"]
    vd = vdir(cfg)
    assert_not_frozen(vd)
    md = mdir(cfg)
    thr = cfg["questions"]["min_horizontal_sep_m"]
    groups = merged_pre_prune(cfg)
    sids = {g["scene_id"] for g in groups}
    link_scenes(cfg, sids)
    write_groups(md / "_premerge" / "test_groups.jsonl", groups)
    n_pre = describe(groups)["counts"]
    # 1. standing-point horizontal separation
    geo = {sid: scene_geometry(cfg, sid) for sid in sids}
    drops = {sid: DropLog(md / sid.split("_", 1)[1], "merge_sep") for sid in sids}
    kill, reasons = set(), {}
    for gi, g in enumerate(groups):
        centers, cams = geo[g["scene_id"]]
        for qi, q in enumerate(g["questions"]):
            r = separation(q, centers, cams[g["init_frame"]])
            if r is not None and r[0] < thr:
                kill.add((gi, qi))
                reasons[(gi, qi)] = "DEGENERATE_SEPARATION"
    removed_sep = step8_debias.remove_questions(groups, kill, reasons, drops, cfg["groups"], cfg_keep=True)
    # 2. option-letter balancing (after merging)
    rebalance_letters_by_dataset(groups, cfg["seed"])
    write_groups(md / "_step7" / "test_groups.jsonl", groups)
    # 3. Step 8 (shared module, without quadrant balancing by removal)
    c8 = step8_cfg(cfg, f"{V}_step8")
    assert_not_frozen(md / "_step8" / f"{V}_step8")
    with dataset_as(DATASET):
        step8_debias.run(c8)
    s8 = md / "_step8" / f"{V}_step8"
    groups = read_groups(s8 / "test_groups.jsonl")
    m8 = read_json(s8 / "manifest.json")
    # 4. quadrant balancing by removal (current ±5% rule; bias scores are the final Step 8 bias_score)
    dq = {sid: DropLog(md / sid.split("_", 1)[1], "merge_quadrant") for sid in {g["scene_id"] for g in groups}}
    algo_q = sorted(r[0] for r in quadrant_rebalance(copy.deepcopy(groups), cfg)[0] if r[2] == "QUADRANT_REBALANCE")   # for the record only
    fixed = set(cfg["merge"]["quadrant_rebalance_qids"])
    kq = {(gi, qi) for gi, g in enumerate(groups) for qi, q in enumerate(g["questions"]) if q["qid"] in fixed}
    if len(kq) != len(fixed):
        raise SystemExit(f"{len(fixed) - len(kq)} questions in quadrant_rebalance_qids are not in the Step 8 output")
    removed_q = step8_debias.remove_questions(groups, kq, {k: "QUADRANT_REBALANCE" for k in kq}, dq, cfg["groups"], cfg_keep=True)
    status_q = "ok" if step8_debias.quadrant_violation(groups, cfg) is None else "quadrant still out of tolerance"
    # 5. option-letter rebalancing; text-only blind-test recheck (two controls + stability check)
    rebalance_letters_by_dataset(groups, cfg["seed"])
    with dataset_as(DATASET):
        _, names, _ = step8_debias.load(c8)
    after = step8_debias.evaluate(groups, names, c8, controls=True)
    stab = step8_debias.stability(groups, names, c8)
    for t, r in after.items():
        for b, (gi, qi) in zip(r["bias"], r["items"]):
            groups[gi]["questions"][qi]["bias_score"] = round(float(b), 4)
    letters = letter_bias(groups)
    letters_ds = letter_bias(groups, by_dataset=True)
    tol = cfg["questions"]["quadrant_balance_tol"]
    checks = {"blind_all_ok": all(r["ok"] for r in after.values()),
              "quadrant_ok": step8_debias.quadrant_violation(groups, cfg) is None,
              # merged set and each dataset: strictly uniform per (template, number of options) (letter counts differ by <= 1)
              "letters_ok": all(max(v["counts"].values()) - min(v["counts"].get(step8_debias.LETTERS[i], 0) for i in range(int(k.split("|")[-1]))) <= 1
                                for k, v in letters.items()),
              "letters_by_dataset_ok": all(max(v["counts"].values()) - min([v["counts"].get(step8_debias.LETTERS[i], 0) for i in range(int(k.split("|")[-1]))]) <= 1
                                           for k, v in letters_ds.items()),
              "step8_status": m8["status"], "quadrant_status": status_q,
              "quadrant_algorithm_would_remove": algo_q, "quadrant_algorithm_same_as_decided": set(algo_q) == fixed}
    status = "ok" if checks["blind_all_ok"] and checks["quadrant_ok"] and checks["letters_ok"] and checks["letters_by_dataset_ok"] \
        and status_q == "ok" and m8["status"] == "ok" \
        else "check_failed"
    vd.mkdir(parents=True, exist_ok=True)
    write_groups(vd / "test_groups.jsonl", groups)
    (vd / "pre_prune").mkdir(exist_ok=True)
    shutil.copyfile(md / "_premerge" / "test_groups.jsonl", vd / "pre_prune" / "premerge_test_groups.jsonl")
    shutil.copyfile(md / "_step7" / "test_groups.jsonl", vd / "pre_prune" / "step8_input_test_groups.jsonl")
    src = {ds: {"version": cfg["merge"]["sources"][ds]["version"],
                "freeze_manifest_sha256": sha256(version_dir(cfg, ds) / "freeze_manifest.json"),
                "pre_prune": str(cfg["merge"]["sources"][ds]["pre_prune"]), "pre_prune_sha256": sha256(source_path(cfg, ds))} for ds in TAGS}
    blind = {t: {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                 if k not in ("bias", "items")} for t, r in after.items()}
    man = {"version": V, "status": status, "checks": checks, "sources": src,
           "min_horizontal_sep_m": thr, "groups": len(groups), "questions": sum(len(g["questions"]) for g in groups),
           "stages": {"premerge": {"groups": sum(n_pre["groups"].values()), "questions": sum(n_pre["n_questions"].values())},
                      "separation": [{"qid": q, "template": t, "reason": r} for q, t, r in removed_sep],
                      "step8": {"version": f"{V}_step8", "status": m8["status"], "rounds": m8["rounds"], "removed": m8["removed"]},
                      "quadrant": [{"qid": q, "template": t, "reason": r} for q, t, r in removed_q]},
           "blind_after": blind, "stability_after": {t: {"mean_pp": round(100 * v[0], 2), "std_pp": round(100 * v[1], 2), "runs_pp": v[2]}
                                                      for t, v in stab.items()},
           "letters": letters, "letters_by_dataset": letters_ds, "describe": describe(groups)}
    write_json(vd / "manifest.json", man)
    # Step 8 report: replace the fixed header of the shared module with a description of the merged version
    rep = (mdir(cfg) / "_reports" / f"step8_debias_{V}_step8.md").read_text()
    lines = rep.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Result:"))
    head = [f"# Step 8: blind test and debiasing (ARKit + ScanNet++ merged -> {V})", "",
            "> Generated by `python -m actspace_build.merge_build build`; reuses the shared module `step8_debias` (unchanged, with the "
            "dataset switched to merged at run time). The input is the merged pre-pruning question set after the standing-point horizontal "
            "separation filter and option-letter balancing, `work/merged/_step7/test_groups.jsonl`.", "",
            "> **Difference from the shared method description below**: Step 8 runs without quadrant balancing by removal (T3/T4 quadrant "
            "tolerance and T4 front/back tolerance set to 1.0); quadrant balancing is done separately after Step 8 with ±5% (`QUADRANT_REBALANCE`). "
            f"`{V}_step8` below is the intermediate Step 8 output; the final version is `{V}`.", ""]
    body = "\n".join(lines[i:]).replace("work/arkitscenes/", "work/merged/")
    (path_of(cfg, "reports") / f"step8_debias_{V}.md").write_text("\n".join(head) + "\n" + body)
    print(json.dumps({k: man[k] for k in ("status", "checks", "groups", "questions")}, ensure_ascii=False))
    return man


# ---------------------------------------------------------------- integrity checks

def integrity_check(cfg) -> dict:
    from .scannetpp_step3 import load_blacklist as sp_blacklist
    from .step56_questions import dedup_key
    V = cfg["merge"]["version"]
    stage = f"_step8/{V}"
    out_name = f"{V}_integrity_check.md"
    old = integrity.DATASET
    integrity.DATASET = DATASET
    try:
        groups, res, letters, quads, nonref = integrity.check(cfg, stage)
        integrity.write_report(cfg, groups, res, letters, quads, nonref, out_name=out_name, stage=stage)
    finally:
        integrity.DATASET = old
    extra = defaultdict(lambda: [0, []])

    def rec(item, ok, detail):
        if ok:
            extra[item][0] += 1
        else:
            extra[item][1].append(detail)
    thr = cfg["questions"]["min_horizontal_sep_m"]
    rho_thr = {"arkitscenes": cfg["visibility"]["visible_min_ratio"], "scannetpp": cfg["scannetpp"]["visible_min_ratio"]}
    sp = cfg["scannetpp"]
    pre = {(g["source_dataset"], q["source_qid"]): q for g in read_groups(mdir(cfg) / "_premerge" / "test_groups.jsonl") for q in g["questions"]}
    seen_g, seen_q = set(), set()
    keys = Counter()
    vis_cache = {}
    n_all = sum(len(g["questions"]) for g in groups)
    for g in groups:
        ds, sid = g["source_dataset"], g["scene_id"]
        vid = sid.split("_", 1)[1]
        tag = TAGS[ds]
        rec("Group ids are unique, carry the dataset prefix and match the scene", g["group_id"] not in seen_g and g["group_id"] == f"{tag}_{g['source_group_id']}"
            and dataset_of_scene(sid) == ds, g["group_id"])
        seen_g.add(g["group_id"])
        ts = [q["template"] for q in g["questions"]]
        rec("Group conditions: 3-5 questions, >= 2 templates", 3 <= len(ts) <= 5 and len(set(ts)) >= 2, g["group_id"])
        rec("Constraint questions are all in the same group", all(x in {q["qid"] for q in g["questions"]} for c in g["constraints"] for x in c["qids"]), g["group_id"])
        centers, cams = scene_geometry(cfg, sid)
        black = sp_blacklist(cfg, vid) if ds == "scannetpp" else {}
        if vid not in vis_cache:
            vis_cache[vid] = np.load(real_work(cfg) / ds / vid / "visibility.npz")
        z = vis_cache[vid]
        row = {int(o): i for i, o in enumerate(z["obj_ids"])}
        col = {str(n): j for j, n in enumerate(z["frame_ids"])}
        for q in g["questions"]:
            rec("Question ids are unique and carry the dataset prefix", q["qid"] not in seen_q and q["qid"] == f"{tag}_{q['source_qid']}", q["qid"])
            seen_q.add(q["qid"])
            r = separation(q, centers, cams[g["init_frame"]])
            if r is not None:
                rec(f"Direction questions: standing-point horizontal separation >= {thr} m (T3 / T5 standing point; T4 object vs. start camera)", r[0] >= thr, f"{q['qid']}: {r[0]:.3f} m {r[1]}")
            named = list(q["slots"].values()) + list(q.get("option_objs") or [])
            if ds == "scannetpp":
                rec("ScanNet++ questions contain no ScanNet++ blacklisted objects (configs/object_blacklist_scannetpp.yaml)",
                    not (set(named) | set(q["evidence_objs"])) & set(black), q["qid"])
                keys[(sid, dedup_key(q, True))] += 1
            # identical to the source pre-pruning question (only option order and letters may change)
            p = pre[(ds, q["source_qid"])]
            same = all(p.get(k) == q.get(k) for k in ("template", "variant", "slots", "answer_type", "evidence_objs", "evidence_label",
                                                       "oracle_nodes", "link_views", "path_views", "text", "answer_raw"))
            if q["answer_type"] == "numeric":
                same &= p["answer"] == q["answer"]
            else:
                same &= integrity.stored_choice(p) == integrity.stored_choice(q) and \
                    sorted(map(str, p.get("option_objs") or p["option_values"])) == sorted(map(str, q.get("option_objs") or q["option_values"]))
            rec("Question content matches the pre-pruning question of the frozen source version (only option order / letters reassigned)", same, q["qid"])
            # per-dataset visibility criterion: visible at oracle evidence nodes satisfies this dataset's ρ threshold
            if q["evidence_label"] != "unanswerable":
                nodes = [g["init_frame"], *(q.get("oracle_nodes") or [])]
                ok = all(any(z["state"][row[o], col[n]] == 2 and float(z["rho"][row[o], col[n]]) >= rho_thr[ds] - 1e-3 for n in nodes)
                         for o in q["evidence_objs"])
                rec("Oracle evidence: each evidence object is visible at some node of p0 + oracle_nodes with ρ >= the dataset threshold (ARKit 0.30 / ScanNet++ 0.40)", ok, q["qid"])
    for (sid, k), n in keys.items():
        lim = cfg["questions"]["max_t4_per_object_per_scene"]
        rec("ScanNet++: no duplicate T2 within a scene by reference object + direction + correct answer", n <= (lim if k[0] == "T4" else 1), f"{sid}: {k}")
    # scene level: visible pairs satisfy the dataset's visibility criteria
    for g in {g["scene_id"]: g for g in groups}.values():
        ds, vid = g["source_dataset"], g["scene_id"].split("_", 1)[1]
        z = vis_cache[vid]
        v = z["state"] == 2
        ok = bool((z["rho"][v].astype(float) >= rho_thr[ds] - 1e-3).all())
        if ds == "scannetpp":
            ok &= bool((z["min_rect_short_px"][v] >= sp["vis_min_short_side_px"] - 1e-3).all())
            ok &= bool((z["sharpness_ratio"][v].astype(float) >= sp["vis_min_sharpness_ratio"] - 1e-3).all())
            ok &= not bool(v[:, z["motion_fast_node"]].any())
        rec("All visible pairs in the scene satisfy the dataset's visibility criteria (ARKit ρ >= 0.30; ScanNet++ ρ >= 0.40, short side >= 105 px, sharpness >= 0.18, no fast-motion node)",
            ok, g["scene_id"])
    qs = [q for g in groups for q in g["questions"]]
    pair = sum(q.get("variant") == "pair" for q in qs)
    rec(f"T2 pair share of all questions <= {cfg['questions']['pair_max_frac']:.0%}", pair / n_all <= cfg["questions"]["pair_max_frac"], f"{pair}/{n_all}")
    t4 = [q for q in qs if q["template"] == "T4" and q["evidence_label"] != "unanswerable"]
    fr = sum(step8_debias.answer_text(q) in step8_debias.FRONT for q in t4) / len(t4)
    rec("T4 answerable questions: front / behind 50% ± 5% each", abs(fr - 0.5) <= cfg["debias"]["t4_front_back_tol"], f"front {fr:.3f}")
    for t, c in quads.items():
        tot = sum(c.values())
        rec(f"{t}: each of the four quadrants 25% ± 5%", all(abs(c[k] / tot - 0.25) <= cfg["questions"]["quadrant_balance_tol"] for k in step8_debias.QUADS), t)
    for (t, n), c in letters.items():
        tot = sum(c.values())
        rec("Option letters within ±5% per (template, number of options)", all(abs(c[step8_debias.LETTERS[i]] / tot - 1 / n) <= 0.05 for i in range(n)), f"{t}/{n}")
    p = path_of(cfg, "reports") / out_name
    text = p.read_text().replace("# ARKit question integrity check", f"# {V} integrity check (ARKit + ScanNet++ merged)")
    text = text.replace("work/arkitscenes/", "work/merged/")
    text = text.replace("No blacklisted objects or excluded scenes", "No ARKit blacklisted objects or excluded scenes (configs/object_blacklist.yaml; always true for ScanNet++ questions, see the additional checks below)")
    L = ["", "## Additional checks for the merged version", "", "| Check | Passed | Total | Result |", "|---|---|---|---|"]
    fails = []
    for k, (ok, bad) in extra.items():
        tot = ok + len(bad)
        L.append(f"| {k} | {ok} | {tot} | {'100%' if not bad else f'**{ok / tot:.1%}**'} |")
        if bad:
            fails.append((k, bad))
    if fails:
        L += ["", "### Questions failing the additional checks", ""]
        for k, bad in fails:
            L += [f"- {k}: {len(bad)}, e.g. {'; '.join(map(str, bad[:10]))}"]
    p.write_text(f"> `{V}` (`work/merged/{stage}/test_groups.jsonl`); the shared `integrity` module runs with the dataset switched to merged (scene artifacts are symlinks to the official artifacts of both datasets).\n\n"
                 + text + "\n".join(L) + "\n")
    all_items = {**{k: (v[0], len(v[1])) for k, v in res.items()}, **{k: (v[0], len(v[1])) for k, v in extra.items()}}
    out = {"items": {k: {"pass": a, "fail": b} for k, (a, b) in all_items.items()}, "all_pass": all(b == 0 for _, b in all_items.values())}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------- freeze

def hashes(cfg) -> dict:
    vd = vdir(cfg)
    files = {str(f.relative_to(vd)): {"sha256": sha256(f), "bytes": f.stat().st_size} for f in sorted(vd.rglob("*"))
             if f.is_file() and f.name != "freeze_manifest.json"}
    groups = read_groups(vd / "test_groups.jsonl")
    scenes = {}
    for sid in sorted({g["scene_id"] for g in groups}):
        ds, vid = dataset_of_scene(sid), sid.split("_", 1)[1]
        wd = real_work(cfg) / ds / vid
        scenes[sid] = {n: sha256(wd / n) for n in SCENE_FILES[ds] if (wd / n).exists()}
    configs = {n: sha256(ROOT / n) for n in CONFIG_FILES}
    sources = {ds: sha256(version_dir(cfg, ds) / "freeze_manifest.json") for ds in TAGS}
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src" / "actspace_build").glob("*.py"))}
    return {"files": files, "scene_files_sha256": scenes, "configs_sha256": configs, "source_freeze_manifests_sha256": sources,
            "code_sha256": code}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def freeze_version(cfg) -> dict:
    vd = vdir(cfg)
    if (vd / "freeze_manifest.json").exists():
        raise SystemExit(f"{vd} is already frozen")
    man = read_json(vd / "manifest.json")
    if man["status"] != "ok":
        raise SystemExit(f"manifest status = {man['status']}; not freezing")
    groups = read_groups(vd / "test_groups.jsonl")
    qs = [q for g in groups for q in g["questions"]]
    m = {"version": cfg["merge"]["version"], "label": "ARKitScenes + ScanNet++ merged release candidate", "frozen": True, "frozen_on": date.today().isoformat(),
         "decided_by": "merge decisions recorded in configs/merge.yaml (quadrant rebalancing, dataset prefixes in question IDs)",
         "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head_at_freeze": git("rev-parse", "HEAD"),
                 "uncommitted_at_freeze": [l[3:] for l in git("status", "--porcelain").splitlines() if l],
                 "remote": None},
         "counts": {"scenes": dict(Counter(ds_of(g) for g in {g["scene_id"]: g for g in groups}.values())),
                    "groups": dict(Counter(ds_of(g) for g in groups)), "questions": len(qs),
                    "by_dataset_template": dict(sorted(Counter(f"{ds_of(g)}|{q['template']}" for g in groups for q in g["questions"]).items())),
                    "unanswerable": sum(q["evidence_label"] == "unanswerable" for q in qs)},
         **hashes(cfg)}
    write_json(vd / "freeze_manifest.json", m)
    ro = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for p in sorted(vd.rglob("*"), reverse=True):
        os.chmod(p, p.stat().st_mode & ro)
    os.chmod(vd, vd.stat().st_mode & ro)
    print(f"froze {m['version']}: {len(groups)} groups, {len(qs)} questions; manifest sha256 {sha256(vd / 'freeze_manifest.json')}")
    return m


def verify_version(cfg) -> bool:
    vd = vdir(cfg)
    m = read_json(vd / "freeze_manifest.json")
    now = hashes(cfg)
    bad = [f"file {k}" for k in set(m["files"]) | set(now["files"])
           if (m["files"].get(k) or {}).get("sha256") != (now["files"].get(k) or {}).get("sha256")]
    bad += [f"scene {s} {n}" for s, d in m["scene_files_sha256"].items() for n, h in d.items() if now["scene_files_sha256"].get(s, {}).get(n) != h]
    path_only = [k for k, h in m["configs_sha256"].items() if now["configs_sha256"].get(k) != h and config_matches_frozen(k, h)]
    bad += [f"config {k}" for k, h in m["configs_sha256"].items() if now["configs_sha256"].get(k) != h and k not in path_only]
    bad += [f"source {k}" for k, h in m["source_freeze_manifests_sha256"].items() if now["source_freeze_manifests_sha256"].get(k) != h]
    code_changed = [k for k, h in m["code_sha256"].items() if now["code_sha256"].get(k) != h]
    # read-only: check the permission bits (other machines may run as root, for which os.access always reports writable)
    ro = not any(p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in [vd, *vd.rglob("*")])
    n = len(m["files"]) + sum(map(len, m["scene_files_sha256"].values())) + len(m["configs_sha256"]) + len(m["source_freeze_manifests_sha256"])
    print(f"verify {m['version']}: {n} hashed files (version dir {len(m['files'])}, scene files {sum(map(len, m['scene_files_sha256'].values()))}, "
          f"configs {len(m['configs_sha256'])}, source manifests {len(m['source_freeze_manifests_sha256'])}); mismatches {len(bad)}; read-only {ro}; "
          f"code files changed since freeze: {code_changed or 'none'}"
          + (f"; configs differing only in machine-specific paths: {path_only}" if path_only else ""))
    for b in bad[:50]:
        print("  MISMATCH", b)
    return not bad and ro


def verify_all(cfg) -> bool:
    ok = verify_version(cfg)
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for cmd in (["-m", "actspace_build.freeze", "verify", "--version", cfg["merge"]["sources"]["arkitscenes"]["version"]],
                ["-m", "actspace_build.scannetpp_freeze", "verify"]):
        r = subprocess.run([py, *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr.strip()[-500:])
        ok &= r.returncode == 0
    return ok


# ---------------------------------------------------------------- comparison with the dry run

def compare(cfg) -> dict:
    V, D = cfg["merge"]["version"], cfg["merge"]["dryrun_version"]
    sb = sandbox(cfg) / "merged" / "work" / "merged"
    md = mdir(cfg)
    out = {"step8_input_identical": (sb / "_step7" / "test_groups.jsonl").read_bytes() == (md / "_step7" / "test_groups.jsonl").read_bytes(),
           "step8_output_identical": (sb / "_step8" / D / "test_groups.jsonl").read_bytes() == (md / "_step8" / f"{V}_step8" / "test_groups.jsonl").read_bytes()}
    sim = read_json(sandbox(cfg) / "results" / "analyze.json")["quadrant_rebalance_simulation"]
    man = read_json(vdir(cfg) / "manifest.json")
    sim_q = {r["qid"] for r in sim.get("removed", [])}
    got_q = {r["qid"] for r in man["stages"]["quadrant"]}
    out.update(quadrant_removed=sorted(got_q), quadrant_removed_matches_dryrun=sim_q == got_q if sim_q else None,
               dryrun_simulated_n=sim["n_removed"], formal_n=len(got_q))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "integrity", "freeze", "verify", "compare"])
    args = ap.parse_args()
    cfg = load_merge_config()
    if args.cmd == "build":
        r = build(cfg)
    elif args.cmd == "integrity":
        r = integrity_check(cfg)
    elif args.cmd == "freeze":
        r = freeze_version(cfg)
    elif args.cmd == "compare":
        r = compare(cfg)
    else:
        sys.exit(0 if verify_all(cfg) else 1)
    write_json(sandbox(cfg) / "results" / f"formal_{args.cmd}.json", r)
    if args.cmd == "integrity" and not r["all_pass"]:
        sys.exit(1)
    if args.cmd == "build" and r["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
