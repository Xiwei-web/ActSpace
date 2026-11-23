"""ARKit + ScanNet++ merge: preparation and evaluation (dry run only: no actual pruning, freezing or packaging).

All trial computations are written under merge.sandbox_root (configs/merge.yaml); the official artifacts under work/, the two
frozen versions and eval_runs/ are not modified. The shared Step 8 (step8_debias) is unchanged: in the sandbox, paths.work /
paths.reports point into the sandbox and the module-level DATASET is switched at run time. Each scene directory in the sandbox only
holds a symlink to the official objects.json; the dropped.jsonl written by Step 8 / this module lands in the sandbox scene directory.

  regress   with the merged code and each dataset's configuration at freeze time, rerun Step 8 in the sandbox from each
            pre-pruning state and compare byte by byte with the frozen test_groups.jsonl
  stats     statistics of the two pre-pruning question sets combined (questions and groups per dataset x template x evidence
            label; option-letter bias; T3/T4 quadrant bias; T4 front/back bias)
  sepfilter standing-point horizontal separation filter (questions.min_horizontal_sep_m, DEGENERATE_SEPARATION), producing the
            merged pre-pruning question set; afterwards option letters are redistributed uniformly per (template, number of
            options) (Step 8 only reassigns letters after removing questions)
  step8     run Step 8 on the merged set after sepfilter (version merge.dryrun_version), without quadrant balancing by removal
            (only simulated); option letters are rebalanced as usual
  analyze   final statistics; the ±5% quadrant balancing by removal is only simulated; single-feature blind test on "source dataset"
  vis       read-only estimate of applying the ScanNet++ visibility conditions to ARKit v0.11 (visibility is not recomputed)

Usage: python -m actspace_build.merge_dryrun regress|stats|sepfilter|step8|analyze|vis
Results are written to <sandbox>/results/<subcommand>.json.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from . import step8_debias
from .common import ROOT, DropLog, load_config, path_of, read_json, write_json

TAGS = {"arkitscenes": "arkit", "scannetpp": "scannetpp"}
DATASET_OF_TAG = {v: k for k, v in TAGS.items()}
TEMPLATES = ("T1", "T2", "T3", "T4", "T5")
POSE_UNVERIFIED = 2   # step1_arkit: 0 good, 1 bad, 2 unverified
LABELS = ("explore", "init", "unanswerable")
QUADS = step8_debias.QUADS
FRONT = step8_debias.FRONT


# ---------------------------------------------------------------- configuration and sandbox

def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def load_merge_config() -> dict:
    """configs/build.yaml + configs/merge.yaml (key-by-key override); sandbox_root and release_root are overridden by sandbox and
    release in configs/paths.yaml; relative paths are relative to the repository root."""
    from .common import abs_path, site_paths
    cfg = load_config()
    cfg = deep_update(cfg, yaml.safe_load(open(ROOT / "configs" / "merge.yaml")))
    sp = site_paths()
    for key, site in (("sandbox_root", "sandbox"), ("release_root", "release")):
        cfg["merge"][key] = str(sp[site]) if site in sp else str(abs_path(cfg["merge"][key]))
    return cfg


def sandbox(cfg: dict) -> Path:
    return Path(cfg["merge"]["sandbox_root"])


def real_work(cfg: dict) -> Path:
    return path_of(cfg, "work")


def sandbox_cfg(base: dict, root: Path) -> dict:
    c = copy.deepcopy(base)
    c["paths"]["work"] = str(root / "work")
    c["paths"]["reports"] = str(root / "reports")
    (root / "reports").mkdir(parents=True, exist_ok=True)
    return c


def frozen_config(commit: str) -> tuple[dict, str]:
    """configs/build.yaml at freeze time (git show); returns (config, SHA-256)."""
    raw = subprocess.run(["git", "show", f"{commit}:configs/build.yaml"], cwd=ROOT, check=True, capture_output=True).stdout
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_path(cfg: dict, ds: str) -> Path:
    return real_work(cfg).parent / cfg["merge"]["sources"][ds]["pre_prune"]


def version_dir(cfg: dict, ds: str) -> Path:
    return real_work(cfg) / ds / "_step8" / cfg["merge"]["sources"][ds]["version"]


def read_groups(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def stage(sb_work: Path, sb_dataset: str, groups: list[dict], real: Path, ds_of_scene) -> None:
    """Sandbox input: <sb_work>/<sb_dataset>/_step7/test_groups.jsonl; each scene directory gets a symlink to the official objects.json."""
    d = sb_work / sb_dataset
    (d / "_step7").mkdir(parents=True, exist_ok=True)
    with open(d / "_step7" / "test_groups.jsonl", "w") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    for sid in sorted({g["scene_id"] for g in groups}):
        vid = sid.split("_", 1)[1]
        sd = d / vid
        sd.mkdir(parents=True, exist_ok=True)
        link = sd / "objects.json"
        if not link.exists():
            link.symlink_to(real / ds_of_scene(sid) / vid / "objects.json")


@contextlib.contextmanager
def dataset_as(name: str):
    old = step8_debias.DATASET
    step8_debias.DATASET = name
    try:
        yield
    finally:
        step8_debias.DATASET = old


def dataset_of_scene(sid: str) -> str:
    return "arkitscenes" if sid.startswith("arkit_") else "scannetpp"


# ---------------------------------------------------------------- regression

def regress(cfg: dict) -> dict:
    from .scannetpp_step78 import step8_cfg as sp_step8_cfg
    out = {}
    for ds in ("arkitscenes", "scannetpp"):
        src = cfg["merge"]["sources"][ds]
        vd = version_dir(cfg, ds)
        fm = read_json(vd / "freeze_manifest.json")
        base, cfg_hash = frozen_config(src["frozen_config_commit"])
        rec_hash = fm["configs_sha256"]["configs/build.yaml"]
        pre = source_path(cfg, ds)
        root = sandbox(cfg) / "regress" / ds
        if root.exists():
            shutil.rmtree(root)
        c = sandbox_cfg(base, root)
        stage(Path(c["paths"]["work"]), ds, read_groups(pre), real_work(cfg), lambda s, ds=ds: ds)
        with dataset_as(ds):
            if ds == "scannetpp":
                step8_debias.run(sp_step8_cfg(c))
            else:
                step8_debias.run(c)
        ver = sp_step8_cfg(c)["debias"]["output_version"] if ds == "scannetpp" else c["debias"]["output_version"]
        new = Path(c["paths"]["work"]) / ds / "_step8" / ver / "test_groups.jsonl"
        frozen = vd / "test_groups.jsonl"
        out[ds] = {"version": ver, "config_commit": src["frozen_config_commit"], "config_sha256": cfg_hash,
                   "config_sha256_recorded_at_freeze": rec_hash, "config_matches_freeze": cfg_hash == rec_hash,
                   "pre_prune": str(pre), "pre_prune_sha256": sha256(pre),
                   "frozen_test_groups_sha256": sha256(frozen), "rerun_test_groups_sha256": sha256(new),
                   "byte_identical": frozen.read_bytes() == new.read_bytes(),
                   "frozen_manifest_removed": len(read_json(vd / "manifest.json")["removed"]),
                   "rerun_manifest_removed": len(read_json(new.parent / "manifest.json")["removed"])}
        print(ds, json.dumps(out[ds], ensure_ascii=False))
    return out


# ---------------------------------------------------------------- statistics

def answer_text(q):
    return step8_debias.answer_text(q)


def tag_groups(groups: list[dict], ds: str) -> list[dict]:
    """Prefix group and question ids with the dataset when merging (both datasets number from g_000000)."""
    t = TAGS[ds]
    out = []
    for g in copy.deepcopy(groups):
        g["source_dataset"] = ds
        g["source_group_id"] = g["group_id"]
        g["group_id"] = f"{t}_{g['group_id']}"
        for q in g["questions"]:
            q["source_qid"] = q["qid"]
            q["qid"] = f"{t}_{q['qid']}"
        for c in g["constraints"]:
            c["qids"] = [f"{t}_{x}" for x in c["qids"]]
        out.append(g)
    return out


def merged_pre_prune(cfg: dict) -> list[dict]:
    return [g for ds in ("arkitscenes", "scannetpp") for g in tag_groups(read_groups(source_path(cfg, ds)), ds)]


def ds_of(g):
    return g.get("source_dataset") or dataset_of_scene(g["scene_id"])


def counts(groups: list[dict]) -> dict:
    """Questions per dataset x template x evidence label; groups per dataset x template (groups containing the template); totals."""
    q = Counter()
    gr = defaultdict(set)
    for g in groups:
        for x in g["questions"]:
            q[(ds_of(g), x["template"], x["evidence_label"])] += 1
            gr[(ds_of(g), x["template"])].add(g["group_id"])
    out = {"questions": {"|".join(k): v for k, v in sorted(q.items())},
           "groups_with_template": {"|".join(k): len(v) for k, v in sorted(gr.items())},
           "groups": dict(Counter(ds_of(g) for g in groups)),
           "n_questions": dict(Counter(ds_of(g) for g in groups for _ in g["questions"])),
           "scenes": dict(Counter(ds_of(g) for g in {g["scene_id"]: g for g in groups}.values()))}
    return out


def letter_bias(groups: list[dict], by_dataset=False) -> dict:
    """Per (template, number of options): maximum absolute difference between the correct-letter shares and uniform (percentage
    points), and the count per letter."""
    b = defaultdict(Counter)
    for g in groups:
        for q in g["questions"]:
            if q["answer_type"] != "numeric":
                key = (ds_of(g),) if by_dataset else ()
                b[(*key, q["template"], len(q["options"]))][q["answer"]] += 1
    out = {}
    for k, c in sorted(b.items()):
        n, m = sum(c.values()), k[-1]
        out["|".join(map(str, k))] = {"n": n, "counts": dict(sorted(c.items())),
                                      "max_dev_pp": round(100 * max(abs(c[step8_debias.LETTERS[i]] / n - 1 / m) for i in range(m)), 1)}
    return out


def quadrant_bias(groups: list[dict], scope=None) -> dict:
    """T3 / T4 quadrant shares (all questions, as in the Step 8 check) and maximum deviation; front share of answerable T4
    questions. scope: restrict to one dataset."""
    out = {}
    for t in ("T3", "T4"):
        qs = [q for g in groups if scope in (None, ds_of(g)) for q in g["questions"] if q["template"] == t]
        c = Counter(answer_text(q) for q in qs)
        n = sum(c.values())
        if not n:
            continue
        r = {"n": n, "shares": {k: round(c[k] / n, 3) for k in QUADS},
             "max_dev_pp": round(100 * max(abs(c[k] / n - 0.25) for k in QUADS), 1)}
        if t == "T4":
            ans = [q for q in qs if q["evidence_label"] != "unanswerable"]
            fr = sum(answer_text(q) in FRONT for q in ans) / max(len(ans), 1)
            r.update(front_n=len(ans), front_share=round(fr, 3), front_dev_pp=round(100 * abs(fr - 0.5), 1))
        out[t] = r
    return out


def describe(groups: list[dict]) -> dict:
    return {"counts": counts(groups), "letters": letter_bias(groups), "letters_by_dataset": letter_bias(groups, True),
            "quadrants": quadrant_bias(groups), "quadrants_by_dataset": {ds: quadrant_bias(groups, ds) for ds in TAGS}}


def stats(cfg: dict) -> dict:
    return describe(merged_pre_prune(cfg))


# ---------------------------------------------------------------- standing-point horizontal separation

def scene_geometry(cfg: dict, sid: str):
    ds, vid = dataset_of_scene(sid), sid.split("_", 1)[1]
    d = real_work(cfg) / ds / vid
    centers = {o["obj_id"]: np.asarray(o["center"][:2], float) for o in read_json(d / "objects.json")}
    cams = {f["frame_id"]: np.asarray(f["position"][:2], float) for f in read_json(d / "frames_all.json")}
    return centers, cams


def separation(q: dict, centers: dict, cam_xy) -> tuple[float, str] | None:
    """Minimum horizontal distance (m) relative to the standing point, and the corresponding pair; None for templates it does not apply to.
    T3 / T5: standing object OBJ1 vs. the facing, queried and option objects; T4: queried object OBJ1 vs. the start camera."""
    t = q["template"]
    if t == "T4":
        o = q["slots"]["OBJ1"]
        return float(np.linalg.norm(centers[o] - cam_xy)), f"START camera–{o}"
    if t not in ("T3", "T5"):
        return None
    s = q["slots"]["OBJ1"]
    others = [v for k, v in q["slots"].items() if k != "OBJ1"] + list(q.get("option_objs") or [])
    return min((float(np.linalg.norm(centers[s] - centers[b])), f"{s}–{b}") for b in dict.fromkeys(others) if b != s)


def sepfilter(cfg: dict) -> dict:
    thr = cfg["questions"]["min_horizontal_sep_m"]
    groups = merged_pre_prune(cfg)
    root = sandbox(cfg) / "merged"
    if root.exists():
        shutil.rmtree(root)
    c = sandbox_cfg(cfg, root)
    sb = Path(c["paths"]["work"])
    stage(sb, "merged", groups, real_work(cfg), dataset_of_scene)   # create the scene directories first (DropLog writes into the sandbox)
    geo = {sid: scene_geometry(cfg, sid) for sid in {g["scene_id"] for g in groups}}
    drops = {sid: DropLog(sb / "merged" / sid.split("_", 1)[1], "merge_sep") for sid in geo}
    kill, reasons, detail = set(), {}, {}
    for gi, g in enumerate(groups):
        centers, cams = geo[g["scene_id"]]
        for qi, q in enumerate(g["questions"]):
            r = separation(q, centers, cams[g["init_frame"]])
            if r is not None and r[0] < thr:
                kill.add((gi, qi))
                reasons[(gi, qi)] = "DEGENERATE_SEPARATION"
                detail[q["qid"]] = {"template": q["template"], "dataset": ds_of(g), "sep_m": round(r[0], 3), "pair": r[1],
                                    "evidence_label": q["evidence_label"], "group_id": g["group_id"]}
    before = copy.deepcopy(groups)
    removed = step8_debias.remove_questions(groups, kill, reasons, drops, cfg["groups"], cfg_keep=True)
    group_ids_after = {g["group_id"] for g in groups}
    dropped_groups = sorted({g["group_id"] for g in before} - group_ids_after)
    lost_constraints = sum(len(g["constraints"]) for g in before) - sum(len(g["constraints"]) for g in groups) \
        - sum(len(g["constraints"]) for g in before if g["group_id"] in dropped_groups)
    letters_before = letter_bias(groups)
    # rebalance option letters after merging: Step 8 only reassigns letters when questions are removed, so redistribute them
    # uniformly per (template, number of options) once after merging (and this filter)
    step8_debias.rebalance_letters(groups, cfg["seed"])
    stage(sb, "merged", groups, real_work(cfg), dataset_of_scene)
    by = Counter((detail.get(q, {}).get("dataset") or DATASET_OF_TAG[q.split("_", 1)[0]], t, r) for q, t, r in removed)
    excl = yaml.safe_load(open(ROOT / "configs" / "eval_exclude.yaml"))
    ex = {f"arkit_{x['qid']}" for x in excl["questions"]}
    out = {"threshold_m": thr, "removed": [{"qid": q, "template": t, "reason": r, **detail.get(q, {})} for q, t, r in removed],
           "removed_by": {"|".join(k): v for k, v in sorted(by.items())},
           "dropped_groups": dropped_groups, "constraints_lost_in_kept_groups": lost_constraints,
           "eval_exclude_12_all_removed": ex <= set(detail), "eval_exclude_missing": sorted(ex - set(detail)),
           "arkit_removed_not_in_eval_exclude": sorted(q for q in detail if q.startswith("arkit_") and q not in ex),
           "letters_before_rebalance": letters_before, "after": describe(groups)}
    return out


# ---------------------------------------------------------------- Step 8 dry run

def dryrun_cfg(cfg: dict) -> dict:
    c = sandbox_cfg(cfg, sandbox(cfg) / "merged")
    c["debias"]["output_version"] = cfg["merge"]["dryrun_version"]
    c["debias"]["keep_group_on_constraint_loss"] = True
    # quadrant balancing by removal is only simulated -> disabled during the run (as for ScanNet++); option letters are rebalanced as usual
    c["questions"]["quadrant_balance_tol"] = 1.0
    c["debias"]["t4_front_back_tol"] = 1.0
    return c


def step8(cfg: dict) -> dict:
    c = dryrun_cfg(cfg)
    with dataset_as("merged"):
        step8_debias.run(c)
    d = Path(c["paths"]["work"]) / "merged" / "_step8" / c["debias"]["output_version"]
    return {"version": c["debias"]["output_version"], "test_groups": str(d / "test_groups.jsonl"),
            "sha256": sha256(d / "test_groups.jsonl"), "manifest": read_json(d / "manifest.json")}


class _NoLog:
    def add(self, *a, **k):
        pass


def quadrant_rebalance(groups: list[dict], cfg: dict, drops=None) -> tuple[list, str]:
    """Quadrant balancing by removal (±quadrant_balance_tol; T4 answerable front/back ±t4_front_back_tol), in place: from the
    over-represented quadrant, remove the non-constraint, non-unanswerable question with the highest bias score (selected as in Step 8's
    RATIO_BALANCE, using the final Step 8 bias_score), reason code QUADRANT_REBALANCE; a group that no longer has 3-5 questions and
    >= 2 templates is dropped entirely. Returns (removal records, status)."""
    removed = []
    drops = drops if drops is not None else defaultdict(_NoLog)
    status = "ok"
    while True:
        v = step8_debias.quadrant_violation(groups, cfg)
        if v is None:
            break
        t, quads = v
        cand = [((q.get("bias_score") or 0.0), (gi, qi)) for gi, g in enumerate(groups) for qi, q in enumerate(g["questions"])
                if q["template"] == t and q["evidence_label"] != "unanswerable" and answer_text(q) in quads
                and not any(q["qid"] in c_["qids"] for c_ in g["constraints"])]
        if not cand:
            status = f"infeasible for {t}"
            break
        it = max(cand)[1]
        removed += step8_debias.remove_questions(groups, {it}, {it: "QUADRANT_REBALANCE"}, drops, cfg["groups"], cfg_keep=True)
    return removed, status


def simulate_quadrant_rebalance(groups: list[dict], cfg: dict) -> dict:
    """Simulate quadrant_rebalance without applying it (runs on a copy)."""
    g2 = copy.deepcopy(groups)
    removed, status = quadrant_rebalance(g2, cfg)
    before_g = {g["group_id"]: g for g in groups}
    after_g = {g["group_id"] for g in g2}
    by = Counter((DATASET_OF_TAG[q.split("_", 1)[0]], t, r) for q, t, r in removed)
    return {"status": status, "removed_by": {"|".join(k): v for k, v in sorted(by.items())}, "n_removed": len(removed),
            "removed": [{"qid": q, "template": t, "reason": r} for q, t, r in removed],
            "groups_dropped": sorted(set(before_g) - after_g),
            "groups_dropped_by_dataset": dict(Counter(ds_of(before_g[x]) for x in set(before_g) - after_g)),
            "after": describe(g2)}


def source_probe(groups: list[dict], cfg: dict) -> dict:
    """Predict the answer from the single feature "source dataset" (unanswerable questions excluded) with a random forest and
    GroupKFold by scene (effective_folds). Multiple choice: predict the correct letter, accuracy vs. chance (mean of 1 / number of
    options); for T3 / T4 also predict the quadrant, vs. the training-fold majority baseline. Numeric: predict the distance, MRA vs.
    the MRA of the training-fold mean."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    n_trees, seed = cfg["merge"]["source_probe_trees"], cfg["seed"]
    out = {}
    for t in TEMPLATES:
        items = [(g, q) for g in groups for q in g["questions"] if q["template"] == t and q["evidence_label"] != "unanswerable"]
        if not items:
            continue
        X = np.array([[1.0 if ds_of(g) == "scannetpp" else 0.0] for g, _ in items])
        sids = np.array([g["scene_id"] for g, _ in items])
        k = step8_debias.effective_folds(len(set(sids)), cfg["debias"]["n_folds"])
        folds = list(GroupKFold(n_splits=k).split(X, groups=sids))
        frac_sp = float(X.mean())
        if items[0][1]["answer_type"] == "numeric":
            y = np.array([q["answer"] for _, q in items], float)
            pred, base = np.zeros_like(y), np.zeros_like(y)
            for tr, te in folds:
                m = RandomForestRegressor(n_estimators=n_trees, random_state=seed).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
                base[te] = y[tr].mean()
            out[t] = {"n": len(y), "scannetpp_frac": round(frac_sp, 3), "mra_source": round(float(step8_debias.mra_each(pred, y).mean()), 3),
                      "mra_train_mean": round(float(step8_debias.mra_each(base, y).mean()), 3),
                      "mean_distance_by_dataset": {ds: round(float(y[X[:, 0] == v].mean()), 2) for ds, v in (("arkitscenes", 0), ("scannetpp", 1))}}
            continue
        y = np.array([q["answer"] for _, q in items])
        chance = float(np.mean([1 / len(q["options"]) for _, q in items]))
        pred = np.empty_like(y)
        for tr, te in folds:
            m = RandomForestClassifier(n_estimators=n_trees, random_state=seed).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        r = {"n": len(y), "scannetpp_frac": round(frac_sp, 3), "letter_acc": round(float((pred == y).mean()), 3), "chance": round(chance, 3)}
        if t in ("T3", "T4"):
            yc = np.array([answer_text(q) for _, q in items])
            pc, bc = np.empty_like(yc), np.empty_like(yc)
            for tr, te in folds:
                pc[te] = RandomForestClassifier(n_estimators=n_trees, random_state=seed).fit(X[tr], yc[tr]).predict(X[te])
                bc[te] = Counter(yc[tr]).most_common(1)[0][0]
            r.update(quadrant_acc=round(float((pc == yc).mean()), 3), quadrant_majority=round(float((bc == yc).mean()), 3))
        out[t] = r
    return out


def analyze(cfg: dict) -> dict:
    c = dryrun_cfg(cfg)
    d = Path(c["paths"]["work"]) / "merged"
    pre = read_groups(d / "_step7" / "test_groups.jsonl")
    final = read_groups(d / "_step8" / c["debias"]["output_version"] / "test_groups.jsonl")
    man = read_json(d / "_step8" / c["debias"]["output_version"] / "manifest.json")
    n0 = Counter((ds_of(g), q["template"]) for g in pre for q in g["questions"])
    n1 = Counter((ds_of(g), q["template"]) for g in final for q in g["questions"])
    reasons = Counter((DATASET_OF_TAG[r["qid"].split("_", 1)[0]], r["template"], r["reason"]) for r in man["removed"])
    prune = {}
    for t in TEMPLATES:
        for ds in (*TAGS, "all"):
            a = sum(v for (d_, t_), v in n0.items() if t_ == t and ds in ("all", d_))
            b = sum(v for (d_, t_), v in n1.items() if t_ == t and ds in ("all", d_))
            prune[f"{ds}|{t}"] = {"before": a, "after": b, "pruned_frac": round(1 - b / a, 3) if a else None}
    un = Counter((ds_of(g), q["evidence_label"] == "unanswerable") for g in final for q in g["questions"])
    unans = {ds: {"unanswerable": un[(ds, True)], "total": un[(ds, True)] + un[(ds, False)],
                  "frac": round(un[(ds, True)] / max(1, un[(ds, True)] + un[(ds, False)]), 3)} for ds in TAGS}
    tot_u = sum(v["unanswerable"] for v in unans.values())
    unans["all"] = {"unanswerable": tot_u, "total": sum(v["total"] for v in unans.values()),
                    "frac": round(tot_u / max(1, sum(v["total"] for v in unans.values())), 3)}
    qcfg = copy.deepcopy(cfg)   # current ±5% rule
    return {"final": describe(final), "pre": describe(pre), "prune": prune,
            "removed_by_reason": {"|".join(k): v for k, v in sorted(reasons.items())},
            "status": man["status"], "rounds": man["rounds"], "unanswerable": unans,
            "quadrant_rebalance_simulation": simulate_quadrant_rebalance(final, qcfg),
            "source_probe": source_probe(final, cfg)}


# ---------------------------------------------------------------- visibility rule differences (read-only estimate)

def _rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def arkit_node_motion(frames: list[dict], nodes: list[str], half_window_s: float) -> dict:
    """Camera angular velocity (deg/s) and linear velocity (m/s) at ARKit nodes, from the two frames closest in time to
    t ± half_window_s (T_wc and timestamp from frames_all); same definition as for ScanNet++ (scannetpp_motion: ±2 frames @ 60 FPS
    ≈ ±1/30 s)."""
    ts = np.array([f["timestamp"] for f in frames], float)
    order = np.argsort(ts)
    ts = ts[order]
    fr = [frames[i] for i in order]
    idx = {f["frame_id"]: i for i, f in enumerate(fr)}
    out = {}
    for n in nodes:
        i = idx[n]
        a = int(np.clip(np.searchsorted(ts, ts[i] - half_window_s), 0, len(ts) - 1))
        b = int(np.clip(np.searchsorted(ts, ts[i] + half_window_s), 0, len(ts) - 1))
        dt = ts[b] - ts[a]
        if dt <= 0:
            out[n] = (None, None)
            continue
        Ta, Tb = np.asarray(fr[a]["T_wc"], float), np.asarray(fr[b]["T_wc"], float)
        out[n] = (_rot_deg(Ta[:3, :3], Tb[:3, :3]) / dt, float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) / dt)
    return out


def vis(cfg: dict) -> dict:
    """ARKit v0.11: with the ScanNet++ visibility conditions, how many visible (object, node) pairs would be downgraded and how many
    questions would lose their oracle evidence. Read-only; visibility is not recomputed."""
    sp = cfg["scannetpp"]
    rho_thr = sp["visible_min_ratio"]
    ang_thr, perr_ratio, perr_lin = sp["motion_max_ang_deg_s"], sp["pose_err_ratio_max"], sp["pose_err_min_lin_m_s"]
    half = sp["motion_half_window_frames"] / 60.0
    groups = read_groups(version_dir(cfg, "arkitscenes") / "test_groups.jsonl")
    V = 2
    pairs = Counter()
    down = defaultdict(set)   # condition -> {(sid, obj, node)}
    per_scene = {}
    for sid in sorted({g["scene_id"] for g in groups}):
        vid = sid.split("_", 1)[1]
        d = real_work(cfg) / "arkitscenes" / vid
        z = np.load(d / "visibility.npz")
        objs, nodes, st, rho = z["obj_ids"], list(z["frame_ids"]), z["state"], z["rho"].astype(float)
        frames = read_json(d / "frames_all.json")
        mot = arkit_node_motion(frames, nodes, half)
        pc = np.load(d / "pose_check.npz")
        ts_of = {f["frame_id"]: f["timestamp"] for f in frames}
        err = pc["err"].astype(float)
        ok = np.isfinite(err) & (pc["status"] != POSE_UNVERIFIED)   # GATE A per-pose depth error; unverified poses have no error
        node_err = {}
        for n in nodes:
            j = int(np.argmin(np.abs(pc["traj_ts"] - ts_of[n])))
            node_err[n] = float(err[j]) if ok[j] else np.nan
        ne = np.array([v for v in node_err.values() if np.isfinite(v)])
        med = float(np.median(ne)) if len(ne) else np.nan   # as for ScanNet++: median error over the scene's nodes
        per_scene[sid] = {"nodes": len(nodes), "median_pose_err_m": med}
        for i, o in enumerate(objs):
            for j, n in enumerate(nodes):
                if st[i, j] != V:
                    continue
                key = (sid, int(o), n)
                pairs["visible"] += 1
                if rho[i, j] < rho_thr:
                    down["rho"].add(key)
                ang, lin = mot[n]
                if ang is not None and ang >= ang_thr:
                    down["motion"].add(key)
                if lin is not None and np.isfinite(node_err[n]) and np.isfinite(med) and node_err[n] >= perr_ratio * med and lin >= perr_lin:
                    down["pose_offset"].add(key)
    allset = set().union(*down.values()) if down else set()
    downs = {k: len(v) for k, v in down.items()}
    downs["any_of_computable"] = len(allset)

    def q_effect(cond_set):
        """Effect on questions: an evidence object of an answerable question is no longer visible at p0 + oracle_nodes (oracle
        evidence fails); an evidence object is no longer visible at any node (would become AMBIGUOUS_EVIDENCE under the generation
        rules); an init question's evidence is no longer all visible at p0. Masked nodes of masked groups are not counted."""
        res = Counter()
        bad_q = []
        for g in groups:
            sid = g["scene_id"]
            vid = sid.split("_", 1)[1]
            z = np.load(real_work(cfg) / "arkitscenes" / vid / "visibility.npz")
            oi = {int(o): i for i, o in enumerate(z["obj_ids"])}
            ni = {n: j for j, n in enumerate(z["frame_ids"])}
            st = z["state"]
            masked = set((g.get("view_graph_mask") or {}).get("masked_nodes") or [])

            def vis_ok(o, n):
                return st[oi[o], ni[n]] == V and (sid, o, n) not in cond_set
            for q in g["questions"]:
                if q["evidence_label"] == "unanswerable":
                    continue
                res["answerable"] += 1
                ev = q["evidence_objs"]
                seen = [g["init_frame"], *(q.get("oracle_nodes") or [])]
                oracle_fail = any(not any(vis_ok(o, n) for n in seen) for o in ev)
                never = any(not any(vis_ok(o, n) for n in ni if n not in masked) for o in ev)
                init_fail = q["evidence_label"] == "init" and not all(vis_ok(o, g["init_frame"]) for o in ev)
                res["oracle_evidence_fails"] += oracle_fail
                res["evidence_never_visible"] += never
                res["init_label_fails"] += init_fail
                if oracle_fail:
                    bad_q.append(q["qid"])
        return dict(res), bad_q
    effects = {}
    for k, s in [*down.items(), ("any_of_computable", allset)]:
        e, bad = q_effect(s)
        effects[k] = {**e, "qids_oracle_fail_sample": bad[:20], "by_template": dict(Counter(
            q["template"] for g in groups for q in g["questions"] if q["qid"] in set(bad)))}
    return {"thresholds": {"rho": rho_thr, "motion_max_ang_deg_s": ang_thr, "pose_err_ratio_max": perr_ratio,
                           "pose_err_min_lin_m_s": perr_lin, "half_window_s": round(half, 4)},
            "visible_pairs": pairs["visible"], "downgraded": downs, "question_effects": effects,
            "scenes": len(per_scene), "groups": len(groups), "questions": sum(len(g["questions"]) for g in groups)}


# ---------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regress", "stats", "sepfilter", "step8", "analyze", "vis"])
    args = ap.parse_args()
    cfg = load_merge_config()
    res = {"regress": regress, "stats": stats, "sepfilter": sepfilter, "step8": step8, "analyze": analyze, "vis": vis}[args.cmd](cfg)
    out = sandbox(cfg) / "results" / f"{args.cmd}.json"
    write_json(out, res)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
