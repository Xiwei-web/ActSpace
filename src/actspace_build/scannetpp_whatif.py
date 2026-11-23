"""Read-only what-if analyses (sandboxed; no official output is modified).

- rho: ScanNet++ visible_min_ratio 0.30 -> 0.40. Raising ρ can only downgrade visible to partial, so setting visible pairs with
  ρ < 0.40 to partial in the official classification is equivalent to rerunning Step 4; Steps 5-6 are then rerun to compare question
  counts, and the pairs among the 50 of full-run spot-check round 1 that would flip are listed.
- t5_right: how many extra T5 questions ScanNet++ and ARKit would each get with a T5 "right" variant ("Among the objects on your right,
  which one is closest to you?", rules symmetric to left; reported only). R1 = use the right side when the left side is invalid (same
  dedup key); R2 = left and right are separate questions (T5 dedup key includes the direction).
- motion: effect of the motion-blur condition (fast camera motion) on question counts.

Sandbox root: <parent of paths.work>/<scannetpp.whatif_work_root>/<dataset>/; required files are copied before each run.
Usage: python -m actspace_build.scannetpp_whatif rho|t5_right|motion
"""
from __future__ import annotations

import argparse
import copy
import itertools
import shutil

import numpy as np
import pandas as pd

from . import geometry as g
from . import step56_questions as s56
from .common import load_config, path_of, read_json, work_dir, write_json
from .scannetpp import sp_cfg, sp_work
from .scannetpp_step56 import config_for_mode, primary_mode
from .step4_graph_visibility import STATE_PARTIAL, STATE_VISIBLE

FILES = ("visibility.npz", "view_graph.json", "frames_all.json", "objects.json", "scene_meta.json", "keyframes.json",
         "step3_summary.json", "step4_summary.json")
T5_RIGHT_TEXT = "Imagine you are standing at the {OBJ1} and facing the {OBJ2}. Among the objects on your right, which one is closest to you?"


def sandbox_cfg(cfg, dataset, scenes, state_fn=None):
    """Copy scene files into the sandbox; state_fn(sid, vis_dict) may modify the classification. Returns a config copy pointing at the sandbox."""
    root = path_of(cfg, "work").parent / sp_cfg(cfg)["whatif_work_root"]
    for sid in scenes:
        src, dst = work_dir(cfg, dataset, sid), root / dataset / sid
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        if state_fn is not None:
            v = dict(np.load(src / "visibility.npz"))
            v["state"] = state_fn(sid, v)
            np.savez_compressed(dst / "visibility.npz", **v)
    c = copy.deepcopy(cfg)
    c["paths"]["work"] = str(root)
    return c


def sp_scenes(cfg):
    """Same order as the official run (scannetpp_step56 sorts scenes): option-letter balancing, the T2:pair cap, etc. work across scenes, so scene order affects the result."""
    return sorted(l.strip() for l in open(sp_work(cfg) / "_full" / "passed_scenes.txt") if l.strip())


def summarize(st):
    return {"num_groups": st["num_groups"], "num_questions": st["num_questions"], "per_template": st["per_template"], "evidence_label": st["evidence_label"]}


# ---------------------------------------------------------------- ρ 0.40
def rho(cfg, new_rho=0.40) -> dict:
    scenes = sp_scenes(cfg)
    base = read_json(sp_work(cfg) / "_step6" / "step6_stats.json")
    flips = {}

    def demote(sid, v):
        st = v["state"].copy()
        d = (st == STATE_VISIBLE) & (v["rho"].astype(np.float32) < new_rho)
        st[d] = STATE_PARTIAL
        flips[sid] = int(d.sum())
        return st
    c = config_for_mode(sandbox_cfg(cfg, "scannetpp", scenes, demote), primary_mode(cfg))
    st = s56.run(c, "scannetpp", scenes, out_dir=path_of(c, "work") / "scannetpp" / "_step6_rho")
    # pairs among the 50 of full-run spot-check round 1 that would flip
    pairs = pd.read_csv(sp_work(cfg) / "_full" / "spotcheck_round1_pairs.csv", dtype={"pair": str})
    j = pd.read_csv(sp_work(cfg) / "_full" / "spotcheck_round1_judgments.tsv", sep="\t", dtype={"pair": str}, keep_default_na=False)
    pj = pairs.merge(j[["pair", "ok"]], on="pair")
    flip_pairs = [{"pair": r.pair, "scene_id": r.scene_id, "name": r.name, "rho": r.rho, "ok_round1": r.ok}
                  for r in pj.itertuples() if r.rho < new_rho]
    res = {"new_rho": new_rho, "baseline": summarize(base), "whatif": summarize(st), "visible_pairs_demoted": flips,
           "question_change_frac": st["num_questions"] / base["num_questions"] - 1, "round1_flips": flip_pairs}
    write_json(sp_work(cfg) / "_full" / "whatif_rho.json", res)
    print({k: v for k, v in res.items() if k != "visible_pairs_demoted"})
    return res


# ---------------------------------------------------------------- motion blur
def motion(cfg, variants=(("ang>=73", 73.0, None), ("lin>=0.9", None, 0.9), ("ang>=73 or lin>=0.9", 73.0, 0.9))) -> dict:
    """Set visible to partial at fast-motion nodes in the official classification (equivalent to rerunning Step 4), rerun Steps 5-6 and compare with the official result. Scenes are ordered as in the official run."""
    from .scannetpp_step4 import motion_fast_nodes
    scenes = sp_scenes(cfg)
    base = read_json(sp_work(cfg) / "_step6" / "step6_stats.json")
    res = {"baseline": summarize(base), "variants": {}}
    for name, max_ang, max_lin in variants:
        cnt = {"fast_nodes": 0, "visible_demoted": 0}

        def demote(sid, v):
            st = v["state"].copy()
            fast = motion_fast_nodes(v["cam_ang_deg_s"], v["cam_lin_m_s"], max_ang, max_lin)
            d = (st == STATE_VISIBLE) & fast[None, :]
            st[d] = STATE_PARTIAL
            cnt["fast_nodes"] += int(fast.sum())
            cnt["visible_demoted"] += int(d.sum())
            return st
        c = config_for_mode(sandbox_cfg(cfg, "scannetpp", scenes, demote), primary_mode(cfg))
        st = s56.run(c, "scannetpp", scenes, out_dir=path_of(c, "work") / "scannetpp" / "_step6_motion")
        res["variants"][name] = {**summarize(st), **cnt, "question_change_frac": st["num_questions"] / base["num_questions"] - 1}
        print(name, res["variants"][name]["num_questions"], f"{res['variants'][name]['question_change_frac']:+.1%}", cnt, flush=True)
    write_json(sp_work(cfg) / "_full" / "whatif_motion.json", res)
    return res


# ---------------------------------------------------------------- T5 right
def q_T5_right(sc, a, b, opts, qc):
    """Mirror of q_T5: asks for the closest object on the right (theta < 0)."""
    if sc.dist(a, b) < qc["min_standpoint_distance_m"] or any(sc.dist(a, o) < qc["min_standpoint_distance_m"] for o in opts):
        return None
    th = {o: g.direction_theta_facing(sc.center[a], sc.center[b], sc.center[o]) for o in opts}
    if any(min(abs(t), 180.0 - abs(t)) < qc["angle_margin_deg"] for t in th.values()):
        return None
    right = sorted((sc.dist(a, o), o) for o in opts if th[o] < 0)
    if not right or len(right) == len(opts):
        return None
    if len(right) >= 2 and not s56.rank_margin_ok(right[0][0], right[1][0], qc):
        return None
    return {"template": "T5", "variant": "right", "text_template": T5_RIGHT_TEXT, "slots": {"OBJ1": a, "OBJ2": b},
            "answer_type": "multiple_choice", "option_objs": list(opts), "answer_obj": right[0][1],
            "evidence_objs": [a, b, *opts], "margins": {"theta_deg": {str(o): round(t, 1) for o, t in th.items()}}}


def run_variant(c, dataset, scenes, out_name, mode):
    """mode: base / R1 / R2. Temporarily replaces functions in step56_questions and restores them afterwards."""
    orig_q, orig_fill, orig_key = s56.q_T5, s56.fill_candidates, s56.dedup_key
    try:
        if mode == "R1":
            s56.q_T5 = lambda sc, a, b, opts, qc: orig_q(sc, a, b, opts, qc) or q_T5_right(sc, a, b, opts, qc)
        elif mode == "R2":
            def fill(sc, cl, p0, qc, rng):
                c_ = orig_fill(sc, cl, p0, qc, rng)
                lo5, hi5 = qc["t5_options"]
                for a in cl:
                    for b in [o for o in cl if o != a]:
                        rest = [o for o in sc.referable if o not in (a, b)]
                        for n in range(lo5, min(hi5, len(rest)) + 1):
                            for opts in itertools.combinations(rest, n):
                                q = q_T5_right(sc, a, b, opts, qc)
                                if q:
                                    c_["T5"].append(q)
                return c_

            def key(q, t2_by_answer=False):
                k = orig_key(q, t2_by_answer)
                return k + (q.get("variant", "left"),) if q["template"] == "T5" else k
            s56.fill_candidates, s56.dedup_key = fill, key
        st = s56.run(c, dataset, scenes, out_dir=path_of(c, "work") / dataset / out_name)
    finally:
        s56.q_T5, s56.fill_candidates, s56.dedup_key = orig_q, orig_fill, orig_key
    out = summarize(st)
    out["t5_variants"] = st["per_variant"]
    return out


def t5_right(cfg) -> dict:
    res = {}
    sp_sc = sp_scenes(cfg)
    c_sp = config_for_mode(sandbox_cfg(cfg, "scannetpp", sp_sc), primary_mode(cfg))
    ark = sorted(p.parent.name for p in (path_of(cfg, "work") / "arkitscenes").glob("*/step4_summary.json"))
    c_ar = sandbox_cfg(cfg, "arkitscenes", ark)
    for ds, c, scenes in (("scannetpp", c_sp, sp_sc), ("arkitscenes", c_ar, ark)):
        res[ds] = {m: run_variant(c, ds, scenes, f"_step6_t5_{m}", m) for m in ("base", "R1", "R2")}
        print(ds, {m: (r["num_questions"], r["per_template"].get("T5", 0)) for m, r in res[ds].items()}, flush=True)
    write_json(sp_work(cfg) / "_full" / "whatif_t5_right.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rho", "t5_right", "motion"])
    args = ap.parse_args()
    cfg = load_config()
    {"rho": rho, "t5_right": t5_right, "motion": motion}[args.cmd](cfg)


if __name__ == "__main__":
    main()
