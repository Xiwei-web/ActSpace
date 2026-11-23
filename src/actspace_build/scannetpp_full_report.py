"""Deliverables after the full ScanNet++ run.

- whatif_area: option B (visibility area threshold 0.02 -> 0.01) recomputed on the real data: copy each scene's Step 4 outputs into a sandbox
  work directory (scannetpp.whatif_work_root), re-classify visible from the raw values in visibility.npz (mirror / magenta occlusion are
  already in the raw values; the extra shape / sharpness conditions apply as usual), then rerun Steps 5-6. Official outputs are not modified.
- spotcheck_images / spotcheck: full-run visibility spot check (new seed, 50 pairs, at least half from oracle / link views actually used by
  questions, covering as many scenes as possible).
- report: reports/scannetpp_full_summary.md, one card per scene, a question-preview jsonl, and a list of suspected mirrors with images.

Usage: python -m actspace_build.scannetpp_full_report whatif_area|spotcheck_images|spotcheck|report
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from . import arkit
from . import geometry as g
from .common import load_config, path_of, read_json, scene_rng, write_json
from .scannetpp import read_split, sp_cfg, sp_root, sp_work
from .scannetpp_gateC import ARKIT_V011, TEMPLATES, LABELS, card, fmt, scene_data
from .scannetpp_step4 import face_corners, mirror_face
from .scannetpp_step4_report import base_image, box_of, put
from .scannetpp_step56 import config_for_mode, primary_mode
from .step1_arkit import draw_obbs
from .step4_graph_visibility import STATE_INVISIBLE, STATE_PARTIAL, STATE_VISIBLE
from .step56_questions import run, scene_id_of

DATASET = "scannetpp"
SPOT_N = 50
SPOT_SEED = ("scannetpp_full_visibility_spotcheck", "spotcheck_scannetpp_full_round5")  # final confirmation spot check (rounds 1-4 used ..._round1 ... _round4)
S19_SCENES = ()  # full-run round 2 required >= 1 pair from each of the 4 re-admitted scenes ("25f3b7a318", "bde1e479ad", "21d970d8de", "cc5237fd77"); not required from round 3 on
STEP4_FILES = ("visibility.npz", "view_graph.json", "frames_all.json", "objects.json", "scene_meta.json", "keyframes.json",
               "step3_summary.json", "step4_summary.json")


def full_dir(cfg):
    d = sp_work(cfg) / "_full"
    d.mkdir(parents=True, exist_ok=True)
    return d


def passed_scenes(cfg) -> list[str]:
    return [l.strip() for l in open(full_dir(cfg) / "passed_scenes.txt") if l.strip()]


def step6_scenes(cfg) -> list[str]:
    """Scenes that pass GATE A and all Step 2-4 scene-level checks (the scene set actually used by Steps 5-6; same test as step56_questions.scene_eligible)."""
    from .step56_questions import scene_eligible
    return [sid for sid in passed_scenes(cfg) if (sp_work(cfg) / sid / "step4_summary.json").exists() and not scene_eligible(cfg, DATASET, sid)]


# ---------------------------------------------------------------- option B: sandbox recomputation
def reclassify(v, vc, sp, area_min) -> np.ndarray:
    """Re-classify from the raw values in visibility.npz using the three-state rule + the extra shape / sharpness conditions (only the area threshold changes)."""
    vis = ((v["rho"] >= vc["visible_min_ratio"]) & (v["area_frac"] >= area_min) & (v["n_visible"] >= vc["visible_min_points"])
           & (v["completeness"] >= vc["visible_min_completeness"]) & (v["in_frame_frac"] >= vc["visible_min_in_frame_frac"])
           & (v["min_rect_short_px"] >= sp["vis_min_short_side_px"]) & (v["sharpness_ratio"] >= sp["vis_min_sharpness_ratio"]))
    inv = (v["n_infrustum"] == 0) | (v["rho"] < vc["invisible_max_ratio"])
    st = np.full(v["state"].shape, STATE_PARTIAL, np.uint8)
    st[inv] = STATE_INVISIBLE
    st[vis] = STATE_VISIBLE
    return st


def whatif_area(cfg, area_min=0.01) -> dict:
    sp = sp_cfg(cfg)
    root = path_of(cfg, "work").parent / sp["whatif_work_root"]
    scenes = step6_scenes(cfg)
    sw = root / sp["work_subdir"]
    base_check = {}
    for sid in scenes:
        src, dst = sp_work(cfg) / sid, sw / sid
        dst.mkdir(parents=True, exist_ok=True)
        for f in STEP4_FILES:
            shutil.copy2(src / f, dst / f)
        v = dict(np.load(src / "visibility.npz"))
        base_check[sid] = int((reclassify(v, cfg["visibility"], sp, cfg["visibility"]["visible_min_area_frac"]) != v["state"]).sum())
        v["state"] = reclassify(v, cfg["visibility"], sp, area_min)
        np.savez_compressed(dst / "visibility.npz", **v)
    c = copy.deepcopy(cfg)
    c["paths"]["work"] = str(root)
    c = config_for_mode(c, primary_mode(cfg))
    st = run(c, DATASET, scenes, out_dir=sw / "_step6")
    usable = {}
    for sid in scenes:
        objs = read_json(sw / sid / "objects.json")
        v = np.load(sw / sid / "visibility.npz")
        usable[sid] = sum(1 for k, o in enumerate(objs) if o["referable"] and (v["state"][k] == STATE_VISIBLE).any())
    res = {"area_min": area_min, "sandbox": str(sw), "num_groups": st["num_groups"], "num_questions": st["num_questions"],
           "per_template": st["per_template"], "evidence_label": st["evidence_label"], "usable_per_scene": usable,
           "reclassify_mismatch_at_current_threshold": base_check}
    write_json(full_dir(cfg) / "whatif_area.json", res)
    print({k: v for k, v in res.items() if k != "usable_per_scene"})
    return res


# ---------------------------------------------------------------- full-run visibility spot check
def spotcheck_images(cfg) -> None:
    scenes = step6_scenes(cfg)
    groups = [json.loads(l) for l in open(sp_work(cfg) / "_step6" / "test_groups.jsonl")]
    out_dir = path_of(cfg, "reports") / "scannetpp_full_spotcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    rng = scene_rng(cfg, *SPOT_SEED)
    pool_or, pool_rd = {}, {}
    data = {}
    for sid in scenes:
        wd = sp_work(cfg) / sid
        objs, v = read_json(wd / "objects.json"), np.load(wd / "visibility.npz")
        frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
        nodes = [str(x) for x in v["frame_ids"]]
        data[sid] = (objs, v, frames, nodes)
        row = {o["obj_id"]: k for k, o in enumerate(objs)}
        used = set()
        for grp in groups:
            if grp["scene_id"] != scene_id_of(DATASET, sid):
                continue
            for q in grp["questions"]:
                for n in (q.get("oracle_nodes") or []) + (q.get("link_views") or []):
                    for o in q["evidence_objs"]:
                        if objs[row[o]]["referable"] and v["state"][row[o], nodes.index(n)] == STATE_VISIBLE:
                            used.add((sid, row[o], nodes.index(n)))
        allv = {(sid, int(r), int(c)) for r, c in zip(*np.where(v["state"] == STATE_VISIBLE)) if objs[r]["referable"]}
        if used:
            pool_or[sid] = sorted(used)
        rest = sorted(allv - used)
        if rest:
            pool_rd[sid] = rest

    def draw(pools, n, avoid_scenes):
        """Round-robin sampling over scenes: each round shuffles the scenes with pairs left and draws 1 pair per scene; scenes outside avoid_scenes go first."""
        picks, pools = [], {k: list(v) for k, v in pools.items()}
        while len(picks) < n and any(pools.values()):
            order = [s for s in rng.permutation(sorted(pools)) if pools[s]]
            order = [s for s in order if s not in avoid_scenes] + [s for s in order if s in avoid_scenes]
            for s in order:
                if len(picks) >= n:
                    break
                i = int(rng.integers(len(pools[s])))
                picks.append(pools[s].pop(i))
            avoid_scenes = set()
        return picks
    # draw 1 pair from each re-admitted scene first (oracle / link views preferred)
    pre_or, pre_rd = [], []
    for s_ in S19_SCENES:
        if pool_or.get(s_):
            pre_or.append(pool_or[s_].pop(int(rng.integers(len(pool_or[s_])))))
        elif pool_rd.get(s_):
            pre_rd.append(pool_rd[s_].pop(int(rng.integers(len(pool_rd[s_])))))
    n_or = SPOT_N // 2
    p_or = pre_or + draw(pool_or, n_or - len(pre_or), {p[0] for p in pre_or + pre_rd})
    p_rd = pre_rd + draw(pool_rd, SPOT_N - len(p_or) - len(pre_rd), {p[0] for p in p_or + pre_rd})
    picks = [(p, "oracle/link") for p in p_or] + [(p, "random") for p in p_rd]
    picks = [picks[i] for i in rng.permutation(len(picks))]
    rows = []
    for k, ((sid, r, c), src) in enumerate(picks, 1):
        objs, v, frames, nodes = data[sid]
        o, fid = objs[r], nodes[c]
        img, K, T = base_image(cfg, frames[fid], 1280)
        draw_obbs(img, [box_of(o, o["ref_name"])], K, T, np.eye(3), color=(0, 220, 0))
        put(img, f"#{k:02d} {sid} {fid}   object: {o['ref_name']}", (10, 30), 0.8)
        put(img, f"rho={float(v['rho'][r, c]):.2f} completeness={float(v['completeness'][r, c]):.2f} area={float(v['area_frac'][r, c]):.3f} "
                 f"in_frame={float(v['in_frame_frac'][r, c]):.2f}", (10, img.shape[0] - 14), 0.6)
        cv2.imwrite(str(out_dir / f"pair_{k:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({"pair": f"{k:02d}", "scene_id": sid, "frame_id": fid, "obj_id": o["obj_id"], "name": o["ref_name"], "source_label": o["source_label"],
                     "rho": round(float(v["rho"][r, c]), 2), "completeness": round(float(v["completeness"][r, c]), 2),
                     "area_frac": round(float(v["area_frac"][r, c]), 4), "in_frame_frac": round(float(v["in_frame_frac"][r, c]), 2), "source": src})
    pd.DataFrame(rows).to_csv(full_dir(cfg) / "spotcheck_pairs.csv", index=False)
    print(f"oracle/link pool: {sum(map(len, pool_or.values()))} in {len(pool_or)} scenes; random pool: {sum(map(len, pool_rd.values()))};",
          f"picked {len(p_or)} oracle/link + {len(p_rd)} random from {len({r['scene_id'] for r in rows})} scenes")


def spotcheck_section(cfg) -> list[str]:
    p = full_dir(cfg) / "spotcheck_pairs.csv"
    j = path_of(cfg, "reports") / "scannetpp_full_spotcheck_judgments.tsv"
    if not (p.exists() and j.exists()):
        return ["## Full-run visibility spot check", "", "(not done yet)", ""]
    df = pd.read_csv(p, dtype={"pair": str, "scene_id": str}).merge(pd.read_csv(j, sep="\t", dtype={"pair": str}, keep_default_na=False), on="pair")
    n_ok = int((df.ok == "yes").sum())
    fails = df[df.ok != "yes"]
    s19 = dict(Counter(r for r in df.scene_id if r in S19_SCENES))
    s19_txt = f" (at least 1 pair from each of the 4 re-admitted scenes: {s19})" if S19_SCENES else ""
    L = ["## 7. Full-run visibility spot check", "",
         f"Final confirmation spot check (full-run round 5, after the last category-rule change), new seed (`scene_rng{SPOT_SEED}`); only visible pairs of referable, non-blacklisted objects are sampled, {len(df)} pairs in total: "
         f"{int((df.source == 'oracle/link').sum())} pairs from oracle / link views actually used by questions, {int((df.source == 'random').sum())} random; "
         f"round-robin over scenes, covering {df.scene_id.nunique()} scenes{s19_txt}. One image per pair (original image at 1280 px width + object box + referring name). Each pair is judged on two items: (1) can a person recognize the object and roughly tell its position; (2) does the name used in questions match the object in the photo (is the boxed thing called this name). "
         "A pair passes only if both are \"yes\"; borderline cases count as \"no\".", "",
         f"**Result: {n_ok} / {len(df)} ({n_ok / len(df):.0%})**, required >= 95%: {'**passed**' if n_ok / len(df) >= 0.95 else '**failed** (stopped as required; thresholds not tuned)'}.", ""]
    two = "recognizable" in df.columns
    if two:
        L += [f"- (1) recognizable with rough position: {int((df.recognizable == 'yes').sum())} / {len(df)}; (2) name matches object: {int((df.name_match == 'yes').sum())} / {len(df)}.", ""]
    if len(fails):
        L += ["| # | Scene | Frame | Object | Source | ρ | Completeness | Area | In frame | " + ("(1) Recognized | (2) Name | " if two else "") + "Type | Note |",
              "|---" * (11 + 2 * two) + "|",
              *[f"| {r.pair} | `{r.scene_id}` | {r.frame_id} | {r.name} ({r.source_label}) #{r.obj_id} | {r.source} | {r.rho} | {r.completeness} | {r.area_frac} | "
                f"{r.in_frame_frac} | " + (f"{r.recognizable} | {r.name_match} | " if two else "") + f"{r.failure_type} | {r.note} |" for r in fails.itertuples()], ""]
    if "unsure" in df.columns and (df.unsure == "yes").any():
        L += ["### Uncertain samples (for manual review)", ""]
        for r in df[df.unsure == "yes"].itertuples():
            L += [f"- **#{r.pair}** `{r.scene_id}` {r.frame_id} {r.name} ({r.source_label}) #{r.obj_id}: judged {'pass' if r.ok == 'yes' else 'fail'}. {r.note}  ",
                  f"  ![#{r.pair}](scannetpp_full_spotcheck/pair_{r.pair}.jpg)"]
        L += [""]
    L += ["<details><summary>Judgments for all 50 pairs</summary>", "", "| # | Source | Scene | Object | ρ | Completeness | Area | Recognizable | Note |", "|---|---|---|---|---|---|---|---|---|",
          *[f"| {r.pair} | {r.source} | `{r.scene_id}` | {r.name} #{r.obj_id} | {r.rho} | {r.completeness} | {r.area_frac} | {'yes' if r.ok == 'yes' else '**no**'} | {r.note} |"
            for r in df.itertuples()], "", "</details>", "",
          "Spot-check images: `reports/scannetpp_full_spotcheck/pair_XX.jpg` (not in git).", ""]
    review = full_dir(cfg) / "spotcheck_review.md"
    if review.exists():
        L += [review.read_text().rstrip("\n"), ""]
    return L


def v02_section(cfg) -> list[str]:
    """scannetpp_only_v0.2 (final version after Step 8): comparison with v0.1, total with ARKit v0.11, gap to 200 per template, human baseline."""
    from collections import Counter as C
    W = sp_work(cfg) / "_step8"
    if not (W / "scannetpp_only_v0.2" / "test_groups.jsonl").exists():
        return []

    def load(v):
        gs = [json.loads(l) for l in open(W / v / "test_groups.jsonl")]
        qs = [q for g_ in gs for q in g_["questions"]]
        return {"groups": len(gs), "questions": len(qs), "t": C(q["template"] for q in qs), "l": C(q["evidence_label"] for q in qs),
                "scenes": len({g_["scene_id"] for g_ in gs})}
    v1, v2 = load("scannetpp_only_v0.1"), load("scannetpp_only_v0.2")
    tot = ARKIT_V011["questions"] + v2["questions"]
    L = ["## 6a. Final version scannetpp_only_v0.2 (after Step 8)", "",
         "v0.2 = after the latest blacklist and category-rule changes, Step 7 (same paraphrase bank) and Step 8 (same rules as v0.1) rerun from the pre-pruning state. Integrity check: `scannetpp_only_v0.2_integrity_check.md`.", "",
         "| Version | Scenes | Groups | Questions | " + " | ".join(TEMPLATES) + " | " + " | ".join(LABELS) + " | Unanswerable share |",
         "|---" * (5 + len(TEMPLATES) + len(LABELS)) + "|"]
    for name, x in (("v0.1 (GATE D verified version)", v1), ("**v0.2 (final)**", v2)):
        L.append(f"| {name} | {x['scenes']} | {x['groups']} | {x['questions']} | " + " | ".join(str(x["t"][t]) for t in TEMPLATES) + " | "
                 + " | ".join(str(x["l"][l]) for l in LABELS) + f" | {x['l']['unanswerable'] / max(x['questions'], 1):.1%} |")
    L.append(f"| Change | {v2['scenes'] - v1['scenes']:+d} | {v2['groups'] - v1['groups']:+d} | {v2['questions'] - v1['questions']:+d} | "
             + " | ".join(f"{v2['t'][t] - v1['t'][t]:+d}" for t in TEMPLATES) + " | " + " | ".join(f"{v2['l'][l] - v1['l'][l]:+d}" for l in LABELS) + " | |")
    L += ["", "| | Total questions | " + " | ".join(TEMPLATES) + " |", "|---" * (len(TEMPLATES) + 2) + "|",
          f"| ScanNet++ v0.2 | {v2['questions']} | " + " | ".join(str(v2["t"][t]) for t in TEMPLATES) + " |",
          f"| ARKit v0.11 (frozen) | {ARKIT_V011['questions']} | " + " | ".join(str(ARKIT_V011[t]) for t in TEMPLATES) + " |",
          f"| **Total** | **{tot}** | " + " | ".join(f"**{ARKIT_V011[t] + v2['t'][t]}**" for t in TEMPLATES) + " |",
          "| Short of 200 per template | | " + " | ".join(str(max(0, 200 - ARKIT_V011[t] - v2["t"][t])) for t in TEMPLATES) + " |", "",
          f"- **Total {tot} questions" + (", below the lower bound of 1000" if tot < 1000 else ", reaching the lower bound of 1000") + f"**; ScanNet++ share {v2['questions'] / tot:.0%}."
          + (" Reported as is; rules were not relaxed to reach the count." if tot < 1000 else ""),
          "- Both are post-Step 8 numbers; after merging, Step 8 is rerun from each pre-pruning state and the numbers will change again.", ""]
    bp = path_of(cfg, "reports") / "gateD" / "scannetpp" / "baseline_v0.2.md"
    if bp.exists():
        t = bp.read_text().split("\n", 2)[2]
        L += ["### GATE D human baseline (recomputed after removing questions involving blacklisted objects)", "", t.strip(), ""]
    return L


def round_changes_section(cfg, st, ga) -> list[str]:
    """Comparison with earlier full-run rounds, the impact of the newer rules, and the sandbox evaluations (ρ 0.40, T5 right)."""
    fd = full_dir(cfg)
    L = []
    hist = [(n, fd / f"step6_stats_{k}.json") for n, k in (("Full run, round 1", "round1"), ("Full run, round 2", "round2"), ("Full run, round 3", "round3"), ("Round 4 (source of v0.1)", "round4"))]
    hist = [(n, read_json(p_)) for n, p_ in hist if p_.exists()]
    if hist:
        r1 = hist[-1][1]
        L += ["## 6b. Comparison with earlier rounds", "",
              "- Round 2: 2 objects blacklisted; visible downgraded to partial when > 30% of in-frustum points fall on magenta pixels; ScanNet++ ρ threshold 0.40; GATE A scene retention uses only the per-pose rule (4 scenes re-admitted).",
              f"- Round 3: 2 more objects blacklisted (`c50d2d1d42` window blind #64, `e7af285f7d` shelf #104); new motion-blur condition"
              f" (nodes with camera angular speed >= {sp_cfg(cfg).get('motion_max_ang_deg_s')}°/s cannot serve as visible evidence).",
              f"- **Round 4 (after visibility spot-check round 3 had passed)**: 2 more built-in appliances blacklisted (`3db0a1c8f3` refrigerator #36, `3864514494` dishwasher #18); "
              f"pose lateral-offset condition: at nodes whose per-pose depth error >= scene median node error x {sp_cfg(cfg).get('pose_err_ratio_max')} and camera linear speed >= {sp_cfg(cfg).get('pose_err_min_lin_m_s')} m/s, "
              f"visible is downgraded to partial for all objects: {sum(read_json(sp_work(cfg) / s_ / 'step4_summary.json').get('pose_suspect', {}).get('nodes', 0) for s_ in step6_scenes(cfg))} nodes and "
              f"{sum(read_json(sp_work(cfg) / s_ / 'step4_summary.json').get('pose_suspect', {}).get('visible_downgraded', 0) for s_ in step6_scenes(cfg))} (object, node) pairs in the full run. "
              "Both changes only remove visible pairs, so the spot-check conclusion is unaffected and the spot check was not redone.",
              "- **This round (after GATE D)**: 3 more objects blacklisted (`3e8bba0176` printer #43, `5942004064` pot #56, `38d58a7a31` trash can #71); plant pot and pot are treated as one category for uniqueness.", "",
              "| | Groups | Questions | " + " | ".join(TEMPLATES) + " | " + " | ".join(LABELS) + " |", "|---" * (3 + len(TEMPLATES) + len(LABELS)) + "|"]
        for name, x in hist + [("This round", st)]:
            L.append(f"| {name} | {x['num_groups']} | {x['num_questions']} | " + " | ".join(str(x["per_template"].get(t, 0)) for t in TEMPLATES)
                     + " | " + " | ".join(str(x["evidence_label"].get(l, 0)) for l in LABELS) + " |")
        L.append(f"| This round vs previous | {st['num_groups'] - r1['num_groups']:+d} | {st['num_questions'] - r1['num_questions']:+d} | "
                 + " | ".join(f"{st['per_template'].get(t, 0) - r1['per_template'].get(t, 0):+d}" for t in TEMPLATES) + " | "
                 + " | ".join(f"{st['evidence_label'].get(l, 0) - r1['evidence_label'].get(l, 0):+d}" for l in LABELS) + " |")
        L.append("")
    # magenta-fraction condition
    rows, tot = [], 0
    for sid in step6_scenes(cfg):
        s4 = read_json(sp_work(cfg) / sid / "step4_summary.json")
        n = s4.get("magenta_frac", {}).get("visible_downgraded", 0)
        tot += n
        if n:
            rows.append(f"`{sid}` {n}")
    L += [f"- **Magenta-occlusion supplementary condition** (share of in-frustum points on magenta pixels > {sp_cfg(cfg).get('vis_max_magenta_frac')}): {tot} (object, node) pairs downgraded from visible in the full run"
          + (f" ({', '.join(rows)})" if rows else "") + ".", ""]
    wm = fd / "whatif_motion.json"
    if wm.exists():
        w = read_json(wm)
        mt = [read_json(sp_work(cfg) / s_ / "step4_summary.json").get("motion", {}) for s_ in step6_scenes(cfg)]
        ms = pd.read_csv(fd / "motion_samples.csv", dtype={"pair": str}) if (fd / "motion_samples.csv").exists() else None
        L += ["### Motion-blur condition", "",
              "Camera angular and linear speed at each keyframe, computed from the 60 FPS poses over +-2 frames (~ +-1/30 s). Values on all earlier spot-check samples (2 pilot rounds + 2 full-run rounds, 200 pairs):", ""]
        if ms is not None:
            ok = ms[ms.ok == "yes"]
            # round labels must match the values in motion_samples.csv (full-run round 2, pilot round 2, pilot round 1)
            blur = [("全量第2轮", "16"), ("试点第2轮", "19"), ("试点第1轮", "14"), ("试点第1轮", "39")]
            L += ["| Sample | Object | Angular speed (°/s) | Linear speed (m/s) |", "|---|---|---|---|",
                  *[f"| {r['round']} #{r['pair']} (known blurry) | {r['name']} | {r['ang']:.1f} | {r['lin']:.2f} |" for _, r in ms.iterrows() if (r["round"], r["pair"]) in blur],
                  *[f"| {r['round']} #{r['pair']} (box offset, manual judgment) | {r['name']} | {r['ang']:.1f} | {r['lin']:.2f} |" for _, r in ms.iterrows() if (r["round"], r["pair"]) == ("全量第2轮", "26")],
                  "", f"{len(ok)} samples judged recognizable: angular speed P50 {ok.ang.median():.1f}, P90 {ok.ang.quantile(.9):.1f}, P95 {ok.ang.quantile(.95):.1f}, max {ok.ang.max():.1f}; "
                  f"linear speed P50 {ok.lin.median():.2f}, P95 {ok.lin.quantile(.95):.2f}, max {ok.lin.max():.2f}.",
                  "", "| Angular-speed threshold (°/s) | Known blurry samples flagged (of 4) | Recognizable samples wrongly flagged |", "|---|---|---|",
                  *[f"| {t} | {sum(1 for _, r in ms.iterrows() if (r['round'], r['pair']) in blur and r['ang'] >= t)} | {int((ok.ang >= t).sum())} / {len(ok)} |" for t in (50, 60, 65, 73, 80, 90)], ""]
        L += [f"Chosen: **73°/s**: flags 2 / 4 known blurry samples (#16 97°/s, pilot #19 74°/s) and wrongly flags 4 / 181 recognizable ones; flagging the other two (53, 62°/s) would need 50°/s and wrongly flag 20, which is not conservative enough. "
              "Linear speed shows no clear relation to blur (all known blurry samples are < 0.4 m/s) and is not used.", "",
              "| Variant | Fast nodes | Visible pairs downgraded | Questions | Change |", "|---|---|---|---|---|",
              *[f"| {k} | {v['fast_nodes']} | {v['visible_demoted']} | {v['num_questions']} | {v['question_change_frac']:+.1%} |" for k, v in w["variants"].items()], "",
              f"In the sandbox, none of the three variants changes the question count ({w['baseline']['num_questions']} -> {w['baseline']['num_questions']}, <= 5%), so by the rule angular speed >= 73°/s is applied in the official run. "
              f"The official run has {sum(m.get('fast_nodes', 0) for m in mt)} fast nodes and {sum(m.get('visible_downgraded', 0) for m in mt)} downgraded visible pairs, and the question count is still {st['num_questions']}: "
              "groups and questions are determined by object-node combinations, and downgraded nodes are replaced as evidence by other nodes (in the official run, oracle evidence views contain 0 fast nodes; link_views contain 20, which is allowed because the partial_partial linking mode only requires partial), so the count is unchanged.", ""]
    s23 = fd / "s23_offset.md"
    if s23.exists():
        L += [s23.read_text().rstrip("\n"), ""]
    wr = fd / "whatif_rho.json"
    if wr.exists():
        w = read_json(wr)
        L += [f"### Sandbox evaluation of ρ 0.30 -> {w['new_rho']} (before the official run)", "",
              "Raising ρ can only downgrade visible to partial; in the sandbox, visible pairs with ρ < 0.40 in the official classification are set to partial (equivalent to rerunning Step 4), then Steps 5-6 are rerun.", "",
              "| | Groups | Questions | " + " | ".join(TEMPLATES) + " |", "|---" * (3 + len(TEMPLATES)) + "|"]
        for name, x in (("ρ 0.30 (baseline)", w["baseline"]), (f"ρ {w['new_rho']}", w["whatif"])):
            L.append(f"| {name} | {x['num_groups']} | {x['num_questions']} | " + " | ".join(str(x["per_template"].get(t, 0)) for t in TEMPLATES) + " |")
        L.append(f"| Change | {w['whatif']['num_groups'] - w['baseline']['num_groups']:+d} | {w['whatif']['num_questions'] - w['baseline']['num_questions']:+d}"
                 f" ({w['question_change_frac']:+.1%}) | " + " | ".join(f"{w['whatif']['per_template'].get(t, 0) - w['baseline']['per_template'].get(t, 0):+d}" for t in TEMPLATES) + " |")
        L += ["", f"{sum(w['visible_pairs_demoted'].values())} visible pairs downgraded in total. Rule: apply officially if the question count drops by <= 10%, do not apply if > 10%. "
              f"Result {w['question_change_frac']:+.1%}: **{'applied' if w['question_change_frac'] >= -0.10 else 'not applied'}**.", "",
              *([f"**Correction**: {w['note']} Question count of the official ρ 0.40 run: {w['official_after']['num_groups']} groups, {w['official_after']['num_questions']} questions ("
                 + ", ".join(f"{t} {w['official_after']['per_template'].get(t, 0)}" for t in TEMPLATES) + ").", ""] if "official_after" in w else []),
              "Pairs among the 50 of full-run spot-check round 1 that would flip because of this (ρ < 0.40):", "",
              "| # | Scene | Object | ρ | Round-1 judgment |", "|---|---|---|---|---|",
              *[f"| {r['pair']} | `{r['scene_id']}` | {r['name']} | {r['rho']} | {'yes' if r['ok_round1'] == 'yes' else '**no**'} |" for r in w["round1_flips"]], ""]
    wt = fd / "whatif_t5_right.json"
    if wt.exists():
        w = read_json(wt)
        L += ["### Sandbox estimate of a T5 \"right\" variant (reported only, not adopted)", "",
              "\"Among the objects on your right, which one is closest to you?\", with rules symmetric to left. R1 = ask about the right side when the left-side question is invalid (still at most 1 T5 per standpoint / facing); "
              "R2 = left and right are separate questions (the T5 within-scene dedup key includes the direction; at most 1 of each per standpoint / facing). ScanNet++ uses this round's official rules; ARKit reruns Steps 5-6 with the shared rules on all Step 4 scenes"
              " (before Step 8 debiasing, so the baseline differs from the frozen v0.11 question count).", "",
              "| Dataset | Variant | Groups | Questions | T5 | T5 increase | Of which right |", "|---|---|---|---|---|---|---|"]
        for ds, rr in w.items():
            b5 = rr["base"]["per_template"].get("T5", 0)
            for m in ("base", "R1", "R2"):
                x = rr[m]
                t5 = x["per_template"].get("T5", 0)
                L.append(f"| {ds} | {m} | {x['num_groups']} | {x['num_questions']} | {t5} | {t5 - b5:+d} | {x['t5_variants'].get('T5:right', 0)} |")
        L += ["", "Adopting it would also require writing and reviewing paraphrases for right (Step 7).", ""]
    return L


# ---------------------------------------------------------------- suspected mirrors
def mirror_image(cfg, sid, obj_id, out_path) -> str | None:
    """Draw on the node with the most pixels where sensor depth inside the box face is farther than the face: far pixels in red, the face as a blue outline. Returns the node id."""
    sp = sp_cfg(cfg)
    wd = sp_work(cfg) / sid
    objs = read_json(wd / "objects.json")
    frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
    nodes = read_json(wd / "view_graph.json")["nodes"]
    kf = read_json(wd / "keyframes.json")["keyframe_ids"]
    fc = mirror_face(objs[obj_id], np.mean([frames[i]["position"] for i in kf], axis=0))
    best = (0, None, None)
    for fid in nodes:
        f = frames[fid]
        Kd, T, (Wd, Hd) = np.asarray(f["depth_K"]), np.asarray(f["T_wc"]), f["depth_size"]
        C, R = T[:3, 3], T[:3, :3]
        if (C - fc["center"]) @ fc["normal"] <= 0:
            continue
        uv, z = g.project(face_corners(fc), Kd, T)
        if (z <= 0.05).any():
            continue
        m = np.zeros((Hd, Wd), np.uint8)
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
        vv, uu = np.nonzero(m)
        if not len(vv):
            continue
        dirs = np.stack([(uu - Kd[0, 2]) / Kd[0, 0], (vv - Kd[1, 2]) / Kd[1, 1], np.ones(len(uu))], 1) @ R.T
        zf = ((fc["center"] - C) @ fc["normal"]) / (dirs @ fc["normal"])
        d = arkit.read_depth_m(sp_root(cfg) / f["depth_rel_path"], sp["depth_scale"])[vv, uu]
        far = (d > 0) & (d > zf + sp["mirror_scan_far_m"])
        if far.sum() > best[0]:
            mask = np.zeros((Hd, Wd), bool)
            mask[vv[far], uu[far]] = True
            best = (int(far.sum()), fid, mask)
    if best[1] is None:
        return None
    f = frames[best[1]]
    img, K, T = base_image(cfg, f, 960)
    m = cv2.resize(best[2].astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[m] = (0.5 * img[m] + [0, 0, 127]).astype(np.uint8)
    uv, z = g.project(face_corners(fc), K, T)
    cv2.polylines(img, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (255, 120, 0), 2, cv2.LINE_AA)
    put(img, f"{sid} {best[1]}: {objs[obj_id]['source_label']} #{obj_id}  red = sensor depth > box face + {sp['mirror_scan_far_m']} m", (8, 24), 0.55)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return best[1]


def cards_overview(card_dir, out_path, width=480, cols=6) -> None:
    """Shrink all cards and tile them into one overview image."""
    ims = []
    for f in sorted(card_dir.glob("*.jpg")):
        a = cv2.imread(str(f))
        ims.append(cv2.resize(a, (width, int(a.shape[0] * width / a.shape[1])), interpolation=cv2.INTER_AREA))
    if not ims:
        return
    H = max(i.shape[0] for i in ims)
    ims = [cv2.copyMakeBorder(i, 0, H - i.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in ims]
    while len(ims) % cols:
        ims.append(np.full_like(ims[0], 255))
    cv2.imwrite(str(out_path), np.concatenate([np.concatenate(ims[k:k + cols], 1) for k in range(0, len(ims), cols)], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- report
def write_report(cfg) -> None:
    sp, qc = sp_cfg(cfg), cfg["questions"]
    rep = path_of(cfg, "reports")
    out_img = rep / "scannetpp_full"
    out_img.mkdir(parents=True, exist_ok=True)
    allsc = read_split(cfg)
    ga = {r["scene_id"]: r for r in read_json(full_dir(cfg) / "gate_a.json")["scenes"]}
    s6 = step6_scenes(cfg)
    groups = [json.loads(l) for l in open(sp_work(cfg) / "_step6" / "test_groups.jsonl")]
    st = read_json(sp_work(cfg) / "_step6" / "step6_stats.json")
    qs = [q for grp in groups for q in grp["questions"]]
    types = json.load(open(sp_root(cfg) / "metadata" / "scene_types.json"))
    by_scene = {sid: [x for x in groups if x["scene_id"] == scene_id_of(DATASET, sid)] for sid in allsc}
    L = ["# ScanNet++ full run, Steps 1-6 (50 nvs_sem_val scenes)", "",
         "> Generated by `python -m actspace_build.scannetpp_full_report report`. Rules are identical to the pilot (plus the adjustments after GATE C2: "
         "T2 deduplicated by reference object + direction + correct answer, singular name for window blind, no drop-based quadrant balancing), and the later rule changes described below. "
         "Visibility spot-check round 3 passed 48 / 50 (manually reviewed) and Steps 1-6 were accepted; the question counts in this file are the final results after round 4.", "",
         f"**Result: {sum(r['passed'] for r in ga.values())} of {len(allsc)} scenes pass GATE A, {sum(1 for s in s6 if by_scene[s])} scenes yield questions; "
         f"{st['num_groups']} groups, {st['num_questions']} questions**; per template {fmt(dict(sorted(st['per_template'].items())))}; evidence labels {fmt(st['evidence_label'])}.", ""]

    # ---- 1. per scene
    L += ["## 1. Per scene", "",
          "| Scene | Type | GATE A | Decoded frames | Frames dropped (bad / unverified pose) | Keyframes | View-graph nodes | Referable objects | Usable objects (visible at >= 1 node) | Groups | Questions | Scene-level exclusion |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = Counter()
    scene_drop = Counter()
    for sid in allsc:
        r = ga[sid]
        wd = sp_work(cfg) / sid
        kf = read_json(wd / "keyframes.json") if (wd / "keyframes.json").exists() and r["passed"] else None
        s3 = read_json(wd / "step3_summary.json") if (wd / "step3_summary.json").exists() and r["passed"] else None
        s4 = read_json(wd / "step4_summary.json") if (wd / "step4_summary.json").exists() and r["passed"] else None
        drop = [] if r["passed"] else [f"GATE A: {', '.join(r['reasons'])}"]
        if kf and not kf["scene_ok"]:
            drop.append("TOO_FEW_KEYFRAMES")
        if s3 and not s3["scene_ok"]:
            drop.append("TOO_FEW_OBJECTS")
        if s4 and not s4["ok"]:
            drop.append("GRAPH_DISCONNECTED")
        for d in drop:
            scene_drop[d.split(": ")[0]] += 1
        n_ref = s3["num_referable"] if s3 else None
        n_use = sum(1 for o in s4["objects"] if o["referable"] and o["num_visible_nodes"] > 0) if s4 else None
        gs = by_scene[sid]
        nq = sum(len(x["questions"]) for x in gs)
        if r["passed"]:
            tot["kf"] += kf["num_keyframes"] if kf else 0
            tot["ref"] += n_ref or 0
            tot["use"] += n_use or 0
        tot["g"] += len(gs)
        tot["q"] += nq
        L.append(f"| `{sid}` | {types.get(sid, '?')} | {'pass' if r['passed'] else '**fail**'} | {r.get('n_decoded_frames', '—')} | {r.get('n_frames_dropped', '—')} | "
                 f"{kf['num_keyframes'] if kf else '—'} | {s4['num_graph_nodes'] if s4 else '—'} | {n_ref if n_ref is not None else '—'} | {n_use if n_use is not None else '—'} | "
                 f"{len(gs)} | {nq} | {'; '.join(drop) or '—'} |")
    L += [f"| **Total** | | {sum(r['passed'] for r in ga.values())} / {len(allsc)} | | | {tot['kf']} | | {tot['ref']} | {tot['use']} | {tot['g']} | {tot['q']} | |", ""]
    hist = Counter(min(sum(1 for o in read_json(sp_work(cfg) / s / "step4_summary.json")["objects"] if o["referable"] and o["num_visible_nodes"] > 0), 9) for s in s6)
    L += [f"- Scene-level exclusions: {fmt(scene_drop) if scene_drop else 'none'}.",
          f"- Distribution of usable objects per scene (9 = >= 9): {fmt(dict(sorted(hist.items())))}; scenes with <= 2 usable objects cannot form clusters.", ""]

    # ---- 2. whole-dataset statistics
    L += ["## 2. Whole dataset", "", "| Template | " + " | ".join(TEMPLATES) + " | Total |", "|---" * (len(TEMPLATES) + 2) + "|",
          "| Questions | " + " | ".join(str(st["per_template"].get(t, 0)) for t in TEMPLATES) + f" | {st['num_questions']} |", "",
          "| Evidence label | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|",
          "| Questions | " + " | ".join(str(st["evidence_label"].get(l, 0)) for l in LABELS) + " |", "",
          f"- Breakdown: {fmt(st['per_variant'])}. Constraints: {fmt(st['constraints'])}.",
          f"- Difficulty (explore questions by oracle_steps tertiles, cut points {[round(c, 1) for c in st['difficulty_cuts_oracle_steps']]}): {fmt(st['difficulty'])}.",
          f"- Number of source scenes per template: " + ", ".join(f"{t} {len({x['scene_id'] for x in groups if any(q['template'] == t for q in x['questions'])})}" for t in TEMPLATES) + ".", ""]

    # ---- 3. drop reasons
    keys, per_step = Counter(), Counter()
    for sid in allsc:
        p = sp_work(cfg) / sid / "dropped.jsonl"
        if p.exists():
            for l in open(p):
                d = json.loads(l)
                keys[(d["step"], d.get("kind", ""), d["reason_code"])] += 1
    for sid in s6:
        n_nv = sum(1 for o in read_json(sp_work(cfg) / sid / "step4_summary.json")["objects"] if o["referable"] and o["num_visible_nodes"] == 0)
        keys[("step4", "object", "NEVER_VISIBLE (referable but no visible node; derived)")] += n_nv
    order = ["step1_gate_a", "step2_keyframes", "step3_objects", "step3_scene", "step4", "step56"]
    L += ["## 3. Drop reasons (by reason code, all 50 scenes)", "", "| Step | Item | Reason code | Count |", "|---|---|---|---|",
          *[f"| {k[0]} | {k[1]} | `{k[2]}` | {v} |" for k, v in sorted(keys.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 9, -kv[1]))], "",
          f"Candidate-question filter hits (enumeration level): {fmt(st['candidate_filter_counts'])}.", ""]

    # ---- 4. unanswerable
    reason_cn = {"masked_frac_too_large": "masked share over limit", "group_incompatible": "other questions in group unanswerable / evidence not linkable", "p0_masked": "object visible at p0",
                 "never_visible_evidence": "masked object never visible", "margin_fail": "T4 angle / T1 distance margin not met", "duplicate_in_scene": "duplicate within scene",
                 "template_diversity": "fewer than 2 templates after replacement", "evidence_too_large": "too many evidence objects", "dataset_cap_reached": "dataset-wide share cap reached"}
    dom, tried, succ, raw = Counter(), 0, 0, Counter()
    for sid, spc in st["special_attempts"].items():
        sa = spc["unanswerable"]
        added = {a["group_index"] for a in (sa.get("added") or [])}
        has = set(sa.get("groups_with_candidate", []))
        raw.update(sa.get("fail_counts", {}))
        for gi, fc in sa.get("fail_counts_by_group", {}).items():
            tried += 1
            if int(gi) in added:
                succ += 1
                continue
            if int(gi) in has:
                dom["had candidates but none selected (each object used once / share cap)"] += 1
                continue
            fc2 = {k: v for k, v in fc.items() if k != "never_visible_evidence"} or fc
            dom[reason_cn.get(max(fc2, key=fc2.get), max(fc2, key=fc2.get)) if fc2 else "no candidate"] += 1
    n_ua = st["evidence_label"].get("unanswerable", 0)
    L += ["## 4. unanswerable (masking method, attempted for every group)", "",
          f"- Attempted {tried} groups, succeeded **{succ}** ({succ / max(tried, 1):.0%}); {n_ua} unanswerable questions, {n_ua / max(len(qs), 1):.1%} of all (target {qc['target_unanswerable_frac'][0]:.0%}-{qc['target_unanswerable_frac'][1]:.0%}).",
          f"- Main reasons for unsuccessful groups (number of groups): {fmt(dict(dom.most_common()))}.",
          f"- Attempt-level failure counts (group x masked object): {fmt({reason_cn.get(k, k): v for k, v in raw.most_common()})}.", ""]

    # ---- 5. balance
    L += ["## 5. Option-letter and quadrant deviation (no drop-based balancing)", "", "| Template | Options | Questions | Correct letter | Max deviation |", "|---|---|---|---|---|"]
    for t in TEMPLATES:
        for n in (2, 3, 4):
            items = [q for q in qs if q["template"] == t and q["answer_type"] == "multiple_choice" and len(q["options"]) == n]
            if items:
                c = Counter(q["answer"] for q in items)
                L.append(f"| {t} | {n} | {len(items)} | {fmt(dict(sorted(c.items())))} | {max(abs(c.get(l, 0) / len(items) - 1 / n) for l in 'ABCD'[:n]):.0%} |")
    L += ["", "| Template | Questions | front-left | front-right | back-left | back-right | Max deviation from 25% |", "|---|---|---|---|---|---|---|"]
    for t in ("T3", "T4"):
        c = Counter(q["answer_value"] for q in qs if q["template"] == t)
        n = sum(c.values())
        if n:
            L.append(f"| {t} | {n} | " + " | ".join(str(c.get(k, 0)) for k in ("front-left", "front-right", "back-left", "back-right"))
                     + f" | {max(abs(c.get(k, 0) / n - 0.25) for k in g.QUADRANTS):.0%} |")
    L += ["", "Quadrant balancing is deferred to the Step 8 rerun after merging with ARKit.", ""]

    # ---- 6. comparison with targets, and option B
    tot_q = ARKIT_V011["questions"] + st["num_questions"]
    L += ["## 6. Actual totals vs targets", "",
          "| | Total questions | " + " | ".join(TEMPLATES) + " |", "|---" * (len(TEMPLATES) + 2) + "|",
          "| ScanNet++ (this full run, Step 6) | " + f"{st['num_questions']} | " + " | ".join(str(st["per_template"].get(t, 0)) for t in TEMPLATES) + " |",
          "| ARKit v0.11 (frozen) | " + f"{ARKIT_V011['questions']} | " + " | ".join(str(ARKIT_V011[t]) for t in TEMPLATES) + " |",
          "| **Total** | " + f"**{tot_q}** | " + " | ".join(f"**{ARKIT_V011[t] + st['per_template'].get(t, 0)}**" for t in TEMPLATES) + " |",
          "| Short of 200 per template | | " + " | ".join(str(max(0, 200 - ARKIT_V011[t] - st["per_template"].get(t, 0))) or "0" for t in TEMPLATES) + " |", "",
          f"- Total {tot_q}, target 1000-2000: {'lower bound reached' if tot_q >= 1000 else f'**{1000 - tot_q} questions short of the lower bound**'}; ScanNet++ share {st['num_questions'] / tot_q:.0%}. "
          "The ScanNet++ numbers are Step 6 results; Step 8 debias pruning will remove some more (ARKit v0.11 pruned 3%).", ""]
    wf = full_dir(cfg) / "whatif_area.json"
    if wf.exists():
        w = read_json(wf)
        base_use = {s: sum(1 for o in read_json(sp_work(cfg) / s / "step4_summary.json")["objects"] if o["referable"] and o["num_visible_nodes"] > 0) for s in s6}
        L += ["### Option B recomputed on real data: visibility area threshold 0.02 -> 0.01 (ScanNet++ only)", "",
              f"Each scene's Step 4 outputs are copied into a sandbox work directory, visible is re-classified from the raw values in visibility.npz (mirror / magenta occlusion are already in the raw values; the two extra shape / sharpness conditions apply as usual), and Steps 5-6 are rerun (same rules as the official run); official outputs are untouched. "
              f"(Self-check: re-classifying at the current threshold disagrees with the official result on {sum(w['reclassify_mismatch_at_current_threshold'].values())} (object, node) pairs.)", "",
              "| | Usable objects | Groups | Questions | " + " | ".join(TEMPLATES) + " |", "|---" * (len(TEMPLATES) + 4) + "|",
              f"| Current (0.02) | {sum(base_use.values())} | {st['num_groups']} | {st['num_questions']} | " + " | ".join(str(st["per_template"].get(t, 0)) for t in TEMPLATES) + " |",
              f"| Option B (0.01) | {sum(w['usable_per_scene'].values())} | {w['num_groups']} | {w['num_questions']} | " + " | ".join(str(w["per_template"].get(t, 0)) for t in TEMPLATES) + " |",
              f"| Increase | {sum(w['usable_per_scene'].values()) - sum(base_use.values()):+d} | {w['num_groups'] - st['num_groups']:+d} | {w['num_questions'] - st['num_questions']:+d}"
              f" ({w['num_questions'] / max(st['num_questions'], 1) - 1:+.0%}) | " + " | ".join(f"{w['per_template'].get(t, 0) - st['per_template'].get(t, 0):+d}" for t in TEMPLATES) + " |", "",
              "Option B mainly recovers small objects (visible pairs with area 0.01-0.02); adopting it would require a new visibility spot check at the new threshold.", ""]

    L += v02_section(cfg)
    L += round_changes_section(cfg, st, ga)
    L += spotcheck_section(cfg)

    # ---- 8. suspected mirrors
    md = out_img / "mirrors"
    md.mkdir(parents=True, exist_ok=True)
    for f in md.glob("*.jpg"):
        f.unlink()
    rows = []
    for sid in s6:
        s4 = read_json(sp_work(cfg) / sid / "step4_summary.json")
        objs = read_json(sp_work(cfg) / sid / "objects.json")
        for r in s4["suspected_mirrors"]:
            if r["suspected"]:
                fn = f"{sid}_{r['obj_id']}.jpg"
                node = mirror_image(cfg, sid, r["obj_id"], md / fn)
                o = objs[r["obj_id"]]
                rows.append((sid, r, o, node, fn))
    L += ["## 8. Suspected mirrors (reported only, classification unchanged; for manual confirmation)", "",
          f"Same criterion as the pilot (within the box face of a flat object, share of pixels whose sensor depth is more than {sp['mirror_scan_far_m']} m beyond the face >= {sp['mirror_scan_min_far_frac']:.0%}, counted nodes >= {sp['mirror_scan_min_nodes']}). "
          "Both mirrors and transparent glass (windows, glass doors / partitions) are listed. Objects whose label contains mirror and objects in `mirror_object_overrides` are already treated as mirrors, shown in the \"Treated as mirror\" column.", "",
          "| Scene | Object | Referable | Box size (m) | Counted nodes | \"Farther\" pixel share | Treated as mirror | Image |", "|---|---|---|---|---|---|---|---|"]
    for sid, r, o, node, fn in rows:
        mirror_ids = set(np.load(sp_work(cfg) / sid / "visibility.npz")["mirror_obj_ids"].tolist())
        L.append(f"| `{sid}` | {o['source_label']} #{o['obj_id']} | {'yes' if o['referable'] else 'no'} | {' × '.join(map(str, r['size']))} | {r['nodes']} | {r['far_frac']:.0%} | "
                 f"{'yes' if o['obj_id'] in mirror_ids else 'no'} | ![](scannetpp_full/mirrors/{fn}) |" if node else
                 f"| `{sid}` | {o['source_label']} #{o['obj_id']} | {'yes' if o['referable'] else 'no'} | {' × '.join(map(str, r['size']))} | {r['nodes']} | {r['far_frac']:.0%} | — | — |")
    labelled = [(sid, o) for sid in s6 for o in read_json(sp_work(cfg) / sid / "objects.json") if "mirror" in o["source_label"].lower()]
    lab_txt = ", ".join(f"`{s}` {o['source_label']} #{o['obj_id']}" for s, o in labelled) or "none"
    L += ["", f"{len(rows)} suspected in total. Objects whose label contains mirror in the full run: {lab_txt}. "
          "In the images, red = pixels inside the box face whose sensor depth is beyond the face; blue outline = box face.", ""]

    # ---- 9. cards and outputs
    cd = out_img / "cards"
    cd.mkdir(parents=True, exist_ok=True)
    for f in cd.glob("*.jpg"):
        f.unlink()
    L += ["## 9. One card per scene", "", "For each scene with questions, 1 group is shown (groups with masked questions first, then the group with the most questions). Card format as in GATE C2. "
          f"Thumbnail overview of all {sum(1 for s_ in s6 if by_scene[s_])} cards: `reports/scannetpp_full/cards_overview.jpg`.", ""]
    for sid in s6:
        gs = by_scene[sid]
        if not gs:
            continue
        grp = sorted(gs, key=lambda x: (not x.get("view_graph_mask"), -len(x["questions"]), x["group_id"]))[0]
        fn = f"{sid}_{grp['group_id']}.jpg"
        card(cfg, scene_data(cfg, sid), grp, cd / fn)
        L += [f"### `{sid}` ({types.get(sid, '?')}, {len(gs)} groups) · {grp['group_id']}", "", f"![{sid}](scannetpp_full/cards/{fn})", ""]
    cards_overview(cd, out_img / "cards_overview.jpg")
    prev = rep / "scannetpp_full_preview.jsonl"
    with open(prev, "w") as f:
        for grp in groups:
            f.write(json.dumps(grp, ensure_ascii=False) + "\n")
    L += ["## 10. Outputs", "",
          "- Questions: `work/scannetpp/_step6/test_groups.jsonl`, `step6_stats.json`; GATE A summary `work/scannetpp/_full/gate_a.json`.",
          f"- Question preview (all {len(qs)} questions, with gold answers): `reports/scannetpp_full_preview.jsonl` (not in git).",
          "- Cards: `reports/scannetpp_full/cards/`; suspected-mirror images: `reports/scannetpp_full/mirrors/`; visibility spot-check images: `reports/scannetpp_full_spotcheck/` (none in git).",
          f"- Option B sandbox: `{sp['whatif_work_root']}/{sp['work_subdir']}/` (separate from the official work directory).", ""]
    review = full_dir(cfg) / "agent_review.md"
    if review.exists():
        L += [review.read_text().rstrip("\n"), ""]
    (rep / "scannetpp_full_summary.md").write_text("\n".join(L) + "\n")
    print("wrote", rep / "scannetpp_full_summary.md", st["num_groups"], st["num_questions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["whatif_area", "spotcheck_images", "spotcheck", "report"])
    args = ap.parse_args()
    cfg = load_config()
    {"whatif_area": whatif_area, "spotcheck_images": spotcheck_images, "spotcheck": write_report, "report": write_report}[args.cmd](cfg)


if __name__ == "__main__":
    main()
