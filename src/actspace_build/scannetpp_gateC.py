"""ScanNet++ GATE C2 pilot material: reports/scannetpp_gateC_pilot.md, cards (reports/scannetpp_gateC/cards/), question preview jsonl.

Requires scannetpp_step56 to have run first (primary result in work/scannetpp/_step6/, comparisons in _step6_<variant>/).
The scale extrapolation uses work/scannetpp/_gateC/label_candidates.json (referable-candidate counts per scene, estimated from the
annotations alone for all 50 scenes; see _gateC/label_candidates.py).
Usage: python -m actspace_build.scannetpp_gateC
"""
from __future__ import annotations

import json
import textwrap
from collections import Counter

import cv2
import numpy as np

from . import geometry as g
from .common import load_config, path_of, read_json
from .scannetpp import scale_intrinsics, sp_cfg, sp_root, sp_work
from .scannetpp_step4_report import put
from .step1_arkit import draw_obbs
from .step4_graph_visibility import STATE_INVISIBLE, STATE_PARTIAL, STATE_VISIBLE
from .scannetpp_vocab import load_mapping
from .step56_questions import scene_id_of

DATASET = "scannetpp"
TEMPLATES = ("T1", "T2", "T3", "T4", "T5")
LABELS = ("init", "explore", "unanswerable")
STATE_COLOR = {STATE_VISIBLE: (0, 200, 0), STATE_PARTIAL: (0, 150, 255)}
CARDS_PER_SCENE = 3
TILE_W = 640
MAX_TILES = 9


def load_groups(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def scene_data(cfg, sid):
    wd = sp_work(cfg) / sid
    v = np.load(wd / "visibility.npz")
    return {"objs": read_json(wd / "objects.json"), "state": v["state"], "nodes": [str(x) for x in v["frame_ids"]],
            "frames": {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}, "raw": read_json(wd / "groups_raw.json"),
            "drops": [json.loads(l) for l in open(wd / "dropped.jsonl")], "s4": read_json(wd / "step4_summary.json"), "vis": v}


def fmt(d):
    return ", ".join(f"{k} {v}" for k, v in d.items()) if d else "—"


# ---------------------------------------------------------------- cards
def tile(cfg, D, fid, ev, name, title, title_color=(0, 0, 0)):
    f = D["frames"][fid]
    img = cv2.imread(str(sp_root(cfg) / f["rel_path"]))
    W0, H0 = f["image_size"]
    img = cv2.resize(img, (TILE_W, int(round(H0 * TILE_W / W0))), interpolation=cv2.INTER_AREA)
    K = scale_intrinsics(np.asarray(f["K"]), (W0, H0), (img.shape[1], img.shape[0]))
    T = np.asarray(f["T_wc"])
    v = D["nodes"].index(fid)
    row = {o["obj_id"]: k for k, o in enumerate(D["objs"])}
    for oid in ev:
        st = int(D["state"][row[oid], v])
        if st == STATE_INVISIBLE:
            continue
        o = D["objs"][row[oid]]["obb"]
        box = {"center": np.asarray(o["center"]), "axes": np.asarray(o["axes"]), "size": np.asarray(o["size"]), "label": ""}
        draw_obbs(img, [box], K, T, np.eye(3), color=STATE_COLOR[st])
        uv, z = g.project(box["center"][None], K, T)
        if z[0] > 0.05:
            p = np.clip(np.round(uv[0]).astype(int), [4, 40], [img.shape[1] - 120, img.shape[0] - 8])
            put(img, f"{name[oid]} ({'visible' if st == STATE_VISIBLE else 'partial'})", (int(p[0]), int(p[1])), 0.5, STATE_COLOR[st])
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    while tw > TILE_W - 12 and len(title) > 10:
        title = title[:-4] + "…"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (TILE_W, 24), title_color, -1)
    cv2.putText(img, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def card(cfg, D, grp, out_path):
    """Initial view + evidence views (oracle_nodes, link_views, T4 path_views; masked groups add 1 masked node) + questions, options, answers."""
    name = {o["obj_id"]: o["ref_name"] or o["category"] for o in D["objs"]}
    ev = sorted({o for q in grp["questions"] for o in q["evidence_objs"]} | set(grp["cluster_obj_ids"]))
    roles = {}
    for qi, q in enumerate(grp["questions"], 1):
        for n in q.get("oracle_nodes") or []:
            roles.setdefault(n, []).append(f"Q{qi} oracle")
        for n in q.get("link_views") or []:
            roles.setdefault(n, []).append(f"Q{qi} link")
        for k, n in enumerate(q.get("path_views") or [], 1):
            roles.setdefault(n, []).append(f"Q{qi} path {k}")
    roles.pop(grp["init_frame"], None)
    tiles = [tile(cfg, D, grp["init_frame"], ev, name, f"p0 initial view  {grp['init_frame']}", (90, 60, 0))]
    order = list(roles)
    for n in order[:MAX_TILES - 1]:
        tiles.append(tile(cfg, D, n, ev, name, f"{', '.join(roles[n])}  {n}"))
    mask = grp.get("view_graph_mask")
    if mask:
        x = mask["masked_obj"]
        row = {o["obj_id"]: k for k, o in enumerate(D["objs"])}
        seen = [n for n in mask["masked_nodes"] if D["state"][row[x], D["nodes"].index(n)] != STATE_INVISIBLE]
        if seen:
            tiles.append(tile(cfg, D, seen[0], ev, name, f"MASKED node {seen[0]}: {name[x]} seen here", (30, 30, 170)))
    cols = 3
    blank = np.full_like(tiles[0], 255)
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    W = grid.shape[1]
    lines = [f"{grp['group_id']}   {grp['scene_id']}   cluster: {', '.join(name[o] for o in grp['cluster_obj_ids'])}   "
             f"constraint: {grp['constraints'][0]['type'] if grp['constraints'] else 'none'}",
             "box colours: green = visible at that node, orange = partial (invisible objects are not drawn)"]
    if len(order) > MAX_TILES - 1:
        lines.append(f"(further evidence views not shown: {', '.join(order[MAX_TILES - 1:])})")
    if mask:
        lines.append(f"MASKED view graph: object '{name[mask['masked_obj']]}' hidden; {len(mask['masked_nodes'])} masked nodes: "
                     + ", ".join(mask["masked_nodes"][:24]) + (" ..." if len(mask["masked_nodes"]) > 24 else ""))
    for qi, q in enumerate(grp["questions"], 1):
        ans = f"{q['answer']} m" if q["answer_type"] == "numeric" else f"{q['answer']} ({q.get('answer_value') or name[q['answer_obj']]})"
        if q["evidence_label"] == "unanswerable":
            ans = f"cannot be determined (object masked; true value {ans})"
        lines += textwrap.wrap(f"Q{qi} [{q['template']}{':' + q['variant'] if q.get('variant') else ''}] {q['text']}", 150)
        if q.get("options"):
            lines.append("      " + "   ".join(q["options"]))
        extra = []
        if q.get("link_views"):
            extra.append("link_views " + ",".join(q["link_views"]))
        if q.get("path_views"):
            extra.append("path_views " + ",".join(q["path_views"]))
        lines.append(f"      answer: {ans}   evidence: {q['evidence_label']}   oracle_views={q['oracle_views']} oracle_steps={q['oracle_steps']}"
                     + ("   " + "; ".join(extra) if extra else ""))
    lines = [w for t in lines for w in (textwrap.wrap(t, 150, subsequent_indent="      ") or [""])]
    panel = np.full((24 * len(lines) + 16, W, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.concatenate([grid, panel], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- extrapolation and comparison
ARKIT_V011 = {"groups": 104, "questions": 425, "T1": 105, "T2": 115, "T3": 78, "T4": 93, "T5": 34}


def q_per_scene(u: int, cap: int = 8) -> tuple[int, float]:
    """Empirical pilot relation: usable objects u -> (groups, questions). u <= 2 cannot form a cluster; 3 allow only one 3-object
    cluster; 4 give about 4 groups; 5 give 6-8 groups; >= 6 reach the per-scene group cap.
    (For the 5 pilot scenes this gives 4 + 1 + 7 + 7 + 1 = 20 groups, matching the actual result.)"""
    groups = min({0: 0, 1: 0, 2: 0, 3: 1, 4: 4, 5: 7}.get(u, cap), cap)
    return groups, groups * 4.3


def extrapolate(cands: dict, ratios, cap: int = 8) -> list[dict]:
    out = []
    for name, r in ratios:
        G = Q = 0
        for c in cands.values():
            gq = q_per_scene(int(round(c * r)), cap)
            G += gq[0]
            Q += gq[1]
        out.append({"name": name, "ratio": r, "groups": G, "questions": Q})
    return out


def usable_whatif(cfg, S, **over) -> dict:
    """Re-decide `visible` on the raw pilot visibility values with modified thresholds (the short-side and sharpness conditions
    still apply); return, per scene, the number of referable objects visible at >= 1 node."""
    vc = {**cfg["visibility"], **{k: v for k, v in over.items() if k.startswith("visible_")}}
    sp = sp_cfg(cfg)
    n = {}
    for sid, D in S.items():
        v = D["vis"]
        vis = ((v["rho"] >= vc["visible_min_ratio"]) & (v["area_frac"] >= vc["visible_min_area_frac"]) & (v["n_visible"] >= vc["visible_min_points"])
               & (v["completeness"] >= vc["visible_min_completeness"]) & (v["in_frame_frac"] >= vc["visible_min_in_frame_frac"])
               & (v["min_rect_short_px"] >= sp["vis_min_short_side_px"]) & (v["sharpness_ratio"] >= sp["vis_min_sharpness_ratio"]))
        n[sid] = sum(1 for k, o in enumerate(D["objs"]) if o["referable"] and vis[k].any())
    return n


# ---------------------------------------------------------------- report
def write(cfg):
    sp = sp_cfg(cfg)
    pilots = sp["pilot_scenes"]
    W6 = path_of(cfg, "work") / DATASET
    rep = path_of(cfg, "reports")
    groups = load_groups(W6 / "_step6" / "test_groups.jsonl")
    st = read_json(W6 / "_step6" / "step6_stats.json")
    variants = read_json(W6 / "_step6" / "evidence_link_modes.json")
    S = {sid: scene_data(cfg, sid) for sid in pilots}
    cands = read_json(W6 / "_gateC" / "label_candidates.json")
    prim = [k for k, v in variants.items() if v["primary"]][0]
    allowed_cats = load_mapping(cfg)[1]
    qc, gc = cfg["questions"], cfg["groups"]
    by_scene = {sid: [g_ for g_ in groups if g_["scene_id"] == scene_id_of(DATASET, sid)] for sid in pilots}
    qs = [q for g_ in groups for q in g_["questions"]]

    L = ["# ScanNet++ GATE C2: pilot question generation (5 scenes)", "",
         "> Branch `scannetpp`. Generated by `python -m actspace_build.scannetpp_gateC`. Based on Steps 5–6 in docs/data_construction.md "
         "plus later rule changes (`groups.require_constraint: false`, T4 `path_views`, etc.). This round stops at GATE C2; **the full run has not started**.", "",
         f"**Result (primary, evidence-link mode `{variants[prim]['evidence_link_mode']}`): {st['num_groups']} question groups, {st['num_questions']} questions**; "
         f"per template {fmt(dict(sorted(st['per_template'].items())))}; evidence labels {fmt(st['evidence_label'])}.", ""]

    # ---- 0. Adjustments in this round
    L += ["## 0. Adjustments and rules in this round", "",
          "- **Names default to the coarse (merged) category name**: the fine-grained name is used only when the coarse name would be a misnomer; "
          "currently only monitor (not called TV) and copier (not called printer). "
          "Affected name in the pilot: bookshelf → the shelf (`f3d64c30f8`). The name mapping table is in `scannetpp_step3_objects.md`.",
          "- **Separate blacklist file**: the ScanNet++ blacklist lives in `configs/object_blacklist_scannetpp.yaml` (`scannetpp.blacklist_file`); `configs/object_blacklist.yaml` is identical to main. The label-name check when blacklisting is kept.",
          "- **Evidence-link mode**: ARKit currently uses `partial_partial` (`questions.evidence_link_mode`; adopted since v0.10 because visible_partial yielded < 500 questions). "
          "The ScanNet++ primary result follows it (`scannetpp.evidence_link_mode: null`); see Section 5 for the comparison.",
          f"- Step 5: clusters are drawn only from referable, non-blacklisted objects that are visible at >= 1 node; {gc['objects_per_cluster'][0]}–{gc['objects_per_cluster'][1]} objects per cluster; "
          f"scenes with <= {gc['small_scene_max_referable']} referable objects sample {gc['small_scene_cluster_size']}-object clusters first; pairwise distance within a cluster >= {gc['min_pair_distance_m']} m, never visible together at one node; "
          f"objects shared between groups <= {gc['max_object_overlap_between_groups']}; at most {gc['max_groups_per_scene']} groups per scene. Initial views follow the relaxed rule (>= 1 annotated object visible or >= 2 visible / partial, "
          f"invisible fraction of cluster objects >= {gc['init_view_min_invisible_frac']:.0%}).",
          f"- Step 6: {gc['questions_per_group'][0]}–{gc['questions_per_group'][1]} questions per group, >= 2 templates; constraints optional (`groups.require_constraint: false`); "
          "evidence labelling, oracle, same-scene deduplication, the T2:pair share cap and T4 `path_views` all use the shared code (`step56_questions`, only a dataset parameter was added; ARKit behaviour unchanged).",
          "- **Balancing**: option letters are balanced within the 5 scenes per (template, number of options) by shuffling, without removing questions. The shared T3/T4 quadrant balancing removes questions (±5%); "
          "since the pilot is small and balancing should not cost many questions, the primary result does not remove questions for quadrant balance (`scannetpp.pilot_quadrant_balance_tol: null`, **pending confirmation**); "
          f"Section 4 lists the questions the shared rule would remove.",
          "- Since the last merge, main only received evaluation-related commits and no changes to the data-construction code or rules, so no merge was done in this round.", ""]

    # ---- 1. Per scene
    L += ["## 1. Per scene", "",
          "| Scene | Type | Usable objects (visible at >= 1 node) | Clusters | Groups | Questions | " + " | ".join(TEMPLATES) + " | init | explore | unanswerable |",
          "|---" * (6 + len(TEMPLATES) + 3) + "|"]
    types = json.load(open(sp_root(cfg) / "metadata" / "scene_types.json"))
    tot = Counter()
    for sid in pilots:
        D, gs = S[sid], by_scene[sid]
        usable = sum(1 for o in D["s4"]["objects"] if o["referable"] and o["num_visible_nodes"] > 0)
        sq = [q for g_ in gs for q in g_["questions"]]
        tc, lc = Counter(q["template"] for q in sq), Counter(q["evidence_label"] for q in sq)
        row = [usable, len(D["raw"]["clusters"]), len(gs), len(sq)] + [tc[t] for t in TEMPLATES] + [lc[l] for l in LABELS]
        for k, x in enumerate(row):
            tot[k] += x
        L.append(f"| `{sid}` | {types.get(sid)} | " + " | ".join(str(x) for x in row) + " |")
    L += ["| **Total** | | " + " | ".join(f"**{tot[k]}**" for k in range(len(row))) + " |", ""]
    L += [f"- Constraints: {fmt(st['constraints'])} (none = group without constraint, allowed since v0.11). T2 breakdown: {fmt({k: v for k, v in st['per_variant'].items() if k.startswith('T2')})}.",
          f"- Difficulty (explore questions split into terciles of oracle_steps, cut points {[round(c, 1) for c in st['difficulty_cuts_oracle_steps']]}, provisional pilot values): {fmt(st['difficulty'])}.",
          f"- `oracle_steps` distribution: {fmt(dict(sorted((int(k), v) for k, v in st['oracle_steps'].items() if str(k).isdigit())))}.", ""]

    # ---- 2. Funnel and bottlenecks
    L += ["## 2. Drop reasons and bottlenecks", "",
          "### 2.1 Object funnel (objects remaining after each step)", "",
          "| Scene | Annotated objects of allowed categories | Unique by coarse category | After geometry filters | After blacklist (Step 3 referable) | Visible at >= 1 node (Step 4, usable) | In a cluster of a group |",
          "|---|---|---|---|---|---|---|"]
    tf = Counter()
    for sid in pilots:
        D = S[sid]
        sch = lambda o: o["schemes"][sp["vocab_granularity"]]
        a = [o for o in D["objs"] if o["kind"] == "object" and o["category_merged"] in allowed_cats and o["category_fine"] in allowed_cats]
        b = [o for o in a if "NOT_REFERABLE" not in sch(o)["drop_reasons"]]
        c = [o for o in b if not ({"UNRELIABLE_GEOMETRY", "TOO_SMALL"} & set(sch(o)["drop_reasons"]))]
        d = [o for o in c if o["referable"]]
        vis_n = {o["obj_id"]: o["num_visible_nodes"] for o in D["s4"]["objects"]}
        e = [o for o in d if vis_n[o["obj_id"]] > 0]
        f = {o for g_ in by_scene[sid] for o in g_["cluster_obj_ids"]}
        row = [len(a), len(b), len(c), len(d), len(e), len(f)]
        for k, x in enumerate(row):
            tf[k] += x
        L.append(f"| `{sid}` | " + " | ".join(map(str, row)) + " |")
    L += ["| **Total** | " + " | ".join(f"**{tf[k]}**" for k in range(6)) + " |", "",
          f"**Main bottleneck: uniqueness by coarse category (Step 3).** Of {tf[0]} objects in allowed categories only {tf[1]} are unique in their scene (objects with >= 2 of the same category are not referable; size references are disabled); "
          f"next is visibility (Step 4): {tf[3] - tf[4]} of {tf[3]} referable objects are not visible at any node (mostly small objects with area < 0.02). "
          "Steps 5–6 themselves drop almost no groups: scenes with 3 usable objects can form only one 3-object cluster (`31a2c91c43`, `5748ce6f01`), scenes with 5 usable objects form 6–8 groups.", ""]
    # 2.2 reason codes
    L += ["### 2.2 Drop reasons per step (by reason code)", "",
          "| Step | Item | Reason code | " + " | ".join(f"`{s}`" for s in pilots) + " | Total |", "|---" * (4 + len(pilots)) + "|"]
    keys = Counter()
    per = {sid: Counter() for sid in pilots}
    for sid in pilots:
        for d in S[sid]["drops"]:
            k = (d["step"], d.get("kind", ""), d["reason_code"])
            per[sid][k] += 1
            keys[k] += 1
        n_nv = sum(1 for o in S[sid]["s4"]["objects"] if o["referable"] and o["num_visible_nodes"] == 0)
        per[sid][("step4", "object", "NEVER_VISIBLE (referable but no visible node; derived)")] += n_nv
        keys[("step4", "object", "NEVER_VISIBLE (referable but no visible node; derived)")] += n_nv
    order = ["step1_gate_a", "step2_keyframes", "step3_objects", "step4", "step56"]
    for k in sorted(keys, key=lambda k: (order.index(k[0]) if k[0] in order else 9, -keys[k])):
        L.append(f"| {k[0]} | {k[1]} | `{k[2]}` | " + " | ".join(str(per[s][k]) for s in pilots) + f" | {keys[k]} |")
    L += ["", "- `NOT_REFERABLE` in step3 covers both disallowed categories (wall, floor, ceiling, `object`, etc.) and non-unique categories.",
          "- `AMBIGUOUS_EVIDENCE` in step56 is per candidate question: T2 / T5 option objects are drawn from all referable objects in the scene, and an object that is not visible at any node invalidates the candidate; "
          "it does not cause dropped groups, the count is large because many candidates are enumerated. `EVIDENCE_NOT_LINKED` means the evidence objects cannot be linked in the view graph (primary mode partial_partial).",
          f"- Candidate filter hits (per enumerated candidate, not the number of dropped questions): {fmt(st['candidate_filter_counts'])}. T5 is almost always filtered by the angular gap and the 'objects required on the left' rule (few usable objects, often only 0–1 on the left).",
          "- Scene level: all 5 scenes pass (>= 40 keyframes, >= 3 referable objects, connected view graph); no group was dropped for having too few questions (`GROUP_INCOMPLETE` 0), no cluster for lacking an initial view (`NO_VALID_INIT_VIEW` 0).", ""]

    # ---- 3. unanswerable
    L += ["## 3. unanswerable (masking): attempted for every group", "",
          f"Rules follow the shared code: all nodes where the masked object is visible / partial are removed, as are nodes disconnected from p0; p0 cannot be masked; masked fraction <= {qc['unanswerable_max_masked_frac']:.0%}; "
          f"after masking the other questions of the group must remain answerable with linked evidence; candidate questions are T4(X) or T1(X, cluster object); dataset-wide unanswerable share cap {qc['target_unanswerable_frac'][1]:.0%}. "
          "Each group tries every referable object in the scene in turn.", "",
          "| Scene | Groups | Succeeded | Main reason for failed groups (groups) | Per-attempt failure counts (group × masked object) |", "|---|---|---|---|---|"]
    dom_all, tried, succ = Counter(), 0, 0
    reason_cn = {"masked_frac_too_large": "masked fraction too large", "group_incompatible": "other questions in the group unanswerable / evidence not linked after masking", "p0_masked": "object visible at p0",
                 "never_visible_evidence": "masked object never visible", "margin_fail": "T4 angle / T1 distance margin not met", "duplicate_in_scene": "duplicate within scene",
                 "template_diversity": "fewer than 2 templates after replacement", "evidence_too_large": "too many evidence objects", "dataset_cap_reached": "dataset-wide share cap reached"}
    for sid in pilots:
        sa = st["special_attempts"][sid]["unanswerable"]
        n_g = len(S[sid]["raw"]["groups"])
        ok = len(sa.get("added") or [])
        dom = Counter()
        has_cand = set(sa.get("groups_with_candidate", []))
        added_g = {a["group_index"] for a in (sa.get("added") or [])}
        for gi, fc in sa.get("fail_counts_by_group", {}).items():
            if int(gi) in added_g:
                continue
            if int(gi) in has_cand:
                dom["had a candidate but not selected (each object used once / share cap)"] += 1
                continue
            fc2 = {k: v for k, v in fc.items() if k != "never_visible_evidence"} or fc
            dom[reason_cn.get(max(fc2, key=fc2.get), max(fc2, key=fc2.get)) if fc2 else "no candidate"] += 1
        dom_all.update(dom)
        tried, succ = tried + n_g, succ + ok
        L.append(f"| `{sid}` | {n_g} | {ok} | {fmt(dom)} | {fmt({reason_cn.get(k, k): v for k, v in sa.get('fail_counts', {}).items()})} |")
    L += [f"| **Total** | {tried} | **{succ}** ({succ / max(tried, 1):.0%}) | {fmt(dom_all)} | |", "",
          "'Main reason' = the most frequent failure reason among the group's attempts ('masked object never visible' is only counted when there is no other reason, since it comes from trying to mask referable objects that are never visible and is unrelated to the group).",
          f"unanswerable share {Counter(q['evidence_label'] for q in qs)['unanswerable'] / max(len(qs), 1):.1%} (target {qc['target_unanswerable_frac'][0]:.0%}–{qc['target_unanswerable_frac'][1]:.0%}). "
          "ScanNet++ scenes are small with dense keyframes, so objects are visible / partial at many nodes; masking often exceeds the 30% cap or also masks nodes needed by other questions in the group.", ""]
    init_r = {sid: st["special_attempts"][sid]["init"] for sid in pilots}
    init_txt = "; ".join(f"`{s}` {r.get('reason', '')}" for s, r in init_r.items() if not r.get("added"))
    L += [f"init: {sum(1 for r in init_r.values() if r.get('added'))} questions. {init_txt}. "
          "init requires >= 2 referable objects visible together at the group's p0, but p0 is chosen where most cluster objects are invisible and usable objects are few, so this was not achievable in the pilot.", ""]

    # ---- 4. Balancing
    L += ["## 4. Option-letter and quadrant balance (within the 5 scenes)", "",
          "| Template | Options | Questions | Correct-letter distribution | Max deviation from uniform |", "|---|---|---|---|---|"]
    for t in TEMPLATES:
        for n in (2, 3, 4):
            items = [q for q in qs if q["template"] == t and q["answer_type"] == "multiple_choice" and len(q["options"]) == n]
            if not items:
                continue
            c = Counter(q["answer"] for q in items)
            dev = max(abs(c.get(l, 0) / len(items) - 1 / n) for l in "ABCD"[:n])
            L.append(f"| {t} | {n} | {len(items)} | {fmt(dict(sorted(c.items())))} | {dev:.0%} |")
    L += ["", "Letters are assigned exactly uniformly within each (template, number of options) bucket and then shuffled; deviations only come from question counts not divisible by the number of options.", "",
          "| Template | Questions | front-left | front-right | back-left | back-right | Max deviation from 25% |", "|---|---|---|---|---|---|---|"]
    for t in ("T3", "T4"):
        c = Counter(q["answer_value"] for q in qs if q["template"] == t)
        n = sum(c.values())
        if n:
            L.append(f"| {t} | {n} | " + " | ".join(str(c.get(k, 0)) for k in ("front-left", "front-right", "back-left", "back-right"))
                     + f" | {max(abs(c.get(k, 0) / n - 0.25) for k in g.QUADRANTS):.0%} |")
    shared = variants.get(f"{prim}_shared_quadrant_rule")
    if shared:
        L += ["", f"Quadrant balancing by removal under the shared rule (±{qc['quadrant_balance_tol']:.0%}) would remove {sum(shared['quadrant_rebalance_removed'].values())} questions "
              f"({fmt(shared['quadrant_rebalance_removed'])}), questions {variants[prim]['num_questions']} → {shared['num_questions']}, T4 {variants[prim]['per_template'].get('T4', 0)} → {shared['per_template'].get('T4', 0)}. "
              "With only about 20 questions per template in the pilot, ±5% is nearly impossible, so no questions are removed and only the deviation is reported; the shared rule applies to the full run.", ""]

    # ---- 5. Evidence-link modes
    L += ["## 5. Two evidence-link modes", "",
          "| Mode | Primary | Groups | Questions | " + " | ".join(TEMPLATES) + " |", "|---" * (4 + len(TEMPLATES)) + "|"]
    for k, v in variants.items():
        if "shared_quadrant" in k:
            continue
        L.append(f"| `{v['evidence_link_mode']}` | {'yes' if v['primary'] else ''} | {v['num_groups']} | {v['num_questions']} | "
                 + " | ".join(str(v["per_template"].get(t, 0)) for t in TEMPLATES) + " |")
    L += ["", "Neither mode removes questions for quadrant balance, so they are directly comparable. The difference is concentrated in `c50d2d1d42` (4 groups under partial_partial, 2 under visible_partial).", ""]

    # ---- 6. Scale extrapolation
    a = np.array(list(cands.values()))
    ratios = [("low (candidates per scene × 0.45)", 0.45), ("mid (× 0.625, pilot total)", 0.625), ("high (× 0.80)", 0.80)]
    ex = extrapolate(cands, ratios)
    share = {t: st["per_template"].get(t, 0) / max(st["num_questions"], 1) for t in TEMPLATES}
    L += ["## 6. Scale extrapolation: all 50 nvs_sem_val scenes", "",
          f"1. Using only the annotations of all 50 scenes and the current vocabulary rules (unique by coarse category, allowed category, not a plural annotation, longest official box side >= 0.10 m), referable candidates per scene: median {np.median(a):.0f}, mean {a.mean():.1f}, "
          f"distribution {fmt(dict(sorted(Counter(a.tolist()).items())))} (candidates: scenes).",
          f"2. The 5 pilot scenes: {sum(cands[s] for s in pilots)} annotation candidates → {tot[0]} usable objects (after geometry, blacklist and visibility), ratio {tot[0] / sum(cands[s] for s in pilots):.3f}; "
          "per-scene ratios range from 0.43 to 1.0. The extrapolation uses three levels: 0.45 / 0.625 / 0.80.",
          "3. Pilot relation between usable objects and groups: <= 2 → 0 groups; 3 → 1 group; 4 → 4 groups; 5 → 6–8 groups; >= 6 → the cap of 8 groups per scene. 4.3 questions per group on average.", "",
          "| Case | Groups | Questions | " + " | ".join(TEMPLATES) + " |", "|---" * (3 + len(TEMPLATES)) + "|"]
    for r in ex:
        L.append(f"| {r['name']} | {r['groups']} | {r['questions']:.0f} | " + " | ".join(f"{r['questions'] * share[t]:.0f}" for t in TEMPLATES) + " |")
    lo, hi = ex[0]["questions"], ex[-1]["questions"]
    L += ["", f"**Estimate: about {ex[0]['groups']}–{ex[-1]['groups']} groups and {lo:.0f}–{hi:.0f} questions for the 50 scenes** (templates in pilot proportions; T5 has only 1 question in the pilot, so its extrapolation is close to 0).", "",
          "Comparison with the targets (1000–2000 in total, >= 200 per template, as large a ScanNet++ share as possible), adding the frozen ARKit release arkit_only_v0.11 (104 groups, 425 questions):", "",
          "| Case | Total questions | ScanNet++ share | " + " | ".join(f"{t} (ARKit {ARKIT_V011[t]})" for t in TEMPLATES) + " |", "|---" * (3 + len(TEMPLATES)) + "|"]
    for r in ex:
        L.append(f"| {r['name']} | {ARKIT_V011['questions'] + r['questions']:.0f} | {r['questions'] / (ARKIT_V011['questions'] + r['questions']):.0%} | "
                 + " | ".join(("**{:.0f}**" if ARKIT_V011[t] + r["questions"] * share[t] < 200 else "{:.0f}").format(ARKIT_V011[t] + r["questions"] * share[t]) for t in TEMPLATES) + " |")
    L += ["", f"Bold = below the target of 200 per template. **Total about {425 + lo:.0f}–{425 + hi:.0f} questions; the low case is below the minimum of 1000**; **T5 is far below 200 in all three cases**, T4 is below 200 in the low case, T3 is close to the minimum. "
          "Note that the ScanNet++ numbers are Step 6 results; Step 8 debiasing will prune a few more (3% for ARKit v0.11).", ""]
    base_u = usable_whatif(cfg, S)
    mid = ex[1]
    pilot_groups = lambda u: sum(q_per_scene(x)[0] for x in u.values())

    def whatif_row(u):
        """Pilot: usable objects and groups estimated from the relation; 50 scenes: mid-case estimate with the usable ratio scaled by the pilot increase."""
        r = 0.625 * sum(u.values()) / sum(base_u.values())
        e = extrapolate(cands, [("", r)])[0]
        return (f"usable objects {sum(base_u.values())} → {sum(u.values())} ({'/'.join(str(u[s]) for s in pilots)}), pilot groups {pilot_groups(base_u)} → {pilot_groups(u)}",
                f"mid case {mid['questions']:.0f} → {e['questions']:.0f} questions ({e['questions'] / mid['questions'] - 1:+.0%})")
    wa1 = whatif_row(usable_whatif(cfg, S, visible_min_area_frac=0.01))
    wa2 = whatif_row(usable_whatif(cfg, S, visible_min_area_frac=0.005))
    wf = whatif_row(usable_whatif(cfg, S, visible_min_in_frame_frac=0.40))
    e12 = extrapolate(cands, [("", 0.625)], cap=12)[0]
    per_scene_q = mid["questions"] / 50
    L += ["### Options (estimates only, not executed; to be decided)", "",
          "| Option | Basis (pilot) | Expected gain (50 scenes) | Cost / risk |", "|---|---|---|---|",
          f"| A. Use `nvs_sem_train` (856 scenes with semantic annotations, same kind of data as nvs_sem_val) | about {per_scene_q:.1f} questions per scene | about +{per_scene_q * 100:.0f} questions per 100 scenes; "
          f"about +{per_scene_q * 856:.0f} for all 856 (T5 still rare) | download about 0.7–1.5 GB per scene (0.6–1.3 TB in total) + decoding; needs approval to download; these are official training scenes, which must be stated in the evaluation |",
          f"| B. Visibility area threshold 0.02 → 0.01 (ScanNet++ only) | {wa1[0]} | {wa1[1]} | recovers small objects (tap, clock, plant, etc.) that are even smaller in the image; visibility must be spot-checked again; inconsistent with the ARKit rule |",
          f"| B'. Area threshold → 0.005 | {wa2[0]} | {wa2[1]} | as above, higher risk (in spot-check round 1 small objects were only a few dozen pixels) |",
          f"| C. In-frame fraction 0.60 → 0.40 | {wf[0]} | {wf[1]} | recovers tall objects (cabinets, doors); the round-1 spot-check failure where only the lower half of a refrigerator was shown is of this kind |",
          "| D. Re-allow some categories banned at GATE B (e.g. socket, light switch, book) | these categories usually occur several times per scene, few are unique | not quantified; estimated below B | the reasons for the ban (too small, easily moved) still hold |",
          f"| E. Per-scene group cap 8 → 12 | only helps scenes with >= 6 usable objects (pilot 0 / 5) | mid case {mid['questions']:.0f} → {e12['questions']:.0f} questions ({e12['questions'] / mid['questions'] - 1:+.0%}) | the same objects are asked about repeatedly, raising question correlation |",
          "| F. Restore size references (larger / smaller) | permanently disabled | — | not recommended |", "",
          "Note: the gains for B–E are rough estimates from the pilot usable-object counts and the 'usable objects → groups' relation (per-scene usable objects are rounded and looked up, so B and B' land on the same level and give the same result; coarse, for order of magnitude only); A is the only option that clearly raises both the total and the ScanNet++ share. The T5 shortfall is a constraint of the template itself (it needs several referable objects on the left), "
          "so no option is likely to bring T5 to 200; the T5 target or rules may need to be adjusted.", ""]

    # ---- 7. Cards
    card_dir = rep / "scannetpp_gateC" / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    for f in card_dir.glob("*.jpg"):
        f.unlink()
    L += ["## 7. Cards (3 groups per scene, or all if fewer)", "",
          "Each card: p0 initial view (dark blue title) + evidence views (oracle nodes, link_views, T4 path_views; the title states the role); masked groups add 1 masked node (red title). "
          "Boxes and names of evidence objects: green = visible at that node, orange = partial (invisible not drawn). Questions, options and ground-truth answers are listed below.", ""]
    for sid in pilots:
        gs = by_scene[sid]
        pick = sorted(gs, key=lambda g_: (not g_.get("view_graph_mask"), g_["group_id"]))[:CARDS_PER_SCENE]
        pick = sorted(pick, key=lambda g_: g_["group_id"])
        L += [f"### `{sid}` ({types.get(sid)}, {len(gs)} groups)", ""]
        for g_ in pick:
            fn = f"{sid}_{g_['group_id']}.jpg"
            card(cfg, S[sid], g_, card_dir / fn)
            L += [f"**{g_['group_id']}**: {len(g_['questions'])} questions ({', '.join(q['template'] for q in g_['questions'])})"
                  + (f", masked object {g_['view_graph_mask']['masked_obj']}, {len(g_['view_graph_mask']['masked_nodes'])} masked nodes" if g_.get("view_graph_mask") else ""), "",
                  f"![{g_['group_id']}](scannetpp_gateC/cards/{fn})", ""]

    # ---- 8. Outputs
    prev = rep / "scannetpp_gateC_preview.jsonl"
    with open(prev, "w") as f:
        for g_ in groups:
            f.write(json.dumps(g_, ensure_ascii=False) + "\n")
    L += ["## 8. Outputs", "",
          "- Questions: `work/scannetpp/_step6/test_groups.jsonl` (primary), `step6_stats.json`; comparisons: `_step6_visible_partial/`, `_step6_partial_partial_shared_quadrant_rule/`; "
          "per scene `groups_raw.json` (intermediate clusters and groups).",
          f"- Question preview (all {len(qs)} pilot questions, with ground truth): `reports/scannetpp_gateC_preview.jsonl` (not tracked in git).",
          "- Cards: `reports/scannetpp_gateC/cards/` (not tracked in git).",
          "- Code: `scannetpp_step56.py` (calls the shared `step56_questions.run`; mode and quadrant comparisons), `scannetpp_gateC.py` (this report, cards, preview); "
          "the shared `step56_questions.py` only gained the `run()` entry point, a dataset parameter and per-group unanswerable failure counts (ARKit behaviour unchanged).", ""]
    # ---- 9. Observations
    dup = []
    for g_ in groups:
        c = Counter((q["slots"].get("OBJ1"), q.get("variant"), q.get("answer_obj")) for q in g_["questions"] if q["template"] == "T2")
        dup += [(g_["group_id"], g_["scene_id"], k) for k, v in c.items() if v > 1]
    ua = Counter(q["evidence_label"] for q in qs)["unanswerable"]
    L += ["## 9. Observations (open, not acted on)", "",
          f"1. **Near-duplicate T2 within a group**: in {len(dup)} groups two T2 questions share the reference object, comparison direction and correct answer and differ only in the option set ("
          + "; ".join(f"{gid} reference obj {k[0]} {k[1]}" for gid, _, k in dup) + "). The shared same-scene dedup key is 'reference + direction + option set' (ARKit rule), so this is allowed. "
          "Optional: change the T2 dedup key for ScanNet++ (or both datasets) to 'reference + direction + correct answer'." if dup else "1. No near-duplicate T2 within a group.",
          "2. **Quadrant balance**: the primary pilot result does not remove questions for quadrant balance (Section 4); to be confirmed, together with whether the shared removal-based balancing is restored for the full run.",
          f"3. **Only {ua} unanswerable questions ({ua / max(len(qs), 1):.1%}, target 8%–12%)**, all from 1 scene; mainly because the masked fraction exceeds 30% (dense ScanNet++ keyframes, objects visible at many nodes). "
          "To raise it, `unanswerable_max_masked_frac` could be relaxed for ScanNet++ only (not tried).",
          "4. **0 init questions**: fewer than 2 referable objects are visible together at p0; with few usable objects in the pilot, few are expected in the full run as well.",
          "5. **Only 1 T5 question**: with few usable objects, there are rarely >= 2 referable objects to the left of the standing point; the same holds for ARKit (T5 in v0.11 also comes from only 6 scenes).", ""]
    (rep / "scannetpp_gateC_pilot.md").write_text("\n".join(L) + "\n")
    print("wrote", rep / "scannetpp_gateC_pilot.md", len(groups), len(qs))


def main():
    write(load_config())


if __name__ == "__main__":
    main()
