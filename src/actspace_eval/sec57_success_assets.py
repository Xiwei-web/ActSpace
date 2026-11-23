"""Section 5.7: selection and material export of representative success cases, plus adding raw dataset files to the already exported cases
and packaging them.
Reads only existing logs, the frozen data and the raw datasets; no model is called, no finished figures are drawn (except preview contact
sheets, which are rough sketches for selection), and no annotation or visibility definition is changed.

1. Selection: merged_v1.0 main-table definition (groups in budget_infeasible excluded), active_v2 answers of the closed-source models.
   3 candidates for each of 4 categories; rules in select().
2. Export: the top candidate of each category (S1–S4) is exported to success_assets/ with the same rules as sec57_case_assets; for S4 the
   failure trajectory of the same group is stored as well.
3. raw/: S1–S4 and the already exported C1, C4, C9 each get a raw/ subdirectory with the original dataset files.
4. Sizes are estimated first; over the limit nothing is copied and the exit code is 3.
5. Each case is packed into its own tar.gz, with SHA256SUMS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import sys
import tarfile
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from actspace_build import arkit, geometry as G
from actspace_build.common import path_of, read_json
from actspace_build.obb_convention import draw_box

from .agents import oracle_targets, plan_route
from .data import MOVE_ACTIONS, FrozenData, dataset_of_scene
from .sec53_behavior import ROOT, RUNS
from .sec57_case_assets import CASE_ROOT, CASES, OUT as CASE_OUT, Case, answers_record, dir_listing, export_all, jdump, trajectory_record
from .trajectory import read_episodes

ORDER = ["opus-5.5", "sol", "sonnet-5.5", "terra", "haiku-5.5", "luna"]     # closed-source models from strongest to weakest (main-table order)
STRONG = ORDER[:4]
TYPES = {"S1": "purposeful_evidence", "S2": "correct_abstention", "S3": "pose_reasoning", "S4": "same_group_contrast"}
ALL9 = ("early_answer_arkit_g_000004", "early_answer_arkit_g_000013", "early_answer_arkit_g_000015", "evidence_left_context_arkit_g_000016",
        "evidence_left_context_arkit_g_000040", "evidence_left_context_arkit_g_000070", "wrong_with_evidence_arkit_g_000023",
        "wrong_with_evidence_arkit_g_000053", "wrong_with_evidence_scannetpp_g_000015")     # C1–C9 (export order of sec57_errors)
SUCC_OUT = ROOT / "_reports" / "sec57" / "success_assets"
CAND_OUT = ROOT / "_reports" / "sec57" / "success_candidates"
T1_OK = 0.5              # minimum score for a T1 (numeric) answer to count as "correct"; multiple-choice questions must score 1
S1_RATIO = 1.5           # S1: steps used ≤ this value × steps of the shortest evidence route
S1_MIN_SHORTEST = 3      # S1: the shortest evidence route has at least this many steps (otherwise there is little evidence gathering)
S1_MIN_PROGRESS = 0.6    # S1: while some evidence is unseen, share of successful moves that reduce "graph distance to the nearest viewpoint of unseen evidence"
S2_MIN_STEPS = 5         # S2: "after exploring" = at least this many steps taken and this many distinct viewpoints visited
S2_MIN_NODES = 4
CLEAR_AREA = 0.01        # automatic proxy for "clearly recognisable": in a strictly visible frame seen by the model, object area fraction ≥ this value and in-frame fraction ≥ CLEAR_INFRAME
CLEAR_INFRAME = 0.8
N_CAND = 3
GB = 1 << 30


@lru_cache(maxsize=None)
def vis_of(wd: Path):
    v = np.load(wd / "visibility.npz")
    return ({int(o): i for i, o in enumerate(v["obj_ids"])}, {str(n): i for i, n in enumerate(v["frame_ids"])}, v["state"], v["area_frac"], v["in_frame_frac"])


def graph_dist(scene, masked, src, targets):
    """Number of steps from src to the nearest node in targets on the view graph (masked nodes removed); None if unreachable."""
    if src in targets:
        return 0
    seen, dq = {src}, deque([(src, 0)])
    while dq:
        v, d = dq.popleft()
        for a in MOVE_ACTIONS:
            w = scene.edges[v][a]
            if w is None or w in masked or w in seen:
                continue
            if w in targets:
                return d + 1
            seen.add(w)
            dq.append((w, d + 1))
    return None


class Ep:
    """Summary of one active_v2 episode (used only for selection)."""

    def __init__(self, data: FrozenData, model: str, run: str, ep: list):
        s, end = ep[0], ep[-1]
        self.model, self.run, self.gid = model, run, s["group_id"]
        self.g = g = data.groups[self.gid]
        self.sid, self.ds = g["scene_id"], dataset_of_scene(g["scene_id"])
        self.scene = scene = data.scene(self.sid)
        self.masked = data.masked_nodes(self.gid)
        row, col, st, area, infr = vis_of(scene.wd)
        steps = [r for r in ep if r["type"] == "step"]
        self.seen = [(0, s["start_node"])] + [(r["t"], r["to"]) for r in steps if r["moved"]]
        self.steps_used, self.forced, self.notes = end["steps_used"], end.get("forced"), end.get("notes")
        self.n_nodes = len({n for _, n in self.seen})
        self.n_failed_moves = sum(1 for r in steps if not r["moved"])
        pub = {v: k for k, v in s["question_ids"].items()}
        self.first = {}                      # object → step at which it was first strictly visible
        self.qs = []
        for q in g["questions"]:
            k = pub[q["qid"]]
            unans = q["evidence_label"] == "unanswerable"
            sc, ans = end["scores"].get(k), end["answers"].get(k)
            ev = [int(o) for o in q["evidence_objs"]]
            for o in ev:
                if o not in self.first:
                    self.first[o] = next((t for t, n in self.seen if st[row[o], col[n]] == 2), None)
            numeric = q["answer_type"] == "numeric"
            ok = sc is not None and (sc >= T1_OK if (numeric and not unans) else sc == 1.0)
            seen_all = all(self.first[o] is not None for o in ev)
            clear = min((max((float(area[row[o], col[n]]) for _, n in self.seen if st[row[o], col[n]] == 2 and infr[row[o], col[n]] >= CLEAR_INFRAME), default=0.0)
                         for o in ev), default=0.0)
            self.qs.append({"id": k, "qid": q["qid"], "template": q["template"], "text": q["text"], "options": q.get("options"), "unanswerable": unans,
                            "gold": "cannot_determine" if unans else (q.get("answer_raw") if numeric else q["answer"]), "answer_value": q.get("answer_value"),
                            "answer": ans, "score": sc, "ok": bool(ok), "exact": sc == 1.0, "evidence_objs": ev, "seen_all": seen_all,
                            "evidence_step": max(self.first[o] for o in ev) if seen_all and ev else None,
                            "start_state": {o: int(st[row[o], col[s["start_node"]]]) for o in ev}, "clear": round(clear, 4)})
        self.answerable = [q for q in self.qs if not q["unanswerable"]]
        self.unans = [q for q in self.qs if q["unanswerable"]]

    @property
    def shortest(self) -> int:
        return shortest_steps(self.gid, self)

    def progress(self) -> float | None:
        """While some evidence objects are not yet strictly seen, whether each successful move reduces "graph distance to the nearest viewpoint of unseen evidence"; returns the share that do."""
        row, col, st, _, _ = vis_of(self.scene.wd)
        objs = sorted({o for q in self.answerable for o in q["evidence_objs"]})
        tgt = {o: {n for n in self.scene.nodes if n not in self.masked and st[row[o], col[n]] == 2} for o in objs}
        good = tot = 0
        for (t0, a), (t1, b) in zip(self.seen[:-1], self.seen[1:]):
            unseen = [o for o in objs if self.first[o] is None or self.first[o] > t0]
            if not unseen:
                break
            T = set().union(*(tgt[o] for o in unseen))
            da, db = graph_dist(self.scene, self.masked, a, T), graph_dist(self.scene, self.masked, b, T)
            tot += 1
            good += da is not None and db is not None and db < da
        return good / tot if tot else None


_SHORT = {}


def shortest_steps(gid, e: Ep) -> int:
    if gid not in _SHORT:
        _SHORT[gid] = len(plan_route(e.scene, e.masked, e.g["init_frame"], oracle_targets(e.g))[0])
    return _SHORT[gid]


def load_eps(data) -> dict:
    """{(model, run): {group: Ep}}, without budget_infeasible groups."""
    out = {}
    for m in ORDER:
        base, runs, _ = RUNS[m]
        for run in runs:
            d = {}
            for ep in read_episodes(base / "active_v2" / run / "trajectories.jsonl"):
                if ep[0].get("budget_infeasible") or ep[-1]["type"] != "episode_end":
                    continue
                d[ep[0]["group_id"]] = Ep(data, m, run, ep)
            out[(m, run)] = d
    return out


def cand(e: Ep, cat: str, key: tuple, focus: dict | None, **extra) -> dict:
    return {"cat": cat, "key": key, "model": e.model, "run": e.run, "group_id": e.gid, "scene_id": e.sid, "dataset": e.ds, "steps_used": e.steps_used,
            "forced": e.forced, "shortest_steps": e.shortest, "n_nodes_visited": e.n_nodes, "n_failed_moves": e.n_failed_moves, "n_masked_nodes": len(e.masked),
            "focus_qid": (focus or e.qs[0])["qid"], "questions": e.qs, "notes": e.notes,
            "first_strict_step": {str(o): t for o, t in sorted(e.first.items())}, "all_ok": all(q["ok"] for q in e.qs), "all_exact": all(q["exact"] for q in e.qs), **extra}


def candidates(data, eps: dict) -> dict:
    out = {k: [] for k in TYPES}
    strong = [(mr, d) for mr, d in eps.items() if mr[0] in STRONG]
    mi = ORDER.index
    for (m, run), d in strong:
        for e in d.values():
            ans_ok = all(q["ok"] and q["seen_all"] for q in e.answerable)
            clear = min((q["clear"] for q in e.answerable), default=0.0)
            # S1 purposeful evidence gathering
            # (the automatic "clearly recognisable" and "progress towards evidence" metrics are used only for ranking, not as hard conditions: very few groups would pass them as hard conditions)
            if not e.unans and e.answerable and ans_ok and e.shortest >= S1_MIN_SHORTEST and e.steps_used <= S1_RATIO * e.shortest:
                p = e.progress() or 0.0
                exact = all(q["exact"] for q in e.qs)
                out["S1"].append(cand(e, "S1", (-(clear >= CLEAR_AREA), -(p >= S1_MIN_PROGRESS), -exact, -round(p, 3), round(e.steps_used / e.shortest, 3), mi(m)), None,
                                      progress=round(p, 3), step_ratio=round(e.steps_used / e.shortest, 3), clear=clear))
            # S2 correct abstention
            if e.unans and e.answerable and ans_ok and all(q["ok"] for q in e.unans) and e.steps_used >= S2_MIN_STEPS and e.n_nodes >= S2_MIN_NODES:
                cov = e.n_nodes / max(1, len(e.scene.nodes) - len(e.masked))
                exact = all(q["exact"] for q in e.qs)
                out["S2"].append(cand(e, "S2", (-(clear >= CLEAR_AREA), -exact, -round(cov, 3), mi(m)), e.unans[0], coverage=round(cov, 3), clear=clear))
            # S3 pose reasoning: T4, target not visible at START (level 0), answered correctly after reaching a strictly visible viewpoint
            t4 = [q for q in e.answerable if q["template"] == "T4" and q["ok"] and q["seen_all"] and all(v == 0 for v in q["start_state"].values())
                  and (q["evidence_step"] or 0) >= 1 and q["clear"] >= CLEAR_AREA]
            if t4:
                q = max(t4, key=lambda q: ("back" in str(q["answer_value"]), q["clear"]))
                rest = all(x["ok"] for x in e.qs) and ans_ok
                out["S3"].append(cand(e, "S3", (-rest, -("back" in str(q["answer_value"])), -round(sum(x["ok"] for x in e.qs) / len(e.qs), 3), bool(e.masked),
                                                -q["clear"], mi(m)), q, target_quadrant=q["answer_value"], clear=q["clear"], rest_ok=rest))
    # S4 same-group contrast: the same groups as C1–C9, solved by a stronger model (skipped when the failing model is opus-5.5, since there is no stronger one)
    exported = {n for _, n in CASES}
    for i, name in enumerate(ALL9, 1):
        src = read_json(CASE_ROOT / name / "trajectory.json")
        for (m, run), d in eps.items():
            e = d.get(src["group_id"])
            if e is None or mi(m) >= mi(src["model"]):
                continue
            fq = next(q for q in e.qs if q["qid"] == src["focus_qid"])
            if not (fq["ok"] and fq["seen_all"]):
                continue
            rest = all(q["ok"] for q in e.qs) and all(q["seen_all"] for q in e.answerable)
            out["S4"].append(cand(e, "S4", (0, -(name in exported), -rest, -round(sum(x["ok"] for x in e.qs) / len(e.qs), 3), mi(m)), fq, rest_ok=rest,
                                  failure={"case": f"C{i}", "name": name, "model": src["model"], "run": src["run"], "situation": src["situation"],
                                           "steps_used": src["steps_used"], "exported": name in exported}))
    # S4 fallback: same-group pairs "weak model fails, strong model succeeds" outside C1–C9 (used only if fewer than 3 groups were found above)
    c_groups = {c["group_id"] for c in out["S4"]}
    for (m, run), d in strong:
        for e in d.values():
            if e.gid in c_groups or not (all(q["ok"] for q in e.qs) and all(q["seen_all"] for q in e.answerable)):
                continue
            for wm in ("haiku-5.5", "luna"):
                w = eps[(wm, RUNS[wm][1][0])].get(e.gid)
                bad = [q for q in (w.answerable if w else []) if not q["ok"]]
                if bad:
                    fq = next(q for q in e.qs if q["qid"] == bad[0]["qid"])
                    out["S4"].append(cand(e, "S4", (1, 0, -len(bad), mi(m), 0), fq, rest_ok=True,
                                          failure={"case": None, "name": None, "model": wm, "run": w.run, "situation": "other_pair", "steps_used": w.steps_used, "exported": False}))
                    break
    for k, L in out.items():          # keep only the best episode per group in each category
        best = {}
        for c in sorted(L, key=lambda c: (c["key"], c["group_id"])):
            best.setdefault(c["group_id"], c)
        out[k] = sorted(best.values(), key=lambda c: (c["key"], c["group_id"]))
    return out


def select(pool: dict) -> dict:
    """3 candidates per category. Top candidate: among the top 10 of each category, the combination with the smallest sum of ranks, all scenes different, both datasets present; 2nd and 3rd: categories take turns picking the next one whose scene is unused,
    preferring a dataset not yet present in the category (searched only within the category's top 10)."""
    tops = {k: pool[k][:10] for k in TYPES}
    best = None
    for combo in itertools.product(*(range(len(tops[k])) for k in TYPES)):
        cs = [tops[k][i] for k, i in zip(TYPES, combo)]
        if len({c["scene_id"] for c in cs}) < len(cs) or len({c["dataset"] for c in cs}) < 2:
            continue
        if best is None or sum(combo) < best[0]:
            best = (sum(combo), cs)
    if best is None:
        raise SystemExit("no combination of top candidates with 4 different scenes covering both datasets; see the candidate counts")
    sel = {k: [c] for k, c in zip(TYPES, best[1])}
    used = {c["scene_id"] for c in best[1]}
    for _ in range(N_CAND - 1):
        for k in TYPES:
            have = {c["dataset"] for c in sel[k]}       # prefer a dataset not yet present in the category (only within the category's top 10), otherwise the next one
            nxt = (next((c for c in pool[k][:10] if c["scene_id"] not in used and c["dataset"] not in have), None)
                   or next((c for c in pool[k] if c["scene_id"] not in used), None))
            if nxt is not None:
                sel[k].append(nxt)
                used.add(nxt["scene_id"])
    for k, L in sel.items():
        for r, c in enumerate(L, 1):
            c["rank"], c["pool_rank"], c["pool_size"] = r, pool[k].index(c) + 1, len(pool[k])
    return sel


# ---------------------------------------------------------------- preview contact sheet (rough sketch, only for selection)
def coarse_topdown(c: Case, res=0.02):
    import open3d as o3d
    c.topdown_shape()
    V, F, col = c._mesh
    lo, hi = c._td
    W, H = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x, y = lo[0] + (u + 0.5) * res, hi[1] - (v + 0.5) * res
    z0 = float(np.median([c.scene.pose[n][0][2] for n in c.scene.nodes])) + 0.5
    rays = np.stack([x, y, np.full_like(x, z0), np.zeros_like(x), np.zeros_like(x), -np.ones_like(x)], -1).reshape(-1, 6).astype(np.float32)
    r = arkit.raycasting_scene(V, F).cast_rays(o3d.core.Tensor(rays))
    t, pid, uv = r["t_hit"].numpy(), r["primitive_ids"].numpy(), r["primitive_uvs"].numpy()
    hit = np.isfinite(t)
    img = np.full((H * W, 3), 255, np.uint8)
    if col is not None:
        tri = F[pid[hit]]
        b1, b2 = uv[hit, 0:1], uv[hit, 1:2]
        img[hit] = np.clip(np.round(((1 - b1 - b2) * col[tri[:, 0]] + b1 * col[tri[:, 1]] + b2 * col[tri[:, 2]]) * 255), 0, 255).astype(np.uint8)
    else:
        img[hit] = 150
    img = (0.55 * img.reshape(H, W, 3) + 0.45 * 255).astype(np.uint8)[:, :, ::-1].copy()      # lightened, BGR
    return img, (lambda p: (int(round((p[0] - lo[0]) / res)), int(round((hi[1] - p[1]) / res))))


def preview(c: Case, cd: dict, path: Path, fail: Case | None = None, first: list | None = None):
    """first: [(step, [object ids])], placed before the other evidence frames (default: by the step at which evidence was first strictly visible)."""
    name = {int(o["obj_id"]): o for o in c.objects}
    td, px = coarse_topdown(c)
    sp = plan_route(c.scene, c.masked, c.g["init_frame"], oracle_targets(c.g))[0]
    pts = lambda nodes: np.array([px(c.scene.pose[n][0]) for n in nodes], np.int32).reshape(-1, 1, 2)     # noqa: E731
    cv2.polylines(td, [pts([c.g["init_frame"]] + [n for _, n in sp])], False, (0, 160, 0), 2, cv2.LINE_AA)          # green: shortest evidence route
    if fail is not None:
        cv2.polylines(td, [pts(fail.route)], False, (0, 0, 220), 2, cv2.LINE_AA)                                    # red: failure trajectory
    cv2.polylines(td, [pts(c.route)], False, (220, 80, 0), 2, cv2.LINE_AA)                                          # blue: model route
    for n in dict.fromkeys(c.route):
        cv2.circle(td, px(c.scene.pose[n][0]), 3, (220, 80, 0), -1, cv2.LINE_AA)
    p0, yaw = c.scene.pose[c.route[0]]
    cv2.circle(td, px(p0), 7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(td, px(p0), px(p0 + 0.5 * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0])), (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.4)
    for o in c.ev_objs:
        ob = name[o]["obb"]
        ce, ax, sz = np.asarray(ob["center"], float), np.asarray(ob["axes"], float), np.asarray(ob["size"], float)
        cor = [ce + 0.5 * (sx * sz[0] * ax[0] + sy * sz[1] * ax[1]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        cv2.polylines(td, [np.array([px(p) for p in cor], np.int32).reshape(-1, 1, 2)], True, (200, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(td, f"{o}", px(ce), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1, cv2.LINE_AA)
    tiles = []

    def tile(n, label, objs=()):
        K, T, _, _ = c.camera(n)
        bgr = np.ascontiguousarray(c.scene.image(n)[:, :, ::-1])
        for o in objs:
            ob = name[o]["obb"]
            draw_box(bgr, np.asarray(ob["center"], float), np.asarray(ob["axes"], float), np.asarray(ob["size"], float), K, T, (0, 255, 0))
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (255, 255, 255), -1)
        cv2.putText(bgr, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(bgr)

    tile(c.route[0], "START")
    step_node = {0: c.route[0], **{r["t"]: r["to"] for r in c.steps}}
    fq = next(q for q in cd["questions"] if q["qid"] == cd["focus_qid"])
    order = list(fq["evidence_objs"]) + [o for q in cd["questions"] for o in q["evidence_objs"]]
    by_step = {}
    for o in dict.fromkeys(order):
        t = cd["first_strict_step"].get(str(o))
        if t is not None:
            by_step.setdefault(t, []).append(o)
    shown = [(t, o) for t, o in (first or [])] + [(t, o) for t, o in by_step.items() if t not in {x for x, _ in (first or [])}]
    for t, objs in shown[:3]:
        tile(step_node[t], f"step {t}: " + ", ".join(f"{(name[o].get('ref_name') or name[o].get('category'))} ({o})" for o in objs)[:48], objs)
    Hh = 640
    fit = lambda im: cv2.resize(im, (round(im.shape[1] * Hh / im.shape[0]), Hh), interpolation=cv2.INTER_AREA)     # noqa: E731
    row = np.hstack([fit(t) for t in tiles] + [fit(td)])
    head = np.full((34, row.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{cd['cat']} #{cd['rank']}  {cd['group_id']}  {cd['model']}  steps {cd['steps_used']} / shortest {cd['shortest_steps']}   "
                      "blue = model route, green = shortest evidence route" + (", red = failure route" if fail is not None else "") + ", magenta = evidence objects",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([head, row]), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- raw/: original dataset files
def link_or_copy(src: Path, dst: Path, twin: Path | None = None):
    """Hard-link when twin is the same file already present in the case directory (stored only once in the tar); otherwise copy from the raw dataset."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if twin is not None and twin.exists() and twin.stat().st_size == src.stat().st_size:
        try:
            os.link(twin, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def raw_frames(c: Case) -> list[str]:
    return sorted(set(c.route) | {n for o in c.ev_objs for n in c.best_frames(o)})


def raw_plan(c: Case) -> list[tuple[Path, str, Path | None]]:
    """[(file in the raw dataset, relative path under raw/, relative path of the same file already in the case directory)], excluding per-frame text extracts."""
    fr = raw_frames(c)
    P = []
    if c.ds == "arkitscenes":
        f3 = arkit.scene_files_3dod(path_of(c.cfg, "arkit_3dod_root"), c.rid)
        P += [(f3["annotation"], f3["annotation"].name, Path("annotations") / f"raw_{f3['annotation'].name}"), (f3["mesh"], f3["mesh"].name, Path("geometry") / f3["mesh"].name),
              (f3["lowres_wide.traj"], "lowres_wide.traj", None)]
        intr_dir = c.image_root / Path(c.frames[fr[0]]["rel_path"]).parent.parent / "vga_wide_intrinsics"
        intr = arkit.list_frames(intr_dir, ".pincam")
        its = np.array([t for t, _ in intr])
        c._raw_rows = []
        for n in fr:
            f = c.frames[n]
            src = c.image_root / f["rel_path"]
            pin = intr[int(np.argmin(np.abs(its - f["timestamp"])))][1]
            P += [(src, f"vga_wide/{src.name}", None), (intr_dir / pin, f"vga_wide_intrinsics/{pin}", None)]
            sd = c.sensor_depth_raw(n)
            dn = None
            if sd is not None:
                dn = sd[1].name
                P += [(sd[1], f"lowres_depth/{dn}", None), (f3["lowres_wide_intrinsics"] / dn.replace(".png", ".pincam"), f"lowres_wide_intrinsics/{dn.replace('.png', '.pincam')}", None)]
            c._raw_rows.append({"frame_id": n, "timestamp": f["timestamp"], "vga_wide": f"vga_wide/{src.name}", "vga_wide_intrinsics": f"vga_wide_intrinsics/{pin}",
                                "lowres_depth": f"lowres_depth/{dn}" if dn else "", "lowres_depth_dt_s": round(sd[2], 4) if sd else "",
                                "rotation_cw_quarters": f["rotation_cw_quarters"], "on_route": int(n in c.route)})
    else:
        from actspace_build.scannetpp import scene_paths, sp_root
        sc = scene_paths(c.cfg, c.rid)
        mesh = Path(sc.scan_mesh_path)
        sem = mesh.with_name(mesh.stem + "_semantic.ply")
        P += [(Path(sc.scan_anno_json_path), "scans/segments_anno.json", Path("annotations") / "raw_segments_anno.json"),
              (Path(sc.scan_mesh_segs_path), "scans/segments.json", Path("annotations") / "raw_segments.json"),
              (mesh, f"scans/{mesh.name}", Path("geometry") / mesh.name), (sem, f"scans/{sem.name}", None)]
        c._raw_rows = []
        for n in fr:
            f = c.frames[n]
            src, dp = sp_root(c.cfg) / f["rel_path"], sp_root(c.cfg) / f["depth_rel_path"]
            P += [(src, f"iphone/rgb/{src.name}", None)] + ([(dp, f"iphone/depth/{dp.name}", None)] if dp.exists() else [])
            c._raw_rows.append({"frame_id": n, "source_frame": f["source_frame"], "timestamp": f["timestamp"], "rgb": f"iphone/rgb/{src.name}",
                                "depth": f"iphone/depth/{dp.name}" if dp.exists() else "", "rotation_cw_quarters": f["rotation_cw_quarters"], "on_route": int(n in c.route)})
    return [(s, r, t) for s, r, t in dict.fromkeys(P)]


def raw_estimate(c: Case, existing: Path | None) -> int:
    """Bytes added by raw/ (upper bound): files already in the case directory (hard links) are not counted; the ScanNet++ pose_intrinsic_imu extract counts 4 KB per frame."""
    return sum(s.stat().st_size for s, _, t in raw_plan(c) if not (t is not None and existing is not None)) + 4096 * len(raw_frames(c)) + (1 << 20)


def export_raw(c: Case, case_dir: Path) -> dict:
    d = case_dir / "raw"
    d.mkdir()
    plan = raw_plan(c)
    src_rows, miss = [], []
    for s, rel, twin in plan:
        if not s.exists():
            miss.append(str(s))
            continue
        how = link_or_copy(s, d / rel, case_dir / twin if twin is not None else None)
        src_rows.append([f"raw/{rel}", str(s), s.stat().st_size, how, str(twin) if twin is not None else ""])
    rows = c._raw_rows
    if c.ds == "arkitscenes":
        A = path_of(c.cfg, "arkit_3dod_root").resolve().parents[1]
        for nm, p in (("metadata_3dod.csv", A / "3dod" / "metadata.csv"), ("metadata_raw.csv", A / "raw" / "metadata.csv")):
            L = p.read_text().splitlines()
            keep = [L[0]] + [x for x in L[1:] if x.split(",")[0] == c.rid]
            (d / nm).write_text("\n".join(keep) + "\n")
            src_rows.append([f"raw/{nm}", f"{p} (header + {len(keep) - 1} rows with video_id = {c.rid})", (d / nm).stat().st_size, "extract", ""])
        f3 = arkit.scene_files_3dod(path_of(c.cfg, "arkit_3dod_root"), c.rid)
        lines = sorted((float(x.split()[0]), x) for x in f3["lowres_wide.traj"].read_text().splitlines() if x.strip())
        ts = np.array([t for t, _ in lines])
        ac = c.cfg["arkitscenes"]
        for r in rows:
            idx = G.pose_indices_at_timestamp(r["timestamp"], ts, ac["vga_pose_nearest_tol_s"], ac["vga_pose_interp_max_gap_s"])
            js, alpha = idx if idx is not None else ([], None)
            r.update({"traj_rule": "nearest" if alpha is None else "interpolate", "traj_alpha": "" if alpha is None else round(float(alpha), 6),
                      "traj_line_a": lines[js[0]][1] if js else "", "traj_line_b": lines[js[1]][1] if len(js) > 1 else ""})
        src_rows.append(["raw/frame_map.csv", f"{f3['lowres_wide.traj']} (the 1 or 2 original lines used per frame) + frames_all.json", 0, "extract", "annotations/view_graph.json"])
    else:
        from actspace_build.scannetpp import scene_paths
        pp = Path(scene_paths(c.cfg, c.rid).iphone_pose_intrinsic_imu_path)
        poses = read_json(pp)
        jdump(d / "iphone" / "pose_intrinsic_imu_frames.json", {r["source_frame"]: poses[r["source_frame"]] for r in rows})
        src_rows.append(["raw/iphone/pose_intrinsic_imu_frames.json", f"{pp} (only the entries of these {len(rows)} frames, keyed by original frame name)", (d / "iphone" / "pose_intrinsic_imu_frames.json").stat().st_size,
                         "extract", "annotations/view_graph.json"])
    with open(d / "frame_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(d / "sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "source_path", "bytes", "how", "same_file_in_case_dir"])
        w.writerows(src_rows)
    n_depth = sum(1 for r in rows if r.get("lowres_depth") or r.get("depth"))
    L = [f"# raw/: original dataset files of {c.tag}", "",
         f"Scene `{c.sid}` ({c.ds}). All files here are original files of the raw dataset (copied as is, or hard-linked to the same file already in the case directory), or verbatim lines / frames extracted from original files. "
         "The source path of every file is listed in `sources.csv`.", "",
         f"- Per-frame files cover only the {len(rows)} frames used by the route (`images/route/`) and the evidence frames (`images/evidence/`); frame id ↔ original file name in `frame_map.csv`.", ""]
    ident = c.ds != "arkitscenes" or bool(np.allclose(np.asarray(c.meta["R_up"], float), np.eye(3)))
    if c.ds == "arkitscenes":
        L += ["## Files", "", "| File | Source |", "|---|---|",
              f"| `{c.rid}_3dod_annotation.json` | `3dod/Validation/{c.rid}/`, original 3D box annotation |",
              f"| `{c.rid}_3dod_mesh.ply` | same directory, original mesh (same file as in `geometry/`) |",
              "| `metadata_3dod.csv`, `metadata_raw.csv` | header and this scene's rows of `3dod/metadata.csv`, `raw/metadata.csv` |",
              "| `lowres_wide.traj` | the scene's full pose file (per line: timestamp, 3 axis-angle values, 3 translation values) |",
              f"| `vga_wide/*.png` | `raw/Validation/{c.rid}/vga_wide/`, original 640×480 images (not upright), {len(rows)} images |",
              "| `vga_wide_intrinsics/*.pincam` | intrinsics with the same name (nearest timestamp): width height fx fy cx cy |",
              f"| `lowres_depth/*.png` | `3dod/…_frames/lowres_depth/`, 256×192, 16-bit, millimeters; {n_depth} / {len(rows)} frames have depth within 0.02 s |",
              "| `lowres_wide_intrinsics/*.pincam` | 256×192 intrinsics with the same name as the depth map |",
              "| `frame_map.csv` | frame id ↔ original file name, depth time offset, traj lines used for the pose |", "",
              "## Correspondence with annotations/", "",
              "- Objects: `source_uid` in `annotations/objects.json` = `data[].uid` in `*_3dod_annotation.json`; `source_label` = `data[].label`. "
              "The OBB comes from `segments.obbAligned`: `centroid` → center, `axesLengths` → size, `normalizedAxes` arranged row-wise as 3×3 → axes (boxes may have been normalised when building the questions; `objects.json` is authoritative).",
              "- Frames: `frame_id` in `frame_map.csv` = node id in `annotations/view_graph.json`; `vga_wide` is its original image.",
              f"- Image orientation: rotating the original image {90 * int(c.frames[rows[0]['frame_id']]['rotation_cw_quarters'])}° clockwise gives the image in `images/`; `K` and `T_wc` in `view_graph.json` are rotated accordingly, "
              "so they are not the same numbers as the `.pincam` files here (not upright).",
              f"- Pose: each traj line (axis-angle, translation) forms a 4×4 matrix M; this scene's convention is `{c.meta.get('traj_convention')}` (`inv_opencv` means T_wc = inverse of M, OpenCV camera frame), "
              f"then left-multiplied by R_up ({'identity for this scene' if ident else 'see R_up in annotations/scene_meta.json'}). If a frame's timestamp is within 0.02 s of the nearest traj line, that line is used directly (`traj_rule = nearest`), "
              "otherwise the pose is interpolated between the two neighbouring lines (`interpolate`; `traj_alpha` is the weight of the later line).",
              "- World frame: meters, z up; " + ("the original mesh and annotations are already in this frame (R_up is the identity)." if ident else "original mesh vertices and annotations must be right-multiplied by the transpose of R_up to get into this frame (`geometry/mesh_zup.ply` is already converted).")]
    else:
        L += ["## Files", "", "| File | Source |", "|---|---|",
              f"| `scans/segments_anno.json`, `scans/segments.json` | `data/{c.rid}/scans/`, original instance annotation and vertex segmentation |",
              "| `scans/mesh_aligned_0.05.ply` | original mesh (same file as in `geometry/`) |",
              "| `scans/mesh_aligned_0.05_semantic.ply` | original semantic mesh |",
              f"| `iphone/rgb/*.jpg` | `data/{c.rid}/iphone/rgb/`, 1920×1440, {len(rows)} images |",
              f"| `iphone/depth/*.png` | `data/{c.rid}/iphone/depth/`, 256×192, 16-bit, millimeters, {n_depth} images |",
              "| `iphone/pose_intrinsic_imu_frames.json` | entries of these frames in `iphone/pose_intrinsic_imu.json` (timestamp, pose, intrinsic, imu, aligned_pose) |",
              "| `frame_map.csv` | frame id ↔ original frame name and files |", "",
              "## Correspondence with annotations/", "",
              "- The single-frame files in `iphone/rgb/` and `iphone/depth/` were decoded at build time from `rgb.mkv` and `depth.bin` with the dataset's own tools; `rgb.mkv` (about 210 MB) and `depth.bin` (about 333 MB) were not copied.",
              "- Objects: `source_object_id` in `annotations/objects.json` = `segGroups[].objectId` in `segments_anno.json`; `source_label` = `segGroups[].label`. "
              "An object's vertices = the mesh vertices of the segments listed in `segGroups[].segments`, looked up in `segIndices` of `segments.json`; the OBB was computed from these vertices at build time and is not part of the raw data.",
              "- Frames: `frame_id` in `frame_map.csv` = node id in `annotations/view_graph.json`; `source_frame` is the original frame name (the key in `pose_intrinsic_imu_frames.json`).",
              f"- Pose: `aligned_pose` is directly `T_wc` (camera-to-world, OpenCV camera frame; convention `{c.meta.get('traj_convention')}`), `intrinsic` is K of the 1920×1440 image. "
              "The images in `images/` are scaled to 640×480, so their K is 1/3 of it (`upright.K` in `view_graph.json`). The images need no rotation.",
              "- Depth and RGB share the frame index and the field of view; intrinsics scale with resolution.",
              "- World frame: meters, z up; the original mesh is already in this frame."]
    if miss:
        L += ["", "## Missing files", ""] + [f"- `{m}` does not exist in the raw dataset." for m in miss]
    (d / "README.md").write_text("\n".join(L) + "\n")
    return {"n_frames": len(rows), "n_depth": n_depth, "missing": miss, "bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file()),
            "n_files": sum(1 for p in d.rglob("*") if p.is_file()), "hardlinked": [r[0] for r in src_rows if r[3] == "hardlink"]}


def audit_existing(d: Path) -> dict:
    """Which original dataset files the already exported failure-case directory contained."""
    return {"raw_mesh_in_geometry": sorted(p.name for p in (d / "geometry").glob("*") if p.name not in ("mesh_zup.ply", "geometry_info.json")),
            "raw_annotation_in_annotations": sorted(p.name for p in (d / "annotations").glob("raw_*")), "had_raw_dir": (d / "raw").exists()}


def package(root: Path, names: list[str]) -> list[dict]:
    pk = root / "_packages"
    pk.mkdir(exist_ok=True)
    out = []
    for n in names:
        t = pk / f"{n}.tar.gz"
        with tarfile.open(t, "w:gz", compresslevel=6) as tf:
            tf.add(root / n, arcname=n)
        h = hashlib.sha256()
        with open(t, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        out.append({"file": str(t), "bytes": t.stat().st_size, "sha256": h.hexdigest()})
        print(f"== packed {t.name}: {t.stat().st_size / (1 << 20):.1f} MB", flush=True)
    (pk / "SHA256SUMS").write_text("".join(f"{o['sha256']}  {Path(o['file']).name}\n" for o in out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-case-gb", type=float, default=3.0)
    ap.add_argument("--max-total-gb", type=float, default=12.0)
    ap.add_argument("--select-only", action="store_true", help="only select and print candidate counts; write no files")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    data = FrozenData(verify=not a.no_verify)
    eps = load_eps(data)
    pool = candidates(data, eps)
    for k, L in pool.items():
        print(f"== {k}: {len(L)} qualifying groups (arkitscenes {sum(c['dataset'] == 'arkitscenes' for c in L)}, scannetpp {sum(c['dataset'] == 'scannetpp' for c in L)})", flush=True)
    sel = select(pool)
    for k, L in sel.items():
        for c in L:
            print(f"   {k} #{c['rank']} (rank {c['pool_rank']} / {c['pool_size']} in category) {c['group_id']} {c['model']} {c['run']} steps {c['steps_used']} / shortest {c['shortest_steps']} key {c['key']}"
                  + (f" failure case {c['failure']['case']} {c['failure']['model']}" if k == "S4" else ""), flush=True)
    if a.select_only:
        return
    for p in (SUCC_OUT, CAND_OUT):
        if p.exists():
            print(f"{p} already exists; refusing to overwrite")
            sys.exit(2)
    done = [CASE_OUT / f"{t}_{n}" for t, n in CASES]
    if any((d / "raw").exists() for d in done) or (CASE_OUT / "_packages").exists():
        print("case_assets already contains raw/ or _packages/; refusing to overwrite")
        sys.exit(2)
    # preview contact sheets and candidate list
    CAND_OUT.mkdir(parents=True)
    spec = lambda c: {"model": c["model"], "run": c["run"], "group_id": c["group_id"], "focus_qid": c["focus_qid"], "situation": TYPES[c["cat"]]}     # noqa: E731
    fail_case = lambda c: (Case(data, c["failure"]["case"], c["failure"]["name"]) if c["failure"]["name"] else     # noqa: E731
                           Case(data, "F", "failure", {"model": c["failure"]["model"], "run": c["failure"]["run"], "group_id": c["group_id"],
                                                       "focus_qid": c["focus_qid"], "situation": "failure_of_pair"}))
    cases = {}
    for k, L in sel.items():
        for c in L:
            name = f"{k}_{TYPES[k]}_{c['group_id']}"
            cs = Case(data, k, name, spec(c))
            c["preview"] = str(CAND_OUT / "previews" / f"{k}_rank{c['rank']}_{c['group_id']}_{c['model']}.jpg")
            preview(cs, c, Path(c["preview"]), fail_case(c) if k == "S4" else None)
            if c["rank"] == 1:
                cases[k] = (cs, c)
            else:
                del cs._mesh
    others = []
    for i, n in enumerate(ALL9, 1):
        t = read_json(CASE_ROOT / n / "trajectory.json")
        others.append({"case": f"C{i}", "name": n, "exported": n in {x for _, x in CASES}, **{x: t[x] for x in ("situation", "model", "run", "group_id", "scene_id", "dataset", "steps_used", "focus_qid")},
                       "focus_evidence": [o["name"] for o in t["focus_evidence_objects"]]})
    jdump(CAND_OUT / "candidates.json", {"rules": {"T1_OK": T1_OK, "S1_RATIO": S1_RATIO, "S1_MIN_SHORTEST": S1_MIN_SHORTEST, "S1_MIN_PROGRESS": S1_MIN_PROGRESS, "S2_MIN_STEPS": S2_MIN_STEPS,
                                                   "S2_MIN_NODES": S2_MIN_NODES, "CLEAR_AREA": CLEAR_AREA, "CLEAR_INFRAME": CLEAR_INFRAME, "strength_order": ORDER},
                                         "pool_sizes": {k: {"n": len(L), "arkitscenes": sum(c["dataset"] == "arkitscenes" for c in L), "scannetpp": sum(c["dataset"] == "scannetpp" for c in L)}
                                                        for k, L in pool.items()},
                                         "pool_top": {k: [{x: c[x] for x in ("group_id", "scene_id", "dataset", "model", "run", "steps_used", "shortest_steps", "key")} for c in L[:15]]
                                                      for k, L in pool.items()},
                                         "selected": sel, "failure_cases": others})
    print(f"== candidate list and {sum(len(L) for L in sel.values())} preview images written to {CAND_OUT}", flush=True)
    # size estimate
    old = {t: Case(data, t, n) for t, n in CASES}
    est = {k: {**cs.estimate(), "raw": raw_estimate(cs, Path("x"))} for k, (cs, _) in cases.items()}
    for k in est:
        est[k]["package"] = int(0.9 * sum(est[k].values()))
    for (t, n), d in zip(CASES, done):
        r = raw_estimate(old[t], d)
        est[t] = {"raw": r, "package": int(0.9 * (r + dir_listing(d)["total_bytes"]))}
    for k, e in est.items():
        print(f"== size estimate {k}: new total {sum(e.values()) / GB:.3f} GB; " + ", ".join(f"{x} {v / (1 << 20):.1f} MB" for x, v in e.items()), flush=True)
    tot = sum(sum(e.values()) for e in est.values())
    over = [k for k, e in est.items() if sum(e.values()) + (dir_listing(CASE_OUT / f"{k}_{dict(CASES)[k]}")["total_bytes"] if k in old else 0) > a.max_case_gb * GB]
    print(f"== size estimate, total new in this run: {tot / GB:.3f} GB (limits: {a.max_case_gb} GB per case, {a.max_total_gb} GB new in total)", flush=True)
    jdump(CAND_OUT / "size_estimate.json", est)
    if over or tot > a.max_total_gb * GB:
        print(f"over the limit ({over}); nothing copied")
        sys.exit(3)
    # export S1–S4
    summary = {"estimate_bytes": est, "success": {}, "failure_raw": {}}
    SUCC_OUT.mkdir(parents=True)
    for k, (cs, c) in cases.items():
        d = SUCC_OUT / cs.name
        d.mkdir()
        info = export_all(cs, d)
        if k == "S4":
            fc = fail_case(c)
            jdump(d / "episode" / "failure_trajectory.json", {
                "note": "failure trajectory of the same group (same scene and coordinate frame as trajectory.json). world is the ground-truth pose of the node reached at each step; the images of this trajectory are not copied into this directory"
                        + ("; the full material is in case_assets/" + f"{c['failure']['case']}_{c['failure']['name']}/" if c["failure"]["exported"] else ""),
                "failure_case": c["failure"], "trajectory": trajectory_record(fc, images=False), "answers": answers_record(fc)})
        info["raw"] = export_raw(cs, d)
        info.update(dir_listing(d))
        info["candidate"] = {x: c[x] for x in c if x not in ("questions", "notes")}
        summary["success"][k] = info
        del cs._mesh
        print(f"== {k} raw/: {info['raw']['n_files']} files, {info['raw']['bytes'] / (1 << 20):.1f} MB; case total {info['total_bytes'] / (1 << 20):.1f} MB", flush=True)
    # add raw/ to C1, C4, C9
    for (t, n), d in zip(CASES, done):
        before = audit_existing(d)
        r = export_raw(old[t], d)
        summary["failure_raw"][t] = {"dir": str(d), "name": n, "before": before, "raw": r, **dir_listing(d)}
        print(f"== {t} raw/: {r['n_files']} files, {r['bytes'] / (1 << 20):.1f} MB; case total {summary['failure_raw'][t]['total_bytes'] / (1 << 20):.1f} MB", flush=True)
    summary["packages"] = {"success": package(SUCC_OUT, [cs.name for cs, _ in cases.values()]), "failure": package(CASE_OUT, [f"{t}_{n}" for t, n in CASES])}
    jdump(SUCC_OUT / "success_assets_summary.json", summary)
    print("== all done", flush=True)


if __name__ == "__main__":
    main()
