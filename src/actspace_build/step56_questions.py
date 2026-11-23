"""Step 5 (object clusters and initial views) + Step 6 (template questions, gold answers, constraints, evidence labels).

Step 6 balances quadrants and answer letters across scenes, so this step runs all scenes at once, sorted by scene.
Usage: python -m actspace_build.step56_questions --dataset arkitscenes --scenes ...
Outputs: work/<dataset>/<vid>/groups.json; work/<dataset>/_step6/test_groups.jsonl, step6_stats.json
Other datasets (ScanNet++) call run() (scannetpp_step56); their scene IDs are "<dataset>_<scene_id>" and objects are read from that dataset's directory. ARKit behaviour is unchanged.
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter, deque

import numpy as np

from . import geometry as g
from .common import DropLog, load_config, path_of, read_json, scene_rng, work_dir, write_json
from .step4_graph_visibility import ACTIONS, STATE_INVISIBLE, STATE_VISIBLE

TEMPLATES = {
    "T1": "What is the distance between the center of the {OBJ1} and the center of the {OBJ2}, in meters?",
    "T2_closest": "Which of the following objects is closest to the {OBJ1}?",
    "T2_farthest": "Which of the following objects is farthest from the {OBJ1}?",
    "T2_pair": "Which is closer to the {OBJ1}: the {OBJ2} or the {OBJ3}?",
    "T3": "Imagine you are standing at the {OBJ1} and facing the {OBJ2}. Where is the {OBJ3} relative to you?",
    "T4": "Relative to your starting position and viewing direction, where is the {OBJ1}?",
    "T5": "Imagine you are standing at the {OBJ1} and facing the {OBJ2}. Among the objects on your left, which one is closest to you?",
}
ABILITY = {"T1": "absolute_distance", "T2": "relative_distance", "T3": "perspective_taking",
           "T4": "unseen_object_localization", "T5": "multi_hop"}
LETTERS = "ABCD"


# ---------------------------------------------------------------- scene context

def scene_id_of(dataset: str, vid: str) -> str:
    """ARKit keeps "arkit_<video_id>"; other datasets use "<dataset>_<scene_id>"."""
    return f"arkit_{vid}" if dataset == "arkitscenes" else f"{dataset}_{vid}"


def objects_dataset(dataset: str) -> str:
    """Dataset directory holding objects.json: ARKit always reads arkitscenes (original behaviour); other datasets read their own directory."""
    return "arkitscenes" if dataset == "arkitscenes" else dataset


def covisible(state, rows, mode="visible_visible") -> bool:
    """Whether these objects (rows of the visibility matrix) are "in the same frame" in at least one node. mode as in link_edges:
    visible_visible = all visible (current); visible_partial = all at least partial and at least one visible; partial_partial = all at least partial."""
    S = state[rows]
    seen = (S != STATE_INVISIBLE).all(0)
    if mode == "visible_visible":
        return bool((S == STATE_VISIBLE).all(0).any())
    if mode == "visible_partial":
        return bool((seen & (S == STATE_VISIBLE).any(0)).any())
    return bool(seen.any())


LINKED_TEMPLATES = ("T1", "T2", "T3", "T5")


LINK_MODES = ("visible_visible", "visible_partial", "partial_partial")


def link_edges(state, rows, keep=None, mode="visible_partial"):
    """Whether each pair of evidence objects has a "same frame" edge at each node: E[i, j, v].
    mode: visible_visible = both visible; visible_partial = one visible and the other at least partial (used);
    partial_partial = both at least partial. If keep (boolean array) is given, only kept nodes count (masked question groups); other nodes have no edges."""
    assert mode in LINK_MODES, mode
    vis = state[rows] == STATE_VISIBLE
    seen = state[rows] != STATE_INVISIBLE
    if mode == "visible_visible":
        E = vis[:, None, :] & vis[None, :, :]
    elif mode == "visible_partial":
        E = (vis[:, None, :] & seen[None, :, :]) | (seen[:, None, :] & vis[None, :, :])
    else:
        E = seen[:, None, :] & seen[None, :, :]
    if keep is not None:
        E = E & np.asarray(keep, bool)[None, None, :]
    return E


def evidence_linked(state, rows, keep=None, mode="visible_partial") -> bool:
    """Evidence is linkable: build a graph with evidence objects as vertices and "in the same frame at some node" (see mode in link_edges) as edges; the graph must be connected."""
    adj = link_edges(state, rows, keep, mode).any(2)
    n = len(rows)
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if j not in seen and adj[i, j]:
                seen.add(j)
                stack.append(j)
    return len(seen) == n


def greedy_link_views(state, rows, keep=None, mode="visible_partial", prefer=()):
    """Link views: the co-visibility nodes needed to make the evidence graph connected, chosen greedily to be as few as possible.
    Each round picks the node that merges the most connected components; ties go to nodes in prefer (the question's oracle views and initial view), then to the smallest index.
    Returns a list of node indices; empty if there are fewer than 2 evidence objects; None if the graph cannot be connected."""
    n = len(rows)
    if n < 2:
        return []
    E = link_edges(state, rows, keep, mode)
    E[np.arange(n), np.arange(n), :] = False
    cand = np.flatnonzero(E.any((0, 1)))
    parent = list(range(n))

    def find(p, i):
        while p[i] != i:
            i = p[i]
        return i

    def merged(p, v):
        p, k = list(p), 0
        for i, j in zip(*np.nonzero(np.triu(E[:, :, v], 1))):
            a, b = find(p, i), find(p, j)
            if a != b:
                p[a] = b
                k += 1
        return k, p
    prefer, chosen = set(prefer), []
    while len({find(parent, i) for i in range(n)}) > 1:
        best = max(cand, key=lambda v: (merged(parent, v)[0], v in prefer, -v), default=None)
        if best is None or merged(parent, best)[0] == 0:
            return None
        parent = merged(parent, best)[1]
        chosen.append(int(best))
    return chosen


def path_view_indices(n, k=4):
    """Pick at most k of n waypoint nodes, evenly spaced and in order (including the last one)."""
    if n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def add_link_views(sc, groups):
    """Write link_views (list of node IDs; masked question groups use kept nodes only) for every question that needs linking; other questions get an empty list.
    Answerable T4 questions also get path_views: along the shortest path of oracle_steps, at most 4 waypoint nodes after the start, evenly spaced and in order;
    masked question groups compute the path on kept nodes, so masked nodes never appear. The definitions of oracle_views / oracle_steps are unchanged."""
    qc = sc.cfg["questions"]
    for grp in groups:
        keep = None
        if grp.get("view_graph_mask"):
            keep = np.ones(len(sc.nodes), bool)
            keep[[sc.node_index[n] for n in grp["view_graph_mask"]["masked_nodes"]]] = False
        p0 = sc.node_index[grp["init_frame"]]
        for q in grp["questions"]:
            q["link_views"], q["path_views"] = [], []
            if q["template"] == "T4" and q["evidence_label"] != "unanswerable":
                _, _, _, path = oracle(sc, q["evidence_objs"], p0, keep)
                walk = [v for v in dict.fromkeys(path) if v != p0]
                q["path_views"] = [sc.nodes[walk[i]] for i in path_view_indices(len(walk))]
            if not (qc["require_evidence_linked"] and q["template"] in LINKED_TEMPLATES and q["evidence_label"] != "unanswerable"):
                continue
            prefer = [sc.node_index[n] for n in q["oracle_nodes"]] + [p0]
            lv = greedy_link_views(sc.state, [sc.row[o] for o in dict.fromkeys(q["evidence_objs"])], keep, qc["evidence_link_mode"], prefer)
            assert lv is not None, f"{sc.scene_id} {grp['init_frame']} {q['template']}: evidence not linkable"
            q["link_views"] = [sc.nodes[v] for v in lv]


class Scene:
    def __init__(self, cfg, dataset, vid):
        self.cfg, self.vid = cfg, vid
        self.scene_id = scene_id_of(dataset, vid)
        wd = work_dir(cfg, dataset, vid)
        self.wd = wd
        self.objs = read_json(work_dir(cfg, objects_dataset(dataset), vid) / "objects.json")
        vis = np.load(wd / "visibility.npz")
        self.state = vis["state"]
        self.nodes = [str(x) for x in vis["frame_ids"]]
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        graph = read_json(wd / "view_graph.json")
        self.adj = [[self.node_index[graph["edges"][n][a]] for a in ACTIONS if graph["edges"][n][a] is not None] for n in self.nodes]
        self.radj = [[] for _ in self.nodes]
        for v, ws in enumerate(self.adj):
            for w in ws:
                self.radj[w].append(v)
        frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
        self.frames = [frames[n] for n in self.nodes]
        self.center = {o["obj_id"]: np.asarray(o["center"]) for o in self.objs}
        # Referring name in the question text: "TV" or "largest cabinet" ("largest/smallest X" references)
        self.cat = {o["obj_id"]: o.get("ref_name") or o["category"] for o in self.objs}
        self.size_group = {o["obj_id"]: o.get("size_group") or [o["obj_id"]] for o in self.objs}
        self.row = {o["obj_id"]: k for k, o in enumerate(self.objs)}
        self.referable = [o["obj_id"] for o in self.objs if o["referable"]]
        # larger / smaller references require both instances to be visible together in at least one node; otherwise the category falls back to "referable only if unique"
        # Size references are permanently disabled (size_ref_enabled: false); size-referenced objects in older Step 3 outputs are never referable
        self.size_ref_dropped, self.size_ref_disabled = [], []
        size_ref_on = cfg["objects"].get("size_ref_enabled", False)
        for o in self.objs:
            if o["referable"] and o.get("ref_kind") not in ("unique", None):
                if not size_ref_on:
                    self.size_ref_disabled.append(o["obj_id"])
                elif cfg["objects"]["size_ref_require_covisible"] and o.get("ref_kind") in ("larger", "smaller") and not covisible(
                        self.state, [self.row[m] for m in o["size_group"]], cfg["objects"].get("size_ref_covisible_mode", "visible_visible")):
                    self.size_ref_dropped.append(o["obj_id"])
        self.referable = [o for o in self.referable if o not in self.size_ref_dropped + self.size_ref_disabled]

    def dist(self, a, b) -> float:
        return float(np.linalg.norm(self.center[a] - self.center[b]))

    def st(self, obj, node) -> int:
        return int(self.state[self.row[obj], node])

    def visible_anywhere(self, obj) -> bool:
        return bool((self.state[self.row[obj]] == STATE_VISIBLE).any())


# ---------------------------------------------------------------- Step 5

def sample_groups(sc: Scene, drops: DropLog):
    gc = sc.cfg["groups"]
    rng = scene_rng(sc.cfg, sc.scene_id, "clusters")
    # Object clusters are drawn only from referable objects that are visible in at least one node
    ref = [o for o in sc.referable if sc.visible_anywhere(o)] if gc["cluster_require_visible_anywhere"] else sc.referable
    lo, hi = gc["objects_per_cluster"]
    accepted, seen = [], set()
    vis_any_node = sc.state == STATE_VISIBLE
    n_visible_per_node = vis_any_node.sum(0)
    n_seen_per_node = (sc.state != STATE_INVISIBLE).sum(0)
    # Scenes with few referable objects first sample 3-object clusters only; after half the attempts, sample sizes 3–5 uniformly
    small = len(ref) <= gc["small_scene_max_referable"]
    n_attempts = gc["cluster_sample_attempts"]
    for attempt in range(n_attempts):
        if len(accepted) >= gc["max_groups_per_scene"] or len(ref) < lo:
            break
        if small and attempt < n_attempts // 2:
            k = gc["small_scene_cluster_size"]
        else:
            k = int(rng.integers(lo, min(hi, len(ref)) + 1))
        cl = tuple(sorted(rng.choice(ref, size=k, replace=False).tolist()))
        if cl in seen:
            continue
        seen.add(cl)
        if any(sc.dist(a, b) < gc["min_pair_distance_m"] for a, b in itertools.combinations(cl, 2)):
            continue
        rows = [sc.row[o] for o in cl]
        if vis_any_node[rows].all(0).any():  # all objects visible together in one frame
            continue
        if any(len(set(cl) & set(a["cluster"])) > gc["max_object_overlap_between_groups"] for a in accepted):
            continue
        inv_frac = (sc.state[rows] == STATE_INVISIBLE).mean(0)
        if gc["init_view_relaxed"]:
            # Relaxed rule: ≥ 1 annotated object visible, or ≥ 2 visible / partial
            ok_view = (n_visible_per_node >= gc["init_view_relaxed_min_visible"]) | (n_seen_per_node >= gc["init_view_relaxed_min_seen"])
        else:
            ok_view = n_visible_per_node >= gc["init_view_min_visible_objects"]
        cand = np.where(ok_view & (inv_frac >= gc["init_view_min_invisible_frac"]))[0]
        if len(cand) == 0:
            drops.add("cluster", list(cl), "NO_VALID_INIT_VIEW")
            continue
        p0 = int(rng.choice(cand))
        accepted.append({"cluster": list(cl), "p0": p0, "num_p0_candidates": int(len(cand))})
    return accepted


# ---------------------------------------------------------------- evidence and oracle

def evidence_label(sc: Scene, ev, p0, never_visible_is_ambiguous=True):
    """Evidence label (see docs/data_construction.md). In ARKit, evidence objects that are invisible in every view mostly come from mesh artefacts;
    they are dropped as AMBIGUOUS_EVIDENCE (returns None), and unanswerable questions are generated only by masking."""
    if all(sc.st(o, p0) == STATE_VISIBLE for o in ev):
        return "init"
    if any((sc.state[sc.row[o]] == STATE_INVISIBLE).all() for o in ev):
        return None if never_visible_is_ambiguous else "unanswerable"
    if all(sc.visible_anywhere(o) for o in ev):
        return "explore"
    return None  # AMBIGUOUS_EVIDENCE


def oracle(sc: Scene, ev, p0, keep=None):
    """Returns (oracle_views, oracle_steps, list of observed nodes, path). Exact: a bitmask DP finds the minimum number of observations,
    then a 0-1 BFS over (node, covered mask, number observed) finds the minimum number of actions; equivalent to enumerating all minimum covers and visiting orders.
    If keep (boolean array) is given, computes only on the view graph of kept nodes (question groups containing a masked unanswerable question)."""
    need = [o for o in ev if sc.st(o, p0) != STATE_VISIBLE]
    if not need:
        return 0, 0, [], [p0]
    full = (1 << len(need)) - 1
    cover = np.zeros(len(sc.nodes), dtype=np.int64)
    for b, o in enumerate(need):
        cover |= (sc.state[sc.row[o]] == STATE_VISIBLE).astype(np.int64) << b
    if keep is not None:
        cover[~keep] = 0
    masks = set(int(m) for m in cover if m)
    reach, k = {0}, 0
    while full not in reach:
        k += 1
        reach = reach | {r | m for r in reach for m in masks}
        if k > len(need):
            return None, None, [], []
    K = k
    start = (p0, 0, 0)
    dist = {start: 0}
    parent = {start: None}
    dq = deque([start])
    goal = None
    while dq:
        s = dq.popleft()
        v, m, used = s
        if m == full:
            goal = s
            break
        steps = [((v, m | int(cover[v]), used + 1), 0)] if used < K and int(cover[v]) & ~m else []
        steps += [((w, m, used), 1) for w in sc.adj[v] if keep is None or keep[w]]
        for ns, c in steps:
            nd = dist[s] + c
            if nd < dist.get(ns, 1 << 30):
                dist[ns] = nd
                parent[ns] = s
                (dq.appendleft if c == 0 else dq.append)(ns)
    if goal is None:
        return K, None, [], []
    path, observed, s = [], [], goal
    while s is not None:
        p = parent[s]
        if p is not None and p[0] == s[0] and p[2] + 1 == s[2]:
            observed.append(s[0])
        elif p is None or p[0] != s[0]:
            path.append(s[0])
        s = p
    return K, dist[goal], observed[::-1], path[::-1]


# ---------------------------------------------------------------- within-scene deduplication

def dedup_key(q, t2_by_answer=False):
    """Question content key: T1 unordered object pair; T2 reference + comparison direction + set of option objects; T3 (stand, face, queried); T5 (stand, face); T4 queried object.
    With t2_by_answer=True, T2 uses reference + comparison direction + correct answer instead (distractors ignored; used for ScanNet++, ARKit unchanged)."""
    s, t = q["slots"], q["template"]
    if t == "T1":
        return ("T1", frozenset((s["OBJ1"], s["OBJ2"])))
    if t == "T2":
        if t2_by_answer:
            return ("T2", s["OBJ1"], q.get("variant"), "answer", q["answer_obj"])
        return ("T2", s["OBJ1"], q.get("variant"), frozenset(q["option_objs"]))
    if t == "T3":
        return ("T3", s["OBJ1"], s["OBJ2"], s["OBJ3"])
    if t == "T5":
        return ("T5", s["OBJ1"], s["OBJ2"])
    return ("T4", s["OBJ1"])


class SceneUsed:
    """Scene-level set of "used questions": T1/T2/T3/T5 content appears only once; the same T4 object appears at most max_t4 times (different starts)."""

    def __init__(self, max_t4, t2_by_answer=False):
        self.count = Counter()
        self.max_t4 = max_t4
        self.t2_by_answer = t2_by_answer

    def key(self, q):
        return dedup_key(q, self.t2_by_answer)

    def limit(self, key):
        return self.max_t4 if key[0] == "T4" else 1

    def ok(self, q):
        k = self.key(q)
        return self.count[k] < self.limit(k)

    def ok_all(self, qs):
        c = Counter(self.key(q) for q in qs)
        return all(self.count[k] + n <= self.limit(k) for k, n in c.items())

    def add(self, q):
        self.count[self.key(q)] += 1

    def remove(self, q):
        self.count[self.key(q)] -= 1

    def copy(self):
        u = SceneUsed(self.max_t4, self.t2_by_answer)
        u.count = self.count.copy()
        return u


# ---------------------------------------------------------------- ambiguity filters and question generation

# Candidate filter statistics (template:reason → count). Candidates enumerate object combinations within a cluster, and the same combination may be enumerated once for constraints and once for supplementary questions,
# so these are "filter trigger counts", not numbers of dropped questions; for the latter see dropped.jsonl
CANDIDATE_REJECTS = Counter()


def reject(template, reason):
    CANDIDATE_REJECTS[f"{template}:{reason}"] += 1
    return None


def rank_margin_ok(d_first, d_second, qc) -> bool:
    diff = abs(d_second - d_first)
    return diff >= qc["rank_margin_abs_m"] and diff / max(d_first, d_second) >= qc["rank_margin_rel"]


def q_T1(sc, a, b, qc):
    d = sc.dist(a, b)
    lo, hi = qc["t1_distance_range_m"]
    if not lo <= d <= hi:
        return reject("T1", "DISTANCE_OUT_OF_RANGE")
    return {"template": "T1", "text_template": TEMPLATES["T1"], "slots": {"OBJ1": a, "OBJ2": b},
            "answer_type": "numeric", "answer_raw": round(d, qc["store_round_decimals"]),
            "answer": round(round(d / qc["answer_round_m"]) * qc["answer_round_m"], 1),
            "evidence_objs": [a, b], "margins": {"distance_m": round(d, 2)}}


def q_pair(sc, a, b, c, qc):
    db, dc = sc.dist(a, b), sc.dist(a, c)
    if not rank_margin_ok(min(db, dc), max(db, dc), qc):
        return reject("T2_pair", "MARGIN_FAIL_rank")
    return {"template": "T2", "variant": "pair", "text_template": TEMPLATES["T2_pair"], "slots": {"OBJ1": a, "OBJ2": b, "OBJ3": c},
            "answer_type": "multiple_choice", "option_objs": [b, c], "answer_obj": b if db < dc else c,
            "evidence_objs": [a, b, c], "margins": {"dist_diff_m": round(abs(db - dc), 2)}}


def q_T2(sc, a, opts, farthest, qc):
    d = sorted(((sc.dist(a, o), o) for o in opts), reverse=farthest)
    if not rank_margin_ok(d[0][0], d[1][0], qc):
        return reject("T2", "MARGIN_FAIL_rank")
    key = "T2_farthest" if farthest else "T2_closest"
    return {"template": "T2", "variant": "farthest" if farthest else "closest", "text_template": TEMPLATES[key],
            "slots": {"OBJ1": a}, "answer_type": "multiple_choice", "option_objs": list(opts), "answer_obj": d[0][1],
            "evidence_objs": [a, *opts], "margins": {"dist_diff_m": round(abs(d[1][0] - d[0][0]), 2)}}


def q_T3(sc, a, b, c, qc):
    if min(sc.dist(a, b), sc.dist(a, c)) < qc["min_standpoint_distance_m"]:
        return reject("T3", "STANDPOINT_TOO_CLOSE")
    th = g.direction_theta_facing(sc.center[a], sc.center[b], sc.center[c])
    if g.angle_margin(th) < qc["angle_margin_deg"]:
        return reject("T3", "MARGIN_FAIL_angle")
    return {"template": "T3", "text_template": TEMPLATES["T3"], "slots": {"OBJ1": a, "OBJ2": b, "OBJ3": c},
            "answer_type": "multiple_choice", "option_values": list(g.QUADRANTS), "answer_value": g.quadrant(th),
            "evidence_objs": [a, b, c], "margins": {"theta_deg": round(th, 1), "angle_margin_deg": round(g.angle_margin(th), 1)}}


def q_T4(sc, x, p0, qc):
    if sc.st(x, p0) != STATE_INVISIBLE:
        return reject("T4", "NOT_INVISIBLE_AT_P0")
    f = sc.frames[p0]
    th = g.theta_from_yaw(f["position"], f["yaw_deg"], sc.center[x])
    if g.angle_margin(th) < qc["angle_margin_deg"]:
        return reject("T4", "MARGIN_FAIL_angle")
    return {"template": "T4", "text_template": TEMPLATES["T4"], "slots": {"OBJ1": x},
            "answer_type": "multiple_choice", "option_values": list(g.QUADRANTS), "answer_value": g.quadrant(th),
            "evidence_objs": [x], "margins": {"theta_deg": round(th, 1), "angle_margin_deg": round(g.angle_margin(th), 1)}}


def q_T5(sc, a, b, opts, qc):
    if sc.dist(a, b) < qc["min_standpoint_distance_m"] or any(sc.dist(a, o) < qc["min_standpoint_distance_m"] for o in opts):
        return reject("T5", "STANDPOINT_TOO_CLOSE")
    th = {o: g.direction_theta_facing(sc.center[a], sc.center[b], sc.center[o]) for o in opts}
    m = qc["angle_margin_deg"]
    if any(min(abs(t), 180.0 - abs(t)) < m for t in th.values()):  # too close to 0° / 180°
        return reject("T5", "MARGIN_FAIL_angle")
    left = sorted((sc.dist(a, o), o) for o in opts if th[o] > 0)
    if not left or len(left) == len(opts):  # need a mix of left and right
        return reject("T5", "NO_LEFT_RIGHT_MIX")
    if len(left) >= 2 and not rank_margin_ok(left[0][0], left[1][0], qc):
        return reject("T5", "MARGIN_FAIL_rank")
    return {"template": "T5", "text_template": TEMPLATES["T5"], "slots": {"OBJ1": a, "OBJ2": b},
            "answer_type": "multiple_choice", "option_objs": list(opts), "answer_obj": left[0][1],
            "evidence_objs": [a, b, *opts], "margins": {"theta_deg": {str(o): round(t, 1) for o, t in th.items()}}}


class Balancer:
    """Cross-scene template counts and T3/T4 quadrant counts, used for resampling during sampling."""

    def __init__(self):
        self.template = Counter()
        self.variant = Counter()
        self.constraint = Counter()
        self.quadrant = {"T3": Counter(), "T4": Counter()}
        self.n_questions = 0
        self.n_pairs = 0
        self.n_unanswerable = 0

    def pick(self, rng, cands):
        """T3/T4 candidates: least-counted quadrant first; T2: least-counted variant first (pair / closest / farthest); then random."""
        if not cands:
            return None
        t = cands[0]["template"]
        if t in self.quadrant:
            qs = sorted({c["answer_value"] for c in cands}, key=lambda q: (self.quadrant[t][q], g.QUADRANTS.index(q)))
            cands = [c for c in cands if c["answer_value"] == qs[0]]
        variants = sorted({c.get("variant") or "" for c in cands}, key=lambda v: (self.variant[(t, v)], v))
        cands = [c for c in cands if (c.get("variant") or "") == variants[0]]
        return cands[int(rng.integers(len(cands)))]

    def pick_constraint(self, rng, cons, qc, gcfg, dt_used_in_scene):
        """Binary comparison questions (T2:pair) make up ≤ pair_max_frac of all questions; constraints prefer numeric_vs_comparison and
        direction_inverse; distance_transitivity is used for at most max_distance_transitivity_per_scene groups per scene, and only when neither of the other two is feasible.
        The comparison share is estimated assuming this group is filled to the questions_per_group maximum; constraints that would exceed the cap are not chosen. Returns None if no constraint is available."""
        hi = gcfg["questions_per_group"][1]
        n_pairs = {"numeric_vs_comparison": 1, "direction_inverse": 0, "distance_transitivity": 3}

        def within_cap(c):
            return (self.n_pairs + n_pairs[c[0]]) / (self.n_questions + hi) <= qc["pair_max_frac"]
        ok = [c for c in cons if within_cap(c)]
        main = [c for c in ok if c[0] in ("numeric_vs_comparison", "direction_inverse")]
        if main:
            return sorted(main, key=lambda c: (self.constraint[c[0]], c[0]))[0]
        dt = [c for c in ok if c[0] == "distance_transitivity" and dt_used_in_scene < qc["max_distance_transitivity_per_scene"]]
        return dt[0] if dt else None

    def remove(self, q):
        """Undo add (used when a masked unanswerable question replaces a supplementary question)."""
        self.n_questions -= 1
        self.n_unanswerable -= q.get("evidence_label") == "unanswerable"
        self.n_pairs -= q.get("variant") == "pair"
        self.template["T2_pair" if q.get("variant") == "pair" else q["template"]] -= 1
        self.variant[(q["template"], q.get("variant") or "")] -= 1
        if q["template"] in self.quadrant:
            self.quadrant[q["template"]][q["answer_value"]] -= 1

    def add(self, q):
        self.n_questions += 1
        self.n_unanswerable += q.get("evidence_label") == "unanswerable"
        self.n_pairs += q.get("variant") == "pair"
        # Binary comparison questions inside constraints are counted separately and do not use up the T2 (≥ 3 options) quota of supplementary questions
        self.template["T2_pair" if q.get("variant") == "pair" else q["template"]] += 1
        self.variant[(q["template"], q.get("variant") or "")] += 1
        if q["template"] in self.quadrant:
            self.quadrant[q["template"]][q["answer_value"]] += 1


def expand_evidence(sc, ev):
    """An object referred to as "largest/smallest X" is compared with all instances of its category, so all instances count as evidence objects."""
    out = []
    for o in ev:
        for m in [o] + [m for m in sc.size_group[o] if m != o]:
            if m not in out:
                out.append(m)
    return out


def annotate(sc, q, p0, drops, group_tag):
    """Add the evidence label and oracle to a question; AMBIGUOUS returns None and is logged."""
    q["evidence_objs"] = expand_evidence(sc, q["evidence_objs"])
    if len(q["evidence_objs"]) > sc.cfg["questions"]["max_evidence_objects"]:
        drops.add("question", f"{group_tag}:{q['template']}:{q['slots']}", "EVIDENCE_TOO_LARGE", n_evidence=len(q["evidence_objs"]))
        return None
    lab = evidence_label(sc, q["evidence_objs"], p0, sc.cfg["questions"]["never_visible_is_ambiguous"])
    if lab is None:
        drops.add("question", f"{group_tag}:{q['template']}:{q['slots']}", "AMBIGUOUS_EVIDENCE")
        return None
    if (lab != "unanswerable" and q["template"] in LINKED_TEMPLATES and sc.cfg["questions"]["require_evidence_linked"]
            and not evidence_linked(sc.state, [sc.row[o] for o in q["evidence_objs"]], mode=sc.cfg["questions"]["evidence_link_mode"])):
        drops.add("question", f"{group_tag}:{q['template']}:{q['slots']}", "EVIDENCE_NOT_LINKED", n_evidence=len(q["evidence_objs"]))
        return None
    q["evidence_label"] = lab
    if lab == "unanswerable":
        q.update(oracle_views=None, oracle_steps=None, oracle_nodes=[])
    else:
        k, s, observed, _ = oracle(sc, q["evidence_objs"], p0)
        q.update(oracle_views=k, oracle_steps=s, oracle_nodes=[sc.nodes[v] for v in observed])
    return q


def constraint_candidates(sc, cl, p0, qc, rng, bal, annot, used):
    """Return feasible constraint candidates: (type, [questions], expected). Each question passes the ambiguity filters on its own."""
    out = []
    perms3 = list(itertools.permutations(cl, 3))
    rng.shuffle(perms3)
    # numeric_vs_comparison
    for a, b, c in perms3:
        if b > c:
            continue
        qs = [q_T1(sc, a, b, qc), q_T1(sc, a, c, qc), q_pair(sc, a, b, c, qc)]
        qs = [annot(q) for q in qs] if all(qs) and used.ok_all(qs) else [None]
        if all(qs):
            out.append(("numeric_vs_comparison", qs, "q3 answer = argmin(q1, q2)"))
            break
    # direction_inverse (only triples where C projects outside segment AB)
    cands = []
    for a, b, c in perms3:
        if a > b or not g.projection_outside_segment(sc.center[a], sc.center[b], sc.center[c]):
            continue
        q1, q2 = q_T3(sc, a, b, c, qc), q_T3(sc, b, a, c, qc)
        if q1 and q2 and g.INVERSE_QUADRANT[q1["answer_value"]] == q2["answer_value"] and used.ok_all([q1, q2]):
            cands.append((q1, q2))
    if cands:
        qmin = sorted({c[0]["answer_value"] for c in cands}, key=lambda q: (bal.quadrant["T3"][q], g.QUADRANTS.index(q)))[0]
        cands = [c for c in cands if c[0]["answer_value"] == qmin]
        q1, q2 = (annot(q) for q in cands[int(rng.integers(len(cands)))])
        if q1 and q2:
            out.append(("direction_inverse", [q1, q2], "q2 quadrant = inverse(q1 quadrant)"))
    # distance_transitivity
    if len(cl) >= 4:
        quads = [(a, *sorted(r)) for a in cl for r in itertools.combinations([o for o in cl if o != a], 3)]
        rng.shuffle(quads)
        for a, x, y, z in quads:
            qs = [q_pair(sc, a, x, y, qc), q_pair(sc, a, y, z, qc), q_pair(sc, a, x, z, qc)]
            qs = [annot(q) for q in qs] if all(qs) and used.ok_all(qs) else [None]
            if all(qs):
                out.append(("distance_transitivity", qs, "pairwise answers form no cycle"))
                break
    return out


def fill_candidates(sc, cl, p0, qc, rng):
    """Candidates for supplementary questions in a group (evidence not yet labelled). Reference/standing objects come from the cluster; option objects of T2 and T5 are drawn from all referable objects in the scene
    and also count as evidence_objs. Binary comparison questions appear only in constraints; supplementary T2 questions have at least 3 options."""
    c = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
    for a, b in itertools.combinations(cl, 2):
        c["T1"].append(q_T1(sc, a, b, qc))
    lo2, hi2 = qc["t2_options"]
    lo5, hi5 = qc["t5_options"]
    for a in cl:
        rest = [o for o in cl if o != a]
        scene_rest = [o for o in sc.referable if o != a]
        for n in range(lo2, min(hi2, len(scene_rest)) + 1):
            for opts in itertools.combinations(scene_rest, n):
                c["T2"].append(q_T2(sc, a, opts, bool(rng.random() < qc["t2_farthest_frac"]), qc))
        for b, x in itertools.permutations(rest, 2):
            c["T3"].append(q_T3(sc, a, b, x, qc))
        for b in rest:
            scene_rest2 = [o for o in sc.referable if o not in (a, b)]
            for n in range(lo5, min(hi5, len(scene_rest2)) + 1):
                for opts in itertools.combinations(scene_rest2, n):
                    c["T5"].append(q_T5(sc, a, b, opts, qc))
        c["T4"].append(q_T4(sc, a, p0, qc))
    return {t: [q for q in v if q] for t, v in c.items()}


def build_group(sc, grp, gi, qc, gcfg, rng, bal, drops, scene_used, dt_used_in_scene=0):
    cl, p0 = grp["cluster"], grp["p0"]
    tag = f"{sc.scene_id}:g{gi}"
    cache = {}

    def annot(q):
        key = (q["template"], q.get("variant"), tuple(sorted(q["slots"].items())), tuple(q.get("option_objs", [])))
        if key not in cache:
            cache[key] = annotate(sc, q, p0, drops, tag)
        return cache[key]

    cons = constraint_candidates(sc, cl, p0, qc, rng, bal, annot, scene_used)
    picked = bal.pick_constraint(rng, cons, qc, gcfg, dt_used_in_scene) if cons else None
    if picked is None and gcfg.get("require_constraint", True):
        if not cons:
            drops.add("group", tag, "GROUP_INCOMPLETE", detail="no constraint passes ambiguity/evidence/dedup filters", cluster=cl)
        else:
            drops.add("group", tag, "GROUP_INCOMPLETE", detail="no constraint within T2:pair cap / distance_transitivity limit", cluster=cl,
                      feasible=[c[0] for c in cons])
        return None
    # With groups.require_constraint: false (read-only what-if analysis), groups without an available constraint have no constraint and contain only regular questions
    ctype, cqs, expected = picked if picked is not None else (None, [], None)
    lo, hi = gcfg["questions_per_group"]
    # Groups containing a binary comparison question are filled to the maximum, lowering the comparison share
    has_pair = any(q.get("variant") == "pair" for q in cqs)
    target = hi if has_pair else int(rng.integers(max(lo, len(cqs)), hi + 1))
    questions = list(cqs)
    if sum(q["evidence_label"] == "init" for q in questions) > qc["max_init_per_group"]:
        drops.add("group", tag, "AMBIGUOUS_EVIDENCE", detail="constraint has more than max_init_per_group init questions")
        return None
    pool = fill_candidates(sc, cl, p0, qc, rng)
    used = {(q["template"], tuple(sorted(q["slots"].items())), tuple(q.get("option_objs", []))) for q in questions}
    group_used = scene_used.copy()  # within-scene dedup: record this group's questions on a copy and merge into the scene set only if the whole group succeeds
    for q in questions:
        group_used.add(q)
    while len(questions) < target:
        types_now = {q["template"] for q in questions}
        n_init = sum(q["evidence_label"] == "init" for q in questions)
        avail = []
        for t, qs in pool.items():
            ok = []
            for q in qs:
                k = (q["template"], tuple(sorted(q["slots"].items())), tuple(q.get("option_objs", [])))
                if k in used or not group_used.ok(q):
                    continue
                aq = annot(q)
                if aq is None or (aq["evidence_label"] == "init" and n_init >= qc["max_init_per_group"]):
                    continue
                ok.append(aq)
            if ok:
                avail.append((t, ok))
        if not avail:
            break
        # First ensure at least two question types, then prefer the least-counted global template to balance the type mix
        avail.sort(key=lambda x: (x[0] in types_now and len(types_now) < 2,
                                  bal.template[x[0]] + sum(q["template"] == x[0] and q.get("variant") != "pair" for q in questions), x[0]))
        q = bal.pick(rng, avail[0][1])
        questions.append(q)
        group_used.add(q)
        used.add((q["template"], tuple(sorted(q["slots"].items())), tuple(q.get("option_objs", []))))
    if len(questions) < lo or len({q["template"] for q in questions}) < 2:
        drops.add("group", tag, "GROUP_INCOMPLETE", detail="cannot reach group size / template diversity", n=len(questions))
        return None
    for q in questions:
        bal.add(q)
    if ctype is not None:
        bal.constraint[ctype] += 1
    scene_used.count = group_used.count
    return {"scene_id": sc.scene_id, "init_frame": sc.nodes[p0], "cluster_obj_ids": cl, "num_p0_candidates": grp["num_p0_candidates"],
            "questions": [dict(q) for q in questions], "constraint": {"type": ctype, "n": len(cqs), "expected": expected}}


# ---------------------------------------------------------------- pilot: init / unanswerable pipeline check

def scc_of(sc, keep: np.ndarray, start: int) -> set:
    """Nodes of the strongly connected component containing start when only the nodes in keep are kept."""
    def reach(adj):
        seen, stack = {start}, [start]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if keep[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen
    return reach(sc.adj) & reach(sc.radj)


def masked_unanswerable(sc, grp, x, qc, replace_idx=None, scene_used=None):
    """Masking: remove all nodes where x is visible / partial from the view graph; if the remaining graph is disconnected,
    also mask nodes not in the same strongly connected component as p0 (x is invisible at those nodes anyway). Requirements: p0 is not masked; the masked-node share
    ≤ unanswerable_max_masked_frac; other questions in the group remain answerable after masking (the whole group shares this masked view graph).
    If replace_idx is given, the new question replaces the non-constraint supplementary question at that position; the answerability check excludes the replaced question,
    and the group must still have at least 2 question types after replacement. Returns (question, keep, failure reason)."""
    p0 = sc.node_index[grp["init_frame"]]
    keep = sc.state[sc.row[x]] == STATE_INVISIBLE
    if not keep[p0]:
        return None, None, "p0_masked"
    n_masked_by_obj = int((~keep).sum())
    comp = scc_of(sc, keep, p0)
    keep = np.zeros_like(keep)
    keep[list(comp)] = True
    frac = 1.0 - keep.sum() / len(keep)
    if frac > qc["unanswerable_max_masked_frac"]:
        return None, None, "masked_frac_too_large"
    kept = np.where(keep)[0]
    others = [oq for k, oq in enumerate(grp["questions"]) if k != replace_idx]
    compatible = all(any(sc.st(o, v) == STATE_VISIBLE for v in kept) for oq in others
                     if oq["evidence_label"] != "unanswerable" for o in oq["evidence_objs"])
    if compatible and sc.cfg["questions"]["require_evidence_linked"]:
        # After masking, the evidence of other questions in the group must still be linkable (on kept nodes)
        compatible = all(evidence_linked(sc.state, [sc.row[o] for o in oq["evidence_objs"]], keep, sc.cfg["questions"]["evidence_link_mode"]) for oq in others
                         if oq["evidence_label"] != "unanswerable" and oq["template"] in LINKED_TEMPLATES)
    if not compatible:
        return None, None, "group_incompatible"
    cands = [q_T4(sc, x, p0, qc)] + [q_T1(sc, x, y, qc) for y in grp["cluster_obj_ids"] if y != x]
    cands = [q for q in cands if q]
    if not cands:
        return None, None, "margin_fail"
    if scene_used is not None:
        # Within-scene dedup: the replaced question releases its content key
        u = scene_used.copy()
        if replace_idx is not None:
            u.remove(grp["questions"][replace_idx])
        cands = [q for q in cands if u.ok(q)]
        if not cands:
            return None, None, "duplicate_in_scene"
    q = dict(cands[0])
    # The masked object and the other evidence objects must all have visible nodes in the full view graph (objects never seen cannot be used for questions)
    if not all(sc.visible_anywhere(o) for o in expand_evidence(sc, q["evidence_objs"])):
        return None, None, "never_visible_evidence"
    if replace_idx is not None and len({oq["template"] for oq in others} | {q["template"]}) < 2:
        return None, None, "template_diversity"
    q["evidence_objs"] = expand_evidence(sc, q["evidence_objs"])
    if len(q["evidence_objs"]) > qc["max_evidence_objects"]:
        return None, None, "evidence_too_large"
    masked = [sc.nodes[v] for v in np.where(~keep)[0]]
    q.update(evidence_label="unanswerable", oracle_views=None, oracle_steps=None, oracle_nodes=[], supplement="unanswerable",
             unanswerable_method="mask", masked_obj=x, masked_nodes=masked, masked_frac=round(float(frac), 3),
             masked_by_object=n_masked_by_obj, group_compatible=True)
    return q, keep, None


def add_masked_unanswerable(sc, groups, order, qc, room, rng, bal, drops, scene_used):
    """Try 1 masked unanswerable question per question group (up to the dataset-wide target_unanswerable_frac cap). Append if the group has room;
    if there is no room or appending is infeasible, a non-constraint supplementary question in the group may be replaced (questions in constraints cannot be replaced). Within-scene dedup applies.
    Selection order: append before replace; T4 candidates prefer the quadrant with the lowest current global count; then increasing masked-node share."""
    cap = qc["target_unanswerable_frac"][1]
    fails = Counter()
    fails_by_group = {i: Counter() for i in order}  # failure reasons per group (for reporting each group's attempts)
    cands = []
    for i in order:
        grp = groups[i]
        n_con = grp["constraint"]["n"]
        options = ([None] if room(grp) else []) + [k for k in range(n_con, len(grp["questions"]))
                                                    if not grp["questions"][k].get("supplement")]
        for x in [int(o) for o in rng.permutation(sc.referable)]:
            for rep_idx in options:
                q, keep, why = masked_unanswerable(sc, grp, x, qc, rep_idx, scene_used)
                if q is None:
                    fails[why] += 1
                    fails_by_group[i][why] += 1
                    if why in ("p0_masked", "masked_frac_too_large", "margin_fail", "evidence_too_large", "never_visible_evidence"):
                        break  # failure unrelated to which question is replaced
                    continue
                cands.append({"rep": rep_idx, "i": i, "x": x, "q": q, "keep": keep})
                break
    added, used_g, used_x = [], set(), set()

    def key(c):
        quad = bal.quadrant["T4"][c["q"]["answer_value"]] - min(bal.quadrant["T4"].get(k, 0) for k in g.QUADRANTS) \
            if c["q"]["template"] == "T4" else 0
        return (c["rep"] is not None, quad, c["q"]["masked_frac"], c["i"], c["x"])
    while cands:
        cands = [c for c in cands if c["i"] not in used_g and c["x"] not in used_x]
        if not cands:
            break
        c = min(cands, key=key)  # re-pick each time using current quadrant counts
        cands.remove(c)
        i, x, rep_idx, q, keep = c["i"], c["x"], c["rep"], c["q"], c["keep"]
        grp = groups[i]
        if rep_idx is None and not room(grp):
            continue
        if not scene_used.copy().ok(q) and rep_idx is None:
            fails["duplicate_in_scene"] += 1
            continue
        if (bal.n_unanswerable + 1) / (bal.n_questions + (rep_idx is None)) > cap:
            fails["dataset_cap_reached"] += 1
            fails_by_group[i]["dataset_cap_reached"] += 1
            continue
        replaced = None
        if rep_idx is not None:
            replaced = grp["questions"][rep_idx]
            u = scene_used.copy()
            u.remove(replaced)
            if not u.ok(q):
                fails["duplicate_in_scene"] += 1
                continue
            bal.remove(replaced)
            scene_used.remove(replaced)
            drops.add("question", f"{sc.scene_id}:{grp['init_frame']}:{replaced['template']}:{replaced['slots']}",
                      "REPLACED_BY_UNANSWERABLE", masked_obj=sc.cat[x])
            q["replaced_template"] = replaced["template"]
            grp["questions"][rep_idx] = q
        else:
            grp["questions"].append(q)
        # The whole group shares the masked view graph: recompute the oracle of other questions in the group on the masked graph
        p0 = sc.node_index[grp["init_frame"]]
        for oq in grp["questions"]:
            if oq["evidence_label"] != "unanswerable":
                k, st_, observed, _ = oracle(sc, oq["evidence_objs"], p0, keep)
                oq.update(oracle_views=k, oracle_steps=st_, oracle_nodes=[sc.nodes[v] for v in observed])
        grp["view_graph_mask"] = {"masked_obj": x, "masked_nodes": q["masked_nodes"]}
        bal.add(q)
        scene_used.add(q)
        used_g.add(i)
        used_x.add(x)
        added.append({"group_index": i, "template": q["template"], "object": sc.cat[x], "masked_nodes": len(q["masked_nodes"]),
                      "masked_frac": q["masked_frac"], "masked_by_object": q["masked_by_object"],
                      "replaced": None if replaced is None else replaced["template"] + (":" + replaced["variant"] if replaced.get("variant") else "")})
    r = {"target": len(groups), "added": added or None, "n_candidates": len(added), "fail_counts": dict(fails),
         "fail_counts_by_group": {str(i): dict(c) for i, c in fails_by_group.items()},
         "groups_with_candidate": sorted({int(c["i"]) for c in cands} | {a["group_index"] for a in added})}
    if not added:
        r["reason"] = "no candidate satisfies the conditions (failure counts in fail_counts)"
    return r


def add_special_questions(sc, groups, qc, gcfg, rng, bal, drops, scene_used):
    """Pilot stage: each scene tries at least 1 unanswerable and 1 init question to check that the pipeline works;
    visibility thresholds are not relaxed to meet counts. Questions are appended at the end of a group (constraint positions unaffected), and group size does not exceed the maximum. Returns the attempt records."""
    hi = gcfg["questions_per_group"][1]
    order = [int(i) for i in rng.permutation(len(groups))]
    report = {}

    def room(grp):
        return len(grp["questions"]) < hi

    def append(grp, q, kind):
        if not scene_used.ok(q):
            return False
        q = annotate(sc, dict(q), sc.node_index[grp["init_frame"]], drops, f"{sc.scene_id}:{kind}")
        if q is None or q["evidence_label"] != kind:
            return False
        q["supplement"] = kind
        grp["questions"].append(q)
        bal.add(q)
        scene_used.add(q)
        return True

    # unanswerable: masking
    report["unanswerable"] = add_masked_unanswerable(sc, groups, order, qc, room, rng, bal, drops, scene_used)

    # init: all evidence objects visible at p0; cluster objects are mostly invisible at p0, so use scene referable objects visible at p0
    r = {"added": None}
    tried, max_vis = 0, 0
    for i in order:
        grp = groups[i]
        if not room(grp) or any(q["evidence_label"] == "init" for q in grp["questions"]):
            continue
        p0 = sc.node_index[grp["init_frame"]]
        vis = [o for o in sc.referable if sc.st(o, p0) == STATE_VISIBLE]
        tried, max_vis = tried + 1, max(max_vis, len(vis))
        cands = [q_T1(sc, a, b, qc) for a, b in itertools.combinations(vis, 2)]
        cands += [q_T3(sc, a, b, c, qc) for a, b, c in itertools.permutations(vis, 3)]
        if any(q and append(grp, q, "init") for q in cands):
            q = grp["questions"][-1]
            r["added"] = {"group_index": i, "template": q["template"], "objects": [sc.cat[o] for o in q["evidence_objs"]]}
            break
    if not r["added"]:
        r["reason"] = ("all question groups are full or already have an init question" if tried == 0 else
                       f"tried {tried} question groups with free slots; at most {max_vis} referable objects visible together at p0"
                       + (" (< 2, no question possible)" if max_vis < 2 else "; the candidate questions failed the ambiguity filter"))
    report["init"] = r
    return report


def rebalance_quadrants(groups, gcfg, tol, drop_logs, rng):
    """T3/T4: all four quadrants must be within 25% ± tol. Otherwise remove non-constraint, non-supplementary questions from the most frequent quadrant
    (one at a time; the group must keep ≥ 3 questions and ≥ 2 question types) until balanced or nothing can be removed. Returns removal statistics."""
    lo = gcfg["questions_per_group"][0]
    removed = Counter()
    for t in ("T3", "T4"):
        while True:
            c = Counter(q["answer_value"] for g_ in groups for q in g_["questions"] if q["template"] == t)
            n = sum(c.values())
            if not n or all(abs(c[k] / n - 0.25) <= tol for k in g.QUADRANTS):
                break
            if min(c[k] for k in g.QUADRANTS) == 0:
                removed[f"{t}:infeasible_empty_quadrant"] += 1  # a quadrant is empty, removal cannot reach balance; remove nothing
                break
            over = max(g.QUADRANTS, key=lambda k: c[k])
            cands = []
            for gi, g_ in enumerate(groups):
                qs = g_["questions"]
                if len(qs) - 1 < lo:
                    continue
                for qi in range(g_["constraint"]["n"], len(qs)):
                    q = qs[qi]
                    if q["template"] == t and q["answer_value"] == over and not q.get("supplement"):
                        rest = {x["template"] for k, x in enumerate(qs) if k != qi}
                        if len(rest) >= 2:
                            cands.append((gi, qi))
            if not cands:
                break
            gi, qi = cands[int(rng.integers(len(cands)))]
            q = groups[gi]["questions"].pop(qi)
            removed[f"{t}:{over}"] += 1
            drop_logs[groups[gi]["scene_id"]].add("question", f"{groups[gi]['scene_id']}:{q['template']}:{q['slots']}",
                                                  "QUADRANT_REBALANCE", quadrant=over)
    return dict(removed)


def enforce_pair_cap(groups, cap, drop_logs):
    """Hard constraint: binary comparison questions make up ≤ cap of all questions. If still exceeded (some groups have fewer than 5 questions), drop whole groups in decreasing order of their comparison share and log them."""
    is_pair = lambda q: q.get("variant") == "pair"
    groups = list(groups)
    while True:
        n = sum(len(g_["questions"]) for g_ in groups)
        p = sum(is_pair(q) for g_ in groups for q in g_["questions"])
        if not n or p / n <= cap:
            return groups
        worst = max(range(len(groups)), key=lambda i: (sum(map(is_pair, groups[i]["questions"])) / len(groups[i]["questions"]), i))
        g_ = groups.pop(worst)
        drop_logs[g_["scene_id"]].add("group", f"{g_['scene_id']}:init={g_['init_frame']}", "GROUP_INCOMPLETE",
                                      detail="dropped to keep T2:pair fraction within cap", pair_frac_before=p / n)


# ---------------------------------------------------------------- global: IDs, answer letters, difficulty

def finalize(groups, cfg, objs_by_scene):
    rng = np.random.default_rng([cfg["seed"], 6])
    for gi, grp in enumerate(groups):
        grp["group_id"] = f"g_{gi:06d}"
        for qi, q in enumerate(grp["questions"]):
            q["qid"] = f"{grp['group_id']}_q{qi + 1}"
            q["ability"] = ABILITY[q["template"]]
        n = grp["constraint"].pop("n")
        grp["constraints"] = [] if grp["constraint"]["type"] is None else \
            [{"type": grp["constraint"]["type"], "qids": [q["qid"] for q in grp["questions"][:n]], "expected": grp["constraint"]["expected"]}]
        del grp["constraint"]
        grp["human_verified"] = False
    # Answer letters: group by (template, number of options), assign target positions exactly uniformly, then shuffle
    buckets = {}
    for grp in groups:
        for q in grp["questions"]:
            if q["answer_type"] == "multiple_choice":
                n = len(q.get("option_objs") or q.get("option_values"))
                buckets.setdefault((q["template"], n), []).append((grp, q))
    for (t, n), items in sorted(buckets.items()):
        pos = np.tile(np.arange(n), -(-len(items) // n))[:len(items)]
        rng.shuffle(pos)
        for (grp, q), p in zip(items, pos):
            cat = objs_by_scene[grp["scene_id"]]
            if "option_objs" in q:
                others = [o for o in q["option_objs"] if o != q["answer_obj"]]
                rng.shuffle(others)
                order = others[:p] + [q["answer_obj"]] + others[p:]
                q["option_objs"] = order
                q["options"] = [f"{LETTERS[i]}. {cat[o]}" for i, o in enumerate(order)]
            else:
                others = [v for v in q["option_values"] if v != q["answer_value"]]
                rng.shuffle(others)
                order = others[:p] + [q["answer_value"]] + others[p:]
                q["option_values"] = order
                q["options"] = [f"{LETTERS[i]}. {v}" for i, v in enumerate(order)]
            q["answer"] = LETTERS[int(p)]
    for grp in groups:
        cat = objs_by_scene[grp["scene_id"]]
        for q in grp["questions"]:
            q.setdefault("options", None)
            q["text"] = q["text_template"].format(**{k: cat[v] for k, v in q["slots"].items()})  # unrewritten version before Step 7
            q["bias_score"] = None
    # Difficulty: explore questions by oracle_steps terciles (provisional during the pilot, recomputed on the full set)
    steps = np.array([q["oracle_steps"] for grp in groups for q in grp["questions"] if q["evidence_label"] == "explore" and q["oracle_steps"] is not None])
    cuts = np.percentile(steps, [100 / 3, 200 / 3]) if len(steps) else [0, 0]
    for grp in groups:
        for q in grp["questions"]:
            s = q["oracle_steps"]
            q["difficulty"] = None if q["evidence_label"] != "explore" or s is None else ("easy" if s <= cuts[0] else "medium" if s <= cuts[1] else "hard")
    return groups, [float(c) for c in cuts]


def stats(groups):
    qs = [q for grp in groups for q in grp["questions"]]
    c = lambda key: dict(Counter(key(q) for q in qs))
    return {"num_groups": len(groups), "num_questions": len(qs),
            "per_template": c(lambda q: q["template"]), "per_variant": c(lambda q: f"{q['template']}:{q.get('variant', '-')}"),
            "evidence_label": c(lambda q: q["evidence_label"]), "difficulty": c(lambda q: q["difficulty"]),
            "answer_letter_per_template": {t: dict(Counter(q["answer"] for q in qs if q["template"] == t and q["answer_type"] == "multiple_choice"))
                                           for t in sorted({q["template"] for q in qs})},
            "quadrant_per_template": {t: dict(Counter(q["answer_value"] for q in qs if q["template"] == t)) for t in ("T3", "T4")},
            "oracle_views": c(lambda q: q["oracle_views"]), "oracle_steps": c(lambda q: q["oracle_steps"]),
            "constraints": dict(Counter(grp["constraints"][0]["type"] if grp["constraints"] else "none" for grp in groups))}


def scene_eligible(cfg, dataset, vid):
    """Scene-level exclusions from earlier steps: POSE_INVALID / ORIENTATION_CONFLICT / TOO_FEW_KEYFRAMES / TOO_FEW_OBJECTS / GRAPH_DISCONNECTED."""
    wd = work_dir(cfg, dataset, vid)
    st = read_json(wd / "scene_meta.json")["status"]
    reasons = [k for k in ("pose_valid", "orientation_ok", "vga_image_orientation_ok") if st.get(k) is False]
    if dataset == "arkitscenes" and not st.get("resolution_ok", False):
        reasons.append("resolution_ok")
    if not read_json(wd / "keyframes.json")["scene_ok"]:
        reasons.append("TOO_FEW_KEYFRAMES")
    if not read_json(work_dir(cfg, objects_dataset(dataset), vid) / "step3_summary.json")["scene_ok"]:
        reasons.append("TOO_FEW_OBJECTS")
    if not read_json(wd / "step4_summary.json")["ok"]:
        reasons.append("GRAPH_DISCONNECTED")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arkitscenes")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--ignore-scene-checks", action="store_true", help="debugging only: skip scene-level exclusion checks")
    args = ap.parse_args()
    cfg = load_config()
    scenes = sorted(str(s) for s in (args.scenes or cfg["arkitscenes"]["pilot_scenes"]))
    import json
    st = run(cfg, args.dataset, scenes, args.ignore_scene_checks)
    print(json.dumps(st, indent=1, default=str))


def run(cfg, dataset, scenes, ignore_scene_checks=False, out_dir=None, raw_name="groups_raw.json"):
    """Main Step 5–6 pipeline (formerly the body of main). out_dir defaults to work/<dataset>/_step6; raw_name is the file name of each scene's intermediate result. Returns statistics."""
    CANDIDATE_REJECTS.clear()
    bal = Balancer()
    groups, objs_by_scene, special_by_scene, drop_logs = [], {}, {}, {}
    for vid in scenes:
        bad = scene_eligible(cfg, dataset, vid)
        if bad and not ignore_scene_checks:
            print(vid, "skipped:", bad)
            continue
        sc = Scene(cfg, dataset, vid)
        objs_by_scene[sc.scene_id] = sc.cat
        drops = DropLog(sc.wd, "step56")
        drop_logs[sc.scene_id] = drops
        for o in sc.size_ref_dropped:
            drops.add("object", o, "NOT_REFERABLE", detail="larger/smaller instances never visible in the same node", name=sc.cat[o])
        for o in sc.size_ref_disabled:
            drops.add("object", o, "NOT_REFERABLE", detail="size reference disabled (size_ref_enabled: false)", name=sc.cat[o])
        rng = scene_rng(cfg, sc.scene_id, "questions")
        grps = sample_groups(sc, drops)
        scene_used = SceneUsed(cfg["questions"]["max_t4_per_object_per_scene"], cfg["questions"].get("t2_dedup_by_answer", False))
        out = []
        for gi, grp in enumerate(grps):
            n_dt = sum(r["constraint"]["type"] == "distance_transitivity" for r in out)
            r = build_group(sc, grp, gi, cfg["questions"], cfg["groups"], rng, bal, drops, scene_used, n_dt)
            if r:
                out.append(r)
        special = add_special_questions(sc, out, cfg["questions"], cfg["groups"], scene_rng(cfg, sc.scene_id, "special"), bal, drops, scene_used) if out else \
            {"unanswerable": {"added": None, "reason": "scene has no question groups"}, "init": {"added": None, "reason": "scene has no question groups"}}
        special_by_scene[vid] = special
        add_link_views(sc, out)
        write_json(sc.wd / raw_name, {"clusters": grps, "groups": out, "special": special})
        groups += out
        print(vid, "referable", len(sc.referable), "clusters", len(grps), "groups", len(out), "questions", sum(len(x["questions"]) for x in out))
    rebalanced = rebalance_quadrants(groups, cfg["groups"], cfg["questions"]["quadrant_balance_tol"], drop_logs,
                                     np.random.default_rng([cfg["seed"], 23]))
    groups = enforce_pair_cap(groups, cfg["questions"]["pair_max_frac"], drop_logs)
    groups, cuts = finalize(groups, cfg, objs_by_scene)
    out_dir = out_dir or path_of(cfg, "work") / dataset / "_step6"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "test_groups.jsonl", "w") as f:
        for grp in groups:
            f.write(json.dumps(grp, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else int(o)) + "\n")
    st = stats(groups)
    st["difficulty_cuts_oracle_steps"] = cuts
    st["candidate_filter_counts"] = dict(sorted(CANDIDATE_REJECTS.items()))
    st["special_attempts"] = special_by_scene
    st["quadrant_rebalance_removed"] = rebalanced
    write_json(out_dir / "step6_stats.json", st)
    return st


if __name__ == "__main__":
    main()
