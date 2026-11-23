"""ARKit question integrity checks (the applicable items of the integrity checklist in docs/data_construction.md) and preview
export (question format of docs/data_construction.md).

Usage: python -m actspace_build.integrity check | export
Outputs: reports/arkit_integrity_check.md; reports/arkit_test_groups_preview.jsonl
The checks recompute everything independently of the question-generation code: answers use only the object centers in
objects.json, the pose of p0 in frames_all.json and the direction functions in geometry.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict

import numpy as np

from . import geometry as g
from .common import load_config, path_of, read_json, work_dir
from .step4_graph_visibility import ACTIONS, STATE_INVISIBLE, STATE_VISIBLE
from .step56_questions import dedup_key

DATASET = "arkitscenes"
LETTERS = "ABCD"


def load_groups(cfg, stage="_step6"):
    return [json.loads(l) for l in open(path_of(cfg, "work") / DATASET / stage / "test_groups.jsonl")]


class SceneData:
    def __init__(self, cfg, vid):
        wd = work_dir(cfg, DATASET, vid)
        self.objs = {o["obj_id"]: o for o in read_json(wd / "objects.json")}
        vis = np.load(wd / "visibility.npz")
        self.state = vis["state"]
        self.row = {int(o): k for k, o in enumerate(vis["obj_ids"])}
        self.nodes = [str(x) for x in vis["frame_ids"]]
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.graph = read_json(wd / "view_graph.json")
        fr = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
        self.pose = {n: fr[n] for n in self.nodes}

    def c(self, o):
        return np.asarray(self.objs[o]["center"])

    def dist(self, a, b):
        return float(np.linalg.norm(self.c(a) - self.c(b)))

    def keep_mask(self, grp):
        keep = np.ones(len(self.nodes), bool)
        m = grp.get("view_graph_mask")
        if m:
            for n in m["masked_nodes"]:
                keep[self.node_index[n]] = False
        return keep

    def strongly_connected(self, keep):
        idx = [i for i in range(len(self.nodes)) if keep[i]]
        if not idx:
            return False
        adj = defaultdict(list)
        radj = defaultdict(list)
        for n, e in self.graph["edges"].items():
            for a in ACTIONS:
                t = e.get(a)
                if t is not None and keep[self.node_index[n]] and keep[self.node_index[t]]:
                    adj[n].append(t)
                    radj[t].append(n)

        def reach(start, A):
            seen, stack = {start}, [start]
            while stack:
                for w in A[stack.pop()]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            return seen
        s = self.nodes[idx[0]]
        return len(reach(s, adj)) == len(idx) == len(reach(s, radj))


def expected_answer(sd: SceneData, q, init_frame, qc):
    """Recompute the ground truth independently: numeric questions return (answer, answer_raw); multiple-choice questions return
    the value of the correct option (object id or quadrant)."""
    s = q["slots"]
    t = q["template"]
    if t == "T1":
        d = sd.dist(s["OBJ1"], s["OBJ2"])
        return round(round(d / qc["answer_round_m"]) * qc["answer_round_m"], 1), round(d, qc["store_round_decimals"])
    if t == "T2":
        a, opts = s["OBJ1"], q["option_objs"]
        d = [sd.dist(a, o) for o in opts]
        return opts[int(np.argmax(d) if q.get("variant") == "farthest" else np.argmin(d))]
    if t == "T3":
        th = g.direction_theta_facing(sd.c(s["OBJ1"]), sd.c(s["OBJ2"]), sd.c(s["OBJ3"]))
        return g.quadrant(th)
    if t == "T4":
        f = sd.pose[init_frame]
        return g.quadrant(g.theta_from_yaw(f["position"], f["yaw_deg"], sd.c(s["OBJ1"])))
    if t == "T5":
        a, b = s["OBJ1"], s["OBJ2"]
        left = [o for o in q["option_objs"] if g.direction_theta_facing(sd.c(a), sd.c(b), sd.c(o)) > 0]
        return min(left, key=lambda o: sd.dist(a, o)) if left else None
    raise ValueError(t)


def stored_choice(q):
    k = LETTERS.index(q["answer"])
    return q["option_objs"][k] if "option_objs" in q else q["option_values"][k]


def linked(sd, ev, keep, mode, nodes=None):
    """Whether the graph with evidence objects as vertices and "co-visible at the same (unmasked) node" as edges is connected
    (recomputed independently of step56_questions.evidence_linked). If nodes is given, only those nodes are used (checks that
    link_views suffice to link the evidence)."""
    ok = keep.copy()
    if nodes is not None:
        ok &= np.isin(np.arange(len(keep)), [sd.node_index[n] for n in nodes])
    vis = {o: set(np.flatnonzero((sd.state[sd.row[o]] == STATE_VISIBLE) & ok)) for o in set(ev)}
    seen_ = {o: set(np.flatnonzero((sd.state[sd.row[o]] != STATE_INVISIBLE) & ok)) for o in set(ev)}

    def edge(a, b):
        if mode == "visible_visible":
            return bool(vis[a] & vis[b])
        if mode == "visible_partial":
            return bool(vis[a] & seen_[b]) or bool(seen_[a] & vis[b])
        return bool(seen_[a] & seen_[b])
    objs = list(vis)
    seen, todo = {objs[0]}, [objs[0]]
    while todo:
        a = todo.pop()
        for b in objs:
            if b not in seen and edge(a, b):
                seen.add(b); todo.append(b)
    return len(seen) == len(objs)


def check(cfg, stage="_step6"):
    qc = cfg["questions"]
    groups = load_groups(cfg, stage)
    scenes = {}
    res = defaultdict(lambda: [0, []])  # check item -> [number passed, failure details]

    def record(item, ok, detail):
        if ok:
            res[item][0] += 1
        else:
            res[item][1].append(detail)

    from .step3_objects import load_blacklist
    from .step56_questions import LINKED_TEMPLATES
    black_objs, black_scenes = load_blacklist(cfg)
    oc = cfg["objects"]
    scene_keys = defaultdict(Counter)
    letters = defaultdict(Counter)
    quads = defaultdict(Counter)
    nonref_evidence = 0
    for grp in groups:
        vid = grp["scene_id"].split("_", 1)[1]
        sd = scenes.setdefault(vid, SceneData(cfg, vid))
        gid = grp["group_id"]
        keep = sd.keep_mask(grp)
        record("init_frame is a view-graph node", grp["init_frame"] in sd.node_index, gid)
        qby = {q["qid"]: q for q in grp["questions"]}
        for q in grp["questions"]:
            qid = q["qid"]
            # 1. ground truth
            exp = expected_answer(sd, q, grp["init_frame"], qc)
            if q["answer_type"] == "numeric":
                record("Recomputed ground truth matches", (q["answer"], q["answer_raw"]) == exp, f"{qid}: stored {q['answer']}/{q['answer_raw']}, recomputed {exp}")
            else:
                record("Recomputed ground truth matches", stored_choice(q) == exp, f"{qid}: stored {stored_choice(q)}, recomputed {exp}")
                n = len(q.get("option_objs") or q["option_values"])
                letters[(q["template"], n)][q["answer"]] += 1
                if q["template"] in ("T3", "T4"):
                    quads[q["template"]][stored_choice(q)] += 1
            # 2. evidence label vs. visibility matrix (masked groups use the masked view graph)
            ev = q["evidence_objs"]
            st = {o: sd.state[sd.row[o]] for o in ev}
            p0 = sd.node_index[grp["init_frame"]]
            lab = q["evidence_label"]
            if lab == "init":
                ok = all(st[o][p0] == STATE_VISIBLE for o in ev)
            elif lab == "explore":
                ok = any(st[o][p0] != STATE_VISIBLE for o in ev) and all((st[o][keep] == STATE_VISIBLE).any() for o in ev)
            else:  # unanswerable: at least one evidence object is invisible at all (unmasked) nodes
                ok = any((st[o][keep] == STATE_INVISIBLE).all() for o in ev)
            record("Evidence label consistent with the visibility matrix", ok, f"{qid}: {lab}")
            nonref_evidence += sum(not sd.objs[o]["referable"] for o in ev)
            # Integrity checklist: all objects named in the question (slot and option objects) are referable;
            # with larger/smaller references, other instances of the same category may be non-referable evidence objects
            named = list(q["slots"].values()) + list(q.get("option_objs") or [])
            record("All named objects are referable", all(sd.objs[o]["referable"] for o in named), qid)
            extra = [o for o in ev if not sd.objs[o]["referable"]]
            record("Non-referable evidence objects are same-category instances of a size-referenced object",
                   all(any(o in sd.objs[n].get("size_group", []) and sd.objs[n]["ref_kind"] != "unique" for n in named) for o in extra), qid)
            scene_keys[grp["scene_id"]][dedup_key(q)] += 1
            # T4 path views are unmasked nodes, at most 4, exclude the start, and the last one is the oracle node of the last observation
            # (a size-referenced T4 has two evidence objects, the target and another instance of its category; the path ends at the one observed later)
            if q["template"] == "T4" and lab != "unanswerable" and "path_views" in q:
                pv = q["path_views"]
                ok = len(pv) <= 4 and grp["init_frame"] not in pv and all(n in sd.node_index and keep[sd.node_index[n]] for n in pv)
                if lab == "explore":
                    ok = ok and bool(pv) and pv[-1] == q["oracle_nodes"][-1] and any(st[o][sd.node_index[pv[-1]]] == STATE_VISIBLE for o in ev)
                record("T4 path views are unmasked nodes and end at the last oracle view", ok, qid)
            # rules added after the first GATE D round
            if oc["blacklist_enabled"]:
                record("No blacklisted objects or excluded scenes", vid not in black_scenes and not any((vid, o) in black_objs for o in ev + named), qid)
            if not oc["armchair_enabled"] or oc["size_disambig_max_instances"] <= 2:
                record("No armchair / largest / smallest", not re.search(r"\b(armchair|largest|smallest)\b", q["text"], re.I), qid)
            if not oc.get("size_ref_enabled", False):  # size references (larger / smaller) are disabled
                record("No size references (larger / smaller); every named object is the only one of its category in the scene",
                       not re.search(r"\b(larger|smaller)\b", q["text"], re.I) and all(sd.objs[o]["ref_kind"] == "unique" for o in named), qid)
            elif oc.get("size_ref_require_covisible"):
                sized = [n for n in named if sd.objs[n]["ref_kind"] not in ("unique", None)]
                from .step56_questions import covisible
                cm = oc.get("size_ref_covisible_mode", "visible_visible")
                record(f"Same-category instances of a size reference are co-visible at some node ({cm})",
                       all(covisible(sd.state, [sd.row[m] for m in sd.objs[n]["size_group"]], cm) for n in sized), qid)
            if qc.get("require_evidence_linked") and q["template"] in LINKED_TEMPLATES and lab != "unanswerable":
                record("Evidence objects are linked by co-visibility at shared nodes", linked(sd, ev, keep, qc["evidence_link_mode"]), qid)
                lv = q.get("link_views") or []
                record("Link views are unmasked nodes and suffice on their own to link the evidence",
                       all(n in sd.node_index and keep[sd.node_index[n]] for n in lv) and (len(set(ev)) < 2 or linked(sd, ev, keep, qc["evidence_link_mode"], lv)), qid)
            # 3. masking-method questions
            if q.get("unanswerable_method") == "mask":
                masked = set(q["masked_nodes"])
                record("Masked nodes exclude init_frame", grp["init_frame"] not in masked, qid)
                record("Remaining graph is strongly connected after masking", sd.strongly_connected(keep), qid)
                x = q["masked_obj"]
                record("Masked object is invisible at all remaining nodes", (st[x][keep] == STATE_INVISIBLE).all(), qid)
                record("Mask recorded in the group matches the question", grp.get("view_graph_mask", {}).get("masked_nodes") == q["masked_nodes"], qid)
        # 4. constraints
        for con in grp["constraints"]:
            qs = [qby[i] for i in con["qids"]]
            if con["type"] == "numeric_vs_comparison":
                q1, q2, q3 = qs
                b, c = q1["slots"]["OBJ2"], q2["slots"]["OBJ2"]
                ok = stored_choice(q3) == (b if q1["answer_raw"] < q2["answer_raw"] else c)
            elif con["type"] == "direction_inverse":
                ok = g.INVERSE_QUADRANT[stored_choice(qs[0])] == stored_choice(qs[1])
            else:  # distance_transitivity: the three comparison results do not form a cycle
                closer = {}
                for q in qs:
                    o1, o2 = q["option_objs"]
                    w = stored_choice(q)
                    closer[(w, o2 if w == o1 else o1)] = True
                objs = {o for q in qs for o in q["option_objs"]}
                ok = not any(closer.get((x, y)) and closer.get((y, z)) and closer.get((z, x)) for x, y, z in itertools.permutations(objs, 3))
            record("Constraint relation holds on the ground truth", ok, f"{grp['group_id']}: {con['type']}")
    # 5. no duplicate questions within a scene: the same T1/T2/T3/T5 content occurs once; the same T4 object at most max_t4 times
    for sid, c in scene_keys.items():
        for k, n in c.items():
            lim = qc["max_t4_per_object_per_scene"] if k[0] == "T4" else 1
            record("No duplicate questions within a scene", n <= lim, f"{sid}: {k[0]} {sorted(k[1]) if isinstance(k[1], frozenset) else k[1:]} occurs {n} times")
    # 6. endpoints of view-graph edges
    for vid, sd in scenes.items():
        ns = set(sd.nodes)
        ok = all(t is None or t in ns for e in sd.graph["edges"].values() for t in e.values())
        record("All view_graph edge endpoints are nodes", ok and set(sd.graph["edges"]) <= ns, vid)
    return groups, res, letters, quads, nonref_evidence


def write_report(cfg, groups, res, letters, quads, nonref_evidence, out_name="arkit_integrity_check.md", stage="_step6"):
    n_q = sum(len(g_["questions"]) for g_ in groups)
    L = ["# ARKit question integrity check", "",
         f"Scope: all {len(groups)} question groups and {n_q} questions ({len({g_['scene_id'] for g_ in groups})} scenes) in `work/arkitscenes/{stage}/test_groups.jsonl`. "
         "Answers are recomputed independently from the object centers in `objects.json` and the pose of p0 in `frames_all.json`; "
         "masking-method groups use the view graph masked by their `view_graph_mask`.", "",
         "| Check | Passed | Total | Result |", "|---|---|---|---|"]
    fails = []
    for item, (ok, bad) in res.items():
        tot = ok + len(bad)
        L.append(f"| {item} | {ok} | {tot} | {'100%' if not bad else f'**{ok / tot:.1%}**'} |")
        if bad:
            fails.append((item, bad))
    L += ["", "## Option-letter and quadrant distribution (each within ± 5% of uniform)", "",
          "| Template | Options | Questions | Share per letter | Within ±5% |", "|---|---|---|---|---|"]
    for (t, n), c in sorted(letters.items()):
        tot = sum(c.values())
        fr = [c[LETTERS[i]] / tot for i in range(n)]
        ok = all(abs(x - 1 / n) <= 0.05 for x in fr)
        L.append(f"| {t} | {n} | {tot} | " + ", ".join(f"{LETTERS[i]} {fr[i]:.1%}" for i in range(n)) + f" | {'yes' if ok else '**no**'} |")
    L += ["", "| Template | Questions | Share per quadrant | Within 25% ± 5% |", "|---|---|---|---|"]
    for t, c in sorted(quads.items()):
        tot = sum(c.values())
        fr = {k: c[k] / tot for k in g.QUADRANTS}
        L.append(f"| {t} | {tot} | " + ", ".join(f"{k} {v:.1%}" for k, v in fr.items()) + f" | {'yes' if all(abs(v - 0.25) <= 0.05 for v in fr.values()) else '**no**'} |")
    L += ["", "## Notes on the checks", "",
          "- \"All evidence_objs are referable\" is checked as: all objects named in the question (slot and option objects) are referable; "
          f"with larger / smaller references, other instances of the same category may be non-referable evidence objects (two rows above). There are {nonref_evidence} such evidence objects in total.",
          "- No duplicate questions within a scene: counted once per (scene, content key); T1 by unordered object pair, T2 by reference object + comparison direction + option set, "
          "T3 by (standing, facing, queried), T5 by (standing, facing), T4 by queried object (at most 2 per object).",
          "- \"At least one evidence object of an unanswerable question is invisible at all nodes\": unanswerable questions are generated only by the masking method, "
          "so this is checked on the masked view graph (row \"Evidence label consistent with the visibility matrix\" above).", ""]
    if fails:
        L += ["## Failed questions", ""]
        for item, bad in fails:
            L += [f"### {item} ({len(bad)})", "", *[f"- {b}" for b in bad[:200]], ""]
    (path_of(cfg, "reports") / out_name).write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))


def export(cfg, stage="_step6"):
    """Export with the question fields of docs/data_construction.md; masking-method questions also keep unanswerable_method /
    masked_obj / masked_nodes, and groups keep view_graph_mask."""
    fields = ["qid", "template", "ability", "text", "text_template", "slots", "answer_type", "options", "answer", "answer_raw",
              "evidence_objs", "evidence_label", "oracle_views", "oracle_steps", "difficulty", "margins", "bias_score"]
    out = path_of(cfg, "reports") / "arkit_test_groups_preview.jsonl"
    groups = load_groups(cfg, stage)
    with open(out, "w") as f:
        for grp in groups:
            qs = []
            for q in grp["questions"]:
                r = {k: q.get(k) for k in fields}
                if r["answer_type"] != "numeric":
                    r.pop("answer_raw")
                if q.get("unanswerable_method") == "mask":
                    r.update(unanswerable_method="mask", masked_obj=q["masked_obj"], masked_nodes=q["masked_nodes"])
                if "paraphrase_id" in q:
                    r.update(paraphrase_key=q["paraphrase_key"], paraphrase_id=q["paraphrase_id"])
                qs.append(r)
            rec = {"group_id": grp["group_id"], "scene_id": grp["scene_id"], "init_frame": grp["init_frame"],
                   "cluster_obj_ids": grp["cluster_obj_ids"], "questions": qs, "constraints": grp["constraints"],
                   "human_verified": grp["human_verified"]}
            if grp.get("view_graph_mask"):
                rec["view_graph_mask"] = grp["view_graph_mask"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"exported {len(groups)} groups to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "export"])
    ap.add_argument("--stage", default="_step6", help="question source: _step6 / _step7 / _step8/<version>")
    ap.add_argument("--out", default="arkit_integrity_check.md", help="report file name (under reports/)")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "check":
        write_report(cfg, *check(cfg, args.stage), out_name=args.out, stage=args.stage)
    else:
        export(cfg, args.stage)


if __name__ == "__main__":
    main()
