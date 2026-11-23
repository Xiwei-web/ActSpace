"""Per-group / per-question results and run summaries, derived only from trajectory records (can be recomputed from trajectories.jsonl when resuming). Shared by run.py and llm_run.py."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import answers as A
from .agents import evidence_seen
from .data import MOVE_ACTIONS, dataset_of_scene
from .meta import efficiency


def nodes_from_records(records) -> list:
    """Nodes the agent has seen (private): for static settings, the given photos; for Active, the start and visited nodes."""
    s = records[0]
    if s.get("mode", "active") != "active":
        return list(s["shown_nodes"])
    return [s["start_node"]] + [r["to"] for r in records if r["type"] == "step"]


def evidence_check(data, group, nodes):
    """Whether all evidence of each answerable question was seen (evidence_seen); whether the masked objects of unanswerable questions were never visible (should always hold)."""
    scene = data.scene(group["scene_id"])
    hit, leak = {}, {}
    for q in group["questions"]:
        if q["evidence_label"] == "unanswerable":
            x = q.get("masked_obj", (group.get("view_graph_mask") or {}).get("masked_obj"))
            leak[q["qid"]] = x is not None and any(scene.visible(x, n) for n in nodes)
        else:
            hit[q["qid"]] = evidence_seen(scene, q, nodes)
    return hit, leak


def episode_rows(data, meta, records):
    """One episode → (per-group record, list of per-question records).

    Blind does not accept cannot_determine and only counts answerable questions; unanswerable questions are left out of the per-question records (not in the denominator),
    and excluded_unanswerable in the per-group record counts the excluded questions.

    Groups whose shortest evidence route exceeds the budget (env.budget_infeasible, 7 ScanNet++ groups) run as usual; per-group / per-question records are flagged
    budget_infeasible; efficiency metrics are computed only on feasible groups (efficiency = None for infeasible groups). Per-question records also store dataset, gold (gold answer) and
    cannot_determine (whether the model answered cannot_determine)."""
    s, end = records[0], records[-1]
    gid, mode, budget = s["group_id"], s.get("mode", "active"), s["budget"]
    g, m = data.groups[gid], meta[gid]
    nodes = nodes_from_records(records)
    hit, leak = evidence_check(data, g, nodes)
    all_seen = all(hit.values())
    infeasible = gid in set(data.ecfg["env"].get("budget_infeasible") or [])
    qid2pub = {v: k for k, v in s["question_ids"].items()}
    counted = [q for q in g["questions"] if not (mode == "blind" and q["evidence_label"] == "unanswerable")]
    ep = {"group_id": gid, "scene_id": g["scene_id"], "mode": mode, "steps_used": end["steps_used"],
          "forced": end["forced"], "parse_ok": end["parse_ok"],
          "cannot_move": sum(r["type"] == "step" and not r["moved"] and r["action"] in MOVE_ACTIONS for r in records),
          "invalid_steps": sum(r["type"] == "step" and r["action"] not in MOVE_ACTIONS for r in records),
          "n_photos": len(set(nodes)), "group_route_steps": m["group_route_steps"],
          "route_feasible": m["group_route_steps"] <= budget, "oracle_steps_max": m["oracle_steps_max"],
          "evidence_hit_all": all_seen, "masked_obj_seen": any(leak.values()),
          "efficiency": efficiency(m["group_route_steps"], end["steps_used"]) if mode == "active" and all_seen and not infeasible else None,
          "budget_infeasible": infeasible, "dataset": dataset_of_scene(g["scene_id"]),
          "mean_score": float(np.mean([end["scores"][qid2pub[q["qid"]]] for q in counted])),
          "excluded_unanswerable": len(g["questions"]) - len(counted), "notes": sum("notes" in r for r in records)}
    if "route" in s:  # random_walk / oracle_route: route steps
        ep["route_moves"] = len(s["route"])
    per_q = []
    for q in counted:
        pub = qid2pub[q["qid"]]
        per_q.append({"group_id": gid, "scene_id": g["scene_id"], "qid": q["qid"], "template": q["template"],
                      "difficulty": q.get("difficulty"), "answerable": q["evidence_label"] != "unanswerable",
                      "masked_group": m["masked"], "score": end["scores"][pub], "answer": end["answers"][pub],
                      "parse_error": end["parse_errors"].get(pub), "evidence_hit": hit.get(q["qid"]),
                      "oracle_steps": q.get("oracle_steps"), "dataset": dataset_of_scene(g["scene_id"]), "gold": A.gold(q),
                      "cannot_determine": end["answers"][pub] == A.CANNOT,
                      "budget_infeasible": infeasible})
    return ep, per_q


def build_summary(head: dict, per_ep: list, per_q: list) -> dict:
    def mean(xs):
        return float(np.mean(xs)) if xs else None
    by_t = defaultdict(list)
    for x in per_q:
        by_t[(x["template"], "answerable" if x["answerable"] else "unanswerable")].append(x["score"])
    return {**head, "episodes": len(per_ep), "questions": len(per_q),
            "mean_steps": mean([e["steps_used"] for e in per_ep]),
            "mean_photos": mean([e["n_photos"] for e in per_ep]),
            "steps_hist": dict(sorted(Counter(e["steps_used"] for e in per_ep).items())),
            "forced_answers": dict(Counter(str(e["forced"]) for e in per_ep)),
            "parse_failures": sum(not e["parse_ok"] for e in per_ep),
            "invalid_steps": sum(e.get("invalid_steps", 0) for e in per_ep),
            "excluded_unanswerable": sum(e.get("excluded_unanswerable", 0) for e in per_ep),
            "mean_cannot_move_steps": mean([e["cannot_move"] for e in per_ep]),
            "mean_score_all_questions": mean([x["score"] for x in per_q]),
            "score_by_template": {f"{t}/{k}": {"mean": mean(v), "n": len(v)} for (t, k), v in sorted(by_t.items())},
            "evidence_hit_frac_answerable_questions": mean([x["evidence_hit"] for x in per_q if x["answerable"]]),
            "evidence_hit_answerable": [int(sum(x["evidence_hit"] for x in per_q if x["answerable"])), sum(x["answerable"] for x in per_q)],
            "episodes_with_masked_obj_seen": sum(e["masked_obj_seen"] for e in per_ep),
            "route_feasible_groups": sum(e["route_feasible"] for e in per_ep),
            "budget_infeasible_groups": sorted(e["group_id"] for e in per_ep if e.get("budget_infeasible")),
            "mean_efficiency_feasible": mean([e["efficiency"] for e in per_ep if e.get("efficiency") is not None]),
            "per_episode": per_ep, "per_question": per_q}
