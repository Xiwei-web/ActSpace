"""Per-group evaluation metadata (primary efficiency reference for Active: group_route_steps).

group_route_steps = on the (masked) view graph, the minimum number of steps from the start to visit every oracle view, link view and T4 path view of the group
(agents.plan_route, solved exactly when there are ≤ 14 targets; all exact on v0.11). Per-question oracle_steps are kept as a secondary reference.

The metadata are derived deterministically from the frozen data and written to <paths.eval_runs>/meta/<version>_group_meta.json (not in the repository; frozen data untouched);
the file records the SHA-256 of test_groups.jsonl and is regenerated when it does not match the frozen data.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agents import oracle_targets, plan_route
from .data import FrozenData


def compute_group_meta(data: FrozenData, gid: str) -> dict:
    g = data.groups[gid]
    scene, masked = data.scene(g["scene_id"]), data.masked_nodes(gid)
    targets = oracle_targets(g)
    route, unreachable = plan_route(scene, masked, g["init_frame"], targets)
    assert not unreachable, (gid, unreachable)
    answerable = [q for q in g["questions"] if q["evidence_label"] != "unanswerable"]
    return {"group_id": gid, "scene_id": g["scene_id"], "n_questions": len(g["questions"]),
            "masked": bool(masked), "group_route_steps": len(route), "n_targets": len([t for t in dict.fromkeys(targets) if t != g["init_frame"]]),
            "route_exact": len(set(targets) - {g["init_frame"]}) <= 14,
            "oracle_steps": {q["qid"]: q.get("oracle_steps") for q in g["questions"]},
            "oracle_steps_max": max([q["oracle_steps"] or 0 for q in answerable], default=0),
            "oracle_steps_sum": sum(q["oracle_steps"] or 0 for q in answerable)}


def group_meta(data: FrozenData, rebuild: bool = False) -> dict:
    """{group_id: metadata}; read the cache file directly when it matches the frozen data."""
    sha = data.manifest["files"]["test_groups.jsonl"]["sha256"]
    path = Path(data.ecfg["paths"]["eval_runs"]) / "meta" / f"{data.version}_group_meta.json"
    if path.exists() and not rebuild:
        m = json.loads(path.read_text())
        if m.get("test_groups_sha256") == sha and len(m["groups"]) == len(data.groups):
            return m["groups"]
    groups = {gid: compute_group_meta(data, gid) for gid in data.group_ids}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": data.version, "test_groups_sha256": sha,
                                "definition": "group_route_steps: minimum number of steps to visit all oracle_nodes, link_views and T4 path_views of the group (masked view graph)",
                                "groups": groups}, ensure_ascii=False, indent=1))
    return groups


def feasible_groups(meta: dict, budget: int) -> set:
    """Groups whose group-level shortest route fits within this budget."""
    return {gid for gid, m in meta.items() if m["group_route_steps"] <= budget}


def efficiency(route_steps: int, steps_used: int) -> float:
    """Efficiency = group_route_steps / steps used, capped at 1; computed only for groups where all evidence was seen (decided by the caller)."""
    if steps_used <= 0:
        return 1.0
    return min(1.0, route_steps / steps_used)
