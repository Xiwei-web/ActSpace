"""Synthetic frozen dataset for tests: no real data, no images from any dataset.

make_synthetic(root) writes a tiny "merged_v1.0" with the same layout as a real frozen version:

    <root>/work/merged/_step8/merged_v1.0/{test_groups.jsonl, freeze_manifest.json}
    <root>/work/<dataset>/<scene>/{view_graph.json, frames_all.json, visibility.npz, objects.json}
    <root>/frames/<scene_id>/<frame_id>.png|.jpg     (rebuild_frames.py layout: full resolution, upright)
    <root>/paths.yaml                                 (work, frames, results; point ACTSPACE_PATHS here)

The version directory gets a real freeze manifest and is made read-only, so the startup verification passes.
Command line: `python tests/synthetic.py <dir>` (needs PYTHONPATH=src).

Two scenes (one per dataset), 3 x 2 positions x 4 headings = 24 view-graph nodes each, all six actions. One group per
scene with T1-T5; the ScanNet++ group is masked and contains one unanswerable T1 question.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

GROUPS = ["arkit_g_000000", "scannetpp_g_000001"]   # scannetpp_g_000000 is in env.budget_infeasible
HEADINGS = (0, 90, 180, 270)
CELLS = [(x, y) for y in range(2) for x in range(3)]
VIS_KEYS = ("rho", "completeness", "area_frac", "in_frame_frac", "n_infrustum", "n_visible")


def _nodes():
    return [(f"f_{i:06d}", c, h) for i, (c, h) in enumerate((c, h) for c in CELLS for h in HEADINGS)]


def _graph(nodes):
    by = {(c, h): n for n, c, h in nodes}
    step = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}
    edges = {}
    for n, (x, y), h in nodes:
        def move(dh):
            dx, dy = step[(h + dh) % 360]
            return by.get(((x + dx, y + dy), h))
        edges[n] = {"forward": move(0), "backward": move(180), "left": move(90), "right": move(270),
                    "turn_left": by[((x, y), (h + 90) % 360)], "turn_right": by[((x, y), (h - 90) % 360)]}
    return {"nodes": [n for n, _, _ in nodes], "edges": edges}


OBJECTS = [(0, "chair", (2.6, 0.0)), (1, "table", (1.0, 1.8)), (2, "lamp", (-0.7, 1.0)), (3, "plant", (1.0, -0.8))]


def _visibility(nodes):
    """state 2 when the object is within 40 degrees of the heading and closer than 2.2 m, 1 within 60 degrees and 3 m."""
    st = np.zeros((len(OBJECTS), len(nodes)), np.int8)
    for j, (n, (x, y), h) in enumerate(nodes):
        for i, (_, _, (ox, oy)) in enumerate(OBJECTS):
            dx, dy = ox - x, oy - y
            d = float(np.hypot(dx, dy))
            ang = abs((np.degrees(np.arctan2(dy, dx)) - h + 180) % 360 - 180)
            st[i, j] = 2 if (ang < 40 and d < 2.2) else 1 if (ang < 60 and d < 3.0) else 0
    return st


def _image(color, size):
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), [255 - c for c in color], -1)
    return img


def _question(qid, template, text, answer_type, answer, evidence, oracle, steps, options=None, variant=None, answer_raw=None):
    q = {"qid": qid, "template": template, "text": text, "answer_type": answer_type, "answer": answer, "options": options,
         "evidence_label": "explore", "evidence_objs": evidence, "oracle_nodes": oracle, "link_views": [], "path_views": [],
         "oracle_views": len(oracle), "oracle_steps": steps, "difficulty": "medium", "ability": template.lower(),
         "slots": {f"OBJ{k + 1}": o for k, o in enumerate(evidence)}, "margins": {}, "paraphrase_key": template, "paraphrase_id": 0,
         "source_qid": qid, "bias_score": 0.0, "text_template": text}
    if variant:
        q["variant"] = variant
    if answer_raw is not None:
        q["answer_raw"] = answer_raw
    return q


def _questions(prefix, n):
    """n(index) → node id of position index × heading index."""
    quads = ["A. front-left", "B. front-right", "C. back-left", "D. back-right"]
    return [
        _question(f"{prefix}_q1", "T1", "How far is the center of the chair from the center of the table, in meters?", "numeric", 2.4,
                  [0, 1], [n(2, 0), n(1, 1)], 3, answer_raw=2.43),
        _question(f"{prefix}_q2", "T2", "Which object is closest to the lamp?", "multiple_choice", "B", [0, 1, 2],
                  [n(0, 2), n(1, 1)], 2, options=["A. chair", "B. table", "C. plant"], variant="closest"),
        _question(f"{prefix}_q3", "T3", "If you stood at the table facing the chair, where would the lamp be?", "multiple_choice", "C",
                  [0, 1, 2], [n(1, 1), n(0, 2)], 2, options=quads),
        _question(f"{prefix}_q4", "T4", "Relative to your starting position and direction, where is the chair?", "multiple_choice", "A",
                  [0], [n(2, 0)], 2, options=quads),
        _question(f"{prefix}_q5", "T5", "Among the objects on your left, which is closest to you?", "multiple_choice", "A", [1, 2],
                  [n(1, 1)], 1, options=["A. table", "B. lamp", "C. plant"]),
    ]


def make_synthetic(root: Path) -> Path:
    root = Path(root).resolve()
    work, frames = root / "work", root / "frames"
    vdir = work / "merged" / "_step8" / "merged_v1.0"
    if vdir.exists():                       # regenerate: the previous synthetic version directory was made read-only
        for f in [vdir, *vdir.rglob("*")]:
            f.chmod(f.stat().st_mode | 0o200)
    vdir.mkdir(parents=True, exist_ok=True)
    nodes = _nodes()
    graph = _graph(nodes)
    vis = _visibility(nodes)
    idx = {(c, h): n for n, c, h in nodes}

    def n(cell, head):
        return idx[(CELLS[cell], HEADINGS[head])]

    groups = []
    for ds, sid, size, ext, rot in (("arkitscenes", "arkit_10000001", (640, 480), ".png", 0),
                                    ("scannetpp", "scannetpp_0000aaaa01", (1920, 1440), ".jpg", 0)):
        sd = work / ds / sid.split("_", 1)[1]
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "view_graph.json").write_text(json.dumps(graph))
        fa = [{"frame_id": nid, "position": [float(x), float(y), 1.4], "yaw_deg": float(h), "rel_path": f"synthetic/{nid}{ext}",
               "rotation_cw_quarters": rot, "in_graph": True, "image_size": list(size)} for nid, (x, y), h in nodes]
        (sd / "frames_all.json").write_text(json.dumps(fa))
        np.savez(sd / "visibility.npz", obj_ids=np.array([o[0] for o in OBJECTS]), frame_ids=np.array(graph["nodes"]), state=vis,
                 **{k: np.zeros(vis.shape, np.float32) for k in VIS_KEYS})
        (sd / "objects.json").write_text(json.dumps([{"obj_id": i, "category": c, "ref_name": c, "referable": True,
                                                      "center": [x, y, 0.5]} for i, c, (x, y) in OBJECTS]))
        fd = frames / sid
        fd.mkdir(parents=True, exist_ok=True)
        for k, (nid, _, _) in enumerate(nodes):
            cv2.imwrite(str(fd / f"{nid}{ext}"), _image((40 + 8 * k, 120, 200 - 6 * k), size))
        gid = GROUPS[0] if ds == "arkitscenes" else GROUPS[1]
        g = {"group_id": gid, "scene_id": sid, "source_dataset": ds, "source_group_id": gid.split("_", 1)[1], "init_frame": n(0, 0),
             "cluster_obj_ids": [0, 1, 2], "num_p0_candidates": 1, "constraints": [], "human_verified": False,
             "questions": _questions(gid, n)}
        if ds == "scannetpp":           # masked group: remove the views that see the plant (3) and add an unanswerable T1
            masked = [graph["nodes"][j] for j in np.nonzero(vis[3] > 0)[0]]
            keep = {g["init_frame"]} | {x for q in g["questions"] for x in q["oracle_nodes"]}
            masked = [m for m in masked if m not in keep]
            g["view_graph_mask"] = {"masked_obj": 3, "masked_nodes": masked}
            q = _question(f"{gid}_q6", "T1", "How far is the center of the plant from the center of the chair, in meters?", "numeric",
                          1.8, [3, 0], [], None, answer_raw=1.84)
            q.update(evidence_label="unanswerable", oracle_views=None, difficulty=None, masked_obj=3, masked_nodes=masked)
            g["questions"].append(q)
        groups.append(g)
    tg = vdir / "test_groups.jsonl"
    tg.write_text("".join(json.dumps(g) + "\n" for g in groups))
    paths = root / "paths.yaml"
    paths.write_text(f"paths:\n  work: {work}\n  frames: {frames}\n  results: {root / 'results'}\n  models: {root / 'models'}\n")
    freeze_synthetic(paths)
    return paths


def freeze_synthetic(paths: Path) -> None:
    """Write freeze_manifest.json with the same hashes `actspace_build.merge_build verify` checks (version dir, scene files,
    configs, source manifests) and make the version directory read-only, so FrozenData(verify=True) accepts the data."""
    import os
    import stat
    old = os.environ.get("ACTSPACE_PATHS")
    os.environ["ACTSPACE_PATHS"] = str(paths)
    try:
        from actspace_build.merge_build import hashes, vdir
        from actspace_build.merge_dryrun import load_merge_config, version_dir
        cfg = load_merge_config()
        for ds in ("arkitscenes", "scannetpp"):                      # stand-ins for the two single-dataset source versions
            src = version_dir(cfg, ds) / "freeze_manifest.json"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(json.dumps({"version": cfg["merge"]["sources"][ds]["version"], "frozen": True, "synthetic": True}))
        vd = vdir(cfg)
        h = hashes(cfg)
        n_groups = sum(1 for _ in open(vd / "test_groups.jsonl"))
        (vd / "freeze_manifest.json").write_text(json.dumps({"version": cfg["merge"]["version"], "frozen": True, "synthetic": True,
                                                             "counts": {"groups": n_groups}, **h}, indent=1))
        for f in [*vd.rglob("*"), vd]:
            f.chmod(f.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    finally:
        if old is None:
            os.environ.pop("ACTSPACE_PATHS", None)
        else:
            os.environ["ACTSPACE_PATHS"] = old


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
    print(f"export ACTSPACE_PATHS={make_synthetic(out).resolve()}")
