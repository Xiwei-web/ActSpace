"""Step 3 (ARKitScenes): object preparation, unified categories, size / point-count / unique-referability filtering, and GATE B vocabulary frequency statistics.

Subcommands
  vocab_stats  read annotations only, over all deduplicated selected videos; count category frequencies and objects with a "unique category" (GATE B review material)
  objects      extract and filter object point sets for the given scenes; writes work/arkitscenes/<vid>/objects.json, object_points.npz
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

from . import arkit
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, work_dir, write_json


def load_vocab(cfg):
    vdir = path_of(cfg, "vocab")
    cats = pd.read_csv(vdir / "categories.tsv", sep="\t", dtype=str, keep_default_na=False)
    allowed = {r.category: r.allowed.strip().lower() == "true" for r in cats.itertuples()}
    mapping = pd.read_csv(vdir / "label_mapping_arkitscenes.tsv", sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(mapping.source_label, mapping.category)), allowed


def obb_yaw_and_size(axes: np.ndarray, size: np.ndarray):
    """Gravity-aligned OBB -> (yaw_deg, [main horizontal edge, second horizontal edge, height], box axes in the same order). Rows of axes are box axes (z-up world frame).
    The returned edge lengths and axes must be in the same order, otherwise length and width are swapped when drawing boxes (the cause of a box-drawing bug in earlier report cards)."""
    k_up = int(np.argmax(np.abs(axes[:, 2])))
    horiz = [i for i in range(3) if i != k_up]
    i0, i1 = sorted(horiz, key=lambda i: -size[i])
    a = axes[i0]
    yaw = float(np.degrees(np.arctan2(a[1], a[0])))
    yaw = (yaw + 90.0) % 180.0 - 90.0  # box axes have no sign, so normalise to [-90, 90)
    return yaw, [float(size[i0]), float(size[i1]), float(size[k_up])], axes[[i0, i1, k_up]]


def points_in_obb(V: np.ndarray, center, axes, size, inflate, floor_cut) -> np.ndarray:
    """Object point set: the OBB is inflated by `inflate` only horizontally and upwards, not downwards, and vertices with z < box bottom + floor_cut are dropped
    (so floor and carpet vertices near the box bottom are not counted as part of the object). Rows of axes are box axes (z-up world frame)."""
    axes, size, center = np.asarray(axes), np.asarray(size), np.asarray(center)
    k_up = int(np.argmax(np.abs(axes[:, 2])))
    half = 0.5 * size
    bottom_z = center[2] - half[k_up] * abs(axes[k_up, 2])
    top_z = center[2] + half[k_up] * abs(axes[k_up, 2])
    r = float(np.linalg.norm(half + inflate))
    near = np.where(np.all(np.abs(V - center) <= r, axis=1))[0]
    local = (V[near] - center) @ axes.T
    horiz = [i for i in range(3) if i != k_up]
    ok = np.all(np.abs(local[:, horiz]) <= half[horiz] + inflate, axis=1)
    z = V[near, 2]
    ok &= (z >= bottom_z + floor_cut) & (z <= top_z + inflate)
    return near[ok]


def inside_obb(P: np.ndarray, center, axes, size) -> np.ndarray:
    """Whether points lie inside the (non-inflated) OBB; rows of axes are box axes."""
    local = (P - np.asarray(center)) @ np.asarray(axes).T
    return np.all(np.abs(local) <= 0.5 * np.asarray(size), axis=1)


_VOL_GRID = (np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, 11)] * 3, indexing="ij"), -1).reshape(-1, 3))


def volume_frac_inside(a: dict, b: dict) -> float:
    """Fraction of box a's volume that lies inside box b (estimated with 11^3 uniform grid points inside a)."""
    P = np.asarray(a["center"]) + (_VOL_GRID * np.asarray(a["size"])) @ np.asarray(a["axes"])
    return float(inside_obb(P, b["center"], b["axes"], b["size"]).mean())


def object_point_sets(V: np.ndarray, boxes: list[dict], ocfg) -> tuple[list[np.ndarray], list[int], dict]:
    """Mesh vertex indices per object: points_in_obb, then vertices that also fall inside another object's (non-inflated) OBB are handled as follows:
    - nested: >= nested_min_volume_frac of the smaller box's volume lies inside the larger box -> overlapping vertices go to the smaller object and are removed only from the larger one;
    - adjacent: otherwise -> overlapping vertices are removed from both (e.g. a small table box reaching into a sofa armrest).
    boxes: [{center, axes, size}] (z-up world frame). Returns (point sets, number of vertices removed per object, statistics)."""
    sets = [points_in_obb(V, b["center"], b["axes"], b["size"], ocfg["arkit_obb_inflate_m"], ocfg["floor_cut_m"]) for b in boxes]
    vol = [float(np.prod(b["size"])) for b in boxes]
    n = len(boxes)
    nested_in = {}  # (small, large) -> True
    for k in range(n):
        for j in range(n):
            if k != j and vol[k] <= vol[j] and volume_frac_inside(boxes[k], boxes[j]) >= ocfg["nested_min_volume_frac"]:
                nested_in[(k, j)] = True
    out, removed = [], []
    kept_by_nesting = set()
    for k, idx in enumerate(sets):
        drop = np.zeros(len(idx), bool)
        for j, b in enumerate(boxes):
            if j == k or not len(idx):
                continue
            shared = inside_obb(V[idx], b["center"], b["axes"], b["size"])
            if not shared.any():
                continue
            if nested_in.get((k, j)):
                kept_by_nesting.add(k)  # k is nested in j: overlapping vertices go to k
                continue
            drop |= shared  # k is the larger box (j nested in k) or the two are adjacent
        out.append(idx[~drop])
        removed.append(int(drop.sum()))
    stats = {"nested_pairs": len(nested_in), "objects_kept_by_nesting": len(kept_by_nesting),
             "objects_with_vertices_removed": int(sum(r > 0 for r in removed))}
    return out, removed, stats


def filter_reasons(cat, ref_kind, allowed, n_points, size, ocfg) -> list[str]:
    """Step 3 object filtering. ref_kind None means the object cannot be referred to uniquely by name (category not unique and no size disambiguation)."""
    reasons = []
    if cat is None or not allowed.get(cat, False) or cat in ocfg["banned_categories"] or ref_kind is None:
        reasons.append("NOT_REFERABLE")
    if n_points < ocfg["min_object_points"]:
        reasons.append("UNRELIABLE_GEOMETRY")
    if max(size) < ocfg["min_longest_edge_m"]:
        reasons.append("TOO_SMALL")
    return reasons


def load_blacklist(cfg):
    """Objects in configs/object_blacklist.yaml are no longer used for questions, and listed scenes are dropped entirely. Returns (object set, scene set)."""
    import yaml
    from .common import ROOT
    p = ROOT / cfg["objects"]["blacklist_file"]
    if not cfg["objects"]["blacklist_enabled"] or not p.exists():
        return set(), set()
    d = yaml.safe_load(open(p))
    return ({(str(o["video_id"]), int(o["obj_id"])) for o in d.get("objects") or []},
            {str(x["video_id"]) for x in d.get("scenes") or []})


def assign_references(raw: list[dict], cats: list, ocfg, blacklisted=frozenset()) -> list[dict]:
    """Determine how each annotated object is referred to:
    - if exactly one sofa instance has longest edge < armchair_max_longest_edge_m, it is called "armchair" (and takes part in uniqueness as its own category);
    - a category with exactly 1 instance in the scene -> "unique", and the question uses the category name;
    - with 2..size_disambig_max_instances instances of a category whose centres are pairwise >= size_disambig_min_center_dist_m apart, size references are enabled
      by volume ratio: "larger / smaller X" for 2 instances, "largest / smallest X" for 3; all instances of the category are recorded in size_group.
      Size references are permanently disabled (size_ref_enabled: false, for both ARKit and ScanNet++): objects are referred to only by a category name unique in the scene.
    Returns {eff_cat, ref_kind, ref_name, size_group} per object; ref_kind None means the object cannot be referred to uniquely by name."""
    eff = list(cats)
    if ocfg["armchair_enabled"]:  # the armchair rule is disabled (GATE D repeatedly found it invalid)
        sofas = [k for k, c in enumerate(cats) if c == "sofa"]
        small = [k for k in sofas if max(raw[k]["size"]) < ocfg["armchair_max_longest_edge_m"]]
        if len(small) == 1:
            eff[small[0]] = "armchair"
    count = Counter(eff)
    out = [{"eff_cat": c, "ref_kind": "unique" if c is not None and count[c] == 1 else None, "ref_name": c, "size_group": [k]}
           for k, c in enumerate(eff)]
    # Tightened: exactly 2 instances of the category (size_disambig_max_instances: 2, no more largest/smallest), volume ratio >= 2.5;
    # "a node where both are visible" needs the visibility matrix and is checked when Steps 5-6 load the scene (step56_questions.Scene)
    if not ocfg.get("size_ref_enabled", False):
        return out
    for c in sorted({c for c in eff if c is not None and 2 <= count[c] <= ocfg["size_disambig_max_instances"]}):
        members = [k for k, ck in enumerate(eff) if ck == c]
        if any(m in blacklisted for m in members):
            continue  # the category has an instance flagged in human review as badly annotated; no size reference
        centers = np.array([raw[m]["center"] for m in members])
        if min(np.linalg.norm(a - b) for a, b in itertools.combinations(centers, 2)) < ocfg["size_disambig_min_center_dist_m"]:
            continue  # most likely parts of the same object (double sink, side-by-side cabinets)
        for i, kind in size_disambiguable([raw[m] for m in members], ocfg["size_disambig_min_volume_ratio"]):
            word = {"largest": "larger", "smallest": "smaller"}[kind] if len(members) == 2 else kind
            out[members[i]].update(ref_kind=word, ref_name=f"{word} {c}", size_group=members)
    return out


def size_disambiguable(objs_of_cat: list[dict], ratio: float) -> list[tuple[int, str]]:
    """Objects among a category (>= 2 instances) that can be referred to as "the largest / smallest X": the largest is referable when largest / second-largest volume >= ratio;
    the smallest is referable when second-smallest / smallest volume >= ratio. Returns [(index, "largest"/"smallest")]."""
    if len(objs_of_cat) < 2:
        return []
    vol = np.array([float(np.prod(o["size"])) for o in objs_of_cat])
    order = np.argsort(-vol)
    out = []
    if vol[order[0]] >= ratio * vol[order[1]]:
        out.append((int(order[0]), "largest"))
    if vol[order[-2]] >= ratio * vol[order[-1]]:
        out.append((int(order[-1]), "smallest"))
    return out


def objects_scene(vid: str, force: bool = False):
    cfg = load_config()
    ocfg = cfg["objects"]
    wd = work_dir(cfg, "arkitscenes", vid)
    if is_done(wd, "objects") and not force:
        return read_json(wd / "objects.json")
    gate = read_json(wd / "gate_a.json")
    R_up = np.asarray(gate["R_up"])
    files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    mesh = arkit.load_mesh_legacy(files["mesh"])
    V = np.asarray(mesh.vertices) @ R_up.T
    mapping, allowed = load_vocab(cfg)
    drops = DropLog(wd, "step3_objects")

    raw = arkit.load_annotation(files["annotation"])
    # uniqueness counts include all annotated instances (including small instances filtered later)
    cats = [mapping.get(o["label"]) for o in raw]
    bl_objs, bl_scenes = load_blacklist(cfg)
    blacklisted = {k for (v, k) in bl_objs if v == vid}
    refs = assign_references(raw, cats, ocfg, blacklisted)
    cat_count = Counter(r["eff_cat"] for r in refs)
    boxes = [{"center": o["center"] @ R_up.T, "axes": o["axes"] @ R_up.T, "size": o["size"]} for o in raw]
    point_sets, n_shared, overlap_stats = object_point_sets(V, boxes, ocfg)
    out, pts = [], {}
    for obj_id, o in enumerate(raw):
        center, axes = boxes[obj_id]["center"], boxes[obj_id]["axes"]
        ref = refs[obj_id]
        cat = ref["eff_cat"]
        idx = point_sets[obj_id]
        yaw, size_ordered, axes_ordered = obb_yaw_and_size(axes, o["size"])
        reasons = filter_reasons(cat, ref["ref_kind"], allowed, len(idx), o["size"], ocfg)
        if obj_id in blacklisted:
            reasons.append("HUMAN_REJECTED")  # flagged label wrong / unannotated same-category object in GATE D
        for r in reasons:
            drops.add("object", obj_id, r, label=o["label"], category=cat, num_points=len(idx),
                      category_count=cat_count[cat], longest_edge_m=float(max(o["size"])))
        out.append({
            "obj_id": obj_id, "category": cat, "source_label": o["label"], "source_uid": o["uid"],
            "referable": not reasons, "drop_reasons": reasons,
            "ref_kind": ref["ref_kind"], "ref_name": ref["ref_name"], "size_group": ref["size_group"],
            "center": center.tolist(),
            "obb": {"center": center.tolist(), "size": size_ordered, "yaw_deg": yaw, "axes": axes_ordered.tolist()},
            "num_points": int(len(idx)), "num_points_shared_removed": n_shared[obj_id],
        })
        pts[f"obj_{obj_id}"] = idx.astype(np.int32)
    np.savez_compressed(wd / "object_points.npz", **pts)
    write_json(wd / "objects.json", out)
    n_ref = sum(o["referable"] for o in out)
    scene_ok = n_ref >= ocfg["min_referable_objects_per_scene"]
    scene_log = DropLog(wd, "step3_scene")
    if not scene_ok:
        scene_log.add("scene", vid, "TOO_FEW_OBJECTS", num_referable=n_ref)
    if vid in bl_scenes:
        scene_ok = False
        scene_log.add("scene", vid, "HUMAN_REJECTED", detail="whole scene dropped after human review")
    write_json(wd / "step3_summary.json", {"num_objects": len(out), "num_referable": n_ref, "scene_ok": scene_ok, **overlap_stats,
                                           "num_referable_by_size": sum(o["referable"] and o["ref_kind"] != "unique" for o in out),
                                           "num_blacklisted": len(blacklisted), "scene_excluded": vid in bl_scenes})
    mark_done(wd, "objects")
    return out


def vocab_stats(cfg):
    mapping, allowed = load_vocab(cfg)
    sel = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "selected_videos.csv", dtype={"video_id": str})
    root = path_of(cfg, "arkit_3dod_root")
    inst, scenes, uniq = Counter(), Counter(), Counter()
    per_scene_unique = []
    for vid in sel.video_id:
        cats = [mapping.get(o["label"], f"UNMAPPED:{o['label']}") for o in arkit.load_annotation(arkit.scene_files_3dod(root, vid)["annotation"])]
        c = Counter(cats)
        inst.update(cats)
        scenes.update(c.keys())
        u = [k for k, n in c.items() if n == 1 and allowed.get(k, False)]
        uniq.update(u)
        per_scene_unique.append(len(u))
    df = pd.DataFrame([{"category": k, "instances": inst[k], "scenes_present": scenes[k], "scenes_unique": uniq[k]}
                       for k in sorted(inst, key=lambda k: -inst[k])])
    df.to_csv(path_of(cfg, "vocab") / "category_frequency_arkitscenes.tsv", sep="\t", index=False)
    return df, np.array(per_scene_unique), len(sel)


def _referability_one(vid):
    """Read-only statistics for one scene: the number of referable objects after all Step 3 filters, and the number of extra objects referable through "largest/smallest X" disambiguation."""
    cfg = load_config()
    ocfg = cfg["objects"]
    mapping, allowed = load_vocab(cfg)
    files = arkit.scene_files_3dod(path_of(cfg, "arkit_3dod_root"), vid)
    V = np.asarray(arkit.load_mesh_legacy(files["mesh"]).vertices)  # mesh and annotations share the original frame, so counting points needs no up-axis transform
    raw = arkit.load_annotation(files["annotation"])
    cats = [mapping.get(o["label"]) for o in raw]
    point_sets, _, overlap_stats = object_point_sets(V, [{"center": o["center"], "axes": o["axes"], "size": o["size"]} for o in raw], ocfg)
    npts = [len(p) for p in point_sets]
    # referable count with category uniqueness only (no armchair / size references), and with all current rules
    count = Counter(cats)
    base = [not filter_reasons(c, "unique" if c is not None and count[c] == 1 else None, allowed, n, o["size"], ocfg)
            for c, n, o in zip(cats, npts, raw)]
    refs = assign_references(raw, cats, ocfg)
    full = [not filter_reasons(r["eff_cat"], r["ref_kind"], allowed, n, o["size"], ocfg) for r, n, o in zip(refs, npts, raw)]
    extra = sum(full) - sum(base)
    extra_cats = [r["ref_name"] for r, b, f in zip(refs, base, full) if f and not b]
    return {"video_id": vid, **overlap_stats, "referable": sum(base), "extra_size_disambig": extra, "extra_examples": "; ".join(sorted(extra_cats))}


def referability_stats(cfg, workers: int = 16):
    from multiprocessing import Pool
    sel = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "selected_videos.csv", dtype={"video_id": str})
    with Pool(workers) as pool:
        rows = pool.map(_referability_one, list(sel.video_id))
    df = pd.DataFrame(rows)
    diag = pd.read_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "orientation_diagnostic.csv", dtype={"video_id": str})
    df = df.merge(diag[["video_id", "sky_direction", "sky_auto"]], on="video_id", how="left")
    df["orientation_conflict"] = df.sky_direction != df.sky_auto
    df.to_csv(path_of(cfg, "work") / "arkitscenes" / "_step0" / "referability_stats.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vocab_stats", "objects", "referability_stats"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "vocab_stats":
        df, per_scene, n = vocab_stats(cfg)
        print(df.to_string(index=False))
        thr = cfg["objects"]["min_referable_objects_per_scene"]
        print(f"scenes={n} unique-category objects per scene: median={np.median(per_scene)} "
              f"<{thr}: {(per_scene < thr).sum()} ({(per_scene < thr).mean():.0%})")
        print("distribution:", dict(sorted(Counter(per_scene.tolist()).items())))
    elif args.cmd == "referability_stats":
        df = referability_stats(cfg)
        for k in (3, 4, 5):
            print(k, int((df.referable >= k).sum()), int(((df.referable + df.extra_size_disambig) >= k).sum()))
    else:
        scenes = [str(s) for s in (args.scenes or cfg["arkitscenes"]["pilot_scenes"])]
        for vid in scenes:
            objs = objects_scene(vid, args.force)
            print(vid, "objects", len(objs), "referable", sum(o["referable"] for o in objs),
                  [o["category"] for o in objs if o["referable"]])


if __name__ == "__main__":
    main()
