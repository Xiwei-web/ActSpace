"""ScanNet++ Step 3: object instances, unified categories, unique referability (see docs/data_construction.md), plus GATE B review material.

- Objects: each segGroup of segments_anno.json -> mesh vertices (segIndices of segments.json map vertices to segments; in ScanNet++
  object vertex sets do not overlap, so ARKit's shared-vertex handling is not needed). Box: scannetpp.object_obb (minimum-area
  bounding rectangle in xy + z extent).
- Filtering: points < objects.min_object_points -> UNRELIABLE_GEOMETRY; longest box edge < objects.min_longest_edge_m -> TOO_SMALL;
  category not allowed, not unique and no size reference, or a group annotation (one annotation covering several objects) -> NOT_REFERABLE;
  objects listed in scannetpp.blacklist_file (configs/object_blacklist_scannetpp.yaml) -> BLACKLISTED (still counted for uniqueness).
- Uniqueness counts include every annotated instance (including filtered small instances and group annotations); annotation-tool
  markers (SPLIT / REMOVE) are not objects and are not counted.
- Four category schemes (scannetpp.vocab_granularity selects the one written to the category / referable / ... fields of objects.json;
  all of them are stored under schemes):
  fine: unique by fine category; merged: unique after merging subclasses into parents; hybrid: specific names (fine category != parent
  and allowed) are unique by fine category, generic names by the whole parent class;
  merged_unique_fine_name (adopted at GATE B): uniqueness by merged class (both the merged class and the fine category must be allowed);
  objects are named by the merged class by default, and by the fine name only for fine categories marked name_ok in the vocabulary
  (monitor, copier, where the merged name would be a misnomer).
- Size references (larger / smaller): permanently disabled (scannetpp.size_reference_enabled: false); only category names that are
  unique in the scene are used. The code path is kept behind the switch (when enabled: exactly 2 instances with the same name, neither
  a group annotation, volume ratio >= 2.5, centres >= 1.0 m apart).

Usage: python -m actspace_build.scannetpp_step3 objects [--scenes ...] [--force]
       python -m actspace_build.scannetpp_step3 views [--scenes ...]     # report figures: best views, top-down views, box comparisons and examples
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2
import numpy as np
import open3d as o3d

from . import arkit
from . import geometry as g
from .common import DropLog, is_done, load_config, mark_done, path_of, read_json, write_json
from .scannetpp import object_obb, object_obb_pca, scale_intrinsics, scene_paths, sp_cfg, sp_work
from .scannetpp_vocab import load_display_names, load_mapping, load_name_ok, norm
from .step1_arkit import draw_obbs
from .step3_objects import filter_reasons, obb_yaw_and_size, size_disambiguable

SCHEMES = ("fine", "merged", "hybrid", "merged_unique_fine_name")


# ---------------------------------------------------------------- referring names
def scheme_keys(objs: list[dict], scheme: str, allowed: set, name_ok: set = frozenset()) -> list[tuple[str, tuple]]:
    """(referring name, count key) of each object under the scheme. hybrid: specific names count by fine category, generic names by
    parent class (including all subclass instances). merged_unique_fine_name: count key is the merged class; the name is the fine name
    (if in name_ok) or the merged class name."""
    out = []
    for o in objs:
        f, m = o["category_fine"], o["category_merged"]
        if scheme == "merged_unique_fine_name":
            out.append((f if f in name_ok else m, ("merged", m)))
        elif scheme == "fine":
            out.append((f, ("fine", f)))
        elif scheme == "merged":
            out.append((m, ("merged", m)))
        elif f != m and f in allowed:
            out.append((f, ("fine", f)))
        else:
            out.append((m, ("merged", m)))
    return out


def count_keys(objs: list[dict]) -> Counter:
    c = Counter()
    for o in objs:
        c[("fine", o["category_fine"])] += 1
        c[("merged", o["category_merged"])] += 1
    return c


def assign_references(objs: list[dict], scheme: str, allowed: set, ocfg: dict, size_refs: bool = False,
                      name_ok: set = frozenset()) -> list[dict]:
    """Return {name, ref_kind, ref_name, size_group} for each object; ref_kind None means the object cannot be referred to uniquely by name.
    objs need category_fine / category_merged / kind / center / size (size is used for the volume).
    With size_refs=False (size references are permanently disabled) only unique references are produced."""
    keys = scheme_keys(objs, scheme, allowed, name_ok)
    cnt = count_keys(objs)
    out = []
    for k, (o, (name, key)) in enumerate(zip(objs, keys)):
        uniq = cnt[key] == 1 and o["kind"] == "object" and name in allowed
        if scheme == "merged_unique_fine_name":
            uniq = uniq and o["category_fine"] in allowed and o["category_merged"] in allowed
        out.append({"name": name, "ref_kind": "unique" if uniq else None, "ref_name": name if uniq else None, "size_group": [k]})
    if not size_refs:
        return out
    by_key = {}
    for k, (_, key) in enumerate(keys):
        by_key.setdefault(key, []).append(k)
    for key, members in sorted(by_key.items()):
        if cnt[key] != 2 or len(members) != 2:
            continue  # exactly 2 instances under the count key, both with the same name (not used for a hybrid generic group containing a specific-name instance)
        names = {keys[m][0] for m in members}
        if len(names) != 1 or any(objs[m]["kind"] != "object" for m in members):
            continue
        name = names.pop()
        if name not in allowed:
            continue
        a, b = (np.asarray(objs[m]["center"]) for m in members)
        if np.linalg.norm(a - b) < ocfg["size_disambig_min_center_dist_m"]:
            continue
        for i, kind in size_disambiguable([objs[m] for m in members], ocfg["size_disambig_min_volume_ratio"]):
            word = {"largest": "larger", "smallest": "smaller"}[kind]
            out[members[i]].update(ref_kind="size", ref_word=word, ref_name=f"{word} {name}", size_group=members)
    return out


# ---------------------------------------------------------------- objects
def load_blacklist(cfg: dict, sid: str) -> dict[int, str]:
    """Objects of this scene in the ScanNet++ blacklist file (scannetpp.blacklist_file, separate from ARKit's): obj_id -> source_label
    (used for cross-checking). The switch is the shared objects.blacklist_enabled."""
    import yaml
    from .common import ROOT
    p = ROOT / sp_cfg(cfg)["blacklist_file"]
    if not cfg["objects"]["blacklist_enabled"] or not p.exists():
        return {}
    d = yaml.safe_load(open(p)) or {}
    return {int(o["obj_id"]): o["source_label"] for o in d.get("objects") or [] if str(o["scene_id"]) == sid}


def load_mesh(cfg: dict, sid: str):
    sc = scene_paths(cfg, sid)
    mesh = o3d.io.read_triangle_mesh(str(sc.scan_mesh_path))
    return sc, np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def raw_objects(cfg: dict, sid: str, V: np.ndarray, sc) -> tuple[list[dict], dict]:
    """Objects of segments_anno (SPLIT / REMOVE markers removed) with their vertex indices and both box types."""
    mapping, _ = load_mapping(cfg)
    seg = np.asarray(read_json(sc.scan_mesh_segs_path)["segIndices"])
    identity = len(seg) == len(V) and bool(np.all(seg == np.arange(len(seg))))
    if not identity:
        order = np.argsort(seg, kind="stable")
        starts = np.searchsorted(seg[order], np.unique(seg))
        groups = dict(zip(np.unique(seg).tolist(), np.split(order, starts[1:])))
    objs, n_art = [], Counter()
    for gobj in read_json(sc.scan_anno_json_path)["segGroups"]:
        lab = gobj["label"]
        m = mapping.get(lab) or {"kind": "object", "category_fine": norm(lab), "category_merged": norm(lab)}
        if m["kind"] == "artifact":
            n_art[lab] += 1
            continue
        segs = np.asarray(gobj["segments"], dtype=np.int64)
        vids = np.unique(segs) if identity else np.unique(np.concatenate([groups.get(int(s), np.empty(0, np.int64)) for s in segs]))
        o = {"source_object_id": int(gobj["objectId"]), "source_label": lab, "kind": m["kind"],
             "category_fine": m["category_fine"], "category_merged": m["category_merged"], "vertex_ids": vids}
        if len(vids):
            box = object_obb(V[vids])
            pca = object_obb_pca(V[vids])
        else:
            box = pca = {"center": np.zeros(3), "axes": np.eye(3), "size": np.zeros(3)}
        o.update(center=box["center"], axes=box["axes"], size=box["size"], pca=pca)
        objs.append(o)
    return objs, {"segindices_identity": identity, "artifacts": dict(n_art)}


def objects_scene(sid: str, force: bool = False) -> dict:
    cfg = load_config()
    sp = sp_cfg(cfg)
    ocfg = cfg["objects"]
    wd = sp_work(cfg) / sid
    if is_done(wd, "objects") and not force:
        return read_json(wd / "step3_summary.json")
    _, allowed = load_mapping(cfg)
    name_ok = load_name_ok(cfg)
    sc, V, _ = load_mesh(cfg, sid)
    objs, meta = raw_objects(cfg, sid, V, sc)
    refs = {s: assign_references(objs, s, allowed, ocfg, sp["size_reference_enabled"], name_ok) for s in SCHEMES}
    display = load_display_names(cfg)  # names are singular (blinds -> window blind)
    for rs in refs.values():
        for r in rs:
            if r["ref_name"] in display:
                r["ref_name"] = display[r["ref_name"]]
    allowed_d = {c: True for c in allowed}
    black = load_blacklist(cfg, sid)
    for k, lab in black.items():
        if objs[k]["source_label"] != lab:
            raise SystemExit(f"{sid}: blacklisted obj {k} has label {lab!r} but the current annotation is {objs[k]['source_label']!r} (object ids may have changed)")
    chosen = sp["vocab_granularity"]
    drops = DropLog(wd, "step3_objects")
    out, pts = [], {}
    summary_reasons = {s: Counter() for s in SCHEMES}
    for obj_id, o in enumerate(objs):
        yaw, size_o, axes_o = obb_yaw_and_size(np.asarray(o["axes"]), np.asarray(o["size"]))
        pyaw, psize, paxes = obb_yaw_and_size(np.asarray(o["pca"]["axes"]), np.asarray(o["pca"]["size"]))
        sch = {}
        small_sofa = (o["category_merged"] == "sofa" and sp.get("small_sofa_max_longest_m") is not None
                      and float(max(o["size"])) < sp["small_sofa_max_longest_m"])
        for s in SCHEMES:
            r = refs[s][obj_id]
            reasons = filter_reasons(r["name"], r["ref_kind"], allowed_d, len(o["vertex_ids"]), o["size"], ocfg)
            if o["kind"] == "group" and "NOT_REFERABLE" not in reasons:
                reasons.insert(0, "NOT_REFERABLE")
            if obj_id in black:
                reasons.append("BLACKLISTED")
            if small_sofa and "NOT_REFERABLE" not in reasons:  # a sofa whose longest edge is below the threshold (single armchair) is not referable but still counts for sofa uniqueness
                reasons.append("NOT_REFERABLE")
            summary_reasons[s].update(reasons)
            sch[s] = {"category": r["name"], "referable": not reasons, "drop_reasons": reasons, "ref_kind": r["ref_kind"],
                      "ref_name": r["ref_name"], "size_group": r["size_group"]}
        c = sch[chosen]
        for reason in c["drop_reasons"]:
            extra = {"detail": "small_sofa"} if small_sofa and reason == "NOT_REFERABLE" else {}
            drops.add("object", obj_id, reason, label=o["source_label"], category=c["category"], annotation_kind=o["kind"], **extra,
                      num_points=len(o["vertex_ids"]), longest_edge_m=float(max(o["size"])))
        out.append({"obj_id": obj_id, "category": c["category"], "source_label": o["source_label"], "source_object_id": o["source_object_id"],
                    "kind": o["kind"], "category_fine": o["category_fine"], "category_merged": o["category_merged"],
                    "referable": c["referable"], "drop_reasons": c["drop_reasons"], "ref_kind": c["ref_kind"], "ref_name": c["ref_name"],
                    "size_group": c["size_group"], "blacklisted": obj_id in black, "center": np.asarray(o["center"]).tolist(),
                    "obb": {"center": np.asarray(o["center"]).tolist(), "size": size_o, "yaw_deg": yaw, "axes": axes_o.tolist()},
                    "obb_pca": {"center": np.asarray(o["pca"]["center"]).tolist(), "size": psize, "yaw_deg": pyaw, "axes": paxes.tolist()},
                    "num_points": int(len(o["vertex_ids"])), "schemes": sch})
        pts[f"obj_{obj_id}"] = o["vertex_ids"].astype(np.int32)
    np.savez_compressed(wd / "object_points.npz", **pts)
    write_json(wd / "objects.json", out)
    n_ref = {s: sum(x["schemes"][s]["referable"] for x in out) for s in SCHEMES}
    scene_ok = n_ref[chosen] >= ocfg["min_referable_objects_per_scene"]
    scene_log = DropLog(wd, "step3_scene")
    if not scene_ok:
        scene_log.add("scene", sid, "TOO_FEW_OBJECTS", num_referable=n_ref[chosen])
    summary = {"scene_id": sid, "vocab_granularity": chosen, "num_objects": len(out),
               "num_groups": sum(x["kind"] == "group" for x in out), "num_blacklisted": len(black), "num_referable": n_ref[chosen], "num_referable_by_scheme": n_ref,
               "num_referable_by_size": {s: sum(x["schemes"][s]["referable"] and x["schemes"][s]["ref_kind"] == "size" for x in out) for s in SCHEMES},
               "drop_reasons_by_scheme": {s: dict(v) for s, v in summary_reasons.items()},
               "scene_ok": scene_ok, "scene_ok_by_scheme": {s: n_ref[s] >= ocfg["min_referable_objects_per_scene"] for s in SCHEMES}, **meta}
    write_json(wd / "step3_summary.json", summary)
    mark_done(wd, "objects")
    return summary


# ---------------------------------------------------------------- report figures
def _box(o: dict, key: str = "obb") -> dict:
    b = o[key]
    return {"center": np.asarray(b["center"]), "axes": np.asarray(b["axes"]), "size": np.asarray(b["size"]), "label": o["source_label"]}


def best_views(cfg: dict, sid: str, objs: list[dict], pts: dict, V: np.ndarray, F: np.ndarray, n_sample: int = 300) -> list[dict]:
    """Best keyframe view of each object: the keyframe with the largest fraction of sampled vertices visible (inside the image and not
    occluded by the mesh); ties go to the larger projected area. Only used for report figures (Step 4 computes visibility separately)."""
    sp = sp_cfg(cfg)
    wd = sp_work(cfg) / sid
    frames = {f["frame_id"]: f for f in read_json(wd / "frames_all.json")}
    kfs = [frames[i] for i in read_json(wd / "keyframes.json")["keyframe_ids"]]
    W, H = sp["best_view_render_size"]
    rc = arkit.raycasting_scene(V, F)
    tol = cfg["visibility"]["occlusion_tolerance_m"]
    rng = np.random.default_rng(0)
    samples = []
    for o in objs:
        v = pts[f"obj_{o['obj_id']}"]
        samples.append(V[v if len(v) <= n_sample else rng.choice(v, n_sample, replace=False)] if len(v) else np.zeros((0, 3)))
    best = [{"frame_id": None, "visible_frac": 0.0, "area": 0.0} for _ in objs]
    for f in kfs:
        K = scale_intrinsics(np.asarray(f["K"]), f["image_size"], (W, H))
        T = np.asarray(f["T_wc"])
        d = g.render_depth(rc, K, T, W, H)
        for k, P in enumerate(samples):
            if not len(P):
                continue
            uv, z = g.project(P, K, T)
            ins = g.in_image(uv, z, W, H)
            if not ins.any():
                continue
            ui = np.clip(np.round(uv[ins]).astype(int), 0, [W - 1, H - 1])
            vis = z[ins] <= d[ui[:, 1], ui[:, 0]] + tol
            frac = float(vis.sum() / len(P))
            area = float(np.ptp(uv[ins][:, 0]) * np.ptp(uv[ins][:, 1])) if ins.sum() > 1 else 0.0
            b = best[k]
            if frac > b["visible_frac"] + 1e-9 or (abs(frac - b["visible_frac"]) <= 1e-9 and area > b["area"]):
                best[k] = {"frame_id": f["frame_id"], "visible_frac": frac, "area": area}
    return best


def render_object(cfg: dict, sid: str, frame: dict, boxes: list[tuple[dict, tuple]], points=None, out_w: int = 960,
                  crop: bool = False, title: str = "") -> np.ndarray:
    """Draw boxes on the original keyframe (boxes: [(box, BGR colour)]), optionally the object points (small magenta dots).
    crop=True crops to the area around the projected boxes."""
    from .scannetpp import sp_root
    img = cv2.imread(str(sp_root(cfg) / frame["rel_path"]))
    W0, H0 = frame["image_size"]
    s = out_w / W0
    img = cv2.resize(img, (out_w, int(round(H0 * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    K = scale_intrinsics(np.asarray(frame["K"]), (W0, H0), (W, H))
    T = np.asarray(frame["T_wc"])
    if points is not None and len(points):
        uv, z = g.project(points, K, T)
        ok = g.in_image(uv, z, W, H)
        for p in np.round(uv[ok]).astype(int):
            cv2.circle(img, tuple(p), 1, (255, 0, 255), -1)
    for b, color in boxes:
        draw_obbs(img, [b], K, T, np.eye(3), color=color)
    if crop:
        from .step1_arkit import obb_corners
        C = np.concatenate([obb_corners(b) for b, _ in boxes])
        uv, z = g.project(C, K, T)
        uv = uv[z > 0.05]
        if len(uv):
            x0, y0 = np.clip(uv.min(0) - 40, 0, [W, H]).astype(int)
            x1, y1 = np.clip(uv.max(0) + 40, 0, [W, H]).astype(int)
            if x1 - x0 > 20 and y1 - y0 > 20:
                img = img[y0:y1, x0:x1]
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(img, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def topdown(cfg: dict, sid: str, objs: list[dict], pts: dict, V: np.ndarray, out_path, px: int = 1400) -> None:
    """Top-down view: each object's point set in its own colour, unlabelled vertices light grey, banned categories (wall, floor,
    ceiling, ...) dark grey; allowed objects get the base of the new box and a label (bold if referable)."""
    rng = np.random.default_rng(0)
    xy = V[:, :2]
    lo, hi = xy.min(0) - 0.2, xy.max(0) + 0.2
    s = (px - 40) / float(max(hi - lo))
    Wd, Hd = int((hi - lo)[0] * s) + 40, int((hi - lo)[1] * s) + 40
    to_px = lambda p: np.stack([20 + (p[:, 0] - lo[0]) * s, Hd - 20 - (p[:, 1] - lo[1]) * s], 1).round().astype(int)
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    labelled = np.zeros(len(V), bool)
    for o in objs:
        labelled[pts[f"obj_{o['obj_id']}"]] = True
    bg = np.flatnonzero(~labelled)
    bg = bg if len(bg) <= 150000 else rng.choice(bg, 150000, replace=False)
    for p in to_px(xy[bg]):
        img[p[1], p[0]] = (225, 225, 225)
    order = sorted(objs, key=lambda o: -max(o["obb"]["size"][:2]))  # draw large objects first so small ones are on top
    struct = {"wall", "floor", "ceiling", "doorframe", "window frame", "windowsill", "shower wall", "pipe", "electrical duct", "structure"}
    for o in order:
        v = pts[f"obj_{o['obj_id']}"]
        v = v if len(v) <= 20000 else rng.choice(v, 20000, replace=False)
        color = (150, 150, 150) if o["category_fine"] in struct or o["category_merged"] in {"object"} else tuple(int(c) for c in rng.integers(30, 220, 3))
        P = to_px(xy[v])
        img[np.clip(P[:, 1], 0, Hd - 1), np.clip(P[:, 0], 0, Wd - 1)] = color
    from .step1_arkit import obb_corners
    for o in objs:
        if o["category_fine"] in struct or o["kind"] == "artifact":
            continue
        sch = o["schemes"][sp_cfg(cfg)["vocab_granularity"]]
        if sch["category"] is None:
            continue
        C = obb_corners(_box(o))[:4]
        P = to_px(C[:, :2])
        cv2.polylines(img, [P.reshape(-1, 1, 2)], True, (0, 0, 0), 2 if sch["referable"] else 1, cv2.LINE_AA)
        if sch["referable"]:
            c = to_px(np.asarray(o["center"])[None, :2])[0]
            cv2.putText(img, sch["ref_name"], (int(c[0]) + 3, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"{sid}: object point sets (colour per object; light grey = unlabelled, dark grey = structure/object); "
                     f"boxes = new OBB; red text = referable ({sp_cfg(cfg)['vocab_granularity']})",
                (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def robust_longest(P: np.ndarray, axes: np.ndarray, pct: float) -> float:
    """Longest horizontal edge after trimming pct% of the points at both ends of each box axis (to detect stray outliers)."""
    loc = P @ np.asarray(axes).T
    lo, hi = np.percentile(loc, pct, axis=0), np.percentile(loc, 100 - pct, axis=0)
    return float(max((hi - lo)[:2]))


def views_scene(sid: str) -> dict:
    """Report preparation: best view and outlier metric of each object; top-down view. Writes work/scannetpp/<sid>/step3_views.json."""
    cfg = load_config()
    sp = sp_cfg(cfg)
    wd = sp_work(cfg) / sid
    objs = read_json(wd / "objects.json")
    pts = dict(np.load(wd / "object_points.npz"))
    _, V, F = load_mesh(cfg, sid)
    bv = best_views(cfg, sid, objs, pts, V, F)
    rows = []
    for o, b in zip(objs, bv):
        P = V[pts[f"obj_{o['obj_id']}"]]
        rl = robust_longest(P, o["obb"]["axes"], sp["outlier_robust_pct"]) if len(P) >= 20 else None
        rows.append({"obj_id": o["obj_id"], **b, "robust_longest_m": rl})
    write_json(wd / "step3_views.json", rows)
    topdown(cfg, sid, objs, pts, V, path_of(cfg, "reports") / "scannetpp_step3" / f"{sid}_topdown.jpg")
    return {"scene_id": sid, "objects": len(objs), "with_view": sum(r["frame_id"] is not None for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["objects", "views"])
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    scenes = args.scenes or sp_cfg(cfg)["pilot_scenes"]
    from multiprocessing import Pool
    with Pool(min(args.workers, len(scenes))) as pool:
        for r in pool.imap(_run, [(args.cmd, s, args.force) for s in scenes]):
            print({k: v for k, v in r.items() if k not in ("drop_reasons_by_scheme",)}, flush=True)


def _run(a):
    cmd, sid, force = a
    return objects_scene(sid, force) if cmd == "objects" else views_scene(sid)


if __name__ == "__main__":
    main()
