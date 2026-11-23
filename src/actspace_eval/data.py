"""Read the frozen data version (docs/evaluation_protocol.md).

- `FrozenData`: at start-up, verifies the SHA-256 of all data files and scene dependency files against freeze_manifest.json and refuses to run on mismatch; read-only.
  The official evaluation uses the merged version merged_v1.0 (version in configs/eval.yaml), version directory work/merged/_step8/<version>/;
  at start-up runs the verify of actspace_build.merge_build; scene outputs are read from work/<dataset>/<scene>/ according to the scene's dataset.
  Older single-dataset versions (arkit_only_v0.11 etc.) remain readable to reproduce earlier reports.
- `SceneData`: one scene's view graph (edges), poses and image paths — the public part the environment needs; plus the visibility matrix (private, only for privileged agents and scoring).
  Images: after upright rotation, downscaled proportionally (INTER_AREA) if the long side exceeds env.max_image_long_side (640) — ScanNet++ 1920×1440 → 640×480,
  the same size as ARKit; visibility and other annotations are still computed at the original resolution and are unaffected.

Local paths: configs/paths.yaml (copied from configs/paths.example.yaml, or set via the environment variable ACTSPACE_PATHS) gives the locations of raw data, intermediate outputs (work),
photos, models and results (apply_local). Path keys and model directories are always resolved to absolute paths; relative paths are relative to the repository root.
- Data: version directory <work>/merged/_step8/<version>/ and scene outputs <work>/<dataset>/<scene>/; verify checks hashes against freeze_manifest.json.
- Photos (paths.frames_root; unset by default = read from the raw datasets and rotate upright):
  - output of scripts/rebuild_frames.py, <frames_root>/<scene_id>/<frame_id>.png|.jpg (original resolution, upright): downscaled to max_image_long_side after reading;
  - or the exported evaluation JPEGs <frames_root>/<dataset>/<scene_id>/<frame_id>.jpg (upright, downscaled): these bytes are used as is (models.encode_jpeg).
- paths.model_root: each model directory becomes <model_root>/<original directory name>.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from actspace_build import freeze
from actspace_build.common import ROOT, load_config, path_of, read_json
from actspace_build.geometry import rotate_image_cw

DATASET = "arkitscenes"   # older single-dataset versions
DATASETS = ("arkitscenes", "scannetpp")
MOVE_ACTIONS = ("forward", "backward", "left", "right", "turn_left", "turn_right")
STATE_VISIBLE = 2


def load_eval_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs" / "eval.yaml") as f:
        ec = yaml.safe_load(f)
    return apply_local(ec) if path is None else ec


EVAL_PATH_KEYS = {"paths": ("data_root", "annotations_root", "frames_root", "model_root", "eval_runs"),
                  "open_models": ("results_root", "model_dir", "vllm_env", "hf_home", "cambrian_serve", "cambrian_python", "cambrian_repo")}


def _unshare(x):
    """YAML <<: merges make several models share one sub-dict (e.g. vllm); split them before overriding so changing one model does not change the others."""
    if isinstance(x, dict):
        return {k: _unshare(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unshare(v) for v in x]
    return x


def apply_local(ec: dict) -> dict:
    """Apply local paths (configs/paths.yaml, see configs/paths.example.yaml) to the evaluation config; afterwards path keys and model directories are resolved to absolute paths
    (relative paths are relative to the repository root). When paths.model_root is given, each model directory becomes <model_root>/<directory name>."""
    from actspace_build.common import abs_path, site_paths
    ec = _unshare(ec)
    sp = site_paths()
    P, om = ec.setdefault("paths", {}), ec.setdefault("open_models", {})
    if "annotations" in sp:
        P["annotations_root"] = str(sp["annotations"])
    if "frames" in sp:
        P["frames_root"] = str(sp["frames"])
    if "models" in sp:
        P["model_root"] = str(sp["models"])
    if "results" in sp:
        om["results_root"] = str(sp["results"])
        P["eval_runs"] = str(sp["results"] / "eval_runs")
    for k in ("hf_home", "vllm_env", "cambrian_python", "cambrian_repo"):
        if k in sp:
            om[k] = str(sp[k])
    for sec, keys in EVAL_PATH_KEYS.items():
        d = ec.get(sec) or {}
        for k in keys:
            if d.get(k):
                d[k] = str(abs_path(d[k]))
    root = P.get("model_root") or om.get("model_dir")
    for m in ec["models"].values():
        if isinstance(m, dict) and m.get("path"):
            m["path"] = str(Path(root) / Path(m["path"]).name) if P.get("model_root") else str(abs_path(m["path"]))
    return ec


def results_root() -> Path:
    """Results root (results in configs/paths.yaml, default results/ in the repository): open models <model>/<setting>/<run>/, API models api/<model>/…, figures _reports/."""
    return Path(load_eval_config()["open_models"]["results_root"])


class JpegImage(np.ndarray):
    """Image in frames/ mode: the decoded RGB array, carrying the original JPEG bytes (.jpeg). These bytes are sent to the model directly (models.encode_jpeg).
    New arrays from slicing, resizing etc. do not carry .jpeg (they are re-encoded), so only images passed through unchanged use the original bytes."""
    jpeg = b""


def jpeg_image(data: bytes) -> JpegImage:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decoding failed")
    a = bgr[:, :, ::-1].copy().view(JpegImage)
    a.jpeg = bytes(data)
    return a


class FreezeError(RuntimeError):
    pass


def resolve_step_scale(scale: dict | None, dataset: str) -> dict | None:
    """env.step_scale can be one scale ({move_m, turn_deg}, shared by all datasets) or given per dataset ({arkitscenes: {...}, scannetpp: {...}})."""
    if not scale or "move_m" in scale:
        return scale
    return scale[dataset]


def dataset_of_scene(scene_id: str) -> str:
    """The scene ID prefix determines the dataset: arkit_<video_id> / scannetpp_<scene_id>."""
    return "arkitscenes" if scene_id.startswith("arkit_") else "scannetpp"


class SceneData:
    def __init__(self, wd: Path, image_root: Path | None, max_long_side: int | None = None, frames_dir: Path | None = None,
                 frames_kind: str = "export"):
        self.wd = wd
        self.frames_dir = frames_dir
        self.frames_kind = frames_kind          # export (evaluation JPEGs) or rebuilt (output of rebuild_frames.py)
        self.max_long_side = max_long_side
        graph = read_json(wd / "view_graph.json")
        self.nodes = list(graph["nodes"])
        self.edges = {n: {a: graph["edges"][n].get(a) for a in MOVE_ACTIONS} for n in self.nodes}
        # frozen build layout: frames_all.json; annotation package: frames.json (keyframes, a superset of the graph nodes)
        ff = wd / "frames_all.json" if (wd / "frames_all.json").exists() else wd / "frames.json"
        frames = {f["frame_id"]: f for f in read_json(ff)}
        self.pose = {n: (np.asarray(frames[n]["position"], float), float(frames[n]["yaw_deg"])) for n in self.nodes}
        self._img = {n: ((image_root / frames[n]["rel_path"]) if image_root is not None else None, int(frames[n]["rotation_cw_quarters"])) for n in self.nodes}
        self._vis = None

    def image(self, node: str) -> np.ndarray:
        """Upright image (RGB uint8); downscaled proportionally when the long side exceeds max_long_side (ScanNet++ 1920×1440 → 640×480).
        When frames_dir holds exported evaluation JPEGs, read the JPEG (JpegImage, with original bytes); when it is the output of rebuild_frames.py, read the original-resolution upright image."""
        if self.frames_dir is not None:
            if self.frames_kind == "export":
                return jpeg_image((self.frames_dir / f"{node}.jpg").read_bytes())
            p = next((self.frames_dir / f"{node}{ext}" for ext in (".png", ".jpg") if (self.frames_dir / f"{node}{ext}").exists()),
                     self.frames_dir / f"{node}.png")
            bgr, q = cv2.imread(str(p)), 0
        else:
            p, q = self._img[node]
            bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        img = rotate_image_cw(bgr, q)
        h, w = img.shape[:2]
        if self.max_long_side and max(h, w) > self.max_long_side:
            k = self.max_long_side / max(h, w)
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        return img[:, :, ::-1].copy()

    # ---- private information below: only for the privileged agent (scripted_oracle) and scoring; never exposed by the environment
    def visible(self, obj_id: int, node: str) -> bool:
        if self._vis is None:
            v = np.load(self.wd / "visibility.npz")
            self._vis = ({int(o): k for k, o in enumerate(v["obj_ids"])}, {str(n): k for k, n in enumerate(v["frame_ids"])}, v["state"])
        row, col, st = self._vis
        return bool(st[row[obj_id], col[node]] == STATE_VISIBLE)


def group_from_annotations(gg: dict) -> dict:
    """Group record of the annotation package -> the frozen record used by the evaluation. Reverses the export conventions:
    unanswerable questions carry answer = cannot_determine and the geometric answer in geometric_answer / geometric_answer_raw;
    T1 answers are the unrounded distance (answer_raw, the value used for scoring); the package adds masked_group,
    in_main_table and qids. Not in the package: the group-level human_verified flag and the display-rounded T1 answer."""
    g = {k: v for k, v in gg.items() if k not in ("masked_group", "in_main_table", "qids", "questions")}
    qs = []
    for q in gg["questions"]:
        q = dict(q)
        geo = q.pop("geometric_answer", None)
        if q["evidence_label"] == "unanswerable":
            if "geometric_answer_raw" in q:
                q["answer_raw"] = q.pop("geometric_answer_raw")
            q["answer"] = float(geo) if q["answer_type"] == "numeric" else geo
        elif q["answer_type"] == "numeric":
            q["answer"] = float(q["answer"])
        qs.append(q)
    g["questions"] = qs
    return g


class FrozenData:
    """Read-only view of a frozen version. verify=True checks against the freeze manifest (on by default; unit tests may turn it off for speed)."""

    def __init__(self, ecfg: dict | None = None, verify: bool = True, source: str | None = None):
        """source: "frozen" (the frozen build layout under paths.work), "annotations" (the released annotation package,
        `annotations` in configs/paths.yaml) or None (annotations when configured, else frozen)."""
        self.ecfg = ecfg or load_eval_config()
        self.cfg = load_config(ROOT / self.ecfg["build_config"])
        paths = self.ecfg.get("paths") or {}
        self.data_root = paths.get("data_root")
        self.frames_root = Path(paths["frames_root"]) if paths.get("frames_root") else None
        self.source = source or ("annotations" if paths.get("annotations_root") else "frozen")
        if self.source == "annotations":
            self._init_annotations(Path(paths["annotations_root"]), verify)
            return
        if self.data_root:
            self.cfg["paths"]["work"] = str(self.data_root)
        self.version = self.ecfg["version"]
        self.merged = self.version.startswith("merged")
        if self.merged:
            from actspace_build.merge_dryrun import load_merge_config
            self.mcfg = load_merge_config()
            if self.data_root:
                self.mcfg["paths"]["work"] = str(self.data_root)
            if self.mcfg["merge"]["version"] != self.version:
                raise FreezeError(f"version in configs/merge.yaml ({self.mcfg['merge']['version']}) does not match eval.yaml ({self.version})")
            self.vdir = path_of(self.cfg, "work") / "merged" / "_step8" / self.version
        else:
            self.vdir = freeze.version_dir(self.cfg, self.version)
        self.manifest_path = mpath = self.vdir / "freeze_manifest.json"
        if not mpath.exists():
            raise FreezeError(f"{self.version} has no freeze_manifest.json; not a frozen version")
        self.manifest = read_json(mpath)
        self.groups = {}
        with open(self.vdir / "test_groups.jsonl") as f:
            for line in f:
                g = json.loads(line)
                self.groups[g["group_id"]] = g
        self.group_ids = list(self.groups)
        if verify:
            self.verify()
        self.image_roots = {"arkitscenes": path_of(self.cfg, "arkit_3dod_root").resolve().parents[1],
                            "scannetpp": (ROOT / self.cfg["scannetpp"]["root"]).resolve()}
        self.image_root = self.image_roots["arkitscenes"]
        self.max_long_side = self.ecfg["env"].get("max_image_long_side")

    def _init_annotations(self, pkg: Path, verify: bool):
        """Released annotation package (Hugging Face layout): data/groups/test.jsonl, scenes/<scene_id>/{scene_meta,objects,
        frames,view_graph}.json + visibility.npz, metadata/ (freeze manifest, tiny groups, eval subsets), SHA256SUMS."""
        self.pkg = pkg
        if not (pkg / "data" / "groups" / "test.jsonl").exists():
            raise FreezeError(f"{pkg} is not an ActSpace annotation package (data/groups/test.jsonl missing)")
        self.version = self.ecfg["version"]
        self.merged = self.version.startswith("merged")
        self.manifest_path = pkg / "metadata" / "freeze_manifest.json"
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("version") != self.version:
            raise FreezeError(f"annotation package version {self.manifest.get('version')} != configs/eval.yaml version {self.version}")
        self.vdir = pkg
        self.groups = {}
        with open(pkg / "data" / "groups" / "test.jsonl") as f:
            for line in f:
                g = group_from_annotations(json.loads(line))
                self.groups[g["group_id"]] = g
        self.group_ids = list(self.groups)
        if verify:
            self.verify()
        self.image_roots = {"arkitscenes": path_of(self.cfg, "arkit_3dod_root").resolve().parents[1],
                            "scannetpp": (ROOT / self.cfg["scannetpp"]["root"]).resolve()}
        self.image_root = self.image_roots["arkitscenes"]
        self.max_long_side = self.ecfg["env"].get("max_image_long_side")

    def verify_annotations(self):
        """Every file listed in SHA256SUMS must exist with that hash, every file of the package must be listed, and the
        package's tiny set and budget-infeasible groups must equal configs/eval.yaml."""
        import hashlib
        pkg = self.pkg
        want = dict(reversed(line.split("  ", 1)) for line in (pkg / "SHA256SUMS").read_text().splitlines() if line.strip())
        have = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                and not any(part.startswith(".") for part in p.relative_to(pkg).parts)}
        bad = sorted(n for n, h in want.items() if not (pkg / n).is_file() or hashlib.sha256((pkg / n).read_bytes()).hexdigest() != h)
        extra = sorted(have - set(want))
        tiny = (pkg / "metadata" / "tiny_groups.txt").read_text().split()
        subsets = read_json(pkg / "metadata" / "eval_subsets.json")
        if tiny != list(self.ecfg["llm"]["tiny_groups"]):
            bad.append("metadata/tiny_groups.txt differs from llm.tiny_groups")
        if sorted(subsets["budget_infeasible"]["groups"]) != sorted(self.ecfg["env"]["budget_infeasible"]):
            bad.append("metadata/eval_subsets.json budget_infeasible differs from env.budget_infeasible")
        print(f"verify annotation package {pkg.name}: {len(want)} files in SHA256SUMS; mismatches {len(bad)}; unlisted files {len(extra)}")
        for b in (bad + [f"unlisted {x}" for x in extra])[:50]:
            print("  MISMATCH", b)
        if bad or extra:
            raise FreezeError(f"annotation package {pkg} does not match its SHA256SUMS (see MISMATCH above)")

    def verify(self):
        if self.source == "annotations":
            return self.verify_annotations()
        if self.merged:   # at start-up run the merged_v1.0 verify (version directory, scene outputs, configs, source freeze manifests)
            from actspace_build.merge_build import verify_version
            if not verify_version(self.mcfg):
                raise FreezeError(f"{self.version} does not match freeze_manifest.json (see MISMATCH above)")
            return
        groups = list(self.groups.values())
        files, scenes, _ = freeze.hashes(self.cfg, self.version, groups)
        m = self.manifest
        bad = [k for k in m["files"] if files.get(k, {}).get("sha256") != m["files"][k]["sha256"]]
        bad += [f"{s}/{n}" for s, d in m["scene_files_sha256"].items() for n, h in d.items() if scenes.get(s, {}).get(n) != h]
        if bad or len(groups) != m["counts"]["groups"]:
            raise FreezeError(f"frozen data do not match freeze_manifest.json: {bad[:10]}")

    @lru_cache(maxsize=None)
    def scene(self, scene_id: str) -> SceneData:
        ds = dataset_of_scene(scene_id)
        if self.source == "annotations":
            wd = self.pkg / "scenes" / scene_id
        else:
            wd = path_of(self.cfg, "work") / ds / scene_id.split("_", 1)[1]
        if self.frames_root is not None:
            if (self.frames_root / scene_id).is_dir():          # output of rebuild_frames.py: <frames_root>/<scene_id>/
                return SceneData(wd, None, self.max_long_side, frames_dir=self.frames_root / scene_id, frames_kind="rebuilt")
            return SceneData(wd, None, self.max_long_side, frames_dir=self.frames_root / ds / scene_id)
        return SceneData(wd, self.image_roots[ds], self.max_long_side)

    def dataset(self, group_id: str) -> str:
        """Dataset of the question group (arkitscenes / scannetpp)."""
        return dataset_of_scene(self.groups[group_id]["scene_id"])

    def step_scale(self, group_id: str) -> dict | None:
        """Per-step scale in the active_v2 prompt: env.step_scale for the question group's dataset."""
        return resolve_step_scale(self.ecfg["env"].get("step_scale"), self.dataset(group_id))

    def cfg_work(self) -> Path:
        return path_of(self.cfg, "work") / DATASET

    def masked_nodes(self, group_id: str) -> set:
        return set((self.groups[group_id].get("view_graph_mask") or {}).get("masked_nodes", []))
