"""Configuration loading, deterministic seeds, output I/O and drop logging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "build.yaml"


PATHS_ENV = "ACTSPACE_PATHS"
# Keys available in configs/paths.yaml (see configs/paths.example.yaml). Relative paths are relative to the directory of paths.yaml; ~ and $VAR are supported.
SITE_KEYS = ("data_raw", "work", "annotations", "release", "sandbox", "frames", "models", "results", "hf_home", "vllm_env",
             "cambrian_python", "cambrian_repo", "scannetpp_toolbox", "arkitscenes_repo", "logs")


def paths_file() -> Path:
    """Machine-specific path settings: the file named by the ACTSPACE_PATHS environment variable, default configs/paths.yaml (not in the repository; copy it from configs/paths.example.yaml)."""
    import os
    return Path(os.environ.get(PATHS_ENV) or ROOT / "configs" / "paths.yaml")


def site_paths() -> dict[str, Path]:
    """Read paths.yaml -> {key: absolute path}. Returns {} if the file does not exist (relative defaults in the configs are then relative to the repository root)."""
    import os
    p = paths_file()
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text()) or {}
    d = d.get("paths", d)
    unknown = sorted(set(d) - set(SITE_KEYS))
    if unknown:
        raise ValueError(f"{p}: unknown keys {unknown}; allowed: {list(SITE_KEYS)}")
    out = {}
    for k, v in d.items():
        if v in (None, ""):
            continue
        q = Path(os.path.expandvars(os.path.expanduser(str(v))))
        out[k] = q if q.is_absolute() else (p.parent / q).resolve()
    return out


def apply_site_paths_build(cfg: dict) -> dict:
    """Apply paths.yaml to the build config: data_raw prefix, work, logs, ScanNet++ toolbox, official ARKitScenes repository."""
    sp = site_paths()
    if "data_raw" in sp:
        def swap(v):
            if isinstance(v, str) and (v == "data_raw" or v.startswith("data_raw/")):
                return str(sp["data_raw"] / v[len("data_raw/"):]) if v != "data_raw" else str(sp["data_raw"])
            return v
        cfg["paths"] = {k: swap(v) for k, v in cfg["paths"].items()}
        if "scannetpp" in cfg:
            cfg["scannetpp"]["root"] = swap(cfg["scannetpp"]["root"])
    if "work" in sp:
        cfg["paths"]["work"] = str(sp["work"])
    if "arkitscenes_repo" in sp:
        cfg["paths"]["arkit_repo"] = str(sp["arkitscenes_repo"])
    if "scannetpp" in cfg:
        if "scannetpp_toolbox" in sp:
            cfg["scannetpp"]["toolbox"] = str(sp["scannetpp_toolbox"])
        if "logs" in sp:
            cfg["scannetpp"]["logs"] = str(sp["logs"] / "scannetpp")
    return cfg


# Hashed YAML configuration files may differ from the frozen bytes only in comments, formatting and the keys below, which hold
# machine-specific paths. configs/frozen_config_fingerprints.json stores, for each hashed YAML file, the SHA-256 at freeze time
# and a fingerprint of its parsed content without these keys, so verification accepts such differences and nothing else.
SITE_CONFIG_KEYS = {"configs/build.yaml": (("paths", "work"), ("paths", "arkit_repo"), ("scannetpp", "toolbox"), ("scannetpp", "logs")),
                    "configs/merge.yaml": (("merge", "sandbox_root"), ("merge", "release_root"))}
FROZEN_FINGERPRINTS = ROOT / "configs" / "frozen_config_fingerprints.json"


def config_fingerprint(rel: str, text: str) -> str:
    """SHA-256 of the parsed YAML without the machine-specific path keys (comments and formatting ignored)."""
    import copy
    d = copy.deepcopy(yaml.safe_load(text))
    for keys in SITE_CONFIG_KEYS.get(rel, ()):
        node = d
        for k in keys[:-1]:
            node = node.get(k) or {}
        node.pop(keys[-1], None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def config_matches_frozen(rel: str, frozen_sha256: str) -> bool:
    """True if the YAML config differs from the frozen file (hash frozen_sha256) only in comments, formatting or SITE_CONFIG_KEYS."""
    if not rel.endswith(".yaml") or not FROZEN_FINGERPRINTS.exists():
        return False
    refs = json.loads(FROZEN_FINGERPRINTS.read_text()).get(rel, [])
    fp = config_fingerprint(rel, (ROOT / rel).read_text())
    return any(r["sha256"] == frozen_sha256 and r["fingerprint"] == fp for r in refs)


def load_config(path: str | Path | None = None) -> dict:
    """Read configs/build.yaml by default; the ACTSPACE_CONFIG environment variable can name another config (for comparison experiments that add rules one at a time).
    Machine-specific paths (data, intermediate outputs, etc.) are overridden by configs/paths.yaml (apply_site_paths_build)."""
    import os
    path = path or os.environ.get("ACTSPACE_CONFIG") or DEFAULT_CONFIG
    with open(path) as f:
        return apply_site_paths_build(yaml.safe_load(f))


def abs_path(v) -> Path:
    """A path from the config: relative paths are relative to the repository root."""
    q = Path(v)
    return q if q.is_absolute() else ROOT / q


def path_of(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def scene_seed(cfg: dict, scene_id: str) -> int:
    """Derive a scene-level seed from hash(seed, scene_id); stable across processes."""
    h = hashlib.sha256(f"{cfg['seed']}:{scene_id}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def scene_rng(cfg: dict, scene_id: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng([scene_seed(cfg, scene_id), int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:4], "little")])


def work_dir(cfg: dict, dataset: str, scene_id: str) -> Path:
    d = path_of(cfg, "work") / dataset / scene_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_to_jsonable)
    tmp.replace(path)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def assert_not_frozen(version_dir: Path) -> None:
    """A frozen data version (its directory has freeze_manifest.json) is never modified again; see freeze.py."""
    if (version_dir / "freeze_manifest.json").exists():
        raise SystemExit(f"{version_dir} is frozen (freeze_manifest.json) and must not be modified; use a new version name in the config")


def is_done(wd: Path, step: str) -> bool:
    return (wd / f".done_{step}").exists()


def mark_done(wd: Path, step: str) -> None:
    (wd / f".done_{step}").touch()


class DropLog:
    """Write filtered frames/objects/questions to dropped.jsonl (with a reason_code from the reason-code table).

    Rerunning a step first clears that step's old records, so the log is idempotent.
    """

    def __init__(self, wd: Path, step: str):
        self.path = wd / "dropped.jsonl"
        self.step = step
        kept = []
        if self.path.exists():
            with open(self.path) as f:
                kept = [l for l in f if json.loads(l).get("step") != step]
        with open(self.path, "w") as f:
            f.writelines(kept)

    def add(self, kind: str, item_id, reason_code: str, **detail) -> None:
        rec = {"step": self.step, "kind": kind, "id": item_id, "reason_code": reason_code, **detail}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_to_jsonable) + "\n")
