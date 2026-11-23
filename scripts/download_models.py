"""Download open-model weights and verify them against SHA-256 manifests.

    python scripts/download_models.py [--models-yaml docs/models/models.yaml] [--models A B …] [--verify-only] [--workers 8]

- Downloads hf_repo at revision (a pinned commit) from models.yaml to <models>/<local_dir> (models in configs/paths.yaml, default
  models/ in the repository), then checks the SHA-256 of every file against models/manifests/<model>.sha256 (excluding .cache/):
  missing files, extra files and hash mismatches are all errors.
- Repositories in hf_cache_extra are downloaded into HF_HOME (open_models.hf_home): SenseNova's remote code, Cambrian-S's vision
  tower. The evaluation runs offline.
- After successful verification the manifest is copied to <results>/_manifests/; during evaluation env.json records the total
  weights hash (weights_total_sha256).
- Gated repositories (RoboBrain2.0-7B): accept the license on the Hugging Face website first, then `read -s HF_TOKEN && export HF_TOKEN` (read from the
  environment only; do not put it in files or command-line arguments).
- Without --models, all models in models.yaml are downloaded.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actspace_eval.data import load_eval_config  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def verify(d: Path, manifest: Path, workers: int) -> dict:
    want = {}
    for line in manifest.read_text().splitlines():
        h, name = line.split("  ", 1)
        want[name] = h
    have = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and ".cache" not in p.relative_to(d).parts)
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    with ThreadPoolExecutor(workers) as ex:
        got = dict(zip(sorted(set(want) & set(have)), ex.map(lambda n: sha256(d / n), sorted(set(want) & set(have)))))
    bad = sorted(n for n, h in got.items() if h != want[n])
    return {"files": len(want), "missing": missing, "extra": extra, "sha256_mismatch": bad, "ok": not (missing or extra or bad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-yaml", default=str(Path(__file__).resolve().parents[1] / "docs" / "models" / "models.yaml"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ec = load_eval_config()
    model_root = ec["paths"].get("model_root") or ec["open_models"]["model_dir"]
    hf_home = ec["open_models"].get("hf_home")
    results = Path(ec["open_models"]["results_root"])
    if not model_root:
        raise SystemExit("no model directory: set models in configs/paths.yaml")
    my = Path(a.models_yaml)
    doc = yaml.safe_load(my.read_text())["models"]
    names = a.models or list(doc)
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download     # reads HF_TOKEN from the environment (if set); no token is passed here
    report = {}
    for k in names:
        m = doc[k]
        d = Path(model_root) / m["local_dir"]
        if not a.verify_only:
            print(f"downloading {k}: {m['hf_repo']}@{m['revision']} -> {d}", flush=True)
            snapshot_download(m["hf_repo"], revision=m["revision"], local_dir=str(d), max_workers=a.workers)
            for x in m.get("hf_cache_extra") or []:
                print(f"  also into HF_HOME: {x['repo']}@{x['revision']}", flush=True)
                snapshot_download(x["repo"], revision=x["revision"], allow_patterns=x.get("allow_patterns"))
        r = verify(d, my.parent / m["weights_sha256_manifest"], a.workers)
        report[k] = r
        print(f"{k}: {'OK' if r['ok'] else 'FAILED'} ({r['files']} files; missing {len(r['missing'])}, extra {len(r['extra'])}, hash mismatches {len(r['sha256_mismatch'])})", flush=True)
        if r["ok"]:
            (results / "_manifests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(my.parent / m["weights_sha256_manifest"], results / "_manifests" / Path(m["weights_sha256_manifest"]).name)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if all(r["ok"] for r in report.values()) else 1)


if __name__ == "__main__":
    main()
