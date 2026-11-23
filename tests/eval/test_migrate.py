"""Local paths (configs/paths.yaml), photo directories (exported JPEGs and rebuild_frames.py output), truncation statistics of
result snapshots, the local run queue."""
import json

import numpy as np
import pytest


def test_paths_yaml_applies_to_eval_build_and_merge_configs(tmp_path, monkeypatch):
    from actspace_build.common import ROOT, load_config
    from actspace_build.merge_dryrun import load_merge_config
    from actspace_eval.data import load_eval_config
    from actspace_eval.providers import registry as R
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg/paths.yaml").write_text("paths:\n  data_raw: ../raw\n  work: ../work\n  frames: /abs/frames\n  models: ../w\n"
                                             "  results: ../results\n  release: ../rel\n")
    monkeypatch.setenv("ACTSPACE_PATHS", str(tmp_path / "cfg/paths.yaml"))
    monkeypatch.delenv("API_OUT_ROOT", raising=False)
    ec = load_eval_config()
    assert ec["paths"]["frames_root"] == "/abs/frames"
    assert ec["open_models"]["results_root"] == str((tmp_path / "results").resolve())
    assert ec["paths"]["eval_runs"] == str((tmp_path / "results" / "eval_runs").resolve())
    assert ec["models"]["Qwen3.5-2B"]["path"] == str((tmp_path / "w").resolve() / "Qwen3.5-2B")
    assert ec["models"]["Qwen3.5-9B"]["vllm"]["max_model_len"] == 65536
    cfg = load_config()
    assert cfg["paths"]["work"] == str((tmp_path / "work").resolve())
    assert cfg["paths"]["arkit_3dod_root"] == str((tmp_path / "raw").resolve() / "arkitscenes/3dod")
    assert cfg["scannetpp"]["root"] == str((tmp_path / "raw").resolve() / "scannetpp")
    assert load_merge_config()["merge"]["release_root"] == str((tmp_path / "rel").resolve())
    assert R.results_root(R.load_registry()) == (tmp_path / "results").resolve() / "api"
    monkeypatch.setenv("ACTSPACE_PATHS", str(tmp_path / "missing.yaml"))      # no paths.yaml: relative defaults are relative to the repository root
    ec = load_eval_config()
    assert ec["open_models"]["results_root"] == str(ROOT / "results") and ec["models"]["Qwen3.5-2B"]["path"] == str(ROOT / "models" / "Qwen3.5-2B")


def test_paths_yaml_rejects_unknown_keys(tmp_path, monkeypatch):
    from actspace_build.common import site_paths
    (tmp_path / "paths.yaml").write_text("paths:\n  wrok: /x\n")
    monkeypatch.setenv("ACTSPACE_PATHS", str(tmp_path / "paths.yaml"))
    with pytest.raises(ValueError):
        site_paths()


def test_jpeg_bytes_pass_through_unchanged():
    import cv2
    from actspace_eval.data import jpeg_image
    from actspace_eval.models import encode_jpeg
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    b = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()    # deliberately a different quality: returned as is, not re-encoded
    img = jpeg_image(b)
    assert img.shape == (48, 64, 3) and encode_jpeg(img) == b
    assert encode_jpeg(img[:, :32]) != b and encode_jpeg(np.ascontiguousarray(img)) != b      # new arrays are re-encoded


def test_scene_reads_frames_dir(tmp_path):
    import cv2
    from actspace_eval.data import SceneData
    from actspace_eval.models import encode_jpeg
    wd = tmp_path / "scene"
    wd.mkdir()
    (wd / "view_graph.json").write_text(json.dumps({"nodes": ["f_1"], "edges": {"f_1": {}}}))
    (wd / "frames_all.json").write_text(json.dumps([{"frame_id": "f_1", "position": [0, 0, 0], "yaw_deg": 0, "rel_path": "x.png", "rotation_cw_quarters": 1}]))
    fr = tmp_path / "frames"
    fr.mkdir()
    b = cv2.imencode(".jpg", np.full((480, 640, 3), 90, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    (fr / "f_1.jpg").write_bytes(b)
    sc = SceneData(wd, None, 640, frames_dir=fr)
    assert encode_jpeg(sc.image("f_1")) == b and sc.image("f_1").shape == (480, 640, 3)


def test_scene_reads_rebuilt_frames(tmp_path):
    """rebuild_frames.py output (original resolution, already upright): not rotated again, downscaled when the long side exceeds
    max_long_side."""
    import cv2
    from actspace_eval.data import SceneData
    wd = tmp_path / "scene"
    wd.mkdir()
    (wd / "view_graph.json").write_text(json.dumps({"nodes": ["f_1"], "edges": {"f_1": {}}}))
    (wd / "frames_all.json").write_text(json.dumps([{"frame_id": "f_1", "position": [0, 0, 0], "yaw_deg": 0, "rel_path": "x.jpg", "rotation_cw_quarters": 1}]))
    fr = tmp_path / "rebuilt"
    fr.mkdir()
    img = np.zeros((1440, 1920, 3), np.uint8)
    img[:, :960] = 200
    cv2.imwrite(str(fr / "f_1.png"), img)
    sc = SceneData(wd, None, 640, frames_dir=fr, frames_kind="rebuilt")
    out = sc.image("f_1")
    assert out.shape == (480, 640, 3) and out[:, :300].mean() > 150 and out[:, 340:].mean() < 50


def _calls(d, rows):
    d.mkdir(parents=True)
    (d / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "trajectories.jsonl").write_text(json.dumps({"type": "episode_end", "forced": "context_overflow"}) + "\n")


def test_digest_gives_same_truncation_stats(tmp_path):
    from actspace_eval.report_open_full import max_input, run_extra
    from actspace_eval.report_open_smoke import digest_call, truncation
    loop = " ".join(["the same words again"] * 400)
    rows = [{"type": "call", "group_id": "g1", "tag": {"t": 0}, "finish_reason": "length", "response": loop, "usage": {"input_tokens": 900},
             "request": {"messages": [{"role": "user", "content": "x"}]}, "parse": {"json": False}},
            {"type": "call", "group_id": "g1", "tag": {"t": 1}, "finish_reason": "stop", "response": "{}", "usage": {"input_tokens": 1200},
             "timeout_truncated": False, "parse": {"json": True}},
            {"type": "call", "group_id": "g2", "tag": {"t": 0}, "finish_reason": "length", "reasoning": "a b c d e f g h i j k l m n o p", "timeout_truncated": True,
             "usage": {"input_tokens": 300}}]
    _calls(tmp_path / "M/s/full", rows)
    d2 = tmp_path / "S/M/s/full"
    d2.mkdir(parents=True)
    (d2 / "calls_digest.jsonl").write_text("".join(json.dumps(digest_call(r)) + "\n" for r in rows))
    (d2 / "trajectories.jsonl").write_text((tmp_path / "M/s/full/trajectories.jsonl").read_text())
    a, b = truncation(tmp_path / "M/s/full"), truncation(d2)
    assert a == b and a["loop"] == 1 and a["unfinished"] == 1
    assert run_extra(tmp_path / "M/s/full") == run_extra(d2)
    assert max_input(tmp_path, "M", {"s": ["s/full"]}) == max_input(tmp_path / "S", "M", {"s": ["s/full"]}) == 1200
    assert "request_sha256" in digest_call(rows[0]) and "response" not in digest_call(rows[0])


def _plan_dir(tmp_path, models):
    (tmp_path / "_queue_local").mkdir(exist_ok=True)
    plan = {"prefix": "full", "seeds": [0], "models": models, "settings": ["blind"], "smoke_gate": {}, "groups_tiny": ["g0"], "groups_all": ["g0"],
            "groups_smoke": ["g0"], "report_dir": str(tmp_path)}
    (tmp_path / "_queue_local/plan.json").write_text(json.dumps(plan))


def test_queue_without_slurm_is_local(tmp_path, monkeypatch):
    import subprocess as sp
    from actspace_eval.open_queue import Queue
    for k in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "QUEUE_STOP_AFTER_S", "CUDA_DEVICE_ORDER", "QUEUE_LOCAL_HOURS"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    monkeypatch.setenv("QUEUE_LOCAL", "1")
    monkeypatch.setenv("QUEUE_DIR", "_queue_local")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    _plan_dir(tmp_path, ["RoboBrain2.0-7B"])
    calls = []
    real = sp.run
    monkeypatch.setattr(sp, "run", lambda cmd, **k: calls.append(cmd) or real(["true"], **k) if cmd[0] in ("sbatch", "squeue", "scancel", "scontrol") else real(cmd, **k))
    q = Queue(tmp_path)
    assert q.local and q.job.startswith("local-") and q.free == ["2", "3"] and q.stop_at - q.t0 > 1e8
    q.ensure_next()
    q.cancel_next()
    q.leave_node()
    assert q.stopping.is_set() and not [c for c in calls if c[0] in ("sbatch", "scancel", "squeue")]
    monkeypatch.setenv("QUEUE_LOCAL_HOURS", "2")
    assert abs(Queue(tmp_path).deadline() - (Queue(tmp_path).t0 + 7200 - 1800)) < 5


@pytest.mark.requires_data
def test_run_local_writes_plan_and_reuses_it(tmp_path, monkeypatch):
    from argparse import Namespace
    from actspace_eval import open_queue as oq
    ran = []
    monkeypatch.setattr(oq.Queue, "run", lambda self: ran.append((self.local, list(self.free), sorted({t.model for t in self.tasks}), len(self.tasks))))
    for k in [k for k in __import__("os").environ if k.startswith("SLURM_")]:
        monkeypatch.delenv(k)
    for k in ("QUEUE_DIR", "QUEUE_LOCAL", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONPATH", "HF_HOME", "QUEUE_LOCAL_HOURS", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CAMBRIAN_PYTHON", "CAMBRIAN_REPO", "CAMBRIAN_MODEL"):
        monkeypatch.setenv(k, "x")          # record the original values first: cmd_run_local modifies os.environ directly; restored at the end of the test
        monkeypatch.delenv(k)
    a = Namespace(root=str(tmp_path), gpus="0,1", models=["RoboBrain2.0-7B", "Qwen3.5-2B"], settings=None, seeds="0", queue_dir="_queue_local",
                  report_dir=str(tmp_path / "rep"), hours=None, appendix=["Qwen3.5-2B"], smoke_only=False)
    oq.cmd_run_local(a)
    plan = json.loads((tmp_path / "_queue_local/plan.json").read_text())
    assert plan["models"] == ["RoboBrain2.0-7B", "Qwen3.5-2B"] and plan["appendix"][0]["suffix"] == "_nothink" and plan["paper_table"]
    assert plan["report_names"]["final"] == "local_queue_local_full.md"
    assert ran[-1][:3] == (True, ["0", "1"], ["Qwen3.5-2B", "RoboBrain2.0-7B"]) and ran[-1][3] == 2 * 8 + 2 * 8 + 8   # smoke ×2 (both in smoke_gate) + evaluation ×2 + appendix
    oq.cmd_run_local(Namespace(**{**vars(a), "models": None, "appendix": None}))      # second time: reuse plan.json
    assert ran[-1][3] == 40


@pytest.mark.requires_data
def test_run_local_smoke_only(tmp_path, monkeypatch):
    from argparse import Namespace
    from actspace_eval import open_queue as oq
    ran = []
    monkeypatch.setattr(oq.Queue, "run", lambda self: ran.append(sorted((t.kind, t.run) for t in self.tasks)))
    for k in ("QUEUE_DIR", "QUEUE_LOCAL", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONPATH", "HF_HOME", "QUEUE_LOCAL_HOURS", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "CAMBRIAN_PYTHON", "CAMBRIAN_REPO", "CAMBRIAN_MODEL", "SLURM_JOB_ID"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    oq.cmd_run_local(Namespace(root=str(tmp_path), gpus="0", models=["RoboBrain2.0-7B"], settings=None, seeds="0", queue_dir="_queue_check",
                               report_dir=str(tmp_path / "rep"), hours=None, appendix=None, smoke_only=True))
    plan = json.loads((tmp_path / "_queue_check/plan.json").read_text())
    assert plan["smoke_only"] and not plan["paper_table"] and ran[-1] == [("smoke", "smoke1_r0")] * 8


def test_config_fingerprint_ignores_only_machine_paths():
    """Hashed configs may differ from the frozen files only in the machine-specific path keys (SITE_CONFIG_KEYS)."""
    import json
    from actspace_build.common import FROZEN_FINGERPRINTS, ROOT as BROOT, config_fingerprint
    import yaml
    refs = json.loads(FROZEN_FINGERPRINTS.read_text())
    for rel in ("configs/build.yaml", "configs/merge.yaml"):
        text = (BROOT / rel).read_text()
        assert config_fingerprint(rel, text) in {r["fingerprint"] for r in refs[rel]}
        d = yaml.safe_load(text)
        sec, key = {"configs/build.yaml": ("paths", "work"), "configs/merge.yaml": ("merge", "sandbox_root")}[rel]
        d[sec][key] = "/somewhere/else"
        assert config_fingerprint(rel, yaml.safe_dump(d)) == config_fingerprint(rel, text)
        d["seed" if rel.endswith("build.yaml") else "merge"] = 1 if rel.endswith("build.yaml") else {**d["merge"], "version": "x"}
        assert config_fingerprint(rel, yaml.safe_dump(d)) != config_fingerprint(rel, text)
