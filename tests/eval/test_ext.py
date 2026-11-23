"""Six additional open models: server arguments, plugin environment, further smoke-gate checks, MiniCPM segmented placeholders."""
import json

import pytest
import sys
from pathlib import Path

from actspace_eval.data import load_eval_config
from actspace_eval.open_serve import PLUGIN_DIR, server_env, vllm_args

EXT = ["LLaVA-OneVision-7B", "RoboBrain2.0-7B", "SenseNova-SI-1.1-InternVL3-2B", "MiniCPM-V-4_5", "Qwen3.5-2B", "GLM-4.6V-Flash"]


def get(a, k):
    return a[a.index(k) + 1]


def test_ext_models_serve_args():
    ec = load_eval_config()
    for k in EXT:
        a = vllm_args(ec, k, 1)
        assert get(a, "--tensor-parallel-size") == "1" and get(a, "--max-model-len") in ("65536", "40960", "32768")
    a = vllm_args(ec, "LLaVA-OneVision-7B", 1)
    assert get(a, "--dtype") == "float16" and get(a, "--max-model-len") == "32768"
    assert json.loads(get(a, "--hf-overrides")) == {"architectures": ["LlavaOnevisionBaseOnly"], "image_grid_pinpoints": [[384, 384]]}
    assert get(vllm_args(ec, "RoboBrain2.0-7B", 1), "--dtype") == "bfloat16"
    assert json.loads(get(vllm_args(ec, "SenseNova-SI-1.1-InternVL3-2B", 1), "--hf-overrides")) == {"max_dynamic_patch": 1}
    assert get(vllm_args(ec, "GLM-4.6V-Flash", 1), "--reasoning-parser") == "glm45"
    m = ec["models"]
    assert m["MiniCPM-V-4_5"]["chat_template_kwargs"] == {"enable_thinking": True} and m["GLM-4.6V-Flash"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert m["RoboBrain2.0-7B"]["sampling"] == {"temperature": 1e-6, "repetition_penalty": 1.05}
    assert not m["LLaVA-OneVision-7B"]["thinking"]["enabled"] and not m["RoboBrain2.0-7B"]["thinking"]["enabled"]
    q = ec["open_models"]["queue"]
    # LLaVA and MiniCPM re-ran the smoke test (smoke2) after switching to format-adapted templates; the others use smoke1
    redo = ("LLaVA-OneVision-7B", "MiniCPM-V-4_5")
    assert q["order"][-6:] == EXT and all(q["smoke_gate"][k] == ("smoke2" if k in redo else "smoke1") for k in EXT)
    for k, f in (("LLaVA-OneVision-7B", "llava_onevision_interleaved.jinja"), ("MiniCPM-V-4_5", "minicpm_v_interleaved.jinja")):
        assert get(vllm_args(ec, k, 1), "--chat-template").endswith("configs/chat_templates/" + f)
        assert get(vllm_args(ec, k, 1), "--chat-template-content-format") == "openai"


def test_plugin_env_only_for_plugin_models():
    ec = load_eval_config()
    env = server_env(ec, "LLaVA-OneVision-7B", {"PYTHONPATH": "src"})
    assert env["PYTHONPATH"].split(":") == [str(PLUGIN_DIR), "src"] and "ACTSPACE_MINICPMV_TOKENIZER" not in env
    env = server_env(ec, "MiniCPM-V-4_5", {})
    assert env["PYTHONPATH"] == str(PLUGIN_DIR) and env["ACTSPACE_MINICPMV_TOKENIZER"] == "1"
    assert server_env(ec, "Qwen3.5-2B", {"PYTHONPATH": "src"}) == {"PYTHONPATH": "src"}
    ep = (PLUGIN_DIR / "actspace_vllm_plugins-0.1.dist-info" / "entry_points.txt").read_text()
    assert "[vllm.general_plugins]" in ep and "actspace_vllm_plugins:register" in ep


def test_minicpmv_tokenizer_patch_matches_official_names():
    sys.path.insert(0, str(PLUGIN_DIR))
    from actspace_vllm_plugins import MINICPMV_TOKENS
    from actspace_eval.data import load_eval_config
    src = Path(load_eval_config()["models"]["MiniCPM-V-4_5"]["path"]) / "tokenization_minicpmv_fast.py"
    if not src.exists():
        pytest.skip("MiniCPM-V-4_5 weights are not available locally")
    if src.exists():
        text = src.read_text()
        for name, tok in MINICPMV_TOKENS.items():
            assert f'self.{name} = "{tok}"' in text


def _call(gid, w, h, text, parsed=True, retries=0, inp=100):
    return {"type": "call", "tag": {"t": 0, "episode": gid}, "group_id": gid, "rendered_prompt": {"text": text}, "parse": {"json": parsed},
            "retries": retries, "usage": {"input_tokens": inp},
            "request": {"messages": [{"role": "user", "content": [{"type": "image", "w": w, "h": h, "photo_id": "photo_1"}]}]}}


def _run(d, calls, sealed=True, episodes=2):
    d.mkdir(parents=True, exist_ok=True)
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "summary.json").write_text(json.dumps({"episodes": episodes}))
    if sealed:
        (d / "SHA256SUMS").write_text("x\n")


def test_minicpm_segments_are_summed_per_image(tmp_path):
    from actspace_eval.report_open_full import tokens_by_orientation
    _run(tmp_path / "r", [_call("arkit_g_1", 480, 640, "<unk>×64 a <unk>×64 b <unk>×64"), _call("scannetpp_g_1", 640, 480, "<unk>×64 <unk>×64 <unk>×64")])
    assert tokens_by_orientation(tmp_path / "r") == {"arkit 480×640": [192], "scannetpp 640×480": [192]}


def test_gate_extra_checks(tmp_path):
    from actspace_eval.report_open_full import gate_result
    good = [_call("arkit_g_1", 480, 640, "<image>×730", inp=31000), _call("scannetpp_g_1", 640, 480, "<image>×730", inp=900)]
    _run(tmp_path / "M/passive_8/s", good)
    g = gate_result(tmp_path, "M", "s", ["passive_8"], 2, extra=True)
    assert g["pass"] and g["checks"]["parse_rate"] == 1.0 and g["checks"]["max_input"] == {"passive_8": 31000}
    _run(tmp_path / "M/passive_20/s", [dict(c, parse={"json": False}) for c in good])
    g = gate_result(tmp_path, "M", "s", ["passive_20"], 2, extra=True)
    assert g["pass"] and g["checks"]["parse_rate"] == 0.0 and g["why"] == []     # the parse rate is recorded only, not a reason to pause
    _run(tmp_path / "M/passive_40/s", good, sealed=False)
    g = gate_result(tmp_path, "M", "s", ["passive_40"], 2, extra=True)
    assert not g["pass"] and "incomplete records" in g["why"][0]
    assert gate_result(tmp_path, "M", "s", ["passive_20"])["pass"]       # without extra: unchanged behaviour (token counts only)


# ---------------------------------------------------------------- GPUs, env_error, appendix comparisons

SMI4 = [{"index": str(i), "uuid": f"GPU-{i}", "bus": f"0000:{i:02x}:00.0", "used": u, "total": 40960} for i, u in enumerate([26921, 0, 17013, 0])]


def test_allocated_gpus_follow_slurm():
    from actspace_eval.open_queue import allocated_gpus
    # no device isolation (ConstrainDevices=no): node indices, SLURM_JOB_GPUS is authoritative
    assert allocated_gpus({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "1,3", "CUDA_VISIBLE_DEVICES": "1,3"}, SMI4)[0] == ["1", "3"]
    g, note = allocated_gpus({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2,3", "CUDA_VISIBLE_DEVICES": "0,1"}, SMI4)
    assert g == ["2", "3"] and "disagrees" in note
    # device isolation: only the 2 allocated GPUs are visible, indices start at 0
    assert allocated_gpus({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2,3", "CUDA_VISIBLE_DEVICES": "0,1"}, SMI4[:2])[0] == ["0", "1"]
    assert allocated_gpus({"SLURM_JOB_ID": "1"}, SMI4)[0] == []                     # no guessing
    assert allocated_gpus({"CUDA_VISIBLE_DEVICES": "0,1"}, [])[0] == ["0", "1"]      # not inside a Slurm job (tests)
    assert allocated_gpus({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "1,7"}, SMI4)[0] == ["1"]


def test_env_error_patterns():
    from actspace_eval.open_queue import env_error_of
    log = ("INFO x\n(EngineCore pid=1) ValueError: Free memory on device cuda:0 (12.8/39.49 GiB) on startup is less than desired GPU memory "
           "utilization (0.9, 35.55 GiB). Decrease GPU memory utilization\nRuntimeError: Engine core initialization failed\n")
    assert "Free memory on device" in env_error_of(log)
    assert env_error_of("RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable")
    assert env_error_of("OSError: [Errno 98] Address already in use")
    assert env_error_of("ValueError: Model architectures ['Foo'] are not supported") == ""
    assert env_error_of("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate") == ""      # a model that does not fit is a model / parameter problem


def _ext_queue(tmp_path, monkeypatch, models, free="0,1", **plan):
    from actspace_eval.open_queue import Queue
    (tmp_path / "_queue").mkdir(exist_ok=True)
    base = {"prefix": "full", "seeds": [0], "models": models, "settings": ["blind", "active_v2"], "smoke_gate": {}, "groups_tiny": ["g0"],
            "groups_all": ["g0", "g1"], "groups_smoke": ["g0"], "report_dir": str(tmp_path)}
    (tmp_path / "_queue/plan.json").write_text(json.dumps({**base, **plan}))
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setenv("QUEUE_STOP_AFTER_S", "100000")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", free)
    q = Queue(tmp_path)
    started = []
    monkeypatch.setattr(q, "model_thread", lambda m, g, p, phase="main": started.append((phase, m, tuple(g))))
    monkeypatch.setattr(q, "reload_config", lambda: None)
    q.q = {**q.q, "order": models}
    return q, started


def test_card_with_foreign_memory_is_skipped(tmp_path, monkeypatch):
    import time as _t
    from actspace_eval import open_queue as oq
    q, started = _ext_queue(tmp_path, monkeypatch, ["LLaVA-OneVision-7B", "RoboBrain2.0-7B"])
    monkeypatch.setattr(oq, "smi_gpus", lambda: [dict(g) for g in SMI4])
    q.launch()
    _t.sleep(0.05)
    assert started == [("main", "LLaVA-OneVision-7B", ("1",))]          # GPU 0 has 26.9 GiB used by someone else: do not start there
    assert q.free == ["0"] and "not enough for" in (tmp_path / "_queue/events.log").read_text()
    # GPU 0 becomes usable once it is freed
    monkeypatch.setattr(oq, "smi_gpus", lambda: [dict(g, used=0) for g in SMI4])
    q.launch()
    _t.sleep(0.05)
    assert started[-1] == ("main", "RoboBrain2.0-7B", ("0",))


def test_all_cards_blocked_leaves_node_and_next_job_excludes_it(tmp_path, monkeypatch):
    import subprocess as sp
    from types import SimpleNamespace
    from actspace_eval import open_queue as oq
    q, started = _ext_queue(tmp_path, monkeypatch, ["LLaVA-OneVision-7B"], free="0")
    monkeypatch.setattr(oq, "smi_gpus", lambda: [dict(g) for g in SMI4])
    calls, queued = [], ["555"]

    def run(cmd, **k):
        calls.append(cmd)
        if cmd[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="".join(j + "\n" for j in queued), stderr="")
        return SimpleNamespace(returncode=0, stdout="777\n", stderr="")
    monkeypatch.setattr(sp, "run", run)
    monkeypatch.setattr(oq.os, "uname", lambda: SimpleNamespace(nodename="node-b"))
    q.q = {**q.q, "foreign_wait_s": 0}
    q.launch()
    assert started == [] and q.launchable()
    q.leave_node()
    assert q.stopping.is_set() and q.plan["exclude_nodes"] == ["node-b"] and ["scancel", "555"] in calls
    q.ensure_next()                                   # squeue still shows the just-cancelled 555: ignored
    sb = [c for c in calls if c[0] == "sbatch"]
    assert len(sb) == 1 and "--exclude=node-b" in sb[0]
    assert json.loads((tmp_path / "_queue/plan.json").read_text())["exclude_nodes"] == ["node-b"]


def test_env_error_does_not_count_as_model_try(tmp_path, monkeypatch):
    from actspace_eval import open_queue as oq
    _ext_queue(tmp_path, monkeypatch, ["LLaVA-OneVision-7B"], free="0,1")
    q2 = oq.Queue(tmp_path)                            # new instance: model_thread not replaced
    q2.q = {**q2.q, "max_server_tries": 3}
    msg = "ValueError: Free memory on device cuda:0 (12.8/39.49 GiB) on startup is less than desired GPU memory utilization (0.9, 35.55 GiB)."
    n = []

    def fake_start(self):
        n.append(self.gpus)
        self.proc = type("P", (), {"poll": lambda s: 1, "returncode": 1})()
        self.log_path.write_text(msg + "\n")
        self.log_from = 0
        self.check_env()
        return False
    monkeypatch.setattr(oq.Server, "start", fake_start)
    monkeypatch.setattr(oq.Server, "stop", lambda self: None)
    q2.running.add("LLaVA-OneVision-7B")
    q2.launched.add("LLaVA-OneVision-7B")
    q2.free = ["1"]
    q2.model_thread("LLaVA-OneVision-7B", ["0"], 8100)
    assert len(n) == 1                                 # environment cause: no 3 retries, GPUs released right away
    ts = q2.of_model("LLaVA-OneVision-7B")
    assert all(t.state == "pending" and t.reason.startswith("env_error") and t.tries == 0 for t in ts)
    assert "LLaVA-OneVision-7B" not in q2.launched and q2.bad["0"] > 0 and sorted(q2.free) == ["0", "1"]
    q2.mark_bad(["0"], "again")
    assert q2.bad["0"] == float("inf")                 # 2nd time on the same GPU: no longer used by this job


def test_appendix_tasks_wait_for_main_and_pause_at_end(tmp_path, monkeypatch):
    import time as _t
    from actspace_eval.open_queue import plan_tasks
    ap = [{"model": "Qwen3.5-2B", "suffix": "_nothink", "llm_args": ["--no-thinking"]}]
    q, started = _ext_queue(tmp_path, monkeypatch, ["RoboBrain2.0-7B", "Qwen3.5-2B"], appendix=ap)
    app = [t for t in q.tasks if t.kind == "appendix"]
    assert [t.run for t in app] == ["full_r0_nothink", "full_r0_nothink"] and {t.setting for t in app} == {"blind", "active_v2"}
    assert len(plan_tasks(q.plan)) == 2 * 2 + 2
    for t in q.of_model("Qwen3.5-2B", "full"):
        t.state = "done"
    q.launched.add("Qwen3.5-2B")
    q.launch()
    _t.sleep(0.05)
    assert started == [("main", "RoboBrain2.0-7B", ("0",))]          # RoboBrain not finished: appendix does not start
    for t in q.of_model("RoboBrain2.0-7B", "full"):
        t.state = "done"
    q.launch()
    _t.sleep(0.05)
    assert started[-1] == ("appendix", "Qwen3.5-2B", ("1",))
    # llm_run arguments
    cmds = []

    class P:
        def wait(self):
            return 0
    import subprocess as sp
    monkeypatch.setattr(sp, "Popen", lambda cmd, **k: cmds.append(cmd) or P())
    from actspace_eval.open_queue import Server
    q.spawn(app[0], Server(q, "Qwen3.5-2B", ["1"], 8201, tag="_appendix"), ["g0"], True)
    assert cmds[0][-1] == "--no-thinking" and cmds[0][cmds[0].index("--name") + 1] == "full_r0_nothink" and "--control" not in cmds[0]


def test_no_thinking_override():
    from actspace_eval.data import load_eval_config
    from actspace_eval.llm_run import without_thinking
    m = load_eval_config()["models"]["Qwen3.5-2B"]
    x = without_thinking(m)
    assert x["chat_template_kwargs"] == {"enable_thinking": False} and x["thinking"]["enabled"] is False
    assert x["sampling"] == m["sampling"] and x["max_output_tokens"] == m["max_output_tokens"] and m["chat_template_kwargs"] == {"enable_thinking": True}
    import pytest
    with pytest.raises(ValueError):
        without_thinking(load_eval_config()["models"]["LLaVA-OneVision-7B"])


def test_appendix_runs_stay_out_of_main_tables(tmp_path):
    from actspace_eval.report_open_full import runs_of
    for name in ("full_r0", "full_r0_nothink"):
        d = tmp_path / "M" / "blind" / name
        d.mkdir(parents=True)
        (d / "SHA256SUMS").write_text("x")
    assert runs_of(tmp_path, "M", "blind", "full") == ["blind/full_r0"]
    assert runs_of(tmp_path, "M", "blind", "full", suffix="_nothink") == ["blind/full_r0_nothink"]
