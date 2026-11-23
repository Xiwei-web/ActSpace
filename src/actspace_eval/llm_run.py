"""Run one setting with a model agent (docs/evaluation_protocol.md). Real calls require llm.allow_real_calls and llm.prompts_approved to be on.

    python -m actspace_eval.llm_run --model mock --mode active_v2 --budget 40 --set tiny --name s3_mock_active_v2_b40 [--resume]
            [--history last_k|full] [--estimate-only] [--no-render] [--smoke] [--run-seed 0] [--effort high]

- --mode is a setting name (llm_agent.SETTINGS): blind, passive (--budget is k), oracle_min, oracle_route, random_walk (--run-seed selects
  one of 3 seeds per group), active_v1, active_v2; the legacy names active / oracle equal active_v1 / oracle_min.
- --effort: overrides the model's configured reasoning_effort; written to every record and to status.json.

- --smoke: smoke test. Real calls are allowed even when llm.allow_real_calls / prompts_approved are off,
  but only on groups in llm.smoke_groups (the default); confirmed prices and OPENAI_API_KEY are still required.
- Every trajectory record stores the model name and reasoning_effort; status.json stores total token usage (input / cached / output / reasoning).

- Before running, prints a cost estimate: tiny set × four settings × this budget × this model, and this run.
- Cost accumulates in real time at the configured prices; exceeding cost_cap_usd stops immediately; completed episodes are already in trajectories.jsonl, and --resume continues
  (skips completed groups; the spent amount is restored from calls.jsonl and the cap applies to the cumulative cost). Non-retryable errors or exhausted retries also stop the run, which can be resumed.
- Outputs: <paths.eval_runs>/<name>/trajectories.jsonl, calls.jsonl (every call and retry), summary.json, status.json.

Open models (provider: vllm):
    python -m actspace_eval.llm_run --model Qwen3.5-9B --mode active_v2 --budget 40 --set smoke --run-seed 0 \
            --base-url http://127.0.0.1:8000 --out-root results --name smoke_r0 --workers 3 \
            --server-info <job dir>/server_Qwen3.5-9B.json --seal [--resume]
- Outputs go to <out-root>/<model>/<setting dir>/<run name>/ (setting dirs: blind, passive_8, passive_20, passive_40, random_walk_40, oracle_min,
  oracle_route, active_v2); plus env.json (run environment: model path and overall weight hash, chat template hash, sampling parameters, seed, thinking, max_tokens,
  vLLM / torch / CUDA versions, GPU model and count, tensor parallel size, concurrency, data version and freeze manifest hash, code commit, start / end time)
  and logprobs.jsonl.gz. --seal: once the run is complete (status = complete), write SHA256SUMS and make the run directory read-only.
- --workers: number of episodes run concurrently (each worker has its own environment, agent and client; trajectories and call logs are appended under a lock).
- Context overflow: when a request after Active step 1 (the conversation already contains the model's own replies) exceeds the model context, the episode ends there,
  episode_end.forced = context_overflow, unanswered questions score 0, and the run continues. Overflow already at step 0 is handled as in the next item.
- Model capability insufficient (context too small, too many images per request; models_vllm.ModelLimit): stop the run, status = not_applicable, na_reason gives the reason.
- Server error (retries exhausted): stop the run, status = stopped_server_error; unfinished episodes are not written to the trajectory; resumable with --resume.
- --max-tokens: overrides the model's max_output_tokens, only for comparisons (_mt32k runs, not in the main tables); written to env.json and every record.
- --no-thinking: disables thinking (chat_template_kwargs.enable_thinking: false, thinking recorded as false), only for the appendix comparison
  (_nothink runs of Qwen3.5-2B, not in the main tables); only works for models that toggle thinking via chat_template_kwargs.enable_thinking.
- Resume (open models): drop half-written lines at the end of trajectories.jsonl and calls.jsonl; calls of unfinished episodes move to calls_dropped.jsonl and are excluded from truncation-rate and similar statistics.
- --control (to raise cluster utilisation): several server instances of the same model. The control file (JSON, written by the scheduler) lists the instances
  (index, address, GPUs, server info) and this run's concurrency on each instance (workers: static / active_v2); reread every 10 s; workers are started for new instances,
  added when concurrency goes up, and surplus workers finish their current episode and exit when it goes down or an instance is removed. All workers take groups from one shared queue,
  and each group is run exactly once by one worker; generation parameters and seeds depend only on the run, question group and call index, not on the instance. Every completed episode is logged as a line in
  instances.jsonl (group, instance, address, GPUs, process, start / end time); env.json records the instances at start and the number of episodes completed per instance.
- SIGTERM (open models, sent by actspace_eval.open_queue at the time limit): wait until records being written are finished, then exit with code 143; status.json is not written.
- Extra trajectory fields: cannot_move (move action without movement) and invalid (not a legal action) in step records; budget_infeasible,
  run_seed, thinking, sampling, max_output_tokens (open models) in episode_start.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import queue
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

from .cost import CostCapExceeded, CostTracker, estimate, format_estimate
from .data import MOVE_ACTIONS, FrozenData
from .env import ActSpaceEnv
from .llm_agent import SETTINGS, LLMAgent, setting_env
from .meta import group_meta
from .env import EndEpisode
from .models import _LOG_LOCK, ModelError, make_client
from .models_vllm import _LP_LOCK
from .results import build_summary, episode_rows
from .run import run_episode
from .trajectory import TrajectoryWriter, complete_episodes


def group_set(data, name, groups=None):
    if groups:
        return list(groups)
    if name == "smoke":
        return list(data.ecfg["llm"]["smoke_groups"])
    if name == "tiny":
        return list(data.ecfg["llm"]["tiny_groups"])
    if name == "all":
        return list(data.group_ids)
    raise ValueError(name)


def env_factory(data, render=False):
    """(setting, budget, run_seed) → environment."""
    def make(setting, budget, run_seed=0):
        mode, _, pose = setting_env(setting)
        return ActSpaceEnv(data, budget=budget, render=render, mode=mode, cumulative_pose=pose, run_seed=run_seed)
    return make


def estimate_plan(budget, active_steps=None):
    """tiny set × each setting: Active estimated both with the full budget and with an average of 20 steps; random_walk with 3 seeds per group."""
    return [("blind", budget, 0), ("passive", budget, 0), ("passive", 20, 0), ("passive", 8, 0), ("oracle_min", budget, 0),
            ("oracle_route", budget, 0), ("random_walk", budget, 0, (0, 1, 2)),
            ("active_v2", budget, budget), ("active_v2", budget, active_steps or 20)]


def default_effort(ec, model_key, effort):
    """The official evaluation uses reasoning_effort = llm.reasoning_effort (high) for all models and all settings;
    --effort can still override it (only for comparison experiments). Not set for models without reasoning (mock etc.)."""
    if effort is not None:
        return effort
    if ec["models"][model_key].get("reasoning", {}).get("enabled"):
        return ec["llm"].get("reasoning_effort")
    return None


def with_effort(mcfg, effort):
    """Override reasoning_effort (only for models with reasoning enabled)."""
    if effort is None:
        return mcfg
    if not mcfg.get("reasoning", {}).get("enabled"):
        raise ValueError("--effort only applies to models with reasoning enabled")
    return {**mcfg, "reasoning": {**mcfg["reasoning"], "effort": effort}}


def setting_dir(setting: str, budget: int) -> str:
    """Setting name in the results directory: blind, passive_8 / 20 / 40, random_walk_40, oracle_min, oracle_route, active_v2."""
    if setting in ("passive", "random_walk"):
        return f"{setting}_{budget}"
    return setting


def annotate_steps(records):
    """Add cannot_move (move action without movement) and invalid (not a legal action) to step records; ignored during replay (trajectory._IGNORE)."""
    for r in records:
        if r["type"] == "step":
            r["invalid"] = r["action"] not in MOVE_ACTIONS
            r["cannot_move"] = (not r["invalid"]) and not r["moved"]
    return records


def without_thinking(mcfg):
    """--no-thinking: for models whose thinking switch is chat_template_kwargs.enable_thinking, turn it off; other parameters unchanged (appendix comparison)."""
    t = mcfg.get("thinking") or {}
    if t.get("switch") != "chat_template_kwargs.enable_thinking":
        raise ValueError(f"the thinking switch of {mcfg.get('model')} is not chat_template_kwargs.enable_thinking; --no-thinking is not possible")
    return {**mcfg, "thinking": {**t, "enabled": False}, "chat_template_kwargs": {**(mcfg.get("chat_template_kwargs") or {}), "enable_thinking": False}}


def thinking_of(mcfg):
    """thinking setting of an open model (written to every record); None for API models."""
    t = mcfg.get("thinking")
    return None if t is None else bool(t.get("enabled"))


def git_commit() -> dict:
    """Code version: in a git repository, HEAD and whether there are uncommitted changes; without .git (code exported with git archive)
    read CODE_COMMIT written at export time (dirty recorded as None: cannot tell whether the code changed after export)."""
    root = Path(__file__).resolve().parents[2]
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                        capture_output=True, text=True, timeout=20).stdout.strip())
            return {"commit": r.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        pass
    f = root / "CODE_COMMIT"
    if f.exists():
        return {"commit": f.read_text().split()[0], "dirty": None, "source": "CODE_COMMIT (code exported with git archive)"}
    return {"commit": None, "dirty": None}


def seal_run(out: Path):
    """Completed run: write SHA256SUMS (all files in the directory, sorted by relative path), then make files and directories read-only."""
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = []
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for p in files + [out / "SHA256SUMS"]:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True) + [out]:
        d.chmod(d.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def trim_partial_line(path: Path):
    """Drop a half-written line at the end of the file (possible if the process was killed)."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def drop_unfinished_calls(cpath: Path, done: set):
    """Before resuming, move calls of unfinished episodes to calls_dropped.jsonl: these episodes will be rerun, and keeping them would double count statistics."""
    if not cpath.exists():
        return
    keep, drop = [], []
    with open(cpath) as f:
        for line in f:
            ep = (json.loads(line).get("tag") or {}).get("episode")
            (drop if ep is not None and ep not in done else keep).append(line)
    if drop:
        with open(cpath.with_name("calls_dropped.jsonl"), "a") as f:
            f.writelines(drop)
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text("".join(keep))
        tmp.replace(cpath)


def run_llm(data, model_key, mode, budget, gids, out: Path, resume=False, history=None, render=True,
            cost_cap=None, client_kwargs=None, print_estimate=True, smoke=False, run_seed=0, effort=None,
            workers=1, env_info=None, seal=False, max_tokens=None, control=None, no_thinking=False, tracker_cls=CostTracker, gate=None,
            on_episode=None, env_cls=None, agent_cls=None, label=None) -> dict:
    """mode is a setting name (llm_agent.SETTINGS). With workers > 1, several episodes run concurrently. control: vLLM multi-instance control file (see module docstring).
    tracker_cls: cost tracker for API registry models (reserves cost under concurrency). gate: a no-argument callable returning a context manager in which each episode runs
    (episode slots shared by several runs in one API job); on_episode(done, total, spent). env_cls / agent_cls / label: ablations (providers/ablations.py)
    substitute subclasses of ActSpaceEnv / LLMAgent; label is the setting name written to records (default: mode). If env_info contains "code", it is used as the code version."""
    ec = data.ecfg
    effort = default_effort(ec, model_key, effort)
    mcfg, llm_cfg = with_effort(ec["models"][model_key], effort), ec["llm"]
    if max_tokens:
        mcfg = {**mcfg, "max_output_tokens": int(max_tokens)}
    if no_thinking:
        mcfg = without_thinking(mcfg)
    setting = mode
    env_mode, version, pose = setting_env(setting)
    label = label or setting
    scale = ec["env"].get("step_scale")   # may be given per dataset; each episode uses data.step_scale(gid)
    if smoke:
        extra = sorted(set(gids) - set(llm_cfg["smoke_groups"]))
        if extra:
            raise ValueError(f"smoke tests may only run on llm.smoke_groups; extra groups: {extra}")
        client_kwargs = {**(client_kwargs or {}), "smoke": True}
    effort = mcfg["reasoning"]["effort"] if mcfg.get("reasoning", {}).get("enabled") else None
    thinking = thinking_of(mcfg)
    infeasible = set(ec["env"].get("budget_infeasible") or [])
    history = history or llm_cfg["history"]
    if env_mode != "active":
        history = None
    cap = ec["cost_cap_usd"] if cost_cap is None else cost_cap
    meta = group_meta(data)
    if print_estimate:
        fac = env_factory(data)
        print(format_estimate(estimate(fac, gids, [(setting, budget, budget, (run_seed,))], mcfg, history, scale),
                              f"This run ({setting}, {len(gids)} groups, upper-bound estimate; configured prices, including estimated cache hits)"))
    out.mkdir(parents=True, exist_ok=True)
    tpath, cpath, ipath = out / "trajectories.jsonl", out / "calls.jsonl", out / "instances.jsonl"
    done_eps = []
    if resume:
        trim_partial_line(tpath)
        trim_partial_line(cpath)
        trim_partial_line(ipath)
        done_eps = complete_episodes(tpath)
        with open(tpath, "w") as f:  # drop incomplete episodes from an interruption
            for ep in done_eps:
                for r in ep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif tpath.exists() or cpath.exists():
        raise FileExistsError(f"{out} already has run outputs; add --resume to resume, or use another run name")
    done = {ep[0]["group_id"] for ep in done_eps}
    if resume and mcfg["provider"] == "vllm":
        drop_unfinished_calls(cpath, done)
    cost = tracker_cls(mcfg["prices_usd_per_mtok"], cap, spent=CostTracker.spent_in_log(cpath))
    agent_name = f"{model_key}:{mcfg['model']}"
    t_start = time.time()
    climit = None
    if mcfg["provider"] == "vllm" and mcfg.get("call_timeout_s") == "auto" and out.parent.parent.name == model_key:   # <root>/<model>/<setting>/<run>
        # per-model limit = 3 × 32768 ÷ normal decoding speed (call_limit.py), written to env.json
        from .call_limit import CallLimit
        climit = CallLimit(out.parent.parent, mcfg, llm_cfg, formal=not out.name.startswith("smoke"))
        climit.ensure()
        client_kwargs = {**(client_kwargs or {}), "call_limit": climit}
    base_url = (client_kwargs or {}).get("base_url")
    if control:
        ctl = read_control(control, setting) or {"servers": [], "workers": workers}
    else:
        ctl = {"servers": [{"id": "main", "url": base_url}], "workers": workers}
    if env_info is not None:
        env_doc = {**env_info, "run": out.name, "model": model_key, "setting": label, "budget": budget, "run_seed": run_seed,
                   "thinking": thinking, "sampling": mcfg.get("sampling"), "sampling_source": mcfg.get("sampling_source"),
                   "chat_template_kwargs": mcfg.get("chat_template_kwargs"), "max_output_tokens": mcfg.get("max_output_tokens"),
                   "json_mode": mcfg.get("json_mode"), "logprobs_top": mcfg.get("logprobs_top"), "history": history,
                   "workers": workers, "groups": list(gids), "data_version": data.version,
                   "freeze_manifest_sha256": hashlib.sha256(data.manifest_path.read_bytes()).hexdigest(), "data_source": data.source,
                   "code": (env_info or {}).get("code") or git_commit(), "resume": resume,
                   "call_timeout": climit.snapshot() if climit else {"limit_s": mcfg.get("call_timeout_s", llm_cfg.get("call_timeout_s"))},
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start))}
        if control:
            env_doc["control"] = {"path": str(control), "workers_per_instance": ctl["workers"], "instances": [server_doc(x) for x in ctl["servers"]]}
        prev = json.loads((out / "env.json").read_text()) if resume and (out / "env.json").exists() else None
        if prev:
            env_doc["previous_sessions"] = prev.get("previous_sessions", []) + [{k: prev.get(k) for k in ("started_at", "ended_at", "status", "code", "server")}]
        (out / "env.json").write_text(json.dumps(env_doc, ensure_ascii=False, indent=1))

    def make_worker(srv):
        kw = dict(client_kwargs or {})
        if srv.get("url"):
            kw["base_url"] = srv["url"]
        client = make_client(mcfg, llm_cfg, cost, cpath, **kw)
        client.cache_run = out.name
        if hasattr(client, "run_seed"):
            client.run_seed = run_seed
        agent = (agent_cls or LLMAgent)(client, env_mode, history, name=agent_name, version=version)
        env = (env_cls or ActSpaceEnv)(data, budget=budget, render=render, mode=env_mode, cumulative_pose=pose, run_seed=run_seed)
        return env, agent

    writer = TrajectoryWriter(tpath, append=True)
    wlock, stop = threading.Lock(), threading.Event()
    state = {"status": "complete", "err": None, "na": None}

    used = {}

    def run_one(env, agent, gid, srv):
        t_ep = time.time()
        records, seed = run_episode(env, _TagAgent(agent, gid, data.step_scale(gid), env), gid, run_seed)
        ep_extra = {}
        if thinking is not None:
            ep_extra = {"thinking": thinking, "sampling": mcfg.get("sampling"), "max_output_tokens": mcfg.get("max_output_tokens")}
        with wlock:
            writer.write_episode(annotate_steps(records), run=out.name, agent=agent.name, agent_seed=seed, model=model_key, setting=label,
                                 dataset=data.dataset(gid), step_scale=agent.scale if env_mode == "active" else None,
                                 prompt_version=version, model_name=mcfg["model"], reasoning_effort=effort,
                                 history=history, cost_usd=round(sum(c.cost_usd for c in agent.calls), 6), calls=len(agent.calls),
                                 budget_infeasible=gid in infeasible, run_seed=run_seed, **ep_extra)
            done.add(gid)
            if control or env_info is not None:
                with open(ipath, "a") as f:
                    f.write(json.dumps({"group_id": gid, "instance": srv["id"], "url": srv.get("url"), "gpus": srv.get("gpus"), "pid": os.getpid(),
                                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_ep)),
                                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False) + "\n")
            used.setdefault(srv["id"], {**{k: srv.get(k) for k in ("url", "gpus", "info")}, "episodes": 0})["episodes"] += 1
            n_done = len(done & set(gids))
        if on_episode is not None:
            on_episode(n_done, len(gids), cost.spent)

    if mcfg["provider"] == "vllm" and threading.current_thread() is threading.main_thread():
        def on_term(*_):
            wlock.acquire()
            _LOG_LOCK.acquire()
            _LP_LOCK.acquire()
            writer.f.flush()
            os._exit(143)
        signal.signal(signal.SIGTERM, on_term)

    def fail(status, e, na=None):
        with wlock:
            if state["status"] == "complete" or na is not None:
                state.update(status=status, err=str(e), na=na)
        stop.set()

    todo = queue.Queue()
    for gid in gids:
        if gid not in done:
            todo.put(gid)

    def worker_loop(srv, retire):
        try:
            env, agent = make_worker(srv)
        except Exception as e:  # noqa: BLE001  client could not be created (e.g. real-call switches off): re-raise in the main thread
            fail("error", e)
            state["exc"] = e
            return
        slot = gate or contextlib.nullcontext
        while not stop.is_set() and not retire.is_set():
            try:
                gid = todo.get_nowait()
            except queue.Empty:
                return
            try:
                with slot():
                    run_one(env, agent, gid, srv)
            except CostCapExceeded as e:
                fail("stopped_cost_cap", e)
            except ModelError as e:
                kind = getattr(e, "kind", None)
                if kind:
                    fail("not_applicable", e, {"kind": kind, "group_id": gid, "message": str(e)[:1500]})
                else:
                    fail(getattr(e, "stop_status", None) or
                         ("stopped_server_error" if mcfg["provider"] == "vllm" or mcfg.get("api_registry") else "stopped_model_error"), e)
            except Exception as e:  # noqa: BLE001
                fail("error", e)
                state["exc"] = e

    live = []          # [thread, instance index, stop flag]
    idle_since = None

    def adjust():
        """Add workers according to the control file; let surplus workers finish their current episode and exit."""
        nonlocal ctl
        if control:
            ctl = read_control(control, setting) or ctl
        want = {x["id"]: (x, per_instance(ctl["workers"], setting)) for x in ctl["servers"]}
        for th, sid, retire in live:
            if th.is_alive() and sid not in want:
                retire.set()
        for sid, (srv, n) in want.items():
            mine = [(th, r) for th, s, r in live if s == sid and th.is_alive() and not r.is_set()]
            for th, r in mine[n:]:
                r.set()
            for _ in range(n - len(mine)):
                if todo.empty() or stop.is_set():
                    break
                r = threading.Event()
                th = threading.Thread(target=worker_loop, args=(srv, r), daemon=True)
                live.append((th, sid, r))
                th.start()

    try:
        while True:
            adjust()
            alive = [th for th, _, _ in live if th.is_alive()]
            if not alive and (stop.is_set() or todo.empty()):
                break
            if not alive:      # groups left to run but no instance: wait for the scheduler to add one, give up after too long
                idle_since = idle_since or time.time()
                if time.time() - idle_since > float(os.environ.get("ACTSPACE_CONTROL_IDLE_S", 900)):
                    fail("stopped_server_error", "no server instance available in the control file for 900 s")
                    break
            else:
                idle_since = None
            stop.wait(float(os.environ.get("ACTSPACE_CONTROL_POLL_S", 10)) if control else 0.5)
    finally:
        writer.close()
    if state.get("exc") is not None and state["status"] == "error":
        raise state["exc"]
    status, err = state["status"], state["err"]
    eps = complete_episodes(tpath)
    per_ep, per_q = [], []
    for ep in eps:
        a, b = episode_rows(data, meta, ep)
        per_ep.append(a)
        per_q += b
    head = {"run": out.name, "agent": agent_name, "model": model_key, "model_name": mcfg["model"], "reasoning_effort": effort,
            "thinking": thinking, "mode": env_mode, "setting": label, "prompt_version": version, "run_seed": run_seed,
            "version": data.version, "budget": budget, "history": history, "render": render, "smoke": smoke}
    (out / "summary.json").write_text(json.dumps(build_summary(head, per_ep, per_q), ensure_ascii=False, indent=1))
    usage = CostTracker.usage_in_log(cpath)
    st = {"status": status, "error": err, "na_reason": state["na"], "spent_usd": round(cost.spent, 6), "cost_cap_usd": cap,
          "model_name": mcfg["model"], "reasoning_effort": effort, "thinking": thinking, "setting": label,
          "usage": usage, "truncation_rate": round(usage["truncated_calls"] / usage["calls"], 6) if usage["calls"] else None,
          "completed": len(eps), "remaining": [g for g in gids if g not in {e[0]["group_id"] for e in eps}],
          "seconds": round(time.time() - t_start, 1), "workers": workers}
    (out / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    if env_info is not None:
        env_doc = json.loads((out / "env.json").read_text())
        env_doc.update(ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), status=status, seconds=st["seconds"])
        if climit is not None:
            climit.limit_s()
            env_doc["call_timeout_end"] = climit.snapshot()     # differs from the start value if it was updated automatically during the run
        if control:
            env_doc["instances_used"] = {k: {"episodes": v["episodes"], **server_doc({**v, "id": k})} for k, v in used.items()}
        (out / "env.json").write_text(json.dumps(env_doc, ensure_ascii=False, indent=1))
    if seal and status == "complete":
        seal_run(out)
    return st


def read_control(path, setting):
    """Multi-instance control file (written by the scheduler, see module docstring); None if unreadable or malformed (keep the previous content)."""
    try:
        doc = json.loads(Path(path).read_text())
        servers = [x for x in doc["servers"] if x.get("id") and x.get("url")]
        return {"servers": servers, "workers": doc["workers"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def per_instance(workers, setting) -> int:
    """This run's concurrency on each instance: an integer, or {static, active_v2} (same format as open_models.queue.workers)."""
    if isinstance(workers, dict):
        return max(0, int(workers.get(setting, workers["active_v2"] if setting == "active_v2" else workers["static"])))
    return max(1, int(workers))


def server_doc(srv) -> dict:
    """Server info of an instance (JSON written by server_info.py); without it only the index and address are recorded."""
    out = {k: srv.get(k) for k in ("id", "url", "gpus")}
    if srv.get("info") and Path(srv["info"]).exists():
        try:
            out["server"] = json.loads(Path(srv["info"]).read_text())
        except ValueError:
            pass
    return out


class _TagAgent:
    """Carry the question group ID into the call log (agent.reset clears the tag); open-model clients also log photo ID → raw frame ID (only logged, never sent to the model)."""

    def __init__(self, agent, gid, scale=None, env=None):
        self.agent, self.gid, self.scale, self.env = agent, gid, scale, env
        self.name, self.privileged = agent.name, agent.privileged

    def reset(self, *a, **k):
        self.agent.reset(*a, **k)
        self.agent.scale = self.scale   # per-step scale for the episode's dataset
        self.agent.episode_tag = self.gid
        c = self.agent.client
        c.cache_key = f"actspace-{getattr(c, 'cache_run', 'run')}-{self.gid}"
        if hasattr(c, "frame_of") and self.env is not None:
            env = self.env
            c.frame_of = lambda pid: {v: k for k, v in env._photo_ids.items()}.get(pid)
            c.group_id = self.gid

    def act(self, obs):
        try:
            return self.agent.act(obs)
        except ModelError as e:
            env = self.env
            if getattr(e, "kind", None) == "context" and env is not None and env.mode == "active" and env.steps_used > 0:
                raise EndEpisode("context_overflow") from e
            raise


def open_model_env_info(ec, model_key, server_info=None) -> dict:
    """Model and server part of env.json: model path, overall weight hash (SHA-256 of _manifests/<model>.sha256), chat template hash, server info."""
    m = ec["models"][model_key]
    om = ec["open_models"]
    info = {"model_path": m.get("path"), "served_model_name": m["model"], "vllm_args": m.get("vllm"),
            "thinking_config": m.get("thinking"), "provider": m["provider"]}
    man = Path(om["results_root"]) / "_manifests" / f"{Path(m['path']).name}.sha256"
    info["weights_manifest"] = str(man)
    info["weights_total_sha256"] = hashlib.sha256(man.read_bytes()).hexdigest() if man.exists() else None
    info["chat_template"] = chat_template_hash(m.get("path"))
    custom = (m.get("vllm") or {}).get("chat_template")
    if custom:   # template actually sent to the server (only the passing format changes, see configs/chat_templates/)
        info["chat_template_served"] = {"source": custom, "sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / custom).read_bytes()).hexdigest()}
    if server_info and Path(server_info).exists():
        info["server"] = json.loads(Path(server_info).read_text())
    return info


def chat_template_hash(path) -> dict | None:
    """chat template in the model directory (chat_template.jinja / chat_template.json / chat_template in tokenizer_config.json) and its SHA-256."""
    if not path:
        return None
    p = Path(path)
    for name in ("chat_template.jinja", "chat_template.json"):
        if (p / name).exists():
            return {"source": name, "sha256": hashlib.sha256((p / name).read_bytes()).hexdigest()}
    tc = p / "tokenizer_config.json"
    if tc.exists():
        t = json.loads(tc.read_text()).get("chat_template")
        if t:
            t = t if isinstance(t, str) else json.dumps(t, sort_keys=True)
            return {"source": "tokenizer_config.json:chat_template", "sha256": hashlib.sha256(t.encode()).hexdigest()}
    return {"source": None, "sha256": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=list(SETTINGS), help="setting name (see module docstring)")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--set", default="tiny", choices=["smoke", "tiny", "all"])
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--name", help="run name (not needed with --estimate-only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--history", choices=["last_k", "full"])
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="smoke test: run only on llm.smoke_groups (see module docstring)")
    ap.add_argument("--run-seed", type=int, default=0, help="random_walk seed (3 per group: 0, 1, 2)")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="override the model's configured reasoning_effort")
    ap.add_argument("--cost-cap", type=float, help="cost cap of this run (including resumes), default cost_cap_usd; "
                    "jobs with several settings use it so the four settings share one total cap")
    ap.add_argument("--base-url", help="open models: address of vllm serve, e.g. http://127.0.0.1:8000")
    ap.add_argument("--out-root", help="open models: results root; the run directory is <out-root>/<model>/<setting dir>/<run name>")
    ap.add_argument("--workers", type=int, default=1, help="number of episodes run concurrently")
    ap.add_argument("--server-info", help="open models: server info JSON written by the job (vLLM / torch / CUDA versions, GPU, tensor parallel size, etc.), merged into env.json")
    ap.add_argument("--seal", action="store_true", help="write SHA256SUMS and make read-only when complete")
    ap.add_argument("--max-tokens", type=int, help="override max_output_tokens (only for comparisons, see module docstring)")
    ap.add_argument("--control", help="open models: multi-instance control file (written by the scheduler, see module docstring)")
    ap.add_argument("--no-thinking", action="store_true", help="disable thinking, only for the appendix comparison (see module docstring)")
    args = ap.parse_args()
    data = FrozenData()
    budget = args.budget or data.ecfg["env"]["budget"]
    history = None
    if args.history == "full":
        history = {"mode": "full"}
    elif args.history == "last_k":
        history = data.ecfg["llm"]["history"]
    gids = group_set(data, args.set, args.groups or (data.ecfg["llm"]["smoke_groups"] if args.smoke else None))
    if args.estimate_only:
        mcfg = with_effort(data.ecfg["models"][args.model], default_effort(data.ecfg, args.model, args.effort))
        h = history or data.ecfg["llm"]["history"]
        fac = env_factory(data)
        print(format_estimate(estimate(fac, data.ecfg["llm"]["tiny_groups"], estimate_plan(budget), mcfg, h, data.ecfg["env"].get("step_scale")),
                              f"Cost estimate: tiny set × settings × budget {budget} × {args.model} (configured prices; including estimated cache hits)"))
        return
    if not args.name:
        ap.error("--name is required unless --estimate-only")
    mcfg = data.ecfg["models"][args.model]
    client_kwargs, env_info = None, None
    if args.out_root:
        out = Path(args.out_root) / args.model / setting_dir(args.mode, budget) / args.name
    else:
        out = Path(data.ecfg["paths"]["eval_runs"]) / args.name
    if mcfg["provider"] == "vllm":
        if not args.base_url:
            ap.error("open models need --base-url")
        client_kwargs = {"base_url": args.base_url}
        env_info = open_model_env_info(data.ecfg, args.model, args.server_info)
    st = run_llm(data, args.model, args.mode, budget, gids, out, resume=args.resume, history=history, render=not args.no_render,
                 smoke=args.smoke, cost_cap=args.cost_cap, run_seed=args.run_seed, effort=args.effort,
                 client_kwargs=client_kwargs, workers=args.workers, env_info=env_info, seal=args.seal, max_tokens=args.max_tokens,
                 control=args.control, no_thinking=args.no_thinking)
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
