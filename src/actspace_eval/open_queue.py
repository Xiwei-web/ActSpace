"""Task queue for the open-model evaluation runs: one allocation, tasks run back to back on the node.

    python -m actspace_eval.open_queue init --prefix full [--seeds 0] [--report-dir <dir>] [--models …] [--settings …] [--groups …] [--tiny-groups …]
    python -m actspace_eval.open_queue run             # inside a GPU job (scripts/run_open_queue.sbatch)
    python -m actspace_eval.open_queue status          # show progress at any time (read-only)
    python -m actspace_eval.open_queue reset <model> [<setting>]   # set failed / paused tasks back to pending (after a manual decision)
    python -m actspace_eval.open_queue watchdog        # daily check (e.g. from scrontab)

- Task = model × setting × run. States live in <results root>/_queue/tasks.tsv (pending / running / done / failed / paused, with
  a reason); events go to events.log. Only the scheduler writes these files; completion is decided by the run directory
  (a run with SHA256SUMS is done).
- In the order of open_models.queue.order, the next model that fits is placed on free GPUs (GPUs = tensor parallelism): start
  its server, run all its settings concurrently, then stop the server and release the GPUs.
- Each run has two phases: first the tiny groups, then the remaining groups (same run directory, --resume). When all tiny runs
  of a model are done, a tiny draft report is written.
- Models in open_models.queue.smoke_gate run a smoke test first; the evaluation runs are released only if the tokens per image
  agree between landscape and portrait images, otherwise the evaluation tasks are paused.
- A failing task does not affect the others: on server errors or process crashes the server is restarted and the task retried;
  when the retries are exhausted the task is marked failed. Insufficient context (not_applicable) is a protocol issue: the task
  is paused and noted in the progress file.
- Continuation jobs: at its start a job submits the next continuation job (--dependency=afterany:<current job>), so the
  continuation starts whether the current job ends normally, times out or is killed. At most one at any time: nothing is
  submitted if a job with the same name is already queued (nor if the query fails; it is retried later). When the queue is
  finished, the queued continuation job is cancelled. A continuation job with no pending tasks exits right away.
  To stop completely, cancel both jobs: scancel -n open_queue.
- stop_margin_s before the job's time limit: wait for records being written, stop the servers and exit; unfinished episodes are
  discarded and rerun.
- watchdog (daily, e.g. via scrontab): if no job is running or queued and tasks remain, resubmit and note it in the progress file.
- After each model the three score tables are written (report_open_full) and a line is appended to the progress file; when
  everything is done, open_models_full.md and _queue/DONE are written.
- To add repeats 1 and 2 later: init --seeds 1,2 and submit a job; the score tables merge all completed runs in the directory.

Utilisation (scheduling only; evaluation rules and parameters are unchanged):
- The start order comes from open_models.queue.order (re-read from the config the next time a model is started). In that order:
  a model not yet started that has pending tasks is started if it fits; for a model already running evaluation tasks, if more
  groups remain than episodes in flight, another server instance of the same model with the same parameters is started
  (GPUs = tensor parallelism). Freed GPUs therefore go to models earlier in the order first.
- Multiple instances: the scheduler writes the instance list of a model and the concurrency per instance to
  <job dir>/ctl_<model>.json; every run (llm_run --control) re-reads it every 10 s and takes groups from one shared pending
  queue, each group exactly once; the instance that ran each group is recorded in the run directory's instances.jsonl.
  When the batch of a model ends, all its instances are stopped and the GPUs released. An instance that exits midway is removed
  from the list; after 2 failed instance starts for a model no more are started.
- Concurrency: default open_models.queue.workers(_by_model) (per instance); _queue/workers.json ({model: {static, active_v2}})
  can be changed while running and is synced to the control files every minute. GPUs listed in _queue/hold_gpus (IDs separated
  by spaces) are not allocated (for testing); deleting the file releases them.
- Takeover (switching the scheduler inside the same job): an external helper may freeze the old scheduler and start a new one in
  the same job (srun --overlap). _queue/takeover.json records the job and the old scheduler process; the daily check cancels the
  job if it is still alive but the takeover step is gone, so that the continuation job takes over.

GPU and environment errors (GPUs may already be partly occupied by processes outside this job):
- Only the GPUs Slurm allocated to this job are used (allocated_gpus). Without device isolation (ConstrainDevices=no) the job sees
  all GPUs of the node and nvidia-smi indices are node indices; SLURM_JOB_GPUS is authoritative and is cross-checked against
  CUDA_VISIBLE_DEVICES. Child processes get CUDA_DEVICE_ORDER=PCI_BUS_ID so that CUDA indices match nvidia-smi.
  At job start the node name, SLURM_JOB_GPUS, CUDA_VISIBLE_DEVICES and the nvidia-smi output (including processes per GPU) are
  logged and saved to <job dir>/nvidia_smi_start.txt.
- Before starting each server the free memory of its GPUs is checked (vLLM needs gpu_memory_utilization × total memory, plus
  1 GiB): if it is not enough, the server is not started on that GPU; wait until it frees up or use another allocated GPU.
  If none of the allocated GPUs is usable, no model of this job is running, and this lasts open_models.queue.foreign_wait_s
  seconds: stop, and the continuation job gets --exclude for the current node (exclude_nodes in plan.json).
- Server start failures or server exits caused by the environment (memory occupied, driver / NVML errors, port in use, etc.;
  see ENV_PATTERNS) count as env_error: they do not count against the model's retries, the tasks stay pending, the GPU is
  disabled for ENV_COOLDOWN_S seconds (not used again by this job after its 2nd env_error), and the model is restarted later on a
  usable GPU.

Local runs (machines without Slurm):
    python -m actspace_eval.open_queue run-local --gpus 0,1 [--models …] [--settings …] [--seeds 0] [--root <results dir>] [--hours 48]
- Same task queue, resume, sealing, score tables and paper tables; only the GPUs given by --gpus are used (CUDA_VISIBLE_DEVICES,
  nvidia-smi indices), with the same free-memory check before starting servers.
- Queue directory defaults to <results dir>/_queue_local; with --models, plan.json is (re)written; completed runs are recognised
  by their directories, failed / paused tasks keep their state. After an interruption, rerun the same command to continue.
- --smoke-only: only the smoke runs (same run names as on the cluster, e.g. smoke1_r0), for consistency checks after a migration
  (scripts/compare_runs.py); no score tables or paper tables.
- No continuation jobs and no daily check; stops when --hours is reached (default unlimited). If none of the given GPUs is
  usable, it stops and states why; free the memory and rerun the same command.
- Child process environment: PYTHONPATH=src, HF offline, HF_HOME = open_models.hf_home, vLLM and inductor caches under /tmp.

Appendix comparisons: the appendix of plan.json ([{model, suffix, llm_args}]) adds one task per setting whose run name carries the
suffix (kind appendix, e.g. full_r0_nothink). They start only after all main-table tasks (smoke, evaluation) have ended; appendix
tasks not run by the end of the job are marked paused (no continuation, not counted as unfinished). They are excluded from the
main and paper tables (runs_of only matches <prefix>_[rs]<seed>).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .data import ROOT, load_eval_config
from .open_serve import server_env, vllm_args
from .report_open_full import SETTINGS, gate_result, run_name

PENDING, RUNNING, DONE, FAILED, PAUSED = "pending", "running", "done", "failed", "paused"
MAIN = ("smoke", "full")
ENV_COOLDOWN_S = 600
ENV_MARGIN_MIB = 1024
# Server failures caused by the environment: not counted against the model's retries. Problems of the model itself (unsupported
# architecture, unreadable weights, not enough memory for the model, etc.) are not listed here.
ENV_PATTERNS = [r"Free memory on device .* is less than desired GPU memory utilization",
                r"CUDA-capable device\(s\) is/are busy or unavailable", r"no CUDA-capable device is detected", r"cudaErrorDevicesUnavailable",
                r"cudaErrorNoDevice", r"Failed to initialize NVML", r"NVML_ERROR|NVMLError", r"CUDA driver version is insufficient",
                r"CUDA error: unknown error", r"uncorrectable ECC error", r"CUDA error: an illegal memory access was encountered.*Xid",
                r"Address already in use"]
PY = sys.executable
CAMBRIAN_SERVE = str(ROOT / "scripts" / "cambrian" / "serve_cambrian.sh")     # default; open_models.cambrian_serve / cambrian_python in the config take precedence
CAMBRIAN_PY = None       # cambrian_python must be given in configs/paths.yaml
MAX_JOBS = 8     # cap on the total number of jobs: do not keep submitting forever when the same tasks never finish
JOB_NAME = "open_queue"


def queue_dir(root: Path) -> Path:
    """Queue directory. QUEUE_DIR selects an independent queue that coexists with the main queue (e.g. _queue_q38)."""
    return root / os.environ.get("QUEUE_DIR", "_queue")


def env_error_of(text: str) -> str:
    """Environment cause (ENV_PATTERNS) in a server log: returns the matching line, or an empty string."""
    for pat in ENV_PATTERNS:
        m = re.search(rf"[^\n]*(?:{pat})[^\n]*", text)
        if m:
            return m.group(0).strip()[-300:]
    return ""


def smi_gpus() -> list[dict]:
    """GPUs visible to nvidia-smi: index, uuid, bus ID, used / total memory (MiB). Returns [] if the query fails."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.strip().splitlines() if r.returncode == 0 else []:
        c = [x.strip() for x in line.split(",")]
        if len(c) == 5 and c[3].isdigit() and c[4].isdigit():
            out.append({"index": c[0], "uuid": c[1], "bus": c[2], "used": int(c[3]), "total": int(c[4])})
    return out


def allocated_gpus(env, smi: list[dict]) -> tuple[list[str], str]:
    """GPUs this job may use (nvidia-smi indices) and an explanation. Only GPUs allocated by Slurm are used.
    - More GPUs visible than allocated (no device isolation, ConstrainDevices=no): indices are node indices and SLURM_JOB_GPUS is
      authoritative; a disagreement with CUDA_VISIBLE_DEVICES is noted.
    - Only the allocated GPUs visible (device isolation): indices start at 0, use CUDA_VISIBLE_DEVICES.
    - Neither available inside a Slurm job: return [] (no guessing). Outside a Slurm job (tests): CUDA_VISIBLE_DEVICES, default 0."""
    split = lambda v: [x.strip() for x in (v or "").split(",") if x.strip()]
    slurm = split(env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS"))
    cvd = split(env.get("CUDA_VISIBLE_DEVICES"))
    if not env.get("SLURM_JOB_ID"):
        return cvd or ["0"], "not inside a Slurm job"
    if not slurm and not cvd:
        return [], "Slurm reported no allocated GPUs (neither SLURM_JOB_GPUS nor CUDA_VISIBLE_DEVICES is set)"
    n = len(slurm or cvd)
    if smi and len(smi) > n:
        gpus, note = (slurm or cvd), "all GPUs of the node are visible (no device isolation); using the node indices allocated by Slurm"
        if slurm and cvd and sorted(slurm) != sorted(cvd):
            note += f"; CUDA_VISIBLE_DEVICES {','.join(cvd)} disagrees with SLURM_JOB_GPUS {','.join(slurm)}, using the latter"
    else:
        gpus, note = (cvd or [str(i) for i in range(n)]), "only the allocated GPUs are visible, indices start at 0"
    idx = {g["index"] for g in smi}
    if idx and not set(gpus) <= idx:
        note += f"; {sorted(set(gpus) - idx)} not in nvidia-smi, not used"
        gpus = [g for g in gpus if g in idx]
    return gpus, note


def exclude_args(plan: dict) -> list[str]:
    """--exclude for the continuation job: the --exclude already in the sbatch script plus exclude_nodes from plan.json
    (an --exclude on the command line overrides the one in the script)."""
    nodes = list(plan.get("exclude_nodes") or [])
    if not nodes:
        return []
    try:
        m = re.search(r"^#SBATCH\s+--exclude=(\S+)", (ROOT / "scripts/run_open_queue.sbatch").read_text(), re.M)
    except OSError:
        m = None
    base = m.group(1).split(",") if m else []
    return [f"--exclude={','.join(dict.fromkeys(base + nodes))}"]


@dataclass
class Task:
    model: str
    setting: str
    run: str
    seed: int
    kind: str = "full"       # full / smoke
    state: str = PENDING
    reason: str = ""
    done: int = 0
    total: int = 0
    tiny: bool = False
    tries: int = 0
    updated: str = ""

    def key(self):
        return self.model, self.setting, self.run


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def done_groups(run_dir: Path) -> set:
    """Question groups of completed episodes (looks only at episode_start / episode_end in trajectories.jsonl)."""
    p = run_dir / "trajectories.jsonl"
    out, cur = set(), None
    if not p.exists():
        return out
    with open(p, errors="replace") as f:
        for line in f:
            if '"type": "episode_start"' in line:
                m = re.search(r'"group_id": "([^"]+)"', line)
                cur = m.group(1) if m else None
            elif '"type": "episode_end"' in line and cur and line.endswith("\n"):
                out.add(cur)
    return out


def plan_tasks(plan: dict) -> list[Task]:
    tasks = []
    gate = plan.get("smoke_gate") or {}
    for m in plan["models"]:
        if m in gate:
            tasks += [Task(m, s, f"{gate[m]}_r0", 0, "smoke", total=len(plan["groups_smoke"])) for s in plan["settings"]]
        if plan.get("smoke_only"):            # smoke runs only (consistency check after a migration)
            continue
        for s in plan["settings"]:
            for k in (plan["seeds"] if SETTINGS[s][2] else [0]):
                tasks.append(Task(m, s, run_name(plan["prefix"], s, k), k, total=len(plan["groups_all"])))
    for a in plan.get("appendix") or []:          # appendix comparison: 1 run per setting
        tasks += [Task(a["model"], s, run_name(plan["prefix"], s, 0) + a["suffix"], 0, "appendix", total=len(plan["groups_all"])) for s in plan["settings"]]
    return tasks


def appendix_of(plan: dict, t: "Task") -> dict:
    return next((a for a in plan.get("appendix") or [] if a["model"] == t.model and t.run.endswith(a["suffix"])), {})


def read_tsv(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            c = line.split("\t")
            out[(c[0], c[1], c[2])] = {"state": c[3], "reason": c[8] if len(c) > 8 else ""}
    return out


def restore(tasks: list[Task], root: Path, plan: dict, old: dict):
    """Task states at startup: runs with SHA256SUMS are done; failed and paused stay as they are (only reset sets them back
    to pending); everything else is pending."""
    tiny = set(plan["groups_tiny"])
    for t in tasks:
        d = root / t.model / t.setting / t.run
        g = done_groups(d)
        t.done, t.tiny = len(g), tiny <= g
        o = old.get(t.key())
        if (d / "SHA256SUMS").exists():
            t.state = DONE
        elif o and o["state"] in (FAILED, PAUSED):
            t.state, t.reason = o["state"], o["reason"]


def write_tsv(path: Path, tasks: list[Task], job: str):
    tmp = path.with_suffix(".tmp")
    rows = ["model\tsetting\trun\tstate\tgroups_done\tgroups_total\ttiny\tjob\treason\tupdated"]
    rows += ["\t".join([t.model, t.setting, t.run, t.state, str(t.done), str(t.total), "yes" if t.tiny else "no", job,
                        t.reason.replace("\t", " ").replace("\n", " ")[:300], t.updated]) for t in tasks]
    tmp.write_text("\n".join(rows) + "\n")
    tmp.replace(path)


def worker_count(q: dict, model: str, setting: str) -> int:
    w = (q.get("workers_by_model") or {}).get(model) or q["workers"]
    return int(w["active_v2"] if setting == "active_v2" else w["static"])


def queued_jobs(name: str, me: str = "") -> list | None:
    """Jobs with this name in the queue (running or pending), excluding me; None if the query fails (must not be read as
    "no jobs")."""
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", name, "-o", "%i"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return None if r.returncode != 0 else [x for x in r.stdout.split() if x != me]


class Server:
    """Inference server of one model: its own process group, terminated as a whole on shutdown (the vLLM engine exits together
    with its tensor-parallel workers). tag distinguishes several instances of the same model."""

    def __init__(self, q, model, gpus, port, tag=""):
        self.q, self.model, self.gpus, self.port, self.tag = q, model, gpus, port, tag
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.info = q.jobdir / f"server_{model}{tag}.json"
        self.log_path = q.jobdir / f"server_{model}{tag}.log"
        self.log_from = 0
        self.base_used = {}      # memory used per GPU before the server started (other processes); after shutdown wait until it is back to this level
        self.env_error = ""      # failure caused by the environment

    def desc(self) -> dict:
        return {"id": f"{self.model}@{self.port}", "url": self.url, "gpus": ",".join(self.gpus), "info": str(self.info)}

    def start(self) -> bool:
        ec = self.q.ec
        m = ec["models"][self.model]
        kind = m.get("server", "vllm")
        venv = os.environ.get("QUEUE_VLLM_ENV", ec["open_models"]["vllm_env"])   # the environment variable is for tests only
        if kind == "vllm" and not venv:
            raise SystemExit("vllm_env (the conda environment directory with vLLM) is missing in configs/paths.yaml")
        if kind == "vllm":
            args = vllm_args(ec, self.model, self.port)
            cmd, vpy = [f"{venv}/bin/vllm", *args], f"{venv}/bin/python"
        else:
            args = ["cambrian_server", str(self.port)]
            om = ec["open_models"]
            cmd, vpy = ["bash", om.get("cambrian_serve") or CAMBRIAN_SERVE, str(self.port)], om.get("cambrian_python") or CAMBRIAN_PY
            if not vpy:
                raise SystemExit("cambrian_python (the python of the Cambrian-S environment) is missing in configs/paths.yaml")
        dev = {"CUDA_VISIBLE_DEVICES": ",".join(self.gpus), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        env = server_env(ec, self.model, {**os.environ, **dev}) if kind == "vllm" else {**os.environ, **dev}
        t0 = int(time.time())
        self.base_used = {g["index"]: g["used"] for g in smi_gpus() if g["index"] in self.gpus}
        self.log_from = self.log_path.stat().st_size if self.log_path.exists() else 0
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.q.log(f"{self.model}{self.tag}: GPUs {','.join(self.gpus)} port {self.port}, starting server (memory used before start {self.base_used} MiB)")
        ready = self.wait_ready()
        if ready is None:
            self.check_env()
            return False
        tp = (m.get("vllm") or {}).get("tp", 1)
        with open(self.info, "w") as f:
            subprocess.run([vpy, str(ROOT / "scripts/server_info.py"), self.model, ",".join(self.gpus), str(tp), str(self.port), kind, str(t0), json.dumps(ready), *args],
                           env=env, stdout=f, check=False)
        self.q.log(f"{self.model}: server ready ({ready['ready_after_s']} s)")
        self.q.gpu_snap(f"ready:{self.model}")
        return True

    def wait_ready(self, timeout_s=3600):
        import requests
        t0 = time.time()
        while time.time() - t0 < timeout_s and not self.q.stopping.is_set():
            if self.proc.poll() is not None:
                self.q.log(f"{self.model}{self.tag}: server process exited ({self.proc.returncode}), see server_{self.model}{self.tag}.log")
                return None
            try:
                if requests.get(self.url + "/health", timeout=5).status_code == 200:
                    return {"ready_after_s": round(time.time() - t0, 1), "models": requests.get(self.url + "/v1/models", timeout=10).json()}
            except requests.exceptions.RequestException:
                pass
            self.q.tick.wait(10)
        return None

    def check_env(self) -> str:
        """When the server process has exited, check whether the server log since this start shows an environment cause (ENV_PATTERNS)."""
        if self.proc is None or self.proc.poll() is None or self.env_error:
            return self.env_error
        try:
            with open(self.log_path, errors="replace") as f:
                f.seek(self.log_from)
                self.env_error = env_error_of(f.read())
        except OSError:
            pass
        return self.env_error

    def stop(self):
        if self.proc is None:
            return
        for sig, wait_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 20)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        self.wait_gpus_free()

    def wait_gpus_free(self, timeout_s=180, limit_mib=2000):
        """Wait until this server's GPU memory is released: used memory per GPU back to the pre-start level + limit_mib
    (memory of other processes does not count)."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            smi = smi_gpus()
            used = {g["index"]: g["used"] for g in smi}
            if not smi or all(used.get(g, 0) < self.base_used.get(g, 0) + limit_mib for g in self.gpus):
                return
            self.q.tick.wait(5)
        self.q.log(f"{self.model}: memory of GPUs {','.join(self.gpus)} did not return to the pre-start level ({self.base_used} MiB) within {timeout_s} s")


class Queue:
    def __init__(self, root: Path):
        self.root = root
        self.qdir = queue_dir(root)
        self.plan = json.loads((self.qdir / "plan.json").read_text())
        self.ec = load_eval_config()
        self.q = self.ec["open_models"]["queue"]
        self.local = os.environ.get("QUEUE_LOCAL") == "1" or not os.environ.get("SLURM_JOB_ID")     # local run (no Slurm)
        self.job = os.environ.get("SLURM_JOB_ID") or ("local-" + time.strftime("%Y%m%d-%H%M%S") if os.environ.get("QUEUE_LOCAL") == "1" else "local")
        self.jobdir = root / "_jobs" / self.job
        self.jobdir.mkdir(parents=True, exist_ok=True)
        self.tasks = plan_tasks(self.plan)
        restore(self.tasks, root, self.plan, read_tsv(self.qdir / "tasks.tsv"))
        self.requeued = self.apply_requeue()
        self.lock = threading.RLock()
        self.report_lock = threading.Lock()
        self.stopping = threading.Event()
        self.tick = threading.Event()      # only used for waiting, never set: tick.wait(s) waits s seconds
        self.wake = threading.Event()
        self.procs = {}
        self.smi_start = smi_gpus()
        self.free, self.gpu_note = allocated_gpus(os.environ, self.smi_start)     # only GPUs allocated by Slurm
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                             # CUDA indices match nvidia-smi
        self.n_gpus = len(self.free)
        self.bad, self.env_count, self.card_ok = {}, {}, {}     # GPU -> disabled until / env_error count / usable at the last check
        self.blocked_since = None
        self.cancelled = set()
        self.running, self.launched = set(), set()
        self.ctl, self.batch_end, self.n_inst, self.replica_fail = {}, {}, {}, {}   # multiple instances (see the module docstring)
        self.next_port = 8300
        self.cfg_key = self.config_key()
        self.warned_max = False
        self.t0 = time.time()
        self.stop_at = self.deadline()
        self.no_start_at = self.stop_at - (0 if os.environ.get("QUEUE_STOP_AFTER_S") else float(self.q["no_start_margin_s"]) - float(self.q["stop_margin_s"]))

    def apply_requeue(self) -> list:
        """requeue.tsv (model, setting, run; tab-separated): paused or failed tasks go back to pending with their retry count reset;
        the file is renamed and kept afterwards. A running job does not re-read the task table, so tasks to rerun are listed here
        and picked up when the next job starts."""
        p = self.qdir / "requeue.tsv"
        if not p.exists():
            return []
        want = {tuple(x.split("\t")[:3]) for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")}
        done = []
        for t in self.tasks:
            if (t.model, t.setting, t.run) in want and t.state in (FAILED, PAUSED):
                t.state, t.reason, t.tries = PENDING, "", 0
                done.append(t.key())
        p.rename(self.qdir / f"requeue.done.{self.job}.tsv")
        return done

    # ------------------------------------------------------------------ records
    def log(self, msg):
        line = f"{now()} [{self.job}] {msg}"
        print(line, flush=True)
        with self.lock, open(self.qdir / "events.log", "a") as f:
            f.write(line + "\n")

    def progress(self, msg):
        p = Path(self.plan["report_dir"]) / "open_models_progress.md"
        with self.lock, open(p, "a") as f:
            f.write(f"- {now()} {msg}\n")

    def gpu_snap(self, tag):
        r = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
        with open(self.jobdir / "gpu.csv", "a") as f:
            f.writelines(f"{tag},{x}\n" for x in r.stdout.splitlines())

    def set_state(self, t: Task, state, reason=""):
        with self.lock:
            t.state, t.reason, t.updated = state, reason, now()
            self.flush()
        self.log(f"{t.model} {t.setting} {t.run}: {state}" + (f" ({reason[:200]})" if reason else ""))
        self.wake.set()

    def flush(self):
        with self.lock:
            write_tsv(self.qdir / "tasks.tsv", self.tasks, self.job)

    def refresh(self):
        for t in self.tasks:
            if t.state == RUNNING:
                t.done = len(done_groups(self.root / t.model / t.setting / t.run))
        self.flush()

    # ------------------------------------------------------------------ time
    def deadline(self) -> float:
        """Time to stop: job end time − stop_margin_s. QUEUE_STOP_AFTER_S is only for testing continuation."""
        if os.environ.get("QUEUE_STOP_AFTER_S"):
            return self.t0 + float(os.environ["QUEUE_STOP_AFTER_S"])
        if os.environ.get("QUEUE_LOCAL_HOURS"):
            return self.t0 + float(os.environ["QUEUE_LOCAL_HOURS"]) * 3600 - float(self.q["stop_margin_s"])
        if self.local:
            return self.t0 + 1e9
        try:
            r = subprocess.run(["scontrol", "show", "job", self.job, "-o"], capture_output=True, text=True, check=False)
        except OSError:
            return self.t0 + 1e9
        m = re.search(r"EndTime=(\S+)", r.stdout)
        if not m:
            return self.t0 + 1e9
        end = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
        return end - float(self.q["stop_margin_s"])

    # ------------------------------------------------------------------ tasks
    def of_model(self, model, kind=None, state=None):
        return [t for t in self.tasks if t.model == model and (kind is None or t.kind == kind) and (state is None or t.state == state)]

    def pending(self, model, kinds=MAIN):
        return [t for t in self.tasks if t.model == model and t.kind in kinds and t.state == PENDING]

    def main_left(self) -> bool:
        return any(t.state in (PENDING, RUNNING) for t in self.tasks if t.kind in MAIN)

    @staticmethod
    def app_key(model) -> str:
        return f"{model}[appendix]"

    # ------------------------------------------------------------------ GPUs
    def need_mib(self, model, total) -> float | None:
        """Free memory needed to start this model's server: for vLLM gpu_memory_utilization × total memory + ENV_MARGIN_MIB;
        other servers return None (they require < 2000 MiB used on the GPU)."""
        m = self.ec["models"][model]
        if m.get("server", "vllm") != "vllm":
            return None
        util = float((m.get("vllm") or {}).get("gpu_memory_utilization", self.ec["open_models"]["gpu_memory_utilization"]))
        return util * total + ENV_MARGIN_MIB

    def usable(self, model, cards, smi) -> list:
        """GPUs in cards with enough free memory for this model. No blocking if nvidia-smi is unavailable. Logs only when a
        GPU's state changes."""
        if not smi:
            return list(cards)
        by = {g["index"]: g for g in smi}
        ok = []
        for c in cards:
            g = by.get(c)
            if g is None:
                continue
            need = self.need_mib(model, g["total"])
            free = g["total"] - g["used"]
            good = g["used"] < 2000 if need is None else free >= need
            if good:
                ok.append(c)
            if self.card_ok.get(c) != good:
                if not good:
                    self.log(f"GPU {c}: {free / 1024:.1f} GiB free, not enough for {model} (needs {(need or g['total']) / 1024:.1f} GiB); "
                             f"the {g['used']} MiB in use are not from this job's servers; not starting on this GPU, waiting for it to free up or using another allocated GPU")
                elif c in self.card_ok:
                    self.log(f"GPU {c}: {free / 1024:.1f} GiB free, usable again")
                self.card_ok[c] = good
        return ok

    def mark_bad(self, gpus, why):
        """A GPU with an env_error is disabled for ENV_COOLDOWN_S seconds; from its 2nd env_error on, this job no longer uses it."""
        with self.lock:
            for g in gpus:
                self.env_count[g] = self.env_count.get(g, 0) + 1
                self.bad[g] = float("inf") if self.env_count[g] >= 2 else time.time() + ENV_COOLDOWN_S
        self.log(f"GPUs {','.join(gpus)}: env_error ({why[:200]}), " + "; ".join(
            f"GPU {g} {'no longer used by this job' if self.bad[g] == float('inf') else f'disabled for {ENV_COOLDOWN_S} s'}" for g in gpus))

    def env_fail(self, model, key, gpus, why, kinds):
        """Failure caused by the environment: not counted against the model's retries, tasks stay pending (reason env_error),
        the GPU is disabled, and the model is restarted later on a usable GPU."""
        with self.lock:
            for t in self.pending(model, kinds):
                t.reason = f"env_error: {why[:200]}"
            self.launched.discard(key)
            self.flush()
        self.mark_bad(gpus, why)
        self.log(f"{model}: env_error, not counted against the model's retries; tasks stay pending and the model is restarted later on a usable GPU")
        self.progress(f"{model}: environment error on GPUs {','.join(gpus)} (env_error, not counted against the model's retries): {why[:200]}")

    def log_node(self):
        """At job start, record the node, the GPUs allocated by Slurm and the nvidia-smi output."""
        host = os.uname().nodename
        self.log(f"node {host}; SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', '')}; SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', '')}; "
                 f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}; using GPUs {self.free} ({self.gpu_note})")
        for g in self.smi_start:
            self.log(f"  nvidia-smi GPU {g['index']} ({g['bus']}) used {g['used']} / {g['total']} MiB" + (", allocated to this job" if g["index"] in self.free else ""))
        try:
            a = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60, check=False).stdout
            b = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv"],
                               capture_output=True, text=True, timeout=60, check=False).stdout
            (self.jobdir / "nvidia_smi_start.txt").write_text(f"{now()} {host}\n{a}\n{b}")
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"  nvidia-smi query failed: {e}")

    def leave_node(self):
        """The allocated GPUs stay unusable: stop, and the continuation job gets --exclude for the current node (a continuation job
        already queued has no --exclude, so it is cancelled and resubmitted at the end)."""
        host = os.uname().nodename
        if self.local:
            self.log(f"the --gpus GPUs {self.free} have been unusable for {self.foreign_wait():.0f} s (memory held by other processes or environment errors); stopping. Free the memory and rerun the same command")
            self.progress(f"local run: none of the --gpus GPUs {self.free} is usable, stopping (see events.log)")
            self.stopping.set()
            return
        why = "; ".join(f"GPU {g}: " + ("disabled after env_error" if self.bad.get(g, 0) > time.time() else "not enough free memory") for g in self.free)
        self.log(f"the allocated GPUs {self.free} have been unusable for {self.foreign_wait():.0f} s ({why}); stopping. The continuation job will not use node {host}")
        self.progress(f"job {self.job}: none of the allocated GPUs {self.free} on node {host} is usable ({why}); stopping, continuation job gets --exclude {host}")
        nodes = [n for n in self.plan.get("exclude_nodes", []) if n != host] + [host]
        self.plan["exclude_nodes"] = nodes[-4:]
        self.save_plan()
        for j in self.queued_jobs() or []:
            subprocess.run(["scancel", j], check=False)
            self.cancelled.add(j)
            self.log(f"queued continuation job {j} has no --exclude: cancelled, will be resubmitted at the end")
        self.stopping.set()

    def foreign_wait(self) -> float:
        return float(self.q.get("foreign_wait_s", 900))

    def config_key(self):
        try:
            st = (ROOT / "configs/eval.yaml").stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def reload_config(self):
        """Re-read the config file if it changed (start order, tensor parallelism etc. take effect the next time a model is
        started; running servers are not affected)."""
        k = self.config_key()
        if k != self.cfg_key:
            try:
                ec = load_eval_config()
            except Exception as e:  # noqa: BLE001
                self.log(f"re-reading the config failed, keeping the old config: {str(e)[:200]}")
                return
            self.cfg_key, self.ec, self.q = k, ec, ec["open_models"]["queue"]
            self.log(f"config re-read: start order {self.order()}")

    def order(self) -> list:
        o = [m for m in self.q.get("order", []) if m in self.plan["models"]]
        return o + [m for m in self.plan["models"] if m not in o]

    def tp(self, model) -> int:
        return int((self.ec["models"][model].get("vllm") or {}).get("tp", 1))

    def held(self) -> set:
        p = self.qdir / "hold_gpus"
        try:
            return set(p.read_text().split()) if p.exists() else set()
        except OSError:
            return set()

    def workers_of(self, model) -> dict:
        """Concurrency per setting on each instance: from the config, overridable by _queue/workers.json (can be changed while
        running, synced every minute)."""
        w = dict((self.q.get("workers_by_model") or {}).get(model) or self.q["workers"])
        try:
            o = json.loads((self.qdir / "workers.json").read_text()).get(model) if (self.qdir / "workers.json").exists() else None
            if o:
                w.update({k: int(v) for k, v in o.items()})
        except (OSError, ValueError, AttributeError):
            pass
        return w

    def ctl_path(self, model) -> Path:
        return self.jobdir / f"ctl_{model}.json"

    def write_control(self, model):
        with self.lock:
            servers = list(self.ctl.get(model) or [])
            doc = {"model": model, "workers": self.workers_of(model), "servers": [x.desc() for x in servers], "updated": now()}
            p = self.ctl_path(model)
            old = json.loads(p.read_text()) if p.exists() else None
            if old and {k: v for k, v in old.items() if k != "updated"} == {k: v for k, v in doc.items() if k != "updated"}:
                return
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            tmp.replace(p)
        if old and old.get("workers") != doc["workers"]:
            self.log(f"{model}: concurrency per instance {old.get('workers')} -> {doc['workers']}")

    def sync_controls(self):
        for m in list(self.ctl):
            self.write_control(m)

    def replica_wanted(self, model) -> bool:
        """Start another instance: evaluation tasks are running, more groups remain than episodes in flight, and neither the
        instance count nor the failure count has reached its limit."""
        if model not in self.ctl or self.replica_fail.get(model, 0) >= 2:
            return False
        if self.n_inst.get(model, 0) >= int(self.q.get("max_instances", 4)):
            return False
        tasks = [t for t in self.of_model(model, "full") if t.state == RUNNING]
        if not tasks:
            return False
        w = self.workers_of(model)
        rem = sum(max(0, t.total - t.done) for t in tasks)
        inflight = self.n_inst.get(model, 1) * sum(int(w["active_v2"] if t.setting == "active_v2" else w["static"]) for t in tasks)
        return rem > inflight

    def spawn(self, t: Task, srv: Server, groups, seal) -> int:
        mode, budget, _ = SETTINGS[t.setting]
        cmd = [PY, "-m", "actspace_eval.llm_run", "--model", t.model, "--mode", mode, "--budget", str(budget), "--run-seed", str(t.seed),
               "--groups", *groups, "--name", t.run, "--base-url", srv.url, "--out-root", str(self.root),
               "--workers", str(3 if t.kind == "smoke" else worker_count(self.q, t.model, t.setting)),
               "--server-info", str(srv.info), "--resume"] + (["--seal"] if seal else []) \
            + (["--control", str(self.ctl_path(t.model))] if t.kind == "full" and t.model in self.ctl else []) \
            + (list(appendix_of(self.plan, t).get("llm_args") or []) if t.kind == "appendix" else [])
        with open(self.jobdir / f"run_{t.model}_{t.setting}_{t.run}.log", "a") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        with self.lock:
            self.procs[t.key()] = p
        rc = p.wait()
        with self.lock:
            self.procs.pop(t.key(), None)
        return rc

    def run_task(self, t: Task, srv: Server):
        self.set_state(t, RUNNING)
        d = self.root / t.model / t.setting / t.run
        if t.kind == "smoke":
            phases = [(self.plan["groups_smoke"], True, False)]
        elif t.kind == "appendix":
            phases = [(self.plan["groups_all"], True, False)]
        else:
            phases = [] if t.tiny else [(self.plan["groups_tiny"], False, True)]
            phases.append((self.plan["groups_all"], True, False))
        try:
            for groups, seal, is_tiny in phases:
                rc = self.spawn(t, srv, groups, seal)
                t.done = len(done_groups(d))
                if self.stopping.is_set():
                    self.set_state(t, DONE if (d / "SHA256SUMS").exists() else PENDING, "" if (d / "SHA256SUMS").exists() else "stopped at the time limit, the continuation job resumes it")
                    return
                st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
                if rc == 0 and st.get("status") == "complete":
                    if is_tiny:
                        t.tiny = True
                        self.after_tiny(t.model)
                    continue
                if st.get("status") == "not_applicable" and rc == 0:
                    msg = f"not_applicable: {json.dumps(st.get('na_reason'), ensure_ascii=False)[:280]}"
                    self.set_state(t, PAUSED, msg)
                    self.progress(f"**Decision needed**: {t.model} {t.setting} {t.run} paused, {msg}")
                    return
                env = srv.check_env()
                if env:                         # server exited for an environment reason: not counted as a retry
                    self.set_state(t, PENDING, f"env_error: {env[:200]}")
                    return
                t.tries += 1
                msg = f"exit {rc}, status {st.get('status')}: {str(st.get('error'))[:220]}"
                self.set_state(t, FAILED if t.tries >= int(self.q["max_server_tries"]) else PENDING, msg)
                return
            self.set_state(t, DONE)
        except Exception:  # noqa: BLE001  an error in one task does not affect the others
            t.tries += 1
            self.set_state(t, FAILED, traceback.format_exc()[-280:])

    def run_batch(self, tasks, srv):
        th = [threading.Thread(target=self.run_task, args=(t, srv), daemon=True) for t in tasks]
        for x in th:
            x.start()
        for x in th:
            x.join()
        self.gpu_snap(f"done:{tasks[0].model}")

    def gate(self, model, srv) -> bool:
        """smoke_gate: run the smoke test first; release only if the tokens per image agree."""
        prefix = (self.plan.get("smoke_gate") or {}).get(model)
        if not prefix:
            return True
        path = self.qdir / f"gate_{model}.json"
        if path.exists() and json.loads(path.read_text()).get("pass"):
            return True
        todo = self.of_model(model, "smoke", PENDING)
        if todo:
            self.run_batch(todo, srv)
        if self.stopping.is_set():
            return False
        smoke = self.of_model(model, "smoke")
        if any(t.state != DONE for t in smoke):
            if not self.of_model(model, "smoke", PENDING):
                self.pause_full(model, "smoke runs not all done, evaluation tasks not started")
            return False
        extra = bool(self.plan.get("gate_extra"))     # also check complete records; the parse rate is recorded only
        g = gate_result(self.root, model, f"{prefix}_r0", [t.setting for t in smoke], len(self.plan["groups_smoke"]), extra=extra)
        path.write_text(json.dumps(g, ensure_ascii=False, indent=1))
        self.log(f"{model}: smoke gate {'passed' if g['pass'] else 'failed'}: {g['tokens']}")
        c = g.get("checks")
        more = (f"; parsed {c['parse_rate']:.0%}, retried calls {c['retried_calls']}/{c['calls']}, max input {c['max_input']}" if c else "")
        if c and c.get("order"):          # image/text interleaving order (content_order)
            n = {k: sum(v.get(k, 0) for v in c["order"].values()) for k in ("interleaved", "images_first", "mismatch", "text_not_found", "no_render")}
            more += f"; image/text interleaving: {n['interleaved']} turn-0 calls correct" + "".join(f", {k} {v}" for k, v in n.items() if k != "interleaved" and v)
        self.progress(f"{model} smoke (`{prefix}_r0`): tokens per image {g['tokens']}{more}; {'passed, starting the evaluation runs' if g['pass'] else '**failed, evaluation tasks paused, decision needed**: ' + '; '.join(g.get('why') or ['tokens per image differ'])}")
        if not g["pass"]:
            self.pause_full(model, "smoke gate failed: " + "; ".join(g.get("why") or [f"tokens per image differ: {g['tokens']}"]))
        return g["pass"]

    def pause_full(self, model, reason):
        for t in self.of_model(model, "full", PENDING):
            self.set_state(t, PAUSED, reason)

    def report(self, model, what):
        """Write score tables (subprocess): a failure is only logged and does not affect the tasks."""
        with self.report_lock:
            cmd = [PY, "-m", "actspace_eval.report_open_full", what, "--root", str(self.root), "--prefix", self.plan["prefix"], "--report-dir", self.plan["report_dir"]]
            if model:
                cmd += ["--model", model]
            if what in ("final", "tiny_all") and self.plan.get("final_queues"):
                cmd += ["--queues", *self.plan["final_queues"]]
            names = self.plan.get("report_names") or {}      # summary report names of an independent queue
            if what in names:
                cmd += ["--name", names[what]]
            if what in ("final", "tiny_all") and self.plan.get("report_title"):
                cmd += ["--title", self.plan["report_title"]]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
        with open(self.jobdir / "reports.log", "a") as f:
            f.write(f"{now()} {what} {model}\n{r.stdout}\n{r.stderr}\n")
        if r.returncode != 0:
            self.log(f"{what} {model}: report failed, see reports.log")
            return None
        return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""

    def after_tiny(self, model):
        with self.lock:
            full = self.of_model(model, "full")
            ready = all(t.tiny or t.state in (FAILED, PAUSED) for t in full) and model not in self.plan.setdefault("tiny_reported", [])
            if ready:
                self.plan["tiny_reported"].append(model)
                self.save_plan()
        if ready:
            line = self.report(model, "tiny")
            if line is not None:
                self.progress(f"**Note**: tiny draft of {model} written: {line}")
            with self.lock:
                last = set(self.plan["tiny_reported"]) >= set(self.plan["models"]) and not self.plan.get("tiny_all")
                if last:
                    self.plan["tiny_all"] = True
                    self.save_plan()
            if last:
                line = self.report(None, "tiny_all")
                self.progress(f"**Note**: tiny runs of all models are done, summary draft written. {line or ''}")

    def save_plan(self):
        tmp = self.qdir / "plan.tmp"
        tmp.write_text(json.dumps(self.plan, ensure_ascii=False, indent=1))
        tmp.replace(self.qdir / "plan.json")

    def finish_model(self, model):
        tasks = self.of_model(model, "full")
        if not tasks or any(t.state in (PENDING, RUNNING) for t in tasks):     # smoke-only queues write no score tables and no paper tables
            return
        n = sum(t.state == DONE for t in tasks)
        line = self.report(model, "model")
        bad = [f"{t.setting} ({t.state})" for t in tasks if t.state != DONE]
        self.progress(f"{model} finished: {n}/{len(tasks)} runs done" + (f", unfinished {', '.join(bad)}" if bad else "") + (f". {line}" if line else ""))
        if self.plan.get("paper_table"):
            self.paper_table()

    def paper_table(self):
        """Regenerate the paper tables after each model (all open and closed models, with self-check). A failure is only logged."""
        with self.report_lock:
            r = subprocess.run([PY, "-m", "actspace_eval.report_paper", "--root", str(self.root), "--out-dir", str(Path(self.plan["report_dir"]) / "paper_table")],
                               capture_output=True, text=True, cwd=ROOT, check=False)
        with open(self.jobdir / "reports.log", "a") as f:
            f.write(f"{now()} paper_table\n{r.stdout}\n{r.stderr}\n")
        line = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else "failed, see reports.log"
        self.log(f"paper tables: {line}")
        if r.returncode == 0:
            self.progress(f"paper tables regenerated (`paper_table/`): {line.split('; ', 1)[-1]}")

    def model_thread(self, model, gpus, port, phase="main"):
        """phase main: smoke (gate) + evaluation tasks; appendix: appendix comparison tasks (after all main-table tasks have
        ended). Failures caused by the environment: see env_fail."""
        kinds = MAIN if phase == "main" else ("appendix",)
        key = model if phase == "main" else self.app_key(model)
        tries, env = 0, ""
        try:
            if phase == "main":
                self.after_tiny(model)      # the previous job may have stopped right when tiny completed
            while not self.stopping.is_set() and self.pending(model, kinds):
                if tries >= int(self.q["max_server_tries"]):
                    for t in self.pending(model, kinds):
                        self.set_state(t, FAILED, t.reason or f"server failed to start {tries} times, see server_{model}.log")
                    break
                tries += 1
                srv = Server(self, model, gpus, port, tag="" if phase == "main" else "_appendix")
                try:
                    if srv.start():
                        if phase != "main":
                            todo = self.pending(model, kinds)
                            if todo:
                                self.run_batch(todo, srv)
                        elif self.gate(model, srv):
                            todo = self.of_model(model, "full", PENDING)
                            if todo:
                                ev = threading.Event()
                                with self.lock:
                                    self.batch_end[model] = ev
                                    self.ctl[model] = [srv]
                                    self.n_inst[model] = 1
                                self.write_control(model)
                                try:
                                    self.run_batch(todo, srv)
                                finally:
                                    ev.set()           # the other instances shut down with it
                                    with self.lock:
                                        self.ctl.pop(model, None)
                    if srv.check_env():
                        tries -= 1
                        env = srv.env_error
                        break
                finally:
                    srv.stop()
            if env:
                self.env_fail(model, key, gpus, env, kinds)
            elif phase == "main":
                self.finish_model(model)
            elif not self.pending(model, kinds):
                line = self.report(model, "model")
                self.progress(f"{model} appendix comparison finished: {sum(t.state == DONE for t in self.of_model(model, 'appendix'))}/{len(self.of_model(model, 'appendix'))} runs done"
                              + (f". {line}" if line else ""))
        except Exception:  # noqa: BLE001
            self.log(f"{model}: scheduler error\n{traceback.format_exc()}")
            for t in [t for t in self.tasks if t.model == model and t.kind in kinds and t.state in (PENDING, RUNNING)]:
                self.set_state(t, FAILED, "scheduler error, see events.log")
        finally:
            with self.lock:
                self.free += gpus
                self.running.discard(key)
            self.wake.set()

    def replica_thread(self, model, gpus, port, ev):
        """Another server instance of the same model: added to the control file once ready; stopped and its GPUs released when
        the batch ends (or on stop)."""
        srv = Server(self, model, gpus, port, tag=f"_p{port}")
        try:
            if not srv.start():
                if srv.env_error:
                    self.mark_bad(gpus, srv.env_error)
                else:
                    with self.lock:
                        self.replica_fail[model] = self.replica_fail.get(model, 0) + 1
                return
            with self.lock:
                if ev.is_set() or self.stopping.is_set() or model not in self.ctl:
                    return
                self.ctl[model].append(srv)
            self.write_control(model)
            self.log(f"{model}: instance {srv.desc()['id']} (GPUs {','.join(gpus)}) joined, {len(self.ctl.get(model, []))} in total")
            while not ev.is_set() and not self.stopping.is_set():
                if srv.proc.poll() is not None:
                    self.log(f"{model}: server of instance {srv.desc()['id']} exited ({srv.proc.returncode}), removed from the instance list")
                    with self.lock:
                        self.replica_fail[model] = self.replica_fail.get(model, 0) + 1
                    break
                ev.wait(30)
        except Exception:  # noqa: BLE001
            self.log(f"{model}: instance error\n{traceback.format_exc()}")
        finally:
            with self.lock:
                if srv in self.ctl.get(model, []):
                    self.ctl[model].remove(srv)
                live = model in self.ctl
                self.n_inst[model] = max(1, self.n_inst.get(model, 1) - 1)
            if live:
                self.write_control(model)
            srv.stop()
            with self.lock:
                self.free += gpus
            self.log(f"{model}: instance {srv.desc()['id']} stopped, GPUs {','.join(gpus)} freed")
            self.wake.set()

    # ------------------------------------------------------------------ main loop
    def launch(self):
        """In start order: start a model not yet started if it fits; add instances to running models when needed (see the
        module docstring)."""
        if time.time() > self.no_start_at:
            return
        self.reload_config()
        held = self.held()
        main_left = self.main_left()
        smi = None
        for m in self.order():
            i = self.plan["models"].index(m)
            tp = self.tp(m)
            new = m not in self.launched and bool(self.pending(m, MAIN))
            more = not new and m in self.running and self.replica_wanted(m)
            app = not (new or more) and not main_left and self.app_key(m) not in self.launched and bool(self.pending(m, ("appendix",)))
            if not (new or more or app):
                continue
            with self.lock:
                t_now = time.time()
                avail = [g for g in self.free if g not in held and self.bad.get(g, 0) <= t_now]
                if tp > len(avail):
                    continue
                if smi is None:
                    smi = smi_gpus()
                ok = self.usable(m, avail, smi)       # start on a GPU only if it has enough free memory
                if tp > len(ok):
                    continue
                gpus = ok[:tp]
                self.free = [g for g in self.free if g not in gpus]
                for g in gpus:                        # memory changes once the server starts; the next model queries again
                    smi = [x for x in smi if x["index"] != g]
                if new:
                    self.running.add(m)
                    self.launched.add(m)
                elif app:
                    self.running.add(self.app_key(m))
                    self.launched.add(self.app_key(m))
                else:
                    self.n_inst[m] = self.n_inst.get(m, 1) + 1
                    port, self.next_port = self.next_port, self.next_port + 1
                    ev = self.batch_end[m]
            if new:
                threading.Thread(target=self.model_thread, args=(m, gpus, 8100 + i), daemon=True).start()
            elif app:
                self.log(f"{m}: all main-table tasks have ended, GPUs {','.join(gpus)} run the appendix comparison")
                threading.Thread(target=self.model_thread, args=(m, gpus, 8200 + i, "appendix"), daemon=True).start()
            else:
                self.log(f"{m}: GPUs {','.join(gpus)} free, starting instance {self.n_inst[m]}")
                threading.Thread(target=self.replica_thread, args=(m, gpus, port, ev), daemon=True).start()

    def begin_stop(self, why):
        if self.stopping.is_set():
            return
        self.log(f"stopping: {why}")
        self.stopping.set()
        with self.lock:
            procs = list(self.procs.values())
        for p in procs:
            p.terminate()
        self.kill_at = time.time() + 120

    def queued_jobs(self):
        """Other jobs with the same name in the queue (running or pending); None if the query fails. Local runs have no jobs."""
        if self.local:
            return []
        return queued_jobs(os.environ.get("SLURM_JOB_NAME", JOB_NAME), self.job)

    def ensure_next(self):
        """Make sure exactly one continuation job is queued."""
        if self.local:
            return
        q = self.queued_jobs()
        if q is None:
            self.log("queue query failed; not submitting a continuation job now, will check again later")
            return
        if [j for j in q if j not in self.cancelled]:
            return
        if len(self.plan.get("jobs", [])) >= MAX_JOBS:
            if not self.warned_max:
                self.warned_max = True
                self.progress(f"**The queue has used all {MAX_JOBS} jobs; no continuation job submitted**, needs attention")
            return
        r = subprocess.run(["sbatch", "--parsable", f"--dependency=afterany:{self.job}", *self.plan.get("sbatch_args", []), *exclude_args(self.plan),
                            "scripts/run_open_queue.sbatch"], capture_output=True, text=True, cwd=ROOT, check=False)
        self.log(f"continuation job: {r.stdout.strip()} {r.stderr.strip()[:300]}")

    def cancel_next(self):
        for j in self.queued_jobs() or []:
            subprocess.run(["scancel", j], check=False)
            self.log(f"queue finished, cancelling continuation job {j}")

    def catch_up(self):
        """The previous job died before writing its reports: write the missing tiny drafts and score tables of finished models."""
        for m in self.plan["models"]:
            if self.pending(m, MAIN):
                continue
            self.after_tiny(m)
            if not (Path(self.plan["report_dir"]) / "open" / f"{m}.md").exists():
                self.finish_model(m)

    def finish_queue(self):
        if not (self.qdir / "DONE").exists():
            line = self.report(None, "final")
            self.progress(f"queue finished: {sum(t.state == DONE for t in self.tasks)}/{len(self.tasks)} tasks done. {line or ''}")
            (self.qdir / "DONE").write_text(now() + "\n")

    def run(self):
        if not any(t.state in (PENDING, RUNNING) for t in self.tasks):
            self.log("no pending tasks, exiting")
            self.catch_up()
            self.finish_queue()
            return
        self.log_node()
        if not self.free:
            self.log(f"no usable GPUs ({self.gpu_note}), exiting")
            self.progress(f"**Needs attention**: job {self.job} got no GPUs allocated by Slurm ({self.gpu_note}) and ran no tasks")
            return
        (self.qdir / "DONE").unlink(missing_ok=True)
        if self.requeued:
            self.flush()
            self.log(f"requeue.tsv: {len(self.requeued)} tasks set back to pending {self.requeued}")
        signal.signal(signal.SIGTERM, lambda *_: self.begin_stop("received SIGTERM"))
        if self.local:      # local run: Ctrl-C acts like SIGTERM, waiting for records and stopping the servers (servers run in their own process group and do not get the terminal's Ctrl-C)
            signal.signal(signal.SIGINT, lambda *_: self.begin_stop("received Ctrl-C"))
        self.plan.setdefault("jobs", []).append({"job": self.job, "host": os.uname().nodename, "gpus": self.n_gpus, "code": str(ROOT), "started": now(),
                                                 **({"takeover": True} if os.environ.get("QUEUE_TAKEOVER") else {})})
        self.save_plan()
        self.ensure_next()
        self.catch_up()
        self.log(f"start: {sum(t.state == DONE for t in self.tasks)}/{len(self.tasks)} done, GPUs {self.free}, {(self.stop_at - time.time()) / 3600:.1f} h until stop")
        last = mem = 0
        checked = time.time()
        while True:
            if time.time() >= self.stop_at:
                self.begin_stop("approaching the job time limit")
            if self.stopping.is_set() and time.time() > getattr(self, "kill_at", 0):
                with self.lock:
                    for p in self.procs.values():
                        p.kill()
            if not self.stopping.is_set():
                self.launch()
            if time.time() - last > 60:
                self.refresh()
                self.sync_controls()
                last = time.time()
            if time.time() - checked > 600:     # resubmit if the continuation job was cancelled or its submission failed
                self.ensure_next()
                checked = time.time()
            if time.time() - mem > 1800:
                r = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "rss="], capture_output=True, text=True, check=False)
                self.log(f"memory: {sum(int(x) for x in r.stdout.split()) / 2**20:.0f} GiB in total for this user's processes; GPUs {self.free} free; running {sorted(self.running)}")
                mem = time.time()
            with self.lock:
                idle = not self.running
            if idle and not self.stopping.is_set() and self.launchable():
                # there are startable models but none started: none of the allocated GPUs is usable (memory held by others, or disabled after env_error)
                if self.blocked_since is None:
                    self.blocked_since = time.time()
                    self.log(f"none of the allocated GPUs {self.free} is usable now, waiting at most {self.foreign_wait():.0f} s")
                elif time.time() - self.blocked_since >= self.foreign_wait():
                    self.leave_node()
            else:
                self.blocked_since = None
            if idle and (self.stopping.is_set() or not self.launchable()):
                break
            self.wake.wait(20)
            self.wake.clear()
        self.refresh()
        self.plan["jobs"][-1].update(ended=now(), seconds=round(time.time() - self.t0))
        self.save_plan()
        pending = [t for t in self.tasks if t.state in (PENDING, RUNNING) and t.kind in MAIN]
        if pending:
            self.ensure_next()
        else:
            for t in self.tasks:              # appendix comparisons only use the time left in this job
                if t.kind == "appendix" and t.state in (PENDING, RUNNING):
                    self.set_state(t, PAUSED, "appendix comparison: job time used up before it finished (no continuation)")
            self.cancel_next()
            self.finish_queue()
        self.log(f"end: {len(pending)} pending")

    def launchable(self):
        """Is there still a model that can be started in this job?"""
        if time.time() > self.no_start_at:
            return False
        if any(m not in self.launched and self.pending(m, MAIN) for m in self.plan["models"]):
            return True
        return not self.main_left() and any(self.app_key(m) not in self.launched and self.pending(m, ("appendix",)) for m in self.plan["models"])


def takeover_guard(qdir: Path) -> str:
    """The taken-over job is still alive but its takeover step (open_takeover) is gone: the old scheduler is still frozen and the
    GPUs would stay idle, so cancel the job and let the continuation job take over."""
    p = qdir / "takeover.json"
    if not p.exists():
        return ""
    t = json.loads(p.read_text())
    try:
        r = subprocess.run(["squeue", "-h", "-s", "-j", str(t["job"]), "-o", "%j"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return "job step query failed"
    if r.returncode != 0 and "Invalid job id" not in r.stderr:
        return "job step query failed"
    steps = r.stdout.split() if r.returncode == 0 else []
    if not steps:
        p.rename(qdir / f"takeover.done.{t['job']}.json")
        return f"taken-over job {t['job']} has ended"
    if "open_takeover" in steps:
        return f"taken-over job {t['job']} is fine"
    subprocess.run(["scancel", str(t["job"])], check=False)
    p.rename(qdir / f"takeover.done.{t['job']}.json")
    return f"**takeover step is gone: cancelled job {t['job']}, the continuation job takes over**"


def cmd_init(a):
    ec = load_eval_config()
    from .data import FrozenData
    q = ec["open_models"]["queue"]
    root = Path(a.root)
    qdir = queue_dir(root)
    qdir.mkdir(parents=True, exist_ok=True)
    tiny = a.tiny_groups or list(ec["llm"]["tiny_groups"])
    gall = a.groups or list(FrozenData(verify=False).group_ids)
    seeds = [int(x) for x in a.seeds.split(",")]
    old = json.loads((qdir / "plan.json").read_text()) if (qdir / "plan.json").exists() else {}
    if old and old.get("prefix") == a.prefix:     # adding runs: keep the existing seeds and job records
        seeds = sorted(set(old["seeds"]) | set(seeds))
    plan = {**old, "prefix": a.prefix, "seeds": seeds, "models": a.models or list(q["order"]), "settings": a.settings or list(SETTINGS),
            "groups_tiny": tiny, "groups_all": tiny + [g for g in gall if g not in set(tiny)], "groups_smoke": list(ec["llm"]["smoke_groups"]),
            "smoke_gate": {k: v for k, v in (q.get("smoke_gate") or {}).items() if k in (a.models or q["order"])},
            "report_dir": a.report_dir, "sbatch_args": a.sbatch_args or []}
    for k in ("job_name", "report_title"):
        if getattr(a, k, None):
            plan[k] = getattr(a, k)
    if a.report_names:
        plan["report_names"] = dict(x.split("=", 1) for x in a.report_names)
    if a.final_queues:
        plan["final_queues"] = a.final_queues
    if a.gate_extra:
        plan["gate_extra"] = True
    if a.paper_table:
        plan["paper_table"] = True
    (qdir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    tasks = plan_tasks(plan)
    restore(tasks, root, plan, read_tsv(qdir / "tasks.tsv"))
    write_tsv(qdir / "tasks.tsv", tasks, "-")
    rd = Path(a.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / "open_models_progress.md"
    if not p.exists():
        p.write_text("# Open-model evaluation progress\n\nTask states: `<results>/_queue/tasks.tsv`.\n\n")
    print(f"{len(tasks)} tasks -> {qdir / 'tasks.tsv'}")


def local_env(ec: dict):
    """Child process environment for local runs (same settings as scripts/run_open_queue.sbatch)."""
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([str(ROOT / "src"), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if ec["open_models"].get("hf_home"):
        os.environ["HF_HOME"] = str(ec["open_models"]["hf_home"])
    om = ec["open_models"]                       # paths needed by the Cambrian-S server (scripts/cambrian/serve_cambrian.sh)
    if om.get("cambrian_python"):
        os.environ.setdefault("CAMBRIAN_PYTHON", str(om["cambrian_python"]))
    if om.get("cambrian_repo"):
        os.environ.setdefault("CAMBRIAN_REPO", str(om["cambrian_repo"]))
    if (ec["models"].get("Cambrian-S-7B") or {}).get("path"):
        os.environ.setdefault("CAMBRIAN_MODEL", str(ec["models"]["Cambrian-S-7B"]["path"]))
    tag = f"{os.environ.get('USER', 'u')}_{os.getpid()}"
    os.environ.setdefault("VLLM_CACHE_ROOT", f"/tmp/{tag}_vllm_cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/{tag}_inductor")


def cmd_run_local(a):
    """Local run (see the module docstring): write plan.json if --models is given (or there is no plan.json yet), then run the
    queue in this process."""
    ec = load_eval_config()
    root = Path(a.root)
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        os.environ.pop(k)
    os.environ.update(QUEUE_LOCAL="1", QUEUE_DIR=a.queue_dir, CUDA_VISIBLE_DEVICES=a.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if a.hours:
        os.environ["QUEUE_LOCAL_HOURS"] = str(a.hours)
    local_env(ec)
    qdir = queue_dir(root)
    if a.models or not (qdir / "plan.json").exists():
        report_dir = a.report_dir or str(ROOT / "reports" / "eval")
        cmd_init(argparse.Namespace(
            root=str(root), prefix="full", seeds=a.seeds, report_dir=report_dir, models=a.models, settings=a.settings, groups=None, tiny_groups=None,
            sbatch_args=None, job_name=None, report_names=[f"final=local_{a.queue_dir.strip('_')}_full.md", f"tiny_all=local_{a.queue_dir.strip('_')}_tiny.md"],
            report_title="Local run: open models", final_queues=[a.queue_dir], gate_extra=True, paper_table=True))
        if a.smoke_only or a.appendix is not None:
            plan = json.loads((qdir / "plan.json").read_text())
            if a.smoke_only:
                plan.update(smoke_only=True, paper_table=False)
            if a.appendix is not None:
                plan["appendix"] = [{"model": m, "suffix": "_nothink", "llm_args": ["--no-thinking"]} for m in a.appendix]
            (qdir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
            write_tsv(qdir / "tasks.tsv", (lambda t: (restore(t, root, plan, read_tsv(qdir / "tasks.tsv")), t)[1])(plan_tasks(plan)), "-")
    Queue(root).run()


def cmd_status(a):
    from collections import Counter
    root = Path(a.root)
    rows = [x.split("\t") for x in (queue_dir(root) / "tasks.tsv").read_text().splitlines()[1:]]
    by = {}
    for c in rows:
        by.setdefault(c[0], Counter())[c[3]] += 1
    for m, c in by.items():
        grp = sum(int(r[4]) for r in rows if r[0] == m)
        tot = sum(int(r[5]) for r in rows if r[0] == m)
        print(f"{m:32} " + " ".join(f"{k} {v}" for k, v in c.items()) + f"  groups {grp}/{tot}")
    for c in rows:
        if c[3] in (FAILED, PAUSED):
            print(f"{c[3]} {c[0]} {c[1]} {c[2]}: {c[8]}")


def cmd_watchdog(a):
    """Daily check: resubmit if no job is running or queued and tasks remain. If the query fails, submit nothing and wait for
    the next check."""
    root = Path(a.root)
    qdir = queue_dir(root)
    plan = json.loads((qdir / "plan.json").read_text())
    rows = [x.split("\t") for x in (qdir / "tasks.tsv").read_text().splitlines()[1:]]
    left = sum(c[3] in (PENDING, RUNNING) for c in rows)
    q = queued_jobs(plan.get("job_name", JOB_NAME))
    msg = f"unfinished {left}, paused or failed {sum(c[3] in (FAILED, PAUSED) for c in rows)}, jobs in queue {q}"
    g = takeover_guard(qdir)
    if g:
        msg += f"; {g}"
        if g.startswith("**"):
            with open(Path(plan["report_dir"]) / "open_models_progress.md", "a") as f:
                f.write(f"- {now()} daily check: {g}\n")
    if left and q == []:
        r = subprocess.run(["sbatch", "--parsable", *plan.get("sbatch_args", []), *exclude_args(plan), "scripts/run_open_queue.sbatch"],
                           capture_output=True, text=True, cwd=ROOT, check=False)
        job = r.stdout.strip() or "submission failed: " + r.stderr.strip()[:200]
        with open(Path(plan["report_dir"]) / "open_models_progress.md", "a") as f:
            f.write(f"- {now()} daily check: no job in the queue, {left} tasks unfinished, continuation resubmitted ({job})\n")
        msg += f"; resubmitted {job}"
    with open(qdir / "watchdog.log", "a") as f:
        f.write(f"{now()} {msg}\n")
    print(msg)


def cmd_reset(a):
    root = Path(a.root)
    p = queue_dir(root) / "tasks.tsv"
    lines = p.read_text().splitlines()
    out = [lines[0]]
    n = 0
    for x in lines[1:]:
        c = x.split("\t")
        if c[0] == a.model and (a.setting is None or c[1] == a.setting) and c[3] in (FAILED, PAUSED):
            c[3], c[8] = PENDING, ""
            n += 1
        out.append("\t".join(c))
    p.write_text("\n".join(out) + "\n")
    print(f"{n} tasks set back to pending")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("QUEUE_ROOT") or str(load_eval_config()["open_models"]["results_root"]))
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init")
    p.add_argument("--prefix", required=True)
    p.add_argument("--seeds", default="0")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--models", nargs="*")
    p.add_argument("--settings", nargs="*")
    p.add_argument("--groups", nargs="*")
    p.add_argument("--tiny-groups", nargs="*")
    p.add_argument("--sbatch-args", nargs="*", help="extra sbatch arguments for continuation jobs (GPU count, job name, etc. of an independent queue)")
    p.add_argument("--job-name", help="job name (the daily check looks it up in the queue; default open_queue)")
    p.add_argument("--report-names", nargs="*", help="summary report file names, e.g. final=ext_full.md tiny_all=ext_tiny.md")
    p.add_argument("--report-title", help="title prefix of the summary reports")
    p.add_argument("--final-queues", nargs="*", help="queue directories included in the summary reports (default only _queue)")
    p.add_argument("--gate-extra", action="store_true", help="smoke gate also checks complete records (the parse rate is recorded only, not a reason to pause)")
    p.add_argument("--paper-table", action="store_true", help="regenerate the paper tables after each model")
    sp.add_parser("run")
    p = sp.add_parser("run-local", help="local run (no Slurm; see the module docstring)")
    p.add_argument("--gpus", required=True, help="GPUs to use (nvidia-smi indices, comma-separated)")
    p.add_argument("--models", nargs="*", help="models to run (default: keep the existing plan.json; required the first time)")
    p.add_argument("--settings", nargs="*", help="settings (default all 8; for adding repeats give only active_v2 random_walk_40)")
    p.add_argument("--seeds", default="0", help="run numbers, e.g. 0 or 1,2 (repeats of Active / random_walk)")
    p.add_argument("--queue-dir", default="_queue_local", help="queue directory name (under the results directory)")
    p.add_argument("--report-dir", help="report directory (default <code>/reports/eval)")
    p.add_argument("--hours", type=float, help="run for at most this many hours (default unlimited)")
    p.add_argument("--appendix", nargs="*", help="models that additionally run the thinking-disabled appendix comparison (_nothink) after all main-table tasks")
    p.add_argument("--smoke-only", action="store_true", help="smoke runs only (for consistency checks; no score tables, no paper tables)")
    sp.add_parser("status")
    sp.add_parser("watchdog")
    p = sp.add_parser("reset")
    p.add_argument("model")
    p.add_argument("setting", nargs="?")
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "run":
        local_env(load_eval_config())
        Queue(Path(a.root)).run()
    elif a.cmd == "run-local":
        cmd_run_local(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "watchdog":
        cmd_watchdog(a)
    else:
        cmd_reset(a)


if __name__ == "__main__":
    main()
