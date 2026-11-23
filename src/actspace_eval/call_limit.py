"""Per-call wall-clock limit for vLLM models.

Principle: the time limit only stops stuck or runaway requests; it does not limit normal reasoning length (normal length is limited by max_tokens 32768).
- Limit = factor (3) × tokens (32768) ÷ the model's normal decoding speed. Decoding speed = median generation speed of completed calls (output tokens ÷ (total time − time to first token));
  only calls with output ≥ min_output_tokens (100) are used, since speed is meaningless for very short outputs.
- Sources in order: this model's formal runs (≥ min_calls calls) → this model's smoke runs (run name starting with smoke) → the configured proxy model (call_timeout_proxy, a model with the same architecture).
- Every call of a formal run appends its speed to <model dir>/call_speed.tsv; when the initial value did not come from formal runs, it is updated once automatically after min_calls (50) calls, recorded in
  the history of <model dir>/call_timeout.json. Each process checks before every call whether the file was replaced and rereads it, so updates also apply to running runs.
- Several runs starting at once: a file lock ensures it is computed only once. The limit is written to each run's env.json (at start and at end) and to every call record (call_timeout_s).
"""
from __future__ import annotations

import fcntl
import json
import statistics
import time
from pathlib import Path


def call_speed(output_tokens, latency_s, ttft_s, min_output_tokens=100):
    """Generation speed of one call (tokens/s, excluding time to first token); None if the output is too short or the timing is invalid."""
    dt = (latency_s or 0) - (ttft_s or 0)
    if not output_tokens or output_tokens < min_output_tokens or dt <= 0:
        return None
    return output_tokens / dt


class CallLimit:
    def __init__(self, model_dir: Path, mcfg: dict, llm_cfg: dict, formal: bool = True):
        rule = llm_cfg.get("call_timeout_auto") or {}
        self.factor = float(rule.get("factor", 3))
        self.tokens = int(rule.get("tokens", 32768))
        self.min_calls = int(rule.get("min_calls", 50))
        self.min_out = int(rule.get("min_output_tokens", 100))
        self.dir = Path(model_dir)
        self.model = self.dir.name
        self.path = self.dir / "call_timeout.json"
        self.samples = self.dir / "call_speed.tsv"
        self.lockfile = self.dir / ".call_timeout.lock"
        self.proxy = mcfg.get("call_timeout_proxy")
        self.formal = formal
        self.state, self._mtime = None, None

    # ------------------------------------------------------------------ computation
    def rule(self) -> dict:
        return {"factor": self.factor, "tokens": self.tokens, "min_calls": self.min_calls, "min_output_tokens": self.min_out}

    def scan(self, model_dir: Path, smoke: bool) -> list[float]:
        """Generation speeds of completed calls under one model directory: smoke = True looks only at smoke runs, False only at formal runs."""
        out = []
        for f in sorted(model_dir.glob("*/*/calls.jsonl")):
            if f.parent.name.startswith("smoke") != smoke:
                continue
            with open(f, errors="replace") as h:
                for line in h:
                    if not line.endswith("\n") or '"type": "call"' not in line:
                        continue
                    r = json.loads(line)
                    if r.get("type") != "call" or r.get("timeout_truncated"):
                        continue
                    v = call_speed((r.get("usage") or {}).get("output_tokens"), r.get("latency_s"), r.get("ttft_s"), self.min_out)
                    if v is not None:
                        out.append(v)
        return out

    def _make(self, stage: str, speeds: list[float], source: str) -> dict:
        v = statistics.median(speeds)
        return {"stage": stage, "source": source, "n": len(speeds), "speed_tok_s": round(v, 3),
                "limit_s": round(self.factor * self.tokens / v, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def _initial(self) -> dict:
        s = self.scan(self.dir, smoke=False)
        if len(s) >= self.min_calls:
            return self._make("formal", s, f"{self.model} formal runs")
        s = self.scan(self.dir, smoke=True)
        if s:
            return self._make("smoke", s, f"{self.model} smoke runs")
        if self.proxy:
            pd = self.dir.parent / self.proxy
            s = self.scan(pd, smoke=False) or self.scan(pd, smoke=True)
            if s:
                return self._make("proxy", s, f"proxy model {self.proxy} (no call data for {self.model} yet)")
        return {"stage": "none", "source": "no call data available; no limit", "n": 0, "speed_tok_s": None, "limit_s": None,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # ------------------------------------------------------------------ files
    def _write(self, cur: dict):
        prev = self.state or {}
        doc = {"model": self.model, "rule": self.rule(), **cur, "history": prev.get("history", []) + [cur]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        tmp.replace(self.path)
        self.state = doc
        self._mtime = self._key()

    def _key(self):
        """Whether the file was replaced: every write replaces it with a new file, so compare inode, nanosecond mtime and size."""
        st = self.path.stat()
        return st.st_ino, st.st_mtime_ns, st.st_size

    def _load(self):
        self.state = json.loads(self.path.read_text())
        self._mtime = self._key()

    def ensure(self) -> dict:
        """Read the limit; compute and write it if missing. If it came from the proxy or no data, switch to smoke data once this model has some."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.lockfile, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if self.path.exists():
                self._load()
                if self.state["stage"] in ("proxy", "none"):
                    s = self.scan(self.dir, smoke=True)
                    if s:
                        self._write(self._make("smoke", s, f"{self.model} smoke runs"))
            else:
                self.state = None
                self._write(self._initial())
        return self.snapshot()

    def snapshot(self) -> dict:
        return {k: v for k, v in (self.state or {}).items() if k != "history"}

    def limit_s(self):
        try:
            if self._key() != self._mtime:
                self._load()
        except (OSError, ValueError):
            pass
        return (self.state or {}).get("limit_s")

    def record(self, speed):
        """One call of a formal run: append its speed; update once when the initial value did not come from formal runs and min_calls calls have accumulated."""
        if not self.formal or speed is None:
            return
        self.limit_s()
        if (self.state or {}).get("stage") == "formal":
            return
        with open(self.lockfile, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            with open(self.samples, "a") as f:
                f.write(f"{speed:.4f}\n")
            self._load()
            if self.state.get("stage") == "formal":
                return
            vals = [float(x) for x in self.samples.read_text().split()]
            if len(vals) >= self.min_calls:
                self._write(self._make("formal", vals, f"{self.model} first {len(vals)} calls of formal runs (automatic update)"))
