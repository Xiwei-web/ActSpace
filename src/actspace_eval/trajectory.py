"""Trajectory recording and replay (docs/evaluation_protocol.md).

Each episode's private records (episode_start / step… / episode_end) are written line by line to jsonl. Replay re-executes the recorded seed and action
sequence in a fresh environment and compares the records one by one (render settings excepted); any mismatch means the environment or data changed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .env import ActSpaceEnv


class TrajectoryWriter:
    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a" if append else "w")

    def write_episode(self, records: list[dict], **extra):
        for r in records:
            self.f.write(json.dumps({**extra, **r} if r["type"] == "episode_start" else r, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def complete_episodes(path: Path) -> list[list[dict]]:
    """Keep only complete episodes (ending with episode_end); used to clean up the file before resuming."""
    if not Path(path).exists():
        return []
    return [ep for ep in read_episodes(path) if ep[-1]["type"] == "episode_end"]


def read_episodes(path: Path) -> list[list[dict]]:
    eps, cur = [], None
    with open(path) as f:
        for line in f:
            if not line.endswith("\n"):    # another process is writing, or the process was killed
                break
            r = json.loads(line)
            if r["type"] == "episode_start":
                cur = [r]
                eps.append(cur)
            else:
                cur.append(r)
    return eps


_IGNORE = {"render", "run", "agent", "agent_seed", "model", "model_name", "reasoning_effort", "history", "cost_usd", "calls",
           "setting", "prompt_version", "dataset", "step_scale",
           "budget_infeasible", "run_seed", "thinking", "sampling", "max_output_tokens", "cannot_move", "invalid"}  # fields added by the runner


def replay_episode(data, ep: list[dict]) -> list[str]:
    """Replay one episode and return the mismatches (empty list = identical)."""
    s = ep[0]
    env = ActSpaceEnv(data, budget=s["budget"], odometry_on=s["odometry"], render=False, mode=s.get("mode", "active"),
                      cumulative_pose=s.get("cumulative_pose", False))
    env.reset(s["group_id"], seed=s["seed"])
    for r in ep[1:]:
        if r["type"] == "step":
            env.step({"action": r["action"], "notes": r["notes"]} if "notes" in r else r["action"])
        elif r["forced"] == "invalid_replies":  # ended automatically by the last invalid step
            continue
        elif r["forced"] == "context_overflow":  # ended early by the runner
            env.end("context_overflow")
        elif r["forced"] in ("move_after_budget", "move_in_static_mode"):
            env.step("forward")
        else:
            env.step({"action": "answer", "content": r["raw_answer"], "notes": r.get("notes")})
    diffs = []
    if len(env.records) != len(ep):
        diffs.append(f"{s['group_id']}: {len(ep)} records logged, {len(env.records)} replayed")
    for a, b in zip(ep, env.records):
        a = {k: v for k, v in a.items() if k not in _IGNORE}
        b = {k: v for k, v in b.items() if k not in _IGNORE}
        if json.loads(json.dumps(a)) != json.loads(json.dumps(b)):
            diffs.append(f"{s['group_id']} {a['type']} t={a.get('t')}: logged {a} != replayed {b}")
    return diffs


def replay(data, path: Path) -> dict:
    eps = read_episodes(path)
    diffs = defaultdict(list)
    for ep in eps:
        d = replay_episode(data, ep)
        if d:
            diffs[ep[0]["group_id"]] = d
    return {"episodes": len(eps), "mismatched_episodes": len(diffs), "diffs": dict(diffs)}
