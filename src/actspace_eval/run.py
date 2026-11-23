"""Run one agent with one setting on all (or selected) question groups; write the trajectory jsonl and per-question / per-group results (docs/evaluation_protocol.md).

    python -m actspace_eval.run --agent scripted_oracle --mode active --budget 40 --name s2_scripted_oracle_active_b40
            [--groups g_000001 ...] [--run-seed 0] [--no-odometry] [--no-render] [--replay]

Outputs: <paths.eval_runs>/<name>/trajectories.jsonl, summary.json (with per-group per_episode and per-question per_question, aggregated by score).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .agents import make_agent
from .data import FrozenData
from .env import MODES, ActSpaceEnv, derive_seed, EndEpisode
from .meta import group_meta
from .results import build_summary, episode_rows
from .trajectory import TrajectoryWriter, replay


def run_episode(env, agent, group_id, run_seed=0):
    obs = env.reset(group_id)
    agent_seed = derive_seed(env.data.ecfg["seed"], run_seed, group_id, agent.name)
    agent.reset(obs, seed=agent_seed, group_id=group_id if agent.privileged else None, env=env if agent.privileged else None)
    done = False
    while not done:
        try:
            action = agent.act(obs)
        except EndEpisode as e:     # e.g. context_overflow: the episode ends without an answer
            env.end(e.reason)
            break
        obs, _, done = env.step(action)
    return env.records, agent_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["random", "scripted_oracle", "seen_oracle"])
    ap.add_argument("--mode", default="active", choices=MODES)
    ap.add_argument("--budget", type=int, default=None, help="step budget for active; k for passive (default: main budget)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--no-odometry", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--replay", action="store_true", help="replay all episodes after the run to check reproducibility")
    args = ap.parse_args()

    t0 = time.time()
    data = FrozenData()  # verify the freeze manifest
    meta = group_meta(data)
    env = ActSpaceEnv(data, budget=args.budget, odometry_on=False if args.no_odometry else None,
                      render=False if args.no_render else None, run_seed=args.run_seed, mode=args.mode)
    agent = make_agent(args.agent, data)
    out = Path(data.ecfg["paths"]["eval_runs"]) / args.name
    writer = TrajectoryWriter(out / "trajectories.jsonl")
    gids = args.groups or data.group_ids
    per_q, per_ep = [], []
    for gid in gids:
        records, agent_seed = run_episode(env, agent, gid, args.run_seed)
        writer.write_episode(records, run=args.name, agent=agent.name, agent_seed=agent_seed)
        ep, rows = episode_rows(data, meta, records)
        if agent.name == "scripted_oracle" and args.mode == "active":
            logged = [(r["action"], r["to"]) for r in records if r["type"] == "step"]
            ep |= {"route_len": len(agent.route), "targets": len(agent.targets), "unreachable": agent.unreachable,
                   "route_followed": logged == [tuple(x) for x in agent.route[:len(logged)]],
                   "route_completed": len(logged) == len(agent.route)}
            assert len(agent.route) == meta[gid]["group_route_steps"], gid
        per_ep.append(ep)
        per_q += rows
    writer.close()

    def mean(xs):
        return float(np.mean(xs)) if xs else None
    summary = build_summary({"run": args.name, "agent": agent.name, "mode": args.mode, "version": data.version,
                             "budget": env.budget, "odometry": env.odometry_on, "render": env.render, "run_seed": args.run_seed},
                            per_ep, per_q)
    if agent.name == "scripted_oracle" and args.mode == "active":
        summary["oracle"] = {"route_followed_all": all(e["route_followed"] for e in per_ep),
                             "routes_completed": sum(e["route_completed"] for e in per_ep),
                             "routes_over_budget": [e["group_id"] for e in per_ep if e["route_len"] > env.budget],
                             "unreachable_targets": {e["group_id"]: e["unreachable"] for e in per_ep if e["unreachable"]},
                             "mean_route_len": mean([e["route_len"] for e in per_ep])}
    if args.replay:
        summary["replay"] = {k: v for k, v in replay(data, out / "trajectories.jsonl").items() if k != "diffs"}
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    brief = {k: v for k, v in summary.items() if k not in ("per_episode", "per_question", "score_by_template", "steps_hist")}
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == "__main__":
    main()
