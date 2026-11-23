"""Agent interface and fake agents that call no API (docs/evaluation_protocol.md).

- RandomAgent: at each step picks uniformly among the 6 move actions; answers randomly once the budget is used up.
- SeenOracleAgent (auxiliary, only for checking scoring and derived quantities): moves like random; when answering, it gives the gold answer for questions whose
  evidence objects were each visible at least once in the photos it has seen, and guesses randomly otherwise. Its score therefore depends only on "which evidence was seen", which makes it predictable for checking evidence hits, derived quantities and the bootstrap.
- ScriptedOracleAgent: privileged agent, only for validating the environment. Reads oracle_nodes + link_views (plus path_views for T4) from the frozen data,
  walks the shortest route visiting all targets on the (masked) view graph (solved exactly when there are ≤ 14 targets), then answers with the gold answers.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

from . import answers as A
from .data import MOVE_ACTIONS


class Agent:
    name = "agent"
    privileged = False  # for True, the runner additionally passes group_id and the environment (only for validating the environment and scoring)

    def reset(self, start_obs: dict, *, seed: int, group_id: str | None = None, env=None) -> None:
        raise NotImplementedError

    def act(self, obs: dict):
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def __init__(self, t1_range=(0.5, 5.0)):
        self.t1_range = t1_range

    def reset(self, start_obs, *, seed, group_id=None, env=None):
        self.rng = np.random.default_rng([seed, 2])
        self.questions = start_obs["questions"]

    def guess(self, q):
        if q["answer_type"] == "numeric":
            return round(float(self.rng.uniform(*self.t1_range)), 2)
        return q["options"][int(self.rng.integers(len(q["options"])))].split(".", 1)[0]

    def act(self, obs):
        if not obs["must_answer"]:
            return MOVE_ACTIONS[int(self.rng.integers(len(MOVE_ACTIONS)))]
        return {"action": "answer", "content": json.dumps({q["id"]: self.guess(q) for q in self.questions})}


def seen_nodes(env) -> list:
    """Nodes the agent has seen (private): for active, the start and visited nodes; for static settings, the given photos."""
    if env.mode != "active":
        return list(env.shown)
    return [env.records[0]["start_node"]] + [r["to"] for r in env.records if r["type"] == "step"]


def evidence_seen(scene, q, nodes) -> bool:
    """Evidence hit: every evidence object is visible at least once among the seen nodes."""
    return all(any(scene.visible(o, n) for n in nodes) for o in set(q["evidence_objs"]))


class SeenOracleAgent(RandomAgent):
    name = "seen_oracle"
    privileged = True

    def __init__(self, data, t1_range=(0.5, 5.0)):
        super().__init__(t1_range)
        self.data = data

    def reset(self, start_obs, *, seed, group_id=None, env=None):
        super().reset(start_obs, seed=seed)
        self.group, self.env = self.data.groups[group_id], env

    def act(self, obs):
        if not obs["must_answer"]:
            return MOVE_ACTIONS[int(self.rng.integers(len(MOVE_ACTIONS)))]
        nodes, scene = seen_nodes(self.env), self.env.scene
        ans = {}
        for qp, q in zip(self.questions, self.group["questions"]):
            ok = q["evidence_label"] != "unanswerable" and evidence_seen(scene, q, nodes)
            ans[qp["id"]] = A.gold(q) if ok else self.guess(qp)
        return {"action": "answer", "content": json.dumps(ans)}


def shortest_path(scene, masked, src, dst):
    """Shortest action sequence [(action, reached node)] from src to dst on the view graph (masked nodes removed); None if unreachable."""
    if src == dst:
        return []
    prev = {src: None}
    dq = deque([src])
    while dq:
        v = dq.popleft()
        for a in MOVE_ACTIONS:  # fixed action order, deterministic result
            w = scene.edges[v][a]
            if w is None or w in masked or w in prev:
                continue
            prev[w] = (v, a)
            if w == dst:
                path = []
                while prev[w] is not None:
                    v_, a_ = prev[w]
                    path.append((a_, w))
                    w = v_
                return path[::-1]
            dq.append(w)
    return None


def oracle_targets(group) -> list[str]:
    out = []
    for q in group["questions"]:
        for n in (q.get("oracle_nodes") or []) + (q.get("link_views") or []) + (q.get("path_views") or []):
            if n not in out:
                out.append(n)
    return out


def plan_route(scene, masked, start, targets, exact_max=14):
    """Action sequence visiting all target nodes. With ≤ exact_max targets, solve the exact shortest route (bitmask DP over BFS distances between targets,
    i.e. an open-path travelling salesman problem; passing through other targets costs nothing extra), otherwise greedy. Returns (actions, unreachable targets)."""
    todo = [t for t in dict.fromkeys(targets) if t != start]
    paths = {}
    for a in [start] + todo:
        for b in todo:
            if a != b:
                paths[(a, b)] = shortest_path(scene, masked, a, b)
    unreachable = [t for t in todo if paths[(start, t)] is None]
    todo = [t for t in todo if t not in unreachable]
    if not todo or len(todo) > exact_max:
        return _greedy_route(scene, masked, start, todo, unreachable)
    n, INF = len(todo), 1 << 30
    dist = lambda a, b: len(paths[(a, b)])
    dp = {(1 << i, i): (dist(start, todo[i]), None) for i in range(n)}
    for m in range(1, 1 << n):
        for i in range(n):
            if (m, i) not in dp:
                continue
            c = dp[(m, i)][0]
            for j in range(n):
                if not m >> j & 1:
                    k, v = (m | 1 << j, j), c + dist(todo[i], todo[j])
                    if v < dp.get(k, (INF,))[0]:
                        dp[k] = (v, i)
    full = (1 << n) - 1
    i = min(range(n), key=lambda i: (dp[(full, i)][0], i))
    order, m = [], full
    while i is not None:
        order.append(todo[i])
        m, i = m & ~(1 << i), dp[(m, i)][1]
    route, cur = [], start
    for t in order[::-1]:
        route += paths[(cur, t)]
        cur = t
    return route, unreachable


def _greedy_route(scene, masked, start, todo, unreachable):
    """Greedy: always go to the nearest unvisited target (fewest BFS steps; ties go to the earlier target in the list)."""
    route, cur = [], start
    while todo:
        best = None
        for t in todo:
            p = shortest_path(scene, masked, cur, t)
            if p is None:
                continue
            if best is None or len(p) < len(best[1]):
                best = (t, p)
        if best is None:
            unreachable = unreachable + todo
            break
        t, p = best
        # other targets passed on the way also count as visited
        visited = {n for _, n in p}
        todo = [x for x in todo if x not in visited]
        route += p
        cur = t
    return route, unreachable


class ScriptedOracleAgent(Agent):
    name = "scripted_oracle"
    privileged = True

    def __init__(self, data):
        self.data = data

    def reset(self, start_obs, *, seed, group_id=None, env=None):
        g = self.data.groups[group_id]
        scene = self.data.scene(g["scene_id"])
        masked = self.data.masked_nodes(group_id)
        self.targets = oracle_targets(g)
        self.route, self.unreachable = plan_route(scene, masked, g["init_frame"], self.targets)
        self.k = 0
        self.answer = {qp["id"]: A.gold(q) for qp, q in zip(start_obs["questions"], g["questions"])}

    def act(self, obs):
        if self.k < len(self.route) and not obs["must_answer"]:
            a = self.route[self.k][0]
            self.k += 1
            return a
        return {"action": "answer", "content": json.dumps(self.answer)}


def make_agent(name, data):
    if name == "random":
        return RandomAgent(tuple(data.ecfg["agents"]["random"]["t1_range_m"]))
    if name == "scripted_oracle":
        return ScriptedOracleAgent(data)
    if name == "seen_oracle":
        return SeenOracleAgent(data, tuple(data.ecfg["agents"]["random"]["t1_range_m"]))
    raise ValueError(f"unknown agent {name}")
