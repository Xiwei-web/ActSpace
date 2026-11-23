"""ActSpace interactive environment (Gym style, docs/evaluation_protocol.md).

    env = ActSpaceEnv(data)
    obs = env.reset("g_000040", seed=...)            # initial observation: questions, start photo
    obs, feedback, done = env.step("forward")        # every move action costs 1 step (including "cannot move" and invalid actions)
    obs, feedback, done = env.step({"action": "answer", "content": '{"Q1": "B", ...}'})

Public observations contain only the questions, photos (upright originals), random photo IDs, feedback, odometry and step count; raw frame IDs, the view graph, visibility, oracle information and
gold answers are never exposed. Each episode's private records (with frame IDs and scores) are in env.records and written to jsonl by trajectory.

Four settings: mode = active (interactive exploration, usage above); blind (questions only); passive (start photo + k = budget photos drawn uniformly at random from unmasked nodes;
the seed depends only on the question group and k, so all agents see the same photos); oracle (start + oracle_nodes + link_views,
plus path_views for T4). The last three give all photos at reset (obs["photos"]) and then accept only answer.
The start photo is always first; the other photos are randomly ordered (seed depends only on the question group).

Invalid replies:
- An invalid reply (unparseable / illegal action) costs 1 step; after max_consecutive_invalid (3) consecutive invalid replies the episode ends and is scored as unanswered (forced = invalid_replies).
- An action may be {"action": ..., "notes": ...}; notes are not scored and are truncated to notes_max_chars (300) characters before being written to the trajectory.

Setting revisions:
- cumulative_pose (active_v2): every observation also gives the cumulative pose relative to START, obs["pose"] = {forward_m, left_m, heading_deg},
  computed directly from the true poses (not by summing rounded per-step odometry), rounded like per-step odometry (0.1 m, 5°); written to the "pose" of the step record.
- The original oracle is renamed oracle_min ("oracle" is kept as a legacy name with identical behaviour, so old trajectories replay as before).
- Two route settings (answer once; photos are given in visiting order, together with each step's action and cumulative pose):
  random_walk: from START, at each step pick uniformly among the moves that succeed (with an unmasked neighbouring viewpoint), for budget steps; randomness is derived from the episode seed,
  so different run_seed values give different walks (3 seeds per group). oracle_route: follows scripted_oracle's group-level shortest evidence route (agents.plan_route).
  obs["photos"] holds START and every viewpoint at its first visit; obs["route"] holds each step {t, action, photo_id, pose}.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import answers as A
from .data import MOVE_ACTIONS, FrozenData
from .odometry import describe, odometry

ACTIONS = MOVE_ACTIONS + ("answer",)
MODES = ("active", "blind", "passive", "oracle", "oracle_min", "oracle_route", "random_walk")
ROUTE_MODES = ("random_walk", "oracle_route")
CANNOT_MOVE = "Cannot move: no view in that direction."


def derive_seed(*parts) -> int:
    """Seed derivation that is stable across processes."""
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little")



class EndEpisode(Exception):
    """The agent cannot continue (e.g. the conversation exceeds the model context): run_episode catches it and calls env.end(reason)."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

class ActSpaceEnv:
    def __init__(self, data: FrozenData, budget: int | None = None, odometry_on: bool | None = None,
                 render: bool | None = None, run_seed: int = 0, mode: str = "active", cumulative_pose: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        ec = data.ecfg["env"]
        self.data, self.ec = data, ec
        self.budget = int(budget if budget is not None else ec["budget"])
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        self.odometry_on = ec["odometry"] if odometry_on is None else odometry_on
        self.render = ec["render"] if render is None else render
        self.run_seed = run_seed
        self.cumulative_pose = bool(cumulative_pose)
        self.thresholds = tuple(data.ecfg["scoring"]["mra_thresholds"])
        self.group = None

    # ------------------------------------------------------------ public interface
    def reset(self, group_id: str, seed: int | None = None) -> dict:
        g = self.data.groups[group_id]
        self.group, self.scene = g, self.data.scene(g["scene_id"])
        self.masked = self.data.masked_nodes(group_id)
        self.node = g["init_frame"]
        assert self.node not in self.masked
        self.seed = derive_seed(self.data.ecfg["seed"], self.run_seed, group_id) if seed is None else int(seed)
        self._id_rng = np.random.default_rng([self.seed, 1])
        self._photo_ids, self._used_ids = {}, set()
        self.steps_used, self.done, self.forced, self.invalid_streak = 0, False, None, 0
        self.q_public = A.public_questions(g)
        start = {"type": "episode_start", "mode": self.mode, "version": self.data.version, "group_id": group_id,
                 "scene_id": g["scene_id"], "seed": self.seed, "budget": self.budget,
                 "odometry": self.odometry_on, "render": self.render, "start_node": self.node,
                 "question_ids": {q["id"]: gq["qid"] for q, gq in zip(self.q_public, g["questions"])},
                 "masked_nodes": len(self.masked)}
        if self.cumulative_pose:
            start["cumulative_pose"] = True
        self.start_pose = self.scene.pose[self.node]
        if self.mode in ROUTE_MODES:
            return self._reset_route(start)
        if self.mode != "active":
            self.shown = self._static_nodes(group_id)
            photos = [{**self._view(n), "start": n == self.node} for n in self.shown]
            self.records = [{**start, "shown_nodes": self.shown, "shown_photo_ids": [p["photo_id"] for p in photos]}]
            return {"questions": self.q_public, "photos": photos, "actions": ["answer"],
                    "steps_used": 0, "steps_left": 0, "must_answer": True}
        self.records = [{**start, "start_photo_id": self._photo_id(self.node)}]
        obs = {"questions": self.q_public, **self._view(self.node), "budget": self.budget, "actions": list(ACTIONS),
               "odometry_enabled": self.odometry_on, **self._counters()}
        if self.cumulative_pose:
            obs["pose"] = self.pose_rel_start(self.node)
        return obs

    def pose_rel_start(self, node) -> dict:
        """Cumulative pose relative to START: forward / left in the START heading (m, rounded to odometry_round_m) and heading change (degrees, left positive, rounded to odometry_round_deg)."""
        od = odometry(self.start_pose, self.scene.pose[node], self.ec["odometry_round_m"], self.ec["odometry_round_deg"])
        h = od["turn_deg"]
        return {"forward_m": od["forward_m"], "left_m": od["left_m"], "heading_deg": 180 if h == -180 else h}

    def _route_moves(self):
        """Action sequence of a route setting [(action, reached node)]."""
        if self.mode == "oracle_route":
            from .agents import oracle_targets, plan_route
            route, unreachable = plan_route(self.scene, self.masked, self.node, oracle_targets(self.group))
            assert not unreachable, unreachable
            return [tuple(x) for x in route]
        rng = np.random.default_rng([self.seed, 3])
        cur, out = self.node, []
        for _ in range(self.budget):
            ok = [a for a in MOVE_ACTIONS if self.scene.edges[cur][a] is not None and self.scene.edges[cur][a] not in self.masked]
            if not ok:
                break
            a = ok[int(rng.integers(len(ok)))]
            cur = self.scene.edges[cur][a]
            out.append((a, cur))
        return out

    def _reset_route(self, start):
        moves = self._route_moves()
        self.shown = list(dict.fromkeys([self.node] + [n for _, n in moves]))
        photos = [{**self._view(n), "start": n == self.node} for n in self.shown]
        route = [{"t": t, "action": a, "photo_id": self._photo_id(n), "pose": self.pose_rel_start(n)} for t, (a, n) in enumerate(moves, 1)]
        self.records = [{**start, "shown_nodes": self.shown, "shown_photo_ids": [p["photo_id"] for p in photos],
                         "route": [[a, n] for a, n in moves]}]
        return {"questions": self.q_public, "photos": photos, "route": route, "start_pose": self.pose_rel_start(self.node),
                "actions": ["answer"], "steps_used": 0, "steps_left": 0, "must_answer": True}

    def _static_nodes(self, group_id):
        """blind: no photos; passive: start + k photos drawn uniformly at random from unmasked nodes (k = budget); oracle / oracle_min: start + scripted_oracle's target nodes."""
        if self.mode == "blind":
            return []
        start = self.node
        if self.mode == "passive":
            pool = sorted(n for n in self.scene.nodes if n not in self.masked and n != start)
            rng = np.random.default_rng(derive_seed(self.data.ecfg["seed"], "passive", group_id, self.budget))
            return [start] + rng.choice(pool, size=min(self.budget, len(pool)), replace=False).tolist()
        from .agents import oracle_targets
        rest = [n for n in oracle_targets(self.group) if n != start]
        rng = np.random.default_rng(derive_seed(self.data.ecfg["seed"], "oracle", group_id))
        return [start] + [rest[i] for i in rng.permutation(len(rest))]

    def step(self, action):
        if self.group is None or self.done:
            raise RuntimeError("call reset() first; the episode has ended")
        name, content, notes = self._normalize(action)
        if name == "answer":
            return self._finish(content, forced_reason=self.forced, notes=notes)
        if self.mode != "active":  # static settings accept only answers
            return self._finish(None, forced_reason="move_in_static_mode")
        if self.steps_used >= self.budget:  # move submitted after the budget is used up: not executed, episode ends with an empty answer
            return self._finish(None, forced_reason="move_after_budget")
        prev = self.node
        self.steps_used += 1
        if name not in MOVE_ACTIONS:
            self.invalid_streak += 1
            moved, od, fb = False, None, f"Invalid action: {name[:80]!r}. Valid actions: {', '.join(ACTIONS)}."
        else:
            self.invalid_streak = 0
            nxt = self.scene.edges[prev][name]
            if nxt is None or nxt in self.masked:
                moved, od, fb = False, None, CANNOT_MOVE
            else:
                self.node, moved = nxt, True
                od = odometry(self.scene.pose[prev], self.scene.pose[nxt], self.ec["odometry_round_m"], self.ec["odometry_round_deg"])
                fb = describe(od) if self.odometry_on else "Moved."
                if not self.odometry_on:
                    od = None
        if self.steps_used >= self.budget:
            self.forced = "budget_exhausted"
        view = self._view(self.node)
        rec = {"type": "step", "t": self.steps_used, "action": name[:200],
               "from": prev, "to": self.node, "moved": moved, "photo_id": view["photo_id"],
               "feedback": fb, "odometry": od, "steps_used": self.steps_used}
        if self.cumulative_pose:
            rec["pose"] = self.pose_rel_start(self.node)
        if notes is not None:
            rec["notes"] = notes
        self.records.append(rec)
        if self.invalid_streak >= self.ec.get("max_consecutive_invalid", 3):
            return self._finish(None, forced_reason="invalid_replies")
        obs = {**view, "moved": moved, "feedback": fb, "odometry": od, **self._counters()}
        if self.cumulative_pose:
            obs["pose"] = rec["pose"]
        return obs, fb, False

    # ------------------------------------------------------------ internals
    def _normalize(self, action):
        """→ (action name, answer content, notes)."""
        if isinstance(action, dict):
            name = str(action.get("action", "")).strip().lower()
            notes = action.get("notes")
            if notes is not None:
                notes = str(notes)[:self.ec.get("notes_max_chars", 300)]
            return name, action.get("content"), notes
        return str(action).strip().lower(), None, None

    def _counters(self):
        left = self.budget - self.steps_used
        return {"steps_used": self.steps_used, "steps_left": left, "must_answer": left <= 0}

    def _photo_id(self, node):
        if node not in self._photo_ids:
            n = 10 ** self.ec["photo_id_digits"]
            while True:
                k = int(self._id_rng.integers(n))
                if k not in self._used_ids:
                    break
            self._used_ids.add(k)
            self._photo_ids[node] = f"photo_{k:0{self.ec['photo_id_digits']}d}"
        return self._photo_ids[node]

    def _view(self, node):
        assert node not in self.masked, "masked nodes must not be returned"
        return {"photo_id": self._photo_id(node), "image": self.scene.image(node) if self.render else None}

    def end(self, forced_reason):
        """End the episode early on behalf of the runner (unanswered, scored 0): e.g. context_overflow."""
        return self._finish(None, forced_reason=forced_reason)

    def _finish(self, content, forced_reason=None, notes=None):
        self.done = True
        parsed = A.parse_answers(content, self.q_public) if content is not None else \
            {"ok": False, "answers": {q["id"]: None for q in self.q_public}, "errors": {q["id"]: "no answer submitted" for q in self.q_public}}
        scores = {}
        for qp, q in zip(self.q_public, self.group["questions"]):
            scores[qp["id"]] = A.score_question(q, parsed["answers"][qp["id"]], self.thresholds)
        self.records.append({"type": "episode_end", "raw_answer": content, "parse_ok": parsed["ok"], "answers": parsed["answers"],
                             "parse_errors": parsed["errors"], "scores": scores, "steps_used": self.steps_used,
                             "forced": forced_reason, "final_node": self.node, **({"notes": notes} if notes is not None else {})})
        fb = {"move_after_budget": "The step budget is used up; the episode ended without a valid answer.",
              "move_in_static_mode": "Only an answer is accepted in this setting; the episode ended without a valid answer.",
              "invalid_replies": "Three invalid replies in a row; the episode ended without a valid answer.",
              "context_overflow": "The conversation no longer fits the model's context; the episode ended without a valid answer."
              }.get(forced_reason, "Answer received.")
        return {"feedback": fb, **self._counters(), "done": True}, fb, True
