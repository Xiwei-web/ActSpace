"""Model agent: assembles public observations into prompts, calls the model, parses the reply into an action (docs/evaluation_protocol.md).

- Static settings (Blind / Passive / Oracle): one call, reply {"action": "answer", "answers": {...}}.
- Active: multi-turn conversation. Turn 0 gives the questions and the start photo; each later turn gives the result of the previous action (feedback / odometry), the new photo (when the move succeeded) and the step count.
  Image history per config: last_k (latest k photos as images, the rest as text placeholders) or full.
- Reply parsing: take the first JSON object. If action is a move action name → move; if answer → answer with answers (the whole object if missing);
  no JSON or an invalid action name → passed to the environment as an invalid action (costs 1 step; the environment ends the episode after 3 consecutive invalid actions). The raw reply is kept in the call log.
- Optional "notes" field: not scored, truncated beyond 300 characters; passed to the environment with the action and written to the trajectory.
  The Active conversation keeps the model's raw reply; when notes are too long, the truncated JSON is kept instead.
- Active prompts come in v1 / v2 (v2 adds the per-step scale and cumulative pose; the environment must enable cumulative_pose);
  random_walk / oracle_route answer once; the user message gives the photos and per-step records in visiting order (prompts.user_route).
"""
from __future__ import annotations

import json

from .agents import Agent
from .answers import _first_json_object
from .data import MOVE_ACTIONS
from . import prompts as P

INVALID = "invalid_reply"
NOTES_MAX = 300

# setting name → (environment mode, Active prompt version). active_v2 is the main setting, active_v1 is kept;
# the original oracle is renamed oracle_min; added baselines random_walk (3 seeds per group, distinguished by run_seed), oracle_route, passive_k (passive with budget = k).
# "active" and "oracle" are legacy names (pilot runs), equivalent to active_v1 and oracle_min.
SETTINGS = {"blind": ("blind", None), "passive": ("passive", None), "oracle": ("oracle", None), "oracle_min": ("oracle_min", None),
            "oracle_route": ("oracle_route", None), "random_walk": ("random_walk", None),
            "active": ("active", "v1"), "active_v1": ("active", "v1"), "active_v2": ("active", "v2")}


def setting_env(setting: str) -> tuple[str, str | None, bool]:
    """→ (environment mode, prompt version, whether to give the cumulative pose)."""
    mode, version = SETTINGS[setting]
    return mode, version, version == "v2"


def reply_notes(obj) -> str | None:
    """notes in the reply (not scored), truncated beyond NOTES_MAX characters."""
    if not isinstance(obj, dict) or obj.get("notes") is None:
        return None
    n = obj["notes"] if isinstance(obj["notes"], str) else json.dumps(obj["notes"], ensure_ascii=False)
    return n[:NOTES_MAX]


def parse_reply(text: str):
    """Return (action, answer content). The action is a move action name, "answer" or an invalid action name."""
    obj = _first_json_object(text or "")
    if obj is None:
        return INVALID, None
    act = str(obj.get("action", "")).strip().lower()
    if act in MOVE_ACTIONS:
        return act, None
    if act == "answer":
        return "answer", obj["answers"] if isinstance(obj.get("answers"), dict) else obj
    if not act and any(str(k).strip().lower().startswith("q") for k in obj):  # only {"Q1": ...} given: treat as an answer
        return "answer", obj
    return (act[:40] or INVALID), None


def parse_detail(text: str) -> dict:
    """Parse result and failure reason for the call log: json (whether a JSON object was found), action, error, answers."""
    if not (text or "").strip():
        return {"json": False, "action": INVALID, "error": "empty reply"}
    obj = _first_json_object(text)
    if obj is None:
        return {"json": False, "action": INVALID, "error": "no JSON object in reply"}
    act, content = parse_reply(text)
    out = {"json": True, "action": act, "error": None}
    if act == "answer":
        out["answers"] = content
    elif act not in MOVE_ACTIONS:
        out["error"] = "missing action field" if not str(obj.get("action", "")).strip() else f"invalid action {act!r}"
    return out


def history_text(text: str) -> str:
    """assistant reply kept in the Active conversation: the raw text; replaced by the truncated JSON when notes are too long."""
    obj = _first_json_object(text or "")
    if isinstance(obj, dict) and isinstance(obj.get("notes"), str) and len(obj["notes"]) > NOTES_MAX:
        return json.dumps({**obj, "notes": obj["notes"][:NOTES_MAX]}, ensure_ascii=False)
    return text


def static_messages(mode: str, obs: dict):
    """One-shot answer settings: (system prompt, conversation)."""
    if mode in P.ROUTE_MODES:
        return P.system_prompt(mode), [{"role": "user", "parts": P.user_route(obs)}]
    return P.system_prompt(mode, n_photos=len(obs["photos"])), [{"role": "user", "parts": P.user_static(obs)}]


class LLMAgent(Agent):
    privileged = False

    def __init__(self, client, mode: str, history: dict, name: str | None = None, version: str = "v1", scale: dict | None = None):
        self.client, self.mode, self.history = client, mode, history
        self.version, self.scale = version, scale
        self.name = name or f"llm:{client.model_name}"
        client.parse_hook = parse_detail

    def reset(self, start_obs, *, seed, group_id=None, env=None):
        self.start = start_obs
        self.turns, self.t, self.seen_ids = [], 0, set()  # t: call index within this episode
        self.budget = start_obs.get("budget")
        self.calls = []
        self.episode_tag = None

    def _call(self, system: str, turns: list[dict]):
        comp = self.client.complete(system, turns, tag={"episode": self.episode_tag, "t": self.t, "mode": self.mode})
        self.calls.append(comp)
        return comp

    def act(self, obs):
        if self.mode != "active":
            comp = self._call(*static_messages(self.mode, obs))
            act, content = parse_reply(comp.text)
            notes = reply_notes(_first_json_object(comp.text or ""))
            return {"action": "answer", "content": content if act == "answer" else comp.text, "notes": notes}
        if not self.turns:
            self.turns.append({"role": "user", "parts": P.user_active_start(obs, self.version)})
            self.seen_ids.add(obs["photo_id"])
        else:
            revisit = obs.get("moved") and obs["photo_id"] in self.seen_ids
            self.seen_ids.add(obs["photo_id"])
            self.turns.append({"role": "user", "parts": P.user_active_step(
                obs["steps_used"], self.last_action, obs, self.budget, bool(revisit), self.last_invalid, self.version)})
        system = P.system_prompt("active", budget=self.budget, history=self.history, version=self.version, scale=self.scale)
        comp = self._call(system, P.apply_history(self.turns, self.history))
        self.turns.append({"role": "assistant", "text": history_text(comp.text)})
        act, content = parse_reply(comp.text)
        notes = reply_notes(_first_json_object(comp.text or ""))
        self.t += 1
        self.last_action = act
        self.last_invalid = act not in MOVE_ACTIONS and act != "answer"
        if act == "answer":
            return {"action": "answer", "content": content, "notes": notes}
        return {"action": act, "notes": notes}
