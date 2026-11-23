"""Ablations and cognitive-map probe. One job runs all settings of a model, sharing one cost cap.

    python -m actspace_eval.providers.ablations job --model <short name>[,<short name>...] --cap <USD> [--resume] [--key-file <file>] [--set all] [--source-prefix <prefix>] [--settings a,b]
    python -m actspace_eval.providers.ablations check --model <short name> [--set all] [--source-prefix <prefix>]     # sends no requests: checks routes and image hashes against the existing runs

--model may list several short names (comma-separated): the models run one after another in the same job, and --cap caps the total
cost of these models under prefix abl.
--settings runs only the given settings (e.g. rw40_last8_v2).
Order: start_only -> rw40_last8 -> cogmap_active_v2 -> cogmap_oracle_route -> rw40_last8_v2 -> active_v2_medium. Prefix abl, run
directory <results root>/<short name>/<setting>/abl_r0/.
Existing main runs (sources) are read only: <source prefix>_r0 of random_walk_40, active_v2 and oracle_route (all for terra, all_v1
for haiku-5.5).

(A) rw40_last8: reuses the route of random_walk_40 seed 0 (the environment regenerates it with the same seed, and actions, nodes and
    photo IDs are checked group by group against the source run; it stops on any mismatch).
    The text record is unchanged (action, photo ID and accumulated pose of each step); photos are given only for START and for "the
    last k = 8 distinct viewpoints in reverse visit order" (excluding START; START is always given).
    Prompt: in SYSTEM_ROUTE the sentence "each viewpoint's photograph is given once" is replaced with Active's
    HISTORY_RULE_LAST_K_PIN sentence verbatim (without the leading "- "); viewpoints whose photo is not given use Active's
    IMAGE_OMITTED placeholder on the line of their first visit. Everything else is unchanged word for word.
    rw40_last8_v2 (the reported result): on a return to a viewpoint whose photo is not given, "(photograph shown above)" on that line
    is replaced with the format of Active's placeholder, "(photo_xxxxxx: image no longer shown)"; lines of viewpoints whose photo is
    given are unchanged. Otherwise identical to rw40_last8.
(B) start_only: Passive prompt verbatim, with START as the only photo (n_photos = 1); cannot_determine is allowed.
(C) active_v2_medium: the active_v2 main setting with only the reasoning effort changed to medium (OpenAI reasoning_effort,
    Anthropic output_config.effort).
(D) cogmap: no new exploration. For the source active_v2 and oracle_route runs, take the call in which the model answered for each
    group and rebuild its context verbatim from the call log (system prompt, text, photos; photos are re-read by frame_id and
    re-encoded, and their SHA-256 must match the log, otherwise ImageMismatch is raised and the run stops), then append PROBE to the
    end of the last user message, asking for the position of every object mentioned in the group's questions relative to the start.
    Ground truth: (forward, left) of the object's OBB center relative to the start camera, in the same frame as T4
    (geometry.theta_from_yaw) and the accumulated poses (odometry.relative_motion).
    Retries: when the reply is not in probe format (no JSON, or no objects list, e.g. an answer JSON instead), the identical request is
    resent, at most 2 times; every call stays in calls.jsonl, with tag.attempt giving the attempt number (no such field means the
    first attempt). Groups that still fail get status = unparsed for all objects, which counts as known = false in the statistics and
    is listed separately.
    If a sealed run has failed groups that have not used up their attempts, --resume unseals it, retries only those groups and
    seals it again.
    Records: probes.jsonl with one line per group (objects, ground truth, visibility level on the route / in the context photos at
    answer time, the model's answer); calls.jsonl, env.json, status.json and SHA256SUMS as in the main runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .. import prompts as P
from ..answers import _first_json_object
from ..cost import CostCapExceeded, CostTracker
from ..data import dataset_of_scene
from ..env import ActSpaceEnv
from ..llm_agent import LLMAgent, parse_reply, reply_notes
from ..llm_run import git_commit, group_set, run_llm, seal_run
from ..models import ModelError, encode_jpeg, make_client
from ..trajectory import read_episodes
from . import api_eval as E
from . import registry as R
from .base import CostPool, repair_tail

PREFIX = "abl"
RUN = f"{PREFIX}_r0"
ORDER = ("start_only", "rw40_last8", "cogmap_active_v2", "cogmap_oracle_route", "rw40_last8_v2", "active_v2_medium")
SOURCE_PREFIX = {"terra": "all", "haiku-5.5": "all_v1", "sol": "all", "luna": "all", "opus-5.5": "all_v1", "sonnet-5.5": "all_v1"}
SOURCES = {"rw40_last8": "random_walk_40", "rw40_last8_v2": "random_walk_40", "cogmap_active_v2": "active_v2", "cogmap_oracle_route": "oracle_route"}
MEDIUM = "medium"
PROBE_ATTEMPTS = 3      # cogmap: when the reply is not in probe format, the identical request is resent, at most 2 times

# ---------------------------------------------------------------- (A) rw40_last8 / (B) start_only
ROUTE_ALL_PHOTOS = ("Each viewpoint's photograph is given once, in the order in which the viewpoints were first visited; "
                    "a later return to the same viewpoint refers back to its photograph by ID.")


def last_k_nodes(start, visited, k: int) -> set:
    """The last k distinct viewpoints in reverse visit order (excluding START)."""
    keep = []
    for n in reversed(visited):
        if n != start and n not in keep:
            keep.append(n)
            if len(keep) == k:
                break
    return set(keep)


class StartOnlyEnv(ActSpaceEnv):
    """Passive setting with START as the only photo."""

    def _static_nodes(self, group_id):
        return [self.node]


class Last8Env(ActSpaceEnv):
    """random_walk setting: route and photo IDs as in random_walk_40 (same seed), with photos only for START and the last k distinct viewpoints."""

    def _reset_route(self, start):
        k = self.data.ecfg["llm"]["history"]["k"]
        moves = self._route_moves()
        nodes = list(dict.fromkeys([self.node] + [n for _, n in moves]))
        ids = [self._photo_id(n) for n in nodes]      # same ID assignment order as random_walk, so the photo IDs are the same
        keep = last_k_nodes(self.node, [n for _, n in moves], k)
        self.shown = [n for n in nodes if n == self.node or n in keep]
        photos = [{**self._view(n), "start": n == self.node} for n in self.shown]
        route = [{"t": t, "action": a, "photo_id": self._photo_id(n), "pose": self.pose_rel_start(n)} for t, (a, n) in enumerate(moves, 1)]
        self.records = [{**start, "shown_nodes": self.shown, "shown_photo_ids": [p["photo_id"] for p in photos],
                         "route": [[a, n] for a, n in moves], "route_nodes": nodes, "route_photo_ids": ids, "window_k": k}]
        return {"questions": self.q_public, "photos": photos, "route": route, "start_pose": self.pose_rel_start(self.node), "window_k": k,
                "actions": ["answer"], "steps_used": 0, "steps_left": 0, "must_answer": True}


def last8_system(k: int) -> str:
    s = P.system_prompt("random_walk")
    assert s.count(ROUTE_ALL_PHOTOS) == 1
    return s.replace(ROUTE_ALL_PHOTOS, P.HISTORY_RULE_LAST_K_PIN.format(k=k)[2:])


SHOWN_ABOVE = "(photograph shown above)"


def last8_user(obs: dict, v2: bool = False) -> list:
    """The record of prompts.user_route verbatim; viewpoints whose photo is not given use the IMAGE_OMITTED placeholder on the line of their first visit.
    v2: on a return to a viewpoint whose photo is not given, SHOWN_ABOVE is replaced with the IMAGE_OMITTED text (square brackets changed to parentheses)."""
    img = {p["photo_id"]: p["image"] for p in obs["photos"]}
    start = obs["photos"][0]["photo_id"]
    lines, images, seen = [P.ROUTE_START.format(photo_id=start, pose=P.pose_text(obs["start_pose"]))], [(start, img[start])], {start}
    for r in obs["route"]:
        pid = r["photo_id"]
        if pid in seen:
            line = P.ROUTE_MOVE_REVISIT.format(t=r["t"], action=r["action"], photo_id=pid, pose=P.pose_text(r["pose"]))
            if v2 and pid not in img:
                assert line.count(SHOWN_ABOVE) == 1
                line = line.replace(SHOWN_ABOVE, "(" + P.IMAGE_OMITTED.format(photo_id=pid)[1:-1] + ")")
            lines.append(line)
            continue
        seen.add(pid)
        line = P.ROUTE_MOVE_NEW.format(t=r["t"], action=r["action"], photo_id=pid, pose=P.pose_text(r["pose"]))
        if pid in img:
            images.append((pid, img[pid]))
        else:
            line = line.replace("{image}", P.IMAGE_OMITTED.format(photo_id=pid))
        lines.append(line)
    text = P.USER_ROUTE.format(questions=P.format_questions(obs["questions"]), n_moves=len(obs["route"]), n_photos=len(obs["photos"]),
                               record="\n".join(lines))
    return P._parts(text, images)


class Last8Agent(LLMAgent):
    v2 = False

    def act(self, obs):
        comp = self._call(last8_system(obs["window_k"]), [{"role": "user", "parts": last8_user(obs, self.v2)}])
        act, content = parse_reply(comp.text)
        return {"action": "answer", "content": content if act == "answer" else comp.text, "notes": reply_notes(_first_json_object(comp.text or ""))}


class Last8AgentV2(Last8Agent):
    v2 = True


def source_episodes(src: Path) -> dict:
    return {ep[0]["group_id"]: ep for ep in read_episodes(src / "trajectories.jsonl") if ep[-1]["type"] == "episode_end"}


def check_route_source(data, src: Path, gids) -> int:
    """Route, nodes and photo IDs of rw40_last8 match the source random_walk_40 group by group; raises on any mismatch. Returns the number of groups checked."""
    eps = source_episodes(src)
    env = Last8Env(data, budget=40, render=False, mode="random_walk", run_seed=0)
    for g in gids:
        s = eps[g][0]
        env.reset(g)
        r = env.records[0]
        same = (r["seed"] == s["seed"] and r["route"] == [list(x) for x in s["route"]] and r["route_nodes"] == s["shown_nodes"]
                and r["route_photo_ids"] == s["shown_photo_ids"] and set(r["shown_nodes"]) <= set(s["shown_nodes"]))
        if not same:
            raise RuntimeError(f"rw40_last8: route of {g} is inconsistent with the source run {src}")
    return len(gids)


# ---------------------------------------------------------------- (D) cogmap
PROBE = """Position probe. Do not answer the questions and do not choose an action now. Instead, estimate where the center of each object listed below is, relative to your starting position and starting viewing direction, measured on the floor plane: "forward_m" is the distance along the starting viewing direction (negative = behind the starting position), "left_m" is the distance to the left of it (negative = to the right), both in meters. These are the same fixed coordinates as the poses relative to START. If you have not seen an object or cannot judge where it is, set "known" to false and give null for both numbers.

Objects:
{objects}

Reply with a single JSON object and nothing else, with one entry per object, in the order listed:
{{"objects": [{{"name": "<object name>", "known": true, "forward_m": <number>, "left_m": <number>}}, {{"name": "<object name>", "known": false, "forward_m": null, "left_m": null}}]}}"""
_NUM_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}
COGMAP_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["name", "known", "forward_m", "left_m"],
    "properties": {"name": {"type": "string"}, "known": {"type": "boolean"}, "forward_m": _NUM_OR_NULL, "left_m": _NUM_OR_NULL}}}}}


class ImageMismatch(RuntimeError):
    pass


def probe_text(names) -> str:
    return PROBE.format(objects="\n".join(f"- {n}" for n in names))


_OBJ, _VIS = {}, {}
_LOCK = threading.Lock()


def scene_objects(scene) -> dict:
    with _LOCK:
        if scene.wd not in _OBJ:
            _OBJ[scene.wd] = {int(o["obj_id"]): o for o in json.loads((scene.wd / "objects.json").read_text())}
        return _OBJ[scene.wd]


def vis_level(scene, obj: int, nodes) -> int:
    """Highest visibility level of the object at these nodes: 2 = strictly visible, 1 = at least partial, 0 = not visible anywhere."""
    from actspace_build.step4_graph_visibility import STATE_INVISIBLE, STATE_VISIBLE
    with _LOCK:
        if scene.wd not in _VIS:
            v = np.load(scene.wd / "visibility.npz")
            _VIS[scene.wd] = ({int(o): k for k, o in enumerate(v["obj_ids"])}, {str(n): k for k, n in enumerate(v["frame_ids"])}, v["state"])
        row, col, st = _VIS[scene.wd]
    if obj not in row:
        return 0
    s = [int(st[row[obj], col[n]]) for n in nodes if n in col]
    return 2 if STATE_VISIBLE in s else (1 if any(x != STATE_INVISIBLE for x in s) else 0)


def group_objects(data, gid: str) -> list[dict]:
    """Every object mentioned in the group's questions: ID, the name used in the questions, whether it is masked, and the ground truth (forward, left) relative to the start camera (meters, unrounded)."""
    g = data.groups[gid]
    scene = data.scene(g["scene_id"])
    objs = scene_objects(scene)
    masked = (g.get("view_graph_mask") or {}).get("masked_obj")
    pos, yaw = scene.pose[g["init_frame"]]
    r = np.radians(yaw)
    out = []
    for i in sorted({int(o) for q in g["questions"] for o in q["evidence_objs"]}):
        d = np.asarray(objs[i]["obb"]["center"], float)[:2] - np.asarray(pos, float)[:2]
        out.append({"obj_id": i, "name": objs[i]["ref_name"], "masked": i == masked,
                    "truth_forward_m": float(d @ np.array([np.cos(r), np.sin(r)])), "truth_left_m": float(d @ np.array([-np.sin(r), np.cos(r)]))})
    names = [o["name"] for o in out]
    assert all(names) and len(set(names)) == len(names), (gid, names)
    return out


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) else None


def parse_probe(text: str, names) -> dict:
    """-> {"json", "error", "objects": {name: {"status": known | unknown | invalid | missing, "forward_m", "left_m"}} or None (unparsable)}."""
    obj = _first_json_object(text or "")
    if obj is None:
        return {"json": False, "error": "no JSON object in reply" if (text or "").strip() else "empty reply", "objects": None}
    if not isinstance(obj.get("objects"), list):
        return {"json": True, "error": "no objects list", "objects": None}
    by = {}
    for e in obj["objects"]:
        if isinstance(e, dict) and isinstance(e.get("name"), str):
            by.setdefault(e["name"].strip().lower(), e)
    out = {}
    for n in names:
        e = by.get(n.strip().lower())
        if e is None:
            out[n] = {"status": "missing", "forward_m": None, "left_m": None}
            continue
        f, left = _num(e.get("forward_m")), _num(e.get("left_m"))
        if e.get("known") is True and f is not None and left is not None:
            out[n] = {"status": "known", "forward_m": f, "left_m": left}
        elif e.get("known") is False:
            out[n] = {"status": "unknown", "forward_m": None, "left_m": None}
        else:
            out[n] = {"status": "invalid", "forward_m": f, "left_m": left}
    return {"json": True, "error": None, "objects": out}


def answer_calls(src: Path) -> dict:
    """The last call of each group in the source run (file order) = the call in which the model answered."""
    out = {}
    with open(src / "calls.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "call":
                out[r["tag"]["episode"]] = r
    return out


def rebuild_context(data, gid: str, call: dict):
    """Call log -> (system prompt, conversation, list of frame IDs in the context, {photo ID: frame ID}). Photos are re-read and re-encoded; raises ImageMismatch if the SHA-256 does not match the log."""
    scene = data.scene(data.groups[gid]["scene_id"])
    msgs = call["request"]["messages"]
    assert msgs[0]["role"] == "system"
    turns, frames, frame_of, cache = [], [], {}, {}
    for m in msgs[1:]:
        if isinstance(m["content"], str):
            turns.append({"role": m["role"], "text": m["content"]})
            continue
        parts = []
        for c in m["content"]:
            if c["type"] == "text":
                parts.append(("text", c["text"]))
                continue
            fid = c["frame_id"]
            if fid not in cache:
                img = scene.image(fid)
                jpg = encode_jpeg(img)
                sha = hashlib.sha256(jpg).hexdigest()
                if sha != c["sha256"] or len(jpg) != c["jpeg_bytes"]:
                    raise ImageMismatch(f"cogmap: {c['photo_id']} ({fid}) of {gid} after rebuilding has SHA-256 {sha}, inconsistent with the log {c['sha256']}")
                cache[fid] = img
            parts.append(("image", c["photo_id"], cache[fid]))
            frames.append(fid)
            frame_of[c["photo_id"]] = fid
        turns.append({"role": m["role"], "parts": parts})
    assert turns and turns[-1]["role"] == "user" and "parts" in turns[-1], gid
    return msgs[0]["content"], turns, frames, frame_of


def cogmap_request(data, gid: str, call: dict, ep: list):
    """-> (system prompt, conversation with the probe prompt, object list (with visibility levels), {photo ID: frame ID})."""
    system, turns, frames, frame_of = rebuild_context(data, gid, call)
    s, end = ep[0], ep[-1]
    if (call.get("parse") or {}).get("action") == "answer" and isinstance(end["raw_answer"], dict):
        if json.loads(json.dumps(call["parse"].get("answers"))) != json.loads(json.dumps(end["raw_answer"])):
            raise RuntimeError(f"cogmap: the last call of {gid} is inconsistent with the answer in the trajectory")
    route = list(s["shown_nodes"]) if s.get("mode", "active") != "active" else [s["start_node"]] + [r["to"] for r in ep if r["type"] == "step"]
    scene = data.scene(data.groups[gid]["scene_id"])
    objects = [{**o, "seen_route": vis_level(scene, o["obj_id"], route), "seen_window": vis_level(scene, o["obj_id"], frames)} for o in group_objects(data, gid)]
    turns = turns[:-1] + [{**turns[-1], "parts": turns[-1]["parts"] + [("text", "\n\n" + probe_text([o["name"] for o in objects]))]}]
    return system, turns, objects, frame_of


def probe_records(out: Path) -> list[dict]:
    p = out / "probes.jsonl"
    return [json.loads(x) for x in open(p)] if p.exists() else []


def pending_retries(out: Path) -> list[str]:
    """Groups that failed and have not used up PROBE_ATTEMPTS attempts."""
    return [r["group_id"] for r in probe_records(out) if not r["parse_ok"] and r.get("attempts", 1) < PROBE_ATTEMPTS]


def unseal(out: Path):
    """Unseals a sealed run (for retries): restores write permission and deletes SHA256SUMS; seal_run seals it again afterwards."""
    for p in [out, *out.rglob("*")]:
        p.chmod(p.stat().st_mode | 0o200)
    (out / "SHA256SUMS").unlink(missing_ok=True)


def run_cogmap(data, reg, model: str, source: str, src: Path, gids, out: Path, pool, workers: int, resume: bool, code: dict, on_group=None) -> dict:
    ec = data.ecfg
    mcfg = {**ec["models"][model], "reply_schema": COGMAP_SCHEMA}
    label = f"cogmap_{source}"
    infeasible = set(ec["env"].get("budget_infeasible") or [])
    out.mkdir(parents=True, exist_ok=True)
    ppath, cpath = out / "probes.jsonl", out / "calls.jsonl"
    done, prior = set(), {}
    if resume:
        if (out / "SHA256SUMS").exists():
            unseal(out)
        repair_tail(ppath)
        repair_tail(cpath)
        recs = probe_records(out)
        prior = {r["group_id"]: r for r in recs if not r["parse_ok"] and r.get("attempts", 1) < PROBE_ATTEMPTS}     # groups to retry: take out their old records first
        if prior:
            ppath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs if r["group_id"] not in prior))
        done = {r["group_id"] for r in recs if r["group_id"] not in prior}
    elif ppath.exists() or cpath.exists():
        raise FileExistsError(f"{out} already has run artifacts; add --resume to continue")
    calls, eps = answer_calls(src), source_episodes(src)
    missing = [g for g in gids if g not in calls or g not in eps]
    if missing:
        raise RuntimeError(f"cogmap: source run {src} is missing {missing[:5]}")
    cost = pool.view(CostTracker.spent_in_log(cpath))
    t0 = time.time()
    env_doc = {**R.env_info(model, R.entries(reg)[model]), "run": out.name, "model": model, "setting": label, "run_seed": 0,
               "source": {"setting": source, "run": str(src), "calls_sha256": hashlib.sha256((src / "calls.jsonl").read_bytes()).hexdigest()},
               "probe": PROBE, "reply_schema": COGMAP_SCHEMA if mcfg.get("json_schema") else None, "max_output_tokens": mcfg["max_output_tokens"],
               "workers": workers, "groups": list(gids), "data_version": data.version,
               "freeze_manifest_sha256": hashlib.sha256(data.manifest_path.read_bytes()).hexdigest(), "data_source": data.source,
               "code": code, "resume": resume, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
               "probe_attempts_max": PROBE_ATTEMPTS, "retry_groups": sorted(prior)}
    prev = json.loads((out / "env.json").read_text()) if resume and (out / "env.json").exists() else None
    if prev:
        env_doc["previous_sessions"] = prev.get("previous_sessions", []) + [{k: prev.get(k) for k in ("started_at", "ended_at", "status", "code", "retry_groups")}]
    (out / "env.json").write_text(json.dumps(env_doc, ensure_ascii=False, indent=1))
    todo = queue.Queue()
    for g in gids:
        if g not in done:
            todo.put(g)
    wlock, stop = threading.Lock(), threading.Event()
    state = {"status": "complete", "err": None, "exc": None}

    def fail(status, e):
        with wlock:
            if state["status"] == "complete":
                state.update(status=status, err=str(e))
        stop.set()

    def one(client, gid):
        system, turns, objects, frame_of = cogmap_request(data, gid, calls[gid], eps[gid])
        names = [o["name"] for o in objects]
        client.cache_key, client.group_id, client.frame_of = f"actspace-{label}-{RUN}-{gid}", gid, frame_of.get
        client.parse_hook = lambda text: parse_probe(text, names)
        old = prior.get(gid)
        attempt, cost_usd = (old.get("attempts", 1) if old else 0), (old["cost_usd"] if old else 0.0)
        while True:      # reply not in probe format: resend the identical request, at most PROBE_ATTEMPTS times in total
            attempt += 1
            comp = client.complete(system, turns, tag={"episode": gid, "t": 0, "mode": "cogmap", "source": source, "attempt": attempt})
            cost_usd += comp.cost_usd
            parsed = parse_probe(comp.text, names)
            if parsed["objects"] is not None or attempt >= PROBE_ATTEMPTS:
                break
        rec = {"type": "probe", "run": out.name, "model": model, "model_name": mcfg["model"], "setting": label, "source_setting": source,
               "source_call_id": calls[gid].get("call_id"), "source_action": (calls[gid].get("parse") or {}).get("action"),
               "source_steps_used": eps[gid][-1]["steps_used"], "group_id": gid, "scene_id": data.groups[gid]["scene_id"],
               "dataset": dataset_of_scene(data.groups[gid]["scene_id"]), "budget_infeasible": gid in infeasible,
               "parse_ok": parsed["objects"] is not None, "parse_error": parsed["error"], "finish_reason": comp.finish_reason,
               "cost_usd": round(cost_usd, 6), "attempts": attempt, "first_parse_ok": attempt == 1 and parsed["objects"] is not None,
               "objects": [{**o, **(parsed["objects"][o["name"]] if parsed["objects"] else {"status": "unparsed", "forward_m": None, "left_m": None})} for o in objects]}
        with wlock:
            with open(ppath, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(gid)
            n = len(done & set(gids))
        if on_group is not None:
            on_group(n, len(gids), cost.spent)

    def loop():
        try:
            client = make_client(mcfg, ec["llm"], cost, cpath)
            client.cache_run = out.name
        except Exception as e:  # noqa: BLE001
            fail("error", e)
            state["exc"] = e
            return
        while not stop.is_set():
            try:
                gid = todo.get_nowait()
            except queue.Empty:
                return
            try:
                one(client, gid)
            except CostCapExceeded as e:
                fail("stopped_cost_cap", e)
            except ModelError as e:
                fail(getattr(e, "stop_status", None) or "stopped_server_error", e)
            except Exception as e:  # noqa: BLE001  code error (including ImageMismatch): stop
                fail("error", e)
                state["exc"] = e

    ths = [threading.Thread(target=loop, daemon=True) for _ in range(max(1, workers))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if state["exc"] is not None and state["status"] == "error":
        raise state["exc"]
    recs = probe_records(out)
    usage = CostTracker.usage_in_log(cpath)
    st = {"status": state["status"], "error": state["err"], "na_reason": None, "spent_usd": round(cost.spent, 6), "cost_cap_usd": pool.cap,
          "model_name": mcfg["model"], "reasoning_effort": mcfg["reasoning"]["effort"], "thinking": None, "setting": label, "usage": usage,
          "truncation_rate": round(usage["truncated_calls"] / usage["calls"], 6) if usage["calls"] else None,
          "completed": len(recs), "remaining": [g for g in gids if g not in {r["group_id"] for r in recs}],
          "unparsed": sum(not r["parse_ok"] for r in recs), "first_failed": sum(not r.get("first_parse_ok", r["parse_ok"]) for r in recs),
          "retried_ok": sum(r["parse_ok"] and not r.get("first_parse_ok", True) for r in recs), "probe_attempts_max": PROBE_ATTEMPTS, "seconds": round(time.time() - t0, 1), "workers": workers}
    (out / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    cnt = {}
    for r in recs:
        for o in r["objects"]:
            cnt[o["status"]] = cnt.get(o["status"], 0) + 1
    (out / "summary.json").write_text(json.dumps({"run": out.name, "model": model, "model_name": mcfg["model"], "setting": label, "source_setting": source,
                                                  "version": data.version, "groups": len(recs), "objects": sum(cnt.values()), "object_status": cnt,
                                                  "unparsed_groups": st["unparsed"], "first_failed_groups": st["first_failed"], "retried_ok_groups": st["retried_ok"]}, ensure_ascii=False, indent=1))
    env_doc.update(ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), status=st["status"], seconds=st["seconds"])
    (out / "env.json").write_text(json.dumps(env_doc, ensure_ascii=False, indent=1))
    if st["status"] == "complete":
        seal_run(out)
    return st


# ---------------------------------------------------------------- job
def source_dirs(root: Path, model: str, src_prefix: str) -> dict:
    return {s: root / model / s / f"{src_prefix}_r0" for s in ("random_walk_40", "active_v2", "oracle_route")}


def check_sources(root: Path, model: str, src_prefix: str, gids, settings) -> dict:
    """Source runs to reuse must be complete, sealed and contain all groups to run."""
    dirs = source_dirs(root, model, src_prefix)
    for name in sorted({SOURCES[s] for s in settings if s in SOURCES}):
        d = dirs[name]
        st = E.read_status(d)
        if st.get("status") != "complete" or not (d / "SHA256SUMS").exists():
            sys.exit(print(f"source run incomplete: {d} ({st.get('status')})") or 2)
        if set(gids) - set(source_episodes(d)):
            sys.exit(print(f"source run {d} does not contain all groups to run") or 2)
    return dirs


def medium_env_info(info: dict) -> dict:
    """reasoning for env.json: the registry says high; this run actually sends medium."""
    r = copy.deepcopy(info["reasoning"])
    r["target"] = MEDIUM
    for k in ("reasoning_effort",):
        if k in r["request"]:
            r["request"][k] = MEDIUM
    if "output_config" in r["request"]:
        r["request"]["output_config"]["effort"] = MEDIUM
    return {**info, "reasoning": r, "reasoning_registry_target": info["reasoning"]["target"]}


def settings_arg(s) -> list[str]:
    out = list(ORDER) if not s or s == "all" else s.split(",")
    bad = [x for x in out if x not in ORDER]
    if bad:
        raise SystemExit(f"unknown settings {bad}; choices {list(ORDER)}")
    return [x for x in ORDER if x in out]


def cmd_job(a):
    reg, data = E.load()
    models = a.model.split(",")
    unknown = [m for m in models if m not in R.entries(reg)]
    if unknown or len(set(models)) != len(models):
        sys.exit(print(f"--model has unknown or duplicate short names: {models}") or 2)
    root = R.results_root(reg)
    code = git_commit()     # the code version is taken once at job start and shared by all models and settings: Markdown written under reports/ later does not get later runs marked dirty
    for m in models:        # run in order; once any one stops (sys.exit), the rest are not started
        job_model(reg, data, argparse.Namespace(**{**vars(a), "model": m}), lambda: sum(E.scope_spent(root, x, PREFIX) for x in models), code)
    if len(models) > 1:
        print(f"== all models complete: {models}, prefix {PREFIX} total ${sum(E.scope_spent(root, x, PREFIX) for x in models):.4f} (cap ${a.cap:.2f})", flush=True)


def job_model(reg, data, a, spent_all, code):
    """All settings of one model. spent_all(): total already spent under prefix abl by all models sharing the cap. code: code version taken at job start."""
    root = R.results_root(reg)
    e = R.entries(reg)[a.model]
    src_prefix = a.source_prefix or SOURCE_PREFIX.get(a.model)
    if not src_prefix:
        sys.exit(print(f"{a.model} has no default source prefix; give --source-prefix") or 2)
    gids = group_set(data, a.set)
    settings = settings_arg(a.settings)
    print(f"== job {os.environ.get('SLURM_JOB_ID', 'local')} on {os.uname().nodename}: {a.model} ablations {settings}, {a.set} ({len(gids)} groups), prefix {PREFIX}, "
          f"source prefix {src_prefix}, cap ${a.cap:.2f}", flush=True)
    src = check_sources(root, a.model, src_prefix, gids, settings)
    todo = []
    for s in settings:
        out = root / a.model / s / RUN
        if (out / "SHA256SUMS").exists():
            retry = pending_retries(out) if s.startswith("cogmap") else []
            if retry and a.resume:
                print(f"already complete, retrying {len(retry)} groups whose replies were not in probe format: {out}")
                todo.append((s, out, True))
            else:
                print(f"already complete, skipping: {out}" + (f" ({len(retry)} failed groups can be retried; add --resume)" if retry else ""))
            continue
        resume = any((out / f).exists() for f in ("trajectories.jsonl", "calls.jsonl", "probes.jsonl"))
        if resume and not a.resume:
            sys.exit(print(f"{out} already has run artifacts; add --resume to continue") or 2)
        todo.append((s, out, resume))
    if not todo:
        print("all settings already complete")
        return
    print(E.load_keys(reg, a.model, a.key_file)["note"], flush=True)
    print("== probe", flush=True)
    E.step_probe(data, reg, argparse.Namespace(model=a.model, prefix=PREFIX, resume=a.resume, cap=a.cap - spent_all() + E.scope_spent(root, a.model, PREFIX)))
    conc = e["concurrency"]["episodes"]
    pool = CostPool(data.ecfg["models"][a.model]["prices_usd_per_mtok"], a.cap, spent_all(), slots=conc)     # prices of this model; already-spent includes the other models in the same job
    print(f"== runs: total cap ${a.cap:.2f}, already spent ${pool.spent:.4f}, concurrency {conc}, code {code}", flush=True)
    prog_dir = root / a.model / "_progress" / PREFIX
    prog_dir.mkdir(parents=True, exist_ok=True)
    info = {**R.env_info(a.model, e), "code": code}
    for s, out, resume in todo:
        if resume and not (out / "SHA256SUMS").exists():      # sealed cogmap runs (retries) are unsealed by run_cogmap
            for f in ("trajectories.jsonl", "calls.jsonl"):
                repair_tail(out / f)
        with open(prog_dir / f"{s}_r0.log", "a", buffering=1) as prog:
            def tick(done, total, spent):
                prog.write(f"{time.strftime('%F %T')} {done}/{total} run ${spent:.4f} job total ${pool.spent:.4f}\n")
            print(f"== [{time.strftime('%F %T')}] {s} started", flush=True)
            kw = dict(resume=resume, cost_cap=a.cap, run_seed=0, workers=conc, seal=True, print_estimate=False,
                      tracker_cls=lambda prices, cap, spent: pool.view(spent), on_episode=tick, label=s)
            try:
                if s == "start_only":
                    st = run_llm(data, a.model, "passive", 40, gids, out, env_cls=StartOnlyEnv,
                                 env_info={**info, "ablation": {"name": s, "photos": "START only", "prompt": "passive, n_photos = 1"}}, **kw)
                elif s in ("rw40_last8", "rw40_last8_v2"):
                    n = check_route_source(data, src["random_walk_40"], gids)
                    print(f"   routes match the source {src['random_walk_40']}: {n} groups", flush=True)
                    st = run_llm(data, a.model, "random_walk", 40, gids, out, env_cls=Last8Env, agent_cls=Last8AgentV2 if s.endswith("_v2") else Last8Agent,
                                 env_info={**info, "ablation": {"name": s, "source": str(src["random_walk_40"]), "photos": "START + last 8 distinct viewpoints",
                                                                "revisit_unshown": "(photo_id: image no longer shown)" if s.endswith("_v2") else SHOWN_ABOVE,
                                                                "system_prompt_removed": ROUTE_ALL_PHOTOS,
                                                                "system_prompt_added": P.HISTORY_RULE_LAST_K_PIN.format(k=data.ecfg["llm"]["history"]["k"])[2:]}}, **kw)
                elif s == "active_v2_medium":
                    st = run_llm(data, a.model, "active_v2", 40, gids, out, effort=MEDIUM,
                                 env_info={**medium_env_info(info), "ablation": {"name": s, "effort": MEDIUM}}, **kw)
                else:
                    source = SOURCES[s]
                    st = run_cogmap(data, reg, a.model, source, src[source], gids, out, pool, conc, resume, code, on_group=tick)
            except Exception as x:  # noqa: BLE001  code error: stop the job and log the original error
                prog.write(f"{time.strftime('%F %T')} error: {x}\n")
                print(f"=== {s}_r0 error: {type(x).__name__}: {x}", flush=True)
                sys.exit(1)
            prog.write(f"{time.strftime('%F %T')} {st['status']} {st['completed']}/{len(gids)} run ${st['spent_usd']:.4f} truncation rate {st['truncation_rate']}\n")
        print(f"=== {s}_r0 {st['status']} {st['completed']}/{len(gids)} ${st['spent_usd']:.4f} truncation {st['truncation_rate']} {st['seconds']} s" +
              (f" unparsed {st['unparsed']} first_failed {st['first_failed']} retried_ok {st['retried_ok']}" if "unparsed" in st else "") + (f" error: {st['error']}" if st["error"] else ""), flush=True)
        if st["status"] != "complete":
            code_ = E.EXIT_CODES.get(st["status"], 1)
            print(f"stopped, exit code {code_} (3 cost cap; 4 fatal error; 5 server-error retries exhausted; 6 all keys exhausted; 1 code error); completed settings and episodes are kept; continue with --resume", flush=True)
            sys.exit(code_)
    print(f"== [{time.strftime('%F %T')}] all complete, job total ${pool.spent:.4f} (including probe)", flush=True)


def cmd_check(a):
    """Sends no requests: rw40_last8 routes match the source; for both cogmap sources, every group's context can be rebuilt with matching image hashes."""
    reg, data = E.load(verify=False)
    root = R.results_root(reg)
    gids = group_set(data, a.set)
    src = check_sources(root, a.model, a.source_prefix or SOURCE_PREFIX[a.model], gids, ORDER)
    print(f"rw40_last8: routes match for {check_route_source(data, src['random_walk_40'], gids)} groups")
    for source in ("active_v2", "oracle_route"):
        calls, eps = answer_calls(src[source]), source_episodes(src[source])
        n_img = n_obj = 0
        acts = {}
        for g in gids:
            _, turns, objects, _ = cogmap_request(data, g, calls[g], eps[g])
            n_img += sum(p[0] == "image" for t in turns for p in t.get("parts", []))
            n_obj += len(objects)
            k = (calls[g].get("parse") or {}).get("action")
            acts[k] = acts.get(k, 0) + 1
        print(f"cogmap_{source}: rebuilt {len(gids)} groups, all {n_img} image hashes match, {n_obj} objects, actions of the last source calls {acts}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("job", cmd_job), ("check", cmd_check)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--model", required=True)
        p.add_argument("--set", default="all", choices=["smoke", "tiny", "all"])
        p.add_argument("--source-prefix")
        if name == "job":
            p.add_argument("--cap", type=float, required=True)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--key-file")
            p.add_argument("--settings")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
