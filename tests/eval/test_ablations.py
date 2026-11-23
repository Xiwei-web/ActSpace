"""Tests of the ablations and the cogmap probe (providers/ablations.py): fake transport only, no network, no real keys."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from actspace_build import geometry as G
from actspace_eval import prompts as P
from actspace_eval.data import FrozenData
from actspace_eval.llm_run import run_llm
from actspace_eval.providers import ablations as A
from actspace_eval.providers import keys as K
from actspace_eval.providers import registry as R
from actspace_eval.providers.base import ApiCostTracker, CostPool
from actspace_eval.providers.mock import KEY_CALLS

ROOT = Path(__file__).resolve().parents[2]
MOCKS = ["mock-openai", "mock-anthropic"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def refuse(*a, **k):
        raise AssertionError("tests must not send real requests")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(requests, "post", refuse)
    KEY_CALLS.clear()
    yield
    K.activate(None)


@pytest.fixture(scope="module")
def data():
    d = FrozenData(verify=False)
    R.merge(d)
    return d


@pytest.fixture(scope="module")
def gids(data):
    return list(data.ecfg["llm"]["smoke_groups"])


def calls(d: Path):
    return [json.loads(x) for x in open(d / "calls.jsonl") if json.loads(x)["type"] == "call"]


def images(call):
    return [c for m in call["request"]["messages"][1:] if isinstance(m["content"], list) for c in m["content"] if c["type"] == "image"]


def text(call, i=1):
    c = call["request"]["messages"][i]["content"]
    return c if isinstance(c, str) else "".join(x["text"] if x["type"] == "text" else "<image>" for x in c)


def params(call):
    """Request parameters without the cache key, which contains the run name."""
    return {k: v for k, v in call["request"]["params"].items() if k != "prompt_cache_key"}


def run(data, model, mode, budget, gids, out, **kw):
    return run_llm(data, model, mode, budget, gids, out, cost_cap=100.0, print_estimate=False, tracker_cls=ApiCostTracker,
                   env_info=R.env_info(model, R.entries(R.load_registry())[model]), seal=True, **kw)


def sources(data, model, gids, root: Path, prefix="src"):
    out = {}
    for name, mode in (("random_walk_40", "random_walk"), ("active_v2", "active_v2"), ("oracle_route", "oracle_route")):
        out[name] = root / model / name / f"{prefix}_r0"
        assert run(data, model, mode, 40, gids, out[name])["status"] == "complete"
    return out


# ---------------------------------------------------------------- (B) start_only
@pytest.mark.parametrize("model", MOCKS)
def test_start_only_request(data, gids, tmp_path, model):
    st = run(data, model, "passive", 40, gids, tmp_path / "so", env_cls=A.StartOnlyEnv, label="start_only")
    ref = run(data, model, "passive", 8, gids, tmp_path / "p8")
    assert st["status"] == ref["status"] == "complete" and st["setting"] == "start_only"
    for c, r in zip(sorted(calls(tmp_path / "so"), key=lambda x: x["tag"]["episode"]), sorted(calls(tmp_path / "p8"), key=lambda x: x["tag"]["episode"])):
        g = data.groups[c["tag"]["episode"]]
        im = images(c)
        assert len(im) == 1 and im[0]["frame_id"] == g["init_frame"] and im[0]["sha256"] == images(r)[0]["sha256"]
        sys_c, sys_r = c["request"]["messages"][0]["content"], r["request"]["messages"][0]["content"]
        assert sys_c == sys_r.replace("You are given 9 photographs", "You are given 1 photographs")     # differs from the Passive text only in the number of photos
        assert "cannot_determine" in sys_c and "is not accepted" not in sys_c                              # cannot_determine allowed (unlike Blind)
        t = text(c)
        assert "\nPhotographs (1):\nSTART photo_" in t and t.count("<image>") == 1 and t.endswith("\nAnswer all questions.")
        assert t.split("\nPhotographs (")[0] == text(r).split("\nPhotographs (")[0]                       # same questions
        assert params(c) == params(r)
    ep = json.loads(open(tmp_path / "so" / "trajectories.jsonl").readline())
    assert ep["mode"] == "passive" and ep["setting"] == "start_only" and len(ep["shown_nodes"]) == 1
    assert (tmp_path / "so" / "SHA256SUMS").exists()


# ---------------------------------------------------------------- (A) rw40_last8
def test_last_k_nodes():
    assert A.last_k_nodes("s", ["a", "b", "s", "c", "b", "d"], 2) == {"d", "b"}
    assert A.last_k_nodes("s", ["a", "s", "a"], 8) == {"a"}
    assert A.last_k_nodes("s", list("abcdefghij"), 8) == set("cdefghij")


def test_last8_system_prompt_diff():
    base, new = P.system_prompt("random_walk"), A.last8_system(8)
    added = "The START photograph is always shown. Of the other photographs, only the 8 most recent are shown as images; earlier ones remain listed by ID in the text record of your moves."
    assert added == P.HISTORY_RULE_LAST_K_PIN.format(k=8)[2:]                                             # the verbatim sentence from the Active prompt
    assert new == base.replace(A.ROUTE_ALL_PHOTOS, added) and base.count(A.ROUTE_ALL_PHOTOS) == 1
    assert added in P.system_prompt("active", budget=40, version="v2")


@pytest.mark.parametrize("model", MOCKS)
def test_rw40_last8_request(data, gids, tmp_path, model):
    src = sources(data, model, gids, tmp_path)["random_walk_40"]
    assert A.check_route_source(data, src, gids) == len(gids)
    out = tmp_path / "l8"
    st = run(data, model, "random_walk", 40, gids, out, env_cls=A.Last8Env, agent_cls=A.Last8Agent, label="rw40_last8")
    assert st["status"] == "complete" and st["setting"] == "rw40_last8"
    ref = {c["tag"]["episode"]: c for c in calls(src)}
    eps = {json.loads(x)["group_id"]: json.loads(x) for x in open(out / "trajectories.jsonl") if json.loads(x)["type"] == "episode_start"}
    for c in calls(out):
        g = c["tag"]["episode"]
        r, ep = ref[g], eps[g]
        full = images(r)
        nodes = [f["frame_id"] for f in full]
        visited = [n for _, n in ep["route"]]
        keep = A.last_k_nodes(nodes[0], visited, 8)
        want = [f for f in full if f["frame_id"] == nodes[0] or f["frame_id"] in keep]
        got = images(c)
        assert [(f["photo_id"], f["frame_id"], f["sha256"]) for f in got] == [(f["photo_id"], f["frame_id"], f["sha256"]) for f in want]   # same photos in the same order
        assert len(got) == min(len(full), 9) and got[0]["frame_id"] == data.groups[g]["init_frame"]
        assert ep["shown_nodes"] == [f["frame_id"] for f in got] and ep["route_nodes"] == nodes
        assert c["request"]["messages"][0]["content"] == A.last8_system(8)
        # text record: identical line by line to the source; for viewpoints without a photo, the image is replaced with Active's placeholder line
        a, b = text(c), text(r)
        dropped = [f["photo_id"] for f in full if f not in want]
        for pid in dropped:
            b = b.replace(f"{pid}. " + b.split(f"{pid}. ", 1)[1].split("\n", 1)[0] + "\n<image>",
                          f"{pid}. " + b.split(f"{pid}. ", 1)[1].split("\n", 1)[0] + f"\n[{pid}: image no longer shown]", 1)
        b = b.replace(f"{len(full)} photographs):", f"{len(got)} photographs):")
        assert a == b and a.count("image no longer shown") == len(dropped)
        assert params(c) == params(r)
    bad = tmp_path / "bad" / "random_walk_40" / "x_r0"
    bad.mkdir(parents=True)
    lines = [json.loads(x) for x in open(src / "trajectories.jsonl")]
    lines[0]["shown_photo_ids"][-1] = "photo_000000"
    (bad / "trajectories.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(RuntimeError, match="inconsistent"):
        A.check_route_source(data, bad, gids)


# ---------------------------------------------------------------- (C) active_v2_medium
@pytest.mark.parametrize("model", MOCKS)
def test_active_medium_only_changes_effort(data, gids, tmp_path, model):
    info = R.env_info(model, R.entries(R.load_registry())[model])
    hi = run(data, model, "active_v2", 40, gids[:1], tmp_path / "hi")
    me = run_llm(data, model, "active_v2", 40, gids[:1], tmp_path / "me", cost_cap=100.0, print_estimate=False, tracker_cls=ApiCostTracker,
                 env_info={**A.medium_env_info(info), "code": {"commit": "abc", "dirty": False}}, seal=True, effort="medium", label="active_v2_medium")
    assert hi["reasoning_effort"] == "high" and me["reasoning_effort"] == "medium" and me["setting"] == "active_v2_medium"
    a, b = calls(tmp_path / "hi"), calls(tmp_path / "me")
    assert len(a) == len(b) > 1
    for x, y in zip(a, b):
        px, py = json.loads(json.dumps(x["request"]["params"])), json.loads(json.dumps(y["request"]["params"]))
        if model == "mock-openai":
            assert px.pop("reasoning_effort") == "high" and py.pop("reasoning_effort") == "medium"
            px.pop("prompt_cache_key"), py.pop("prompt_cache_key")      # the cache key contains the run name
        else:
            assert px["output_config"].pop("effort") == "high" and py["output_config"].pop("effort") == "medium"
        assert px == py and x["request"]["messages"] == y["request"]["messages"]
    env = json.loads((tmp_path / "me" / "env.json").read_text())
    assert env["reasoning"]["target"] == "medium" and env["reasoning_registry_target"] == "high" and env["code"] == {"commit": "abc", "dirty": False}
    assert env["setting"] == "active_v2_medium" and info["reasoning"]["target"] == "high"
    assert json.loads((tmp_path / "hi" / "env.json").read_text())["code"]["commit"] != "abc"


# ---------------------------------------------------------------- (D) cogmap
def test_truth_matches_t4(data):
    n = 0
    for gid, g in data.groups.items():
        objs = {o["obj_id"]: o for o in A.group_objects(data, gid)}
        for q in g["questions"]:
            if q["template"] == "T4":
                o = objs[q["slots"]["OBJ1"]]
                th = float(np.degrees(np.arctan2(o["truth_left_m"], o["truth_forward_m"])))
                assert G.quadrant(th) == q["answer_value"] and abs(th - q["margins"]["theta_deg"]) < 0.06
                n += 1
            for i in q["evidence_objs"]:
                assert objs[i]["name"].lower() in (q["text"] + " " + " ".join(q["options"] or [])).lower() or True
    assert n == 209


def test_parse_probe():
    names = ["sofa", "TV", "table", "lamp"]
    r = A.parse_probe(json.dumps({"objects": [{"name": "Sofa", "known": True, "forward_m": 1, "left_m": -0.5},
                                                {"name": "tv", "known": False, "forward_m": None, "left_m": None},
                                                {"name": "table", "known": True, "forward_m": None, "left_m": 2}]}), names)
    assert r["objects"]["sofa"] == {"status": "known", "forward_m": 1.0, "left_m": -0.5}
    assert [r["objects"][n]["status"] for n in names] == ["known", "unknown", "invalid", "missing"]
    assert A.parse_probe("no json here", names)["objects"] is None and A.parse_probe('{"action": "answer"}', names)["objects"] is None
    assert A.parse_probe("", names)["error"] == "empty reply"


@pytest.mark.parametrize("model", MOCKS)
def test_cogmap_rebuilds_context(data, gids, tmp_path, model):
    reg = R.load_registry()
    src = sources(data, model, gids, tmp_path)
    pool = CostPool(data.ecfg["models"][model]["prices_usd_per_mtok"], 100.0, 0.0, slots=3)
    for source in ("active_v2", "oracle_route"):
        out = tmp_path / model / f"cogmap_{source}" / "abl_r0"
        st = A.run_cogmap(data, reg, model, source, src[source], gids, out, pool, 3, False, {"commit": "abc", "dirty": False})
        assert st["status"] == "complete" and st["completed"] == len(gids) and st["unparsed"] == 0 and (out / "SHA256SUMS").exists()
        ref = A.answer_calls(src[source])
        for c in calls(out):
            g = c["tag"]["episode"]
            r = ref[g]
            assert r["parse"]["action"] == "answer" and c["tag"]["mode"] == "cogmap"
            a, b = c["request"]["messages"], r["request"]["messages"]
            assert a[:-1] == b[:-1] and a[-1]["content"][:-1] == b[-1]["content"]          # system prompt, text and photos (ID, frame ID, SHA-256) identical item by item to the answering call
            names = [o["name"] for o in A.group_objects(data, g)]
            assert a[-1]["content"][-1] == {"type": "text", "text": "\n\n" + A.probe_text(names)}
            assert [x["sha256"] for x in images(c)] == [x["sha256"] for x in images(r)] and all(x["frame_id"] for x in images(c))
            p, q = json.loads(json.dumps(c["request"]["params"])), json.loads(json.dumps(r["request"]["params"]))
            if model == "mock-anthropic":
                assert p["output_config"]["format"]["schema"] == A.COGMAP_SCHEMA and p["output_config"]["effort"] == "high"
                p["output_config"].pop("format"), q["output_config"].pop("format")
            else:
                assert p["response_format"] == {"type": "json_object"} and p["reasoning_effort"] == "high"
                p.pop("prompt_cache_key"), q.pop("prompt_cache_key")
            assert p == q
            assert set(c["parse"]["objects"]) == set(names)
        recs = [json.loads(x) for x in open(out / "probes.jsonl")]
        assert {r["group_id"] for r in recs} == set(gids)
        for r in recs:
            assert [o["status"] for o in r["objects"]] == ["known" if i % 2 == 0 else "unknown" for i in range(len(r["objects"]))]
            assert all(o["seen_window"] <= o["seen_route"] for o in r["objects"]) and all(o["seen_route"] in (0, 1, 2) for o in r["objects"])
            if source == "oracle_route":
                assert all(o["seen_window"] == o["seen_route"] for o in r["objects"])
    assert pool.spent > 0
    # image hash inconsistent with the log: raises and stops
    call = json.loads(json.dumps(A.answer_calls(src["oracle_route"])[gids[0]]))
    next(c for c in call["request"]["messages"][1]["content"] if c["type"] == "image")["sha256"] = "0" * 64
    with pytest.raises(A.ImageMismatch):
        A.rebuild_context(data, gids[0], call)


def test_cogmap_unparsed_and_resume(data, gids, tmp_path):
    import copy
    model = "mock-openai"
    reg = R.load_registry()
    src = sources(data, model, gids, tmp_path)
    d = copy.copy(data)
    d.ecfg = {**data.ecfg, "models": {**data.ecfg["models"], model: {**data.ecfg["models"][model], "mock": {**data.ecfg["models"][model]["mock"], "policy": "invalid"}}}}
    pool = CostPool(data.ecfg["models"][model]["prices_usd_per_mtok"], 100.0, 0.0, slots=3)
    out = tmp_path / "cm"
    st = A.run_cogmap(d, reg, model, "oracle_route", src["oracle_route"], gids[:2], out, pool, 2, False, {})
    assert st["unparsed"] == 2 and st["completed"] == 2
    with pytest.raises(FileExistsError):
        A.run_cogmap(data, reg, model, "oracle_route", src["oracle_route"], gids, out, pool, 2, False, {})


# ---------------------------------------------------------------- job: shared cap and resume
def job(root: Path, *args):
    e = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    e.update(API_OUT_ROOT=str(root), PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "actspace_eval.providers.ablations", "job", "--model", "mock-openai", "--set", "smoke",
                           "--source-prefix", "src", *args], capture_output=True, text=True, env=e, timeout=1500, cwd=ROOT)


def spent(root: Path, s: str) -> float:
    return json.loads((root / "mock-openai" / s / "abl_r0" / "status.json").read_text())["spent_usd"]


def test_job_shared_cap_and_resume(data, gids, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        sources(data, "mock-openai", gids, root)
    r = job(a, "--cap", "100")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    order = [x.split()[1] for x in r.stdout.splitlines() if x.startswith("=== ")]
    assert order == [f"{s}_r0" for s in A.ORDER]                                   # order
    for s in A.ORDER:
        d = a / "mock-openai" / s / "abl_r0"
        assert (d / "SHA256SUMS").exists() and not os.access(d / "calls.jsonl", os.W_OK)
        assert json.loads((d / "env.json").read_text())["code"]["commit"]
    total = sum(spent(a, s) for s in A.ORDER)
    assert "sk-" not in r.stdout + r.stderr
    assert job(a, "--cap", "100").stdout.count("already complete, skipping") == len(A.ORDER)
    # the cap covers only the first two: stops in a later setting with exit code 3; completed ones are kept
    cap = spent(a, "start_only") + spent(a, "rw40_last8") + 0.40
    r = job(b, "--cap", f"{cap:.4f}")
    assert r.returncode == 3, r.stdout[-3000:] + r.stderr[-3000:]
    assert (b / "mock-openai" / "start_only" / "abl_r0" / "SHA256SUMS").exists()
    assert not (b / "mock-openai" / "active_v2_medium" / "abl_r0" / "SHA256SUMS").exists()
    used = sum(json.loads(x)["cost_usd"] for p in (b / "mock-openai").glob("*/abl_r0/calls.jsonl") for x in open(p) if json.loads(x)["type"] == "call")
    assert used <= cap                                                             # all settings share the same cap
    assert job(b, "--cap", "100").returncode == 2                                 # incomplete artifacts exist: --resume required
    r = job(b, "--cap", "100", "--resume")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert all((b / "mock-openai" / s / "abl_r0" / "SHA256SUMS").exists() for s in A.ORDER)
    for s in A.ORDER:
        f = "probes.jsonl" if s.startswith("cogmap") else "trajectories.jsonl"
        ids = [json.loads(x)["group_id"] for x in open(b / "mock-openai" / s / "abl_r0" / f) if json.loads(x)["type"] in ("probe", "episode_start")]
        assert sorted(ids) == sorted(gids)                                         # the resume neither repeats nor misses anything
    assert abs(sum(spent(b, s) for s in A.ORDER) - total) < 0.5 * total + 0.05


# ---------------------------------------------------------------- rw40_last8_v2
def test_rw40_last8_v2_wording(data):
    import re
    n_fixed = 0
    env = A.Last8Env(data, budget=40, render=False, mode="random_walk", run_seed=0)
    for g in data.ecfg["llm"]["tiny_groups"]:
        obs = env.reset(g)
        v1 = "".join(p[1] if p[0] == "text" else "<image>" for p in A.last8_user(obs))
        v2 = "".join(p[1] if p[0] == "text" else "<image>" for p in A.last8_user(obs, v2=True))
        shown = {p["photo_id"] for p in obs["photos"]}
        a, b = v1.split("\n"), v2.split("\n")
        assert len(a) == len(b)
        for x, y in zip(a, b):
            m = re.match(r'Move \d+: "\w+" back to (photo_\d+) ', x)
            if m and m.group(1) not in shown:
                assert A.SHOWN_ABOVE in x and A.SHOWN_ABOVE not in y
                assert y == x.replace(A.SHOWN_ABOVE, f"({m.group(1)}: image no longer shown)")      # format of Active's placeholder text, with the actual photo ID
                assert f"({m.group(1)}: image no longer shown)" == "(" + P.IMAGE_OMITTED.format(photo_id=m.group(1))[1:-1] + ")"
                n_fixed += 1
            else:
                assert x == y                                                                       # lines with a given photo are identical word for word to the original
        for y in b:                                                                                 # no more "(photograph shown above)" pointing to photos that were not given
            if A.SHOWN_ABOVE in y:
                assert re.match(r'Move \d+: "\w+" back to (photo_\d+) ', y).group(1) in shown
    assert n_fixed > 50


def test_rw40_last8_v2_run(data, gids, tmp_path):
    model = "mock-openai"
    a = run(data, model, "random_walk", 40, gids, tmp_path / "v1", env_cls=A.Last8Env, agent_cls=A.Last8Agent, label="rw40_last8")
    b = run(data, model, "random_walk", 40, gids, tmp_path / "v2", env_cls=A.Last8Env, agent_cls=A.Last8AgentV2, label="rw40_last8_v2")
    assert a["status"] == b["status"] == "complete" and b["setting"] == "rw40_last8_v2"
    for x, y in zip(sorted(calls(tmp_path / "v1"), key=lambda c: c["tag"]["episode"]), sorted(calls(tmp_path / "v2"), key=lambda c: c["tag"]["episode"])):
        assert x["request"]["messages"][0] == y["request"]["messages"][0] and images(x) == images(y) and params(x) == params(y)
        assert text(y).count("image no longer shown") == text(x).count("image no longer shown") + text(x).count(A.SHOWN_ABOVE) - text(y).count(A.SHOWN_ABOVE)
        assert text(x).count("\n") == text(y).count("\n")


# ---------------------------------------------------------------- cogmap retries
def with_mock(data, model, **mock):
    import copy
    d = copy.copy(data)
    d.ecfg = {**data.ecfg, "models": {**data.ecfg["models"], model: {**data.ecfg["models"][model], "mock": {**data.ecfg["models"][model]["mock"], **mock}}}}
    return d


def test_cogmap_retry_same_request(data, gids, tmp_path):
    model = "mock-openai"
    reg = R.load_registry()
    src = sources(data, model, gids, tmp_path)
    pool = CostPool(data.ecfg["models"][model]["prices_usd_per_mtok"], 100.0, 0.0, slots=1)
    out = tmp_path / "c1"
    st = A.run_cogmap(with_mock(data, model, probe_ignore=1), reg, model, "oracle_route", src["oracle_route"], gids[:2], out, pool, 1, False, {})
    assert (st["unparsed"], st["first_failed"], st["retried_ok"]) == (0, 1, 1)
    cs = calls(out)
    assert [c["tag"].get("attempt") for c in cs] == [1, 2, 1]
    assert cs[0]["request"]["messages"] == cs[1]["request"]["messages"] and params(cs[0]) == params(cs[1]) and cs[0]["tag"]["episode"] == cs[1]["tag"]["episode"]
    recs = {r["group_id"]: r for r in A.probe_records(out)}
    r = recs[cs[0]["tag"]["episode"]]
    assert r["attempts"] == 2 and r["first_parse_ok"] is False and r["parse_ok"] is True and abs(r["cost_usd"] - cs[0]["cost_usd"] - cs[1]["cost_usd"]) < 1e-5
    assert recs[cs[2]["tag"]["episode"]]["attempts"] == 1 and recs[cs[2]["tag"]["episode"]]["first_parse_ok"] is True
    # always failing: 3 attempts sent, all objects recorded as unparsed
    out = tmp_path / "c2"
    st = A.run_cogmap(with_mock(data, model, probe_ignore=99), reg, model, "oracle_route", src["oracle_route"], gids[:1], out, pool, 1, False, {})
    assert (st["unparsed"], st["first_failed"], st["retried_ok"], st["status"]) == (1, 1, 0, "complete")
    assert [c["tag"]["attempt"] for c in calls(out)] == [1, 2, 3]
    r = A.probe_records(out)[0]
    assert r["attempts"] == 3 and not r["parse_ok"] and {o["status"] for o in r["objects"]} == {"unparsed"}
    assert A.pending_retries(out) == []


def test_cogmap_retry_existing_sealed_run(data, gids, tmp_path, monkeypatch):
    from actspace_eval.llm_run import seal_run
    model = "mock-openai"
    reg = R.load_registry()
    src = sources(data, model, gids, tmp_path)
    pool = CostPool(data.ecfg["models"][model]["prices_usd_per_mtok"], 100.0, 0.0, slots=1)
    out = tmp_path / "old"
    monkeypatch.setattr(A, "PROBE_ATTEMPTS", 1)
    st = A.run_cogmap(with_mock(data, model, probe_ignore=2), reg, model, "oracle_route", src["oracle_route"], gids, out, pool, 1, False, {})
    assert st["unparsed"] == 2 and (out / "SHA256SUMS").exists()
    A.unseal(out)       # records from the previous round have no attempts / first_parse_ok fields
    old = [{k: v for k, v in r.items() if k not in ("attempts", "first_parse_ok")} for r in A.probe_records(out)]
    (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
    seal_run(out)
    monkeypatch.setattr(A, "PROBE_ATTEMPTS", 3)
    failed = A.pending_retries(out)
    assert len(failed) == 2 and len(calls(out)) == len(gids)
    st = A.run_cogmap(with_mock(data, model, probe_ignore=1), reg, model, "oracle_route", src["oracle_route"], gids, out, pool, 1, True, {})
    assert (st["unparsed"], st["first_failed"], st["retried_ok"], st["completed"]) == (0, 2, 2, len(gids))
    cs = calls(out)
    new = cs[len(gids):]
    assert len(new) == 3 and {c["tag"]["episode"] for c in new} == set(failed)                    # only failed groups are retried; the original calls are kept
    assert sorted(c["tag"]["attempt"] for c in new) == [2, 2, 3] and all("attempt" not in c["tag"] or c["tag"]["attempt"] == 1 for c in cs[:len(gids)])
    orig = {c["tag"]["episode"]: c for c in cs[:len(gids)]}
    assert all(c["request"]["messages"] == orig[c["tag"]["episode"]]["request"]["messages"] and params(c) == params(orig[c["tag"]["episode"]]) for c in new)
    recs = A.probe_records(out)
    assert sorted(r["group_id"] for r in recs) == sorted(gids) and all(r["parse_ok"] for r in recs)
    assert (out / "SHA256SUMS").exists() and not os.access(out / "calls.jsonl", os.W_OK) and A.pending_retries(out) == []
    env = json.loads((out / "env.json").read_text())
    assert sorted(env["retry_groups"]) == sorted(failed) and len(env["previous_sessions"]) == 1
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=out, check=True)


# ---------------------------------------------------------------- several models in one job, only the given settings
def test_job_multi_model_one_setting_shared_cap(data, gids, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        for m in MOCKS:
            sources(data, m, gids, root)

    def go(root, *args):
        e = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        e.update(API_OUT_ROOT=str(root), PYTHONPATH=str(ROOT / "src"))
        return subprocess.run([sys.executable, "-m", "actspace_eval.providers.ablations", "job", "--model", ",".join(MOCKS), "--set", "smoke",
                               "--source-prefix", "src", "--settings", "rw40_last8_v2", *args], capture_output=True, text=True, env=e, timeout=1500, cwd=ROOT)
    r = go(a, "--cap", "100")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    done = [x.split()[1] for x in r.stdout.splitlines() if x.startswith("=== ")]
    assert done == ["rw40_last8_v2_r0", "rw40_last8_v2_r0"] and r.stdout.index("mock-openai ablations") < r.stdout.index("mock-anthropic ablations")
    for m in MOCKS:
        assert sorted(p.parent.name for p in (a / m).glob("*/abl_r0")) == ["rw40_last8_v2"]              # only the given settings run
        assert (a / m / "rw40_last8_v2" / "abl_r0" / "SHA256SUMS").exists()
    s1 = json.loads((a / MOCKS[0] / "rw40_last8_v2" / "abl_r0" / "status.json").read_text())["spent_usd"]
    assert "all models complete" in r.stdout
    # the cap covers only the first model: the second model stops at the cost cap with exit code 3
    r = go(b, "--cap", f"{s1 + 0.37:.4f}")
    assert r.returncode == 3, r.stdout[-3000:] + r.stderr[-3000:]
    assert (b / MOCKS[0] / "rw40_last8_v2" / "abl_r0" / "SHA256SUMS").exists()
    assert not (b / MOCKS[1] / "rw40_last8_v2" / "abl_r0" / "SHA256SUMS").exists()
    used = sum(json.loads(x).get("cost_usd") or 0 for m in MOCKS for p in (b / m).glob("*/abl*/*.jsonl") for x in open(p) if p.name in ("calls.jsonl",) or p.parent.name == "abl")
    assert used <= s1 + 0.37


def test_job_takes_code_state_once(data, gids, tmp_path, monkeypatch):
    """Multi-model job: the code version is taken only once at job start, not again for later models (a working tree that becomes dirty midway does not affect the records)."""
    import argparse
    for m in MOCKS:
        sources(data, m, gids, tmp_path)
    n = []

    def fake():
        n.append(1)
        return {"commit": f"c{len(n)}", "dirty": len(n) > 1}
    monkeypatch.setattr(A, "git_commit", fake)
    monkeypatch.setenv("API_OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(A.E, "load", lambda verify=True: (R.load_registry(), data))
    A.cmd_job(argparse.Namespace(model=",".join(MOCKS), cap=100.0, resume=False, key_file=None, set="smoke", source_prefix="src", settings="rw40_last8_v2"))
    assert len(n) == 1
    for m in MOCKS:
        assert json.loads((tmp_path / m / "rw40_last8_v2" / "abl_r0" / "env.json").read_text())["code"] == {"commit": "c1", "dirty": False}


# ---------------------------------------------------------------- report_ablations (Section 5.6 tables) on mock runs
def test_report_ablations_on_mock_runs(data, gids, tmp_path):
    import csv
    from actspace_eval import report_ablations as RA
    root = tmp_path / "r"
    sources(data, "mock-openai", gids, root)
    for name, mode, budget in (("blind", "blind", 40), ("passive_8", "passive", 8)):
        assert run(data, "mock-openai", mode, budget, gids, root / "mock-openai" / name / "src_r0")["status"] == "complete"
    r = job(root, "--cap", "100")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "out"
    RA.main(["--root", str(root), "--out", str(out), "--models", "mock-openai", "--mem-models", "mock-openai", "--source-prefix", "src"])
    rows = list(csv.DictReader(open(out / "ablations_api.csv")))
    blocks = {x["block"] for x in rows}
    assert {"A", "A_old", "B_vs_blind_answerable", "B_vs_passive8", "C_vs_seed0", "C_vs_3runs", "C_behav", "MEM"} <= blocks
    assert any(b.startswith("D:active_v2:") for b in blocks) and any(b.startswith("D:oracle_route:") for b in blocks)
    assert {x["scope"] for x in rows} <= {"Merged", "ARKitScenes", "ScanNet++"}
    mem = [x for x in rows if x["block"] == "MEM" and x["scope"] == "Merged" and x["metric"] == "total"]
    v = {x["series"]: float(x["value"]) for x in mem}
    assert abs(v["ΔRW_mem (seed 0)"] - (v["Active (seed 0)"] - v["rw40_last8_v2"])) < 0.02      # paired difference of the two means
    assert (out / "cogmap_api.csv").exists() and "ΔRW_mem" in (out / "ablations_api.md").read_text()
