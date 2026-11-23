"""Official-evaluation changes: merged_v1.0, two datasets, ScanNet++ downscaling, per-dataset step scale, high by default,
repeated runs and "scene bootstrap + repeats" intervals, per-dataset score tables, new tiny set. Mock only, no real calls."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from actspace_eval.data import FrozenData, dataset_of_scene
from actspace_eval.llm_run import run_llm
from actspace_eval.score import RepBoot
from actspace_eval.trajectory import complete_episodes, replay

ROOT = Path(__file__).resolve().parents[2]
TWO = ["arkit_g_000000", "scannetpp_g_000009"]   # one group per dataset (scannetpp_g_000009 is a masked group)


@pytest.fixture()
def data():
    return FrozenData(verify=False)


def run(data, tmp_path, setting, name, gids=TWO, budget=40, **kw):
    return run_llm(data, "mock", setting, budget, gids, tmp_path / name, print_estimate=False, **kw)


def test_merged_version_and_datasets(data):
    assert data.version == "merged_v1.0" and data.merged and len(data.groups) == 234
    assert {data.dataset(g) for g in data.group_ids} == {"arkitscenes", "scannetpp"}
    assert sum(len(g["questions"]) for g in data.groups.values()) == 994


def test_scannetpp_frames_downscaled_to_640x480(data):
    sp = data.scene(data.groups["scannetpp_g_000009"]["scene_id"])
    ak = data.scene(data.groups["arkit_g_000000"]["scene_id"])
    img = sp.image(data.groups["scannetpp_g_000009"]["init_frame"])
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert ak.image(data.groups["arkit_g_000000"]["init_frame"]).shape[:2] in ((480, 640), (640, 480))


@pytest.mark.parametrize("setting,budget,seed", [("blind", 40, 0), ("passive", 8, 0), ("passive", 20, 0), ("passive", 40, 0),
                                                 ("random_walk", 40, 2), ("oracle_min", 40, 0), ("oracle_route", 40, 0),
                                                 ("active_v2", 40, 1)])
def test_all_settings_both_datasets_rendered(data, tmp_path, setting, budget, seed):
    """All settings × both datasets, rendering images: complete, replayable, dataset recorded; every image sent is 640×480 (ScanNet++ downscaled)."""
    st = run(data, tmp_path, setting, "r", budget=budget, run_seed=seed)
    assert st["status"] == "complete" and st["completed"] == 2
    p = tmp_path / "r" / "trajectories.jsonl"
    assert replay(data, p)["mismatched_episodes"] == 0
    eps = complete_episodes(p)
    assert {e[0]["dataset"] for e in eps} == {"arkitscenes", "scannetpp"}
    log = (tmp_path / "r" / "calls.jsonl").read_text()
    if setting != "blind":
        assert "<image " in log and "1920x1440" not in log and ("640x480" in log or "480x640" in log)
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert {dataset_of_scene(x["scene_id"]) for x in s["per_question"]} == {"arkitscenes", "scannetpp"}


def test_step_scale_per_dataset(data, tmp_path):
    """The per-step scale is filled into the prompt according to the episode's dataset and written to the trajectory records."""
    d = copy.copy(data)
    d.ecfg = copy.deepcopy(data.ecfg)
    d.ecfg["env"]["step_scale"]["scannetpp"] = {"move_m": 0.4, "turn_deg": 40}
    assert run(d, tmp_path, "active_v2", "r", render=False)["status"] == "complete"
    calls = [json.loads(l) for l in open(tmp_path / "r" / "calls.jsonl")]
    sys_by_ep = {c["tag"]["episode"]: c["request"]["messages"][0]["content"] for c in calls}
    assert "A move covers about 0.3 m." in sys_by_ep["arkit_g_000000"] and "by about 45 degrees" in sys_by_ep["arkit_g_000000"]
    assert "A move covers about 0.4 m." in sys_by_ep["scannetpp_g_000009"] and "by about 40 degrees" in sys_by_ep["scannetpp_g_000009"]
    starts = {e[0]["group_id"]: e[0] for e in complete_episodes(tmp_path / "r" / "trajectories.jsonl")}
    assert starts["scannetpp_g_000009"]["step_scale"] == {"move_m": 0.4, "turn_deg": 40}
    assert starts["arkit_g_000000"]["step_scale"] == {"move_m": 0.3, "turn_deg": 45}
    assert data.ecfg["env"]["step_scale"] == {"arkitscenes": {"move_m": 0.3, "turn_deg": 45}, "scannetpp": {"move_m": 0.3, "turn_deg": 45}}


def test_effort_high_recorded(data, tmp_path):
    """The official evaluation always uses high: models with reasoning use llm.reasoning_effort when --effort is not given, written to every record (mock has no reasoning, recorded as None)."""
    from actspace_eval.llm_run import default_effort, with_effort
    for m, mc in data.ecfg["models"].items():
        if mc.get("reasoning", {}).get("enabled"):
            assert with_effort(mc, default_effort(data.ecfg, m, None))["reasoning"]["effort"] == "high"
    st = run(data, tmp_path, "blind", "r", render=False)
    assert "reasoning_effort" in st
    ep = complete_episodes(tmp_path / "r" / "trajectories.jsonl")[0][0]
    assert "reasoning_effort" in ep


def test_repboot_properties():
    rng = np.random.default_rng(0)
    scenes = {"arkitscenes": [f"arkit_{i}" for i in range(10)], "scannetpp": [f"scannetpp_{i}" for i in range(8)]}
    one = [(s, float(rng.random() < 0.6)) for ds in scenes.values() for s in ds for _ in range(4)]
    b = RepBoot(scenes)
    a1 = b.mean_ci([one], "x")
    a3 = b.mean_ci([one, one, one], "x")        # three identical runs: point estimate and interval unchanged
    assert a1["mean"] == pytest.approx(a3["mean"]) and a1["lo"] == pytest.approx(a3["lo"]) and a1["hi"] == pytest.approx(a3["hi"])
    zero = [(s, 0.0) for s, _ in one]             # three systematically different runs: with repeats resampled, the interval is wider than when they are pooled as one run
    w = b.mean_ci([one, zero, one], "y")
    pooled = b.mean_ci([one + zero + one], "y")
    assert w["mean"] == pytest.approx(pooled["mean"]) and (w["hi"] - w["lo"]) > (pooled["hi"] - pooled["lo"])
    # a single dataset: only that dataset's scenes
    ba = RepBoot(scenes, scope="arkitscenes")
    ca = ba.mean_ci([one], "x")
    assert ca["n"] == 40 and ca["mean"] == pytest.approx(np.mean([v for s, v in one if s.startswith("arkit_")]))
    # pooled results are stratified by dataset: each resample keeps the scene count of both datasets
    assert (b.w[:, :10].sum(1) == 10).all() and (b.w[:, 10:].sum(1) == 8).all()
    d = b.diff_ci([one], "x", [one], "z")
    assert d["mean"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_report_eval_with_repeats(data, tmp_path):
    from actspace_eval import report_eval as RE
    for setting, name, seed in [("blind", "b", 0), ("passive", "p40", 0), ("random_walk", "rw0", 0), ("random_walk", "rw1", 1),
                                ("active_v2", "a0", 0), ("active_v2", "a1", 1)]:
        assert run(data, tmp_path, setting, name, render=False, run_seed=seed)["status"] == "complete"
    spec = {"blind": ["b"], "passive_40": ["p40"], "random_walk_40": ["rw0", "rw1"], "active_v2": ["a0", "a1"]}
    R = RE.build(tmp_path, spec)
    text = "\n".join(RE.render(R))
    assert "## 1. ARKitScenes" in text and "## 2. ScanNet++" in text and "## 3. Merged" in text
    assert "Active − random_walk_40 (primary metric)" in text and "Active − Passive-40 (secondary metric)" in text
    heads = [l for l in text.splitlines() if l.startswith("| Setting | Runs")]
    assert len(heads) == 3 and all("Answer rate" in h and "Selective score" in h and "correct abstention" in h for h in heads)
    assert ["Evidence hit" in h for h in heads] == [True, True, False]    # evidence hits only per dataset
    assert "| Active v2 | 2 |" in text and "| random_walk_40 | 2 |" in text


def test_tiny_merged_config_reproducible(data):
    from actspace_eval.meta import group_meta
    from actspace_eval.tiny import select_merged
    tiny = data.ecfg["llm"]["tiny_groups"]
    assert len(tiny) == 32 and sum(data.dataset(g) == "arkitscenes" for g in tiny) == 16
    picked, s, _, _ = select_merged(data, group_meta(data))
    assert picked == sorted(tiny) and s == 76
    assert set(data.ecfg["llm"]["smoke_groups"]) <= set(data.group_ids)


def test_cost_plan_tables_plan_b_halves_only_astra_active():
    """Cost plans: A = Active and random_walk 3 times each; B = only astra's Active reduced to once."""
    from actspace_eval.cost_plan import tables
    rows = [{"model": m, "setting": s, "set": "tiny32", "repeats": r, "per_run": {"standard_usd": 1.0, "upper_usd": 2.0}}
            for m in ("gpt-6.1-sol", "gpt-6-astra") for s, r in (("blind", 1), ("random_walk_40", 3), ("active_v2", 3))]
    T = tables({"rows": rows})
    a, b = T["tiny32|A"], T["tiny32|B"]
    assert a["gpt-6-astra"]["active_v2"] == (3.0, 6.0, 3) and b["gpt-6-astra"]["active_v2"] == (1.0, 2.0, 1)
    assert b["gpt-6.1-sol"]["active_v2"] == (3.0, 6.0, 3) and b["gpt-6-astra"]["random_walk_40"] == (3.0, 6.0, 3)
