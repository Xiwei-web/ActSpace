"""End-to-end mock smoke test on the synthetic dataset (tests/synthetic.py): every evaluation setting on both datasets with the
mock model, replay of the trajectories, image sizes sent to the model, scoring and the score report. No real data, no API calls."""
import json

import numpy as np
import pytest

from actspace_eval.data import dataset_of_scene
from actspace_eval.llm_run import run_llm
from actspace_eval.trajectory import complete_episodes, replay
from synthetic import GROUPS


def run(data, tmp_path, setting, name, budget=40, **kw):
    return run_llm(data, "mock", setting, budget, GROUPS, tmp_path / name, print_estimate=False, **kw)


def test_synthetic_data_layout(synth):
    assert synth.merged and set(synth.group_ids) == set(GROUPS)
    assert {synth.dataset(g) for g in GROUPS} == {"arkitscenes", "scannetpp"}
    g = synth.groups["scannetpp_g_000001"]
    img = synth.scene(g["scene_id"]).image(g["init_frame"])
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8          # ScanNet++ 1920x1440 downscaled
    assert synth.masked_nodes("scannetpp_g_000001") and not synth.masked_nodes("arkit_g_000000")


@pytest.mark.parametrize("setting,budget,seed", [("blind", 40, 0), ("passive", 8, 0), ("passive", 20, 0), ("passive", 40, 0),
                                                 ("random_walk", 40, 2), ("oracle_min", 40, 0), ("oracle_route", 40, 0),
                                                 ("active_v2", 40, 1)])
def test_all_settings_both_datasets(synth, tmp_path, setting, budget, seed):
    st = run(synth, tmp_path, setting, "r", budget=budget, run_seed=seed)
    assert st["status"] == "complete" and st["completed"] == 2
    p = tmp_path / "r" / "trajectories.jsonl"
    assert replay(synth, p)["mismatched_episodes"] == 0
    assert {e[0]["dataset"] for e in complete_episodes(p)} == {"arkitscenes", "scannetpp"}
    log = (tmp_path / "r" / "calls.jsonl").read_text()
    if setting != "blind":
        assert "<image " in log and "1920x1440" not in log and "640x480" in log
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert {dataset_of_scene(x["scene_id"]) for x in s["per_question"]} == {"arkitscenes", "scannetpp"}
    assert len(s["per_question"]) + s["excluded_unanswerable"] == sum(len(synth.groups[g]["questions"]) for g in GROUPS)
    assert all(0.0 <= x["score"] <= 1.0 for x in s["per_question"])


def test_resume_continues_an_interrupted_run(synth, tmp_path):
    first = run_llm(synth, "mock", "active_v2", 40, GROUPS[:1], tmp_path / "r", print_estimate=False)
    assert first["completed"] == 1
    st = run_llm(synth, "mock", "active_v2", 40, GROUPS, tmp_path / "r", print_estimate=False, resume=True)
    assert st["status"] == "complete" and st["completed"] == 2
    assert len(complete_episodes(tmp_path / "r" / "trajectories.jsonl")) == 2
