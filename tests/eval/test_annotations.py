"""Equivalence of the released annotation package and the frozen merged_v1.0, field by field.

Needs both: the frozen version (`work` in configs/paths.yaml) and the annotation package (`annotations`). Skipped otherwise.
Documented differences (by design of the package): the group-level `human_verified` flag is not published, and for numeric (T1)
questions the package's `answer` is the unrounded distance used for scoring (`answer_raw`) instead of the display-rounded value.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from actspace_eval.answers import gold
from actspace_eval.data import FrozenData, dataset_of_scene, load_eval_config
from actspace_eval.llm_run import run_llm
from actspace_eval.trajectory import complete_episodes

VIS_KEYS = ("obj_ids", "frame_ids", "state", "rho", "completeness", "area_frac", "in_frame_frac", "n_infrustum", "n_visible")
FRAME_KEYS = ("source_frame", "rel_path", "timestamp", "rotation_cw_quarters", "T_wc", "K", "image_size", "position", "yaw_deg")
OBJECT_KEYS = ("obj_id", "source_label", "referable", "ref_kind", "ref_name", "center", "num_points")
META_KEYS = ("source_id", "split", "up_axis", "units", "pose_convention", "image_size", "frame_source", "rotation_cw_quarters")


@pytest.fixture(scope="module")
def pair():
    ec = load_eval_config()
    if not ec["paths"].get("annotations_root"):
        pytest.skip("no annotation package configured (`annotations` in configs/paths.yaml)")
    try:
        frozen = FrozenData(source="frozen", verify=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"frozen merged_v1.0 not available: {e}")
    return frozen, FrozenData(source="annotations", verify=True)


def read(p: Path):
    return json.loads(p.read_text())


def test_groups_and_questions_identical(pair):
    frozen, ann = pair
    assert ann.group_ids == frozen.group_ids and len(ann.group_ids) == 234
    n_q = n_numeric = 0
    for gid in frozen.group_ids:
        a, f = ann.groups[gid], frozen.groups[gid]
        assert set(f) - set(a) == {"human_verified"} and set(a) <= set(f), gid
        for k in a:
            if k != "questions":
                assert a[k] == f[k], (gid, k)
        assert [q["qid"] for q in a["questions"]] == [q["qid"] for q in f["questions"]]
        for qa, qf in zip(a["questions"], f["questions"]):
            assert set(qa) == set(qf), qf["qid"]
            for k in qf:
                if k == "answer" and qf["answer_type"] == "numeric":
                    assert qa["answer"] == qf["answer_raw"] and round(qa["answer"], 1) == pytest.approx(qf["answer"], abs=0.1 + 1e-9)
                    n_numeric += 1
                else:
                    assert qa[k] == qf[k], (qf["qid"], k)
            assert gold(qa) == gold(qf)
            n_q += 1
    assert n_q == 994 and n_numeric > 0


def test_scene_files_identical(pair):
    frozen, ann = pair
    scenes = sorted({g["scene_id"] for g in frozen.groups.values()})
    assert len(scenes) == 72
    for sid in scenes:
        fw, aw = frozen.scene(sid).wd, ann.scene(sid).wd
        g_f, g_a = read(fw / "view_graph.json"), read(aw / "view_graph.json")
        assert g_a == {"nodes": g_f["nodes"], "edges": g_f["edges"]}, sid
        zf, za = np.load(fw / "visibility.npz"), np.load(aw / "visibility.npz")
        for k in VIS_KEYS:
            assert np.array_equal(zf[k], za[k]) and zf[k].dtype == za[k].dtype, (sid, k)
        nodes = set(g_f["nodes"])
        frames_all = {f["frame_id"]: f for f in read(fw / "frames_all.json")}
        frames = read(aw / "frames.json")
        assert [f["frame_id"] for f in frames] == read(fw / "keyframes.json")["keyframe_ids"], sid
        assert nodes <= {f["frame_id"] for f in frames}
        for f in frames:
            src = frames_all[f["frame_id"]]
            assert {k: f[k] for k in FRAME_KEYS} == {k: src[k] for k in FRAME_KEYS}, (sid, f["frame_id"])
            assert f["in_graph"] == (f["frame_id"] in nodes)
        objs_f, objs_a = read(fw / "objects.json"), read(aw / "objects.json")
        assert [o["obj_id"] for o in objs_a] == [o["obj_id"] for o in objs_f], sid
        for oa, of in zip(objs_a, objs_f):
            assert {k: oa[k] for k in OBJECT_KEYS} == {k: of.get(k) for k in OBJECT_KEYS}, (sid, of["obj_id"])
            assert oa["category"] == (of.get("category_merged") or of["category"])
            assert oa["obb"] == {k: of["obb"][k] for k in ("center", "axes", "size", "yaw_deg")}
        mf, ma = read(fw / "scene_meta.json"), read(aw / "scene_meta.json")
        assert {k: ma[k] for k in META_KEYS} == {k: mf.get(k) for k in META_KEYS} and ma["dataset"] == dataset_of_scene(sid)


def test_scene_data_identical(pair):
    """What the environment reads: graph, poses, image paths and rotations, visibility."""
    frozen, ann = pair
    for sid in sorted({g["scene_id"] for g in frozen.groups.values()}):
        sf, sa = frozen.scene(sid), ann.scene(sid)
        assert sa.nodes == sf.nodes and sa.edges == sf.edges
        assert all(np.array_equal(sa.pose[n][0], sf.pose[n][0]) and sa.pose[n][1] == sf.pose[n][1] for n in sf.nodes)
        assert sa._img == sf._img                                    # raw image path and rotation per node
        obj_ids = [int(o) for o in np.load(sf.wd / "visibility.npz")["obj_ids"]]
        assert [[sa.visible(o, n) for n in sa.nodes] for o in obj_ids] == [[sf.visible(o, n) for n in sf.nodes] for o in obj_ids]


@pytest.mark.parametrize("setting,budget,seed", [("active_v2", 40, 0), ("oracle_route", 40, 0), ("random_walk", 40, 1), ("passive", 40, 0),
                                                 ("blind", 40, 0)])
def test_episodes_identical(pair, tmp_path, setting, budget, seed):
    """All 234 groups with the mock model (no images): identical trajectories and scores from both sources."""
    frozen, ann = pair
    out = {}
    for name, d in (("frozen", frozen), ("annotations", ann)):
        st = run_llm(d, "mock", setting, budget, d.group_ids, tmp_path / name, render=False, run_seed=seed, print_estimate=False, cost_cap=1e9)
        assert st["status"] == "complete" and st["completed"] == 234
        eps = complete_episodes(tmp_path / name / "trajectories.jsonl")
        steps = {e[0]["group_id"]: [(r.get("to"), r.get("photo_id"), r.get("feedback")) for r in e if r["type"] == "step"] for e in eps}
        summary = json.loads((tmp_path / name / "summary.json").read_text())
        out[name] = (steps, sorted((x["qid"], x["score"], json.dumps(x.get("answer"))) for x in summary["per_question"]))
    assert out["frozen"] == out["annotations"]


def test_tampered_package_is_refused(pair, tmp_path):
    """A changed or unlisted file in the package stops the evaluation at start-up."""
    import shutil
    from actspace_eval.data import FreezeError
    _, ann = pair
    pkg = tmp_path / "pkg"
    shutil.copytree(ann.pkg, pkg, ignore=shutil.ignore_patterns(".*"))
    ec = {**ann.ecfg, "paths": {**ann.ecfg["paths"], "annotations_root": str(pkg)}}
    FrozenData(ec, verify=True)
    vg = pkg / "scenes" / ann.groups[ann.group_ids[0]]["scene_id"] / "view_graph.json"
    vg.write_text(vg.read_text().replace("turn_left", "turn_lefT", 1))
    with pytest.raises(FreezeError):
        FrozenData(ec, verify=True)
