"""ScanNet++ Steps 5-6: dataset parameters of the shared step56_questions (ARKit behaviour unchanged) and the ScanNet++ evidence-linking mode override."""
from actspace_build.scannetpp_step56 import config_for_mode
from actspace_build.step56_questions import objects_dataset, scene_id_of


def test_scene_id_and_objects_dir():
    assert scene_id_of("arkitscenes", "42897508") == "arkit_42897508"
    assert objects_dataset("arkitscenes") == "arkitscenes"
    assert scene_id_of("scannetpp", "31a2c91c43") == "scannetpp_31a2c91c43"
    assert objects_dataset("scannetpp") == "scannetpp"


def test_config_for_mode_does_not_touch_shared():
    cfg = {"questions": {"evidence_link_mode": "partial_partial"}, "scannetpp": {"evidence_link_mode": None}}
    c2 = config_for_mode(cfg, "visible_partial")
    assert c2["questions"]["evidence_link_mode"] == "visible_partial" and cfg["questions"]["evidence_link_mode"] == "partial_partial"


def test_pilot_quadrant_tolerance():
    cfg = {"questions": {"evidence_link_mode": "partial_partial", "quadrant_balance_tol": 0.05},
           "scannetpp": {"evidence_link_mode": None, "quadrant_balance_tol": None, "t2_dedup_by_answer": True}}
    assert config_for_mode(cfg, "partial_partial")["questions"]["quadrant_balance_tol"] == 1.0
    assert config_for_mode(cfg, "partial_partial", "shared")["questions"]["quadrant_balance_tol"] == 0.05
    assert config_for_mode(cfg, "partial_partial")["questions"]["t2_dedup_by_answer"] is True
