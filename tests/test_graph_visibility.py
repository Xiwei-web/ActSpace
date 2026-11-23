"""Occlusion and action-semantics tests; three-level visibility classification."""
import numpy as np
import pytest

from actspace_build import step4_graph_visibility as s4
from actspace_build.common import load_config

CFG = load_config()


def test_action_semantics_synthetic_graph():
    # nodes: position (x, y, z), heading yaw (degrees). Node 0 is at the origin facing +x.
    pos = np.array([[0.0, 0.0, 1.5],    # 0
                    [1.0, 0.0, 1.5],    # 1 straight ahead, 1 m
                    [0.0, 1.0, 1.5],    # 2 to the left, 1 m
                    [0.0, -0.8, 1.5],   # 3 to the right, 0.8 m
                    [0.1, 0.0, 1.5]])   # 4 near the origin, turned 50 deg left
    yaw = np.array([0.0, 0.0, 0.0, 10.0, 50.0])
    e = s4.action_edges(pos, yaw, CFG["view_graph"], CFG["view_graph"]["move_max_m"])
    assert e[0] == {"forward": 1, "backward": None, "left": 2, "right": 3, "turn_left": 4, "turn_right": None}
    # node 1: node 0 is 1 m straight behind (dyaw=0); node 4 is also behind but dyaw=50 > 45, so not backward
    assert e[1]["backward"] == 0
    assert e[1]["forward"] is None
    # node 4: node 0 is within 0.1 m and dyaw=-50 -> turn_right
    assert e[4]["turn_right"] == 0
    assert e[4]["turn_left"] is None


def test_scc_keeps_only_mutually_reachable():
    edges = [{"forward": 1}, {"backward": 0}, {"forward": 0}]  # 2 -> 0 one-way
    assert s4.largest_scc(edges) == [0, 1]


def test_occlusion_plate_in_front_of_cube():
    o3d = pytest.importorskip("open3d")
    cube = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 0.5).translate((-1.0, -1.0, 3.0))   # front face at z=3
    plate = o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 0.01).translate((-1.0, -1.0, 2.0))  # blocks the x<0 half
    scene = o3d.t.geometry.RaycastingScene()
    for m in (cube, plate):
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    K = np.array([[300.0, 0, 319.5], [0, 300.0, 239.5], [0, 0, 1]])
    occluded = np.array([[-0.5, 0.0, 3.0], [-0.3, 0.2, 3.0]])
    visible = np.array([[0.5, 0.0, 3.0], [0.3, -0.2, 3.0]])
    res = s4.visibility_for_node(scene, K, np.eye(4), 640, 480, [occluded, visible], CFG["visibility"])
    assert (res[0]["n_in"], res[0]["n_vis"]) == (2, 0)
    assert (res[1]["n_in"], res[1]["n_vis"]) == (2, 2)

    # sensor depth: door open during the mesh scan but closed during capture -> sensor depth is in front of the point, so it is occluded
    closed_door = np.full((480, 640), 1.5, np.float32)
    res = s4.visibility_for_node(scene, K, np.eye(4), 640, 480, [occluded, visible], CFG["visibility"], closed_door)
    assert res[1]["n_vis"] == 0 and res[1]["n_vis_mesh"] == 2
    # sensor depth agrees with the point (within tolerance) -> mesh decision unchanged; invalid depth (0) does not veto
    agree = np.full((480, 640), 2.95, np.float32)
    assert s4.visibility_for_node(scene, K, np.eye(4), 640, 480, [visible], CFG["visibility"], agree)[0]["n_vis"] == 2
    invalid = np.zeros((480, 640), np.float32)
    r = s4.visibility_for_node(scene, K, np.eye(4), 640, 480, [visible], CFG["visibility"], invalid)[0]
    assert r["n_vis"] == 2 and r["n_no_depth"] == 2


def test_three_state_classification():
    v = CFG["visibility"]
    assert s4.classify(0, 0, 0.0, v) == s4.STATE_INVISIBLE
    assert s4.classify(1000, 900, 0.05, v) == s4.STATE_VISIBLE
    assert s4.classify(1000, 900, 0.01, v) == s4.STATE_PARTIAL       # area too small (< 0.02)
    assert s4.classify(1000, 40, 0.01, v) == s4.STATE_INVISIBLE      # ρ < 0.05
    assert s4.classify(1000, 200, 0.01, v) == s4.STATE_PARTIAL       # 0.05 ≤ ρ < 0.30
    assert s4.classify(100, 40, 0.01, v) == s4.STATE_PARTIAL         # visible points < 50


def test_completeness_condition():
    """Only a corner shows: almost all in-frustum points are visible (high rho), but they are a small part of all object points -> partial."""
    v = CFG["visibility"]
    assert s4.classify(200, 190, 0.01, v, n_total=2000) == s4.STATE_PARTIAL   # completeness 0.095 < 0.30
    assert s4.classify(1000, 900, 0.05, v, n_total=2000) == s4.STATE_VISIBLE  # completeness 0.45


def test_obb_in_frame_fraction():
    """Box fully inside the image -> 1; half outside -> about 0.5; behind the camera -> 0."""
    K = np.array([[300.0, 0, 319.5], [0, 300.0, 239.5], [0, 0, 1]])
    size = np.array([0.4, 0.4, 0.4])
    assert abs(s4.obb_in_frame_frac([0, 0, 3.0], np.eye(3), size, K, np.eye(4), 640, 480) - 1.0) < 1e-6
    x_edge = 319.5 / 300.0 * 3.0       # x of the right image edge at z=3
    f = s4.obb_in_frame_frac([x_edge, 0, 3.0], np.eye(3), size, K, np.eye(4), 640, 480)
    assert 0.35 < f < 0.65
    assert s4.obb_in_frame_frac([0, 0, -3.0], np.eye(3), size, K, np.eye(4), 640, 480) == 0.0
    v = CFG["visibility"]
    assert s4.classify(1000, 900, 0.05, v, n_total=2000, in_frame=0.5) == s4.STATE_PARTIAL
