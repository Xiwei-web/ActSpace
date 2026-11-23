"""ScanNet++ Step 4: mirror occluder faces (S8) and magenta patches (S10)."""
import numpy as np

from actspace_build.scannetpp_step4 import magenta_mask, mirror_face, mirror_object_ids, mirror_occluded

SP = {"magenta_r_min": 200, "magenta_b_min": 200, "magenta_g_max": 60, "magenta_dilate_px": 1,
      "mirror_label_exclude": ["mirror lamp"], "mirror_object_overrides": {"s1": [7]}}


def _mirror():
    # mirror on a wall: centre (0, 0, 1.5) in the x = 0 plane, 1 m wide (y), 1 m high (z), 0.1 m thick (x); the room is on the x > 0 side
    o = {"obj_id": 3, "obb": {"center": [0.0, 0.0, 1.5], "axes": np.eye(3).tolist(), "size": [0.1, 1.0, 1.0]}}
    return mirror_face(o, toward=np.array([2.0, 0.0, 1.5]))


def test_mirror_face_points_into_room():
    f = _mirror()
    assert np.allclose(f["normal"], [1, 0, 0]) and np.allclose(f["center"], [0.05, 0, 1.5])
    assert np.isclose(f["hu"], 0.5) and np.isclose(f["hv"], 0.5)


def test_mirror_occludes_points_behind_only():
    f = _mirror()
    C = np.array([2.0, 0.0, 1.5])
    P = np.array([[-1.0, 0.0, 1.5],    # behind the mirror, line of sight crosses the mirror -> occluded
                  [0.5, 0.0, 1.5],     # in front of the mirror -> not occluded
                  [-1.0, 3.0, 1.5],    # behind, but line of sight misses the mirror rectangle -> not occluded
                  [0.02, 0.0, 1.5]])   # beyond the mirror by less than the tolerance -> not occluded
    assert mirror_occluded(P, C, f, 0.05).tolist() == [True, False, False, False]
    # not applied when the camera is behind the mirror
    assert not mirror_occluded(P[:1], np.array([-2.0, 0.0, 1.5]), f, 0.05).any()


def test_mirror_object_ids():
    objs = [{"obj_id": 1, "source_label": "Mirror"}, {"obj_id": 2, "source_label": "mirror lamp"}, {"obj_id": 7, "source_label": "window"}]
    assert mirror_object_ids(SP, "s1", objs) == [1, 7]
    assert mirror_object_ids(SP, "s2", objs) == [1]


def test_magenta_mask_dilated():
    img = np.full((20, 20, 3), 128, np.uint8)
    img[10, 10] = (255, 0, 255)  # BGR pure magenta
    img[2, 2] = (250, 50, 120)   # not magenta enough
    m = magenta_mask(img, SP)
    assert m[10, 10] and m[9, 11] and not m[10, 13] and not m[2, 2]
    assert not magenta_mask(np.zeros((5, 5, 3), np.uint8), SP).any()


def test_region_metrics_short_side_and_sharpness():
    """S12: short side of the minimum-area bounding rectangle of an elongated projection; Laplacian variance inside the convex hull (0 for flat regions)."""
    from actspace_build.scannetpp_step4 import region_metrics
    uv = np.array([[10, 10], [210, 10], [210, 40], [10, 40], [100, 20]], float)
    lap = np.zeros((100, 300))
    mag = np.zeros((100, 300), bool)
    short, v = region_metrics(uv, lap, mag)
    assert abs(short - 30) < 1e-6 and v == 0.0
    lap[10:41, 10:211] = np.tile([1.0, -1.0], (31, 101))[:, :201]
    assert region_metrics(uv, lap, mag)[1] > 0.9
    mag[:] = True  # not counted when all pixels are magenta
    assert region_metrics(uv, lap, mag)[1] == 0.0
    assert region_metrics(uv[:2], lap, mag) == (0.0, 0.0)


def test_apply_s12_downgrades_only_visible():
    from actspace_build.scannetpp_step4 import apply_s12
    state = np.array([[2, 2, 1], [2, 0, 2]], np.uint8)
    short = np.array([[50, 300, 50], [300, 300, 300]], float)
    region = np.array([[10, 10, 10], [1, 10, 10]], float)
    frame = np.array([10.0, 10.0, 10.0])
    sp = {"vis_min_short_side_px": 105, "vis_min_sharpness_ratio": 0.18, "vis_sharpness_ref": "scene_frames"}
    out, ratio, d_shape, d_sharp = apply_s12(state, short, region, frame, sp)
    assert out.tolist() == [[1, 2, 1], [1, 0, 2]]
    assert d_shape.tolist() == [[True, False, False], [False, False, False]]
    assert d_sharp.tolist() == [[False, False, False], [True, False, False]]
    off = apply_s12(state, short, region, frame, {**sp, "vis_min_short_side_px": None, "vis_min_sharpness_ratio": None})[0]
    assert (off == state).all()


def test_apply_magenta_frac():
    from actspace_build.scannetpp_step4 import apply_magenta_frac
    state = np.array([[2, 2, 1]], np.uint8)
    n_mag = np.array([[40, 20, 90]])
    n_in = np.array([[100, 100, 100]])
    out, d = apply_magenta_frac(state, n_mag, n_in, 0.30)
    assert out.tolist() == [[1, 2, 1]] and d.tolist() == [[True, False, False]]
    assert (apply_magenta_frac(state, n_mag, n_in, None)[0] == state).all()


def test_motion_fast_nodes_and_rot_angle():
    from actspace_build.scannetpp_motion import rot_angle_deg
    from actspace_build.scannetpp_step4 import motion_fast_nodes
    ang, lin = np.array([10.0, 80.0, 20.0]), np.array([0.2, 0.3, 1.0])
    assert motion_fast_nodes(ang, lin, 73, None).tolist() == [False, True, False]
    assert motion_fast_nodes(ang, lin, 73, 0.9).tolist() == [False, True, True]
    assert not motion_fast_nodes(ang, lin, None, None).any()
    c, s_ = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert abs(rot_angle_deg(np.eye(3), np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])) - 30) < 1e-6


def test_pose_suspect_nodes():
    from actspace_build.scannetpp_step4 import pose_suspect_nodes
    err = np.array([0.02, 0.02, 0.02, 0.06, 0.06, np.nan], np.float32)  # median 0.02 -> threshold 0.05
    lin = np.array([1.0, 0.2, 0.3, 1.0, 0.5, 2.0], np.float32)
    assert pose_suspect_nodes(err, lin, 2.5, 0.9).tolist() == [False, False, False, True, False, False]
    assert not pose_suspect_nodes(err, lin, None, 0.9).any()


def test_topdown_structure_optional():
    """Top-down structure layer is optional; by default (structure=None) the output is pixel-identical to before."""
    from actspace_build import topdown
    obb = {"center": [0.0, 0.0, 0.5], "axes": np.eye(3).tolist(), "size": [1.0, 0.5, 1.0]}
    door = {"center": [2.0, 0.0, 1.0], "axes": np.eye(3).tolist(), "size": [0.1, 0.9, 2.0]}
    a = topdown.render({1: {"obb": obb}}, {1: "table"}, [1])
    b = topdown.render({1: {"obb": obb}}, {1: "table"}, [1], structure=None)
    assert (a == b).all()
    c = topdown.render({1: {"obb": obb}}, {1: "table"}, [1], structure={"wall_xy": np.array([[1.5, 0.0]]), "doors": [door], "windows": []})
    assert c.shape != a.shape or (c != a).any()
