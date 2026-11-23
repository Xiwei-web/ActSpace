"""Geometry unit tests + image uprighting tests."""
import numpy as np
import pytest

from actspace_build import geometry as g


# ---------------------------------------------------------------- direction judgement

P, F = (0.0, 0.0), (1.0, 0.0)


def test_straight_ahead_is_boundary():
    th = g.direction_theta_facing(P, F, (2.0, 0.0))
    assert th == pytest.approx(0.0)
    assert g.quadrant(th) is None
    assert g.angle_margin(th) == pytest.approx(0.0)


@pytest.mark.parametrize("C, theta, quad", [
    ((1.0, 1.0), 45.0, "front-left"),
    ((1.0, -1.0), -45.0, "front-right"),
    ((-1.0, 1.0), 135.0, "back-left"),
    ((-1.0, -1.0), -135.0, "back-right"),
])
def test_quadrants(C, theta, quad):
    th = g.direction_theta_facing(P, F, C)
    assert th == pytest.approx(theta)
    assert g.quadrant(th) == quad


A_, B_ = np.array([0.0, 0.0]), np.array([2.0, 0.0])


@pytest.mark.parametrize("C, q_ab, q_ba", [
    ((3.0, 5.0), "front-left", "back-right"),    # projection beyond B, left of line AB
    ((-1.0, -5.0), "back-right", "front-left"),  # projection beyond A, right side
    ((4.0, -2.0), "front-right", "back-left"),
    ((-2.0, 3.0), "back-left", "front-right"),
])
def test_inverse_consistency(C, q_ab, q_ba):
    """When the projection of C onto line AB falls outside segment AB, the quadrants satisfy the front-left <-> back-right mapping."""
    assert g.projection_outside_segment(A_, B_, C)
    a = g.quadrant(g.direction_theta_facing(A_, B_, C))
    b = g.quadrant(g.direction_theta_facing(B_, A_, C))
    assert (a, b) == (q_ab, q_ba)
    assert g.INVERSE_QUADRANT[a] == b


def test_inverse_only_flips_side_when_projection_between():
    """When the projection falls inside segment AB, both ends see C in front and the inverse mapping does not hold."""
    C = (1.0, 5.0)
    assert not g.projection_outside_segment(A_, B_, C)
    assert g.quadrant(g.direction_theta_facing(A_, B_, C)) == "front-left"
    assert g.quadrant(g.direction_theta_facing(B_, A_, C)) == "front-right"


def test_theta_from_yaw_matches_facing():
    rng = np.random.default_rng(0)
    for _ in range(100):
        Pp, Fp, C = rng.normal(size=(3, 2))
        yaw = np.degrees(np.arctan2(*(Fp - Pp)[::-1]))
        assert g.theta_from_yaw(Pp, yaw, C) == pytest.approx(g.direction_theta_facing(Pp, Fp, C))


def test_wrap_deg_range():
    assert g.wrap_deg(180.0) == 180.0
    assert g.wrap_deg(-180.0) == 180.0
    assert g.wrap_deg(190.0) == pytest.approx(-170.0)
    assert g.wrap_deg(-370.0) == pytest.approx(-10.0)


# ---------------------------------------------------------------- coordinate transforms

def test_y_up_to_z_up():
    assert np.allclose(g.Y_UP_TO_Z_UP @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(g.Y_UP_TO_Z_UP @ g.Y_UP_TO_Z_UP.T, np.eye(3))
    assert np.linalg.det(g.Y_UP_TO_Z_UP) == pytest.approx(1.0)


def test_yaw_from_Twc():
    # camera forward (z axis) points to world +y -> yaw = 90 deg
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    assert g.yaw_deg_from_Twc(g.make_T(R, [0, 0, 1])) == pytest.approx(90.0)


def test_interpolate_pose_endpoints_and_mid():
    T0 = g.make_T(g.axis_angle_to_matrix([0, 0, 0]), [0, 0, 0])
    T1 = g.make_T(g.axis_angle_to_matrix([0, 0, np.pi / 2]), [2, 0, 0])
    assert np.allclose(g.interpolate_pose(T0, T1, 0.0), T0)
    assert np.allclose(g.interpolate_pose(T0, T1, 1.0), T1)
    Tm = g.interpolate_pose(T0, T1, 0.5)
    assert np.allclose(Tm[:3, 3], [1, 0, 0])
    assert np.allclose(Tm[:3, :3], g.axis_angle_to_matrix([0, 0, np.pi / 4]))


def test_pose_at_timestamp_rules():
    ts = np.array([0.0, 0.1, 0.3])
    poses = np.stack([g.make_T(np.eye(3), [t, 0, 0]) for t in ts])
    near, gap = 0.02, 0.10
    assert np.allclose(g.pose_at_timestamp(0.11, ts, poses, near, gap)[:3, 3], [0.1, 0, 0])   # nearest
    assert np.allclose(g.pose_at_timestamp(0.05, ts, poses, near, gap)[:3, 3], [0.05, 0, 0])  # interpolated
    assert g.pose_at_timestamp(0.2, ts, poses, near, gap) is None                              # gap 0.2 > 0.1
    assert g.pose_at_timestamp(0.5, ts, poses, near, gap) is None                              # out of range


def test_pose_indices_at_timestamp():
    """POSE_FRAME_INVALID relies on the pose indices returned here: nearest neighbour uses 1, interpolation uses the 2 around it."""
    ts = np.array([0.0, 0.1, 0.3])
    near, gap = 0.02, 0.10
    assert g.pose_indices_at_timestamp(0.11, ts, near, gap) == ([1], None)
    idx, alpha = g.pose_indices_at_timestamp(0.05, ts, near, gap)
    assert idx == [0, 1] and np.isclose(alpha, 0.5)
    assert g.pose_indices_at_timestamp(0.2, ts, near, gap) is None
    assert g.pose_indices_at_timestamp(-0.5, ts, near, gap) is None


# ---------------------------------------------------------------- image uprighting

def _random_camera(rng):
    R = g.axis_angle_to_matrix(rng.normal(size=3))
    return g.make_T(R, rng.normal(size=3))


@pytest.mark.parametrize("quarters", [0, 1, 2, 3])
@pytest.mark.parametrize("W, H", [(640, 480), (256, 192), (7, 5)])
def test_rotation_projection_matches_image_rotation(quarters, W, H):
    """Same 3D point: a dot drawn at an integer pixel of the original image must, after rotating the image,
    sit where the point reprojects with the updated K, T_wc."""
    rng = np.random.default_rng(quarters * 100 + W)
    K = np.array([[0.8 * W, 0, (W - 1) / 2 + 0.3], [0, 0.75 * W, (H - 1) / 2 - 0.2], [0, 0, 1]])
    T = _random_camera(rng)
    K2, T2, W2, H2 = g.rotate_camera_cw(K, T, W, H, quarters)
    assert (W2, H2) == ((W, H) if quarters % 2 == 0 else (H, W))
    for _ in range(20):
        u, v = rng.integers(0, W), rng.integers(0, H)
        z = rng.uniform(0.5, 5.0)
        pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        pw = g.transform_points(T, pc[None])
        img = np.zeros((H, W), dtype=np.uint8)
        img[v, u] = 255
        rot = g.rotate_image_cw(img, quarters)
        assert rot.shape == (H2, W2)
        v_exp, u_exp = np.argwhere(rot == 255)[0]
        uv2, z2 = g.project(pw, K2, T2)
        assert uv2[0] == pytest.approx([u_exp, v_exp], abs=1e-6)
        assert z2[0] == pytest.approx(z)


@pytest.mark.parametrize("sky", ["Up", "Left", "Down", "Right"])
def test_sky_direction_and_upright_after_rotation(sky):
    # build a level camera, then roll it about the optical axis so world "up" appears in the sky direction of the image
    level = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # columns: world directions of camera x, y, z
    # after rolling the camera +90 deg about its z axis, the new x axis points down in the world and up is on the image's left
    roll = {"Up": 0, "Left": 90, "Down": 180, "Right": 270}[sky]
    R_roll = g.axis_angle_to_matrix([0, 0, np.radians(roll)])
    T = g.make_T(level @ R_roll, [0, 0, 1.5])
    assert g.sky_direction_from_pose(T) == sky
    K = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
    _, T2, _, _ = g.rotate_camera_cw(K, T, 640, 480, g.SKY_TO_CW_QUARTERS[sky])
    assert g.sky_direction_from_pose(T2) == "Up"
    # after uprighting, the camera y axis (image down) should point down in the world
    assert np.allclose(T2[:3, 1], [0, 0, -1], atol=1e-9)


# ---------------------------------------------------------------- projection / occlusion

def test_project_and_in_image():
    K = np.array([[100.0, 0, 49.5], [0, 100.0, 39.5], [0, 0, 1]])
    T = np.eye(4)
    uv, z = g.project(np.array([[0, 0, 2.0], [0, 0, -1.0]]), K, T)
    assert uv[0] == pytest.approx([49.5, 39.5])
    assert list(g.in_image(uv, z, 100, 80)) == [True, False]


def test_render_depth_plane():
    o3d = pytest.importorskip("open3d")
    mesh = o3d.geometry.TriangleMesh.create_box(10, 10, 0.01).translate((-5, -5, 3.0))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    K = np.array([[50.0, 0, 15.5], [0, 50.0, 11.5], [0, 0, 1]])
    depth = g.render_depth(scene, K, np.eye(4), 32, 24)
    assert np.allclose(depth, 3.0, atol=1e-4)  # z depth, not ray length
