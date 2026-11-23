"""ScanNet++: object OBB (minimum-area xy bounding rectangle + z extent), intrinsics scaling, drawn corners in the same order as the computed OBB."""
import numpy as np

from actspace_build import geometry as g
from actspace_build.scannetpp import object_obb, object_obb_pca, scale_intrinsics
from actspace_build.step1_arkit import obb_corners


def _box_points(center, yaw_deg, size, n=11):
    """Regular grid points inside a known gravity-aligned rotated box (including the 8 corners; the grid is symmetric, so the
    principal axes of the xy covariance are exactly the box axes). Returns the points, the true axes (rows) and the true corners."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    axes = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    t = np.linspace(-0.5, 0.5, n)
    local = np.stack(np.meshgrid(t, t, t, indexing="ij"), axis=-1).reshape(-1, 3) * size
    return np.asarray(center) + local @ axes, axes, np.asarray(center) + (0.5 * signs * size) @ axes


def test_obb_recovers_rotated_box():
    size = np.array([2.0, 0.6, 0.9])       # longest along x → PCA principal direction
    pts, axes_true, corners_true = _box_points([1.0, -2.0, 0.45], 30.0, size)
    o = object_obb(pts)
    np.testing.assert_allclose(o["center"], [1.0, -2.0, 0.45], atol=1e-6)
    np.testing.assert_allclose(o["size"], size, atol=1e-6)
    np.testing.assert_allclose(np.abs(o["axes"] @ axes_true.T), np.eye(3), atol=1e-6)   # same set of axes (sign flips allowed)
    assert np.isclose(np.linalg.det(o["axes"]), 1.0)                                   # right-handed
    assert o["axes"][0, 0] >= 0                                                         # fixed sign


def test_draw_corners_match_computed_obb():
    """Drawing uses step1_arkit.obb_corners(o): the corner set must match the true box, and size[i] must correspond to axes[i] (same order)."""
    size = np.array([1.5, 0.4, 2.0])
    pts, _, corners_true = _box_points([0.0, 0.0, 1.0], -65.0, size)
    o = object_obb(pts)
    C = obb_corners(o)
    d = np.linalg.norm(C[:, None, :] - corners_true[None, :, :], axis=-1)
    assert np.all(d.min(axis=1) < 1e-6) and np.all(d.min(axis=0) < 1e-6)
    # edge length along each axis equals the matching size: span of the projection onto axes[i]
    for i in range(3):
        proj = C @ o["axes"][i]
        assert np.isclose(proj.max() - proj.min(), o["size"][i])
    # all input points lie inside the box
    local = (pts - o["center"]) @ o["axes"].T
    assert np.all(np.abs(local) <= 0.5 * o["size"] + 1e-9)


def test_scale_intrinsics_matches_pixel_scaling():
    K = np.array([[1429.5, 0, 954.2], [0, 1429.5, 724.2], [0, 0, 1.0]])
    Kd = scale_intrinsics(K, (1920, 1440), (256, 192))
    T = np.eye(4)
    P = np.array([[0.3, -0.2, 2.0], [-1.0, 0.5, 3.5]])
    uv, _ = g.project(P, K, T)
    uvd, _ = g.project(P, Kd, T)
    s = 256 / 1920
    np.testing.assert_allclose(uvd, (uv + 0.5) * s - 0.5, atol=1e-9)
    assert np.isclose(Kd[0, 0], 1429.5 * s) and np.isclose(Kd[1, 1], 1429.5 * 192 / 1440)


def _assert_box_matches(o, center, size_true, axes_true):
    """o matches the true box: centre, same set of axes (sign flips allowed), sizes in axis order, right-handed, fixed sign."""
    np.testing.assert_allclose(o["center"], center, atol=1e-6)
    np.testing.assert_allclose(o["size"], size_true, atol=1e-6)
    np.testing.assert_allclose(np.abs(o["axes"] @ axes_true.T), np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(o["axes"]), 1.0) and o["axes"][0, 0] >= 0
    C = obb_corners(o)                                  # drawn corners: span along axes[i] equals size[i]
    for i in range(3):
        proj = C @ o["axes"][i]
        assert np.isclose(proj.max() - proj.min(), o["size"][i])


def test_min_rect_square_footprint():
    """Square base: the PCA direction is arbitrary (isotropic covariance); the minimum bounding rectangle must fit the true edges."""
    size = np.array([0.6, 0.6, 0.9])
    pts, axes_true, _ = _box_points([2.0, 1.0, 0.45], 30.0, size)
    o = object_obb(pts)
    _assert_box_matches(o, [2.0, 1.0, 0.45], size, axes_true)
    assert np.isclose(o["size"][0] * o["size"][1], 0.36)


def test_min_rect_elongated():
    size = np.array([2.0, 0.4, 0.75])
    pts, axes_true, _ = _box_points([-1.0, 0.5, 0.375], -20.0, size)
    o = object_obb(pts)
    _assert_box_matches(o, [-1.0, 0.5, 0.375], size, axes_true)
    assert o["size"][0] > o["size"][1]                  # axes[0] is the longer edge


def test_min_rect_uneven_points():
    """Uneven point distribution: 8 corners + many dense points along the base diagonal. PCA is pulled towards the diagonal and the box is too large; the minimum bounding rectangle is unaffected."""
    size = np.array([1.2, 0.8, 1.0])
    c, s = np.cos(np.radians(25.0)), np.sin(np.radians(25.0))
    axes_true = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([0.3, -0.7, 0.5])
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = center + (0.5 * signs * size) @ axes_true
    t = np.linspace(-0.5, 0.5, 400)[:, None]
    diag = center + (t * np.array([size[0], size[1], 0.0])) @ axes_true
    pts = np.vstack([corners, diag])
    o = object_obb(pts)
    _assert_box_matches(o, center, size, axes_true)
    o_pca = object_obb_pca(pts)
    assert o_pca["size"][0] * o_pca["size"][1] > 1.3 * size[0] * size[1]
    C_pca = obb_corners(o_pca)         # the PCA box also keeps size and axes in the same order
    for i in range(3):
        proj = C_pca @ o_pca["axes"][i]
        assert np.isclose(proj.max() - proj.min(), o_pca["size"][i])


def test_min_rect_degenerate_inputs():
    """Collinear or very few points fall back to the PCA direction without raising."""
    line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.1]])
    o = object_obb(line)
    assert np.isclose(abs(o["axes"][0] @ np.array([1, 1, 0]) / np.sqrt(2)), 1.0)
    o2 = object_obb(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
    assert np.all(np.isfinite(o2["size"]))
