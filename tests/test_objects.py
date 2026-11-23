"""Step 3 object OBB: stored edge lengths and box axes must be in the same order (otherwise length and width are swapped when drawing)."""
import numpy as np

from actspace_build.step1_arkit import obb_corners
from actspace_build.step3_objects import obb_yaw_and_size


def test_stored_obb_has_same_corners_as_raw():
    # a slanted thin box (TV): original order [thickness, width, height]; stored edges are reordered to [width, thickness, height]
    yaw = np.radians(30)
    axes = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    size = np.array([0.06, 1.23, 0.73])
    center = np.array([1.0, 2.0, 0.8])
    _, size_o, axes_o = obb_yaw_and_size(axes, size)
    raw = obb_corners({"center": center, "axes": axes, "size": size})
    stored = obb_corners({"center": center, "axes": axes_o, "size": np.asarray(size_o)})
    key = lambda C: sorted(map(tuple, np.round(C, 6)))
    assert key(raw) == key(stored)
    assert size_o[0] >= size_o[1]


def test_points_in_obb_excludes_floor():
    from actspace_build.step3_objects import points_in_obb
    axes = np.eye(3)
    center, size = np.array([0, 0, 0.5]), np.array([1.0, 1.0, 1.0])  # box bottom z = 0
    V = np.array([[0, 0, 0.0],    # floor
                  [0, 0, 0.02],   # carpet (< bottom + 0.03)
                  [0, 0, 0.5],    # inside the object
                  [0.53, 0, 0.5], # within the horizontal inflation
                  [0, 0, 1.04],   # within the upward inflation
                  [0, 0, -0.04]]) # below the box (no downward inflation)
    idx = points_in_obb(V, center, axes, size, 0.05, 0.03)
    assert sorted(idx.tolist()) == [2, 3, 4]


def _box(center, size):
    return {"center": np.asarray(center, float), "axes": np.eye(3), "size": np.asarray(size, float), "label": "x"}


OCFG = {"size_disambig_min_volume_ratio": 2.5, "size_disambig_max_instances": 2,
        "size_disambig_min_center_dist_m": 1.0, "armchair_enabled": False, "armchair_max_longest_edge_m": 1.2,
        "size_ref_enabled": True}


def test_size_references():
    """larger / smaller only with exactly 2 instances of a category and volume ratio >= 2.5; no more largest / smallest."""
    from actspace_build.step3_objects import assign_references
    raw = [_box([0, 0, 0], [2, 1, 1]), _box([3, 0, 0], [1, 0.5, 1])]          # volume ratio 4
    r = assign_references(raw, ["table", "table"], OCFG)
    assert [x["ref_name"] for x in r] == ["larger table", "smaller table"] and r[0]["size_group"] == [0, 1]
    raw_close = [_box([0, 0, 0], [2, 1, 1]), _box([3, 0, 0], [1.2, 1, 1])]    # volume ratio 1.67 < 2.5
    assert all(x["ref_kind"] is None for x in assign_references(raw_close, ["table", "table"], OCFG))
    raw3 = raw + [_box([0, 3, 0], [0.5, 0.5, 0.5])]                              # 3 instances -> disabled
    assert all(x["ref_kind"] is None for x in assign_references(raw3, ["table"] * 3, OCFG))
    sinks = [_box([0, 0, 0], [0.9, 0.6, 0.2]), _box([0.45, 0, 0], [0.3, 0.3, 0.2])]  # double sink: centres < 1.0 m apart
    assert all(x["ref_kind"] is None for x in assign_references(sinks, ["sink", "sink"], OCFG))


def test_size_references_disabled():
    """Size references are permanently disabled: objects of a category with >= 2 instances are not referable (disabled by default)."""
    from actspace_build.step3_objects import assign_references
    raw = [_box([0, 0, 0], [2, 1, 1]), _box([3, 0, 0], [1, 0.5, 1]), _box([0, 3, 0], [1, 1, 1])]
    cats = ["table", "table", "sink"]
    for ocfg in (dict(OCFG, size_ref_enabled=False), {k: v for k, v in OCFG.items() if k != "size_ref_enabled"}):
        r = assign_references(raw, cats, ocfg)
        assert [x["ref_kind"] for x in r] == [None, None, "unique"] and [x["ref_name"] for x in r] == cats


def test_armchair_disabled():
    from actspace_build.step3_objects import assign_references
    raw = [_box([0, 0, 0], [2.0, 0.9, 0.8]), _box([3, 0, 0], [0.9, 0.8, 0.8])]   # volume ratio 2.84
    r = assign_references(raw, ["sofa", "sofa"], OCFG)
    assert [x["ref_name"] for x in r] == ["larger sofa", "smaller sofa"]       # no longer renamed to armchair
    r_old = assign_references(raw, ["sofa", "sofa"], dict(OCFG, armchair_enabled=True))
    assert [x["ref_name"] for x in r_old] == ["sofa", "armchair"]


def test_shared_vertices_removed():
    """A vertex in the overlap of two boxes (sofa armrest reaching into a small table's box) belongs to both: it is removed from both point sets."""
    from actspace_build.step3_objects import object_point_sets
    ocfg = {"arkit_obb_inflate_m": 0.05, "floor_cut_m": 0.03, "nested_min_volume_frac": 0.8}
    sofa = {"center": np.array([0, 0, 0.4]), "axes": np.eye(3), "size": np.array([2.0, 1.0, 0.8])}
    table = {"center": np.array([1.2, 0, 0.3]), "axes": np.eye(3), "size": np.array([0.5, 0.5, 0.6])}
    V = np.array([[0.98, 0, 0.5],   # sofa armrest, in the overlap of both boxes (x in [0.95, 1.0])
                  [1.3, 0, 0.4],    # the small table itself
                  [0.0, 0, 0.5]])   # middle of the sofa
    (s_sofa, s_table), removed, _ = object_point_sets(V, [sofa, table], ocfg)
    assert s_sofa.tolist() == [2] and s_table.tolist() == [1] and removed == [1, 1]


def test_nested_object_keeps_shared_vertices():
    """A sink box entirely inside a counter cabinet box (nested) -> overlapping vertices go to the sink and are removed only from the cabinet."""
    from actspace_build.step3_objects import object_point_sets
    ocfg = {"arkit_obb_inflate_m": 0.05, "floor_cut_m": 0.03, "nested_min_volume_frac": 0.8}
    cabinet = {"center": np.array([0, 0, 0.45]), "axes": np.eye(3), "size": np.array([2.0, 0.6, 0.9])}
    sink = {"center": np.array([0, 0, 0.8]), "axes": np.eye(3), "size": np.array([0.6, 0.4, 0.15])}
    V = np.array([[0.0, 0, 0.8],    # sink (also inside the cabinet box)
                  [0.8, 0, 0.5]])   # the cabinet itself
    (s_cab, s_sink), removed, stats = object_point_sets(V, [cabinet, sink], ocfg)
    assert s_sink.tolist() == [0] and s_cab.tolist() == [1] and stats["nested_pairs"] == 1
