"""ScanNet++ Step 3: uniqueness under the four category schemes (including merged_unique_fine_name adopted at GATE B); the size-reference
switch (permanently disabled, off by default); two-level raw-label mapping and GATE B vocabulary rules."""
import numpy as np

from actspace_build.scannetpp_step3 import assign_references
from actspace_build.scannetpp_vocab import map_label

OCFG = {"size_disambig_min_center_dist_m": 1.0, "size_disambig_min_volume_ratio": 2.5}
ALLOWED = {"chair", "office chair", "table", "cabinet", "kitchen cabinet", "book"}


def _o(fine, merged, kind="object", center=(0, 0, 0), size=(0.5, 0.5, 0.8)):
    return {"category_fine": fine, "category_merged": merged, "kind": kind, "center": np.array(center, float), "size": np.array(size, float)}


def _names(refs):
    return [r["ref_name"] for r in refs]


def test_schemes_chair_and_office_chair():
    """1 chair + 1 office chair: fine -> both unique; merged -> neither unique; hybrid -> only the specific name office chair is unique."""
    objs = [_o("chair", "chair", center=(0, 0, 0)), _o("office chair", "chair", center=(0.5, 0, 0))]
    assert _names(assign_references(objs, "fine", ALLOWED, OCFG)) == ["chair", "office chair"]
    assert _names(assign_references(objs, "merged", ALLOWED, OCFG)) == [None, None]
    assert _names(assign_references(objs, "hybrid", ALLOWED, OCFG)) == [None, "office chair"]


def test_hybrid_falls_back_to_parent_when_specific_not_allowed():
    """If the fine category is not allowed (e.g. coffee table), hybrid uses the parent name and counts over the whole parent class."""
    objs = [_o("coffee table", "table"), _o("chair", "chair", center=(3, 0, 0))]
    r = assign_references(objs, "hybrid", ALLOWED, OCFG)
    assert _names(r) == ["table", "chair"]
    assert _names(assign_references(objs, "fine", ALLOWED, OCFG)) == [None, "chair"]


def test_group_annotation_blocks_uniqueness():
    """1 book + 1 group annotation books: book is not unique; a group annotation is never referable and never takes part in size references."""
    objs = [_o("book", "book", size=(0.2, 0.15, 0.03)), _o("book", "book", kind="group", center=(3, 0, 0), size=(1.0, 0.3, 0.3))]
    for s in ("fine", "merged", "hybrid"):
        assert _names(assign_references(objs, s, ALLOWED, OCFG)) == [None, None]
    assert _names(assign_references([objs[1]], "fine", ALLOWED, OCFG)) == [None]


def test_size_reference_rules():
    """Exactly 2 with the same name, volume ratio >= 2.5, centres >= 1 m apart -> larger / smaller; not used if they are too close."""
    big, small = _o("table", "table", center=(0, 0, 0), size=(2.0, 1.0, 0.8)), _o("table", "table", center=(2, 0, 0), size=(0.6, 0.6, 0.7))
    r = assign_references([big, small], "fine", ALLOWED, OCFG, size_refs=True)
    assert _names(r) == ["larger table", "smaller table"] and r[0]["size_group"] == [0, 1]
    near = _o("table", "table", center=(0.5, 0, 0), size=(0.6, 0.6, 0.7))
    assert _names(assign_references([big, near], "fine", ALLOWED, OCFG, size_refs=True)) == [None, None]


def test_size_reference_disabled_by_default():
    """Size references are permanently disabled: by default (and with size_reference_enabled: false) neither of 2 same-class objects is referable."""
    big, small = _o("table", "table", center=(0, 0, 0), size=(2.0, 1.0, 0.8)), _o("table", "table", center=(2, 0, 0), size=(0.6, 0.6, 0.7))
    for s in ("fine", "merged", "hybrid"):
        r = assign_references([big, small], s, ALLOWED, OCFG)
        assert _names(r) == [None, None] and all(x["ref_kind"] is None for x in r)


def test_hybrid_generic_size_group_requires_same_name():
    """hybrid: if the parent count is 2 but one of them uses a specific name (chair + office chair), no size reference for the generic name."""
    objs = [_o("chair", "chair", center=(0, 0, 0), size=(1.0, 1.0, 1.0)), _o("office chair", "chair", center=(3, 0, 0), size=(0.4, 0.4, 0.4))]
    r = assign_references(objs, "hybrid", ALLOWED, OCFG, size_refs=True)
    assert r[0]["ref_kind"] is None and r[1]["ref_name"] == "office chair"


def test_map_label_levels():
    official = (["chair", "office chair", "socket", "tv", "table"], ["chair", "office chair", "socket", "tv", "table"],
                {"power socket": "socket", "desk": "table", "trash cans": "trash can"}, {"trash cans"})
    m = map_label("Power Socket", official)
    assert (m["kind"], m["category_fine"], m["category_merged"]) == ("object", "socket", "socket")
    m = map_label("office chair", official)
    assert (m["category_fine"], m["category_merged"]) == ("office chair", "chair")
    assert map_label("tv", official)["category_fine"] == "TV"
    assert map_label("monitor", official)["category_merged"] == "TV"
    assert map_label("SPLIT", official)["kind"] == "artifact"
    assert map_label("trash cans", official)["kind"] == "group"
    assert map_label("books", official)["kind"] == "group"


def test_merged_unique_fine_name():
    """GATE B scheme: uniqueness by merged class; fine names in name_ok are used as names, otherwise the merged name (name_ok is only monitor, copier)."""
    allowed = ALLOWED | {"TV", "monitor", "printer", "copier", "ottoman chair"}
    name_ok = allowed - {"ottoman chair"}
    s = "merged_unique_fine_name"
    # a unique monitor is called the monitor; monitor and TV share a family and are counted together
    assert _names(assign_references([_o("monitor", "TV")], s, allowed, OCFG, name_ok=name_ok)) == ["monitor"]
    assert _names(assign_references([_o("monitor", "TV"), _o("TV", "TV", center=(3, 0, 0))], s, allowed, OCFG, name_ok=name_ok)) == [None, None]
    # copier is unique via printer; chair + office chair are both not unique
    objs = [_o("copier", "printer"), _o("chair", "chair", center=(3, 0, 0)), _o("office chair", "chair", center=(5, 0, 0))]
    assert _names(assign_references(objs, s, allowed, OCFG, name_ok=name_ok)) == ["copier", None, None]
    # fall back to the merged name if the fine name is not in name_ok
    assert _names(assign_references([_o("ottoman chair", "chair")], s, allowed, OCFG, name_ok=name_ok)) == ["chair"]
    # a banned fine category (not in allowed) is not referable even if the merged class is allowed
    assert _names(assign_references([_o("spray bottle", "table")], s, allowed, OCFG, name_ok=name_ok)) == [None]
    # group annotations count towards the merged-class instances
    assert _names(assign_references([_o("monitor", "TV"), _o("TV", "TV", kind="group")], s, allowed, OCFG, name_ok=name_ok)) == [None, None]


def test_gateb_vocab_rules():
    """GATE B: ceiling lamp is not merged into lamp; subclasses of a banned parent are banned; subclasses of an allowed parent are allowed."""
    from actspace_build.scannetpp_vocab import allowed_categories
    official = (["lamp"], ["ceiling lamp", "table lamp"], {}, set())
    assert map_label("ceiling lamp", official)["category_merged"] == "ceiling lamp"
    assert map_label("table lamp", official)["category_merged"] == "lamp"
    base = {"printer": "x", "bottle": "x", "spray bottle": "x", "ceiling lamp": "x", "lamp": "x", "socket": "x", "door": "x"}
    a = allowed_categories(["printer", "copier", "bottle", "soap bottle", "spray bottle", "ceiling lamp", "lamp", "socket", "door"], base)
    assert set(a) == {"printer", "copier", "lamp", "door"} and a["copier"] == "inherit:printer"


def test_scannetpp_blacklist_separate_file(tmp_path, monkeypatch):
    """The ScanNet++ blacklist lives in a separate file (scannetpp.blacklist_file) and is read per scene; ARKit's load_blacklist only reads the shared file."""
    import actspace_build.common as common
    from actspace_build.scannetpp_step3 import load_blacklist
    from actspace_build.step3_objects import load_blacklist as arkit_load
    (tmp_path / "bl.yaml").write_text("objects:\n- video_id: '42897508'\n  obj_id: 5\n")
    (tmp_path / "bl_sp.yaml").write_text("objects:\n- scene_id: '31a2c91c43'\n  obj_id: 25\n  source_label: window\n")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    cfg = {"objects": {"blacklist_file": "bl.yaml", "blacklist_enabled": True}, "scannetpp": {"blacklist_file": "bl_sp.yaml"}}
    assert load_blacklist(cfg, "31a2c91c43") == {25: "window"} and load_blacklist(cfg, "5ee7c22ba0") == {}
    assert arkit_load(cfg) == ({("42897508", 5)}, set())


def test_default_merged_name():
    """Names default to the merged class; only monitor and copier use the fine name."""
    from actspace_build.scannetpp_vocab import NAME_FINE
    allowed = ALLOWED | {"shelf", "bookshelf", "storage cabinet", "monitor", "TV"}
    s = "merged_unique_fine_name"
    names = lambda o: _names(assign_references([o], s, allowed, OCFG, name_ok=set(NAME_FINE)))
    assert names(_o("bookshelf", "shelf")) == ["shelf"]
    assert names(_o("office chair", "chair")) == ["chair"]
    assert names(_o("storage cabinet", "cabinet")) == ["cabinet"]
    assert names(_o("monitor", "TV")) == ["monitor"]


def test_lookalike_children_stay_banned():
    """Look-alike banned categories are merged into the parent for uniqueness but are not allowed via the parent."""
    from actspace_build.scannetpp_vocab import allowed_categories, map_label
    official = (["box"], ["box"], {}, set())
    assert map_label("cardboard", official)["category_merged"] == "box"
    assert map_label("bowl", official)["category_merged"] == "pot"
    assert map_label("mattress", official)["category_merged"] == "mattress"  # part-of relations are not merged
    a = allowed_categories(["box", "cardboard", "pot", "bowl", "door", "sliding door"], {"box": "x", "pot": "x", "door": "x"})
    assert set(a) == {"box", "pot", "door"}
