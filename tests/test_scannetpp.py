"""ScanNet++ decoding: sampled frame indices, depth record format, never overwriting existing files."""
import zlib

import lz4.block
import numpy as np
import pytest

from actspace_build.scannetpp import (count_depth_records, decode_frame_ids, iter_depth_records,
                                      move_no_overwrite)


def test_frame_ids_keep_original_index():
    # 60 FPS -> 10 FPS: keep 1 frame in 6, indices start at 0
    assert decode_frame_ids(20, 60, 10) == [0, 6, 12, 18]
    assert decode_frame_ids(7207, 60.0, 10)[-1] == 7206


def test_frame_ids_reject_non_integer_ratio():
    with pytest.raises(ValueError):
        decode_frame_ids(100, 30, 7)


def _write_per_frame(path, frames, codec):
    with open(path, "wb") as f:
        for d in frames:
            if codec == "lz4":
                payload = lz4.block.compress(d.astype(np.uint16).tobytes(), store_size=False)
            else:
                c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
                payload = c.compress((d / 1000.0).astype(np.float32).tobytes()) + c.flush()
            f.write(len(payload).to_bytes(4, "little") + payload)


@pytest.mark.parametrize("codec", ["lz4", "zlib"])
def test_depth_records_roundtrip(tmp_path, codec):
    h, w = 4, 6
    frames = [np.full((h, w), 1000 * (i + 1), dtype=np.uint16) for i in range(5)]
    p = tmp_path / "depth.bin"
    _write_per_frame(p, frames, codec)
    out = list(iter_depth_records(p, h, w))
    assert [i for i, _ in out] == list(range(5))
    for (_, d), ref in zip(out, frames):
        assert d.dtype == np.uint16 and np.array_equal(d, ref)
    assert count_depth_records(p, h, w) == 5


def test_move_never_overwrites(tmp_path):
    src, dst = tmp_path / "a.txt", tmp_path / "out" / "a.txt"
    dst.parent.mkdir()
    dst.write_text("original")
    src.write_text("new")
    assert move_no_overwrite(src, dst) is False
    assert dst.read_text() == "original"
    src2 = tmp_path / "b.txt"
    src2.write_text("b")
    assert move_no_overwrite(src2, tmp_path / "out" / "b.txt") is True


def test_snapshot_detects_changes(tmp_path):
    from actspace_build.scannetpp import changed_files, snapshot_files
    (tmp_path / "iphone").mkdir()
    (tmp_path / "iphone" / "rgb.mkv").write_bytes(b"x" * 10)
    snap = snapshot_files(tmp_path)
    (tmp_path / "iphone" / "rgb").mkdir()
    (tmp_path / "iphone" / "rgb" / "frame_000000.jpg").write_bytes(b"new")   # a newly added file does not count as a change
    assert changed_files(tmp_path, snap) == []
    (tmp_path / "iphone" / "rgb.mkv").write_bytes(b"y" * 11)
    assert changed_files(tmp_path, snap) == ["iphone/rgb.mkv"]


def test_stratified_pilots_distinct_types_and_deterministic():
    from actspace_build.scannetpp import draw_stratified_pilots
    types = {f"a{i}": "office" for i in range(10)} | {f"b{i}": "bath" for i in range(6)} | \
            {f"c{i}": "apt" for i in range(6)} | {f"d{i}": "kitchen" for i in range(4)} | \
            {f"e{i}": "lab" for i in range(4)} | {f"f{i}": "conf" for i in range(4)} | {"g0": "server"}
    r1 = draw_stratified_pilots(types, 5, np.random.default_rng([1, 2]))
    r2 = draw_stratified_pilots(types, 5, np.random.default_rng([1, 2]))
    assert r1 == r2
    picked = [p["scene_type"] for p in r1["picks"]]
    assert len(set(picked)) == 5
    assert picked[:3] == ["office", "bath", "apt"] or picked[:3] == ["office", "apt", "bath"]
    assert set(picked[3:]) <= {"kitchen", "lab", "conf"}          # 2 of the 4 tied scene types
    assert all(types[p["scene_id"]] == p["scene_type"] for p in r1["picks"])


def test_stratified_pilots_too_few_types():
    from actspace_build.scannetpp import draw_stratified_pilots
    with pytest.raises(ValueError):
        draw_stratified_pilots({"a": "x", "b": "y"}, 5, np.random.default_rng(0))
