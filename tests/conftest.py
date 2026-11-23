"""Shared test setup.

- `synth`: a synthetic frozen dataset (tests/synthetic.py) used by the end-to-end mock smoke tests; no real data needed.
- Tests that use the `data` fixture read the real frozen merged_v1.0 and its frames (configs/paths.yaml). They are skipped
  automatically when that data is not available locally.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

_HAVE_DATA = None


def have_real_data() -> bool:
    global _HAVE_DATA
    if _HAVE_DATA is None:
        try:
            from actspace_eval.data import FrozenData
            d = FrozenData(verify=False)
            g = d.groups[d.group_ids[0]]
            d.scene(g["scene_id"]).image(g["init_frame"])
            _HAVE_DATA = True
        except Exception:  # noqa: BLE001
            _HAVE_DATA = False
    return _HAVE_DATA


def pytest_collection_modifyitems(config, items):
    need = [it for it in items if "data" in getattr(it, "fixturenames", ()) or it.get_closest_marker("requires_data")]
    if need and not have_real_data():
        skip = pytest.mark.skip(reason="needs the frozen merged_v1.0 data and frames (set work/frames in configs/paths.yaml)")
        for it in need:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def synth_paths(tmp_path_factory):
    """Synthetic dataset, generated once per session; returns its paths.yaml."""
    from synthetic import make_synthetic
    return make_synthetic(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture
def synth(synth_paths, monkeypatch):
    """FrozenData of the synthetic dataset (verified like real data); ACTSPACE_PATHS points to it for this test only."""
    monkeypatch.setenv("ACTSPACE_PATHS", str(synth_paths))
    from actspace_eval.data import FrozenData
    return FrozenData(verify=True)
