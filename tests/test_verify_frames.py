"""scripts/verify_frames.py on synthetic images: write the list, check it, detect a changed and a missing frame."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("verify_frames", Path(__file__).resolve().parents[1] / "scripts" / "verify_frames.py")
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)


def test_verify_frames_roundtrip(tmp_path):
    frames = tmp_path / "frames" / "arkit_1"
    frames.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(frames / f"f_{i}.png"), np.full((48, 64, 3), 40 * i, np.uint8))
    cv2.imwrite(str(frames / "f_9.jpg"), np.full((48, 64, 3), 200, np.uint8))
    sums = tmp_path / "sums.txt"
    assert vf.main(["--frames", str(tmp_path / "frames"), "--sums", str(sums), "--write"]) == 0
    assert len(sums.read_text().splitlines()) == 4
    assert vf.main(["--frames", str(tmp_path / "frames"), "--sums", str(sums)]) == 0
    # same pixels, different PNG compression: still equal
    cv2.imwrite(str(frames / "f_0.png"), np.full((48, 64, 3), 0, np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    assert vf.main(["--frames", str(tmp_path / "frames"), "--sums", str(sums)]) == 0
    cv2.imwrite(str(frames / "f_1.png"), np.full((48, 64, 3), 41, np.uint8))
    assert vf.main(["--frames", str(tmp_path / "frames"), "--sums", str(sums)]) == 1
    cv2.imwrite(str(frames / "f_1.png"), np.full((48, 64, 3), 40, np.uint8))
    (frames / "f_2.png").unlink()
    assert vf.main(["--frames", str(tmp_path / "frames"), "--sums", str(sums)]) == 1
