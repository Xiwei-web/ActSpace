#!/usr/bin/env python3
"""Check frames rebuilt by scripts/rebuild_frames.py against a reference list of pixel hashes.

    python scripts/verify_frames.py --frames <dir> --sums FRAMES_SHA256.txt            # check (exit code 1 on any mismatch)
    python scripts/verify_frames.py --frames <dir> --sums FRAMES_SHA256.txt --write    # write the list for <dir>

The hash is SHA-256 over the decoded pixels (uint8, BGR, height x width x 3) plus the shape, not over the file bytes: PNG and JPEG
encoders of different library versions write different bytes for the same image. ScanNet++ frames decoded with another ffmpeg
build can differ slightly; mismatches are reported with the mean absolute pixel difference when --reference-dir is given.
List format, one line per frame: "<sha256>  <scene_id>/<frame_id>".
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = (".png", ".jpg")


def pixel_sha256(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    h = hashlib.sha256(np.asarray(img.shape, np.int64).tobytes())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def frame_files(root: Path) -> dict[str, Path]:
    return {f"{p.parent.name}/{p.stem}": p for p in sorted(root.glob("*/*")) if p.suffix.lower() in EXTS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", required=True, help="output directory of rebuild_frames.py (<scene_id>/<frame_id>.png|.jpg)")
    ap.add_argument("--sums", required=True, help="reference list")
    ap.add_argument("--write", action="store_true", help="write the list instead of checking")
    ap.add_argument("--reference-dir", help="optional directory with reference frames, to report pixel differences")
    a = ap.parse_args(argv)
    files = frame_files(Path(a.frames))
    if a.write:
        Path(a.sums).write_text("".join(f"{pixel_sha256(p)}  {k}\n" for k, p in files.items()))
        print(f"wrote {len(files)} hashes to {a.sums}")
        return 0
    ref = dict(reversed(line.split(None, 1)) for line in Path(a.sums).read_text().splitlines() if line.strip())
    ref = {k.strip(): v for k, v in ref.items()}
    missing = sorted(set(ref) - set(files))
    bad = [k for k in sorted(set(ref) & set(files)) if pixel_sha256(files[k]) != ref[k]]
    for k in bad[:20]:
        msg = f"MISMATCH {k}"
        if a.reference_dir:
            r = frame_files(Path(a.reference_dir)).get(k)
            if r is not None:
                x, y = cv2.imread(str(files[k])), cv2.imread(str(r))
                if x is not None and y is not None and x.shape == y.shape:
                    msg += f" (mean abs pixel difference {np.abs(x.astype(int) - y.astype(int)).mean():.3f})"
        print(msg)
    for k in missing[:20]:
        print(f"MISSING {k}")
    extra = len(set(files) - set(ref))
    print(f"checked {len(ref) - len(missing)} of {len(ref)} frames: {len(bad)} mismatches, {len(missing)} missing, {extra} not in the list")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
