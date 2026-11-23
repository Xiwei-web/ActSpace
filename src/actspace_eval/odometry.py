"""Odometry (docs/evaluation_protocol.md): poses of adjacent nodes → relative displacement (forward / leftward components in the starting heading) and turn angle, rounded and rendered as text."""
from __future__ import annotations

import numpy as np

from actspace_build.geometry import wrap_deg


def _round(x: float, step: float) -> float:
    # Round half away from zero; +1e-9 offsets float error such as 0.25 / 0.1 = 2.4999…, so 0.25 does not become 0.2
    r = float(np.floor(abs(x) / step + 0.5 + 1e-9) * step) * (1 if x >= 0 else -1)
    return round(r, 6) + 0.0  # drop -0.0


def relative_motion(pose_a, pose_b) -> tuple[float, float, float]:
    """(forward m, left m, turn deg, positive = left turn), unrounded. pose = (position xyz, yaw_deg)."""
    (pa, ya), (pb, yb) = pose_a, pose_b
    d = np.asarray(pb, float)[:2] - np.asarray(pa, float)[:2]
    r = np.radians(ya)
    fwd = float(d @ np.array([np.cos(r), np.sin(r)]))
    left = float(d @ np.array([-np.sin(r), np.cos(r)]))
    return fwd, left, float(wrap_deg(yb - ya))


def odometry(pose_a, pose_b, round_m=0.1, round_deg=5) -> dict:
    fwd, left, turn = relative_motion(pose_a, pose_b)
    return {"forward_m": _round(fwd, round_m), "left_m": _round(left, round_m), "turn_deg": int(_round(turn, round_deg))}


def describe(od: dict) -> str:
    """E.g. "Moved forward about 0.6 m and left about 0.1 m; turned left about 40 degrees."; components that round to 0 are omitted."""
    parts = []
    f, l, t = od["forward_m"], od["left_m"], od["turn_deg"]
    if f:
        parts.append(f"{'forward' if f > 0 else 'backward'} about {abs(f):.1f} m")
    if l:
        parts.append(f"{'left' if l > 0 else 'right'} about {abs(l):.1f} m")
    move = ("Moved " + " and ".join(parts)) if parts else ""
    turn = f"turned {'left' if t > 0 else 'right'} about {abs(t)} degrees" if t else ""
    if move and turn:
        return f"{move}; {turn}."
    if move:
        return move + "."
    if turn:
        return turn[0].upper() + turn[1:] + "."
    return "Moved; the change in position and heading is below the reporting resolution."
