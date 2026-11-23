"""Top-down view: object centres and names, OBB footprints; direction questions add the standpoint, facing arrow and the four quadrants; T4 adds the start position and heading.

Coordinates: world z points up; seen from above, x points right and y points up (y is flipped in the image), so counter-clockwise = left, consistent with geometry.direction_theta.
Used for the C3 direction-question check and the GATE D review-phase pages.
"""
from __future__ import annotations

import cv2
import numpy as np

QUAD_ANGLES = {"front-left": 45, "back-left": 135, "back-right": -135, "front-right": -45}
INK, MUTED, FAINT = (40, 40, 40), (150, 150, 150), (215, 215, 215)
HL, STAND, TARGET, START = (40, 110, 230), (200, 90, 30), (30, 30, 200), (40, 150, 40)  # BGR
WALL, DOOR, WINDOW = (110, 110, 110), (40, 90, 150), (200, 150, 60)  # optional structure layer (wall points, doors, windows)


def footprint(obb):
    """The four corners of an OBB in the xy plane (using the two horizontal axes)."""
    axes, size, c = np.asarray(obb["axes"]), np.asarray(obb["size"]), np.asarray(obb["center"])
    k_up = int(np.argmax(np.abs(axes[:, 2])))
    h = [i for i in range(3) if i != k_up]
    corners = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        corners.append(c[:2] + 0.5 * (sx * size[h[0]] * axes[h[0], :2] + sy * size[h[1]] * axes[h[1], :2]))
    return np.array(corners)


def render(objs, names, highlight, background_xy=None, stand=None, face=None, target=None, start=None,
           title="", size_px=760, margin=40, structure=None):
    """objs: {obj_id: {"obb": ...}}; names: {obj_id: name}; highlight: objects to label with their name;
    stand / face / target: standing, facing and queried objects of a direction question (obj_id); start: (xy, yaw_deg) (T4 start).
    structure (optional, not drawn by default): {"wall_xy": Nx2 wall points, "doors": [OBB], "windows": [OBB]}, drawn below the objects to help match photos with the top-down view."""
    pts = [footprint(o["obb"]) for o in objs.values()]
    if structure:
        pts += [footprint(b) for b in structure.get("doors", []) + structure.get("windows", [])]
    if start is not None:
        pts.append(np.asarray(start[0])[None, :2])
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - 0.5, allp.max(0) + 0.5
    s = (size_px - 2 * margin) / max(hi - lo)
    H = int((hi[1] - lo[1]) * s) + 2 * margin + 30
    W = int((hi[0] - lo[0]) * s) + 2 * margin
    img = np.full((H, W, 3), 255, np.uint8)

    def px(p):
        return (int(margin + (p[0] - lo[0]) * s), int(H - margin - (p[1] - lo[1]) * s))
    if background_xy is not None:
        for p in background_xy:
            cv2.circle(img, px(p), 1, FAINT, -1)
    if structure:
        for p in structure.get("wall_xy", []):
            if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]:
                cv2.circle(img, px(p), 1, WALL, -1)
        for key, col, label in (("doors", DOOR, "door"), ("windows", WINDOW, "window")):
            for b in structure.get(key, []):
                poly = np.array([px(p) for p in footprint(b)], np.int32)
                cv2.fillPoly(img, [poly], tuple(int(0.35 * c_ + 0.65 * 255) for c_ in col))
                cv2.polylines(img, [poly], True, col, 2, cv2.LINE_AA)
                q = poly.mean(0).astype(int)
                cv2.putText(img, label, (int(q[0]) + 4, int(q[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for oid, o in objs.items():
        poly = np.array([px(p) for p in footprint(o["obb"])], np.int32)
        cv2.polylines(img, [poly], True, HL if oid in highlight else MUTED, 2 if oid in highlight else 1, cv2.LINE_AA)
    c = {oid: np.asarray(o["obb"]["center"])[:2] for oid, o in objs.items()}

    def axes_from(p, fwd, color, label):
        fwd = fwd / np.linalg.norm(fwd)
        left = np.array([-fwd[1], fwd[0]])
        L = 2.0 * max(hi - lo)
        for d in (fwd, left):
            cv2.line(img, px(p - L * d), px(p + L * d), color, 1, cv2.LINE_AA)
        cv2.arrowedLine(img, px(p), px(p + 0.9 * fwd), color, 3, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, px(p), 7, color, -1, cv2.LINE_AA)
        r = 0.35 * min(hi - lo)
        for q, a in QUAD_ANGLES.items():
            ang = np.radians(a)
            d = np.cos(ang) * fwd + np.sin(ang) * left
            t = px(p + r * d)
            cv2.putText(img, q, (t[0] - 38, t[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, label, (px(p)[0] + 9, px(p)[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if stand is not None and face is not None:
        axes_from(c[stand], c[face] - c[stand], STAND, "you stand here")
    if start is not None:
        y = np.radians(start[1])
        axes_from(np.asarray(start[0])[:2], np.array([np.cos(y), np.sin(y)]), START, "start (view direction)")
    for oid in highlight:
        p = px(c[oid])
        col = TARGET if oid == target else HL
        cv2.circle(img, p, 6 if oid == target else 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, names[oid], (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, names[oid], (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
    # 1 m scale bar
    cv2.line(img, (W - margin - int(s), H - 15), (W - margin, H - 15), INK, 2)
    cv2.putText(img, "1 m", (W - margin - int(s), H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    return img


def mesh_background(vertices, n=25000, seed=0):
    """Sparse projection of mesh vertices onto the xy plane, used as the room-outline background."""
    V = np.asarray(vertices)
    idx = np.random.default_rng(seed).choice(len(V), min(n, len(V)), replace=False)
    return V[idx, :2]
