"""Geometry utilities: the single dataset-wide implementation of direction judgements, coordinate transforms, pose interpolation, image uprighting and projection.

Conventions: right-handed world frame, z up, metres; OpenCV camera frame (x right, y down, z forward);
all poses are T_wc (camera to world); yaw = atan2(f_y, f_x), where f is the camera forward vector projected onto the xy plane.
Pixel coordinates put pixel centres at integers (consistent with how cv2.rotate maps integer pixel indices).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# y up -> z up: rotate +90 deg about the x axis, (0,1,0) -> (0,0,1)
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])

QUADRANTS = ("front-left", "back-left", "front-right", "back-right")
INVERSE_QUADRANT = {"front-left": "back-right", "back-right": "front-left",
                    "front-right": "back-left", "back-left": "front-right"}
_BOUNDARIES_DEG = (0.0, 90.0, -90.0, 180.0, -180.0)


def wrap_deg(a):
    """Normalise an angle to (-180, 180]."""
    a = np.mod(np.asarray(a, dtype=float) + 180.0, 360.0) - 180.0
    a = np.where(a == -180.0, 180.0, a)
    return a if a.ndim else float(a)


# ---------------------------------------------------------------- direction judgement

def direction_theta(P, f, C) -> float:
    """Azimuth of C (degrees, left positive / right negative, range (-180, 180]) when standing at P with forward vector f (xy vector)."""
    f = np.asarray(f, dtype=float)[:2]
    f = f / np.linalg.norm(f)
    v = (np.asarray(C, dtype=float) - np.asarray(P, dtype=float))[:2]
    cross = f[0] * v[1] - f[1] * v[0]
    dot = f[0] * v[0] + f[1] * v[1]
    return float(wrap_deg(np.degrees(np.arctan2(cross, dot))))


def direction_theta_facing(P, F, C) -> float:
    """Azimuth of C when standing at P facing point F."""
    return direction_theta(P, (np.asarray(F, dtype=float) - np.asarray(P, dtype=float))[:2], C)


def theta_from_yaw(P, yaw_deg, C) -> float:
    """Azimuth of C when standing at P with heading yaw_deg."""
    y = np.radians(yaw_deg)
    return direction_theta(P, (np.cos(y), np.sin(y)), C)


def angle_margin(theta: float) -> float:
    """Minimum angular distance (degrees) from theta to the quadrant boundaries {0, +-90, 180}."""
    return float(min(abs(theta - b) for b in _BOUNDARIES_DEG))


def quadrant(theta: float):
    """Map theta to one of the four quadrants; returns None when it lies exactly on a boundary."""
    if theta in _BOUNDARIES_DEG:
        return None
    if 0 < theta < 90:
        return "front-left"
    if 90 < theta <= 180:
        return "back-left"
    if -90 < theta < 0:
        return "front-right"
    return "back-right"


def is_left(theta: float) -> bool:
    return theta > 0


def projection_outside_segment(A, B, C) -> bool:
    """Whether the projection of C onto line AB in the xy plane falls outside segment AB.

    Only then does the front-left <-> back-right mapping of the direction_inverse constraint hold strictly;
    if it falls inside the segment, both ends see C in front and only left/right are swapped.
    """
    A, B, C = (np.asarray(x, dtype=float)[:2] for x in (A, B, C))
    ab = B - A
    s = float(np.dot(C - A, ab) / np.dot(ab, ab))
    return s < 0.0 or s > 1.0


# ---------------------------------------------------------------- poses and coordinates

def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def axis_angle_to_matrix(rotvec) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()


def world_transform(R_world: np.ndarray) -> np.ndarray:
    """Wrap a 3x3 world rotation into a 4x4 matrix (left-multiplied onto T_wc or points)."""
    return make_T(R_world, np.zeros(3))


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def yaw_deg_from_Twc(T_wc: np.ndarray) -> float:
    f = T_wc[:3, 2]  # world direction of the OpenCV camera forward (z) axis
    return float(np.degrees(np.arctan2(f[1], f[0])))


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation of translation, SLERP of rotation. alpha in [0,1]; 0 corresponds to T0."""
    rots = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    R = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return make_T(R, t)


def pose_indices_at_timestamp(ts_query: float, ts: np.ndarray, nearest_tol_s: float, interp_max_gap_s: float):
    """Determine which traj poses are used for a timestamp (Step 1 rule).

    If the time gap to the nearest pose is <= nearest_tol_s, only that pose is used and ([j], None) is returned; otherwise interpolate between the
    poses before and after, which must be <= interp_max_gap_s apart, and return ([i-1, i], alpha). Returns None if neither holds. ts must be ascending.
    """
    i = int(np.searchsorted(ts, ts_query))
    cand = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    if not cand:
        return None
    j_near = min(cand, key=lambda j: abs(ts[j] - ts_query))
    if abs(ts[j_near] - ts_query) <= nearest_tol_s:
        return [j_near], None
    if i == 0 or i >= len(ts):
        return None
    t0, t1 = ts[i - 1], ts[i]
    if t1 - t0 > interp_max_gap_s:
        return None
    return [i - 1, i], (ts_query - t0) / (t1 - t0)


def pose_at_timestamp(ts_query: float, ts: np.ndarray, poses: np.ndarray,
                      nearest_tol_s: float, interp_max_gap_s: float):
    """Pose at a timestamp (rule in pose_indices_at_timestamp); None if the conditions are not met."""
    r = pose_indices_at_timestamp(ts_query, ts, nearest_tol_s, interp_max_gap_s)
    if r is None:
        return None
    idx, alpha = r
    return poses[idx[0]] if alpha is None else interpolate_pose(poses[idx[0]], poses[idx[1]], alpha)


# ---------------------------------------------------------------- image uprighting

# scene orientation label -> number of 90-degree clockwise image rotations; names match sky_direction in ARKitScenes metadata.csv
SKY_TO_CW_QUARTERS = {"Up": 0, "Left": 1, "Down": 2, "Right": 3}


def sky_direction_from_pose(T_wc: np.ndarray, up_world=(0.0, 0.0, 1.0)) -> str:
    """Which way world "up" points in the image: Up / Down / Left / Right."""
    up_c = T_wc[:3, :3].T @ np.asarray(up_world, dtype=float)
    x, y = up_c[0], up_c[1]
    if abs(y) >= abs(x):
        return "Up" if y < 0 else "Down"
    return "Right" if x > 0 else "Left"


def _rot90_cw_once(K: np.ndarray, T_wc: np.ndarray, W: int, H: int):
    """Intrinsics, pose and size after rotating the image 90 deg clockwise (cv2.ROTATE_90_CLOCKWISE).

    Pixel mapping u' = (H-1) - v, v' = u; in new camera coordinates X' = -Y, Y' = X, Z' = Z.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_new = np.array([[fy, 0.0, (H - 1) - cy],
                      [0.0, fx, cx],
                      [0.0, 0.0, 1.0]])
    R_newcam_from_old = np.array([[0.0, -1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]])
    T_new = T_wc.copy()
    T_new[:3, :3] = T_wc[:3, :3] @ R_newcam_from_old.T
    return K_new, T_new, H, W


def rotate_camera_cw(K: np.ndarray, T_wc: np.ndarray, W: int, H: int, quarters: int):
    """Rotate the image clockwise by quarters x 90 deg and update K, T_wc and (W, H) accordingly."""
    K, T_wc = np.asarray(K, dtype=float), np.asarray(T_wc, dtype=float)
    for _ in range(quarters % 4):
        K, T_wc, W, H = _rot90_cw_once(K, T_wc, W, H)
    return K, T_wc, W, H


def rotate_image_cw(img: np.ndarray, quarters: int) -> np.ndarray:
    """Image rotation matching rotate_camera_cw (k of np.rot90 counts counter-clockwise turns)."""
    return np.ascontiguousarray(np.rot90(img, k=-(quarters % 4)))


# ---------------------------------------------------------------- projection and rendering

def project(points_w: np.ndarray, K: np.ndarray, T_wc: np.ndarray):
    """Project world points to pixels. Returns (uv[N,2], z[N]), z being camera-frame depth."""
    T_cw = np.linalg.inv(T_wc)
    pc = transform_points(T_cw, points_w)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def in_image(uv: np.ndarray, z: np.ndarray, W: int, H: int) -> np.ndarray:
    """With pixel centres at integers, the image spans [-0.5, W-0.5) x [-0.5, H-0.5)."""
    return (z > 0) & (uv[:, 0] >= -0.5) & (uv[:, 0] < W - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < H - 0.5)


def pixel_rays(K: np.ndarray, T_wc: np.ndarray, W: int, H: int) -> np.ndarray:
    """Rays through every pixel centre, [H*W, 6] (origin + direction). The camera-frame z component of the direction is 1,
    so the t_hit returned by open3d equals z depth directly."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)
    d_w = d_c @ T_wc[:3, :3].T
    o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
    return np.concatenate([o_w, d_w], axis=1).astype(np.float32)


def render_depth(raycast_scene, K: np.ndarray, T_wc: np.ndarray, W: int, H: int) -> np.ndarray:
    """Render a z-depth map (metres) with open3d RaycastingScene; misses are inf."""
    import open3d as o3d
    rays = o3d.core.Tensor(pixel_rays(K, T_wc, W, H))
    t = raycast_scene.cast_rays(rays)["t_hit"].numpy()
    return t.reshape(H, W)
