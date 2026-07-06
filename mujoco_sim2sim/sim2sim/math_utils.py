"""Small numerical helpers shared by the MuJoCo runner."""

from __future__ import annotations

import math

import numpy as np


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return world-from-body rotation for a MuJoCo wxyz quaternion."""
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def projected_gravity_from_quat(q: np.ndarray) -> np.ndarray:
    """Gravity vector expressed in body frame, matching IsaacLab projected_gravity."""
    rot_wb = quat_wxyz_to_rotmat(q)
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return rot_wb.T @ gravity_w


def yaw_quat(rad: float) -> np.ndarray:
    half = 0.5 * rad
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def apply_yaw_correction(q: np.ndarray, yaw_rad: float) -> np.ndarray:
    if abs(yaw_rad) < 1e-9:
        return q.astype(np.float32, copy=True)
    out = quat_mul(yaw_quat(yaw_rad), q.astype(np.float32))
    return out / np.linalg.norm(out)


def pd_torque(
    target_pos: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    effort_limit: np.ndarray,
) -> np.ndarray:
    tau = kp * (target_pos - qpos) - kd * qvel
    return np.clip(tau, -effort_limit, effort_limit).astype(np.float32)
