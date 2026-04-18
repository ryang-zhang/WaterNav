"""Geometric utilities: 2-D rotations, coordinate transforms, etc."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """2x2 rotation matrix for angle *theta* (radians, CCW positive)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points(
    points: NDArray[np.float64],
    tx: float,
    ty: float,
    theta: float,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Apply similarity transform (scale, rotate, translate) to Nx2 points.

    q_i = R(theta) @ (scale * p_i) + [tx, ty]
    """
    R = rotation_matrix(theta)
    return (scale * points) @ R.T + np.array([tx, ty])


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def pose_to_matrix(x: float, y: float, theta: float) -> NDArray[np.float64]:
    """Pose (x, y, theta) -> 3x3 homogeneous SE(2) matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1],
    ], dtype=np.float64)


def matrix_to_pose(T: NDArray[np.float64]) -> tuple[float, float, float]:
    """3x3 SE(2) matrix -> (x, y, theta)."""
    x = float(T[0, 2])
    y = float(T[1, 2])
    theta = float(np.arctan2(T[1, 0], T[0, 0]))
    return x, y, theta


def pose_difference(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Position error (Euclidean) between Nx3 pose arrays [x, y, theta].

    Returns Nx1 array of translational errors.
    """
    return np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
