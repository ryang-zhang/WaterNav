"""Core data structures shared across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class Mode(Enum):
    A = auto()  # feature matching
    B = auto()  # shoreline contour SDF alignment
    C = auto()  # constant-velocity propagation


@dataclass
class PoseEstimate:
    """Output of a single-frame localisation step."""
    x: float
    y: float
    theta: float
    covariance: NDArray[np.float64]  # 3x3
    mode: Mode
    timestamp: float = 0.0

    @property
    def position(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y])

    @property
    def pose_vector(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.theta])


@dataclass
class ModeAOutput:
    """Raw output from Mode-A feature matching."""
    x: float
    y: float
    theta: float
    covariance: NDArray[np.float64]  # 3x3
    inlier_ratio: float = 0.0
    num_inliers: int = 0
    reproj_error: float = 0.0
    success: bool = False


@dataclass
class ModeBOutput:
    """Raw output from Mode-B contour alignment."""
    x: float
    y: float
    theta: float
    covariance: NDArray[np.float64]  # 3x3, Hessian-based anisotropic
    cost: float = 0.0
    seg_confidence: float = 0.0
    shoreline_length: float = 0.0
    constraint_eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    success: bool = False


@dataclass
class ModeCOutput:
    """Raw output from Mode-C propagation."""
    x: float
    y: float
    theta: float
    covariance: NDArray[np.float64]  # 3x3
