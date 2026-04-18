"""Pose estimation from feature correspondences (Mode A)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeAConfig
from waternav.core import ModeAOutput
from waternav.mode_a.matcher import MatchResult


class PoseEstimatorA:
    """Estimate (x, y, theta) + isotropic covariance from feature matches."""

    def __init__(self, cfg: ModeAConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def estimate(self, matches: MatchResult) -> ModeAOutput:
        """Estimate pose from *matches* via RANSAC + least-squares.

        Returns
        -------
        ModeAOutput with pose, covariance Sigma_A, quality metrics.
        """
        raise NotImplementedError("PoseEstimatorA.estimate")
