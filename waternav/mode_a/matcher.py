"""Unified wrapper for feature matchers (LoFTR / SuperGlue)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeAConfig


@dataclass
class MatchResult:
    """Keypoint correspondences between drone and satellite images."""
    kpts_drone: NDArray[np.float64]   # Nx2
    kpts_sat: NDArray[np.float64]     # Nx2
    confidences: NDArray[np.float64]  # N
    num_matches: int = 0


class FeatureMatcher:
    """Wraps a pretrained matcher (LoFTR or SuperGlue).

    The actual model loading and inference will be filled in during
    the implementation phase.  The interface is fixed here.
    """

    def __init__(self, cfg: ModeAConfig) -> None:
        self.cfg = cfg
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    def match(
        self,
        drone_img: NDArray[np.uint8],
        sat_patch: NDArray[np.uint8],
    ) -> MatchResult:
        """Find correspondences between *drone_img* and *sat_patch*.

        Returns
        -------
        MatchResult with filtered keypoints above ``confidence_thresh``.
        """
        raise NotImplementedError("FeatureMatcher.match")
