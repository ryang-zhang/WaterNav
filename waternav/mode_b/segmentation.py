"""Water / land segmentation for shoreline contour extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeBConfig


@dataclass
class SegmentationResult:
    """Output of the water/land segmentor."""
    mask: NDArray[np.uint8]               # H x W binary mask (water=1)
    contour_points: NDArray[np.float64]   # Nx2 shoreline contour in image coords
    confidence: float = 0.0
    shoreline_length: float = 0.0         # in pixels
    valid: bool = False


class Segmentor:
    """Wraps a pretrained segmentation model (SAM2 or custom).

    The segmentation model itself is pluggable; this class handles
    morphological post-processing and contour extraction regardless
    of the upstream model.
    """

    def __init__(self, cfg: ModeBConfig) -> None:
        self.cfg = cfg
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    def segment(self, drone_img: NDArray[np.uint8]) -> SegmentationResult:
        """Segment *drone_img* into water / land and extract shoreline contour.

        Pipeline:
        1. Run segmentation model -> raw mask
        2. Morphological closing/opening with ``cfg.morphology_kernel_size``
        3. Extract longest contour via ``cv2.findContours``
        4. Smooth contour, compute confidence and length
        """
        raise NotImplementedError("Segmentor.segment")
