"""Signed Distance Field construction and querying (pure Python).

Uses ``scipy.ndimage.distance_transform_edt`` for the heavy lifting.
The SDF is a 2-D array where:
  - shoreline pixels = 0
  - water side < 0
  - land side > 0
Gradients are pre-computed with ``np.gradient`` for fast lookup.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from waternav.config import ModeBConfig


class SDFField:
    """Pre-computed signed distance field on the satellite map."""

    def __init__(
        self,
        sdf_array: NDArray[np.float64],
        grad_x: NDArray[np.float64],
        grad_y: NDArray[np.float64],
        origin: tuple[float, float],
        resolution: float,
    ) -> None:
        self.sdf = sdf_array
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.origin = origin  # world coords of pixel (0,0)
        self.resolution = resolution  # meters per pixel

    @property
    def shape(self) -> tuple[int, int]:
        return self.sdf.shape  # type: ignore[return-value]

    # ------------------------------------------------------------------
    @classmethod
    def from_satellite_map(
        cls,
        sat_map: object,
        cfg: ModeBConfig,
    ) -> SDFField:
        """Build SDF from a ``SatelliteMap`` instance.

        Steps:
        1. Obtain binary water/land mask from satellite map
        2. Compute distance transform for water side and land side
        3. Combine into signed field (water negative, land positive)
        4. Pre-compute spatial gradients
        """
        raise NotImplementedError("SDFField.from_satellite_map")

    # ------------------------------------------------------------------
    @classmethod
    def from_mask(
        cls,
        water_mask: NDArray[np.uint8],
        origin: tuple[float, float] = (0.0, 0.0),
        resolution: float = 1.0,
    ) -> SDFField:
        """Build SDF directly from a binary mask (water=1, land=0).

        This is the low-level constructor used by ``from_satellite_map``
        and also useful for unit testing.
        """
        land_mask = 1 - water_mask.astype(bool)
        dist_water = distance_transform_edt(water_mask)
        dist_land = distance_transform_edt(land_mask)
        sdf = dist_land - dist_water  # positive on land, negative on water

        grad_y, grad_x = np.gradient(sdf)

        return cls(
            sdf_array=sdf.astype(np.float64),
            grad_x=grad_x.astype(np.float64),
            grad_y=grad_y.astype(np.float64),
            origin=origin,
            resolution=resolution,
        )

    # ------------------------------------------------------------------
    def query(
        self,
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Query SDF values and gradients at world-coordinate *points* (Nx2).

        Returns
        -------
        values : (N,) SDF values via bilinear interpolation
        gradients : (N, 2) spatial gradients [dφ/dx, dφ/dy]
        """
        px = (points[:, 0] - self.origin[0]) / self.resolution
        py = (points[:, 1] - self.origin[1]) / self.resolution

        values = self._bilinear(self.sdf, px, py)
        gx = self._bilinear(self.grad_x, px, py)
        gy = self._bilinear(self.grad_y, px, py)
        # np.gradient gives d(sdf)/d(pixel), convert to d(sdf)/d(world)
        gradients = np.column_stack([gx, gy]) / self.resolution
        return values, gradients

    # ------------------------------------------------------------------
    @staticmethod
    def _bilinear(
        field: NDArray[np.float64],
        px: NDArray[np.float64],
        py: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Bilinear interpolation on a 2-D array at sub-pixel locations."""
        h, w = field.shape
        x0 = np.clip(np.floor(px).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(py).astype(int), 0, h - 2)
        x1, y1 = x0 + 1, y0 + 1
        wx = px - x0
        wy = py - y0
        return (
            field[y0, x0] * (1 - wx) * (1 - wy)
            + field[y0, x1] * wx * (1 - wy)
            + field[y1, x0] * (1 - wx) * wy
            + field[y1, x1] * wx * wy
        )
