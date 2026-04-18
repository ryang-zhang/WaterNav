"""Visualisation functions for the experiment design figures.

All plotting functions return a matplotlib Figure so callers can
either show or save them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse
except ImportError:
    plt = None  # type: ignore[assignment]


# -----------------------------------------------------------------------
# Hero Figure
# -----------------------------------------------------------------------
def plot_hero_figure(
    timestamps: NDArray[np.float64],
    errors: dict[str, NDArray[np.float64]],
    zone_labels: NDArray[np.int32],
    zone_names: Sequence[str] = ("high-texture", "low-texture", "no-texture"),
    zone_colors: Sequence[str] = ("#4CAF50", "#FFC107", "#F44336"),
) -> Figure:
    """Error-over-time curve with coloured background bands per zone type.

    Parameters
    ----------
    timestamps : T-length array
    errors : method_name -> T-length error array
    zone_labels : T-length int array (0/1/2 matching *zone_names*)
    """
    raise NotImplementedError("plot_hero_figure")


# -----------------------------------------------------------------------
# Uncertainty evolution
# -----------------------------------------------------------------------
def plot_uncertainty_evolution(
    timestamps: NDArray[np.float64],
    covariances: NDArray[np.float64],
    modes: NDArray[np.int32],
) -> Figure:
    """Plot covariance trace / eigenvalues over time, coloured by mode.

    Parameters
    ----------
    covariances : Tx3x3 array
    modes : T-length int array (0=A, 1=B, 2=C)
    """
    raise NotImplementedError("plot_uncertainty_evolution")


# -----------------------------------------------------------------------
# Error ellipses on satellite map
# -----------------------------------------------------------------------
def plot_error_ellipses(
    sat_image: NDArray[np.uint8],
    poses: NDArray[np.float64],
    covariances: NDArray[np.float64],
    every_n: int = 10,
    scale: float = 3.0,
) -> Figure:
    """Overlay 2-D error ellipses on the satellite image.

    Parameters
    ----------
    sat_image : H x W x 3
    poses : Nx3 [x, y, theta]
    covariances : Nx3x3
    every_n : draw one ellipse every N frames
    scale : sigma multiplier (default 3-sigma)
    """
    raise NotImplementedError("plot_error_ellipses")


# -----------------------------------------------------------------------
# Shoreline geometry vs accuracy
# -----------------------------------------------------------------------
def plot_shoreline_vs_accuracy(
    curvature_complexity: NDArray[np.float64],
    mode_b_errors: NDArray[np.float64],
    shoreline_lengths: NDArray[np.float64] | None = None,
) -> Figure:
    """Scatter plot: shoreline curvature complexity vs Mode-B localisation error.

    Parameters
    ----------
    curvature_complexity : per-frame scalar measure
    mode_b_errors : translational error when in Mode B
    shoreline_lengths : optional, for a second subplot
    """
    raise NotImplementedError("plot_shoreline_vs_accuracy")
