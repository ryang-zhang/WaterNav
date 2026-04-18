"""Evaluation metrics as defined in the experiment design document.

All functions accept Nx3 arrays of [x, y, theta] poses.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_ate(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
) -> float:
    """Absolute Trajectory Error — RMSE of translational errors over all frames."""
    errs = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_ate_nt(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
    nt_mask: NDArray[np.bool_],
) -> float:
    """ATE computed only on no-texture segments (the most important metric)."""
    errs = np.linalg.norm(pred[nt_mask, :2] - gt[nt_mask, :2], axis=1)
    if len(errs) == 0:
        return 0.0
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_max_e_nt(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
    nt_mask: NDArray[np.bool_],
) -> float:
    """Maximum translational error within no-texture segments."""
    errs = np.linalg.norm(pred[nt_mask, :2] - gt[nt_mask, :2], axis=1)
    if len(errs) == 0:
        return 0.0
    return float(np.max(errs))


def compute_rpe(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
) -> float:
    """Relative Pose Error — RMSE of frame-to-frame relative translational errors."""
    pred_delta = np.diff(pred[:, :2], axis=0)
    gt_delta = np.diff(gt[:, :2], axis=0)
    errs = np.linalg.norm(pred_delta - gt_delta, axis=1)
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_recov_e(
    pred: NDArray[np.float64],
    gt: NDArray[np.float64],
    nt_mask: NDArray[np.bool_],
    normal_thresh: float = 5.0,
) -> int:
    """Recovery frames — number of frames after leaving no-texture zone
    until translational error drops below *normal_thresh* (meters).

    Returns the maximum recovery count across all no-texture exit points.
    """
    errs = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    transitions = np.where(np.diff(nt_mask.astype(int)) == -1)[0]
    max_recovery = 0
    for t in transitions:
        for k in range(t + 1, len(errs)):
            if errs[k] < normal_thresh:
                max_recovery = max(max_recovery, k - t)
                break
        else:
            max_recovery = max(max_recovery, len(errs) - t)
    return max_recovery


def compute_anees(
    errors: NDArray[np.float64],
    covariances: NDArray[np.float64],
) -> float:
    """Average Normalised Estimation Error Squared.

    ANEES = (1/N) sum_t  e_t^T  Sigma_t^{-1}  e_t
    Ideal value = 3 (state dimension).

    Parameters
    ----------
    errors : Nx3 array of estimation errors [dx, dy, dtheta]
    covariances : Nx3x3 array of covariance matrices
    """
    N = len(errors)
    nees_vals = np.zeros(N)
    for i in range(N):
        e = errors[i]
        try:
            nees_vals[i] = e @ np.linalg.solve(covariances[i], e)
        except np.linalg.LinAlgError:
            nees_vals[i] = np.nan
    return float(np.nanmean(nees_vals))
