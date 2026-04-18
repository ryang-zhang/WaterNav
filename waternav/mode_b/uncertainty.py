"""Hessian-based anisotropic uncertainty estimation for Mode B.

At the optimum T*, the covariance is  Sigma_B ≈ sigma^2 * H^{-1}
where H = J^T J is the 3x3 Gauss-Newton Hessian.

Eigenvalue decomposition reveals constraint geometry:
  - straight shoreline -> constrains 1 DOF (perpendicular direction)
  - L-shaped corner    -> constrains 2 DOFs
  - complex bend       -> near-full 3-DOF constraint
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeBConfig
from waternav.mode_b.sdf import SDFField
from waternav.utils.geo import transform_points


@dataclass
class CovarianceResult:
    covariance: NDArray[np.float64]      # 3x3
    eigenvalues: NDArray[np.float64]     # 3, ascending
    eigenvectors: NDArray[np.float64]    # 3x3, columns
    constrained_dofs: int = 0            # how many DOFs are well-constrained


class UncertaintyEstimator:
    """Compute Hessian-based uncertainty at the alignment optimum."""

    def __init__(self, cfg: ModeBConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def compute(
        self,
        pose: NDArray[np.float64],
        contour_pts: NDArray[np.float64],
        sdf: SDFField,
    ) -> CovarianceResult:
        """Compute anisotropic covariance from the alignment Hessian.

        Parameters
        ----------
        pose : [x, y, theta] — optimised pose from ContourAligner
        contour_pts : Nx2 contour points (image frame)
        sdf : signed distance field
        """
        H = self._build_hessian(pose, contour_pts, sdf)
        sigma2 = self.cfg.sigma_sdf ** 2

        try:
            cov = sigma2 * np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = sigma2 * np.eye(3) * 1e6  # degenerate fallback

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)

        well_constrained_thresh = sigma2 * 100
        constrained = int(np.sum(eigenvalues < well_constrained_thresh))

        return CovarianceResult(
            covariance=cov,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            constrained_dofs=constrained,
        )

    # ------------------------------------------------------------------
    def _build_hessian(
        self,
        pose: NDArray[np.float64],
        pts: NDArray[np.float64],
        sdf: SDFField,
    ) -> NDArray[np.float64]:
        """Build approximate Hessian H = J^T J at the given pose."""
        transformed = transform_points(pts, pose[0], pose[1], pose[2])
        _, grads = sdf.query(transformed)

        N = len(pts)
        J = np.zeros((N, 3))
        J[:, 0] = grads[:, 0]
        J[:, 1] = grads[:, 1]

        s, c = np.sin(pose[2]), np.cos(pose[2])
        dq_dtheta = pts @ np.array([[-s, c], [-c, -s]])
        J[:, 2] = grads[:, 0] * dq_dtheta[:, 0] + grads[:, 1] * dq_dtheta[:, 1]

        return J.T @ J
