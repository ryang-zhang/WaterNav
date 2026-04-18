"""SDF-based contour-to-map alignment optimisation (Mode B).

Cost function: E(x, y, theta) = sum_i rho( phi( R(theta)*s*p_i + t(x,y) ) )
where rho is the Huber robust kernel.

Optimisation: coarse grid search + Gauss-Newton refinement (3-DOF).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeBConfig
from waternav.mode_b.sdf import SDFField
from waternav.utils.geo import transform_points


@dataclass
class AlignmentResult:
    pose: NDArray[np.float64]  # [x, y, theta]
    cost: float = np.inf
    converged: bool = False
    iterations: int = 0


class ContourAligner:
    """Align drone contour points to satellite SDF."""

    def __init__(self, cfg: ModeBConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def align(
        self,
        contour_pts: NDArray[np.float64],
        sdf: SDFField,
        init_pose: NDArray[np.float64],
    ) -> AlignmentResult:
        """Full alignment pipeline: grid search → Gauss-Newton.

        Parameters
        ----------
        contour_pts : Nx2 shoreline points in drone image coordinates
        sdf : pre-built signed distance field
        init_pose : [x, y, theta] initial guess (from ESKF prediction)
        """
        best_pose = self._grid_search(contour_pts, sdf, init_pose)
        result = self._gauss_newton(contour_pts, sdf, best_pose)
        return result

    # ------------------------------------------------------------------
    def _grid_search(
        self,
        pts: NDArray[np.float64],
        sdf: SDFField,
        center: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Coarse grid search around *center* to find a good starting point."""
        cfg = self.cfg
        best_cost = np.inf
        best_pose = center.copy()

        xs = np.arange(
            center[0] - cfg.grid_search_range,
            center[0] + cfg.grid_search_range,
            cfg.grid_search_step,
        )
        ys = np.arange(
            center[1] - cfg.grid_search_range,
            center[1] + cfg.grid_search_range,
            cfg.grid_search_step,
        )
        thetas = np.arange(
            center[2] - cfg.grid_search_angle_range,
            center[2] + cfg.grid_search_angle_range,
            cfg.grid_search_angle_step,
        )

        for tx in xs:
            for ty in ys:
                for th in thetas:
                    cost = self._cost(pts, sdf, tx, ty, th)
                    if cost < best_cost:
                        best_cost = cost
                        best_pose = np.array([tx, ty, th])
        return best_pose

    # ------------------------------------------------------------------
    def _gauss_newton(
        self,
        pts: NDArray[np.float64],
        sdf: SDFField,
        x0: NDArray[np.float64],
    ) -> AlignmentResult:
        """Gauss-Newton refinement on the 3-DOF pose."""
        pose = x0.copy()
        cfg = self.cfg

        for it in range(cfg.gn_max_iters):
            transformed = transform_points(pts, pose[0], pose[1], pose[2])
            vals, grads = sdf.query(transformed)

            # Huber weights
            abs_v = np.abs(vals)
            w = np.where(abs_v <= cfg.huber_delta, 1.0, cfg.huber_delta / abs_v)

            # Jacobian: d(phi(q_i))/d[tx, ty, theta]
            N = len(pts)
            J = np.zeros((N, 3))
            J[:, 0] = grads[:, 0]  # d/dtx
            J[:, 1] = grads[:, 1]  # d/dty
            s, c = np.sin(pose[2]), np.cos(pose[2])
            dq_dtheta = pts @ np.array([[-s, c], [-c, -s]])
            J[:, 2] = grads[:, 0] * dq_dtheta[:, 0] + grads[:, 1] * dq_dtheta[:, 1]

            W = np.diag(w)
            JtWJ = J.T @ W @ J
            JtWr = J.T @ W @ vals

            try:
                delta = np.linalg.solve(JtWJ, -JtWr)
            except np.linalg.LinAlgError:
                break

            pose += delta

            if np.linalg.norm(delta) < cfg.gn_convergence_eps:
                return AlignmentResult(
                    pose=pose,
                    cost=float(np.sum(w * vals ** 2)),
                    converged=True,
                    iterations=it + 1,
                )

        return AlignmentResult(
            pose=pose,
            cost=float(self._cost(pts, sdf, pose[0], pose[1], pose[2])),
            converged=False,
            iterations=cfg.gn_max_iters,
        )

    # ------------------------------------------------------------------
    def _cost(
        self,
        pts: NDArray[np.float64],
        sdf: SDFField,
        tx: float,
        ty: float,
        theta: float,
    ) -> float:
        """Evaluate Huber cost at a single pose."""
        transformed = transform_points(pts, tx, ty, theta)
        vals, _ = sdf.query(transformed)
        delta = self.cfg.huber_delta
        abs_v = np.abs(vals)
        loss = np.where(
            abs_v <= delta,
            0.5 * vals ** 2,
            delta * (abs_v - 0.5 * delta),
        )
        return float(np.sum(loss))
