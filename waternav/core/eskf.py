"""Error-State Kalman Filter for degradation-aware temporal fusion.

State: (x, y, theta) — 3-DOF planar pose.
Observation noise R_t is mode-dependent: Mode-B feeds the Hessian-based
anisotropic covariance directly, so the Kalman gain automatically adjusts
update strength per direction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from waternav.config import ESKFConfig


class ESKF:
    """Planar Error-State Kalman Filter (pure NumPy)."""

    def __init__(self, cfg: ESKFConfig) -> None:
        self.cfg = cfg
        self.x = np.zeros(3)  # [x, y, theta]
        self.P = np.diag([
            cfg.init_pos_std ** 2,
            cfg.init_pos_std ** 2,
            cfg.init_theta_std ** 2,
        ])
        self.Q = np.diag([
            cfg.process_noise_xy ** 2,
            cfg.process_noise_xy ** 2,
            cfg.process_noise_theta ** 2,
        ])
        self._velocity = np.zeros(2)  # estimated velocity for prediction

    # ------------------------------------------------------------------
    def reset(self, x0: NDArray[np.float64], P0: NDArray[np.float64]) -> None:
        self.x = x0.copy()
        self.P = P0.copy()

    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        """Constant-velocity prediction step."""
        self.x[:2] += self._velocity * dt
        F = np.eye(3)
        self.P = F @ self.P @ F.T + self.Q * dt

    # ------------------------------------------------------------------
    def update(
        self,
        z: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> None:
        """Observation update.

        Parameters
        ----------
        z : (3,) observation [x, y, theta]
        R : (3, 3) observation noise covariance — comes directly from
            mode-A isotropic or mode-B anisotropic Hessian covariance.
        """
        H = np.eye(3)
        y = z - self.x  # innovation
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi  # wrap angle

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(3) - K @ H) @ self.P

    # ------------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float) -> None:
        self._velocity = np.array([vx, vy])

    @property
    def state(self) -> NDArray[np.float64]:
        return self.x.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self.P.copy()
