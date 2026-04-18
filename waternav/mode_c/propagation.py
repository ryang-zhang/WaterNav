"""Mode C: constant-velocity propagation with map feasible-region truncation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from waternav.config import ModeCConfig
from waternav.core import ModeCOutput


class ConstantVelocityPropagator:
    """Dead-reckoning with optional map constraint truncation.

    Role: bridge strategy that maintains a rough pose prior until
    Mode A or B can re-engage.
    """

    def __init__(self, cfg: ModeCConfig) -> None:
        self.cfg = cfg
        self._velocity = np.zeros(2)  # (vx, vy) in m/s
        self._omega = 0.0             # angular velocity rad/s

    # ------------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float, omega: float = 0.0) -> None:
        self._velocity = np.array([vx, vy])
        self._omega = omega

    # ------------------------------------------------------------------
    def propagate(
        self,
        pose: NDArray[np.float64],
        covariance: NDArray[np.float64],
        dt: float,
    ) -> ModeCOutput:
        """Propagate pose forward by *dt* seconds.

        Parameters
        ----------
        pose : [x, y, theta]
        covariance : 3x3 current covariance
        dt : time step in seconds
        """
        cfg = self.cfg
        new_pose = pose.copy()
        new_pose[:2] += self._velocity * dt
        new_pose[2] += self._omega * dt
        new_pose[2] = (new_pose[2] + np.pi) % (2 * np.pi) - np.pi

        # decay velocity
        self._velocity *= cfg.velocity_decay
        self._omega *= cfg.velocity_decay

        # inflate covariance
        Q = np.diag([
            (cfg.process_noise_xy * dt) ** 2,
            (cfg.process_noise_xy * dt) ** 2,
            (cfg.process_noise_theta * dt) ** 2,
        ])
        new_cov = covariance + Q

        return ModeCOutput(
            x=new_pose[0],
            y=new_pose[1],
            theta=new_pose[2],
            covariance=new_cov,
        )

    # ------------------------------------------------------------------
    def truncate_to_feasible_region(
        self,
        pose: NDArray[np.float64],
        feasible_mask: NDArray[np.uint8],
        origin: tuple[float, float],
        resolution: float,
    ) -> NDArray[np.float64]:
        """Snap pose to nearest feasible point if it falls outside the map.

        This is a simple nearest-neighbour projection.  A more sophisticated
        version could use the SDF to find the closest feasible point.
        """
        raise NotImplementedError("truncate_to_feasible_region")
