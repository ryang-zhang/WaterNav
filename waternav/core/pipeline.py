"""Main pipeline that orchestrates Modes A / B / C + ESKF fusion."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from waternav.config import Config
from waternav.core import Mode, ModeAOutput, ModeBOutput, ModeCOutput, PoseEstimate
from waternav.core.eskf import ESKF
from waternav.core.mode_switcher import ModeSwitcher
from waternav.data.satellite import SatelliteMap
from waternav.mode_a.matcher import FeatureMatcher
from waternav.mode_a.pose_estimator import PoseEstimatorA
from waternav.mode_b.alignment import ContourAligner
from waternav.mode_b.segmentation import Segmentor
from waternav.mode_b.sdf import SDFField
from waternav.mode_b.uncertainty import UncertaintyEstimator
from waternav.mode_c.propagation import ConstantVelocityPropagator

logger = logging.getLogger("waternav")


class WaterNavPipeline:
    """End-to-end localisation pipeline."""

    def __init__(self, cfg: Config, sat_map: SatelliteMap) -> None:
        self.cfg = cfg
        self.sat_map = sat_map

        # core
        self.eskf = ESKF(cfg.eskf)
        self.switcher = ModeSwitcher(cfg.mode_switch)

        # mode A
        self.matcher = FeatureMatcher(cfg.mode_a)
        self.pose_est_a = PoseEstimatorA(cfg.mode_a)

        # mode B
        self.segmentor = Segmentor(cfg.mode_b)
        self.sdf: SDFField | None = None  # built lazily from sat_map
        self.aligner = ContourAligner(cfg.mode_b)
        self.uncertainty_est = UncertaintyEstimator(cfg.mode_b)

        # mode C
        self.propagator = ConstantVelocityPropagator(cfg.mode_c)

        self._prev_timestamp: float | None = None
        self._frame_idx = 0

    # ------------------------------------------------------------------
    def initialise(self, x0: float, y0: float, theta0: float) -> None:
        """Set initial pose (e.g. from first GPS fix)."""
        P0 = np.diag([
            self.cfg.eskf.init_pos_std ** 2,
            self.cfg.eskf.init_pos_std ** 2,
            self.cfg.eskf.init_theta_std ** 2,
        ])
        self.eskf.reset(np.array([x0, y0, theta0]), P0)
        logger.info("Pipeline initialised at (%.2f, %.2f, %.4f)", x0, y0, theta0)

    # ------------------------------------------------------------------
    def build_sdf(self) -> None:
        """Offline: build SDF from satellite map shoreline."""
        self.sdf = SDFField.from_satellite_map(self.sat_map, self.cfg.mode_b)
        logger.info("SDF built, shape=%s", self.sdf.shape)

    # ------------------------------------------------------------------
    def process_frame(
        self,
        drone_img: NDArray[np.uint8],
        timestamp: float,
    ) -> PoseEstimate:
        """Process one drone frame and return the fused pose estimate."""
        dt = 0.0
        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
        self._prev_timestamp = timestamp

        # --- ESKF predict ---
        if dt > 0:
            self.eskf.predict(dt)

        # --- try mode A ---
        mode_a_out = self._run_mode_a(drone_img)

        # --- try mode B (only if SDF is ready) ---
        mode_b_out: ModeBOutput | None = None
        if self.sdf is not None:
            mode_b_out = self._run_mode_b(drone_img)

        # --- mode switch ---
        active = self.switcher.update(mode_a_out, mode_b_out)

        # --- ESKF update or propagation ---
        if active == Mode.A and mode_a_out is not None and mode_a_out.success:
            z = np.array([mode_a_out.x, mode_a_out.y, mode_a_out.theta])
            self.eskf.update(z, mode_a_out.covariance)
        elif active == Mode.B and mode_b_out is not None and mode_b_out.success:
            z = np.array([mode_b_out.x, mode_b_out.y, mode_b_out.theta])
            self.eskf.update(z, mode_b_out.covariance)
        else:
            pass  # mode C: prediction-only, no observation update

        state = self.eskf.state
        cov = self.eskf.covariance
        self._frame_idx += 1

        return PoseEstimate(
            x=state[0],
            y=state[1],
            theta=state[2],
            covariance=cov,
            mode=active,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    def _run_mode_a(self, drone_img: NDArray[np.uint8]) -> ModeAOutput | None:
        """Run feature matching against satellite map patch."""
        sat_patch = self.sat_map.get_patch(
            self.eskf.state[0], self.eskf.state[1], margin=200,
        )
        if sat_patch is None:
            return None
        matches = self.matcher.match(drone_img, sat_patch)
        return self.pose_est_a.estimate(matches)

    # ------------------------------------------------------------------
    def _run_mode_b(self, drone_img: NDArray[np.uint8]) -> ModeBOutput | None:
        """Run shoreline segmentation + SDF alignment."""
        assert self.sdf is not None
        seg_result = self.segmentor.segment(drone_img)
        if not seg_result.valid:
            return None
        init_pose = self.eskf.state
        align_result = self.aligner.align(
            seg_result.contour_points, self.sdf, init_pose,
        )
        if not align_result.converged:
            return None
        cov = self.uncertainty_est.compute(
            align_result.pose, seg_result.contour_points, self.sdf,
        )
        return ModeBOutput(
            x=align_result.pose[0],
            y=align_result.pose[1],
            theta=align_result.pose[2],
            covariance=cov.covariance,
            cost=align_result.cost,
            seg_confidence=seg_result.confidence,
            shoreline_length=seg_result.shoreline_length,
            constraint_eigenvalues=cov.eigenvalues,
            success=True,
        )
