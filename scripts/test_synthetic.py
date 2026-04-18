#!/usr/bin/env python3
"""Synthetic validation of WaterNav Mode-B pipeline + ESKF + metrics.

No real images, models, or GPS data needed. All geometry is constructed
analytically so that ground truth is known exactly.

Run:  python scripts/test_synthetic.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waternav.config import Config, ESKFConfig, ModeBConfig, ModeSwitchConfig
from waternav.core import Mode, ModeAOutput, ModeBOutput
from waternav.core.eskf import ESKF
from waternav.core.mode_switcher import ModeSwitcher
from waternav.evaluation.metrics import (
    compute_anees,
    compute_ate,
    compute_ate_nt,
    compute_max_e_nt,
    compute_recov_e,
    compute_rpe,
)
from waternav.mode_b.alignment import ContourAligner
from waternav.mode_b.sdf import SDFField
from waternav.mode_b.uncertainty import UncertaintyEstimator
from waternav.utils.geo import rotation_matrix, transform_points
from waternav.utils.logger import setup_logger

logger = setup_logger(name="test_synthetic")

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    if condition:
        _PASS += 1
        logger.info(msg)
    else:
        _FAIL += 1
        logger.error(msg)
    return condition


# ======================================================================
# Helpers: synthetic mask generators
# ======================================================================

def make_straight_mask(size: int = 500, shoreline_y: int = 250) -> np.ndarray:
    """Water in bottom half (y > shoreline_y)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[shoreline_y:, :] = 1
    return mask


def make_l_corner_mask(size: int = 500) -> np.ndarray:
    """Water in bottom-right quadrant — L-shaped shoreline."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[250:, 250:] = 1
    return mask


def make_curved_mask(size: int = 500, cx: int = 250, cy: int = 250,
                     radius: int = 100) -> np.ndarray:
    """Circular lake centred at (cx, cy)."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (dist <= radius).astype(np.uint8)
    return mask


def shoreline_from_mask(mask: np.ndarray) -> np.ndarray:
    """Extract zero-crossing boundary points (Nx2 float64, [x, y])."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    longest = max(contours, key=len)
    pts = longest.reshape(-1, 2).astype(np.float64)  # [x, y]
    return pts


def world_to_local(world_pts: np.ndarray, tx: float, ty: float,
                   theta: float) -> np.ndarray:
    """Inverse of transform_points: world -> drone-local frame."""
    R_gt = rotation_matrix(theta)  # 2x2
    return (world_pts - np.array([tx, ty])) @ R_gt


# ======================================================================
# Test 1: SDF construction
# ======================================================================

def test_sdf_construction() -> None:
    logger.info("=" * 60)
    logger.info("TEST 1: SDF construction")
    logger.info("=" * 60)

    mask = make_straight_mask(200, shoreline_y=100)
    sdf = SDFField.from_mask(mask, origin=(0.0, 0.0), resolution=1.0)

    check("SDF shape", sdf.shape == (200, 200))
    check("land positive (y=50)", sdf.sdf[50, 100] > 0,
          f"val={sdf.sdf[50, 100]:.2f}")
    check("water negative (y=150)", sdf.sdf[150, 100] < 0,
          f"val={sdf.sdf[150, 100]:.2f}")
    check("shoreline near zero (y=100)", abs(sdf.sdf[100, 100]) < 1.5,
          f"val={sdf.sdf[100, 100]:.2f}")

    # on land near shore (y=95), SDF decreases toward water (+y direction),
    # so grad_y is negative (steepest ascent points away from water = -y)
    check("grad_y negative on land near shore",
          sdf.grad_y[95, 100] < 0,
          f"grad_y={sdf.grad_y[95, 100]:.3f}")

    # query interface
    pts = np.array([[100.0, 50.0], [100.0, 150.0], [100.0, 100.0]])
    vals, grads = sdf.query(pts)
    check("query: land val > 0", vals[0] > 0, f"val={vals[0]:.2f}")
    check("query: water val < 0", vals[1] < 0, f"val={vals[1]:.2f}")
    check("query: shoreline val ≈ 0", abs(vals[2]) < 1.5,
          f"val={vals[2]:.2f}")

    # non-unit resolution
    sdf2 = SDFField.from_mask(mask, origin=(0.0, 0.0), resolution=0.5)
    pts_world = np.array([[50.0, 25.0]])  # maps to pixel (100, 50) = land
    v2, g2 = sdf2.query(pts_world)
    check("non-unit resolution: correct sign", v2[0] > 0,
          f"res=0.5, val={v2[0]:.2f}")


# ======================================================================
# Test 2: Alignment — straight horizontal shoreline
# ======================================================================

def test_alignment_straight() -> None:
    logger.info("=" * 60)
    logger.info("TEST 2: Alignment — straight horizontal shoreline")
    logger.info("=" * 60)

    mask = make_straight_mask(500, shoreline_y=250)
    sdf = SDFField.from_mask(mask)

    # GT pose: drone at (250, 250, 0.0)
    # Note: SDF zero-crossing is at y≈249.5 (between land y=249 and water y=250)
    # so points at y=250 have SDF ≈ -1 — ~1 px discretisation offset is expected
    gt_pose = np.array([250.0, 250.0, 0.0])

    world_shore = np.column_stack([
        np.linspace(150, 350, 200),
        np.full(200, 250.0),
    ])

    local_pts = world_to_local(world_shore, *gt_pose)

    world_check = transform_points(local_pts, *gt_pose)
    vals_check, _ = sdf.query(world_check)
    check("GT cost small (discretisation offset)", np.mean(vals_check ** 2) < 2.0,
          f"mean_sq={np.mean(vals_check ** 2):.4f}")

    # perturb pose — only perturb y (the constrained direction)
    init_pose = gt_pose + np.array([0.0, 8.0, 0.0])

    cfg = ModeBConfig(
        grid_search_range=15.0,
        grid_search_step=3.0,
        grid_search_angle_range=0.1,
        grid_search_angle_step=0.02,
        gn_max_iters=50,
        gn_convergence_eps=1e-6,
        huber_delta=5.0,
    )
    aligner = ContourAligner(cfg)
    result = aligner.align(local_pts, sdf, init_pose)

    # y should be well-recovered — within 2 px of GT (1 px discretisation + 1 px tolerance)
    err_y = abs(result.pose[1] - gt_pose[1])
    check("y error < 2 px (perpendicular to shore)", err_y < 2.0,
          f"err_y={err_y:.3f}")

    # check Hessian eigenvalues directly (not covariance, which is ill-conditioned)
    unc = UncertaintyEstimator(cfg)
    H = unc._build_hessian(result.pose, local_pts, sdf)
    h_evals = np.sort(np.linalg.eigvalsh(H))
    logger.info("  Hessian eigenvalues: %s", np.array2string(h_evals, precision=4))
    check("Hessian: 1 large eigenvalue (perpendicular DOF)",
          h_evals[-1] > 10.0, f"max={h_evals[-1]:.2f}")
    check("Hessian: smallest eigenvalue near 0 (along-shore unconstrained)",
          h_evals[0] < 1.0, f"min={h_evals[0]:.4f}")
    ratio = h_evals[-1] / (h_evals[0] + 1e-12)
    check("Hessian anisotropy ratio > 100", ratio > 100,
          f"ratio={ratio:.1f}")


# ======================================================================
# Test 3: Alignment — L-shaped corner
# ======================================================================

def test_alignment_l_corner() -> None:
    logger.info("=" * 60)
    logger.info("TEST 3: Alignment — L-shaped corner")
    logger.info("=" * 60)

    mask = make_l_corner_mask(500)
    sdf = SDFField.from_mask(mask)

    gt_pose = np.array([250.0, 250.0, 0.0])

    # L-shaped contour: horizontal segment + vertical segment
    horiz = np.column_stack([np.linspace(250, 400, 100), np.full(100, 250.0)])
    vert = np.column_stack([np.full(100, 250.0), np.linspace(250, 400, 100)])
    world_shore = np.vstack([horiz, vert])

    local_pts = world_to_local(world_shore, *gt_pose)
    init_pose = gt_pose + np.array([5.0, 5.0, 0.05])

    cfg = ModeBConfig(
        grid_search_range=15.0,
        grid_search_step=3.0,
        grid_search_angle_range=0.15,
        grid_search_angle_step=0.03,
        gn_max_iters=50,
        gn_convergence_eps=1e-8,
        huber_delta=5.0,
    )
    aligner = ContourAligner(cfg)
    result = aligner.align(local_pts, sdf, init_pose)

    err_xy = np.linalg.norm(result.pose[:2] - gt_pose[:2])
    err_theta = abs(result.pose[2] - gt_pose[2])

    check("converged", result.converged, f"iters={result.iterations}")
    check("position error < 3 px", err_xy < 3.0, f"err={err_xy:.3f}")
    check("angle error < 0.05 rad", err_theta < 0.05,
          f"err={err_theta:.4f}")

    unc = UncertaintyEstimator(cfg)
    cov_result = unc.compute(result.pose, local_pts, sdf)
    evals = cov_result.eigenvalues
    logger.info("  Hessian eigenvalues: %s", np.array2string(evals, precision=4))
    check("constrained DOFs >= 2 for L-corner",
          cov_result.constrained_dofs >= 2,
          f"dofs={cov_result.constrained_dofs}")


# ======================================================================
# Test 4: Alignment — curved shoreline (circular lake)
# ======================================================================

def make_river_bend_mask(size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Create an asymmetric river bend — water below a wavy shoreline.

    Returns (mask, shoreline_world_pts).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    xs = np.arange(size)
    # y_shore(x) = 250 + 40*sin(2π*x/300) — one full period of wave
    y_shore = 250.0 + 40.0 * np.sin(2.0 * np.pi * xs / 300.0)
    for x in range(size):
        y_cut = int(round(y_shore[x]))
        if y_cut < size:
            mask[y_cut:, x] = 1  # water below shoreline

    # sample the shoreline curve for contour points
    sample_xs = np.linspace(100, 400, 200)
    sample_ys = 250.0 + 40.0 * np.sin(2.0 * np.pi * sample_xs / 300.0)
    shore_pts = np.column_stack([sample_xs, sample_ys])
    return mask, shore_pts


def test_alignment_curved() -> None:
    logger.info("=" * 60)
    logger.info("TEST 4: Alignment — curved shoreline (river bend)")
    logger.info("=" * 60)

    mask, world_shore = make_river_bend_mask(500)
    sdf = SDFField.from_mask(mask)

    gt_pose = np.array([250.0, 250.0, 0.05])
    local_pts = world_to_local(world_shore, *gt_pose)

    init_pose = gt_pose + np.array([4.0, 4.0, 0.04])

    cfg = ModeBConfig(
        grid_search_range=12.0,
        grid_search_step=2.0,
        grid_search_angle_range=0.15,
        grid_search_angle_step=0.03,
        gn_max_iters=50,
        gn_convergence_eps=1e-8,
        huber_delta=5.0,
    )
    aligner = ContourAligner(cfg)
    result = aligner.align(local_pts, sdf, init_pose)

    err_xy = np.linalg.norm(result.pose[:2] - gt_pose[:2])
    err_theta = abs(result.pose[2] - gt_pose[2])

    check("converged", result.converged, f"iters={result.iterations}")
    check("position error < 3 px", err_xy < 3.0, f"err={err_xy:.3f}")
    check("angle error < 0.03 rad", err_theta < 0.03,
          f"err={err_theta:.4f}")

    # Hessian should show near-full 3-DOF constraint for wavy shoreline
    unc = UncertaintyEstimator(cfg)
    H = unc._build_hessian(result.pose, local_pts, sdf)
    h_evals = np.sort(np.linalg.eigvalsh(H))
    logger.info("  Hessian eigenvalues: %s", np.array2string(h_evals, precision=4))
    check("Hessian: all eigenvalues > 1 (full constraint)",
          h_evals[0] > 1.0, f"min={h_evals[0]:.4f}")
    ratio = h_evals[-1] / (h_evals[0] + 1e-12)
    check("Hessian: finite anisotropy (all dirs constrained)", ratio < 1e6,
          f"ratio={ratio:.1f}")


# ======================================================================
# Test 5: ESKF predict / update cycle
# ======================================================================

def test_eskf() -> None:
    logger.info("=" * 60)
    logger.info("TEST 5: ESKF predict / update")
    logger.info("=" * 60)

    cfg = ESKFConfig(
        init_pos_std=5.0,
        init_theta_std=0.1,
        process_noise_xy=0.5,
        process_noise_theta=0.01,
    )
    kf = ESKF(cfg)
    kf.reset(np.array([100.0, 200.0, 0.5]), np.diag([25.0, 25.0, 0.01]))
    kf.set_velocity(1.0, 0.5)

    # predict 1 second
    kf.predict(1.0)
    check("predict: x advanced", abs(kf.state[0] - 101.0) < 0.01,
          f"x={kf.state[0]:.3f}")
    check("predict: y advanced", abs(kf.state[1] - 200.5) < 0.01,
          f"y={kf.state[1]:.3f}")
    check("predict: P grew", kf.covariance[0, 0] > 25.0,
          f"P00={kf.covariance[0, 0]:.3f}")

    # update with observation near truth
    z = np.array([101.2, 200.6, 0.51])
    R = np.diag([1.0, 1.0, 0.01])
    kf.update(z, R)
    check("update: x closer to obs",
          abs(kf.state[0] - 101.2) < abs(101.0 - 101.2),
          f"x={kf.state[0]:.3f}")
    check("update: P shrank", kf.covariance[0, 0] < 25.0,
          f"P00={kf.covariance[0, 0]:.3f}")

    # multiple updates should converge
    gt = np.array([105.0, 203.0, 0.55])
    R_tight = np.diag([0.5, 0.5, 0.005])
    for _ in range(20):
        kf.predict(0.1)
        kf.update(gt, R_tight)

    err = np.linalg.norm(kf.state[:2] - gt[:2])
    check("20 updates: converge to obs", err < 1.0, f"err={err:.3f}")

    # anisotropic update: tight in x, loose in y
    kf.reset(np.array([0.0, 0.0, 0.0]), np.diag([100.0, 100.0, 0.1]))
    R_aniso = np.diag([0.1, 1000.0, 0.1])
    kf.update(np.array([5.0, 5.0, 0.1]), R_aniso)
    check("aniso update: x pulled strongly",
          abs(kf.state[0] - 5.0) < abs(kf.state[1] - 5.0),
          f"x={kf.state[0]:.2f}, y={kf.state[1]:.2f}")


# ======================================================================
# Test 6: Mode switcher
# ======================================================================

def test_mode_switcher() -> None:
    logger.info("=" * 60)
    logger.info("TEST 6: Mode switcher")
    logger.info("=" * 60)

    cfg = ModeSwitchConfig(min_hold_frames=3)
    sw = ModeSwitcher(cfg)

    cov = np.eye(3)

    # start in A, good A output → stay A
    a_good = ModeAOutput(0, 0, 0, cov, inlier_ratio=0.5, reproj_error=5.0,
                         success=True)
    for _ in range(5):
        mode = sw.update(a_good, None)
    check("stay A when A is good", mode == Mode.A)

    # A becomes bad, B available → switch to B
    a_bad = ModeAOutput(0, 0, 0, cov, inlier_ratio=0.05, reproj_error=20.0,
                        success=True)
    b_ok = ModeBOutput(0, 0, 0, cov, seg_confidence=0.6,
                       shoreline_length=60.0, success=True)
    for _ in range(5):
        mode = sw.update(a_bad, b_ok)
    check("switch A->B when A bad, B ok", mode == Mode.B)

    # min hold: should not switch immediately
    sw2 = ModeSwitcher(ModeSwitchConfig(min_hold_frames=10))
    for _ in range(3):
        m = sw2.update(a_good, None)
    # even with bad A, min_hold prevents switch
    m = sw2.update(a_bad, b_ok)
    check("min_hold prevents early switch", m == Mode.A)

    # B bad, A bad → fall to C
    sw3 = ModeSwitcher(cfg)
    for _ in range(5):
        sw3.update(a_good, None)
    b_bad = ModeBOutput(0, 0, 0, cov, seg_confidence=0.1,
                        shoreline_length=10.0, success=True)
    for _ in range(5):
        mode = sw3.update(a_bad, b_bad)
    check("fall to C when A and B bad", mode == Mode.C)

    # from C, good B → recover to B (hysteresis: need b_good not just b_ok)
    b_good = ModeBOutput(0, 0, 0, cov, seg_confidence=0.7,
                         shoreline_length=60.0, success=True)
    for _ in range(5):
        mode = sw3.update(a_bad, b_good)
    check("recover C->B with good B", mode == Mode.B)


# ======================================================================
# Test 7: Evaluation metrics
# ======================================================================

def test_metrics() -> None:
    logger.info("=" * 60)
    logger.info("TEST 7: Evaluation metrics")
    logger.info("=" * 60)

    N = 100
    gt = np.zeros((N, 3))
    gt[:, 0] = np.arange(N) * 1.0
    gt[:, 1] = 50.0

    pred = gt.copy()
    pred[:, 0] += 2.0  # constant 2m error in x

    ate = compute_ate(pred, gt)
    check("ATE = 2.0", abs(ate - 2.0) < 0.01, f"ate={ate:.4f}")

    rpe = compute_rpe(pred, gt)
    check("RPE ≈ 0 (constant offset)", rpe < 0.01, f"rpe={rpe:.4f}")

    nt_mask = np.zeros(N, dtype=bool)
    nt_mask[30:60] = True

    ate_nt = compute_ate_nt(pred, gt, nt_mask)
    check("ATE-NT = 2.0", abs(ate_nt - 2.0) < 0.01, f"ate_nt={ate_nt:.4f}")

    max_e = compute_max_e_nt(pred, gt, nt_mask)
    check("MaxE-NT = 2.0", abs(max_e - 2.0) < 0.01, f"max_e={max_e:.4f}")

    # recovery: error drops to 0 right after NT zone
    pred2 = gt.copy()
    pred2[30:60, 0] += 10.0  # large error in NT zone only
    recov = compute_recov_e(pred2, gt, nt_mask, normal_thresh=5.0)
    check("RecovE = 1 (instant recovery)", recov == 1, f"recov={recov}")

    # ANEES with perfectly calibrated covariance
    errors = np.random.randn(200, 3)
    covs = np.tile(np.eye(3), (200, 1, 1))
    anees = compute_anees(errors, covs)
    check("ANEES ≈ 3 (calibrated)", abs(anees - 3.0) < 1.0,
          f"anees={anees:.2f}")


# ======================================================================
# Test 8: End-to-end Mode-B → ESKF mini-sequence
# ======================================================================

def test_end_to_end_sequence() -> None:
    logger.info("=" * 60)
    logger.info("TEST 8: End-to-end Mode-B + ESKF sequence")
    logger.info("=" * 60)

    mask = make_l_corner_mask(500)
    sdf = SDFField.from_mask(mask)

    b_cfg = ModeBConfig(
        grid_search_range=12.0,
        grid_search_step=2.0,
        grid_search_angle_range=0.12,
        grid_search_angle_step=0.02,
        gn_max_iters=50,
        gn_convergence_eps=1e-8,
        huber_delta=5.0,
    )
    aligner = ContourAligner(b_cfg)
    unc_est = UncertaintyEstimator(b_cfg)

    kf_cfg = ESKFConfig(
        init_pos_std=10.0,
        init_theta_std=0.1,
        process_noise_xy=0.3,
        process_noise_theta=0.005,
    )
    kf = ESKF(kf_cfg)

    # L-corner contour in world frame
    horiz = np.column_stack([np.linspace(250, 380, 80), np.full(80, 250.0)])
    vert = np.column_stack([np.full(80, 250.0), np.linspace(250, 380, 80)])
    world_shore = np.vstack([horiz, vert])

    # simulate drone drifting along the corner
    gt_trajectory = []
    for i in range(10):
        gt_trajectory.append(np.array([250.0 + i * 0.5, 250.0 + i * 0.3, 0.01 * i]))
    gt_trajectory = np.array(gt_trajectory)

    # initialize ESKF with noisy first pose
    kf.reset(
        gt_trajectory[0] + np.array([3.0, 3.0, 0.02]),
        np.diag([100.0, 100.0, 0.04]),
    )

    estimates = []
    for i, gt_pose in enumerate(gt_trajectory):
        if i > 0:
            kf.predict(0.1)

        local_pts = world_to_local(world_shore, *gt_pose)
        result = aligner.align(local_pts, sdf, kf.state)

        if result.converged:
            cov_result = unc_est.compute(result.pose, local_pts, sdf)
            kf.update(result.pose, cov_result.covariance)

        estimates.append(kf.state.copy())

    estimates = np.array(estimates)
    final_err = np.linalg.norm(estimates[-1, :2] - gt_trajectory[-1, :2])
    ate = compute_ate(estimates, gt_trajectory)

    check("e2e: ATE < 5 px", ate < 5.0, f"ate={ate:.3f}")
    check("e2e: final error < 3 px", final_err < 3.0,
          f"err={final_err:.3f}")

    logger.info("  Per-frame errors:")
    for i in range(len(gt_trajectory)):
        e = np.linalg.norm(estimates[i, :2] - gt_trajectory[i, :2])
        logger.info("    frame %d: err=%.3f", i, e)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    tests = [
        test_sdf_construction,
        test_alignment_straight,
        test_alignment_l_corner,
        test_alignment_curved,
        test_eskf,
        test_mode_switcher,
        test_metrics,
        test_end_to_end_sequence,
    ]

    for fn in tests:
        try:
            fn()
        except Exception:
            logger.error("EXCEPTION in %s:\n%s", fn.__name__,
                         traceback.format_exc())
            global _FAIL
            _FAIL += 1

    logger.info("=" * 60)
    logger.info("SUMMARY:  %d passed,  %d failed", _PASS, _FAIL)
    logger.info("=" * 60)

    sys.exit(1 if _FAIL > 0 else 0)


if __name__ == "__main__":
    main()
