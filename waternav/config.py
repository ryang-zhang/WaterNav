"""All configuration parameters in one place, as plain dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Mode A – feature matching
# ---------------------------------------------------------------------------
@dataclass
class ModeAConfig:
    matcher_type: str = "loftr"  # "loftr" | "superglue"
    confidence_thresh: float = 0.5
    min_inliers: int = 20
    reproj_err_thresh: float = 8.0  # pixels


# ---------------------------------------------------------------------------
# Mode B – shoreline contour SDF alignment (core innovation)
# ---------------------------------------------------------------------------
@dataclass
class ModeBConfig:
    # segmentation
    seg_model: str = "sam2"  # "sam2" | "custom"
    seg_confidence_thresh: float = 0.6
    morphology_kernel_size: int = 5

    # SDF
    sdf_resolution: float = 0.5  # meters per pixel

    # alignment optimisation
    grid_search_range: float = 50.0  # meters, half-width of coarse grid
    grid_search_step: float = 5.0  # meters
    grid_search_angle_range: float = 0.3  # radians
    grid_search_angle_step: float = 0.05  # radians
    gn_max_iters: int = 30
    gn_convergence_eps: float = 1e-6
    huber_delta: float = 2.0  # SDF units (pixels)

    # uncertainty
    sigma_sdf: float = 1.0  # assumed SDF noise std


# ---------------------------------------------------------------------------
# Mode C – constant-velocity propagation with map constraint
# ---------------------------------------------------------------------------
@dataclass
class ModeCConfig:
    velocity_decay: float = 0.98  # per-frame multiplicative decay
    process_noise_xy: float = 2.0  # m/s²
    process_noise_theta: float = 0.05  # rad/s


# ---------------------------------------------------------------------------
# ESKF (Error-State Kalman Filter)
# ---------------------------------------------------------------------------
@dataclass
class ESKFConfig:
    init_pos_std: float = 10.0  # meters
    init_theta_std: float = 0.2  # radians
    process_noise_xy: float = 1.0
    process_noise_theta: float = 0.02


# ---------------------------------------------------------------------------
# Mode switching
# ---------------------------------------------------------------------------
@dataclass
class ModeSwitchConfig:
    # A -> B thresholds (drop below these to leave A)
    a_inlier_ratio_low: float = 0.15
    a_reproj_err_high: float = 15.0  # pixels

    # B -> A thresholds (exceed these to re-enter A)  – hysteresis
    a_inlier_ratio_high: float = 0.25
    a_reproj_err_low: float = 10.0

    # B -> C thresholds (drop below these to leave B)
    b_seg_confidence_low: float = 0.3
    b_shoreline_length_low: float = 30  # pixels

    # C -> B thresholds (exceed these to re-enter B)  – hysteresis
    b_seg_confidence_high: float = 0.5
    b_shoreline_length_high: float = 50

    min_hold_frames: int = 5  # minimum frames before allowing mode switch


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    mode_a: ModeAConfig = field(default_factory=ModeAConfig)
    mode_b: ModeBConfig = field(default_factory=ModeBConfig)
    mode_c: ModeCConfig = field(default_factory=ModeCConfig)
    eskf: ESKFConfig = field(default_factory=ESKFConfig)
    mode_switch: ModeSwitchConfig = field(default_factory=ModeSwitchConfig)

    log_dir: str = "debug_file"
    device: str = "cuda"  # "cuda" | "cpu"
    image_size: tuple[int, int] = (480, 640)  # H, W for processing
