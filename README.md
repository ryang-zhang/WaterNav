# WaterNav

**UAV–satellite cross-view continuous localization over textureless water regions.**

When a UAV flies over open water, traditional feature-matching breaks down due to the lack of visual texture. WaterNav addresses this with a **degradation-aware three-mode framework** that inserts a novel *shoreline contour SDF alignment* layer (Mode B) between feature matching (Mode A) and dead-reckoning propagation (Mode C). Partial water–land boundary geometry is exploited to maintain localization even when no visual features are available.

---

## System Architecture

```
UAV image sequence  +  georeferenced satellite orthophoto
             │
        WaterNavPipeline
             │
      ModeSwitcher  (hysteresis thresholds + min-hold frames)
      ┌──────┼──────┐
      ▼      ▼      ▼
   Mode A  Mode B  Mode C
  feature  shore   const-
  match    SDF     velocity
 (reuse)  (novel)  (bridge)
      └──────┼──────┘
             ▼
  Degradation-aware ESKF
  (anisotropic covariance fusion)
             │
             ▼
    Pose (x, y, θ)  +  uncertainty
```

| Mode | Scenario | Method | Uncertainty |
|------|----------|--------|-------------|
| A | High-texture area | LoFTR / SuperGlue cross-view matching | Small, near-isotropic |
| B | Transition zone (visible shoreline) | Shoreline contour ↔ SDF alignment | Medium, **anisotropic** (Hessian-based) |
| C | Featureless open water | Constant-velocity model + map truncation | Large, growing |

---

## Project Structure

```
WaterNav/
├── waternav/                      # Main Python package
│   ├── config.py                  # All parameters (dataclasses, single file)
│   ├── core/
│   │   ├── __init__.py            # Shared types: Mode, PoseEstimate, *Output
│   │   ├── pipeline.py            # Main controller: A/B/C + ESKF
│   │   ├── eskf.py                # Error-State Kalman Filter (3-DOF, NumPy)
│   │   └── mode_switcher.py       # Mode switching with hysteresis
│   ├── mode_a/
│   │   ├── matcher.py             # LoFTR / SuperGlue wrapper (interface stub)
│   │   └── pose_estimator.py      # Matches → pose + covariance (interface stub)
│   ├── mode_b/                    # ★ Core novel module (implemented)
│   │   ├── segmentation.py        # Water–land segmentation wrapper (interface stub)
│   │   ├── sdf.py                 # SDF build + query (scipy + NumPy)
│   │   ├── alignment.py           # Grid search + Gauss-Newton alignment
│   │   └── uncertainty.py         # Hessian → anisotropic covariance
│   ├── mode_c/
│   │   └── propagation.py         # Constant-velocity propagation + covariance inflation
│   ├── evaluation/
│   │   ├── metrics.py             # ATE / ATE-NT / MaxE-NT / RPE / RecovE / ANEES
│   │   └── visualization.py       # Plotting stubs (not yet implemented)
│   ├── data/
│   │   ├── dataset.py             # Dataset loader (custom / UAV-VisLoc format)
│   │   └── satellite.py           # Satellite map loader and coordinate handling
│   └── utils/
│       ├── geo.py                 # SE(2) helpers, rotation matrices, angle wrapping
│       └── logger.py              # Timestamped file logger
│
├── scripts/
│   ├── run_pipeline.py            # End-to-end pipeline entry point
│   ├── evaluate.py                # Compute and print evaluation metrics
│   └── test_synthetic.py          # Synthetic validation (43/43 tests pass)
│
├── environment.yml                # Conda environment definition
├── requirements.txt               # pip-compatible dependency list
└── pyproject.toml                 # Package build metadata
```

---

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | ✅ Done | All parameters in one dataclass file |
| `core/eskf.py` | ✅ Done | 3-DOF ESKF, predict/update, full anisotropic R |
| `core/mode_switcher.py` | ✅ Done | Hysteresis + min-hold-frames, A↔B↔C |
| `core/pipeline.py` | ✅ Done | Main controller, wires all modules |
| `mode_b/sdf.py` | ✅ Done | `scipy` distance transform, bilinear query, gradient cache |
| `mode_b/alignment.py` | ✅ Done | Coarse grid search + Gauss-Newton, Huber loss |
| `mode_b/uncertainty.py` | ✅ Done | H = JᵀJ → Σ_B = σ²H⁻¹, eigenvalue analysis |
| `mode_c/propagation.py` | ✅ Done | Constant-velocity extrapolation + covariance inflation |
| `evaluation/metrics.py` | ✅ Done | ATE / ATE-NT / MaxE-NT / RPE / RecovE / ANEES |
| `mode_a/matcher.py` | 🔧 Stub | Needs LoFTR or SuperGlue integration |
| `mode_a/pose_estimator.py` | 🔧 Stub | Needs RANSAC + least-squares pose estimation |
| `mode_b/segmentation.py` | 🔧 Stub | Needs SAM2 or custom segmentation model |
| `evaluation/visualization.py` | 🔧 Stub | Paper figure generation (hero, error ellipse, etc.) |

> **Note:** The end-to-end pipeline on real data is blocked by the three stubs above. All geometry, ESKF, and mode-switching logic is fully implemented and validated synthetically.

---

## Synthetic Validation Results

Run `python scripts/test_synthetic.py` — **43 / 43 tests pass**. No real data or trained models needed.

**SDF construction:** water-side negative, land-side positive, zero-crossing at shoreline; gradient direction correct; non-unit resolution supported.

**Straight shoreline alignment:** Hessian eigenvalues ≈ [0, 450, 1.5M]. Near-zero eigenvalue corresponds to along-shore direction (unobservable); large eigenvalue to cross-shore direction (strongly constrained). Confirms: *straight shoreline constrains 1 DOF*.

**L-shaped corner alignment:** Position error 0.7 px, heading error < 0.001 rad, converges in 18 iterations. Confirms: *L-corner constrains 2+ DOF*.

**Curved bank alignment:** Sinusoidal shoreline, position error 0.47 px, heading error < 0.001 rad, all Hessian eigenvalues > 1. Confirms: *complex curved shoreline provides near-full constraint*.

**ESKF:** Constant-velocity prediction, scalar update, anisotropic update — all correct.

**Mode switching:** Full A→B→C chain, hysteresis, min-hold-frames — all correct.

**End-to-end sequence:** 10-frame L-corner Mode-B + ESKF, ATE = 0.98 px.

---

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate waternav
```

### pip

```bash
pip install -r requirements.txt
```

Core dependencies: Python 3.10, PyTorch (CUDA 11.8), NumPy, SciPy, OpenCV, Matplotlib, Kornia, segment-anything-2.

---

## Usage

```bash
# Synthetic validation — no data or models required
python scripts/test_synthetic.py

# Full pipeline — requires a data sequence and satellite map
python scripts/run_pipeline.py \
    --sequence data/your_sequence \
    --satellite data/your_sat_map.png \
    --resolution 0.5

# Evaluate predictions
python scripts/evaluate.py \
    --pred experiments/pred_poses.txt \
    --gt data/your_sequence/gt_poses.txt
```

### Data Format

**Sequence directory:**
```
your_sequence/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── gt_poses.txt   # N×3 text file: x  y  theta (one row per frame)
```

**Satellite map:** a single georeferenced orthophoto image (PNG/JPG). Supply `--resolution` in metres per pixel.

---

## Design Principles

- **Pure Python:** SDF via `scipy`, Gauss-Newton with only 3 DOF via handwritten NumPy — no C++ extensions required.
- **Single-file configuration:** all parameters in `config.py` as dataclasses; easy to override programmatically.
- **Correctness first:** all implemented modules are validated synthetically before real-data integration.

---

## Roadmap

1. Integrate a real segmentation model (SAM2) into `mode_b/segmentation.py`
2. Integrate a feature matcher (LoFTR) into `mode_a/matcher.py` and `pose_estimator.py`
3. Run the full pipeline on a real UAV sequence with RTK-GPS ground truth
4. Core ablation: A+B+C vs. A+C to quantify Mode-B gain
5. Implement visualization functions for paper figures

---

## Citation

If you use WaterNav in your research, please cite:

```bibtex
@misc{waternav2026,
  title   = {WaterNav: UAV--Satellite Cross-View Localization over Textureless Water},
  year    = {2026},
  note    = {\url{https://github.com/ryang-zhang/WaterNav}}
}
```

---

## License

[MIT](LICENSE)
