#!/usr/bin/env python3
"""Evaluate predicted poses against ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waternav.evaluation.metrics import (
    compute_anees,
    compute_ate,
    compute_ate_nt,
    compute_max_e_nt,
    compute_recov_e,
    compute_rpe,
)
from waternav.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WaterNav results")
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to predicted poses (Nx3 txt)")
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to ground truth poses (Nx3 txt)")
    parser.add_argument("--nt-mask", type=str, default=None,
                        help="Path to no-texture boolean mask (N txt, 0/1)")
    args = parser.parse_args()

    logger = setup_logger(name="evaluate")

    pred = np.loadtxt(args.pred)
    gt = np.loadtxt(args.gt)
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"

    logger.info("ATE:    %.3f m", compute_ate(pred, gt))
    logger.info("RPE:    %.3f m", compute_rpe(pred, gt))

    if args.nt_mask is not None:
        nt_mask = np.loadtxt(args.nt_mask).astype(bool)
        logger.info("ATE-NT: %.3f m", compute_ate_nt(pred, gt, nt_mask))
        logger.info("MaxE-NT: %.3f m", compute_max_e_nt(pred, gt, nt_mask))
        logger.info("RecovE: %d frames", compute_recov_e(pred, gt, nt_mask))


if __name__ == "__main__":
    main()
