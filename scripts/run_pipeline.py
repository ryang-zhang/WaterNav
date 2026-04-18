#!/usr/bin/env python3
"""Main entry point: run WaterNav pipeline on a data sequence."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waternav.config import Config
from waternav.core.pipeline import WaterNavPipeline
from waternav.data.dataset import WaterNavDataset
from waternav.data.satellite import SatelliteMap
from waternav.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WaterNav pipeline")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Path to data sequence directory")
    parser.add_argument("--satellite", type=str, required=True,
                        help="Path to satellite map image")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Satellite map resolution (m/px)")
    parser.add_argument("--output", type=str, default="experiments",
                        help="Output directory for results")
    args = parser.parse_args()

    cfg = Config()
    logger = setup_logger(log_dir=cfg.log_dir)
    logger.info("Config: %s", cfg)

    dataset = WaterNavDataset(args.sequence)
    logger.info("Loaded %d frames from %s", len(dataset), args.sequence)

    sat_map = SatelliteMap.load(args.satellite, resolution=args.resolution)
    logger.info("Satellite map shape: %s", sat_map.shape)

    pipeline = WaterNavPipeline(cfg, sat_map)
    pipeline.build_sdf()

    first = dataset[0]
    pipeline.initialise(first.gt_x, first.gt_y, first.gt_theta)

    results = []
    t0 = time.perf_counter()
    for i in range(len(dataset)):
        frame = dataset[i]
        img = cv2.imread(frame.image_path)
        if img is None:
            logger.warning("Cannot read frame %d: %s", i, frame.image_path)
            continue
        estimate = pipeline.process_frame(img, frame.timestamp)
        results.append(estimate)
        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d frames", i + 1, len(dataset))

    elapsed = time.perf_counter() - t0
    logger.info("Done. %d frames in %.1f s (%.1f fps)",
                len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = np.array([[r.x, r.y, r.theta] for r in results])
    np.savetxt(out_dir / "pred_poses.txt", poses, fmt="%.6f")
    logger.info("Saved predictions to %s", out_dir / "pred_poses.txt")


if __name__ == "__main__":
    main()
