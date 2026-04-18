"""Logging utility that writes to debug_file/<timestamp>/ directories."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


_initialised = False


def setup_logger(
    name: str = "waternav",
    log_dir: str = "debug_file",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or return existing) logger that writes to console + .log file.

    The .log file is placed under ``<log_dir>/<YYYY-MM-DD_HH-MM-SS>/``,
    one folder per run so that outputs never collide.
    """
    global _initialised
    logger = logging.getLogger(name)

    if _initialised:
        return logger

    logger.setLevel(level)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"{name}.log"

    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _initialised = True
    logger.info("Log directory: %s", run_dir)
    return logger
