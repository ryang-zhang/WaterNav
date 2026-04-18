"""Mode switching logic with hysteresis and minimum-hold-frame constraints."""

from __future__ import annotations

import logging

from waternav.config import ModeSwitchConfig
from waternav.core import Mode, ModeAOutput, ModeBOutput

logger = logging.getLogger("waternav")


class ModeSwitcher:
    """Decides the active localisation mode (A / B / C) each frame."""

    def __init__(self, cfg: ModeSwitchConfig) -> None:
        self.cfg = cfg
        self.current_mode = Mode.A
        self._frames_in_current = 0

    # ------------------------------------------------------------------
    def update(
        self,
        mode_a_output: ModeAOutput | None,
        mode_b_output: ModeBOutput | None,
    ) -> Mode:
        """Evaluate switching criteria and return the mode for this frame.

        Both outputs can be ``None`` when the respective module has not been
        run (e.g. mode-B is skipped while in mode-A).
        """
        self._frames_in_current += 1

        if self._frames_in_current < self.cfg.min_hold_frames:
            return self.current_mode

        prev = self.current_mode
        next_mode = self._decide(mode_a_output, mode_b_output)
        if next_mode != prev:
            logger.info(
                "Mode switch %s -> %s at hold=%d",
                prev.name,
                next_mode.name,
                self._frames_in_current,
            )
            self._frames_in_current = 0
        self.current_mode = next_mode
        return self.current_mode

    # ------------------------------------------------------------------
    def _decide(
        self,
        a: ModeAOutput | None,
        b: ModeBOutput | None,
    ) -> Mode:
        cfg = self.cfg

        a_good = (
            a is not None
            and a.success
            and a.inlier_ratio >= cfg.a_inlier_ratio_high
            and a.reproj_error <= cfg.a_reproj_err_low
        )
        a_ok = (
            a is not None
            and a.success
            and a.inlier_ratio >= cfg.a_inlier_ratio_low
            and a.reproj_error <= cfg.a_reproj_err_high
        )
        b_good = (
            b is not None
            and b.success
            and b.seg_confidence >= cfg.b_seg_confidence_high
            and b.shoreline_length >= cfg.b_shoreline_length_high
        )
        b_ok = (
            b is not None
            and b.success
            and b.seg_confidence >= cfg.b_seg_confidence_low
            and b.shoreline_length >= cfg.b_shoreline_length_low
        )

        if self.current_mode == Mode.A:
            if a_ok:
                return Mode.A
            if b_ok:
                return Mode.B
            return Mode.C

        if self.current_mode == Mode.B:
            if a_good:
                return Mode.A
            if b_ok:
                return Mode.B
            return Mode.C

        # current_mode == Mode.C
        if a_good:
            return Mode.A
        if b_good:
            return Mode.B
        return Mode.C
