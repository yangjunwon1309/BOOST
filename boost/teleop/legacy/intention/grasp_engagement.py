"""Grasp-phase detection shared by the teleoperation bridge and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GraspEngagementDetector:
    """Detect a completed grasp, with a command-delay fallback without jaw FK.

    A measured non-empty jaw gap must stall before it is considered a grasp.
    The current bridge has no Dex1 jaw-state stream, so ``jaw_gap_m=None``
    uses the nominal close travel delay instead and preserves the reach phase
    while the fingers are closing.
    """

    stall_gap_min_m: float = 0.008
    stall_rate_max_mps: float = 0.004
    engage_ticks: int = 5
    release_ticks: int = 3
    # No measured Dex1 jaw state reaches the live bridge.  Keep belief
    # position authority through a deliberate, continuous 1.5 s close hold
    # before handing position back to VR for transport.
    command_delay_sec: float = 1.5
    engaged: bool = False
    stalled_ticks: int = 0
    free_ticks: int = 0
    close_elapsed_sec: float = 0.0
    previous_gap_m: float | None = None

    def update(
        self,
        *,
        commanded_closed: bool,
        jaw_gap_m: float | None,
        dt: float,
    ) -> bool:
        step = max(0.0, float(dt))
        if not commanded_closed:
            self.engaged = False
            self.stalled_ticks = 0
            self.free_ticks = 0
            self.close_elapsed_sec = 0.0
            self.previous_gap_m = None
            return False

        self.close_elapsed_sec += step
        finite_gap = jaw_gap_m is not None and math.isfinite(float(jaw_gap_m))
        if not finite_gap:
            self.engaged = self.close_elapsed_sec >= max(0.0, self.command_delay_sec)
            return self.engaged

        gap = float(jaw_gap_m)
        has_previous_gap = self.previous_gap_m is not None
        rate = float("inf")
        if has_previous_gap and step > 0.0:
            rate = abs(gap - self.previous_gap_m) / step
        self.previous_gap_m = gap
        # The first jaw observation has no motion history and therefore cannot
        # prove a stall; treating it as stationary would engage one tick early.
        stalled = (
            has_previous_gap
            and gap > self.stall_gap_min_m
            and rate < self.stall_rate_max_mps
        )
        if stalled:
            self.stalled_ticks += 1
            self.free_ticks = 0
        else:
            self.free_ticks += 1
            if self.free_ticks >= self.release_ticks:
                self.stalled_ticks = 0
        if self.stalled_ticks >= self.engage_ticks:
            self.engaged = True
        return self.engaged
