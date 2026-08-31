"""Joint-index layouts shared by direct PATH control and legacy replay tools."""

from __future__ import annotations


# q29 order projected to the legacy planner's upper-body vector.  Keeping the
# layout outside the SMPL/SONIC replay server prevents direct control from
# importing that retired execution path.
UPPER_BODY_PLANNER_INDICES: list[int] = [
    12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
