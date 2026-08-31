"""Controller-trigger to 7-DoF Dex1/G1 hand pose mapping."""

from __future__ import annotations

import numpy as np


class G1GripperInverseKinematicsSolver:
    """Map synthetic fingertip distances to the established hand targets."""

    def __init__(self, side: str) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot) -> None:
        del robot

    def __call__(self, finger_data: dict) -> np.ndarray:
        positions = np.asarray(
            [finger[:3, 3] for finger in finger_data["position"]], dtype=np.float64
        ).reshape(-1, 3)
        thumb = positions[4]
        grips = [
            float(np.clip(1.0 - np.linalg.norm(thumb - positions[index]), 0.0, 1.0))
            for index in (9, 14, 19, 24)
        ]
        grips = [0.0 if grip < 0.05 else grip for grip in grips]
        maximum = max(grips)
        if maximum == 0.0:
            return np.zeros(7)
        if grips[0] == maximum:
            closed = self._index_closed()
        elif grips[1] == maximum:
            closed = self._middle_closed()
        elif grips[2] == maximum:
            closed = self._ring_closed()
        else:
            closed = np.zeros(7)
        return maximum * closed

    def _with_side(self, values: np.ndarray) -> np.ndarray:
        return values if self.side == "L" else -values

    def _index_closed(self) -> np.ndarray:
        values = np.zeros(7)
        values[0] = -0.5 if self.side == "L" else 0.5
        values[1:3] = 0.7
        values[3:] = [-1.5, -1.5, -0.6, -1.5]
        return self._with_side(values)

    def _middle_closed(self) -> np.ndarray:
        values = np.zeros(7)
        values[1:3] = 0.7
        values[3:] = [-1.0, -1.5, -1.0, -1.5]
        return self._with_side(values)

    def _ring_closed(self) -> np.ndarray:
        values = np.zeros(7)
        values[0] = 0.5 if self.side == "L" else -0.5
        values[1:3] = 0.7
        values[3:] = [-0.6, -1.5, -1.5, -1.5]
        return self._with_side(values)
