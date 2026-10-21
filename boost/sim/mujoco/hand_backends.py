"""REVO2 joint/command mapping, ported from PATH-G1 sonic_runtime.

Original source: sonic_runtime/gear_sonic/utils/mujoco_sim/hand_backends.py
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEX3_BACKEND = "dex3"
REVO2_BACKEND = "brainco_revo2"

REVO2_COMMAND_SUFFIXES = (
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)

# Keep actuated joints first.  This lets the DDS state publisher expose the
# first six physical states without confusing them with the five mimic joints.
REVO2_JOINT_SUFFIXES = REVO2_COMMAND_SUFFIXES + (
    "thumb_distal_joint",
    "index_distal_joint",
    "middle_distal_joint",
    "ring_distal_joint",
    "pinky_distal_joint",
)


@dataclass(frozen=True)
class Revo2CommandAdapter:
    """Convert BrainCo normalized commands into REVO2 joint targets.

    q=0 is fully open and q=1 is the configured close limit.  The default
    binary close pose deliberately limits thumb opposition to 70 percent;
    this avoids the thumb driving through the fingers during initial tuning.
    """

    q_min: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q_max: tuple[float, ...] = (1.51, 1.04, 1.46, 1.46, 1.46, 1.46)
    binary_open: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    binary_close: tuple[float, ...] = (0.70, 1.0, 1.0, 1.0, 1.0, 1.0)
    legacy_close_threshold: float = 0.10

    def _six(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
        if array.size != 6:
            raise ValueError(f"REVO2 expects 6 commands, got {array.size}")
        return array

    def clamp_normalized(self, values: Iterable[float]) -> np.ndarray:
        return np.clip(self._six(values), 0.0, 1.0)

    def normalized_to_actuated(self, values: Iterable[float]) -> np.ndarray:
        normalized = self.clamp_normalized(values)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        return q_min + normalized * (q_max - q_min)

    def actuated_to_normalized(self, values: Iterable[float]) -> np.ndarray:
        physical = self._six(values)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        return np.clip((physical - q_min) / np.maximum(q_max - q_min, 1e-9), 0.0, 1.0)

    def normalized_velocity(self, physical_velocity: Iterable[float]) -> np.ndarray:
        velocity = self._six(physical_velocity)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        return velocity / np.maximum(q_max - q_min, 1e-9)

    def normalized_to_all_joints(self, values: Iterable[float]) -> np.ndarray:
        actuated = self.normalized_to_actuated(values)
        thumb_meta, thumb_prox, index, middle, ring, pinky = actuated
        # These ratios match the equality constraints in the supplied REVO2
        # MJCF.  Distal joints are included for kinematics-only simulation;
        # dynamic simulation lets MuJoCo enforce the same constraints.
        distal = np.asarray(
            [
                min(1.0472, 1.155 * thumb_prox),
                min(1.6930, 1.000 * index),
                min(1.6930, 1.155 * middle),
                min(1.6930, 1.155 * ring),
                min(1.6930, 1.155 * pinky),
            ],
            dtype=np.float64,
        )
        return np.concatenate(([thumb_meta, thumb_prox, index, middle, ring, pinky], distal))

    def binary_target(self, closed: bool) -> np.ndarray:
        pose = self.binary_close if closed else self.binary_open
        return self.clamp_normalized(pose)

    def legacy_dex3_is_closed(self, dex3_values: Iterable[float]) -> bool:
        values = np.asarray(tuple(dex3_values), dtype=np.float64).reshape(-1)
        if values.size != 7:
            raise ValueError(f"legacy DEX3 input expects 7 values, got {values.size}")
        # This is the same discriminator used by the real LCM action bridge.
        return bool(abs(values[3]) > self.legacy_close_threshold)

    def legacy_dex3_to_normalized(self, dex3_values: Iterable[float]) -> np.ndarray:
        return self.binary_target(self.legacy_dex3_is_closed(dex3_values))
