"""Canonical BrainCo Revo2 command/state mapping shared by policy and MuJoCo.

The six DDS channels use normalized positions in the order Thumb, Thumb_aux,
Index, Middle, Ring, Pinky. The six commanded MJCF joints are ordered to match
that contract; the distal joints are coupled by the MuJoCo model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


BRAINCO_CHANNELS = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
BRAINCO_SIDES = ("left", "right")
BRAINCO_COMMAND_TOPICS = {
    side: f"rt/brainco/{side}/cmd" for side in BRAINCO_SIDES
}
BRAINCO_STATE_TOPICS = {
    side: f"rt/brainco/{side}/state" for side in BRAINCO_SIDES
}

REVO2_COMMAND_SUFFIXES = (
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
REVO2_JOINT_SUFFIXES = REVO2_COMMAND_SUFFIXES + (
    "thumb_distal_joint",
    "index_distal_joint",
    "middle_distal_joint",
    "ring_distal_joint",
    "pinky_distal_joint",
)


@dataclass(frozen=True)
class BrainCoRevo2Adapter:
    """Convert Mimic/Revo2 joint positions and DDS channels.

    ``q=0`` is open and ``q=1`` is the configured close limit. The defaults
    match ``boost/sim/mujoco/hand_backends.py`` and the checked-in Revo2 MJCF.
    """

    q_min: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q_max: tuple[float, ...] = (1.57, 1.03, 1.41, 1.41, 1.41, 1.41)
    binary_open: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    binary_close: tuple[float, ...] = (0.70, 1.0, 1.0, 1.0, 1.0, 1.0)
    legacy_close_threshold: float = 0.10

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "BrainCoRevo2Adapter":
        """Build the adapter from the checked-in BrainCo YAML mapping."""
        return cls(
            q_min=tuple(config.get("q_min", cls.q_min)),
            q_max=tuple(config.get("q_max", cls.q_max)),
            binary_open=tuple(config.get("binary_open", cls.binary_open)),
            binary_close=tuple(config.get("binary_close", cls.binary_close)),
            legacy_close_threshold=float(
                config.get("legacy_close_threshold", cls.legacy_close_threshold)
            ),
        )

    def _six(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
        if array.size != len(BRAINCO_CHANNELS):
            raise ValueError(f"Revo2 expects 6 values, got {array.size}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Revo2 command/state values must all be finite")
        return array

    def normalized_command(self, values: Iterable[float]) -> np.ndarray:
        """Validate and clamp the six BrainCo DDS position targets to [0, 1]."""
        return np.clip(self._six(values), 0.0, 1.0)

    # Backward-compatible spelling used by the existing MuJoCo DDS bridge.
    def clamp_normalized(self, values: Iterable[float]) -> np.ndarray:
        return self.normalized_command(values)

    def normalized_to_actuated(self, values: Iterable[float]) -> np.ndarray:
        normalized = self.normalized_command(values)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        if q_min.shape != (6,) or q_max.shape != (6,) or np.any(q_max <= q_min):
            raise ValueError("Revo2 physical joint limits must contain six increasing ranges")
        return q_min + normalized * (q_max - q_min)

    def actuated_to_normalized(self, values: Iterable[float]) -> np.ndarray:
        physical = self._six(values)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        if q_min.shape != (6,) or q_max.shape != (6,) or np.any(q_max <= q_min):
            raise ValueError("Revo2 physical joint limits must contain six increasing ranges")
        return np.clip((physical - q_min) / (q_max - q_min), 0.0, 1.0)

    def actuated_velocity_to_normalized(self, values: Iterable[float]) -> np.ndarray:
        velocity = self._six(values)
        q_min = np.asarray(self.q_min, dtype=np.float64)
        q_max = np.asarray(self.q_max, dtype=np.float64)
        if q_min.shape != (6,) or q_max.shape != (6,) or np.any(q_max <= q_min):
            raise ValueError("Revo2 physical joint limits must contain six increasing ranges")
        return velocity / (q_max - q_min)

    # Kept for existing callers. It accepts physical velocity and returns its
    # normalized channel rate, matching the current MuJoCo state publisher.
    def normalized_velocity(self, values: Iterable[float]) -> np.ndarray:
        return self.actuated_velocity_to_normalized(values)

    def mimicked_joint_targets(self, values: Iterable[float]) -> np.ndarray:
        """Return all 11 MJCF joint targets from the six commanded values."""
        thumb_meta, thumb_prox, index, middle, ring, pinky = self.normalized_to_actuated(values)
        # The checked-in MJCF defines its equality as q_prox = ratio*q_distal
        # (joint1=proximal, joint2=distal), so solve for the distal joint here.
        distal = np.asarray(
            [
                min(1.03, thumb_prox / 1.155),
                min(1.63, index),
                min(1.63, middle / 1.155),
                min(1.63, ring / 1.155),
                min(1.63, pinky / 1.155),
            ],
            dtype=np.float64,
        )
        return np.concatenate(([thumb_meta, thumb_prox, index, middle, ring, pinky], distal))

    def normalized_to_all_joints(self, values: Iterable[float]) -> np.ndarray:
        return self.mimicked_joint_targets(values)

    def joint_targets_to_normalized(self, targets: Mapping[str, float]) -> np.ndarray:
        """Map six Mimic joint targets in radians into BrainCo channel order."""
        missing = [name for name in REVO2_COMMAND_SUFFIXES if name not in targets]
        if missing:
            raise ValueError(f"Mimic target is missing Revo2 joints: {', '.join(missing)}")
        physical = [targets[name] for name in REVO2_COMMAND_SUFFIXES]
        return self.actuated_to_normalized(physical)

    def binary_target(self, closed: bool) -> np.ndarray:
        pose = self.binary_close if closed else self.binary_open
        return self.normalized_command(pose)

    def legacy_dex3_is_closed(self, dex3_values: Iterable[float]) -> bool:
        values = np.asarray(tuple(dex3_values), dtype=np.float64).reshape(-1)
        if values.size != 7:
            raise ValueError(f"legacy DEX3 input expects 7 values, got {values.size}")
        if not np.all(np.isfinite(values)):
            return False
        return bool(abs(values[3]) > self.legacy_close_threshold)

    def legacy_dex3_to_normalized(self, dex3_values: Iterable[float]) -> np.ndarray:
        return self.binary_target(self.legacy_dex3_is_closed(dex3_values))


# Backward-compatible name used by the existing simulation DDS bridge.
Revo2CommandAdapter = BrainCoRevo2Adapter
