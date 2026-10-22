"""Mimic hand-model adapter for the official BrainCo Revo2 MJCF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mimic_retargeter_lab.hand_models.robot_hand_base import RobotHandModel
from mimic_retargeter_lab.types import Chirality, HandLandmark, MujocoLandmark

from boost.control.hand_mppi.revo2 import (
    BRAINCO_CHANNELS,
    REVO2_COMMAND_SUFFIXES,
    REVO2_JOINT_SUFFIXES,
)


_TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class BrainCoRevo2HandModel(RobotHandModel):
    """Expose one official Revo2 hand MJCF as Mimic's six-DoF robot model.

    The upstream MJCF has eleven scalar joints and no actuators. Revo2 exposes
    six command channels; distal joints are derived from their proximal command
    using the same ratios as the combined BOOST simulator model.
    """

    def __init__(
        self,
        robot_hand_base_path: Path,
        chirality: Chirality,
        distal_couplings: dict[str, dict[str, object]],
        mjx_device: Any | None = None,
    ) -> None:
        super().__init__(Path(robot_hand_base_path), chirality)
        self.side = chirality.value
        self.hand_model_path = Path(robot_hand_base_path) / f"revo2_{self.side}.xml"
        if not self.hand_model_path.is_file():
            raise FileNotFoundError(
                f"official Revo2 MJCF not found: {self.hand_model_path}; "
                "initialize external/brainco-description"
            )

        self._command_joint_names = [
            f"{self.side}_{suffix}" for suffix in REVO2_COMMAND_SUFFIXES
        ]
        self._qpos_joint_names = [
            f"{self.side}_{suffix}" for suffix in REVO2_JOINT_SUFFIXES
        ]
        self._distal_couplings = distal_couplings
        self.num_fingertips = 5
        self.num_actuated_dofs = len(self._command_joint_names)
        self.num_qpos_dofs = len(self._qpos_joint_names)

        prefix = self.side
        self._landmark_config = {
            HandLandmark.ARM_ATTACHMENT: MujocoLandmark(
                name=f"{prefix}_hand_base_link", object_type="body"
            ),
            HandLandmark.WRIST: MujocoLandmark(
                name=f"{prefix}_hand_base_link", object_type="body"
            ),
            HandLandmark.PALM: MujocoLandmark(
                name=f"{prefix}_hand_base_link", object_type="body"
            ),
            HandLandmark.THUMB_BASE: MujocoLandmark(
                name=f"{prefix}_thumb_metacarpal_joint", object_type="joint"
            ),
            HandLandmark.INDEX_BASE: MujocoLandmark(
                name=f"{prefix}_index_proximal_joint", object_type="joint"
            ),
            HandLandmark.MIDDLE_BASE: MujocoLandmark(
                name=f"{prefix}_middle_proximal_joint", object_type="joint"
            ),
            HandLandmark.RING_BASE: MujocoLandmark(
                name=f"{prefix}_ring_proximal_joint", object_type="joint"
            ),
            HandLandmark.PINKY_BASE: MujocoLandmark(
                name=f"{prefix}_pinky_proximal_joint", object_type="joint"
            ),
        }
        for finger in _TIP_NAMES:
            self._landmark_config[getattr(HandLandmark, f"{finger.upper()}_DP")] = (
                MujocoLandmark(
                    name=f"{prefix}_{finger}_distal_link", object_type="body"
                )
            )
            self._landmark_config[getattr(HandLandmark, f"{finger.upper()}_TIP")] = (
                MujocoLandmark(
                    name=f"{prefix}_{finger}_tip_link", object_type="body"
                )
            )

        self.joint_map = self.compute_joint_map()
        self.create_mjx_kinematic_model(device=mjx_device)

    def get_qpos_joint_names(self) -> list[str]:
        return list(self._qpos_joint_names)

    def get_actuated_joint_names(self) -> list[str]:
        return list(self._command_joint_names)

    def compute_joint_map(self) -> np.ndarray:
        joint_map = np.zeros(
            (self.num_qpos_dofs, self.num_actuated_dofs), dtype=np.float32
        )
        qpos_index = {name: index for index, name in enumerate(self._qpos_joint_names)}
        command_index = {
            name: index for index, name in enumerate(self._command_joint_names)
        }

        for joint_name, channel_index in command_index.items():
            joint_map[qpos_index[joint_name], channel_index] = 1.0

        for coupling in self._distal_couplings.values():
            parent = f"{self.side}_{coupling['parent']}"
            child = f"{self.side}_{coupling['joint']}"
            ratio = float(coupling["multiplier"])
            if parent not in command_index or child not in qpos_index or ratio <= 0:
                raise ValueError(f"invalid Revo2 distal coupling: {coupling}")
            # MJCF equality in BOOST is q_proximal = ratio * q_distal.
            joint_map[qpos_index[child], command_index[parent]] = 1.0 / ratio

        if np.any(np.count_nonzero(joint_map, axis=1) != 1):
            raise ValueError("Revo2 qpos-to-command map must define all 11 joints")
        if len(BRAINCO_CHANNELS) != self.num_actuated_dofs:
            raise ValueError("BrainCo channel order and Revo2 command DoF count differ")
        return joint_map

