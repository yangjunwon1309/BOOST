"""Sealed FlashSAC DreamWaQ and G1 actuator contracts.

This module deliberately supports one policy family: the 29-DoF lower-body
FlashSAC reference export. The earlier 102-D PPO/CENet deployment contract is
not accepted here; admitting it beside FlashSAC made it too easy to run an
artifact with a shape-compatible but semantically different observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SDK_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Isaac's 29-joint proprioceptive order. It is also the order of q/dq/action
# channels in the FlashSAC observation; only the policy head is lower-body.
POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
)
SPINE_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_NAMES = tuple(
    name for name in SDK_JOINT_NAMES if name not in set(LOWER_BODY_JOINT_NAMES + SPINE_JOINT_NAMES)
)

FLASH_SAC_ASSET_PROFILE = "g1_29dof_flashsac_reference"
FLASH_SAC_WAIST_CONTROL_ASSET_PROFILE = "g1_29dof_flashsac_reference_waist_control"
FLASH_SAC_OBSERVATION_LAYOUT = "g1_lower_body_reference_v1"
FLASH_SAC_MODE_OBSERVATION_LAYOUT = "g1_lower_body_reference_mode_v2"
WAIST_CONTROL_JOINT_NAMES = LOWER_BODY_JOINT_NAMES + SPINE_JOINT_NAMES

# Exact FlashSAC reference reset pose, in Unitree SDK order.
DEFAULT_Q_SDK = np.asarray(
    [
        -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

JOINT_LOWER_SDK = np.asarray(
    [
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.618, -0.52, -0.52,
        -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558,
        -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558,
    ],
    dtype=np.float32,
)
JOINT_UPPER_SDK = np.asarray(
    [
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
        2.618, 0.52, 0.52,
        2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558,
        2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558,
    ],
    dtype=np.float32,
)
JOINT_VELOCITY_LIMIT_SDK = np.asarray(
    [
        32, 20, 32, 20, 37, 37,
        32, 20, 32, 20, 37, 37,
        32, 37, 37,
        37, 37, 37, 37, 37, 22, 22,
        37, 37, 37, 37, 37, 22, 22,
    ],
    dtype=np.float32,
)

_OMEGA = 20.0 * math.pi
_K_7520_14 = 0.010177520 * _OMEGA**2
_D_7520_14 = 4.0 * 0.010177520 * _OMEGA
_K_7520_22 = 0.025101925 * _OMEGA**2
_D_7520_22 = 4.0 * 0.025101925 * _OMEGA
_K_5020 = 0.003609725 * _OMEGA**2
_D_5020 = 4.0 * 0.003609725 * _OMEGA
_K_4010 = 0.00425 * _OMEGA**2
_D_4010 = 4.0 * 0.00425 * _OMEGA

# Original Isaac reference actuator gains. Hardware calibration applies a
# reviewed multiplicative envelope; these vectors must never silently fall
# back to unrelated PPO gains.
KP_SDK = np.asarray(
    [
        _K_7520_14, _K_7520_22, _K_7520_14, _K_7520_22, 2.0 * _K_5020, 2.0 * _K_5020,
        _K_7520_14, _K_7520_22, _K_7520_14, _K_7520_22, 2.0 * _K_5020, 2.0 * _K_5020,
        1200.0, 1200.0, 1200.0,
        _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
        _K_5020, _K_5020, _K_5020, _K_5020, _K_5020, _K_4010, _K_4010,
    ],
    dtype=np.float32,
)
KD_SDK = np.asarray(
    [
        _D_7520_14, _D_7520_22, _D_7520_14, _D_7520_22, 2.0 * _D_5020, 2.0 * _D_5020,
        _D_7520_14, _D_7520_22, _D_7520_14, _D_7520_22, 2.0 * _D_5020, 2.0 * _D_5020,
        80.0, 80.0, 80.0,
        _D_5020, _D_5020, _D_5020, _D_5020, _D_5020, _D_4010, _D_4010,
        _D_5020, _D_5020, _D_5020, _D_5020, _D_5020, _D_4010, _D_4010,
    ],
    dtype=np.float32,
)

# Later FlashSAC runs intentionally trained the three waist joints with the
# physical deployment gains.  Both receipts are admitted, but only as exact
# profiles: arbitrary per-joint gain drift still invalidates the export.
DEPLOYMENT_MATCHED_KP_SDK = KP_SDK.copy()
DEPLOYMENT_MATCHED_KD_SDK = KD_SDK.copy()
DEPLOYMENT_MATCHED_KP_SDK[12:15] = 300.0
DEPLOYMENT_MATCHED_KD_SDK[12:15] = 3.0
# RB8 was trained with the production HOMIE arm-gain profile while controlling
# the three waist joints. Arms remain owned by the separate deployment overlay,
# but the exact receipt is admitted so their observed dynamics stay auditable.
WAIST_CONTROL_KP_SDK = DEPLOYMENT_MATCHED_KP_SDK.copy()
WAIST_CONTROL_KD_SDK = DEPLOYMENT_MATCHED_KD_SDK.copy()
WAIST_CONTROL_KP_SDK[15:29] = np.asarray([80.0] * 4 + [40.0] * 3 + [80.0] * 4 + [40.0] * 3)
WAIST_CONTROL_KD_SDK[15:29] = np.asarray([3.0] * 4 + [1.5] * 3 + [3.0] * 4 + [1.5] * 3)
TORQUE_LIMIT_SDK = np.asarray(
    [
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 250, 250,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float32,
)
# LS69 TR Round6 retained the same 12-DoF actor and 300/3 waist PD gains,
# while training waist roll/pitch against the G1 model's conservative 50 Nm
# effort envelope.  Keep this as a separate exact receipt; never turn torque
# admission into a range check.
WAIST50_TORQUE_LIMIT_SDK = TORQUE_LIMIT_SDK.copy()
WAIST50_TORQUE_LIMIT_SDK[13:15] = 50.0

# HOMIE's upper-body controller is deliberately a separate deployment overlay:
# DreamWaQ owns only the lower-body policy and waist targets.  These values are
# pinned from HOMIE's ``g1_homie.yaml`` and its ``rt/arm_sdk`` bridge so the
# fourteen arm joints use the same static bring-up gains, teleoperation gain
# envelope, and feed-forward torque clamp in either deployment.
#
# Arm order is SDK indices 15..28: left seven, then right seven.
HOMIE_ARM_DEFAULT_Q_SDK = np.asarray(
    [0.0] * 14,
    dtype=np.float32,
)
HOMIE_ARM_DEFAULT_KP_SDK = np.asarray(
    [40.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0] * 2,
    dtype=np.float32,
)
HOMIE_ARM_DEFAULT_KD_SDK = np.asarray(
    [5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0] * 2,
    dtype=np.float32,
)
# XR teleoperation and PATH's direct arm bridge publish these active values
# alongside their arm targets. They are a hard ceiling, never a gain
# multiplier. PATH's two 4010 wrist joints use Kd=1.548805, so retain a small
# margin above that measured controller value rather than rejecting a valid
# arm command at a nominal 1.5 ceiling.
HOMIE_ARM_ACTIVE_KP_MAX_SDK = np.asarray(
    [80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0] * 2,
    dtype=np.float32,
)
HOMIE_ARM_ACTIVE_KD_MAX_SDK = np.asarray(
    [3.0, 3.0, 3.0, 3.0, 1.6, 1.6, 1.6] * 2,
    dtype=np.float32,
)
HOMIE_ARM_TAU_FF_LIMIT_SDK = np.asarray(
    [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * 2,
    dtype=np.float32,
)

# Physical radians per normalized actor action in LOWER_BODY_JOINT_NAMES
# order. These are the product of the FlashSAC wrapper range and Isaac's G1
# action-manager scale, not the old scalar action_scale=0.25.
REFERENCE_PHYSICAL_ACTION_SCALE = np.asarray(
    [
        2.809274911880493, 2.809274911880493,
        2.792564630508423, 2.792564630508423,
        2.481839895248413, 2.481839895248413,
        2.31144642829895, 2.31144642829895,
        0.6837863922119141, 0.6837863922119141,
        0.23561996221542358, 0.23561996221542358,
    ],
    dtype=np.float32,
)
WAIST_CONTROL_PHYSICAL_ACTION_SCALE = np.concatenate(
    (REFERENCE_PHYSICAL_ACTION_SCALE, np.asarray((0.15, 0.08, 0.08), dtype=np.float32))
)
REFERENCE_COMMAND_LOWER = np.asarray((-0.6, -0.5, -0.5), dtype=np.float32)
REFERENCE_COMMAND_UPPER = np.asarray((1.0, 0.5, 0.5), dtype=np.float32)
# LS27 was trained with a deliberately symmetric, narrower command envelope.
# Keep both reviewed receipts exact: accepting an arbitrary range from a
# shape-compatible manifest would silently change the policy semantics.
LS27_COMMAND_LOWER = np.asarray((-0.4, -0.4, -0.4), dtype=np.float32)
LS27_COMMAND_UPPER = np.asarray((0.4, 0.4, 0.4), dtype=np.float32)
NEW3_COMMAND_LOWER = np.asarray((-0.8, -0.8, -1.0), dtype=np.float32)
NEW3_COMMAND_UPPER = np.asarray((1.0, 0.8, 1.0), dtype=np.float32)
_REVIEWED_COMMAND_RANGES = (
    (REFERENCE_COMMAND_LOWER, REFERENCE_COMMAND_UPPER),
    (LS27_COMMAND_LOWER, LS27_COMMAND_UPPER),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _exact_names(value: Any, expected: Sequence[str], label: str) -> None:
    names = tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()
    if names != tuple(expected):
        raise ValueError("%s is not the exact G1 29-DoF deployment order: got %d names" % (label, len(names)))


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("%s must contain exactly %d finite values" % (label, size))
    return result


class JointOrderMap:
    """Name-derived Unitree SDK <-> FlashSAC policy permutation."""

    def __init__(self) -> None:
        sdk_by_name = {name: index for index, name in enumerate(SDK_JOINT_NAMES)}
        policy_by_name = {name: index for index, name in enumerate(POLICY_JOINT_NAMES)}
        if len(sdk_by_name) != 29 or len(policy_by_name) != 29 or set(sdk_by_name) != set(policy_by_name):
            raise ValueError("G1 29-DoF joint orders are incomplete or inconsistent")
        self._sdk_for_policy = np.asarray([sdk_by_name[name] for name in POLICY_JOINT_NAMES], dtype=np.int64)
        self._policy_for_sdk = np.asarray([policy_by_name[name] for name in SDK_JOINT_NAMES], dtype=np.int64)
        self.lower_policy_indices = np.asarray([policy_by_name[name] for name in LOWER_BODY_JOINT_NAMES], dtype=np.int64)
        self.lower_sdk_indices = np.asarray([sdk_by_name[name] for name in LOWER_BODY_JOINT_NAMES], dtype=np.int64)
        self.spine_sdk_indices = np.asarray([sdk_by_name[name] for name in SPINE_JOINT_NAMES], dtype=np.int64)
        self.arm_sdk_indices = np.asarray([sdk_by_name[name] for name in ARM_JOINT_NAMES], dtype=np.int64)

    def policy_indices(self, names: Sequence[str]) -> np.ndarray:
        by_name = {name: index for index, name in enumerate(POLICY_JOINT_NAMES)}
        return np.asarray([by_name[str(name)] for name in names], dtype=np.int64)

    def sdk_indices(self, names: Sequence[str]) -> np.ndarray:
        by_name = {name: index for index, name in enumerate(SDK_JOINT_NAMES)}
        return np.asarray([by_name[str(name)] for name in names], dtype=np.int64)

    @staticmethod
    def _vector(values: Any, label: str) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1:] != (29,):
            raise ValueError("%s must end in exactly 29 values, got %s" % (label, array.shape))
        return array

    def sdk_to_policy(self, values: Any) -> np.ndarray:
        return self._vector(values, "SDK joint vector")[..., self._sdk_for_policy]

    def policy_to_sdk(self, values: Any) -> np.ndarray:
        return self._vector(values, "policy joint vector")[..., self._policy_for_sdk]


@dataclass(frozen=True)
class DreamWaQ29Contract:
    """Validated FlashSAC lower-body export and its physical profile."""

    export_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    policy_path: Path
    cenet_path: Path
    cenet_norm_path: Path
    artifact_sha256: Mapping[str, str]
    fingerprint: str
    default_q_sdk: np.ndarray
    kp_sdk: np.ndarray
    kd_sdk: np.ndarray
    torque_limit_sdk: np.ndarray
    physical_action_scale: np.ndarray
    action_joint_names: tuple[str, ...]
    command_lower: np.ndarray
    command_upper: np.ndarray
    observation_dim: int = 99
    raw_observation_dim: int = 96
    measurable_observation_dim: int = 93
    history_length: int = 5
    history_dim: int = 465
    latent_dim: int = 32
    velocity_dim: int = 3
    action_dim: int = 12
    actor_input_dim: int = 131
    clip_actions: float = 1.0
    clip_observations: float = 100.0
    joint_velocity_obs_scale: float = 0.05
    velocity_obs_scale: float = 1.0
    explicit_locomotion_mode: bool = False

    @classmethod
    def load(cls, export: Any) -> "DreamWaQ29Contract":
        raw = Path(export).expanduser()
        manifest_path = raw / "manifest.json" if raw.is_dir() else raw
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("DreamWaQ deployment manifest must be one regular file: %s" % manifest_path)
        export_dir = manifest_path.parent.resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read DreamWaQ deployment manifest: %s" % manifest_path) from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("DreamWaQ deployment manifest must contain one object")
        if manifest.get("format") != "dreamwaq_export_v2" or manifest.get("sim2sim_contract") != "g1_dreamwaq_sim2sim_v1":
            raise ValueError("unsupported DreamWaQ export format/contract")
        if manifest.get("source_algorithm") != "FlashSACDreamWaQ":
            raise ValueError("G1 deployment accepts FlashSACDreamWaQ exports only")
        layout = manifest.get("flashsac_observation_layout")
        if layout not in (FLASH_SAC_OBSERVATION_LAYOUT, FLASH_SAC_MODE_OBSERVATION_LAYOUT):
            raise ValueError("unsupported FlashSAC lower-body observation layout")
        explicit_mode = layout == FLASH_SAC_MODE_OBSERVATION_LAYOUT
        mode_contract = manifest.get("locomotion_mode_contract")
        if explicit_mode:
            if not isinstance(mode_contract, Mapping) or any(
                mode_contract.get(key) != value
                for key, value in (("standing", 0), ("walking", 1), ("actor_index", 99))
            ):
                raise ValueError("NEW3 export requires standing=0/walking=1 at actor index 99")
            standing_command = _finite_vector(
                mode_contract.get("standing_command"), 3, "locomotion_mode_contract.standing_command"
            )
            if np.any(standing_command != 0.0) or mode_contract.get("walking_zero_command_allowed") is not True:
                raise ValueError("NEW3 mode contract requires zero standing command and zero-speed walking")
        elif mode_contract is not None:
            raise ValueError("locomotion_mode_contract requires the NEW3 mode observation layout")
        asset_profile = manifest.get("g1_asset_profile")
        try:
            action_dim = int(manifest.get("num_actions", -1))
        except (TypeError, ValueError):
            action_dim = -1
        reviewed_actions = {
            FLASH_SAC_ASSET_PROFILE: LOWER_BODY_JOINT_NAMES,
            FLASH_SAC_WAIST_CONTROL_ASSET_PROFILE: WAIST_CONTROL_JOINT_NAMES,
        }
        action_joint_names = reviewed_actions.get(asset_profile)
        if action_joint_names is None or action_dim != len(action_joint_names):
            raise ValueError("FlashSAC export asset/action profile is not a reviewed G1 deployment contract")

        expected_ints = {
            "policy_obs_dim": 99 + int(explicit_mode), "dreamwaq_o1_dim": 93, "dreamwaq_hist_feat_dim": 93,
            "dreamwaq_hist_len": 5, "dreamwaq_hist_dim": 465, "dreamwaq_z_dim": 32,
        }
        for key, expected in expected_ints.items():
            try:
                actual = int(manifest.get(key, -1))
            except (TypeError, ValueError):
                actual = -1
            if actual != expected:
                raise ValueError("%s=%s; expected %d for FlashSAC G1 29-DoF" % (key, actual, expected))
        if manifest.get("include_gait_phase_obs") is not False:
            raise ValueError("FlashSAC G1 deployment does not accept gait-clock observations")
        if manifest.get("policy_action_squash") != "tanh" or manifest.get("g1_action_mode", "direct") != "direct":
            raise ValueError("FlashSAC export does not have the required direct tanh lower-body action contract")
        _exact_names(manifest.get("physical_joint_names"), POLICY_JOINT_NAMES, "physical_joint_names")
        _exact_names(manifest.get("policy_joint_names"), action_joint_names, "policy_joint_names")
        if "effective_policy_joint_names" in manifest:
            _exact_names(manifest.get("effective_policy_joint_names"), action_joint_names, "effective_policy_joint_names")
        nested = manifest.get("g1_action_contract")
        if nested not in ({}, None):
            if not isinstance(nested, Mapping):
                raise ValueError("g1_action_contract must be an object")
            for key, names in (
                ("physical_joint_names", POLICY_JOINT_NAMES),
                ("logical_action_names", action_joint_names),
                ("policy_joint_names", action_joint_names),
                ("effective_policy_joint_names", action_joint_names),
            ):
                if key in nested:
                    _exact_names(nested[key], names, "g1_action_contract." + key)
            if "asset_profile" in nested and nested.get("asset_profile") != asset_profile:
                raise ValueError("g1_action_contract.asset_profile differs from the export profile")
        if action_dim == 15:
            if manifest.get("action_contract_version") != "g1_lower_body_action_v2":
                raise ValueError("waist-control exports require the v2 action contract")
            if not isinstance(nested, Mapping) or nested.get("spine_fixed") is not False:
                raise ValueError("waist-control export must explicitly assign the spine to the actor")
            nested_scale = _finite_vector(
                nested.get("reference_physical_action_scale_policy_order"), 15,
                "g1_action_contract.reference_physical_action_scale_policy_order",
            )
            if not np.allclose(nested_scale, WAIST_CONTROL_PHYSICAL_ACTION_SCALE, atol=1.0e-6, rtol=0.0):
                raise ValueError("nested waist-control action scale differs from the reviewed profile")

        delay = manifest.get("action_delay_contract")
        if not isinstance(delay, Mapping) or delay.get("enabled") is not False or int(delay.get("max_delay_steps", -1)) != 0:
            raise ValueError("FlashSAC reference deployment requires the no-action-delay export")
        scales = manifest.get("obs_scales")
        if not isinstance(scales, Mapping):
            raise ValueError("manifest obs_scales must be an object")
        expected_scales = {"ang_vel": 1.0, "projected_gravity": 1.0, "commands": 1.0, "joint_pos": 1.0, "joint_vel": 0.05, "actions": 1.0}
        for key, expected in expected_scales.items():
            try:
                value = float(scales.get(key, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value) or abs(value - expected) > 1.0e-8:
                raise ValueError("obs_scales.%s differs from the FlashSAC reference contract" % key)
        for key, expected in (("clip_actions", 1.0), ("clip_observations", 100.0), ("dreamwaq_velocity_obs_scale", 1.0)):
            try:
                value = float(manifest.get(key, expected))
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value) or abs(value - expected) > 1.0e-8:
                raise ValueError("%s differs from the FlashSAC reference contract" % key)

        action_scale = _finite_vector(
            manifest.get("physical_action_scale_policy_order"), action_dim,
            "physical_action_scale_policy_order",
        )
        if np.any(action_scale <= 0.0):
            raise ValueError("physical_action_scale_policy_order must be positive")
        expected_action_scale = (
            WAIST_CONTROL_PHYSICAL_ACTION_SCALE
            if action_dim == 15 else REFERENCE_PHYSICAL_ACTION_SCALE
        )
        if not np.allclose(action_scale, expected_action_scale, atol=1.0e-6, rtol=0.0):
            raise ValueError("physical_action_scale_policy_order differs from the reviewed action profile")
        default_q_sdk = DEFAULT_Q_SDK.copy()
        default_policy = manifest.get("default_joint_pos_policy_order")
        if default_policy is not None:
            expected = JointOrderMap().sdk_to_policy(default_q_sdk)
            supplied = _finite_vector(default_policy, 29, "default_joint_pos_policy_order")
            if not np.allclose(supplied, expected, atol=1.0e-7, rtol=0.0):
                raise ValueError("manifest default joint pose differs from the FlashSAC reference")
        mapper = JointOrderMap()
        nominal_kp_sdk = KP_SDK.copy()
        nominal_kd_sdk = KD_SDK.copy()
        serialized_pd = manifest.get("flashsac_reference_pd_gains_policy_order")
        if serialized_pd is not None:
            supplied_pd = np.asarray(serialized_pd, dtype=np.float32)
            reviewed_profiles = (
                (KP_SDK, KD_SDK),
                (DEPLOYMENT_MATCHED_KP_SDK, DEPLOYMENT_MATCHED_KD_SDK),
                (WAIST_CONTROL_KP_SDK, WAIST_CONTROL_KD_SDK),
            )
            matched_profile = None
            if supplied_pd.shape == (29, 2) and np.all(np.isfinite(supplied_pd)):
                for profile_kp_sdk, profile_kd_sdk in reviewed_profiles:
                    expected_pd = np.stack(
                        (mapper.sdk_to_policy(profile_kp_sdk), mapper.sdk_to_policy(profile_kd_sdk)), axis=-1
                    )
                    if np.allclose(supplied_pd, expected_pd, atol=1.0e-6, rtol=0.0):
                        matched_profile = (profile_kp_sdk, profile_kd_sdk)
                        break
            if matched_profile is None:
                raise ValueError("serialized FlashSAC PD gains differ from the reviewed actuator profiles")
            nominal_kp_sdk = matched_profile[0].copy()
            nominal_kd_sdk = matched_profile[1].copy()
        nominal_torque_limit_sdk = TORQUE_LIMIT_SDK.copy()
        serialized_torque = manifest.get("flashsac_reference_torque_limits_policy_order")
        if serialized_torque is not None:
            supplied_torque = _finite_vector(serialized_torque, 29, "flashsac_reference_torque_limits_policy_order")
            matched_torque = next(
                (
                    profile
                    for profile in (TORQUE_LIMIT_SDK, WAIST50_TORQUE_LIMIT_SDK)
                    if np.allclose(
                        supplied_torque,
                        mapper.sdk_to_policy(profile),
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                ),
                None,
            )
            if matched_torque is None:
                raise ValueError("serialized FlashSAC torque limits differ from the reviewed actuator profiles")
            nominal_torque_limit_sdk = matched_torque.copy()
        command_lower, command_upper = REFERENCE_COMMAND_LOWER.copy(), REFERENCE_COMMAND_UPPER.copy()
        command_contract = manifest.get("command_range")
        if command_contract is not None:
            if not isinstance(command_contract, Mapping):
                raise ValueError("command_range must be an object")
            command_lower = _finite_vector(command_contract.get("lower"), 3, "command_range.lower")
            command_upper = _finite_vector(command_contract.get("upper"), 3, "command_range.upper")
            reviewed = any(
                np.allclose(command_lower, lower, atol=1.0e-7, rtol=0.0)
                and np.allclose(command_upper, upper, atol=1.0e-7, rtol=0.0)
                for lower, upper in (
                    _REVIEWED_COMMAND_RANGES
                    + (((NEW3_COMMAND_LOWER, NEW3_COMMAND_UPPER),) if explicit_mode else ())
                )
            )
            if not reviewed:
                raise ValueError("command range differs from the reviewed FlashSAC training envelopes")

        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for key, fallback in (("policy", "policy.pt"), ("cenet", "cenet.pt"), ("cenet_norm", "cenet_norm.pt")):
            filename = manifest.get(key) or fallback
            if Path(str(filename)).name != str(filename):
                raise ValueError("%s must name a file inside the export directory" % key)
            path = export_dir / str(filename)
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing regular DreamWaQ artifact: %s" % path)
            paths[key], hashes[key] = path.resolve(), _sha256_file(path)

        fingerprint_payload = {
            "schema": "dreamwaq.g1_29dof.flashsac_runtime_contract.v1",
            "manifest": dict(manifest), "artifact_sha256": hashes,
            "sdk_joint_names": list(SDK_JOINT_NAMES),
            "default_q_sdk": default_q_sdk.tolist(), "kp_sdk": nominal_kp_sdk.tolist(),
            "kd_sdk": nominal_kd_sdk.tolist(),
            "torque_limit_sdk": nominal_torque_limit_sdk.tolist(), "command_lower": command_lower.tolist(),
            "command_upper": command_upper.tolist(),
        }
        return cls(
            export_dir=export_dir, manifest_path=manifest_path.resolve(), manifest=dict(manifest),
            policy_path=paths["policy"], cenet_path=paths["cenet"], cenet_norm_path=paths["cenet_norm"],
            artifact_sha256=hashes, fingerprint=_stable_hash(fingerprint_payload),
            default_q_sdk=default_q_sdk, kp_sdk=nominal_kp_sdk, kd_sdk=nominal_kd_sdk,
            torque_limit_sdk=nominal_torque_limit_sdk, physical_action_scale=action_scale.copy(),
            action_joint_names=tuple(action_joint_names), command_lower=command_lower,
            command_upper=command_upper, action_dim=action_dim,
            observation_dim=99 + int(explicit_mode), actor_input_dim=131 + int(explicit_mode),
            explicit_locomotion_mode=explicit_mode,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "dreamwaq.g1_29dof.flashsac_runtime_contract.v1", "fingerprint": self.fingerprint,
            "manifest_path": str(self.manifest_path), "artifact_sha256": dict(self.artifact_sha256),
            "source_algorithm": "FlashSACDreamWaQ", "sdk_joint_names": list(SDK_JOINT_NAMES),
            "policy_joint_names": list(POLICY_JOINT_NAMES), "lower_body_joint_names": list(LOWER_BODY_JOINT_NAMES),
            "action_joint_names": list(self.action_joint_names),
            "raw_observation_dim": self.raw_observation_dim, "observation_dim": self.observation_dim,
            "measurable_observation_dim": self.measurable_observation_dim, "history_length": self.history_length,
            "history_dim": self.history_dim, "actor_input_dim": self.actor_input_dim, "action_dim": self.action_dim,
            "locomotion_mode_contract": self.manifest.get("locomotion_mode_contract"),
            "default_q_sdk": self.default_q_sdk.tolist(), "physical_action_scale": self.physical_action_scale.tolist(),
            "kp_sdk": self.kp_sdk.tolist(), "kd_sdk": self.kd_sdk.tolist(),
            "torque_limit_sdk": self.torque_limit_sdk.tolist(),
            "command_range": {"lower": self.command_lower.tolist(), "upper": self.command_upper.tolist()},
        }


__all__ = [
    "ARM_JOINT_NAMES", "DEFAULT_Q_SDK", "DreamWaQ29Contract", "FLASH_SAC_ASSET_PROFILE",
    "FLASH_SAC_WAIST_CONTROL_ASSET_PROFILE",
    "FLASH_SAC_OBSERVATION_LAYOUT", "JOINT_LOWER_SDK", "JOINT_UPPER_SDK", "JOINT_VELOCITY_LIMIT_SDK",
    "DEPLOYMENT_MATCHED_KD_SDK", "DEPLOYMENT_MATCHED_KP_SDK", "KD_SDK", "KP_SDK",
    "LOWER_BODY_JOINT_NAMES", "LS27_COMMAND_LOWER", "LS27_COMMAND_UPPER", "POLICY_JOINT_NAMES",
    "REFERENCE_COMMAND_LOWER", "REFERENCE_COMMAND_UPPER", "REFERENCE_PHYSICAL_ACTION_SCALE",
    "SDK_JOINT_NAMES", "SPINE_JOINT_NAMES", "WAIST_CONTROL_JOINT_NAMES",
    "WAIST_CONTROL_KD_SDK", "WAIST_CONTROL_KP_SDK", "WAIST_CONTROL_PHYSICAL_ACTION_SCALE",
    "TORQUE_LIMIT_SDK", "WAIST50_TORQUE_LIMIT_SDK", "JointOrderMap", "HOMIE_ARM_DEFAULT_Q_SDK",
]
