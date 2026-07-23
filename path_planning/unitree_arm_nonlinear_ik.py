#!/usr/bin/env python3
"""Unitree-style arm-only nonlinear IK for PATH references.

This module mirrors the G1_29 IK objective used in Unitree's xr_teleop:

  50 * position_error^2 + 1 * orientation_log_error^2
  + 0.02 * q_regularization^2 + 0.1 * q_smooth^2

The reduced model locks legs, waist, and hands.  The decision variable is the
14 arm joints themselves, not a delta around SONIC's PD target.  For single-arm
static targets, the opposite arm is hard-locked by setting its dynamic joint
bounds to the latched hold q.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

try:
    import casadi
except ImportError as exc:  # pragma: no cover - exercised on machines without CasADi.
    casadi = None
    _CASADI_IMPORT_ERROR = exc
else:
    _CASADI_IMPORT_ERROR = None


ARM_HW_INDICES = np.array(
    [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    dtype=np.int64,
)
LEFT_ARM_HW = np.array([15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
RIGHT_ARM_HW = np.array([22, 23, 24, 25, 26, 27, 28], dtype=np.int64)

LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LOCKED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


@dataclass
class ArmIkResult:
    success: bool
    q_ref: np.ndarray
    tau_ff_ref: np.ndarray
    solve_time_ms: float
    position_error_m: float
    rotation_error_rad: float
    message: str = "ok"


def quat_wxyz_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.eye(3)
    quat = quat / norm
    return pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()


def transform_from_pos_quat(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = quat_wxyz_to_rot(quat_wxyz)
    tf[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return tf


def _casadi_mat(value: np.ndarray):
    mat = casadi.SX(value.shape[0], value.shape[1])
    for r in range(value.shape[0]):
        for c in range(value.shape[1]):
            mat[r, c] = float(value[r, c])
    return mat


def _casadi_eye4():
    mat = casadi.SX.zeros(4, 4)
    for i in range(4):
        mat[i, i] = 1.0
    return mat


def _casadi_se3(rotation: np.ndarray, translation: np.ndarray):
    mat = _casadi_eye4()
    mat[:3, :3] = _casadi_mat(np.asarray(rotation, dtype=np.float64))
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    mat[0, 3] = float(t[0])
    mat[1, 3] = float(t[1])
    mat[2, 3] = float(t[2])
    return mat


def _rot_x(theta):
    c = casadi.cos(theta)
    s = casadi.sin(theta)
    mat = _casadi_eye4()
    mat[1, 1] = c
    mat[1, 2] = -s
    mat[2, 1] = s
    mat[2, 2] = c
    return mat


def _rot_y(theta):
    c = casadi.cos(theta)
    s = casadi.sin(theta)
    mat = _casadi_eye4()
    mat[0, 0] = c
    mat[0, 2] = s
    mat[2, 0] = -s
    mat[2, 2] = c
    return mat


def _rot_z(theta):
    c = casadi.cos(theta)
    s = casadi.sin(theta)
    mat = _casadi_eye4()
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    return mat


def _joint_motion_transform(shortname: str, theta):
    if shortname == "JointModelRX":
        return _rot_x(theta)
    if shortname == "JointModelRY":
        return _rot_y(theta)
    if shortname == "JointModelRZ":
        return _rot_z(theta)
    raise RuntimeError(f"unsupported reduced G1 arm joint type for CasADi FK: {shortname}")


def _so3_log_casadi(rotation):
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = casadi.fmin(1.0 - 1.0e-9, casadi.fmax(-1.0 + 1.0e-9, cos_theta))
    theta = casadi.acos(cos_theta)
    sin_theta = casadi.sin(theta)
    scale = casadi.if_else(theta < 1.0e-6, 0.5, theta / (2.0 * sin_theta))
    return scale * casadi.vertcat(
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    )


class UnitreeG1ArmNonlinearIK:
    def __init__(
        self,
        *,
        urdf_path: Path,
        model_dir: Path,
        max_iter: int = 30,
        position_weight: float = 50.0,
        orientation_weight: float = 1.0,
        regularization_weight: float = 0.02,
        smooth_weight: float = 0.1,
        ee_offset_m: float = 0.05,
        use_smooth_filter: bool = True,
    ) -> None:
        if casadi is None:
            raise RuntimeError(
                "CasADi is required for Unitree-style nonlinear IK. "
                "Install it in the PATH Python environment, e.g. `pip install casadi`."
            ) from _CASADI_IMPORT_ERROR

        self.urdf_path = Path(urdf_path).expanduser()
        self.model_dir = Path(model_dir).expanduser()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(self.urdf_path), [str(self.model_dir)]
        )
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=LOCKED_JOINTS,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                0,
                pin.SE3(np.eye(3), np.array([ee_offset_m, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                0,
                pin.SE3(np.eye(3), np.array([ee_offset_m, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.reduced_robot.model.createData()
        self.left_frame_id = self.reduced_robot.model.getFrameId("L_ee")
        self.right_frame_id = self.reduced_robot.model.getFrameId("R_ee")
        self.left_q_indices = self._joint_q_indices(LEFT_ARM_NAMES)
        self.right_q_indices = self._joint_q_indices(RIGHT_ARM_NAMES)

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cT_l = self._symbolic_wrist_transform(
            self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
            ee_offset_m,
        )
        cT_r = self._symbolic_wrist_transform(
            self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
            ee_offset_m,
        )
        cR_l_err = cT_l[:3, :3] @ self.cTf_l[:3, :3].T
        cR_r_err = cT_r[:3, :3] @ self.cTf_r[:3, :3].T

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cT_l[:3, 3] - self.cTf_l[:3, 3],
                    cT_r[:3, 3] - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    _so3_log_casadi(cR_l_err),
                    _so3_log_casadi(cR_r_err),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_lower = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_upper = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(self.param_lower, self.var_q, self.param_upper)
        )
        self.opti.minimize(
            position_weight * translational_cost
            + orientation_weight * rotation_cost
            + regularization_weight * regularization_cost
            + smooth_weight * smooth_cost
        )

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.lower = np.asarray(self.reduced_robot.model.lowerPositionLimit, dtype=np.float64)
        self.upper = np.asarray(self.reduced_robot.model.upperPositionLimit, dtype=np.float64)
        self.init_data = np.zeros(self.reduced_robot.model.nq, dtype=np.float64)
        self.use_smooth_filter = bool(use_smooth_filter)
        self.filter_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        self.filter_buffer: list[np.ndarray] = []

    def _joint_chain_to_root(self, joint_id: int) -> list[int]:
        chain: list[int] = []
        current = int(joint_id)
        while current > 0:
            chain.append(current)
            current = int(self.reduced_robot.model.parents[current])
        chain.reverse()
        return chain

    def _symbolic_wrist_transform(self, wrist_joint_id: int, ee_offset_m: float):
        transform = _casadi_eye4()
        for joint_id in self._joint_chain_to_root(wrist_joint_id):
            joint = self.reduced_robot.model.joints[joint_id]
            placement = self.reduced_robot.model.jointPlacements[joint_id]
            transform = (
                transform
                @ _casadi_se3(placement.rotation, placement.translation)
                @ _joint_motion_transform(joint.shortname(), self.cq[int(joint.idx_q)])
            )
        offset = _casadi_eye4()
        offset[0, 3] = float(ee_offset_m)
        return transform @ offset

    def _joint_q_indices(self, joint_names: list[str]) -> np.ndarray:
        indices: list[int] = []
        for name in joint_names:
            joint_id = self.reduced_robot.model.getJointId(name)
            if joint_id <= 0:
                raise RuntimeError(f"joint not found in reduced G1 model: {name}")
            indices.append(int(self.reduced_robot.model.joints[joint_id].idx_q))
        return np.asarray(indices, dtype=np.int64)

    def q29_to_arm_q(self, q29: np.ndarray) -> np.ndarray:
        return np.asarray(q29, dtype=np.float64).reshape(29)[ARM_HW_INDICES].copy()

    def arm_q_to_q29(self, q_base: np.ndarray, arm_q: np.ndarray) -> np.ndarray:
        q29 = np.asarray(q_base, dtype=np.float64).reshape(29).copy()
        q29[ARM_HW_INDICES] = np.asarray(arm_q, dtype=np.float64).reshape(14)
        return q29

    def wrist_transforms_from_q29(self, q29: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arm_q = self.q29_to_arm_q(q29)
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, arm_q)
        left = np.eye(4, dtype=np.float64)
        right = np.eye(4, dtype=np.float64)
        left[:3, :3] = self.data.oMf[self.left_frame_id].rotation
        left[:3, 3] = self.data.oMf[self.left_frame_id].translation
        right[:3, :3] = self.data.oMf[self.right_frame_id].rotation
        right[:3, 3] = self.data.oMf[self.right_frame_id].translation
        return left, right

    def _filtered(self, q: np.ndarray) -> np.ndarray:
        if not self.use_smooth_filter:
            return q
        self.filter_buffer.insert(0, q.copy())
        del self.filter_buffer[len(self.filter_weights):]
        out = np.zeros_like(q)
        total = 0.0
        for weight, item in zip(self.filter_weights, self.filter_buffer):
            out += weight * item
            total += weight
        return out / max(total, 1.0e-9)

    def solve_single_target(
        self,
        *,
        target_link: str,
        target_position: np.ndarray,
        target_quaternion_wxyz: np.ndarray,
        q_seed29: np.ndarray,
        q_hold29: np.ndarray,
        hard_lock_inactive_arm: bool = True,
    ) -> ArmIkResult:
        started = time.perf_counter()
        q_seed29 = np.asarray(q_seed29, dtype=np.float64).reshape(29)
        q_hold29 = np.asarray(q_hold29, dtype=np.float64).reshape(29)
        target_side = "right" if "right" in target_link else "left"

        left_tf, right_tf = self.wrist_transforms_from_q29(q_hold29)
        target_tf = transform_from_pos_quat(target_position, target_quaternion_wxyz)
        if target_side == "left":
            left_tf = target_tf
        else:
            right_tf = target_tf

        lower = self.lower.copy()
        upper = self.upper.copy()
        hold_arm_q = self.q29_to_arm_q(q_hold29)
        if hard_lock_inactive_arm:
            if target_side == "left":
                lower[self.right_q_indices] = hold_arm_q[self.right_q_indices]
                upper[self.right_q_indices] = hold_arm_q[self.right_q_indices]
            else:
                lower[self.left_q_indices] = hold_arm_q[self.left_q_indices]
                upper[self.left_q_indices] = hold_arm_q[self.left_q_indices]

        q_last = self.q29_to_arm_q(q_seed29)
        self.opti.set_initial(self.var_q, np.clip(self.init_data, lower, upper))
        self.opti.set_value(self.var_q_last, q_last)
        self.opti.set_value(self.param_lower, lower)
        self.opti.set_value(self.param_upper, upper)
        self.opti.set_value(self.param_tf_l, left_tf)
        self.opti.set_value(self.param_tf_r, right_tf)

        try:
            sol = self.opti.solve()
            arm_q = np.asarray(sol.value(self.var_q), dtype=np.float64).reshape(14)
            arm_q = self._filtered(arm_q)
            self.init_data = arm_q.copy()
            tau_arm = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                arm_q,
                np.zeros(14, dtype=np.float64),
                np.zeros(14, dtype=np.float64),
            )
            q_ref = self.arm_q_to_q29(q_seed29, arm_q)
            tau_ff = np.zeros(29, dtype=np.float64)
            tau_ff[ARM_HW_INDICES] = np.asarray(tau_arm, dtype=np.float64).reshape(14)

            solved_left, solved_right = self.wrist_transforms_from_q29(q_ref)
            solved_tf = solved_left if target_side == "left" else solved_right
            pos_error = float(np.linalg.norm(solved_tf[:3, 3] - target_tf[:3, 3]))
            rot_error = float(np.linalg.norm(pin.log3(solved_tf[:3, :3] @ target_tf[:3, :3].T)))
            return ArmIkResult(
                success=True,
                q_ref=q_ref,
                tau_ff_ref=tau_ff,
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                position_error_m=pos_error,
                rotation_error_rad=rot_error,
            )
        except Exception as exc:
            return ArmIkResult(
                success=False,
                q_ref=q_seed29.copy(),
                tau_ff_ref=np.zeros(29, dtype=np.float64),
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                position_error_m=float("inf"),
                rotation_error_rad=float("inf"),
                message=str(exc),
            )
