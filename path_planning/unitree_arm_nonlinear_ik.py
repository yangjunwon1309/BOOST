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

from path_core.teleop.xr_arm_postprocessing import XrArmCommandPostprocessor

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

G1_29_JOINT_NAMES = [
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
    *LEFT_ARM_NAMES,
    *RIGHT_ARM_NAMES,
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
    left_position_error_m: float = float("inf")
    right_position_error_m: float = float("inf")
    left_rotation_error_rad: float = float("inf")
    right_rotation_error_rad: float = float("inf")
    left_orientation_weight_scale: float = 0.0
    right_orientation_weight_scale: float = 0.0
    target_projected: bool = False
    postprocess_rate_limited: bool = False
    postprocess_motion_scale: float = 1.0
    postprocess_max_step_rad: float = 0.0


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
        orientation_full_error_m: float = 0.01,
        orientation_zero_error_m: float = 0.03,
        orientation_min_scale: float = 1.0,
        residual_projection_threshold_m: float = 0.03,
        residual_success_threshold_m: float = 0.05,
        projection_step_m: float = 0.05,
        output_velocity_limit_rad_s: float = 20.0,
        output_control_dt_s: float = 1.0 / 250.0,
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
        self.robot.model.addFrame(
            pin.Frame(
                "PATH_full_L_ee",
                self.robot.model.getJointId("left_wrist_yaw_joint"),
                0,
                pin.SE3(np.eye(3), np.array([ee_offset_m, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.model.addFrame(
            pin.Frame(
                "PATH_full_R_ee",
                self.robot.model.getJointId("right_wrist_yaw_joint"),
                0,
                pin.SE3(np.eye(3), np.array([ee_offset_m, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.full_data = self.robot.model.createData()
        self.full_left_frame_id = self.robot.model.getFrameId("PATH_full_L_ee")
        self.full_right_frame_id = self.robot.model.getFrameId("PATH_full_R_ee")
        self.full_q_indices = np.asarray(
            [
                int(self.robot.model.joints[
                    self.robot.model.getJointId(name)
                ].idx_q)
                for name in G1_29_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self.full_v_indices = np.asarray(
            [
                int(self.robot.model.joints[
                    self.robot.model.getJointId(name)
                ].idx_v)
                for name in G1_29_JOINT_NAMES
            ],
            dtype=np.int64,
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
        self.param_orientation_scale_l = self.opti.parameter()
        self.param_orientation_scale_r = self.opti.parameter()

        translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        rotation_error = self.rotational_error(
            self.var_q, self.param_tf_l, self.param_tf_r
        )
        left_rotation_cost = casadi.sumsqr(rotation_error[:3])
        right_rotation_cost = casadi.sumsqr(rotation_error[3:])
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(self.param_lower, self.var_q, self.param_upper)
        )
        self.opti.minimize(
            position_weight * translational_cost
            + orientation_weight
            * (
                self.param_orientation_scale_l * left_rotation_cost
                + self.param_orientation_scale_r * right_rotation_cost
            )
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
        self.orientation_full_error_m = float(orientation_full_error_m)
        self.orientation_zero_error_m = float(orientation_zero_error_m)
        self.orientation_min_scale = float(orientation_min_scale)
        self.residual_projection_threshold_m = float(
            residual_projection_threshold_m
        )
        self.residual_success_threshold_m = float(residual_success_threshold_m)
        self.projection_step_m = float(projection_step_m)
        self.arm_postprocessor = XrArmCommandPostprocessor(
            velocity_limit_rad_s=output_velocity_limit_rad_s,
            control_dt_s=output_control_dt_s,
            data_size=self.reduced_robot.model.nq,
        )

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
        """Arm-solver FK with legs and waist locked at identity."""

        arm_q = self.q29_to_arm_q(q29)
        pin.framesForwardKinematics(self.reduced_robot.model, self.data, arm_q)
        left = np.eye(4, dtype=np.float64)
        right = np.eye(4, dtype=np.float64)
        left[:3, :3] = self.data.oMf[self.left_frame_id].rotation
        left[:3, 3] = self.data.oMf[self.left_frame_id].translation
        right[:3, :3] = self.data.oMf[self.right_frame_id].rotation
        right[:3, 3] = self.data.oMf[self.right_frame_id].translation
        return left, right

    def gmr_wrist_transforms_from_q29(
        self, q29: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pelvis-identity FK of the complete GMR waist+arm configuration."""

        q_full = np.zeros(self.robot.model.nq, dtype=np.float64)
        q_full[self.full_q_indices] = np.asarray(q29, dtype=np.float64).reshape(29)
        pin.framesForwardKinematics(self.robot.model, self.full_data, q_full)
        left = np.eye(4, dtype=np.float64)
        right = np.eye(4, dtype=np.float64)
        left[:3, :3] = self.full_data.oMf[self.full_left_frame_id].rotation
        left[:3, 3] = self.full_data.oMf[self.full_left_frame_id].translation
        right[:3, :3] = self.full_data.oMf[self.full_right_frame_id].rotation
        right[:3, 3] = self.full_data.oMf[self.full_right_frame_id].translation
        return left, right

    def gravity_torque_from_q29(
        self,
        q29: np.ndarray,
        base_quaternion_wxyz: np.ndarray | None = None,
        gravity_direction_pelvis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return base-oriented gravity compensation in G1 hardware order."""

        q_full = np.zeros(self.robot.model.nq, dtype=np.float64)
        q_full[self.full_q_indices] = np.asarray(q29, dtype=np.float64).reshape(29)
        original_gravity = self.robot.model.gravity.linear.copy()
        if gravity_direction_pelvis is not None:
            gravity_direction = np.asarray(
                gravity_direction_pelvis, dtype=np.float64
            ).reshape(3)
            norm = float(np.linalg.norm(gravity_direction))
            if norm > 1.0e-8 and np.isfinite(gravity_direction).all():
                self.robot.model.gravity.linear = (
                    9.81 * gravity_direction / norm
                )
        elif base_quaternion_wxyz is not None:
            # The IMU quaternion rotates pelvis-frame vectors into world.
            # R_world_pelvis.T therefore expresses world gravity in pelvis.
            rotation_world_pelvis = quat_wxyz_to_rot(base_quaternion_wxyz)
            self.robot.model.gravity.linear = (
                rotation_world_pelvis.T
                @ np.array([0.0, 0.0, -9.81], dtype=np.float64)
            )
        try:
            tau_full = pin.rnea(
                self.robot.model,
                self.full_data,
                q_full,
                np.zeros(self.robot.model.nv, dtype=np.float64),
                np.zeros(self.robot.model.nv, dtype=np.float64),
            )
        finally:
            self.robot.model.gravity.linear = original_gravity
        return np.asarray(tau_full, dtype=np.float64)[self.full_v_indices].copy()

    def _orientation_scale(self, position_error_m: float) -> float:
        """Use the XR-teleop orientation objective at full weight."""

        del position_error_m
        return 1.0

    def _solution_metrics(
        self,
        q_ref: np.ndarray,
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
    ) -> tuple[float, float, float, float]:
        solved_left, solved_right = self.wrist_transforms_from_q29(q_ref)
        left_position_error = float(
            np.linalg.norm(solved_left[:3, 3] - left_target_tf[:3, 3])
        )
        right_position_error = float(
            np.linalg.norm(solved_right[:3, 3] - right_target_tf[:3, 3])
        )
        left_rotation_error = float(
            np.linalg.norm(
                pin.log3(solved_left[:3, :3] @ left_target_tf[:3, :3].T)
            )
        )
        right_rotation_error = float(
            np.linalg.norm(
                pin.log3(solved_right[:3, :3] @ right_target_tf[:3, :3].T)
            )
        )
        return (
            left_position_error,
            right_position_error,
            left_rotation_error,
            right_rotation_error,
        )

    def _project_target_from_current(
        self,
        current_tf: np.ndarray,
        requested_tf: np.ndarray,
    ) -> np.ndarray:
        projected = np.asarray(requested_tf, dtype=np.float64).copy()
        delta = requested_tf[:3, 3] - current_tf[:3, 3]
        distance = float(np.linalg.norm(delta))
        if distance > self.projection_step_m > 0.0:
            projected[:3, 3] = (
                current_tf[:3, 3]
                + delta * (self.projection_step_m / distance)
            )
        return projected

    def _solve_once(
        self,
        *,
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
        q_last: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        initial: np.ndarray,
        left_orientation_scale: float,
        right_orientation_scale: float,
    ) -> np.ndarray:
        self.opti.set_initial(self.var_q, np.clip(initial, lower, upper))
        self.opti.set_value(self.var_q_last, q_last)
        self.opti.set_value(self.param_lower, lower)
        self.opti.set_value(self.param_upper, upper)
        self.opti.set_value(self.param_tf_l, left_target_tf)
        self.opti.set_value(self.param_tf_r, right_target_tf)
        self.opti.set_value(
            self.param_orientation_scale_l, float(left_orientation_scale)
        )
        self.opti.set_value(
            self.param_orientation_scale_r, float(right_orientation_scale)
        )
        sol = self.opti.solve()
        return np.asarray(sol.value(self.var_q), dtype=np.float64).reshape(14)

    def _solve_transforms(
        self,
        *,
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
        q_seed29: np.ndarray,
        q_smooth29: np.ndarray | None = None,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        base_quaternion_wxyz: np.ndarray | None = None,
    ) -> ArmIkResult:
        """Solve both arm joint vectors against explicit pelvis-local EE poses."""

        started = time.perf_counter()
        q_seed29 = np.asarray(q_seed29, dtype=np.float64).reshape(29)
        left_target_tf = np.asarray(left_target_tf, dtype=np.float64).reshape(4, 4)
        right_target_tf = np.asarray(right_target_tf, dtype=np.float64).reshape(4, 4)
        lower = self.lower.copy() if lower is None else np.asarray(lower, dtype=np.float64).copy()
        upper = self.upper.copy() if upper is None else np.asarray(upper, dtype=np.float64).copy()
        smooth_source = q_seed29 if q_smooth29 is None else np.asarray(
            q_smooth29, dtype=np.float64
        ).reshape(29)
        q_last = self.q29_to_arm_q(smooth_source)

        try:
            current_left, current_right = self.wrist_transforms_from_q29(q_seed29)
            left_request_distance = float(
                np.linalg.norm(current_left[:3, 3] - left_target_tf[:3, 3])
            )
            right_request_distance = float(
                np.linalg.norm(current_right[:3, 3] - right_target_tf[:3, 3])
            )
            left_orientation_scale = self._orientation_scale(
                left_request_distance
            )
            right_orientation_scale = self._orientation_scale(
                right_request_distance
            )
            arm_q = self._solve_once(
                left_target_tf=left_target_tf,
                right_target_tf=right_target_tf,
                q_last=q_last,
                lower=lower,
                upper=upper,
                initial=self.init_data,
                left_orientation_scale=left_orientation_scale,
                right_orientation_scale=right_orientation_scale,
            )
            q_raw = self.arm_q_to_q29(q_seed29, arm_q)
            first_metrics = self._solution_metrics(
                q_raw, left_target_tf, right_target_tf
            )

            target_projected = (
                first_metrics[0] > self.residual_projection_threshold_m
                or first_metrics[1] > self.residual_projection_threshold_m
            )
            applied_left_tf = left_target_tf
            applied_right_tf = right_target_tf
            if target_projected:
                if first_metrics[0] > self.residual_projection_threshold_m:
                    applied_left_tf = self._project_target_from_current(
                        current_left, left_target_tf
                    )
                if first_metrics[1] > self.residual_projection_threshold_m:
                    applied_right_tf = self._project_target_from_current(
                        current_right, right_target_tf
                    )
                left_projected_distance = float(
                    np.linalg.norm(
                        current_left[:3, 3] - applied_left_tf[:3, 3]
                    )
                )
                right_projected_distance = float(
                    np.linalg.norm(
                        current_right[:3, 3] - applied_right_tf[:3, 3]
                    )
                )
                left_orientation_scale = self._orientation_scale(
                    left_projected_distance
                )
                right_orientation_scale = self._orientation_scale(
                    right_projected_distance
                )
                arm_q = self._solve_once(
                    left_target_tf=applied_left_tf,
                    right_target_tf=applied_right_tf,
                    q_last=q_last,
                    lower=lower,
                    upper=upper,
                    initial=arm_q,
                    left_orientation_scale=left_orientation_scale,
                    right_orientation_scale=right_orientation_scale,
                )

            q_raw = self.arm_q_to_q29(q_seed29, arm_q)
            raw_metrics = self._solution_metrics(
                q_raw, applied_left_tf, applied_right_tf
            )
            postprocessed = self.arm_postprocessor.apply(
                arm_q,
                self.q29_to_arm_q(q_seed29),
            )
            q_ref = self.arm_q_to_q29(q_seed29, postprocessed.command_q)
            command_metrics = self._solution_metrics(
                q_ref, applied_left_tf, applied_right_tf
            )

            # XR teleop warms IPOPT from the filtered solution; controller-side
            # velocity clipping does not alter the next optimization seed.
            self.init_data = postprocessed.filtered_q.copy()
            q_ref[12:15] = 0.0
            tau_ff = self.gravity_torque_from_q29(
                q_ref, base_quaternion_wxyz=base_quaternion_wxyz
            )
            # Legs and waist remain under SONIC/PD ownership. PATH supplies
            # RNEA feedforward only for the two IPOPT-controlled arms.
            tau_ff[:15] = 0.0

            (
                left_position_error,
                right_position_error,
                left_rotation_error,
                right_rotation_error,
            ) = command_metrics
            position_error = max(left_position_error, right_position_error)
            rotation_error = max(left_rotation_error, right_rotation_error)
            raw_solution_success = (
                max(raw_metrics[:2]) <= self.residual_success_threshold_m
            )
            success = raw_solution_success
            message = (
                "ok_projected_postprocessed"
                if success and target_projected
                else "ok_postprocessed"
                if success
                else (
                    "ee_residual_exceeded:"
                    f"left={raw_metrics[0]:.4f}m,"
                    f"right={raw_metrics[1]:.4f}m"
                )
            )
            return ArmIkResult(
                success=success,
                q_ref=q_ref,
                tau_ff_ref=tau_ff,
                solve_time_ms=(time.perf_counter() - started) * 1000.0,
                position_error_m=position_error,
                rotation_error_rad=rotation_error,
                message=message,
                left_position_error_m=left_position_error,
                right_position_error_m=right_position_error,
                left_rotation_error_rad=left_rotation_error,
                right_rotation_error_rad=right_rotation_error,
                left_orientation_weight_scale=left_orientation_scale,
                right_orientation_weight_scale=right_orientation_scale,
                target_projected=target_projected,
                postprocess_rate_limited=postprocessed.rate_limited,
                postprocess_motion_scale=postprocessed.motion_scale,
                postprocess_max_step_rad=postprocessed.max_command_step_rad,
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

    def solve_bimanual_targets(
        self,
        *,
        left_target_tf: np.ndarray,
        right_target_tf: np.ndarray,
        q_seed29: np.ndarray,
        q_smooth29: np.ndarray | None = None,
        base_quaternion_wxyz: np.ndarray | None = None,
    ) -> ArmIkResult:
        """Solve the authoritative 14-DoF arm target used by PATH q/e modes."""

        return self._solve_transforms(
            left_target_tf=left_target_tf,
            right_target_tf=right_target_tf,
            q_seed29=q_seed29,
            q_smooth29=q_smooth29,
            base_quaternion_wxyz=base_quaternion_wxyz,
        )

    def solve_gmr_reference(
        self,
        *,
        q_gmr29: np.ndarray,
        q_seed29: np.ndarray,
        left_position_override: np.ndarray | None = None,
        right_position_override: np.ndarray | None = None,
        base_quaternion_wxyz: np.ndarray | None = None,
    ) -> ArmIkResult:
        """Solve GMR EE poses, optionally replacing only an e-mode position."""

        left_tf, right_tf = self.gmr_wrist_transforms_from_q29(q_gmr29)
        if left_position_override is not None:
            left_tf[:3, 3] = np.asarray(left_position_override, dtype=np.float64).reshape(3)
        if right_position_override is not None:
            right_tf[:3, 3] = np.asarray(right_position_override, dtype=np.float64).reshape(3)
        return self.solve_bimanual_targets(
            left_target_tf=left_tf,
            right_target_tf=right_tf,
            q_seed29=q_seed29,
            base_quaternion_wxyz=base_quaternion_wxyz,
        )

    def solve_single_target(
        self,
        *,
        target_link: str,
        target_position: np.ndarray,
        target_quaternion_wxyz: np.ndarray,
        q_seed29: np.ndarray,
        q_hold29: np.ndarray,
        hard_lock_inactive_arm: bool = True,
        base_quaternion_wxyz: np.ndarray | None = None,
    ) -> ArmIkResult:
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

        return self._solve_transforms(
            left_target_tf=left_tf,
            right_target_tf=right_tf,
            q_seed29=q_seed29,
            lower=lower,
            upper=upper,
            base_quaternion_wxyz=base_quaternion_wxyz,
        )
