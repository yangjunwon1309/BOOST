# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python pico_manager_thread_server.py --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python pico_manager_thread_server.py --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python pico_manager_thread_server.py --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
#   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

# RECORDED SESSION REPLAY (no Pico/XR service required):
#   python pico_manager_thread_server.py --manager \
#       --replay_session gear_sonic/scripts/exp_qp_8 --target_fps 50

"""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, IntEnum
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

PATH_G1_ROOT = Path(__file__).resolve().parents[1]
HUMAN_JOINTS_INFO_PATH = (
    PATH_G1_ROOT / "sonic_runtime" / "gear_sonic" / "data" / "human" / "human_joints_info.pkl"
)

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)
from path_core.teleop.pico_joystick_locomotion import (
    PicoButtonGestureInterpreter,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        build_command_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5
    ALIGNING = 6
    TRACKING = 7


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


T_ROBOT_OPENXR = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
T_OPENXR_ROBOT = np.array(
    [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
T_TO_UNITREE_HUMANOID_LEFT_ARM = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
T_TO_UNITREE_HUMANOID_RIGHT_ARM = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
WAIST_OFFSET_XYZ = np.array([0.15, 0.0, 0.45], dtype=np.float64)
CONST_HEAD_POSE = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_LEFT_ARM_POSE = np.array(
    [[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)
CONST_RIGHT_ARM_POSE = np.array(
    [[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]],
    dtype=np.float64,
)

# G1 URDF zero pose: upper arms hang vertically from the shoulders while the
# forearms point forward (+X), giving the requested geometric 90-degree elbow.
Q_READY = np.zeros(29, dtype=np.float64)


def _pico_pose_to_mat_openxr(
    pose7: np.ndarray | list[float],
) -> tuple[np.ndarray, bool]:
    """PICO SDK [x,y,z,qx,qy,qz,qw] to OpenXR-basis SE(3)."""

    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    if not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1.0e-8:
        return np.eye(4, dtype=np.float64), False
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = sRot.from_quat(p[3:]).as_matrix()
    transform[:3, 3] = p[:3]
    return transform, True


def world_arm_to_waist_head_position(
    arm_world: np.ndarray,
    head_world: np.ndarray,
) -> np.ndarray:
    """Current head position to waist; no yaw, freeze, anchor, or gain."""

    transform = np.asarray(arm_world, dtype=np.float64).reshape(4, 4).copy()
    head = np.asarray(head_world, dtype=np.float64).reshape(4, 4)
    transform[:3, 3] = (
        transform[:3, 3] - head[:3, 3] + WAIST_OFFSET_XYZ
    )
    return transform


def controller_waist_targets(
    left_pose7: np.ndarray | list[float],
    right_pose7: np.ndarray | list[float],
    head_pose7: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """PICO raw controller poses to absolute waist-frame wrist SE(3)."""

    head_xr, head_ok = _pico_pose_to_mat_openxr(head_pose7)
    if not head_ok:
        head_xr = CONST_HEAD_POSE.copy()
    left_xr, left_ok = _pico_pose_to_mat_openxr(left_pose7)
    if not left_ok:
        left_xr = CONST_LEFT_ARM_POSE.copy()
    right_xr, right_ok = _pico_pose_to_mat_openxr(right_pose7)
    if not right_ok:
        right_xr = CONST_RIGHT_ARM_POSE.copy()

    head_robot = T_ROBOT_OPENXR @ head_xr @ T_OPENXR_ROBOT
    left_robot = T_ROBOT_OPENXR @ left_xr @ T_OPENXR_ROBOT
    right_robot = T_ROBOT_OPENXR @ right_xr @ T_OPENXR_ROBOT
    left_waist = world_arm_to_waist_head_position(left_robot, head_robot)
    right_waist = world_arm_to_waist_head_position(right_robot, head_robot)
    return left_waist, right_waist, bool(head_ok and left_ok and right_ok)


@dataclass
class AlignmentGate:
    ctrl_fwd_tol_rad: float = np.deg2rad(30.0)
    ctrl_speed_tol_mps: float = 0.05
    hold_frames: int = 25

    def check(
        self,
        left_waist: np.ndarray,
        right_waist: np.ndarray,
        left_velocity: np.ndarray,
        right_velocity: np.ndarray,
    ) -> tuple[bool, str]:
        for name, transform in (("L", left_waist), ("R", right_waist)):
            angle = np.arccos(
                np.clip(
                    np.asarray(transform, dtype=np.float64)[:3, 0]
                    @ np.array([1.0, 0.0, 0.0]),
                    -1.0,
                    1.0,
                )
            )
            if angle > self.ctrl_fwd_tol_rad:
                return False, f"ctrl_{name}_not_forward"
        if max(np.linalg.norm(left_velocity), np.linalg.norm(right_velocity)) > (
            self.ctrl_speed_tol_mps
        ):
            return False, "operator_moving"
        return True, "ok"


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
        human_joints_info_path=str(HUMAN_JOINTS_INFO_PATH),
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    # The real G1 LCM bridge exposes a binary grip flag. Treat either analog
    # controller input as a close request so the grip button is not silently
    # ignored and use the same middle-finger target that maps to hand joint 3.
    if max(float(trigger), float(grip)) > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    global_quats = body_poses_np[:, [6, 3, 4, 5]]

    # Convert to local rotations
    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


class PicoButtonEventPublisher:
    """Poll controller input independently from body/FK work.

    ABXY gestures are emitted as discrete events.  Raw thumbstick state is
    published at the polling rate so the PATH state machine owns all safety
    limits and the q-only locomotion gate.
    """

    def __init__(self, *, port: int = 5565, poll_hz: float = 100.0):
        self.port = int(port)
        self.poll_hz = max(20.0, float(poll_hz))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = (False, False, False, False)
        self._sequence = 0
        self._controller_sequence = 0
        self._thread = threading.Thread(
            target=self._run, name="pico-abxy-poller", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> tuple[bool, bool, bool, bool]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 32)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://*:{self.port}")
        print(
            f"[PicoButtons] polling ABXY at {self.poll_hz:.0f}Hz; "
            f"publishing tcp://*:{self.port}/pico_button_event + "
            "pico_controller_state",
            flush=True,
        )
        gestures = PicoButtonGestureInterpreter()
        period = 1.0 / self.poll_hz
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                buttons = get_abxy_buttons()
                lx, ly, rx, ry = get_controller_axes()
                left_axis_click, right_axis_click = get_axis_clicks()
                # Keep analogue hand controls on this dedicated 100 Hz path.
                # vr_pose_debug is conflated downstream for Cartesian arm
                # tracking, so a short trigger press must not depend on which
                # body-pose frame happens to survive that conflation.
                (
                    _,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs()
                read_timestamp_us = time.monotonic_ns() // 1000
                with self._lock:
                    self._latest = buttons
                a, b, x, y = buttons
                self._controller_sequence += 1
                controller_payload = {
                    "schema": "path.pico_controller_state.v1",
                    "sequence": self._controller_sequence,
                    "controller_read_timestamp_us": int(read_timestamp_us),
                    "publish_timestamp_us": int(time.monotonic_ns() // 1000),
                    "left_axis": [float(lx), float(ly)],
                    "right_axis": [float(rx), float(ry)],
                    "left_axis_click": bool(left_axis_click),
                    "right_axis_click": bool(right_axis_click),
                    "left_trigger": float(left_trigger),
                    "right_trigger": float(right_trigger),
                    "left_grip": float(left_grip),
                    "right_grip": float(right_grip),
                    "a": bool(a),
                    "b": bool(b),
                    "x": bool(x),
                    "y": bool(y),
                }
                socket.send(
                    b"pico_controller_state"
                    + msgpack.packb(controller_payload, use_bin_type=True)
                )

                events = gestures.update(buttons)
                for event in events:
                    self._sequence += 1
                    payload = {
                        "schema": "path.pico_button_event.v1",
                        "event": event,
                        "sequence": self._sequence,
                        "button_read_timestamp_us": int(read_timestamp_us),
                        "publish_timestamp_us": int(time.monotonic_ns() // 1000),
                        "a": bool(a),
                        "b": bool(b),
                        "x": bool(x),
                        "y": bool(y),
                    }
                    socket.send(
                        b"pico_button_event"
                        + msgpack.packb(payload, use_bin_type=True)
                    )
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                elif delay < -period:
                    deadline = time.monotonic()
        finally:
            socket.close()
            context.term()


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two quaternions and renormalize. Input shape (4,), xyzw order.
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Shared pose streaming loop used by run_pico."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )

    # Create reader and start it
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = LivePoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()


class ThreePointPose:
    """Legacy SMPL visualization plus a tracking-session counter.

    Arm targets never pass through this class.
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis

        self._robot_model = robot_model
        if enable_vis_vr3pt and self._robot_model is None:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            self._robot_model = instantiate_g1_robot_model()
            print(f"[{log_prefix}] Robot model loaded for visualization")

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        self._calibration_lock = threading.RLock()
        self._calibration_revision = 0

    @property
    def is_pending(self) -> bool:
        return False

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def calibration_revision(self) -> int:
        with self._calibration_lock:
            return int(self._calibration_revision)

    def start_tracking_session(self) -> int:
        with self._calibration_lock:
            self._calibration_revision += 1
            return int(self._calibration_revision)

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        vr_3pt_pose = _process_3pt_pose(smpl_pose_np)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def debug_pose(
        self, body_poses_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return legacy body diagnostics; never used as an arm target."""

        raw = _process_3pt_pose(body_poses_np)
        return raw, raw.copy(), True


class PicoVrPoseDebugPublisher:
    """Publish absolute PICO controller targets in the waist frame."""

    def __init__(
        self,
        *,
        reader: PicoReader,
        three_point: ThreePointPose,
        port: int = 5566,
        publish_hz: float = 90.0,
        stream_mode_getter=None,
    ):
        self.reader = reader
        self.three_point = three_point
        self.port = int(port)
        self.publish_hz = min(90.0, max(50.0, float(publish_hz)))
        self.stream_mode_getter = stream_mode_getter
        self.arm_reference_mode = "head_position"
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self._align_ok = False
        self._align_reason = "inactive"
        self._align_ok_frames = 0
        self._align_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="pico-vr-pose-publisher", daemon=True
        )

    @property
    def alignment_ready(self) -> bool:
        # Absolute controller-to-waist targets have no zero-pose calibration.
        # This remains only as a compatibility status property for old callers.
        return True

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://*:{self.port}")
        period = 1.0 / self.publish_hz
        next_deadline = time.monotonic()
        last_timestamp_ns = -1
        previous_left_position: np.ndarray | None = None
        previous_right_position: np.ndarray | None = None
        previous_time = time.monotonic()
        last_alignment_log = 0.0
        print(
            f"[PICO-VR] publishing tcp://*:{self.port}/vr_pose_debug "
            f"at up to {self.publish_hz:.0f}Hz "
            "(absolute controller head-position -> waist targets)"
        )
        try:
            while not self._stop.is_set():
                sample = self.reader.get_latest()
                timestamp_ns = int(sample.get("timestamp_ns", 0)) if sample else 0
                if sample is not None and timestamp_ns > 0 and timestamp_ns != last_timestamp_ns:
                    try:
                        mode = (
                            self.stream_mode_getter()
                            if self.stream_mode_getter is not None
                            else StreamMode.OFF
                        )
                        active = mode in (StreamMode.ALIGNING, StreamMode.TRACKING)
                        left_waist = np.full((4, 4), np.nan, dtype=np.float64)
                        right_waist = np.full((4, 4), np.nan, dtype=np.float64)
                        targets_ok = False
                        if active:
                            left_waist, right_waist, targets_ok = (
                                controller_waist_targets(
                                    xrt.get_left_controller_pose(),
                                    xrt.get_right_controller_pose(),
                                    xrt.get_headset_pose(),
                                )
                            )

                        now = time.monotonic()
                        dt = max(now - previous_time, 1.0e-6)
                        if (
                            targets_ok
                            and previous_left_position is not None
                            and previous_right_position is not None
                        ):
                            left_velocity = (
                                left_waist[:3, 3] - previous_left_position
                            ) / dt
                            right_velocity = (
                                right_waist[:3, 3] - previous_right_position
                            ) / dt
                        else:
                            left_velocity = np.full(3, np.inf)
                            right_velocity = np.full(3, np.inf)

                        # Do not gate absolute PICO targets on a hand-forward
                        # pose or a stationary 25-frame initialization window.
                        # The only remaining gate is finite raw tracking data.
                        align_ok = bool(targets_ok and active)
                        align_reason = "bypassed" if align_ok else "inactive"

                        with self._align_lock:
                            self._align_ok_frames = 1 if align_ok else 0
                            self._align_ok = align_ok
                            self._align_reason = align_reason
                            align_ok_frames = self._align_ok_frames

                        if active and now - last_alignment_log >= 1.0:
                            max_speed = max(
                                float(np.linalg.norm(left_velocity)),
                                float(np.linalg.norm(right_velocity)),
                            )
                            print(
                                "[PICO-VR] absolute-target stream "
                                f"reason={align_reason} "
                                f"valid={align_ok_frames} "
                                f"controller_speed={max_speed:.3f}m/s"
                            )
                            last_alignment_log = now

                        if targets_ok:
                            previous_left_position = left_waist[:3, 3].copy()
                            previous_right_position = right_waist[:3, 3].copy()
                            previous_time = now

                        (
                            _,
                            left_trigger,
                            right_trigger,
                            left_grip,
                            right_grip,
                        ) = get_controller_inputs()
                        left_hand_joints, right_hand_joints = (
                            compute_hand_joints_from_inputs(
                                self.left_hand_ik_solver,
                                self.right_hand_ik_solver,
                                left_trigger,
                                left_grip,
                                right_trigger,
                                right_grip,
                            )
                        )
                        socket.send(
                            pack_pose_message(
                                {
                                    "wrist_left_waist": (
                                        left_waist.reshape(-1).astype(np.float32)
                                        if targets_ok
                                        else np.full(16, np.nan, dtype=np.float32)
                                    ),
                                    "wrist_right_waist": (
                                        right_waist.reshape(-1).astype(np.float32)
                                        if targets_ok
                                        else np.full(16, np.nan, dtype=np.float32)
                                    ),
                                    "targets_valid": np.array(
                                        [targets_ok and active], dtype=bool
                                    ),
                                    "frame_index": np.array(
                                        [timestamp_ns], dtype=np.int64
                                    ),
                                    "timestamp_monotonic": np.array(
                                        [sample.get("timestamp_monotonic", 0.0)],
                                        dtype=np.float64,
                                    ),
                                    "pico_dt": np.array(
                                        [sample.get("dt", 0.0)], dtype=np.float32
                                    ),
                                    "pico_fps": np.array(
                                        [sample.get("fps", 0.0)], dtype=np.float32
                                    ),
                                    "ee_frame_offset_m": np.array(
                                        [0.05], dtype=np.float32
                                    ),
                                    "stream_mode": np.array(
                                        [mode.value], dtype=np.int32
                                    ),
                                    "session_id": np.array(
                                        [self.three_point.calibration_revision],
                                        dtype=np.int32,
                                    ),
                                    "arm_reference_mode": np.array(
                                        [0], dtype=np.int32
                                    ),
                                    "align_ok": np.array(
                                        [align_ok], dtype=bool
                                    ),
                                    "align_reason": np.array(
                                        [
                                            {
                                                "ok": 0,
                                                "tracking": 1,
                                                "inactive": 2,
                                                "operator_moving": 3,
                                                "ctrl_L_not_forward": 4,
                                                "ctrl_R_not_forward": 5,
                                            }.get(align_reason, 6)
                                        ],
                                        dtype=np.int32,
                                    ),
                                    "left_trigger": np.array(
                                        [left_trigger], dtype=np.float32
                                    ),
                                    "right_trigger": np.array(
                                        [right_trigger], dtype=np.float32
                                    ),
                                    "left_grip": np.array(
                                        [left_grip], dtype=np.float32
                                    ),
                                    "right_grip": np.array(
                                        [right_grip], dtype=np.float32
                                    ),
                                    "left_hand_joints": left_hand_joints.reshape(
                                        -1
                                    ).astype(np.float32),
                                    "right_hand_joints": right_hand_joints.reshape(
                                        -1
                                    ).astype(np.float32),
                                },
                                topic="vr_pose_debug",
                            )
                        )
                        last_timestamp_ns = timestamp_ns
                    except Exception as exc:
                        print(f"[PICO-VR] skipped frame: {exc}")
                next_deadline += period
                sleep_sec = next_deadline - time.monotonic()
                if sleep_sec > 0.0:
                    self._stop.wait(sleep_sec)
                else:
                    next_deadline = time.monotonic()
        finally:
            socket.close()
            context.term()


class LivePoseStreamer:
    """Streams live Pico poses after the normal SMPL preprocessing pipeline."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        log_prefix: str = "PoseLoop",
    ):
        self.socket = socket
        self.reader = reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix

        # Injected dependencies
        self.reader = reader
        self.three_point = three_point

        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.parent_indices = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        # NOTE: Sleep budget set to 95% of the ideal frame period so that the actual
        # FPS lands closer to target_fps despite per-frame processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()

        # Data collection button state tracking (edge-triggered)
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False

        self.buffer_cleared = (
            True  # Start with buffer cleared - wait for full buffer before first send
        )
        self.yaw_accumulator = YawAccumulator()

    def reset_yaw(self):
        """Called when entering a legacy pose mode. Resets yaw only."""
        self.yaw_accumulator.reset()

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()

        if sample is None:
            time.sleep(0.005)
            return

        latest_data = compute_from_body_poses(
            self.parent_indices, self.device, sample["body_poses_np"]
        )
        (left_menu_button, left_trigger, right_trigger, left_grip, right_grip) = (
            get_controller_inputs()
        )
        # Get A and B button states for data collection control
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

        # Data collection toggle logic (edge-triggered)
        # Left grip + A = toggle_data_collection
        # Left grip + B = toggle_data_abort
        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5

        # Detect rising edge
        toggle_data_collection = toggle_data_collection_tmp and not self.toggle_data_collection_last
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        )
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(
            np.float32
        )
        use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(
            np.float32
        )
        N = len(self.frame_buffer["frame_index"])

        ##### From @Jiefeng for directly setting the joint position ######
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX = 17
        SMPL_L_WRIST_IDX = 19
        SMPL_R_ELBOW_IDX = 18
        SMPL_R_WRIST_IDX = 20

        # G1_L_ELBOW_IDX = 0
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27

        # G1_R_ELBOW_IDX = 0
        G1_R_WRIST_ROLL_IDX = 24  # Done
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(
            smpl_l_elbow_aa, g1_l_elbow_axis
        )

        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(
            smpl_r_elbow_aa, g1_r_elbow_axis
        )

        # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL
        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )

        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
        g1_l_wrist_pitch = -l_wrist_euler[:, 1]
        g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

        g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
        g1_r_wrist_pitch = -r_wrist_euler[:, 1]
        g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

        joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

        # Process SMPL pose to get calibrated 3-point VR pose and update visualization
        # Pass SMPL local joints for optional body visualization in the VR3Pt viewer
        smpl_joints_for_vis = (
            latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            if self.three_point.enable_smpl_vis
            else None
        )
        vr_3pt_pose_raw = _process_3pt_pose(sample["body_poses_np"])
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )
        ##### From @Jiefeng for directly setting the joint position ######

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        N = len(self.frame_buffer["frame_index"])

        # Wait for buffer to be completely filled before sending first message after clearing
        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            # Buffer is now full with fresh data, can start sending
            self.buffer_cleared = False

        # Get joystick axes for yaw accumulation
        _, _, rx, _ = get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        # Only send if buffer is full and we're not waiting for fresh data
        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "vr_position_raw": vr_3pt_pose_raw[:, :3].flatten(),
                "vr_orientation_raw": vr_3pt_pose_raw[:, 3:].flatten(),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }

            packed_message = pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


class RecordedPoseStreamer:
    """Replays pose protocol-v3 windows captured by record_sonic_motion_triplet.py."""

    _FRAME_RE = re.compile(r"frame_(\d+)\.npz$")
    _REQUIRED_FIELDS = {
        "smpl_pose": (3, (21, 3)),
        "smpl_joints": (3, (24, 3)),
        "body_quat_w": (2, (4,)),
        "joint_pos": (2, (29,)),
        "joint_vel": (2, (29,)),
        "frame_index": (1, ()),
    }
    _OPTIONAL_FIELDS = (
        "vr_position",
        "vr_orientation",
        "vr_position_raw",
        "vr_orientation_raw",
        "vr_compliance",
        "left_trigger",
        "right_trigger",
        "left_grip",
        "right_grip",
        "pico_dt",
        "pico_fps",
        "left_hand_joints",
        "right_hand_joints",
        "heading_increment",
    )

    def __init__(
        self,
        socket: zmq.Socket,
        session_dir: str,
        target_fps: float,
        playback_speed: float = 1.0,
        timing: str = "fixed",
        loop: bool = False,
        max_recorded_gap_sec: float = 0.1,
    ):
        if target_fps <= 0.0:
            raise ValueError("replay target FPS must be positive")
        if playback_speed <= 0.0:
            raise ValueError("replay playback speed must be positive")
        if timing not in {"fixed", "recorded"}:
            raise ValueError("replay timing must be 'fixed' or 'recorded'")

        self.socket = socket
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.target_fps = float(target_fps)
        self.playback_speed = float(playback_speed)
        self.timing = timing
        self.loop = loop
        self.max_recorded_gap_sec = max(0.0, float(max_recorded_gap_sec))
        self.entries = self._discover_entries()
        self.cursor = 0
        self.loop_count = 0
        self.wire_last_index = -1
        self.previous_source_last = None
        self.previous_record_time = None
        self.next_deadline = None

        first = self._load_pose_data(self.entries[0][0])
        self.window_size = int(first["frame_index"].shape[0])
        print(
            f"[Replay] Loaded {len(self.entries)} unique pose windows from "
            f"{self.session_dir} (window={self.window_size}, timing={self.timing}, "
            f"fps={self.target_fps:g}, speed={self.playback_speed:g})"
        )

    @staticmethod
    def _frame_sort_key(path: Path) -> int:
        match = RecordedPoseStreamer._FRAME_RE.fullmatch(path.name)
        return int(match.group(1)) if match else -1

    def _discover_entries(self) -> list[tuple[Path, int, float]]:
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"Replay session directory not found: {self.session_dir}")
        paths = sorted(
            (p for p in self.session_dir.glob("frame_*.npz") if self._FRAME_RE.fullmatch(p.name)),
            key=self._frame_sort_key,
        )
        if not paths:
            raise FileNotFoundError(f"No frame_*.npz files in {self.session_dir}")

        entries: list[tuple[Path, int, float]] = []
        duplicate_count = 0
        previous_last = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if "pose_frame_index" not in data:
                    continue
                indices = np.asarray(data["pose_frame_index"]).reshape(-1)
                if not len(indices):
                    continue
                source_last = int(indices[-1])
                if source_last == previous_last:
                    duplicate_count += 1
                    continue
                record_time = float(
                    np.asarray(data.get("record_monotonic_time", [np.nan])).reshape(-1)[0]
                )
                entries.append((path, source_last, record_time))
                previous_last = source_last

        if not entries:
            raise ValueError(f"No recorded pose windows found in {self.session_dir}")
        if duplicate_count:
            print(f"[Replay] Removed {duplicate_count} duplicate recorder ticks")
        return entries

    def _load_pose_data(self, path: Path) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as recorded:
            for name in self._REQUIRED_FIELDS:
                key = f"pose_{name}"
                if key not in recorded:
                    raise ValueError(f"{path.name}: missing required field '{key}'")
                output[name] = np.asarray(recorded[key]).copy()
            for name in self._OPTIONAL_FIELDS:
                key = f"pose_{name}"
                if key in recorded:
                    output[name] = np.asarray(recorded[key]).copy()

        frame_count = int(output["frame_index"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"{path.name}: empty frame_index")
        for name, (ndim, trailing_shape) in self._REQUIRED_FIELDS.items():
            value = output[name]
            if value.ndim != ndim or tuple(value.shape[1:]) != trailing_shape:
                raise ValueError(
                    f"{path.name}: invalid {name} shape {value.shape}; "
                    f"expected [N{',' if trailing_shape else ''}"
                    f"{','.join(map(str, trailing_shape))}]"
                )
            if value.shape[0] != frame_count:
                raise ValueError(
                    f"{path.name}: {name} has {value.shape[0]} frames, expected {frame_count}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{path.name}: {name} contains non-finite values")

        output["toggle_data_collection"] = np.array([False], dtype=bool)
        output["toggle_data_abort"] = np.array([False], dtype=bool)
        output["timestamp_realtime"] = np.array([time.time()], dtype=np.float64)
        output["timestamp_monotonic"] = np.array([time.monotonic()], dtype=np.float64)
        return output

    def _interval_for(self, source_last: int, record_time: float) -> float:
        source_step = 1
        if self.previous_source_last is not None and source_last > self.previous_source_last:
            source_step = source_last - self.previous_source_last
        fixed_interval = source_step / self.target_fps
        if self.timing != "recorded" or self.previous_record_time is None:
            return fixed_interval / self.playback_speed
        interval = record_time - self.previous_record_time
        if not np.isfinite(interval) or interval <= 0.0:
            interval = fixed_interval
        if self.max_recorded_gap_sec > 0.0:
            interval = min(interval, self.max_recorded_gap_sec)
        return interval / self.playback_speed

    def _pace(self, interval: float) -> None:
        now = time.monotonic()
        if self.next_deadline is None:
            self.next_deadline = now
        else:
            self.next_deadline += interval
            if self.next_deadline < now - 0.5:
                self.next_deadline = now
        delay = self.next_deadline - now
        if delay > 0.0:
            time.sleep(delay)

    def _reindex(self, data: dict[str, np.ndarray], source_last: int) -> None:
        frame_count = int(data["frame_index"].shape[0])
        source_step = 1
        if self.previous_source_last is not None and source_last > self.previous_source_last:
            source_step = source_last - self.previous_source_last
        if self.wire_last_index < 0:
            self.wire_last_index = frame_count - 1
        else:
            self.wire_last_index += source_step
        data["frame_index"] = np.arange(
            self.wire_last_index - frame_count + 1,
            self.wire_last_index + 1,
            dtype=np.int64,
        )

    def prime(self, repeat: int = 3, interval_sec: float = 0.05) -> None:
        path, _, _ = self.entries[0]
        data = self._load_pose_data(path)
        data["frame_index"] = np.arange(len(data["frame_index"]), dtype=np.int64)
        packed = pack_pose_message(data, topic="pose", version=3)
        for _ in range(max(1, repeat)):
            self.socket.send(packed)
            time.sleep(max(0.0, interval_sec))

    def send_next(self) -> bool:
        if self.cursor >= len(self.entries):
            if not self.loop:
                return False
            self.cursor = 0
            self.loop_count += 1
            self.previous_source_last = None
            self.previous_record_time = None
            print(f"[Replay] Loop {self.loop_count}")

        path, source_last, record_time = self.entries[self.cursor]
        interval = self._interval_for(source_last, record_time)
        self._pace(interval)
        data = self._load_pose_data(path)
        self._reindex(data, source_last)
        data["timestamp_realtime"] = np.array([time.time()], dtype=np.float64)
        data["timestamp_monotonic"] = np.array([time.monotonic()], dtype=np.float64)
        self.socket.send(pack_pose_message(data, topic="pose", version=3))

        self.previous_source_last = source_last
        self.previous_record_time = record_time
        self.cursor += 1
        return True

    def send_hold(self) -> None:
        path, _, _ = self.entries[-1]
        self._pace(1.0 / (self.target_fps * self.playback_speed))
        data = self._load_pose_data(path)
        self.wire_last_index += 1
        frame_count = len(data["frame_index"])
        data["frame_index"] = np.arange(
            self.wire_last_index - frame_count + 1,
            self.wire_last_index + 1,
            dtype=np.int64,
        )
        data["timestamp_realtime"] = np.array([time.time()], dtype=np.float64)
        data["timestamp_monotonic"] = np.array([time.monotonic()], dtype=np.float64)
        self.socket.send(pack_pose_message(data, topic="pose", version=3))


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """Run Pico body tracking with real-time visualization and ZMQ streaming."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
        )
    finally:
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


# Desk ("hands-on-desk") ready pose, 29-DOF in MuJoCo joint order. MUST stay in
# sync with `ready_angles` in
# gear_sonic_deploy/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp.
# Only the upper-body entries (indices 12-28) are used — they are extracted via
# FeedbackReader.upper_body_joint_indices and sent as the frozen upper_body_position
# while in plain PLANNER mode when --desk-pose is enabled, so the arms hold the desk
# pose until POSE mode takes over.
DESK_POSE_MUJOCO_29 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                          # waist: yaw, roll, pitch
    0.0, 0.15, 0.0, 1.3, 0.0, 0.0, 0.0,     # left arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
    0.0, -0.15, 0.0, 1.3, 0.0, 0.0, 0.0,    # right arm
]


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5557):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="g1_debug")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self):
        """Poll for feedback once, and update internal state."""
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets()
        print("[PlannerLoop] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        data = self.poller.get_data()

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None

        unpacked = msgpack.unpackb(data, raw=False)
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, left_hand_q, right_hand_q, full_body_q


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        lock_waist: bool = False,
        desk_pose: bool = False,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        self.lock_waist = lock_waist
        # When True, hold the arms at the desk ready pose during plain PLANNER mode.
        self.desk_pose = desk_pose
        self._desk_log_counter = 0
        print(
            f"[DESK POSE] PlannerStreamer desk_pose={'ENABLED' if desk_pose else 'disabled'}"
        )
        # Current locomotion mode, default IDLE
        self.mode = LocomotionMode.IDLE
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self.last_xrt_timestamp = None

        # Hand IK solvers for trigger-controlled hand open/close in VR 3PT mode
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback()

    def run_once(self, stream_mode: StreamMode):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # A+B => next mode; X+Y => previous mode (rising edges)
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            ab_now = bool(a_pressed) and bool(b_pressed)
            xy_now = bool(x_pressed) and bool(y_pressed)
            if ab_now and not self.prev_ab:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            if xy_now and not self.prev_xy:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            self.prev_ab = ab_now
            self.prev_xy = xy_now

            # Read axes/joysticks to control movement, facing, speed and mode
            lx, ly, rx, ry = get_controller_axes()

            # Facing from RIGHT stick: continuous yaw based on rx (right = turn right, left = turn left)
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = np.hypot(lx, ly)
            raw_mag = np.clip(raw_mag, 0.0, 1.0)
            if np.abs(raw_mag) < JOYSTICK_DEADZONE:
                mag = 0.0
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                if mag > 1.0:
                    mag = 1.0
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag  # 1.5 .. 4.5
                else:
                    speed = mag  # default 0 .. 1.0

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local

            movement = [movement_global[0], movement_global[1], 0.0]

            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target
            elif self.desk_pose and stream_mode == StreamMode.PLANNER:
                # Desk pose: hold the arms at the hands-on-desk ready pose during plain
                # PLANNER so they don't relax before POSE mode (A+X) takes over. Ordered
                # to match FeedbackReader.upper_body_joint_indices (MuJoCo order).
                upper_body_position = np.array(
                    [DESK_POSE_MUJOCO_29[i] for i in self.feedback_reader.upper_body_joint_indices],
                    dtype=np.float32,
                )
                self._desk_log_counter += 1
                if self._desk_log_counter % 20 == 1:  # ~once/sec at 20 Hz
                    print(
                        "[DESK POSE] PLANNER: sending desk upper_body_position "
                        f"(elbow L/R={upper_body_position[9]:.2f}/{upper_body_position[10]:.2f})"
                    )

            if self.lock_waist and upper_body_position is not None:
                upper_body_position = np.asarray(upper_body_position, dtype=np.float32).copy()
                upper_body_position[:3] = 0.0  # lock waist position to zero
            
            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    print("[PlannerLoop] Sending VR 3-point pose as target")
                    vr_3pt_pose = self.three_point.process_smpl_pose(sample["body_poses_np"])
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                # Compute hand joints from trigger/grip inputs so operator can
                # control hand open/close while in VR 3PT mode
                (
                    left_menu_button,
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = get_controller_inputs()
                lh_joints, rh_joints = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    left_trigger,
                    left_grip,
                    right_trigger,
                    right_grip,
                )
                left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()


def _send_manager_state(socket: zmq.Socket, mode: StreamMode) -> None:
    socket.send(
        pack_pose_message(
            {
                "stream_mode": np.array([mode.value], dtype=np.int32),
                "toggle_data_collection": np.array([False], dtype=bool),
                "toggle_data_abort": np.array([False], dtype=bool),
            },
            topic="manager_state",
        )
    )


def _wait_for_replay_subscribers(socket: zmq.Socket, timeout_sec: float) -> None:
    """Wait until SONIC has subscribed to every topic needed for the handshake."""
    if timeout_sec <= 0.0:
        print("[ReplayManager] Subscriber readiness wait disabled")
        return
    required = {"pose", "command", "planner"}
    subscribed: set[str] = set()
    deadline = time.monotonic() + timeout_sec
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(
        f"[ReplayManager] Waiting up to {timeout_sec:.1f}s for SONIC subscriptions: "
        f"{sorted(required)}"
    )
    while required - subscribed:
        remaining_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            missing = sorted(required - subscribed)
            raise TimeoutError(
                f"SONIC did not subscribe to replay topics before timeout: {missing}. "
                "Start g1_deploy_onnx_ref with --input-type zmq_manager."
            )
        events = dict(poller.poll(min(remaining_ms, 500)))
        if socket not in events:
            continue
        event = socket.recv()
        if not event:
            continue
        is_subscribe = event[0] == 1
        topic = event[1:].decode("utf-8", errors="replace")
        if is_subscribe:
            subscribed.add(topic)
        else:
            subscribed.discard(topic)
        print(f"[ReplayManager] subscriptions ready: {sorted(required & subscribed)}")
    print("[ReplayManager] SONIC pose/command/planner subscribers are ready")


def run_recorded_pose_manager(
    replay_session: str,
    port: int = 5556,
    target_fps: float = 50.0,
    replay_speed: float = 1.0,
    replay_timing: str = "fixed",
    replay_loop: bool = False,
    replay_subscriber_timeout: float = 120.0,
    replay_start_delay: float = 2.0,
    replay_alignment_delay: float = 6.0,
    replay_max_recorded_gap_sec: float = 0.1,
    replay_hold_last: bool = False,
    replay_resume_control: bool = False,
    replay_stop_at_end: bool = False,
) -> None:
    """Own the manager PUB socket and replay captured pose windows deterministically."""
    context = zmq.Context()
    # XPUB remains wire-compatible with SUB clients and exposes subscription
    # events, preventing one-shot start/mode commands from being lost to slow joiners.
    socket = context.socket(zmq.XPUB)
    socket.setsockopt(zmq.XPUB_VERBOSE, 1)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://*:{port}"
    try:
        try:
            socket.bind(endpoint)
        except zmq.ZMQError as exc:
            if exc.errno == zmq.EADDRINUSE:
                raise RuntimeError(
                    f"Cannot bind replay manager to {endpoint}: address already in use. "
                    "Stop the live Pico manager or any other publisher using this port."
                ) from exc
            raise RuntimeError(f"Cannot bind replay manager to {endpoint}: {exc}") from exc

        streamer = RecordedPoseStreamer(
            socket=socket,
            session_dir=replay_session,
            target_fps=target_fps,
            playback_speed=replay_speed,
            timing=replay_timing,
            loop=replay_loop,
            max_recorded_gap_sec=replay_max_recorded_gap_sec,
        )
        print(f"[ReplayManager] ZMQ PUB bound to {endpoint}")
        _wait_for_replay_subscribers(socket, replay_subscriber_timeout)
        print(
            f"[ReplayManager] Waiting {replay_start_delay:.2f}s for SONIC, GMR, "
            "and recorder subscribers"
        )
        time.sleep(max(0.0, replay_start_delay))

        idle_planner = build_planner_message(
            LocomotionMode.IDLE.value,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            speed=-1.0,
            height=-1.0,
        )

        if replay_resume_control:
            print(
                "[ReplayManager] Resuming existing CONTROL state; skipping the "
                "A+B+X+Y start/alignment step"
            )
        else:
            # Mirror A+B+X+Y: OFF -> PLANNER and initialize CONTROL/alignment.
            for _ in range(5):
                socket.send(idle_planner)
                socket.send(build_command_message(start=True, stop=False, planner=True))
                _send_manager_state(socket, StreamMode.PLANNER)
                time.sleep(0.1)

            print(
                f"[ReplayManager] PLANNER control started; waiting "
                f"{replay_alignment_delay:.2f}s for initialization/alignment"
            )
            alignment_deadline = time.monotonic() + max(0.0, replay_alignment_delay)
            while time.monotonic() < alignment_deadline:
                socket.send(idle_planner)
                _send_manager_state(socket, StreamMode.PLANNER)
                time.sleep(0.05)

        print("[ReplayManager] Switching PLANNER -> STREAMED_MOTION")
        for _ in range(5):
            socket.send(build_command_message(
                start=not replay_resume_control, stop=False, planner=False
            ))
            _send_manager_state(socket, StreamMode.POSE)
            time.sleep(0.1)

        # Prime after the mode-switch safety reset has cleared the pose merger.
        streamer.prime(repeat=3, interval_sec=0.05)
        time.sleep(0.1)

        print("[ReplayManager] Playback started in STREAMED_MOTION mode")
        sent = 0
        while streamer.send_next():
            _send_manager_state(socket, StreamMode.POSE)
            sent += 1
            if sent == 1 or sent % max(1, int(target_fps * 5)) == 0:
                print(
                    f"[ReplayManager] sent={sent}/{len(streamer.entries)} "
                    f"loop={streamer.loop_count} wire_frame={streamer.wire_last_index}"
                )

        if replay_hold_last:
            print("[ReplayManager] Playback complete; holding the final pose")
            while True:
                streamer.send_hold()
                _send_manager_state(socket, StreamMode.POSE)
        elif replay_stop_at_end:
            print("[ReplayManager] Playback complete; sending STOP")
            for _ in range(3):
                socket.send(build_command_message(start=False, stop=True, planner=False))
                _send_manager_state(socket, StreamMode.OFF)
                time.sleep(0.05)
        else:
            # Mirror A+X at the end: POSE -> PLANNER without toggling CONTROL off.
            print("[ReplayManager] Playback complete; returning to PLANNER (policy stays active)")
            for _ in range(5):
                socket.send(idle_planner)
                socket.send(build_command_message(start=False, stop=False, planner=True))
                _send_manager_state(socket, StreamMode.PLANNER)
                time.sleep(0.05)
            planner_deadline = time.monotonic() + max(0.0, replay_alignment_delay)
            while time.monotonic() < planner_deadline:
                socket.send(idle_planner)
                _send_manager_state(socket, StreamMode.PLANNER)
                time.sleep(0.05)
            print(
                "[ReplayManager] PLANNER is active. Use --replay_resume_control "
                "for the next replay without another start toggle."
            )
    except KeyboardInterrupt:
        if replay_stop_at_end:
            print("\n[ReplayManager] Interrupted; sending STOP")
            for _ in range(3):
                try:
                    socket.send(build_command_message(start=False, stop=True, planner=False))
                except zmq.ZMQError:
                    break
                time.sleep(0.05)
        else:
            print("\n[ReplayManager] Interrupted; leaving policy state unchanged")
    finally:
        socket.close()
        context.term()
        print("[ReplayManager] Shutdown complete")


def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    lock_waist: bool = False,
    desk_pose: bool = False,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    replay_session: str = "",
    replay_speed: float = 1.0,
    replay_timing: str = "fixed",
    replay_loop: bool = False,
    replay_subscriber_timeout: float = 120.0,
    replay_start_delay: float = 2.0,
    replay_alignment_delay: float = 6.0,
    replay_max_recorded_gap_sec: float = 0.1,
    replay_hold_last: bool = False,
    replay_resume_control: bool = False,
    replay_stop_at_end: bool = False,
    button_event_port: int = 5565,
    button_poll_hz: float = 100.0,
    vr_pose_debug_port: int = 5566,
    vr_pose_debug_hz: float = 90.0,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Controller input:
      A+X: PATH tracking arm/disarm only
      A+B+X+Y: immediately start/stop absolute PICO wrist targets
    """
    if replay_session:
        if enable_vis_vr3pt or enable_smpl_vis:
            print("[ReplayManager] Live VR visualization options are ignored in replay mode")
        return run_recorded_pose_manager(
            replay_session=replay_session,
            port=port,
            target_fps=target_fps,
            replay_speed=replay_speed,
            replay_timing=replay_timing,
            replay_loop=replay_loop,
            replay_subscriber_timeout=replay_subscriber_timeout,
            replay_start_delay=replay_start_delay,
            replay_alignment_delay=replay_alignment_delay,
            replay_max_recorded_gap_sec=replay_max_recorded_gap_sec,
            replay_hold_last=replay_hold_last,
            replay_resume_control=replay_resume_control,
            replay_stop_at_end=replay_stop_at_end,
        )

    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    # Create shared reader and 3-point pose processor
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()
    button_publisher = PicoButtonEventPublisher(
        port=button_event_port, poll_hz=button_poll_hz
    )
    button_publisher.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    pose_streamer = LivePoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        lock_waist=lock_waist,
        desk_pose=desk_pose,
    )

    print(
        "Manager controls: A+B+X+Y=immediate PICO target start/stop; "
        "A+X=PATH tracking arm/disarm only"
    )
    current_mode = StreamMode.OFF
    vr_pose_publisher = PicoVrPoseDebugPublisher(
        reader=reader,
        three_point=three_point,
        port=vr_pose_debug_port,
        publish_hz=vr_pose_debug_hz,
        stream_mode_getter=lambda: current_mode,
    )
    vr_pose_publisher.start()
    prev_toggle_dc = False
    prev_toggle_da = False
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = button_publisher.latest()

            left_menu_button, _, _, left_grip_mgr, _ = get_controller_inputs()

            left_axis_click, _ = get_axis_clicks()

            ax_pressed = (a_pressed) and (x_pressed)
            by_pressed = (b_pressed) and (y_pressed)
            start_combo = (a_pressed) and (b_pressed) and (x_pressed) and (y_pressed)

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    three_point.start_tracking_session()
                    new_mode = StreamMode.TRACKING
            elif current_mode == StreamMode.TRACKING:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if new_mode == StreamMode.TRACKING:
                    planner_streamer.reset_yaw()

            # Run one iteration of the new mode
            if new_mode in (StreamMode.ALIGNING, StreamMode.TRACKING):
                planner_streamer.run_once(StreamMode.PLANNER)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                elif new_mode in (StreamMode.ALIGNING, StreamMode.TRACKING):
                    socket.send(build_command_message(start=True, stop=False, planner=True))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            # Mode-independent: send manager_state for data exporter
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp
            socket.send(
                pack_pose_message(
                    {
                        "stream_mode": np.array([current_mode.value], dtype=np.int32),
                        "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                        "toggle_data_abort": np.array([toggle_da], dtype=bool),
                    },
                    topic="manager_state",
                )
            )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        vr_pose_publisher.stop()
        reader.stop()
        button_publisher.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--button_event_port",
        type=int,
        default=5565,
        help="Dedicated ABXY event PUB port (default: 5565)",
    )
    parser.add_argument(
        "--button_poll_hz",
        type=float,
        default=100.0,
        help="Dedicated ABXY polling rate (default: 100Hz)",
    )
    parser.add_argument(
        "--vr_pose_debug_port",
        type=int,
        default=5566,
        help="Independent calibrated PICO wrist PUB port (default: 5566)",
    )
    parser.add_argument(
        "--vr_pose_debug_hz",
        type=float,
        default=90.0,
        help="Independent calibrated PICO wrist publish rate, clamped to 50-90Hz",
    )
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--replay_session",
        type=str,
        default="",
        help="Replay a record_sonic_motion_triplet frame_*.npz directory instead of live Pico",
    )
    parser.add_argument(
        "--replay_speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--replay_timing",
        choices=("fixed", "recorded"),
        default="fixed",
        help="Use fixed target FPS or recorded monotonic intervals (default: fixed)",
    )
    parser.add_argument(
        "--replay_loop",
        action="store_true",
        help="Loop recorded motion while keeping wire frame indices monotonic",
    )
    parser.add_argument(
        "--replay_subscriber_timeout",
        type=float,
        default=120.0,
        help="Wait for SONIC pose/command/planner subscriptions before handshake (default: 120s; 0 disables)",
    )
    parser.add_argument(
        "--replay_start_delay",
        type=float,
        default=2.0,
        help="Subscriber connection delay before replay starts (default: 2.0s)",
    )
    parser.add_argument(
        "--replay_alignment_delay",
        type=float,
        default=6.0,
        help="Planner initialization/alignment time before switching to replay (default: 6.0s)",
    )
    parser.add_argument(
        "--replay_max_recorded_gap_sec",
        type=float,
        default=0.1,
        help="Maximum pause preserved by recorded timing mode (default: 0.1s)",
    )
    parser.add_argument(
        "--replay_hold_last",
        action="store_true",
        help="Keep publishing the final pose instead of sending STOP",
    )
    parser.add_argument(
        "--replay_resume_control",
        action="store_true",
        help="Assume CONTROL is already active in PLANNER and switch directly to replay",
    )
    parser.add_argument(
        "--replay_stop_at_end",
        action="store_true",
        help="Send STOP when replay ends; default returns to PLANNER and keeps policy active",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--lock_waist",
        action="store_true",
        help="Lock G1 robot waist position to zero (disabled by default for performance)",
    )
    parser.add_argument(
        "--desk-pose",
        "--desk_pose",
        dest="desk_pose",
        action="store_true",
        help="Hold the arms at the hands-on-desk ready pose during plain PLANNER mode "
             "(pairs with the deploy's --desk-pose). Off by default.",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    args = parser.parse_args()

    if args.replay_session and not args.manager:
        parser.error("--replay_session requires --manager")

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            lock_waist=args.lock_waist,
            desk_pose=args.desk_pose,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            replay_session=args.replay_session,
            replay_speed=args.replay_speed,
            replay_timing=args.replay_timing,
            replay_loop=args.replay_loop,
            replay_subscriber_timeout=args.replay_subscriber_timeout,
            replay_start_delay=args.replay_start_delay,
            replay_alignment_delay=args.replay_alignment_delay,
            replay_max_recorded_gap_sec=args.replay_max_recorded_gap_sec,
            replay_hold_last=args.replay_hold_last,
            replay_resume_control=args.replay_resume_control,
            replay_stop_at_end=args.replay_stop_at_end,
            button_event_port=args.button_event_port,
            button_poll_hz=args.button_poll_hz,
            vr_pose_debug_port=args.vr_pose_debug_port,
            vr_pose_debug_hz=args.vr_pose_debug_hz,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
