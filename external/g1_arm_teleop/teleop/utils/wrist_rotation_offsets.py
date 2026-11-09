"""Local wrist-frame mounting corrections, applied to IK targets only."""

import numpy as np

from .ee_kinematics import euler_xyz_to_rotation


def _rpy_degrees(values):
    angles = np.asarray(values, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("wrist offset must contain three finite roll/pitch/yaw angles in degrees")
    return angles


def rotation_offset_from_degrees(rpy_deg):
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), with degree-valued inputs."""
    roll, pitch, yaw = np.deg2rad(_rpy_degrees(rpy_deg))
    return euler_xyz_to_rotation(roll, pitch, yaw)


def apply_wrist_rotation_offset(pose, offset_rotation):
    """Copy a target and postmultiply its orientation in the local wrist frame.

    Translation is unchanged. In the G1 IK model local +X points towards the
    fingers, so an X-only correction rolls the palm about that direction.
    """
    target = np.asarray(pose, dtype=np.float64)
    rotation = np.asarray(offset_rotation, dtype=np.float64)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("wrist target must be a finite 4x4 pose")
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("wrist offset must be a proper 3x3 rotation")
    corrected = target.copy()
    corrected[:3, :3] = target[:3, :3] @ rotation
    return corrected


def wrist_rotation_offset_metadata(left_rpy_deg, right_rpy_deg):
    """Record enough information to reproduce the command-frame correction."""
    return {
        "frame": "local_ik_wrist",
        "angle_unit": "degree",
        "rpy_order": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "application": "R_corrected = R_target @ R_offset; target position unchanged",
        "left_rpy_deg": _rpy_degrees(left_rpy_deg).tolist(),
        "right_rpy_deg": _rpy_degrees(right_rpy_deg).tolist(),
    }
