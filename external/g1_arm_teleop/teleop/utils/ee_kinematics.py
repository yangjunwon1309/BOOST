"""End-effector pose helpers for teleop_ikea.py.

Two different things get recorded and they must not be confused:

  * the IK *target* -- the wrist pose the operator's hand asked for. It exists
    only inside this process for one loop iteration and cannot be recovered from
    anything else, so it has to be written down as it is produced.
  * the IK *solution's* pose -- FK of sol_q. This one is a pure function of the
    joint command that is already being recorded, so strictly it could be
    recomputed offline. It is written anyway because it costs ~50 us and it makes
    the IK residual available per frame, which is the cheapest data-quality
    filter there is: when the two poses disagree, the arm did not go where the
    operator pointed.

The frame comes from the IK model itself rather than a second URDF load, so the
recorded poses are by construction in the same frame the deployed policy's IK
will invert. That frame is: pelvis origin, waist joints held at zero (which makes
it the torso frame translated to the pelvis), end point 5 cm along the wrist_yaw
link's local +x. It is exactly what G1_29_ArmIK registers as 'L_ee' / 'R_ee'.

Nothing here mutates the IK object: FK runs against a private pin.Data.
"""

import hashlib
import math
import os

import numpy as np

# Written into the episode's info block. A dataset that does not carry this is a
# dataset whose EE numbers have to be reverse engineered later.
EE_FRAME_CONVENTION = {
    "source": "G1_29_ArmIK.reduced_robot (pinocchio), frames L_ee / R_ee",
    "reference_link": "pelvis",
    "waist_joints": "excluded_from_fk",
    # Spelled out because "waist locked" reads as a statement about the robot,
    # and it is not one.
    "waist_note": ("the IK's reduced model holds waist yaw/roll/pitch at zero, so these poses are "
                   "torso-relative: the torso frame translated to the pelvis origin by a constant "
                   "[-0.004, 0, 0.044] m. Whether the physical waist moved is body.qpos[12:15], "
                   "which is recorded independently. A waist-included variant can be derived from "
                   "those joints offline, but it does not round-trip through the deployed IK."),
    "ee_offset_parent": "{side}_wrist_yaw_joint",
    "ee_offset_xyz_m": [0.05, 0.0, 0.0],
    "position_unit": "m",
    "rotation": "euler_extrinsic_xyz_rad (URDF/ROS roll-pitch-yaw)",
    "pose_vec_layout": ["x", "y", "z", "roll", "pitch", "yaw"],
    "matrix_layout": "4x4 homogeneous, row-major, 16 floats",
}


def _file_digest(path):
    """(absolute path, sha256) for a model file, or None when it is not readable."""
    if not path:
        return None
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return {"path": abs_path, "sha256": None}
    return {"path": abs_path, "sha256": digest}


def frame_convention(arm_ik=None):
    """EE_FRAME_CONVENTION plus which model files produced it.

    Deriving EE poses at conversion time is only safe if the conversion uses the
    same kinematics the recording assumed. G1_29_ArmIK loads its reduced model
    from a pickle cache when one exists, so the URDF on disk is not by itself
    proof of what was used -- both are fingerprinted here, and the dataset
    carries the result.
    """
    out = dict(EE_FRAME_CONVENTION)
    if arm_ik is not None:
        out["urdf"] = _file_digest(getattr(arm_ik, "urdf_path", None))
        out["model_cache"] = _file_digest(getattr(arm_ik, "cache_path", None))
    return out


def rotation_to_euler_xyz(R):
    """Extrinsic x-y-z Euler angles (URDF/ROS roll-pitch-yaw) from a rotation matrix.

    Extrinsic xyz means R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the same convention
    scipy spells as as_euler("xyz") and the one the BCT export turned out to use.
    """
    sin_pitch = -float(R[2, 0])
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    if abs(sin_pitch) > 1.0 - 1e-9:
        # Gimbal lock: only roll +/- yaw is observable, so pin roll to zero.
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    else:
        pitch = math.asin(sin_pitch)
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return [roll, pitch, yaw]


def euler_xyz_to_rotation(roll, pitch, yaw):
    """Inverse of rotation_to_euler_xyz; used by the self-test."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_pose_vec(T):
    """4x4 homogeneous transform -> [x, y, z, roll, pitch, yaw]."""
    T = np.asarray(T, dtype=np.float64)
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])] + rotation_to_euler_xyz(T[:3, :3])


def mat_to_flat(T):
    """4x4 homogeneous transform -> 16 floats, row-major.

    Stored alongside the Euler vector because it is lossless and convention-free:
    if a later decision changes how rotations should be parameterised, the matrix
    is still there to re-derive from.
    """
    return [float(v) for v in np.asarray(T, dtype=np.float64).reshape(-1)]


def pose_error(T_target, T_actual):
    """(position error in mm, rotation error in degrees) between two transforms."""
    T_target = np.asarray(T_target, dtype=np.float64)
    T_actual = np.asarray(T_actual, dtype=np.float64)
    pos_mm = float(np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3]) * 1000.0)
    R_err = T_target[:3, :3].T @ T_actual[:3, :3]
    cos_angle = max(-1.0, min(1.0, (float(np.trace(R_err)) - 1.0) / 2.0))
    return pos_mm, float(math.degrees(math.acos(cos_angle)))


class ArmEEKinematics:
    """FK for the L_ee / R_ee frames of an already-built G1_29_ArmIK.

    Borrows the IK's pinocchio model but allocates its own Data, so calling this
    from the record path cannot disturb the solver's own rnea() scratch space.
    """

    def __init__(self, arm_ik):
        import pinocchio as pin

        self._pin = pin
        self.model = arm_ik.reduced_robot.model
        if self.model.nq != 14:
            raise ValueError(
                f"expected a 14-dof reduced arm model, got nq={self.model.nq}; "
                "ee_kinematics assumes G1_29 with legs, waist and hands locked")
        self.data = self.model.createData()
        self.left_frame_id = arm_ik.L_hand_id
        self.right_frame_id = arm_ik.R_hand_id

    def poses(self, q14):
        """Left and right EE transforms (4x4 numpy) for a 14-dof arm configuration."""
        q = np.asarray(q14, dtype=np.float64).reshape(-1)
        self._pin.framesForwardKinematics(self.model, self.data, q)
        return (self.data.oMf[self.left_frame_id].homogeneous.copy(),
                self.data.oMf[self.right_frame_id].homogeneous.copy())


def _self_test():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20000):
        roll = rng.uniform(-math.pi, math.pi)
        pitch = rng.uniform(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3)
        yaw = rng.uniform(-math.pi, math.pi)
        R = euler_xyz_to_rotation(roll, pitch, yaw)
        back = rotation_to_euler_xyz(R)
        worst = max(worst, float(np.abs(euler_xyz_to_rotation(*back) - R).max()))
    assert worst < 1e-9, f"euler round trip drifted by {worst}"
    print(f"euler xyz round trip: max matrix error {worst:.3e} over 20000 random rotations")

    # Gimbal lock must still reproduce the rotation even though the angles differ.
    for pitch in (math.pi / 2, -math.pi / 2):
        R = euler_xyz_to_rotation(0.4, pitch, -1.1)
        assert np.abs(euler_xyz_to_rotation(*rotation_to_euler_xyz(R)) - R).max() < 1e-9

    # Agree with scipy where it is installed; the BCT analysis used as_euler("xyz").
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("scipy not installed, skipping cross-check")
    else:
        diff = 0.0
        for _ in range(2000):
            R = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            diff = max(diff, float(np.abs(
                np.array(rotation_to_euler_xyz(R)) - Rotation.from_matrix(R).as_euler("xyz")).max()))
        assert diff < 1e-9, f"disagrees with scipy by {diff}"
        print(f"scipy as_euler('xyz') cross-check: max diff {diff:.3e}")

    T = np.eye(4)
    T[:3, 3] = [0.3, 0.1, 0.2]
    T2 = T.copy()
    T2[:3, 3] += [0.001, 0.0, 0.0]
    T2[:3, :3] = euler_xyz_to_rotation(0.0, 0.0, math.radians(2.0))
    pos_mm, rot_deg = pose_error(T, T2)
    assert abs(pos_mm - 1.0) < 1e-6 and abs(rot_deg - 2.0) < 1e-6, (pos_mm, rot_deg)
    print(f"pose_error sanity: {pos_mm:.4f} mm, {rot_deg:.4f} deg")
    print("ee_kinematics self-test OK")


if __name__ == "__main__":
    _self_test()
