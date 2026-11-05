"""Math helpers vendored verbatim (numpy-only) from robofinals wbc_policy/utils/homie_utils.py.

Kept byte-for-byte equivalent so the deployed observation matches training.
"""
import numpy as np
import yaml


def load_config(config_path):
    """Load the HOMIE policy YAML and convert list fields to numpy arrays."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    array_keys = ["default_angles", "cmd_scale", "cmd_init", "kps", "kds", "rpy_cmd"]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (q is [w, x, y, z])."""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ]
    )


def get_gravity_orientation(quat):
    """Gravity direction expressed in the base frame. quat is [w, x, y, z]."""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)
