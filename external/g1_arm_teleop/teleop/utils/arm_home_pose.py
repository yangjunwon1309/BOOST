"""Resting arm targets, independent of the live XR wrist mounting offsets."""

import numpy as np


def g1_29_home_q(end_effector):
    """Return the 14 arm joints, left seven followed by right seven, in radians.

    With this rig's BrainCo mounts the zero-joint pose has both palms facing
    up. The two +X wrist-roll axes need opposite quarter turns to face inward.
    Other end effectors retain the original zero-joint home pose.
    """
    target = np.zeros(14, dtype=float)
    if end_effector == 'brainco':
        target[4] = np.pi / 2   # left wrist roll, hardware joint 19
        target[11] = -np.pi / 2  # right wrist roll, hardware joint 26
    return target
