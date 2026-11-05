"""Hand a running robot between two DreamWaQ exports on a remote button.

Both exports are loaded in their own policy processes at startup and stay
loaded, so a switch either way costs one 50-Hz tick: the writer keeps holding
the last accepted leg target, the next state goes to the other worker, and the
leg target blends from the held pose to that policy's output over
``--ramp-time``.  The LowCmd writer and the DDS readers are untouched, so the
robot never passes through damping.  That is only sound if both exports agree
on everything the writer and the bring-up use, which ``incompatible_fields``
checks.
"""

from __future__ import annotations

import numpy as np


# Fixed at startup for the whole process: joint map, default pose for the
# ramps, gains from the hardware config and the writer's torque envelope.
SHARED_CONTRACT_FIELDS = (
    "action_dim", "action_joint_names", "default_q_sdk", "kp_sdk", "kd_sdk", "torque_limit_sdk",
)


def incompatible_fields(first, second):
    return [
        name for name in SHARED_CONTRACT_FIELDS
        if not np.array_equal(np.asarray(getattr(first, name)), np.asarray(getattr(second, name)))
    ]


class SoloButtonRelease:
    """Fire once when a button is released after being held on its own.

    Firing on release rather than press keeps chords out.  With
    ``--controller-switch``, L1+Y also sets the Y bit, and an operator who
    lands Y a tick before L1 would otherwise trigger both.  Any other button
    seen during the hold cancels the press.
    """

    def __init__(self, mask, initial_keys=0):
        self.mask = int(mask)
        # None: released.  True: held alone so far.  False: held, disqualified,
        # including a press already in progress when this switch was built.
        self._solo = False if int(initial_keys) & self.mask else None

    def update(self, keys):
        keys = int(keys)
        if keys & self.mask:
            if self._solo is None:
                self._solo = keys == self.mask
            elif keys != self.mask:
                self._solo = False
            return False
        fired = self._solo is True
        self._solo = None
        return fired


class Stillness:
    """Whether the legs and the base have been measured quiet for long enough.

    The way back from an export without a standing mode cannot go by the
    command: a zero command does not mean that policy is not stepping, and the
    held leg target is a standing pose only if the joints themselves are
    still.  Fed once per control tick; a non-finite sample counts as moving.
    """

    def __init__(self, max_leg_dq, max_gyro, ticks):
        self.max_leg_dq = float(max_leg_dq)
        self.max_gyro = float(max_gyro)
        self.ticks = max(1, int(ticks))
        self.leg_dq = self.gyro = float("nan")
        self._quiet = 0

    def update(self, leg_dq, gyro):
        self.leg_dq = float(np.max(np.abs(leg_dq)))
        self.gyro = float(np.max(np.abs(gyro)))
        quiet = self.leg_dq <= self.max_leg_dq and self.gyro <= self.max_gyro
        self._quiet = self._quiet + 1 if quiet else 0

    def reset(self):
        self._quiet = 0

    @property
    def still(self):
        return self._quiet >= self.ticks
