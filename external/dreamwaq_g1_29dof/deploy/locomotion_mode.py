"""Operator-selected NEW3 mode, independent of joystick speed or UDP commands."""

import numpy as np


TOGGLE_BUTTON_MASKS = {"X": 0x0400, "Y": 0x0800, "B": 0x0200}


class LocomotionModeSwitch:
    """Toggle once per button press and gate every velocity source in standing."""

    def __init__(self, *, enabled, initial_mode="standing", button="X", initial_keys=0):
        if initial_mode not in ("standing", "walking"):
            raise ValueError("initial locomotion mode must be standing or walking")
        self.enabled = bool(enabled)
        self.mode = int(initial_mode == "walking")
        self.button_mask = TOGGLE_BUTTON_MASKS[button]
        self._pressed = bool(int(initial_keys) & self.button_mask)

    @property
    def name(self):
        return "walking" if self.mode else "standing"

    def set_mode(self, name):
        """Select a mode directly, as Thor's request does; True if it changed.

        The button edge state is left alone, so an X held at the time still
        needs its release before it toggles again.
        """
        if name not in ("standing", "walking"):
            raise ValueError("locomotion mode must be standing or walking")
        if not self.enabled:
            raise RuntimeError("this export has no standing/walking mode")
        mode = int(name == "walking")
        changed = mode != self.mode
        self.mode = mode
        return changed

    def update(self, keys, command):
        value = np.asarray(command, dtype=np.float32).reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("velocity command must contain three finite values")
        pressed = bool(int(keys) & self.button_mask)
        changed = self.enabled and pressed and not self._pressed
        self._pressed = pressed
        if changed:
            self.mode = 1 - self.mode
        effective = value.copy()
        if self.enabled and self.mode == 0:
            effective.fill(0.0)
        return effective, self.mode, changed
