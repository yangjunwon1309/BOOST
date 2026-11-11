"""Command-following-error contact estimates for BrainCo Revo2 fingers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


CHANNELS = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_FINGER_CHANNELS = {
    # Thumb proximal is the flexion channel; metacarpal is opposition/base motion.
    "thumb": "thumb_aux",
    "index": "index",
    "middle": "middle",
    "ring": "ring",
    "pinky": "pinky",
}


def _six(values: Sequence[float] | None, *, name: str) -> np.ndarray | None:
    if values is None:
        return None
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite values")
    return result


def tracking_error_and_raw_contact(
    q_cmd: Sequence[float],
    q_meas: Sequence[float],
    *,
    tau_on: float = 0.08,
    finger_channels: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, list[bool]]:
    """Purely map normalized command-following error to raw finger bits."""
    command = _six(q_cmd, name="q_cmd")
    measured = _six(q_meas, name="q_meas")
    threshold = float(tau_on)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tau_on must be finite and nonnegative")
    names = dict(DEFAULT_FINGER_CHANNELS if finger_channels is None else finger_channels)
    if set(names) != set(FINGERS) or any(name not in CHANNELS for name in names.values()):
        raise ValueError("finger_channels must map all five fingers to Revo2 channel names")
    error = command - measured
    raw = [bool(error[CHANNELS.index(names[finger])] > threshold) for finger in FINGERS]
    return error, raw


class HandContactStateEstimator:
    """Debounce per-finger positive command-following errors with hysteresis."""

    def __init__(
        self,
        *,
        tau_on: float = 0.08,
        tau_off: float = 0.04,
        minimum_duration_s: float = 0.10,
        stale_after_s: float = 0.25,
        finger_channels: Mapping[str, str] | None = None,
    ) -> None:
        values = (tau_on, tau_off, minimum_duration_s, stale_after_s)
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("contact thresholds and durations must be finite and nonnegative")
        if tau_off > tau_on or stale_after_s <= 0.0:
            raise ValueError("contact requires tau_off <= tau_on and positive stale_after_s")
        self.tau_on = float(tau_on)
        self.tau_off = float(tau_off)
        self.minimum_duration_s = float(minimum_duration_s)
        self.stale_after_s = float(stale_after_s)
        names = dict(DEFAULT_FINGER_CHANNELS if finger_channels is None else finger_channels)
        if set(names) != set(FINGERS) or any(name not in CHANNELS for name in names.values()):
            raise ValueError("finger_channels must map all five fingers to Revo2 channel names")
        self.finger_channels = names
        self._channel_indices = {finger: CHANNELS.index(channel) for finger, channel in names.items()}
        self._stable = {finger: False for finger in FINGERS}
        self._pending_since: dict[str, float | None] = {finger: None for finger in FINGERS}
        self._contact_started: dict[str, float | None] = {finger: None for finger in FINGERS}
        self._contact_count = {finger: 0 for finger in FINGERS}
        self._contact_duration_s = {finger: 0.0 for finger in FINGERS}
        self._last_command: np.ndarray | None = None
        self._last_command_time: float | None = None
        self._last_state_q: np.ndarray | None = None
        self._last_state_dq: np.ndarray | None = None
        self._last_state_tau: np.ndarray | None = None
        self._last_state_time: float | None = None

    def update_command(self, q_cmd: Sequence[float], received_monotonic: float) -> None:
        self._last_command = _six(q_cmd, name="q_cmd")
        self._last_command_time = float(received_monotonic)

    def update_state(
        self,
        q_meas: Sequence[float],
        dq_meas: Sequence[float],
        tau_est: Sequence[float],
        received_monotonic: float,
    ) -> None:
        self._last_state_q = _six(q_meas, name="q_meas")
        self._last_state_dq = _six(dq_meas, name="dq_meas")
        self._last_state_tau = _six(tau_est, name="tau_est")
        self._last_state_time = float(received_monotonic)
        self._update_contacts(float(received_monotonic))

    def _update_contacts(self, now: float) -> None:
        if self._last_command is None or self._last_state_q is None:
            return
        if self._last_command_time is None or self._last_state_time is None:
            return
        if (
            now - self._last_command_time > self.stale_after_s
            or now - self._last_state_time > self.stale_after_s
        ):
            for finger in FINGERS:
                self._pending_since[finger] = None
            return
        errors, _ = tracking_error_and_raw_contact(
            self._last_command,
            self._last_state_q,
            tau_on=self.tau_on,
            finger_channels=self.finger_channels,
        )
        for finger in FINGERS:
            error = float(errors[self._channel_indices[finger]])
            active = self._stable[finger]
            transition = error < self.tau_off if active else error > self.tau_on
            if not transition:
                self._pending_since[finger] = None
                continue
            if self._pending_since[finger] is None:
                self._pending_since[finger] = now
            if now - float(self._pending_since[finger]) < self.minimum_duration_s:
                continue
            self._pending_since[finger] = None
            self._stable[finger] = not active
            if self._stable[finger]:
                self._contact_started[finger] = now
                self._contact_count[finger] += 1
            else:
                started = self._contact_started[finger]
                if started is not None:
                    self._contact_duration_s[finger] += max(0.0, now - started)
                self._contact_started[finger] = None

    def snapshot(self, now: float) -> dict:
        now = float(now)
        command_age = (
            None if self._last_command_time is None else max(0.0, now - self._last_command_time)
        )
        state_age = (
            None if self._last_state_time is None else max(0.0, now - self._last_state_time)
        )
        available = bool(
            command_age is not None
            and state_age is not None
            and command_age <= self.stale_after_s
            and state_age <= self.stale_after_s
        )
        if available and self._last_command is not None and self._last_state_q is not None:
            errors, raw = tracking_error_and_raw_contact(
                self._last_command,
                self._last_state_q,
                tau_on=self.tau_on,
                finger_channels=self.finger_channels,
            )
        else:
            errors = None
            raw = [None] * len(FINGERS)
        contacts = (
            [bool(self._stable[finger]) for finger in FINGERS]
            if available
            else [None] * len(FINGERS)
        )
        return {
            "available": available,
            "cmd_age_ms": None if command_age is None else command_age * 1000.0,
            "state_age_ms": None if state_age is None else state_age * 1000.0,
            "channels": list(CHANNELS),
            "q_cmd": None if self._last_command is None else self._last_command.tolist(),
            "q_meas": None if self._last_state_q is None else self._last_state_q.tolist(),
            "dq_meas": None if self._last_state_dq is None else self._last_state_dq.tolist(),
            "tau_est": None if self._last_state_tau is None else self._last_state_tau.tolist(),
            "err": None if errors is None else errors.tolist(),
            "fingers": list(FINGERS),
            "contact_raw": raw,
            "contact": contacts,
        }

    def summary(self, now: float) -> dict[str, dict[str, float | int]]:
        now = float(now)
        result = {}
        for finger in FINGERS:
            duration = self._contact_duration_s[finger]
            started = self._contact_started[finger]
            if started is not None:
                duration += max(0.0, now - started)
            result[finger] = {
                "contact_intervals": self._contact_count[finger],
                "contact_time_s": duration,
            }
        return result
