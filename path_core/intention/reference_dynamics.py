"""Analytic second-order reference generators for PATH control authority."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-9


def _second_order_transition(omega_n: float, zeta: float, dt: float) -> tuple[float, float, float, float]:
    """Exact transition for e_ddot=-2*zeta*wn*e_dot-wn^2*e."""
    wn = max(EPS, float(omega_n))
    # The overdamped closed form divides by 2*wn*sqrt(zeta^2 - 1).
    # Close to one, subtraction between the two exponential modes loses
    # precision, so retain an explicitly numerically safe overdamped margin.
    z = max(1.05, float(zeta))
    root = float(np.sqrt(z * z - 1.0))
    s1, s2 = -wn * (z - root), -wn * (z + root)
    gap = s1 - s2
    e1, e2 = float(np.exp(s1 * max(0.0, float(dt)))), float(np.exp(s2 * max(0.0, float(dt))))
    return ((s1 * e2 - s2 * e1) / gap, (e1 - e2) / gap,
            -wn * wn * (e1 - e2) / gap, (s1 * e1 - s2 * e2) / gap)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    sine = float(np.sin(angle))
    if abs(sine) < 1.0e-6:
        values, vectors = np.linalg.eigh(r + np.eye(3))
        axis = vectors[:, int(np.argmax(values))]
        return axis / max(EPS, float(np.linalg.norm(axis))) * angle
    return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * angle / (2.0 * sine)


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


@dataclass
class ContactResistanceEstimator:
    """Detect when commanded reference motion is not producing real motion.

    Motion resistance is detected from low commanded/achieved progress while
    the reference is advancing.  Once the reference is nearly stationary, a
    persistent tracking error is used instead because the progress ratio is
    no longer meaningful.
    """

    efficiency_enter: float = 0.35
    efficiency_exit: float = 0.60
    min_commanded_mps: float = 0.02
    static_error_m: float = 0.015
    static_error_rate_mps: float = 0.01
    enter_ticks: int = 5
    block_ticks: int = 15
    release_ticks: int = 25
    ema_alpha: float = 0.25

    state: str = "free"
    commanded_ema: float = 0.0
    achieved_ema: float = 0.0
    efficiency: float = 1.0
    error_m: float = 0.0
    error_rate_mps: float = 0.0
    resisted_ticks: int = 0
    free_ticks: int = 0
    previous_reference: np.ndarray | None = None
    previous_measured: np.ndarray | None = None
    previous_error_m: float | None = None

    def reset(self) -> None:
        self.state = "free"
        self.commanded_ema = 0.0
        self.achieved_ema = 0.0
        self.efficiency = 1.0
        self.error_m = 0.0
        self.error_rate_mps = 0.0
        self.resisted_ticks = 0
        self.free_ticks = 0
        self.previous_reference = None
        self.previous_measured = None
        self.previous_error_m = None

    def update(
        self,
        reference: np.ndarray,
        measured: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> str:
        reference = np.asarray(reference, dtype=np.float64).reshape(3)
        measured = np.asarray(measured, dtype=np.float64).reshape(3)
        target = np.asarray(target, dtype=np.float64).reshape(3)
        dt = max(1.0e-4, float(dt))

        self.error_m = float(np.linalg.norm(reference - measured))
        self.error_rate_mps = (
            0.0
            if self.previous_error_m is None
            else (self.error_m - self.previous_error_m) / dt
        )
        self.previous_error_m = self.error_m

        resisted = False
        if self.previous_reference is not None and self.previous_measured is not None:
            direction = target - reference
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1.0e-6:
                unit_direction = direction / direction_norm
                commanded = float(
                    np.dot(reference - self.previous_reference, unit_direction)
                ) / dt
                achieved = float(
                    np.dot(measured - self.previous_measured, unit_direction)
                ) / dt
                alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
                self.commanded_ema = (
                    alpha * commanded + (1.0 - alpha) * self.commanded_ema
                )
                self.achieved_ema = (
                    alpha * achieved + (1.0 - alpha) * self.achieved_ema
                )

            if self.commanded_ema > self.min_commanded_mps:
                self.efficiency = self.achieved_ema / max(EPS, self.commanded_ema)
                threshold = (
                    self.efficiency_enter
                    if self.state == "free"
                    else self.efficiency_exit
                )
                resisted = self.efficiency < threshold
            else:
                self.efficiency = 1.0
                resisted = (
                    self.error_m > self.static_error_m
                    and self.error_rate_mps > -self.static_error_rate_mps
                )

        self.previous_reference = reference.copy()
        self.previous_measured = measured.copy()

        if resisted:
            self.resisted_ticks += 1
            self.free_ticks = 0
        else:
            self.free_ticks += 1
            if self.state == "free":
                # Entry requires consecutive resistance samples.
                self.resisted_ticks = 0

        if self.state != "free" and self.free_ticks >= max(1, int(self.release_ticks)):
            self.state = "free"
            self.resisted_ticks = 0
        elif self.resisted_ticks >= max(1, int(self.block_ticks)):
            self.state = "blocked"
        elif self.resisted_ticks >= max(1, int(self.enter_ticks)):
            self.state = "resisted"
        return self.state


@dataclass
class BimanualCouplingCorrector:
    """Conditionally remove internal stretch between two blocked hands.

    This is deliberately not an always-on object-frame constraint.  It can
    affect references only after both hands independently become blocked and
    the caller confirms a shared grasp, so unrelated teleoperation targets do
    not contaminate one another.
    """

    engage_error_m: float = 0.01
    release_error_m: float = 0.004
    gain: float = 0.3
    max_correction_m: float = 0.03
    latched_distance_m: float | None = None
    engaged: bool = False
    correction_m: float = 0.0

    def reset(self) -> None:
        self.latched_distance_m = None
        self.engaged = False
        self.correction_m = 0.0

    def update(
        self,
        left_ref: np.ndarray,
        right_ref: np.ndarray,
        *,
        both_blocked: bool,
        same_object: bool,
        measured_distance_m: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del dt  # Reserved for a future rate-limited correction law.
        left = np.asarray(left_ref, dtype=np.float64).reshape(3)
        right = np.asarray(right_ref, dtype=np.float64).reshape(3)
        if not (both_blocked and same_object):
            self.reset()
            return left, right
        if self.latched_distance_m is None:
            self.latched_distance_m = float(measured_distance_m)

        offset = left - right
        distance = float(np.linalg.norm(offset))
        if distance < 1.0e-6:
            return left, right
        error = distance - self.latched_distance_m
        threshold = self.release_error_m if self.engaged else self.engage_error_m
        if abs(error) < threshold:
            self.engaged = False
            self.correction_m = 0.0
            return left, right

        self.engaged = True
        magnitude = float(
            np.clip(self.gain * error, -self.max_correction_m, self.max_correction_m)
        )
        self.correction_m = magnitude
        direction = offset / distance
        return (
            left - 0.5 * magnitude * direction,
            right + 0.5 * magnitude * direction,
        )


@dataclass
class SecondOrderPositionReference:
    zeta: float = 2.0
    omega_n: float = 3.0
    max_speed_mps: float = 0.35
    resync_distance_m: float = 0.06
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None

    def reset(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3).copy()

    def step(self, target: np.ndarray, dt: float, *, measured: np.ndarray | None = None,
             omega_n: float | None = None, zeta: float | None = None,
             resistance_state: str = "free",
             equilibrium_shift: np.ndarray | None = None) -> np.ndarray:
        """Advance the reference, clamping its equilibrium while blocked.

        Contact tracking error is legitimate, so measured-position resync is
        disabled while blocked to prevent a push-snap-push limit cycle.
        """
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if self.position is None or self.velocity is None:
            self.reset(measured if measured is not None else target)
        if equilibrium_shift is not None:
            target = target + np.asarray(equilibrium_shift, dtype=np.float64).reshape(3)
        if resistance_state == "blocked":
            target = self.position.copy()
        if (
            measured is not None
            and resistance_state == "free"
            and np.linalg.norm(
                self.position - np.asarray(measured, dtype=np.float64).reshape(3)
            ) > self.resync_distance_m
        ):
            self.reset(measured, self.velocity)
        a11, a12, a21, a22 = _second_order_transition(
            self.omega_n if omega_n is None else omega_n,
            self.zeta if zeta is None else zeta,
            dt,
        )
        error = self.position - target
        self.position, self.velocity = target + a11 * error + a12 * self.velocity, a21 * error + a22 * self.velocity
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.max_speed_mps:
            self.velocity *= self.max_speed_mps / max(EPS, speed)
        return self.position.copy()


@dataclass
class SecondOrderOrientationReference:
    zeta: float = 2.0
    omega_n: float = 4.0
    max_rate_rps: float = 1.5
    resync_angle_rad: float = 0.6
    rotation: np.ndarray | None = None
    rate: np.ndarray | None = None

    def reset(self, rotation: np.ndarray, rate: np.ndarray | None = None) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
        self.rate = np.zeros(3) if rate is None else np.asarray(rate, dtype=np.float64).reshape(3).copy()

    def step(self, target: np.ndarray, dt: float, *, measured: np.ndarray | None = None,
             omega_n: float | None = None, zeta: float | None = None) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(3, 3)
        if self.rotation is None or self.rate is None:
            self.reset(measured if measured is not None else target)
        if measured is not None and np.linalg.norm(rotation_vector(np.asarray(measured).reshape(3, 3) @ self.rotation.T)) > self.resync_angle_rad:
            self.reset(measured, self.rate)
        a11, a12, a21, a22 = _second_order_transition(
            self.omega_n if omega_n is None else omega_n,
            self.zeta if zeta is None else zeta,
            dt,
        )
        error = -rotation_vector(target @ self.rotation.T)
        new_error = a11 * error + a12 * self.rate
        self.rate = a21 * error + a22 * self.rate
        rate = float(np.linalg.norm(self.rate))
        if rate > self.max_rate_rps:
            self.rate *= self.max_rate_rps / max(EPS, rate)
        # log(R_new R_target^T) == new_error, therefore
        # R_new = exp(new_error) R_target.  Negating new_error creates the
        # target mirror image and a period-two orientation limit cycle.
        self.rotation = rotation_from_vector(new_error) @ target
        return self.rotation.copy()


@dataclass
class OrientationTargetGate:
    """Reject discontinuous or contradictory orientation constraints.

    The gate belongs at the reference-control boundary: a perception-side
    posterior can be temporally consistent for a few frames while still
    describing an invalid surface normal.
    """

    max_angle_rad: float = 1.047
    max_step_rate_rps: float = 3.0
    accept_dwell: int = 3
    previous_target: np.ndarray | None = None
    pending_count: int = 0
    reject_count: int = 0
    last_reason: str = ""

    def reset(self) -> None:
        self.previous_target = None
        self.pending_count = 0
        self.last_reason = ""

    def accept(self, target: np.ndarray | None, reference: np.ndarray, dt: float) -> bool:
        if target is None:
            self.previous_target = None
            self.pending_count = 0
            self.last_reason = "no_target"
            return False

        target = np.asarray(target, dtype=np.float64).reshape(3, 3)
        reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
        angle = float(np.linalg.norm(rotation_vector(target @ reference.T)))
        if angle > float(self.max_angle_rad):
            self.previous_target = None
            self.pending_count = 0
            self.reject_count += 1
            self.last_reason = f"angle_{np.degrees(angle):.0f}deg"
            return False

        if self.previous_target is not None and dt > 0.0:
            step = float(np.linalg.norm(rotation_vector(target @ self.previous_target.T)))
            if step > float(self.max_step_rate_rps) * float(dt):
                self.previous_target = target.copy()
                self.pending_count = 0
                self.reject_count += 1
                self.last_reason = f"jump_{np.degrees(step):.0f}deg"
                return False

        self.previous_target = target.copy()
        self.pending_count += 1
        if self.pending_count < max(1, int(self.accept_dwell)):
            self.last_reason = f"dwell_{self.pending_count}"
            return False
        self.last_reason = "ok"
        return True
