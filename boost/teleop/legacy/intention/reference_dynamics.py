"""Standalone reference filters; the active XR arm bridge bypasses these for Televuer goals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

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


def slerp_so3(source: np.ndarray, target: np.ndarray, weight: float) -> np.ndarray:
    """Geodesically interpolate finite SO(3) rotations without quaternion sign rules."""
    source = np.asarray(source, dtype=np.float64).reshape(3, 3)
    target = np.asarray(target, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("SO(3) interpolation requires finite rotations")
    fraction = float(np.clip(weight if np.isfinite(weight) else 0.0, 0.0, 1.0))
    return rotation_from_vector(fraction * rotation_vector(target @ source.T)) @ source


@dataclass
class OperatorFrameOffset:
    """Keep an operator-frame position target continuous after assistance ends.

    The offset is refreshed while assistance owns the reference, then washed
    out only by deliberate operator travel.  It never decays while the held
    object and the controller are stationary.
    """

    washout_per_m: float = 4.6
    max_offset_m: float = 0.30
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    previous_human: np.ndarray | None = None
    peak_offset_m: float = 0.0

    def reset(self) -> None:
        self.offset = np.zeros(3, dtype=np.float64)
        self.previous_human = None
        self.peak_offset_m = 0.0

    def track(self, reference: np.ndarray, human: np.ndarray) -> None:
        reference = np.asarray(reference, dtype=np.float64).reshape(3)
        human = np.asarray(human, dtype=np.float64).reshape(3)
        delta = reference - human
        norm = float(np.linalg.norm(delta))
        self.peak_offset_m = max(self.peak_offset_m, norm)
        limit = max(0.0, float(self.max_offset_m))
        if limit > 0.0 and norm > limit:
            delta *= limit / norm
        self.offset = delta
        self.previous_human = human.copy()

    def apply(self, human: np.ndarray) -> np.ndarray:
        human = np.asarray(human, dtype=np.float64).reshape(3)
        if self.previous_human is not None:
            travel = float(np.linalg.norm(human - self.previous_human))
            self.offset *= float(
                np.exp(-max(0.0, float(self.washout_per_m)) * travel)
            )
        self.previous_human = human.copy()
        return human + self.offset


@dataclass
class FirstOrderPositionTargetFilter:
    """Per-axis EWMA low-pass filter for an already selected Cartesian target.

    ``alpha`` is the new-target weight.  The caller may supply ``seed`` on the
    first assisted tick so enabling an alpha blend begins from the operator
    command instead of jumping directly to the blended belief target.
    """

    alpha: float = 0.30
    position: np.ndarray | None = None

    def reset(self) -> None:
        self.position = None

    def step(self, target: np.ndarray, *, seed: np.ndarray | None = None) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if self.position is None:
            self.position = np.asarray(
                target if seed is None else seed, dtype=np.float64
            ).reshape(3).copy()
            return self.position.copy()
        alpha = float(np.clip(self.alpha, 0.0, 1.0))
        self.position = (1.0 - alpha) * self.position + alpha * target
        return self.position.copy()


@dataclass
class OperatorRotationOffset:
    """Rotation analogue of :class:`OperatorFrameOffset` for grasp handoff."""

    washout_per_rad: float = 2.0
    offset: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    previous_human: np.ndarray | None = None

    def reset(self) -> None:
        self.offset = np.eye(3, dtype=np.float64)
        self.previous_human = None

    def capture(self, reference: np.ndarray, human: np.ndarray) -> None:
        reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
        human = np.asarray(human, dtype=np.float64).reshape(3, 3)
        self.offset = reference @ human.T
        self.previous_human = human.copy()

    def apply(self, human: np.ndarray) -> np.ndarray:
        human = np.asarray(human, dtype=np.float64).reshape(3, 3)
        if self.previous_human is not None:
            travel = float(
                np.linalg.norm(rotation_vector(human @ self.previous_human.T))
            )
            self.offset = rotation_from_vector(
                rotation_vector(self.offset)
                * float(np.exp(-max(0.0, float(self.washout_per_rad)) * travel))
            )
        self.previous_human = human.copy()
        return self.offset @ human


@dataclass
class GraspTransitionDetector:
    """Physical REACHING -> CLOSING -> HOLDING grasp-state machine.

    Jaw stall and object/hand co-motion are independent evidence channels.
    The timeout only leaves CLOSING; it never declares a grasp by itself.
    ``manual_override`` retains the established long close hold as an explicit
    operator fallback when neither physical channel is available.
    """

    jaw_stall_gap_min_m: float = 0.008
    jaw_stall_rate_max_mps: float = 0.004
    jaw_stall_ticks: int = 5
    comotion_window_s: float = 0.5
    comotion_enter_log_ratio: float = -0.7
    comotion_exit_log_ratio: float = 0.0
    comotion_min_excitation_m: float = 0.02
    closing_timeout_s: float = 0.60
    state: str = "reaching"
    channels: dict[str, bool] = field(
        default_factory=lambda: {"jaw": False, "comotion": False}
    )
    closing_since: float | None = None
    closing_elapsed_s: float = 0.0
    grasp_failed: bool = False
    override_active: bool = False
    override_count: int = 0
    comotion_log_ratio: float | None = None
    _jaw_ticks: int = 0
    _previous_gap: float | None = None
    _history: deque[tuple[float, np.ndarray, np.ndarray]] = field(
        default_factory=deque
    )
    _timeout_latched: bool = False

    def reset(self) -> None:
        self.state = "reaching"
        self.channels = {"jaw": False, "comotion": False}
        self.closing_since = None
        self.closing_elapsed_s = 0.0
        self.grasp_failed = False
        self.override_active = False
        self.comotion_log_ratio = None
        self._jaw_ticks = 0
        self._previous_gap = None
        self._history.clear()
        self._timeout_latched = False

    def _update_jaw(
        self, *, commanded_closed: bool, gap_m: float | None, dt: float
    ) -> None:
        if not commanded_closed or gap_m is None or not math.isfinite(float(gap_m)):
            self.channels["jaw"] = False
            self._jaw_ticks = 0
            self._previous_gap = None
            return
        gap = float(gap_m)
        rate = float("inf")
        if self._previous_gap is not None and dt > 0.0:
            rate = abs(gap - self._previous_gap) / dt
        has_previous = self._previous_gap is not None
        self._previous_gap = gap
        if gap <= float(self.jaw_stall_gap_min_m):
            self.grasp_failed = True
            self.channels["jaw"] = False
            self._jaw_ticks = 0
            return
        if has_previous and rate < float(self.jaw_stall_rate_max_mps):
            self._jaw_ticks += 1
        else:
            self._jaw_ticks = 0
        self.channels["jaw"] = self._jaw_ticks >= max(1, int(self.jaw_stall_ticks))

    def _update_comotion(
        self,
        p_camera: np.ndarray | None,
        p_goal: np.ndarray | None,
        now_s: float,
    ) -> None:
        if p_camera is None or p_goal is None:
            self.channels["comotion"] = False
            return
        try:
            camera = np.asarray(p_camera, dtype=np.float64).reshape(3)
            goal = np.asarray(p_goal, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            self.channels["comotion"] = False
            return
        if not np.isfinite(camera).all() or not np.isfinite(goal).all():
            self.channels["comotion"] = False
            return
        self._history.append((float(now_s), camera, goal))
        while self._history and now_s - self._history[0][0] > max(
            0.0, float(self.comotion_window_s)
        ):
            self._history.popleft()
        if len(self._history) < 5:
            return
        camera_history = np.stack([sample[1] for sample in self._history])
        goal_history = np.stack([sample[2] for sample in self._history])
        var_camera = float(np.sum(np.var(camera_history, axis=0)))
        var_goal = float(np.sum(np.var(goal_history, axis=0)))
        if max(math.sqrt(var_camera), math.sqrt(var_goal)) < float(
            self.comotion_min_excitation_m
        ):
            return
        self.comotion_log_ratio = float(
            np.log10(max(var_camera, 1.0e-12) / max(var_goal, 1.0e-12))
        )
        threshold = (
            float(self.comotion_exit_log_ratio)
            if self.channels["comotion"]
            else float(self.comotion_enter_log_ratio)
        )
        self.channels["comotion"] = self.comotion_log_ratio < threshold

    def update(
        self,
        *,
        commanded_closed: bool,
        jaw_gap_m: float | None,
        p_camera: np.ndarray | None,
        p_goal: np.ndarray | None,
        now_s: float,
        dt: float,
        manual_override: bool = False,
    ) -> str:
        step = max(0.0, float(dt))
        if not commanded_closed:
            self.reset()
            return self.state
        self.closing_elapsed_s += step
        self._update_jaw(
            commanded_closed=commanded_closed, gap_m=jaw_gap_m, dt=step
        )
        self._update_comotion(p_camera, p_goal, now_s)
        engaged = self.channels["jaw"] or self.channels["comotion"]
        self.override_active = bool(manual_override)

        if self.state == "reaching":
            if not self._timeout_latched:
                self.state = "closing"
                self.closing_since = float(now_s)
                self.grasp_failed = False
        elif self.state == "closing":
            elapsed = float(now_s) - float(
                now_s if self.closing_since is None else self.closing_since
            )
            if engaged:
                self.state = "holding"
            elif self.grasp_failed:
                self.state = "reaching"
                self._timeout_latched = True
            elif elapsed >= max(0.0, float(self.closing_timeout_s)):
                self.state = "reaching"
                self._timeout_latched = True
        if manual_override and self.state != "holding":
            self.state = "holding"
            self.override_count += 1
        return self.state


@dataclass
class OrientationAuthority:
    """Smooth, hysteretic, rate-limited perception orientation authority."""

    enter_confidence: float = 0.45
    exit_confidence: float = 0.25
    ema_alpha: float = 0.15
    max_authority_rate_per_s: float = 1.2
    smoothed_confidence: float = 0.0
    authority_weight: float = 0.0
    engaged: bool = False

    def reset(self) -> None:
        self.smoothed_confidence = 0.0
        self.authority_weight = 0.0
        self.engaged = False

    def update(self, confidence: float | None, *, gate_ok: bool, dt: float) -> float:
        value = float(confidence) if confidence is not None else 0.0
        value = value if np.isfinite(value) else 0.0
        value = float(np.clip(value, 0.0, 1.0))
        alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.smoothed_confidence += alpha * (value - self.smoothed_confidence)
        enter = float(np.clip(self.enter_confidence, 0.0, 1.0))
        exit_ = float(np.clip(min(self.exit_confidence, enter), 0.0, 1.0))
        if not self.engaged and self.smoothed_confidence >= enter:
            self.engaged = True
        elif self.engaged and self.smoothed_confidence < exit_:
            self.engaged = False
        requested = 0.0
        if self.engaged and gate_ok:
            requested = (self.smoothed_confidence - exit_) / max(EPS, 1.0 - exit_)
        requested = float(np.clip(requested, 0.0, 1.0))
        delta_limit = max(0.0, float(self.max_authority_rate_per_s)) * max(0.0, float(dt))
        self.authority_weight += float(np.clip(requested - self.authority_weight, -delta_limit, delta_limit))
        return float(np.clip(self.authority_weight, 0.0, 1.0))


@dataclass
class SecondOrderPositionReference:
    zeta: float = 2.0
    omega_n: float = 3.0
    max_speed_mps: float = 0.35
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None

    def reset(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3).copy()

    def step(self, target: np.ndarray, dt: float, *, measured: np.ndarray | None = None,
             omega_n: float | None = None, zeta: float | None = None) -> np.ndarray:
        """Advance toward the input target regardless of measured tracking lag."""
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if self.position is None or self.velocity is None:
            self.reset(measured if measured is not None else target)
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
    taper_start_fraction: float = 0.667
    previous_target: np.ndarray | None = None
    pending_count: int = 0
    reject_count: int = 0
    last_reason: str = ""

    def reset(self) -> None:
        self.previous_target = None
        self.pending_count = 0
        self.last_reason = ""

    def authority(self, target: np.ndarray | None, reference: np.ndarray, dt: float) -> float:
        if target is None:
            self.previous_target = None
            self.pending_count = 0
            self.last_reason = "no_target"
            return 0.0

        try:
            target = np.asarray(target, dtype=np.float64).reshape(3, 3)
            reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError):
            self.last_reason = "invalid_target"
            return 0.0
        if not np.isfinite(target).all() or not np.isfinite(reference).all():
            self.last_reason = "invalid_target"
            return 0.0
        angle = float(np.linalg.norm(rotation_vector(target @ reference.T)))
        if angle > float(self.max_angle_rad):
            self.previous_target = None
            self.pending_count = 0
            self.reject_count += 1
            self.last_reason = f"angle_{np.degrees(angle):.0f}deg"
            return 0.0

        if self.previous_target is not None and dt > 0.0:
            step = float(np.linalg.norm(rotation_vector(target @ self.previous_target.T)))
            if step > float(self.max_step_rate_rps) * float(dt):
                self.previous_target = target.copy()
                self.pending_count = 0
                self.reject_count += 1
                self.last_reason = f"jump_{np.degrees(step):.0f}deg"
                return 0.0

        self.previous_target = target.copy()
        self.pending_count += 1
        if self.pending_count < max(1, int(self.accept_dwell)):
            self.last_reason = f"dwell_{self.pending_count}"
            return 0.0
        taper_start = float(np.clip(self.taper_start_fraction, 0.0, 0.999999))
        taper_angle = taper_start * float(self.max_angle_rad)
        if angle > taper_angle:
            weight = (float(self.max_angle_rad) - angle) / max(EPS, float(self.max_angle_rad) - taper_angle)
            self.last_reason = "angle_taper"
            return float(np.clip(weight, 0.0, 1.0))
        self.last_reason = "ok"
        return 1.0

    def accept(self, target: np.ndarray | None, reference: np.ndarray, dt: float) -> bool:
        """Compatibility predicate for callers migrating to :meth:`authority`."""
        return self.authority(target, reference, dt) > 0.0
