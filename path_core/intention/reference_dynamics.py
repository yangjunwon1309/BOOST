"""Analytic second-order reference generators for PATH control authority."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-9


def _second_order_transition(omega_n: float, zeta: float, dt: float) -> tuple[float, float, float, float]:
    """Exact transition for e_ddot=-2*zeta*wn*e_dot-wn^2*e."""
    wn = max(EPS, float(omega_n))
    z = max(1.0 + 1.0e-6, float(zeta))
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
class SecondOrderPositionReference:
    zeta: float = 2.0
    omega_n: float = 3.0
    max_speed_mps: float = 0.35
    resync_distance_m: float = 0.15
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None

    def reset(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3).copy()

    def step(self, target: np.ndarray, dt: float, *, measured: np.ndarray | None = None, omega_n: float | None = None) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if self.position is None or self.velocity is None:
            self.reset(measured if measured is not None else target)
        if measured is not None and np.linalg.norm(self.position - np.asarray(measured, dtype=np.float64).reshape(3)) > self.resync_distance_m:
            self.reset(measured, self.velocity)
        a11, a12, a21, a22 = _second_order_transition(self.omega_n if omega_n is None else omega_n, self.zeta, dt)
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

    def step(self, target: np.ndarray, dt: float, *, measured: np.ndarray | None = None, omega_n: float | None = None) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).reshape(3, 3)
        if self.rotation is None or self.rate is None:
            self.reset(measured if measured is not None else target)
        if measured is not None and np.linalg.norm(rotation_vector(np.asarray(measured).reshape(3, 3) @ self.rotation.T)) > self.resync_angle_rad:
            self.reset(measured, self.rate)
        a11, a12, a21, a22 = _second_order_transition(self.omega_n if omega_n is None else omega_n, self.zeta, dt)
        error = -rotation_vector(target @ self.rotation.T)
        new_error = a11 * error + a12 * self.rate
        self.rate = a21 * error + a22 * self.rate
        rate = float(np.linalg.norm(self.rate))
        if rate > self.max_rate_rps:
            self.rate *= self.max_rate_rps / max(EPS, rate)
        self.rotation = rotation_from_vector(-new_error) @ target
        return self.rotation.copy()
