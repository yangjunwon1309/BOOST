"""Bounded second-order SE(3) reference dynamics for wrist targets."""

from __future__ import annotations

import math

import numpy as np


def _vector(value, size: int, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(size)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def so3_exp(rotation_vector) -> np.ndarray:
    """Return the matrix exponential of a 3-vector in radians."""
    vector = _vector(rotation_vector, 3, name="rotation_vector")
    angle = float(np.linalg.norm(vector))
    skew = np.array(
        [[0.0, -vector[2], vector[1]],
         [vector[2], 0.0, -vector[0]],
         [-vector[1], vector[0], 0.0]],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    axis_skew = skew / angle
    return np.eye(3) + math.sin(angle) * axis_skew + (1.0 - math.cos(angle)) * (axis_skew @ axis_skew)


def so3_log(rotation) -> np.ndarray:
    """Return the shortest rotation vector for a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("rotation must contain only finite values")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_vector = np.array(
        [matrix[2, 1] - matrix[1, 2],
         matrix[0, 2] - matrix[2, 0],
         matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return 0.5 * skew_vector
    if math.pi - angle < 1.0e-5:
        # The antisymmetric part vanishes at pi. Recover an axis from R+I,
        # choosing the largest diagonal for numerical stability.
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        pivot = int(np.argmax(diagonal))
        axis = np.zeros(3, dtype=np.float64)
        axis[pivot] = math.sqrt(float(diagonal[pivot]))
        if axis[pivot] > 1.0e-8:
            for index in range(3):
                if index != pivot:
                    axis[index] = (matrix[pivot, index] + matrix[index, pivot]) / (4.0 * axis[pivot])
        else:
            axis[pivot] = 1.0
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-8:
            axis[pivot] = 1.0
            norm = 1.0
        axis /= norm
        if np.linalg.norm(skew_vector) > 1.0e-8 and float(axis @ skew_vector) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * math.sin(angle))) * skew_vector


def _clip_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= maximum or norm == 0.0:
        return value
    return value * (maximum / norm)


def _decay_offset(value: np.ndarray, maximum_step: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= maximum_step or norm == 0.0:
        return np.zeros_like(value)
    return value * ((norm - maximum_step) / norm)


class DampedSE3Reference:
    """Semi-implicit second-order SE(3) tracker with norm-bounded speeds.

    ``advance`` integrates position in the parent frame and angular velocity
    in the current/body frame. ``release_to_direct`` stores the remaining
    target residual; ``direct_target`` transfers it to a moving target while
    removing the residual at the configured linear/angular speed limits.
    """

    def __init__(
        self,
        *,
        zeta: float = 1.5,
        omega_n: float = 5.0,
        v_max: float = 0.06,
        omega_max: float = 0.6,
    ) -> None:
        values = (zeta, omega_n, v_max, omega_max)
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("damped SE(3) parameters must be finite and positive")
        self.zeta = float(zeta)
        self.omega_n = float(omega_n)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float64)
        self.position_offset = np.zeros(3, dtype=np.float64)
        self.rotation_offset = np.zeros(3, dtype=np.float64)
        self.initialized = False

    @staticmethod
    def _transform(value) -> np.ndarray:
        transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(transform).all():
            raise ValueError("SE(3) transform must contain only finite values")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
            raise ValueError("SE(3) transform has an invalid homogeneous row")
        return transform.copy()

    def reset(self, transform) -> None:
        """Initialize the reference pose and clear all velocity/residual state."""
        target = self._transform(transform)
        self.position = target[:3, 3].copy()
        self.rotation = target[:3, :3].copy()
        self.velocity.fill(0.0)
        self.angular_velocity.fill(0.0)
        self.position_offset.fill(0.0)
        self.rotation_offset.fill(0.0)
        self.initialized = True

    def transform(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.position
        return result

    def advance(self, target, dt: float) -> np.ndarray:
        """Advance toward ``target`` by one positive time step."""
        goal = self._transform(target)
        if not self.initialized:
            self.reset(goal)
            return goal
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and positive")
        error = goal[:3, 3] - self.position
        acceleration = self.omega_n**2 * error - 2.0 * self.zeta * self.omega_n * self.velocity
        self.velocity = _clip_norm(self.velocity + acceleration * step, self.v_max)
        self.position = self.position + self.velocity * step

        rotation_error = so3_log(self.rotation.T @ goal[:3, :3])
        angular_acceleration = (
            self.omega_n**2 * rotation_error
            - 2.0 * self.zeta * self.omega_n * self.angular_velocity
        )
        self.angular_velocity = _clip_norm(
            self.angular_velocity + angular_acceleration * step,
            self.omega_max,
        )
        self.rotation = self.rotation @ so3_exp(self.angular_velocity * step)
        return self.transform()

    def release_to_direct(self, target) -> None:
        """Preserve pose continuity while switching to a moving direct goal."""
        goal = self._transform(target)
        if not self.initialized:
            self.reset(goal)
        self.position_offset = self.position - goal[:3, 3]
        self.rotation_offset = so3_log(goal[:3, :3].T @ self.rotation)
        self.velocity.fill(0.0)
        self.angular_velocity.fill(0.0)

    def direct_target(self, target, dt: float) -> np.ndarray:
        """Apply and decay the continuity residual on a direct target."""
        goal = self._transform(target)
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and positive")
        result = goal.copy()
        result[:3, 3] += self.position_offset
        result[:3, :3] = goal[:3, :3] @ so3_exp(self.rotation_offset)
        self.position_offset = _decay_offset(self.position_offset, self.v_max * step)
        self.rotation_offset = _decay_offset(self.rotation_offset, self.omega_max * step)
        self.position = result[:3, 3].copy()
        self.rotation = result[:3, :3].copy()
        return result
