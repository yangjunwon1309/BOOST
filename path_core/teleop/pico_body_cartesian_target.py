"""Validated pelvis-local Cartesian wrist targets from PICO body tracking.

``vr_position`` and ``vr_orientation`` are generated from PICO body tracking
and anchored by the PICO manager to G1 FK. They use the same wrist-yaw + 5 cm
EE frame as the arm IPOPT model. Native controller poses are recorded only as
diagnostics. This tracker validates the authoritative body/G1 stream and does
not create another coordinate transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PicoBodyTrackingSample:
    left_position: np.ndarray
    right_position: np.ndarray
    neck_position: np.ndarray
    left_orientation_wxyz: np.ndarray
    right_orientation_wxyz: np.ndarray
    neck_orientation_wxyz: np.ndarray
    source_timestamp_us: int
    frame_index: int
    source: str = "vr_pose_debug"


@dataclass(frozen=True)
class PicoCartesianTargetConfig:
    calibration_sec: float = 0.5
    gap_hold_sec: float = 0.1
    position_jump_m: float = 0.05
    orientation_jump_rad: float = np.deg2rad(12.0)
    recovery_frames: int = 3
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_calibration_frames: int = 10


@dataclass
class PicoCartesianTargets:
    left_transform: np.ndarray
    right_transform: np.ndarray
    source_timestamp_us: int
    frame_index: int
    valid: bool
    held: bool
    reason: str
    left_raw_position: np.ndarray
    right_raw_position: np.ndarray
    calibration_frames: int
    recovery_count: int
    source: str


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-10:
        raise ValueError("invalid quaternion")
    return q / norm


def _quaternion_mean_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Markley quaternion mean with deterministic hemisphere selection."""

    q = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1.0e-12)
    reference = q[0]
    q[np.sum(q * reference, axis=1) < 0.0] *= -1.0
    values, vectors = np.linalg.eigh(q.T @ q)
    mean = vectors[:, int(np.argmax(values))]
    if float(np.dot(mean, reference)) < 0.0:
        mean *= -1.0
    return _normalize_quaternion(mean)


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    q0 = _normalize_quaternion(first)
    q1 = _normalize_quaternion(second)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0)))


def _validate_sample(sample: PicoBodyTrackingSample) -> PicoBodyTrackingSample:
    positions = [
        np.asarray(sample.left_position, dtype=np.float64).reshape(3),
        np.asarray(sample.right_position, dtype=np.float64).reshape(3),
        np.asarray(sample.neck_position, dtype=np.float64).reshape(3),
    ]
    orientations = [
        _normalize_quaternion(sample.left_orientation_wxyz),
        _normalize_quaternion(sample.right_orientation_wxyz),
        _normalize_quaternion(sample.neck_orientation_wxyz),
    ]
    if not all(np.isfinite(value).all() for value in positions + orientations):
        raise ValueError("non-finite PICO body pose")
    return PicoBodyTrackingSample(
        *[value.copy() for value in positions],
        *[value.copy() for value in orientations],
        int(sample.source_timestamp_us),
        int(sample.frame_index),
        str(sample.source),
    )


class PicoBodyCartesianTargetTracker:
    """Validate and expose direct PICO pelvis-frame wrist targets."""

    def __init__(self, config: PicoCartesianTargetConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._calibration_samples: list[PicoBodyTrackingSample] = []
        self._calibration_start_us = 0
        self._calibrated = False
        self._left_position_anchor: np.ndarray | None = None
        self._right_position_anchor: np.ndarray | None = None
        self._previous_observed: PicoBodyTrackingSample | None = None
        self._last_accepted_sample: PicoBodyTrackingSample | None = None
        self._last_accepted_source_us = 0
        self._recovering = False
        self._recovery_count = 0
        self._last_targets: PicoCartesianTargets | None = None

    @property
    def initialized(self) -> bool:
        return self._calibrated

    def _jump_reason(
        self,
        previous: PicoBodyTrackingSample,
        current: PicoBodyTrackingSample,
    ) -> str | None:
        position_jump = max(
            float(np.linalg.norm(current.left_position - previous.left_position)),
            float(np.linalg.norm(current.right_position - previous.right_position)),
        )
        if position_jump > self.config.position_jump_m:
            return f"position_jump_{position_jump:.3f}m"
        orientation_jump = max(
            _quaternion_distance_rad(
                previous.left_orientation_wxyz, current.left_orientation_wxyz
            ),
            _quaternion_distance_rad(
                previous.right_orientation_wxyz, current.right_orientation_wxyz
            ),
        )
        if orientation_jump > self.config.orientation_jump_rad:
            return f"orientation_jump_{np.rad2deg(orientation_jump):.1f}deg"
        return None

    def _reset_calibration(self, sample: PicoBodyTrackingSample) -> None:
        self._calibration_samples = [sample]
        self._calibration_start_us = sample.source_timestamp_us

    def _calibrate(
        self,
        *,
        sample: PicoBodyTrackingSample,
        robot_left_transform: np.ndarray | None = None,
        robot_right_transform: np.ndarray | None = None,
    ) -> PicoCartesianTargets:
        if not self._calibration_samples:
            self._reset_calibration(sample)
        else:
            previous = self._calibration_samples[-1]
            gap_sec = (sample.source_timestamp_us - previous.source_timestamp_us) / 1.0e6
            reason = self._jump_reason(previous, sample)
            if gap_sec > self.config.gap_hold_sec or gap_sec <= 0.0 or reason is not None:
                self._reset_calibration(sample)
            else:
                self._calibration_samples.append(sample)

        elapsed_sec = (
            sample.source_timestamp_us - self._calibration_start_us
        ) / 1.0e6
        if (
            elapsed_sec < self.config.calibration_sec
            or len(self._calibration_samples) < self.config.min_calibration_frames
        ):
            return self._empty_targets(
                sample,
                reason=(
                    f"calibrating_{elapsed_sec:.2f}/"
                    f"{self.config.calibration_sec:.2f}s"
                ),
            )

        # PICO manager already performs the single pelvis-frame alignment.
        # This window only confirms a stable stream and selects a robust first
        # direct target. Robot FK arguments remain for API compatibility only.
        del robot_left_transform, robot_right_transform
        samples = self._calibration_samples
        left_position = np.median(
            np.stack([value.left_position for value in samples]), axis=0
        )
        right_position = np.median(
            np.stack([value.right_position for value in samples]), axis=0
        )
        left_orientation = _quaternion_mean_wxyz(
            np.stack([value.left_orientation_wxyz for value in samples])
        )
        right_orientation = _quaternion_mean_wxyz(
            np.stack([value.right_orientation_wxyz for value in samples])
        )
        left_tf = np.eye(4, dtype=np.float64)
        right_tf = np.eye(4, dtype=np.float64)
        left_tf[:3, 3] = left_position
        right_tf[:3, 3] = right_position
        left_tf[:3, :3] = _rotation_from_wxyz(left_orientation)
        right_tf[:3, :3] = _rotation_from_wxyz(right_orientation)
        self._calibrated = True
        self._left_position_anchor = left_position.copy()
        self._right_position_anchor = right_position.copy()
        self._previous_observed = sample
        self._last_accepted_sample = PicoBodyTrackingSample(
            left_position.copy(),
            right_position.copy(),
            sample.neck_position.copy(),
            left_orientation.copy(),
            right_orientation.copy(),
            sample.neck_orientation_wxyz.copy(),
            int(sample.source_timestamp_us),
            int(sample.frame_index),
            str(sample.source),
        )
        self._last_accepted_source_us = sample.source_timestamp_us
        self._recovering = False
        self._recovery_count = 0
        output = PicoCartesianTargets(
            left_transform=left_tf,
            right_transform=right_tf,
            source_timestamp_us=int(sample.source_timestamp_us),
            frame_index=int(sample.frame_index),
            valid=True,
            held=False,
            reason="initialized",
            left_raw_position=left_position.copy(),
            right_raw_position=right_position.copy(),
            calibration_frames=len(self._calibration_samples),
            recovery_count=0,
            source=str(sample.source),
        )
        self._last_targets = output
        return output

    def update(
        self,
        *,
        sample: PicoBodyTrackingSample,
        robot_left_transform: np.ndarray | None = None,
        robot_right_transform: np.ndarray | None = None,
    ) -> PicoCartesianTargets:
        try:
            sample = _validate_sample(sample)
        except (TypeError, ValueError):
            return self.hold(reason="invalid_body_pose")

        if not self.initialized:
            return self._calibrate(
                sample=sample,
                robot_left_transform=robot_left_transform,
                robot_right_transform=robot_right_transform,
            )

        previous = self._previous_observed
        self._previous_observed = sample
        if previous is None:
            self._recovering = True
            self._recovery_count = 0
            return self.hold(
                source_timestamp_us=sample.source_timestamp_us,
                frame_index=sample.frame_index,
                reason="missing_previous_sample",
            )

        gap_sec = (sample.source_timestamp_us - previous.source_timestamp_us) / 1.0e6
        if gap_sec > self.config.gap_hold_sec or gap_sec <= 0.0:
            self._recovering = True
            self._recovery_count = 0
            return self.hold(
                source_timestamp_us=sample.source_timestamp_us,
                frame_index=sample.frame_index,
                reason=f"timestamp_gap_{1000.0 * gap_sec:.1f}ms",
            )

        continuity_reference = self._last_accepted_sample or previous
        jump_reason = self._jump_reason(continuity_reference, sample)
        if jump_reason is not None:
            self._recovering = True
            self._recovery_count = 0
            return self.hold(
                source_timestamp_us=sample.source_timestamp_us,
                frame_index=sample.frame_index,
                reason=jump_reason,
            )

        if self._recovering:
            self._recovery_count += 1
            if self._recovery_count < max(1, self.config.recovery_frames):
                return self.hold(
                    source_timestamp_us=sample.source_timestamp_us,
                    frame_index=sample.frame_index,
                    reason=(
                        f"recovering_{self._recovery_count}/"
                        f"{self.config.recovery_frames}"
                    ),
                )
            self._recovering = False
            self._recovery_count = 0

        self._last_accepted_source_us = sample.source_timestamp_us
        self._last_accepted_sample = sample
        return self._targets_from_sample(sample, reason="tracking")

    def _targets_from_sample(
        self,
        sample: PicoBodyTrackingSample,
        *,
        reason: str,
    ) -> PicoCartesianTargets:
        if self._left_position_anchor is None or self._right_position_anchor is None:
            raise RuntimeError("PICO position gain anchors are unavailable")
        scale = np.asarray(self.config.scale_xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("PICO position gains must be finite and positive")
        left_position = self._left_position_anchor + scale * (
            sample.left_position - self._left_position_anchor
        )
        right_position = self._right_position_anchor + scale * (
            sample.right_position - self._right_position_anchor
        )
        left_tf = np.eye(4, dtype=np.float64)
        right_tf = np.eye(4, dtype=np.float64)
        left_tf[:3, 3] = left_position
        right_tf[:3, 3] = right_position
        left_tf[:3, :3] = _rotation_from_wxyz(sample.left_orientation_wxyz)
        right_tf[:3, :3] = _rotation_from_wxyz(sample.right_orientation_wxyz)
        output = PicoCartesianTargets(
            left_transform=left_tf,
            right_transform=right_tf,
            source_timestamp_us=int(sample.source_timestamp_us),
            frame_index=int(sample.frame_index),
            valid=True,
            held=False,
            reason=str(reason),
            left_raw_position=sample.left_position.copy(),
            right_raw_position=sample.right_position.copy(),
            calibration_frames=len(self._calibration_samples),
            recovery_count=int(self._recovery_count),
            source=str(sample.source),
        )
        self._last_targets = output
        return output

    def _empty_targets(
        self,
        sample: PicoBodyTrackingSample,
        *,
        reason: str,
    ) -> PicoCartesianTargets:
        return PicoCartesianTargets(
            left_transform=np.eye(4, dtype=np.float64),
            right_transform=np.eye(4, dtype=np.float64),
            source_timestamp_us=int(sample.source_timestamp_us),
            frame_index=int(sample.frame_index),
            valid=False,
            held=True,
            reason=str(reason),
            left_raw_position=np.zeros(3, dtype=np.float64),
            right_raw_position=np.zeros(3, dtype=np.float64),
            calibration_frames=len(self._calibration_samples),
            recovery_count=int(self._recovery_count),
            source=str(sample.source),
        )

    def hold(
        self,
        *,
        source_timestamp_us: int | None = None,
        frame_index: int | None = None,
        reason: str = "held",
    ) -> PicoCartesianTargets:
        if self._last_targets is None:
            sample = PicoBodyTrackingSample(
                *(np.zeros(3, dtype=np.float64) for _ in range(3)),
                *(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) for _ in range(3)),
                int(source_timestamp_us or 0),
                int(frame_index if frame_index is not None else -1),
                "vr_pose_debug",
            )
            return self._empty_targets(sample, reason=reason)
        previous = self._last_targets
        return PicoCartesianTargets(
            left_transform=previous.left_transform.copy(),
            right_transform=previous.right_transform.copy(),
            source_timestamp_us=int(
                source_timestamp_us
                if source_timestamp_us is not None
                else previous.source_timestamp_us
            ),
            frame_index=int(
                frame_index if frame_index is not None else previous.frame_index
            ),
            valid=True,
            held=True,
            reason=str(reason),
            left_raw_position=previous.left_raw_position.copy(),
            right_raw_position=previous.right_raw_position.copy(),
            calibration_frames=int(previous.calibration_frames),
            recovery_count=int(self._recovery_count),
            source=str(previous.source),
        )
