from __future__ import annotations

import dataclasses
from pathlib import Path
import time

import numpy as np

from boost.teleop.path_streaming.xr_source_v15 import (
    ArrivalMonitor,
    T_ROBOT_OPENXR,
    TeleVuerSource,
    XrSample,
    XrSource,
    controller_waist_targets_se3,
)


def _sample(*, arrival: float = 1.0) -> XrSample:
    identity = np.eye(4)
    return XrSample(
        head=identity.copy(), left=identity.copy(), right=identity.copy(),
        head_valid=True, left_valid=True, right_valid=True,
        left_trigger=0.8, right_trigger=0.6, left_grip=0.4, right_grip=0.2,
        left_axis=(0.1, 0.2), right_axis=(0.3, 0.4),
        buttons={"A": True, "B": True, "X": True, "Y": True},
        arrival_monotonic=arrival, source_timestamp_ns=None,
    )


def test_controller_waist_targets_se3_uses_head_relative_translation() -> None:
    head = np.eye(4)
    left = np.eye(4)
    right = np.eye(4)
    head[:3, 3] = [1.0, 2.0, 3.0]
    left[:3, 3] = [1.3, 2.2, 3.1]
    right[:3, 3] = [0.8, 1.9, 3.4]

    left_waist, right_waist, valid = controller_waist_targets_se3(left, right, head)

    assert valid is True
    np.testing.assert_allclose(left_waist[:3, 3], [0.05, -0.30, 0.65])
    np.testing.assert_allclose(right_waist[:3, 3], [-0.25, 0.20, 0.35])
    assert np.isclose(np.linalg.det(left_waist[:3, :3]), 1.0)


def test_xr_stale_releases_consent_before_invalidating_pose() -> None:
    sample = _sample()
    released = XrSource.stale_safe(sample, now=1.25)
    assert released is not None
    assert released.head_valid is True
    assert released.left_trigger == 0.0
    assert not any(released.buttons.values())

    invalid = XrSource.stale_safe(sample, now=1.36)
    assert invalid is not None
    assert invalid.head_valid is False
    assert invalid.left_valid is False
    assert invalid.right_valid is False


def test_arrival_monitor_reports_long_gaps() -> None:
    monitor = ArrivalMonitor()
    for timestamp in (1.0, 1.012, 1.026, 1.300):
        monitor.record(timestamp)
    report = monitor.report()
    assert report["source_dropouts_over_150ms"] == 1
    assert np.isclose(report["source_gap_max_ms"], 274.0)


def test_televuer_raw_controller_mapping_uses_xr_standard_sides() -> None:
    class RawTeleVuer:
        motion_data_ready = True
        head_pose = np.eye(4)
        left_arm_pose = np.eye(4)
        right_arm_pose = np.eye(4)
        left_ctrl_triggerValue = 0.2
        right_ctrl_triggerValue = 0.7
        left_ctrl_squeezeValue = 0.3
        right_ctrl_squeezeValue = 0.8
        left_ctrl_thumbstickValue = np.array([0.1, -0.2])
        right_ctrl_thumbstickValue = np.array([-0.3, 0.4])
        left_ctrl_aButton = True
        left_ctrl_bButton = False
        right_ctrl_aButton = False
        right_ctrl_bButton = True

    source = TeleVuerSource(cert_file=Path("unused-cert"), key_file=Path("unused-key"))
    source._raw = RawTeleVuer()  # Raw source is deliberately the adapter boundary.
    sample = source.poll()

    assert sample is not None
    assert sample.buttons == {"A": False, "B": True, "X": True, "Y": False, "left_menu": False}
    assert sample.left_trigger == 0.2
    assert sample.right_grip == 0.8
    assert sample.left_axis == (0.1, -0.2)
    assert np.isclose(np.linalg.det(T_ROBOT_OPENXR[:3, :3]), 1.0)


def test_televuer_hand_pinch_maps_to_path_y_b_with_shorter_stale_release() -> None:
    class RawTeleVuer:
        motion_data_ready = True
        head_pose = np.eye(4)
        left_arm_pose = np.eye(4)
        right_arm_pose = np.eye(4)
        left_hand_pinch = True
        right_hand_pinch = True
        left_hand_squeezeValue = 0.4
        right_hand_squeezeValue = 0.5

    source = TeleVuerSource(
        cert_file=Path("unused-cert"), key_file=Path("unused-key"), use_hand_tracking=True
    )
    source._raw = RawTeleVuer()
    sample = source.poll()

    assert sample is not None
    assert sample.buttons == {"A": False, "B": True, "X": False, "Y": True, "left_menu": False}
    assert sample.left_trigger == 1.0
    source._latest = dataclasses.replace(sample, arrival_monotonic=time.monotonic() - 0.11)
    released = source.latest()
    assert released is not None
    assert released.head_valid is True
    assert not any(released.buttons.values())
