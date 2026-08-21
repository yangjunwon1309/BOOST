from __future__ import annotations

from path_core.intention.grasp_engagement import GraspEngagementDetector


def test_grasp_detector_waits_for_close_command_fallback_delay() -> None:
    detector = GraspEngagementDetector(command_delay_sec=0.5)

    for _ in range(4):
        assert detector.update(commanded_closed=True, jaw_gap_m=None, dt=0.1) is False
    assert detector.update(commanded_closed=True, jaw_gap_m=None, dt=0.1) is True
    assert detector.update(commanded_closed=False, jaw_gap_m=None, dt=0.1) is False


def test_grasp_detector_requires_stalled_nonempty_jaws_when_gap_is_available() -> None:
    detector = GraspEngagementDetector(engage_ticks=3, release_ticks=2)

    # A freely closing jaw is not a grasp even while close is commanded.
    assert detector.update(commanded_closed=True, jaw_gap_m=0.040, dt=0.1) is False
    assert detector.update(commanded_closed=True, jaw_gap_m=0.030, dt=0.1) is False
    assert detector.update(commanded_closed=True, jaw_gap_m=0.020, dt=0.1) is False

    # A non-empty, stalled gap is an object held between the fingers.
    assert detector.update(commanded_closed=True, jaw_gap_m=0.020, dt=0.1) is False
    assert detector.update(commanded_closed=True, jaw_gap_m=0.020, dt=0.1) is False
    assert detector.update(commanded_closed=True, jaw_gap_m=0.020, dt=0.1) is True
