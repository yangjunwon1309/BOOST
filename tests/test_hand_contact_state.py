from __future__ import annotations

from boost.control.hand_mppi.contact_state import HandContactStateEstimator


def _sample_state(estimator, command, measured, stamp):
    estimator.update_command(command, stamp)
    estimator.update_state(measured, [0.0] * 6, [0.0] * 6, stamp)


def test_channel_mapping_hysteresis_and_minimum_contact_duration() -> None:
    estimator = HandContactStateEstimator(tau_on=0.08, tau_off=0.04, minimum_duration_s=0.10)
    command = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    measured = [0.0] * 6
    _sample_state(estimator, command, measured, 1.00)
    assert estimator.snapshot(1.00)["contact_raw"] == [False, True, False, False, False]
    assert estimator.snapshot(1.00)["contact"][1] is False

    _sample_state(estimator, command, measured, 1.11)
    assert estimator.snapshot(1.11)["contact"][1] is True
    assert estimator.summary(1.11)["index"]["contact_intervals"] == 1

    relaxed = command.copy()
    relaxed[2] = 0.02
    _sample_state(estimator, relaxed, measured, 1.12)
    assert estimator.snapshot(1.12)["contact"][1] is True
    _sample_state(estimator, relaxed, measured, 1.23)
    assert estimator.snapshot(1.23)["contact"][1] is False


def test_thumb_uses_the_proximal_channel_and_stale_inputs_make_contacts_null() -> None:
    estimator = HandContactStateEstimator(minimum_duration_s=0.0)
    command = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    measured = [0.0] * 6
    _sample_state(estimator, command, measured, 2.0)
    result = estimator.snapshot(2.0)
    assert result["contact_raw"] == [True, False, False, False, False]
    assert estimator.snapshot(2.30)["available"] is False
    assert estimator.snapshot(2.30)["contact"] == [None] * 5
    assert estimator.snapshot(2.30)["contact_raw"] == [None] * 5
