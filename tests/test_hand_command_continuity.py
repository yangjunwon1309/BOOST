import numpy as np

from boost.control.hand_mppi.command_continuity import HandCommandContinuity


def test_brief_loss_holds_last_command_then_recovers_without_a_step() -> None:
    continuity = HandCommandContinuity()
    continuity.submit(np.full(6, 0.4), 1.0, 1.0)
    initial, _ = continuity.command(1.0)
    continuity.mark_lost()
    held, mode = continuity.command(1.5)
    np.testing.assert_array_equal(held, initial)
    assert mode == "tracking-hold"
    continuity.submit(np.ones(6), 1.6, 1.6, measured=np.zeros(6))
    recovered, mode = continuity.command(1.6)
    np.testing.assert_array_equal(recovered, held)
    assert mode == "recovering"
    for stamp in (1.63, 1.66, 1.69, 1.72):
        continuity.submit(np.ones(6), stamp, stamp)
        output, _ = continuity.command(stamp)
        assert np.all(output >= recovered)
        assert np.max(output - recovered) <= 2.0 * 0.03 + 1e-10
        recovered = output


def test_long_loss_stops_and_restarts_at_measured_pose_with_bounded_time_step() -> None:
    continuity = HandCommandContinuity()
    continuity.submit(np.ones(6), 1.0, 1.0)
    continuity.command(1.0)
    command, mode = continuity.command(2.01)
    assert command is None and mode == "stale-stop"
    measured = np.full(6, 0.15)
    continuity.submit(np.ones(6), 2.1, 2.1, measured=measured)
    output, _ = continuity.command(2.1)
    np.testing.assert_array_equal(output, measured)
    # A paused publisher must not use the whole pause as its movement budget.
    continuity.submit(np.ones(6), 2.6, 2.6, measured=measured)
    output, _ = continuity.command(2.6)
    np.testing.assert_array_equal(output, measured)


def test_stale_or_duplicate_source_is_never_refreshed_by_a_manager_heartbeat() -> None:
    continuity = HandCommandContinuity()
    continuity.submit(np.full(6, 0.3), 1.0, 1.0)
    continuity.command(1.0)
    assert not continuity.submit(np.ones(6), 1.0, 1.5)
    held, mode = continuity.command(1.5)
    np.testing.assert_array_equal(held, np.full(6, 0.3))
    assert mode == "tracking-hold"
    assert continuity.command(2.1)[0] is None


def test_normal_tracking_has_no_lpf_but_large_valid_jumps_are_rate_limited() -> None:
    continuity = HandCommandContinuity()
    continuity.submit(np.zeros(6), 1.0, 1.0)
    for index in range(1, 40):
        stamp = 1.0 + index * 0.02
        continuity.submit(np.zeros(6), stamp, stamp)
        continuity.command(stamp)
    continuity.submit(np.full(6, 0.05), 1.8, 1.8)
    output, mode = continuity.command(1.8)
    np.testing.assert_allclose(output, 0.05)
    assert mode == "tracking"
    continuity.submit(np.ones(6), 1.82, 1.82)
    limited, _ = continuity.command(1.82)
    assert np.max(limited - output) <= 5.0 * 0.02 + 1e-10


def test_dds_state_loss_reseeds_from_measurement_even_after_a_brief_gap() -> None:
    continuity = HandCommandContinuity()
    continuity.submit(np.ones(6), 1.0, 1.0)
    continuity.command(1.0)
    continuity.mark_lost(released=True)
    assert continuity.command(1.05)[0] is None
    measured = np.full(6, 0.1)
    continuity.submit(np.ones(6), 1.1, 1.1, measured=measured)
    output, mode = continuity.command(1.1)
    np.testing.assert_array_equal(output, measured)
    assert mode == "recovering"
