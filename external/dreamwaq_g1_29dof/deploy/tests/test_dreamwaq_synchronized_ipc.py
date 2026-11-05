import multiprocessing as mp
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)
ROOT = os.path.dirname(DEPLOY)
sys.path.insert(0, DEPLOY)
sys.path.insert(0, ROOT)

from dreamwaq_action_feedback import PublishedActionTracker  # noqa: E402
from dreamwaq_actor_history import AcceptedActorActionHistory  # noqa: E402
from dreamwaq_policy_ipc import (  # noqa: E402
    DreamWaQPolicyIPC,
    PolicyActionResponse,
)
from dreamwaq_shadow_timing import (  # noqa: E402
    ShadowWriterCadenceProbe,
    build_shadow_summary,
    write_shadow_report,
)
from run_shadow import ReadOnlyG1State  # noqa: E402
from run_dreamwaq import parse_args  # noqa: E402
from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract, JointOrderMap  # noqa: E402


EXPORT = os.path.join(ROOT, "checkpoints/new2_3_main_plus50k_249999/exported")


def test_arm_smoothing_tau_cli_override_is_optional_and_nonnegative():
    assert parse_args(["--net", "eth0"]).arm_smoothing_tau is None
    assert parse_args(["--net", "eth0", "--arm-smoothing-tau", "0.0"]).arm_smoothing_tau == 0.0
    assert parse_args(["--net", "eth0", "--arm-smoothing-tau", "0.04"]).arm_smoothing_tau == 0.04


def _state(sequence, previous):
    q = np.zeros(29, dtype=np.float32)
    dq = np.zeros(29, dtype=np.float32)
    quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    gyro = np.zeros(3, dtype=np.float32)
    command = np.asarray((0.2, 0.0, 0.0), dtype=np.float32)
    return dict(
        q29=q,
        dq29=dq,
        quaternion_wxyz=quat,
        angular_velocity_body=gyro,
        command=command,
        sequence=sequence,
        monotonic_ns=1_000_000 + sequence,
        previous_lower_action=previous,
        previous_action_sequence=sequence - 1,
    )


def _response(sequence, value=0.25):
    return PolicyActionResponse(
        source_state_sequence=sequence,
        source_state_monotonic_ns=1_000_000 + sequence,
        source_state_published_ns=1_500_000 + sequence,
        inference_started_ns=2_000_000 + sequence,
        inference_finished_ns=3_000_000 + sequence,
        skipped_state_count=0,
        q_target15=np.full(15, value, dtype=np.float32),
        normalized_lower_action=np.full(12, value, dtype=np.float32),
        mode="walk",
        inference_ms=1.0,
        max_action=abs(value),
    )


def _hold_state_lock(ipc, entered, release):
    lock = ipc.state.get_lock()
    lock.acquire()
    try:
        entered.set()
        release.wait(timeout=1.0)
    finally:
        lock.release()
        # Match publish_state(): readers are woken only after the lock is free.
        ipc.state_event.set()


def test_ipc_preserves_state_action_sequence_and_previous_action():
    ipc = DreamWaQPolicyIPC(mp.get_context("spawn"))
    previous = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    ipc.publish_state(**_state(0, previous))
    request = ipc.wait_for_state(-1, timeout_s=0.01)
    assert request is not None
    assert request.sequence == 0
    assert request.request_published_ns > request.monotonic_ns
    assert request.previous_action_sequence == -1
    np.testing.assert_allclose(request.previous_lower_action, previous)

    ipc.publish_action(_response(0))
    response = ipc.wait_for_action(0, timeout_s=0.01)
    assert response is not None
    assert response.source_state_sequence == request.sequence
    np.testing.assert_allclose(response.normalized_lower_action, 0.25)


def test_state_lock_contention_retries_without_killing_policy_ipc():
    ctx = mp.get_context("spawn")
    ipc = DreamWaQPolicyIPC(ctx, lock_timeout_s=0.002)
    ipc.publish_state(**_state(0, np.zeros(12, dtype=np.float32)))
    entered = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(target=_hold_state_lock, args=(ipc, entered, release))
    holder.start()
    assert entered.wait(timeout=1.0)

    releaser = threading.Timer(0.010, release.set)
    releaser.start()
    try:
        request = ipc.wait_for_state(-1, timeout_s=0.100)
    finally:
        releaser.cancel()
        release.set()
        holder.join(timeout=1.0)
    assert holder.exitcode == 0
    assert request is not None
    assert request.sequence == 0


def test_ipc_supports_waist_control_action_dimension():
    ipc = DreamWaQPolicyIPC(mp.get_context("spawn"), action_dim=15)
    previous = np.linspace(-0.5, 0.5, 15, dtype=np.float32)
    payload = _state(0, previous)
    ipc.publish_state(**payload)
    request = ipc.wait_for_state(-1, timeout_s=0.01)
    np.testing.assert_allclose(request.previous_lower_action, previous)
    response = _response(0)
    response = PolicyActionResponse(
        **{**response.__dict__, "normalized_lower_action": previous}
    )
    ipc.publish_action(response)
    received = ipc.wait_for_action(0, timeout_s=0.01)
    np.testing.assert_allclose(received.normalized_lower_action, previous)


def test_ipc_ignores_late_action_and_waits_for_exact_sequence():
    ipc = DreamWaQPolicyIPC(mp.get_context("spawn"))
    ipc.publish_state(**_state(0, np.zeros(12, dtype=np.float32)))
    ipc.publish_action(_response(0))
    ipc.publish_state(**_state(1, np.zeros(12, dtype=np.float32)))
    assert ipc.wait_for_action(1, timeout_s=0.002) is None

    def publish_current():
        time.sleep(0.005)
        ipc.publish_action(_response(1, value=-0.1))

    thread = threading.Thread(target=publish_current)
    thread.start()
    response = ipc.wait_for_action(1, timeout_s=0.05)
    thread.join()
    assert response is not None
    assert response.source_state_sequence == 1
    np.testing.assert_allclose(response.normalized_lower_action, -0.1)


def test_ipc_exposes_an_injected_late_action_as_a_deadline_miss():
    ipc = DreamWaQPolicyIPC(mp.get_context("spawn"))
    ipc.publish_state(**_state(0, np.zeros(12, dtype=np.float32)))

    def publish_too_late():
        time.sleep(0.020)
        ipc.publish_action(_response(0))

    thread = threading.Thread(target=publish_too_late)
    thread.start()
    assert ipc.wait_for_action(0, timeout_s=0.005) is None
    thread.join()
    assert ipc.latest_action().source_state_sequence == 0


def test_published_action_feedback_uses_command_before_state_timestamp():
    contract = DreamWaQ29Contract.load(EXPORT)
    tracker = PublishedActionTracker(contract)
    lower = JointOrderMap().lower_sdk_indices

    first = contract.default_q_sdk.copy()
    first[lower] += contract.physical_action_scale * 0.2
    second = contract.default_q_sdk.copy()
    second[lower] -= contract.physical_action_scale * 0.4
    tracker.record(first, {"action_sequence": 7}, 100)
    tracker.record(second, {"action_sequence": 8}, 200)

    before = tracker.at_or_before(99)
    between = tracker.at_or_before(150)
    after = tracker.at_or_before(250)
    assert before.sequence == -1
    np.testing.assert_allclose(before.normalized, 0.0)
    assert between.sequence == 7
    np.testing.assert_allclose(between.normalized, 0.2, atol=1.0e-6)
    assert after.sequence == 8
    np.testing.assert_allclose(after.normalized, -0.4, atol=1.0e-6)


def test_accepted_actor_history_holds_on_timeout_and_ignores_writer_state():
    history = AcceptedActorActionHistory()
    initial = history.snapshot()
    assert initial.sequence == -1
    np.testing.assert_allclose(initial.normalized, 0.0)

    actor = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    history.accept(7, actor)
    committed = history.snapshot()
    assert committed.sequence == 7
    np.testing.assert_allclose(committed.normalized, actor)

    # A policy deadline miss is a hold: no new actor output is committed.
    held = history.snapshot()
    assert held.sequence == 7
    np.testing.assert_allclose(held.normalized, actor)


def test_policy_uses_externally_supplied_actor_history():
    import pytest

    pytest.importorskip("torch")
    from dreamwaq_policy import DreamWaQPolicy

    policy = DreamWaQPolicy(EXPORT, device="cpu")
    try:
        q = policy.contract.default_q_sdk.copy()
        dq = np.zeros(29, dtype=np.float32)
        quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        gyro = np.zeros(3, dtype=np.float32)
        previous_actor = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        policy.set_command((0.2, 0.0, 0.0))
        policy.update(q, dq, quat, gyro, previous_lower_action=previous_actor)
        np.testing.assert_allclose(policy.previous_lower_action, previous_actor)
        assert policy.last_lower_action.shape == (12,)
        assert np.all(np.isfinite(policy.last_lower_action))
    finally:
        policy.close()


def test_spawned_policy_worker_replies_to_the_exact_request():
    import pytest

    pytest.importorskip("torch")
    import dreamwaq_policy_process

    context = mp.get_context("spawn")
    ipc = DreamWaQPolicyIPC(context)
    process = context.Process(
        target=dreamwaq_policy_process.run,
        args=(ipc, EXPORT, "cpu", 0.02, None),
    )
    process.start()
    try:
        assert ipc.ready_event.wait(10.0)
        contract = DreamWaQ29Contract.load(EXPORT)
        previous = np.zeros(12, dtype=np.float32)
        for sequence in range(8):
            state = _state(sequence, previous)
            state["q29"] = contract.default_q_sdk.copy()
            state["monotonic_ns"] = time.monotonic_ns()
            request_published_ns = ipc.publish_state(**state)
            response = ipc.wait_for_action(sequence, timeout_s=0.015)
            assert response is not None
            assert response.source_state_sequence == sequence
            assert response.source_state_monotonic_ns == state["monotonic_ns"]
            assert response.source_state_published_ns == request_published_ns
            assert response.skipped_state_count == 0
            previous = response.normalized_lower_action
    finally:
        ipc.request_stop()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    assert process.exitcode == 0


def test_spawned_policy_worker_can_idle_before_first_state():
    import pytest

    pytest.importorskip("torch")
    import dreamwaq_policy_process

    context = mp.get_context("spawn")
    ipc = DreamWaQPolicyIPC(context)
    process = context.Process(
        target=dreamwaq_policy_process.run,
        args=(ipc, EXPORT, "cpu", 0.02, None),
    )
    process.start()
    try:
        assert ipc.ready_event.wait(10.0)
        # Real/shadow setup can take much longer than one worker wait period.
        # Deadline expiry before the first request must be an idle poll, not an
        # attempted zero-timeout lock acquisition or worker failure.
        time.sleep(0.10)
        assert process.is_alive()
    finally:
        ipc.request_stop()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    assert process.exitcode == 0


def _timing_row(sequence, *, accepted=True, state_age_ms=1.0):
    state_received_ns = 1_000_000_000 + sequence * 20_000_000
    tick_started_ns = state_received_ns + int(state_age_ms * 1.0e6)
    request_published_ns = tick_started_ns + 100_000
    response_received_ns = request_published_ns + 1_100_000
    return {
        "sequence": sequence,
        "source_tick": 100 + sequence * 20,
        "source_tick_delta": None if sequence == 0 else 20,
        "source_sequence": sequence,
        "state_sample_repeated": 0,
        "accepted": int(accepted),
        "response_sequence": sequence if accepted else sequence - 1,
        "response_sequence_lag": 0 if accepted else 1,
        "skipped_state_count": 0 if accepted else None,
        "tick_started_ns": tick_started_ns,
        "state_received_ns": state_received_ns,
        "request_published_ns": request_published_ns,
        "inference_started_ns": request_published_ns + 200_000 if accepted else None,
        "inference_finished_ns": request_published_ns + 1_000_000 if accepted else None,
        "response_received_ns": response_received_ns,
        "state_age_ms": state_age_ms,
        "parent_prepare_ms": 0.1,
        "ipc_wakeup_ms": 0.2 if accepted else None,
        "worker_compute_ms": 0.8 if accepted else None,
        "backend_inference_ms": 0.6 if accepted else None,
        "worker_overhead_ms": 0.2 if accepted else None,
        "response_transfer_ms": 0.1 if accepted else None,
        "request_roundtrip_ms": 1.1 if accepted else None,
        "state_to_response_ms": 2.1 if accepted else None,
        "loop_work_ms": 1.2,
        "tick_period_ms": 20.0,
        "tick_lateness_ms": 0.1,
        "deadline_margin_ms": 13.9 if accepted else None,
        "shadow_writer_handoff_ms": 2.0 if accepted else None,
        "shadow_writer_callback_ms": 0.2 if accepted else None,
        "shadow_writer_ready_ms": 2.2 if accepted else None,
        "state_to_shadow_writer_ms": 4.1 if accepted else None,
        "state_to_shadow_writer_ready_ms": 4.3 if accepted else None,
        "shadow_writer_tick_lateness_ms": 0.1 if accepted else None,
        "shadow_writer_tick_ns": response_received_ns + 2_000_000 if accepted else None,
        "mode": "walk" if accepted else "timeout",
        "max_action": 0.2 if accepted else None,
        "target_delta": 0.1 if accepted else None,
    }


def _summary(rows):
    return build_shadow_summary(
        rows,
        policy_timeout_ms=15.0,
        control_dt_ms=20.0,
        max_state_age_ms=10.0,
        max_tick_jitter_ms=5.0,
        max_writer_handoff_ms=8.0,
        expected_lowstate_hz=1000.0,
        reader_diagnostics={
            "read_count": 500,
            "unique_count": 500,
            "duplicate_count": 0,
            "replay_count": 0,
        },
        reader_elapsed_s=2.0,
    )


def test_shadow_summary_detects_deadline_and_state_age_failures(tmp_path):
    clean_rows = [_timing_row(index) for index in range(20)]
    clean = _summary(clean_rows)
    assert clean["verdict"] == "PASS_NO_DELAY"
    assert clean["timeout_count"] == 0

    delayed_rows = clean_rows + [_timing_row(20, accepted=False)]
    delayed = _summary(delayed_rows)
    assert delayed["verdict"] == "FAIL_DELAY_DETECTED"
    assert delayed["timeout_count"] == 1

    stale_rows = [_timing_row(index, state_age_ms=12.0) for index in range(20)]
    stale = _summary(stale_rows)
    assert stale["verdict"] == "FAIL_DELAY_DETECTED"
    assert any("LowState age" in message for message in stale["errors"])

    json_path, csv_path = write_shadow_report(tmp_path / "shadow.json", clean, clean_rows)
    assert json_path.is_file()
    assert csv_path.is_file()


def test_shadow_summary_detects_hardware_tick_backlog_and_writer_delay():
    backlog_rows = [_timing_row(index) for index in range(20)]
    for index, row in enumerate(backlog_rows):
        row["source_tick"] = 100 + index * 4
        row["source_tick_delta"] = None if index == 0 else 4
    backlog = _summary(backlog_rows)
    assert backlog["verdict"] == "FAIL_DELAY_DETECTED"
    assert any("DDS backlog/stall" in message for message in backlog["errors"])

    writer_rows = [_timing_row(index) for index in range(20)]
    for row in writer_rows:
        row["shadow_writer_ready_ms"] = 12.0
    writer_delay = _summary(writer_rows)
    assert writer_delay["verdict"] == "FAIL_DELAY_DETECTED"
    assert any("response-to-ready" in message for message in writer_delay["errors"])


class _FakeMotor:
    q = 0.0
    dq = 0.0


class _FakeImu:
    quaternion = (1.0, 0.0, 0.0, 0.0)
    gyroscope = (0.0, 0.0, 0.0)


class _FakeLowState:
    def __init__(self, tick):
        self.tick = tick
        self.motor_state = [_FakeMotor() for _ in range(35)]
        self.imu_state = _FakeImu()
        self.wireless_remote = bytes(40)
        self.mode_machine = 5


class _FakeSubscriber:
    def __init__(self, ticks):
        self.messages = [_FakeLowState(tick) for tick in ticks]

    def Init(self):
        return None

    def Read(self):
        return self.messages.pop(0) if self.messages else None

    def Close(self):
        return None


class _FakeConfig:
    joint2motor_idx = list(range(29))


def test_shadow_lowstate_reader_does_not_refresh_duplicate_tick_timestamp():
    state = ReadOnlyG1State(
        "unused",
        _FakeConfig(),
        reader_hz=1000.0,
        subscriber=_FakeSubscriber((10, 10, 11)),
    )
    try:
        deadline = time.monotonic() + 0.1
        while state.diagnostics()["unique_count"] < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        diagnostics = state.diagnostics()
        assert diagnostics["unique_count"] == 2
        assert diagnostics["duplicate_count"] == 1
        assert diagnostics["replay_count"] == 0
        assert state.reader_fault is None
    finally:
        state.close()


def test_shadow_lowstate_reader_flags_replayed_tick():
    state = ReadOnlyG1State(
        "unused",
        _FakeConfig(),
        reader_hz=1000.0,
        subscriber=_FakeSubscriber((20, 19)),
    )
    try:
        deadline = time.monotonic() + 0.1
        while state.reader_fault is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert state.reader_fault is not None
        assert state.diagnostics()["replay_count"] == 1
    finally:
        state.close()


def test_shadow_writer_probe_records_next_250hz_slot():
    probe = ShadowWriterCadenceProbe(250.0)
    probe.start()
    try:
        response_ns = time.monotonic_ns()
        probe.submit(3, response_ns - 1_000_000, response_ns)
        deadline = time.monotonic() + 0.05
        while 3 not in probe.results() and time.monotonic() < deadline:
            time.sleep(0.001)
    finally:
        probe.stop()
    result = probe.results()[3]
    assert 0.0 <= result["shadow_writer_handoff_ms"] < 12.0
    assert result["shadow_writer_ready_ms"] >= result["shadow_writer_handoff_ms"]
    assert result["state_to_shadow_writer_ms"] >= result["shadow_writer_handoff_ms"]


def test_full_shadow_runner_writes_no_delay_report(monkeypatch, tmp_path):
    import json
    import run_shadow as run_shadow_module
    import dreamwaq_policy_process
    from robot_control.config import Config

    config = Config(os.path.join(DEPLOY, "robot_control/configs/g1_homie.yaml"))

    class FakeReadOnlyState:
        def __init__(self, _net, config, **_kwargs):
            self.config = config
            self.sequence = 0
            self.started = time.monotonic()
            self.reader_fault = None
            self.reader_cpu_affinity = None

        def read_controller_input(self):
            return SimpleNamespace(keys=0, lx=0.0, ly=0.0, rx=0.0, ry=0.0)

        def get_robot_state(self):
            self.sequence += 1
            msg = _FakeLowState(self.sequence * 20)
            return (
                np.asarray(self.config.default_angles, dtype=np.float32),
                np.zeros(29, dtype=np.float32),
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                msg,
                time.monotonic_ns() - 1_000_000,
                msg.tick,
                self.sequence,
            )

        def diagnostics(self):
            elapsed = max(0.01, time.monotonic() - self.started)
            unique = int(elapsed * 250.0)
            return {
                "read_count": unique,
                "unique_count": unique,
                "duplicate_count": 0,
                "replay_count": 0,
                "configured_rate_hz": 250.0,
            }

        def close(self):
            return None

    monkeypatch.setattr(run_shadow_module, "ReadOnlyG1State", FakeReadOnlyState)
    context = mp.get_context("spawn")
    ipc = DreamWaQPolicyIPC(context)
    process = context.Process(
        target=dreamwaq_policy_process.run,
        args=(ipc, EXPORT, "cpu", 0.02, None),
    )
    process.start()
    report = tmp_path / "full_shadow.json"
    args = SimpleNamespace(
        net="unused",
        shadow_reader_hz=250.0,
        writer_rate_hz=250.0,
        policy_timeout_ms=15.0,
        duration=0.12,
        shadow_warmup_s=0.0,
        status_every=10.0,
        expected_mode_machine=5,
        max_policy_timeouts=3,
        shadow_max_state_age_ms=10.0,
        shadow_max_tick_jitter_ms=5.0,
        shadow_max_writer_handoff_ms=8.0,
        shadow_expected_lowstate_hz=1000.0,
        shadow_production_load=False,
        shadow_report=str(report),
        policy_cpu=None,
        lowstate_reader_cpu=None,
        arm_reader_cpu=None,
        writer_cpu=None,
        control_cpu=None,
        io_cpus=tuple(os.sched_getaffinity(0)),
        device="cpu",
    )
    try:
        assert ipc.ready_event.wait(10.0)
        result = run_shadow_module.run_shadow(args, config, ipc, process)
    finally:
        ipc.request_stop()
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"] in ("PASS_NO_DELAY", "PASS_WITH_WARNINGS")
    assert payload["timeout_count"] == 0
    assert payload["sequence_error_count"] == 0
    assert payload["metrics_ms"]["request_roundtrip_ms"]["count"] > 0
    assert report.with_suffix(".csv").is_file()
