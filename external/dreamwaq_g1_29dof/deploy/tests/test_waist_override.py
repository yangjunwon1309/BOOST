"""Thor's waist-yaw request: read from rt/arm_sdk, applied by the writer, hidden from the policy."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import arm_target_source as source  # noqa: E402
from dreamwaq_interface.g1_29dof.contract import (  # noqa: E402
    DreamWaQ29Contract,
    JointOrderMap,
    LOWER_BODY_JOINT_NAMES,
    WAIST_CONTROL_JOINT_NAMES,
)
from waist_override import WAIST_YAW_SDK_INDEX, WaistYawOverride  # noqa: E402

TURN = -0.7854  # 45 deg to the robot's right
CHECKPOINTS = ROOT / "checkpoints"


def run_for(waist, request, seconds, start_ns=0, dt=0.02):
    now = start_ns
    events = []
    for _ in range(int(round(seconds / dt))):
        now += int(dt * 1e9)
        event = waist.step(request, now)
        if event:
            events.append(event)
    return now, events


def test_turn_ramps_at_the_rate_limit_and_returns_to_default():
    waist = WaistYawOverride(enabled=True)
    assert waist.step(TURN, 0) == "begin"  # the request alone starts it
    now, events = run_for(waist, TURN, 0.5)
    assert waist.offset == pytest.approx(-0.25, abs=1e-6)
    assert waist.rate == pytest.approx(-0.5, abs=1e-6)
    assert events == []
    now, _ = run_for(waist, TURN, 1.2, now)
    assert waist.offset == pytest.approx(TURN)
    assert waist.rate == pytest.approx(0.0)
    # The request stops (toggled off, disengaged or stale): back to default.
    _, events = run_for(waist, None, 1.7, now)
    assert waist.offset == pytest.approx(0.0) and not waist.active
    assert events == ["end"]


def test_request_is_clamped_and_long_gaps_do_not_step():
    waist = WaistYawOverride(enabled=True)
    waist.step(-2.0, 0)
    waist.step(-2.0, int(5e9))  # 5 s gap: at most 0.1 s worth of motion
    assert waist.offset == pytest.approx(-0.05)
    run_for(waist, -2.0, 3.0, int(5e9))
    assert waist.offset == pytest.approx(-waist.limit_rad)
    waist.step(float("nan"), int(9e9))
    assert waist.target == 0.0


def test_policy_reads_the_waist_at_its_default_while_the_writer_turns_it():
    waist = WaistYawOverride(enabled=True)
    waist.step(TURN, 0)
    run_for(waist, TURN, 0.5)
    q_target = np.zeros(29, dtype=np.float32)
    commanded = waist.command(q_target)
    assert commanded[WAIST_YAW_SDK_INDEX] == pytest.approx(waist.offset)
    assert q_target[WAIST_YAW_SDK_INDEX] == 0.0  # caller's array untouched
    # Measured: the joint lags the target by 0.01 rad while turning.
    q29 = np.zeros(29, dtype=np.float32)
    dq29 = np.zeros(29, dtype=np.float32)
    q29[WAIST_YAW_SDK_INDEX] = waist.offset + 0.01
    dq29[WAIST_YAW_SDK_INDEX] = waist.rate
    q_obs, dq_obs = waist.observe(q29, dq29)
    assert q_obs[WAIST_YAW_SDK_INDEX] == pytest.approx(0.01)
    assert dq_obs[WAIST_YAW_SDK_INDEX] == pytest.approx(0.0, abs=1e-6)
    assert q29[WAIST_YAW_SDK_INDEX] == pytest.approx(waist.offset + 0.01)
    assert np.array_equal(np.delete(q_obs, WAIST_YAW_SDK_INDEX), np.delete(q29, WAIST_YAW_SDK_INDEX))


def test_untouched_when_no_turn_was_ever_requested():
    waist = WaistYawOverride(enabled=True)
    run_for(waist, None, 1.0)
    q = np.arange(29, dtype=np.float32)
    assert waist.command(q) is q
    assert waist.observe(q, q) == (q, q)


def test_writer_and_policy_threads_can_share_it():
    waist = WaistYawOverride(enabled=True)
    stop = threading.Event()

    def writer():
        now = 0
        while not stop.is_set():
            now += 4_000_000
            waist.step(TURN, now)
            waist.command(np.zeros(29, dtype=np.float32))

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(2000):
            q, dq = waist.observe(np.zeros(29, dtype=np.float32), np.zeros(29, dtype=np.float32))
            assert np.isfinite(q).all() and np.isfinite(dq).all()
    finally:
        stop.set()
        thread.join()


def test_disabled_override_ignores_requests():
    waist = WaistYawOverride(enabled=False)
    run_for(waist, TURN, 1.0)
    assert waist.offset == 0.0 and waist.ignored_requests > 0


def test_only_exports_that_leave_the_spine_to_the_writer_take_a_turn():
    mapper = JointOrderMap()
    assert WAIST_YAW_SDK_INDEX not in list(mapper.sdk_indices(LOWER_BODY_JOINT_NAMES))
    assert WAIST_YAW_SDK_INDEX in list(mapper.sdk_indices(WAIST_CONTROL_JOINT_NAMES))
    for name, owned in (("new3_4_widerange_quality_390000", False), ("rb8_waist15_99999_0915", True)):
        export = CHECKPOINTS / name / "exported"
        if not export.is_dir():
            pytest.skip(f"{name} export is not checked out")
        contract = DreamWaQ29Contract.load(str(export))
        assert (WAIST_YAW_SDK_INDEX in list(mapper.sdk_indices(contract.action_joint_names))) is owned


def arm_sdk_source(weight=1.0, enable=1.0, waist=TURN, age_s=0.0):
    src = object.__new__(source.ArmSdkArmTargetSource)
    src.default_arm = np.zeros(14, dtype=np.float32)
    src.default_kp = np.ones(14, dtype=np.float32)
    src.default_kd = np.ones(14, dtype=np.float32)
    src.last_target = src.default_arm.copy()
    src.smoothed_target = src.default_arm.copy()
    src.last_kp, src.last_kd = src.default_kp.copy(), src.default_kd.copy()
    src.last_tau = np.zeros(14, dtype=np.float32)
    src.last_waist_yaw_request = None
    src.timeout, src.smoothing_tau, src._last_smooth_time = 0.2, 0.0, None
    src.tau_scale, src.tau_clamp = 1.0, np.full(14, 5.0, dtype=np.float32)
    src._lock = threading.Lock()
    src._last_stale_warn = src._last_nan_warn = 0.0
    src._on_diagnostic = None
    motors = [SimpleNamespace(q=0.0, dq=0.0, kp=1.0, kd=1.0, tau=0.0) for _ in range(35)]
    motors[source.ARM_SDK_WEIGHT_ID].q = weight
    motors[source.WAIST_ENABLE_ID].q = enable
    motors[source.WAIST_YAW_ID].q = waist
    src._latest_msg = SimpleNamespace(motor_cmd=motors)
    src._latest_msg_time = source.time.monotonic() - age_s
    return src


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, TURN),
        ({"enable": 0.0}, None),       # a publisher that only locks the waist
        ({"weight": 0.0}, None),       # teleop disengaged
        ({"age_s": 1.0}, None),        # sender gone: never hold a stale turn
        ({"waist": float("nan")}, None),
    ],
)
def test_arm_sdk_waist_request_needs_enable_weight_and_fresh_sample(kwargs, expected):
    src = arm_sdk_source(**kwargs)
    src.get_arm_command()
    assert src.get_waist_yaw_request() == (None if expected is None else pytest.approx(expected))


def test_other_arm_sources_never_request_a_turn():
    hold = source.HoldDefaultArmSource(np.zeros(14), np.ones(14), np.ones(14))
    assert hold.get_waist_yaw_request() is None
