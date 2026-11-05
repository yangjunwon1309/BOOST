"""Live switch between two DreamWaQ exports and back, without DDS or TorchScript."""

import dataclasses
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import arm_target_source
from command_source import CommandSource, set_command_limits
from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract
from policy_switch import SoloButtonRelease, Stillness, incompatible_fields
from run_dreamwaq import parse_args
import run_dreamwaq

L1, SELECT, X, Y = 0x0002, 0x0008, 0x0400, 0x0800
NEW3_4 = str(ROOT / "checkpoints/new3_4_widerange_quality_390000/exported")
NEW3_4_BACKPACK = str(ROOT / "checkpoints/new3_4_backpack_440000/exported")
NEW2_3 = str(ROOT / "checkpoints/new2_3_main_plus50k_249999/exported")
NEW2_3_BACKPACK = str(ROOT / "checkpoints/new2_3_backpack_lower8cm_299999/exported")
NATURE_ARGS = [
    "--net", "eth0", "--export", NEW3_4, "--ramp-time", "0.04",
    "--max-vx", "0.7", "--max-vy", "0.6", "--max-yaw", "0.6", "--arm-smoothing-tau", "0",
    "--switch-export", NEW2_3, "--switch-max-vx", "0.3", "--switch-max-vy", "0.2",
    "--switch-max-yaw", "0.4", "--switch-arm-smoothing-tau", "0.04",
    "--switch-disable-lower-joint-target-clip",
    # Three ticks: the fake legs are still unless a test moves them.
    "--switch-back-still-time", "0.06",
]


class SoloButtonReleaseTest(unittest.TestCase):
    def run_keys(self, keys, initial_keys=0):
        button = SoloButtonRelease(Y, initial_keys)
        return [button.update(k) for k in keys]

    def test_fires_once_on_release_of_a_solo_press(self):
        self.assertEqual(self.run_keys((0, Y, Y, Y, 0, 0)), [False] * 4 + [True, False])

    def test_chords_never_fire_in_either_order(self):
        # L1+Y is the controller handoff; landing Y a tick early must not count.
        self.assertFalse(any(self.run_keys((L1, L1 | Y, L1, 0))))
        self.assertFalse(any(self.run_keys((Y, L1 | Y, Y, 0))))
        self.assertFalse(any(self.run_keys((Y, X | Y, 0))))

    def test_press_in_progress_at_construction_is_ignored(self):
        self.assertEqual(self.run_keys((Y, 0, Y, 0), initial_keys=Y), [False, False, False, True])


class StillnessTest(unittest.TestCase):
    def test_needs_quiet_legs_and_base_for_the_whole_window(self):
        still = Stillness(max_leg_dq=0.5, max_gyro=0.3, ticks=3)
        quiet, stepping, turning = np.full(12, 0.1), np.r_[np.zeros(11), 2.0], np.array([0.0, 0.0, 0.5])
        seen = []
        for legs, gyro in ((quiet, np.zeros(3)),) * 3 + ((stepping, np.zeros(3)), (quiet, turning)) \
                + ((quiet, np.zeros(3)),) * 3:
            still.update(legs, gyro)
            seen.append(still.still)
        self.assertEqual(seen, [False, False, True, False, False, False, False, True])
        self.assertEqual((still.leg_dq, still.gyro), (0.1, 0.0))

    def test_a_non_finite_sample_counts_as_moving(self):
        still = Stillness(max_leg_dq=0.5, max_gyro=0.3, ticks=1)
        still.update(np.r_[np.zeros(11), np.nan], np.zeros(3))
        self.assertFalse(still.still)


class ArmReturnTest(unittest.TestCase):
    """ArmSdkArmTargetSource.return_to_default, without DDS."""

    def make_source(self, *, weight=1.0, q=0.8, kp=40.0, age_s=0.0):
        src = object.__new__(arm_target_source.ArmSdkArmTargetSource)
        src.default_arm = np.zeros(14, np.float32)
        src.default_kp = np.full(14, 100.0, np.float32)
        src.default_kd = np.full(14, 2.0, np.float32)
        src.last_target = src.smoothed_target = src.default_arm.copy()
        src.last_kp, src.last_kd = src.default_kp.copy(), src.default_kd.copy()
        src.last_tau = np.zeros(14, np.float32)
        src.last_waist_yaw_request = None
        src.timeout, src.smoothing_tau, src._last_smooth_time = 0.2, 0.0, None
        src.tau_scale, src.tau_clamp = 1.0, np.full(14, 5.0, np.float32)
        src._lock = arm_target_source.threading.Lock()
        src._last_stale_warn = src._last_nan_warn = 0.0
        src._on_diagnostic = None
        self.publish(src, weight=weight, q=q, kp=kp, age_s=age_s)
        return src

    @staticmethod
    def publish(src, *, weight, q, kp=40.0, age_s=0.0):
        motors = [SimpleNamespace(q=q, dq=0.0, kp=kp, kd=1.0, tau=1.0) for _ in range(35)]
        motors[arm_target_source.ARM_SDK_WEIGHT_ID].q = weight
        motors[arm_target_source.WAIST_ENABLE_ID].q = 0.0
        src._latest_msg = SimpleNamespace(motor_cmd=motors)
        src._latest_msg_time = time.monotonic() - age_s

    def test_a_sender_counts_only_while_engaged_and_fresh(self):
        self.assertTrue(self.make_source().sender_active())
        self.assertFalse(self.make_source(weight=0.0).sender_active())
        self.assertFalse(self.make_source(age_s=1.0).sender_active())

    def test_ramps_from_the_last_command_to_default_and_ignores_older_samples(self):
        # The mission client exited mid-command: its last sample is engaged.
        src = self.make_source(age_s=1.0)
        q, kp, kd, tau = src.get_arm_command()
        np.testing.assert_allclose((q[0], kp[0], tau[0]), (0.8, 40.0, 0.0))
        now = time.monotonic()
        with patch.object(arm_target_source.time, "monotonic", return_value=now):
            src.return_to_default(2.0)
            q, kp, kd, tau = src.get_arm_command()
        np.testing.assert_allclose((q[0], kp[0], kd[0]), (0.8, 40.0, 1.0))
        with patch.object(arm_target_source.time, "monotonic", return_value=now + 1.0):
            q, kp, kd, tau = src.get_arm_command()
        np.testing.assert_allclose((q[0], kp[0], kd[0], tau[0]), (0.4, 70.0, 1.5, 0.0))
        with patch.object(arm_target_source.time, "monotonic", return_value=now + 3.0):
            q, kp, kd, _ = src.get_arm_command()
        np.testing.assert_allclose((q, kp, kd), (src.default_arm, src.default_kp, src.default_kd))
        # Done: the pre-return sample stays ignored, a quiet one leaves the default.
        q, *_ = src.get_arm_command()
        np.testing.assert_allclose(q, src.default_arm)
        self.publish(src, weight=0.0, q=0.5)
        q, *_ = src.get_arm_command()
        np.testing.assert_allclose(q, src.default_arm)
        # The next mission's sender takes the arms again.
        self.publish(src, weight=1.0, q=0.3)
        q, kp, _, tau = src.get_arm_command()
        np.testing.assert_allclose((q[0], kp[0], tau[0]), (0.3, 40.0, 1.0))

    def test_an_engaging_sender_ends_the_ramp_where_it_is(self):
        src = self.make_source(weight=0.0)
        src.get_arm_command()
        src.return_to_default(2.0)
        src.get_arm_command()
        self.assertIsNotNone(src._arm_return)
        self.publish(src, weight=1.0, q=0.6)
        q, *_ = src.get_arm_command()
        np.testing.assert_allclose(q[0], 0.6)
        self.assertIsNone(src._arm_return)


class CompatibilityTest(unittest.TestCase):
    def test_bundled_nature_exports_are_switchable(self):
        # run_dreamwaq_nature.sh switches between the two backpack exports.
        for first_export, second_export in ((NEW3_4_BACKPACK, NEW2_3_BACKPACK), (NEW3_4, NEW2_3)):
            first, second = DreamWaQ29Contract.load(first_export), DreamWaQ29Contract.load(second_export)
            self.assertEqual(incompatible_fields(first, second), [])
        changed = dataclasses.replace(second, kp_sdk=second.kp_sdk * 2.0)
        self.assertEqual(incompatible_fields(first, changed), ["kp_sdk"])

    def test_command_limits_reach_the_stick_fallback(self):
        config = SimpleNamespace(max_cmd=np.array([0.7, 0.6, 0.6]),
                                 min_cmd=np.array([-0.7, -0.6, -0.6]), cmd_deadzone=0.1)
        source = SimpleNamespace(sticks=CommandSource(config))
        set_command_limits(source, [-0.3, -0.2, -0.4], [0.3, 0.2, 0.4])
        full = SimpleNamespace(lx=-1.0, ly=1.0, rx=1.0)
        np.testing.assert_allclose(source.sticks.get_command(full), [0.3, 0.2, -0.4], atol=1e-6)
        np.testing.assert_allclose(source.max_cmd, [0.3, 0.2, 0.4])


class CliTest(unittest.TestCase):
    def test_switch_options_are_validated(self):
        self.assertIsNone(parse_args(["--net", "eth0"]).switch_export)
        self.assertEqual(parse_args(NATURE_ARGS).switch_button, "Y")
        defaults = parse_args(["--net", "eth0", "--switch-export", NEW2_3])
        self.assertEqual((defaults.switch_back_max_leg_dq, defaults.switch_back_max_gyro,
                          defaults.switch_back_still_time, defaults.switch_back_arm_time),
                         (0.5, 0.3, 0.5, 3.0))
        for argv in (["--switch-max-vx", "0.3"],
                     ["--switch-export", NEW2_3, "--dry-run"],
                     ["--switch-export", NEW2_3, "--switch-button", "X"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(["--net", "eth0", *argv])


class FakeIPC:
    def __init__(self, q15):
        self.q15 = np.asarray(q15, dtype=np.float32)
        self.requests = []
        self.stopped = False

    def publish_state(self, q29, dq29, quat, gyro, command, *, sequence, monotonic_ns, **kwargs):
        published_ns = time.monotonic_ns()
        self.requests.append(dict(kwargs, sequence=sequence, command=np.array(command)))
        self._source = (monotonic_ns, published_ns)
        return published_ns

    def wait_for_action(self, sequence, timeout_s):
        return SimpleNamespace(
            source_state_monotonic_ns=self._source[0], source_state_published_ns=self._source[1],
            q_target15=self.q15.copy(), normalized_lower_action=np.zeros(12, np.float32),
            inference_ms=1.0, max_action=0.0, mode="policy", state_to_action_ms=1.0,
            skipped_state_count=0,
        )

    def request_stop(self):
        self.stopped = True


class FakeEnv:
    controller_mapping = {"select": SELECT}
    writer_fault = reader_fault = None

    def __init__(self, keys, *, config, clip_lower_joint_targets, leg_speeds=(), **kwargs):
        self.config = config
        self.clip_lower_joint_targets = clip_lower_joint_targets
        self.keys = iter(keys)
        # One leg joint speed per state read (rad/s); still once they run out.
        self.leg_speeds = iter(leg_speeds)
        self.get_command = None
        self.published_legs = []

    def read_controller_input(self):
        # One read per 50-Hz tick: record what the writer is publishing now.
        if self.get_command is not None:
            self.published_legs.append(self.get_command()[0][:15].copy())
        return SimpleNamespace(keys=next(self.keys, SELECT), lx=0.0, ly=0.0, rx=0.0, ry=0.0)

    def move_to_default_pos(self, **kwargs):
        pass

    def default_pos_state(self, **kwargs):
        pass

    def start_writer(self, get_command, **kwargs):
        self.get_command = get_command

    def get_robot_state_timed(self):
        dq = np.zeros(29)
        dq[3] = next(self.leg_speeds, 0.0)
        return (np.asarray(self.config.default_angles, np.float32).copy(), dq,
                np.array([1.0, 0, 0, 0]), np.zeros(3), time.monotonic_ns())

    def state_reader_diagnostics(self):
        return dict(state_age_ms=0, source_tick=1, mode_machine=5,
                    nonzero_motorstate=(), max_temperature=20, reader_cpu_affinity=())

    def writer_diagnostics(self):
        return dict(achieved_hz=250, last_publish_age_ms=0, overruns=0,
                    get_ms_max=0, send_ms_max=0, cpu_affinity=())

    def close(self):
        return {"damping_sent": True, "writer_stopped": True}


class FakeRemote:
    """Scripted Thor requests: one list per control tick, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.answers = []
        self.statuses = []
        self.closed = False

    def pending(self):
        if not self.script:
            return []
        return [SimpleNamespace(session="demo", seq=index, action=action)
                for index, action in self.script.pop(0)]

    def resolve(self, request, result, detail=""):
        self.answers.append((request.seq, request.action, result))

    def publish_status(self, status):
        self.statuses.append(status)

    def close(self):
        self.closed = True


class HoldableCommands:
    """A bridge sending `requested` until Thor latches the external hold."""

    def __init__(self, requested):
        self.requested = np.array(requested, np.float32)
        self.held = False

    def get_command(self, controller):
        return np.zeros(3, np.float32) if self.held else self.requested.copy()

    def hold_external(self):
        latched, self.held = not self.held, True
        return latched

    def release_external(self):
        released, self.held = self.held, False
        return released

    def external_held(self):
        return self.held

    def external_status(self):
        return {"state": "hold" if self.held else "autonomy_active", "reason": None, "age_s": 0.0}


class SwitchLoopTest(unittest.TestCase):
    def run_main(self, keys, requested=(0.0, 0.0, 0.0), argv=NATURE_ARGS, remote=None,
                 commands=None, leg_speeds=(), arm_sender=False):
        args = parse_args(argv)
        default15 = None
        self.envs = []

        def make_env(**kwargs):
            nonlocal default15
            env = FakeEnv(keys, leg_speeds=leg_speeds, **kwargs)
            default15 = np.asarray(kwargs["config"].default_angles[:15], np.float32)
            self.first.q15 = default15 + 0.01
            self.second.q15 = default15 - 0.03
            self.envs.append(env)
            return env

        self.first, self.second = FakeIPC(np.zeros(15)), FakeIPC(np.zeros(15))
        workers = iter(((self.first, Mock()), (self.second, Mock())))
        self.commands = commands or SimpleNamespace(
            get_command=lambda controller: np.array(requested, np.float32))
        self.arm = SimpleNamespace(
            _reader_cpu_affinity=None, smoothing_tau=0.0,
            get_arm_command=lambda: (np.zeros(14, np.float32), np.ones(14), np.ones(14), np.zeros(14)),
            sender_active=Mock(return_value=arm_sender), return_to_default=Mock(),
        )
        self.reporter = Mock(path="test.log")
        module = ModuleType("robot_control.g1_wrapper")
        module.G1RealWorldEnv = make_env
        with patch.dict(sys.modules, {"robot_control.g1_wrapper": module}), \
             patch.object(run_dreamwaq, "parse_args", return_value=args), \
             patch.object(run_dreamwaq.os, "sched_getaffinity", return_value=set(range(8))), \
             patch.object(run_dreamwaq, "pin_control_thread", return_value=[7]), \
             patch.object(run_dreamwaq, "start_policy_process", side_effect=lambda *a: next(workers)), \
             patch.object(run_dreamwaq, "stop_policy_process") as self.stop_policy, \
             patch.object(run_dreamwaq, "make_command_source", return_value=self.commands), \
             patch("arm_target_source.make_arm_source", return_value=self.arm), \
             patch.object(run_dreamwaq, "RuntimeReporter", return_value=self.reporter), \
             patch.object(run_dreamwaq, "RemoteModeServer", return_value=remote), \
             patch("builtins.print"):
            self.assertEqual(run_dreamwaq.main(), 0, self.reporter.emit.call_args_list)
        return self.envs[0]

    def test_y_hands_the_standing_robot_to_the_second_export_without_damping(self):
        # Two reads build the mode and switch detectors; then one read per tick.
        env = self.run_main((0, 0, 0, 0, Y, 0, 0, 0, 0, SELECT))
        self.assertEqual(len(self.first.requests), 3)
        # Kept loaded for the way back; only shutdown stops it.
        self.assertFalse(self.first.stopped)
        self.assertEqual(len(self.second.requests), 3)
        self.assertEqual({r["locomotion_mode"] for r in self.second.requests}, {0})
        # Ramp of two ticks: hold, halfway, then the second policy's target.
        # Legs only; the waist index belongs to the waist override.
        np.testing.assert_allclose(env.published_legs[-4][:12], self.first.q15[:12], atol=1e-6)
        np.testing.assert_allclose(env.published_legs[-3][:12],
                                   (self.first.q15[:12] + self.second.q15[:12]) / 2, atol=1e-6)
        np.testing.assert_allclose(env.published_legs[-2][:12], self.second.q15[:12], atol=1e-6)
        self.assertFalse(env.clip_lower_joint_targets)
        np.testing.assert_allclose(self.commands.max_cmd, [0.3, 0.2, 0.4], atol=1e-6)
        np.testing.assert_allclose(self.commands.min_cmd, [-0.3, -0.2, -0.4], atol=1e-6)
        self.assertEqual(self.arm.smoothing_tau, 0.04)
        self.arm.return_to_default.assert_not_called()
        self.assertEqual(self.stop_policy.call_count, 2)

    # Ticks below: two reads build the detectors, Y lands on tick 3 and fires
    # on its release at tick 4; the two-tick leg blend is over after tick 6.
    THERE_AND_BACK = (0, 0, 0, 0, Y, 0, 0, 0, Y, 0, 0, 0, SELECT)

    def test_y_again_hands_the_robot_back_to_the_first_export(self):
        env = self.run_main(self.THERE_AND_BACK)
        self.assertEqual(len(self.second.requests), 3)
        self.assertEqual(len(self.first.requests), 5)
        # Each switch is a new epoch, so the returning worker resets CENet.
        self.assertEqual([r["policy_epoch"] for r in self.first.requests], [0, 0, 0, 2, 2])
        self.assertEqual({r["policy_epoch"] for r in self.second.requests}, {1})
        self.assertEqual([r["locomotion_mode"] for r in self.first.requests[3:]], [0, 0])
        self.assertFalse(self.first.stopped or self.second.stopped)
        # Legs blend from the second policy's held target back to the first's.
        np.testing.assert_allclose(env.published_legs[-3][:12], self.second.q15[:12], atol=1e-6)
        np.testing.assert_allclose(env.published_legs[-2][:12],
                                   (self.first.q15[:12] + self.second.q15[:12]) / 2, atol=1e-6)
        np.testing.assert_allclose(env.published_legs[-1][:12], self.first.q15[:12], atol=1e-6)
        self.assertTrue(env.clip_lower_joint_targets)
        np.testing.assert_allclose(self.commands.max_cmd, [0.7, 0.6, 0.6], atol=1e-6)
        np.testing.assert_allclose(self.commands.min_cmd, [-0.7, -0.6, -0.6], atol=1e-6)
        self.assertEqual(self.arm.smoothing_tau, 0.0)
        self.arm.return_to_default.assert_called_once_with(3.0)

    def test_y_back_waits_for_the_legs_to_be_still(self):
        # NEW2-3 steps in place on a zero command; the way back must not take that.
        env = self.run_main(self.THERE_AND_BACK, leg_speeds=(0.0,) * 3 + (2.0,) * 9)
        self.assertEqual(len(self.first.requests), 3)
        self.assertFalse(env.clip_lower_joint_targets)
        self.arm.return_to_default.assert_not_called()
        self.assertTrue(any("[switch] Y ignored: legs not still" in str(call)
                            for call in self.reporter.emit.call_args_list))

    def test_y_back_waits_for_the_mission_client_to_let_go_of_the_arms(self):
        env = self.run_main(self.THERE_AND_BACK, arm_sender=True)
        self.assertEqual(len(self.first.requests), 3)
        self.assertFalse(env.clip_lower_joint_targets)
        self.arm.return_to_default.assert_not_called()
        self.assertTrue(any("rt/arm_sdk still drives the arms" in str(call)
                            for call in self.reporter.emit.call_args_list))

    def test_y_during_the_leg_blend_is_refused(self):
        env = self.run_main((0, 0, 0, 0, Y, 0, Y, 0, 0, SELECT))
        self.assertEqual(len(self.first.requests), 3)
        self.assertFalse(env.clip_lower_joint_targets)
        self.assertTrue(any("still blending" in str(call)
                            for call in self.reporter.emit.call_args_list))

    def test_y_while_walking_is_refused(self):
        env = self.run_main((0, 0, X, 0, Y, 0, SELECT))
        self.assertEqual(len(self.first.requests), 4)
        self.assertFalse(self.first.stopped)
        self.assertEqual(self.second.requests, [])
        self.assertTrue(env.clip_lower_joint_targets)
        self.assertTrue(any("[switch] Y ignored" in str(call)
                            for call in self.reporter.emit.call_args_list))


    def test_y_is_refused_while_standing_gates_a_live_command(self):
        # NEW3 standing zeroes this for the first policy, but NEW2-3 has no
        # gate: a bridge out of HOLD would walk the robot right after the switch.
        env = self.run_main((0, 0, 0, 0, Y, 0, SELECT), requested=(0.2, 0.0, 0.0))
        self.assertEqual({float(np.abs(r["command"]).max()) for r in self.first.requests}, {0.0})
        self.assertFalse(self.first.stopped)
        self.assertEqual(self.second.requests, [])
        self.assertTrue(env.clip_lower_joint_targets)


class RemoteLoopTest(SwitchLoopTest):
    REMOTE_ARGS = [*NATURE_ARGS, "--remote-bind", "127.0.0.1", "--remote-allowed-host", "127.0.0.1"]

    def test_thor_walks_and_stands_the_robot_without_the_remote(self):
        remote = FakeRemote([[(0, "walking")], [(1, "walking")], [], [(2, "standing")]])
        commands = HoldableCommands((0.2, 0.0, 0.0))
        self.run_main((0, 0, 0, 0, 0, 0, SELECT), argv=self.REMOTE_ARGS, remote=remote,
                      commands=commands)
        self.assertEqual([r["locomotion_mode"] for r in self.first.requests], [1, 1, 1, 0])
        np.testing.assert_allclose([r["command"][0] for r in self.first.requests],
                                   [0.2, 0.2, 0.2, 0.0], atol=1e-6)
        self.assertEqual(remote.answers, [(0, "walking", "applied"), (1, "walking", "already"),
                                          (2, "standing", "applied")])
        self.assertEqual([s["mode"] for s in remote.statuses], ["walking"] * 3 + ["standing"])
        self.assertTrue(remote.closed)

    def test_thor_switches_only_after_the_hold_and_while_standing(self):
        remote = FakeRemote([
            [(0, "switch")],                          # no hold yet
            [(1, "walking"), (2, "hold_external")],
            [(3, "switch")],                          # still walking
            [(4, "standing"), (5, "switch")],
            [(6, "switch"), (7, "walking")],          # already switched; no modes left
        ])
        commands = HoldableCommands((0.2, 0.0, 0.0))
        self.run_main((0, 0, 0, 0, 0, 0, 0, 0, SELECT), argv=self.REMOTE_ARGS, remote=remote,
                      commands=commands)
        self.assertEqual(remote.answers, [
            (0, "switch", "refused"), (1, "walking", "applied"), (2, "hold_external", "applied"),
            (3, "switch", "refused"), (4, "standing", "applied"), (5, "switch", "applied"),
            # Within a tick the mode requests go first, then the switch.
            (7, "walking", "refused"), (6, "switch", "already"),
        ])
        self.assertFalse(self.first.stopped)
        # The hold zeroed the live bridge command before the walk took it.
        self.assertEqual({float(np.abs(r["command"]).max()) for r in self.first.requests}, {0.0})
        self.assertGreater(len(self.second.requests), 0)
        last = remote.statuses[-1]
        self.assertEqual((last["switched"], last["mode"], last["external_hold"], last["export"]),
                         (True, None, True, "new2_3_main_plus50k_249999"))

    def test_y_holds_the_bridge_and_only_thor_releases_it_in_standing(self):
        """The second Nature run: Y there and back, sticks to the start, then the demo."""
        remote = FakeRemote([
            [], [], [], [],                           # Y switches there on tick 4
            [(0, "release_external")],                # NEW2-3 has no standing mode
            [(1, "walking")],
            [], [],                                   # Y switches back on tick 8
            [(2, "walking"), (3, "release_external")],
            [(4, "standing"), (5, "release_external")],
            [(6, "release_external")],
        ])
        commands = HoldableCommands((0.0, 0.0, 0.0))
        self.run_main((0, 0, 0, 0, Y, 0, 0, 0, Y, 0, 0, 0, 0, SELECT), argv=self.REMOTE_ARGS,
                      remote=remote, commands=commands)
        self.assertEqual(remote.answers, [
            (0, "release_external", "refused"), (1, "walking", "refused"),
            (2, "walking", "applied"), (3, "release_external", "refused"),
            (4, "standing", "applied"), (5, "release_external", "applied"),
            (6, "release_external", "already"),
        ])
        # Held from the switch there (Y, not Thor) through the way back.
        holds = [(s["switched"], s["external_hold"]) for s in remote.statuses]
        self.assertEqual(holds, [(False, False)] * 3 + [(True, True)] * 3
                         + [(False, True), (False, False), (False, False)])
        last = remote.statuses[-1]
        self.assertEqual((last["export"], last["mode"]), ("new3_4_widerange_quality_390000", "standing"))
        self.arm.return_to_default.assert_called_once_with(3.0)

    def test_remote_options_are_validated(self):
        for argv in (["--remote-allowed-host", "127.0.0.1"],
                     ["--remote-bind", "127.0.0.1"],
                     ["--remote-bind", "127.0.0.1", "--remote-allowed-host", "127.0.0.1",
                      "--dry-run"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(["--net", "eth0", *argv])


if __name__ == "__main__":
    unittest.main()
