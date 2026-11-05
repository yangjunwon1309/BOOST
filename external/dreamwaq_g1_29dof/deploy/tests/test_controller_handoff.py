"""Controller ownership, RPC failures and motor-writer exclusion without DDS."""

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

from controller_handoff import (
    CONTROLLER_TOGGLE_MASK, ControllerHandoff, MotionServiceError, UnitreeMotionService,
)
from run_dreamwaq import parse_args
import run_dreamwaq


class FakeEnv:
    def __init__(self, events=None, **kwargs):
        self.events = events if events is not None else []
        self.lowcmd_enabled = kwargs.get("lowcmd_enabled", False)
        self.writer_fault = self.reader_fault = None
        self.controller_mapping = {"select": 8}
        self.config = kwargs.get("config")
        self.keys = iter((0, 0, CONTROLLER_TOGGLE_MASK, 0, CONTROLLER_TOGGLE_MASK,
                          CONTROLLER_TOGGLE_MASK, 0, CONTROLLER_TOGGLE_MASK, 8))

    def suspend_lowcmd(self):
        self.events.append("drain")
        self.lowcmd_enabled = False

    def require_fresh_state(self):
        self.events.append("fresh")

    def set_lowcmd_enabled(self, enabled):
        self.events.append("lowcmd_on" if enabled else "lowcmd_off")
        self.lowcmd_enabled = enabled

    def read_controller_input(self):
        return SimpleNamespace(keys=next(self.keys, 8))

    def move_to_default_pos(self, **kwargs):
        self.events.append("initial_ramp")

    def default_pos_state(self, **kwargs):
        self.events.append("initial_A")

    def ramp_to_default_pos(self, **kwargs):
        assert self.lowcmd_enabled
        self.events.append("resume_ramp")

    def start_writer(self, *args, **kwargs):
        assert self.lowcmd_enabled
        self.events.append("writer_start")

    def close(self):
        self.events.append("close_damping" if self.lowcmd_enabled else "close_no_write")
        return {"damping_sent": self.lowcmd_enabled, "writer_stopped": True}

    def get_robot_state_timed(self):
        return (self.config.default_angles.copy(), np.zeros(29), np.array([1, 0, 0, 0]),
                np.zeros(3), time.monotonic_ns())

    def state_reader_diagnostics(self):
        return dict(state_age_ms=0, source_tick=1, mode_machine=5,
                    nonzero_motorstate=(), max_temperature=20, reader_cpu_affinity=())

    def writer_diagnostics(self):
        return dict(achieved_hz=250, last_publish_age_ms=0, overruns=0,
                    get_ms_max=0, send_ms_max=0, cpu_affinity=())


class FakeService:
    def __init__(self, events, env):
        self.events = events
        self.env = env
        self.cancel = lambda: None

    def release(self):
        self.events.append("release")

    def select(self, name):
        assert not self.env.lowcmd_enabled
        self.events.append("select_" + name)

    def start_walking(self):
        assert not self.env.lowcmd_enabled
        self.events.append("walking")


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.env = FakeEnv(self.events)
        self.service = FakeService(self.events, self.env)
        self.logs = []
        self.handoff = ControllerHandoff(self.env, self.service, notify=self.logs.append)

    def test_round_trip_and_cleanup_release_before_any_lowcmd(self):
        self.handoff.enter_dreamwaq()
        self.handoff.enter_unitree()
        self.assertFalse(self.env.lowcmd_enabled)
        self.assertTrue(self.handoff.walking_confirmed)
        self.handoff.enter_dreamwaq()
        self.assertEqual(self.events, ["release", "fresh", "lowcmd_on", "drain",
                                      "release", "select_ai", "walking",
                                      "release", "fresh", "lowcmd_on"])
        self.handoff.enter_unitree()
        self.events.clear()
        self.handoff.prepare_shutdown()
        self.assertEqual(self.events, ["drain", "release", "fresh", "lowcmd_on"])

    def test_chord_edges_and_extra_buttons(self):
        for keys, expected in ((0, False), (CONTROLLER_TOGGLE_MASK, True),
                               (CONTROLLER_TOGGLE_MASK, False), (0, False),
                               (CONTROLLER_TOGGLE_MASK | 8, False),
                               (CONTROLLER_TOGGLE_MASK, False), (0, False),
                               (CONTROLLER_TOGGLE_MASK, True)):
            self.assertEqual(self.handoff.toggle_requested(keys), expected)
        held = ControllerHandoff(self.env, self.service, initial_keys=CONTROLLER_TOGGLE_MASK)
        self.assertFalse(held.toggle_requested(CONTROLLER_TOGGLE_MASK))

    def test_missing_sport_rpc_leaves_native_owned_and_reports_manual_walking(self):
        self.service.start_walking = Mock(side_effect=MotionServiceError("SDK code=3102"))
        self.handoff.enter_unitree()
        self.assertEqual(self.handoff.controller, "unitree")
        self.assertFalse(self.handoff.walking_confirmed)
        self.assertFalse(self.env.lowcmd_enabled)
        self.assertIn("R1+X", self.logs[-1])
        self.handoff.enter_dreamwaq()
        self.assertTrue(self.env.lowcmd_enabled)

    def test_failed_release_never_enables_lowcmd(self):
        self.service.release = Mock(side_effect=MotionServiceError("release failed"))
        with self.assertRaises(MotionServiceError):
            self.handoff.enter_dreamwaq()
        self.assertFalse(self.env.lowcmd_enabled)

    def test_failed_writer_drain_never_starts_native_controller(self):
        self.env.suspend_lowcmd = Mock(side_effect=RuntimeError("writer stuck"))
        with self.assertRaises(RuntimeError):
            self.handoff.enter_unitree()
        self.assertEqual(self.events, [])

    def test_cli_is_opt_in_and_rejects_non_motion_runs(self):
        self.assertFalse(parse_args(["--net", "eth0"]).controller_switch)
        args = parse_args(["--net", "eth0", "--controller-switch", "--initial-controller", "unitree"])
        self.assertEqual(args.initial_controller, "unitree")
        for extra in (("--initial-controller", "unitree"),
                      ("--controller-switch", "--shadow"),
                      ("--controller-switch", "--dry-run")):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(["--net", "eth0", *extra])

    def test_main_keeps_process_and_pauses_ipc_in_native_mode(self):
        self.exercise_main("dreamwaq")

    def test_main_can_start_native_and_enter_dreamwaq_without_start_or_A(self):
        self.exercise_main("unitree")

    def exercise_main(self, initial_controller):
        args = parse_args(["--net", "eth0", "--controller-switch",
                           "--initial-controller", initial_controller])
        requests = []
        ipc = SimpleNamespace(
            publish_state=lambda *a, **kw: requests.append(kw),
            wait_for_action=lambda *a: None,
        )
        # publish_state must return a non-None timestamp to exercise the worker
        # wait while still counting calls.
        def publish(*a, **kw):
            requests.append(kw)
            return time.monotonic_ns()
        ipc.publish_state = publish
        reporter = Mock(path="test.log")
        env = None
        def make_env(**kwargs):
            nonlocal env
            env = FakeEnv(self.events, **kwargs)
            if initial_controller == "unitree":
                env.keys = iter((0, 0, CONTROLLER_TOGGLE_MASK, CONTROLLER_TOGGLE_MASK,
                                 0, CONTROLLER_TOGGLE_MASK, 8))
            return env
        def connect(**kwargs):
            return FakeService(self.events, env)
        module = ModuleType("robot_control.g1_wrapper")
        module.G1RealWorldEnv = make_env
        with patch.dict(sys.modules, {"robot_control.g1_wrapper": module}), \
             patch.object(run_dreamwaq, "parse_args", return_value=args), \
             patch.object(run_dreamwaq.os, "sched_getaffinity", return_value=set(range(8))), \
             patch.object(run_dreamwaq, "pin_control_thread", return_value=[7]), \
             patch.object(run_dreamwaq, "start_policy_process", return_value=(ipc, Mock())), \
             patch.object(run_dreamwaq, "stop_policy_process") as stop_policy, \
             patch.object(run_dreamwaq, "make_command_source", return_value=Mock(get_command=lambda c: np.zeros(3))), \
             patch("arm_target_source.make_arm_source", return_value=SimpleNamespace(_reader_cpu_affinity=None)), \
             patch.object(run_dreamwaq, "RuntimeReporter", return_value=reporter), \
             patch.object(UnitreeMotionService, "connect", side_effect=connect):
            self.assertEqual(run_dreamwaq.main(), 0, reporter.emit.call_args_list)
        self.assertEqual([r["policy_epoch"] for r in requests], [1])
        self.assertEqual([r["locomotion_mode"] for r in requests], [0])
        self.assertEqual(self.events.count("writer_start"), 2 if initial_controller == "dreamwaq" else 1)
        self.assertEqual(self.events.count("initial_A"), int(initial_controller == "dreamwaq"))
        self.assertEqual(self.events.count("resume_ramp"), 1)
        self.assertEqual(self.events.count("walking"), 2)
        self.assertEqual(self.events[-4:], ["release", "fresh", "lowcmd_on", "close_damping"])
        stop_policy.assert_called_once()


class ServiceTest(unittest.TestCase):
    def test_rpc_codes_are_checked_and_invalid_checkmode_is_rejected(self):
        client = Mock(CheckMode=Mock(return_value=(3102, None)))
        service = UnitreeMotionService(client, Mock())
        with self.assertRaisesRegex(MotionServiceError, "3102"):
            service.check_mode()
        client.CheckMode.return_value = (3204, None)
        with self.assertRaisesRegex(MotionServiceError, "3204"):
            service.release()
        client.ReleaseMode.assert_not_called()
        client.CheckMode.return_value = (0, {"name": None})
        with self.assertRaises(MotionServiceError):
            service.check_mode()

    def test_native_release_verification_and_direct_walking(self):
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": "ai"}), (0, {"name": ""})]),
                        ReleaseMode=Mock(return_value=(0, None)))
        loco = Mock(GetFsmId=Mock(side_effect=[(0, 4), (0, 500)]),
                    SetFsmId=Mock(return_value=0))
        service = UnitreeMotionService(switcher, loco)
        service.release()
        service.start_walking()
        loco.SetFsmId.assert_called_once_with(500)

    def test_select_timeout_and_restart_read_errors_still_complete_handoff(self):
        # Real failure trace: SelectMode responds 3104 and CheckMode is briefly
        # unavailable while the native service restarts. Never resend SelectMode.
        env = FakeEnv()
        switcher = Mock(
            CheckMode=Mock(side_effect=[(0, {"name": ""}), (0, {"name": ""}),
                                       (3104, None), (3102, None),
                                       (0, {"name": ""}), (0, {"name": "ai"})]),
            SelectMode=Mock(return_value=(3104, None)),
        )
        logs = []
        service = UnitreeMotionService(switcher, Mock(), notify=logs.append)
        service.start_walking = Mock(side_effect=MotionServiceError("sport unavailable"))
        handoff = ControllerHandoff(env, service, notify=logs.append)
        with patch("controller_handoff.time.sleep"):
            handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertFalse(env.lowcmd_enabled)
        switcher.SelectMode.assert_called_once_with("ai")
        self.assertEqual(switcher.CheckMode.call_count, 6)
        self.assertTrue(any("state confirmation" in line for line in logs))

    def test_release_timeout_requires_empty_mode_before_enabling_lowcmd(self):
        events = []
        env = FakeEnv(events)
        replies = iter([(3102, None), (0, {"name": "ai"}), (3104, None),
                        (0, {"name": "ai"}), (0, {"name": ""})])
        def check():
            self.assertFalse(env.lowcmd_enabled)
            reply = next(replies)
            if reply == (0, {"name": ""}):
                events.append("release_confirmed")
            return reply
        switcher = Mock(CheckMode=Mock(side_effect=check),
                        ReleaseMode=Mock(return_value=(3104, None)))
        service = UnitreeMotionService(switcher, Mock(), notify=Mock())
        handoff = ControllerHandoff(env, service, notify=Mock())
        with patch("controller_handoff.time.sleep"):
            handoff.enter_dreamwaq()
        self.assertEqual(events, ["release_confirmed", "fresh", "lowcmd_on"])
        switcher.ReleaseMode.assert_called_once()
        self.assertEqual(handoff.controller, "dreamwaq")

    def test_unconfirmed_release_timeout_never_enables_lowcmd(self):
        env = FakeEnv()
        switcher = Mock(CheckMode=Mock(return_value=(0, {"name": "ai"})),
                        ReleaseMode=Mock(return_value=(3104, None)))
        service = UnitreeMotionService(switcher, Mock(), timeout_s=0.001, notify=Mock())
        handoff = ControllerHandoff(env, service, notify=Mock())
        with patch("controller_handoff.time.monotonic", side_effect=[0, 0, 0, 0.002]), \
             patch("controller_handoff.time.sleep"):
            with self.assertRaisesRegex(MotionServiceError, "timed out.*release"):
                handoff.enter_dreamwaq()
        self.assertFalse(env.lowcmd_enabled)
        self.assertNotEqual(handoff.controller, "dreamwaq")
        switcher.ReleaseMode.assert_called_once()

    def test_unconfirmed_select_timeout_does_not_claim_native_control(self):
        env = FakeEnv()
        switcher = Mock(
            CheckMode=Mock(side_effect=[(0, {"name": ""}), (0, {"name": ""}),
                                       (3104, None), (3104, None)]),
            SelectMode=Mock(return_value=(3104, None)),
        )
        service = UnitreeMotionService(switcher, Mock(), timeout_s=0.001, notify=Mock())
        service.start_walking = Mock()
        handoff = ControllerHandoff(env, service, notify=Mock())
        with patch("controller_handoff.time.monotonic", side_effect=[0, 0, 0, 0, 0, 0.002]), \
             patch("controller_handoff.time.sleep"):
            with self.assertRaisesRegex(MotionServiceError, "timed out.*startup.*3104"):
                handoff.enter_unitree()
        self.assertFalse(env.lowcmd_enabled)
        self.assertEqual(handoff.controller, "switching")
        switcher.SelectMode.assert_called_once_with("ai")
        service.start_walking.assert_not_called()

    def test_cleanup_retries_restart_read_errors_before_releasing_and_damping(self):
        env = FakeEnv()
        switcher = Mock(
            CheckMode=Mock(side_effect=[(3104, None), (0, {"name": "ai"}),
                                       (3102, None), (0, {"name": ""})]),
            ReleaseMode=Mock(return_value=(3104, None)),
        )
        service = UnitreeMotionService(switcher, Mock(), notify=Mock())
        handoff = ControllerHandoff(env, service)
        handoff.controller = "unitree"
        cancel = service.cancel
        with patch("controller_handoff.time.sleep"):
            handoff.prepare_shutdown()
        self.assertIs(service.cancel, cancel)
        self.assertEqual(env.events, ["drain", "fresh", "lowcmd_on"])
        self.assertEqual(handoff.controller, "dreamwaq")
        switcher.ReleaseMode.assert_called_once()

    def test_permanent_transport_loss_is_bounded_in_startup_and_cleanup(self):
        for cleanup in (False, True):
            with self.subTest(cleanup=cleanup):
                env = FakeEnv()
                switcher = Mock(CheckMode=Mock(return_value=(3104, None)))
                service = UnitreeMotionService(switcher, Mock(), timeout_s=0.001)
                handoff = ControllerHandoff(env, service)
                call = handoff.prepare_shutdown if cleanup else handoff.enter_dreamwaq
                with patch("controller_handoff.time.monotonic", side_effect=[0, 0, 0.002]), \
                     patch("controller_handoff.time.sleep"):
                    with self.assertRaisesRegex(MotionServiceError, "timed out.*3104"):
                        call()
                self.assertFalse(env.lowcmd_enabled)
                self.assertEqual(switcher.CheckMode.call_count, 2)
                switcher.ReleaseMode.assert_not_called()

    def test_send_failures_and_server_rejections_are_not_accepted_as_select(self):
        for code in (3102, 3204):
            with self.subTest(code=code):
                switcher = Mock(SelectMode=Mock(return_value=(code, None)))
                service = UnitreeMotionService(switcher, Mock())
                with self.assertRaisesRegex(MotionServiceError, str(code)):
                    service.select("ai")
                switcher.CheckMode.assert_not_called()

    def test_walking_timeouts_are_verified_without_resending_motion_requests(self):
        loco = Mock(GetFsmId=Mock(side_effect=[(3102, None), (0, 4),
                                             (3104, None), (0, 4), (0, 501)]),
                    SetFsmId=Mock(return_value=3104))
        service = UnitreeMotionService(Mock(), loco, notify=Mock())
        with patch("controller_handoff.time.sleep"):
            service.start_walking()
        loco.SetFsmId.assert_called_once_with(500)

    def test_early_walking_acknowledgement_is_renewed_while_standing_prepares(self):
        loco = Mock(GetFsmId=Mock(side_effect=[(0, 4), (0, 4), (0, 500)]),
                    SetFsmId=Mock(return_value=0))
        service = UnitreeMotionService(Mock(), loco, notify=Mock())
        with patch("controller_handoff.time.sleep"):
            service.start_walking()
        self.assertEqual([c.args[0] for c in loco.SetFsmId.call_args_list], [500, 500])

    def test_acknowledged_but_ignored_walking_is_bounded_and_not_reported_confirmed(self):
        env = FakeEnv()
        logs = []
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": ""}),
                                                  (0, {"name": ""}),
                                                  (0, {"name": "ai"})]),
                        SelectMode=Mock(return_value=(0, None)))
        loco = Mock(GetFsmId=Mock(return_value=(0, 4)),
                    SetFsmId=Mock(return_value=0))
        service = UnitreeMotionService(switcher, loco, timeout_s=0.001, notify=logs.append)
        handoff = ControllerHandoff(env, service, notify=logs.append)
        clock = [0.0]
        with patch("controller_handoff.time.monotonic", side_effect=lambda: clock[0]), \
             patch("controller_handoff.time.sleep", side_effect=lambda _: clock.__setitem__(0, 0.002)):
            handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertFalse(env.lowcmd_enabled)
        self.assertFalse(handoff.walking_confirmed)
        self.assertEqual([c.args[0] for c in loco.SetFsmId.call_args_list], [500, 500])
        self.assertIn("timed out waiting for regular walking", logs[-1])

    def test_native_startup_waits_for_loco_state_then_confirms_walking(self):
        env = FakeEnv()
        logs = []
        switcher = Mock(
            CheckMode=Mock(side_effect=[(0, {"name": ""}), (0, {"name": ""}),
                                       (3104, None), (0, {"name": "ai"})]),
            SelectMode=Mock(return_value=(3104, None)),
        )
        loco = Mock(GetFsmId=Mock(side_effect=[(3102, None), (7301, None),
                                             (7301, None), (0, 0),
                                             (7301, None), (0, 1),
                                             (7301, None), (0, 4),
                                             (7301, None), (0, 500)]))
        def set_fsm(fsm):
            self.assertFalse(env.lowcmd_enabled)
            self.assertIn(fsm, (1, 4, 500))
            self.assertEqual(loco.GetFsmId.call_count, {1: 4, 4: 6, 500: 8}[fsm])
            return 0
        loco.SetFsmId.side_effect = set_fsm
        service = UnitreeMotionService(switcher, loco, notify=logs.append)
        handoff = ControllerHandoff(env, service, notify=logs.append)
        with patch("controller_handoff.time.sleep"):
            handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertTrue(handoff.walking_confirmed)
        self.assertFalse(env.lowcmd_enabled)
        self.assertEqual([c.args[0] for c in loco.SetFsmId.call_args_list], [1, 4, 500])
        self.assertIn("regular walking confirmed", logs[-1])
        self.assertTrue(any("initializing (7301)" in line for line in logs))
        switcher.SelectMode.assert_called_once_with("ai")

    def test_already_walking_needs_no_additional_fsm_request(self):
        for fsm in (500, 501):
            with self.subTest(fsm=fsm):
                loco = Mock(GetFsmId=Mock(return_value=(0, fsm)))
                service = UnitreeMotionService(Mock(), loco)
                service.start_walking()
                loco.SetFsmId.assert_not_called()

    def test_direct_walking_rejection_retains_native_control_without_standing_request(self):
        env = FakeEnv()
        logs = []
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": ""}),
                                                  (0, {"name": ""}),
                                                  (0, {"name": "ai"})]),
                        SelectMode=Mock(return_value=(0, None)))
        loco = Mock(GetFsmId=Mock(return_value=(0, 4)),
                    SetFsmId=Mock(return_value=7302))
        service = UnitreeMotionService(switcher, loco, notify=logs.append)
        handoff = ControllerHandoff(env, service, notify=logs.append)
        handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertFalse(env.lowcmd_enabled)
        self.assertFalse(handoff.walking_confirmed)
        loco.SetFsmId.assert_called_once_with(500)
        self.assertIn("7302", logs[-1])

    def test_missing_loco_state_is_bounded_and_keeps_native_without_motion_requests(self):
        env = FakeEnv()
        logs = []
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": ""}),
                                                  (0, {"name": ""}),
                                                  (0, {"name": "ai"})]),
                        SelectMode=Mock(return_value=(0, None)))
        loco = Mock(GetFsmId=Mock(return_value=(7301, None)))
        service = UnitreeMotionService(switcher, loco, timeout_s=0.001, notify=logs.append)
        handoff = ControllerHandoff(env, service, notify=logs.append)
        with patch("controller_handoff.time.monotonic", side_effect=[0, 0, 0, 0, 0, 0, 0.002]), \
             patch("controller_handoff.time.sleep"):
            handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertFalse(handoff.walking_confirmed)
        self.assertFalse(env.lowcmd_enabled)
        loco.SetFsmId.assert_not_called()
        self.assertEqual(loco.GetFsmId.call_count, 2)
        self.assertEqual(sum("initializing (7301)" in line for line in logs), 1)
        self.assertIn("timed out", logs[-1])
        self.assertIn("7301", logs[-1])
        self.assertIn("R1+X", logs[-1])

    def test_loco_retries_do_not_hide_invalid_fsm_or_switcher_server_errors(self):
        loco = Mock(GetFsmId=Mock(return_value=(7302, None)))
        service = UnitreeMotionService(Mock(), loco)
        with self.assertRaisesRegex(MotionServiceError, "7302"):
            service.start_walking()
        self.assertEqual(loco.GetFsmId.call_count, 1)
        loco.SetFsmId.assert_not_called()
        switcher = Mock(CheckMode=Mock(return_value=(7301, None)))
        service = UnitreeMotionService(switcher, Mock())
        with self.assertRaisesRegex(MotionServiceError, "7301"):
            service.release()
        self.assertEqual(switcher.CheckMode.call_count, 1)
        switcher.ReleaseMode.assert_not_called()

    def test_wait_is_bounded_and_cancellable(self):
        service = UnitreeMotionService(Mock(), Mock(), timeout_s=0.001)
        with patch("controller_handoff.time.sleep"):
            with self.assertRaisesRegex(MotionServiceError, "timed out"):
                service._wait(lambda: False, "release")
        service.cancel = Mock(side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            service.release()

    def test_busy_switcher_retries_rejected_requests_but_not_accepted_timeouts(self):
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": "ai"}),
                                                  (0, {"name": ""}),
                                                  (3104, None), (0, {"name": "ai"})]),
                        ReleaseMode=Mock(side_effect=[(7002, None), (0, None)]),
                        SelectMode=Mock(side_effect=[(7002, None), (3104, None)]))
        service = UnitreeMotionService(switcher, Mock(), notify=Mock())
        with patch("controller_handoff.time.sleep"):
            service.release()
            service.select("ai")
        self.assertEqual(switcher.ReleaseMode.call_count, 2)
        self.assertEqual(switcher.SelectMode.call_count, 2)
        self.assertEqual(switcher.CheckMode.call_count, 4)

    def test_permanent_switcher_busy_is_bounded_and_never_enables_lowcmd(self):
        env = FakeEnv()
        switcher = Mock(CheckMode=Mock(return_value=(0, {"name": "ai"})),
                        ReleaseMode=Mock(return_value=(7002, None)))
        service = UnitreeMotionService(switcher, Mock(), timeout_s=0.001, notify=Mock())
        handoff = ControllerHandoff(env, service, notify=Mock())
        clock = [0.0]
        with patch("controller_handoff.time.monotonic", side_effect=lambda: clock[0]), \
             patch("controller_handoff.time.sleep", side_effect=lambda _: clock.__setitem__(0, 0.002)):
            with self.assertRaisesRegex(MotionServiceError, "timed out.*7002"):
                handoff.enter_dreamwaq()
        self.assertFalse(env.lowcmd_enabled)
        self.assertEqual(switcher.ReleaseMode.call_count, 2)

    def test_native_damping_must_be_confirmed_before_standing_or_walking(self):
        env = FakeEnv()
        logs = []
        switcher = Mock(CheckMode=Mock(side_effect=[(0, {"name": ""}),
                                                  (0, {"name": ""}),
                                                  (0, {"name": "ai"})]),
                        SelectMode=Mock(return_value=(0, None)))
        loco = Mock(GetFsmId=Mock(return_value=(0, 0)),
                    SetFsmId=Mock(return_value=3104))
        service = UnitreeMotionService(switcher, loco, timeout_s=0.001, notify=logs.append)
        handoff = ControllerHandoff(env, service, notify=logs.append)
        clock = [0.0]
        with patch("controller_handoff.time.monotonic", side_effect=lambda: clock[0]), \
             patch("controller_handoff.time.sleep", side_effect=lambda _: clock.__setitem__(0, 0.002)):
            handoff.enter_unitree()
        self.assertEqual(handoff.controller, "unitree")
        self.assertFalse(env.lowcmd_enabled)
        self.assertFalse(handoff.walking_confirmed)
        loco.SetFsmId.assert_called_once_with(1)
        self.assertIn("timed out waiting for native damping", logs[-1])


def load_wrapper_without_dds():
    stubs = {}
    exports = {
        "unitree_sdk2py.core.channel": {name: Mock() for name in (
            "ChannelFactoryInitialize", "ChannelPublisher", "ChannelSubscriber")},
        "unitree_sdk2py.idl.default": {"unitree_hg_msg_dds__LowCmd_": Mock()},
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": {"LowCmd_": Mock(), "LowState_": Mock()},
        "unitree_sdk2py.utils.crc": {"CRC": Mock()},
    }
    for name, values in exports.items():
        stubs[name] = ModuleType(name)
        stubs[name].__dict__.update(values)
    spec = importlib.util.spec_from_file_location(
        "handoff_test_wrapper", ROOT / "deploy/robot_control/g1_wrapper.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.G1RealWorldEnv


class WriterExclusionTest(unittest.TestCase):
    def setUp(self):
        self.cls = load_wrapper_without_dds()
        self.env = self.cls.__new__(self.cls)
        self.env._writer_thread = self.env._final_damping_thread = None
        self.env._writer_on_fault = None
        self.env._lowcmd_enabled = True

    def test_disabled_send_and_start_never_touch_publisher(self):
        self.env.set_lowcmd_enabled(False)
        with self.assertRaises(RuntimeError):
            self.env.send_cmd(None, None, None)
        with self.assertRaises(RuntimeError):
            self.env.start_writer(lambda: None)

    def test_inflight_damping_blocks_handoff_and_no_second_write_is_started(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_write(**kwargs):
            entered.set()
            release.wait(2.0)
        self.env.send_damping = Mock(side_effect=blocked_write)
        try:
            self.assertFalse(self.env._send_damping_bounded(timeout_s=0.005))
            self.assertTrue(entered.is_set())
            with self.assertRaisesRegex(RuntimeError, "in flight"):
                self.env.set_lowcmd_enabled(False)
            self.assertFalse(self.env._send_damping_bounded(timeout_s=0.005))
            self.assertEqual(self.env.send_damping.call_count, 1)
        finally:
            release.set()
            self.env._final_damping_thread.join(1.0)

    def test_close_while_native_owned_does_not_publish_damping(self):
        self.env._lowcmd_enabled = False
        self.env.stop_writer = Mock(return_value=True)
        self.env._send_damping_bounded = Mock()
        self.env._stop = threading.Event()
        self.env._reader_thread = Mock(is_alive=Mock(return_value=False))
        self.assertFalse(self.env.close()["damping_sent"])
        self.env._send_damping_bounded.assert_not_called()


if __name__ == "__main__":
    unittest.main()
