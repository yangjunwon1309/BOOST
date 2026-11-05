"""Autonomy gates and damping hooks, without any DDS/motor construction."""

import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_source import (CommandSource, RosUdpNavCommandSource,
                            check_command_damping, make_command_source, set_command_limits)
from ros_autonomy_bridge import AutonomyBridgeState, parse_args
from robot_control.config import Config
from locomotion_mode import LocomotionModeSwitch


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.bridge = AutonomyBridgeState(clock=lambda: self.now)

    def fresh(self):
        self.bridge.update_odometry()
        self.bridge.update_command(0.2, -0.1, 0.3)

    def test_startup_hold_and_release_need_new_command(self):
        self.fresh()
        packet = self.bridge.packet()
        self.assertEqual(packet['state'], 'hold')
        self.assertEqual(packet['vx'], 0)
        self.bridge.set_hold(False)
        self.assertEqual(self.bridge.packet()['vx'], 0)
        self.fresh()
        self.assertEqual(self.bridge.packet()['vx'], 0.2)

    def test_command_and_odometry_watchdogs_use_wall_clock(self):
        self.bridge.set_hold(False)
        self.fresh()
        self.now += 0.31
        self.assertEqual(self.bridge.packet()['reason'], 'command_stale')
        self.assertEqual(self.bridge.packet()['vx'], 0)
        self.bridge.update_command(0.2, 0, 0)
        self.now += 0.8
        self.bridge.update_command(0.2, 0, 0)
        packet = self.bridge.packet()
        self.assertEqual(packet['state'], 'hold')
        self.assertEqual(packet['reason'], 'odometry_stale')
        self.assertEqual(packet['vx'], 0)

    def test_nonfinite_command_zeros_previous_valid_sample(self):
        self.bridge.set_hold(False)
        self.fresh()
        self.assertFalse(self.bridge.update_command(float('nan'), 0, 0))
        self.assertEqual(self.bridge.packet()['vx'], 0)

    def test_damping_is_latched_and_sequences_increase(self):
        self.bridge.request_damping()
        self.bridge.set_hold(False)
        self.fresh()
        first, second = self.bridge.packet(), self.bridge.packet()
        self.assertEqual(first['state'], 'damping')
        self.assertEqual(first['vx'], 0)
        self.assertEqual(second['seq'], first['seq'] + 1)
        self.assertEqual(first['session'], second['session'])

    def test_cli_rejects_invalid_rates_timeouts_and_ports(self):
        for option, value in (('--rate', 'nan'), ('--cmd-timeout', '0'),
                              ('--odom-timeout', '-1'), ('--port', '0')):
            with self.assertRaises(SystemExit):
                parse_args([option, value])


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        config = SimpleNamespace(max_cmd=np.array([0.95, 0.75, 0.75]),
                                 min_cmd=np.array([-0.8, -0.75, -0.75]), cmd_deadzone=0.1)
        self.source = RosUdpNavCommandSource.__new__(RosUdpNavCommandSource)
        self.source.sticks = CommandSource(config)
        self.source.max_cmd, self.source.min_cmd = config.max_cmd, config.min_cmd
        self.source.timeout = 0.3
        self.source._lock = threading.Lock()
        self.source._latest_cmd = np.zeros(3, dtype=np.float32)
        self.source._latest_time = 0.0
        self.source._latest_state = 'hold'
        self.source._damping = threading.Event()
        self.source._last_session = None
        self.source._last_seq = -1
        self.source._retired_sessions = set()
        self.source._last_bad_warn = self.source._last_gap_warn = 0.0
        self.controller = SimpleNamespace(lx=0, ly=0, rx=0)
        self.seq = 0

    def accept(self, state='autonomy_active', **kwargs):
        self.seq += 1
        packet = dict(vx=2, vy=-2, wz=2, seq=self.seq, session='test', state=state)
        packet.update(kwargs)
        parsed = self.source._parse(json.dumps(packet))
        if parsed is not None:
            with self.source._lock:
                self.source._latest_cmd, self.source._latest_state = parsed
                self.source._latest_time = time.monotonic()
        return parsed

    def test_clipping_sticks_hold_and_timeout(self):
        self.accept()
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.95, -0.75, 0.75])
        self.controller.ly = 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.95, 0, 0])
        self.accept('hold')
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.95, 0, 0])
        self.controller.ly = 0
        np.testing.assert_allclose(self.source.get_command(self.controller), [0, 0, 0])
        self.accept()
        self.source._latest_time -= 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0, 0, 0])
        self.controller.ly = 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.95, 0, 0])

    def test_policy_switch_caps_bridge_and_stick_commands(self):
        # run_dreamwaq_nature.sh retargets this source to NEW2-3's caps on Y.
        set_command_limits(self.source, [-0.3, -0.2, -0.4], [0.3, 0.2, 0.4])
        self.accept()
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.3, -0.2, 0.4])
        self.controller.ly = 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.3, 0, 0])

    def test_manual_control_without_bridge_and_standing_still_stops(self):
        self.controller.ly = 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0.95, 0, 0])
        switch = LocomotionModeSwitch(enabled=True)
        result, mode, _ = switch.update(0, self.source.get_command(self.controller))
        np.testing.assert_allclose(result, [0, 0, 0])
        self.assertEqual(mode, 0)

    def test_standing_gates_fresh_ros_input(self):
        self.accept()
        switch = LocomotionModeSwitch(enabled=True)
        result, mode, _ = switch.update(0, self.source.get_command(self.controller))
        np.testing.assert_allclose(result, [0, 0, 0])
        self.assertEqual(mode, 0)
        result, mode, _ = switch.update(0x0400, self.source.get_command(self.controller))
        np.testing.assert_allclose(result, [0.95, -0.75, 0.75])
        self.assertEqual(mode, 1)

    def test_invalid_replay_and_new_session_cannot_trigger_damping(self):
        self.accept()
        self.assertIsNone(self.accept('damping', seq=1))
        self.assertIsNone(self.accept('damping', vx=float('nan')))
        self.assertIsNone(self.accept('damping', session='new'))
        self.assertFalse(self.source.damping_requested())
        self.assertIsNone(self.accept('human_active'))

    def test_damping_latch_calls_cleanup_and_writer_fault_paths(self):
        self.accept('damping')
        self.accept()
        self.controller.ly = 1
        np.testing.assert_allclose(self.source.get_command(self.controller), [0, 0, 0])
        with self.assertRaises(KeyboardInterrupt):
            check_command_damping(self.source)
        with self.assertRaises(RuntimeError):
            check_command_damping(self.source, writer=True)
        check_command_damping(SimpleNamespace())
        check_command_damping(Mock())

    def test_autonomy_config_factory_requires_loopback(self):
        path = Path(__file__).resolve().parents[1] / 'robot_control/configs/g1_ros_autonomy.yaml'
        config = Config(path)
        self.assertEqual(config.cmd_source, 'ros_udp')
        config.cmd_udp_bind_host = '0.0.0.0'
        with self.assertRaises(ValueError):
            make_command_source(config)


if __name__ == '__main__':
    unittest.main()
