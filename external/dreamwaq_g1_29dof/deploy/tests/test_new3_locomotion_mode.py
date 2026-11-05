"""NEW3 actor/CENet parity and operator mode transitions, without robot I/O."""

import copy
import csv
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import dreamwaq_policy_process
from dreamwaq_policy import DreamWaQPolicy
from dreamwaq_policy_ipc import DreamWaQPolicyIPC
from locomotion_mode import LocomotionModeSwitch
from run_dreamwaq import parse_args
from command_source import CommandSource, configure_command_limits
from robot_control.config import Config
from run_shadow import run_shadow
from dreamwaq_interface.g1_29dof.contract import DreamWaQ29Contract, JointOrderMap
from dreamwaq_interface.g1_29dof.observation import (
    DreamWaQObservationBuilder, FlashSACHistoryBuffer, RobotState29,
)

EXPORT = ROOT / "checkpoints/new3_4_widerange_quality_390000/exported"


class New3ModeTest(unittest.TestCase):
    def test_contract_dimensions_and_malformed_mode_receipts(self):
        contract = DreamWaQ29Contract.load(EXPORT)
        self.assertTrue(contract.explicit_locomotion_mode)
        self.assertEqual((contract.observation_dim, contract.actor_input_dim, contract.history_dim), (100, 132, 465))
        np.testing.assert_allclose(contract.command_lower, (-0.8, -0.8, -1.0))
        np.testing.assert_allclose(contract.command_upper, (1.0, 0.8, 1.0))
        np.testing.assert_allclose(contract.kp_sdk[12:15], 300.0)
        np.testing.assert_allclose(contract.kd_sdk[12:15], 3.0)
        bad_receipts = []
        for key, value in (("actor_index", 98), ("standing_command", [0.1, 0, 0]), ("walking_zero_command_allowed", False)):
            payload = copy.deepcopy(dict(contract.manifest))
            payload["locomotion_mode_contract"][key] = value
            bad_receipts.append(payload)
        payload = copy.deepcopy(dict(contract.manifest))
        payload["policy_obs_dim"] = 99
        bad_receipts.append(payload)
        payload = copy.deepcopy(dict(contract.manifest))
        payload.pop("locomotion_mode_contract")
        bad_receipts.append(payload)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "exported"
            shutil.copytree(EXPORT, target)
            for payload in bad_receipts:
                with self.subTest(payload=payload.get("locomotion_mode_contract")):
                    (target / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        DreamWaQ29Contract.load(target)

    def test_press_hold_release_and_command_gating(self):
        switch = LocomotionModeSwitch(enabled=True)
        source = np.array([0.3, 0.1, -0.2], dtype=np.float32)
        original_source = source.copy()
        command, mode, changed = switch.update(0, source)
        np.testing.assert_array_equal(command, 0.0)
        self.assertEqual((mode, changed), (0, False))
        # Other held buttons must not hide X's rising edge.
        command, mode, changed = switch.update(0x0401, source)
        np.testing.assert_array_equal(command, source)
        self.assertEqual((mode, changed), (1, True))
        for _ in range(100):
            _, mode, changed = switch.update(0x0401, source)
            self.assertEqual((mode, changed), (1, False))
        command, mode, _ = switch.update(0, np.zeros(3))
        self.assertEqual(mode, 1)  # zero command remains walking
        command, mode, changed = switch.update(0x0400, source)
        np.testing.assert_array_equal(command, 0.0)
        self.assertEqual((mode, changed), (0, True))
        np.testing.assert_array_equal(source, original_source)

    def test_held_bringup_button_and_legacy_policy(self):
        switch = LocomotionModeSwitch(enabled=True, initial_keys=0x0400)
        self.assertEqual(switch.update(0x0400, (0.3, 0, 0))[1:], (0, False))
        switch.update(0, (0, 0, 0))
        self.assertEqual(switch.update(0x0400, (0.3, 0, 0))[1:], (1, True))
        legacy = LocomotionModeSwitch(enabled=False)
        command, _, changed = legacy.update(0x0400, (0.3, 0, 0))
        np.testing.assert_allclose(command, (0.3, 0, 0))
        self.assertFalse(changed)

    def test_default_export_and_initial_mode_cli(self):
        args = parse_args(["--net", "eth0"])
        self.assertEqual(Path(args.export).resolve(), EXPORT.resolve())
        self.assertEqual((args.locomotion_mode, args.mode_toggle_button), ("standing", "X"))
        args = parse_args(["--net", "eth0", "--locomotion-mode", "walking", "--mode-toggle-button", "Y"])
        self.assertEqual((args.locomotion_mode, args.mode_toggle_button), ("walking", "Y"))

    def test_requested_joystick_speeds_reach_the_policy_input(self):
        args = parse_args(["--net", "eth0", "--max-vx", "0.95",
                           "--max-vy", "0.75", "--max-yaw", "0.75"])
        policy = DreamWaQPolicy(EXPORT)
        try:
            config = Config(ROOT / "deploy/robot_control/configs/g1_homie.yaml")
            configure_command_limits(config, policy.contract.command_lower,
                                     policy.contract.command_upper,
                                     (args.max_vx, args.max_vy, args.max_yaw))
            sticks = CommandSource(config)
            for ly, other, expected in ((1, -1, (0.95, 0.75, 0.75)),
                                       (-1, 1, (-0.8, -0.75, -0.75))):
                command = sticks.get_command(SimpleNamespace(ly=ly, lx=other, rx=other))
                policy.set_command(command)
                for mode in (1, 0):
                    policy.set_locomotion_mode(mode)
                    with patch.object(policy.backend, "infer_action", wraps=policy.backend.infer_action) as infer:
                        target, label, _ = policy.update(
                            policy.contract.default_q_sdk, np.zeros(29), (1, 0, 0, 0), (0, 0, 0),
                            previous_lower_action=np.zeros(policy.contract.action_dim),
                        )
                    observation = infer.call_args.args[0]
                    np.testing.assert_allclose(observation[96:99], expected if mode else (0, 0, 0))
                    self.assertEqual(label, "walk" if mode else "stand")
                    self.assertTrue(np.isfinite(target).all())
            # These runtime caps do not rewrite the selected model's contract.
            np.testing.assert_allclose(policy.contract.command_lower, (-0.8, -0.8, -1.0))
            np.testing.assert_allclose(policy.contract.command_upper, (1.0, 0.8, 1.0))
        finally:
            policy.close()

    def test_observation_and_fused_action_match_sim2sim_assembly(self):
        import torch

        policy = DreamWaQPolicy(EXPORT)
        try:
            contract = policy.contract
            mapper = JointOrderMap()
            history = FlashSACHistoryBuffer()
            builder = DreamWaQObservationBuilder(contract=contract)
            previous = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
            # A varied measured trace exercises joint ordering, previous actor
            # output and CENet term-major history across both mode transitions.
            for frame, (mode, command) in enumerate(((0, (0.3, 0, 0)), (1, (0, 0, 0)), (1, (0.3, 0, 0)), (0, (0.2, 0.1, 0.2)))):
                state = RobotState29.make(
                    q_sdk=contract.default_q_sdk + np.linspace(-0.02, 0.02, 29, dtype=np.float32) * frame,
                    dq_sdk=np.linspace(-0.1, 0.1, 29, dtype=np.float32) * frame,
                    quaternion_wxyz=(1, 0, 0, 0), angular_velocity_body=(0.01, -0.02, 0.03),
                    monotonic_ns=time.monotonic_ns(), mode_machine=5,
                )
                observation = builder.build(state, command=command, previous_lower_action=previous, locomotion_mode=mode)
                previous29 = np.zeros(29, np.float32)
                previous29[mapper.policy_indices(contract.action_joint_names)] = previous
                # Same observable field order as the training/sim2sim NEW3
                # frame. The mode belongs only in actor current, not CENet.
                expected_measurable = np.concatenate((
                    state.angular_velocity_body, (0, 0, -1),
                    mapper.sdk_to_policy(state.q_sdk - contract.default_q_sdk),
                    mapper.sdk_to_policy(state.dq_sdk) * 0.05, previous29,
                )).astype(np.float32)
                effective_command = np.array(command if mode else (0, 0, 0), np.float32)
                expected_current = np.concatenate((np.zeros(3, np.float32), expected_measurable, effective_command, [mode])).astype(np.float32)
                np.testing.assert_allclose(observation.measurable_current, expected_measurable, atol=1e-7)
                np.testing.assert_allclose(observation.policy_current, expected_current, atol=1e-7)
                flat = history.append(expected_measurable)
                direct = policy.backend.infer(observation.policy_current, flat)
                fused = policy.backend.infer_action(observation.policy_current, flat)
                with torch.inference_mode():
                    velocity, latent = policy.backend.cenet(torch.from_numpy(flat).unsqueeze(0))
                    expected_current[:3] = velocity.numpy().reshape(3)
                    expected_actor = np.concatenate((expected_current, latent.numpy().reshape(-1)))
                    expected_action = torch.tanh(policy.backend.policy(torch.from_numpy(expected_actor).unsqueeze(0))).numpy().reshape(-1)
                np.testing.assert_allclose(direct.actor_input, expected_actor, atol=1e-6)
                # Frozen fusion changes FP32 operation ordering slightly.
                np.testing.assert_allclose(fused.lower_action, expected_action, atol=3e-6)
                self.assertEqual(direct.actor_input[99], mode)
                policy.set_command(command)
                policy.set_locomotion_mode(mode)
                target, label, _ = policy.update(state.q_sdk, state.dq_sdk, state.quaternion_wxyz, state.angular_velocity_body, previous_lower_action=previous)
                self.assertEqual(label, "walk" if mode else "stand")
                self.assertTrue(np.isfinite(target).all())
                np.testing.assert_array_equal(policy.previous_lower_action, previous)
                np.testing.assert_allclose(policy.last_lower_action, expected_action, atol=3e-6)
        finally:
            policy.close()

    def test_mode_is_atomic_with_sequence_in_spawned_worker(self):
        context = mp.get_context("spawn")
        ipc = DreamWaQPolicyIPC(context)
        process = context.Process(target=dreamwaq_policy_process.run, args=(ipc, str(EXPORT), "cpu", 0.02))
        process.start()
        try:
            self.assertTrue(ipc.ready_event.wait(30.0))
            contract = DreamWaQ29Contract.load(EXPORT)
            previous = np.zeros(12, dtype=np.float32)
            for sequence, mode in enumerate((0, 1, 0)):
                stamp = time.monotonic_ns()
                published = ipc.publish_state(
                    contract.default_q_sdk, np.zeros(29), (1, 0, 0, 0), (0, 0, 0), (0.3, 0, 0),
                    sequence=sequence, monotonic_ns=stamp,
                    previous_lower_action=previous, previous_action_sequence=sequence - 1,
                    locomotion_mode=mode,
                )
                request = ipc.wait_for_state(sequence - 1, 1.0)
                self.assertEqual(request.locomotion_mode, mode)
                response = ipc.wait_for_action(sequence, 5.0)
                self.assertIsNotNone(response)
                self.assertEqual(response.source_state_sequence, sequence)
                self.assertEqual(response.source_state_monotonic_ns, stamp)
                self.assertEqual(response.source_state_published_ns, published)
                self.assertEqual(response.mode, "walk" if mode else "stand")
                self.assertTrue(np.isfinite(response.q_target15).all())
                previous = response.normalized_lower_action
        finally:
            ipc.request_stop()
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)

    def test_new_controller_epoch_resets_cenet_in_the_same_worker(self):
        context = mp.get_context("spawn")
        ipc = DreamWaQPolicyIPC(context)
        process = context.Process(target=dreamwaq_policy_process.run, args=(ipc, str(EXPORT), "cpu", 0.02))
        reference = DreamWaQPolicy(EXPORT)
        process.start()
        try:
            self.assertTrue(ipc.ready_event.wait(30.0))
            contract = reference.contract
            zero_action = np.zeros(contract.action_dim, np.float32)
            for sequence, epoch in enumerate((0, 0, 1)):
                q = contract.default_q_sdk.copy()
                dq = np.zeros(29, np.float32)
                if epoch == 0:
                    q += 0.05
                    dq.fill(0.2)
                published = ipc.publish_state(
                    q, dq, (1, 0, 0, 0), (0, 0, 0), (0, 0, 0),
                    sequence=sequence, monotonic_ns=time.monotonic_ns(),
                    previous_lower_action=zero_action, previous_action_sequence=-1,
                    locomotion_mode=0, policy_epoch=epoch,
                )
                response = ipc.wait_for_action(sequence, 5.0)
                self.assertIsNotNone(response)
                self.assertEqual(response.source_state_published_ns, published)
                request = ipc.wait_for_state(sequence - 1, 1.0)
                self.assertEqual(request.policy_epoch, epoch)
            # After native motion, the resumed response must match a freshly
            # initialized worker, not include the pre-handoff measured history.
            reference.set_command((0, 0, 0))
            reference.set_locomotion_mode(0)
            target, _, _ = reference.update(q, dq, (1, 0, 0, 0), (0, 0, 0), zero_action)
            np.testing.assert_allclose(response.q_target15, target, atol=3e-6)
            np.testing.assert_allclose(response.normalized_lower_action, reference.last_lower_action, atol=3e-6)
            self.assertTrue(process.is_alive())
        finally:
            reference.close()
            ipc.request_stop()
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)

    def test_shadow_records_button_modes_and_gated_commands(self):
        contract = DreamWaQ29Contract.load(EXPORT)

        class FakeState:
            def __init__(self, *_args, **_kwargs):
                self.sequence = 0
                self.started = time.monotonic()
                self.reader_fault = None
                self.reader_cpu_affinity = None

            def read_controller_input(self):
                keys = (0, 0x0400, 0x0400, 0, 0x0400)[min(self.sequence, 4)]
                return SimpleNamespace(keys=keys, lx=0.0, ly=0.6, rx=0.0, ry=0.0)

            def get_robot_state(self):
                self.sequence += 1
                tick = self.sequence * 20
                return (contract.default_q_sdk, np.zeros(29), np.array([1, 0, 0, 0]), np.zeros(3),
                        SimpleNamespace(mode_machine=5), time.monotonic_ns() - 1_000_000, tick, self.sequence)

            def diagnostics(self):
                count = int((time.monotonic() - self.started) * 250)
                return dict(read_count=count, unique_count=count, duplicate_count=0, replay_count=0)

            def close(self):
                pass

        context = mp.get_context("spawn")
        ipc = DreamWaQPolicyIPC(context)
        process = context.Process(target=dreamwaq_policy_process.run, args=(ipc, str(EXPORT), "cpu", 0.02))
        process.start()
        try:
            self.assertTrue(ipc.ready_event.wait(30.0))
            args = parse_args(["--net", "offline", "--shadow", "--no-shadow-production-load", "--duration", "0.14", "--shadow-warmup-s", "0"])
            args.explicit_locomotion_mode = True
            args.policy_cpu = args.control_cpu = args.writer_cpu = args.lowstate_reader_cpu = None
            config = Config(args.config)
            with tempfile.TemporaryDirectory() as tmp:
                args.shadow_report = str(Path(tmp) / "mode.json")
                with patch("run_shadow.ReadOnlyG1State", FakeState):
                    run_shadow(args, config, ipc, process, contract.torque_limit_sdk)
                with Path(args.shadow_report).with_suffix(".csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertGreaterEqual(len(rows), 5)
                self.assertEqual([int(row["locomotion_mode"]) for row in rows[:5]], [0, 1, 1, 1, 0])
                for row in rows:
                    if int(row["locomotion_mode"]) == 0:
                        self.assertEqual(float(row["command_vx"]), 0.0)
                    else:
                        self.assertGreater(float(row["command_vx"]), 0.0)
                    if int(row["accepted"]):
                        self.assertEqual(row["mode"], "walk" if int(row["locomotion_mode"]) else "stand")
                summary = json.loads(Path(args.shadow_report).read_text())
                self.assertFalse(summary["topology"]["lowcmd_publisher_constructed"])
        finally:
            ipc.request_stop()
            process.join(5.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)


if __name__ == "__main__":
    unittest.main()
