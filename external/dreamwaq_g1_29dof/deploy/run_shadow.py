"""Read-only DreamWaQ shadow runner.

This module deliberately creates no ``rt/lowcmd`` publisher and performs no
DDS write. It subscribes to ``rt/lowstate`` and can reproduce the read-only arm
and pre-write CPU load on a publisher-free cadence probe.
"""

from __future__ import annotations

import os
import struct
import threading
import time

import numpy as np
from cpu_affinity import pin_current_thread

from command_source import CommandSource
from locomotion_mode import LocomotionModeSwitch
from dreamwaq_shadow_timing import (
    ShadowWriterCadenceProbe,
    build_shadow_summary,
    metric_summary,
    write_shadow_report,
)


class _Controller:
    __slots__ = ("keys", "lx", "ly", "rx", "ry")

    def __init__(self, keys, lx, ly, rx, ry):
        self.keys = keys
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry


def _decode_wireless_remote(payload) -> _Controller:
    if not payload or len(payload) < 24:
        return _Controller(0, 0.0, 0.0, 0.0, 0.0)
    payload = bytes(payload)
    keys = struct.unpack("<H", payload[2:4])[0]
    lx, rx, ry = struct.unpack("<fff", payload[4:16])
    ly = struct.unpack("<f", payload[20:24])[0]
    return _Controller(keys, lx, ly, rx, ry)


class ReadOnlyG1State:
    """Production-shaped 250-Hz LowState cache with no LowCmd side effects."""

    def __init__(
        self,
        net,
        config,
        connect_timeout=5.0,
        reader_hz=250.0,
        subscriber=None,
        reader_cpu=None,
    ):
        self.config = config
        reader_hz = float(reader_hz)
        if not np.isfinite(reader_hz) or reader_hz <= 0.0:
            raise ValueError("shadow LowState reader rate must be finite and positive")
        self.period = 1.0 / reader_hz
        self.reader_hz = reader_hz
        self.reader_cpu = None if reader_cpu is None else int(reader_cpu)
        self.reader_cpu_affinity = None
        if subscriber is None:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            ChannelFactoryInitialize(0, net)
            subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber = subscriber
        self.subscriber.Init()
        self._latest = None
        self._latest_received_ns = 0
        self._latest_tick = None
        self._latest_sequence = 0
        self._read_count = 0
        self._duplicate_count = 0
        self._replay_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader_fault = None
        self._reader_ready = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="dreamwaq-shadow-lowstate-reader",
            daemon=True,
        )
        self._reader_thread.start()
        if not self._reader_ready.wait(timeout=1.0):
            self._stop.set()
            raise RuntimeError("shadow LowState reader did not initialize CPU affinity")
        if self._reader_fault is not None:
            raise RuntimeError("shadow LowState reader CPU affinity failed: %s" % self._reader_fault)
        deadline = time.monotonic() + float(connect_timeout)
        while self._latest is None:
            if self._reader_fault is not None:
                raise RuntimeError("shadow LowState reader faulted: %s" % self._reader_fault)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "no 'rt/lowstate' sample received within %.1fs (net=%s)"
                    % (connect_timeout, net)
                )
            time.sleep(0.002)

    @staticmethod
    def _field(value):
        return value() if callable(value) else value

    def _reader_loop(self):
        try:
            self.reader_cpu_affinity = pin_current_thread(self.reader_cpu)
        except Exception as exc:
            self._reader_fault = RuntimeError(
                "could not pin shadow LowState reader to CPU %s: %s"
                % (self.reader_cpu, exc)
            )
            self._reader_ready.set()
            return
        self._reader_ready.set()
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                msg = self.subscriber.Read()
                if msg is not None:
                    tick = int(self._field(getattr(msg, "tick")))
                    with self._lock:
                        self._read_count += 1
                        if self._latest_tick is not None:
                            delta = (tick - self._latest_tick) & 0xFFFFFFFF
                            if delta == 0:
                                self._duplicate_count += 1
                                msg = None
                            elif delta >= 0x80000000:
                                self._replay_count += 1
                                self._reader_fault = RuntimeError(
                                    "LowState tick replayed (%d -> %d)"
                                    % (self._latest_tick, tick)
                                )
                                msg = None
                        if msg is not None:
                            self._latest = msg
                            self._latest_tick = tick
                            self._latest_sequence += 1
                            self._latest_received_ns = time.monotonic_ns()
            except Exception as exc:
                self._reader_fault = exc

            next_deadline += self.period
            remaining = next_deadline - time.monotonic()
            if remaining > 0.0:
                self._stop.wait(remaining)
            else:
                next_deadline = time.monotonic()

    def get_robot_state(self):
        with self._lock:
            msg = self._latest
            received_ns = int(self._latest_received_ns)
            tick = -1 if self._latest_tick is None else int(self._latest_tick)
            source_sequence = int(self._latest_sequence)
        q = np.asarray(
            [msg.motor_state[m].q for m in self.config.joint2motor_idx],
            dtype=np.float32,
        )
        dq = np.asarray(
            [msg.motor_state[m].dq for m in self.config.joint2motor_idx],
            dtype=np.float32,
        )
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        return q, dq, quat, gyro, msg, received_ns, tick, source_sequence

    def read_controller_input(self):
        with self._lock:
            latest = self._latest
        return _decode_wireless_remote(latest.wireless_remote)

    def latest_message(self):
        """Return the immutable latest DDS sample for read-only writer emulation."""

        with self._lock:
            return self._latest

    @property
    def reader_fault(self):
        return self._reader_fault

    def diagnostics(self):
        with self._lock:
            return {
                "read_count": int(self._read_count),
                "unique_count": int(self._latest_sequence),
                "duplicate_count": int(self._duplicate_count),
                "replay_count": int(self._replay_count),
                "latest_tick": -1 if self._latest_tick is None else int(self._latest_tick),
                "latest_received_ns": int(self._latest_received_ns),
                "configured_rate_hz": float(self.reader_hz),
                "reader_cpu_affinity": self.reader_cpu_affinity,
            }

    def close(self):
        self._stop.set()
        self._reader_thread.join(timeout=1.0)
        close = getattr(self.subscriber, "Close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def run_shadow(args, config, ipc, policy_process, torque_limit_sdk=None) -> int:
    """Measure the live no-delay contract without a LowCmd publisher or write."""

    state = ReadOnlyG1State(
        args.net,
        config,
        reader_hz=float(args.shadow_reader_hz),
        reader_cpu=args.lowstate_reader_cpu,
    )
    if torque_limit_sdk is None:
        from dreamwaq_interface.g1_29dof.contract import TORQUE_LIMIT_SDK

        torque_limit_sdk = TORQUE_LIMIT_SDK
    torque_limit_sdk = np.asarray(torque_limit_sdk, dtype=np.float32).reshape(29)
    arm_source = None
    shadow_lowcmd_factory = None
    shadow_crc = None
    if bool(args.shadow_production_load):
        from arm_target_source import make_arm_source
        from dreamwaq_interface.g1_29dof.contract import (
            JOINT_LOWER_SDK,
            JOINT_UPPER_SDK,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        arm_source = make_arm_source(
            config, net=args.net, reader_cpu=args.arm_reader_cpu
        )
        shadow_lowcmd_factory = unitree_hg_msg_dds__LowCmd_
        shadow_crc = CRC()
    default_lower = np.asarray(config.default_angles[:15], dtype=np.float32)
    lower_kp = np.asarray(config.kps[:15], dtype=np.float32)
    lower_kd = np.asarray(config.kds[:15], dtype=np.float32)
    arm_gain_scale = float(getattr(args, "gain_scale", 1.0))

    def production_load_callback():
        if arm_source is None:
            return
        arm_q, arm_kp, arm_kd, arm_tau = arm_source.get_arm_command()
        # Reproduce the production writer's Python-side command composition,
        # safety clamp, message construction, and CRC, but never create a
        # publisher or write this message.
        values = (
            np.concatenate((default_lower, np.asarray(arm_q, dtype=np.float32))),
            np.concatenate(
                (lower_kp, np.asarray(arm_kp, dtype=np.float32) * arm_gain_scale)
            ),
            np.concatenate(
                (lower_kd, np.asarray(arm_kd, dtype=np.float32) * arm_gain_scale)
            ),
            np.concatenate(
                (
                    np.zeros(15, dtype=np.float32),
                    np.asarray(arm_tau, dtype=np.float32) * arm_gain_scale,
                )
            ),
        )
        if not all(value.shape == (29,) and np.all(np.isfinite(value)) for value in values):
            raise RuntimeError("shadow production-load callback produced an invalid command")
        q_target, kp, kd, tau_ff = values
        low_state = state.latest_message()
        measured_q = np.asarray(
            [low_state.motor_state[index].q for index in config.joint2motor_idx],
            dtype=np.float32,
        )
        measured_dq = np.asarray(
            [low_state.motor_state[index].dq for index in config.joint2motor_idx],
            dtype=np.float32,
        )
        q_target = np.clip(q_target, JOINT_LOWER_SDK[:29], JOINT_UPPER_SDK[:29])
        torque_limit = torque_limit_sdk
        tau_ff = np.clip(tau_ff, -torque_limit, torque_limit)
        positive_error = (
            torque_limit - tau_ff + kd * measured_dq
        ) / np.maximum(kp, 1.0e-6)
        negative_error = (
            -torque_limit - tau_ff + kd * measured_dq
        ) / np.maximum(kp, 1.0e-6)
        active_kp = kp > 1.0e-6
        q_target[active_kp] = np.clip(
            q_target[active_kp],
            measured_q[active_kp] + negative_error[active_kp],
            measured_q[active_kp] + positive_error[active_kp],
        )
        q_target = np.clip(q_target, JOINT_LOWER_SDK[:29], JOINT_UPPER_SDK[:29])
        message = shadow_lowcmd_factory()
        message.mode_pr = 0
        message.mode_machine = low_state.mode_machine
        for index in range(35):
            motor = message.motor_cmd[index]
            if index < 29:
                motor.mode = 1
                motor.q = float(q_target[index])
                motor.dq = 0.0
                motor.tau = float(tau_ff[index])
                motor.kp = float(max(0.0, kp[index]))
                motor.kd = float(max(0.0, kd[index]))
            else:
                motor.mode = 0
                motor.q = float(low_state.motor_state[index].q)
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 0.0
        message.crc = shadow_crc.Crc(message)

    writer_probe = ShadowWriterCadenceProbe(
        float(args.writer_rate_hz),
        callback=production_load_callback if args.shadow_production_load else None,
        cpu=args.writer_cpu,
    )
    writer_probe.start()
    control_cpu = getattr(args, "control_cpu", None)
    if control_cpu is not None:
        os.sched_setaffinity(0, {int(control_cpu)})
    control_affinity = sorted(os.sched_getaffinity(0))
    commands = CommandSource(config)
    mode_switch = LocomotionModeSwitch(
        enabled=getattr(args, "explicit_locomotion_mode", False),
        initial_mode=getattr(args, "locomotion_mode", "standing"),
        button=getattr(args, "mode_toggle_button", "X"),
        initial_keys=state.read_controller_input().keys,
    )
    dt = float(config.control_dt)
    dt_ns = int(round(dt * 1.0e9))
    timeout_s = float(args.policy_timeout_ms) / 1000.0
    duration = float(args.duration)
    warmup_s = float(args.shadow_warmup_s)
    status_every_ns = int(max(0.1, float(args.status_every)) * 1.0e9)
    reader_baseline = state.diagnostics()
    run_started_ns = time.monotonic_ns()
    measurement_started_ns = run_started_ns + int(warmup_s * 1.0e9)
    end_ns = None if duration <= 0.0 else measurement_started_ns + int(duration * 1.0e9)
    next_deadline_ns = run_started_ns
    next_status_ns = run_started_ns + status_every_ns
    reader_started_ns = run_started_ns
    previous_tick_started_ns = None
    previous_source_sequence = None
    previous_source_tick = None
    previous_action = np.zeros(ipc.action_dim, dtype=np.float32)
    previous_action_sequence = -1
    rows = []
    sequence = 0
    consecutive_timeouts = 0
    max_consecutive_timeouts = 0
    warmup_timeouts = 0
    fatal_error = None
    reader_finished_ns = run_started_ns

    print(
        "[G1-29][shadow] READ-ONLY: no LowCmd publisher/writer; arm_load=%s "
        "reader_hz=%.1f policy_hz=%.1f warmup=%.1fs measure=%s deadline=%.1fms"
        % (
            "on" if arm_source is not None else "off",
            args.shadow_reader_hz,
            1.0 / dt,
            warmup_s,
            "unbounded" if duration <= 0.0 else "%.1fs" % duration,
            args.policy_timeout_ms,
        ),
        flush=True,
    )
    try:
        while end_ns is None or time.monotonic_ns() < end_ns:
            scheduled_ns = next_deadline_ns
            tick_started_ns = time.monotonic_ns()
            measured = tick_started_ns >= measurement_started_ns
            if state.reader_fault is not None:
                raise RuntimeError("shadow LowState reader faulted: %s" % state.reader_fault)
            if not policy_process.is_alive():
                raise RuntimeError("DreamWaQ policy process exited during shadow")

            controller = state.read_controller_input()
            if controller.keys & 0x0008:
                break
            command, mode_value, changed = mode_switch.update(
                controller.keys, commands.get_command(controller)
            )
            if changed:
                print("[G1-29][shadow] locomotion_mode=%s" % mode_switch.name, flush=True)
            (
                q29,
                dq29,
                quat,
                gyro,
                low_state,
                state_received_ns,
                source_tick,
                source_sequence,
            ) = state.get_robot_state()
            if int(low_state.mode_machine) != int(args.expected_mode_machine):
                raise RuntimeError(
                    "LowState mode_machine=%d expected=%d"
                    % (int(low_state.mode_machine), int(args.expected_mode_machine))
                )
            state_sample_repeated = (
                previous_source_sequence is not None
                and source_sequence == previous_source_sequence
            )
            source_tick_delta = (
                None
                if previous_source_tick is None
                else (int(source_tick) - int(previous_source_tick)) & 0xFFFFFFFF
            )
            previous_source_sequence = source_sequence
            previous_source_tick = source_tick
            request_published_ns = ipc.publish_state(
                q29,
                dq29,
                quat,
                gyro,
                command,
                sequence=sequence,
                monotonic_ns=state_received_ns,
                previous_lower_action=previous_action,
                previous_action_sequence=previous_action_sequence,
                locomotion_mode=mode_value,
            )
            response = ipc.wait_for_action(sequence, timeout_s)
            response_received_ns = time.monotonic_ns()

            response_sequence = None
            response_lag = None
            skipped = None
            mode = "timeout"
            max_action = None
            target_delta = None
            metrics = {
                "ipc_wakeup_ms": None,
                "worker_compute_ms": None,
                "backend_inference_ms": None,
                "worker_overhead_ms": None,
                "response_transfer_ms": None,
                "request_roundtrip_ms": None,
                "state_to_response_ms": None,
                "deadline_margin_ms": None,
            }
            if response is None:
                consecutive_timeouts += 1
                max_consecutive_timeouts = max(
                    max_consecutive_timeouts, consecutive_timeouts
                )
                if not measured:
                    warmup_timeouts += 1
                latest = ipc.latest_action()
                if latest is not None:
                    response_sequence = latest.source_state_sequence
                    response_lag = sequence - response_sequence
            else:
                if (
                    response.source_state_sequence != sequence
                    or response.source_state_monotonic_ns != state_received_ns
                    or response.source_state_published_ns != request_published_ns
                ):
                    raise RuntimeError("policy response does not match the current state request")
                consecutive_timeouts = 0
                response_sequence = response.source_state_sequence
                response_lag = sequence - response_sequence
                skipped = response.skipped_state_count
                mode = response.mode
                max_action = response.max_action
                target_delta = float(
                    np.max(
                        np.abs(
                            response.q_target15
                            - np.asarray(config.default_angles[:15], dtype=np.float32)
                        )
                    )
                )
                # Counterfactual only: no command is published in shadow mode.
                # This preserves representative policy history while timing the
                # same approval path used by production.
                previous_action = response.normalized_lower_action.copy()
                previous_action_sequence = response.source_state_sequence
                writer_probe.submit(sequence, state_received_ns, response_received_ns)
                worker_compute_ms = (
                    response.inference_finished_ns - response.inference_started_ns
                ) / 1.0e6
                metrics = {
                    "ipc_wakeup_ms": (
                        response.inference_started_ns
                        - response.source_state_published_ns
                    ) / 1.0e6,
                    "worker_compute_ms": worker_compute_ms,
                    "backend_inference_ms": response.inference_ms,
                    "worker_overhead_ms": worker_compute_ms - response.inference_ms,
                    "response_transfer_ms": (
                        response_received_ns - response.inference_finished_ns
                    ) / 1.0e6,
                    "request_roundtrip_ms": (
                        response_received_ns - response.source_state_published_ns
                    ) / 1.0e6,
                    "state_to_response_ms": (
                        response_received_ns - response.source_state_monotonic_ns
                    ) / 1.0e6,
                    "deadline_margin_ms": float(args.policy_timeout_ms)
                    - (
                        response_received_ns - response.source_state_published_ns
                    )
                    / 1.0e6,
                }

            loop_finished_ns = time.monotonic_ns()
            row = {
                "sequence": sequence,
                "source_tick": source_tick,
                "source_tick_delta": source_tick_delta,
                "source_sequence": source_sequence,
                "state_sample_repeated": int(state_sample_repeated),
                "locomotion_mode": mode_value if mode_switch.enabled else None,
                "command_vx": float(command[0]),
                "command_vy": float(command[1]),
                "command_yaw": float(command[2]),
                "accepted": int(response is not None),
                "response_sequence": response_sequence,
                "response_sequence_lag": response_lag,
                "skipped_state_count": skipped,
                "tick_started_ns": tick_started_ns,
                "state_received_ns": state_received_ns,
                "request_published_ns": request_published_ns,
                "inference_started_ns": None
                if response is None
                else response.inference_started_ns,
                "inference_finished_ns": None
                if response is None
                else response.inference_finished_ns,
                "response_received_ns": response_received_ns,
                "state_age_ms": (request_published_ns - state_received_ns) / 1.0e6,
                "parent_prepare_ms": (request_published_ns - tick_started_ns) / 1.0e6,
                "loop_work_ms": (loop_finished_ns - tick_started_ns) / 1.0e6,
                "tick_period_ms": None
                if previous_tick_started_ns is None
                else (tick_started_ns - previous_tick_started_ns) / 1.0e6,
                "tick_lateness_ms": max(0.0, (tick_started_ns - scheduled_ns) / 1.0e6),
                "mode": mode,
                "max_action": max_action,
                "target_delta": target_delta,
                **metrics,
            }
            previous_tick_started_ns = tick_started_ns
            if measured:
                rows.append(row)

            now_ns = time.monotonic_ns()
            if now_ns >= next_status_ns:
                scored = rows[-250:] if rows else []
                roundtrip = metric_summary(scored, "request_roundtrip_ms")
                state_age = metric_summary(scored, "state_age_ms")
                timeouts = sum(not bool(item["accepted"]) for item in scored)
                print(
                    "[G1-29][shadow] phase=%s seq=%d tick=%d state_age_p99=%s "
                    "roundtrip_p99=%s timeout=%d skipped=%s max_consecutive=%d"
                    % (
                        "measure" if measured else "warmup",
                        sequence,
                        source_tick,
                        "n/a" if state_age["p99"] is None else "%.3fms" % state_age["p99"],
                        "n/a" if roundtrip["p99"] is None else "%.3fms" % roundtrip["p99"],
                        timeouts,
                        "n/a" if skipped is None else skipped,
                        max_consecutive_timeouts,
                    ),
                    flush=True,
                )
                next_status_ns = now_ns + status_every_ns

            sequence += 1
            next_deadline_ns += dt_ns
            remaining_ns = next_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1.0e9)
            elif remaining_ns <= -dt_ns:
                next_deadline_ns = time.monotonic_ns()
    except KeyboardInterrupt:
        print("[G1-29][shadow] interrupted; writing partial report", flush=True)
    except Exception as exc:
        fatal_error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        reader_finished_ns = time.monotonic_ns()
        reader_diagnostics = state.diagnostics()
        state.close()
        writer_probe.stop()
        close_arm = getattr(arm_source, "close", None)
        if callable(close_arm):
            close_arm()

    probe_results = writer_probe.results()
    for row in rows:
        row.update(probe_results.get(int(row["sequence"]), {}))

    for key in ("read_count", "unique_count", "duplicate_count", "replay_count"):
        reader_diagnostics[key] = max(
            0,
            int(reader_diagnostics.get(key, 0))
            - int(reader_baseline.get(key, 0)),
        )
    reader_elapsed_s = max(1.0e-9, (reader_finished_ns - reader_started_ns) / 1.0e9)
    summary = build_shadow_summary(
        rows,
        policy_timeout_ms=float(args.policy_timeout_ms),
        control_dt_ms=dt * 1000.0,
        max_state_age_ms=float(args.shadow_max_state_age_ms),
        max_tick_jitter_ms=float(args.shadow_max_tick_jitter_ms),
        max_writer_handoff_ms=float(args.shadow_max_writer_handoff_ms),
        expected_lowstate_hz=float(args.shadow_expected_lowstate_hz),
        reader_diagnostics=reader_diagnostics,
        reader_elapsed_s=reader_elapsed_s,
    )
    summary["topology"] = {
        "lowcmd_publisher_constructed": False,
        "lowcmd_writer_constructed": False,
        "arm_source_constructed": bool(arm_source is not None),
        "policy_process_isolated": True,
        "policy_previous_action": "counterfactual_last_accepted_shadow_action",
        "writer_probe": (
            "250Hz scheduling plus arm, 35-slot command construction, and CRC; no LowCmd publisher or DDS write"
            if arm_source is not None
            else "250Hz scheduling only; no LowCmd or DDS write"
        ),
    }
    summary["limitations"] = [
        "read-only shadow does not call LowCmd DDS Write or measure robot-side command acceptance",
        "a constant LowState network offset cannot be inferred from an unsynchronized hardware tick; tick-rate checks detect stalls/backlog changes",
        "previous action is counterfactual because no command is physically applied",
    ]
    summary["warmup_s"] = warmup_s
    summary["warmup_timeout_count"] = warmup_timeouts
    summary["max_consecutive_timeout_count"] = max_consecutive_timeouts
    try:
        parent_affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        parent_affinity = None
    try:
        policy_affinity = sorted(os.sched_getaffinity(policy_process.pid))
    except (AttributeError, OSError, TypeError):
        policy_affinity = None
    summary["runtime"] = {
        "parent_pid": os.getpid(),
        "policy_pid": policy_process.pid,
        "parent_cpu_affinity": parent_affinity,
        "control_cpu_affinity": control_affinity,
        "policy_cpu_affinity": policy_affinity,
        "configured_policy_cpu": args.policy_cpu,
        "dds_io_cpus": list(args.io_cpus),
        "lowstate_reader_cpu_affinity": state.reader_cpu_affinity,
        "arm_reader_cpu_affinity": getattr(arm_source, "_reader_cpu_affinity", None),
        "writer_probe_cpu_affinity": writer_probe.cpu_affinity,
        "policy_prewake_us": float(getattr(args, "policy_prewake_us", 0.0)),
        "device": args.device,
        "reader_hz": float(args.shadow_reader_hz),
        "expected_lowstate_hz": float(args.shadow_expected_lowstate_hz),
        "writer_probe_hz": float(args.writer_rate_hz),
    }
    if fatal_error is not None:
        summary["errors"].append(fatal_error)
        summary["verdict"] = "FAIL_DELAY_DETECTED"
    if writer_probe.fault is not None:
        summary["errors"].append("shadow writer callback faulted: %s" % writer_probe.fault)
        summary["verdict"] = "FAIL_DELAY_DETECTED"
    json_path, csv_path = write_shadow_report(args.shadow_report, summary, rows)

    print(
        "[G1-29][shadow] verdict=%s samples=%d accepted=%d timeouts=%d "
        "skipped=%d reader_unique_hz=%.1f"
        % (
            summary["verdict"],
            summary["sample_count"],
            summary["accepted_count"],
            summary["timeout_count"],
            summary["worker_skipped_state_count"],
            summary["reader_unique_rate_hz"],
        ),
        flush=True,
    )
    p99 = summary["metrics_ms"]

    def p99_text(name):
        value = p99[name]["p99"]
        return "n/a" if value is None else "%.3fms" % value

    hardware_tick_rate = summary["hardware_tick_rate_hz"]
    print(
        "[G1-29][shadow] p99 state_age=%s ipc_wakeup=%s inference=%s "
        "roundtrip=%s writer_ready=%s hardware_tick=%s"
        % (
            p99_text("state_age_ms"),
            p99_text("ipc_wakeup_ms"),
            p99_text("backend_inference_ms"),
            p99_text("request_roundtrip_ms"),
            p99_text("shadow_writer_ready_ms"),
            "n/a" if hardware_tick_rate is None else "%.1fHz" % hardware_tick_rate,
        ),
        flush=True,
    )
    for message in summary["errors"]:
        print("[G1-29][shadow][ERROR] " + message, flush=True)
    for message in summary["warnings"]:
        print("[G1-29][shadow][WARN] " + message, flush=True)
    print(
        "[G1-29][shadow][NOTE] PASS covers the pre-write software path; "
        "LowCmd DDS Write/network/robot acceptance are not measured",
        flush=True,
    )
    print("[G1-29][shadow] summary=%s samples=%s" % (json_path, csv_path), flush=True)
    return 2 if summary["verdict"] == "FAIL_DELAY_DETECTED" else 0
