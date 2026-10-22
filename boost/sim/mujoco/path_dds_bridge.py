"""Bridge PATH's packed LCM packets to the local G1 DDS simulation."""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass

import lcm
import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
from boost.io.path_transport.g1_direct_protocol import (
    ARM_COUNT,
    ARM_START,
    ROBOT_STATE_CHANNEL,
    UPPER_COMMAND_CHANNEL,
    UpperCommand,
    pack_robot_state,
    unpack_upper_command,
)
from boost.io.unitree_crc import make_unitree_crc

LOWSTATE_TOPIC = "rt/lowstate"
TORSO_IMU_TOPIC = "rt/secondary_imu"
ARM_SDK_TOPIC = "rt/arm_sdk"
ARM_MARKER_INDEX = 29
BODY_MOTOR_COUNT = 29
ARM_COMMAND_TIMEOUT_S = 0.150
COMMAND_PERIOD_S = 0.002
STATE_PERIOD_S = 0.010


def _motor_gains() -> tuple[np.ndarray, np.ndarray]:
    # Keep the arm SDK gains aligned with boost/io/g1_bridge/include/robot_parameters.hpp.
    natural_frequency = 10.0 * 2.0 * 3.1415926535
    damping_ratio = 2.0
    armatures = np.asarray(
        [0.003609725] * 5
        + [0.00425] * 2
        + [0.003609725] * 5
        + [0.00425] * 2,
        dtype=np.float32,
    )
    kp = armatures * natural_frequency**2
    kd = 2.0 * damping_ratio * armatures * natural_frequency
    return kp, kd


ARM_KP, ARM_KD = _motor_gains()


@dataclass(frozen=True)
class _RobotSnapshot:
    sequence: int
    mode_machine: int
    base_quat: tuple[float, ...]
    base_gyro: tuple[float, ...]
    base_accel: tuple[float, ...]
    torso_quat: tuple[float, ...] | None
    torso_gyro: tuple[float, ...] | None
    torso_accel: tuple[float, ...] | None
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau_est: tuple[float, ...]


class PathDdsBridge:
    """Simulation-only replacement for the native PATH/G1 DDS bridge."""

    def __init__(self, *, interface: str, domain_id: int, lcm_url: str):
        self.interface = interface
        self.domain_id = domain_id
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state: _RobotSnapshot | None = None
        self._command: tuple[UpperCommand, float] | None = None
        self._lowstate_sequence = 0
        self._state_packet_sequence = 0
        self._crc = make_unitree_crc()

        self._lcm_rx = lcm.LCM(lcm_url)
        self._lcm_tx = lcm.LCM(lcm_url)
        self._lcm_rx.subscribe(UPPER_COMMAND_CHANNEL, self._on_path_command)

        ChannelFactoryInitialize(self.domain_id, self.interface)
        self._arm_publisher = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self._lowstate_subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._torso_imu_subscriber = ChannelSubscriber(TORSO_IMU_TOPIC, IMUState_)

    @staticmethod
    def _array(values, size: int) -> tuple[float, ...]:
        result = np.asarray(values, dtype=np.float32).reshape(-1)
        if result.size != size or not np.isfinite(result).all():
            raise ValueError(f"DDS vector must contain {size} finite values")
        return tuple(float(value) for value in result)

    def _on_lowstate(self, message: LowState_) -> None:
        imu = message.imu_state
        q = tuple(float(message.motor_state[i].q) for i in range(BODY_MOTOR_COUNT))
        dq = tuple(float(message.motor_state[i].dq) for i in range(BODY_MOTOR_COUNT))
        tau = tuple(float(message.motor_state[i].tau_est) for i in range(BODY_MOTOR_COUNT))
        with self._lock:
            self._lowstate_sequence += 1
            previous_torso = self._state
            self._state = _RobotSnapshot(
                sequence=self._lowstate_sequence,
                mode_machine=int(message.mode_machine),
                base_quat=self._array(imu.quaternion, 4),
                base_gyro=self._array(imu.gyroscope, 3),
                base_accel=self._array(imu.accelerometer, 3),
                torso_quat=previous_torso.torso_quat if previous_torso else None,
                torso_gyro=previous_torso.torso_gyro if previous_torso else None,
                torso_accel=previous_torso.torso_accel if previous_torso else None,
                q=q,
                dq=dq,
                tau_est=tau,
            )

    def _on_torso_imu(self, message: IMUState_) -> None:
        with self._lock:
            state = self._state
            if state is None:
                return
            self._state = _RobotSnapshot(
                **{
                    **state.__dict__,
                    "torso_quat": self._array(message.quaternion, 4),
                    "torso_gyro": self._array(message.gyroscope, 3),
                    "torso_accel": self._array(message.accelerometer, 3),
                }
            )

    def _on_path_command(self, _channel: str, payload: bytes) -> None:
        try:
            command = unpack_upper_command(payload)
        except (ValueError, TypeError) as exc:
            print(f"[path-dds] ignoring malformed PATH command: {exc}", flush=True)
            return
        with self._lock:
            self._command = (command, time.monotonic())

    def _snapshot(self):
        with self._lock:
            return self._state, self._command

    def _publish_robot_state(self) -> None:
        state, _ = self._snapshot()
        if state is None:
            return
        packet = pack_robot_state(
            sequence=self._state_packet_sequence,
            lowstate_sequence=state.sequence,
            timestamp_us=time.time_ns() // 1_000,
            mode_machine=state.mode_machine,
            base_quat_wxyz=state.base_quat,
            base_gyro=state.base_gyro,
            base_accel=state.base_accel,
            torso_quat_wxyz=state.torso_quat,
            torso_gyro=state.torso_gyro,
            torso_accel=state.torso_accel,
            q=state.q,
            dq=state.dq,
            tau_est=state.tau_est,
        )
        self._lcm_tx.publish(ROBOT_STATE_CHANNEL, packet)
        self._state_packet_sequence += 1

    def _publish_arm_sdk(self) -> None:
        state, command_sample = self._snapshot()
        if state is None:
            return

        active_command = None
        if command_sample is not None:
            command, received_at = command_sample
            if time.monotonic() - received_at <= ARM_COMMAND_TIMEOUT_S:
                active_command = command

        message = unitree_hg_msg_dds__LowCmd_()
        message.mode_pr = 0
        message.mode_machine = state.mode_machine
        for index in range(BODY_MOTOR_COUNT):
            motor = message.motor_cmd[index]
            motor.mode = 1
            motor.q = state.q[index]
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

        for arm_index in range(ARM_COUNT):
            index = ARM_START + arm_index
            motor = message.motor_cmd[index]
            motor.kp = float(ARM_KP[arm_index])
            motor.kd = float(ARM_KD[arm_index])
            if active_command is not None:
                motor.q = float(active_command.q_arm[arm_index])
                motor.dq = float(active_command.dq_arm[arm_index])
                motor.tau = float(active_command.tau_arm[arm_index])

        marker = message.motor_cmd[ARM_MARKER_INDEX]
        marker.mode = 1
        marker.q = 1.0 if active_command is not None else 0.0
        marker.dq = 0.0
        marker.tau = 0.0
        marker.kp = 300.0
        marker.kd = 3.0
        message.crc = self._crc.Crc(message)
        self._arm_publisher.Write(message)

    def run(self) -> None:
        self._arm_publisher.Init()
        self._lowstate_subscriber.Init(self._on_lowstate, 1)
        self._torso_imu_subscriber.Init(self._on_torso_imu, 1)
        command_thread = threading.Thread(
            target=self._loop,
            args=(COMMAND_PERIOD_S, self._publish_arm_sdk),
            name="path-arm-sdk-writer",
            daemon=True,
        )
        state_thread = threading.Thread(
            target=self._loop,
            args=(STATE_PERIOD_S, self._publish_robot_state),
            name="path-lcm-state-writer",
            daemon=True,
        )
        command_thread.start()
        state_thread.start()
        print(
            f"[path-dds] DDS domain={self.domain_id} interface={self.interface}; "
            f"LCM upper={UPPER_COMMAND_CHANNEL} state={ROBOT_STATE_CHANNEL}",
            flush=True,
        )
        try:
            while not self._stop.is_set():
                self._lcm_rx.handle_timeout(100)
        finally:
            self._stop.set()
            command_thread.join(timeout=1.0)
            state_thread.join(timeout=1.0)

    def _loop(self, period: float, callback) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += period
            callback()
            self._stop.wait(max(0.0, next_tick - time.monotonic()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--lcm-url", default=os.getenv("PATH_LCM_URL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lcm_url:
        raise SystemExit("PATH_LCM_URL or --lcm-url is required")
    PathDdsBridge(
        interface=args.interface,
        domain_id=args.domain_id,
        lcm_url=args.lcm_url,
    ).run()


if __name__ == "__main__":
    main()
