"""Read-only/mock transports and the explicitly authorized Unitree DDS writer."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from .calibration import HardwareCalibration
from .contract import (
    HOMIE_ARM_ACTIVE_KD_MAX_SDK,
    HOMIE_ARM_ACTIVE_KP_MAX_SDK,
    HOMIE_ARM_DEFAULT_KD_SDK,
    HOMIE_ARM_DEFAULT_KP_SDK,
    HOMIE_ARM_DEFAULT_Q_SDK,
    HOMIE_ARM_TAU_FF_LIMIT_SDK,
    JOINT_LOWER_SDK,
    JOINT_UPPER_SDK,
)
from .observation import RobotState29
from .upper_body import ArmTarget14, UpperBodyTargetMux
from .safety import SafetyMonitor
from .unitree_crc import PurePythonHGCRC


REAL_MOTION_TOKEN = "I_UNDERSTAND_THIS_WRITES_UNITREE_G1_LOWCMD"


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandFrame29:
    q_target_sdk: np.ndarray
    kp_sdk: np.ndarray
    kd_sdk: np.ndarray
    tau_ff_sdk: np.ndarray
    sequence: int
    monotonic_ns: int

    @classmethod
    def make(
        cls,
        *,
        q_target_sdk: Any,
        kp_sdk: Any,
        kd_sdk: Any,
        tau_ff_sdk: Any = None,
        sequence: int,
        monotonic_ns: int | None = None,
    ) -> "CommandFrame29":
        vectors = []
        if tau_ff_sdk is None:
            tau_ff_sdk = np.zeros(29, dtype=np.float32)
        for value, label in (
            (q_target_sdk, "q_target_sdk"),
            (kp_sdk, "kp_sdk"),
            (kd_sdk, "kd_sdk"),
            (tau_ff_sdk, "tau_ff_sdk"),
        ):
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape != (29,) or not np.all(np.isfinite(array)):
                raise ValueError("%s must contain exactly 29 finite values" % label)
            vectors.append(array.copy())
        if np.any(vectors[1] < 0.0) or np.any(vectors[2] < 0.0):
            raise ValueError("PD gains must be non-negative")
        stamp = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
        if int(sequence) < 0 or stamp <= 0:
            raise ValueError("command frame identity is invalid")
        return cls(vectors[0], vectors[1], vectors[2], vectors[3], int(sequence), stamp)

    def with_q_target(self, q_target_sdk: Any) -> "CommandFrame29":
        """Return the same frame identity with a projected position target."""

        return CommandFrame29.make(
            q_target_sdk=q_target_sdk,
            kp_sdk=self.kp_sdk,
            kd_sdk=self.kd_sdk,
            tau_ff_sdk=self.tau_ff_sdk,
            sequence=self.sequence,
            monotonic_ns=self.monotonic_ns,
        )


@dataclass(frozen=True)
class ArmCommand14:
    """One atomic ``rt/arm_sdk`` arm command in SDK order (15..28)."""

    q_arm_sdk: np.ndarray
    kp_arm_sdk: np.ndarray
    kd_arm_sdk: np.ndarray
    tau_ff_arm_sdk: np.ndarray
    authority_weight: float
    monotonic_ns: int
    sequence: int

    @classmethod
    def make(
        cls,
        *,
        q_arm_sdk: Any,
        kp_arm_sdk: Any,
        kd_arm_sdk: Any,
        tau_ff_arm_sdk: Any,
        authority_weight: float,
        monotonic_ns: int,
        sequence: int,
    ) -> "ArmCommand14":
        vectors: list[np.ndarray] = []
        for value, label in (
            (q_arm_sdk, "arm q"),
            (kp_arm_sdk, "arm kp"),
            (kd_arm_sdk, "arm kd"),
            (tau_ff_arm_sdk, "arm tau"),
        ):
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.shape != (14,) or not np.all(np.isfinite(array)):
                raise ValueError("%s must contain exactly fourteen finite values" % label)
            vectors.append(array.copy())
        if np.any(vectors[1] < 0.0) or np.any(vectors[2] < 0.0):
            raise ValueError("arm SDK PD gains must be non-negative")
        target = ArmTarget14.make(
            vectors[0],
            authority_weight=authority_weight,
            monotonic_ns=monotonic_ns,
            sequence=sequence,
        )
        return cls(
            q_arm_sdk=target.q_arm_sdk,
            kp_arm_sdk=vectors[1],
            kd_arm_sdk=vectors[2],
            tau_ff_arm_sdk=vectors[3],
            authority_weight=target.authority_weight,
            monotonic_ns=target.monotonic_ns,
            sequence=target.sequence,
        )

    def target(self) -> ArmTarget14:
        return ArmTarget14.make(
            self.q_arm_sdk,
            authority_weight=self.authority_weight,
            monotonic_ns=self.monotonic_ns,
            sequence=self.sequence,
        )


class StateSource(Protocol):
    def sample(self) -> RobotState29:
        ...


class CommandSink(Protocol):
    is_hardware: bool

    def update(self, frame: CommandFrame29) -> None:
        ...

    def close(self) -> None:
        ...


class ShadowCommandSink:
    """In-memory sink with no Unitree import and no publication capability."""

    is_hardware = False

    def __init__(self) -> None:
        self.latest: CommandFrame29 | None = None
        self.count = 0

    def update(self, frame: CommandFrame29) -> None:
        self.latest = frame
        self.count += 1

    def close(self) -> None:
        return None


class UnitreeArmSDKTargetSource:
    """Read the 14 arm q/Kp/Kd/tau fields from the existing ``rt/arm_sdk`` topic.

    The message's leg/waist targets, gains and feed-forward torques are always
    ignored. DreamWaQ remains the only process that publishes ``rt/lowcmd``.
    """

    def __init__(
        self,
        *,
        verify_crc: bool = True,
        poll_period_s: float = 0.002,
        reader_cpu: int | None = None,
        subscriber: Any = None,
    ) -> None:
        self.verify_crc = bool(verify_crc)
        self.poll_period_s = float(poll_period_s)
        if self.poll_period_s <= 0.0:
            raise ValueError("arm_sdk poll period must be positive")
        self.reader_cpu = None if reader_cpu is None else int(reader_cpu)
        if self.reader_cpu is not None and self.reader_cpu < 0:
            raise ValueError("arm_sdk reader CPU must be non-negative")
        self._subscriber = subscriber
        # ``rt/arm_sdk`` can arrive at the same 250-Hz cadence as the LowCmd
        # writer.  Keep the reference implementation only as an offline
        # fallback/parity check; using it per arm packet monopolizes the GIL.
        self._pure_crc = PurePythonHGCRC()
        self._vendor_crc: Any | None = None
        self._crc_parity_preflight_done = False
        self._packet_crc_verified = False
        self._latest_raw: tuple[Any, int, int] | None = None
        self._raw_sequence = 0
        self._latest: ArmTarget14 | None = None
        self._latest_command: ArmCommand14 | None = None
        self._decoded_sequence = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self.last_error: str | None = None
        self.read_count = 0
        self.decode_count = 0
        self.crc_error_count = 0

    @staticmethod
    def _field(value: Any) -> Any:
        return value() if callable(value) else value

    def start(self) -> None:
        if self._thread is not None:
            raise TransportError("arm_sdk reader is already started")
        if self._subscriber is None:
            try:
                from unitree_sdk2py.core.channel import ChannelSubscriber
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            except (ImportError, OSError) as exc:
                raise TransportError("unitree_sdk2py is required only for G1 DDS mode") from exc
            self._subscriber = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        try:
            self._subscriber.Init()
        except BaseException as exc:
            raise TransportError("cannot initialize read-only rt/arm_sdk subscriber") from exc
        if self.verify_crc:
            try:
                from unitree_sdk2py.utils.crc import CRC

                self._vendor_crc = CRC()
                print("[G1-29][transport] ArmSDK CRC backend=unitree_sdk2py", flush=True)
            except (ImportError, OSError):
                # Unit tests and offline injected subscribers retain the
                # deterministic reference implementation.
                self._vendor_crc = None
        self._thread = threading.Thread(target=self._loop, name="dreamwaq-g1-arm-sdk", daemon=True)
        self._thread.start()

    def _message_crc(self, message: Any) -> int:
        engine = self._vendor_crc if self._vendor_crc is not None else self._pure_crc
        calculated = int(engine.Crc(message))
        if self._vendor_crc is not None and not self._crc_parity_preflight_done:
            reference = int(self._pure_crc.Crc(message))
            if calculated != reference:
                raise TransportError("vendor/pure rt/arm_sdk CRC parity preflight failure")
            self._crc_parity_preflight_done = True
            print(
                "[G1-29][transport] ArmSDK CRC parity preflight=PASS; "
                "runtime backend=unitree_sdk2py",
                flush=True,
            )
        return calculated

    def _verify_packet_once(self, message: Any) -> None:
        if self.verify_crc and not self._packet_crc_verified:
            received = int(self._field(getattr(message, "crc")))
            calculated = self._message_crc(message)
            if received != calculated:
                self.crc_error_count += 1
                raise TransportError("arm_sdk CRC mismatch")
            # ``rt/arm_sdk`` is a local, reliable DDS arm-command input, not
            # the hardware LowState safety boundary.  Match HOMIE: prove CRC
            # parity and one real packet before activation, then avoid a full
            # 35-slot CRC traversal at every 250-Hz arm update.  Every packet
            # still undergoes finite-vector validation, arm gain/tau clipping,
            # target bounds and total PD-effort projection in the sole writer.
            self._packet_crc_verified = True

    def _message_to_command(self, message: Any, stamp: int) -> ArmCommand14:
        self._verify_packet_once(message)
        motors = tuple(self._field(getattr(message, "motor_cmd")))
        if len(motors) != 35:
            raise TransportError("HG arm_sdk command must expose exactly 35 motor slots")
        q_arm = [float(self._field(getattr(motors[index], "q"))) for index in range(15, 29)]
        kp_arm = [float(self._field(getattr(motors[index], "kp"))) for index in range(15, 29)]
        kd_arm = [float(self._field(getattr(motors[index], "kd"))) for index in range(15, 29)]
        tau_arm = [float(self._field(getattr(motors[index], "tau"))) for index in range(15, 29)]
        authority = float(self._field(getattr(motors[29], "q")))
        result = ArmCommand14.make(
            q_arm_sdk=q_arm,
            kp_arm_sdk=kp_arm,
            kd_arm_sdk=kd_arm,
            tau_ff_arm_sdk=tau_arm,
            authority_weight=authority,
            monotonic_ns=stamp,
            sequence=self._sequence,
        )
        self._sequence += 1
        return result

    def _loop(self) -> None:
        try:
            self._configure_reader_thread()
        except BaseException as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            return
        while not self._stop.is_set():
            try:
                message = self._subscriber.Read()
                if message is None:
                    # Real unitree_sdk2py Read() blocks like HOMIE.  Keep a
                    # bounded fallback for injected/non-blocking subscribers
                    # so a temporary empty read cannot become a GIL spin.
                    time.sleep(self.poll_period_s)
                    continue
                # Match HOMIE: this 250-Hz reader stores the newest raw DDS
                # command only. Parsing 14 q/Kp/Kd/tau vectors here would hold
                # the shared Python GIL on every teleop update and disrupt the
                # 50-Hz FlashSAC policy. The writer/policy decodes on demand.
                stamp = time.monotonic_ns()
                with self._lock:
                    self._raw_sequence += 1
                    self._latest_raw = (message, stamp, self._raw_sequence)
                    self.read_count += 1
                self._ready.set()
            except BaseException as exc:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                time.sleep(self.poll_period_s)

    def _configure_reader_thread(self) -> None:
        if self.reader_cpu is None:
            return
        try:
            os.sched_setaffinity(0, {self.reader_cpu})
        except OSError as exc:
            raise TransportError(
                "cannot pin rt/arm_sdk reader to CPU %d" % self.reader_cpu
            ) from exc
        print(
            "[G1-29][transport] ArmSDK reader affinity=%s"
            % sorted(os.sched_getaffinity(0)),
            flush=True,
        )

    def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        return bool(self._ready.wait(float(timeout_s)))

    def sample(self) -> ArmTarget14 | None:
        command = self.sample_command()
        return None if command is None else command.target()

    def sample_command(self) -> ArmCommand14 | None:
        """Return q/kp/kd/tau from one verified arm_sdk sample atomically."""

        with self._lock:
            raw = self._latest_raw
            cached = self._latest_command
            decoded_sequence = self._decoded_sequence
        if raw is None:
            return None
        message, stamp, raw_sequence = raw
        if cached is not None and decoded_sequence == raw_sequence:
            return cached
        # A DDS read creates a distinct message object.  Consume this captured
        # snapshot even when the reader publishes a newer object while it is
        # being decoded.  Retrying at 250 Hz can otherwise livelock the
        # 250-Hz writer under load; HOMIE uses this same latest-snapshot model.
        command = self._message_to_command(message, stamp)
        with self._lock:
            self._latest_command = command
            self._latest = command.target()
            self._decoded_sequence = raw_sequence
            self.decode_count += 1
        return command

    def writer_raw_snapshot(self) -> tuple[Any, int, int] | None:
        """Return HOMIE-style latest ``rt/arm_sdk`` packet without decoding it.

        HOMIE's reader only caches the incoming DDS object.  Its LowCmd writer
        consumes that one immutable snapshot at write rate; it does not create
        deployment-domain frame objects, muxes, or retry loops.  Keep this
        accessor equally small so the FlashSAC writer can follow that path.
        """

        with self._lock:
            return self._latest_raw

    def verify_writer_packet_once(self, message: Any) -> None:
        """Perform the same one-packet CRC proof used by HOMIE-style writer input."""

        self._verify_packet_once(message)

    def close(self) -> None:
        self._stop.set()
        close = getattr(self._subscriber, "Close", None)
        if callable(close):
            try:
                # Read() is intentionally blocking like HOMIE's arm reader;
                # close the subscriber first so the blocked call can return.
                close()
            except BaseException as exc:
                self.last_error = self.last_error or (
                    "arm_sdk subscriber close failed: %s: %s" % (type(exc).__name__, exc)
                )
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None


class MockPlantTransport:
    """Deterministic CPU plant used for loop/timing/failure-injection checks."""

    is_hardware = False

    def __init__(self, *, default_q_sdk: Any, mode_machine: int = 5, policy_dt: float = 0.02) -> None:
        self.q = np.asarray(default_q_sdk, dtype=np.float32).reshape(-1)
        if self.q.shape != (29,) or not np.all(np.isfinite(self.q)):
            raise ValueError("FlashSAC mock plant requires 29 finite default-joint values")
        self.q = self.q.copy()
        self.dq = np.zeros(29, dtype=np.float32)
        self.target = self.q.copy()
        self.mode_machine = int(mode_machine)
        self.policy_dt = float(policy_dt)
        self.sequence = 0
        self.latest: CommandFrame29 | None = None
        self.remote = bytes(40)
        self.inject_stale = False
        self.inject_tilt_rad = 0.0

    def update(self, frame: CommandFrame29) -> None:
        self.latest = frame
        self.target = frame.q_target_sdk.copy()

    def sample(self) -> RobotState29:
        max_delta = 8.0 * self.policy_dt
        delta = np.clip(self.target - self.q, -max_delta, max_delta)
        self.dq = (delta / self.policy_dt).astype(np.float32)
        self.q += delta
        self.sequence += 1
        half = 0.5 * float(self.inject_tilt_rad)
        quat = (float(np.cos(half)), float(np.sin(half)), 0.0, 0.0)
        stamp = time.monotonic_ns() - (100_000_000 if self.inject_stale else 0)
        return RobotState29.make(
            q_sdk=self.q,
            dq_sdk=self.dq,
            quaternion_wxyz=quat,
            angular_velocity_body=(0.0, 0.0, 0.0),
            monotonic_ns=stamp,
            mode_machine=self.mode_machine,
            source_tick=self.sequence,
            wireless_remote=self.remote,
            reserved_q_raw=np.zeros(6, dtype=np.float32),
        )

    def close(self) -> None:
        return None


class _OwnershipLease:
    def __init__(self, path: Any) -> None:
        target = Path(path).expanduser()
        if target.is_symlink():
            raise TransportError("LowCmd ownership lock path must not be a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._file = target.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            raise TransportError("another local process owns the G1 LowCmd lease") from exc
        self._file.seek(0)
        self._file.truncate()
        self._file.write("pid=%d\n" % os.getpid())
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


class UnitreeDDSStateSource:
    """Read-only polling subscriber for verified 35-slot HG LowState."""

    def __init__(
        self,
        *,
        network_interface: str,
        calibration: HardwareCalibration,
        verify_crc: bool = True,
        poll_period_s: float = 0.002,
        decode_rate_hz: float = 250.0,
        reader_cpu: int | None = None,
        initialize_channel_factory: bool = True,
        subscriber: Any = None,
    ) -> None:
        self.network_interface = str(network_interface)
        self.calibration = calibration
        self.verify_crc = bool(verify_crc)
        self.poll_period_s = float(poll_period_s)
        self.decode_rate_hz = float(decode_rate_hz)
        self.reader_cpu = None if reader_cpu is None else int(reader_cpu)
        if self.poll_period_s <= 0.0:
            raise ValueError("LowState poll period must be positive")
        if not np.isfinite(self.decode_rate_hz) or not 50.0 <= self.decode_rate_hz <= 500.0:
            raise ValueError("LowState reader rate must be finite and in [50, 500] Hz")
        self.reader_period_ns = int(round(1.0e9 / self.decode_rate_hz))
        if self.reader_cpu is not None and self.reader_cpu < 0:
            raise ValueError("LowState reader CPU must be non-negative")
        self._initialize_channel_factory = bool(initialize_channel_factory)
        self._subscriber = subscriber
        self._crc = PurePythonHGCRC()
        self._vendor_crc: Any | None = None
        self._latest_raw: tuple[Any, int, int] | None = None
        self._raw_sequence = 0
        self._latest: RobotState29 | None = None
        self._decoded_sequence = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_tick: int | None = None
        self.last_error: str | None = None
        self.read_count = 0
        self.decode_count = 0
        self.crc_error_count = 0

    @staticmethod
    def _field(value: Any) -> Any:
        return value() if callable(value) else value

    def start(self) -> None:
        if self._thread is not None:
            raise TransportError("LowState reader is already started")
        if self._subscriber is None:
            try:
                from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            except (ImportError, OSError) as exc:
                raise TransportError("unitree_sdk2py is required only for G1 DDS mode") from exc
            if self._initialize_channel_factory:
                ChannelFactoryInitialize(0, networkInterface=self.network_interface)
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        try:
            self._subscriber.Init()
        except BaseException as exc:
            raise TransportError("cannot initialize rt/lowstate subscriber") from exc
        if self.verify_crc:
            try:
                from unitree_sdk2py.utils.crc import CRC

                self._vendor_crc = CRC()
                print("[G1-29][transport] LowState CRC backend=unitree_sdk2py", flush=True)
            except (ImportError, OSError):
                # Offline/injected transports retain the reference Python CRC.
                self._vendor_crc = None
        print(
            "[G1-29][transport] LowState raw-cache reader_rate_hz=%g"
            % self.decode_rate_hz,
            flush=True,
        )
        self._thread = threading.Thread(target=self._loop, name="dreamwaq-g1-lowstate", daemon=True)
        self._thread.start()

    def _message_crc(self, message: Any) -> int:
        engine = self._vendor_crc if self._vendor_crc is not None else self._crc
        return int(engine.Crc(message))

    def _message_to_state(self, message: Any, stamp: int) -> RobotState29:
        tick = int(self._field(getattr(message, "tick")))
        if self._last_tick is not None:
            delta = (tick - self._last_tick) & 0xFFFFFFFF
            if delta == 0:
                # ``ChannelSubscriber.Read()`` can expose the same latest DDS
                # sample more than once between hardware publications.  HOMIE
                # keeps consuming its last decoded state in that case.  Keep
                # the original monotonic timestamp so the independent 40-ms
                # freshness guard still faults a *real* LowState stall.
                if self._latest is not None:
                    return self._latest
                raise TransportError("LowState tick repeated before a verified state was cached")
            if delta >= 0x80000000:
                raise TransportError("LowState tick replayed")
        if self.verify_crc:
            received = int(self._field(getattr(message, "crc")))
            calculated = self._message_crc(message)
            if received != calculated:
                self.crc_error_count += 1
                raise TransportError("LowState CRC mismatch")
        motors = tuple(self._field(getattr(message, "motor_state")))
        if len(motors) != 35:
            raise TransportError("HG LowState must expose exactly 35 motor slots")
        q_raw = np.asarray([float(self._field(getattr(motor, "q"))) for motor in motors], dtype=np.float32)
        dq = np.asarray([float(self._field(getattr(motor, "dq"))) for motor in motors], dtype=np.float32)
        tau = np.asarray([float(self._field(getattr(motor, "tau_est"))) for motor in motors[:29]], dtype=np.float32)
        imu = self._field(getattr(message, "imu_state"))
        quat = tuple(float(value) for value in self._field(getattr(imu, "quaternion")))
        gyro = tuple(float(value) for value in self._field(getattr(imu, "gyroscope")))
        remote = bytes(int(value) & 0xFF for value in self._field(getattr(message, "wireless_remote")))
        mode = int(self._field(getattr(message, "mode_machine")))
        state = RobotState29.make(
            q_sdk=q_raw[:29] - self.calibration.encoder_zero_offsets_rad,
            dq_sdk=dq[:29],
            quaternion_wxyz=quat,
            angular_velocity_body=gyro,
            monotonic_ns=stamp,
            mode_machine=mode,
            source_tick=tick,
            wireless_remote=remote,
            reserved_q_raw=q_raw[29:],
            tau_est_sdk=tau,
        )
        self._last_tick = tick
        return state

    def _loop(self) -> None:
        try:
            self._configure_reader_thread()
        except BaseException as exc:
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            return
        next_read = time.monotonic_ns()
        while not self._stop.is_set():
            try:
                remaining_ns = next_read - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(min(self.poll_period_s, remaining_ns / 1.0e9))
                    continue
                message = self._subscriber.Read()
                if message is None:
                    time.sleep(self.poll_period_s)
                    continue
                # Match HOMIE's I/O boundary: reader work only stores the
                # newest native DDS message. Decoding 35 motors plus IMU into
                # Python/NumPy here at the source rate monopolizes the process
                # GIL and starves the 50-Hz policy. ``sample()`` verifies and
                # converts this raw cache only when the policy consumes it.
                stamp = time.monotonic_ns()
                with self._lock:
                    self._raw_sequence += 1
                    self._latest_raw = (message, stamp, self._raw_sequence)
                    self.read_count += 1
                self._ready.set()
                next_read += self.reader_period_ns
                if next_read < time.monotonic_ns():
                    # Do not burst through missed read deadlines after a
                    # scheduler stall.
                    next_read = time.monotonic_ns()
            except BaseException as exc:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                time.sleep(self.poll_period_s)

    def _configure_reader_thread(self) -> None:
        if self.reader_cpu is None:
            return
        try:
            os.sched_setaffinity(0, {self.reader_cpu})
        except OSError as exc:
            raise TransportError(
                "cannot pin LowState reader to CPU %d" % self.reader_cpu
            ) from exc
        print(
            "[G1-29][transport] LowState reader affinity=%s"
            % sorted(os.sched_getaffinity(0)),
            flush=True,
        )

    def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        return bool(self._ready.wait(float(timeout_s)))

    def sample(self) -> RobotState29:
        """Return a freshly verified state for the 50-Hz policy path."""

        with self._lock:
            raw = self._latest_raw
            cached = self._latest
            decoded_sequence = self._decoded_sequence
        if raw is None:
            raise TransportError("no rt/lowstate sample is available")
        message, stamp, raw_sequence = raw
        if cached is not None and decoded_sequence == raw_sequence:
            return cached
        # Keep one coherent, CRC-verified packet per policy tick.  A newer raw
        # packet may arrive during conversion, but retrying the conversion
        # until the reader pauses creates a starvation loop at 250 Hz.  The
        # captured packet is still bounded by the existing 40-ms freshness
        # check, and the next tick consumes the newer snapshot.
        state = self._message_to_state(message, stamp)
        with self._lock:
            self._latest = state
            self._decoded_sequence = raw_sequence
            self.decode_count += 1
        return state

    def sample_cached(self) -> RobotState29:
        """Return the most recently policy-verified state without decoding."""

        with self._lock:
            if self._latest is None:
                raise TransportError("no policy-verified rt/lowstate sample is available")
            return self._latest

    def writer_raw_snapshot(self) -> tuple[Any, int, int]:
        """Return HOMIE's latest raw LowState cache for the LowCmd writer.

        The 50-Hz policy path is responsible for full CRC/decode/safety
        validation.  The high-rate writer needs only a current hardware
        heartbeat, mode, and the six reserved HG positions while publishing.
        Keeping that split avoids a second full 29-joint safety pass under the
        same Python GIL on every 4-ms write period.
        """

        with self._lock:
            if self._latest_raw is None:
                raise TransportError("no raw rt/lowstate sample is available")
            return self._latest_raw

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        close = getattr(self._subscriber, "Close", None)
        if callable(close):
            close()


class UnitreeDDSCommandSink:
    """Explicitly authorized HOMIE-style 250-Hz LowCmd writer."""

    is_hardware = True

    def __init__(
        self,
        *,
        state_source: UnitreeDDSStateSource,
        calibration: HardwareCalibration,
        safety: SafetyMonitor,
        expected_mode_machine: int,
        motion_token: str,
        ownership_lock_path: Any = "/tmp/dreamwaq-g1-29dof-lowcmd.lock",
        max_frame_age_ns: int = 60_000_000,
        max_writer_lateness_ns: int = 10_000_000,
        max_active_duration_s: float | None = None,
        writer_rate_hz: float = 250.0,
        writer_cpu: int | None = None,
        writer_spin_us: int = 0,
        writer_rt_priority: int = 0,
        high_rate_arm_source: UnitreeArmSDKTargetSource | None = None,
        arm_default_q_sdk: Any = None,
        arm_target_max_age_ns: int = 100_000_000,
        arm_velocity_limit_rad_s: float = 10.0,
        publisher: Any = None,
    ) -> None:
        if motion_token != REAL_MOTION_TOKEN:
            raise TransportError("physical LowCmd authority token is missing")
        self.state_source = state_source
        self.calibration = calibration
        self._kp_cap = calibration.kp_sdk
        self._kd_cap = calibration.kd_sdk
        self.safety = safety
        self.expected_mode_machine = int(expected_mode_machine)
        self.max_frame_age_ns = int(max_frame_age_ns)
        self.max_active_duration_ns = (
            None if max_active_duration_s is None else int(float(max_active_duration_s) * 1.0e9)
        )
        self.writer_rate_hz = float(writer_rate_hz)
        if not np.isfinite(self.writer_rate_hz) or not 100.0 <= self.writer_rate_hz <= 500.0:
            raise ValueError("LowCmd writer rate must be finite and in [100, 500] Hz")
        self.writer_period_ns = int(round(1.0e9 / self.writer_rate_hz))
        # HOMIE publishes LowCmd at 250 Hz on this G1.  Permit at most four
        # writer periods of scheduler delay while retaining the existing 10 ms
        # envelope at 500 Hz.  The independent 60 ms command-frame freshness
        # guard remains stricter than this for a stalled policy source.
        self.max_writer_lateness_ns = max(
            int(max_writer_lateness_ns), 4 * self.writer_period_ns
        )
        self.writer_cpu = None if writer_cpu is None else int(writer_cpu)
        self.writer_spin_ns = int(writer_spin_us) * 1_000
        self.writer_rt_priority = int(writer_rt_priority)
        if not 0 <= self.expected_mode_machine <= 255:
            raise ValueError("expected_mode_machine must be a uint8")
        if min(self.max_frame_age_ns, self.max_writer_lateness_ns) <= 0:
            raise ValueError("LowCmd writer timing bounds must be positive")
        if self.max_active_duration_ns is not None and self.max_active_duration_ns <= 0:
            raise ValueError("LowCmd writer maximum duration must be positive when set")
        if self.writer_cpu is not None and self.writer_cpu < 0:
            raise ValueError("LowCmd writer CPU must be non-negative")
        if not 0 <= self.writer_spin_ns <= 1_000_000:
            raise ValueError("LowCmd writer spin must be in [0, 1000] microseconds")
        if not 0 <= self.writer_rt_priority <= 99:
            raise ValueError("LowCmd writer real-time priority must be in [0, 99]")
        self._high_rate_arm_source = high_rate_arm_source
        # HOMIE reads the already-verified rt/arm_sdk snapshot directly in
        # its first LowCmd callback.  Arm ownership is independent of
        # lower-body startup, so do not defer this until a handoff event.
        self._high_rate_arm_enabled = high_rate_arm_source is not None
        self._homie_arm_default_kp_sdk = HOMIE_ARM_DEFAULT_KP_SDK.copy()
        self._homie_arm_default_kd_sdk = HOMIE_ARM_DEFAULT_KD_SDK.copy()
        self._homie_arm_kp_max_sdk = HOMIE_ARM_ACTIVE_KP_MAX_SDK.copy()
        self._homie_arm_kd_max_sdk = HOMIE_ARM_ACTIVE_KD_MAX_SDK.copy()
        self._homie_arm_tau_ff_limit_sdk = HOMIE_ARM_TAU_FF_LIMIT_SDK.copy()
        self._last_high_rate_arm_kp_sdk = self._homie_arm_default_kp_sdk.copy()
        self._last_high_rate_arm_kd_sdk = self._homie_arm_default_kd_sdk.copy()
        self._last_writer_work_ns = 0
        self._last_writer_arm_ns = 0
        self._last_writer_state_ns = 0
        self._last_writer_safety_ns = 0
        self._last_writer_projection_ns = 0
        self._last_writer_validation_ns = 0
        self._last_writer_write_ns = 0
        # Keep the actual 250-Hz writer as close as possible to HOMIE's
        # ``get_command() -> send_cmd()`` path: a fixed lower-body frame plus
        # a directly cached arm SDK packet.  The buffers are writer-thread
        # owned, so arm updates need no CommandFrame/ArmTarget allocation or
        # whole-vector rebuilding at 250 Hz.
        self._writer_frame_sequence = -1
        self._writer_q_sdk = np.zeros(29, dtype=np.float32)
        self._writer_kp_sdk = np.zeros(29, dtype=np.float32)
        self._writer_kd_sdk = np.zeros(29, dtype=np.float32)
        self._writer_tau_ff_sdk = np.zeros(29, dtype=np.float32)
        self._writer_q_raw_target = np.zeros(29, dtype=np.float32)
        self._writer_reserved_q_raw = np.zeros(6, dtype=np.float32)
        self._writer_arm_raw_sequence = -1
        self._writer_arm_stamp_ns = 0
        self._writer_arm_weight = 0.0
        self._writer_arm_tau_input = np.zeros(14, dtype=np.float32)
        self._arm_indices = slice(15, 29)
        self._arm_lower_sdk = JOINT_LOWER_SDK[self._arm_indices].copy()
        self._arm_upper_sdk = JOINT_UPPER_SDK[self._arm_indices].copy()
        if high_rate_arm_source is not None:
            # The external arm topic may be absent during bring-up.  HOMIE
            # holds this conservative neutral pose until it receives an active
            # arm packet, never the FlashSAC training arm reference.
            self._writer_q_sdk[self._arm_indices] = HOMIE_ARM_DEFAULT_Q_SDK
            self._writer_kp_sdk[self._arm_indices] = self._homie_arm_default_kp_sdk
            self._writer_kd_sdk[self._arm_indices] = self._homie_arm_default_kd_sdk
        self._arm_target_max_age_ns = int(arm_target_max_age_ns)
        self._lease = _OwnershipLease(ownership_lock_path)
        self._publisher = publisher
        self._message = None
        self._vendor_crc = None
        self._pure_crc = PurePythonHGCRC()
        self._crc_parity_preflight_done = False
        self._frame: CommandFrame29 | None = None
        self._frame_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._enabled_ns = 0
        self.fault: str | None = None
        self.write_count = 0
        self._disable_published = False
        self._last_sequence = -1
        try:
            if self._publisher is None:
                from unitree_sdk2py.core.channel import ChannelPublisher
                from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
                from unitree_sdk2py.utils.crc import CRC

                self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
                self._publisher.Init()
                self._message = unitree_hg_msg_dds__LowCmd_()
                self._vendor_crc = CRC()
            else:
                factory = getattr(self._publisher, "message_factory", None)
                if not callable(factory):
                    raise TransportError("injected LowCmd publisher must expose message_factory()")
                self._message = factory()
                self._vendor_crc = getattr(self._publisher, "crc", None)
            if len(self._message.motor_cmd) != 35:
                raise TransportError("HG LowCmd must expose exactly 35 motor slots")
            self._validate_crc_backend_once()
        except BaseException:
            self._lease.close()
            raise

    def update(self, frame: CommandFrame29) -> None:
        with self._frame_lock:
            if self.fault is not None:
                raise TransportError("LowCmd writer is faulted: %s" % self.fault)
            if frame.sequence <= self._last_sequence:
                raise TransportError("LowCmd frame sequence is stale or replayed")
            if np.any(frame.kp_sdk > self._kp_cap + 1.0e-6):
                raise TransportError("LowCmd kp exceeds the calibration gain envelope")
            if np.any(frame.kd_sdk > self._kd_cap + 1.0e-6):
                raise TransportError("LowCmd kd exceeds the calibration gain envelope")
            self._frame = frame
            self._last_sequence = frame.sequence

    @staticmethod
    def _activation_frame(frame: CommandFrame29, *, monotonic_ns: int) -> CommandFrame29:
        """Mark the already-validated first shared-memory policy frame active."""

        return CommandFrame29.make(
            q_target_sdk=frame.q_target_sdk,
            kp_sdk=frame.kp_sdk,
            kd_sdk=frame.kd_sdk,
            tau_ff_sdk=frame.tau_ff_sdk,
            sequence=frame.sequence,
            monotonic_ns=int(monotonic_ns),
        )

    def reset_sequence_after_handoff(self) -> None:
        """Begin the policy sequence domain without exposing private writer state."""

        with self._frame_lock:
            if not self._running or self.fault is not None or self._frame is None:
                raise TransportError("cannot reset sequence outside a healthy active handoff")
            self._last_sequence = -1
            self._writer_frame_sequence = -1

    def enable(self) -> None:
        if self._running:
            raise TransportError("LowCmd writer is already enabled")
        with self._frame_lock:
            if self._frame is None:
                raise TransportError("set a fresh handoff frame before enabling LowCmd")
        state = self.state_source.sample()
        self.safety.check_state(state)
        if state.mode_machine != self.expected_mode_machine:
            raise TransportError("verified LowState mode does not match expected_mode_machine")
        activation_ns = time.monotonic_ns()
        with self._frame_lock:
            # This is intentionally only the one-time pre-writer activation
            # frame. Once the writer runs, update() preserves each policy
            # process timestamp for the ordinary freshness watchdog.
            if self._frame is None:
                raise TransportError("handoff frame disappeared before enabling LowCmd")
            self._frame = self._activation_frame(self._frame, monotonic_ns=activation_ns)
        self._enabled_ns = activation_ns
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, name="dreamwaq-g1-lowcmd", daemon=True)
        self._thread.start()

    def enable_high_rate_arm_override(self) -> None:
        """Compatibility hook for the writer-rate HOMIE arm source.

        New real-mode sessions enable this source from the first writer tick,
        matching HOMIE.  This remains for legacy callers that enable it later.
        """

        if self._high_rate_arm_source is None or self._high_rate_arm_mux is None:
            raise TransportError("high-rate arm override was not configured")
        if not self._running or self.fault is not None:
            raise TransportError("high-rate arm override requires a healthy active LowCmd writer")
        if self._high_rate_arm_enabled:
            return
        self._high_rate_arm_mux.reset()
        self._writer_arm_raw_sequence = -1
        self._high_rate_arm_enabled = True

    def _apply_high_rate_arm_override(self, frame: CommandFrame29, *, now_ns: int) -> CommandFrame29:
        if not self._high_rate_arm_enabled:
            # The physical arm-sdk route uses HOMIE's conservative static arm
            # impedance during the measured default-pose handoff as well. Only
            # target/tau ownership waits for ``enable_high_rate_arm_override``.
            if self._high_rate_arm_source is None:
                return frame
            kp = frame.kp_sdk.copy()
            kd = frame.kd_sdk.copy()
            tau_ff = frame.tau_ff_sdk.copy()
            arm_indices = self._high_rate_arm_mux.mapper.arm_sdk_indices if self._high_rate_arm_mux is not None else slice(15, 29)
            kp[arm_indices] = self._homie_arm_default_kp_sdk
            kd[arm_indices] = self._homie_arm_default_kd_sdk
            tau_ff[arm_indices] = 0.0
            return CommandFrame29.make(
                q_target_sdk=frame.q_target_sdk,
                kp_sdk=kp,
                kd_sdk=kd,
                tau_ff_sdk=tau_ff,
                sequence=frame.sequence,
                monotonic_ns=frame.monotonic_ns,
            )
        source = self._high_rate_arm_source
        mux = self._high_rate_arm_mux
        if source is None or mux is None:
            raise TransportError("high-rate arm override lost its configured source")
        # A reader thread can publish an arm sample after the writer captured
        # its initial timestamp.  Timestamp *after* sampling to avoid treating
        # that valid concurrent target as future-dated.
        command_sample = getattr(source, "sample_command", None)
        source_command = command_sample() if callable(command_sample) else None
        # HOMIE considers teleoperation active only at weight >= 0.5.  Below
        # that threshold it holds the last q/Kp/Kd and sends no tau_ff.
        source_frame = (
            # HOMIE uses the arm_sdk weight as an engaged/not-engaged gate for
            # q/Kp/Kd, not as a position interpolation factor. tau_ff remains
            # weight-scaled below, exactly as HOMIE does.
            ArmTarget14.make(
                source_command.q_arm_sdk,
                authority_weight=1.0,
                monotonic_ns=source_command.monotonic_ns,
                sequence=source_command.sequence,
            )
            if source_command is not None and source_command.authority_weight >= 0.5
            else (source.sample() if source_command is None else None)
        )
        arm_target = mux.resolve(source_frame, now_ns=max(int(now_ns), time.monotonic_ns()))
        merged = frame.q_target_sdk.copy()
        merged[mux.mapper.arm_sdk_indices] = arm_target
        kp = frame.kp_sdk.copy()
        kd = frame.kd_sdk.copy()
        tau_ff = frame.tau_ff_sdk.copy()
        arm_indices = mux.mapper.arm_sdk_indices
        # Start and stale/disengaged behavior match HOMIE: use its conservative
        # defaults until a fresh active arm_sdk command arrives, then hold the
        # last active gains while dropping tau_ff to zero.
        tau_arm = np.zeros(14, dtype=np.float32)
        if source_command is not None and mux.active:
            # ``rt/arm_sdk`` is an external input.  Preserve its supported
            # per-joint impedance profile, but do not turn a slightly newer
            # teleop/PATH gain into a full LowCmd writer fault.  Clamp it at
            # the same sealed envelope used for tau_ff and total PD effort.
            clipped_kp = np.clip(
                source_command.kp_arm_sdk,
                0.0,
                self._homie_arm_kp_max_sdk,
            ).astype(np.float32)
            clipped_kd = np.clip(
                source_command.kd_arm_sdk,
                0.0,
                self._homie_arm_kd_max_sdk,
            ).astype(np.float32)
            if (
                np.any(clipped_kp != source_command.kp_arm_sdk)
                or np.any(clipped_kd != source_command.kd_arm_sdk)
            ):
                last_report_ns = int(getattr(self, "_last_high_rate_arm_clamp_report_ns", 0))
                if int(now_ns) - last_report_ns >= 1_000_000_000:
                    print(
                        "[G1-29][upper-body] clipped rt/arm_sdk Kp/Kd to the PATH/HOMIE arm envelope",
                        flush=True,
                    )
                    self._last_high_rate_arm_clamp_report_ns = int(now_ns)
            self._last_high_rate_arm_kp_sdk = clipped_kp
            self._last_high_rate_arm_kd_sdk = clipped_kd
            tau_arm = np.clip(
                source_command.tau_ff_arm_sdk * source_command.authority_weight,
                -self._homie_arm_tau_ff_limit_sdk,
                self._homie_arm_tau_ff_limit_sdk,
            ).astype(np.float32)
        kp[arm_indices] = self._last_high_rate_arm_kp_sdk
        kd[arm_indices] = self._last_high_rate_arm_kd_sdk
        tau_ff[arm_indices] = tau_arm
        return CommandFrame29.make(
            q_target_sdk=merged,
            kp_sdk=kp,
            kd_sdk=kd,
            tau_ff_sdk=tau_ff,
            sequence=frame.sequence,
            monotonic_ns=frame.monotonic_ns,
        )

    def _refresh_writer_lower_frame(self, frame: CommandFrame29) -> None:
        """Copy a 50-Hz policy frame only when it actually changes.

        This mirrors HOMIE's writer-side cached leg state.  The direct arm
        merge below overwrites only joints 15..28 at each writer period.
        """

        if frame.sequence == self._writer_frame_sequence:
            return
        if self._high_rate_arm_source is None:
            np.copyto(self._writer_q_sdk, frame.q_target_sdk)
            np.copyto(self._writer_kp_sdk, frame.kp_sdk)
            np.copyto(self._writer_kd_sdk, frame.kd_sdk)
            np.copyto(self._writer_tau_ff_sdk, frame.tau_ff_sdk)
        else:
            # The FlashSAC frame owns legs+spine only.  Never copy policy arm
            # defaults back over a fresh rt/arm_sdk target between writer
            # ticks; this was the remaining competing arm-owner path.
            np.copyto(self._writer_q_sdk[:15], frame.q_target_sdk[:15])
            np.copyto(self._writer_kp_sdk[:15], frame.kp_sdk[:15])
            np.copyto(self._writer_kd_sdk[:15], frame.kd_sdk[:15])
            np.copyto(self._writer_tau_ff_sdk[:15], frame.tau_ff_sdk[:15])
        self._writer_frame_sequence = frame.sequence

    def _merge_homie_arm_into_writer_buffers(self, *, now_ns: int) -> None:
        """HOMIE-style direct ``rt/arm_sdk`` merge for the writer hot path.

        HOMIE keeps a raw arm DDS snapshot in its reader, then directly reads
        q/Kp/Kd/tau for motors 15..28 in its LowCmd callback.  Keep that
        shape here: no ArmCommand/ArmTarget objects, no mux and no 29-vector
        concatenation in the 250-Hz writer.  On inactive/stale arm input,
        positions and gains hold while tau_ff is explicitly zeroed, exactly
        like HOMIE's arm source.
        """

        source = self._high_rate_arm_source
        if source is None:
            return
        raw_snapshot = source.writer_raw_snapshot()
        if raw_snapshot is not None:
            message, stamp, raw_sequence = raw_snapshot
            if raw_sequence != self._writer_arm_raw_sequence:
                source.verify_writer_packet_once(message)
                motors = message.motor_cmd
                if len(motors) != 35:
                    raise TransportError("HG arm_sdk command must expose exactly 35 motor slots")
                authority = float(motors[29].q)
                self._writer_arm_weight = (
                    min(1.0, max(0.0, authority)) if np.isfinite(authority) else 0.0
                )
                if authority >= 0.5 and np.isfinite(authority):
                    q_arm = np.asarray([float(motors[index].q) for index in range(15, 29)], dtype=np.float32)
                    kp_arm = np.asarray([float(motors[index].kp) for index in range(15, 29)], dtype=np.float32)
                    kd_arm = np.asarray([float(motors[index].kd) for index in range(15, 29)], dtype=np.float32)
                    tau_arm = np.asarray([float(motors[index].tau) for index in range(15, 29)], dtype=np.float32)
                    # Preserve HOMIE's hold-on-bad-input behavior for position
                    # and impedance.  Feed-forward torque is never held.
                    if np.all(np.isfinite(q_arm)) and np.all(np.isfinite(kp_arm)) and np.all(np.isfinite(kd_arm)):
                        np.clip(q_arm, self._arm_lower_sdk, self._arm_upper_sdk, out=self._writer_q_sdk[self._arm_indices])
                        np.clip(kp_arm, 0.0, self._homie_arm_kp_max_sdk, out=self._writer_kp_sdk[self._arm_indices])
                        np.clip(kd_arm, 0.0, self._homie_arm_kd_max_sdk, out=self._writer_kd_sdk[self._arm_indices])
                        if np.all(np.isfinite(tau_arm)):
                            np.copyto(self._writer_arm_tau_input, tau_arm)
                        else:
                            self._writer_arm_tau_input.fill(0.0)
                        self._writer_arm_stamp_ns = int(stamp)
                    else:
                        self._writer_arm_weight = 0.0
                        self._writer_arm_tau_input.fill(0.0)
                else:
                    # Teleop disengaged: keep q/Kp/Kd, never hold tau_ff.
                    self._writer_arm_tau_input.fill(0.0)
                self._writer_arm_raw_sequence = int(raw_sequence)

        tau_out = self._writer_tau_ff_sdk[self._arm_indices]
        tau_out.fill(0.0)
        if (
            self._writer_arm_weight > 0.0
            and int(now_ns) - self._writer_arm_stamp_ns <= self._arm_target_max_age_ns
        ):
            np.multiply(self._writer_arm_tau_input, self._writer_arm_weight, out=tau_out)
            np.clip(
                tau_out,
                -self._homie_arm_tau_ff_limit_sdk,
                self._homie_arm_tau_ff_limit_sdk,
                out=tau_out,
            )

    def _seal_crc(self) -> None:
        if self._vendor_crc is None:
            crc = int(self._pure_crc.Crc(self._message))
        else:
            # This is the same native CRC path used by HOMIE.  The reference
            # Python implementation is checked once at construction, not in
            # the 250/500-Hz loop where it can starve the LowState reader.
            crc = int(self._vendor_crc.Crc(self._message))
        self._message.crc = crc

    def _validate_crc_backend_once(self) -> None:
        if self._crc_parity_preflight_done or self._vendor_crc is None:
            return
        pure = int(self._pure_crc.Crc(self._message))
        vendor = int(self._vendor_crc.Crc(self._message))
        if pure != vendor:
            raise TransportError("vendor/pure LowCmd CRC parity preflight failure")
        self._crc_parity_preflight_done = True
        print("[G1-29][transport] LowCmd CRC parity preflight=PASS; runtime backend=unitree_sdk2py", flush=True)

    def _writer_lowstate_raw(self) -> np.ndarray:
        """Read the minimal latest-LowState subset HOMIE uses while writing.

        Full state decoding and deployment safety checks run in the isolated
        50-Hz policy process.  Match HOMIE's writer exactly here: consume the
        newest available raw DDS state without imposing a second timestamp-age
        watchdog.  The writer still requires a received packet, the pinned
        mode, all 35 slots, and finite reserved-HG positions.
        """

        raw, _stamp, _sequence = self.state_source.writer_raw_snapshot()
        field = UnitreeDDSStateSource._field
        mode_machine = int(field(getattr(raw, "mode_machine")))
        if mode_machine != self.expected_mode_machine:
            raise TransportError("raw LowState mode does not match expected_mode_machine")
        motors = field(getattr(raw, "motor_state"))
        if len(motors) != 35:
            raise TransportError("raw HG LowState must expose exactly 35 motor slots")
        for offset, index in enumerate(range(29, 35)):
            q = float(field(getattr(motors[index], "q")))
            if not np.isfinite(q):
                raise TransportError("raw LowState reserved HG position is non-finite")
            self._writer_reserved_q_raw[offset] = q
        return self._writer_reserved_q_raw

    def _write_vectors(
        self,
        reserved_q_raw: np.ndarray,
        q_target_sdk: np.ndarray,
        kp_sdk: np.ndarray,
        kd_sdk: np.ndarray,
        tau_ff_sdk: np.ndarray,
    ) -> None:
        if reserved_q_raw.shape != (6,) or not np.all(np.isfinite(reserved_q_raw)):
            raise TransportError("raw LowState lacks six finite reserved HG positions")
        message = self._message
        message.mode_pr = 0
        message.mode_machine = self.expected_mode_machine
        if hasattr(message, "reserve"):
            for index in range(len(message.reserve)):
                message.reserve[index] = 0
        np.add(q_target_sdk, self.calibration.encoder_zero_offsets_rad, out=self._writer_q_raw_target)
        for index, motor in enumerate(message.motor_cmd):
            if index < 29:
                motor.mode = 1
                motor.q = float(self._writer_q_raw_target[index])
                motor.dq = 0.0
                motor.tau = float(tau_ff_sdk[index])
                motor.kp = float(kp_sdk[index])
                motor.kd = float(kd_sdk[index])
            else:
                motor.mode = 0
                motor.q = float(reserved_q_raw[index - 29])
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 0.0
            if hasattr(motor, "reserve"):
                motor.reserve = 0
        self._seal_crc()
        # unitree_sdk2py and the HOMIE binding both treat a non-throwing Write
        # as success; some SDK builds return ``None`` instead of ``True``.
        self._publisher.Write(message)
        self.write_count += 1

    def _publish_disable(self) -> str | None:
        """Send HOMIE's final damping frame after an active session."""

        # Do not manufacture the process's first LowCmd during cleanup.  A
        # damping frame is required only after this writer has driven once.
        if self._message is None or self.write_count <= 0 or self._disable_published:
            return None
        try:
            self._message.mode_pr = 0
            self._message.mode_machine = self.expected_mode_machine
            if hasattr(self._message, "reserve"):
                for index in range(len(self._message.reserve)):
                    self._message.reserve[index] = 0
            for index, motor in enumerate(self._message.motor_cmd):
                motor.mode = 1 if index < 29 else 0
                # Kp=0 makes q immaterial.  Retaining the last HG-reserved
                # values for 29..34 mirrors the ordinary writer path.
                if index < 29:
                    motor.q = 0.0
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 8.0 if index < 29 else 0.0
                if hasattr(motor, "reserve"):
                    motor.reserve = 0
            self._seal_crc()
            self._publisher.Write(self._message)
            self._disable_published = True
            return None
        except BaseException as exc:
            error = "disable publish failed: %s: %s" % (type(exc).__name__, exc)
            self.fault = (self.fault + "; " if self.fault else "") + error
            return error

    def _writer_loop(self) -> None:
        try:
            self._configure_writer_thread()
        except BaseException as exc:
            self.fault = "%s: %s" % (type(exc).__name__, exc)
            self._running = False
            return
        next_tick = time.monotonic_ns()
        while self._running:
            next_tick += self.writer_period_ns
            try:
                now = time.monotonic_ns()
                lateness_ns = now - next_tick
                if lateness_ns > self.max_writer_lateness_ns:
                    # Match HOMIE: retain the most recent valid command and
                    # resynchronise the next period rather than turning one
                    # Linux scheduling slip into a controller shutdown.
                    last = int(getattr(self, "_last_writer_lateness_report_ns", 0))
                    if now - last >= 1_000_000_000:
                        print(
                            "[G1-29][transport] LowCmd writer late (lateness_ms=%.3f work_ms=%.3f); continuing with latest frame"
                            % (lateness_ns / 1.0e6, int(getattr(self, "_last_writer_work_ns", 0)) / 1.0e6),
                            flush=True,
                        )
                        self._last_writer_lateness_report_ns = now
                if (
                    self.max_active_duration_ns is not None
                    and now - self._enabled_ns > self.max_active_duration_ns
                ):
                    raise TransportError("bounded physical test duration elapsed")
                with self._frame_lock:
                    frame = self._frame
                if frame is None:
                    raise TransportError("policy command frame watchdog fired")
                self._refresh_writer_lower_frame(frame)
                validation_now = time.monotonic_ns()
                if validation_now - frame.monotonic_ns > self.max_frame_age_ns:
                    last = int(getattr(self, "_last_frame_age_report_ns", 0))
                    if validation_now - last >= 1_000_000_000:
                        print(
                            "[G1-29][transport] policy frame age_ms=%.3f; holding latest frame"
                            % ((validation_now - frame.monotonic_ns) / 1.0e6),
                            flush=True,
                        )
                        self._last_frame_age_report_ns = validation_now
                # Deliberately match HOMIE's high-rate division of work:
                # the isolated 50-Hz policy process owns full CRC/decode and
                # state/target/PD safety checks, while this writer reads only
                # the raw heartbeat/mode/reserved HG slots it needs to emit a
                # valid LowCmd.  Repeating every vector safety operation here
                # costs more than the 4-ms writer period on this G1.
                reserved_q_raw = self._writer_lowstate_raw()
                state_finished = time.monotonic_ns()
                arm_started = time.monotonic_ns()
                if self._high_rate_arm_enabled:
                    self._merge_homie_arm_into_writer_buffers(now_ns=validation_now)
                arm_finished = time.monotonic_ns()
                self._write_vectors(
                    reserved_q_raw,
                    self._writer_q_sdk,
                    self._writer_kp_sdk,
                    self._writer_kd_sdk,
                    self._writer_tau_ff_sdk,
                )
                write_finished = time.monotonic_ns()
                self._last_writer_state_ns = state_finished - now
                self._last_writer_safety_ns = 0
                self._last_writer_arm_ns = arm_finished - arm_started
                self._last_writer_projection_ns = 0
                self._last_writer_validation_ns = 0
                self._last_writer_write_ns = write_finished - arm_finished
                self._last_writer_work_ns = write_finished - now
            except BaseException as exc:
                self.fault = "%s: %s" % (type(exc).__name__, exc)
                self._running = False
                self._publish_disable()
                break
            remaining = next_tick - time.monotonic_ns()
            if remaining > self.writer_spin_ns:
                time.sleep((remaining - self.writer_spin_ns) / 1.0e9)
            elif remaining <= -self.writer_period_ns:
                # Copy HOMIE's writer behavior: one late DDS publish must
                # never create a catch-up burst that accumulates deadline
                # debt over subsequent 4-ms periods.  A genuine scheduler
                # stall is still caught by the independent 16-ms envelope at
                # the start of a later loop.
                next_tick = time.monotonic_ns()
            while self._running and remaining > 0 and time.monotonic_ns() < next_tick:
                # Keep the final short wakeup window on the dedicated writer
                # CPU.  This avoids normal scheduler wake jitter at 500 Hz
                # without turning an entire core into a permanent busy loop.
                pass

    def _configure_writer_thread(self) -> None:
        """Apply writer-only scheduling isolation from inside the writer TID."""

        rt_priority = int(getattr(self, "writer_rt_priority", 0))
        if self.writer_cpu is not None:
            try:
                # Linux affinity is per thread.  A launch-level ``taskset -c
                # 2`` therefore leaves the policy on CPU 2 while this writer
                # thread can safely move itself to CPU 3.
                os.sched_setaffinity(0, {self.writer_cpu})
            except OSError as exc:
                raise TransportError(
                    "cannot pin LowCmd writer to CPU %d" % self.writer_cpu
                ) from exc
        if rt_priority:
            try:
                os.sched_setscheduler(
                    0,
                    os.SCHED_FIFO,
                    os.sched_param(rt_priority),
                )
            except (AttributeError, OSError) as exc:
                raise TransportError(
                    "cannot set LowCmd writer SCHED_FIFO priority %d; run real mode through sudo -E"
                    % rt_priority
                ) from exc
        if self.writer_cpu is not None or self.writer_spin_ns or rt_priority:
            affinity = sorted(os.sched_getaffinity(0))
            print(
                "[G1-29][transport] LowCmd writer rate_hz=%g affinity=%s spin_us=%d rt_priority=%d lateness_limit_ms=%.3f"
                % (
                    self.writer_rate_hz,
                    affinity,
                    self.writer_spin_ns // 1_000,
                    rt_priority,
                    self.max_writer_lateness_ns / 1.0e6,
                ),
                flush=True,
            )

    def close(self) -> None:
        # Parent teardown can be reached from a policy failure and then again
        # from process cleanup.  It is idempotent and never issues a second
        # driving command.
        if self._closed:
            return
        self._closed = True
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                self.fault = self.fault or "LowCmd writer thread did not stop; ownership lease retained"
                raise TransportError(self.fault)
        self._thread = None
        disable_error = self._publish_disable()
        close = getattr(self._publisher, "Close", None)
        publisher_error = None
        try:
            if callable(close):
                close()
        except BaseException as exc:
            # unitree_sdk2py's Python writer wrapper can release its private
            # ``__writer`` during process teardown before its public Close()
            # method is called.  The writer thread is already stopped and the
            # disable frame was attempted above, so this vendor cleanup-only
            # AttributeError has no remaining motion authority to revoke.
            # Do not let it replace the primary handoff/safety fault.
            if isinstance(exc, AttributeError) and "_Writer__writer" in str(exc):
                print(
                    "[G1-29][transport] ignoring known unitree_sdk2py publisher Close() teardown bug: %s"
                    % exc,
                    flush=True,
                )
            else:
                publisher_error = "publisher close failed: %s: %s" % (type(exc).__name__, exc)
        finally:
            # The writer thread is proven stopped above.  Releasing the lease
            # here cannot create another write, and prevents a failed DDS
            # Close() from leaving a stale local lock until process teardown.
            self._lease.close()
            if self._high_rate_arm_source is not None:
                self._high_rate_arm_source.close()
        errors = [value for value in (disable_error, publisher_error) if value]
        if errors:
            raise TransportError("; ".join(errors))
