"""G1 low-level DDS interface, pure `unitree_sdk2py` (no `unitree_interface`).

Talks rt/lowcmd + rt/lowstate directly over DDS via the same binding xr_teleoperate and
`arm_target_source.ArmSdkArmTargetSource` use. Using ONE DDS binding for the whole process
(state read/write AND the `rt/arm_sdk` arm-target subscription) avoids the crash risk of
loading two different native `libddsc.so` builds (`unitree_interface` + `unitree_sdk2py`)
in one process -- confirmed by hand: initializing a second, differently-built DDS
participant on the same domain after `unitree_interface` already opened one fails outright
("[ChannelFactory] create domain error"). `ChannelFactoryInitialize` itself is an
idempotent per-process singleton, so `arm_target_source.py` calling it again is a no-op.

HOMIE PD gains and default pose come from the deploy Config.
"""
import struct
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from cpu_affinity import pin_current_thread

from dreamwaq_interface.g1_29dof.contract import (
    JOINT_LOWER_SDK,
    JOINT_UPPER_SDK,
    TORQUE_LIMIT_SDK,
)

ControllerMapping = {
    "A": 0x0100, "B": 0x0200, "X": 0x0400, "Y": 0x0800,
    "R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008,
    "R2": 0x0010, "L2": 0x0020, "F1": 0x0040, "F2": 0x0080,
    "up": 0x1000, "right": 0x2000, "down": 0x4000, "left": 0x8000,
}

LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
NUM_HG_MOTORS = 35  # HG LowCmd/LowState always carry 35 motor slots


class _WirelessController:
    __slots__ = ("keys", "lx", "ly", "rx", "ry")

    def __init__(self, keys, lx, ly, rx, ry):
        self.keys = keys
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry


def _decode_wireless_remote(payload):
    """Unitree wireless_remote layout (40 bytes): 2 header bytes, keys uint16 at
    [2:4], then float32 stick axes lx[4:8], rx[8:12], ry[12:16], L2[16:20], ly[20:24].
    """
    if not payload or len(payload) < 24:
        return _WirelessController(keys=0, lx=0.0, ly=0.0, rx=0.0, ry=0.0)
    payload = bytes(payload)
    keys = struct.unpack("<H", payload[2:4])[0]
    lx, rx, ry = struct.unpack("<fff", payload[4:16])
    ly = struct.unpack("<f", payload[20:24])[0]
    return _WirelessController(keys=keys, lx=lx, ly=ly, rx=rx, ry=ry)


class G1RealWorldEnv:
    def __init__(self, net, config, connect_timeout=5.0, domain_id=0,
                 lowcmd_topic=LOWCMD_TOPIC, lowstate_topic=LOWSTATE_TOPIC,
                 lowstate_reader_hz=250.0, expected_mode_machine=5,
                 lowstate_max_age_s=0.040,
                 clip_lower_joint_targets=True,
                 torque_limit_sdk=None, reader_cpu=None, writer_cpu=None,
                 lowcmd_enabled=True):
        """`domain_id`/`lowcmd_topic`/`lowstate_topic` default to the production DDS
        domain (0) and topics (`rt/lowcmd`/`rt/lowstate`) -- override ONLY for test
        isolation (e.g. a non-production domain id and a `test/...` topic prefix),
        never in normal deploy use."""
        self.config = config
        self.lowcmd_topic = str(lowcmd_topic)
        self.lowstate_topic = str(lowstate_topic)
        lowstate_reader_hz = float(lowstate_reader_hz)
        if not np.isfinite(lowstate_reader_hz) or lowstate_reader_hz <= 0.0:
            raise ValueError("lowstate_reader_hz must be finite and > 0")
        self._reader_period = 1.0 / lowstate_reader_hz
        self._reader_cpu = None if reader_cpu is None else int(reader_cpu)
        self._reader_cpu_affinity = None
        self._writer_cpu = None if writer_cpu is None else int(writer_cpu)
        self._writer_cpu_affinity = None
        self._writer_ready = None

        ChannelFactoryInitialize(domain_id, net)
        self._crc = CRC()

        self._latest_state = None
        self._latest_state_monotonic_ns = 0
        self._latest_state_tick = None
        self._latest_state_sequence = 0
        self._state_duplicate_count = 0
        self._state_replay_count = 0
        self._reader_fault = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._writer_thread = None
        self._final_damping_thread = None
        self._lowcmd_enabled = bool(lowcmd_enabled)
        self._writer_fault = None
        self._writer_damping_fault = None
        self._writer_on_fault = None
        self._writer_publish_count = 0
        self._writer_last_publish_ns = 0
        self._writer_achieved_hz = 0.0
        self._writer_overruns = 0
        self._writer_get_ms_max = 0.0
        self._writer_send_ms_max = 0.0
        self.expected_mode_machine = int(expected_mode_machine)
        self.clip_lower_joint_targets = bool(clip_lower_joint_targets)
        self.lowstate_max_age_s = float(lowstate_max_age_s)
        if not np.isfinite(self.lowstate_max_age_s) or self.lowstate_max_age_s <= 0.0:
            raise ValueError("lowstate_max_age_s must be finite and > 0")
        if torque_limit_sdk is None:
            torque_limit_sdk = TORQUE_LIMIT_SDK
        self.torque_limit_sdk = np.asarray(torque_limit_sdk, dtype=np.float32).reshape(-1).copy()
        if (
            self.torque_limit_sdk.shape != (29,)
            or not np.all(np.isfinite(self.torque_limit_sdk))
            or np.any(self.torque_limit_sdk <= 0.0)
        ):
            raise ValueError("torque_limit_sdk must contain 29 finite positive values")

        self._lowstate_sub = ChannelSubscriber(self.lowstate_topic, LowState_)
        self._lowstate_sub.Init()
        self._lowcmd_pub = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self._lowcmd_pub.Init()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="g1-lowstate-reader", daemon=True
        )
        self._reader_thread.start()

        t0 = time.monotonic()
        while self._latest_state is None:
            if self._reader_fault is not None:
                raise RuntimeError("LowState reader faulted: %s" % self._reader_fault)
            if time.monotonic() - t0 > connect_timeout:
                raise RuntimeError(
                    f"no '{self.lowstate_topic}' sample received within {connect_timeout}s "
                    f"(net={net}) -- is the robot powered on and reachable?"
                )
            time.sleep(0.01)

        n = config.num_actions
        self.qj = np.zeros(n, dtype=np.float32)
        self.dqj = np.zeros(n, dtype=np.float32)
        self.tau_est = np.zeros(n, dtype=np.float32)
        self.counter = 0

        self.controller_mapping = ControllerMapping
        print("[green]Successfully connected to the robot[/green]")

    def _reader_loop(self):
        try:
            self._reader_cpu_affinity = pin_current_thread(self._reader_cpu)
        except Exception as exc:
            self._reader_fault = RuntimeError(
                "could not pin LowState reader to CPU %s: %s"
                % (self._reader_cpu, exc)
            )
            self._stop.set()
            return
        # The robot publishes lowstate at roughly 1 kHz, but Homie's command writer
        # and policy only need the latest sample.  Reading/deserializing every sample
        # needlessly monopolizes the Python GIL, so sample the latest state at 250 Hz.
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            msg = self._lowstate_sub.Read()
            if msg is not None:
                try:
                    raw_tick = getattr(msg, "tick")
                    tick = int(raw_tick() if callable(raw_tick) else raw_tick)
                except Exception as exc:
                    self._reader_fault = RuntimeError(
                        "LowState tick is unavailable: %s" % exc
                    )
                    self._stop.set()
                    continue
                with self._state_lock:
                    if self._latest_state_tick is not None:
                        delta = (tick - self._latest_state_tick) & 0xFFFFFFFF
                        if delta == 0:
                            self._state_duplicate_count += 1
                            msg = None
                        elif delta >= 0x80000000:
                            self._state_replay_count += 1
                            self._reader_fault = RuntimeError(
                                "LowState tick replayed (%d -> %d)"
                                % (self._latest_state_tick, tick)
                            )
                            msg = None
                    if msg is not None:
                        stamp_ns = time.monotonic_ns()
                        self._latest_state = msg
                        self._latest_state_tick = tick
                        self._latest_state_sequence += 1
                        self._latest_state_monotonic_ns = stamp_ns

            next_deadline += self._reader_period
            wait_s = next_deadline - time.monotonic()
            if wait_s > 0.0:
                self._stop.wait(wait_s)
            else:
                # Do not burst through missed reads after a scheduler stall.
                next_deadline = time.monotonic()

    # --- reads -------------------------------------------------------------
    def read_robot_state(self):
        with self._state_lock:
            return self._latest_state

    def read_controller_input(self):
        state = self.read_robot_state()
        return _decode_wireless_remote(state.wireless_remote)

    def get_robot_state(self):
        """Return (dof_pos, dof_vel, base_quat[wxyz], base_ang_vel)."""
        return self.get_robot_state_timed()[:4]

    def get_robot_state_timed(self):
        """Return robot state plus its local monotonic receive timestamp.

        The timestamp lets the policy parent pair a measured state with the
        last LowCmd target that had actually been published before that sample
        arrived, instead of the actor's most recently computed output.
        """
        self.counter += 1
        with self._state_lock:
            low_state = self._latest_state
            state_monotonic_ns = int(self._latest_state_monotonic_ns)
            for i in range(len(self.config.joint2motor_idx)):
                m = self.config.joint2motor_idx[i]
                self.qj[i] = low_state.motor_state[m].q
                self.dqj[i] = low_state.motor_state[m].dq
                self.tau_est[i] = low_state.motor_state[m].tau_est
            quat = np.array(low_state.imu_state.quaternion, dtype=np.float32)
            ang_vel = np.array(low_state.imu_state.gyroscope, dtype=np.float32)
        return self.qj.copy(), self.dqj.copy(), quat, ang_vel, state_monotonic_ns

    # --- writes ------------------------------------------------------------
    def _apply_joint_target_limits(self, q_target):
        """Keep non-policy joints bounded; optionally leave the 12 leg targets raw."""

        target = np.asarray(q_target, dtype=np.float32).reshape(self.config.num_actions).copy()
        first_clipped_index = 0 if self.clip_lower_joint_targets else 12
        np.clip(
            target[first_clipped_index:],
            JOINT_LOWER_SDK[first_clipped_index:self.config.num_actions],
            JOINT_UPPER_SDK[first_clipped_index:self.config.num_actions],
            out=target[first_clipped_index:],
        )
        return target

    def send_cmd(self, q_target, kp, kd, dq_target=None, tau_ff=None,
                 allow_stale_state=False):
        """Build and publish a 35-slot HG LowCmd from 29-dim targets/gains.

        The 6 reserved HG slots (29..34) are not policy-controlled; pass their
        measured position through with mode=0 so they are reported but never driven.
        """
        if not self._lowcmd_enabled:
            raise RuntimeError("DreamWaQ LowCmd is disabled while Unitree owns control")
        n = self.config.num_actions
        q_target = np.asarray(q_target, dtype=np.float32).reshape(n)
        kp = np.asarray(kp, dtype=np.float32).reshape(n)
        kd = np.asarray(kd, dtype=np.float32).reshape(n)
        dq_target = np.zeros(n, dtype=np.float32) if dq_target is None else np.asarray(dq_target, dtype=np.float32).reshape(n)
        tau_ff = np.zeros(n, dtype=np.float32) if tau_ff is None else np.asarray(tau_ff, dtype=np.float32).reshape(n)
        if not all(np.all(np.isfinite(value)) for value in (q_target, kp, kd, dq_target, tau_ff)):
            raise ValueError("LowCmd vectors must contain only finite values")
        kp = np.maximum(kp, 0.0)
        kd = np.maximum(kd, 0.0)

        with self._state_lock:
            low_state = self._latest_state
            state_received_ns = int(self._latest_state_monotonic_ns)
            reader_fault = self._reader_fault
        if reader_fault is not None and not allow_stale_state:
            raise RuntimeError("LowState reader faulted: %s" % reader_fault)
        state_age_ns = time.monotonic_ns() - state_received_ns
        if state_age_ns < 0:
            raise RuntimeError(
                "raw LowState timestamp is in the future (age_ms=%.3f)"
                % (state_age_ns / 1.0e6)
            )
        # A stale LowState (communication delay) is tolerated: keep commanding
        # with the newest cached state instead of faulting the writer.
        if int(low_state.mode_machine) != self.expected_mode_machine:
            raise RuntimeError(
                "LowState mode_machine=%s, expected=%s"
                % (int(low_state.mode_machine), self.expected_mode_machine)
            )

        measured_q = np.asarray(
            [low_state.motor_state[m].q for m in self.config.joint2motor_idx],
            dtype=np.float32,
        )
        measured_dq = np.asarray(
            [low_state.motor_state[m].dq for m in self.config.joint2motor_idx],
            dtype=np.float32,
        )
        q_target = self._apply_joint_target_limits(q_target)
        torque_limit = self.torque_limit_sdk[:n]
        tau_ff = np.clip(tau_ff, -torque_limit, torque_limit)
        # Convert the total PD effort envelope into an admissible q-target
        # interval. This preserves the absolute-position interface while
        # preventing a large measured-pose error from becoming a destructive
        # instantaneous torque request.
        positive_error = (torque_limit - tau_ff + kd * measured_dq) / np.maximum(kp, 1.0e-6)
        negative_error = (-torque_limit - tau_ff + kd * measured_dq) / np.maximum(kp, 1.0e-6)
        active_kp = kp > 1.0e-6
        q_target[active_kp] = np.clip(
            q_target[active_kp],
            measured_q[active_kp] + negative_error[active_kp],
            measured_q[active_kp] + positive_error[active_kp],
        )
        q_target = self._apply_joint_target_limits(q_target)
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = low_state.mode_machine
        for i in range(NUM_HG_MOTORS):
            motor = msg.motor_cmd[i]
            if i < n:
                motor.mode = 1
                motor.q = float(q_target[i])
                motor.dq = float(dq_target[i])
                motor.tau = float(tau_ff[i])
                motor.kp = float(kp[i])
                motor.kd = float(kd[i])
            else:
                motor.mode = 0
                motor.q = float(low_state.motor_state[i].q)
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 0.0
        msg.crc = self._crc.Crc(msg)
        self._lowcmd_pub.Write(msg)
        return q_target.copy()

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0, kp=None, kd=None,
                           tau_ff=None):
        """`kp`/`kd`, if given, are full per-joint gain arrays (e.g. teleop-supplied arm
        gains blended with the HOMIE leg/waist gains) and are used as-is, ignoring
        kp_scale/kd_scale. Otherwise falls back to config.kps/kds * scale (unchanged
        default behavior, used by bring-up and sim2sim).

        `tau_ff`, if given, is a full 29-dim feedforward torque array (e.g. teleop's arm
        gravity-comp forwarded by arm_target_source.py, legs/waist always zero). Omitted
        -> send_cmd's own default of all-zero tau_ff, unchanged from before."""
        if kp is None:
            kp = [self.config.kps[i] * kp_scale for i in range(len(self.config.kps))]
        if kd is None:
            kd = [self.config.kds[i] * kd_scale for i in range(len(self.config.kds))]
        return self.send_cmd(target_dof_pos, kp, kd, tau_ff=tau_ff)

    def send_damping(self, kd=8.0, *, allow_stale_state=False):
        """Zero-stiffness damping hold -- the safe fallback."""
        n = self.config.num_actions
        self.send_cmd(
            np.zeros(n, dtype=np.float32),
            [0.0] * n,
            [kd] * n,
            allow_stale_state=allow_stale_state,
        )

    # --- decoupled high-rate command writer ---------------------------------
    def start_writer(self, get_command, rate_hz=250.0, on_published=None,
                     on_fault=None):
        """Publish `rt/lowcmd` on a dedicated thread at `rate_hz`, decoupled from the
        caller's own control loop.

        `get_command()` is called once per writer cycle and must return either
        `(q_target29, kp29, kd29, tau_ff29, metadata)`, the older four-tuple, or
        `(q_target29, kp29, kd29)` (backward compatible: tau_ff is then treated as
        all-zero). ``on_published(actual_q_target, metadata, monotonic_ns)`` runs only
        after a successful DDS write and can acknowledge the exact command to its
        producer. This exists so a
        slow, fixed-rate decision loop (e.g. a 50 Hz RL policy for legs/waist) can still
        resample a fast-changing target (e.g. teleop arm commands arriving at ~250 Hz
        over `rt/arm_sdk`) every write instead of only once per policy tick -- holding an
        external command to the policy's own rate would otherwise throttle it far below
        what its own source publishes at.
        """
        if not self._lowcmd_enabled:
            raise RuntimeError("cannot start LowCmd writer without low-level ownership")
        if self._writer_thread is not None:
            raise RuntimeError("writer is already running")
        rate_hz = float(rate_hz)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and > 0")
        self._writer_get_command = get_command
        self._writer_on_published = on_published
        self._writer_on_fault = on_fault
        self._writer_fault = None
        self._writer_damping_fault = None
        self._writer_publish_count = 0
        self._writer_last_publish_ns = 0
        self._writer_period = 1.0 / rate_hz
        self._writer_stop = threading.Event()
        self._writer_ready = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="g1-lowcmd-writer", daemon=True
        )
        self._writer_thread.start()
        if not self._writer_ready.wait(timeout=1.0):
            self._writer_stop.set()
            raise RuntimeError("LowCmd writer did not initialize its CPU affinity")
        if self._writer_fault is not None:
            raise RuntimeError("LowCmd writer CPU affinity failed: %s" % self._writer_fault)

    def _writer_loop(self):
        """Diagnostic counters (see also arm_target_source.py's own [arm] report):
        this prints achieved write Hz, per-iteration cost, and how often a cycle
        overran its target period -- i.e. whether stutter/lag is coming from THIS
        process (slow callback/DDS write, GIL contention) rather than upstream
        (teleop's own publish cadence, network). Report every ~1s.
        """
        try:
            self._writer_cpu_affinity = pin_current_thread(self._writer_cpu)
        except Exception as exc:
            self._writer_fault = RuntimeError(
                "could not pin LowCmd writer to CPU %s: %s"
                % (self._writer_cpu, exc)
            )
            self._writer_ready.set()
            return
        self._writer_ready.set()
        n = 0
        overruns = 0
        get_ms_sum = get_ms_max = 0.0
        send_ms_sum = send_ms_max = 0.0
        report_t0 = time.monotonic()
        next_deadline = time.monotonic()
        while not self._writer_stop.is_set():
            t0 = time.monotonic()
            try:
                result = self._writer_get_command()
                metadata = None
                if len(result) == 5:
                    q_target, kp, kd, tau_ff, metadata = result
                elif len(result) == 4:
                    q_target, kp, kd, tau_ff = result
                else:
                    q_target, kp, kd = result
                    tau_ff = None
                t_got = time.monotonic()
                if q_target is not None:
                    published_target = self.send_robot_action(
                        q_target, kp=kp, kd=kd, tau_ff=tau_ff
                    )
                    published_ns = time.monotonic_ns()
                    self._writer_publish_count += 1
                    self._writer_last_publish_ns = published_ns
                    if self._writer_on_published is not None:
                        self._writer_on_published(
                            published_target, metadata, published_ns
                        )
                t_sent = time.monotonic()
            except Exception as exc:
                # Publish the fault before doing anything that could block.  In
                # particular, never print from this 250-Hz safety path: a
                # back-pressured SSH terminal can hold TextIOWrapper's lock and
                # prevent both the fault transition and emergency damping.
                self._writer_fault = exc
                self._notify_writer_fault("LowCmd writer faulted: %s" % exc)
                self._damping_writer_loop()
                return
            get_ms = (t_got - t0) * 1000.0
            send_ms = (t_sent - t_got) * 1000.0
            n += 1
            get_ms_sum += get_ms
            get_ms_max = max(get_ms_max, get_ms)
            send_ms_sum += send_ms
            send_ms_max = max(send_ms_max, send_ms)
            # Anchor every cycle to an absolute monotonic deadline.  This corrects
            # normal sleep/wakeup jitter instead of accumulating it every iteration.
            next_deadline += self._writer_period
            wait_s = next_deadline - time.monotonic()
            if wait_s > 0.0:
                self._writer_stop.wait(wait_s)
            else:
                overruns += 1  # callback+write/wakeup missed the target deadline
                if wait_s <= -self._writer_period:
                    # A long scheduler stall must not trigger a catch-up burst.
                    next_deadline = time.monotonic()

            if t_sent - report_t0 >= 1.0:
                achieved_hz = n / (t_sent - report_t0)
                target_hz = 1.0 / self._writer_period
                self._writer_achieved_hz = achieved_hz
                self._writer_overruns = overruns
                self._writer_get_ms_max = get_ms_max
                self._writer_send_ms_max = send_ms_max
                # Periodic writer-performance logging disabled to keep the console quiet.
                # print(f"[writer] {achieved_hz:5.0f} Hz (target {target_hz:.0f})  "
                #       f"get_ms avg={get_ms_sum / n:.2f} max={get_ms_max:.2f}  "
                #       f"send_ms avg={send_ms_sum / n:.2f} max={send_ms_max:.2f}  "
                #       f"overruns={overruns}/{n}")
                n = 0
                overruns = 0
                get_ms_sum = get_ms_max = 0.0
                send_ms_sum = send_ms_max = 0.0
                report_t0 = t_sent

    def _damping_writer_loop(self):
        """Keep publishing damping if the command producer or writer faults.

        The policy parent normally observes ``writer_fault`` on its next 50-Hz
        tick and closes the environment. This loop is the independent fallback
        for a blocked or stalled parent process.
        """

        next_deadline = time.monotonic()
        while not self._writer_stop.is_set():
            try:
                # A stale LowState is itself one reason this path is entered.
                # Damping has Kp=0 and does not use a position target, so allow
                # the latest cached mode/state to form this fallback command.
                self.send_damping(allow_stale_state=True)
            except Exception as exc:
                self._writer_damping_fault = exc
                self._notify_writer_fault(
                    "emergency damping write failed: %s" % exc
                )
                return
            next_deadline += self._writer_period
            wait_s = next_deadline - time.monotonic()
            if wait_s > 0.0:
                self._writer_stop.wait(wait_s)
            else:
                next_deadline = time.monotonic()

    def _notify_writer_fault(self, message):
        callback = self._writer_on_fault
        if callback is None:
            return
        try:
            callback(str(message))
        except Exception:
            # Diagnostics can never become part of the command path.
            pass

    def stop_writer(self, timeout_s=1.0):
        if self._writer_thread is None:
            return True
        thread = self._writer_thread
        self._writer_stop.set()
        thread.join(timeout=max(0.0, float(timeout_s)))
        if thread.is_alive():
            fault = RuntimeError(
                "LowCmd writer did not stop within %.3f s" % float(timeout_s)
            )
            if self._writer_fault is None:
                self._writer_fault = fault
            self._notify_writer_fault(str(fault))
            return False
        self._writer_thread = None
        return True

    def _send_damping_bounded(self, timeout_s=0.25):
        """Attempt the final damping write without trapping shutdown in DDS."""

        if self._final_damping_thread is not None and self._final_damping_thread.is_alive():
            return False
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        done = threading.Event()
        result = {"error": None}

        def write_once():
            try:
                self.send_damping(allow_stale_state=True)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=write_once,
            name="g1-final-damping-writer",
            daemon=True,
        )
        self._final_damping_thread = thread
        thread.start()
        if not done.wait(max(0.0, float(timeout_s))):
            self._notify_writer_fault(
                "final damping write did not finish within %.3f s"
                % float(timeout_s)
            )
            return False
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            return False
        if result["error"] is not None:
            self._notify_writer_fault(
                "final damping write failed: %s" % result["error"]
            )
            return False
        return True

    @property
    def lowcmd_enabled(self):
        return self._lowcmd_enabled

    def require_fresh_state(self):
        """Fresh feedback is required when acquiring control after an RPC handoff."""
        diag = self.state_reader_diagnostics()
        if self.reader_fault is not None:
            raise RuntimeError("LowState reader faulted: %s" % self.reader_fault)
        if diag["state_age_ms"] > self.lowstate_max_age_s * 1000.0:
            raise RuntimeError("cannot acquire control with stale LowState")
        if diag["mode_machine"] != self.expected_mode_machine:
            raise RuntimeError("cannot acquire control: unexpected G1 hardware type")

    def set_lowcmd_enabled(self, enabled):
        if self._writer_thread is not None and self._writer_thread.is_alive():
            raise RuntimeError("stop the LowCmd writer before changing ownership")
        if self._final_damping_thread is not None and self._final_damping_thread.is_alive():
            raise RuntimeError("damping Write is still in flight; ownership cannot change")
        self._lowcmd_enabled = bool(enabled)

    def suspend_lowcmd(self):
        """Drain all writes before allowing the native controller to start."""
        if not self.stop_writer():
            raise RuntimeError("LowCmd writer is still alive; Unitree handoff cancelled")
        if self._lowcmd_enabled and not self._send_damping_bounded():
            raise RuntimeError("final damping did not complete; Unitree handoff cancelled")
        self.set_lowcmd_enabled(False)

    @property
    def writer_fault(self):
        return self._writer_fault

    @property
    def reader_fault(self):
        return self._reader_fault

    def state_reader_diagnostics(self):
        with self._state_lock:
            low_state = self._latest_state
            received_ns = int(self._latest_state_monotonic_ns)
            mode_machine = (
                -1 if low_state is None else int(low_state.mode_machine)
            )
            nonzero_motorstate = []
            max_temperature = None
            if low_state is not None:
                temperatures = []
                for index in self.config.joint2motor_idx:
                    motor = low_state.motor_state[index]
                    motorstate = int(motor.motorstate)
                    if motorstate != 0:
                        nonzero_motorstate.append((int(index), motorstate))
                    temperatures.extend(int(value) for value in motor.temperature)
                if temperatures:
                    max_temperature = max(temperatures)
            return {
                "source_tick": -1 if self._latest_state_tick is None else int(self._latest_state_tick),
                "source_sequence": int(self._latest_state_sequence),
                "duplicate_reads": int(self._state_duplicate_count),
                "replay_reads": int(self._state_replay_count),
                "received_monotonic_ns": received_ns,
                "state_age_ms": (
                    float("inf")
                    if received_ns <= 0
                    else max(0.0, (time.monotonic_ns() - received_ns) / 1.0e6)
                ),
                "mode_machine": mode_machine,
                "nonzero_motorstate": tuple(nonzero_motorstate),
                "max_temperature": max_temperature,
                "reader_cpu_affinity": self._reader_cpu_affinity,
            }

    def writer_diagnostics(self):
        last_publish_ns = int(self._writer_last_publish_ns)
        return {
            "publish_count": int(self._writer_publish_count),
            "last_publish_age_ms": (
                float("inf")
                if last_publish_ns <= 0
                else max(0.0, (time.monotonic_ns() - last_publish_ns) / 1.0e6)
            ),
            "achieved_hz": float(self._writer_achieved_hz),
            "overruns": int(self._writer_overruns),
            "get_ms_max": float(self._writer_get_ms_max),
            "send_ms_max": float(self._writer_send_ms_max),
            "cpu_affinity": self._writer_cpu_affinity,
            "fault": None if self._writer_fault is None else str(self._writer_fault),
            "damping_fault": (
                None
                if self._writer_damping_fault is None
                else str(self._writer_damping_fault)
            ),
        }

    # --- bring-up ----------------------------------------------------------
    def move_to_default_pos(self, total_time=2.0, notify=print, cancel=lambda: None):
        notify("[green]Press START on the remote to ramp to the default pose...[/green]")
        while self.read_controller_input().keys != self.controller_mapping["start"]:
            cancel()
            time.sleep(self.config.control_dt)
        self.ramp_to_default_pos(total_time=total_time, notify=notify, cancel=cancel)

    def ramp_to_default_pos(self, total_time=2.0, notify=print, cancel=lambda: None):
        """Prepare the pose again after handoff without waiting for START/A."""
        notify("[green]Ramping to default pose.[/green]")

        total_time = float(total_time)
        if not np.isfinite(total_time) or total_time <= 0.0:
            raise ValueError("default-pose ramp duration must be finite and > 0")
        num_step = max(1, int(total_time / self.config.control_dt))
        q0 = self.get_robot_state()[0]
        q_tgt = self.config.default_angles.copy()
        for i in range(1, num_step + 1):
            cancel()
            alpha = i / num_step
            self.send_robot_action(q0 * (1 - alpha) + q_tgt * alpha)
            time.sleep(self.config.control_dt)

    def default_pos_state(self, notify=print, cancel=lambda: None):
        notify("Holding default pose. Press A to start the policy...")
        while self.read_controller_input().keys != self.controller_mapping["A"]:
            cancel()
            self.send_robot_action(self.config.default_angles.copy())
            time.sleep(self.config.control_dt)

    def close(self):
        writer_stopped = self.stop_writer()
        # Never issue a concurrent Write() through the same publisher when the
        # original writer is stuck in DDS.  The firmware command timeout is the
        # bounded fallback in that exceptional case.
        damping_sent = False
        if writer_stopped and self._lowcmd_enabled:
            damping_sent = self._send_damping_bounded()
        elif not writer_stopped:
            self._notify_writer_fault(
                "final damping write skipped because LowCmd writer is still alive"
            )
        self._stop.set()
        self._reader_thread.join(timeout=1.0)
        reader_stopped = not self._reader_thread.is_alive()
        if not reader_stopped:
            self._notify_writer_fault("LowState reader did not stop within 1.000 s")
        return {
            "writer_stopped": bool(writer_stopped),
            "damping_sent": bool(damping_sent),
            "reader_stopped": bool(reader_stopped),
        }
