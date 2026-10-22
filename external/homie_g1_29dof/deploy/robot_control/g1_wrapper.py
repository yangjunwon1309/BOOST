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
from boost.io.unitree_crc import make_unitree_crc

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
                 lowstate_reader_hz=250.0):
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

        ChannelFactoryInitialize(domain_id, net)
        self._crc = make_unitree_crc()

        self._latest_state = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._writer_thread = None

        self._lowstate_sub = ChannelSubscriber(self.lowstate_topic, LowState_)
        self._lowstate_sub.Init()
        self._lowcmd_pub = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self._lowcmd_pub.Init()

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="g1-lowstate-reader", daemon=True
        )
        self._reader_thread.start()

        t0 = time.time()
        while self._latest_state is None:
            if time.time() - t0 > connect_timeout:
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
        # The robot publishes lowstate at roughly 1 kHz, but Homie's command writer
        # and policy only need the latest sample.  Reading/deserializing every sample
        # needlessly monopolizes the Python GIL, so sample the latest state at 250 Hz.
        next_deadline = time.monotonic()
        while not self._stop.is_set():
            msg = self._lowstate_sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._latest_state = msg

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
        self.counter += 1
        low_state = self.read_robot_state()
        for i in range(len(self.config.joint2motor_idx)):
            m = self.config.joint2motor_idx[i]
            self.qj[i] = low_state.motor_state[m].q
            self.dqj[i] = low_state.motor_state[m].dq
            self.tau_est[i] = low_state.motor_state[m].tau_est

        quat = np.array(low_state.imu_state.quaternion, dtype=np.float32)   # [w, x, y, z]
        ang_vel = np.array(low_state.imu_state.gyroscope, dtype=np.float32)  # rad/s
        return self.qj.copy(), self.dqj.copy(), quat, ang_vel

    # --- writes ------------------------------------------------------------
    def send_cmd(self, q_target, kp, kd, dq_target=None, tau_ff=None):
        """Build and publish a 35-slot HG LowCmd from 29-dim targets/gains.

        The 6 reserved HG slots (29..34) are not policy-controlled; pass their
        measured position through with mode=0 so they are reported but never driven.
        """
        n = self.config.num_actions
        q_target = np.asarray(q_target, dtype=np.float32)
        dq_target = np.zeros(n, dtype=np.float32) if dq_target is None else np.asarray(dq_target, dtype=np.float32)
        tau_ff = np.zeros(n, dtype=np.float32) if tau_ff is None else np.asarray(tau_ff, dtype=np.float32)

        low_state = self.read_robot_state()
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
        self.send_cmd(target_dof_pos, kp, kd, tau_ff=tau_ff)

    def send_damping(self, kd=8.0):
        """Zero-stiffness damping hold -- the safe fallback."""
        n = self.config.num_actions
        self.send_cmd(np.zeros(n, dtype=np.float32), [0.0] * n, [kd] * n)

    # --- decoupled high-rate command writer ---------------------------------
    def start_writer(self, get_command, rate_hz=250.0):
        """Publish `rt/lowcmd` on a dedicated thread at `rate_hz`, decoupled from the
        caller's own control loop.

        `get_command()` is called once per writer cycle and must return either
        `(q_target29, kp29, kd29, tau_ff29)` or the older `(q_target29, kp29, kd29)`
        (backward compatible: tau_ff is then treated as all-zero, same as before this
        was added) -- or all-`None` in either arity to skip a cycle. This exists so a
        slow, fixed-rate decision loop (e.g. a 50 Hz RL policy for legs/waist) can still
        resample a fast-changing target (e.g. teleop arm commands arriving at ~250 Hz
        over `rt/arm_sdk`) every write instead of only once per policy tick -- holding an
        external command to the policy's own rate would otherwise throttle it far below
        what its own source publishes at.
        """
        if self._writer_thread is not None:
            raise RuntimeError("writer is already running")
        rate_hz = float(rate_hz)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and > 0")
        self._writer_get_command = get_command
        self._writer_period = 1.0 / rate_hz
        self._writer_stop = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="g1-lowcmd-writer", daemon=True
        )
        self._writer_thread.start()

    def _writer_loop(self):
        """Diagnostic counters (see also arm_target_source.py's own [arm] report):
        this prints achieved write Hz, per-iteration cost, and how often a cycle
        overran its target period -- i.e. whether stutter/lag is coming from THIS
        process (slow callback/DDS write, GIL contention) rather than upstream
        (teleop's own publish cadence, network). Report every ~1s.
        """
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
                if len(result) == 4:
                    q_target, kp, kd, tau_ff = result
                else:
                    q_target, kp, kd = result
                    tau_ff = None
                t_got = time.monotonic()
                if q_target is not None:
                    self.send_robot_action(q_target, kp=kp, kd=kd, tau_ff=tau_ff)
                t_sent = time.monotonic()
            except Exception as exc:
                print(f"[writer] command callback failed: {exc}")
                t_got = t_sent = time.monotonic()
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

    def stop_writer(self):
        if self._writer_thread is None:
            return
        self._writer_stop.set()
        self._writer_thread.join(timeout=1.0)
        self._writer_thread = None

    # --- bring-up ----------------------------------------------------------
    def move_to_default_pos(self):
        print("[green]Press START on the remote to ramp to the default pose...[/green]")
        while self.read_controller_input().keys != self.controller_mapping["start"]:
            time.sleep(self.config.control_dt)
        print("[green]Ramping to default pose.[/green]")

        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)
        q0 = self.get_robot_state()[0]
        q_tgt = self.config.default_angles.copy()
        for i in range(num_step):
            alpha = i / num_step
            self.send_robot_action(q0 * (1 - alpha) + q_tgt * alpha)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Holding default pose. Press A to start the policy...")
        while self.read_controller_input().keys != self.controller_mapping["A"]:
            self.send_robot_action(self.config.default_angles.copy())
            time.sleep(self.config.control_dt)

    def close(self):
        self.stop_writer()
        self.send_damping()
        self._stop.set()
        self._reader_thread.join(timeout=1.0)
