"""Read-only DDS probes for the channels teleop_ikea.py records on top of the arm state.

The arm controller already subscribes to rt/lowstate, but it keeps only q and dq
for the 35 motor slots and throws the IMU away. Lower-body learning needs more
than that, and all of it is already on the wire:

  rt/lowstate      (unitree_hg::LowState_,   ~1060 Hz)  whole-body q/dq/tau_est + pelvis IMU
  rt/odommodestate (unitree_go::SportModeState_, ~500 Hz)  base position / velocity / yaw_speed
  rt/secondary_imu (unitree_hg::IMUState_,   ~1055 Hz)  torso IMU (the head cameras' own tilt)

Each probe owns its own subscriber and reader thread, so nothing here perturbs
the control path. Deliberately NOT used:

  rt/lf/*            every low-frequency mirror publishes at 20 Hz. The BCT export
                     recorded base pose from rt/lf/odommodestate, which is why its
                     root channel was a 20 Hz signal zero-order-held into 30 fps.
  SportModeState_.body_height / foot_force / foot_position_body / stamp
                     quadruped leftovers, verified to stay 0 on G1.
  MotorState_.ddq    the firmware never fills it.
"""

import math
import threading
import time

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

kTopicLowState = "rt/lowstate"
kTopicOdomModeState = "rt/odommodestate"
kTopicSecondaryImu = "rt/secondary_imu"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightState = "rt/dex1/right/state"

# G1 29-DoF: motor slots 0..28 are the real joints, 29..34 are unused padding.
G1_29_NUM_JOINTS = 29


def quat_to_projected_gravity(quat_wxyz):
    """Gravity direction expressed in the body frame, from a Unitree IMU quaternion.

    Unitree publishes [w, x, y, z] and does NOT canonicalise the sign, so the
    same attitude shows up as q or -q from one sample to the next. Feeding those
    four numbers to a policy therefore feeds it a channel that flips sign for no
    physical reason -- and two of the four components are almost pure yaw, which
    is an arbitrary per-session offset anyway.

    The projected gravity vector has neither problem: it is invariant to the
    quaternion sign and to yaw, and it carries exactly the roll/pitch information
    that is not recoverable from the joint encoders.

    Returns [gx, gy, gz], a unit vector, ~[0, 0, -1] when the pelvis is upright.
    """
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return [0.0, 0.0, -1.0]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # -R[2, :], i.e. minus the bottom row of the body->world rotation. Every term
    # is a product of two components, so q and -q give the same answer.
    return [2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            2.0 * (x * x + y * y) - 1.0]


def decode_lowstate(msg, num_joints=G1_29_NUM_JOINTS, with_wireless_remote=False):
    """rt/lowstate -> whole-body q/dq/tau_est plus the pelvis IMU.

    Kept module level and free of DDS so it can be exercised against a plain
    stand-in message; the field names are the part most likely to rot when the
    SDK moves.
    """
    imu = msg.imu_state
    out = {
        # dq and tau_est for every joint, not just the arms: the legs and waist
        # are the whole point of reading this a second time.
        "q": [float(msg.motor_state[i].q) for i in range(num_joints)],
        "dq": [float(msg.motor_state[i].dq) for i in range(num_joints)],
        "tau_est": [float(msg.motor_state[i].tau_est) for i in range(num_joints)],
        "imu_quat": [float(v) for v in imu.quaternion],          # [w, x, y, z], sign not canonical
        "imu_rpy": [float(v) for v in imu.rpy],
        "imu_gyro": [float(v) for v in imu.gyroscope],           # base angular velocity, measured
        "imu_accel": [float(v) for v in imu.accelerometer],
        "tick": int(msg.tick),
    }
    out["projected_gravity"] = quat_to_projected_gravity(out["imu_quat"])
    if with_wireless_remote:
        # Raw Unitree remote bytes. Zero throughout XR-controller teleoperation,
        # so off by default; useful only when the base is driven from the remote.
        out["wireless_remote"] = [int(b) for b in msg.wireless_remote]
    return out


def decode_odom(msg):
    """rt/odommodestate -> base pose and twist.

    position and velocity are in DIFFERENT frames. Measured on the robot, not
    assumed: fitting d(position)/dt = R(alpha * yaw) @ velocity over 23k samples
    spanning 32 degrees of yaw gives alpha = 1.000, bootstrap 95% CI
    [0.90, 1.11], which excludes the world hypothesis (alpha = 0) outright.

        position   world (integrated in the odom frame)
        velocity   body  (forward/left/up of the robot)
        yaw_speed  identical either way, the z axis is shared

    So they must not be concatenated into one state vector. The body-frame
    velocity is the one a policy wants -- it is invariant to heading -- and it
    agrees with the pelvis gyro, whose IMU mounts with rpy="0 0 0" and is
    therefore already in the pelvis frame too.

    body_height, foot_force, foot_position_body and stamp are quadruped leftovers
    that stay 0 on G1, so they are not read.
    """
    imu = msg.imu_state
    return {
        # World frame, and session-local: fine for analysis and evaluation, not a
        # policy input. It drifts and its origin is wherever the estimator last
        # reset, so across clips it behaves like a clip id.
        "position": [float(v) for v in msg.position],
        # Body frame. Already differentiated and filtered by the onboard
        # estimator; do not recompute it from position, which at 20-30 Hz is
        # mostly finite-difference noise.
        "velocity": [float(v) for v in msg.velocity],
        "yaw_speed": float(msg.yaw_speed),
        "imu_quat": [float(v) for v in imu.quaternion],
        "imu_rpy": [float(v) for v in imu.rpy],
    }


def decode_imu(msg):
    """rt/secondary_imu -> torso IMU."""
    out = {
        "imu_quat": [float(v) for v in msg.quaternion],
        "imu_rpy": [float(v) for v in msg.rpy],
        "imu_gyro": [float(v) for v in msg.gyroscope],
        "imu_accel": [float(v) for v in msg.accelerometer],
    }
    out["projected_gravity"] = quat_to_projected_gravity(out["imu_quat"])
    return out


def decode_gripper(msg):
    """rt/dex1/{side}/state -> the fields that are actually populated.

    ddq, temperature and the *_raw fields come through as 0 and are skipped.
    """
    motor = msg.states[0]
    return {
        "q": [float(motor.q)],
        "dq": [float(motor.dq)],
        "tau_est": [float(motor.tau_est)],
    }


class _Latest:
    """Single-slot buffer holding the newest sample and when it was received."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._recv_us = 0
        self._count = 0

    def set(self, value):
        with self._lock:
            self._value = value
            self._recv_us = int(time.time() * 1e6)
            self._count += 1

    def get(self):
        with self._lock:
            return self._value, self._recv_us, self._count


class _Probe:
    """Common plumbing: subscribe, poll on a daemon thread, expose the newest sample."""

    topic = None
    msg_type = None
    label = "probe"

    def __init__(self, poll_dt=0.002, wait_timeout=3.0):
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._latest = _Latest()
        self._stop = threading.Event()
        self._subscriber = ChannelSubscriber(self.topic, self.msg_type)
        self._subscriber.Init()
        self._poll_dt = poll_dt
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if wait_timeout > 0 and not self.wait_ready(wait_timeout):
            logger_mp.warning(f"[{self.label}] no sample from {self.topic} after {wait_timeout}s; "
                              "recording will emit nulls for it.")

    def _run(self):
        while not self._stop.is_set():
            try:
                msg = self._subscriber.Read()
            except Exception as error:                       # a dead reader must not kill recording
                logger_mp.warning(f"[{self.label}] read failed: {error}")
                msg = None
            if msg is not None:
                try:
                    self._latest.set(self._decode(msg))
                except Exception as error:
                    logger_mp.warning(f"[{self.label}] decode failed: {error}")
            time.sleep(self._poll_dt)

    def _decode(self, msg):
        raise NotImplementedError

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._latest.get()[0] is not None:
                return True
            time.sleep(0.02)
        return self._latest.get()[0] is not None

    def snapshot(self, now_us=None):
        """Newest decoded sample plus its receive time and age, or None if nothing arrived."""
        value, recv_us, count = self._latest.get()
        if value is None:
            return None
        if now_us is None:
            now_us = int(time.time() * 1e6)
        out = dict(value)
        out["recv_wall_time_us"] = recv_us
        out["age_ms"] = round((now_us - recv_us) / 1000.0, 3)
        out["sample_count"] = count
        return out

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._subscriber.Close()
        except Exception:
            pass


class LowStateProbe(_Probe):
    """Whole-body motor state and pelvis IMU from rt/lowstate."""

    label = "lowstate"

    def __init__(self, num_joints=G1_29_NUM_JOINTS, with_wireless_remote=False, **kwargs):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

        self.topic = kTopicLowState
        self.msg_type = hg_LowState
        self._num_joints = num_joints
        self._with_wireless_remote = with_wireless_remote
        super().__init__(**kwargs)

    def _decode(self, msg):
        return decode_lowstate(msg, self._num_joints, self._with_wireless_remote)


class OdomProbe(_Probe):
    """Base odometry from rt/odommodestate (the 500 Hz topic, not the 20 Hz lf mirror)."""

    label = "odommodestate"

    def __init__(self, **kwargs):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as go_SportModeState

        self.topic = kTopicOdomModeState
        self.msg_type = go_SportModeState
        super().__init__(**kwargs)

    def _decode(self, msg):
        return decode_odom(msg)


class SecondaryImuProbe(_Probe):
    """Torso IMU from rt/secondary_imu.

    The head cameras are bolted to torso_link, so this is the cameras' own tilt.
    Its yaw is integrated independently from the pelvis IMU's, so the two yaws
    are NOT differenced to recover the waist angle -- use motor_state[12..14].q
    for that.
    """

    label = "secondary_imu"

    def __init__(self, **kwargs):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_ as hg_IMUState

        self.topic = kTopicSecondaryImu
        self.msg_type = hg_IMUState
        super().__init__(**kwargs)

    def _decode(self, msg):
        return decode_imu(msg)


class Dex1GripperProbe(_Probe):
    """One Dex1 gripper's motor state.

    The gripper controller already reads this topic, but it only keeps q. dq and
    tau_est are on the wire too and they are what fills the qvel/torque slots
    that the recorder has always written as empty lists. ddq, temperature and the
    *_raw fields come through as 0 and are skipped.
    """

    label = "dex1"

    def __init__(self, topic, side, **kwargs):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_ as go_MotorStates

        self.topic = topic
        self.msg_type = go_MotorStates
        self.label = f"dex1/{side}"
        super().__init__(**kwargs)

    def _decode(self, msg):
        return decode_gripper(msg)


class RobotStateProbes:
    """All probes together, each optional so a missing topic degrades gracefully."""

    def __init__(self, enable_lowstate=True, enable_odom=True, enable_secondary_imu=True,
                 enable_gripper=True, num_joints=G1_29_NUM_JOINTS, wait_timeout=3.0,
                 with_wireless_remote=False, poll_dt=0.002):
        if not math.isfinite(poll_dt) or poll_dt <= 0:
            raise ValueError('recording probe poll_dt must be finite and positive')
        self.lowstate = self._build(
            LowStateProbe, enable_lowstate, num_joints=num_joints, wait_timeout=wait_timeout,
            with_wireless_remote=with_wireless_remote, poll_dt=poll_dt)
        self.odom = self._build(OdomProbe, enable_odom, wait_timeout=wait_timeout, poll_dt=poll_dt)
        self.secondary_imu = self._build(
            SecondaryImuProbe, enable_secondary_imu, wait_timeout=wait_timeout, poll_dt=poll_dt)
        self.left_gripper = self._build(
            Dex1GripperProbe, enable_gripper,
            topic=kTopicGripperLeftState, side="left", wait_timeout=wait_timeout, poll_dt=poll_dt)
        self.right_gripper = self._build(
            Dex1GripperProbe, enable_gripper,
            topic=kTopicGripperRightState, side="right", wait_timeout=wait_timeout, poll_dt=poll_dt)

    @staticmethod
    def _build(cls, enabled, **kwargs):
        if not enabled:
            return None
        try:
            return cls(**kwargs)
        except Exception as error:
            # A missing IDL type or an absent topic must not stop a recording
            # session; the affected keys simply come out null.
            logger_mp.warning(f"[{cls.label}] disabled: {error}")
            return None

    def _all(self):
        return (("lowstate", self.lowstate), ("odom", self.odom),
                ("secondary_imu", self.secondary_imu),
                ("left_gripper", self.left_gripper), ("right_gripper", self.right_gripper))

    def snapshot(self, now_us=None):
        if now_us is None:
            now_us = int(time.time() * 1e6)
        return {name: (probe.snapshot(now_us) if probe is not None else None)
                for name, probe in self._all()}

    def status(self):
        """Human-readable summary of which probes actually delivered data."""
        parts = []
        for name, probe in self._all():
            if probe is None:
                parts.append(f"{name}=off")
            else:
                snap = probe.snapshot()
                parts.append(f"{name}={'ok' if snap else 'NO DATA'}")
        return ", ".join(parts)

    def close(self):
        for name, probe in self._all():
            if probe is not None:
                try:
                    probe.close()
                except Exception as error:
                    logger_mp.warning(f"Failed to close {name}: {error}")


def _self_test():
    """Check projected gravity against a real sample read off the robot, no DDS needed."""
    # rt/lowstate sample: quaternion and rpy captured in the same message, so the
    # gravity vector derived from the quaternion must agree with the rpy.
    quat = [-0.8078, -0.0064, -0.0045, 0.5893]
    roll, pitch = 0.0051, 0.0148
    got = quat_to_projected_gravity(quat)
    want = [math.sin(pitch),
            -math.cos(pitch) * math.sin(roll),
            -math.cos(pitch) * math.cos(roll)]
    for name, g, w in zip("xyz", got, want):
        assert abs(g - w) < 2e-3, f"g{name}: {g:.6f} != {w:.6f}"
    # Unitree does not canonicalise the quaternion sign; the answer must not care.
    flipped = quat_to_projected_gravity([-v for v in quat])
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, flipped)), "sign flip changed the result"
    # Upright pelvis.
    assert quat_to_projected_gravity([1.0, 0.0, 0.0, 0.0]) == [0.0, -0.0, -1.0]
    print(f"projected_gravity(real lowstate sample) = {[round(v, 6) for v in got]}")
    print(f"expected from its own rpy              = {[round(v, 6) for v in want]}")
    print("robot_state_probe self-test OK")


if __name__ == "__main__":
    _self_test()
