"""External 14-dim arm-target source for joints 15..28 (HOMIE does not control arms).

Implementations:
  * HoldDefaultArmSource -- always returns the config default arm pose.
  * ArmSdkArmTargetSource -- subscribes the Unitree `rt/arm_sdk` DDS topic that
    xr_teleoperate publishes and reads motor_cmd[15..28].q (and .kp/.kd, see below).
    This needs NO changes to xr_teleoperate: run it normally (motion mode) and it
    drives our arms. Falls back to the default pose while the teleop "weight"
    (kNotUsedJoint0.q) is ~0 or the channel is silent.
  * RedisArmTargetSource -- reads a JSON list of 14 floats from a Redis key
    (hardware arm order 15..28). On stale/missing input it holds the last valid
    target. Handy if you'd rather bridge arm targets over Redis than DDS.

All sources return arms in hardware order 15..28 == G1_29_JointArmIndex
(left arm 7 then right arm 7), which is exactly xr_teleoperate's IK `sol_q`.

Every source also exposes `get_arm_gains() -> (kp14, kd14)`. xr_teleoperate's own arm
controller (`robot_arm.py`) publishes real per-joint tuned gains alongside `q` in the
same `rt/arm_sdk` message: shoulder (pitch/roll/yaw) + elbow get `kp_low=80`/`kd_low=3.0`,
wrist (roll/pitch/yaw) gets `kp_wrist=40`/`kd_wrist=1.5` -- `kp_high=300`/`kd_high=3.0` is
NOT used for arms at all, it's what `robot_arm.py` applies to the *locked, non-arm*
joints (legs/waist) while teleop owns the whole body; HOMIE never sees or forwards that
value since we only read ARM_MOTOR_IDS (15..28). These teleop gains are already softer
for the shoulder than HOMIE's own conservative bring-up default would suggest -- if the
shoulder feels sluggish, that softness is coming from xr_teleoperate's own `kp_low=80`
choice (same value used with or without HOMIE in the loop), not from anything getting
lost in translation here. If `run_homie.py` always published the config default gains
instead of forwarding the teleop's, tracking would feel wrong/overridden even though the
*positions* were correct -- this is exactly what `get_arm_gains()` fixes.

Every source also exposes `get_arm_command() -> (q14, kp14, kd14, tau14)`, an atomic
one-call accessor covering position, gains, AND feedforward torque (tau) from the same
underlying sample -- prefer this over calling get_arm_target()/get_arm_gains()
separately when you need all of it consistently (see ArmSdkArmTargetSource's docstring
for why tau specifically needs its own careful staleness/validity handling that q/kp/kd
don't).

`make_arm_source(config, net)` picks the implementation from config.arm_source.
"""
import json
import threading
import time

import numpy as np

ARM_DIM = 14
ARM_SLICE = slice(15, 29)
ARM_MOTOR_IDS = list(range(15, 29))       # G1_29_JointArmIndex
ARM_SDK_TOPIC = "rt/arm_sdk"
ARM_SDK_WEIGHT_ID = 29                    # kNotUsedJoint0 carries the enable weight
ARM_SDK_WEIGHT_ON = 0.5                   # treat weight >= this as "teleop active"

# Per-joint index within the 14-dim arm arrays (ARM_MOTOR_IDS order == G1_29_JointArmIndex):
#   0 L shoulder pitch  1 L shoulder roll  2 L shoulder yaw  3 L elbow
#   4 L wrist roll      5 L wrist pitch    6 L wrist yaw
#   7 R shoulder pitch  8 R shoulder roll  9 R shoulder yaw  10 R elbow
#   11 R wrist roll     12 R wrist pitch   13 R wrist yaw


def _normalize_tau_clamp(tau_clamp):
    """Accept either a scalar (broadcast to all 14 joints) or a length-14
    array/list (per-joint, e.g. G1 URDF effort limits) and return a validated
    (14,) float32 array. Raises ValueError on a wrong-length input or on any
    element that isn't finite and strictly positive -- a clamp of 0 or less
    would silently zero all torque, and NaN/Inf would defeat the clamp entirely."""
    arr = np.asarray(tau_clamp, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full(ARM_DIM, float(arr), dtype=np.float32)
    else:
        if arr.shape != (ARM_DIM,):
            raise ValueError(
                f"arm_tau_clamp must be a scalar or a length-{ARM_DIM} array, "
                f"got shape {arr.shape}"
            )
        arr = arr.reshape(ARM_DIM).astype(np.float32)
    if not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        raise ValueError(f"arm_tau_clamp must be finite and > 0 for every joint, got {arr}")
    return arr


def _normalize_tau_scale(tau_scale):
    """Validate the tau_ff multiplier: must be finite and in [0, 1]. This is applied
    ON TOP of the teleop weight scale, not instead of it -- >1 would let this single
    knob amplify teleop's own torque beyond what it intended, and NaN/Inf would defeat
    every downstream clamp (0 * inf == nan, which np.clip does not sanitize)."""
    value = float(tau_scale)
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"arm_tau_scale must be finite and in [0, 1], got {tau_scale}")
    return value


class HoldDefaultArmSource:
    def __init__(self, default_arm, default_kp, default_kd):
        self.default_arm = np.asarray(default_arm, dtype=np.float32).reshape(ARM_DIM)
        self.default_kp = np.asarray(default_kp, dtype=np.float32).reshape(ARM_DIM)
        self.default_kd = np.asarray(default_kd, dtype=np.float32).reshape(ARM_DIM)
        self.default_tau = np.zeros(ARM_DIM, dtype=np.float32)

    def get_arm_target(self):
        return self.default_arm.copy()

    def get_arm_gains(self):
        return self.default_kp.copy(), self.default_kd.copy()

    def get_arm_command(self):
        """Atomic (q14, kp14, kd14, tau14). No feedforward torque source here -> tau
        is always zero."""
        return self.default_arm.copy(), self.default_kp.copy(), self.default_kd.copy(), \
            self.default_tau.copy()


class RedisArmTargetSource:
    def __init__(self, default_arm, default_kp, default_kd, key, timeout,
                 host="localhost", port=6379, db=0):
        import redis  # imported lazily so the dep is optional
        self.default_arm = np.asarray(default_arm, dtype=np.float32).reshape(ARM_DIM)
        self.default_kp = np.asarray(default_kp, dtype=np.float32).reshape(ARM_DIM)
        self.default_kd = np.asarray(default_kd, dtype=np.float32).reshape(ARM_DIM)
        self.key = key
        self.timeout = float(timeout)
        self.client = redis.Redis(host=host, port=port, db=db)
        self.last_target = self.default_arm.copy()

    def get_arm_target(self):
        try:
            raw = self.client.get(self.key)
            if raw is not None:
                payload = json.loads(raw)
                target = np.asarray(payload.get("arm", payload), dtype=np.float32)
                ts = float(payload["t"]) if isinstance(payload, dict) and "t" in payload else time.time()
                if target.shape[0] == ARM_DIM and (time.time() - ts) <= self.timeout:
                    self.last_target = target
        except Exception as exc:  # never let arm I/O crash the control loop
            print(f"[arm] redis read failed, holding last target: {exc}")
        return self.last_target.copy()

    def get_arm_gains(self):
        # The Redis channel only carries positions (see arm_sdk_to_redis.py); gains
        # always fall back to the config default.
        return self.default_kp.copy(), self.default_kd.copy()

    def get_arm_command(self):
        """Atomic (q14, kp14, kd14, tau14). The Redis channel carries positions only
        (see arm_sdk_to_redis.py) -> gains fall back to default and tau is always zero."""
        q = self.get_arm_target()
        kp, kd = self.get_arm_gains()
        return q, kp, kd, np.zeros(ARM_DIM, dtype=np.float32)


class ArmSdkArmTargetSource:
    """Read arm joint targets from the `rt/arm_sdk` topic xr_teleoperate publishes.

    Uses unitree_sdk2py (the same binding xr_teleoperate uses). It shares DDS domain 0
    with the control interface, so `net` must match. When the teleop enable weight is
    below threshold (or nothing has arrived yet), holds the default arm pose.

    The DDS reader itself runs on a dedicated background thread that blocks on each
    `Read()` -- cyclonedds's default (no-timeout) read blocks for ~infinity
    (`duration(weeks=99999)`) rather than returning `None` when idle, so calling it
    straight from the 50 Hz control loop would freeze the whole policy the moment
    xr_teleoperate isn't actively publishing. `get_arm_target()` only ever reads the
    cached latest sample, so it's always non-blocking regardless of teleop state.

    Output smoothing: xr_teleoperate's own `rt/arm_sdk` publish rate is observed to be
    irregular in practice (Thor-side IK/vision/recording load), and without HOMIE the
    G1 firmware's native `rt/arm_sdk` overlay apparently smooths/interpolates between
    updates on its own -- that overlay is bypassed once HOMIE owns `rt/lowcmd` (see
    g1_wrapper.py's docstring), and this class used to just repeat the last raw sample
    verbatim until a new one arrived, which reproduces as visible "stepping" whenever the
    source updates arrive unevenly. `get_arm_target()` now runs a simple time-constant
    low-pass filter (see `smoothing_tau`) toward the latest raw sample on every call
    (i.e. at the writer thread's own high rate), so the commanded position advances
    continuously between raw updates instead of holding-then-jumping. Gains are NOT
    smoothed (kp/kd should track the teleop weight/mode transitions immediately).

    Feedforward torque (tau): xr_teleoperate also fills `motor_cmd[15:29].tau` (its own
    arm gravity compensation). `get_arm_command()` forwards it, scaled by `tau_scale` and
    hard-clamped to `tau_clamp` N*m per joint, on top of the (0..1-clamped) teleop weight
    itself -- so torque ramps in/out with the weight rather than snapping. Unlike q/kp/kd,
    tau is NOT held across a stale/disengaged period: it is recomputed to exactly zero on
    every single call unless the very latest sample is actively engaged, fresh (age <=
    `timeout`), and fully finite (no NaN/Inf). This is deliberately more conservative than
    the position hold-on-stale behavior -- a stale or garbage feedforward torque is a
    correctness/safety risk in a way a stale *position* target (which the PD loop still
    actively corrects toward) is not.
    """

    def __init__(self, default_arm, default_kp, default_kd, net, timeout=0.2,
                 init_channel=False, smoothing_tau=0.03, tau_scale=1.0, tau_clamp=5.0,
                 domain_id=0, topic=ARM_SDK_TOPIC):
        """`domain_id`/`topic` default to the production DDS domain (0) and topic
        (`rt/arm_sdk`) -- override ONLY for test isolation (e.g. a non-production domain
        id and a `test/...` topic prefix), never in normal deploy use."""
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd

        self.default_arm = np.asarray(default_arm, dtype=np.float32).reshape(ARM_DIM)
        self.default_kp = np.asarray(default_kp, dtype=np.float32).reshape(ARM_DIM)
        self.default_kd = np.asarray(default_kd, dtype=np.float32).reshape(ARM_DIM)
        self.last_target = self.default_arm.copy()   # raw target from the latest rt/arm_sdk sample
        self.smoothed_target = self.default_arm.copy()  # low-pass output actually returned
        self.last_kp = self.default_kp.copy()
        self.last_kd = self.default_kd.copy()
        self.last_tau = np.zeros(ARM_DIM, dtype=np.float32)
        self.timeout = float(timeout)
        self.smoothing_tau = float(smoothing_tau)
        self._last_smooth_time = None
        self.topic = str(topic)

        # tau_ff safety: conservative unless explicitly configured otherwise (see
        # robot_control/config.py's arm_tau_scale/arm_tau_clamp defaults).
        self.tau_scale = _normalize_tau_scale(tau_scale)
        self.tau_clamp = _normalize_tau_clamp(tau_clamp)
        if np.all(self.tau_clamp == self.tau_clamp[0]):
            clamp_str = f"±{self.tau_clamp[0]:g} N*m/joint"
        else:
            clamp_str = f"±{list(np.round(self.tau_clamp, 2))} N*m (per joint)"
        print(f"[arm] tau_ff safety: scale={self.tau_scale:g}  clamp={clamp_str}  "
              f"timeout={self.timeout:g}s")

        if init_channel:
            # Only if the control interface has NOT already initialized DDS; double
            # init raises, so this is opt-in.
            try:
                ChannelFactoryInitialize(domain_id, net)
            except Exception as exc:
                print(f"[arm] ChannelFactoryInitialize skipped: {exc}")

        self.sub = ChannelSubscriber(self.topic, hg_LowCmd)
        self.sub.Init()

        self._latest_msg = None
        self._latest_msg_time = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_stale_warn = 0.0
        self._last_nan_warn = 0.0
        self._thread = threading.Thread(
            target=self._reader_loop, name="arm-sdk-reader", daemon=True
        )
        self._thread.start()
        print(f"[arm] subscribed '{self.topic}' (xr_teleoperate arm targets)")

    def _reader_loop(self):
        """Diagnostic counters, reported every ~1s: how often rt/arm_sdk actually
        arrives and how evenly spaced. A low/irregular receive rate here means the
        choppiness is upstream (xr_teleoperate's own publish cadence, or the network/
        DDS transport between the two boards) -- NOT this process. Compare against
        g1_wrapper.py's [writer] report, which covers the sending side instead."""
        n = 0
        max_gap_ms = 0.0
        last_recv = None
        report_t0 = time.time()
        while not self._stop.is_set():
            try:
                msg = self.sub.Read()  # blocks until the next sample; fine off-loop
            except Exception as exc:
                print(f"[arm] arm_sdk read failed: {exc}")
                time.sleep(0.1)
                continue
            if msg is not None:
                now = time.time()
                if last_recv is not None:
                    max_gap_ms = max(max_gap_ms, (now - last_recv) * 1000.0)
                last_recv = now
                n += 1
                with self._lock:
                    self._latest_msg = msg
                    self._latest_msg_time = now

                if now - report_t0 >= 1.0:
                    weight = msg.motor_cmd[ARM_SDK_WEIGHT_ID].q
                    # Periodic receive-rate logging disabled to keep the teleop console quiet.
                    # print(f"[arm] {self.topic} recv {n / (now - report_t0):5.0f} Hz  "
                    #       f"max_gap={max_gap_ms:.1f}ms  weight={weight:.2f}")
                    n = 0
                    max_gap_ms = 0.0
                    report_t0 = now

    def get_arm_target(self):
        self._sync()
        return self._smoothed_position()

    def get_arm_gains(self):
        """Per-joint (kp14, kd14), forwarded from the same rt/arm_sdk message so
        teleop tracking uses xr_teleoperate's own tuned gains, not HOMIE's conservative
        bring-up default. Falls back to the default gains while disengaged/idle, same
        as get_arm_target()'s position fallback."""
        self._sync()
        return self.last_kp.copy(), self.last_kd.copy()

    def get_arm_command(self):
        """Atomic (q14, kp14, kd14, tau14) from ONE synced rt/arm_sdk sample.

        Calling get_arm_target()+get_arm_gains() separately each does its own _sync(),
        and the background reader thread could swap in a newer sample in between --
        harmless for q/kp/kd (they're just a slightly newer valid sample), but not
        something to risk for tau. This method syncs exactly once so q/kp/kd/tau are
        always derived from the same underlying message.
        """
        self._sync()
        q = self._smoothed_position()
        return q, self.last_kp.copy(), self.last_kd.copy(), self.last_tau.copy()

    def _smoothed_position(self):
        now = time.time()
        if self._last_smooth_time is None or self.smoothing_tau <= 0.0:
            self.smoothed_target = self.last_target.copy()
        else:
            dt = max(0.0, now - self._last_smooth_time)
            alpha = 1.0 - np.exp(-dt / self.smoothing_tau)
            self.smoothed_target = self.smoothed_target + alpha * (self.last_target - self.smoothed_target)
        self._last_smooth_time = now
        return self.smoothed_target.copy()

    def _sync(self):
        """Pull the latest cached rt/arm_sdk sample into last_target/last_kp/last_kd/
        last_tau, all from the SAME captured message (see get_arm_command()).

        Position/gains: unchanged behavior -- updated only while the teleop weight is
        engaged, otherwise held at whatever was last commanded (no snap).

        tau (feedforward torque) is safety-critical, so unlike position it is recomputed
        from scratch on EVERY call and forced to exactly zero unless ALL of:
          - a message has ever arrived
          - the teleop weight is engaged (>= ARM_SDK_WEIGHT_ON)
          - the message is fresh (age <= self.timeout) -- stale tau is never held, even
            though a stale position is (see class docstring for why)
          - every element is finite (no NaN/Inf)
        When all hold, tau is scaled by the (0..1-clamped) teleop weight -- so torque
        ramps in/out with the weight instead of snapping -- then by tau_scale, then
        hard-clamped per joint to tau_clamp.
        """
        with self._lock:
            msg = self._latest_msg
            msg_time = self._latest_msg_time

        if msg is None:
            self.last_tau = np.zeros(ARM_DIM, dtype=np.float32)
            return

        weight_raw = float(msg.motor_cmd[ARM_SDK_WEIGHT_ID].q)
        engaged = weight_raw >= ARM_SDK_WEIGHT_ON
        age_s = (time.time() - msg_time) if msg_time else float("inf")
        stale = age_s > self.timeout

        if engaged:
            age_ms = age_s * 1000.0
            if age_ms > 30.0 and time.time() - self._last_stale_warn > 1.0:
                print(f"[arm] WARN: consuming a {age_ms:.0f}ms-old rt/arm_sdk sample "
                      f"(teleop weight={weight_raw:.2f}) -- this process is falling behind")
                self._last_stale_warn = time.time()
            self.last_target = np.array(
                [msg.motor_cmd[i].q for i in ARM_MOTOR_IDS], dtype=np.float32
            )
            self.last_kp = np.array(
                [msg.motor_cmd[i].kp for i in ARM_MOTOR_IDS], dtype=np.float32
            )
            self.last_kd = np.array(
                [msg.motor_cmd[i].kd for i in ARM_MOTOR_IDS], dtype=np.float32
            )
            # else: teleop disengaged -> hold whatever it last commanded (no snap).

        if engaged and not stale:
            tau_raw = np.array(
                [msg.motor_cmd[i].tau for i in ARM_MOTOR_IDS], dtype=np.float32
            )
            if not np.all(np.isfinite(tau_raw)):
                if time.time() - self._last_nan_warn > 1.0:
                    print("[arm] WARN: non-finite tau in rt/arm_sdk sample -- zeroing arm tau_ff")
                    self._last_nan_warn = time.time()
                tau_raw = np.zeros(ARM_DIM, dtype=np.float32)
            weight_clamped = float(np.clip(weight_raw, 0.0, 1.0))
            tau = tau_raw * weight_clamped * self.tau_scale
            tau = np.clip(tau, -self.tau_clamp, self.tau_clamp)
            self.last_tau = tau.astype(np.float32)
        else:
            self.last_tau = np.zeros(ARM_DIM, dtype=np.float32)

    def close(self):
        self._stop.set()
        close = getattr(self.sub, "Close", None)
        if callable(close):
            close()
        self._thread.join(timeout=1.0)


def make_arm_source(config, net=None):
    default_arm = np.asarray(config.default_angles, dtype=np.float32)[ARM_SLICE]
    default_kp = np.asarray(config.kps, dtype=np.float32)[ARM_SLICE]
    default_kd = np.asarray(config.kds, dtype=np.float32)[ARM_SLICE]
    src = getattr(config, "arm_source", "hold")
    if src == "arm_sdk":
        print("[arm] using rt/arm_sdk (xr_teleoperate) -- gains forwarded from teleop too")
        return ArmSdkArmTargetSource(
            default_arm, default_kp, default_kd, net,
            timeout=config.arm_redis_timeout, init_channel=True,
            smoothing_tau=getattr(config, "arm_smoothing_tau", 0.03),
            tau_scale=getattr(config, "arm_tau_scale", 1.0),
            tau_clamp=getattr(config, "arm_tau_clamp", 5.0),
        )
    if src == "redis":
        print(f"[arm] using Redis arm-target channel '{config.arm_redis_key}'")
        return RedisArmTargetSource(
            default_arm, default_kp, default_kd,
            config.arm_redis_key, config.arm_redis_timeout,
        )
    print("[arm] holding default arm pose")
    return HoldDefaultArmSource(default_arm, default_kp, default_kd)
