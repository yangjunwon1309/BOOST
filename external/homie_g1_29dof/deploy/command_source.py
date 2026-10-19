"""cmd_vel [vx, vy, wz] sources for the HOMIE control loop.

Implementations:
  * CommandSource -- decodes the G1 wireless controller sticks (unchanged default).
      Left stick  -> vx (forward, ly) / vy (lateral, lx)
      Right stick -> wz (yaw, rx)
  * UdpNavCommandSource -- receives the production Thor-to-G1 unicast feed,
      validates sender/session/sequence, and expires it to zero after 0.3 s.
  * DdsNavCommandSource -- wraps CommandSource with an external nav cmd_vel read off
      the `rt/nav_cmd` DDS topic (e.g. a vision/nav stack running off-robot, publishing
      the same way xr_teleoperate publishes `rt/arm_sdk` -- see arm_target_source.py's
      ArmSdkArmTargetSource, which this mirrors). The wireless sticks are NEVER
      disabled: touching a stick beyond the deadzone always overrides the nav feed on
      that same call, giving the operator an instant, unconditional take-back --
      effectively a second software e-stop on top of the hardware Select button. A
      watchdog zeros the nav-derived command (does NOT hold the last value) once no
      message has arrived within `timeout` seconds, so a dead/killed sender degrades
      to stand rather than to a stale, possibly-large velocity.

Wire format (UDP bytes, or DDS `std_msgs.msg.dds_.String_.data` in compatibility mode):
  JSON {"vx": <float>, "vy": <float>, "wz": <float>, "seq": <int>,
  "session": <str>, "state": <str>}. vx/vy/wz are body-frame m/s, m/s, rad/s -- the SAME units/frame
  HOMIE's own stick command already uses (see CommandSource.get_command). `state` is
  informational only (logged, not consumed). `seq` is used only to detect drops for
  logging; there is no timestamp field by design -- staleness is judged purely by
  local arrival time, so the sender does not need clock sync with this process. Any
  message that fails to parse as JSON, is missing vx/vy/wz, or decodes to a
  non-finite value, is dropped and logged (rate-limited) rather than applied.

Sticks report in [-1, 1]. A deadzone is applied before scaling by max_cmd so a
released stick yields exactly zero -> the policy switches to stand.onnx.
Flip the SIGN_* constants if an axis drives the robot the wrong way on your unit.
"""
import json
import socket
import threading
import time

import numpy as np

SIGN_VX = 1.0
SIGN_VY = -1.0   # flipped: left stick left was driving the robot right
SIGN_WZ = -1.0   # right stick left -> turn left (+yaw) by default

NAV_CMD_TOPIC = "rt/nav_cmd"
NAV_CMD_UDP_PORT = 15001


class CommandSource:
    def __init__(self, config):
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)   # [vx, vy, wz]
        self.deadzone = float(config.cmd_deadzone)

    def _dz(self, x):
        if abs(x) < self.deadzone:
            return 0.0
        # rescale so the response is continuous just outside the deadzone
        return (x - np.sign(x) * self.deadzone) / (1.0 - self.deadzone)

    def stick_active(self, controller):
        """True if any stick this class reads from is outside its deadzone."""
        return (abs(self._dz(float(controller.lx))) > 0.0
                or abs(self._dz(float(controller.ly))) > 0.0
                or abs(self._dz(float(controller.rx))) > 0.0)

    def get_command(self, controller):
        lx = self._dz(float(controller.lx))
        ly = self._dz(float(controller.ly))
        rx = self._dz(float(controller.rx))

        cmd = np.array([
            SIGN_VX * ly * self.max_cmd[0],
            SIGN_VY * lx * self.max_cmd[1],
            SIGN_WZ * rx * self.max_cmd[2],
        ], dtype=np.float32)
        return cmd


class DdsNavCommandSource:
    """External nav cmd_vel over the `rt/nav_cmd` DDS topic, with the wireless sticks
    retained as an unconditional override and a zero-on-stale watchdog. See module
    docstring for the wire format and the override/watchdog rationale.

    Uses unitree_sdk2py (the same binding xr_teleoperate/HOMIE use elsewhere). The DDS
    reader runs on a dedicated background thread that blocks on each `Read()` --
    cyclonedds's default (no-timeout) read blocks for ~infinity rather than returning
    None when idle, so calling it straight from the 50 Hz control loop would freeze the
    whole policy the moment the nav stack isn't actively publishing (see
    ArmSdkArmTargetSource's docstring for the same issue on `rt/arm_sdk`).
    `get_command()` only ever reads the cached latest sample, so it's always
    non-blocking regardless of publisher state.
    """

    def __init__(self, config, net, timeout=0.3, init_channel=True, domain_id=0,
                 topic=NAV_CMD_TOPIC):
        """`domain_id`/`topic` default to the production DDS domain (0) and topic
        (`rt/nav_cmd`) -- override ONLY for test isolation, never in normal deploy
        use. `init_channel=True` by default (unlike ArmSdkArmTargetSource) because this
        source may be constructed before G1RealWorldEnv has initialized the channel
        factory; the double-init case (env already did it) is caught and ignored
        below, same as make_arm_source's "arm_sdk" path does."""
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self.sticks = CommandSource(config)
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.timeout = float(timeout)
        self.topic = str(topic)

        self._lock = threading.Lock()
        self._latest_cmd = np.zeros(3, dtype=np.float32)
        self._latest_time = 0.0
        self._last_seq = None

        if init_channel:
            try:
                ChannelFactoryInitialize(domain_id, net)
            except Exception as exc:
                print(f"[cmd] ChannelFactoryInitialize skipped: {exc}")

        self._String_ = String_
        self.sub = ChannelSubscriber(self.topic, String_)
        self.sub.Init()

        self._stop = threading.Event()
        self._last_bad_warn = 0.0
        self._last_gap_warn = 0.0
        self._thread = threading.Thread(
            target=self._reader_loop, name="nav-cmd-reader", daemon=True
        )
        self._thread.start()
        print(f"[cmd] subscribed '{self.topic}' (nav cmd_vel, "
              f"timeout={self.timeout:g}s, sticks always override)")

    def _reader_loop(self):
        n = 0
        last_cmd = None
        last_state = None
        report_t0 = time.time()
        while not self._stop.is_set():
            try:
                msg = self.sub.Read()  # blocks until the next sample; fine off-loop
            except Exception as exc:
                print(f"[cmd] nav_cmd read failed: {exc}")
                time.sleep(0.1)
                msg = None

            if msg is not None:
                parsed = self._parse(msg.data)
                if parsed is not None:
                    cmd, state = parsed
                    with self._lock:
                        self._latest_cmd = cmd
                        self._latest_time = time.time()
                    n += 1
                    last_cmd, last_state = cmd, state

            now = time.time()
            if now - report_t0 >= 1.0:
                if n == 0:
                    print(f"[cmd] nav_cmd recv 0.0 Hz -- no valid '{self.topic}' "
                          f"messages in the last {now - report_t0:.1f}s")
                else:
                    print(f"[cmd] nav_cmd recv {n / (now - report_t0):5.1f} Hz  "
                          f"latest={last_cmd.tolist()}  state={last_state}")
                n = 0
                report_t0 = now

    def _parse(self, data):
        try:
            payload = json.loads(data)
            vx = float(payload["vx"])
            vy = float(payload["vy"])
            wz = float(payload["wz"])
        except Exception as exc:
            self._warn_bad(f"unparseable payload {data!r}: {exc}")
            return None

        cmd = np.array([vx, vy, wz], dtype=np.float32)
        if not np.all(np.isfinite(cmd)):
            self._warn_bad(f"non-finite values {cmd.tolist()}")
            return None

        seq = payload.get("seq")
        if isinstance(seq, int) and self._last_seq is not None:
            gap = seq - self._last_seq
            if gap > 1 and time.time() - self._last_gap_warn > 1.0:
                print(f"[cmd] WARN: nav_cmd seq gap detected ({gap - 1} message(s) "
                      f"lost, {self._last_seq} -> {seq})")
                self._last_gap_warn = time.time()
        if isinstance(seq, int):
            self._last_seq = seq

        return np.clip(cmd, -self.max_cmd, self.max_cmd), payload.get("state")

    def _warn_bad(self, reason):
        now = time.time()
        if now - self._last_bad_warn > 1.0:
            print(f"[cmd] WARN: dropping bad nav_cmd message: {reason}")
            self._last_bad_warn = now

    def get_command(self, controller):
        # The operator's sticks always win: any deflection on this same call takes
        # over instantly, with no latch -- releasing the sticks hands control straight
        # back to the (possibly-updated) nav feed on the very next call.
        if self.sticks.stick_active(controller):
            return self.sticks.get_command(controller)

        with self._lock:
            cmd, age = self._latest_cmd.copy(), time.time() - self._latest_time

        if age > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return cmd

    def close(self):
        self._stop.set()
        close = getattr(self.sub, "Close", None)
        if callable(close):
            close()
        self._thread.join(timeout=1.0)


class UdpNavCommandSource:
    """Thor unicast input with sender filtering and zero-on-stale behavior."""

    def __init__(self, config, timeout=None, socket_cls=None):
        self.sticks = CommandSource(config)
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.timeout = float(
            config.cmd_udp_timeout if timeout is None else timeout
        )
        if not np.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError("cmd_udp_timeout must be finite and positive")
        self.bind_host = str(config.cmd_udp_bind_host)
        self.port = int(config.cmd_udp_port)
        self.allowed_host = str(config.cmd_udp_allowed_host).strip()
        if not 1 <= self.port <= 65535:
            raise ValueError("cmd_udp_port must be in [1, 65535]")
        if not self.allowed_host:
            raise ValueError("cmd_udp_allowed_host must not be empty")

        self._lock = threading.Lock()
        self._latest_cmd = np.zeros(3, dtype=np.float32)
        self._latest_time = 0.0
        self._last_session = None
        self._last_seq = -1
        self._retired_sessions = set()
        self._last_bad_warn = 0.0
        self._last_gap_warn = 0.0
        if socket_cls is None:
            socket_cls = socket.socket
        self._socket = socket_cls(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.1)
        self._socket.bind((self.bind_host, self.port))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="nav-cmd-udp-reader",
            daemon=True,
        )
        self._thread.start()
        print(
            "[cmd] listening udp://%s:%d for Thor %s "
            "(timeout=%gs, sticks always override)"
            % (
                self.bind_host,
                self.port,
                self.allowed_host,
                self.timeout,
            )
        )

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                data, address = self._socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    break
                self._warn_bad("UDP receive failed: %s" % exc)
                continue
            if address[0] != self.allowed_host:
                self._warn_bad("packet from disallowed host %s" % address[0])
                continue
            parsed = self._parse(data)
            if parsed is None:
                continue
            command, state = parsed
            if state == "receiver_probe":
                acknowledgement = json.dumps(
                    {
                        "type": "nav_ack",
                        "session": self._last_session,
                        "seq": self._last_seq,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                try:
                    self._socket.sendto(acknowledgement, address)
                except OSError as exc:
                    self._warn_bad("UDP acknowledgement failed: %s" % exc)
            with self._lock:
                self._latest_cmd = command
                self._latest_time = time.monotonic()

    def _parse(self, data):
        try:
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            payload = json.loads(data)
            command = np.asarray(
                [payload["vx"], payload["vy"], payload["wz"]],
                dtype=np.float32,
            )
            session = str(payload["session"])
            sequence = payload["seq"]
        except Exception as exc:
            self._warn_bad("unparseable UDP payload %r: %s" % (data, exc))
            return None
        if command.shape != (3,) or not np.isfinite(command).all():
            self._warn_bad("non-finite or malformed command %s" % command.tolist())
            return None
        if not session or not isinstance(sequence, int) or sequence < 0:
            self._warn_bad("invalid session/sequence")
            return None
        if session in self._retired_sessions:
            return None
        if self._last_session is not None and session != self._last_session:
            with self._lock:
                age = time.monotonic() - self._latest_time
            if age <= self.timeout:
                self._warn_bad("new session rejected while current sender is fresh")
                return None
            self._retired_sessions.add(self._last_session)
            self._last_seq = -1
        if session == self._last_session:
            if sequence <= self._last_seq:
                return None
            gap = sequence - self._last_seq
            if (
                self._last_seq >= 0
                and gap > 1
                and time.monotonic() - self._last_gap_warn > 1.0
            ):
                print(
                    "[cmd] WARN: UDP nav_cmd seq gap detected "
                    "(%d message(s) lost, %d -> %d)"
                    % (gap - 1, self._last_seq, sequence)
                )
                self._last_gap_warn = time.monotonic()
        self._last_session = session
        self._last_seq = sequence
        return np.clip(command, -self.max_cmd, self.max_cmd), payload.get("state")

    def _warn_bad(self, reason):
        now = time.monotonic()
        if now - self._last_bad_warn > 1.0:
            print("[cmd] WARN: dropping bad UDP nav_cmd message: %s" % reason)
            self._last_bad_warn = now

    def get_command(self, controller):
        if self.sticks.stick_active(controller):
            return self.sticks.get_command(controller)
        with self._lock:
            command = self._latest_cmd.copy()
            age = time.monotonic() - self._latest_time
        if age > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return command

    def close(self):
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=1.0)


def make_command_source(config, net=None):
    src = getattr(config, "cmd_source", "wireless")
    if src == "udp":
        return UdpNavCommandSource(config)
    if src == "dds":
        return DdsNavCommandSource(config, net, timeout=config.cmd_dds_timeout)
    if src != "wireless":
        raise ValueError("unsupported cmd_source: %r" % src)
    return CommandSource(config)
