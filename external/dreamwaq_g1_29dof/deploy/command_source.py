"""Safe velocity sources shared by the DreamWaQ real-robot loop.

The default external source is verified Thor-to-G1 unicast UDP.
``cmd_source: dds`` retains the legacy ``rt/nav_cmd`` compatibility path. A
physical wireless-stick deflection always overrides either source and an
expired external command becomes exactly zero.

Wire format (``std_msgs.msg.dds_.String_``)::

    {"vx": 0.1, "vy": 0.0, "wz": 0.0, "seq": 12,
     "session": "...", "state": "human_active"}
"""

import json
import socket
import threading
import time

import numpy as np


SIGN_VX = 1.0
SIGN_VY = -1.0
SIGN_WZ = -1.0
NAV_CMD_TOPIC = "rt/nav_cmd"
NAV_CMD_UDP_PORT = 15001


def configure_command_limits(config, command_lower, command_upper, overrides):
    """Apply requested caps separately to the export's positive/negative ranges."""
    requested = np.asarray(config.max_cmd, dtype=np.float32).copy()
    for index, override in enumerate(overrides):
        if override is not None:
            requested[index] = float(override)
    config.min_cmd = np.maximum(-requested, command_lower).astype(np.float32)
    config.max_cmd = np.minimum(requested, command_upper).astype(np.float32)


def set_command_limits(source, min_cmd, max_cmd):
    """Retarget a running source, and its wireless-stick fallback, to new caps."""
    for target in (source, getattr(source, "sticks", None)):
        if target is not None:
            target.min_cmd = np.asarray(min_cmd, dtype=np.float32).copy()
            target.max_cmd = np.asarray(max_cmd, dtype=np.float32).copy()


class CommandSource:
    def __init__(self, config):
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.min_cmd = np.asarray(getattr(config, "min_cmd", -self.max_cmd), dtype=np.float32)
        self.deadzone = float(config.cmd_deadzone)

    def _dz(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return (value - np.sign(value) * self.deadzone) / (1.0 - self.deadzone)

    def stick_active(self, controller):
        return (
            abs(self._dz(float(controller.lx))) > 0.0
            or abs(self._dz(float(controller.ly))) > 0.0
            or abs(self._dz(float(controller.rx))) > 0.0
        )

    def get_command(self, controller):
        lx = self._dz(float(controller.lx))
        ly = self._dz(float(controller.ly))
        rx = self._dz(float(controller.rx))
        axes = np.asarray([SIGN_VX * ly, SIGN_VY * lx, SIGN_WZ * rx], dtype=np.float32)
        # Scale each side over its full stick travel, so forward can reach the
        # positive training bound without extending the reverse command range.
        return axes * np.where(axes >= 0.0, self.max_cmd, -self.min_cmd)


class ExternalHold:
    """Thor's hold_external: zero the external command until released.

    Taken where the robot must not walk on the bridge: before the switch to
    the manipulation export, which has no standing mode, and kept on the way
    back, so only the sticks drive the robot to the next start.  Only Thor's
    release_external lifts it, which run_dreamwaq.py takes only from a
    standing export in standing.  The physical sticks still drive, and a
    latched damping request still stops everything.
    """

    _external_held = False

    def hold_external(self):
        """Latch; True if this call is what latched it."""
        latched = not self._external_held
        self._external_held = True
        return latched

    def release_external(self):
        """Unlatch; True if this call is what released it."""
        released = self._external_held
        self._external_held = False
        return released

    def external_held(self):
        return bool(self._external_held)


class DdsNavCommandSource(ExternalHold):
    """DDS nav input with wireless override and a zero-on-stale watchdog."""

    def __init__(
        self,
        config,
        net,
        timeout=0.3,
        init_channel=True,
        domain_id=0,
        topic=NAV_CMD_TOPIC,
    ):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self.sticks = CommandSource(config)
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.min_cmd = np.asarray(getattr(config, "min_cmd", -self.max_cmd), dtype=np.float32)
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError("cmd_dds_timeout must be finite and positive")
        self.topic = str(topic)
        self._lock = threading.Lock()
        self._latest_cmd = np.zeros(3, dtype=np.float32)
        self._latest_time = 0.0
        self._last_seq = None
        self._last_bad_warn = 0.0
        self._last_gap_warn = 0.0

        if init_channel:
            try:
                ChannelFactoryInitialize(domain_id, net)
            except Exception as exc:
                print(f"[cmd] ChannelFactoryInitialize skipped: {exc}")
        self.sub = ChannelSubscriber(self.topic, String_)
        self.sub.Init()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="nav-cmd-reader",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[cmd] subscribed '{self.topic}' (nav cmd_vel, "
            f"timeout={self.timeout:g}s, sticks always override)"
        )

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                message = self.sub.Read()
            except Exception as exc:
                print(f"[cmd] nav_cmd read failed: {exc}")
                time.sleep(0.1)
                continue
            if message is None:
                continue
            parsed = self._parse(message.data)
            if parsed is None:
                continue
            command, _state = parsed
            with self._lock:
                self._latest_cmd = command
                self._latest_time = time.monotonic()

    def _parse(self, data):
        try:
            payload = json.loads(data)
            command = np.asarray(
                [payload["vx"], payload["vy"], payload["wz"]],
                dtype=np.float32,
            )
        except Exception as exc:
            self._warn_bad(f"unparseable payload {data!r}: {exc}")
            return None
        if command.shape != (3,) or not np.isfinite(command).all():
            self._warn_bad(f"non-finite or malformed command {command.tolist()}")
            return None

        sequence = payload.get("seq")
        if isinstance(sequence, int) and self._last_seq is not None:
            gap = sequence - self._last_seq
            if gap > 1 and time.monotonic() - self._last_gap_warn > 1.0:
                print(
                    f"[cmd] WARN: nav_cmd seq gap detected "
                    f"({gap - 1} message(s) lost, {self._last_seq} -> {sequence})"
                )
                self._last_gap_warn = time.monotonic()
        if isinstance(sequence, int):
            self._last_seq = sequence
        return np.clip(command, self.min_cmd, self.max_cmd), payload.get("state")

    def _warn_bad(self, reason):
        now = time.monotonic()
        if now - self._last_bad_warn > 1.0:
            print(f"[cmd] WARN: dropping bad nav_cmd message: {reason}")
            self._last_bad_warn = now

    def get_command(self, controller):
        if self.sticks.stick_active(controller):
            return self.sticks.get_command(controller)
        if self.external_held():
            return np.zeros(3, dtype=np.float32)
        with self._lock:
            command = self._latest_cmd.copy()
            age = time.monotonic() - self._latest_time
        if age > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return command

    def close(self):
        self._stop.set()
        close = getattr(self.sub, "Close", None)
        if callable(close):
            close()
        self._thread.join(timeout=1.0)


class UdpNavCommandSource(ExternalHold):
    """Thor unicast input with sender filtering and zero-on-stale behavior."""

    # The sender's own explanation of `state`, when it gives one (the ROS
    # bridge does: operator_hold, odometry_stale, command_stale, active).
    # Diagnostic only: nothing here acts on it.
    _latest_reason = None
    _parsed_reason = None

    def __init__(self, config, timeout=None, socket_cls=None):
        self.sticks = CommandSource(config)
        self.max_cmd = np.asarray(config.max_cmd, dtype=np.float32)
        self.min_cmd = np.asarray(getattr(config, "min_cmd", -self.max_cmd), dtype=np.float32)
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
            "[cmd] listening udp://%s:%d for sender %s "
            "(timeout=%gs, physical sticks override navigation input)"
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
            reason = self._parsed_reason
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
                self._latest_state = state
                self._latest_reason = reason
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
        reason = payload.get("reason")
        self._parsed_reason = reason if isinstance(reason, str) else None
        return np.clip(command, self.min_cmd, self.max_cmd), payload.get("state")

    def _warn_bad(self, reason):
        now = time.monotonic()
        if now - self._last_bad_warn > 1.0:
            print("[cmd] WARN: dropping bad UDP nav_cmd message: %s" % reason)
            self._last_bad_warn = now

    def get_command(self, controller):
        if self.sticks.stick_active(controller):
            return self.sticks.get_command(controller)
        if self.external_held():
            return np.zeros(3, dtype=np.float32)
        with self._lock:
            command = self._latest_cmd.copy()
            age = time.monotonic() - self._latest_time
        if age > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return command

    def external_status(self):
        """The latest packet's state and reason and its age (s), for Thor."""
        with self._lock:
            received = self._latest_time
            state = getattr(self, "_latest_state", None)
            reason = self._latest_reason
        if received <= 0.0:
            return {"state": None, "reason": None, "age_s": None}
        return {"state": state, "reason": reason, "age_s": time.monotonic() - received}

    def close(self):
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=1.0)


class RosUdpNavCommandSource(UdpNavCommandSource):
    """Local ROS input with physical-stick priority and a latched damping stop."""

    def __init__(self, config, **kwargs):
        if (config.cmd_udp_bind_host != "127.0.0.1"
                or config.cmd_udp_allowed_host != "127.0.0.1"):
            raise ValueError("ros_udp requires a loopback-only bind and sender")
        self._latest_state = "hold"
        self._damping = threading.Event()
        super().__init__(config, **kwargs)

    def _parse(self, data):
        try:
            payload = json.loads(data)
            if payload.get("state") not in ("autonomy_active", "hold", "damping"):
                return None
        except (ValueError, TypeError, AttributeError, UnicodeError):
            return None
        parsed = super()._parse(data)
        if parsed is None:
            return None
        command, state = parsed
        if state == "damping":
            self._damping.set()
        if state != "autonomy_active":
            command.fill(0.0)
        return command, state

    def damping_requested(self):
        return self._damping.is_set()

    def get_command(self, controller):
        if self.damping_requested():
            return np.zeros(3, dtype=np.float32)
        # The operator must retain manual control when navigation is held,
        # localization is unavailable, or the ROS bridge is not running.
        # NEW3 standing still gates this command in LocomotionModeSwitch.
        if self.sticks.stick_active(controller):
            return self.sticks.get_command(controller)
        if self.external_held():
            return np.zeros(3, dtype=np.float32)
        with self._lock:
            command = self._latest_cmd.copy()
            state = self._latest_state
            age = time.monotonic() - self._latest_time
        if state != "autonomy_active" or age > self.timeout:
            return np.zeros(3, dtype=np.float32)
        return command

    def diagnostics(self, controller):
        """Snapshot the last accepted UDP input and the current source gate."""
        with self._lock:
            udp_command = self._latest_cmd.copy()
            state = self._latest_state
            received_at = self._latest_time
        age_ms = (None if received_at <= 0.0 else
                  max(0.0, (time.monotonic() - received_at) * 1000.0))
        if self.damping_requested():
            selection = "damping"
        elif self.sticks.stick_active(controller):
            selection = "physical_sticks"
        elif age_ms is None:
            selection = "no_udp_packet"
        elif state != "autonomy_active":
            selection = "bridge_%s" % state
        elif age_ms > self.timeout * 1000.0:
            selection = "udp_stale"
        else:
            selection = "udp_active"
        return {
            "udp_command": udp_command,
            "state": state,
            "age_ms": age_ms,
            "selection": selection,
        }


def check_command_damping(commands, *, writer=False):
    """Use the existing real-runtime cleanup path for a latched bridge stop."""
    requested = getattr(commands, "damping_requested", None)
    if callable(requested) and requested() is True:
        if writer:
            raise RuntimeError("ROS autonomy bridge requested damping")
        raise KeyboardInterrupt("ROS autonomy bridge requested damping")


def make_command_source(config, net=None):
    source = getattr(config, "cmd_source", "wireless")
    if source == "ros_udp":
        return RosUdpNavCommandSource(config)
    if source == "udp":
        return UdpNavCommandSource(config)
    if source == "dds":
        return DdsNavCommandSource(
            config,
            net,
            timeout=config.cmd_dds_timeout,
        )
    if source != "wireless":
        raise ValueError(f"unsupported cmd_source: {source!r}")
    return CommandSource(config)
