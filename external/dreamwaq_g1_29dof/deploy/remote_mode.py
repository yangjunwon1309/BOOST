"""Thor's stand-ins for X and Y, and the runtime state Thor needs to see.

The Nature demo on Thor walks the robot to the table, stands it and hands it
to the manipulation export without anyone touching the remote.  The runtime
otherwise takes X and Y only from ``LowState.wireless_remote``, so this is a
second, narrow way in: unicast UDP from one allowed host.

Requests name the state wanted, not a button, so a retried request cannot
toggle the mode twice::

    {"type": "dreamwaq_request", "session": "...", "seq": 3, "action": "walking"}

``action`` is ``status`` (register for status only), ``walking``,
``standing``, ``hold_external`` (zero every external velocity until
``release_external``; the sticks still drive), ``release_external`` (only
from the first export, in standing) or ``switch`` (Y's switch to the second
export, under Y's own checks and only once ``hold_external`` is latched; the
way back is Y's alone).  Each request gets exactly one
acknowledgement, sent again unchanged if the same session and seq arrive
again::

    {"type": "dreamwaq_ack", "session": "...", "seq": 3, "action": "walking",
     "result": "applied" | "already" | "refused" | "unavailable" | "expired",
     "detail": "..."}

A request the control loop has not taken within ``max_age_s`` is answered
``expired`` rather than applied late: a ``walking`` held back while Unitree
owned the robot must not fire when DreamWaQ gets it back.

The control loop publishes a status snapshot; this thread sends it at
``status_hz`` to whoever sent the latest valid request.  All socket I/O stays
on this thread, so the 50-Hz loop only takes and puts small objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import socket
import threading
import time

ACTIONS = ("status", "walking", "standing", "hold_external", "release_external", "switch")
RESULTS = ("applied", "already", "refused", "unavailable", "expired")
DEFAULT_PORT = 15003


@dataclass(frozen=True)
class RemoteRequest:
    session: str
    seq: int
    action: str
    address: tuple
    received_ns: int


def parse_request(data):
    """(session, seq, action) from one datagram, or None if it is not a request."""
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != "dreamwaq_request":
        return None
    session, seq, action = payload.get("session"), payload.get("seq"), payload.get("action")
    if not isinstance(session, str) or not session or len(session) > 128:
        return None
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        return None
    if not isinstance(action, str):
        return None
    return session, seq, action


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _finite(value):
    """Status values cross JSON, which has no NaN; send null instead."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


class RemoteModeServer:
    def __init__(self, bind_host, port, allowed_host, *, status_hz=10.0, max_age_s=0.5,
                 socket_cls=None, clock_ns=time.monotonic_ns, notify=None):
        self.bind_host = str(bind_host)
        self.port = int(port)
        self.allowed_host = str(allowed_host).strip()
        if not 1 <= self.port <= 65535:
            raise ValueError("remote port must be in [1, 65535]")
        if not self.allowed_host:
            raise ValueError("remote allowed host must not be empty")
        if not (math.isfinite(status_hz) and status_hz > 0 and math.isfinite(max_age_s) and max_age_s > 0):
            raise ValueError("remote status rate and request age must be finite and positive")
        self._status_period_ns = int(1e9 / float(status_hz))
        self._max_age_ns = int(float(max_age_s) * 1e9)
        self._clock_ns = clock_ns
        self._notify = notify or (lambda message: None)
        self._lock = threading.Lock()
        self._pending = deque()
        self._outbox = deque()
        # session -> {seq: ack}; only the current session's answers are kept.
        self._session = None
        self._acks = {}
        self._waiting = set()
        self._last_seq = -1
        self._client = None
        self._status = None
        self._status_seq = 0
        self._next_status_ns = 0
        socket_cls = socket.socket if socket_cls is None else socket_cls
        self._socket = socket_cls(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.02)
        self._socket.bind((self.bind_host, self.port))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="dreamwaq-remote", daemon=True)
        self._thread.start()
        self._notify(
            "[remote] Thor requests on udp://%s:%d from %s (walking, standing, hold_external, "
            "release_external, switch); status at %.0f Hz"
            % (self.bind_host, self.port, self.allowed_host, status_hz)
        )

    # -- control loop side ------------------------------------------------ #
    def pending(self):
        """Requests to apply this tick; stale ones are answered ``expired``."""
        now = self._clock_ns()
        fresh = []
        with self._lock:
            while self._pending:
                request = self._pending.popleft()
                if now - request.received_ns > self._max_age_ns:
                    self._resolve_locked(request, "expired", "not taken by the control loop in time")
                else:
                    fresh.append(request)
        return fresh

    def resolve(self, request, result, detail=""):
        if result not in RESULTS:
            raise ValueError("unknown remote result %r" % (result,))
        with self._lock:
            self._resolve_locked(request, result, detail)

    def publish_status(self, status):
        with self._lock:
            self._status = dict(status)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close()

    # -- socket thread side ----------------------------------------------- #
    def _resolve_locked(self, request, result, detail):
        ack = {"type": "dreamwaq_ack", "session": request.session, "seq": request.seq,
               "action": request.action, "result": result, "detail": str(detail)}
        self._waiting.discard((request.session, request.seq))
        if request.session == self._session:
            self._acks[request.seq] = ack
            while len(self._acks) > 64:
                self._acks.pop(min(self._acks))
        self._outbox.append((ack, request.address))
        if request.action != "status":
            self._notify("[remote] %s seq=%d -> %s%s" % (
                request.action, request.seq, result, " (%s)" % detail if detail else ""))

    def handle_datagram(self, data, address):
        """Queue a request, answer a repeat, or drop it; returns what it did."""
        if address[0] != self.allowed_host:
            return "disallowed"
        parsed = parse_request(data)
        if parsed is None:
            return "malformed"
        session, seq, action = parsed
        with self._lock:
            if session != self._session:
                # A restarted demo opens a new session.  The old one's answers
                # go, and nothing it asked for is applied after this point.
                self._session, self._acks, self._last_seq = session, {}, -1
                for stale in [r for r in self._pending if r.session != session]:
                    self._pending.remove(stale)
                    self._resolve_locked(stale, "expired", "superseded by a new session")
            self._client = address
            if seq in self._acks:
                self._outbox.append((self._acks[seq], address))
                return "repeat"
            if (session, seq) in self._waiting:
                return "waiting"
            if seq <= self._last_seq:
                return "old"
            self._last_seq = seq
            request = RemoteRequest(session, seq, action, tuple(address), self._clock_ns())
            if action not in ACTIONS:
                self._resolve_locked(request, "refused", "unknown action; use %s" % ", ".join(ACTIONS))
                return "unknown"
            if action == "status":
                self._resolve_locked(request, "applied", "")
                return "status"
            self._waiting.add((session, seq))
            self._pending.append(request)
            return "queued"

    def _flush(self):
        now = self._clock_ns()
        with self._lock:
            outgoing = list(self._outbox)
            self._outbox.clear()
            if self._client is not None and self._status is not None and now >= self._next_status_ns:
                self._status_seq += 1
                status = dict(self._status, type="dreamwaq_status", seq=self._status_seq)
                outgoing.append((_finite(status), self._client))
                self._next_status_ns = now + self._status_period_ns
        for message, address in outgoing:
            try:
                self._socket.sendto(_json(message), address)
            except (OSError, ValueError) as exc:
                self._notify("[remote] send to %s failed: %s" % (address, exc))

    def _loop(self):
        while not self._stop.is_set():
            try:
                data, address = self._socket.recvfrom(4096)
            except socket.timeout:
                data = None
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.02)
                data = None
            if data is not None:
                self.handle_datagram(data, address)
            self._flush()
