"""Terminal teleop keys with a quit gesture confirmed by actual key repeats."""

import math
import os
import select
import sys
import termios
import time
import tty


class KeyHoldTracker:
    """Recognize key presses using caller-supplied monotonic timestamps.

    Terminals supply characters rather than key-up events. Silence infers a
    release, but can only produce a short q. A long q requires received repeat
    characters spanning the full hold duration; waiting after a tap never quits.
    Call feed and poll from the same input thread.
    """

    def __init__(self, hold_seconds=3.0, first_repeat_wait=0.75, repeat_gap=0.20):
        for name, value in (("hold_seconds", hold_seconds),
                            ("first_repeat_wait", first_repeat_wait),
                            ("repeat_gap", repeat_gap)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.hold_seconds = hold_seconds
        self.first_repeat_wait = first_repeat_wait
        self.repeat_gap = repeat_gap
        self._key = None
        self._first = self._last = 0.0
        self._repeated = False
        self._long_sent = False

    def _release(self):
        events = [("short_q", None)] if self._key == "q" and not self._long_sent else []
        self._key = None
        return events

    def poll(self, now):
        """Infer release after silence, without ever producing a long q."""
        gap = self.repeat_gap if self._repeated else self.first_repeat_wait
        if self._key is not None and now - self._last > gap:
            return self._release()
        return []

    def feed(self, key, now):
        """Return ordered (kind, key) events for one received character."""
        key = key.lower()
        events = self.poll(now)
        if key != self._key:
            events.extend(self._release())
            if key not in ("r", "s", "c", "q", "w") and not key.isdigit():
                return events
            self._key = key
            self._first = self._last = now
            self._repeated = self._long_sent = False
            if key != "q":
                events.append(("press", key))
        else:
            self._last = now
            self._repeated = True
            if key == "q" and not self._long_sent and now - self._first >= self.hold_seconds:
                self._long_sent = True
                events.append(("long_q", None))
        return events


def listen_teleop_keys(on_press, on_short_q, on_long_q, stop_event, *, stream=None):
    """Read a Linux TTY until stopped or EOF, restoring its settings on exit.

    Callbacks run sequentially in this listener's thread and should only update
    control state. on_press receives r/s/c/w and digits; the two q callbacks
    take no argument.
    Cbreak mode keeps ISIG enabled, so Ctrl+C retains its usual signal behavior.
    The optional stream argument permits isolated PTY tests.
    """
    stream = sys.stdin if stream is None else stream
    fd = stream.fileno()
    if not os.isatty(fd):
        raise ValueError("Teleop keyboard input requires a TTY")
    original = termios.tcgetattr(fd)
    tracker = KeyHoldTracker()
    escape_mode = None
    escape_started = 0.0

    def dispatch(events):
        for kind, key in events:
            if stop_event.is_set():
                return
            if kind == "press":
                on_press(key)
            elif kind == "short_q":
                on_short_q()
            elif kind == "long_q":
                on_long_q()

    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            readable, _, _ = select.select([fd], [], [], 0.02)
            now = time.monotonic()
            if readable:
                data = os.read(fd, 64)
                if not data:
                    break
                for value in data:
                    if stop_event.is_set():
                        break
                    # Do not turn the C in an arrow's ESC [ C into recenter.
                    if escape_mode is not None and now - escape_started > 0.1:
                        escape_mode = None
                    if escape_mode == "esc":
                        escape_mode = "sequence" if value in (ord("["), ord("O")) else None
                        continue
                    if escape_mode == "sequence":
                        if 0x40 <= value <= 0x7E:
                            escape_mode = None
                        continue
                    if value == 0x1B:
                        dispatch(tracker.feed("esc", now))
                        escape_mode, escape_started = "esc", now
                    else:
                        dispatch(tracker.feed(chr(value), now))
            dispatch(tracker.poll(time.monotonic()))
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)
