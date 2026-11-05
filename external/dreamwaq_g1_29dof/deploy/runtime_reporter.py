"""Non-blocking runtime diagnostics for the real-robot control path.

The 50-Hz parent and 250-Hz writer must never wait for an SSH terminal.  A
bounded queue hands messages to this daemon; it writes the persistent log first
and mirrors to ``/dev/tty`` with a non-blocking file descriptor when available.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import termios


class RuntimeReporter:
    def __init__(self, path: str, *, queue_size: int = 128) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop = threading.Event()
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run,
            name="dreamwaq-runtime-reporter",
            daemon=True,
        )
        self._thread.start()

    def emit(self, message: object, *, level: str = "INFO") -> None:
        """Queue one diagnostic without ever blocking a control thread."""

        line = "%s monotonic_ns=%d %s %s\n" % (
            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            time.monotonic_ns(),
            str(level).upper(),
            str(message).replace("\n", " | "),
        )
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._dropped += 1
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                return

    def close(self, timeout_s: float = 0.25) -> None:
        if self._dropped:
            self.emit(
                "runtime reporter dropped %d old messages" % self._dropped,
                level="WARNING",
            )
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(timeout_s)))
        # If the PTY was accidentally flow-stopped (Ctrl+S), resume it so the
        # returning shell prompt and Ctrl+C cleanup remain visible.
        tty_fd = None
        try:
            tty_fd = os.open(
                "/dev/tty", os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC
            )
            termios.tcflow(tty_fd, termios.TCOON)
        except (OSError, termios.error):
            pass
        finally:
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except OSError:
                    pass

    def _run(self) -> None:
        log_fd = None
        tty_fd = None
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            log_fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC,
                0o644,
            )
            try:
                tty_fd = os.open(
                    "/dev/tty",
                    os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC,
                )
            except OSError:
                tty_fd = None

            while not self._stop.is_set() or not self._queue.empty():
                try:
                    line = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                payload = line.encode("utf-8", errors="replace")
                try:
                    os.write(log_fd, payload)
                except OSError:
                    pass
                if tty_fd is not None:
                    try:
                        os.write(tty_fd, payload)
                    except (BlockingIOError, OSError):
                        # A stalled SSH/PTY must be invisible to control timing.
                        pass
        finally:
            for fd in (tty_fd, log_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass


__all__ = ["RuntimeReporter"]
