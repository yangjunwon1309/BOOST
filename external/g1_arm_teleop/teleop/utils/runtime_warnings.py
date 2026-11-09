"""Operator diagnostics that never request a robot stop or change commands."""

import threading
import time


class RuntimeWarnings:
    def __init__(self, logger, interval_s=5.0):
        self.logger = logger
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._active = {}
        self._last_log = {}

    def report(self, source, message, *, log=True):
        message = str(message)[:400]
        now = time.monotonic()
        with self._lock:
            changed = source not in self._active
            self._active[source] = message
            emit = log and (changed or now - self._last_log.get(source, float('-inf')) >= self.interval_s)
            if emit:
                self._last_log[source] = now
        if emit:
            try:
                self.logger.warning('%s: %s; continuing teleoperation with the existing control authority.', source, message)
            except Exception:
                # Losing a log sink must not become a second control failure.
                pass

    def clear(self, source):
        with self._lock:
            self._active.pop(source, None)

    def messages(self):
        with self._lock:
            return [f'{source}: {message}' for source, message in self._active.items()]
