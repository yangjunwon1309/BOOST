"""Monotonic loop pacing without accumulating OS sleep overshoot."""

import math
import time


class LoopRate:
    def __init__(self, frequency, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("Loop frequency must be positive and finite")
        self.period = 1.0 / frequency
        self.clock, self._sleep = clock, sleep
        self.reset()

    def reset(self):
        self.deadline = self.clock()

    def sleep(self):
        self.deadline += self.period
        now = self.clock()
        if now - self.deadline >= self.period:
            # Discard missed ticks after a pause/stall instead of replaying them.
            self.deadline = now
        delay = max(0.0, self.deadline - now)
        if delay:
            self._sleep(delay)
        return delay
