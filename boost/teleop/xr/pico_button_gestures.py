"""Discrete PICO button chords used by PATH input publishers."""

from __future__ import annotations

from typing import Sequence


class PicoButtonGestureInterpreter:
    """Resolve arm tracking, belief selection, and recorder button events."""

    def __init__(self) -> None:
        self.previous = (False, False, False, False)
        self.x_active = False
        self.x_chorded = False

    def update(self, buttons: Sequence[bool]) -> list[str]:
        a, b, x, y = (bool(value) for value in buttons)
        pa, pb, px, py = self.previous

        if x and not px:
            self.x_active = True
            self.x_chorded = bool(a or b or y)
        if x and (a or b or y):
            self.x_chorded = True

        events: list[str] = []
        if y and not py and not (a or b or x):
            events.append("select_left")
        if b and not pb and not (a or x or y):
            events.append("select_right")

        start_combo = a and b and x and y
        previous_start_combo = pa and pb and px and py
        if start_combo and not previous_start_combo:
            events.append("start_combo")
        elif a and x and not (pa and px):
            events.append("ax")
        elif b and y and not (pb and py):
            events.append("by")

        if px and not x:
            if self.x_active and not self.x_chorded:
                events.append("record_toggle")
            self.x_active = False
            self.x_chorded = False

        if (a, b, x, y) != self.previous:
            events.append("button_state")
        self.previous = (a, b, x, y)
        return events
