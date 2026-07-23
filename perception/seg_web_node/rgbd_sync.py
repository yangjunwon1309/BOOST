"""Timestamp pairing for robot-monotonic RGB and aligned depth frames."""

from __future__ import annotations

import collections
import dataclasses

import cv2

from .lcm_protocol import CameraFrame, DepthFrame


@dataclasses.dataclass(frozen=True)
class RgbdPair:
    rgb: CameraFrame
    depth: DepthFrame
    delta_us: int


class RgbdSynchronizer:
    def __init__(self, tolerance_ms: float = 50.0, capacity: int = 30):
        if tolerance_ms < 0 or capacity <= 0:
            raise ValueError("invalid synchronizer settings")
        self.tolerance_us = round(tolerance_ms * 1000)
        self.rgb = collections.deque(maxlen=capacity)
        self.depth = collections.deque(maxlen=capacity)
        self.matched = 0
        self.dropped_rgb = 0
        self.dropped_depth = 0

    def _match(self) -> RgbdPair | None:
        if not self.rgb or not self.depth:
            return None
        best = None
        for rgb_index, rgb in enumerate(self.rgb):
            for depth_index, depth in enumerate(self.depth):
                delta = abs(rgb.timestamp_us - depth.timestamp_us)
                candidate = (delta, rgb_index, depth_index, rgb, depth)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        delta, rgb_index, depth_index, rgb, depth = best
        if delta > self.tolerance_us:
            return None
        del self.rgb[rgb_index]
        del self.depth[depth_index]
        self.matched += 1
        return RgbdPair(rgb, depth, delta)

    def add_rgb(self, frame: CameraFrame) -> RgbdPair | None:
        if len(self.rgb) == self.rgb.maxlen:
            self.dropped_rgb += 1
        self.rgb.append(frame)
        return self._match()

    def add_depth(self, frame: DepthFrame) -> RgbdPair | None:
        if len(self.depth) == self.depth.maxlen:
            self.dropped_depth += 1
        self.depth.append(frame)
        return self._match()


def depth_at_rgb_size(pair: RgbdPair):
    if pair.depth.width == pair.rgb.width and pair.depth.height == pair.rgb.height:
        return pair.depth.depth
    return cv2.resize(
        pair.depth.depth,
        (pair.rgb.width, pair.rgb.height),
        interpolation=cv2.INTER_NEAREST,
    )
