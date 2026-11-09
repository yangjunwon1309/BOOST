"""Heads-up notices drawn inside the XR headset during teleoperation.

Why this exists
---------------
In ``--display-mode pass-through`` the headset shows the real world and no robot
image is streamed to it, so the person wearing it cannot see the terminal where
the recorder prints its state.  Episodes get started, saved -- or never started
at all -- without the operator noticing, and a teleimager server that dies mid
session looks exactly like one that is fine.  This module draws a small banner
into the XR scene instead: recording state on one channel, faults on another.

Placement
---------
The banner is a small camera-facing plane, lifted by ``position`` -- which Vuer
applies as a camera-space offset on top of ``distanceToCamera`` whenever the
plane is not ``fixed``.  So the geometry is stated the way the operator
experiences it, in degrees above the line of sight, and the plane is only as
large as the banner itself rather than a view-sized sheet that is transparent
almost everywhere.

Split of responsibility
-----------------------
``XRNotifier`` owns both the text and the drawing; ``televuer`` only asks it for
a frame and upserts whatever comes back.  That keeps the vendored televuer
package free of any import back into ``teleop`` -- it never learns what a level
or an episode is, and the banner layout can change without touching it.

The notice is published through ``multiprocessing`` shared memory because
televuer runs its Vuer app in a forked child process.  An ``XRNotifier`` must
therefore be constructed *before* ``TeleVuerWrapper``, so the child inherits it.

Text is drawn with OpenCV's Hershey fonts, which are ASCII-only, so non-ASCII
characters are replaced rather than drawn as boxes.  Keep notice text ASCII.
"""

import math
import time
from collections import namedtuple
from multiprocessing import Array, Value

import cv2
import numpy as np

LEVEL_INFO = 0
LEVEL_RECORDING = 1
LEVEL_GOOD = 2
LEVEL_WARN = 3
LEVEL_ERROR = 4

LINK_LOST = "lost"
LINK_RESTORED = "restored"

# Long enough for "REC 0012 | 3600 frames | 120.0 s | leg insert", short enough
# that the shared buffer stays trivial.
TEXT_CAPACITY = 240

# (background, accent) as RGB. televuer hands the array to Vuer as RGB -- its
# own image path runs BGR2RGB before writing the display buffer -- so these are
# authored in RGB even though cv2 draws them.
_PALETTE = {
    LEVEL_INFO:      ((30, 34, 42), (150, 162, 178)),
    LEVEL_RECORDING: ((44, 20, 22), (236, 64, 52)),
    LEVEL_GOOD:      ((20, 40, 28), (64, 202, 110)),
    LEVEL_WARN:      ((48, 38, 14), (242, 178, 32)),
    LEVEL_ERROR:     ((54, 18, 18), (250, 74, 60)),
}
_TEXT_RGB = (246, 248, 252)
_FONT = cv2.FONT_HERSHEY_DUPLEX
_MIN_TEXT_SCALE = 0.45

# clear=True asks the renderer to drop the element; otherwise rgb/alpha/props
# are everything televuer needs to build one ImageBackground.
XRNoticeFrame = namedtuple("XRNoticeFrame", ("clear", "rgb", "alpha", "props"))


def _ascii(text):
    """Hershey fonts have no glyphs outside ASCII; make that visible, not silent."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _text_width(text, scale, thickness):
    (width, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return width


def _ellipsize(text, scale, thickness, max_width):
    if _text_width(text, scale, thickness) <= max_width:
        return text
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut] + "..."
        if _text_width(candidate, scale, thickness) <= max_width:
            return candidate
    return "..."


def render_banner(text, level, width=1018, height=160, margin=0.05):
    """Draw one notice, returning (rgb HxWx3, alpha HxW).

    The canvas is the banner, so the plane it goes on is only banner-sized. The
    margin is left fully transparent so the panel does not read as a hard-edged
    rectangle pasted over the room, and the panel itself is translucent while
    the accent bar stays opaque.
    """
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.uint8)

    y0 = int(height * margin)
    y1 = height - y0
    x0 = int(width * margin * height / width)      # equal margin in pixels, not in fractions
    x1 = width - x0
    background, accent = _PALETTE.get(level, _PALETTE[LEVEL_INFO])

    cv2.rectangle(rgb, (x0, y0), (x1, y1), background, -1)
    # Not fully opaque: a slightly translucent panel still reads clearly while
    # letting the operator see what is behind it.
    cv2.rectangle(alpha, (x0, y0), (x1, y1), 226, -1)

    bar = max(6, int(height * 0.075))
    cv2.rectangle(rgb, (x0, y0), (x0 + bar, y1), accent, -1)
    cv2.rectangle(alpha, (x0, y0), (x0 + bar, y1), 255, -1)

    pad = int(height * 0.22)
    _draw_fitted_text(rgb, _ascii(text), x0 + bar + pad, y0, x1 - pad, y1)
    return rgb, alpha


def _draw_fitted_text(canvas, text, x_left, y_top, x_right, y_bottom):
    box_width = max(1, x_right - x_left)
    box_height = max(1, y_bottom - y_top)

    scale = box_height / 52.0
    thickness = max(1, int(round(scale * 1.4)))
    while scale > _MIN_TEXT_SCALE and _text_width(text, scale, thickness) > box_width:
        scale -= 0.05
        thickness = max(1, int(round(scale * 1.4)))
    text = _ellipsize(text, scale, thickness, box_width)

    (_, text_height), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    baseline_y = y_top + (box_height + text_height) // 2
    cv2.putText(canvas, text, (x_left, baseline_y), _FONT, scale,
                _TEXT_RGB, thickness, cv2.LINE_AA)


class _Channel:
    """One notice slot, written in the teleop process and read in the Vuer child."""

    def __init__(self):
        self._text = Array("c", TEXT_CAPACITY, lock=True)
        self._level = Value("i", LEVEL_INFO, lock=True)
        # 0 means the notice is sticky; anything else is a CLOCK_MONOTONIC
        # deadline, which is system-wide on Linux and so valid across the fork.
        self._expire_ns = Value("q", 0, lock=True)

    def set(self, text, level, ttl_s=None):
        payload = _ascii(text)[:TEXT_CAPACITY - 1].encode("ascii")
        expire = 0 if ttl_s is None else time.monotonic_ns() + int(ttl_s * 1e9)
        with self._text.get_lock():
            self._text.value = payload
            self._level.value = int(level)
            self._expire_ns.value = expire

    def clear(self):
        with self._text.get_lock():
            self._text.value = b""
            self._expire_ns.value = 0

    def read(self, now_ns):
        with self._text.get_lock():
            payload = self._text.value
            level = self._level.value
            expire = self._expire_ns.value
        if not payload:
            return None
        if expire and now_ns >= expire:
            return None
        return payload.decode("ascii"), level


class XRNotifier:
    """Two-channel notice board for the headset.

    ``status`` is the sticky line -- what the recorder is doing right now.
    ``alert`` is the louder one and wins whenever it is set, which is what makes
    a dead camera link cover the recording indicator instead of racing with it.
    """

    def __init__(self, element_key="teleop-notice", elevation_deg=22.0,
                 banner_deg=5.5, width_deg=34.0, distance_to_camera=1.0,
                 canvas_height=160):
        """Geometry is given in degrees of the operator's field of view.

        elevation_deg is where the banner sits above the line of sight. It has
        to clear the working area -- manipulation happens at and below gaze
        centre -- while staying close enough to read with a glance rather than
        a head tilt. banner_deg and width_deg are its angular size there.
        """
        self.element_key = element_key
        self.elevation_deg = float(elevation_deg)
        self.banner_deg = float(banner_deg)
        self.width_deg = float(width_deg)
        self.distance_to_camera = float(distance_to_camera)

        # Vuer offsets the plane in camera space, so the banner centre lands at
        # (0, y, -d): further from the eye than d once it is lifted, which is
        # what eye_distance corrects for when sizing it.
        elevation = math.radians(self.elevation_deg)
        self.offset_y = self.distance_to_camera * math.tan(elevation)
        eye_distance = self.distance_to_camera / math.cos(elevation)
        self.plane_height = 2 * eye_distance * math.tan(math.radians(self.banner_deg) / 2)
        self.plane_width = 2 * eye_distance * math.tan(math.radians(self.width_deg) / 2)
        self.aspect = self.plane_width / self.plane_height

        self.height = int(canvas_height)
        self.width = max(1, int(round(self.height * self.aspect)))

        self._status = _Channel()
        self._alert = _Channel()
        # Render cache. The fork copies it, so each process tracks what its own
        # session has already been shown; it is never shared state. It starts at
        # None, matching an empty notice board, so a session that never gets a
        # notice never sends a remove for an element it never created.
        self._shown = None

    # ---------------- producer side (teleop process) ----------------

    def status(self, text, level=LEVEL_INFO):
        """Set the sticky line. Replaces whatever was there."""
        self._status.set(text, level, None)

    def clear_status(self):
        self._status.clear()

    def alert(self, text, level=LEVEL_WARN, ttl_s=4.0):
        """Raise a notice over the status line. ttl_s=None keeps it until cleared."""
        self._alert.set(text, level, ttl_s)

    def clear_alert(self):
        self._alert.clear()

    def clear(self):
        self.clear_alert()
        self.clear_status()

    # ---------------- renderer side (Vuer child process) ----------------

    def current(self, now_ns=None):
        """(text, level) that should be on screen, or None. Honours TTL and priority."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        alert = self._alert.read(now)
        return alert if alert is not None else self._status.read(now)

    def poll_render(self, now_ns=None):
        """Return an XRNoticeFrame when the screen must change, else None.

        Returning None on the unchanged case is what keeps this cheap enough to
        call at display rate: the banner is only encoded when its text or level
        actually moves, not once per frame.
        """
        notice = self.current(now_ns)
        if notice == self._shown:
            return None
        self._shown = notice
        if notice is None:
            return XRNoticeFrame(True, None, None, {"key": self.element_key})

        text, level = notice
        rgb, alpha = render_banner(text, level, self.width, self.height)
        ok, encoded = cv2.imencode(".png", alpha)
        if not ok:
            return None
        props = {
            "key": self.element_key,
            "aspect": self.aspect,
            "height": self.plane_height,
            "distanceToCamera": self.distance_to_camera,
            # Camera-space offset, applied on top of distanceToCamera. Only
            # honoured while the plane is not "fixed", which is the default and
            # is also what keeps it following the head.
            "position": [0.0, self.offset_y, 0.0],
            "interpolate": True,
            "format": "jpeg",
            "quality": 92,
        }
        return XRNoticeFrame(False, rgb, encoded.tobytes(), props)


class StreamHealthMonitor:
    """Turns "did a frame arrive recently" into link-down / link-up edges.

    Two frame flavours reach this class and they fail differently.  teleimager's
    ``TeleImage`` goes falsy as soon as its SUB socket stops receiving, so a
    truthy object means the frame is fresh *now*.  ``RecordingImageClient``'s
    ``RecordingFrame`` keeps handing back the last frame forever, so freshness
    has to come from its ``receive_monotonic_ns`` stamp instead.
    """

    def __init__(self, stale_after_s=1.5):
        self.stale_after_ns = int(stale_after_s * 1e9)
        self._last_rx_ns = None
        self._down = False

    @property
    def is_down(self):
        return self._down

    def down_for_s(self, now_ns=None):
        if self._last_rx_ns is None:
            return 0.0
        now = time.monotonic_ns() if now_ns is None else now_ns
        return max(0.0, (now - self._last_rx_ns) / 1e9)

    def update(self, frame, now_ns=None):
        """Feed the newest frame object; returns LINK_LOST/LINK_RESTORED on an edge."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        received = self._receive_ns(frame, now)
        if received is not None:
            self._last_rx_ns = received
        if self._last_rx_ns is None:
            # First poll before anything has arrived: start the clock here so a
            # slow first frame is not reported as a link that went down.
            self._last_rx_ns = now
            return None

        stale = (now - self._last_rx_ns) > self.stale_after_ns
        if stale and not self._down:
            self._down = True
            return LINK_LOST
        if not stale and self._down:
            self._down = False
            return LINK_RESTORED
        return None

    @staticmethod
    def _receive_ns(frame, now):
        if frame is None:
            return None
        stamp = getattr(frame, "receive_monotonic_ns", None)
        if stamp is not None:
            return int(stamp)
        return now if frame else None
