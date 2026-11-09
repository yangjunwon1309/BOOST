"""Validate one WebXR hand before changing shared tracking state."""

from typing import NamedTuple

import numpy as np


class HandFrame(NamedTuple):
    wrist: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray


def decode_hand_frame(raw) -> HandFrame:
    # msgpackr encodes WebXR's undefined pose as ExtType(0, b"\x00").
    # Inspect its structure so this pure decoder need not import msgpack.
    if raw is None or (getattr(raw, "code", None) == 0
                       and getattr(raw, "data", None) == b"\x00"):
        raise ValueError("hand pose unavailable (WebXR tracking lost)")
    values = np.asarray(raw, dtype=np.float64)
    if values.shape not in ((400,), (25, 16)):
        raise ValueError(f"expected 25 joint matrices / 400 values, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("hand pose contains non-finite values")
    joints = values.reshape(25, 16)
    wrist = joints[0].reshape(4, 4, order="F")
    rotation = wrist[:3, :3]
    if (not np.allclose(wrist[3], [0, 0, 0, 1], atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=0.05)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.05)):
        raise ValueError("hand wrist is not a valid rigid transform")
    return HandFrame(
        joints[0].copy(),
        joints[:, 12:15].reshape(-1).copy(),
        joints[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]].reshape(-1).copy(),
    )


def payload_length(raw) -> int:
    try:
        return len(raw)
    except TypeError:
        return 0


def hand_gesture_defaults(frame: HandFrame) -> dict:
    """Match Vuer's gesture distances when optional state metadata is absent.

    Pinch is thumb-tip to index-tip distance, in metres. A zero placeholder
    would mean a closed Dex1 gripper, so derive it from the received geometry.
    """
    points = frame.positions.reshape(25, 3)
    pinch = float(np.linalg.norm(points[4] - points[9]))
    squeeze = float(np.linalg.norm(points[14] - points[11]))
    return {"pinch": pinch < 0.01, "pinchValue": pinch,
            "squeeze": squeeze < 0.07, "squeezeValue": squeeze}


HAND_STAT_FIELDS = (
    "event_count", "valid_count", "error_count",
    "last_event_monotonic_ns", "last_valid_monotonic_ns",
    "left_length", "right_length", "left_valid", "right_valid",
)
