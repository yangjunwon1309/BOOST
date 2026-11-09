"""Camera payload and timestamp contract shared by teleop recorders."""

from __future__ import annotations

from typing import Any


def recording_image_payload(frame: Any):
    """Return pixels exactly as the teleop recorder stores them.

    TIM2/legacy frames that carry an encoded JPEG bypass OpenCV re-encoding.
    A decoded BGR array is used only when the source has no encoded payload.
    ``CameraFrame`` in url_groot_deploy calls the encoded field ``jpeg`` while
    teleimager calls it ``jpg``; accepting both keeps one storage contract.
    """
    encoded = getattr(frame, "jpg", None)
    if encoded is None:
        encoded = getattr(frame, "jpeg", None)
    return encoded if encoded is not None else frame.bgr


def recording_camera_timestamp(
    frame: Any,
    action_wall_time_us: int,
) -> dict[str, Any] | None:
    """Build the camera timestamp block used by ``teleop_ikea.py``.

    Legacy ZMQ has no camera capture clock, so it deliberately contributes no
    camera block. TIM2 contributes the established three fields; clock-aligned
    frames also preserve the raw capture clocks and calibration uncertainty.
    """
    capture_us = getattr(frame, "capture_wall_time_us", None)
    if capture_us is None:
        return None
    result = {
        "capture_wall_time_us": capture_us,
        "sequence": getattr(frame, "sequence", None),
        "lag_ms": round((action_wall_time_us - capture_us) / 1000.0, 3),
    }
    source_us = getattr(frame, "source_capture_wall_time_us", None)
    if source_us is not None:
        result.update(
            source_capture_wall_time_us=source_us,
            capture_monotonic_time_ns=frame.capture_monotonic_time_ns,
            clock_uncertainty_us=frame.clock_uncertainty_us,
        )
    return result


def recording_camera_layout(config, head_eye="both"):
    """Describe stored colors; selecting one stereo eye leaves three images."""
    if head_eye not in ("both", "left", "right"):
        raise ValueError(f"unknown recording head eye: {head_eye!r}")
    binocular = bool(config["head_camera"].get("binocular", False))
    if head_eye != "both" and not binocular:
        raise ValueError("--record-head-eye requires a binocular head camera")
    eyes = ("left", "right") if binocular and head_eye == "both" else (head_eye if binocular else None,)
    layout = {f"color_{index}": {"camera": "head_camera", "eye": eye}
              for index, eye in enumerate(eyes)}
    for name, index in (("left_wrist_camera", len(eyes)),
                        ("right_wrist_camera", len(eyes) + 1)):
        camera = config.get(name, {})
        if camera.get("enable_recording_zmq") or camera.get("enable_zmq"):
            layout[f"color_{index}"] = {"camera": name, "eye": None}
    return layout


def recording_camera_subdirs(config, head_eye="both"):
    """Group RGB files by camera; color keys distinguish the two head images."""
    return {
        key: source['camera']
        for key, source in recording_camera_layout(config, head_eye).items()
    }


def recording_camera_images(config, frames, head_eye="both"):
    """Build a complete color row, preserving wrist JPEGs and physical eye identity.

    teleimager rotates the entire side-by-side source before streaming it. At
    180 degrees the physical right eye occupies the left half; at 90/270 the
    eyes are stacked vertically. Cropping here also leaves XR's full frame alone.
    """
    colors = {}
    for key, source in recording_camera_layout(config, head_eye).items():
        name, eye = source["camera"], source["eye"]
        frame = frames.get(name)
        if frame is None:
            raise ValueError(f"{name}: missing recording frame")
        if eye is None:
            payload = recording_image_payload(frame)
        else:
            pixels = frame.bgr
            if pixels is None:
                raise ValueError(f"{name}: cannot decode stereo frame")
            rotation = int(config[name].get("rotation", 0))
            if rotation not in (0, 90, 180, 270):
                raise ValueError(f"{name}: unsupported stereo rotation {rotation}")
            vertical = rotation in (90, 270)
            size = pixels.shape[0 if vertical else 1]
            if size % 2:
                raise ValueError(f"{name}: odd stereo split dimension {size}")
            first = (eye == "left") == (rotation in (0, 90))
            section = slice(0, size // 2) if first else slice(size // 2, size)
            payload = pixels[section, :] if vertical else pixels[:, section]
        if payload is None:
            raise ValueError(f"{name}: empty recording image")
        colors[key] = payload
    return colors


def recording_body_state(qpos, lowstate, require_full=True, motor_slots=35):
    """Keep all G1 joint slots; refuse missing velocities/torques by default."""
    state = {"qpos": qpos, "qvel": (lowstate or {}).get("dq", []),
             "torque": (lowstate or {}).get("tau_est", [])}
    if require_full and any(len(values) != motor_slots for values in state.values()):
        raise ValueError(f"whole-body recording requires {motor_slots} qpos/qvel/torque slots; "
                         "check rt/lowstate or explicitly use --no-lowstate-probe")
    return state
