#!/usr/bin/env python3
"""Verify the live, unmodified G1 teleimager server contract for GR00T."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml
import zmq


ADDON_DIR = Path(__file__).resolve().parent
TELEIMAGER_DIR = ADDON_DIR.parent
SOURCE_DIR = TELEIMAGER_DIR / "src"
if SOURCE_DIR.is_dir() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def live_config(host: str = "127.0.0.1", port: int = 60000) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    socket.setsockopt(zmq.SNDTIMEO, 2000)
    socket.connect(f"tcp://{host}:{port}")
    try:
        socket.send(b"GET_DATA")
        config = socket.recv_json()
    except zmq.error.Again as exc:
        raise RuntimeError(
            f"live teleimager config service did not respond at {host}:{port}"
        ) from exc
    finally:
        socket.close(linger=0)
        context.term()
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid live camera config: {type(config).__name__}")
    return config


def validate_config(config: dict, expected_config: dict | None = None) -> None:
    if expected_config is not None and config != expected_config:
        raise RuntimeError("live camera config differs from the requested config file")
    expected = {
        "head_camera": ((480, 1280), True),
        "left_wrist_camera": ((480, 640), False),
        "right_wrist_camera": ((480, 640), False),
    }
    if expected_config is not None:
        expected = {
            name: (tuple(expected_config[name]["image_shape"]),
                   bool(expected_config[name].get("binocular", False)))
            for name in expected
        }
    for name, (shape, binocular) in expected.items():
        camera = config.get(name)
        if not isinstance(camera, dict):
            raise RuntimeError(f"live config is missing {name}")
        if not camera.get("enable_zmq", False):
            raise RuntimeError(f"{name} enable_zmq is false")
        actual_shape = tuple(int(value) for value in camera.get("image_shape", ()))
        if actual_shape != shape:
            raise RuntimeError(
                f"{name} image_shape={actual_shape}, expected {shape}"
            )
        if bool(camera.get("binocular", False)) != binocular:
            raise RuntimeError(
                f"{name} binocular={camera.get('binocular')}, expected {binocular}"
            )


def wait_for_streams(
    timeout_s: float = 10.0, host: str = "127.0.0.1", port: int = 60000,
) -> dict[str, tuple[tuple[int, ...], float]]:
    from teleimager.image_client import ImageClient

    client = ImageClient(
        host=host,
        request_port=port,
        request_bgr=False,
    )
    getters = {
        "head_camera": client.get_head_frame,
        "left_wrist_camera": client.get_left_wrist_frame,
        "right_wrist_camera": client.get_right_wrist_frame,
    }
    latest: dict[str, tuple[tuple[int, ...], float]] = {}
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline and len(latest) != len(getters):
            for name, getter in getters.items():
                frame = getter()
                jpeg = getattr(frame, "jpg", None) if frame is not None else None
                if not jpeg:
                    continue
                image = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise RuntimeError(f"{name} JPEG decode failed")
                latest[name] = (
                    tuple(int(value) for value in image.shape),
                    float(getattr(frame, "fps", 0.0)),
                )
            time.sleep(0.02)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    missing = sorted(set(getters) - set(latest))
    if missing:
        raise RuntimeError(f"camera streams did not become live: {missing}")
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--request-port", type=int, default=60000)
    parser.add_argument("--config", type=Path, help="Require this exact live config")
    args = parser.parse_args()
    expected = yaml.safe_load(args.config.read_text()) if args.config else None
    config = live_config(args.host, args.request_port)
    validate_config(config, expected)
    print("Live config service: OK")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    streams = wait_for_streams(host=args.host, port=args.request_port)
    for name, (shape, fps) in streams.items():
        expected_shape = tuple(config[name]["image_shape"]) + (3,)
        if shape != expected_shape:
            raise RuntimeError(f"{name}: decoded shape={shape}, expected {expected_shape}")
        print(f"{name}: shape={shape} fps={fps:.1f} OK")
    print("G1 teleimager GR00T camera contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
