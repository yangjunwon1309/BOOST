#!/usr/bin/env python3
"""Inspect the Galaxy XR manager's local ZMQ wrist/hand stream on Thor."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import zmq

from boost.io.path_transport.zmq_wire import unpack_pose_message


def _last_scalar(fields: dict[str, np.ndarray], name: str, default):
    value = np.asarray(fields.get(name, [default])).reshape(-1)
    return default if not value.size else value[-1].item()


def _wrist(fields: dict[str, np.ndarray], side: str, targets_valid: bool) -> str:
    name = f"wrist_{side}_waist"
    try:
        transform = np.asarray(fields[name], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return "missing"
    if not targets_valid or not np.isfinite(transform).all():
        return "invalid"
    position = transform[:3, 3]
    return "({:+.3f},{:+.3f},{:+.3f})".format(*position)


def _hand(fields: dict[str, np.ndarray], side_index: int, side: str) -> str:
    try:
        valid = np.asarray(fields["hand_landmarks_valid"], dtype=bool).reshape(-1)
        points = np.asarray(fields[f"hand_landmarks_{side}"], dtype=np.float32).reshape(25, 3)
    except (KeyError, TypeError, ValueError):
        return "missing"
    finite_count = int(np.isfinite(points).all(axis=1).sum())
    if valid.size != 2:
        return f"schema-invalid({finite_count}/25)"
    return f"{int(valid[side_index])}({finite_count}/25)"


def _describe(raw: bytes, topic: str) -> dict[str, object]:
    fields = unpack_pose_message(raw, topic=topic)
    targets_valid = bool(_last_scalar(fields, "targets_valid", False))
    source_time = float(_last_scalar(fields, "timestamp_monotonic", 0.0))
    source_age_ms = (
        max(0.0, (time.monotonic() - source_time) * 1000.0)
        if np.isfinite(source_time) and source_time > 0.0
        else None
    )
    return {
        "session": int(_last_scalar(fields, "session_id", -1)),
        "frame": int(_last_scalar(fields, "frame_index", -1)),
        "source_age_ms": source_age_ms,
        "targets_valid": targets_valid,
        "left_wrist": _wrist(fields, "left", targets_valid),
        "right_wrist": _wrist(fields, "right", targets_valid),
        "left_hand": _hand(fields, 0, "left"),
        "right_hand": _hand(fields, 1, "right"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("PATH_XR_HAND_ENDPOINT", "tcp://127.0.0.1:5566"),
        help="Galaxy XR manager ZMQ PUB endpoint (default: Thor localhost:5566)",
    )
    parser.add_argument("--topic", default="vr_pose_debug")
    parser.add_argument("--report-period", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.report_period) or args.report_period <= 0.0:
        parser.error("--report-period must be positive and finite")

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))
    subscriber.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    print(
        f"[XR-stream] SUB {args.endpoint}/{args.topic}; "
        "waiting for Galaxy XR manager frames (Ctrl+C to stop)",
        flush=True,
    )
    last_report = time.monotonic()
    last_receive: float | None = None
    frames = 0
    malformed = 0
    latest: dict[str, object] | None = None
    last_error = ""
    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if subscriber in events:
                raw = subscriber.recv()
                now = time.monotonic()
                frames += 1
                last_receive = now
                try:
                    latest = _describe(raw, args.topic)
                    last_error = ""
                except (KeyError, TypeError, ValueError) as error:
                    malformed += 1
                    message = str(error)
                    if message != last_error:
                        print(f"[XR-stream] malformed frame: {message}", flush=True)
                        last_error = message

            now = time.monotonic()
            elapsed = now - last_report
            if elapsed >= args.report_period:
                rate = frames / max(elapsed, 1e-6)
                if latest is None:
                    state = "no frames received"
                else:
                    age = latest["source_age_ms"]
                    age_text = "n/a" if age is None else f"{age:.1f}ms"
                    state = (
                        f"session={latest['session']} frame={latest['frame']} "
                        f"source_age={age_text} targets_valid={int(latest['targets_valid'])} "
                        f"wrist_L={latest['left_wrist']} wrist_R={latest['right_wrist']} "
                        f"hand_valid_L={latest['left_hand']} hand_valid_R={latest['right_hand']}"
                    )
                receive_age = (
                    "never"
                    if last_receive is None
                    else f"{(now - last_receive) * 1000.0:.0f}ms"
                )
                print(
                    f"[XR-stream] rate={rate:.1f}Hz total={frames} "
                    f"last_rx={receive_age} malformed={malformed} {state}",
                    flush=True,
                )
                frames = 0
                last_report = now
    except KeyboardInterrupt:
        return 0
    finally:
        poller.unregister(subscriber)
        subscriber.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
