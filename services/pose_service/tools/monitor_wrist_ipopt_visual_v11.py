#!/usr/bin/env python3
"""Verify that this laptop receives the current wrist IPOPT targets over Wi-Fi LCM.

The publisher runs in the PATH state-machine bridge.  This monitor subscribes
directly to its diagnostic channel, so it remains useful even when the wrist
webview is stopped or cannot render a marker.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    import lcm
except ModuleNotFoundError:  # Keep --help usable outside the deployment image.
    lcm = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep the monitor's help and dependency error readable outside Pixi/Docker.
WRIST_IPOPT_VISUAL_CHANNEL = "PATH_V11_WRIST_IPOPT_VISUAL"


def _monotonic_delta_ms(later_us: Any, earlier_us: Any) -> str:
    try:
        delta = (int(later_us) - int(earlier_us)) / 1_000.0
    except (TypeError, ValueError):
        return "n/a"
    return f"{delta:.1f}"


def _format_target(hands: Any, side: str) -> str:
    hand = hands.get(side) if isinstance(hands, dict) else None
    target = hand.get("ipopt_wrist_target_m") if isinstance(hand, dict) else None
    try:
        point = [float(value) for value in target]
    except (TypeError, ValueError):
        return "missing"
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        return "invalid"
    return "[" + ", ".join(f"{value:+.4f}" for value in point) + "] m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wifi-lcm-url",
        required=True,
        help="Wi-Fi multicast URL used by both Thor and this laptop.",
    )
    parser.add_argument(
        "--channel",
        default=WRIST_IPOPT_VISUAL_CHANNEL,
        help=f"LCM channel to monitor (default: {WRIST_IPOPT_VISUAL_CHANNEL}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of valid packets required for success (default: 3).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=5.0,
        help="Total receive timeout in non-watch mode (default: 5).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Print packets continuously until Ctrl-C.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.timeout_sec <= 0.0:
        raise SystemExit("--timeout-sec must be positive")
    if lcm is None:
        raise SystemExit(
            "python-lcm is required. Run this inside the laptop perception "
            "environment/container that runs the wrist webview."
        )
    try:
        from remote_offload.wrist_d405_protocol_v11 import unpack_ipopt_visual
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error.name} is required. From the BOOST repository root run "
            "`pixi run --locked python services/pose_service/tools/"
            "monitor_wrist_ipopt_visual_v11.py ...`."
        ) from error

    bus = lcm.LCM(args.wifi_lcm_url)
    received = 0
    rejected = 0
    last_received_us: int | None = None

    print(
        f"[ipopt-visual] listening channel={args.channel} url={args.wifi_lcm_url}",
        flush=True,
    )

    def on_visual(_channel: str, data: bytes) -> None:
        nonlocal received, rejected, last_received_us
        received_us = time.time_ns() // 1_000
        try:
            payload = unpack_ipopt_visual(data)
        except Exception as error:
            rejected += 1
            print(f"[ipopt-visual] rejected packet: {error}", file=sys.stderr, flush=True)
            return

        received += 1
        interval = (
            "first"
            if last_received_us is None
            else f"{(received_us - last_received_us) / 1_000.0:.1f} ms"
        )
        last_received_us = received_us
        hands = payload.get("hands", {})
        print(
            "[ipopt-visual] "
            f"packet={received} state={payload.get('state', 'unknown')} "
            # PATH timestamps are monotonic-clock values from Thor. They cannot
            # be compared to the laptop wall clock, but their same-host delta
            # exposes any bridge-side delay before the LCM publish.
            f"bridge_publish_delta={_monotonic_delta_ms(payload.get('published_timestamp_us'), payload.get('source_timestamp_us'))} ms "
            f"interval={interval} "
            f"left={_format_target(hands, 'left')} "
            f"right={_format_target(hands, 'right')}",
            flush=True,
        )

    bus.subscribe(args.channel, on_visual)
    deadline = None if args.watch else time.monotonic() + float(args.timeout_sec)
    try:
        while args.watch or received < args.count:
            if deadline is not None and time.monotonic() >= deadline:
                break
            bus.handle_timeout(100)
    except KeyboardInterrupt:
        print("\n[ipopt-visual] stopped", flush=True)
        return 0 if received else 1

    if received < args.count:
        print(
            f"[ipopt-visual] FAILED: received {received}/{args.count} valid packet(s) "
            f"within {args.timeout_sec:.1f}s (rejected={rejected}).",
            file=sys.stderr,
        )
        return 1
    print(
        f"[ipopt-visual] OK: received {received} valid packet(s) "
        f"(rejected={rejected}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
