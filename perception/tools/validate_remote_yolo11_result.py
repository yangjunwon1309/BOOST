#!/usr/bin/env python3
"""Validate remote YOLO11 perception results on the LCM result channel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lcm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.protocol import (  # noqa: E402
    REMOTE_RESULT_CHANNEL,
    ValidationContext,
    decode_mask,
    unpack_result,
    validate_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--result-channel", default=REMOTE_RESULT_CHANNEL)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--result-max-age-ms", type=float, default=1500.0)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    bus = lcm.LCM(args.lcm_url)
    accepted = 0
    rejected = 0
    last_sequence: int | None = None
    deadline = time.monotonic() + max(0.1, float(args.timeout_s))

    def on_result(_channel: str, data: bytes) -> None:
        nonlocal accepted, rejected, last_sequence
        try:
            payload = unpack_result(data)
            errors = validate_result(
                payload,
                ValidationContext(
                    result_max_age_ms=args.result_max_age_ms,
                    previous_sequence=last_sequence,
                ),
            )
            if payload.get("backend") != "yolo11_pytorch":
                errors.append(f"backend is {payload.get('backend')!r}, expected 'yolo11_pytorch'")
            objects = payload.get("objects", [])
            if not args.allow_empty and isinstance(objects, list) and not objects:
                errors.append("objects is empty; pass --allow-empty for no-detection frames")
            if errors:
                rejected += 1
                print(
                    "[remote-yolo11-validate] rejected "
                    f"seq={payload.get('sequence')} errors={errors}",
                    flush=True,
                )
                return
            labels = []
            contact_count = 0
            mask_pixels = []
            for obj in objects:
                labels.append(str(obj.get("label")))
                contact_count += len(obj.get("contact_candidates") or [])
                mask = decode_mask(
                    obj["mask_data"],
                    int(obj["mask_height"]),
                    int(obj["mask_width"]),
                )
                mask_pixels.append(int(mask.sum()))
            accepted += 1
            last_sequence = int(payload["sequence"])
            print(
                "[remote-yolo11-validate] accepted "
                f"seq={payload['sequence']} "
                f"objects={len(objects)} "
                f"labels={labels[:8]} "
                f"mask_pixels={mask_pixels[:8]} "
                f"contacts={contact_count} "
                f"inference_ms={payload.get('inference_ms')}",
                flush=True,
            )
        except Exception as error:
            rejected += 1
            print(f"[remote-yolo11-validate] decode rejected: {error}", flush=True)

    bus.subscribe(args.result_channel, on_result)
    while accepted < args.count and time.monotonic() < deadline:
        bus.handle_timeout(200)
    print(
        "[remote-yolo11-validate] done "
        f"accepted={accepted} rejected={rejected}",
        flush=True,
    )
    return 0 if accepted >= args.count and rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
