#!/usr/bin/env python3
"""Print remote perception LCM heartbeats and result summaries."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lcm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from remote_offload.protocol import (  # noqa: E402
    REMOTE_HEARTBEAT_CHANNEL,
    REMOTE_RESULT_CHANNEL,
    unpack_heartbeat,
    unpack_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", required=True)
    parser.add_argument("--heartbeat-channel", default=REMOTE_HEARTBEAT_CHANNEL)
    parser.add_argument("--result-channel", default=REMOTE_RESULT_CHANNEL)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    bus = lcm.LCM(args.lcm_url)
    started = time.monotonic()
    counts = {"heartbeat": 0, "result": 0}

    def on_heartbeat(_channel: str, payload: bytes) -> None:
        counts["heartbeat"] += 1
        heartbeat = unpack_heartbeat(payload)
        print(
            "[remote-monitor] heartbeat "
            f"model_ready={heartbeat.get('model_ready')} "
            f"backend={heartbeat.get('backend')} "
            f"input={heartbeat.get('input_frame_count')} "
            f"processed={heartbeat.get('processed_count')} "
            f"dropped={heartbeat.get('dropped_count')} "
            f"errors={heartbeat.get('decode_errors')} "
            f"latency_ms={heartbeat.get('inference_latency_ms')} "
            f"last_error={heartbeat.get('last_error')}",
            flush=True,
        )

    def on_result(_channel: str, payload: bytes) -> None:
        counts["result"] += 1
        result = unpack_result(payload)
        objects = result.get("objects", [])
        labels = [str(obj.get("label")) for obj in objects[:5] if isinstance(obj, dict)]
        geometry = result.get("geometry_status")
        geometry = geometry if isinstance(geometry, dict) else {}
        object_geometry = []
        if isinstance(objects, list):
            for obj in objects[:5]:
                if not isinstance(obj, dict):
                    continue
                point_filter = obj.get("point_filter")
                point_filter = point_filter if isinstance(point_filter, dict) else {}
                object_geometry.append(
                    {
                        "label": obj.get("label"),
                        "depth": point_filter.get("valid"),
                        "depth_reason": point_filter.get("reason"),
                        "depth_points": obj.get("valid_depth_points"),
                        "contacts": len(obj.get("contact_candidates") or []),
                        "primitives": len(obj.get("surface_primitives") or []),
                    }
                )
        print(
            "[remote-monitor] result "
            f"seq={result.get('sequence')} "
            f"backend={result.get('backend')} "
            f"valid={result.get('valid')} "
            f"objects={len(objects) if isinstance(objects, list) else 'invalid'} "
            f"labels={labels} "
            f"geometry={{state={geometry.get('state_available')}, "
            f"T_goal_cam={geometry.get('t_goal_cam_available')}, "
            f"calibration={geometry.get('calibration_available')}, "
            f"intrinsics={geometry.get('intrinsics_available')}, "
            f"image_size={geometry.get('image_size')}, "
            f"depth_size={geometry.get('depth_size')}, "
            f"intrinsics_size={geometry.get('intrinsics_size')}, "
            f"geometry_intrinsics_size={geometry.get('geometry_intrinsics_size')}, "
            f"geometry_valid_depth_ratio={geometry.get('geometry_valid_depth_ratio')}, "
            f"state_delta_ms={geometry.get('state_delta_ms')}, "
            f"valid_depth_objects={geometry.get('objects_with_valid_depth')}, "
            f"contacts={geometry.get('contact_candidate_count')}, "
            f"primitives={geometry.get('primitive_count')}}} "
            f"object_geometry={object_geometry} "
            f"error={result.get('error')}",
            flush=True,
        )

    bus.subscribe(args.heartbeat_channel, on_heartbeat)
    bus.subscribe(args.result_channel, on_result)
    deadline = started + max(0.1, float(args.timeout_s))
    while time.monotonic() < deadline:
        bus.handle_timeout(200)
    print(
        "[remote-monitor] done "
        f"heartbeats={counts['heartbeat']} results={counts['result']}",
        flush=True,
    )
    return 0 if counts["heartbeat"] or counts["result"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
