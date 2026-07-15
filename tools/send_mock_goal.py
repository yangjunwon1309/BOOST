#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import msgpack
import zmq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--topic", default="path_goal")
    parser.add_argument("--target-link", default="left_wrist_yaw_link")
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--quaternion", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--settle-sec", type=float, default=0.5)
    args = parser.parse_args()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(args.endpoint)
    time.sleep(args.settle_sec)
    payload = {
        "target_link": args.target_link,
        "position": args.position,
        "quaternion": args.quaternion,
    }
    socket.send(args.topic.encode() + msgpack.packb(payload, use_bin_type=True))
    print(f"sent {args.topic} -> {args.endpoint}: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

