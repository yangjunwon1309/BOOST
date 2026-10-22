#!/usr/bin/env python3
"""Translate PATH's PDU1 grip flags into BrainCo Revo2 DDS commands.

PATH's arm bridge remains the owner of rt/arm_sdk. This small side channel
handles only the two binary grip flags already present in G1_PATH_UPPER_COMMAND.
The MuJoCo Revo2 backend consumes the same six normalized BrainCo channels as
the real service.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import lcm
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

from boost.control.hand_mppi.revo2 import BRAINCO_COMMAND_TOPICS
from boost.io.path_transport.g1_direct_protocol import (
    UPPER_COMMAND_CHANNEL,
    unpack_upper_command,
)

ROOT = Path(__file__).resolve().parents[3]


def _new_message(values: tuple[float, ...]) -> MotorCmds_:
    message = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in values])
    for command, value in zip(message.cmds, values):
        command.q = float(value)
        command.dq = 1.0
    return message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sim/mujoco_revo2.yaml")
    parser.add_argument("--lcm-url", default=None)
    parser.add_argument("--lcm-channel", default=None)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--stale-open-after-sec", type=float, default=None)
    parser.add_argument("--thumb-close", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    hand_config_path = (ROOT / config["hand_config"]).resolve()
    with hand_config_path.open("r", encoding="utf-8") as stream:
        hand_config = yaml.safe_load(stream)
    args.lcm_url = args.lcm_url or os.environ.get(
        "PATH_LCM_URL", "udpm://239.255.76.67:7667?ttl=1"
    )
    args.lcm_channel = args.lcm_channel or config.get(
        "grip_lcm_channel", UPPER_COMMAND_CHANNEL
    )
    args.interface = args.interface or config.get("interface", "lo")
    args.domain_id = args.domain_id if args.domain_id is not None else int(config.get("domain_id", 0))
    args.stale_open_after_sec = (
        args.stale_open_after_sec
        if args.stale_open_after_sec is not None
        else float(hand_config["control"]["stale_command_timeout_sec"])
    )
    if args.thumb_close is None:
        args.thumb_close = float(hand_config["binary_close"][0])

    ChannelFactoryInitialize(args.domain_id, args.interface)
    left_pub = ChannelPublisher(BRAINCO_COMMAND_TOPICS["left"], MotorCmds_)
    right_pub = ChannelPublisher(BRAINCO_COMMAND_TOPICS["right"], MotorCmds_)
    left_pub.Init()
    right_pub.Init()
    bus = lcm.LCM(args.lcm_url)
    latest = {"left": False, "right": False, "received": 0.0}

    def on_command(_channel: str, raw: bytes) -> None:
        try:
            command = unpack_upper_command(raw)
        except ValueError as exc:
            print(f"[Revo2 bridge] ignored malformed PATH command: {exc}", flush=True)
            return
        latest["left"] = command.left_grip
        latest["right"] = command.right_grip
        latest["received"] = time.monotonic()

    bus.subscribe(args.lcm_channel, on_command)
    period = 1.0 / max(1.0, float(args.rate_hz))
    next_tick = time.monotonic()
    print(
        f"[Revo2 bridge] LCM {args.lcm_url}/{args.lcm_channel} -> "
        "rt/brainco/{left,right}/cmd on "
        f"DDS domain={args.domain_id} interface={args.interface}",
        flush=True,
    )
    try:
        while True:
            bus.handle_timeout(1)
            now = time.monotonic()
            fresh = latest["received"] > 0 and now - latest["received"] <= args.stale_open_after_sec
            left_closed = fresh and bool(latest["left"])
            right_closed = fresh and bool(latest["right"])
            open_pose = tuple(float(value) for value in hand_config["binary_open"])
            close_pose = tuple(float(value) for value in hand_config["binary_close"])
            close_pose = (args.thumb_close, *close_pose[1:])
            left_pub.Write(_new_message(close_pose if left_closed else open_pose))
            right_pub.Write(_new_message(close_pose if right_closed else open_pose))
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -5.0 * period:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
