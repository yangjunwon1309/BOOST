"""Decode G1 Teleimager TIM2 wrist frames and relay JPEG plus metadata to LCM."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import lcm
import yaml
import zmq

BOOST_ROOT = Path(__file__).resolve().parents[3]
TELEIMAGER_SRC = BOOST_ROOT / "external" / "teleimager" / "src"
if TELEIMAGER_SRC.is_dir() and str(TELEIMAGER_SRC) not in sys.path:
    sys.path.insert(0, str(TELEIMAGER_SRC))

from teleimager.recording_protocol import decode_recording_frame

from .wrist_rgb_protocol_v11 import WRIST_RGB_FRAME_CHANNEL, now_us, pack_rgb_frame


LOG = logging.getLogger("thor_teleimager_wrist_relay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get(
            "BOOST_TELEIMAGER_RELAY_CONFIG",
            BOOST_ROOT / "configs/teleop/teleimager_relay.yaml",
        )),
    )
    parser.add_argument("--g1-host", default=None)
    parser.add_argument("--teleimager-config", type=Path, default=None)
    parser.add_argument("--left-port", type=int, default=None)
    parser.add_argument("--right-port", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--wifi-lcm-url",
        default=None,
    )
    parser.add_argument("--report-period-s", type=float, default=2.0)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        relay_config = yaml.safe_load(stream) or {}
    teleimager_config_path = args.teleimager_config or Path(
        os.environ.get(
            "BOOST_TELEIMAGER_G1_CONFIG",
            relay_config.get(
                "teleimager_config",
                "external/teleimager/config/hbvcam_4m2214hd_g1.yaml",
            ),
        )
    )
    if not teleimager_config_path.is_absolute():
        teleimager_config_path = BOOST_ROOT / teleimager_config_path
    with teleimager_config_path.open("r", encoding="utf-8") as stream:
        teleimager_config = yaml.safe_load(stream)

    head = teleimager_config["head_camera"]
    left = teleimager_config["left_wrist_camera"]
    right = teleimager_config["right_wrist_camera"]
    args.g1_host = args.g1_host or os.environ.get(
        "G1_TELEIMAGER_HOST", relay_config.get("g1_host", "192.168.123.164")
    )
    args.wifi_lcm_url = args.wifi_lcm_url or os.environ.get(
        "WIFI_LCM_URL",
        relay_config.get("wifi_lcm_url", "udpm://239.255.77.67:7667?ttl=1"),
    )
    args.left_port = (
        int(left["recording_zmq_port"]) if args.left_port is None else args.left_port
    )
    args.right_port = (
        int(right["recording_zmq_port"]) if args.right_port is None else args.right_port
    )
    image_shape = left.get("image_shape", [480, 640])
    args.height = int(image_shape[0]) if args.height is None else args.height
    args.width = int(image_shape[1]) if args.width is None else args.width
    if min(args.left_port, args.right_port, args.width, args.height) <= 0:
        parser.error("ports and image dimensions must be positive")
    for name, camera in (("left_wrist_camera", left), ("right_wrist_camera", right)):
        if not camera.get("enable_recording_zmq", False):
            parser.error(f"{name} must enable the TIM2 recording ZMQ stream")
        shape = camera.get("image_shape", [args.height, args.width])
        if (int(shape[1]), int(shape[0])) != (args.width, args.height):
            parser.error("left/right Teleimager wrist cameras must use the same image dimensions")

    bus = lcm.LCM(args.wifi_lcm_url)
    context = zmq.Context()
    poller = zmq.Poller()
    sockets: dict[zmq.Socket, tuple[str, int]] = {}
    for camera_id, port in (("left_wrist", args.left_port), ("right_wrist", args.right_port)):
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(f"tcp://{args.g1_host}:{port}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = (camera_id, port)

    counts = {"left_wrist": 0, "right_wrist": 0}
    byte_counts = {"left_wrist": 0, "right_wrist": 0}
    sequence_gaps = {"left_wrist": 0, "right_wrist": 0}
    last_sequence: dict[str, int] = {}
    last_report = time.monotonic()
    LOG.info(
        "subscribed to G1 Teleimager TIM2 %s:%d/%d (head TIM2 port %d), "
        "publishing LCM v2 %s on %s; JPEG dimensions=%dx%d; config=%s",
        args.g1_host,
        args.left_port,
        args.right_port,
        int(head["recording_zmq_port"]),
        WRIST_RGB_FRAME_CHANNEL,
        args.wifi_lcm_url,
        args.width,
        args.height,
        teleimager_config_path,
    )

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            for socket in events:
                camera_id, _port = sockets[socket]
                try:
                    source_frame = decode_recording_frame(socket.recv())
                    if (source_frame.width, source_frame.height) != (args.width, args.height):
                        raise ValueError(
                            "TIM2 dimensions %dx%d differ from configured %dx%d"
                            % (source_frame.width, source_frame.height, args.width, args.height)
                        )
                    previous_sequence = last_sequence.get(camera_id)
                    if previous_sequence is not None:
                        if source_frame.sequence > previous_sequence + 1:
                            gap = source_frame.sequence - previous_sequence - 1
                            sequence_gaps[camera_id] += gap
                            LOG.warning(
                                "%s TIM2 sequence gap: previous=%d current=%d missing=%d",
                                camera_id,
                                previous_sequence,
                                source_frame.sequence,
                                gap,
                            )
                        elif source_frame.sequence < previous_sequence:
                            LOG.info(
                                "%s TIM2 sequence restarted: previous=%d current=%d",
                                camera_id,
                                previous_sequence,
                                source_frame.sequence,
                            )
                    last_sequence[camera_id] = source_frame.sequence
                    packet = pack_rgb_frame(
                        camera_id=camera_id,
                        width=source_frame.width,
                        height=source_frame.height,
                        thor_receive_timestamp_us=now_us(),
                        jpeg=source_frame.jpeg,
                        camera_sequence=source_frame.sequence,
                        g1_capture_wall_time_us=source_frame.capture_wall_time_us,
                        g1_capture_monotonic_time_ns=source_frame.capture_monotonic_time_ns,
                    )
                    bus.publish(WRIST_RGB_FRAME_CHANNEL, packet)
                except Exception:
                    LOG.exception("failed to decode or relay %s TIM2 frame", camera_id)
                    continue
                counts[camera_id] += 1
                byte_counts[camera_id] += len(source_frame.jpeg)

            now = time.monotonic()
            if now - last_report >= args.report_period_s:
                elapsed = now - last_report
                LOG.info(
                    "relay fps left=%.1f right=%.1f last_jpeg_bytes left=%d right=%d "
                    "missing_tim2 left=%d right=%d",
                    counts["left_wrist"] / elapsed,
                    counts["right_wrist"] / elapsed,
                    byte_counts["left_wrist"],
                    byte_counts["right_wrist"],
                    sequence_gaps["left_wrist"],
                    sequence_gaps["right_wrist"],
                )
                counts = {"left_wrist": 0, "right_wrist": 0}
                byte_counts = {"left_wrist": 0, "right_wrist": 0}
                sequence_gaps = {"left_wrist": 0, "right_wrist": 0}
                last_report = now
    except KeyboardInterrupt:
        LOG.info("stopping G1 teleimager relay")
    finally:
        for socket in sockets:
            socket.close(0)
        context.term()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
