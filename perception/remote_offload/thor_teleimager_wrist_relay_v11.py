"""Relay G1 teleimager's raw wrist JPEG streams onto Thor's Wi-Fi LCM bus."""

from __future__ import annotations

import argparse
import logging
import os
import time

import lcm
import zmq

from .wrist_rgb_protocol_v11 import WRIST_RGB_FRAME_CHANNEL, now_us, pack_rgb_frame


LOG = logging.getLogger("thor_teleimager_wrist_relay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-host", default=os.environ.get("G1_TELEIMAGER_HOST", "192.168.123.164"))
    parser.add_argument("--left-port", type=int, default=55556)
    parser.add_argument("--right-port", type=int, default=55557)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--wifi-lcm-url",
        default=os.environ.get("WIFI_LCM_URL", "udpm://239.255.77.67:7667?ttl=1"),
    )
    parser.add_argument("--report-period-s", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.left_port, args.right_port, args.width, args.height) <= 0:
        parser.error("ports and image dimensions must be positive")

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
    last_report = time.monotonic()
    LOG.info(
        "subscribed to G1 teleimager %s:%d/%d, publishing LCM %s on %s; JPEG dimensions=%dx%d",
        args.g1_host,
        args.left_port,
        args.right_port,
        WRIST_RGB_FRAME_CHANNEL,
        args.wifi_lcm_url,
        args.width,
        args.height,
    )

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            for socket in events:
                camera_id, _port = sockets[socket]
                jpeg = socket.recv()
                try:
                    packet = pack_rgb_frame(
                        camera_id=camera_id,
                        width=args.width,
                        height=args.height,
                        thor_receive_timestamp_us=now_us(),
                        jpeg=jpeg,
                    )
                    bus.publish(WRIST_RGB_FRAME_CHANNEL, packet)
                except Exception:
                    LOG.exception("failed to relay %s frame (%d bytes)", camera_id, len(jpeg))
                    continue
                counts[camera_id] += 1
                byte_counts[camera_id] += len(jpeg)

            now = time.monotonic()
            if now - last_report >= args.report_period_s:
                elapsed = now - last_report
                LOG.info(
                    "relay fps left=%.1f right=%.1f last_jpeg_bytes left=%d right=%d",
                    counts["left_wrist"] / elapsed,
                    counts["right_wrist"] / elapsed,
                    byte_counts["left_wrist"],
                    byte_counts["right_wrist"],
                )
                counts = {"left_wrist": 0, "right_wrist": 0}
                byte_counts = {"left_wrist": 0, "right_wrist": 0}
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
