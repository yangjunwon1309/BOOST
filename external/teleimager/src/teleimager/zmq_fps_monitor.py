"""Monitor all timestamped ZMQ camera streams declared in a config."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping, Optional, Sequence

import yaml
import zmq

from .recording_protocol import decode_recording_frame


DEFAULT_CONFIG = Path(__file__).parents[2] / "cam_config_server.yaml"


def recording_streams(config: Mapping) -> dict[str, int]:
    streams = {
        name: camera["recording_zmq_port"]
        for name, camera in config.items()
        if isinstance(camera, Mapping) and camera.get("enable_recording_zmq", False)
    }
    if len(set(streams.values())) != len(streams):
        raise ValueError("recording_zmq_port values must be unique")
    return streams


@dataclass
class StreamStats:
    frames: int = 0
    packet_bytes: int = 0
    sequence_gaps: int = 0
    last_sequence: Optional[int] = None
    width: int = 0
    height: int = 0

    def observe(self, *, sequence: int, packet_size: int, width: int, height: int) -> None:
        if self.last_sequence is not None and sequence > self.last_sequence + 1:
            self.sequence_gaps += sequence - self.last_sequence - 1
        self.last_sequence = sequence
        self.frames += 1
        self.packet_bytes += packet_size
        self.width = width
        self.height = height

    def report(self, elapsed_seconds: float) -> tuple[float, float, int, Optional[int], int, int]:
        fps = self.frames / elapsed_seconds
        mbps = self.packet_bytes * 8 / elapsed_seconds / 1_000_000
        result = (
            fps,
            mbps,
            self.sequence_gaps,
            self.last_sequence,
            self.width,
            self.height,
        )
        self.frames = 0
        self.packet_bytes = 0
        self.sequence_gaps = 0
        return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report FPS for every enabled TIM2/ZMQ camera stream."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    streams = recording_streams(config)
    if not streams:
        raise SystemExit(f"No recording ZMQ streams are enabled in {args.config}")

    context = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    stats = {name: StreamStats() for name in streams}

    for name, port in streams.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(f"tcp://{args.host}:{port}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = name
        print(f"{name}: tcp://{args.host}:{port}")

    last_report = time.monotonic()
    try:
        while True:
            for socket in dict(poller.poll(timeout=100)):
                packet = socket.recv()
                frame = decode_recording_frame(packet)
                stats[sockets[socket]].observe(
                    sequence=frame.sequence,
                    packet_size=len(packet),
                    width=frame.width,
                    height=frame.height,
                )

            now = time.monotonic()
            elapsed = now - last_report
            if elapsed >= args.interval:
                for name, stream_stats in stats.items():
                    fps, mbps, gaps, sequence, width, height = stream_stats.report(elapsed)
                    print(
                        f"{name:20} {fps:5.1f} FPS | {mbps:5.1f} Mbps | "
                        f"gaps={gaps} seq={sequence} frame={width}x{height}"
                    )
                print()
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()
