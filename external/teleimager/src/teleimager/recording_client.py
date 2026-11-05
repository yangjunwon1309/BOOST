"""Live cv2.imshow viewer for the TIM2 recording ZMQ streams.

teleimager-client only ever looks at the legacy `enable_zmq` / `zmq_port`
streams, which carry raw JPEG bytes with no sequence numbers, so a dropped
frame just means the viewer shows the previous frame again -- it never looks
"stuck" or gappy. The `enable_recording_zmq` / `recording_zmq_port` streams
carry TIM2-enveloped frames (recording_protocol.py) with a monotonic sequence
number, so drops are visible as sequence gaps. This tool subscribes to those
streams the same way teleimager-client does (SUB + CONFLATE, latest frame
wins) and overlays live FPS / gap counters so the two can be compared side by
side while chasing where a recording stream is losing frames.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
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


class GapCounter:
    def __init__(self):
        self.last_sequence: Optional[int] = None
        self.gaps = 0

    def observe(self, sequence: int) -> None:
        if self.last_sequence is not None and sequence > self.last_sequence + 1:
            self.gaps += sequence - self.last_sequence - 1
        self.last_sequence = sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cv2.imshow viewer for TIM2 recording ZMQ streams (compare against teleimager-client)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="IP address of the teleimager server")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="cam_config yaml to read recording ports from")
    args = parser.parse_args(argv)
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
    counters = {name: GapCounter() for name in streams}
    fps_windows = {name: [] for name in streams}

    for name, port in streams.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(f"tcp://{args.host}:{port}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = name
        print(f"{name}: tcp://{args.host}:{port}")
    print("Press q (with a window focused) to exit.")

    try:
        while True:
            events = dict(poller.poll(timeout=100))
            for socket, name in sockets.items():
                if socket not in events:
                    continue
                packet = socket.recv()
                frame = decode_recording_frame(packet)
                counters[name].observe(frame.sequence)

                now = time.perf_counter()
                window = fps_windows[name]
                window.append(now)
                cutoff = now - 1.0
                while window and window[0] < cutoff:
                    window.pop(0)
                fps = max(len(window) - 1, 0)

                img = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                cv2.putText(
                    img,
                    f"{name} seq={frame.sequence} gaps={counters[name].gaps} fps~={fps}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(f"[recording_zmq] {name}", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
