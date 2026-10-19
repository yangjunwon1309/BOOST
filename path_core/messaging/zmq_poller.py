"""Small non-blocking ZMQ subscriber used by PICO and Galaxy XR inputs."""

from __future__ import annotations

import zmq


class ZMQPoller:
    """Keep only the latest subscribed packet for low-latency inputs."""

    def __init__(self, host: str = "localhost", port: int = 5555, topic: str = ""):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic

    def get_data(self) -> bytes | None:
        if not self._socket.poll(timeout=0):
            return None
        data = self._socket.recv(zmq.NOBLOCK)
        return None if data is None else data[len(self._topic) :]

    def close(self) -> None:
        self._socket.close()
        self._context.term()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
