"""Map remote capture clocks to the recording computer without changing either OS clock."""

from dataclasses import asdict, dataclass
import time

import zmq


def clock_response():
    return {
        "clock_version": 1,
        "wall_time_us": time.time_ns() // 1000,
        "monotonic_time_ns": time.monotonic_ns(),
    }


@dataclass(frozen=True)
class ClockCalibration:
    client_wall_time_us: int
    server_wall_time_us: int
    server_monotonic_time_ns: int
    round_trip_ns: int

    @property
    def uncertainty_us(self):
        return (self.round_trip_ns + 1999) // 2000

    def capture_wall_time_us(self, capture_monotonic_time_ns):
        return self.client_wall_time_us + (
            capture_monotonic_time_ns - self.server_monotonic_time_ns
        ) // 1000

    def metadata(self):
        return dict(asdict(self), uncertainty_us=self.uncertainty_us,
                    clock_offset_us=self.client_wall_time_us - self.server_wall_time_us,
                    method="lowest-RTT ZMQ midpoint; remote monotonic capture clock")


def calibrate_camera_clock(host, port=60000, samples=5, timeout_ms=2000):
    """Use the shortest round trip to bound network asymmetry and queue delay.

    The recording host's wall clock anchors remote *monotonic* capture times.
    A wrong or later corrected robot wall clock cannot shift the episode.
    """
    if samples < 1 or timeout_ms < 1:
        raise ValueError("clock calibration needs positive samples and timeout")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://{host}:{port}")
    measurements = []
    try:
        for _ in range(samples):
            start_wall_us = time.time_ns() // 1000
            start_ns = time.monotonic_ns()
            socket.send(b"GET_CLOCK")
            response = socket.recv_json()
            elapsed_ns = time.monotonic_ns() - start_ns
            if not isinstance(response, dict) or response.get("clock_version") != 1:
                raise RuntimeError(f"{host}:{port} does not support GET_CLOCK; update Nature teleimager")
            for name in ("wall_time_us", "monotonic_time_ns"):
                if type(response.get(name)) is not int or response[name] < 0:
                    raise RuntimeError(f"invalid clock response field {name} from {host}:{port}")
            measurements.append(ClockCalibration(
                start_wall_us + elapsed_ns // 2000,
                response["wall_time_us"], response["monotonic_time_ns"], elapsed_ns,
            ))
    except zmq.Again as error:
        raise RuntimeError(f"camera clock request timed out at {host}:{port}") from error
    finally:
        socket.close()
        context.term()
    return min(measurements, key=lambda sample: sample.round_trip_ns)
