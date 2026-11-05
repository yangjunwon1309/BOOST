"""Same-frame RGB and color-aligned RealSense Z16 depth on a separate PUB port."""
import json
import multiprocessing
from multiprocessing import shared_memory
import time
import zlib
import numpy as np
import zmq


class RgbdPublisher:
    def __init__(self, port):
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{int(port)}")

    def publish(self, jpeg, depth, metadata):
        depth = np.asarray(depth, dtype="<u2")
        if depth.shape != (metadata["height"], metadata["width"]):
            raise ValueError("RGBD depth dimensions differ from color")
        compressed = zlib.compress(depth.tobytes(), 1)
        header = dict(metadata, schema="RGBD1", depth_dtype="<u2", depth_codec="zlib",
                      publish_monotonic_ns=time.monotonic_ns())
        self.socket.send_multipart(
            [json.dumps(header).encode(), jpeg, compressed],
            flags=zmq.NOBLOCK,
        )

    def close(self):
        self.socket.close(linger=0)


class NativeDepthAligner:
    """SDK projection geometry for zero-distortion D435 depth/color profiles."""
    def __init__(self, calibration):
        self.calibration = calibration
        for name in ("depth", "color"):
            coefficients = np.asarray(calibration[name]["coeffs"])
            if not np.isfinite(coefficients).all() or np.any(np.abs(coefficients) > 1e-8):
                raise ValueError("native RGBD alignment requires zero-distortion SDK profiles")
        intr = calibration["depth"]
        self.y, self.x = np.indices((intr["height"], intr["width"]), dtype=np.float32)
        self.rotation = np.asarray(calibration["rotation"], np.float32).reshape(3, 3).T
        self.translation = np.asarray(calibration["translation"], np.float32)

    def compute(self, raw):
        depth, color = self.calibration["depth"], self.calibration["color"]
        z = raw.astype(np.float32) * np.float32(self.calibration["scale"])
        corners = []
        for offset in (-0.5, 0.5):
            x = ((self.x + offset - depth["ppx"]) / depth["fx"]) * z
            y = ((self.y + offset - depth["ppy"]) / depth["fy"]) * z
            r, t = self.rotation, self.translation
            tx = r[0, 0] * x + r[0, 1] * y + r[0, 2] * z + t[0]
            ty = r[1, 0] * x + r[1, 1] * y + r[1, 2] * z + t[1]
            tz = r[2, 0] * x + r[2, 1] * y + r[2, 2] * z + t[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                px = tx / tz * color["fx"] + color["ppx"]
                py = ty / tz * color["fy"] + color["ppy"]
            corners.append((np.nan_to_num(px + 0.5, nan=-1., posinf=-1., neginf=-1.).astype(np.int32),
                            np.nan_to_num(py + 0.5, nan=-1., posinf=-1., neginf=-1.).astype(np.int32), tz))
        x0, y0, z0 = corners[0]
        x1, y1, z1 = corners[1]
        width, height = color["width"], color["height"]
        valid = ((raw > 0) & (z0 > 0) & (z1 > 0) & (x0 >= 0) & (y0 >= 0)
                 & (x1 < width) & (y1 < height) & (x1 >= x0) & (y1 >= y0))
        output = np.full(width * height, 65535, dtype=np.uint16)
        if valid.any():
            dx, dy = x1 - x0, y1 - y0
            max_dx, max_dy = int(dx[valid].max()), int(dy[valid].max())
            if max_dx > 8 or max_dy > 8:
                raise ValueError("unexpected RGBD pixel footprint; camera calibration/profile mismatch")
            for oy in range(max_dy + 1):
                for ox in range(max_dx + 1):
                    use = valid & (dx >= ox) & (dy >= oy)
                    indices = (y0[use] + oy) * width + x0[use] + ox
                    np.minimum.at(output, indices, raw[use])
        output[output == 65535] = 0
        return output.reshape(height, width)

    def close(self):
        pass


def _alignment_worker(memory_name, shape, lengths, lock, updated, stopped, port, calibration):
    memory = shared_memory.SharedMemory(name=memory_name)
    depth_bytes = int(np.prod(shape)) * 2
    depth_view = np.ndarray(shape, dtype=np.uint16, buffer=memory.buf)
    aligner, publisher = NativeDepthAligner(calibration), RgbdPublisher(port)
    try:
        while not stopped.is_set():
            if not updated.wait(timeout=0.1):
                continue
            if stopped.is_set():
                break
            with lock:
                updated.clear()
                raw_depth = depth_view.copy()
                jpeg_size, header_size = lengths[0], lengths[1]
                jpeg = bytes(memory.buf[depth_bytes:depth_bytes + jpeg_size])
                metadata = json.loads(bytes(memory.buf[depth_bytes + 2_000_000:
                                                     depth_bytes + 2_000_000 + header_size]))
            publisher.publish(jpeg, aligner.compute(raw_depth), metadata)
    finally:
        publisher.close()
        aligner.close()
        del depth_view
        memory.close()


class RgbdAlignmentProcess:
    """One shared latest-frame slot; alignment cannot block the capture GIL."""
    def __init__(self, port, calibration):
        context = multiprocessing.get_context("spawn")
        self.shape = (calibration["depth"]["height"], calibration["depth"]["width"])
        self.depth_bytes = int(np.prod(self.shape)) * 2
        self.memory = shared_memory.SharedMemory(create=True, size=self.depth_bytes + 2_000_000 + 4096)
        self.depth_view = np.ndarray(self.shape, dtype=np.uint16, buffer=self.memory.buf)
        self.lengths = context.Array("i", 2, lock=False)
        self.lock, self.updated, self.stopped = context.Lock(), context.Event(), context.Event()
        self.process = context.Process(
            target=_alignment_worker,
            args=(self.memory.name, self.shape, self.lengths, self.lock, self.updated,
                  self.stopped, port, calibration), daemon=True,
        )
        try:
            self.process.start()
        except Exception:
            del self.depth_view
            self.memory.close()
            self.memory.unlink()
            raise

    def submit(self, raw_depth, jpeg, metadata):
        if not self.process.is_alive():
            raise RuntimeError("native RGBD alignment process exited")
        header = json.dumps(metadata).encode()
        if raw_depth.shape != self.shape or len(jpeg) > 2_000_000 or len(header) > 4096:
            raise ValueError("RGBD frame exceeds the shared capture slot")
        with self.lock:
            np.copyto(self.depth_view, raw_depth)
            self.memory.buf[self.depth_bytes:self.depth_bytes + len(jpeg)] = jpeg
            self.memory.buf[self.depth_bytes + 2_000_000:
                            self.depth_bytes + 2_000_000 + len(header)] = header
            self.lengths[0], self.lengths[1] = len(jpeg), len(header)
            self.updated.set()

    def close(self):
        self.stopped.set()
        self.updated.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        del self.depth_view
        self.memory.close()
        self.memory.unlink()
