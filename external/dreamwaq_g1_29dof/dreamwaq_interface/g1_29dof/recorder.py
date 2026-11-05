"""Chunked, non-pickle signal recorder shared by play and deployment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np


class SignalTraceRecorder:
    """Write each DreamWaQ signal to its own atomic ``.npy`` chunk stream."""

    def __init__(
        self,
        root: Any,
        *,
        signal_shapes: Mapping[str, Sequence[int]],
        metadata: Mapping[str, Any],
        chunk_steps: int = 512,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.chunk_steps = int(chunk_steps)
        if self.chunk_steps <= 0:
            raise ValueError("trace chunk_steps must be positive")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("trace directory already exists and is not empty: %s" % self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.signal_shapes = {str(name): tuple(int(value) for value in shape) for name, shape in signal_shapes.items()}
        if not self.signal_shapes or any(not name for name in self.signal_shapes):
            raise ValueError("trace signal schema must be non-empty")
        self._buffers: dict[str, list[np.ndarray]] = {name: [] for name in self.signal_shapes}
        self._chunk_index = 0
        self._steps = 0
        self._closed = False
        # Trace persistence must never run on the 50-Hz policy thread.  A
        # synchronous fsync of the 20 signal files took ~76 ms on the G1 and
        # showed up as a false policy deadline miss at steps 512 and 1024.
        self._write_queue: queue.Queue[tuple[int, dict[str, list[np.ndarray]]] | None] = queue.Queue()
        self._writer_error: BaseException | None = None
        self._metadata = dict(metadata)
        trace_schema = str(self._metadata.get("trace_schema", "dreamwaq.g1_29dof.signal_trace.v1"))
        self._metadata.update(
            {
                "trace_schema": trace_schema,
                "created_unix_ns": time.time_ns(),
                "signal_shapes": {name: list(shape) for name, shape in self.signal_shapes.items()},
                "chunk_steps": self.chunk_steps,
            }
        )
        for name in self.signal_shapes:
            (self.root / "signals" / name).mkdir(parents=True, exist_ok=True)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="dreamwaq-trace-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                chunk_index, values = item
                self._write_chunk(chunk_index, values)
            except BaseException as exc:  # surface the error on append/close
                self._writer_error = exc
            finally:
                self._write_queue.task_done()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("DreamWaQ trace writer failed") from self._writer_error

    def append(self, **signals: Any) -> None:
        if self._closed:
            raise RuntimeError("trace recorder is closed")
        self._raise_writer_error()
        if set(signals) != set(self.signal_shapes):
            missing = sorted(set(self.signal_shapes) - set(signals))
            extra = sorted(set(signals) - set(self.signal_shapes))
            raise ValueError("trace signal set mismatch: missing=%s extra=%s" % (missing, extra))
        converted = {}
        for name, shape in self.signal_shapes.items():
            value = np.asarray(signals[name])
            if value.shape != shape:
                raise ValueError("trace signal %s has shape %s; expected %s" % (name, value.shape, shape))
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError("trace signal %s contains NaN or infinity" % name)
            converted[name] = value.copy()
        for name, value in converted.items():
            self._buffers[name].append(value)
        self._steps += 1
        if len(next(iter(self._buffers.values()))) >= self.chunk_steps:
            self._flush()

    def _atomic_save(self, path: Path, value: np.ndarray) -> None:
        temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write_chunk(self, chunk_index: int, values: dict[str, list[np.ndarray]]) -> None:
        filename = "chunk_%06d.npy" % chunk_index
        for name, buffered_values in values.items():
            value = np.stack(buffered_values, axis=0)
            self._atomic_save(self.root / "signals" / name / filename, value)

    def _flush(self) -> None:
        count = len(next(iter(self._buffers.values())))
        if count == 0:
            return
        values = {name: buffer[:] for name, buffer in self._buffers.items()}
        for buffer in self._buffers.values():
            buffer.clear()
        self._write_queue.put((self._chunk_index, values))
        self._chunk_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._write_queue.put(None)
        self._write_queue.join()
        self._writer_thread.join(timeout=5.0)
        self._raise_writer_error()
        self._metadata.update({"steps": self._steps, "chunks": self._chunk_index})
        target = self.root / "metadata.json"
        temporary = target.with_name(".metadata.json.%d.tmp" % os.getpid())
        temporary.write_text(
            json.dumps(self._metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        self._closed = True

    def __enter__(self) -> "SignalTraceRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def load_signal(root: Any, name: str) -> np.ndarray:
        signal_dir = Path(root).expanduser().resolve() / "signals" / str(name)
        chunks = sorted(signal_dir.glob("chunk_*.npy"))
        if not chunks:
            raise FileNotFoundError("trace contains no chunks for signal %s" % name)
        return np.concatenate([np.load(path, allow_pickle=False) for path in chunks], axis=0)


DEPLOYMENT_TRACE_SHAPES = {
    "monotonic_ns": (),
    "command": (3,),
    "q_sdk": (29,),
    "dq_sdk": (29,),
    "quaternion_wxyz": (4,),
    "angular_velocity_body": (3,),
    "projected_gravity_body": (3,),
    "raw_control_frame": (96,),
    "policy_current": (99,),
    "cenet_history": (465,),
    "v_hat": (3,),
    "z": (32,),
    "actor_input": (131,),
    "policy_mean": (12,),
    "lower_action": (12,),
    "previous_action_policy": (29,),
    "q_target_sdk": (29,),
    "inference_latency_ns": (),
}


def deployment_trace_shapes(action_dim: int = 12) -> dict[str, tuple[int, ...]]:
    shapes = dict(DEPLOYMENT_TRACE_SHAPES)
    shapes["policy_mean"] = (int(action_dim),)
    shapes["lower_action"] = (int(action_dim),)
    return shapes
