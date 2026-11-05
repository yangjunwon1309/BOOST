"""Record timestamped teleimager JPEG streams without image conversion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping, Optional, Sequence

import yaml
import zmq

from .recording_protocol import RecordingFrame, decode_recording_frame
from .zmq_fps_monitor import recording_streams


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/hbvcam_4m2214hd_g1.yaml"
)


@dataclass(frozen=True)
class FrameJob:
    camera: str
    frame: RecordingFrame
    epoch: int
    sequence_gap: int
    packet_size: int
    receive_wall_time_ns: int
    receive_monotonic_time_ns: int


@dataclass
class CameraStats:
    received: int = 0
    malformed: int = 0
    sequence_gaps: int = 0
    queue_drops: int = 0
    duplicates: int = 0
    packet_bytes: int = 0
    interval_received: int = 0
    interval_gaps: int = 0
    interval_drops: int = 0
    interval_bytes: int = 0

    def observe(self, packet_size: int, sequence_gap: int) -> None:
        self.received += 1
        self.sequence_gaps += sequence_gap
        self.packet_bytes += packet_size
        self.interval_received += 1
        self.interval_gaps += sequence_gap
        self.interval_bytes += packet_size

    def report_interval(self, elapsed_seconds: float) -> tuple[float, float, int, int]:
        report = (
            self.interval_received / elapsed_seconds,
            self.interval_bytes * 8 / elapsed_seconds / 1_000_000,
            self.interval_gaps,
            self.interval_drops,
        )
        self.interval_received = 0
        self.interval_gaps = 0
        self.interval_drops = 0
        self.interval_bytes = 0
        return report


class SequenceTracker:
    def __init__(self) -> None:
        self.epoch = 0
        self.last_sequence: Optional[int] = None

    def observe(self, sequence: int) -> tuple[int, int, bool]:
        if self.last_sequence is None:
            self.last_sequence = sequence
            return self.epoch, 0, False
        if sequence == self.last_sequence:
            return self.epoch, 0, True
        if sequence < self.last_sequence:
            self.epoch += 1
            gap = 0
        else:
            gap = sequence - self.last_sequence - 1
        self.last_sequence = sequence
        return self.epoch, gap, False


def enqueue_latest(work_queue: queue.Queue, item: Any) -> Any:
    dropped = None
    while True:
        try:
            work_queue.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                dropped = work_queue.get_nowait()
            except queue.Empty:
                continue
            work_queue.task_done()


def write_frames(
    work_queue: queue.Queue,
    output_dir: Path,
    metadata_path: Path,
    written: dict[str, int],
) -> None:
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        while True:
            job = work_queue.get()
            try:
                if job is None:
                    return
                relative_path = Path(job.camera) / (
                    f"epoch_{job.epoch:03d}_seq_{job.frame.sequence:010d}.jpg"
                )
                image_path = output_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(job.frame.jpeg)
                metadata_file.write(
                    json.dumps(
                        {
                            "camera": job.camera,
                            "epoch": job.epoch,
                            "sequence": job.frame.sequence,
                            "sequence_gap": job.sequence_gap,
                            "capture_wall_time_us": job.frame.capture_wall_time_us,
                            "capture_monotonic_time_ns": job.frame.capture_monotonic_time_ns,
                            "receive_wall_time_ns": job.receive_wall_time_ns,
                            "receive_monotonic_time_ns": job.receive_monotonic_time_ns,
                            "width": job.frame.width,
                            "height": job.frame.height,
                            "packet_size": job.packet_size,
                            "jpeg_size": len(job.frame.jpeg),
                            "path": relative_path.as_posix(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                written[job.camera] += 1
            finally:
                work_queue.task_done()


def load_recording_streams(config_path: Path) -> dict[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    streams = recording_streams(config)
    if not streams:
        raise ValueError(f"no recording ZMQ streams are enabled in {config_path}")
    for name, port in streams.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe camera name: {name!r}")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"invalid recording port for {name}: {port!r}")
    return streams


def write_summary(
    path: Path,
    stats: Mapping[str, CameraStats],
    written: Mapping[str, int],
    *,
    duration_seconds: float,
) -> None:
    cameras = {}
    for name, camera in stats.items():
        cameras[name] = {
            "received": camera.received,
            "written": written[name],
            "malformed": camera.malformed,
            "sequence_gaps": camera.sequence_gaps,
            "queue_drops": camera.queue_drops,
            "duplicates": camera.duplicates,
            "packet_bytes": camera.packet_bytes,
            "average_receive_fps": camera.received / duration_seconds
            if duration_seconds > 0
            else 0.0,
            "average_mbps": camera.packet_bytes * 8 / duration_seconds / 1_000_000
            if duration_seconds > 0
            else 0.0,
        }
    path.write_text(
        json.dumps(
            {"duration_seconds": duration_seconds, "cameras": cameras},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_manifest(
    path: Path,
    *,
    host: str,
    config_path: Path,
    streams: Mapping[str, int],
    queue_size: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "created_wall_time_ns": time.time_ns(),
                "host": host,
                "config": str(config_path),
                "queue_size": queue_size,
                "streams": {
                    name: {"transport": "tim2_zmq", "port": port}
                    for name, port in streams.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record every enabled teleimager TIM2/ZMQ stream on Thor."
    )
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to record; zero records until Ctrl+C.",
    )
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    if args.queue_size <= 0:
        parser.error("--queue-size must be positive")
    return args


def _print_status(
    stats: Mapping[str, CameraStats],
    written: Mapping[str, int],
    elapsed_seconds: float,
    queued: int,
) -> None:
    for name, camera in stats.items():
        fps, mbps, gaps, drops = camera.report_interval(elapsed_seconds)
        print(
            f"[recorder] {name:20} {fps:5.1f} FPS | {mbps:5.1f} Mbps | "
            f"gaps={gaps} queue_drops={drops} written={written[name]}"
        )
    print(f"[recorder] writer queue: {queued}\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    streams = load_recording_streams(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}\n"
            "Use --overwrite or choose a new --output-dir."
        )

    write_manifest(
        args.output_dir / "manifest.json",
        host=args.host,
        config_path=args.config,
        streams=streams,
        queue_size=args.queue_size,
    )

    stats = {name: CameraStats() for name in streams}
    trackers = {name: SequenceTracker() for name in streams}
    written = {name: 0 for name in streams}
    work_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    writer_errors: queue.Queue = queue.Queue(maxsize=1)

    def writer_target() -> None:
        try:
            write_frames(
                work_queue,
                args.output_dir,
                args.output_dir / "frames.jsonl",
                written,
            )
        except BaseException as error:
            writer_errors.put(error)

    writer_thread = threading.Thread(
        target=writer_target,
        name="teleimager-frame-writer",
        daemon=True,
    )
    writer_thread.start()

    context = zmq.Context()
    poller = zmq.Poller()
    sockets: dict[zmq.Socket, str] = {}
    for name, port in streams.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{args.host}:{port}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = name
        print(f"[recorder] {name}: tcp://{args.host}:{port}")

    started_mono = time.monotonic()
    last_status = started_mono
    failure: Optional[BaseException] = None
    print(f"[recorder] writing to {args.output_dir}; Ctrl+C to stop")

    try:
        while args.duration == 0 or time.monotonic() - started_mono < args.duration:
            if not writer_errors.empty():
                raise RuntimeError("frame writer failed") from writer_errors.get_nowait()

            for socket in dict(poller.poll(timeout=100)):
                name = sockets[socket]
                packet = socket.recv()
                receive_wall_ns = time.time_ns()
                receive_mono_ns = time.monotonic_ns()
                try:
                    frame = decode_recording_frame(packet)
                except Exception:
                    stats[name].malformed += 1
                    continue

                epoch, gap, duplicate = trackers[name].observe(frame.sequence)
                if duplicate:
                    stats[name].duplicates += 1
                    continue
                stats[name].observe(len(packet), gap)
                job = FrameJob(
                    camera=name,
                    frame=frame,
                    epoch=epoch,
                    sequence_gap=gap,
                    packet_size=len(packet),
                    receive_wall_time_ns=receive_wall_ns,
                    receive_monotonic_time_ns=receive_mono_ns,
                )
                dropped = enqueue_latest(work_queue, job)
                if dropped is not None:
                    stats[dropped.camera].queue_drops += 1
                    stats[dropped.camera].interval_drops += 1

            now = time.monotonic()
            if now - last_status >= 1.0:
                _print_status(stats, written, now - last_status, work_queue.qsize())
                last_status = now
    except KeyboardInterrupt:
        print("\n[recorder] stopping...")
    except BaseException as error:
        failure = error
    finally:
        for socket in sockets:
            poller.unregister(socket)
            socket.close(linger=0)
        context.term()

        while writer_thread.is_alive():
            try:
                work_queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        writer_thread.join()

        duration = time.monotonic() - started_mono
        write_summary(
            args.output_dir / "summary.json",
            stats,
            written,
            duration_seconds=duration,
        )
        if not writer_errors.empty() and failure is None:
            failure = RuntimeError("frame writer failed")
            failure.__cause__ = writer_errors.get_nowait()
        print(f"[recorder] summary: {args.output_dir / 'summary.json'}")

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
