import argparse
from collections import deque
import contextlib
import time
from typing import Optional, Sequence

import cv2


class FPSMeter:
    """Measure frame arrival rate over a recent time window."""

    def __init__(self, window_seconds: float = 1.0):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_ns = int(window_seconds * 1_000_000_000)
        self._timestamps = deque()

    def tick(self, timestamp_ns: Optional[int] = None) -> None:
        now_ns = time.perf_counter_ns() if timestamp_ns is None else timestamp_ns
        if self._timestamps and now_ns < self._timestamps[-1]:
            raise ValueError("timestamps must be monotonic")

        self._timestamps.append(now_ns)
        cutoff_ns = now_ns - self._window_ns
        while self._timestamps[0] < cutoff_ns:
            self._timestamps.popleft()

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed_ns = self._timestamps[-1] - self._timestamps[0]
        if elapsed_ns == 0:
            return 0.0
        return (len(self._timestamps) - 1) * 1_000_000_000 / elapsed_ns


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview an HBVCAM stereo stream with measured capture FPS."
    )
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def _fourcc_name(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open camera: {args.device}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = _fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
    print(
        f"Opened {args.device}: requested {args.width}x{args.height}@{args.fps}, "
        f"negotiated {actual_width}x{actual_height}@{actual_fps:.1f} {actual_fourcc}"
    )
    print("Press q to exit.")

    meter = FPSMeter(window_seconds=1.0)
    last_report_ns = time.perf_counter_ns()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Failed to read a frame from {args.device}")

            now_ns = time.perf_counter_ns()
            meter.tick(now_ns)
            height, width = frame.shape[:2]
            cv2.line(frame, (width // 2, 0), (width // 2, height - 1), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Capture FPS: {meter.fps:.1f} / target {args.fps}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("HBVCAM stereo preview", frame)

            if now_ns - last_report_ns >= 1_000_000_000:
                print(f"Capture FPS: {meter.fps:.1f} | frame: {width}x{height}")
                last_report_ns = now_ns

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as exc:
        raise SystemExit(
            "OpenCV GUI is unavailable. Run teleimager-preview in the teleimager "
            "environment with GUI-enabled opencv-python."
        ) from exc
    finally:
        cap.release()
        with contextlib.suppress(cv2.error):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
