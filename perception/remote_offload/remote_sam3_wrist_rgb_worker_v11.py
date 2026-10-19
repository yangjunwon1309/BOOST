"""Run SAM3 on Thor-relayed UVC wrist RGB frames, without 3-D targeting."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import lcm
import numpy as np

from seg_web_node.sam3_backend import TensorRTSam3Backend, TorchSam3Backend

from .wrist_rgb_protocol_v11 import WRIST_RGB_FRAME_CHANNEL, now_us, unpack_rgb_frame


LOG = logging.getLogger("remote_sam3_wrist_rgb")


def parse_concepts(text: str) -> list[str]:
    concepts = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not concepts:
        raise ValueError("at least one SAM3 concept is required")
    return concepts


class WristRgbOnlySam3Worker:
    def __init__(self, *, lcm_url: str, concepts: list[str], backend: Any, max_frame_age_ms: float) -> None:
        self.bus = lcm.LCM(lcm_url)
        self.concepts = concepts
        self.backend = backend
        self.max_frame_age_us = max(1, int(max_frame_age_ms * 1_000.0))
        self.latest: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.last_processed: tuple[int, int] | None = None
        self.sequence = 0
        self.bus.subscribe(WRIST_RGB_FRAME_CHANNEL, self._on_frame)

    def _on_frame(self, _channel: str, data: bytes) -> None:
        try:
            frame = unpack_rgb_frame(data)
        except Exception as error:
            LOG.warning("discarding malformed UVC wrist frame: %s", error)
            return
        with self.lock:
            self.latest[str(frame["camera_id"])] = frame

    def _process_latest(self) -> None:
        with self.lock:
            left = self.latest.get("left_wrist")
            right = self.latest.get("right_wrist")
        if left is None or right is None:
            return
        signature = (
            int(left["thor_receive_timestamp_us"]),
            int(right["thor_receive_timestamp_us"]),
        )
        if signature == self.last_processed:
            return
        self.last_processed = signature

        now = now_us()
        ages_ms = [(now - timestamp_us) / 1_000.0 for timestamp_us in signature]
        if max(ages_ms) > self.max_frame_age_us / 1_000.0:
            LOG.warning("waiting for fresh left/right UVC frames; receive ages_ms=%s", [round(x, 1) for x in ages_ms])
            return

        images: list[np.ndarray] = []
        for frame in (left, right):
            jpeg_array = np.frombuffer(frame["jpeg"], dtype=np.uint8)
            image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
            if image is None:
                LOG.warning("failed to decode %s JPEG", frame["camera_id"])
                return
            expected_shape = (int(frame["height"]), int(frame["width"]))
            if image.shape[:2] != expected_shape:
                LOG.warning(
                    "%s JPEG size %s differs from configured shape %s",
                    frame["camera_id"],
                    image.shape[:2],
                    expected_shape,
                )
                return
            images.append(image)

        if images[0].shape != images[1].shape:
            LOG.warning("left/right UVC images have different dimensions")
            return
        split_y = images[0].shape[0]
        stacked = np.vstack(images)
        _masks, boxes, scores, labels = self.backend.infer(stacked, self.concepts)
        boxes = np.asarray(boxes).reshape((-1, 4)) if len(boxes) else np.empty((0, 4))
        detections = {"left_wrist": [], "right_wrist": []}
        for index, box in enumerate(boxes):
            side = "left_wrist" if (float(box[1]) + float(box[3])) * 0.5 < split_y else "right_wrist"
            label = str(labels[index]) if index < len(labels) else "object"
            score = float(scores[index]) if index < len(scores) else 0.0
            detections[side].append((label, score))

        self.sequence += 1
        LOG.info(
            "rgb SAM3 seq=%d image=%dx%d receive_skew_ms=%.1f detections_left=%d detections_right=%d labels=%s",
            self.sequence,
            images[0].shape[1],
            images[0].shape[0],
            abs(signature[0] - signature[1]) / 1_000.0,
            len(detections["left_wrist"]),
            len(detections["right_wrist"]),
            {
                side: [(label, round(score, 3)) for label, score in values]
                for side, values in detections.items()
            },
        )

    def run(self) -> None:
        LOG.info(
            "RGB-only SAM3 worker ready; subscribing %s concepts=%s backend=%s; geometry and target publishing are disabled",
            WRIST_RGB_FRAME_CHANNEL,
            self.concepts,
            getattr(self.backend, "name", type(self.backend).__name__),
        )
        while True:
            self.bus.handle_timeout(20)
            while self.bus.handle_timeout(0) > 0:
                pass
            self._process_latest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--concepts", default="object")
    parser.add_argument("--backend", choices=("pytorch", "tensorrt"), default="pytorch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trt-engine", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-px", type=int, default=100)
    parser.add_argument("--max-frame-age-ms", type=float, default=500.0)
    args = parser.parse_args()
    concepts = parse_concepts(args.concepts)
    if args.backend == "pytorch":
        backend: Any = TorchSam3Backend(
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            min_area_px=args.min_area_px,
        )
    else:
        if args.trt_engine is None:
            parser.error("--trt-engine is required for the TensorRT backend")
        backend = TensorRTSam3Backend(
            engine_path=args.trt_engine,
            model_id="facebook/sam3",
            threshold=args.threshold,
            min_area_px=args.min_area_px,
        )
    WristRgbOnlySam3Worker(
        lcm_url=args.wifi_lcm_url,
        concepts=concepts,
        backend=backend,
        max_frame_age_ms=args.max_frame_age_ms,
    ).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
