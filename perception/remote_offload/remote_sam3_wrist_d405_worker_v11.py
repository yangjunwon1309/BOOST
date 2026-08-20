"""Remote SAM3 worker for one vertically stacked pair of wrist D405 frames."""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import lcm
import numpy as np

from seg_web_node.lcm_protocol import decode_robot_state
from seg_web_node.robot_geometry import RobotStateBuffer
from seg_web_node.sam3_backend import TensorRTSam3Backend, TorchSam3Backend

from .wrist_d405_geometry_pipeline_v11 import (
    load_wrist_extrinsics,
    object_geometry,
)
from .wrist_d405_protocol_v11 import (
    CAMERA_IDS,
    WRIST_BUNDLE_CHANNEL,
    WRIST_RESULT_CHANNEL,
    WRIST_ROBOT_STATE_CHANNEL,
    make_result,
    pack_mask,
    pack_result,
    unpack_bundle,
    unpack_depth_z16,
)


LOG = logging.getLogger(__name__)


def parse_concepts(text: str) -> list[str]:
    concepts = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not concepts:
        raise ValueError("at least one SAM3 concept is required")
    return concepts


def stack_wrist_images(frames: dict[str, dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Return left-above-right BGR input, refusing mismatched camera grids."""
    decoded: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for camera_id in CAMERA_IDS:
        frame = frames[camera_id]
        image = cv2.imdecode(np.frombuffer(frame["color_jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"invalid JPEG from {camera_id}")
        expected = (int(frame["height"]), int(frame["width"]))
        if image.shape[:2] != expected:
            raise ValueError(f"JPEG grid mismatch from {camera_id}")
        if shape is None:
            shape = expected
        elif shape != expected:
            raise ValueError("left/right D405 images must have a common native grid")
        decoded[camera_id] = image
    assert shape is not None
    return np.vstack((decoded["left_wrist"], decoded["right_wrist"])), shape[0]


def attribute_stacked_masks(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: list[str],
                            *, split_y: int, seam_guard_px: int) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Attribute only seam-free masks to a local wrist grid; discard all others."""
    output: dict[str, list[dict[str, Any]]] = {camera_id: [] for camera_id in CAMERA_IDS}
    discarded = 0
    seam_start, seam_end = int(split_y) - int(seam_guard_px), int(split_y) + int(seam_guard_px)
    for mask, box, score, label in zip(np.asarray(masks, dtype=bool), boxes, scores, labels):
        if mask.ndim != 2 or mask.shape[0] != 2 * int(split_y):
            raise ValueError("SAM3 mask does not match the vertical stack")
        if mask[max(0, seam_start):min(mask.shape[0], seam_end + 1), :].any():
            discarded += 1
            continue
        top, bottom = int(mask[:split_y].sum()), int(mask[split_y:].sum())
        if top == bottom or max(top, bottom) == 0:
            discarded += 1
            continue
        camera_id = "left_wrist" if top > bottom else "right_wrist"
        local_mask = mask[:split_y] if camera_id == "left_wrist" else mask[split_y:]
        local_box = np.asarray(box, dtype=np.float64).copy()
        if camera_id == "right_wrist":
            local_box[[1, 3]] -= split_y
        local_box[[0, 2]] = np.clip(local_box[[0, 2]], 0, local_mask.shape[1] - 1)
        local_box[[1, 3]] = np.clip(local_box[[1, 3]], 0, local_mask.shape[0] - 1)
        output[camera_id].append({"mask": local_mask, "box": local_box, "score": float(score), "label": str(label)})
    return output, discarded


class MockSam3Backend:
    name = "mock"

    def infer(self, image_bgr: np.ndarray, concepts: list[str]):
        height, width = image_bgr.shape[:2]
        mask = np.zeros((1, height, width), dtype=bool)
        mask[0, height // 8:height // 3, width // 4:3 * width // 4] = True
        return mask, np.asarray([[width / 4, height / 8, 3 * width / 4, height / 3]]), np.asarray([0.99]), [concepts[0]]


class WristD405Sam3Worker:
    def __init__(self, *, wifi_lcm_url: str, concepts: list[str], backend: Any, extrinsics: dict[str, np.ndarray],
                 seam_guard_px: int, min_depth_m: float, max_depth_m: float, min_depth_points: int,
                 robot_state_max_delta_ms: float) -> None:
        self.bus = lcm.LCM(wifi_lcm_url)
        self.concepts, self.backend, self.extrinsics = concepts, backend, extrinsics
        self.seam_guard_px, self.min_depth_m, self.max_depth_m = int(seam_guard_px), float(min_depth_m), float(max_depth_m)
        self.min_depth_points, self.robot_state_max_delta_ms = int(min_depth_points), float(robot_state_max_delta_ms)
        self.states = RobotStateBuffer()
        self.lock = threading.Lock()
        self.latest_bundle: dict[str, Any] | None = None
        self.running = False
        self.sequence = 0
        self.next_track_id = {camera_id: 0 for camera_id in CAMERA_IDS}
        self.tracks: dict[str, dict[str, tuple[np.ndarray, int]]] = {camera_id: {} for camera_id in CAMERA_IDS}
        self.bus.subscribe(WRIST_BUNDLE_CHANNEL, self._on_bundle)
        self.bus.subscribe(WRIST_ROBOT_STATE_CHANNEL, self._on_robot_state)

    def _on_bundle(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                self.latest_bundle = unpack_bundle(data)
        except Exception as error:
            LOG.warning("rejected wrist D405 bundle: %s", error)

    def _on_robot_state(self, _channel: str, data: bytes) -> None:
        try:
            with self.lock:
                self.states.add(decode_robot_state(data))
        except Exception as error:
            LOG.warning("rejected wrist robot state: %s", error)

    def process_latest(self) -> dict[str, Any] | None:
        with self.lock:
            bundle = self.latest_bundle
            self.latest_bundle = None
        if bundle is None:
            return None
        frames = bundle["cameras"]
        stacked, split_y = stack_wrist_images(frames)
        masks, boxes, scores, labels = self.backend.infer(stacked, self.concepts)
        attributed, discarded = attribute_stacked_masks(masks, boxes, scores, labels, split_y=split_y, seam_guard_px=self.seam_guard_px)
        objects: list[dict[str, Any]] = []
        for camera_id in CAMERA_IDS:
            frame = frames[camera_id]
            state_match = self.states.nearest(int(frame["host_timestamp_us"]), self.robot_state_max_delta_ms)
            state, state_age_ms = state_match if state_match is not None else (None, None)
            depth = unpack_depth_z16(frame["depth_z16_zlib"], width=int(frame["width"]), height=int(frame["height"]))
            for detected in attributed[camera_id]:
                obj = object_geometry(
                    camera_id=camera_id, serial=str(frame["serial"]), mask=detected["mask"], box_xyxy=detected["box"].tolist(),
                    label=detected["label"], score=detected["score"], depth_z16=depth, depth_scale_m=float(frame["depth_scale_m"]),
                    intrinsics=frame["intrinsics"], source_timestamp_us=int(frame["host_timestamp_us"]),
                    calibration_id=str(frame.get("calibration_id", "factory")), state=state, state_age_ms=state_age_ms,
                    wrist_extrinsics=self.extrinsics, min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
                    min_depth_points=self.min_depth_points,
                )
                obj.update(
                    mask_encoding="bitpack-zlib-row-major",
                    mask_data=pack_mask(detected["mask"]),
                )
                obj["track_id"] = self._assign_track(camera_id, obj)
                objects.append(obj)
        self.sequence += 1
        result = make_result(sequence=self.sequence, source_bundle_timestamp_us=int(bundle["bundle_timestamp_us"]),
                             objects=objects, discarded_seam_masks=discarded)
        self.bus.publish(WRIST_RESULT_CHANNEL, pack_result(result))
        return result

    def _assign_track(self, camera_id: str, obj: dict[str, Any]) -> str:
        """Associate only within one camera namespace; never cross wrist seam."""
        center = obj.get("center_goal_m") or obj.get("center_camera_m")
        if center is None:
            track_id = self.next_track_id[camera_id]
            self.next_track_id[camera_id] += 1
            return f"{camera_id}:{obj['label']}:track{track_id:04d}"
        point = np.asarray(center, dtype=np.float64).reshape(3)
        label = str(obj["label"])
        candidates = [
            (float(np.linalg.norm(point - old_center)), key)
            for key, (old_center, _last_sequence) in self.tracks[camera_id].items()
            if key.startswith(f"{label}:track")
        ]
        if candidates and min(candidates)[0] <= 0.25:
            local_id = min(candidates)[1]
        else:
            local_id = f"{label}:track{self.next_track_id[camera_id]:04d}"
            self.next_track_id[camera_id] += 1
        self.tracks[camera_id][local_id] = (point, self.sequence)
        for key, (_center, last_sequence) in list(self.tracks[camera_id].items()):
            if self.sequence - last_sequence > 5:
                del self.tracks[camera_id][key]
        return f"{camera_id}:{local_id}"

    def run(self) -> None:
        self.running = True
        while self.running:
            self.bus.handle_timeout(20)
            self.process_latest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wifi-lcm-url", required=True)
    parser.add_argument("--concepts", default="object")
    parser.add_argument("--backend", choices=("mock", "pytorch", "tensorrt"), default="pytorch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--trt-engine", type=Path)
    parser.add_argument("--extrinsics", type=Path, default=Path("config/wrist_d405_extrinsics_v11.yaml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area-px", type=int, default=100)
    parser.add_argument("--seam-guard-px", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.07)
    parser.add_argument("--max-depth-m", type=float, default=0.50)
    parser.add_argument("--min-depth-points", type=int, default=100)
    parser.add_argument("--robot-state-max-delta-ms", type=float, default=15.0)
    args = parser.parse_args()
    if args.backend == "mock":
        backend: Any = MockSam3Backend()
    elif args.backend == "pytorch":
        backend = TorchSam3Backend(checkpoint=args.checkpoint, threshold=args.threshold, min_area_px=args.min_area_px)
    else:
        backend = TensorRTSam3Backend(engine_path=args.trt_engine, model_id="facebook/sam3", threshold=args.threshold, min_area_px=args.min_area_px)
    WristD405Sam3Worker(wifi_lcm_url=args.wifi_lcm_url, concepts=parse_concepts(args.concepts), backend=backend,
                        extrinsics=load_wrist_extrinsics(args.extrinsics), seam_guard_px=args.seam_guard_px,
                        min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m, min_depth_points=args.min_depth_points,
                        robot_state_max_delta_ms=args.robot_state_max_delta_ms).run()


if __name__ == "__main__":
    main()
