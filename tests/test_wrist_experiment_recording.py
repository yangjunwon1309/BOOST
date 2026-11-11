from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from services.pose_service.tools.record_wrist_experiment_v11 import (
    CSV_FIELDS,
    SCHEMA_VERSION,
    CameraImageWriter,
)
from teleimager.recording_protocol import RecordingFrame, decode_recording_frame, encode_recording_frame


def test_schema_v3_keeps_pose_and_hand_data_without_legacy_belief_columns() -> None:
    assert SCHEMA_VERSION == 3
    assert "left_hand_json" in CSV_FIELDS and "right_hand_json" in CSV_FIELDS
    assert not any("belief" in field or "objects_json" in field for field in CSV_FIELDS)


def test_fake_tim2_frame_is_saved_as_original_jpeg_and_indexed(tmp_path: Path) -> None:
    jpeg = b"\xff\xd8boost-test\xff\xd9"
    packet = encode_recording_frame(
        RecordingFrame(
            sequence=17,
            capture_wall_time_us=123456,
            capture_monotonic_time_ns=789012,
            width=640,
            height=480,
            jpeg=jpeg,
        )
    )
    decoded = decode_recording_frame(packet)
    frame = SimpleNamespace(
        jpg=decoded.jpeg,
        sequence=decoded.sequence,
        capture_wall_time_us=decoded.capture_wall_time_us,
        capture_monotonic_time_ns=decoded.capture_monotonic_time_ns,
        receive_monotonic_ns=345678,
        width=decoded.width,
        height=decoded.height,
    )
    trial_dir = tmp_path / "path_wrist_experiment_001"
    writer = CameraImageWriter(trial_dir)
    writer.on_frame("left_wrist_camera", frame)
    summary = writer.close()

    image_path = trial_dir / "images" / "left_wrist" / "17.jpg"
    assert image_path.read_bytes() == jpeg
    with (trial_dir / "images_index.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "camera": "left_wrist",
            "sequence": "17",
            "capture_wall_time_us": "123456",
            "capture_monotonic_time_ns": "789012",
            "receive_monotonic_ns": "345678",
            "width": "640",
            "height": "480",
            "file": "images/left_wrist/17.jpg",
        }
    ]
    assert summary["left_wrist"]["saved_frames"] == 1
