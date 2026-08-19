import threading

import numpy as np

from remote_offload.protocol import (
    VERSION,
    ValidationContext,
    decode_mask,
    encode_mask,
    make_empty_result,
    make_object,
    now_us,
    normalize_concepts,
    pack_concept_control,
    pack_result,
    unpack_concept_control,
    unpack_result,
    validate_result,
)
from remote_offload.remote_sam3_worker import RemoteSam3Worker
from remote_offload.remote_web_server import RemoteWebRuntime


class _ResetRecorder:
    def __init__(self):
        self.reset_count = 0

    def reset_tracking(self):
        self.reset_count += 1


def test_remote_concept_control_round_trip_and_validation():
    payload = {
        "version": 1,
        "published_timestamp_us": now_us(),
        "revision": 7,
        "concepts": [" bottle ", "red   cup", "bottle"],
    }
    decoded = unpack_concept_control(pack_concept_control(payload))
    assert decoded["concepts"] == ["bottle", "red cup"]
    assert decoded["revision"] == 7
    assert normalize_concepts("shoe, cup") == ["shoe", "cup"]


def test_web_concept_command_is_applied_once_at_worker_frame_boundary():
    web = RemoteWebRuntime.__new__(RemoteWebRuntime)
    web.lock = threading.Lock()
    web.concept_revision = 100
    web.pending_control_packets = []
    web.stats = {"concept_commands": 0}
    command = web.set_concepts("shoe, red cup")

    assert command["concepts"] == ["shoe", "red cup"]
    assert len(web.pending_control_packets) == 3
    assert len(set(web.pending_control_packets)) == 1

    worker = RemoteSam3Worker.__new__(RemoteSam3Worker)
    worker.lock = threading.Lock()
    worker.concepts = ["bottle"]
    worker.pending_concepts = None
    worker.concept_revision = 0
    worker.geometry_pipeline = _ResetRecorder()
    worker._record_error = lambda error: (_ for _ in ()).throw(error)

    packet = web.pending_control_packets[0]
    worker._on_concept_control("unused", packet)
    assert worker._apply_pending_concepts() == ["shoe", "red cup"]
    assert worker.geometry_pipeline.reset_count == 1

    worker._on_concept_control("unused", packet)
    assert worker.pending_concepts is None
    assert worker._apply_pending_concepts() == ["shoe", "red cup"]
    assert worker.geometry_pipeline.reset_count == 1


def test_remote_mask_bitpack_round_trip_little_endian():
    mask = np.zeros((5, 9), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[3, 7] = True
    decoded = decode_mask(encode_mask(mask), 5, 9)
    np.testing.assert_array_equal(decoded, mask)


def test_remote_result_msgpack_schema_validates():
    timestamp = now_us()
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    payload = make_empty_result(
        sequence=7,
        published_timestamp_us=timestamp,
        source_rgb_timestamp_us=111,
        source_depth_timestamp_us=112,
        calibration_id=3,
        image_width=6,
        image_height=4,
        backend="sam3_pytorch",
        inference_ms=12.5,
    )
    payload["objects"].append(
        make_object(
            track_id="object:track0007",
            label="object",
            score=0.91,
            box_xyxy=[2, 1, 4, 2],
            mask=mask,
            median_depth_m=0.8,
            center_pixel_uv=[3, 2],
            center_camera_m=[0.1, 0.2, 0.8],
            center_goal_m=[0.3, 0.2, 0.5],
            valid_depth_points=6,
        )
    )
    payload["stage_us"] = {
        "source_rgb": 111,
        "worker_dequeue": timestamp + 100,
        "decode_done": timestamp + 200,
        "infer_done": timestamp + 300,
        "geometry_done": timestamp + 400,
        "packed": timestamp + 500,
    }
    unpacked = unpack_result(pack_result(payload))
    errors = validate_result(
        unpacked,
        ValidationContext(
            result_max_age_ms=800,
            expected_rgb_timestamp_us=111,
            expected_depth_timestamp_us=112,
            expected_calibration_id=3,
            expected_image_width=6,
            expected_image_height=4,
            previous_sequence=6,
            now_timestamp_us=timestamp + 1000,
        ),
    )
    assert errors == []
    decoded = decode_mask(unpacked["objects"][0]["mask_data"], 4, 6)
    np.testing.assert_array_equal(decoded, mask)


def test_remote_result_stage_timestamps_are_optional_but_validated_when_present():
    timestamp = now_us()
    payload = make_empty_result(
        sequence=1,
        published_timestamp_us=timestamp,
        source_rgb_timestamp_us=101,
        source_depth_timestamp_us=102,
        image_width=8,
        image_height=8,
        backend="mock",
    )
    context = ValidationContext(
        result_max_age_ms=800,
        now_timestamp_us=timestamp,
    )
    assert validate_result(payload, context) == []
    payload["stage_us"] = {"source_rgb": 101}
    errors = validate_result(payload, context)
    assert any("stage_us.worker_dequeue is invalid" in error for error in errors)


def test_remote_result_validator_rejects_stale_mismatched_or_nonmonotonic():
    timestamp = now_us()
    payload = make_empty_result(
        sequence=4,
        published_timestamp_us=timestamp - 2_000_000,
        source_rgb_timestamp_us=10,
        source_depth_timestamp_us=11,
        calibration_id=1,
        image_width=8,
        image_height=8,
        backend="sam3_pytorch",
    )
    payload["version"] = VERSION
    errors = validate_result(
        payload,
        ValidationContext(
            result_max_age_ms=800,
            expected_rgb_timestamp_us=99,
            expected_depth_timestamp_us=11,
            expected_calibration_id=2,
            expected_image_width=8,
            expected_image_height=8,
            previous_sequence=4,
            now_timestamp_us=timestamp,
        ),
    )
    assert any("result age" in error for error in errors)
    assert any("source_rgb_timestamp_us mismatch" in error for error in errors)
    assert any("calibration_id mismatch" in error for error in errors)
    assert any("sequence is not monotonic" in error for error in errors)
