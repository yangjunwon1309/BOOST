import numpy as np

from remote_offload.protocol import (
    VERSION,
    ValidationContext,
    decode_mask,
    encode_mask,
    make_empty_result,
    make_object,
    now_us,
    pack_result,
    unpack_result,
    validate_result,
)


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

