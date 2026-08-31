import json

import numpy as np

from path_core.messaging.zmq_wire import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


def _header(packet: bytes, topic: bytes) -> dict:
    assert packet.startswith(topic)
    raw = packet[len(topic) : len(topic) + HEADER_SIZE].split(b"\0", 1)[0]
    return json.loads(raw.decode("utf-8"))


def test_command_wire_header_and_timing_payload_are_stable():
    packet = build_command_message(
        True,
        False,
        True,
        0.25,
        button_sequence=9,
        button_read_timestamp_us=10,
    )
    header = _header(packet, b"command")
    assert header["v"] == 1
    assert [field["name"] for field in header["fields"]] == [
        "start",
        "stop",
        "planner",
        "delta_heading",
        "button_sequence",
        "button_read_timestamp_us",
    ]


def test_planner_and_pose_packets_preserve_field_names_and_dtypes():
    planner = build_planner_message(
        7,
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        left_hand_position=[0.1, 0.2],
    )
    assert [field["name"] for field in _header(planner, b"planner")["fields"]][-1] == "left_hand_joints"

    pose = pack_pose_message(
        {
            "position": np.array([1.0, 2.0], dtype=np.float32),
            "timestamp": np.array([3], dtype=np.int64),
        },
        topic="vr_pose_debug",
    )
    fields = _header(pose, b"vr_pose_debug")["fields"]
    assert fields == [
        {"name": "position", "dtype": "f32", "shape": [2]},
        {"name": "timestamp", "dtype": "i64", "shape": [1]},
    ]
