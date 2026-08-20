from __future__ import annotations

import numpy as np

from remote_offload.remote_sam3_wrist_d405_worker_v11 import (
    attribute_stacked_masks,
    stack_wrist_images,
)
from remote_offload.wrist_d405_protocol_v11 import (
    pack_depth_z16,
    unpack_depth_z16,
)


def test_d405_depth_round_trip_preserves_native_grid() -> None:
    depth = np.arange(640 * 360, dtype=np.uint16).reshape(360, 640)
    restored = unpack_depth_z16(pack_depth_z16(depth), width=640, height=360)
    np.testing.assert_array_equal(restored, depth)


def test_vertical_masks_map_to_their_local_wrist_grid() -> None:
    masks = np.zeros((2, 720, 640), dtype=bool)
    masks[0, 20:40, 10:30] = True
    masks[1, 500:520, 10:30] = True
    attributed, discarded = attribute_stacked_masks(
        masks, np.array([[10, 20, 30, 40], [10, 500, 30, 520]], dtype=float),
        np.array([0.9, 0.8]), ["top", "bottom"], split_y=360, seam_guard_px=8,
    )
    assert discarded == 0
    assert len(attributed["left_wrist"]) == 1
    assert len(attributed["right_wrist"]) == 1
    assert attributed["right_wrist"][0]["box"][1] == 140
    assert attributed["left_wrist"][0]["mask"].shape == (360, 640)


def test_seam_touching_mask_is_discarded_not_split() -> None:
    masks = np.zeros((1, 720, 640), dtype=bool)
    masks[0, 351:370, 100:120] = True
    attributed, discarded = attribute_stacked_masks(
        masks, np.array([[100, 351, 120, 370]], dtype=float), np.array([0.9]), ["unsafe"],
        split_y=360, seam_guard_px=8,
    )
    assert discarded == 1
    assert not attributed["left_wrist"] and not attributed["right_wrist"]
