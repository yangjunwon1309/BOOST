import numpy as np

from seg_web_node.depth_based_filter import DepthBasedRobotFilter


def test_depth_veto_removes_robot_depth_but_keeps_closer_object():
    filt = DepthBasedRobotFilter(
        {
            "enabled": True,
            "margin_high_m": 0.06,
            "margin_low_m": 0.03,
            "bilateral_diameter": 0,
        }
    )
    robot_depth = np.full((1, 3), 1.0, dtype=np.float32)
    measured = np.asarray([[0.90, 0.98, 1.04]], dtype=np.float32)

    robot_mask = filt._depth_veto(measured, robot_depth)

    assert robot_mask.tolist() == [[False, True, True]]


def test_depth_veto_hysteresis_holds_then_releases():
    filt = DepthBasedRobotFilter(
        {
            "enabled": True,
            "margin_high_m": 0.06,
            "margin_low_m": 0.03,
            "bilateral_diameter": 0,
        }
    )
    robot_depth = np.asarray([[1.0]], dtype=np.float32)

    assert filt._depth_veto(np.asarray([[0.98]], dtype=np.float32), robot_depth)[0, 0]
    assert filt._depth_veto(np.asarray([[0.95]], dtype=np.float32), robot_depth)[0, 0]
    assert not filt._depth_veto(np.asarray([[0.92]], dtype=np.float32), robot_depth)[0, 0]


def test_filter_mask_uses_current_robot_mask():
    filt = DepthBasedRobotFilter({"enabled": True})
    filt._robot_mask = np.zeros((4, 4), dtype=bool)
    filt._robot_mask[1:3, 1:3] = True
    filt.last_status = filt._empty_status(None)
    mask = np.ones((4, 4), dtype=bool)

    filtered = filt.filter_mask(mask, min_area_px=4)

    assert filtered is not None
    assert int(filtered.sum()) == 12
    assert filt.metadata()["trimmed_masks"] == 1


def test_filter_mask_scales_geometry_robot_mask_to_full_rgb_mask_grid():
    filt = DepthBasedRobotFilter({"enabled": True})
    filt._robot_mask = np.zeros((2, 2), dtype=bool)
    filt._robot_mask[0, 0] = True
    filt.last_status = filt._empty_status(None)
    mask = np.ones((4, 4), dtype=bool)

    filtered = filt.filter_mask(mask, min_area_px=4)

    assert filtered is not None
    assert not filtered[:2, :2].any()
    assert filtered[2:, 2:].all()
    assert int(filtered.sum()) == 12
