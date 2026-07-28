import numpy as np

from seg_web_node.geometry import central_mask_geometry_filter


def _plane_xyz(height: int = 20, width: int = 20, z: float = 0.4) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return np.stack([x * 0.001, y * 0.001, np.full_like(x, z, dtype=float)], axis=-1)


def test_central_geometry_rejects_area_and_range():
    xyz = _plane_xyz()
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, :, :10] = True  # 50% area
    masks[1, 2:6, 2:6] = True
    xyz[2:6, 2:6, 2] = 0.8
    keep, rejected = central_mask_geometry_filter(
        masks, xyz,
        max_area_ratio=0.20,
        max_center_distance_m=0.60,
        plane_min_points=300,
        plane_max_rms_m=0.015,
        support_normal_up_dot_min=0.85,
        t_goal_cam=np.eye(4),
    )
    assert len(keep) == 0
    assert {item["reason"] for item in rejected} == {"area", "distance"}


def test_central_geometry_rejects_flat_horizontal_support_plane():
    xyz = _plane_xyz()
    masks = np.ones((1, 20, 20), dtype=bool)
    keep, rejected = central_mask_geometry_filter(
        masks, xyz,
        max_area_ratio=1.0,
        max_center_distance_m=0.60,
        plane_min_points=100,
        plane_max_rms_m=0.015,
        support_normal_up_dot_min=0.85,
        t_goal_cam=np.eye(4),
    )
    assert len(keep) == 0
    assert rejected[0]["reason"] == "horizontal_flat_plane"
