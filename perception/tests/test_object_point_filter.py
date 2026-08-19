import numpy as np

from seg_web_node.geometry import CameraIntrinsics
from seg_web_node.object_point_filter import (
    ObjectPointFilterConfig,
    compute_depth_edge_mask,
    extract_filtered_object_cloud,
)


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(width, height, 200.0, 200.0, width / 2, height / 2)


def test_object_filter_rejects_mask_edges_depth_jumps_and_mad_outliers():
    depth = np.ones((80, 80), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[10:70, 10:70] = True
    depth[10:70, 10:12] = 1.40
    depth[35:38, 35:38] = 1.35
    depth[45, 45] = 0.45
    config = ObjectPointFilterConfig(
        erosion_px=2,
        depth_edge_threshold_m=0.03,
        mad_scale=3.5,
        min_depth_band_m=0.02,
        voxel_size_m=0.005,
        radius_outlier_m=0.018,
        min_radius_neighbors=3,
    )

    cloud = extract_filtered_object_cloud(
        depth,
        mask,
        _intrinsics(80, 80),
        config,
        min_depth_m=0.15,
        max_depth_m=3.0,
        min_valid_points=100,
        max_points=8000,
    )

    assert cloud is not None
    np.testing.assert_allclose(cloud.center_camera_m[2], 1.0, atol=1.0e-6)
    assert cloud.stats["after_erosion_pixels"] < cloud.stats["input_mask_pixels"]
    assert cloud.stats["after_mad_pixels"] < cloud.stats["after_erosion_pixels"]
    assert cloud.valid_depth_points >= 100


def test_radius_filter_removes_a_disconnected_mask_point():
    depth = np.ones((80, 80), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[25:55, 25:55] = True
    mask[2, 2] = True
    config = ObjectPointFilterConfig(
        erosion_px=0,
        depth_edge_threshold_m=0.0,
        mad_scale=0.0,
        voxel_size_m=0.005,
        radius_outlier_m=0.018,
        min_radius_neighbors=3,
    )

    cloud = extract_filtered_object_cloud(
        depth,
        mask,
        _intrinsics(80, 80),
        config,
        min_depth_m=0.15,
        max_depth_m=3.0,
        min_valid_points=100,
        max_points=8000,
    )

    assert cloud is not None
    assert cloud.stats["after_radius_points"] == cloud.stats["after_voxel_points"] - 1
    assert np.linalg.norm(cloud.center_camera_m[:2]) < 0.01


def test_injected_depth_edge_mask_is_bit_identical_to_internal_path():
    depth = np.ones((80, 80), dtype=np.float32)
    depth[:, 42:] = 1.4
    depth[30:33, 25:28] = np.nan
    mask = np.zeros_like(depth, dtype=bool)
    mask[10:70, 10:70] = True
    config = ObjectPointFilterConfig(
        erosion_px=1,
        depth_edge_threshold_m=0.03,
        mad_scale=0.0,
        voxel_size_m=0.0,
        radius_outlier_m=0.0,
    )
    arguments = {
        "min_depth_m": 0.15,
        "max_depth_m": 3.0,
        "min_valid_points": 100,
        "max_points": 8000,
    }
    internal = extract_filtered_object_cloud(
        depth,
        mask,
        _intrinsics(80, 80),
        config,
        **arguments,
    )
    edge_mask = compute_depth_edge_mask(
        depth,
        min_depth_m=arguments["min_depth_m"],
        max_depth_m=arguments["max_depth_m"],
        threshold_m=config.depth_edge_threshold_m,
    )
    injected = extract_filtered_object_cloud(
        depth,
        mask,
        _intrinsics(80, 80),
        config,
        depth_edge_mask=edge_mask,
        **arguments,
    )

    assert internal is not None and injected is not None
    np.testing.assert_array_equal(injected.clean_mask, internal.clean_mask)
    np.testing.assert_array_equal(injected.points_camera_m, internal.points_camera_m)
    assert injected.stats == internal.stats
