import numpy as np

from seg_web_node.geometry_fallback import LastValidGeometryFallback
from seg_web_node.robot_geometry import RobotStateBuffer, camera_goal_transform


class _SurfaceTracker:
    def status(self, track_id):
        return {
            "type": "cylinder",
            "center_goal_m": [0.3, 0.1, 0.2],
            "axis_goal": [0.0, 0.0, 1.0],
            "radius_m": 0.03,
            "height_m": 0.2,
            "fit_rmse_m": 0.004,
            "inlier_ratio": 0.8,
            "angular_coverage": 0.5,
            "updates": 4,
            "stale_frames": 1,
        }


def _object():
    return {
        "track_id": "bottle:track0001",
        "label": "bottle",
        "score": 0.9,
        "mask_width": 640,
        "mask_height": 360,
        "center_goal_m": [0.3, 0.1, 0.2],
        "target_contact_goal_m": [0.32, 0.12, 0.18],
    }


def test_geometry_fallback_holds_then_expires():
    cache = LastValidGeometryFallback(ttl_frames=2, score_decay=0.9)
    tracker = _SurfaceTracker()

    fresh = cache.complete_frame([_object()], tracker, image_width=640, image_height=360)
    first = cache.complete_frame([], tracker, image_width=640, image_height=360)
    second = cache.complete_frame([], tracker, image_width=640, image_height=360)
    expired = cache.complete_frame([], tracker, image_width=640, image_height=360)

    assert fresh[0]["geometry_fallback"] is False
    assert first[0]["geometry_fallback"] is True
    assert first[0]["missing_frames"] == 1
    assert np.isclose(first[0]["score"], 0.81)
    assert second[0]["missing_frames"] == 2
    assert second[0]["surface_model"]["stale_frames"] == 1
    assert expired == []


def test_identity_goal_transform_does_not_require_robot_state():
    t_torso_cam = np.eye(4)
    t_torso_cam[:3, 3] = [0.1, 0.2, 0.3]

    transform = camera_goal_transform(
        t_torso_cam,
        "identity",
        RobotStateBuffer(),
        123456,
        40.0,
    )

    np.testing.assert_allclose(transform, t_torso_cam)
