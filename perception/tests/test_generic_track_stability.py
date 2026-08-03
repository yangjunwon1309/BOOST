import numpy as np

from seg_web_node.depth_based_filter import DepthBasedRobotFilter
from seg_web_node.sam2_backend import Sam2HieraTinyBackend
from seg_web_node.yolo11_backend import Yolo11SegBackend


class _PredictNothing:
    def predict(self, **_kwargs):
        return np.empty((0, 8, 8), dtype=bool), np.empty(0, dtype=np.float32), None


def _backend() -> Sam2HieraTinyBackend:
    """Construct only the state used by pure tracking helpers (no CUDA model)."""
    backend = Sam2HieraTinyBackend.__new__(Sam2HieraTinyBackend)
    backend.threshold = 0.5
    backend.min_area_px = 4
    backend.max_masks = 4
    backend.track_missing_ttl_frames = 3
    backend.track_fallback_score_decay = 0.9
    backend.reinitialize_match_iou = 0.15
    backend.depth_self_filter = DepthBasedRobotFilter({"enabled": False})
    backend.predictor = _PredictNothing()
    backend.track_priors = []
    return backend


def test_box_tracking_keeps_last_mask_through_short_predictor_dropout():
    backend = _backend()
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    backend.track_priors = [{"box": np.array([2, 2, 5, 5], dtype=np.float32), "mask": mask, "score": 0.9, "missing": 0}]

    output = backend._track()

    assert len(output) == 1
    np.testing.assert_array_equal(output[0][0], mask)
    assert output[0][1] == 0.81
    assert backend.track_priors[0]["missing"] == 1


def test_reinitialize_preserves_unmatched_track_until_ttl():
    backend = _backend()
    old = np.zeros((8, 8), dtype=bool)
    old[1:5, 1:5] = True
    backend.track_priors = [{"box": np.array([1, 1, 4, 4], dtype=np.float32), "mask": old, "score": 0.9, "missing": 0}]
    backend._discover = lambda _image: []

    output = backend._reinitialize(np.zeros((8, 8, 3), dtype=np.uint8))

    assert len(output) == 1
    np.testing.assert_array_equal(output[0][0], old)
    assert backend.track_priors[0]["missing"] == 1


def test_yolo_association_keeps_a_missing_coco_instance_for_ttl():
    backend = Yolo11SegBackend.__new__(Yolo11SegBackend)
    backend.min_area_px = 4
    backend.max_masks = 4
    backend.track_missing_ttl_frames = 3
    backend.track_fallback_score_decay = 0.9
    backend.reinitialize_match_iou = 0.15
    backend.confidence = 0.8
    backend.depth_self_filter = DepthBasedRobotFilter({"enabled": False})
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    backend.track_priors = [{
        "label": "bottle", "mask": mask,
        "box": np.array([2, 2, 5, 5], dtype=np.float32), "score": 0.9, "missing": 0,
    }]

    output = backend._update_tracks([])

    assert output == []  # retained internally, never a stale visible/belief mask
    assert len(backend.track_priors) == 1
    assert backend.track_priors[0]["label"] == "bottle"
    assert backend.track_priors[0]["missing"] == 1
    assert backend.track_priors[0]["score"] == 0.81


def test_yolo_new_same_label_replaces_stale_viewpoint_mask_without_ghost():
    backend = Yolo11SegBackend.__new__(Yolo11SegBackend)
    backend.min_area_px = 4
    backend.max_masks = 4
    backend.track_missing_ttl_frames = 6
    backend.track_fallback_score_decay = 0.985
    backend.reinitialize_match_iou = 0.15
    backend.confidence = 0.8
    backend.depth_self_filter = DepthBasedRobotFilter({"enabled": False})
    old = np.zeros((12, 12), dtype=bool); old[1:5, 1:5] = True
    new = np.zeros((12, 12), dtype=bool); new[7:11, 7:11] = True
    backend.track_priors = [{
        "label": "bottle", "mask": old,
        "box": np.array([1, 1, 4, 4], dtype=np.float32), "score": 0.95, "missing": 0,
    }]

    output = backend._update_tracks([{
        "label": "bottle", "mask": new,
        "box": np.array([7, 7, 10, 10], dtype=np.float32), "score": 0.93, "missing": 0,
    }])

    assert len(output) == 1
    np.testing.assert_array_equal(output[0]["mask"], new)
    assert len(backend.track_priors) == 1
