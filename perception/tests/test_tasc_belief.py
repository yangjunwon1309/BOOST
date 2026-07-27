import numpy as np

from seg_web_node.feature_cost import (
    FeatureConfig,
    MultiFeatureCost,
    huber_like,
)
from seg_web_node.grip_candidates import (
    GripCandidate,
    GripCandidateConfig,
    generate_grip_candidates,
)
from seg_web_node.robot_geometry import g1_wrist_ee_position
from seg_web_node.tasc_belief import (
    BeliefInputObject,
    HierarchicalTascBelief,
    StableObjectIdTracker,
)


def box_points(center, size, n=9):
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    axes = [np.linspace(-0.5 * size[i], 0.5 * size[i], n) for i in range(3)]
    surfaces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            grids = np.meshgrid(*axes, indexing="ij")
            grids[axis][...] = sign * 0.5 * size[axis]
            surfaces.append(np.stack(grids, axis=-1).reshape(-1, 3) + center)
    return np.concatenate(surfaces, axis=0)


def test_huber_like_midpoint_convexity():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a, b = sorted(rng.uniform(0.0, 1.0, size=2))
        mid = 0.5 * (a + b)
        f_mid = huber_like(mid, k=1.5, delta=0.1, c=0.1)
        f_avg = 0.5 * (
            huber_like(a, k=1.5, delta=0.1, c=0.1)
            + huber_like(b, k=1.5, delta=0.1, c=0.1)
        )
        assert f_mid <= f_avg + 1.0e-9


def test_distance_advantage_positive_when_action_moves_toward_target():
    feature = MultiFeatureCost(
        FeatureConfig(action_apply_time=1.0, w_ray=0.0, w_agreement=0.0)
    )
    advantage = feature.advantage(
        np.array([1.0, 0.0, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([0.2, 0.0, 0.0]),
    )
    assert advantage > 0.0


def test_ray_feature_prefers_pointed_object():
    feature = MultiFeatureCost(FeatureConfig(w_distance=0.0, w_ray=1.0, w_agreement=0.0))
    origin = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    pointed = feature.advantage(
        np.array([1.0, 0.0, 0.0]),
        eef_position=origin,
        ray_origin=origin,
        ray_direction=direction,
    )
    off_axis = feature.advantage(
        np.array([1.0, 1.0, 0.0]),
        eef_position=origin,
        ray_origin=origin,
        ray_direction=direction,
    )
    assert pointed > off_axis


def test_agreement_direction_override_uses_robot_view_direction():
    feature = MultiFeatureCost(
        FeatureConfig(w_distance=0.0, w_ray=0.0, w_agreement=1.0)
    )
    aligned = feature.advantage(
        np.array([0.5, 0.2, 0.1]),
        eef_position=np.zeros(3),
        user_translation=np.array([1.0, 0.0, 0.0]),
        agreement_direction=np.array([1.0, 0.0, 0.0]),
    )
    off_axis = feature.advantage(
        np.array([0.5, 0.2, 0.1]),
        eef_position=np.zeros(3),
        user_translation=np.array([1.0, 0.0, 0.0]),
        agreement_direction=np.array([0.0, 1.0, 0.0]),
    )
    assert aligned > off_axis


def test_feature_breakdown_exposes_values_and_weighted_contributions():
    feature = MultiFeatureCost(FeatureConfig(w_distance=1.0, w_ray=0.8, w_agreement=0.6))
    breakdown = feature.breakdown(
        np.array([1.0, 0.0, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([0.2, 0.0, 0.0]),
        ray_origin=np.zeros(3),
        ray_direction=np.array([1.0, 0.0, 0.0]),
        agreement_direction=np.array([1.0, 0.0, 0.0]),
    )
    assert breakdown["distance_m"] == 1.0
    assert breakdown["distance_advantage"] > 0.0
    assert breakdown["ray_cost"] == 0.0
    assert breakdown["weighted_contributions"]["distance"] > 0.0
    assert "weighted_total" in breakdown


def test_g1_wrist_fk_matches_pelvis_side_convention():
    q = np.zeros(29, dtype=np.float64)
    left = g1_wrist_ee_position(q, "left_wrist_yaw_link", 0.05)
    right = g1_wrist_ee_position(q, "right_wrist_yaw_link", 0.05)
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()
    assert left[1] > 0.0
    assert right[1] < 0.0
    assert left[0] > 0.15


def test_grip_candidates_are_depth_surface_points_not_only_center():
    center = np.array([0.4, 0.1, 0.2])
    points = box_points(center, np.array([0.12, 0.08, 0.2]))
    candidates = generate_grip_candidates(
        points, GripCandidateConfig(top_k=4, surface_augment=4)
    )
    assert len(candidates) == 4
    assert all(np.min(np.linalg.norm(points - c.center, axis=1)) < 1.0e-9 for c in candidates)
    assert any(np.linalg.norm(c.center - center) > 0.02 for c in candidates)



def test_stable_object_tracker_keeps_label_nearest_id_across_index_changes():
    tracker = StableObjectIdTracker(max_distance_m=0.10, missing_ttl_frames=2)
    first = tracker.assign([
        {"index": 0, "label": "bottle", "center_goal_m": [0.4, 0.1, 0.2]},
        {"index": 1, "label": "box", "center_goal_m": [0.4, -0.1, 0.2]},
    ])
    second = tracker.assign([
        {"index": 1, "label": "bottle", "center_goal_m": [0.405, 0.102, 0.2]},
        {"index": 0, "label": "box", "center_goal_m": [0.398, -0.101, 0.2]},
    ])
    assert second[1] == first[0]
    assert second[0] == first[1]


def test_grip_nearest_matching_preserves_belief_when_candidate_order_changes():
    belief = HierarchicalTascBelief({
        "feature_ema_alpha": 0.25,
        "grip": {"match_max_distance_m": 0.05},
    })
    candidates_a = generate_grip_candidates(
        box_points(np.array([0.4, 0.1, 0.2]), np.array([0.12, 0.08, 0.2])),
        GripCandidateConfig(top_k=4, surface_augment=4),
    )
    belief.log_grip_distributions["bottle:track0000"] = np.log(
        np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
    )
    belief.previous_grip_candidates["bottle:track0000"] = candidates_a
    candidates_b = [candidates_a[2], candidates_a[0], candidates_a[1], candidates_a[3]]
    match = belief._match_previous_grips("bottle:track0000", candidates_b)
    log_grip = belief._align_grip_distribution("bottle:track0000", candidates_b, match)
    probs = np.exp(log_grip)
    assert int(np.argmax(probs)) == 1
    assert probs[1] > 0.65


def test_persistent_grip_bank_keeps_missing_candidate_for_ttl():
    belief = HierarchicalTascBelief({
        "grip": {"match_max_distance_m": 0.05, "track_missing_ttl_frames": 2}
    })
    object_id = "bottle:track0000"
    candidates_a = [
        GripCandidate(np.array([0.30, 0.00, 0.10]), np.array([0.0, 0.0, -1.0]), 0.0, 1.0),
        GripCandidate(np.array([0.36, 0.00, 0.10]), np.array([0.0, 0.0, -1.0]), 0.0, 0.8),
    ]
    persisted_a, ids_a, generated_a = belief._persistent_grip_candidates(
        object_id, candidates_a
    )
    assert generated_a == 2
    assert len(persisted_a) == 2

    candidates_b = [
        GripCandidate(np.array([0.302, 0.001, 0.10]), np.array([0.0, 0.0, -1.0]), 0.0, 1.0),
    ]
    persisted_b, ids_b, generated_b = belief._persistent_grip_candidates(
        object_id, candidates_b
    )
    assert generated_b == 1
    assert ids_a[0] in ids_b
    assert ids_a[1] in ids_b
    assert len(persisted_b) == 2


def test_feature_advantage_ema_uses_alpha_quarter():
    belief = HierarchicalTascBelief({"feature_ema_alpha": 0.25})
    obj = BeliefInputObject("bottle:track0000", "bottle", 1.0, np.array([0.4, 0, 0]))
    first = belief._smooth_object_advantages([obj], np.array([1.0]))
    second = belief._smooth_object_advantages([obj], np.array([-1.0]))
    assert first.tolist() == [1.0]
    np.testing.assert_allclose(second, [0.5])


def test_grip_hysteresis_requires_two_consecutive_frames_to_switch():
    belief = HierarchicalTascBelief({
        "grip": {"switch_dwell_frames": 2, "switch_margin": 0.0}
    })
    object_id = "bottle:track0000"
    belief.selected_grip_indices[object_id] = 0
    probs = np.array([0.4, 0.6], dtype=np.float64)
    first = belief._select_grip_index(object_id, 1, probs, {})
    second = belief._select_grip_index(object_id, 1, probs, {})
    assert first == 0
    assert second == 1

def test_hierarchical_belief_selects_ray_target_and_contact_point():
    config = {
        "features": {"w_distance": 0.2, "w_ray": 1.0, "w_agreement": 0.2},
        "grip": {"top_k": 4, "surface_augment": 4, "grip_update_hz": 100.0},
        "leak": {"kappa0": 0.0, "gamma": 5.0},
    }
    belief = HierarchicalTascBelief(config)
    bottle_center = np.array([0.42, 0.10, 0.18])
    box_center = np.array([0.42, -0.12, 0.18])
    objects = [
        BeliefInputObject(
            "bottle:0",
            "bottle",
            0.95,
            bottle_center,
            box_points(bottle_center, np.array([0.07, 0.07, 0.22])),
        ),
        BeliefInputObject(
            "box:1",
            "box",
            0.94,
            box_center,
            box_points(box_center, np.array([0.16, 0.12, 0.08])),
        ),
    ]
    telemetry = {}
    for tick in range(6):
        telemetry = belief.update(
            objects,
            eef_position_m=np.zeros(3),
            user_translation_mps=np.array([0.4, 0.1, 0.05]),
            ray_origin_m=np.array([0.0, 0.0, 0.1]),
            ray_direction=bottle_center - np.array([0.0, 0.0, 0.1]),
            now_monotonic=float(tick) * 0.02,
        )
    target = telemetry["target"]
    assert target["object_id"] == "bottle:0"
    contact = np.asarray(target["position_goal_m"])
    assert np.linalg.norm(contact - bottle_center) > 0.015
    target_object = next(
        item for item in telemetry["objects"] if item["object_id"] == target["object_id"]
    )
    breakdown = target_object["selected_grip_feature_breakdown"]
    assert breakdown["distance_m"] > 0.0
    assert set(breakdown["weighted_contributions"]) >= {"distance", "ray", "agreement"}
