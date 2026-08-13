import numpy as np

from seg_web_node.feature_cost import (
    FeatureConfig,
    MultiFeatureCost,
    huber_like,
    robust_standardize,
)
from seg_web_node.grip_candidates import (
    CylinderSurfaceConfig,
    GripCandidate,
    GripCandidateConfig,
    PersistentCylinderSurfaceTracker,
    fit_cylinder_surface,
    generate_grip_candidates,
    _merge_nearby_candidates,
    extract_pseudo_contacts,
    fit_circle_in_plane,
    sample_cylinder_candidates,
    PseudoContactConfig,
    _fit_plane_ransac,
)
from seg_web_node.robot_geometry import g1_wrist_ee_position
from seg_web_node.tasc_belief import (
    BeliefInputObject,
    HierarchicalTascBelief,
    OrientationCandidate,
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


def cylinder_points(
    center=np.array([0.4, 0.1, 0.2]),
    radius=0.04,
    height=0.20,
    angles=48,
    levels=20,
):
    center = np.asarray(center, dtype=np.float64)
    theta = np.linspace(-0.75 * np.pi, 0.75 * np.pi, angles)
    z = np.linspace(-0.5 * height, 0.5 * height, levels)
    theta_grid, z_grid = np.meshgrid(theta, z, indexing="ij")
    return np.stack(
        [
            center[0] + radius * np.cos(theta_grid),
            center[1] + radius * np.sin(theta_grid),
            center[2] + z_grid,
        ],
        axis=-1,
    ).reshape(-1, 3)


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
        FeatureConfig(action_apply_time=1.0, w_progress=0.0, w_agreement=0.0, w_reach=0.0)
    )
    advantage = feature.advantage(
        np.array([1.0, 0.0, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([0.2, 0.0, 0.0]),
    )
    assert advantage > 0.0


def test_ray_feature_is_compatibility_only_and_has_zero_weight():
    feature = MultiFeatureCost(FeatureConfig(w_distance=0.0, w_progress=0.0, w_agreement=0.0, w_reach=0.0))
    origin = np.zeros(3)
    pointed = feature.advantage(
        np.array([1.0, 0.0, 0.0]),
        eef_position=origin,
    )
    off_axis = feature.advantage(
        np.array([1.0, 1.0, 0.0]),
        eef_position=origin,
    )
    assert pointed == off_axis == 0.0


def test_agreement_uses_hand_to_target_direction():
    feature = MultiFeatureCost(
        FeatureConfig(w_distance=0.0, w_progress=0.0, w_agreement=1.0, w_reach=0.0)
    )
    aligned = feature.advantage(
        np.array([0.5, 0.0, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([1.0, 0.0, 0.0]),
    )
    off_axis = feature.advantage(
        np.array([0.0, 0.5, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([1.0, 0.0, 0.0]),
    )
    assert aligned > off_axis


def test_robust_standardize_ignores_tiny_signal_spread():
    update = robust_standardize(
        np.array([0.0001, 0.0002, 0.0003]),
        min_signal_spread=0.002,
        scale_floor=0.01,
        clip=2.0,
    )
    np.testing.assert_allclose(update, np.zeros(3))


def test_robust_standardize_uses_floor_and_clip_for_real_signal():
    update = robust_standardize(
        np.array([-0.10, 0.0, 0.10]),
        min_signal_spread=0.002,
        scale_floor=0.20,
        clip=0.4,
    )
    np.testing.assert_allclose(update, [-0.4, 0.0, 0.4])


def test_feature_breakdown_exposes_values_and_weighted_contributions():
    feature = MultiFeatureCost(FeatureConfig(w_distance=1.0, w_progress=1.0, w_agreement=0.6, w_reach=0.0))
    breakdown = feature.breakdown(
        np.array([1.0, 0.0, 0.0]),
        eef_position=np.zeros(3),
        user_translation=np.array([0.2, 0.0, 0.0]),
    )
    assert breakdown["distance_m"] == 1.0
    assert breakdown["distance_advantage"] > 0.0
    assert breakdown["ray_cost"] == 0.0
    assert breakdown["weights"]["ray"] == 0.0
    assert "progress_advantage" in breakdown
    assert breakdown["weighted_contributions"]["distance"] > 0.0
    assert "weighted_total" in breakdown


def test_reach_feature_penalizes_target_outside_shoulder_annulus():
    feature = MultiFeatureCost(
        FeatureConfig(w_distance=0.0, w_progress=0.0, w_agreement=0.0, w_reach=1.0)
    )
    reachable = feature.breakdown(
        np.array([0.40, 0.0, 0.0]), eef_position=np.zeros(3), shoulder_position=np.zeros(3)
    )
    unreachable = feature.breakdown(
        np.array([0.90, 0.0, 0.0]), eef_position=np.zeros(3), shoulder_position=np.zeros(3)
    )
    assert reachable["reach_advantage"] == 0.0
    assert unreachable["reach_advantage"] < 0.0


def test_g1_wrist_fk_matches_pelvis_side_convention():
    q = np.zeros(29, dtype=np.float64)
    left = g1_wrist_ee_position(q, "left_wrist_yaw_link", 0.05)
    right = g1_wrist_ee_position(q, "right_wrist_yaw_link", 0.05)
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()
    assert left[1] > 0.0
    assert right[1] < 0.0
    assert left[0] > 0.15


def test_grip_candidates_merge_nearby_contacts_before_topk():
    candidates = [
        GripCandidate(np.array([0.30, 0.00, 0.10]), np.array([0.0, 0.0, -1.0]), 0.02, 1.0, "a"),
        GripCandidate(np.array([0.32, 0.00, 0.10]), np.array([0.0, 0.0, -1.0]), 0.04, 0.5, "b"),
        GripCandidate(np.array([0.40, 0.00, 0.10]), np.array([0.0, 0.0, -1.0]), 0.01, 0.7, "c"),
    ]
    merged = _merge_nearby_candidates(candidates, 0.03)
    assert len(merged) == 2
    np.testing.assert_allclose(
        merged[0].center,
        (np.array([0.30, 0.00, 0.10]) * 1.0 + np.array([0.32, 0.00, 0.10]) * 0.5) / 1.5,
    )
    assert merged[0].geom_score == 1.0
    assert merged[0].source == "merged(a+b)"


def test_grip_candidates_are_depth_surface_points_not_only_center():
    center = np.array([0.4, 0.1, 0.2])
    points = box_points(center, np.array([0.12, 0.08, 0.2]))
    candidates = generate_grip_candidates(
        points, GripCandidateConfig(top_k=4, surface_augment=4)
    )
    assert len(candidates) == 4
    assert all(np.min(np.linalg.norm(points - c.center, axis=1)) < 1.0e-9 for c in candidates)
    assert any(np.linalg.norm(c.center - center) > 0.02 for c in candidates)


def test_robust_cylinder_fit_recovers_partial_surface_with_outliers():
    rng = np.random.default_rng(12)
    points = cylinder_points()
    points += rng.normal(scale=0.0015, size=points.shape)
    outliers = rng.uniform([0.30, 0.00, 0.08], [0.52, 0.20, 0.32], size=(80, 3))
    model = fit_cylinder_surface(
        np.concatenate([points, outliers]),
        CylinderSurfaceConfig(min_points=100, min_angular_coverage=0.15),
    )
    assert model is not None
    assert abs(model.radius - 0.04) < 0.008
    assert abs(model.height - 0.20) < 0.025
    assert abs(float(np.dot(model.axis, np.array([0.0, 0.0, 1.0])))) > 0.98
    assert model.fit_rmse < 0.006


def test_cylinder_tracker_retains_surface_during_short_occlusion():
    config = GripCandidateConfig.from_dict(
        {
            "top_k": 4,
            "surface_model": {
                "min_points": 100,
                "min_angular_coverage": 0.15,
                "stale_ttl_frames": 2,
                "max_candidates": 4,
            },
        }
    )
    tracker = PersistentCylinderSurfaceTracker(config)
    first = tracker.update("bottle:track0000", "bottle", cylinder_points())
    assert len(first) == 4
    assert all(candidate.source == "cylinder_surface" for candidate in first)
    first_centers = np.stack([candidate.center for candidate in first])

    held = tracker.update(
        "bottle:track0000", "bottle", np.empty((0, 3), dtype=np.float64)
    )
    assert len(held) == 4
    np.testing.assert_allclose(
        np.stack([candidate.center for candidate in held]), first_centers
    )
    assert tracker.status("bottle:track0000")["stale_frames"] == 1


def test_cylinder_candidates_preserve_orthonormal_surface_frame():
    model = fit_cylinder_surface(
        cylinder_points(), CylinderSurfaceConfig(min_points=100, min_angular_coverage=0.15)
    )
    assert model is not None
    candidate = sample_cylinder_candidates(model, GripCandidateConfig())[0]
    assert candidate.surface_normal is not None
    assert candidate.surface_tangent is not None
    assert candidate.surface_axis is not None
    frame = [candidate.surface_normal, candidate.surface_tangent, candidate.surface_axis]
    np.testing.assert_allclose([np.linalg.norm(vector) for vector in frame], 1.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(frame[0], frame[1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(frame[0], frame[2]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(frame[1], frame[2]), 0.0, atol=1e-6)


def test_pseudo_contact_plane_and_circle_fit_refines_hole_center():
    center = np.array([0.42, -0.08, 0.20])
    radius = 0.03
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    ring = center + np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))
    fitted = fit_circle_in_plane(ring, center, np.array([0.0, 0.0, 1.0]))
    assert fitted is not None
    fit_center, fit_radius, rmse = fitted
    np.testing.assert_allclose(fit_center, center, atol=1e-6)
    assert abs(fit_radius - radius) < 1e-6 and rmse < 1e-6
    pseudo = extract_pseudo_contacts(
        ring, PseudoContactConfig(min_inlier_ratio=0.2), label="white round circled hole"
    )
    assert pseudo and pseudo[0].source == "hole_circle"


def test_orientation_advantage_prefers_aligned_stationary_hand():
    feature = MultiFeatureCost(FeatureConfig())
    aligned = feature.orientation_advantage(
        surface_normal=np.array([0.0, 0.0, 1.0]), hand_axis=np.array([0.0, 0.0, 1.0]),
        hand_angular_velocity=np.zeros(3), relation="parallel",
    )
    orthogonal = feature.orientation_advantage(
        surface_normal=np.array([0.0, 0.0, 1.0]), hand_axis=np.array([1.0, 0.0, 0.0]),
        hand_angular_velocity=np.zeros(3), relation="parallel",
    )
    assert aligned > orthogonal


def test_tasc_uses_fitted_cylinder_surface_as_contact_domain():
    belief = HierarchicalTascBelief(
        {
            "grip": {
                "top_k": 4,
                "surface_model": {
                    "min_points": 100,
                    "min_angular_coverage": 0.15,
                    "max_candidates": 6,
                },
            }
        }
    )
    telemetry = belief.update(
        [
            BeliefInputObject(
                object_id="bottle:track0000",
                label="bottle",
                score=0.95,
                center_goal_m=np.array([0.4, 0.1, 0.2]),
                points_goal_m=cylinder_points(),
            )
        ],
        eef_position_m=np.array([0.2, 0.1, 0.2]),
    )
    assert telemetry["target"] is not None
    assert telemetry["objects"][0]["surface_model"]["type"] == "cylinder"
    candidates = telemetry["objects"][0]["grip_candidates"]
    assert len(candidates) == 6
    assert all(candidate["source"] == "cylinder_surface" for candidate in candidates)



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


def test_persist_leak_is_elapsed_time_not_callback_rate():
    belief = HierarchicalTascBelief(
        {"jump": {"enabled": False, "persist_leak": 0.02}}
    )
    initial = np.log(np.array([0.8, 0.2], dtype=np.float64))
    advantage = np.zeros(2, dtype=np.float64)
    one_step, _, _ = belief._jump_predict(initial, advantage, None, 1.0)

    many_steps = initial.copy()
    for _ in range(10):
        many_steps, _, _ = belief._jump_predict(
            many_steps, advantage, None, 0.1
        )
    np.testing.assert_allclose(
        np.exp(one_step), np.exp(many_steps), rtol=1.0e-10, atol=1.0e-12
    )
    one_tick, _, _ = belief._jump_predict(initial, advantage, None, 1.0 / 30.0)
    np.testing.assert_allclose(np.exp(one_tick), [0.794, 0.206], atol=1.0e-12)


def test_pseudo_contacts_never_return_more_than_one_surface_target():
    sparse = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    plane = np.array([[x, y, 0.0] for x in np.linspace(-.1, .1, 8) for y in np.linspace(-.1, .1, 8)])
    ring = np.array([[.03*np.cos(t), .03*np.sin(t), 0.0] for t in np.linspace(0, 2*np.pi, 32, endpoint=False)])
    for cloud, label in ((sparse, "point"), (plane, "table"), (ring, "circle hole")):
        assert len(extract_pseudo_contacts(cloud, PseudoContactConfig(min_inlier_ratio=.2), label=label)) <= 1


def test_orientation_candidate_bank_carries_posterior_across_domain_change():
    belief = HierarchicalTascBelief({"orientation": {"switch_dwell_frames": 1}})
    first = [
        OrientationCandidate("obj", 0, relation, np.array([.3, .1, .2]), np.array([0, 0, 1.]), 1.0, track_id=f"obj:{relation}")
        for relation in ("parallel", "perpendicular")
    ]
    belief.orientation_candidates = first
    belief.log_orientation_distribution = np.log(np.array([.9, .1]))
    belief.orientation_advantage_ema = np.array([.4, -.2])
    expanded = first + [
        OrientationCandidate(f"other{i}", 0, "parallel", np.array([.4+i*.01, 0, .2]), np.array([0,0,1.]), 1.0, track_id=f"other{i}:parallel")
        for i in range(10)
    ]
    match = belief._match_orientation_candidates(expanded)
    assert match[0] == 0 and match[1] == 1
    assert belief.log_orientation_distribution[match[0]] > belief.log_orientation_distribution[match[1]]


def test_orientation_disabled_retains_position_belief_compatibility():
    belief = HierarchicalTascBelief({"orientation": {"enabled": False}, "jump": {"enabled": False}})
    telemetry = belief.update(
        [BeliefInputObject("obj", "bottle", 1.0, np.array([.3, 0., .2]))],
        eef_position_m=np.zeros(3), now_monotonic=1.0,
    )
    assert telemetry["target"] is not None
    assert telemetry["orientation"]["enabled"] is False


def test_perpendicular_orientation_singularity_returns_no_quaternion():
    belief = HierarchicalTascBelief({"orientation": {"hand_axis_local": [1, 0, 0]}})
    candidate = OrientationCandidate(
        "surface", 0, "perpendicular", np.array([0.3, 0.0, 0.2]),
        np.array([1.0, 0.0, 0.0]), 1.0,
    )
    assert belief._quaternion_from_relation(candidate, np.eye(3)) is None


def test_standalone_orientation_ghost_stays_in_lockstep():
    belief = HierarchicalTascBelief({"orientation": {"switch_dwell_frames": 1}})
    belief.orientation_candidates = [
        OrientationCandidate(
            "old", 0, "parallel", np.array([0.3, 0.0, 0.2]),
            np.array([0.0, 0.0, 1.0]), 1.0, track_id="old:parallel",
        )
    ]
    belief.log_orientation_distribution = np.log(np.array([1.0]))
    belief.orientation_advantage_ema = np.array([0.0])
    target, telemetry, _, _ = belief._update_orientation_belief(
        [BeliefInputObject("new", "surface", 1.0, np.array([0.4, 0.0, 0.2]))],
        hand_rotation_matrix=np.eye(3),
        hand_angular_velocity=np.zeros(3),
        dt=1.0 / 30.0,
    )
    assert target is not None
    assert len(telemetry["candidates"]) == belief.orientation_advantage_ema.size


def test_stationary_jump_kernels_are_uniform():
    belief = HierarchicalTascBelief()
    objects = [
        BeliefInputObject("a", "a", 1.0, np.array([0.2, 0.0, 0.1])),
        BeliefInputObject("b", "b", 1.0, np.array([0.4, 0.0, 0.1])),
    ]
    position_kernel = belief._switch_weights_position(objects, np.zeros(3), None)
    np.testing.assert_allclose(np.exp(position_kernel), [0.5, 0.5])
    orientation_kernel = belief._orientation_switch_weights(
        [
            OrientationCandidate("a", 0, "parallel", np.zeros(3), np.array([0, 0, 1.0]), 1.0),
            OrientationCandidate("b", 0, "perpendicular", np.zeros(3), np.array([1.0, 0, 0]), 1.0),
        ],
        np.array([0.0, 0.0, 1.0]),
        None,
    )
    np.testing.assert_allclose(np.exp(orientation_kernel), [0.5, 0.5])


def test_new_orientation_candidate_enters_at_existing_posterior_floor():
    belief = HierarchicalTascBelief({
        "jump": {"enabled": False, "persist_leak": 0.0},
        "orientation": {"beta": 0.0, "w_sc": 1.0, "switch_dwell_frames": 1},
    })
    old = [
        OrientationCandidate(f"old{i}", 0, "parallel", np.array([.3 + .01 * i, 0, .2]), np.array([0, 0, 1.]), 1.0, track_id=f"old{i}:parallel")
        for i in range(3)
    ]
    belief.orientation_candidates = old
    belief.log_orientation_distribution = np.log(np.array([.90, .08, .02]))
    belief.orientation_advantage_ema = np.zeros(3)
    candidates = old + [
        OrientationCandidate("new", 0, "parallel", np.array([.5, 0, .2]), np.array([0, 0, 1.]), 1.0, track_id="new:parallel")
    ]
    _target, telemetry, _, _ = belief._update_orientation_belief(
        [], hand_rotation_matrix=np.eye(3), hand_angular_velocity=np.zeros(3),
        dt=1.0 / 30.0, orientation_candidates=candidates,
    )
    probabilities = np.asarray([item["probability"] for item in telemetry["candidates"]])
    new_index = next(i for i, item in enumerate(telemetry["candidates"]) if item["track_id"] == "new:parallel")
    assert probabilities[new_index] <= probabilities[2] + 1.0e-12


def test_plane_fit_quality_improves_with_lower_residual_noise():
    rng = np.random.default_rng(31)
    base = np.array([[x, y, 0.0] for x in np.linspace(-.1, .1, 16) for y in np.linspace(-.1, .1, 16)])
    config = PseudoContactConfig(min_inlier_ratio=.2, plane_inlier_thresh_m=.012)
    clean = _fit_plane_ransac(base + rng.normal(scale=0.0003, size=base.shape), config)
    noisy = _fit_plane_ransac(base + rng.normal(scale=0.0040, size=base.shape), config)
    assert clean is not None and noisy is not None
    assert clean[3] > noisy[3]


def test_grip_candidates_use_supplied_gravity_axis():
    points = box_points(np.array([0.4, 0.0, 0.2]), np.array([0.12, 0.08, 0.04]))
    up = np.array([0.0, 0.5, np.sqrt(0.75)])
    candidates = generate_grip_candidates(
        points, GripCandidateConfig(top_k=1, surface_augment=0), up_goal=up
    )
    assert candidates
    np.testing.assert_allclose(candidates[0].approach, -up, atol=1.0e-6)
