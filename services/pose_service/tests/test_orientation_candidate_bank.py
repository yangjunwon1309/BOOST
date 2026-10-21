import numpy as np
from seg_web_node.grip_candidates import SurfacePrimitive
from seg_web_node.robot_geometry import pelvis_to_world_rotation
from seg_web_node.tasc_belief import (
    AxisEstimator,
    BeliefInputObject,
    HandInput,
    OrientationBankConfig,
    OrientationCandidateBank,
    ParallelTascBelief,
    TascBeliefConfig,
)


def _plane(
    *, origin: tuple[float, float, float] = (0.3, 0.0, 0.2),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    quality: float = 0.9,
) -> SurfacePrimitive:
    return SurfacePrimitive(
        kind="plane",
        origin=np.asarray(origin, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
        fit_quality=quality,
        source="test",
    )


def test_axis_estimator_is_invariant_to_alternating_normal_sign() -> None:
    estimator = AxisEstimator(alpha=0.3)
    for index in range(80):
        estimator.update(
            np.array([0.0, 0.0, 1.0 if index % 2 == 0 else -1.0])
        )
    resolved = estimator.axis_and_concentration()
    assert resolved is not None
    axis, concentration = resolved
    assert abs(float(np.dot(axis, [0.0, 0.0, 1.0]))) > 1.0 - 1.0e-12
    assert concentration > 1.0 - 1.0e-12


def test_axis_estimator_isotropic_observations_converge_to_one_third() -> None:
    estimator = AxisEstimator(alpha=0.01)
    basis = np.eye(3)
    for _ in range(400):
        for axis in basis:
            estimator.update(axis)
    resolved = estimator.axis_and_concentration()
    assert resolved is not None
    _axis, concentration = resolved
    assert abs(concentration - 1.0 / 3.0) < 0.005


def test_axis_estimator_decay_preserves_axis() -> None:
    estimator = AxisEstimator(alpha=0.4)
    expected = np.array([1.0, 2.0, 3.0])
    for _ in range(20):
        estimator.update(expected)
    before = estimator.axis_and_concentration()
    estimator.decay(0.17)
    after = estimator.axis_and_concentration()
    assert before is not None and after is not None
    assert abs(float(np.dot(before[0], after[0]))) > 1.0 - 1.0e-12
    assert abs(before[1] - after[1]) < 1.0e-12


def test_pelvis_world_axis_round_trip_is_identity() -> None:
    angle = np.deg2rad(37.0)
    quaternion_wxyz = np.array(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
    )
    rotation = pelvis_to_world_rotation(quaternion_wxyz)
    assert rotation is not None
    axis_goal = np.array([0.2, -0.5, 0.7])
    np.testing.assert_allclose(
        rotation.T @ (rotation @ axis_goal), axis_goal, atol=1.0e-12
    )


def test_gravity_candidates_are_detection_independent_pair() -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(gravity_candidate_enabled=True)
    )
    no_gravity = bank.candidates(
        relations=("parallel", "perpendicular"),
        base_quat_wxyz=None,
        up_goal=None,
        label_relation_prior={},
        role_keywords={},
        max_candidates=10,
    )
    assert no_gravity == []

    candidates = bank.candidates(
        relations=("parallel", "perpendicular"),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        up_goal=np.array([0.0, 0.0, 1.0]),
        label_relation_prior={
            "gravity": {"parallel": -0.3, "perpendicular": -0.3}
        },
        role_keywords={},
        max_candidates=0,
    )
    assert [candidate.track_id for candidate in candidates] == [
        "__gravity__:parallel",
        "__gravity__:perpendicular",
    ]


def test_reassociation_gate_expands_by_drift_budget_after_two_seconds() -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(
            axis_ema_alpha=1.0,
            assoc_base_distance_m=0.10,
            assoc_drift_mps=0.15,
            gravity_candidate_enabled=False,
        )
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    bank.observe(
        {"bottle:old": _plane(origin=(0.0, 0.0, 0.0))},
        {"bottle:old": "bottle"},
        base_quat_wxyz=identity,
        up_goal=np.array([0.0, 0.0, 1.0]),
        now_us=0,
    )
    bank.advance(0)
    uid = next(iter(bank.tracks))

    bank.observe(
        {"bottle:new": _plane(origin=(0.39, 0.0, 0.0))},
        {"bottle:new": "bottle"},
        base_quat_wxyz=identity,
        up_goal=np.array([0.0, 0.0, 1.0]),
        now_us=2_000_000,
    )
    bank.advance(2_000_000)
    assert list(bank.tracks) == [uid]
    assert bank.tracks[uid].total_observations == 2
    assert bank.reassociations == 1
    assert bank.new_tracks == 1


def test_coasting_axis_reprojects_through_current_pelvis_rotation() -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(
            axis_ema_alpha=1.0,
            gravity_candidate_enabled=False,
        )
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    bank.observe(
        {"surface:0": _plane(axis=(1.0, 0.0, 0.0))},
        {"surface:0": "surface"},
        base_quat_wxyz=identity,
        up_goal=np.array([0.0, 0.0, 1.0]),
        now_us=1_000_000,
    )
    bank.advance(1_000_000)

    yaw = np.deg2rad(90.0)
    rotated = np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    )
    bank.observe(
        {},
        {},
        base_quat_wxyz=rotated,
        up_goal=np.array([0.0, 0.0, 1.0]),
        now_us=1_125_000,
    )
    bank.advance(1_125_000)
    candidates = bank.candidates(
        relations=("parallel",),
        base_quat_wxyz=rotated,
        up_goal=np.array([0.0, 0.0, 1.0]),
        label_relation_prior={},
        role_keywords={},
        max_candidates=10,
    )
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0].normal, [0.0, -1.0, 0.0], atol=1e-12)
    assert candidates[0].source == "bank_coasting"


def test_fresh_track_ranks_ahead_of_equivalent_coasting_track() -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(
            axis_ema_alpha=1.0,
            coast_decay=0.9,
            gravity_candidate_enabled=False,
        )
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    bank.observe(
        {"old:0": _plane()},
        {"old:0": "old"},
        base_quat_wxyz=identity,
        up_goal=None,
        now_us=1_000_000,
    )
    bank.advance(1_000_000)
    bank.observe(
        {"fresh:0": _plane()},
        {"fresh:0": "fresh"},
        base_quat_wxyz=identity,
        up_goal=None,
        now_us=1_125_000,
    )
    bank.advance(1_125_000)
    candidates = bank.candidates(
        relations=("parallel",),
        base_quat_wxyz=identity,
        up_goal=None,
        label_relation_prior={},
        role_keywords={},
        max_candidates=10,
    )
    assert [candidate.label for candidate in candidates[:2]] == ["fresh", "old"]
    assert candidates[0].fit_quality > candidates[1].fit_quality


def test_parallel_belief_keeps_gravity_orientation_without_objects() -> None:
    belief = ParallelTascBelief(
        {
            "orientation": {
                "switch_dwell_frames": 1,
                "bank": {"gravity_candidate_enabled": True},
            }
        }
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    shared = belief.build_shared_geometry(
        [],
        up_goal=np.array([0.0, 0.0, 1.0]),
        base_quat_wxyz=identity,
        stamp_us=1_000_000,
    )
    output = belief.update_hands(
        shared,
        {
            "left": HandInput(
                eef_position=np.zeros(3),
                user_translation=None,
                hand_rotation=np.eye(3),
                hand_axis=None,
                approach_axis=None,
                angular_velocity=np.zeros(3),
                button_pressed=False,
            )
        },
        now_monotonic=1.0,
    )["left"]
    assert output["target"] is None
    assert output["orientation"]["orientation_valid"] is True
    assert output["orientation_only_target"]["object_id"] == "__gravity__"
    assert len(output["orientation"]["candidates"]) == 2


def test_coasting_bank_primitive_is_axis_only_not_contact_geometry() -> None:
    belief = ParallelTascBelief(
        {
            "orientation": {
                "bank": {
                    "gravity_candidate_enabled": False,
                    "contact_requires_active": True,
                }
            }
        }
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    primitive = _plane()
    belief.build_shared_geometry(
        [
            BeliefInputObject(
                "surface:0",
                "surface",
                1.0,
                primitive.origin,
                surface_primitives=[primitive],
            )
        ],
        base_quat_wxyz=identity,
        stamp_us=1_000_000,
    )
    coast = belief.build_shared_geometry(
        [], base_quat_wxyz=identity, stamp_us=1_125_000
    )
    assert coast.surface_primitives == {}
    assert coast.orientation_candidates
    assert all(
        candidate.source == "bank_coasting"
        for candidate in coast.orientation_candidates
    )
    assert coast.orientation_bank_status["coasting_count"] == 1


def test_missing_imu_uses_short_goal_fallback_ttl_and_warns_once(caplog) -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(
            goal_fallback_ttl_frames=3,
            gravity_candidate_enabled=False,
        )
    )
    for frame in range(2):
        bank.observe(
            {"surface:0": _plane()} if frame == 0 else {},
            {"surface:0": "surface"} if frame == 0 else {},
            base_quat_wxyz=None,
            up_goal=None,
            now_us=frame * 100_000,
        )
        bank.advance(frame * 100_000)
    assert bank.status(
        base_quat_wxyz=None, up_goal=None
    )["frame_mode"] == "goal_fallback"
    assert caplog.text.count("short-TTL goal-frame fallback") == 1

    for frame in range(2, 5):
        bank.observe(
            {},
            {},
            base_quat_wxyz=None,
            up_goal=None,
            now_us=frame * 100_000,
        )
        bank.advance(frame * 100_000)
    assert bank.tracks == {}


def test_detector_track_id_is_a_strong_reassociation_hint() -> None:
    bank = OrientationCandidateBank(
        OrientationBankConfig(
            axis_ema_alpha=1.0,
            gravity_candidate_enabled=False,
        )
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    bank.observe(
        {"detector:7": _plane(axis=(0.0, 0.0, 1.0))},
        {"detector:7": "old label"},
        base_quat_wxyz=identity,
        up_goal=None,
        now_us=1_000_000,
    )
    bank.advance(1_000_000)
    uid = next(iter(bank.tracks))
    bank.observe(
        {
            "detector:7": _plane(
                origin=(0.8, 0.0, 0.0), axis=(1.0, 0.0, 0.0)
            )
        },
        {"detector:7": "new label"},
        base_quat_wxyz=identity,
        up_goal=None,
        now_us=1_125_000,
    )
    bank.advance(1_125_000)
    assert list(bank.tracks) == [uid]
    assert bank.tracks[uid].total_observations == 2


def test_bank_track_id_and_orientation_posterior_survive_occlusion() -> None:
    belief = ParallelTascBelief(
        {
            "orientation": {
                "switch_dwell_frames": 1,
                "bank": {"gravity_candidate_enabled": False},
            }
        }
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    primitive = _plane()
    shared_visible = belief.build_shared_geometry(
        [
            BeliefInputObject(
                "surface:7",
                "surface",
                1.0,
                primitive.origin,
                surface_primitives=[primitive],
            )
        ],
        base_quat_wxyz=identity,
        stamp_us=1_000_000,
    )
    hand = HandInput(
        eef_position=np.array([0.3, 0.0, 0.5]),
        user_translation=None,
        hand_rotation=np.eye(3),
        hand_axis=None,
        approach_axis=None,
        angular_velocity=np.zeros(3),
        button_pressed=False,
    )
    first = belief.update_hands(
        shared_visible, {"left": hand}, now_monotonic=1.0
    )["left"]
    first_ids = [
        item["track_id"] for item in first["orientation"]["candidates"]
    ]
    first_selected = first["orientation"]["selected_track_id"]

    shared_hidden = belief.build_shared_geometry(
        [], base_quat_wxyz=identity, stamp_us=1_125_000
    )
    hidden = belief.update_hands(
        shared_hidden, {"left": hand}, now_monotonic=1.125
    )["left"]
    hidden_ids = [
        item["track_id"] for item in hidden["orientation"]["candidates"]
    ]
    assert hidden_ids == first_ids
    assert hidden["orientation"]["selected_track_id"] == first_selected
    assert hidden["orientation_bank"]["coasting_count"] == 1


def test_gravity_prior_configuration_merges_with_existing_label_defaults() -> None:
    config = TascBeliefConfig.from_dict(
        {
            "orientation": {
                "bank": {"gravity_label_prior": 0.2},
                "label_relation_prior": {
                    "gravity": {
                        "parallel": -0.1,
                        "perpendicular": 0.4,
                    }
                },
            }
        }
    )
    assert config.label_relation_prior["surface"]["perpendicular"] == 1.0
    assert config.label_relation_prior["gravity"] == {
        "parallel": -0.1,
        "perpendicular": 0.4,
    }


def test_gravity_pair_is_exempt_from_orientation_candidate_limit() -> None:
    belief = ParallelTascBelief(
        {
            "max_orientation_candidates": 1,
            "orientation": {
                "relations": ["parallel", "perpendicular"],
                "bank": {"gravity_candidate_enabled": True},
            },
        }
    )
    primitive = _plane()
    shared = belief.build_shared_geometry(
        [
            BeliefInputObject(
                "surface:0",
                "surface",
                1.0,
                primitive.origin,
                surface_primitives=[primitive],
            )
        ],
        up_goal=np.array([0.0, 0.0, 1.0]),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        stamp_us=1_000_000,
    )
    gravity = [
        candidate
        for candidate in shared.orientation_candidates
        if candidate.object_id == "__gravity__"
    ]
    non_gravity = [
        candidate
        for candidate in shared.orientation_candidates
        if candidate.object_id != "__gravity__"
    ]
    assert len(gravity) == 2
    assert len(non_gravity) == 1
    output = belief.update_hands(
        shared,
        {
            "left": HandInput(
                eef_position=np.array([0.3, 0.0, 0.5]),
                user_translation=None,
                hand_rotation=np.eye(3),
                hand_axis=None,
                approach_axis=None,
                angular_velocity=np.zeros(3),
                button_pressed=False,
            )
        },
        now_monotonic=1.0,
    )["left"]
    assert len(output["orientation"]["candidates"]) == 3
