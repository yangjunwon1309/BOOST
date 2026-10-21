import numpy as np

from seg_web_node.feature_cost import FeatureConfig, MultiFeatureCost
from seg_web_node.grip_candidates import SurfacePrimitive, contact_frame
from seg_web_node.tasc_belief import (
    BeliefInputObject,
    HandInput,
    ParallelTascBelief,
    grasp_frame_from_cylinder,
)


def _cylinder() -> SurfacePrimitive:
    return SurfacePrimitive(
        kind="cylinder",
        origin=np.array([0.40, 0.0, 0.20]),
        axis=np.array([0.0, 0.0, 1.0]),
        radius=0.04,
        half_height=0.10,
        fit_quality=0.9,
        source="test",
    )


def test_cylinder_contact_normal_is_hand_relative_around_full_surface():
    primitive = _cylinder()
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        frame = contact_frame(primitive, primitive.origin + 0.20 * radial)
        np.testing.assert_allclose(frame.normal, radial, atol=1.0e-9)
        np.testing.assert_allclose(
            frame.point, primitive.origin + primitive.radius * radial, atol=1.0e-9
        )


def test_cylinder_axis_singularity_preserves_previous_normal():
    primitive = _cylinder()
    previous = np.array([0.0, -1.0, 0.0])
    frame = contact_frame(primitive, primitive.origin, previous)
    np.testing.assert_allclose(frame.normal, previous, atol=1.0e-9)


def test_plane_contact_normal_selects_the_hand_side_without_camera_data():
    primitive = SurfacePrimitive(
        kind="plane", origin=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]),
        fit_quality=0.8, source="plane",
    )
    np.testing.assert_allclose(contact_frame(primitive, np.array([0, 0, 1.0])).normal, [0, 0, 1])
    np.testing.assert_allclose(contact_frame(primitive, np.array([0, 0, -1.0])).normal, [0, 0, -1])


def test_cylinder_grasp_frame_is_a_right_handed_rotation_without_sign_jump():
    reference = np.eye(3)
    first = grasp_frame_from_cylinder([0, 0, 1], [1, 0, 0], reference)
    second = grasp_frame_from_cylinder([0, 0, 1], [0.999, 0.045, 0], first)
    assert first is not None and second is not None
    np.testing.assert_allclose(first.T @ first, np.eye(3), atol=1.0e-9)
    assert np.linalg.det(first) > 0.999999
    assert abs(float(np.dot(first[:, 2], np.array([0.0, 0.0, 1.0])))) > 0.999999
    assert float(np.trace(first.T @ second)) > 2.0
    np.testing.assert_allclose(first[:, 1], [-1, 0, 0], atol=1.0e-9)


def test_alignment_advantage_prefers_head_on_approach_and_respects_quality_gate():
    feature = MultiFeatureCost(FeatureConfig(w_align=0.9, align_min_quality=0.25))
    head_on = feature.alignment_advantage(
        approach_axis=np.array([1.0, 0.0, 0.0]),
        contact_normal=np.array([-1.0, 0.0, 0.0]),
        fit_quality=0.8,
    )
    away = feature.alignment_advantage(
        approach_axis=np.array([1.0, 0.0, 0.0]),
        contact_normal=np.array([1.0, 0.0, 0.0]),
        fit_quality=0.8,
    )
    assert head_on == 1.0 and away == -1.0
    assert feature.alignment_advantage(
        approach_axis=np.array([1.0, 0.0, 0.0]),
        contact_normal=np.array([-1.0, 0.0, 0.0]),
        fit_quality=0.1,
    ) == 0.0


def test_gravity_axis_regularization_does_not_flip_cylinder_contact_normal():
    primitive = _cylinder()
    frame = contact_frame(primitive, np.array([0.40, 0.20, 0.20]))
    np.testing.assert_allclose(frame.normal, [0.0, 1.0, 0.0], atol=1.0e-9)


def test_parallel_hands_share_the_primitive_but_get_independent_contact_normals():
    primitive = _cylinder()
    belief = ParallelTascBelief({"orientation": {"switch_dwell_frames": 1}})
    shared = belief.build_shared_geometry([
        BeliefInputObject("bottle:0", "bottle", 1.0, primitive.origin, surface_primitives=[primitive])
    ])
    outputs = belief.update_hands(shared, {
        "left": HandInput(np.array([0.60, 0.0, 0.20]), None, np.eye(3), None, np.array([0, 1, 0]), None, False),
        "right": HandInput(np.array([0.20, 0.0, 0.20]), None, np.eye(3), None, np.array([0, 1, 0]), None, False),
    }, now_monotonic=1.0)
    left = outputs["left"]["orientation"]["candidates"][0]["normal"]
    right = outputs["right"]["orientation"]["candidates"][0]["normal"]
    np.testing.assert_allclose(left, [1, 0, 0], atol=1.0e-9)
    np.testing.assert_allclose(right, [-1, 0, 0], atol=1.0e-9)
