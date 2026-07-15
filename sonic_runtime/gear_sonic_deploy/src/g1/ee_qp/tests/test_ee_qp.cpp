#include <cassert>
#include <cmath>
#include <memory>

#include "ee_qp/ee_qp_corrector.hpp"

namespace {

class LinearTestKinematics final : public g1::ee_qp::EeKinematics {
 public:
  bool reject_corrected_candidates = false;
  mutable g1::ee_qp::JointVector last_q = g1::ee_qp::JointVector::Zero();

  bool Evaluate(
      const g1::ee_qp::JointVector& q,
      const Eigen::Quaterniond& pelvis_orientation,
      g1::ee_qp::WristKinematics& output) const override {
    last_q = q;
    if (reject_corrected_candidates &&
        (std::abs(q[15] - 0.02) > 1e-9 ||
         std::abs(q[22] + 0.02) > 1e-9)) {
      return false;
    }
    const Eigen::Matrix3d root = pelvis_orientation.toRotationMatrix();
    output.left_position_world =
        root * Eigen::Vector3d(q[12] + q[15], q[13] + q[16], q[14] + q[17]);
    output.right_position_world =
        root * Eigen::Vector3d(q[12] + q[22], q[13] + q[23], q[14] + q[24]);
    output.position_jacobian_world.setZero();
    output.position_jacobian_world.block<3, 3>(0, 0) = root;
    output.position_jacobian_world.block<3, 3>(3, 0) = root;
    output.position_jacobian_world.block<3, 3>(0, 3) = root;
    output.position_jacobian_world.block<3, 3>(3, 10) = root;
    return true;
  }
};

}  // namespace

int main() {
  g1::ee_qp::EeQpConfig config;
  config.max_solve_time_ms = 100.0;
  config.startup_ramp_sec = 0.0;
  auto kinematics = std::make_shared<LinearTestKinematics>();
  g1::ee_qp::EeQpCorrector corrector(config, kinematics);

  g1::ee_qp::EeQpInput input;
  input.reference.valid = true;
  input.reference.timestamp_us = 1000;
  input.now_us = 1000;
  input.reference.left_wrist_in_reference_pelvis = {0.05, 0.0, 0.0};
  input.reference.right_wrist_in_reference_pelvis = {-0.05, 0.0, 0.0};
  input.q_sonic[15] = 0.02;
  input.q_sonic[22] = -0.02;
  input.q_sonic[12] = 0.3;
  input.q_measured[12] = -0.2;
  input.q_measured[15] = 0.8;
  input.q_measured[22] = -0.8;
  input.reference.q_ref[15] = 0.05;
  input.reference.q_ref[22] = -0.05;

  const auto result = corrector.Compute(input);
  assert(result.valid);
  assert(result.error_before > result.predicted_error_after);
  assert(result.error_before > result.nonlinear_error_after);
  assert(std::isfinite(result.predicted_reference_joint_error_after));
  assert(result.q_corrected.head<12>().isApprox(input.q_sonic.head<12>()));
  assert(std::abs(result.delta_q[12]) > 0.0);
  assert(result.backtracking_alpha == 1.0);
  assert(result.reference_age_ms == 0.0);

  input.reference.valid = false;
  const auto fallback = corrector.Compute(input);
  assert(!fallback.valid);
  assert(fallback.fallback_active);
  assert(fallback.fallback_reason ==
         g1::ee_qp::EeQpFallbackReason::kMissingReference);
  assert(fallback.delta_q.norm() > 0.0);
  assert(fallback.delta_q.norm() < result.delta_q.norm());

  config.nonlinear_min_improvement_m = 1.0;
  g1::ee_qp::EeQpCorrector rejecting_corrector(config, kinematics);
  input.reference.valid = true;
  const auto rejected = rejecting_corrector.Compute(input);
  assert(!rejected.valid);
  assert(rejected.fallback_active);
  assert(rejected.candidate_rejected);
  assert(rejected.q_corrected.isApprox(input.q_sonic));

  config.nonlinear_min_improvement_m = 0.0;
  auto stateful_kinematics = std::make_shared<LinearTestKinematics>();
  g1::ee_qp::EeQpCorrector preserving_corrector(config, stateful_kinematics);
  const auto seed = preserving_corrector.Compute(input);
  assert(seed.valid);
  stateful_kinematics->reject_corrected_candidates = true;
  const auto rejected_after_seed = preserving_corrector.Compute(input);
  assert(rejected_after_seed.delta_q.allFinite());
  return 0;
}
