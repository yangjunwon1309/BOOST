#include "ee_qp/ee_qp_corrector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <proxsuite/proxqp/dense/dense.hpp>
#include <yaml-cpp/yaml.h>

namespace g1::ee_qp {

namespace {

bool AllFinite(const EeQpInput& input) {
  const bool terminal_left_ok = !input.reference.has_terminal_left_wrist_target ||
      input.reference.terminal_left_wrist_in_reference_pelvis.allFinite();
  const bool terminal_right_ok = !input.reference.has_terminal_right_wrist_target ||
      input.reference.terminal_right_wrist_in_reference_pelvis.allFinite();
  return input.q_measured.allFinite() && input.dq_measured.allFinite() &&
         input.q_sonic.allFinite() &&
         input.measured_pelvis_orientation.coeffs().allFinite() &&
         input.reference.left_wrist_in_reference_pelvis.allFinite() &&
         input.reference.right_wrist_in_reference_pelvis.allFinite() &&
         input.reference.reference_pelvis_orientation.coeffs().allFinite() &&
         input.reference.q_ref.allFinite() &&
         terminal_left_ok && terminal_right_ok &&
         std::isfinite(input.reference.terminal_capture_radius_m) &&
         std::isfinite(input.dt) && input.dt > 0.0;
}

ActiveVector ExtractActive(const JointVector& q) {
  ActiveVector active;
  for (int i = 0; i < kNumUpperJoints; ++i) {
    active[i] = q[kUpperHardwareIndices[i]];
  }
  return active;
}

void InsertActive(const ActiveVector& active, JointVector& q) {
  for (int i = 0; i < kNumUpperJoints; ++i) {
    q[kUpperHardwareIndices[i]] = active[i];
  }
}

double SaturationFraction(
    const ActiveVector& value, const ActiveVector& lower, const ActiveVector& upper) {
  constexpr double kSaturationToleranceRad = 1e-4;
  int saturated = 0;
  for (int i = 0; i < kNumUpperJoints; ++i) {
    if (std::abs(value[i] - lower[i]) <= kSaturationToleranceRad ||
        std::abs(value[i] - upper[i]) <= kSaturationToleranceRad) {
      ++saturated;
    }
  }
  return static_cast<double>(saturated) / kNumUpperJoints;
}

ActiveVector SaturationMask(
    const ActiveVector& value, const ActiveVector& lower, const ActiveVector& upper) {
  constexpr double kSaturationToleranceRad = 1e-4;
  ActiveVector mask = ActiveVector::Zero();
  for (int i = 0; i < kNumUpperJoints; ++i) {
    if (std::abs(value[i] - lower[i]) <= kSaturationToleranceRad ||
        std::abs(value[i] - upper[i]) <= kSaturationToleranceRad) {
      mask[i] = 1.0;
    }
  }
  return mask;
}

bool LeftOnlyCartesianTarget(const EeReference& reference) {
  return reference.reference_type == ReferenceType::kCartesianTarget &&
         reference.has_terminal_left_wrist_target &&
         !reference.has_terminal_right_wrist_target;
}

bool RightOnlyCartesianTarget(const EeReference& reference) {
  return reference.reference_type == ReferenceType::kCartesianTarget &&
         reference.has_terminal_right_wrist_target &&
         !reference.has_terminal_left_wrist_target;
}

void LockInactiveArmBounds(
    const EeReference& reference, ActiveVector& lower, ActiveVector& upper) {
  if (LeftOnlyCartesianTarget(reference)) {
    for (int i = 10; i < 17; ++i) {
      lower[i] = 0.0;
      upper[i] = 0.0;
    }
  } else if (RightOnlyCartesianTarget(reference)) {
    for (int i = 3; i < 10; ++i) {
      lower[i] = 0.0;
      upper[i] = 0.0;
    }
  }
}

void LockCartesianWaistBounds(
    const EeReference& reference, ActiveVector& lower, ActiveVector& upper) {
  if (reference.reference_type != ReferenceType::kCartesianTarget) return;
  for (int i = 0; i < 3; ++i) {
    lower[i] = 0.0;
    upper[i] = 0.0;
  }
}

void ZeroInactiveArmDelta(const EeReference& reference, ActiveVector& delta) {
  if (LeftOnlyCartesianTarget(reference)) {
    for (int i = 10; i < 17; ++i) delta[i] = 0.0;
  } else if (RightOnlyCartesianTarget(reference)) {
    for (int i = 3; i < 10; ++i) delta[i] = 0.0;
  }
}

void ZeroCartesianWaistDelta(const EeReference& reference, ActiveVector& delta) {
  if (reference.reference_type != ReferenceType::kCartesianTarget) return;
  for (int i = 0; i < 3; ++i) delta[i] = 0.0;
}

void ZeroLockedDelta(const EeReference& reference, ActiveVector& delta) {
  ZeroCartesianWaistDelta(reference, delta);
  ZeroInactiveArmDelta(reference, delta);
}

void CopyInactiveArmFromReference(const EeReference& reference, JointVector& q) {
  if (LeftOnlyCartesianTarget(reference)) {
    for (int hw = 22; hw <= 28; ++hw) q[hw] = reference.q_ref[hw];
  } else if (RightOnlyCartesianTarget(reference)) {
    for (int hw = 15; hw <= 21; ++hw) q[hw] = reference.q_ref[hw];
  }
}

template <typename Fn>
void ForEachTargetArmIndex(const EeReference& reference, Fn&& fn) {
  if (LeftOnlyCartesianTarget(reference)) {
    for (int i = 3; i < 10; ++i) fn(i);
  } else if (RightOnlyCartesianTarget(reference)) {
    for (int i = 10; i < 17; ++i) fn(i);
  } else if (reference.reference_type == ReferenceType::kCartesianTarget) {
    if (reference.has_terminal_left_wrist_target) {
      for (int i = 3; i < 10; ++i) fn(i);
    }
    if (reference.has_terminal_right_wrist_target) {
      for (int i = 10; i < 17; ++i) fn(i);
    }
  }
}

struct CartesianServoSchedule {
  double position_scale = 1.0;
  double active_arm_joint_scale = 1.0;
  double active_arm_velocity_rad_s = 0.0;
  bool active = false;
};

TaskVector MakeTaskVector(const WristKinematics& wrist) {
  TaskVector position;
  position.head<3>() = wrist.left_position_world;
  position.tail<3>() = wrist.right_position_world;
  return position;
}

ActiveVector ReadActiveVector(
    const YAML::Node& node, const std::string& key, const ActiveVector& fallback) {
  const YAML::Node value = node[key];
  if (!value) return fallback;
  ActiveVector output;
  if (value.IsSequence()) {
    if (value.size() != kNumUpperJoints) {
      throw std::invalid_argument(
          key + " must contain exactly " + std::to_string(kNumUpperJoints) +
          " values");
    }
    for (int i = 0; i < kNumUpperJoints; ++i) {
      output[i] = value[i].as<double>();
    }
  } else {
    output.setConstant(value.as<double>());
  }
  return output;
}

}  // namespace

class EeQpCorrector::Impl {
 public:
  Impl(EeQpConfig config, std::shared_ptr<const EeKinematics> kinematics)
      : config_(std::move(config)),
        kinematics_(std::move(kinematics)),
        qp_(kNumUpperJoints, 0, kNumUpperJoints) {
    if (!kinematics_) throw std::invalid_argument("EeKinematics is required");
    if (config_.position_tolerance_m <= 0.0 ||
        (config_.joint_tolerance_rad.array() <= 0.0).any() ||
        config_.invalid_decay_time_sec < 0.0 ||
        (config_.position_weight.array() < 0.0).any() ||
        (config_.joint_tracking_weight.array() < 0.0).any() ||
        config_.inactive_arm_hold_weight < 0.0 ||
        config_.inactive_arm_hold_tolerance_rad <= 0.0 ||
        (config_.max_delta_rad.array() <= 0.0).any() ||
        (config_.max_velocity_rad_s.array() <= 0.0).any() ||
        config_.previous_delta_weight < 0.0 ||
        config_.cartesian_relinearization_iterations < 1 ||
        config_.backtracking_initial_alpha <= 0.0 ||
        config_.backtracking_initial_alpha > 1.0 ||
        config_.backtracking_factor <= 0.0 ||
        config_.backtracking_factor >= 1.0 ||
        config_.backtracking_min_alpha <= 0.0 ||
        config_.backtracking_min_alpha > config_.backtracking_initial_alpha ||
        config_.cartesian_far_distance_m <= config_.cartesian_mid_distance_m ||
        config_.cartesian_mid_distance_m <= config_.cartesian_near_distance_m ||
        config_.cartesian_near_distance_m < 0.0 ||
        config_.cartesian_position_weight_scale_far <= 0.0 ||
        config_.cartesian_position_weight_scale_mid <= 0.0 ||
        config_.cartesian_position_weight_scale_near <= 0.0 ||
        config_.cartesian_position_weight_scale_close <= 0.0 ||
        config_.cartesian_active_arm_joint_weight_scale_far < 0.0 ||
        config_.cartesian_active_arm_joint_weight_scale_mid < 0.0 ||
        config_.cartesian_active_arm_joint_weight_scale_near < 0.0 ||
        config_.cartesian_active_arm_joint_weight_scale_close < 0.0 ||
        config_.cartesian_active_arm_max_velocity_far_rad_s <= 0.0 ||
        config_.cartesian_active_arm_max_velocity_mid_rad_s <= 0.0 ||
        config_.cartesian_active_arm_max_velocity_near_rad_s <= 0.0 ||
        config_.cartesian_active_arm_max_velocity_close_rad_s <= 0.0) {
      throw std::invalid_argument("EE QP weights and tolerances must be valid");
    }
    qp_.settings.eps_abs = 1e-7;
    qp_.settings.eps_rel = 1e-7;
  }

  EeQpResult Compute(const EeQpInput& input) {
    if (!input.reference.valid) {
      return Fallback(input, EeQpFallbackReason::kMissingReference);
    }
    if (!input.reference.feasible) {
      return Fallback(input, EeQpFallbackReason::kInfeasibleReference);
    }
    if (!AllFinite(input)) {
      return Fallback(input, EeQpFallbackReason::kNonFiniteInput);
    }
    if (input.now_us < input.reference.timestamp_us) {
      return Fallback(input, EeQpFallbackReason::kFutureReference);
    }
    if (input.now_us - input.reference.timestamp_us >
        config_.reference_timeout_us) {
      return Fallback(input, EeQpFallbackReason::kStaleReference);
    }

    // Corrections are commanded relative to q_sonic, but FK/Jacobian are
    // linearized at a one-step measured-state prediction. This keeps the QP
    // objective close to the state where the next command will actually land.
    constexpr double kPredictionHorizonSec = 0.01;
    constexpr double kMaxPredictionStepRad = 0.04;
    JointVector q_linearization = input.q_measured;
    for (int hw = 0; hw < kNumJoints; ++hw) {
      const double step = std::clamp(
          input.dq_measured[hw] * kPredictionHorizonSec,
          -kMaxPredictionStepRad, kMaxPredictionStepRad);
      q_linearization[hw] += step;
    }
    WristKinematics wrist;
    if (!kinematics_->Evaluate(
            q_linearization, input.measured_pelvis_orientation.normalized(), wrist) ||
        !wrist.left_position_world.allFinite() ||
        !wrist.right_position_world.allFinite() ||
        !wrist.position_jacobian_world.allFinite()) {
      return Fallback(input, EeQpFallbackReason::kKinematicsFailure);
    }

    EeQpResult result;
    result.q_corrected = input.q_sonic;
    result.reference_age_ms =
        static_cast<double>(input.now_us - input.reference.timestamp_us) / 1000.0;
    result.reference_source = input.reference.source;
    result.control_mode = input.reference.control_mode;
    result.plan_id = input.reference.plan_id;
    result.plan_phase = input.reference.phase;
    result.reference_type = input.reference.reference_type;
    result.reference_feasible = input.reference.feasible;
    const Eigen::Quaterniond reference_root =
        input.reference.reference_pelvis_orientation.normalized();
    const bool cartesian_target_mode =
        input.reference.reference_type == ReferenceType::kCartesianTarget;
    TaskVector target;
    TaskVector current = MakeTaskVector(wrist);
    result.left_wrist_position = current.head<3>();
    result.right_wrist_position = current.tail<3>();
    result.active_wrist_position = current.tail<3>();
    if (cartesian_target_mode) {
      target = current;
    } else {
      target.head<3>() = reference_root *
          input.reference.left_wrist_in_reference_pelvis;
      target.tail<3>() = reference_root *
          input.reference.right_wrist_in_reference_pelvis;
    }
    result.terminal_capture_radius_m =
        std::max(0.0, input.reference.terminal_capture_radius_m);
    if (input.reference.has_terminal_left_wrist_target) {
      const Eigen::Vector3d terminal =
          reference_root * input.reference.terminal_left_wrist_in_reference_pelvis;
      const double distance = (terminal - current.head<3>()).norm();
      result.terminal_target_distance_m = distance;
      result.terminal_target_position = terminal;
      if (cartesian_target_mode || distance <= result.terminal_capture_radius_m) {
        target.head<3>() = terminal;
        result.terminal_target_active = true;
        result.terminal_target_side |= 1;
        result.active_wrist_position = current.head<3>();
      }
    }
    if (input.reference.has_terminal_right_wrist_target) {
      const Eigen::Vector3d terminal =
          reference_root * input.reference.terminal_right_wrist_in_reference_pelvis;
      const double distance = (terminal - current.tail<3>()).norm();
      if (result.terminal_target_distance_m < 0.0 ||
          distance < result.terminal_target_distance_m) {
        result.terminal_target_distance_m = distance;
        result.terminal_target_position = terminal;
      }
      if (cartesian_target_mode || distance <= result.terminal_capture_radius_m) {
        target.tail<3>() = terminal;
        result.terminal_target_active = true;
        result.terminal_target_side |= 2;
        result.active_wrist_position = current.tail<3>();
      }
    }
    const TaskVector error = target - current;
    result.error_before = error.norm();
    result.arm_only_error_before = DecomposedError(
        input.reference.q_ref, q_linearization, target, reference_root,
        kArmHardwareIndices);
    result.waist_only_error_before = DecomposedError(
        input.reference.q_ref, q_linearization, target, reference_root,
        kWaistHardwareIndices);
    result.lower_only_error_before = LowerOnlyError(
        input.reference.q_ref, q_linearization, target, reference_root);
    result.upper_only_error_before = DecomposedError(
        input.reference.q_ref, q_linearization, target, reference_root,
        kUpperHardwareIndices);

    const CartesianServoSchedule cartesian_schedule =
        SelectCartesianServoSchedule(input.reference, result.terminal_target_distance_m);
    Eigen::Vector3d scheduled_position_weight = config_.position_weight;
    ActiveVector scheduled_joint_tracking_weight = config_.joint_tracking_weight;
    ActiveVector scheduled_max_velocity_rad_s = config_.max_velocity_rad_s;
    if (cartesian_schedule.active) {
      scheduled_position_weight *= cartesian_schedule.position_scale;
      ForEachTargetArmIndex(input.reference, [&](int i) {
        scheduled_joint_tracking_weight[i] *= cartesian_schedule.active_arm_joint_scale;
        scheduled_max_velocity_rad_s[i] =
            cartesian_schedule.active_arm_velocity_rad_s;
      });
    }

    TaskVector task_weights;
    const Eigen::Vector3d normalized_position_weights =
        scheduled_position_weight.cwiseSqrt() / config_.position_tolerance_m;
    task_weights.head<3>() = normalized_position_weights;
    task_weights.tail<3>() = normalized_position_weights;
    const JointVector& reference_prior = input.reference.q_ref;
    const ActiveVector reference_delta =
        ExtractActive(reference_prior) - ExtractActive(input.q_sonic);
    result.reference_joint_error_before = reference_delta.norm();
    const ActiveVector normalized_joint_weight =
        scheduled_joint_tracking_weight.array() /
        config_.joint_tolerance_rad.array().square();
    ActiveVector inactive_arm_hold_weight = ActiveVector::Zero();
    if (cartesian_target_mode) {
      const double hold_weight = config_.inactive_arm_hold_weight /
          (config_.inactive_arm_hold_tolerance_rad *
           config_.inactive_arm_hold_tolerance_rad);
      if (input.reference.has_terminal_left_wrist_target &&
          !input.reference.has_terminal_right_wrist_target) {
        inactive_arm_hold_weight.tail<7>().setConstant(hold_weight);
      } else if (input.reference.has_terminal_right_wrist_target &&
                 !input.reference.has_terminal_left_wrist_target) {
        inactive_arm_hold_weight.segment<7>(3).setConstant(hold_weight);
      }
    }
    const ActiveVector sonic_active = ExtractActive(input.q_sonic);
    const ActiveVector correction_lower =
        (config_.joint_lower - sonic_active).cwiseMax(-config_.max_delta_rad);
    const ActiveVector correction_upper =
        (config_.joint_upper - sonic_active).cwiseMin(config_.max_delta_rad);
    const ActiveVector projected_previous_delta =
        previous_delta_.cwiseMax(correction_lower).cwiseMin(correction_upper);

    ActiveVector lower = correction_lower;
    ActiveVector upper = correction_upper;

    const ActiveVector max_rate_step = scheduled_max_velocity_rad_s * input.dt;
    const ActiveVector rate_lower = projected_previous_delta - max_rate_step;
    const ActiveVector rate_upper = projected_previous_delta + max_rate_step;
    lower = lower.cwiseMax(rate_lower);
    upper = upper.cwiseMin(rate_upper);
    LockCartesianWaistBounds(input.reference, lower, upper);
    LockInactiveArmBounds(input.reference, lower, upper);
    if ((lower.array() > upper.array()).any()) {
      return Fallback(input, EeQpFallbackReason::kInfeasibleBounds);
    }

    Eigen::MatrixXd equality(0, kNumUpperJoints);
    Eigen::VectorXd equality_target(0);
    const Eigen::MatrixXd box = Eigen::MatrixXd::Identity(
        kNumUpperJoints, kNumUpperJoints);

    auto solve_total_delta_at = [&](const ActiveVector& linearization_delta,
                                    const ActiveVector& initial_guess,
                                    ActiveVector& solution) {
      JointVector q_iter = q_linearization;
      JointVector full_linearization_delta = JointVector::Zero();
      InsertActive(linearization_delta, full_linearization_delta);
      q_iter += full_linearization_delta;

      WristKinematics iter_wrist;
      if (!kinematics_->Evaluate(
              q_iter, input.measured_pelvis_orientation.normalized(), iter_wrist) ||
          !iter_wrist.left_position_world.allFinite() ||
          !iter_wrist.right_position_world.allFinite() ||
          !iter_wrist.position_jacobian_world.allFinite()) {
        return EeQpFallbackReason::kKinematicsFailure;
      }

      const auto weighted_jacobian = task_weights.asDiagonal() *
          iter_wrist.position_jacobian_world;
      // The QP variable remains the total correction relative to q_sonic.
      // Around q_linearization + linearization_delta:
      //   p(q + y) ~= p_iter + J_iter * (y - linearization_delta)
      // so the linear least-squares target is
      //   J_iter * y ~= target - p_iter + J_iter * linearization_delta.
      const TaskVector effective_error =
          target - MakeTaskVector(iter_wrist) +
          iter_wrist.position_jacobian_world * linearization_delta;
      const TaskVector weighted_error =
          task_weights.asDiagonal() * effective_error;

      Eigen::MatrixXd hessian =
          weighted_jacobian.transpose() * weighted_jacobian;
      hessian.diagonal().array() +=
          normalized_joint_weight.array() + inactive_arm_hold_weight.array() +
          config_.previous_delta_weight;
      Eigen::VectorXd gradient =
          -weighted_jacobian.transpose() * weighted_error -
          normalized_joint_weight.asDiagonal() * reference_delta -
          config_.previous_delta_weight * projected_previous_delta;

      const auto started = std::chrono::steady_clock::now();
      if (!initialized_) {
        qp_.init(hessian, gradient, equality, equality_target, box, lower, upper);
        initialized_ = true;
      } else {
        qp_.update(hessian, gradient, equality, equality_target, box, lower, upper);
      }
      ActiveVector guess = initial_guess.cwiseMax(lower).cwiseMin(upper);
      ZeroLockedDelta(input.reference, guess);
      qp_.solve(guess, proxsuite::nullopt, proxsuite::nullopt);
      result.solve_time_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();

      if (qp_.results.info.status !=
          proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
        return EeQpFallbackReason::kSolverStatus;
      }
      if (!qp_.results.x.allFinite()) {
        return EeQpFallbackReason::kNonFiniteSolution;
      }
      solution = qp_.results.x;
      ZeroLockedDelta(input.reference, solution);
      return EeQpFallbackReason::kNone;
    };

    ActiveVector initial_guess = reference_delta.cwiseMax(lower).cwiseMin(upper);
    ZeroLockedDelta(input.reference, initial_guess);
    ActiveVector qp_delta = ActiveVector::Zero();
    EeQpFallbackReason solve_failure =
        solve_total_delta_at(ActiveVector::Zero(), initial_guess, qp_delta);
    const int relinearization_iterations = cartesian_target_mode
        ? std::clamp(config_.cartesian_relinearization_iterations, 1, 4)
        : 1;
    for (int iter = 1;
         solve_failure == EeQpFallbackReason::kNone &&
         iter < relinearization_iterations;
         ++iter) {
      ActiveVector refined_delta = qp_delta;
      solve_failure = solve_total_delta_at(qp_delta, qp_delta, refined_delta);
      if (solve_failure == EeQpFallbackReason::kNone) {
        qp_delta = refined_delta;
      }
    }
    if (solve_failure == EeQpFallbackReason::kNone &&
        result.solve_time_ms > config_.max_solve_time_ms) {
      solve_failure = EeQpFallbackReason::kSolveTimeout;
    }
    if (solve_failure != EeQpFallbackReason::kNone) {
      EeQpResult fallback = Fallback(input, solve_failure);
      fallback.solve_time_ms = result.solve_time_ms;
      fallback.error_before = result.error_before;
      fallback.left_wrist_position = result.left_wrist_position;
      fallback.right_wrist_position = result.right_wrist_position;
      fallback.active_wrist_position = result.active_wrist_position;
      fallback.terminal_target_active = result.terminal_target_active;
      fallback.terminal_target_side = result.terminal_target_side;
      fallback.terminal_target_position = result.terminal_target_position;
      fallback.terminal_target_distance_m = result.terminal_target_distance_m;
      fallback.terminal_capture_radius_m = result.terminal_capture_radius_m;
      return fallback;
    }

    result.correction_bound_saturation_fraction = SaturationFraction(
        qp_delta, correction_lower, correction_upper);
    result.rate_bound_saturation_fraction = SaturationFraction(
        qp_delta, rate_lower, rate_upper);
    result.correction_bound_saturation_mask = SaturationMask(
        qp_delta, correction_lower, correction_upper);
    result.rate_bound_saturation_mask = SaturationMask(
        qp_delta, rate_lower, rate_upper);
    ActiveVector proposed_delta = qp_delta;
    ZeroLockedDelta(input.reference, proposed_delta);
    elapsed_sec_ += input.dt;
    const double ramp = config_.startup_ramp_sec > 0.0
        ? std::clamp(elapsed_sec_ / config_.startup_ramp_sec, 0.0, 1.0)
        : 1.0;
    proposed_delta *= ramp;
    ActiveVector accepted_delta = proposed_delta;
    bool accepted = !config_.nonlinear_validation_enabled;

    if (config_.nonlinear_validation_enabled) {
      for (double alpha = config_.backtracking_initial_alpha;
           alpha + 1e-12 >= config_.backtracking_min_alpha;
           alpha *= config_.backtracking_factor) {
        // Interpolate from the previous correction so reducing alpha cannot
        // introduce a new rate-limit jump toward zero.
        ActiveVector candidate_delta =
            previous_delta_ + alpha * (proposed_delta - previous_delta_);
        ZeroLockedDelta(input.reference, candidate_delta);
        JointVector candidate_q = q_linearization;
        JointVector candidate_full_delta = JointVector::Zero();
        InsertActive(candidate_delta, candidate_full_delta);
        candidate_q += candidate_full_delta;

        WristKinematics candidate_wrist;
        if (!kinematics_->Evaluate(
                candidate_q, input.measured_pelvis_orientation.normalized(),
                candidate_wrist) ||
            !candidate_wrist.left_position_world.allFinite() ||
            !candidate_wrist.right_position_world.allFinite()) {
          continue;
        }
        TaskVector nonlinear_position = MakeTaskVector(candidate_wrist);
        const double candidate_error = (target - nonlinear_position).norm();
        if (candidate_error + config_.nonlinear_min_improvement_m <=
            result.error_before) {
          accepted_delta = candidate_delta;
          result.nonlinear_error_after = candidate_error;
          result.backtracking_alpha = alpha;
          accepted = true;
          break;
        }
      }
    }

    result.predicted_error_after =
        (wrist.position_jacobian_world * accepted_delta - error).norm();
    result.predicted_reference_joint_error_after =
        (accepted_delta - reference_delta).norm();
    if (!config_.nonlinear_validation_enabled) {
      result.nonlinear_error_after = result.predicted_error_after;
      result.backtracking_alpha = 1.0;
    }

    if (!accepted) {
      EeQpResult fallback =
          Fallback(input, EeQpFallbackReason::kCandidateRejected);
      fallback.error_before = result.error_before;
      fallback.predicted_error_after = result.predicted_error_after;
      fallback.nonlinear_error_after = result.nonlinear_error_after;
      fallback.solve_time_ms = result.solve_time_ms;
      fallback.reference_age_ms = result.reference_age_ms;
      fallback.left_wrist_position = result.left_wrist_position;
      fallback.right_wrist_position = result.right_wrist_position;
      fallback.active_wrist_position = result.active_wrist_position;
      fallback.correction_bound_saturation_fraction =
          result.correction_bound_saturation_fraction;
      fallback.rate_bound_saturation_fraction =
          result.rate_bound_saturation_fraction;
      fallback.correction_bound_saturation_mask =
          result.correction_bound_saturation_mask;
      fallback.rate_bound_saturation_mask =
          result.rate_bound_saturation_mask;
      fallback.terminal_target_active = result.terminal_target_active;
      fallback.terminal_target_side = result.terminal_target_side;
      fallback.terminal_target_position = result.terminal_target_position;
      fallback.terminal_target_distance_m = result.terminal_target_distance_m;
      fallback.terminal_capture_radius_m = result.terminal_capture_radius_m;
      fallback.candidate_rejected = true;
      return fallback;
    }

    ZeroLockedDelta(input.reference, accepted_delta);
    JointVector full_delta = JointVector::Zero();
    InsertActive(accepted_delta, full_delta);
    result.q_corrected = input.q_sonic + full_delta;
    CopyInactiveArmFromReference(input.reference, result.q_corrected);
    result.delta_q = result.q_corrected - input.q_sonic;

    previous_delta_ = accepted_delta;
    result.valid = true;
    return result;
  }

  void Reset() {
    previous_delta_.setZero();
    elapsed_sec_ = 0.0;
    initialized_ = false;
    qp_.cleanup();
  }

 private:
  CartesianServoSchedule SelectCartesianServoSchedule(
      const EeReference& reference, double target_distance_m) const {
    CartesianServoSchedule schedule;
    if (!config_.cartesian_servo_schedule_enabled ||
        reference.reference_type != ReferenceType::kCartesianTarget ||
        (!reference.has_terminal_left_wrist_target &&
         !reference.has_terminal_right_wrist_target) ||
        !std::isfinite(target_distance_m) ||
        target_distance_m < 0.0) {
      return schedule;
    }
    schedule.active = true;
    if (target_distance_m > config_.cartesian_far_distance_m) {
      schedule.position_scale = config_.cartesian_position_weight_scale_far;
      schedule.active_arm_joint_scale =
          config_.cartesian_active_arm_joint_weight_scale_far;
      schedule.active_arm_velocity_rad_s =
          config_.cartesian_active_arm_max_velocity_far_rad_s;
    } else if (target_distance_m > config_.cartesian_mid_distance_m) {
      schedule.position_scale = config_.cartesian_position_weight_scale_mid;
      schedule.active_arm_joint_scale =
          config_.cartesian_active_arm_joint_weight_scale_mid;
      schedule.active_arm_velocity_rad_s =
          config_.cartesian_active_arm_max_velocity_mid_rad_s;
    } else if (target_distance_m > config_.cartesian_near_distance_m) {
      schedule.position_scale = config_.cartesian_position_weight_scale_near;
      schedule.active_arm_joint_scale =
          config_.cartesian_active_arm_joint_weight_scale_near;
      schedule.active_arm_velocity_rad_s =
          config_.cartesian_active_arm_max_velocity_near_rad_s;
    } else {
      schedule.position_scale = config_.cartesian_position_weight_scale_close;
      schedule.active_arm_joint_scale =
          config_.cartesian_active_arm_joint_weight_scale_close;
      schedule.active_arm_velocity_rad_s =
          config_.cartesian_active_arm_max_velocity_close_rad_s;
    }
    return schedule;
  }

  template <std::size_t N>
  double DecomposedError(
      const JointVector& q_ref,
      const JointVector& q_linearization,
      const TaskVector& target,
      const Eigen::Quaterniond& reference_root,
      const std::array<int, N>& indices) const {
    JointVector q = q_ref;
    for (const int hw : indices) {
      q[hw] = q_linearization[hw];
    }
    WristKinematics wrist;
    if (!kinematics_->Evaluate(q, reference_root, wrist) ||
        !wrist.left_position_world.allFinite() ||
        !wrist.right_position_world.allFinite()) {
      return 0.0;
    }
    return (target - MakeTaskVector(wrist)).norm();
  }

  double LowerOnlyError(
      const JointVector& q_ref,
      const JointVector& q_linearization,
      const TaskVector& target,
      const Eigen::Quaterniond& reference_root) const {
    JointVector q = q_ref;
    for (int hw = 0; hw < kWaistHardwareIndices.front(); ++hw) {
      q[hw] = q_linearization[hw];
    }
    WristKinematics wrist;
    if (!kinematics_->Evaluate(q, reference_root, wrist) ||
        !wrist.left_position_world.allFinite() ||
        !wrist.right_position_world.allFinite()) {
      return 0.0;
    }
    return (target - MakeTaskVector(wrist)).norm();
  }

  EeQpResult Fallback(
      const EeQpInput& input, EeQpFallbackReason reason) {
    EeQpResult result;
    result.q_corrected = input.q_sonic;
    result.fallback_active = true;
    result.fallback_reason = reason;
    if (input.reference.valid && input.now_us >= input.reference.timestamp_us) {
      result.reference_age_ms =
          static_cast<double>(input.now_us - input.reference.timestamp_us) /
          1000.0;
    }
    result.reference_source = input.reference.source;
    result.control_mode = input.reference.control_mode;
    result.plan_id = input.reference.plan_id;
    result.plan_phase = input.reference.phase;
    result.reference_type = input.reference.reference_type;
    result.reference_feasible = input.reference.feasible;

    switch (config_.invalid_fallback_mode) {
      case InvalidFallbackMode::kSonic:
        previous_delta_.setZero();
        break;
      case InvalidFallbackMode::kHold:
        break;
      case InvalidFallbackMode::kDecay: {
        const double factor = config_.invalid_decay_time_sec > 0.0 &&
                std::isfinite(input.dt) && input.dt > 0.0
            ? std::exp(-input.dt / config_.invalid_decay_time_sec)
            : 0.0;
        previous_delta_ *= factor;
        break;
      }
    }

    if (input.q_sonic.allFinite()) {
      const ActiveVector sonic_active = ExtractActive(input.q_sonic);
      const ActiveVector lower = (config_.joint_lower - sonic_active)
          .cwiseMax(-config_.max_delta_rad);
      const ActiveVector upper = (config_.joint_upper - sonic_active)
          .cwiseMin(config_.max_delta_rad);
      if ((lower.array() <= upper.array()).all()) {
        previous_delta_ = previous_delta_.cwiseMax(lower).cwiseMin(upper);
      } else {
        previous_delta_.setZero();
      }
    } else {
      previous_delta_.setZero();
    }
    ZeroLockedDelta(input.reference, previous_delta_);

    JointVector full_delta = JointVector::Zero();
    InsertActive(previous_delta_, full_delta);
    result.q_corrected += full_delta;
    CopyInactiveArmFromReference(input.reference, result.q_corrected);
    result.delta_q = result.q_corrected - input.q_sonic;
    return result;
  }

  EeQpConfig config_;
  std::shared_ptr<const EeKinematics> kinematics_;
  proxsuite::proxqp::dense::QP<double> qp_;
  ActiveVector previous_delta_ = ActiveVector::Zero();
  double elapsed_sec_ = 0.0;
  bool initialized_ = false;
};

EeQpConfig LoadEeQpConfig(const std::string& yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path);
  EeQpConfig config;
  const auto weights = root["task"]["position_weight"];
  for (int i = 0; i < 3; ++i) config.position_weight[i] = weights[i].as<double>();
  config.position_tolerance_m =
      root["task"]["position_tolerance_m"].as<double>();
  config.joint_tracking_weight = ReadActiveVector(
      root["task"], "joint_tracking_weight", config.joint_tracking_weight);
  config.joint_tolerance_rad = ReadActiveVector(
      root["task"], "joint_tolerance_rad", config.joint_tolerance_rad);
  if (root["task"]["inactive_arm_hold_weight"]) {
    config.inactive_arm_hold_weight =
        root["task"]["inactive_arm_hold_weight"].as<double>();
  }
  if (root["task"]["inactive_arm_hold_tolerance_rad"]) {
    config.inactive_arm_hold_tolerance_rad =
        root["task"]["inactive_arm_hold_tolerance_rad"].as<double>();
  }
  if (const auto schedule = root["task"]["cartesian_servo_schedule"]) {
    if (schedule["enabled"]) {
      config.cartesian_servo_schedule_enabled =
          schedule["enabled"].as<bool>();
    }
    if (schedule["far_distance_m"]) {
      config.cartesian_far_distance_m =
          schedule["far_distance_m"].as<double>();
    }
    if (schedule["mid_distance_m"]) {
      config.cartesian_mid_distance_m =
          schedule["mid_distance_m"].as<double>();
    }
    if (schedule["near_distance_m"]) {
      config.cartesian_near_distance_m =
          schedule["near_distance_m"].as<double>();
    }
    if (schedule["position_weight_scale"]) {
      const auto scales = schedule["position_weight_scale"];
      config.cartesian_position_weight_scale_far =
          scales["far"].as<double>();
      config.cartesian_position_weight_scale_mid =
          scales["mid"].as<double>();
      config.cartesian_position_weight_scale_near =
          scales["near"].as<double>();
      config.cartesian_position_weight_scale_close =
          scales["close"].as<double>();
    }
    if (schedule["active_arm_joint_weight_scale"]) {
      const auto scales = schedule["active_arm_joint_weight_scale"];
      config.cartesian_active_arm_joint_weight_scale_far =
          scales["far"].as<double>();
      config.cartesian_active_arm_joint_weight_scale_mid =
          scales["mid"].as<double>();
      config.cartesian_active_arm_joint_weight_scale_near =
          scales["near"].as<double>();
      config.cartesian_active_arm_joint_weight_scale_close =
          scales["close"].as<double>();
    }
    if (schedule["active_arm_max_velocity_rad_s"]) {
      const auto velocities = schedule["active_arm_max_velocity_rad_s"];
      config.cartesian_active_arm_max_velocity_far_rad_s =
          velocities["far"].as<double>();
      config.cartesian_active_arm_max_velocity_mid_rad_s =
          velocities["mid"].as<double>();
      config.cartesian_active_arm_max_velocity_near_rad_s =
          velocities["near"].as<double>();
      config.cartesian_active_arm_max_velocity_close_rad_s =
          velocities["close"].as<double>();
    }
  }
  config.previous_delta_weight = root["previous_delta_weight"].as<double>();
  config.reference_timeout_us =
      root["reference_timeout_ms"].as<std::int64_t>() * 1000;
  YAML::Node reference = root["reference"];
  if (!reference) reference = root["gmr_reference"];
  if (reference) {
    config.reference_host = reference["host"].as<std::string>();
    config.reference_port = reference["port"].as<int>();
    config.reference_topic = reference["topic"].as<std::string>();
  }
  config.startup_ramp_sec = root["startup_ramp_sec"].as<double>();
  config.max_solve_time_ms = root["max_solve_time_ms"].as<double>();
  config.max_delta_rad =
      ReadActiveVector(root, "max_delta_rad", config.max_delta_rad);
  config.max_velocity_rad_s =
      ReadActiveVector(root, "max_velocity_rad_s", config.max_velocity_rad_s);
  if (const auto validation = root["nonlinear_validation"]) {
    config.nonlinear_validation_enabled = validation["enabled"].as<bool>();
    if (validation["cartesian_relinearization_iterations"]) {
      config.cartesian_relinearization_iterations =
          validation["cartesian_relinearization_iterations"].as<int>();
    }
    config.nonlinear_min_improvement_m =
        validation["min_improvement_m"].as<double>();
    if (validation["backtracking_initial_alpha"]) {
      config.backtracking_initial_alpha =
          validation["backtracking_initial_alpha"].as<double>();
    }
    if (validation["backtracking_factor"]) {
      config.backtracking_factor =
          validation["backtracking_factor"].as<double>();
    }
    if (validation["backtracking_min_alpha"]) {
      config.backtracking_min_alpha =
          validation["backtracking_min_alpha"].as<double>();
    }
  }
  if (const auto fallback = root["invalid_fallback"]) {
    const std::string mode = fallback["mode"].as<std::string>();
    if (mode == "sonic") {
      config.invalid_fallback_mode = InvalidFallbackMode::kSonic;
    } else if (mode == "hold") {
      config.invalid_fallback_mode = InvalidFallbackMode::kHold;
    } else if (mode == "decay") {
      config.invalid_fallback_mode = InvalidFallbackMode::kDecay;
    } else {
      throw std::invalid_argument("Unknown EE QP invalid fallback mode: " + mode);
    }
    config.invalid_decay_time_sec = fallback["decay_time_sec"].as<double>();
  }
  const auto lower = root["joint_lower_rad"];
  const auto upper = root["joint_upper_rad"];
  if (!lower || !upper || lower.size() != kNumUpperJoints ||
      upper.size() != kNumUpperJoints) {
    throw std::invalid_argument(
        "joint_lower_rad and joint_upper_rad must contain exactly " +
        std::to_string(kNumUpperJoints) + " values");
  }
  for (int i = 0; i < kNumUpperJoints; ++i) {
    config.joint_lower[i] = lower[i].as<double>();
    config.joint_upper[i] = upper[i].as<double>();
  }
  return config;
}

EeQpCorrector::EeQpCorrector(
    EeQpConfig config, std::shared_ptr<const EeKinematics> kinematics)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(kinematics))) {}

EeQpCorrector::~EeQpCorrector() = default;

EeQpResult EeQpCorrector::Compute(const EeQpInput& input) {
  return impl_->Compute(input);
}

void EeQpCorrector::Reset() { impl_->Reset(); }

}  // namespace g1::ee_qp
