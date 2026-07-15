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
  return input.q_measured.allFinite() && input.dq_measured.allFinite() &&
         input.q_sonic.allFinite() &&
         input.measured_pelvis_orientation.coeffs().allFinite() &&
         input.reference.left_wrist_in_reference_pelvis.allFinite() &&
         input.reference.right_wrist_in_reference_pelvis.allFinite() &&
         input.reference.reference_pelvis_orientation.coeffs().allFinite() &&
         input.reference.q_ref.allFinite() &&
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
        (config_.max_delta_rad.array() <= 0.0).any() ||
        (config_.max_velocity_rad_s.array() <= 0.0).any() ||
        config_.previous_delta_weight < 0.0 ||
        config_.backtracking_initial_alpha <= 0.0 ||
        config_.backtracking_initial_alpha > 1.0 ||
        config_.backtracking_factor <= 0.0 ||
        config_.backtracking_factor >= 1.0 ||
        config_.backtracking_min_alpha <= 0.0 ||
        config_.backtracking_min_alpha > config_.backtracking_initial_alpha) {
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
    result.reference_feasible = input.reference.feasible;
    const Eigen::Quaterniond reference_root =
        input.reference.reference_pelvis_orientation.normalized();
    TaskVector target;
    target.head<3>() = reference_root *
        input.reference.left_wrist_in_reference_pelvis;
    target.tail<3>() = reference_root *
        input.reference.right_wrist_in_reference_pelvis;

    TaskVector current = MakeTaskVector(wrist);
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

    TaskVector task_weights;
    const Eigen::Vector3d normalized_position_weights =
        config_.position_weight.cwiseSqrt() / config_.position_tolerance_m;
    task_weights.head<3>() = normalized_position_weights;
    task_weights.tail<3>() = normalized_position_weights;
    const auto weighted_jacobian = task_weights.asDiagonal() *
        wrist.position_jacobian_world;
    const TaskVector weighted_error = task_weights.asDiagonal() * error;
    const ActiveVector reference_delta =
        ExtractActive(input.reference.q_ref) - ExtractActive(input.q_sonic);
    result.reference_joint_error_before = reference_delta.norm();
    const ActiveVector normalized_joint_weight =
        config_.joint_tracking_weight.array() /
        config_.joint_tolerance_rad.array().square();

    Eigen::MatrixXd hessian = weighted_jacobian.transpose() * weighted_jacobian;
    hessian.diagonal().array() +=
        normalized_joint_weight.array() + config_.previous_delta_weight;
    Eigen::VectorXd gradient = -weighted_jacobian.transpose() * weighted_error -
        normalized_joint_weight.asDiagonal() * reference_delta -
        config_.previous_delta_weight * previous_delta_;

    const ActiveVector sonic_active = ExtractActive(input.q_sonic);
    const ActiveVector correction_lower =
        (config_.joint_lower - sonic_active).cwiseMax(-config_.max_delta_rad);
    const ActiveVector correction_upper =
        (config_.joint_upper - sonic_active).cwiseMin(config_.max_delta_rad);
    ActiveVector lower = correction_lower;
    ActiveVector upper = correction_upper;

    const ActiveVector max_rate_step = config_.max_velocity_rad_s * input.dt;
    const ActiveVector rate_lower = previous_delta_ - max_rate_step;
    const ActiveVector rate_upper = previous_delta_ + max_rate_step;
    lower = lower.cwiseMax(rate_lower);
    upper = upper.cwiseMin(rate_upper);
    if ((lower.array() > upper.array()).any()) {
      return Fallback(input, EeQpFallbackReason::kInfeasibleBounds);
    }

    Eigen::MatrixXd equality(0, kNumUpperJoints);
    Eigen::VectorXd equality_target(0);
    const Eigen::MatrixXd box = Eigen::MatrixXd::Identity(
        kNumUpperJoints, kNumUpperJoints);

    const auto started = std::chrono::steady_clock::now();
    if (!initialized_) {
      qp_.init(hessian, gradient, equality, equality_target, box, lower, upper);
      initialized_ = true;
    } else {
      qp_.update(hessian, gradient, equality, equality_target, box, lower, upper);
    }
    const ActiveVector initial_guess = reference_delta.cwiseMax(lower).cwiseMin(upper);
    qp_.solve(initial_guess, proxsuite::nullopt, proxsuite::nullopt);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    EeQpFallbackReason solve_failure = EeQpFallbackReason::kNone;
    if (qp_.results.info.status !=
        proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
      solve_failure = EeQpFallbackReason::kSolverStatus;
    } else if (result.solve_time_ms > config_.max_solve_time_ms) {
      solve_failure = EeQpFallbackReason::kSolveTimeout;
    } else if (!qp_.results.x.allFinite()) {
      solve_failure = EeQpFallbackReason::kNonFiniteSolution;
    }
    if (solve_failure != EeQpFallbackReason::kNone) {
      EeQpResult fallback = Fallback(input, solve_failure);
      fallback.solve_time_ms = result.solve_time_ms;
      fallback.error_before = result.error_before;
      return fallback;
    }

    const ActiveVector qp_delta = qp_.results.x;
    result.correction_bound_saturation_fraction = SaturationFraction(
        qp_delta, correction_lower, correction_upper);
    result.rate_bound_saturation_fraction = SaturationFraction(
        qp_delta, rate_lower, rate_upper);
    result.correction_bound_saturation_mask = SaturationMask(
        qp_delta, correction_lower, correction_upper);
    result.rate_bound_saturation_mask = SaturationMask(
        qp_delta, rate_lower, rate_upper);
    ActiveVector proposed_delta = qp_delta;
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
        const ActiveVector candidate_delta =
            previous_delta_ + alpha * (proposed_delta - previous_delta_);
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
      fallback.correction_bound_saturation_fraction =
          result.correction_bound_saturation_fraction;
      fallback.rate_bound_saturation_fraction =
          result.rate_bound_saturation_fraction;
      fallback.correction_bound_saturation_mask =
          result.correction_bound_saturation_mask;
      fallback.rate_bound_saturation_mask =
          result.rate_bound_saturation_mask;
      fallback.candidate_rejected = true;
      return fallback;
    }

    JointVector full_delta = JointVector::Zero();
    InsertActive(accepted_delta, full_delta);
    result.delta_q = full_delta;
    result.q_corrected = input.q_sonic + full_delta;

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

    JointVector full_delta = JointVector::Zero();
    InsertActive(previous_delta_, full_delta);
    result.delta_q = full_delta;
    result.q_corrected += full_delta;
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
