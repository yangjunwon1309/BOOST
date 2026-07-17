#pragma once

#include <array>
#include <chrono>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g1::ee_qp {

inline constexpr int kNumJoints = 29;
inline constexpr int kNumUpperJoints = 17;
inline constexpr int kNumArmJoints = 14;
inline constexpr int kTaskDimension = 6;

using JointVector = Eigen::Matrix<double, kNumJoints, 1>;
using ActiveVector = Eigen::Matrix<double, kNumUpperJoints, 1>;
using ArmVector = Eigen::Matrix<double, kNumArmJoints, 1>;
using TaskVector = Eigen::Matrix<double, kTaskDimension, 1>;
using ActiveJacobian = Eigen::Matrix<double, kTaskDimension, kNumUpperJoints>;

inline constexpr std::array<int, 3> kWaistHardwareIndices = {12, 13, 14};

inline constexpr std::array<int, kNumUpperJoints> kUpperHardwareIndices = {
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};

inline constexpr std::array<int, kNumArmJoints> kArmHardwareIndices = {
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};

enum class EeQpFallbackReason : std::uint8_t {
  kNone = 0,
  kMissingReference,
  kNonFiniteInput,
  kFutureReference,
  kStaleReference,
  kKinematicsFailure,
  kInfeasibleBounds,
  kSolverStatus,
  kSolveTimeout,
  kNonFiniteSolution,
  kCandidateRejected,
  kInfeasibleReference,
};

enum class ReferenceSource : std::uint8_t {
  kUnknown = 0,
  kGmr,
  kCurobo,
  kReplay,
  kDampedReturn,
};

enum class ControlMode : std::uint8_t {
  kUnknown = 0,
  kTeleop,
  kPlanPending,
  kPlanExec,
  kBlendBack,
  kIdle,
};

inline const char* FallbackReasonName(EeQpFallbackReason reason) {
  switch (reason) {
    case EeQpFallbackReason::kNone: return "none";
    case EeQpFallbackReason::kMissingReference: return "missing_reference";
    case EeQpFallbackReason::kNonFiniteInput: return "nonfinite_input";
    case EeQpFallbackReason::kFutureReference: return "future_reference";
    case EeQpFallbackReason::kStaleReference: return "stale_reference";
    case EeQpFallbackReason::kKinematicsFailure: return "kinematics_failure";
    case EeQpFallbackReason::kInfeasibleBounds: return "infeasible_bounds";
    case EeQpFallbackReason::kSolverStatus: return "solver_status";
    case EeQpFallbackReason::kSolveTimeout: return "solve_timeout";
    case EeQpFallbackReason::kNonFiniteSolution: return "nonfinite_solution";
    case EeQpFallbackReason::kCandidateRejected: return "candidate_rejected";
    case EeQpFallbackReason::kInfeasibleReference: return "infeasible_reference";
  }
  return "unknown";
}

inline const char* ReferenceSourceName(ReferenceSource source) {
  switch (source) {
    case ReferenceSource::kUnknown: return "unknown";
    case ReferenceSource::kGmr: return "gmr";
    case ReferenceSource::kCurobo: return "curobo";
    case ReferenceSource::kReplay: return "replay";
    case ReferenceSource::kDampedReturn: return "damped_return";
  }
  return "unknown";
}

inline const char* ControlModeName(ControlMode mode) {
  switch (mode) {
    case ControlMode::kUnknown: return "unknown";
    case ControlMode::kTeleop: return "teleop";
    case ControlMode::kPlanPending: return "plan_pending";
    case ControlMode::kPlanExec: return "plan_exec";
    case ControlMode::kBlendBack: return "blend_back";
    case ControlMode::kIdle: return "idle";
  }
  return "unknown";
}

struct EeReference {
  Eigen::Vector3d left_wrist_in_reference_pelvis = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_wrist_in_reference_pelvis = Eigen::Vector3d::Zero();
  Eigen::Quaterniond reference_pelvis_orientation = Eigen::Quaterniond::Identity();
  JointVector q_ref = JointVector::Zero();
  Eigen::Vector3d terminal_left_wrist_in_reference_pelvis = Eigen::Vector3d::Zero();
  Eigen::Vector3d terminal_right_wrist_in_reference_pelvis = Eigen::Vector3d::Zero();
  double terminal_capture_radius_m = 0.07;
  std::int64_t timestamp_us = 0;
  std::uint64_t plan_id = 0;
  double phase = 0.0;
  ReferenceSource source = ReferenceSource::kUnknown;
  ControlMode control_mode = ControlMode::kUnknown;
  bool has_terminal_left_wrist_target = false;
  bool has_terminal_right_wrist_target = false;
  bool feasible = true;
  bool valid = false;
};

struct EeQpInput {
  JointVector q_measured = JointVector::Zero();
  JointVector dq_measured = JointVector::Zero();
  JointVector q_sonic = JointVector::Zero();
  Eigen::Quaterniond measured_pelvis_orientation = Eigen::Quaterniond::Identity();
  EeReference reference;
  double dt = 0.02;
  std::int64_t now_us = 0;
};

struct EeQpResult {
  JointVector q_corrected = JointVector::Zero();
  JointVector delta_q = JointVector::Zero();
  double error_before = 0.0;
  double predicted_error_after = 0.0;
  double nonlinear_error_after = 0.0;
  double reference_joint_error_before = 0.0;
  double predicted_reference_joint_error_after = 0.0;
  double arm_only_error_before = 0.0;
  double waist_only_error_before = 0.0;
  double lower_only_error_before = 0.0;
  double upper_only_error_before = 0.0;
  double solve_time_ms = 0.0;
  double reference_age_ms = -1.0;
  Eigen::Vector3d terminal_target_position = Eigen::Vector3d::Zero();
  double terminal_target_distance_m = -1.0;
  double terminal_capture_radius_m = 0.0;
  double correction_bound_saturation_fraction = 0.0;
  double rate_bound_saturation_fraction = 0.0;
  ActiveVector correction_bound_saturation_mask = ActiveVector::Zero();
  ActiveVector rate_bound_saturation_mask = ActiveVector::Zero();
  double backtracking_alpha = 0.0;
  EeQpFallbackReason fallback_reason = EeQpFallbackReason::kNone;
  ReferenceSource reference_source = ReferenceSource::kUnknown;
  ControlMode control_mode = ControlMode::kUnknown;
  std::uint64_t plan_id = 0;
  double plan_phase = 0.0;
  bool fallback_active = false;
  bool candidate_rejected = false;
  bool reference_feasible = false;
  bool terminal_target_active = false;
  int terminal_target_side = 0;  // 0 none, 1 left, 2 right, 3 both.
  bool valid = false;
};

struct EeQpTelemetry {
  JointVector q_ref = JointVector::Zero();
  JointVector q_corrected = JointVector::Zero();
  JointVector delta_q = JointVector::Zero();
  JointVector joint_tracking_error = JointVector::Zero();
  double error_before_m = 0.0;
  double predicted_error_after_m = 0.0;
  double nonlinear_error_after_m = 0.0;
  double reference_joint_error_before_rad = 0.0;
  double predicted_reference_joint_error_after_rad = 0.0;
  double arm_only_error_before_m = 0.0;
  double waist_only_error_before_m = 0.0;
  double lower_only_error_before_m = 0.0;
  double upper_only_error_before_m = 0.0;
  double joint_tracking_rmse_rad = 0.0;
  double joint_tracking_max_abs_rad = 0.0;
  double pd_tracking_lag_rmse_rad = 0.0;
  double pd_tracking_lag_max_abs_rad = 0.0;
  double solve_time_ms = 0.0;
  double reference_age_ms = -1.0;
  Eigen::Vector3d terminal_target_position = Eigen::Vector3d::Zero();
  double terminal_target_distance_m = -1.0;
  double terminal_capture_radius_m = 0.0;
  double correction_bound_saturation_fraction = 0.0;
  double rate_bound_saturation_fraction = 0.0;
  ActiveVector correction_bound_saturation_mask = ActiveVector::Zero();
  ActiveVector rate_bound_saturation_mask = ActiveVector::Zero();
  double backtracking_alpha = 0.0;
  EeQpFallbackReason fallback_reason = EeQpFallbackReason::kNone;
  ReferenceSource reference_source = ReferenceSource::kUnknown;
  ControlMode control_mode = ControlMode::kUnknown;
  std::uint64_t plan_id = 0;
  double plan_phase = 0.0;
  bool enabled = false;
  bool fallback_active = false;
  bool candidate_rejected = false;
  bool reference_feasible = false;
  bool terminal_target_active = false;
  int terminal_target_side = 0;
  bool valid = false;
};

}  // namespace g1::ee_qp
