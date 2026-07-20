#pragma once

#include <memory>
#include <string>

#include "ee_qp/ee_kinematics.hpp"
#include "ee_qp/types.hpp"

namespace g1::ee_qp {

enum class InvalidFallbackMode {
  kSonic,
  kHold,
  kDecay,
};

struct EeQpConfig {
  Eigen::Vector3d position_weight = Eigen::Vector3d::Ones();
  double position_tolerance_m = 0.03;
  ActiveVector joint_tracking_weight = ActiveVector::Constant(0.8);
  ActiveVector joint_tolerance_rad = ActiveVector::Constant(0.15);
  double inactive_arm_hold_weight = 3.0;
  double inactive_arm_hold_tolerance_rad = 0.08;
  bool cartesian_servo_schedule_enabled = true;
  double cartesian_far_distance_m = 0.10;
  double cartesian_mid_distance_m = 0.05;
  double cartesian_near_distance_m = 0.02;
  double cartesian_position_weight_scale_far = 6.0;
  double cartesian_position_weight_scale_mid = 12.0;
  double cartesian_position_weight_scale_near = 24.0;
  double cartesian_position_weight_scale_close = 36.0;
  double cartesian_active_arm_joint_weight_scale_far = 0.75;
  double cartesian_active_arm_joint_weight_scale_mid = 0.55;
  double cartesian_active_arm_joint_weight_scale_near = 0.35;
  double cartesian_active_arm_joint_weight_scale_close = 0.25;
  double cartesian_active_arm_max_velocity_far_rad_s = 0.65;
  double cartesian_active_arm_max_velocity_mid_rad_s = 0.60;
  double cartesian_active_arm_max_velocity_near_rad_s = 0.45;
  double cartesian_active_arm_max_velocity_close_rad_s = 0.30;
  double previous_delta_weight = 1e-2;
  ActiveVector max_delta_rad = ActiveVector::Constant(0.05);
  ActiveVector max_velocity_rad_s = ActiveVector::Constant(0.5);
  double startup_ramp_sec = 1.5;
  double max_solve_time_ms = 2.0;
  bool nonlinear_validation_enabled = true;
  int cartesian_relinearization_iterations = 2;
  double nonlinear_min_improvement_m = 0.0;
  double backtracking_initial_alpha = 1.0;
  double backtracking_factor = 0.5;
  double backtracking_min_alpha = 0.125;
  InvalidFallbackMode invalid_fallback_mode = InvalidFallbackMode::kDecay;
  double invalid_decay_time_sec = 0.2;
  std::int64_t reference_timeout_us = 100000;
  std::string reference_host = "127.0.0.1";
  int reference_port = 5558;
  std::string reference_topic = "gmr_ref";
  ActiveVector joint_lower = ActiveVector::Constant(-3.14);
  ActiveVector joint_upper = ActiveVector::Constant(3.14);
};

EeQpConfig LoadEeQpConfig(const std::string& yaml_path);

class EeQpCorrector {
 public:
  EeQpCorrector(EeQpConfig config, std::shared_ptr<const EeKinematics> kinematics);
  ~EeQpCorrector();

  EeQpCorrector(const EeQpCorrector&) = delete;
  EeQpCorrector& operator=(const EeQpCorrector&) = delete;

  EeQpResult Compute(const EeQpInput& input);
  void Reset();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g1::ee_qp
