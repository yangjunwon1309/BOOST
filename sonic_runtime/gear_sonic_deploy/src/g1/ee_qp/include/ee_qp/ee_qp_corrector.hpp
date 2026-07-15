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
  double previous_delta_weight = 1e-2;
  ActiveVector max_delta_rad = ActiveVector::Constant(0.05);
  ActiveVector max_velocity_rad_s = ActiveVector::Constant(0.5);
  double startup_ramp_sec = 1.5;
  double max_solve_time_ms = 2.0;
  bool nonlinear_validation_enabled = true;
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
