#ifndef UPPER_BODY_IK_ORIGIN_SOLVER_HPP
#define UPPER_BODY_IK_ORIGIN_SOLVER_HPP

#include <array>
#include <deque>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>

#include <Eigen/Dense>

#include "fk.hpp"
#include "math_utils.hpp"
#include "robot_parameters.hpp"
#include "policy_parameters.hpp"

// ---------------------------------------------------------------------------
// WeightedMovingFilterOrigin — C++ port of xr_teleoperate's filter
// ---------------------------------------------------------------------------
class WeightedMovingFilterOrigin {
 public:
  WeightedMovingFilterOrigin(const std::vector<double>& weights, int data_size)
      : weights_(weights), data_size_(data_size),
        filtered_(data_size, 0.0) {}

  void AddData(const std::vector<double>& data) {
    if (static_cast<int>(data.size()) != data_size_) return;
    if (!queue_.empty() && data == queue_.back()) return;

    queue_.push_back(data);
    if (queue_.size() > weights_.size()) queue_.pop_front();

    if (queue_.size() < weights_.size()) {
      filtered_ = queue_.back();
      return;
    }
    // weighted convolution (newest sample gets weights_[0])
    std::fill(filtered_.begin(), filtered_.end(), 0.0);
    for (size_t w = 0; w < weights_.size(); ++w) {
      const auto& sample = queue_[queue_.size() - 1 - w];
      for (int d = 0; d < data_size_; ++d) {
        filtered_[d] += weights_[w] * sample[d];
      }
    }
  }

  const std::vector<double>& Filtered() const { return filtered_; }

 private:
  std::vector<double> weights_;
  int data_size_;
  std::deque<std::vector<double>> queue_;
  std::vector<double> filtered_;
};

// ---------------------------------------------------------------------------
// UpperBodyIKSolver — xr_teleoperate style
//
// Solves IK for arms only (14 DOF: 7 left + 7 right).
// Waist joints (12-14) keep SONIC network output values.
//
// Cost function mirrors xr_teleoperate/G1_29_ArmIK:
//   50 * translational_cost + 1 * rotational_cost
//   + 0.02 * regularization_cost + 0.1 * smooth_cost
// ---------------------------------------------------------------------------
class UpperBodyIKOriginSolver {
 public:
  static constexpr int kNumBodies = 30;  // pelvis + 29 joints
  static constexpr int kNumJoints = 29;
  static constexpr int kNumArmJoints = 14; // 7 left arm + 7 right arm
  static constexpr double kEpsilon = 1e-6;

  // xr_teleoperate cost weights
  struct Config {
    int num_iterations = 8;
    double damping = 0.01;
    double translation_weight = 50.0;  // xr_teleoperate: 50
    double rotation_weight = 1.0;      // xr_teleoperate: 1
    double regularization_weight = 0.02; // xr_teleoperate: 0.02
    double smooth_weight = 0.1;          // xr_teleoperate: 0.1
  };

  UpperBodyIKOriginSolver() = default;

  bool Initialize(const std::string& mujoco_xml_path) {
    try {
      fk_ = std::make_unique<RobotFK>(mujoco_xml_path);
    } catch (const std::exception& e) {
      std::cerr << "[UpperBodyIKOrigin] Failed to load model: " << e.what() << '\n';
      return false;
    }

    if (fk_->NumJoints() != kNumBodies) {
      std::cerr << "[UpperBodyIKOrigin] Expected " << kNumBodies << " bodies, got "
                << fk_->NumJoints() << '\n';
      return false;
    }

    ResolveBodyIndices();

    for (int i = 0; i < kNumJoints; ++i) {
      default_q_[i] = default_angles[i];
    }

    // Warm start: init with default arm angles
    for (int j = 0; j < kNumArmJoints; ++j) {
      warm_q_[j] = default_q_[arm_hw_indices_[j]];
    }

    // Smooth filter: weights [0.4, 0.3, 0.2, 0.1] matching xr_teleoperate
    smooth_filter_ = std::make_unique<WeightedMovingFilterOrigin>(
        std::vector<double>{0.4, 0.3, 0.2, 0.1}, kNumArmJoints);

    std::cout << "[UpperBodyIKOrigin] Initialized (xr_teleoperate style). "
              << "Arms only (14 DOF), waist from SONIC. "
              << "left_wrist_body=" << left_wrist_body_idx_
              << " right_wrist_body=" << right_wrist_body_idx_ << '\n';
    return true;
  }

  void OverrideUpperBody(
      std::array<float, G1_NUM_MOTOR>& q_target,
      const std::array<double, 9>& vr_3point_position,
      const std::array<double, 12>& vr_3point_orientation,
      const std::array<double, 4>& /*base_quat*/) {

    // Full q for FK — lower body + waist from SONIC, arms from warm start
    std::array<double, kNumJoints> q_full;
    for (int i = 0; i < kNumJoints; ++i) {
      q_full[i] = static_cast<double>(q_target[i]);
    }

    // Init arm joints from warm start (previous IK solution)
    for (int j = 0; j < kNumArmJoints; ++j) {
      q_full[arm_hw_indices_[j]] = warm_q_[j];
    }

    std::array<double, 3> root_pos = {0.0, 0.0, 0.0};
    std::array<double, 4> root_rot = {1.0, 0.0, 0.0, 0.0};

    // VR targets: left wrist (idx 0) and right wrist (idx 1) only
    // Torso (idx 2) is not used — waist is controlled by SONIC
    struct Target {
      int body_idx;
      std::array<double, 3> pos;
      std::array<double, 4> quat; // w, x, y, z
    };
    std::array<Target, 2> targets = {{
      {left_wrist_body_idx_,
       {vr_3point_position[0], vr_3point_position[1], vr_3point_position[2]},
       {vr_3point_orientation[0], vr_3point_orientation[1],
        vr_3point_orientation[2], vr_3point_orientation[3]}},
      {right_wrist_body_idx_,
       {vr_3point_position[3], vr_3point_position[4], vr_3point_position[5]},
       {vr_3point_orientation[4], vr_3point_orientation[5],
        vr_3point_orientation[6], vr_3point_orientation[7]}}
    }};

    // -- Iterative IK (damped least squares with xr_teleoperate costs) --
    for (int iter = 0; iter < cfg_.num_iterations; ++iter) {
      // FK at current q
      std::array<std::array<double, 3>, kNumBodies> pos_world;
      std::array<std::array<double, 4>, kNumBodies> rot_world;
      fk_->DoFK(pos_world.data(), rot_world.data(), root_pos, root_rot, q_full.data());

      // Error vector: 2 targets × 6 + regularization(14) + smooth(14) = 40
      constexpr int kTaskDim = 2 * 6;       // 2 wrists × (pos3 + ori3)
      constexpr int kRegDim = kNumArmJoints; // 14
      constexpr int kSmoothDim = kNumArmJoints; // 14
      constexpr int kTotalErr = kTaskDim + kRegDim + kSmoothDim;

      Eigen::VectorXd error(kTotalErr);
      Eigen::MatrixXd J(kTotalErr, kNumArmJoints);
      error.setZero();
      J.setZero();

      // --- Task errors: translation + rotation for each wrist ---
      for (int t = 0; t < 2; ++t) {
        const auto& tgt = targets[t];
        int off = t * 6;
        // Translation error (weighted by sqrt(50) ≈ 7.07 so squared cost = 50)
        double w_t = std::sqrt(cfg_.translation_weight);
        error(off + 0) = w_t * (tgt.pos[0] - pos_world[tgt.body_idx][0]);
        error(off + 1) = w_t * (tgt.pos[1] - pos_world[tgt.body_idx][1]);
        error(off + 2) = w_t * (tgt.pos[2] - pos_world[tgt.body_idx][2]);

        // Rotation error (weighted by sqrt(1) = 1)
        double w_r = std::sqrt(cfg_.rotation_weight);
        auto ori_err = QuatError(tgt.quat, rot_world[tgt.body_idx]);
        error(off + 3) = w_r * ori_err[0];
        error(off + 4) = w_r * ori_err[1];
        error(off + 5) = w_r * ori_err[2];

        // Numerical Jacobian w.r.t. arm joints only
        for (int j = 0; j < kNumArmJoints; ++j) {
          int hw = arm_hw_indices_[j];
          constexpr double eps = 1e-4;

          std::array<double, kNumJoints> q_pert = q_full;
          q_pert[hw] += eps;

          std::array<std::array<double, 3>, kNumBodies> pos_p;
          std::array<std::array<double, 4>, kNumBodies> rot_p;
          fk_->DoFK(pos_p.data(), rot_p.data(), root_pos, root_rot, q_pert.data());

          J(off + 0, j) = w_t * (pos_p[tgt.body_idx][0] - pos_world[tgt.body_idx][0]) / eps;
          J(off + 1, j) = w_t * (pos_p[tgt.body_idx][1] - pos_world[tgt.body_idx][1]) / eps;
          J(off + 2, j) = w_t * (pos_p[tgt.body_idx][2] - pos_world[tgt.body_idx][2]) / eps;

          auto ori_p = QuatError(tgt.quat, rot_p[tgt.body_idx]);
          J(off + 3, j) = w_r * (ori_p[0] - ori_err[0]) / eps;
          J(off + 4, j) = w_r * (ori_p[1] - ori_err[1]) / eps;
          J(off + 5, j) = w_r * (ori_p[2] - ori_err[2]) / eps;
        }
      }

      // --- Regularization: pull toward default pose ---
      double w_reg = std::sqrt(cfg_.regularization_weight);
      for (int j = 0; j < kNumArmJoints; ++j) {
        int hw = arm_hw_indices_[j];
        int idx = kTaskDim + j;
        error(idx) = w_reg * (default_q_[hw] - q_full[hw]);
        J(idx, j) = w_reg;  // d(default - q)/dq = -1, but in error form = +1
      }

      // --- Smooth: penalize deviation from previous solution (warm start) ---
      double w_smooth = std::sqrt(cfg_.smooth_weight);
      for (int j = 0; j < kNumArmJoints; ++j) {
        int hw = arm_hw_indices_[j];
        int idx = kTaskDim + kRegDim + j;
        error(idx) = w_smooth * (warm_q_[j] - q_full[hw]);
        J(idx, j) = w_smooth;
      }

      // Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
      Eigen::MatrixXd JJt = J * J.transpose();
      JJt.diagonal().array() += cfg_.damping * cfg_.damping;
      Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(error);

      // Update arm joints only
      for (int j = 0; j < kNumArmJoints; ++j) {
        q_full[arm_hw_indices_[j]] += dq(j);
      }

      ClampArmJoints(q_full);
    }

    // Extract solved arm angles
    std::vector<double> sol_q(kNumArmJoints);
    for (int j = 0; j < kNumArmJoints; ++j) {
      sol_q[j] = q_full[arm_hw_indices_[j]];
    }

    // Apply smooth filter (matching xr_teleoperate)
    smooth_filter_->AddData(sol_q);
    const auto& filtered = smooth_filter_->Filtered();

    // Update warm start for next tick
    for (int j = 0; j < kNumArmJoints; ++j) {
      warm_q_[j] = filtered[j];
    }

    // Write filtered arm joints to q_target (waist stays as SONIC output)
    for (int j = 0; j < kNumArmJoints; ++j) {
      q_target[arm_hw_indices_[j]] = static_cast<float>(filtered[j]);
    }
  }

  Config& GetConfig() { return cfg_; }
  const Config& GetConfig() const { return cfg_; }

 private:
  static std::array<double, 3> QuatError(
      const std::array<double, 4>& q_target,
      const std::array<double, 4>& q_current) {
    auto q_cur_inv = quat_conjugate_d(q_current);
    auto q_err = quat_mul_d(q_target, q_cur_inv);

    if (q_err[0] < 0.0) {
      q_err[0] = -q_err[0]; q_err[1] = -q_err[1];
      q_err[2] = -q_err[2]; q_err[3] = -q_err[3];
    }

    double half_angle = std::acos(std::clamp(q_err[0], -1.0, 1.0));
    double sin_ha = std::sin(half_angle);
    if (sin_ha < kEpsilon) return {0.0, 0.0, 0.0};

    double scale = 2.0 * half_angle / sin_ha;
    return {scale * q_err[1], scale * q_err[2], scale * q_err[3]};
  }

  void ClampArmJoints(std::array<double, kNumJoints>& q) {
    // Arm joint limits matching G1 URDF (only clamp arm indices)
    static constexpr std::array<std::pair<double, double>, kNumArmJoints> limits = {{
      {-2.87, 2.87},  //  0 → hw15 left_shoulder_pitch
      {-1.34, 3.11},  //  1 → hw16 left_shoulder_roll
      {-1.57, 1.57},  //  2 → hw17 left_shoulder_yaw
      {-1.05, 1.4},   //  3 → hw18 left_elbow
      {-1.57, 1.57},  //  4 → hw19 left_wrist_roll
      {-0.52, 0.52},  //  5 → hw20 left_wrist_pitch
      {-0.52, 0.52},  //  6 → hw21 left_wrist_yaw
      {-2.87, 2.87},  //  7 → hw22 right_shoulder_pitch
      {-3.11, 1.34},  //  8 → hw23 right_shoulder_roll
      {-1.57, 1.57},  //  9 → hw24 right_shoulder_yaw
      {-1.05, 1.4},   // 10 → hw25 right_elbow
      {-1.57, 1.57},  // 11 → hw26 right_wrist_roll
      {-0.52, 0.52},  // 12 → hw27 right_wrist_pitch
      {-0.52, 0.52},  // 13 → hw28 right_wrist_yaw
    }};

    for (int j = 0; j < kNumArmJoints; ++j) {
      int hw = arm_hw_indices_[j];
      q[hw] = std::clamp(q[hw], limits[j].first, limits[j].second);
    }
  }

  void ResolveBodyIndices() {
    left_wrist_body_idx_ = fk_->FindBodyIndex("left_wrist_yaw_link");
    right_wrist_body_idx_ = fk_->FindBodyIndex("right_wrist_yaw_link");

    if (left_wrist_body_idx_ < 0 || right_wrist_body_idx_ < 0) {
      std::cerr << "[UpperBodyIKOrigin] WARNING: Could not find wrist bodies. "
                << "left=" << left_wrist_body_idx_
                << " right=" << right_wrist_body_idx_ << '\n';
    }
  }

  // Arm hardware indices only (no waist)
  // Left arm: shoulder_pitch(15), shoulder_roll(16), shoulder_yaw(17),
  //           elbow(18), wrist_roll(19), wrist_pitch(20), wrist_yaw(21)
  // Right arm: shoulder_pitch(22), shoulder_roll(23), shoulder_yaw(24),
  //            elbow(25), wrist_roll(26), wrist_pitch(27), wrist_yaw(28)
  static constexpr std::array<int, kNumArmJoints> arm_hw_indices_ = {
    15, 16, 17, 18, 19, 20, 21,  // left arm
    22, 23, 24, 25, 26, 27, 28   // right arm
  };

  Config cfg_;
  std::unique_ptr<RobotFK> fk_;
  std::array<double, kNumJoints> default_q_{};
  std::array<double, kNumArmJoints> warm_q_{};  // warm start from previous tick
  std::unique_ptr<WeightedMovingFilterOrigin> smooth_filter_;
  int left_wrist_body_idx_ = -1;
  int right_wrist_body_idx_ = -1;
};

#endif // UPPER_BODY_IK_ORIGIN_SOLVER_HPP
