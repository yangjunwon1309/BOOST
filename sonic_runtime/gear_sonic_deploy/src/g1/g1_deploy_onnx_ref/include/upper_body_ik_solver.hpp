#ifndef UPPER_BODY_IK_SOLVER_HPP
#define UPPER_BODY_IK_SOLVER_HPP

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
// WeightedMovingFilter — C++ port of xr_teleoperate's filter
// ---------------------------------------------------------------------------
class WeightedMovingFilter {
 public:
  WeightedMovingFilter(const std::vector<double>& weights, int data_size)
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
// Solves IK for upper body (17 DOF: waist 3 + 7 left arm + 7 right arm).
//
// Cost function mirrors xr_teleoperate/G1_29_ArmIK:
//   50 * translational_cost + 1 * rotational_cost
//   + 0.02 * regularization_cost + 0.1 * smooth_cost
// ---------------------------------------------------------------------------
class UpperBodyIKSolver {
 public:
  static constexpr int kNumJoints = 29;
  static constexpr int kNumArmJoints = 14; // 7 left arm + 7 right arm
  static constexpr int kNumIkJoints = 17;  // waist 3 + arms 14
  static constexpr int kNumTargets = 3;    // left wrist + right wrist + torso
  static constexpr double kEpsilon = 1e-6;

  enum class Mode {
    WaistAndArms,
    ArmsOnly,
  };

  // xr_teleoperate cost weights
  struct Config {
    int num_iterations = 8;
    double damping = 0.01;
    double translation_weight = 50.0;  // xr_teleoperate: 50
    double rotation_weight = 1.0;      // xr_teleoperate: 1
    double torso_translation_weight = 0.0; // use torso orientation for waist
    double torso_rotation_weight = 2.0;
    double regularization_weight = 0.02; // xr_teleoperate: 0.02
    double smooth_weight = 0.1;          // xr_teleoperate: 0.1
    double max_solver_step = 0.03;       // rad per IK iteration
    double max_joint_step_per_tick = 0.006; // rad per 500 Hz command tick
  };

  UpperBodyIKSolver() = default;

  void SetMode(Mode mode) { mode_ = mode; }
  Mode GetMode() const { return mode_; }

  bool Initialize(const std::string& mujoco_xml_path) {
    try {
      fk_ = std::make_unique<RobotFK>(mujoco_xml_path);
    } catch (const std::exception& e) {
      std::cerr << "[UpperBodyIK] Failed to load model: " << e.what() << '\n';
      return false;
    }

    // RobotFK::NumJoints() currently returns the number of parsed FK nodes.
    fk_body_count_ = fk_->NumJoints();
    if (fk_body_count_ <= 0) {
      std::cerr << "[UpperBodyIK] No FK bodies loaded from model\n";
      return false;
    }

    ResolveBodyIndices();
    if (!HasValidTargetBodies()) {
      std::cerr << "[UpperBodyIK] Invalid IK target body index. "
                << "fk_bodies=" << fk_body_count_
                << " left=" << left_wrist_body_idx_
                << " right=" << right_wrist_body_idx_
                << " torso=" << torso_body_idx_ << '\n';
      return false;
    }

    for (int i = 0; i < kNumJoints; ++i) {
      default_q_[i] = default_angles[i];
    }

    // Warm start: init with default upper-body angles
    for (int j = 0; j < kNumIkJoints; ++j) {
      warm_q_[j] = default_q_[ik_hw_indices_[j]];
    }

    // Smooth filter: weights [0.4, 0.3, 0.2, 0.1] matching xr_teleoperate
    smooth_filter_ = std::make_unique<WeightedMovingFilter>(
        std::vector<double>{0.4, 0.3, 0.2, 0.1}, kNumIkJoints);

    std::cout << "[UpperBodyIK] Initialized (xr_teleoperate style). "
              << ModeDescription() << " (" << ActiveIkDof() << " DOF). "
              << "fk_bodies=" << fk_body_count_ << " "
              << "left_wrist_body=" << left_wrist_body_idx_
              << " right_wrist_body=" << right_wrist_body_idx_
              << " torso_body=" << torso_body_idx_ << '\n';
    return true;
  }

  void OverrideUpperBody(
      std::array<float, G1_NUM_MOTOR>& q_target,
      const std::array<double, 9>& vr_3point_position,
      const std::array<double, 12>& vr_3point_orientation,
      const std::array<double, 4>& /*base_quat*/) {
    if (!fk_ || fk_body_count_ <= 0) return;
    if (!AllFinite(vr_3point_position) || !AllFinite(vr_3point_orientation)) return;

    // Full q for FK — lower body from SONIC, upper body from warm start
    std::array<double, kNumJoints> q_full;
    for (int i = 0; i < kNumJoints; ++i) {
      q_full[i] = static_cast<double>(q_target[i]);
    }

    // Init active IK joints from warm start. In arms-only mode, waist remains
    // the SONIC target so we can isolate whether waist IK is causing motion drift.
    for (int j = 0; j < kNumIkJoints; ++j) {
      if (IsActiveIkJoint(j)) {
        q_full[ik_hw_indices_[j]] = warm_q_[j];
      } else {
        warm_q_[j] = q_full[ik_hw_indices_[j]];
      }
    }
    const auto previous_warm_q = warm_q_;

    std::array<double, 3> root_pos = {0.0, 0.0, 0.0};
    std::array<double, 4> root_rot = {1.0, 0.0, 0.0, 0.0};

    // VR targets: left wrist (idx 0), right wrist (idx 1), torso/head (idx 2).
    // The third target drives torso orientation for waist IK; translation weight
    // defaults to 0 because the HMD/head point is not the torso origin.
    struct Target {
      int body_idx;
      std::array<double, 3> pos;
      std::array<double, 4> quat; // w, x, y, z
      double translation_weight;
      double rotation_weight;
    };
    const std::array<double, 3> left_vr_pos = {
      vr_3point_position[0], vr_3point_position[1], vr_3point_position[2]};
    const std::array<double, 3> right_vr_pos = {
      vr_3point_position[3], vr_3point_position[4], vr_3point_position[5]};
    const std::array<double, 3> torso_vr_pos = {
      vr_3point_position[6], vr_3point_position[7], vr_3point_position[8]};
    const auto left_vr_quat = NormalizeQuat(std::array<double, 4>{
      vr_3point_orientation[0], vr_3point_orientation[1],
      vr_3point_orientation[2], vr_3point_orientation[3]});
    const auto right_vr_quat = NormalizeQuat(std::array<double, 4>{
      vr_3point_orientation[4], vr_3point_orientation[5],
      vr_3point_orientation[6], vr_3point_orientation[7]});
    const auto torso_vr_quat = NormalizeQuat(std::array<double, 4>{
      vr_3point_orientation[8], vr_3point_orientation[9],
      vr_3point_orientation[10], vr_3point_orientation[11]});

    std::array<Target, kNumTargets> targets = {{
      {left_wrist_body_idx_,
       left_vr_pos,
       left_vr_quat,
       cfg_.translation_weight,
       cfg_.rotation_weight},
      {right_wrist_body_idx_,
       right_vr_pos,
       right_vr_quat,
       cfg_.translation_weight,
       cfg_.rotation_weight},
      {torso_body_idx_,
       torso_vr_pos,
       torso_vr_quat,
       SolvesWaist() ? cfg_.torso_translation_weight : 0.0,
       SolvesWaist() ? cfg_.torso_rotation_weight : 0.0}
    }};

    std::vector<std::array<double, 3>> pos_world(fk_body_count_);
    std::vector<std::array<double, 4>> rot_world(fk_body_count_);
    std::vector<std::array<double, 3>> pos_p(fk_body_count_);
    std::vector<std::array<double, 4>> rot_p(fk_body_count_);

    // -- Iterative IK (damped least squares with xr_teleoperate costs) --
    for (int iter = 0; iter < cfg_.num_iterations; ++iter) {
      // FK at current q
      fk_->DoFK(pos_world.data(), rot_world.data(), root_pos, root_rot, q_full.data());

      // Error vector: 3 targets × 6 + regularization(17) + smooth(17) = 52
      constexpr int kTaskDim = kNumTargets * 6;
      constexpr int kRegDim = kNumIkJoints;
      constexpr int kSmoothDim = kNumIkJoints;
      constexpr int kTotalErr = kTaskDim + kRegDim + kSmoothDim;

      Eigen::VectorXd error(kTotalErr);
      Eigen::MatrixXd J(kTotalErr, kNumIkJoints);
      error.setZero();
      J.setZero();

      // --- Task errors: translation + rotation for each target ---
      for (int t = 0; t < kNumTargets; ++t) {
        const auto& tgt = targets[t];
        if (tgt.body_idx < 0) continue;
        int off = t * 6;
        const auto tracked_pos = TrackedPointPosition(
            t, pos_world[tgt.body_idx], rot_world[tgt.body_idx]);

        double w_t = std::sqrt(tgt.translation_weight);
        error(off + 0) = w_t * (tgt.pos[0] - tracked_pos[0]);
        error(off + 1) = w_t * (tgt.pos[1] - tracked_pos[1]);
        error(off + 2) = w_t * (tgt.pos[2] - tracked_pos[2]);

        double w_r = std::sqrt(tgt.rotation_weight);
        auto ori_err = QuatError(tgt.quat, rot_world[tgt.body_idx]);
        error(off + 3) = w_r * ori_err[0];
        error(off + 4) = w_r * ori_err[1];
        error(off + 5) = w_r * ori_err[2];

        // Numerical Jacobian w.r.t. waist + arm joints
        for (int j = 0; j < kNumIkJoints; ++j) {
          if (!IsActiveIkJoint(j)) continue;
          int hw = ik_hw_indices_[j];
          constexpr double eps = 1e-4;

          std::array<double, kNumJoints> q_pert = q_full;
          q_pert[hw] += eps;

          fk_->DoFK(pos_p.data(), rot_p.data(), root_pos, root_rot, q_pert.data());
          const auto tracked_pos_p = TrackedPointPosition(
              t, pos_p[tgt.body_idx], rot_p[tgt.body_idx]);

          J(off + 0, j) = w_t * (tracked_pos_p[0] - tracked_pos[0]) / eps;
          J(off + 1, j) = w_t * (tracked_pos_p[1] - tracked_pos[1]) / eps;
          J(off + 2, j) = w_t * (tracked_pos_p[2] - tracked_pos[2]) / eps;

          auto ori_p = QuatError(tgt.quat, rot_p[tgt.body_idx]);
          J(off + 3, j) = w_r * (ori_p[0] - ori_err[0]) / eps;
          J(off + 4, j) = w_r * (ori_p[1] - ori_err[1]) / eps;
          J(off + 5, j) = w_r * (ori_p[2] - ori_err[2]) / eps;
        }
      }

      // --- Regularization: pull toward default pose ---
      double w_reg = std::sqrt(cfg_.regularization_weight);
      for (int j = 0; j < kNumIkJoints; ++j) {
        if (!IsActiveIkJoint(j)) continue;
        int hw = ik_hw_indices_[j];
        int idx = kTaskDim + j;
        error(idx) = w_reg * (default_q_[hw] - q_full[hw]);
        J(idx, j) = w_reg;  // d(default - q)/dq = -1, but in error form = +1
      }

      // --- Smooth: penalize deviation from previous solution (warm start) ---
      double w_smooth = std::sqrt(cfg_.smooth_weight);
      for (int j = 0; j < kNumIkJoints; ++j) {
        if (!IsActiveIkJoint(j)) continue;
        int hw = ik_hw_indices_[j];
        int idx = kTaskDim + kRegDim + j;
        error(idx) = w_smooth * (warm_q_[j] - q_full[hw]);
        J(idx, j) = w_smooth;
      }

      // Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
      Eigen::MatrixXd JJt = J * J.transpose();
      JJt.diagonal().array() += cfg_.damping * cfg_.damping;
      Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(error);

      // Update waist + arm joints only
      for (int j = 0; j < kNumIkJoints; ++j) {
        if (!IsActiveIkJoint(j)) continue;
        double step = std::isfinite(dq(j)) ? dq(j) : 0.0;
        step = std::clamp(step, -cfg_.max_solver_step, cfg_.max_solver_step);
        q_full[ik_hw_indices_[j]] += step;
      }

      ClampUpperBodyJoints(q_full);
    }

    // Extract solved upper-body angles
    std::vector<double> sol_q(kNumIkJoints);
    for (int j = 0; j < kNumIkJoints; ++j) {
      sol_q[j] = q_full[ik_hw_indices_[j]];
    }

    // Apply smooth filter (matching xr_teleoperate)
    smooth_filter_->AddData(sol_q);
    const auto& filtered = smooth_filter_->Filtered();

    // Update warm start for next tick. Real robot control should never jump to
    // a newly solved IK pose in one command tick.
    for (int j = 0; j < kNumIkJoints; ++j) {
      if (!IsActiveIkJoint(j)) {
        warm_q_[j] = q_full[ik_hw_indices_[j]];
        continue;
      }
      const double proposed = std::isfinite(filtered[j]) ? filtered[j] : previous_warm_q[j];
      const double delta = proposed - previous_warm_q[j];
      if (cfg_.max_joint_step_per_tick > 0.0) {
        warm_q_[j] = previous_warm_q[j] + std::clamp(
            delta, -cfg_.max_joint_step_per_tick, cfg_.max_joint_step_per_tick);
      } else {
        warm_q_[j] = proposed;
      }
    }

    // Write filtered waist + arm joints to q_target
    for (int j = 0; j < kNumIkJoints; ++j) {
      if (!IsActiveIkJoint(j)) continue;
      q_target[ik_hw_indices_[j]] = static_cast<float>(warm_q_[j]);
    }
  }

  Config& GetConfig() { return cfg_; }
  const Config& GetConfig() const { return cfg_; }

 private:
  static constexpr int kFirstArmIkIndex = 3;

  bool SolvesWaist() const {
    return mode_ == Mode::WaistAndArms;
  }

  bool IsActiveIkJoint(int ik_idx) const {
    return SolvesWaist() || ik_idx >= kFirstArmIkIndex;
  }

  int ActiveIkDof() const {
    return SolvesWaist() ? kNumIkJoints : kNumArmJoints;
  }

  const char* ModeDescription() const {
    return SolvesWaist() ? "Waist + arms" : "Arms only";
  }

  template <size_t N>
  static bool AllFinite(const std::array<double, N>& values) {
    return std::all_of(values.begin(), values.end(), [](double v) {
      return std::isfinite(v);
    });
  }

  static std::array<double, 4> NormalizeQuat(std::array<double, 4> q) {
    const double n = std::sqrt(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!std::isfinite(n) || n < kEpsilon) {
      return {1.0, 0.0, 0.0, 0.0};
    }
    for (double& v : q) v /= n;
    return q;
  }

  static std::array<double, 3> TrackedPointPosition(
      int target_idx,
      const std::array<double, 3>& body_pos,
      const std::array<double, 4>& body_quat) {
    // The policy/sender's VR point is offset from the tracked body, so the IK
    // translation task tracks that same local point instead of the link origin.
    static constexpr std::array<std::array<double, 3>, kNumTargets> offsets = {{
      {0.18, -0.025, 0.0},
      {0.18,  0.025, 0.0},
      {0.0,   0.0,   0.35}
    }};

    const auto offset_world = quat_rotate_d(body_quat, offsets[target_idx]);
    return {
      body_pos[0] + offset_world[0],
      body_pos[1] + offset_world[1],
      body_pos[2] + offset_world[2],
    };
  }

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

  void ClampUpperBodyJoints(std::array<double, kNumJoints>& q) {
    // Waist + arm joint limits matching G1 MJCF/URDF.
    static constexpr std::array<std::pair<double, double>, kNumIkJoints> limits = {{
      {-2.618, 2.618}, //  0 → hw12 waist_yaw
      {-0.52, 0.52},   //  1 → hw13 waist_roll
      {-0.52, 0.52},   //  2 → hw14 waist_pitch
      {-2.87, 2.87},   //  3 → hw15 left_shoulder_pitch
      {-1.34, 3.11},   //  4 → hw16 left_shoulder_roll
      {-1.57, 1.57},   //  5 → hw17 left_shoulder_yaw
      {-1.05, 1.4},    //  6 → hw18 left_elbow
      {-1.57, 1.57},   //  7 → hw19 left_wrist_roll
      {-0.52, 0.52},   //  8 → hw20 left_wrist_pitch
      {-0.52, 0.52},   //  9 → hw21 left_wrist_yaw
      {-2.87, 2.87},   // 10 → hw22 right_shoulder_pitch
      {-3.11, 1.34},   // 11 → hw23 right_shoulder_roll
      {-1.57, 1.57},   // 12 → hw24 right_shoulder_yaw
      {-1.05, 1.4},    // 13 → hw25 right_elbow
      {-1.57, 1.57},   // 14 → hw26 right_wrist_roll
      {-0.52, 0.52},   // 15 → hw27 right_wrist_pitch
      {-0.52, 0.52},   // 16 → hw28 right_wrist_yaw
    }};

    for (int j = 0; j < kNumIkJoints; ++j) {
      if (!IsActiveIkJoint(j)) continue;
      int hw = ik_hw_indices_[j];
      q[hw] = std::clamp(q[hw], limits[j].first, limits[j].second);
    }
  }

  void ResolveBodyIndices() {
    left_wrist_body_idx_ = fk_->FindBodyIndex("left_wrist_yaw_link");
    right_wrist_body_idx_ = fk_->FindBodyIndex("right_wrist_yaw_link");
    torso_body_idx_ = fk_->FindBodyIndex("torso_link");

    if (left_wrist_body_idx_ < 0 || right_wrist_body_idx_ < 0 || torso_body_idx_ < 0) {
      std::cerr << "[UpperBodyIK] WARNING: Could not find IK target bodies. "
                << "left=" << left_wrist_body_idx_
                << " right=" << right_wrist_body_idx_
                << " torso=" << torso_body_idx_ << '\n';
    }
  }

  bool HasValidBodyIndex(int idx) const {
    return idx >= 0 && idx < fk_body_count_;
  }

  bool HasValidTargetBodies() const {
    return HasValidBodyIndex(left_wrist_body_idx_) &&
           HasValidBodyIndex(right_wrist_body_idx_) &&
           HasValidBodyIndex(torso_body_idx_);
  }

  // Upper-body hardware indices: waist + arms.
  // Waist: yaw(12), roll(13), pitch(14)
  // Left arm: shoulder_pitch(15), shoulder_roll(16), shoulder_yaw(17),
  //           elbow(18), wrist_roll(19), wrist_pitch(20), wrist_yaw(21)
  // Right arm: shoulder_pitch(22), shoulder_roll(23), shoulder_yaw(24),
  //            elbow(25), wrist_roll(26), wrist_pitch(27), wrist_yaw(28)
  static constexpr std::array<int, kNumIkJoints> ik_hw_indices_ = {
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,  // left arm
    22, 23, 24, 25, 26, 27, 28   // right arm
  };

  Config cfg_;
  Mode mode_ = Mode::WaistAndArms;
  std::unique_ptr<RobotFK> fk_;
  std::array<double, kNumJoints> default_q_{};
  std::array<double, kNumIkJoints> warm_q_{};  // warm start from previous tick
  std::unique_ptr<WeightedMovingFilter> smooth_filter_;
  int fk_body_count_ = 0;
  int left_wrist_body_idx_ = -1;
  int right_wrist_body_idx_ = -1;
  int torso_body_idx_ = -1;
};

#endif // UPPER_BODY_IK_SOLVER_HPP
