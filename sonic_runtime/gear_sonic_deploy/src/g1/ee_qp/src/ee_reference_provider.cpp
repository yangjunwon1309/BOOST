#include "ee_qp/ee_reference_provider.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace g1::ee_qp {

BufferedEeReferenceProvider::BufferedEeReferenceProvider(std::int64_t timeout_us)
    : timeout_us_(timeout_us) {}

void BufferedEeReferenceProvider::Update(const EeReference& reference) {
  std::lock_guard<std::mutex> lock(mutex_);
  reference_ = reference;
}

std::optional<EeReference> BufferedEeReferenceProvider::Latest(
    std::int64_t now_us) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!reference_.valid || now_us < reference_.timestamp_us ||
      now_us - reference_.timestamp_us > timeout_us_) {
    return std::nullopt;
  }
  return reference_;
}

BsplineEeReferenceProvider::BsplineEeReferenceProvider(
    std::int64_t heartbeat_timeout_us,
    std::shared_ptr<EeKinematics> kinematics)
    : kinematics_(std::move(kinematics)),
      heartbeat_timeout_us_(heartbeat_timeout_us) {}

void BsplineEeReferenceProvider::SetLookaheadUs(std::int64_t lookahead_us) {
  std::lock_guard<std::mutex> lock(mutex_);
  lookahead_us_ = std::max<std::int64_t>(0, lookahead_us);
}

void BsplineEeReferenceProvider::UpdatePlan(
    const std::vector<JointVector>& control_points,
    double control_dt_sec,
    std::int64_t t_start_us,
    std::int64_t received_us,
    std::uint64_t plan_id,
    bool feasible,
    ControlMode control_mode) {
  std::lock_guard<std::mutex> lock(mutex_);
  Plan plan;
  plan.control_points = control_points;
  plan.control_dt_sec = control_dt_sec;
  plan.t_start_us = t_start_us;
  plan.received_us = received_us;
  plan.plan_id = plan_id;
  plan.feasible = feasible;
  plan.control_mode = control_mode;
  plan_ = std::move(plan);
}

std::optional<JointVector> BsplineEeReferenceProvider::EvaluateLocked(
    const Plan& plan,
    std::int64_t eval_us,
    double& phase) const {
  const auto n = plan.control_points.size();
  if (n < 4 || plan.control_dt_sec <= 0.0 || !std::isfinite(plan.control_dt_sec)) {
    return std::nullopt;
  }

  const double dt_us = plan.control_dt_sec * 1.0e6;
  const double max_segment = static_cast<double>(n - 4);
  const double raw_s =
      static_cast<double>(eval_us - plan.t_start_us) / std::max(1.0, dt_us);
  const double s = std::clamp(raw_s, 0.0, max_segment);
  const int i = std::clamp(static_cast<int>(std::floor(s)), 0, static_cast<int>(n - 4));
  const double u = std::clamp(s - static_cast<double>(i), 0.0, 1.0);

  const double one_minus_u = 1.0 - u;
  const double b0 = one_minus_u * one_minus_u * one_minus_u / 6.0;
  const double b1 = (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0;
  const double b2 = (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0;
  const double b3 = u * u * u / 6.0;

  JointVector q = JointVector::Zero();
  q += b0 * plan.control_points[i + 0];
  q += b1 * plan.control_points[i + 1];
  q += b2 * plan.control_points[i + 2];
  q += b3 * plan.control_points[i + 3];
  phase = max_segment > 0.0 ? std::clamp(s / max_segment, 0.0, 1.0) : 1.0;
  return q;
}

std::optional<EeReference> BsplineEeReferenceProvider::Latest(
    std::int64_t now_us) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!plan_) return std::nullopt;

  const Plan& plan = *plan_;
  if (heartbeat_timeout_us_ > 0 && now_us - plan.received_us > heartbeat_timeout_us_) {
    return std::nullopt;
  }

  EeReference reference;
  reference.source = ReferenceSource::kCurobo;
  reference.control_mode = plan.control_mode;
  reference.feasible = plan.feasible;
  reference.plan_id = plan.plan_id;
  reference.timestamp_us = now_us;
  reference.reference_pelvis_orientation = Eigen::Quaterniond::Identity();

  if (!plan.feasible) {
    reference.valid = true;
    return reference;
  }

  double phase = 0.0;
  const auto q_ref = EvaluateLocked(plan, now_us + lookahead_us_, phase);
  if (!q_ref || !kinematics_) return std::nullopt;

  WristKinematics wrist;
  if (!kinematics_->Evaluate(*q_ref, Eigen::Quaterniond::Identity(), wrist)) {
    return std::nullopt;
  }
  reference.left_wrist_in_reference_pelvis = wrist.left_position_world;
  reference.right_wrist_in_reference_pelvis = wrist.right_position_world;
  reference.q_ref = *q_ref;
  reference.phase = phase;
  reference.valid = true;
  return reference;
}


DynamicEeReferenceProvider::DynamicEeReferenceProvider(
    std::int64_t sample_timeout_us,
    std::int64_t heartbeat_timeout_us,
    std::shared_ptr<EeKinematics> kinematics)
    : kinematics_(std::move(kinematics)),
      sample_timeout_us_(sample_timeout_us),
      heartbeat_timeout_us_(heartbeat_timeout_us) {}

void DynamicEeReferenceProvider::SetLookaheadUs(std::int64_t lookahead_us) {
  std::lock_guard<std::mutex> lock(mutex_);
  lookahead_us_ = std::max<std::int64_t>(0, lookahead_us);
}

void DynamicEeReferenceProvider::UpdateSample(const EeReference& reference) {
  std::lock_guard<std::mutex> lock(mutex_);
  sample_ = reference;
  active_reference_ = ActiveReference::kSample;
}

void DynamicEeReferenceProvider::UpdatePlan(
    const std::vector<JointVector>& control_points,
    double control_dt_sec,
    std::int64_t t_start_us,
    std::int64_t received_us,
    std::uint64_t plan_id,
    bool feasible,
    ControlMode control_mode,
    ReferenceSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  Plan plan;
  plan.control_points = control_points;
  plan.control_dt_sec = control_dt_sec;
  plan.t_start_us = t_start_us;
  plan.received_us = received_us;
  plan.plan_id = plan_id;
  plan.feasible = feasible;
  plan.control_mode = control_mode;
  plan.source = source;
  plan_ = std::move(plan);
  active_reference_ = ActiveReference::kBspline;
}

std::optional<JointVector> DynamicEeReferenceProvider::EvaluatePlanLocked(
    const Plan& plan,
    std::int64_t eval_us,
    double& phase) const {
  const auto n = plan.control_points.size();
  if (n < 4 || plan.control_dt_sec <= 0.0 || !std::isfinite(plan.control_dt_sec)) {
    return std::nullopt;
  }

  const double dt_us = plan.control_dt_sec * 1.0e6;
  const double max_segment = static_cast<double>(n - 4);
  const double raw_s =
      static_cast<double>(eval_us - plan.t_start_us) / std::max(1.0, dt_us);
  const double s = std::clamp(raw_s, 0.0, max_segment);
  const int i = std::clamp(static_cast<int>(std::floor(s)), 0, static_cast<int>(n - 4));
  const double u = std::clamp(s - static_cast<double>(i), 0.0, 1.0);

  const double one_minus_u = 1.0 - u;
  const double b0 = one_minus_u * one_minus_u * one_minus_u / 6.0;
  const double b1 = (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0;
  const double b2 = (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0;
  const double b3 = u * u * u / 6.0;

  JointVector q = JointVector::Zero();
  q += b0 * plan.control_points[i + 0];
  q += b1 * plan.control_points[i + 1];
  q += b2 * plan.control_points[i + 2];
  q += b3 * plan.control_points[i + 3];
  phase = max_segment > 0.0 ? std::clamp(s / max_segment, 0.0, 1.0) : 1.0;
  return q;
}

std::optional<EeReference> DynamicEeReferenceProvider::Latest(
    std::int64_t now_us) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_reference_ == ActiveReference::kSample) {
    if (!sample_ || !sample_->valid || now_us < sample_->timestamp_us ||
        now_us - sample_->timestamp_us > sample_timeout_us_) {
      return std::nullopt;
    }
    return sample_;
  }

  if (active_reference_ != ActiveReference::kBspline || !plan_) return std::nullopt;
  const Plan& plan = *plan_;
  if (heartbeat_timeout_us_ > 0 && now_us - plan.received_us > heartbeat_timeout_us_) {
    return std::nullopt;
  }

  EeReference reference;
  reference.source = plan.source;
  reference.control_mode = plan.control_mode;
  reference.feasible = plan.feasible;
  reference.plan_id = plan.plan_id;
  reference.timestamp_us = now_us;
  reference.reference_pelvis_orientation = Eigen::Quaterniond::Identity();

  if (!plan.feasible) {
    reference.valid = true;
    return reference;
  }

  double phase = 0.0;
  const auto q_ref = EvaluatePlanLocked(plan, now_us + lookahead_us_, phase);
  if (!q_ref || !kinematics_) return std::nullopt;

  WristKinematics wrist;
  if (!kinematics_->Evaluate(*q_ref, Eigen::Quaterniond::Identity(), wrist)) {
    return std::nullopt;
  }
  reference.left_wrist_in_reference_pelvis = wrist.left_position_world;
  reference.right_wrist_in_reference_pelvis = wrist.right_position_world;
  reference.q_ref = *q_ref;
  reference.phase = phase;
  reference.valid = true;
  return reference;
}

}  // namespace g1::ee_qp
