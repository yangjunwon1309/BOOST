#pragma once

#include <cstdint>
#include <optional>
#include <mutex>
#include <memory>
#include <vector>

#include "ee_qp/ee_kinematics.hpp"
#include "ee_qp/types.hpp"

namespace g1::ee_qp {

class EeReferenceProvider {
 public:
  virtual ~EeReferenceProvider() = default;
  virtual std::optional<EeReference> Latest(std::int64_t now_us) const = 0;
};

class BufferedEeReferenceProvider final : public EeReferenceProvider {
 public:
  explicit BufferedEeReferenceProvider(std::int64_t timeout_us);
  void Update(const EeReference& reference);
  std::optional<EeReference> Latest(std::int64_t now_us) const override;

 private:
  mutable std::mutex mutex_;
  EeReference reference_;
  std::int64_t timeout_us_;
};

class BsplineEeReferenceProvider final : public EeReferenceProvider {
 public:
  explicit BsplineEeReferenceProvider(
      std::int64_t heartbeat_timeout_us,
      std::shared_ptr<EeKinematics> kinematics);

  void SetLookaheadUs(std::int64_t lookahead_us);

  void UpdatePlan(
      const std::vector<JointVector>& control_points,
      double control_dt_sec,
      std::int64_t t_start_us,
      std::int64_t received_us,
      std::uint64_t plan_id,
      bool feasible,
      ControlMode control_mode = ControlMode::kPlanExec);

  std::optional<EeReference> Latest(std::int64_t now_us) const override;

 private:
  struct Plan {
    std::vector<JointVector> control_points;
    double control_dt_sec = 0.02;
    std::int64_t t_start_us = 0;
    std::int64_t received_us = 0;
    std::uint64_t plan_id = 0;
    bool feasible = false;
    ControlMode control_mode = ControlMode::kPlanExec;
  };

  std::optional<JointVector> EvaluateLocked(
      const Plan& plan,
      std::int64_t eval_us,
      double& phase) const;

  mutable std::mutex mutex_;
  std::shared_ptr<EeKinematics> kinematics_;
  std::optional<Plan> plan_;
  std::int64_t heartbeat_timeout_us_;
  std::int64_t lookahead_us_ = 0;
};

}  // namespace g1::ee_qp
