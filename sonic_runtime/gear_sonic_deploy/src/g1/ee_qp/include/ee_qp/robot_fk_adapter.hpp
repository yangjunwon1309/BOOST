#pragma once

#include <memory>
#include <string>

#include "ee_qp/ee_kinematics.hpp"

class RobotFK;

namespace g1::ee_qp {

class RobotFkAdapter final : public EeKinematics {
 public:
  explicit RobotFkAdapter(const std::string& mjcf_path);
  ~RobotFkAdapter() override;

  bool Evaluate(
      const JointVector& q_mujoco_order,
      const Eigen::Quaterniond& pelvis_orientation,
      WristKinematics& output) const override;

 private:
  std::unique_ptr<RobotFK> fk_;
};

}  // namespace g1::ee_qp
