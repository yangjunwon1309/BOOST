#pragma once

#include "ee_qp/types.hpp"

namespace g1::ee_qp {

struct WristKinematics {
  Eigen::Vector3d left_position_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_position_world = Eigen::Vector3d::Zero();
  ActiveJacobian position_jacobian_world = ActiveJacobian::Zero();
};

class EeKinematics {
 public:
  virtual ~EeKinematics() = default;

  // Root translation is intentionally zero. The supplied orientation rotates
  // both wrist positions and Jacobian columns into the world-oriented frame.
  virtual bool Evaluate(
      const JointVector& q,
      const Eigen::Quaterniond& pelvis_orientation,
      WristKinematics& output) const = 0;
};

}  // namespace g1::ee_qp
