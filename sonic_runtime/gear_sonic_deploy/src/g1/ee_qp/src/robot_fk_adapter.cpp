#include "ee_qp/robot_fk_adapter.hpp"

#include <array>

#include "fk.hpp"
#include "policy_parameters.hpp"

namespace g1::ee_qp {

RobotFkAdapter::RobotFkAdapter(const std::string& mjcf_path)
    : fk_(std::make_unique<RobotFK>(mjcf_path)) {}

RobotFkAdapter::~RobotFkAdapter() = default;

bool RobotFkAdapter::Evaluate(
    const JointVector& q_mujoco_order,
    const Eigen::Quaterniond& pelvis_orientation,
    WristKinematics& output) const {
  std::array<double, kNumJoints> q_isaac_order{};
  for (int hw = 0; hw < kNumJoints; ++hw) {
    q_isaac_order[isaaclab_to_mujoco[hw]] = q_mujoco_order[hw];
  }

  const std::array<double, 3> root_position = {0.0, 0.0, 0.0};
  const Eigen::Quaterniond root = pelvis_orientation.normalized();
  const std::array<double, 4> root_rotation = {
      root.w(), root.x(), root.y(), root.z()};

  std::array<std::array<double, kNumJoints>, 3> left_full{};
  std::array<std::array<double, kNumJoints>, 3> right_full{};
  std::array<double, 3> left_position{};
  std::array<double, 3> right_position{};
  constexpr std::array<double, 3> left_tracking_point = {0.18, -0.025, 0.0};
  constexpr std::array<double, 3> right_tracking_point = {0.18, 0.025, 0.0};

  if (!fk_->ComputeBodyPositionJacobian(
          "left_wrist_yaw_link", left_full, left_position,
          left_tracking_point, root_position, root_rotation, q_isaac_order.data()) ||
      !fk_->ComputeBodyPositionJacobian(
          "right_wrist_yaw_link", right_full, right_position,
          right_tracking_point, root_position, root_rotation, q_isaac_order.data())) {
    return false;
  }

  output.left_position_world = Eigen::Map<Eigen::Vector3d>(left_position.data());
  output.right_position_world = Eigen::Map<Eigen::Vector3d>(right_position.data());
  output.position_jacobian_world.setZero();
  for (int column = 0; column < kNumUpperJoints; ++column) {
    const int hw = kUpperHardwareIndices[column];
    const int isaac = isaaclab_to_mujoco[hw];
    for (int row = 0; row < 3; ++row) {
      output.position_jacobian_world(row, column) = left_full[row][isaac];
      output.position_jacobian_world(row + 3, column) = right_full[row][isaac];
    }
  }
  return true;
}

}  // namespace g1::ee_qp
