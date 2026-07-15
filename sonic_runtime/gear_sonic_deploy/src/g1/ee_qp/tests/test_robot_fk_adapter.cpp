#include <algorithm>
#include <cassert>
#include <cmath>

#include "ee_qp/robot_fk_adapter.hpp"

int main() {
  g1::ee_qp::RobotFkAdapter adapter(G1_EE_QP_MODEL_PATH);
  g1::ee_qp::JointVector q = g1::ee_qp::JointVector::Zero();
  for (int i = 0; i < g1::ee_qp::kNumUpperJoints; ++i) {
    q[g1::ee_qp::kUpperHardwareIndices[i]] = 0.03 * (i - 8);
  }
  q[12] = 0.12;
  q[13] = -0.08;
  q[14] = 0.05;

  const Eigen::Quaterniond root(
      Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-0.12, Eigen::Vector3d::UnitY()));
  g1::ee_qp::WristKinematics nominal;
  assert(adapter.Evaluate(q, root, nominal));

  constexpr double epsilon = 1e-6;
  double max_error = 0.0;
  for (int column = 0; column < g1::ee_qp::kNumUpperJoints; ++column) {
    const int hw = g1::ee_qp::kUpperHardwareIndices[column];
    auto q_plus = q;
    auto q_minus = q;
    q_plus[hw] += epsilon;
    q_minus[hw] -= epsilon;

    g1::ee_qp::WristKinematics plus;
    g1::ee_qp::WristKinematics minus;
    assert(adapter.Evaluate(q_plus, root, plus));
    assert(adapter.Evaluate(q_minus, root, minus));

    Eigen::Matrix<double, 6, 1> finite_difference;
    finite_difference.head<3>() =
        (plus.left_position_world - minus.left_position_world) / (2.0 * epsilon);
    finite_difference.tail<3>() =
        (plus.right_position_world - minus.right_position_world) / (2.0 * epsilon);
    max_error = std::max(
        max_error,
        (finite_difference - nominal.position_jacobian_world.col(column))
            .cwiseAbs().maxCoeff());
  }

  assert(std::isfinite(max_error));
  assert(max_error < 1e-5);
  return 0;
}
