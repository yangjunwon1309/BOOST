#ifndef G1_DEPLOY_REAL_LCM_PROTOCOL_HPP
#define G1_DEPLOY_REAL_LCM_PROTOCOL_HPP

#include <array>
#include <cstdint>

#include "robot_parameters.hpp"

namespace g1_lcm {

constexpr const char* kDefaultActionChannel = "G1_POLICY_ACTION";
constexpr const char* kDefaultStateChannel = "G1_ROBOT_STATE";
constexpr uint32_t kPolicyActionMagic = 0x31414347U;  // "GCA1" little-endian.
constexpr uint32_t kRobotStateMagic = 0x31545347U;    // "GST1" little-endian.
constexpr uint16_t kProtocolVersion = 1;
constexpr uint16_t kActionFlagEmergencyStop = 1U << 0;
constexpr uint16_t kActionFlagLeftGrip      = 1U << 1;  // left hand close
constexpr uint16_t kActionFlagRightGrip     = 1U << 2;  // right hand close
constexpr uint16_t kStateFlagHasTorsoImu = 1U << 0;


#pragma pack(push, 1)
struct PolicyActionPacket {
  uint32_t magic = kPolicyActionMagic;
  uint16_t version = kProtocolVersion;
  uint16_t flags = 0;
  uint64_t sequence = 0;
  int64_t send_time_us = 0;
  std::array<float, G1_NUM_MOTOR> action = {};
};

struct RobotStatePacket {
  uint32_t magic = kRobotStateMagic;
  uint16_t version = kProtocolVersion;
  uint16_t flags = 0;
  uint64_t sequence = 0;
  uint64_t lowstate_sequence = 0;
  int64_t send_time_us = 0;
  uint8_t mode_machine = 0;
  std::array<float, 4> base_quat = {};
  std::array<float, 3> base_gyro = {};
  std::array<float, 3> base_accel = {};
  std::array<float, 4> torso_quat = {};
  std::array<float, 3> torso_gyro = {};
  std::array<float, 3> torso_accel = {};
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
  std::array<float, G1_NUM_MOTOR> tau_est = {};

};
#pragma pack(pop)

}  // namespace g1_lcm

#endif
