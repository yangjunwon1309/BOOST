#ifndef G1_DEPLOY_REAL_ROBOT_PARAMETERS_HPP
#define G1_DEPLOY_REAL_ROBOT_PARAMETERS_HPP

#include <array>
#include <cstdint>
#include <string>

static const std::string HG_ARM_SDK_TOPIC = "rt/arm_sdk";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

constexpr int G1_NUM_MOTOR = 29;
constexpr int G1_ARM_START_INDEX = 15;
constexpr int G1_ARM_MOTOR_COUNT = 14;
// LowCmd_ has extra non-actuated entries. This slot is Unitree's arm-SDK
// ownership marker and is intentionally outside the 29 body actuators.
constexpr int G1_ARM_SDK_ENABLE_INDEX = 29;

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

// Ported from g1_ws/include/policy_parameters.hpp. The bridge uses these
// per-joint gains when it builds rt/arm_sdk commands.
constexpr double ARMATURE_5020 = 0.003609725;
constexpr double ARMATURE_7520_14 = 0.010177520;
constexpr double ARMATURE_7520_22 = 0.025101925;
constexpr double ARMATURE_4010 = 0.00425;
constexpr double NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535;
constexpr double DAMPING_RATIO = 2.0;

constexpr double STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
constexpr double DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
constexpr double DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
constexpr double DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

constexpr float ToMotorGain(double gain) { return static_cast<float>(gain); }

constexpr std::array<float, G1_NUM_MOTOR> kps = {
    ToMotorGain(STIFFNESS_7520_22), ToMotorGain(STIFFNESS_7520_22),
    ToMotorGain(STIFFNESS_7520_14), ToMotorGain(STIFFNESS_7520_22),
    ToMotorGain(2.0 * STIFFNESS_5020), ToMotorGain(2.0 * STIFFNESS_5020),
    ToMotorGain(STIFFNESS_7520_22), ToMotorGain(STIFFNESS_7520_22),
    ToMotorGain(STIFFNESS_7520_14), ToMotorGain(STIFFNESS_7520_22),
    ToMotorGain(2.0 * STIFFNESS_5020), ToMotorGain(2.0 * STIFFNESS_5020),
    ToMotorGain(STIFFNESS_7520_14), ToMotorGain(2.0 * STIFFNESS_5020),
    ToMotorGain(2.0 * STIFFNESS_5020), ToMotorGain(STIFFNESS_5020),
    ToMotorGain(STIFFNESS_5020), ToMotorGain(STIFFNESS_5020),
    ToMotorGain(STIFFNESS_5020), ToMotorGain(STIFFNESS_5020),
    ToMotorGain(STIFFNESS_4010), ToMotorGain(STIFFNESS_4010),
    ToMotorGain(STIFFNESS_5020), ToMotorGain(STIFFNESS_5020),
    ToMotorGain(STIFFNESS_5020), ToMotorGain(STIFFNESS_5020),
    ToMotorGain(STIFFNESS_5020), ToMotorGain(STIFFNESS_4010),
    ToMotorGain(STIFFNESS_4010),
};

constexpr std::array<float, G1_NUM_MOTOR> kds = {
    ToMotorGain(DAMPING_7520_22), ToMotorGain(DAMPING_7520_22),
    ToMotorGain(DAMPING_7520_14), ToMotorGain(DAMPING_7520_22),
    ToMotorGain(2.0 * DAMPING_5020), ToMotorGain(2.0 * DAMPING_5020),
    ToMotorGain(DAMPING_7520_22), ToMotorGain(DAMPING_7520_22),
    ToMotorGain(DAMPING_7520_14), ToMotorGain(DAMPING_7520_22),
    ToMotorGain(2.0 * DAMPING_5020), ToMotorGain(2.0 * DAMPING_5020),
    ToMotorGain(DAMPING_7520_14), ToMotorGain(2.0 * DAMPING_5020),
    ToMotorGain(2.0 * DAMPING_5020), ToMotorGain(DAMPING_5020),
    ToMotorGain(DAMPING_5020), ToMotorGain(DAMPING_5020),
    ToMotorGain(DAMPING_5020), ToMotorGain(DAMPING_5020),
    ToMotorGain(DAMPING_4010), ToMotorGain(DAMPING_4010),
    ToMotorGain(DAMPING_5020), ToMotorGain(DAMPING_5020),
    ToMotorGain(DAMPING_5020), ToMotorGain(DAMPING_5020),
    ToMotorGain(DAMPING_5020), ToMotorGain(DAMPING_4010),
    ToMotorGain(DAMPING_4010),
};

enum class Mode : uint8_t {
  PR = 0,
  AB = 1,
};

enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,
  WaistA = 13,
  WaistPitch = 14,
  WaistB = 14,
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,
  LeftWristYaw = 21,
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,
  RightWristYaw = 28,
};

inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t crc32 = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;

  for (uint32_t i = 0; i < len; ++i) {
    xbit = 1U << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (crc32 & 0x80000000) {
        crc32 <<= 1;
        crc32 ^= polynomial;
      } else {
        crc32 <<= 1;
      }
      if (data & xbit) {
        crc32 ^= polynomial;
      }
      xbit >>= 1;
    }
  }

  return crc32;
}

#endif
