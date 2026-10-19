#ifndef G1_DEPLOY_REAL_PATH_DIRECT_PROTOCOL_HPP
#define G1_DEPLOY_REAL_PATH_DIRECT_PROTOCOL_HPP

#include <array>
#include <cstdint>

namespace g1_path {

constexpr int kArmStart = 15;
constexpr int kArmCount = 14;
constexpr const char* kDefaultUpperCommandChannel = "G1_PATH_UPPER_COMMAND";
constexpr uint32_t kUpperCommandMagic = 0x31554450U;  // "PDU1" little-endian.
constexpr uint16_t kUpperCommandVersion = 1;
constexpr uint16_t kFlagLeftGrip = 1U << 1;
constexpr uint16_t kFlagRightGrip = 1U << 2;

#pragma pack(push, 1)
struct UpperCommandPacket {
  uint32_t magic = kUpperCommandMagic;
  uint16_t version = kUpperCommandVersion;
  uint16_t flags = 0;
  uint64_t sequence = 0;
  int64_t send_time_us = 0;
  std::array<float, kArmCount> q_arm = {};
  std::array<float, kArmCount> dq_arm = {};
  std::array<float, kArmCount> tau_arm = {};
};
#pragma pack(pop)

static_assert(sizeof(UpperCommandPacket) == 192,
              "UpperCommandPacket size changed");

}  // namespace g1_path

#endif
