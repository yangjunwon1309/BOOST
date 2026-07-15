#ifndef G1_AGX_LCM_SPLIT_ACTION_SENDER_HPP
#define G1_AGX_LCM_SPLIT_ACTION_SENDER_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include <lcm/lcm.h>

#include "g1_lcm_protocol.hpp"

namespace g1_lcm {

class ActionSender {
 public:
  ActionSender(std::string lcm_url = "", std::string channel = kDefaultActionChannel)
      : lcm_url_(std::move(lcm_url)), channel_(std::move(channel)) {}

  bool Start() {
    lcm_ = lcm_create(lcm_url_.empty() ? nullptr : lcm_url_.c_str());
    if (!lcm_) {
      std::cerr << "[g1-lcm-action-sender] Failed to create LCM";
      if (!lcm_url_.empty()) {
        std::cerr << " for URL: " << lcm_url_;
      }
      std::cerr << '\n';
      return false;
    }
    std::cout << "[g1-lcm-action-sender] Publishing channel: " << channel_ << '\n';
    return true;
  }

  void Stop() {
    if (lcm_) {
      lcm_destroy(lcm_);
      lcm_ = nullptr;
    }
  }

  ~ActionSender() { Stop(); }

  bool Publish(const std::array<float, kNumMotors>& action, bool emergency_stop = false) {
    if (!lcm_) {
      return false;
    }

    PolicyActionPacket packet;
    packet.sequence = sequence_++;
    packet.send_time_us = NowUnixMicros();
    packet.flags = emergency_stop ? kActionFlagEmergencyStop : 0;
    packet.action = action;

    const int result = lcm_publish(lcm_, channel_.c_str(), &packet, sizeof(packet));
    if (result != 0) {
      std::cerr << "[g1-lcm-action-sender] Publish failed. seq=" << packet.sequence << '\n';
      return false;
    }
    return true;
  }

 private:
  static int64_t NowUnixMicros() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  }

  std::string lcm_url_;
  std::string channel_;
  uint64_t sequence_ = 0;
  lcm_t* lcm_ = nullptr;
};

}  // namespace g1_lcm

#endif
