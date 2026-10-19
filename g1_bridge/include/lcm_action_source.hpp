#ifndef G1_DEPLOY_REAL_LCM_ACTION_SOURCE_HPP
#define G1_DEPLOY_REAL_LCM_ACTION_SOURCE_HPP

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "action_source.hpp"
#include "path_direct_protocol.hpp"

#ifdef G1_HAS_LCM
#include <lcm/lcm.h>
#endif

class LcmActionSource final : public ActionSource {
 public:
  LcmActionSource(std::string lcm_url, std::string channel)
      : lcm_url_(std::move(lcm_url)), channel_(std::move(channel)) {}

  bool Start() override {
#ifdef G1_HAS_LCM
    lcm_ = lcm_create(lcm_url_.empty() ? nullptr : lcm_url_.c_str());
    if (!lcm_) {
      std::cerr << "[lcm-action] Failed to create LCM instance";
      if (!lcm_url_.empty()) std::cerr << " for URL: " << lcm_url_;
      std::cerr << '\n';
      return false;
    }

    subscription_ =
        lcm_subscribe(lcm_, channel_.c_str(), &LcmActionSource::OnMessage, this);
    if (!subscription_) {
      std::cerr << "[lcm-action] Failed to subscribe channel: " << channel_ << '\n';
      lcm_destroy(lcm_);
      lcm_ = nullptr;
      return false;
    }

    running_.store(true);
    thread_ = std::thread(&LcmActionSource::HandleLoop, this);
    std::cout << "[lcm-action] Subscribed to PATH arm channel: " << channel_ << '\n';
    return true;
#else
    std::cerr << "[lcm-action] This binary was built without LCM support.\n";
    return false;
#endif
  }

  void Stop() override {
    running_.store(false);
#ifdef G1_HAS_LCM
    if (thread_.joinable()) thread_.join();
    if (lcm_) {
      if (subscription_) {
        lcm_unsubscribe(lcm_, subscription_);
        subscription_ = nullptr;
      }
      lcm_destroy(lcm_);
      lcm_ = nullptr;
    }
#endif
  }

  std::optional<ActionCommand> Poll() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

  std::string Name() const override { return "lcm:" + channel_; }

 private:
#ifdef G1_HAS_LCM
  static void OnMessage(const lcm_recv_buf_t* buffer, const char*, void* user) {
    static_cast<LcmActionSource*>(user)->HandleMessage(buffer);
  }

  void HandleLoop() {
    while (running_.load()) {
      if (!lcm_) return;
      const int result = lcm_handle_timeout(lcm_, 100);
      if (result < 0 && running_.load()) {
        std::cerr << "[lcm-action] lcm_handle_timeout failed.\n";
      }
    }
  }

  void HandleMessage(const lcm_recv_buf_t* buffer) {
    if (!buffer ||
        buffer->data_size != static_cast<int>(sizeof(g1_path::UpperCommandPacket))) {
      ReportBadPacket("unexpected PATH arm packet size");
      return;
    }

    g1_path::UpperCommandPacket packet{};
    std::memcpy(&packet, buffer->data, sizeof(packet));
    constexpr uint16_t kPathGripFlags =
        g1_path::kFlagLeftGrip | g1_path::kFlagRightGrip;
    const auto finite_arm = [](const auto& values) {
      for (const float value : values) {
        if (!std::isfinite(value)) return false;
      }
      return true;
    };
    if (packet.magic != g1_path::kUpperCommandMagic ||
        packet.version != g1_path::kUpperCommandVersion ||
        (packet.flags & ~kPathGripFlags) != 0 || !finite_arm(packet.q_arm) ||
        !finite_arm(packet.dq_arm) || !finite_arm(packet.tau_arm)) {
      ReportBadPacket("invalid PATH arm packet header, flags, or joint values");
      return;
    }

    ActionCommand command;
    for (int i = 0; i < g1_path::kArmCount; ++i) {
      const int motor = g1_path::kArmStart + i;
      command.q_target[motor] = packet.q_arm[i];
      command.dq_target[motor] = packet.dq_arm[i];
      command.tau_ff[motor] = packet.tau_arm[i];
    }
    command.sequence = packet.sequence;
    command.received_at = std::chrono::steady_clock::now();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_ = command;
    }

    const uint64_t count = ++packet_count_;
    if (count == 1) {
      std::cout << "[lcm-action] First PATH arm packet received. seq="
                << packet.sequence << " protocol=PDU1\n";
    } else if (packet.sequence > last_sequence_ + 1) {
      std::cerr << "[lcm-action] Sequence jump: last=" << last_sequence_
                << " current=" << packet.sequence << '\n';
    }
    last_sequence_ = packet.sequence;
  }

  void ReportBadPacket(const char* message) {
    const uint64_t count = ++bad_packet_count_;
    if (count <= 5 || count % 100 == 0) {
      std::cerr << "[lcm-action] " << message << " count=" << count << '\n';
    }
  }
#endif

  std::string lcm_url_;
  std::string channel_;
  mutable std::mutex mutex_;
  std::optional<ActionCommand> latest_;
  std::atomic_bool running_{false};
  std::atomic<uint64_t> packet_count_{0};
  std::atomic<uint64_t> bad_packet_count_{0};
  uint64_t last_sequence_ = 0;

#ifdef G1_HAS_LCM
  lcm_t* lcm_ = nullptr;
  lcm_subscription_t* subscription_ = nullptr;
  std::thread thread_;
#endif
};

#endif
