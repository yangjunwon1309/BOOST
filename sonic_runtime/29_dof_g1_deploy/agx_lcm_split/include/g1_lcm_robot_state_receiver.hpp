#ifndef G1_AGX_LCM_SPLIT_ROBOT_STATE_RECEIVER_HPP
#define G1_AGX_LCM_SPLIT_ROBOT_STATE_RECEIVER_HPP

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <lcm/lcm.h>

#include "g1_lcm_protocol.hpp"

namespace g1_lcm {

struct TimestampedRobotState {
  RobotStatePacket packet;
  std::chrono::steady_clock::time_point received_at;
};

class RobotStateReceiver {
 public:
  RobotStateReceiver(std::string lcm_url = "", std::string channel = kDefaultStateChannel)
      : lcm_url_(std::move(lcm_url)), channel_(std::move(channel)) {}

  bool Start() {
    lcm_ = lcm_create(lcm_url_.empty() ? nullptr : lcm_url_.c_str());
    if (!lcm_) {
      std::cerr << "[g1-lcm-state-receiver] Failed to create LCM";
      if (!lcm_url_.empty()) {
        std::cerr << " for URL: " << lcm_url_;
      }
      std::cerr << '\n';
      return false;
    }

    subscription_ = lcm_subscribe(lcm_, channel_.c_str(), &RobotStateReceiver::OnMessage, this);
    if (!subscription_) {
      std::cerr << "[g1-lcm-state-receiver] Failed to subscribe channel: " << channel_ << '\n';
      lcm_destroy(lcm_);
      lcm_ = nullptr;
      return false;
    }

    running_.store(true);
    thread_ = std::thread(&RobotStateReceiver::HandleLoop, this);
    std::cout << "[g1-lcm-state-receiver] Subscribed channel: " << channel_ << '\n';
    return true;
  }

  void Stop() {
    running_.store(false);
    if (thread_.joinable()) {
      thread_.join();
    }
    if (lcm_) {
      if (subscription_) {
        lcm_unsubscribe(lcm_, subscription_);
        subscription_ = nullptr;
      }
      lcm_destroy(lcm_);
      lcm_ = nullptr;
    }
  }

  ~RobotStateReceiver() { Stop(); }

  std::optional<TimestampedRobotState> Latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

  uint64_t PacketCount() const { return packet_count_.load(); }

 private:
  static void OnMessage(const lcm_recv_buf_t* buffer, const char*, void* user) {
    static_cast<RobotStateReceiver*>(user)->HandleMessage(buffer);
  }

  void HandleLoop() {
    while (running_.load()) {
      if (!lcm_) {
        return;
      }
      const int result = lcm_handle_timeout(lcm_, 100);
      if (result < 0 && running_.load()) {
        std::cerr << "[g1-lcm-state-receiver] lcm_handle_timeout failed.\n";
      }
    }
  }

  void HandleMessage(const lcm_recv_buf_t* buffer) {
    if (!buffer || buffer->data_size != sizeof(RobotStatePacket)) {
      ++bad_packet_count_;
      if (bad_packet_count_ <= 5 || bad_packet_count_ % 100 == 0) {
        std::cerr << "[g1-lcm-state-receiver] Bad packet size count="
                  << bad_packet_count_.load()
                  << " size=" << (buffer ? buffer->data_size : 0)
                  << " expected=" << sizeof(RobotStatePacket) << '\n';
      }
      return;
    }

    RobotStatePacket packet;
    std::memcpy(&packet, buffer->data, sizeof(packet));
    if (packet.magic != kRobotStateMagic || packet.version != kProtocolVersion) {
      ++bad_packet_count_;
      if (bad_packet_count_ <= 5 || bad_packet_count_ % 100 == 0) {
        std::cerr << "[g1-lcm-state-receiver] Bad packet header count="
                  << bad_packet_count_.load() << '\n';
      }
      return;
    }

    TimestampedRobotState state;
    state.packet = packet;
    state.received_at = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_ = state;
    }

    const uint64_t count = ++packet_count_;
    if (count == 1) {
      std::cout << "[g1-lcm-state-receiver] First robot state. seq="
                << packet.sequence << " mode=" << static_cast<int>(packet.mode_machine)
                << '\n';
    } else if (packet.sequence != last_sequence_ + 1 && packet.sequence > last_sequence_ + 1) {
      std::cerr << "[g1-lcm-state-receiver] Sequence jump: last=" << last_sequence_
                << " current=" << packet.sequence << '\n';
    }
    last_sequence_ = packet.sequence;
  }

  std::string lcm_url_;
  std::string channel_;
  mutable std::mutex mutex_;
  std::optional<TimestampedRobotState> latest_;
  std::atomic_bool running_{false};
  std::atomic<uint64_t> packet_count_{0};
  std::atomic<uint64_t> bad_packet_count_{0};
  uint64_t last_sequence_ = 0;
  lcm_t* lcm_ = nullptr;
  lcm_subscription_t* subscription_ = nullptr;
  std::thread thread_;
};

}  // namespace g1_lcm

#endif
