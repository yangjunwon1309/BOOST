#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include "g1_lcm_robot_state_receiver.hpp"

namespace {
std::atomic_bool should_stop{false};

void SignalHandler(int) {
  should_stop.store(true);
}
}  // namespace

int main(int argc, char** argv) {
  std::string lcm_url;
  if (argc == 2) {
    lcm_url = argv[1];
  } else if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [lcm_url]\n";
    return 1;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  g1_lcm::RobotStateReceiver receiver(lcm_url);
  if (!receiver.Start()) {
    return 1;
  }

  while (!should_stop.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto latest = receiver.Latest();
    if (!latest) {
      std::cout << "[state-listener] Waiting for G1_ROBOT_STATE...\n";
      continue;
    }
    const auto& packet = latest->packet;
    const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - latest->received_at).count();
    std::cout << "[state-listener] packets=" << receiver.PacketCount()
              << " seq=" << packet.sequence
              << " age_ms=" << age_ms
              << " mode=" << static_cast<int>(packet.mode_machine)
              << " q0=" << packet.q[0]
              << " dq0=" << packet.dq[0]
              << " base_quat_w=" << packet.base_quat[0]
              << '\n';
  }

  receiver.Stop();
  return 0;
}
