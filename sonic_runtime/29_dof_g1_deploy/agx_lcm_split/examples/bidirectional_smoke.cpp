#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "g1_lcm_action_sender.hpp"
#include "g1_lcm_robot_state_receiver.hpp"

namespace {
std::atomic_bool should_stop{false};

void SignalHandler(int) {
  should_stop.store(true);
}
}  // namespace

int main(int argc, char** argv) {
  std::string lcm_url;
  int action_rate_hz = 50;
  int duration_sec = 0;

  if (argc >= 2) {
    lcm_url = argv[1];
  }
  if (argc >= 3) {
    action_rate_hz = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    duration_sec = std::atoi(argv[3]);
  }
  if (argc > 4 || action_rate_hz <= 0 || action_rate_hz > 500 || duration_sec < 0) {
    std::cerr << "Usage: " << argv[0] << " [lcm_url] [action_rate_hz] [duration_sec]\n"
              << "  duration_sec=0 means run until Ctrl-C.\n";
    return 1;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  g1_lcm::RobotStateReceiver receiver(lcm_url);
  if (!receiver.Start()) {
    return 1;
  }

  g1_lcm::ActionSender sender(lcm_url);
  if (!sender.Start()) {
    return 1;
  }

  std::array<float, g1_lcm::kNumMotors> zero_action = {};
  const auto action_period = std::chrono::microseconds(1000000 / action_rate_hz);
  auto next_action_tick = std::chrono::steady_clock::now();
  auto next_report_tick = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  const auto start_time = std::chrono::steady_clock::now();
  uint64_t action_count = 0;

  std::cout << "[bidirectional-smoke] Running. action_rate_hz=" << action_rate_hz
            << " duration_sec=" << duration_sec << '\n';

  while (!should_stop.load()) {
    const auto now = std::chrono::steady_clock::now();
    if (duration_sec > 0 && now - start_time >= std::chrono::seconds(duration_sec)) {
      break;
    }

    if (now >= next_action_tick) {
      next_action_tick += action_period;
      if (!sender.Publish(zero_action, false)) {
        return 1;
      }
      ++action_count;
    }

    if (now >= next_report_tick) {
      next_report_tick += std::chrono::seconds(1);
      auto latest = receiver.Latest();
      if (!latest) {
        std::cout << "[bidirectional-smoke] sent_actions=" << action_count
                  << " received_states=0 waiting_for_state\n";
      } else {
        const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - latest->received_at).count();
        const auto& packet = latest->packet;
        std::cout << "[bidirectional-smoke] sent_actions=" << action_count
                  << " received_states=" << receiver.PacketCount()
                  << " last_state_seq=" << packet.sequence
                  << " state_age_ms=" << age_ms
                  << " mode=" << static_cast<int>(packet.mode_machine)
                  << " q0=" << packet.q[0]
                  << '\n';
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  sender.Publish(zero_action, true);
  sender.Stop();
  receiver.Stop();
  std::cout << "[bidirectional-smoke] Done. sent_actions=" << action_count
            << " received_states=" << receiver.PacketCount() << '\n';
  return 0;
}
