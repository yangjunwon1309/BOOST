#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "g1_lcm_action_sender.hpp"

namespace {
std::atomic_bool should_stop{false};

void SignalHandler(int) {
  should_stop.store(true);
}
}  // namespace

int main(int argc, char** argv) {
  std::string lcm_url;
  int rate_hz = 50;
  if (argc >= 2) {
    lcm_url = argv[1];
  }
  if (argc >= 3) {
    rate_hz = std::atoi(argv[2]);
  }
  if (argc > 3 || rate_hz <= 0 || rate_hz > 500) {
    std::cerr << "Usage: " << argv[0] << " [lcm_url] [rate_hz]\n";
    return 1;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  g1_lcm::ActionSender sender(lcm_url);
  if (!sender.Start()) {
    return 1;
  }

  std::array<float, g1_lcm::kNumMotors> zero_action = {};
  const auto period = std::chrono::microseconds(1000000 / rate_hz);
  auto next_tick = std::chrono::steady_clock::now();
  uint64_t count = 0;

  while (!should_stop.load()) {
    next_tick += period;
    if (!sender.Publish(zero_action, false)) {
      return 1;
    }
    ++count;
    if (count == 1 || count % static_cast<uint64_t>(rate_hz) == 0) {
      std::cout << "[action-smoke-sender] Published zero action count=" << count << '\n';
    }
    std::this_thread::sleep_until(next_tick);
  }

  sender.Publish(zero_action, true);
  sender.Stop();
  return 0;
}
