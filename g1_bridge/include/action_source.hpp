#ifndef G1_DEPLOY_REAL_ACTION_SOURCE_HPP
#define G1_DEPLOY_REAL_ACTION_SOURCE_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "robot_parameters.hpp"


struct ActionCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
  bool left_grip = false;
  bool right_grip = false;
  uint64_t sequence = 0;
  std::chrono::steady_clock::time_point received_at = std::chrono::steady_clock::now();
};

class ActionSource {
 public:
  virtual ~ActionSource() = default;

  virtual bool Start() { return true; }
  virtual void Stop() {}
  virtual std::optional<ActionCommand> Poll() = 0;
  virtual std::string Name() const = 0;
};

#endif
