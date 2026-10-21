#ifndef G1_DEPLOY_REAL_DEX1_HANDS_HPP
#define G1_DEPLOY_REAL_DEX1_HANDS_HPP

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace dex1 {

using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree_go::msg::dds_::MotorCmds_;
using unitree_go::msg::dds_::MotorStates_;

// Dex1-1 has one M4010 actuator per hand. q=0 is the calibrated closed stop.
// Although the SDK exposes a 7 rad logical range, Unitree's XR teleoperation
// controller limits physical motion to 5.40 rad (roughly the 9 cm rail stroke)
// and ramps there from measured state at 0.18 rad per 200 Hz tick.
constexpr int kMotorCount = 1;
constexpr float kClosedPositionRad = 0.0F;
constexpr float kOpenPositionRad = 5.40F;
constexpr float kGripKp = 5.0F;
constexpr float kReleaseKp = 5.0F;
constexpr float kKd = 0.05F;
constexpr float kMaxCommandStepRad = 0.18F;
constexpr auto kKeepalivePeriod = std::chrono::milliseconds(5);  // 200 Hz
constexpr auto kShutdownPeriod = std::chrono::milliseconds(200);
constexpr auto kInitializationTimeout = std::chrono::seconds(5);
constexpr auto kStateStaleTimeout = std::chrono::milliseconds(100);
constexpr auto kResponseCheckDelay = std::chrono::milliseconds(300);
constexpr float kResponseErrorThresholdRad = 0.25F;
constexpr float kResponseMovementThresholdRad = 0.03F;
constexpr float kResponseVelocityThresholdRadS = 0.02F;
constexpr float kResponseTorqueThresholdNm = 0.10F;

static const std::string kLeftCommandTopic = "rt/dex1/left/cmd";
static const std::string kRightCommandTopic = "rt/dex1/right/cmd";
static const std::string kLeftStateTopic = "rt/dex1/left/state";
static const std::string kRightStateTopic = "rt/dex1/right/state";

class Dex1Hands {
 public:
  Dex1Hands() {
    SetReleaseCommand(left_);
    SetReleaseCommand(right_);
  }

  Dex1Hands(const Dex1Hands&) = delete;
  Dex1Hands& operator=(const Dex1Hands&) = delete;

  ~Dex1Hands() { Shutdown(); }

  bool Initialize() {
    if (initialized_.exchange(true)) return true;
    shutting_down_.store(false);
    InitializeHand(left_, true, kLeftCommandTopic, kLeftStateTopic);
    InitializeHand(right_, false, kRightCommandTopic, kRightStateTopic);

    const auto deadline = std::chrono::steady_clock::now() + kInitializationTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (HasLeftState() && HasRightState()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!HasLeftState() || !HasRightState()) {
      std::cerr << "[dex1] Initialization failed: no real state from "
                << (!HasLeftState() && !HasRightState() ? "left and right hands"
                    : (!HasLeftState() ? "left hand" : "right hand"))
                << " within "
                << std::chrono::duration_cast<std::chrono::seconds>(
                       kInitializationTimeout).count()
                << " s. Dex1 writer was not started.\n";
      left_.subscriber.reset();
      right_.subscriber.reset();
      left_.publisher.reset();
      right_.publisher.reset();
      initialized_.store(false);
      return false;
    }

    writer_running_.store(true);
    writer_thread_ = std::thread(&Dex1Hands::WriterLoop, this);

    std::cout << "[dex1] Initialized left/right Dex1-1 DDS endpoints.\n"
              << "[dex1] Command topics: " << kLeftCommandTopic << ", "
              << kRightCommandTopic << '\n'
              << "[dex1] State topics: " << kLeftStateTopic << ", "
              << kRightStateTopic << '\n'
              << "[dex1] One motor per hand; q=0 closed, q=5.4 open; "
              << "dedicated 200 Hz writer, 0.18 rad/tick ramp, "
              << "3-tap smoothing, state timeout="
              << kStateStaleTimeout.count() << " ms.\n";
    return true;
  }

  void SetGripCommand(bool is_left, bool do_close) {
    SetPositionCommand(
        is_left ? left_ : right_,
        do_close ? kClosedPositionRad : kOpenPositionRad,
        do_close ? kGripKp : kReleaseKp);
  }

  void ReleaseAll() {
    SetPositionCommand(left_, kOpenPositionRad, kReleaseKp);
    SetPositionCommand(right_, kOpenPositionRad, kReleaseKp);
  }

  void StopAll() { ReleaseAll(); }

  void Shutdown() {
    if (!initialized_.load() || shutting_down_.exchange(true)) return;

    writer_running_.store(false);
    if (writer_thread_.joinable()) writer_thread_.join();

    ReleaseAll();
    constexpr int kShutdownWrites =
        static_cast<int>(kShutdownPeriod / kKeepalivePeriod);
    for (int i = 0; i < kShutdownWrites; ++i) {
      const auto next_tick = std::chrono::steady_clock::now() + kKeepalivePeriod;
      WriteHand(left_, "left", true);
      WriteHand(right_, "right", true);
      std::this_thread::sleep_until(next_tick);
    }
    initialized_.store(false);
    std::cout << "[dex1] Shutdown complete: OPEN published at 200 Hz for 200 ms.\n";
  }

  bool HasLeftState() const { return left_.state_count.load() > 0; }
  bool HasRightState() const { return right_.state_count.load() > 0; }
  uint64_t LeftStateCount() const { return left_.state_count.load(); }
  uint64_t RightStateCount() const { return right_.state_count.load(); }

 private:
  struct HandContext {
    mutable std::mutex mutex;
    MotorCmds_ command;
    ChannelPublisherPtr<MotorCmds_> publisher;
    ChannelSubscriberPtr<MotorStates_> subscriber;
    std::atomic<uint64_t> state_count{0};
    float desired_position{kOpenPositionRad};
    float desired_kp{kReleaseKp};
    float measured_position{kClosedPositionRad};
    float measured_velocity{0.0F};
    float measured_tau_est{0.0F};
    float last_sent_position{kClosedPositionRad};
    std::array<float, 3> command_history{};
    int command_history_count{0};
    bool has_state{false};
    bool stale_reported{false};
    bool response_warning{false};
    std::chrono::steady_clock::time_point last_state_time{};
    std::chrono::steady_clock::time_point target_changed_at{};
    float target_start_position{kClosedPositionRad};
    std::chrono::steady_clock::time_point last_progress_time{};
    float last_progress_position{kClosedPositionRad};
    uint64_t diagnostic_state_count{0};
  };

  static void SizeCommand(MotorCmds_& command) {
    command.cmds().resize(kMotorCount);
  }

  static void FillCommand(MotorCmds_& command, float position_rad, float kp) {
    SizeCommand(command);
    auto& motor = command.cmds()[0];
    // Match Unitree's official Dex1-1 DDS test publisher.  The current
    // serial2dds service derives FOC from a fresh command, but mode=1 keeps
    // this command compatible with service revisions that honour the field.
    motor.mode() = 1;
    motor.q() = position_rad;
    motor.dq() = 0.0F;
    motor.tau() = 0.0F;
    motor.kp() = kp;
    motor.kd() = kKd;
    motor.reserve().fill(0);
  }

  static void SetReleaseCommand(HandContext& hand) {
    std::lock_guard<std::mutex> lock(hand.mutex);
    // Start from the calibrated closed state and let the same bounded path
    // open the gripper after state feedback arrives.
    FillCommand(hand.command, kClosedPositionRad, kReleaseKp);
    hand.desired_position = kOpenPositionRad;
    hand.desired_kp = kReleaseKp;
    hand.last_sent_position = kClosedPositionRad;
    hand.command_history.fill(kClosedPositionRad);
    hand.command_history_count = 0;
    hand.target_changed_at = std::chrono::steady_clock::now();
    hand.target_start_position = kClosedPositionRad;
    hand.last_progress_time = hand.target_changed_at;
    hand.last_progress_position = kClosedPositionRad;
  }

  static void SetPositionCommand(HandContext& hand, float position_rad, float kp) {
    std::lock_guard<std::mutex> lock(hand.mutex);
    const float desired = std::clamp(
        position_rad, kClosedPositionRad, kOpenPositionRad);
    if (std::abs(desired - hand.desired_position) > 1.0e-5F) {
      hand.desired_position = desired;
      hand.target_changed_at = std::chrono::steady_clock::now();
      hand.target_start_position = hand.has_state
          ? hand.measured_position : hand.last_sent_position;
      hand.last_progress_time = hand.target_changed_at;
      hand.last_progress_position = hand.target_start_position;
      hand.response_warning = false;
    }
    hand.desired_kp = kp;
  }

  void InitializeHand(HandContext& hand,
                      bool is_left,
                      const std::string& command_topic,
                      const std::string& state_topic) {
    {
      std::lock_guard<std::mutex> lock(hand.mutex);
      SizeCommand(hand.command);
    }
    hand.publisher.reset(new ChannelPublisher<MotorCmds_>(command_topic));
    hand.publisher->InitChannel();
    hand.subscriber.reset(new ChannelSubscriber<MotorStates_>(state_topic));
    hand.subscriber->InitChannel(
        [this, is_left](const void* message) { StateHandler(is_left, message); }, 1);
  }

  void StateHandler(bool is_left, const void* message) {
    HandContext& hand = is_left ? left_ : right_;
    const auto* state = static_cast<const MotorStates_*>(message);
    if (state == nullptr || state->states().size() < kMotorCount) return;
    const auto& motor = state->states()[0];
    const float measured_position = motor.q();
    if (!std::isfinite(measured_position)) return;

    const auto now = std::chrono::steady_clock::now();
    bool reconnected = false;
    {
      std::lock_guard<std::mutex> lock(hand.mutex);
      reconnected = !hand.has_state
          || now - hand.last_state_time > kStateStaleTimeout;
      hand.measured_position = measured_position;
      hand.measured_velocity = std::isfinite(motor.dq()) ? motor.dq() : 0.0F;
      hand.measured_tau_est = std::isfinite(motor.tau_est())
          ? motor.tau_est() : 0.0F;
      hand.last_state_time = now;
      hand.has_state = true;
      if (reconnected) {
        // Restart the bounded trajectory from the newly observed physical
        // position. Never continue a filter trajectory based on stale state.
        hand.last_sent_position = measured_position;
        hand.command_history.fill(measured_position);
        hand.command_history_count = 3;
        FillCommand(hand.command, measured_position, hand.desired_kp);
        hand.target_changed_at = now;
        hand.target_start_position = measured_position;
        hand.last_progress_time = now;
        hand.last_progress_position = measured_position;
        hand.stale_reported = false;
        hand.response_warning = false;
      } else if (std::abs(measured_position - hand.last_progress_position)
                 >= kResponseMovementThresholdRad) {
        hand.last_progress_time = now;
        hand.last_progress_position = measured_position;
      }
    }
    const uint64_t count = ++hand.state_count;
    if (count == 1) {
      std::cout << "[dex1] First " << (is_left ? "left" : "right")
                << " MotorStates received at q=" << measured_position << ".\n";
    } else if (reconnected) {
      std::cout << "[dex1] " << (is_left ? "left" : "right")
                << " state reconnected; trajectory reset to measured q="
                << measured_position << ".\n";
    }
  }

  static void WriteHand(HandContext& hand,
                        const char* side,
                        bool shutdown_release) {
    if (!hand.publisher) return;
    MotorCmds_ command;
    const auto now = std::chrono::steady_clock::now();
    bool report_stale = false;
    {
      std::lock_guard<std::mutex> lock(hand.mutex);
      const bool state_fresh = hand.state_count.load() > 0
          && hand.has_state
          && now - hand.last_state_time <= kStateStaleTimeout;

      if (!state_fresh && !shutdown_release) {
        // Keep the DDS server alive, but do not advance the position ramp from
        // an old measured q. The trajectory will be reset by StateHandler when
        // fresh feedback returns.
        if (!hand.stale_reported) {
          hand.stale_reported = true;
          report_stale = true;
        }
      } else {
        const float reference_position = shutdown_release
            ? hand.last_sent_position : hand.measured_position;
        const float bounded_position = std::clamp(
            hand.desired_position,
            reference_position - kMaxCommandStepRad,
            reference_position + kMaxCommandStepRad);
        hand.command_history[2] = hand.command_history[1];
        hand.command_history[1] = hand.command_history[0];
        hand.command_history[0] = bounded_position;
        hand.command_history_count = std::min(3, hand.command_history_count + 1);
        float smoothed_position = bounded_position;
        if (hand.command_history_count == 2) {
          smoothed_position = 0.625F * hand.command_history[0]
              + 0.375F * hand.command_history[1];
        } else if (hand.command_history_count >= 3) {
          smoothed_position = 0.5F * hand.command_history[0]
              + 0.3F * hand.command_history[1]
              + 0.2F * hand.command_history[2];
        }
        hand.last_sent_position = std::clamp(
            smoothed_position, kClosedPositionRad, kOpenPositionRad);
      }
      FillCommand(hand.command, hand.last_sent_position, hand.desired_kp);
      command = hand.command;
    }
    hand.publisher->Write(command);
    if (report_stale) {
      std::cerr << "[dex1] WARNING: " << side
                << " state became stale; position ramp frozen "
                << "at last sent q until feedback reconnects.\n";
    }
  }

  static void LogDiagnostics(HandContext& hand,
                             const char* side,
                             double interval_sec) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(hand.mutex);
    const uint64_t count = hand.state_count.load();
    const uint64_t state_delta = count - hand.diagnostic_state_count;
    hand.diagnostic_state_count = count;
    const double state_hz = interval_sec > 0.0
        ? static_cast<double>(state_delta) / interval_sec : 0.0;
    const double state_age_ms = hand.has_state
        ? std::chrono::duration<double, std::milli>(
              now - hand.last_state_time).count()
        : -1.0;
    const bool fresh = hand.has_state
        && now - hand.last_state_time <= kStateStaleTimeout;

    const float target_error = std::abs(
        hand.desired_position - hand.measured_position);
    const bool check_response = fresh
        && now - hand.target_changed_at >= kResponseCheckDelay
        && target_error > kResponseErrorThresholdRad;
    const bool no_recent_progress =
        now - hand.last_progress_time >= kResponseCheckDelay;
    const bool stationary =
        std::abs(hand.measured_velocity) < kResponseVelocityThresholdRadS;
    const bool nonresponsive = check_response && no_recent_progress && stationary
        && std::abs(hand.measured_tau_est) < kResponseTorqueThresholdNm;
    const bool blocked_or_contact = check_response && no_recent_progress && stationary
        && !nonresponsive;
    if (nonresponsive && !hand.response_warning) {
      std::cerr << "[dex1] WARNING: " << side
                << " hand is not responding to the latest target "
                << "(large q error, negligible motion/velocity/torque).\n";
    }
    hand.response_warning = nonresponsive;

    std::cout << std::fixed << std::setprecision(3)
              << "[dex1-status] side=" << side
              << " state_hz=" << state_hz
              << " state_age_ms=" << state_age_ms
              << " measured_q=" << hand.measured_position
              << " sent_q=" << hand.last_sent_position
              << " desired_q=" << hand.desired_position
              << " dq=" << hand.measured_velocity
              << " tau_est=" << hand.measured_tau_est
              << " state=" << (fresh ? "fresh" : "STALE")
              << " response="
              << (nonresponsive ? "NO_RESPONSE"
                  : (blocked_or_contact ? "BLOCKED_OR_CONTACT" : "ok"))
              << '\n';
  }

  void WriterLoop() {
    auto next_tick = std::chrono::steady_clock::now();
    auto last_diagnostic = next_tick;
    while (writer_running_.load()) {
      next_tick += kKeepalivePeriod;
      WriteHand(left_, "left", false);
      WriteHand(right_, "right", false);

      const auto now = std::chrono::steady_clock::now();
      if (now - last_diagnostic >= std::chrono::seconds(1)) {
        const double interval_sec =
            std::chrono::duration<double>(now - last_diagnostic).count();
        LogDiagnostics(left_, "left", interval_sec);
        LogDiagnostics(right_, "right", interval_sec);
        last_diagnostic = now;
      }
      if (now > next_tick + kKeepalivePeriod) next_tick = now;
      std::this_thread::sleep_until(next_tick);
    }
  }

  HandContext left_;
  HandContext right_;
  std::atomic_bool initialized_{false};
  std::atomic_bool shutting_down_{false};
  std::atomic_bool writer_running_{false};
  std::thread writer_thread_;
};

}  // namespace dex1

#endif  // G1_DEPLOY_REAL_DEX1_HANDS_HPP
