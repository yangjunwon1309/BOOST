#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "lcm_action_source.hpp"
#include "lcm_protocol.hpp"
#include "lcm_robot_state_publisher.hpp"
#include "null_action_source.hpp"
#include "path_direct_protocol.hpp"
#include "robot_parameters.hpp"


using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree_hg::msg::dds_::IMUState_;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;


namespace {

std::atomic_bool g_should_stop{false};

void SignalHandler(int) {
  g_should_stop.store(true);
}

struct BridgeOptions {
  std::string network_interface;
  std::string action_source = "null";
  std::string lcm_url;
  std::string lcm_action_channel = g1_path::kDefaultUpperCommandChannel;
  std::string lcm_state_channel = g1_lcm::kDefaultStateChannel;
  bool publish_lcm_state = false;
  bool dry_run_actions = false;
  int lcm_state_rate_hz = 100;
};

template <typename T>
class TimestampedBuffer {
 public:
  void Set(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = value;
    updated_at_ = std::chrono::steady_clock::now();
  }

  std::optional<T> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<T> value_;
  std::chrono::steady_clock::time_point updated_at_{};
};

class G1LowLevelBridge {
 public:
  explicit G1LowLevelBridge(BridgeOptions options)
      : options_(std::move(options)),
        network_interface_(options_.network_interface) {
    if (options_.action_source == "lcm") {
      action_source_ =
          std::make_unique<LcmActionSource>(options_.lcm_url, options_.lcm_action_channel);
    } else {
      action_source_ = std::make_unique<NullActionSource>();
    }

    if (options_.publish_lcm_state) {
      lcm_state_publisher_ =
          std::make_unique<LcmRobotStatePublisher>(options_.lcm_url, options_.lcm_state_channel);
    }
  }

  bool Start() {
    std::cout << "[bridge] Initializing Unitree DDS on interface: " << network_interface_ << '\n';
    ChannelFactory::Instance()->Init(0, network_interface_);

    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_ARM_SDK_TOPIC));
    lowcmd_publisher_->InitChannel();

    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1LowLevelBridge::LowStateHandler, this, std::placeholders::_1), 1);

    imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber_->InitChannel(
        std::bind(&G1LowLevelBridge::ImuTorsoHandler, this, std::placeholders::_1), 1);

    if (!action_source_->Start()) {
      std::cerr << "[bridge] Failed to start action source: " << action_source_->Name() << '\n';
      return false;
    }

    if (lcm_state_publisher_ && !lcm_state_publisher_->Start()) {
      std::cerr << "[bridge] Failed to start LCM state publisher.\n";
      action_source_->Stop();
      return false;
    }

    SetSafeCommand();
    running_.store(true);
    command_writer_thread_ = std::thread(&G1LowLevelBridge::CommandWriterLoop, this);
    control_thread_ = std::thread(&G1LowLevelBridge::ControlLoop, this);
    if (lcm_state_publisher_) {
      lcm_state_thread_ = std::thread(&G1LowLevelBridge::LcmStatePublisherLoop, this);
    }

    std::cout << "[bridge] Started. Action source: " << action_source_->Name() << '\n';
    std::cout << "[bridge] DDS output: arm-only topic=" << HG_ARM_SDK_TOPIC
              << " joints=15..28\n";
    if (options_.dry_run_actions) {
      std::cout << "[bridge] Dry-run actions enabled; received actions will not move motors.\n";
    } else if (options_.action_source == "null") {
      std::cout << "[bridge] PATH LCM action input is disabled.\n";
    } else {
      std::cout << "[bridge] External actions are enabled; stale/missing actions "
                << "release arm-SDK ownership.\n";
    }
    return true;
  }

  void Wait() {
    while (!g_should_stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void Stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
      return;
    }

    std::cout << "[bridge] Stopping. Releasing arm-SDK ownership before exit.\n";
    action_source_->Stop();

    if (lcm_state_thread_.joinable()) {
      lcm_state_thread_.join();
    }
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
    if (command_writer_thread_.joinable()) {
      command_writer_thread_.join();
    }
    if (lcm_state_publisher_) {
      lcm_state_publisher_->Stop();
    }

    SetSafeCommand();

    // In arm-SDK mode these writes contain only the ownership-release marker;
    // waist and lower body remain owned by the resident G1 controller.
    for (int i = 0; i < 100; ++i) {
      WriteLowCommand();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  ~G1LowLevelBridge() {
    Stop();
  }

 private:
  static constexpr auto kCommandWriterPeriod = std::chrono::microseconds(2000);
  static constexpr auto kControlPeriod = std::chrono::microseconds(20000);
  static constexpr auto kActionTimeout = std::chrono::milliseconds(150);
  static constexpr float kArmSdkMarkerKp = 300.0F;
  static constexpr float kArmSdkMarkerKd = 3.0F;

  void LowStateHandler(const void* message) {
    LowState_ low_state = *(const LowState_*)message;
    const uint32_t received_crc = low_state.crc();
    const uint32_t calculated_crc =
        Crc32Core((uint32_t*)&low_state, (sizeof(LowState_) >> 2) - 1);

    if (received_crc != calculated_crc) {
      ++crc_error_count_;
      if (crc_error_count_ <= 5 || crc_error_count_ % 100 == 0) {
        std::cerr << "[bridge] LowState CRC error count=" << crc_error_count_.load() << '\n';
      }
      return;
    }

    low_state_buffer_.Set(low_state);
    mode_machine_.store(low_state.mode_machine());

    const uint64_t count = ++lowstate_count_;
    if (count == 1) {
      std::cout << "[bridge] First LowState received. G1 type: "
                << static_cast<unsigned>(low_state.mode_machine()) << '\n';
    } else if (count % 2000 == 0) {
      const auto& rpy = low_state.imu_state().rpy();
      std::cout << "[bridge] LowState OK count=" << count
                << " pelvis_rpy=[" << rpy[0] << ", " << rpy[1] << ", " << rpy[2]
                << "]\n";
    }
  }

  void ImuTorsoHandler(const void* message) {
    IMUState_ imu = *(const IMUState_*)message;
    imu_torso_buffer_.Set(imu);
  }

  void CommandWriterLoop() {
    auto next_tick = std::chrono::steady_clock::now();
    while (running_.load()) {
      next_tick += kCommandWriterPeriod;
      WriteLowCommand();
      std::this_thread::sleep_until(next_tick);
    }
  }

  void ControlLoop() {
    auto next_tick = std::chrono::steady_clock::now();
    while (running_.load()) {
      next_tick += kControlPeriod;
      UpdateCommandFromActionSource();
      std::this_thread::sleep_until(next_tick);
    }
  }

  void LcmStatePublisherLoop() {
    const int safe_rate_hz = options_.lcm_state_rate_hz > 0 ? options_.lcm_state_rate_hz : 100;
    const auto period = std::chrono::microseconds(1000000 / safe_rate_hz);
    auto next_tick = std::chrono::steady_clock::now();
    while (running_.load()) {
      next_tick += period;
      PublishLcmRobotState();
      std::this_thread::sleep_until(next_tick);
    }
  }

  void PublishLcmRobotState() {
    if (!lcm_state_publisher_) {
      return;
    }
    auto maybe_low_state = low_state_buffer_.Get();
    if (!maybe_low_state) {
      return;
    }
    lcm_state_publisher_->Publish(*maybe_low_state,
                                  imu_torso_buffer_.Get(),
                                  lowstate_count_.load());
  }

  void UpdateCommandFromActionSource() {
    auto maybe_action = action_source_->Poll();
    if (!maybe_action) {
      SetSafeCommand();
      return;
    }

    const ActionCommand& command = *maybe_action;
    const auto age = std::chrono::steady_clock::now() - command.received_at;
    if (age > kActionTimeout) {
      static auto last_timeout_report = std::chrono::steady_clock::time_point{};
      const auto now = std::chrono::steady_clock::now();
      if (now - last_timeout_report > std::chrono::seconds(1)) {
        std::cerr << "[bridge] Action timeout. Releasing arm-SDK ownership.\n";
        last_timeout_report = now;
      }
      SetSafeCommand();
      return;
    }

    if (options_.dry_run_actions) {
      static uint64_t last_reported_sequence = 0;
      if (command.sequence != last_reported_sequence) {
        std::cout << "[bridge] Dry-run action received seq=" << command.sequence << '\n';
        last_reported_sequence = command.sequence;
      }
      SetSafeCommand();
      return;
    }

    SetDirectMotorCommand(command);
    const uint64_t applied_count = ++action_applied_count_;
    if (applied_count == 1 || applied_count % 500 == 0) {
      const auto action_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - command.received_at).count();
      std::cout << "[bridge] Applied action count=" << applied_count
                << " seq=" << command.sequence
                << " age_ms=" << action_age_ms << '\n';
    }
  }

  void SetDirectMotorCommand(const ActionCommand& command) {
    MotorCommand motor_command;
    for (int i = G1_ARM_START_INDEX;
         i < G1_ARM_START_INDEX + G1_ARM_MOTOR_COUNT; ++i) {
      motor_command.q_target[i] = command.q_target[i];
      motor_command.dq_target[i] = command.dq_target[i];
      motor_command.tau_ff[i] = command.tau_ff[i];
      motor_command.kp[i] = kps[i];
      motor_command.kd[i] = kds[i];
    }
    arm_sdk_armed_.store(true);
    motor_command_buffer_.Set(motor_command);
  }

  void SetSafeCommand() {
    arm_sdk_armed_.store(false);
    motor_command_buffer_.Set(MotorCommand{});
  }

  void WriteLowCommand() {
    if (!lowstate_count_.load()) {
      return;
    }
    auto maybe_command = motor_command_buffer_.Get();
    auto maybe_low_state = low_state_buffer_.Get();
    if (!maybe_command || !maybe_low_state) {
      return;
    }

    const MotorCommand& raw = *maybe_command;
    LowCmd_ dds_low_command;
    dds_low_command.mode_pr(static_cast<uint8_t>(Mode::PR));
    dds_low_command.mode_machine(mode_machine_.load());

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      auto& motor = dds_low_command.motor_cmd().at(i);
      motor.mode() = 1;
      motor.q() = maybe_low_state->motor_state().at(i).q();
      motor.dq() = 0.0F;
      motor.tau() = 0.0F;
      motor.kp() = kps[i];
      motor.kd() = kds[i];
    }
    if (arm_sdk_armed_.load()) {
      for (int i = G1_ARM_START_INDEX;
           i < G1_ARM_START_INDEX + G1_ARM_MOTOR_COUNT; ++i) {
        auto& motor = dds_low_command.motor_cmd().at(i);
        motor.q() = raw.q_target[i];
        motor.dq() = raw.dq_target[i];
        motor.tau() = raw.tau_ff[i];
        motor.kp() = raw.kp[i];
        motor.kd() = raw.kd[i];
      }
    }

    auto& marker = dds_low_command.motor_cmd().at(G1_ARM_SDK_ENABLE_INDEX);
    marker.mode() = 1;
    marker.q() = arm_sdk_armed_.load() ? 1.0F : 0.0F;
    marker.dq() = 0.0F;
    marker.tau() = 0.0F;
    marker.kp() = kArmSdkMarkerKp;
    marker.kd() = kArmSdkMarkerKd;

    dds_low_command.crc() =
        Crc32Core((uint32_t*)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher_->Write(dds_low_command);
  }

  BridgeOptions options_;
  std::string network_interface_;
  std::unique_ptr<ActionSource> action_source_;
  std::unique_ptr<LcmRobotStatePublisher> lcm_state_publisher_;

  TimestampedBuffer<LowState_> low_state_buffer_;
  TimestampedBuffer<IMUState_> imu_torso_buffer_;
  TimestampedBuffer<MotorCommand> motor_command_buffer_;
  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;

  std::atomic_bool running_{false};
  std::thread command_writer_thread_;
  std::thread control_thread_;
  std::thread lcm_state_thread_;

  std::atomic<uint8_t> mode_machine_{0};
  std::atomic<uint64_t> lowstate_count_{0};
  std::atomic<uint64_t> action_applied_count_{0};
  std::atomic<uint64_t> crc_error_count_{0};
  std::atomic_bool arm_sdk_armed_{false};
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " <network_interface> [options]\n"
            << "Options:\n"
            << "  --action-source <null|lcm>       Action source (default: null)\n"
            << "  --lcm-url <url>                  LCM URL (default: LCM default/env)\n"
            << "  --lcm-action-channel <name>      PATH arm channel (default: "
            << g1_path::kDefaultUpperCommandChannel << ")\n"
            << "  --publish-lcm-state              Publish robot state over LCM\n"
            << "  --lcm-state-channel <name>       State channel (default: "
            << g1_lcm::kDefaultStateChannel << ")\n"
            << "  --lcm-state-rate <hz>            State publish rate (default: 100)\n"
            << "  --dry-run-actions                Receive commands without arm SDK ownership\n";
}

bool ParseOptions(int argc, char** argv, BridgeOptions& options) {
  if (argc < 2 || std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
    return false;
  }

  options.network_interface = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--action-source") {
      const char* value = require_value("--action-source");
      if (!value) return false;
      options.action_source = value;
      if (options.action_source != "null" && options.action_source != "lcm") {
        std::cerr << "--action-source must be 'null' or 'lcm'\n";
        return false;
      }
    } else if (arg == "--lcm-url") {
      const char* value = require_value("--lcm-url");
      if (!value) return false;
      options.lcm_url = value;
    } else if (arg == "--lcm-action-channel") {
      const char* value = require_value("--lcm-action-channel");
      if (!value) return false;
      options.lcm_action_channel = value;
    } else if (arg == "--publish-lcm-state") {
      options.publish_lcm_state = true;
    } else if (arg == "--lcm-state-channel") {
      const char* value = require_value("--lcm-state-channel");
      if (!value) return false;
      options.lcm_state_channel = value;
    } else if (arg == "--lcm-state-rate") {
      const char* value = require_value("--lcm-state-rate");
      if (!value) return false;
      options.lcm_state_rate_hz = std::atoi(value);
      if (options.lcm_state_rate_hz <= 0 || options.lcm_state_rate_hz > 1000) {
        std::cerr << "--lcm-state-rate must be in 1..1000\n";
        return false;
      }
    } else if (arg == "--dry-run-actions") {
      options.dry_run_actions = true;
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      return false;
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  BridgeOptions options;
  if (!ParseOptions(argc, argv, options)) {
    PrintUsage(argv[0]);
    return (argc >= 2 && (std::strcmp(argv[1], "-h") == 0 ||
                          std::strcmp(argv[1], "--help") == 0)) ? 0 : 1;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  try {
    G1LowLevelBridge bridge(std::move(options));
    if (!bridge.Start()) {
      return 1;
    }
    bridge.Wait();
    bridge.Stop();
  } catch (const std::exception& e) {
    std::cerr << "[bridge] Fatal error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
