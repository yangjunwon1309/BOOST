#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <lcm/lcm.h>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

#include "camera_frame_packet.hpp"
#include "jpeg_resize.hpp"

namespace {

constexpr int kOutputWidth = 640;
constexpr int kOutputHeight = 360;

std::atomic_bool g_running{true};

void SignalHandler(int) {
  g_running.store(false);
}

struct Options {
  std::string network_interface;
  std::string lcm_url;
  std::string lcm_channel = g1_camera::kDefaultCameraChannel;
  int poll_rate_hz = 30;
  int jpeg_quality = 80;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " <network_interface> [options]\n\n"
      << "Options:\n"
      << "  --lcm-url <url>              LCM URL (default: env/default)\n"
      << "  --lcm-channel <name>         LCM channel (default: G1_CAMERA_FRAME)\n"
      << "  --poll-rate <hz>             VideoHub RPC poll rate (default: 30)\n"
      << "  --jpeg-quality <0-100>       Output JPEG quality (default: 80)\n";
}

bool ParseOptions(int argc, char** argv, Options& options) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
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

    if (arg == "--lcm-url") {
      const char* value = require_value("--lcm-url");
      if (!value) return false;
      options.lcm_url = value;
    } else if (arg == "--lcm-channel") {
      const char* value = require_value("--lcm-channel");
      if (!value) return false;
      options.lcm_channel = value;
    } else if (arg == "--poll-rate") {
      const char* value = require_value("--poll-rate");
      if (!value) return false;
      options.poll_rate_hz = std::atoi(value);
    } else if (arg == "--jpeg-quality") {
      const char* value = require_value("--jpeg-quality");
      if (!value) return false;
      options.jpeg_quality = std::atoi(value);
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      return false;
    }
  }

  if (options.poll_rate_hz <= 0 || options.poll_rate_hz > 120) {
    std::cerr << "--poll-rate must be in 1..120\n";
    return false;
  }
  if (options.jpeg_quality < 0 || options.jpeg_quality > 100) {
    std::cerr << "--jpeg-quality must be in 0..100\n";
    return false;
  }
  return true;
}

uint64_t WallTimeUs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// Single-slot handoff between the network (producer) thread and the transcode
// (consumer) thread. Holds only the most recent unprocessed frame: if the
// consumer is still busy when a new frame arrives, the older one is dropped so
// teleop always sees the freshest image and latency never accumulates.
struct FrameSlot {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<uint8_t> jpeg;
  uint64_t capture_us = 0;
  bool has_frame = false;
};

// Shared statistics (written from both threads).
std::atomic<uint64_t> g_requests{0};
std::atomic<uint64_t> g_duplicates{0};
std::atomic<uint64_t> g_published{0};
std::atomic<uint64_t> g_errors{0};
std::atomic<uint64_t> g_dropped{0};

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, options)) {
    PrintUsage(argv[0]);
    return argc < 2 ? 1 : 2;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  unitree::robot::ChannelFactory::Instance()->Init(0, options.network_interface);
  unitree::robot::go2::VideoClient video_client;
  video_client.SetTimeout(1.0f);
  video_client.Init();

  lcm_t* lcm = lcm_create(options.lcm_url.empty() ? nullptr : options.lcm_url.c_str());
  if (!lcm) {
    std::cerr << "[videohub-lcm] Failed to create LCM instance\n";
    return 1;
  }

  std::cout << "[videohub-lcm] VideoHub RPC on interface="
            << options.network_interface << '\n'
            << "[videohub-lcm] Output=" << kOutputWidth << 'x' << kOutputHeight
            << " JPEG quality=" << options.jpeg_quality << '\n'
            << "[videohub-lcm] LCM channel=" << options.lcm_channel
            << " poll_rate=" << options.poll_rate_hz << "Hz\n";

  FrameSlot slot;

  // Producer: poll the VideoHub RPC and hand the latest frame to the consumer.
  std::thread producer([&] {
    const auto period =
        std::chrono::microseconds(1000000 / options.poll_rate_hz);
    auto next_tick = std::chrono::steady_clock::now();
    std::vector<uint8_t> source_jpeg;
    std::vector<uint8_t> previous_source_jpeg;

    while (g_running.load()) {
      source_jpeg.clear();
      const int result = video_client.GetImageSample(source_jpeg);
      g_requests.fetch_add(1, std::memory_order_relaxed);

      if (result != 0 || source_jpeg.empty()) {
        g_errors.fetch_add(1, std::memory_order_relaxed);
      } else if (source_jpeg == previous_source_jpeg) {
        g_duplicates.fetch_add(1, std::memory_order_relaxed);
      } else {
        previous_source_jpeg = source_jpeg;
        const uint64_t capture_us = WallTimeUs();
        {
          std::lock_guard<std::mutex> lock(slot.mutex);
          if (slot.has_frame) {
            g_dropped.fetch_add(1, std::memory_order_relaxed);
          }
          // Swap the freshly fetched frame in; source_jpeg reclaims the slot's
          // old buffer so its capacity is reused on the next fetch.
          slot.jpeg.swap(source_jpeg);
          slot.capture_us = capture_us;
          slot.has_frame = true;
        }
        slot.cv.notify_one();
      }

      // Drift-safe pacing: if a slow RPC pushed us past the tick, resync to now
      // instead of bursting to "catch up".
      next_tick += period;
      const auto now = std::chrono::steady_clock::now();
      if (next_tick < now) {
        next_tick = now;
      } else {
        std::this_thread::sleep_until(next_tick);
      }
    }
    // Wake the consumer so it can observe shutdown.
    slot.cv.notify_all();
  });

  // Consumer (this thread): transcode the latest frame and publish it.
  g1_camera::JpegTranscoder transcoder(options.jpeg_quality);
  std::vector<uint8_t> work_jpeg;
  auto last_report = std::chrono::steady_clock::now();

  while (g_running.load()) {
    uint64_t capture_us = 0;
    bool have_frame = false;
    {
      std::unique_lock<std::mutex> lock(slot.mutex);
      // Wake at least once a second even without a frame so the periodic
      // report keeps ticking when the stream stalls.
      slot.cv.wait_for(lock, std::chrono::seconds(1),
                       [&] { return slot.has_frame || !g_running.load(); });
      if (slot.has_frame) {
        work_jpeg.swap(slot.jpeg);
        capture_us = slot.capture_us;
        slot.has_frame = false;
        have_frame = true;
      }
    }

    if (have_frame) {
      try {
        const auto& resized_jpeg =
            transcoder.Transcode(work_jpeg, kOutputWidth, kOutputHeight);
        const auto packet = g1_camera::BuildCameraFramePacket(
            resized_jpeg, kOutputWidth, kOutputHeight, capture_us);
        if (lcm_publish(lcm, options.lcm_channel.c_str(), packet.data(),
                        static_cast<unsigned int>(packet.size())) < 0) {
          g_errors.fetch_add(1, std::memory_order_relaxed);
        } else {
          g_published.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (const std::exception& error) {
        g_errors.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[videohub-lcm] Frame conversion failed: " << error.what()
                  << '\n';
      }
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_report >= std::chrono::seconds(5)) {
      std::cout << "[videohub-lcm] requests="
                << g_requests.load(std::memory_order_relaxed)
                << " published=" << g_published.load(std::memory_order_relaxed)
                << " duplicates="
                << g_duplicates.load(std::memory_order_relaxed)
                << " dropped=" << g_dropped.load(std::memory_order_relaxed)
                << " errors=" << g_errors.load(std::memory_order_relaxed)
                << '\n';
      last_report = now;
    }
  }

  producer.join();

  std::cout << "[videohub-lcm] Stopping. requests="
            << g_requests.load(std::memory_order_relaxed)
            << " published=" << g_published.load(std::memory_order_relaxed)
            << " duplicates=" << g_duplicates.load(std::memory_order_relaxed)
            << " dropped=" << g_dropped.load(std::memory_order_relaxed)
            << " errors=" << g_errors.load(std::memory_order_relaxed) << '\n';
  lcm_destroy(lcm);
  return 0;
}
