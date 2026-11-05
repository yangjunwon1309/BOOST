#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <unitree/idl/go2/Go2FrontVideoData_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree_go::msg::dds_::Go2FrontVideoData_;

namespace {

std::atomic_bool running{true};
std::atomic<uint64_t> received_count{0};

void SignalHandler(int) {
    running.store(false);
}

void PrintHeader(
    const std::string& name,
    const std::vector<uint8_t>& data
) {
    std::cout << "  " << name
              << ": bytes=" << data.size();

    if (!data.empty()) {
        std::cout << ", header=";

        const std::size_t count =
            std::min<std::size_t>(data.size(), 12);

        for (std::size_t i = 0; i < count; ++i) {
            std::cout
                << std::hex
                << std::setw(2)
                << std::setfill('0')
                << static_cast<unsigned int>(data[i])
                << ' ';
        }

        std::cout << std::dec;
    }

    std::cout << '\n';
}

void SaveFirstFrame(
    const std::string& name,
    const std::vector<uint8_t>& data
) {
    if (data.empty()) {
        return;
    }

    const std::string path =
        "/tmp/front_video_" + name + ".bin";

    std::ofstream output(path, std::ios::binary);

    if (!output.is_open()) {
        std::cerr << "Failed to open " << path << '\n';
        return;
    }

    output.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size())
    );

    std::cout << "  saved first frame: " << path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string network_interface =
        argc > 1 ? argv[1] : "eth0";

    const std::string topic =
        argc > 2 ? argv[2] : "rt/frontvideostream";

    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    std::cout << "Network interface: "
              << network_interface << '\n';

    std::cout << "DDS topic: "
              << topic << '\n';

    ChannelFactory::Instance()->Init(
        0,
        network_interface
    );

    ChannelSubscriber<Go2FrontVideoData_> subscriber(
        topic
    );

    auto first_message_time =
        std::chrono::steady_clock::time_point{};

    auto previous_report_time =
        std::chrono::steady_clock::now();

    uint64_t previous_report_count = 0;

    subscriber.InitChannel(
        [&](const void* message) {
            if (message == nullptr) {
                return;
            }

            const auto* frame =
                static_cast<const Go2FrontVideoData_*>(
                    message
                );

            const uint64_t count =
                ++received_count;

            if (count == 1) {
                first_message_time =
                    std::chrono::steady_clock::now();

                std::cout << "\nFirst frame received\n";
                std::cout << "  time_frame="
                          << frame->time_frame() << '\n';

                PrintHeader(
                    "720p",
                    frame->video720p()
                );

                PrintHeader(
                    "360p",
                    frame->video360p()
                );

                PrintHeader(
                    "180p",
                    frame->video180p()
                );

                SaveFirstFrame(
                    "720p",
                    frame->video720p()
                );

                SaveFirstFrame(
                    "360p",
                    frame->video360p()
                );

                SaveFirstFrame(
                    "180p",
                    frame->video180p()
                );

                std::cout << std::endl;
            }

            const auto now =
                std::chrono::steady_clock::now();

            const double elapsed =
                std::chrono::duration<double>(
                    now - previous_report_time
                ).count();

            if (elapsed >= 2.0) {
                const uint64_t interval_count =
                    count - previous_report_count;

                const double fps =
                    static_cast<double>(interval_count)
                    / elapsed;

                std::cout
                    << "frames=" << count
                    << ", estimated_fps=" << fps
                    << ", time_frame="
                    << frame->time_frame()
                    << ", sizes=("
                    << frame->video720p().size()
                    << ", "
                    << frame->video360p().size()
                    << ", "
                    << frame->video180p().size()
                    << ")\n";

                previous_report_time = now;
                previous_report_count = count;
            }
        },
        1
    );

    const auto started =
        std::chrono::steady_clock::now();

    while (running.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );

        const auto elapsed =
            std::chrono::steady_clock::now()
            - started;

        if (
            received_count.load() == 0
            && elapsed > std::chrono::seconds(10)
        ) {
            std::cerr
                << "\nNo frame received for 10 seconds.\n"
                << "Check the network interface and topic name.\n";

            return 2;
        }
    }

    std::cout << "\nTotal frames received: "
              << received_count.load() << '\n';

    return 0;
}
