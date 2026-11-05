#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::go2::VideoClient;

void PrintHeader(const std::vector<uint8_t>& data) {
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

int main(int argc, char** argv) {
    const std::string network_interface =
        argc > 1 ? argv[1] : "eth0";

    std::cout
        << "Network interface: "
        << network_interface
        << '\n';

    ChannelFactory::Instance()->Init(
        0,
        network_interface
    );

    VideoClient client;
    client.SetTimeout(3.0f);
    client.Init();

    constexpr int kRequests = 30;

    int success_count = 0;
    double total_request_ms = 0.0;

    const auto total_started =
        std::chrono::steady_clock::now();

    for (int index = 0; index < kRequests; ++index) {
        std::vector<uint8_t> image;

        const auto started =
            std::chrono::steady_clock::now();

        const int ret =
            client.GetImageSample(image);

        const auto finished =
            std::chrono::steady_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                finished - started
            ).count();

        total_request_ms += elapsed_ms;

        std::cout
            << '['
            << std::setw(2)
            << std::setfill('0')
            << index
            << "] ret=" << ret
            << ", bytes=" << image.size()
            << ", elapsed_ms=" << elapsed_ms
            << ", header=";

        PrintHeader(image);
        std::cout << '\n';

        if (ret == 0 && !image.empty()) {
            ++success_count;

            if (success_count == 1) {
                const std::string output_path =
                    "/tmp/videohub_rpc_frame.bin";

                std::ofstream output(
                    output_path,
                    std::ios::binary
                );

                output.write(
                    reinterpret_cast<const char*>(
                        image.data()
                    ),
                    static_cast<std::streamsize>(
                        image.size()
                    )
                );

                std::cout
                    << "Saved first frame: "
                    << output_path
                    << '\n';
            }
        }
    }

    const double total_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now()
            - total_started
        ).count();

    std::cout
        << "\nSummary\n"
        << "  success=" << success_count
        << '/' << kRequests << '\n'
        << "  average_request_ms="
        << total_request_ms / kRequests
        << '\n'
        << "  achieved_rate_hz="
        << kRequests / total_seconds
        << '\n';

    return success_count > 0 ? 0 : 2;
}
