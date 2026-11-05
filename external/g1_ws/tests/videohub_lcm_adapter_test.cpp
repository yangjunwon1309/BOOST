#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "camera_frame_packet.hpp"

namespace {

template <typename T>
T Read(const std::vector<uint8_t>& data, std::size_t offset) {
  T value{};
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  const std::vector<uint8_t> resized_jpeg = {0xff, 0xd8, 0xff, 0xd9};

  constexpr uint64_t timestamp_us = 123456789ULL;
  const auto packet = g1_camera::BuildCameraFramePacket(
      resized_jpeg, 640, 360, timestamp_us);

  Require(packet.size() == 24 + resized_jpeg.size(), "Packet size mismatch");
  Require(Read<uint32_t>(packet, 0) == g1_camera::kCameraFrameMagic,
          "Packet magic mismatch");
  Require(Read<uint32_t>(packet, 4) == 640U, "Packet width mismatch");
  Require(Read<uint32_t>(packet, 8) == 360U, "Packet height mismatch");
  Require(Read<uint32_t>(packet, 12) == resized_jpeg.size(),
          "Packet JPEG size mismatch");
  Require(Read<uint64_t>(packet, 16) == timestamp_us,
          "Packet timestamp mismatch");
  Require(std::memcmp(packet.data() + 24, resized_jpeg.data(), resized_jpeg.size()) == 0,
          "Packet JPEG payload mismatch");

  std::cout << "videohub_lcm_adapter_test passed\n";
  return 0;
}
