#ifndef G1_DEPLOY_REAL_CAMERA_FRAME_PACKET_HPP
#define G1_DEPLOY_REAL_CAMERA_FRAME_PACKET_HPP

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace g1_camera {

constexpr uint32_t kCameraFrameMagic = 0x4D524143U;  // "CRAM"
constexpr const char* kDefaultCameraChannel = "G1_CAMERA_FRAME";

inline std::vector<uint8_t> BuildCameraFramePacket(
    const std::vector<uint8_t>& jpeg,
    uint32_t width,
    uint32_t height,
    uint64_t timestamp_us) {
  if (jpeg.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("JPEG is too large for CameraFramePacket");
  }

  const uint32_t jpeg_size = static_cast<uint32_t>(jpeg.size());
  std::vector<uint8_t> packet(24 + jpeg_size);
  uint8_t* data = packet.data();

  std::memcpy(data + 0, &kCameraFrameMagic, sizeof(kCameraFrameMagic));
  std::memcpy(data + 4, &width, sizeof(width));
  std::memcpy(data + 8, &height, sizeof(height));
  std::memcpy(data + 12, &jpeg_size, sizeof(jpeg_size));
  std::memcpy(data + 16, &timestamp_us, sizeof(timestamp_us));
  if (!jpeg.empty()) {
    std::memcpy(data + 24, jpeg.data(), jpeg.size());
  }
  return packet;
}

}  // namespace g1_camera

#endif  // G1_DEPLOY_REAL_CAMERA_FRAME_PACKET_HPP
