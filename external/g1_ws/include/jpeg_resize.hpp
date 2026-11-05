#ifndef G1_DEPLOY_REAL_JPEG_RESIZE_HPP
#define G1_DEPLOY_REAL_JPEG_RESIZE_HPP

#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

namespace g1_camera {

// Decodes a JPEG at 1/2 scale (libjpeg DCT scaling, ~4x cheaper than a full
// decode) and, if the decoded size differs from the requested output size,
// resizes to it. For a 1920x1080 source the reduced decode yields 960x540,
// which is then downscaled to the 640x360 output.
//
// Standalone convenience form. Allocates fresh buffers on every call; for the
// hot path use JpegTranscoder below, which reuses its buffers.
inline std::vector<uint8_t> ResizeJpeg(
    const std::vector<uint8_t>& source_jpeg,
    int width,
    int height,
    int jpeg_quality) {
  if (source_jpeg.empty()) {
    throw std::invalid_argument("Source JPEG is empty");
  }
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Output dimensions must be positive");
  }
  if (jpeg_quality < 0 || jpeg_quality > 100) {
    throw std::invalid_argument("JPEG quality must be in 0..100");
  }

  const cv::Mat decoded = cv::imdecode(source_jpeg, cv::IMREAD_REDUCED_COLOR_2);
  if (decoded.empty()) {
    throw std::runtime_error("VideoHub payload is not a decodable JPEG");
  }

  cv::Mat resized;
  const cv::Mat* to_encode = &decoded;
  if (decoded.cols != width || decoded.rows != height) {
    cv::resize(decoded, resized, cv::Size(width, height), 0.0, 0.0,
               cv::INTER_AREA);
    to_encode = &resized;
  }

  std::vector<uint8_t> output;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
  if (!cv::imencode(".jpg", *to_encode, output, params)) {
    throw std::runtime_error("Failed to encode resized JPEG");
  }
  return output;
}

// Reusable transcoder for the streaming hot path. Holds its decode/resize/
// encode buffers across frames so no per-frame heap allocation is required.
// Not thread-safe: use one instance per consumer thread.
class JpegTranscoder {
 public:
  explicit JpegTranscoder(int jpeg_quality)
      : params_{cv::IMWRITE_JPEG_QUALITY, jpeg_quality} {
    if (jpeg_quality < 0 || jpeg_quality > 100) {
      throw std::invalid_argument("JPEG quality must be in 0..100");
    }
  }

  // Transcodes source_jpeg to width x height and returns a reference to the
  // internal output buffer (valid until the next Transcode call).
  const std::vector<uint8_t>& Transcode(
      const std::vector<uint8_t>& source_jpeg, int width, int height) {
    if (source_jpeg.empty()) {
      throw std::invalid_argument("Source JPEG is empty");
    }
    cv::imdecode(source_jpeg, cv::IMREAD_REDUCED_COLOR_2, &decoded_);
    if (decoded_.empty()) {
      throw std::runtime_error("VideoHub payload is not a decodable JPEG");
    }

    const cv::Mat* to_encode = &decoded_;
    if (decoded_.cols != width || decoded_.rows != height) {
      cv::resize(decoded_, resized_, cv::Size(width, height), 0.0, 0.0,
                 cv::INTER_AREA);
      to_encode = &resized_;
    }

    if (!cv::imencode(".jpg", *to_encode, output_, params_)) {
      throw std::runtime_error("Failed to encode resized JPEG");
    }
    return output_;
  }

 private:
  std::vector<int> params_;
  cv::Mat decoded_;
  cv::Mat resized_;
  std::vector<uint8_t> output_;
};

}  // namespace g1_camera

#endif  // G1_DEPLOY_REAL_JPEG_RESIZE_HPP
