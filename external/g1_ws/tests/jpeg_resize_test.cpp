#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "jpeg_resize.hpp"

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  cv::Mat source(1080, 1920, CV_8UC3, cv::Scalar(20, 100, 220));
  std::vector<uint8_t> source_jpeg;
  const bool encoded = cv::imencode(".jpg", source, source_jpeg);
  Require(encoded, "Failed to create source JPEG");

  const auto resized_jpeg = g1_camera::ResizeJpeg(source_jpeg, 640, 360, 80);
  const cv::Mat decoded = cv::imdecode(resized_jpeg, cv::IMREAD_COLOR);

  Require(!decoded.empty(), "Failed to decode resized JPEG");
  Require(decoded.cols == 640, "Resized JPEG width is not 640");
  Require(decoded.rows == 360, "Resized JPEG height is not 360");
  std::cout << "jpeg_resize_test passed\n";
  return 0;
}
