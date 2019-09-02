#include "../inference_engine/image_util.hpp"
#include "util.hpp"
#include <iostream>

bool test_gray_image_to_hw() {
  cv::Mat image_mat = cv::imread("gray.jpg", cv::IMREAD_COLOR);
  image_mat.convertTo(image_mat, CV_32FC3);
  float *image = new float[image_mat.rows * image_mat.cols];
  memset(image, 0, sizeof(float) * image_mat.rows * image_mat.cols);

  inference_engine::image_util::gray_image_to_hw(image_mat, image);
  if (image == nullptr) {
    return false;
  }

  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      if (image[i * 28 + j] != 128.0) {
        std::cout << "`128.0` is expected, but got `" << image[i * 28 + j]
                  << "` at " << i << "," << j << std::endl;
        return false;
      }
    }
  }

  delete[] image;
  return true;
}

int main() {
  if (!test_gray_image_to_hw()) {
    std::cout << "test_gray_image_to_hw failed" << std::endl;
    return 1;
  }

  std::cout << "all tests are passed" << std::endl;

  return 0;
}