#ifndef CATCH_CONFIG_MAIN
#define CATCH_CONFIG_MAIN
#endif

#include <catch2/catch.hpp>
#include "../inference_engine/image_util.hpp"
#include "util.hpp"
#include <iostream>

bool test_gray_image_to_hw() {
  cv::Mat image_mat = cv::imread("test/gray.jpg", cv::IMREAD_COLOR);
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

TEST_CASE("gray_image_to_hw",
          "[inference_engine::image_util::gray_image_to_hw]") {
  SECTION("gray image") { REQUIRE(test_gray_image_to_hw() == true); }
}