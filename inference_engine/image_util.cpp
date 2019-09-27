#include "image_util.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace inference_engine {
namespace image_util {
void gray_image_to_hw(const cv::Mat &image_mat, float *image) {
  for (int y = 0; y < image_mat.rows; ++y) {
    for (int x = 0; x < image_mat.cols; ++x) {
      image[y * image_mat.cols + x] = image_mat.at<cv::Vec3f>(y, x)[0];
    }
  }
}

void rgb_image_to_chw(const cv::Mat &image_mat, float *image) {
  assert(image_mat.channels() == 3);

  int channels = image_mat.channels();
  for (int y = 0; y < image_mat.rows; ++y) {
    for (int x = 0; x < image_mat.cols; ++x) {
      for (int c = 0; c < channels; ++c) {
        // C * H * W
        image[(c * image_mat.rows * image_mat.cols) + (y * image_mat.cols) +
              x] = image_mat.at<cv::Vec3f>(y, x)[2 - c];
      }
    }
  }
}

} // namespace image_util
} // namespace inference_engine