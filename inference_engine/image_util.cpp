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
} // namespace image_util
} // namespace inference_engine