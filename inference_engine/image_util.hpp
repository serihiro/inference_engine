#ifndef IMAGE_UTIL_HPP
#define IMAGE_UTIL_HPP

#include <opencv2/opencv.hpp>

namespace inference_engine {
namespace image_util {
void gray_image_to_hw(const cv::Mat &image_mat, float *image);
}
} // namespace inference_engine
#endif