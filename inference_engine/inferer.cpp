#include <cmath>
#include <utility>

namespace inference_engine {
namespace inferer {
std::pair<int, int> calculate_conv_matrix_dims(int h, int w, int k, int pad,
                                               int stride) {
  std::pair<int, int> result;
  result.first = static_cast<int>(floor((h - k + 2 * pad) / float(stride)) + 1);
  result.second =
      static_cast<int>(floor((w - k + 2 * pad) / float(stride)) + 1);

  return result;
}
} // namespace inferer
} // namespace inference_engine