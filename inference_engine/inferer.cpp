#include <cmath>
#include <utility>

namespace inference_engine {
namespace inferer {
std::pair<long, long> calculate_conv_matrix_dims(long h, long w, long k,
                                                 long pad, long stride) {
  std::pair<long, long> result;
  result.first =
      static_cast<long>(floor((h - k + 2 * pad) / float(stride)) + 1);
  result.second =
      static_cast<long>(floor((w - k + 2 * pad) / float(stride)) + 1);

  return result;
}
} // namespace inferer
} // namespace inference_engine