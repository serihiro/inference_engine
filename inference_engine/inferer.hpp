#include <utility>

namespace inference_engine {
namespace inferer {
std::pair<int, int> calculate_conv_matrix_dims(int h, int w, int k, int pad,
                                               int stride);
} // namespace inferer
} // namespace inference_engine