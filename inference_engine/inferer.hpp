#include <utility>

namespace inference_engine {
namespace inferer {
std::pair<long, long> calculate_conv_matrix_dims(long h, long w, long k,
                                                 long pad, long stride);
} // namespace inferer
} // namespace inference_engine