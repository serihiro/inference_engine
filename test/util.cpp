#include <iostream>

namespace inference_engine {
namespace test {
bool assert_array_eq_float(float *actual, float *expected,
                           long long array_size) {
  for (long long i = 0; i < array_size; ++i) {
    if (actual[i] != expected[i]) {
      std::cout << expected[i] << " is expected, but got " << actual[i]
                << " at index " << i << std::endl;
      return false;
    }
  }

  return true;
}
} // namespace test
} // namespace inference_engine
