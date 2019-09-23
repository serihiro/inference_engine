#include "backend.hpp"
#include <algorithm>
#include <iostream>

namespace inference_engine {
namespace backend {

// Calculate C[m * n] = A[m x k] * B[k * n] + D[m * n]
void gemm(int m, int n, int k, float *a, float *b, float *c, float *d) {
  for (int m_i = 0; m_i < m; ++m_i) {
    // loop exchange
    for (int k_i = 0; k_i < k; ++k_i) {
      for (int n_i = 0; n_i < n; ++n_i) {
        c[m_i * n + n_i] += a[m_i * k + k_i] * b[k_i * n + n_i];
      }
    }
  }
  for (int m_i = 0; m_i < m; ++m_i) {
    for (int n_i = 0; n_i < n; ++n_i) {
      c[m_i * n + n_i] += d[m_i * n + n_i];
    }
  }
}

void relu(int m, int n, float *a, float *b) {
  for (long long i = 0; i < m * n; ++i) {
    b[i] = std::max(0.0f, a[i]);
  }
}
} // namespace backend
} // namespace inference_engine
