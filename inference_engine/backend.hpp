#ifndef BACKEND_HPP
#define BACKEND_HPP

namespace inference_engine {
namespace backend {

// Calculate C[m * n] = A[m x k] * B[k * n] + D[k]
void gemm(int m, int n, int k, float *a, float *b, float *c, float *d);

// Apply relu function to all elements of matrix a
// b[m * n] = relu(a[m * n])
void relu(int m, int n, float *a, float *b);
} // namespace backend
} // namespace inference_engine

#endif
