#include "../inference_engine/backend.hpp"
#include "util.hpp"
#include <iostream>

bool test_relu1() {
  float result[4];
  float test[4] = {-1.0, 1.0, 0.0, 0.5};
  float expected[4] = {0.0, 1.0, 0.0, 0.5};
  inference_engine::backend::relu(2, 2, test, result);

  return inference_engine::test::assert_array_eq_float(result, expected, 4ll);
}

bool test_relu2() {
  float result[4];
  float test[4] = {-1.0, -100.0, 0.0, -0.1};
  float expected[4] = {0.0, 0.0, 0.0, 0.0};
  inference_engine::backend::relu(2, 2, test, result);

  return inference_engine::test::assert_array_eq_float(result, expected, 4ll);
}

void array_arange(float *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = float(i);
  }
}

void array_zeros(float *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = 0.0;
  }
}

bool test_gem1() {
  int m = 2;
  int k = 2;
  int n = 2;
  float a[m * k];
  float b[k * n];
  float c[m * n];
  float d[m * n];

  array_arange(a, m * k);
  array_arange(b, k * n);
  array_zeros(c, m * n);
  array_arange(d, m * n);

  inference_engine::backend::gemm(m, n, k, a, b, c, d);
  float expected[4] = {2, 4, 8, 14};
  return inference_engine::test::assert_array_eq_float(c, expected, m * n);
}

bool test_gem2() {
  int m = 3;
  int k = 4;
  int n = 5;
  float a[m * k];
  float b[k * n];
  float c[m * n];
  float d[m * n];
  array_arange(a, m * k);
  array_arange(b, k * n);
  array_zeros(c, m * n);
  array_arange(d, m * n);

  inference_engine::backend::gemm(m, n, k, a, b, c, d);
  float expected[15] = {70,  77,  84,  91,  98,  195, 218, 241,
                        264, 287, 320, 359, 398, 437, 476};
  return inference_engine::test::assert_array_eq_float(c, expected, m * n);
}

bool test_gem3() {
  int m = 10;
  int k = 10;
  int n = 1;
  float a[m * k];
  float b[k * n];
  float c[m * n];
  float d[m * n];
  array_arange(a, m * k);
  array_arange(b, k * n);
  array_zeros(c, m * n);
  array_arange(d, m * n);

  inference_engine::backend::gemm(m, n, k, a, b, c, d);
  float expected[10] = {285,  736,  1187, 1638, 2089,
                        2540, 2991, 3442, 3893, 4344};
  return inference_engine::test::assert_array_eq_float(c, expected, m * n);
}

int main() {
  if (!test_relu1()) {
    std::cout << "test_relu1 failed" << std::endl;
    return 1;
  }

  if (!test_relu2()) {
    std::cout << "test_relu2 failed" << std::endl;
    return 1;
  }

  if (!test_gem1()) {
    std::cout << "test_gem1 failed" << std::endl;
    return 1;
  }

  if (!test_gem2()) {
    std::cout << "test_gem2 failed" << std::endl;
    return 1;
  }

  if (!test_gem3()) {
    std::cout << "test_gem3 failed" << std::endl;
    return 1;
  }

  std::cout << "all tests are passed" << std::endl;

  return 0;
}
