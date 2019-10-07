#include "backend.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace inference_engine {
namespace backend {

void gemm(long m, long n, long k, float *a, float *b, float *c, float *d) {
  for (long m_i = 0; m_i < m; ++m_i) {
    // loop exchange
    for (long k_i = 0; k_i < k; ++k_i) {
      for (long n_i = 0; n_i < n; ++n_i) {
        c[m_i * n + n_i] += a[m_i * k + k_i] * b[k_i * n + n_i];
      }
    }
  }
  for (long m_i = 0; m_i < m; ++m_i) {
    for (long n_i = 0; n_i < n; ++n_i) {
      c[m_i * n + n_i] += d[m_i * n + n_i];
    }
  }
}

void conv_with_padding(long c_in, long c_out, long x_h, long x_w, long y_h,
                       long y_w, long k, long pad, long stride, float *x,
                       float *w, float *b, float *y) {

  long padded_height = x_h + 2 * pad;
  long padded_width = x_w + 2 * pad;
  long long total_padded_x_size = c_in * padded_height * padded_width;

  std::unique_ptr<float[]> padded_x =
      std::make_unique<float[]>(total_padded_x_size);
  memset(padded_x.get(), 0, sizeof(float) * total_padded_x_size);

  long target_x_index_offset = 0l;
  long target_padded_x_index_offset = 0l;
  long target_y_index = 0l;
  long target_w_index_offset = 0l;

  for (long cc_i = 0; cc_i < c_in; ++cc_i) {
    for (long xx_h = 0; xx_h < x_h; ++xx_h) {
      target_padded_x_index_offset = (cc_i * (padded_height) * (padded_width)) +
                                     ((xx_h + pad) * (padded_width)) + pad;
      target_x_index_offset = (cc_i * x_h * x_w) + (xx_h * x_w);

      for (long xx_w = 0; xx_w < x_w; ++xx_w) {
        padded_x[target_padded_x_index_offset + xx_w] =
            x[target_x_index_offset + xx_w];
      }
    }
  }

  for (long cc_out = 0; cc_out < c_out; ++cc_out) {
    for (long yy_h = 0; yy_h < y_h; ++yy_h) {
      for (long yy_w = 0; yy_w < y_w; ++yy_w) {
        target_y_index = (cc_out * y_h * y_w) + (yy_h * y_w) + yy_w;
        y[target_y_index] += b[cc_out];

        for (long k_h = 0; k_h < k; ++k_h) {
          for (long cc_in = 0; cc_in < c_in; ++cc_in) {
            target_x_index_offset = (cc_in * (padded_height) * (padded_width)) +
                                    ((yy_h * stride + k_h) * (padded_width)) +
                                    yy_w * stride;
            target_w_index_offset =
                (cc_out * (c_in * k * k)) + (cc_in * k * k) + k_h * k;

            for (long k_w = 0; k_w < k; ++k_w) {
              y[target_y_index] += padded_x[target_x_index_offset + k_w] *
                                   w[target_w_index_offset + k_w];
            }
          }
        }
      }
    }
  }
}

void conv(long c_in, long c_out, long x_h, long x_w, long y_h, long y_w, long k,
          long pad, long stride, float *x, float *w, float *b, float *y) {
  assert(k <= x_h && k <= x_w);
  assert(y_h == floor((x_h - k + 2 * pad) / float(stride)) + 1);
  assert(y_w == floor((x_w - k + 2 * pad) / float(stride)) + 1);

  if (pad > 0) {
    return conv_with_padding(c_in, c_out, x_h, x_w, y_h, y_w, k, pad, stride, x,
                             w, b, y);
  }

  long target_y_index = 0l;
  long target_x_index_offset = 0l;
  long target_w_index_offset = 0l;

  for (long cc_out = 0; cc_out < c_out; ++cc_out) {
    for (long yy_h = 0; yy_h < y_h; ++yy_h) {
      for (long yy_w = 0; yy_w < y_w; ++yy_w) {
        target_y_index = (cc_out * y_h * y_w) + (yy_h * y_w) + yy_w;
        y[target_y_index] += b[cc_out];

        for (long k_h = 0; k_h < k; ++k_h) {
          for (long cc_in = 0; cc_in < c_in; ++cc_in) {
            target_x_index_offset = (cc_in * x_h * x_w) +
                                    ((yy_h * stride + k_h) * x_w) +
                                    yy_w * stride;
            target_w_index_offset =
                (cc_out * (c_in * k * k)) + (cc_in * k * k) + k_h * k;

            for (long k_w = 0; k_w < k; ++k_w) {
              y[target_y_index] += x[target_x_index_offset + k_w] *
                                   w[target_w_index_offset + k_w];
            }
          }
        }
      }
    }
  }
}

constexpr float MAX_POOL_INITIAL_MAX_ELEMENT = 1 << 31;

void max_pool_with_padding(long c, long x_h, long x_w, long y_h, long y_w,
                           long k, long pad, long stride, float *x, float *y) {

  long padded_height = x_h + 2 * pad;
  long padded_width = x_w + 2 * pad;
  long long total_padded_x_size = c * padded_height * padded_width;

  std::unique_ptr<float[]> padded_x =
      std::make_unique<float[]>(total_padded_x_size);
  memset(padded_x.get(), 0, sizeof(float) * total_padded_x_size);

  long target_x_index_offset = 0l;
  long target_padded_x_index_offset = 0l;
  long target_y_index = 0l;

  for (long cc = 0; cc < c; ++cc) {
    for (long xx_h = 0; xx_h < x_h; ++xx_h) {
      target_padded_x_index_offset = (cc * (padded_height) * (padded_width)) +
                                     ((xx_h + pad) * (padded_width)) + pad;
      target_x_index_offset = (cc * x_h * x_w) + (xx_h * x_w);

      for (long xx_w = 0; xx_w < x_w; ++xx_w) {
        padded_x[target_padded_x_index_offset + xx_w] =
            x[target_x_index_offset + xx_w];
      }
    }
  }

  float max_element = inference_engine::backend::MAX_POOL_INITIAL_MAX_ELEMENT;

  for (long cc = 0; cc < c; ++cc) {
    for (long yy_h = 0; yy_h < y_h; ++yy_h) {
      for (long yy_w = 0; yy_w < y_w; ++yy_w) {
        max_element = inference_engine::backend::MAX_POOL_INITIAL_MAX_ELEMENT;
        target_y_index = (cc * y_h * y_w) + (yy_h * y_w) + yy_w;

        for (long k_h = 0; k_h < k; ++k_h) {
          target_x_index_offset = (cc * padded_height * padded_width) +
                                  ((yy_h * stride + k_h) * padded_width) +
                                  yy_w * stride;

          for (long k_w = 0; k_w < k; ++k_w) {
            max_element =
                std::max(padded_x[target_x_index_offset + k_w], max_element);
          }
        }
        y[target_y_index] = max_element;
      }
    }
  }
}

void max_pool(long c, long x_h, long x_w, long y_h, long y_w, long k, long pad,
              long stride, float *x, float *y) {
  assert(k <= x_h && k <= x_w);
  assert(y_h == floor((x_h - k + 2 * pad) / float(stride)) + 1);
  assert(y_w == floor((x_w - k + 2 * pad) / float(stride)) + 1);

  if (pad > 0) {
    return max_pool_with_padding(c, x_h, x_w, y_h, y_w, k, pad, stride, x, y);
  }

  float max_element = inference_engine::backend::MAX_POOL_INITIAL_MAX_ELEMENT;
  long target_y_index = 0l;
  long target_x_index_offset = 0l;
  for (long cc = 0; cc < c; ++cc) {
    for (long yy_h = 0; yy_h < y_h; ++yy_h) {
      for (long yy_w = 0; yy_w < y_w; ++yy_w) {
        max_element = inference_engine::backend::MAX_POOL_INITIAL_MAX_ELEMENT;
        target_y_index = (cc * y_h * y_w) + (yy_h * y_w) + yy_w;

        for (long k_h = 0; k_h < k; ++k_h) {
          target_x_index_offset =
              (cc * x_h * x_w) + ((yy_h * stride + k_h) * x_w) + yy_w * stride;

          for (long k_w = 0; k_w < k; ++k_w) {
            max_element = std::max(x[target_x_index_offset + k_w], max_element);
          }
        }
        y[target_y_index] = max_element;
      }
    }
  }
}

void drop_out(long long n, float ratio, float *x, float *y, float *mask) {
  std::default_random_engine generator;
  std::binomial_distribution<long> distribution(1, 1.0 - ratio);

  for (long long i = 0; i < n; ++i) {
    mask[i] = distribution(generator);
    y[i] = x[i] * mask[i];
  }
}

void relu(long long n, float *x, float *y) {
  for (long long i = 0; i < n; ++i) {
    y[i] = std::max(0.0f, x[i]);
  }
}

void softmax(long long n, float *x, float *y) {
  float max_value = *std::max_element(x, x + n);
  std::transform(x, x + n, y,
                 ([&](float v) -> float { return std::exp(v - max_value); }));
  float sum_x = std::accumulate(y, y + n, 0.0);
  std::transform(y, y + n, y, ([&](float v) -> float { return v / sum_x; }));
}

} // namespace backend
} // namespace inference_engine
