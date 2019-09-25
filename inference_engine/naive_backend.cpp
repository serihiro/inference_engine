#include "backend.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

namespace inference_engine {
namespace backend {

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

void conv_with_padding(int c_in, int c_out, int x_h, int x_w, int y_h, int y_w,
                       int k, int pad, int stride, float *x, float *w, float *b,
                       float *y) {

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

  for (int cc_i = 0; cc_i < c_in; ++cc_i) {
    for (int xx_h = 0; xx_h < x_h; ++xx_h) {
      target_padded_x_index_offset = (cc_i * (padded_height) * (padded_width)) +
                                     ((xx_h + pad) * (padded_width)) + pad;
      target_x_index_offset = (cc_i * x_h * x_w) + (xx_h * x_w);

      for (int xx_w = 0; xx_w < x_w; ++xx_w) {
        padded_x.get()[target_padded_x_index_offset + xx_w] =
            x[target_x_index_offset + xx_w];
      }
    }
  }

  for (int cc_in = 0; cc_in < c_in; ++cc_in) {
    for (int cc_out = 0; cc_out < c_out; ++cc_out) {
      for (int yy_h = 0; yy_h < y_h; ++yy_h) {
        for (int yy_w = 0; yy_w < y_w; ++yy_w) {
          target_y_index = (cc_out * y_h * y_w) + (yy_h * y_w) + yy_w;

          for (int k_h = 0; k_h < k; ++k_h) {
            target_x_index_offset = (cc_in * (padded_height) * (padded_width)) +
                                    ((yy_h * stride + k_h) * (padded_width)) +
                                    yy_w * stride;
            target_w_index_offset =
                (cc_out * (c_in * k * k)) + (cc_in * k * k) + k_h * k;

            for (int k_w = 0; k_w < k; ++k_w) {
              y[target_y_index] += padded_x.get()[target_x_index_offset + k_w] *
                                   w[target_w_index_offset + k_w];
            }
          }
          y[target_y_index] += b[cc_out];
        }
      }
    }
  }
}

void conv(int c_in, int c_out, int x_h, int x_w, int y_h, int y_w, int k,
          int pad, int stride, float *x, float *w, float *b, float *y) {
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

  for (int cc_in = 0; cc_in < c_in; ++cc_in) {
    for (int cc_out = 0; cc_out < c_out; ++cc_out) {
      for (int yy_h = 0; yy_h < y_h; ++yy_h) {
        for (int yy_w = 0; yy_w < y_w; ++yy_w) {
          target_y_index = (cc_out * y_h * y_w) + (yy_h * y_w) + yy_w;

          for (int k_h = 0; k_h < k; ++k_h) {
            target_x_index_offset = (cc_in * x_h * x_w) +
                                    ((yy_h * stride + k_h) * x_w) +
                                    yy_w * stride;
            target_w_index_offset =
                (cc_out * (c_in * k * k)) + (cc_in * k * k) + k_h * k;

            for (int k_w = 0; k_w < k; ++k_w) {
              y[target_y_index] += x[target_x_index_offset + k_w] *
                                   w[target_w_index_offset + k_w];
            }
          }
          y[target_y_index] += b[cc_out];
        }
      }
    }
  }
}

} // namespace backend
} // namespace inference_engine
