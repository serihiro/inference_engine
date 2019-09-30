#ifndef BACKEND_HPP
#define BACKEND_HPP

namespace inference_engine {
namespace backend {

// Calculate C[m * n] = A[m x k] * B[k * n] + D[m * n]
void gemm(long m, long n, long k, float *a, float *b, float *c, float *d);

// Apply Conv
// long x_h/x_w: the size of height and width of input x
// long c_in: the size of channel size of input x
// long c_out: the size of kernel
// long k: kernel size. Note that the kernel is assumed to be square.
// long pad: padding size. Note that the padding is assumed to extend the input
// x
//  as square.
// long stride: stride size. Note that the striding is assumed that
//  the width of vertical and horizontal striding are same.
// long y_h/y_w: the size of height and width of output y
// float *x: the input float array with c_in * x_h * x_w
// float *w: the kernel array with c_out * c_in * k * k
// float *b: the bias array with c_out
// float *y: the output array with c_o * y_h * y_w
//   where y_h is `floor((x_h - k + 2 * pad) / float(stride)) + 1`
//   and y_w is `floor((x_w - k + 2 * pad) / float(stride)) + 1`
void conv(long c_in, long c_out, long x_h, long x_w, long y_h, long y_w, long k,
          long pad, long stride, float *x, float *w, float *b, float *y);

// Apply MaxPool
// long x_h/x_w: the size of height and width of input x
// long c: the size of channel size of input x and output y
// long k: kernel size. Note that the kernel is assumed to be square.
// long pad: padding size. Note that the padding is assumed to extend
//   the input x as square.
// long stride: stride size. Note that the striding is assumed that
//  the width of vertical and horizontal striding are same.
// long y_h/y_w: the size of height and width of output y
// float *x: the input float array with c * x_h * x_w
// float *y: the output array with c * y_h * y_w
//   where y_h is `floor((x_h - k + 2 * pad) / float(stride)) + 1`
//   and y_w is `floor((x_w - k + 2 * pad) / float(stride)) + 1`
void max_pool(long c, long x_h, long x_w, long y_h, long y_w, long k, long pad,
              long stride, float *x, float *y);

// Apply Dropout
// long c: the size of channel size of input x, output y, and output mask
// long h: the size of height of input x, output y, and output mask
// long w: the size of width of input x, output y, and output mask
// float ratio: the ratio of random dropout. Note that this parameter is
//    used for sampling from binomial distribution to generate a 0/1 mask.
// float *x: the input array with n
// float *y: the output array with n
// float *mask: the output array with n
void drop_out(long long n, float ratio, float *x, float *y, float *mask);

// Apply relu function to all elements of matrix a
// long long n: the size of input x and output y
// float *x: the input vector with n
// float *y: the output vector with n
void relu(long long n, float *x, float *y);

// Apply Softmax
// long long n: the size of input x and output y
// float *x: the input vector with n
// float *y: the output vector with n
void softmax(long long n, float *x, float *y);
} // namespace backend
} // namespace inference_engine

#endif
