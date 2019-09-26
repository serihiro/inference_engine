#ifndef BACKEND_HPP
#define BACKEND_HPP

namespace inference_engine {
namespace backend {

// Calculate C[m * n] = A[m x k] * B[k * n] + D[m * n]
void gemm(int m, int n, int k, float *a, float *b, float *c, float *d);

// Apply relu function to all elements of matrix a
// b[m * n] = relu(a[m * n])
void relu(int m, int n, float *a, float *b);

// Apply Conv
// int x_h/x_w: the size of height and width of input x
// int c_in: the size of channel size of input x
// int c_out: the size of kernel
// int k: kernel size. Note that the kernel is assumed to be square.
// int pad: padding size. Note that the padding is assumed to extend the input x
//  as square.
// int stride: stride size. Note that the striding is assumed that
//  the width of vertical and horizontal striding are same.
// int y_h/y_w: the size of height and width of output y
// float *x: the input float array with c_in * x_h * x_w
// float *w: the kernel array with c_out * c_in * k * k
// float *b: the bias array with c_out
// float *y: the output array with c_o * y_h * y_w
//   where y_h is `floor((x_h - k + 2 * pad) / float(stride)) + 1`
//   and y_w is `floor((x_w - k + 2 * pad) / float(stride)) + 1`
void conv(int c_in, int c_out, int x_h, int x_w, int y_h, int y_w, int k,
          int pad, int stride, float *x, float *w, float *b, float *y);

// Apply MaxPool
// int x_h/x_w: the size of height and width of input x
// int c: the size of channel size of input x and output y
// int k: kernel size. Note that the kernel is assumed to be square.
// int pad: padding size. Note that the padding is assumed to extend the input x
//  as square.
// int stride: stride size. Note that the striding is assumed that
//  the width of vertical and horizontal striding are same.
// int y_h/y_w: the size of height and width of output y
// float *x: the input float array with c * x_h * x_w
// float *y: the output array with c * y_h * y_w
//   where y_h is `floor((x_h - k + 2 * pad) / float(stride)) + 1`
//   and y_w is `floor((x_w - k + 2 * pad) / float(stride)) + 1`
void max_pool(int c, int x_h, int x_w, int y_h, int y_w, int k, int pad,
              int stride, float *x, float *y);
} // namespace backend
} // namespace inference_engine

#endif
