#ifndef CATCH_CONFIG_MAIN
#define CATCH_CONFIG_MAIN
#endif

#include "../external/catch.hpp"
#include "../inference_engine/backend.hpp"
#include "util.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

void array_arange(float *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = float(i);
  }
}

void array_full(float *a, int n, float v) {
  for (int i = 0; i < n; ++i) {
    a[i] = v;
  }
}

void array_zeros(float *a, int n) { array_full(a, n, 0.0); }

TEST_CASE("relu") {
  SECTION("case 1") {
    float result[4];
    float test[4] = {-1.0, 1.0, 0.0, 0.5};
    float expected[4] = {0.0, 1.0, 0.0, 0.5};
    inference_engine::backend::relu(2, 2, test, result);

    REQUIRE(
        inference_engine::test::assert_array_eq_float(result, expected, 4ll));
  }

  SECTION("case 2") {
    float result[4];
    float test[4] = {-1.0, -100.0, 0.0, -0.1};
    float expected[4] = {0.0, 0.0, 0.0, 0.0};
    inference_engine::backend::relu(2, 2, test, result);

    REQUIRE(
        inference_engine::test::assert_array_eq_float(result, expected, 4ll));
  }
}

TEST_CASE("gemm") {
  SECTION("case 1") {
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
    REQUIRE(inference_engine::test::assert_array_eq_float(c, expected, m * n));
  }

  SECTION("case 2") {
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
    REQUIRE(inference_engine::test::assert_array_eq_float(c, expected, m * n));
  }

  SECTION("case 3") {
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
    REQUIRE(inference_engine::test::assert_array_eq_float(c, expected, m * n));
  }

  SECTION("performance test") {
    int m = 1024;
    int k = 1024;
    int n = 1024;
    // To avoid stack over flow, allocate arrays on heap.
    float *a = new float[m * k];
    float *b = new float[k * n];
    float *c = new float[m * n];
    float *d = new float[m * n];
    array_arange(a, m * k);
    array_arange(b, k * n);
    array_zeros(c, m * n);
    array_arange(d, m * n);

    auto start = std::chrono::system_clock::now();
    inference_engine::backend::gemm(m, n, k, a, b, c, d);
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    auto elapsed_msec =
        std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
    REQUIRE(elapsed_msec < 2000);
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
  }
}

TEST_CASE("conv") {
  SECTION("1x3x3 image, 1x2x2 kernel") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 3;
    int x_w = 3;
    int k = 2;
    int stride = 1;
    int pad = 0;
    int y_h = 2;
    int y_w = 2;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[4] = {16.0, 24.0, 40.0, 48.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x3x3 image, 1x1x1 kernel") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 3;
    int x_w = 3;
    int k = 1;
    int stride = 1;
    int pad = 0;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[9] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x3x3 image, 1x2x2 kernel, 2 stride") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 3;
    int x_w = 3;
    int k = 2;
    int stride = 2;
    int pad = 0;
    int y_h = 1;
    int y_w = 1;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[1] = {16.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x10x10 image, 1x2x2 kernel, 3 stride") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 3;
    int pad = 0;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[9] = {44.0,  68.0,  92.0,  284.0, 308.0,
                         332.0, 524.0, 548.0, 572.0};
    for (int i = 0; i < 9; ++i) {
      expected[i] += 0.125;
    }
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_full(b, c_out, 0.125);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x10x10 image, 1x2x2 kernel, 1 padding") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 1;
    int pad = 1;
    int y_h = 11;
    int y_w = 11;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[121] = {
        0.0,   2.0,   6.0,   10.0,  14.0,  18.0,  22.0,  26.0,  30.0,  34.0,
        18.0,  20.0,  44.0,  52.0,  60.0,  68.0,  76.0,  84.0,  92.0,  100.0,
        108.0, 56.0,  60.0,  124.0, 132.0, 140.0, 148.0, 156.0, 164.0, 172.0,
        180.0, 188.0, 96.0,  100.0, 204.0, 212.0, 220.0, 228.0, 236.0, 244.0,
        252.0, 260.0, 268.0, 136.0, 140.0, 284.0, 292.0, 300.0, 308.0, 316.0,
        324.0, 332.0, 340.0, 348.0, 176.0, 180.0, 364.0, 372.0, 380.0, 388.0,
        396.0, 404.0, 412.0, 420.0, 428.0, 216.0, 220.0, 444.0, 452.0, 460.0,
        468.0, 476.0, 484.0, 492.0, 500.0, 508.0, 256.0, 260.0, 524.0, 532.0,
        540.0, 548.0, 556.0, 564.0, 572.0, 580.0, 588.0, 296.0, 300.0, 604.0,
        612.0, 620.0, 628.0, 636.0, 644.0, 652.0, 660.0, 668.0, 336.0, 340.0,
        684.0, 692.0, 700.0, 708.0, 716.0, 724.0, 732.0, 740.0, 748.0, 376.0,
        180.0, 362.0, 366.0, 370.0, 374.0, 378.0, 382.0, 386.0, 390.0, 394.0,
        198.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x10x10 image, 1x2x2 kernel, 2 padding, 3 stride") {
    int c_in = 1;
    int c_out = 1;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 3;
    int pad = 2;
    int y_h = 5;
    int y_w = 5;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[25] = {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   132.0,
                          156.0, 180.0, 0.0,   0.0,   372.0, 396.0, 420.0,
                          0.0,   0.0,   612.0, 636.0, 660.0, 0.0,   0.0,
                          0.0,   0.0,   0.0,   0.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("2x10x10 image, 1x2x2 kernel, 0 padding, 1 stride") {
    int c_in = 2;
    int c_out = 1;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 1;
    int pad = 0;
    int y_h = 9;
    int y_w = 9;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[81] = {
        888.0,  904.0,  920.0,  936.0,  952.0,  968.0,  984.0,  1000.0, 1016.0,
        1048.0, 1064.0, 1080.0, 1096.0, 1112.0, 1128.0, 1144.0, 1160.0, 1176.0,
        1208.0, 1224.0, 1240.0, 1256.0, 1272.0, 1288.0, 1304.0, 1320.0, 1336.0,
        1368.0, 1384.0, 1400.0, 1416.0, 1432.0, 1448.0, 1464.0, 1480.0, 1496.0,
        1528.0, 1544.0, 1560.0, 1576.0, 1592.0, 1608.0, 1624.0, 1640.0, 1656.0,
        1688.0, 1704.0, 1720.0, 1736.0, 1752.0, 1768.0, 1784.0, 1800.0, 1816.0,
        1848.0, 1864.0, 1880.0, 1896.0, 1912.0, 1928.0, 1944.0, 1960.0, 1976.0,
        2008.0, 2024.0, 2040.0, 2056.0, 2072.0, 2088.0, 2104.0, 2120.0, 2136.0,
        2168.0, 2184.0, 2200.0, 2216.0, 2232.0, 2248.0, 2264.0, 2280.0, 2296.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("1x10x10 image, 2x2x2 kernel, 0 padding, 1 stride") {
    int c_in = 1;
    int c_out = 2;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 1;
    int pad = 0;
    int y_h = 9;
    int y_w = 9;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[162] = {
        44.0,  52.0,  60.0,  68.0,  76.0,  84.0,  92.0,  100.0, 108.0, 124.0,
        132.0, 140.0, 148.0, 156.0, 164.0, 172.0, 180.0, 188.0, 204.0, 212.0,
        220.0, 228.0, 236.0, 244.0, 252.0, 260.0, 268.0, 284.0, 292.0, 300.0,
        308.0, 316.0, 324.0, 332.0, 340.0, 348.0, 364.0, 372.0, 380.0, 388.0,
        396.0, 404.0, 412.0, 420.0, 428.0, 444.0, 452.0, 460.0, 468.0, 476.0,
        484.0, 492.0, 500.0, 508.0, 524.0, 532.0, 540.0, 548.0, 556.0, 564.0,
        572.0, 580.0, 588.0, 604.0, 612.0, 620.0, 628.0, 636.0, 644.0, 652.0,
        660.0, 668.0, 684.0, 692.0, 700.0, 708.0, 716.0, 724.0, 732.0, 740.0,
        748.0, 44.0,  52.0,  60.0,  68.0,  76.0,  84.0,  92.0,  100.0, 108.0,
        124.0, 132.0, 140.0, 148.0, 156.0, 164.0, 172.0, 180.0, 188.0, 204.0,
        212.0, 220.0, 228.0, 236.0, 244.0, 252.0, 260.0, 268.0, 284.0, 292.0,
        300.0, 308.0, 316.0, 324.0, 332.0, 340.0, 348.0, 364.0, 372.0, 380.0,
        388.0, 396.0, 404.0, 412.0, 420.0, 428.0, 444.0, 452.0, 460.0, 468.0,
        476.0, 484.0, 492.0, 500.0, 508.0, 524.0, 532.0, 540.0, 548.0, 556.0,
        564.0, 572.0, 580.0, 588.0, 604.0, 612.0, 620.0, 628.0, 636.0, 644.0,
        652.0, 660.0, 668.0, 684.0, 692.0, 700.0, 708.0, 716.0, 724.0, 732.0,
        740.0, 748.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("2x10x10 image, 3x2x2 kernel, 0 padding, 1 stride") {
    int c_in = 2;
    int c_out = 3;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 1;
    int pad = 0;
    int y_h = 9;
    int y_w = 9;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[243] = {
        888.0,  904.0,  920.0,  936.0,  952.0,  968.0,  984.0,  1000.0, 1016.0,
        1048.0, 1064.0, 1080.0, 1096.0, 1112.0, 1128.0, 1144.0, 1160.0, 1176.0,
        1208.0, 1224.0, 1240.0, 1256.0, 1272.0, 1288.0, 1304.0, 1320.0, 1336.0,
        1368.0, 1384.0, 1400.0, 1416.0, 1432.0, 1448.0, 1464.0, 1480.0, 1496.0,
        1528.0, 1544.0, 1560.0, 1576.0, 1592.0, 1608.0, 1624.0, 1640.0, 1656.0,
        1688.0, 1704.0, 1720.0, 1736.0, 1752.0, 1768.0, 1784.0, 1800.0, 1816.0,
        1848.0, 1864.0, 1880.0, 1896.0, 1912.0, 1928.0, 1944.0, 1960.0, 1976.0,
        2008.0, 2024.0, 2040.0, 2056.0, 2072.0, 2088.0, 2104.0, 2120.0, 2136.0,
        2168.0, 2184.0, 2200.0, 2216.0, 2232.0, 2248.0, 2264.0, 2280.0, 2296.0,
        888.0,  904.0,  920.0,  936.0,  952.0,  968.0,  984.0,  1000.0, 1016.0,
        1048.0, 1064.0, 1080.0, 1096.0, 1112.0, 1128.0, 1144.0, 1160.0, 1176.0,
        1208.0, 1224.0, 1240.0, 1256.0, 1272.0, 1288.0, 1304.0, 1320.0, 1336.0,
        1368.0, 1384.0, 1400.0, 1416.0, 1432.0, 1448.0, 1464.0, 1480.0, 1496.0,
        1528.0, 1544.0, 1560.0, 1576.0, 1592.0, 1608.0, 1624.0, 1640.0, 1656.0,
        1688.0, 1704.0, 1720.0, 1736.0, 1752.0, 1768.0, 1784.0, 1800.0, 1816.0,
        1848.0, 1864.0, 1880.0, 1896.0, 1912.0, 1928.0, 1944.0, 1960.0, 1976.0,
        2008.0, 2024.0, 2040.0, 2056.0, 2072.0, 2088.0, 2104.0, 2120.0, 2136.0,
        2168.0, 2184.0, 2200.0, 2216.0, 2232.0, 2248.0, 2264.0, 2280.0, 2296.0,
        888.0,  904.0,  920.0,  936.0,  952.0,  968.0,  984.0,  1000.0, 1016.0,
        1048.0, 1064.0, 1080.0, 1096.0, 1112.0, 1128.0, 1144.0, 1160.0, 1176.0,
        1208.0, 1224.0, 1240.0, 1256.0, 1272.0, 1288.0, 1304.0, 1320.0, 1336.0,
        1368.0, 1384.0, 1400.0, 1416.0, 1432.0, 1448.0, 1464.0, 1480.0, 1496.0,
        1528.0, 1544.0, 1560.0, 1576.0, 1592.0, 1608.0, 1624.0, 1640.0, 1656.0,
        1688.0, 1704.0, 1720.0, 1736.0, 1752.0, 1768.0, 1784.0, 1800.0, 1816.0,
        1848.0, 1864.0, 1880.0, 1896.0, 1912.0, 1928.0, 1944.0, 1960.0, 1976.0,
        2008.0, 2024.0, 2040.0, 2056.0, 2072.0, 2088.0, 2104.0, 2120.0, 2136.0,
        2168.0, 2184.0, 2200.0, 2216.0, 2232.0, 2248.0, 2264.0, 2280.0, 2296.0};
    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_zeros(b, c_out);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }

  SECTION("3x10x10 image, 3x2x2 kernel, 5 padding, 3 stride") {
    int c_in = 3;
    int c_out = 3;
    int x_h = 10;
    int x_w = 10;
    int k = 2;
    int stride = 3;
    int pad = 5;
    int y_h = 7;
    int y_w = 7;
    float *x = new float[c_in * x_h * x_w];
    float *w = new float[c_out * c_in * k * k];
    float *y = new float[c_out * y_h * y_w];
    float *b = new float[c_out];
    float expected[147] = {
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    2796.0, 2868.0,
        2940.0, 0.0,    0.0,    0.0,    0.0,    3516.0, 3588.0, 3660.0, 0.0,
        0.0,    0.0,    0.0,    4236.0, 4308.0, 4380.0, 0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    2796.0, 2868.0, 2940.0, 0.0,    0.0,    0.0,    0.0,
        3516.0, 3588.0, 3660.0, 0.0,    0.0,    0.0,    0.0,    4236.0, 4308.0,
        4380.0, 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    2796.0, 2868.0, 2940.0,
        0.0,    0.0,    0.0,    0.0,    3516.0, 3588.0, 3660.0, 0.0,    0.0,
        0.0,    0.0,    4236.0, 4308.0, 4380.0, 0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        0.0,    0.0,    0.0};

    // bias
    for (int i = 0; i < 147; ++i) {
      expected[i] += 0.125;
    }

    array_arange(x, c_in * x_h * x_w);
    array_full(w, c_out * c_in * k * k, 2.0);
    array_zeros(y, c_out * y_h * y_w);
    array_full(b, c_out, 0.125);

    inference_engine::backend::conv(c_in, c_out, x_h, x_w, y_h, y_w, k, pad,
                                    stride, x, w, b, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c_out * y_h * y_w));
    delete[] x;
    delete[] w;
    delete[] y;
    delete[] b;
  }
}

TEST_CASE("max_pool") {
  SECTION("1x3x3 image, 1x1 kernel") {
    int c = 1;
    int x_h = 3;
    int x_w = 3;
    int k = 1;
    int pad = 0;
    int stride = 1;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[9] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("1x4x4 image, 2x2 kernel") {
    int c = 1;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 0;
    int stride = 1;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[9] = {5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("2x4x4 image, 2x2 kernel") {
    int c = 2;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 0;
    int stride = 1;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[18] = {5.0,  6.0,  7.0,  9.0,  10.0, 11.0, 13.0, 14.0, 15.0,
                          21.0, 22.0, 23.0, 25.0, 26.0, 27.0, 29.0, 30.0, 31.0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("1x4x4 image, 2x2 kernel, 1 padding") {
    int c = 1;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 1;
    int stride = 1;
    int y_h = 5;
    int y_w = 5;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[25] = {0,  1,  2,  3,  3,  4,  5,  6,  7,  7,  8,  9, 10,
                          11, 11, 12, 13, 14, 15, 15, 12, 13, 14, 15, 15};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("1x4x4 image, 2x2 kernel, 2 padding") {
    int c = 1;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 2;
    int stride = 1;
    int y_h = 7;
    int y_w = 7;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[49] = {0,  0,  0, 0,  0,  0,  0,  0,  0, 1, 2,  3,  3,
                          0,  0,  4, 5,  6,  7,  7,  0,  0, 8, 9,  10, 11,
                          11, 0,  0, 12, 13, 14, 15, 15, 0, 0, 12, 13, 14,
                          15, 15, 0, 0,  0,  0,  0,  0,  0, 0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("1x4x4 image, 2x2 kernel, 2 stride") {
    int c = 1;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 0;
    int stride = 2;
    int y_h = 2;
    int y_w = 2;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[4] = {5.0, 7.0, 13.0, 15.0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("1x4x4 image, 2x2 kernel, 1 padding, 2 stride") {
    int c = 1;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 1;
    int stride = 2;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[9] = {0.0, 2.0, 3.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0};
    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("2x4x4 image, 2x2 kernel, 1 padding") {
    int c = 2;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 1;
    int stride = 1;
    int y_h = 5;
    int y_w = 5;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[50] = {0,  1,  2,  3,  3,  4,  5,  6,  7,  7,  8,  9,  10,
                          11, 11, 12, 13, 14, 15, 15, 12, 13, 14, 15, 15,

                          16, 17, 18, 19, 19, 20, 21, 22, 23, 23, 24, 25, 26,
                          27, 27, 28, 29, 30, 31, 31, 28, 29, 30, 31, 31};

    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }

  SECTION("2x4x4 image, 2x2 kernel, 1 padding, 2 stride") {
    int c = 2;
    int x_h = 4;
    int x_w = 4;
    int k = 2;
    int pad = 1;
    int stride = 2;
    int y_h = 3;
    int y_w = 3;
    float *x = new float[c * x_h * x_w];
    float *y = new float[c * y_h * y_w];
    float expected[18] = {0.0,  2.0,  3.0,  8.0,  10.0, 11.0, 12.0, 14.0, 15.0,
                          16.0, 18.0, 19.0, 24.0, 26.0, 27.0, 28.0, 30.0, 31.0};

    array_arange(x, c * x_h * x_w);
    array_zeros(y, c * y_h * y_w);

    inference_engine::backend::max_pool(c, x_h, x_w, y_h, y_w, k, pad, stride,
                                        x, y);

    REQUIRE(inference_engine::test::assert_array_eq_float(y, expected,
                                                          c * y_h * y_w));
    delete[] x;
    delete[] y;
  }
}

TEST_CASE("drop_out") {
  SECTION("1x100x100 image, ratio 0.5") {
    int c = 1;
    int h = 100;
    int w = 100;
    float ratio = 0.5;
    float *x = new float[c * h * w];
    float *y = new float[c * h * w];
    float *mask = new float[c * h * w];
    array_full(x, c * h * w, 2.0);
    array_zeros(y, c * h * w);
    array_zeros(mask, c * h * w);
    inference_engine::backend::drop_out(c, h, w, ratio, x, y, mask);

    int zero_count = 0;
    int as_is_count = 0;
    int other_count = 0;

    for (int i = 0; i < c * h * w; ++i) {
      if (y[i] == 0) {
        ++zero_count;
      } else if (y[i] == 2.0) {
        ++as_is_count;
      } else {
        ++other_count;
      }
    }

    REQUIRE(other_count == 0);
    REQUIRE(zero_count + as_is_count == c * h * w);
    REQUIRE(c * h * w * (ratio - 0.05) < zero_count);
    REQUIRE(zero_count < c * h * w * (ratio + 0.05));
  }

  SECTION("3x100x100 image, ratio 0.5") {
    int c = 3;
    int h = 100;
    int w = 100;
    float ratio = 0.5;
    float *x = new float[c * h * w];
    float *y = new float[c * h * w];
    float *mask = new float[c * h * w];
    array_full(x, c * h * w, 2.0);
    array_zeros(y, c * h * w);
    array_zeros(mask, c * h * w);
    inference_engine::backend::drop_out(c, h, w, ratio, x, y, mask);

    int zero_count = 0;
    int as_is_count = 0;
    int other_count = 0;

    for (int i = 0; i < c * h * w; ++i) {
      if (y[i] == 0) {
        ++zero_count;
      } else if (y[i] == 2.0) {
        ++as_is_count;
      } else {
        ++other_count;
      }
    }

    REQUIRE(other_count == 0);
    REQUIRE(zero_count + as_is_count == c * h * w);
    REQUIRE(c * h * w * (ratio - 0.05) < zero_count);
    REQUIRE(zero_count < c * h * w * (ratio + 0.05));
  }

  SECTION("3x100x100 image, ratio 0.3") {
    int c = 3;
    int h = 100;
    int w = 100;
    float ratio = 0.3;
    float *x = new float[c * h * w];
    float *y = new float[c * h * w];
    float *mask = new float[c * h * w];
    array_full(x, c * h * w, 2.0);
    array_zeros(y, c * h * w);
    array_zeros(mask, c * h * w);
    inference_engine::backend::drop_out(c, h, w, ratio, x, y, mask);

    int zero_count = 0;
    int as_is_count = 0;
    int other_count = 0;

    for (int i = 0; i < c * h * w; ++i) {
      if (y[i] == 0) {
        ++zero_count;
      } else if (y[i] == 2.0) {
        ++as_is_count;
      } else {
        ++other_count;
      }
    }

    REQUIRE(other_count == 0);
    REQUIRE(zero_count + as_is_count == c * h * w);
    REQUIRE(c * h * w * (ratio - 0.05) < zero_count);
    REQUIRE(zero_count < c * h * w * (ratio + 0.05));
  }
}

TEST_CASE("softmax") {
  SECTION("case 1") {
    float x[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
    float y[10];

    inference_engine::backend::softmax(10, x, y);

    float expected[10] = {0.08510981, 0.09406088, 0.10395335, 0.11488622,
                          0.12696892, 0.11488622, 0.10395335, 0.09406088,
                          0.08510981, 0.07701054};

    constexpr float e = std::numeric_limits<float>::epsilon();
    for (int i = 0; i < 10; ++i) {
      REQUIRE(std::abs(expected[i] - y[i]) < e);
    }
  }
}