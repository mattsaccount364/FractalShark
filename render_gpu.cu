#include <stdio.h>
#include <iostream>

#include "render_gpu.h"
#include "dbldbl.cuh"
#include "dblflt.cuh"
#include "../QuadDouble/gqd_basic.cuh"
#include "../QuadFloat/gqf_basic.cuh"

// Match in Fractal.cpp
constexpr static auto NB_THREADS_W = 16;
constexpr static auto NB_THREADS_H = 8;


__global__
void mandel_4x_float(int* iter_matrix,
    int width,
    int height,
    GQF::gqf_real cx,
    GQF::gqf_real cy,
    GQF::gqf_real dx,
    GQF::gqf_real dy,
    int n_iterations)
{
    using namespace GQF;
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    // Approach 2
    //// For reference
    ////{
    ////    x = 0;
    ////    y = 0;
    ////    float zrsqr = x * x;
    ////    float zisqr = y * y;
    ////    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    ////    {
    ////        y = x * y;
    ////        y += y; // Multiply by two
    ////        y += y0;
    ////        x = zrsqr - zisqr + x0;
    ////        zrsqr = x * x;
    ////        zisqr = y * y;
    ////        iter++;
    ////    }
    ////}

    int iter = 0;
    gqf_real x = make_qf(0.0f, 0.0f, 0.0f, 0.0f);
    gqf_real y = make_qf(0.0f, 0.0f, 0.0f, 0.0f);

    gqf_real y0;
    gqf_real Y_QF = make_qf(Y, 0.0f, 0.0f, 0.0f);
    y0 = cy + dy * Y_QF;

    gqf_real x0;
    gqf_real X_QF = make_qf(X, 0.0f, 0.0f, 0.0f);
    x0 = cx + dx * X_QF;

    gqf_real four;
    four = make_qf(4.0f, 0.0f, 0.0f, 0.0f);

    gqf_real zrsqr = sqr(x);
    gqf_real zisqr = sqr(y);
    while (zrsqr + zisqr <= four && iter < n_iterations)
    {
        y = x * y;
        y = mul_pwr2(y, 2.0f); // Multiply by two
        y = y + y0;
        x = zrsqr - zisqr + x0;
        zrsqr = sqr(x);
        zisqr = sqr(y);
        iter++;
    }

    iter_matrix[idx] = iter;
}

__global__
void mandel_4x_double(int* iter_matrix,
    int width,
    int height,
    GQD::gqd_real cx,
    GQD::gqd_real cy,
    GQD::gqd_real dx,
    GQD::gqd_real dy,
    int n_iterations)
{
    using namespace GQD;
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    // Approach 2
    //// For reference
    ////{
    ////    x = 0;
    ////    y = 0;
    ////    float zrsqr = x * x;
    ////    float zisqr = y * y;
    ////    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    ////    {
    ////        y = x * y;
    ////        y += y; // Multiply by two
    ////        y += y0;
    ////        x = zrsqr - zisqr + x0;
    ////        zrsqr = x * x;
    ////        zisqr = y * y;
    ////        iter++;
    ////    }
    ////}

    int iter = 0;
    gqd_real x = make_qd(0, 0, 0, 0);
    gqd_real y = make_qd(0, 0, 0, 0);
    gqd_real y0;
    y0 = cy + dy * Y;

    gqd_real x0;
    x0 = cx + dx * X;

    gqd_real zrsqr = x * x;
    gqd_real zisqr = y * y;
    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    {
        y = x * y;
        y = y * 2.0; // Multiply by two
        y = y + y0;
        x = zrsqr - zisqr + x0;
        zrsqr = x * x;
        zisqr = y * y;
        iter++;
    }

    iter_matrix[idx] = iter;
}


/*
* 
* Nvidia double double library (128-bit precision)

We have released a library that contains code for negation, addition, subtraction, multiplication, division, and square root of double-double operands using a simple C-style interface.

Developers whose applications require precision beyond double precision will likely find this helpful, as double-double offers almost twice the precision of double precision.

It is available in the CUDA Registered Developer Page. The tar file also contains a simple example ( solution of a quadratic equation with different precisions)

$ ./example_dd

Solving quadratic equation with a = 1 b = -100000 c = 1

Using double precision (std. quadratic formula):
x1 = 9.99999999900e+04 ax1**2+bx1+c = 0.00000000000e+00
x2 = 1.00000033854e-05 ax2**2+bx2+c =-3.38435755864e-07

Using double-double (std. quadratic formula):
x1 = 9.99999999900e+04 ax1**2+bx1+c = 0.00000000000e+00
x2 = 1.00000000010e-05 ax2**2+bx2+c = 0.00000000000e+00

Using double precision (more robust formula):
x1 = 9.99999999900e+04 ax1**2+bx1+c = 0.00000000000e+00
x2 = 1.00000000010e-05 ax2**2+bx2+c = 0.00000000000e+00

*/

__global__
void mandel_2x_double(int* iter_matrix,
    int width,
    int height,
    dbldbl cx,
    dbldbl cy,
    dbldbl dx,
    dbldbl dy,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    //// Approach 1
    // TODO need to take dbldbl as parameters to this
    // convert boost high precision to dbldbl?

    dbldbl cx2 = add_double_to_dbldbl(cx.y, cx.x);
    dbldbl cy2 = add_double_to_dbldbl(cy.y, cy.x);
    
    dbldbl dx2 = add_double_to_dbldbl(dx.y, dx.x);
    dbldbl dy2 = add_double_to_dbldbl(dy.y, dy.x);

    dbldbl X2 = add_double_to_dbldbl(X, 0);
    dbldbl Y2 = add_double_to_dbldbl(Y, 0);

    dbldbl x0;
    x0 = add_dbldbl(cx2, mul_dbldbl(dx2, X2));

    dbldbl y0 = add_dbldbl(cy2, mul_dbldbl(dy2, Y2));

    dbldbl x = add_double_to_dbldbl(0,0);
    dbldbl y = add_double_to_dbldbl(0, 0);

    int iter = 0;
    dbldbl xtemp;
    const dbldbl two = add_double_to_dbldbl(2.0, 0);

    dbldbl zrsqr = sqr_dbldbl(x);
    dbldbl zisqr = sqr_dbldbl(y);

    while (zrsqr.y + zisqr.y < 4.0 && iter < n_iterations)
    {
        xtemp = add_dbldbl(sub_dbldbl(zrsqr, zisqr), x0);
        y = add_dbldbl(mul_dbldbl(two, mul_dbldbl(x, y)), y0);
        x = xtemp;
        zrsqr = sqr_dbldbl(x);
        zisqr = sqr_dbldbl(y);
        iter++;
    }

    iter_matrix[idx] = iter;

    // // Approach 2
    // // For reference
    //{
    //    x = 0;
    //    y = 0;
    //    float zrsqr = x * x;
    //    float zisqr = y * y;
    //    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    //    {
    //        y = x * y;
    //        y += y; // Multiply by two
    //        y += y0;
    //        x = zrsqr - zisqr + x0;
    //        zrsqr = x * x;
    //        zisqr = y * y;
    //        iter++;
    //    }
    //}

    //dbldbl cx2 = add_double_to_dbldbl(cx.y, cx.x);
    //dbldbl cy2 = add_double_to_dbldbl(cy.y, cy.x);

    //dbldbl dx2 = add_double_to_dbldbl(dx.y, dx.x);
    //dbldbl dy2 = add_double_to_dbldbl(dy.y, dy.x);

    //dbldbl X2 = add_double_to_dbldbl(X, 0);
    //dbldbl Y2 = add_double_to_dbldbl(Y, 0);

    //dbldbl x0;
    //x0 = add_dbldbl(cx2, mul_dbldbl(dx2, X2));

    //dbldbl y0;
    //y0 = add_dbldbl(cy2, mul_dbldbl(dy2, Y2));

    //dbldbl x = add_double_to_dbldbl(0, 0);
    //dbldbl y = add_double_to_dbldbl(0, 0);

    //dbldbl two;
    //two = add_double_to_dbldbl(2.0, 0);

    //int iter = 0;
    //dbldbl zrsqr = mul_dbldbl(x, x);
    //dbldbl zisqr = mul_dbldbl(y, y);

    //while (get_dbldbl_head(add_dbldbl(zrsqr, zisqr)) < 4.0 && iter < n_iterations)
    //{
    //    y = mul_dbldbl(x, y);
    //    y = shiftleft_dbldbl(y);
    //    y = add_dbldbl(y, y0);
    //    x = sub_dbldbl(zrsqr, zisqr);
    //    x = add_dbldbl(x, x0);
    //    zrsqr = sqr_dbldbl(x);
    //    zisqr = sqr_dbldbl(y);
    //    iter++;
    //}

    //iter_matrix[idx] = iter;
}

template<int iteration_precision>
__global__
void mandel_1x_double(int* iter_matrix,
                 int width,
                 int height,
                 double cx,
                 double cy,
                 double dx,
                 double dy,
                 int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    double x0 = cx + dx * X;
    double y0 = cy + dy * Y;

    double x = 0.0;
    double y = 0.0;

    n_iterations -= iteration_precision - 1;

    int iter = 0;
    double xtemp, xtemp2;
    double ytemp;

    auto MANDEL_1X_DOUBLE = [&]() {
        ytemp = __fma_rd(-y, y, x0);
        xtemp = __fma_rd(x, x, ytemp);
        xtemp2 = 2.0 * x;
        y = __fma_rd(xtemp2, y, y0);
        x = xtemp;
    };

    while (x * x + y * y < 4.0 && iter < n_iterations)
    {
        //xtemp = x * x - y * y + x0;
        //y = 2.0 * x * y + y0;
        //x = xtemp;
        //iter++;

        if (iteration_precision == 1) {
            MANDEL_1X_DOUBLE();
            iter++;
        }
        else if (iteration_precision == 2) {
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            iter += 2;
        }
        else if (iteration_precision == 4) {
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            iter += 4;
        }
        else if (iteration_precision == 8) {
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            iter += 8;
        }
        else if (iteration_precision == 16) {
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            MANDEL_1X_DOUBLE();
            iter += 16;
        }
    }

    iter_matrix[idx] = iter;
}

template<int iteration_precision>
__global__
void mandel_2x_float(int* iter_matrix,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    ////// Approach 1
    //// TODO need to take dblflt as parameters to this
    //// convert boost high precision to dblflt?

    //dblflt cx2 = add_float_to_dblflt(cx.y, cx.x);
    //dblflt cy2 = add_float_to_dblflt(cy.y, cy.x);

    //dblflt dx2 = add_float_to_dblflt(dx.y, dx.x);
    //dblflt dy2 = add_float_to_dblflt(dy.y, dy.x);

    //dblflt X2 = add_float_to_dblflt(X, 0);
    //dblflt Y2 = add_float_to_dblflt(Y, 0);

    //dblflt x0;
    //x0 = add_dblflt(cx2, mul_dblflt(dx2, X2));

    //dblflt y0;
    //y0 = add_dblflt(cy2, mul_dblflt(dy2, Y2));

    //dblflt x = add_float_to_dblflt(0, 0);
    //dblflt y = add_float_to_dblflt(0, 0);

    //int iter = 0;
    //dblflt xtemp;
    //dblflt two;
    //two = add_float_to_dblflt(2.0, 0);

    //dblflt xtemp2, xtemp3;
    //dblflt ytemp2, ytemp3;
    //// while (x * x + y * y < 4.0 && iter < n_iterations)
    //while (get_dblflt_head(add_dblflt(mul_dblflt(x, x), mul_dblflt(y, y))) < 4.0 && iter < n_iterations)
    //{
    //    //xtemp = x * x - y * y + x0;
    //    //y = 2.0 * x * y + y0;
    //    //x = xtemp;
    //    //iter++;

    //    //xtemp = add_dblflt(sub_dblflt(mul_dblflt(x, x), mul_dblflt(y, y)), x0);
    //    //y = add_dblflt(mul_dblflt(two, mul_dblflt(x, y)), y0);
    //    //x = xtemp;
    //    //iter++;

    //    xtemp2 = mul_dblflt(x, x);
    //    ytemp2 = mul_dblflt(y, y);
    //    ytemp3 = sub_dblflt(xtemp2, ytemp2);
    //    xtemp = add_dblflt(ytemp3, x0);
    //    y = add_dblflt(mul_dblflt(two, mul_dblflt(x, y)), y0);
    //    x = xtemp;
    //    iter++;
    //}


    // Approach 2
    // // For reference
    //{
    //    x = 0;
    //    y = 0;
    //    float zrsqr = x * x;
    //    float zisqr = y * y;
    //    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    //    {
    //        y = x * y;
    //        y += y; // Multiply by two
    //        y += y0;
    //        x = zrsqr - zisqr + x0;
    //        zrsqr = x * x;
    //        zisqr = y * y;
    //        iter++;
    //    }
    //}

    dblflt cx2 = add_float_to_dblflt(cx.y, cx.x);
    dblflt cy2 = add_float_to_dblflt(cy.y, cy.x);

    dblflt dx2 = add_float_to_dblflt(dx.y, dx.x);
    dblflt dy2 = add_float_to_dblflt(dy.y, dy.x);

    dblflt X2 = add_float_to_dblflt(X, 0);
    dblflt Y2 = add_float_to_dblflt(Y, 0);

    dblflt x0;
    x0 = add_dblflt(cx2, mul_dblflt(dx2, X2));

    dblflt y0;
    y0 = add_dblflt(cy2, mul_dblflt(dy2, Y2));

    dblflt x = {};
    dblflt y = {};

    int iter = 0;
    dblflt zrsqr = {};
    dblflt zisqr = {};

    auto MANDEL_2X_FLOAT = [&]() {
        y = mul_dblflt2x(x, y);
        y = add_dblflt(y, y0);
        x = sub_dblflt(zrsqr, zisqr);
        x = add_dblflt(x, x0);
        zrsqr = sqr_dblflt(x);
        zisqr = sqr_dblflt(y);
    };

    while (zrsqr.y + zisqr.y < 4.0f && iter < n_iterations)
    {
        if (iteration_precision == 1) {
            MANDEL_2X_FLOAT();
            iter++;
        }
        else if (iteration_precision == 2) {
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            iter += 2;
        }
        else if (iteration_precision == 4) {
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            iter += 4;
        }
        else if (iteration_precision == 8) {
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            iter += 8;
        }
        else if (iteration_precision == 16) {
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            MANDEL_2X_FLOAT();
            iter += 16;
        }
    }

    iter_matrix[idx] = iter;
}

template<int iteration_precision>
__global__
void mandel_1x_float(int* iter_matrix,
    int width,
    int height,
    float cx,
    float cy,
    float dx,
    float dy,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;

    float x0 = cx + dx * X;
    float y0 = cy + dy * Y;

    float x = 0.0;
    float y = 0.0;

    n_iterations -= iteration_precision - 1;

    int iter = 0;

    //{
    //    x = 0;
    //    y = 0;
    //    float zrsqr = x * x;
    //    float zisqr = y * y;
    //    while (zrsqr + zisqr <= 4.0 && iter < n_iterations)
    //    {
    //        y = x * y;
    //        y += y; // Multiply by two
    //        y += y0;
    //        x = zrsqr - zisqr + x0;
    //        zrsqr = x * x;
    //        zisqr = y * y;
    //        iter++;
    //    }
    //}

    float xtemp, xtemp2;
    float ytemp;

    auto MANDEL_1X_FLOAT = [&]() {
        ytemp = __fmaf_rd(-y, y, x0);
        xtemp = __fmaf_rd(x, x, ytemp);
        xtemp2 = 2.0f * x;
        y = __fmaf_rd(xtemp2, y, y0);
        x = xtemp;
    };

    while (x * x + y * y < 4.0 && iter < n_iterations)
    {
        //xtemp = x * x - y * y + x0;
        //y = 2.0 * x * y + y0;
        //x = xtemp;
        //iter++;

        if (iteration_precision == 1) {
            MANDEL_1X_FLOAT();
            iter++;
        }
        else if (iteration_precision == 2) {
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            iter += 2;
        }
        else if (iteration_precision == 4) {
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            iter += 4;
        }
        else if (iteration_precision == 8) {
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            iter += 8;
        }
        else if (iteration_precision == 16) {
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            MANDEL_1X_FLOAT();
            iter += 16;
        }
    }

    iter_matrix[idx] = iter;
}

__global__
void mandel_float_double_combo(int* iter_matrix,
    uint8_t *ratio_matrix,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    double cx_orig,
    double cy_orig,
    double dx_orig,
    double dy_orig,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * (height - Y - 1) + X;
    size_t ratio_idx = blockIdx.y * width + blockIdx.x;
    int iter = 0;

    if (ratio_matrix[ratio_idx] == 'f') {

        double x0 = cx_orig + dx_orig * X;
        double y0 = cy_orig + dy_orig * Y;

        double x = 0.0;
        double y = 0.0;

        double xtemp, xtemp2;
        double ytemp;
        while (x * x + y * y < 4.0 && iter < n_iterations)
        {
            //xtemp = x * x - y * y + x0;
            //y = 2.0 * x * y + y0;
            //x = xtemp;
            //iter++;

            ytemp = __fma_rd(-y, y, x0);
            xtemp = __fma_rd(x, x, ytemp);
            xtemp2 = 2.0 * x;
            y = __fma_rd(xtemp2, y, y0);
            x = xtemp;

            iter++;
        }

        iter += 100;
    } 
    else if (ratio_matrix[ratio_idx] == 'D')
    {
        // TODO need to take dblflt as parameters to this
        // convert boost high precision to dblflt?

        dblflt cx2 = add_float_to_dblflt(cx.y, cx.x);
        dblflt cy2 = add_float_to_dblflt(cy.y, cy.x);

        dblflt dx2 = add_float_to_dblflt(dx.y, dx.x);
        dblflt dy2 = add_float_to_dblflt(dy.y, dy.x);

        dblflt X2 = add_float_to_dblflt(X, 0);
        dblflt Y2 = add_float_to_dblflt(Y, 0);

        dblflt x0;
        x0 = add_dblflt(cx2, mul_dblflt(dx2, X2));

        dblflt y0 =
            y0 = add_dblflt(cy2, mul_dblflt(dy2, Y2));

        dblflt x = add_float_to_dblflt(0, 0);
        dblflt y = add_float_to_dblflt(0, 0);

        dblflt xtemp;
        dblflt two;
        two = add_float_to_dblflt(2.0, 0);

        while (get_dblflt_head(add_dblflt(mul_dblflt(x, x), mul_dblflt(y, y))) < 4.0 && iter < n_iterations)
        {
            xtemp = add_dblflt(sub_dblflt(mul_dblflt(x, x), mul_dblflt(y, y)), x0);
            y = add_dblflt(mul_dblflt(two, mul_dblflt(x, y)), y0);
            x = xtemp;
            iter++;
        }
    }
    else {
        iter = 100;
    }
    
    iter_matrix[idx] = iter;
}

GPURenderer::~GPURenderer() {
    ResetRatioMemory();
}

void GPURenderer::ResetRatioMemory() {
    if (ratioMemory_cu != nullptr) {
        cudaFree(ratioMemory_cu);
        ratioMemory_cu = nullptr;
    }
}

void GPURenderer::SetRatioMemory(uint8_t* ratioMemory, size_t MaxFractalSize) {
    cudaError_t err;
    ResetRatioMemory();

    if (ratioMemory && ratioMemory_cu == nullptr) {
        err = cudaMalloc(&ratioMemory_cu, MaxFractalSize * MaxFractalSize * sizeof(uint8_t));
        if (err != cudaSuccess) {
            ratioMemory_cu = nullptr;
            return;
        }

        err = cudaMemcpy(ratioMemory_cu, ratioMemory, MaxFractalSize * MaxFractalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ratioMemory_cu);
            ratioMemory_cu = nullptr;
            return;
        }
    }
}

void GPURenderer::render_gpu2(
    uint32_t algorithm,
    uint32_t* buffer,
    size_t MaxFractalSize,
    size_t width,
    size_t height,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    int n_iterations,
    int antialiasing,
    int iteration_precision)
{
    unsigned int local_width = (unsigned int)width * antialiasing;
    unsigned int local_height = (unsigned int)height * antialiasing;
    unsigned int w_block = local_width / NB_THREADS_W + (local_width % NB_THREADS_W != 0);
    unsigned int h_block = local_height / NB_THREADS_H + (local_height % NB_THREADS_H != 0);
    size_t N_cu = w_block * NB_THREADS_W * h_block * NB_THREADS_H;
    int* iter_matrix = new int[N_cu];
    int* iter_matrix_cu;
    cudaError_t err;

    err = cudaMalloc(&iter_matrix_cu, N_cu * sizeof(int));
    if (err != cudaSuccess) {
        delete[] iter_matrix;
        return;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == 'f') {
        switch (iteration_precision) {
        case 1:
            mandel_1x_double<1> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
                n_iterations);
            break;
        case 4:
            mandel_1x_double<4> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
                n_iterations);
            break;
        case 8:
            mandel_1x_double<8> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
                n_iterations);
            break;
        case 16:
            mandel_1x_double<16> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
                n_iterations);
            break;
        default:
            break;
        }
    }
    else if (algorithm == 'd') {
        dbldbl cx2{ cx.dbl.head, cx.dbl.tail };
        dbldbl cy2{ cy.dbl.head, cy.dbl.tail };
        dbldbl dx2{ dx.dbl.head, dx.dbl.tail };
        dbldbl dy2{ dy.dbl.head, dy.dbl.tail };

        mandel_2x_double << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            local_width, local_height, cx2, cy2, dx2, dy2,
            n_iterations);
    }
    else if (algorithm == 'q') {
        using namespace GQD;
        gqd_real cx2;
        cx2 = make_qd(cx.qdbl.v1, cx.qdbl.v2, cx.qdbl.v3, cx.qdbl.v4);

        gqd_real cy2;
        cy2 = make_qd(cy.qdbl.v1, cy.qdbl.v2, cy.qdbl.v3, cy.qdbl.v4);

        gqd_real dx2;
        dx2 = make_qd(dx.qdbl.v1, dx.qdbl.v2, dx.qdbl.v3, dx.qdbl.v4);

        gqd_real dy2;
        dy2 = make_qd(dy.qdbl.v1, dy.qdbl.v2, dy.qdbl.v3, dy.qdbl.v4);

        mandel_4x_double << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            local_width, local_height, cx2, cy2, dx2, dy2,
            n_iterations);
    }
    else if (algorithm == 'F') {
        switch (iteration_precision) {
        case 1:
            mandel_1x_float<1> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
                n_iterations);
            break;
        case 4:
            mandel_1x_float<4> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
                n_iterations);
            break;
        case 8:
            mandel_1x_float<8> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
                n_iterations);
            break;
        case 16:
            mandel_1x_float<16> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
                n_iterations);
            break;
        default:
            break;
        }
    }
    else if (algorithm == 'D') {
        dblflt cx2{ cx.flt.head, cx.flt.tail };
        dblflt cy2{ cy.flt.head, cy.flt.tail };
        dblflt dx2{ dx.flt.head, dx.flt.tail };
        dblflt dy2{ dy.flt.head, dy.flt.tail };

        switch (iteration_precision) {
        case 1:
            mandel_2x_float<1> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
            break;
        case 4:
            mandel_2x_float<4> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
            break;
        case 8:
            mandel_2x_float<8> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
            break;
        case 16:
            mandel_2x_float<16> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
            break;
        default:
            break;
        }
    }
    else if (algorithm == 'Q') {
        using namespace GQF;
        gqf_real cx2;
        cx2 = make_qf(cx.qflt.v1, cx.qflt.v2, cx.qflt.v3, cx.qflt.v4);

        gqf_real cy2;
        cy2 = make_qf(cy.qflt.v1, cy.qflt.v2, cy.qflt.v3, cy.qflt.v4);

        gqf_real dx2;
        dx2 = make_qf(dx.qflt.v1, dx.qflt.v2, dx.qflt.v3, dx.qflt.v4);

        gqf_real dy2;
        dy2 = make_qf(dy.qflt.v1, dy.qflt.v2, dy.qflt.v3, dy.qflt.v4);

        mandel_4x_float << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            local_width, local_height, cx2, cy2, dx2, dy2,
            n_iterations);
    }
    else if (algorithm == 'B') {
        dblflt cx2{ cx.flt.head, cx.flt.tail };
        dblflt cy2{ cy.flt.head, cy.flt.tail };
        dblflt dx2{ dx.flt.head, dx.flt.tail };
        dblflt dy2{ dy.flt.head, dy.flt.tail };

        mandel_float_double_combo << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            ratioMemory_cu, local_width, local_height, cx2, cy2, dx2, dy2,
            cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            n_iterations);
    }
    else {
        delete[] iter_matrix;
        cudaFree(iter_matrix_cu);
        return;
    }

    err = cudaMemcpy(iter_matrix, iter_matrix_cu, N_cu * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] iter_matrix;
        cudaFree(iter_matrix_cu);
        return;
    }

    size_t aax, aay;
    double temp;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {

            buffer[y * MaxFractalSize + x] = 0;
            temp = 0.0;
            for (aay = y * antialiasing; aay < y * antialiasing + antialiasing; aay++) {
                for (aax = x * antialiasing; aax < x * antialiasing + antialiasing; aax++) {
                    temp += iter_matrix[aay * local_width + aax];
                }
            }

            temp /= antialiasing * antialiasing;
            buffer[y * MaxFractalSize + x] = (int) temp;
        }
    }

    delete[] iter_matrix;
    cudaFree(iter_matrix_cu);
}