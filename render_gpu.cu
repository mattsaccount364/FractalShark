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

__global__
void mandel_1x_double_perturb_glitchy(int* iter_matrix,
    double* results_x,
    double* results_x2,
    double* results_y,
    double* results_y2,
    double* results_tolerancy,
    size_t sz,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    int iter = 0;

    double deltaReal = dx * X - centerX;
    double deltaImaginary = -dy * Y - centerY;

    double DeltaSub0X = deltaReal;
    double DeltaSub0Y = deltaImaginary;
    double DeltaSubNX, DeltaSubNY;

    DeltaSubNX = DeltaSub0X;
    DeltaSubNY = DeltaSub0Y;

    double zn_size;
    bool glitched = false;

    do {
        const double DeltaSubNXOrig = DeltaSubNX;
        const double DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX = DeltaSubNXOrig * (results_x2[iter] + DeltaSubNXOrig) -
            DeltaSubNYOrig * (results_y2[iter] + DeltaSubNYOrig);
        DeltaSubNX += DeltaSub0X;

        DeltaSubNY = DeltaSubNXOrig * (results_y2[iter] + DeltaSubNYOrig) +
            DeltaSubNYOrig * (results_x2[iter] + DeltaSubNXOrig);
        DeltaSubNY += DeltaSub0Y;

        ++iter;

        const double tempX = results_x[iter] + DeltaSubNX;
        const double tempY = results_y[iter] + DeltaSubNY;
        zn_size = tempX * tempX + tempY * tempY;

        if (zn_size < results_tolerancy[iter]) {
            glitched = true;
            break;
        }
    } while (zn_size < 256 && iter < n_iterations);

    if (glitched == false) {
        iter_matrix[idx] = iter;
    }
    else {
        iter_matrix[idx] = 0;
    }
}

__global__
void mandel_1x_double_perturb_bla(int* iter_matrix,
    double* results_x,
    double* results_x2,
    double* results_y,
    double* results_y2,
    double* results_tolerancy,
    size_t sz,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    double deltaReal = dx * X - centerX;
    double deltaImaginary = -dy * Y - centerY;

    double DeltaSub0X = deltaReal;
    double DeltaSub0Y = deltaImaginary;
    double DeltaSubNX = 0;
    double DeltaSubNY = 0;

    while (iter < n_iterations) {
        const double DeltaSubNXOrig = DeltaSubNX;
        const double DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX =
            DeltaSubNXOrig * (results_x2[RefIteration] + DeltaSubNXOrig) -
            DeltaSubNYOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY =
            DeltaSubNXOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
            DeltaSubNYOrig * (results_x2[RefIteration] + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;

        const double tempZX = results_x[RefIteration] + DeltaSubNX;
        const double tempZY = results_y[RefIteration] + DeltaSubNY;
        const double zn_size = tempZX * tempZX + tempZY * tempZY;
        const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > 256) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == sz - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
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

__global__
void mandel_2x_float_perturb_glitchy_setup(
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    dblflt* results_tolerancy,
    size_t sz)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0)
        return;

    for (size_t i = 0; i < sz; i++) {
        results_x[i] = add_float_to_dblflt(results_x[i].y, results_x[i].x);
        results_x2[i] = add_float_to_dblflt(results_x2[i].y, results_x2[i].x);
        results_y[i] = add_float_to_dblflt(results_y[i].y, results_y[i].x);
        results_y2[i] = add_float_to_dblflt(results_y2[i].y, results_y2[i].x);
        results_tolerancy[i] = add_float_to_dblflt(results_tolerancy[i].y, results_tolerancy[i].x);
    }
}

__global__
void mandel_2x_float_perturb_glitchy(int* iter_matrix,
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    dblflt* results_tolerancy,
    size_t sz,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    dblflt centerX,
    dblflt centerY,
    int n_iterations)
{

    //int X = blockIdx.x * blockDim.x + threadIdx.x;
    //int Y = blockIdx.y * blockDim.y + threadIdx.y;

    //if (X >= width || Y >= height)
    //    return;

    ////size_t idx = width * (height - Y - 1) + X;
    //size_t idx = width * Y + X;

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    //int iter = 0;

    //double deltaReal = dx * X - centerX;
    //double deltaImaginary = -dy * Y - centerY;

    //double DeltaSub0X = deltaReal;
    //double DeltaSub0Y = deltaImaginary;
    //double DeltaSubNX, DeltaSubNY;

    //DeltaSubNX = DeltaSub0X;
    //DeltaSubNY = DeltaSub0Y;

    //double zn_size;
    //bool glitched = false;

    //do {
    //    const double DeltaSubNXOrig = DeltaSubNX;
    //    const double DeltaSubNYOrig = DeltaSubNY;

    //    DeltaSubNX = DeltaSubNXOrig * (results_x2[iter] + DeltaSubNXOrig) -
    //        DeltaSubNYOrig * (results_y2[iter] + DeltaSubNYOrig);
    //    DeltaSubNX += DeltaSub0X;

    //    DeltaSubNY = DeltaSubNXOrig * (results_y2[iter] + DeltaSubNYOrig) +
    //        DeltaSubNYOrig * (results_x2[iter] + DeltaSubNXOrig);
    //    DeltaSubNY += DeltaSub0Y;

    //    ++iter;

    //    const double tempX = results_x[iter] + DeltaSubNX;
    //    const double tempY = results_y[iter] + DeltaSubNY;
    //    zn_size = tempX * tempX + tempY * tempY;

    //    if (zn_size < results_tolerancy[iter]) {
    //        glitched = true;
    //        break;
    //    }
    //} while (zn_size < 256 && iter < n_iterations);

    //if (glitched == false) {
    //    iter_matrix[idx] = iter;
    //}
    //else {
    //    iter_matrix[idx] = 0;
    //}

    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    int iter = 0;

    dblflt X2 = add_float_to_dblflt(X, 0);
    dblflt Y2 = add_float_to_dblflt(Y, 0);
    dblflt MinusY2 = add_float_to_dblflt(-Y, 0);

    dblflt deltaReal = sub_dblflt(mul_dblflt(dx, X2), centerX);
    dblflt deltaImaginary = sub_dblflt(mul_dblflt(dy, MinusY2), centerY);

    dblflt DeltaSub0X = deltaReal;
    dblflt DeltaSub0Y = deltaImaginary;
    dblflt DeltaSubNX, DeltaSubNY;

    DeltaSubNX = DeltaSub0X;
    DeltaSubNY = DeltaSub0Y;

    dblflt zn_size;
    bool glitched = false;

    do {
        const dblflt DeltaSubNXOrig = DeltaSubNX;
        const dblflt DeltaSubNYOrig = DeltaSubNY;

        const dblflt tempTermX1 = add_dblflt(results_x2[iter], DeltaSubNXOrig);
        const dblflt tempTermX2 = add_dblflt(results_y2[iter], DeltaSubNYOrig);

        DeltaSubNX =
            sub_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX1),
                mul_dblflt(DeltaSubNYOrig, tempTermX2)
                );
        DeltaSubNX = add_dblflt(DeltaSubNX, DeltaSub0X);

        DeltaSubNY =
            add_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX2),
                mul_dblflt(DeltaSubNYOrig, tempTermX1)
            );
        DeltaSubNY = add_dblflt(DeltaSubNY, DeltaSub0Y);

        ++iter;

        const dblflt tempX = add_dblflt(results_x[iter], DeltaSubNX);
        const dblflt tempY = add_dblflt(results_y[iter], DeltaSubNY);
        zn_size = add_dblflt(sqr_dblflt(tempX), sqr_dblflt(tempY));

        if (zn_size.y < results_tolerancy[iter].y) {
            glitched = true;
            break;
        }
    } while (zn_size.y < 256 && iter < n_iterations);

    if (glitched == false) {
        iter_matrix[idx] = iter;
    }
    else {
        iter_matrix[idx] = 0;
    }
}

__global__
void mandel_2x_float_perturb_bla_setup(
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    dblflt* results_tolerancy,
    size_t sz)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0)
        return;

    for (size_t i = 0; i < sz; i++) {
        results_x[i] = add_float_to_dblflt(results_x[i].y, results_x[i].x);
        results_x2[i] = add_float_to_dblflt(results_x2[i].y, results_x2[i].x);
        results_y[i] = add_float_to_dblflt(results_y[i].y, results_y[i].x);
        results_y2[i] = add_float_to_dblflt(results_y2[i].y, results_y2[i].x);
        results_tolerancy[i] = add_float_to_dblflt(results_tolerancy[i].y, results_tolerancy[i].x);
    }
}

__global__
void mandel_2x_float_perturb_bla(int* iter_matrix,
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    dblflt* results_tolerancy,
    size_t sz,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    dblflt centerX,
    dblflt centerY,
    int n_iterations)
{

    //int X = blockIdx.x * blockDim.x + threadIdx.x;
    //int Y = blockIdx.y * blockDim.y + threadIdx.y;

    //if (X >= width || Y >= height)
    //    return;

    ////size_t idx = width * (height - Y - 1) + X;
    //size_t idx = width * Y + X;

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    //size_t iter = 0;
    //size_t RefIteration = 0;
    //double deltaReal = dx * X - centerX;
    //double deltaImaginary = -dy * Y - centerY;

    //double DeltaSub0X = deltaReal;
    //double DeltaSub0Y = deltaImaginary;
    //double DeltaSubNX = 0;
    //double DeltaSubNY = 0;

    //while (iter < n_iterations) {
    //    const double DeltaSubNXOrig = DeltaSubNX;
    //    const double DeltaSubNYOrig = DeltaSubNY;

    //    DeltaSubNX =
    //        DeltaSubNXOrig * (results_x2[RefIteration] + DeltaSubNXOrig) -
    //        DeltaSubNYOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
    //        DeltaSub0X;
    //    DeltaSubNY =
    //        DeltaSubNXOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
    //        DeltaSubNYOrig * (results_x2[RefIteration] + DeltaSubNXOrig) +
    //        DeltaSub0Y;

    //    ++RefIteration;

    //    const double tempZX = results_x[RefIteration] + DeltaSubNX;
    //    const double tempZY = results_y[RefIteration] + DeltaSubNY;
    //    const double zn_size = tempZX * tempZX + tempZY * tempZY;
    //    const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

    //    if (zn_size > 256) {
    //        break;
    //    }

    //    if (zn_size < normDeltaSubN ||
    //        RefIteration == sz - 1) {
    //        DeltaSubNX = tempZX;
    //        DeltaSubNY = tempZY;
    //        RefIteration = 0;
    //    }

    //    ++iter;
    //}

    //iter_matrix[idx] = (uint32_t)iter;

    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    int iter = 0;
    size_t RefIteration = 0;

    dblflt X2 = add_float_to_dblflt(X, 0);
    dblflt Y2 = add_float_to_dblflt(Y, 0);
    dblflt MinusY2 = add_float_to_dblflt(-Y, 0);

    dblflt deltaReal = sub_dblflt(mul_dblflt(dx, X2), centerX);
    dblflt deltaImaginary = sub_dblflt(mul_dblflt(dy, MinusY2), centerY);

    dblflt DeltaSub0X = deltaReal;
    dblflt DeltaSub0Y = deltaImaginary;
    dblflt DeltaSubNX, DeltaSubNY;

    DeltaSubNX = add_float_to_dblflt(0, 0);
    DeltaSubNY = add_float_to_dblflt(0, 0);

    while (iter < n_iterations) {
        const dblflt DeltaSubNXOrig = DeltaSubNX;
        const dblflt DeltaSubNYOrig = DeltaSubNY;

        const dblflt tempTermX1 = add_dblflt(results_x2[RefIteration], DeltaSubNXOrig);
        const dblflt tempTermX2 = add_dblflt(results_y2[RefIteration], DeltaSubNYOrig);

        DeltaSubNX =
            sub_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX1),
                mul_dblflt(DeltaSubNYOrig, tempTermX2)
            );
        DeltaSubNX = add_dblflt(DeltaSubNX, DeltaSub0X);

        DeltaSubNY =
            add_dblflt(
                mul_dblflt(DeltaSubNXOrig, tempTermX2),
                mul_dblflt(DeltaSubNYOrig, tempTermX1)
            );
        DeltaSubNY = add_dblflt(DeltaSubNY, DeltaSub0Y);

        ++RefIteration;

        const dblflt tempZX = add_dblflt(results_x[RefIteration], DeltaSubNX);
        const dblflt tempZY = add_dblflt(results_y[RefIteration], DeltaSubNY);
        const dblflt zn_size = add_dblflt(sqr_dblflt(tempZX), sqr_dblflt(tempZY));
        const dblflt normDeltaSubN = add_dblflt(sqr_dblflt(DeltaSubNX), sqr_dblflt(DeltaSubNY));

        if (zn_size.y > 256) {
            break;
        }

        if (zn_size.y < normDeltaSubN.y ||
            RefIteration == sz - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
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
void mandel_1x_float_perturb_bla(int* iter_matrix,
    float* results_x,
    float* results_x2,
    float* results_y,
    float* results_y2,
    float* results_tolerancy,
    size_t sz,
    int width,
    int height,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    int n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    float deltaReal = dx * X - centerX;
    float deltaImaginary = -dy * Y - centerY;

    float DeltaSub0X = deltaReal;
    float DeltaSub0Y = deltaImaginary;
    float DeltaSubNX = 0;
    float DeltaSubNY = 0;

    while (iter < n_iterations) {
        const float DeltaSubNXOrig = DeltaSubNX;
        const float DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX =
            DeltaSubNXOrig * (results_x2[RefIteration] + DeltaSubNXOrig) -
            DeltaSubNYOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY =
            DeltaSubNXOrig * (results_y2[RefIteration] + DeltaSubNYOrig) +
            DeltaSubNYOrig * (results_x2[RefIteration] + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;

        const float tempZX = results_x[RefIteration] + DeltaSubNX;
        const float tempZY = results_y[RefIteration] + DeltaSubNY;
        const float zn_size = tempZX * tempZX + tempZY * tempZY;
        const float normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > 256) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == sz - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
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

GPURenderer::GPURenderer() {
    ClearLocals();
}

GPURenderer::~GPURenderer() {
    ResetRatioMemory();
}

void GPURenderer::ResetRatioMemory() {
    if (ratioMemory_cu != nullptr) {
        cudaFree(ratioMemory_cu);
    }

    if (iter_matrix_cu != nullptr) {
        cudaFree(iter_matrix_cu);
    }

    ClearLocals();
}

void GPURenderer::ClearLocals() {
    // Assumes memory is freed
    ratioMemory_cu = nullptr;
    iter_matrix_cu = nullptr;

    width = 0;
    height = 0;
    local_width = 0;
    local_height = 0;
    w_block = 0;
    h_block = 0;
    N_cu = 0;
}

void GPURenderer::SetRatioMemory(uint8_t* ratioMemory, size_t MaxFractSize) {
    cudaError_t err;
    ResetRatioMemory();

    if (ratioMemory && ratioMemory_cu == nullptr) {
        err = cudaMalloc(&ratioMemory_cu, MaxFractSize * MaxFractSize * sizeof(uint8_t));
        if (err != cudaSuccess) {
            ratioMemory_cu = nullptr;
            return;
        }

        err = cudaMemcpy(ratioMemory_cu, ratioMemory, MaxFractSize * MaxFractSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ratioMemory_cu);
            ratioMemory_cu = nullptr;
            return;
        }
    }
}

void GPURenderer::ClearMemory() {
    if (iter_matrix_cu == nullptr) {
        return;
    }

    cudaMemset(iter_matrix_cu, 0, N_cu * sizeof(int));
}

void GPURenderer::InitializeMemory(
    size_t w,
    size_t h,
    uint32_t aa,
    size_t MaxFractSize)
{
    if ((local_width == w * aa) &&
        (local_height == h * aa)) {
        return;
    }

    width = (uint32_t)w;
    height = (uint32_t)h;
    antialiasing = aa;
    local_width = width * antialiasing;
    local_height = height * antialiasing;
    w_block = local_width / NB_THREADS_W + (local_width % NB_THREADS_W != 0);
    h_block = local_height / NB_THREADS_H + (local_height % NB_THREADS_H != 0);
    N_cu = w_block * NB_THREADS_W * h_block * NB_THREADS_H;
    MaxFractalSize = MaxFractSize;

    if (iter_matrix_cu != nullptr) {
        cudaFree(iter_matrix_cu);
    }

    cudaError_t err = cudaMallocManaged(&iter_matrix_cu, N_cu * sizeof(int), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        ClearLocals();
        return;
    }

    ClearMemory();
}

void GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    int n_iterations,
    int iteration_precision)
{
    if (iter_matrix_cu == nullptr) {
        return;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64) {
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
    else if (algorithm == RenderAlgorithm::Gpu2x64) {
        dbldbl cx2{ cx.dbl.head, cx.dbl.tail };
        dbldbl cy2{ cy.dbl.head, cy.dbl.tail };
        dbldbl dx2{ dx.dbl.head, dx.dbl.tail };
        dbldbl dy2{ dy.dbl.head, dy.dbl.tail };

        mandel_2x_double << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            local_width, local_height, cx2, cy2, dx2, dy2,
            n_iterations);
    }
    else if (algorithm == RenderAlgorithm::Gpu4x64) {
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
    else if (algorithm == RenderAlgorithm::Gpu1x32) {
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
    else if (algorithm == RenderAlgorithm::Gpu2x32) {
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
    else if (algorithm == RenderAlgorithm::Gpu4x32) {
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
    else if (algorithm == RenderAlgorithm::Blend) {
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
        return;
    }

    ExtractIters(buffer);
}

void GPURenderer::RenderPerturbGlitchy(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults* results,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    int n_iterations,
    int /*iteration_precision*/)
{
    if (iter_matrix_cu == nullptr) {
        return;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64PerturbedGlitchy) {
        double* results_x;
        double* results_y;
        double* results_x2;
        double* results_y2;
        double* results_tolerancy;

        cudaError_t err;

        err = cudaMallocManaged(&results_x, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_x2, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y2, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_tolerancy, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        cudaMemcpy(results_x, results->x.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_x2, results->x2.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_y, results->y.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_y2, results->y2.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_tolerancy, results->tolerancy.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);

        mandel_1x_double_perturb_glitchy << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        cudaFree(results_x);
        cudaFree(results_x2);
        cudaFree(results_y);
        cudaFree(results_y2);
        cudaFree(results_tolerancy);
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedGlitchy) {
        dblflt* results_x;
        dblflt* results_y;
        dblflt* results_x2;
        dblflt* results_y2;
        dblflt* results_tolerancy;

        cudaError_t err;

        err = cudaMallocManaged(&results_x, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_x2, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y2, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_tolerancy, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        dblflt cx2{ cx.flt.head, cx.flt.tail };
        dblflt cy2{ cy.flt.head, cy.flt.tail };
        dblflt dx2{ dx.flt.head, dx.flt.tail };
        dblflt dy2{ dy.flt.head, dy.flt.tail };
        dblflt centerX2{ centerX.flt.head, centerX.flt.tail };
        dblflt centerY2{ centerY.flt.head, centerY.flt.tail };

        cudaMemcpy(results_x, results->x.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_x2, results->x2.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_y, results->y.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_y2, results->y2.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_tolerancy, results->tolerancy.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);

        mandel_2x_float_perturb_glitchy_setup << <nb_blocks, threads_per_block >> > (
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size);

        mandel_2x_float_perturb_glitchy << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size,
            local_width, local_height, cx2, cy2, dx2, dy2,
            centerX2, centerY2,
            n_iterations);

        cudaFree(results_x);
        cudaFree(results_x2);
        cudaFree(results_y);
        cudaFree(results_y2);
        cudaFree(results_tolerancy);
    }

    ExtractIters(buffer);
}

void GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults* results,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    int n_iterations,
    int /*iteration_precision*/)
{
    if (iter_matrix_cu == nullptr) {
        return;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64PerturbedBLA) {
        double* results_x;
        double* results_y;
        double* results_x2;
        double* results_y2;
        double* results_tolerancy;

        cudaError_t err;

        err = cudaMallocManaged(&results_x, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_x2, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y2, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_tolerancy, results->size * sizeof(double), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        cudaMemcpy(results_x, results->x.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_x2, results->x2.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_y, results->y.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_y2, results->y2.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(results_tolerancy, results->tolerancy.doubleOnly, results->size * sizeof(double), cudaMemcpyDefault);

        mandel_1x_double_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        cudaFree(results_x);
        cudaFree(results_x2);
        cudaFree(results_y);
        cudaFree(results_y2);
        cudaFree(results_tolerancy);
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedBLA) {
        dblflt* results_x;
        dblflt* results_y;
        dblflt* results_x2;
        dblflt* results_y2;
        dblflt* results_tolerancy;

        cudaError_t err;

        err = cudaMallocManaged(&results_x, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_x2, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y2, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_tolerancy, results->size * sizeof(dblflt), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        dblflt cx2{ cx.flt.head, cx.flt.tail };
        dblflt cy2{ cy.flt.head, cy.flt.tail };
        dblflt dx2{ dx.flt.head, dx.flt.tail };
        dblflt dy2{ dy.flt.head, dy.flt.tail };
        dblflt centerX2{ centerX.flt.head, centerX.flt.tail };
        dblflt centerY2{ centerY.flt.head, centerY.flt.tail };

        cudaMemcpy(results_x, results->x.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_x2, results->x2.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_y, results->y.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_y2, results->y2.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);
        cudaMemcpy(results_tolerancy, results->tolerancy.flt, results->size * sizeof(dblflt), cudaMemcpyDefault);

        mandel_2x_float_perturb_bla_setup << <nb_blocks, threads_per_block >> > (
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size);

        mandel_2x_float_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size,
            local_width, local_height, cx2, cy2, dx2, dy2,
            centerX2, centerY2,
            n_iterations);

        cudaFree(results_x);
        cudaFree(results_x2);
        cudaFree(results_y);
        cudaFree(results_y2);
        cudaFree(results_tolerancy);
    } else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedBLA) {
        float* results_x;
        float* results_y;
        float* results_x2;
        float* results_y2;
        float* results_tolerancy;

        cudaError_t err;

        err = cudaMallocManaged(&results_x, results->size * sizeof(float), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_x2, results->size * sizeof(float), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y, results->size * sizeof(float), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_y2, results->size * sizeof(float), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        err = cudaMallocManaged(&results_tolerancy, results->size * sizeof(float), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ClearLocals();
            return;
        }

        cudaMemcpy(results_x, results->x.floatOnly, results->size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(results_x2, results->x2.floatOnly, results->size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(results_y, results->y.floatOnly, results->size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(results_y2, results->y2.floatOnly, results->size * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(results_tolerancy, results->tolerancy.floatOnly, results->size * sizeof(float), cudaMemcpyDefault);

        mandel_1x_float_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            results_x, results_x2, results_y, results_y2, results_tolerancy, results->size,
            local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
            centerX.floatOnly, centerY.floatOnly,
            n_iterations);

        cudaFree(results_x);
        cudaFree(results_x2);
        cudaFree(results_y);
        cudaFree(results_y2);
        cudaFree(results_tolerancy);
    }

    ExtractIters(buffer);
}

void GPURenderer::ExtractIters(uint32_t* buffer) {
    size_t aax, aay;
    double temp;

    cudaDeviceSynchronize();

    if (antialiasing != 1) {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {

                buffer[y * MaxFractalSize + x] = 0;
                temp = 0.0;
                for (aay = y * antialiasing; aay < y * antialiasing + antialiasing; aay++) {
                    for (aax = x * antialiasing; aax < x * antialiasing + antialiasing; aax++) {
                        temp += iter_matrix_cu[aay * local_width + aax];
                    }
                }

                temp /= antialiasing * antialiasing;

                buffer[y * MaxFractalSize + x] = (int)temp;
            }
        }
    }
    else {
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                buffer[y * MaxFractalSize + x] = (int)iter_matrix_cu[y * local_width + x];
            }
        }
    }
}