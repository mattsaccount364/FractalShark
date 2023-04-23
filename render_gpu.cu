#include <stdio.h>
#include <iostream>

#include "render_gpu.h"
#include "dbldbl.cuh"
#include "dblflt.cuh"
#include "../QuadDouble/gqd_basic.cuh"
#include "../QuadFloat/gqf_basic.cuh"


template<typename Type>
struct MattPerturbSingleResults {
    Type* x;
    Type* x2;
    Type* y;
    Type* y2;
    uint8_t* bad;
    uint32_t* bad_counts;
    size_t size;
    size_t bad_counts_size;

    MattPerturbSingleResults(
        size_t sz,
        Type* in_x,
        Type* in_x2,
        Type* in_y,
        Type* in_y2,
        uint8_t* in_bad,
        uint32_t* in_bad_counts,
        size_t in_bad_counts_size)
        : size(sz),
        x(nullptr),
        x2(nullptr),
        y(nullptr),
        y2(nullptr),
        bad(nullptr),
        bad_counts(nullptr),
        bad_counts_size(in_bad_counts_size) {

        cudaError_t err;

        err = cudaMallocManaged(&x, size * sizeof(Type), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        err = cudaMallocManaged(&x2, size * sizeof(Type), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        err = cudaMallocManaged(&y, size * sizeof(Type), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        err = cudaMallocManaged(&y2, size * sizeof(Type), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        err = cudaMallocManaged(&bad, size * sizeof(uint8_t), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        err = cudaMallocManaged(&bad_counts, bad_counts_size * sizeof(uint8_t), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            return;
        }

        cudaMemcpy(x, in_x, size * sizeof(Type), cudaMemcpyDefault);
        cudaMemcpy(x2, in_x2, size * sizeof(Type), cudaMemcpyDefault);
        cudaMemcpy(y, in_y, size * sizeof(Type), cudaMemcpyDefault);
        cudaMemcpy(y2, in_y2, size * sizeof(Type), cudaMemcpyDefault);
        cudaMemcpy(bad, in_bad, size * sizeof(uint8_t), cudaMemcpyDefault);
        cudaMemcpy(bad_counts, in_bad_counts, bad_counts_size * sizeof(uint32_t), cudaMemcpyDefault);
    }

    ~MattPerturbSingleResults() {
        cudaFree(x);
        cudaFree(x2);
        cudaFree(y);
        cudaFree(y2);
        cudaFree(bad_counts);
    }
};


// Match in Fractal.cpp
constexpr static auto NB_THREADS_W = 16;
constexpr static auto NB_THREADS_H = 8;


__global__
void mandel_4x_float(uint32_t* iter_matrix,
    int width,
    int height,
    GQF::gqf_real cx,
    GQF::gqf_real cy,
    GQF::gqf_real dx,
    GQF::gqf_real dy,
    uint32_t n_iterations)
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
void mandel_4x_double(uint32_t* iter_matrix,
    int width,
    int height,
    GQD::gqd_real cx,
    GQD::gqd_real cy,
    GQD::gqd_real dx,
    GQD::gqd_real dy,
    uint32_t n_iterations)
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
void mandel_2x_double(uint32_t* iter_matrix,
    int width,
    int height,
    dbldbl cx,
    dbldbl cy,
    dbldbl dx,
    dbldbl dy,
    uint32_t n_iterations)
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
void mandel_1x_double(uint32_t* iter_matrix,
                 int width,
                 int height,
                 double cx,
                 double cy,
                 double dx,
                 double dy,
                 uint32_t n_iterations)
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
void mandel_1x_double_perturb_bla(uint32_t* iter_matrix,
    double* results_x,
    double* results_x2,
    double* results_y,
    double* results_y2,
    size_t sz,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations)
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
void mandel_2x_float(uint32_t* iter_matrix,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    uint32_t n_iterations)
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
void mandel_2x_float_perturb_bla_setup(
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    size_t sz)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0)
        return;

    for (size_t i = 0; i < sz; i++) {
        results_x[i] = add_float_to_dblflt(results_x[i].y, results_x[i].x);
        results_x2[i] = add_float_to_dblflt(results_x2[i].y, results_x2[i].x);
        results_y[i] = add_float_to_dblflt(results_y[i].y, results_y[i].x);
        results_y2[i] = add_float_to_dblflt(results_y2[i].y, results_y2[i].x);
    }
}

__global__
void mandel_2x_float_perturb_bla(uint32_t* iter_matrix,
    dblflt* results_x,
    dblflt* results_x2,
    dblflt* results_y,
    dblflt* results_y2,
    size_t sz,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    dblflt centerX,
    dblflt centerY,
    uint32_t n_iterations)
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
void mandel_1x_float(uint32_t* iter_matrix,
    int width,
    int height,
    float cx,
    float cy,
    float dx,
    float dy,
    uint32_t n_iterations)
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
void mandel_1x_float_perturb_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> results,
    int width,
    int height,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint32_t n_iterations)
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

        const float tempSubX = results.x2[RefIteration] + DeltaSubNXOrig;
        const float tempSubY = results.y2[RefIteration] + DeltaSubNYOrig;

        DeltaSubNX =
            DeltaSubNXOrig * tempSubX -
            DeltaSubNYOrig * tempSubY +
            DeltaSub0X;
        DeltaSubNY =
            DeltaSubNXOrig * tempSubY +
            DeltaSubNYOrig * tempSubX +
            DeltaSub0Y;

        ++RefIteration;

        const float tempZX = results.x[RefIteration] + DeltaSubNX;
        const float tempZY = results.y[RefIteration] + DeltaSubNY;
        const float zn_size = tempZX * tempZX + tempZY * tempZY;
        const float normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > 256) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == results.size - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

/*
__global__
void mandel_1x_float_perturb_bla_scaled(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> results,
    MattPerturbSingleResults<double> doubleResults,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    const double deltaReal = dx * X - centerX;
    const double deltaImaginary = -dy * Y - centerY;

    // DeltaSubNWX = 2 * DeltaSubNWX * results.x[RefIteration] - 2 * DeltaSubNWY * results.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY * results.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    double S = sqrt(deltaReal * deltaReal + deltaImaginary * deltaImaginary);

    //double S = 1;
    float DeltaSub0DX = (float)(deltaReal / S);
    float DeltaSub0DY = (float)(deltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    float s = (float)S;
    float s2 = s * s;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    const double W2threshold = exp(log(LARGE_MANTISSA) / 2);
    size_t cur_iter = 0;
    size_t cur_bad_count_iter = 0;

    while (iter < n_iterations) {
        cur_iter = 0;
        while (cur_iter < results.bad_counts[cur_bad_count_iter] && iter < n_iterations) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * results.x2[RefIteration] - DeltaSubNWYOrig * results.y2[RefIteration] +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * results.y2[RefIteration] + DeltaSubNWYOrig * results.x2[RefIteration] +
                twos * DeltaSubNWXOrig * DeltaSubNWYOrig +
                DeltaSub0DY;

            ++RefIteration;

            const float tempZX =
                results.x[RefIteration] + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                results.y[RefIteration] + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const float normDeltaSubN =
                DeltaSubNWX * DeltaSubNWX * s2 +
                DeltaSubNWY * DeltaSubNWY * s2;

            if (zn_size > 256.0f) {
                goto end;
            }

            double DoubleTempZX;
            double DoubleTempZY;

            if (zn_size < normDeltaSubN ||
                RefIteration == results.size - 1) {
                DoubleTempZX = (doubleResults.x[RefIteration] + (double)DeltaSubNWX * S); // Xxrd, xr
                DoubleTempZY = (doubleResults.y[RefIteration] + (double)DeltaSubNWY * S); // Xxid, xi

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = (float)S;
                s2 = s * s;
                twos = 2 * s;

                DeltaSub0DX = (float)(deltaReal / S);
                DeltaSub0DY = (float)(deltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                RefIteration = 0;
                cur_iter = 0;
                cur_bad_count_iter = 0;
                continue;
            }
            else {
                const float w2 = DeltaSubNWX * DeltaSubNWX + DeltaSubNWY * DeltaSubNWY;
                if (w2 >= w2threshold)
                {
                    DoubleTempZX = DeltaSubNWX * S;
                    DoubleTempZY = DeltaSubNWY * S;

                    S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                    s = (float)S;
                    s2 = s * s;
                    twos = 2 * s;

                    DeltaSub0DX = (float)(deltaReal / S);
                    DeltaSub0DY = (float)(deltaImaginary / S);
                    DeltaSubNWX = (float)(DoubleTempZX / S);
                    DeltaSubNWY = (float)(DoubleTempZY / S);
                }
            }

            ++iter;
            ++cur_iter;
        }

        ++cur_bad_count_iter;
        cur_iter = 0;
        while (cur_iter < results.bad_counts[cur_bad_count_iter] && iter < n_iterations) {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = DeltaSubNWX;
            double DeltaSubNWYOrig = DeltaSubNWY;

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * doubleResults.x2[RefIteration] - DeltaSubNWYOrig * doubleResults.y2[RefIteration] +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                deltaReal / S;
            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * doubleResults.y2[RefIteration] + DeltaSubNWYOrig * doubleResults.x2[RefIteration] +
                2 * S * DeltaSubNWXOrig * DeltaSubNWYOrig +
                deltaImaginary / S;

            ++RefIteration;

            const double tempZX =
                doubleResults.x[RefIteration] +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                doubleResults.y[RefIteration] +
                DoubleTempDeltaSubNWY * S; // Xxid

            const double zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const double normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * S * S +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * S * S;

            if (zn_size > 256.0) {
                goto end;
            }

            double DeltaSubNWXNew;
            double DeltaSubNWYNew;

            if (zn_size < normDeltaSubN ||
                RefIteration == doubleResults.size - 1) {
                DeltaSubNWXNew = (doubleResults.x[RefIteration] + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (doubleResults.y[RefIteration] + DoubleTempDeltaSubNWY * S); // Xxid, xi

                S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
                s = (float)S;
                s2 = s * s;
                twos = 2 * s;

                DeltaSub0DX = (float)(deltaReal / S);
                DeltaSub0DY = (float)(deltaImaginary / S);
                DeltaSubNWX = (float)(DeltaSubNWXNew / S);
                DeltaSubNWY = (float)(DeltaSubNWYNew / S);

                RefIteration = 0;
                cur_iter = 0;
                cur_bad_count_iter = -1;
                break;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;

                S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
                s = (float)S;
                s2 = s * s;
                twos = 2 * s;

                DeltaSub0DX = (float)(deltaReal / S);
                DeltaSub0DY = (float)(deltaImaginary / S);
                DeltaSubNWX = (float)(DeltaSubNWXNew / S);
                DeltaSubNWY = (float)(DeltaSubNWYNew / S);
            }

            ++iter;
            ++cur_iter;
        }

        ++cur_bad_count_iter;
    }

end:
    iter_matrix[idx] = (uint32_t)iter;
}
*/

__global__
void mandel_1x_float_perturb_bla_scaled(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> results,
    MattPerturbSingleResults<double> doubleResults,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    const double deltaReal = dx * X - centerX;
    const double deltaImaginary = -dy * Y - centerY;

    // DeltaSubNWX = 2 * DeltaSubNWX * results.x[RefIteration] - 2 * DeltaSubNWY * results.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY * results.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    double S = sqrt(deltaReal * deltaReal + deltaImaginary * deltaImaginary);

    //double S = 1;
    float DeltaSub0DX = (float)(deltaReal / S);
    float DeltaSub0DY = (float)(deltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    float s = (float)S;
    float s2 = s * s;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    const double W2threshold = exp(log(LARGE_MANTISSA) / 2);
    bool full_iteration = false;

    while (iter < n_iterations) {
        full_iteration = results.bad[RefIteration];

        if (full_iteration == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * results.x2[RefIteration] -
                DeltaSubNWYOrig * results.y2[RefIteration] +
                s * (DeltaSubNWXOrig * DeltaSubNWXOrig - DeltaSubNWYOrig * DeltaSubNWYOrig) +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (results.y2[RefIteration] + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * results.x2[RefIteration] +
                DeltaSub0DY;

            ++RefIteration;

            const float tempZX =
                results.x[RefIteration] + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                results.y[RefIteration] + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            if (zn_size > 256.0f) {
                break;
            }

            const float DeltaSubNWXSquared = DeltaSubNWX * DeltaSubNWX;
            const float DeltaSubNWYSquared = DeltaSubNWY * DeltaSubNWY;

            const float normDeltaSubN =
                DeltaSubNWXSquared * s2 +
                DeltaSubNWYSquared * s2;

            double DoubleTempZX;
            double DoubleTempZY;

            if (zn_size < normDeltaSubN ||
                RefIteration == results.size - 1) {
                DoubleTempZX = (doubleResults.x[RefIteration] + (double)DeltaSubNWX * S); // Xxrd, xr
                DoubleTempZY = (doubleResults.y[RefIteration] + (double)DeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = (float)S;
                s2 = s * s;
                twos = 2 * s;

                DeltaSub0DX = (float)(deltaReal / S);
                DeltaSub0DY = (float)(deltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);
            }
            else {
                const float w2 = DeltaSubNWXSquared + DeltaSubNWYSquared;
                if (w2 >= w2threshold)
                {
                    DoubleTempZX = DeltaSubNWX * S;
                    DoubleTempZY = DeltaSubNWY * S;

                    S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                    s = (float)S;
                    s2 = s * s;
                    twos = 2 * s;

                    DeltaSub0DX = (float)(deltaReal / S);
                    DeltaSub0DY = (float)(deltaImaginary / S);
                    DeltaSubNWX = (float)(DoubleTempZX / S);
                    DeltaSubNWY = (float)(DoubleTempZY / S);
                }
            }
        } else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = DeltaSubNWX;
            double DeltaSubNWYOrig = DeltaSubNWY;

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * doubleResults.x2[RefIteration] -
                DeltaSubNWYOrig * doubleResults.y2[RefIteration] +
                S * (DeltaSubNWXOrig * DeltaSubNWXOrig - DeltaSubNWYOrig * DeltaSubNWYOrig) +
                deltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (doubleResults.y2[RefIteration] + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * doubleResults.x2[RefIteration] +
                deltaImaginary / S;

            ++RefIteration;

            const double tempZX =
                doubleResults.x[RefIteration] +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                doubleResults.y[RefIteration] +
                DoubleTempDeltaSubNWY * S; // Xxid

            const double zn_size =
                tempZX * tempZX + tempZY * tempZY;

            if (zn_size > 256.0) {
                break;
            }

            const double TwoS = S * S;
            const double normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * TwoS +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * TwoS;

            double DeltaSubNWXNew;
            double DeltaSubNWYNew;

            if (zn_size < normDeltaSubN ||
                RefIteration == doubleResults.size - 1) {
                DeltaSubNWXNew = (doubleResults.x[RefIteration] + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (doubleResults.y[RefIteration] + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            s = (float)S;
            s2 = s * s;
            twos = 2 * s;

            DeltaSub0DX = (float)(deltaReal / S);
            DeltaSub0DY = (float)(deltaImaginary / S);
            DeltaSubNWX = (float)(DeltaSubNWXNew / S);
            DeltaSubNWY = (float)(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

GPURenderer::GPURenderer() {
    ClearLocals();
}

GPURenderer::~GPURenderer() {
    ResetMemory();
}

void GPURenderer::ResetMemory() {
    if (iter_matrix_cu != nullptr) {
        cudaFree(iter_matrix_cu);
    }

    ClearLocals();
}

void GPURenderer::ClearLocals() {
    // Assumes memory is freed
    iter_matrix_cu = nullptr;

    width = 0;
    height = 0;
    local_width = 0;
    local_height = 0;
    w_block = 0;
    h_block = 0;
    N_cu = 0;
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
    uint32_t n_iterations,
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
    else {
        return;
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
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    if (iter_matrix_cu == nullptr) {
        return;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64PerturbedBLA) {
        MattPerturbSingleResults<double> cudaResults(
            results->size,
            results->x.doubleOnly,
            results->x2.doubleOnly,
            results->y.doubleOnly,
            results->y2.doubleOnly,
            results->bad,
            results->bad_counts,
            results->bad_counts_size);

        mandel_1x_double_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults.x, cudaResults.x2, cudaResults.y, cudaResults.y2, cudaResults.size,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedBLA) {
        MattPerturbSingleResults<dblflt> cudaResults(
            results->size,
            (dblflt*)results->x.flt,
            (dblflt*)results->x2.flt,
            (dblflt*)results->y.flt,
            (dblflt*)results->y2.flt,
            results->bad,
            results->bad_counts,
            results->bad_counts_size);

        dblflt cx2{ cx.flt.head, cx.flt.tail };
        dblflt cy2{ cy.flt.head, cy.flt.tail };
        dblflt dx2{ dx.flt.head, dx.flt.tail };
        dblflt dy2{ dy.flt.head, dy.flt.tail };
        dblflt centerX2{ centerX.flt.head, centerX.flt.tail };
        dblflt centerY2{ centerY.flt.head, centerY.flt.tail };

        mandel_2x_float_perturb_bla_setup << <nb_blocks, threads_per_block >> > (
            cudaResults.x, cudaResults.x2, cudaResults.y, cudaResults.y2, results->size);

        mandel_2x_float_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults.x, cudaResults.x2, cudaResults.y, cudaResults.y2, results->size,
            local_width, local_height, cx2, cy2, dx2, dy2,
            centerX2, centerY2,
            n_iterations);
    } else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedBLA ||
               algorithm == RenderAlgorithm::Gpu1x32PerturbedScaled) {
        MattPerturbSingleResults<float> cudaResults(
            results->size,
            results->x.floatOnly,
            results->x2.floatOnly,
            results->y.floatOnly,
            results->y2.floatOnly,
            results->bad,
            results->bad_counts,
            results->bad_counts_size);

        if (algorithm == RenderAlgorithm::Gpu1x32PerturbedBLA) {
            mandel_1x_float_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
                centerX.floatOnly, centerY.floatOnly,
                n_iterations);
        }
        else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaled) {
            MattPerturbSingleResults<double> cudaResultsDouble(
                results->size,
                results->x.doubleOnly,
                results->x2.doubleOnly,
                results->y.doubleOnly,
                results->y2.doubleOnly,
                results->bad,
                results->bad_counts,
                results->bad_counts_size);

            mandel_1x_float_perturb_bla_scaled << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults, cudaResultsDouble,
                local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
                centerX.doubleOnly, centerY.doubleOnly,
                n_iterations);
        }
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