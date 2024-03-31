//////////////////////////////////////////////////////////////////////////////

template<typename IterType>
__global__
void mandel_4x_float(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    GQF::gqf_real cx,
    GQF::gqf_real cy,
    GQF::gqf_real dx,
    GQF::gqf_real dy,
    IterType n_iterations)
{
    using namespace GQF;
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

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

    IterType iter = 0;
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

    OutputIterMatrix[idx] = iter;
}

template<typename IterType>
__global__
void mandel_4x_double(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    GQD::gqd_real cx,
    GQD::gqd_real cy,
    GQD::gqd_real dx,
    GQD::gqd_real dy,
    IterType n_iterations)
{
    using namespace GQD;
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

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

    IterType iter = 0;
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

    OutputIterMatrix[idx] = iter;
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

template<typename IterType>
__global__
void mandel_2x_double(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    dbldbl cx,
    dbldbl cy,
    dbldbl dx,
    dbldbl dy,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

    //// Approach 1
    // TODO need to take dbldbl as parameters to this
    // convert high precision to dbldbl?

    dbldbl cx2 = add_double_to_dbldbl(cx.y, cx.x);
    dbldbl cy2 = add_double_to_dbldbl(cy.y, cy.x);

    dbldbl dx2 = add_double_to_dbldbl(dx.y, dx.x);
    dbldbl dy2 = add_double_to_dbldbl(dy.y, dy.x);

    dbldbl X2 = add_double_to_dbldbl(X, 0);
    dbldbl Y2 = add_double_to_dbldbl(Y, 0);

    dbldbl x0;
    x0 = add_dbldbl(cx2, mul_dbldbl(dx2, X2));

    dbldbl y0 = add_dbldbl(cy2, mul_dbldbl(dy2, Y2));

    dbldbl x = add_double_to_dbldbl(0, 0);
    dbldbl y = add_double_to_dbldbl(0, 0);

    IterType iter = 0;
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

    OutputIterMatrix[idx] = iter;

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

    //OutputIterMatrix[idx] = iter;
}

template<typename IterType, int iteration_precision>
__global__
void mandel_1x_double(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    double cx,
    double cy,
    double dx,
    double dy,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

    double x0 = cx + dx * X;
    double y0 = cy + dy * Y;

    double x = 0.0;
    double y = 0.0;

    n_iterations -= iteration_precision - 1;

    IterType iter = 0;
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

    OutputIterMatrix[idx] = iter;
}

template<typename IterType, int iteration_precision>
__global__
void mandel_2x_float(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    dblflt cx,
    dblflt cy,
    dblflt dx,
    dblflt dy,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

    ////// Approach 1
    //// TODO need to take dblflt as parameters to this
    //// convert high precision to dblflt?

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

    dblflt cx2 = add_float_to_dblflt(cx.head, cx.tail);
    dblflt cy2 = add_float_to_dblflt(cy.head, cy.tail);

    dblflt dx2 = add_float_to_dblflt(dx.head, dx.tail);
    dblflt dy2 = add_float_to_dblflt(dy.head, dy.tail);

    dblflt X2 = add_float_to_dblflt(X, 0);
    dblflt Y2 = add_float_to_dblflt(Y, 0);

    dblflt x0;
    x0 = add_dblflt(cx2, mul_dblflt(dx2, X2));

    dblflt y0;
    y0 = add_dblflt(cy2, mul_dblflt(dy2, Y2));

    dblflt x = {};
    dblflt y = {};

    IterType iter = 0;
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

    while (zrsqr.head + zisqr.head < 4.0f && iter < n_iterations)
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

    OutputIterMatrix[idx] = iter;
}

template<typename IterType, int iteration_precision>
__global__
void mandel_hdr_float(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    HDRFloat<CudaDblflt<dblflt>> cx,
    HDRFloat<CudaDblflt<dblflt>> cy,
    HDRFloat<CudaDblflt<dblflt>> dx,
    HDRFloat<CudaDblflt<dblflt>> dy,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

    HDRFloat<CudaDblflt<dblflt>> X2{ (float)X };
    HDRFloat<CudaDblflt<dblflt>> Y2{ (float)Y };

    X2.Reduce();
    Y2.Reduce();

    HDRFloat<CudaDblflt<dblflt>> x0{ cx + dx * X2 };
    HDRFloat<CudaDblflt<dblflt>> y0{ cy + dy * Y2 };

    x0.Reduce();
    y0.Reduce();

    HDRFloat<CudaDblflt<dblflt>> x{};
    HDRFloat<CudaDblflt<dblflt>> y{};

    IterType iter = 0;
    HDRFloat<CudaDblflt<dblflt>> zrsqr{};
    HDRFloat<CudaDblflt<dblflt>> zisqr{};

    //HDRFloat<CudaDblflt<dblflt>> zrsq2{};
    //HDRFloat<CudaDblflt<dblflt>> zisq2{};
    HDRFloat<CudaDblflt<dblflt>> zsq_sum{};

    HDRFloat<CudaDblflt<dblflt>> Two{ 2.0f };
    HDRFloat<CudaDblflt<dblflt>> Four{ 4.0f };

    //auto MANDEL_2X_FLOAT = [&]() {
    //    y.Reduce();
    //    x.Reduce();
    //    y = x * y * Two + y0;
    //    x = zrsqr - zisqr + x0;
    //    zrsqr = x.square();
    //    zrsqr.Reduce();
    //    zisqr = y.square();
    //    zisqr.Reduce();
    //    zsq_sum = zrsqr + zisqr;
    //    zsq_sum.Reduce();
    //};

    auto MANDEL_2X_FLOAT = [&]() {
        y.Reduce();
        x.Reduce();
        //double tempProd = x.toDouble() * y.toDouble() * 2.0 + y0.toDouble();
        y = x * y * Two + y0;
        //y = HDRFloat<CudaDblflt<dblflt>>(tempProd);
        x = zrsqr - zisqr + x0;
        //double tempProd2 = zrsqr.toDouble() - zisqr.toDouble() + x0.toDouble();
        //x = HDRFloat<CudaDblflt<dblflt>>(tempProd2);
        zrsqr = x * x;
        zrsqr.Reduce();
        zisqr = y * y;
        zisqr.Reduce();
        zsq_sum = zrsqr + zisqr;
        zsq_sum.Reduce();
        };

    while (zsq_sum.compareToBothPositiveReduced(Four) < 0 && iter < n_iterations)
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

    OutputIterMatrix[idx] = iter;
}

template<typename IterType, int iteration_precision>
__global__
void mandel_1x_float(
    IterType* OutputIterMatrix,
    AntialiasedColors OutputColorMatrix,
    int width,
    int height,
    float cx,
    float cy,
    float dx,
    float dy,
    IterType n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = ConvertLocToIndex(X, height - Y - 1, width);

    float x0 = cx + dx * X;
    float y0 = cy + dy * Y;

    float x = 0.0f;
    float y = 0.0f;

    n_iterations -= iteration_precision - 1;

    IterType iter = 0;

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

    OutputIterMatrix[idx] = iter;
}
