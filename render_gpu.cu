// TODO: 2x32 perturb is busted, do git diff
// Re-run  profile on current default view
#include <stdio.h>
#include <iostream>

#include "render_gpu.h"
#include "dbldbl.cuh"
#include "dblflt.cuh"
#include "../QuadDouble/gqd_basic.cuh"
#include "../QuadFloat/gqf_basic.cuh"

#include "GPUBLAS.h"

#include "BLA.h"
#include "HDRFloat.h"

#ifdef __CUDACC__
__device__ double twoPowExpData[2048];

#ifdef __CUDA_ARCH__
__device__ void InitStatics()
{
    //LN2 = ::log(2);
    //LN2_REC = 1.0 / LN2;

    twoPowExp = twoPowExpData;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    //twoPowExp.resize(MaxDoubleExponent - MinDoubleExponent + 1);
    for (int i = MinDoubleExponent; i <= MaxDoubleExponent; i++) {
        double d = scalbn(1.0, i);
        int index = i - MinDoubleExponent;
        twoPowExp[index] = d;
    }
}
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Bilinear approximation
////////////////////////////////////////////////////////////////////////////////////////

template<class T>
CUDA_CRAP BLA<T>::BLA(T r2, T RealA, T ImagA, T RealB, T ImagB, int l) {
    this->Ax = RealA;
    this->Ay = ImagA;
    this->Bx = RealB;
    this->By = ImagB;
    this->r2 = r2;
    this->l = l;

    HdrReduce(this->Ax);
    HdrReduce(this->Ay);
    HdrReduce(this->Bx);
    HdrReduce(this->By);
    HdrReduce(this->r2);
}

template<class T>
CUDA_CRAP void BLA<T>::getValue(
    T &RealDeltaSubN,
    T &ImagDeltaSubN,
    T RealDeltaSub0,
    T ImagDeltaSub0
) {

    //T zxn = Ax * zx - Ay * zy + Bx * cx - By * cy;
    //T zyn = Ax * zy + Ay * zx + Bx * cy + By * cx;
    T NewRealValue = Ax * RealDeltaSubN - Ay * ImagDeltaSubN + Bx * RealDeltaSub0 - By * ImagDeltaSub0;
    T NewImagValue = Ax * ImagDeltaSubN + Ay * RealDeltaSubN + Bx * ImagDeltaSub0 + By * RealDeltaSub0;
    RealDeltaSubN = NewRealValue;
    HdrReduce(RealDeltaSubN);

    ImagDeltaSubN = NewImagValue;
    HdrReduce(ImagDeltaSubN);
}

template<class T>
CUDA_CRAP T BLA<T>::hypotA() const {
    auto ret = HdrSqrt<T>(Ax * Ax + Ay * Ay);
    HdrReduce(ret);
    return ret;
}

template<class T>
CUDA_CRAP T BLA<T>::hypotB() const {
    auto ret = HdrSqrt<T>(Bx * Bx + By * By);
    HdrReduce(ret);
    return ret;
}

template<class T>
CUDA_CRAP BLA<T> BLA<T>::getGenericStep(
    T r2,
    T RealA,
    T ImagA,
    T RealB,
    T ImagB,
    int l
) {
    return BLA(r2, RealA, ImagA, RealB, ImagB, l);
}

// A = y.A * x.A
template<class T>
CUDA_CRAP void BLA<T>::getNewA(const BLA &x, const BLA &y, T &RealValue, T &ImagValue) {
    RealValue = y.Ax * x.Ax - y.Ay * x.Ay;
    HdrReduce(RealValue);

    ImagValue = y.Ax * x.Ay + y.Ay * x.Ax;
    HdrReduce(ImagValue);
}

// B = y.A * x.B + y.B
template<class T>
CUDA_CRAP void BLA<T>::getNewB(const BLA &x, const BLA &y, T& RealValue, T& ImagValue) {
    T xBx = x.Bx;
    T xBy = x.By;
    RealValue = y.Ax * xBx - y.Ay * xBy + y.Bx;
    HdrReduce(RealValue);

    ImagValue = y.Ax * xBy + y.Ay * xBx + y.By;
    HdrReduce(ImagValue);
}

template<class T>
CUDA_CRAP int BLA<T>::getL() const {
    return l;
}

template<class T>
CUDA_CRAP T BLA<T>::getR2() const {
    return r2;
}

template class BLA<float>;
template class BLA<double>;
template class BLA<HDRFloat<float>>;
template class BLA<HDRFloat<double>>;

////////////////////////////////////////////////////////////////////////////////////////
// Bilinear approximation.  GPU copy.
////////////////////////////////////////////////////////////////////////////////////////

template<class T, class GPUBLA_TYPE>
GPUBLAS<T, GPUBLA_TYPE>::GPUBLAS(const std::vector<std::vector<GPUBLA_TYPE>>& B,
    int32_t LM2,
    size_t FirstLevel)
    : m_ElementsPerLevel(nullptr),
    m_NumLevels(0),
    m_B(nullptr),
    m_LM2(LM2),
    m_FirstLevel(FirstLevel),
    m_Err(),
    m_Owned(true) {

    m_NumLevels = B.size();

    m_Err = cudaMallocManaged(&m_B, m_NumLevels * sizeof(GPUBLA_TYPE*), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    cudaMemset(m_B, 0, m_NumLevels * sizeof(GPUBLA_TYPE*));

    m_Err = cudaMallocManaged(&m_ElementsPerLevel,
        m_NumLevels * sizeof(size_t),
        cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    for (size_t i = 0; i < B.size(); i++) {
        m_ElementsPerLevel[i] = B[i].size();
    }

    for (size_t i = 0; i < B.size(); i++) {
        m_Err = cudaMallocManaged(&m_B[i],
            sizeof(GPUBLA_TYPE) * m_ElementsPerLevel[i],
            cudaMemAttachGlobal);

        if (m_Err != cudaSuccess) {
            return;
        }

        cudaMemcpy(m_B[i],
            B[i].data(),
            sizeof(GPUBLA_TYPE) * m_ElementsPerLevel[i],
            cudaMemcpyDefault);
    }
}

template<class T, class GPUBLA_TYPE>
GPUBLAS<T, GPUBLA_TYPE>::~GPUBLAS() {
    if (m_Owned) {
        if (m_ElementsPerLevel != nullptr) {
            cudaFree(m_ElementsPerLevel);
            m_ElementsPerLevel = nullptr;
        }

        for (size_t i = 0; i < m_NumLevels; i++) {
            cudaFree(m_B[i]);
            m_B[i] = nullptr;
        }

        if (m_B != nullptr) {
            cudaFree(m_B);
            m_B = nullptr;
        }
    }
}

template<class T, class GPUBLA_TYPE>
GPUBLAS<T, GPUBLA_TYPE>::GPUBLAS(const GPUBLAS& other) {
    if (this == &other) {
        return;
    }

    m_ElementsPerLevel = other.m_ElementsPerLevel;
    m_NumLevels = other.m_NumLevels;
    m_B = other.m_B;
    //for (size_t i = 0; i < m_NumLevels; i++) {
    //    m_B[i] = other.m_B[i];
    //}

    m_LM2 = other.m_LM2;
    m_FirstLevel = other.m_FirstLevel;

    m_Owned = false;
}

template<class T, class GPUBLA_TYPE>
uint32_t GPUBLAS<T, GPUBLA_TYPE>::CheckValid() const {
    return m_Err;
}

template<class T, class GPUBLA_TYPE>
CUDA_CRAP GPUBLA_TYPE* GPUBLAS<T, GPUBLA_TYPE>::LookupBackwards(size_t m, T z2) {

    if (m == 0) {
        return nullptr;
    }

    GPUBLA_TYPE* tempB = nullptr;

    int32_t k = (int32_t)m - 1;

    if ((k & 1) == 1) { // m - 1 is odd
        return nullptr;
    }

    int32_t zeros;
    uint32_t ix;
    if (k == 0) {
        // k >> m_FirstLevel,
        // This could be done for all K values, but it was shown through statistics that
        // most effort is done on k == 0
        if (z2 >= m_B[m_FirstLevel][0].getR2()) {
            return nullptr;
        }
        zeros = 32;
        ix = 0;
    }
    else {
        float v = (float)(k & -k);
        uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
        zeros = (bits >> 23) - 0x7f;
        ix = k >> zeros;
    }

    int32_t startLevel = ((zeros <= m_LM2) ? zeros : m_LM2);
    for (int32_t level = startLevel; level >= m_FirstLevel; --level) {
        if (z2 < (tempB = &m_B[level][ix])->getR2()) {
            return tempB;
        }
        ix = ix << 1;
    }
    return nullptr;
}


template class GPUBLAS<float, BLA<double>>;
template class GPUBLAS<double, BLA<double>>;
template class GPUBLAS<HDRFloat<double>, BLA<HDRFloat<double>>>;
template class GPUBLAS<HDRFloat<float>, BLA<HDRFloat<float>>>;


////////////////////////////////////////////////////////////////////////////////////////
// Perturbation results
////////////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(MattReferenceSingleIter<float>) == 24, "Float");
static_assert(sizeof(MattReferenceSingleIter<double>) == 40, "Double");
static_assert(sizeof(MattReferenceSingleIter<dblflt>) == 40, "Dblflt");

//char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
//char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
//char(*__kaboom3)[sizeof(MattReferenceSingleIter<dblflt>)] = 1;

template<typename Type>
struct MattPerturbSingleResults {
    MattReferenceSingleIter<Type>* iters;
    size_t size;
    bool own;
    cudaError_t err;

    MattPerturbSingleResults(
        size_t sz,
        MattReferenceSingleIter<Type> *in_iters)
        : size(sz),
        iters(nullptr),
        own(true),
        err(cudaSuccess) {

        static_assert(sizeof(MattDblflt) == sizeof(dblflt), "No");

        err = cudaMallocManaged(&iters, size * sizeof(MattReferenceSingleIter<Type>), cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            size = 0;
            return;
        }

        cudaMemcpy(iters, in_iters, size * sizeof(MattReferenceSingleIter<Type>), cudaMemcpyDefault);
    }

    // funny semantics, copy doesn't own the pointers.
    MattPerturbSingleResults(const MattPerturbSingleResults& other) {
        if (this == &other) {
            return;
        }

        iters = other.iters;
        size = other.size;
        own = false;
    }

    uint32_t CheckValid() const {
        return err;
    }

    MattPerturbSingleResults(MattPerturbSingleResults&& other) = delete;
    MattPerturbSingleResults &operator=(const MattPerturbSingleResults& other) = delete;
    MattPerturbSingleResults &operator=(MattPerturbSingleResults&& other) = delete;

    ~MattPerturbSingleResults() {
        if (own) {
            if (iters != nullptr) {
                cudaFree(iters);
            }
        }
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
void mandel_1x_double_perturb(uint32_t* iter_matrix,
    MattPerturbSingleResults<double> PerturbDouble,
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
    double DeltaReal = dx * X - centerX;
    double DeltaImaginary = -dy * Y - centerY;

    double DeltaSub0X = DeltaReal;
    double DeltaSub0Y = DeltaImaginary;
    double DeltaSubNX = 0;
    double DeltaSubNY = 0;
    size_t MaxRefIteration = PerturbDouble.size - 1;

    while (iter < n_iterations) {
        MattReferenceSingleIter<double> *CurIter = &PerturbDouble.iters[RefIteration];

        const double DeltaSubNXOrig = DeltaSubNX;
        const double DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX =
            DeltaSubNXOrig * (CurIter->x2 + DeltaSubNXOrig) -
            DeltaSubNYOrig * (CurIter->y2 + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY =
            DeltaSubNXOrig * (CurIter->y2 + DeltaSubNYOrig) +
            DeltaSubNYOrig * (CurIter->x2 + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;
        CurIter = &PerturbDouble.iters[RefIteration];

        const double tempZX = CurIter->x + DeltaSubNX;
        const double tempZY = CurIter->y + DeltaSubNY;
        const double zn_size = tempZX * tempZX + tempZY * tempZY;
        const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > 256) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

__global__
void mandel_1x_double_perturb_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<double> PerturbDouble,
    GPUBLAS<double, BLA<double>> doubleBlas,
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
    double DeltaReal = dx * X - centerX;
    double DeltaImaginary = -dy * Y - centerY;

    double DeltaSub0X = DeltaReal;
    double DeltaSub0Y = DeltaImaginary;
    double DeltaSubNX = 0;
    double DeltaSubNY = 0;
    double DeltaNormSquared = 0;

    while (iter < n_iterations) {
        BLA<double>* b = nullptr;
        while ((b = doubleBlas.LookupBackwards(RefIteration, DeltaNormSquared)) != nullptr) {
            int l = b->getL();

            // TODO this first RefIteration + l check bugs me
            if (RefIteration + l >= PerturbDouble.size) {
                break;
            }

            if (iter + l >= n_iterations) {
                break;
            }

            iter += l;
            RefIteration += l;

            b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

            const double tempZX = PerturbDouble.iters[RefIteration].x + DeltaSubNX;
            const double tempZY = PerturbDouble.iters[RefIteration].y + DeltaSubNY;
            const double normSquared = tempZX * tempZX + tempZY * tempZY;
            DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

            if (normSquared > 256) {
                break;
            }

            if (normSquared < DeltaNormSquared ||
                RefIteration >= PerturbDouble.size - 1) {
                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;
                DeltaNormSquared = normSquared;
                RefIteration = 0;
            }
        }

        if (iter >= n_iterations) {
            break;
        }

        const double DeltaSubNXOrig = DeltaSubNX;
        const double DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX = DeltaSubNXOrig * (PerturbDouble.iters[RefIteration].x2 + DeltaSubNXOrig) -
            DeltaSubNYOrig * (PerturbDouble.iters[RefIteration].y2 + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY = DeltaSubNXOrig * (PerturbDouble.iters[RefIteration].y2 + DeltaSubNYOrig) +
            DeltaSubNYOrig * (PerturbDouble.iters[RefIteration].x2 + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;

        const double tempZX = PerturbDouble.iters[RefIteration].x + DeltaSubNX;
        const double tempZY = PerturbDouble.iters[RefIteration].y + DeltaSubNY;
        const double normSquared = tempZX * tempZX + tempZY * tempZY;
        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (normSquared > 256) {
            break;
        }

        if (normSquared < DeltaNormSquared ||
            RefIteration >= PerturbDouble.size - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            DeltaNormSquared = normSquared;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

__global__
void mandel_1xHDR_InitStatics()
{
    if (blockIdx.x == 0 &&
        threadIdx.x == 0 &&
        blockIdx.y == 0 &&
        threadIdx.y == 0) {
        InitStatics();
    }
}

template<class HDRFloatType>
__global__
void mandel_1xHDR_float_perturb_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<HDRFloatType> Perturb,
    GPUBLAS<HDRFloatType, BLA<HDRFloatType>> blas,
    int width,
    int height,
    HDRFloatType cx,
    HDRFloatType cy,
    HDRFloatType dx,
    HDRFloatType dy,
    HDRFloatType centerX,
    HDRFloatType centerY,
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

    HdrReduce(centerX);
    HdrReduce(centerY);

    size_t iter = 0;
    size_t RefIteration = 0;
    HDRFloatType DeltaReal = dx * X - centerX;
    HDRFloatType DeltaImaginary = -dy * Y - centerY;

    HDRFloatType DeltaSub0X = DeltaReal;
    HDRFloatType DeltaSub0Y = DeltaImaginary;
    HDRFloatType DeltaSubNX = 0;
    HDRFloatType DeltaSubNY = 0;
    HDRFloatType DeltaNormSquared = 0;

    while (iter < n_iterations) {
        BLA<HDRFloatType>* b = nullptr;
        while ((b = blas.LookupBackwards(RefIteration, DeltaNormSquared)) != nullptr) {
            int l = b->getL();

            // TODO this first RefIteration + l check bugs me
            if (RefIteration + l >= Perturb.size) {
                break;
            }

            if (iter + l >= n_iterations) {
                break;
            }

            iter += l;
            RefIteration += l;

            b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

            HDRFloatType tempZX = Perturb.iters[RefIteration].x + DeltaSubNX;
            HDRFloatType tempZY = Perturb.iters[RefIteration].y + DeltaSubNY;
            HDRFloatType normSquared = tempZX * tempZX + tempZY * tempZY;
            HdrReduce(normSquared);

            DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
            HdrReduce(DeltaNormSquared);

            if (normSquared > 256) {
                break;
            }

            if (normSquared < DeltaNormSquared ||
                RefIteration >= Perturb.size - 1) {
                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;
                DeltaNormSquared = normSquared;
                RefIteration = 0;
            }
        }

        if (iter >= n_iterations) {
            break;
        }

        HDRFloatType DeltaSubNXOrig = DeltaSubNX;
        HDRFloatType DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX = DeltaSubNXOrig * (Perturb.iters[RefIteration].x2 + DeltaSubNXOrig) -
            DeltaSubNYOrig * (Perturb.iters[RefIteration].y2 + DeltaSubNYOrig) +
            DeltaSub0X;
        HdrReduce(DeltaSubNX);

        DeltaSubNY = DeltaSubNXOrig * (Perturb.iters[RefIteration].y2 + DeltaSubNYOrig) +
            DeltaSubNYOrig * (Perturb.iters[RefIteration].x2 + DeltaSubNXOrig) +
            DeltaSub0Y;
        HdrReduce(DeltaSubNY);

        ++RefIteration;
        if (RefIteration >= Perturb.size) {
            // TODO this first RefIteration + l check bugs me
            iter_matrix[idx] = 255;
            return;
        }

        HDRFloatType tempZX = Perturb.iters[RefIteration].x + DeltaSubNX;
        HDRFloatType tempZY = Perturb.iters[RefIteration].y + DeltaSubNY;
        HDRFloatType normSquared = tempZX * tempZX + tempZY * tempZY;
        HdrReduce(normSquared);

        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
        HdrReduce(DeltaNormSquared);

        if (normSquared > 256) {
            break;
        }

        if (normSquared < DeltaNormSquared ||
            RefIteration >= Perturb.size - 1) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            DeltaNormSquared = normSquared;
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
void mandel_2x_float_perturb_setup(MattPerturbSingleResults<dblflt> PerturbDblFlt)
{
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0)
        return;

    for (size_t i = 0; i < PerturbDblFlt.size; i++) {
        PerturbDblFlt.iters[i].x = add_float_to_dblflt(PerturbDblFlt.iters[i].x.y, PerturbDblFlt.iters[i].x.x);
        PerturbDblFlt.iters[i].x2 = add_float_to_dblflt(PerturbDblFlt.iters[i].x2.y, PerturbDblFlt.iters[i].x2.x);
        PerturbDblFlt.iters[i].y = add_float_to_dblflt(PerturbDblFlt.iters[i].y.y, PerturbDblFlt.iters[i].y.x);
        PerturbDblFlt.iters[i].y2 = add_float_to_dblflt(PerturbDblFlt.iters[i].y2.y, PerturbDblFlt.iters[i].y2.x);
    }
}

__global__
void mandel_2x_float_perturb(uint32_t* iter_matrix,
    MattPerturbSingleResults<dblflt> PerturbDblFlt,
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
    //double DeltaReal = dx * X - centerX;
    //double DeltaImaginary = -dy * Y - centerY;

    //double DeltaSub0X = DeltaReal;
    //double DeltaSub0Y = DeltaImaginary;
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

    dblflt DeltaReal = sub_dblflt(mul_dblflt(dx, X2), centerX);
    dblflt DeltaImaginary = sub_dblflt(mul_dblflt(dy, MinusY2), centerY);

    dblflt DeltaSub0X = DeltaReal;
    dblflt DeltaSub0Y = DeltaImaginary;
    dblflt DeltaSubNX, DeltaSubNY;

    size_t MaxRefIteration = PerturbDblFlt.size - 1;

    DeltaSubNX = add_float_to_dblflt(0, 0);
    DeltaSubNY = add_float_to_dblflt(0, 0);

    while (iter < n_iterations) {
        MattReferenceSingleIter<dblflt>* CurIter = &PerturbDblFlt.iters[RefIteration];

        const dblflt DeltaSubNXOrig = DeltaSubNX;
        const dblflt DeltaSubNYOrig = DeltaSubNY;

        const dblflt tempTermX1 = add_dblflt(CurIter->x2, DeltaSubNXOrig);
        const dblflt tempTermX2 = add_dblflt(CurIter->y2, DeltaSubNYOrig);

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
        CurIter = &PerturbDblFlt.iters[RefIteration];

        const dblflt tempZX = add_dblflt(CurIter->x, DeltaSubNX);
        const dblflt tempZY = add_dblflt(CurIter->y, DeltaSubNY);
        const dblflt zn_size = add_dblflt(sqr_dblflt(tempZX), sqr_dblflt(tempZY));
        const dblflt normDeltaSubN = add_dblflt(sqr_dblflt(DeltaSubNX), sqr_dblflt(DeltaSubNY));

        if (zn_size.y > 256) {
            break;
        }

        if (zn_size.y < normDeltaSubN.y ||
            RefIteration == MaxRefIteration) {
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
void mandel_1x_float_perturb(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> PerturbFloat,
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
    float DeltaReal = dx * X - centerX;
    float DeltaImaginary = -dy * Y - centerY;

    float DeltaSub0X = DeltaReal;
    float DeltaSub0Y = DeltaImaginary;
    float DeltaSubNX = 0;
    float DeltaSubNY = 0;
    size_t MaxRefIteration = PerturbFloat.size - 1;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<float> *curIter = &PerturbFloat.iters[RefIteration];

        const float DeltaSubNXOrig = DeltaSubNX;
        const float DeltaSubNYOrig = DeltaSubNY;

        const float tempSubX = curIter->x2 + DeltaSubNXOrig;
        const float tempSubY = curIter->y2 + DeltaSubNYOrig;

        DeltaSubNX =
            DeltaSubNXOrig * tempSubX -
            DeltaSubNYOrig * tempSubY +
            DeltaSub0X;
        DeltaSubNY =
            DeltaSubNXOrig * tempSubY +
            DeltaSubNYOrig * tempSubX +
            DeltaSub0Y;

        ++RefIteration;
        curIter = &PerturbFloat.iters[RefIteration];

        const float tempZX = curIter->x + DeltaSubNX;
        const float tempZY = curIter->y + DeltaSubNY;
        const float zn_size = tempZX * tempZX + tempZY * tempZY;
        const float normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > 256) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

__global__
void mandel_1x_float_perturb_scaled(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> PerturbFloat,
    MattPerturbSingleResults<double> PerturbDouble,
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

    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    const double DeltaReal = dx * X - centerX;
    const double DeltaImaginary = -dy * Y - centerY;

    // DeltaSubNWX = 2 * DeltaSubNWX * PerturbFloat.x[RefIteration] - 2 * DeltaSubNWY * PerturbFloat.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * PerturbFloat.y[RefIteration] + 2 * DeltaSubNWY * PerturbFloat.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    double S = sqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);

    //double S = 1;
    float DeltaSub0DX = (float)(DeltaReal / S);
    float DeltaSub0DY = (float)(DeltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    float s = (float)S;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    size_t MaxRefIteration = PerturbFloat.size - 1;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<float> *curFloatIter = &PerturbFloat.iters[RefIteration];
        const MattReferenceSingleIter<double> *curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curFloatIter->bad == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * curFloatIter->x2 -
                DeltaSubNWYOrig * curFloatIter->y2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x2 +
                DeltaSub0DY;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const float tempZX =
                curFloatIter->x + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                curFloatIter->y + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const float DeltaSubNWXSquared = DeltaSubNWX * DeltaSubNWX;
            const float DeltaSubNWYSquared = DeltaSubNWY * DeltaSubNWY;
            const float w2 = DeltaSubNWXSquared + DeltaSubNWYSquared;
            const float normDeltaSubN = w2 * s * s;

            double DoubleTempZX;
            double DoubleTempZY;

            const bool zn_size_OK = (zn_size < 256.0f);
            const bool test1a = (zn_size < normDeltaSubN);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2 >= w2threshold) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            } else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + (double)DeltaSubNWX * S); // Xxrd, xr
                DoubleTempZY = (curDoubleIter->y + (double)DeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = DeltaSubNWX * S;
                DoubleTempZY = DeltaSubNWY * S;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        } else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = DeltaSubNWX;
            double DeltaSubNWYOrig = DeltaSubNWY;

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * curDoubleIter->x2 -
                DeltaSubNWYOrig * curDoubleIter->y2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x2 +
                DeltaImaginary / S;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const double tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                curDoubleIter->y +
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
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            s = (float)S;
            twos = 2 * s;

            DeltaSub0DX = (float)(DeltaReal / S);
            DeltaSub0DY = (float)(DeltaImaginary / S);
            DeltaSubNWX = (float)(DeltaSubNWXNew / S);
            DeltaSubNWY = (float)(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

__global__
void mandel_1x_float_perturb_scaled_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> PerturbFloat,
    MattPerturbSingleResults<double> PerturbDouble,
    GPUBLAS<double, BLA<double>> doubleBlas,
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

    size_t idx = width * Y + X;

    if (iter_matrix[idx] != 0) {
        return;
    }

    size_t iter = 0;
    size_t RefIteration = 0;
    const double DeltaReal = dx * X - centerX;
    const double DeltaImaginary = -dy * Y - centerY;
    double S = sqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);
    float DeltaSub0DX = (float)(DeltaReal / S);
    float DeltaSub0DY = (float)(DeltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    double DeltaNormSquared = 0;
    
    float s = (float)S;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    size_t MaxRefIteration = PerturbFloat.size - 1;

    while (iter < n_iterations) {
        MattReferenceSingleIter<float> *curFloatIter = &PerturbFloat.iters[RefIteration];
        MattReferenceSingleIter<double> *curDoubleIter = &PerturbDouble.iters[RefIteration];

        double DeltaSubNX = DeltaSubNWX * S;
        double DeltaSubNY = DeltaSubNWY * S;

        BLA<double>* b = nullptr;

        b = doubleBlas.LookupBackwards(RefIteration, DeltaNormSquared);
        if (b != nullptr) {
            for (;;) {
                int l = b->getL();

                // TODO this first RefIteration + l check bugs me
                if (RefIteration + l >= PerturbDouble.size) {
                    break;
                }

                if (iter + l >= n_iterations) {
                    break;
                }

                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaReal, DeltaImaginary);

                curDoubleIter = &PerturbDouble.iters[RefIteration];
                const double tempZX = curDoubleIter->x + DeltaSubNX;
                const double tempZY = curDoubleIter->y + DeltaSubNY;
                const double normSquared = tempZX * tempZX + tempZY * tempZY;
                DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                if (normSquared > 256) {
                    break;
                }

                if (normSquared < DeltaNormSquared ||
                    RefIteration >= PerturbFloat.size - 1) {
                    DeltaSubNX = tempZX;
                    DeltaSubNY = tempZY;
                    DeltaNormSquared = normSquared;
                    RefIteration = 0;
                }

                b = doubleBlas.LookupBackwards(RefIteration, DeltaNormSquared);
                if (b == nullptr) {
                    double DoubleTempZX = DeltaSubNX;
                    double DoubleTempZY = DeltaSubNY;

                    DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                    S = sqrt(DeltaNormSquared);
                    s = (float)S;
                    twos = 2 * s;

                    DeltaSub0DX = (float)(DeltaReal / S);
                    DeltaSub0DY = (float)(DeltaImaginary / S);
                    DeltaSubNWX = (float)(DoubleTempZX / S);
                    DeltaSubNWY = (float)(DoubleTempZY / S);
                    break;
                }
            }
        }

        if (iter >= n_iterations) {
            break;
        }

        curFloatIter = &PerturbFloat.iters[RefIteration];
        curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curFloatIter->bad == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * curFloatIter->x2 -
                DeltaSubNWYOrig * curFloatIter->y2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x2 +
                DeltaSub0DY;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const float tempZX =
                curFloatIter->x + DeltaSubNWX * s; // Xxrd

            const float tempZY =
                curFloatIter->y + DeltaSubNWY * s; // Xxid

            const float zn_size =
                tempZX * tempZX + tempZY * tempZY;

            const float DeltaSubNWXSquared = DeltaSubNWX * DeltaSubNWX;
            const float DeltaSubNWYSquared = DeltaSubNWY * DeltaSubNWY;
            const float w2 = DeltaSubNWXSquared + DeltaSubNWYSquared;
            const float normDeltaSubN = w2 * s * s;
            DeltaNormSquared = normDeltaSubN; // Yay

            double DoubleTempZX;
            double DoubleTempZY;

            const bool zn_size_OK = (zn_size < 256.0f);
            const bool test1a = (zn_size < normDeltaSubN);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2 >= w2threshold) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            } else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + (double)DeltaSubNWX * S); // Xxrd, xr
                DoubleTempZY = (curDoubleIter->y + (double)DeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;

                DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                S = sqrt(DeltaNormSquared);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = DeltaSubNWX * S;
                DoubleTempZY = DeltaSubNWY * S;

                DeltaNormSquared = DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY; // Yay
                S = sqrt(DeltaNormSquared);
                s = (float)S;
                twos = 2 * s;

                DeltaSub0DX = (float)(DeltaReal / S);
                DeltaSub0DY = (float)(DeltaImaginary / S);
                DeltaSubNWX = (float)(DoubleTempZX / S);
                DeltaSubNWY = (float)(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        } else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = DeltaSubNWX;
            double DeltaSubNWYOrig = DeltaSubNWY;

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * curDoubleIter->x2 -
                DeltaSubNWYOrig * curDoubleIter->y2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x2 +
                DeltaImaginary / S;

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const double tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                curDoubleIter->y +
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
            DeltaNormSquared = normDeltaSubN; // Yay

            double DeltaSubNWXNew;
            double DeltaSubNWYNew;

            if (zn_size < normDeltaSubN ||
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            DeltaNormSquared = DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew; // Yay
            S = sqrt(DeltaNormSquared);
            s = (float)S;
            twos = 2 * s;

            DeltaSub0DX = (float)(DeltaReal / S);
            DeltaSub0DY = (float)(DeltaImaginary / S);
            DeltaSubNWX = (float)(DeltaSubNWXNew / S);
            DeltaSubNWY = (float)(DeltaSubNWYNew / S);
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

__global__
void mandel_2x_float_perturb_scaled(uint32_t* iter_matrix,
    MattPerturbSingleResults<dblflt> PerturbDoubleFlt,
    MattPerturbSingleResults<double> PerturbDouble,
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
    const double DeltaReal = dx * X - centerX;
    const double DeltaImaginary = -dy * Y - centerY;

    // DeltaSubNWX = 2 * DeltaSubNWX * PerturbDoubleFlt.x[RefIteration] - 2 * DeltaSubNWY * PerturbDoubleFlt.y[RefIteration] +
    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
    //               dX
    // DeltaSubNWY = 2 * DeltaSubNWX * PerturbDoubleFlt.y[RefIteration] + 2 * DeltaSubNWY * PerturbDoubleFlt.x[RefIteration] +
    //               2 * S * DeltaSubNWX * DeltaSubNWY +
    //               dY
    // 
    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
    //     = 2 * Xr * wr + wr * wr * s - 2 * Xi * wi - wi * wi * s + ur;
    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;

    double S = sqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);

    //double S = 1;
    dblflt DeltaSub0DX = double_to_dblflt(DeltaReal / S);
    dblflt DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
    dblflt DeltaSubNWX = add_float_to_dblflt(0, 0);
    dblflt DeltaSubNWY = add_float_to_dblflt(0, 0);

    dblflt s = double_to_dblflt(S);
    dblflt twos = add_dblflt(s, s);
    const dblflt w2threshold = double_to_dblflt(exp(log(LARGE_MANTISSA) / 2));
    size_t MaxRefIteration = PerturbDoubleFlt.size - 1;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<dblflt>* curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
        const MattReferenceSingleIter<double>* curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curDblFloatIter->bad == false) {
            const dblflt DeltaSubNWXOrig = DeltaSubNWX;
            const dblflt DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX = mul_dblflt(DeltaSubNWXOrig, curDblFloatIter->x2);
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(DeltaSubNWYOrig, curDblFloatIter->y2));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWXOrig), DeltaSubNWXOrig));
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWYOrig), DeltaSubNWYOrig));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, DeltaSub0DX);

            DeltaSubNWY = mul_dblflt(DeltaSubNWXOrig, (add_dblflt(curDblFloatIter->y2, mul_dblflt(twos, DeltaSubNWYOrig))));
            DeltaSubNWY = add_dblflt(DeltaSubNWY, mul_dblflt(DeltaSubNWYOrig, curDblFloatIter->x2));
            DeltaSubNWY = add_dblflt(DeltaSubNWY, DeltaSub0DY);

            ++RefIteration;
            curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const dblflt tempZX =
                add_dblflt(curDblFloatIter->x, mul_dblflt(DeltaSubNWX, s)); // Xxrd

            const dblflt tempZY =
                add_dblflt(curDblFloatIter->y, mul_dblflt(DeltaSubNWY, s)); // Xxid

            const dblflt zn_size =
                add_dblflt(sqr_dblflt(tempZX), sqr_dblflt(tempZY));

            const dblflt DeltaSubNWXSquared = sqr_dblflt(DeltaSubNWX);
            const dblflt DeltaSubNWYSquared = sqr_dblflt(DeltaSubNWY);
            const dblflt w2 = add_dblflt(DeltaSubNWXSquared, DeltaSubNWYSquared);
            const dblflt normDeltaSubN = mul_dblflt(w2, sqr_dblflt(s));

            double DoubleTempZX;
            double DoubleTempZY;

            const bool zn_size_OK = (zn_size.y < 256.0f);
            const bool test1a = (zn_size.y < normDeltaSubN.y);
            const bool test1b = (RefIteration == MaxRefIteration);
            const bool test1ab = test1a || (test1b && zn_size_OK);
            const bool testw2 = (w2.y >= w2threshold.y) && zn_size_OK;
            const bool none = !test1ab && !testw2 && zn_size_OK;

            if (none) {
                ++iter;
                continue;
            }
            else if (test1ab) {
                DoubleTempZX = (curDoubleIter->x + dblflt_to_double(DeltaSubNWX) * S); // Xxrd, xr
                DoubleTempZY = (curDoubleIter->y + dblflt_to_double(DeltaSubNWY) * S); // Xxid, xi

                RefIteration = 0;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = double_to_dblflt(S);
                twos = add_dblflt(s, s);

                DeltaSub0DX = double_to_dblflt(DeltaReal / S);
                DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
                DeltaSubNWX = double_to_dblflt(DoubleTempZX / S);
                DeltaSubNWY = double_to_dblflt(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else if (testw2)
            {
                DoubleTempZX = dblflt_to_double(DeltaSubNWX) * S;
                DoubleTempZY = dblflt_to_double(DeltaSubNWY) * S;

                S = sqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                s = double_to_dblflt(S);
                twos = add_dblflt(s, s);

                DeltaSub0DX = double_to_dblflt(DeltaReal / S);
                DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
                DeltaSubNWX = double_to_dblflt(DoubleTempZX / S);
                DeltaSubNWY = double_to_dblflt(DoubleTempZY / S);

                ++iter;
                continue;
            }
            else {
                // zn_size fail
                break;
            }
        }
        else {
            // Do full iteration at double precision
            double DeltaSubNWXOrig = dblflt_to_double(DeltaSubNWX);
            double DeltaSubNWYOrig = dblflt_to_double(DeltaSubNWY);

            const double DoubleTempDeltaSubNWX =
                DeltaSubNWXOrig * curDoubleIter->x2 -
                DeltaSubNWYOrig * curDoubleIter->y2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x2 +
                DeltaImaginary / S;

            ++RefIteration;
            curDblFloatIter = &PerturbDoubleFlt.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const double tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const double tempZY =
                curDoubleIter->y +
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
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = sqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            s = double_to_dblflt(S);
            twos = add_dblflt(s, s);

            DeltaSub0DX = double_to_dblflt(DeltaReal / S);
            DeltaSub0DY = double_to_dblflt(DeltaImaginary / S);
            DeltaSubNWX = double_to_dblflt(DeltaSubNWXNew / S);
            DeltaSubNWY = double_to_dblflt(DeltaSubNWYNew / S);
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

uint32_t GPURenderer::InitializeMemory(
    size_t w,
    size_t h)
{
    if ((local_width == w) &&
        (local_height == h)) {
        return 0;
    }

    if (local_width % NB_THREADS_W != 0) {
        return 100;
    }

    if (local_height % NB_THREADS_H != 0) {
        return 101;
    }

    width = (uint32_t)w;
    height = (uint32_t)h;
    local_width = width;
    local_height = height;
    w_block = local_width / NB_THREADS_W;
    h_block = local_height / NB_THREADS_H;
    N_cu = w_block * NB_THREADS_W * h_block * NB_THREADS_H;

    if (iter_matrix_cu != nullptr) {
        cudaFree(iter_matrix_cu);
    }

    cudaError_t err = cudaMallocManaged(&iter_matrix_cu, N_cu * sizeof(int), cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        ClearLocals();
        return err;
    }

    ClearMemory();
    return 0;
}

uint32_t GPURenderer::Render(
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
        return cudaSuccess;
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
        dblflt cx2{ cx.flt.x, cx.flt.y };
        dblflt cy2{ cy.flt.x, cy.flt.y };
        dblflt dx2{ dx.flt.x, dx.flt.y };
        dblflt dy2{ dy.flt.x, dy.flt.y };

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
        return cudaSuccess;
    }

    return ExtractIters(buffer);
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<float>* float_perturb,
    BLAS<float> * /*blas*/, // TODO
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<float> cudaResults(
        float_perturb->size,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu1x32Perturbed) {
        mandel_1x_float_perturb << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults,
            local_width, local_height, cx.floatOnly, cy.floatOnly, dx.floatOnly, dy.floatOnly,
            centerX.floatOnly, centerY.floatOnly,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<double>* double_perturb,
    BLAS<double> *blas,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64Perturbed) {
        MattPerturbSingleResults<double> cudaResults(
            double_perturb->size,
            double_perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        mandel_1x_double_perturb << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        result = ExtractIters(buffer);
    } else if (algorithm == RenderAlgorithm::Gpu1x64PerturbedBLA) {
        MattPerturbSingleResults<double> cudaResults(
            double_perturb->size,
            double_perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        GPUBLAS<double, BLA<double>> gpu_blas(blas->m_B, blas->m_LM2, blas->m_FirstLevel);
        result = gpu_blas.CheckValid();
        if (result != 0) {
            return result;
        }

        mandel_1x_double_perturb_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults,
            gpu_blas,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        result = ExtractIters(buffer);
    }
    //else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedScaled) {
    //    MattPerturbSingleResults<dblflt> cudaResults(
    //        Perturb->size,
    //        Perturb->iters);

    //    result = cudaResults.CheckValid();
    //    if (result != 0) {
    //        return result;
    //    }

    //    MattPerturbSingleResults<double> cudaResultsDouble(
    //        Perturb->size,
    //        Perturb->iters);

    //    result = cudaResultsDouble.CheckValid();
    //    if (result != 0) {
    //        return result;
    //    }

    //    mandel_2x_float_perturb_setup << <nb_blocks, threads_per_block >> > (cudaResults);

    //    mandel_2x_float_perturb_scaled << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
    //        cudaResults, cudaResultsDouble,
    //        local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
    //        centerX.doubleOnly, centerY.doubleOnly,
    //        n_iterations);

    //    result = ExtractIters(buffer);
    //}

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<double>* double_perturb,
    MattPerturbResults<float>* float_perturb,
    BLAS<double>* blas,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<float> cudaResults(
        float_perturb->size,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    MattPerturbSingleResults<double> cudaResultsDouble(
        double_perturb->size,
        double_perturb->iters);

    result = cudaResultsDouble.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaled) {
        mandel_1x_float_perturb_scaled << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults, cudaResultsDouble,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        result = ExtractIters(buffer);
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaledBLA) {
        GPUBLAS<double, BLA<double>> doubleGpuBlas(blas->m_B, blas->m_LM2, blas->m_FirstLevel);
        result = doubleGpuBlas.CheckValid();
        if (result != 0) {
            return result;
        }

        //GPUBLAS<float, BLA<double>> floatGpuBlas(blas->m_B, blas->m_LM2, blas->m_FirstLevel);
        //result = floatGpuBlas.CheckValid();
        //if (result != 0) {
        //    return result;
        //}

        mandel_1x_float_perturb_scaled_bla << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults, cudaResultsDouble, doubleGpuBlas,
            local_width, local_height, cx.doubleOnly, cy.doubleOnly, dx.doubleOnly, dy.doubleOnly,
            centerX.doubleOnly, centerY.doubleOnly,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<dblflt>* dblflt_perturb,
    BLAS<dblflt>* /*blas*/,  // TODO
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<dblflt> cudaResults(
        dblflt_perturb->size,
        dblflt_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu2x32Perturbed) {
        dblflt cx2{ cx.flt.x, cx.flt.y };
        dblflt cy2{ cy.flt.x, cy.flt.y };
        dblflt dx2{ dx.flt.x, dx.flt.y };
        dblflt dy2{ dy.flt.x, dy.flt.y };
        dblflt centerX2{ centerX.flt.x, centerX.flt.y };
        dblflt centerY2{ centerY.flt.x, centerY.flt.y };

        mandel_2x_float_perturb_setup << <nb_blocks, threads_per_block >> > (cudaResults);

        mandel_2x_float_perturb << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults,
            local_width, local_height, cx2, cy2, dx2, dy2,
            centerX2, centerY2,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<float>>* perturb,
    BLAS<HDRFloat<float>>* blas,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::GpuHDRx32PerturbedBLA) {
        MattPerturbSingleResults<HDRFloat<float>> cudaResults(
            perturb->size,
            perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        GPUBLAS<HDRFloat<float>, BLA<HDRFloat<float>>> gpu_blas(
            blas->m_B,
            blas->m_LM2,
            blas->m_FirstLevel);
        result = gpu_blas.CheckValid();
        if (result != 0) {
            return result;
        }

        mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

        mandel_1xHDR_float_perturb_bla<HDRFloat<float>> << <nb_blocks, threads_per_block >> > (
            iter_matrix_cu,
            cudaResults,
            gpu_blas,
            local_width, local_height, cx.hdrflt, cy.hdrflt, dx.hdrflt, dy.hdrflt,
            centerX.hdrflt, centerY.hdrflt,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<double>>* perturb,
    BLAS<HDRFloat<double>>* blas,
    MattCoords cx,
    MattCoords cy,
    MattCoords dx,
    MattCoords dy,
    MattCoords centerX,
    MattCoords centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::GpuHDRx64PerturbedBLA) {
        MattPerturbSingleResults<HDRFloat<double>> cudaResults(
            perturb->size,
            perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        GPUBLAS<HDRFloat<double>, BLA<HDRFloat<double>>> gpu_blas(blas->m_B, blas->m_LM2, blas->m_FirstLevel);
        result = gpu_blas.CheckValid();
        if (result != 0) {
            return result;
        }

        mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

        mandel_1xHDR_float_perturb_bla<HDRFloat<double>> << <nb_blocks, threads_per_block >> > (
            iter_matrix_cu,
            cudaResults,
            gpu_blas,
            local_width, local_height, cx.hdrdbl, cy.hdrdbl, dx.hdrdbl, dy.hdrdbl,
            centerX.hdrdbl, centerY.hdrdbl,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

uint32_t GPURenderer::ExtractIters(uint32_t* buffer) {
    const size_t ERROR_COLOR = 255;
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        cudaMemset(buffer, ERROR_COLOR, sizeof(int) * width * height);
        return result;
    }

    result = cudaMemcpy(buffer,
                        iter_matrix_cu,
                        sizeof(int) * N_cu,
                        cudaMemcpyDefault);
    if (result != cudaSuccess) {
        return result;
    }

    return cudaSuccess;
}