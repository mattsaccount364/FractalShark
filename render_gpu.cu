// TODO: 2x32 perturb is busted, do git diff
// Re-run  profile on current default view
#include <stdio.h>
#include <iostream>

#include "render_gpu.h"
#include "dbldbl.cuh"
#include "dblflt.cuh"
#include "../QuadDouble/gqd_basic.cuh"
#include "../QuadFloat/gqf_basic.cuh"

#include "GPU_BLAS.h"

#include "HDRFloatComplex.h"
#include "BLA.h"
#include "HDRFloat.h"

#include "GPU_LAReference.h"

#include "GPU_LAInfoDeep.h"
#include "LAReference.h"

#include <type_traits>
//#include <cuda/pipeline>
//#include <cuda_pipeline.h>

#ifdef __CUDACC__
__device__ __constant__ double twoPowExpDataDbl[2048];
__device__ __constant__ float twoPowExpDataFlt[256];

#ifdef __CUDA_ARCH__
__device__ void InitStatics()
{
    //LN2 = ::log(2);
    //LN2_REC = 1.0 / LN2;

    twoPowExpDbl = twoPowExpDataDbl;
    twoPowExpFlt = twoPowExpDataFlt;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    static constexpr int MaxFloatExponent = 127;
    static constexpr int MinFloatExponent = -126;

    //twoPowExp.resize(MaxDoubleExponent - MinDoubleExponent + 1);
    for (int i = MinDoubleExponent; i <= MaxDoubleExponent; i++) {
        double d = scalbn(1.0, i);
        int index = i - MinDoubleExponent;
        twoPowExpDbl[index] = d;
    }

    for (int i = MinFloatExponent; i <= MaxFloatExponent; i++) {
        float f = scalbn(1.0, i);
        int index = i - MinFloatExponent;
        twoPowExpFlt[index] = f;
    }
}
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Bilinear approximation
////////////////////////////////////////////////////////////////////////////////////////

template<class T>
CUDA_CRAP constexpr BLA<T>::BLA(T r2, T RealA, T ImagA, T RealB, T ImagB, int l)
    : Ax(RealA),
      Ay(ImagA),
      Bx(RealB),
      By(ImagB),
      r2(r2),
      l(l) {
    //HdrReduce(this->Ax);
    //HdrReduce(this->Ay);
    //HdrReduce(this->Bx);
    //HdrReduce(this->By);
    //HdrReduce(this->r2);
}

template<class T>
CUDA_CRAP void BLA<T>::getValue(
    T &RealDeltaSubN,
    T &ImagDeltaSubN,
    const T &RealDeltaSub0,
    const T &ImagDeltaSub0
) const {

    //T zxn = Ax * zx - Ay * zy + Bx * cx - By * cy;
    //T zyn = Ax * zy + Ay * zx + Bx * cy + By * cx;
    T NewRealValue = Ax * RealDeltaSubN - Ay * ImagDeltaSubN + Bx * RealDeltaSub0 - By * ImagDeltaSub0;
    T NewImagValue = Ax * ImagDeltaSubN + Ay * RealDeltaSubN + Bx * ImagDeltaSub0 + By * RealDeltaSub0;
    RealDeltaSubN = NewRealValue;
    //HdrReduce(RealDeltaSubN);

    ImagDeltaSubN = NewImagValue;
    //HdrReduce(ImagDeltaSubN);
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

template<class T>
CUDA_CRAP const T *BLA<T>::getR2Addr() const {
    return &r2;
}

template class BLA<float>;
template class BLA<double>;
template class BLA<HDRFloat<float>>;
template class BLA<HDRFloat<double>>;

////////////////////////////////////////////////////////////////////////////////////////
// Bilinear approximation.  GPU copy.
////////////////////////////////////////////////////////////////////////////////////////

template<class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<T, GPUBLA_TYPE, LM2>::GPU_BLAS(const std::vector<std::vector<GPUBLA_TYPE>>& B)
    : m_B(nullptr),
      m_Err(),
      m_Owned(true) {

    GPUBLA_TYPE** tempB;
    m_Err = cudaMallocManaged(&tempB, m_NumLevels * sizeof(GPUBLA_TYPE*), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    m_B = tempB;
    cudaMemset(m_B, 0, m_NumLevels * sizeof(GPUBLA_TYPE*));

    size_t total = 0;

    for (size_t i = 0; i < B.size(); i++) {
        total += sizeof(GPUBLA_TYPE) * B[i].size();
    }

    m_Err = cudaMalloc(&m_BMem, total);
    if (m_Err != cudaSuccess) {
        return;
    }

    size_t curTotal = 0;
    for (size_t i = 0; i < B.size(); i++) {
        m_B[i] = &m_BMem[curTotal];
        curTotal += B[i].size();

        cudaMemcpy(m_B[i],
            B[i].data(),
            sizeof(GPUBLA_TYPE) * B[i].size(),
            cudaMemcpyDefault);
    }
}

template<class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<T, GPUBLA_TYPE, LM2>::~GPU_BLAS() {
    if (m_Owned) {
        if (m_BMem != nullptr) {
            cudaFree(m_BMem);
            m_BMem = nullptr;
        }

        if (m_B != nullptr) {
            cudaFree(m_B);
            m_B = nullptr;
        }
    }
}

template<class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<T, GPUBLA_TYPE, LM2>::GPU_BLAS(const GPU_BLAS& other) : m_Owned(false) {
    if (this == &other) {
        return;
    }

    m_BMem = other.m_BMem;
    m_B = other.m_B;
    //for (size_t i = 0; i < m_NumLevels; i++) {
    //    m_B[i] = other.m_B[i];
    //}
}

template<class T, class GPUBLA_TYPE, int32_t LM2>
uint32_t GPU_BLAS<T, GPUBLA_TYPE, LM2>::CheckValid() const {
    return m_Err;
}

#ifdef __CUDA_ARCH__
template<class T, class GPUBLA_TYPE, int32_t LM2>
CUDA_CRAP const GPUBLA_TYPE* GPU_BLAS<T, GPUBLA_TYPE, LM2>::LookupBackwards(
    const GPUBLA_TYPE* __restrict__ *altB,
    //const GPUBLA_TYPE* __restrict__ nullBla,
    /*T* curBR2,*/
    size_t m,
    T z2) const {

    const int32_t k = (int32_t)m - 1;

    //// Option A:
    //const GPUBLA_TYPE* __restrict__ tempB = nullptr;
    //const float v = (float)(k & -k);
    //const uint32_t bits = *reinterpret_cast<const uint32_t * __restrict__>(&v);
    //const uint32_t zeros = (bits >> 23) - 0x7f;
    //uint32_t ix = k >> zeros;

    //// Option B: pretty similar results:
    //const GPUBLA_TYPE* __restrict__ tempB = nullptr;
    //uint32_t zeros;
    //uint32_t ix;
    //int r;           // result goes here
    //static constexpr int MultiplyDeBruijnBitPosition[32] =
    //{
    //  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
    //  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    //};
    //zeros = MultiplyDeBruijnBitPosition[((uint32_t)((k & -k) * 0x077CB531U)) >> 27];
    //ix = k >> zeros;

    //// Option C:
    // Get position of low-order 1 bit, subtract 1.
    //const GPUBLA_TYPE* __restrict__ tempB = nullptr;
    //const uint32_t zeros = __ffs(k) - 1;
    //uint32_t ix = k >> zeros;

    // Option D:
    // Reverse bit order, count high order zeros.
    const GPUBLA_TYPE* __restrict__ tempB = nullptr;
    const uint32_t zeros = __clz(__brev(k));
    uint32_t ix = k >> zeros;

    const int32_t startLevel =
        (LM2 == 0) ? 0 : (((zeros < LM2) ? zeros : LM2));

    //for (int32_t level = startLevel; level >= m_FirstLevel; --level) {
    //    __pipeline_memcpy_async(
    //        &curBR2[level],
    //        altB[level][ix].getR2Addr(),
    //        sizeof(T),
    //        0);
    //    ix = ix << 1;
    //}
    //__pipeline_commit();

    //ix = ixcopy;

    //__pipeline_wait_prior(0);

    for (int32_t level = startLevel; level >= m_FirstLevel; --level) {
        if (HdrCompareToBothPositiveReducedLT(z2, (tempB = &altB[level][ix])->getR2())) {
        //if (z2 < curBR2[level]) {
            return tempB;
        }
        ix = ix << 1;
    }
    return nullptr;
    //return nullBla;

    //GPUBLA_TYPE* __restrict__ tempB = nullptr;
    //uint32_t zeros;
    //uint32_t ix;
    //float v = (float)(k & -k);
    //uint32_t bits = *reinterpret_cast<const uint32_t * __restrict__>(&v);
    //zeros = (bits >> 23) - 0x7f;
    //ix = k >> zeros;
    //int32_t startLevel = ((zeros <= m_LM2) ? zeros : m_LM2);
    //ix = ix << (startLevel - m_FirstLevel);
    //for (int32_t level = m_FirstLevel; level <= startLevel; level++) {
    //    tempB = (z2 < m_B[level][ix].getR2()) ? &m_B[level][ix] : tempB;
    //    ix = ix >> 1;
    //}
    //return tempB;
}
#endif

#define LargeSwitch \
        switch (blas->m_LM2) {                                         \
        case  0: result = Run.template operator()<0> (); break;     \
        case  1: result = Run.template operator()<1> (); break;     \
        case  2: result = Run.template operator()<2> (); break;     \
        case  3: result = Run.template operator()<3> (); break;     \
        case  4: result = Run.template operator()<4> (); break;     \
        case  5: result = Run.template operator()<5> (); break;     \
        case  6: result = Run.template operator()<6> (); break;     \
        case  7: result = Run.template operator()<7> (); break;     \
        case  8: result = Run.template operator()<8> (); break;     \
        case  9: result = Run.template operator()<9> (); break;     \
        case 10: result = Run.template operator()<10> (); break;    \
        case 11: result = Run.template operator()<11> (); break;    \
        case 12: result = Run.template operator()<12> (); break;    \
        case 13: result = Run.template operator()<13> (); break;    \
        case 14: result = Run.template operator()<14> (); break;    \
        case 15: result = Run.template operator()<15> (); break;    \
        case 16: result = Run.template operator()<16> (); break;    \
        case 17: result = Run.template operator()<17> (); break;    \
        case 18: result = Run.template operator()<18> (); break;    \
        case 19: result = Run.template operator()<19> (); break;    \
        case 20: result = Run.template operator()<20> (); break;    \
        case 21: result = Run.template operator()<21> (); break;    \
        case 22: result = Run.template operator()<22> (); break;    \
        case 23: result = Run.template operator()<23> (); break;    \
        case 24: result = Run.template operator()<24> (); break;    \
        case 25: result = Run.template operator()<25> (); break;    \
        case 26: result = Run.template operator()<26> (); break;    \
        case 27: result = Run.template operator()<27> (); break;    \
        case 28: result = Run.template operator()<28> (); break;    \
        case 29: result = Run.template operator()<29> (); break;    \
        case 30: result = Run.template operator()<30> (); break;    \
        case 31: result = Run.template operator()<31> (); break;    \
        default: break;                                                \
        }

////////////////////////////////////////////////////////////////////////////////////////
// Perturbation results
////////////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(MattReferenceSingleIter<float>) == 16, "Float");
static_assert(sizeof(MattReferenceSingleIter<double>) == 24, "Double");
static_assert(sizeof(MattReferenceSingleIter<dblflt>) == 24, "Dblflt");

//char(*__kaboom1)[sizeof(MattReferenceSingleIter<float>)] = 1;
//char(*__kaboom2)[sizeof(MattReferenceSingleIter<double>)] = 1;
//char(*__kaboom3)[sizeof(MattReferenceSingleIter<dblflt>)] = 1;

template<typename Type>
struct MattPerturbSingleResults {
    MattReferenceSingleIter<Type>* __restrict__ iters;
    size_t size;
    bool own;
    cudaError_t err;
    size_t PeriodMaybeZero;

    MattPerturbSingleResults(
        size_t sz,
        size_t PeriodMaybeZero,
        MattReferenceSingleIter<Type> *in_iters)
        : size(sz),
        PeriodMaybeZero(PeriodMaybeZero),
        iters(nullptr),
        own(true),
        err(cudaSuccess) {

        static_assert(sizeof(MattDblflt) == sizeof(dblflt), "No");

        MattReferenceSingleIter<Type>* tempIters;
        err = cudaMalloc(&tempIters, size * sizeof(MattReferenceSingleIter<Type>));
        if (err != cudaSuccess) {
            size = 0;
            return;
        }

        iters = tempIters;
        cudaMemcpy(iters, in_iters, size * sizeof(MattReferenceSingleIter<Type>), cudaMemcpyDefault);

        //err = cudaMemAdvise(iters,
        //    size * sizeof(MattReferenceSingleIter<Type>),
        //    cudaMemAdviseSetReadMostly,
        //    0);
        //if (err != cudaSuccess) {
        //    size = 0;
        //    return;
        //}
    }

    // funny semantics, copy doesn't own the pointers.
    MattPerturbSingleResults(const MattPerturbSingleResults& other) {
        if (this == &other) {
            return;
        }

        iters = other.iters;
        size = other.size;
        PeriodMaybeZero = other.PeriodMaybeZero;
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


//////////////////////////////////////////////////////////////////////////////

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
struct SharedMemStruct {
    using GPUBLA_TYPE = BLA<HDRFloatType>;
    const GPUBLA_TYPE* __restrict__ altB[32];
    //GPUBLA_TYPE nullBla;
    //struct {
    //    //HDRFloatType curBR2[16];
    //    //MattReferenceSingleIter<HDRFloatType> CurResult;
    //    //HDRFloatType NextX1;
    //    //HDRFloatType NextY1;
    //} PerThread[NB_THREADS_W][NB_THREADS_H];
};

template<int32_t LM2>
__global__
void mandel_1x_double_perturb_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<double> PerturbDouble,
    GPU_BLAS<double, BLA<double>, LM2> doubleBlas,
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

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    using GPUBLA_TYPE = BLA<double>;
    char __shared__ SharedMem[sizeof(SharedMemStruct<double>)];
    auto* shared =
        reinterpret_cast<SharedMemStruct<double>*>(SharedMem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        GPUBLA_TYPE** elts = doubleBlas.GetB();

        for (size_t i = 0; i < doubleBlas.m_NumLevels; i++) {
            shared->altB[i] = elts[i];
        }
    }

    __syncthreads();

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
        const BLA<double>* b = nullptr;
        while ((b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared)) != nullptr) {
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

        DeltaSubNX = DeltaSubNXOrig * (PerturbDouble.iters[RefIteration].x * 2 + DeltaSubNXOrig) -
            DeltaSubNYOrig * (PerturbDouble.iters[RefIteration].y * 2 + DeltaSubNYOrig) +
            DeltaSub0X;
        DeltaSubNY = DeltaSubNXOrig * (PerturbDouble.iters[RefIteration].y * 2 + DeltaSubNYOrig) +
            DeltaSubNYOrig * (PerturbDouble.iters[RefIteration].x * 2 + DeltaSubNXOrig) +
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

//#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

//#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
//#define PXL_GLOBAL_PTR   "l"
//#else
//#define PXL_GLOBAL_PTR   "r"
//#endif

//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l1(const void* const ptr)
//{
//    asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}
//
//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
//{
//    asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}
//
//DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l2(const void* const ptr)
//{
//    asm("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
//}

template<class HDRFloatType, int32_t LM2>
__global__
void
//__launch_bounds__(NB_THREADS_W * NB_THREADS_H, 2)
mandel_1xHDR_float_perturb_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<HDRFloatType> Perturb,
    GPU_BLAS<HDRFloatType, BLA<HDRFloatType>, LM2> blas,
    int width,
    int height,
    const HDRFloatType cx,
    const HDRFloatType cy,
    const HDRFloatType dx,
    const HDRFloatType dy,
    const HDRFloatType centerX,
    const HDRFloatType centerY,
    uint32_t n_iterations)
{
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    //if (iter_matrix[idx] != 0) { 
    //    return;
    //}

    using GPUBLA_TYPE = BLA<HDRFloatType>;
    char __shared__ SharedMem[sizeof(SharedMemStruct<HDRFloatType>)];
    auto* shared =
        reinterpret_cast<SharedMemStruct<HDRFloatType>*>(SharedMem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        GPUBLA_TYPE**elts = blas.GetB();

        for (size_t i = 0; i < blas.m_NumLevels; i++) {
            shared->altB[i] = elts[i];
        }

        //const GPUBLA_TYPE nullBla{
        //    HDRFloatType(),
        //    HDRFloatType(1.0f),
        //    HDRFloatType(),
        //    HDRFloatType(),
        //    HDRFloatType(),
        //    0 };
        //shared->nullBla = nullBla;
    }

    __syncthreads();

    size_t iter = 0;
    size_t RefIteration = 0;
    const HDRFloatType DeltaReal = dx * X - centerX;
    const HDRFloatType DeltaImaginary = -dy * Y - centerY;

    const HDRFloatType DeltaSub0X = DeltaReal;
    const HDRFloatType DeltaSub0Y = DeltaImaginary;
    HDRFloatType DeltaSubNX = HDRFloatType(0);
    HDRFloatType DeltaSubNY = HDRFloatType(0);
    HDRFloatType DeltaNormSquared = HDRFloatType(0);
    const HDRFloatType Two = HDRFloatType(2);

    while (iter < n_iterations) {
        //auto* next1X = &shared->PerThread[threadIdx.x][threadIdx.y].NextX1;
        //auto* next1Y = &shared->PerThread[threadIdx.x][threadIdx.y].NextY1;
        //__pipeline_memcpy_async(
        //    next1X,
        //    &Perturb.iters[RefIteration + 1].x,
        //    sizeof(Perturb.iters[RefIteration + 1].x),
        //    0);
        //__pipeline_memcpy_async(
        //    next1Y,
        //    &Perturb.iters[RefIteration + 1].y,
        //    sizeof(Perturb.iters[RefIteration + 1].y),
        //    0);
        //__pipeline_commit();

        const HDRFloatType DeltaSubNXOrig = DeltaSubNX;
        const HDRFloatType DeltaSubNYOrig = DeltaSubNY;

        //__prefetch_global_l2(&Perturb.iters[RefIteration + 1].x);
        //__prefetch_global_l2(&Perturb.iters[RefIteration + 1].y);

        const auto tempMulX2 = Perturb.iters[RefIteration].x * Two;
        const auto tempMulY2 = Perturb.iters[RefIteration].y * Two;

        ++RefIteration;

        //if (RefIteration >= Perturb.size) {
        //    // TODO this first RefIteration + l check bugs me
        //    iter = 255;
        //    break;
        //}

        const auto tempSum1 = (tempMulY2 + DeltaSubNYOrig);
        const auto tempSum2 = (tempMulX2 + DeltaSubNXOrig);

        //DeltaSubNX = DeltaSubNXOrig * tempSum2 -
        //    DeltaSubNYOrig * tempSum1 +
        //    DeltaSub0X;
        //HdrReduce(DeltaSubNX);

        //DeltaSubNX = HDRFloatType::custom_perturb1<false>(
        //    DeltaSubNXOrig,
        //    tempSum2,
        //    DeltaSubNYOrig,
        //    tempSum1,
        //    DeltaSub0X);

        //DeltaSubNY = HDRFloatType::custom_perturb1<true>(
        //    DeltaSubNXOrig,
        //    tempSum1,
        //    DeltaSubNYOrig,
        //    tempSum2,
        //    DeltaSub0Y);

        HDRFloatType::custom_perturb2(
            DeltaSubNX,
            DeltaSubNY,
            DeltaSubNXOrig,
            tempSum2,
            DeltaSubNYOrig,
            tempSum1,
            DeltaSub0X,
            DeltaSub0Y);

        //DeltaSubNY = DeltaSubNXOrig * tempSum1 +
        //    DeltaSubNYOrig * tempSum2 +
        //    DeltaSub0Y;
        //HdrReduce(DeltaSubNY);

        //__pipeline_wait_prior(0);

        const auto tempVal1X = Perturb.iters[RefIteration].x;
        const auto tempVal1Y = Perturb.iters[RefIteration].y;

        const HDRFloatType tempZX = tempVal1X + DeltaSubNX;
        const HDRFloatType tempZY = tempVal1Y + DeltaSubNY;
        HDRFloatType normSquared = tempZX * tempZX + tempZY * tempZY;
        HdrReduce(normSquared);

        if (HdrCompareToBothPositiveReducedLT<HDRFloatType, 256>(normSquared) && iter < n_iterations) {
            DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
            HdrReduce(DeltaNormSquared);

            if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                RefIteration >= Perturb.size - 1) {
                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;
                DeltaNormSquared = normSquared;
                RefIteration = 0;
            }

            ++iter;
        }
        else {
            break;
        }

        const BLA<HDRFloatType>* b = nullptr;

        for (;;) {
            b = blas.LookupBackwards(
                shared->altB,
                /*shared->PerThread[threadIdx.x][threadIdx.y].curBR2,*/
                //&shared->nullBla,
                RefIteration,
                DeltaNormSquared);
            if (b == nullptr) {
                break;
            }

            const int l = b->getL();

            // TODO this first RefIteration + l check bugs me
            const bool res1 = (RefIteration + l >= Perturb.size);
            const bool res2 = (iter + l >= n_iterations);
            const bool res3 = (RefIteration + l < Perturb.size - 1);
            //const bool res4 = l == 0; // nullBla
            const bool res12 = (/*res4 || */res1 || res2) == false;
            if (res12 && res3) {
                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                HdrReduce(DeltaNormSquared);
                continue;
            }
            else if (res12 && !res3) {
                iter += l;
                RefIteration += l;

                b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                //__pipeline_memcpy_async(
                //    &shared->PerThread[threadIdx.x][threadIdx.y].CurResult.x,
                //    &Perturb.iters[RefIteration].x,
                //    sizeof(Perturb.iters[RefIteration].x),
                //    0);
                //__pipeline_memcpy_async(
                //    &shared->PerThread[threadIdx.x][threadIdx.y].CurResult.y,
                //    &Perturb.iters[RefIteration].y,
                //    sizeof(Perturb.iters[RefIteration].y),
                //    0);
                //__pipeline_commit();
                //__pipeline_wait_prior(0);

                //HDRFloatType tempZX = shared->PerThread[threadIdx.x][threadIdx.y].CurResult.x + DeltaSubNX;
                //HDRFloatType tempZY = shared->PerThread[threadIdx.x][threadIdx.y].CurResult.y + DeltaSubNY;
                HDRFloatType tempZX = Perturb.iters[RefIteration].x + DeltaSubNX;
                HDRFloatType tempZY = Perturb.iters[RefIteration].y + DeltaSubNY;

                DeltaSubNX = tempZX;
                DeltaSubNY = tempZY;

                DeltaNormSquared = tempZX.square_mutable() + tempZY.square_mutable();
                HdrReduce(DeltaNormSquared);
                RefIteration = 0;
                break;
            }
            else {
                break;
            }
        }
    }

    iter_matrix[idx] = (uint32_t)iter;
}

template<class SubType>
__global__
void
//__launch_bounds__(NB_THREADS_W * NB_THREADS_H, 2)
mandel_1xHDR_float_perturb_lav2(uint32_t* iter_matrix,
    MattPerturbSingleResults<HDRFloat<SubType>> Perturb,
    GPU_LAReference<SubType> LaReference, // "copy"
    int width,
    int height,
    const HDRFloat<SubType> cx,
    const HDRFloat<SubType> cy,
    const HDRFloat<SubType> dx,
    const HDRFloat<SubType> dy,
    const HDRFloat<SubType> centerX,
    const HDRFloat<SubType> centerY,
    uint32_t n_iterations)
{
    using HDRFloatType = HDRFloat<SubType>;

    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    int32_t idx = width * Y + X;
    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    int32_t iter = 0;
    int32_t RefIteration = 0;
    const HDRFloatType DeltaReal = dx * X - centerX;
    const HDRFloatType DeltaImaginary = -dy * Y - centerY;

    const HDRFloatType DeltaSub0X = DeltaReal;
    const HDRFloatType DeltaSub0Y = DeltaImaginary;
    HDRFloatType DeltaSubNX = HDRFloatType(0);
    HDRFloatType DeltaSubNY = HDRFloatType(0);

    using TComplex = HDRFloatComplex<SubType>;

    ////////////
    TComplex DeltaSub0;
    TComplex DeltaSubN;

    DeltaSub0 = { DeltaReal, DeltaImaginary };
    DeltaSubN = { 0, 0 };

    if (LaReference.isValid && LaReference.UseAT && LaReference.AT.isValid(DeltaSub0)) {
        ATResult<SubType> res;
        LaReference.AT.PerformAT(n_iterations, DeltaSub0, res);
        iter = res.bla_iterations;
        DeltaSubN = res.dz;
    }

    int32_t MaxRefIteration = Perturb.size - 1;
    TComplex complex0{ DeltaReal, DeltaImaginary };
    int32_t CurrentLAStage{ LaReference.isValid ? LaReference.LAStageCount : 0 };

    if (iter != 0 && RefIteration < MaxRefIteration) {
        complex0 = TComplex{ Perturb.iters[RefIteration].x, Perturb.iters[RefIteration].y } + DeltaSubN;
    }
    else if (iter != 0 && Perturb.PeriodMaybeZero != 0) {
        RefIteration = RefIteration % Perturb.PeriodMaybeZero;
        complex0 = TComplex{ Perturb.iters[RefIteration].x, Perturb.iters[RefIteration].y } + DeltaSubN;
    }

    while (CurrentLAStage > 0) {
        CurrentLAStage--;

        const int32_t LAIndex{ LaReference.getLAIndex(CurrentLAStage) };

        if (LaReference.isLAStageInvalid(LAIndex, DeltaSub0)) {
            continue;
        }

        const int32_t MacroItCount{ LaReference.getMacroItCount(CurrentLAStage) };
        int32_t j = RefIteration;

        while (iter < n_iterations) {
            const GPU_LAstep las{ LaReference.getLA(LAIndex, DeltaSubN, j, iter, n_iterations) };

            if (las.unusable) {
                RefIteration = las.nextStageLAindex;
                break;
            }

            iter += las.step;
            DeltaSubN = las.Evaluate(DeltaSub0);
            complex0 = las.getZ(DeltaSubN);
            j++;

            const auto complex0Norm{ HdrReduce(complex0.chebychevNorm()) };
            const auto DeltaSubNNorm{ HdrReduce(DeltaSubN.chebychevNorm()) };
            if (HdrCompareToBothPositiveReducedLT(complex0Norm, DeltaSubNNorm) || j >= MacroItCount) {
                DeltaSubN = complex0;
                j = 0;
            }
        }

        if (iter >= n_iterations) {
            break;
        }
    }

    DeltaSubNX = DeltaSubN.getRe();
    DeltaSubNY = DeltaSubN.getIm();

    //////////////////////

    auto perturbLoop = [&](uint32_t maxIterations) {
        for (;;) {
            const HDRFloatType DeltaSubNXOrig{ DeltaSubNX };
            const HDRFloatType DeltaSubNYOrig{ DeltaSubNY };

            const auto tempMulX2{ Perturb.iters[RefIteration].x.multiply2() };
            const auto tempMulY2{ Perturb.iters[RefIteration].y.multiply2() };

            ++RefIteration;

            const auto tempSum1{ tempMulY2 + DeltaSubNYOrig };
            const auto tempSum2{ tempMulX2 + DeltaSubNXOrig };

            HDRFloatType::custom_perturb2(
                DeltaSubNX,
                DeltaSubNY,
                DeltaSubNXOrig,
                tempSum2,
                DeltaSubNYOrig,
                tempSum1,
                DeltaSub0X,
                DeltaSub0Y);

            const auto tempVal1X{ Perturb.iters[RefIteration].x };
            const auto tempVal1Y{ Perturb.iters[RefIteration].y };

            const HDRFloatType tempZX{ tempVal1X + DeltaSubNX };
            const HDRFloatType tempZY{ tempVal1Y + DeltaSubNY };
            const HDRFloatType normSquared{ HdrReduce(tempZX.square() + tempZY.square())};

            if (HdrCompareToBothPositiveReducedLT<HDRFloatType, 256>(normSquared) && iter < maxIterations) {
                const auto DeltaNormSquared{ HdrReduce(DeltaSubNX.square() + DeltaSubNY.square()) };

                if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                    RefIteration >= Perturb.size - 1) {
                    DeltaSubNX = tempZX;
                    DeltaSubNY = tempZY;
                    RefIteration = 0;
                }

                ++iter;
            }
            else {
                break;
            }
        }
    };

//    for (;;) {
        __syncthreads();

        //bool differences = false;
        int maxRefIteration = 0;
        const int XStart = blockIdx.x * blockDim.x;
        const int YStart = blockIdx.y * blockDim.y;

        int32_t curidx = width * YStart + XStart;
        maxRefIteration = iter_matrix[curidx];

        for (auto yiter = YStart; yiter < YStart + blockDim.y; yiter++) {
            for (auto xiter = XStart; xiter < XStart + blockDim.x; xiter++) {
                curidx = width * yiter + xiter;
                if (maxRefIteration < iter_matrix[curidx]) {
                    //differences = true;
                    maxRefIteration = iter_matrix[curidx];
                }
            }
        }

        //if (differences == false) {
        //    break;
        //}

        __syncthreads();

        // Give it one chance to synchronize memory access
        if (RefIteration < maxRefIteration) {
            perturbLoop(maxRefIteration);
        }
    //}

    __syncthreads();

    perturbLoop(n_iterations);

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
        PerturbDblFlt.iters[i].y = add_float_to_dblflt(PerturbDblFlt.iters[i].y.y, PerturbDblFlt.iters[i].y.x);
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

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    int iter = 0;
    size_t RefIteration = 0;

    dblflt Two = add_float_to_dblflt(0, 2);

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

        const dblflt tempX = mul_dblflt(CurIter->x, Two);
        const dblflt tempY = mul_dblflt(CurIter->y, Two);

        const dblflt tempTermX1 = add_dblflt(tempX, DeltaSubNXOrig);
        const dblflt tempTermX2 = add_dblflt(tempY, DeltaSubNYOrig);

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

template<class T, bool Periodic>
__global__
void mandel_1x_float_perturb(uint32_t* iter_matrix,
    MattPerturbSingleResults<T> PerturbFloat,
    int width,
    int height,
    T cx,
    T cy,
    T dx,
    T dy,
    T centerX,
    T centerY,
    uint32_t n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    //size_t idx = width * (height - Y - 1) + X;
    size_t idx = width * Y + X;

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    size_t iter = 0;
    size_t RefIteration = 0;
    T DeltaReal = dx * X - centerX;
    T DeltaImaginary = -dy * Y - centerY;

    T DeltaSub0X = DeltaReal;
    T DeltaSub0Y = DeltaImaginary;
    T DeltaSubNX = T(0.0);
    T DeltaSubNY = T(0.0);
    size_t MaxRefIteration = PerturbFloat.size - 1;

    //T dzdcX = max(max(x.dzdc), 1.0f);
    T scalingFactor = T(1.0f) / (HdrMaxPositiveReduced(HdrMaxPositiveReduced(HdrAbs(dx), HdrAbs(dy)), T(1.0f)));
    //T scalingFactor = 1.0f;
    T dzdcX = scalingFactor;
    T dzdcY = T(0.0);

    T maxRadius = HdrMaxPositiveReduced(HdrAbs(dx), HdrAbs(dy));
    T maxRadiusSq = maxRadius * maxRadius;

    while (iter < n_iterations) {
        const MattReferenceSingleIter<T> *curIter = &PerturbFloat.iters[RefIteration];

        const T DeltaSubNXOrig = DeltaSubNX;
        const T DeltaSubNYOrig = DeltaSubNY;

        const T tempSubX = curIter->x * T(2) + DeltaSubNXOrig;
        const T tempSubY = curIter->y * T(2) + DeltaSubNYOrig;

        ++RefIteration;
        curIter = &PerturbFloat.iters[RefIteration];

        DeltaSubNX =
            DeltaSubNXOrig * tempSubX -
            DeltaSubNYOrig * tempSubY +
            DeltaSub0X;
        HdrReduce(DeltaSubNX);

        DeltaSubNY =
            DeltaSubNXOrig * tempSubY +
            DeltaSubNYOrig * tempSubX +
            DeltaSub0Y;
        HdrReduce(DeltaSubNY);

        const T tempZX = curIter->x + DeltaSubNX;
        const T tempZY = curIter->y + DeltaSubNY;
        
        T zn_size = tempZX * tempZX + tempZY * tempZY;
        HdrReduce(zn_size);

        T normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedGE(zn_size, T(256))) {
            break;
        }

        // Just finds the interesting Misiurewicz points.  Breaks so they're colored differently
        if constexpr (Periodic) {
            auto n3 = maxRadiusSq * (dzdcX * dzdcX + dzdcY * dzdcY);
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedGE(zn_size, n3)) {
                // dzdc = dzdc * 2.0 * z + ScalingFactor;
                // dzdc = dzdc * 2.0 * tempZ + ScalingFactor;
                // dzdc = (dzdcX + dzdcY * i) * 2.0 * (tempZX + tempZY * i) + ScalingFactor;
                // dzdc = (dzdcX * 2.0 + dzdcY * i * 2.0) * (tempZX + tempZY * i) + ScalingFactor;
                // dzdc = (dzdcX * 2.0) * tempZX +
                //        (dzdcX * 2.0) * (tempZY * i) +
                //        (dzdcY * i * 2.0) * tempZX +
                //        (dzdcY * i * 2.0) * tempZY * i
                //
                // dzdcX = (dzdcX * 2.0) * tempZX -
                //         (dzdcY * 2.0) * tempZY
                // dzdcY = (dzdcX * 2.0) * (tempZY) +
                //         (dzdcY * 2.0) * tempZX
                auto dzdcXOrig = dzdcX;
                dzdcX = T(2.0f) * tempZX * dzdcX - T(2.0f) * tempZY * dzdcY + scalingFactor;
                HdrReduce(dzdcX);

                dzdcY = T(2.0f) * tempZY * dzdcXOrig + T(2.0f) * tempZX * dzdcY;
                HdrReduce(dzdcY);
            }
            else {
                //iter = n_iterations;
                break;
            }
        }

        if (HdrCompareToBothPositiveReducedLE(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        ++iter;
    }

    iter_matrix[idx] = (uint32_t)iter;
}

template<class T>
__global__
void mandel_1x_float_perturb_scaled(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> PerturbFloat,
    MattPerturbSingleResults<T> PerturbDouble,
    int width,
    int height,
    T cx,
    T cy,
    T dx,
    T dy,
    T centerX,
    T centerY,
    uint32_t n_iterations)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    const float LARGE_MANTISSA = 1e30;

    if (X >= width || Y >= height)
        return;

    size_t idx = width * Y + X;

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    size_t iter = 0;
    size_t RefIteration = 0;
    T DeltaReal = dx * X - centerX;
    HdrReduce(DeltaReal);

    T DeltaImaginary = -dy * Y - centerY;
    HdrReduce(DeltaImaginary);

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

    T S = HdrSqrt(DeltaReal * DeltaReal + DeltaImaginary * DeltaImaginary);
    HdrReduce(S);

    //double S = 1;
    float DeltaSub0DX = (float)(DeltaReal / S);
    float DeltaSub0DY = (float)(DeltaImaginary / S);
    float DeltaSubNWX = 0;
    float DeltaSubNWY = 0;

    float s = (float)S;
    float twos = 2 * s;
    const float w2threshold = exp(log(LARGE_MANTISSA) / 2);
    size_t MaxRefIteration = PerturbFloat.size - 1;

    T TwoFiftySix = T(256.0);

    while (iter < n_iterations) {
        const MattReferenceSingleIter<float> *curFloatIter = &PerturbFloat.iters[RefIteration];
        const MattReferenceSingleIter<T> *curDoubleIter = &PerturbDouble.iters[RefIteration];

        if (curFloatIter->bad == false) {
            const float DeltaSubNWXOrig = DeltaSubNWX;
            const float DeltaSubNWYOrig = DeltaSubNWY;

            DeltaSubNWX =
                DeltaSubNWXOrig * curFloatIter->x * 2 -
                DeltaSubNWYOrig * curFloatIter->y * 2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y * 2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x * 2 +
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

            T DoubleTempZX;
            T DoubleTempZY;

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
                DoubleTempZX = (curDoubleIter->x + (T)DeltaSubNWX * S); // Xxrd, xr
                //HdrReduce(DoubleTempZX);
                DoubleTempZY = (curDoubleIter->y + (T)DeltaSubNWY * S); // Xxid, xi
                //HdrReduce(DoubleTempZY);

                RefIteration = 0;
                S = HdrSqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                HdrReduce(S);
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
                DoubleTempZX = (T)DeltaSubNWX * S;
                //HdrReduce(DoubleTempZX);
                DoubleTempZY = (T)DeltaSubNWY * S;
                //HdrReduce(DoubleTempZY);

                S = HdrSqrt(DoubleTempZX * DoubleTempZX + DoubleTempZY * DoubleTempZY);
                HdrReduce(S);
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
            T DeltaSubNWXOrig = (T)DeltaSubNWX;
            T DeltaSubNWYOrig = (T)DeltaSubNWY;

            T DoubleTempDeltaSubNWX = DeltaSubNWXOrig * curDoubleIter->x * 2;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX -= DeltaSubNWYOrig * curDoubleIter->y * 2;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX += S * DeltaSubNWXOrig * DeltaSubNWXOrig;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX -= S * DeltaSubNWYOrig * DeltaSubNWYOrig;
            //HdrReduce(DoubleTempDeltaSubNWX);
            DoubleTempDeltaSubNWX += DeltaReal / S;
            HdrReduce(DoubleTempDeltaSubNWX);

            T DoubleTempDeltaSubNWY = DeltaSubNWXOrig * (curDoubleIter->y * 2 + T(2) * S * DeltaSubNWYOrig);
            //HdrReduce(DoubleTempDeltaSubNWY);
            DoubleTempDeltaSubNWY += DeltaSubNWYOrig * curDoubleIter->x * 2;
            //HdrReduce(DoubleTempDeltaSubNWY);
            DoubleTempDeltaSubNWY += DeltaImaginary / S;
            HdrReduce(DoubleTempDeltaSubNWY);

            ++RefIteration;
            curFloatIter = &PerturbFloat.iters[RefIteration];
            curDoubleIter = &PerturbDouble.iters[RefIteration];

            const T tempZX =
                curDoubleIter->x +
                DoubleTempDeltaSubNWX * S; // Xxrd

            const T tempZY =
                curDoubleIter->y +
                DoubleTempDeltaSubNWY * S; // Xxid

            T zn_size =
                tempZX * tempZX + tempZY * tempZY;
            HdrReduce(zn_size);

            if (!HdrCompareToBothPositiveReducedLT<T, 256>(zn_size)) {
                break;
            }

            const T TwoS = S * S;
            T normDeltaSubN =
                DoubleTempDeltaSubNWX * DoubleTempDeltaSubNWX * TwoS +
                DoubleTempDeltaSubNWY * DoubleTempDeltaSubNWY * TwoS;
            HdrReduce(normDeltaSubN);

            T DeltaSubNWXNew;
            T DeltaSubNWYNew;

            if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
                RefIteration == MaxRefIteration) {
                DeltaSubNWXNew = (curDoubleIter->x + DoubleTempDeltaSubNWX * S); // Xxrd, xr
                DeltaSubNWYNew = (curDoubleIter->y + DoubleTempDeltaSubNWY * S); // Xxid, xi

                RefIteration = 0;
            }
            else {
                DeltaSubNWXNew = DoubleTempDeltaSubNWX * S;
                DeltaSubNWYNew = DoubleTempDeltaSubNWY * S;
            }

            S = HdrSqrt(DeltaSubNWXNew * DeltaSubNWXNew + DeltaSubNWYNew * DeltaSubNWYNew);
            HdrReduce(S);
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

template<int32_t LM2>
__global__
void mandel_1x_float_perturb_scaled_bla(uint32_t* iter_matrix,
    MattPerturbSingleResults<float> PerturbFloat,
    MattPerturbSingleResults<double> PerturbDouble,
    GPU_BLAS<double, BLA<double>, LM2> doubleBlas,
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

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

    using GPUBLA_TYPE = BLA<double>;
    char __shared__ SharedMem[sizeof(SharedMemStruct<double>)];
    auto* shared =
        reinterpret_cast<SharedMemStruct<double>*>(SharedMem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        GPUBLA_TYPE** elts = doubleBlas.GetB();

        for (size_t i = 0; i < doubleBlas.m_NumLevels; i++) {
            shared->altB[i] = elts[i];
        }
    }

    __syncthreads();

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
        const MattReferenceSingleIter<float> *curFloatIter = &PerturbFloat.iters[RefIteration];
        const MattReferenceSingleIter<double> *curDoubleIter = &PerturbDouble.iters[RefIteration];

        double DeltaSubNX = DeltaSubNWX * S;
        double DeltaSubNY = DeltaSubNWY * S;

        const BLA<double>* b = nullptr;

        b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared);
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

                b = doubleBlas.LookupBackwards(shared->altB, RefIteration, DeltaNormSquared);
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
                DeltaSubNWXOrig * curFloatIter->x * 2 -
                DeltaSubNWYOrig * curFloatIter->y * 2 +
                s * DeltaSubNWXOrig * DeltaSubNWXOrig - s * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaSub0DX;

            DeltaSubNWY =
                DeltaSubNWXOrig * (curFloatIter->y * 2 + twos * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curFloatIter->x * 2 +
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
                DeltaSubNWXOrig * curDoubleIter->x * 2 -
                DeltaSubNWYOrig * curDoubleIter->y * 2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y * 2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x * 2 +
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

    //if (iter_matrix[idx] != 0) {
    //    return;
    //}

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

            const dblflt Two = add_float_to_dblflt(0, 2);
            const dblflt tempX = mul_dblflt(curDblFloatIter->x, Two);
            const dblflt tempY = mul_dblflt(curDblFloatIter->y, Two);

            DeltaSubNWX = mul_dblflt(DeltaSubNWXOrig, tempX);
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(DeltaSubNWYOrig, tempY));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWXOrig), DeltaSubNWXOrig));
            DeltaSubNWX = sub_dblflt(DeltaSubNWX, mul_dblflt(mul_dblflt(s, DeltaSubNWYOrig), DeltaSubNWYOrig));
            DeltaSubNWX = add_dblflt(DeltaSubNWX, DeltaSub0DX);

            DeltaSubNWY = mul_dblflt(DeltaSubNWXOrig, (add_dblflt(tempY, mul_dblflt(twos, DeltaSubNWYOrig))));
            DeltaSubNWY = add_dblflt(DeltaSubNWY, mul_dblflt(DeltaSubNWYOrig, tempX));
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
                DeltaSubNWXOrig * curDoubleIter->x * 2 -
                DeltaSubNWYOrig * curDoubleIter->y * 2 +
                S * DeltaSubNWXOrig * DeltaSubNWXOrig - S * DeltaSubNWYOrig * DeltaSubNWYOrig +
                DeltaReal / S;

            const double DoubleTempDeltaSubNWY =
                DeltaSubNWXOrig * (curDoubleIter->y * 2 + 2 * S * DeltaSubNWYOrig) +
                DeltaSubNWYOrig * curDoubleIter->x * 2 +
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
        return 10000;
    }

    if (local_height % NB_THREADS_H != 0) {
        return 10001;
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

template<class T>
uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<T> cx,
    MattCoords<T> cy,
    MattCoords<T> dx,
    MattCoords<T> dy,
    uint32_t n_iterations,
    int iteration_precision)
{
    if (iter_matrix_cu == nullptr) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64) {
        // all are doubleOnly
        if constexpr (std::is_same<T, double>::value) {
            switch (iteration_precision) {
            case 1:
                mandel_1x_double<1> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 4:
                mandel_1x_double<4> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 8:
                mandel_1x_double<8> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 16:
                mandel_1x_double<16> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            default:
                break;
            }
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu2x64) {
        if constexpr (std::is_same<T, dbldbl>::value) {
            dbldbl cx2{ cx.val.head, cx.val.tail };
            dbldbl cy2{ cy.val.head, cy.val.tail };
            dbldbl dx2{ dx.val.head, dx.val.tail };
            dbldbl dy2{ dy.val.head, dy.val.tail };

            mandel_2x_double << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu4x64) {
        // qdbl
        if constexpr (std::is_same<T, GQD::gqd_real>::value) {
            using namespace GQD;
            gqd_real cx2;
            cx2 = make_qd(cx.val.x, cx.val.y, cx.val.z, cx.val.w);

            gqd_real cy2;
            cy2 = make_qd(cy.val.x, cy.val.y, cy.val.z, cy.val.w);

            gqd_real dx2;
            dx2 = make_qd(dx.val.x, dx.val.y, dx.val.z, dx.val.w);

            gqd_real dy2;
            dy2 = make_qd(dy.val.x, dy.val.y, dy.val.z, dy.val.w);

            mandel_4x_double << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32) {
        if constexpr (std::is_same<T, float>::value) {
            // floatOnly
            switch (iteration_precision) {
            case 1:
                mandel_1x_float<1> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 4:
                mandel_1x_float<4> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 8:
                mandel_1x_float<8> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            case 16:
                mandel_1x_float<16> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    n_iterations);
                break;
            default:
                break;
            }
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32) {
        // flt
        if constexpr (std::is_same<T, dblflt>::value) {
            dblflt cx2{ cx.val.x, cx.val.y };
            dblflt cy2{ cy.val.x, cy.val.y };
            dblflt dx2{ dx.val.x, dx.val.y };
            dblflt dy2{ dy.val.x, dy.val.y };

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
    }
    else if (algorithm == RenderAlgorithm::Gpu4x32) {
        // qflt
        if constexpr (std::is_same<T, GQF::gqf_real>::value) {
            using namespace GQF;
            gqf_real cx2;
            cx2 = make_qf(cx.val.x, cx.val.y, cx.val.z, cx.val.w);

            gqf_real cy2;
            cy2 = make_qf(cy.val.x, cy.val.y, cy.val.z, cy.val.w);

            gqf_real dx2;
            dx2 = make_qf(dx.val.x, dx.val.y, dx.val.z, dx.val.w);

            gqf_real dy2;
            dy2 = make_qf(dy.val.x, dy.val.y, dy.val.z, dy.val.w);

            mandel_4x_float << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                local_width, local_height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else {
        return cudaSuccess;
    }

    return ExtractIters(buffer);
}

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<double> cx,
    MattCoords<double> cy,
    MattCoords<double> dx,
    MattCoords<double> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<float> cx,
    MattCoords<float> cy,
    MattCoords<float> dx,
    MattCoords<float> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<MattDbldbl> cx,
    MattCoords<MattDbldbl> cy,
    MattCoords<MattDbldbl> dx,
    MattCoords<MattDbldbl> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<MattQDbldbl> cx,
    MattCoords<MattQDbldbl> cy,
    MattCoords<MattQDbldbl> dx,
    MattCoords<MattQDbldbl> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<MattDblflt> cx,
    MattCoords<MattDblflt> cy,
    MattCoords<MattDblflt> dx,
    MattCoords<MattDblflt> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattCoords<MattQFltflt> cx,
    MattCoords<MattQFltflt> cy,
    MattCoords<MattQFltflt> dx,
    MattCoords<MattQFltflt> dy,
    uint32_t n_iterations,
    int iteration_precision);

template<class T>
uint32_t GPURenderer::RenderPerturb(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<T>* float_perturb,
    MattCoords<T> cx,
    MattCoords<T> cy,
    MattCoords<T> dx,
    MattCoords<T> dy,
    MattCoords<T> centerX,
    MattCoords<T> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<T> cudaResults(
        float_perturb->size,
        float_perturb->PeriodMaybeZero,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu1x32Perturbed) {
        // floatOnly
        if constexpr (std::is_same<T, float>::value) {
            mandel_1x_float_perturb<T, false> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x64Perturbed) {
        // doubleOnly
        if constexpr (std::is_same<T, double>::value) {
            mandel_1x_float_perturb<T, false> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedPeriodic) {
        // floatOnly
        if constexpr (std::is_same<T, float>::value) {
            mandel_1x_float_perturb<T, true> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    }
    else if (algorithm == RenderAlgorithm::GpuHDRx32Perturbed) {
        if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            // hdrflt
            mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

            mandel_1x_float_perturb<T, false> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    }

    return result;
}

template uint32_t GPURenderer::RenderPerturb<float>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<float>* float_perturb,
    MattCoords<float> cx,
    MattCoords<float> cy,
    MattCoords<float> dx,
    MattCoords<float> dy,
    MattCoords<float> centerX,
    MattCoords<float> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturb<HDRFloat<float>>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<float>>* float_perturb,
    MattCoords<HDRFloat<float>> cx,
    MattCoords<HDRFloat<float>> cy,
    MattCoords<HDRFloat<float>> dx,
    MattCoords<HDRFloat<float>> dy,
    MattCoords<HDRFloat<float>> centerX,
    MattCoords<HDRFloat<float>> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturb<double>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<double>* float_perturb,
    MattCoords<double> cx,
    MattCoords<double> cy,
    MattCoords<double> dx,
    MattCoords<double> dy,
    MattCoords<double> centerX,
    MattCoords<double> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturb<HDRFloat<double>>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<double>>* float_perturb,
    MattCoords<HDRFloat<double>> cx,
    MattCoords<HDRFloat<double>> cy,
    MattCoords<HDRFloat<double>> dx,
    MattCoords<HDRFloat<double>> dy,
    MattCoords<HDRFloat<double>> centerX,
    MattCoords<HDRFloat<double>> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

uint32_t GPURenderer::RenderPerturbLAv2(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<float>>* float_perturb,
    const LAReference<float> &LaReference,
    MattCoords<HDRFloat<float>> cx,
    MattCoords<HDRFloat<float>> cy,
    MattCoords<HDRFloat<float>> dx,
    MattCoords<HDRFloat<float>> dy,
    MattCoords<HDRFloat<float>> centerX,
    MattCoords<HDRFloat<float>> centerY,
    uint32_t n_iterations)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<HDRFloat<float>> cudaResults(
        float_perturb->size,
        float_perturb->PeriodMaybeZero,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    GPU_LAReference laReferenceCuda{LaReference};
    result = laReferenceCuda.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::GpuHDRx32PerturbedLAv2) {
        mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

        // hdrflt
        mandel_1xHDR_float_perturb_lav2<float> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults, laReferenceCuda,
            local_width, local_height, cx.val, cy.val, dx.val, dy.val,
            centerX.val, centerY.val,
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
    MattCoords<double> cx,
    MattCoords<double> cy,
    MattCoords<double> dx,
    MattCoords<double> dy,
    MattCoords<double> centerX,
    MattCoords<double> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64PerturbedBLA) {
        MattPerturbSingleResults<double> cudaResults(
            double_perturb->size,
            double_perturb->PeriodMaybeZero,
            double_perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        auto Run = [&]<int32_t LM2>() -> uint32_t {
            GPU_BLAS<double, BLA<double>, LM2> gpu_blas(blas->m_B);
            result = gpu_blas.CheckValid();
            if (result != 0) {
                return result;
            }

            mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

            // doubleOnly
            mandel_1x_double_perturb_bla<LM2> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults,
                gpu_blas,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            return ExtractIters(buffer);
        };

        LargeSwitch
    }
    //else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedScaled) {
    //    MattPerturbSingleResults<dblflt> cudaResults(
    //        Perturb->size,
    //        Perturb->PeriodMaybeZero,
    //        Perturb->iters);

    //    result = cudaResults.CheckValid();
    //    if (result != 0) {
    //        return result;
    //    }

    //    MattPerturbSingleResults<double> cudaResultsDouble(
    //        Perturb->size,
    //        Perturb->PeriodMaybeZero,
    //        Perturb->iters);

    //    result = cudaResultsDouble.CheckValid();
    //    if (result != 0) {
    //        return result;
    //    }

    //    // doubleOnly
    //    mandel_2x_float_perturb_setup << <nb_blocks, threads_per_block >> > (cudaResults);

    //    mandel_2x_float_perturb_scaled << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
    //        cudaResults, cudaResultsDouble,
    //        local_width, local_height, cx.val, cy.val, dx.val, dy.val,
    //        centerX.val, centerY.val,
    //        n_iterations);

    //    result = ExtractIters(buffer);
    //}

    return result;
}

uint32_t GPURenderer::RenderPerturbLAv2(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<double>>* float_perturb,
    const LAReference<double>& LaReference,
    MattCoords<HDRFloat<double>> cx,
    MattCoords<HDRFloat<double>> cy,
    MattCoords<HDRFloat<double>> dx,
    MattCoords<HDRFloat<double>> dy,
    MattCoords<HDRFloat<double>> centerX,
    MattCoords<HDRFloat<double>> centerY,
    uint32_t n_iterations)
{
    uint32_t result = cudaSuccess;

    if (iter_matrix_cu == nullptr) {
        return result;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    MattPerturbSingleResults<HDRFloat<double>> cudaResults(
        float_perturb->size,
        float_perturb->PeriodMaybeZero,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    GPU_LAReference laReferenceCuda{ LaReference };
    result = laReferenceCuda.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::GpuHDRx64PerturbedLAv2) {
        // hdrdbl
        mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

        mandel_1xHDR_float_perturb_lav2<double> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
            cudaResults, laReferenceCuda,
            local_width, local_height, cx.val, cy.val, dx.val, dy.val,
            centerX.val, centerY.val,
            n_iterations);

        result = ExtractIters(buffer);
    }

    return result;
}

template<class T>
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<T>* double_perturb,
    MattPerturbResults<float>* float_perturb,
    BLAS<T>* blas,
    MattCoords<T> cx,
    MattCoords<T> cy,
    MattCoords<T> dx,
    MattCoords<T> dy,
    MattCoords<T> centerX,
    MattCoords<T> centerY,
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
        float_perturb->PeriodMaybeZero,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    MattPerturbSingleResults<T> cudaResultsDouble(
        double_perturb->size,
        float_perturb->PeriodMaybeZero,
        double_perturb->iters);

    result = cudaResultsDouble.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaled) {
        if constexpr (std::is_same<T, double>::value) {
            // doubleOnly
            mandel_1x_float_perturb_scaled<T> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults, cudaResultsDouble,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    } else if (algorithm == RenderAlgorithm::GpuHDRx32PerturbedScaled) {
        if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            // hdrflt
            mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

            mandel_1x_float_perturb_scaled<T> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                cudaResults, cudaResultsDouble,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            result = ExtractIters(buffer);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaledBLA) {
        if constexpr (std::is_same<T, double>::value) {
            // doubleOnly
            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<double, BLA<double>, LM2> doubleGpuBlas(blas->m_B);
                result = doubleGpuBlas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

                mandel_1x_float_perturb_scaled_bla<LM2> << <nb_blocks, threads_per_block >> > (iter_matrix_cu,
                    cudaResults, cudaResultsDouble, doubleGpuBlas,
                    local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                    centerX.val, centerY.val,
                    n_iterations);

                return ExtractIters(buffer);
            };

            LargeSwitch
        }
    }

    return result;
}

template uint32_t GPURenderer::RenderPerturbBLA<double>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<double>* double_perturb,
    MattPerturbResults<float>* float_perturb,
    BLAS<double>* blas,
    MattCoords<double> cx,
    MattCoords<double> cy,
    MattCoords<double> dx,
    MattCoords<double> dy,
    MattCoords<double> centerX,
    MattCoords<double> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/
);

template uint32_t GPURenderer::RenderPerturbBLA<HDRFloat<float>>(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<float>>* double_perturb,
    MattPerturbResults<float>* float_perturb,
    BLAS<HDRFloat<float>>* blas,
    MattCoords<HDRFloat<float>> cx,
    MattCoords<HDRFloat<float>> cy,
    MattCoords<HDRFloat<float>> dx,
    MattCoords<HDRFloat<float>> dy,
    MattCoords<HDRFloat<float>> centerX,
    MattCoords<HDRFloat<float>> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/
);


uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<float2>* dblflt_perturb,
    BLAS<float2>* /*blas*/,  // TODO
    MattCoords<float2> cx,
    MattCoords<float2> cy,
    MattCoords<float2> dx,
    MattCoords<float2> dy,
    MattCoords<float2> centerX,
    MattCoords<float2> centerY,
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
        dblflt_perturb->PeriodMaybeZero,
        dblflt_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu2x32Perturbed) {
        // flt
        dblflt cx2{ cx.val.x, cx.val.y };
        dblflt cy2{ cy.val.x, cy.val.y };
        dblflt dx2{ dx.val.x, dx.val.y };
        dblflt dy2{ dy.val.x, dy.val.y };
        dblflt centerX2{ centerX.val.x, centerX.val.y };
        dblflt centerY2{ centerY.val.x, centerY.val.y };

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
    MattCoords<HDRFloat<float>> cx,
    MattCoords<HDRFloat<float>> cy,
    MattCoords<HDRFloat<float>> dx,
    MattCoords<HDRFloat<float>> dy,
    MattCoords<HDRFloat<float>> centerX,
    MattCoords<HDRFloat<float>> centerY,
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
            perturb->PeriodMaybeZero,
            perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        auto Run = [&]<int32_t LM2>() -> uint32_t {
            GPU_BLAS<HDRFloat<float>, BLA<HDRFloat<float>>, LM2> gpu_blas(blas->m_B);
            result = gpu_blas.CheckValid();
            if (result != 0) {
                return result;
            }

            mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

            // hdrflt
            mandel_1xHDR_float_perturb_bla<HDRFloat<float>, LM2> << <nb_blocks, threads_per_block >> > (
                iter_matrix_cu,
                cudaResults,
                gpu_blas,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            return ExtractIters(buffer);
        };

        LargeSwitch
    }

    return result;
}

uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* buffer,
    MattPerturbResults<HDRFloat<double>>* perturb,
    BLAS<HDRFloat<double>>* blas,
    MattCoords<HDRFloat<double>> cx,
    MattCoords<HDRFloat<double>> cy,
    MattCoords<HDRFloat<double>> dx,
    MattCoords<HDRFloat<double>> dy,
    MattCoords<HDRFloat<double>> centerX,
    MattCoords<HDRFloat<double>> centerY,
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
            perturb->PeriodMaybeZero,
            perturb->iters);

        result = cudaResults.CheckValid();
        if (result != 0) {
            return result;
        }

        auto Run = [&]<int32_t LM2>() -> uint32_t {
            GPU_BLAS<HDRFloat<double>, BLA<HDRFloat<double>>, LM2> gpu_blas(blas->m_B);
            result = gpu_blas.CheckValid();
            if (result != 0) {
                return result;
            }

            mandel_1xHDR_InitStatics << <nb_blocks, threads_per_block >> > ();

            // hdrflt -- that looks like a bug and probably should be hdrdbl
            mandel_1xHDR_float_perturb_bla<HDRFloat<double>, LM2> << <nb_blocks, threads_per_block >> > (
                iter_matrix_cu,
                cudaResults,
                gpu_blas,
                local_width, local_height, cx.val, cy.val, dx.val, dy.val,
                centerX.val, centerY.val,
                n_iterations);

            return ExtractIters(buffer);
        };

        LargeSwitch
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

const char* GPURenderer::ConvertErrorToString(uint32_t err) {
    auto typeNotExposedOutSideHere = static_cast<cudaError_t>(err);
    return cudaGetErrorString(typeNotExposedOutSideHere);
}