////////////////////////////////////////////////////////////////////////////////////////
// Bilinear approximation
////////////////////////////////////////////////////////////////////////////////////////

template<class T>
BLA<T>::BLA() : r2(0), Ax(0), Ay(0), Bx(0), By(0), l(0) {
}

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
CUDA_CRAP void BLA<T>::getNewB(const BLA &x, const BLA &y, T &RealValue, T &ImagValue) {
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

template<typename IterType, class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<IterType, T, GPUBLA_TYPE, LM2>::GPU_BLAS(const std::vector<std::vector<GPUBLA_TYPE>> &B)
    : m_B(nullptr),
    m_Err(),
    m_Owned(true) {

    GPUBLA_TYPE **tempB;
    m_Err = cudaMallocManaged(&tempB, m_NumLevels * sizeof(GPUBLA_TYPE *), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    m_B = tempB;
    cudaMemset(m_B, 0, m_NumLevels * sizeof(GPUBLA_TYPE *));

    size_t total = 0;

    for (size_t i = 0; i < B.size(); i++) {
        total += sizeof(GPUBLA_TYPE) * B[i].size();
    }

    m_Err = cudaMallocManaged(&m_BMem, total);
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

template<typename IterType, class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<IterType, T, GPUBLA_TYPE, LM2>::~GPU_BLAS() {
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

template<typename IterType, class T, class GPUBLA_TYPE, int32_t LM2>
GPU_BLAS<IterType, T, GPUBLA_TYPE, LM2>::GPU_BLAS(const GPU_BLAS &other) : m_Owned(false) {
    if (this == &other) {
        return;
    }

    m_BMem = other.m_BMem;
    m_B = other.m_B;
    //for (size_t i = 0; i < m_NumLevels; i++) {
    //    m_B[i] = other.m_B[i];
    //}
}

template<typename IterType, class T, class GPUBLA_TYPE, int32_t LM2>
uint32_t GPU_BLAS<IterType, T, GPUBLA_TYPE, LM2>::CheckValid() const {
    return m_Err;
}

#ifdef __CUDA_ARCH__
template<typename IterType, class T, class GPUBLA_TYPE, int32_t LM2>
CUDA_CRAP const GPUBLA_TYPE *GPU_BLAS<IterType, T, GPUBLA_TYPE, LM2>::LookupBackwards(
    const GPUBLA_TYPE *__restrict__ *altB,
    //const GPUBLA_TYPE* __restrict__ nullBla,
    /*T* curBR2,*/
    IterType m,
    T z2) const {

    const IterType k = m - 1;

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
    const GPUBLA_TYPE *__restrict__ tempB = nullptr;
    const IterType zeros = __clz(__brev(k));
    IterType ix = k >> zeros;

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
