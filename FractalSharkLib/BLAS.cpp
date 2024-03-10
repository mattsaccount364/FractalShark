#include "stdafx.h"

#include "Fractal.h"
#include "PerturbationResults.h"
#include "BLAS.h"
#include "HDRFloat.h"

template<typename IterType, class T, PerturbExtras PExtras>
BLAS<IterType, T, PExtras>::BLAS(PerturbationResults<IterType, T, PExtras>& results) :
    m_PerturbationResults{ results },
    m_CompressionHelper{ std::make_unique<CompressionHelper<IterType, T, PExtras>>(results) },
    m_OldChunk{},
    m_M{},
    m_L{},
    m_B{},
    m_LM2{},
    m_ElementsPerLevel{}
{
}

template<typename IterType, class T, PerturbExtras PExtras>
void BLAS<IterType, T, PExtras>::InitLStep(size_t level, size_t m, T blaSize, T epsilon) {
    m_B[level][m - 1] = CreateLStep(level, m, blaSize, epsilon);
}

template<typename IterType, class T, PerturbExtras PExtras>
BLA<T> BLAS<IterType, T, PExtras>::MergeTwoBlas(BLA<T> x, BLA<T> y, T blaSize) {
    uint32_t l = x.getL() + y.getL();
    // A = y.A * x.A
    T RealA, ImagA;
    BLA<T>::getNewA(x, y, RealA, ImagA);

    // B = y.A * x.B + y.B
    T RealB, ImagB;
    BLA<T>::getNewB(x, y, RealB, ImagB);
    T xA = x.hypotA();
    T xB = x.hypotB();

    // TODO
    T tempR = (HdrSqrt(y.getR2()) - xB * blaSize) / xA;
    HdrReduce(tempR);

    T r = HdrMinPositiveReduced(HdrSqrt(x.getR2()), HdrMaxPositiveReduced(T(0), tempR));
    T r2 = r * r;
    return BLA<T>::getGenericStep(r2, RealA, ImagA, RealB, ImagB, l);
}

template<typename IterType, class T, PerturbExtras PExtras>
BLA<T> BLAS<IterType, T, PExtras>::CreateLStep(size_t level, size_t m, T blaSize, T epsilon) {

    if (level == 0) {
        return CreateOneStep(m, epsilon);
    }

    size_t m2 = m << 1;
    size_t mx = m2 - 1;
    size_t my = m2;
    size_t levelm1 = level - 1;
    if (my <= m_ElementsPerLevel[levelm1]) {

        BLA<T> x = CreateLStep(levelm1, mx, blaSize, epsilon);

        BLA<T> y = CreateLStep(levelm1, my, blaSize, epsilon);

        return MergeTwoBlas(x, y, blaSize);
    }
    else {
        return CreateLStep(levelm1, mx, blaSize, epsilon);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
BLA<T> BLAS<IterType, T, PExtras>::CreateOneStep(size_t m, T epsilon) {
    const auto Complex = m_PerturbationResults.GetComplex(*m_CompressionHelper, m);
    T RealA = static_cast<T>(Complex.getRe() * 2);
    T ImagA = static_cast<T>(Complex.getIm() * 2);

    T mA = HdrSqrt(RealA * RealA + ImagA * ImagA);

    T r = mA * epsilon;

    T r2 = r * r;

    const T RealB = T(1);
    const T ImagB = T(0);
    const int l = 1;
    return BLA<T>(r2, RealA, ImagA, RealB, ImagB, l); // BLA1Step
}

constexpr size_t WorkThreshholdForThreads = 5000;

template<typename IterType, class T, PerturbExtras PExtras>
void BLAS<IterType, T, PExtras>::InitInternal(T blaSize, T epsilon) {

    std::vector<std::unique_ptr<std::thread>> threads;

    auto RunInit = [&](size_t firstLevel, size_t mStart, size_t mEnd, T blaSize, T epsilon) {
        for (size_t m = mStart; m < mEnd; m++) {
            InitLStep(firstLevel, m, blaSize, epsilon);
        }
    };

    size_t elements = m_ElementsPerLevel[m_FirstLevel] + 1;
    size_t optThreads = elements / WorkThreshholdForThreads;
    if (optThreads > std::thread::hardware_concurrency()) {
        optThreads = std::thread::hardware_concurrency();
    }
    else if (optThreads == 0) {
        optThreads = 1;
    }

    size_t mDelta = elements / optThreads;
    size_t mConsumed = 1;

    if (mDelta > WorkThreshholdForThreads && optThreads > 1) {
        for (size_t i = 0; i < optThreads - 1; i++) {
            threads.push_back(std::make_unique<std::thread>(RunInit, m_FirstLevel, mConsumed, mConsumed + mDelta, blaSize, epsilon));
            mConsumed += mDelta;
        }

        threads.push_back(std::make_unique<std::thread>(RunInit, m_FirstLevel, mConsumed, elements, blaSize, epsilon));

        for (size_t i = 0; i < threads.size(); i++) {
            threads[i]->join();
        }
    }
    else {
        RunInit(m_FirstLevel, 1, elements, blaSize, epsilon);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void BLAS<IterType, T, PExtras>::MergeOneStep(size_t m, size_t elementsSrc, size_t src, size_t dest, T blaSize) {
    size_t mx = m << 1;
    size_t my = mx + 1;
    if (my < elementsSrc) {
        BLA<T> x = m_B[src][mx];
        BLA<T> y = m_B[src][my];

        m_B[dest][m] = MergeTwoBlas(x, y, blaSize);
    }
    else {
        m_B[dest][m] = m_B[src][mx];
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void BLAS<IterType, T, PExtras>::Merge(T blaSize) {

    size_t elementsDst = 0;
    size_t src = m_FirstLevel;
    size_t maxLevel = m_ElementsPerLevel.size() - 1;
    for (size_t elementsSrc = m_ElementsPerLevel[src]; src < maxLevel && elementsSrc > 1; src++) {
        std::vector<std::unique_ptr<std::thread>> threads;

        size_t srcp1 = src + 1;
        elementsDst = m_ElementsPerLevel[srcp1];
        size_t dst = srcp1;

        size_t optThreads = elementsDst / WorkThreshholdForThreads;
        if (optThreads > std::thread::hardware_concurrency()) {
            optThreads = std::thread::hardware_concurrency();
        }
        else if (optThreads == 0) {
            optThreads = 1;
        }

        const size_t elementsSrcFinal = elementsSrc;
        const size_t srcFinal = src;
        const size_t destFinal = dst;

        auto SubMerge = [&](size_t mStart, size_t mEnd) {
            for (size_t m = mStart; m < mEnd; m++) {
                MergeOneStep(m, elementsSrcFinal, srcFinal, destFinal, blaSize);
            }
        };

        size_t mDelta = elementsDst / optThreads;
        size_t mConsumed = 0;

        if (mDelta > WorkThreshholdForThreads && optThreads > 1) {
            for (size_t i = 0; i < optThreads - 1; i++) {
                threads.push_back(std::make_unique<std::thread>(SubMerge, mConsumed, mConsumed + mDelta));
                mConsumed += mDelta;
            }

            threads.push_back(std::make_unique<std::thread>(SubMerge, mConsumed, elementsDst));

            for (size_t i = 0; i < threads.size(); i++) {
                threads[i]->join();
            }
        }
        else {
            SubMerge(0, elementsDst);
        }

        elementsSrc = elementsDst;
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void BLAS<IterType, T, PExtras>::Init(size_t InM, T blaSize) {
    T precision = T(1) / T{ 1L << BLA_BITS };

    this->m_M = InM;

    size_t m = m_M - 1;

    if (m <= 0) {
        return;
    }

    m_ElementsPerLevel.clear();

    for (; m > 1; m = (m + 1) >> 1) {
        m_ElementsPerLevel.push_back(m);
    }

    m_ElementsPerLevel.push_back(m);

    m_L = m_ElementsPerLevel.size();
    m_B.clear();
    m_B.resize(m_L);
    m_LM2 = (int32_t)m_L - 2;
    if (m_LM2 < 0) {
        m_LM2 = 0;
    }

    if (m_FirstLevel >= m_ElementsPerLevel.size()) {
        return;
    }

    for (size_t l = m_FirstLevel; l < m_B.size(); l++) {
        m_B[l].clear();
        m_B[l].resize(m_ElementsPerLevel[l]);
    }

    InitInternal(blaSize, precision);

    Merge(blaSize);
}

template<typename IterType, class T, PerturbExtras PExtras>
BLA<T>* BLAS<IterType, T, PExtras>::LookupBackwards(size_t m, T z2) {

    if (m == 0) {
        return nullptr;
    }

    BLA<T>* tempB = nullptr;

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
        if constexpr (std::is_same<T, HDRFloat<float>>::value || std::is_same<T, HDRFloat<double>>::value) {
            if (z2.compareToBothPositiveReduced(m_B[m_FirstLevel][0].getR2()) >= 0) {
                return nullptr;
            }
        }
        else {
            if (z2 >= m_B[m_FirstLevel][0].getR2()) {
                return nullptr;
            }
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
        assert(level < m_B.size());
        assert(ix < m_B[level].size());
        if (HdrCompareToBothPositiveReducedLT(z2, (tempB = &m_B[level][ix])->getR2())) {
            return tempB;
        }
        ix = ix << 1;
    }
    return nullptr;
}

template class BLAS<uint32_t, float, PerturbExtras::Disable>;
template class BLAS<uint32_t, double, PerturbExtras::Disable>;
template class BLAS<uint32_t, HDRFloat<double>, PerturbExtras::Disable>;
template class BLAS<uint32_t, HDRFloat<float>, PerturbExtras::Disable>;

template class BLAS<uint32_t, float, PerturbExtras::Bad>;
template class BLAS<uint32_t, double, PerturbExtras::Bad>;
template class BLAS<uint32_t, HDRFloat<double>, PerturbExtras::Bad>;
template class BLAS<uint32_t, HDRFloat<float>, PerturbExtras::Bad>;

template class BLAS<uint64_t, float, PerturbExtras::Disable>;
template class BLAS<uint64_t, double, PerturbExtras::Disable>;
template class BLAS<uint64_t, HDRFloat<double>, PerturbExtras::Disable>;
template class BLAS<uint64_t, HDRFloat<float>, PerturbExtras::Disable>;

template class BLAS<uint64_t, float, PerturbExtras::Bad>;
template class BLAS<uint64_t, double, PerturbExtras::Bad>;
template class BLAS<uint64_t, HDRFloat<double>, PerturbExtras::Bad>;
template class BLAS<uint64_t, HDRFloat<float>, PerturbExtras::Bad>;