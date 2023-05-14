#include "stdafx.h"
#include "PerturbationResults.h"
#include "BLAS.h"

template<class T>
BLAS<T>::BLAS(PerturbationResults& results) :
    m_PerturbationResults(results) {
}

template<class T>
void BLAS<T>::InitLStep(size_t level, size_t m, T blaSize, T epsilon) {
    m_B[level][m - 1] = CreateLStep(level, m, blaSize, epsilon);
}

template<class T>
BLA<T> BLAS<T>::MergeTwoBlas(BLA<T> x, BLA<T> y, T blaSize) {
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

    T r = min(sqrt(x.getR2()), max(0, (sqrt(y.getR2()) - xB * blaSize) / xA));
    T r2 = r * r;

    return BLA<T>::getGenericStep(r2, RealA, ImagA, RealB, ImagB, l);
}

template<class T>
BLA<T> BLAS<T>::CreateLStep(size_t level, size_t m, T blaSize, T epsilon) {

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

template<class T>
BLA<T> BLAS<T>::CreateOneStep(size_t m, T epsilon) {
    T RealA = static_cast<T>(m_PerturbationResults.x2[m]);
    T ImagA = static_cast<T>(m_PerturbationResults.y2[m]);

    T mA = sqrt(RealA * RealA + ImagA * ImagA);

    T r = mA * epsilon;

    T r2 = r * r;

    const T RealB = 1;
    const T ImagB = 0;
    const int l = 1;
    return BLA<T>(r2, RealA, ImagA, RealB, ImagB, l); // BLA1Step
}

template<class T>
void BLAS<T>::InitInternal(T blaSize, T epsilon) {

    size_t elements = m_ElementsPerLevel[m_FirstLevel] + 1;
    m_Done = 0;
    for (size_t m = 1; m < elements; m++) {
        InitLStep(m_FirstLevel, m, blaSize, epsilon);
    }
}

template<class T>
void BLAS<T>::MergeOneStep(size_t m, size_t elementsSrc, size_t src, size_t dest, T blaSize) {
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

template<class T>
void BLAS<T>::Merge(T blaSize) {

    size_t elementsDst = 0;
    size_t src = m_FirstLevel;
    size_t maxLevel = m_ElementsPerLevel.size() - 1;
    for (size_t elementsSrc = m_ElementsPerLevel[src]; src < maxLevel && elementsSrc > 1; src++) {

        size_t srcp1 = src + 1;
        elementsDst = m_ElementsPerLevel[srcp1];
        size_t dst = srcp1;

        const size_t elementsSrcFinal = elementsSrc;
        const size_t srcFinal = src;
        const size_t destFinal = dst;

        for (size_t m = 0; m < elementsDst; m++) {
            MergeOneStep(m, elementsSrcFinal, srcFinal, destFinal, blaSize);
        }

        elementsSrc = elementsDst;
    }
}

template<class T>
void BLAS<T>::Init(size_t InM, T blaSize) {
    T precision = 1 / ((T)(1L << BLA_BITS));
    m_FirstLevel = BLA_STARTING_LEVEL - 1;

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

template<class T>
BLA<T>* BLAS<T>::LookupBackwards(size_t m, T z2) {

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
        assert(level < m_B.size());
        assert(ix < m_B[level].size());
        if (z2 < (tempB = &m_B[level][ix])->getR2()) {
            return tempB;
        }
        ix = ix << 1;
    }
    return nullptr;
}

template class BLAS<float>;
template class BLAS<double>;