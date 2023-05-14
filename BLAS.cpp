#include "stdafx.h"
#include "PerturbationResults.h"
#include "BLAS.h"

BLAS::BLAS(PerturbationResults& results) :
    m_PerturbationResults(results) {
}

void BLAS::initLStep(size_t level, size_t m, double blaSize, double epsilon) {
    m_B[level][m - 1] = createLStep(level, m, blaSize, epsilon);
}

BLA BLAS::mergeTwoBlas(BLA x, BLA y, double blaSize) {
    uint32_t l = x.getL() + y.getL();
    // A = y.A * x.A
    Complex A = BLA::getNewA(x, y);
    // B = y.A * x.B + y.B
    Complex B = BLA::getNewB(x, y);
    double xA = x.hypotA();
    double xB = x.hypotB();

    // TODO

    double r = min(sqrt(x.getR2()), max(0, (sqrt(y.getR2()) - xB * blaSize) / xA));
    double r2 = r * r;

    return BLA::getGenericStep(r2, A, B, l);
}

BLA BLAS::createLStep(size_t level, size_t m, double blaSize, double epsilon) {

    if (level == 0) {
        return createOneStep(m, epsilon);
    }

    size_t m2 = m << 1;
    size_t mx = m2 - 1;
    size_t my = m2;
    size_t levelm1 = level - 1;
    if (my <= m_ElementsPerLevel[levelm1]) {

        BLA x = createLStep(levelm1, mx, blaSize, epsilon);

        BLA y = createLStep(levelm1, my, blaSize, epsilon);

        return mergeTwoBlas(x, y, blaSize);
    }
    else {
        return createLStep(levelm1, mx, blaSize, epsilon);
    }
}

BLA BLAS::createOneStep(size_t m, double epsilon) {
    Complex A = Complex(m_PerturbationResults.x2[m], m_PerturbationResults.y2[m]);

    double mA = sqrt(A.Real * A.Real + A.Imag * A.Imag);

    double r = mA * epsilon;

    double r2 = r * r;

    return BLA(r2, A, Complex(1, 0), 1); // BLA1Step
}

void BLAS::initInternal(double blaSize, double epsilon) {

    size_t elements = m_ElementsPerLevel[m_FirstLevel] + 1;
    m_Done = 0;
    for (size_t m = 1; m < elements; m++) {
        initLStep(m_FirstLevel, m, blaSize, epsilon);
    }
}

void BLAS::mergeOneStep(size_t m, size_t elementsSrc, size_t src, size_t dest, double blaSize) {
    size_t mx = m << 1;
    size_t my = mx + 1;
    if (my < elementsSrc) {
        BLA x = m_B[src][mx];
        BLA y = m_B[src][my];

        m_B[dest][m] = mergeTwoBlas(x, y, blaSize);
    }
    else {
        m_B[dest][m] = m_B[src][mx];
    }
}

void BLAS::merge(double blaSize) {

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
            mergeOneStep(m, elementsSrcFinal, srcFinal, destFinal, blaSize);
        }

        elementsSrc = elementsDst;
    }
}

void BLAS::init(size_t InM, double blaSize) {
    double precision = 1 / ((double)(1L << BLA_BITS));
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

    initInternal(blaSize, precision);

    merge(blaSize);
}

BLA* BLAS::lookupBackwards(size_t m, double z2) {

    if (m == 0) {
        return nullptr;
    }

    BLA* tempB = nullptr;

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