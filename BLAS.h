#pragma once

#include "BLA.h"
#include <vector>

class BLAS {
public:
    uint32_t M;
    uint32_t L;
    std::vector<std::vector<BLA>> b;

private:
    long done;
    long old_chunk;

    uint32_t LM2;//Level -1 is not attainable due to Zero R

    std::vector<uint32_t> elementsPerLevel;

    int firstLevel;
    bool returnL1;

    static constexpr uint32_t BLA_BITS = 23;
    static constexpr int BLA_STARTING_LEVEL = 2;

    using Complex = std::complex<double>;

    PerturbationResults& m_PerturbationResults;

public:
    BLAS(PerturbationResults &results) :
        m_PerturbationResults(results) {
    }

private:
    void initLStep(uint32_t level, uint32_t m, double blaSize, double epsilon) {

        b[level][m - 1] = createLStep(level, m, blaSize, epsilon);

    }

    BLA mergeTwoBlas(BLA x, BLA y, double blaSize) {
        uint32_t l = x.getL() + y.getL();
        // A = y.A * x.A
        Complex A = BLA::getNewA(x, y);
        // B = y.A * x.B + y.B
        Complex B = BLA::getNewB(x, y);
        double xA = x.hypotA();
        double xB = x.hypotB();
        if (xB != 1.0) {
            volatile Complex temp = B;
        }
        double r = min(sqrt(x.r2), max(0, (sqrt(y.r2) - xB * blaSize) / xA));
        double r2;
        if (r > 0.0000001) {
            r2 = r * r;
        }
        else {
            r2 = r * r;
        }
        
        return BLA::getGenericStep(r2, A, B, l);
    }

    BLA createLStep(uint32_t level, uint32_t m, double blaSize, double epsilon) {

        if (level == 0) {
            return createOneStep(m, epsilon);
        }

        uint32_t m2 = m << 1;
        uint32_t mx = m2 - 1;
        uint32_t my = m2;
        uint32_t levelm1 = level - 1;
        if (my <= elementsPerLevel[levelm1]) {

            BLA x = createLStep(levelm1, mx, blaSize, epsilon);

            BLA y = createLStep(levelm1, my, blaSize, epsilon);

            return mergeTwoBlas(x, y, blaSize);
        }
        else {
            return createLStep(levelm1, mx, blaSize, epsilon);
        }
    }

    BLA createOneStep(uint32_t m, double epsilon) {
        Complex A = Complex(m_PerturbationResults.x2[m], m_PerturbationResults.y2[m]);

        double mA = sqrt(std::norm(A));

        double r = mA * epsilon;

        double r2 = r * r;

        return BLA(r2, A, Complex(1, 0), 1); // BLA1Step
    }

    void initInternal(double blaSize, double epsilon) {

        uint32_t elements = elementsPerLevel[firstLevel] + 1;
        done = 0;
        for (uint32_t m = 1; m < elements; m++) {
            initLStep(firstLevel, m, blaSize, epsilon);
        }
    }

    void mergeOneStep(uint32_t m, uint32_t elementsSrc, uint32_t src, uint32_t dest, double blaSize) {
        uint32_t mx = m << 1;
        uint32_t my = mx + 1;
        if (my < elementsSrc) {
            BLA x = b[src][mx];
            BLA y = b[src][my];

            b[dest][m] = mergeTwoBlas(x, y, blaSize);
        }
        else {
            b[dest][m] = b[src][mx];
        }
    }

    void merge(double blaSize) {

        uint32_t elementsDst = 0;
        uint32_t src = firstLevel;
        uint32_t maxLevel = elementsPerLevel.size() - 1;
        for (uint32_t elementsSrc = elementsPerLevel[src]; src < maxLevel && elementsSrc > 1; src++) {

            uint32_t srcp1 = src + 1;
            elementsDst = elementsPerLevel[srcp1];
            uint32_t dst = srcp1;

            const uint32_t elementsSrcFinal = elementsSrc;
            const uint32_t srcFinal = src;
            const uint32_t destFinal = dst;

            for (uint32_t m = 0; m < elementsDst; m++) {
                mergeOneStep(m, elementsSrcFinal, srcFinal, destFinal, blaSize);
            }

            elementsSrc = elementsDst;
        }
    }

public:
    void init(uint32_t InM, double blaSize) {
        double precision = 1 / ((double)(1L << BLA_BITS));
        firstLevel = BLA_STARTING_LEVEL - 1;

        this->M = InM;

        uint32_t m = M - 1;

        if (m <= 0) {
            return;
        }

        elementsPerLevel.clear();

        uint32_t count = 1;
        for (; m > 1; m = (m + 1) >> 1) {
            elementsPerLevel.push_back(m);
            count++;
        }

        elementsPerLevel.push_back(m);

        returnL1 = firstLevel == 0;

        L = count;
        b.clear();
        b.resize(count);
        LM2 = L - 2;

        if (firstLevel >= elementsPerLevel.size()) {
            return;
        }

        for (uint32_t l = firstLevel; l < b.size(); l++) {
            b[l].clear();
            b[l].resize(elementsPerLevel[l]);
        }

        initInternal(blaSize, precision);

        merge(blaSize);
    }

    BLA *lookup(uint32_t m, double z2) {
        if (m == 0) {
            return nullptr;
        }

        BLA *B = nullptr;
        BLA *tempB;
        uint32_t ix = (m - 1) >> firstLevel;
        for (uint32_t level = firstLevel; level < L; ++level) {
            uint32_t ixm = (ix << level) + 1;
            if (m == ixm && z2 < (tempB = &b[level][ix])->r2) {
                B = tempB;
            }
            else {
                break;
            }
            ix = ix >> 1;
        }
        return B;
    }

    BLA *lookupBackwards(uint32_t m, double z2) {

        if (m == 0) {
            return nullptr;
        }

        BLA *tempB = nullptr;

        uint32_t k = m - 1;

        if ((k & 1) == 1) { // m - 1 is odd
            if (returnL1 && z2 < (tempB = &b[0][k])->r2) {
                return tempB;
            }
            return nullptr;
        }

        int zeros;
        uint32_t ix;
        if (k == 0) {
            if (z2 >= b[firstLevel][0].r2) { //k >> firstLevel, This could be done for all K values, but it was shown through statistics that most effort is done on k == 0
                return nullptr;
            }
            zeros = 32;
            ix = 0;
        }
        else if (k > 10) {
            float v = (int)((int)k & (int)-k);
            uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
            zeros = (bits >> 23) - 0x7f;
            ix = k >> zeros;
        }
        else {
            float v = (int)((int)k & (int)-k);
            uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
            zeros = (bits >> 23) - 0x7f;
            ix = k >> zeros;
        }

        int startLevel = (zeros <= LM2) ? zeros : LM2;
        for (int level = startLevel; level >= firstLevel; --level) {
            assert(level < b.size());
            assert(ix < b[level].size());
            if (z2 < (tempB = &b[level][ix])->r2) {
                return tempB;
            }
            ix = ix << 1;
        }
        return nullptr;
    }
};


