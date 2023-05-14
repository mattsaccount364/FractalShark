#pragma once

#include "BLA.h"
#include <vector>

class PerturbationResults;

class BLAS {
public:
    size_t m_M;
    size_t m_L;
    std::vector<std::vector<BLA>> m_B;
    int32_t m_LM2;//Level -1 is not attainable due to Zero R
    size_t m_FirstLevel;

    BLAS(PerturbationResults& results);

private:
    void initLStep(size_t level, size_t m, double blaSize, double epsilon);
    BLA mergeTwoBlas(BLA x, BLA y, double blaSize);
    BLA createLStep(size_t level, size_t m, double blaSize, double epsilon);
    BLA createOneStep(size_t m, double epsilon);
    void initInternal(double blaSize, double epsilon);
    void mergeOneStep(size_t m, size_t elementsSrc, size_t src, size_t dest, double blaSize);
    void merge(double blaSize);

public:
    void init(size_t InM, double blaSize);
    BLA* lookupBackwards(size_t m, double z2);

private:
    long m_Done;
    long m_OldChunk;

    std::vector<size_t> m_ElementsPerLevel;

    static constexpr size_t BLA_BITS = 23;
    static constexpr size_t BLA_STARTING_LEVEL = 2;

    PerturbationResults& m_PerturbationResults;
};


