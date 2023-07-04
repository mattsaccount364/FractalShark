#pragma once

#include "BLA.h"
#include <vector>

template<class T>
class PerturbationResults;

template<class T>
class BLAS {
public:
    size_t m_M;
    size_t m_L;
    std::vector<std::vector<BLA<T>>> m_B;
    int32_t m_LM2;//Level -1 is not attainable due to Zero R
    size_t m_FirstLevel;

    BLAS(PerturbationResults<T>& results);

private:
    void InitLStep(size_t level, size_t m, T blaSize, T epsilon);
    BLA<T> MergeTwoBlas(BLA<T> x, BLA<T> y, T blaSize);
    BLA<T> CreateLStep(size_t level, size_t m, T blaSize, T epsilon);
    BLA<T> CreateOneStep(size_t m, T epsilon);
    void InitInternal(T blaSize, T epsilon);
    void MergeOneStep(size_t m, size_t elementsSrc, size_t src, size_t dest, T blaSize);
    void Merge(T blaSize);

public:
    void Init(size_t InM, T blaSize);
    BLA<T>* LookupBackwards(size_t m, T z2);

private:
    long m_OldChunk;

    std::vector<size_t> m_ElementsPerLevel;

    static constexpr size_t BLA_BITS = 23;
    static constexpr size_t BLA_STARTING_LEVEL = 2;

    PerturbationResults<T>& m_PerturbationResults;
};


