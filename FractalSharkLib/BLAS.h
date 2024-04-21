#pragma once

#include "BLA.h"
#include "PerturbationResultsHelpers.h"
#include <vector>
#include <memory>

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

template<typename IterType, class T, PerturbExtras PExtras = PerturbExtras::Disable>
class BLAS {
private:
    static constexpr size_t BLA_BITS = 23;
    static constexpr int32_t BLA_STARTING_LEVEL = 3;

public:
    size_t m_M;
    size_t m_L;
    std::vector<std::vector<BLA<T>>> m_B;
    int32_t m_LM2;//Level -1 is not attainable due to Zero R
    static constexpr int32_t m_FirstLevel = BLA_STARTING_LEVEL - 1;

    BLAS(PerturbationResults<IterType, T, PExtras> &results);

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
    BLA<T> *LookupBackwards(size_t m, T z2);

private:
    long m_OldChunk;

    std::vector<size_t> m_ElementsPerLevel;

    PerturbationResults<IterType, T, PExtras> &m_PerturbationResults;
    std::unique_ptr<RuntimeDecompressor<IterType, T, PExtras>> m_CompressionHelper;
};
