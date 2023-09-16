#pragma once

#include "HDRFloatComplex.h"
#include "LAstep.h"
#include "LAInfoI.h"
#include "ATInfo.h"
#include "GPU_LAInfoDeep.h"

class GPU_LAReference {
private:
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    static const int lowBound = 64;
    static const double log16;

public:
    __host__ GPU_LAReference(const LAReference& other);
    __host__ GPU_LAReference(const GPU_LAReference& other);
    ~GPU_LAReference();

    GPU_LAReference(GPU_LAReference&& other) = delete;
    GPU_LAReference& operator=(const GPU_LAReference& other) = delete;
    GPU_LAReference& operator=(GPU_LAReference&& other) = delete;

    bool UseAT;

    ATInfo<HDRFloat> AT;

    int32_t LAStageCount;

    bool isValid;

    cudaError_t m_Err;

    const bool m_Owned;

    uint32_t CheckValid() const {
        return m_Err;
    }

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;

    GPU_LAInfoDeep<float> * __restrict__ LAs;
    LAStageInfo * __restrict__ LAStages;

public:
    CUDA_CRAP bool isLAStageInvalid(int32_t LAIndex, HDRFloatComplex dc) const;
    CUDA_CRAP int32_t getLAIndex(int32_t CurrentLAStage) const;
    CUDA_CRAP int32_t getMacroItCount(int32_t CurrentLAStage) const;
    CUDA_CRAP GPU_LAstep<HDRFloatComplex> getLA(int32_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */int32_t j, int32_t iterations, int32_t max_iterations) const;
};

CUDA_CRAP
bool GPU_LAReference::isLAStageInvalid(int32_t LAIndex, HDRFloatComplex dc) const {
    //return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
    const auto temp1 = LAs[LAIndex];
    const auto temp2 = temp1.getLAThresholdC();
    const auto temp3 = dc.chebychevNorm();
    const auto temp4 = temp3.compareToBothPositiveReduced(temp2);
    const auto finalres = (temp4 >= 0);
    return finalres;
}

CUDA_CRAP
int32_t GPU_LAReference::getLAIndex(int32_t CurrentLAStage) const {
    return LAStages[CurrentLAStage].LAIndex;
}

CUDA_CRAP
int32_t GPU_LAReference::getMacroItCount(int32_t CurrentLAStage) const {
    return LAStages[CurrentLAStage].MacroItCount;
}

CUDA_CRAP
GPU_LAstep<GPU_LAReference::HDRFloatComplex>
GPU_LAReference::getLA(int32_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ int32_t j, int32_t iterations, int32_t max_iterations) const {

    const int32_t LAIndexj = LAIndex + j;
    const LAInfoI &LAIj = LAs[LAIndexj].GetLAi();

    GPU_LAstep<HDRFloatComplex> las;

    const int32_t l = LAIj.StepLength;
    const bool usable = iterations + l <= max_iterations;

    if (usable) {
        const GPU_LAInfoDeep<float>& LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (HDRFloatComplex)LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;
}
