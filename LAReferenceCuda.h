#pragma once

#include "HDRFloatComplex.h"
#include "LAstep.h"
#include "LAInfoI.h"
#include "ATInfo.h"
#include "LAInfoDeep.h"

class LAReferenceCuda {
private:
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    static const int lowBound = 64;
    static const double log16;

public:
    __host__ LAReferenceCuda(const LAReference& other);
    ~LAReferenceCuda();
    __host__ LAReferenceCuda(const LAReferenceCuda& other);

    LAReferenceCuda(LAReferenceCuda&& other) = delete;
    LAReferenceCuda& operator=(const LAReferenceCuda& other) = delete;
    LAReferenceCuda& operator=(LAReferenceCuda&& other) = delete;

    bool UseAT;

    ATInfo<HDRFloat> AT;

    size_t LAStageCount;

    bool isValid;

    cudaError_t m_Err;

    const bool m_Owned;

    uint32_t CheckValid() const {
        return m_Err;
    }

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;

    LAInfoDeep<float> * __restrict__ LAs;
    LAInfoI * __restrict__ LAIs;
    LAStageInfo * __restrict__ LAStages;

public:
    CUDA_CRAP bool isLAStageInvalid(size_t LAIndex, HDRFloatComplex dc) const;
    CUDA_CRAP size_t getLAIndex(size_t CurrentLAStage) const;
    CUDA_CRAP size_t getMacroItCount(size_t CurrentLAStage) const;
    CUDA_CRAP LAstep<HDRFloatComplex> getLA(size_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */size_t j, size_t iterations, size_t max_iterations) const;
};

CUDA_CRAP
bool LAReferenceCuda::isLAStageInvalid(size_t LAIndex, HDRFloatComplex dc) const {
    return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
}

CUDA_CRAP
size_t LAReferenceCuda::getLAIndex(size_t CurrentLAStage) const {
    return LAStages[CurrentLAStage].LAIndex;
}

CUDA_CRAP
size_t LAReferenceCuda::getMacroItCount(size_t CurrentLAStage) const {
    return LAStages[CurrentLAStage].MacroItCount;
}

CUDA_CRAP
LAstep<LAReferenceCuda::HDRFloatComplex>
LAReferenceCuda::getLA(size_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ size_t j, size_t iterations, size_t max_iterations) const {

    const size_t LAIndexj = LAIndex + j;
    const LAInfoI &LAIj = LAIs[LAIndexj];

    LAstep<HDRFloatComplex> las;

    const size_t l = LAIj.StepLength;
    const bool usable = iterations + l <= max_iterations;

    if (usable) {
        LAInfoDeep<float>& LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (HDRFloatComplex)LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }
    else {
        las = LAstep<HDRFloatComplex>();
        las.unusable = true;
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;

}
