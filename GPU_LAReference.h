#pragma once

#include "HDRFloatComplex.h"
#include "LAstep.h"
#include "LAInfoI.h"
#include "ATInfo.h"
#include "GPU_LAInfoDeep.h"

template<class SubType>
class GPU_LAReference {
private:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;

public:
    __host__ GPU_LAReference(const LAReference<SubType>& other);
    __host__ GPU_LAReference(const GPU_LAReference& other);
    ~GPU_LAReference();

    GPU_LAReference(GPU_LAReference&& other) = delete;
    GPU_LAReference& operator=(const GPU_LAReference& other) = delete;
    GPU_LAReference& operator=(GPU_LAReference&& other) = delete;

    bool UseAT;

    ATInfo<SubType> AT;

    IterType LAStageCount;

    bool isValid;

    cudaError_t m_Err;

    const bool m_Owned;

    uint32_t CheckValid() const {
        return m_Err;
    }

private:
    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;

    GPU_LAInfoDeep<SubType> * __restrict__ LAs;
    size_t NumLAs;

    LAStageInfo * __restrict__ LAStages;
    size_t NumLAStages;


public:
    CUDA_CRAP bool isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const;
    CUDA_CRAP IterType getLAIndex(IterType CurrentLAStage) const;
    CUDA_CRAP IterType getMacroItCount(IterType CurrentLAStage) const;
    CUDA_CRAP GPU_LAstep<SubType> getLA(
        IterType LAIndex,
        HDRFloatComplex dz,
        IterType j,
        IterType iterations,
        IterType max_iterations) const;
};

template<class SubType>
__host__
GPU_LAReference<SubType>::GPU_LAReference(const LAReference<SubType>& other) :
    UseAT{ other.UseAT },
    AT{ other.AT },
    LAStageCount{ other.LAStageCount },
    isValid{ other.isValid },
    m_Err{},
    m_Owned(true),
    LAs{},
    NumLAs{},
    LAStages{},
    NumLAStages{} {

    GPU_LAInfoDeep<SubType>* tempLAs;
    LAStageInfo* tempLAStages;

    m_Err = cudaMallocManaged(&tempLAs, other.LAs.size() * sizeof(GPU_LAInfoDeep<SubType>), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    LAs = tempLAs;
    NumLAs = other.LAs.size();

    m_Err = cudaMallocManaged(&tempLAStages, other.LAStages.size() * sizeof(LAStageInfo), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    LAStages = tempLAStages;
    NumLAStages = other.LAStages.size();

    //for (size_t i = 0; i < other.LAs.size(); i++) {
    //    LAs[i] = other.LAs[i];
    //}

    static_assert(sizeof(GPU_LAInfoDeep<SubType>) == sizeof(LAInfoDeep<SubType>), "!");
    
    m_Err = cudaMemcpy(LAs,
        other.LAs.data(),
        sizeof(GPU_LAInfoDeep<SubType>) * other.LAs.size(),
        cudaMemcpyDefault);
    if (m_Err != cudaSuccess) {
        return;
    }

    //for (size_t i = 0; i < other.LAStages.size(); i++) {
    //    LAStages[i] = other.LAStages[i];
    //}

    m_Err = cudaMemcpy(LAStages,
        other.LAStages.data(),
        sizeof(LAStageInfo) * other.LAStages.size(),
        cudaMemcpyDefault);
    if (m_Err != cudaSuccess) {
        return;
    }

}

template<class SubType>
GPU_LAReference<SubType>::~GPU_LAReference() {
    if (m_Owned) {
        if (LAs != nullptr) {
            cudaFree(LAs);
            LAs = nullptr;
        }

        if (LAStages != nullptr) {
            cudaFree(LAStages);
            LAStages = nullptr;
        }
    }
}

template<class SubType>
__host__
GPU_LAReference<SubType>::GPU_LAReference(const GPU_LAReference& other) : m_Owned(false) {
    this->UseAT = other.UseAT;
    this->AT = other.AT;
    this->LAStageCount = other.LAStageCount;
    this->isValid = other.isValid;
    this->m_Err = cudaSuccess;
    this->LAs = other.LAs;
    this->LAStages = other.LAStages;
}

template<class SubType>
CUDA_CRAP
bool GPU_LAReference<SubType>::isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const {
    //return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
    const auto temp1 = LAs[LAIndex];
    const auto temp2 = temp1.getLAThresholdC();
    const auto temp3 = dc.chebychevNorm();
    const auto temp4 = temp3.compareToBothPositiveReduced(temp2);
    const auto finalres = (temp4 >= 0);
    return finalres;
}

template<class SubType>
CUDA_CRAP
IterType GPU_LAReference<SubType>::getLAIndex(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].LAIndex;
}

template<class SubType>
CUDA_CRAP
IterType GPU_LAReference<SubType>::getMacroItCount(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].MacroItCount;
}

template<class SubType>
CUDA_CRAP
GPU_LAstep<SubType>
GPU_LAReference<SubType>::getLA(IterType LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ IterType j, IterType iterations, IterType max_iterations) const {

    const IterType LAIndexj = LAIndex + j;
    const LAInfoI &LAIj = LAs[LAIndexj].GetLAi();

    GPU_LAstep<SubType> las;

    const IterType l = LAIj.StepLength;
    const bool usable = iterations + l <= max_iterations;

    if (usable) {
        const GPU_LAInfoDeep<SubType>& LAj = LAs[LAIndexj];

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
