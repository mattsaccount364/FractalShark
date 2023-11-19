#pragma once

#include "HDRFloatComplex.h"
#include "LAstep.h"
#include "LAInfoI.h"
#include "ATInfo.h"
#include "GPU_LAInfoDeep.h"

template<typename IterType, class SubType>
class GPU_LAReference {
private:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;

public:
    template<class SubType2>
    __host__ GPU_LAReference(const LAReference<IterType, SubType2>& other);
    __host__ GPU_LAReference(const GPU_LAReference& other);
    ~GPU_LAReference();

    GPU_LAReference() = delete;
    GPU_LAReference(GPU_LAReference&& other) = delete;
    GPU_LAReference& operator=(const GPU_LAReference& other) = delete;
    GPU_LAReference& operator=(GPU_LAReference&& other) = delete;

    bool UseAT;

    ATInfo<IterType, SubType> AT;

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

    GPU_LAInfoDeep<IterType, SubType> * __restrict__ LAs;
    size_t NumLAs;

    LAStageInfo<IterType> * __restrict__ LAStages;
    size_t NumLAStages;


public:
    CUDA_CRAP bool isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const;
    CUDA_CRAP IterType getLAIndex(IterType CurrentLAStage) const;
    CUDA_CRAP IterType getMacroItCount(IterType CurrentLAStage) const;
    CUDA_CRAP GPU_LAstep<IterType, SubType> getLA(
        IterType LAIndex,
        HDRFloatComplex dz,
        IterType j,
        IterType iterations,
        IterType max_iterations) const;
};

template<typename IterType, class SubType>
template<class SubType2>
__host__
GPU_LAReference<IterType, SubType>::GPU_LAReference<SubType2>(const LAReference<IterType, SubType2>& other) :
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

    GPU_LAInfoDeep<IterType, SubType>* tempLAs;
    LAStageInfo<IterType>* tempLAStages;

    m_Err = cudaMallocManaged(&tempLAs, other.LAs.size() * sizeof(GPU_LAInfoDeep<IterType, SubType>), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    LAs = tempLAs;
    NumLAs = other.LAs.size();

    m_Err = cudaMallocManaged(&tempLAStages, other.LAStages.size() * sizeof(LAStageInfo<IterType>), cudaMemAttachGlobal);
    if (m_Err != cudaSuccess) {
        return;
    }

    LAStages = tempLAStages;
    NumLAStages = other.LAStages.size();

    if constexpr (std::is_same<SubType, SubType2>::value) {
        static_assert(sizeof(GPU_LAInfoDeep<IterType, SubType>) == sizeof(LAInfoDeep<IterType, SubType>), "!");

        m_Err = cudaMemcpy(LAs,
            other.LAs.data(),
            sizeof(GPU_LAInfoDeep<IterType, SubType>) * other.LAs.size(),
            cudaMemcpyDefault);
        if (m_Err != cudaSuccess) {
            return;
        }
    }
    else {
        for (size_t i = 0; i < other.LAs.size(); i++) {
            LAs[i] = other.LAs[i];
        }
    }

    //for (size_t i = 0; i < other.LAStages.size(); i++) {
    //    LAStages[i] = other.LAStages[i];
    //}

    m_Err = cudaMemcpy(LAStages,
        other.LAStages.data(),
        sizeof(LAStageInfo<IterType>) * other.LAStages.size(),
        cudaMemcpyDefault);
    if (m_Err != cudaSuccess) {
        return;
    }

}

template<typename IterType, class SubType>
GPU_LAReference<IterType, SubType>::~GPU_LAReference() {
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

template<typename IterType, class SubType>
__host__
GPU_LAReference<IterType, SubType>::GPU_LAReference(const GPU_LAReference& other)
    : UseAT{ other.UseAT },
    AT{ other.AT },
    LAStageCount{ other.LAStageCount },
    isValid{ other.isValid },
    m_Err{ cudaSuccess },
    m_Owned{ false },
    LAs{ other.LAs },
    NumLAs{ other.NumLAs },
    LAStages{ other.LAStages },
    NumLAStages{ other.NumLAStages } {
}

template<typename IterType, class SubType>
CUDA_CRAP
bool GPU_LAReference<IterType, SubType>::isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const {
    //return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
    const auto temp1 = LAs[LAIndex];
    const auto temp2 = temp1.getLAThresholdC();
    const auto temp3 = dc.chebychevNorm();
    const auto temp4 = temp3.compareToBothPositiveReduced(temp2);
    const auto finalres = (temp4 >= 0);
    return finalres;
}

template<typename IterType, class SubType>
CUDA_CRAP
IterType GPU_LAReference<IterType, SubType>::getLAIndex(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].LAIndex;
}

template<typename IterType, class SubType>
CUDA_CRAP
IterType GPU_LAReference<IterType, SubType>::getMacroItCount(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].MacroItCount;
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType>
GPU_LAReference<IterType, SubType>::getLA(IterType LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ IterType j, IterType iterations, IterType max_iterations) const {

    const IterType LAIndexj = LAIndex + j;
    const LAInfoI<IterType> &LAIj = LAs[LAIndexj].GetLAi();

    GPU_LAstep<IterType, SubType> las;

    const IterType l = LAIj.StepLength;
    const bool usable = iterations + l <= max_iterations;

    if (usable) {
        const GPU_LAInfoDeep<IterType, SubType>& LAj = LAs[LAIndexj];

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
