#pragma once

#include "HDRFloatComplex.h"
#include "LAstep.h"
#include "LAInfoI.h"
#include "ATInfo.h"
#include "GPU_LAInfoDeep.h"

template<typename IterType, class Float, class SubType>
class GPU_LAReference {
private:
    static constexpr bool IsHDR =
        std::is_same<Float, HDRFloat<float>>::value ||
        std::is_same<Float, HDRFloat<double>>::value ||
        std::is_same<Float, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    using HDRFloatComplex =
        std::conditional<
            std::is_same<Float, ::HDRFloat<float>>::value ||
            std::is_same<Float, ::HDRFloat<double>>::value ||
            std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

public:
    template<class T2, class SubType2, PerturbExtras OtherPExtras>
    __host__ GPU_LAReference(const LAReference<IterType, T2, SubType2, OtherPExtras>& other);
    __host__ GPU_LAReference(const GPU_LAReference& other);
    __host__ GPU_LAReference(GPU_LAReference&& other);
    __host__ GPU_LAReference();
    ~GPU_LAReference();

    
    GPU_LAReference& operator=(const GPU_LAReference& other) = delete;
    GPU_LAReference& operator=(GPU_LAReference&& other) = delete;

    bool UseAT;

    ATInfo<IterType, Float, SubType> AT;

    IterType LAStageCount;

    bool isValid;

    cudaError_t m_Err;

    bool m_Owned;

    uint32_t CheckValid() const {
        return m_Err;
    }

private:
    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;

    GPU_LAInfoDeep<IterType, Float, SubType> * __restrict__ LAs;
    size_t NumLAs;

    LAStageInfo<IterType> * __restrict__ LAStages;
    size_t NumLAStages;

    bool AllocHostLA;
    bool AllocHostLAStages;


public:
    CUDA_CRAP bool isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const;
    CUDA_CRAP IterType getLAIndex(IterType CurrentLAStage) const;
    CUDA_CRAP IterType getMacroItCount(IterType CurrentLAStage) const;
    CUDA_CRAP GPU_LAstep<IterType, Float, SubType> getLA(
        IterType LAIndex,
        HDRFloatComplex dz,
        IterType j,
        IterType iterations,
        IterType max_iterations) const;
};

template<typename IterType, class Float, class SubType>
template<class T2, class SubType2, PerturbExtras OtherPExtras>
__host__
GPU_LAReference<IterType, Float, SubType>::GPU_LAReference<T2, SubType2, OtherPExtras>(
    const LAReference<IterType, T2, SubType2, OtherPExtras>& other) :
    UseAT{ other.UseAT() },
    AT{ other.GetAT() },
    LAStageCount{ other.GetLAStageCount() },
    isValid{ other.IsValid() },
    m_Err{},
    m_Owned(true),
    LAs{},
    NumLAs{},
    LAStages{},
    NumLAStages{},
    AllocHostLA{},
    AllocHostLAStages{} {

    GPU_LAInfoDeep<IterType, Float, SubType>* tempLAs;
    LAStageInfo<IterType>* tempLAStages;

    const auto LAMemToAllocate = other.GetLAs().GetSize() * sizeof(GPU_LAInfoDeep<IterType, Float, SubType>);
    m_Err = cudaMalloc(&tempLAs, LAMemToAllocate);
    if (m_Err != cudaSuccess) {
        AllocHostLA = true;
        m_Err = cudaMallocHost(&tempLAs, LAMemToAllocate);
        if (m_Err != cudaSuccess) {
            return;
        }
    }

    LAs = tempLAs;
    NumLAs = other.GetLAs().GetSize();

    const auto LAStageMemoryToAllocate = other.GetLAStages().GetSize() * sizeof(LAStageInfo<IterType>);
    m_Err = cudaMalloc(&tempLAStages, LAStageMemoryToAllocate);
    if (m_Err != cudaSuccess) {
        AllocHostLAStages = true;
        m_Err = cudaMallocHost(&tempLAStages, LAStageMemoryToAllocate);
        if (m_Err != cudaSuccess) {
            return;
        }
    }

    LAStages = tempLAStages;
    NumLAStages = other.GetLAStages().GetSize();

    if constexpr (
        std::is_same<Float, T2>::value &&
        std::is_same<SubType, SubType2>::value) {

        check_size<Float, T2>();
        check_size<SubType, SubType2>();
        check_size<
            GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex,
            LAInfoDeep<IterType, Float, SubType, OtherPExtras>::HDRFloatComplex>();
        check_size<
            GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloat,
            LAInfoDeep<IterType, Float, SubType, OtherPExtras>::HDRFloat>();

        static_assert(
            &static_cast<GPU_LAInfoDeep<IterType, Float, SubType> *>(0)->CCoeff ==
            &static_cast<LAInfoDeep<IterType, Float, SubType, OtherPExtras> *>(0)->CCoeff, "!");

        check_size<
            GPU_LAInfoDeep<IterType, Float, SubType>,
            LAInfoDeep<IterType, Float, SubType, OtherPExtras>>();

        m_Err = cudaMemcpy(LAs,
            other.GetLAs().GetData(),
            sizeof(GPU_LAInfoDeep<IterType, Float, SubType>) * other.GetLAs().GetSize(),
            cudaMemcpyDefault);
        if (m_Err != cudaSuccess) {
            return;
        }
    }
    else {
        for (size_t i = 0; i < other.GetLAs().GetSize(); i++) {
            LAs[i] = other.GetLAs()[i];
        }
    }

    //for (size_t i = 0; i < other.m_LAStages.GetSize(); i++) {
    //    m_LAStages[i] = other.m_LAStages[i];
    //}

    m_Err = cudaMemcpy(LAStages,
        other.GetLAStages().GetData(),
        sizeof(LAStageInfo<IterType>) * other.GetLAStages().GetSize(),
        cudaMemcpyDefault);
    if (m_Err != cudaSuccess) {
        return;
    }

}

template<typename IterType, class Float, class SubType>
GPU_LAReference<IterType, Float, SubType>::~GPU_LAReference() {
    if (m_Owned) {
        if (LAs != nullptr) {
            if (AllocHostLA) {
                cudaFreeHost(LAs);
            }
            else {
                cudaFree(LAs);
            }

            LAs = nullptr;
        }

        if (LAStages != nullptr) {
            if (AllocHostLAStages) {
                cudaFreeHost(LAStages);
            }
            else {
                cudaFree(LAStages);
            }

            LAStages = nullptr;
        }
    }
}

template<typename IterType, class Float, class SubType>
__host__
GPU_LAReference<IterType, Float, SubType>::GPU_LAReference()
    : UseAT{ false },
    AT{},
    LAStageCount{},
    isValid{ false },
    m_Err{ cudaSuccess },
    m_Owned{ true },
    LAs{},
    NumLAs{},
    LAStages{},
    NumLAStages{},
    AllocHostLA{},
    AllocHostLAStages{} {}

template<typename IterType, class Float, class SubType>
__host__
GPU_LAReference<IterType, Float, SubType>::GPU_LAReference(GPU_LAReference&& other) {
    UseAT = other.UseAT;
    AT = other.AT;
    LAStageCount = other.LAStageCount;
    isValid = other.isValid;
    m_Err = other.m_Err;
    m_Owned = other.m_Owned;
    LAs = other.LAs;
    NumLAs = other.NumLAs;
    LAStages = other.LAStages;
    NumLAStages = other.NumLAStages;
    AllocHostLA = other.AllocHostLA;
    AllocHostLAStages = other.AllocHostLAStages;

    other.UseAT = false;
    other.AT = ATInfo<IterType, Float, SubType>{};
    other.LAStageCount = 0;
    other.isValid = false;
    other.m_Err = cudaSuccess;
    other.m_Owned = false;
    other.LAs = nullptr;
    other.NumLAs = 0;
    other.LAStages = nullptr;
    other.NumLAStages = 0;
    other.AllocHostLA = false;
    other.AllocHostLAStages = false;
}

template<typename IterType, class Float, class SubType>
__host__
GPU_LAReference<IterType, Float, SubType>::GPU_LAReference(const GPU_LAReference& other)
    : UseAT{ other.UseAT },
    AT{ other.AT },
    LAStageCount{ other.LAStageCount },
    isValid{ other.isValid },
    m_Err{ cudaSuccess },
    m_Owned{ false },
    LAs{ other.LAs },
    NumLAs{ other.NumLAs },
    LAStages{ other.LAStages },
    NumLAStages{ other.NumLAStages },
    AllocHostLA{ other.AllocHostLA },
    AllocHostLAStages{ other.AllocHostLAStages } {
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool GPU_LAReference<IterType, Float, SubType>::isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc) const {
    //return (dc.chebychevNorm().compareToBothPositiveReduced((m_LAs[LAIndex]).getLAThresholdC()) >= 0);
    const auto temp1 = LAs[LAIndex];
    const auto temp2 = temp1.getLAThresholdC();
    const auto temp3 = dc.chebychevNorm();

    if constexpr (IsHDR) {
        return temp3.compareToBothPositiveReduced(temp2) >= 0;
    }
    else {
        return temp3 >= temp2;
    }
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
IterType GPU_LAReference<IterType, Float, SubType>::getLAIndex(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].LAIndex;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
IterType GPU_LAReference<IterType, Float, SubType>::getMacroItCount(IterType CurrentLAStage) const {
    return LAStages[CurrentLAStage].MacroItCount;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, Float, SubType>
GPU_LAReference<IterType, Float, SubType>::getLA(IterType LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ IterType j, IterType iterations, IterType max_iterations) const {

    const IterType LAIndexj = LAIndex + j;
    const LAInfoI<IterType> &LAIj = LAs[LAIndexj].GetLAi();

    GPU_LAstep<IterType, Float, SubType> las;

    const IterType l = LAIj.StepLength;
    const bool usable = iterations + l <= max_iterations;

    if (usable) {
        const GPU_LAInfoDeep<IterType, Float, SubType>& LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;
}
