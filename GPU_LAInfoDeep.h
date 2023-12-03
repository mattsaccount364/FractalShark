#pragma once

#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"
#include "LAInfoDeep.h"

template<typename IterType, class Float, class SubType>
class GPU_LAInfoDeep {
public:
    using HDRFloat = Float;
    using HDRFloatComplex =
        std::conditional<
            std::is_same<Float, ::HDRFloat<float>>::value ||
            std::is_same<Float, ::HDRFloat<double>>::value ||
            std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

    HDRFloatComplex Ref;
    HDRFloatComplex ZCoeff;
    HDRFloatComplex CCoeff;
    HDRFloat LAThreshold;
    HDRFloat LAThresholdC;
    HDRFloat MinMag;
    LAInfoI<IterType> LAi;

    template<class Float2, class SubType2>
    CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType>& operator=(
        const GPU_LAInfoDeep<IterType, Float2, SubType2>& other);

    template<class Float2, class SubType2>
    CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType>& operator=(
        const LAInfoDeep<IterType, Float2, SubType2>& other);

    CUDA_CRAP GPU_LAstep<IterType, Float, SubType> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloat getLAThresholdC() const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const;
    CUDA_CRAP const LAInfoI<IterType>& GetLAi() const;
};

template<typename IterType, class Float, class SubType>
template<class Float2, class SubType2>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType> &GPU_LAInfoDeep<IterType, Float, SubType>::operator=(
    const GPU_LAInfoDeep<IterType, Float2, SubType2>& other) {
    if (this == &other) {
        return *this;
    }

    this->Ref = static_cast<HDRFloatComplex>(other.Ref);
    this->LAThreshold = static_cast<HDRFloat>(other.LAThreshold);
    this->ZCoeff = static_cast<HDRFloatComplex>(other.ZCoeff);
    this->CCoeff = static_cast<HDRFloatComplex>(other.CCoeff);
    this->LAThresholdC = static_cast<HDRFloat>(other.LAThresholdC);
    this->LAi = other.LAi;

    // Note: not copying MinMag
    // Its not needed for GPU_LAInfoDeep but included for padding purposes

    return *this;
}

template<typename IterType, class Float, class SubType>
template<class Float2, class SubType2>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>& GPU_LAInfoDeep<IterType, Float, SubType>::operator=(
    const LAInfoDeep<IterType, Float2, SubType2>& other) {

    this->Ref = static_cast<HDRFloatComplex>(other.Ref);
    this->LAThreshold = static_cast<HDRFloat>(other.LAThreshold);
    this->ZCoeff = static_cast<HDRFloatComplex>(other.ZCoeff);
    this->CCoeff = static_cast<HDRFloatComplex>(other.CCoeff);
    this->LAThresholdC = static_cast<HDRFloat>(other.LAThresholdC);
    this->LAi = other.LAi;

    return *this;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, Float, SubType> GPU_LAInfoDeep<IterType, Float, SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (Ref * HDRFloat(2) + dz);
    newdz.Reduce();

    GPU_LAstep<IterType, Float, SubType> temp;
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(LAThreshold) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, Float, SubType>::getRef() const {
    return Ref;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloat GPU_LAInfoDeep<IterType, Float, SubType>::getLAThresholdC() const {
    return LAThresholdC;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, Float, SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const {
    return newdz * ZCoeff + dc * CCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP const LAInfoI<IterType>& GPU_LAInfoDeep<IterType, Float, SubType>::GetLAi() const {
    return this->LAi;
}
