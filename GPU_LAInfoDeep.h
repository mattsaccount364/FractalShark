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

    SubType RefRe, RefIm;
    int32_t RefExp;

    SubType LAThresholdMant;
    int32_t LAThresholdExp;

    SubType ZCoeffRe, ZCoeffIm;
    int32_t ZCoeffExp;

    SubType CCoeffRe, CCoeffIm;
    int32_t CCoeffExp;

    SubType LAThresholdCMant;
    int32_t LAThresholdCExp;

    LAInfoI<IterType> LAi;

    SubType MinMagMant;
    int32_t MinMagExp;

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

    this->RefRe = (SubType)other.RefRe;
    this->RefIm = (SubType)other.RefIm;
    this->RefExp = other.RefExp;
    this->LAThresholdMant = (SubType)other.LAThresholdMant;
    this->LAThresholdExp = other.LAThresholdExp;
    
    this->ZCoeffRe = (SubType)other.ZCoeffRe;
    this->ZCoeffIm = (SubType)other.ZCoeffIm;
    this->ZCoeffExp = other.ZCoeffExp;
    
    this->CCoeffRe = (SubType)other.CCoeffRe;
    this->CCoeffIm = (SubType)other.CCoeffIm;
    this->CCoeffExp = other.CCoeffExp;

    this->LAThresholdCMant = (SubType)other.LAThresholdCMant;
    this->LAThresholdCExp = other.LAThresholdCExp;
    
    this->LAi = other.LAi;
    return *this;
}

template<typename IterType, class Float, class SubType>
template<class Float2, class SubType2>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>& GPU_LAInfoDeep<IterType, Float, SubType>::operator=(
    const LAInfoDeep<IterType, Float2, SubType2>& other) {
    this->RefRe = (SubType)other.RefRe;
    this->RefIm = (SubType)other.RefIm;
    this->RefExp = other.RefExp;
    this->LAThresholdMant = (SubType)other.LAThresholdMant;
    this->LAThresholdExp = other.LAThresholdExp;

    this->ZCoeffRe = (SubType)other.ZCoeffRe;
    this->ZCoeffIm = (SubType)other.ZCoeffIm;
    this->ZCoeffExp = other.ZCoeffExp;

    this->CCoeffRe = (SubType)other.CCoeffRe;
    this->CCoeffIm = (SubType)other.CCoeffIm;
    this->CCoeffExp = other.CCoeffExp;

    this->LAThresholdCMant = (SubType)other.LAThresholdCMant;
    this->LAThresholdCExp = other.LAThresholdCExp;

    this->LAi = other.LAi;
    return *this;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, Float, SubType> GPU_LAInfoDeep<IterType, Float, SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (HDRFloatComplex(RefExp + 1, RefRe, RefIm) + dz);
    newdz.Reduce();

    GPU_LAstep<IterType, Float, SubType> temp;
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, Float, SubType>::getRef() const {
    return HDRFloatComplex(RefExp, RefRe, RefIm);
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloat GPU_LAInfoDeep<IterType, Float, SubType>::getLAThresholdC() const {
    return HDRFloat(LAThresholdCExp, LAThresholdCMant);
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, Float, SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const {
    return newdz * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) + dc * HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP const LAInfoI<IterType>& GPU_LAInfoDeep<IterType, Float, SubType>::GetLAi() const {
    return this->LAi;
}
