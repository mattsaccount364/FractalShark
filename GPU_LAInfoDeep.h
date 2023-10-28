#pragma once

#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"
#include "LAInfoDeep.h"

template<typename IterType, class SubType>
class GPU_LAInfoDeep {
public:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;

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

    CUDA_CRAP GPU_LAInfoDeep<IterType, SubType> &operator=(const GPU_LAInfoDeep<IterType, SubType>& other);
    CUDA_CRAP GPU_LAInfoDeep<IterType, SubType>& operator=(const LAInfoDeep<IterType, SubType>& other);

    CUDA_CRAP GPU_LAstep<IterType, SubType> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP GPU_LAInfoDeep<IterType, SubType>::HDRFloat getLAThresholdC() const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const;
    CUDA_CRAP const LAInfoI<IterType>& GetLAi() const;
};

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, SubType> &GPU_LAInfoDeep<IterType, SubType>::operator=(const GPU_LAInfoDeep<IterType, SubType>& other) {
    if (this == &other) {
        return *this;
    }

    this->RefRe = other.RefRe;
    this->RefIm = other.RefIm;
    this->RefExp = other.RefExp;
    this->LAThresholdMant = other.LAThresholdMant;
    this->LAThresholdExp = other.LAThresholdExp;
    
    this->ZCoeffRe = other.ZCoeffRe;
    this->ZCoeffIm = other.ZCoeffIm;
    this->ZCoeffExp = other.ZCoeffExp;
    
    this->CCoeffRe = other.CCoeffRe;
    this->CCoeffIm = other.CCoeffIm;
    this->CCoeffExp = other.CCoeffExp;

    this->LAThresholdCMant = other.LAThresholdCMant;
    this->LAThresholdCExp = other.LAThresholdCExp;
    
    this->LAi = other.LAi;
    return *this;
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, SubType>& GPU_LAInfoDeep<IterType, SubType>::operator=(const LAInfoDeep<IterType, SubType>& other) {
    this->RefRe = other.RefRe;
    this->RefIm = other.RefIm;
    this->RefExp = other.RefExp;
    this->LAThresholdMant = other.LAThresholdMant;
    this->LAThresholdExp = other.LAThresholdExp;

    this->ZCoeffRe = other.ZCoeffRe;
    this->ZCoeffIm = other.ZCoeffIm;
    this->ZCoeffExp = other.ZCoeffExp;

    this->CCoeffRe = other.CCoeffRe;
    this->CCoeffIm = other.CCoeffIm;
    this->CCoeffExp = other.CCoeffExp;

    this->LAThresholdCMant = other.LAThresholdCMant;
    this->LAThresholdCExp = other.LAThresholdCExp;

    this->LAi = other.LAi;
    return *this;
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAstep<IterType, SubType> GPU_LAInfoDeep<IterType, SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (HDRFloatComplex(RefExp + 1, RefRe, RefIm) + dz);
    newdz.Reduce();

    GPU_LAstep<IterType, SubType> temp;
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, SubType>::getRef() const {
    return HDRFloatComplex(RefExp, RefRe, RefIm);
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, SubType>::HDRFloat GPU_LAInfoDeep<IterType, SubType>::getLAThresholdC() const {
    return HDRFloat(LAThresholdCExp, LAThresholdCMant);
}

template<typename IterType, class SubType>
CUDA_CRAP
GPU_LAInfoDeep<IterType, SubType>::HDRFloatComplex GPU_LAInfoDeep<IterType, SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const {
    return newdz * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) + dc * HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<typename IterType, class SubType>
CUDA_CRAP const LAInfoI<IterType>& GPU_LAInfoDeep<IterType, SubType>::GetLAi() const {
    return this->LAi;
}
