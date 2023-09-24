#pragma once

#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"
#include "LAInfoDeep.h"

template<class SubType>
class GPU_LAInfoDeep {
public:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;
    using T = SubType;

    T RefRe, RefIm;
    int32_t RefExp;

    T LAThresholdMant;
    int32_t LAThresholdExp;

    T ZCoeffRe, ZCoeffIm;
    int32_t ZCoeffExp;

    T CCoeffRe, CCoeffIm;
    int32_t CCoeffExp;

    T LAThresholdCMant;
    int32_t LAThresholdCExp;

    LAInfoI LAi;

    T MinMagMant;
    int32_t MinMagExp;

    CUDA_CRAP GPU_LAInfoDeep<SubType> &operator=(const GPU_LAInfoDeep<SubType>& other);
    CUDA_CRAP GPU_LAInfoDeep<SubType>& operator=(const LAInfoDeep<SubType>& other);

    CUDA_CRAP GPU_LAstep<HDRFloatComplex> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP GPU_LAInfoDeep<SubType>::HDRFloat getLAThresholdC() const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const;
    CUDA_CRAP const LAInfoI& GetLAi() const;
};

template<class SubType>
CUDA_CRAP
GPU_LAInfoDeep<SubType> &GPU_LAInfoDeep<SubType>::operator=(const GPU_LAInfoDeep<SubType>& other) {
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

template<class SubType>
CUDA_CRAP
GPU_LAInfoDeep<SubType>& GPU_LAInfoDeep<SubType>::operator=(const LAInfoDeep<SubType>& other) {
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

template<class SubType>
CUDA_CRAP
GPU_LAstep<HDRFloatComplex<SubType>> GPU_LAInfoDeep<SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (HDRFloatComplex(RefExp + 1, RefRe, RefIm) + dz);
    newdz.Reduce();

    GPU_LAstep<HDRFloatComplex> temp;
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<class SubType>
CUDA_CRAP
GPU_LAInfoDeep<SubType>::HDRFloatComplex GPU_LAInfoDeep<SubType>::getRef() const {
    return HDRFloatComplex(RefExp, RefRe, RefIm);
}

template<class SubType>
CUDA_CRAP
GPU_LAInfoDeep<SubType>::HDRFloat GPU_LAInfoDeep<SubType>::getLAThresholdC() const {
    return HDRFloat(LAThresholdCExp, LAThresholdCMant);
}

template<class SubType>
CUDA_CRAP
GPU_LAInfoDeep<SubType>::HDRFloatComplex GPU_LAInfoDeep<SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const {
    return newdz * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) + dc * HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<class SubType>
CUDA_CRAP const LAInfoI& GPU_LAInfoDeep<SubType>::GetLAi() const {
    return this->LAi;
}
