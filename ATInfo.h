#pragma once

#include <complex>

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "ATResult.h"

template<class SubType>
class ATInfo {
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;


public:
    CUDA_CRAP ATInfo();

public:
    int32_t StepLength;
    HDRFloat ThresholdC;
    SubType SqrEscapeRadius;
    HDRFloatComplex RefC;
    HDRFloatComplex ZCoeff, CCoeff, InvZCoeff;
    HDRFloatComplex CCoeffSqrInvZCoeff;
    HDRFloatComplex CCoeffInvZCoeff;
    HDRFloat CCoeffNormSqr;
    HDRFloat RefCNormSqr;
    HDRFloat factor;

public:
    CUDA_CRAP bool Usable(HDRFloat SqrRadius) {
        auto result = CCoeffNormSqr * SqrRadius * factor;
        return result.compareToBothPositive(RefCNormSqr) > 0 && SqrEscapeRadius > 4.0f;
    }

    CUDA_CRAP bool isValid(HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex getC(HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex getDZ(HDRFloatComplex z);

    CUDA_CRAP void PerformAT(int32_t max_iterations, HDRFloatComplex DeltaSub0, ATResult<SubType> &result);
};


template<class SubType>
CUDA_CRAP
ATInfo<SubType>::ATInfo() :
    StepLength{},
    ThresholdC{},
    SqrEscapeRadius{},
    RefC{},
    ZCoeff{},
    CCoeff{},
    InvZCoeff{},
    CCoeffSqrInvZCoeff{},
    CCoeffInvZCoeff{},
    CCoeffNormSqr{},
    RefCNormSqr{} {
    factor = HDRFloat(0x1.0p32);
}

template<class SubType>
CUDA_CRAP
bool ATInfo<SubType>::isValid(HDRFloatComplex DeltaSub0) {
    return DeltaSub0.chebychevNorm().compareToBothPositiveReduced(ThresholdC) <= 0;
}

template<class SubType>
CUDA_CRAP
ATInfo<SubType>::HDRFloatComplex ATInfo<SubType>::getC(HDRFloatComplex dc) {
    HDRFloatComplex temp = dc * CCoeff + RefC;
    temp.Reduce();
    return temp;
}

template<class SubType>
CUDA_CRAP
ATInfo<SubType>::HDRFloatComplex ATInfo<SubType>::getDZ(HDRFloatComplex z) {
    HDRFloatComplex temp = z * InvZCoeff;
    temp.Reduce();
    return temp;
}

template<class SubType>
CUDA_CRAP
void ATInfo<SubType>::PerformAT(
    int32_t max_iterations,
    HDRFloatComplex DeltaSub0,
    ATResult<SubType>& result) {
    //int ATMaxIt = (max_iterations - 1) / StepLength + 1;
    HDRFloat nsq;
    const int32_t ATMaxIt = max_iterations / StepLength;
    const HDRFloatComplex c = getC((HDRFloatComplex)DeltaSub0);
    HDRFloatComplex z{};
    int32_t i;

    for (i = 0; i < ATMaxIt; i++) {

        nsq = z.norm_squared();
        HdrReduce(nsq);
        if (nsq.compareToBothPositiveReduced(HDRFloat(SqrEscapeRadius)) > 0) {
            break;
        }

        z = z * z + c;
    }

    result.dz = getDZ(z);
    result.bla_iterations = i * StepLength;
    result.bla_steps = i;
}