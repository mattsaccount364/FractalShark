#pragma once

#include <complex>

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "ATResult.h"

template<typename IterType, class SubType>
class ATInfo {
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;


public:
    CUDA_CRAP ATInfo();

    template<class SubType2>
    CUDA_CRAP ATInfo(const ATInfo<IterType, SubType2>& other)
        : StepLength(other.StepLength),
          ThresholdC(other.ThresholdC),
          SqrEscapeRadius(other.SqrEscapeRadius),
          RefC(other.RefC),
          ZCoeff(other.ZCoeff),
          CCoeff(other.CCoeff),
          InvZCoeff(other.InvZCoeff),
          CCoeffSqrInvZCoeff(other.CCoeffSqrInvZCoeff),
          CCoeffInvZCoeff(other.CCoeffInvZCoeff),
          CCoeffNormSqr(other.CCoeffNormSqr),
          RefCNormSqr(other.RefCNormSqr),
          factor(other.factor) {
    }

public:
    IterType StepLength;
    HDRFloat ThresholdC;
    HDRFloat SqrEscapeRadius;
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
        HdrReduce(result);
        auto Four = HDRFloat(4.0f);
        HdrReduce(Four);

        return
            result.compareToBothPositiveReduced(RefCNormSqr) > 0 &&
            SqrEscapeRadius.compareToBothPositiveReduced(Four) > 0;
    }

    CUDA_CRAP bool isValid(HDRFloatComplex DeltaSub0);
    CUDA_CRAP HDRFloatComplex getC(HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex getDZ(HDRFloatComplex z);

    CUDA_CRAP void PerformAT(IterType max_iterations, HDRFloatComplex DeltaSub0, ATResult<IterType, SubType> &result);
};


template<typename IterType, class SubType>
CUDA_CRAP
ATInfo<IterType, SubType>::ATInfo() :
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

template<typename IterType, class SubType>
CUDA_CRAP
bool ATInfo<IterType, SubType>::isValid(HDRFloatComplex DeltaSub0) {
    return DeltaSub0.chebychevNorm().compareToBothPositiveReduced(ThresholdC) <= 0;
}

template<typename IterType, class SubType>
CUDA_CRAP
ATInfo<IterType, SubType>::HDRFloatComplex ATInfo<IterType, SubType>::getC(HDRFloatComplex dc) {
    HDRFloatComplex temp = dc * CCoeff + RefC;
    temp.Reduce();
    return temp;
}

template<typename IterType, class SubType>
CUDA_CRAP
ATInfo<IterType, SubType>::HDRFloatComplex ATInfo<IterType, SubType>::getDZ(HDRFloatComplex z) {
    HDRFloatComplex temp = z * InvZCoeff;
    temp.Reduce();
    return temp;
}

template<typename IterType, class SubType>
CUDA_CRAP
void ATInfo<IterType, SubType>::PerformAT(
    IterType max_iterations,
    HDRFloatComplex DeltaSub0,
    ATResult<IterType, SubType>& result) {
    //int ATMaxIt = (max_iterations - 1) / StepLength + 1;
    HDRFloat nsq;
    const IterType ATMaxIt = max_iterations / StepLength;
    const HDRFloatComplex c = getC((HDRFloatComplex)DeltaSub0);
    HDRFloatComplex z{};
    IterType i;

    for (i = 0; i < ATMaxIt; i++) {

        nsq = z.norm_squared();
        HdrReduce(nsq);
        if (nsq.compareToBothPositiveReduced(SqrEscapeRadius) > 0) {
            break;
        }

        z = z * z + c;
    }

    result.dz = getDZ(z);
    result.bla_iterations = i * StepLength;
    result.bla_steps = i;
}