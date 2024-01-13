#pragma once

#include <complex>

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "ATResult.h"

template<typename IterType, class HDRFloat, class SubType>
class ATInfo {
    static constexpr bool IsHDR =
        std::is_same<HDRFloat, ::HDRFloat<float>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<double>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<dblflt>>>::value;
    using HDRFloatComplex =
        std::conditional<
        IsHDR,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

public:
    CUDA_CRAP ATInfo();

    template<class T2, class SubType2>
    CUDA_CRAP ATInfo(const ATInfo<IterType, T2, SubType2>& other)
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

        if constexpr (IsHDR) {
            return
                result.compareToBothPositiveReduced(RefCNormSqr) > 0 &&
                SqrEscapeRadius.compareToBothPositiveReduced(Four) > 0;
        }
        else {
            return
                result > RefCNormSqr && SqrEscapeRadius > 4;
        }
    }

    CUDA_CRAP bool isValid(HDRFloatComplex DeltaSub0) const;
    CUDA_CRAP HDRFloatComplex getC(HDRFloatComplex dc) const;
    CUDA_CRAP HDRFloatComplex getDZ(HDRFloatComplex z) const;

    CUDA_CRAP void PerformAT(
        IterType max_iterations,
        HDRFloatComplex DeltaSub0,
        ATResult<IterType, HDRFloat, SubType> &result) const;
};


template<typename IterType, class HDRFloat, class SubType>
CUDA_CRAP
ATInfo<IterType, HDRFloat, SubType>::ATInfo() :
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

template<typename IterType, class HDRFloat, class SubType>
CUDA_CRAP
bool ATInfo<IterType, HDRFloat, SubType>::isValid(HDRFloatComplex DeltaSub0) const {
    if constexpr (IsHDR) {
        return DeltaSub0.chebychevNorm().compareToBothPositiveReduced(ThresholdC) <= 0;
    } else {
        return DeltaSub0.chebychevNorm() <= ThresholdC;
    }
}

template<typename IterType, class HDRFloat, class SubType>
CUDA_CRAP
ATInfo<IterType, HDRFloat, SubType>::HDRFloatComplex ATInfo<IterType, HDRFloat, SubType>::getC(HDRFloatComplex dc) const {
    HDRFloatComplex temp = dc * CCoeff + RefC;
    temp.Reduce();
    return temp;
}

template<typename IterType, class HDRFloat, class SubType>
CUDA_CRAP
ATInfo<IterType, HDRFloat, SubType>::HDRFloatComplex ATInfo<IterType, HDRFloat, SubType>::getDZ(HDRFloatComplex z) const {
    HDRFloatComplex temp = z * InvZCoeff;
    temp.Reduce();
    return temp;
}

template<typename IterType, class HDRFloat, class SubType>
CUDA_CRAP
void ATInfo<IterType, HDRFloat, SubType>::PerformAT(
    IterType max_iterations,
    HDRFloatComplex DeltaSub0,
    ATResult<IterType, HDRFloat, SubType>& result) const {
    //int ATMaxIt = (max_iterations - 1) / StepLength + 1;
    HDRFloat nsq;
    const IterType ATMaxIt = max_iterations / StepLength;
    const HDRFloatComplex c = getC((HDRFloatComplex)DeltaSub0);
    HDRFloatComplex z{};
    IterType i;

    for (i = 0; i < ATMaxIt; i++) {

        nsq = z.norm_squared();
        HdrReduce(nsq);
        if constexpr (IsHDR) {
            if (nsq.compareToBothPositiveReduced(SqrEscapeRadius) > 0) {
                break;
            }
        }
        else {
            if (nsq > SqrEscapeRadius) {
                break;
            }
        }

        z = z * z + c;
    }

    result.dz = getDZ(z);
    result.bla_iterations = i * StepLength;
    result.bla_steps = i;
}