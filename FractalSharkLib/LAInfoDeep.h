#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"
#include "LAParameters.h"

template<typename IterType, class Float, class SubType>
class GPU_LAInfoDeep;

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAInfoDeep {
public:
    using HDRFloat = Float;
    using HDRFloatComplex =
        std::conditional<
        std::is_same<Float, ::HDRFloat<float>>::value ||
        std::is_same<Float, ::HDRFloat<double>>::value ||
        std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;
    using T = SubType;
    static constexpr bool IsHDR =
        std::is_same<HDRFloat, ::HDRFloat<float>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<double>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<dblflt>>>::value;

    friend class GPU_LAInfoDeep<IterType, Float, SubType>;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAInfoDeep<IterType, Float, double, PExtras>;
    friend class LAInfoDeep<IterType, Float, CudaDblflt<MattDblflt>, PExtras>;

public:

    HDRFloatComplex Ref;
    HDRFloatComplex ZCoeff;
    HDRFloatComplex CCoeff;
    HDRFloat LAThreshold;
    HDRFloat LAThresholdC;
    HDRFloat MinMag;
    LAInfoI<IterType> LAi;


public:
    CUDA_CRAP LAInfoDeep();

    template<class Float2, class SubType2, PerturbExtras PExtras2>
    CUDA_CRAP LAInfoDeep(const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other);
    CUDA_CRAP LAInfoDeep(const LAParameters &la_parameters, HDRFloatComplex z);
    CUDA_CRAP bool DetectPeriod(const LAParameters &la_parameters, HDRFloatComplex z);
    CUDA_CRAP HDRFloatComplex getRef();
    CUDA_CRAP HDRFloatComplex getZCoeff();
    CUDA_CRAP HDRFloatComplex getCCoeff();
    CUDA_CRAP bool Step(
        const LAParameters &la_parameters,
        LAInfoDeep &out,
        HDRFloatComplex z) const;

    CUDA_CRAP bool isLAThresholdZero();
    CUDA_CRAP bool isZCoeffZero();
    CUDA_CRAP LAInfoDeep Step(
        const LAParameters &la_parameters,
        HDRFloatComplex z);

    CUDA_CRAP bool Composite(
        const LAParameters &la_parameters,
        LAInfoDeep &out,
        LAInfoDeep LA);
    CUDA_CRAP LAInfoDeep Composite(const LAParameters &la_parameters, LAInfoDeep LA);
    CUDA_CRAP LAstep<IterType, Float, SubType, PExtras> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP void CreateAT(ATInfo<IterType, Float, SubType> &Result, LAInfoDeep Next, bool UseSmallExponents);
    CUDA_CRAP HDRFloat getLAThreshold();
    CUDA_CRAP HDRFloat getLAThresholdC();
    CUDA_CRAP void SetLAi(const LAInfoI<IterType> &other);
    CUDA_CRAP const LAInfoI<IterType> &GetLAi() const;
};

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep() :
    Ref{},
    LAThreshold{},
    ZCoeff{},
    CCoeff{},
    LAThresholdC{},
    LAi{},
    MinMag{} {
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<class Float2, class SubType2, PerturbExtras PExtras2>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep(
    const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other) {

    this->Ref = static_cast<HDRFloatComplex>(other.Ref);
    this->LAThreshold = static_cast<HDRFloat>(other.LAThreshold);
    this->ZCoeff = static_cast<HDRFloatComplex>(other.ZCoeff);
    this->CCoeff = static_cast<HDRFloatComplex>(other.CCoeff);
    this->LAThresholdC = static_cast<HDRFloat>(other.LAThresholdC);
    this->MinMag = static_cast<HDRFloat>(other.MinMag);
    this->LAi = other.LAi;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep(
    const LAParameters &la_parameters,
    HDRFloatComplex z)
    : MinMag{} {

    Ref = z;

    HDRFloatComplex ZCoeffLocal = HDRFloatComplex(1.0, 0);
    HDRFloatComplex CCoeffLocal = HDRFloatComplex(1.0, 0);
    HDRFloat LAThresholdLocal = HDRFloat{ 1 };
    HDRFloat LAThresholdCLocal = HDRFloat{ 1 };

    this->ZCoeff = ZCoeffLocal;
    this->CCoeff = CCoeffLocal;
    this->LAThreshold = LAThresholdLocal;
    this->LAThresholdC = LAThresholdCLocal;

    if (la_parameters.GetDetectionMethod() == 1) {
        this->MinMag = HDRFloat{ 4 };
    }

    LAi = {};
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType, PExtras>::DetectPeriod(const LAParameters &la_parameters, HDRFloatComplex z) {
    if (la_parameters.GetDetectionMethod() == 1) {
        if constexpr (IsHDR) {
            //return z.chebychevNorm().compareToBothPositive(HDRFloat(MinMagExp, MinMagMant).multiply(PeriodDetectionThreshold2)) < 0;
            return z.chebychevNorm().compareToBothPositive(MinMag * la_parameters.GetPeriodDetectionThreshold2()) < 0;
        } else {
            return z.chebychevNorm() < (MinMag * la_parameters.GetPeriodDetectionThreshold2());
        }
    } else {
        //return z.chebychevNorm().divide(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm()).multiply_mutable(LAThresholdScale).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant).multiply(PeriodDetectionThreshold)) < 0;
        if constexpr (IsHDR) {
            return (z.chebychevNorm() /
                ZCoeff.chebychevNorm() * la_parameters.GetLAThresholdScale())
                .compareToBothPositive(LAThreshold * la_parameters.GetPeriodDetectionThreshold()) < 0;
        } else {
            return (z.chebychevNorm() /
                ZCoeff.chebychevNorm() * la_parameters.GetLAThresholdScale()) < (LAThreshold * la_parameters.GetPeriodDetectionThreshold());
        }
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::getRef() {
    return Ref;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::getZCoeff() {
    return ZCoeff;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::getCCoeff() {
    return CCoeff;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType, PExtras>::Step(
    const LAParameters &la_parameters,
    LAInfoDeep &out,
    HDRFloatComplex z) const {

    const HDRFloat ChebyMagz = z.chebychevNorm();

    const HDRFloat ChebyMagZCoeff{ ZCoeff.chebychevNorm() };
    const HDRFloat ChebyMagCCoeff{ CCoeff.chebychevNorm() };

    if (la_parameters.GetDetectionMethod() == 1) {
        if constexpr (IsHDR) {
            HDRFloat outMin = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
            out.MinMag = outMin;
        } else {
            HDRFloat outMin = std::min(ChebyMagz, MinMag);
            out.MinMag = outMin;
        }
    }

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * la_parameters.GetLAThresholdScale();
    HdrReduce(temp1);

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * la_parameters.GetLAThresholdCScale();
    HdrReduce(temp2);

    HDRFloat outLAThreshold;
    HDRFloat outLAThresholdC;
    if constexpr (IsHDR) {
        outLAThreshold = HDRFloat::minBothPositiveReduced(LAThreshold, temp1);
        outLAThresholdC = HDRFloat::minBothPositiveReduced(LAThresholdC, temp2);
    } else {
        outLAThreshold = std::min(LAThreshold, temp1);
        outLAThresholdC = std::min(LAThresholdC, temp2);
    }

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;

    const HDRFloatComplex z2{ z * HDRFloat(2) };
    HDRFloatComplex outZCoeff{ z2 * ZCoeff };
    HdrReduce(outZCoeff);
    HDRFloatComplex outCCoeff{ z2 * CCoeff + HDRFloat{ 1 } };
    HdrReduce(outCCoeff);

    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;

    out.Ref = Ref;

    if (la_parameters.GetDetectionMethod() == 1) {
        // TODO: Double check the right thresholds here.  Look at the name of the variables and 
        // look at the default detection method number.  Obviously 1 != 2
        if constexpr (IsHDR) {
            return out.MinMag.compareToBothPositive(MinMag * (la_parameters.GetStage0PeriodDetectionThreshold2())) < 0;
        } else {
            return out.MinMag < (MinMag * la_parameters.GetStage0PeriodDetectionThreshold2());
        }
        //return HDRFloat(out.MinMagExp, out.MinMagMant) < HDRFloat(MinMagExp, MinMagMant) * PeriodDetectionThreshold2;
    } else {
        if constexpr (IsHDR) {
            return out.LAThreshold.compareToBothPositive(LAThreshold * (la_parameters.GetStage0PeriodDetectionThreshold())) < 0;
        } else {
            return out.LAThreshold < (LAThreshold * (la_parameters.GetStage0PeriodDetectionThreshold()));
        }

        //return HDRFloat(LAThresholdExp, LAThresholdMant) < HDRFloat(out.LAThresholdExp, out.LAThresholdMant) * PeriodDetectionThreshold;
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType, PExtras>::isLAThresholdZero() {
    if constexpr (IsHDR) {
        return LAThreshold.compareTo(HDRFloat{ 0 }) == 0;
    } else {
        return LAThreshold == 0;
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType, PExtras>::isZCoeffZero() {
    if constexpr (IsHDR) {
        return ZCoeff.getRe().compareTo(HDRFloat{ 0 }) == 0 && ZCoeff.getIm().compareTo(HDRFloat{ 0 }) == 0;
    } else {
        return ZCoeff.getRe() == 0 && ZCoeff.getIm() == 0;
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras> LAInfoDeep<IterType, Float, SubType, PExtras>::Step(
    const LAParameters &la_parameters,
    HDRFloatComplex z) {

    LAInfoDeep Result = LAInfoDeep();

    Step(la_parameters, Result, z);
    return Result;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType, PExtras>::Composite(
    const LAParameters &la_parameters,
    LAInfoDeep &out,
    LAInfoDeep LA) {

    HDRFloatComplex z = LA.Ref;
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
    HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * la_parameters.GetLAThresholdScale();
    HdrReduce(temp1);

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * la_parameters.GetLAThresholdCScale();
    HdrReduce(temp2);

    HDRFloat outLAThreshold;
    HDRFloat outLAThresholdC;
    if constexpr (IsHDR) {
        outLAThreshold = HDRFloat::minBothPositiveReduced(LAThreshold, temp1);
        outLAThresholdC = HDRFloat::minBothPositiveReduced(LAThresholdC, temp2);
    } else {
        outLAThreshold = std::min(LAThreshold, temp1);
        outLAThresholdC = std::min(LAThresholdC, temp2);
    }

    HDRFloatComplex z2 = z * HDRFloat(2);
    HDRFloatComplex outZCoeff = z2 * ZCoeff;
    HdrReduce(outZCoeff);
    //double RescaleFactor = out.LAThreshold / LAThreshold;
    HDRFloatComplex outCCoeff = z2 * CCoeff;
    HdrReduce(outCCoeff);

    ChebyMagZCoeff = outZCoeff.chebychevNorm();
    ChebyMagCCoeff = outCCoeff.chebychevNorm();
    HDRFloat temp = outLAThreshold;

    HDRFloat LA_LAThreshold = LA.LAThreshold;
    HDRFloatComplex LAZCoeff = LA.ZCoeff;
    HDRFloatComplex LACCoeff = LA.CCoeff;

    temp1 = LA_LAThreshold / ChebyMagZCoeff;
    HdrReduce(temp1);

    temp2 = LA_LAThreshold / ChebyMagCCoeff;
    HdrReduce(temp2);

    if constexpr (IsHDR) {
        outLAThreshold = HDRFloat::minBothPositiveReduced(outLAThreshold, temp1);
        outLAThresholdC = HDRFloat::minBothPositiveReduced(outLAThresholdC, temp2);
    } else {
        outLAThreshold = std::min(outLAThreshold, temp1);
        outLAThresholdC = std::min(outLAThresholdC, temp2);
    }
    outZCoeff = outZCoeff * LAZCoeff;
    HdrReduce(outZCoeff);
    //RescaleFactor = out.LAThreshold / temp;
    outCCoeff = outCCoeff * LAZCoeff + LACCoeff;
    HdrReduce(outCCoeff);

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;
    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;
    out.Ref = Ref;

    if (la_parameters.GetDetectionMethod() == 1) {
        if constexpr (IsHDR) {
            temp = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
            out.MinMag = HDRFloat::minBothPositiveReduced(temp, LA.MinMag);
            return temp.compareToBothPositive(MinMag * la_parameters.GetPeriodDetectionThreshold2()) < 0;
        } else {
            temp = std::min(ChebyMagz, MinMag);
            out.MinMag = std::min(temp, LA.MinMag);
            return temp < (MinMag * la_parameters.GetPeriodDetectionThreshold2());
        }
    } else {
        if constexpr (IsHDR) {
            return temp.compareToBothPositive(LAThreshold * la_parameters.GetPeriodDetectionThreshold()) < 0;
        } else {
            return temp < (LAThreshold * la_parameters.GetPeriodDetectionThreshold());
        }
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras> LAInfoDeep<IterType, Float, SubType, PExtras>::Composite(
    const LAParameters &la_parameters,
    LAInfoDeep LA) {

    LAInfoDeep Result = LAInfoDeep();

    Composite(la_parameters, Result, LA);
    return Result;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAstep<IterType, Float, SubType, PExtras> LAInfoDeep<IterType, Float, SubType, PExtras>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (Ref * HDRFloat(2) + dz);
    HdrReduce(newdz);

    LAstep<IterType, Float, SubType, PExtras> temp{};

    if constexpr (IsHDR) {
        temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(LAThreshold) >= 0;
    } else {
        temp.unusable = newdz.chebychevNorm() >= LAThreshold;
    }
    temp.newDzDeep = newdz;
    return temp;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) {
    return newdz * ZCoeff + dc * CCoeff;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return  dzdc * HDRFloat(2) * z * ZCoeff + CCoeff;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return (dzdc2 * z + dzdc.square()) * HDRFloat(2) * ZCoeff;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
void LAInfoDeep<IterType, Float, SubType, PExtras>::CreateAT(
    ATInfo<IterType, Float, SubType> &Result,
    LAInfoDeep Next,
    bool UseSmallExponents) {
    Result.ZCoeff = ZCoeff;
    Result.CCoeff = ZCoeff * CCoeff;
    HdrReduce(Result.CCoeff);

    Result.InvZCoeff = ZCoeff.reciprocal();
    HdrReduce(Result.InvZCoeff);

    Result.CCoeffSqrInvZCoeff = Result.CCoeff * Result.CCoeff * Result.InvZCoeff;
    HdrReduce(Result.CCoeffSqrInvZCoeff);

    Result.CCoeffInvZCoeff = Result.CCoeff * Result.InvZCoeff;
    HdrReduce(Result.CCoeffInvZCoeff);

    Result.RefC = Next.getRef() * ZCoeff;
    HdrReduce(Result.RefC);

    Result.CCoeffNormSqr = Result.CCoeff.norm_squared();
    HdrReduce(Result.CCoeffNormSqr);

    Result.RefCNormSqr = Result.RefC.norm_squared();
    HdrReduce(Result.RefCNormSqr);

    HDRFloat lim;
    if constexpr (IsHDR) {
        lim = HDRFloat(32, 1);
        if constexpr (std::is_same<HDRFloat, ::HDRFloat<double>>::value) {
            if (!UseSmallExponents) {
                lim.setExp(256);
            }
        }
        HdrReduce(lim);
        Result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim);
        HdrReduce(Result.SqrEscapeRadius);

        Result.ThresholdC = HDRFloat::minBothPositive(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    } else {
        lim = 4294967296.0f;
        Result.SqrEscapeRadius = std::min(ZCoeff.norm_squared() * LAThreshold, lim);
        Result.ThresholdC = std::min(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    }
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloat LAInfoDeep<IterType, Float, SubType, PExtras>::getLAThreshold() {
    return LAThreshold;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloat LAInfoDeep<IterType, Float, SubType, PExtras>::getLAThresholdC() {
    return LAThresholdC;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void LAInfoDeep<IterType, Float, SubType, PExtras>::SetLAi(const LAInfoI<IterType> &other) {
    this->LAi = other;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP const LAInfoI<IterType> &LAInfoDeep<IterType, Float, SubType, PExtras>::GetLAi() const {
    return this->LAi;
}
