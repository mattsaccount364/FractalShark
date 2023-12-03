#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"

template<typename IterType, class Float, class SubType>
class GPU_LAInfoDeep;

template<typename IterType, class Float, class SubType>
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

    friend class GPU_LAInfoDeep<IterType, Float, SubType>;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAInfoDeep<IterType, Float, double>;
    friend class LAInfoDeep<IterType, Float, CudaDblflt<MattDblflt>>;

    static constexpr int DEFAULT_DETECTION_METHOD = 2;
    static constexpr float DefaultStage0PeriodDetectionThreshold = 0x1.0p-10;
    static constexpr float DefaultPeriodDetectionThreshold = 0x1.0p-10;
    static constexpr float DefaultStage0PeriodDetectionThreshold2 = 0x1.0p-6;
    static constexpr float DefaultPeriodDetectionThreshold2 = 0x1.0p-3;
    //static constexpr float DefaultLAThresholdScale =
    //    std::is_same<T, double>::value ? 0x1.0p-24 : 0x1.0p-12;
    //static constexpr float DefaultLAThresholdCScale =
    //    std::is_same<T, double>::value ? 0x1.0p-24 : 0x1.0p-12;
    static constexpr float DefaultLAThresholdScale = 0x1.0p-12;
    static constexpr float DefaultLAThresholdCScale = 0x1.0p-12;
    
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

    template<class Float2, class SubType2>
    CUDA_CRAP LAInfoDeep(const LAInfoDeep<IterType, Float2, SubType2> &other);
    CUDA_CRAP LAInfoDeep(HDRFloatComplex z);
    CUDA_CRAP bool DetectPeriod(HDRFloatComplex z);
    CUDA_CRAP HDRFloatComplex getRef();
    CUDA_CRAP HDRFloatComplex getZCoeff();
    CUDA_CRAP HDRFloatComplex getCCoeff();
    CUDA_CRAP bool Step(LAInfoDeep &out, HDRFloatComplex z);
    CUDA_CRAP bool isLAThresholdZero();
    CUDA_CRAP bool isZCoeffZero();
    CUDA_CRAP LAInfoDeep Step(HDRFloatComplex z);
    CUDA_CRAP bool Composite(LAInfoDeep &out, LAInfoDeep LA);
    CUDA_CRAP LAInfoDeep Composite(LAInfoDeep LA);
    CUDA_CRAP LAstep<IterType, Float, SubType> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP void CreateAT(ATInfo<IterType, Float, SubType> &Result, LAInfoDeep Next);
    CUDA_CRAP HDRFloat getLAThreshold();
    CUDA_CRAP HDRFloat getLAThresholdC();
    CUDA_CRAP void SetLAi(const LAInfoI<IterType>&other);
    CUDA_CRAP const LAInfoI<IterType>&GetLAi() const;
};

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::LAInfoDeep() :
    Ref{},
    LAThreshold{},
    ZCoeff{},
    CCoeff{},
    LAThresholdC{},
    LAi{},
    MinMag{} {
}

template<typename IterType, class Float, class SubType>
template<class Float2, class SubType2>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType>::LAInfoDeep(const LAInfoDeep<IterType, Float2, SubType2>& other) {
    this->Ref = static_cast<HDRFloatComplex>(other.Ref);
    this->LAThreshold = static_cast<HDRFloat>(other.LAThreshold);
    this->ZCoeff = static_cast<HDRFloatComplex>(other.ZCoeff);
    this->CCoeff = static_cast<HDRFloatComplex>(other.CCoeff);
    this->LAThresholdC = static_cast<HDRFloat>(other.LAThresholdC);
    this->MinMag = static_cast<HDRFloat>(other.MinMag);
    this->LAi = other.LAi;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::LAInfoDeep(HDRFloatComplex z) {
    Ref = z;

    HDRFloatComplex ZCoeffLocal = HDRFloatComplex(1.0, 0);
    HDRFloatComplex CCoeffLocal = HDRFloatComplex(1.0, 0);
    HDRFloat LAThresholdLocal = HDRFloat{ 1 };
    HDRFloat LAThresholdCLocal = HDRFloat{ 1 };

    this->ZCoeff = ZCoeffLocal;
    this->CCoeff = CCoeffLocal;
    this->LAThreshold = LAThresholdLocal;
    this->LAThresholdC = LAThresholdCLocal;

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        this->MinMag = HDRFloat{ 4 };
    }

    LAi = {};
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType>::DetectPeriod(HDRFloatComplex z) {
    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        //return z.chebychevNorm().compareToBothPositive(HDRFloat(MinMagExp, MinMagMant).multiply(PeriodDetectionThreshold2)) < 0;
        return z.chebychevNorm().compareToBothPositive(MinMag * DefaultPeriodDetectionThreshold2) < 0;
    }
    else {
        //return z.chebychevNorm().divide(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm()).multiply_mutable(LAThresholdScale).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant).multiply(PeriodDetectionThreshold)) < 0;
        return (z.chebychevNorm() /
            ZCoeff.chebychevNorm() * DefaultLAThresholdScale)
            .compareToBothPositive(LAThreshold * DefaultPeriodDetectionThreshold) < 0;
    }
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::getRef() {
    return Ref;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::getZCoeff() {
    return ZCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::getCCoeff() {
    return CCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType>::Step(LAInfoDeep& out, HDRFloatComplex z) {
    const HDRFloat ChebyMagz = z.chebychevNorm();

    const HDRFloat ChebyMagZCoeff{ ZCoeff.chebychevNorm() };
    const HDRFloat ChebyMagCCoeff{ CCoeff.chebychevNorm() };

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        HDRFloat outMin = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
        out.MinMag = outMin;
    }

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * DefaultLAThresholdScale;
    temp1.Reduce();

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * DefaultLAThresholdCScale;
    temp2.Reduce();

    const HDRFloat outLAThreshold{ HDRFloat::minBothPositiveReduced(LAThreshold, temp1) };
    const HDRFloat outLAThresholdC{ HDRFloat::minBothPositiveReduced(LAThresholdC, temp2) };

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;

    const HDRFloatComplex z2{ z * HDRFloat(2) };
    HDRFloatComplex outZCoeff{ z2 * ZCoeff };
    outZCoeff.Reduce();
    HDRFloatComplex outCCoeff{ z2 * CCoeff + HDRFloat{ 1 } };
    outCCoeff.Reduce();

    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;

    out.Ref = Ref;

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        return out.MinMag.compareToBothPositive(MinMag * (DefaultStage0PeriodDetectionThreshold2)) < 0;
        //return HDRFloat(out.MinMagExp, out.MinMagMant) < HDRFloat(MinMagExp, MinMagMant) * PeriodDetectionThreshold2;
    }
    else {
        return out.LAThreshold.compareToBothPositive(LAThreshold * (DefaultStage0PeriodDetectionThreshold)) < 0;
        //return HDRFloat(LAThresholdExp, LAThresholdMant) < HDRFloat(out.LAThresholdExp, out.LAThresholdMant) * PeriodDetectionThreshold;
    }
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType>::isLAThresholdZero() {
    return LAThreshold.compareTo(HDRFloat{ 0 }) == 0;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType>::isZCoeffZero() {
    return ZCoeff.getRe().compareTo(HDRFloat{ 0 }) == 0 && ZCoeff.getIm().compareTo(HDRFloat{ 0 }) == 0;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType> LAInfoDeep<IterType, Float, SubType>::Step(HDRFloatComplex z) {
    LAInfoDeep Result = LAInfoDeep();

    Step(Result, z);
    return Result;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
bool LAInfoDeep<IterType, Float, SubType>::Composite(LAInfoDeep& out, LAInfoDeep LA) {
    HDRFloatComplex z = LA.Ref;
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
    HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * DefaultLAThresholdScale;
    temp1.Reduce();

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * DefaultLAThresholdCScale;
    temp2.Reduce();

    HDRFloat outLAThreshold = HDRFloat::minBothPositiveReduced(LAThreshold, temp1);
    HDRFloat outLAThresholdC = HDRFloat::minBothPositiveReduced(LAThresholdC, temp2);

    HDRFloatComplex z2 = z * HDRFloat(2);
    HDRFloatComplex outZCoeff = z2 * ZCoeff;
    outZCoeff.Reduce();
    //double RescaleFactor = out.LAThreshold / LAThreshold;
    HDRFloatComplex outCCoeff = z2 * CCoeff;
    outCCoeff.Reduce();

    ChebyMagZCoeff = outZCoeff.chebychevNorm();
    ChebyMagCCoeff = outCCoeff.chebychevNorm();
    HDRFloat temp = outLAThreshold;

    HDRFloat LA_LAThreshold = LA.LAThreshold;
    HDRFloatComplex LAZCoeff = LA.ZCoeff;
    HDRFloatComplex LACCoeff = LA.CCoeff;

    temp1 = LA_LAThreshold / ChebyMagZCoeff;
    temp1.Reduce();

    temp2 = LA_LAThreshold / ChebyMagCCoeff;
    temp2.Reduce();

    outLAThreshold = HDRFloat::minBothPositiveReduced(outLAThreshold, temp1);
    outLAThresholdC = HDRFloat::minBothPositiveReduced(outLAThresholdC, temp2);
    outZCoeff = outZCoeff * LAZCoeff;
    outZCoeff.Reduce();
    //RescaleFactor = out.LAThreshold / temp;
    outCCoeff = outCCoeff * LAZCoeff + LACCoeff;
    outCCoeff.Reduce();

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;
    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;
    out.Ref = Ref;

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        temp = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
        HDRFloat outMinMag = HDRFloat::minBothPositiveReduced(temp, LA.MinMag);
        out.MinMag = outMinMag;

        return temp.compareToBothPositive(MinMag * DefaultPeriodDetectionThreshold2) < 0;
    }
    else {
        return temp.compareToBothPositive(LAThreshold * DefaultPeriodDetectionThreshold) < 0;
    }
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType> LAInfoDeep<IterType, Float, SubType>::Composite(LAInfoDeep LA) {
    LAInfoDeep Result = LAInfoDeep();

    Composite(Result, LA);
    return Result;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAstep<IterType, Float, SubType> LAInfoDeep<IterType, Float, SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (Ref * HDRFloat(2) + dz);
    newdz.Reduce();

    LAstep<IterType, Float, SubType> temp = {};
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(LAThreshold) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) {
    return newdz * ZCoeff + dc * CCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return  dzdc * HDRFloat(2) * z * ZCoeff + CCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex LAInfoDeep<IterType, Float, SubType>::EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return (dzdc2 * z + dzdc.square()) * HDRFloat(2) * ZCoeff;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
void LAInfoDeep<IterType, Float, SubType>::CreateAT(ATInfo<IterType, Float, SubType>& Result, LAInfoDeep Next) {
    Result.ZCoeff = ZCoeff;
    Result.CCoeff = ZCoeff * CCoeff;
    Result.CCoeff.Reduce();

    Result.InvZCoeff = ZCoeff.reciprocal();
    Result.InvZCoeff.Reduce();

    Result.CCoeffSqrInvZCoeff = Result.CCoeff * Result.CCoeff * Result.InvZCoeff;
    Result.CCoeffSqrInvZCoeff.Reduce();

    Result.CCoeffInvZCoeff = Result.CCoeff * Result.InvZCoeff;
    Result.CCoeffInvZCoeff.Reduce();

    Result.RefC = Next.getRef() * ZCoeff;
    Result.RefC.Reduce();

    Result.CCoeffNormSqr = Result.CCoeff.norm_squared();
    Result.CCoeffNormSqr.Reduce();

    Result.RefCNormSqr = Result.RefC.norm_squared();
    Result.RefCNormSqr.Reduce();

    HDRFloat lim(32, 1);
    if constexpr (std::is_same<LAInfoDeep<IterType, Float, SubType>::HDRFloat, ::HDRFloat<double>>::value) {
        lim.setExp(256);
    }

    HdrReduce(lim);

    Result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim);
    HdrReduce(Result.SqrEscapeRadius);

    Result.ThresholdC = HDRFloat::minBothPositive(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloat LAInfoDeep<IterType, Float, SubType>::getLAThreshold() {
    return LAThreshold;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType>::HDRFloat LAInfoDeep<IterType, Float, SubType>::getLAThresholdC() {
    return LAThresholdC;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP void LAInfoDeep<IterType, Float, SubType>::SetLAi(const LAInfoI<IterType>& other) {
    this->LAi = other;
}

template<typename IterType, class Float, class SubType>
CUDA_CRAP const LAInfoI<IterType>& LAInfoDeep<IterType, Float, SubType>::GetLAi() const {
    return this->LAi;
}
