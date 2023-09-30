#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
#include  "LAInfoI.h"

template<class SubType>
class GPU_LAInfoDeep;

template<class SubType>
class LAInfoDeep {
public:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;
    using T = SubType;

    friend class GPU_LAInfoDeep<SubType>;

    static constexpr int DEFAULT_DETECTION_METHOD = 1;
    static constexpr T DefaultStage0PeriodDetectionThreshold = 0x1.0p-10;
    static constexpr T DefaultPeriodDetectionThreshold = 0x1.0p-10;
    static constexpr T DefaultStage0PeriodDetectionThreshold2 = 0x1.0p-6;
    static constexpr T DefaultPeriodDetectionThreshold2 = 0x1.0p-3;
    static constexpr T DefaultLAThresholdScale = 0x1.0p-24;
    static constexpr T DefaultLAThresholdCScale = 0x1.0p-24;
    
public:

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


public:
    CUDA_CRAP LAInfoDeep();
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
    CUDA_CRAP LAstep<HDRFloatComplex> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP void CreateAT(ATInfo<HDRFloat> &Result, LAInfoDeep Next);
    CUDA_CRAP HDRFloat getLAThreshold();
    CUDA_CRAP HDRFloat getLAThresholdC();
    CUDA_CRAP void SetLAi(const LAInfoI &other);
    CUDA_CRAP const LAInfoI &GetLAi() const;
};

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::LAInfoDeep() :
    RefRe{},
    RefIm{},
    RefExp{},
    ZCoeffRe{},
    ZCoeffIm{},
    ZCoeffExp{},
    CCoeffRe{},
    CCoeffIm{},
    CCoeffExp{},
    LAThresholdMant{},
    LAThresholdExp{},
    LAThresholdCMant{},
    LAThresholdCExp{},
    MinMagMant{},
    MinMagExp{},
    LAi{} {
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::LAInfoDeep(HDRFloatComplex z) {
    RefRe = z.getMantissaReal();
    RefIm = z.getMantissaImag();
    RefExp = z.getExp();


    HDRFloatComplex ZCoeff = HDRFloatComplex(1.0, 0);
    HDRFloatComplex CCoeff = HDRFloatComplex(1.0, 0);
    HDRFloat LAThreshold = HDRFloat{ 1 };
    HDRFloat LAThresholdC = HDRFloat{ 1 };

    ZCoeffRe = ZCoeff.getMantissaReal();
    ZCoeffIm = ZCoeff.getMantissaImag();
    ZCoeffExp = ZCoeff.getExp();

    CCoeffRe = CCoeff.getMantissaReal();
    CCoeffIm = CCoeff.getMantissaImag();
    CCoeffExp = CCoeff.getExp();

    LAThresholdMant = LAThreshold.getMantissa();
    LAThresholdExp = LAThreshold.getExp();

    LAThresholdCMant = LAThresholdC.getMantissa();
    LAThresholdCExp = LAThresholdC.getExp();

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        HDRFloat MinMag{ 4 };
        MinMagMant = MinMag.getMantissa();
        MinMagExp = MinMag.getExp();
    }

    LAi = {};
}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::DetectPeriod(HDRFloatComplex z) {
    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        //return z.chebychevNorm().compareToBothPositive(HDRFloat(MinMagExp, MinMagMant).multiply(PeriodDetectionThreshold2)) < 0;
        return z.chebychevNorm().compareToBothPositive(HDRFloat(MinMagExp, MinMagMant) * DefaultPeriodDetectionThreshold2) < 0;
    }
    else {
        //return z.chebychevNorm().divide(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm()).multiply_mutable(LAThresholdScale).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant).multiply(PeriodDetectionThreshold)) < 0;
        return (z.chebychevNorm() /
            (HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm)
            .chebychevNorm())
            * DefaultLAThresholdScale)
            .compareToBothPositive(
                HDRFloat(LAThresholdExp, LAThresholdMant) * DefaultPeriodDetectionThreshold) < 0;
    }
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::getRef() {
    return HDRFloatComplex(RefExp, RefRe, RefIm);
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::getZCoeff() {
    return HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::getCCoeff() {
    return HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::Step(LAInfoDeep& out, HDRFloatComplex z) {
    const HDRFloat ChebyMagz = z.chebychevNorm();

    const HDRFloatComplex ZCoeff{ ZCoeffExp, ZCoeffRe, ZCoeffIm };
    const HDRFloatComplex CCoeff{ CCoeffExp, CCoeffRe, CCoeffIm };

    const HDRFloat ChebyMagZCoeff{ ZCoeff.chebychevNorm() };
    const HDRFloat ChebyMagCCoeff{ CCoeff.chebychevNorm() };

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        HDRFloat outMinMag = HDRFloat::minBothPositiveReduced(ChebyMagz, HDRFloat(MinMagExp, MinMagMant));
        out.MinMagExp = outMinMag.getExp();
        out.MinMagMant = outMinMag.getMantissa();
    }

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * DefaultLAThresholdScale;
    temp1.Reduce();

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * DefaultLAThresholdCScale;
    temp2.Reduce();

    const HDRFloat outLAThreshold{ HDRFloat::minBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant), temp1) };
    const HDRFloat outLAThresholdC{ HDRFloat::minBothPositiveReduced(HDRFloat(LAThresholdCExp, LAThresholdCMant), temp2) };

    out.LAThresholdExp = outLAThreshold.getExp();
    out.LAThresholdMant = outLAThreshold.getMantissa();

    out.LAThresholdCExp = outLAThresholdC.getExp();
    out.LAThresholdCMant = outLAThresholdC.getMantissa();

    const HDRFloatComplex z2{ z * HDRFloat(2) };
    HDRFloatComplex outZCoeff{ z2 * ZCoeff };
    outZCoeff.Reduce();
    HDRFloatComplex outCCoeff{ z2 * CCoeff + HDRFloat{ 1 } };
    outCCoeff.Reduce();

    out.ZCoeffExp = outZCoeff.getExp();
    out.ZCoeffRe = outZCoeff.getMantissaReal();
    out.ZCoeffIm = outZCoeff.getMantissaImag();

    out.CCoeffExp = outCCoeff.getExp();
    out.CCoeffRe = outCCoeff.getMantissaReal();
    out.CCoeffIm = outCCoeff.getMantissaImag();

    out.RefRe = RefRe;
    out.RefIm = RefIm;
    out.RefExp = RefExp;

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        return HDRFloat(out.MinMagExp, out.MinMagMant).compareToBothPositive(HDRFloat(MinMagExp, MinMagMant) * (DefaultStage0PeriodDetectionThreshold2)) < 0;
        //return HDRFloat(out.MinMagExp, out.MinMagMant) < HDRFloat(MinMagExp, MinMagMant) * PeriodDetectionThreshold2;
    }
    else {
        return HDRFloat(out.LAThresholdExp, out.LAThresholdMant).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant) * (DefaultStage0PeriodDetectionThreshold)) < 0;
        //return HDRFloat(LAThresholdExp, LAThresholdMant) < HDRFloat(out.LAThresholdExp, out.LAThresholdMant) * PeriodDetectionThreshold;
    }
}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::isLAThresholdZero() {
    return HDRFloat(LAThresholdExp, LAThresholdMant).compareTo(HDRFloat{ 0 }) == 0;
}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::isZCoeffZero() {
    HDRFloatComplex ZCoeff = HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    return ZCoeff.getRe().compareTo(HDRFloat{ 0 }) == 0 && ZCoeff.getIm().compareTo(HDRFloat{ 0 }) == 0;
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType> LAInfoDeep<SubType>::Step(HDRFloatComplex z) {
    LAInfoDeep Result = LAInfoDeep();

    Step(Result, z);
    return Result;
}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::Composite(LAInfoDeep& out, LAInfoDeep LA) {
    HDRFloatComplex z = HDRFloatComplex(LA.RefExp, LA.RefRe, LA.RefIm);
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloatComplex ZCoeff = HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    HDRFloatComplex CCoeff = HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
    HDRFloat LAThreshold = HDRFloat(LAThresholdExp, LAThresholdMant);
    HDRFloat LAThresholdC = HDRFloat(LAThresholdCExp, LAThresholdCMant);

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

    HDRFloat LA_LAThreshold = HDRFloat(LA.LAThresholdExp, LA.LAThresholdMant);
    HDRFloatComplex LAZCoeff = HDRFloatComplex(LA.ZCoeffExp, LA.ZCoeffRe, LA.ZCoeffIm);
    HDRFloatComplex LACCoeff = HDRFloatComplex(LA.CCoeffExp, LA.CCoeffRe, LA.CCoeffIm);

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

    out.LAThresholdExp = outLAThreshold.getExp();
    out.LAThresholdMant = outLAThreshold.getMantissa();

    out.LAThresholdCExp = outLAThresholdC.getExp();
    out.LAThresholdCMant = outLAThresholdC.getMantissa();

    out.ZCoeffExp = outZCoeff.getExp();
    out.ZCoeffRe = outZCoeff.getMantissaReal();
    out.ZCoeffIm = outZCoeff.getMantissaImag();

    out.CCoeffExp = outCCoeff.getExp();
    out.CCoeffRe = outCCoeff.getMantissaReal();
    out.CCoeffIm = outCCoeff.getMantissaImag();

    out.RefExp = RefExp;
    out.RefRe = RefRe;
    out.RefIm = RefIm;

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        HDRFloat MinMag = HDRFloat(MinMagExp, MinMagMant);
        temp = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
        HDRFloat outMinMag = HDRFloat::minBothPositiveReduced(temp, HDRFloat(LA.MinMagExp, LA.MinMagMant));

        out.MinMagExp = outMinMag.getExp();
        out.MinMagMant = outMinMag.getMantissa();

        return temp.compareToBothPositive(MinMag * DefaultPeriodDetectionThreshold2) < 0;
    }
    else {
        return temp.compareToBothPositive(LAThreshold * DefaultPeriodDetectionThreshold) < 0;
    }
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType> LAInfoDeep<SubType>::Composite(LAInfoDeep LA) {
    LAInfoDeep Result = LAInfoDeep();

    Composite(Result, LA);
    return Result;
}

template<class SubType>
CUDA_CRAP
LAstep<HDRFloatComplex<SubType>> LAInfoDeep<SubType>::Prepare(HDRFloatComplex dz) const {
    //*2 is + 1
    HDRFloatComplex newdz = dz * (HDRFloatComplex(RefExp + 1, RefRe, RefIm) + dz);
    newdz.Reduce();

    LAstep<HDRFloatComplex> temp = LAstep<HDRFloatComplex>();
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) {
    return newdz * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) + dc * HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return  dzdc * HDRFloat(2) * z * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) + HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return (dzdc2 * z + dzdc.square()) * HDRFloat(2) * HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
}

template<class SubType>
CUDA_CRAP
void LAInfoDeep<SubType>::CreateAT(ATInfo<HDRFloat>& Result, LAInfoDeep Next) {
    HDRFloatComplex ZCoeff = HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    HDRFloatComplex CCoeff = HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
    HDRFloat LAThreshold = HDRFloat(LAThresholdExp, LAThresholdMant);
    HDRFloat LAThresholdC = HDRFloat(LAThresholdCExp, LAThresholdCMant);

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

    if constexpr (std::is_same<LAInfoDeep<SubType>::HDRFloat, ::HDRFloat<double>>::value) {
        HDRFloat lim(0x1.0p256); // TODO should be constexpr
        Result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim).toDouble();
        Result.ThresholdC = HDRFloat::minBothPositive(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    }
    else {
        HDRFloat lim(0x1.0p64);
        Result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim).toDouble();
        Result.ThresholdC = HDRFloat::minBothPositive(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    }
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloat LAInfoDeep<SubType>::getLAThreshold() {
    return HDRFloat(LAThresholdExp, LAThresholdMant);
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloat LAInfoDeep<SubType>::getLAThresholdC() {
    return HDRFloat(LAThresholdCExp, LAThresholdCMant);
}

template<class SubType>
CUDA_CRAP void LAInfoDeep<SubType>::SetLAi(const LAInfoI& other) {
    this->LAi = other;
}

template<class SubType>
CUDA_CRAP const LAInfoI& LAInfoDeep<SubType>::GetLAi() const {
    return this->LAi;
}
