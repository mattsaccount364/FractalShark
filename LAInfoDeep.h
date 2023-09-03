#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"

template<class SubType>
class LAInfoDeep {
public:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;
    using T = SubType;

    static constexpr int DEFAULT_DETECTION_METHOD = 1;
    static constexpr T DefaultStage0PeriodDetectionThreshold = 0x1.0p-10;
    static constexpr T DefaultPeriodDetectionThreshold = 0x1.0p-10;
    static constexpr T DefaultStage0PeriodDetectionThreshold2 = 0x1.0p-6;
    static constexpr T DefaultPeriodDetectionThreshold2 = 0x1.0p-3;
    static constexpr T DefaultLAThresholdScale = 0x1.0p-24;
    static constexpr T DefaultLAThresholdCScale = 0x1.0p-24;
    static constexpr int DEFAULT_SERIES_APPROXIMATION_TERMS = 5;
    static constexpr long DEFAULT_SERIES_APPROXIMATION_OOM_DIFFERENCE = 2;
    static constexpr int DEFAULT_SERIES_APPROXIMATION_MAX_SKIP_ITER = INT32_MAX;
    static constexpr int DEFAULT_BLA_BITS = 23;
    static constexpr int DEFAULT_BLA_STARTING_LEVEL = 2;
    static constexpr int DEFAULT_NANOMB1_N = 8;
    static constexpr int DEFAULT_NANOMB1_M = 16;

    //LAInfoDeep::HDRFloat LAInfoDeep::Stage0PeriodDetectionThreshold(DefaultStage0PeriodDetectionThreshold);
    //LAInfoDeep::HDRFloat LAInfoDeep::PeriodDetectionThreshold(DefaultPeriodDetectionThreshold);
    //LAInfoDeep::HDRFloat LAInfoDeep::Stage0PeriodDetectionThreshold2(DefaultStage0PeriodDetectionThreshold2);
    //LAInfoDeep::HDRFloat LAInfoDeep::PeriodDetectionThreshold2(DefaultPeriodDetectionThreshold2);
    //LAInfoDeep::HDRFloat LAInfoDeep::LAThresholdScale(DefaultLAThresholdScale);
    //LAInfoDeep::HDRFloat LAInfoDeep::LAThresholdCScale(DefaultLAThresholdCScale);

    T Stage0PeriodDetectionThreshold = DefaultStage0PeriodDetectionThreshold;
    T PeriodDetectionThreshold = DefaultPeriodDetectionThreshold;
    T Stage0PeriodDetectionThreshold2 = DefaultStage0PeriodDetectionThreshold2;
    T PeriodDetectionThreshold2 = DefaultPeriodDetectionThreshold2;
    T LAThresholdScale = DefaultLAThresholdScale;
    T LAThresholdCScale = DefaultLAThresholdCScale;

private:

    T RefRe, RefIm;
    long RefExp;

    T ZCoeffRe, ZCoeffIm;
    long ZCoeffExp;

    T CCoeffRe, CCoeffIm;
    long CCoeffExp;

    T LAThresholdMant;
    long LAThresholdExp;

    T LAThresholdCMant;
    long LAThresholdCExp;

    T MinMagMant;
    long MinMagExp;

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
    CUDA_CRAP LAstep<HDRFloatComplex> Prepare(HDRFloatComplex dz);
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    CUDA_CRAP void CreateAT(ATInfo<HDRFloat> &Result, LAInfoDeep Next);
    CUDA_CRAP HDRFloat getLAThreshold();
    CUDA_CRAP HDRFloat getLAThresholdC();
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
    MinMagExp{} {
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

}

template<class SubType>
CUDA_CRAP
bool LAInfoDeep<SubType>::DetectPeriod(HDRFloatComplex z) {
    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        //return z.chebychevNorm().compareToBothPositive(HDRFloat(MinMagExp, MinMagMant).multiply(PeriodDetectionThreshold2)) < 0;
        return z.chebychevNorm() < HDRFloat(MinMagExp, MinMagMant) * Stage0PeriodDetectionThreshold2;
    }
    else {
        //return z.chebychevNorm().divide(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm()).multiply_mutable(LAThresholdScale).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant).multiply(PeriodDetectionThreshold)) < 0;
        return z.chebychevNorm() / HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm() * LAThresholdScale < HDRFloat(LAThresholdExp, LAThresholdMant) * Stage0PeriodDetectionThreshold;
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
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloatComplex ZCoeff = HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    HDRFloatComplex CCoeff = HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);

    HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
    HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

    if constexpr (DEFAULT_DETECTION_METHOD == 1) {
        HDRFloat outMinMag = HDRFloat::minBothPositiveReduced(ChebyMagz, HDRFloat(MinMagExp, MinMagMant));
        out.MinMagExp = outMinMag.getExp();
        out.MinMagMant = outMinMag.getMantissa();
    }

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * LAThresholdScale;
    temp1.Reduce();

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * LAThresholdCScale;
    temp2.Reduce();

    HDRFloat outLAThreshold = HDRFloat::minBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant), temp1);
    HDRFloat outLAThresholdC = HDRFloat::minBothPositiveReduced(HDRFloat(LAThresholdCExp, LAThresholdCMant), temp2);

    out.LAThresholdExp = outLAThreshold.getExp();
    out.LAThresholdMant = outLAThreshold.getMantissa();

    out.LAThresholdCExp = outLAThresholdC.getExp();
    out.LAThresholdCMant = outLAThresholdC.getMantissa();

    HDRFloatComplex z2 = z.times2();
    HDRFloatComplex outZCoeff = z2.times(ZCoeff);
    outZCoeff.Reduce();
    HDRFloatComplex outCCoeff = z2.times(CCoeff).plus_mutable(HDRFloat{ 1 });
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
        return HDRFloat(out.MinMagExp, out.MinMagMant).compareToBothPositive(HDRFloat(MinMagExp, MinMagMant) * (Stage0PeriodDetectionThreshold2)) < 0;
        //return HDRFloat(out.MinMagExp, out.MinMagMant) < HDRFloat(MinMagExp, MinMagMant) * PeriodDetectionThreshold2;
    }
    else {
        return HDRFloat(out.LAThresholdExp, out.LAThresholdMant).compareToBothPositive(HDRFloat(LAThresholdExp, LAThresholdMant) * (Stage0PeriodDetectionThreshold)) < 0;
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

    HDRFloat temp1 = ChebyMagz / ChebyMagZCoeff * LAThresholdScale;
    temp1.Reduce();

    HDRFloat temp2 = ChebyMagz / ChebyMagCCoeff * LAThresholdCScale;
    temp2.Reduce();

    HDRFloat outLAThreshold = HDRFloat::minBothPositiveReduced(LAThreshold, temp1);
    HDRFloat outLAThresholdC = HDRFloat::minBothPositiveReduced(LAThresholdC, temp2);

    HDRFloatComplex z2 = z.times2();
    HDRFloatComplex outZCoeff = z2.times(ZCoeff);
    outZCoeff.Reduce();
    //double RescaleFactor = out.LAThreshold / LAThreshold;
    HDRFloatComplex outCCoeff = z2.times(CCoeff);
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
    outZCoeff = outZCoeff.times(LAZCoeff);
    outZCoeff.Reduce();
    //RescaleFactor = out.LAThreshold / temp;
    outCCoeff = outCCoeff.times(LAZCoeff).plus_mutable(LACCoeff);
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

        return temp.compareToBothPositive(MinMag * PeriodDetectionThreshold2) < 0;
    }
    else {
        return temp.compareToBothPositive(LAThreshold * PeriodDetectionThreshold) < 0;
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
LAstep<HDRFloatComplex<SubType>> LAInfoDeep<SubType>::Prepare(HDRFloatComplex dz) {
    //*2 is + 1
    HDRFloatComplex newdz = dz.times(HDRFloatComplex(RefExp + 1, RefRe, RefIm).plus_mutable(dz));
    newdz.Reduce();

    LAstep<HDRFloatComplex> temp = LAstep<HDRFloatComplex>();
    temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
    temp.newDzDeep = newdz;
    return temp;
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) {
    return newdz.times(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm)).plus_mutable(dc.times(HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm)));
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc) {
    return  dzdc.times2().times_mutable(z).times_mutable(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm)).plus_mutable(HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm));
}

template<class SubType>
CUDA_CRAP
LAInfoDeep<SubType>::HDRFloatComplex LAInfoDeep<SubType>::EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
    return  dzdc2.times(z)
        .plus_mutable(dzdc.square()).times2_mutable().times_mutable(HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm));
}

template<class SubType>
CUDA_CRAP
void LAInfoDeep<SubType>::CreateAT(ATInfo<HDRFloat>& Result, LAInfoDeep Next) {
    HDRFloatComplex ZCoeff = HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    HDRFloatComplex CCoeff = HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
    HDRFloat LAThreshold = HDRFloat(LAThresholdExp, LAThresholdMant);
    HDRFloat LAThresholdC = HDRFloat(LAThresholdCExp, LAThresholdCMant);

    Result.ZCoeff = ZCoeff;
    Result.CCoeff = ZCoeff.times(CCoeff);
    Result.CCoeff.Reduce();

    Result.InvZCoeff = ZCoeff.reciprocal();
    Result.InvZCoeff.Reduce();

    Result.CCoeffSqrInvZCoeff = Result.CCoeff.square().times_mutable(Result.InvZCoeff);
    Result.CCoeffSqrInvZCoeff.Reduce();

    Result.CCoeffInvZCoeff = Result.CCoeff.times(Result.InvZCoeff);
    Result.CCoeffInvZCoeff.Reduce();

    Result.RefC = Next.getRef().times(ZCoeff);
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

