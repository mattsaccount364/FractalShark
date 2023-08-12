#pragma once

#include <HDRFloat.h>
#include <HDRFloatComplex.h>

//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.HDRFloat;
//import fractalzoomer.core.HDRFloatComplex;

class LAInfoDeep {
public:
    using HDRFloat = HDRFloat<float>;

    static HDRFloat Stage0PeriodDetectionThreshold = new HDRFloat(LAInfo.Stage0PeriodDetectionThreshold);
    static HDRFloat PeriodDetectionThreshold = new HDRFloat(LAInfo.PeriodDetectionThreshold);
    static HDRFloat Stage0PeriodDetectionThreshold2 = new HDRFloat(LAInfo.Stage0PeriodDetectionThreshold2);
    static HDRFloat PeriodDetectionThreshold2 = new HDRFloat(LAInfo.PeriodDetectionThreshold2);
    static HDRFloat LAThresholdScale = new HDRFloat(LAInfo.LAThresholdScale);
    static HDRFloat LAThresholdCScale = new HDRFloat(LAInfo.LAThresholdCScale);
private:
    static HDRFloat atLimit = new HDRFloat(0x1.0p256);

    double RefRe, RefIm;
    long RefExp;

    double ZCoeffRe, ZCoeffIm;
    long ZCoeffExp;

    double CCoeffRe, CCoeffIm;
    long CCoeffExp;

    double LAThresholdMant;
    long LAThresholdExp;

    double LAThresholdCMant;
    long LAThresholdCExp;

    double MinMagMant;
    long MinMagExp;

public:
    LAInfoDeep() {

    }

    LAInfoDeep(HDRFloatComplex z) {
        RefRe = z.getMantissaReal();
        RefIm = z.getMantissaImag();
        RefExp = z.getExp();


        HDRFloatComplex ZCoeff = new HDRFloatComplex(1.0, 0);
        HDRFloatComplex CCoeff = new HDRFloatComplex(1.0, 0);
        HDRFloat LAThreshold = HDRFloat.ONE;
        HDRFloat LAThresholdC = HDRFloat.ONE;

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

        if(LAInfo.DETECTION_METHOD == 1) {
            HDRFloat MinMag = HDRFloat.FOUR;
            MinMagMant = MinMag.getMantissa();
            MinMagExp = MinMag.getExp();
        }

    }

protected:
    bool DetectPeriod(HDRFloatComplex z) {
        if(LAInfo.DETECTION_METHOD == 1) {
            return z.chebychevNorm().compareToBothPositive(new HDRFloat(MinMagExp, MinMagMant).multiply(PeriodDetectionThreshold2)) < 0;
        }
        else {
            return z.chebychevNorm().divide(new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm).chebychevNorm()).multiply_mutable(LAThresholdScale).compareToBothPositive(new HDRFloat(LAThresholdExp, LAThresholdMant).multiply(PeriodDetectionThreshold)) < 0;
        }
    }

    
    GenericComplex getRef() {
        return new HDRFloatComplex(RefExp, RefRe, RefIm);
    }

    
    GenericComplex getZCoeff() {
        return new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
    }

    
    GenericComplex getCCoeff() {
        return new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
    }

    
    bool Step(LAInfoDeep out, HDRFloatComplex z) {
        HDRFloat ChebyMagz = z.chebychevNorm();

        HDRFloatComplex ZCoeff = new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
        HDRFloatComplex CCoeff = new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);

        HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
        HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

        if(LAInfo.DETECTION_METHOD == 1) {
            HDRFloat outMinMag = HDRFloat.minBothPositiveReduced(ChebyMagz, new HDRFloat(MinMagExp, MinMagMant));
            out.MinMagExp = outMinMag.getExp();
            out.MinMagMant = outMinMag.getMantissa();
        }

        HDRFloat temp1 = ChebyMagz.divide(ChebyMagZCoeff).multiply_mutable(LAThresholdScale);
        temp1.Reduce();

        HDRFloat temp2 = ChebyMagz.divide(ChebyMagCCoeff).multiply_mutable(LAThresholdCScale);
        temp2.Reduce();

        HDRFloat outLAThreshold = HDRFloat.minBothPositiveReduced(new HDRFloat(LAThresholdExp, LAThresholdMant), temp1);
        HDRFloat outLAThresholdC = HDRFloat.minBothPositiveReduced(new HDRFloat(LAThresholdCExp, LAThresholdCMant), temp2);

        out.LAThresholdExp = outLAThreshold.getExp();
        out.LAThresholdMant = outLAThreshold.getMantissa();

        out.LAThresholdCExp = outLAThresholdC.getExp();
        out.LAThresholdCMant = outLAThresholdC.getMantissa();

        HDRFloatComplex z2 = z.times2();
        HDRFloatComplex outZCoeff = z2.times(ZCoeff);
        outZCoeff.Reduce();
        HDRFloatComplex outCCoeff = z2.times(CCoeff).plus_mutable(HDRFloat.ONE);
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

        if(LAInfo.DETECTION_METHOD == 1) {
            return new HDRFloat(out.MinMagExp, out.MinMagMant).compareToBothPositive(new HDRFloat(MinMagExp, MinMagMant).multiply(Stage0PeriodDetectionThreshold2)) < 0;
        }
        else {
            return new HDRFloat(out.LAThresholdExp, out.LAThresholdMant).compareToBothPositive(new HDRFloat(LAThresholdExp, LAThresholdMant).multiply(Stage0PeriodDetectionThreshold)) < 0;
        }
    }

    
    bool Step(LAInfo out, Complex z) {
        return false;
    }

    
    bool isLAThresholdZero() {
        return new HDRFloat(LAThresholdExp, LAThresholdMant).compareTo(HDRFloat.ZERO) == 0;
    }

    
    bool isZCoeffZero() {
        HDRFloatComplex ZCoeff = new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
        return ZCoeff.getRe().compareTo(HDRFloat.ZERO) == 0 && ZCoeff.getIm().compareTo(HDRFloat.ZERO) == 0;
    }

    
    bool DetectPeriod(Complex z) {
        return false;
    }

    
    GenericLAInfo Step(HDRFloatComplex z)  {
        LAInfoDeep Result = new LAInfoDeep();

        Step(Result, z);
        return Result;
    }

    
    bool Composite(LAInfoDeep out, LAInfoDeep LA) {
        HDRFloatComplex z = new HDRFloatComplex(LA.RefExp, LA.RefRe, LA.RefIm);
        HDRFloat ChebyMagz = z.chebychevNorm();

        HDRFloatComplex ZCoeff = new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
        HDRFloatComplex CCoeff = new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
        HDRFloat LAThreshold = new HDRFloat(LAThresholdExp, LAThresholdMant);
        HDRFloat LAThresholdC = new HDRFloat(LAThresholdCExp, LAThresholdCMant);

        HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
        HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

        HDRFloat temp1 = ChebyMagz.divide(ChebyMagZCoeff).multiply_mutable(LAThresholdScale);
        temp1.Reduce();

        HDRFloat temp2 = ChebyMagz.divide(ChebyMagCCoeff).multiply_mutable(LAThresholdCScale);
        temp2.Reduce();

        HDRFloat outLAThreshold = HDRFloat.minBothPositiveReduced(LAThreshold, temp1);
        HDRFloat outLAThresholdC = HDRFloat.minBothPositiveReduced(LAThresholdC, temp2);

        HDRFloatComplex z2 = z.times2();
        HDRFloatComplex outZCoeff = z2.times(ZCoeff);
        outZCoeff.Reduce();
        //double RescaleFactor = out.LAThreshold / LAThreshold;
        HDRFloatComplex outCCoeff = z2.times(CCoeff);
        outCCoeff.Reduce();

        ChebyMagZCoeff = outZCoeff.chebychevNorm();
        ChebyMagCCoeff = outCCoeff.chebychevNorm();
        HDRFloat temp = outLAThreshold;

        HDRFloat LA_LAThreshold = new HDRFloat(LA.LAThresholdExp, LA.LAThresholdMant);
        HDRFloatComplex LAZCoeff = new HDRFloatComplex(LA.ZCoeffExp, LA.ZCoeffRe, LA.ZCoeffIm);
        HDRFloatComplex LACCoeff = new HDRFloatComplex(LA.CCoeffExp, LA.CCoeffRe, LA.CCoeffIm);

        temp1 = LA_LAThreshold.divide(ChebyMagZCoeff);
        temp1.Reduce();

        temp2 = LA_LAThreshold.divide(ChebyMagCCoeff);
        temp2.Reduce();

        outLAThreshold = HDRFloat.minBothPositiveReduced(outLAThreshold, temp1);
        outLAThresholdC = HDRFloat.minBothPositiveReduced(outLAThresholdC, temp2);
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

        if(LAInfo.DETECTION_METHOD == 1) {
            HDRFloat MinMag = new HDRFloat(MinMagExp, MinMagMant);
            temp = HDRFloat.minBothPositiveReduced(ChebyMagz, MinMag);
            HDRFloat outMinMag = HDRFloat.minBothPositiveReduced (temp, new HDRFloat(LA.MinMagExp, LA.MinMagMant));

            out.MinMagExp = outMinMag.getExp();
            out.MinMagMant = outMinMag.getMantissa();

            return temp.compareToBothPositive(MinMag.multiply(PeriodDetectionThreshold2)) < 0;
        }
        else {
            return temp.compareToBothPositive(LAThreshold.multiply(PeriodDetectionThreshold)) < 0;
        }
    }

    
    GenericLAInfo Composite(LAInfo LA) {
        return null;
    }

    
    LAInfoDeep Composite(LAInfoDeep LA)  {
        LAInfoDeep Result = new LAInfoDeep();

        Composite(Result, LA);
        return Result;
    }

    
    GenericLAInfo Step(Complex z) {
        return null;
    }

    
    bool Composite(LAInfo out, LAInfo LA) {
        return false;
    }

    LAstep Prepare(HDRFloatComplex dz, HDRFloatComplex dc)  {
        //*2 is + 1
        HDRFloatComplex newdz = dz.times(new HDRFloatComplex(RefExp + 1, RefRe, RefIm).plus_mutable(dz));
        newdz.Reduce();

        LAstep temp = new LAstep();
        temp.unusable = newdz.chebychevNorm().compareToBothPositiveReduced(new HDRFloat(LAThresholdExp, LAThresholdMant)) >= 0;
        temp.newDzDeep = newdz;
        return temp;
    }

    public:
    HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) {
        return newdz.times(new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm)).plus_mutable(dc.times(new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm)));
    }

    HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc)  {
        return  dzdc.times2().times_mutable(z).times_mutable(new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm) ).plus_mutable(new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm));
    }

    HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc)  {
        return  dzdc2.times(z)
                .plus_mutable(dzdc.square()).times2_mutable().times_mutable(new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm));
    }
    
    protected:
    ATInfo CreateAT(GenericLAInfo Next) {
        ATInfo Result = new ATInfo();

        HDRFloatComplex ZCoeff = new HDRFloatComplex(ZCoeffExp, ZCoeffRe, ZCoeffIm);
        HDRFloatComplex CCoeff = new HDRFloatComplex(CCoeffExp, CCoeffRe, CCoeffIm);
        HDRFloat LAThreshold = new HDRFloat(LAThresholdExp, LAThresholdMant);
        HDRFloat LAThresholdC = new HDRFloat(LAThresholdCExp, LAThresholdCMant);

        Result.ZCoeff = ZCoeff;
        Result.CCoeff = ZCoeff.times(CCoeff);
        Result.CCoeff.Reduce();

        Result.InvZCoeff = ZCoeff.reciprocal();
        Result.InvZCoeff.Reduce();

        Result.CCoeffSqrInvZCoeff = Result.CCoeff.square().times_mutable(Result.InvZCoeff);
        Result.CCoeffSqrInvZCoeff.Reduce();

        Result.CCoeffInvZCoeff = Result.CCoeff.times(Result.InvZCoeff);
        Result.CCoeffInvZCoeff.Reduce();

        Result.RefC = Next.getRef().toMantExpComplex().times(ZCoeff);
        Result.RefC.Reduce();

        Result.CCoeffNormSqr = Result.CCoeff.norm_squared();
        Result.CCoeffNormSqr.Reduce();

        Result.RefCNormSqr = Result.RefC.norm_squared();
        Result.RefCNormSqr.Reduce();

        Result.SqrEscapeRadius = HDRFloat.minBothPositive(ZCoeff.norm_squared().multiply(LAThreshold), atLimit).toDouble();

        Result.ThresholdC = HDRFloat.minBothPositive(LAThresholdC, atLimit.divide(Result.CCoeff.chebychevNorm()));

        return Result;
    }

    public:
    HDRFloat getLAThreshold() {
        return new HDRFloat(LAThresholdExp, LAThresholdMant);
    }

    
    HDRFloat getLAThresholdC() {
        return new HDRFloat(LAThresholdCExp, LAThresholdCMant);
    }
};
