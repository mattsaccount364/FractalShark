#pragma once

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAStep.h"
#include "ATInfo.h"
//#include "LAInfo.h"

//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.HDRFloat;
//import fractalzoomer.core.HDRFloatComplex;

class LAInfoDeep {
public:
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;
    using T = float;

    //public static int DETECTION_METHOD = ApproximationDefaultSettings.DETECTION_METHOD;
    //public static double Stage0PeriodDetectionThreshold = ApproximationDefaultSettings.Stage0PeriodDetectionThreshold;
    //public static double PeriodDetectionThreshold = ApproximationDefaultSettings.PeriodDetectionThreshold;
    //public static double Stage0PeriodDetectionThreshold2 = ApproximationDefaultSettings.Stage0PeriodDetectionThreshold2;
    //public static double PeriodDetectionThreshold2 = ApproximationDefaultSettings.PeriodDetectionThreshold2;
    //public static double LAThresholdScale = ApproximationDefaultSettings.LAThresholdScale;
    //public static double LAThresholdCScale = ApproximationDefaultSettings.LAThresholdCScale;

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

    static HDRFloat Stage0PeriodDetectionThreshold;
    static HDRFloat PeriodDetectionThreshold;
    static HDRFloat Stage0PeriodDetectionThreshold2;
    static HDRFloat PeriodDetectionThreshold2;
    static HDRFloat LAThresholdScale;
    static HDRFloat LAThresholdCScale;

private:
    static HDRFloat atLimit;

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
    LAInfoDeep();
    LAInfoDeep(HDRFloatComplex z);
    bool DetectPeriod(HDRFloatComplex z);
    HDRFloatComplex getRef();
    HDRFloatComplex getZCoeff();
    HDRFloatComplex getCCoeff();
    bool Step(LAInfoDeep out, HDRFloatComplex z);
    bool isLAThresholdZero();
    bool isZCoeffZero();
    LAInfoDeep Step(HDRFloatComplex z);
    bool Composite(LAInfoDeep out, LAInfoDeep LA);
    LAInfoDeep Composite(LAInfoDeep LA);
    LAstep Prepare(HDRFloatComplex dz);
    HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc);
    HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    ATInfo CreateAT(LAInfoDeep Next);
    HDRFloat getLAThreshold();
    HDRFloat getLAThresholdC();
};
