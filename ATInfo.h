#pragma once

#include <complex>

#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "ATResult.h"

//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.HDRFloat;
//import fractalzoomer.core.HDRFloatComplex;

class ATInfo {
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    static HDRFloat factor;

public:
    ATInfo();

    size_t StepLength;
    HDRFloat ThresholdC;
    double SqrEscapeRadius;
    HDRFloatComplex RefC;
    HDRFloatComplex ZCoeff, CCoeff, InvZCoeff;
    HDRFloatComplex CCoeffSqrInvZCoeff;
    HDRFloatComplex CCoeffInvZCoeff;

    HDRFloat CCoeffNormSqr;
    HDRFloat RefCNormSqr;

    using Complex = std::complex<float>;

    bool Usable(HDRFloat SqrRadius) {
        auto result = CCoeffNormSqr * SqrRadius * factor;
        return result.compareToBothPositive(RefCNormSqr) > 0 && SqrEscapeRadius > 4.0;
    }

public:
    bool isValid(HDRFloatComplex DeltaSub0);
    HDRFloatComplex getDZ(Complex z);
    HDRFloatComplex getDZDC(Complex dzdc);
    HDRFloatComplex getDZDC2(Complex dzdc2);
    Complex getC(HDRFloatComplex dc);
    ATResult PerformAT(size_t max_iterations, HDRFloatComplex DeltaSub0, size_t derivatives);
};
