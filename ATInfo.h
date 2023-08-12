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

protected:  
    int StepLength;
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
       return CCoeffNormSqr.multiply(SqrRadius).multiply_mutable(factor)
                .compareToBothPositive(RefCNormSqr) > 0 && SqrEscapeRadius > 4.0;
    }

public:

    bool isValid(HDRFloatComplex DeltaSub0) {
        return DeltaSub0.chebychevNorm().compareToBothPositiveReduced(ThresholdC) <= 0;
    }

    HDRFloatComplex getDZ(Complex z) {
        HDRFloatComplex temp = HDRFloatComplex(z).times(InvZCoeff);
        temp.Reduce();
        return temp;
    }

    HDRFloatComplex getDZDC(Complex dzdc) {
        HDRFloatComplex temp = HDRFloatComplex(dzdc).times(CCoeffInvZCoeff);
        temp.Reduce();
        return temp;
    }

    HDRFloatComplex getDZDC2(Complex dzdc2) {
        HDRFloatComplex temp = HDRFloatComplex(dzdc2).times(CCoeffSqrInvZCoeff);
        temp.Reduce();
        return temp;
    }

    Complex getC(HDRFloatComplex dc) {
        HDRFloatComplex temp = dc.times(CCoeff).plus_mutable(RefC);
        temp.Reduce();
        return temp.toComplex();
    }

    ATResult PerformAT(int max_iterations, HDRFloatComplex DeltaSub0, int derivatives) {
        //int ATMaxIt = (max_iterations - 1) / StepLength + 1;
        int ATMaxIt = max_iterations / StepLength;

        Complex c;
        c = getC((HDRFloatComplex) DeltaSub0);

        Complex z = Complex();
        Complex dzdc = Complex();
        Complex dzdc2 = Complex();

        int i;
        for(i = 0; i < ATMaxIt; i++) {

            if(std::norm(z) > SqrEscapeRadius) {
                break;
            }

            if(derivatives > 1) {
                //dzdc2 = dzdc2.times(z).plus_mutable(dzdc.square()).times2_mutable();
                dzdc2 = dzdc2 * z + dzdc * dzdc * 2.0f;
            }
            if(derivatives > 0) {
                //dzdc = dzdc.times2().times_mutable(z).plus_mutable(1);
                dzdc = dzdc * 2.0f * z + 1.0f;
            }

            z = z * z + c;
        }

        ATResult res = ATResult();
        res.dz = getDZ(z);

        if(derivatives > 1) {
            res.dzdc2 = getDZDC2(dzdc2);
        }

        if(derivatives > 0) {
            res.dzdc = getDZDC(dzdc);
        }

        res.bla_iterations = i * StepLength;
        res.bla_steps = i;

        return res;
    }
};
