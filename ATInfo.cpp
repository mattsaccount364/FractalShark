#include "StdAfx.h"

#include "ATInfo.h"
#include "ATResult.h"

HDRFloat<float> ATInfo::factor(0x1.0p32);

bool ATInfo::isValid(HDRFloatComplex DeltaSub0) {
    return DeltaSub0.chebychevNorm().compareToBothPositiveReduced(ThresholdC) <= 0;
}

ATInfo::HDRFloatComplex ATInfo::getDZ(Complex z) {
    HDRFloatComplex temp = HDRFloatComplex(z).times(InvZCoeff);
    temp.Reduce();
    return temp;
}

ATInfo::HDRFloatComplex ATInfo::getDZDC(Complex dzdc) {
    HDRFloatComplex temp = HDRFloatComplex(dzdc).times(CCoeffInvZCoeff);
    temp.Reduce();
    return temp;
}

ATInfo::HDRFloatComplex ATInfo::getDZDC2(Complex dzdc2) {
    HDRFloatComplex temp = HDRFloatComplex(dzdc2).times(CCoeffSqrInvZCoeff);
    temp.Reduce();
    return temp;
}

ATInfo::Complex ATInfo::getC(HDRFloatComplex dc) {
    HDRFloatComplex temp = dc.times(CCoeff).plus_mutable(RefC);
    temp.Reduce();
    return temp.toComplex();
}

ATResult ATInfo::PerformAT(int max_iterations, HDRFloatComplex DeltaSub0, int derivatives) {
    //int ATMaxIt = (max_iterations - 1) / StepLength + 1;
    int ATMaxIt = max_iterations / StepLength;

    Complex c;
    c = getC((HDRFloatComplex)DeltaSub0);

    Complex z = Complex();
    Complex dzdc = Complex();
    Complex dzdc2 = Complex();

    int i;
    for (i = 0; i < ATMaxIt; i++) {

        if (std::norm(z) > SqrEscapeRadius) {
            break;
        }

        if (derivatives > 1) {
            //dzdc2 = dzdc2.times(z).plus_mutable(dzdc.square()).times2_mutable();
            dzdc2 = dzdc2 * z + dzdc * dzdc * 2.0f;
        }
        if (derivatives > 0) {
            //dzdc = dzdc.times2().times_mutable(z).plus_mutable(1);
            dzdc = dzdc * 2.0f * z + 1.0f;
        }

        z = z * z + c;
    }

    ATResult res = ATResult();
    res.dz = getDZ(z);

    if (derivatives > 1) {
        res.dzdc2 = getDZDC2(dzdc2);
    }

    if (derivatives > 0) {
        res.dzdc = getDZDC(dzdc);
    }

    res.bla_iterations = i * StepLength;
    res.bla_steps = i;

    return res;
}