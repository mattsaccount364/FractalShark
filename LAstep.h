#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.MantExpComplex;

#include "LAInfoDeep.h"

class LAstep {
public:
    int step;
    int nextStageLAindex;

protected:
    LAInfoDeep LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0) {
        return LAjdeep.Evaluate(newDzDeep, DeltaSub0);
    }

    HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc) {
        return LAj.EvaluateDzdc(z, dzdc);
    }

    HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
        return LAj.EvaluateDzdc2(z, dzdc2, dzdc);
    }

    HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc) {
        return LAjdeep.EvaluateDzdc(z, dzdc);
    }

    HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc) {
        return LAjdeep.EvaluateDzdc2(z, dzdc2, dzdc);
    }

    HDRFloatComplex getZ(HDRFloatComplex DeltaSubN) { return Refp1Deep.plus(DeltaSubN); }

};
