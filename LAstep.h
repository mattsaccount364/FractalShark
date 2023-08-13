#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.Complex;
//import fractalzoomer.core.MantExpComplex;

#include "HDRFloatComplex.h"
class LAInfoDeep;

class LAstep {
public:
    int step;
    int nextStageLAindex;

public:
    using HDRFloatComplex = HDRFloatComplex<float>;
    LAInfoDeep *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;

public:
    bool unusable;

    HDRFloatComplex Evaluate(HDRFloatComplex DeltaSub0);
    HDRFloatComplex EvaluateDzdcDeep(HDRFloatComplex z, HDRFloatComplex dzdc);
    HDRFloatComplex EvaluateDzdc2Deep(HDRFloatComplex z, HDRFloatComplex dzdc2, HDRFloatComplex dzdc);
    HDRFloatComplex getZ(HDRFloatComplex DeltaSubN);
};
