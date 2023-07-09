#include <StdAfx.h>
#include "HDRFloat.h"

double twoPowExpDataDbl[2048];
float twoPowExpDataFlt[256];
double* twoPowExpDbl;
float* twoPowExpFlt;

void InitStatics()
{
    //LN2 = ::log(2);
    //LN2_REC = 1.0 / LN2;

    twoPowExpDbl = twoPowExpDataDbl;
    twoPowExpFlt = twoPowExpDataFlt;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    static constexpr int MaxFloatExponent = 127;
    static constexpr int MinFloatExponent = -126;

    //twoPowExp.resize(MaxDoubleExponent - MinDoubleExponent + 1);
    for (int i = MinDoubleExponent; i <= MaxDoubleExponent; i++) {
        double d = scalbn(1.0, i);
        int index = i - MinDoubleExponent;
        twoPowExpDbl[index] = d;
    }

    for (int i = MinFloatExponent; i <= MaxFloatExponent; i++) {
        float f = scalbn(1.0f, i);
        int index = i - MinFloatExponent;
        twoPowExpFlt[index] = f;
    }
}