#include <StdAfx.h>
#include "HDRFloat.h"

double twoPowExpData[2048];
double* twoPowExp;

void InitStatics()
{
    //LN2 = ::log(2);
    //LN2_REC = 1.0 / LN2;

    twoPowExp = twoPowExpData;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    //twoPowExp.resize(MaxDoubleExponent - MinDoubleExponent + 1);
    for (int i = MinDoubleExponent; i <= MaxDoubleExponent; i++) {
        double d = scalbn(1.0, i);
        int index = i - MinDoubleExponent;
        twoPowExp[index] = d;
    }
}