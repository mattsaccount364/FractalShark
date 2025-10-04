#pragma once

#include "HighPrecision.h"
#include "HDRFloat.h"

__device__ __constant__ double twoPowExpDataDbl[2048];
__device__ __constant__ float twoPowExpDataFlt[256];

__global__
void GlobalInitStaticsKernel() {
    if (blockIdx.x == 0 &&
        threadIdx.x == 0 &&
        blockIdx.y == 0 &&
        threadIdx.y == 0) {

        // LN2 = ::log(2);
        // LN2_REC = 1.0 / LN2;

        twoPowExpDbl = twoPowExpDataDbl;
        twoPowExpFlt = twoPowExpDataFlt;

        static constexpr int MaxDoubleExponent = 1023;
        static constexpr int MinDoubleExponent = -1022;

        static constexpr int MaxFloatExponent = 127;
        static constexpr int MinFloatExponent = -126;

        // twoPowExp.resize(MaxDoubleExponent - MinDoubleExponent + 1);
        for (int i = MinDoubleExponent; i <= MaxDoubleExponent; i++) {
            double d = scalbn(1.0, i);
            int index = i - MinDoubleExponent;
            twoPowExpDbl[index] = d;
        }

        for (int i = MinFloatExponent; i <= MaxFloatExponent; i++) {
            float f = scalbn(1.0, i);
            int index = i - MinFloatExponent;
            twoPowExpFlt[index] = f;
        }
    }
}

void
InitHdrStaticsOnGpu()
{
    // 1 thread, 1 block
    GlobalInitStaticsKernel<<<1, 1>>>();
}
