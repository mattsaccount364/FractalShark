#pragma once

#include "HDRFloat.h"

class LAStageInfo {
public:
    int32_t LAIndex;
    int32_t MacroItCount;
};

class  LAInfoI {
public:
    int32_t StepLength;
    int32_t NextStageLAIndex;

    CUDA_CRAP LAInfoI() {
        StepLength = 0;
        NextStageLAIndex = 0;
    }

    CUDA_CRAP LAInfoI(const LAInfoI& other) {
        StepLength = other.StepLength;
        NextStageLAIndex = other.NextStageLAIndex;
    }
};
