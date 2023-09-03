#pragma once

#include "HDRFloat.h"

class LAStageInfo {
public:
    size_t LAIndex;
    size_t MacroItCount;
};

class  LAInfoI {
public:
    size_t StepLength;
    size_t NextStageLAIndex;

    CUDA_CRAP LAInfoI() {
        StepLength = 0;
        NextStageLAIndex = 0;
    }

    CUDA_CRAP LAInfoI(const LAInfoI& other) {
        StepLength = other.StepLength;
        NextStageLAIndex = other.NextStageLAIndex;
    }
};
