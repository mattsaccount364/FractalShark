#pragma once

#include "HDRFloat.h"

class LAStageInfo {
public:
    IterType LAIndex;
    IterType MacroItCount;
};

class  LAInfoI {
public:
    IterType StepLength;
    IterType NextStageLAIndex;

    CUDA_CRAP LAInfoI() {
        StepLength = 0;
        NextStageLAIndex = 0;
    }

    CUDA_CRAP LAInfoI(const LAInfoI& other) {
        StepLength = other.StepLength;
        NextStageLAIndex = other.NextStageLAIndex;
    }
};
