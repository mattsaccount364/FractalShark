#pragma once

#include "HDRFloat.h"

template<typename IterType>
class LAStageInfo {
public:
    IterType LAIndex;
    IterType MacroItCount;
};

template<typename IterType>
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
