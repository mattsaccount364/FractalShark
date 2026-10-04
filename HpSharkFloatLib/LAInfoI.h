#pragma once

#include "HDRFloat.h"

template <typename IterType, class Float> class LAStageInfo {
public:
    IterType LAIndex{};
    IterType MacroItCount{};
    Float LAThresholdC{};

    template <class OtherFloat>
    CUDA_CRAP LAStageInfo &
    operator=(const LAStageInfo<IterType, OtherFloat> &other)
    {
        LAIndex = other.LAIndex;
        MacroItCount = other.MacroItCount;
        LAThresholdC = static_cast<Float>(other.LAThresholdC);
        return *this;
    }
};

template <typename IterType> class LAInfoI {
public:
    IterType StepLength{};
    IterType NextStageLAIndex{};
};
