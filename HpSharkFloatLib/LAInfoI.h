#pragma once

#include "HDRFloat.h"

template <typename IterType> class LAStageInfo {
public:
    IterType LAIndex{};
    IterType MacroItCount{};
};

template <typename IterType> class LAInfoI {
public:
    IterType StepLength{};
    IterType NextStageLAIndex{};
};
