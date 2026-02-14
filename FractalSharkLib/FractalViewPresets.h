#pragma once

#include "HighPrecision.h"
#include "ItersMemoryContainer.h"
#include "LAParameters.h"

#include <string>

struct ViewPresetResult {
    HighPrecision minX;
    HighPrecision minY;
    HighPrecision maxX;
    HighPrecision maxY;

    IterTypeFull numIterations;
    uint32_t gpuAntialiasing;
    int32_t compressionErrorExpLow;
    int32_t compressionErrorExpIntermediate;
    IterTypeEnum iterType;
    bool setLADefaultsMaxPerf;
    std::wstring warningMessage;
};

ViewPresetResult GetViewPreset(
    size_t view,
    IterTypeFull defaultIterations,
    int32_t defaultCompressionExpLow,
    int32_t defaultCompressionExpIntermediate);
