#pragma once

#include "HighPrecision.h"

struct LAParameters {
    CUDA_CRAP int GetDefaultDetectionMethod() const;
    CUDA_CRAP float GetDefaultLAThresholdScale() const;
    CUDA_CRAP float GetDefaultLAThresholdCScale() const;
    CUDA_CRAP float GetDefaultStage0PeriodDetectionThreshold2() const;
    CUDA_CRAP float GetDefaultPeriodDetectionThreshold2() const;
    CUDA_CRAP float GetDefaultStage0PeriodDetectionThreshold() const;
    CUDA_CRAP float GetDefaultPeriodDetectionThreshold() const;
};

