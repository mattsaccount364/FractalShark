#pragma once

enum class RenderAlgorithm;
enum class IterTypeEnum;

#include "HighPrecision.h"

struct RecommendedSettings {
    RecommendedSettings();
    RecommendedSettings(IterTypeFull iterationLimit);
    RecommendedSettings(RenderAlgorithm renderAlg, IterTypeEnum iterType, IterTypeFull numIterations);

    RenderAlgorithm RenderAlg;
    IterTypeEnum IterType;
    IterTypeFull NumIterations;
};