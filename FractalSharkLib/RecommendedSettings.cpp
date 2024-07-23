#include "stdafx.h"
#include "RecommendedSettings.h"
#include "Fractal.h"

RecommendedSettings::RecommendedSettings()
    : RenderAlg{},
    IterType{},
    NumIterations{} {
}

RecommendedSettings::RecommendedSettings(IterTypeFull iterationLimit)
    : RecommendedSettings{} {

    if (iterationLimit <= Fractal::GetMaxIterations<uint32_t>()) {
        IterType = IterTypeEnum::Bits32;
    } else {
        IterType = IterTypeEnum::Bits64;
    }

    NumIterations = iterationLimit;
}

RecommendedSettings::RecommendedSettings(
    RenderAlgorithm renderAlg,
    IterTypeEnum iterType,
    IterTypeFull numIterations) :
    RenderAlg(renderAlg),
    IterType(iterType),
    NumIterations(numIterations) {
    }