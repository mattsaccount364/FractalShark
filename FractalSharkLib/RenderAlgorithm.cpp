#include "stdafx.h"
#include "RenderAlgorithm.h"

RenderAlgorithm &RenderAlgorithm::operator=(const RenderAlgorithm &other) {
    if (this != &other) {
        this->Algorithm = other.Algorithm;
        this->AlgorithmStr = other.AlgorithmStr;
        this->UseLocalColor = other.UseLocalColor;
        this->RequiresCompression = other.RequiresCompression;
        this->RequiresReferencePoints = other.RequiresReferencePoints;
        this->TestInclude = other.TestInclude;
    }

    return *this;
}

RenderAlgorithm &RenderAlgorithm::operator=(RenderAlgorithm &&other) {
    return *this = other;
}

RenderAlgorithmEnum RenderAlgorithm::GetAlgorithmEnum(const char *algorithmStr) {
    for (size_t i = 0; i < (size_t)RenderAlgorithmEnum::MAX; i++) {
        if (algorithmStr == RenderAlgorithmStr[i]) {
            return (RenderAlgorithmEnum)i;
        }
    }
    return RenderAlgorithmEnum::AUTO;
}

