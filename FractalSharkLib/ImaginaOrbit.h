#pragma once

// Lifted from Imagina

#include <complex>
#include <stdint.h>

#include "HDRFloat.h"

using ImaginaHRReal = HDRFloat<double, HDROrder::Left, int64_t>;
using ImaginaSRReal = double;

struct ReferenceHeader {
    bool ExtendedRange;
};

struct ReferenceTrivialContent {
    ImaginaHRReal AbsolutePrecision;
    ImaginaHRReal RelativePrecision;
    ImaginaHRReal ValidRadius;
};

struct ImaginaATInfo {
    size_t StepLength;
    ImaginaHRReal ThresholdC;
    ImaginaSRReal SqrEscapeRadius;
    std::complex<ImaginaSRReal> RefC;
    std::complex<ImaginaHRReal> ZCoeff, CCoeff, InvZCoeff;
};

struct LAReferenceTrivialContent {
    std::complex<ImaginaSRReal> Refc;
    size_t RefIt;
    size_t MaxIt;
    bool DoublePrecisionPT;
    bool DirectEvaluate;
    bool IsPeriodic;
    bool UseAT;

    ImaginaATInfo AT;

    size_t LAStageCount;
};
