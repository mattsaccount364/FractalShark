#pragma once

// Lifted from Imagina

#include <complex>
#include <stdint.h>

#include "HDRFloat.h"

namespace Imagina {
    using HRReal = HDRFloat<double, HDROrder::Left, int64_t>;
    using SRReal = double;

    constexpr uint64_t IMMagicNumber = 0x000A0D56504D49FF;

    // Encode ASCII "Sharks:)" into a 64-bit number, for
    // a slight derivative of the format.
    constexpr uint64_t SharksMagicNumber = 0x536861726b733a29;
    

    struct IMFileHeader {
        uint64_t Magic;
        uint64_t Reserved;
        uint64_t LocationOffset;
        uint64_t ReferenceOffset;
    };

    struct ReferenceHeader {
        bool ExtendedRange;
    };

    struct ReferenceTrivialContent {
        HRReal AbsolutePrecision;
        HRReal RelativePrecision;
        HRReal ValidRadius;
    };

    struct ImaginaATInfo {
        size_t StepLength;
        HRReal ThresholdC;
        SRReal SqrEscapeRadius;
        std::complex<SRReal> RefC;
        std::complex<HRReal> ZCoeff, CCoeff, InvZCoeff;
    };

    struct LAReferenceTrivialContent {
        std::complex<SRReal> Refc;
        size_t RefIt;
        size_t MaxIt;
        bool DoublePrecisionPT;
        bool DirectEvaluate;
        bool IsPeriodic;
        bool UseAT;

        ImaginaATInfo AT;

        size_t LAStageCount;
    };
} // namespace Imagina