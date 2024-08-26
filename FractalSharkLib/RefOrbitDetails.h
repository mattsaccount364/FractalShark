#pragma once

#include "HighPrecision.h"

struct RefOrbitDetails {
    uint64_t InternalPeriodMaybeZero;
    uint64_t CompressedIters;
    uint64_t UncompressedIters;
    uint64_t CompressedIntermediateIters;
    int32_t CompressionErrorExp;
    int32_t IntermediateCompressionErrorExp;
    int64_t DeltaIntermediatePrecision;
    int64_t ExtraIntermediatePrecision;
    uint64_t OrbitMilliseconds;
    uint64_t LAMilliseconds;
    uint64_t LASize;
    std::string PerturbationAlg;
    HighPrecision ZoomFactor;

    RefOrbitDetails() : InternalPeriodMaybeZero{},
        CompressedIters{},
        UncompressedIters{},
        CompressedIntermediateIters{},
        CompressionErrorExp{},
        IntermediateCompressionErrorExp{},
        DeltaIntermediatePrecision{},
        ExtraIntermediatePrecision{},
        OrbitMilliseconds{},
        LAMilliseconds{},
        LASize{},
        PerturbationAlg{},
        ZoomFactor{} {
    }

    RefOrbitDetails(
        uint64_t InternalPeriodMaybeZero,
        uint64_t CompressedIters,
        uint64_t UncompressedIters,
        uint64_t CompressedIntermediateIters,
        int32_t CompressionErrorExp,
        int32_t IntermediateCompressionErrorExp,
        int64_t DeltaIntermediatePrecision,
        int64_t ExtraIntermediatePrecision,
        uint64_t OrbitMilliseconds,
        uint64_t LAMilliseconds,
        uint64_t LASize,
        std::string PerturbationAlg,
        HighPrecision ZoomFactor) :
        InternalPeriodMaybeZero{ InternalPeriodMaybeZero },
        CompressedIters{ CompressedIters },
        UncompressedIters{ UncompressedIters },
        CompressedIntermediateIters{ CompressedIntermediateIters },
        CompressionErrorExp{ CompressionErrorExp },
        IntermediateCompressionErrorExp{ IntermediateCompressionErrorExp },
        DeltaIntermediatePrecision{ DeltaIntermediatePrecision },
        ExtraIntermediatePrecision{ ExtraIntermediatePrecision },
        OrbitMilliseconds{ OrbitMilliseconds },
        LAMilliseconds{ LAMilliseconds },
        LASize{ LASize },
        PerturbationAlg{ PerturbationAlg },
        ZoomFactor{ ZoomFactor } {
    }
};