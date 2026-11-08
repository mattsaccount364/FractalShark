#pragma once

#include <cstdint>

namespace FractalShark {

enum class RefOrbitPrecisionFailure { None, AmbiguousSample, UnsupportedConversion, InvalidBound };

// Counts classify fixed-MT9 equivalence certification, not rendering: an uncertain
// attempt counts once as a failure even though calculation continues. Only clean,
// noncancelled attempts with at least one step count as successes.
struct RefOrbitDynamicPrecisionCounters {
    uint64_t Successes{};
    uint64_t Failures{};
};

// Per-attempt diagnostics are deliberately independent of the saved orbit format.
// GenerationNumber identifies the dynamic attempt; PublishedGenerationNumber also
// follows conversions into another cached sample type. Failure records the first
// fixed-equivalence uncertainty; it does not imply calculation or publication failure.
struct RefOrbitDynamicPrecisionStats {
    uint64_t GenerationNumber{};
    uint64_t PublishedGenerationNumber{};
    uint64_t FailureIteration{};
    uint64_t FailureWorkingBits{};
    uint64_t FullBits{};
    uint64_t MinimumBits{};
    uint64_t Steps{};
    uint64_t ReducedSteps{};
    // Routes count real/imaginary component checks, not iterations or MPIR calls.
    // Structural real-axis skips and on-demand diagnostic replay do not count.
    uint64_t FastCertifiedComponents{};
    uint64_t RegularCertificationComponents{};
    double MeanBits{};
    uint64_t ElapsedMilliseconds{};
    RefOrbitPrecisionFailure Failure{RefOrbitPrecisionFailure::None};
};

} // namespace FractalShark
