#pragma once

#include <cstdint>

namespace FractalShark {

enum class RefOrbitPrecisionFailure { None, AmbiguousSample, UnsupportedConversion, InvalidBound };

// Process-wide counts of successful dynamic attempts and certificate failures.
// Successful startup exits without steps and cancelled attempts are excluded.
struct RefOrbitDynamicPrecisionCounters {
    uint64_t Successes{};
    uint64_t Failures{};
};

// Per-attempt diagnostics are deliberately independent of the saved orbit format.
// Failed-attempt statistics remain available after the exception; their generation
// identifies the discarded attempt, never a previously cached valid orbit.
struct RefOrbitDynamicPrecisionStats {
    uint64_t GenerationNumber{};
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
