#pragma once

#include <cstdint>
#include <memory>
#include <vector>

template <class SharkFloatParams> struct DebugHostCombo;
template <class SharkFloatParams> struct HpSharkFloat;

enum class PeriodicityResult;

template <class SharkFloatParams> struct NewtonRaphsonResult {
    HpSharkFloat<SharkFloatParams> RefinedCReal;
    HpSharkFloat<SharkFloatParams> RefinedCImag;
    uint32_t NewtonIterations;
    bool Converged;
};

template <class SharkFloatParams> struct ReferenceOrbitResult {
    std::vector<typename SharkFloatParams::ReferenceIterT> Orbit;
    uint64_t IterationsExecuted;
    PeriodicityResult PeriodResult;
    HpSharkFloat<SharkFloatParams> FinalZReal;
    HpSharkFloat<SharkFloatParams> FinalZImag;
};
