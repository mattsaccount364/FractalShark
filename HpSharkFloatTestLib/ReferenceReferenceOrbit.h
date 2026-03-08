#pragma once

#include <cstdint>
#include <vector>

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

// Forward declaration — defined in HpSharkFloat.h
enum class PeriodicityResult;

// Result of a CPU reference orbit computation.
template <class SharkFloatParams> struct ReferenceOrbitResult {
    std::vector<typename SharkFloatParams::ReferenceIterT> Orbit;
    uint64_t IterationsExecuted;
    PeriodicityResult PeriodResult;
};

// Computes a full Mandelbrot reference orbit on the CPU using
// HpSharkFloat-based arithmetic (MultiplyHelperFFT2 + AddHelper).
// Mirrors the GPU HpSharkReferenceGpuLoop kernel for validation.
//
// z_(n+1) = z_n^2 + c
// where c = (cReal, cImag), z_0 = (cReal, cImag) [same as GPU init]
//
// Periodicity detection and escape checking are included when
// SharkFloatParams::EnablePeriodicity is true.
template <class SharkFloatParams>
ReferenceOrbitResult<SharkFloatParams>
ReferenceOrbitHelper(const HpSharkFloat<SharkFloatParams> *cReal,
                     const HpSharkFloat<SharkFloatParams> *cImag,
                     const typename SharkFloatParams::Float &radiusY,
                     uint64_t maxIters,
                     DebugHostCombo<SharkFloatParams> &debugHostCombo);
