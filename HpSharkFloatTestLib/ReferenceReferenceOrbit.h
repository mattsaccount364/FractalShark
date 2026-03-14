#pragma once

#include <cstdint>
#include <memory>
#include <vector>

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

// Forward declaration — defined in HpSharkFloat.h
enum class PeriodicityResult;

// Result of a Newton-Raphson period-refinement step.
template <class SharkFloatParams> struct NewtonRaphsonResult {
    HpSharkFloat<SharkFloatParams> RefinedCReal;
    HpSharkFloat<SharkFloatParams> RefinedCImag;
    uint32_t NewtonIterations;
    bool Converged;
};

// Result of a CPU reference orbit computation.
template <class SharkFloatParams> struct ReferenceOrbitResult {
    std::vector<typename SharkFloatParams::ReferenceIterT> Orbit;
    uint64_t IterationsExecuted;
    PeriodicityResult PeriodResult;
    HpSharkFloat<SharkFloatParams> FinalZReal;
    HpSharkFloat<SharkFloatParams> FinalZImag;
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
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>
ReferenceOrbitHelper(const HpSharkFloat<SharkFloatParams> *cReal,
                     const HpSharkFloat<SharkFloatParams> *cImag,
                     const typename SharkFloatParams::Float &radiusY,
                     uint64_t maxIters,
                     DebugHostCombo<SharkFloatParams> &debugHostCombo);

// Newton-Raphson inner loop: iterates z = z^2 + c for `period`
// steps, tracking both z_p and dz/dc_p using HpSharkFloat arithmetic.
// Also accumulates d2 (second derivative) in SharkFloatParams::Float
// (HDRFloat), matching production EvaluateCriticalOrbitAndDerivsST.
template <class SharkFloatParams>
void EvaluateOrbitAndDerivative(
    const HpSharkFloat<SharkFloatParams> *cReal,
    const HpSharkFloat<SharkFloatParams> *cImag,
    uint64_t period,
    HpSharkFloat<SharkFloatParams> *outZReal,
    HpSharkFloat<SharkFloatParams> *outZImag,
    HpSharkFloat<SharkFloatParams> *outDzdcReal,
    HpSharkFloat<SharkFloatParams> *outDzdcImag,
    typename SharkFloatParams::Float *outD2Real,
    typename SharkFloatParams::Float *outD2Imag,
    DebugHostCombo<SharkFloatParams> &debugHostCombo);
