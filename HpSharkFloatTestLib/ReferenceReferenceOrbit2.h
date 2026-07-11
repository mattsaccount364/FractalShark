#pragma once

#include "ReferenceReferenceOrbit.h"

// Test-only second reference-orbit implementation.  This exercises the
// HpSharkFloat CPU NTT multiply and add-resolution path directly; it
// intentionally has no GPU entry point.
template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>> ReferenceOrbit2Helper(
    const HpSharkFloat<SharkFloatParams> *cReal,
    const HpSharkFloat<SharkFloatParams> *cImag,
    const typename SharkFloatParams::Float &radiusY,
    uint64_t maxIters,
    DebugHostCombo<SharkFloatParams> &debugHostCombo);

template <class SharkFloatParams>
void EvaluateOrbitAndDerivative2(const HpSharkFloat<SharkFloatParams> *cReal,
                                 const HpSharkFloat<SharkFloatParams> *cImag,
                                 uint64_t period,
                                 HpSharkFloat<SharkFloatParams> *outZReal,
                                 HpSharkFloat<SharkFloatParams> *outZImag,
                                 HpSharkFloat<SharkFloatParams> *outDzdcReal,
                                 HpSharkFloat<SharkFloatParams> *outDzdcImag,
                                 typename SharkFloatParams::Float *outD2Real,
                                 typename SharkFloatParams::Float *outD2Imag,
                                 DebugHostCombo<SharkFloatParams> &debugHostCombo);
