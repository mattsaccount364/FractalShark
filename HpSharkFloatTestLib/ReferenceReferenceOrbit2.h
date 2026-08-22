#pragma once

#include "ReferenceOrbitResult.h"

namespace HpShark {
template <class SharkFloatParams> class ReferencePreparedTables;
}

// Test-only CPU oracle for the fused GPU reference-orbit implementation.
template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>> ReferenceOrbit2Helper(
    const HpSharkFloat<SharkFloatParams> *cReal,
    const HpSharkFloat<SharkFloatParams> *cImag,
    const typename SharkFloatParams::Float &radiusY,
    uint64_t maxIters,
    uint32_t actualPrecisionLimbs,
    DebugHostCombo<SharkFloatParams> &debugHostCombo,
    HpShark::ReferencePreparedTables<SharkFloatParams> *preparedTables);

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
                                 uint32_t actualPrecisionLimbs,
                                 DebugHostCombo<SharkFloatParams> &debugHostCombo,
                                 HpShark::ReferencePreparedTables<SharkFloatParams> *preparedTables);
