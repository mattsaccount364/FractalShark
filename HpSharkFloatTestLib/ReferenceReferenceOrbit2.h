#pragma once

#include "ReferenceReferenceOrbit.h"

namespace HpShark {
template <class SharkFloatParams> class Reference2PreparedTables;
}

// Test-only second reference-orbit implementation. This exercises the
// HpSharkFloat CPU arithmetic path while consuming tables prepared by the
// shared Ref2 CUDA setup kernel.
template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>> ReferenceOrbit2Helper(
    const HpSharkFloat<SharkFloatParams> *cReal,
    const HpSharkFloat<SharkFloatParams> *cImag,
    const typename SharkFloatParams::Float &radiusY,
    uint64_t maxIters,
    uint32_t actualPrecisionLimbs,
    DebugHostCombo<SharkFloatParams> &debugHostCombo,
    HpShark::Reference2PreparedTables<SharkFloatParams> *preparedTables = nullptr);

template <class SharkFloatParams>
void EvaluateOrbitAndDerivative2(
    const HpSharkFloat<SharkFloatParams> *cReal,
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
    HpShark::Reference2PreparedTables<SharkFloatParams> *preparedTables = nullptr);
