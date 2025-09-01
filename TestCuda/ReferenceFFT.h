#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams>
struct DebugHostCombo;

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
void MultiplyHelperFFT(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXX,
    HpSharkFloat<SharkFloatParams> *OutXY,
    HpSharkFloat<SharkFloatParams> *OutYY,
    DebugHostCombo<SharkFloatParams> &debugCombo
);

template<class SharkFloatParams>
void MultiplyHelperFFT2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXX,
    HpSharkFloat<SharkFloatParams> *OutXY,
    HpSharkFloat<SharkFloatParams> *OutYY,
    DebugHostCombo<SharkFloatParams> &debugCombo
);