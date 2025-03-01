#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams>
struct DebugStateHost;

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXX,
    HpSharkFloat<SharkFloatParams> *OutXY,
    HpSharkFloat<SharkFloatParams> *OutYY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
);