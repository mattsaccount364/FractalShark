#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams>
struct DebugStateHost;

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
void AddHelper(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
);