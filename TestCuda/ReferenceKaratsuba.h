#pragma once

#include <stdint.h>

template<class SharkFloatParams>
struct HpSharkFloat;

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV1(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/, // Unused
    uint64_t * /*carryOuts_phase6*/, // Unused
    uint64_t * /*carryIns*/,         // Unused
    uint64_t * /*tempProducts*/      // Unused
);

template<class SharkFloatParams>
void MultiplyHelperKaratsubaV2(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t * /*carryOuts_phase3*/, // Unused
    uint64_t * /*carryOuts_phase6*/, // Unused
    uint64_t * /*carryIns*/,         // Unused
    uint64_t * /*tempProducts*/      // Unused
);