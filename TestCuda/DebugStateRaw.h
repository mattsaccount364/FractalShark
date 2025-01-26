#pragma once

#include <stdint.h>

enum class DebugStatePurpose {
    Invalid,
    ADigits,
    BDigits,
    AHalfHigh,
    AHalfLow,
    BHalfHigh,
    BHalfLow,
    XDiff,
    YDiff,
    Z0,
    Z1,
    Z2,
    Z1_offset,
    Final128,
    Convolution_offset,
    Result_offset,
    XDiff_offset,
    YDiff_offset,
    GlobalCarryOffset,
    SubtractionOffset1,
    SubractionOffset2,
    SubtractionOffset3,
    SubtractionOffset4,
    BorrowAnyOffset,
    NumPurposes
};

struct DebugStateRaw {
    int Block;
    int Thread;
    uint64_t ArraySize;
    uint64_t Checksum;
    DebugStatePurpose ChecksumPurpose;
    int CallIndex;
};
