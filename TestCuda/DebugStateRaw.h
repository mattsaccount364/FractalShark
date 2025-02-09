#pragma once

#include <stdint.h>

enum class UseConvolution {
    No,
    Yes
};

enum class RecordIt {
    No,
    Yes
};

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
    Result_offset,
    NumPurposes
};

struct DebugStateRaw {
    int Block;
    int Thread;
    uint64_t ArraySize;
    uint64_t Checksum;
    DebugStatePurpose ChecksumPurpose;
    int RecursionDepth;
    int CallIndex;
    UseConvolution Convolution;
};
