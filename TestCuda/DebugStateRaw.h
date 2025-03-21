#pragma once

#include <stdint.h>
#include <array>

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
    Z0XX,
    Z0XY,
    Z0YY,
    Z1XX,
    Z1XY,
    Z1YY,
    Z2XX,
    Z2XY,
    Z2YY,
    Z1_offsetXX,
    Z1_offsetXY,
    Z1_offsetYY,
    Final128XX,
    Final128XY,
    Final128YY,
    Result_offsetXX,
    Result_offsetXY,
    Result_offsetYY,
    NumPurposes
};

inline constexpr std::array<const char *, static_cast<size_t>(DebugStatePurpose::NumPurposes)>
DebugStatePurposeStrings {
    "Invalid",
    "ADigits",
    "BDigits",
    "AHalfHigh",
    "AHalfLow",
    "BHalfHigh",
    "BHalfLow",
    "XDiff",
    "YDiff",
    "Z0XX",
    "Z0XY",
    "Z0YY",
    "Z1XX",
    "Z1XY",
    "Z1YY",
    "Z2XX",
    "Z2XY",
    "Z2YY",
    "Z1_offsetXX",
    "Z1_offsetXY",
    "Z1_offsetYY",
    "Final128XX",
    "Final128XY",
    "Final128YY",
    "Result_offsetXX",
    "Result_offsetXY",
    "Result_offsetYY"
};

inline constexpr const char *
DebugStatePurposeToString(DebugStatePurpose purpose) {
    if (static_cast<size_t>(purpose) < DebugStatePurposeStrings.size()) {
        return DebugStatePurposeStrings[static_cast<size_t>(purpose)];
    }
    return "Unknown";
}

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
