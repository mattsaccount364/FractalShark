#pragma once

#include <stdint.h>
#include <array>
#include <vector>

enum class UseConvolution {
    No,
    Yes
};

enum class DebugStatePurpose {
    Invalid,
    ADigits,
    BDigits,
    CDigits,
    DDigits,
    EDigits,
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
    Z3XX,
    Z3XY,
    Z3YY,
    Z4XX,
    Z4XY,
    Z4YY,
    Z2_Perm1,
    Z2_Perm2,
    Z2_Perm3,
    Z2_Perm4,
    Z2_Perm5,
    Z2_Perm6,
    Z1_offsetXX,
    Z1_offsetXY,
    Z1_offsetYY,
    Final128XX,
    Final128XY,
    Final128YY,
    FinalAdd1,
    FinalAdd2,
    FinalAdd3,
    Result_offsetXX,
    Result_offsetXY,
    Result_offsetYY,
    Result_Add1,
    Result_Add2,
    NumPurposes
};

inline constexpr std::array<const char *, static_cast<size_t>(DebugStatePurpose::NumPurposes)>
DebugStatePurposeStrings {
    "Invalid",
    "ADigits",
    "BDigits",
    "CDigits",
    "DDigits",
    "EDigits",
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
    "Z3XX",
    "Z3XY",
    "Z3YY",
    "Z4XX",
    "Z4XY",
    "Z4YY",
    "Z2_Perm1",
    "Z2_Perm2",
    "Z2_Perm3",
    "Z2_Perm4",
    "Z2_Perm5",
    "Z2_Perm6",
    "Z1_offsetXX",
    "Z1_offsetXY",
    "Z1_offsetYY",
    "Final128XX",
    "Final128XY",
    "Final128YY",
    "FinalAdd1",
    "FinalAdd2",
    "FinalAdd3",
    "Result_offsetXX",
    "Result_offsetXY",
    "Result_offsetYY",
    "Result_Add1",
    "Result_Add2"
};

inline constexpr const char *
DebugStatePurposeToString(DebugStatePurpose purpose) {
    static_assert (static_cast<size_t>(DebugStatePurpose::NumPurposes) == DebugStatePurposeStrings.size(),
        "DebugStatePurposeStrings size mismatch with DebugStatePurpose enum");
    if (static_cast<size_t>(purpose) < DebugStatePurposeStrings.size()) {
        return DebugStatePurposeStrings[static_cast<size_t>(purpose)];
    }
    return "Unknown";
}

struct DebugStateRaw {
    int Initialized;
    int Block;
    int Thread;
    uint64_t ArraySize;
    uint64_t Checksum;
    DebugStatePurpose ChecksumPurpose;
    int RecursionDepth;
    int CallIndex;
    UseConvolution Convolution;
};

struct DebugGlobalCountRaw {
    int blockIdx;
    int threadIdx;
    uint64_t multiplyCount;
    uint64_t carryCount;
};

class DebugGpuCombo {
public:
    std::vector<DebugStateRaw> States;
    std::vector<DebugGlobalCountRaw> MultiplyCounts;
};