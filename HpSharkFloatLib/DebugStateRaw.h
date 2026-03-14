#pragma once

#include <array>
#include <stdint.h>
#include <vector>

enum class UseConvolution { No, Yes };

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
    // NR multiply derivative stages (W0-W3)
    Z0W0,
    Z1W0,
    Z2W0,
    Z3W0,
    Z2_PermW0,
    Z2_PermW0b,
    Final128W0,
    Z0W1,
    Z1W1,
    Z2W1,
    Z3W1,
    Z2_PermW1,
    Z2_PermW1b,
    Final128W1,
    Z0W2,
    Z1W2,
    Z2W2,
    Z3W2,
    Z2_PermW2,
    Z2_PermW2b,
    Final128W2,
    Z0W3,
    Z1W3,
    Z2W3,
    Z3W3,
    Z2_PermW3,
    Z2_PermW3b,
    Final128W3,
    // NR add derivative stages
    FinalAddDzdc1,
    FinalAddDzdc2,
    FinalAddDzdc3,
    Result_AddDzdc1,
    Result_AddDzdc2,
    NumPurposes
};

inline constexpr std::array<const char *, static_cast<size_t>(DebugStatePurpose::NumPurposes)>
    DebugStatePurposeStrings{"Invalid",
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
                             "Result_Add2",
                             "Z0W0",
                             "Z1W0",
                             "Z2W0",
                             "Z3W0",
                             "Z2_PermW0",
                             "Z2_PermW0b",
                             "Final128W0",
                             "Z0W1",
                             "Z1W1",
                             "Z2W1",
                             "Z3W1",
                             "Z2_PermW1",
                             "Z2_PermW1b",
                             "Final128W1",
                             "Z0W2",
                             "Z1W2",
                             "Z2W2",
                             "Z3W2",
                             "Z2_PermW2",
                             "Z2_PermW2b",
                             "Final128W2",
                             "Z0W3",
                             "Z1W3",
                             "Z2W3",
                             "Z3W3",
                             "Z2_PermW3",
                             "Z2_PermW3b",
                             "Final128W3",
                             "FinalAddDzdc1",
                             "FinalAddDzdc2",
                             "FinalAddDzdc3",
                             "Result_AddDzdc1",
                             "Result_AddDzdc2"};

inline constexpr const char *
DebugStatePurposeToString(DebugStatePurpose purpose)
{
    static_assert(static_cast<size_t>(DebugStatePurpose::NumPurposes) == DebugStatePurposeStrings.size(),
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
    uint64_t normalizeCount;
};

class DebugGpuCombo {
public:
    std::vector<DebugStateRaw> States;
    std::vector<DebugGlobalCountRaw> MultiplyCounts;
};
