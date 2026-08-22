#pragma once

#include "NTTConstexprGenerator.h"

#include <cstdint>

template <class SharkFloatParams> struct DebugHostCombo;

namespace SharkNTT {

struct Plan {
    int n32 = 0;
    int b = 0;
    int L = 0;
    int N = 0;
    int stages = 0;
    bool ok = false;
};

constexpr uint32_t
NextPow2U32(uint32_t value)
{
    if (value <= 1u)
        return 1u;
    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

constexpr uint32_t
CeilDivU32(uint32_t dividend, uint32_t divisor)
{
    return (dividend + divisor - 1u) / divisor;
}

constexpr uint32_t
CeilLog2U32(uint32_t value)
{
    uint32_t result = 0;
    uint32_t remaining = value == 0u ? 0u : value - 1u;
    while (remaining != 0u) {
        remaining >>= 1u;
        ++result;
    }
    return result;
}

constexpr Plan
BuildPlan(int limbCount)
{
    constexpr int BitsPerCoefficient = 16;
    const uint64_t totalBits = static_cast<uint64_t>(limbCount) * 32ull;
    return {limbCount,
            BitsPerCoefficient,
            static_cast<int>(CeilDivU32(static_cast<uint32_t>(totalBits), BitsPerCoefficient)),
            0,
            0,
            true};
}

struct RootTables {
    int32_t stages;
    uint64_t *stage_omegas;
    uint64_t *stage_omegas_inv;
    int32_t N;
    uint64_t *omega_pows;
    uint64_t Ninvm_mont;
    uint64_t Ninv;
    uint64_t *stage_twiddles_fwd;
    uint64_t *stage_twiddles_inv;
    uint32_t total_twiddles;
    uint64_t InputScaleR;
};

template <class SharkFloatParams> uint64_t MontgomeryMul(uint64_t a, uint64_t b);

template <class SharkFloatParams>
uint64_t MontgomeryMul(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a, uint64_t b);

template <class SharkFloatParams> uint64_t ToMontgomery(uint64_t value);

template <class SharkFloatParams>
uint64_t ToMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value);

template <class SharkFloatParams> uint64_t FromMontgomery(uint64_t value);

template <class SharkFloatParams>
uint64_t FromMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value);

template <class SharkFloatParams> uint64_t MontgomeryPow(uint64_t value, uint64_t exponent);

template <class SharkFloatParams>
uint64_t MontgomeryPow(DebugHostCombo<SharkFloatParams> &debugCombo,
                       uint64_t value,
                       uint64_t exponent);

} // namespace SharkNTT
