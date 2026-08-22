#pragma once

#include "ReferenceNTT.h"

#include "DebugChecksumHost.h"
#include "HpSharkFloat.h"

namespace SharkNTT {

static void
MultiplyWide(uint64_t a, uint64_t b, uint64_t &low, uint64_t &high)
{
#if defined(_MSC_VER) && defined(_M_X64)
    low = _umul128(a, b, &high);
#else
    const unsigned __int128 product =
        static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    low = static_cast<uint64_t>(product);
    high = static_cast<uint64_t>(product >> 64u);
#endif
}

static uint64_t
AddWithCarry(uint64_t a, uint64_t b, uint64_t &carry)
{
    const uint64_t sum = a + b;
    const uint64_t firstCarry = sum < a;
    const uint64_t result = sum + carry;
    carry = firstCarry | (result < sum);
    return result;
}

template <class SharkFloatParams>
uint64_t
MontgomeryMul(uint64_t a, uint64_t b)
{
    uint64_t productLow;
    uint64_t productHigh;
    MultiplyWide(a, b, productLow, productHigh);

    const uint64_t multiplier = productLow * MagicPrimeInv;
    uint64_t modulusLow;
    uint64_t modulusHigh;
    MultiplyWide(multiplier, MagicPrime, modulusLow, modulusHigh);

    uint64_t carry = 0;
    (void)AddWithCarry(productLow, modulusLow, carry);
    uint64_t highCarry = carry;
    uint64_t result = AddWithCarry(productHigh, modulusHigh, highCarry);
    if (highCarry != 0u || result >= MagicPrime)
        result -= MagicPrime;
    return result;
}

template <class SharkFloatParams>
uint64_t
MontgomeryMul(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a, uint64_t b)
{
    debugCombo.MultiplyCounts.DebugMultiplyIncrement(7);
    return MontgomeryMul<SharkFloatParams>(a, b);
}

template <class SharkFloatParams>
uint64_t
ToMontgomery(uint64_t value)
{
    return MontgomeryMul<SharkFloatParams>(value, R2);
}

template <class SharkFloatParams>
uint64_t
ToMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value)
{
    return MontgomeryMul<SharkFloatParams>(debugCombo, value, R2);
}

template <class SharkFloatParams>
uint64_t
FromMontgomery(uint64_t value)
{
    return MontgomeryMul<SharkFloatParams>(value, 1u);
}

template <class SharkFloatParams>
uint64_t
FromMontgomery(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value)
{
    return MontgomeryMul<SharkFloatParams>(debugCombo, value, 1u);
}

template <class SharkFloatParams>
uint64_t
MontgomeryPow(uint64_t value, uint64_t exponent)
{
    uint64_t result = ToMontgomery<SharkFloatParams>(1u);
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u)
            result = MontgomeryMul<SharkFloatParams>(result, value);
        value = MontgomeryMul<SharkFloatParams>(value, value);
        exponent >>= 1u;
    }
    return result;
}

template <class SharkFloatParams>
uint64_t
MontgomeryPow(DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value, uint64_t exponent)
{
    uint64_t result = ToMontgomery(debugCombo, 1u);
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u)
            result = MontgomeryMul(debugCombo, result, value);
        value = MontgomeryMul(debugCombo, value, value);
        exponent >>= 1u;
    }
    return result;
}

} // namespace SharkNTT
