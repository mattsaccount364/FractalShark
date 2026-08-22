#pragma once

#include <array>

namespace SharkNTT {

// Goldilocks prime p = 2^64 - 2^32 + 1
static constexpr uint64_t MagicPrime = 0xFFFF'FFFF'0000'0001ull;
static constexpr uint64_t MagicPrimeInv = 0xFFFF'FFFE'FFFF'FFFFull; // -p^{-1} mod 2^64
static constexpr uint64_t MontgomeryR = 0x0000'0000'FFFF'FFFFull;   // 2^64 mod p
static constexpr uint64_t R2 = 0xFFFF'FFFE'0000'0001ull;            // (2^64)^2 mod p
static constexpr uint64_t SqrtInverseTwo = 0x0000'007F'FF7F'FF80ull;
static constexpr uint64_t PHI = 0xFFFF'FFFF'0000'0000ull;

struct U128 {
    uint64_t lo, hi;
};

// Portable 64x64→128 multiply (by-hand 32-bit partials)
constexpr U128
Mul64x64To128(uint64_t a, uint64_t b)
{
    const uint64_t a0 = static_cast<uint32_t>(a);
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = static_cast<uint32_t>(b);
    const uint64_t b1 = b >> 32;

    const uint64_t p00 = a0 * b0; // 32x32 -> 64
    const uint64_t p01 = a0 * b1; // 32x32 -> 64
    const uint64_t p10 = a1 * b0; // 32x32 -> 64
    const uint64_t p11 = a1 * b1; // 32x32 -> 64

    const uint64_t mid = p01 + p10;
    uint64_t lo = p00 + (mid << 32);
    uint64_t carry = (lo < p00);
    uint64_t hi = p11 + (mid >> 32) + carry;
    return {lo, hi};
}

// Reduce (hi:lo) mod p, using 2^64 ≡ 2^32 - 1 (mod p).
constexpr uint64_t
ReduceModP(uint64_t hi, uint64_t lo)
{
    // r = lo + (hi << 32) - hi  (mod 2^64), then correct into [0, p)
    uint64_t r = lo + (hi << 32);
    r -= hi;
    // Up to two corrections suffice for this p
    if (r >= MagicPrime)
        r -= MagicPrime;
    if (r >= MagicPrime)
        r -= MagicPrime;
    return r;
}

constexpr uint64_t
MulModP(uint64_t a, uint64_t b)
{
    const auto prod = Mul64x64To128(a, b);
    return ReduceModP(prod.hi, prod.lo);
}

constexpr uint64_t
PowModP(uint64_t a, uint64_t e)
{
    uint64_t base = (a >= MagicPrime ? a % MagicPrime : a);
    uint64_t r = 1;
    while (e) {
        if (e & 1ull)
            r = MulModP(r, base);
        base = MulModP(base, base);
        e >>= 1;
    }
    return r;
}

constexpr uint64_t
ReferenceInputScale(uint32_t stages)
{
    // These constants are s = (N * R)^(-1/2), generated with arbitrary-precision modular
    // arithmetic.  Keeping the table constexpr avoids widening the existing generator's
    // compile-time modular reducer, whose fast 64-bit fold is intended for its original use.
    switch (stages) {
        case 1u:
            return 0x0080'007F'FF80'0000ull;
        case 2u:
            return 0x7FFF'FFFF'0000'0001ull;
        case 3u:
            return 0x0040'003F'FFC0'0000ull;
        case 4u:
            return 0xBFFF'FFFF'0000'0001ull;
        case 5u:
            return 0x0020'001F'FFE0'0000ull;
        case 6u:
            return 0xDFFF'FFFF'0000'0001ull;
        case 7u:
            return 0x0010'000F'FFF0'0000ull;
        case 8u:
            return 0xEFFF'FFFF'0000'0001ull;
        case 9u:
            return 0x0008'0007'FFF8'0000ull;
        case 10u:
            return 0xF7FF'FFFF'0000'0001ull;
        case 11u:
            return 0x0004'0003'FFFC'0000ull;
        case 12u:
            return 0xFBFF'FFFF'0000'0001ull;
        case 13u:
            return 0x0002'0001'FFFE'0000ull;
        case 14u:
            return 0xFDFF'FFFF'0000'0001ull;
        case 15u:
            return 0x0001'0000'FFFF'0000ull;
        case 16u:
            return 0xFEFF'FFFF'0000'0001ull;
        case 17u:
            return 0x0000'8000'7FFF'8000ull;
        case 18u:
            return 0xFF7F'FFFF'0000'0001ull;
        case 19u:
            return 0x0000'4000'3FFF'C000ull;
        case 20u:
            return 0xFFBF'FFFF'0000'0001ull;
        case 21u:
            return 0x0000'2000'1FFF'E000ull;
        case 22u:
            return 0xFFDF'FFFF'0000'0001ull;
        case 23u:
            return 0x0000'1000'0FFF'F000ull;
        case 24u:
            return 0xFFEF'FFFF'0000'0001ull;
        case 25u:
            return 0x0000'0800'07FF'F800ull;
        default:
            return 0ull;
    }
}

constexpr uint64_t
ReferenceInputScaleR2(uint32_t stages)
{
    switch (stages) {
        case 1u:
            return 0xFFFF'FF7F'0080'0081ull;
        case 2u:
            return 0x7FFF'FFFF'8000'0000ull;
        case 3u:
            return 0xFFFF'FFBF'0040'0041ull;
        case 4u:
            return 0x3FFF'FFFF'C000'0000ull;
        case 5u:
            return 0xFFFF'FFDF'0020'0021ull;
        case 6u:
            return 0x1FFF'FFFF'E000'0000ull;
        case 7u:
            return 0xFFFF'FFEF'0010'0011ull;
        case 8u:
            return 0x0FFF'FFFF'F000'0000ull;
        case 9u:
            return 0xFFFF'FFF7'0008'0009ull;
        case 10u:
            return 0x07FF'FFFF'F800'0000ull;
        case 11u:
            return 0xFFFF'FFFB'0004'0005ull;
        case 12u:
            return 0x03FF'FFFF'FC00'0000ull;
        case 13u:
            return 0xFFFF'FFFD'0002'0003ull;
        case 14u:
            return 0x01FF'FFFF'FE00'0000ull;
        case 15u:
            return 0xFFFF'FFFE'0001'0002ull;
        case 16u:
            return 0x00FF'FFFF'FF00'0000ull;
        case 17u:
            return 0x7FFF'FFFF'0000'8001ull;
        case 18u:
            return 0x007F'FFFF'FF80'0000ull;
        case 19u:
            return 0xBFFF'FFFF'0000'4001ull;
        case 20u:
            return 0x003F'FFFF'FFC0'0000ull;
        case 21u:
            return 0xDFFF'FFFF'0000'2001ull;
        case 22u:
            return 0x001F'FFFF'FFE0'0000ull;
        case 23u:
            return 0xEFFF'FFFF'0000'1001ull;
        case 24u:
            return 0x000F'FFFF'FFF0'0000ull;
        case 25u:
            return 0xF7FF'FFFF'0000'0801ull;
        default:
            return 0ull;
    }
}

constexpr uint64_t
ReferenceInputScaleR(uint32_t stages)
{
    return MulModP(ReferenceInputScale(stages), MontgomeryR);
}

constexpr bool
ValidateReferenceInputScales()
{
    for (uint32_t stages = 1u; stages <= 25u; ++stages) {
        if (ReferenceInputScale(stages) == 0ull || ReferenceInputScaleR2(stages) == 0ull)
            return false;
    }
    return true;
}

constexpr bool
ValidateReferenceInputScaleMontgomeryValues()
{
    for (uint32_t stages = 1u; stages <= 25u; ++stages) {
        if (MulModP(ReferenceInputScaleR(stages), MontgomeryR) != ReferenceInputScaleR2(stages))
            return false;
    }
    return true;
}

constexpr bool
ValidateReferenceEvenScaleShifts()
{
    for (uint32_t stages = 2u; stages <= 24u; stages += 2u) {
        const uint32_t shift = 32u - stages / 2u;
        if (ReferenceInputScaleR(stages) != (1ull << shift))
            return false;
    }
    return true;
}

static_assert(MulModP(SqrtInverseTwo, SqrtInverseTwo) == (MagicPrime + 1ull) / 2ull);
static_assert(ValidateReferenceInputScales());
static_assert(ValidateReferenceInputScaleMontgomeryValues());
static_assert(ValidateReferenceEvenScaleShifts());
static_assert(ReferenceInputScale(16u) == 0xFEFF'FFFF'0000'0001ull);
static_assert(ReferenceInputScaleR2(16u) == 0x00FF'FFFF'FF00'0000ull);
static_assert(ReferenceInputScaleR(16u) == 0x0000'0000'0100'0000ull);

// Prime factorization of phi = p-1 = 2^32 * (2^32 - 1), and 2^32 - 1 = 3*5*17*257*65537
constexpr std::array<uint64_t, 6> PHI_PRIME_FACTORS = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};

constexpr bool
IsPrimitiveRoot(uint64_t g)
{
    if (g <= 1 || g >= MagicPrime)
        return false;
    for (uint64_t q : PHI_PRIME_FACTORS) {
        const uint64_t e = PHI / q;
        if (PowModP(g, e) == 1ull)
            return false;
    }
    return true;
}

// --------- Compile-time generator search (bounded) ----------
consteval uint64_t
FindGeneratorConstexpr()
{
    // Try small integers; Goldilocks has small generators (e.g., 7)
    for (uint64_t g = 7; g < 2000; ++g) {
        if (IsPrimitiveRoot(g))
            return g;
    }
    // If we ever get here, widen the bound or seed with a known generator.
    return 0; // signals failure at compile time
}

static constexpr uint64_t GoldilocksGenerator = SharkNTT::FindGeneratorConstexpr();

// Optional compile-time sanity checks:
static_assert(GoldilocksGenerator != 0, "Failed to find generator at compile time");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 2) != 1, "Not primitive: factor 2");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 3) != 1, "Not primitive: factor 3");

} // namespace SharkNTT
