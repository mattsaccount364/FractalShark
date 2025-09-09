#pragma once

namespace SharkNTT {

// Goldilocks prime p = 2^64 - 2^32 + 1
static constexpr uint64_t P = 0xFFFF'FFFF'0000'0001ull;
static constexpr uint64_t NINV = 0xFFFF'FFFE'FFFF'FFFFull; // -p^{-1} mod 2^64
static constexpr uint64_t R2 = 0xFFFF'FFFE'0000'0001ull;   // (2^64)^2 mod p


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
    if (r >= P)
        r -= P;
    if (r >= P)
        r -= P;
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
    uint64_t base = (a >= P ? a % P : a);
    uint64_t r = 1;
    while (e) {
        if (e & 1ull)
            r = MulModP(r, base);
        base = MulModP(base, base);
        e >>= 1;
    }
    return r;
}

// Prime factorization of phi = p-1 = 2^32 * (2^32 - 1), and 2^32 - 1 = 3*5*17*257*65537
constexpr std::array<uint64_t, 6> PHI_PRIME_FACTORS = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};
constexpr uint64_t PHI = 0xFFFF'FFFF'0000'0000ull;

constexpr bool
IsPrimitiveRoot(uint64_t g)
{
    if (g <= 1 || g >= P)
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

static constexpr uint64_t GoldilocksP = SharkNTT::P;
static constexpr uint64_t GoldilocksGenerator = SharkNTT::FindGeneratorConstexpr();

// Optional compile-time sanity checks:
static_assert(GoldilocksGenerator != 0, "Failed to find generator at compile time");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 2) != 1, "Not primitive: factor 2");
static_assert(SharkNTT::PowModP(GoldilocksGenerator, SharkNTT::PHI / 3) != 1, "Not primitive: factor 3");

} // namespace SharkNTT