#pragma once

namespace SharkNTT {

// -------------------------------------------
// constexpr 32-bit–limb helpers (MSVC-safe)
// -------------------------------------------
struct u128 {
    uint64_t lo, hi;
};

static constexpr uint64_t
add64_carry(uint64_t a, uint64_t b, uint64_t &carry)
{
    uint64_t s = a + b;
    uint64_t c0 = (s < a) ? 1ull : 0ull;
    s += carry;
    uint64_t c1 = (s < carry) ? 1ull : 0ull;
    carry = c0 + c1;
    return s;
}

static constexpr u128
add128(u128 x, u128 y)
{
    uint64_t c = 0;
    uint64_t lo = add64_carry(x.lo, y.lo, c);
    uint64_t hi = x.hi + y.hi + c;
    return {lo, hi};
}

// 64×64 -> 128 using 32-bit limbs (all intermediates fit in 64-bit)
static constexpr u128
mul64x64(uint64_t a, uint64_t b)
{
    const uint64_t a0 = static_cast<uint32_t>(a);
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = static_cast<uint32_t>(b);
    const uint64_t b1 = b >> 32;

    const uint64_t p00 = a0 * b0; // <= 2^64-2^33+1
    const uint64_t p01 = a0 * b1;
    const uint64_t p10 = a1 * b0;
    const uint64_t p11 = a1 * b1;

    // low 32 bits
    uint64_t lo = (p00 & 0xFFFF'FFFFull);

    // middle 32 bits: (p00>>32) + (p01 & 0xFFFF'FFFF) + (p10 & 0xFFFF'FFFF)
    uint64_t mid = (p00 >> 32);
    mid += (p01 & 0xFFFF'FFFFull);
    mid += (p10 & 0xFFFF'FFFFull);

    // compose low 64
    lo |= (mid & 0xFFFF'FFFFull) << 32;

    // high 64: (mid>>32) + (p01>>32) + (p10>>32) + p11
    uint64_t hi = (mid >> 32);
    hi += (p01 >> 32);
    hi += (p10 >> 32);
    hi += p11;

    return {lo, hi};
}

// -------------------------------------------
// constexpr Montgomery (MSVC-safe, no intrinsics)
// -------------------------------------------
static constexpr uint64_t
MontgomeryMulConstexpr(uint64_t a, uint64_t b)
{
    // t = a*b
    const u128 t = mul64x64(a, b);

    // m = (t.lo * NINV) mod 2^64
    const uint64_t m = mul64x64(t.lo, MagicPrimeInv).lo;

    // m * p
    const u128 mp = mul64x64(m, MagicPrime);

    // u = t + mp
    const u128 u = add128(t, mp);

    // r = u >> 64, then conditional subtract p
    uint64_t r = u.hi;
    if (r >= MagicPrime)
        r -= MagicPrime;
    return r;
}

static constexpr uint64_t
ToMontgomeryConstexpr(uint64_t x)
{
    return MontgomeryMulConstexpr(x, R2);
}

// Precomputed 1 in Montgomery form (for Goldilocks, this is 2^32)
static constexpr uint64_t OneMont = ToMontgomeryConstexpr(1);

// Sanity checks
static_assert(MagicPrime == 0xFFFF'FFFF'0000'0001ull, "p mismatch");
static_assert(MagicPrimeInv == 0xFFFF'FFFE'FFFF'FFFFull, "-p^{-1} mod 2^64 mismatch");
static_assert(R2 == 0xFFFF'FFFE'0000'0001ull, "(2^64)^2 mod p mismatch");
static_assert(OneMont == 0x0000'0000'FFFF'FFFFull, "1 (Montgomery) mismatch");


} // namespace SharkNTT
