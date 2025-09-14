#include <algorithm>
#include <assert.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <unordered_map>

#include <array>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "DebugChecksumHost.h"
#include "HpSharkFloat.cuh" // your header (provides HpSharkFloat<>, DebugStateHost<>)
#include "ReferenceFFT.h"
#include "TestVerbose.h"

static inline uint32_t
reverse_bits32(uint32_t x, int bits)
{
    // Perform full 32-bit bit reversal manually since MSVC doesn't have __builtin_bitreverse32
    x = (x >> 16) | (x << 16);
    x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
    x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
    x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
    return x >> (32 - bits);
}

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Does not allocate; does two passes over Final128 to (1) find 'significant',
// then (2) write the window and set the Karatsuba-style exponent.
// NOTE: Does not set sign; do that at the call site.
template <class P>
static void
NormalizeFromFinal128LikeKaratsuba(
    HpSharkFloat<P>& Out,
    const HpSharkFloat<P>& A,
    const HpSharkFloat<P>& B,
    const uint64_t* final128,     // len = 2*Ddigits
    size_t Ddigits,               // number of 32-bit positions represented in Final128
    int32_t additionalFactorOfTwo // 0 for XX/YY; 1 for XY if you did NOT shift digits
)
{
    constexpr int N = P::GlobalNumUint32;

    auto pass_once = [&](bool write_window, size_t start, size_t needN) -> std::pair<int, int> {
        uint64_t carry_lo = 0, carry_hi = 0;
        size_t idx = 0;           // index in base-2^32 digits being produced
        int highest_nonzero = -1; // last non-zero digit index seen
        int out_written = 0;

        // Helper to emit one 32-bit digit (optionally writing into Out.Digits)
        auto emit_digit = [&](uint32_t dig) {
            if (dig != 0)
                highest_nonzero = (int)idx;
            if (write_window && idx >= start && out_written < (int)needN) {
                Out.Digits[out_written++] = dig;
            }
            ++idx;
        };

        // Main Ddigits loop
        for (size_t d = 0; d < Ddigits; ++d) {
            uint64_t lo = final128[2 * d + 0];
            uint64_t hi = final128[2 * d + 1];

            uint64_t s_lo = lo + carry_lo;
            uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
            uint64_t s_hi = hi + carry_hi + c0;

            emit_digit(static_cast<uint32_t>(s_lo & 0xffffffffull));

            carry_lo = (s_lo >> 32) | (s_hi << 32);
            carry_hi = (s_hi >> 32);
        }

        // Flush remaining carry into more digits
        while (carry_lo || carry_hi) {
            emit_digit(static_cast<uint32_t>(carry_lo & 0xffffffffull));
            uint64_t nlo = (carry_lo >> 32) | (carry_hi << 32);
            uint64_t nhi = (carry_hi >> 32);
            carry_lo = nlo;
            carry_hi = nhi;
        }

        // If we were asked to write a window and didn't fill all N yet, pad zeros
        if (write_window) {
            while (out_written < (int)needN) {
                Out.Digits[out_written++] = 0u;
            }
        }

        return {highest_nonzero, out_written};
    };

    // Pass 1: find 'significant' (last non-zero + 1)
    auto [highest_nonzero, _] = pass_once(/*write_window=*/false, /*start=*/0, /*needN=*/0);
    if (highest_nonzero < 0) {
        // Zero result
        std::memset(Out.Digits, 0, N * sizeof(uint32_t));
        Out.Exponent = A.Exponent + B.Exponent; // Karatsuba zero convention
        return;
    }
    const int significant = highest_nonzero + 1;
    int shift_digits = significant - N;
    if (shift_digits < 0)
        shift_digits = 0;

    // Karatsuba-style exponent
    Out.Exponent = A.Exponent + B.Exponent + 32 * shift_digits + additionalFactorOfTwo;

    // Pass 2: write exactly N digits starting at shift_digits
    (void)pass_once(/*write_window=*/true, /*start=*/(size_t)shift_digits, /*needN=*/(size_t)N);
}

namespace FirstFailingAttempt {

// ============================================================================
// Helpers: 64x64->128 multiply (MSVC-friendly) and tiny utils
// ============================================================================

struct U128 {
    uint64_t lo, hi;
};

static inline U128
mul_64x64_128(uint64_t a, uint64_t b)
{
#if defined(_MSC_VER) && defined(_M_X64)
    U128 r;
    r.lo = _umul128(a, b, &r.hi);
    return r;
#else
    const uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    const uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t p00 = a0 * b0;
    uint64_t p01 = a0 * b1;
    uint64_t p10 = a1 * b0;
    uint64_t p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    U128 r;
    r.lo = (p00 & 0xffffffffull) | (mid << 32);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
#endif
}

static inline uint64_t
gcd_u64(uint64_t a, uint64_t b)
{
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
static inline uint64_t
lcm_u64(uint64_t a, uint64_t b)
{
    return (a / gcd_u64(a, b)) * b;
}

static inline size_t
align_up(size_t x, size_t A)
{
    return (x + (A - 1)) & ~(A - 1);
}

struct Plan {
    // n32 — number of 32-bit limbs in each input mantissa window we pack.
    // -------------------------------------------------------------------
    // Units: limbs (32-bit each).
    // This determines the total bit-length of the raw mantissa we may need
    // to pack: totalBits = n32 * 32. For the “8×32-bit limbs” case,
    // n32 = 8 and totalBits = 256.
    int n32 = 0;

    // b — bits per packed coefficient (base 2^b).
    // -------------------------------------------
    // Units: bits.
    // We pack the integer mantissa into base-2^b digits (windows) before the FFT.
    // Larger b ⇒ fewer coefficients L, but larger dynamic range needed in the ring.
    // To avoid overflow in the ring during butterflies and pointwise products,
    // we choose b so that:
    //     2*b + log2(N) + margin ≤ Kbits
    // (two b-bit values are multiplied, then N-term accumulation spreads energy).
    // Typical choices: b in [16..30]. Example: with Kbits=64 and N=32, b≈26.
    int b = 0;

    // L — number of packed coefficients actually used by the inputs.
    // --------------------------------------------------------------
    // Units: count (coeffs).
    // Computed as L = ceil( totalBits / b ). This is the length of the
    // base-2^b digit vector for each operand. The true linear convolution
    // length is (2*L - 1), so we must ensure N ≥ 2*L (no wrap in iFFT).
    // Example (n32=8, b=26): L = ceil(256/26) = 10.
    int L = 0;

    // N — FFT length (power of two).
    // -------------------------------
    // Units: count (points).
    // This is the negacyclic transform length. We use power-of-two radix-2 FFTs,
    // so N = 2^stages. Must satisfy:
    //   • No-wrap:     2*L - 1 ≤ N
    //   • Twiddle step integrality: N | (2*K)   (we quantize Kbits so this holds)
    // Example (L=10): pick N = nextPow2(2*L) = 32.
    int N = 0;

    // stages — log2(N).
    // -----------------
    // Units: bits (exponent).
    // The number of radix-2 stages in the FFT. Also used when forming the iFFT
    // scale N^{-1} ≡ -2^{K - stages} (mod 2^K+1).
    // Example: N=32 ⇒ stages=5.
    int stages = 0;

    // Kbits — the ring size parameter in 2^K + 1 (bit length K).
    // ----------------------------------------------------------
    // Units: bits.
    // We work in the negacyclic ring ℤ/(2^K+1). K must be large enough to
    // hold products of b-bit digits across the FFT:
    //     2*b + log2(N) + margin ≤ Kbits
    // and we quantize Kbits so that both:
    //     Kbits is a multiple of 64   (word alignment ⇒ simpler, W64 integral)
    //     N divides Kbits             (so 2*K/N is an integer twiddle exponent step)
    // (Implementation uses Kbits as a multiple of lcm(64, N).)
    // Example: with N=32 and the bound ≲ 59, choose Kbits=64.
    int Kbits = 0;

    // W64 — number of 64-bit words per K-bit ring element.
    // ----------------------------------------------------
    // Units: 64-bit words.
    // Each coefficient is stored as W64 64-bit words. We quantize Kbits to a
    // multiple of 64, so W64 = Kbits/64 exactly (no ceiling needed).
    // Example: Kbits=64 ⇒ W64=1; Kbits=128 ⇒ W64=2.
    int W64 = 0;

    // baseShift — global twiddle exponent step (2*K)/N.
    // -------------------------------------------------
    // Units: exponent (an integer modulo 2*K).
    // In the negacyclic FFT the twiddle is w = 2^{(2*K)/N}. At stage s with
    // m = 2^s, the per-butterfly twiddle exponent stride is (2*K)/m.
    // Caching baseShift = (2*K)/N is handy if you want to build per-stage
    // strides as (baseShift * (N/m)). (If you don’t use it directly, it can
    // still serve as a debug sanity value.)
    // Example: Kbits=64, N=32 ⇒ 2*K/N = 128/32 = 4.
    uint64_t baseShift = 0;

    // ok — planning success flag.
    // ---------------------------
    // True if all derived constraints were satisfied and the plan is usable:
    //   • N is power of two
    //   • 2*L - 1 ≤ N (no wrap)
    //   • Kbits is a multiple of lcm(64, N) (⇒ W64 integral, N | Kbits)
    //   • Selected b satisfies the safety bound
    // If any of these fail we mark ok=false and the caller can fall back.
    bool ok = false;

    void
    print() const
    {
        std::cout << "Plan: " << std::endl;
        std::cout << " n32 = " << n32 << std::endl;
        std::cout << " b = " << b << std::endl;
        std::cout << " L = " << L << std::endl;
        std::cout << " N = " << N << std::endl;
        std::cout << " stages = " << stages << std::endl;
        std::cout << " Kbits = " << Kbits << std::endl;
        std::cout << " W64 = " << W64 << std::endl;
        std::cout << " baseShift = " << baseShift << std::endl;
        std::cout << " ok = " << ok << std::endl;
    }
};

// ---------- Headroom / overflow guards (no 128-bit temporaries) ----------

// Asserts a sufficient theoretical bound: during FFT butterflies you need
//    2*b + log2(N) + margin <= Kbits
// The "+1" covers exact-boundary cases; keep margin>=1 (I use 2).
static inline void
AssertNoOverflowByParameters(const Plan& pl, int margin_bits = 2)
{
    [[maybe_unused]] const int need = 2 * pl.b + pl.stages + margin_bits;
    assert(need <= pl.Kbits &&
           "Ring too small: 2*b + log2(N) exceeds Kbits headroom (lower b or raise Kbits).");
}

// Build the K-bit vector for 2^{e}. (All Kbits % 64 done already in your plan.)
static inline void
set_pow2_kbits(uint64_t* x, int W64, int Kbits, int e)
{
    std::memset(x, 0, size_t(W64) * 8);
    if (e < 0 || e >= Kbits)
        return;
    x[e >> 6] = (1ull << (e & 63));
}

// Returns true iff x == 2^{K-1}+1 (the "worst" balanced magnitude, i.e., |x| = 2^{K-1})
static inline bool
is_half_plus_one(const uint64_t* x, int W64, int Kbits)
{
    // danger = 2^{K-1} + 1
    std::vector<uint64_t> danger(W64);
    set_pow2_kbits(danger.data(), W64, Kbits, Kbits - 1);
    // +1 (mod 2^K) — this is ordinary K-bit increment
    uint64_t c = 1ull;
    for (int i = 0; i < W64 && c; ++i) {
        uint64_t p = danger[i];
        danger[i] += c;
        c = (danger[i] < p) ? 1ull : 0ull;
    }
    for (int i = 0; i < W64; ++i)
        if (x[i] != danger[i])
            return false;
    return true;
}

// Debug-only assert to trip when a coefficient lands exactly at ±2^{K-1}.
// (Hitting this means zero headroom; it’s the first value that can flip sign
// under tiny roundoff/bug and is exactly the kind of pattern you saw with
// specific limbs like 0xFFFFFFFF.)
static inline void
AssertNotAtBalancedBoundary(const uint64_t* x, const Plan& pl)
{
    if (is_half_plus_one(x, pl.W64, pl.Kbits)) {
        assert(false && "Coefficient reached magnitude 2^{K-1} (2^{K-1}+1 in Z/(2^K+1)); decrease b or "
                        "increase Kbits.");
    }
}

template <class P>
static Plan
build_plan(int n32)
{
    Plan pl{};
    pl.ok = false;
    pl.n32 = n32;

    auto next_pow2 = [](uint32_t x) -> uint32_t {
        if (x <= 1)
            return 1u;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    };
    auto ceil_div_u32 = [](uint32_t a, uint32_t b) -> uint32_t { return (a + b - 1u) / b; };
    auto gcd_u64 = [](uint64_t a, uint64_t b) -> uint64_t {
        while (b) {
            uint64_t t = a % b;
            a = b;
            b = t;
        }
        return a;
    };
    auto lcm_u64 = [&](uint64_t a, uint64_t b) -> uint64_t {
        if (!a || !b)
            return 0;
        return (a / gcd_u64(a, b)) * b;
    };
    auto ceil_log2_u32 = [](uint32_t x) -> int {
        int l = 0;
        uint32_t v = x - 1u;
        while (v) {
            v >>= 1;
            ++l;
        }
        return l;
    };

    // Conservative, safe b selector: 2*b + ceil_log2(N) + margin <= Kbits
    auto select_b_safely = [&](int Kbits, int N, int b_hint = 26, int margin = 2) -> int {
        const int lgN = ceil_log2_u32((uint32_t)N);
        const int bmax = (Kbits - margin - lgN) / 2;
        int b = b_hint;
        if (b > bmax)
            b = bmax;
        if (b < 16)
            b = 16;
        if (b > 30)
            b = 30;
        return b;
    };

    // -------- Pass 0: crude seed ----------
    const uint32_t totalBits = (uint32_t)pl.n32 * 32u;
    int b_guess = 26;
    int L_guess = (int)ceil_div_u32(totalBits, (uint32_t)b_guess);
    uint32_t N = next_pow2((uint32_t)(2 * L_guess)); // no-wrap target

    // Choose K as multiple of lcm(64, N) so baseShift = K/N is integer
    uint64_t step = lcm_u64(64ull, (uint64_t)N);
    // Lower bound for K: log2(2N) + 2*b + small margin
    double rhs_log2 = std::log2(2.0 * (double)N) + 2.0 * (double)b_guess + 1.0;
    uint64_t Kmin = (uint64_t)std::ceil(rhs_log2);
    uint64_t K = ((Kmin + step - 1) / step) * step; // quantize

    // -------- Pass 1: compute b, L, N with (Kbits,N) consistent ----------
    pl.Kbits = (int)K;
    pl.W64 = (int)(K / 64ull);
    pl.b = select_b_safely(pl.Kbits, (int)N, /*b_hint*/ 26);
    pl.L = (int)ceil_div_u32(totalBits, (uint32_t)pl.b);
    pl.N = (int)next_pow2((uint32_t)(2 * pl.L)); // ensure 2L-1 <= N

    // Because N may have changed, re-quantize K so K%N==0, then recompute b/L/N once more.
    step = lcm_u64(64ull, (uint64_t)pl.N);
    if ((K % step) != 0)
        K = ((K + step - 1) / step) * step;
    pl.Kbits = (int)K;
    pl.W64 = (int)(K / 64ull);
    pl.b = select_b_safely(pl.Kbits, pl.N, /*b_hint*/ 26);
    pl.L = (int)ceil_div_u32(totalBits, (uint32_t)pl.b);
    pl.N = (int)next_pow2((uint32_t)(2 * pl.L)); // final N

    // Final sanity: baseShift integral and no-wrap must hold now
    if (pl.Kbits % pl.N != 0) {
        // quantize up once more if needed
        step = lcm_u64(64ull, (uint64_t)pl.N);
        K = ((K + step - 1) / step) * step;
        pl.Kbits = (int)K;
        pl.W64 = (int)(K / 64ull);
    }
    if (2 * pl.L - 1 > pl.N) {
        // Should not happen, but keep a defensive check
        pl.N = (int)next_pow2((uint32_t)(2 * pl.L));
    }

    // Use w = 2^( (2*Kbits)/N ), so baseShift is (2*K)/N.
    pl.baseShift = ((uint64_t)2 * pl.Kbits) / (uint64_t)pl.N;
    pl.stages = ceil_log2_u32((uint32_t)pl.N);

    // Assert: baseShift should be correct for negacyclic FFT
    assert(pl.baseShift * pl.N == 2u * pl.Kbits &&
           "baseShift calculation error - should be (2*Kbits)/N");
    assert(pl.Kbits % pl.N == 0 && "Kbits must be divisible by N");

    pl.ok = (pl.N >= 2) && (pl.W64 >= 1);

    // Optional: assert the no-wrap condition explicitly
    // 2L-1 <= N  ⇒ linear product fits in negacyclic length-N iFFT
    if (2 * pl.L - 1 > pl.N) {
        std::cerr << "Plan invalid: 2L-1 > N (wrap would occur)\n";
        assert(false);
    }
    return pl;
}

// ============================================================================
// Ring ops on raw K-bit words (arrays of length W64)
// ============================================================================
static inline void
kbits_zero(uint64_t* x, int W64)
{
    std::memset(x, 0, size_t(W64) * 8);
}
static inline bool
kbits_is_zero(const uint64_t* x, int W64)
{
    for (int i = 0; i < W64; ++i)
        if (x[i])
            return false;
    return true;
}

static inline void
kbits_copy(uint64_t* dst, const uint64_t* src, int W64)
{
    std::memcpy(dst, src, size_t(W64) * 8);
}

// K-bit increment by one (wrap)
static inline void
add1_kbits(uint64_t* x, int W64)
{
    uint64_t c = 1ull;
    for (int i = 0; i < W64 && c; ++i) {
        uint64_t p = x[i];
        x[i] += c;
        c = (x[i] < p) ? 1ull : 0ull;
    }
}

// K-bit addition modulo (2^K + 1):
// out = a + b (mod 2^K+1).
// Compute K-bit sum s with carry c. Then out = s - c (i.e., subtract 1 if overflow).
static inline void
add_mod(const uint64_t* a, const uint64_t* b, uint64_t* out, int W64)
{
    uint64_t carry = 0;
    for (int i = 0; i < W64; ++i) {
        uint64_t ai = a[i], bi = b[i];

        // sum = ai + carry
        uint64_t s = ai + carry;
        uint64_t c1 = (s < ai) ? 1ull : 0ull;

        // sum += bi
        uint64_t s2 = s + bi;
        uint64_t c2 = (s2 < bi) ? 1ull : 0ull;

        out[i] = s2;
        carry = (c1 | c2);
    }

    if (carry) {
        // subtract 1 modulo 2^K
        uint64_t borrow = 1ull;
        for (int i = 0; i < W64 && borrow; ++i) {
            uint64_t v = out[i];
            out[i] = v - borrow;
            borrow = (v < borrow) ? 1ull : 0ull;
        }

        if (W64 == 1) {
            assert(out[0] != 0x8000000000000001ULL && "add_mod produced invalid result in Z/(2^64+1)");
        }
    }
}

// out = -a mod (2^K + 1). Convention: 0 -> 0; else (~a)+2 within K bits
static inline void
neg_mod(const uint64_t* a, uint64_t* out, int W64)
{
    if (kbits_is_zero(a, W64)) {
        kbits_zero(out, W64);
        return;
    }

    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        std::cout << "neg_mod input: 0x" << std::hex << a[0] << std::dec << std::endl;
    }

    for (int i = 0; i < W64; ++i)
        out[i] = ~a[i];
    add1_kbits(out, W64);
    add1_kbits(out, W64);

    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        std::cout << "neg_mod output: 0x" << std::hex << out[0] << std::dec << std::endl;
    }
}

// Enhanced sub_mod with edge case debugging
static inline void
sub_mod(const uint64_t* a, const uint64_t* b, uint64_t* out, int W64)
{
    bool debug_this = false;
    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        debug_this = (a[0] <= 0x100 && b[0] <= 0x100);

        if (debug_this) {
            std::cout << "=== DETAILED sub_mod DEBUG ===" << std::endl;
            std::cout << "  Computing: 0x" << std::hex << a[0] << " - 0x" << b[0] << " mod (2^64+1)"
                      << std::dec << std::endl;
        }
    }

    uint64_t borrow = 0;
    for (int i = 0; i < W64; ++i) {
        uint64_t ai = a[i], bi = b[i];

        uint64_t s = ai - bi;
        uint64_t d1 = (ai < bi) ? 1ull : 0ull;

        uint64_t s2 = s - borrow;
        uint64_t d2 = (s < borrow) ? 1ull : 0ull;

        out[i] = s2;
        borrow = (d1 | d2);

        if (debug_this) {
            std::cout << "  Step " << i << ": ai=0x" << std::hex << ai << " bi=0x" << bi << " s=0x" << s
                      << " s2=0x" << s2 << " borrow=" << std::dec << borrow << std::endl;
        }
    }

    if (borrow) {
        if (debug_this) {
            std::cout << "  BORROW case: adding 1 to handle underflow in Z/(2^K+1)" << std::endl;
            std::cout << "  Before compensation: 0x" << std::hex << out[0] << std::dec << std::endl;
        }

        // add 1 modulo 2^K
        uint64_t carry = 1ull;
        for (int i = 0; i < W64 && carry; ++i) {
            uint64_t v = out[i];
            uint64_t s = v + carry;
            out[i] = s;
            carry = (s < v) ? 1ull : 0ull;
        }

        if (debug_this) {
            std::cout << "  After compensation: 0x" << std::hex << out[0] << std::dec << std::endl;
        }

        if (W64 == 1) {
            if (out[0] == 0x8000000000000001ULL) {
                std::cout << "CRITICAL: sub_mod produced invalid result 0x8000000000000001 in Z/(2^64+1)"
                          << std::endl;
                std::cout
                    << "This represents -2^63 which should not occur in normal negacyclic arithmetic"
                    << std::endl;
                assert(false && "sub_mod produced invalid result in Z/(2^64+1)");
            }
        }
    } else if (debug_this) {
        std::cout << "  NO BORROW: direct subtraction result 0x" << std::hex << out[0] << std::dec
                  << std::endl;
    }

    if (debug_this) {
        std::cout << "=== END sub_mod DEBUG ===" << std::endl;
    }
}

// K-bit left shift by r (0<=r<K)
static inline void
shl_kbits(const uint64_t* x, uint32_t r, uint64_t* out, int W64)
{
    if (r == 0) {
        kbits_copy(out, x, W64);
        return;
    }
    uint32_t ws = r >> 6, bs = r & 63u;
    for (int i = W64 - 1; i >= 0; --i) {
        uint64_t lo = 0, hi = 0;
        int s = i - (int)ws;
        if (s >= 0) {
            lo = x[s] << bs;
            if (bs && s > 0)
                hi = (x[s - 1] >> (64 - bs));
        }
        out[i] = lo | hi;
    }
}

// K-bit right shift by r (0<=r<K)
static inline void
shr_kbits(const uint64_t* x, uint32_t r, uint64_t* out, int W64)
{
    if (r == 0) {
        kbits_copy(out, x, W64);
        return;
    }
    uint32_t ws = r >> 6, bs = r & 63u;
    for (int i = 0; i < W64; ++i) {
        uint64_t lo = 0, hi = 0;
        uint32_t s = i + ws;
        if (s < (uint32_t)W64) {
            lo = x[s] >> bs;
            if (bs && (s + 1) < (uint32_t)W64)
                hi = (x[s + 1] << (64 - bs));
        }
        out[i] = lo | hi;
    }
}

// Multiply by 2^e (mod 2^K + 1). Assumes Kbits == 64*W64.
static inline void
mul_pow2(const uint64_t* x,
         uint64_t e,
         int Kbits, // keep for e reduction
         uint64_t* out,
         uint64_t* tmpA, // W64 scratch
         uint64_t* tmpB, // W64 scratch
         int W64)
{

    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        std::cout << "mul_pow2: x=0x" << std::hex << x[0] << " e=" << std::dec << e << " Kbits=" << Kbits
                  << std::endl;
    }

    const uint64_t twoK = (uint64_t)(2 * Kbits);
    e %= twoK;

    // e = q*K + r with q in {0,1}, r in [0, K)
    const bool neg = (e >= (uint64_t)Kbits);
    const int r = (int)(neg ? (e - (uint64_t)Kbits) : e);

    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        std::cout << "  reduced e=" << e << " neg=" << neg << " r=" << r << std::endl;
    }

    if (r == 0) {
        // 2^0 == 1; 2^K == -1 handled by 'neg' below
        kbits_copy(out, x, W64);
    } else {
        // y = (x << r) - (x >> (K - r))  in Z/(2^K+1)
        shl_kbits(x, r, tmpA, W64);         // tmpA = x << r  (within K bits)
        shr_kbits(x, Kbits - r, tmpB, W64); // tmpB = x >> (K-r)

        if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
            std::cout << "    shl result: 0x" << std::hex << tmpA[0] << std::dec << std::endl;
            std::cout << "    shr result: 0x" << std::hex << tmpB[0] << std::dec << std::endl;
        }

        sub_mod(tmpA, tmpB, out, W64); // out = tmpA - tmpB (mod 2^K+1)
    }

    if (neg) {
        // multiply by (-1): out = (0 - out) mod (2^K+1)
        for (int i = 0; i < W64; ++i)
            tmpA[i] = 0ull;
        sub_mod(tmpA, out, out, W64);

        if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
            std::cout << "  after negation: 0x" << std::hex << out[0] << std::dec << std::endl;
        }
    }

    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        std::cout << "mul_pow2 final result: 0x" << std::hex << out[0] << std::dec << std::endl;
    }
}

// Schoolbook: prod[0..2W-1] = a*b (full 2K-bit)
// prod length is 2*W64 64-bit words
static inline void
mul_kbits_schoolbook(const uint64_t* a, const uint64_t* b, uint64_t* prod, int W64)
{
    std::memset(prod, 0, size_t(2 * W64) * sizeof(uint64_t));
    for (int i = 0; i < W64; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < W64; ++j) {
            U128 p = mul_64x64_128(a[i], b[j]);

            // Low limb: prod[i+j] += p.lo
            uint64_t t0 = prod[i + j];
            uint64_t s0 = t0 + p.lo;
            uint64_t c1 = (s0 < t0) ? 1ull : 0ull; // carry from low add
            prod[i + j] = s0;

            // High limb with full carry chain:
            // prod[i+j+1] += p.hi + c1 + carry
            uint64_t t1 = prod[i + j + 1];

            // Pre-compute the three potential carry sources BEFORE we mutate u
            //  1) hi add overflow, 2) low-limb carry (c1), 3) incoming carry
            uint64_t hi_over_pre =
                ((uint64_t)(~t1) < p.hi) ? 1ull : 0ull; // overflow check for t1 + p.hi
            uint64_t multi_sources = hi_over_pre + c1 + carry;

            if (multi_sources > 1ull) {
                std::fprintf(stderr,
                             "[mul_kbits_schoolbook] multi-source carry detected at i=%d j=%d "
                             "(hi_over=%llu, c1=%llu, carry_in=%llu) "
                             "t0=%016llx s0=%016llx t1=%016llx hi=%016llx lo=%016llx\n",
                             i,
                             j,
                             (unsigned long long)hi_over_pre,
                             (unsigned long long)c1,
                             (unsigned long long)carry,
                             (unsigned long long)t0,
                             (unsigned long long)s0,
                             (unsigned long long)t1,
                             (unsigned long long)p.hi,
                             (unsigned long long)p.lo);
                assert(multi_sources <= 1ull &&
                       "multi-source carry occurred; need robust 128-bit accumulate");
            }

            uint64_t u = t1 + p.hi;
            uint64_t c2 = (u < t1) ? 1ull : 0ull;
            u += c1;
            c2 += (u < c1) ? 1ull : 0ull;
            u += carry;
            c2 += (u < carry) ? 1ull : 0ull;

            if (c2 > 1ull) {
                std::fprintf(stderr,
                             "[mul_kbits_schoolbook] aggregated high carry >1 at i=%d j=%d: c2=%llu\n",
                             i,
                             j,
                             (unsigned long long)c2);
                assert(c2 <= 1ull && "aggregated carry grew >1; need robust 128-bit accumulate");
            }

            prod[i + j + 1] = u;
            carry = c2; // keep original behavior (no saturation) so symptom remains identical unless
                        // assert fires
        }

        // Sanity: we only ever expect final carry 0/1 in this algorithm
        if (carry > 1ull) {
            std::fprintf(stderr,
                         "[mul_kbits_schoolbook] tail carry >1 at i=%d: carry=%llu\n",
                         i,
                         (unsigned long long)carry);
            assert(false && "tail carry >1; need robust 128-bit accumulate");
        }

        // Any leftover carry goes to the next limb after (i+W64)
        int k = i + W64 + 1;

        assert(k <= 2 * W64 && "initial tail-carry index out of bounds");
        while (carry) {
            uint64_t t = prod[k];
            prod[k] += carry;                    // original behavior preserved
            carry = (prod[k] < t) ? 1ull : 0ull; // becomes 1 only on overflow
            ++k;

            assert(k <= 2 * W64 && "carry propagation ran past output buffer");
        }
    }
}

// Enhanced reduce_mod_2k1 with detailed debugging
static inline void
reduce_mod_2k1(const uint64_t* prod, uint64_t* out, int W64)
{
    bool debug_this = false;
    if (SharkVerbose == VerboseMode::Debug && W64 == 1) {
        // Debug when we have a simple case that might reveal the issue
        debug_this = (prod[0] <= 0x100 && prod[1] == 0) || (prod[0] == 0x2 && prod[1] == 0);

        if (debug_this) {
            std::cout << "=== DETAILED reduce_mod_2k1 DEBUG ===" << std::endl;
            std::cout << "  Input: low=0x" << std::hex << prod[0] << " high=0x" << prod[1] << std::dec
                      << std::endl;
        }
    }

    sub_mod(prod, prod + W64, out, W64);

    if (debug_this) {
        std::cout << "  sub_mod(low, high) result: 0x" << std::hex << out[0] << std::dec << std::endl;

        // For the specific failing case (2, 0) -> should give 2
        if (prod[0] == 0x2 && prod[1] == 0x0) {
            if (out[0] != 0x2) {
                std::cout << "  CRITICAL ERROR: (2 - 0) mod (2^64+1) should be 2, got 0x" << std::hex
                          << out[0] << std::dec << std::endl;
            } else {
                std::cout << "  OK: (2 - 0) mod (2^64+1) = 2 as expected" << std::endl;
            }
        }

        std::cout << "=== END reduce_mod_2k1 DEBUG ===" << std::endl;
    }
}

static inline void
mul_mod(const uint64_t* a, const uint64_t* b, uint64_t* out, uint64_t* prod2W, int W64)
{

    // Enhanced debug: capture inputs for specific problematic cases
    bool debug_this_call = false;
    if (SharkVerbose == VerboseMode::Debug) {
        // Trigger detailed debug for small values that might cause issues
        debug_this_call = (a[0] <= 0x100 || b[0] <= 0x100) && (W64 == 1);

        if (debug_this_call) {
            std::cout << "=== DETAILED mul_mod DEBUG ===" << std::endl;
            std::cout << "  mul_mod inputs: a=0x" << std::hex << a[0] << " b=0x" << b[0] << std::dec
                      << std::endl;
        }
    }

    mul_kbits_schoolbook(a, b, prod2W, W64);

    if (debug_this_call) {
        std::cout << "  schoolbook result (2K-bit): low=0x" << std::hex << prod2W[0] << " high=0x"
                  << prod2W[1] << std::dec << std::endl;

        // Check for specific patterns that might cause issues
        uint64_t low = prod2W[0];
        uint64_t high = prod2W[1];

        if (high == 0 && low <= 0x100) {
            std::cout << "  PATTERN: Small multiplication (high=0, low <= 0x100)" << std::endl;
        }

        if (low == 0x2 && high == 0) {
            std::cout << "  EXACT CASE: This is the 1*2 = 2 case that's failing!" << std::endl;
        }
    }

    if (SharkVerbose == VerboseMode::Debug && !debug_this_call) {
        std::cout << "  mul_mod: a = ";
        for (int i = W64 - 1; i >= 0; --i) {
            std::cout << std::hex << a[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "  mul_mod: b = ";
        for (int i = W64 - 1; i >= 0; --i) {
            std::cout << std::hex << b[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "  mul_mod: prod= ";
        for (int i = 2 * W64 - 1; i >= 0; --i) {
            std::cout << std::hex << prod2W[i] << " ";
        }
        std::cout << std::endl;
    }

    reduce_mod_2k1(prod2W, out, W64);

    if (debug_this_call) {
        std::cout << "  reduce_mod_2k1 result: 0x" << std::hex << out[0] << std::dec << std::endl;

        // Validate the reduction manually for K=64
        if (W64 == 1) {
            uint64_t expected_manual = prod2W[0] - prod2W[1]; // (low - high) mod 2^64

            // Handle underflow in 64-bit arithmetic
            if (prod2W[0] < prod2W[1]) {
                // In Z/(2^64+1), underflow means we add 1
                expected_manual = (prod2W[0] - prod2W[1]) + 1; // This will wrap naturally
                std::cout << "  UNDERFLOW detected: low < high, adding compensation" << std::endl;
            }

            std::cout << "  Manual calculation: (0x" << std::hex << prod2W[0] << " - 0x" << prod2W[1]
                      << ") = 0x" << expected_manual << std::dec << std::endl;

            if (out[0] != expected_manual) {
                std::cout << "  ERROR: reduce_mod_2k1 result doesn't match manual calculation!"
                          << std::endl;
                std::cout << "  Got: 0x" << std::hex << out[0] << std::endl;
                std::cout << "  Expected: 0x" << expected_manual << std::dec << std::endl;
            }
        }

        // Check for the problematic pattern
        if (out[0] == 0x8000000000000002ULL) {
            std::cout << "  CRITICAL: Produced the problematic value 0x8000000000000002!" << std::endl;
            std::cout << "  This suggests incorrect negacyclic reduction" << std::endl;
        }

        std::cout << "=== END DETAILED mul_mod DEBUG ===" << std::endl;
    }

    if (SharkVerbose == VerboseMode::Debug && !debug_this_call) {
        std::cout << "  mul_mod: out= ";
        for (int i = W64 - 1; i >= 0; --i) {
            std::cout << std::hex << out[i] << " ";
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// FFT over arrays of K-bit coefficients (AoS: element i at base + i*W64)
// ============================================================================
static inline uint64_t*
elem_ptr(uint64_t* base, int idx, int W64)
{
    return base + size_t(idx) * size_t(W64);
}

static inline const uint64_t*
elem_ptr(const uint64_t* base, int idx, int W64)
{
    return base + size_t(idx) * size_t(W64);
}

static void
bit_reverse_inplace(uint64_t* A, int N, int stages, int W64, uint64_t* tmpSwap)
{
    // Preconditions
    assert(N > 0 && (N & (N - 1)) == 0);   // N is power of two
    assert((1u << stages) == (uint32_t)N); // stages matches N
    (void)W64;                             // W64 >= 1 by construction

    for (int i = 0; i < N; ++i) {
        uint32_t j = reverse_bits32((uint32_t)i, stages);
        j &= (uint32_t)(N - 1); // belt & suspenders
        if (j > (uint32_t)i) {
            uint64_t* pi = elem_ptr(A, i, W64);
            uint64_t* pj = elem_ptr(A, (int)j, W64);
            // swap W64 words using tmpSwap (must hold W64 uint64_t)
            std::memcpy(tmpSwap, pi, size_t(W64) * 8);
            std::memcpy(pi, pj, size_t(W64) * 8);
            std::memcpy(pj, tmpSwap, size_t(W64) * 8);
        }
    }
}

template <bool Inverse>
static void
FFT_pow2(uint64_t* A,
         const Plan& pl,
         uint64_t* scratch64,
         const uint64_t* Ninv, // required when Inverse=true
         uint64_t* prod2W)     // required when Inverse=true
{
    // scratch aliases: [tmpA | tmpB | t | sum | diff | swap]
    uint64_t* tmpA = scratch64;
    uint64_t* tmpB = scratch64 + pl.W64;
    uint64_t* t = scratch64 + 2 * pl.W64;
    uint64_t* sum = scratch64 + 3 * pl.W64;
    uint64_t* diff = scratch64 + 4 * pl.W64;
    uint64_t* swp = scratch64 + 5 * pl.W64;

    // --- helpers -------------------------------------------------------------

    auto trace_coeff5 = [&](const char* stage) {
        if (SharkVerbose == VerboseMode::Debug) {
            const uint64_t* c5 = elem_ptr(A, 5, pl.W64);
            std::cout << "Coeff[5] " << stage << ": 0x" << std::hex << c5[0] << std::dec << std::endl;
        }
    };

    auto dump_nonzero_coeffs = [&](const char* stage) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "=== " << stage << " - Non-zero coefficients ===" << std::endl;
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t* ci = elem_ptr(A, i, pl.W64);
                if (ci[0] != 0) {
                    std::cout << "  A[" << i << "] = 0x" << std::hex << ci[0] << std::dec;
                    if (ci[0] <= 0x100)
                        std::cout << " (decimal: " << ci[0] << ")";
                    if (ci[0] == 0x8000000000000002ULL) {
                        std::cout << " *** PROBLEMATIC VALUE ***";
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << "=== End " << stage << " ===" << std::endl;
        }
    };

    auto validate_coeff_range = [&](const char* stage) {
        if (SharkVerbose == VerboseMode::Debug) {
            bool found_issue = false;
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t* ci = elem_ptr(A, i, pl.W64);
                if (ci[0] >= 0x8000000000000000ULL && ci[0] != 0x8000000000000000ULL) {
                    if (!found_issue) {
                        std::cout << "=== RANGE VALIDATION ISSUES at " << stage << " ===" << std::endl;
                        found_issue = true;
                    }
                    std::cout << "  A[" << i << "] = 0x" << std::hex << ci[0]
                              << " (>= 2^63, possible negative representation)" << std::dec << std::endl;
                }
            }
            if (found_issue) {
                std::cout << "=== End Range Validation ===" << std::endl;
            }
        }
    };

    // Equality / zero helpers over pl.W64 limbs
    auto kbits_equal = [&](const uint64_t* x, const uint64_t* y) {
        for (int w = 0; w < pl.W64; ++w)
            if (x[w] != y[w])
                return false;
        return true;
    };
    auto kbits_is_zero = [&](const uint64_t* x) {
        for (int w = 0; w < pl.W64; ++w)
            if (x[w] != 0)
                return false;
        return true;
    };
    auto set_one_into = [&](uint64_t* x) {
        for (int w = 0; w < pl.W64; ++w)
            x[w] = 0;
        x[0] = 1;
    };

    // --- prologue ------------------------------------------------------------

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\n=== FFT_pow2 START (Inverse=" << Inverse << ") ===" << std::endl;
        std::cout << "Plan: N=" << pl.N << ", Kbits=" << pl.Kbits << ", stages=" << pl.stages
                  << std::endl;
    }

    dump_nonzero_coeffs("Initial Input");
    validate_coeff_range("Initial Input");

    // --- Permutation placement (DIT butterflies) ---
    if constexpr (!Inverse) {
        // Forward DIT: expects bit-reversed input, emits natural order.
        trace_coeff5("before bit_reverse");
        bit_reverse_inplace(A, pl.N, pl.stages, pl.W64, swp);
        trace_coeff5("after bit_reverse");
        dump_nonzero_coeffs("After Bit Reverse (Forward)");
    } else {
        // Inverse DIT: input is in natural order (from forward),
        // so reverse up front and DO NOT reverse again at the end.
        trace_coeff5("before bit_reverse (inverse)");
        bit_reverse_inplace(A, pl.N, pl.stages, pl.W64, swp);
        trace_coeff5("after bit_reverse (inverse)");
        dump_nonzero_coeffs("After Bit Reverse (Inverse)");
    }

    const uint64_t twoK = (uint64_t)2 * (uint64_t)pl.Kbits;

    // --- main stages ---------------------------------------------------------

    for (int s = 1; s <= pl.stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;
        const uint64_t stride = (uint64_t)(2 * pl.Kbits) / (uint64_t)m;

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\n--- FFT Stage " << s << " ---" << std::endl;
            std::cout << "  m=" << m << ", half=" << half << ", stride=" << stride << std::endl;
        }

        trace_coeff5("start of stage");
        assert(stride * m == twoK);

        for (uint32_t k = 0; k < (uint32_t)pl.N; k += m) {
            uint64_t tw = 0;

            for (uint32_t j = 0; j < half; ++j) {
                uint64_t* U = elem_ptr(A, (int)(k + j), pl.W64);
                uint64_t* V = elem_ptr(A, (int)(k + j + half), pl.W64);

                // Lightweight debug filter
                bool affects_coeff5 = ((k + j) == 5) || ((k + j + half) == 5);
                bool debug_butterfly =
                    affects_coeff5 || (U[0] != 0 && U[0] <= 0x100) || (V[0] != 0 && V[0] <= 0x100);

                if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                    std::cout << "\n=== BUTTERFLY DEBUG (Stage " << s << ") ===" << std::endl;
                    std::cout << "  Indices: U[" << (k + j) << "] <-> V[" << (k + j + half) << "]"
                              << std::endl;
                    std::cout << "  Before: U=0x" << std::hex << U[0] << ", V=0x" << V[0] << std::dec
                              << std::endl;
                    std::cout << "  Twiddle exponent: tw=" << tw << std::endl;
                    if (affects_coeff5)
                        std::cout << "  *** AFFECTS COEFFICIENT 5 ***" << std::endl;
                }

                // Twiddle
                if constexpr (!Inverse) {
                    if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                        std::cout << "  Forward: Computing V * 2^" << tw << " mod (2^" << pl.Kbits
                                  << "+1)" << std::endl;
                    }
                    mul_pow2(V, tw, (uint32_t)pl.Kbits, t, tmpA, tmpB, pl.W64);
                } else {
                    // inverse: multiply by 2^{-tw} ≡ 2^{2K - tw}
                    uint64_t inv_tw = (tw == 0) ? 0 : (twoK - tw);
                    if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                        std::cout << "  Inverse: Computing V * 2^" << inv_tw
                                  << " (i.e., 2^{-tw}) mod (2^" << pl.Kbits << "+1)"
                                  << "  [tw=" << tw << ", twoK=" << twoK << "]" << std::endl;
                    }
                    mul_pow2(V, inv_tw, (uint32_t)pl.Kbits, t, tmpA, tmpB, pl.W64);
                }

                if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                    std::cout << "  After twiddle: t=0x" << std::hex << t[0] << std::dec << std::endl;
                    if (t[0] == 0x8000000000000002ULL) {
                        std::cout << "  *** CRITICAL: Twiddle produced problematic value! ***"
                                  << std::endl;
                    }
                }

                // Butterfly: (U, V) -> (U+t, U-t)
                add_mod(U, t, sum, pl.W64);
                sub_mod(U, t, diff, pl.W64);

                if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                    std::cout << "  Butterfly results: sum=0x" << std::hex << sum[0] << ", diff=0x"
                              << diff[0] << std::dec << std::endl;
                    if (sum[0] == 0x8000000000000002ULL)
                        std::cout << "  *** CRITICAL: sum produced problematic value! ***" << std::endl;
                    if (diff[0] == 0x8000000000000002ULL)
                        std::cout << "  *** CRITICAL: diff produced problematic value! ***" << std::endl;
                }

                // Range guard
                AssertNotAtBalancedBoundary(sum, pl);
                AssertNotAtBalancedBoundary(diff, pl);

                // --- Representation-aware algebra sanity check (debug only) ---
                if (SharkVerbose == VerboseMode::Debug) {
                    // twoU = 2U ; twoT = 2t
                    add_mod(U, U, tmpA, pl.W64); // tmpA := 2U
                    add_mod(t, t, tmpB, pl.W64); // tmpB := 2t

                    // Check identity 1: (U+t)+(U-t) == 2U
                    add_mod(sum, diff, swp, pl.W64); // swp := sum+diff
                    bool id1_ok = kbits_equal(swp, tmpA);

                    // Check identity 2: (U+t)-(U-t) == 2t
                    sub_mod(sum, diff, swp, pl.W64); // swp := sum-diff
                    bool id2_ok = kbits_equal(swp, tmpB);

                    // Correct for collapsed (-1 ≡ 2^K) when diff==0 but U!=t
                    if ((!id1_ok || !id2_ok) && kbits_is_zero(diff) && !kbits_equal(U, t)) {
                        // Re-evaluate id1 as (sum - 1) == 2U
                        if (!id1_ok) {
                            set_one_into(tmpA);              // tmpA := 1
                            sub_mod(sum, tmpA, swp, pl.W64); // swp := sum - 1  (≡ sum + 2^K)
                            add_mod(U, U, tmpA, pl.W64);     // tmpA := 2U (recompute)
                            id1_ok = kbits_equal(swp, tmpA);
                        }
                        // Re-evaluate id2 as (sum + 1) == 2t
                        if (!id2_ok) {
                            set_one_into(tmpA);              // tmpA := 1
                            add_mod(sum, tmpA, swp, pl.W64); // swp := sum + 1  (≡ sum - 2^K)
                            id2_ok = kbits_equal(swp, tmpB); // tmpB still holds 2t
                        }
                    }

                    if (!id1_ok || !id2_ok) {
                        std::cerr << "Butterfly identity FAIL at stage " << s << " (m=" << m
                                  << ", half=" << half << ", stride=" << stride << ")"
                                  << " k=" << k << " j=" << j << "  tw=" << tw
                                  << (Inverse ? " (inv)" : " (fwd)") << "\n"
                                  << "  U=" << std::hex << U[0] << "  V=" << V[0] << "  t=" << t[0]
                                  << "  sum=" << sum[0] << "  diff=" << diff[0] << std::dec << "\n";
                        assert(false && "Butterfly identities violated");
                    }
                }
                // ----------------------------------------------------------------

                // Store outputs
                kbits_copy(U, sum, pl.W64);
                kbits_copy(V, diff, pl.W64);

                if (SharkVerbose == VerboseMode::Debug && debug_butterfly) {
                    std::cout << "  Final: U[" << (k + j) << "]=0x" << std::hex << U[0] << ", V["
                              << (k + j + half) << "]=0x" << V[0] << std::dec << std::endl;
                    std::cout << "=== END BUTTERFLY DEBUG ===" << std::endl;
                }

                tw += stride;
            }
        }

        if (SharkVerbose == VerboseMode::Debug) {
            trace_coeff5("end of stage");
            std::cout << "--- End Stage " << s << " ---" << std::endl;
            dump_nonzero_coeffs(("After Stage " + std::to_string(s)).c_str());
            validate_coeff_range(("After Stage " + std::to_string(s)).c_str());
        }
    }

    // --- inverse scale -------------------------------------------------------

    if constexpr (Inverse) {
        // Scale by N^{-1} = -2^{K - log2(N)} in Z/(2^K+1)
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\n=== INVERSE FFT SCALING ===" << std::endl;
            std::cout << "Scaling by Ninv = 0x" << std::hex << Ninv[0] << std::dec << std::endl;
        }

        trace_coeff5("before final scaling");
        dump_nonzero_coeffs("Before iFFT Scaling");

        for (int i = 0; i < pl.N; ++i) {
            uint64_t* Xi = elem_ptr(A, i, pl.W64);
            uint64_t orig_value = Xi[0];

            bool debug_this = (SharkVerbose == VerboseMode::Debug) &&
                              (i == 5 || (orig_value != 0 && orig_value <= 0x100));

            if (debug_this) {
                std::cout << "  Scaling A[" << i << "]: 0x" << std::hex << orig_value << " * 0x"
                          << Ninv[0] << std::dec << std::endl;
            }

            mul_mod(Xi, Ninv, Xi, prod2W, pl.W64);

            if (debug_this) {
                std::cout << "    Result: 0x" << std::hex << Xi[0] << std::dec << std::endl;
                if (Xi[0] == 0x8000000000000002ULL) {
                    std::cout << "    *** CRITICAL: iFFT scaling produced problematic value! ***"
                              << "  (input 0x" << std::hex << orig_value << std::dec << ")\n";
                }
            }
        }

        trace_coeff5("after final scaling");
        dump_nonzero_coeffs("After iFFT Scaling");
        validate_coeff_range("After iFFT Scaling");

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "=== Final iFFT output summary ===" << std::endl;
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t* Xi = elem_ptr(A, i, pl.W64);
                if (Xi[0] != 0) {
                    std::cout << "  X[" << i << "] = 0x" << std::hex << Xi[0] << std::dec;
                    if (Xi[0] <= 0x100)
                        std::cout << " (decimal: " << Xi[0] << ")";
                    if (Xi[0] == 0x8000000000000002ULL)
                        std::cout << " *** PROBLEMATIC ***";
                    std::cout << std::endl;
                }
            }
        }
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== FFT_pow2 END (Inverse=" << Inverse << ") ===\n" << std::endl;
    }
}

// ============================================================================
// Pack / Unpack and finalization
// ============================================================================

// Helper: read a b-bit window starting at absolute bit index q from (Digits[], Exponent)
// This applies the inverse of the normalization shift implicitly.
template <class P>
static uint64_t
read_bits_with_exponent(const HpSharkFloat<P>* X, int64_t q, int b)
{
    // Effective bit index inside the mantissa = q + shiftBack
    // shiftBack is the inverse of the left-shift used at construction.
    // It places the value’s LSB at absolute bit 0.
    //
    // Derivation: the implementation stores mantissa shifted up; its exponent tracks the binary point.
    // Using the same logic that prints your “Shifted data: … shiftBits: S”, we need to shift BACK by S.
    // That “S” is (DefaultPrecBits - LowPrec) - 1 - (X->Exponent_lowlevel_mapping).
    // Rather than rely on private details, we compute S from the visible pair (Digits, Exponent):
    // Find where the value’s bit 0 sits relative to the mantissa by combining Exponent with the
    // mantissa’s placement.
    //
    // Practically: compute the bit index of the mantissa’s MSB (mm) and the value’s MSB (vv).
    // The inverse shift is: shiftBack = vv - mm. For integers, vv = floor(log2(value)).
    // Because we don’t want to touch GMP here, we scan the mantissa to get mm and use X->Exponent to get
    // vv.

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "read_bits_with_exponent entry: q=" << q << " b=" << b << std::endl;
    }

    // For now, just extract bits directly from the mantissa without exponent adjustment
    // This will work for testing the FFT pipeline itself
    const int B = P::GlobalNumUint32 * 32; // mantissa width in bits

    // Simple direct bit extraction from mantissa
    int64_t bit = q; // Start at bit position q

    if (bit >= B || bit < 0) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "  read_bits_with_exponent: bit position " << bit
                      << " outside mantissa range [0, " << B << ")" << std::endl;
        }

        return 0;
    }

    // Gather up to 64 bits starting at t (may straddle words); clip at mantissa edges
    uint64_t v = 0;
    int need = b;
    int outPos = 0;
    while (need > 0 && bit < B) {
        int64_t w = bit / 32;
        int off = (int)(bit % 32);
        uint32_t limb = (w >= 0) ? X->Digits[(int)w] : 0u;
        uint32_t chunk = (off ? (limb >> off) : limb);
        int take = std::min(32 - off, need);
        v |= (uint64_t)(chunk & ((1u << take) - 1u)) << outPos;
        outPos += take;
        need -= take;
        bit += take;
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "  extracted value=0x" << std::hex << v << std::dec << std::endl;
    }

    return v & ((b == 64) ? ~0ull : ((1ull << b) - 1ull));
}

template <class P>
static void
pack_base2b_exponent_aware(const HpSharkFloat<P>* X, const Plan& pl, uint64_t* out /*N*W64*/)
{
    std::memset(out, 0, size_t(pl.N) * size_t(pl.W64) * 8);
    for (int i = 0; i < pl.L; ++i) {
        // Read the i-th base-2^b window starting at bit q = i*b of the *value* (not the raw mantissa)

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "pack_base2b_exponent_aware: i=" << i
                      << " reading bits at q=" << (int64_t)i * pl.b << std::endl;
        }

        uint64_t coeff = read_bits_with_exponent(X, /*q=*/(int64_t)i * pl.b, /*b=*/pl.b);
        uint64_t* ci = out + size_t(i) * size_t(pl.W64);
        ci[0] = coeff; // low word; higher words remain zero for chosen b
        // (if you later let b > 64, you’d spill into more words)

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "  pack_base2b_exponent_aware: i=" << i << std::hex << " coeff=0x" << coeff
                      << std::dec << std::endl;
        }
    }
    // Remaining coeffs [pl.L..pl.N-1] are zero (already zeroed)
}

// Accumulate a 32-bit add into interleaved 128-bit buckets at digit index d
static inline void
add32_to_final128(uint64_t* final128 /*len=2*D*/, size_t Ddigits, uint32_t d, uint32_t add32)
{
    if (d >= Ddigits)
        return; // safety
    uint64_t& lo = final128[2ull * d + 0];
    uint64_t& hi = final128[2ull * d + 1];
    uint64_t prev = lo;
    lo += (uint64_t)add32;
    hi += (lo < prev) ? 1ull : 0ull;
}

static inline void
final128_accum_add32(uint64_t* final128, size_t Ddigits, size_t j, uint32_t add)
{
    if (j >= Ddigits || add == 0)
        return;
    uint64_t* lo = &final128[2 * j + 0];
    uint64_t* hi = &final128[2 * j + 1];
    uint64_t old = *lo;
    uint64_t sum = old + (uint64_t)add;
    *lo = sum;
    if (sum < old)
        (*hi) += 1ull; // carry into hi64
}

static inline void
final128_accum_sub32(uint64_t* final128, size_t Ddigits, size_t j, uint32_t sub)
{
    if (j >= Ddigits || sub == 0)
        return;
    uint64_t* lo = &final128[2 * j + 0];
    uint64_t* hi = &final128[2 * j + 1];
    uint64_t old = *lo;
    uint64_t dif = old - (uint64_t)sub;
    *lo = dif;
    if (old < (uint64_t)sub) { // borrow out of lo64
        (*hi) -= 1ull;
    }
}

static void
unpack_to_final128(const uint64_t* A, const Plan& pl, uint64_t* Final128, size_t Ddigits)
{
    std::memset(Final128, 0, sizeof(uint64_t) * 2 * Ddigits);

    const uint64_t HALF = (pl.Kbits == 64) ? (1ull << 63) : 0ull; // K=64 in your plan
    std::span<const uint64_t> Aspan(A, pl.N * pl.W64);
    std::span<const uint64_t> FinalSpan(Final128, 2 * Ddigits);

    for (int i = 0; i < pl.N; ++i) {
        const uint64_t* ci = elem_ptr(A, i, pl.W64);
        uint64_t v = ci[0]; // Kbits=64 => one word

        if (v == 0)
            continue;

        // ----- balanced lift in Z/(2^K+1) -----
        bool neg = (pl.Kbits == 64) && (v > HALF);
        uint64_t mag64;
        if (!neg) {
            mag64 = v; // positive: magnitude = v
        } else {
            // magnitude m = (2^K + 1) - v = (~v) + 2 (since 2^K ≡ 0 in 64-bit)
            mag64 = ~v;
            mag64 += 2ull;
        }

        // bit offset for coefficient i: s = i*b
        const uint64_t sBits = (uint64_t)i * (uint64_t)pl.b;
        const size_t q = (size_t)(sBits >> 5); // /32
        const int r = (int)(sBits & 31);       // %32

        const uint64_t lo64 = (r ? (mag64 << r) : mag64);
        const uint64_t hi64 = (r ? (mag64 >> (64 - r)) : 0ull);

        // split into four 32-bit lanes
        uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu);
        uint32_t d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu);
        uint32_t d2 = (uint32_t)(hi64 & 0xffffffffu);
        uint32_t d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);

        if (!neg) {
            final128_accum_add32(Final128, Ddigits, q + 0, d0);
            final128_accum_add32(Final128, Ddigits, q + 1, d1);
            final128_accum_add32(Final128, Ddigits, q + 2, d2);
            final128_accum_add32(Final128, Ddigits, q + 3, d3);
        } else {
            final128_accum_sub32(Final128, Ddigits, q + 0, d0);
            final128_accum_sub32(Final128, Ddigits, q + 1, d1);
            final128_accum_sub32(Final128, Ddigits, q + 2, d2);
            final128_accum_sub32(Final128, Ddigits, q + 3, d3);
        }

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "  unpack_to_final128: i=" << i << " coeff=" << v << (neg ? " (neg)" : " (pos)")
                      << " => mag=0x" << std::hex << mag64 << std::dec << " lo64=0x" << std::hex << lo64
                      << std::dec << " hi64=0x" << std::hex << hi64 << std::dec << " d32=[" << d0 << ","
                      << d1 << "," << d2 << "," << d3 << "]"
                      << " at q=" << q << " r=" << r << std::endl;
        }
    }
}

// Check function for bit_reverse_inplace
static bool
check_bit_reverse_inplace(int N, int stages)
{
    if (SharkVerbose != VerboseMode::Debug)
        return true;

    // Create test data
    std::vector<uint64_t> test_data(N);
    std::vector<uint64_t> original_data(N);
    std::vector<uint64_t> tmpSwap(1); // W64=1 for this test

    // Initialize with pattern that makes bit reversal obvious
    for (int i = 0; i < N; ++i) {
        test_data[i] = i + 1; // 1, 2, 3, 4, ...
        original_data[i] = test_data[i];
    }

    std::cout << "Testing bit_reverse_inplace with N=" << N << ", stages=" << stages << std::endl;

    // Print original
    std::cout << "Original: ";
    for (int i = 0; i < N; ++i) {
        std::cout << test_data[i] << " ";
    }
    std::cout << std::endl;

    // Apply bit reversal
    bit_reverse_inplace(test_data.data(), N, stages, 1, tmpSwap.data());

    // Print after reversal
    std::cout << "After bit_reverse: ";
    for (int i = 0; i < N; ++i) {
        std::cout << test_data[i] << " ";
    }
    std::cout << std::endl;

    // Verify each element is in the correct position
    bool success = true;
    std::cout << "Verification:" << std::endl;
    for (int i = 0; i < N; ++i) {
        uint32_t expected_j = reverse_bits32((uint32_t)i, stages);
        expected_j &= (uint32_t)(N - 1);
        uint64_t expected_value = original_data[expected_j];

        if (test_data[i] != expected_value) {
            std::cout << "  MISMATCH at i=" << i << ": got " << test_data[i] << ", expected "
                      << expected_value << " (from original[" << expected_j << "])" << std::endl;
            success = false;
        } else {
            std::cout << "  OK: test_data[" << i << "] = " << test_data[i] << " (from original["
                      << expected_j << "])" << std::endl;
        }
    }

    // Apply bit reversal again - should get back to original
    bit_reverse_inplace(test_data.data(), N, stages, 1, tmpSwap.data());

    std::cout << "After double bit_reverse: ";
    for (int i = 0; i < N; ++i) {
        std::cout << test_data[i] << " ";
    }
    std::cout << std::endl;

    // Check if we got back to original
    for (int i = 0; i < N; ++i) {
        if (test_data[i] != original_data[i]) {
            std::cout << "  DOUBLE REVERSAL FAILED at i=" << i << ": got " << test_data[i]
                      << ", expected " << original_data[i] << std::endl;
            success = false;
        }
    }

    std::cout << "Bit reversal test: " << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Also create a function to check reverse_bits32
static bool
check_reverse_bits32()
{
    if (SharkVerbose != VerboseMode::Debug)
        return true;

    std::cout << "Testing reverse_bits32:" << std::endl;

    struct TestCase {
        uint32_t input;
        int bits;
        uint32_t expected;
    };

    TestCase tests[] = {
        {0, 4, 0},  // 0000 -> 0000
        {1, 4, 8},  // 0001 -> 1000
        {2, 4, 4},  // 0010 -> 0100
        {3, 4, 12}, // 0011 -> 1100
        {8, 4, 1},  // 1000 -> 0001
        {0, 5, 0},  // 00000 -> 00000
        {1, 5, 16}, // 00001 -> 10000
        {16, 5, 1}, // 10000 -> 00001
    };

    bool success = true;
    for (const auto& test : tests) {
        uint32_t result = reverse_bits32(test.input, test.bits);
        if (result != test.expected) {
            std::cout << "  FAIL: reverse_bits32(" << test.input << ", " << test.bits << ") = " << result
                      << ", expected " << test.expected << std::endl;
            success = false;
        } else {
            std::cout << "  OK: reverse_bits32(" << test.input << ", " << test.bits << ") = " << result
                      << std::endl;
        }
    }

    std::cout << "reverse_bits32 test: " << (success ? "PASSED" : "FAILED") << std::endl;
    return success;
}

// Verifies Ninv for negacyclic FFT over Z/(2^K + 1).
// Checks three things:
//  (1) Ninv == -2^(K - m)           (additive form)
//  (2) (2^m) * Ninv ≡ 1             (multiplicative form)
//  (3) ((X * Ninv) * 2^m) == X      (round-trip on a few patterns, incl. 0xFFFFFFFF)
static bool
SelfCheckNinv(const Plan& pl, const uint64_t* Ninv)
{
    const int m = pl.stages; // N = 2^m
    const int Kbits = pl.Kbits;
    const int W64 = pl.W64;

    auto dump_kbits = [&](const char* tag, const uint64_t* x) {
        if (SharkVerbose != VerboseMode::Debug)
            return;
        std::cout << tag << " = ";
        for (int i = W64 - 1; i >= 0; --i)
            std::cout << std::hex << x[i] << " ";
        std::cout << std::dec << "\n";
    };

    auto kbits_equal = [&](const uint64_t* a, const uint64_t* b) -> bool {
        for (int i = 0; i < W64; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    };

    auto kzero = [&](uint64_t* x) { std::memset(x, 0, size_t(W64) * 8); };
    auto kone = [&](uint64_t* x) {
        kzero(x);
        x[0] = 1ull;
    };

    // x = 2^e as a K-bit vector (0 <= e < Kbits)
    auto set_pow2 = [&](uint64_t* x, int e) {
        kzero(x);
        if (e < 0 || e >= Kbits)
            return; // defensive
        const int word = e >> 6;
        const int bit = e & 63;
        x[word] = (1ull << bit);
    };

    bool ok = true;

    // Working buffers
    std::vector<uint64_t> P(W64), M(W64), Sum(W64), TwoToM(W64), One(W64), Check(W64);
    std::vector<uint64_t> Tmp2W(2 * W64), tA(W64), tB(W64);

    // --- (1) Additive form: Ninv must equal -2^{K-m} ---------------------
    set_pow2(P.data(), Kbits - m);    // P = 2^{K-m}
    neg_mod(P.data(), M.data(), W64); // M = -2^{K-m} in Z/(2^K+1)
    if (!kbits_equal(Ninv, M.data())) {
        dump_kbits("Expected -2^{K-m}", M.data());
        dump_kbits("Built Ninv        ", Ninv);
        ok = false;
    }

    // Equivalent check: 2^{K-m} + Ninv ≡ 0
    add_mod(P.data(), Ninv, Sum.data(), W64);
    for (int i = 0; i < W64; ++i) {
        if (Sum[i] != 0) {
            dump_kbits("2^{K-m} + Ninv (should be 0)", Sum.data());
            ok = false;
            break;
        }
    }

    // --- (2) Multiplicative form: (2^m) * Ninv ≡ 1 -----------------------
    set_pow2(TwoToM.data(), m);
    kone(One.data());

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "SelfCheckNinv: " << std::endl;
    }

    mul_mod(TwoToM.data(), Ninv, Check.data(), Tmp2W.data(), W64); // Check = (2^m)*Ninv
    if (!kbits_equal(Check.data(), One.data())) {
        dump_kbits("(2^m)*Ninv", Check.data());
        dump_kbits("Expected 1", One.data());
        ok = false;
    }

    // Also cross-check via mul_pow2 (should give 1 as well)
    mul_pow2(const_cast<uint64_t*>(Ninv), (uint64_t)m, Kbits, Check.data(), tA.data(), tB.data(), W64);
    if (!kbits_equal(Check.data(), One.data())) {
        dump_kbits("mul_pow2(Ninv, m)", Check.data());
        dump_kbits("Expected 1        ", One.data());
        ok = false;
    }

    // --- (3) Round-trip identity on a few patterns ------------------------
    // Should satisfy: ((X * Ninv) * 2^m) == X
    if (SharkVerbose == VerboseMode::Debug) {
        for (int trial = 0; trial < 5; ++trial) {
            std::vector<uint64_t> X(W64), Y(W64), Z(W64);
            kzero(X.data());
            switch (trial) {
                case 0:
                    X[0] = 0xFFFFFFFFull;
                    break; // your corner case
                case 1:
                    X[0] = 0x8000000000000000ull;
                    break; // high bit
                case 2:
                    X[0] = 0x123456789ABCDEF0ull;
                    break; // randomish
                case 3:
                    if (W64 > 1) {
                        X[1] = 1ull;
                    } else {
                        X[0] = 1ull;
                    }
                    break; // cross-word
                case 4:
                    if (W64 > 1) {
                        X[0] = 0xAAAAAAAAAAAAAAAAull;
                        X[1] = 0x5555555555555555ull;
                    } else {
                        X[0] = 0xAAAAAAAAAAAAAAAAull;
                    }
                    break;
            }

            mul_mod(X.data(), Ninv, Y.data(), Tmp2W.data(), W64);                        // Y = X * Ninv
            mul_pow2(Y.data(), (uint64_t)m, Kbits, Z.data(), tA.data(), tB.data(), W64); // Z = Y * 2^m

            if (!kbits_equal(X.data(), Z.data())) {
                dump_kbits("X               ", X.data());
                dump_kbits("((X*Ninv)<<m)   ", Z.data());
                ok = false; // continue to print all mismatches in debug
            }
        }
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Ninv self-check: " << (ok ? "OK" : "FAILED") << "\n";
    }
    return ok;
}

} // namespace FirstFailingAttempt

// ============================================================================
// Public API (drop-in): MultiplyHelperKaratsubaV2 (CPU, single-thread, S–S)
// ============================================================================
template <class SharkFloatParams>
void
MultiplyHelperFFT(const HpSharkFloat<SharkFloatParams>* A,
                  const HpSharkFloat<SharkFloatParams>* B,
                  HpSharkFloat<SharkFloatParams>* OutXX,
                  HpSharkFloat<SharkFloatParams>* OutXY,
                  HpSharkFloat<SharkFloatParams>* OutYY,
                  DebugHostCombo<SharkFloatParams>& /*debugCombo*/)
{
    using namespace FirstFailingAttempt;
    using P = SharkFloatParams;
    const Plan pl = build_plan<P>(P::GlobalNumUint32);
    AssertNoOverflowByParameters(pl);
    if (!pl.ok) {
        // Produce zeros if planning fails (extremely unlikely with our choices)
        std::memset(OutXX->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        std::memset(OutXY->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        std::memset(OutYY->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        return;
    }

    // ---- Single arena allocation (64B aligned slices) ----
    const size_t W64 = (size_t)pl.W64;
    const size_t coeffWords = (size_t)pl.N * W64;                   // words per coeff array
    const size_t Ddigits = ((size_t)pl.N * (size_t)pl.b + 31) / 32; // final128 digits

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "MultiplyHelperFFT: " << std::endl;

        pl.print();

        std::cout << "W64=" << W64 << " coeffWords=" << coeffWords << " Ddigits=" << Ddigits
                  << std::endl;
    }

    // Byte sizes
    const size_t bytesCoeff = coeffWords * sizeof(uint64_t);
    const size_t bytesFinal128 = (2 * Ddigits) * sizeof(uint64_t);
    const size_t bytesProd2K = (2 * W64) * sizeof(uint64_t);
    const size_t bytesNinv = (1 * W64) * sizeof(uint64_t);
    const size_t bytesScratch = (6 * W64) * sizeof(uint64_t); // tmpA,tmpB,t,sum,diff,swap

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " bytesCoeff=" << bytesCoeff << " bytesFinal128=" << bytesFinal128
                  << " bytesProd2K=" << bytesProd2K << " bytesNinv=" << bytesNinv
                  << " bytesScratch=" << bytesScratch << std::endl;
    }

    // We keep: Apacked, Bpacked, Awork, Bwork, Ftmp (5 coeff arrays)
    size_t off = 0;
    off = align_up(off, 64);
    size_t off_Apacked = off;
    off += bytesCoeff;
    off = align_up(off, 64);
    size_t off_Bpacked = off;
    off += bytesCoeff;
    off = align_up(off, 64);
    size_t off_Awork = off;
    off += bytesCoeff;
    off = align_up(off, 64);
    size_t off_Bwork = off;
    off += bytesCoeff;
    off = align_up(off, 64);
    size_t off_Ftmp = off;
    off += bytesCoeff;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " off_Apacked=" << off_Apacked << " off_Bpacked=" << off_Bpacked
                  << " off_Awork=" << off_Awork << " off_Bwork=" << off_Bwork << " off_Ftmp=" << off_Ftmp
                  << std::endl;
    }

    off = align_up(off, 64);
    size_t off_Final128 = off;
    off += bytesFinal128;

    off = align_up(off, 64);
    size_t off_Prod2K = off;
    off += bytesProd2K;
    off = align_up(off, 64);
    size_t off_Ninv = off;
    off += bytesNinv;
    off = align_up(off, 64);
    size_t off_Scratch = off;
    off += bytesScratch;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " off_Final128=" << off_Final128 << " off_Prod2K=" << off_Prod2K
                  << " off_Ninv=" << off_Ninv << " off_Scratch=" << off_Scratch << std::endl;
    }

    const size_t arenaBytes = align_up(off, 64);
    std::unique_ptr<std::byte[]> arena(new std::byte[arenaBytes]);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " arenaBytes=" << arenaBytes << std::endl;
    }

    // Slices
    auto Apacked = reinterpret_cast<uint64_t*>(arena.get() + off_Apacked);
    auto Bpacked = reinterpret_cast<uint64_t*>(arena.get() + off_Bpacked);
    auto Awork = reinterpret_cast<uint64_t*>(arena.get() + off_Awork);
    auto Bwork = reinterpret_cast<uint64_t*>(arena.get() + off_Bwork);
    auto Ftmp = reinterpret_cast<uint64_t*>(arena.get() + off_Ftmp);

    auto Final128 = reinterpret_cast<uint64_t*>(arena.get() + off_Final128);

    auto Prod2K = reinterpret_cast<uint64_t*>(arena.get() + off_Prod2K);
    auto Ninv = reinterpret_cast<uint64_t*>(arena.get() + off_Ninv);
    auto Scratch = reinterpret_cast<uint64_t*>(arena.get() + off_Scratch);

    const auto ApackedSize = off_Bpacked - off_Apacked;
    std::span<uint64_t> span_Apacked(Apacked, ApackedSize);

    const auto BpackedSize = off_Awork - off_Bpacked;
    std::span<uint64_t> span_Bpacked(Bpacked, BpackedSize);

    const auto AworkSize = off_Bwork - off_Awork;
    std::span<uint64_t> span_Awork(Awork, AworkSize);

    const auto BworkSize = off_Ftmp - off_Bwork;
    std::span<uint64_t> span_Bwork(Bwork, BworkSize);

    const auto FtmpSize = off_Final128 - off_Ftmp;
    std::span<uint64_t> span_Ftmp(Ftmp, FtmpSize);

    const auto Final128Size = off_Prod2K - off_Final128;
    std::span<uint64_t> span_Final128(Final128, Final128Size);

    const auto Prod2KSize = off_Ninv - off_Prod2K;
    std::span<uint64_t> span_Prod2K(Prod2K, Prod2KSize);

    const auto NinvSize = off_Scratch - off_Ninv;
    std::span<uint64_t> span_Ninv(Ninv, NinvSize);

    const auto ScratchSize = arenaBytes - off_Scratch;
    std::span<uint64_t> span_Scratch(Scratch, ScratchSize);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " Apacked @" << (void*)Apacked << " Bpacked @" << (void*)Bpacked << " Awork @"
                  << (void*)Awork << " Bwork @" << (void*)Bwork << " Ftmp @" << (void*)Ftmp << std::endl;
        std::cout << " Final128 @" << (void*)Final128 << std::endl;
        std::cout << " Prod2K @" << (void*)Prod2K << " Ninv @" << (void*)Ninv << " Scratch @"
                  << (void*)Scratch << std::endl;
    }

    // ---- Build N^{-1} = -2^{K-m} (bits [K-m .. K-1] set) ----
    std::memset(Ninv, 0, W64 * 8);
    const int m = pl.stages;
    if (m <= pl.Kbits) {
        const int start = pl.Kbits - m;

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << " Ninv = -2^(K-m), m=" << m << " start=" << start << std::endl;
        }

        // Set bits across 64-bit words
        for (int bit = start; bit < pl.Kbits; ++bit) {
            Ninv[bit >> 6] |= (1ull << (bit & 63));
        }
    }

    // Add +1 in the K-bit ring to complete -2^{K-m} ***
    add1_kbits(Ninv, pl.W64);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " Ninv = ";
        for (int w = 0; w < pl.W64; ++w) {
            std::cout << std::hex << Ninv[w] << " ";
        }
        std::cout << std::dec << std::endl;
    }

    // Verify Ninv is exactly what we think it is.
    [[maybe_unused]] bool ninv_ok = SelfCheckNinv(pl, Ninv);
    assert(ninv_ok);

    // ---- Pack inputs ----
    pack_base2b_exponent_aware<P>(A, pl, Apacked);
    pack_base2b_exponent_aware<P>(B, pl, Bpacked);

    if (SharkVerbose == VerboseMode::Debug) {
        auto printPacked = [&](const char* name, const uint64_t* Xpacked) {
            std::cout << name << " packed: " << std::endl;
            for (int i = 0; i < pl.N; ++i) {
                std::cout << i << ":";
                std::cout << std::hex << "0x" << Xpacked[i] << std::dec << std::endl;
            }
            std::cout << std::dec << std::endl;
        };

        printPacked("A", Apacked);
        printPacked("B", Bpacked);
    }

    auto verifyFunction = [&]() {
        // First check the bit reversal functions
        [[maybe_unused]] bool bit_reverse_ok = check_reverse_bits32();
        assert(bit_reverse_ok);

        [[maybe_unused]] bool bit_reverse_inplace_ok = check_bit_reverse_inplace(pl.N, pl.stages);
        assert(bit_reverse_inplace_ok);

        auto fft_roundtrip_check = [&](const uint64_t* src) {
            std::vector<uint64_t> tmp(pl.N * pl.W64);
            std::memcpy(tmp.data(), src, tmp.size() * 8);

            // Add debug output for the specific failing case
            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "=== FFT Roundtrip Debug ===" << std::endl;
                std::cout << "Input data:" << std::endl;
                for (int i = 0; i < pl.N; ++i) {
                    const uint64_t* ci = elem_ptr(src, i, pl.W64);
                    if (ci[0] != 0 || i == 5) { // Always show index 5
                        std::cout << "  src[" << i << "] = 0x" << std::hex << ci[0] << std::dec
                                  << std::endl;
                    }
                }
            }

            // Assert pre-conditions
            assert(pl.W64 == 1 && "Expected W64=1 for this debug");
            assert(pl.Kbits == 64 && "Expected Kbits=64 for this debug");

            // Forward FFT
            FFT_pow2<false>(tmp.data(), pl, Scratch, /*Ninv*/ nullptr, /*Prod2K*/ nullptr);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After forward FFT:" << std::endl;
                for (int i = 0; i < pl.N; ++i) {
                    const uint64_t* ci = elem_ptr(tmp.data(), i, pl.W64);
                    if (ci[0] != 0 || i == 5) {
                        std::cout << "  fwd[" << i << "] = 0x" << std::hex << ci[0] << std::dec
                                  << std::endl;
                    }
                }
            }

            // Inverse FFT
            FFT_pow2<true>(tmp.data(), pl, Scratch, Ninv, Prod2K);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After inverse FFT:" << std::endl;
                for (int i = 0; i < pl.N; ++i) {
                    const uint64_t* ci = elem_ptr(tmp.data(), i, pl.W64);
                    if (ci[0] != 0 || i == 5) {
                        std::cout << "  inv[" << i << "] = 0x" << std::hex << ci[0] << std::dec
                                  << std::endl;
                    }
                }
            }

            const uint64_t* ref = src;
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t* xi = elem_ptr(tmp.data(), i, pl.W64);
                const uint64_t* ri = elem_ptr(ref, i, pl.W64);
                for (int w = 0; w < pl.W64; ++w) {
                    if (xi[w] != ri[w]) {
                        // Enhanced debug for the mismatch
                        std::cout << "=== MISMATCH DETECTED ===" << std::endl;
                        std::cout << "Index: " << i << ", Word: " << w << std::endl;
                        std::cout << "Got:      0x" << std::hex << xi[w] << std::dec << std::endl;
                        std::cout << "Expected: 0x" << std::hex << ri[w] << std::dec << std::endl;
                        std::cout << "Plan details: N=" << pl.N << ", Kbits=" << pl.Kbits
                                  << ", stages=" << pl.stages << std::endl;

                        // Check if this looks like a sign flip in the negacyclic ring
                        if (pl.Kbits == 64 && xi[w] == (~ri[w] + 2)) {
                            std::cout << "ERROR: This looks like incorrect negacyclic reduction!"
                                      << std::endl;
                            std::cout << "Got value appears to be -ref in Z/(2^64+1)" << std::endl;
                        }

                        // Check for common bit patterns
                        if (xi[w] & 0x8000000000000000ULL) {
                            std::cout << "WARNING: Result has high bit set (value >= 2^63)" << std::endl;
                        }

                        assert(false && "FFT roundtrip mismatch - check negacyclic ring arithmetic");
                        return false;
                    }
                }
            }
            return true;
        };

        // Check on the packed time-domain inputs (or on a saved copy of Awork/Bwork if you want).
        bool okA = fft_roundtrip_check(Apacked);
        bool okB = fft_roundtrip_check(Bpacked);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Checking FFT roundtrip on packed inputs (A):\n";
        }

        assert(okA);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Checking FFT roundtrip on packed inputs (B):\n";
        }

        assert(okB);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "FFT roundtrip A=" << okA << " B=" << okB << "\n";
        }
    };

    verifyFunction();

    // ---- Forward FFT on copies (Awork/Bwork) ----
    std::memcpy(Awork, Apacked, bytesCoeff);
    std::memcpy(Bwork, Bpacked, bytesCoeff);

    auto beforeForwardFFT = [](const char* name, const uint64_t* Apacked, const Plan& pl) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Before forward FFT - packed coefficients: " << name << std::endl;
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t* ci = elem_ptr(Apacked, i, pl.W64);
                if (ci[0] != 0) {
                    std::cout << "Apacked[" << i << "] = " << std::hex << ci[0] << std::dec << std::endl;
                }
            }
        }
    };

    beforeForwardFFT("A", Awork, pl);
    beforeForwardFFT("B", Bwork, pl);

    FFT_pow2<false>(Awork, pl, Scratch, nullptr, nullptr);
    FFT_pow2<false>(Bwork, pl, Scratch, nullptr, nullptr);

    auto dumpSpectral = [&](const char* name, uint64_t* W) {
        if (SharkVerbose != VerboseMode::Debug)
            return;

        std::cout << name << "[0]=" << std::hex << elem_ptr(W, 0, pl.W64)[0]
                  << " [1]=" << elem_ptr(W, 1, pl.W64)[0]
                  << " [N/2]=" << elem_ptr(W, pl.N / 2, pl.W64)[0]
                  << " [N-1]=" << elem_ptr(W, pl.N - 1, pl.W64)[0] << std::dec << "\n";
    };
    dumpSpectral("Awork", Awork);
    dumpSpectral("Bwork", Bwork);

    auto printDebug = [](const char* name, const uint64_t* Ftmp, int N, int W64) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << name << ": After pointwise multiplication (Ftmp):" << std::endl;
            for (int i = 0; i < N; ++i) {
                const uint64_t* fi = elem_ptr(Ftmp, i, W64);
                if (fi[0] != 0) {
                    std::cout << "Ftmp[" << i << "] = " << std::hex << fi[0] << std::dec << std::endl;
                }
            }
        }
    };

    // Lambda to compute FFT-based multiplication and finalize result
    auto computeProduct = [&](const uint64_t* workA,
                              const uint64_t* workB,
                              const char* debugName,
                              HpSharkFloat<P>* output,
                              const HpSharkFloat<P>& inputA,
                              const HpSharkFloat<P>& inputB,
                              int32_t additionalFactorOfTwo,
                              bool isNegative) {
        // Pointwise multiplication in frequency domain
        for (int i = 0; i < pl.N; ++i) {
            const uint64_t* ai = elem_ptr(workA, i, pl.W64);
            const uint64_t* bi = elem_ptr(workB, i, pl.W64);
            uint64_t* fo = elem_ptr(Ftmp, i, pl.W64);
            mul_mod(ai, bi, fo, Prod2K, pl.W64);
            AssertNotAtBalancedBoundary(fo, pl);
        }
        printDebug(debugName, Ftmp, pl.N, pl.W64);

        // Inverse FFT
        FFT_pow2<true>(Ftmp, pl, Scratch, Ninv, Prod2K);
        unpack_to_final128(Ftmp, pl, Final128, Ddigits);

        // Set sign and normalize with Karatsuba-style exponent handling
        output->SetNegative(isNegative);
        NormalizeFromFinal128LikeKaratsuba<P>(
            *output, inputA, inputB, Final128, Ddigits, additionalFactorOfTwo);
    };

    // ============================================================
    // XX = A^2
    // ============================================================
    computeProduct(Awork, Awork, "XX", OutXX, *A, *A, 0, false);

    // ============================================================
    // YY = B^2
    // ============================================================
    computeProduct(Bwork, Bwork, "YY", OutYY, *B, *B, 0, false);

    // ============================================================
    // XY = 2*(A*B)
    // ============================================================
    computeProduct(Awork, Bwork, "XY", OutXY, *A, *B, 1, A->GetNegative() ^ B->GetNegative());
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void MultiplyHelperFFT<SharkFloatParams>(                                                  \
        const HpSharkFloat<SharkFloatParams>*,                                                          \
        const HpSharkFloat<SharkFloatParams>*,                                                          \
        HpSharkFloat<SharkFloatParams>*,                                                                \
        HpSharkFloat<SharkFloatParams>*,                                                                \
        HpSharkFloat<SharkFloatParams>*,                                                                \
        DebugHostCombo<SharkFloatParams>& debugHostCombo);

ExplicitInstantiateAll();
