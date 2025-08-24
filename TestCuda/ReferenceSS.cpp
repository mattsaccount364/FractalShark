#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "TestVerbose.h"
#include "ReferenceSS.h"
#include "HpSharkFloat.cuh"   // your header (provides HpSharkFloat<>, DebugStateHost<>)
#include "DebugChecksumHost.h"

// ============================================================================
// Helpers: 64x64->128 multiply (MSVC-friendly) and tiny utils
// ============================================================================
struct U128 { uint64_t lo, hi; };

static inline U128 mul_64x64_128(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && defined(_M_X64)
    U128 r; r.lo = _umul128(a, b, &r.hi); return r;
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

static inline uint32_t reverse_bits32(uint32_t x, int bits) {
#if defined(_MSC_VER)
    x = _byteswap_ulong(x);
    return x >> (32 - bits);
#else
    x = (x >> 16) | (x << 16);
    x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
    x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
    x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
    x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
    return x >> (32 - bits);
#endif
}

static inline uint64_t gcd_u64(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }
static inline uint64_t lcm_u64(uint64_t a, uint64_t b) { return (a / gcd_u64(a, b)) * b; }

static inline size_t align_up(size_t x, size_t A) { return (x + (A - 1)) & ~(A - 1); }

// ============================================================================
// Plan selection (N, K, b)
// ============================================================================
struct Plan {
    int n32 = 0;            // input limbs (32-bit)
    int b = 0;            // bits per packed coefficient
    int L = 0;            // number of packed coeffs
    int N = 0;            // FFT length (power of two)
    int stages = 0;         // log2(N)
    int Kbits = 0;          // ring modulus 2^K + 1
    int W64 = 0;          // Kbits / 64
    uint64_t baseShift = 0; // (2*K)/N
    bool ok = false;

    void print() const {
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

static inline uint32_t next_pow2(uint32_t x) {
    if (x <= 1) return 1u;
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

// Safe bound: 2^K > 2 N (2^b - 1)^2  ⇒  K >= ceil(log2(2N)) + 2b
static inline int safe_b_for(uint32_t N, uint32_t K) {
    int need = 0; uint32_t v = 2u * N; while ((1u << need) < v) ++need;
    int b = (int)((K - need) / 2);
    if (b > 26) b = 26; if (b < 12) b = 12; return b;
}

template<class P>
static Plan build_plan(int n32) {
    Plan pl;
    pl.n32 = n32;

    // Initial guess
    int b_guess = 26;
    int L_guess = (int)(((int64_t)n32 * 32 + b_guess - 1) / b_guess);
    uint32_t N = next_pow2((uint32_t)L_guess);

    // Choose K as multiple of lcm(64, N/2), large enough for the bound
    uint64_t step = lcm_u64(64ull, (uint64_t)(N / 2 ? (N / 2) : 1));
    double rhs_log2 = std::log2(2.0 * (double)N) + 2.0 * (double)b_guess;
    uint64_t Kmin = (uint64_t)std::ceil(rhs_log2 + 1.0);
    uint64_t K = ((Kmin + step - 1) / step) * step;

    pl.Kbits = (int)K;
    pl.W64 = (int)(K / 64ull);
    pl.b = safe_b_for(N, (uint32_t)K);
    pl.L = (int)(((int64_t)n32 * 32 + pl.b - 1) / pl.b);
    pl.N = (int)next_pow2((uint32_t)pl.L);
    pl.stages = 0; { int t = pl.N; while ((t >>= 1) != 0) ++pl.stages; }

    // Ensure divisibility for final N
    step = lcm_u64(64ull, (uint64_t)(pl.N / 2 ? (pl.N / 2) : 1));
    if ((K % step) != 0) K = ((K + step - 1) / step) * step;
    pl.Kbits = (int)K;
    pl.W64 = (int)(K / 64ull);

    // baseShift = (2K)/N
    pl.baseShift = (uint64_t)(2ull * (uint64_t)pl.Kbits) / (uint64_t)pl.N;

    pl.ok = (pl.N >= 2) && (pl.W64 >= 1);
    return pl;
}

// ============================================================================
// Ring ops on raw K-bit words (arrays of length W64)
// ============================================================================
static inline void kbits_zero(uint64_t *x, int W64) { std::memset(x, 0, size_t(W64) * 8); }
static inline bool kbits_is_zero(const uint64_t *x, int W64) {
    for (int i = 0; i < W64; ++i) if (x[i]) return false; return true;
}

static inline void kbits_copy(uint64_t *dst, const uint64_t *src, int W64) {
    std::memcpy(dst, src, size_t(W64) * 8);
}

// K-bit increment by one (wrap)
static inline void add1_kbits(uint64_t *x, int W64) {
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
static inline void add_mod(const uint64_t *a, const uint64_t *b, uint64_t *out, int W64) {
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
    }
}


// out = -a mod (2^K + 1). Convention: 0 -> 0; else (~a)+2 within K bits
static inline void neg_mod(const uint64_t *a, uint64_t *out, int W64) {
    if (kbits_is_zero(a, W64)) { kbits_zero(out, W64); return; }
    for (int i = 0; i < W64; ++i) out[i] = ~a[i];
    add1_kbits(out, W64);
    add1_kbits(out, W64);
}

// K-bit subtraction modulo (2^K + 1):
// out = a - b (mod 2^K+1).
// Compute K-bit diff s with borrow d. Then out = s + d (i.e., add 1 if underflow).
static inline void sub_mod(const uint64_t *a, const uint64_t *b, uint64_t *out, int W64) {
    uint64_t borrow = 0;
    for (int i = 0; i < W64; ++i) {
        uint64_t ai = a[i], bi = b[i];

        uint64_t s = ai - bi;
        uint64_t d1 = (ai < bi) ? 1ull : 0ull;

        uint64_t s2 = s - borrow;
        uint64_t d2 = (s < borrow) ? 1ull : 0ull;

        out[i] = s2;
        borrow = (d1 | d2);
    }

    if (borrow) {
        // add 1 modulo 2^K
        uint64_t carry = 1ull;
        for (int i = 0; i < W64 && carry; ++i) {
            uint64_t v = out[i];
            uint64_t s = v + carry;
            out[i] = s;
            carry = (s < v) ? 1ull : 0ull;
        }
    }
}

// K-bit left shift by r (0<=r<K)
static inline void shl_kbits(const uint64_t *x, uint32_t r, uint64_t *out, int W64) {
    if (r == 0) { kbits_copy(out, x, W64); return; }
    uint32_t ws = r >> 6, bs = r & 63u;
    for (int i = W64 - 1; i >= 0; --i) {
        uint64_t lo = 0, hi = 0;
        int s = i - (int)ws;
        if (s >= 0) {
            lo = x[s] << bs;
            if (bs && s > 0) hi = (x[s - 1] >> (64 - bs));
        }
        out[i] = lo | hi;
    }
}

// K-bit right shift by r (0<=r<K)
static inline void shr_kbits(const uint64_t *x, uint32_t r, uint64_t *out, int W64) {
    if (r == 0) { kbits_copy(out, x, W64); return; }
    uint32_t ws = r >> 6, bs = r & 63u;
    for (int i = 0; i < W64; ++i) {
        uint64_t lo = 0, hi = 0;
        uint32_t s = i + ws;
        if (s < (uint32_t)W64) {
            lo = x[s] >> bs;
            if (bs && (s + 1) < (uint32_t)W64) hi = (x[s + 1] << (64 - bs));
        }
        out[i] = lo | hi;
    }
}

// t = x * 2^s in Z/(2^K+1), where 2^K ≡ −1 (negacyclic wrap).
// For 0<r<K: x*2^r = (x<<r) − (x>>(K−r)); if s crosses multiples of K, flip sign.
static inline void mul_pow2(const uint64_t *x, uint64_t s, int Kbits,
    uint64_t *t, uint64_t *tmpA, uint64_t *tmpB, int W64) {
    if (kbits_is_zero(x, W64)) { kbits_zero(t, W64); return; }
    const uint64_t twoK = 2ull * (uint64_t)Kbits;
    s %= twoK;
    uint64_t q = (Kbits ? s / (uint64_t)Kbits : 0ull) & 1ull;
    uint32_t r = (uint32_t)(Kbits ? (s % (uint64_t)Kbits) : 0ull);

    if (r == 0) {
        kbits_copy(t, x, W64);
    } else {
        shl_kbits(x, r, tmpA, W64);
        shr_kbits(x, (uint32_t)(Kbits - r), tmpB, W64);
        sub_mod(tmpA, tmpB, t, W64);  // (x<<r) - (x>>(K-r))
    }
    if (q) {
        uint64_t *tmp = tmpA;
        neg_mod(t, tmp, W64);
        kbits_copy(t, tmp, W64);
    }
}

// Schoolbook: prod[0..2W-1] = a*b (full 2K-bit)
// prod length is 2*W64 64-bit words
static inline void mul_kbits_schoolbook(const uint64_t *a, const uint64_t *b,
    uint64_t *prod, int W64) {
    std::memset(prod, 0, size_t(2 * W64) * 8);
    for (int i = 0; i < W64; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < W64; ++j) {
            U128 p = mul_64x64_128(a[i], b[j]);
            uint64_t s = prod[i + j] + p.lo;
            uint64_t c1 = (s < prod[i + j]) ? 1ull : 0ull;
            prod[i + j] = s;
            uint64_t s1 = prod[i + j + 1] + p.hi + c1 + carry;
            uint64_t c2 = (s1 < prod[i + j + 1]) ? 1ull : 0ull;
            prod[i + j + 1] = s1;
            carry = c2;
        }
        int k = i + W64 + 1;
        while (carry) {
            uint64_t p = prod[k];
            prod[k] += carry;
            carry = (prod[k] < p) ? 1ull : 0ull;
            ++k;
        }
    }
}

// out = (low - high) mod (2^K+1), where prod = [low(0..W-1) | high(W..2W-1)]
static inline void reduce_mod_2k1(const uint64_t *prod, uint64_t *out, int W64) {
    sub_mod(prod, prod + W64, out, W64);
}

static inline void mul_mod(const uint64_t *a, const uint64_t *b,
    uint64_t *out, uint64_t *prod2W, int W64) {
    mul_kbits_schoolbook(a, b, prod2W, W64);
    reduce_mod_2k1(prod2W, out, W64);
}

// ============================================================================
// FFT over arrays of K-bit coefficients (AoS: element i at base + i*W64)
// ============================================================================
static inline uint64_t *elem_ptr(uint64_t *base, int idx, int W64) { return base + size_t(idx) * size_t(W64); }
static inline const uint64_t *elem_ptr(const uint64_t *base, int idx, int W64) { return base + size_t(idx) * size_t(W64); }

static void bit_reverse_inplace(uint64_t *A, int N, int stages, int W64, uint64_t *tmpSwap) {
    for (int i = 0; i < N; ++i) {
        uint32_t j = reverse_bits32((uint32_t)i, stages);
        if ((uint32_t)j > (uint32_t)i) {
            uint64_t *pi = elem_ptr(A, i, W64);
            uint64_t *pj = elem_ptr(A, j, W64);
            // swap W64 words using tmpSwap
            std::memcpy(tmpSwap, pi, size_t(W64) * 8);
            std::memcpy(pi, pj, size_t(W64) * 8);
            std::memcpy(pj, tmpSwap, size_t(W64) * 8);
        }
    }
}

static void FFT_forward(uint64_t *A, const Plan &pl, uint64_t *scratch64) {
    // scratch64 layout: [tmpA(W) | tmpB(W) | t(W) | sum(W) | diff(W) | swap(W)]
    uint64_t *tmpA = scratch64;
    uint64_t *tmpB = scratch64 + pl.W64;
    uint64_t *t = scratch64 + 2 * pl.W64;
    uint64_t *sum = scratch64 + 3 * pl.W64;
    uint64_t *diff = scratch64 + 4 * pl.W64;
    uint64_t *swp = scratch64 + 5 * pl.W64;

    bit_reverse_inplace(A, pl.N, pl.stages, pl.W64, swp);

    for (int s = 1; s <= pl.stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;
        const uint64_t stride = pl.baseShift * (uint64_t)(pl.N / m);
        for (uint32_t k = 0; k < (uint32_t)pl.N; k += m) {
            uint64_t tw = 0;
            for (uint32_t j = 0; j < half; ++j) {
                uint64_t *U = elem_ptr(A, (int)(k + j), pl.W64);
                uint64_t *V = elem_ptr(A, (int)(k + j + half), pl.W64);

                mul_pow2(V, tw, pl.Kbits, t, tmpA, tmpB, pl.W64); // t = V * w^j
                add_mod(U, t, sum, pl.W64);
                sub_mod(U, t, diff, pl.W64);
                kbits_copy(U, sum, pl.W64);
                kbits_copy(V, diff, pl.W64);

                tw += stride;
            }
        }
    }
}

static void FFT_inverse(uint64_t *A, const Plan &pl, uint64_t *scratch64,
    const uint64_t *Ninv, uint64_t *prod2W) {
    // scratch64 layout: [tmpA(W) | tmpB(W) | t(W) | sum(W) | diff(W) | swap(W)]
    uint64_t *tmpA = scratch64;
    uint64_t *tmpB = scratch64 + pl.W64;
    uint64_t *t = scratch64 + 2 * pl.W64;
    uint64_t *sum = scratch64 + 3 * pl.W64;
    uint64_t *diff = scratch64 + 4 * pl.W64;
    uint64_t *swp = scratch64 + 5 * pl.W64;

    bit_reverse_inplace(A, pl.N, pl.stages, pl.W64, swp);

    for (int s = 1; s <= pl.stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;
        const uint64_t stride = pl.baseShift * (uint64_t)(pl.N / m);
        for (uint32_t k = 0; k < (uint32_t)pl.N; k += m) {
            uint64_t tw = 0;
            for (uint32_t j = 0; j < half; ++j) {
                uint64_t *U = elem_ptr(A, (int)(k + j), pl.W64);
                uint64_t *V = elem_ptr(A, (int)(k + j + half), pl.W64);

                // conjugate twiddle: w^{-j} = w^{(2K - j)}
                mul_pow2(V, (uint64_t)(2 * pl.Kbits) - tw, pl.Kbits, t, tmpA, tmpB, pl.W64);
                add_mod(U, t, sum, pl.W64);
                sub_mod(U, t, diff, pl.W64);
                kbits_copy(U, sum, pl.W64);
                kbits_copy(V, diff, pl.W64);

                tw += stride;
            }
        }
    }
    // Scale by N^{-1}
    for (int i = 0; i < pl.N; ++i) {
        uint64_t *Xi = elem_ptr(A, i, pl.W64);
        mul_mod(Xi, Ninv, Xi, prod2W, pl.W64);
    }
}

// ============================================================================
// Pack / Unpack and finalization
// ============================================================================

// Helper: read a b-bit window starting at absolute bit index q from (Digits[], Exponent)
// This applies the inverse of the normalization shift implicitly.
template<class P>
static uint64_t read_bits_with_exponent(const HpSharkFloat<P> *X, int64_t q, int b) {
    // Effective bit index inside the mantissa = q + shiftBack
    // shiftBack is the inverse of the left-shift used at construction.
    // It places the value’s LSB at absolute bit 0.
    //
    // Derivation: the implementation stores mantissa shifted up; its exponent tracks the binary point.
    // Using the same logic that prints your “Shifted data: … shiftBits: S”, we need to shift BACK by S.
    // That “S” is (DefaultPrecBits - LowPrec) - 1 - (X->Exponent_lowlevel_mapping).
    // Rather than rely on private details, we compute S from the visible pair (Digits, Exponent):
    // Find where the value’s bit 0 sits relative to the mantissa by combining Exponent with the mantissa’s placement.
    //
    // Practically: compute the bit index of the mantissa’s MSB (mm) and the value’s MSB (vv).
    // The inverse shift is: shiftBack = vv - mm. For integers, vv = floor(log2(value)).
    // Because we don’t want to touch GMP here, we scan the mantissa to get mm and use X->Exponent to get vv.

    const int B = P::GlobalNumUint32 * 32;  // mantissa width in bits
    // mm: MSB index in the mantissa (0..B-1)  (count within Digits[] bitstring)
    int msw = -1;
    for (int i = P::GlobalNumUint32 - 1; i >= 0; --i) { if (X->Digits[i]) { msw = i; break; } }
    if (msw < 0) return 0; // zero
#if defined(_MSC_VER)
    unsigned long clz;
    _BitScanReverse(&clz, X->Digits[msw]); // clz in [0..31], position of MSB
    int msb_in_word = (int)clz;
#else
    int msb_in_word = 31 - __builtin_clz(X->Digits[msw]);
#endif
    int mm = msw * 32 + msb_in_word;

    // vv: MSB index of the VALUE (relative to bit 0 of the value)
    // The class maintains: value = mantissa << (E_val_shift), for some E that’s consistent across your prints.
    // From your traces for small integers (7→shift 61, 19→59) and the 32-bit limb layout,
    // the “binary point” is tracked entirely by X->Exponent: the value’s MSB index = mm + X->Exponent_adjust.
    // We only need the *difference* (shiftBack), not vv itself:
    //     shiftBack = - (normalization left-shift) = some linear function of X->Exponent.
    //
    // In practice you already compute “Shift bits: S” when printing.
    // The safest production approach is to expose that exact S (or a helper) and reuse it here.
    // For now we assume a helper GetShiftBitsForValue(X) that returns S (the left-shift at construction).
    int64_t shiftBack = -(int64_t)X->Exponent;    // mantissa bit = value bit - Exponent

    // Effective bit index inside the mantissa:
    int64_t t = q + shiftBack;
    if (t + b <= 0) return 0; // entirely below mantissa
    if (t >= B)     return 0; // entirely above mantissa

    // Gather up to 64 bits starting at t (may straddle words); clip at mantissa edges
    uint64_t v = 0;
    int need = b;
    int64_t bit = t;
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
    return v & ((b == 64) ? ~0ull : ((1ull << b) - 1ull));
}

template<class P>
static void pack_base2b_exponent_aware(const HpSharkFloat<P> *X, const Plan &pl, uint64_t *out /*N*W64*/) {
    std::memset(out, 0, size_t(pl.N) * size_t(pl.W64) * 8);
    for (int i = 0; i < pl.L; ++i) {
        // Read the i-th base-2^b window starting at bit q = i*b of the *value* (not the raw mantissa)
        uint64_t coeff = read_bits_with_exponent(X, /*q=*/(int64_t)i * pl.b, /*b=*/pl.b);
        uint64_t *ci = out + size_t(i) * size_t(pl.W64);
        ci[0] = coeff;                 // low word; higher words remain zero for chosen b
        // (if you later let b > 64, you’d spill into more words)
    }
    // Remaining coeffs [pl.L..pl.N-1] are zero (already zeroed)
}


// Accumulate a 32-bit add into interleaved 128-bit buckets at digit index d
static inline void add32_to_final128(uint64_t *final128 /*len=2*D*/, size_t Ddigits,
    uint32_t d, uint32_t add32) {
    if (d >= Ddigits) return; // safety
    uint64_t &lo = final128[2ull * d + 0];
    uint64_t &hi = final128[2ull * d + 1];
    uint64_t prev = lo;
    lo += (uint64_t)add32;
    hi += (lo < prev) ? 1ull : 0ull;
}

// Unpack coefficients (AoS K-bit) into interleaved final128 buckets
static void unpack_to_final128(const uint64_t *coeffs /*N*W64*/, const Plan &pl,
    uint64_t *final128 /*2*Ddigits*/, size_t Ddigits) {
    std::memset(final128, 0, size_t(2 * Ddigits) * 8);

    for (int i = 0; i < pl.N; ++i) {
        const uint64_t *c = elem_ptr(coeffs, i, pl.W64);
        // skip fast if zero
        bool allz = true; for (int w = 0; w < pl.W64; ++w) if (c[w]) { allz = false; break; }
        if (allz) continue;

        uint64_t baseBits = (uint64_t)i * (uint64_t)pl.b;
        uint32_t baseDigit = (uint32_t)(baseBits >> 5);   // /32
        uint32_t off = (uint32_t)(baseBits & 31u);  // %32

        for (int w = 0; w < pl.W64; ++w) {
            uint64_t v = c[w];
            if (!v) continue;

            uint32_t d = baseDigit + 2u * (uint32_t)w;
            uint64_t low = (off ? (v << off) : v);
            uint64_t high = (off ? (v >> (64 - off)) : 0ull);

            add32_to_final128(final128, Ddigits, d + 0, (uint32_t)(low & 0xffffffffull));
            add32_to_final128(final128, Ddigits, d + 1, (uint32_t)((low >> 32) & 0xffffffffull));
            if (off) add32_to_final128(final128, Ddigits, d + 2, (uint32_t)(high & 0xffffffffull));
        }
    }
}

// Left shift interleaved final128 (lo64,hi64 per digit) by 1 bit
static void final128_shl1(uint64_t *final128, size_t Ddigits) {
    uint64_t carry = 0;
    for (size_t d = 0; d < Ddigits; ++d) {
        uint64_t lo = final128[2 * d + 0];
        uint64_t hi = final128[2 * d + 1];
        uint64_t nlo = (lo << 1) | carry;
        uint64_t nhi = (hi << 1) | (lo >> 63);
        carry = (hi >> 63);
        final128[2 * d + 0] = nlo;
        final128[2 * d + 1] = nhi;
    }
}

// Carry-stream final128 → HpSharkFloat<P>::Digits[0..N32-1]
template<class P>
static void finalize_to_digits(const uint64_t *final128, size_t Ddigits,
    HpSharkFloat<P> *Out) {
    using U64 = uint64_t; using U32 = uint32_t;
    const int OUTN = P::GlobalNumUint32;
    // zero output
    for (int i = 0; i < OUTN; ++i) Out->Digits[i] = 0; // ensure no UB on some compilers
    std::memset(Out->Digits, 0, sizeof(typename HpSharkFloat<P>::DigitType) * OUTN);

    U64 carry_lo = 0, carry_hi = 0;
    int written = 0;

    for (size_t d = 0; d < Ddigits && written < OUTN; ++d) {
        U64 lo = final128[2 * d + 0];
        U64 hi = final128[2 * d + 1];

        U64 s_lo = lo + carry_lo;
        U64 c0 = (s_lo < lo) ? 1ull : 0ull;
        U64 s_hi = hi + carry_hi + c0;

        Out->Digits[written++] = (uint32_t)(s_lo & 0xffffffffull);

        carry_lo = (s_lo >> 32) | (s_hi << 32);
        carry_hi = (s_hi >> 32);
    }
    // flush remaining carry into digits, up to OUTN
    while ((carry_lo || carry_hi) && written < OUTN) {
        Out->Digits[written++] = (uint32_t)(carry_lo & 0xffffffffull);
        U64 nlo = (carry_lo >> 32) | (carry_hi << 32);
        U64 nhi = (carry_hi >> 32);
        carry_lo = nlo; carry_hi = nhi;
    }
}

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Does not allocate; does two passes over Final128 to (1) find 'significant',
// then (2) write the window and set the Karatsuba-style exponent.
// NOTE: Does not set sign; do that at the call site.
template<class P>
static void NormalizeFromFinal128LikeKaratsuba(
    HpSharkFloat<P> &Out,
    const HpSharkFloat<P> &A,
    const HpSharkFloat<P> &B,
    const uint64_t *final128,   // len = 2*Ddigits
    size_t Ddigits,             // number of 32-bit positions represented in Final128
    int32_t additionalFactorOfTwo // 0 for XX/YY; 1 for XY if you did NOT shift digits
) {
    constexpr int N = P::GlobalNumUint32;

    auto pass_once = [&](bool write_window, size_t start, size_t needN) -> std::pair<int, int> {
        uint64_t carry_lo = 0, carry_hi = 0;
        size_t idx = 0;              // index in base-2^32 digits being produced
        int highest_nonzero = -1;    // last non-zero digit index seen
        int out_written = 0;

        // Helper to emit one 32-bit digit (optionally writing into Out.Digits)
        auto emit_digit = [&](uint32_t dig) {
            if (dig != 0) highest_nonzero = (int)idx;
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

        return { highest_nonzero, out_written };
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
    if (shift_digits < 0) shift_digits = 0;

    // Karatsuba-style exponent
    Out.Exponent = /*A.Exponent + B.Exponent +*/ 32 * shift_digits + additionalFactorOfTwo;

    // Pass 2: write exactly N digits starting at shift_digits
    (void)pass_once(/*write_window=*/true, /*start=*/(size_t)shift_digits, /*needN=*/(size_t)N);
}



// ============================================================================
// Public API (drop-in): MultiplyHelperKaratsubaV2 (CPU, single-thread, S–S)
// ============================================================================
template<class SharkFloatParams>
void MultiplyHelperSS(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXX,
    HpSharkFloat<SharkFloatParams> *OutXY,
    HpSharkFloat<SharkFloatParams> *OutYY,
    DebugHostCombo<SharkFloatParams> &debugCombo) {

    using P = SharkFloatParams;
    const Plan pl = build_plan<P>(P::GlobalNumUint32);
    if (!pl.ok) {
        // Produce zeros if planning fails (extremely unlikely with our choices)
        std::memset(OutXX->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        std::memset(OutXY->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        std::memset(OutYY->Digits, 0, sizeof(uint32_t) * P::GlobalNumUint32);
        return;
    }

    // ---- Single arena allocation (64B aligned slices) ----
    const size_t W64 = (size_t)pl.W64;
    const size_t coeffWords = (size_t)pl.N * W64; // words per coeff array
    const size_t Ddigits = ((size_t)pl.N * (size_t)pl.b + 31) / 32; // final128 digits

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "MultiplyHelperSS: " << std::endl;

        pl.print();

        std::cout
            << "W64=" << W64
            << " coeffWords=" << coeffWords
            << " Ddigits=" << Ddigits
            << std::endl;
    }

    // Byte sizes
    const size_t bytesCoeff = coeffWords * sizeof(uint64_t);
    const size_t bytesFinal128 = (2 * Ddigits) * sizeof(uint64_t);
    const size_t bytesProd2K = (2 * W64) * sizeof(uint64_t);
    const size_t bytesNinv = (1 * W64) * sizeof(uint64_t);
    const size_t bytesScratch = (6 * W64) * sizeof(uint64_t); // tmpA,tmpB,t,sum,diff,swap

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout
            << " bytesCoeff=" << bytesCoeff
            << " bytesFinal128=" << bytesFinal128
            << " bytesProd2K=" << bytesProd2K
            << " bytesNinv=" << bytesNinv
            << " bytesScratch=" << bytesScratch
            << std::endl;
    }

    // We keep: Apacked, Bpacked, Awork, Bwork, Ftmp (5 coeff arrays)
    size_t off = 0;
    off = align_up(off, 64); size_t off_Apacked = off; off += bytesCoeff;
    off = align_up(off, 64); size_t off_Bpacked = off; off += bytesCoeff;
    off = align_up(off, 64); size_t off_Awork = off; off += bytesCoeff;
    off = align_up(off, 64); size_t off_Bwork = off; off += bytesCoeff;
    off = align_up(off, 64); size_t off_Ftmp = off; off += bytesCoeff;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout
            << " off_Apacked=" << off_Apacked
            << " off_Bpacked=" << off_Bpacked
            << " off_Awork=" << off_Awork
            << " off_Bwork=" << off_Bwork
            << " off_Ftmp=" << off_Ftmp
            << std::endl;
    }

    off = align_up(off, 64); size_t off_Final128 = off; off += bytesFinal128;

    off = align_up(off, 64); size_t off_Prod2K = off; off += bytesProd2K;
    off = align_up(off, 64); size_t off_Ninv = off; off += bytesNinv;
    off = align_up(off, 64); size_t off_Scratch = off; off += bytesScratch;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout
            << " off_Final128=" << off_Final128
            << " off_Prod2K=" << off_Prod2K
            << " off_Ninv=" << off_Ninv
            << " off_Scratch=" << off_Scratch
            << std::endl;
    }

    const size_t arenaBytes = align_up(off, 64);
    std::unique_ptr<std::byte[]> arena(new std::byte[arenaBytes]);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " arenaBytes=" << arenaBytes << std::endl;
    }

    // Slices
    auto Apacked = reinterpret_cast<uint64_t *>(arena.get() + off_Apacked);
    auto Bpacked = reinterpret_cast<uint64_t *>(arena.get() + off_Bpacked);
    auto Awork = reinterpret_cast<uint64_t *>(arena.get() + off_Awork);
    auto Bwork = reinterpret_cast<uint64_t *>(arena.get() + off_Bwork);
    auto Ftmp = reinterpret_cast<uint64_t *>(arena.get() + off_Ftmp);

    auto Final128 = reinterpret_cast<uint64_t *>(arena.get() + off_Final128);

    auto Prod2K = reinterpret_cast<uint64_t *>(arena.get() + off_Prod2K);
    auto Ninv = reinterpret_cast<uint64_t *>(arena.get() + off_Ninv);
    auto Scratch = reinterpret_cast<uint64_t *>(arena.get() + off_Scratch);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout
            << " Apacked @" << (void *)Apacked
            << " Bpacked @" << (void *)Bpacked
            << " Awork @" << (void *)Awork
            << " Bwork @" << (void *)Bwork
            << " Ftmp @" << (void *)Ftmp
            << std::endl;
        std::cout
            << " Final128 @" << (void *)Final128
            << std::endl;
        std::cout
            << " Prod2K @" << (void *)Prod2K
            << " Ninv @" << (void *)Ninv
            << " Scratch @" << (void *)Scratch
            << std::endl;
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

    // *** Fix: add +1 in the K-bit ring to complete -2^{K-m} ***
    add1_kbits(Ninv, pl.W64);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << " Ninv = ";
        for (int w = 0; w < pl.W64; ++w) {
            std::cout << std::hex << Ninv[w] << " ";
        }
        std::cout << std::dec << std::endl;
    }

    // ---- Pack inputs ----
    pack_base2b_exponent_aware<P>(A, pl, Apacked);
    pack_base2b_exponent_aware<P>(B, pl, Bpacked);

    if (SharkVerbose == VerboseMode::Debug) {
        auto printPacked = [&](const char *name, const uint64_t *Xpacked) {
            std::cout << " " << name << " packed: ";
            for (int i = 0; i < pl.N; ++i) {
                const uint64_t *ci = elem_ptr(Xpacked, i, pl.W64);
                bool allz = true; for (int w = 0; w < pl.W64; ++w) if (ci[w]) { allz = false; break; }
                if (allz) continue;
                std::cout << "[" << i << ":";
                for (int w = 0; w < pl.W64; ++w) {
                    std::cout << std::hex << ci[w] << (w + 1 < pl.W64 ? "," : "");
                }
                std::cout << "] ";
            }
            std::cout << std::dec << std::endl;
        };

        printPacked("A", Apacked);
        printPacked("B", Bpacked);
    }

    // ---- Forward FFT on copies (Awork/Bwork) ----
    std::memcpy(Awork, Apacked, bytesCoeff);
    std::memcpy(Bwork, Bpacked, bytesCoeff);

    FFT_forward(Awork, pl, Scratch);
    FFT_forward(Bwork, pl, Scratch);

    // ============================================================
    // XX = A^2
    // ============================================================
    {
        // Ftmp = elementwise square of Awork
        for (int i = 0; i < pl.N; ++i) {
            const uint64_t *ai = elem_ptr(Awork, i, pl.W64);
            uint64_t *fo = elem_ptr(Ftmp, i, pl.W64);
            mul_mod(ai, ai, fo, Prod2K, pl.W64);
        }
        FFT_inverse(Ftmp, pl, Scratch, Ninv, Prod2K);
        unpack_to_final128(Ftmp, pl, Final128, Ddigits);
        //finalize_to_digits<P>(Final128, Ddigits, OutXX);

        // --- NEW: set sign & exponent like Karatsuba ---
        constexpr auto additionalFactorOfTwoXX = 0;
        OutXX->SetNegative(false);
        NormalizeFromFinal128LikeKaratsuba<P>(*OutXX, *A, *A, Final128, Ddigits, additionalFactorOfTwoXX);
    }

    // ============================================================
    // YY = B^2
    // ============================================================
    {
        for (int i = 0; i < pl.N; ++i) {
            const uint64_t *bi = elem_ptr(Bwork, i, pl.W64);
            uint64_t *fo = elem_ptr(Ftmp, i, pl.W64);
            mul_mod(bi, bi, fo, Prod2K, pl.W64);
        }
        FFT_inverse(Ftmp, pl, Scratch, Ninv, Prod2K);
        unpack_to_final128(Ftmp, pl, Final128, Ddigits);
        //finalize_to_digits<P>(Final128, Ddigits, OutYY);

        constexpr auto additionalFactorOfTwoYY = 0;
        OutYY->SetNegative(false);
        NormalizeFromFinal128LikeKaratsuba<P>(*OutYY, *B, *B, Final128, Ddigits, additionalFactorOfTwoYY);
    }

    // ============================================================
    // XY = 2*(A*B)
    // ============================================================
    {
        for (int i = 0; i < pl.N; ++i) {
            const uint64_t *ai = elem_ptr(Awork, i, pl.W64);
            const uint64_t *bi = elem_ptr(Bwork, i, pl.W64);
            uint64_t *fo = elem_ptr(Ftmp, i, pl.W64);
            mul_mod(ai, bi, fo, Prod2K, pl.W64);
        }
        FFT_inverse(Ftmp, pl, Scratch, Ninv, Prod2K);
        unpack_to_final128(Ftmp, pl, Final128, Ddigits);
        //final128_shl1(Final128, Ddigits);               // ×2
        //finalize_to_digits<P>(Final128, Ddigits, OutXY);

        constexpr auto additionalFactorOfTwoXY = 1;
        OutXY->SetNegative(A->GetNegative() ^ B->GetNegative());
        NormalizeFromFinal128LikeKaratsuba<P>(*OutXY, *A, *B, Final128, Ddigits, additionalFactorOfTwoXY);
    }
}


#define ExplicitlyInstantiate(SharkFloatParams) \
    template void MultiplyHelperSS<SharkFloatParams>( \
        const HpSharkFloat<SharkFloatParams> *, \
        const HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        HpSharkFloat<SharkFloatParams> *, \
        DebugHostCombo<SharkFloatParams> &debugHostCombo);

ExplicitInstantiateAll();