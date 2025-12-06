/*
  This currently doesn't work.  Idea is to look into an
  alternative approach staying in standard domain instead of Montgomery.

  To use set:
  - static constexpr bool TestGpu = false;
  - static constexpr bool TestReferenceImpl = true;
  - Ifdef out the other reference implementation of NTT and enable this one.
  - Debug mode not release
  - Also fix the plan builder and BuildRoots.  Those are not set up to
    work with this idea and this implementation is definitely busted without
    changes there.
  */

#if 0
#include "DbgHeap.h"

#include <algorithm>
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
#include <type_traits>

#include "DebugChecksumHost.h"
#include "HpSharkFloat.cuh"
#include "MultiplyNTTPlanBuilder.cuh"
#include "NTTConstexprGenerator.h"
#include "TestVerbose.h"

namespace {

    //--------------------------------------------------------------------------------------------------
    // Bit utilities
    //--------------------------------------------------------------------------------------------------

    // Reverse the lowest `bit_count` bits of a 32-bit value (manual since MSVC lacks
    // __builtin_bitreverse32).
    static inline uint32_t ReverseBits32(uint32_t value, int bit_count)
{
    value = (value >> 16) | (value << 16);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);

    const int shift = 32 - bit_count;
    return value >> shift;
}

    
template <class SharkFloatParams, DebugStatePurpose Purpose, typename ArrayType>
const DebugStateHost<SharkFloatParams> &
GetCurrentDebugState(std::vector<DebugStateHost<SharkFloatParams>> &debugStates,
                     const ArrayType *arrayToChecksum,
                     size_t arraySize)
{
    constexpr auto curPurpose = static_cast<int>(Purpose);
    constexpr auto CallIndex = 0;
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolution = UseConvolution::No;

    auto &retval = debugStates[curPurpose];
    retval.Reset(arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex, UseConvolution);
    return retval;
}


static void
BitReverseInplace64(uint64_t *A, uint32_t N, uint32_t stages)
{
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t j = ReverseBits32(i, stages) & (N - 1);
        if (j > i)
            std::swap(A[i], A[j]);
    }
}

//--------------------------------------------------------------------------------------------------
// 64-bit Goldilocks prime, helpers, and modular arithmetic
//--------------------------------------------------------------------------------------------------

#include "NTTConstexprGenerator.h"

static inline uint64_t
AddP(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    if (s < a || s >= SharkNTT::MagicPrime)
        s -= SharkNTT::MagicPrime;
    return s;
}

static inline uint64_t
SubP(uint64_t a, uint64_t b)
{
    return (a >= b) ? (a - b) : (a + SharkNTT::MagicPrime - b);
}

// Fast modular multiply in the standard domain for the Goldilocks prime
// p = 2^64 - 2^32 + 1. This uses a Solinas-style reduction on the 128-bit
// product of a and b. Inputs must already be reduced into [0, p).
static inline uint64_t
MulP(uint64_t a, uint64_t b)
{
    constexpr uint64_t P = SharkNTT::MagicPrime; // 0xFFFFFFFF00000001
    constexpr uint64_t MASK32 = 0xFFFFFFFFull;

    // Compute 128-bit product a*b = (hi:lo)
    uint64_t lo, hi;
#if defined(_MSC_VER)
    lo = _umul128(a, b, &hi); // MSVC intrinsic: lo = a*b, hi = high 64 bits
#else
    __uint128_t z = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
    lo = static_cast<uint64_t>(z);
    hi = static_cast<uint64_t>(z >> 64);
#endif

    // Decompose z = 2^96 A + 2^64 B + 2^32 C + D
    const uint64_t A = hi >> 32;
    const uint64_t B = hi & MASK32;
    const uint64_t C = lo >> 32;
    const uint64_t D = lo & MASK32;

    // Target: r ≡ 2^32 (B + C) - A - B + D (mod P)

    const uint64_t s = B + C; // <= 2^33 - 2
    const uint64_t s_lo = s & MASK32;
    const uint64_t s_hi = s >> 32; // 0 or 1

    // Start with 2^32 * s_lo + D
    uint64_t r = (s_lo << 32) + D;

    // Add contribution of 2^64 * s_hi using 2^64 ≡ 2^32 - 1 (mod P):
    //   2^64 * s_hi ≡ (2^32 - 1) * s_hi = (s_hi << 32) - s_hi
    if (s_hi) {
        r += (uint64_t(1) << 32);
        r -= 1u;
    }

    // Subtract A + B with modular correction
    const uint64_t ab = A + B;
    if (r >= ab) {
        r -= ab;
    } else {
        r = (r + P) - ab;
    }

    if (r >= P)
        r -= P;

    return r;
}

// Simple modular exponentiation in the standard domain, using MulP.
static uint64_t
PowP(uint64_t base, uint64_t exp)
{
    constexpr uint64_t P = SharkNTT::MagicPrime;
    base %= P;
    uint64_t result = 1ull;

    while (exp) {
        if (exp & 1ull)
            result = MulP(result, base);
        base = MulP(base, base);
        exp >>= 1ull;
    }
    return result;
}

// Runtime generator search (bounded). Used only for debug prints; correctness is guarded by table
// checks.
template <class SharkFloatParams>
static uint64_t
FindGenerator()
{
    using namespace SharkNTT;
    // p-1 = 2^32 * (2^32 - 1), with distinct prime factors:
    // {2, 3, 5, 17, 257, 65537}
    static const uint64_t factors[] = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};

    for (uint64_t g = 3; g < 1'000; ++g) {
        uint64_t g_std = g % MagicPrime;
        bool ok = true;
        for (uint64_t q : factors) {
            // if g^{(p-1)/q} == 1, then g is NOT a generator
            uint64_t t = PowP(g_std, PHI / q);
            if (t == 1ull) {
                ok = false;
                break;
            }
        }
        if (ok)
            return g_std; // found a primitive root in standard domain
    }
    // Fallback (shouldn't happen): 7 is often fine, but order-checks will catch issues.
    return 7ull % MagicPrime;
}

//--------------------------------------------------------------------------------------------------
// Diagnostics
//--------------------------------------------------------------------------------------------------

template <typename SharkFloatParams>
bool
VerifyMontgomeryConstants(uint64_t Pval, uint64_t NINVval, uint64_t R2val)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== Enhanced Montgomery Verification ===\n";
        std::cout << "SharkNTT::MagicPrime   = 0x" << std::hex << Pval << std::dec << "\n";
        std::cout << "NINV= 0x" << std::hex << NINVval << std::dec << "\n";
        std::cout << "R2  = 0x" << std::hex << R2val << std::dec << "\n";

        // Test 1: Basic Montgomery property (SharkNTT::MagicPrime * NINV ≡ 2^64-1 mod 2^64)
        uint64_t test1 = Pval * NINVval; // unsigned wrap is modulo 2^64
        std::cout << "SharkNTT::MagicPrime * NINV = 0x" << std::hex << test1 << std::dec;
        if (test1 == 0xFFFFFFFFFFFFFFFFull) {
            std::cout << "  CORRECT\n";
        } else {
            std::cout << "  INCORRECT (should be 0xFFFFFFFFFFFFFFFF)\n";
            return false;
        }

        // Test 2: Montgomery multiplication of 1
        uint64_t one_m = SharkNTT::ToMontgomery<SharkFloatParams>(1);
        std::cout << "ToMontgomery(1) = 0x" << std::hex << one_m << std::dec << "\n";

        uint64_t back1 = SharkNTT::FromMontgomery<SharkFloatParams>(one_m);
        std::cout << "FromMontgomery(ToMontgomery(1)) = " << back1 << "\n";
        if (back1 == 1ull) {
            std::cout << "Roundtrip(1) CORRECT\n";
        } else {
            std::cout << "Roundtrip(1) INCORRECT\n";
            return false;
        }

        // Test 3: Roundtrip of a random-ish value
        uint64_t val = 0x123456789ABCDEFull;
        uint64_t val_m = SharkNTT::ToMontgomery<SharkFloatParams>(val);
        uint64_t back_val = SharkNTT::FromMontgomery<SharkFloatParams>(val_m);
        std::cout << "Roundtrip(val) -> " << back_val << "\n";
        if (back_val == (val % Pval)) {
            std::cout << "Roundtrip(val) CORRECT\n";
        } else {
            std::cout << "Roundtrip(val) INCORRECT\n";
            return false;
        }

        std::cout << "========================================\n";
    }
    return true;
}

template <class SharkFloatParams>
static void
VerifyNTTRoots(const SharkNTT::RootTables &roots, const SharkNTT::PlanPrime &plan)
{
    using namespace SharkNTT;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "  Generator g = 0x" << std::hex << FindGenerator<SharkFloatParams>() << std::dec
                  << "\n";
        uint64_t psi = roots.psi_pows[1];
        std::cout << "  psi = 0x" << std::hex << psi << std::dec << " (order " << (2 * plan.N) << ")\n";

        uint64_t omega = MulP(psi, psi);
        std::cout << "  omega = 0x" << std::hex << omega << std::dec << " (order " << plan.N << ")\n";
    }

    const uint64_t one = 1ull;

    // ψ^(2N) = 1
    uint64_t psi = roots.psi_pows[1];
    uint64_t psi2N = PowP(psi, 2ull * (uint64_t)plan.N);
    assert(psi2N == one && "psi^(2N) != 1 (check N and roots)");

    // ω = ψ^2, check ω^N = 1
    uint64_t omega = MulP(psi, psi);
    uint64_t omegaN = PowP(omega, (uint64_t)plan.N);
    assert(omegaN == one && "omega^N != 1 (check N and roots)");
}

template <class SharkFloatParams>
static void
ConvertRootsToStandard(SharkNTT::RootTables &roots, const SharkNTT::PlanPrime &plan)
{
    using namespace SharkNTT;

    // psi and psi^{-1} powers: length N
    for (int i = 0; i < plan.N; ++i) {
        roots.psi_pows[(size_t)i] = FromMontgomery<SharkFloatParams>(roots.psi_pows[(size_t)i]);
        roots.psi_inv_pows[(size_t)i] = FromMontgomery<SharkFloatParams>(roots.psi_inv_pows[(size_t)i]);
    }

    // Stage omegas
    for (int s = 0; s < plan.stages; ++s) {
        roots.stage_omegas[(size_t)s] = FromMontgomery<SharkFloatParams>(roots.stage_omegas[(size_t)s]);
        roots.stage_omegas_inv[(size_t)s] =
            FromMontgomery<SharkFloatParams>(roots.stage_omegas_inv[(size_t)s]);
    }

    // Twiddle tables: length >= N (convert first N entries, which are all that NTTRadix2 touches)
    for (int i = 0; i < plan.N; ++i) {
        roots.stage_twiddles_fwd[(size_t)i] =
            FromMontgomery<SharkFloatParams>(roots.stage_twiddles_fwd[(size_t)i]);
        roots.stage_twiddles_inv[(size_t)i] =
            FromMontgomery<SharkFloatParams>(roots.stage_twiddles_inv[(size_t)i]);
    }

    // N^{-1} (previously Ninvm_mont) into standard domain
    roots.Ninvm_mont = FromMontgomery<SharkFloatParams>(roots.Ninvm_mont);
}

//--------------------------------------------------------------------------------------------------
// Iterative radix-2 NTT (Cooley–Tukey) over standard domain (Goldilocks)
//--------------------------------------------------------------------------------------------------

template <class SharkFloatParams, bool inverse>
static void
NTTRadix2(DebugHostCombo<SharkFloatParams> & /*debugCombo*/,
          uint64_t *A,
          uint32_t N,
          uint32_t stages,
          SharkNTT::RootTables &rootTables)
{
    uint64_t *stage_omegas;
    uint64_t *stage_twiddles;

    if constexpr (inverse) {
        stage_omegas = rootTables.stage_omegas_inv;
        stage_twiddles = rootTables.stage_twiddles_inv;
    } else {
        stage_omegas = rootTables.stage_omegas;
        stage_twiddles = rootTables.stage_twiddles_fwd;
    }

    for (uint32_t s = 1; s <= stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;

        const uint64_t w_m = stage_omegas[s - 1];

        const uint32_t numTwid = (1u << (s - 1));
        const uint32_t tw_base = numTwid - 1u;

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half; ++j) {
                const uint32_t i0 = k + j;
                const uint32_t i1 = i0 + half;

                const uint64_t U = A[i0];
                const uint64_t V = A[i1];

                const uint64_t w = stage_twiddles[tw_base + j]; // fwd or inv

                // Optional: stage-local cross-check (only in debug), now in standard domain.
                {
                    static thread_local bool checked_stage[64] = {false};
                    if (!checked_stage[s - 1] && j + 1 == half) {
                        uint64_t w_inc = 1ull;
                        for (uint32_t jj = 0; jj < half; ++jj) {
                            const uint64_t w_tab = stage_twiddles[tw_base + jj];
                            if (w_tab != w_inc) {
                                std::cerr << (inverse ? "Inv" : "Fwd") << " twiddle mismatch at stage "
                                          << s << " j=" << jj << std::endl;
                                break;
                            }
                            w_inc = MulP(w_inc, w_m);
                        }
                        checked_stage[s - 1] = true;
                    }
                }

                const uint64_t t = MulP(V, w);
                A[i0] = AddP(U, t);
                A[i1] = SubP(U, t);
            }
        }
    }
}

//==================================================================================================
//                       Pack (base-2^b) and Unpack (to Final128)
//==================================================================================================

template <class SharkFloatParams>
[[nodiscard]] static uint64_t
ReadBitsSimple(const HpSharkFloat<SharkFloatParams> &X, int64_t q, int b)
{
    const int B = SharkFloatParams::GlobalNumUint32 * 32;
    if (q >= B || q < 0)
        return 0;

    uint64_t v = 0;
    int need = b;
    int outPos = 0;
    int64_t bit = q;

    while (need > 0 && bit < B) {
        const int32_t digit_index = static_cast<int32_t>(bit / 32);
        const int shift = static_cast<int>(bit % 32);

        const uint32_t digit = X.Digits[digit_index];

        const int take = std::min(need, 32 - shift);
        const uint32_t mask = (take == 32) ? 0xFFFFFFFFu : ((1u << take) - 1u);

        const uint32_t chunk = (digit >> shift) & mask;
        v |= (static_cast<uint64_t>(chunk) << outPos);

        need -= take;
        outPos += take;
        bit += take;
    }

    return v;
}

template <class SharkFloatParams>
static void
PackBase2bPrime_NoAlloc(DebugHostCombo<SharkFloatParams> & /*debugCombo*/,
                        const HpSharkFloat<SharkFloatParams> &Xsrc,
                        const SharkNTT::PlanPrime &plan,
                        std::span<uint64_t> outMod) // len >= plan.N
{
    using namespace SharkNTT;
    const uint64_t zero = 0ull;
    for (int i = 0; i < plan.N; ++i)
        outMod[(size_t)i] = zero;

    for (int i = 0; i < plan.L; ++i) {
        const uint64_t coeff = ReadBitsSimple(Xsrc, (int64_t)i * plan.b, plan.b);
        const uint64_t cmod = coeff % SharkNTT::MagicPrime;
        outMod[(size_t)i] = cmod; // standard-domain coefficient
    }
}

static void
UnpackPrimeToFinal128(const uint64_t *A_norm,
                      const SharkNTT::PlanPrime &plan,
                      uint64_t *Final128,
                      size_t Ddigits)
{
    using namespace SharkNTT;
    std::memset(Final128, 0, sizeof(uint64_t) * 2 * Ddigits);

    const uint64_t HALF = (SharkNTT::MagicPrime - 1ull) >> 1;
    const int Imax = std::min(plan.N, 2 * plan.L - 1);

    for (int i = 0; i < Imax; ++i) {
        uint64_t v = A_norm[i];
        if (!v)
            continue;

        bool neg = (v > HALF);
        uint64_t mag64 = neg ? (SharkNTT::MagicPrime - v) : v;

        const uint64_t sBits = (uint64_t)i * (uint64_t)plan.b;
        const uint64_t word_index = sBits >> 6;
        const uint64_t bit_offset = sBits & 63ull;

        uint64_t *p = Final128 + word_index;
        *p += (mag64 << bit_offset);

        if (bit_offset > 0) {
            uint64_t hiPart = mag64 >> (64ull - bit_offset);
            if (hiPart)
                *(p + 1) += hiPart;
        }

        if (neg) {
            // Two's complement in-place: FINAL128 -= mag64 * 2^{sBits}.
            uint64_t *q = Final128 + word_index;
            uint64_t borrow = 0;
            uint64_t sub = mag64 << bit_offset;
            uint64_t old = *q;
            uint64_t res = old - sub;
            borrow = (old < sub);
            *q = res;

            if (bit_offset > 0) {
                ++q;
                uint64_t sub_hi = mag64 >> (64ull - bit_offset);
                old = *q;
                res = old - sub_hi - borrow;
                borrow = (old < sub_hi + borrow);
                *q = res;
            }

            while (borrow && ++q < Final128 + 2 * Ddigits) {
                old = *q;
                res = old - 1;
                borrow = (old == 0);
                *q = res;
            }
        }
    }
}

//==================================================================================================
//                         Normalize Final128 -> HpSharkFloat
//==================================================================================================

template <class SharkFloatParams>
static void
DigitNormalizeFromFinal128(const uint64_t *final128,
                           size_t Ddigits,
                           std::vector<uint32_t> &digitsOut,
                           int &highest_nonzero)
{
    using u32 = uint32_t;
    using u64 = uint64_t;

    constexpr int N = SharkFloatParams::GlobalNumUint32;

    digitsOut.clear();
    digitsOut.reserve(Ddigits + 4);

    u64 carry_lo = 0;
    u64 carry_hi = 0;
    highest_nonzero = -1;

    const u64 *p = final128;
    //const u64 *end = final128 + 2 * Ddigits;

    for (size_t i = 0; i < Ddigits; ++i, p += 2) {
        const u64 lo = p[0];
        const u64 hi = p[1];

        // ----- low 64 bits + carry_lo -----
        u64 digit0 = lo + carry_lo;
        // carry if overflow occurred
        u64 new_carry_lo = (digit0 < carry_lo) ? 1u : 0u;
        carry_lo = new_carry_lo;

        // ----- high 64 bits + carry_hi -----
        u64 digit1 = hi + carry_hi;
        u64 new_carry_hi = (digit1 < carry_hi) ? 1u : 0u;
        carry_hi = new_carry_hi;

        // Re-pack into 32-bit digits
        u64 merged = (digit1 << 32) | (digit0 >> 32);

        u32 outDigit = static_cast<u32>(digit0 & 0xFFFFFFFFu);
        digitsOut.push_back(outDigit);
        if (outDigit != 0)
            highest_nonzero = static_cast<int>(digitsOut.size()) - 1;

        digitsOut.push_back(static_cast<u32>(merged & 0xFFFFFFFFu));
        if (digitsOut.back() != 0)
            highest_nonzero = static_cast<int>(digitsOut.size()) - 1;
    }

    while (digitsOut.size() < N)
        digitsOut.push_back(0);

    for (size_t i = N; i < digitsOut.size(); ++i) {
        if (digitsOut[i] != 0)
            highest_nonzero = static_cast<int>(i);
    }

    if (highest_nonzero >= N)
        highest_nonzero = N - 1;
}

template <class SharkFloatParams>
static void
NormalizeCombineExponents(const HpSharkFloat<SharkFloatParams> &a,
                          const HpSharkFloat<SharkFloatParams> &b,
                          const std::vector<uint32_t> &digits,
                          int highest_nonzero,
                          int additionalFactorOfTwo,
                          HpSharkFloat<SharkFloatParams> &out)
{
    using u32 = uint32_t;

    constexpr int N = SharkFloatParams::GlobalNumUint32;

    if (highest_nonzero < 0) {
        std::fill_n(out.Digits, N, u32{0});
        out.Exponent = a.Exponent + b.Exponent;
        return;
    }

    const int significant = highest_nonzero + 1;
    int shift_digits = significant - N;
    if (shift_digits < 0)
        shift_digits = 0;

    out.Exponent = a.Exponent + b.Exponent + 32 * shift_digits + additionalFactorOfTwo;

    const size_t start = static_cast<size_t>(shift_digits);
    const size_t total = digits.size();
    const size_t copyCount = std::min<size_t>(N, total - start);

    for (size_t i = 0; i < copyCount; ++i)
        out.Digits[i] = digits[start + i];
    for (size_t i = copyCount; i < (size_t)N; ++i)
        out.Digits[i] = 0;
}

template <class SharkFloatParams>
static void
Normalize(HpSharkFloat<SharkFloatParams> &out,
          const HpSharkFloat<SharkFloatParams> &a,
          const HpSharkFloat<SharkFloatParams> &b,
          const uint64_t *Final128,
          size_t Ddigits,
          int additionalFactorOfTwo)
{
    std::vector<uint32_t> digits;
    int highest_nonzero = -1;

    DigitNormalizeFromFinal128<SharkFloatParams>(Final128, Ddigits, digits, highest_nonzero);
    NormalizeCombineExponents<SharkFloatParams>(
        a, b, digits, highest_nonzero, additionalFactorOfTwo, out);
}


} // namespace

//==================================================================================================
//                          MultiplyHelperFFT2 (prime backend)
//==================================================================================================

template <class SharkFloatParams>
void
MultiplyHelperFFT2(const HpSharkFloat<SharkFloatParams> *A,
                   const HpSharkFloat<SharkFloatParams> *B,
                   HpSharkFloat<SharkFloatParams> *OutXX,
                   HpSharkFloat<SharkFloatParams> *OutXY,
                   HpSharkFloat<SharkFloatParams> *OutYY,
                   DebugHostCombo<SharkFloatParams> &debugCombo)
{
    using namespace SharkNTT;

    auto &debugStates = debugCombo.States;

    if constexpr (HpShark::DebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // --------- Plan and tables ---------
    PlanPrime plan = BuildPlanPrime(SharkFloatParams::GlobalNumUint32, /*b_hint=*/26, /*margin=*/2);
    plan.Print();

    assert(plan.ok && "Prime plan build failed (check b/N headroom constraints)");
    assert(plan.N >= 2 * plan.L && "No-wrap condition violated: need N >= 2*L");
    assert((PHI % (2ull * (uint64_t)plan.N)) == 0ull);

    if constexpr (HpShark::DebugChecksums) {
        const auto &debugAState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, A->Digits, SharkFloatParams::GlobalNumUint32);
        const auto &debugBState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, B->Digits, SharkFloatParams::GlobalNumUint32);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "debugAState checksum: " << debugAState.GetStr() << "\n";
            std::cout << "debugBState checksum: " << debugBState.GetStr() << "\n";
        }
    }

    const int Ddigits = ((uint64_t)((2 * plan.L - 2) * plan.b + 64) + 31u) / 32u + 2u;
    const size_t totalU64 = (size_t)plan.N * 2 + (size_t)2 * Ddigits;

    std::unique_ptr<uint64_t[]> buffer = std::make_unique<uint64_t[]>(totalU64);

    size_t off = 0;
    std::span<uint64_t> X{buffer.get() + off, (size_t)plan.N};
    off += (size_t)plan.N;
    std::span<uint64_t> Y{buffer.get() + off, (size_t)plan.N};
    off += (size_t)plan.N;
    std::span<uint64_t> Final128{buffer.get() + off, (size_t)2 * Ddigits};
    off += (size_t)2 * Ddigits;

    RootTables roots;
    BuildRoots<SharkFloatParams>(plan.N, plan.stages, roots);

    // Convert all root tables from Montgomery to standard domain for the Goldilocks field.
    ConvertRootsToStandard<SharkFloatParams>(roots, plan);

    // Verify constants and roots (optional but helpful in debug builds)
    [[maybe_unused]] const bool constants_ok = VerifyMontgomeryConstants<SharkFloatParams>(
        SharkNTT::MagicPrime, SharkNTT::MagicPrimeInv, SharkNTT::R2);
    assert(constants_ok && "Montgomery constants verification failed");
    VerifyNTTRoots<SharkFloatParams>(roots, plan);

    auto run_conv = [&]<DebugStatePurpose step1X,
                        DebugStatePurpose step1Y,
                        DebugStatePurpose step2X,
                        DebugStatePurpose step2Y,
                        DebugStatePurpose step3,
                        DebugStatePurpose step4,
                        DebugStatePurpose step5>(HpSharkFloat<SharkFloatParams> *out,
                                                 const HpSharkFloat<SharkFloatParams> &inA,
                                                 const HpSharkFloat<SharkFloatParams> &inB,
                                                 int addFactorOfTwo,
                                                 bool isNegative) {
        // ============================
        // HOT PATH (single allocation)
        // ============================

        // 1) Pack (into X and Y)
        PackBase2bPrime_NoAlloc(debugCombo, inA, plan, X);
        PackBase2bPrime_NoAlloc(debugCombo, inB, plan, Y);

        // 2) Twist by ψ^i (standard domain)
        for (int i = 0; i < plan.N; ++i) {
            X[(size_t)i] = MulP(X[(size_t)i], roots.psi_pows[(size_t)i]);
        }

        for (int i = 0; i < plan.N; ++i) {
            Y[(size_t)i] = MulP(Y[(size_t)i], roots.psi_pows[(size_t)i]);
        }

        if constexpr (HpShark::DebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step1X>(debugStates, X.data(), (size_t)plan.N);
            const auto &debugYState =
                GetCurrentDebugState<SharkFloatParams, step1Y>(debugStates, Y.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After twist, debugXState checksum: " << debugXState.GetStr() << "\n";
                std::cout << "After twist, debugYState checksum: " << debugYState.GetStr() << "\n";
            }
        }

        // 3) Forward NTT (in place)
        BitReverseInplace64(X.data(), (uint32_t)plan.N, (uint32_t)plan.stages);
        BitReverseInplace64(Y.data(), (uint32_t)plan.N, (uint32_t)plan.stages);

        NTTRadix2<SharkFloatParams, false>(
            debugCombo, X.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots);
        NTTRadix2<SharkFloatParams, false>(
            debugCombo, Y.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots);

        if constexpr (HpShark::DebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step2X>(debugStates, X.data(), (size_t)plan.N);
            const auto &debugYState =
                GetCurrentDebugState<SharkFloatParams, step2Y>(debugStates, Y.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After NTT, debugXState checksum: " << debugXState.GetStr() << "\n";
                std::cout << "After NTT, debugYState checksum: " << debugYState.GetStr() << "\n";
            }
        }

        // 4) Pointwise multiply (X *= Y)
        for (int i = 0; i < plan.N; ++i)
            X[(size_t)i] = MulP(X[(size_t)i], Y[(size_t)i]);

        // 5) Inverse NTT (in place on X)
        BitReverseInplace64(X.data(), (uint32_t)plan.N, (uint32_t)plan.stages);

        if constexpr (HpShark::DebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step3>(debugStates, X.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After bit reverse, debugXState checksum: " << debugXState.GetStr() << "\n";
            }
        }

        NTTRadix2<SharkFloatParams, true>(
            debugCombo, X.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots);

        // 6) Untwist + scale by N^{-1} (write back into X), still in standard domain
        for (int i = 0; i < plan.N; ++i) {
            // uint64_t v = MulP(X[(size_t)i], roots.psi_inv_pows[(size_t)i]);
            // X[(size_t)i] = MulP(v, roots.Ninvm_mont);

            // After BuildRoots + ConvertRootsToStandard + VerifyNTTRoots:
            const uint64_t N_mod_p = (uint64_t)plan.N % SharkNTT::MagicPrime;

            // Fermat: x^(p-2) ≡ x^{-1} mod p
            const uint64_t Ninv_std = PowP(N_mod_p, SharkNTT::MagicPrime - 2ull);

            uint64_t v = MulP(X[(size_t)i], roots.psi_inv_pows[(size_t)i]);
            X[(size_t)i] = MulP(v, Ninv_std);

        }

        // 7) Unpack -> Final128 -> Normalize (already in standard domain)
        // std::fill_n(Final128.data(), Final128.size(), 0ull);

        if constexpr (HpShark::DebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step4>(debugStates, X.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After untwist, debugXState checksum: " << debugXState.GetStr() << "\n";
            }
        }

        UnpackPrimeToFinal128(X.data(), plan, Final128.data(), Ddigits);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << std::endl;
            std::cout << "=== Final128 Dump ===" << std::endl;
            for (size_t i = 0; i < Ddigits; ++i) {
                std::cout << " 0x" << std::hex << Final128[2 * i + 0] << " 0x" << Final128[2 * i + 1]
                          << " " << std::dec;
            }

            std::cout << std::endl;
        }

        out->SetNegative(isNegative);
        Normalize<SharkFloatParams>(*out, inA, inB, Final128.data(), Ddigits, addFactorOfTwo);

        if constexpr (HpShark::DebugChecksums) {
            const auto &debugXState = GetCurrentDebugState<SharkFloatParams, step5>(
                debugStates, out->Digits, SharkFloatParams::GlobalNumUint32);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After normalize, debugXState checksum: " << debugXState.GetStr() << "\n";
            }
        }
    };

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "=== MultiplyHelperFFT2 Debug Output ===" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "XX:" << std::endl;
        std::cout << "==============================" << std::endl;
    }

    const auto noAdditionalFactorOfTwo = 0;
    const auto squaresNegative = false;
    run_conv.template operator()<DebugStatePurpose::Z0XX,
                                 DebugStatePurpose::Z1XX,
                                 DebugStatePurpose::Z2XX,
                                 DebugStatePurpose::Z3XX,
                                 DebugStatePurpose::Z2_Perm1,
                                 DebugStatePurpose::Z2_Perm4,
                                 DebugStatePurpose::Final128XX>(
        OutXX, *A, *A, noAdditionalFactorOfTwo, squaresNegative);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "==============================" << std::endl;
        std::cout << "YY:" << std::endl;
        std::cout << "==============================" << std::endl;
    }

    run_conv.template operator()<DebugStatePurpose::Z0YY,
                                 DebugStatePurpose::Z1YY,
                                 DebugStatePurpose::Z2YY,
                                 DebugStatePurpose::Z3YY,
                                 DebugStatePurpose::Z2_Perm2,
                                 DebugStatePurpose::Z2_Perm5,
                                 DebugStatePurpose::Final128YY>(
        OutYY, *B, *B, noAdditionalFactorOfTwo, squaresNegative);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "==============================" << std::endl;
        std::cout << "XY:" << std::endl;
        std::cout << "==============================" << std::endl;
    }

    const auto includeAdditionalFactorOfTwo = 1;
    const auto OutXYIsNegative = (A->GetNegative() ^ B->GetNegative());
    run_conv.template operator()<DebugStatePurpose::Z0XY,
                                 DebugStatePurpose::Z1XY,
                                 DebugStatePurpose::Z2XY,
                                 DebugStatePurpose::Z3XY,
                                 DebugStatePurpose::Z2_Perm3,
                                 DebugStatePurpose::Z2_Perm6,
                                 DebugStatePurpose::Final128XY>(
        OutXY, *A, *B, includeAdditionalFactorOfTwo, OutXYIsNegative);

    SharkNTT::DestroyRoots<SharkFloatParams>(false, roots);
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void MultiplyHelperFFT2<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *,          \
                                                       const HpSharkFloat<SharkFloatParams> *,          \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       DebugHostCombo<SharkFloatParams> &);

ExplicitInstantiateAll();


#endif 