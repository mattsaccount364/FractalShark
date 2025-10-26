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

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "DebugChecksumHost.h"
#include "HpSharkFloat.cuh" // your header (provides HpSharkFloat<>, DebugStateHost<>)
#include "MultiplyNTTCudaSetup.h"
#include "NTTConstexprGenerator.h"
#include "ReferenceNTT2.h"
#include "TestVerbose.h"

namespace SharkNTT {

//--------------------------------------------------------------------------------------------------
// Bit utilities
//--------------------------------------------------------------------------------------------------

// Reverse the lowest `bit_count` bits of a 32-bit value (manual since MSVC lacks
// __builtin_bitreverse32).
static inline uint32_t
ReverseBits32(uint32_t value, int bit_count)
{
    value = (value >> 16) | (value << 16);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);
    return value >> (32 - bit_count);
}

//--------------------------------------------------------------------------------------------------
// Normalization (final carry propagation and windowing into HpSharkFloat<SharkFloatParams>)
//--------------------------------------------------------------------------------------------------

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Does not allocate; does two passes over Final128 to (1) find 'significant',
// then (2) write the window and set the Karatsuba-style exponent.
// NOTE: Does not set sign; do that at the call site.
template <class SharkFloatParams>
static void
Normalize(HpSharkFloat<SharkFloatParams> &out,
          const HpSharkFloat<SharkFloatParams> &a,
          const HpSharkFloat<SharkFloatParams> &b,
          const uint64_t *final128,     // len = 2*Ddigits
          size_t Ddigits,               // number of 32-bit positions represented in final128
          int32_t additionalFactorOfTwo // 0 for XX/YY; 1 for XY if you did NOT shift digits
)
{
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    auto pass_once = [&](bool write_window, size_t start, size_t needN) -> std::pair<int, int> {
        uint64_t carry_lo = 0, carry_hi = 0;
        size_t idx = 0;           // index in base-2^32 digits being produced
        int highest_nonzero = -1; // last non-zero digit index seen
        int out_written = 0;

        // Helper to emit one 32-bit digit (optionally writing into out.Digits)
        auto emit_digit = [&](uint32_t dig) {
            if (dig != 0)
                highest_nonzero = (int)idx;
            if (write_window && idx >= start && out_written < (int)needN) {
                out.Digits[out_written++] = dig;
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
                out.Digits[out_written++] = 0u;
            }
        }

        return {highest_nonzero, out_written};
    };

    // Pass 1: find 'significant' (last non-zero + 1)
    auto [highest_nonzero, _] = pass_once(/*write_window=*/false, /*start=*/0, /*needN=*/0);
    if (highest_nonzero < 0) {
        // Zero result
        std::memset(out.Digits, 0, N * sizeof(uint32_t));
        out.Exponent = a.Exponent + b.Exponent; // Karatsuba zero convention
        return;
    }
    const int significant = highest_nonzero + 1;
    int shift_digits = significant - N;
    if (shift_digits < 0)
        shift_digits = 0;

    // Karatsuba-style exponent
    out.Exponent = a.Exponent + b.Exponent + 32 * shift_digits + additionalFactorOfTwo;

    // Pass 2: write exactly N digits starting at shift_digits
    (void)pass_once(/*write_window=*/true, /*start=*/(size_t)shift_digits, /*needN=*/(size_t)N);
}

//--------------------------------------------------------------------------------------------------
// 64-bit Goldilocks prime, helpers, and Montgomery arithmetic
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
        uint64_t g_m = ToMontgomery<SharkFloatParams>(g);
        bool ok = true;
        for (uint64_t q : factors) {
            // if g^{(p-1)/q} == 1, then g is NOT a generator
            uint64_t t = MontgomeryPow<SharkFloatParams>(g_m, PHI / q);
            if (t == ToMontgomery<SharkFloatParams>(1)) {
                ok = false;
                break;
            }
        }
        if (ok)
            return g; // found a primitive root
    }
    // Fallback (shouldn't happen): 7 is often fine, but order-checks will catch issues.
    return 7ull;
}

//--------------------------------------------------------------------------------------------------
// In-place bit-reversal permutation on Montgomery residues (64-bit words)
//--------------------------------------------------------------------------------------------------

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
// Iterative radix-2 NTT (Cooley–Tukey) over Montgomery domain
//--------------------------------------------------------------------------------------------------

template <class SharkFloatParams>
static void
NTTRadix2(DebugHostCombo<SharkFloatParams> &debugCombo,
          uint64_t *A,
          uint32_t N,
          uint32_t stages,
          const uint64_t *stage_base)
{
    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint64_t w_m = stage_base[s - 1];
        for (uint32_t k = 0; k < N; k += m) {
            uint64_t w = ToMontgomery(debugCombo, 1);
            for (uint32_t j = 0; j < half; ++j) {
                uint64_t U = A[k + j];
                uint64_t V = A[k + j + half];
                uint64_t t = MontgomeryMul(debugCombo, V, w);
                A[k + j] = AddP(U, t);
                A[k + j + half] = SubP(U, t);
                w = MontgomeryMul(debugCombo, w, w_m);
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
        int64_t w = bit / 32;
        int off = (int)(bit % 32);
        uint32_t limb = (w >= 0) ? X.Digits[(int)w] : 0u;
        uint32_t chunk = (off ? (limb >> off) : limb);
        int take = std::min(32 - off, need);

        v |= (uint64_t)(chunk & ((take == 32) ? 0xFFFFFFFFu : ((1u << take) - 1u))) << outPos;

        outPos += take;
        need -= take;
        bit += take;
    }
    return (b == 64) ? v : (v & ((1ull << b) - 1ull));
}

template <class SharkFloatParams>
static void
PackBase2bPrime_NoAlloc(DebugHostCombo<SharkFloatParams> &debugCombo,
                        const HpSharkFloat<SharkFloatParams> &Xsrc,
                        const PlanPrime &plan,
                        std::span<uint64_t> outMont) // len >= plan.N
{
    using namespace SharkNTT;
    const uint64_t zero_m = ToMontgomery(debugCombo, 0);
    for (int i = 0; i < plan.N; ++i)
        outMont[(size_t)i] = zero_m;

    for (int i = 0; i < plan.L; ++i) {
        const uint64_t coeff = ReadBitsSimple(Xsrc, (int64_t)i * plan.b, plan.b);
        const uint64_t cmod = coeff % SharkNTT::MagicPrime;
        outMont[(size_t)i] = ToMontgomery(debugCombo, cmod);
    }
}

static void
UnpackPrimeToFinal128(const uint64_t *A_norm, const PlanPrime &plan, uint64_t *Final128, size_t Ddigits)
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
        const size_t q = (size_t)(sBits >> 5);
        const int r = (int)(sBits & 31);

        const uint64_t lo64 = (r ? (mag64 << r) : mag64);
        const uint64_t hi64 = (r ? (mag64 >> (64 - r)) : 0ull);

        uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu);
        uint32_t d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu);
        uint32_t d2 = (uint32_t)(hi64 & 0xffffffffu);
        uint32_t d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);

        auto add32 = [&](size_t j, uint32_t add) {
            if (!add)
                return;
            uint64_t &lo = Final128[2 * j + 0];
            uint64_t &hi = Final128[2 * j + 1];
            uint64_t old = lo;
            lo += (uint64_t)add;
            if (lo < old)
                hi += 1ull;
        };
        auto sub32 = [&](size_t j, uint32_t sub) {
            if (!sub)
                return;
            uint64_t &lo = Final128[2 * j + 0];
            uint64_t &hi = Final128[2 * j + 1];
            uint64_t old = lo;
            uint64_t dif = old - (uint64_t)sub;
            lo = dif;
            if (old < (uint64_t)sub)
                hi -= 1ull;
        };

        if (!neg) {
            add32(q + 0, d0);
            add32(q + 1, d1);
            add32(q + 2, d2);
            add32(q + 3, d3);
        } else {
            sub32(q + 0, d0);
            sub32(q + 1, d1);
            sub32(q + 2, d2);
            sub32(q + 3, d3);
        }
    }
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

        uint64_t one_squared = SharkNTT::MontgomeryMul<SharkFloatParams>(one_m, one_m);
        std::cout << "MontgomeryMul(ToMontgomery(1), ToMontgomery(1)) = 0x" << std::hex << one_squared
                  << std::dec << "\n";

        uint64_t back_to_one = SharkNTT::FromMontgomery<SharkFloatParams>(one_squared);
        std::cout << "FromMontgomery(result) = " << std::hex << back_to_one << std::dec;
        if (back_to_one == 1) {
            std::cout << "  CORRECT\n";
        } else {
            std::cout << "  INCORRECT\n";
            return false;
        }

        // Test 3: Verify R2 is correct (MontgomeryMul(1, R2) == ToMontgomery(1))
        uint64_t mont_1 = SharkNTT::MontgomeryMul<SharkFloatParams>(1, R2val);
        if (mont_1 == one_m) {
            std::cout << "R2 verification: CORRECT\n";
        } else {
            std::cout << "R2 verification: INCORRECT\n";
            std::cout << "  MontgomeryMul(1, R2) = 0x" << std::hex << mont_1 << std::dec << "\n";
            std::cout << "  ToMontgomery(1)      = 0x" << std::hex << one_m << std::dec << "\n";
            return false;
        }
    }

    return true;
}

template <class SharkFloatParams>
static void
VerifyNTTRoots(const SharkNTT::RootTables &roots, const PlanPrime &plan)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "  NINV = 0x" << std::hex << SharkNTT::MagicPrimeInv << std::dec << "\n";
        std::cout << "  R2   = 0x" << std::hex << SharkNTT::R2 << std::dec << "\n";
        std::cout << "  Generator g = 0x" << std::hex << FindGenerator<SharkFloatParams>() << std::dec
                  << "\n";
        std::cout << "  psi = 0x" << std::hex << FromMontgomery<SharkFloatParams>(roots.psi_pows[1])
                  << std::dec << " (order " << (2 * plan.N) << ")\n";
        std::cout << "  omega = 0x" << std::hex
                  << FromMontgomery<SharkFloatParams>(
                         MontgomeryMul<SharkFloatParams>(roots.psi_pows[1], roots.psi_pows[1]))
                  << std::dec << " (order " << plan.N << ")\n";
    }

    // Use the table's own 1 in Montgomery domain
    [[maybe_unused]] const uint64_t one_m = roots.psi_pows[0];

    // ψ is roots.psi_pows[1] (Montgomery). Check ψ^(2N) = 1
    [[maybe_unused]] uint64_t psi2N =
        SharkNTT::MontgomeryPow<SharkFloatParams>(roots.psi_pows[1], 2ull * (uint64_t)plan.N);
    assert(psi2N == one_m && "psi^(2N) != 1 (check N and roots)");

    // ω = ψ^2. Check ω^N = 1
    [[maybe_unused]] uint64_t omega =
        SharkNTT::MontgomeryMul<SharkFloatParams>(roots.psi_pows[1], roots.psi_pows[1]);
    [[maybe_unused]] uint64_t omegaN =
        SharkNTT::MontgomeryPow<SharkFloatParams>(omega, (uint64_t)plan.N);
    assert(omegaN == one_m && "omega^N != 1 (check N and roots)");
}

} // namespace SharkNTT

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

    if constexpr (SharkDebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // x must be a positive constant expression

    // Verify power of 2
    static_assert(SharkFloatParams::GlobalNumUint32 > 0 &&
                      (SharkFloatParams::GlobalNumUint32 & (SharkFloatParams::GlobalNumUint32 - 1)) == 0,
                  "GlobalNumUint32 must be a power of 2");

    // --------- Plan and tables ---------
    PlanPrime plan = BuildPlanPrime(SharkFloatParams::GlobalNumUint32, /*b_hint=*/26, /*margin=*/2);
    plan.Print();

    assert(plan.ok && "Prime plan build failed (check b/N headroom constraints)");
    assert(plan.N >= 2 * plan.L && "No-wrap condition violated: need N >= 2*L");
    assert((PHI % (2ull * (uint64_t)plan.N)) == 0ull);

    if constexpr (SharkDebugChecksums) {
        const auto &debugAState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, A->Digits, SharkFloatParams::GlobalNumUint32);
        const auto &debugBState = GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, B->Digits, SharkFloatParams::GlobalNumUint32);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "debugAState checksum: " << debugAState.GetStr() << "\n";
            std::cout << "debugBState checksum: " << debugBState.GetStr() << "\n";
        }
    }

    // Compute Final128 digit budget once
    const uint32_t Ddigits = ((uint64_t)((2 * plan.L - 2) * plan.b + 64) + 31u) / 32u + 2u;

    // ---- Single allocation for entire core path ----
    const size_t buf_count = (size_t)2 * (size_t)plan.N     // X + Y
                             + (size_t)2 * (size_t)Ddigits; // Final128 (lo,hi per 32-bit slot)
    std::unique_ptr<uint64_t[]> buffer(new uint64_t[buf_count]);

    // Slice buffer into spans
    size_t off = 0;
    std::span<uint64_t> X{buffer.get() + off, (size_t)plan.N};
    off += (size_t)plan.N;
    std::span<uint64_t> Y{buffer.get() + off, (size_t)plan.N};
    off += (size_t)plan.N;
    std::span<uint64_t> Final128{buffer.get() + off, (size_t)2 * Ddigits};
    off += (size_t)2 * Ddigits;

    RootTables roots;
    BuildRoots<SharkFloatParams>(plan.N, plan.stages, roots);

    // Verify constants and roots
    [[maybe_unused]] const bool constants_ok = VerifyMontgomeryConstants<SharkFloatParams>(
        SharkNTT::MagicPrime, SharkNTT::MagicPrimeInv, SharkNTT::R2);
    assert(constants_ok && "Montgomery constants verification failed");
    VerifyNTTRoots<SharkFloatParams>(roots, plan);

    auto run_conv = [&]<DebugStatePurpose step1X,
                        DebugStatePurpose step1Y,
                        DebugStatePurpose step2X,
                        DebugStatePurpose step2Y>(HpSharkFloat<SharkFloatParams> *out,
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

        if constexpr (SharkDebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step1X>(debugStates, X.data(), (size_t)plan.N);
            const auto &debugYState =
                GetCurrentDebugState<SharkFloatParams, step1Y>(debugStates, Y.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After twist, debugXState checksum: " << debugXState.GetStr() << "\n";
                std::cout << "After twist, debugYState checksum: " << debugYState.GetStr() << "\n";
            }
        }

        // 2) Twist by ψ^i
        for (int i = 0; i < plan.N; ++i) {
            X[(size_t)i] = MontgomeryMul(debugCombo, X[(size_t)i], roots.psi_pows[(size_t)i]);
        }

        for (int i = 0; i < plan.N; ++i) {
            Y[(size_t)i] = MontgomeryMul(debugCombo, Y[(size_t)i], roots.psi_pows[(size_t)i]);
        }

        if constexpr (SharkDebugChecksums) {
            const auto &debugXState =
                GetCurrentDebugState<SharkFloatParams, step2X>(debugStates, X.data(), (size_t)plan.N);
            const auto &debugYState =
                GetCurrentDebugState<SharkFloatParams, step2Y>(debugStates, Y.data(), (size_t)plan.N);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "After NTT, debugXState checksum: " << debugXState.GetStr() << "\n";
                std::cout << "After NTT, debugYState checksum: " << debugYState.GetStr() << "\n";
            }
        }

        // 3) Forward NTT (in place)
        BitReverseInplace64(X.data(), (uint32_t)plan.N, (uint32_t)plan.stages);
        BitReverseInplace64(Y.data(), (uint32_t)plan.N, (uint32_t)plan.stages);

        NTTRadix2(debugCombo, X.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots.stage_omegas);
        NTTRadix2(debugCombo, Y.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots.stage_omegas);

        // 4) Pointwise multiply (X *= Y)
        for (int i = 0; i < plan.N; ++i)
            X[(size_t)i] = MontgomeryMul(debugCombo, X[(size_t)i], Y[(size_t)i]);

        // 5) Inverse NTT (in place on X)
        BitReverseInplace64(X.data(), (uint32_t)plan.N, (uint32_t)plan.stages);
        NTTRadix2(debugCombo, X.data(), (uint32_t)plan.N, (uint32_t)plan.stages, roots.stage_omegas_inv);

        // 6) Untwist + scale by N^{-1} (write back into X)
        for (int i = 0; i < plan.N; ++i) {
            uint64_t v = MontgomeryMul(debugCombo, X[(size_t)i], roots.psi_inv_pows[(size_t)i]);
            X[(size_t)i] = MontgomeryMul(debugCombo, v, roots.Ninvm_mont);
        }

        // 7) Convert out of Montgomery: reuse Y as normal-domain buffer
        for (int i = 0; i < plan.N; ++i)
            Y[(size_t)i] = FromMontgomery(debugCombo, X[(size_t)i]);

        // 8) Unpack -> Final128 -> Normalize
        // std::fill_n(Final128.data(), Final128.size(), 0ull);
        UnpackPrimeToFinal128(Y.data(), plan, Final128.data(), Ddigits);
        out->SetNegative(isNegative);
        Normalize<SharkFloatParams>(*out, inA, inB, Final128.data(), Ddigits, addFactorOfTwo);
    };

    // XX = A^2
    const auto noAdditionalFactorOfTwo = 0;
    const auto squaresNegative = false;
    run_conv.template operator()<DebugStatePurpose::Z0XX,
                                 DebugStatePurpose::Z1XX,
                                 DebugStatePurpose::Z2XX,
                                 DebugStatePurpose::Z3XX>(
        OutXX, *A, *A, noAdditionalFactorOfTwo, squaresNegative);

    // YY = B^2
    run_conv.template operator()<DebugStatePurpose::Z0YY,
                                 DebugStatePurpose::Z1YY,
                                 DebugStatePurpose::Z2YY,
                                 DebugStatePurpose::Z3YY>(
        OutYY, *B, *B, noAdditionalFactorOfTwo, squaresNegative);

    // XY = 2*(A*B)
    const auto includeAdditionalFactorOfTwo = 1;
    const auto OutXYIsNegative = (A->GetNegative() ^ B->GetNegative());
    run_conv.template operator()<DebugStatePurpose::Z0XY,
                                 DebugStatePurpose::Z1XY,
                                 DebugStatePurpose::Z2XY,
                                 DebugStatePurpose::Z3XY>(
        OutXY, *A, *B, includeAdditionalFactorOfTwo, OutXYIsNegative);

    DestroyRoots<SharkFloatParams>(false, roots);
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void MultiplyHelperFFT2<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *,          \
                                                       const HpSharkFloat<SharkFloatParams> *,          \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       HpSharkFloat<SharkFloatParams> *,                \
                                                       DebugHostCombo<SharkFloatParams> &);

ExplicitInstantiateAll();
