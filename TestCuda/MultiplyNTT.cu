#include "MultiplyNTT.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "BenchmarkTimer.h"
#include "DebugChecksum.cuh"
#include "HpSharkFloat.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <span>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "NTTConstexprGenerator.h"

//#define TEMP_DISABLEALL

namespace cg = cooperative_groups;

static constexpr auto [[maybe_unused]]
CalcAlign16Bytes64BitIndex(uint64_t Sixty4BitIndex)
{
    return Sixty4BitIndex % 2 == 0 ? 0 : 1;
}

static constexpr auto [[maybe_unused]]
CalcAlign16Bytes32BitIndex(uint64_t Thirty2BitIndex)
{
    return 4 - (Thirty2BitIndex % 4);
}

#ifndef TEMP_DISABLEALL

namespace SharkNTT {

//--------------------------------------------------------------------------------------------------
// Bit utilities
//--------------------------------------------------------------------------------------------------

// Reverse the lowest `bit_count` bits of a 32-bit value (manual since MSVC lacks
// __builtin_bitreverse32).
static __device__ SharkForceInlineReleaseOnly uint32_t
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
static __device__ void
Normalize(HpSharkFloat<SharkFloatParams>& out,
          const HpSharkFloat<SharkFloatParams>& a,
          const HpSharkFloat<SharkFloatParams>& b,
          const uint64_t* final128,     // len = 2*Ddigits
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
// 64×64→128 helpers (compiler/ABI specific intrinsics)
//--------------------------------------------------------------------------------------------------

static __device__ SharkForceInlineReleaseOnly void
Mul64Wide(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
    lo = a * b;
    hi = __umul64hi(a, b);
}

static __device__ SharkForceInlineReleaseOnly uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t& carry)
{
    uint64_t s = a + b;
    uint64_t c = (s < a);
    uint64_t out = s + carry;
    carry = c | (out < s);
    return out;
}

//--------------------------------------------------------------------------------------------------
// Prime field ops + Montgomery core
//--------------------------------------------------------------------------------------------------

static __device__ SharkForceInlineReleaseOnly uint64_t
AddP(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    if (s < a || s >= SharkNTT::MagicPrime)
        s -= SharkNTT::MagicPrime;
    return s;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
SubP(uint64_t a, uint64_t b)
{
    return (a >= b) ? (a - b) : (a + SharkNTT::MagicPrime - b);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
MontgomeryMul(cooperative_groups::grid_group& grid,
              cooperative_groups::thread_block& block,
              DebugMultiplyCount<SharkFloatParams>* debugCombo,
              uint64_t a,
              uint64_t b)
{
    // We'll count 128-bit multiplications here as 3x64  (so 3 + 3 + 1)
    if constexpr (SharkPrintMultiplyCounts) {
        DebugMultiplyIncrement<SharkFloatParams>(
            debugCombo, grid, block, 7);
    }

    // t = a*b (128-bit)
    uint64_t t_lo, t_hi;
    Mul64Wide(a, b, t_lo, t_hi);

    // m = (t_lo * NINV) mod 2^64
    uint64_t m = t_lo * SharkNTT::MagicPrimeInv;

    // m*SharkNTT::MagicPrime (128-bit)
    uint64_t mp_lo, mp_hi;
    Mul64Wide(m, SharkNTT::MagicPrime, mp_lo, mp_hi);

    // u = t + m*SharkNTT::MagicPrime
    // low 64 + carry0
    uint64_t carry0 = 0;
    (void)Add64WithCarry(t_lo, mp_lo, carry0); // updates carry0

    // high 64 + carry0  -> also track carry-out (carry1)
    uint64_t carry1 = carry0;
    uint64_t u_hi = Add64WithCarry(t_hi, mp_hi, carry1); // returns sum, updates carry1

    // r = u / 2^64; ensure r < SharkNTT::MagicPrime (include the high-limb carry-out)
    uint64_t r = u_hi;
    if (carry1 || r >= SharkNTT::MagicPrime)
        r -= SharkNTT::MagicPrime;

    return r;
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
ToMontgomery(cooperative_groups::grid_group& grid,
             cooperative_groups::thread_block& block,
             DebugMultiplyCount<SharkFloatParams>* debugCombo,
             uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, R2);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
FromMontgomery(cooperative_groups::grid_group& grid,
               cooperative_groups::thread_block& block,
               DebugMultiplyCount<SharkFloatParams>* debugCombo,
               uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, 1);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
MontgomeryPow(cooperative_groups::grid_group& grid,
              cooperative_groups::thread_block& block,
              DebugMultiplyCount<SharkFloatParams>* debugCombo,
              uint64_t a_mont,
              uint64_t e)
{
    uint64_t x = ToMontgomery(grid, block, debugCombo, 1);
    uint64_t y = a_mont;
    while (e) {
        if (e & 1)
            x = MontgomeryMul(grid, block, debugCombo, x, y);
        y = MontgomeryMul(grid, block, debugCombo, y, y);
        e >>= 1;
    }
    return x;
}

// Runtime generator search (bounded). Used only for debug prints; correctness is guarded by table
// checks.
template <class SharkFloatParams>
static __device__ uint64_t
FindGenerator(DebugMultiplyCount<SharkFloatParams>* debugCombo)
{
    using namespace SharkNTT;
    // p-1 = 2^32 * (2^32 - 1), with distinct prime factors:
    // {2, 3, 5, 17, 257, 65537}
    static const uint64_t factors[] = {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull};

    for (uint64_t g = 3; g < 1'000; ++g) {
        uint64_t g_m = ToMontgomery(debugCombo, g);
        bool ok = true;
        for (uint64_t q : factors) {
            // if g^{(p-1)/q} == 1, then g is NOT a generator
            uint64_t t = MontgomeryPow(debugCombo, g_m, PHI / q);
            if (t == ToMontgomery(debugCombo, 1)) {
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

static __device__ void
BitReverseInplace64(uint64_t* A, uint32_t N, uint32_t stages)
{
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t j = ReverseBits32(i, stages) & (N - 1);
        if (j > i) {
            // std::swap(A[i], A[j]);
            uint64_t t = A[i];
            A[i] = A[j];
            A[j] = t;
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Iterative radix-2 NTT (Cooley–Tukey) over Montgomery domain
//--------------------------------------------------------------------------------------------------

template <class SharkFloatParams>
static __device__ void
NTTRadix2(cooperative_groups::grid_group& grid,
          cooperative_groups::thread_block& block,
          DebugMultiplyCount<SharkFloatParams>* debugCombo,
          uint64_t* A,
          uint32_t N,
          uint32_t stages,
          const uint64_t *stage_base)
{
    for (uint32_t s = 1; s <= stages; ++s) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint64_t w_m = stage_base[s - 1];
        for (uint32_t k = 0; k < N; k += m) {
            uint64_t w = ToMontgomery(grid, block, debugCombo, 1);
            for (uint32_t j = 0; j < half; ++j) {
                uint64_t U = A[k + j];
                uint64_t V = A[k + j + half];
                uint64_t t = MontgomeryMul(grid, block, debugCombo, V, w);
                A[k + j] = AddP(U, t);
                A[k + j + half] = SubP(U, t);
                w = MontgomeryMul(grid, block, debugCombo, w, w_m);
            }
        }
    }
}

// Grid-stride Cooley–Tukey radix-2 NTT (in-place), with the inner w-update
// identical in structure to the original (w starts at 1 and is multiplied by w_m each j).
// Parallelizes over k-blocks; each thread processes full butterflies for its k.
template <class SharkFloatParams>
static __device__ void
NTTRadix2_GridStride(cooperative_groups::grid_group& grid,
                     cooperative_groups::thread_block& block,
                     DebugMultiplyCount<SharkFloatParams>* debugCombo,
                     uint64_t* SharkRestrict A,
                     uint32_t N,
                     uint32_t stages,
                     const uint64_t* SharkRestrict stage_base)
{
    const uint64_t one_m = ToMontgomery(grid, block, debugCombo, 1ull);

    for (uint32_t s = 1; s <= stages; ++s) {
        const uint32_t m = 1u << s;             // span
        const uint32_t half = m >> 1;           // butterflies per span
        const uint64_t w_m = stage_base[s - 1]; // stage increment (Montgomery)
        const uint32_t nblocks = N / m;         // number of k-blocks this stage

        // Grid-stride over k-blocks; each thread does the full inner j-loop for its k.
        const size_t gsize = grid.size();
        const size_t grank = grid.thread_rank();
        for (size_t blk = grank; blk < nblocks; blk += gsize) {
            const uint32_t k = static_cast<uint32_t>(blk) * m;

            // ---- Inner loop identical to original: w increments by w_m each j ----
            uint64_t w = one_m;
            for (uint32_t j = 0; j < half; ++j) {
                const uint32_t i0 = k + j;
                const uint32_t i1 = i0 + half;

                const uint64_t U = A[i0];
                const uint64_t V = A[i1];
                const uint64_t t = MontgomeryMul(grid, block, debugCombo, V, w);

                A[i0] = AddP(U, t);
                A[i1] = SubP(U, t);

                w = MontgomeryMul(grid, block, debugCombo, w, w_m);
            }
        }

        // All butterflies for this stage must complete before advancing to next stage
        grid.sync();
    }
}



//==================================================================================================
//                       Pack (base-2^b) and Unpack (to Final128)
//==================================================================================================

template <class SharkFloatParams>
[[nodiscard]] static __device__ uint64_t
ReadBitsSimple(const HpSharkFloat<SharkFloatParams> &Z0_OutDigits, int64_t q, int b)
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
        uint32_t limb = (w >= 0) ? Z0_OutDigits.Digits[(int)w] : 0u;
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
static __device__ void
PackBase2bPrime_NoAlloc(cooperative_groups::grid_group& grid,
                        cooperative_groups::thread_block& block,
                        DebugMultiplyCount<SharkFloatParams>* debugCombo,
                        const HpSharkFloat<SharkFloatParams> &Xsrc,
                        const PlanPrime& plan,
                        uint64_t *outMont) // len >= plan.N
{
    using namespace SharkNTT;
    const uint64_t zero_m = ToMontgomery(grid, block, debugCombo, 0);
    for (int i = 0; i < plan.N; ++i)
        outMont[(size_t)i] = zero_m;

    for (int i = 0; i < plan.L; ++i) {
        const uint64_t coeff = ReadBitsSimple(Xsrc, (int64_t)i * plan.b, plan.b);
        const uint64_t cmod = coeff % SharkNTT::MagicPrime;
        outMont[(size_t)i] = ToMontgomery(grid, block, debugCombo, cmod);
    }
}

static __device__ void
UnpackPrimeToFinal128(const uint64_t* A_norm, const PlanPrime& plan, uint64_t* Final128, size_t Ddigits)
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
            uint64_t& lo = Final128[2 * j + 0];
            uint64_t& hi = Final128[2 * j + 1];
            uint64_t old = lo;
            lo += (uint64_t)add;
            if (lo < old)
                hi += 1ull;
        };
        auto sub32 = [&](size_t j, uint32_t sub) {
            if (!sub)
                return;
            uint64_t& lo = Final128[2 * j + 0];
            uint64_t& hi = Final128[2 * j + 1];
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

// Grid-strided version: minimize distinct loops, add grid.sync between phases.
// A once -> (XX1, XX2, XY1), then B once -> (YY1, YY2, XY2)
template <class SharkFloatParams>
static __device__ inline void
PackTwistFwdNTT_Fused_AB_ToSixOutputs(cooperative_groups::grid_group& grid,
                                      cooperative_groups::thread_block& block,
                                      DebugMultiplyCount<SharkFloatParams>* debugMultiplyCounts,
                                      const HpSharkFloat<SharkFloatParams>& inA,
                                      const HpSharkFloat<SharkFloatParams>& inB,
                                      const SharkNTT::PlanPrime& plan,
                                      const SharkNTT::RootTables& roots,
                                      // six outputs (Montgomery domain, length plan.N)
                                      uint64_t* SharkRestrict tempDigitsXX1,
                                      uint64_t* SharkRestrict tempDigitsXX2,
                                      uint64_t* SharkRestrict tempDigitsYY1,
                                      uint64_t* SharkRestrict tempDigitsYY2,
                                      uint64_t* SharkRestrict tempDigitsXY1,
                                      uint64_t* SharkRestrict tempDigitsXY2)
{
    const uint32_t N = static_cast<uint32_t>(plan.N);
    const uint32_t L = static_cast<uint32_t>(plan.L);
    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    const uint64_t zero_m = ToMontgomery(grid, block, debugMultiplyCounts, 0ull);

    // -------------------- Phase A: pack+twist with tail zero (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        if (i < L) {
            const uint64_t coeff = ReadBitsSimple(inA, (int64_t)i * plan.b, plan.b);
            const uint64_t cmod = coeff % MagicPrime; // match original
            const uint64_t xm = ToMontgomery(grid, block, debugMultiplyCounts, cmod);
            const uint64_t psik = roots.psi_pows[i]; // Montgomery domain
            tempDigitsXX1[i] = MontgomeryMul(grid, block, debugMultiplyCounts, xm, psik);
        } else {
            tempDigitsXX1[i] = zero_m;
        }
    }
    grid.sync();

    // A: forward NTT (grid-wide helpers)
    BitReverseInplace64(tempDigitsXX1, N, (uint32_t)plan.stages);
    NTTRadix2_GridStride(
        grid, block, debugMultiplyCounts, tempDigitsXX1, N, (uint32_t)plan.stages, roots.stage_omegas);
    grid.sync();

    // -------------------- Combined Loop: replicate A AND prepare B (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        // Replicate A spectrum
        const uint64_t vA = tempDigitsXX1[i];
        tempDigitsXX2[i] = vA;
        tempDigitsXY1[i] = vA;

        // Prepare B (pack+twist or zero)
        if (i < L) {
            const uint64_t coeffB = ReadBitsSimple(inB, (int64_t)i * plan.b, plan.b);
            const uint64_t cmodB = coeffB % MagicPrime;
            const uint64_t xmB = ToMontgomery(grid, block, debugMultiplyCounts, cmodB);
            const uint64_t psiB = roots.psi_pows[i]; // Montgomery domain
            tempDigitsYY1[i] = MontgomeryMul(grid, block, debugMultiplyCounts, xmB, psiB);
        } else {
            tempDigitsYY1[i] = zero_m;
        }
    }
    grid.sync();

    // B: forward NTT (grid-wide helpers)
    BitReverseInplace64(tempDigitsYY1, N, (uint32_t)plan.stages);
    NTTRadix2_GridStride(
        grid, block, debugMultiplyCounts, tempDigitsYY1, N, (uint32_t)plan.stages, roots.stage_omegas);
    grid.sync();

    // -------------------- Final replicate of B (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        const uint64_t vB = tempDigitsYY1[i];
        tempDigitsYY2[i] = vB;
        tempDigitsXY2[i] = vB;
    }
    grid.sync(); // ready for immediate consumers
}

// Fused grid-stride: untwist by psi^{-i}, scale by N^{-1} (Montgomery),
// then convert out of Montgomery — for XX1, YY1, XY1 in-place.
//
// Equivalent to:
//  for i: XX1[i] = FromMont( (XX1[i] * psi_inv[i]) * Ninv );
//  same for YY1, XY1.
//
// Requires psi_inv_pows[] and Ninvm_mont to be in Montgomery domain.
// Adds a grid sync at the end to make results visible to subsequent phases.
template <class SharkFloatParams>
static __device__ inline void
UntwistScaleFromMont_3Way_GridStride(cooperative_groups::grid_group& grid,
                                     cooperative_groups::thread_block& block,
                                     DebugMultiplyCount<SharkFloatParams>* debugMultiplyCounts,
                                     const SharkNTT::PlanPrime& plan,
                                     const SharkNTT::RootTables& roots,
                                     uint64_t* SharkRestrict tempDigitsXX1,
                                     uint64_t* SharkRestrict tempDigitsYY1,
                                     uint64_t* SharkRestrict tempDigitsXY1)
{
    using namespace SharkNTT;

    const size_t N = static_cast<size_t>(plan.N);
    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    const uint64_t Ninvm = roots.Ninvm_mont; // Montgomery-domain 1/N

    for (size_t i = grank; i < N; i += gsize) {
        const uint64_t psi_inv_i = roots.psi_inv_pows[i]; // Montgomery-domain psi^{-i}

        // XX
        {
            uint64_t v = MontgomeryMul(grid, block, debugMultiplyCounts, tempDigitsXX1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugMultiplyCounts, v, Ninvm);
            tempDigitsXX1[i] = FromMontgomery(grid, block, debugMultiplyCounts, v);
        }

        // YY
        {
            uint64_t v = MontgomeryMul(grid, block, debugMultiplyCounts, tempDigitsYY1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugMultiplyCounts, v, Ninvm);
            tempDigitsYY1[i] = FromMontgomery(grid, block, debugMultiplyCounts, v);
        }

        // XY
        {
            uint64_t v = MontgomeryMul(grid, block, debugMultiplyCounts, tempDigitsXY1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugMultiplyCounts, v, Ninvm);
            tempDigitsXY1[i] = FromMontgomery(grid, block, debugMultiplyCounts, v);
        }
    }

    grid.sync(); // ensure all writes are complete before consumers run
}



} // namespace SharkNTT

#endif

template <class SharkFloatParams, DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly static void
EraseCurrentDebugState(RecordIt record,
                       DebugState<SharkFloatParams>* debugStates,
                       cooperative_groups::grid_group& grid,
                       cooperative_groups::thread_block& block)
{
    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugStates[curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams, DebugStatePurpose Purpose, typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState(RecordIt record,
                       DebugState<SharkFloatParams>* SharkRestrict debugStates,
                       cooperative_groups::grid_group& grid,
                       cooperative_groups::thread_block& block,
                       const ArrayType* arrayToChecksum,
                       size_t arraySize)
{

    constexpr auto CurPurpose = static_cast<int32_t>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;
    constexpr auto CallIndex = 0;

    debugStates[CurPurpose].Reset(record,
                                  UseConvolutionHere,
                                  grid,
                                  block,
                                  arrayToChecksum,
                                  arraySize,
                                  Purpose,
                                  RecursionDepth,
                                  CallIndex);
}


// Look for CalculateMultiplyFrameSize and ScratchMemoryArraysForMultiply
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArraysForMultiply.
static_assert(AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets()                                                            \
    const int threadIdxGlobal =                                                                         \
        block.group_index().x * SharkFloatParams::GlobalThreadsPerBlock + block.thread_index().x;       \
    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;                                            \
    constexpr int TestMultiplier = 1;                                                                   \
    constexpr auto Multiplies_offset = AdditionalGlobalSyncSpace;                                       \
    constexpr auto Checksum_offset = Multiplies_offset + AdditionalGlobalMultipliesPerThread;           \
    constexpr auto GlobalsDoneOffset = Checksum_offset + AdditionalGlobalChecksumSpace;                 \
    constexpr auto Z0_offsetXX = GlobalsDoneOffset; \
    constexpr auto Z0_offsetXY = Z0_offsetXX + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 4 */         \
    constexpr auto Z0_offsetYY = Z0_offsetXY + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 8 */         \
    constexpr auto Z2_offsetXX = Z0_offsetYY + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 12 */        \
    constexpr auto Z2_offsetXY = Z2_offsetXX + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 16 */        \
    constexpr auto Z2_offsetYY = Z2_offsetXY + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 20 */        \
    constexpr auto Z1_temp_offsetXX = Z2_offsetYY + 4 * NewN * TestMultiplier +                         \
                                      CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 24 */   \
    constexpr auto Z1_temp_offsetXY = Z1_temp_offsetXX + 4 * NewN * TestMultiplier +                    \
                                      CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 28 */   \
    constexpr auto Z1_temp_offsetYY = Z1_temp_offsetXY + 4 * NewN * TestMultiplier +                    \
                                      CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 32 */   \
    constexpr auto Z1_offsetXX = Z1_temp_offsetYY + 4 * NewN * TestMultiplier +                         \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 36 */        \
    constexpr auto Z1_offsetXY = Z1_offsetXX + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 40 */        \
    constexpr auto Z1_offsetYY = Z1_offsetXY + 4 * NewN * TestMultiplier +                              \
                                 CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 44 */        \
    constexpr auto Convolution_offsetXX =                                                               \
        Z1_offsetYY + 4 * NewN * TestMultiplier +                                                       \
        CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 48 */                                 \
    constexpr auto Convolution_offsetXY =                                                               \
        Convolution_offsetXX + 4 * NewN * TestMultiplier +                                              \
        CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 52 */                                 \
    constexpr auto Convolution_offsetYY =                                                               \
        Convolution_offsetXY + 4 * NewN * TestMultiplier +                                              \
        CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 56 */                                 \
    constexpr auto Result_offsetXX = Convolution_offsetYY + 4 * NewN * TestMultiplier +                 \
                                     CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 60 */    \
    constexpr auto Result_offsetXY = Result_offsetXX + 4 * NewN * TestMultiplier +                      \
                                     CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 64 */    \
    constexpr auto Result_offsetYY = Result_offsetXY + 4 * NewN * TestMultiplier +                      \
                                     CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 68 */    \
    constexpr auto XDiff_offset = Result_offsetYY + 2 * NewN * TestMultiplier +                         \
                                  CalcAlign16Bytes64BitIndex(2 * NewN * TestMultiplier); /* 70 */       \
    constexpr auto YDiff_offset = XDiff_offset + 1 * NewN * TestMultiplier +                            \
                                  CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 71 */       \
    constexpr auto GlobalCarryOffset = YDiff_offset + 1 * NewN * TestMultiplier +                       \
                                       CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 72 */  \
    constexpr auto SubtractionOffset1 = GlobalCarryOffset + 1 * NewN * TestMultiplier +                 \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 73 */ \
    constexpr auto SubtractionOffset2 = SubtractionOffset1 + 1 * NewN * TestMultiplier +                \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 74 */ \
    constexpr auto SubtractionOffset3 = SubtractionOffset2 + 1 * NewN * TestMultiplier +                \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 75 */ \
    constexpr auto SubtractionOffset4 = SubtractionOffset3 + 1 * NewN * TestMultiplier +                \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 76 */ \
    constexpr auto SubtractionOffset5 = SubtractionOffset4 + 1 * NewN * TestMultiplier +                \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 77 */ \
    constexpr auto SubtractionOffset6 = SubtractionOffset5 + 1 * NewN * TestMultiplier +                \
                                        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 78 */ \
    constexpr auto CarryInsOffset =                                                                     \
        SubtractionOffset6 + 1 * NewN * TestMultiplier +                                                \
        CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* requires 3xNewN 79 */                 \
    constexpr auto CarryInsEnd = CarryInsOffset + 3 * NewN + CalcAlign16Bytes64BitIndex(3 * NewN); \




template <class SharkFloatParams>
static __device__ void
RunNTT_3Way_Multiply(HpSharkFloat<SharkFloatParams>* outXX,
         HpSharkFloat<SharkFloatParams>* outYY,
         HpSharkFloat<SharkFloatParams>* outXY,
         const HpSharkFloat<SharkFloatParams>& inA,
         const HpSharkFloat<SharkFloatParams>& inB,
         const SharkNTT::PlanPrime& plan,
         const SharkNTT::RootTables& roots,
         cg::grid_group& grid,
         cg::thread_block& block,
         DebugMultiplyCount<SharkFloatParams>* debugMultiplyCounts,
         DebugState<SharkFloatParams>* debugStates,
         RecordIt record,
         uint64_t* tempDigitsXX1,
         uint64_t* tempDigitsXX2,
         uint64_t* tempDigitsYY1,
         uint64_t* tempDigitsYY2,
         uint64_t* tempDigitsXY1,
         uint64_t* tempDigitsXY2,
         uint64_t* Final128_XX,
         uint64_t* Final128_YY,
         uint64_t* Final128_XY,
         size_t Ddigits)
{
    PackTwistFwdNTT_Fused_AB_ToSixOutputs<SharkFloatParams>(grid,
                                                            block,
                                                            debugMultiplyCounts,
                                                            inA,
                                                            inB,
                                                            plan,
                                                            roots,
                                                            tempDigitsXX1,
                                                            tempDigitsXX2,
                                                            tempDigitsYY1,
                                                            tempDigitsYY2,
                                                            tempDigitsXY1,
                                                            tempDigitsXY2);

    grid.sync();

    // 4) Pointwise multiply (Z0_OutDigits *= Z2_OutDigits)
    for (int i = 0; i < plan.N; ++i) {
        tempDigitsXX1[(size_t)i] = SharkNTT::MontgomeryMul(
            grid, block, debugMultiplyCounts, tempDigitsXX1[(size_t)i], tempDigitsXX2[(size_t)i]);

        tempDigitsYY1[(size_t)i] = SharkNTT::MontgomeryMul(
            grid, block, debugMultiplyCounts, tempDigitsYY1[(size_t)i], tempDigitsYY2[(size_t)i]);

        tempDigitsXY1[(size_t)i] = SharkNTT::MontgomeryMul(
            grid, block, debugMultiplyCounts, tempDigitsXY1[(size_t)i], tempDigitsXY2[(size_t)i]);
    }

    // 5) Inverse NTT (in place on Z0_OutDigits)
    SharkNTT::BitReverseInplace64(tempDigitsXX1, (uint32_t)plan.N, (uint32_t)plan.stages);
    SharkNTT::BitReverseInplace64(tempDigitsYY1, (uint32_t)plan.N, (uint32_t)plan.stages);
    SharkNTT::BitReverseInplace64(tempDigitsXY1, (uint32_t)plan.N, (uint32_t)plan.stages);

    SharkNTT::NTTRadix2_GridStride(grid,
                        block,
                        debugMultiplyCounts,
                        tempDigitsXX1,
                        (uint32_t)plan.N,
                        (uint32_t)plan.stages,
                        roots.stage_omegas_inv);

    SharkNTT::NTTRadix2_GridStride(grid,
                        block,
                        debugMultiplyCounts,
                        tempDigitsYY1,
                        (uint32_t)plan.N,
                        (uint32_t)plan.stages,
                        roots.stage_omegas_inv);

    SharkNTT::NTTRadix2_GridStride(grid,
                        block,
                        debugMultiplyCounts,
                        tempDigitsXY1,
                        (uint32_t)plan.N,
                        (uint32_t)plan.stages,
                        roots.stage_omegas_inv);

    //// 6) Untwist + scale by N^{-1} (write back into Z0_OutDigits)
    //for (int i = 0; i < plan.N; ++i) {
    //    uint64_t vXX = SharkNTT::MontgomeryMul(
    //        grid, block, debugMultiplyCounts, tempDigitsXX1[(size_t)i], roots.psi_inv_pows[(size_t)i]);
    //    tempDigitsXX1[(size_t)i] =
    //        SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, vXX, roots.Ninvm_mont);

    //    uint64_t vYY = SharkNTT::MontgomeryMul(
    //        grid, block, debugMultiplyCounts, tempDigitsYY1[(size_t)i], roots.psi_inv_pows[(size_t)i]);
    //    tempDigitsYY1[(size_t)i] =
    //        SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, vYY, roots.Ninvm_mont);

    //    uint64_t vXY = SharkNTT::MontgomeryMul(
    //        grid, block, debugMultiplyCounts, tempDigitsXY1[(size_t)i], roots.psi_inv_pows[(size_t)i]);
    //    tempDigitsXY1[(size_t)i] =
    //        SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, vXY, roots.Ninvm_mont);
    //}

    //// 7) Convert out of Montgomery: reuse Z2_OutDigits as normal-domain buffer
    //for (int i = 0; i < plan.N; ++i) {
    //    tempDigitsXX1[(size_t)i] =
    //        SharkNTT::FromMontgomery(grid, block, debugMultiplyCounts, tempDigitsXX1[(size_t)i]);
    //    tempDigitsYY1[(size_t)i] =
    //        SharkNTT::FromMontgomery(grid, block, debugMultiplyCounts, tempDigitsYY1[(size_t)i]);
    //    tempDigitsXY1[(size_t)i] =
    //        SharkNTT::FromMontgomery(grid, block, debugMultiplyCounts, tempDigitsXY1[(size_t)i]);
    //}

    // --- After inverse NTTs (XX1 / YY1 / XY1 are in Montgomery domain) ---
    grid.sync(); // make sure prior writes (inv-NTT) are visible

    UntwistScaleFromMont_3Way_GridStride<SharkFloatParams>(grid,
                                                           block,
                                                           debugMultiplyCounts,
                                                           plan,
                                                           roots,
                                                           /* XX1 */ tempDigitsXX1,
                                                           /* YY1 */ tempDigitsYY1,
                                                           /* XY1 */ tempDigitsXY1);

    // The helper does a final grid.sync() internally.
    // At this point, tempDigitsXX1/YY1/XY1 are back in the normal domain (not Montgomery).


    // 8) Unpack -> Final128 -> Normalize
    SharkNTT::UnpackPrimeToFinal128(tempDigitsXX1, plan, Final128_XX, Ddigits);
    SharkNTT::UnpackPrimeToFinal128(tempDigitsYY1, plan, Final128_YY, Ddigits);
    SharkNTT::UnpackPrimeToFinal128(tempDigitsXY1, plan, Final128_XY, Ddigits);

    outXX->SetNegative(false);
    outYY->SetNegative(false);

    const auto OutXYIsNegative = (inA.GetNegative() ^ inB.GetNegative());
    outXY->SetNegative(OutXYIsNegative);

    const auto addFactorOfTwoXX = 0;
    const auto addFactorOfTwoYY = 0;
    const auto addFactorOfTwoXY = 1;

    SharkNTT::Normalize<SharkFloatParams>(*outXX, inA, inA, Final128_XX, Ddigits, addFactorOfTwoXX);
    SharkNTT::Normalize<SharkFloatParams>(*outYY, inB, inB, Final128_YY, Ddigits, addFactorOfTwoYY);
    SharkNTT::Normalize<SharkFloatParams>(*outXY, inA, inB, Final128_XY, Ddigits, addFactorOfTwoXY);
}


template <class SharkFloatParams>
static __device__ void
MultiplyHelperNTTV2Separates(const SharkNTT::PlanPrime &plan,
                             const SharkNTT::RootTables &roots,
                             const HpSharkFloat<SharkFloatParams>* SharkRestrict A,
                             const HpSharkFloat<SharkFloatParams>* SharkRestrict B,
                             HpSharkFloat<SharkFloatParams>* SharkRestrict OutXX,
                             HpSharkFloat<SharkFloatParams>* SharkRestrict OutXY,
                             HpSharkFloat<SharkFloatParams>* SharkRestrict OutYY,
                             cg::grid_group& grid,
                             cg::thread_block& block,
                             uint64_t* SharkRestrict tempProducts)
{

    extern __shared__ uint32_t shared_data[];

    constexpr auto ExecutionBlockBase = 0;
    constexpr auto ExecutionNumBlocks = SharkFloatParams::GlobalNumBlocks;

    DefineTempProductsOffsets();

    // TODO: indexes
    auto* SharkRestrict debugMultiplyCounts =
        reinterpret_cast<DebugMultiplyCount<SharkFloatParams>*>(&tempProducts[Multiplies_offset]);
    auto* SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&tempProducts[Checksum_offset]);

    if constexpr (SharkPrintMultiplyCounts) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();
    }

    if constexpr (SharkDebugChecksums) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();

        const RecordIt record =
            (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) ? RecordIt::Yes
                                                                                         : RecordIt::No;
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::AHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::BHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XX>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4YY>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::FinalAdd3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 47,
                      "Unexpected number of purposes");
    }

    const RecordIt record = (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase)
                                ? RecordIt::Yes
                                : RecordIt::No;

    if constexpr (SharkDebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::ADigits,
                               uint32_t>(
            record, debugStates, grid, block, A->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams,
                               DebugStatePurpose::BDigits,
                               uint32_t>(
            record, debugStates, grid, block, B->Digits, NewN);
    }

#ifndef TEMP_DISABLEALL
    // x must be a positive constant expression

    // Verify power of 2
    static_assert(
        SharkFloatParams::GlobalNumUint32 > 0 &&
            (SharkFloatParams::GlobalNumUint32 & (SharkFloatParams::GlobalNumUint32 - 1)) == 0,
        "GlobalNumUint32 must be a power of 2");

    // Compute Final128 digit budget once
    const uint32_t Ddigits = ((uint64_t)((2 * plan.L - 2) * plan.b + 64) + 31u) / 32u + 2u;

    // ---- Single allocation for entire core path ----
    uint64_t* buffer = &tempProducts[GlobalsDoneOffset];

    // TODO: Assert or something somewhere
    //const size_t buf_count = (size_t)2 * (size_t)plan.N     // Z0_OutDigits + Z2_OutDigits
    //                         + (size_t)2 * (size_t)Ddigits; // Final128 (lo,hi per 32-bit slot)

    // Slice buffer into spans
    size_t off = 0;
    uint64_t *tempDigitsXX1 = buffer + off; // plan.N
    off += (size_t)plan.N;
    uint64_t* tempDigitsXX2 = buffer + off; // plan.N
    off += (size_t)plan.N;

    uint64_t* tempDigitsYY1 = buffer + off; // plan.N
    off += (size_t)plan.N;
    uint64_t* tempDigitsYY2 = buffer + off; // plan.N
    off += (size_t)plan.N;

    uint64_t* tempDigitsXY1 = buffer + off; // plan.N
    off += (size_t)plan.N;
    uint64_t* tempDigitsXY2 = buffer + off; // plan.N
    off += (size_t)plan.N;

    uint64_t *Final128_XX = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t* Final128_YY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t* Final128_XY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    // XX = A^2
    RunNTT_3Way_Multiply<SharkFloatParams>(OutXX,
                                            OutYY,
                                            OutXY,
                                            *A,
                                            *B,
                                            plan,
                                            roots,
                                            grid,
                                            block,
                                            debugMultiplyCounts,
                                            debugStates,
                                            record,
                                            tempDigitsXX1,
                                            tempDigitsXX2,
                                            tempDigitsYY1,
                                            tempDigitsYY2,
                                            tempDigitsXY1,
                                            tempDigitsXY2,
                                            Final128_XX,
                                            Final128_YY,
                                            Final128_XY,
                                            Ddigits);

    grid.sync();
#endif
}

template <class SharkFloatParams>
void
PrintMaxActiveBlocks(void* kernelFn, int sharedAmountBytes)
{
    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;

    {
        // Check the maximum number of active blocks per multiprocessor
        // with the given shared memory size
        // This is useful to determine if we can fit more blocks
        // in the shared memory

        const auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, kernelFn, SharkFloatParams::GlobalThreadsPerBlock, sharedAmountBytes);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max active blocks per multiprocessor: " << numBlocks << std::endl;
    }

    {
        size_t availableSharedMemory = 0;
        const auto err = cudaOccupancyAvailableDynamicSMemPerBlock(
            &availableSharedMemory, kernelFn, numBlocks, SharkFloatParams::GlobalThreadsPerBlock);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyAvailableDynamicSMemPerBlock: "
                      << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Available shared memory per block: " << availableSharedMemory << std::endl;
    }

    // Check the number of multiprocessors on the device
    int numSM;

    {
        const auto err = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        std::cout << "Number of multiprocessors: " << numSM << std::endl;
    }

    int maxConcurrentBlocks = numSM * numBlocks;

    std::cout << "Max concurrent blocks: " << maxConcurrentBlocks << std::endl;
    if (maxConcurrentBlocks < SharkFloatParams::GlobalNumBlocks) {
        std::cout << "Warning: Max concurrent blocks exceeds the number of blocks requested."
                  << std::endl;
    }

    {
        // Check the maximum number of threads per block
        int maxThreadsPerBlock;
        const auto err = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    }

    {
        // Check the maximum number of threads per multiprocessor
        int maxThreadsPerMultiprocessor;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }
        std::cout << "Max threads per multiprocessor: " << maxThreadsPerMultiprocessor << std::endl;
    }

    // Check if this device supports cooperative launches
    int cooperativeLaunch;

    {
        const auto err = cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, 0);

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err)
                      << std::endl;
            return;
        }

        if (cooperativeLaunch) {
            std::cout << "This device supports cooperative launches." << std::endl;
        } else {
            std::cout << "This device does not support cooperative launches." << std::endl;
        }
    }
}

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
template <class SharkFloatParams>
static __device__ void
MultiplyHelperNTT(HpSharkComboResults<SharkFloatParams>* SharkRestrict combo,
                  cg::grid_group& grid,
                  cg::thread_block& block,
                  uint64_t* SharkRestrict tempProducts)
{
    MultiplyHelperNTTV2Separates<SharkFloatParams>(combo->Plan,
                                                   combo->Roots,
                                                   &combo->A,
                                                   &combo->B,
                                                   &combo->ResultX2,
                                                   &combo->Result2XY,
                                                   &combo->ResultY2,
                                                   grid,
                                                   block,
                                                   tempProducts);
}
