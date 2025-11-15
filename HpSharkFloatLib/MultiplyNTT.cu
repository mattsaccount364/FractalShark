#include "MultiplyNTT.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "DebugChecksum.cuh"
#include "HpSharkFloat.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <vector>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

#include "NTTConstexprGenerator.h"
#include "MontgomeryCoreConstexpr.cuh"

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

namespace SharkNTT {

//--------------------------------------------------------------------------------------------------
// Normalization (final carry propagation and windowing into HpSharkFloat<SharkFloatParams>)
//--------------------------------------------------------------------------------------------------

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Does not allocate; does two passes over Final128 to (1) find 'significant',
// then (2) write the window and set the Karatsuba-style exponent.
// NOTE: Does not set sign; do that at the call site.
template <class SharkFloatParams>
static __device__ void
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

template <class SharkFloatParams>
static __device__ inline void
Normalize_GridStride_3Way(cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          DebugState<SharkFloatParams> *debugStates,
                          // outputs
                          HpSharkFloat<SharkFloatParams> &outXX,
                          HpSharkFloat<SharkFloatParams> &outYY,
                          HpSharkFloat<SharkFloatParams> &outXY,
                          // input exponents
                          const HpSharkFloat<SharkFloatParams> &inA,
                          const HpSharkFloat<SharkFloatParams> &inB,
                          // convolution sums (lo,hi per 32-bit position), NORMAL domain
                          const uint64_t *SharkRestrict final128XX,
                          const uint64_t *SharkRestrict final128YY,
                          const uint64_t *SharkRestrict final128XY,
                          size_t Ddigits,
                          int32_t addTwoXX,
                          int32_t addTwoYY,
                          int32_t addTwoXY,
                          // global workspaces (NOT shared memory)
                          uint64_t *SharkRestrict CarryPropagationBuffer2, // >= 6 + 6*lanes u64
                          uint64_t *SharkRestrict /*block_carry_outs*/,    // unused
                          uint64_t *SharkRestrict globalCarryCheck,        // 1 u64
                          uint64_t *SharkRestrict resultXX,                // len >= Ddigits
                          uint64_t *SharkRestrict resultYY,                // len >= Ddigits
                          uint64_t *SharkRestrict resultXY)                // len >= Ddigits
{
    constexpr int N32 = SharkFloatParams::GlobalNumUint32;

    // We only ever produce digits in [0, Ddigits).
    const int ProducedDigits = static_cast<int>(Ddigits);

    const int T_all = static_cast<int>(grid.size());
    const int tid = static_cast<int>(grid.thread_rank());

    // Active participants are min(ProducedDigits, T_all)
    const int lanes = (ProducedDigits < T_all) ? ProducedDigits : T_all;
    const bool active = (tid < lanes);
    const int lane = tid; // 0..lanes-1 for active threads

    // Linear partition of [0, ProducedDigits) across 'lanes'
    const int64_t PD = static_cast<int64_t>(ProducedDigits);
    const int64_t LNS = static_cast<int64_t>(lanes);
    const int64_t LN = static_cast<int64_t>(lane);
    const int start = active ? static_cast<int>((PD * LN) / LNS) : 0;
    const int end = active ? static_cast<int>((PD * (LN + 1)) / LNS) : 0;

    // --- 1) Initial pass over our slice (no tail beyond Ddigits) ---
    uint64_t carry_xx_lo = 0ull, carry_xx_hi = 0ull;
    uint64_t carry_yy_lo = 0ull, carry_yy_hi = 0ull;
    uint64_t carry_xy_lo = 0ull, carry_xy_hi = 0ull;

    if (active) {
        for (int idx = start; idx < end; ++idx) {
            const bool in = (idx < ProducedDigits);
            const size_t s = static_cast<size_t>(idx) * 2u;

            // XX
            {
                const uint64_t lo = in ? final128XX[s + 0] : 0ull;
                const uint64_t hi = in ? final128XX[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_xx_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_xx_hi + c0;

                resultXX[idx] = static_cast<uint32_t>(s_lo);
                carry_xx_lo = (s_lo >> 32) | (s_hi << 32);
                carry_xx_hi = (s_hi >> 32);
            }
            // YY
            {
                const uint64_t lo = in ? final128YY[s + 0] : 0ull;
                const uint64_t hi = in ? final128YY[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_yy_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_yy_hi + c0;

                resultYY[idx] = static_cast<uint32_t>(s_lo);
                carry_yy_lo = (s_lo >> 32) | (s_hi << 32);
                carry_yy_hi = (s_hi >> 32);
            }
            // XY
            {
                const uint64_t lo = in ? final128XY[s + 0] : 0ull;
                const uint64_t hi = in ? final128XY[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_xy_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_xy_hi + c0;

                resultXY[idx] = static_cast<uint32_t>(s_lo);
                carry_xy_lo = (s_lo >> 32) | (s_hi << 32);
                carry_xy_hi = (s_hi >> 32);
            }
        }

        // Publish residual to the next lane except for the **last lane**.
        // By design (matches simplified scalar), we DROP any residual that would flow beyond Ddigits.
        if (lane == lanes - 1) {
            carry_xx_lo = carry_xx_hi = 0ull;
            carry_xy_lo = carry_xy_hi = 0ull;
            carry_yy_lo = carry_yy_hi = 0ull;
        }

        const int base = 6 + lane * 6;
        CarryPropagationBuffer2[base + 0] = carry_xx_lo;
        CarryPropagationBuffer2[base + 1] = carry_xx_hi;
        CarryPropagationBuffer2[base + 2] = carry_xy_lo;
        CarryPropagationBuffer2[base + 3] = carry_xy_hi;
        CarryPropagationBuffer2[base + 4] = carry_yy_lo;
        CarryPropagationBuffer2[base + 5] = carry_yy_hi;
    }
    grid.sync();

    // --- 2) Iterative carry propagation within [0, Ddigits) (drop at right edge) ---
    while (true) {
        if (tid == 0)
            *globalCarryCheck = 0ull;

        uint64_t in_xx_lo = 0ull, in_xx_hi = 0ull;
        uint64_t in_yy_lo = 0ull, in_yy_hi = 0ull;
        uint64_t in_xy_lo = 0ull, in_xy_hi = 0ull;

        if (active) {
            if (lane > 0) {
                const int prev = 6 + (lane - 1) * 6;
                in_xx_lo = CarryPropagationBuffer2[prev + 0];
                in_xx_hi = CarryPropagationBuffer2[prev + 1];
                in_xy_lo = CarryPropagationBuffer2[prev + 2];
                in_xy_hi = CarryPropagationBuffer2[prev + 3];
                in_yy_lo = CarryPropagationBuffer2[prev + 4];
                in_yy_hi = CarryPropagationBuffer2[prev + 5];
            }
        }

        grid.sync();

        if (active) {
            for (int idx = start; idx < end; ++idx) {
                // XX
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_xx_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultXX[idx]) + add32;
                    resultXX[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_xx_lo >> 32) | (in_xx_hi << 32);
                    const uint64_t next_hi = (in_xx_hi >> 32);
                    in_xx_lo = next_lo + c32;
                    in_xx_hi = next_hi;
                }
                // YY
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_yy_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultYY[idx]) + add32;
                    resultYY[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_yy_lo >> 32) | (in_yy_hi << 32);
                    const uint64_t next_hi = (in_yy_hi >> 32);
                    in_yy_lo = next_lo + c32;
                    in_yy_hi = next_hi;
                }
                // XY
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_xy_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultXY[idx]) + add32;
                    resultXY[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_xy_lo >> 32) | (in_xy_hi << 32);
                    const uint64_t next_hi = (in_xy_hi >> 32);
                    in_xy_lo = next_lo + c32;
                    in_xy_hi = next_hi;
                }
            }

            // Drop residual at the right boundary (last lane), publish otherwise.
            if (lane == lanes - 1) {
                in_xx_lo = in_xx_hi = 0ull;
                in_xy_lo = in_xy_hi = 0ull;
                in_yy_lo = in_yy_hi = 0ull;
            }

            const int base = 6 + lane * 6;
            CarryPropagationBuffer2[base + 0] = in_xx_lo;
            CarryPropagationBuffer2[base + 1] = in_xx_hi;
            CarryPropagationBuffer2[base + 2] = in_xy_lo;
            CarryPropagationBuffer2[base + 3] = in_xy_hi;
            CarryPropagationBuffer2[base + 4] = in_yy_lo;
            CarryPropagationBuffer2[base + 5] = in_yy_hi;

            // Only signal continuation if something remains to hand to the *next* lane.
            if (in_xx_lo | in_xx_hi | in_xy_lo | in_xy_hi | in_yy_lo | in_yy_hi)
                atomicAdd(globalCarryCheck, 1ull);
        }

        grid.sync();
        // Atomic read to avoid any visibility doubt
        if (atomicAdd(globalCarryCheck, 0ull) == 0ull)
            break;
        grid.sync();
    }

    // --- 3) Scan [0, Ddigits) for highest-nonzero; compute shifts/exponents (single thread) ---
    int h_xx = -1, h_yy = -1, h_xy = -1;
    if (tid == 0) {
        for (int i = ProducedDigits - 1; i >= 0; --i) {
            if (h_xx < 0 && static_cast<uint32_t>(resultXX[i]) != 0u)
                h_xx = i;
            if (h_yy < 0 && static_cast<uint32_t>(resultYY[i]) != 0u)
                h_yy = i;
            if (h_xy < 0 && static_cast<uint32_t>(resultXY[i]) != 0u)
                h_xy = i;
            if (h_xx >= 0 && h_yy >= 0 && h_xy >= 0)
                break;
        }

        auto shift_exp = [&](int h, int32_t add, int32_t ea, int32_t eb) -> std::pair<int, int32_t> {
            if (h < 0)
                return {0, ea + eb};
            const int significant = h + 1;
            int shift = significant - N32;
            if (shift < 0)
                shift = 0;
            return {shift, ea + eb + 32 * shift + add};
        };

        auto [sXX, eXX] = shift_exp(h_xx, addTwoXX, inA.Exponent, inA.Exponent);
        auto [sYY, eYY] = shift_exp(h_yy, addTwoYY, inB.Exponent, inB.Exponent);
        auto [sXY, eXY] = shift_exp(h_xy, addTwoXY, inA.Exponent, inB.Exponent);

        outXX.Exponent = eXX;
        outYY.Exponent = eYY;
        outXY.Exponent = eXY;

        // Broadcast shifts + zero flags
        CarryPropagationBuffer2[0] = static_cast<uint64_t>(sXX);
        CarryPropagationBuffer2[1] = static_cast<uint64_t>(sYY);
        CarryPropagationBuffer2[2] = static_cast<uint64_t>(sXY);
        CarryPropagationBuffer2[3] = static_cast<uint64_t>(h_xx < 0);
        CarryPropagationBuffer2[4] = static_cast<uint64_t>(h_yy < 0);
        CarryPropagationBuffer2[5] = static_cast<uint64_t>(h_xy < 0);
    }
    grid.sync();

    const int sXX = static_cast<int>(CarryPropagationBuffer2[0]);
    const int sYY = static_cast<int>(CarryPropagationBuffer2[1]);
    const int sXY = static_cast<int>(CarryPropagationBuffer2[2]);
    const bool zXX = (CarryPropagationBuffer2[3] != 0);
    const bool zYY = (CarryPropagationBuffer2[4] != 0);
    const bool zXY = (CarryPropagationBuffer2[5] != 0);

    // --- 4) Grid-stride write the N32-digit windows (bounds-safe into [0, Ddigits)) ---
    for (int i = tid; i < N32; i += T_all) {
        // XX
        outXX.Digits[i] =
            zXX ? 0u : ((sXX + i < ProducedDigits) ? static_cast<uint32_t>(resultXX[sXX + i]) : 0u);

        // YY
        outYY.Digits[i] =
            zYY ? 0u : ((sYY + i < ProducedDigits) ? static_cast<uint32_t>(resultYY[sYY + i]) : 0u);

        // XY
        outXY.Digits[i] =
            zXY ? 0u : ((sXY + i < ProducedDigits) ? static_cast<uint32_t>(resultXY[sXY + i]) : 0u);
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XX, uint32_t>(
            debugStates, grid, block, outXX.Digits, SharkFloatParams::GlobalNumUint32);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128YY, uint32_t>(
            debugStates, grid, block, outYY.Digits, SharkFloatParams::GlobalNumUint32);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY, uint32_t>(
            debugStates, grid, block, outXY.Digits, SharkFloatParams::GlobalNumUint32);
        grid.sync();
    } else {
        grid.sync();
    }
}

// Normalize directly from Final128 (lo64,hi64 per 32-bit digit position).
// Grid-stride is used where it helps (zeroing the output window up front).
// The core carry propagation is inherently sequential, so one thread performs it.
// This preserves the original math and write-window behavior (no allocations).
//
// NOTE: Does not set sign; do that at the call site.
template <class SharkFloatParams>
static __device__ void
Normalize_GridStride(cooperative_groups::grid_group &grid,
                     HpSharkFloat<SharkFloatParams> &out,
                     const HpSharkFloat<SharkFloatParams> &a,
                     const HpSharkFloat<SharkFloatParams> &b,
                     const uint64_t *SharkRestrict final128, // len = 2*Ddigits
                     size_t Ddigits,                         // number of 32-bit positions in final128
                     int32_t additionalFactorOfTwo) // 0 for XX/YY; 1 for XY (if not pre-shifted)
{
    using std::size_t;
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    //// -------------------- Grid-stride zero the output digits up front --------------------
    //{
    //    const size_t gsize = grid.size();
    //    const size_t grank = grid.thread_rank();
    //    for (size_t i = grank; i < static_cast<size_t>(N); i += gsize) {
    //        out.Digits[i] = 0u;
    //    }
    //}
    // grid.sync(); // make zeros visible

    // -------------------- Single-thread carry propagation & window selection --------------------
    if (grid.thread_rank() == 0) {
        // ---------- Pass 1: find 'significant' (last non-zero + 1) ----------
        uint64_t carry_lo = 0, carry_hi = 0;
        size_t idx = 0;           // index in base-2^32 digits being produced
        int highest_nonzero = -1; // last non-zero digit index seen

        // Main Ddigits loop
        for (size_t d = 0; d < Ddigits; ++d) {
            const uint64_t lo = final128[2 * d + 0];
            const uint64_t hi = final128[2 * d + 1];

            const uint64_t s_lo = lo + carry_lo;
            const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
            const uint64_t s_hi = hi + carry_hi + c0;

            const uint32_t dig = static_cast<uint32_t>(s_lo & 0xffffffffull);
            if (dig != 0)
                highest_nonzero = static_cast<int>(idx);
            ++idx;

            carry_lo = (s_lo >> 32) | (s_hi << 32);
            carry_hi = (s_hi >> 32);
        }

        if (highest_nonzero < 0) {
            // Zero result: exponent is sum; digits already zeroed grid-wide.
            out.Exponent = a.Exponent + b.Exponent; // Karatsuba zero convention
        } else {
            const int significant = highest_nonzero + 1;
            int shift_digits = significant - N;
            if (shift_digits < 0)
                shift_digits = 0;

            // Karatsuba-style exponent
            out.Exponent = a.Exponent + b.Exponent + 32 * shift_digits + additionalFactorOfTwo;

            // ---------- Pass 2: write exactly N digits starting at shift_digits ----------
            carry_lo = 0;
            carry_hi = 0;
            idx = 0;
            int out_written = 0;
            const size_t start = static_cast<size_t>(shift_digits);
            const size_t needN = static_cast<size_t>(N);

            // Main loop again, now emitting a window
            for (size_t d = 0; d < Ddigits && out_written < N; ++d) {
                const uint64_t lo = final128[2 * d + 0];
                const uint64_t hi = final128[2 * d + 1];

                const uint64_t s_lo = lo + carry_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_hi + c0;

                const uint32_t dig = static_cast<uint32_t>(s_lo & 0xffffffffull);
                if (idx >= start && static_cast<size_t>(out_written) < needN) {
                    out.Digits[out_written++] = dig;
                }
                ++idx;

                carry_lo = (s_lo >> 32) | (s_hi << 32);
                carry_hi = (s_hi >> 32);
            }

            // Any remaining tail (if window exceeded produced digits) is already zero
            // from the grid-stride zeroing above.
        }
    }

    grid.sync(); // ensure exponent and digits are finalized before consumers
}

//--------------------------------------------------------------------------------------------------
// 64×64→128 helpers (compiler/ABI specific intrinsics)
//--------------------------------------------------------------------------------------------------

static __device__ SharkForceInlineReleaseOnly uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t &carry)
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
MontgomeryMul(cooperative_groups::grid_group &grid,
              cooperative_groups::thread_block &block,
              DebugMultiplyCount<SharkFloatParams> *debugCombo,
              uint64_t a,
              uint64_t b)
{
    // We'll count 128-bit multiplications here as 3x64  (so 3 + 3 + 1)
    if constexpr (HpShark::PrintMultiplyCounts) {
        DebugMultiplyIncrement<SharkFloatParams>(debugCombo, grid, block, 7);
    }

    // t = a*b (128-bit)
    uint64_t t_lo, t_hi;

    t_lo = a * b;
    t_hi = __umul64hi(a, b);

    // m = (t_lo * NINV) mod 2^64
    uint64_t m = t_lo * SharkNTT::MagicPrimeInv;

    // m*SharkNTT::MagicPrime (128-bit)
    uint64_t mp_lo, mp_hi;
    mp_lo = m * SharkNTT::MagicPrime;
    mp_hi = __umul64hi(m, SharkNTT::MagicPrime);

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
ToMontgomery(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             DebugMultiplyCount<SharkFloatParams> *debugCombo,
             uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, R2);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
FromMontgomery(cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugMultiplyCount<SharkFloatParams> *debugCombo,
               uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, 1);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
MontgomeryPow(cooperative_groups::grid_group &grid,
              cooperative_groups::thread_block &block,
              DebugMultiplyCount<SharkFloatParams> *debugCombo,
              uint64_t a_mont,
              uint32_t e)
{
    uint64_t x = ToMontgomeryConstexpr(1);
    uint64_t y = a_mont;
    while (e) {
        if (e & 1)
            x = MontgomeryMul(grid, block, debugCombo, x, y);
        y = MontgomeryMul(grid, block, debugCombo, y, y);
        e >>= 1;
    }
    return x;
}

enum class Multiway { OneWay, TwoWay, ThreeWay };

// Grid-stride in-place bit-reversal permutation (uint64 elements).
// Safe without atomics: each index participates in exactly one swap pair (i, j=rev(i));
// only the lower index performs the swap (j > i), so pairs are disjoint.
template <Multiway OneTwoThree>
static __device__ SharkForceInlineReleaseOnly void
BitReverseInplace64_GridStride(cooperative_groups::grid_group &grid,
                               uint64_t *SharkRestrict A,
                               uint64_t *SharkRestrict B,
                               uint64_t *SharkRestrict C,
                               uint32_t N,
                               uint32_t stages)
{
    // Compile-time selection of which arrays to process
    constexpr bool DoA = true;
    constexpr bool DoB = (OneTwoThree != Multiway::OneWay);
    constexpr bool DoC = (OneTwoThree == Multiway::ThreeWay);

    const uint32_t gsz = static_cast<uint32_t>(grid.size());
    const uint32_t tid = static_cast<uint32_t>(grid.thread_rank());

    // Reverse the lowest `stages` bits via __brev; drop the high bits.
    const uint32_t sh = 32u - stages; // assumes N <= 2^32

    // Swap helper for one array (loads to registers first to avoid rereads).
    auto swap_one = [](uint64_t *SharkRestrict arr, uint32_t i, uint32_t j) {
        const uint64_t ai = arr[i];
        const uint64_t aj = arr[j];
        arr[i] = aj;
        arr[j] = ai;
    };

    // Process helper for a single index/pair across the enabled arrays.
    auto process_idx = [&](uint32_t i, uint32_t j) {
        if (i >= N)
            return;
        if (i == j)
            return; // fixed point
        if (i > j)
            return; // only one owner does the swap

        if constexpr (DoA)
            swap_one(A, i, j);
        if constexpr (DoB)
            swap_one(B, i, j);
        if constexpr (DoC)
            swap_one(C, i, j);
    };

    // Grid-stride loop, unrolled by 4 when possible.
    const uint32_t step4 = gsz << 2; // 4 * gsz
    for (uint32_t base = tid; base < N; base += step4) {
        const uint32_t i0 = base;
        const uint32_t i1 = i0 + gsz;
        const uint32_t i2 = i1 + gsz;
        const uint32_t i3 = i2 + gsz;

        // Fast compute of reversed partners using __brev (32-bit path).
        const uint32_t j0 = __brev(i0) >> sh;
        const uint32_t j1 = (i1 < N) ? (__brev(i1) >> sh) : 0u;
        const uint32_t j2 = (i2 < N) ? (__brev(i2) >> sh) : 0u;
        const uint32_t j3 = (i3 < N) ? (__brev(i3) >> sh) : 0u;

        // Handle up to 4 indices per iteration.
        process_idx(i0, j0);
        process_idx(i1, j1);
        process_idx(i2, j2);
        process_idx(i3, j3);
    }
}

//--------------------------------------------------------------------------------------------------
// Iterative radix-2 NTT (Cooley–Tukey) over Montgomery domain
//--------------------------------------------------------------------------------------------------

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
NTTRadix2(cooperative_groups::grid_group &grid,
          cooperative_groups::thread_block &block,
          DebugMultiplyCount<SharkFloatParams> *debugCombo,
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
            uint64_t w = ToMontgomeryConstexpr(1);
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

template <class SharkFloatParams, Multiway OneTwoThree, uint32_t TS_log>
static __device__ inline uint32_t
SmallRadixPhase1_SM(uint64_t *shared_data,
                    cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugMultiplyCount<SharkFloatParams> *debugCombo,
                    uint64_t *__restrict A,
                    uint64_t *__restrict B,
                    uint64_t *__restrict C,
                    uint32_t N,
                    uint32_t stages,
                    const uint64_t *__restrict stage_base)
{
    constexpr uint32_t TS = 1u << TS_log;

    auto ctz_u32 = [](uint32_t x) -> uint32_t {
        uint32_t c = 0;
        while ((x & 1u) == 0u) {
            x >>= 1u;
            ++c;
        }
        return c;
    };
    const uint32_t S0 = stages < TS_log ? stages : TS_log;
    const uint32_t rem = (N & (TS - 1u));
    const uint32_t tail_len = (rem == 0u) ? TS : rem;
    const uint32_t tail_cap = (rem == 0u) ? TS_log : ctz_u32(tail_len);
    const uint32_t S1 = (S0 < tail_cap) ? S0 : tail_cap;
    if (S1 == 0)
        return 0;

    const uint32_t tiles = (N + TS - 1u) / TS;

    // Carve tile base from shared_data (A lives in shared)
    auto *const s_dataA = reinterpret_cast<uint64_t *>(shared_data);
    auto *const s_dataB = reinterpret_cast<uint64_t *>(shared_data) + TS;
    auto *const s_dataC = reinterpret_cast<uint64_t *>(shared_data) + 2 * TS;

    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const bool is_last = (tile == tiles - 1);
        const uint32_t len = is_last ? tail_len : TS; // divisible by 2^S1

        // Load A tile only (keep B,C in global)
        if constexpr (OneTwoThree == Multiway::OneWay || OneTwoThree == Multiway::TwoWay ||
                      OneTwoThree == Multiway::ThreeWay) {
            cg::memcpy_async(block, s_dataA, &A[tile * TS], len * sizeof(uint64_t));
        }

        if constexpr (OneTwoThree == Multiway::TwoWay || OneTwoThree == Multiway::ThreeWay) {
            cg::memcpy_async(block, s_dataB, &B[tile * TS], len * sizeof(uint64_t));
        }

        if constexpr (OneTwoThree == Multiway::ThreeWay) {
            cg::memcpy_async(block, s_dataC, &C[tile * TS], len * sizeof(uint64_t));
        }

        cg::wait(block);
        block.sync();

        // Stages s=1..S1 — single set of loops; reuse the same syncs for all three
        for (uint32_t s = 1; s <= S1; ++s) {
            const uint32_t m = 1u << s;
            const uint32_t half = m >> 1;
            const uint64_t w_m = stage_base[s - 1];

            const uint32_t total_pairs = (len >> 1);
            const uint32_t tid = block.thread_index().x;
            const uint32_t step = block.size();

            for (uint32_t p = tid; p < total_pairs; p += step) {
                const uint32_t group = p / half;
                const uint32_t j = p - group * half; // p % half
                const uint32_t i0 = group * m + j;
                const uint32_t i1 = i0 + half;

                const uint64_t wj = MontgomeryPow(grid, block, debugCombo, w_m, j);

                // ---- A (shared) ----
                if constexpr (OneTwoThree == Multiway::OneWay) {
                    const uint64_t U1 = s_dataA[i0];
                    const uint64_t V1 = s_dataA[i1];

                    const uint64_t t = MontgomeryMul(grid, block, debugCombo, V1, wj);

                    s_dataA[i0] = AddP(U1, t);
                    s_dataA[i1] = SubP(U1, t);
                }

                if constexpr (OneTwoThree == Multiway::TwoWay) {
                    const uint64_t U1 = s_dataA[i0];
                    const uint64_t V1 = s_dataA[i1];

                    const uint64_t U2 = s_dataB[i0];
                    const uint64_t V2 = s_dataB[i1];

                    const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, V1, wj);

                    const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, V2, wj);

                    s_dataA[i0] = AddP(U1, t1);
                    s_dataA[i1] = SubP(U1, t1);

                    s_dataB[i0] = AddP(U2, t2);
                    s_dataB[i1] = SubP(U2, t2);
                }

                if constexpr (OneTwoThree == Multiway::ThreeWay) {
                    const uint64_t U1 = s_dataA[i0];
                    const uint64_t V1 = s_dataA[i1];

                    const uint64_t U2 = s_dataB[i0];
                    const uint64_t V2 = s_dataB[i1];

                    const uint64_t U3 = s_dataC[i0];
                    const uint64_t V3 = s_dataC[i1];

                    const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, V1, wj);
                    const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, V2, wj);
                    const uint64_t t3 = MontgomeryMul(grid, block, debugCombo, V3, wj);

                    s_dataA[i0] = AddP(U1, t1);
                    s_dataA[i1] = SubP(U1, t1);

                    s_dataB[i0] = AddP(U2, t2);
                    s_dataB[i1] = SubP(U2, t2);

                    s_dataC[i0] = AddP(U3, t3);
                    s_dataC[i1] = SubP(U3, t3);
                }
            }
            block.sync();
        }

        // Store A
        for (uint32_t t = block.thread_index().x; t < len; t += block.size()) {
            if constexpr (OneTwoThree == Multiway::OneWay || OneTwoThree == Multiway::TwoWay ||
                          OneTwoThree == Multiway::ThreeWay) {
                A[tile * TS + t] = s_dataA[t];
            }

            if constexpr (OneTwoThree == Multiway::TwoWay || OneTwoThree == Multiway::ThreeWay) {
                B[tile * TS + t] = s_dataB[t];
            }

            if constexpr (OneTwoThree == Multiway::ThreeWay) {
                C[tile * TS + t] = s_dataC[t];
            }
        }
    }

    grid.sync();
    return S1;
}


// Unified 1-way / 3-way radix-2 NTT with warp-strided twiddles,
// early shared-memory microkernel, and Phase-2 persistent task queue.
// ThreeWay=false: operates on A only (matches 1-way behavior)
// ThreeWay=true : operates on A, B, C in lockstep
template <class SharkFloatParams, Multiway OneTwoThree>
static __device__ SharkForceInlineReleaseOnly void
NTTRadix2_GridStride(uint64_t *shared_data,
                     cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugMultiplyCount<SharkFloatParams> *debugCombo,
                     uint64_t *SharkRestrict globalSync1,
                     uint64_t *SharkRestrict A,
                     uint64_t *SharkRestrict B,
                     uint64_t *SharkRestrict C,
                     uint32_t N,
                     uint32_t stages,
                     const uint64_t *SharkRestrict stage_base)
{
    constexpr auto TS_log = 9u;
    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    uint32_t S0 = 0;
    {
        if constexpr (OneTwoThree == Multiway::OneWay) {
            S0 = SmallRadixPhase1_SM<SharkFloatParams, Multiway::OneWay, TS_log>(
                shared_data, grid, block, debugCombo, A, nullptr, nullptr, N, stages, stage_base);
        } else if constexpr (OneTwoThree == Multiway::TwoWay) {
            S0 = SmallRadixPhase1_SM<SharkFloatParams, Multiway::TwoWay, TS_log>(
                shared_data, grid, block, debugCombo, A, B, nullptr, N, stages, stage_base);
        } else if constexpr (OneTwoThree == Multiway::ThreeWay) {
            S0 = SmallRadixPhase1_SM<SharkFloatParams, Multiway::ThreeWay, TS_log>(
                shared_data, grid, block, debugCombo, A, B, C, N, stages, stage_base);
        }
    }

// =========================
    // Phase 2: static contiguous striping by warp (no atomics)
    //          + scalarized micro-tiling (register-only, U=4/2)
    // =========================
    for (uint32_t s = S0 + 1; s <= stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;
        const uint64_t w_m = stage_base[s - 1];

        // Per-stage tiling in j: J_total = ceil(half / W)
        constexpr uint32_t W = 32u;
        const uint32_t J_total = (half + (W - 1u)) / W; // ceil(half/W)
        const uint32_t nblocks = N / m;
        const size_t total_tasks =
            static_cast<size_t>(nblocks) * static_cast<size_t>(J_total); // == N/64, invariant in s

        // Per-warp/per-lane constants
        const size_t gsize = grid.size();
        const size_t grank = grid.thread_rank();
        const uint32_t lane = static_cast<uint32_t>(grank % W);
        const uint32_t warp_id = static_cast<uint32_t>(grank / W);
        const uint32_t warp_count = static_cast<uint32_t>(gsize / W);

        const uint64_t w_strideW = MontgomeryPow(grid, block, debugCombo, w_m, W);      // w_m^W
        const uint64_t w_lane_base = MontgomeryPow(grid, block, debugCombo, w_m, lane); // w_m^lane

        // -------- Static contiguous partition: each warp gets one equal-sized range --------
        const size_t tasks_per_warp = (total_tasks + warp_count - 1ull) / warp_count; // ceil
        size_t T0 = static_cast<size_t>(warp_id) * tasks_per_warp;
        size_t Tend = min(total_tasks, T0 + tasks_per_warp);

        if (T0 < Tend) {
            // Decode the first ticket in our contiguous range
            uint32_t blk_id = static_cast<uint32_t>(T0 / J_total);
            uint32_t j_chunk =
                static_cast<uint32_t>(T0 - static_cast<size_t>(blk_id) * J_total); // faster than %

            // Twiddle at start of our range: w = w_lane_base * (w_strideW ^ j_chunk)
            uint64_t w = (j_chunk == 0)
                             ? w_lane_base
                             : MontgomeryMul(grid,
                                             block,
                                             debugCombo,
                                             w_lane_base,
                                             MontgomeryPow(grid, block, debugCombo, w_strideW, j_chunk));

            uint32_t k_base = blk_id * m;
            size_t remaining = Tend - T0;

            // Micro-tiling width: 4 for OneWay, 2 for Two/ThreeWay (keeps regs in check)
            constexpr int U = (OneTwoThree == Multiway::OneWay ? 4 : 2);

            // ------------- process our contiguous slice -------------
            while (remaining) {
                // Tile cannot cross block boundary or our assigned range
                const uint32_t room_in_block = J_total - j_chunk;
                const uint32_t span = static_cast<uint32_t>(std::min<size_t>(remaining, room_in_block));
                const uint32_t tile =
                    (U == 4) ? std::min<uint32_t>(4u, span) : std::min<uint32_t>(2u, span);

                // ===== position 0 =====
                const uint32_t j0_0 = lane + j_chunk * W;
                const bool a0 = (j0_0 < half);
                uint32_t i0_0 = 0, i1_0 = 0;
                uint64_t Au0 = 0, Av0 = 0, Bu0 = 0, Bv0 = 0, Cu0 = 0, Cv0 = 0;

                if (a0) {
                    i0_0 = k_base + j0_0;
                    i1_0 = i0_0 + half;
                    Au0 = A[i0_0];
                    Av0 = A[i1_0];
                    if constexpr (OneTwoThree != Multiway::OneWay) {
                        Bu0 = B[i0_0];
                        Bv0 = B[i1_0];
                        if constexpr (OneTwoThree == Multiway::ThreeWay) {
                            Cu0 = C[i0_0];
                            Cv0 = C[i1_0];
                        }
                    }
                }
                const uint64_t tw0 = w;

                // Pre-advance twiddle and optionally prefetch pos1 operands
                uint64_t tw1 = MontgomeryMul(grid, block, debugCombo, tw0, w_strideW);

                // ===== position 1 (if any) =====
                bool a1 = false;
                uint32_t i0_1 = 0, i1_1 = 0;
                uint64_t Au1 = 0, Av1 = 0, Bu1 = 0, Bv1 = 0, Cu1 = 0, Cv1 = 0;
                if (tile >= 2) {
                    const uint32_t j0_1 = j0_0 + W;
                    a1 = (j0_1 < half);
                    if (a1) {
                        i0_1 = k_base + j0_1;
                        i1_1 = i0_1 + half;
                        Au1 = A[i0_1];
                        Av1 = A[i1_1];
                        if constexpr (OneTwoThree != Multiway::OneWay) {
                            Bu1 = B[i0_1];
                            Bv1 = B[i1_1];
                            if constexpr (OneTwoThree == Multiway::ThreeWay) {
                                Cu1 = C[i0_1];
                                Cv1 = C[i1_1];
                            }
                        }
                    }
                }

                // ===== position 2/3 (OneWay only) =====
                uint32_t i0_2 = 0, i1_2 = 0, i0_3 = 0, i1_3 = 0;
                uint64_t Au2 = 0, Av2 = 0, Au3 = 0, Av3 = 0;
                bool a2 = false, a3 = false;
                uint64_t tw2 = 0, tw3 = 0;

                if constexpr (U == 4) {
                    tw2 = MontgomeryMul(grid, block, debugCombo, tw1, w_strideW);
                    if (tile >= 3) {
                        const uint32_t j0_2 = j0_0 + 2u * W;
                        a2 = (j0_2 < half);
                        if (a2) {
                            i0_2 = k_base + j0_2;
                            i1_2 = i0_2 + half;
                            Au2 = A[i0_2];
                            Av2 = A[i1_2];
                        }
                        if (tile >= 4)
                            tw3 = MontgomeryMul(grid, block, debugCombo, tw2, w_strideW);
                    }
                    if (tile >= 4) {
                        const uint32_t j0_3 = j0_0 + 3u * W;
                        a3 = (j0_3 < half);
                        if (a3) {
                            i0_3 = k_base + j0_3;
                            i1_3 = i0_3 + half;
                            Au3 = A[i0_3];
                            Av3 = A[i1_3];
                        }
                    }
                }

                // ---- compute/store: position 0 ----
                if (a0) {
                    if constexpr (OneTwoThree == Multiway::OneWay) {
                        const uint64_t t0 = MontgomeryMul(grid, block, debugCombo, Av0, tw0);
                        A[i0_0] = AddP(Au0, t0);
                        A[i1_0] = SubP(Au0, t0);
                    } else if constexpr (OneTwoThree == Multiway::TwoWay) {
                        const uint64_t tA0 = MontgomeryMul(grid, block, debugCombo, Av0, tw0);
                        const uint64_t tB0 = MontgomeryMul(grid, block, debugCombo, Bv0, tw0);
                        A[i0_0] = AddP(Au0, tA0);
                        A[i1_0] = SubP(Au0, tA0);
                        B[i0_0] = AddP(Bu0, tB0);
                        B[i1_0] = SubP(Bu0, tB0);
                    } else {
                        const uint64_t tA0 = MontgomeryMul(grid, block, debugCombo, Av0, tw0);
                        const uint64_t tB0 = MontgomeryMul(grid, block, debugCombo, Bv0, tw0);
                        const uint64_t tC0 = MontgomeryMul(grid, block, debugCombo, Cv0, tw0);
                        A[i0_0] = AddP(Au0, tA0);
                        A[i1_0] = SubP(Au0, tA0);
                        B[i0_0] = AddP(Bu0, tB0);
                        B[i1_0] = SubP(Bu0, tB0);
                        C[i0_0] = AddP(Cu0, tC0);
                        C[i1_0] = SubP(Cu0, tC0);
                    }
                }

                // ---- compute/store: position 1 ----
                if (tile >= 2 && a1) {
                    if constexpr (OneTwoThree == Multiway::OneWay) {
                        const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, Av1, tw1);
                        A[i0_1] = AddP(Au1, t1);
                        A[i1_1] = SubP(Au1, t1);
                    } else if constexpr (OneTwoThree == Multiway::TwoWay) {
                        const uint64_t tA1 = MontgomeryMul(grid, block, debugCombo, Av1, tw1);
                        const uint64_t tB1 = MontgomeryMul(grid, block, debugCombo, Bv1, tw1);
                        A[i0_1] = AddP(Au1, tA1);
                        A[i1_1] = SubP(Au1, tA1);
                        B[i0_1] = AddP(Bu1, tB1);
                        B[i1_1] = SubP(Bu1, tB1);
                    } else {
                        const uint64_t tA1 = MontgomeryMul(grid, block, debugCombo, Av1, tw1);
                        const uint64_t tB1 = MontgomeryMul(grid, block, debugCombo, Bv1, tw1);
                        const uint64_t tC1 = MontgomeryMul(grid, block, debugCombo, Cv1, tw1);
                        A[i0_1] = AddP(Au1, tA1);
                        A[i1_1] = SubP(Au1, tA1);
                        B[i0_1] = AddP(Bu1, tB1);
                        B[i1_1] = SubP(Bu1, tB1);
                        C[i0_1] = AddP(Cu1, tC1);
                        C[i1_1] = SubP(Cu1, tC1);
                    }
                }

                // ---- compute/store: positions 2 & 3 (OneWay only) ----
                if constexpr (U == 4) {
                    if (tile >= 3 && a2) {
                        const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, Av2, tw2);
                        A[i0_2] = AddP(Au2, t2);
                        A[i1_2] = SubP(Au2, t2);
                    }
                    if (tile >= 4 && a3) {
                        const uint64_t t3 = MontgomeryMul(grid, block, debugCombo, Av3, tw3);
                        A[i0_3] = AddP(Au3, t3);
                        A[i1_3] = SubP(Au3, t3);
                    }
                }

                // ---- advance within block by 'tile' ----
                j_chunk += tile;
                remaining -= tile;

                if (j_chunk == J_total) {
                    // wrap to next block
                    j_chunk = 0;
                    blk_id += 1;
                    k_base += m;
                    w = w_lane_base;
                } else {
                    // seed twiddle for next tile start
                    if (tile == 1)
                        w = tw1;
                    else if (tile == 2)
                        w = MontgomeryMul(grid, block, debugCombo, tw1, w_strideW);
                    else if (tile == 3)
                        w = tw3; // built above when U==4
                    else         /*tile==4*/
                        w = MontgomeryMul(grid, block, debugCombo, tw3, w_strideW);
                }
            } // while (remaining)
        } // if (T0<Tend)

        // One grid-wide barrier per stage (still required for correctness)
        grid.sync();
    }
}

//==================================================================================================
//                       Pack (base-2^b) and Unpack (to Final128)
//==================================================================================================

template <class SharkFloatParams>
[[nodiscard]] static __device__ SharkForceInlineReleaseOnly uint64_t
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

// Fused GRID-STRIDE unpack for 3 vectors (XX1, YY1, XY1) into 128-bit accumulators,
// WITHOUT atomics. Each thread owns disjoint output word indices j, accumulates locally,
// and writes the 128-bit (lo,hi) pair once.
//
// Final128_* layout: for output word j (32-bit lane), store a 128-bit sum in
//   Final128_[2*j + 0] = lo64, Final128_[2*j + 1] = hi64.
//
// Preconditions:
//  - AXX_norm / AYY_norm / AXY_norm are in NORMAL domain (not Montgomery).
//  - Ddigits is the number of 32-bit words in the destination (covers all q..q+3).
//  - SharkFloatParams::NTTPlan.b is the limb bit-width (<=32).
//  - MagicPrime, HALF, etc., follow your existing defs.
//
template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
UnpackPrimeToFinal128_3Way(cooperative_groups::grid_group &grid,
                           // inputs (normal domain)
                           const uint64_t *SharkRestrict AXX_norm,
                           const uint64_t *SharkRestrict AYY_norm,
                           const uint64_t *SharkRestrict AXY_norm,
                           // outputs (len = 2 * Ddigits; pairs of 64-bit lo/hi)
                           uint64_t *SharkRestrict Final128_XX,
                           uint64_t *SharkRestrict Final128_YY,
                           uint64_t *SharkRestrict Final128_XY,
                           size_t Ddigits)
{
    using namespace SharkNTT;

    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    // Helper: ceil_div for positive integers.
    auto ceil_div_u64 = [](uint64_t a, uint32_t b) -> uint64_t {
        return (a + (uint64_t)b - 1ull) / (uint64_t)b;
    };

    // Local add/sub of a 32-bit quantity into a 128-bit (lo,hi) accumulator.
    auto add32_local = [](uint64_t &lo, uint64_t &hi, uint32_t add32) {
        if (!add32)
            return;
        uint64_t old = lo;
        lo += (uint64_t)add32;
        if (lo < old)
            hi += 1ull;
    };
    auto sub32_local = [](uint64_t &lo, uint64_t &hi, uint32_t sub32) {
        if (!sub32)
            return;
        uint64_t old = lo;
        uint64_t dif = old - (uint64_t)sub32;
        lo = dif;
        if (old < (uint64_t)sub32)
            hi -= 1ull;
    };

    const uint64_t HALF = (SharkNTT::MagicPrime - 1ull) >> 1;
    const int Imax = min(SharkFloatParams::NTTPlan.N, 2 * SharkFloatParams::NTTPlan.L - 1); // same bound as original

    // Grid-stride over output word indices j (each thread owns distinct j).
    for (size_t j = grank; j < Ddigits; j += gsize) {
        // Local 128-bit accumulators for each channel.
        uint64_t xx_lo = 0ull, xx_hi = 0ull;
        uint64_t yy_lo = 0ull, yy_hi = 0ull;
        uint64_t xy_lo = 0ull, xy_hi = 0ull;

        // For each offset t=0..3, collect all i such that j = floor(i*b/32) + t.
        // That implies floor(i*b/32) = j - t.
        // Solve k = j - t; i in [ceil(32*k / b), ceil(32*(k+1)/b) - 1].
        for (int t = 0; t < 4; ++t) {
            if ((int)j - t < 0)
                continue;
            const uint64_t k = (uint64_t)((int)j - t);
            const uint64_t i_lo = ceil_div_u64(32ull * k, (uint32_t)SharkFloatParams::NTTPlan.b);
            const uint64_t i_hi_raw = ceil_div_u64(32ull * (k + 1ull), (uint32_t)SharkFloatParams::NTTPlan.b);
            uint64_t i_hi = (i_hi_raw == 0 ? 0 : (i_hi_raw - 1ull));
            if ((int64_t)i_lo > (int64_t)(Imax - 1))
                continue;
            if (i_hi > (uint64_t)(Imax - 1))
                i_hi = (uint64_t)(Imax - 1);
            if (i_lo > i_hi)
                continue;

            for (uint64_t iu = i_lo; iu <= i_hi; ++iu) {
                const int i = (int)iu;
                const uint64_t sBits = (uint64_t)i * (uint64_t)SharkFloatParams::NTTPlan.b;
                const int r = (int)(sBits & 31);          // shift amount within 32-bit word
                const uint64_t lsh = (r ? (64 - r) : 64); // guard; (mag >> 64)==0 if r==0

                // --------- XX channel ---------
                {
                    const uint64_t v = AXX_norm[i];
                    if (v) {
                        const bool neg = (v > HALF);
                        const uint64_t mag64 = neg ? (SharkNTT::MagicPrime - v) : v;

                        const uint64_t lo64 = r ? (mag64 << r) : mag64;
                        const uint64_t hi64 = r ? (mag64 >> lsh) : 0ull;

                        const uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu);
                        const uint32_t d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu);
                        const uint32_t d2 = (uint32_t)(hi64 & 0xffffffffu);
                        const uint32_t d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);

                        const uint32_t dt = (t == 0) ? d0 : (t == 1) ? d1 : (t == 2) ? d2 : d3;
                        if (!neg)
                            add32_local(xx_lo, xx_hi, dt);
                        else
                            sub32_local(xx_lo, xx_hi, dt);
                    }
                }
                // --------- YY channel ---------
                {
                    const uint64_t v = AYY_norm[i];
                    if (v) {
                        const bool neg = (v > HALF);
                        const uint64_t mag64 = neg ? (SharkNTT::MagicPrime - v) : v;

                        const uint64_t lo64 = r ? (mag64 << r) : mag64;
                        const uint64_t hi64 = r ? (mag64 >> lsh) : 0ull;

                        const uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu);
                        const uint32_t d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu);
                        const uint32_t d2 = (uint32_t)(hi64 & 0xffffffffu);
                        const uint32_t d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);

                        const uint32_t dt = (t == 0) ? d0 : (t == 1) ? d1 : (t == 2) ? d2 : d3;
                        if (!neg)
                            add32_local(yy_lo, yy_hi, dt);
                        else
                            sub32_local(yy_lo, yy_hi, dt);
                    }
                }
                // --------- XY channel ---------
                {
                    const uint64_t v = AXY_norm[i];
                    if (v) {
                        const bool neg = (v > HALF);
                        const uint64_t mag64 = neg ? (SharkNTT::MagicPrime - v) : v;

                        const uint64_t lo64 = r ? (mag64 << r) : mag64;
                        const uint64_t hi64 = r ? (mag64 >> lsh) : 0ull;

                        const uint32_t d0 = (uint32_t)(lo64 & 0xffffffffu);
                        const uint32_t d1 = (uint32_t)((lo64 >> 32) & 0xffffffffu);
                        const uint32_t d2 = (uint32_t)(hi64 & 0xffffffffu);
                        const uint32_t d3 = (uint32_t)((hi64 >> 32) & 0xffffffffu);

                        const uint32_t dt = (t == 0) ? d0 : (t == 1) ? d1 : (t == 2) ? d2 : d3;
                        if (!neg)
                            add32_local(xy_lo, xy_hi, dt);
                        else
                            sub32_local(xy_lo, xy_hi, dt);
                    }
                }
            } // iu
        } // t

        // Write back this thread's 128-bit totals for word j.
        Final128_XX[2 * j + 0] = xx_lo;
        Final128_XX[2 * j + 1] = xx_hi;
        Final128_YY[2 * j + 0] = yy_lo;
        Final128_YY[2 * j + 1] = yy_hi;
        Final128_XY[2 * j + 0] = xy_lo;
        Final128_XY[2 * j + 1] = xy_hi;
    }

    grid.sync(); // ensure all j-words are written before consumers proceed
}

// Grid-strided version: minimize distinct loops, add grid.sync between phases.
// A once -> (XX1, XX2, XY1), then B once -> (YY1, YY2, XY2)
template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
PackTwistFwdNTT_Fused_AB_ToSixOutputs(uint64_t *shared_data,
                                      cooperative_groups::grid_group &grid,
                                      cooperative_groups::thread_block &block,
                                      DebugState<SharkFloatParams> *debugStates,
                                      DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
                                      const HpSharkFloat<SharkFloatParams> &inA,
                                      const HpSharkFloat<SharkFloatParams> &inB,
                                      const SharkNTT::RootTables &roots,
                                      uint64_t *carryPropagationSync,
                                      // six outputs (Montgomery domain, length SharkFloatParams::NTTPlan.N)
                                      uint64_t *SharkRestrict tempDigitsXX1,
                                      uint64_t *SharkRestrict tempDigitsXX2,
                                      uint64_t *SharkRestrict tempDigitsYY1,
                                      uint64_t *SharkRestrict tempDigitsYY2,
                                      uint64_t *SharkRestrict tempDigitsXY1,
                                      uint64_t *SharkRestrict tempDigitsXY2)
{
    const uint32_t N = static_cast<uint32_t>(SharkFloatParams::NTTPlan.N);
    const uint32_t L = static_cast<uint32_t>(SharkFloatParams::NTTPlan.L);
    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    const uint64_t zero_m = ToMontgomeryConstexpr(0ull);

    // -------------------- Phase A: pack+twist with tail zero (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        if (i < L) {
            const uint64_t coeff = ReadBitsSimple(inA, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmod = coeff % MagicPrime; // match original
            const uint64_t xm = ToMontgomery(grid, block, debugMultiplyCounts, cmod);
            const uint64_t psik = roots.psi_pows[i]; // Montgomery domain
            tempDigitsXX1[i] = MontgomeryMul(grid, block, debugMultiplyCounts, xm, psik);
        } else {
            tempDigitsXX1[i] = zero_m;
        }

        if (i < L) {
            const uint64_t coeffB = ReadBitsSimple(inB, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmodB = coeffB % MagicPrime;
            const uint64_t xmB = ToMontgomery(grid, block, debugMultiplyCounts, cmodB);
            const uint64_t psiB = roots.psi_pows[i]; // Montgomery domain
            tempDigitsYY1[i] = MontgomeryMul(grid, block, debugMultiplyCounts, xmB, psiB);
        } else {
            tempDigitsYY1[i] = zero_m;
        }
    }

    //
    // Note: the next couple checksums have some redundancy because of the
    // way the reference implementation works.
    //

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        debugStates[static_cast<int>(DebugStatePurpose::Z1XX)].Reset(
            DebugStatePurpose::Z1XX, debugStates[static_cast<int>(DebugStatePurpose::Z0XX)]);
        debugStates[static_cast<int>(DebugStatePurpose::Z0XY)].Reset(
            DebugStatePurpose::Z0XY, debugStates[static_cast<int>(DebugStatePurpose::Z0XX)]);

        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        debugStates[static_cast<int>(DebugStatePurpose::Z1YY)].Reset(
            DebugStatePurpose::Z1YY, debugStates[static_cast<int>(DebugStatePurpose::Z0YY)]);
        debugStates[static_cast<int>(DebugStatePurpose::Z1XY)].Reset(
            DebugStatePurpose::Z1XY, debugStates[static_cast<int>(DebugStatePurpose::Z0YY)]);
        grid.sync();
    } else {
        grid.sync();
    }

    // A: forward NTT (grid-wide helpers)
    BitReverseInplace64_GridStride<Multiway::TwoWay>(
        grid, tempDigitsXX1, tempDigitsYY1, nullptr, N, (uint32_t)SharkFloatParams::NTTPlan.stages);

    grid.sync();

    NTTRadix2_GridStride<SharkFloatParams, Multiway::TwoWay>(shared_data,
                                                             grid,
                                                             block,
                                                             debugMultiplyCounts,
                                                             carryPropagationSync,
                                                             tempDigitsXX1,
                                                             tempDigitsYY1,
                                                             nullptr,
                                                             N,
                                                             (uint32_t)SharkFloatParams::NTTPlan.stages,
                                                             roots.stage_omegas);

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        debugStates[static_cast<int>(DebugStatePurpose::Z3XX)].Reset(
            DebugStatePurpose::Z3XX, debugStates[static_cast<int>(DebugStatePurpose::Z2XX)]);
        debugStates[static_cast<int>(DebugStatePurpose::Z2XY)].Reset(
            DebugStatePurpose::Z2XY, debugStates[static_cast<int>(DebugStatePurpose::Z2XX)]);

        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        debugStates[static_cast<int>(DebugStatePurpose::Z3YY)].Reset(
            DebugStatePurpose::Z3YY, debugStates[static_cast<int>(DebugStatePurpose::Z2YY)]);
        debugStates[static_cast<int>(DebugStatePurpose::Z3XY)].Reset(
            DebugStatePurpose::Z3XY, debugStates[static_cast<int>(DebugStatePurpose::Z2YY)]);
        grid.sync();
    } else {
        grid.sync();
    }

    // -------------------- Final replicate of B (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        // Replicate A spectrum
        const uint64_t vA = tempDigitsXX1[i];
        tempDigitsXX2[i] = vA;
        tempDigitsXY1[i] = vA;

        // Replicate B spectrum
        const uint64_t vB = tempDigitsYY1[i];
        tempDigitsYY2[i] = vB;
        tempDigitsXY2[i] = vB;
    }
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
static __device__ SharkForceInlineReleaseOnly void
UntwistScaleFromMont_3Way_GridStride(cooperative_groups::grid_group &grid,
                                     cooperative_groups::thread_block &block,
                                     DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
                                     const SharkNTT::RootTables &roots,
                                     uint64_t *SharkRestrict tempDigitsXX1,
                                     uint64_t *SharkRestrict tempDigitsYY1,
                                     uint64_t *SharkRestrict tempDigitsXY1)
{
    using namespace SharkNTT;

    const size_t N = static_cast<size_t>(SharkFloatParams::NTTPlan.N);
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
}

} // namespace SharkNTT

template <class SharkFloatParams, DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly static void
EraseCurrentDebugState(DebugState<SharkFloatParams> *debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block)
{
    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugStates[curPurpose].Erase(
        grid, block, Purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams, DebugStatePurpose Purpose, typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState(DebugState<SharkFloatParams> *SharkRestrict debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       const ArrayType *arrayToChecksum,
                       size_t arraySize)
{

    constexpr auto CurPurpose = static_cast<int32_t>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;
    constexpr auto CallIndex = 0;

    debugStates[CurPurpose].Reset(UseConvolutionHere,
                                  grid,
                                  block,
                                  arrayToChecksum,
                                  arraySize,
                                  Purpose,
                                  RecursionDepth,
                                  CallIndex);
}

// Look for CalculateNTTFrameSize
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArraysForMultiply.
static_assert(AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets()                                                                     \
    const int threadIdxGlobal =                                                                         \
        block.group_index().x * SharkFloatParams::GlobalThreadsPerBlock + block.thread_index().x;       \
    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;                                            \
    constexpr int TestMultiplier = 1;                                                                   \
    constexpr auto Multiplies_offset = AdditionalGlobalSyncSpace;                                       \
    constexpr auto Checksum_offset = Multiplies_offset + AdditionalGlobalMultipliesPerThread;           \
    constexpr auto GlobalsDoneOffset = Checksum_offset + AdditionalGlobalChecksumSpace;                 \
    constexpr auto Z0_offsetXX = GlobalsDoneOffset;                                                     \
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
    constexpr auto CarryInsEnd = CarryInsOffset + 3 * NewN + CalcAlign16Bytes64BitIndex(3 * NewN);

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunNTT_3Way_Multiply(uint64_t *shared_data,
                     HpSharkFloat<SharkFloatParams> *outXX,
                     HpSharkFloat<SharkFloatParams> *outYY,
                     HpSharkFloat<SharkFloatParams> *outXY,
                     const HpSharkFloat<SharkFloatParams> &inA,
                     const HpSharkFloat<SharkFloatParams> &inB,
                     const SharkNTT::RootTables &roots,
                     cg::grid_group &grid,
                     cg::thread_block &block,
                     DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
                     DebugState<SharkFloatParams> *debugStates,
                     uint64_t *tempDigitsXX1,
                     uint64_t *tempDigitsXX2,
                     uint64_t *tempDigitsYY1,
                     uint64_t *tempDigitsYY2,
                     uint64_t *tempDigitsXY1,
                     uint64_t *tempDigitsXY2,
                     uint64_t *Final128_XX,
                     uint64_t *Final128_YY,
                     uint64_t *Final128_XY,
                     uint64_t *CarryPropagationBuffer,
                     uint64_t *CarryPropagationBuffer2,
                     uint64_t *CarryPropagationSync,
                     size_t Ddigits)
{
    PackTwistFwdNTT_Fused_AB_ToSixOutputs<SharkFloatParams>(shared_data,
                                                            grid,
                                                            block,
                                                            debugStates,
                                                            debugMultiplyCounts,
                                                            inA,
                                                            inB,
                                                            roots,
                                                            CarryPropagationSync,
                                                            tempDigitsXX1,
                                                            tempDigitsXX2,
                                                            tempDigitsYY1,
                                                            tempDigitsYY2,
                                                            tempDigitsXY1,
                                                            tempDigitsXY2);

    // Note: no grid.sync.  The last operation done in the prior function
    // is grid-wide and the next loop operates on the same data per-thread
    // so there is no hazard.

    const size_t N = static_cast<size_t>(SharkFloatParams::NTTPlan.N);
    const size_t gsize = grid.size();
    const size_t grank = grid.thread_rank();

    for (size_t i = grank; i < N; i += gsize) {
        const uint64_t aXX = tempDigitsXX1[i];
        const uint64_t bXX = tempDigitsXX2[i];
        tempDigitsXX1[i] = SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, aXX, bXX);

        const uint64_t aYY = tempDigitsYY1[i];
        const uint64_t bYY = tempDigitsYY2[i];
        tempDigitsYY1[i] = SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, aYY, bYY);

        const uint64_t aXY = tempDigitsXY1[i];
        const uint64_t bXY = tempDigitsXY2[i];
        tempDigitsXY1[i] = SharkNTT::MontgomeryMul(grid, block, debugMultiplyCounts, aXY, bXY);
    }

    grid.sync();

    // 5) Inverse NTT (in place on Z0_OutDigits)
    SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::ThreeWay>(
        grid, tempDigitsXX1, tempDigitsYY1, tempDigitsXY1, (uint32_t)SharkFloatParams::NTTPlan.N, (uint32_t)SharkFloatParams::NTTPlan.stages);

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3, uint64_t>(
            debugStates, grid, block, tempDigitsXY1, SharkFloatParams::NTTPlan.N);
        grid.sync();
    } else {
        grid.sync();
    }

    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::ThreeWay>(
        shared_data,
        grid,
        block,
        debugMultiplyCounts,
        CarryPropagationSync,
        tempDigitsXX1,
        tempDigitsYY1,
        tempDigitsXY1,
        (uint32_t)SharkFloatParams::NTTPlan.N,
        (uint32_t)SharkFloatParams::NTTPlan.stages,
        roots.stage_omegas_inv);

    // --- After inverse NTTs (XX1 / YY1 / XY1 are in Montgomery domain) ---
    grid.sync(); // make sure prior writes (inv-NTT) are visible

    UntwistScaleFromMont_3Way_GridStride<SharkFloatParams>(grid,
                                                           block,
                                                           debugMultiplyCounts,
                                                           roots,
                                                           /* XX1 */ tempDigitsXX1,
                                                           /* YY1 */ tempDigitsYY1,
                                                           /* XY1 */ tempDigitsXY1);

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6, uint64_t>(
            debugStates, grid, block, tempDigitsXY1, SharkFloatParams::NTTPlan.N);
        grid.sync();
    } else {
        grid.sync();
    }

    // The helper does a final grid.sync() internally.
    // At this point, tempDigitsXX1/YY1/XY1 are back in the normal domain (not Montgomery).

    // 8) Unpack (fused 3-way) -> Final128 -> Normalize
    SharkNTT::UnpackPrimeToFinal128_3Way<SharkFloatParams>(grid,
                               /* AXX_norm */ tempDigitsXX1,
                               /* AYY_norm */ tempDigitsYY1,
                               /* AXY_norm */ tempDigitsXY1,
                               /* FinalXX */ Final128_XX,
                               /* FinalYY */ Final128_YY,
                               /* FinalXY */ Final128_XY,
                               /* Ddigits */ Ddigits);

    grid.sync(); // subsequent phases depend on Final128_* fully written

    outXX->SetNegative(false);
    outYY->SetNegative(false);

    const auto OutXYIsNegative = (inA.GetNegative() ^ inB.GetNegative());
    outXY->SetNegative(OutXYIsNegative);

    const auto addFactorOfTwoXX = 0;
    const auto addFactorOfTwoYY = 0;
    const auto addFactorOfTwoXY = 1;

    // --- Workspaces ---
    // dynamic shared mem: need >= 6 * blockDim.x * sizeof(uint64_t)
    //   - you already pass dynamic SMEM at launch; just alias it here

    // scratch result digits (2*N32 per channel, uint64_t each; low 32 bits used)
    uint64_t *resultXX = tempDigitsXX1; /* device buffer length 2*N32 */
    uint64_t *resultYY = tempDigitsYY1; /* device buffer length 2*N32 */
    uint64_t *resultXY = tempDigitsXY1; /* device buffer length 2*N32 */

    // ---- Single fused normalize for XX, YY, XY ----
    SharkNTT::Normalize_GridStride_3Way<SharkFloatParams>(grid,
                                                          block,
                                                          debugStates,
                                                          /* outXX */ *outXX,
                                                          /* outYY */ *outYY,
                                                          /* outXY */ *outXY,
                                                          /* inA   */ inA,
                                                          /* inB   */ inB,
                                                          /* Final128_XX */ Final128_XX,
                                                          /* Final128_YY */ Final128_YY,
                                                          /* Final128_XY */ Final128_XY,
                                                          /* Ddigits     */ Ddigits,
                                                          /* addTwoXX    */ addFactorOfTwoXX,
                                                          /* addTwoYY    */ addFactorOfTwoYY,
                                                          /* addTwoXY    */ addFactorOfTwoXY,
                                                          /* shared_data      */ CarryPropagationBuffer2,
                                                          /* block_carry_outs */ CarryPropagationBuffer,
                                                          /* globalCarryCheck */ CarryPropagationSync,
                                                          /* resultXX scratch */ resultXX,
                                                          /* resultYY scratch */ resultYY,
                                                          /* resultXY scratch */ resultXY);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
MultiplyHelperNTTV2Separates(const SharkNTT::RootTables &roots,
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict A,
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict B,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutXX,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutXY,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutYY,
                             cg::grid_group &grid,
                             cg::thread_block &block,
                             uint64_t *SharkRestrict tempProducts)
{

    extern __shared__ uint64_t shared_data[];

    constexpr auto ExecutionNumBlocks = SharkFloatParams::GlobalNumBlocks;

    DefineTempProductsOffsets();

    // TODO: indexes
    auto *SharkRestrict debugMultiplyCounts =
        reinterpret_cast<DebugMultiplyCount<SharkFloatParams> *>(&tempProducts[Multiplies_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams> *>(&tempProducts[Checksum_offset]);

    if constexpr (HpShark::PrintMultiplyCounts) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();
    }

    if constexpr (HpShark::DebugChecksums) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread]
            .DebugMultiplyErase();

        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfHigh>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfHigh>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetXY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1_offsetYY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128XY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Final128YY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd3>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXX>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetXY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_offsetYY>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add1>(
            debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add2>(
            debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 47,
                      "Unexpected number of purposes");
    }

    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits, uint32_t>(
            debugStates, grid, block, A->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits, uint32_t>(
            debugStates, grid, block, B->Digits, NewN);
    }

    // x must be a positive constant expression

    // Verify power of 2
    static_assert(SharkFloatParams::GlobalNumUint32 > 0 &&
                      (SharkFloatParams::GlobalNumUint32 & (SharkFloatParams::GlobalNumUint32 - 1)) == 0,
                  "GlobalNumUint32 must be a power of 2");

    // Compute Final128 digit budget once
    const uint32_t Ddigits =
        ((uint64_t)((2 * SharkFloatParams::NTTPlan.L - 2) * SharkFloatParams::NTTPlan.b + 64) + 31u) /
            32u +
        2u;

    // ---- Single allocation for entire core path ----
    uint64_t *buffer = &tempProducts[GlobalsDoneOffset];

    // TODO: Assert or something somewhere
    // const size_t buf_count = (size_t)2 * (size_t)SharkFloatParams::NTTPlan.N     // Z0_OutDigits + Z2_OutDigits
    //                         + (size_t)2 * (size_t)Ddigits; // Final128 (lo,hi per 32-bit slot)

    // Slice buffer into spans
    size_t off = 0;
    uint64_t *tempDigitsXX1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;
    uint64_t *tempDigitsXX2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;

    uint64_t *tempDigitsYY1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;
    uint64_t *tempDigitsYY2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;

    uint64_t *tempDigitsXY1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;
    uint64_t *tempDigitsXY2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += (size_t)SharkFloatParams::NTTPlan.N;

    uint64_t *Final128_XX = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *Final128_YY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *Final128_XY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *CarryPropagationBuffer = buffer + off; // 3 * NewN
    off += 3 * NewN;

    uint64_t *CarryPropagationBuffer2 = buffer + off; // 3 * NewN
    off += 6 + grid.size() * 6;

    uint64_t *CarryPropagationSync = &tempProducts[0];

    // XX = A^2
    RunNTT_3Way_Multiply<SharkFloatParams>(shared_data,
                                           OutXX,
                                           OutYY,
                                           OutXY,
                                           *A,
                                           *B,
                                           roots,
                                           grid,
                                           block,
                                           debugMultiplyCounts,
                                           debugStates,
                                           tempDigitsXX1,
                                           tempDigitsXX2,
                                           tempDigitsYY1,
                                           tempDigitsYY2,
                                           tempDigitsXY1,
                                           tempDigitsXY2,
                                           Final128_XX,
                                           Final128_YY,
                                           Final128_XY,
                                           CarryPropagationBuffer,
                                           CarryPropagationBuffer2,
                                           CarryPropagationSync,
                                           Ddigits);

    grid.sync();
}

template <class SharkFloatParams>
void
PrintMaxActiveBlocks(void *kernelFn, int sharedAmountBytes)
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
MultiplyHelperNTT(HpSharkComboResults<SharkFloatParams> *SharkRestrict combo,
                  cg::grid_group &grid,
                  cg::thread_block &block,
                  uint64_t *SharkRestrict tempProducts)
{
    MultiplyHelperNTTV2Separates<SharkFloatParams>(combo->Roots,
                                                   &combo->A,
                                                   &combo->B,
                                                   &combo->ResultX2,
                                                   &combo->Result2XY,
                                                   &combo->ResultY2,
                                                   grid,
                                                   block,
                                                   tempProducts);
}
