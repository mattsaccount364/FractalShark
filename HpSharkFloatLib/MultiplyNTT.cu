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

#include "MontgomeryCoreConstexpr.cuh"
#include "NTTConstexprGenerator.h"

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

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
FinalizeNormalize(cooperative_groups::grid_group &grid,
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
                  const uint32_t Ddigits,
                  const int32_t addTwoXX,
                  const int32_t addTwoYY,
                  const int32_t addTwoXY,
                  // global workspaces (NOT shared memory)
                  uint64_t *SharkRestrict CarryPropagationBuffer2, // >= 6 + 6*lanes u64
                  uint64_t *SharkRestrict CarryPropagationBuffer,
                  uint64_t *SharkRestrict resultXX, // len >= Ddigits
                  uint64_t *SharkRestrict resultYY, // len >= Ddigits
                  uint64_t *SharkRestrict resultXY) // len >= Ddigits
{
    // We only ever produce digits in [0, Ddigits).
    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    // --- 3) Scan [0, Ddigits) for highest-nonzero; compute shifts/exponents (single thread) ---
    int h_xx = -1, h_yy = -1, h_xy = -1;
    if (tid == 0) {
        for (int i = Ddigits - 1; i >= 0; --i) {
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
            int shift = significant - SharkFloatParams::GlobalNumUint32;
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

    // --- 4) Grid-stride write the SharkFloatParams::GlobalNumUint32-digit windows (bounds-safe into [0,
    // Ddigits)) ---
    for (int i = tid; i < SharkFloatParams::GlobalNumUint32; i += T_all) {
        // XX
        outXX.Digits[i] =
            zXX ? 0u : ((sXX + i < Ddigits) ? static_cast<uint32_t>(resultXX[sXX + i]) : 0u);

        // YY
        outYY.Digits[i] =
            zYY ? 0u : ((sYY + i < Ddigits) ? static_cast<uint32_t>(resultYY[sYY + i]) : 0u);

        // XY
        outXY.Digits[i] =
            zXY ? 0u : ((sXY + i < Ddigits) ? static_cast<uint32_t>(resultXY[sXY + i]) : 0u);
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

#include "MultiplyNTT_NormalizePPv3.cuh"
#include "MultiplyNTT_NormalizeV1.cuh"
#include "MultiplyNTT_NormalizeWarpTiledV2.cuh"

template <class SharkFloatParams>
static __device__ inline void
Normalize_GridStride_3WayV2(cooperative_groups::grid_group &grid,
                            cooperative_groups::thread_block &block,
                            DebugGlobalCount<SharkFloatParams> *debugGlobalState,
                            DebugState<SharkFloatParams> *debugStates,
                            // outputs
                            HpSharkFloat<SharkFloatParams> &outXX,
                            HpSharkFloat<SharkFloatParams> &outYY,
                            HpSharkFloat<SharkFloatParams> &outXY,
                            // input exponents
                            const HpSharkFloat<SharkFloatParams> &inA,
                            const HpSharkFloat<SharkFloatParams> &inB,
                            // convolution sums (lo,hi per 32-bit position), NORMAL domain
                            uint64_t *final128XX,
                            uint64_t *final128YY,
                            uint64_t *final128XY,
                            const uint32_t Ddigits,
                            const int32_t addTwoXX,
                            const int32_t addTwoYY,
                            const int32_t addTwoXY,
                            // global workspaces (NOT shared memory)
                            uint64_t *SharkRestrict CarryPropagationBuffer2, // >= 6 + 6*lanes uint64_t
                            uint64_t *SharkRestrict CarryPropagationBuffer,
                            uint64_t *globalSync1,            // 1 uint64_t
                            uint64_t *globalSync2,            // 1 uint64_t
                            uint64_t *SharkRestrict resultXX, // len >= Ddigits
                            uint64_t *SharkRestrict resultYY, // len >= Ddigits
                            uint64_t *SharkRestrict resultXY) // len >= Ddigits
{
    // We only ever produce digits in [0, Ddigits).
#ifdef TEST_SMALL_NORMALIZE_WARP
    constexpr int warpSz = block.dim_threads().x;
#else
    constexpr int warpSz = 32;
#endif

    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int totalThreads = grid.size();
    const int totalWarps = max(1, totalThreads / warpSz);
    const unsigned fullMask = __activemask();
    const int lane = block.thread_index().x & (warpSz - 1);

    auto *cur = CarryPropagationBuffer;
    auto *next = CarryPropagationBuffer2;

    // init cur/next = 0 (length Ddigits+1 to include high slot)
    for (int i = tid; i <= Ddigits * 6; i += totalThreads) {
        cur[i] = 0;
        next[i] = 0;
    }

    // We run the tile chain in a single warp for now (warp 0) so that tile
    // carries are propagated correctly and in order. Other warps do nothing.
    uint64_t prevGlobalSync1 = std::numeric_limits<uint64_t>::max();
    *globalSync1 = 0;
    *globalSync2 = 0;

    auto swap2 = [](uint64_t *&a, uint64_t *&b) {
        auto *t = a;
        a = b;
        b = t;
    };

    grid.sync();

    for (int index = tid; index < Ddigits; index += totalThreads) {
        uint64_t carry_loXX = 0;
        uint64_t carry_loYY = 0;
        uint64_t carry_loXY = 0;

        const auto indexT2 = 2 * index;

        auto ProcessOneStart =
            [](size_t indexT2, uint64_t *final128, uint32_t &outDig, uint64_t &outCarry) -> void {
            const uint64_t lo = final128[indexT2];
            const uint64_t hi = final128[indexT2 + 1];
            const uint32_t dig = static_cast<uint32_t>(lo & 0xffffffffu);

            outDig = dig;
            outCarry = (lo >> 32) | (hi << 32);
        };

        uint32_t outXXDig, outYYDig, outXYDig;

        ProcessOneStart(indexT2, final128XX, outXXDig, carry_loXX);
        ProcessOneStart(indexT2, final128YY, outYYDig, carry_loYY);
        ProcessOneStart(indexT2, final128XY, outXYDig, carry_loXY);

        resultXX[index] = outXXDig;
        cur[index * 3 + 3 + 0] = carry_loXX;

        resultYY[index] = outYYDig;
        cur[index * 3 + 3 + 1] = carry_loYY;

        resultXY[index] = outXYDig;
        cur[index * 3 + 3 + 2] = carry_loXY;
    }

    grid.sync();

    for (int index = tid; index < Ddigits; index += totalThreads) {
        const uint64_t carry_loXX = cur[index * 3 + 0];
        const uint64_t carry_loYY = cur[index * 3 + 1];
        const uint64_t carry_loXY = cur[index * 3 + 2];

        auto ProcessOneStart = [](const size_t index,
                                  const uint64_t *result,
                                  uint32_t &outDig,
                                  const uint64_t carryIn,
                                  uint64_t &carryOut) -> void {
            const uint64_t inDig = result[index]; // 32-bit digit in low bits
            const uint64_t fullDig = inDig + carryIn;
            const uint64_t c0 = (fullDig < inDig) ? 1ull : 0ull;
            outDig = fullDig & 0xffffffffu;
            carryOut = (fullDig >> 32) | (c0 << 32);
        };

        uint32_t outXXDig, outYYDig, outXYDig;
        uint64_t carry_outXX, carry_outYY, carry_outXY;

        ProcessOneStart(index, resultXX, outXXDig, carry_loXX, carry_outXX);
        ProcessOneStart(index, resultYY, outYYDig, carry_loYY, carry_outYY);
        ProcessOneStart(index, resultXY, outXYDig, carry_loXY, carry_outXY);

        resultXX[index] = outXXDig;
        next[index * 3 + 3 + 0] = carry_outXX;

        resultYY[index] = outYYDig;
        next[index * 3 + 3 + 1] = carry_outYY;

        resultXY[index] = outXYDig;
        next[index * 3 + 3 + 2] = carry_outXY;
    }

    grid.sync();

    for (int index = tid; index < Ddigits; index += totalThreads) {
        const uint64_t carry_loXX = next[index * 3 + 0];
        const uint64_t carry_loYY = next[index * 3 + 1];
        const uint64_t carry_loXY = next[index * 3 + 2];

        next[index * 3 + 0] = 0;
        next[index * 3 + 1] = 0;
        next[index * 3 + 2] = 0;

        auto ProcessOneStart = [](const size_t index,
                                  const uint64_t *result,
                                  uint32_t &outDig,
                                  const uint64_t carryIn,
                                  uint64_t &carryOut) -> void {
            // This stage cannot overflow
            const uint64_t inDig = result[index]; // 32-bit digit in low bits
            const uint64_t fullDig = inDig + carryIn;
            outDig = fullDig & 0xffffffffu;
            carryOut = fullDig >> 32; // 0 or 1 yay
        };

        uint32_t outXXDig, outYYDig, outXYDig;
        uint64_t carry_outXX, carry_outYY, carry_outXY;

        ProcessOneStart(index, resultXX, outXXDig, carry_loXX, carry_outXX);
        ProcessOneStart(index, resultYY, outYYDig, carry_loYY, carry_outYY);
        ProcessOneStart(index, resultXY, outXYDig, carry_loXY, carry_outXY);

        resultXX[index] = outXXDig;
        cur[index * 3 + 3 + 0] = carry_outXX;

        resultYY[index] = outYYDig;
        cur[index * 3 + 3 + 1] = carry_outYY;

        resultXY[index] = outXYDig;
        cur[index * 3 + 3 + 2] = carry_outXY;

        if (carry_outXX != 0 || carry_outYY != 0 || carry_outXY != 0) {
            *globalSync2 = 1;
        }
    }

    grid.sync();

    const auto globalResult = *globalSync2;

#ifdef TILE_VERSION
    int32_t iteration = 0;

    const int warpId = tid / warpSz;
    const int numTiles = (Ddigits + warpSz - 1) / warpSz;

    if (globalResult != 0) {

        uint32_t carry_lo;

        for (;;) {

            for (int tile = warpId; tile < numTiles; tile += totalWarps) {
                const int base = tile * warpSz;
                const auto basePlusLane = base + lane;

                if (iteration > 0) {
                    if (lane == 0) {
                        carry_lo = cur[base];
                    } else {
                        carry_lo = 0;
                    }
                } else {
                    // Use carries produced above.
                    carry_lo = 0;
                    carry_lo |= (cur[basePlusLane * 3 + 0] << 0);
                    carry_lo |= (cur[basePlusLane * 3 + 1] << 1);
                    carry_lo |= (cur[basePlusLane * 3 + 2] << 2);

                    cur[basePlusLane * 3 + 0] = 0;
                    cur[basePlusLane * 3 + 1] = 0;
                    cur[basePlusLane * 3 + 2] = 0;
                }

                auto loXX = static_cast<uint32_t>(resultXX[basePlusLane]);
                auto loYY = static_cast<uint32_t>(resultYY[basePlusLane]);
                auto loXY = static_cast<uint32_t>(resultXY[basePlusLane]);

                // Warp-tiled normalize for this tile; operates purely in registers.
                const WarpNormalizeTriple tout = WarpNormalizeTile<SharkFloatParams>(
                    fullMask, Ddigits, lane, tile, iteration, loXX, loYY, loXY, carry_lo);

                resultXX[basePlusLane] = loXX;
                resultYY[basePlusLane] = loYY;
                resultXY[basePlusLane] = loXY;

                const int outIdx = min(base + warpSz, Ddigits);
                if (lane == warpSz - 1 || (base + lane == Ddigits - 1)) {
                    if (outIdx < Ddigits) {
                        next[outIdx] = tout.carry_lo;
                    } else {
                        next[outIdx] |= tout.carry_lo;
                    }

                    if (tout.changedMask) {
                        atomicAdd(globalSync1, 1);
                    }
                }
            }

            grid.sync();

            {
                const auto temp = *globalSync1;
                if (temp == prevGlobalSync1) {
                    break;
                }

                prevGlobalSync1 = temp;

                // Swap only the active streams (mirror of your original logic)
                swap2(cur, next);
                iteration++;
            }

            if constexpr (HpShark::DebugGlobalState) {
                DebugNormalizeIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
            }

            grid.sync();
        }
    }
#else
    if (globalResult != 0) {
        // Reinterpret your existing carry buffers as transfer- and prefix-scan storage.
        auto *digitXfer = reinterpret_cast<DigitTransfer3 *>(final128XX);
        auto *scanTemp = reinterpret_cast<DigitTransfer3 *>(final128XY);

        // Use the "front" of CarryPropagationBuffer2 as carryInMask (3 bits per entry).
        auto *carryInMask = reinterpret_cast<uint32_t *>(final128YY);

        ParallelPrefixNormalize3WayV3<SharkFloatParams>(
            grid, block, cur, Ddigits, resultXX, resultYY, resultXY, digitXfer, scanTemp, carryInMask);
    }
#endif

    FinalizeNormalize<SharkFloatParams>(grid,
                                        block,
                                        debugStates,
                                        outXX,
                                        outYY,
                                        outXY,
                                        inA,
                                        inB,
                                        Ddigits,
                                        addTwoXX,
                                        addTwoYY,
                                        addTwoXY,
                                        CarryPropagationBuffer2,
                                        CarryPropagationBuffer,
                                        resultXX,
                                        resultYY,
                                        resultXY);
}

// This one does work, but is just slow because it's sequential
template <class SharkFloatParams>
static __device__ inline void
Normalize_GridStride_3WaySeqV2(cooperative_groups::grid_group &grid,
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
                               uint64_t *final128XX,
                               uint64_t *final128YY,
                               uint64_t *final128XY,
                               const uint32_t Ddigits,
                               const int32_t addTwoXX,
                               const int32_t addTwoYY,
                               const int32_t addTwoXY,
                               // global workspaces (NOT shared memory)
                               uint64_t *SharkRestrict
                                   CarryPropagationBuffer2, // >= 6 + 6*lanes uint64_t
                               uint64_t *SharkRestrict CarryPropagationBuffer,
                               uint64_t *SharkRestrict globalCarryCheck, // 1 uint64_t
                               uint64_t *SharkRestrict resultXX,         // len >= Ddigits
                               uint64_t *SharkRestrict resultYY,         // len >= Ddigits
                               uint64_t *SharkRestrict resultXY)         // len >= Ddigits
{
    // We only ever produce digits in [0, Ddigits).
    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;

    // --- 1) Initial pass over our slice (no tail beyond Ddigits) ---
    uint64_t *prevXX = final128XX;
    uint64_t *prevYY = final128YY;
    uint64_t *prevXY = final128XY;

    uint64_t *curXX = resultXX;
    uint64_t *curYY = resultYY;
    uint64_t *curXY = resultXY;

    // --- 2) Iterative carry propagation within [0, Ddigits) (drop at right edge) ---
    if (tid == 0) {
        uint64_t carry_loXX = 0;
        uint64_t carry_loXY = 0;
        uint64_t carry_loYY = 0;

        uint64_t carry_hiXX = 0;
        uint64_t carry_hiXY = 0;
        uint64_t carry_hiYY = 0;

        size_t index = 0, indexOut = 0;

        auto ProcessOne = [&](size_t index,
                              size_t indexOut,
                              uint64_t *cur,
                              uint64_t *prev,
                              uint64_t &carry_lo,
                              uint64_t &carry_hi) -> void {
            const uint64_t lo = prev[index];
            const uint64_t hi = prev[index + 1];

            const uint64_t s_lo = lo + carry_lo;
            const uint64_t c0 = (s_lo < lo) ? 1u : 0u;
            const uint64_t s_hi = hi + carry_hi + c0;

            const uint32_t dig = static_cast<uint32_t>(s_lo & 0xffffffffu);
            cur[indexOut] = dig;

            carry_lo = (s_lo >> 32) | (s_hi << 32);
            carry_hi = (s_hi >> 32);
        };

        for (; index < 2 * Ddigits;) {
            ProcessOne(index, indexOut, curXX, prevXX, carry_loXX, carry_hiXX);
            ProcessOne(index, indexOut, curYY, prevYY, carry_loXY, carry_hiXY);
            ProcessOne(index, indexOut, curXY, prevXY, carry_loYY, carry_hiYY);
            indexOut++;
            index += 2;
        }
    }
    grid.sync();

    FinalizeNormalize<SharkFloatParams>(grid,
                                        block,
                                        debugStates,
                                        outXX,
                                        outYY,
                                        outXY,
                                        inA,
                                        inB,
                                        Ddigits,
                                        addTwoXX,
                                        addTwoYY,
                                        addTwoXY,
                                        CarryPropagationBuffer2,
                                        CarryPropagationBuffer,
                                        globalCarryCheck,
                                        resultXX,
                                        resultYY,
                                        resultXY);
}

//--------------------------------------------------------------------------------------------------
// 64×64→128 helpers (compiler/ABI specific intrinsics)
//--------------------------------------------------------------------------------------------------

static __device__ SharkForceInlineReleaseOnly uint64_t
Add64WithCarry(uint64_t a, uint64_t b, uint64_t &carry)
{
    const uint64_t s = a + b;
    const uint64_t c = (s < a);
    const uint64_t out = s + carry;
    carry = c | (out < s);
    return out;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
Add64WithCarryInOnly(uint64_t a, uint64_t b, uint64_t carry)
{
    const uint64_t s = a + b;
    const uint64_t out = s + carry;
    return out;
}

static __device__ SharkForceInlineReleaseOnly void
Add64WithCarryVoid(uint64_t a, uint64_t b, uint64_t &carry)
{
    const uint64_t s = a + b;
    carry = (s < a);
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
              DebugGlobalCount<SharkFloatParams> *debugCombo,
              uint64_t a,
              uint64_t b)
{
    // Debug instrumentation (optionally compiled out via if constexpr).
    // Count as 7 "64-bit mul-equivalents": 3 for a*b, 3 for m*p, 1 for the add path.
    if constexpr (HpShark::DebugGlobalState) {
        DebugMultiplyIncrement<SharkFloatParams>(debugCombo, grid, block, 7);
    }

    // ---------------------------------------------------------------------
    // The modulus for this Montgomery domain is SharkNTT::MagicPrime.
    // Montgomery reduction computes:
    //
    //     r = (a * b * R^{-1}) mod p     (where R = 2^64)
    //
    // Given:
    //   NINV = -p^{-1} mod 2^64
    //
    // We compute:
    //   t      = a * b           (128-bit)
    //   m      = (t_lo * NINV)   (mod R)
    //   mp     = m * p           (128-bit)
    //   u      = t + mp          (128-bit)
    //   r      = u_hi (upper 64 bits)
    //
    // And finally, ensure r < p by subtracting p if needed.
    //
    // PTX is used to explicitly control 128-bit math via mul.lo/mul.hi and add.cc/addc.
    // ---------------------------------------------------------------------

    uint64_t t_lo, t_hi;   // 128-bit product of a*b
    uint64_t m;            // m = t_lo * MagicPrimeInv (mod 2^64)
    uint64_t mp_lo, mp_hi; // 128-bit product m * MagicPrime

    // ---------------------------------------------------------------------
    // Compute:
    //   t_lo  = (a * b) low  64 bits
    //   t_hi  = (a * b) high 64 bits
    //   m     = (t_lo * MagicPrimeInv) mod 2^64   (Montgomery trick)
    //   mp_lo = (m * MagicPrime) low  64 bits
    //   mp_hi = (m * MagicPrime) high 64 bits
    //
    // All in a single asm block so the compiler can't interleave or reorder them.
    // Using "=&l" marks outputs early-clobber, ensuring no operand overlap.
    // ---------------------------------------------------------------------
    asm("{\n\t"
        "  mul.lo.u64 %0, %5, %6;   // t_lo = a * b (low 64 bits)\n\t"
        "  mul.hi.u64 %1, %5, %6;   // t_hi = a * b (high 64 bits)\n\t"
        "  mul.lo.u64 %2, %0, %7;   // m    = t_lo * MagicPrimeInv (mod 2^64)\n\t"
        "  mul.lo.u64 %3, %2, %8;   // mp_lo = m * MagicPrime (low 64 bits)\n\t"
        "  mul.hi.u64 %4, %2, %8;   // mp_hi = m * MagicPrime (high 64 bits)\n\t"
        "}\n\t"
        : "=&l"(t_lo), "=&l"(t_hi), "=&l"(m), "=&l"(mp_lo), "=&l"(mp_hi)
        : "l"(a),
          "l"(b),
          "l"(SharkNTT::MagicPrimeInv), // constant folded into immediate or const space
          "l"(SharkNTT::MagicPrime));   // same

    uint64_t u_hi, carry1;

    // ---------------------------------------------------------------------
    // Now compute 128-bit addition:
    //     u = t + mp
    //
    // We only need u_hi (upper 64 bits) for Montgomery reduction; u_lo is discarded.
    // Reuse mp_lo as the low-sum scratch to reduce register pressure.
    //
    // add.cc        sets the carry flag (CC) from the low-limb addition.
    // addc.cc       adds the high limbs *plus the carry*, again updating CC.
    // addc          writes out the final carry (0 or 1) to carry1.
    // ---------------------------------------------------------------------
    asm("add.cc.u64  %0, %3, %4;\n\t" // mp_lo = t_lo + mp_lo   (sets carry0)
        "addc.cc.u64 %1, %5, %6;\n\t" // u_hi = t_hi + mp_hi + carry0   (sets carry1)
        "addc.u64    %2, 0, 0;\n\t"   // carry1 = final carry out
        : "+l"(mp_lo), "=&l"(u_hi), "=&l"(carry1)
        : "l"(t_lo), "l"(mp_lo), "l"(t_hi), "l"(mp_hi));

    // Candidate Montgomery result before final correction
    uint64_t r = u_hi;

    // ---------------------------------------------------------------------
    // Final conditional subtraction:
    //
    //   if (carry1 || r >= p)
    //       r -= p;
    //
    // Implemented branchlessly using PTX predicates:
    //
    //   p_carry = (carry1 != 0)
    //   p_ge    = (r >= p)
    //   p_do    = p_carry || p_ge
    //
    // We perform:
    //   r = r - p
    //   if (!p_do) r = r + p   // undo subtraction when not needed
    //
    // This avoids warp divergence and yields a constant-latency path.
    // ---------------------------------------------------------------------
    {
        uint64_t p = SharkNTT::MagicPrime;

        asm volatile("{\n\t"
                     "  .reg .pred p_carry, p_ge, p_do;\n\t"
                     "  setp.ne.u64 p_carry, %1, 0;     // p_carry = (carry1 != 0)\n\t"
                     "  setp.ge.u64 p_ge, %0, %2;       // p_ge = (r >= p)\n\t"
                     "  or.pred p_do, p_carry, p_ge;    // p_do = p_carry || p_ge\n\t"
                     "  sub.u64 %0, %0, %2;             // r = r - p   (tentative)\n\t"
                     "  @!p_do add.u64 %0, %0, %2;      // if not doing reduction, restore r += p\n\t"
                     "}\n\t"
                     : "+l"(r)
                     : "l"(carry1), "l"(p));
    }

    return r; // Fully normalized Montgomery product mod p
}


template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
ToMontgomery(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             DebugGlobalCount<SharkFloatParams> *debugCombo,
             uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, R2);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
FromMontgomery(cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t x)
{
    return MontgomeryMul(grid, block, debugCombo, x, 1);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint64_t
MontgomeryPow(cooperative_groups::grid_group &grid,
              cooperative_groups::thread_block &block,
              DebugGlobalCount<SharkFloatParams> *debugCombo,
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
                               cooperative_groups::thread_block &block,
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
    const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;

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
          DebugGlobalCount<SharkFloatParams> *debugCombo,
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
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t *__restrict A,
                    uint64_t *__restrict B,
                    uint64_t *__restrict C,
                    uint32_t N,
                    uint32_t stages,
                    const uint64_t *__restrict s_stages,
                    const uint64_t *__restrict stage_twiddles)
{
    namespace cg = cooperative_groups;

    // Toggle this to select implementation:
    //   true  -> original MontgomeryPow recurrence on device
    //   false -> use precomputed flattened twiddle tables
    constexpr bool UseMontPow = false;
    constexpr uint32_t TS = 1u << TS_log;

    auto ctz_u32 = [](uint32_t x) -> uint32_t {
        uint32_t c = 0;
        while ((x & 1u) == 0u) {
            x >>= 1u;
            ++c;
        }
        return c;
    };

    const uint32_t S0 = (stages < TS_log) ? stages : TS_log;
    const uint32_t rem = (N & (TS - 1u));
    const uint32_t tail_len = (rem == 0u) ? TS : rem;
    const uint32_t tail_cap = (rem == 0u) ? TS_log : ctz_u32(tail_len);
    const uint32_t S1 = (S0 < tail_cap) ? S0 : tail_cap;
    if (S1 == 0)
        return 0;

    const uint32_t tiles = (N + TS - 1u) / TS;

    // Carve tile base from shared_data (A/B/C live in shared)
    auto *const s_dataA = reinterpret_cast<uint64_t *>(shared_data) + stages;
    auto *const s_dataB = s_dataA + TS;
    auto *const s_dataC = s_dataB + TS;

    // ----------------------------------------------------------------------
    // Shared twiddle cache for the *first few stages* (e.g. first 6).
    // We copy all needed twiddles for those stages once, up front.
    // Global flattened layout: stage s uses indices
    //   [tw_base .. tw_base + half-1], tw_base = 2^(s-1) - 1, half = 2^(s-1).
    // Total twiddles for stages 1..K = 2^K - 1.
    // ----------------------------------------------------------------------
    constexpr uint32_t MaxCachedStages = 6;
    [[maybe_unused]] constexpr uint32_t MaxCachedTwiddles =
        (1u << MaxCachedStages) - 1; // = 63 // static assert on shared memory usage

    // Reserve shared space for cached twiddles after C.
    auto *const s_twiddles_cached = s_dataC + TS; // needs +MaxCachedTwiddles uint64_t in smem

    // Only cache for precomputed-table mode, and only up to S1.
    const uint32_t cachedStages =
        (!UseMontPow && S1 > 0) ? ((S1 < MaxCachedStages) ? S1 : MaxCachedStages) : 0u;

    const uint32_t cachedTwiddles = (cachedStages > 0) ? ((1u << cachedStages) - 1u) : 0u;

    if constexpr (!UseMontPow) {
        if (cachedTwiddles > 0) {
            // Copy all twiddles for stages 1..cachedStages in one shot:
            // these are stage_twiddles[0 .. cachedTwiddles-1].
            cg::memcpy_async(
                block, s_twiddles_cached, stage_twiddles, cachedTwiddles * sizeof(uint64_t));
        }
    }

    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();

    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const bool is_last = (tile == tiles - 1);
        const uint32_t len = is_last ? tail_len : TS; // divisible by 2^S1

        // Load tiles into shared memory
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
            const uint64_t w_m = s_stages[s - 1];

            // base into flattened stage_twiddles for this stage:
            // stage s has 'half' twiddles; they live at indices [tw_base .. tw_base + half-1]
            // with tw_base = sum_{t=1}^{s-1} 2^(t-1) = 2^(s-1) - 1 = half - 1
            const uint32_t tw_base = half - 1u;

            const uint32_t total_pairs = (len >> 1);

            // For stages we cached, point into shared twiddle cache;
            // otherwise, fall back to global stage_twiddles.
            const bool useSharedTwiddles = (!UseMontPow && s <= cachedStages && cachedTwiddles > 0u);

            const uint64_t *SharkRestrict twiddleBaseForStage =
                useSharedTwiddles ? (s_twiddles_cached + tw_base) : (stage_twiddles + tw_base);

            for (uint32_t p = tid; p < total_pairs; p += step) {
                const uint32_t group = p / half;
                const uint32_t j = p - group * half; // p % half
                const uint32_t i0 = group * m + j;
                const uint32_t i1 = i0 + half;

                uint64_t wj;
                if constexpr (UseMontPow) {
                    wj = MontgomeryPow(grid, block, debugCombo, w_m, j);
                } else {
                    // from shared cache for s <= cachedStages, else from global
                    wj = twiddleBaseForStage[j];
                }

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

            // Original per-stage barrier
            block.sync();
        }

        // Store tiles back to global
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

// -----------------------------------------------------------------------------
// Per-warp, per-stage micro-tile processor.
// Handles:
//   - Mode = OneWay / TwoWay / ThreeWay
//   - UseMontPow = true/false (recurrence vs precomputed twiddles)
//   - microTileWidth = 4 (OneWay) or 2 (Two/ThreeWay)
// Updates jChunkIndex, tasksRemaining, blockIndex, blockDataBaseIndex,
// and (when UseMontPow) currentTwiddle.
// -----------------------------------------------------------------------------
template <class SharkFloatParams, Multiway Mode, bool UseMontPow, int microTileWidth>
static __device__ SharkForceInlineReleaseOnly void
ProcessTile(cooperative_groups::grid_group &grid,
            cooperative_groups::thread_block &block,
            DebugGlobalCount<SharkFloatParams> *debugCombo,
            uint64_t *SharkRestrict A,
            uint64_t *SharkRestrict B,
            uint64_t *SharkRestrict C,
            const uint64_t *SharkRestrict stageTwiddlesForStage,
            const uint32_t halfSpan,
            const uint32_t warpSize,
            const uint32_t numJChunks,
            const uint32_t laneIndex,
            const uint32_t butterflySpan,
            // mutable state for this warp-range:
            uint32_t &blockIndex,
            uint32_t &jChunkIndex,
            size_t &tasksRemaining,
            uint32_t &blockDataBaseIndex,
            // twiddle state (for UseMontPow == true):
            uint64_t &currentTwiddle,
            const uint64_t twiddleStrideWarp,
            const uint64_t laneTwiddleBase)
{
    // Tile cannot cross block boundary or our assigned range
    const uint32_t roomInBlock = numJChunks - jChunkIndex;
    const uint32_t span = static_cast<uint32_t>(min(tasksRemaining, static_cast<size_t>(roomInBlock)));

    const uint32_t tileWidth =
        (microTileWidth == 4) ? std::min<uint32_t>(4u, span) : std::min<uint32_t>(2u, span);

    // Helper: load A/B/C for a given jIndex
    auto loadABC = [&](uint32_t jIndex,
                       bool &inRange,
                       uint32_t &idxUpper,
                       uint32_t &idxLower,
                       uint64_t &aUpper,
                       uint64_t &aLower,
                       uint64_t &bUpper,
                       uint64_t &bLower,
                       uint64_t &cUpper,
                       uint64_t &cLower) {
        inRange = (jIndex < halfSpan);
        if (!inRange)
            return;

        idxUpper = blockDataBaseIndex + jIndex;
        idxLower = idxUpper + halfSpan;

        aUpper = A[idxUpper];
        aLower = A[idxLower];

        if constexpr (Mode != Multiway::OneWay) {
            bUpper = B[idxUpper];
            bLower = B[idxLower];
            if constexpr (Mode == Multiway::ThreeWay) {
                cUpper = C[idxUpper];
                cLower = C[idxLower];
            }
        }
    };

    // Helper: apply Cooley–Tukey butterfly:
    //   u' = u + t
    //   v' = u - t
    // with t = v * twiddle, where u = upper, v = lower.
    auto applyButterfly = [&](uint32_t idxUpper,
                              uint32_t idxLower,
                              uint64_t aUpper,
                              uint64_t aLower,
                              uint64_t bUpper,
                              uint64_t bLower,
                              uint64_t cUpper,
                              uint64_t cLower,
                              uint64_t twiddle) {
        if constexpr (Mode == Multiway::OneWay) {
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
        } else if constexpr (Mode == Multiway::TwoWay) {
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            const uint64_t tB = MontgomeryMul(grid, block, debugCombo, bLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
            B[idxUpper] = AddP(bUpper, tB);
            B[idxLower] = SubP(bUpper, tB); // subtract from *upper* (Bu), not Bl
        } else {                            // ThreeWay
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            const uint64_t tB = MontgomeryMul(grid, block, debugCombo, bLower, twiddle);
            const uint64_t tC = MontgomeryMul(grid, block, debugCombo, cLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
            B[idxUpper] = AddP(bUpper, tB);
            B[idxLower] = SubP(bUpper, tB); // same pattern
            C[idxUpper] = AddP(cUpper, tC);
            C[idxLower] = SubP(cUpper, tC); // same pattern
        }
    };

    auto loadTwiddlePrecomputed = [&](uint32_t jIndex, bool inRange) -> uint64_t {
        if (!inRange)
            return 0;
        return stageTwiddlesForStage[jIndex];
    };

    // ===== position 0 =====
    const uint32_t jIndex0 = laneIndex + jChunkIndex * warpSize;

    bool inRange0 = false;
    uint32_t indexUpper0 = 0, indexLower0 = 0;
    uint64_t aUpper0 = 0, aLower0 = 0;
    uint64_t bUpper0 = 0, bLower0 = 0;
    uint64_t cUpper0 = 0, cLower0 = 0;

    loadABC(jIndex0,
            inRange0,
            indexUpper0,
            indexLower0,
            aUpper0,
            aLower0,
            bUpper0,
            bLower0,
            cUpper0,
            cLower0);

    uint64_t twiddle0 = 0;
    if (inRange0) {
        if constexpr (UseMontPow) {
            twiddle0 = currentTwiddle;
        } else {
            twiddle0 = loadTwiddlePrecomputed(jIndex0, /*inRange=*/true);
        }
    }

    // ===== position 1 (if any) =====
    bool inRange1 = false;
    uint32_t indexUpper1 = 0, indexLower1 = 0;
    uint64_t aUpper1 = 0, aLower1 = 0;
    uint64_t bUpper1 = 0, bLower1 = 0;
    uint64_t cUpper1 = 0, cLower1 = 0;
    uint64_t twiddle1 = 0;

    if (tileWidth >= 2) {
        const uint32_t jIndex1 = jIndex0 + warpSize;

        loadABC(jIndex1,
                inRange1,
                indexUpper1,
                indexLower1,
                aUpper1,
                aLower1,
                bUpper1,
                bLower1,
                cUpper1,
                cLower1);

        if constexpr (UseMontPow) {
            twiddle1 = MontgomeryMul(grid, block, debugCombo, twiddle0, twiddleStrideWarp);
        } else if (inRange1) {
            twiddle1 = loadTwiddlePrecomputed(jIndex1, /*inRange=*/true);
        }
    }

    // ===== position 2/3 (OneWay only) =====
    bool inRange2 = false, inRange3 = false;
    uint32_t indexUpper2 = 0, indexLower2 = 0;
    uint64_t aUpper2 = 0, aLower2 = 0;
    uint64_t bUpper2 = 0, bLower2 = 0;
    uint64_t cUpper2 = 0, cLower2 = 0;
    uint32_t indexUpper3 = 0, indexLower3 = 0;
    uint64_t aUpper3 = 0, aLower3 = 0;
    uint64_t bUpper3 = 0, bLower3 = 0;
    uint64_t cUpper3 = 0, cLower3 = 0;
    uint64_t twiddle2 = 0, twiddle3 = 0;

    if constexpr (microTileWidth == 4) {
        if (tileWidth >= 3) {
            const uint32_t jIndex2 = jIndex0 + 2u * warpSize;
            loadABC(jIndex2,
                    inRange2,
                    indexUpper2,
                    indexLower2,
                    aUpper2,
                    aLower2,
                    bUpper2,
                    bLower2,
                    cUpper2,
                    cLower2);

            if constexpr (!UseMontPow) {
                if (inRange2) {
                    twiddle2 = loadTwiddlePrecomputed(jIndex2, /*inRange=*/true);
                }
            }
        }
        if (tileWidth >= 4) {
            const uint32_t jIndex3 = jIndex0 + 3u * warpSize;
            loadABC(jIndex3,
                    inRange3,
                    indexUpper3,
                    indexLower3,
                    aUpper3,
                    aLower3,
                    bUpper3,
                    bLower3,
                    cUpper3,
                    cLower3);

            if constexpr (!UseMontPow) {
                if (inRange3) {
                    twiddle3 = loadTwiddlePrecomputed(jIndex3, /*inRange=*/true);
                }
            }
        }

        if constexpr (UseMontPow) {
            if (tileWidth >= 3) {
                twiddle2 = MontgomeryMul(grid, block, debugCombo, twiddle1, twiddleStrideWarp);
                if (tileWidth >= 4) {
                    twiddle3 = MontgomeryMul(grid, block, debugCombo, twiddle2, twiddleStrideWarp);
                }
            }
        }
    }

    // ---- compute/store: position 0 ----
    if (inRange0) {
        applyButterfly(
            indexUpper0, indexLower0, aUpper0, aLower0, bUpper0, bLower0, cUpper0, cLower0, twiddle0);
    }

    // ---- compute/store: position 1 ----
    if (tileWidth >= 2 && inRange1) {
        applyButterfly(
            indexUpper1, indexLower1, aUpper1, aLower1, bUpper1, bLower1, cUpper1, cLower1, twiddle1);
    }

    // ---- compute/store: positions 2 & 3 (OneWay only) ----
    if constexpr (microTileWidth == 4) {
        if (tileWidth >= 3 && inRange2) {
            applyButterfly(indexUpper2,
                           indexLower2,
                           aUpper2,
                           aLower2,
                           bUpper2,
                           bLower2,
                           cUpper2,
                           cLower2,
                           twiddle2);
        }
        if (tileWidth >= 4 && inRange3) {
            applyButterfly(indexUpper3,
                           indexLower3,
                           aUpper3,
                           aLower3,
                           bUpper3,
                           bLower3,
                           cUpper3,
                           cLower3,
                           twiddle3);
        }
    }

    // ---- advance within block by 'tileWidth' and update state ----
    if constexpr (UseMontPow) {
        jChunkIndex += tileWidth;
        tasksRemaining -= tileWidth;

        if (jChunkIndex == numJChunks) {
            // wrap to next block
            jChunkIndex = 0;
            blockIndex += 1;
            blockDataBaseIndex += butterflySpan;
            currentTwiddle = laneTwiddleBase;
        } else {
            // seed twiddle for next tile start
            if (tileWidth == 1) {
                currentTwiddle = twiddle1;
            } else if (tileWidth == 2) {
                currentTwiddle = MontgomeryMul(grid, block, debugCombo, twiddle1, twiddleStrideWarp);
            } else if (tileWidth == 3) {
                currentTwiddle = twiddle3; // built above when microTileWidth == 4
            } else {                       // tileWidth == 4
                currentTwiddle = MontgomeryMul(grid, block, debugCombo, twiddle3, twiddleStrideWarp);
            }
        }
    } else {
        jChunkIndex += tileWidth;
        tasksRemaining -= tileWidth;

        if (jChunkIndex == numJChunks) {
            // wrap to next block
            jChunkIndex = 0;
            blockIndex += 1;
            blockDataBaseIndex += butterflySpan;
        }
        // No twiddle recurrence needed here; next call
        // will reload twiddles from stageTwiddlesForStage.
    }
}

// -----------------------------------------------------------------------------
// Unified 1-way / 3-way radix-2 NTT with warp-strided twiddles,
// early shared-memory microkernel, and Phase-2 static contiguous striping.
// Multiway::OneWay  : operates on A only (matches 1-way behavior)
// Multiway::TwoWay  : operates on A, B
// Multiway::ThreeWay: operates on A, B, C in lockstep
// -----------------------------------------------------------------------------
template <class SharkFloatParams, Multiway OneTwoThree, bool Inverse>
static __device__ SharkForceInlineReleaseOnly void
NTTRadix2_GridStride(uint64_t *shared_data,
                     cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t *SharkRestrict globalSync1,
                     uint64_t *SharkRestrict A,
                     uint64_t *SharkRestrict B,
                     uint64_t *SharkRestrict C,
                     const RootTables &rootTables)
{
    uint32_t transformSize = rootTables.N;
    uint32_t numStages = rootTables.stages;

    const uint64_t *SharkRestrict stageRootBase;     // per-stage base ω_m
    const uint64_t *SharkRestrict stageTwiddleTable; // flattened twiddle table

    if constexpr (!Inverse) {
        stageRootBase = rootTables.stage_omegas;
        stageTwiddleTable = rootTables.stage_twiddles_fwd;
    } else {
        stageRootBase = rootTables.stage_omegas_inv;
        stageTwiddleTable = rootTables.stage_twiddles_inv;
    }

    // Toggle this to select implementation:
    //   true  -> original MontgomeryPow recurrence on device
    //   false -> use precomputed flattened twiddle tables
    constexpr bool UseMontPow = false;

    constexpr auto TileSizeLog2 = 9u;
    constexpr uint32_t warpSize = 32u;

    const size_t gridSize = grid.size();
    const auto globalThreadIndex = block.thread_index().x + block.group_index().x * blockDim.x;

    const uint32_t laneIndex = static_cast<uint32_t>(globalThreadIndex & (warpSize - 1u));
    const uint32_t warpIndex = static_cast<uint32_t>(globalThreadIndex / warpSize);
    const uint32_t numWarpsGrid = static_cast<uint32_t>(gridSize / warpSize);

    uint32_t firstLargeStage = 0;

    // Cache stageRootBase into shared memory
    auto *sharedStageRoots = shared_data;

    cg::memcpy_async(block, sharedStageRoots, stageRootBase, numStages * sizeof(uint64_t));
    cg::wait(block);

    // Phase 1: small-radix microkernel
    if constexpr (OneTwoThree == Multiway::OneWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::OneWay, TileSizeLog2>(shared_data,
                                                                                  grid,
                                                                                  block,
                                                                                  debugCombo,
                                                                                  A,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  transformSize,
                                                                                  numStages,
                                                                                  sharedStageRoots,
                                                                                  stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::TwoWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::TwoWay, TileSizeLog2>(shared_data,
                                                                                  grid,
                                                                                  block,
                                                                                  debugCombo,
                                                                                  A,
                                                                                  B,
                                                                                  nullptr,
                                                                                  transformSize,
                                                                                  numStages,
                                                                                  sharedStageRoots,
                                                                                  stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::ThreeWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::ThreeWay, TileSizeLog2>(shared_data,
                                                                                    grid,
                                                                                    block,
                                                                                    debugCombo,
                                                                                    A,
                                                                                    B,
                                                                                    C,
                                                                                    transformSize,
                                                                                    numStages,
                                                                                    sharedStageRoots,
                                                                                    stageTwiddleTable);
    }

    // =========================
    // Phase 2: static contiguous striping by warp (no atomics)
    //          + scalarized micro-tiling (register-only, U=4/2)
    // =========================
    for (uint32_t stageIndex = firstLargeStage + 1; stageIndex <= numStages; ++stageIndex) {
        const uint32_t butterflySpan = 1u << stageIndex; // m
        const uint32_t halfSpan = butterflySpan >> 1;    // m/2

        [[maybe_unused]] const uint64_t stageRoot = sharedStageRoots[stageIndex - 1];

        // base into flattened stageTwiddleTable for this stage:
        // stage s has 'halfSpan' twiddles; they live at indices
        // [twiddleStageOffset .. twiddleStageOffset + halfSpan-1]
        // with twiddleStageOffset = 2^(s-1) - 1 = halfSpan - 1
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint64_t *SharkRestrict stageTwiddlesForStage = stageTwiddleTable + twiddleStageOffset;

        const uint32_t numJChunks = (halfSpan + (warpSize - 1u)) / warpSize; // ceil(halfSpan/warpSize)

        const uint32_t numBlocksPerStage = transformSize / butterflySpan;
        const size_t totalTasks = static_cast<size_t>(numBlocksPerStage) *
                                  static_cast<size_t>(numJChunks); // == N/64, invariant in s

        uint64_t twiddleStrideWarp = 0; // w_m^W
        uint64_t laneTwiddleBase = 0;   // w_m^lane
        uint64_t currentTwiddle = 0;

        if constexpr (UseMontPow) {
            twiddleStrideWarp = MontgomeryPow(grid, block, debugCombo, stageRoot, warpSize);
            laneTwiddleBase = MontgomeryPow(grid, block, debugCombo, stageRoot, laneIndex);
        }

        // -------- Static contiguous partition: each warp gets one equal-sized range --------
        const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid; // ceil
        size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
        size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

        if (warpTaskBegin < warpTaskEnd) {
            // Decode the first ticket in our contiguous range
            uint32_t blockIndex = static_cast<uint32_t>(warpTaskBegin / numJChunks);
            uint32_t jChunkIndex =
                static_cast<uint32_t>(warpTaskBegin - static_cast<size_t>(blockIndex) * numJChunks);

            if constexpr (UseMontPow) {
                currentTwiddle =
                    (jChunkIndex == 0)
                        ? laneTwiddleBase
                        : MontgomeryMul(
                              grid,
                              block,
                              debugCombo,
                              laneTwiddleBase,
                              MontgomeryPow(grid, block, debugCombo, twiddleStrideWarp, jChunkIndex));
            }

            uint32_t blockDataBaseIndex = blockIndex * butterflySpan;
            size_t tasksRemaining = warpTaskEnd - warpTaskBegin;

            constexpr int microTileWidth = (OneTwoThree == Multiway::OneWay ? 4 : 2);

            // ------------- process our contiguous slice -------------
            while (tasksRemaining) {
                ProcessTile<SharkFloatParams, OneTwoThree, UseMontPow, microTileWidth>(
                    grid,
                    block,
                    debugCombo,
                    A,
                    B,
                    C,
                    stageTwiddlesForStage,
                    halfSpan,
                    warpSize,
                    numJChunks,
                    laneIndex,
                    butterflySpan,
                    blockIndex,
                    jChunkIndex,
                    tasksRemaining,
                    blockDataBaseIndex,
                    currentTwiddle,
                    twiddleStrideWarp,
                    laneTwiddleBase);
            }
        }

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
                           cooperative_groups::thread_block &block,
                           // inputs (normal domain)
                           const uint64_t *SharkRestrict AXX_norm,
                           const uint64_t *SharkRestrict AYY_norm,
                           const uint64_t *SharkRestrict AXY_norm,
                           // outputs (len = 2 * Ddigits; pairs of 64-bit lo/hi)
                           uint64_t *SharkRestrict Final128_XX,
                           uint64_t *SharkRestrict Final128_YY,
                           uint64_t *SharkRestrict Final128_XY,
                           uint32_t Ddigits)
{
    using namespace SharkNTT;

    const size_t gsize = grid.size();
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

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
    const int Imax =
        min(SharkFloatParams::NTTPlan.N, 2 * SharkFloatParams::NTTPlan.L - 1); // same bound as original

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
            const uint64_t i_hi_raw =
                ceil_div_u64(32ull * (k + 1ull), (uint32_t)SharkFloatParams::NTTPlan.b);
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
PackTwistFwdNTT_Fused_AB_ToSixOutputs(
    uint64_t *shared_data,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    DebugState<SharkFloatParams> *debugStates,
    DebugGlobalCount<SharkFloatParams> *debugGlobalState,
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
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

    const uint64_t zero_m = ToMontgomeryConstexpr(0ull);

    // -------------------- Phase A: pack+twist with tail zero (grid-stride) --------------------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        if (i < L) {
            const uint64_t coeff = ReadBitsSimple(
                inA, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmod = coeff % MagicPrime; // match original
            const uint64_t xm = ToMontgomery(grid, block, debugGlobalState, cmod);
            const uint64_t psik = roots.psi_pows[i]; // Montgomery domain
            tempDigitsXX1[i] = MontgomeryMul(grid, block, debugGlobalState, xm, psik);
        } else {
            tempDigitsXX1[i] = zero_m;
        }

        if (i < L) {
            const uint64_t coeffB = ReadBitsSimple(
                inB, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmodB = coeffB % MagicPrime;
            const uint64_t xmB = ToMontgomery(grid, block, debugGlobalState, cmodB);
            const uint64_t psiB = roots.psi_pows[i]; // Montgomery domain
            tempDigitsYY1[i] = MontgomeryMul(grid, block, debugGlobalState, xmB, psiB);
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
    BitReverseInplace64_GridStride<Multiway::TwoWay>(grid,
                                                     block,
                                                     tempDigitsXX1,
                                                     tempDigitsYY1,
                                                     nullptr,
                                                     N,
                                                     (uint32_t)SharkFloatParams::NTTPlan.stages);

    grid.sync();

    NTTRadix2_GridStride<SharkFloatParams, Multiway::TwoWay, false>(shared_data,
                                                                    grid,
                                                                    block,
                                                                    debugGlobalState,
                                                                    carryPropagationSync,
                                                                    tempDigitsXX1,
                                                                    tempDigitsYY1,
                                                                    nullptr,
                                                                    roots);

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
                                     DebugGlobalCount<SharkFloatParams> *debugGlobalState,
                                     const SharkNTT::RootTables &roots,
                                     uint64_t *SharkRestrict tempDigitsXX1,
                                     uint64_t *SharkRestrict tempDigitsYY1,
                                     uint64_t *SharkRestrict tempDigitsXY1)
{
    using namespace SharkNTT;

    const size_t N = static_cast<size_t>(SharkFloatParams::NTTPlan.N);
    const size_t gsize = grid.size();
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

    const uint64_t Ninvm = roots.Ninvm_mont; // Montgomery-domain 1/N

    for (size_t i = grank; i < N; i += gsize) {
        const uint64_t psi_inv_i = roots.psi_inv_pows[i]; // Montgomery-domain psi^{-i}

        // XX
        {
            uint64_t v = MontgomeryMul(grid, block, debugGlobalState, tempDigitsXX1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugGlobalState, v, Ninvm);
            tempDigitsXX1[i] = FromMontgomery(grid, block, debugGlobalState, v);
        }

        // YY
        {
            uint64_t v = MontgomeryMul(grid, block, debugGlobalState, tempDigitsYY1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugGlobalState, v, Ninvm);
            tempDigitsYY1[i] = FromMontgomery(grid, block, debugGlobalState, v);
        }

        // XY
        {
            uint64_t v = MontgomeryMul(grid, block, debugGlobalState, tempDigitsXY1[i], psi_inv_i);
            v = MontgomeryMul(grid, block, debugGlobalState, v, Ninvm);
            tempDigitsXY1[i] = FromMontgomery(grid, block, debugGlobalState, v);
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
    debugStates[curPurpose].Erase(grid, block, Purpose, RecursionDepth, CallIndex);
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

    debugStates[CurPurpose].Reset(
        UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

// Look for CalculateNTTFrameSize
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArraysForMultiply.
static_assert(AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets()                                                                     \
    const int threadIdxGlobal = block.group_index().x * block.dim_threads().x + block.thread_index().x; \
    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;                                            \
    constexpr int TestMultiplier = 1;                                                                   \
    constexpr auto DebugGlobals_offset = AdditionalGlobalSyncSpace;                                     \
    constexpr auto DebugChecksum_offset = DebugGlobals_offset + AdditionalGlobalDebugPerThread;         \
    constexpr auto GlobalsDoneOffset = DebugChecksum_offset + AdditionalGlobalChecksumSpace;            \
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
                     DebugGlobalCount<SharkFloatParams> *debugGlobalState,
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
                     uint64_t *CarryPropagationSync2,
                     uint32_t Ddigits)
{
    PackTwistFwdNTT_Fused_AB_ToSixOutputs<SharkFloatParams>(shared_data,
                                                            grid,
                                                            block,
                                                            debugStates,
                                                            debugGlobalState,
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
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

    for (size_t i = grank; i < N; i += gsize) {
        const uint64_t aXX = tempDigitsXX1[i];
        const uint64_t bXX = tempDigitsXX2[i];
        tempDigitsXX1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aXX, bXX);

        const uint64_t aYY = tempDigitsYY1[i];
        const uint64_t bYY = tempDigitsYY2[i];
        tempDigitsYY1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aYY, bYY);

        const uint64_t aXY = tempDigitsXY1[i];
        const uint64_t bXY = tempDigitsXY2[i];
        tempDigitsXY1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aXY, bXY);
    }

    grid.sync();

    // 5) Inverse NTT (in place on Z0_OutDigits)
    SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::ThreeWay>(
        grid,
        block,
        tempDigitsXX1,
        tempDigitsYY1,
        tempDigitsXY1,
        (uint32_t)SharkFloatParams::NTTPlan.N,
        (uint32_t)SharkFloatParams::NTTPlan.stages);

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

    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::ThreeWay, true>(
        shared_data,
        grid,
        block,
        debugGlobalState,
        CarryPropagationSync,
        tempDigitsXX1,
        tempDigitsYY1,
        tempDigitsXY1,
        roots);

    // --- After inverse NTTs (XX1 / YY1 / XY1 are in Montgomery domain) ---
    grid.sync(); // make sure prior writes (inv-NTT) are visible

    UntwistScaleFromMont_3Way_GridStride<SharkFloatParams>(grid,
                                                           block,
                                                           debugGlobalState,
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
                                                           block,
                                                           tempDigitsXX1,
                                                           tempDigitsYY1,
                                                           tempDigitsXY1,
                                                           Final128_XX,
                                                           Final128_YY,
                                                           Final128_XY,
                                                           Ddigits);

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

    // scratch result digits (2*SharkFloatParams::GlobalNumUint32 per channel, uint64_t each; low 32 bits
    // used)
    uint64_t *resultXX = tempDigitsXX1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */
    uint64_t *resultYY = tempDigitsYY1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */
    uint64_t *resultXY = tempDigitsXY1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */

    // ---- Single fused normalize for XX, YY, XY ----
    // #define FORCE_ORIGINAL_NORMALIZE

#ifdef FORCE_ORIGINAL_NORMALIZE
    SharkNTT::Normalize_GridStride_3WayV1<SharkFloatParams>(
        grid,
        block,
        debugGlobalState,
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
#else
    SharkNTT::Normalize_GridStride_3WayV2<SharkFloatParams>(grid,
                                                            block,
                                                            debugGlobalState,
                                                            debugStates,
                                                            *outXX,
                                                            *outYY,
                                                            *outXY,
                                                            inA,
                                                            inB,
                                                            Final128_XX,
                                                            Final128_YY,
                                                            Final128_XY,
                                                            Ddigits,
                                                            addFactorOfTwoXX,
                                                            addFactorOfTwoYY,
                                                            addFactorOfTwoXY,
                                                            CarryPropagationBuffer2,
                                                            CarryPropagationBuffer,
                                                            CarryPropagationSync,
                                                            CarryPropagationSync2,
                                                            resultXX,
                                                            resultYY,
                                                            resultXY);
#endif
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

    DefineTempProductsOffsets();

    // TODO: indexes
    auto *SharkRestrict debugGlobalState =
        reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempProducts[DebugGlobals_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams> *>(&tempProducts[DebugChecksum_offset]);

#ifdef ENABLE_MULTIPLY_NTT_KERNEL
    if constexpr (HpShark::DebugGlobalState) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugGlobalState[CurBlock * block.dim_threads().x + CurThread].DebugMultiplyErase();
    }
#endif

    if constexpr (HpShark::DebugChecksums) {
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfHigh>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::AHalfLow>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfHigh>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::BHalfLow>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::XDiff>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::YDiff>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XX>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0XY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0YY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XX>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1XY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z1YY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XX>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2YY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XX>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3XY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z3YY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XX>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4XY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z4YY>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6>(debugStates, grid, block);
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
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2>(debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd3>(debugStates, grid, block);
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
    const auto Ddigits =
        (((2 * SharkFloatParams::NTTPlan.L - 2) * SharkFloatParams::NTTPlan.b + 64) + 31) / 32 + 2;

    const auto LameHackBufferSizeWhatShouldItBe = std::max(Ddigits, SharkFloatParams::NTTPlan.N);

    // ---- Single allocation for entire core path ----
    uint64_t *buffer = &tempProducts[GlobalsDoneOffset];

    // TODO: Assert or something somewhere
    // const size_t buf_count = (size_t)2 * (size_t)SharkFloatParams::NTTPlan.N     // Z0_OutDigits +
    // Z2_OutDigits
    //                         + (size_t)2 * (size_t)Ddigits; // Final128 (lo,hi per 32-bit slot)

    // Slice buffer into spans
    size_t off = 0;
    uint64_t *tempDigitsXX1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsXX2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;

    uint64_t *tempDigitsYY1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsYY2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;

    uint64_t *tempDigitsXY1 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsXY2 = buffer + off; // SharkFloatParams::NTTPlan.N
    off += LameHackBufferSizeWhatShouldItBe;

    uint64_t *Final128_XX = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *Final128_YY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *Final128_XY = buffer + off; // (size_t)2 * Ddigits
    off += (size_t)2 * Ddigits;

    uint64_t *CarryPropagationBuffer = buffer + off;
    off += 6 * Ddigits;

    uint64_t *CarryPropagationBuffer2 = buffer + off;
    off += 6 * Ddigits;

    uint64_t *CarryPropagationSync = &tempProducts[0];
    uint64_t *CarryPropagationSync2 = &tempProducts[16];

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
                                           debugGlobalState,
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
                                           CarryPropagationSync2,
                                           Ddigits);

    grid.sync();
}

template <class SharkFloatParams>
void
PrintMaxActiveBlocks(const HpShark::LaunchParams &launchParams, void *kernelFn, int sharedAmountBytes)
{

    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;

    {
        // Check the maximum number of active blocks per multiprocessor
        // with the given shared memory size
        // This is useful to determine if we can fit more blocks
        // in the shared memory

        const auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, kernelFn, launchParams.ThreadsPerBlock, sharedAmountBytes);

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
            &availableSharedMemory, kernelFn, numBlocks, launchParams.ThreadsPerBlock);

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
    if (maxConcurrentBlocks < launchParams.NumBlocks) {
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
