#include "MultiplyNTT.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "DebugChecksum.h"
#include "HpSharkFloat.h"

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

#include "MontgomeryCoreConstexpr.h"
#include "NTTConstexprGenerator.h"

namespace cg = cooperative_groups;

[[maybe_unused]] static constexpr auto
CalcAlign16Bytes64BitIndex(uint64_t Sixty4BitIndex)
{
    return Sixty4BitIndex % 2 == 0 ? 0 : 1;
}

[[maybe_unused]] static constexpr auto
CalcAlign16Bytes32BitIndex(uint64_t Thirty2BitIndex)
{
    return 4 - (Thirty2BitIndex % 4);
}

namespace SharkNTT {

template <class SharkFloatParams,
          DebugStatePurpose purposeXX = DebugStatePurpose::Final128XX,
          DebugStatePurpose purposeYY = DebugStatePurpose::Final128YY,
          DebugStatePurpose purposeXY = DebugStatePurpose::Final128XY>
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
        StoreCurrentDebugState<SharkFloatParams, purposeXX, uint32_t>(
            debugStates, grid, block, outXX.Digits, SharkFloatParams::GlobalNumUint32);
        StoreCurrentDebugState<SharkFloatParams, purposeYY, uint32_t>(
            debugStates, grid, block, outYY.Digits, SharkFloatParams::GlobalNumUint32);
        StoreCurrentDebugState<SharkFloatParams, purposeXY, uint32_t>(
            debugStates, grid, block, outXY.Digits, SharkFloatParams::GlobalNumUint32);
        grid.sync();
    } else {
        grid.sync();
    }
}

// ---- N-channel FinalizeNormalize ----
// Takes pre-computed exponent sums and arrays of pointers for each channel.
template <class SharkFloatParams, int NumChannels>
static __device__ SharkForceInlineReleaseOnly void
FinalizeNormalizeNWay(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugState<SharkFloatParams> *debugStates,
                      HpSharkFloat<SharkFloatParams> **outs,
                      const int32_t *baseExponents,
                      const uint32_t Ddigits,
                      const int32_t *addTwos,
                      uint64_t *SharkRestrict workspace,
                      uint64_t *SharkRestrict *results,
                      const DebugStatePurpose *purposes)
{
    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    // Scan for highest-nonzero per channel; compute shifts/exponents (single thread)
    if (tid == 0) {
        for (int ch = 0; ch < NumChannels; ++ch) {
            int h = -1;
            for (int i = Ddigits - 1; i >= 0; --i) {
                if (static_cast<uint32_t>(results[ch][i]) != 0u) {
                    h = i;
                    break;
                }
            }

            int shift = 0;
            int32_t exp = baseExponents[ch];
            if (h >= 0) {
                const int significant = h + 1;
                shift = significant - SharkFloatParams::GlobalNumUint32;
                if (shift < 0)
                    shift = 0;
                exp = baseExponents[ch] + 32 * shift + addTwos[ch];
            }

            outs[ch]->Exponent = exp;
            workspace[ch] = static_cast<uint64_t>(shift);
            workspace[NumChannels + ch] = static_cast<uint64_t>(h < 0);
        }
    }
    grid.sync();

    int shifts[NumChannels];
    bool zeros[NumChannels];
#pragma unroll
    for (int ch = 0; ch < NumChannels; ++ch) {
        shifts[ch] = static_cast<int>(workspace[ch]);
        zeros[ch] = (workspace[NumChannels + ch] != 0);
    }

    // Grid-stride write the GlobalNumUint32-digit windows
    for (int i = tid; i < SharkFloatParams::GlobalNumUint32; i += T_all) {
#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            outs[ch]->Digits[i] = zeros[ch] ? 0u
                                            : ((shifts[ch] + i < static_cast<int>(Ddigits))
                                                   ? static_cast<uint32_t>(results[ch][shifts[ch] + i])
                                                   : 0u);
        }
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        for (int ch = 0; ch < NumChannels; ++ch) {
            StoreCurrentDebugState<SharkFloatParams, uint32_t>(debugStates,
                                                               grid,
                                                               block,
                                                               purposes[ch],
                                                               outs[ch]->Digits,
                                                               SharkFloatParams::GlobalNumUint32);
        }
        grid.sync();
    } else {
        grid.sync();
    }
}

#include "MultiplyNTT_NormalizePPv3.h"
#include "MultiplyNTT_NormalizeV1.h"
#include "MultiplyNTT_NormalizeWarpTiledV2.h"

// ---- N-channel Normalize_GridStride ----
// Fused normalize for NumChannels products sharing all grid.sync() barriers.
template <class SharkFloatParams, int NumChannels>
static __device__ inline void
Normalize_GridStride_NWay(cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          DebugGlobalCount<SharkFloatParams> *debugGlobalState,
                          DebugState<SharkFloatParams> *debugStates,
                          HpSharkFloat<SharkFloatParams> **outs,
                          const int32_t *baseExponents,
                          uint64_t **final128s,
                          const uint32_t Ddigits,
                          const int32_t *addTwos,
                          uint64_t *SharkRestrict CarryPropagationBuffer2,
                          uint64_t *SharkRestrict CarryPropagationBuffer,
                          uint64_t *globalSync1,
                          uint64_t *globalSync2,
                          uint64_t **results,
                          const DebugStatePurpose *purposes,
                          uint64_t *SharkRestrict shared_data)
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    const int warpSz = block.dim_threads().x;
#else
    constexpr int warpSz = 32;
#endif
    (void)warpSz;

    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int totalThreads = T_all;

    auto *cur = CarryPropagationBuffer;
    auto *next = CarryPropagationBuffer2;

    // Only cur[0..NumChannels-1] and next[0..NumChannels-1] need zeroing:
    // Phase 2 reads cur[0..N-1] for index=0 (no predecessor carry),
    // Phase 3 reads next[0..N-1] for index=0. Everything else is overwritten
    // by the previous phase before being read.
    // tid=0 processes index=0 in Phase 2, so same-thread write→read, no sync needed.
    if (tid == 0) {
#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            cur[ch] = 0;
            next[ch] = 0;
        }
        *globalSync1 = 0;
        *globalSync2 = 0;
    }

    // Phase 1: Extract digits and carries from final128 buffers
    for (int index = tid; index < static_cast<int>(Ddigits); index += totalThreads) {
        const auto indexT2 = 2 * index;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint64_t lo = final128s[ch][indexT2];
            const uint64_t hi = final128s[ch][indexT2 + 1];
            const uint32_t dig = static_cast<uint32_t>(lo & 0xffffffffu);
            const uint64_t carry = (lo >> 32) | (hi << 32);

            results[ch][index] = dig;
            cur[index * NumChannels + NumChannels + ch] = carry;
        }
    }

    grid.sync();

    // Phase 2: First carry propagation round
    for (int index = tid; index < static_cast<int>(Ddigits); index += totalThreads) {
#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint64_t carryIn = cur[index * NumChannels + ch];
            const uint64_t inDig = results[ch][index];
            const uint64_t fullDig = inDig + carryIn;
            const uint64_t c0 = (fullDig < inDig) ? 1ull : 0ull;

            results[ch][index] = fullDig & 0xffffffffu;
            next[index * NumChannels + NumChannels + ch] = (fullDig >> 32) | (c0 << 32);
        }
    }

    grid.sync();

    // Phase 3: Second carry propagation round (produces 0-or-1 carries)
    for (int index = tid; index < static_cast<int>(Ddigits); index += totalThreads) {
        bool anyCarry = false;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint64_t carryIn = next[index * NumChannels + ch];
            next[index * NumChannels + ch] = 0;

            const uint64_t inDig = results[ch][index];
            const uint64_t fullDig = inDig + carryIn;

            results[ch][index] = fullDig & 0xffffffffu;
            const uint64_t carryOut = fullDig >> 32;
            cur[index * NumChannels + NumChannels + ch] = carryOut;

            if (carryOut != 0) {
                anyCarry = true;
            }
        }

        if (anyCarry) {
            *globalSync2 = 1;
        }
    }

    grid.sync();

    const auto globalResult = *globalSync2;

    if (globalResult != 0) {
        // Reuse final128 buffers as PP scratch (they are consumed above).
        auto *digitXfer = reinterpret_cast<DigitTransfer<NumChannels> *>(final128s[0]);
        auto *descBuf = reinterpret_cast<uint32_t *>(final128s[1]);
        auto *carryInMask = reinterpret_cast<uint32_t *>(final128s[2]);

        constexpr bool DLB = true;

        if constexpr (DLB) {
            ParallelPrefixNormalize_DLB<SharkFloatParams, NumChannels>(
                shared_data, grid, block, cur, Ddigits, results, digitXfer, descBuf, carryInMask);
        } else {
            ParallelPrefixNormalize<SharkFloatParams, NumChannels>(
                grid, block, cur, Ddigits, results, digitXfer, descBuf, carryInMask);
        }
    }

    FinalizeNormalizeNWay<SharkFloatParams, NumChannels>(grid,
                                                         block,
                                                         debugStates,
                                                         outs,
                                                         baseExponents,
                                                         Ddigits,
                                                         addTwos,
                                                         CarryPropagationBuffer2,
                                                         results,
                                                         purposes);
}

template <class SharkFloatParams,
          DebugStatePurpose purposeXX = DebugStatePurpose::Final128XX,
          DebugStatePurpose purposeYY = DebugStatePurpose::Final128YY,
          DebugStatePurpose purposeXY = DebugStatePurpose::Final128XY>
static __device__ inline void
Normalize_GridStride_3WayV2(uint64_t *shared_data,
                            cooperative_groups::grid_group &grid,
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
    const int warpSz = block.dim_threads().x;
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

    // Only cur[0..2] and next[0..2] need zeroing (carry-in for digit 0).
    // tid=0 processes index=0 in Phase 2 → same-thread write→read, no sync needed.
    if (tid == 0) {
        cur[0] = 0;
        cur[1] = 0;
        cur[2] = 0;
        next[0] = 0;
        next[1] = 0;
        next[2] = 0;
    }

    // We run the tile chain in a single warp for now (warp 0) so that tile
    // carries are propagated correctly and in order. Other warps do nothing.
    uint64_t prevGlobalSync1 = std::numeric_limits<uint64_t>::max();
    if (tid == 0) {
        *globalSync1 = 0;
        *globalSync2 = 0;
    }

    auto swap2 = [](uint64_t *&a, uint64_t *&b) {
        auto *t = a;
        a = b;
        b = t;
    };

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
                    block, fullMask, Ddigits, lane, tile, iteration, loXX, loYY, loXY, carry_lo);

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

        ParallelPrefixNormalize3WayV3<SharkFloatParams>(shared_data,
                                                        grid,
                                                        block,
                                                        cur,
                                                        Ddigits,
                                                        resultXX,
                                                        resultYY,
                                                        resultXY,
                                                        digitXfer,
                                                        scanTemp,
                                                        carryInMask);
    }
#endif

    FinalizeNormalize<SharkFloatParams, purposeXX, purposeYY, purposeXY>(grid,
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

enum class Multiway { OneWay, TwoWay, ThreeWay, FourWay, SevenWay };

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
                               uint64_t *SharkRestrict D,
                               uint32_t N,
                               uint32_t stages,
                               uint64_t *SharkRestrict E = nullptr,
                               uint64_t *SharkRestrict F = nullptr,
                               uint64_t *SharkRestrict G = nullptr)
{
    // Compile-time selection of which arrays to process
    constexpr bool DoA = true;
    constexpr bool DoB = (OneTwoThree != Multiway::OneWay);
    constexpr bool DoC = (OneTwoThree == Multiway::ThreeWay || OneTwoThree == Multiway::FourWay ||
                          OneTwoThree == Multiway::SevenWay);
    constexpr bool DoD = (OneTwoThree == Multiway::FourWay || OneTwoThree == Multiway::SevenWay);
    constexpr bool DoE = (OneTwoThree == Multiway::SevenWay);
    constexpr bool DoF = (OneTwoThree == Multiway::SevenWay);
    constexpr bool DoG = (OneTwoThree == Multiway::SevenWay);

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
        if constexpr (DoD)
            swap_one(D, i, j);
        if constexpr (DoE)
            swap_one(E, i, j);
        if constexpr (DoF)
            swap_one(F, i, j);
        if constexpr (DoG)
            swap_one(G, i, j);
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

// ---------------------------------------------------------------------------
// Per-tile helpers for the shared-memory phase. Splitting the load from the compute/store path lets
// SevenWay keep the D/E copy in flight while A/B/C are processed.
// ---------------------------------------------------------------------------
template <class SharkFloatParams, Multiway Mode>
static __device__ inline void
LoadOneTilePhase1SM(cooperative_groups::thread_block &block,
                    uint64_t *sharedDataA,
                    uint64_t *sharedDataB,
                    uint64_t *sharedDataC,
                    uint64_t *sharedDataD,
                    const uint64_t *__restrict A,
                    const uint64_t *__restrict B,
                    const uint64_t *__restrict C,
                    const uint64_t *__restrict D,
                    uint32_t tile,
                    uint32_t TS,
                    uint32_t len)
{
    namespace cg = cooperative_groups;

    if constexpr (Mode == Multiway::OneWay || Mode == Multiway::TwoWay || Mode == Multiway::ThreeWay ||
                  Mode == Multiway::FourWay) {
        cg::memcpy_async(
            block, sharedDataA, &A[tile * TS], cuda::aligned_size_t<16>(len * sizeof(uint64_t)));
    }

    if constexpr (Mode == Multiway::TwoWay || Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
        cg::memcpy_async(
            block, sharedDataB, &B[tile * TS], cuda::aligned_size_t<16>(len * sizeof(uint64_t)));
    }

    if constexpr (Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
        cg::memcpy_async(
            block, sharedDataC, &C[tile * TS], cuda::aligned_size_t<16>(len * sizeof(uint64_t)));
    }

    if constexpr (Mode == Multiway::FourWay) {
        cg::memcpy_async(
            block, sharedDataD, &D[tile * TS], cuda::aligned_size_t<16>(len * sizeof(uint64_t)));
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ inline void
ProcessLoadedTilePhase1SM(cooperative_groups::thread_block &block,
                          cooperative_groups::grid_group &grid,
                          DebugGlobalCount<SharkFloatParams> *debugCombo,
                          uint64_t *sharedDataA,
                          uint64_t *sharedDataB,
                          uint64_t *sharedDataC,
                          uint64_t *sharedDataD,
                          uint64_t *__restrict A,
                          uint64_t *__restrict B,
                          uint64_t *__restrict C,
                          uint64_t *__restrict D,
                          const uint64_t *sharedTwiddles,
                          const uint64_t *__restrict stageTwiddles,
                          uint32_t tile,
                          uint32_t TS,
                          uint32_t len,
                          uint32_t S1,
                          uint32_t cachedStages)
{
    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();

    // Stages s=1..S1 — single set of loops; reuse the same syncs for all arrays
    for (uint32_t s = 1; s <= S1; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;

        const uint32_t twiddleBase = half - 1u;

        const uint32_t totalPairs = (len >> 1);

        const bool useSharedTwiddles = (s <= cachedStages && cachedStages > 0u);

        const uint64_t *SharkRestrict twiddleBaseForStage =
            useSharedTwiddles ? (sharedTwiddles + twiddleBase) : (stageTwiddles + twiddleBase);

        for (uint32_t p = tid; p < totalPairs; p += step) {
            const uint32_t group = p / half;
            const uint32_t j = p - group * half; // p % half
            const uint32_t i0 = group * m + j;
            const uint32_t i1 = i0 + half;

            const uint64_t wj = twiddleBaseForStage[j];

            // ---- A (shared) ----
            if constexpr (Mode == Multiway::OneWay) {
                const uint64_t U1 = sharedDataA[i0];
                const uint64_t V1 = sharedDataA[i1];

                const uint64_t t = MontgomeryMul(grid, block, debugCombo, V1, wj);

                sharedDataA[i0] = AddP(U1, t);
                sharedDataA[i1] = SubP(U1, t);
            }

            if constexpr (Mode == Multiway::TwoWay) {
                const uint64_t U1 = sharedDataA[i0];
                const uint64_t V1 = sharedDataA[i1];

                const uint64_t U2 = sharedDataB[i0];
                const uint64_t V2 = sharedDataB[i1];

                const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, V1, wj);
                const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, V2, wj);

                sharedDataA[i0] = AddP(U1, t1);
                sharedDataA[i1] = SubP(U1, t1);

                sharedDataB[i0] = AddP(U2, t2);
                sharedDataB[i1] = SubP(U2, t2);
            }

            if constexpr (Mode == Multiway::ThreeWay) {
                const uint64_t U1 = sharedDataA[i0];
                const uint64_t V1 = sharedDataA[i1];

                const uint64_t U2 = sharedDataB[i0];
                const uint64_t V2 = sharedDataB[i1];

                const uint64_t U3 = sharedDataC[i0];
                const uint64_t V3 = sharedDataC[i1];

                const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, V1, wj);
                const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, V2, wj);
                const uint64_t t3 = MontgomeryMul(grid, block, debugCombo, V3, wj);

                sharedDataA[i0] = AddP(U1, t1);
                sharedDataA[i1] = SubP(U1, t1);

                sharedDataB[i0] = AddP(U2, t2);
                sharedDataB[i1] = SubP(U2, t2);

                sharedDataC[i0] = AddP(U3, t3);
                sharedDataC[i1] = SubP(U3, t3);
            }

            if constexpr (Mode == Multiway::FourWay) {
                const uint64_t U1 = sharedDataA[i0];
                const uint64_t V1 = sharedDataA[i1];

                const uint64_t U2 = sharedDataB[i0];
                const uint64_t V2 = sharedDataB[i1];

                const uint64_t U3 = sharedDataC[i0];
                const uint64_t V3 = sharedDataC[i1];

                const uint64_t U4 = sharedDataD[i0];
                const uint64_t V4 = sharedDataD[i1];

                const uint64_t t1 = MontgomeryMul(grid, block, debugCombo, V1, wj);
                const uint64_t t2 = MontgomeryMul(grid, block, debugCombo, V2, wj);
                const uint64_t t3 = MontgomeryMul(grid, block, debugCombo, V3, wj);
                const uint64_t t4 = MontgomeryMul(grid, block, debugCombo, V4, wj);

                sharedDataA[i0] = AddP(U1, t1);
                sharedDataA[i1] = SubP(U1, t1);

                sharedDataB[i0] = AddP(U2, t2);
                sharedDataB[i1] = SubP(U2, t2);

                sharedDataC[i0] = AddP(U3, t3);
                sharedDataC[i1] = SubP(U3, t3);

                sharedDataD[i0] = AddP(U4, t4);
                sharedDataD[i1] = SubP(U4, t4);
            }
        }

        // Original per-stage barrier
        block.sync();
    }

    // Store tiles back to global
    for (uint32_t t = block.thread_index().x; t < len; t += block.size()) {
        if constexpr (Mode == Multiway::OneWay || Mode == Multiway::TwoWay ||
                      Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
            A[tile * TS + t] = sharedDataA[t];
        }

        if constexpr (Mode == Multiway::TwoWay || Mode == Multiway::ThreeWay ||
                      Mode == Multiway::FourWay) {
            B[tile * TS + t] = sharedDataB[t];
        }

        if constexpr (Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
            C[tile * TS + t] = sharedDataC[t];
        }

        if constexpr (Mode == Multiway::FourWay) {
            D[tile * TS + t] = sharedDataD[t];
        }
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessLoadedTilePhase1DIFSM(cooperative_groups::thread_block &block,
                             cooperative_groups::grid_group &grid,
                             DebugGlobalCount<SharkFloatParams> *debugCombo,
                             uint64_t *sharedDataA,
                             uint64_t *sharedDataB,
                             uint64_t *sharedDataC,
                             uint64_t *sharedDataD,
                             uint64_t *__restrict A,
                             uint64_t *__restrict B,
                             uint64_t *__restrict C,
                             uint64_t *__restrict D,
                             const uint64_t *sharedTwiddles,
                             const uint64_t *__restrict stageTwiddles,
                             uint32_t tile,
                             uint32_t TS,
                             uint32_t len,
                             uint32_t S1,
                             uint32_t cachedStages)
{
    static_assert(Mode != Multiway::SevenWay);

    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();

    // Forward DIF stages run from the largest local span to the smallest one. The upper output
    // remains a sum, while the lower output is the twiddled difference.
    for (uint32_t stage = S1; stage > 0u; --stage) {
        const uint32_t butterflySpan = 1u << stage;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint32_t totalPairs = len >> 1u;
        const bool useSharedTwiddles = stage <= cachedStages && cachedStages > 0u;
        const uint64_t *SharkRestrict twiddleBaseForStage =
            useSharedTwiddles ? sharedTwiddles + twiddleStageOffset : stageTwiddles + twiddleStageOffset;

        for (uint32_t pair = tid; pair < totalPairs; pair += step) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t index0 = group * butterflySpan + j;
            const uint32_t index1 = index0 + halfSpan;
            const uint64_t twiddle = twiddleBaseForStage[j];

            if constexpr (Mode == Multiway::OneWay) {
                const uint64_t upper = sharedDataA[index0];
                const uint64_t lower = sharedDataA[index1];
                sharedDataA[index0] = AddP(upper, lower);
                sharedDataA[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upper, lower), twiddle);
            } else if constexpr (Mode == Multiway::TwoWay) {
                const uint64_t upperA = sharedDataA[index0];
                const uint64_t lowerA = sharedDataA[index1];
                const uint64_t upperB = sharedDataB[index0];
                const uint64_t lowerB = sharedDataB[index1];
                sharedDataA[index0] = AddP(upperA, lowerA);
                sharedDataA[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
                sharedDataB[index0] = AddP(upperB, lowerB);
                sharedDataB[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
            } else if constexpr (Mode == Multiway::ThreeWay) {
                const uint64_t upperA = sharedDataA[index0];
                const uint64_t lowerA = sharedDataA[index1];
                const uint64_t upperB = sharedDataB[index0];
                const uint64_t lowerB = sharedDataB[index1];
                const uint64_t upperC = sharedDataC[index0];
                const uint64_t lowerC = sharedDataC[index1];
                sharedDataA[index0] = AddP(upperA, lowerA);
                sharedDataA[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
                sharedDataB[index0] = AddP(upperB, lowerB);
                sharedDataB[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
                sharedDataC[index0] = AddP(upperC, lowerC);
                sharedDataC[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
            } else {
                const uint64_t upperA = sharedDataA[index0];
                const uint64_t lowerA = sharedDataA[index1];
                const uint64_t upperB = sharedDataB[index0];
                const uint64_t lowerB = sharedDataB[index1];
                const uint64_t upperC = sharedDataC[index0];
                const uint64_t lowerC = sharedDataC[index1];
                const uint64_t upperD = sharedDataD[index0];
                const uint64_t lowerD = sharedDataD[index1];
                sharedDataA[index0] = AddP(upperA, lowerA);
                sharedDataA[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
                sharedDataB[index0] = AddP(upperB, lowerB);
                sharedDataB[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
                sharedDataC[index0] = AddP(upperC, lowerC);
                sharedDataC[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
                sharedDataD[index0] = AddP(upperD, lowerD);
                sharedDataD[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperD, lowerD), twiddle);
            }
        }

        block.sync();
    }

    for (uint32_t index = block.thread_index().x; index < len; index += block.size()) {
        if constexpr (Mode == Multiway::OneWay || Mode == Multiway::TwoWay ||
                      Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
            A[tile * TS + index] = sharedDataA[index];
        }
        if constexpr (Mode == Multiway::TwoWay || Mode == Multiway::ThreeWay ||
                      Mode == Multiway::FourWay) {
            B[tile * TS + index] = sharedDataB[index];
        }
        if constexpr (Mode == Multiway::ThreeWay || Mode == Multiway::FourWay) {
            C[tile * TS + index] = sharedDataC[index];
        }
        if constexpr (Mode == Multiway::FourWay) {
            D[tile * TS + index] = sharedDataD[index];
        }
    }
}

template <class SharkFloatParams, Multiway OneTwoThree, uint32_t TileSizeLog2>
static __device__ inline uint32_t
SmallRadixPhase1_SM(uint64_t *sharedData,
                    cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t *__restrict A,
                    uint64_t *__restrict B,
                    uint64_t *__restrict C,
                    uint64_t *__restrict D,
                    uint32_t N,
                    uint32_t stages,
                    const uint64_t *__restrict stageTwiddles,
                    uint64_t *__restrict E = nullptr,
                    uint64_t *__restrict F = nullptr,
                    uint64_t *__restrict G = nullptr)
{
    namespace cg = cooperative_groups;

    constexpr uint32_t TS = 1u << TileSizeLog2;

    auto countTrailingZeros = [](uint32_t x) -> uint32_t {
        uint32_t c = 0;
        while ((x & 1u) == 0u) {
            x >>= 1u;
            ++c;
        }
        return c;
    };

    const uint32_t S0 = (stages < TileSizeLog2) ? stages : TileSizeLog2;
    const uint32_t rem = (N & (TS - 1u));
    const uint32_t tailLength = (rem == 0u) ? TS : rem;
    const uint32_t tailStageCapacity = (rem == 0u) ? TileSizeLog2 : countTrailingZeros(tailLength);
    const uint32_t S1 = (S0 < tailStageCapacity) ? S0 : tailStageCapacity;
    if (S1 == 0)
        return 0;

    const uint32_t tiles = (N + TS - 1u) / TS;

    // Carve the tile buffers from the 16-byte-aligned shared-memory base.
    auto *const sharedDataA = sharedData;
    auto *const sharedDataB = sharedDataA + TS;
    auto *const sharedDataC = sharedDataB + TS;
    [[maybe_unused]] auto *const sharedDataD = sharedDataC + TS;
    [[maybe_unused]] auto *const sharedDataE = sharedDataD + TS;

    // ----------------------------------------------------------------------
    // Shared twiddle cache for the *first few stages* (e.g. first 6).
    // We copy all needed twiddles for those stages once, up front.
    // Global flattened layout: stage s uses indices
    //   [twiddleBase .. twiddleBase + half-1], twiddleBase = 2^(s-1) - 1, half = 2^(s-1).
    // Total twiddles for stages 1..K = 2^K - 1.
    // ----------------------------------------------------------------------
    constexpr uint32_t MaxCachedStages = 7;
    [[maybe_unused]] constexpr uint32_t MaxCachedTwiddles =
        (1u << MaxCachedStages) - 1; // = 127; the aligned copy includes one padding entry.

    // Reserve shared space for cached twiddles after the last tile buffer.
    auto *const sharedTwiddles =
        OneTwoThree == Multiway::SevenWay
            ? (sharedDataE + TS)
            : (OneTwoThree == Multiway::FourWay ? (sharedDataD + TS) : (sharedDataC + TS));

    const uint32_t cachedStages = (S1 < MaxCachedStages) ? S1 : MaxCachedStages;

    const uint32_t cachedTwiddles = (cachedStages > 0) ? ((1u << cachedStages) - 1u) : 0u;

    if (cachedTwiddles > 0) {
        // Copy all twiddles for stages 1..cachedStages in one shot:
        // these are stageTwiddles[0 .. cachedTwiddles-1].
        auto alignWorkspace = [](size_t value, size_t alignment) {
            return (value + alignment - 1) & ~(alignment - 1);
        };

        constexpr auto alignment = 16u;
        const auto alignedSize = alignWorkspace(cachedTwiddles * sizeof(uint64_t), alignment);
        cg::memcpy_async(
            block, sharedTwiddles, stageTwiddles, cuda::aligned_size_t<alignment>(alignedSize));
    }

    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const bool isLast = (tile == tiles - 1);
        const bool hasNextTile = (tile + gridDim.x < tiles);
        const uint32_t len = isLast ? tailLength : TS; // divisible by 2^S1

        if constexpr (OneTwoThree == Multiway::SevenWay) {
            LoadOneTilePhase1SM<SharkFloatParams, Multiway::ThreeWay>(
                block, sharedDataA, sharedDataB, sharedDataC, nullptr, A, B, C, nullptr, tile, TS, len);
            LoadOneTilePhase1SM<SharkFloatParams, Multiway::TwoWay>(block,
                                                                    sharedDataD,
                                                                    sharedDataE,
                                                                    nullptr,
                                                                    nullptr,
                                                                    D,
                                                                    E,
                                                                    nullptr,
                                                                    nullptr,
                                                                    tile,
                                                                    TS,
                                                                    len);

            // The two newest async-copy groups are D/E. Wait for the earlier twiddle and A/B/C
            // groups, then hide D/E's remaining latency behind the A/B/C shared-memory stages.
            cg::wait_prior<2>(block);
            ProcessLoadedTilePhase1SM<SharkFloatParams, Multiway::ThreeWay>(block,
                                                                            grid,
                                                                            debugCombo,
                                                                            sharedDataA,
                                                                            sharedDataB,
                                                                            sharedDataC,
                                                                            nullptr,
                                                                            A,
                                                                            B,
                                                                            C,
                                                                            nullptr,
                                                                            sharedTwiddles,
                                                                            stageTwiddles,
                                                                            tile,
                                                                            TS,
                                                                            len,
                                                                            S1,
                                                                            cachedStages);

            // A/B/C have been stored, so their shared buffers can hold F/G. D/E should normally
            // already be resident after the A/B/C work, but the wait is required before reuse.
            cg::wait(block);
            LoadOneTilePhase1SM<SharkFloatParams, Multiway::TwoWay>(block,
                                                                    sharedDataA,
                                                                    sharedDataB,
                                                                    nullptr,
                                                                    nullptr,
                                                                    F,
                                                                    G,
                                                                    nullptr,
                                                                    nullptr,
                                                                    tile,
                                                                    TS,
                                                                    len);
            cg::wait(block);
            ProcessLoadedTilePhase1SM<SharkFloatParams, Multiway::FourWay>(block,
                                                                           grid,
                                                                           debugCombo,
                                                                           sharedDataD,
                                                                           sharedDataE,
                                                                           sharedDataA,
                                                                           sharedDataB,
                                                                           D,
                                                                           E,
                                                                           F,
                                                                           G,
                                                                           sharedTwiddles,
                                                                           stageTwiddles,
                                                                           tile,
                                                                           TS,
                                                                           len,
                                                                           S1,
                                                                           cachedStages);
        } else {
            LoadOneTilePhase1SM<SharkFloatParams, OneTwoThree>(
                block, sharedDataA, sharedDataB, sharedDataC, sharedDataD, A, B, C, D, tile, TS, len);
            cg::wait(block);
            ProcessLoadedTilePhase1SM<SharkFloatParams, OneTwoThree>(block,
                                                                     grid,
                                                                     debugCombo,
                                                                     sharedDataA,
                                                                     sharedDataB,
                                                                     sharedDataC,
                                                                     sharedDataD,
                                                                     A,
                                                                     B,
                                                                     C,
                                                                     D,
                                                                     sharedTwiddles,
                                                                     stageTwiddles,
                                                                     tile,
                                                                     TS,
                                                                     len,
                                                                     S1,
                                                                     cachedStages);
        }

        if (hasNextTile) {
            block.sync();
        }
    }

    // Blocks with no assigned tile can still have the one-time twiddle copy outstanding. Drain it
    // before phase 2 reuses the same shared-memory allocation for per-warp pages.
    cg::wait(block);
    grid.sync();
    return S1;
}

template <class SharkFloatParams, Multiway Mode, uint32_t TileSizeLog2>
static __device__ inline uint32_t
SmallRadixPhase1DIF_SM(uint64_t *sharedData,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugGlobalCount<SharkFloatParams> *debugCombo,
                       uint64_t *__restrict A,
                       uint64_t *__restrict B,
                       uint64_t *__restrict C,
                       uint64_t *__restrict D,
                       uint32_t N,
                       uint32_t stages,
                       const uint64_t *__restrict stageTwiddles)
{
    static_assert(Mode != Multiway::SevenWay);

    namespace cg = cooperative_groups;

    constexpr uint32_t TS = 1u << TileSizeLog2;

    auto countTrailingZeros = [](uint32_t value) -> uint32_t {
        uint32_t count = 0;
        while ((value & 1u) == 0u) {
            value >>= 1u;
            ++count;
        }
        return count;
    };

    const uint32_t S0 = stages < TileSizeLog2 ? stages : TileSizeLog2;
    const uint32_t rem = N & (TS - 1u);
    const uint32_t tailLength = rem == 0u ? TS : rem;
    const uint32_t tailStageCapacity = rem == 0u ? TileSizeLog2 : countTrailingZeros(tailLength);
    const uint32_t S1 = S0 < tailStageCapacity ? S0 : tailStageCapacity;
    if (S1 == 0u)
        return 0u;

    const uint32_t tiles = (N + TS - 1u) / TS;
    auto *const sharedDataA = sharedData;
    auto *const sharedDataB = sharedDataA + TS;
    auto *const sharedDataC = sharedDataB + TS;
    auto *const sharedDataD = sharedDataC + TS;
    auto *const sharedTwiddles = Mode == Multiway::FourWay ? sharedDataD + TS : sharedDataC + TS;

    constexpr uint32_t MaxCachedStages = 7u;
    const uint32_t cachedStages = S1 < MaxCachedStages ? S1 : MaxCachedStages;
    const uint32_t cachedTwiddles = cachedStages > 0u ? (1u << cachedStages) - 1u : 0u;

    if (cachedTwiddles > 0u) {
        auto alignWorkspace = [](size_t value, size_t alignment) {
            return (value + alignment - 1u) & ~(alignment - 1u);
        };
        constexpr auto alignment = 16u;
        const auto alignedSize = alignWorkspace(cachedTwiddles * sizeof(uint64_t), alignment);
        cg::memcpy_async(
            block, sharedTwiddles, stageTwiddles, cuda::aligned_size_t<alignment>(alignedSize));
    }

    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const bool isLast = tile == tiles - 1u;
        const bool hasNextTile = tile + gridDim.x < tiles;
        const uint32_t len = isLast ? tailLength : TS;

        LoadOneTilePhase1SM<SharkFloatParams, Mode>(
            block, sharedDataA, sharedDataB, sharedDataC, sharedDataD, A, B, C, D, tile, TS, len);
        cg::wait(block);
        ProcessLoadedTilePhase1DIFSM<SharkFloatParams, Mode>(block,
                                                             grid,
                                                             debugCombo,
                                                             sharedDataA,
                                                             sharedDataB,
                                                             sharedDataC,
                                                             sharedDataD,
                                                             A,
                                                             B,
                                                             C,
                                                             D,
                                                             sharedTwiddles,
                                                             stageTwiddles,
                                                             tile,
                                                             TS,
                                                             len,
                                                             S1,
                                                             cachedStages);

        if (hasNextTile)
            block.sync();
    }

    cg::wait(block);
    grid.sync();
    return S1;
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessLoadedTileDIFInPlace(cooperative_groups::thread_block &block,
                            cooperative_groups::grid_group &grid,
                            DebugGlobalCount<SharkFloatParams> *debugCombo,
                            uint64_t *sharedDataA,
                            uint64_t *sharedDataB,
                            uint64_t *sharedDataC,
                            uint64_t *sharedDataD,
                            const uint64_t *sharedTwiddles,
                            const uint64_t *stageTwiddles,
                            uint32_t len,
                            uint32_t S1,
                            uint32_t cachedStages,
                            uint32_t lowestStageExclusive)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::FourWay);

    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();
    for (uint32_t stage = S1; stage > lowestStageExclusive; --stage) {
        const uint32_t butterflySpan = 1u << stage;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint32_t totalPairs = len >> 1u;
        const bool useSharedTwiddles = stage <= cachedStages && cachedStages > 0u;
        const uint64_t *SharkRestrict twiddleBaseForStage =
            useSharedTwiddles ? sharedTwiddles + twiddleStageOffset : stageTwiddles + twiddleStageOffset;

        for (uint32_t pair = tid; pair < totalPairs; pair += step) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t index0 = group * butterflySpan + j;
            const uint32_t index1 = index0 + halfSpan;
            const uint64_t twiddle = twiddleBaseForStage[j];

            const uint64_t upperA = sharedDataA[index0];
            const uint64_t lowerA = sharedDataA[index1];
            sharedDataA[index0] = AddP(upperA, lowerA);
            sharedDataA[index1] = MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);

            const uint64_t upperB = sharedDataB[index0];
            const uint64_t lowerB = sharedDataB[index1];
            sharedDataB[index0] = AddP(upperB, lowerB);
            sharedDataB[index1] = MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);

            if constexpr (Mode == Multiway::FourWay) {
                const uint64_t upperC = sharedDataC[index0];
                const uint64_t lowerC = sharedDataC[index1];
                const uint64_t upperD = sharedDataD[index0];
                const uint64_t lowerD = sharedDataD[index1];
                sharedDataC[index0] = AddP(upperC, lowerC);
                sharedDataC[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
                sharedDataD[index0] = AddP(upperD, lowerD);
                sharedDataD[index1] =
                    MontgomeryMul(grid, block, debugCombo, SubP(upperD, lowerD), twiddle);
            }
        }
        block.sync();
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessLoadedTileDITInPlace(cooperative_groups::thread_block &block,
                            cooperative_groups::grid_group &grid,
                            DebugGlobalCount<SharkFloatParams> *debugCombo,
                            uint64_t *sharedDataA,
                            uint64_t *sharedDataB,
                            uint64_t *sharedDataC,
                            uint64_t *sharedDataD,
                            const uint64_t *sharedTwiddles,
                            const uint64_t *stageTwiddles,
                            uint32_t len,
                            uint32_t firstStage,
                            uint32_t lastStage,
                            uint32_t cachedStages)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::FourWay);

    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();
    for (uint32_t stage = firstStage; stage <= lastStage; ++stage) {
        const uint32_t butterflySpan = 1u << stage;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint32_t totalPairs = len >> 1u;
        const bool useSharedTwiddles = stage <= cachedStages && cachedStages > 0u;
        const uint64_t *SharkRestrict twiddleBaseForStage =
            useSharedTwiddles ? sharedTwiddles + twiddleStageOffset : stageTwiddles + twiddleStageOffset;

        for (uint32_t pair = tid; pair < totalPairs; pair += step) {
            const uint32_t group = pair / halfSpan;
            const uint32_t j = pair - group * halfSpan;
            const uint32_t index0 = group * butterflySpan + j;
            const uint32_t index1 = index0 + halfSpan;
            const uint64_t twiddle = twiddleBaseForStage[j];

            const uint64_t upperA = sharedDataA[index0];
            const uint64_t lowerA = sharedDataA[index1];
            const uint64_t upperB = sharedDataB[index0];
            const uint64_t lowerB = sharedDataB[index1];
            const uint64_t productA = MontgomeryMul(grid, block, debugCombo, lowerA, twiddle);
            const uint64_t productB = MontgomeryMul(grid, block, debugCombo, lowerB, twiddle);
            sharedDataA[index0] = AddP(upperA, productA);
            sharedDataA[index1] = SubP(upperA, productA);
            sharedDataB[index0] = AddP(upperB, productB);
            sharedDataB[index1] = SubP(upperB, productB);

            if constexpr (Mode == Multiway::FourWay) {
                const uint64_t upperC = sharedDataC[index0];
                const uint64_t lowerC = sharedDataC[index1];
                const uint64_t upperD = sharedDataD[index0];
                const uint64_t lowerD = sharedDataD[index1];
                const uint64_t productC = MontgomeryMul(grid, block, debugCombo, lowerC, twiddle);
                const uint64_t productD = MontgomeryMul(grid, block, debugCombo, lowerD, twiddle);
                sharedDataC[index0] = AddP(upperC, productC);
                sharedDataC[index1] = SubP(upperC, productC);
                sharedDataD[index0] = AddP(upperD, productD);
                sharedDataD[index1] = SubP(upperD, productD);
            }
        }
        block.sync();
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessLoadedTileDITFinalStageToGlobal(cooperative_groups::thread_block &block,
                                       cooperative_groups::grid_group &grid,
                                       DebugGlobalCount<SharkFloatParams> *debugCombo,
                                       const uint64_t *sharedDataA,
                                       const uint64_t *sharedDataB,
                                       const uint64_t *sharedDataC,
                                       const uint64_t *sharedDataD,
                                       uint64_t *SharkRestrict outputA,
                                       uint64_t *SharkRestrict outputB,
                                       uint64_t *SharkRestrict outputC,
                                       uint64_t *SharkRestrict outputD,
                                       const uint64_t *sharedTwiddles,
                                       const uint64_t *stageTwiddles,
                                       uint32_t outputBase,
                                       uint32_t len,
                                       uint32_t stage,
                                       uint32_t cachedStages)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::FourWay);

    const uint32_t butterflySpan = 1u << stage;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleStageOffset = halfSpan - 1u;
    const bool useSharedTwiddles = stage <= cachedStages && cachedStages > 0u;
    const uint64_t *SharkRestrict twiddleBaseForStage =
        useSharedTwiddles ? sharedTwiddles + twiddleStageOffset : stageTwiddles + twiddleStageOffset;
    const uint32_t tid = block.thread_index().x;
    const uint32_t step = block.size();
    const uint32_t totalPairs = len >> 1u;

    for (uint32_t pair = tid; pair < totalPairs; pair += step) {
        const uint32_t group = pair / halfSpan;
        const uint32_t j = pair - group * halfSpan;
        const uint32_t index0 = group * butterflySpan + j;
        const uint32_t index1 = index0 + halfSpan;
        const uint64_t twiddle = twiddleBaseForStage[j];

        const uint64_t upperA = sharedDataA[index0];
        const uint64_t lowerA = sharedDataA[index1];
        const uint64_t productA = MontgomeryMul(grid, block, debugCombo, lowerA, twiddle);
        outputA[outputBase + index0] = AddP(upperA, productA);
        outputA[outputBase + index1] = SubP(upperA, productA);

        const uint64_t upperB = sharedDataB[index0];
        const uint64_t lowerB = sharedDataB[index1];
        const uint64_t productB = MontgomeryMul(grid, block, debugCombo, lowerB, twiddle);
        outputB[outputBase + index0] = AddP(upperB, productB);
        outputB[outputBase + index1] = SubP(upperB, productB);

        if constexpr (Mode == Multiway::FourWay) {
            const uint64_t upperC = sharedDataC[index0];
            const uint64_t lowerC = sharedDataC[index1];
            const uint64_t productC = MontgomeryMul(grid, block, debugCombo, lowerC, twiddle);
            outputC[outputBase + index0] = AddP(upperC, productC);
            outputC[outputBase + index1] = SubP(upperC, productC);

            const uint64_t upperD = sharedDataD[index0];
            const uint64_t lowerD = sharedDataD[index1];
            const uint64_t productD = MontgomeryMul(grid, block, debugCombo, lowerD, twiddle);
            outputD[outputBase + index0] = AddP(upperD, productD);
            outputD[outputBase + index1] = SubP(upperD, productD);
        }
    }
}

// -----------------------------------------------------------------------------
// Per-warp, per-stage micro-tile processor.
// Handles:
//   - Mode = OneWay / TwoWay / ThreeWay / FourWay / SevenWay
//   - microTileWidth = 4 (OneWay) or 2 (Two/Three/Four/SevenWay)
// Updates jChunkIndex, tasksRemaining, blockIndex, and blockDataBaseIndex.
// -----------------------------------------------------------------------------
template <class SharkFloatParams, Multiway Mode, int microTileWidth>
static __device__ SharkForceInlineReleaseOnly void
ProcessTile(cooperative_groups::grid_group &grid,
            cooperative_groups::thread_block &block,
            DebugGlobalCount<SharkFloatParams> *debugCombo,
            uint64_t *SharkRestrict A,
            uint64_t *SharkRestrict B,
            uint64_t *SharkRestrict C,
            uint64_t *SharkRestrict D,
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
            // SevenWay extra arrays (nullptr for other modes)
            uint64_t *SharkRestrict E = nullptr,
            uint64_t *SharkRestrict F = nullptr,
            uint64_t *SharkRestrict G = nullptr)
{
    // Tile cannot cross block boundary or our assigned range
    const uint32_t roomInBlock = numJChunks - jChunkIndex;
    const uint32_t span = static_cast<uint32_t>(min(tasksRemaining, static_cast<size_t>(roomInBlock)));

    const uint32_t tileWidth =
        (microTileWidth == 4) ? std::min<uint32_t>(4u, span) : std::min<uint32_t>(2u, span);

    // Helper: load A/B/C/D/E/F/G for a given jIndex
    auto loadABCD = [&](uint32_t jIndex,
                        bool &inRange,
                        uint32_t &idxUpper,
                        uint32_t &idxLower,
                        uint64_t &aUpper,
                        uint64_t &aLower,
                        uint64_t &bUpper,
                        uint64_t &bLower,
                        uint64_t &cUpper,
                        uint64_t &cLower,
                        uint64_t &dUpper,
                        uint64_t &dLower,
                        uint64_t &eUpper,
                        uint64_t &eLower,
                        uint64_t &fUpper,
                        uint64_t &fLower,
                        uint64_t &gUpper,
                        uint64_t &gLower) {
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
            if constexpr (Mode == Multiway::ThreeWay || Mode == Multiway::FourWay ||
                          Mode == Multiway::SevenWay) {
                cUpper = C[idxUpper];
                cLower = C[idxLower];
            }
            if constexpr (Mode == Multiway::FourWay || Mode == Multiway::SevenWay) {
                dUpper = D[idxUpper];
                dLower = D[idxLower];
            }
            if constexpr (Mode == Multiway::SevenWay) {
                eUpper = E[idxUpper];
                eLower = E[idxLower];
                fUpper = F[idxUpper];
                fLower = F[idxLower];
                gUpper = G[idxUpper];
                gLower = G[idxLower];
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
                              uint64_t dUpper,
                              uint64_t dLower,
                              uint64_t eUpper,
                              uint64_t eLower,
                              uint64_t fUpper,
                              uint64_t fLower,
                              uint64_t gUpper,
                              uint64_t gLower,
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
            B[idxLower] = SubP(bUpper, tB);
        } else if constexpr (Mode == Multiway::ThreeWay) {
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            const uint64_t tB = MontgomeryMul(grid, block, debugCombo, bLower, twiddle);
            const uint64_t tC = MontgomeryMul(grid, block, debugCombo, cLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
            B[idxUpper] = AddP(bUpper, tB);
            B[idxLower] = SubP(bUpper, tB);
            C[idxUpper] = AddP(cUpper, tC);
            C[idxLower] = SubP(cUpper, tC);
        } else if constexpr (Mode == Multiway::FourWay) {
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            const uint64_t tB = MontgomeryMul(grid, block, debugCombo, bLower, twiddle);
            const uint64_t tC = MontgomeryMul(grid, block, debugCombo, cLower, twiddle);
            const uint64_t tD = MontgomeryMul(grid, block, debugCombo, dLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
            B[idxUpper] = AddP(bUpper, tB);
            B[idxLower] = SubP(bUpper, tB);
            C[idxUpper] = AddP(cUpper, tC);
            C[idxLower] = SubP(cUpper, tC);
            D[idxUpper] = AddP(dUpper, tD);
            D[idxLower] = SubP(dUpper, tD);
        } else { // SevenWay
            const uint64_t tA = MontgomeryMul(grid, block, debugCombo, aLower, twiddle);
            const uint64_t tB = MontgomeryMul(grid, block, debugCombo, bLower, twiddle);
            const uint64_t tC = MontgomeryMul(grid, block, debugCombo, cLower, twiddle);
            const uint64_t tD = MontgomeryMul(grid, block, debugCombo, dLower, twiddle);
            const uint64_t tE = MontgomeryMul(grid, block, debugCombo, eLower, twiddle);
            const uint64_t tF = MontgomeryMul(grid, block, debugCombo, fLower, twiddle);
            const uint64_t tG = MontgomeryMul(grid, block, debugCombo, gLower, twiddle);
            A[idxUpper] = AddP(aUpper, tA);
            A[idxLower] = SubP(aUpper, tA);
            B[idxUpper] = AddP(bUpper, tB);
            B[idxLower] = SubP(bUpper, tB);
            C[idxUpper] = AddP(cUpper, tC);
            C[idxLower] = SubP(cUpper, tC);
            D[idxUpper] = AddP(dUpper, tD);
            D[idxLower] = SubP(dUpper, tD);
            E[idxUpper] = AddP(eUpper, tE);
            E[idxLower] = SubP(eUpper, tE);
            F[idxUpper] = AddP(fUpper, tF);
            F[idxLower] = SubP(fUpper, tF);
            G[idxUpper] = AddP(gUpper, tG);
            G[idxLower] = SubP(gUpper, tG);
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
    uint64_t dUpper0 = 0, dLower0 = 0;
    uint64_t eUpper0 = 0, eLower0 = 0;
    uint64_t fUpper0 = 0, fLower0 = 0;
    uint64_t gUpper0 = 0, gLower0 = 0;

    loadABCD(jIndex0,
             inRange0,
             indexUpper0,
             indexLower0,
             aUpper0,
             aLower0,
             bUpper0,
             bLower0,
             cUpper0,
             cLower0,
             dUpper0,
             dLower0,
             eUpper0,
             eLower0,
             fUpper0,
             fLower0,
             gUpper0,
             gLower0);

    uint64_t twiddle0 = 0;
    if (inRange0) {
        twiddle0 = loadTwiddlePrecomputed(jIndex0, /*inRange=*/true);
    }

    // ===== position 1 (if any) =====
    bool inRange1 = false;
    uint32_t indexUpper1 = 0, indexLower1 = 0;
    uint64_t aUpper1 = 0, aLower1 = 0;
    uint64_t bUpper1 = 0, bLower1 = 0;
    uint64_t cUpper1 = 0, cLower1 = 0;
    uint64_t dUpper1 = 0, dLower1 = 0;
    uint64_t eUpper1 = 0, eLower1 = 0;
    uint64_t fUpper1 = 0, fLower1 = 0;
    uint64_t gUpper1 = 0, gLower1 = 0;
    uint64_t twiddle1 = 0;

    if (tileWidth >= 2) {
        const uint32_t jIndex1 = jIndex0 + warpSize;

        loadABCD(jIndex1,
                 inRange1,
                 indexUpper1,
                 indexLower1,
                 aUpper1,
                 aLower1,
                 bUpper1,
                 bLower1,
                 cUpper1,
                 cLower1,
                 dUpper1,
                 dLower1,
                 eUpper1,
                 eLower1,
                 fUpper1,
                 fLower1,
                 gUpper1,
                 gLower1);

        if (inRange1) {
            twiddle1 = loadTwiddlePrecomputed(jIndex1, /*inRange=*/true);
        }
    }

    // ===== position 2/3 (OneWay only) =====
    bool inRange2 = false, inRange3 = false;
    uint32_t indexUpper2 = 0, indexLower2 = 0;
    uint64_t aUpper2 = 0, aLower2 = 0;
    uint64_t bUpper2 = 0, bLower2 = 0;
    uint64_t cUpper2 = 0, cLower2 = 0;
    uint64_t dUpper2 = 0, dLower2 = 0;
    uint64_t eUpper2 = 0, eLower2 = 0;
    uint64_t fUpper2 = 0, fLower2 = 0;
    uint64_t gUpper2 = 0, gLower2 = 0;
    uint32_t indexUpper3 = 0, indexLower3 = 0;
    uint64_t aUpper3 = 0, aLower3 = 0;
    uint64_t bUpper3 = 0, bLower3 = 0;
    uint64_t cUpper3 = 0, cLower3 = 0;
    uint64_t dUpper3 = 0, dLower3 = 0;
    uint64_t eUpper3 = 0, eLower3 = 0;
    uint64_t fUpper3 = 0, fLower3 = 0;
    uint64_t gUpper3 = 0, gLower3 = 0;
    uint64_t twiddle2 = 0, twiddle3 = 0;

    if constexpr (microTileWidth == 4) {
        if (tileWidth >= 3) {
            const uint32_t jIndex2 = jIndex0 + 2u * warpSize;
            loadABCD(jIndex2,
                     inRange2,
                     indexUpper2,
                     indexLower2,
                     aUpper2,
                     aLower2,
                     bUpper2,
                     bLower2,
                     cUpper2,
                     cLower2,
                     dUpper2,
                     dLower2,
                     eUpper2,
                     eLower2,
                     fUpper2,
                     fLower2,
                     gUpper2,
                     gLower2);

            if (inRange2) {
                twiddle2 = loadTwiddlePrecomputed(jIndex2, /*inRange=*/true);
            }
        }
        if (tileWidth >= 4) {
            const uint32_t jIndex3 = jIndex0 + 3u * warpSize;
            loadABCD(jIndex3,
                     inRange3,
                     indexUpper3,
                     indexLower3,
                     aUpper3,
                     aLower3,
                     bUpper3,
                     bLower3,
                     cUpper3,
                     cLower3,
                     dUpper3,
                     dLower3,
                     eUpper3,
                     eLower3,
                     fUpper3,
                     fLower3,
                     gUpper3,
                     gLower3);

            if (inRange3) {
                twiddle3 = loadTwiddlePrecomputed(jIndex3, /*inRange=*/true);
            }
        }
    }

    // ---- compute/store: position 0 ----
    if (inRange0) {
        applyButterfly(indexUpper0,
                       indexLower0,
                       aUpper0,
                       aLower0,
                       bUpper0,
                       bLower0,
                       cUpper0,
                       cLower0,
                       dUpper0,
                       dLower0,
                       eUpper0,
                       eLower0,
                       fUpper0,
                       fLower0,
                       gUpper0,
                       gLower0,
                       twiddle0);
    }

    // ---- compute/store: position 1 ----
    if (tileWidth >= 2 && inRange1) {
        applyButterfly(indexUpper1,
                       indexLower1,
                       aUpper1,
                       aLower1,
                       bUpper1,
                       bLower1,
                       cUpper1,
                       cLower1,
                       dUpper1,
                       dLower1,
                       eUpper1,
                       eLower1,
                       fUpper1,
                       fLower1,
                       gUpper1,
                       gLower1,
                       twiddle1);
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
                           dUpper2,
                           dLower2,
                           eUpper2,
                           eLower2,
                           fUpper2,
                           fLower2,
                           gUpper2,
                           gLower2,
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
                           dUpper3,
                           dLower3,
                           eUpper3,
                           eLower3,
                           fUpper3,
                           fLower3,
                           gUpper3,
                           gLower3,
                           twiddle3);
        }
    }

    // ---- advance within block by 'tileWidth' and update state ----
    jChunkIndex += tileWidth;
    tasksRemaining -= tileWidth;

    if (jChunkIndex == numJChunks) {
        // wrap to next block
        jChunkIndex = 0;
        blockIndex += 1;
        blockDataBaseIndex += butterflySpan;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix2Butterfly(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t *SharkRestrict data,
                     uint32_t upperIndex,
                     uint32_t lowerIndex,
                     uint64_t upper,
                     uint64_t lower,
                     uint64_t twiddle)
{
    const uint64_t product = MontgomeryMul(grid, block, debugCombo, lower, twiddle);
    data[upperIndex] = AddP(upper, product);
    data[lowerIndex] = SubP(upper, product);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix4StagePair(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t *SharkRestrict data,
                     uint32_t index0,
                     uint32_t index1,
                     uint32_t index2,
                     uint32_t index3,
                     uint64_t value0,
                     uint64_t value1,
                     uint64_t value2,
                     uint64_t value3,
                     uint64_t firstStageTwiddle,
                     uint64_t secondStageTwiddle0,
                     uint64_t secondStageTwiddle1)
{
    const uint64_t firstProduct0 = MontgomeryMul(grid, block, debugCombo, value1, firstStageTwiddle);
    const uint64_t firstProduct1 = MontgomeryMul(grid, block, debugCombo, value3, firstStageTwiddle);
    const uint64_t firstValue0 = AddP(value0, firstProduct0);
    const uint64_t firstValue1 = SubP(value0, firstProduct0);
    const uint64_t firstValue2 = AddP(value2, firstProduct1);
    const uint64_t firstValue3 = SubP(value2, firstProduct1);

    const uint64_t secondProduct0 =
        MontgomeryMul(grid, block, debugCombo, firstValue2, secondStageTwiddle0);
    const uint64_t secondProduct1 =
        MontgomeryMul(grid, block, debugCombo, firstValue3, secondStageTwiddle1);

    data[index0] = AddP(firstValue0, secondProduct0);
    data[index2] = SubP(firstValue0, secondProduct0);
    data[index1] = AddP(firstValue1, secondProduct1);
    data[index3] = SubP(firstValue1, secondProduct1);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreRadix4DIFStagePair(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                        uint64_t *SharkRestrict data,
                        uint32_t index0,
                        uint32_t index1,
                        uint32_t index2,
                        uint32_t index3,
                        uint64_t value0,
                        uint64_t value1,
                        uint64_t value2,
                        uint64_t value3,
                        uint64_t firstStageTwiddle0,
                        uint64_t firstStageTwiddle1,
                        uint64_t secondStageTwiddle)
{
    const uint64_t firstValue0 = AddP(value0, value2);
    const uint64_t firstValue1 = AddP(value1, value3);
    const uint64_t firstProduct0 =
        MontgomeryMul(grid, block, debugCombo, SubP(value0, value2), firstStageTwiddle0);
    const uint64_t firstProduct1 =
        MontgomeryMul(grid, block, debugCombo, SubP(value1, value3), firstStageTwiddle1);

    const uint64_t secondValue0 = AddP(firstValue0, firstValue1);
    const uint64_t secondValue1 =
        MontgomeryMul(grid, block, debugCombo, SubP(firstValue0, firstValue1), secondStageTwiddle);
    const uint64_t secondValue2 = AddP(firstProduct0, firstProduct1);
    const uint64_t secondValue3 =
        MontgomeryMul(grid, block, debugCombo, SubP(firstProduct0, firstProduct1), secondStageTwiddle);

    data[index0] = secondValue0;
    data[index1] = secondValue1;
    data[index2] = secondValue2;
    data[index3] = secondValue3;
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessRadix2DIFStage(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t *SharkRestrict A,
                      uint64_t *SharkRestrict B,
                      uint64_t *SharkRestrict C,
                      uint64_t *SharkRestrict D,
                      uint64_t *SharkRestrict E,
                      uint64_t *SharkRestrict F,
                      uint64_t *SharkRestrict G,
                      const uint64_t *SharkRestrict stageTwiddleTable,
                      uint32_t transformSize,
                      uint32_t stageIndex,
                      size_t globalThreadIndex,
                      size_t gridSize)
{
    const uint32_t butterflySpan = 1u << stageIndex;
    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t twiddleStageOffset = halfSpan - 1u;
    const size_t butterflyCount = static_cast<size_t>(transformSize) >> 1u;

    for (size_t task = globalThreadIndex; task < butterflyCount; task += gridSize) {
        const uint32_t blockIndex = static_cast<uint32_t>(task / halfSpan);
        const uint32_t jIndex = static_cast<uint32_t>(task - static_cast<size_t>(blockIndex) * halfSpan);
        const uint32_t upperIndex = blockIndex * butterflySpan + jIndex;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        const uint64_t twiddle = stageTwiddleTable[twiddleStageOffset + jIndex];

        if constexpr (Mode == Multiway::OneWay) {
            const uint64_t upper = A[upperIndex];
            const uint64_t lower = A[lowerIndex];
            A[upperIndex] = AddP(upper, lower);
            A[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upper, lower), twiddle);
        } else if constexpr (Mode == Multiway::TwoWay) {
            const uint64_t upperA = A[upperIndex];
            const uint64_t lowerA = A[lowerIndex];
            const uint64_t upperB = B[upperIndex];
            const uint64_t lowerB = B[lowerIndex];
            A[upperIndex] = AddP(upperA, lowerA);
            A[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
            B[upperIndex] = AddP(upperB, lowerB);
            B[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
        } else if constexpr (Mode == Multiway::ThreeWay) {
            const uint64_t upperA = A[upperIndex];
            const uint64_t lowerA = A[lowerIndex];
            const uint64_t upperB = B[upperIndex];
            const uint64_t lowerB = B[lowerIndex];
            const uint64_t upperC = C[upperIndex];
            const uint64_t lowerC = C[lowerIndex];
            A[upperIndex] = AddP(upperA, lowerA);
            A[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
            B[upperIndex] = AddP(upperB, lowerB);
            B[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
            C[upperIndex] = AddP(upperC, lowerC);
            C[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
        } else if constexpr (Mode == Multiway::FourWay) {
            const uint64_t upperA = A[upperIndex];
            const uint64_t lowerA = A[lowerIndex];
            const uint64_t upperB = B[upperIndex];
            const uint64_t lowerB = B[lowerIndex];
            const uint64_t upperC = C[upperIndex];
            const uint64_t lowerC = C[lowerIndex];
            const uint64_t upperD = D[upperIndex];
            const uint64_t lowerD = D[lowerIndex];
            A[upperIndex] = AddP(upperA, lowerA);
            A[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
            B[upperIndex] = AddP(upperB, lowerB);
            B[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
            C[upperIndex] = AddP(upperC, lowerC);
            C[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
            D[upperIndex] = AddP(upperD, lowerD);
            D[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperD, lowerD), twiddle);
        } else {
            const uint64_t upperA = A[upperIndex];
            const uint64_t lowerA = A[lowerIndex];
            const uint64_t upperB = B[upperIndex];
            const uint64_t lowerB = B[lowerIndex];
            const uint64_t upperC = C[upperIndex];
            const uint64_t lowerC = C[lowerIndex];
            const uint64_t upperD = D[upperIndex];
            const uint64_t lowerD = D[lowerIndex];
            const uint64_t upperE = E[upperIndex];
            const uint64_t lowerE = E[lowerIndex];
            const uint64_t upperF = F[upperIndex];
            const uint64_t lowerF = F[lowerIndex];
            const uint64_t upperG = G[upperIndex];
            const uint64_t lowerG = G[lowerIndex];
            A[upperIndex] = AddP(upperA, lowerA);
            A[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperA, lowerA), twiddle);
            B[upperIndex] = AddP(upperB, lowerB);
            B[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperB, lowerB), twiddle);
            C[upperIndex] = AddP(upperC, lowerC);
            C[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperC, lowerC), twiddle);
            D[upperIndex] = AddP(upperD, lowerD);
            D[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperD, lowerD), twiddle);
            E[upperIndex] = AddP(upperE, lowerE);
            E[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperE, lowerE), twiddle);
            F[upperIndex] = AddP(upperF, lowerF);
            F[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperF, lowerF), twiddle);
            G[upperIndex] = AddP(upperG, lowerG);
            G[lowerIndex] = MontgomeryMul(grid, block, debugCombo, SubP(upperG, lowerG), twiddle);
        }
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ForwardDIFScalar(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 uint64_t *SharkRestrict A,
                 uint64_t *SharkRestrict B,
                 uint64_t *SharkRestrict C,
                 uint64_t *SharkRestrict D,
                 uint64_t *SharkRestrict E,
                 uint64_t *SharkRestrict F,
                 uint64_t *SharkRestrict G,
                 const uint64_t *SharkRestrict stageTwiddleTable,
                 uint32_t transformSize,
                 uint32_t numStages,
                 size_t globalThreadIndex,
                 size_t gridSize)
{
    for (uint32_t stageIndex = numStages; stageIndex > 0u; --stageIndex) {
        ProcessRadix2DIFStage<SharkFloatParams, Mode>(grid,
                                                      block,
                                                      debugCombo,
                                                      A,
                                                      B,
                                                      C,
                                                      D,
                                                      E,
                                                      F,
                                                      G,
                                                      stageTwiddleTable,
                                                      transformSize,
                                                      stageIndex,
                                                      globalThreadIndex,
                                                      gridSize);
        grid.sync();
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessRadix4DIFStagePairPipelined(uint64_t *sharedData,
                                   cooperative_groups::grid_group &grid,
                                   cooperative_groups::thread_block &block,
                                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                                   uint64_t *SharkRestrict A,
                                   uint64_t *SharkRestrict B,
                                   uint64_t *SharkRestrict C,
                                   uint64_t *SharkRestrict D,
                                   const uint64_t *SharkRestrict stageTwiddleTable,
                                   uint32_t transformSize,
                                   uint32_t firstStageIndex,
                                   uint32_t warpIndex,
                                   uint32_t numWarpsGrid)
{
    static_assert(Mode != Multiway::SevenWay);

    namespace cg = cooperative_groups;

    constexpr uint32_t WarpSize = 32u;
    constexpr uint32_t PlanesPerPage = 4u;
    constexpr uint32_t PipelineSlots = 2u;
    constexpr uint32_t MaxWarpsPerBlock = 512u / WarpSize;
    constexpr size_t MaxPipelineBytes =
        PipelineSlots * MaxWarpsPerBlock * PlanesPerPage * WarpSize * sizeof(uint64_t);
    static_assert(MaxPipelineBytes == 32u * 1024u);

    const auto warp = cg::tiled_partition<WarpSize>(block);
    const uint32_t laneIndex = warp.thread_rank();
    const uint32_t localWarpIndex = block.thread_index().x / WarpSize;
    const uint32_t warpsPerBlock = block.size() / WarpSize;
    const uint32_t slotStride = warpsPerBlock * PlanesPerPage * WarpSize;
    uint64_t *const warpPage0 = sharedData + localWarpIndex * PlanesPerPage * WarpSize;
    uint64_t *const warpPage1 = sharedData + slotStride + localWarpIndex * PlanesPerPage * WarpSize;

    const uint32_t firstHalfSpan = 1u << (firstStageIndex - 1u);
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint32_t combinedSpan = firstHalfSpan << 1u;
    const uint64_t *const firstStageTwiddles = stageTwiddleTable + firstHalfSpan - 1u;
    const uint64_t *const secondStageTwiddles = stageTwiddleTable + secondHalfSpan - 1u;
    const uint32_t numJChunks = secondHalfSpan / WarpSize;
    const uint32_t numBlocks = transformSize / combinedSpan;
    const size_t totalTasks = static_cast<size_t>(numBlocks) * numJChunks;
    const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid;
    const size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
    const size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

    if (warpTaskBegin >= warpTaskEnd)
        return;

    constexpr size_t PlaneBytes = WarpSize * sizeof(uint64_t);
    auto issuePage = [&](size_t taskIndex, uint64_t *page) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t pageBase = blockIndex * combinedSpan + jChunkIndex * WarpSize;

        cg::memcpy_async(warp, page, A + pageBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + WarpSize, A + pageBase + secondHalfSpan, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(warp,
                         page + 2u * WarpSize,
                         A + pageBase + firstHalfSpan,
                         cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(warp,
                         page + 3u * WarpSize,
                         A + pageBase + firstHalfSpan + secondHalfSpan,
                         cuda::aligned_size_t<16>(PlaneBytes));
    };

    auto processDirectArray = [&](uint64_t *data,
                                  uint32_t index0,
                                  uint32_t index1,
                                  uint32_t index2,
                                  uint32_t index3,
                                  uint64_t firstTwiddle0,
                                  uint64_t firstTwiddle1,
                                  uint64_t secondTwiddle) {
        StoreRadix4DIFStagePair(grid,
                                block,
                                debugCombo,
                                data,
                                index0,
                                index1,
                                index2,
                                index3,
                                data[index0],
                                data[index1],
                                data[index2],
                                data[index3],
                                firstTwiddle0,
                                firstTwiddle1,
                                secondTwiddle);
    };

    uint32_t currentSlot = 0u;
    issuePage(warpTaskBegin, warpPage0);

    for (size_t taskIndex = warpTaskBegin; taskIndex < warpTaskEnd; ++taskIndex) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t jIndex = jChunkIndex * WarpSize + laneIndex;
        const uint32_t index0 = blockIndex * combinedSpan + jIndex;
        const uint32_t index1 = index0 + secondHalfSpan;
        const uint32_t index2 = index0 + firstHalfSpan;
        const uint32_t index3 = index2 + secondHalfSpan;
        const uint64_t firstTwiddle0 = firstStageTwiddles[jIndex];
        const uint64_t firstTwiddle1 = firstStageTwiddles[jIndex + secondHalfSpan];
        const uint64_t secondTwiddle = secondStageTwiddles[jIndex];

        if constexpr (Mode != Multiway::OneWay)
            processDirectArray(
                B, index0, index1, index2, index3, firstTwiddle0, firstTwiddle1, secondTwiddle);
        if constexpr (Mode == Multiway::ThreeWay || Mode == Multiway::FourWay)
            processDirectArray(
                C, index0, index1, index2, index3, firstTwiddle0, firstTwiddle1, secondTwiddle);
        if constexpr (Mode == Multiway::FourWay)
            processDirectArray(
                D, index0, index1, index2, index3, firstTwiddle0, firstTwiddle1, secondTwiddle);

        cg::wait(warp);

        const bool hasNextTask = taskIndex + 1u < warpTaskEnd;
        uint64_t *const currentPage = currentSlot == 0u ? warpPage0 : warpPage1;
        if (hasNextTask) {
            uint64_t *const nextPage = currentSlot == 0u ? warpPage1 : warpPage0;
            issuePage(taskIndex + 1u, nextPage);
        }

        StoreRadix4DIFStagePair(grid,
                                block,
                                debugCombo,
                                A,
                                index0,
                                index1,
                                index2,
                                index3,
                                currentPage[laneIndex],
                                currentPage[WarpSize + laneIndex],
                                currentPage[2u * WarpSize + laneIndex],
                                currentPage[3u * WarpSize + laneIndex],
                                firstTwiddle0,
                                firstTwiddle1,
                                secondTwiddle);
        currentSlot ^= 1u;
    }
}

// Pair two large radix-2 stages for the regular TwoWay/ThreeWay transforms. One array is staged
// through a four-plane per-warp page while the remaining arrays provide enough independent
// Montgomery work to cover the copy. Pairing the stages removes one complete global read/write pass
// and one grid barrier independently of how much copy latency is hidden.
template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessRadix4StagePairPipelined(uint64_t *sharedData,
                                cooperative_groups::grid_group &grid,
                                cooperative_groups::thread_block &block,
                                DebugGlobalCount<SharkFloatParams> *debugCombo,
                                uint64_t *SharkRestrict A,
                                uint64_t *SharkRestrict B,
                                uint64_t *SharkRestrict C,
                                const uint64_t *SharkRestrict stageTwiddleTable,
                                uint32_t transformSize,
                                uint32_t firstStageIndex,
                                uint32_t warpIndex,
                                uint32_t numWarpsGrid)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::ThreeWay);

    namespace cg = cooperative_groups;
    constexpr uint32_t WarpSize = 32u;
    constexpr uint32_t PlanesPerPage = 4u;
    constexpr uint32_t PipelineSlots = 2u;
    constexpr uint32_t MaxWarpsPerBlock = 512u / WarpSize;
    constexpr size_t MaxPipelineBytes =
        PipelineSlots * MaxWarpsPerBlock * PlanesPerPage * WarpSize * sizeof(uint64_t);
    static_assert(MaxPipelineBytes == 32u * 1024u);

    const auto warp = cg::tiled_partition<WarpSize>(block);
    const uint32_t laneIndex = warp.thread_rank();
    const uint32_t localWarpIndex = block.thread_index().x / WarpSize;
    const uint32_t warpsPerBlock = block.size() / WarpSize;
    const uint32_t slotStride = warpsPerBlock * PlanesPerPage * WarpSize;
    uint64_t *const warpPage0 = sharedData + localWarpIndex * PlanesPerPage * WarpSize;
    uint64_t *const warpPage1 = sharedData + slotStride + localWarpIndex * PlanesPerPage * WarpSize;

    const uint32_t firstHalfSpan = 1u << (firstStageIndex - 1u);
    const uint32_t secondHalfSpan = firstHalfSpan << 1u;
    const uint32_t combinedSpan = secondHalfSpan << 1u;
    const uint64_t *const firstStageTwiddles = stageTwiddleTable + firstHalfSpan - 1u;
    const uint64_t *const secondStageTwiddles = stageTwiddleTable + secondHalfSpan - 1u;
    const uint32_t numJChunks = firstHalfSpan / WarpSize;
    const uint32_t numBlocks = transformSize / combinedSpan;
    const size_t totalTasks = static_cast<size_t>(numBlocks) * numJChunks;
    const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid;
    const size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
    const size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

    if (warpTaskBegin >= warpTaskEnd)
        return;

    constexpr size_t PlaneBytes = WarpSize * sizeof(uint64_t);
    auto issuePage = [&](size_t taskIndex, uint64_t *page) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t pageBase = blockIndex * combinedSpan + jChunkIndex * WarpSize;

        cg::memcpy_async(warp, page, A + pageBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + WarpSize, A + pageBase + firstHalfSpan, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(warp,
                         page + 2u * WarpSize,
                         A + pageBase + secondHalfSpan,
                         cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(warp,
                         page + 3u * WarpSize,
                         A + pageBase + secondHalfSpan + firstHalfSpan,
                         cuda::aligned_size_t<16>(PlaneBytes));
    };

    auto processDirectArray = [&](uint64_t *data,
                                  uint32_t index0,
                                  uint32_t index1,
                                  uint32_t index2,
                                  uint32_t index3,
                                  uint64_t firstTwiddle,
                                  uint64_t secondTwiddle0,
                                  uint64_t secondTwiddle1) {
        const uint64_t value0 = data[index0];
        const uint64_t value1 = data[index1];
        const uint64_t value2 = data[index2];
        const uint64_t value3 = data[index3];
        StoreRadix4StagePair(grid,
                             block,
                             debugCombo,
                             data,
                             index0,
                             index1,
                             index2,
                             index3,
                             value0,
                             value1,
                             value2,
                             value3,
                             firstTwiddle,
                             secondTwiddle0,
                             secondTwiddle1);
    };

    uint32_t currentSlot = 0u;
    issuePage(warpTaskBegin, warpPage0);

    for (size_t taskIndex = warpTaskBegin; taskIndex < warpTaskEnd; ++taskIndex) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t jIndex = jChunkIndex * WarpSize + laneIndex;
        const uint32_t index0 = blockIndex * combinedSpan + jIndex;
        const uint32_t index1 = index0 + firstHalfSpan;
        const uint32_t index2 = index0 + secondHalfSpan;
        const uint32_t index3 = index2 + firstHalfSpan;
        const uint64_t firstTwiddle = firstStageTwiddles[jIndex];
        const uint64_t secondTwiddle0 = secondStageTwiddles[jIndex];
        const uint64_t secondTwiddle1 = secondStageTwiddles[jIndex + firstHalfSpan];

        processDirectArray(
            B, index0, index1, index2, index3, firstTwiddle, secondTwiddle0, secondTwiddle1);
        if constexpr (Mode == Multiway::ThreeWay) {
            processDirectArray(
                C, index0, index1, index2, index3, firstTwiddle, secondTwiddle0, secondTwiddle1);
        }

        cg::wait(warp);

        const bool hasNextTask = taskIndex + 1u < warpTaskEnd;
        uint64_t *const currentPage = currentSlot == 0u ? warpPage0 : warpPage1;
        if (hasNextTask) {
            uint64_t *const nextPage = currentSlot == 0u ? warpPage1 : warpPage0;
            issuePage(taskIndex + 1u, nextPage);
        }

        StoreRadix4StagePair(grid,
                             block,
                             debugCombo,
                             A,
                             index0,
                             index1,
                             index2,
                             index3,
                             currentPage[laneIndex],
                             currentPage[WarpSize + laneIndex],
                             currentPage[2u * WarpSize + laneIndex],
                             currentPage[3u * WarpSize + laneIndex],
                             firstTwiddle,
                             secondTwiddle0,
                             secondTwiddle1);
        currentSlot ^= 1u;
    }
}

// SevenWay keeps E/F/G in a six-plane double buffer. A/B/C/D are read directly and supply four
// independent Montgomery products while each page is moving from global to shared memory.
template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ProcessRadix2StagePipelinedSevenWay(uint64_t *sharedData,
                                    cooperative_groups::grid_group &grid,
                                    cooperative_groups::thread_block &block,
                                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                                    uint64_t *SharkRestrict A,
                                    uint64_t *SharkRestrict B,
                                    uint64_t *SharkRestrict C,
                                    uint64_t *SharkRestrict D,
                                    uint64_t *SharkRestrict E,
                                    uint64_t *SharkRestrict F,
                                    uint64_t *SharkRestrict G,
                                    const uint64_t *SharkRestrict stageTwiddles,
                                    uint32_t transformSize,
                                    uint32_t butterflySpan,
                                    uint32_t warpIndex,
                                    uint32_t numWarpsGrid)
{
    namespace cg = cooperative_groups;
    constexpr uint32_t WarpSize = 32u;
    constexpr uint32_t PlanesPerPage = 6u;
    constexpr uint32_t PipelineSlots = 2u;
    constexpr uint32_t MaxWarpsPerBlock = 512u / WarpSize;
    constexpr size_t MaxPipelineBytes =
        PipelineSlots * MaxWarpsPerBlock * PlanesPerPage * WarpSize * sizeof(uint64_t);
    static_assert(MaxPipelineBytes == 48u * 1024u);

    const auto warp = cg::tiled_partition<WarpSize>(block);
    const uint32_t laneIndex = warp.thread_rank();
    const uint32_t localWarpIndex = block.thread_index().x / WarpSize;
    const uint32_t warpsPerBlock = block.size() / WarpSize;
    const uint32_t slotStride = warpsPerBlock * PlanesPerPage * WarpSize;
    uint64_t *const warpPage0 = sharedData + localWarpIndex * PlanesPerPage * WarpSize;
    uint64_t *const warpPage1 = sharedData + slotStride + localWarpIndex * PlanesPerPage * WarpSize;

    const uint32_t halfSpan = butterflySpan >> 1u;
    const uint32_t numJChunks = halfSpan / WarpSize;
    const uint32_t numBlocks = transformSize / butterflySpan;
    const size_t totalTasks = static_cast<size_t>(numBlocks) * numJChunks;
    const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid;
    const size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
    const size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

    if (warpTaskBegin >= warpTaskEnd)
        return;

    constexpr size_t PlaneBytes = WarpSize * sizeof(uint64_t);
    auto issuePage = [&](size_t taskIndex, uint64_t *page) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t upperBase = blockIndex * butterflySpan + jChunkIndex * WarpSize;
        const uint32_t lowerBase = upperBase + halfSpan;

        cg::memcpy_async(warp, page, E + upperBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(warp, page + WarpSize, E + lowerBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + 2u * WarpSize, F + upperBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + 3u * WarpSize, F + lowerBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + 4u * WarpSize, G + upperBase, cuda::aligned_size_t<16>(PlaneBytes));
        cg::memcpy_async(
            warp, page + 5u * WarpSize, G + lowerBase, cuda::aligned_size_t<16>(PlaneBytes));
    };

    auto processDirectArray =
        [&](uint64_t *data, uint32_t upperIndex, uint32_t lowerIndex, uint64_t twiddle) {
            const uint64_t upper = data[upperIndex];
            const uint64_t lower = data[lowerIndex];
            StoreRadix2Butterfly(
                grid, block, debugCombo, data, upperIndex, lowerIndex, upper, lower, twiddle);
        };

    uint32_t currentSlot = 0u;
    issuePage(warpTaskBegin, warpPage0);

    for (size_t taskIndex = warpTaskBegin; taskIndex < warpTaskEnd; ++taskIndex) {
        const uint32_t blockIndex = static_cast<uint32_t>(taskIndex / numJChunks);
        const uint32_t jChunkIndex =
            static_cast<uint32_t>(taskIndex - static_cast<size_t>(blockIndex) * numJChunks);
        const uint32_t jIndex = jChunkIndex * WarpSize + laneIndex;
        const uint32_t upperIndex = blockIndex * butterflySpan + jIndex;
        const uint32_t lowerIndex = upperIndex + halfSpan;
        const uint64_t twiddle = stageTwiddles[jIndex];

        processDirectArray(A, upperIndex, lowerIndex, twiddle);
        processDirectArray(B, upperIndex, lowerIndex, twiddle);
        processDirectArray(C, upperIndex, lowerIndex, twiddle);
        processDirectArray(D, upperIndex, lowerIndex, twiddle);

        cg::wait(warp);

        const bool hasNextTask = taskIndex + 1u < warpTaskEnd;
        uint64_t *const currentPage = currentSlot == 0u ? warpPage0 : warpPage1;
        if (hasNextTask) {
            uint64_t *const nextPage = currentSlot == 0u ? warpPage1 : warpPage0;
            issuePage(taskIndex + 1u, nextPage);
        }

        StoreRadix2Butterfly(grid,
                             block,
                             debugCombo,
                             E,
                             upperIndex,
                             lowerIndex,
                             currentPage[laneIndex],
                             currentPage[WarpSize + laneIndex],
                             twiddle);
        StoreRadix2Butterfly(grid,
                             block,
                             debugCombo,
                             F,
                             upperIndex,
                             lowerIndex,
                             currentPage[2u * WarpSize + laneIndex],
                             currentPage[3u * WarpSize + laneIndex],
                             twiddle);
        StoreRadix2Butterfly(grid,
                             block,
                             debugCombo,
                             G,
                             upperIndex,
                             lowerIndex,
                             currentPage[4u * WarpSize + laneIndex],
                             currentPage[5u * WarpSize + laneIndex],
                             twiddle);
        currentSlot ^= 1u;
    }
}

// -----------------------------------------------------------------------------
// Unified 1-way / 7-way radix-2 NTT with warp-strided twiddles,
// early shared-memory microkernel, and Phase-2 static contiguous striping.
// Multiway::OneWay   : operates on A only
// Multiway::TwoWay   : operates on A, B
// Multiway::ThreeWay : operates on A, B, C in lockstep
// Multiway::FourWay  : operates on A, B, C, D in lockstep
// Multiway::SevenWay : operates on A, B, C, D, E, F, G in lockstep
// -----------------------------------------------------------------------------
template <class SharkFloatParams, Multiway OneTwoThree, bool Inverse, bool DecimationInFrequency = false>
static __device__ SharkForceInlineReleaseOnly void
NTTRadix2_GridStride(uint64_t *sharedData,
                     cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t *SharkRestrict globalSync1,
                     uint64_t *SharkRestrict A,
                     uint64_t *SharkRestrict B,
                     uint64_t *SharkRestrict C,
                     uint64_t *SharkRestrict D,
                     const RootTables &rootTables,
                     uint64_t *SharkRestrict E = nullptr,
                     uint64_t *SharkRestrict F = nullptr,
                     uint64_t *SharkRestrict G = nullptr)
{
    uint32_t transformSize = rootTables.N;
    uint32_t numStages = rootTables.stages;

    const uint64_t *SharkRestrict stageTwiddleTable; // flattened twiddle table

    if constexpr (!Inverse) {
        stageTwiddleTable = rootTables.stage_twiddles_fwd;
    } else {
        stageTwiddleTable = rootTables.stage_twiddles_inv;
    }

    // 11 crashes currently due to excessive shared memory usage
    constexpr auto TileSizeLog2 = 10u;
    constexpr uint32_t warpSize = 32u;

    const size_t gridSize = grid.size();
    const auto globalThreadIndex = block.thread_index().x + block.group_index().x * blockDim.x;
    const uint32_t laneIndex = static_cast<uint32_t>(globalThreadIndex & (warpSize - 1u));
    const uint32_t warpIndex = static_cast<uint32_t>(globalThreadIndex / warpSize);
    const uint32_t numWarpsGrid = static_cast<uint32_t>(gridSize / warpSize);

    if constexpr (DecimationInFrequency) {
        static_assert(!Inverse);
        constexpr bool useScalarForwardDIF = false;
        if constexpr (OneTwoThree == Multiway::SevenWay || useScalarForwardDIF) {
            ForwardDIFScalar<SharkFloatParams, OneTwoThree>(grid,
                                                            block,
                                                            debugCombo,
                                                            A,
                                                            B,
                                                            C,
                                                            D,
                                                            E,
                                                            F,
                                                            G,
                                                            stageTwiddleTable,
                                                            transformSize,
                                                            numStages,
                                                            static_cast<size_t>(globalThreadIndex),
                                                            gridSize);
            return;
        } else {
            const uint32_t smallStageCount = numStages < TileSizeLog2 ? numStages : TileSizeLog2;
            uint32_t stageIndex = numStages;
            while (stageIndex > smallStageCount + 1u) {
                ProcessRadix4DIFStagePairPipelined<SharkFloatParams, OneTwoThree>(sharedData,
                                                                                  grid,
                                                                                  block,
                                                                                  debugCombo,
                                                                                  A,
                                                                                  B,
                                                                                  C,
                                                                                  D,
                                                                                  stageTwiddleTable,
                                                                                  transformSize,
                                                                                  stageIndex,
                                                                                  warpIndex,
                                                                                  numWarpsGrid);
                grid.sync();
                stageIndex -= 2u;
            }

            if (stageIndex > smallStageCount) {
                ProcessRadix2DIFStage<SharkFloatParams, OneTwoThree>(
                    grid,
                    block,
                    debugCombo,
                    A,
                    B,
                    C,
                    D,
                    E,
                    F,
                    G,
                    stageTwiddleTable,
                    transformSize,
                    stageIndex,
                    static_cast<size_t>(globalThreadIndex),
                    gridSize);
                grid.sync();
            }

            if (smallStageCount > 0u) {
                SmallRadixPhase1DIF_SM<SharkFloatParams, OneTwoThree, TileSizeLog2>(sharedData,
                                                                                    grid,
                                                                                    block,
                                                                                    debugCombo,
                                                                                    A,
                                                                                    B,
                                                                                    C,
                                                                                    D,
                                                                                    transformSize,
                                                                                    numStages,
                                                                                    stageTwiddleTable);
            }
        }
        return;
    }

    uint32_t firstLargeStage = 0;

    // Phase 1: small-radix microkernel
    if constexpr (OneTwoThree == Multiway::OneWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::OneWay, TileSizeLog2>(sharedData,
                                                                                  grid,
                                                                                  block,
                                                                                  debugCombo,
                                                                                  A,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  transformSize,
                                                                                  numStages,
                                                                                  stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::TwoWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::TwoWay, TileSizeLog2>(sharedData,
                                                                                  grid,
                                                                                  block,
                                                                                  debugCombo,
                                                                                  A,
                                                                                  B,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  transformSize,
                                                                                  numStages,
                                                                                  stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::ThreeWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::ThreeWay, TileSizeLog2>(sharedData,
                                                                                    grid,
                                                                                    block,
                                                                                    debugCombo,
                                                                                    A,
                                                                                    B,
                                                                                    C,
                                                                                    nullptr,
                                                                                    transformSize,
                                                                                    numStages,
                                                                                    stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::FourWay) {
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::FourWay, TileSizeLog2>(sharedData,
                                                                                   grid,
                                                                                   block,
                                                                                   debugCombo,
                                                                                   A,
                                                                                   B,
                                                                                   C,
                                                                                   D,
                                                                                   transformSize,
                                                                                   numStages,
                                                                                   stageTwiddleTable);
    } else if constexpr (OneTwoThree == Multiway::SevenWay) {
        // SevenWay: interleaved ThreeWay + FourWay per tile in a single call
        firstLargeStage =
            SmallRadixPhase1_SM<SharkFloatParams, Multiway::SevenWay, TileSizeLog2>(sharedData,
                                                                                    grid,
                                                                                    block,
                                                                                    debugCombo,
                                                                                    A,
                                                                                    B,
                                                                                    C,
                                                                                    D,
                                                                                    transformSize,
                                                                                    numStages,
                                                                                    stageTwiddleTable,
                                                                                    E,
                                                                                    F,
                                                                                    G);
    }

    // =========================
    // Phase 2: static contiguous striping by warp (no atomics). TwoWay/ThreeWay pair stages into a
    // radix-4 pass; SevenWay pipelines three arrays through shared memory for each radix-2 pass.
    // =========================
    for (uint32_t stageIndex = firstLargeStage + 1; stageIndex <= numStages; ++stageIndex) {
        const uint32_t butterflySpan = 1u << stageIndex; // m
        const uint32_t halfSpan = butterflySpan >> 1;    // m/2

        // base into flattened stageTwiddleTable for this stage:
        // stage s has 'halfSpan' twiddles; they live at indices
        // [twiddleStageOffset .. twiddleStageOffset + halfSpan-1]
        // with twiddleStageOffset = 2^(s-1) - 1 = halfSpan - 1
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint64_t *SharkRestrict stageTwiddlesForStage = stageTwiddleTable + twiddleStageOffset;

        if constexpr (OneTwoThree == Multiway::TwoWay || OneTwoThree == Multiway::ThreeWay) {
            if (stageIndex < numStages) {
                ProcessRadix4StagePairPipelined<SharkFloatParams, OneTwoThree>(sharedData,
                                                                               grid,
                                                                               block,
                                                                               debugCombo,
                                                                               A,
                                                                               B,
                                                                               C,
                                                                               stageTwiddleTable,
                                                                               transformSize,
                                                                               stageIndex,
                                                                               warpIndex,
                                                                               numWarpsGrid);
                grid.sync();
                ++stageIndex;
                continue;
            }
        }

        if constexpr (OneTwoThree == Multiway::SevenWay) {
            ProcessRadix2StagePipelinedSevenWay(sharedData,
                                                grid,
                                                block,
                                                debugCombo,
                                                A,
                                                B,
                                                C,
                                                D,
                                                E,
                                                F,
                                                G,
                                                stageTwiddlesForStage,
                                                transformSize,
                                                butterflySpan,
                                                warpIndex,
                                                numWarpsGrid);
            grid.sync();
            continue;
        }

        const uint32_t numJChunks = (halfSpan + (warpSize - 1u)) / warpSize; // ceil(halfSpan/warpSize)

        const uint32_t numBlocksPerStage = transformSize / butterflySpan;
        const size_t totalTasks = static_cast<size_t>(numBlocksPerStage) *
                                  static_cast<size_t>(numJChunks); // == N/64, invariant in s

        // -------- Static contiguous partition: each warp gets one equal-sized range --------
        const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid; // ceil
        size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
        size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

        if (warpTaskBegin < warpTaskEnd) {
            // Decode the first ticket in our contiguous range
            uint32_t blockIndex = static_cast<uint32_t>(warpTaskBegin / numJChunks);
            uint32_t jChunkIndex =
                static_cast<uint32_t>(warpTaskBegin - static_cast<size_t>(blockIndex) * numJChunks);

            uint32_t blockDataBaseIndex = blockIndex * butterflySpan;
            size_t tasksRemaining = warpTaskEnd - warpTaskBegin;

            if constexpr (OneTwoThree != Multiway::SevenWay) {
                constexpr int microTileWidth = (OneTwoThree == Multiway::OneWay ? 4 : 2);
                while (tasksRemaining) {
                    ProcessTile<SharkFloatParams, OneTwoThree, microTileWidth>(grid,
                                                                               block,
                                                                               debugCombo,
                                                                               A,
                                                                               B,
                                                                               C,
                                                                               D,
                                                                               stageTwiddlesForStage,
                                                                               halfSpan,
                                                                               warpSize,
                                                                               numJChunks,
                                                                               laneIndex,
                                                                               butterflySpan,
                                                                               blockIndex,
                                                                               jChunkIndex,
                                                                               tasksRemaining,
                                                                               blockDataBaseIndex);
                }
            }
        }

        // One grid-wide barrier per stage (still required for correctness)
        grid.sync();
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ForwardDIFLargeStages(uint64_t *sharedData,
                      cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t *SharkRestrict A,
                      uint64_t *SharkRestrict B,
                      uint64_t *SharkRestrict C,
                      uint64_t *SharkRestrict D,
                      const RootTables &rootTables,
                      uint32_t highestStageIndex)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::FourWay);
    constexpr uint32_t TileSizeLog2 = 10u;
    constexpr uint32_t WarpSize = 32u;
    const uint32_t transformSize = rootTables.N;
    const uint32_t numStages = rootTables.stages;
    const uint64_t *SharkRestrict stageTwiddleTable = rootTables.stage_twiddles_fwd;
    const size_t gridSize = grid.size();
    const size_t globalThreadIndex = block.thread_index().x + block.group_index().x * blockDim.x;
    const uint32_t warpIndex = static_cast<uint32_t>(globalThreadIndex / WarpSize);
    const uint32_t numWarpsGrid = static_cast<uint32_t>(gridSize / WarpSize);
    const uint32_t smallStageCount = numStages < TileSizeLog2 ? numStages : TileSizeLog2;

    uint32_t stageIndex = highestStageIndex;
    while (stageIndex > smallStageCount + 1u) {
        ProcessRadix4DIFStagePairPipelined<SharkFloatParams, Mode>(sharedData,
                                                                   grid,
                                                                   block,
                                                                   debugCombo,
                                                                   A,
                                                                   B,
                                                                   C,
                                                                   D,
                                                                   stageTwiddleTable,
                                                                   transformSize,
                                                                   stageIndex,
                                                                   warpIndex,
                                                                   numWarpsGrid);
        grid.sync();
        stageIndex -= 2u;
    }

    if (stageIndex > smallStageCount) {
        ProcessRadix2DIFStage<SharkFloatParams, Mode>(grid,
                                                      block,
                                                      debugCombo,
                                                      A,
                                                      B,
                                                      C,
                                                      D,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      stageTwiddleTable,
                                                      transformSize,
                                                      stageIndex,
                                                      globalThreadIndex,
                                                      gridSize);
        grid.sync();
    }
}

template <class SharkFloatParams, Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
InverseDITLargeStages(uint64_t *sharedData,
                      cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t *SharkRestrict A,
                      uint64_t *SharkRestrict B,
                      uint64_t *SharkRestrict C,
                      uint64_t *SharkRestrict D,
                      const RootTables &rootTables)
{
    static_assert(Mode == Multiway::TwoWay || Mode == Multiway::FourWay);
    constexpr uint32_t TileSizeLog2 = 10u;
    constexpr uint32_t WarpSize = 32u;
    const uint32_t transformSize = rootTables.N;
    const uint32_t numStages = rootTables.stages;
    const uint64_t *SharkRestrict stageTwiddleTable = rootTables.stage_twiddles_inv;
    const size_t gridSize = grid.size();
    const size_t globalThreadIndex = block.thread_index().x + block.group_index().x * blockDim.x;
    const uint32_t laneIndex = static_cast<uint32_t>(globalThreadIndex & (WarpSize - 1u));
    const uint32_t warpIndex = static_cast<uint32_t>(globalThreadIndex / WarpSize);
    const uint32_t numWarpsGrid = static_cast<uint32_t>(gridSize / WarpSize);
    const uint32_t firstLargeStage = numStages < TileSizeLog2 ? numStages : TileSizeLog2;

    for (uint32_t stageIndex = firstLargeStage + 1u; stageIndex <= numStages; ++stageIndex) {
        const uint32_t butterflySpan = 1u << stageIndex;
        const uint32_t halfSpan = butterflySpan >> 1u;
        const uint32_t twiddleStageOffset = halfSpan - 1u;
        const uint64_t *SharkRestrict stageTwiddlesForStage = stageTwiddleTable + twiddleStageOffset;

        if constexpr (Mode == Multiway::TwoWay) {
            if (stageIndex < numStages) {
                ProcessRadix4StagePairPipelined<SharkFloatParams, Mode>(sharedData,
                                                                        grid,
                                                                        block,
                                                                        debugCombo,
                                                                        A,
                                                                        B,
                                                                        C,
                                                                        stageTwiddleTable,
                                                                        transformSize,
                                                                        stageIndex,
                                                                        warpIndex,
                                                                        numWarpsGrid);
                grid.sync();
                ++stageIndex;
                continue;
            }
        }

        const uint32_t numJChunks = (halfSpan + (WarpSize - 1u)) / WarpSize;
        const uint32_t numBlocksPerStage = transformSize / butterflySpan;
        const size_t totalTasks = static_cast<size_t>(numBlocksPerStage) * numJChunks;
        const size_t tasksPerWarp = (totalTasks + numWarpsGrid - 1ull) / numWarpsGrid;
        size_t warpTaskBegin = static_cast<size_t>(warpIndex) * tasksPerWarp;
        size_t warpTaskEnd = min(totalTasks, warpTaskBegin + tasksPerWarp);

        if (warpTaskBegin < warpTaskEnd) {
            uint32_t blockIndex = static_cast<uint32_t>(warpTaskBegin / numJChunks);
            uint32_t jChunkIndex =
                static_cast<uint32_t>(warpTaskBegin - static_cast<size_t>(blockIndex) * numJChunks);
            uint32_t blockDataBaseIndex = blockIndex * butterflySpan;
            size_t tasksRemaining = warpTaskEnd - warpTaskBegin;
            while (tasksRemaining) {
                ProcessTile<SharkFloatParams, Mode, 2>(grid,
                                                       block,
                                                       debugCombo,
                                                       A,
                                                       B,
                                                       C,
                                                       D,
                                                       stageTwiddlesForStage,
                                                       halfSpan,
                                                       WarpSize,
                                                       numJChunks,
                                                       laneIndex,
                                                       butterflySpan,
                                                       blockIndex,
                                                       jChunkIndex,
                                                       tasksRemaining,
                                                       blockDataBaseIndex);
            }
        }
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
// N-channel unpack: processes NumChannels products in one call with one trailing grid.sync().
template <class SharkFloatParams, int NumChannels>
static __device__ SharkForceInlineReleaseOnly void
UnpackPrimeToFinal128_NWay(cooperative_groups::grid_group &grid,
                           cooperative_groups::thread_block &block,
                           const uint64_t *SharkRestrict const *inputs,
                           uint64_t *SharkRestrict const *outputs,
                           uint32_t Ddigits)
{
    using namespace SharkNTT;

    const size_t gsize = grid.size();
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

    auto ceil_div_u64 = [](uint64_t a, uint32_t b) -> uint64_t {
        return (a + (uint64_t)b - 1ull) / (uint64_t)b;
    };

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
    const int Imax = min(SharkFloatParams::NTTPlan.N, 2 * SharkFloatParams::NTTPlan.L - 1);

    for (size_t j = grank; j < Ddigits; j += gsize) {
        uint64_t accum_lo[NumChannels];
        uint64_t accum_hi[NumChannels];
#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            accum_lo[ch] = 0ull;
            accum_hi[ch] = 0ull;
        }

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
                const int r = (int)(sBits & 31);
                const uint64_t lsh = (r ? (64 - r) : 64);

#pragma unroll
                for (int ch = 0; ch < NumChannels; ++ch) {
                    const uint64_t v = inputs[ch][i];
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
                            add32_local(accum_lo[ch], accum_hi[ch], dt);
                        else
                            sub32_local(accum_lo[ch], accum_hi[ch], dt);
                    }
                }
            }
        }

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            outputs[ch][2 * j + 0] = accum_lo[ch];
            outputs[ch][2 * j + 1] = accum_hi[ch];
        }
    }

    grid.sync();
}

// Backward-compatible 3-way wrapper
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
// When EnableNewtonRaphson, also NTTs dzdcR/dzdcI and replicates W0-W3 product pairs.
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
    // six orbit outputs (Montgomery domain, length SharkFloatParams::NTTPlan.N)
    uint64_t *SharkRestrict tempDigitsXX1,
    uint64_t *SharkRestrict tempDigitsXX2,
    uint64_t *SharkRestrict tempDigitsYY1,
    uint64_t *SharkRestrict tempDigitsYY2,
    uint64_t *SharkRestrict tempDigitsXY1,
    uint64_t *SharkRestrict tempDigitsXY2,
    // NR inputs (nullptr when NR disabled; never dereferenced thanks to if constexpr)
    const HpSharkFloat<SharkFloatParams> *inDzdcReal,
    const HpSharkFloat<SharkFloatParams> *inDzdcImag,
    // NR product pair buffers (8 total, nullptr when NR disabled)
    uint64_t *SharkRestrict tempDigitsW0_1,
    uint64_t *SharkRestrict tempDigitsW0_2,
    uint64_t *SharkRestrict tempDigitsW1_1,
    uint64_t *SharkRestrict tempDigitsW1_2,
    uint64_t *SharkRestrict tempDigitsW2_1,
    uint64_t *SharkRestrict tempDigitsW2_2,
    uint64_t *SharkRestrict tempDigitsW3_1,
    uint64_t *SharkRestrict tempDigitsW3_2)
{
    const uint32_t N = static_cast<uint32_t>(SharkFloatParams::NTTPlan.N);
    const uint32_t L = static_cast<uint32_t>(SharkFloatParams::NTTPlan.L);
    const size_t gsize = grid.size();
    const auto grank = block.thread_index().x + block.group_index().x * blockDim.x;

    const uint64_t zero_m = ToMontgomeryConstexpr(0ull);

    // Bit-reversal shift for scatter-write (eliminates standalone BitReverse pass)
    const uint32_t brShift = 32u - static_cast<uint32_t>(SharkFloatParams::NTTPlan.stages);

    // -------------------- Phase A: pack+twist with scatter-write to bit-reversed positions ----------
    for (size_t i = grank; i < (size_t)N; i += gsize) {
        const uint32_t rev_i = __brev(static_cast<uint32_t>(i)) >> brShift;

        if (i < L) {
            const uint64_t coeff = ReadBitsSimple(
                inA, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmod = coeff % MagicPrime;
            const uint64_t xm = ToMontgomery(grid, block, debugGlobalState, cmod);
            const uint64_t psik = roots.psi_pows[i];
            tempDigitsXX1[rev_i] = MontgomeryMul(grid, block, debugGlobalState, xm, psik);
        } else {
            tempDigitsXX1[rev_i] = zero_m;
        }

        if (i < L) {
            const uint64_t coeffB = ReadBitsSimple(
                inB, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
            const uint64_t cmodB = coeffB % MagicPrime;
            const uint64_t xmB = ToMontgomery(grid, block, debugGlobalState, cmodB);
            const uint64_t psiB = roots.psi_pows[i];
            tempDigitsYY1[rev_i] = MontgomeryMul(grid, block, debugGlobalState, xmB, psiB);
        } else {
            tempDigitsYY1[rev_i] = zero_m;
        }

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint64_t psi_i = roots.psi_pows[i];

            if (i < L) {
                const uint64_t coeffDR = ReadBitsSimple(
                    *inDzdcReal, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
                const uint64_t cmodDR = coeffDR % MagicPrime;
                const uint64_t xmDR = ToMontgomery(grid, block, debugGlobalState, cmodDR);
                tempDigitsW0_1[rev_i] = MontgomeryMul(grid, block, debugGlobalState, xmDR, psi_i);
            } else {
                tempDigitsW0_1[rev_i] = zero_m;
            }

            if (i < L) {
                const uint64_t coeffDI = ReadBitsSimple(
                    *inDzdcImag, (int64_t)i * SharkFloatParams::NTTPlan.b, SharkFloatParams::NTTPlan.b);
                const uint64_t cmodDI = coeffDI % MagicPrime;
                const uint64_t xmDI = ToMontgomery(grid, block, debugGlobalState, cmodDI);
                tempDigitsW1_1[rev_i] = MontgomeryMul(grid, block, debugGlobalState, xmDI, psi_i);
            } else {
                tempDigitsW1_1[rev_i] = zero_m;
            }
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

        // NR: dzdcR and dzdcI after packing (before forward NTT butterfly)
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0W0, uint64_t>(
                debugStates, grid, block, tempDigitsW0_1, SharkFloatParams::NTTPlan.N);
            // Z1W0 = zR (same as Z0XX); Z0W2 = dzdcR (same as Z0W0)
            debugStates[static_cast<int>(DebugStatePurpose::Z1W0)].Reset(
                DebugStatePurpose::Z1W0, debugStates[static_cast<int>(DebugStatePurpose::Z0XX)]);
            debugStates[static_cast<int>(DebugStatePurpose::Z0W2)].Reset(
                DebugStatePurpose::Z0W2, debugStates[static_cast<int>(DebugStatePurpose::Z0W0)]);
            debugStates[static_cast<int>(DebugStatePurpose::Z1W2)].Reset(
                DebugStatePurpose::Z1W2, debugStates[static_cast<int>(DebugStatePurpose::Z0YY)]);

            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z0W1, uint64_t>(
                debugStates, grid, block, tempDigitsW1_1, SharkFloatParams::NTTPlan.N);
            // Z1W1 = zI (same as Z0YY); Z0W3 = dzdcI (same as Z0W1)
            debugStates[static_cast<int>(DebugStatePurpose::Z1W1)].Reset(
                DebugStatePurpose::Z1W1, debugStates[static_cast<int>(DebugStatePurpose::Z0YY)]);
            debugStates[static_cast<int>(DebugStatePurpose::Z0W3)].Reset(
                DebugStatePurpose::Z0W3, debugStates[static_cast<int>(DebugStatePurpose::Z0W1)]);
            debugStates[static_cast<int>(DebugStatePurpose::Z1W3)].Reset(
                DebugStatePurpose::Z1W3, debugStates[static_cast<int>(DebugStatePurpose::Z0XX)]);
        }

        grid.sync();
    } else {
        grid.sync();
    }

    // Forward bit-reverse merged into pack phase (scatter-write to rev_i).
    // Data is already in bit-reversed order — proceed directly to forward NTT.

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        NTTRadix2_GridStride<SharkFloatParams, Multiway::FourWay, false>(shared_data,
                                                                         grid,
                                                                         block,
                                                                         debugGlobalState,
                                                                         carryPropagationSync,
                                                                         tempDigitsXX1,
                                                                         tempDigitsYY1,
                                                                         tempDigitsW0_1,
                                                                         tempDigitsW1_1,
                                                                         roots);
    } else {
        NTTRadix2_GridStride<SharkFloatParams, Multiway::TwoWay, false>(shared_data,
                                                                        grid,
                                                                        block,
                                                                        debugGlobalState,
                                                                        carryPropagationSync,
                                                                        tempDigitsXX1,
                                                                        tempDigitsYY1,
                                                                        nullptr,
                                                                        nullptr,
                                                                        roots);
    }

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

        // NR: dzdcR and dzdcI after forward NTT butterfly
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2W0, uint64_t>(
                debugStates, grid, block, tempDigitsW0_1, SharkFloatParams::NTTPlan.N);
            // Z3W0 = zR after NTT (second operand of W0=dzdcR*zR)
            debugStates[static_cast<int>(DebugStatePurpose::Z3W0)].Reset(
                DebugStatePurpose::Z3W0, debugStates[static_cast<int>(DebugStatePurpose::Z2XX)]);
            // Z2W2 = dzdcR after NTT (same first operand as W0)
            debugStates[static_cast<int>(DebugStatePurpose::Z2W2)].Reset(
                DebugStatePurpose::Z2W2, debugStates[static_cast<int>(DebugStatePurpose::Z2W0)]);
            // Z3W2 = zI after NTT (second operand of W2=dzdcR*zI)
            debugStates[static_cast<int>(DebugStatePurpose::Z3W2)].Reset(
                DebugStatePurpose::Z3W2, debugStates[static_cast<int>(DebugStatePurpose::Z2YY)]);

            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2W1, uint64_t>(
                debugStates, grid, block, tempDigitsW1_1, SharkFloatParams::NTTPlan.N);
            // Z3W1 = zI after NTT (second operand of W1=dzdcI*zI)
            debugStates[static_cast<int>(DebugStatePurpose::Z3W1)].Reset(
                DebugStatePurpose::Z3W1, debugStates[static_cast<int>(DebugStatePurpose::Z2YY)]);
            // Z2W3 = dzdcI after NTT (same first operand as W1)
            debugStates[static_cast<int>(DebugStatePurpose::Z2W3)].Reset(
                DebugStatePurpose::Z2W3, debugStates[static_cast<int>(DebugStatePurpose::Z2W1)]);
            // Z3W3 = zR after NTT (second operand of W3=dzdcI*zR)
            debugStates[static_cast<int>(DebugStatePurpose::Z3W3)].Reset(
                DebugStatePurpose::Z3W3, debugStates[static_cast<int>(DebugStatePurpose::Z2XX)]);
        }

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

        // NR: replicate spectra into W0-W3 product pairs.
        // W0_1 currently holds NTT(dzdcR), W1_1 holds NTT(dzdcI) — read before overwriting.
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint64_t vDR = tempDigitsW0_1[i]; // NTT(dzdcR)
            const uint64_t vDI = tempDigitsW1_1[i]; // NTT(dzdcI)

            // W0: zR * dzdcR
            tempDigitsW0_1[i] = vA;  // NTT(zR)
            tempDigitsW0_2[i] = vDR; // NTT(dzdcR)

            // W1: zI * dzdcI
            tempDigitsW1_1[i] = vB;  // NTT(zI)
            tempDigitsW1_2[i] = vDI; // NTT(dzdcI)

            // W2: zI * dzdcR
            tempDigitsW2_1[i] = vB;  // NTT(zI)
            tempDigitsW2_2[i] = vDR; // NTT(dzdcR)

            // W3: zR * dzdcI
            tempDigitsW3_1[i] = vA;  // NTT(zR)
            tempDigitsW3_2[i] = vDI; // NTT(dzdcI)
        }
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

// Runtime-purpose overload for N-channel normalize
template <class SharkFloatParams, typename ArrayType>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState(DebugState<SharkFloatParams> *SharkRestrict debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugStatePurpose purpose,
                       const ArrayType *arrayToChecksum,
                       size_t arraySize)
{
    const auto CurPurpose = static_cast<int32_t>(purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No;
    constexpr auto CallIndex = 0;

    debugStates[CurPurpose].Reset(
        UseConvolutionHere, grid, block, arrayToChecksum, arraySize, purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugValue(DebugState<SharkFloatParams> *SharkRestrict debugStates,
                       cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugStatePurpose purpose,
                       const HpSharkFloat<SharkFloatParams> &value)
{
    const auto curPurpose = static_cast<int32_t>(purpose);
    constexpr auto recursionDepth = 0;
    constexpr auto useConvolution = UseConvolution::No;
    constexpr auto callIndex = 0;

    debugStates[curPurpose].Reset(
        useConvolution, grid, block, value, purpose, recursionDepth, callIndex);
}

// Look for CalculateNTTFrameSize
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArraysForMultiply.
static_assert(HpShark::AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets()                                                                     \
    const int threadIdxGlobal = block.group_index().x * block.dim_threads().x + block.thread_index().x; \
    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;                                            \
    constexpr int TestMultiplier = 1;                                                                   \
    constexpr auto DebugGlobals_offset = HpShark::AdditionalGlobalSyncSpace;                            \
    constexpr auto DebugChecksum_offset =                                                               \
        DebugGlobals_offset + HpShark::AdditionalGlobalDebugPerThread;                                  \
    constexpr auto GlobalsDoneOffset = DebugChecksum_offset + HpShark::AdditionalGlobalChecksumSpace;   \
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
    /* NTT path: orbit Z0/Z2 are the only macro-allocated buffers.                                      \
       NR W and all other buffers are in the runtime layout. */                                         \
    constexpr auto NTT_OrbitEnd = Z2_offsetYY + 4 * NewN * TestMultiplier +                             \
                                  CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier);

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
RunNTT_3Way_Multiply(uint64_t *shared_data,
                     HpSharkFloat<SharkFloatParams> *outXX,
                     HpSharkFloat<SharkFloatParams> *outYY,
                     HpSharkFloat<SharkFloatParams> *outXY,
                     const HpSharkFloat<SharkFloatParams> &inA,
                     const HpSharkFloat<SharkFloatParams> &inB,
                     // NR inputs (nullptr when NR disabled; never dereferenced)
                     const HpSharkFloat<SharkFloatParams> *inDzdcReal,
                     const HpSharkFloat<SharkFloatParams> *inDzdcImag,
                     HpSharkFloat<SharkFloatParams> *outW0,
                     HpSharkFloat<SharkFloatParams> *outW1,
                     HpSharkFloat<SharkFloatParams> *outW2,
                     HpSharkFloat<SharkFloatParams> *outW3,
                     // existing params continue:
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
                     uint32_t Ddigits,
                     // NR temp buffers (nullptr when NR disabled)
                     uint64_t *tempDigitsW0_1,
                     uint64_t *tempDigitsW0_2,
                     uint64_t *tempDigitsW1_1,
                     uint64_t *tempDigitsW1_2,
                     uint64_t *tempDigitsW2_1,
                     uint64_t *tempDigitsW2_2,
                     uint64_t *tempDigitsW3_1,
                     uint64_t *tempDigitsW3_2,
                     uint64_t *Final128_W0,
                     uint64_t *Final128_W1,
                     uint64_t *Final128_W2,
                     uint64_t *Final128_W3)
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
                                                            tempDigitsXY2,
                                                            inDzdcReal,
                                                            inDzdcImag,
                                                            tempDigitsW0_1,
                                                            tempDigitsW0_2,
                                                            tempDigitsW1_1,
                                                            tempDigitsW1_2,
                                                            tempDigitsW2_1,
                                                            tempDigitsW2_2,
                                                            tempDigitsW3_1,
                                                            tempDigitsW3_2);

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

        // NR pointwise multiplies — same loop, no new sync
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint64_t aW0 = tempDigitsW0_1[i];
            const uint64_t bW0 = tempDigitsW0_2[i];
            tempDigitsW0_1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aW0, bW0);

            const uint64_t aW1 = tempDigitsW1_1[i];
            const uint64_t bW1 = tempDigitsW1_2[i];
            tempDigitsW1_1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aW1, bW1);

            const uint64_t aW2 = tempDigitsW2_1[i];
            const uint64_t bW2 = tempDigitsW2_2[i];
            tempDigitsW2_1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aW2, bW2);

            const uint64_t aW3 = tempDigitsW3_1[i];
            const uint64_t bW3 = tempDigitsW3_2[i];
            tempDigitsW3_1[i] = SharkNTT::MontgomeryMul(grid, block, debugGlobalState, aW3, bW3);
        }
    }

    grid.sync();

    // 5) Inverse bit-reverse + NTT (in place)
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // Fused SevenWay: all 7 transforms (XX,YY,XY + W0-W3) share grid.sync() barriers
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::SevenWay>(
            grid,
            block,
            tempDigitsXX1,
            tempDigitsYY1,
            tempDigitsXY1,
            tempDigitsW0_1,
            (uint32_t)SharkFloatParams::NTTPlan.N,
            (uint32_t)SharkFloatParams::NTTPlan.stages,
            tempDigitsW1_1,
            tempDigitsW2_1,
            tempDigitsW3_1);
    } else {
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::ThreeWay>(
            grid,
            block,
            tempDigitsXX1,
            tempDigitsYY1,
            tempDigitsXY1,
            nullptr,
            (uint32_t)SharkFloatParams::NTTPlan.N,
            (uint32_t)SharkFloatParams::NTTPlan.stages);
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm3, uint64_t>(
            debugStates, grid, block, tempDigitsXY1, SharkFloatParams::NTTPlan.N);

        // NR: W0-W3 after inverse bit-reverse
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW0, uint64_t>(
                debugStates, grid, block, tempDigitsW0_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW1, uint64_t>(
                debugStates, grid, block, tempDigitsW1_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW2, uint64_t>(
                debugStates, grid, block, tempDigitsW2_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW3, uint64_t>(
                debugStates, grid, block, tempDigitsW3_1, SharkFloatParams::NTTPlan.N);
        }

        grid.sync();
    } else {
        grid.sync();
    }

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // Fused SevenWay inverse NTT: all 7 transforms share Phase 2 grid.sync() barriers
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::SevenWay, true>(
            shared_data,
            grid,
            block,
            debugGlobalState,
            CarryPropagationSync,
            tempDigitsXX1,
            tempDigitsYY1,
            tempDigitsXY1,
            tempDigitsW0_1,
            roots,
            tempDigitsW1_1,
            tempDigitsW2_1,
            tempDigitsW3_1);
    } else {
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::ThreeWay, true>(
            shared_data,
            grid,
            block,
            debugGlobalState,
            CarryPropagationSync,
            tempDigitsXX1,
            tempDigitsYY1,
            tempDigitsXY1,
            nullptr,
            roots);
    }

    // After inverse NTT: the last Phase 2 stage already ended with grid.sync(),
    // so all NTT output writes are visible. No additional sync needed before untwist.

    UntwistScaleFromMont_3Way_GridStride<SharkFloatParams>(grid,
                                                           block,
                                                           debugGlobalState,
                                                           roots,
                                                           /* XX1 */ tempDigitsXX1,
                                                           /* YY1 */ tempDigitsYY1,
                                                           /* XY1 */ tempDigitsXY1);

    // NR untwist (same sync point — grid-strided, no internal sync)
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        using namespace SharkNTT;
        const uint64_t Ninvm = roots.Ninvm_mont;
        for (size_t i = grank; i < N; i += gsize) {
            const uint64_t psi_inv_i = roots.psi_inv_pows[i];

            auto untwist_one = [&](uint64_t *buf) {
                uint64_t v = MontgomeryMul(grid, block, debugGlobalState, buf[i], psi_inv_i);
                v = MontgomeryMul(grid, block, debugGlobalState, v, Ninvm);
                buf[i] = FromMontgomery(grid, block, debugGlobalState, v);
            };

            untwist_one(tempDigitsW0_1);
            untwist_one(tempDigitsW1_1);
            untwist_one(tempDigitsW2_1);
            untwist_one(tempDigitsW3_1);
        }
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm4, uint64_t>(
            debugStates, grid, block, tempDigitsXX1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm5, uint64_t>(
            debugStates, grid, block, tempDigitsYY1, SharkFloatParams::NTTPlan.N);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm6, uint64_t>(
            debugStates, grid, block, tempDigitsXY1, SharkFloatParams::NTTPlan.N);

        // NR: W0-W3 after untwist
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW0b, uint64_t>(
                debugStates, grid, block, tempDigitsW0_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW1b, uint64_t>(
                debugStates, grid, block, tempDigitsW1_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW2b, uint64_t>(
                debugStates, grid, block, tempDigitsW2_1, SharkFloatParams::NTTPlan.N);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_PermW3b, uint64_t>(
                debugStates, grid, block, tempDigitsW3_1, SharkFloatParams::NTTPlan.N);
        }

        grid.sync();
    } else {
        grid.sync();
    }

    // The helper does a final grid.sync() internally.
    // At this point, tempDigitsXX1/YY1/XY1 are back in the normal domain (not Montgomery).

    // 8) Unpack -> Final128
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // 7-way: all products in one call, one trailing grid.sync()
        const uint64_t *unpackInputs[7] = {tempDigitsXX1,
                                           tempDigitsYY1,
                                           tempDigitsXY1,
                                           tempDigitsW0_1,
                                           tempDigitsW1_1,
                                           tempDigitsW2_1,
                                           tempDigitsW3_1};
        uint64_t *unpackOutputs[7] = {
            Final128_XX, Final128_YY, Final128_XY, Final128_W0, Final128_W1, Final128_W2, Final128_W3};
        SharkNTT::UnpackPrimeToFinal128_NWay<SharkFloatParams, 7>(
            grid, block, unpackInputs, unpackOutputs, Ddigits);
    } else {
        SharkNTT::UnpackPrimeToFinal128_3Way<SharkFloatParams>(grid,
                                                               block,
                                                               tempDigitsXX1,
                                                               tempDigitsYY1,
                                                               tempDigitsXY1,
                                                               Final128_XX,
                                                               Final128_YY,
                                                               Final128_XY,
                                                               Ddigits);
    }

    grid.sync(); // subsequent phases depend on Final128_* fully written

    // Post-unpack checksums (before normalize)
    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackXX, uint64_t>(
            debugStates, grid, block, Final128_XX, 2 * Ddigits);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackYY, uint64_t>(
            debugStates, grid, block, Final128_YY, 2 * Ddigits);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackXY, uint64_t>(
            debugStates, grid, block, Final128_XY, 2 * Ddigits);

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackW0, uint64_t>(
                debugStates, grid, block, Final128_W0, 2 * Ddigits);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackW1, uint64_t>(
                debugStates, grid, block, Final128_W1, 2 * Ddigits);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackW2, uint64_t>(
                debugStates, grid, block, Final128_W2, 2 * Ddigits);
            StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::UnpackW3, uint64_t>(
                debugStates, grid, block, Final128_W3, 2 * Ddigits);
        }

        grid.sync();
    }

    outXX->SetNegative(false);
    outYY->SetNegative(false);

    const auto OutXYIsNegative = (inA.GetNegative() ^ inB.GetNegative());
    outXY->SetNegative(OutXYIsNegative);

    const auto addFactorOfTwoXX = 0;
    const auto addFactorOfTwoYY = 0;
    const auto addFactorOfTwoXY = 1;

    // --- Workspaces ---
    // dynamic shared mem: need >= 6 * blockDim.x * sizeof(uint64_t)
    // scratch result digits (2*SharkFloatParams::GlobalNumUint32 per channel, uint64_t each; low 32 bits
    // used)
    uint64_t *resultXX = tempDigitsXX1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */
    uint64_t *resultYY = tempDigitsYY1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */
    uint64_t *resultXY = tempDigitsXY1; /* device buffer length 2*SharkFloatParams::GlobalNumUint32 */

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        // W product signs: XOR of input signs
        outW0->SetNegative(inDzdcReal->GetNegative() ^ inA.GetNegative()); // dzdcR × zR
        outW1->SetNegative(inDzdcImag->GetNegative() ^ inB.GetNegative()); // dzdcI × zI
        outW2->SetNegative(inDzdcReal->GetNegative() ^ inB.GetNegative()); // dzdcR × zI
        outW3->SetNegative(inDzdcImag->GetNegative() ^ inA.GetNegative()); // dzdcI × zR

        // All 7 products in one fused 7-way normalize call.
        // This fixes the aliased-buffer bug in the old 4x3-way approach.
        uint64_t *resultW0 = tempDigitsW0_1;
        uint64_t *resultW1 = tempDigitsW1_1;
        uint64_t *resultW2 = tempDigitsW2_1;
        uint64_t *resultW3 = tempDigitsW3_1;

        HpSharkFloat<SharkFloatParams> *outs7[7] = {outXX, outYY, outXY, outW0, outW1, outW2, outW3};
        uint64_t *final128s7[7] = {
            Final128_XX, Final128_YY, Final128_XY, Final128_W0, Final128_W1, Final128_W2, Final128_W3};
        uint64_t *results7[7] = {resultXX, resultYY, resultXY, resultW0, resultW1, resultW2, resultW3};
        int32_t exps7[7] = {
            inA.Exponent + inA.Exponent,         // XX
            inB.Exponent + inB.Exponent,         // YY
            inA.Exponent + inB.Exponent,         // XY
            inDzdcReal->Exponent + inA.Exponent, // W0 = dzdcR × zR
            inDzdcImag->Exponent + inB.Exponent, // W1 = dzdcI × zI
            inDzdcReal->Exponent + inB.Exponent, // W2 = dzdcR × zI
            inDzdcImag->Exponent + inA.Exponent, // W3 = dzdcI × zR
        };
        int32_t addTwos7[7] = {0, 0, 1, 1, 1, 1, 1};
        DebugStatePurpose purposes7[7] = {DebugStatePurpose::Final128XX,
                                          DebugStatePurpose::Final128YY,
                                          DebugStatePurpose::Final128XY,
                                          DebugStatePurpose::Final128W0,
                                          DebugStatePurpose::Final128W1,
                                          DebugStatePurpose::Final128W2,
                                          DebugStatePurpose::Final128W3};

        SharkNTT::Normalize_GridStride_NWay<SharkFloatParams, 7>(grid,
                                                                 block,
                                                                 debugGlobalState,
                                                                 debugStates,
                                                                 outs7,
                                                                 exps7,
                                                                 final128s7,
                                                                 Ddigits,
                                                                 addTwos7,
                                                                 CarryPropagationBuffer2,
                                                                 CarryPropagationBuffer,
                                                                 CarryPropagationSync,
                                                                 CarryPropagationSync2,
                                                                 results7,
                                                                 purposes7,
                                                                 shared_data);
    } else {
        // Non-NR: existing 3-way call (unchanged)
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
        SharkNTT::Normalize_GridStride_3WayV2<SharkFloatParams>(shared_data,
                                                                grid,
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
}

template <class SharkFloatParams, bool InitializeDebugStates = true>
static __device__ SharkForceInlineReleaseOnly void
MultiplyHelperNTTV2Separates(const SharkNTT::RootTables &roots,
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict A,
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict B,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutXX,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutXY,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutYY,
                             // NR derivative inputs / outputs (nullptr when NR disabled)
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict DzdcReal,
                             const HpSharkFloat<SharkFloatParams> *SharkRestrict DzdcImag,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutW0,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutW1,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutW2,
                             HpSharkFloat<SharkFloatParams> *SharkRestrict OutW3,
                             cg::grid_group &grid,
                             cg::thread_block &block,
                             uint64_t *SharkRestrict tempProducts)
{

    extern __shared__ __align__(16) uint64_t shared_data[];

    DefineTempProductsOffsets();

    // Verify scratch offsets fit within allocated memory
    // NR W buffers are now in the runtime layout (not macro offsets), so NTT_OrbitEnd covers both.
    constexpr auto TotalAlloc =
        HpShark::AdditionalUInt64Global + HpShark::CalculateNTTFrameSize<SharkFloatParams>();
    static_assert(NTT_OrbitEnd <= TotalAlloc, "Scratch offsets exceed total allocation");

    // TODO: indexes
    auto *SharkRestrict debugGlobalState =
        reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempProducts[DebugGlobals_offset]);
    auto *SharkRestrict debugStates =
        reinterpret_cast<DebugState<SharkFloatParams> *>(&tempProducts[DebugChecksum_offset]);

    if constexpr (HpShark::DebugGlobalState) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugGlobalState[CurBlock * block.dim_threads().x + CurThread].DebugMultiplyErase();
    }

    if constexpr (HpShark::DebugChecksums) {
        if constexpr (InitializeDebugStates) {
            EraseAllDebugStates(debugStates, grid, block);
        }

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

    // Each NTT buffer must be large enough for both NTT transforms (NTTPlan.N)
    // AND Final128 reuse (2*Ddigits), since "2" buffers are reused as Final128 after pointwise multiply.
    const uint32_t twoDdigits = static_cast<uint32_t>(2 * Ddigits);
    const uint32_t nttN = SharkFloatParams::NTTPlan.N;
    const auto LameHackBufferSizeWhatShouldItBe = (twoDdigits > nttN) ? twoDdigits : nttN;

    // ---- Single allocation for entire core path ----
    uint64_t *buffer = &tempProducts[GlobalsDoneOffset];

    // Slice buffer into spans
    size_t off = 0;
    uint64_t *tempDigitsXX1 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsXX2 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;

    uint64_t *tempDigitsYY1 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsYY2 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;

    uint64_t *tempDigitsXY1 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;
    uint64_t *tempDigitsXY2 = buffer + off;
    off += LameHackBufferSizeWhatShouldItBe;

    // Final128 buffers: reuse the "2" buffers (dead after pointwise multiply).
    // The "2" buffers are >= 2*Ddigits due to LameHack sizing above.
    uint64_t *Final128_XX = tempDigitsXX2;
    uint64_t *Final128_YY = tempDigitsYY2;
    uint64_t *Final128_XY = tempDigitsXY2;

    // Carry buffers: 7-way for NR (7*Ddigits+7 per buffer), 3-way otherwise (6*Ddigits)
    const size_t carryBufEntries = [&]() -> size_t {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            return static_cast<size_t>(7) * Ddigits + 7;
        } else {
            return static_cast<size_t>(6) * Ddigits;
        }
    }();

    uint64_t *CarryPropagationBuffer = buffer + off;
    off += carryBufEntries;

    uint64_t *CarryPropagationBuffer2 = buffer + off;
    off += carryBufEntries;

    uint64_t *CarryPropagationSync = &tempProducts[0];
    uint64_t *CarryPropagationSync2 = &tempProducts[16];

    // NR temp buffers — allocated from runtime layout (not macro offsets)
    // so they're at LameHack size and "2" buffers can serve as Final128_W*.
    uint64_t *tempDigitsW0_1 = nullptr;
    uint64_t *tempDigitsW0_2 = nullptr;
    uint64_t *tempDigitsW1_1 = nullptr;
    uint64_t *tempDigitsW1_2 = nullptr;
    uint64_t *tempDigitsW2_1 = nullptr;
    uint64_t *tempDigitsW2_2 = nullptr;
    uint64_t *tempDigitsW3_1 = nullptr;
    uint64_t *tempDigitsW3_2 = nullptr;
    uint64_t *Final128_W0 = nullptr;
    uint64_t *Final128_W1 = nullptr;
    uint64_t *Final128_W2 = nullptr;
    uint64_t *Final128_W3 = nullptr;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        tempDigitsW0_1 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW0_2 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW1_1 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW1_2 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW2_1 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW2_2 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW3_1 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;
        tempDigitsW3_2 = buffer + off;
        off += LameHackBufferSizeWhatShouldItBe;

        // NR Final128: reuse "2" buffers (dead after pointwise multiply, >= 2*Ddigits)
        Final128_W0 = tempDigitsW0_2;
        Final128_W1 = tempDigitsW1_2;
        Final128_W2 = tempDigitsW2_2;
        Final128_W3 = tempDigitsW3_2;
    }

    // XX = A^2
    RunNTT_3Way_Multiply<SharkFloatParams>(shared_data,
                                           OutXX,
                                           OutYY,
                                           OutXY,
                                           *A,
                                           *B,
                                           DzdcReal,
                                           DzdcImag,
                                           OutW0,
                                           OutW1,
                                           OutW2,
                                           OutW3,
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
                                           Ddigits,
                                           tempDigitsW0_1,
                                           tempDigitsW0_2,
                                           tempDigitsW1_1,
                                           tempDigitsW1_2,
                                           tempDigitsW2_1,
                                           tempDigitsW2_2,
                                           tempDigitsW3_1,
                                           tempDigitsW3_2,
                                           Final128_W0,
                                           Final128_W1,
                                           Final128_W2,
                                           Final128_W3);

    grid.sync();
}

template <class SharkFloatParams>
void
PrintMaxActiveBlocks(const HpShark::LaunchParams &launchParams, void *kernelFn, int sharedAmountBytes)
{

    std::cout << "Shared memory size bytes: " << sharedAmountBytes << std::endl;

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

        std::cout << "Available shared memory per block bytes: " << availableSharedMemory << std::endl;
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
    MultiplyHelperNTTV2Separates<SharkFloatParams>(
        combo->Roots,
        &combo->A,
        &combo->B,
        &combo->ResultX2,
        &combo->Result2XY,
        &combo->ResultY2,
        SharkFloatParams::EnableNewtonRaphson ? &combo->DzdcReal : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->DzdcImag : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultW0 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultW1 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultW2 : nullptr,
        SharkFloatParams::EnableNewtonRaphson ? &combo->ResultW3 : nullptr,
        grid,
        block,
        tempProducts);
}
