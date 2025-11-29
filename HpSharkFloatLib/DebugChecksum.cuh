#pragma once

#include "HpSharkFloat.cuh"
#include "DebugStateRaw.h"

#define USE_PARALLEL_FLETCHER64

#ifdef __CUDACC__

#include <cstdint>
#include <cooperative_groups.h>

template <class SharkFloatParams>
struct DebugGlobalCount {
    __device__ 
    void DebugMultiplyErase ()
    {
        if constexpr (HpShark::DebugGlobalState) {
            Data = {};
        }
    }

    DebugGlobalCountRaw Data;
};

template <class SharkFloatParams>
__device__ void DebugMultiplyIncrement (
    DebugGlobalCount<SharkFloatParams> *array,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    uint32_t count)
{
    if constexpr (HpShark::DebugGlobalState) {
        auto &index = array[block.group_index().x * block.dim_threads().x + block.thread_index().x];
        index.Data.multiplyCount += count;
        index.Data.blockIdx = block.group_index().x;
        index.Data.threadIdx = block.thread_index().x;
    }
}

template <class SharkFloatParams>
__device__ void
DebugCarryIncrement(DebugGlobalCount<SharkFloatParams> *array,
                    cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    uint32_t count)
{
    if constexpr (HpShark::DebugGlobalState) {
        auto &index = array[block.group_index().x * block.dim_threads().x +
                            block.thread_index().x];
        index.Data.carryCount += count;
        index.Data.blockIdx = block.group_index().x;
        index.Data.threadIdx = block.thread_index().x;
    }
}

template <class SharkFloatParams>
__device__ void
DebugNormalizeIncrement(DebugGlobalCount<SharkFloatParams> *array,
                        cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        uint32_t count)
{
    if constexpr (HpShark::DebugGlobalState) {
        auto &index = array[block.group_index().x * block.dim_threads().x +
                            block.thread_index().x];
        index.Data.normalizeCount += count;
        index.Data.blockIdx = block.group_index().x;
        index.Data.threadIdx = block.thread_index().x;
    }
}

namespace DebugChecksumGlobals {
// Limits (bump if you launch bigger blocks/grids)
static constexpr int kMaxBlocks = 4096;
static constexpr int kMaxThreadsPerBlock = 1024;
static constexpr int kWarpSize = 32;
static constexpr int kMaxWarpsPerBlock = kMaxThreadsPerBlock / kWarpSize; // 32

static __device__ uint32_t d_warp_s1[kMaxBlocks * kMaxWarpsPerBlock];
static __device__ uint32_t d_warp_s2[kMaxBlocks * kMaxWarpsPerBlock];
static __device__ uint32_t d_warp_lm[kMaxBlocks * kMaxWarpsPerBlock]; // length mod M

static __device__ uint32_t d_block_s1[kMaxBlocks];
static __device__ uint32_t d_block_s2[kMaxBlocks];
static __device__ uint32_t d_block_lm[kMaxBlocks];

static __device__ uint32_t d_final_s1;
static __device__ uint32_t d_final_s2;
}

template <class SharkFloatParams> struct DebugState {

    // ---------- Fletcher-64 over 32-bit words ----------
    static constexpr uint64_t FLETCHER64_MOD = 0xFFFFFFFFull; // 2^32 - 1

    struct F64Pair {
        uint32_t s1, s2;
    };

    // ---- helpers (64-bit only; VC++-friendly) ----

    // Fold a 64-bit x modulo (2^32-1) using end-around-carry.
    static __device__ __forceinline__ uint32_t
    modM_u64(uint64_t x)
    {
        // One fold is enough for 64-bit operands:
        uint64_t s = (x & 0xFFFFFFFFull) + (x >> 32);
        // Reduce once if still >= M
        if (s >= FLETCHER64_MOD)
            s -= FLETCHER64_MOD;
        return static_cast<uint32_t>(s);
    }

    // Multiply (a * b) mod (2^32-1), with a,b < M. Uses 64-bit product + one fold.
    static __device__ __forceinline__ uint32_t
    mulModM(uint32_t a, uint32_t b)
    {
        uint64_t prod = uint64_t(a) * uint64_t(b); // fits in 64-bit
        return modM_u64(prod);
    }

    // (x + y) mod (2^32-1) with 64-bit safety.
    static __device__ __forceinline__ uint32_t
    addModM(uint32_t x, uint32_t y)
    {
        uint64_t s = uint64_t(x) + uint64_t(y);
        if (s >= FLETCHER64_MOD)
            s -= FLETCHER64_MOD;
        return static_cast<uint32_t>(s);
    }

    // Combine two Fletcher-64 states (A then B), where lenB_words = # of 32-bit words in B.
    static __device__ __forceinline__ F64Pair
    fletcher64_combine(F64Pair A, F64Pair B, uint64_t lenB_words)
    {
        // s1 = (s1A + s1B) mod M
        uint32_t s1 = addModM(A.s1, B.s1);

        // s2 = (s2A + s2B + (lenB mod M) * s1A) mod M
        uint32_t len_mod = modM_u64(lenB_words); // 64-bit -> mod M using fold
        uint32_t term = mulModM(len_mod, A.s1);  // (lenB mod M) * s1A mod M

        // s2A + s2B
        uint64_t s2_wide = uint64_t(A.s2) + uint64_t(B.s2);
        if (s2_wide >= FLETCHER64_MOD)
            s2_wide -= FLETCHER64_MOD;

        // + term, then reduce once more if needed
        s2_wide += term;
        if (s2_wide >= FLETCHER64_MOD)
            s2_wide -= FLETCHER64_MOD;

        return {s1, static_cast<uint32_t>(s2_wide)};
    }

// Compute Fletcher-64 over [lo, hi) 32-bit words with seed (s1,s2).
    static __device__ __forceinline__ F64Pair
    fletcher64_chunk_words(const uint32_t *w32, uint64_t lo, uint64_t hi, F64Pair seed)
    {
        uint64_t s1 = seed.s1;
        uint64_t s2 = seed.s2;

        // Reduce s1 and s2 on *every* word so s2 sees s1 mod M, matching the host.
        for (uint64_t i = lo; i < hi; ++i) {
            s1 += w32[i];
            s1 = (s1 & 0xFFFFFFFFull) + (s1 >> 32);
            if (s1 >= FLETCHER64_MOD)
                s1 -= FLETCHER64_MOD;

            s2 += s1;
            s2 = (s2 & 0xFFFFFFFFull) + (s2 >> 32);
            if (s2 >= FLETCHER64_MOD)
                s2 -= FLETCHER64_MOD;
        }

        return {static_cast<uint32_t>(s1), static_cast<uint32_t>(s2)};
    }


    // Pack/unpack Fletcher state to/from uint64 (s2<<32 | s1)
    static __device__ __forceinline__ F64Pair
    unpack_seed(uint64_t initial)
    {
        return F64Pair{uint32_t(initial & 0xFFFFFFFFull), uint32_t(initial >> 32)};
    }
    static __device__ __forceinline__ uint64_t
    pack_crc(F64Pair p)
    {
        return (uint64_t(p.s2) << 32) | uint64_t(p.s1);
    }

    static __device__ __forceinline__ void
    compute_span(uint64_t tid, uint64_t T, uint64_t N, uint64_t &lo, uint64_t &hi)
    {
        const uint64_t q = N / T;
        const uint64_t r = N % T;
        lo = tid * q + (tid < r ? tid : r);
        hi = lo + q + (tid < r ? 1ull : 0ull);
    }

    __device__ void
    Reset(DebugStatePurpose purpose, DebugState<SharkFloatParams> &other)
    {
        if constexpr (HpShark::DebugChecksums) {
            Data = other.Data;
            Data.ChecksumPurpose = purpose;
        }
    }

    __device__ void
    Reset(UseConvolution useConvolution,
          cooperative_groups::grid_group &grid,
          cooperative_groups::thread_block &block,
          const uint32_t *arrayToChecksum,
          size_t arraySize,
          DebugStatePurpose purpose,
          int recursionDepth,
          int callIndex)
    {
        if constexpr (HpShark::DebugChecksums) {
            Data.Initialized = 1;
            Data.Checksum = 0;
            Data.Block = block.group_index().x;
            Data.Thread = block.thread_index().x;
            Data.ArraySize = arraySize;
            Data.ChecksumPurpose = purpose;

            auto res = ComputeCRC64(grid, block, arrayToChecksum, arraySize, 0);
            Data.Checksum = res;
            Data.RecursionDepth = recursionDepth;
            Data.CallIndex = callIndex;
            Data.Convolution = useConvolution;
        }
    }

    __device__ void
    Reset(UseConvolution useConvolution,
          cooperative_groups::grid_group &grid,
          cooperative_groups::thread_block &block,
          const uint64_t *arrayToChecksum,
          size_t arraySize,
          DebugStatePurpose purpose,
          int recursionDepth,
          int callIndex)
    {
        if constexpr (HpShark::DebugChecksums) {
            Data.Initialized = 1;
            Data.Checksum = 0;
            Data.Block = block.group_index().x;
            Data.Thread = block.thread_index().x;
            Data.ArraySize = arraySize;
            Data.ChecksumPurpose = purpose;

            auto res = ComputeCRC64(grid, block, arrayToChecksum, arraySize, 0);
            Data.Checksum = res;
            Data.RecursionDepth = recursionDepth;
            Data.CallIndex = callIndex;
            Data.Convolution = useConvolution;
        }
    }

    __device__ void
    Erase(cooperative_groups::grid_group &grid,
          cooperative_groups::thread_block &block,
          DebugStatePurpose purpose,
          int recursionDepth,
          int callIndex)
    {
        if constexpr (HpShark::DebugChecksums) {
            Data.Initialized = 0;
            Data.Checksum = 0;
            Data.Block = 0;
            Data.Thread = 0;
            Data.ArraySize = 0;
            Data.ChecksumPurpose = DebugStatePurpose::Invalid;

            Data.RecursionDepth = 0;
            Data.CallIndex = 0;
            Data.Convolution = UseConvolution::No;
        }
    }

#ifdef USE_PARALLEL_FLETCHER64
    __device__ uint64_t
    ComputeCRC64(cooperative_groups::grid_group &grid,
                                               cooperative_groups::thread_block &block,
                                               const uint32_t *data,
                                               size_t size,
                                               uint64_t initialCrc)
    {
        using namespace DebugChecksumGlobals;

        // Partition using block/thread linearization to match reduction order
        const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;
        const uint64_t T = uint64_t(gridDim.x) * blockDim.x;
        const uint64_t Nw = static_cast<uint64_t>(size);

        uint64_t lo, hi;
        compute_span(tid, T, Nw, lo, hi);

        // Per-thread local (must fold per word)
        F64Pair local = (hi > lo) ? fletcher64_chunk_words(
                                        reinterpret_cast<const uint32_t *>(data), lo, hi, F64Pair{0, 0})
                                  : F64Pair{0, 0};

        uint32_t lmod = modM_u64(hi - lo);
        uint32_t s1 = local.s1;
        uint32_t s2 = local.s2;

        // ---- Warp reduction: order-preserving scan (no shared mem) ----
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warp_base = warp * 32;
        const int warp_active = min(32, max(0, int(blockDim.x) - warp_base));
        unsigned mask = __ballot_sync(0xFFFFFFFFu, lane < warp_active);

        if (warp_active > 0) {
            for (int offset = 1; offset < warp_active; offset <<= 1) {
                uint32_t s1_left = __shfl_up_sync(mask, s1, offset);
                uint32_t s2_left = __shfl_up_sync(mask, s2, offset);
                uint32_t lm_left = __shfl_up_sync(mask, lmod, offset);
                if (lane >= offset) {
                    F64Pair C =
                        fletcher64_combine(F64Pair{s1_left, s2_left}, F64Pair{s1, s2}, /*lenB*/ lmod);
                    s1 = C.s1;
                    s2 = C.s2;
                    uint64_t sum = uint64_t(lm_left) + uint64_t(lmod);
                    if (sum >= FLETCHER64_MOD)
                        sum -= FLETCHER64_MOD;
                    lmod = static_cast<uint32_t>(sum);
                }
            }
            const int last = warp_active - 1;
            uint32_t w_s1 = __shfl_sync(mask, s1, last);
            uint32_t w_s2 = __shfl_sync(mask, s2, last);
            uint32_t w_lm = __shfl_sync(mask, lmod, last);
            if (lane == 0) {
                const int idx = blockIdx.x * kMaxWarpsPerBlock + warp;
                d_warp_s1[idx] = w_s1;
                d_warp_s2[idx] = w_s2;
                d_warp_lm[idx] = w_lm;
            }
        } else if (lane == 0) {
            const int idx = blockIdx.x * kMaxWarpsPerBlock + warp;
            d_warp_s1[idx] = 0;
            d_warp_s2[idx] = 0;
            d_warp_lm[idx] = 0;
        }

        block.sync();

        // ---- Per-block serial over warps (ordered; do NOT skip lm==0) ----
        if (threadIdx.x == 0) {
            const int warpsInBlock = (blockDim.x + 31) / 32;
            uint32_t bs1 = 0, bs2 = 0, blm = 0;
            for (int w = 0; w < warpsInBlock; ++w) {
                const int idx = blockIdx.x * kMaxWarpsPerBlock + w;
                F64Pair C = fletcher64_combine(F64Pair{bs1, bs2},
                                               F64Pair{d_warp_s1[idx], d_warp_s2[idx]},
                                               /*lenB*/ d_warp_lm[idx]);
                bs1 = C.s1;
                bs2 = C.s2;
                uint64_t sum = uint64_t(blm) + uint64_t(d_warp_lm[idx]);
                if (sum >= FLETCHER64_MOD)
                    sum -= FLETCHER64_MOD;
                blm = static_cast<uint32_t>(sum);
            }
            d_block_s1[blockIdx.x] = bs1;
            d_block_s2[blockIdx.x] = bs2;
            d_block_lm[blockIdx.x] = blm;
        }

        grid.sync();

// ---- Parallel grid reduction over per-block tuples (no shared mem) ----
        // Each round halves the number of “live” blocks by combining A=even, B=even+offset into A.
        // Only threadIdx.x == 0 in participating blocks touches the tuples.
        {
            int live = gridDim.x; // number of active block results this round
            int offset = 1;       // distance to the right neighbor

            while (live > 1) {
                // A block participates this round if it's an even "group head"
                // and its partner exists.
                const bool isGroupHead = ((blockIdx.x % (2 * offset)) == 0);
                const int partner = blockIdx.x + offset;

                if (isGroupHead && partner < gridDim.x) {
                    if (threadIdx.x == 0) {
                        // Read A (this block) and B (partner), combine as A ⊕ B (left→right)
                        uint32_t As1 = DebugChecksumGlobals::d_block_s1[blockIdx.x];
                        uint32_t As2 = DebugChecksumGlobals::d_block_s2[blockIdx.x];
                        uint32_t Alm = DebugChecksumGlobals::d_block_lm[blockIdx.x];

                        uint32_t Bs1 = DebugChecksumGlobals::d_block_s1[partner];
                        uint32_t Bs2 = DebugChecksumGlobals::d_block_s2[partner];
                        uint32_t Blm = DebugChecksumGlobals::d_block_lm[partner];

                        F64Pair A{As1, As2};
                        F64Pair B{Bs1, Bs2};
                        F64Pair C = fletcher64_combine(A, B, /*lenB*/ Blm);

                        DebugChecksumGlobals::d_block_s1[blockIdx.x] = C.s1;
                        DebugChecksumGlobals::d_block_s2[blockIdx.x] = C.s2;

                        uint64_t sum = uint64_t(Alm) + uint64_t(Blm);
                        if (sum >= FLETCHER64_MOD)
                            sum -= FLETCHER64_MOD;
                        DebugChecksumGlobals::d_block_lm[blockIdx.x] = static_cast<uint32_t>(sum);
                    }
                }

                // Everyone waits before next round
                grid.sync();

                // Next round: groups double in size, partners move farther away
                live = (live + 1) >> 1; // ceil(live/2)
                offset <<= 1;
            }

            // After the loop, block 0 holds the full concatenation in d_block_*[0].
            if (tid == 0) {
                uint32_t gs1 = DebugChecksumGlobals::d_block_s1[0];
                uint32_t gs2 = DebugChecksumGlobals::d_block_s2[0];
                uint32_t glm = DebugChecksumGlobals::d_block_lm[0];

                // Apply initial seed (packed as s2<<32 | s1)
                F64Pair final = fletcher64_combine(unpack_seed(initialCrc), F64Pair{gs1, gs2}, glm);
                DebugChecksumGlobals::d_final_s1 = final.s1;
                DebugChecksumGlobals::d_final_s2 = final.s2;
            }
        }


        grid.sync();
        return (uint64_t(d_final_s2) << 32) | uint64_t(d_final_s1);
    }

    __device__ uint64_t
    ComputeCRC64(cooperative_groups::grid_group &grid,
                                               cooperative_groups::thread_block &block,
                                               const uint64_t *data,
                                               size_t size,
                                               uint64_t initialCrc)
    {
        using namespace DebugChecksumGlobals;

        const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;
        const uint64_t T = uint64_t(gridDim.x) * blockDim.x;
        const uint64_t Nw = static_cast<uint64_t>(size) * 2ull;

        uint64_t lo, hi;
        compute_span(tid, T, Nw, lo, hi);

        const uint32_t *w32 = reinterpret_cast<const uint32_t *>(data);
        F64Pair local = (hi > lo) ? fletcher64_chunk_words(w32, lo, hi, F64Pair{0, 0}) : F64Pair{0, 0};

        uint32_t lmod = modM_u64(hi - lo);
        uint32_t s1 = local.s1;
        uint32_t s2 = local.s2;

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warp_base = warp * 32;
        const int warp_active = min(32, max(0, int(blockDim.x) - warp_base));
        unsigned mask = __ballot_sync(0xFFFFFFFFu, lane < warp_active);

        if (warp_active > 0) {
            for (int offset = 1; offset < warp_active; offset <<= 1) {
                uint32_t s1_left = __shfl_up_sync(mask, s1, offset);
                uint32_t s2_left = __shfl_up_sync(mask, s2, offset);
                uint32_t lm_left = __shfl_up_sync(mask, lmod, offset);
                if (lane >= offset) {
                    F64Pair C =
                        fletcher64_combine(F64Pair{s1_left, s2_left}, F64Pair{s1, s2}, /*lenB*/ lmod);
                    s1 = C.s1;
                    s2 = C.s2;
                    uint64_t sum = uint64_t(lm_left) + uint64_t(lmod);
                    if (sum >= FLETCHER64_MOD)
                        sum -= FLETCHER64_MOD;
                    lmod = static_cast<uint32_t>(sum);
                }
            }
            const int last = warp_active - 1;
            uint32_t w_s1 = __shfl_sync(mask, s1, last);
            uint32_t w_s2 = __shfl_sync(mask, s2, last);
            uint32_t w_lm = __shfl_sync(mask, lmod, last);
            if (lane == 0) {
                const int idx = blockIdx.x * kMaxWarpsPerBlock + warp;
                d_warp_s1[idx] = w_s1;
                d_warp_s2[idx] = w_s2;
                d_warp_lm[idx] = w_lm;
            }
        } else if (lane == 0) {
            const int idx = blockIdx.x * kMaxWarpsPerBlock + warp;
            d_warp_s1[idx] = 0;
            d_warp_s2[idx] = 0;
            d_warp_lm[idx] = 0;
        }

        block.sync();

        if (threadIdx.x == 0) {
            const int warpsInBlock = (blockDim.x + 31) / 32;
            uint32_t bs1 = 0, bs2 = 0, blm = 0;
            for (int w = 0; w < warpsInBlock; ++w) {
                const int idx = blockIdx.x * kMaxWarpsPerBlock + w;
                F64Pair C = fletcher64_combine(F64Pair{bs1, bs2},
                                               F64Pair{d_warp_s1[idx], d_warp_s2[idx]},
                                               /*lenB*/ d_warp_lm[idx]);
                bs1 = C.s1;
                bs2 = C.s2;
                uint64_t sum = uint64_t(blm) + uint64_t(d_warp_lm[idx]);
                if (sum >= FLETCHER64_MOD)
                    sum -= FLETCHER64_MOD;
                blm = static_cast<uint32_t>(sum);
            }
            d_block_s1[blockIdx.x] = bs1;
            d_block_s2[blockIdx.x] = bs2;
            d_block_lm[blockIdx.x] = blm;
        }

        grid.sync();

// ---- Parallel grid reduction over per-block tuples (no shared mem) ----
        // Each round halves the number of “live” blocks by combining A=even, B=even+offset into A.
        // Only threadIdx.x == 0 in participating blocks touches the tuples.
        {
            int live = gridDim.x; // number of active block results this round
            int offset = 1;       // distance to the right neighbor

            while (live > 1) {
                // A block participates this round if it's an even "group head"
                // and its partner exists.
                const bool isGroupHead = ((blockIdx.x % (2 * offset)) == 0);
                const int partner = blockIdx.x + offset;

                if (isGroupHead && partner < gridDim.x) {
                    if (threadIdx.x == 0) {
                        // Read A (this block) and B (partner), combine as A ⊕ B (left→right)
                        uint32_t As1 = DebugChecksumGlobals::d_block_s1[blockIdx.x];
                        uint32_t As2 = DebugChecksumGlobals::d_block_s2[blockIdx.x];
                        uint32_t Alm = DebugChecksumGlobals::d_block_lm[blockIdx.x];

                        uint32_t Bs1 = DebugChecksumGlobals::d_block_s1[partner];
                        uint32_t Bs2 = DebugChecksumGlobals::d_block_s2[partner];
                        uint32_t Blm = DebugChecksumGlobals::d_block_lm[partner];

                        F64Pair A{As1, As2};
                        F64Pair B{Bs1, Bs2};
                        F64Pair C = fletcher64_combine(A, B, /*lenB*/ Blm);

                        DebugChecksumGlobals::d_block_s1[blockIdx.x] = C.s1;
                        DebugChecksumGlobals::d_block_s2[blockIdx.x] = C.s2;

                        uint64_t sum = uint64_t(Alm) + uint64_t(Blm);
                        if (sum >= FLETCHER64_MOD)
                            sum -= FLETCHER64_MOD;
                        DebugChecksumGlobals::d_block_lm[blockIdx.x] = static_cast<uint32_t>(sum);
                    }
                }

                // Everyone waits before next round
                grid.sync();

                // Next round: groups double in size, partners move farther away
                live = (live + 1) >> 1; // ceil(live/2)
                offset <<= 1;
            }

            // After the loop, block 0 holds the full concatenation in d_block_*[0].
            if (tid == 0) {
                uint32_t gs1 = DebugChecksumGlobals::d_block_s1[0];
                uint32_t gs2 = DebugChecksumGlobals::d_block_s2[0];
                uint32_t glm = DebugChecksumGlobals::d_block_lm[0];

                // Apply initial seed (packed as s2<<32 | s1)
                F64Pair final = fletcher64_combine(unpack_seed(initialCrc), F64Pair{gs1, gs2}, glm);
                DebugChecksumGlobals::d_final_s1 = final.s1;
                DebugChecksumGlobals::d_final_s2 = final.s2;
            }

        }


        grid.sync();
        return (uint64_t(d_final_s2) << 32) | uint64_t(d_final_s1);
    }



#else
    // ---------- uint32_t* overload: single-thread (rank 0) Fletcher-64 ----------
    __device__ uint64_t
    ComputeCRC64(cooperative_groups::grid_group &grid,
                                               cooperative_groups::thread_block &block,
                                               const uint32_t *data,
                                               size_t size,
                                               uint64_t initialCrc)
    {
        using namespace DebugChecksumGlobals;

        // Seed packed as (s2<<32 | s1), same as host
        uint32_t s1 = static_cast<uint32_t>(initialCrc & 0xFFFFFFFFull);
        uint32_t s2 = static_cast<uint32_t>(initialCrc >> 32);
        const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;

        if (tid == 0) {
            constexpr uint64_t M = 0xFFFFFFFFull; // 2^32 - 1
            auto foldM = [](uint64_t v) -> uint32_t {
                v = (v & M) + (v >> 32); // end-around carry fold
                if (v >= M)
                    v -= M; // single subtract suffices
                return static_cast<uint32_t>(v);
            };

            for (size_t i = 0; i < size; ++i) {
                s1 = foldM(uint64_t(s1) + uint64_t(data[i]));
                s2 = foldM(uint64_t(s2) + uint64_t(s1));
            }

            d_final_s1 = s1;
            d_final_s2 = s2;
        }

        grid.sync();
        uint64_t ret = (uint64_t(d_final_s2) << 32) | uint64_t(d_final_s1);
        grid.sync();
        return ret;
    }

    // ---------- uint64_t* overload: single-thread (rank 0), split into two 32-bit words (lo then hi)
    // ----------
    __device__ uint64_t
    ComputeCRC64(cooperative_groups::grid_group &grid,
                                               cooperative_groups::thread_block & /*block*/,
                                               const uint64_t *data,
                                               size_t size,
                                               uint64_t initialCrc)
    {
        using namespace DebugChecksumGlobals;

        uint32_t s1 = static_cast<uint32_t>(initialCrc & 0xFFFFFFFFull);
        uint32_t s2 = static_cast<uint32_t>(initialCrc >> 32);
        const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;

        if (tid == 0) {
            constexpr uint64_t M = 0xFFFFFFFFull; // 2^32 - 1
            auto foldM = [](uint64_t v) -> uint32_t {
                v = (v & M) + (v >> 32);
                if (v >= M)
                    v -= M;
                return static_cast<uint32_t>(v);
            };

            for (size_t i = 0; i < size; ++i) {
                const uint64_t w = data[i];
                const uint32_t lo = static_cast<uint32_t>(w & 0xFFFFFFFFull);
                const uint32_t hi = static_cast<uint32_t>(w >> 32);

                // low 32, then high 32 (matches host split and your device little-endian view)
                s1 = foldM(uint64_t(s1) + uint64_t(lo));
                s2 = foldM(uint64_t(s2) + uint64_t(s1));

                s1 = foldM(uint64_t(s1) + uint64_t(hi));
                s2 = foldM(uint64_t(s2) + uint64_t(s1));
            }

            d_final_s1 = s1;
            d_final_s2 = s2;
        }

        grid.sync();
        uint64_t ret = (uint64_t(d_final_s2) << 32) | uint64_t(d_final_s1);
        grid.sync();
        return ret;
    }


#endif

    DebugStateRaw Data;
};


#endif // __CUDACC__