#include "MultiplyKaratsuba.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "DebugChecksum.cuh"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>


namespace cg = cooperative_groups;

static constexpr auto
CalcAlign16Bytes64BitIndex(uint64_t Sixty4BitIndex) {
    return Sixty4BitIndex % 2 == 0 ? 0 : 1;
}

static constexpr auto
CalcAlign16Bytes32BitIndex(uint64_t Thirty2BitIndex) {
    return 4 - (Thirty2BitIndex % 4);
}


// Initialize the random number generator state.  Note that
// this uses a constant seed.  This is lame and we should be
// using a different seed for each thread.
static void
__device__ DebugInitRandom (
    cg::thread_block &block,
    curandState *state)
{
    int index = block.group_index().x * block.dim_threads().x + block.thread_index().x;
    curand_init(1234, index, 0, &state[index]);
}

//// Introduce a random delay.  This delay is per-thread and is
//// intended to exacerbate any races.  Note that you may want
//// instead a block-level delay.  This isn't it.
//static void
//__device__ DebugRandomDelay (
//    cg::thread_block &block,
//    curandState *state)
//{
//    int idx = block.group_index().x * block.dim_threads().x + block.thread_index().x;
//
//    static constexpr int maxIters = 1000;
//    float myrandf = curand_uniform(&state[idx]);
//    myrandf *= (maxIters + 0.999999);
//    int myrand = (int)truncf(myrandf);
//
//    volatile int dummy = 0;
//    for (auto i = 0; i < myrand; ++i) {
//        auto orig = dummy;
//        dummy = orig + 1;
//    }
//}

// Shared-memory “bands” prefetch model (used by ProcessConvolutionBatchPipelined):
// We cache four bands per tile: A_idx[], B_idx[], A_mirr[], B_mirr[].
// Worst-case unique values per tile (in uint32): 
//   values = 2*B + 2*(K + B - 1) = 2*K + 4*B - 2
// where K = threads per block, B = batch size cap per tile.
// For double buffering, we need 2 * values * sizeof(uint32_t).

template<class SharkFloatParams, int RecursionDepth>
struct SharedMemoryLayout {
    // ---- Explanatory constants (no more magic numbers) ----
    static constexpr int kBytesPerU32 = 4;   // sizeof(uint32_t)
    static constexpr int kIdxBands = 2;   // A_idx, B_idx
    static constexpr int kMirrBands = 2;   // A_mirr, B_mirr
    static constexpr int kDoubleBuffers = 2;   // ping & pong
    static constexpr int kBatchCapPipelined = 8;   // B_max we support in the tiler (tunable)

    // Threads per block (compile-time for the params)
    static constexpr int K = SharkFloatParams::GlobalThreadsPerBlock;
    static constexpr int Bmax = kBatchCapPipelined;

    // Worst-case unique uint32 values needed for one tile (single buffer)
    // values = 2*B + 2*(K + B - 1) = 2*K + 4*B - 2
    static constexpr int kWorstCaseTileU32 = (kIdxBands * Bmax) + (kMirrBands * (K + Bmax - 1));

    // Single-buffer bytes for one tile
    static constexpr int kSingleBufferBytes = kWorstCaseTileU32 * kBytesPerU32;

    // Double-buffer bytes (ping-pong) for overlap
    static constexpr int kDoubleBufferBytes = kDoubleBuffers * kSingleBufferBytes;

    // ---- Existing shared layout components (unchanged here) ----
    static constexpr int SyncVarsSize = 2 * sizeof(uint32_t);

    static constexpr int BaseSharedMemory = HpShark::LoadAllInShared
        ? (SharkFloatParams::GlobalNumUint32 * 4 * 2  // aDigits + bDigits
            + SharkFloatParams::GlobalNumUint32 * 2    // x_diff_abs + y_diff_abs
            + 1024)                                    // misc padding/other
        : (2048 + SyncVarsSize);                      // minimum when not loading all in shared + sync vars

    // ---- New pipeline buffer size expressed via the constants above ----
    static constexpr int PipelineBufferSize = !HpShark::LoadAllInShared
        ? kDoubleBufferBytes                 // double-buffered worst-case tile
        : 0;

    // Offsets & totals by recursion
    static constexpr int RecursionOffset = RecursionDepth * (BaseSharedMemory + PipelineBufferSize);
    static constexpr int TotalSharedMemory = RecursionOffset + BaseSharedMemory + PipelineBufferSize;

    static_assert(TotalSharedMemory <= 48 * 1024, "Shared memory exceeds 48KB limit");
};

// Get sync variables for specific recursion depth
template<class SharkFloatParams, int RecursionDepth>
__device__ uint32_t *GetRecursionSyncVars(uint32_t *base_shared_data) {
    constexpr int base_offset = SharedMemoryLayout<SharkFloatParams, RecursionDepth>::RecursionOffset;
    return base_shared_data + (base_offset / sizeof(uint32_t));
}

// Get pipeline buffer for specific recursion depth (when not loading all in shared)
template<class SharkFloatParams, int RecursionDepth>
__device__ uint32_t *GetRecursionPipelineBuffer(uint32_t *base_shared_data) {
    if constexpr (HpShark::LoadAllInShared) {
        return nullptr; // No pipeline buffer needed
    } else {
        constexpr int base_offset = SharedMemoryLayout<SharkFloatParams, RecursionDepth>::RecursionOffset;
        constexpr int base_size = SharedMemoryLayout<SharkFloatParams, RecursionDepth>::BaseSharedMemory;
        return base_shared_data + ((base_offset + base_size) / sizeof(uint32_t));
    }
}

// Compare two digit arrays, returning 1 if a > b, -1 if a < b, and 0 if equal
template<int n1, int n2>
static __device__ int CompareDigits(
    const uint32_t *SharkRestrict highArray,
    const uint32_t *SharkRestrict lowArray)
{
    // The biggest possible "digit index" is one less
    // than the max of the two sizes.
    int maxLen = std::max(n1, n2);

    // Compare top-down, from maxLen-1 down to 0
    for (int i = maxLen - 1; i >= 0; --i) {
        // Treat out-of-range as zero
        uint32_t a_val = (i < n1) ? highArray[i] : 0u;
        uint32_t b_val = (i < n2) ? lowArray[i] : 0u;

        if (a_val > b_val) {
            return 1;  // A is bigger
        } else if (a_val < b_val) {
            return -1; // B is bigger
        }
    }
    return 0;
}

// Subtract two digit arrays, returning the result.
// This is a serial implementation.
template<int n1, int n2>
__device__ static void
SubtractDigitsSerial(const uint32_t *a, const uint32_t *b, uint32_t *result) {
    uint64_t borrow = 0;
    for (int i = 0; i < n1; ++i) {
        uint64_t ai;
        uint64_t bi;

        ai = a[i];

        if (i >= n2) {
            bi = 0;
        } else {
            bi = b[i];
        }

        uint64_t temp = ai - bi - borrow;
        if (ai < bi + borrow) {
            borrow = 1;
            temp += ((uint64_t)1 << 32);
        } else {
            borrow = 0;
        }
        result[i] = (uint32_t)temp;
    }
}

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/**
 * Parallel subtraction (a1 - b1) and (a2 - b2), stored in global_(x|y)_diff_abs,
 * using a multi-pass approach to propagate borrows.
 *
 * The function attempts to subtract each digit of 'b' from 'a' in parallel,
 * then uses repeated passes (do/while) to handle newly introduced borrows
 * until no more remain or a maximum pass count is reached.
 * 
 * Corrupts x_diff_abs shared memory intentionally
 */
template<
    class SharkFloatParams,
    int a1n,
    int b1n,
    int a2n,
    int b2n,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void
SubtractDigitsParallel(
    uint32_t *SharkRestrict x_diff_abs,
    uint32_t *SharkRestrict y_diff_abs,
    const uint32_t *SharkRestrict a1,
    const uint32_t *SharkRestrict b1,
    const uint32_t *SharkRestrict a2,
    const uint32_t *SharkRestrict b2,
    uint32_t *SharkRestrict subtractionBorrows1a,
    uint32_t *SharkRestrict subtractionBorrows1b,
    uint32_t *SharkRestrict subtractionBorrows2a,
    uint32_t *SharkRestrict subtractionBorrows2b,
    uint32_t *SharkRestrict global_x_diff_abs,
    uint32_t *SharkRestrict global_y_diff_abs,
    uint32_t *SharkRestrict globalBorrowAny,
    cg::grid_group &grid,
    cg::thread_block &block
) {
    // Note: steps on this.
    auto *SharkRestrict sharedBorrowAny = x_diff_abs;

    // Note: not ExecutionBlockBase
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *globalBorrowAny = 0;
    }

    if (block.thread_index().x == 0) {
        *sharedBorrowAny = 0;
    }

    // Constants 
    constexpr int MaxPasses = 5000;     // maximum number of multi-pass sweeps

    // We'll define a grid-stride range covering [0..n) for each pass
    // 1) global thread id
    int tid = (block.group_index().x - ExecutionBlockBase) * block.dim_threads().x + block.thread_index().x;
    // 2) stride
    int stride = block.dim_threads().x * ExecutionNumBlocks;

    constexpr auto n1max = std::max(a1n, b1n);
    constexpr auto n2max = std::max(a2n, b2n);
    constexpr auto nmax = std::max(n1max, n2max);

    // (1) First pass: naive partial difference (a[i] - b[i]) and set borrowBit
    // Instead of dividing digits among blocks, each thread does a grid-stride loop:
    for (int idx = tid; idx < nmax; idx += stride) {
        uint32_t ai1;
        uint32_t bi1;
        uint32_t ai2;
        uint32_t bi2;

        // Fill in with 0s if idx is out of bounds
        if (idx < a1n) {
            ai1 = a1[idx];
        } else {
            ai1 = 0;
        }

        if (idx < a2n) {
            ai2 = a2[idx];
        } else {
            ai2 = 0;
        }

        if (idx < b1n) {
            bi1 = b1[idx];
        } else {
            bi1 = 0;
        }

        if (idx < b2n) {
            bi2 = b2[idx];
        } else {
            bi2 = 0;
        }

        // naive difference
        uint64_t diff1 = (uint64_t)ai1 - (uint64_t)bi1;
        uint64_t diff2 = (uint64_t)ai2 - (uint64_t)bi2;

        uint32_t borrow1 = (ai1 < bi1) ? 1u : 0u;
        uint32_t borrow2 = (ai2 < bi2) ? 1u : 0u;

        global_x_diff_abs[idx] = static_cast<uint32_t>(diff1 & 0xFFFFFFFFu);
        subtractionBorrows1a[idx] = borrow1;

        global_y_diff_abs[idx] = static_cast<uint32_t>(diff2 & 0xFFFFFFFFu);
        subtractionBorrows2a[idx] = borrow2;
    }

    // We'll do repeated passes to fix newly introduced borrows
    uint32_t *curBorrow1 = subtractionBorrows1a;
    uint32_t *newBorrow1 = subtractionBorrows1b;
    uint32_t *curBorrow2 = subtractionBorrows2a;
    uint32_t *newBorrow2 = subtractionBorrows2b;
    int pass = 0;
    uint32_t initialBorrowAny = 0;

    // sync the entire grid before multi-pass fixes
    grid.sync();

    do {
        // (2) For each digit, apply the borrow from the previous digit
        for (int idx = tid; idx < nmax; idx += stride) {
            uint64_t borrow_in1 = 0ULL;
            uint64_t borrow_in2 = 0ULL;
            if (idx > 0) {   // borrow_in is from digit (idx-1)
                borrow_in1 = (uint64_t)(curBorrow1[idx - 1]);
                borrow_in2 = (uint64_t)(curBorrow2[idx - 1]);
            }

            uint32_t digit1 = global_x_diff_abs[idx];
            uint32_t digit2 = global_y_diff_abs[idx];

            // subtract the borrow
            uint64_t sum1 = (uint64_t)digit1 - borrow_in1;
            uint64_t sum2 = (uint64_t)digit2 - borrow_in2;

            // store updated digit
            global_x_diff_abs[idx] = static_cast<uint32_t>(sum1 & 0xFFFFFFFFULL);
            global_y_diff_abs[idx] = static_cast<uint32_t>(sum2 & 0xFFFFFFFFULL);

            // If sum is negative => top bit is 1 => new borrow
            if (sum1 & 0x8000'0000'0000'0000ULL) {
                newBorrow1[idx] = 1;
                atomicAdd(sharedBorrowAny, 1);
            } else {
                newBorrow1[idx] = 0;
            }

            if (sum2 & 0x8000'0000'0000'0000ULL) {
                newBorrow2[idx] = 1;
                atomicAdd(sharedBorrowAny, 1);
            } else {
                newBorrow2[idx] = 0;
            }
        }

        // (a) Block-level synchronization (so all threads see final sharedBorrowAny)
        block.sync();

        // The block's thread 0 aggregates once into globalBorrowAny
        if (block.thread_index().x == 0) {
            // Add sharedBorrowAny to the global counter
            atomicAdd(globalBorrowAny, *sharedBorrowAny);

            // Reset local aggregator for the next pass
            *sharedBorrowAny = 0;
        }

        // sync before checking if any new borrows remain
        grid.sync();

        auto tempCopyGlobalBorrowAny = *globalBorrowAny;
        if (tempCopyGlobalBorrowAny == initialBorrowAny) {
            break;  // no new borrows => done
        }

        grid.sync();
        initialBorrowAny = tempCopyGlobalBorrowAny;

        // swap curBorrow, newBorrow
        uint32_t *tmp = curBorrow1;
        curBorrow1 = newBorrow1;
        newBorrow1 = tmp;

        tmp = curBorrow2;
        curBorrow2 = newBorrow2;
        newBorrow2 = tmp;

        pass++;
    } while (pass < MaxPasses);

    if constexpr (HpShark::Debug) {
        if (pass == MaxPasses && block.group_index().x == 0) {
            // This will deadlock the kernel because this problem is hard to diagnose
            grid.sync();
        }
    }
}


// This implementation is pretty ameteur hour since we're not doing blelloch scan
// Next version maybe I'll try something in that direction.  For now since we're
// mostly dealing with random numbers anyway and not weird cases where every digit
// generates a borrow or something, this should be good enough.
template<
    class SharkFloatParams,
    int a1n, int b1n,
    int a2n, int b2n,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void
SubtractDigitsParallelImproved3(
    // Working arrays (which may be in shared memory)
    uint32_t *SharkRestrict x_diff_abs,
    uint32_t *SharkRestrict y_diff_abs,
    // Input digit arrays (for the two halves)
    const uint32_t *SharkRestrict a1,
    const uint32_t *SharkRestrict b1,
    const uint32_t *SharkRestrict a2,
    const uint32_t *SharkRestrict b2,
    // Two borrow arrays (one for each half)
    uint32_t *SharkRestrict subtractionBorrows1a,
    uint32_t *SharkRestrict subtractionBorrows2a,
    uint32_t *SharkRestrict subtractionBorrows1b,
    uint32_t *SharkRestrict subtractionBorrows2b,
    // An array (of size ExecutionNumBlocks) for storing each block's final borrow
    uint32_t *SharkRestrict blockBorrow1,
    uint32_t *SharkRestrict blockBorrow2,
    // Global buffers to hold the "working" differences
    uint32_t *SharkRestrict global_x_diff_abs,
    uint32_t *SharkRestrict global_y_diff_abs,
    // A single global counter to indicate if any borrow remains
    uint32_t *SharkRestrict globalBorrowAny,
    cg::grid_group &grid,
    cg::thread_block &block) {

    // Note: steps on this.
    auto *SharkRestrict sharedBorrowAny =
        HpShark::LoadAllInShared ?
        x_diff_abs :
        &x_diff_abs[block.group_index().x];

    if constexpr (ExecutionNumBlocks > 1) {
        // Compute maximum digit count from the two halves.
        constexpr int n1max = (a1n > b1n) ? a1n : b1n;
        constexpr int n2max = (a2n > b2n) ? a2n : b2n;
        constexpr int nmax = (n1max > n2max) ? n1max : n2max;

        // Use the same mapping as the original:
        const int tid = (block.group_index().x - ExecutionBlockBase) * block.dim_threads().x + block.thread_index().x;
        const int stride = block.dim_threads().x * ExecutionNumBlocks;

        // Reset the global borrow counter.
        if (block.group_index().x == 0 && block.thread_index().x == 0) {
            *globalBorrowAny = 0;
        }

        // Erase the per-block borrow arrays.
        if (block.thread_index().x == 0) {
            blockBorrow1[block.group_index().x] = 0;
            blockBorrow2[block.group_index().x] = 0;
        }

        auto *SharkRestrict curSubtract1 = subtractionBorrows1a;
        auto *SharkRestrict curSubtract2 = subtractionBorrows2a;
        auto *SharkRestrict newSubtract1 = subtractionBorrows1b;
        auto *SharkRestrict newSubtract2 = subtractionBorrows2b;

        // === (1) INITIAL SUBTRACTION: Process all digits using a grid-stride loop.
        // Only active blocks (those with group_index().x in [ExecutionBlockBase, ExecutionBlockBase+ExecutionNumBlocks))
        // participate.

        // Compute chunk size for each block.
        const int blockIdx = block.group_index().x - ExecutionBlockBase;

        const int baseSize = nmax / ExecutionNumBlocks;         // integer division
        const int remainder = nmax % ExecutionNumBlocks;          // extra digits to distribute
        const int chunkSize = (blockIdx < remainder) ? (baseSize + 1) : baseSize;
        const int blockStart = blockIdx * baseSize + min(blockIdx, remainder);
        const int blockEnd = blockStart + chunkSize;

        {
            // Each thread in the block processes its assigned indices in the contiguous chunk.
            for (int idx = blockStart + block.thread_index().x;
                idx < blockEnd;
                idx += block.dim_threads().x) {

                uint32_t a1_val = (idx < a1n) ? a1[idx] : 0;
                uint32_t b1_val = (idx < b1n) ? b1[idx] : 0;
                uint32_t a2_val = (idx < a2n) ? a2[idx] : 0;
                uint32_t b2_val = (idx < b2n) ? b2[idx] : 0;

                uint64_t diff1 = (uint64_t)a1_val - b1_val;
                uint64_t diff2 = (uint64_t)a2_val - b2_val;

                uint32_t borrow1 = (a1_val < b1_val) ? 1u : 0u;
                uint32_t borrow2 = (a2_val < b2_val) ? 1u : 0u;

                global_x_diff_abs[idx] = static_cast<uint32_t>(diff1);
                global_y_diff_abs[idx] = static_cast<uint32_t>(diff2);

                curSubtract1[idx] = borrow1;
                curSubtract2[idx] = borrow2;

                // Initialize newSubtract as well
                newSubtract1[idx] = 0;
                newSubtract2[idx] = 0;
            }
        }

        grid.sync();

        // === (2b) Each block's last thread writes its final borrow.
        if (block.thread_index().x == block.dim_threads().x - 1) {
            // Each block processes a contiguous chunk.
            // const int blockStart = (block.group_index().x - ExecutionBlockBase) * block.dim_threads().x;
            // const int blockEnd = blockStart + block.dim_threads().x; // exclusive

            const uint32_t finalBorrow1 = curSubtract1[blockEnd - 1];
            blockBorrow1[block.group_index().x] = finalBorrow1;
            curSubtract1[blockEnd - 1] = 0;

            const uint32_t finalBorrow2 = curSubtract2[blockEnd - 1];
            blockBorrow2[block.group_index().x] = finalBorrow2;
            curSubtract2[blockEnd - 1] = 0;
        }

        grid.sync();  // Ensure all initial differences and borrows are computed

        uint32_t initialBorrowAny = 0;

        // === (2) OUTER LOOP: Propagate borrows across blocks.
        const int MaxPasses = 500; // Adjust as needed.
        int outerPass = 0;
        do {
            uint32_t injection1 = 0, injection2 = 0;

            if (block.thread_index().x == 0) {
                injection1 = (block.group_index().x > ExecutionBlockBase)
                    ? blockBorrow1[block.group_index().x - 1]
                    : 0;
                injection2 = (block.group_index().x > ExecutionBlockBase)
                    ? blockBorrow2[block.group_index().x - 1]
                    : 0;
            }
            block.sync();

            // === (2a) LOCAL PROPAGATION WITHIN THE BLOCK.
            // Iterate blockDim.x times so that a borrow created at the block's start
            // immediately cascades through.

            for (int pass = 0; pass < nmax; ++pass) {

                if (block.thread_index().x == 0) {
                    *sharedBorrowAny = 0;
                }

                block.sync();

                for (int localIdx = blockStart + block.thread_index().x;
                    localIdx < blockEnd;
                    localIdx += block.dim_threads().x) {

                    // For x_diff_abs:
                    uint32_t borrow1;
                    if (block.thread_index().x == 0) {
                        // Only on the first pass do we subtract the injected borrow.
                        if (pass == 0 && localIdx == blockStart) {
                            borrow1 = injection1;
                        } else {
                            if (localIdx == blockStart) {
                                borrow1 = 0;
                            } else {
                                // This path occurs when there are few threads but extra digits.
                                // Thread 0 might iterate the inner loop once, and in that case
                                // propagation needs to happen.
                                borrow1 = curSubtract1[localIdx - 1];
                            }
                        }
                    } else {
                        borrow1 = curSubtract1[localIdx - 1];
                    }
                    const uint64_t newVal1 = (uint64_t)global_x_diff_abs[localIdx] - borrow1;
                    global_x_diff_abs[localIdx] = static_cast<uint32_t>(newVal1 & 0xFFFFFFFFULL);

                    // Last thread in the block that actually did anything
                    if (newVal1 & 0x8000000000000000ULL) {
                        atomicAdd(sharedBorrowAny, 1);

                        if (localIdx == blockEnd - 1) {
                            newSubtract1[localIdx] |= 1u;
                            curSubtract1[localIdx] |= 1u;
                        } else {
                            newSubtract1[localIdx] = 1u;
                        }
                    } else {
                        if (localIdx < blockEnd - 1) {
                            newSubtract1[localIdx] = 0u;
                        }
                    }

                    // For y_diff_abs:
                    uint32_t borrow2;
                    if (block.thread_index().x == 0) {
                        // Only on the first pass do we subtract the injected borrow.
                        if (pass == 0 && localIdx == blockStart) {
                            borrow2 = injection2;
                        } else {
                            if (localIdx == blockStart) {
                                borrow2 = 0;
                            } else {
                                // This path occurs when there are few threads but extra digits.
                                // Thread 0 might iterate the inner loop once, and in that case
                                // propagation needs to happen.
                                borrow2 = curSubtract2[localIdx - 1];
                            }
                        }
                    } else {
                        borrow2 = curSubtract2[localIdx - 1];
                    }
                    const uint64_t newVal2 = (uint64_t)global_y_diff_abs[localIdx] - borrow2;
                    global_y_diff_abs[localIdx] = static_cast<uint32_t>(newVal2 & 0xFFFFFFFFULL);

                    // Last thread in the block that actually did anything
                    if (newVal2 & 0x8000000000000000ULL) {
                        atomicAdd(sharedBorrowAny, 1);

                        if (localIdx == blockEnd - 1) {
                            newSubtract2[localIdx] |= 1u;
                            curSubtract2[localIdx] |= 1u;
                        } else {
                            newSubtract2[localIdx] = 1u;
                        }
                    } else {
                        if (localIdx < blockEnd - 1) {
                            newSubtract2[localIdx] = 0u;
                        }
                    }
                }

                // Swap curSubtract and newSubtract.
                auto *SharkRestrict tmp = curSubtract1;
                curSubtract1 = newSubtract1;
                newSubtract1 = tmp;

                tmp = curSubtract2;
                curSubtract2 = newSubtract2;
                newSubtract2 = tmp;

                block.sync();

                auto tmpBorrow = *sharedBorrowAny;

                block.sync();

                if (tmpBorrow == 0)
                    break;  // no new borrows
            }
            grid.sync();

            // === (2b) Each block's last thread writes its final borrow.
            if (block.thread_index().x == block.dim_threads().x - 1) {
                const uint32_t finalBorrow1 = curSubtract1[blockEnd - 1];
                blockBorrow1[block.group_index().x] = finalBorrow1;
                curSubtract1[blockEnd - 1] = 0;
                newSubtract1[blockEnd - 1] = 0;

                const uint32_t finalBorrow2 = curSubtract2[blockEnd - 1];
                blockBorrow2[block.group_index().x] = finalBorrow2;
                curSubtract2[blockEnd - 1] = 0;
                newSubtract2[blockEnd - 1] = 0;
            }
            grid.sync();

            // === (2c) Global aggregation: One designated block sums the per-block borrows.
            if (block.group_index().x == ExecutionBlockBase &&
                block.thread_index().x == 0) {

                uint32_t totalBorrow = 0;
                for (int i = ExecutionBlockBase; i < ExecutionBlockBase + ExecutionNumBlocks; ++i) {
                    totalBorrow += blockBorrow1[i];
                    totalBorrow += blockBorrow2[i];
                }

                atomicAdd(globalBorrowAny, totalBorrow);  // Overwrite with the new total.
            }
            grid.sync();

            uint32_t tempCopyGlobalBorrowAny = *globalBorrowAny;
            if (tempCopyGlobalBorrowAny == initialBorrowAny)
                break;  // no new borrows --> done

            grid.sync();

            initialBorrowAny = tempCopyGlobalBorrowAny;
            outerPass++;
        } while (outerPass < MaxPasses);

        grid.sync();  // Final grid sync to guarantee all blocks are done.


    } else { //////////////////////////////////////////////////////////////////////

        // Compute maximum digit count.
        constexpr int n1max = (a1n > b1n) ? a1n : b1n;
        constexpr int n2max = (a2n > b2n) ? a2n : b2n;
        constexpr int nmax = (n1max > n2max) ? n1max : n2max;

        // For one block, block.group_index().x == ExecutionBlockBase.
        // Define a simple block-stride loop over the contiguous chunk: [0, nmax).
        const int tid = block.thread_index().x;
        const int stride = block.dim_threads().x;
        const int blockStart = 0;
        const int blockEnd = nmax; // all digits in [0, nmax)

        auto *SharkRestrict curSubtract1 = subtractionBorrows1a;
        auto *SharkRestrict curSubtract2 = subtractionBorrows2a;
        auto *SharkRestrict newSubtract1 = subtractionBorrows1b;
        auto *SharkRestrict newSubtract2 = subtractionBorrows2b;

        // INITIAL SUBTRACTION: each thread processes its assigned digits.
        for (int idx = blockStart + tid; idx < blockEnd; idx += stride) {
            uint32_t a1_val = (idx < a1n) ? a1[idx] : 0;
            uint32_t b1_val = (idx < b1n) ? b1[idx] : 0;
            uint32_t a2_val = (idx < a2n) ? a2[idx] : 0;
            uint32_t b2_val = (idx < b2n) ? b2[idx] : 0;

            uint64_t diff1 = (uint64_t)a1_val - b1_val;
            uint64_t diff2 = (uint64_t)a2_val - b2_val;

            uint32_t borrow1 = (a1_val < b1_val) ? 1u : 0u;
            uint32_t borrow2 = (a2_val < b2_val) ? 1u : 0u;

            global_x_diff_abs[idx] = static_cast<uint32_t>(diff1);
            global_y_diff_abs[idx] = static_cast<uint32_t>(diff2);

            curSubtract1[idx] = borrow1;
            curSubtract2[idx] = borrow2;
        }

        block.sync();

        // LOCAL PROPAGATION: For each pass, each thread processes its indices in [blockStart, blockEnd)
        // using a block-stride loop.
        // We run (blockEnd - blockStart + 1) passes to ensure complete propagation.

        for (int pass = 0; pass < (blockEnd - blockStart + 1); ++pass) {
            if (block.thread_index().x == 0) {
                *sharedBorrowAny = 0;
            }

            block.sync();

            for (int idx = blockStart + tid; idx < blockEnd; idx += stride) {
                // For the first digit, there is no previous digit, so the borrow is 0.
                uint32_t borrow1 = (idx == blockStart) ? 0 : curSubtract1[idx - 1];
                uint64_t newVal1 = static_cast<uint64_t>(global_x_diff_abs[idx]) - borrow1;
                global_x_diff_abs[idx] = static_cast<uint32_t>(newVal1 & 0xFFFFFFFFULL);

                if (newVal1 & 0x8000000000000000ULL) {
                    newSubtract1[idx] = 1;
                    atomicAdd(sharedBorrowAny, 1);
                } else {
                    newSubtract1[idx] = 0;
                }

                uint32_t borrow2 = (idx == blockStart) ? 0 : curSubtract2[idx - 1];
                uint64_t newVal2 = static_cast<uint64_t>(global_y_diff_abs[idx]) - borrow2;
                global_y_diff_abs[idx] = static_cast<uint32_t>(newVal2 & 0xFFFFFFFFULL);

                if (newVal2 & 0x8000000000000000ULL) {
                    newSubtract2[idx] = 1;
                    atomicAdd(sharedBorrowAny, 1);
                } else {
                    newSubtract2[idx] = 0;
                }
            }

            // All threads synchronize after processing the entire chunk.
            block.sync();

            auto tmpBorrow = *sharedBorrowAny;
            block.sync();
            if (tmpBorrow == 0) {
                break;  // no new borrows
            }

            // Swap curSubtract and newSubtract.
            auto *SharkRestrict tmp = curSubtract1;
            curSubtract1 = newSubtract1;
            newSubtract1 = tmp;

            tmp = curSubtract2;
            curSubtract2 = newSubtract2;
            newSubtract2 = tmp;
        }
    }
}



// Function to perform addition with carry
__device__ SharkForceInlineReleaseOnly static void
Add128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    result_low = a_low + b_low;
    uint64_t carry = (result_low < a_low) ? 1 : 0;
    result_high = a_high + b_high + carry;
}

__device__ SharkForceInlineReleaseOnly static void
Subtract128(
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high,
    uint64_t &result_low, uint64_t &result_high) {

    uint64_t borrow = 0;

    // Subtract low parts
    result_low = a_low - b_low;
    borrow = (a_low < b_low) ? 1 : 0;

    // Subtract high parts with borrow
    result_high = a_high - b_high - borrow;
}

template<class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly static void
SerialCarryPropagationThread0 (
    uint64_t *SharkRestrict shared_data,
    cg::grid_group &grid,
    cg::thread_block &block,
    int thread_start_idx,
    int thread_end_idx,
    const uint64_t *SharkRestrict final128XX,
    const uint64_t *SharkRestrict final128XY,
    const uint64_t *SharkRestrict final128YY,
    uint64_t *SharkRestrict resultXX,
    uint64_t *SharkRestrict resultXY,
    uint64_t *SharkRestrict resultYY)
{
    constexpr int total_result_digits = 2 * SharkFloatParams::GlobalNumUint32;

    // Stateless lambda to process one component (XX, XY, or YY)
    auto ProcessComponent = [](
        const uint64_t *final128,
        uint64_t *result,
        int total_digits) -> uint64_t {

            uint64_t local_carry = 0;

            // Process all digits serially from lowest to highest
            for (int idx = 0; idx < total_digits; ++idx) {
                int sum_low_idx = idx * 2;
                int sum_high_idx = sum_low_idx + 1;

                uint64_t sum_low = final128[sum_low_idx];
                uint64_t sum_high = final128[sum_high_idx];

                // Add local carry to sum_low
                bool new_sum_low_negative = false;
                uint64_t new_sum_low = sum_low + local_carry;

                // Extract one 32-bit digit from new_sum_low
                auto digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);
                result[idx] = digit;

                bool local_carry_negative = ((local_carry & (1ULL << 63)) != 0);
                local_carry = 0ULL;

                if (!local_carry_negative && new_sum_low < sum_low) {
                    local_carry = 1ULL << 32;
                } else if (local_carry_negative && new_sum_low > sum_low) {
                    new_sum_low_negative = (new_sum_low & 0x8000'0000'0000'0000ULL) != 0;
                }

                // Update local_carry
                if (new_sum_low_negative) {
                    // Shift sum_high by 32 bits and add carry_from_low
                    uint64_t upper_new_sum_low = new_sum_low >> 32;
                    upper_new_sum_low |= 0xFFFF'FFFF'0000'0000ULL;
                    local_carry += upper_new_sum_low;
                    local_carry += sum_high << 32;
                } else {
                    local_carry += new_sum_low >> 32;
                    local_carry += sum_high << 32;
                }
            }

            return local_carry;
        };

    // Process each component using the lambda
    uint64_t final_carry_xx = ProcessComponent(final128XX, resultXX, total_result_digits);
    uint64_t final_carry_xy = ProcessComponent(final128XY, resultXY, total_result_digits);
    uint64_t final_carry_yy = ProcessComponent(final128YY, resultYY, total_result_digits);

    // Store final carries
    if (final_carry_xx > 0) {
        resultXX[total_result_digits] = static_cast<uint32_t>(final_carry_xx & 0xFFFFFFFFULL);
    }
    if (final_carry_xy > 0) {
        resultXY[total_result_digits] = static_cast<uint32_t>(final_carry_xy & 0xFFFFFFFFULL);
    }
    if (final_carry_yy > 0) {
        resultYY[total_result_digits] = static_cast<uint32_t>(final_carry_yy & 0xFFFFFFFFULL);
    }
}

template<class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly static void
SerialCarryPropagation (
    uint64_t *SharkRestrict shared_data,
    cg::grid_group &grid,
    cg::thread_block &block,
    int thread_start_idx,
    int thread_end_idx,
    const uint64_t *SharkRestrict final128XX,
    const uint64_t *SharkRestrict final128XY,
    const uint64_t *SharkRestrict final128YY,
    uint64_t *SharkRestrict resultXX,
    uint64_t *SharkRestrict resultXY,
    uint64_t *SharkRestrict resultYY,
    uint64_t *SharkRestrict block_carry_outs,
    uint64_t *SharkRestrict /*globalCarryCheck*/) {

    // Only execute on a single thread to maintain serial behavior
    if (block.thread_index().x == 0 && block.group_index().x == 0) {
        SerialCarryPropagationThread0<SharkFloatParams>(
            shared_data,
            grid,
            block,
            thread_start_idx,
            thread_end_idx,
            final128XX,
            final128XY,
            final128YY,
            resultXX,
            resultXY,
            resultYY);
    }
}

static __device__ int
CarryGlobalToIndex (
    const bool PriorIndex,
    const int block_idx)
{
    return (block_idx - (PriorIndex ? 1 : 0)) * 3;
}

static __device__ int
CarrySharedToIndex (
    const bool PriorIndex,
    const int thread_idx)
{
    return (thread_idx - (PriorIndex ? 1 : 0)) * 3;
}

template<class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly static void
CarryPropagation (
    uint64_t *SharkRestrict shared_data,
    cg::grid_group &grid,
    cg::thread_block &block,
    int thread_start_idx,
    int thread_end_idx,
    const uint64_t *SharkRestrict final128XX,
    const uint64_t *SharkRestrict final128XY,
    const uint64_t *SharkRestrict final128YY,
    uint64_t *SharkRestrict resultXX,
    uint64_t *SharkRestrict resultXY,
    uint64_t *SharkRestrict resultYY,
    uint64_t *SharkRestrict block_carry_outs,
    uint64_t *SharkRestrict globalCarryCheck) {

    auto *SharkRestrict shared_carries = shared_data;

    // TODO: Ensure we allocate a minimum amount of shared memory to support shared_carries use
    // TODO: Ensure we allocate a minimum amount of global memory to support block_carry_outs use

    // First Pass: Process convolution results to compute initial digits and local carries
    // Initialize local carry
    uint64_t local_carry_xx = 0;
    uint64_t local_carry_xy = 0;
    uint64_t local_carry_yy = 0;

    // TODO: Remove this:
    constexpr auto sharedMem = CalculateMultiplySharedMemorySize<SharkFloatParams>() / sizeof(uint64_t);
    {
        // Erase shared_carries
        for (int i = block.thread_index().x; i < sharedMem; i += block.dim_threads().x) {
            shared_carries[i] = 0;
        }
    }

    grid.sync();

    const auto MaxBlocks = grid.group_dim().x;
    const auto MaxThreads = block.dim_threads().x;

    // Constants and offsets
    constexpr int MaxPasses = 5000; // Maximum number of carry propagation passes
    constexpr int total_result_digits = 2 * SharkFloatParams::GlobalNumUint32;

    uint64_t *carries_remaining_global = globalCarryCheck;

    for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
        const int sum_low_idx = idx * 2;
        const int sum_high_idx = sum_low_idx + 1;

        const uint64_t xx_sum_low = final128XX[sum_low_idx];     // Lower 64 bits
        const uint64_t xx_sum_high = final128XX[sum_high_idx];   // Higher 64 bits

        const uint64_t xy_sum_low = final128XY[sum_low_idx];     // Lower 64 bits
        const uint64_t xy_sum_high = final128XY[sum_high_idx];   // Higher 64 bits

        const uint64_t yy_sum_low = final128YY[sum_low_idx];     // Lower 64 bits
        const uint64_t yy_sum_high = final128YY[sum_high_idx];   // Higher 64 bits

        // Add local carry to sum_low
        auto LocalCarry = [](
            uint64_t &local_carry,
            uint64_t sum_low,
            uint64_t sum_high,
            uint64_t *result,
            int idx) {

            bool new_sum_low_negative = false;
            const uint64_t new_sum_low = sum_low + local_carry;

            // Extract one 32-bit digit from new_sum_low
            const auto digit = static_cast<uint32_t>(new_sum_low & 0xFFFFFFFFULL);
            result[idx] = digit;

            const bool local_carry_negative = ((local_carry & (1ULL << 63)) != 0);
            local_carry = 0ULL;

            if (!local_carry_negative && new_sum_low < sum_low) {
                local_carry = 1ULL << 32;
            } else if (local_carry_negative && new_sum_low > sum_low) {
                new_sum_low_negative = (new_sum_low & 0x8000'0000'0000'0000) != 0;
            }

            // Update local_carry
            if (new_sum_low_negative) {
                // Shift sum_high by 32 bits and add carry_from_low
                uint64_t upper_new_sum_low = new_sum_low >> 32;
                upper_new_sum_low |= 0xFFFF'FFFF'0000'0000;
                local_carry += upper_new_sum_low;
                local_carry += sum_high << 32;
            } else {
                local_carry += new_sum_low >> 32;
                local_carry += sum_high << 32;
            }
        };

        // Process xx_sum
        LocalCarry(local_carry_xx, xx_sum_low, xx_sum_high, resultXX, idx);

        // Process xy_sum
        LocalCarry(local_carry_xy, xy_sum_low, xy_sum_high, resultXY, idx);

        // Process yy_sum
        LocalCarry(local_carry_yy, yy_sum_low, yy_sum_high, resultYY, idx);
    }

    const auto threadIndexInBlock = block.thread_index().x;
    const auto blockIndexInGrid = block.group_index().x;
    const auto carrySharedToIndex = CarrySharedToIndex(
        false,
        threadIndexInBlock);
    if (threadIndexInBlock == SharkFloatParams::GlobalThreadsPerBlock - 1) {
        const auto carryGlobalToIndex = CarryGlobalToIndex(
            false,
            block.group_index().x);

        block_carry_outs[carryGlobalToIndex + 0] = local_carry_xx;
        block_carry_outs[carryGlobalToIndex + 1] = local_carry_xy;
        block_carry_outs[carryGlobalToIndex + 2] = local_carry_yy;

        shared_carries[carrySharedToIndex + 0] = local_carry_xx;
        shared_carries[carrySharedToIndex + 1] = local_carry_xy;
        shared_carries[carrySharedToIndex + 2] = local_carry_yy;
    } else {
        shared_carries[carrySharedToIndex + 0] = local_carry_xx;
        shared_carries[carrySharedToIndex + 1] = local_carry_xy;

        shared_carries[carrySharedToIndex + 2] = local_carry_yy;
    }

    // Inter-Block Carry Propagation
    int pass = 0;

    do {
        // Synchronize all blocks
        grid.sync();

        // Zero out the global carry count for the current pass
        if (block.group_index().x == 0 && block.thread_index().x == 0) {
            *carries_remaining_global = 0;
        }

        // Get carry-in from the previous block
        // The warning here is about the constant template parameter not being used in
        // the parameter list.  I don't understand why it's even a warning?  Maybe because
        // it cannot be inferred?  It's clearly being used in the body?

#pragma nv_diag_suppress 445
        auto LocalCarryIn = []<int XX_XY_YY>(
            cg::thread_block & block,
            const int MaxBlocks,
            const int MaxThreads,
            uint64_t &local_carry,
            uint64_t *SharkRestrict block_carry_outs,
            uint64_t *SharkRestrict shared_carries) {

            local_carry = 0;
            if (block.thread_index().x == 0 && block.group_index().x > 0) {
                const auto block_carry_outs_idx = CarryGlobalToIndex(
                    true,
                    block.group_index().x) + XX_XY_YY;
                local_carry = block_carry_outs[block_carry_outs_idx];
            } else {
                if (block.thread_index().x > 0) {
                    const auto shared_carries_idx = CarrySharedToIndex(
                        true,
                        block.thread_index().x) + XX_XY_YY;
                    local_carry = shared_carries[shared_carries_idx];
                }
            }
            };
#pragma nv_diag_default 445

        // Initialize local carry for this pass
        LocalCarryIn.template operator()<0>(
            block,
            MaxBlocks,
            MaxThreads,
            local_carry_xx,
            block_carry_outs,
            shared_carries);
        LocalCarryIn.template operator()<1>(
            block,
            MaxBlocks,
            MaxThreads,
            local_carry_xy,
            block_carry_outs,
            shared_carries);
        LocalCarryIn.template operator()<2>(
            block,
            MaxBlocks,
            MaxThreads,
            local_carry_yy,
            block_carry_outs,
            shared_carries);

        auto LocalCarry = [](uint64_t *SharkRestrict resultXY, uint64_t &local_carry, int idx) {
            // Read the previously stored digit
            const uint32_t digit = resultXY[idx];

            // Add local_carry to digit
            const uint64_t sum = static_cast<uint64_t>(digit) + local_carry;

            // Update digit
            resultXY[idx] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);

            local_carry = 0;

            // Check negativity of the 64-bit sum
            // If "sum" is negative, its top bit is set. 
            const bool sum_is_negative = ((sum & (1ULL << 63)) != 0ULL);
            if (sum_is_negative) {
                // sign-extend the top 32 bits
                uint64_t upper_bits = (sum >> 32);
                upper_bits |= 0xFFFF'FFFF'0000'0000ULL;  // set top 32 bits to 1
                local_carry += upper_bits;               // incorporate sign-extended bits
            } else {
                // normal path: just add top 32 bits
                local_carry += (sum >> 32);
            }
            };

        // Each thread processes its assigned digits
        for (int idx = thread_start_idx; idx < thread_end_idx; ++idx) {
            // Process xx_sum
            LocalCarry(resultXX, local_carry_xx, idx);

            // Process xy_sum
            LocalCarry(resultXY, local_carry_xy, idx);

            // Process yy_sum
            LocalCarry(resultYY, local_carry_yy, idx);
        }

        // TODO should we interleave each of these instead of separating them?  probably?
        const auto carrySharedToCurIndex = CarrySharedToIndex(
            false,
            block.thread_index().x);
        const auto carryGlobalToCurIndex = CarryGlobalToIndex(
            false,
            block.group_index().x);

        shared_carries[carrySharedToCurIndex + 0] = local_carry_xx;
        shared_carries[carrySharedToCurIndex + 1] = local_carry_xy;
        shared_carries[carrySharedToCurIndex + 2] = local_carry_yy;

        // This sync is required to address the *carries_remaining_global = 0; assignment
        // racing with the atomicAdd below.
        grid.sync();

        // The block's carry-out is the carry from the last thread
        const auto temp_xx = shared_carries[carrySharedToCurIndex + 0];
        if (block.thread_index().x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
            block_carry_outs[carryGlobalToCurIndex + 0] = temp_xx;
        }

        const auto temp_xy = shared_carries[carrySharedToCurIndex + 1];
        if (block.thread_index().x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
            block_carry_outs[carryGlobalToCurIndex + 1] = temp_xy;
        }

        const auto temp_yy = shared_carries[carrySharedToCurIndex + 2];
        if (block.thread_index().x == SharkFloatParams::GlobalThreadsPerBlock - 1) {
            block_carry_outs[carryGlobalToCurIndex + 2] = temp_yy;
        }

        if (temp_xx != 0 || temp_xy != 0 || temp_yy != 0) {
            atomicAdd(carries_remaining_global, 1);
        }

        // Synchronize all blocks before checking if carries remain
        grid.sync();

        // If no carries remain, exit the loop
        if (*carries_remaining_global == 0) {
            break;
        }

        pass++;
    } while (pass < MaxPasses);

    // ---- Handle Final Carry-Out ----

    // TODO is this correct?  remove?
    // Handle final carry-out
    if (block.thread_index().x == 0 && block.group_index().x == grid.dim_blocks().x - 1) {
        const auto carryGlobalToIndex = CarryGlobalToIndex(
            false,
            block.group_index().x);
        const auto block_idx_xx = carryGlobalToIndex + 0;
        uint64_t final_carry_xx = block_carry_outs[block_idx_xx];
        if (final_carry_xx > 0) {
            // Store the final carry as an additional digit
            resultXX[total_result_digits] = static_cast<uint32_t>(final_carry_xx & 0xFFFFFFFFULL);
        }

        const auto block_idx_xy = carryGlobalToIndex + 1;
        uint64_t final_carry_xy = block_carry_outs[block_idx_xy];
        if (final_carry_xy > 0) {
            // Store the final carry as an additional digit
            resultXY[total_result_digits] = static_cast<uint32_t>(final_carry_xy & 0xFFFFFFFFULL);
        }

        const auto block_idx_yy = carryGlobalToIndex + 2;
        uint64_t final_carry_yy = block_carry_outs[block_idx_yy];
        if (final_carry_yy > 0) {
            // Store the final carry as an additional digit
            resultYY[total_result_digits] = static_cast<uint32_t>(final_carry_yy & 0xFFFFFFFFULL);
        }
    }
}

// Look for CalculateKaratsubaFrameSize and ScratchMemoryArraysForMultiply
// and make sure the number of NewN arrays we're using here fits within that limit.
// The list here should go up to ScratchMemoryArraysForMultiply.
static_assert(AdditionalUInt64PerFrame == 256, "See below");
#define DefineTempProductsOffsets(CallIndex) \
    const int threadIdxGlobal = block.group_index().x * SharkFloatParams::GlobalThreadsPerBlock + block.thread_index().x; \
    constexpr int TestMultiplier = 1; \
    constexpr auto Multiplies_offset = AdditionalGlobalSyncSpace; \
    constexpr auto Checksum_offset = Multiplies_offset + AdditionalGlobalMultipliesPerThread; \
    /* Start from AdditionalUInt64PerFrame next, global state is above */ \
    constexpr auto CallOffset = Checksum_offset + AdditionalGlobalChecksumSpace + CallIndex * CalculateKaratsubaFrameSize<SharkFloatParams>(); \
    constexpr auto TempBase = 0; \
    constexpr auto TempBaseOffset = TempBase + CallOffset; \
    constexpr auto Z0_offsetXX = TempBaseOffset + AdditionalUInt64PerFrame + CalcAlign16Bytes64BitIndex(TempBaseOffset + AdditionalUInt64PerFrame); /* 0 */ \
    constexpr auto Z0_offsetXY = Z0_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier);            /* 4 */ \
    constexpr auto Z0_offsetYY = Z0_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier);            /* 8 */ \
    constexpr auto Z2_offsetXX = Z0_offsetYY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier);       /* 12 */ \
    constexpr auto Z2_offsetXY = Z2_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 16 */ \
    constexpr auto Z2_offsetYY = Z2_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 20 */ \
    constexpr auto Z1_temp_offsetXX = Z2_offsetYY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 24 */ \
    constexpr auto Z1_temp_offsetXY = Z1_temp_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 28 */ \
    constexpr auto Z1_temp_offsetYY = Z1_temp_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 32 */ \
    constexpr auto Z1_offsetXX = Z1_temp_offsetYY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 36 */ \
    constexpr auto Z1_offsetXY = Z1_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 40 */ \
    constexpr auto Z1_offsetYY = Z1_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 44 */ \
    constexpr auto Convolution_offsetXX = Z1_offsetYY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 48 */ \
    constexpr auto Convolution_offsetXY = Convolution_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 52 */ \
    constexpr auto Convolution_offsetYY = Convolution_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 56 */ \
    constexpr auto Result_offsetXX = Convolution_offsetYY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 60 */ \
    constexpr auto Result_offsetXY = Result_offsetXX + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 64 */ \
    constexpr auto Result_offsetYY = Result_offsetXY + 4 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(4 * NewN * TestMultiplier); /* 68 */ \
    constexpr auto XDiff_offset = Result_offsetYY + 2 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(2 * NewN * TestMultiplier); /* 70 */ \
    constexpr auto YDiff_offset = XDiff_offset + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 71 */ \
    constexpr auto GlobalCarryOffset = YDiff_offset + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 72 */ \
    constexpr auto SubtractionOffset1 = GlobalCarryOffset + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 73 */ \
    constexpr auto SubtractionOffset2 = SubtractionOffset1 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 74 */ \
    constexpr auto SubtractionOffset3 = SubtractionOffset2 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 75 */ \
    constexpr auto SubtractionOffset4 = SubtractionOffset3 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 76 */ \
    constexpr auto SubtractionOffset5 = SubtractionOffset4 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 77 */ \
    constexpr auto SubtractionOffset6 = SubtractionOffset5 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* 78 */ \
    constexpr auto CarryInsOffset = SubtractionOffset6 + 1 * NewN * TestMultiplier + CalcAlign16Bytes64BitIndex(1 * NewN * TestMultiplier); /* requires 3xNewN 79 */ \
    constexpr auto CarryInsEnd = CarryInsOffset + 3 * NewN + CalcAlign16Bytes64BitIndex(3 * NewN); \


#define TempProductsGlobals(TempBase, CallIndex) \
    constexpr auto BorrowGlobalOffset = 0; \
    constexpr auto BorrowBlockLevelOffset1 = MaxBlocks; \
    constexpr auto BorrowBlockLevelOffset2 = MaxBlocks * 2; \

#define DefineExtraDefinitions() \
    const auto RelativeBlockIndex = block.group_index().x - ExecutionBlockBase; \
    constexpr int total_result_digits = 2 * NewN; \
    constexpr auto digits_per_block = NewN * 2 / ExecutionNumBlocks; \
    const auto block_start_idx = block.group_index().x * digits_per_block; \
    const auto block_end_idx = min(block_start_idx + digits_per_block, total_result_digits); \
    const int digits_per_thread = (digits_per_block + block.dim_threads().x - 1) / block.dim_threads().x; \
    const int thread_start_idx = block_start_idx + block.thread_index().x * digits_per_thread; \
    const int thread_end_idx = min(thread_start_idx + digits_per_thread, block_end_idx);

#define DefineCarryDefinitions() \
    constexpr auto total_result_digits = 2 * NewN; \
    constexpr auto per_thread_multiplier = 1; \
    constexpr auto total_threads = SharkFloatParams::GlobalThreadsPerBlock * SharkFloatParams::GlobalNumBlocks * per_thread_multiplier; \
    const int digits_per_thread = (total_result_digits + total_threads - 1) / total_threads; \
    const int thread_idx = block.group_index().x * SharkFloatParams::GlobalThreadsPerBlock + block.thread_index().x; \
    const int thread_start_idx = thread_idx * digits_per_thread; \
    const int thread_end_idx = min(thread_start_idx + digits_per_thread, total_result_digits);


template<
    class SharkFloatParams,
    int RecursionDepth,
    int CallIndex,
    DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly static void
EraseCurrentDebugState(
    RecordIt record,
    DebugState<SharkFloatParams> *debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugStates[CallIndex * maxPurposes + curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    int RecursionDepth,
    int CallIndex,
    DebugStatePurpose Purpose,
    typename ArrayType>
__device__ SharkForceInlineReleaseOnly static void
StoreCurrentDebugState (
    RecordIt record,
    UseConvolution useConvolution,
    DebugState<SharkFloatParams> *debugStates,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize)
{
    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugStates[CallIndex * maxPurposes + curPurpose].Reset(
        record, useConvolution, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

enum class ConditionalAccess {
    False,
    True
};

// Unified scalar accumulator used for BOTH prologue and epilogue
template<
    class SharkFloatParams,
    ConditionalAccess Cond
>
__device__ SharkForceInlineReleaseOnly static void
accumulate_scalar_span (
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
    int i_lo, int i_hi, int k,
    const uint32_t *__restrict__ aDigits_base,
    const uint32_t *__restrict__ bDigits_base,
    int a_offset, int b_offset,
    const uint32_t *__restrict__ x_diff_abs,
    const uint32_t *__restrict__ y_diff_abs,
    uint64_t &xx_low, uint64_t &xx_high,
    uint64_t &xy_low, uint64_t &xy_high,
    uint64_t &yy_low, uint64_t &yy_high) {
    if (i_lo > i_hi) return;
#pragma unroll
    for (int idx = i_lo; idx <= i_hi; ++idx) {
        const int idx2 = k - idx;
        uint64_t xx_a, xx_b, xy_a, xy_b, yy_a, yy_b;
        if constexpr (Cond == ConditionalAccess::True) {
            xx_a = x_diff_abs[idx];     xx_b = x_diff_abs[idx2];
            xy_a = x_diff_abs[idx];     xy_b = y_diff_abs[idx2];
            yy_a = y_diff_abs[idx];     yy_b = y_diff_abs[idx2];
        } else {
            xx_a = aDigits_base[idx + a_offset];
            xx_b = aDigits_base[idx2 + a_offset];
            xy_a = aDigits_base[idx + a_offset];
            xy_b = bDigits_base[idx2 + b_offset];
            yy_a = bDigits_base[idx + b_offset];
            yy_b = bDigits_base[idx2 + b_offset];
        }
        uint64_t p;
        p = xx_a * xx_b; xx_low += p; if (xx_low < p) xx_high += 1;
        p = xy_a * xy_b; xy_low += p; if (xy_low < p) xy_high += 1;
        p = yy_a * yy_b; yy_low += p; if (yy_low < p) yy_high += 1;

        DebugMultiplyIncrement<SharkFloatParams>(debugMultiplyCounts, grid, block, 3);
    }
}

__device__ __forceinline__ int warpMinI32(int v) {
    unsigned m = __activemask();
    v = min(v, __shfl_down_sync(m, v, 16));
    v = min(v, __shfl_down_sync(m, v, 8));
    v = min(v, __shfl_down_sync(m, v, 4));
    v = min(v, __shfl_down_sync(m, v, 2));
    v = min(v, __shfl_down_sync(m, v, 1));
    return __shfl_sync(m, v, 0);
}

#include "MultiplyHelperVectorLoads.cuh"
#include "MultiplyHelperGlobalUnaligned.cuh"

template<
    class SharkFloatParams,
    int BatchSize,
    ConditionalAccess UseConditionalAccess,
    int RecursionDepth,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void
ProcessConvolutionBatch (
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
    const int RelativeBlockIndex,
    const int outerIteration,
    const int k,
    const int total_k,
    const int i_start,
    const int i_end,
    const int n_limit,
    const uint32_t *aDigits_base,
    const uint32_t *bDigits_base,
    const int a_offset,
    const int b_offset,
    uint64_t &xx_sum_low,
    uint64_t &xx_sum_high,
    uint64_t &xy_sum_low,
    uint64_t &xy_sum_high,
    uint64_t &yy_sum_low,
    uint64_t &yy_sum_high,
    uint32_t *shared_data,
    const uint32_t *x_diff_abs = nullptr,
    const uint32_t *y_diff_abs = nullptr) {


    // ---------------- scalar-only fast path when BatchSize==1 ----------------
    if constexpr (BatchSize == 1) {
        accumulate_scalar_span<SharkFloatParams, UseConditionalAccess>(
            grid,
            block,
            debugMultiplyCounts,
            i_start, i_end, k,
            aDigits_base, bDigits_base, a_offset, b_offset,
            x_diff_abs, y_diff_abs,
            xx_sum_low, xx_sum_high,
            xy_sum_low, xy_sum_high,
            yy_sum_low, yy_sum_high);
        return;
    }

    // ---------------- main path (BatchSize>=2; vectorized steady-state when 4) ----------------
    int i = i_start;

    // If all data already in shared (HpShark::LoadAllInShared), we can skip alignment steering
    // and simply use the 4-stage register pipeline with scalar loads from shared.
    if constexpr (HpShark::LoadAllInShared) {
        // Prologue: scalar until we have >= BatchSize items
        const int pro_end = min(i_end, i + ((i_end - i + 1) % BatchSize) - 1);
        if (pro_end >= i) {
            accumulate_scalar_span<SharkFloatParams, UseConditionalAccess>(
                grid,
                block,
                debugMultiplyCounts,
                i, pro_end, k,
                aDigits_base, bDigits_base, a_offset, b_offset,
                x_diff_abs, y_diff_abs,
                xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high);
            i = pro_end + 1;
        }

        // Steady-state: consume as many full batches as possible (scalar loads from shared)
        while (i + BatchSize - 1 <= i_end) {
            const int remaining = i_end - i + 1;
            const int nFull = remaining / BatchSize;    // >=1

            // Four fixed buffers (compile-time indices)
            uint32_t ax0[BatchSize], bx0[BatchSize], ay0[BatchSize], by0[BatchSize], cy0[BatchSize], dy0[BatchSize];
            uint32_t ax1[BatchSize], bx1[BatchSize], ay1[BatchSize], by1[BatchSize], cy1[BatchSize], dy1[BatchSize];
            uint32_t ax2[BatchSize], bx2[BatchSize], ay2[BatchSize], by2[BatchSize], cy2[BatchSize], dy2[BatchSize];
            uint32_t ax3[BatchSize], bx3[BatchSize], ay3[BatchSize], by3[BatchSize], cy3[BatchSize], dy3[BatchSize];

            // scalar loaders (from shared inputs x_diff_abs/y_diff_abs or a/b + offsets)
            auto load_buf_shared = [&](int base_i, uint32_t(&ax)[BatchSize], uint32_t(&bx)[BatchSize],
                uint32_t(&ay)[BatchSize], uint32_t(&by)[BatchSize],
                uint32_t(&cy)[BatchSize], uint32_t(&dy)[BatchSize]) {
#pragma unroll
                    for (int j = 0; j < BatchSize; ++j) {
                        const int idx = base_i + j;
                        const int idx2 = k - idx;
                        if constexpr (UseConditionalAccess == ConditionalAccess::True) {
                            ax[j] = x_diff_abs[idx];
                            bx[j] = x_diff_abs[idx2];
                            ay[j] = x_diff_abs[idx];
                            by[j] = y_diff_abs[idx2];
                            cy[j] = y_diff_abs[idx];
                            dy[j] = y_diff_abs[idx2];
                        } else {
                            ax[j] = aDigits_base[idx + a_offset];
                            bx[j] = aDigits_base[idx2 + a_offset];
                            ay[j] = aDigits_base[idx + a_offset];
                            by[j] = bDigits_base[idx2 + b_offset];
                            cy[j] = bDigits_base[idx + b_offset];
                            dy[j] = bDigits_base[idx2 + b_offset];
                        }
                    }
                };

            auto compute_buf = [&](uint32_t(&ax)[BatchSize], uint32_t(&bx)[BatchSize],
                uint32_t(&ay)[BatchSize], uint32_t(&by)[BatchSize],
                uint32_t(&cy)[BatchSize], uint32_t(&dy)[BatchSize]) {
#pragma unroll
                    for (int j = 0; j < BatchSize; ++j) {
                        const uint64_t xx_a = (uint64_t)ax[j], xx_b = (uint64_t)bx[j];
                        const uint64_t xy_a = (uint64_t)ay[j], xy_b = (uint64_t)by[j];
                        const uint64_t yy_a = (uint64_t)cy[j], yy_b = (uint64_t)dy[j];
                        uint64_t p;
                        p = xx_a * xx_b; xx_sum_low += p; if (xx_sum_low < p) xx_sum_high += 1;
                        p = xy_a * xy_b; xy_sum_low += p; if (xy_sum_low < p) xy_sum_high += 1;
                        p = yy_a * yy_b; yy_sum_low += p; if (yy_sum_low < p) yy_sum_high += 1;

                        DebugMultiplyIncrement<SharkFloatParams>(debugMultiplyCounts, grid, block, 3);
                    }
                };

            if (nFull == 1) {
                load_buf_shared(i, ax0, bx0, ay0, by0, cy0, dy0);
                compute_buf(ax0, bx0, ay0, by0, cy0, dy0);
                i += BatchSize;
            } else if (nFull == 2) {
                load_buf_shared(i, ax0, bx0, ay0, by0, cy0, dy0);
                load_buf_shared(i + BatchSize, ax1, bx1, ay1, by1, cy1, dy1);
                compute_buf(ax0, bx0, ay0, by0, cy0, dy0);
                compute_buf(ax1, bx1, ay1, by1, cy1, dy1);
                i += 2 * BatchSize;
            } else if (nFull == 3) {
                load_buf_shared(i, ax0, bx0, ay0, by0, cy0, dy0);
                load_buf_shared(i + BatchSize, ax1, bx1, ay1, by1, cy1, dy1);
                load_buf_shared(i + 2 * BatchSize, ax2, bx2, ay2, by2, cy2, dy2);
                compute_buf(ax0, bx0, ay0, by0, cy0, dy0);
                compute_buf(ax1, bx1, ay1, by1, cy1, dy1);
                compute_buf(ax2, bx2, ay2, by2, cy2, dy2);
                i += 3 * BatchSize;
            } else {
                // 4-stage steady-state
                load_buf_shared(i + 0 * BatchSize, ax0, bx0, ay0, by0, cy0, dy0);
                load_buf_shared(i + 1 * BatchSize, ax1, bx1, ay1, by1, cy1, dy1);
                load_buf_shared(i + 2 * BatchSize, ax2, bx2, ay2, by2, cy2, dy2);
                load_buf_shared(i + 3 * BatchSize, ax3, bx3, ay3, by3, cy3, dy3);

                int next = 4;
                const int nf = nFull;
                const int rounds = nf / 4;
                const int rem = nf % 4;

                for (int r = 0; r < rounds; ++r) {
                    compute_buf(ax0, bx0, ay0, by0, cy0, dy0);
                    if (next < nf) load_buf_shared(i + next * BatchSize, ax0, bx0, ay0, by0, cy0, dy0), ++next;

                    compute_buf(ax1, bx1, ay1, by1, cy1, dy1);
                    if (next < nf) load_buf_shared(i + next * BatchSize, ax1, bx1, ay1, by1, cy1, dy1), ++next;

                    compute_buf(ax2, bx2, ay2, by2, cy2, dy2);
                    if (next < nf) load_buf_shared(i + next * BatchSize, ax2, bx2, ay2, by2, cy2, dy2), ++next;

                    compute_buf(ax3, bx3, ay3, by3, cy3, dy3);
                    if (next < nf) load_buf_shared(i + next * BatchSize, ax3, bx3, ay3, by3, cy3, dy3), ++next;
                }
                if (rem >= 1) compute_buf(ax0, bx0, ay0, by0, cy0, dy0);
                if (rem >= 2) compute_buf(ax1, bx1, ay1, by1, cy1, dy1);
                if (rem >= 3) compute_buf(ax2, bx2, ay2, by2, cy2, dy2);

                i += nf * BatchSize;
            }
        }

        // Epilogue: scalar tail
        if (i <= i_end) {
            accumulate_scalar_span<SharkFloatParams, UseConditionalAccess>(
                grid,
                block,
                debugMultiplyCounts,
                i, i_end, k,
                aDigits_base, bDigits_base, a_offset, b_offset,
                x_diff_abs, y_diff_abs,
                xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high);
        }
        return;
    }

    // ---------------- global-memory path with alignment steering (BatchSize==4) ----------------
    if constexpr (
        SharkInnerLoopOption == HpShark::InnerLoopOption::TryUnalignedLoads2 ||
        SharkInnerLoopOption == HpShark::InnerLoopOption::TryUnalignedLoads2Shared) {

        ProcessConvolutionDirectLoad_Unaligned2<
            SharkFloatParams,
            16,
            UseConditionalAccess,
            RecursionDepth,
            ExecutionBlockBase,
            ExecutionNumBlocks>(
                grid,
                block,
                debugMultiplyCounts,
                i, i_start, i_end, k,
                aDigits_base, bDigits_base,
                a_offset, b_offset,
                xx_sum_low, xx_sum_high,
                xy_sum_low, xy_sum_high,
                yy_sum_low, yy_sum_high,
                x_diff_abs, y_diff_abs);
        return;
    }

    if constexpr (SharkInnerLoopOption == HpShark::InnerLoopOption::TryVectorLoads) {
        //
        // This implementation uses BatchSize==8 and vector loads.  It
        // imperfectly handles weird cases where digits don't match number of
        // threads and some weird digit lengths, but should work for e.g.
        // 108 blocks, 128 threads per block, and 7776 total digits.
        // 
        // Performance experiments are showing this approach is worse than direct
        // unaligned load-from-global-to-registers and it's far more complex.
        // It seems like we should be able to get a win in here but the logic for
        // dealing with the unaligned cases probably is what's slowing it down.
        //

        ProcessConvolutionDirectLoad_BS8_FwdAligned<
            SharkFloatParams,
            8,
            UseConditionalAccess,
            RecursionDepth,
            ExecutionBlockBase,
            ExecutionNumBlocks>(
                grid,
                block,
                debugMultiplyCounts,
                i, i_end, k,
                aDigits_base, bDigits_base,
                a_offset, b_offset,
                xx_sum_low, xx_sum_high,
                xy_sum_low, xy_sum_high,
                yy_sum_low, yy_sum_high,
                x_diff_abs, y_diff_abs);
        return;
    }

    if constexpr (SharkInnerLoopOption == HpShark::InnerLoopOption::TryUnalignedLoads) {
        //
        // This implementation uses BatchSize==8 and unaligned loads.
        //

        ProcessConvolutionDirectLoad_Unaligned<SharkFloatParams, 8, UseConditionalAccess,
            RecursionDepth, ExecutionBlockBase, ExecutionNumBlocks>(
                grid,
                block,
                debugMultiplyCounts,
                i, i_end, k,
                aDigits_base, bDigits_base,
                a_offset, b_offset,
                xx_sum_low, xx_sum_high,
                xy_sum_low, xy_sum_high,
                yy_sum_low, yy_sum_high,
                x_diff_abs, y_diff_abs);
        return;
    }
}


template<
    class SharkFloatParams,
    int RecursionDepth,
    int CallIndex,
    int NewN,
    int n1,
    int n2,
    int ExecutionBlockBase,
    int ExecutionNumBlocks,
    int NewNumBlocks,
    int TempBase>
__device__ SharkForceInlineReleaseOnly static void
MultiplyDigitsOnly(
    uint32_t *SharkRestrict shared_data,
    const HpSharkFloat<SharkFloatParams> *SharkRestrict A,
    const HpSharkFloat<SharkFloatParams> *SharkRestrict B,
    const uint32_t *SharkRestrict aDigits,
    const uint32_t *SharkRestrict bDigits,
    uint32_t *SharkRestrict x_diff_abs,
    uint32_t *SharkRestrict y_diff_abs,
    uint64_t *SharkRestrict final128XX,
    uint64_t *SharkRestrict final128XY,
    uint64_t *SharkRestrict final128YY,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *SharkRestrict tempProducts) {

    if ((ExecutionBlockBase > 0 && block.group_index().x < ExecutionBlockBase) ||
        block.group_index().x >= ExecutionBlockBase + ExecutionNumBlocks) {

        return;
    }

    DefineTempProductsOffsets(CallIndex);
    TempProductsGlobals(TempBase, CallIndex);

    auto *SharkRestrict debugMultiplyCounts = reinterpret_cast<DebugMultiplyCount<SharkFloatParams>*>(&tempProducts[Multiplies_offset]);
    auto *SharkRestrict debugStates = reinterpret_cast<DebugState<SharkFloatParams>*>(&tempProducts[Checksum_offset]);

    constexpr auto MaxHalfN = std::max(n1, n2);
    constexpr int total_k = MaxHalfN * 2 - 1; // Total number of k values
    constexpr bool UseConvolutionBool =
        (NewNumBlocks <= std::max(SharkFloatParams::GlobalNumBlocks / SharkFloatParams::ConvolutionLimit, 1) ||
        (NewNumBlocks % 3 != 0));
    constexpr UseConvolution UseConvolutionHere = UseConvolutionBool ? UseConvolution::Yes : UseConvolution::No;
    constexpr bool UseParallelSubtract = true;

    using DebugState = DebugState<SharkFloatParams>;

    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) ?
        RecordIt::Yes :
        RecordIt::No;

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();

        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Invalid>(
            record, debugStates, grid, block);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::ADigits, uint32_t>(
            record, UseConvolutionHere, debugStates, grid, block, aDigits, NewN);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BDigits, uint32_t>(
            record, UseConvolutionHere, debugStates, grid, block, bDigits, NewN);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::CDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::DDigits>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::EDigits>(
            record, debugStates, grid, block);

        grid.sync();
    }

    auto *SharkRestrict Z0_OutDigitsXX = &tempProducts[Z0_offsetXX];
    auto *SharkRestrict Z0_OutDigitsXY = &tempProducts[Z0_offsetXY];
    auto *SharkRestrict Z0_OutDigitsYY = &tempProducts[Z0_offsetYY];

    auto *SharkRestrict Z1_temp_digitsXX = &tempProducts[Z1_temp_offsetXX];
    auto *SharkRestrict Z1_temp_digitsXY = &tempProducts[Z1_temp_offsetXY];
    auto *SharkRestrict Z1_temp_digitsYY = &tempProducts[Z1_temp_offsetYY];

    auto *SharkRestrict Z2_OutDigitsXX = &tempProducts[Z2_offsetXX];
    auto *SharkRestrict Z2_OutDigitsXY = &tempProducts[Z2_offsetXY];
    auto *SharkRestrict Z2_OutDigitsYY = &tempProducts[Z2_offsetYY];

    // Arrays to hold the absolute differences (size n)
    auto *SharkRestrict global_x_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[XDiff_offset]);
    auto *SharkRestrict global_y_diff_abs = reinterpret_cast<uint32_t *>(&tempProducts[YDiff_offset]);

    // ---- Compute Differences x_diff = A1 - A0 and y_diff = B1 - B0 ----

    DefineExtraDefinitions();

    int x_diff_sign = 0; // 0 if positive, 1 if negative
    int y_diff_sign = 0; // 0 if positive, 1 if negative

    // Compute x_diff_abs and x_diff_sign
    auto *SharkRestrict subtractionBorrows = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset1]);
    auto *SharkRestrict subtractionBorrows2 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset2]);
    auto *SharkRestrict subtractionBorrows3 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset3]);
    auto *SharkRestrict subtractionBorrows4 = reinterpret_cast<uint32_t *>(&tempProducts[SubtractionOffset4]);
    auto *SharkRestrict globalBorrowAny = reinterpret_cast<uint32_t *>(&tempProducts[BorrowGlobalOffset]);
    auto *SharkRestrict globalBlockBorrow1 = reinterpret_cast<uint32_t *>(&tempProducts[BorrowBlockLevelOffset1]);
    auto *SharkRestrict globalBlockBorrow2 = reinterpret_cast<uint32_t *>(&tempProducts[BorrowBlockLevelOffset2]);

    const auto *SharkRestrict a_high = aDigits + n1;
    const auto *SharkRestrict b_high = bDigits + n1;
    const auto *SharkRestrict a_low = aDigits;
    const auto *SharkRestrict b_low = bDigits;

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();

        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfHigh>(
            record, UseConvolutionHere, debugStates, grid, block, a_high, n2);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfLow>(
            record, UseConvolutionHere, debugStates, grid, block, a_low, n1);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfHigh>(
            record, UseConvolutionHere, debugStates, grid, block, b_high, n2);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfLow>(
            record, UseConvolutionHere, debugStates, grid, block, b_low, n1);

        grid.sync();
    }

    if constexpr (!SharkFloatParams::DisableSubtraction) {
        if constexpr (UseParallelSubtract) {
            int x_compare = CompareDigits<n2, n1>(a_high, a_low);
            int y_compare = CompareDigits<n2, n1>(b_high, b_low);

            if (x_compare >= 0 && y_compare >= 0) {
                x_diff_sign = 0;
                y_diff_sign = 0;
                SubtractDigitsParallelImproved3<
                    SharkFloatParams,
                    n2,
                    n1,
                    n2,
                    n1,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_high,
                        a_low,
                        b_high,
                        b_low,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        globalBlockBorrow1,
                        globalBlockBorrow2,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else if (x_compare < 0 && y_compare < 0) {
                x_diff_sign = 1;
                y_diff_sign = 1;
                SubtractDigitsParallelImproved3<
                    SharkFloatParams,
                    n1,
                    n2,
                    n1,
                    n2,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_low,
                        a_high,
                        b_low,
                        b_high,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        globalBlockBorrow1,
                        globalBlockBorrow2,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else if (x_compare >= 0 && y_compare < 0) {
                x_diff_sign = 0;
                y_diff_sign = 1;
                SubtractDigitsParallelImproved3<
                    SharkFloatParams,
                    n2,
                    n1,
                    n1,
                    n2,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_high,
                        a_low,
                        b_low,
                        b_high,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        globalBlockBorrow1,
                        globalBlockBorrow2,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            } else {
                x_diff_sign = 1;
                y_diff_sign = 0;
                SubtractDigitsParallelImproved3<
                    SharkFloatParams,
                    n1,
                    n2,
                    n2,
                    n1,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(
                        x_diff_abs,
                        y_diff_abs,
                        a_low,
                        a_high,
                        b_high,
                        b_low,
                        subtractionBorrows,
                        subtractionBorrows2,
                        subtractionBorrows3,
                        subtractionBorrows4,
                        globalBlockBorrow1,
                        globalBlockBorrow2,
                        global_x_diff_abs,
                        global_y_diff_abs,
                        globalBorrowAny,
                        grid,
                        block);
            }
        } else {
            if (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) {
                int x_compare = CompareDigits<n1, n2>(a_high, a_low);

                if (x_compare >= 0) {
                    x_diff_sign = 0;
                    SubtractDigitsSerial<n2, n1>(a_high, a_low, global_x_diff_abs); // x_diff = A1 - A0
                } else {
                    x_diff_sign = 1;
                    SubtractDigitsSerial<n1, n2>(a_low, a_high, global_x_diff_abs); // x_diff = A0 - A1
                }

                // Compute y_diff_abs and y_diff_sign
                int y_compare = CompareDigits<n1, n2>(b_high, b_low);
                if (y_compare >= 0) {
                    y_diff_sign = 0;
                    SubtractDigitsSerial<n2, n1>(b_high, b_low, global_y_diff_abs); // y_diff = B1 - B0
                } else {
                    y_diff_sign = 1;
                    SubtractDigitsSerial<n1, n2>(b_low, b_high, global_y_diff_abs); // y_diff = B0 - B1
                }
            }
        }
    }

    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::XDiff>(
            record, UseConvolutionHere, debugStates, grid, block, global_x_diff_abs, MaxHalfN);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::YDiff>(
            record, UseConvolutionHere, debugStates, grid, block, global_y_diff_abs, MaxHalfN);

        grid.sync();
    }

    constexpr auto SubNewNRoundUp = (NewN + 1) / 2;
    constexpr auto SubNewN2a = SubNewNRoundUp / 2;
    constexpr auto SubNewN1a = SubNewNRoundUp - SubNewN2a;   /* n1 is larger or same */

    constexpr auto SubRemainingNewN = NewN - SubNewNRoundUp;
    constexpr auto SubNewN2b = SubRemainingNewN / 2;
    constexpr auto SubNewN1b = SubRemainingNewN - SubNewN2b;   /* n1 is larger or same */

    // Determine the sign of Z1_temp
    // int z1_sign = x_diff_sign ^ y_diff_sign;

    const int z1_signXX = (x_diff_sign ^ x_diff_sign) ? 1 : 0; // TODO obviously can be simplified
    const int z1_signXY = (x_diff_sign ^ y_diff_sign) ? 1 : 0;
    const int z1_signYY = (y_diff_sign ^ y_diff_sign) ? 1 : 0;

    constexpr auto FinalZ0Size =
        (UseConvolutionHere == UseConvolution::Yes) ?
        (total_k * 2) :
        (SubNewNRoundUp * 2 * 2);
    constexpr auto FinalZ2Size =
        (UseConvolutionHere == UseConvolution::Yes) ?
        (total_k * 2) :
        (SubRemainingNewN * 2 * 2);
    constexpr auto FinalZ1TempSize =
        (UseConvolutionHere == UseConvolution::Yes) ?
        (total_k * 2) :
        (SubNewNRoundUp * 2 * 2);

    if constexpr (UseConvolutionHere == UseConvolution::Yes) {
        // Replace A and B in shared memory with their absolute differences
        if constexpr (HpShark::LoadAllInShared) {
            cg::memcpy_async(block, const_cast<uint32_t *>(x_diff_abs), global_x_diff_abs, sizeof(uint32_t) * MaxHalfN);
            cg::memcpy_async(block, const_cast<uint32_t *>(y_diff_abs), global_y_diff_abs, sizeof(uint32_t) * MaxHalfN);
        }

        const int tid = RelativeBlockIndex * block.dim_threads().x + block.thread_index().x;
        const int stride = block.dim_threads().x * ExecutionNumBlocks;

        if constexpr (HpShark::LoadAllInShared) {
            // Wait for the first batch of A to be loaded
            cg::wait(block);
        }

        constexpr int outerIteration = 0;
        for (int idx = tid; idx < total_k * 3; idx += stride) {

            // Check if idx < total_k => handle Z0, else handle Z2
            if (idx < total_k) {
                // Z0 partial sums
                const int k_base = idx;
                int k = k_base; // shift to [0..total_k-1]
                uint64_t xx_sum_low = 0ULL, xx_sum_high = 0ULL;
                uint64_t xy_sum_low = 0ULL, xy_sum_high = 0ULL;
                uint64_t yy_sum_low = 0ULL, yy_sum_high = 0ULL;

                int i_start = (k < n1) ? 0 : (k - (n1 - 1));
                int i_end = (k < n1) ? k : (n1 - 1);

                ProcessConvolutionBatch<
                    SharkFloatParams,
                    SharkKaratsubaBatchSize,
                    ConditionalAccess::False,
                    RecursionDepth,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(

                    grid,
                    block,
                    debugMultiplyCounts,
                    RelativeBlockIndex,
                    outerIteration,
                    k,
                    total_k,
                    i_start,
                    i_end,
                    n1,
                    aDigits,
                    bDigits,
                    0,
                    0,  // Z0 uses base arrays with no offset
                    xx_sum_low,
                    xx_sum_high,
                    xy_sum_low,
                    xy_sum_high,
                    yy_sum_low,
                    yy_sum_high,
                    shared_data);

                int out_idx = k * 2;
                Z0_OutDigitsXX[out_idx] = xx_sum_low;
                Z0_OutDigitsXX[out_idx + 1] = xx_sum_high;
                Z0_OutDigitsXY[out_idx] = xy_sum_low;
                Z0_OutDigitsXY[out_idx + 1] = xy_sum_high;
                Z0_OutDigitsYY[out_idx] = yy_sum_low;
                Z0_OutDigitsYY[out_idx + 1] = yy_sum_high;
            } else if (idx < 2 * total_k) {
                // Z2 partial sums
                const int k_base = idx - total_k; // shift to [0..total_k-1]
                //int k = (k_base + total_k / 3) % total_k;
                int k = k_base;
                uint64_t xx_sum_low = 0ULL, xx_sum_high = 0ULL;
                uint64_t xy_sum_low = 0ULL, xy_sum_high = 0ULL;
                uint64_t yy_sum_low = 0ULL, yy_sum_high = 0ULL;

                int i_start = (k < n2) ? 0 : (k - (n2 - 1));
                int i_end = (k < n2) ? k : (n2 - 1);

                ProcessConvolutionBatch<
                    SharkFloatParams,
                    SharkKaratsubaBatchSize,
                    ConditionalAccess::False,
                    RecursionDepth,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(

                    grid,
                    block,
                    debugMultiplyCounts,
                    RelativeBlockIndex,
                    outerIteration,
                    k,
                    total_k,
                    i_start,
                    i_end,
                    n2,
                    aDigits,
                    bDigits,
                    n1,
                    n1,  // Z2 uses arrays with n1 offset
                    xx_sum_low,
                    xx_sum_high,
                    xy_sum_low,
                    xy_sum_high,
                    yy_sum_low,
                    yy_sum_high,
                    shared_data);

                int out_idx = k * 2;
                Z2_OutDigitsXX[out_idx] = xx_sum_low;
                Z2_OutDigitsXX[out_idx + 1] = xx_sum_high;
                Z2_OutDigitsXY[out_idx] = xy_sum_low;
                Z2_OutDigitsXY[out_idx + 1] = xy_sum_high;
                Z2_OutDigitsYY[out_idx] = yy_sum_low;
                Z2_OutDigitsYY[out_idx + 1] = yy_sum_high;
            } else {
                const int k_base = idx - 2 * total_k; // shift to [0..total_k-1]
                //int k = (k_base + 2 * total_k / 3) % total_k;
                int k = k_base;
                uint64_t xx_sum_low = 0ULL, xx_sum_high = 0ULL;
                uint64_t xy_sum_low = 0ULL, xy_sum_high = 0ULL;
                uint64_t yy_sum_low = 0ULL, yy_sum_high = 0ULL;

                int i_start = (k < MaxHalfN) ? 0 : (k - (MaxHalfN - 1));
                int i_end = (k < MaxHalfN) ? k : (MaxHalfN - 1);

                ProcessConvolutionBatch<
                    SharkFloatParams,
                    SharkKaratsubaBatchSize,
                    ConditionalAccess::True,
                    RecursionDepth,
                    ExecutionBlockBase,
                    ExecutionNumBlocks>(

                    grid,
                    block,
                    debugMultiplyCounts,
                    RelativeBlockIndex,
                    outerIteration,
                    k,
                    total_k,
                    i_start,
                    i_end,
                    MaxHalfN,
                    nullptr,
                    nullptr,
                    0,
                    0,  // Not used for Z1_temp
                    xx_sum_low,
                    xx_sum_high,
                    xy_sum_low,
                    xy_sum_high,
                    yy_sum_low,
                    yy_sum_high,
                    shared_data,
                    HpShark::LoadAllInShared ? x_diff_abs : global_x_diff_abs,
                    HpShark::LoadAllInShared ? y_diff_abs : global_y_diff_abs);

                int out_idx = k * 2;
                Z1_temp_digitsXX[out_idx] = xx_sum_low;
                Z1_temp_digitsXX[out_idx + 1] = xx_sum_high;
                Z1_temp_digitsXY[out_idx] = xy_sum_low;
                Z1_temp_digitsXY[out_idx + 1] = xy_sum_high;
                Z1_temp_digitsYY[out_idx] = yy_sum_low;
                Z1_temp_digitsYY[out_idx + 1] = yy_sum_high;
            }
        }
    } else {
        static_assert(RecursionDepth <= 5, "Unexpected recursion depth");

        MultiplyDigitsOnly<
            SharkFloatParams,
            RecursionDepth + 1,
            CallIndex * 3 - 1,
            SubNewNRoundUp,
            SubNewN1a,
            SubNewN2a,
            ExecutionBlockBase,
            ExecutionNumBlocks / 3,
            NewNumBlocks / 3,
            TempBase>(
            shared_data,
            A,
            B,
            aDigits,
            bDigits,
            x_diff_abs,
            y_diff_abs,
            Z0_OutDigitsXX,
            Z0_OutDigitsXY,
            Z0_OutDigitsYY,
            grid,
            block,
            tempProducts);

        MultiplyDigitsOnly<
            SharkFloatParams,
            RecursionDepth + 1,
            CallIndex * 3,
            SubRemainingNewN,
            SubNewN1b,
            SubNewN2b,
            ExecutionBlockBase + ExecutionNumBlocks / 3,
            ExecutionNumBlocks / 3,
            NewNumBlocks / 3,
            TempBase>(
            shared_data,
            A,
            B,
            aDigits + n1,
            bDigits + n1,
            x_diff_abs,
            y_diff_abs,
            Z2_OutDigitsXX,
            Z2_OutDigitsXY,
            Z2_OutDigitsYY,
            grid,
            block,
            tempProducts);

        //grid.sync();

        {
            constexpr auto NewExecutionBlockBase = ExecutionBlockBase + 2 * ExecutionNumBlocks / 3;
            constexpr auto NewExecutionNumBlocks = ExecutionNumBlocks / 3;

            const bool ExecuteAtAll =
                !((NewExecutionBlockBase > 0 && block.group_index().x < NewExecutionBlockBase) ||
                    block.group_index().x >= NewExecutionBlockBase + NewExecutionNumBlocks);
            constexpr auto MaxSubNewN = std::max(SubNewN1a, SubNewN2a);

            if (ExecuteAtAll) {
                // Replace A and B in shared memory with their absolute differences
                if constexpr (HpShark::LoadAllInShared) {
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(aDigits),
                        global_x_diff_abs,
                        sizeof(uint32_t) * MaxHalfN);
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(bDigits),
                        global_y_diff_abs,
                        sizeof(uint32_t) * MaxHalfN);
                    cg::wait(block);
                }

                MultiplyDigitsOnly<
                    SharkFloatParams,
                    RecursionDepth + 1,
                    CallIndex * 3 + 1,
                    SubNewNRoundUp,
                    SubNewN1a,
                    SubNewN2a,
                    NewExecutionBlockBase,
                    NewExecutionNumBlocks,
                    NewNumBlocks / 3,
                    TempBase>(
                        shared_data,
                        A,
                        B,
                        HpShark::LoadAllInShared ? aDigits : global_x_diff_abs,
                        HpShark::LoadAllInShared ? bDigits : global_y_diff_abs,
                        x_diff_abs,
                        y_diff_abs,
                        Z1_temp_digitsXX,
                        Z1_temp_digitsXY,
                        Z1_temp_digitsYY,
                        grid,
                        block,
                        tempProducts);

                if constexpr (HpShark::LoadAllInShared) {
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(aDigits),
                        A->Digits,
                        sizeof(uint32_t) * SharkFloatParams::GlobalNumUint32);
                    cg::memcpy_async(block,
                        const_cast<uint32_t *>(bDigits),
                        B->Digits,
                        sizeof(uint32_t) * SharkFloatParams::GlobalNumUint32);
                    cg::wait(block);
                }
            }
        }
    }

    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0XX>(
            record, UseConvolutionHere, debugStates, grid, block, Z0_OutDigitsXX, FinalZ0Size);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0XY>(
            record, UseConvolutionHere, debugStates, grid, block, Z0_OutDigitsXY, FinalZ0Size);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0YY>(
            record, UseConvolutionHere, debugStates, grid, block, Z0_OutDigitsYY, FinalZ0Size);

        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2XX>(
            record, UseConvolutionHere, debugStates, grid, block, Z2_OutDigitsXX, FinalZ2Size);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2XY>(
            record, UseConvolutionHere, debugStates, grid, block, Z2_OutDigitsXY, FinalZ2Size);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2YY>(
            record, UseConvolutionHere, debugStates, grid, block, Z2_OutDigitsYY, FinalZ2Size);

        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm1>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm2>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm3>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm4>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm5>(
            record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm6>(
            record, debugStates, grid, block);

        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetXX>(
            record, UseConvolutionHere, debugStates, grid, block, Z1_temp_digitsXX, FinalZ1TempSize);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetXY>(
            record, UseConvolutionHere, debugStates, grid, block, Z1_temp_digitsXY, FinalZ1TempSize);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetYY>(
            record, UseConvolutionHere, debugStates, grid, block, Z1_temp_digitsYY, FinalZ1TempSize);

        grid.sync();
    }

    auto *SharkRestrict Z1_digitsXX = &tempProducts[Z1_offsetXX];
    auto *SharkRestrict Z1_digitsXY = &tempProducts[Z1_offsetXY];
    auto *SharkRestrict Z1_digitsYY = &tempProducts[Z1_offsetYY];

    if constexpr (!SharkFloatParams::DisableAllAdditions) {

        // After computing Z1_temp (Z1'), we now form Z1 directly:
        // If z1_sign == 0: Z1 = Z2 + Z0 - Z1_temp
        // If z1_sign == 1: Z1 = Z2 + Z0 + Z1_temp

        const int tid = RelativeBlockIndex * block.dim_threads().x + block.thread_index().x;
        const int stride = block.dim_threads().x * ExecutionNumBlocks;

        for (int i = tid; i < total_k; i += stride) {
            // Retrieve Z0
            int z0_idx = i * 2;
            const uint64_t xx_z0_low = Z0_OutDigitsXX[z0_idx];
            const uint64_t xx_z0_high = Z0_OutDigitsXX[z0_idx + 1];

            const uint64_t xy_z0_low = Z0_OutDigitsXY[z0_idx];
            const uint64_t xy_z0_high = Z0_OutDigitsXY[z0_idx + 1];

            const uint64_t yy_z0_low = Z0_OutDigitsYY[z0_idx];
            const uint64_t yy_z0_high = Z0_OutDigitsYY[z0_idx + 1];

            // Retrieve Z2
            int z2_idx = i * 2;
            const uint64_t xx_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsXX[z2_idx] : 0;
            const uint64_t xx_z2_high = z2_idx < FinalZ2Size ? Z2_OutDigitsXX[z2_idx + 1] : 0;

            const uint64_t xy_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsXY[z2_idx] : 0;
            const uint64_t xy_z2_high = z2_idx < FinalZ2Size ? Z2_OutDigitsXY[z2_idx + 1] : 0;

            const uint64_t yy_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsYY[z2_idx] : 0;
            const uint64_t yy_z2_high = z2_idx < FinalZ2Size ? Z2_OutDigitsYY[z2_idx + 1] : 0;

            // Retrieve Z1_temp (Z1')
            int z1_temp_idx = i * 2;
            const uint64_t xx_z1_temp_low = Z1_temp_digitsXX[z1_temp_idx];
            const uint64_t xx_z1_temp_high = Z1_temp_digitsXX[z1_temp_idx + 1];

            const uint64_t xy_z1_temp_low = Z1_temp_digitsXY[z1_temp_idx];
            const uint64_t xy_z1_temp_high = Z1_temp_digitsXY[z1_temp_idx + 1];

            const uint64_t yy_z1_temp_low = Z1_temp_digitsYY[z1_temp_idx];
            const uint64_t yy_z1_temp_high = Z1_temp_digitsYY[z1_temp_idx + 1];

            // Combine Z2 + Z0 first
            uint64_t xx_temp_low, xx_temp_high;
            uint64_t xy_temp_low, xy_temp_high;
            uint64_t yy_temp_low, yy_temp_high;

            Add128(xx_z2_low, xx_z2_high, xx_z0_low, xx_z0_high, xx_temp_low, xx_temp_high);
            Add128(xy_z2_low, xy_z2_high, xy_z0_low, xy_z0_high, xy_temp_low, xy_temp_high);
            Add128(yy_z2_low, yy_z2_high, yy_z0_low, yy_z0_high, yy_temp_low, yy_temp_high);

            // Now combine with Z1_temp
            // Z1 = (Z2 + Z0) +/- Z1_temp
            uint64_t xx_z1_low, xx_z1_high;
            uint64_t xy_z1_low, xy_z1_high;
            uint64_t yy_z1_low, yy_z1_high;

            if (z1_signXX == 0) {
                // same sign: Z1 = (Z2 + Z0) - Z1_temp
                Subtract128(xx_temp_low, xx_temp_high, xx_z1_temp_low, xx_z1_temp_high, xx_z1_low, xx_z1_high);
            } else {
                // opposite signs: Z1 = (Z2 + Z0) + Z1_temp
                Add128(xx_temp_low, xx_temp_high, xx_z1_temp_low, xx_z1_temp_high, xx_z1_low, xx_z1_high);
            }

            if (z1_signXY == 0) {
                // same sign: Z1 = (Z2 + Z0) - Z1_temp
                Subtract128(xy_temp_low, xy_temp_high, xy_z1_temp_low, xy_z1_temp_high, xy_z1_low, xy_z1_high);
            } else {
                // opposite signs: Z1 = (Z2 + Z0) + Z1_temp
                Add128(xy_temp_low, xy_temp_high, xy_z1_temp_low, xy_z1_temp_high, xy_z1_low, xy_z1_high);
            }

            if (z1_signYY == 0) {
                // same sign: Z1 = (Z2 + Z0) - Z1_temp
                Subtract128(yy_temp_low, yy_temp_high, yy_z1_temp_low, yy_z1_temp_high, yy_z1_low, yy_z1_high);
            } else {
                // opposite signs: Z1 = (Z2 + Z0) + Z1_temp
                Add128(yy_temp_low, yy_temp_high, yy_z1_temp_low, yy_z1_temp_high, yy_z1_low, yy_z1_high);
            }

            // Store fully formed Z1
            int z1_idx = i * 2;
            Z1_digitsXX[z1_idx] = xx_z1_low;
            Z1_digitsXX[z1_idx + 1] = xx_z1_high;

            Z1_digitsXY[z1_idx] = xy_z1_low;
            Z1_digitsXY[z1_idx + 1] = xy_z1_high;

            Z1_digitsYY[z1_idx] = yy_z1_low;
            Z1_digitsYY[z1_idx + 1] = yy_z1_high;
        }

        if constexpr (HpShark::DebugChecksums) {
            grid.sync();

            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1XX>(
                record, UseConvolutionHere, debugStates, grid, block, Z1_digitsXX, total_k * 2);
            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1XY>(
                record, UseConvolutionHere, debugStates, grid, block, Z1_digitsXY, total_k * 2);
            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1YY>(
                record, UseConvolutionHere, debugStates, grid, block, Z1_digitsYY, total_k * 2);
        }

        // Synchronize before final combination
        grid.sync();

        // Now the final combination is just:
        // final = Z0 + (Z1 << (32*n)) + (Z2 << (64*n))
        for (int i = tid; i < total_result_digits; i += stride) {
            uint64_t xx_sum_low = 0;
            uint64_t xx_sum_high = 0;

            uint64_t xy_sum_low = 0;
            uint64_t xy_sum_high = 0;

            uint64_t yy_sum_low = 0;
            uint64_t yy_sum_high = 0;

            // Add Z0
            if (i < 2 * n1 - 1) {
                int z0_idx = i * 2;

                const uint64_t xx_z0_low = Z0_OutDigitsXX[z0_idx];
                const uint64_t xx_z0_high = Z0_OutDigitsXX[z0_idx + 1];

                const uint64_t xy_z0_low = Z0_OutDigitsXY[z0_idx];
                const uint64_t xy_z0_high = Z0_OutDigitsXY[z0_idx + 1];

                const uint64_t yy_z0_low = Z0_OutDigitsYY[z0_idx];
                const uint64_t yy_z0_high = Z0_OutDigitsYY[z0_idx + 1];
                
                Add128(xx_sum_low, xx_sum_high, xx_z0_low, xx_z0_high, xx_sum_low, xx_sum_high);
                Add128(xy_sum_low, xy_sum_high, xy_z0_low, xy_z0_high, xy_sum_low, xy_sum_high);
                Add128(yy_sum_low, yy_sum_high, yy_z0_low, yy_z0_high, yy_sum_low, yy_sum_high);
            }

            // Add Z1 shifted by n
            if (i >= n1 && (i - n1) < 2 * n1 - 1) {
                int z1_idx = (i - n1) * 2;

                const uint64_t xx_z1_low = Z1_digitsXX[z1_idx];
                const uint64_t xx_z1_high = Z1_digitsXX[z1_idx + 1];

                const uint64_t xy_z1_low = Z1_digitsXY[z1_idx];
                const uint64_t xy_z1_high = Z1_digitsXY[z1_idx + 1];

                const uint64_t yy_z1_low = Z1_digitsYY[z1_idx];
                const uint64_t yy_z1_high = Z1_digitsYY[z1_idx + 1];

                Add128(xx_sum_low, xx_sum_high, xx_z1_low, xx_z1_high, xx_sum_low, xx_sum_high);
                Add128(xy_sum_low, xy_sum_high, xy_z1_low, xy_z1_high, xy_sum_low, xy_sum_high);
                Add128(yy_sum_low, yy_sum_high, yy_z1_low, yy_z1_high, yy_sum_low, yy_sum_high);
            }

            // Add Z2 shifted by 2*n
            if (i >= 2 * n1 && (i - 2 * n1) < 2 * n1 - 1) {
                int z2_idx = (i - 2 * n1) * 2;

                const uint64_t xx_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsXX[z2_idx] : 0;
                const uint64_t xx_z2_high = z2_idx + 1 < FinalZ2Size ? Z2_OutDigitsXX[z2_idx + 1] : 0;

                const uint64_t xy_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsXY[z2_idx] : 0;
                const uint64_t xy_z2_high = z2_idx + 1 < FinalZ2Size ? Z2_OutDigitsXY[z2_idx + 1] : 0;

                const uint64_t yy_z2_low = z2_idx < FinalZ2Size ? Z2_OutDigitsYY[z2_idx] : 0;
                const uint64_t yy_z2_high = z2_idx + 1 < FinalZ2Size ? Z2_OutDigitsYY[z2_idx + 1] : 0;

                Add128(xx_sum_low, xx_sum_high, xx_z2_low, xx_z2_high, xx_sum_low, xx_sum_high);
                Add128(xy_sum_low, xy_sum_high, xy_z2_low, xy_z2_high, xy_sum_low, xy_sum_high);
                Add128(yy_sum_low, yy_sum_high, yy_z2_low, yy_z2_high, yy_sum_low, yy_sum_high);
            }

            int result_idx = i * 2;

            // Store the final result
            final128XX[result_idx] = xx_sum_low;
            final128XX[result_idx + 1] = xx_sum_high;

            final128XY[result_idx] = xy_sum_low;
            final128XY[result_idx + 1] = xy_sum_high;

            final128YY[result_idx] = yy_sum_low;
            final128YY[result_idx + 1] = yy_sum_high;
        }

        if constexpr (HpShark::DebugChecksums) {
            grid.sync();

            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128XX>(
                record, UseConvolutionHere, debugStates, grid, block, final128XX, total_result_digits * 2);
            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128XY>(
                record, UseConvolutionHere, debugStates, grid, block, final128XY, total_result_digits * 2);
            StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128YY>(
                record, UseConvolutionHere, debugStates, grid, block, final128YY, total_result_digits * 2);

            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd1>(
                record, debugStates, grid, block);
            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd2>(
                record, debugStates, grid, block);
            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd3>(
                record, debugStates, grid, block);

            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXX>(
                record, debugStates, grid, block);
            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXY>(
                record, debugStates, grid, block);
            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetYY>(
                record, debugStates, grid, block);

            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_Add1>(
                record, debugStates, grid, block);
            EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_Add2>(
                record, debugStates, grid, block);
        }

        // Synchronize before carry propagation
        grid.sync();
    }
}

//
// static constexpr int32_t SharkFloatParams::GlobalThreadsPerBlock = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::GlobalNumBlocks = /* power of 2 */;
// static constexpr int32_t SharkFloatParams::GlobalNumUint32 = SharkFloatParams::GlobalThreadsPerBlock * SharkFloatParams::GlobalNumBlocks;
// 

template<class SharkFloatParams>
static __device__ void MultiplyHelperKaratsubaV2Separates(
    const HpSharkFloat<SharkFloatParams> *SharkRestrict A,
    const HpSharkFloat<SharkFloatParams> *SharkRestrict B,
    HpSharkFloat<SharkFloatParams> *SharkRestrict OutXX,
    HpSharkFloat<SharkFloatParams> *SharkRestrict OutXY,
    HpSharkFloat<SharkFloatParams> *SharkRestrict OutYY,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *SharkRestrict tempProducts) {

    extern __shared__ uint32_t shared_data[];

    constexpr auto NewN = SharkFloatParams::GlobalNumUint32;         // Total number of digits
    constexpr auto NewN1 = (NewN + 1) / 2;
    constexpr auto NewN2 = NewN - NewN1;   /* n1 is larger or same */
    constexpr auto CallIndex = 0;
    constexpr auto ExecutionBlockBase = 0;
    constexpr auto ExecutionNumBlocks = SharkFloatParams::GlobalNumBlocks;
    constexpr auto RecursionDepth = 0;

    DefineTempProductsOffsets(CallIndex);

    // Must fit inside the computed frame size (in u64 units)
    // Frame starts just after the 256-u64 per-frame header:
    constexpr auto FrameStart = TempBaseOffset + AdditionalUInt64PerFrame;
    // Frame end you already computed:
    constexpr auto FrameEnd = CarryInsEnd;
    static_assert(
        FrameEnd <= TempBaseOffset + CalculateKaratsubaFrameSize<SharkFloatParams>(),
        "Per-frame buffers overflow CalculateKaratsubaFrameSize"
        );

    // Also ensure we never intrude into the global region
    static_assert(
        TempBaseOffset >= (Checksum_offset + AdditionalGlobalChecksumSpace),
        "Per-frame region overlaps global header"
        );

    // How many NewN-sized slots did we actually consume?
    constexpr auto kSlotsUsedNewN =
        /* up to CarryInsOffset */ 79 +
        /* CarryInsEnd adds 3*NewN */ 3;

    static_assert(
        kSlotsUsedNewN <= ScratchMemoryArraysForMultiply,
        "Used more NewN slots than ScratchMemoryArraysForMultiply allows"
        );

    static_assert((Z0_offsetXX % 2) == 0, "Z0_offsetXX must be 16-byte aligned");
    static_assert((Z0_offsetXY % 2) == 0, "Z0_offsetXY must be 16-byte aligned");
    static_assert((Z0_offsetYY % 2) == 0, "Z0_offsetYY must be 16-byte aligned");


    auto *SharkRestrict debugMultiplyCounts = reinterpret_cast<DebugMultiplyCount<SharkFloatParams>*>(&tempProducts[Multiplies_offset]);
    auto *SharkRestrict debugStates = reinterpret_cast<DebugState<SharkFloatParams>*>(&tempProducts[Checksum_offset]);

    auto *SharkRestrict aDigits =
        HpShark::LoadAllInShared ?
        (shared_data) :
        const_cast<uint32_t *>(A->Digits);
    auto *SharkRestrict bDigits =
        HpShark::LoadAllInShared ?
        (aDigits + NewN + CalcAlign16Bytes32BitIndex(NewN)) :
        const_cast<uint32_t *>(B->Digits);
    auto *SharkRestrict x_diff_abs =
        HpShark::LoadAllInShared ?
        reinterpret_cast<uint32_t *>(bDigits + NewN + CalcAlign16Bytes32BitIndex(NewN)) :
        reinterpret_cast<uint32_t *>(&tempProducts[XDiff_offset]);
    auto *SharkRestrict y_diff_abs =
        HpShark::LoadAllInShared ?
        reinterpret_cast<uint32_t *>(x_diff_abs + (NewN + 1) / 2 + CalcAlign16Bytes32BitIndex((NewN + 1) / 2)) :
        reinterpret_cast<uint32_t *>(&tempProducts[YDiff_offset]);

    if constexpr (HpShark::LoadAllInShared) {
        cg::memcpy_async(block, aDigits, A->Digits, sizeof(uint32_t) * NewN);
        cg::memcpy_async(block, bDigits, B->Digits, sizeof(uint32_t) * NewN);
    }/* else {
        constexpr int AvailableShared = CalculateMultiplySharedMemorySize<SharkFloatParams>();
        memset(shared_data, 0xcd, AvailableShared);
    }*/

    if constexpr (HpShark::PrintMultiplyCounts) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread].DebugMultiplyErase();
    }

    if constexpr (HpShark::DebugChecksums) {
        const auto CurBlock = block.group_index().x;
        const auto CurThread = block.thread_index().x;
        debugMultiplyCounts[CurBlock * SharkFloatParams::GlobalThreadsPerBlock + CurThread].DebugMultiplyErase();

        const RecordIt record =
            (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) ?
            RecordIt::Yes :
            RecordIt::No;
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Invalid>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::ADigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::CDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::DDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::EDigits>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::AHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfHigh>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::BHalfLow>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::XDiff>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::YDiff>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z0YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z3XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z3XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z3YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z4XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z4XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z4YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm4>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm5>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z2_Perm6>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Z1_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128XX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128XY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Final128YY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd2>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::FinalAdd3>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXX>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetYY>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_Add1>(record, debugStates, grid, block);
        EraseCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_Add2>(record, debugStates, grid, block);
        static_assert(static_cast<int32_t>(DebugStatePurpose::NumPurposes) == 41, "Unexpected number of purposes");
    }

    // Wait for the first batch of A to be loaded
    cg::wait(block);

    auto *SharkRestrict final128XX = &tempProducts[Convolution_offsetXX];
    auto *SharkRestrict final128XY = &tempProducts[Convolution_offsetXY];
    auto *SharkRestrict final128YY = &tempProducts[Convolution_offsetYY];

    MultiplyDigitsOnly<
        SharkFloatParams,
        RecursionDepth + 1,
        CallIndex + 1,
        NewN,
        NewN1,
        NewN2,
        ExecutionBlockBase,
        ExecutionNumBlocks,
        SharkFloatParams::GlobalNumBlocks,
        TempBase>(
            shared_data,
            A,
            B,
            aDigits,
            bDigits,
            x_diff_abs,
            y_diff_abs,
            final128XX,
            final128XY,
            final128YY,
            grid,
            block,
            tempProducts);

    grid.sync();

    // ---- Carry Propagation ----

    // Global memory for block carry-outs
    // Allocate space for grid.dim_blocks().x block carry-outs after total_result_digits
    // Note, overlaps:
    uint64_t *block_carry_outs = &tempProducts[CarryInsOffset];

    auto *SharkRestrict resultXX = &tempProducts[Result_offsetXX];
    auto *SharkRestrict resultXY = &tempProducts[Result_offsetXY];
    auto *SharkRestrict resultYY = &tempProducts[Result_offsetYY];

    if constexpr (!SharkFloatParams::DisableCarryPropagation) {

        DefineCarryDefinitions();
        constexpr bool UseParallelCarry = true;
        uint64_t *globalCarryCheck = &tempProducts[GlobalCarryOffset];

        if constexpr (UseParallelCarry) {

            // First Pass: Process convolution results to compute initial digits and local carries
            CarryPropagation<SharkFloatParams>(
                (uint64_t *)shared_data,
                grid,
                block,
                thread_start_idx,
                thread_end_idx,
                final128XX,
                final128XY,
                final128YY,
                resultXX,
                resultXY,
                resultYY,
                block_carry_outs,
                globalCarryCheck
            );

            grid.sync();
        } else {
            SerialCarryPropagation<SharkFloatParams>(
                (uint64_t *)shared_data,
                grid,
                block,
                thread_start_idx,
                thread_end_idx,
                final128XX,
                final128XY,
                final128YY,
                resultXX,
                resultXY,
                resultYY,
                nullptr,
                nullptr
            );

            grid.sync();
        }
    } else {
        grid.sync();
    }

    using DebugState = DebugState<SharkFloatParams>;
    const uint64_t *resultEntriesXX = &tempProducts[Result_offsetXX];
    const uint64_t *resultEntriesXY = &tempProducts[Result_offsetXY];
    const uint64_t *resultEntriesYY = &tempProducts[Result_offsetYY];
    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == ExecutionBlockBase) ?
        RecordIt::Yes :
        RecordIt::No;

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();

        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXX>(
            record, UseConvolution::No, debugStates, grid, block, resultEntriesXX, 2 * NewN);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetXY>(
            record, UseConvolution::No, debugStates, grid, block, resultEntriesXY, 2 * NewN);
        StoreCurrentDebugState<SharkFloatParams, RecursionDepth, CallIndex, DebugStatePurpose::Result_offsetYY>(
            record, UseConvolution::No, debugStates, grid, block, resultEntriesYY, 2 * NewN);

        grid.sync();
    }

    // ---- Finalize the Result ----
    if constexpr (!SharkFloatParams::DisableFinalConstruction) {
        // uint64_t final_carry = carryOuts_phase6[SharkFloatParams::GlobalNumBlocks - 1];

        // Initial total_result_digits is 2 * NewN
        int total_result_digits = 2 * NewN;

        // Determine the highest non-zero digit index in the full result
        int highest_nonzero_index_xx = total_result_digits - 1;
        int highest_nonzero_index_xy = total_result_digits - 1;
        int highest_nonzero_index_yy = total_result_digits - 1;

        auto HighestNonzeroIndex = [](const uint64_t *result, int &highest_nonzero_index) {
            while (highest_nonzero_index >= 0) {
                int result_idx = highest_nonzero_index;
                uint32_t digit = static_cast<uint32_t>(result[result_idx]);
                if (digit != 0) {
                    break;
                }
                highest_nonzero_index--;
            }
            };

        HighestNonzeroIndex(resultEntriesXX, highest_nonzero_index_xx);
        HighestNonzeroIndex(resultEntriesXY, highest_nonzero_index_xy);
        HighestNonzeroIndex(resultEntriesYY, highest_nonzero_index_yy);

        // Determine the number of significant digits
        const int significant_digits_xx = highest_nonzero_index_xx + 1;
        const int significant_digits_xy = highest_nonzero_index_xy + 1;
        const int significant_digits_yy = highest_nonzero_index_yy + 1;

        // Calculate the number of digits to shift to keep the most significant NewN digits
        int shift_digits_xx = significant_digits_xx - NewN;
        if (shift_digits_xx < 0) {
            shift_digits_xx = 0;  // No need to shift if we have fewer than NewN significant digits
        }

        int shift_digits_xy = significant_digits_xy - NewN;
        if (shift_digits_xy < 0) {
            shift_digits_xy = 0;  // No need to shift if we have fewer than NewN significant digits
        }

        int shift_digits_yy = significant_digits_yy - NewN;
        if (shift_digits_yy < 0) {
            shift_digits_yy = 0;  // No need to shift if we have fewer than NewN significant digits
        }

        auto ExponentAndSign = [](
            cg::thread_block &block,
            const HpSharkFloat<SharkFloatParams> *A,
            const HpSharkFloat<SharkFloatParams> *B,
            bool forcePositive,
            HpSharkFloat<SharkFloatParams> *Out,
            int shift_digits,
            int additionalFactorsOfTwo) {

                if (block.group_index().x == 0 && block.thread_index().x == 0) {
                    // Adjust the exponent based on the number of bits shifted
                    Out->Exponent = A->Exponent + B->Exponent + shift_digits * 32 + additionalFactorsOfTwo;

                    // Set the sign of the result
                    Out->SetNegative(forcePositive ? false : (A->GetNegative() ^ B->GetNegative()));
                }
            };

        constexpr auto X2_AdditionalFactorsOfTwo = 0;
        ExponentAndSign(
            block,
            A,
            A,
            true,
            OutXX,
            shift_digits_xx,
            X2_AdditionalFactorsOfTwo);

        constexpr auto XY_AdditionalFactorsOfTwo = 1;
        ExponentAndSign(
            block,
            A,
            B,
            false,
            OutXY,
            shift_digits_xy,
            XY_AdditionalFactorsOfTwo);

        constexpr auto Y2_AdditionalFactorsOfTwo = 0;
        ExponentAndSign(
            block,
            B,
            B,
            true,
            OutYY,
            shift_digits_yy,
            Y2_AdditionalFactorsOfTwo);

        auto Finalize = [](
            cg::grid_group &grid,
            cg::thread_block &block,
            const uint64_t *result,
            int highest_nonzero_index,
            int shift_digits,
            HpSharkFloat<SharkFloatParams> *Out) {

                const int tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
                const int stride = block.dim_threads().x * grid.dim_blocks().x;

                // src_idx is the starting index in tempProducts[] from which we copy
                // TODO:
                const int src_idx = shift_digits;
                const int last_src = highest_nonzero_index; // The last valid index

                // We'll do a grid-stride loop over i in [0 .. NewN)
                for (int i = tid; i < NewN; i += stride) {
                    // Corresponding source index for digit i
                    int src = src_idx + i;

                    if (src <= last_src) {
                        // Copy from tempProducts
                        Out->Digits[i] = result[src];
                    } else {
                        // Pad with zero if we've run out of digits
                        Out->Digits[i] = 0;
                    }
                }
            };

        Finalize(
            grid,
            block,
            resultEntriesXX,
            highest_nonzero_index_xx,
            shift_digits_xx,
            OutXX);

        Finalize(
            grid,
            block,
            resultEntriesXY,
            highest_nonzero_index_xy,
            shift_digits_xy,
            OutXY);

        Finalize(
            grid,
            block,
            resultEntriesYY,
            highest_nonzero_index_yy,
            shift_digits_yy,
            OutYY);
    }
}

template<class SharkFloatParams>
void PrintMaxActiveBlocks(void *kernelFn, int sharedAmountBytes) {
    std::cout << "Shared memory size: " << sharedAmountBytes << std::endl;

    int numBlocks;

    {
        // Check the maximum number of active blocks per multiprocessor
        // with the given shared memory size
        // This is useful to determine if we can fit more blocks
        // in the shared memory

        const auto err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            kernelFn,
            SharkFloatParams::GlobalThreadsPerBlock,
            sharedAmountBytes
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyMaxActiveBlocksPerMultiprocessor: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max active blocks per multiprocessor: " << numBlocks << std::endl;
    }

    {
        size_t availableSharedMemory = 0;
        const auto err = cudaOccupancyAvailableDynamicSMemPerBlock(
            &availableSharedMemory,
            kernelFn,
            numBlocks,
            SharkFloatParams::GlobalThreadsPerBlock
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaOccupancyAvailableDynamicSMemPerBlock: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Available shared memory per block: " << availableSharedMemory << std::endl;
    }

    // Check the number of multiprocessors on the device
    int numSM;

    {
        const auto err = cudaDeviceGetAttribute(
            &numSM,
            cudaDevAttrMultiProcessorCount,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Number of multiprocessors: " << numSM << std::endl;
    }

    int maxConcurrentBlocks = numSM * numBlocks;

    std::cout << "Max concurrent blocks: " << maxConcurrentBlocks << std::endl;
    if (maxConcurrentBlocks < SharkFloatParams::GlobalNumBlocks) {
        std::cout << "Warning: Max concurrent blocks exceeds the number of blocks requested." << std::endl;
    }

    {
        // Check the maximum number of threads per block
        int maxThreadsPerBlock;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerBlock,
            cudaDevAttrMaxThreadsPerBlock,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    }

    {
        // Check the maximum number of threads per multiprocessor
        int maxThreadsPerMultiprocessor;
        const auto err = cudaDeviceGetAttribute(
            &maxThreadsPerMultiprocessor,
            cudaDevAttrMaxThreadsPerMultiProcessor,
            0
        );
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        std::cout << "Max threads per multiprocessor: " << maxThreadsPerMultiprocessor << std::endl;
    }

    // Check if this device supports cooperative launches
    int cooperativeLaunch;

    {
        const auto err = cudaDeviceGetAttribute(
            &cooperativeLaunch,
            cudaDevAttrCooperativeLaunch,
            0
        );

        if (err != cudaSuccess) {
            std::cerr << "CUDA error in cudaDeviceGetAttribute: " << cudaGetErrorString(err) << std::endl;
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
template<class SharkFloatParams>
static __device__ void MultiplyHelperKaratsubaV2 (
    HpSharkComboResults<SharkFloatParams> *SharkRestrict combo,
    cg::grid_group &grid,
    cg::thread_block &block,
    uint64_t *SharkRestrict tempProducts) {

    MultiplyHelperKaratsubaV2Separates<SharkFloatParams>(
        &combo->A,
        &combo->B,
        &combo->ResultX2,
        &combo->Result2XY,
        &combo->ResultY2,
        grid,
        block,
        tempProducts);
}
