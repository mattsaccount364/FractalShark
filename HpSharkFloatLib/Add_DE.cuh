

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
SerialCarryPropagation (
    uint32_t *shared_data,
    uint32_t *globalSync1,
    uint64_t *final128,
    const int32_t numActualDigitsPlusGuard,
    uint32_t *carry,      // global memory array for intermediate carries (length numActualDigitsPlusGuard+1)
    int32_t &outExponent,  // note: if you need the updated exponent outside, you might pass this by reference
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid) {
    // (For brevity, we perform a sequential prefix scan in thread 0.
    // In a production code you would replace this with a parallel prefix-sum algorithm.)
    if (block.thread_index().x == 0 && block.group_index().x == 0) {
        if (sameSign) {
            // Addition: propagate carry.
            uint64_t carry = 0;
            for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
                uint64_t sum = (uint64_t)final128[i] + carry;
                final128[i] = (uint32_t)(sum & 0xFFFFFFFF);
                carry = sum >> 32;
            }
            uint32_t finalCarry = (uint32_t)carry;
            if (finalCarry > 0) {
                outExponent += 1;
                // Right-shift the extended result by one bit.
                uint32_t nextBit = finalCarry & 1;
                for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; i--) {
                    uint32_t current = final128[i];
                    final128[i] = (current >> 1) | (nextBit << 31);
                    nextBit = current & 1;
                }
            }
        } else {
            // Subtraction: propagate borrow.
            uint32_t borrow = 0;
            for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
                int64_t diffVal = (int64_t)final128[i] - borrow;
                if (diffVal < 0) {
                    diffVal += ((uint64_t)1 << 32);
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                final128[i] = (uint32_t)(diffVal & 0xFFFFFFFF);
            }
            // Optionally, check that the final borrow is zero.
            assert(borrow == 0);
        }
    }
}

// Define our custom combine operator as a device lambda.
static __device__ uint32_t
CombineBorrow (
    uint32_t x,
    uint32_t y
) {
    // x and y are encoded as: (sat << 1) | b.
    uint32_t sat_x = (x >> 1) & 1;
    uint32_t b_x = x & 1;
    uint32_t sat_y = (y >> 1) & 1;
    uint32_t b_y = y & 1;
    // The combined borrow propagates if:
    //   new_b = b_y OR (b_x AND sat_x)
    uint32_t new_b = b_y | (b_x & sat_x);
    // The saturation value is simply taken from the right element.
    return (sat_y << 1) | new_b;
}

// A small structure to hold the generate/propagate pair for a digit.
struct GenProp {
    uint32_t g; // generate: indicates that this digit produces a carry regardless of incoming carry.
    uint32_t p; // propagate: indicates that if an incoming carry exists, it will be passed along.
};

// The combine operator for two GenProp pairs.
// If you have a block with operator f(x) = g OR (p AND x),
// then the combination for two adjacent blocks is given by:
static __device__ inline GenProp
Combine (
    const GenProp &left,
    const GenProp &right) {
    GenProp out;
    out.g = right.g | (right.p & left.g);
    out.p = right.p & left.p;
    return out;
}

template<class SharkFloatParams, int32_t CallIndex>
static __device__ SharkForceInlineReleaseOnly void
Phase1_DE (
    cg::thread_block &block,
    cg::grid_group &grid,
    const int32_t idx,
    const bool DIsBiggerMagnitude,
    const bool IsNegativeD,
    const bool IsNegativeE,
    const int32_t numActualDigitsPlusGuard,
    const int32_t numActualDigits,
    const auto *ext_D_2X,
    const auto *ext_E_B,
    const int32_t shiftD,
    const int32_t shiftE,
    const int32_t effExpD,
    const int32_t effExpE,
    const int32_t newDExponent,
    const int32_t newEExponent,
    int32_t &outExponent_DE,
    uint64_t *final128,  // the extended result digits
    DebugState<SharkFloatParams> *debugStates)
{
    const bool sameSignDE = (IsNegativeD == IsNegativeE);
    const int32_t diffDE = DIsBiggerMagnitude ? (effExpD - effExpE) : (effExpE - effExpD);
    outExponent_DE = DIsBiggerMagnitude ? newDExponent : newEExponent;

    // --- Each thread computes its aligned limb.
    for (int32_t i = idx; i < numActualDigitsPlusGuard; i += blockDim.x * gridDim.x) {
        uint64_t alignedA = 0, alignedB = 0;

        uint64_t prelim = 0;
        if (sameSignDE) {
            GetCorrespondingLimbs<SharkFloatParams>(
                ext_D_2X,
                numActualDigits,
                numActualDigitsPlusGuard,
                ext_E_B,
                numActualDigits,
                numActualDigitsPlusGuard,
                shiftD,
                shiftE,
                DIsBiggerMagnitude,
                diffDE,
                i,
                alignedA,
                alignedB);
            prelim = alignedA + alignedB;
        } else {
            // ---- Subtraction Branch ----
            if (DIsBiggerMagnitude) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2X,
                    numActualDigits,
                    numActualDigitsPlusGuard,
                    ext_E_B,
                    numActualDigits,
                    numActualDigitsPlusGuard,
                    shiftD,
                    shiftE,
                    DIsBiggerMagnitude,
                    diffDE,
                    i,
                    alignedA,
                    alignedB);
                int64_t diffVal = (int64_t)alignedA - (int64_t)alignedB;
                prelim = diffVal;
            } else {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(
                    ext_D_2X,
                    numActualDigits,
                    numActualDigitsPlusGuard,
                    ext_E_B,
                    numActualDigits,
                    numActualDigitsPlusGuard,
                    shiftD,
                    shiftE,
                    DIsBiggerMagnitude,
                    diffDE,
                    i,
                    alignedA,
                    alignedB);
                const int64_t diffVal = (int64_t)alignedB - (int64_t)alignedA;
                prelim = diffVal;
            }
        }

        // Write preliminary result (without carry/borrow propagation) to global temporary.
        final128[i] = prelim;
    }

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2XY, uint64_t>(
            debugStates, grid, block, final128, numActualDigitsPlusGuard);
        grid.sync();
    } else {
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ inline void CarryPropagationPP3_DE (
    uint32_t *shared_data,   // allocated with at least 2*n*sizeof(uint32_t); unused here.
    uint32_t *globalSync1,   // unused (provided for interface compatibility)
    uint64_t *final128,     // the extended result digits
    const int32_t numActualDigitsPlusGuard,  // number of digits in final128 to process
    uint32_t *carry1,         // will be reinterpreted as an array of GenProp (working vector)
    uint32_t *carry2,         // will be reinterpreted as an array of GenProp (scratch buffer)
    int32_t &outExponent,     // (unchanged; provided for interface compatibility)
    int32_t &carryAcc_DE, // will be set to 1 if there is an overall carry; 0 otherwise
    bool sameSign,            // true for addition; false for subtraction (borrow propagation)
    cg::thread_block &block,
    cg::grid_group &grid) {
    // ==== boilerplate from your original ====
    const int32_t globalIdx = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int32_t   stride = grid.size();
    constexpr int32_t maxIter = 1000;
    auto *carries_remaining = &globalSync1[0];
    uint32_t *curCarry = carry1;
    uint32_t *nextCarry = carry2;

    // initialize the global counter
    if (globalIdx == 0)
        *carries_remaining = 1;

    // zero out both carry buffers
    for (int32_t i = globalIdx; i <= numActualDigitsPlusGuard; i += stride) {
        carry1[i] = carry2[i] = 0;
    }
    grid.sync();


    //
    // ==== NEW: WARP-LEVEL PROPAGATION ====
    //
    // Within each 32-thread warp, we do a single shuffle-based prefix scan
    // of the per-digit carry-out (or borrow-out) flags.  That handles all
    // dependencies *inside* a warp in one go.
    //
    // We'll write the per-warp prefix results into nextCarry[i+1].
    //
    auto warpLevel = [](
        cg::grid_group &grid,
        cg::thread_block &block,
        uint64_t *final128,
        uint32_t *carries_remaining,
        int32_t globalIdx,
        int32_t numActualDigitsPlusGuard,
        bool    sameSign,
        uint32_t *&curCarry,
        uint32_t *&nextCarry
        ) {
            // We need to synchronize the warp before we start.
            __syncwarp();

            static constexpr int W = 32;
            const unsigned mask = __activemask();

            const auto numThreadsInMask = __popc(mask); // number of threads in this warp

            // total threads in grid
            const auto totalThreads = grid.size();

            // compute how many warps we'd need to cover the digits
            const auto numBlocks = grid.group_dim().x;
            const auto physicalWarpsPerBlock = (block.size() + W - 1) / W;
            //const auto numPhysicalWarps = physicalWarpsPerBlock * numBlocks;
            //const auto numLogicalWarpsRequired = (numActualDigitsPlusGuard + (W - 1)) / W;
            const auto chunksRequired = (numActualDigitsPlusGuard + grid.size() - 1) / grid.size();

            // each thread's flat ID
            const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
            const auto tidInBlock = block.thread_index().x;
            const auto warpId = physicalWarpsPerBlock * block.group_index().x;
            const auto warpIdInBlock = tidInBlock / W;
            const auto lane = tidInBlock % W; // thread ID within the warp

            //
            // Rethink all this logic.  It doesn't work and it's not super-obvious that
            // redoing it would even yield better performance.  The required grid.sync()
            // calls are expensive, and it doesn't seem like this approach is worth it.
            //

            for (int chunk = 0; chunk < chunksRequired; chunk++) {
                const auto base = chunk * totalThreads;
                const auto i = base + physicalWarpsPerBlock * warpIdInBlock + lane;
                const bool doWork = (i >= base) && (i < numActualDigitsPlusGuard);

                uint32_t sum = 0, carry = 0;
                if (doWork) {
                    // STEP-1: add/sub
                    const uint64_t old = final128[i];
                    const uint32_t inc = curCarry[i];
                    const uint64_t tmp = sameSign
                        ? old + inc
                        : uint64_t(int64_t(old) - inc);
                    sum = uint32_t(tmp);
                    carry = uint32_t(tmp >> 32);
                }

                // STEP-2: ripple across the physical warp (all W lanes)
                for (int s = 0; s < W - 1; ++s) {
                    // grab the carry bit from lane-1
                    const uint32_t leftCarry = __shfl_up_sync(mask, carry, 1);

                    // decide whether this digit will propagate any incoming carry:
                    //   for addition: propagate only if sum == 0xFFFFFFFFu
                    //   for subtraction: propagate only if sum == 0
                    const bool willProp = sameSign
                        ? (sum == 0xFFFFFFFFu)
                        : (sum == 0u);

                    if (lane >= 1 && willProp) {
                        // pull in that 1-bit carry from our left neighbor
                        carry = leftCarry;
                        sum += carry;        // update our digit
                    } else {
                        carry = 0;            // either no left carry, or we don't propagate
                    }
                }

                // STEP-3: only the "doWork" threads write out
                if (doWork) {
                    final128[i] = sum;        // now fully in-warp carried
                    nextCarry[i + 1] = carry;
                    if (carry) atomicAdd(carries_remaining, 1u);
                }
            }
        };



    //
    // ==== BELOW: your original grid-wide iterative converge, but it only
    //     has to handle *inter-warp* carries now.  We restrict updates
    //     to positions where i % 32 == 0 (the warp boundaries).
    //

    // Set an initial nonzero carry flag.
    auto *carries_remaining_global = &globalSync1[0];
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *carries_remaining_global = 1;
    }

    auto existing_carries_remaining = 0;

    // Swap carry buffers (also executed by every thread)
    //{
    //    auto *tmp = curCarry;
    //    curCarry = nextCarry;
    //    nextCarry = tmp;
    //}

    if (sameSign) {
        // --- Addition: Propagate carry ---
        // Loop a fixed number of iterations. For random numbers the carry
        // propagation will usually "settle" in only a few iterations.
        for (int32_t iter = 0; iter < maxIter; iter++) {

            warpLevel(
                grid,
                block,
                final128,
                carries_remaining_global,
                globalIdx,
                numActualDigitsPlusGuard,
                sameSign,
                curCarry,
                nextCarry
            );

            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            grid.sync();

            uint32_t totalNewCarry = 0;
            // Each thread updates its assigned digits in a grid-stride loop.
            for (int32_t i = globalIdx; i < numActualDigitsPlusGuard; i += stride) {
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                uint64_t sum = final128[i] + incoming;
                uint32_t newDigit = (uint32_t)(sum & 0xFFFFFFFFu);
                uint32_t newCarry = (uint32_t)(sum >> 32);
                final128[i] = newDigit;
                if (i < numActualDigitsPlusGuard - 1) {
                    nextCarry[i + 1] = newCarry;
                } else {
                    // Always merge the new carry with the previous final carry.
                    nextCarry[i + 1] = curCarry[i + 1] | newCarry;
                }

                totalNewCarry += newCarry;
            }

            grid.sync();

            if (totalNewCarry > 0) {
                // Atomically accumulate the new carry count.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // After the loop, thread 0 checks if a final carry remains.
        uint32_t finalCarry = curCarry[numActualDigitsPlusGuard];
        if (finalCarry > 0u) {
            carryAcc_DE = 1;
        }

    } else {
        // --- Subtraction: Propagate borrow ---
        for (int32_t iter = 0; iter < maxIter; iter++) {

            warpLevel(
                grid,
                block,
                final128,
                carries_remaining_global,
                globalIdx,
                numActualDigitsPlusGuard,
                sameSign,
                curCarry,
                nextCarry
            );

            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            // Reset the counter for the next iteration.
            grid.sync();

            uint32_t totalNewBorrow = 0;
            // Each thread processes its grid-stride subset.
            for (int32_t i = globalIdx; i < numActualDigitsPlusGuard; i += stride) {
                // For digit 0, no incoming borrow.
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                int64_t diffVal = (int64_t)final128[i] - incoming;
                uint32_t newBorrow = 0;

                if (diffVal < 0) {
                    diffVal += (1ULL << 32);
                    newBorrow = 1;
                }

                final128[i] = (uint32_t)(diffVal & 0xFFFFFFFFu);
                nextCarry[i + 1] = newBorrow;
                totalNewBorrow += newBorrow;
            }

            grid.sync();

            if (totalNewBorrow > 0) {
                // Accumulate new borrows into the global counter.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // Finally, thread 0 checks that no borrow remains.
        if (globalIdx == 0) {
            assert(curCarry[numActualDigitsPlusGuard] == 0);
        }
    }

    grid.sync();
}

//
// Works but slow
//

template <class SharkFloatParams>
__device__ inline void CarryPropagationPP2_DE(
    uint32_t *shared_data,   // allocated with at least 2*n*sizeof(uint32_t); unused here.
    uint32_t *globalSync1,   // unused (provided for interface compatibility)
    uint64_t *final128,     // the extended result digits
    const int32_t numActualDigitsPlusGuard,  // number of digits in final128 to process
    uint32_t *carry1,         // will be reinterpreted as an array of GenProp (working vector)
    uint32_t *carry2,         // will be reinterpreted as an array of GenProp (scratch buffer)
    int32_t &outExponent,     // (unchanged; provided for interface compatibility)
    int32_t &carryAcc_DE, // will be set to 1 if there is an overall carry; 0 otherwise
    bool sameSign,            // true for addition; false for subtraction (borrow propagation)
    cg::thread_block &block,
    cg::grid_group &grid) {
    // Determine grid-stride parameters.
    const int32_t totalThreads = grid.size();
    const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    // Reinterpret carry1 and carry2 as arrays of GenProp.
    GenProp *working = reinterpret_cast<GenProp *>(carry1);  // working array
    GenProp *scratch = reinterpret_cast<GenProp *>(carry2);    // scratch array

    //--------------------------------------------------------------------------
    // Phase 1: Initialize the per-digit signals.
    //
    // For addition (sameSign):
    //   working[i].g = (high 32 bits of final128[i] != 0) ? 1 : 0;
    //   working[i].p = (low 32 bits of final128[i] == 0xFFFFFFFF) ? 1 : 0;
    //
    // For subtraction:
    //   working[i].g = (int64_t(final128[i]) < 0) ? 1 : 0;
    //   working[i].p = ((low 32 bits == 0) && (high 32 bits == 0)) ? 1 : 0;
    //--------------------------------------------------------------------------
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        uint64_t digit = final128[i];
        uint32_t lo = static_cast<uint32_t>(digit);
        uint32_t hi = static_cast<uint32_t>(digit >> 32);
        if (sameSign) {
            working[i].g = (hi != 0) ? 1 : 0;
            working[i].p = (lo == 0xFFFFFFFFu) ? 1 : 0;
        } else {
            int64_t raw = static_cast<int64_t>(digit);
            working[i].g = (raw < 0) ? 1 : 0;
            working[i].p = ((lo == 0) && (hi == 0)) ? 1 : 0;
        }
    }
    grid.sync();

    //--------------------------------------------------------------------------
    // Phase 2: Perform an inclusive scan on the working vector.
    //
    // We perform log2(numActualDigitsPlusGuard) passes. In each pass, each thread processes 
    // one or more elements via a grid-stride loop. For each element i >= offset,
    // the new signal is computed as:
    //    scratch[i] = Combine(working[i-offset], working[i])
    // Otherwise, we simply copy working[i] to scratch[i].
    // Then we swap working and scratch.
    //--------------------------------------------------------------------------
    if (numActualDigitsPlusGuard > 1) {
        for (int32_t offset = 1; offset < numActualDigitsPlusGuard; offset *= 2) {
            for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
                if (i >= offset) {
                    scratch[i] = Combine(working[i - offset], working[i]);
                } else {
                    scratch[i] = working[i];
                }
            }
            grid.sync();

            // Swap pointers so that 'working' always points to the most up-to-date array.
            GenProp *temp = working;
            working = scratch;
            scratch = temp;
            grid.sync();
        }
    }
    // At this point, working[0..numActualDigitsPlusGuard-1] holds the inclusive scan results.

    //--------------------------------------------------------------------------
    // Phase 3: Convert to an exclusive scan and update the digits.
    //
    // The exclusive scan is defined by taking an identity for index 0 and
    // for i >= 1 using working[i-1]. (The identity is equivalent to a GenProp 
    // of {0, 1}, but since the overall initial carry is 0, we simply use 0.)
    //
    // For addition: update each digit with final128[i] = (final128[i] + incomingCarry) & 0xFFFFFFFF.
    // For subtraction: subtract the incoming borrow.
    //--------------------------------------------------------------------------
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        uint32_t incoming = (i == 0) ? 0 : working[i - 1].g;
        if (sameSign) {
            uint64_t sum = final128[i] + incoming;
            final128[i] = sum & 0xFFFFFFFFULL;
        } else {
            int64_t diff = static_cast<int64_t>(final128[i]) - incoming;
            final128[i] = static_cast<uint32_t>(diff & 0xFFFFFFFFULL);
        }
    }
    grid.sync();

    //--------------------------------------------------------------------------
    // Phase 4: Final carry/borrrow update.
    //
    // For addition: the overall carry is given by working[numActualDigitsPlusGuard - 1].g.
    // If it is nonzero, then we set carryAcc_DE = 1.
    // For subtraction, we assume the final borrow is zero.
    //--------------------------------------------------------------------------
    if (sameSign) {
        uint32_t overallCarry = working[numActualDigitsPlusGuard - 1].g;
        carryAcc_DE = (overallCarry > 0) ? 1 : 0;
    }
    grid.sync();
}

//
// Older implementation buggy with add samesign == true:
//

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
CarryPropagationPPTry1Buggy_DE (
    uint32_t *shared_data,  // must be allocated with at least 2*n*sizeof(uint32_t)
    uint32_t *globalSync1,  // unused in this version (still provided for interface compatibility)
    uint64_t *final128,
    const int32_t numActualDigitsPlusGuard,
    uint32_t *carry1,
    uint32_t *carry2,
    int32_t &outExponent,
    int32_t &carryAcc_DE,
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid) {
    // We assume that numActualDigitsPlusGuard is small. Let n be the next power of two >= numActualDigitsPlusGuard.
    const auto n = numActualDigitsPlusGuard;

    // We use shared_data to hold two arrays (each of length n):
    // s_g[0..n-1] will hold the "generate" flag (0 or 1) for each digit,
    // s_p[0..n-1] will hold the "propagate" flag (0 or 1).
    // (For the exclusive scan we use the Blelloch algorithm.)
    uint32_t *s_g = carry1;       // first n elements
    uint32_t *s_p = carry2;       // next n elements

    static constexpr auto SequentialBits = true;

    const int32_t totalThreads = grid.size();
    const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    if (sameSign) {
        // --- Initialization ---
        // Only the first numActualDigitsPlusGuard threads load a digit; for i >= numActualDigitsPlusGuard we initialize to identity: (0,1).
        for (int32_t i = tid; i < n; i += totalThreads) {
            // i < numActualDigitsPlusGuard always here
            uint64_t x = final128[i];
            uint32_t low = (uint32_t)x;
            uint32_t hi = (uint32_t)(x >> 32);
            s_g[i] = (hi == 1) ? 1 : 0;
            s_p[i] = (low == 0xFFFFFFFF) ? 1 : 0;
        }
        grid.sync();

        // --- Upsweep phase (reduce) ---
        // For d = 1,2,4,..., n/2, each thread whose index fits combines a pair of nodes.
        if constexpr (!SequentialBits) {
            for (int32_t d = 1; d < n; d *= 2) {
                int32_t index = (tid + 1) * d * 2 - 1;  // each thread works on one index
                if (index < n) {
                    uint32_t g1 = s_g[index - d];
                    uint32_t p1 = s_p[index - d];
                    uint32_t g2 = s_g[index];
                    uint32_t p2 = s_p[index];
                    // Combine according to our operator:
                    // (g, p) = (g2 OR (p2 AND g1), p2 AND p1)
                    s_g[index] = g2 | (p2 & g1);
                    s_p[index] = p2 & p1;
                }
                grid.sync();
            }

            // --- Set the last element to the identity (for exclusive scan) ---
            if (tid == 0) {
                s_g[n - 1] = 0;
                s_p[n - 1] = 1;
            }
        } else {
            // Upsweep phase (reduce) – corrected operator (note the order of operands)
            if (block.thread_index().x == 0 && block.group_index().x == 0) {
                for (int32_t d = 1; d < n; d *= 2) {
                    // Process every segment of 2*d elements.
                    for (int32_t index = 2 * d - 1; index < n; index += 2 * d) {
                        // left child is at index - d, right child is at index.
                        uint32_t left_g = s_g[index - d];
                        uint32_t left_p = s_p[index - d];
                        uint32_t right_g = s_g[index];
                        uint32_t right_p = s_p[index];
                        // Combine using the left-to-right operator:
                        // (g, p) = (left_g OR (left_p & right_g), left_p AND right_p)
                        s_g[index] = left_g | (left_p & right_g);
                        s_p[index] = left_p & right_p;
                    }
                }
            }
            grid.sync();

            // --- Set the last element to the identity (for exclusive scan) ---
            if (tid == 0) {
                s_g[n - 1] = 0;
                s_p[n - 1] = 1;
            }
        }
        grid.sync();

        // --- Downsweep phase (exclusive scan for addition) ---
        // Now perform the downsweep: update only indices k satisfying ((k+1) mod (2*d)) == 0.
        if constexpr (!SequentialBits) {
            const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

            for (int32_t d = n / 2; d >= 1; d /= 2) {
                // Use grid-stride loops to cover all indices.
                for (int32_t k = tid;
                    k < n;
                    k += grid.size()) {
                    // Check if k is the last index of its block of 2*d elements.
                    if (((k + 1) % (2 * d)) == 0) {
                        uint32_t temp = s_g[k - d];
                        s_g[k - d] = s_g[k];
                        if (d == 1) {
                            // On the final downsweep pass, simply assign the left value.
                            s_g[k] = temp;
                        } else {
                            s_g[k] = s_g[k] | (s_p[k] & temp); // standard combine for d > 1
                        }
                    }
                }

                grid.sync();
            }
        } else {
            if (block.thread_index().x == 0 && block.group_index().x == 0) {
                for (int32_t d = n / 2; d >= 1; d /= 2) {
                    for (int32_t base = 0; base < n; base += 2 * d) {
                        int32_t k = base + 2 * d - 1; // rightmost node of the segment
                        uint32_t temp = s_g[k - d]; // save left child's g
                        s_g[k - d] = s_g[k];        // set left value to what the right child held
                        if (d == 1)
                            s_g[k] = temp;
                        else
                            s_g[k] = temp | (s_p[k - d] & s_g[k]);
                    }
                }
            }

            grid.sync();
        }

        // Now s_g[0..numActualDigitsPlusGuard-1] contains the exclusive scan results.
        // In particular, for digit i the carry into that digit is given by s_g[i].
        grid.sync();

        // --- Update digits using the computed carries ---
        auto original_last_digit = final128[numActualDigitsPlusGuard - 1]; // save the original last digit
        grid.sync();
        for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
            uint32_t carryIn = s_g[i]; // carry into digit i
            uint64_t sum = final128[i] + carryIn;
            // Write the 32-bit result back.
            final128[i] = sum & 0xFFFFFFFF;
        }
        grid.sync();

        // --- Determine overall final carry ---
        uint32_t carryIn = s_g[numActualDigitsPlusGuard - 1];
        uint64_t lastVal = original_last_digit; // saved before updating final128[numActualDigitsPlusGuard-1]
        uint64_t S = lastVal + carryIn;
        uint32_t finalCarry = S >> 32;
        if (finalCarry > 0u) {
            carryAcc_DE = 1;
        }
    } else {
        // --- Subtraction branch (parallel custom scan for borrow propagation) ---
        // For subtraction we assume the final result is nonnegative so no borrow should remain.
        // The sequential logic is:
        //
        //   borrow[0] = 0,
        //   for i >= 1:
        //     borrow[i] = ( (final128[i-1] - incoming) < 0 )
        //               and then, if final128[i-1] (as 32-bit) equals 0xFFFFFFFF, the borrow propagates.
        //
        // We encode for each digit i (for i>=1) a 2-bit value:
        //   s_g[i] = (sat << 1) | b,
        // where b = 1 if final128[i-1] (the 64-bit preliminary difference) is negative,
        // and sat = 1 if the lower 32 bits of final128[i-1] equal 0xFFFFFFFF.
        //
        // For i == 0, no incoming borrow: we use 0.
        // For indices i >= numActualDigitsPlusGuard (padding) we use the identity element for our operator,
        // which we choose as (sat=1, b=0) --> value 2.
        for (int32_t i = tid; i < n; i += totalThreads) {
            if (i == 0) {
                // No incoming borrow.
                s_g[0] = 0;
            } else if (i < numActualDigitsPlusGuard) {
                uint32_t b = (((int64_t)final128[i - 1] < 0) ? 1 : 0);
                uint32_t sat = (((uint32_t)final128[i - 1] == 0xFFFFFFFFu) ? 1 : 0);
                s_g[i] = (sat << 1) | b;
            } else {
                // For padded indices use the identity element (sat=1, b=0) so that it does not cancel a borrow.
                s_g[i] = 2;
            }
            // s_p is not used in this branch.
        }
        grid.sync();

        // --- Upsweep phase using custom operator ---
        for (int32_t d = 1; d < n; d *= 2) {
            // Each thread processes one or more indices via a grid-stride loop.
            for (int32_t index = (tid + 1) * d * 2 - 1; index < n; index += grid.size() * d * 2) {
                s_g[index] = CombineBorrow(s_g[index - d], s_g[index]);
            }
            grid.sync();
        }

        // --- Set the last element to the identity (0) for the downsweep ---
        if (tid == 0) {
            s_g[n - 1] = 2;
        }
        grid.sync();

        const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

        // --- Downsweep phase using custom operator ---
        // Parallel Hillis–Steele inclusive scan using CombineBorrow.
        // Note: 'I' is the identity for our CombineBorrow operator; for our encoding, we take I = 2.
        // (Assumes numActualDigitsPlusGuard is <= the number of elements in s_g.)
        for (int32_t d = 1; d < numActualDigitsPlusGuard; d *= 2) {
            // Each thread processes indices in grid stride.
            for (int32_t i = tid;
                i < numActualDigitsPlusGuard;
                i += grid.size()) {
                // For indices with i >= d, combine the element from i-d with s_g[i].
                uint32_t prev = (i >= d) ? s_g[i - d] : 2;  // Use identity for out-of-bound indexes.
                uint32_t cur = s_g[i];
                // Compute the inclusive scan value at index i.
                uint32_t combined = (i >= d) ? CombineBorrow(prev, cur) : cur;
                // Write this back to s_g[i]. (It is ok if the update is not strictly barrier-synchronized
                // between iterations, as long as we put an overall __syncthreads() after each step.)
                s_g[i] = combined;
            }
            grid.sync();
        }

        // Now s_g[0..numActualDigitsPlusGuard-1] holds, in its lower bit, the incoming borrow for each digit.
        // Decode the borrow flag (the lower bit) and update final128 accordingly.
        for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
            uint32_t borrow = s_g[i] & 1;
            uint64_t diffVal = (uint64_t)final128[i] - borrow;
            final128[i] = (uint32_t)(diffVal & 0xFFFFFFFFu);
        }
    }
    grid.sync();
}


template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
CarryPropagationDE (
    uint32_t *shared_data,
    uint32_t *globalSync1,   // global sync array; element 0 is used for borrow/carry count
    uint64_t *final128,
    const int32_t numActualDigitsPlusGuard,
    uint32_t *carry1,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
    uint32_t *carry2,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
    int32_t &carryAcc_DE,
    bool sameSign,
    cg::thread_block &block,
    cg::grid_group &grid) {

    // Compute a grid-global thread id and stride.
    const int32_t globalIdx = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int32_t stride = grid.size();

    // We'll use a fixed number of iterations.
    constexpr int32_t maxIter = 1000;

    // Pointer to a single global counter used for convergence.
    auto *carries_remaining_global = &globalSync1[0];
    auto *curCarry = carry1;
    auto *nextCarry = carry2;

    // Set an initial nonzero carry flag.
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *carries_remaining_global = 1;
    }

    auto existing_carries_remaining = 0;

    // Zero out the carry array.
    for (int32_t i = globalIdx; i < numActualDigitsPlusGuard; i += stride) {
        carry1[i] = 0;
        carry2[i] = 0;
    }
    grid.sync();

    if (sameSign) {
        // --- Addition: Propagate carry ---
        // Loop a fixed number of iterations. For random numbers the carry
        // propagation will usually "settle" in only a few iterations.
        for (int32_t iter = 0; iter < maxIter; iter++) {

            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            grid.sync();

            uint32_t totalNewCarry = 0;
            // Each thread updates its assigned digits in a grid-stride loop.
            for (int32_t i = globalIdx; i < numActualDigitsPlusGuard; i += stride) {
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                uint64_t sum = final128[i] + incoming;
                uint32_t newDigit = (uint32_t)(sum & 0xFFFFFFFFu);
                uint32_t newCarry = (uint32_t)(sum >> 32);
                final128[i] = newDigit;
                if (i < numActualDigitsPlusGuard - 1) {
                    nextCarry[i + 1] = newCarry;
                } else {
                    // Always merge the new carry with the previous final carry.
                    nextCarry[i + 1] = curCarry[i + 1] | newCarry;
                }

                totalNewCarry += newCarry;
            }

            grid.sync();

            if (totalNewCarry > 0) {
                // Atomically accumulate the new carry count.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // After the loop, thread 0 checks if a final carry remains.
        uint32_t finalCarry = curCarry[numActualDigitsPlusGuard];
        if (finalCarry > 0u) {
            carryAcc_DE = 1;
        }

    } else {
        // --- Subtraction: Propagate borrow ---
        for (int32_t iter = 0; iter < maxIter; iter++) {
            // Convergence check and reset the counter.
            // If no carries were produced in the previous iteration, we can exit.
            if (*carries_remaining_global == existing_carries_remaining) {
                // Break out of the loop.
                // (This branch is taken if no thread produced any new carry.)
                // Note: The counter was set during the previous iteration.
                // Reset is done before starting the next pass.
                break;
            }

            // Reset the counter for the next iteration.
            existing_carries_remaining = *carries_remaining_global;

            // Reset the counter for the next iteration.
            grid.sync();

            uint32_t totalNewBorrow = 0;
            // Each thread processes its grid-stride subset.
            for (int32_t i = globalIdx; i < numActualDigitsPlusGuard; i += stride) {
                // For digit 0, no incoming borrow.
                uint32_t incoming = (i == 0) ? 0u : curCarry[i];
                int64_t diffVal = (int64_t)final128[i] - incoming;
                uint32_t newBorrow = 0;

                if (diffVal < 0) {
                    diffVal += (1ULL << 32);
                    newBorrow = 1;
                }

                final128[i] = (uint32_t)(diffVal & 0xFFFFFFFFu);
                nextCarry[i + 1] = newBorrow;
                totalNewBorrow += newBorrow;
            }

            grid.sync();

            if (totalNewBorrow > 0) {
                // Accumulate new borrows into the global counter.
                atomicAdd(carries_remaining_global, 1);
            }

            // Swap the carry arrays for the next iteration.
            auto *temp = curCarry;
            curCarry = nextCarry;
            nextCarry = temp;

            grid.sync();
        }

        // Finally, thread 0 checks that no borrow remains.
        if (globalIdx == 0) {
            assert(curCarry[numActualDigitsPlusGuard] == 0);
        }
    }

    grid.sync();
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void FinalResolutionDE (
    const int32_t idx,
    const int32_t stride,
    const int32_t carryAcc_DE,
    const int32_t numActualDigitsPlusGuard,
    const int32_t numActualDigits,
    const uint64_t *final128_DE,
    HpSharkFloat<SharkFloatParams> *OutSharkFloat,
    int32_t &outExponent_DE
    ) {
        if (carryAcc_DE > 0) {
            // Make sure carryAcc_DE, numActualDigitsPlusGuard, numActualDigits, and final128_DE are computed and
            // available to all threads
            int32_t wordShift = carryAcc_DE / 32;
            int32_t bitShift = carryAcc_DE % 32;

            // Each thread handles a subset of indices.
            for (int32_t i = idx; i < numActualDigits; i += stride) {
                uint32_t lower = (i + wordShift < numActualDigitsPlusGuard) ? final128_DE[i + wordShift] : 0;
                uint32_t upper = (i + wordShift + 1 < numActualDigitsPlusGuard) ? final128_DE[i + wordShift + 1] : 0;
                OutSharkFloat->Digits[i] = (bitShift == 0) ? lower : (lower >> bitShift) | (upper << (32 - bitShift));

                if (i == numActualDigits - 1) {
                    // Set the high-order bit of the last digit.
                    OutSharkFloat->Digits[numActualDigits - 1] |= (1u << 31);
                }
            }

            outExponent_DE += carryAcc_DE;
        } else if (carryAcc_DE < 0) {
            int32_t wordShift = (-carryAcc_DE) / 32;
            int32_t bitShift = (-carryAcc_DE) % 32;

            for (int32_t i = idx; i < numActualDigits; i += stride) {
                int32_t srcIdx = i - wordShift;
                uint32_t lower = (srcIdx >= 0 && srcIdx < numActualDigitsPlusGuard) ? final128_DE[srcIdx] : 0;
                uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < numActualDigitsPlusGuard) ? final128_DE[srcIdx - 1] : 0;
                OutSharkFloat->Digits[i] = (bitShift == 0) ? lower : (lower << bitShift) | (upper >> (32 - bitShift));
            }

            if (idx == 0) {
                outExponent_DE -= (-carryAcc_DE);
            }
        } else {
            // No shifting needed; simply copy.  Convert to uint32_t along the way

            for (int32_t i = idx; i < numActualDigits; i += stride) {
                OutSharkFloat->Digits[i] = final128_DE[i];
            }
        }
    };