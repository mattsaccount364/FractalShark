#include "ThreeWayMagnitude.h"

static __device__ SharkForceInlineReleaseOnly ThreeWayLargestOrdering
CompareMagnitudes3Way (
    const int32_t effExpA,
    const int32_t effExpB,
    const int32_t effExpC,
    const int32_t actualDigits,
    const int32_t numActualDigitsPlusGuard,
    const int32_t shiftA,
    const int32_t shiftB,
    const int32_t shiftC,
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,
    int32_t &outExp
) {
    // Helper: returns true if "first" is strictly bigger than "second"
    auto cmp = [&](const uint32_t *e1, int32_t s1, int32_t exp1,
        const uint32_t *e2, int32_t s2, int32_t exp2) {
            if (exp1 != exp2)
                return exp1 > exp2;
            // exponents equal -> compare normalized digits high->low
            for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
                uint32_t d1 = GetNormalizedDigit(e1, actualDigits, numActualDigitsPlusGuard, s1, i);
                uint32_t d2 = GetNormalizedDigit(e2, actualDigits, numActualDigitsPlusGuard, s2, i);
                if (d1 != d2)
                    return d1 > d2;
            }
            return false;  // treat exact equality as "not greater"
        };

    // 1) Is A the strict max?
    if (cmp(extA, shiftA, effExpA, extB, shiftB, effExpB) &&
        cmp(extA, shiftA, effExpA, extC, shiftC, effExpC)) {
        return ThreeWayLargestOrdering::A_GT_AllOthers;
    }
    // 2) Is B the strict max?
    else if (cmp(extB, shiftB, effExpB, extA, shiftA, effExpA) &&
        cmp(extB, shiftB, effExpB, extC, shiftC, effExpC)) {
        return ThreeWayLargestOrdering::B_GT_AllOthers;
    }
    // 3) Otherwise C is the (strict) max
    else {
        return ThreeWayLargestOrdering::C_GT_AllOthers;
    }
}

// 5) helper to do |±X ±Y ±Z| in one pass, given a fixed X_gtY
static uint64_t __device__ SharkForceInlineReleaseOnly
CoreThreeWayAdd(
    uint64_t X, bool sX,
    uint64_t Y, bool sY,
    uint64_t Z, bool sZ,
    bool     X_gtY,
    bool &outSign
) {
    // (X vs Y)
    uint64_t magXY;
    bool     sXY;
    if (sX == sY) {
        magXY = X + Y;
        sXY = sX;
    } else if (X_gtY) {
        magXY = X - Y;
        sXY = sX;
    } else {
        magXY = Y - X;
        sXY = sY;
    }

    // (magXY vs Z)
    uint64_t mag;
    if (sXY == sZ) {
        mag = magXY + Z;
        outSign = sXY;
    } else if (X_gtY) { // reuse X_gtY as proxy for (magXY >= Z)
        mag = magXY - Z;
        outSign = sXY;
    } else {
        mag = Z - magXY;
        outSign = sZ;
    }
    return mag;
}

template<class SharkFloatParams, int32_t CallIndex>
static __device__ SharkForceInlineReleaseOnly
void Phase1_ABC (
    cg::thread_block &block,
    cg::grid_group &grid,
    const int32_t idx,
    const ThreeWayLargestOrdering ordering,
    const bool IsNegativeA,
    const bool IsNegativeB,
    const bool IsNegativeC,
    const int32_t  numActualDigitsPlusGuard,
    const int32_t  actualDigits,
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,
    const int32_t  shiftA,
    const int32_t  shiftB,
    const int32_t  shiftC,
    const int32_t  effExpA,
    const int32_t  effExpB,
    const int32_t  effExpC,
    const int32_t  bias,

    bool &outSignTrue,      // sign when X_gtY == true
    bool &outSignFalse,     // sign when X_gtY == false
    int32_t &outExpTrue_Orig,  // exponent before bias for true-branch
    int32_t &outExpFalse_Orig, // exponent before bias for false-branch
    uint64_t *extResultTrue,   // result limbs for X_gtY == true
    uint64_t *extResultFalse,   // result limbs for X_gtY == false

    DebugState<SharkFloatParams> *debugStates
)
{
    outSignTrue = false;
    outSignFalse = false;

    // 2) pick the “base” exponent from the largest input
    int32_t baseExp;
    switch (ordering) {
    case ThreeWayLargestOrdering::A_GT_AllOthers:
        baseExp = effExpA;
        break;
    case ThreeWayLargestOrdering::B_GT_AllOthers:
        baseExp = effExpB;
        break;
    case ThreeWayLargestOrdering::C_GT_AllOthers:
        baseExp = effExpC;
        break;
    default:
        assert(false);
        for (;;);
    }

    // 3) single diff per input to align to baseExp
    int32_t diffA = baseExp - effExpA;
    int32_t diffB = baseExp - effExpB;
    int32_t diffC = baseExp - effExpC;

    // 4) pick pointers, signs, shifts and diffs in “X, Y, Z” order
    const uint32_t *extX, *extY, *extZ;
    bool  sX, sY, sZ;
    int32_t shX, shY, shZ, diffY, diffZ;

    switch (ordering) {
    case ThreeWayLargestOrdering::A_GT_AllOthers:
        extX = extA;
        sX = IsNegativeA;
        shX = shiftA;

        extY = extB;
        sY = IsNegativeB;
        shY = shiftB;
        diffY = diffB;

        extZ = extC;
        sZ = IsNegativeC;
        shZ = shiftC;
        diffZ = diffC;
        break;

    case ThreeWayLargestOrdering::B_GT_AllOthers:
        extX = extB;
        sX = IsNegativeB;
        shX = shiftB;

        extY = extA;
        sY = IsNegativeA;
        shY = shiftA;
        diffY = diffA;

        extZ = extC;
        sZ = IsNegativeC;
        shZ = shiftC;
        diffZ = diffC;
        break;

    case ThreeWayLargestOrdering::C_GT_AllOthers:
        extX = extC;
        sX = IsNegativeC;
        shX = shiftC;

        extY = extA;
        sY = IsNegativeA;
        shY = shiftA;
        diffY = diffA;

        extZ = extB;
        sZ = IsNegativeB;
        shZ = shiftB;
        diffZ = diffB;
        break;

    default:
        assert(false);
        for (;;);
    }

    // 6) single pass: two calls per digit, grid-stride version
    {
        int32_t stride = grid.size();
        for (int32_t i = idx; i < numActualDigitsPlusGuard; i += stride) {
            uint64_t Xi = GetNormalizedDigit(
                extX, actualDigits, numActualDigitsPlusGuard, shX, i);
            uint64_t Yi = GetShiftedNormalizedDigit<SharkFloatParams>(
                extY, actualDigits, numActualDigitsPlusGuard, shY, diffY, i);
            uint64_t Zi = GetShiftedNormalizedDigit<SharkFloatParams>(
                extZ, actualDigits, numActualDigitsPlusGuard, shZ, diffZ, i);

            // always-true branch
            extResultTrue[i] = CoreThreeWayAdd(Xi, sX, Yi, sY, Zi, sZ, /*X_gtY=*/true, outSignTrue);
            // always-false branch
            extResultFalse[i] = CoreThreeWayAdd(Xi, sX, Yi, sY, Zi, sZ, /*X_gtY=*/false, outSignFalse);
        }
    }

    // 7) both exponents (before re-bias) are just baseExp - bias
    outExpTrue_Orig = baseExp - bias;
    outExpFalse_Orig = baseExp - bias;

    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm1, uint64_t>(
            debugStates, grid, block, extResultTrue, numActualDigitsPlusGuard);
        StoreCurrentDebugStateAdd<SharkFloatParams, DebugStatePurpose::Z2_Perm2, uint64_t>(
            debugStates, grid, block, extResultFalse, numActualDigitsPlusGuard);
        grid.sync();
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
CarryPropagationSmall_ABC(uint32_t *globalSync1,                   // [0] holds convergence counter
                     const int32_t idx,                      // this thread’s global index
                     const int32_t numActualDigitsPlusGuard, // N
                     uint64_t *extResultTrue,                // Phase1_ABC “true” limbs
                     uint64_t *extResultFalse,               // Phase1_ABC “false” limbs
                     uint64_t *final128_DE,                  // Phase1_DE limbs
                     uint32_t *carry1,                       // length N+1
                     uint32_t *carry2,                       // length N+1
                     uint32_t *carry3,                       // length N+1
                     uint32_t *carry4,                       // length N+1
                     uint32_t *carry5,                       // length N+1
                     uint32_t *carry6,                       // length N+1
                     int32_t &carryAcc_ABC_True,             // out: final signed carry/borrow
                     int32_t &carryAcc_ABC_False,            // out: final signed carry/borrow
                     int32_t &carryAcc_DE,                   // out: final unsigned carry
                     cg::thread_block &block,
                     cg::grid_group &grid)
{
    constexpr uint64_t SHIFT1 = 21;
    constexpr uint64_t SHIFT2 = 42;
    constexpr uint64_t FIELD_MASK = (1ULL << 21) - 1;

    const int32_t N = numActualDigitsPlusGuard;
    const int32_t stride = grid.size();
    auto *global64 = reinterpret_cast<uint64_t *>(globalSync1);

    // six carry buffers
    uint32_t *cur1 = carry1;
    uint32_t *next1 = carry2;
    uint32_t *cur2 = carry3;
    uint32_t *next2 = carry4;
    uint32_t *cur3 = carry5;
    uint32_t *next3 = carry6;

    // one‐time zeroing
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        *global64 = 1ULL;
    }

    for (int32_t i = idx; i < N + 1; i += stride) {
        cur1[i] = next1[i] = 0u;
        cur2[i] = next2[i] = 0u;
        cur3[i] = next3[i] = 0u;
    }

    grid.sync();

    uint64_t prevCount = 0;
    uint32_t prevCount1 = 0;
    uint32_t prevCount2 = 0;
    uint32_t prevCount3 = 0;

    bool need1 = true;
    bool need2 = true;
    bool need3 = true;

    for (int32_t iter = 0;; ++iter) {
        // ── barrier 1: reset convergence mask ──
        if (iter > 0 && *global64 == prevCount) { //  TODO
            break;
        }

        prevCount = *global64;
        prevCount1 = prevCount & FIELD_MASK;
        prevCount2 = (prevCount >> SHIFT1) & FIELD_MASK;
        prevCount3 = (prevCount >> SHIFT2) & FIELD_MASK;

        grid.sync();

        // per‐thread packed flag
        uint64_t localMask = 0ULL;

        // grid‐stride per‐digit work
        for (int32_t i = idx; i < N; i += stride) {
            // ABC_True
            if (need1) {
                const int32_t in1 = (i == 0 ? 0 : static_cast<int32_t>(cur1[i]));
                const int64_t limb1 = static_cast<int64_t>(extResultTrue[i]);
                const int64_t sum1 = limb1 + in1;
                const uint32_t lo1 = static_cast<uint32_t>(sum1);
                extResultTrue[i] = lo1;
                const int32_t new1 = int32_t((sum1 - static_cast<int64_t>(lo1)) >> 32);
                if (i < N - 1) {
                    next1[i + 1] = static_cast<uint32_t>(new1);
                } else {
                    next1[i + 1] = cur1[i + 1] + static_cast<uint32_t>(new1);
                }
                localMask += static_cast<int64_t>(new1 != 0);
            }

            // ABC_False
            if (need2) {
                const int32_t in2 = (i == 0 ? 0 : static_cast<int32_t>(cur2[i]));
                const int64_t limb2 = static_cast<int64_t>(extResultFalse[i]);
                const int64_t sum2 = limb2 + in2;
                const uint32_t lo2 = static_cast<uint32_t>(sum2);
                extResultFalse[i] = lo2;
                const int32_t new2 = int32_t((sum2 - static_cast<int64_t>(lo2)) >> 32);
                if (i < N - 1) {
                    next2[i + 1] = static_cast<uint32_t>(new2);
                } else {
                    next2[i + 1] = cur2[i + 1] + static_cast<uint32_t>(new2);
                }
                localMask += static_cast<int64_t>(new2 != 0) << SHIFT1;
            }

            // D+E (DE)
            if (need3) {
                const int32_t in3 = (i == 0 ? 0u : static_cast<int32_t>(cur3[i]));
                const int64_t limb3 = static_cast<int64_t>(final128_DE[i]);
                const int64_t sum3 = limb3 + in3;
                const uint32_t lo3 = static_cast<uint32_t>(sum3);
                final128_DE[i] = lo3;
                const int32_t new3 = int32_t((sum3 - static_cast<int64_t>(lo3)) >> 32);
                if (i < N - 1) {
                    next3[i + 1] = new3;
                } else {
                    next3[i + 1] = cur3[i + 1] + new3;
                }
                localMask += static_cast<int64_t>(new3 != 0) << SHIFT2;
            }
        }

        // atomic pack of all three flags
        if (localMask) {
            atomicAdd(global64, localMask);
        }

        // ── barrier 2: ensure all atomicAdds done before reading ──
        grid.sync();

        const uint64_t allCounts = *global64;

        need1 = ((allCounts >> 0) & FIELD_MASK) != prevCount1;
        need2 = ((allCounts >> SHIFT1) & FIELD_MASK) != prevCount2;
        need3 = ((allCounts >> SHIFT2) & FIELD_MASK) != prevCount3;

        // swap only the active streams
        if (need1) {
            std::swap(cur1, next1);
        }

        if (need2) {
            std::swap(cur2, next2);
        }

        if (need3) {
            std::swap(cur3, next3);
        }

        // ── barrier 3: ready for next iteration ──
        grid.sync();
    }

    // final carry extraction
    carryAcc_ABC_True = static_cast<int32_t>(cur1[N]);
    carryAcc_ABC_False = static_cast<int32_t>(cur2[N]);
    carryAcc_DE = static_cast<int32_t>(cur3[N]);
    grid.sync();
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ProcessStreamABC(
    const int32_t numActualDigitsPlusGuard,
    int32_t base,
    int32_t lane,
    int64_t &limb,
    int32_t inStep,
    int32_t &r,
    int32_t &c_out,
    uint64_t *extResult,
    uint32_t maskBit,
    uint32_t &changedMask)
{
    const auto N = numActualDigitsPlusGuard;
    constexpr auto warpSz = 32;
    const int64_t sum = limb + inStep;
    const uint32_t lo = static_cast<uint32_t>(sum);

    if (base + lane < N) {
        c_out = static_cast<int32_t>(sum >> 32);

        if ((lane == warpSz - 1) || (base + lane == N - 1)) {
            r += c_out;
        } else {
            r = c_out;
        }

        limb = static_cast<int64_t>(lo);
        extResult[base + lane] = static_cast<uint64_t>(limb);
    }

    if (c_out != 0 && (base + lane < N - 1))
        changedMask |= maskBit;
}

template<class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
CarryPropagation_ABC(
    uint32_t *globalSync1, // [0] holds convergence counter
    uint32_t *globalSync2,
    uint32_t *globalSync3,
    uint64_t *sharedData,
    const int32_t    idx,                       // this thread’s global index
    const int32_t    numActualDigitsPlusGuard,  // N
    uint64_t *extResultTrue,         // Phase1_ABC “true” limbs
    uint64_t *extResultFalse,        // Phase1_ABC “false” limbs
    uint64_t *final128_DE,               // Phase1_DE limbs
    uint32_t *carry1,                    // length N+1
    uint32_t *carry2,                    // length N+1
    uint32_t *carry3,                    // length N+1
    uint32_t *carry4,                    // length N+1
    uint32_t *carry5,                    // length N+1
    uint32_t *carry6,                    // length N+1
    int32_t &carryAcc_ABC_True,         // out: final signed carry/borrow
    int32_t &carryAcc_ABC_False,        // out: final signed carry/borrow
    int32_t &carryAcc_DE,               // out: final unsigned carry
    cg::thread_block &block,
    cg::grid_group &grid
) {

//#define OVERRIDE_CARRY_IMPL

#ifdef OVERRIDE_CARRY_IMPL
    CarryPropagationSmall_ABC<SharkFloatParams>(globalSync1, // [0] holds convergence counter
                                                idx,        // this thread’s global index
                                                numActualDigitsPlusGuard, // N
                                                extResultTrue,            // Phase1_ABC “true” limbs
                                                extResultFalse,           // Phase1_ABC “false” limbs
                                                final128_DE,              // Phase1_DE limbs
                                                carry1,                   // length N+1
                                                carry2,                   // length N+1
                                                carry3,                   // length N+1
                                                carry4,                   // length N+1
                                                carry5,                   // length N+1
                                                carry6,                   // length N+1
                                                carryAcc_ABC_True,  // out: final signed carry/borrow
                                                carryAcc_ABC_False, // out: final signed carry/borrow
                                                carryAcc_DE,        // out: final unsigned carry
                                                block,
                                                grid);
    return;
#else
    if (grid.size() % 32 != 0) {
        CarryPropagationSmall_ABC<SharkFloatParams>(globalSync1,                   // [0] holds convergence counter
                                  idx,                      // this thread’s global index
                                  numActualDigitsPlusGuard, // N
                                  extResultTrue,                // Phase1_ABC “true” limbs
                                  extResultFalse,               // Phase1_ABC “false” limbs
                                  final128_DE,                  // Phase1_DE limbs
                                  carry1,                       // length N+1
                                  carry2,                       // length N+1
                                  carry3,                       // length N+1
                                  carry4,                       // length N+1
                                  carry5,                       // length N+1
                                  carry6,                       // length N+1
                                  carryAcc_ABC_True,             // out: final signed carry/borrow
                                  carryAcc_ABC_False,            // out: final signed carry/borrow
                                  carryAcc_DE,                   // out: final unsigned carry
                                  block,
                                  grid);
        return;
    }

    // --- geometry ---
    const int N = numActualDigitsPlusGuard;
    constexpr int warpSz = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & (warpSz - 1);
    const int totalThreads = gridDim.x * blockDim.x;
    const int totalWarps = max(1, totalThreads / warpSz);
    const int warpId = tid / warpSz;
    const unsigned fullMask = __activemask();
    const int numTiles = (N + warpSz - 1) / warpSz;

    // reinterpret carry buffers as int32
    uint32_t *cur1 = carry1;
    uint32_t *next1 = carry2;
    uint32_t *cur2 = carry3;
    uint32_t *next2 = carry4;
    uint32_t *cur3 = carry5;
    uint32_t *next3 = carry6;

    // init cur/next = 0 (length N+1 to include high slot)
    for (int i = tid; i <= N; i += totalThreads) {
        cur1[i] = cur2[i] = cur3[i] = 0;
        next1[i] = next2[i] = next3[i] = 0;
    }

    // warp-tile processor: all three streams in one 32-step loop
    struct Triple {
        int32_t o1, o2, o3;
        uint32_t changedMask;
    };

    auto warp_process_tile_32_all3 =
        [&](int tileIndex, uint32_t in1, uint32_t in2, uint32_t in3) -> Triple {
        const int base = tileIndex * warpSz;

        int64_t limb1 = 0, limb2 = 0, limb3 = 0;

        int32_t r1 = 0, r2 = 0, r3 = 0;
        uint32_t changedMask = 0u; // bit0=True, bit1=False, bit2=DE

        int32_t c_out1;
        int32_t c_out2;
        int32_t c_out3;

        limb1 = static_cast<int64_t>(extResultTrue[base + lane]);
        limb2 = static_cast<int64_t>(extResultFalse[base + lane]);
        limb3 = static_cast<int64_t>(final128_DE[base + lane]);

        for (int step = 0; step < warpSz; ++step) {
            int32_t inStep1 = __shfl_up_sync(fullMask, r1, 1);
            int32_t inStep2 = __shfl_up_sync(fullMask, r2, 1);
            int32_t inStep3 = __shfl_up_sync(fullMask, r3, 1);

            if (lane == 0) {
                if (step == 0) {
                    inStep1 = in1;
                    inStep2 = in2;
                    inStep3 = in3;
                } else {
                    inStep1 = 0;
                    inStep2 = 0;
                    inStep3 = 0;
                }
            }

            ProcessStreamABC<SharkFloatParams>(N, base, lane, limb1, inStep1, r1, c_out1, extResultTrue, 0x1u, changedMask);
            ProcessStreamABC<SharkFloatParams>(N, base, lane, limb2, inStep2, r2, c_out2, extResultFalse, 0x2u, changedMask);
            ProcessStreamABC<SharkFloatParams>(N, base, lane, limb3, inStep3, r3, c_out3, final128_DE, 0x4u, changedMask);
        }

        // OR reduce the mask within the warp
        changedMask |= __shfl_xor_sync(fullMask, changedMask, 16);
        changedMask |= __shfl_xor_sync(fullMask, changedMask, 8);
        changedMask |= __shfl_xor_sync(fullMask, changedMask, 4);
        changedMask |= __shfl_xor_sync(fullMask, changedMask, 2);
        changedMask |= __shfl_xor_sync(fullMask, changedMask, 1);

        return {r1, r2, r3, changedMask};
    };

    *globalSync1 = 0;
    *globalSync2 = std::numeric_limits<uint32_t>::max();
    *globalSync3 = 0;
    grid.sync();

    uint32_t prevResult1 = std::numeric_limits<uint32_t>::max();
    uint32_t prevResult2 = std::numeric_limits<uint32_t>::max();
    uint32_t loadedResult1 = std::numeric_limits<uint32_t>::max();
    uint32_t loadedResult2 = std::numeric_limits<uint32_t>::max();
    int32_t iteration = 0;
    bool assignedTermination = false;

    while (true) {
        // Each warp walks its tiles in round-robin: tile = warpId, warpId+totalWarps
        // Note: do not propagate from one tile to next in this loop!  That's another
        // global iteration.
        for (int tile = warpId; tile < numTiles; tile += totalWarps) {
            // Load incoming carry for the first digit in this tile
            const int base = tile * warpSz;
            const uint32_t in1 = cur1[base];
            const uint32_t in2 = cur2[base];
            const uint32_t in3 = cur3[base];

            cur1[base] = 0;
            cur2[base] = 0;
            cur3[base] = 0;

            const Triple tout = warp_process_tile_32_all3(tile, in1, in2, in3);

            // Outgoing carry index is the digit just after the tile (or N for the high slot)
            const int outIdx = min(base + warpSz, N);

            if (lane == warpSz - 1 || (base + lane == N - 1)) {
                if (outIdx < N) {
                    next1[outIdx] = static_cast<uint32_t>(tout.o1);
                    next2[outIdx] = static_cast<uint32_t>(tout.o2);
                    next3[outIdx] = static_cast<uint32_t>(tout.o3);
                } else { // outIdx == N
                    next1[N] = static_cast<uint32_t>(cur1[N] + tout.o1);
                    next2[N] = static_cast<uint32_t>(cur2[N] + tout.o2);
                    next3[N] = static_cast<uint32_t>(cur3[N] + tout.o3);
                }

                if (tout.changedMask) {
                    atomicAdd(globalSync1, 1);
                }
            }
        }

        grid.sync(); // all next*[N] writes are visible

        prevResult1 = prevResult2;
        prevResult2 = loadedResult1;
        loadedResult1 = *globalSync1;
        loadedResult2 = *globalSync2;

        if (grid.thread_rank() == 0) {
            cur1[N] = next1[N];
            cur2[N] = next2[N];
            cur3[N] = next3[N];

            if (loadedResult1 == prevResult2 && !assignedTermination) {
                *globalSync2 = iteration + 1;
                assignedTermination = true;
            }
        }

        if (iteration == loadedResult2) {
            break;
        }

        // Swap only the active streams (mirror of your original logic)
        auto swap2 = [](uint32_t *&a, uint32_t *&b) {
            auto *t = a;
            a = b;
            b = t;
        };

        swap2(cur1, next1);
        swap2(cur2, next2);
        swap2(cur3, next3);
        
        iteration++;
    }

    // Final carries are in cur*[N] (unchanged for streams that didn't need swapping)
    grid.sync();
    carryAcc_ABC_True = cur1[N];
    carryAcc_ABC_False = cur2[N];
    carryAcc_DE = cur3[N];
#endif
}
