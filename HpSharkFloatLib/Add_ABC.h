#include "ThreeWayMagnitude.h"

static __device__ SharkForceInlineReleaseOnly bool
cmp_magnitude_warp(const uint32_t *SharkRestrict e1,
                   int32_t s1,
                   int32_t exp1,
                   const uint32_t *SharkRestrict e2,
                   int32_t s2,
                   int32_t exp2,
                   const int32_t actualDigits,
                   const int32_t numActualDigitsPlusGuard)
{
    const int lane = threadIdx.x & 31;
    const unsigned fullMask = 0xffffffff;

    // 1) Exponent compare: same as scalar version
    if (exp1 != exp2) {
        return exp1 > exp2;
    }

    // 2) Exponents equal → compare digits high→low in warp tiles of 32
    const int32_t highestIndex = numActualDigitsPlusGuard - 1; // top digit index
    const int32_t totalDigits = numActualDigitsPlusGuard;      // indices 0..highestIndex
    const int32_t numTiles = (totalDigits + 31) / 32;

    bool decidedGreater = false; // true => e1 > e2
    bool decided = false;        // true once we find a differing digit

    for (int32_t tile = 0; tile < numTiles && !decided; ++tile) {
        const int32_t idxFromTop = tile * 32 + lane;
        const int32_t i = highestIndex - idxFromTop; // walk high→low

        uint32_t d1 = 0u, d2 = 0u;
        if (i >= 0) {
            d1 = GetNormalizedDigit(e1, actualDigits, numActualDigitsPlusGuard, s1, i);
            d2 = GetNormalizedDigit(e2, actualDigits, numActualDigitsPlusGuard, s2, i);
        }

        // Per-lane comparisons
        const bool gt = (d1 > d2);
        const bool lt = (d1 < d2);
        const bool ne = gt || lt;

        // Warp-wide: where do they differ?
        const unsigned maskDiff = __ballot_sync(fullMask, ne);

        if (maskDiff != 0u) {
            // Find the *first* differing digit from the top.
            // Mapping: tile=0,lane=0 => highest index; so "topmost" is
            // the smallest lane index with ne==true -> LSB of maskDiff.
            const int firstLane = __ffs(maskDiff) - 1; // 0..31

            const unsigned bit = 1u << firstLane;

            // Which side is larger at that digit?
            const unsigned maskGT = __ballot_sync(fullMask, gt);
            // maskGT and (maskDiff ^ maskGT) are disjoint; only one has 'bit' set.
            decidedGreater = (maskGT & bit) != 0u;
            decided = true;
        }
    }

    // If never decided, all digits equal => "not greater"
    return decided && decidedGreater;
}

static __device__ SharkForceInlineReleaseOnly ThreeWayLargestOrdering
CompareMagnitudes3Way(const int32_t effExpA,
                      const int32_t effExpB,
                      const int32_t effExpC,
                      const int32_t actualDigits,
                      const int32_t numActualDigitsPlusGuard,
                      const int32_t shiftA,
                      const int32_t shiftB,
                      const int32_t shiftC,
                      const uint32_t *SharkRestrict extA,
                      const uint32_t *SharkRestrict extB,
                      const uint32_t *SharkRestrict extC)
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    static constexpr bool OriginalImpl = true;
#else
    static constexpr bool OriginalImpl = false;
#endif

    if constexpr (OriginalImpl) {
        // Helper: returns true if "first" is strictly bigger than "second"
        auto cmp = [&](const uint32_t *SharkRestrict e1,
                       int32_t s1,
                       int32_t exp1,
                       const uint32_t *SharkRestrict e2,
                       int32_t s2,
                       int32_t exp2) {
            if (exp1 != exp2)
                return exp1 > exp2;
            // exponents equal -> compare normalized digits high->low
            for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
                uint32_t d1 = GetNormalizedDigit(e1, actualDigits, numActualDigitsPlusGuard, s1, i);
                uint32_t d2 = GetNormalizedDigit(e2, actualDigits, numActualDigitsPlusGuard, s2, i);
                if (d1 != d2)
                    return d1 > d2;
            }
            return false; // treat exact equality as "not greater"
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
    } else {
        // 1) Is A the strict max?
        const bool A_gt_B = cmp_magnitude_warp(
            extA, shiftA, effExpA, extB, shiftB, effExpB, actualDigits, numActualDigitsPlusGuard);

        const bool A_gt_C = cmp_magnitude_warp(
            extA, shiftA, effExpA, extC, shiftC, effExpC, actualDigits, numActualDigitsPlusGuard);

        if (A_gt_B && A_gt_C) {
            return ThreeWayLargestOrdering::A_GT_AllOthers;
        }

        // 2) Is B the strict max?
        const bool B_gt_A = cmp_magnitude_warp(
            extB, shiftB, effExpB, extA, shiftA, effExpA, actualDigits, numActualDigitsPlusGuard);

        const bool B_gt_C = cmp_magnitude_warp(
            extB, shiftB, effExpB, extC, shiftC, effExpC, actualDigits, numActualDigitsPlusGuard);

        if (B_gt_A && B_gt_C) {
            return ThreeWayLargestOrdering::B_GT_AllOthers;
        }

        // 3) Otherwise C is the (strict) max (or tied)
        return ThreeWayLargestOrdering::C_GT_AllOthers;
    }
}

// 5) helper to do |±X ±Y ±Z| in one pass, given a fixed X_gtY
static uint64_t __device__ SharkForceInlineReleaseOnly
CoreThreeWayAdd(uint64_t X, bool sX, uint64_t Y, bool sY, uint64_t Z, bool sZ, bool X_gtY, bool &outSign)
{
    // (X vs Y)
    uint64_t magXY;
    bool sXY;
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

template <class SharkFloatParams, int32_t CallIndex>
static __device__ SharkForceInlineReleaseOnly void
Phase1_ABC(cg::thread_block &block,
           cg::grid_group &grid,
           const int32_t idx,
           const ThreeWayLargestOrdering ordering,
           const bool IsNegativeA,
           const bool IsNegativeB,
           const bool IsNegativeC,
           const int32_t numActualDigitsPlusGuard,
           const int32_t actualDigits,
           const uint32_t *SharkRestrict extA,
           const uint32_t *SharkRestrict extB,
           const uint32_t *SharkRestrict extC,
           const int32_t shiftA,
           const int32_t shiftB,
           const int32_t shiftC,
           const int32_t effExpA,
           const int32_t effExpB,
           const int32_t effExpC,
           const int32_t bias,

           bool &outSignTrue,                      // sign when X_gtY == true
           bool &outSignFalse,                     // sign when X_gtY == false
           int32_t &outExpTrue_Orig,               // exponent before bias for true-branch
           int32_t &outExpFalse_Orig,              // exponent before bias for false-branch
           uint64_t *SharkRestrict extResultTrue,  // result limbs for X_gtY == true
           uint64_t *SharkRestrict extResultFalse, // result limbs for X_gtY == false

           DebugState<SharkFloatParams> *SharkRestrict debugStates)
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
            for (;;)
                ;
    }

    // 3) single diff per input to align to baseExp
    int32_t diffA = baseExp - effExpA;
    int32_t diffB = baseExp - effExpB;
    int32_t diffC = baseExp - effExpC;

    // 4) pick pointers, signs, shifts and diffs in “X, Y, Z” order
    const uint32_t *SharkRestrict extX;
    const uint32_t *SharkRestrict extY;
    const uint32_t *SharkRestrict extZ;
    bool sX, sY, sZ;
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
            for (;;)
                ;
    }

    // 6) single pass: two calls per digit, grid-stride version
    {
        int32_t stride = grid.size();
        for (int32_t i = idx; i < numActualDigitsPlusGuard; i += stride) {
            uint64_t Xi = GetNormalizedDigit(extX, actualDigits, numActualDigitsPlusGuard, shX, i);
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
CarryPropagationSmall_ABC(uint32_t *SharkRestrict globalSync1,    // [0] holds convergence counter
                          const int32_t idx,                      // this thread’s global index
                          const int32_t numActualDigitsPlusGuard, // N
                          uint64_t *SharkRestrict extResultTrue,  // Phase1_ABC “true” limbs
                          uint64_t *SharkRestrict extResultFalse, // Phase1_ABC “false” limbs
                          uint64_t *SharkRestrict final128_DE,    // Phase1_DE limbs
                          uint32_t *SharkRestrict carry1,         // length N+1
                          uint32_t *SharkRestrict carry2,         // length N+1
                          uint32_t *SharkRestrict carry3,         // length N+1
                          uint32_t *SharkRestrict carry4,         // length N+1
                          uint32_t *SharkRestrict carry5,         // length N+1
                          uint32_t *SharkRestrict carry6,         // length N+1
                          int32_t &carryAcc_ABC_True,             // out: final signed carry/borrow
                          int32_t &carryAcc_ABC_False,            // out: final signed carry/borrow
                          int32_t &carryAcc_DE,                   // out: final unsigned carry
                          cg::thread_block &block,
                          cg::grid_group &grid,
                          DebugGlobalCount<SharkFloatParams> *SharkRestrict debugGlobalState)
{
    constexpr uint64_t SHIFT1 = 21;
    constexpr uint64_t SHIFT2 = 42;
    constexpr uint64_t FIELD_MASK = (1ULL << 21) - 1;

    const int32_t N = numActualDigitsPlusGuard;
    const int32_t stride = grid.size();
    auto *SharkRestrict global64 = reinterpret_cast<uint64_t *>(globalSync1);

    // six carry buffers
    uint32_t *SharkRestrict cur1 = carry1;
    uint32_t *SharkRestrict next1 = carry2;
    uint32_t *SharkRestrict cur2 = carry3;
    uint32_t *SharkRestrict next2 = carry4;
    uint32_t *SharkRestrict cur3 = carry5;
    uint32_t *SharkRestrict next3 = carry6;

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

        if constexpr (HpShark::DebugGlobalState) {
            DebugCarryIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
        }
    }

    // final carry extraction
    carryAcc_ABC_True = static_cast<int32_t>(cur1[N]);
    carryAcc_ABC_False = static_cast<int32_t>(cur2[N]);
    carryAcc_DE = static_cast<int32_t>(cur3[N]);
    grid.sync();
}

// warp-tile processor: all three streams in one 32-step loop
struct WarpProcessTriple {
    int32_t o1, o2, o3;
    uint32_t changedMask;
};

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly WarpProcessTriple
WarpProcessTileCarry(cg::thread_block &block,
                     const int32_t iteration,
                     unsigned fullMask,
                     const int32_t numActualDigitsPlusGuard,
                     const int lane,
                     const int tileIndex,
                     const uint32_t in1,
                     const uint32_t in2,
                     const uint32_t in3,
                     int64_t &limb1,
                     int64_t &limb2,
                     int64_t &limb3)
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    // untested
    const auto warpSz = block.dim_threads().x;
#else
    constexpr auto warpSz = 32;
#endif

    const int base = tileIndex * warpSz;
    const auto basePlusLane = base + lane;

    int32_t r1 = 0, r2 = 0, r3 = 0; // 010101b : initial carry-in = 0 for all three streams
    uint32_t changedMask = 0u;      // bit0=True, bit1=False, bit2=DE

    const bool isLastLaneInTile = (lane == warpSz - 1);
    const bool isLastDigit = (basePlusLane == numActualDigitsPlusGuard - 1);
    const bool laneIsZero = (lane == 0);

#pragma unroll
    for (int step = 0; step < warpSz; ++step) {
        int32_t inStep1 = __shfl_up_sync(fullMask, r1, 1);
        int32_t inStep2 = __shfl_up_sync(fullMask, r2, 1);
        int32_t inStep3 = __shfl_up_sync(fullMask, r3, 1);

        const bool stepIsZero = (step == 0);

        if (laneIsZero) {
            // Lane 0: always inject at step 0, zero otherwise
            inStep1 = stepIsZero ? static_cast<int32_t>(in1) : 0;
            inStep2 = stepIsZero ? static_cast<int32_t>(in2) : 0;
            inStep3 = stepIsZero ? static_cast<int32_t>(in3) : 0;
        } else {
            // Other lanes: only special for iteration==0, step==0
            if (iteration == 0 && stepIsZero) {
                inStep1 = static_cast<int32_t>(in1);
                inStep2 = static_cast<int32_t>(in2);
                inStep3 = static_cast<int32_t>(in3);
            }
            // else: keep shuffled values
        }

        int32_t c_out1 = 0;
        int32_t c_out2 = 0;
        int32_t c_out3 = 0;

        const int64_t sum1 = static_cast<int64_t>(limb1) + inStep1;
        const uint32_t lo1 = static_cast<uint32_t>(sum1);
        c_out1 = static_cast<int32_t>(sum1 >> 32);

        const int64_t sum2 = static_cast<int64_t>(limb2) + inStep2;
        const uint32_t lo2 = static_cast<uint32_t>(sum2);
        c_out2 = static_cast<int32_t>(sum2 >> 32);

        const int64_t sum3 = static_cast<int64_t>(limb3) + inStep3;
        const uint32_t lo3 = static_cast<uint32_t>(sum3);
        c_out3 = static_cast<int32_t>(sum3 >> 32);

        if (!(isLastLaneInTile || isLastDigit)) {
            r1 = 0;
            r2 = 0;
            r3 = 0;
        }

        // Add c_outs to the packed value
        // Note that we don't explicitly saturate these counters, by
        // construction they should be capped in the range of [-1, 1]

        r1 += c_out1;
        r2 += c_out2;
        r3 += c_out3;

        limb1 = static_cast<int64_t>(lo1);
        limb2 = static_cast<int64_t>(lo2);
        limb3 = static_cast<int64_t>(lo3);

        if (basePlusLane < numActualDigitsPlusGuard - 1) {
            changedMask |= c_out1 | c_out2 | c_out3;
        }
    }

    return {r1, r2, r3, changedMask};
}

struct PPTransfer3 {
    // We use 64 bits now; only the low 18 bits are needed for the S→S mapping
    // (3 streams × 3 input states × 2 bits).
    uint64_t bits;
};

__device__ SharkForceInlineReleaseOnly static uint32_t
PPencode_carry(int32_t c) // c ∈ {-1,0,1} -> {0,1,2}
{
    return static_cast<uint32_t>(c + 1);
}

__device__ SharkForceInlineReleaseOnly static int32_t
PPdecode_carry(uint32_t enc) // enc ∈ {0,1,2} -> {-1,0,1}
{
    return static_cast<int32_t>(enc) - 1;
}

__device__ SharkForceInlineReleaseOnly static int
PPstate_index_for_carry(int32_t c)
{
    // c ∈ {-1,0,1} -> 0,1,2
    return c + 1;
}

__device__ SharkForceInlineReleaseOnly static uint32_t
PPget_field(uint64_t bits, int streamIdx, int32_t c_in)
{
    const int stateIdx = PPstate_index_for_carry(c_in); // 0..2
    const int fieldIdx = streamIdx * 3 + stateIdx;      // 0..8
    const uint32_t shift = static_cast<uint32_t>(fieldIdx * 2);
    return static_cast<uint32_t>((bits >> shift) & 0x3ull);
}

__device__ SharkForceInlineReleaseOnly static void
PPset_field(uint64_t &bits, int streamIdx, int32_t c_in, int32_t c_out)
{
    const int stateIdx = PPstate_index_for_carry(c_in); // 0..2
    const int fieldIdx = streamIdx * 3 + stateIdx;      // 0..8
    const uint32_t shift = static_cast<uint32_t>(fieldIdx * 2);
    const uint64_t mask = 0x3ull << shift;
    const uint64_t enc = static_cast<uint64_t>(PPencode_carry(c_out) & 0x3u);
    bits = (bits & ~mask) | (enc << shift);
}

// Evaluate transfer on a carry in {-1,0,1} for a given stream
__device__ SharkForceInlineReleaseOnly static int32_t
PPeval_transfer(const PPTransfer3 &t, int streamIdx, int32_t c_in)
{
    const uint32_t enc = PPget_field(t.bits, streamIdx, c_in);
    return PPdecode_carry(enc);
}

// Composition: g = b ∘ a, i.e. apply a then b, for ALL streams in one go
// Here f_a and f_b are *block-level* transfer functions:
//    c_mid = a(stream, c_in)
//    c_out = b(stream, c_mid)
// r is their composition.
__device__ SharkForceInlineReleaseOnly static PPTransfer3
PPcompose_transfer(const PPTransfer3 &a, const PPTransfer3 &b)
{
    PPTransfer3 r;
    r.bits = 0ull;

#pragma unroll
    for (int streamIdx = 0; streamIdx < 3; ++streamIdx) {
#pragma unroll
        for (int c_in = -1; c_in <= 1; ++c_in) {
            const int32_t cmid = PPeval_transfer(a, streamIdx, c_in);
            const int32_t c_out = PPeval_transfer(b, streamIdx, cmid);
            PPset_field(r.bits, streamIdx, c_in, c_out);
        }
    }

    return r;
}

constexpr int DIGITS_PER_BLOCK = 3;

// Build block-level transfer for a block of up to 3 digits:
// for each stream and c_in ∈ {-1,0,1}, simulate 3 digits in sequence.
__device__ SharkForceInlineReleaseOnly static PPTransfer3
PPbuild_block_transfer_3digits(const uint64_t *extResultTrue,
                               const uint64_t *extResultFalse,
                               const uint64_t *final128_DE,
                               const int32_t *cur1,
                               const int32_t *cur2,
                               const int32_t *cur3,
                               int baseDigit,
                               int len)
{
    PPTransfer3 t{};
    t.bits = 0ull;

#pragma unroll
    for (int streamIdx = 0; streamIdx < 3; ++streamIdx) {
#pragma unroll
        for (int c_in = -1; c_in <= 1; ++c_in) {
            int32_t c = c_in;

#pragma unroll
            for (int k = 0; k < DIGITS_PER_BLOCK; ++k) {
                if (k >= len)
                    break; // tail block safety
                const int i = baseDigit + k;

                const uint32_t d = (streamIdx == 0)   ? static_cast<uint32_t>(extResultTrue[i])
                                   : (streamIdx == 1) ? static_cast<uint32_t>(extResultFalse[i])
                                                      : static_cast<uint32_t>(final128_DE[i]);

                const int32_t cur = (streamIdx == 0) ? cur1[i] : (streamIdx == 1) ? cur2[i] : cur3[i];

                const int32_t in = c + cur;
                const int64_t sum = static_cast<int64_t>(d) + static_cast<int64_t>(in);
                c = static_cast<int32_t>(sum >> 32); // in [-1,1]
            }

            PPset_field(t.bits, streamIdx, c_in, c);
        }
    }

    return t;
}

// constexpr helper that mirrors PPset_field, but usable at compile time.
// NOTE: no __device__ here; this is purely a host-side constexpr function.
constexpr uint64_t
PPset_field_constexpr(uint64_t bits, int streamIdx, int c_in, int c_out)
{
    // c_in ∈ {-1,0,1} -> stateIdx ∈ {0,1,2}
    const int stateIdx = c_in + 1;
    const int fieldIdx = streamIdx * 3 + stateIdx; // 0..8
    const uint64_t shift = static_cast<uint64_t>(fieldIdx) * 2ull;
    const uint64_t mask = 0x3ull << shift;
    const uint64_t enc = static_cast<uint64_t>((c_out + 1) & 0x3); // encode {-1,0,1}→{0,1,2}
    return (bits & ~mask) | (enc << shift);
}

// C++17/20 constexpr builder: runs entirely at compile time.
constexpr PPTransfer3
make_PPIdentity()
{
    uint64_t bits = 0ull;

    // C++17 constexpr allows loops; everything here is integral / literal.
    for (int s = 0; s < 3; ++s) {
        for (int c = -1; c <= 1; ++c) {
            bits = PPset_field_constexpr(bits, s, c, c); // identity: f(c) = c
        }
    }

    return PPTransfer3{bits};
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
CarryPropagation_ABC_TiledV2(uint32_t *globalSync1, // [0] holds convergence counter
                             uint32_t *globalSync2,
                             uint64_t *SharkRestrict shared_data,
                             const int32_t idx,                      // this thread’s global index
                             const int32_t numActualDigitsPlusGuard, // N
                             uint64_t *SharkRestrict extResultTrue,  // Phase1_ABC “true” limbs
                             uint64_t *SharkRestrict extResultFalse, // Phase1_ABC “false” limbs
                             uint64_t *SharkRestrict final128_DE,    // Phase1_DE limbs
                             uint32_t *SharkRestrict cur1,           // length N+1
                             uint32_t *SharkRestrict next1,          // length N+1
                             uint32_t *SharkRestrict cur2,           // length N+1
                             uint32_t *SharkRestrict next2,          // length N+1
                             uint32_t *SharkRestrict cur3,           // length N+1
                             uint32_t *SharkRestrict next3,          // length N+1
                             int32_t &carryAcc_ABC_True,             // out: final signed carry/borrow
                             int32_t &carryAcc_ABC_False,            // out: final signed carry/borrow
                             int32_t &carryAcc_DE,                   // out: final unsigned carry
                             cg::thread_block &block,
                             cg::grid_group &grid,
                             DebugGlobalCount<SharkFloatParams> *SharkRestrict debugGlobalState)
{

    if (grid.size() % 32 != 0) {
        CarryPropagationSmall_ABC<SharkFloatParams>(globalSync1, // [0] holds convergence counter
                                                    idx,         // this thread’s global index
                                                    numActualDigitsPlusGuard, // N
                                                    extResultTrue,            // Phase1_ABC “true” limbs
                                                    extResultFalse,           // Phase1_ABC “false” limbs
                                                    final128_DE,              // Phase1_DE limbs
                                                    cur1,                     // length N+1
                                                    next1,                    // length N+1
                                                    cur2,                     // length N+1
                                                    next2,                    // length N+1
                                                    cur3,                     // length N+1
                                                    next3,                    // length N+1
                                                    carryAcc_ABC_True,  // out: final signed carry/borrow
                                                    carryAcc_ABC_False, // out: final signed carry/borrow
                                                    carryAcc_DE,        // out: final unsigned carry
                                                    block,
                                                    grid,
                                                    debugGlobalState);
        return;
    }

    // --- geometry ---
#ifdef TEST_SMALL_NORMALIZE_WARP
    // untested
    constexpr auto warpSz = block.dim_threads().x;
#else
    constexpr auto warpSz = 32;
#endif

    const unsigned fullMask = __activemask();
    const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int totalThreads = gridDim.x * blockDim.x;
    const int totalWarps = max(1, totalThreads / warpSz);

    // init cur/next = 0 (length numActualDigitsPlusGuard+1 to include high slot)
    for (int i = tid; i <= numActualDigitsPlusGuard; i += totalThreads) {
        cur1[i] = cur2[i] = cur3[i] = 0;
        next1[i] = next2[i] = next3[i] = 0;
    }

    // The way we initialize all this is very important to ensure the loop converges
    *globalSync1 = 0;
    *globalSync2 = std::numeric_limits<uint32_t>::max() - 1;

    grid.sync();

    // grid‐stride per‐digit work
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        // ABC_True
        auto AddPhase = [](uint64_t *extResult, uint32_t *next, int32_t i) {
            const uint64_t limb = extResult[i];
            const uint32_t lo1 = static_cast<uint32_t>(limb & 0xFFFFFFFFULL);
            const uint32_t hi1 = static_cast<uint32_t>(limb >> 32);
            extResult[i] = lo1;
            next[i + 1] = hi1;
        };

        AddPhase(extResultTrue, next1, i);
        AddPhase(extResultFalse, next2, i);
        AddPhase(final128_DE, next3, i);
    }

    grid.sync();

    // grid‐stride per‐digit work
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        // ABC_True
        auto AddPhase = [](int32_t numActualDigitsPlusGuard,
                           uint64_t *extResult,
                           uint32_t *next,
                           uint32_t *cur,
                           int32_t i) {
            const int64_t limb = static_cast<int64_t>(extResult[i]) + static_cast<int32_t>(next[i]);
            next[i] = 0;

            const uint32_t lo1 = static_cast<uint32_t>(limb & 0xFFFFFFFFULL);
            const uint32_t hi1 = static_cast<uint32_t>(limb >> 32);
            extResult[i] = lo1;

            if (i == numActualDigitsPlusGuard - 1) {
                cur[i + 1] = hi1 + next[i + 1];
            } else {
                cur[i + 1] = hi1;
            }
        };

        AddPhase(numActualDigitsPlusGuard, extResultTrue, next1, cur1, i);
        AddPhase(numActualDigitsPlusGuard, extResultFalse, next2, cur2, i);
        AddPhase(numActualDigitsPlusGuard, final128_DE, next3, cur3, i);
    }

    grid.sync();

    const int warpId = tid / warpSz;
    const int lane = threadIdx.x & (warpSz - 1);
    const int numTiles = (numActualDigitsPlusGuard + warpSz - 1) / warpSz;

    uint32_t prevResult2 = std::numeric_limits<uint32_t>::max() - 1;
    uint32_t loadedResult1 = std::numeric_limits<uint32_t>::max() - 2;
    uint32_t loadedResult2 = std::numeric_limits<uint32_t>::max() - 1;
    int32_t iteration = 0;
    bool assignedTermination = false;

    while (true) {
        prevResult2 = loadedResult1;
        loadedResult1 = *globalSync1;
        loadedResult2 = *globalSync2;

        // Each warp walks its tiles in round-robin: tile = warpId, warpId+totalWarps
        // Note: do not propagate from one tile to next in this loop!  That's another
        // global iteration.
        for (int tile = warpId; tile < numTiles; tile += totalWarps) {
            // Load incoming carry for the first digit in this tile
            const int base = tile * warpSz;
            const auto basePlusLane = base + lane;
            const auto in1 = static_cast<int32_t>(cur1[basePlusLane]);
            const auto in2 = static_cast<int32_t>(cur2[basePlusLane]);
            const auto in3 = static_cast<int32_t>(cur3[basePlusLane]);
            auto limb1 = static_cast<int64_t>(extResultTrue[basePlusLane]);
            auto limb2 = static_cast<int64_t>(extResultFalse[basePlusLane]);
            auto limb3 = static_cast<int64_t>(final128_DE[basePlusLane]);

            if (basePlusLane != numActualDigitsPlusGuard) { // ...
                cur1[basePlusLane] = 0;
                cur2[basePlusLane] = 0;
                cur3[basePlusLane] = 0;
            }

            const WarpProcessTriple tout =
                WarpProcessTileCarry<SharkFloatParams>(block,
                                                       iteration,
                                                       fullMask,
                                                       numActualDigitsPlusGuard,
                                                       lane,
                                                       tile,
                                                       in1,
                                                       in2,
                                                       in3,
                                                       limb1,
                                                       limb2,
                                                       limb3);

            extResultTrue[basePlusLane] = static_cast<uint64_t>(limb1);
            extResultFalse[basePlusLane] = static_cast<uint64_t>(limb2);
            final128_DE[basePlusLane] = static_cast<uint64_t>(limb3);

            // Outgoing carry index is the digit just after the tile (or numActualDigitsPlusGuard for the
            // high slot)
            const int outIdx = min(base + warpSz, numActualDigitsPlusGuard);

            if (lane == warpSz - 1 || (base + lane == numActualDigitsPlusGuard - 1)) {
                const auto o1 = tout.o1;
                const auto o2 = tout.o2;
                const auto o3 = tout.o3;

                if (outIdx < numActualDigitsPlusGuard) {
                    next1[outIdx] = static_cast<uint32_t>(o1);
                    next2[outIdx] = static_cast<uint32_t>(o2);
                    next3[outIdx] = static_cast<uint32_t>(o3);
                } else { // outIdx == numActualDigitsPlusGuard
                    next1[numActualDigitsPlusGuard] =
                        static_cast<uint32_t>(cur1[numActualDigitsPlusGuard] + o1);
                    next2[numActualDigitsPlusGuard] =
                        static_cast<uint32_t>(cur2[numActualDigitsPlusGuard] + o2);
                    next3[numActualDigitsPlusGuard] =
                        static_cast<uint32_t>(cur3[numActualDigitsPlusGuard] + o3);
                }

                if (tout.changedMask) {
                    atomicAdd(globalSync1, 1);
                }
            }
        }

        if (iteration - 1 == loadedResult2) {
            break;
        }

        // Tell everyone when to exit
        if (tid == 0) {
            if (loadedResult1 == prevResult2 && !assignedTermination) {
                *globalSync2 = iteration;
                assignedTermination = true;
            }
        }

        // all next*[numActualDigitsPlusGuard] writes are visible
        grid.sync();

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

        if constexpr (HpShark::DebugGlobalState) {
            DebugCarryIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
        }
    }

    // Note: no grid.sync() here
    carryAcc_ABC_True = next1[numActualDigitsPlusGuard];
    carryAcc_ABC_False = next2[numActualDigitsPlusGuard];
    carryAcc_DE = next3[numActualDigitsPlusGuard];
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
CarryPropagation_ABC_PPv3(uint32_t *globalSync1, // [0] holds convergence counter
                          uint32_t *globalSync2,
                          uint64_t *SharkRestrict shared_data,
                          const int32_t idx,                      // this thread’s global index
                          const int32_t numActualDigitsPlusGuard, // N
                          uint64_t *SharkRestrict extResultTrue,  // Phase1_ABC “true” limbs
                          uint64_t *SharkRestrict extResultFalse, // Phase1_ABC “false” limbs
                          uint64_t *SharkRestrict final128_DE,    // Phase1_DE limbs
                          uint32_t *SharkRestrict cur1,           // length N+1
                          uint32_t *SharkRestrict next1,          // length N+1
                          uint32_t *SharkRestrict cur2,           // length N+1
                          uint32_t *SharkRestrict next2,          // length N+1
                          uint32_t *SharkRestrict cur3,           // length N+1
                          uint32_t *SharkRestrict next3,          // length N+1
                          int32_t &carryAcc_ABC_True,             // out: final signed carry/borrow
                          int32_t &carryAcc_ABC_False,            // out: final signed carry/borrow
                          int32_t &carryAcc_DE,                   // out: final unsigned carry
                          cg::thread_block &block,
                          cg::grid_group &grid,
                          DebugGlobalCount<SharkFloatParams> *SharkRestrict debugGlobalState)
{

#ifdef TEST_SMALL_NORMALIZE_WARP
    if (grid.size() % 32 != 0) {
        CarryPropagationSmall_ABC<SharkFloatParams>(globalSync1, // [0] holds convergence counter
                                                    idx,         // this thread’s global index
                                                    numActualDigitsPlusGuard, // N
                                                    extResultTrue,            // Phase1_ABC “true” limbs
                                                    extResultFalse,           // Phase1_ABC “false” limbs
                                                    final128_DE,              // Phase1_DE limbs
                                                    cur1,                     // length N+1
                                                    next1,                    // length N+1
                                                    cur2,                     // length N+1
                                                    next2,                    // length N+1
                                                    cur3,                     // length N+1
                                                    next3,                    // length N+1
                                                    carryAcc_ABC_True,  // out: final signed carry/borrow
                                                    carryAcc_ABC_False, // out: final signed carry/borrow
                                                    carryAcc_DE,        // out: final unsigned carry
                                                    block,
                                                    grid,
                                                    debugGlobalState);
        return;
    }
#endif

    // --- geometry ---
#ifdef TEST_SMALL_NORMALIZE_WARP
    // untested
    constexpr auto warpSz = block.dim_threads().x;
#else
    constexpr auto warpSz = 32;
#endif

    const unsigned fullMask = __activemask();
    const int32_t tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int totalThreads = gridDim.x * blockDim.x;
    const int totalWarps = max(1, totalThreads / warpSz);

    // init cur/next = 0 (length numActualDigitsPlusGuard+1 to include high slot)
    for (int i = tid; i <= numActualDigitsPlusGuard; i += totalThreads) {
        cur1[i] = cur2[i] = cur3[i] = 0;
        next1[i] = next2[i] = next3[i] = 0;
    }

    // The way we initialize all this is very important to ensure the loop converges
    *globalSync1 = 0;
    *globalSync2 = std::numeric_limits<uint32_t>::max() - 1;

    grid.sync();

    // grid‐stride per‐digit work
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        // ABC_True
        auto AddPhase = [](uint64_t *extResult, uint32_t *next, int32_t i) {
            const uint64_t limb = extResult[i];
            const uint32_t lo1 = static_cast<uint32_t>(limb & 0xFFFFFFFFULL);
            const uint32_t hi1 = static_cast<uint32_t>(limb >> 32);
            extResult[i] = lo1;
            next[i + 1] = hi1;
        };

        AddPhase(extResultTrue, next1, i);
        AddPhase(extResultFalse, next2, i);
        AddPhase(final128_DE, next3, i);
    }

    grid.sync();

    // grid‐stride per‐digit work
    for (int32_t i = tid; i < numActualDigitsPlusGuard; i += totalThreads) {
        // ABC_True
        auto AddPhase = [](int32_t numActualDigitsPlusGuard,
                           uint64_t *extResult,
                           uint32_t *next,
                           uint32_t *cur,
                           int32_t i) {
            const int64_t limb = static_cast<int64_t>(extResult[i]) + static_cast<int32_t>(next[i]);
            next[i] = 0;

            const uint32_t lo1 = static_cast<uint32_t>(limb & 0xFFFFFFFFULL);
            const uint32_t hi1 = static_cast<uint32_t>(limb >> 32);
            extResult[i] = lo1;

            if (i == numActualDigitsPlusGuard - 1) {
                cur[i + 1] = hi1 + next[i + 1];
            } else {
                cur[i + 1] = hi1;
            }
        };

        AddPhase(numActualDigitsPlusGuard, extResultTrue, next1, cur1, i);
        AddPhase(numActualDigitsPlusGuard, extResultFalse, next2, cur2, i);
        AddPhase(numActualDigitsPlusGuard, final128_DE, next3, cur3, i);
    }

    grid.sync();

    // ----------------------------------------------------------------------
    // Parallel prefix-scan over per-block carry-transfer functions,
    // with ALL THREE streams packed into PPTransfer3, and each block
    // representing 3 digits per stream (9 scalar digits total).
    // ----------------------------------------------------------------------

    // Reinterpret scratch as a PPTransfer3 array (one per block of 3 digits)
    auto *SharkRestrict tfBlock = reinterpret_cast<PPTransfer3 *>(next1);

    const int32_t numDigits = numActualDigitsPlusGuard;
    const int32_t numBlocks = (numDigits + DIGITS_PER_BLOCK - 1) / DIGITS_PER_BLOCK;

    // ----------------------------
    // Step 1: build per-block transfer functions F_block
    // ----------------------------
    for (int32_t blockIdx = tid; blockIdx < numBlocks; blockIdx += totalThreads) {
        const int base = blockIdx * DIGITS_PER_BLOCK;
        const int remaining = numDigits - base;
        const int len = (remaining > DIGITS_PER_BLOCK) ? DIGITS_PER_BLOCK : remaining;
        if (len <= 0)
            continue;

        tfBlock[blockIdx] = PPbuild_block_transfer_3digits(extResultTrue,
                                                           extResultFalse,
                                                           final128_DE,
                                                           reinterpret_cast<const int32_t *>(cur1),
                                                           reinterpret_cast<const int32_t *>(cur2),
                                                           reinterpret_cast<const int32_t *>(cur3),
                                                           base,
                                                           len);
    }

    grid.sync();

    // ----------------------------
    // Step 2: hierarchical scan over tfBlock (block-level transfers)
    //         using shared_data for all shared memory.
    // ----------------------------

    {
        const int numCTAs = gridDim.x;
        const int cta = block.group_index().x;
        const int threadsInBlock = block.dim_threads().x;
        const int tidBlock = threadIdx.x;

        // Partition tfBlock into numCTAs contiguous chunks.
        const int chunkSize = (numBlocks + numCTAs - 1) / numCTAs; // ceil(numBlocks / numCTAs)

        const int start = cta * chunkSize;
        const int end = min(start + chunkSize, numBlocks);
        const int len = max(end - start, 0);

        // We'll use shared_data as a PPTransfer3 buffer:
        // [0 .. chunkSize-1] for per-CTA local scan
        // [chunkSize .. chunkSize+numCTAs-1] for CTA-level scan (used only in CTA 0)
        auto *shAll = reinterpret_cast<PPTransfer3 *>(shared_data);
        PPTransfer3 *sh_scan = shAll;            // size >= chunkSize
        PPTransfer3 *sh_cta = shAll + chunkSize; // size >= numCTAs

        // Global scratch for CTA totals (aggregates), one per CTA.
        // This is in global memory, NOT shared.
        PPTransfer3 *tfCTA = reinterpret_cast<PPTransfer3 *>(next2); // ensure next2 is sized for numCTAs

        // ----------------------------------------------------------
        // Step 2A: Each CTA scans its own chunk in shared memory
        // ----------------------------------------------------------

        if (len > 0) {
            // Load tfBlock[start..end) into shared memory
            for (int i = tidBlock; i < len; i += threadsInBlock) {
                sh_scan[i] = tfBlock[start + i];
            }
            block.sync();

            // In-CTA inclusive scan over sh_scan[0..len)
            for (int offset = 1; offset < len; offset <<= 1) {
                for (int i = tidBlock; i < len; i += threadsInBlock) {
                    if (i >= offset) {
                        sh_scan[i] = PPcompose_transfer(sh_scan[i - offset], sh_scan[i]);
                    }
                }
                block.sync();
            }

            // Write back local prefix-scan results
            for (int i = tidBlock; i < len; i += threadsInBlock) {
                tfBlock[start + i] = sh_scan[i];
            }

            // The last element in this chunk is the CTA's total transfer
            if (tidBlock == 0) {
                tfCTA[cta] = sh_scan[len - 1];
            }
        } else {
            // No blocks belong to this CTA: its total is identity
            if (tidBlock == 0) {
                tfCTA[cta] = make_PPIdentity();
            }
        }

        // Wait for all CTAs to write tfCTA[cta]
        grid.sync(); // 1st global barrier

        // ----------------------------------------------------------
        // Step 2B: CTA 0 scans the CTA-level aggregates in shared mem
        // ----------------------------------------------------------

        if (cta == 0) {
            // Load tfCTA[0..numCTAs) into shared memory
            for (int i = tidBlock; i < numCTAs; i += threadsInBlock) {
                sh_cta[i] = tfCTA[i];
            }
            block.sync();

            // Inclusive scan over CTA totals
            for (int offset = 1; offset < numCTAs; offset <<= 1) {
                for (int i = tidBlock; i < numCTAs; i += threadsInBlock) {
                    if (i >= offset) {
                        sh_cta[i] = PPcompose_transfer(sh_cta[i - offset], sh_cta[i]);
                    }
                }
                block.sync();
            }

            // Write back the CTA prefix results
            for (int i = tidBlock; i < numCTAs; i += threadsInBlock) {
                tfCTA[i] = sh_cta[i];
            }
        }

        grid.sync(); // 2nd global barrier: tfCTA now holds CTA prefix transfers

        // ----------------------------------------------------------
        // Step 2C: Apply CTA-level prefix to each CTA's local scan
        // ----------------------------------------------------------

        if (len > 0) {
            // Prefix contribution from all previous CTAs.
            // For cta == 0, there is no previous CTA; prefix is identity.
            PPTransfer3 prefix = make_PPIdentity();
            if (cta > 0) {
                prefix = tfCTA[cta - 1]; // prefix over chunks [0 .. cta-1]
            }

            // Compose prefix ∘ local-scan(tfBlock[start..end))
            for (int i = tidBlock; i < len; i += threadsInBlock) {
                const int globalIdx = start + i;
                tfBlock[globalIdx] = PPcompose_transfer(prefix, tfBlock[globalIdx]);
            }
        }

        grid.sync(); // 3rd global barrier: tfBlock[k] is now full prefix F_block_prefix[k]
    }

    // ----------------------------
    // Step 3: use block prefix functions to compute per-digit incoming carries
    //         and apply them to digits.
    // ----------------------------

    for (int32_t blockIdx = tid; blockIdx < numBlocks; blockIdx += totalThreads) {
        const int base = blockIdx * DIGITS_PER_BLOCK;
        const int remaining = numDigits - base;
        const int len = (remaining > DIGITS_PER_BLOCK) ? DIGITS_PER_BLOCK : remaining;
        if (len <= 0)
            continue;

        // Carry entering this block in each stream: F_block_prefix
        int32_t c1 = (blockIdx > 0) ? PPeval_transfer(tfBlock[blockIdx - 1], 0, 0) : 0;
        int32_t c2 = (blockIdx > 0) ? PPeval_transfer(tfBlock[blockIdx - 1], 1, 0) : 0;
        int32_t c3 = (blockIdx > 0) ? PPeval_transfer(tfBlock[blockIdx - 1], 2, 0) : 0;

#pragma unroll
        for (int k = 0; k < DIGITS_PER_BLOCK; ++k) {
            if (k >= len)
                break;
            const int i = base + k;

            const int32_t in1 = c1 + static_cast<int32_t>(cur1[i]);
            const int32_t in2 = c2 + static_cast<int32_t>(cur2[i]);
            const int32_t in3 = c3 + static_cast<int32_t>(cur3[i]);

            // Stream 0: extResultTrue
            {
                const int64_t sum1 = static_cast<int64_t>(extResultTrue[i]) + static_cast<int64_t>(in1);
                const uint32_t lo1 = static_cast<uint32_t>(sum1);
                extResultTrue[i] = static_cast<uint64_t>(lo1);
                c1 = static_cast<int32_t>(sum1 >> 32);
            }

            // Stream 1: extResultFalse
            {
                const int64_t sum2 = static_cast<int64_t>(extResultFalse[i]) + static_cast<int64_t>(in2);
                const uint32_t lo2 = static_cast<uint32_t>(sum2);
                extResultFalse[i] = static_cast<uint64_t>(lo2);
                c2 = static_cast<int32_t>(sum2 >> 32);
            }

            // Stream 2: final128_DE
            {
                const int64_t sum3 = static_cast<int64_t>(final128_DE[i]) + static_cast<int64_t>(in3);
                const uint32_t lo3 = static_cast<uint32_t>(sum3);
                final128_DE[i] = static_cast<uint64_t>(lo3);
                c3 = static_cast<int32_t>(sum3 >> 32);
            }
        }
    }

    grid.sync();

    // ----------------------------
    // Step 4: final high-slot carry
    // ----------------------------
    if (tid == 0) {
        const int32_t N = numDigits;

        int32_t c1_N = 0;
        int32_t c2_N = 0;
        int32_t c3_N = 0;

        if (numBlocks > 0) {
            const int lastBlock = numBlocks - 1;
            c1_N = PPeval_transfer(tfBlock[lastBlock], 0, 0);
            c2_N = PPeval_transfer(tfBlock[lastBlock], 1, 0);
            c3_N = PPeval_transfer(tfBlock[lastBlock], 2, 0);
        }

        const int32_t final1 = c1_N + static_cast<int32_t>(cur1[N]);
        const int32_t final2 = c2_N + static_cast<int32_t>(cur2[N]);
        const int32_t final3 = c3_N + static_cast<int32_t>(cur3[N]);

        cur1[N] = static_cast<uint32_t>(final1);
        cur2[N] = static_cast<uint32_t>(final2);
        cur3[N] = static_cast<uint32_t>(final3);

        if constexpr (HpShark::DebugGlobalState) {
            DebugCarryIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
        }
    }

    grid.sync();

    carryAcc_ABC_True = static_cast<int32_t>(cur1[numDigits]);
    carryAcc_ABC_False = static_cast<int32_t>(cur2[numDigits]);
    carryAcc_DE = static_cast<int32_t>(cur3[numDigits]);
}
