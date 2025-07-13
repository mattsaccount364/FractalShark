#include "ThreeWayMagnitude.h"

__device__ SharkForceInlineReleaseOnly ThreeWayLargestOrdering
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
uint64_t __device__ SharkForceInlineReleaseOnly CoreThreeWayAdd(
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
__device__ SharkForceInlineReleaseOnly
void Phase1_ABC (
    cg::thread_block &block,
    cg::grid_group &grid,
    const RecordIt record,
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

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2_Perm1, uint64_t>(
            record, debugStates, grid, block, extResultTrue, numActualDigitsPlusGuard);
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2_Perm2, uint64_t>(
            record, debugStates, grid, block, extResultFalse, numActualDigitsPlusGuard);
        grid.sync();
    }
}

//template<class SharkFloatParams>
//__device__
//void CarryPropagation_ABC (
//    uint32_t *sharedData,
//    uint32_t *globalSync,
//    const int32_t idx,
//    const int32_t numActualDigitsPlusGuard,
//    uint64_t *final128_ABC,         // raw signed limbs from Phase1_ABC
//    uint32_t *carry1,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
//    uint32_t *carry2,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
//    int32_t &carryAcc,
//    cg::thread_block &block,
//    cg::grid_group &grid
//)
//{
//    const int32_t N = numActualDigitsPlusGuard;
//    const int32_t stride = grid.size();
//
//    // use the user-passed buffers:
//    uint32_t *curC = carry1;
//    uint32_t *nextC = carry2;
//
//    // only one thread initializes the counter
//    if (block.group_index().x == 0 && block.thread_index().x == 0) {
//        globalSync[0] = 1;
//    }
//
//    // zero out both carry arrays
//    for (int32_t i = idx; i < N + 1; i += stride) {
//        curC[i] = 0u;
//        nextC[i] = 0u;
//    }
//    grid.sync();
//
//    constexpr int maxIter = 1000;
//    uint32_t prevCount = 0;
//
//    // iterative rippling until no new carries
//    for (int iter = 0; iter < maxIter; ++iter) {
//        if (globalSync[0] == prevCount) break;
//        prevCount = globalSync[0];
//
//        grid.sync();
//
//        uint32_t localNew = 0;
//        for (int32_t i = idx; i < N; i += stride) {
//            // incoming carry (borrow if negative)
//            int32_t inC = (i == 0 ? 0 : static_cast<int32_t>(curC[i]));
//            int64_t limb = static_cast<int64_t>(final128_ABC[i]);
//            int64_t sum = limb + inC;
//
//            // write back low 32 bits
//            uint32_t low32 = static_cast<uint32_t>(sum);
//            final128_ABC[i] = low32;
//
//            // arithmetic shift yields signed carry/borrow
//            int32_t newC = static_cast<int32_t>((sum - int64_t(low32)) >> 32);
//            if (i < N - 1) {
//                nextC[i + 1] = static_cast<uint32_t>(newC);
//            } else {
//                nextC[i + 1] = curC[i + 1] + static_cast<uint32_t>(newC);
//            }
//            localNew += (newC != 0);
//        }
//
//        grid.sync();
//
//        if (localNew) {
//            atomicAdd(&globalSync[0], localNew);
//        }
//
//        // swap for next iteration
//        auto *tmp = curC; curC = nextC; nextC = tmp;
//        grid.sync();
//    }
//
//    carryAcc = static_cast<int32_t>(curC[N]);
//    grid.sync();
//}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
CarryPropagation_ABC(
    uint32_t * /*sharedData*/,
    uint32_t *globalSync,                // [0] holds convergence counter
    const int32_t    idx,                       // this thread’s global index
    const int32_t    numActualDigitsPlusGuard,  // N
    uint64_t *final128_ABC_True,         // Phase1_ABC “true” limbs
    uint64_t *final128_ABC_False,        // Phase1_ABC “false” limbs
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
    const int32_t N = numActualDigitsPlusGuard;
    const int32_t stride = grid.size();

    // assign the six working buffers
    uint32_t *curC1 = carry1, *nextC1 = carry2;
    uint32_t *curC2 = carry3, *nextC2 = carry4;
    uint32_t *curC3 = carry5, *nextC3 = carry6;

    // only one thread initializes the global counter
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        globalSync[0] = 1;
    }
    grid.sync();

    // zero all six carry buffers
    for (int32_t i = idx; i < N + 1; i += stride) {
        curC1[i] = nextC1[i] = 0u;
        curC2[i] = nextC2[i] = 0u;
        curC3[i] = nextC3[i] = 0u;
    }
    grid.sync();

    constexpr int maxIter = 1000;
    uint32_t prevCount = 0;

    // convergent rippling loop
    for (int iter = 0; iter < maxIter; ++iter) {
        // if no new carries last iter, we’re done
        if (globalSync[0] == prevCount) break;
        prevCount = globalSync[0];
        grid.sync();

        uint32_t localNew = 0u;

        // each thread processes a grid-stride subset
        for (int32_t i = idx; i < N; i += stride) {
            //
            // ——— ABC True ———
            //
            int32_t in1 = (i == 0 ? 0 : int32_t(curC1[i]));
            int64_t sum1 = int64_t(final128_ABC_True[i]) + in1;
            uint32_t lo1 = uint32_t(sum1);
            final128_ABC_True[i] = lo1;
            int32_t new1 = int32_t((sum1 - int64_t(lo1)) >> 32);
            if (i < N - 1) nextC1[i + 1] = uint32_t(new1);
            else          nextC1[i + 1] = curC1[i + 1] + uint32_t(new1);
            localNew += (new1 != 0);

            //
            // ——— ABC False ———
            //
            int32_t in2 = (i == 0 ? 0 : int32_t(curC2[i]));
            int64_t sum2 = int64_t(final128_ABC_False[i]) + in2;
            uint32_t lo2 = uint32_t(sum2);
            final128_ABC_False[i] = lo2;
            int32_t new2 = int32_t((sum2 - int64_t(lo2)) >> 32);
            if (i < N - 1) nextC2[i + 1] = uint32_t(new2);
            else          nextC2[i + 1] = curC2[i + 1] + uint32_t(new2);
            localNew += (new2 != 0);

            //
            // ——— D + E (DE) ———
            //
            uint32_t in3 = (i == 0 ? 0u : curC3[i]);
            uint64_t sum3 = final128_DE[i] + in3;
            uint32_t lo3 = uint32_t(sum3);
            final128_DE[i] = lo3;
            uint32_t new3 = uint32_t(sum3 >> 32);
            if (i < N - 1) nextC3[i + 1] = new3;
            else          nextC3[i + 1] = curC3[i + 1] + new3;
            localNew += (new3 != 0);
        }

        grid.sync();

        // bump global counter if any thread saw new carries
        if (localNew) {
            atomicAdd(&globalSync[0], localNew);
        }

        // swap buffers
        {
            auto *t = curC1;
            curC1 = nextC1;
            nextC1 = t;
        }

        {
            auto *t = curC2;
            curC2 = nextC2;
            nextC2 = t;
        }

        {
            auto *t = curC3;
            curC3 = nextC3;
            nextC3 = t;
        }

        grid.sync();
    }

    // thread (0,0) writes out the final carries
    if (block.group_index().x == 0 && block.thread_index().x == 0) {
        carryAcc_ABC_True = int32_t(curC1[N]);
        carryAcc_ABC_False = int32_t(curC2[N]);
        carryAcc_DE = int32_t(curC3[N]);
    }
    grid.sync();
}
