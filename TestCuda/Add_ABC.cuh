template<class SharkFloatParams>
__device__ void
ComputeABCComparison (
    // normalized, extended digit arrays
    const uint32_t *extA,
    const uint32_t *extB,
    const uint32_t *extC,
    // sizes
    int32_t actualDigits,
    int32_t numActualDigitsPlusGuard,
    // normalization shifts
    int32_t shiftA,
    int32_t shiftB,
    int32_t shiftC,
    // effective exponents
    int32_t effExpA,
    int32_t effExpB,
    int32_t effExpC,
    // input signs
    bool signA,
    bool signB,
    bool signC,
    // outputs:
    int32_t &expRawAB,         // exponent of raw_signed(A–B)
    bool &ABIsBiggerThanC,  // |raw_signed(A–B)| > |C| ?
    bool &ACIsBiggerThanB,  // |raw_signed(A–C)| > |B| ?
    bool &BCIsBiggerThanA   // |raw_signed(B–C)| > |A| ?
) {
    auto cmpSignedRawVsThird = [](
        const uint32_t *extX,
        const uint32_t *extY,
        const uint32_t *extZ,
        int32_t actualDigits,
        int32_t numActualDigitsPlusGuard,
        int32_t expX,
        int32_t expY,
        int32_t expZ,
        int32_t shiftX,
        int32_t shiftY,
        int32_t shiftZ,
        bool    sX,
        bool    sY,
        int32_t &outExpRawXY,
        bool &outXYgtZ
        ) {
            // 1) Determine which magnitude is larger (for alignment only)
            bool XgtY = CompareMagnitudes2Way(
                expX, expY,
                actualDigits, numActualDigitsPlusGuard,
                shiftX, shiftY,
                extX, extY);

            // 2) Compute exponent-diff and biased exponent
            int32_t diffXY = XgtY ? (expX - expY) : (expY - expX);
            int32_t expXY = XgtY ? expX : expY;

            // 3) Build a signed‐raw array of |(±X) + (±Y)| magnitudes
            std::vector<uint64_t> magXY(numActualDigitsPlusGuard);
            for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
                uint64_t x_l, y_l;
                GetCorrespondingLimbs<SharkFloatParams>(
                    extX, actualDigits, numActualDigitsPlusGuard,
                    extY, actualDigits, numActualDigitsPlusGuard,
                    shiftX, shiftY,
                    XgtY, diffXY,
                    i, x_l, y_l);

                int64_t signedXY =
                    (sX ? -int64_t(x_l) : int64_t(x_l))
                    + (sY ? -int64_t(y_l) : int64_t(y_l));

                magXY[i] = signedXY < 0
                    ? uint64_t(-signedXY)
                    : uint64_t(signedXY);
            }

            // 4) Find MSB of magXY (all 64 bits!) to compute outExpRawXY
            auto clz64 = [](uint64_t v) {
                // on GCC/Clang you could use __builtin_clzll(v), but here's a portable fallback:
                if (v == 0) return 64;
                int c = 0;
                for (int b = 63; b >= 0; --b) {
                    if (v & (1ull << b)) break;
                    ++c;

                }
                return c;
                };

            int32_t msd = -1;
            uint64_t absAtMSD64 = 0;
            for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
                if (magXY[i] != 0) {
                    msd = i;
                    absAtMSD64 = magXY[i];
                    break;
                }
            }

            if (msd < 0) {
                outExpRawXY = -100000000;  // exact zero
            } else {
                int32_t clz = clz64(absAtMSD64);
                // now the “1” bit is at position (63-clz) within this 64-bit limb
                int32_t bitIndex = msd * 32 + (63 - clz);
                outExpRawXY = expXY - ((numActualDigitsPlusGuard * 32 - 1) - bitIndex);
            }

            // 5) Compare magXY vs Z’s magnitude
            if (outExpRawXY > expZ) {
                outXYgtZ = true;
            } else if (outExpRawXY < expZ) {
                outXYgtZ = false;
            } else {
                // tie on exponent → compare limbs high→low
                outXYgtZ = false;
                for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
                    uint32_t z_d = GetNormalizedDigit(extZ, actualDigits, numActualDigitsPlusGuard, shiftZ, i);
                    if (magXY[i] > z_d) {
                        outXYgtZ = true;
                        break;
                    } else if (magXY[i] < z_d) {
                        outXYgtZ = false;
                        break;
                    }
                }
            }
        };

    // compute raw_signed(A–B) vs C
    cmpSignedRawVsThird(
        extA, extB, extC,
        actualDigits, numActualDigitsPlusGuard,
        effExpA, effExpB, effExpC,
        shiftA, shiftB, shiftC,
        signA, signB,
        expRawAB,
        ABIsBiggerThanC);

    // compute raw_signed(A–C) vs B
    {
        int32_t dummyExp;
        cmpSignedRawVsThird(
            extA, extC, extB,
            actualDigits, numActualDigitsPlusGuard,
            effExpA, effExpC, effExpB,
            shiftA, shiftC, shiftB,
            signA, signC,
            dummyExp,
            ACIsBiggerThanB);
    }

    // compute raw_signed(B–C) vs A
    {
        int32_t dummyExp;
        cmpSignedRawVsThird(
            extB, extC, extA,
            actualDigits, numActualDigitsPlusGuard,
            effExpB, effExpC, effExpA,
            shiftB, shiftC, shiftA,
            signB, signC,
            dummyExp,
            BCIsBiggerThanA);
    }
}

// "Strict" ordering of three magnitudes (ignores exact ties - see note below)
enum class ThreeWayMagnitude {
    A_GT_B_GT_C,  // A > B > C
    A_GT_C_GT_B,  // A > C > B
    B_GT_A_GT_C,  // B > A > C
    B_GT_C_GT_A,  // B > C > A
    C_GT_A_GT_B,  // C > A > B
    C_GT_B_GT_A   // C > B > A
};

__device__ ThreeWayMagnitude
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
    auto cmp = [](
        const int32_t actualDigits, const int32_t numActualDigitsPlusGuard,
        const uint32_t *e1, int32_t s1, int32_t exp1,
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
    if (cmp(actualDigits, numActualDigitsPlusGuard, extA, shiftA, effExpA, extB, shiftB, effExpB) &&
        cmp(actualDigits, numActualDigitsPlusGuard, extA, shiftA, effExpA, extC, shiftC, effExpC)) {
        // now order B vs C
        outExp = effExpA;
        if (cmp(actualDigits, numActualDigitsPlusGuard, extB, shiftB, effExpB, extC, shiftC, effExpC))
            return ThreeWayMagnitude::A_GT_B_GT_C;
        else
            return ThreeWayMagnitude::A_GT_C_GT_B;
    }
    // 2) Is B the strict max?
    else if (
        cmp(actualDigits, numActualDigitsPlusGuard, extB, shiftB, effExpB, extA, shiftA, effExpA) &&
        cmp(actualDigits, numActualDigitsPlusGuard, extB, shiftB, effExpB, extC, shiftC, effExpC)) {
        // now order A vs C
        outExp = effExpB;
        if (cmp(actualDigits, numActualDigitsPlusGuard, extA, shiftA, effExpA, extC, shiftC, effExpC))
            return ThreeWayMagnitude::B_GT_A_GT_C;
        else
            return ThreeWayMagnitude::B_GT_C_GT_A;
    }
    // 3) Otherwise C is the (strict) max
    else {
        // order A vs B
        outExp = effExpC;
        if (cmp(actualDigits, numActualDigitsPlusGuard, extA, shiftA, effExpA, extB, shiftB, effExpB))
            return ThreeWayMagnitude::C_GT_A_GT_B;
        else
            return ThreeWayMagnitude::C_GT_B_GT_A;
    }
}

template<class SharkFloatParams, int32_t CallIndex>
__device__
void Phase1_ABC (
    cg::thread_block &block,
    cg::grid_group &grid,
    const RecordIt record,
    const int32_t idx,
    const ThreeWayMagnitude ordering,
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
    const int32_t  biasedExpABC_local,
    const int32_t  bias,
    bool &IsNegativeABC,
    int32_t &outExponent_ABC,
    uint64_t *final128_ABC,  // the extended result digits
    DebugState<SharkFloatParams> *debugStates
)
{
    if (idx > 0) {
        return;
    }

    // Final exponent before bias correction
    outExponent_ABC = biasedExpABC_local - bias;

    bool ABIsBiggerThanC, ACIsBiggerThanB, BCIsBiggerThanA;
    int32_t expRawAB;
    ComputeABCComparison<SharkFloatParams>(
        extA, extB, extC,
        actualDigits, numActualDigitsPlusGuard,
        shiftA, shiftB, shiftC,
        effExpA, effExpB, effExpC,
        IsNegativeA, IsNegativeB, IsNegativeC,
        expRawAB,
        ABIsBiggerThanC,
        ACIsBiggerThanB,
        BCIsBiggerThanA);

    // How far each input must be shifted right to align at biasedExpABC_local
    int32_t diffA = biasedExpABC_local - effExpA;
    int32_t diffB = biasedExpABC_local - effExpB;
    int32_t diffC = biasedExpABC_local - effExpC;

    // --- Fused loop: subtract middle from largest, then add smallest ---
    uint64_t X = 0;
    uint64_t Y = 0;
    uint64_t Z = 0;

    bool signX = false;
    bool signY = false;
    bool signZ = false;

    switch (ordering) {
    case ThreeWayMagnitude::A_GT_B_GT_C:
        signX = IsNegativeA;
        signY = IsNegativeB;
        signZ = IsNegativeC;
        break;
    case ThreeWayMagnitude::A_GT_C_GT_B:
        signX = IsNegativeA;
        signY = IsNegativeC;
        signZ = IsNegativeB;
        break;
    case ThreeWayMagnitude::B_GT_A_GT_C:
        signX = IsNegativeB;
        signY = IsNegativeA;
        signZ = IsNegativeC;
        break;
    case ThreeWayMagnitude::B_GT_C_GT_A:
        signX = IsNegativeB;
        signY = IsNegativeC;
        signZ = IsNegativeA;
        break;
    case ThreeWayMagnitude::C_GT_A_GT_B:
        signX = IsNegativeC;
        signY = IsNegativeA;
        signZ = IsNegativeB;
        break;
    case ThreeWayMagnitude::C_GT_B_GT_A:
        signX = IsNegativeC;
        signY = IsNegativeB;
        signZ = IsNegativeA;
        break;
    default:
        assert(false);
    }

    bool XYgtZ = false;
    switch (ordering) {
    case ThreeWayMagnitude::A_GT_B_GT_C:
    case ThreeWayMagnitude::B_GT_A_GT_C:
        XYgtZ = ABIsBiggerThanC;    break;
    case ThreeWayMagnitude::A_GT_C_GT_B:
    case ThreeWayMagnitude::C_GT_A_GT_B:
        XYgtZ = ACIsBiggerThanB;    break;
    case ThreeWayMagnitude::B_GT_C_GT_A:
    case ThreeWayMagnitude::C_GT_B_GT_A:
        XYgtZ = BCIsBiggerThanA;    break;
    }

    // before the loop: we'll stash the final sign here
    IsNegativeABC = false;
    for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
        // Pick (X, Y, Z) = (largest, middle, smallest) per 'ordering'
        switch (ordering) {
        case ThreeWayMagnitude::A_GT_B_GT_C:
            X = GetNormalizedDigit(extA, actualDigits, numActualDigitsPlusGuard, shiftA, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, numActualDigitsPlusGuard, shiftB, diffB, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, numActualDigitsPlusGuard, shiftC, diffC, i);
            break;
        case ThreeWayMagnitude::A_GT_C_GT_B:
            X = GetNormalizedDigit(extA, actualDigits, numActualDigitsPlusGuard, shiftA, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, numActualDigitsPlusGuard, shiftC, diffC, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, numActualDigitsPlusGuard, shiftB, diffB, i);
            break;
        case ThreeWayMagnitude::B_GT_A_GT_C:
            X = GetNormalizedDigit(extB, actualDigits, numActualDigitsPlusGuard, shiftB, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, numActualDigitsPlusGuard, shiftA, diffA, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, numActualDigitsPlusGuard, shiftC, diffC, i);
            break;
        case ThreeWayMagnitude::B_GT_C_GT_A:
            X = GetNormalizedDigit(extB, actualDigits, numActualDigitsPlusGuard, shiftB, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extC, actualDigits, numActualDigitsPlusGuard, shiftC, diffC, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, numActualDigitsPlusGuard, shiftA, diffA, i);
            break;
        case ThreeWayMagnitude::C_GT_A_GT_B:
            X = GetNormalizedDigit(extC, actualDigits, numActualDigitsPlusGuard, shiftC, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, numActualDigitsPlusGuard, shiftA, diffA, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, numActualDigitsPlusGuard, shiftB, diffB, i);
            break;
        case ThreeWayMagnitude::C_GT_B_GT_A:
            X = GetNormalizedDigit(extC, actualDigits, numActualDigitsPlusGuard, shiftC, i);
            Y = GetShiftedNormalizedDigit<SharkFloatParams>(extB, actualDigits, numActualDigitsPlusGuard, shiftB, diffB, i);
            Z = GetShiftedNormalizedDigit<SharkFloatParams>(extA, actualDigits, numActualDigitsPlusGuard, shiftA, diffA, i);
            break;
        default:
            assert(false);  // unknown ordering
        }

        // 2) always “larger - smaller” when signs differ, otherwise add
        uint64_t magXY = (signX == signY) ? (X + Y)
            : (X - Y);
        bool   signXY = signX;  // if we subtracted, X was the larger so X’s sign wins

        uint64_t magABC;
        if (signXY == signZ) {
            // same sign → addition
            magABC = magXY + Z;
            IsNegativeABC = signXY;
        } else if (XYgtZ) {
            // |X±Y| ≥ |Z| → subtraction in that order
            magABC = magXY - Z;
            IsNegativeABC = signXY;
        } else {
            // |Z| > |X±Y| → subtraction the other way
            magABC = Z - magXY;
            IsNegativeABC = signZ;
        }

        // 3) store the always-nonnegative magnitude
        final128_ABC[i] = magABC;
    }

    if constexpr (SharkDebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, CallIndex, DebugStatePurpose::Z2XX, uint64_t>(
            record, debugStates, grid, block, final128_ABC, numActualDigitsPlusGuard);
        grid.sync();
    }
}

template<class SharkFloatParams>
__device__
void CarryPropagation_ABC (
    uint32_t *sharedData,
    uint32_t *globalSync,
    const int32_t idx,
    const int32_t numActualDigitsPlusGuard,
    uint64_t *final128_ABC,         // raw signed limbs from Phase1_ABC
    uint32_t *carry1,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
    uint32_t *carry2,        // global memory array for intermediate carries/borrows (length numActualDigitsPlusGuard+1)
    int32_t &carryAcc,
    cg::thread_block &block,
    cg::grid_group &grid
)
{
    if (idx != 0) { // TODO
        return;
    }

    // Start with zero carry/borrow
    carryAcc = 0;

    for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
        // reinterpret the 64-bit limb as signed
        int64_t limb = static_cast<int64_t>(final128_ABC[i]);

        // add in the previous carry (or borrow, if negative)
        int64_t sum = limb + carryAcc;

        // low 32 bits become the output digit
        uint32_t low32 = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        final128_ABC[i] = low32;

        // compute next carryAcc = floor(sum/2^32)
        // (sum - low32) is a multiple of 2^32, so this division is exact
        carryAcc = (sum - static_cast<int64_t>(low32)) >> 32;
        // -or equivalently-
        // carryAcc = (sum - static_cast<int64_t>(low32)) / (1LL << 32);
    }
}