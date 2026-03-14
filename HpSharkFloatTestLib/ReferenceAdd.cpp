#include "ReferenceAdd.h"
#include "DbgHeap.h"
#include "DebugChecksum.h"
#include "DebugChecksumHost.h"
#include "HpSharkFloat.h"
#include "TestVerbose.h"
#include "ThreeWayMagnitude.h"

#include <algorithm>
#include <assert.h>
#include <bit> // for std::countl_zero (C++20)
#include <cstdint>
#include <cstring> // for memset
#include <iostream>
#include <vector>

struct ABCInputs {
    bool IsNegA, IsNegB, IsNegC;
    const uint32_t *extA, *extB, *extC;
    int32_t shiftA, shiftB, shiftC;
    int32_t effExpA, effExpB, effExpC;
    ThreeWayLargestOrdering ordering;
};

struct ABCOutputs {
    bool outSignTrue, outSignFalse;
    int32_t outExpTrue, outExpFalse;
    std::vector<uint64_t> *resultTrue;
    std::vector<uint64_t> *resultFalse;
};

struct DEInputs {
    bool DIsBigger;
    bool sameSign;
    const uint32_t *extD, *extE;
    int32_t shiftD, shiftE;
    int32_t effExpD, effExpE;
    int32_t newExpD, newExpE;
    bool IsNegD, IsNegE;
};

struct DEOutputs {
    int32_t outExponent;
    std::vector<uint64_t> *result;
};

static constexpr auto UseBellochPropagation = false;

// A - B + C
// D + E

// Direction tag for our funnel-shift helper
enum class Dir { Left, Right };

/// Funnels two 32-bit words from 'data' around a bit-offset boundary.
/// - For Dir::Right, this emulates a right shift across word boundaries.
/// - For Dir::Left,  this emulates a left  shift across word boundaries.
/// 'N' is the number of valid words in 'data'; out-of-range indices yield 0.
template <Dir D>
static uint32_t
FunnelShift32(const auto *data, int idx, int N, int bitOffset)
{
    int wordOff = bitOffset / 32;
    int b = bitOffset % 32;

    auto pick = [&](int i) -> uint32_t {
        return (i < 0 || i >= N) ? 0u : static_cast<uint32_t>(data[i]);
    };

    uint32_t low, high;
    if constexpr (D == Dir::Right) {
        low = pick(idx + wordOff);
        high = pick(idx + wordOff + 1);
    } else {
        low = pick(idx - wordOff);
        high = pick(idx - wordOff - 1);
    }

    if (b == 0)
        return low;
    if constexpr (D == Dir::Right)
        return (low >> b) | (high << (32 - b));
    else
        return (low << b) | (high >> (32 - b));
}

/// Retrieves the digit at 'idx' after a left shift by 'shiftBits',
/// treating words beyond 'actualDigits' as zero (within an 'numActualDigitsPlusGuard' buffer).
static uint32_t
GetNormalizedDigit(const uint32_t *digits,
                   int32_t actualDigits,
                   int32_t numActualDigitsPlusGuard,
                   int32_t shiftBits,
                   int32_t idx)
{
    // ensure idx is within the extended buffer
    (void)numActualDigitsPlusGuard; // unused in this function, but kept for clarity
    assert(idx >= 0 && idx < numActualDigitsPlusGuard);

    // funnel-shift left within the 'actualDigits' region
    return FunnelShift32<Dir::Left>(digits, idx, actualDigits, shiftBits);
}

// Counts the number of leading zero bits in a 32-bit unsigned integer.
// This is a portable implementation of the count leading zeros operation.
static int32_t
CountLeadingZeros(const uint32_t x)
{
#if defined(__CUDA_ARCH__)
    // __clz returns 0–32 inclusive, even for x==0
    return __clz(static_cast<int>(x));
#else
    // std::countl_zero is constexpr in C++20 and returns 32 for x==0
    return static_cast<int>(std::countl_zero(x));
#endif
}

//
// Multi-word shift routines for little-endian arrays
//

// Generic multi-word shift using the requested parameter names
template <Dir D>
static void
MultiWordShift(const auto *in,
               const int32_t numActualDigitsPlusGuard,
               const int32_t shiftNeeded,
               uint32_t *out,
               const int32_t outSz)
{
    assert(numActualDigitsPlusGuard >= outSz);
    for (int32_t i = 0; i < outSz; ++i) {
        out[i] = FunnelShift32<D>(in,
                                  /* idx       = */ i,
                                  /* N         = */ numActualDigitsPlusGuard,
                                  /* bitOffset = */ shiftNeeded);
    }
}

// Retrieves a limb from an extended array, returning zero for indices beyond the actual digit count.
// This handles the boundary between actual digits and guard digits in extended precision arithmetic.
static uint32_t
GetExtLimb(const uint32_t *ext,
           const int32_t actualDigits,
           const int32_t numActualDigitsPlusGuard,
           const int32_t idx)
{

    (void)numActualDigitsPlusGuard; // unused in this function, but kept for clarity

    if (idx < actualDigits) {
        return ext[idx];
    } else {
        assert(idx < numActualDigitsPlusGuard);
        return 0;
    }
}

//
// New ExtendedNormalize routine
//
// Instead of copying (shifting) the entire array, this routine computes
// a shift offset (L) such that if you were to left-shift the original array by L bits,
// its most-significant set bit would be in the highest bit position of the extended field.
// It then adjusts the stored exponent accordingly and returns the shift offset.
//

// Computes the bit shift offset needed to normalize an extended precision number.
// Returns the shift amount and updates the exponent accordingly, without actually performing the shift.
static int32_t
ExtendedNormalizeShiftIndex(const uint32_t *ext,
                            const int32_t actualDigits,
                            const int32_t numActualDigitsPlusGuard,
                            int32_t &storedExp,
                            bool &isZero)
{
    int32_t msd = numActualDigitsPlusGuard - 1;
    while (msd >= 0 && GetExtLimb(ext, actualDigits, numActualDigitsPlusGuard, msd) == 0)
        msd--;
    if (msd < 0) {
        isZero = true;
        return 0; // For zero, the shift offset is irrelevant.
    }
    isZero = false;
    const int32_t clz = CountLeadingZeros(GetExtLimb(ext, actualDigits, numActualDigitsPlusGuard, msd));
    // In little-endian, the overall bit index of the MSB is:
    //    current_msb = msd * 32 + (31 - clz)
    const int32_t current_msb = msd * 32 + (31 - clz);
    const int32_t totalExtBits = numActualDigitsPlusGuard * 32;
    // Compute the left-shift needed so that the MSB moves to bit (totalExtBits - 1).
    const int32_t L = (totalExtBits - 1) - current_msb;
    // Adjust the exponent as if we had shifted the number left by L bits.
    storedExp -= L;
    return L;
}

// New helper: Computes the aligned digit for the normalized value on the fly.
// 'diffDE' is the additional right shift required for alignment.
template <class SharkFloatParams>
static uint32_t
GetShiftedNormalizedDigit(const uint32_t *ext,
                          const int32_t actualDigits,
                          const int32_t numActualDigitsPlusGuard,
                          const int32_t shiftOffset,
                          const int32_t diff,
                          const int32_t idx)
{
    // const int32_t n = SharkFloatParams::GlobalNumUint32; // normalized length
    const int32_t wordShift = diff / 32;
    const int32_t bitShift = diff % 32;
    const uint32_t lower =
        (idx + wordShift < numActualDigitsPlusGuard)
            ? GetNormalizedDigit(
                  ext, actualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift)
            : 0;
    const uint32_t upper =
        (idx + wordShift + 1 < numActualDigitsPlusGuard)
            ? GetNormalizedDigit(
                  ext, actualDigits, numActualDigitsPlusGuard, shiftOffset, idx + wordShift + 1)
            : 0;
    if (bitShift == 0)
        return lower;
    else
        return (lower >> bitShift) | (upper << (32 - bitShift));
}

template <class SharkFloatParams>
static void
GetCorrespondingLimbs(const uint32_t *extA,
                      const int32_t actualASize,
                      const int32_t extASize,
                      const uint32_t *extB,
                      const int32_t actualBSize,
                      const int32_t extBSize,
                      const int32_t shiftA,
                      const int32_t shiftB,
                      const bool AIsBiggerMagnitude,
                      const int32_t diff,
                      const int32_t index,
                      uint64_t &alignedA,
                      uint64_t &alignedB)
{
    if (AIsBiggerMagnitude) {
        // A is larger: normalized A is used as is.
        // For B, we normalize and then shift right by 'diffDE'.
        alignedA = GetNormalizedDigit(extA, actualASize, extASize, shiftA, index);
        alignedB = GetShiftedNormalizedDigit<SharkFloatParams>(
            extB, actualBSize, extBSize, shiftB, diff, index);
    } else {
        // B is larger: normalized B is used as is.
        // For A, we normalize and shift right by 'diffDE'.
        alignedB = GetNormalizedDigit(extB, actualBSize, extBSize, shiftB, index);
        alignedA = GetShiftedNormalizedDigit<SharkFloatParams>(
            extA, actualASize, extASize, shiftA, diff, index);
    }
}

// Retrieves the current debug state for a given purpose and array.
template <class SharkFloatParams, DebugStatePurpose Purpose, typename ArrayType>
static const DebugStateHost<SharkFloatParams> &
GetCurrentDebugState(std::vector<DebugStateHost<SharkFloatParams>> &debugStates,
                     const ArrayType *arrayToChecksum,
                     size_t arraySize)
{

    constexpr auto curPurpose = static_cast<int>(Purpose);
    constexpr auto CallIndex = 0;
    constexpr auto UseConvolution = UseConvolution::No;
    constexpr auto RecursionDepth = 0;

    auto &retval = debugStates[curPurpose];
    retval.Reset(arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex, UseConvolution);
    return retval;
}

static bool
CompareMagnitudes2Way(const int32_t effExpA,
                      const int32_t effExpB,
                      const int32_t actualDigits,
                      const int32_t numActualDigitsPlusGuard,
                      const int32_t shiftA,
                      const int32_t shiftB,
                      const uint32_t *extA,
                      const uint32_t *extB)
{
    bool AIsBiggerMagnitude;

    if (effExpA > effExpB) {
        AIsBiggerMagnitude = true;
    } else if (effExpA < effExpB) {
        AIsBiggerMagnitude = false;
    } else {
        AIsBiggerMagnitude = false; // default if equal
        for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; i--) {
            uint32_t digitA =
                GetNormalizedDigit(extA, actualDigits, numActualDigitsPlusGuard, shiftA, i);
            uint32_t digitB =
                GetNormalizedDigit(extB, actualDigits, numActualDigitsPlusGuard, shiftB, i);
            if (digitA > digitB) {
                AIsBiggerMagnitude = true;
                break;
            } else if (digitA < digitB) {
                AIsBiggerMagnitude = false;
                break;
            }
        }
    }

    return AIsBiggerMagnitude;
}

// Compares magnitudes of two normalized extended values and returns true if A >= B.
static ThreeWayLargestOrdering
CompareMagnitudes3Way(const int32_t effExpA,
                      const int32_t effExpB,
                      const int32_t effExpC,
                      const int32_t actualDigits,
                      const int32_t numActualDigitsPlusGuard,
                      const int32_t shiftA,
                      const int32_t shiftB,
                      const int32_t shiftC,
                      const uint32_t *extA,
                      const uint32_t *extB,
                      const uint32_t *extC)
{
    // Helper: returns true if "first" is strictly bigger than "second"
    auto cmp =
        [&](const uint32_t *e1, int32_t s1, int32_t exp1, const uint32_t *e2, int32_t s2, int32_t exp2) {
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
}

// Propagates raw 64-bit extended results into 32-bit digits with signed carry support.
template <class SharkFloatParams>
static void
CarryPropagation_ABC_Single(const int32_t numActualDigitsPlusGuard,
                            const std::vector<uint64_t> &extResult,
                            int32_t &carryAcc,
                            std::vector<uint64_t> &propagatedResult)
{
    // Start with zero carry/borrow
    carryAcc = 0;

    assert(numActualDigitsPlusGuard == static_cast<int32_t>(extResult.size()));
    assert(propagatedResult.size() == numActualDigitsPlusGuard);

    for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
        // reinterpret the 64-bit limb as signed
        int64_t limb = static_cast<int64_t>(extResult[i]);

        // add in the previous carry (or borrow, if negative)
        int64_t sum = limb + carryAcc;

        // low 32 bits become the output digit
        uint32_t low32 = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
        propagatedResult[i] = low32;

        // compute next carryAcc = floor(sum/2^32)
        // (sum - low32) is a multiple of 2^32, so this division is exact
        carryAcc = (sum - static_cast<int64_t>(low32)) >> 32;
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "CarryPropagation3 final carryAcc = " << carryAcc << std::endl;
    }
}

// Wrapper: processes orbit carry chain and optionally NR carry chain in one call.
template <class SharkFloatParams>
static void
CarryPropagation_ABC(const int32_t numActualDigitsPlusGuard,
                     const std::vector<uint64_t> &extResult,
                     int32_t &carryAcc,
                     std::vector<uint64_t> &propagatedResult,
                     const std::vector<uint64_t> *nrExtResult,
                     int32_t *nrCarryAcc,
                     std::vector<uint64_t> *nrPropagatedResult)
{
    CarryPropagation_ABC_Single<SharkFloatParams>(
        numActualDigitsPlusGuard, extResult, carryAcc, propagatedResult);

    if (nrExtResult) {
        CarryPropagation_ABC_Single<SharkFloatParams>(
            numActualDigitsPlusGuard, *nrExtResult, *nrCarryAcc, *nrPropagatedResult);
    }
}

template <class SharkFloatParams>
static void
CarryPropagationPP_ABC_Single(const int32_t n,
                              const std::vector<uint64_t> &extResultVector,
                              int32_t &finalCarryOut,
                              std::vector<uint64_t> &propagatedResultVector)
{
    assert(extResultVector.size() == size_t(n));
    assert(propagatedResultVector.size() == size_t(n));

    if (n == 0) {
        finalCarryOut = 0;
        return;
    }

    auto nextPow2 = [](int x) {
        int p = 1;
        while (p < x)
            p <<= 1;
        return p;
    };

    auto toSigned64 = [](uint64_t u) -> int64_t {
        int64_t s;
        std::memcpy(&s, &u, sizeof(s));
        return s;
    };

    struct Pair {
        int64_t c;
        uint32_t d;
    };

    const Pair Z{0, 0};

    auto combine = [](const Pair &a, const Pair &b) -> Pair {
        uint64_t sum32 = uint64_t(a.d) + uint64_t(b.d);
        uint32_t new_d = static_cast<uint32_t>(sum32);
        int64_t k = static_cast<int64_t>(sum32 >> 32);
        int64_t new_c = a.c + b.c + k;
        return Pair{new_c, new_d};
    };

    // -------- CORRECTED decomposition --------
    std::vector<Pair> elems(n);
    for (int i = 0; i < n; ++i) {
        const int64_t limb = toSigned64(extResultVector[i]);
        const uint32_t d_i = static_cast<uint32_t>(limb & 0xFFFFFFFFULL);

        // Compute c_i such that limb = (c_i << 32) + d_i
        // This matches the serial version's: carryAcc = (sum - low32) >> 32
        const int64_t c_i = (limb - static_cast<int64_t>(d_i)) >> 32;

        elems[i] = Pair{c_i, d_i};
    }

    const int m = nextPow2(n);
    std::vector<Pair> scan(m, Z);
    for (int i = 0; i < n; ++i)
        scan[i] = elems[i];

    // upsweep
    for (int offset = 1; offset < m; offset <<= 1) {
        for (int i = (offset << 1) - 1; i < m; i += (offset << 1)) {
            scan[i] = combine(scan[i - offset], scan[i]);
        }
    }

    // downsweep
    scan[m - 1] = Z;
    for (int offset = (m >> 1); offset >= 1; offset >>= 1) {
        for (int i = (offset << 1) - 1; i < m; i += (offset << 1)) {
            Pair left = scan[i - offset];
            Pair right = scan[i];
            scan[i - offset] = right;
            scan[i] = combine(left, right);
        }
    }

    // emit results (use carry-in = scan[i].c, not low32(P_i))
    for (int i = 0; i < n; ++i) {
        const int64_t limb_i = toSigned64(extResultVector[i]);
        const int64_t carry_in = scan[i].c;    // = floor(P_{i-1}/2^32)
        const int64_t sum = limb_i + carry_in; // matches the serial step
        propagatedResultVector[i] = static_cast<uint64_t>(static_cast<uint32_t>(sum & 0xFFFFFFFFULL));
    }

    // final carry-out is floor(P_{n-1}/2^32)
    {
        Pair S_last = combine(scan[n - 1], elems[n - 1]); // inclusive prefix for the last element
        finalCarryOut = static_cast<int32_t>(S_last.c);
    }

    // verification (can be removed after testing)
    {
        int32_t tempFinalCarryOut;
        std::vector<uint64_t> tempPropagatedResultVector(n);

        CarryPropagation_ABC_Single<SharkFloatParams>(
            n, extResultVector, tempFinalCarryOut, tempPropagatedResultVector);

        for (int i = 0; i < n; ++i) {
            assert(tempPropagatedResultVector[i] == propagatedResultVector[i]);
        }
        assert(tempFinalCarryOut == finalCarryOut);
    }
}

// Wrapper: processes orbit carry chain and optionally NR carry chain in one call.
template <class SharkFloatParams>
static void
CarryPropagationPP_ABC(const int32_t n,
                       const std::vector<uint64_t> &extResultVector,
                       int32_t &finalCarryOut,
                       std::vector<uint64_t> &propagatedResultVector,
                       const std::vector<uint64_t> *nrExtResultVector,
                       int32_t *nrFinalCarryOut,
                       std::vector<uint64_t> *nrPropagatedResultVector)
{
    CarryPropagationPP_ABC_Single<SharkFloatParams>(
        n, extResultVector, finalCarryOut, propagatedResultVector);

    if (nrExtResultVector) {
        CarryPropagationPP_ABC_Single<SharkFloatParams>(
            n, *nrExtResultVector, *nrFinalCarryOut, *nrPropagatedResultVector);
    }
}

// Applies straightforward carry or borrow propagation across multi-word signed results.
template <class SharkFloatParams>
void
CarryPropagation_DE(const bool sameSign,
                    const int32_t numActualDigitsPlusGuard,
                    std::vector<uint64_t> &extResult,
                    int32_t &carry,
                    std::vector<uint64_t> &propagatedResult)
{
    uint32_t carryUnsigned = static_cast<uint32_t>(carry);

    if (sameSign) {
        // Propagate carry for addition.
        for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
            int64_t sum = (int64_t)extResult[i] + carryUnsigned;
            propagatedResult[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
            carryUnsigned = sum >> 32;
        }

        // Note we'll handle final carry later.
    } else {
        // Propagate borrow for subtraction.
        int64_t borrow = 0;
        for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
            int64_t diffVal = (int64_t)extResult[i] - borrow;
            if (diffVal < 0) {
                diffVal += (1LL << 32);
                borrow = 1;
            } else {
                borrow = 0;
            }
            propagatedResult[i] = (uint32_t)(diffVal & 0xFFFFFFFFULL);
        }
        assert(borrow == 0 && "Final borrow in subtraction should be zero");
    }

    carry = static_cast<int32_t>(carryUnsigned);
}

// Performs the first phase of D+E addition or subtraction across extended precision digits.
// Processes orbit and optionally NR data in a single call.
template <class SharkFloatParams>
static void
Phase1_DE(int32_t numActualDigitsPlusGuard,
          int32_t actualDigits,
          int32_t bias,
          DEInputs orbInputs,
          DEOutputs &orbOutputs,
          DEInputs nrInputs,
          DEOutputs &nrOutputs,
          std::vector<DebugStateHost<SharkFloatParams>> &debugStates)
{
    auto processOne = [&](const DEInputs &in, DEOutputs &out) {
        const int32_t diffDE = in.DIsBigger ? (in.effExpD - in.effExpE) : (in.effExpE - in.effExpD);
        out.outExponent = in.DIsBigger ? in.newExpD : in.newExpE;

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "diffDE: " << diffDE << std::endl;
            std::cout << "outExponent_DE: " << out.outExponent << std::endl;
        }

        // --- Phase 1: Raw Extended Arithmetic ---
        // Compute the raw limb-wise result without propagation.
        if (in.sameSign) {
            // Addition branch.
            for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
                uint64_t alignedA = 0, alignedB = 0;
                GetCorrespondingLimbs<SharkFloatParams>(in.extD,
                                                        actualDigits,
                                                        numActualDigitsPlusGuard,
                                                        in.extE,
                                                        actualDigits,
                                                        numActualDigitsPlusGuard,
                                                        in.shiftD,
                                                        in.shiftE,
                                                        in.DIsBigger,
                                                        diffDE,
                                                        i,
                                                        alignedA,
                                                        alignedB);
                (*out.result)[i] = alignedA + alignedB;
            }
        } else {
            // Subtraction branch.
            std::vector<uint64_t> alignedDDebug;
            std::vector<uint64_t> alignedEDebug;

            if (in.DIsBigger) {
                for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
                    uint64_t alignedA = 0, alignedB = 0;
                    GetCorrespondingLimbs<SharkFloatParams>(in.extD,
                                                            actualDigits,
                                                            numActualDigitsPlusGuard,
                                                            in.extE,
                                                            actualDigits,
                                                            numActualDigitsPlusGuard,
                                                            in.shiftD,
                                                            in.shiftE,
                                                            in.DIsBigger,
                                                            diffDE,
                                                            i,
                                                            alignedA,
                                                            alignedB);
                    // Compute raw difference (which may be negative).
                    const int64_t rawDiff = (int64_t)alignedA - (int64_t)alignedB;
                    (*out.result)[i] = (uint64_t)rawDiff;

                    alignedDDebug.push_back(alignedA);
                    alignedEDebug.push_back(alignedB);
                }
            } else {
                for (int32_t i = 0; i < numActualDigitsPlusGuard; i++) {
                    uint64_t alignedA = 0, alignedB = 0;
                    GetCorrespondingLimbs<SharkFloatParams>(in.extD,
                                                            actualDigits,
                                                            numActualDigitsPlusGuard,
                                                            in.extE,
                                                            actualDigits,
                                                            numActualDigitsPlusGuard,
                                                            in.shiftD,
                                                            in.shiftE,
                                                            in.DIsBigger,
                                                            diffDE,
                                                            i,
                                                            alignedA,
                                                            alignedB);
                    const int64_t rawDiff = (int64_t)alignedB - (int64_t)alignedA;
                    (*out.result)[i] = (uint64_t)rawDiff;

                    alignedDDebug.push_back(alignedA);
                    alignedEDebug.push_back(alignedB);
                }
            }

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "Phase1_DE - These are effectively the arrays we're adding and subtracting:"
                          << std::endl;
                std::cout << "alignedDDebug: " << VectorUintToHexString(alignedDDebug) << std::endl;
                std::cout << "alignedEDebug: " << VectorUintToHexString(alignedEDebug) << std::endl;
            }
        }
    };

    // Process orbit
    processOne(orbInputs, orbOutputs);

    if constexpr (HpShark::DebugChecksums) {
        const auto &debugResultState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2XY, uint64_t>(
                debugStates, orbOutputs.result->data(), numActualDigitsPlusGuard);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "extResult_D_E checksum: " << debugResultState.GetStr() << std::endl;
            std::cout << "extResult_D_E after arithmetic: " << VectorUintToHexString(*orbOutputs.result)
                      << std::endl;
        }
    }

    // Process NR
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        processOne(nrInputs, nrOutputs);
    }
}

// 5) helper to do |±X ±Y ±Z| in one pass, given a fixed X_gtY
static uint64_t
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

// Executes the first phase of the three-term addition/subtraction (A - B + C).
// Processes orbit and optionally NR data in a single call.
template <class SharkFloatParams>
static void
Phase1_ABC(int32_t numActualDigitsPlusGuard,
           int32_t actualDigits,
           int32_t bias,
           ABCInputs orbInputs,
           ABCOutputs &orbOutputs,
           ABCInputs nrInputs,
           ABCOutputs &nrOutputs,
           std::vector<DebugStateHost<SharkFloatParams>> &debugStates)
{
    auto processOne = [&](const ABCInputs &in, ABCOutputs &out) {
        // 1) prepare the two output arrays and signs
        out.resultTrue->assign(numActualDigitsPlusGuard, 0);
        out.resultFalse->assign(numActualDigitsPlusGuard, 0);
        out.outSignTrue = false;
        out.outSignFalse = false;

        // 2) pick the "base" exponent from the largest input
        int32_t baseExp;
        switch (in.ordering) {
            case ThreeWayLargestOrdering::A_GT_AllOthers:
                baseExp = in.effExpA;
                break;
            case ThreeWayLargestOrdering::B_GT_AllOthers:
                baseExp = in.effExpB;
                break;
            case ThreeWayLargestOrdering::C_GT_AllOthers:
                baseExp = in.effExpC;
                break;
            default:
                assert(false);
                for (;;)
                    ;
        }

        // 3) single diff per input to align to baseExp
        int32_t diffA = baseExp - in.effExpA;
        int32_t diffB = baseExp - in.effExpB;
        int32_t diffC = baseExp - in.effExpC;

        // 4) pick pointers, signs, shifts and diffs in "X, Y, Z" order
        const uint32_t *extX, *extY, *extZ;
        bool sX, sY, sZ;
        int32_t shX, shY, shZ, diffY, diffZ;

        switch (in.ordering) {
            case ThreeWayLargestOrdering::A_GT_AllOthers:
                extX = in.extA;
                sX = in.IsNegA;
                shX = in.shiftA;

                extY = in.extB;
                sY = in.IsNegB;
                shY = in.shiftB;
                diffY = diffB;

                extZ = in.extC;
                sZ = in.IsNegC;
                shZ = in.shiftC;
                diffZ = diffC;
                break;

            case ThreeWayLargestOrdering::B_GT_AllOthers:
                extX = in.extB;
                sX = in.IsNegB;
                shX = in.shiftB;

                extY = in.extA;
                sY = in.IsNegA;
                shY = in.shiftA;
                diffY = diffA;

                extZ = in.extC;
                sZ = in.IsNegC;
                shZ = in.shiftC;
                diffZ = diffC;
                break;

            case ThreeWayLargestOrdering::C_GT_AllOthers:
                extX = in.extC;
                sX = in.IsNegC;
                shX = in.shiftC;

                extY = in.extA;
                sY = in.IsNegA;
                shY = in.shiftA;
                diffY = diffA;

                extZ = in.extB;
                sZ = in.IsNegB;
                shZ = in.shiftB;
                diffZ = diffB;
                break;

            default:
                assert(false);
                for (;;)
                    ;
        }

        // 6) single pass: two calls per digit
        for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
            uint64_t Xi = GetNormalizedDigit(extX, actualDigits, numActualDigitsPlusGuard, shX, i);
            uint64_t Yi = GetShiftedNormalizedDigit<SharkFloatParams>(
                extY, actualDigits, numActualDigitsPlusGuard, shY, diffY, i);
            uint64_t Zi = GetShiftedNormalizedDigit<SharkFloatParams>(
                extZ, actualDigits, numActualDigitsPlusGuard, shZ, diffZ, i);

            // always-true branch
            (*out.resultTrue)[i] = CoreThreeWayAdd(Xi, sX, Yi, sY, Zi, sZ, /*X_gtY=*/true, out.outSignTrue);
            // always-false branch
            (*out.resultFalse)[i] = CoreThreeWayAdd(Xi, sX, Yi, sY, Zi, sZ, /*X_gtY=*/false, out.outSignFalse);
        }

        // 7) both exponents (before re-bias) are just baseExp - bias
        out.outExpTrue = baseExp - bias;
        out.outExpFalse = baseExp - bias;
    };

    // Process orbit
    processOne(orbInputs, orbOutputs);

    // Debug printing
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Phase1_ABC - These are effectively the arrays we're adding and subtracting:\n";
        std::cout << "extResultTrue: " << VectorUintToHexString(*orbOutputs.resultTrue) << "\n";
        std::cout << "extResultFalse: " << VectorUintToHexString(*orbOutputs.resultFalse) << "\n";
    }

    if constexpr (HpShark::DebugChecksums) {
        const auto &debugResultState1 =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm1, uint64_t>(
                debugStates, orbOutputs.resultTrue->data(), numActualDigitsPlusGuard);
        const auto &debugResultState2 =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Z2_Perm2, uint64_t>(
                debugStates, orbOutputs.resultFalse->data(), numActualDigitsPlusGuard);
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Phase1_ABC Z2_Perm1 checksum: " << debugResultState1.GetStr() << "\n";
            std::cout << "Phase1_ABC Z2_Perm2 checksum: " << debugResultState2.GetStr() << "\n";
        }
    }

    // Process NR
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        processOne(nrInputs, nrOutputs);
    }
}

template <class SharkFloatParams>
static void
NormalizeAndCopyResult_Single(const char *prefixOutStr,
                              int32_t actualDigits,
                              int32_t numActualDigitsPlusGuard,
                              int32_t &exponent,
                              int32_t &carry,
                              std::vector<uint64_t> &propagatedResult,
                              HpSharkFloat<SharkFloatParams> *ResultOut,
                              bool outSign) noexcept
{
    // --- 1) Inject any carry/borrow back into the digit stream ---
    if (carry < 0) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr << " negative carry (" << carry << "), skipping branch\n";
        }
        assert(false);
        return;
    }

    assert(propagatedResult.size() == numActualDigitsPlusGuard);
    propagatedResult.push_back(static_cast<uint32_t>(carry));
    numActualDigitsPlusGuard++;

    // --- 2) Locate most‐significant non‐zero word ---
    int32_t msdResult = 0;
    for (int32_t i = numActualDigitsPlusGuard - 1; i >= 0; --i) {
        if (propagatedResult[i] != 0) {
            msdResult = i;
            break;
        }
    }

    // --- 3) Compute current vs desired bit positions ---
    int32_t clzResult = CountLeadingZeros(static_cast<uint32_t>(propagatedResult[msdResult]));
    int32_t currentBit = msdResult * 32 + (31 - clzResult);
    int32_t desiredBit = (SharkFloatParams::GlobalNumUint32 - 1) * 32 + 31;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << prefixOutStr << " CLZ of word[" << msdResult << "] = 0x" << std::hex << clzResult
                  << "\n"
                  << prefixOutStr << " currentBit = 0x" << currentBit << "\n"
                  << prefixOutStr << " desiredBit = 0x" << desiredBit << std::dec << "\n";
    }

    // --- 4) Normalize by shifting left or right ---
    int32_t shiftNeeded = currentBit - desiredBit;
    if (shiftNeeded > 0) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr << " right-shifting by " << shiftNeeded << "\n";
        }
        MultiWordShift<Dir::Right>(propagatedResult.data(),
                                   numActualDigitsPlusGuard,
                                   shiftNeeded,
                                   ResultOut->Digits,
                                   actualDigits);
        exponent += shiftNeeded;

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr
                      << " after right shift: " << VectorUintToHexString(ResultOut->Digits, actualDigits)
                      << "\n"
                      << prefixOutStr << " final exponent = 0x" << std::hex << exponent << std::dec
                      << "\n";
        }
    } else if (shiftNeeded < 0) {
        int32_t L = -shiftNeeded;
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr << " left-shifting by " << L << "\n";
        }
        MultiWordShift<Dir::Left>(
            propagatedResult.data(), numActualDigitsPlusGuard, L, ResultOut->Digits, actualDigits);
        exponent -= L;

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr
                      << " after left shift: " << VectorUintToHexString(ResultOut->Digits, actualDigits)
                      << "\n"
                      << prefixOutStr << " final exponent = 0x" << std::hex << exponent << std::dec
                      << "\n";
        }
    } else {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << prefixOutStr << " no normalization shift needed\n";
        }

        assert(propagatedResult.size() == static_cast<size_t>(numActualDigitsPlusGuard));
        assert(sizeof(ResultOut->Digits) / sizeof(ResultOut->Digits[0]) ==
               static_cast<size_t>(actualDigits));

        //
        // Don't just memcpy because the input digits are 64-bit
        // and the output digits are 32-bit.
        //

        MultiWordShift<Dir::Left>(
            propagatedResult.data(), numActualDigitsPlusGuard, 0, ResultOut->Digits, actualDigits);
    }

    // --- 5) Set final exponent and sign ---
    ResultOut->Exponent = exponent;
    ResultOut->SetNegative(outSign);
}

// Wrapper: normalizes orbit result and optionally NR result in one call.
template <class SharkFloatParams>
static void
NormalizeAndCopyResult(const char *prefixOutStr,
                       int32_t actualDigits,
                       int32_t numActualDigitsPlusGuard,
                       int32_t &exponent,
                       int32_t &carry,
                       std::vector<uint64_t> &propagatedResult,
                       HpSharkFloat<SharkFloatParams> *ResultOut,
                       bool outSign,
                       const char *nrPrefixOutStr,
                       int32_t *nrExponent,
                       int32_t *nrCarry,
                       std::vector<uint64_t> *nrPropagatedResult,
                       HpSharkFloat<SharkFloatParams> *nrResultOut,
                       bool nrOutSign) noexcept
{
    NormalizeAndCopyResult_Single<SharkFloatParams>(
        prefixOutStr, actualDigits, numActualDigitsPlusGuard,
        exponent, carry, propagatedResult, ResultOut, outSign);

    if (nrResultOut) {
        NormalizeAndCopyResult_Single<SharkFloatParams>(
            nrPrefixOutStr, actualDigits, numActualDigitsPlusGuard,
            *nrExponent, *nrCarry, *nrPropagatedResult, nrResultOut, nrOutSign);
    }
}

//
// Ternary operator that calculates:
// OutXY1 = A_X2 - B_Y2 + C_A
// ResultOut = D_2X + E_B
//
// Extended arithmetic using little-endian representation.
// This version uses the new normalization approach, where the extended operands
// are not copied; instead, a shift index is returned and used later to compute
// normalized digits on the fly.
//
template <class SharkFloatParams>
void
AddHelper(const HpSharkFloat<SharkFloatParams> *A_X2,
          const HpSharkFloat<SharkFloatParams> *B_Y2,
          const HpSharkFloat<SharkFloatParams> *C_A,
          const HpSharkFloat<SharkFloatParams> *D_2X,
          const HpSharkFloat<SharkFloatParams> *E_B,
          HpSharkFloat<SharkFloatParams> *OutXY1,
          HpSharkFloat<SharkFloatParams> *OutXY2,
          const HpSharkFloat<SharkFloatParams> *W0,
          const HpSharkFloat<SharkFloatParams> *W1,
          const HpSharkFloat<SharkFloatParams> *W2,
          const HpSharkFloat<SharkFloatParams> *W3,
          HpSharkFloat<SharkFloatParams> *OutDzdcReal,
          HpSharkFloat<SharkFloatParams> *OutDzdcImag,
          DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    auto &debugStates = debugHostCombo.States;

    if constexpr (HpShark::DebugChecksums) {
        constexpr auto NewDebugStateSize = static_cast<int>(DebugStatePurpose::NumPurposes);
        debugStates.resize(NewDebugStateSize);
    }

    // Refer to incoming digit arrays.
    const auto *ext_A_X2 = A_X2->Digits;
    const auto *ext_B_Y2 = B_Y2->Digits;
    const auto *ext_C_A = C_A->Digits;
    const auto *ext_D_2X = D_2X->Digits;
    const auto *ext_E_B = E_B->Digits;

    const bool IsNegativeA = A_X2->GetNegative();
    const bool IsNegativeB = !B_Y2->GetNegative(); // A - B + C
    const bool IsNegativeC = C_A->GetNegative();
    const bool IsNegativeD = D_2X->GetNegative();
    const bool IsNegativeE = E_B->GetNegative();

    // --- NR Derivative Setup ---
    constexpr bool NR = SharkFloatParams::EnableNewtonRaphson;
    HpSharkFloat<SharkFloatParams> nrOne{};
    const uint32_t *extW0 = nullptr;
    const uint32_t *extW1 = nullptr;
    const uint32_t *extOne = nullptr;
    const uint32_t *extW2 = nullptr;
    const uint32_t *extW3 = nullptr;
    bool IsNegativeNR_A = false;
    bool IsNegativeNR_B = false;
    bool IsNegativeNR_C = false;
    bool IsNegativeNR_D = false;
    bool IsNegativeNR_E = false;

    if constexpr (NR) {
        nrOne.template FromHDRFloat<typename SharkFloatParams::SubType>(
            HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});

        extW0 = W0->Digits;
        extW1 = W1->Digits;
        extOne = nrOne.Digits;
        extW2 = W2->Digits;
        extW3 = W3->Digits;

        IsNegativeNR_A = W0->GetNegative();
        IsNegativeNR_B = !W1->GetNegative(); // W0 - W1 + One
        IsNegativeNR_C = nrOne.GetNegative();
        IsNegativeNR_D = W2->GetNegative();
        IsNegativeNR_E = W3->GetNegative();
    }

    // --- Set up extended working precision ---
    constexpr int32_t guard = SharkFloatParams::Guard;
    constexpr int32_t numActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t numActualDigitsPlusGuard = SharkFloatParams::GlobalNumUint32 + guard;

    // Create extended arrays (little-endian, index 0 is LSB).
    std::vector<uint64_t> extResultTrue(numActualDigitsPlusGuard, 0);
    std::vector<uint64_t> extResultFalse(numActualDigitsPlusGuard, 0);
    std::vector<uint64_t> extResult_D_E(numActualDigitsPlusGuard, 0);

    std::vector<uint64_t> propagatedResultTrue(numActualDigitsPlusGuard, 0);
    std::vector<uint64_t> propagatedResultFalse(numActualDigitsPlusGuard, 0);
    std::vector<uint64_t> propagatedResult_DE(numActualDigitsPlusGuard, 0);

    // NR result arrays (only allocated when NR enabled)
    std::vector<uint64_t> extNrResultTrue;
    std::vector<uint64_t> extNrResultFalse;
    std::vector<uint64_t> extNrResult_DE;
    std::vector<uint64_t> propagatedNrResultTrue;
    std::vector<uint64_t> propagatedNrResultFalse;
    std::vector<uint64_t> propagatedNrResult_DE;

    if constexpr (NR) {
        extNrResultTrue.resize(numActualDigitsPlusGuard, 0);
        extNrResultFalse.resize(numActualDigitsPlusGuard, 0);
        extNrResult_DE.resize(numActualDigitsPlusGuard, 0);
        propagatedNrResultTrue.resize(numActualDigitsPlusGuard, 0);
        propagatedNrResultFalse.resize(numActualDigitsPlusGuard, 0);
        propagatedNrResult_DE.resize(numActualDigitsPlusGuard, 0);
    }

    // The guard words(indices GlobalNumUint32 to numActualDigitsPlusGuard-1) are left as zero.

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "ext_A_X2: " << VectorUintToHexString(ext_A_X2, numActualDigits) << std::endl;
        std::cout << "ext_A_X2 exponent: " << A_X2->Exponent << std::endl;
        std::cout << "ext_A_X2 sign: " << (IsNegativeA ? "-" : "+") << std::endl;

        std::cout << "ext_B_Y2: " << VectorUintToHexString(ext_B_Y2, numActualDigits) << std::endl;
        std::cout << "ext_B_Y2 exponent: " << B_Y2->Exponent << std::endl;
        std::cout << "ext_B_Y2 sign: " << (IsNegativeB ? "-" : "+") << std::endl;

        std::cout << "ext_C_A: " << VectorUintToHexString(ext_C_A, numActualDigits) << std::endl;
        std::cout << "ext_C_A exponent: " << C_A->Exponent << std::endl;
        std::cout << "ext_C_A sign: " << (IsNegativeC ? "-" : "+") << std::endl;

        std::cout << "ext_D_2X: " << VectorUintToHexString(ext_D_2X, numActualDigits) << std::endl;
        std::cout << "ext_D_2X exponent: " << D_2X->Exponent << std::endl;
        std::cout << "ext_D_2X sign: " << (IsNegativeD ? "-" : "+") << std::endl;

        std::cout << "ext_E_B: " << VectorUintToHexString(ext_E_B, numActualDigits) << std::endl;
        std::cout << "ext_E_B exponent: " << E_B->Exponent << std::endl;
        std::cout << "ext_E_B sign: " << (IsNegativeE ? "-" : "+") << std::endl;
    }

    if constexpr (HpShark::DebugChecksums) {
        // Compute checksums for the extended arrays.
        // Note: we use the actual digits (not the extended size) for the checksum.

        const auto &debugState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits, uint32_t>(
                debugStates, ext_A_X2, numActualDigits);

        const auto &debugBState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits, uint32_t>(
                debugStates, ext_B_Y2, numActualDigits);

        const auto &debugCState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::CDigits, uint32_t>(
                debugStates, ext_C_A, numActualDigits);

        const auto &debugDState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::DDigits, uint32_t>(
                debugStates, ext_D_2X, numActualDigits);

        const auto &debugEState =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::EDigits, uint32_t>(
                debugStates, ext_E_B, numActualDigits);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "A_X2->Digits checksum: " << debugState.GetStr() << std::endl;
            std::cout << "B_Y2->Digits checksum: " << debugBState.GetStr() << std::endl;
            std::cout << "C_A->Digits checksum: " << debugCState.GetStr() << std::endl;
            std::cout << "D_2X->Digits checksum: " << debugDState.GetStr() << std::endl;
            std::cout << "E_B->Digits checksum: " << debugEState.GetStr() << std::endl;
        }
    }

    // --- Extended Normalization using shift indices ---
    bool normA_isZero = false;
    bool normB_isZero = false;
    bool normC_isZero = false;
    bool normD_isZero = false;
    bool normE_isZero = false;

    int32_t newAExponent = A_X2->Exponent;
    int32_t newBExponent = B_Y2->Exponent;
    int32_t newCExponent = C_A->Exponent;
    int32_t newDExponent = D_2X->Exponent;
    int32_t newEExponent = E_B->Exponent;

    // Normalize the extended operands.
    const int32_t shiftALeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_A_X2, numActualDigits, numActualDigitsPlusGuard, newAExponent, normA_isZero);

    const int32_t shiftBLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_B_Y2, numActualDigits, numActualDigitsPlusGuard, newBExponent, normB_isZero);

    const int32_t shiftCLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_C_A, numActualDigits, numActualDigitsPlusGuard, newCExponent, normC_isZero);

    const int32_t shiftDLeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_D_2X, numActualDigits, numActualDigitsPlusGuard, newDExponent, normD_isZero);

    const int32_t shiftELeftToGetMsb = ExtendedNormalizeShiftIndex(
        ext_E_B, numActualDigits, numActualDigitsPlusGuard, newEExponent, normE_isZero);

    // NR normalization
    bool normW0_isZero = false;
    bool normW1_isZero = false;
    bool normOne_isZero = false;
    bool normW2_isZero = false;
    bool normW3_isZero = false;
    int32_t newW0Exponent = 0;
    int32_t newW1Exponent = 0;
    int32_t newOneExponent = 0;
    int32_t newW2Exponent = 0;
    int32_t newW3Exponent = 0;
    int32_t shiftW0LeftToGetMsb = 0;
    int32_t shiftW1LeftToGetMsb = 0;
    int32_t shiftOneLeftToGetMsb = 0;
    int32_t shiftW2LeftToGetMsb = 0;
    int32_t shiftW3LeftToGetMsb = 0;

    if constexpr (NR) {
        newW0Exponent = W0->Exponent;
        newW1Exponent = W1->Exponent;
        newOneExponent = nrOne.Exponent;
        newW2Exponent = W2->Exponent;
        newW3Exponent = W3->Exponent;

        shiftW0LeftToGetMsb = ExtendedNormalizeShiftIndex(
            extW0, numActualDigits, numActualDigitsPlusGuard, newW0Exponent, normW0_isZero);
        shiftW1LeftToGetMsb = ExtendedNormalizeShiftIndex(
            extW1, numActualDigits, numActualDigitsPlusGuard, newW1Exponent, normW1_isZero);
        shiftOneLeftToGetMsb = ExtendedNormalizeShiftIndex(
            extOne, numActualDigits, numActualDigitsPlusGuard, newOneExponent, normOne_isZero);
        shiftW2LeftToGetMsb = ExtendedNormalizeShiftIndex(
            extW2, numActualDigits, numActualDigitsPlusGuard, newW2Exponent, normW2_isZero);
        shiftW3LeftToGetMsb = ExtendedNormalizeShiftIndex(
            extW3, numActualDigits, numActualDigitsPlusGuard, newW3Exponent, normW3_isZero);
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::hex << "shiftALeftToGetMsb: 0x" << shiftALeftToGetMsb << ", newAExponent: 0x"
                  << std::hex << newAExponent << std::endl;
        std::cout << std::hex << "shiftBLeftToGetMsb: 0x" << shiftBLeftToGetMsb << ", newBExponent: 0x"
                  << std::hex << newBExponent << std::endl;
        std::cout << std::hex << "shiftCLeftToGetMsb: 0x" << shiftCLeftToGetMsb << ", newCExponent: 0x"
                  << std::hex << newCExponent << std::endl;
        std::cout << std::hex << "shiftDLeftToGetMsb: 0x" << shiftDLeftToGetMsb << ", newDExponent: 0x"
                  << std::hex << newDExponent << std::endl;
        std::cout << std::hex << "shiftELeftToGetMsb: 0x" << shiftELeftToGetMsb << ", newEExponent: 0x"
                  << std::hex << newEExponent << std::endl;
    }

    // --- Compute Effective Exponents ---
    const auto bias = (SharkFloatParams::GlobalNumUint32 * 32 - 32);
    const int32_t effExpA = normA_isZero ? -100'000'000 : newAExponent + bias;
    const int32_t effExpB = normB_isZero ? -100'000'000 : newBExponent + bias;
    const int32_t effExpC = normC_isZero ? -100'000'000 : newCExponent + bias;
    const int32_t effExpD = normD_isZero ? -100'000'000 : newDExponent + bias;
    const int32_t effExpE = normE_isZero ? -100'000'000 : newEExponent + bias;

    // NR effective exponents
    int32_t effExpW0 = 0;
    int32_t effExpW1 = 0;
    int32_t effExpOne = 0;
    int32_t effExpW2 = 0;
    int32_t effExpW3 = 0;

    if constexpr (NR) {
        effExpW0 = normW0_isZero ? -100'000'000 : newW0Exponent + bias;
        effExpW1 = normW1_isZero ? -100'000'000 : newW1Exponent + bias;
        effExpOne = normOne_isZero ? -100'000'000 : newOneExponent + bias;
        effExpW2 = normW2_isZero ? -100'000'000 : newW2Exponent + bias;
        effExpW3 = normW3_isZero ? -100'000'000 : newW3Exponent + bias;
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "effExpA: " << effExpA << std::endl;
        std::cout << "effExpB: " << effExpB << std::endl;
        std::cout << "effExpC: " << effExpC << std::endl;
        std::cout << "effExpD: " << effExpD << std::endl;
        std::cout << "effExpE: " << effExpE << std::endl;

        // Print the each array normalized according to
        // their resepective effective exponents

        auto PrintOneNormalizedArray = [&](const uint32_t *ext, int32_t shift, const std::string &name) {
            std::cout << name << " normalized: ";
            std::vector<uint32_t> normalizedDigits(numActualDigitsPlusGuard, 0);
            for (int32_t i = 0; i < numActualDigitsPlusGuard; ++i) {
                uint32_t digit =
                    GetNormalizedDigit(ext, numActualDigits, numActualDigitsPlusGuard, shift, i);
                normalizedDigits[i] = digit;
            }

            std::cout << VectorUintToHexString(normalizedDigits.data(), numActualDigitsPlusGuard)
                      << std::endl;
        };

        PrintOneNormalizedArray(ext_A_X2, shiftALeftToGetMsb, "ext_A_X2");
        PrintOneNormalizedArray(ext_B_Y2, shiftBLeftToGetMsb, "ext_B_Y2");
        PrintOneNormalizedArray(ext_C_A, shiftCLeftToGetMsb, "ext_C_A");
        PrintOneNormalizedArray(ext_D_2X, shiftDLeftToGetMsb, "ext_D_2X");
        PrintOneNormalizedArray(ext_E_B, shiftELeftToGetMsb, "ext_E_B");
    }

    // --- Determine which operand has larger magnitude ---

    // Do a 3-way comparison of the other three operands.
    // We need to compare A_X2, B_Y2, and C_A.
    // The result is a 3-way ordering of the three operands.

    // A, B, C:
    ThreeWayLargestOrdering ordering = CompareMagnitudes3Way(effExpA,
                                                             effExpB,
                                                             effExpC,
                                                             numActualDigits,
                                                             numActualDigitsPlusGuard,
                                                             shiftALeftToGetMsb,
                                                             shiftBLeftToGetMsb,
                                                             shiftCLeftToGetMsb,
                                                             ext_A_X2,
                                                             ext_B_Y2,
                                                             ext_C_A);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "ThreeWayLargestOrdering: " << ThreeWayLargestInfo::ToString(ordering) << std::endl;
    }

    // NR 3-way magnitude comparison (W0, W1, One)
    ThreeWayLargestOrdering nrOrdering{};

    if constexpr (NR) {
        nrOrdering = CompareMagnitudes3Way(effExpW0,
                                           effExpW1,
                                           effExpOne,
                                           numActualDigits,
                                           numActualDigitsPlusGuard,
                                           shiftW0LeftToGetMsb,
                                           shiftW1LeftToGetMsb,
                                           shiftOneLeftToGetMsb,
                                           extW0,
                                           extW1,
                                           extOne);
    }

    // --- Phase 1: A - B + C ---
    int32_t outExponentTrue = 0;
    int32_t outExponentFalse = 0;
    bool outSignTrue = false;
    bool outSignFalse = false;
    int32_t nrOutExponentTrue = 0;
    int32_t nrOutExponentFalse = 0;
    bool nrOutSignTrue = false;
    bool nrOutSignFalse = false;

    ABCInputs orbABC{IsNegativeA, IsNegativeB, IsNegativeC,
                     ext_A_X2, ext_B_Y2, ext_C_A,
                     shiftALeftToGetMsb, shiftBLeftToGetMsb, shiftCLeftToGetMsb,
                     effExpA, effExpB, effExpC,
                     ordering};
    ABCOutputs orbABCOut{outSignTrue, outSignFalse,
                         outExponentTrue, outExponentFalse,
                         &extResultTrue, &extResultFalse};

    ABCInputs nrABC{};
    ABCOutputs nrABCOut{false, false, 0, 0, nullptr, nullptr};
    if constexpr (NR) {
        nrABC = {IsNegativeNR_A, IsNegativeNR_B, IsNegativeNR_C,
                 extW0, extW1, extOne,
                 shiftW0LeftToGetMsb, shiftW1LeftToGetMsb, shiftOneLeftToGetMsb,
                 effExpW0, effExpW1, effExpOne,
                 nrOrdering};
        nrABCOut = {nrOutSignTrue, nrOutSignFalse,
                    nrOutExponentTrue, nrOutExponentFalse,
                    &extNrResultTrue, &extNrResultFalse};
    }

    Phase1_ABC<SharkFloatParams>(numActualDigitsPlusGuard,
                                 numActualDigits,
                                 bias,
                                 orbABC,
                                 orbABCOut,
                                 nrABC,
                                 nrABCOut,
                                 debugStates);

    outSignTrue = orbABCOut.outSignTrue;
    outSignFalse = orbABCOut.outSignFalse;
    outExponentTrue = orbABCOut.outExpTrue;
    outExponentFalse = orbABCOut.outExpFalse;
    if constexpr (NR) {
        nrOutSignTrue = nrABCOut.outSignTrue;
        nrOutSignFalse = nrABCOut.outSignFalse;
        nrOutExponentTrue = nrABCOut.outExpTrue;
        nrOutExponentFalse = nrABCOut.outExpFalse;
    }

    // D + E
    // If effective exponents differ, use them. If equal, compare normalized digits on the fly.
    const bool DIsBiggerMagnitude = CompareMagnitudes2Way(effExpD,
                                                          effExpE,
                                                          numActualDigits,
                                                          numActualDigitsPlusGuard,
                                                          shiftDLeftToGetMsb,
                                                          shiftELeftToGetMsb,
                                                          ext_D_2X,
                                                          ext_E_B);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "DIsBiggerMagnitude: " << DIsBiggerMagnitude << std::endl;
    }

    // --- Phase 1: D+E ---
    int32_t outExponent_DE = 0;
    bool nrDIsBiggerMagnitude = false;
    int32_t nrOutExponent_DE = 0;

    // NR 2-way magnitude comparison (W2, W3)
    if constexpr (NR) {
        nrDIsBiggerMagnitude = CompareMagnitudes2Way(effExpW2,
                                                      effExpW3,
                                                      numActualDigits,
                                                      numActualDigitsPlusGuard,
                                                      shiftW2LeftToGetMsb,
                                                      shiftW3LeftToGetMsb,
                                                      extW2,
                                                      extW3);
    }

    const bool sameSignDE = (IsNegativeD == IsNegativeE);

    DEInputs orbDE{DIsBiggerMagnitude, sameSignDE,
                   ext_D_2X, ext_E_B,
                   shiftDLeftToGetMsb, shiftELeftToGetMsb,
                   effExpD, effExpE,
                   newDExponent, newEExponent,
                   IsNegativeD, IsNegativeE};
    DEOutputs orbDEOut{outExponent_DE, &extResult_D_E};

    DEInputs nrDE{};
    DEOutputs nrDEOut{0, nullptr};
    if constexpr (NR) {
        const bool nrSameSign = (IsNegativeNR_D == IsNegativeNR_E);
        nrDE = {nrDIsBiggerMagnitude, nrSameSign,
                extW2, extW3,
                shiftW2LeftToGetMsb, shiftW3LeftToGetMsb,
                effExpW2, effExpW3,
                newW2Exponent, newW3Exponent,
                IsNegativeNR_D, IsNegativeNR_E};
        nrDEOut = {nrOutExponent_DE, &extNrResult_DE};
    }

    Phase1_DE<SharkFloatParams>(numActualDigitsPlusGuard,
                                numActualDigits,
                                bias,
                                orbDE,
                                orbDEOut,
                                nrDE,
                                nrDEOut,
                                debugStates);

    outExponent_DE = orbDEOut.outExponent;
    if constexpr (NR) {
        nrOutExponent_DE = nrDEOut.outExponent;
    }

    // then carry-propagateextResult_ABC into OutXY1->Digits, set OutXY1->Exponent/GetNegative()

    // --- Phase 2: Propagation ---
    // Propagate carries (if addition) or borrows (if subtraction)
    // and store the corrected 32-bit digit into propagatedResult.

    int32_t carryTrue = 0;
    int32_t carryFalse = 0;
    int32_t carry_DE = 0;
    int32_t nrCarryTrue = 0;
    int32_t nrCarryFalse = 0;
    int32_t nrCarry_DE = 0;

    const std::vector<uint64_t> *nrTruePtr = nullptr;
    int32_t *nrCarryTruePtr = nullptr;
    std::vector<uint64_t> *nrPropTruePtr = nullptr;
    const std::vector<uint64_t> *nrFalsePtr = nullptr;
    int32_t *nrCarryFalsePtr = nullptr;
    std::vector<uint64_t> *nrPropFalsePtr = nullptr;
    const std::vector<uint64_t> *nrDEPtr = nullptr;
    int32_t *nrCarryDEPtr = nullptr;
    std::vector<uint64_t> *nrPropDEPtr = nullptr;

    if constexpr (NR) {
        nrTruePtr = &extNrResultTrue;
        nrCarryTruePtr = &nrCarryTrue;
        nrPropTruePtr = &propagatedNrResultTrue;
        nrFalsePtr = &extNrResultFalse;
        nrCarryFalsePtr = &nrCarryFalse;
        nrPropFalsePtr = &propagatedNrResultFalse;
        nrDEPtr = &extNrResult_DE;
        nrCarryDEPtr = &nrCarry_DE;
        nrPropDEPtr = &propagatedNrResult_DE;
    }

    // Result after propagation (orbit + NR in one call each)
    if constexpr (UseBellochPropagation) {

        CarryPropagationPP_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResultTrue, carryTrue, propagatedResultTrue,
            nrTruePtr, nrCarryTruePtr, nrPropTruePtr);

        CarryPropagationPP_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResultFalse, carryFalse, propagatedResultFalse,
            nrFalsePtr, nrCarryFalsePtr, nrPropFalsePtr);

        CarryPropagationPP_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResult_D_E, carry_DE, propagatedResult_DE,
            nrDEPtr, nrCarryDEPtr, nrPropDEPtr);
    } else {
        CarryPropagation_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResultTrue, carryTrue, propagatedResultTrue,
            nrTruePtr, nrCarryTruePtr, nrPropTruePtr);

        CarryPropagation_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResultFalse, carryFalse, propagatedResultFalse,
            nrFalsePtr, nrCarryFalsePtr, nrPropFalsePtr);

        // Formerly CarryPropagation_DE with sameSignDE passed:
        CarryPropagation_ABC<SharkFloatParams>(
            numActualDigitsPlusGuard, extResult_D_E, carry_DE, propagatedResult_DE,
            nrDEPtr, nrCarryDEPtr, nrPropDEPtr);
    }

    // At this point, the propagatedResult_DE array holds the result of the borrow/carry propagation.
    // A subsequent normalization step would adjust these digits (and the exponent) so that the
    // most-significant bit is in the desired position. This normalization step is omitted here.

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "propagatedResultTrue after arithmetic: "
                  << VectorUintToHexString(propagatedResultTrue) << std::endl;
        std::cout << "carryTrue out: 0x" << std::hex << carryTrue << std::endl;

        std::cout << "propagatedResultFalse after arithmetic: "
                  << VectorUintToHexString(propagatedResultFalse) << std::endl;
        std::cout << "carryFalse out: 0x" << std::hex << carryFalse << std::endl;

        std::cout << "propagatedResult_DE after arithmetic: "
                  << VectorUintToHexString(propagatedResult_DE) << std::endl;
        std::cout << "carry_DE out: 0x" << std::hex << carry_DE << std::endl;
    }

    if constexpr (HpShark::DebugChecksums) {
        const auto &debugPropagatedResultTrue =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd1, uint64_t>(
                debugStates, propagatedResultTrue.data(), numActualDigitsPlusGuard);

        const auto &debugPropagatedResultFalse =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd2, uint64_t>(
                debugStates, propagatedResultFalse.data(), numActualDigitsPlusGuard);

        const auto &debugPropagatedResult_DE =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAdd3, uint64_t>(
                debugStates, propagatedResult_DE.data(), numActualDigitsPlusGuard);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "propagatedResultTrue checksum: " << debugPropagatedResultTrue.GetStr()
                      << std::endl;
            std::cout << "propagatedResultFalse checksum: " << debugPropagatedResultFalse.GetStr()
                      << std::endl;
            std::cout << "propagatedResult_DE checksum: " << debugPropagatedResult_DE.GetStr()
                      << std::endl;
        }

        if constexpr (NR) {
            const auto &debugNrTrue =
                GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAddDzdc1, uint64_t>(
                    debugStates, propagatedNrResultTrue.data(), numActualDigitsPlusGuard);
            const auto &debugNrFalse =
                GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAddDzdc2, uint64_t>(
                    debugStates, propagatedNrResultFalse.data(), numActualDigitsPlusGuard);
            const auto &debugNrDE =
                GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::FinalAddDzdc3, uint64_t>(
                    debugStates, propagatedNrResult_DE.data(), numActualDigitsPlusGuard);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "NR propagatedResultTrue checksum: " << debugNrTrue.GetStr() << std::endl;
                std::cout << "NR propagatedResultFalse checksum: " << debugNrFalse.GetStr() << std::endl;
                std::cout << "NR propagatedResult_DE checksum: " << debugNrDE.GetStr() << std::endl;
            }
        }
    }

    // 1) Decide which A-B+C branch to use
    bool useTrueBranch = (carryTrue >= carryFalse);

    // 2) Normalize & write *only* that branch (orbit A-B+C only; NR handled separately)
    if (useTrueBranch) {
        NormalizeAndCopyResult<SharkFloatParams>(
            /* prefixOutStr  = */ "A - B + C: ",
            /* actualDigits  = */ numActualDigits,
            /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
            /* exponent      = */ outExponentTrue,
            /* carry         = */ carryTrue,
            /* propagatedRes = */ propagatedResultTrue,
            /* ResultOut     = */ OutXY1,
            /* outSign       = */ outSignTrue,
            nullptr, nullptr, nullptr, nullptr, nullptr, false);
    } else {
        NormalizeAndCopyResult<SharkFloatParams>(
            /* prefixOutStr  = */ "A - B + C: ",
            /* actualDigits  = */ numActualDigits,
            /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
            /* exponent      = */ outExponentFalse,
            /* carry         = */ carryFalse,
            /* propagatedRes = */ propagatedResultFalse,
            /* ResultOut     = */ OutXY1,
            /* outSign       = */ outSignFalse,
            nullptr, nullptr, nullptr, nullptr, nullptr, false);
    }

    // 3) Handle D+E (orbit + NR combined in one call)
    bool deSign = sameSignDE ? D_2X->GetNegative()
                             : (DIsBiggerMagnitude ? D_2X->GetNegative() : E_B->GetNegative());

    if constexpr (NR) {
        const bool nrSameSignDE = (W2->GetNegative() == W3->GetNegative());
        bool nrDeSign = nrSameSignDE ? W2->GetNegative()
                                     : (nrDIsBiggerMagnitude ? W2->GetNegative() : W3->GetNegative());

        NormalizeAndCopyResult<SharkFloatParams>(
            /* prefixOutStr  = */ "D + E: ",
            /* actualDigits  = */ numActualDigits,
            /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
            /* exponent      = */ outExponent_DE,
            /* carry         = */ carry_DE,
            /* propagatedRes = */ propagatedResult_DE,
            /* ResultOut     = */ OutXY2,
            /* outSign       = */ deSign,
            "NR D + E: ",
            &nrOutExponent_DE,
            &nrCarry_DE,
            &propagatedNrResult_DE,
            OutDzdcImag,
            nrDeSign);
    } else {
        NormalizeAndCopyResult<SharkFloatParams>(
            /* prefixOutStr  = */ "D + E: ",
            /* actualDigits  = */ numActualDigits,
            /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
            /* exponent      = */ outExponent_DE,
            /* carry         = */ carry_DE,
            /* propagatedRes = */ propagatedResult_DE,
            /* ResultOut     = */ OutXY2,
            /* outSign       = */ deSign,
            nullptr, nullptr, nullptr, nullptr, nullptr, false);
    }

    // NR: normalize A-B+C derivative result
    if constexpr (NR) {
        bool nrUseTrueBranch = (nrCarryTrue >= nrCarryFalse);

        if (nrUseTrueBranch) {
            NormalizeAndCopyResult<SharkFloatParams>(
                /* prefixOutStr  = */ "NR A - B + C: ",
                /* actualDigits  = */ numActualDigits,
                /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
                /* exponent      = */ nrOutExponentTrue,
                /* carry         = */ nrCarryTrue,
                /* propagatedRes = */ propagatedNrResultTrue,
                /* ResultOut     = */ OutDzdcReal,
                /* outSign       = */ nrOutSignTrue,
                nullptr, nullptr, nullptr, nullptr, nullptr, false);
        } else {
            NormalizeAndCopyResult<SharkFloatParams>(
                /* prefixOutStr  = */ "NR A - B + C: ",
                /* actualDigits  = */ numActualDigits,
                /* numActualDigitsPlusGuard     = */ numActualDigitsPlusGuard,
                /* exponent      = */ nrOutExponentFalse,
                /* carry         = */ nrCarryFalse,
                /* propagatedRes = */ propagatedNrResultFalse,
                /* ResultOut     = */ OutDzdcReal,
                /* outSign       = */ nrOutSignFalse,
                nullptr, nullptr, nullptr, nullptr, nullptr, false);
        }
    }

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Final Resolution completed" << std::endl;
        std::cout << "Formatted results: " << std::endl;
        std::cout << "OutXY1: " << OutXY1->ToHexString() << std::endl;
        std::cout << "OutXY2: " << OutXY2->ToHexString() << std::endl;
    }

    if constexpr (HpShark::DebugChecksums) {
        const auto &debugResultState_ABC =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add1, uint32_t>(
                debugStates, OutXY1->Digits, SharkFloatParams::GlobalNumUint32);

        const auto &debugResultState_DE =
            GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_Add2, uint32_t>(
                debugStates, OutXY2->Digits, SharkFloatParams::GlobalNumUint32);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "OutXY1->Digits checksum: " << debugResultState_ABC.GetStr() << std::endl;
            std::cout << "OutXY2->Digits checksum: " << debugResultState_DE.GetStr() << std::endl;
        }

        if constexpr (NR) {
            const auto &debugNrResultDzdcReal =
                GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_AddDzdc1, uint32_t>(
                    debugStates, OutDzdcReal->Digits, SharkFloatParams::GlobalNumUint32);
            const auto &debugNrResultDzdcImag =
                GetCurrentDebugState<SharkFloatParams, DebugStatePurpose::Result_AddDzdc2, uint32_t>(
                    debugStates, OutDzdcImag->Digits, SharkFloatParams::GlobalNumUint32);

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "OutDzdcReal->Digits checksum: " << debugNrResultDzdcReal.GetStr()
                          << std::endl;
                std::cout << "OutDzdcImag->Digits checksum: " << debugNrResultDzdcImag.GetStr()
                          << std::endl;
            }
        }
    }
}

//
// Explicit instantiation macro (assumes ExplicitInstantiateAll is defined elsewhere)
//
#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template void AddHelper<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              HpSharkFloat<SharkFloatParams> *,                         \
                                              HpSharkFloat<SharkFloatParams> *,                         \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              const HpSharkFloat<SharkFloatParams> *,                   \
                                              HpSharkFloat<SharkFloatParams> *,                         \
                                              HpSharkFloat<SharkFloatParams> *,                         \
                                              DebugHostCombo<SharkFloatParams> &);

ExplicitInstantiateAll();
