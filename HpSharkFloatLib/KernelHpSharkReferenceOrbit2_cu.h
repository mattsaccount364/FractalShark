#include "Exceptions.h"
#include "KernelHpSharkReferenceOrbit2.h"
#include "LaunchParamsCalculator.h"
#include "MultiplyNTT.cu"

#include <cuda/atomic>
#include <sstream>

namespace Reference2Detail {

#ifdef _DEBUG
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert(bool cond)
{
    if (!cond) {
        //  asm("brkpt;");
        for (;;)
            ;
    }
}
#else
static __device__ SharkForceInlineReleaseOnly void
MattsCudaAssert(bool)
{
    // no-op in release builds
}
#endif

enum class SpectrumId : uint8_t { ZReal, ZImag, DzdcReal, DzdcImag, CReal, CImag, One };
enum class TermKind : uint8_t { Product, Linear };

template <class SharkFloatParams> struct FusedTerm {
    bool IsZero;
    bool IsNegative;
    int32_t Exponent;
    TermKind Kind;
    SpectrumId A;
    SpectrumId B;
};

template <class SharkFloatParams>
__device__ bool
IsLeader(const cooperative_groups::thread_block &block)
{
    return block.group_index().x == 0 && block.thread_index().x == 0;
}

static __device__ uint32_t
GridThreadRank(const cooperative_groups::thread_block &block)
{
    return block.thread_index().x + block.group_index().x * blockDim.x;
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReference2DebugState(DebugState<SharkFloatParams> *debugStates,
                          cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          DebugStatePurpose purpose,
                          const ArrayType *arrayToChecksum,
                          size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose, arrayToChecksum, arraySize);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
StoreReference2DebugValue(DebugState<SharkFloatParams> *debugStates,
                          cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          DebugStatePurpose purpose,
                          const HpSharkFloat<SharkFloatParams> &value)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose, value);
        grid.sync();
    }
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReference2DebugStateBatch(DebugState<SharkFloatParams> *debugStates,
                               cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugStatePurpose purpose0,
                               const ArrayType *array0,
                               DebugStatePurpose purpose1,
                               const ArrayType *array1,
                               size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize);
        grid.sync();
    }
}

template <class SharkFloatParams, class ArrayType>
__device__ void
StoreReference2DebugStateBatch(DebugState<SharkFloatParams> *debugStates,
                               cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugStatePurpose purpose0,
                               const ArrayType *array0,
                               DebugStatePurpose purpose1,
                               const ArrayType *array1,
                               DebugStatePurpose purpose2,
                               const ArrayType *array2,
                               DebugStatePurpose purpose3,
                               const ArrayType *array3,
                               size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose0, array0, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose1, array1, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose2, array2, arraySize);
        StoreCurrentDebugState<SharkFloatParams, ArrayType>(
            debugStates, grid, block, purpose3, array3, arraySize);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
StoreReference2DebugValueBatch(DebugState<SharkFloatParams> *debugStates,
                               cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugStatePurpose purpose0,
                               const HpSharkFloat<SharkFloatParams> &value0,
                               DebugStatePurpose purpose1,
                               const HpSharkFloat<SharkFloatParams> &value1)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose0, value0);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose1, value1);
        grid.sync();
    }
}

template <class SharkFloatParams>
__device__ void
StoreReference2DebugValueBatch(DebugState<SharkFloatParams> *debugStates,
                               cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugStatePurpose purpose0,
                               const HpSharkFloat<SharkFloatParams> &value0,
                               DebugStatePurpose purpose1,
                               const HpSharkFloat<SharkFloatParams> &value1,
                               DebugStatePurpose purpose2,
                               const HpSharkFloat<SharkFloatParams> &value2,
                               DebugStatePurpose purpose3,
                               const HpSharkFloat<SharkFloatParams> &value3)
{
    if constexpr (HpShark::DebugChecksums) {
        grid.sync();
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose0, value0);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose1, value1);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose2, value2);
        StoreCurrentDebugValue<SharkFloatParams>(debugStates, grid, block, purpose3, value3);
        grid.sync();
    }
}

static __device__ uint64_t
CeilPowerOfTwo(uint64_t value)
{
    if (value <= 1)
        return 1;
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return value + 1;
}

static __device__ uint32_t
CountTrailingZeros(uint32_t value)
{
    uint32_t count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
}

static __device__ uint64_t
AddPSerial(uint64_t a, uint64_t b)
{
    const uint64_t sum = a + b;
    return (sum < a || sum >= SharkNTT::MagicPrime) ? sum - SharkNTT::MagicPrime : sum;
}

static __device__ uint64_t
SubPSerial(uint64_t a, uint64_t b)
{
    return (a >= b) ? a - b : a + SharkNTT::MagicPrime - b;
}

template <class SharkFloatParams>
__device__ bool
IsZero(const HpSharkFloat<SharkFloatParams> &value)
{
    // Ref2 finalization keeps every nonzero value normalized to the high bit of the top limb.
    const uint32_t top = value.Digits[SharkFloatParams::GlobalNumUint32 - 1];
    MattsCudaAssert(top == 0u || (top & 0x8000'0000u) != 0u);
    return top == 0u;
}

template <class SharkFloatParams>
__device__ void
SetZeroDigits(cooperative_groups::grid_group &grid,
              cooperative_groups::thread_block &block,
              HpSharkFloat<SharkFloatParams> *output)
{
    constexpr uint32_t DigitCount = SharkFloatParams::GlobalNumUint32;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t digitIndex = GridThreadRank(block); digitIndex < DigitCount; digitIndex += gridSize) {
        output->Digits[digitIndex] = 0u;
    }
}

template <class SharkFloatParams>
__device__ void
SetZeroMetadata(cooperative_groups::thread_block &block, HpSharkFloat<SharkFloatParams> *output)
{
    if (IsLeader<SharkFloatParams>(block)) {
        output->Exponent = -100'000'000;
        output->SetNegative(false);
    }
}

template <class SharkFloatParams>
__device__ void
SetZeroBatch(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             HpSharkFloat<SharkFloatParams> *output0)
{
    SetZeroDigits(grid, block, output0);
    SetZeroMetadata(block, output0);
}

template <class SharkFloatParams>
__device__ void
SetZeroBatch(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             HpSharkFloat<SharkFloatParams> *output0,
             HpSharkFloat<SharkFloatParams> *output1)
{
    SetZeroDigits(grid, block, output0);
    SetZeroDigits(grid, block, output1);
    SetZeroMetadata(block, output0);
    SetZeroMetadata(block, output1);
}

template <class SharkFloatParams>
__device__ void
SetZeroBatch(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             HpSharkFloat<SharkFloatParams> *output0,
             HpSharkFloat<SharkFloatParams> *output1,
             HpSharkFloat<SharkFloatParams> *output2)
{
    SetZeroDigits(grid, block, output0);
    SetZeroDigits(grid, block, output1);
    SetZeroDigits(grid, block, output2);
    SetZeroMetadata(block, output0);
    SetZeroMetadata(block, output1);
    SetZeroMetadata(block, output2);
}

template <class SharkFloatParams>
__device__ void
SetZeroBatch(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             HpSharkFloat<SharkFloatParams> *output0,
             HpSharkFloat<SharkFloatParams> *output1,
             HpSharkFloat<SharkFloatParams> *output2,
             HpSharkFloat<SharkFloatParams> *output3)
{
    SetZeroDigits(grid, block, output0);
    SetZeroDigits(grid, block, output1);
    SetZeroDigits(grid, block, output2);
    SetZeroDigits(grid, block, output3);
    SetZeroMetadata(block, output0);
    SetZeroMetadata(block, output1);
    SetZeroMetadata(block, output2);
    SetZeroMetadata(block, output3);
}

template <class SharkFloatParams>
__device__ void
SetZero(cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        HpSharkFloat<SharkFloatParams> *out)
{
    SetZeroBatch(grid, block, out);
}

template <class SharkFloatParams>
__device__ typename SharkFloatParams::Float
ToNormalizedHDRFloat(const HpSharkFloat<SharkFloatParams> &value)
{
    using SubType = typename SharkFloatParams::SubType;
    using Hdr = typename SharkFloatParams::Float;
    constexpr int TopIndex = SharkFloatParams::GlobalNumUint32 - 1;
    constexpr int MsbInWindow = 63;
    constexpr int32_t MantissaExponent = TopIndex * 32 + 31;

    const uint32_t high = value.Digits[TopIndex];
    if (high == 0u)
        return Hdr{};

    MattsCudaAssert((high & 0x8000'0000u) != 0u);
    const uint32_t low = TopIndex > 0 ? value.Digits[TopIndex - 1] : 0u;
    const uint64_t window = (static_cast<uint64_t>(high) << 32u) | low;
    const int32_t finalExponent = MantissaExponent + value.Exponent;

    if constexpr (std::is_same_v<SubType, CudaDblflt<dblflt>>) {
        double mantissa = static_cast<double>(window) / static_cast<double>(1ull << MsbInWindow);
        if (value.GetNegative())
            mantissa = -mantissa;
        HDRFloat<double> temporary(finalExponent, mantissa);
        HdrReduce(temporary);
        return Hdr{temporary};
    } else {
        SubType mantissa = SubType(window) / std::ldexp(SubType(1), MsbInWindow);
        if (value.GetNegative())
            mantissa = -mantissa;
        Hdr result(finalExponent, mantissa);
        HdrReduce(result);
        return result;
    }
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly FusedTerm<SharkFloatParams>
MakeProductTerm(const HpSharkFloat<SharkFloatParams> &a,
                SpectrumId aId,
                const HpSharkFloat<SharkFloatParams> &b,
                SpectrumId bId,
                bool negate,
                int32_t exponentOffset,
                uint32_t ignoredPrecisionBits)
{
    if (IsZero(a) || IsZero(b))
        return {true, false, 0, TermKind::Product, aId, bId};
    const int64_t exponent = static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(b.Exponent) +
                             static_cast<int64_t>(exponentOffset) +
                             2ll * static_cast<int64_t>(ignoredPrecisionBits);
    MattsCudaAssert(exponent >= INT32_MIN && exponent <= INT32_MAX);
    return {false,
            static_cast<bool>(a.GetNegative() ^ b.GetNegative() ^ negate),
            static_cast<int32_t>(exponent),
            TermKind::Product,
            aId,
            bId};
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly FusedTerm<SharkFloatParams>
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a,
               SpectrumId aId,
               bool negate,
               uint32_t ignoredPrecisionBits)
{
    if (IsZero(a))
        return {true, false, 0, TermKind::Linear, aId, aId};
    const int64_t exponent =
        static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(ignoredPrecisionBits);
    MattsCudaAssert(exponent >= INT32_MIN && exponent <= INT32_MAX);
    return {false,
            static_cast<bool>(a.GetNegative() ^ negate),
            static_cast<int32_t>(exponent),
            TermKind::Linear,
            aId,
            aId};
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly bool
ResolveAlignedValueExponent(int32_t *commonExponent,
                            const HpSharkFloat<SharkFloatParams> &value0,
                            const HpSharkFloat<SharkFloatParams> &value1)
{
    const bool value0Zero = IsZero(value0);
    const bool value1Zero = IsZero(value1);
    if (value0Zero && value1Zero) {
        *commonExponent = 0;
        return true;
    }
    if (value0Zero) {
        *commonExponent = value1.Exponent;
        return false;
    }
    if (value1Zero) {
        *commonExponent = value0.Exponent;
        return false;
    }
    *commonExponent = value0.Exponent < value1.Exponent ? value0.Exponent : value1.Exponent;
    return false;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly FusedTerm<SharkFloatParams>
MakeAlignedProductTerm(bool isZero, int32_t exponent, SpectrumId aId, SpectrumId bId)
{
    return {isZero, false, isZero ? 0 : exponent, TermKind::Product, aId, bId};
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly uint32_t
LinearLimbCount()
{
    return (SharkFloatParams::GlobalNumUint32 * 32u + 31u) / 32u + 2u;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
IncludeTermInCommonExponent(FusedTerm<SharkFloatParams> term, bool &any, int32_t &common)
{
    if (term.IsZero)
        return;
    common = any && common < term.Exponent ? common : term.Exponent;
    any = true;
}

template <class SharkFloatParams, class... RemainingTerms>
__device__ SharkForceInlineReleaseOnly bool
ResolveCommonExponent(int32_t *commonExponent,
                      FusedTerm<SharkFloatParams> firstTerm,
                      RemainingTerms... remainingTerms)
{
    static_assert((std::is_same_v<FusedTerm<SharkFloatParams>, RemainingTerms> && ...));
    bool any = false;
    int32_t common = 0;
    IncludeTermInCommonExponent(firstTerm, any, common);
    (IncludeTermInCommonExponent(remainingTerms, any, common), ...);
    *commonExponent = any ? common : 0;
    return !any;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint64_t
RequiredCoefficientsForTerm(int32_t commonExponent,
                            const SharkNTT::PlanPrime &plan,
                            FusedTerm<SharkFloatParams> term)
{
    if (term.IsZero)
        return 0;
    MattsCudaAssert(plan.b > 0 && plan.L > 0);
    const int64_t signedShift = static_cast<int64_t>(term.Exponent) - commonExponent;
    MattsCudaAssert(signedShift >= 0);
    // The active aligned path below derives support from each packed operand's last coefficient,
    // including the possible high digit introduced by a residual bit shift.
    const uint64_t coefficientShift = static_cast<uint64_t>(signedShift) / static_cast<uint64_t>(plan.b);
    const uint64_t inputCoefficients = static_cast<uint64_t>(plan.L);
    const uint64_t termCoefficients =
        term.Kind == TermKind::Product ? 2ull * inputCoefficients - 1ull : inputCoefficients;
    return coefficientShift + termCoefficients;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
IncludeTermInRequiredCoefficients(int32_t commonExponent,
                                  const SharkNTT::PlanPrime &plan,
                                  FusedTerm<SharkFloatParams> term,
                                  uint64_t &required)
{
    const uint64_t termCoefficients = RequiredCoefficientsForTerm(commonExponent, plan, term);
    required = required > termCoefficients ? required : termCoefficients;
}

template <class SharkFloatParams, class... RemainingTerms>
__device__ SharkForceInlineReleaseOnly uint64_t
RequiredCoefficientsForTerms(int32_t commonExponent,
                             const SharkNTT::PlanPrime &plan,
                             FusedTerm<SharkFloatParams> firstTerm,
                             RemainingTerms... remainingTerms)
{
    static_assert((std::is_same_v<FusedTerm<SharkFloatParams>, RemainingTerms> && ...));
    uint64_t required = RequiredCoefficientsForTerm(commonExponent, plan, firstTerm);
    (IncludeTermInRequiredCoefficients(commonExponent, plan, remainingTerms, required), ...);
    return required;
}

template <class SharkFloatParams>
__device__ uint64_t
ReadBitsSimple(const HpSharkFloat<SharkFloatParams> &value, int64_t bitIndex, int bitCount)
{
    constexpr int TotalBits = SharkFloatParams::GlobalNumUint32 * 32;
    if (bitIndex < 0 || bitIndex >= TotalBits)
        return 0;

    uint64_t result = 0;
    int needed = bitCount;
    int outputBit = 0;
    while (needed > 0 && bitIndex < TotalBits) {
        const int64_t word = bitIndex / 32;
        const int offset = static_cast<int>(bitIndex % 32);
        const uint32_t limb = value.Digits[static_cast<int>(word)];
        const uint32_t chunk = offset == 0 ? limb : limb >> offset;
        const int take = (32 - offset) < needed ? 32 - offset : needed;
        const uint32_t mask = take == 32 ? 0xffffffffu : (1u << take) - 1u;
        result |= static_cast<uint64_t>(chunk & mask) << outputBit;
        outputBit += take;
        needed -= take;
        bitIndex += take;
    }
    return bitCount == 64 ? result : result & ((1ull << bitCount) - 1ull);
}

template <class SharkFloatParams>
__device__ uint64_t
ReadAlignedBits(const HpSharkFloat<SharkFloatParams> &value,
                uint32_t inputBitOffset,
                int64_t sourceBit,
                int bitCount)
{
    constexpr int TotalBits = SharkFloatParams::GlobalNumUint32 * 32;
    const int64_t lowerBit = static_cast<int64_t>(inputBitOffset);
    const int64_t upperBit = static_cast<int64_t>(TotalBits);
    const int64_t sourceEnd = sourceBit + static_cast<int64_t>(bitCount);
    if (sourceEnd <= lowerBit || sourceBit >= upperBit)
        return 0;

    const int64_t readStart = sourceBit < lowerBit ? lowerBit : sourceBit;
    const int64_t readEnd = sourceEnd > upperBit ? upperBit : sourceEnd;
    if (readStart >= readEnd)
        return 0;

    const int leadingZeroBits = static_cast<int>(readStart - sourceBit);
    const int availableBits = static_cast<int>(readEnd - readStart);
    const uint64_t valueBits = ReadBitsSimple(value, readStart, availableBits);
    return valueBits << leadingZeroBits;
}

template <class SharkFloatParams, bool Inverse, bool ForwardDIF = false>
__device__ void
NTTRadix2Batch(uint64_t *sharedData,
               cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *value0,
               uint32_t n,
               uint32_t stages,
               SharkNTT::RootTables &roots)
{
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == n);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stages);
    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::OneWay, Inverse, ForwardDIF>(
        sharedData, grid, block, debugCombo, nullptr, value0, nullptr, nullptr, nullptr, roots);
}

template <class SharkFloatParams, bool Inverse, bool ForwardDIF = false>
__device__ void
NTTRadix2Batch(uint64_t *sharedData,
               cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *value0,
               uint64_t *value1,
               uint32_t n,
               uint32_t stages,
               SharkNTT::RootTables &roots)
{
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == n);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stages);
    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::TwoWay, Inverse, ForwardDIF>(
        sharedData, grid, block, debugCombo, nullptr, value0, value1, nullptr, nullptr, roots);
}

template <class SharkFloatParams, bool Inverse, bool ForwardDIF = false>
__device__ void
NTTRadix2Batch(uint64_t *sharedData,
               cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *value0,
               uint64_t *value1,
               uint64_t *value2,
               uint32_t n,
               uint32_t stages,
               SharkNTT::RootTables &roots)
{
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == n);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stages);
    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::ThreeWay, Inverse, ForwardDIF>(
        sharedData, grid, block, debugCombo, nullptr, value0, value1, value2, nullptr, roots);
}

template <class SharkFloatParams, bool Inverse, bool ForwardDIF = false>
__device__ void
NTTRadix2Batch(uint64_t *sharedData,
               cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *value0,
               uint64_t *value1,
               uint64_t *value2,
               uint64_t *value3,
               uint32_t n,
               uint32_t stages,
               SharkNTT::RootTables &roots)
{
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == n);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stages);
    SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::FourWay, Inverse, ForwardDIF>(
        sharedData, grid, block, debugCombo, nullptr, value0, value1, value2, value3, roots);
}

template <class SharkFloatParams>
__device__ uint64_t
MontgomeryPowSerial(cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t value,
                    uint64_t exponent)
{
    uint64_t result = SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1);
    while (exponent != 0) {
        if ((exponent & 1ull) != 0)
            result = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, result, value);
        value = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, value, value);
        exponent >>= 1;
    }
    return result;
}

template <class SharkFloatParams>
__device__ void
GenerateCachedPlan(cooperative_groups::grid_group &grid,
                   cooperative_groups::thread_block &block,
                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint32_t activeN,
                   HpSharkReference2Workspace<SharkFloatParams> &workspace)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    MattsCudaAssert(activeN >= workspace.ActiveMinFusedN && activeN <= workspace.ActiveMaxFusedN);
    MattsCudaAssert((activeN & (activeN - 1u)) == 0u);
    const uint32_t stages = CountTrailingZeros(activeN);
    MattsCudaAssert(stages >= workspace.ActiveMinFusedStages &&
                    stages <= workspace.ActiveMaxFusedStages);
    const uint32_t slot = stages - Workspace::MinFusedStages;
    MattsCudaAssert(slot < Workspace::PlanCacheEntryCount);
    const uint32_t planBit = 1u << slot;
    if ((workspace.ValidPlanMask & planBit) != 0u)
        return;

    const SharkNTT::PlanPrime &plan = workspace.Plans[slot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[slot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());

    constexpr uint64_t Generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t generatorMont =
        SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, Generator);
    const uint64_t omega = MontgomeryPowSerial<SharkFloatParams>(
        grid, block, debugCombo, generatorMont, SharkNTT::PHI / activeN);
    const uint64_t omegaInverse =
        MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, SharkNTT::PHI - 1ull);

    const uint32_t firstMissingStage = workspace.GeneratedStages + 1u;
    for (uint32_t stage = firstMissingStage; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1u;
        const uint32_t offset = half - 1u;
        if (IsLeader<SharkFloatParams>(block)) {
            roots.stage_omegas[stage - 1u] =
                MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, activeN / width);
            roots.stage_omegas_inv[stage - 1u] = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, omegaInverse, activeN / width);
        }
        grid.sync();
        if (rank < half) {
            uint64_t forwardTwiddle = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas[stage - 1u], rank);
            uint64_t inverseTwiddle = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas_inv[stage - 1u], rank);
            const uint64_t forwardStride = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas[stage - 1u], gridSize);
            const uint64_t inverseStride = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas_inv[stage - 1u], gridSize);
            for (uint32_t index = rank; index < half; index += gridSize) {
                roots.stage_twiddles_fwd[offset + index] = forwardTwiddle;
                roots.stage_twiddles_inv[offset + index] = inverseTwiddle;
                if (index + gridSize < half) {
                    forwardTwiddle = SharkNTT::MontgomeryMul<SharkFloatParams>(
                        grid, block, debugCombo, forwardTwiddle, forwardStride);
                    inverseTwiddle = SharkNTT::MontgomeryMul<SharkFloatParams>(
                        grid, block, debugCombo, inverseTwiddle, inverseStride);
                }
            }
        }
    }

    if (IsLeader<SharkFloatParams>(block)) {
        if (workspace.GeneratedStages < stages)
            workspace.GeneratedStages = stages;
        const uint64_t inverseTwo = SharkNTT::ToMontgomery<SharkFloatParams>(
            grid, block, debugCombo, (SharkNTT::MagicPrime + 1ull) >> 1u);
        roots.Ninvm_mont =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, inverseTwo, stages);
        roots.Ninv =
            SharkNTT::FromMontgomery<SharkFloatParams>(grid, block, debugCombo, roots.Ninvm_mont);
        workspace.ValidPlanMask |= planBit;
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PackForwardOne(cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               const HpSharkFloat<SharkFloatParams> *value,
               const SharkNTT::PlanPrime &plan,
               uint64_t *output,
               uint32_t inputBitOffset)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
        const uint64_t coefficient =
            i < static_cast<uint32_t>(plan.L)
                ? ReadBitsSimple(*value,
                                 static_cast<int64_t>(inputBitOffset) + static_cast<int64_t>(i) * plan.b,
                                 plan.b)
                : 0;
        const uint64_t mont = SharkNTT::ToMontgomery<SharkFloatParams>(
            grid, block, debugCombo, coefficient % SharkNTT::MagicPrime);
        output[i] = mont;
    }
}

static __device__ SharkForceInlineReleaseOnly uint64_t
MultiplyB16ByMontgomeryConstant(uint64_t coefficient, uint64_t scaleR)
{
    MattsCudaAssert(coefficient <= 0xffffull);

    const uint64_t coefficient32 = static_cast<uint32_t>(coefficient);
    const uint64_t lowProduct = coefficient32 * static_cast<uint32_t>(scaleR);
    const uint64_t highProduct = coefficient32 * static_cast<uint32_t>(scaleR >> 32u);
    const uint64_t low = lowProduct + (highProduct << 32u);
    const uint64_t high = (highProduct >> 32u) + (low < lowProduct ? 1ull : 0ull);

    // The input is only 16 bits, so the product is at most 80 bits. Fold the upper
    // word with 2^64 == 2^32 - 1 (mod p), including a possible carry from the fold.
    const uint64_t folded = (high << 32u) - high;
    uint64_t result = low + folded;
    if (result < low) {
        const uint64_t beforeCarryFold = result;
        result += SharkNTT::MontgomeryR;
        if (result < beforeCarryFold)
            result += SharkNTT::MontgomeryR;
    }
    if (result >= SharkNTT::MagicPrime)
        result -= SharkNTT::MagicPrime;
    return result;
}

template <class SharkFloatParams>
__device__ uint64_t
PackAlignedForwardCoefficient(cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              DebugGlobalCount<SharkFloatParams> *debugCombo,
                              const HpSharkFloat<SharkFloatParams> *value,
                              const SharkNTT::PlanPrime &plan,
                              uint32_t outputIndex,
                              uint32_t inputBitOffset,
                              uint32_t coefficientShift,
                              uint32_t residualBitShift,
                              bool negative)
{
    const bool hasCoefficient = outputIndex >= coefficientShift &&
                                outputIndex - coefficientShift <
                                    static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    uint64_t packed = 0ull;
    if (hasCoefficient) {
        const uint32_t inputIndex = outputIndex - coefficientShift;
        const int64_t sourceBit = static_cast<int64_t>(inputBitOffset) +
                                  static_cast<int64_t>(inputIndex) * static_cast<int64_t>(plan.b) -
                                  static_cast<int64_t>(residualBitShift);
        const uint64_t coefficient =
            ReadAlignedBits(*value, inputBitOffset, sourceBit, static_cast<int>(plan.b));
        if (plan.b == 16) {
            packed = MultiplyB16ByMontgomeryConstant(coefficient, SharkNTT::MontgomeryR);
        } else {
            packed = SharkNTT::ToMontgomery<SharkFloatParams>(
                grid, block, debugCombo, coefficient % SharkNTT::MagicPrime);
        }
        if (negative && coefficient != 0)
            packed = SubPSerial(0ull, packed);
    }
    return packed;
}

template <class SharkFloatParams>
__device__ uint32_t
ReadAlignedB16Half(const HpSharkFloat<SharkFloatParams> &value,
                   uint32_t inputBitOffset,
                   int64_t halfIndex,
                   uint32_t coefficientCount)
{
    if (halfIndex < 0 || halfIndex >= static_cast<int64_t>(coefficientCount))
        return 0u;

    constexpr uint64_t TotalBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    const uint64_t sourceBit =
        static_cast<uint64_t>(inputBitOffset) + static_cast<uint64_t>(halfIndex) * 16ull;
    if ((inputBitOffset & 31u) != 0u || sourceBit + 16ull > TotalBits)
        return static_cast<uint32_t>(ReadAlignedBits(value, inputBitOffset, sourceBit, 16));

    const uint32_t wordIndex = (inputBitOffset >> 5u) + static_cast<uint32_t>(halfIndex >> 1u);
    const uint32_t word = value.Digits[wordIndex];
    return (halfIndex & 1ll) == 0ll ? word & 0xffffu : word >> 16u;
}

template <class SharkFloatParams>
__device__ uint64_t
ReadAlignedB16Coefficient(const HpSharkFloat<SharkFloatParams> &value,
                          uint32_t inputBitOffset,
                          uint32_t inputIndex,
                          uint32_t coefficientCount,
                          uint32_t residualBitShift)
{
    const uint32_t current =
        ReadAlignedB16Half(value, inputBitOffset, static_cast<int64_t>(inputIndex), coefficientCount);
    if (residualBitShift == 0u)
        return current;

    const uint32_t previous = ReadAlignedB16Half(
        value, inputBitOffset, static_cast<int64_t>(inputIndex) - 1ll, coefficientCount);
    return (static_cast<uint64_t>(previous >> (16u - residualBitShift)) |
            (static_cast<uint64_t>(current) << residualBitShift)) &
           0xffffull;
}

template <class SharkFloatParams>
__device__ uint64_t
PackAlignedForwardCoefficientScaled(cooperative_groups::grid_group &grid,
                                    cooperative_groups::thread_block &block,
                                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                                    const HpSharkFloat<SharkFloatParams> *value,
                                    const SharkNTT::PlanPrime &plan,
                                    uint64_t inputScaleR,
                                    uint32_t outputIndex,
                                    uint32_t inputBitOffset,
                                    uint32_t coefficientShift,
                                    uint32_t residualBitShift,
                                    bool negative)
{
    MattsCudaAssert(plan.b == 16);
    const bool hasCoefficient = outputIndex >= coefficientShift &&
                                outputIndex - coefficientShift <
                                    static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    if (!hasCoefficient)
        return 0ull;

    const uint32_t inputIndex = outputIndex - coefficientShift;
    const uint32_t sourceCoefficientCount =
        static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);
    const uint64_t coefficient = ReadAlignedB16Coefficient(
        *value, inputBitOffset, inputIndex, sourceCoefficientCount, residualBitShift);
    uint64_t packed;
    const uint32_t stageCount = static_cast<uint32_t>(plan.stages);
    if ((stageCount & 1u) == 0u) {
        // For even stages the scale is exactly a power of two in the standard domain.
        // The compile-time table validation in NTTConstexprGenerator.h covers every
        // supported even stage, so this remains valid for non-View5 transform lengths.
        const uint32_t shift = 32u - stageCount / 2u;
        packed = coefficient << shift;
    } else {
        packed = MultiplyB16ByMontgomeryConstant(coefficient, inputScaleR);
    }
    if (negative)
        packed = SubPSerial(0ull, packed);
    return packed;
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardDIFPairOne(cooperative_groups::grid_group &grid,
                             cooperative_groups::thread_block &block,
                             DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const HpSharkFloat<SharkFloatParams> *value,
                             uint64_t *output,
                             const SharkNTT::PlanPrime &plan,
                             uint64_t inputScaleR,
                             uint32_t j,
                             uint32_t firstHalfSpan,
                             uint32_t secondHalfSpan,
                             const uint64_t *firstStageTwiddles,
                             const uint64_t *secondStageTwiddles,
                             uint32_t inputBitOffset,
                             uint32_t coefficientShift,
                             uint32_t residualBitShift,
                             bool negative)
{
    const uint32_t index0 = j;
    const uint32_t index1 = j + secondHalfSpan;
    const uint32_t index2 = j + firstHalfSpan;
    const uint32_t index3 = index2 + secondHalfSpan;
    const uint64_t value0 = PackAlignedForwardCoefficientScaled(grid,
                                                                block,
                                                                debugCombo,
                                                                value,
                                                                plan,
                                                                inputScaleR,
                                                                index0,
                                                                inputBitOffset,
                                                                coefficientShift,
                                                                residualBitShift,
                                                                negative);
    const uint64_t value1 = PackAlignedForwardCoefficientScaled(grid,
                                                                block,
                                                                debugCombo,
                                                                value,
                                                                plan,
                                                                inputScaleR,
                                                                index1,
                                                                inputBitOffset,
                                                                coefficientShift,
                                                                residualBitShift,
                                                                negative);
    const uint64_t value2 = PackAlignedForwardCoefficientScaled(grid,
                                                                block,
                                                                debugCombo,
                                                                value,
                                                                plan,
                                                                inputScaleR,
                                                                index2,
                                                                inputBitOffset,
                                                                coefficientShift,
                                                                residualBitShift,
                                                                negative);
    const uint64_t value3 = PackAlignedForwardCoefficientScaled(grid,
                                                                block,
                                                                debugCombo,
                                                                value,
                                                                plan,
                                                                inputScaleR,
                                                                index3,
                                                                inputBitOffset,
                                                                coefficientShift,
                                                                residualBitShift,
                                                                negative);
    SharkNTT::StoreRadix4DIFStagePair<SharkFloatParams>(grid,
                                                        block,
                                                        debugCombo,
                                                        output,
                                                        index0,
                                                        index1,
                                                        index2,
                                                        index3,
                                                        value0,
                                                        value1,
                                                        value2,
                                                        value3,
                                                        firstStageTwiddles[j],
                                                        firstStageTwiddles[j + secondHalfSpan],
                                                        secondStageTwiddles[j]);
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardDIFRadix2One(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               const HpSharkFloat<SharkFloatParams> *value,
                               uint64_t *output,
                               const SharkNTT::PlanPrime &plan,
                               uint64_t inputScaleR,
                               uint32_t j,
                               uint32_t halfSpan,
                               const uint64_t *stageTwiddles,
                               uint32_t inputBitOffset,
                               uint32_t coefficientShift,
                               uint32_t residualBitShift,
                               bool negative)
{
    const uint32_t lowerIndex = j + halfSpan;
    const uint64_t upper = PackAlignedForwardCoefficientScaled(grid,
                                                               block,
                                                               debugCombo,
                                                               value,
                                                               plan,
                                                               inputScaleR,
                                                               j,
                                                               inputBitOffset,
                                                               coefficientShift,
                                                               residualBitShift,
                                                               negative);
    const uint64_t lower = PackAlignedForwardCoefficientScaled(grid,
                                                               block,
                                                               debugCombo,
                                                               value,
                                                               plan,
                                                               inputScaleR,
                                                               lowerIndex,
                                                               inputBitOffset,
                                                               coefficientShift,
                                                               residualBitShift,
                                                               negative);
    SharkNTT::StoreRadix2DIFButterfly<SharkFloatParams>(
        grid, block, debugCombo, output, j, lowerIndex, upper, lower, stageTwiddles[j]);
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardDIFLargeStages(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 const SharkNTT::PlanPrime &plan,
                                 const SharkNTT::RootTables &roots,
                                 uint64_t inputScaleR,
                                 const HpSharkFloat<SharkFloatParams> *value0,
                                 uint64_t *output0,
                                 uint32_t inputBitOffset0,
                                 uint32_t coefficientShift0,
                                 uint32_t residualBitShift0,
                                 bool negative0,
                                 const HpSharkFloat<SharkFloatParams> *value1,
                                 uint64_t *output1,
                                 uint32_t inputBitOffset1,
                                 uint32_t coefficientShift1,
                                 uint32_t residualBitShift1,
                                 bool negative1,
                                 const HpSharkFloat<SharkFloatParams> *value2,
                                 uint64_t *output2,
                                 uint32_t inputBitOffset2,
                                 uint32_t coefficientShift2,
                                 uint32_t residualBitShift2,
                                 bool negative2,
                                 const HpSharkFloat<SharkFloatParams> *value3,
                                 uint64_t *output3,
                                 uint32_t inputBitOffset3,
                                 uint32_t coefficientShift3,
                                 uint32_t residualBitShift3,
                                 bool negative3)
{
    MattsCudaAssert(plan.b == 16);
    MattsCudaAssert(plan.stages > 10);
    const uint32_t firstStageIndex = static_cast<uint32_t>(plan.stages);
    const uint32_t firstHalfSpan = 1u << (firstStageIndex - 1u);
    const uint32_t secondHalfSpan = firstHalfSpan >> 1u;
    const uint64_t *firstStageTwiddles = roots.stage_twiddles_fwd + firstHalfSpan - 1u;
    const uint64_t *secondStageTwiddles = roots.stage_twiddles_fwd + secondHalfSpan - 1u;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = GridThreadRank(block); j < secondHalfSpan; j += gridSize) {
        PackAlignedForwardDIFPairOne(grid,
                                     block,
                                     debugCombo,
                                     value0,
                                     output0,
                                     plan,
                                     inputScaleR,
                                     j,
                                     firstHalfSpan,
                                     secondHalfSpan,
                                     firstStageTwiddles,
                                     secondStageTwiddles,
                                     inputBitOffset0,
                                     coefficientShift0,
                                     residualBitShift0,
                                     negative0);
        PackAlignedForwardDIFPairOne(grid,
                                     block,
                                     debugCombo,
                                     value1,
                                     output1,
                                     plan,
                                     inputScaleR,
                                     j,
                                     firstHalfSpan,
                                     secondHalfSpan,
                                     firstStageTwiddles,
                                     secondStageTwiddles,
                                     inputBitOffset1,
                                     coefficientShift1,
                                     residualBitShift1,
                                     negative1);
        if (value2 != nullptr)
            PackAlignedForwardDIFPairOne(grid,
                                         block,
                                         debugCombo,
                                         value2,
                                         output2,
                                         plan,
                                         inputScaleR,
                                         j,
                                         firstHalfSpan,
                                         secondHalfSpan,
                                         firstStageTwiddles,
                                         secondStageTwiddles,
                                         inputBitOffset2,
                                         coefficientShift2,
                                         residualBitShift2,
                                         negative2);
        if (value3 != nullptr)
            PackAlignedForwardDIFPairOne(grid,
                                         block,
                                         debugCombo,
                                         value3,
                                         output3,
                                         plan,
                                         inputScaleR,
                                         j,
                                         firstHalfSpan,
                                         secondHalfSpan,
                                         firstStageTwiddles,
                                         secondStageTwiddles,
                                         inputBitOffset3,
                                         coefficientShift3,
                                         residualBitShift3,
                                         negative3);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardDIFRadix2(cooperative_groups::grid_group &grid,
                            cooperative_groups::thread_block &block,
                            DebugGlobalCount<SharkFloatParams> *debugCombo,
                            const SharkNTT::PlanPrime &plan,
                            const SharkNTT::RootTables &roots,
                            uint64_t inputScaleR,
                            const HpSharkFloat<SharkFloatParams> *value0,
                            uint64_t *output0,
                            uint32_t inputBitOffset0,
                            uint32_t coefficientShift0,
                            uint32_t residualBitShift0,
                            bool negative0,
                            const HpSharkFloat<SharkFloatParams> *value1,
                            uint64_t *output1,
                            uint32_t inputBitOffset1,
                            uint32_t coefficientShift1,
                            uint32_t residualBitShift1,
                            bool negative1,
                            const HpSharkFloat<SharkFloatParams> *value2,
                            uint64_t *output2,
                            uint32_t inputBitOffset2,
                            uint32_t coefficientShift2,
                            uint32_t residualBitShift2,
                            bool negative2,
                            const HpSharkFloat<SharkFloatParams> *value3,
                            uint64_t *output3,
                            uint32_t inputBitOffset3,
                            uint32_t coefficientShift3,
                            uint32_t residualBitShift3,
                            bool negative3)
{
    MattsCudaAssert(plan.b == 16);
    MattsCudaAssert(plan.stages == 11);
    const uint32_t halfSpan = 1u << (static_cast<uint32_t>(plan.stages) - 1u);
    const uint64_t *stageTwiddles = roots.stage_twiddles_fwd + halfSpan - 1u;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = GridThreadRank(block); j < halfSpan; j += gridSize) {
        PackAlignedForwardDIFRadix2One(grid,
                                       block,
                                       debugCombo,
                                       value0,
                                       output0,
                                       plan,
                                       inputScaleR,
                                       j,
                                       halfSpan,
                                       stageTwiddles,
                                       inputBitOffset0,
                                       coefficientShift0,
                                       residualBitShift0,
                                       negative0);
        PackAlignedForwardDIFRadix2One(grid,
                                       block,
                                       debugCombo,
                                       value1,
                                       output1,
                                       plan,
                                       inputScaleR,
                                       j,
                                       halfSpan,
                                       stageTwiddles,
                                       inputBitOffset1,
                                       coefficientShift1,
                                       residualBitShift1,
                                       negative1);
        if (value2 != nullptr)
            PackAlignedForwardDIFRadix2One(grid,
                                           block,
                                           debugCombo,
                                           value2,
                                           output2,
                                           plan,
                                           inputScaleR,
                                           j,
                                           halfSpan,
                                           stageTwiddles,
                                           inputBitOffset2,
                                           coefficientShift2,
                                           residualBitShift2,
                                           negative2);
        if (value3 != nullptr)
            PackAlignedForwardDIFRadix2One(grid,
                                           block,
                                           debugCombo,
                                           value3,
                                           output3,
                                           plan,
                                           inputScaleR,
                                           j,
                                           halfSpan,
                                           stageTwiddles,
                                           inputBitOffset3,
                                           coefficientShift3,
                                           residualBitShift3,
                                           negative3);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PackAlignedInputsBatch(cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugGlobalCount<SharkFloatParams> *debugCombo,
                       const SharkNTT::PlanPrime &plan,
                       uint64_t inputScaleR,
                       const HpSharkFloat<SharkFloatParams> *value0,
                       uint64_t *output0,
                       uint32_t inputBitOffset0,
                       uint32_t coefficientShift0,
                       uint32_t residualBitShift0,
                       bool negative0,
                       const HpSharkFloat<SharkFloatParams> *value1,
                       uint64_t *output1,
                       uint32_t inputBitOffset1,
                       uint32_t coefficientShift1,
                       uint32_t residualBitShift1,
                       bool negative1)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t outputIndex = rank; outputIndex < activeN; outputIndex += gridSize) {
        output0[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value0,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset0,
                                                                   coefficientShift0,
                                                                   residualBitShift0,
                                                                   negative0);
        output1[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value1,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset1,
                                                                   coefficientShift1,
                                                                   residualBitShift1,
                                                                   negative1);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PackAlignedInputsBatch(cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       DebugGlobalCount<SharkFloatParams> *debugCombo,
                       const SharkNTT::PlanPrime &plan,
                       uint64_t inputScaleR,
                       const HpSharkFloat<SharkFloatParams> *value0,
                       uint64_t *output0,
                       uint32_t inputBitOffset0,
                       uint32_t coefficientShift0,
                       uint32_t residualBitShift0,
                       bool negative0,
                       const HpSharkFloat<SharkFloatParams> *value1,
                       uint64_t *output1,
                       uint32_t inputBitOffset1,
                       uint32_t coefficientShift1,
                       uint32_t residualBitShift1,
                       bool negative1,
                       const HpSharkFloat<SharkFloatParams> *value2,
                       uint64_t *output2,
                       uint32_t inputBitOffset2,
                       uint32_t coefficientShift2,
                       uint32_t residualBitShift2,
                       bool negative2,
                       const HpSharkFloat<SharkFloatParams> *value3,
                       uint64_t *output3,
                       uint32_t inputBitOffset3,
                       uint32_t coefficientShift3,
                       uint32_t residualBitShift3,
                       bool negative3)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t outputIndex = rank; outputIndex < activeN; outputIndex += gridSize) {
        output0[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value0,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset0,
                                                                   coefficientShift0,
                                                                   residualBitShift0,
                                                                   negative0);
        output1[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value1,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset1,
                                                                   coefficientShift1,
                                                                   residualBitShift1,
                                                                   negative1);
        output2[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value2,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset2,
                                                                   coefficientShift2,
                                                                   residualBitShift2,
                                                                   negative2);
        output3[outputIndex] = PackAlignedForwardCoefficientScaled(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   value3,
                                                                   plan,
                                                                   inputScaleR,
                                                                   outputIndex,
                                                                   inputBitOffset3,
                                                                   coefficientShift3,
                                                                   residualBitShift3,
                                                                   negative3);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PackForwardBatch(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 uint64_t *sharedData,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 DebugState<SharkFloatParams> *debugStates,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint32_t inputBitOffset,
                 const HpSharkFloat<SharkFloatParams> *value0,
                 uint64_t *output0,
                 DebugStatePurpose packedPurpose0,
                 DebugStatePurpose forwardPurpose0)
{
    PackForwardOne(grid, block, debugCombo, value0, plan, output0, inputBitOffset);
    grid.sync();
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose0, output0, activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(
        sharedData, grid, block, debugCombo, output0, activeN, plan.stages, roots);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose0, output0, activeN);
}

template <class SharkFloatParams>
__device__ void
PackForwardBatch(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 uint64_t *sharedData,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 DebugState<SharkFloatParams> *debugStates,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint32_t inputBitOffset,
                 const HpSharkFloat<SharkFloatParams> *value0,
                 uint64_t *output0,
                 DebugStatePurpose packedPurpose0,
                 DebugStatePurpose forwardPurpose0,
                 const HpSharkFloat<SharkFloatParams> *value1,
                 uint64_t *output1,
                 DebugStatePurpose packedPurpose1,
                 DebugStatePurpose forwardPurpose1)
{
    PackForwardOne(grid, block, debugCombo, value0, plan, output0, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value1, plan, output1, inputBitOffset);
    grid.sync();
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose1, output1, activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(
        sharedData, grid, block, debugCombo, output0, output1, activeN, plan.stages, roots);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose1, output1, activeN);
}

template <class SharkFloatParams>
__device__ void
PackForwardBatch(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 uint64_t *sharedData,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 DebugState<SharkFloatParams> *debugStates,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint32_t inputBitOffset,
                 const HpSharkFloat<SharkFloatParams> *value0,
                 uint64_t *output0,
                 DebugStatePurpose packedPurpose0,
                 DebugStatePurpose forwardPurpose0,
                 const HpSharkFloat<SharkFloatParams> *value1,
                 uint64_t *output1,
                 DebugStatePurpose packedPurpose1,
                 DebugStatePurpose forwardPurpose1,
                 const HpSharkFloat<SharkFloatParams> *value2,
                 uint64_t *output2,
                 DebugStatePurpose packedPurpose2,
                 DebugStatePurpose forwardPurpose2)
{
    PackForwardOne(grid, block, debugCombo, value0, plan, output0, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value1, plan, output1, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value2, plan, output2, inputBitOffset);
    grid.sync();
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose1, output1, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose2, output2, activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(
        sharedData, grid, block, debugCombo, output0, output1, output2, activeN, plan.stages, roots);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose1, output1, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose2, output2, activeN);
}

template <class SharkFloatParams>
__device__ void
PackForwardBatch(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 uint64_t *sharedData,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 DebugState<SharkFloatParams> *debugStates,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint32_t inputBitOffset,
                 const HpSharkFloat<SharkFloatParams> *value0,
                 uint64_t *output0,
                 DebugStatePurpose packedPurpose0,
                 DebugStatePurpose forwardPurpose0,
                 const HpSharkFloat<SharkFloatParams> *value1,
                 uint64_t *output1,
                 DebugStatePurpose packedPurpose1,
                 DebugStatePurpose forwardPurpose1,
                 const HpSharkFloat<SharkFloatParams> *value2,
                 uint64_t *output2,
                 DebugStatePurpose packedPurpose2,
                 DebugStatePurpose forwardPurpose2,
                 const HpSharkFloat<SharkFloatParams> *value3,
                 uint64_t *output3,
                 DebugStatePurpose packedPurpose3,
                 DebugStatePurpose forwardPurpose3)
{
    PackForwardOne(grid, block, debugCombo, value0, plan, output0, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value1, plan, output1, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value2, plan, output2, inputBitOffset);
    PackForwardOne(grid, block, debugCombo, value3, plan, output3, inputBitOffset);
    grid.sync();
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose1, output1, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose2, output2, activeN);
    StoreReference2DebugState(debugStates, grid, block, packedPurpose3, output3, activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(sharedData,
                                                  grid,
                                                  block,
                                                  debugCombo,
                                                  output0,
                                                  output1,
                                                  output2,
                                                  output3,
                                                  activeN,
                                                  plan.stages,
                                                  roots);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose0, output0, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose1, output1, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose2, output2, activeN);
    StoreReference2DebugState(debugStates, grid, block, forwardPurpose3, output3, activeN);
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardBatch(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        uint64_t *sharedData,
                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                        DebugState<SharkFloatParams> *debugStates,
                        const SharkNTT::PlanPrime &plan,
                        SharkNTT::RootTables &roots,
                        const HpSharkFloat<SharkFloatParams> *value0,
                        uint64_t *output0,
                        uint32_t inputBitOffset0,
                        uint32_t coefficientShift0,
                        uint32_t residualBitShift0,
                        bool negative0,
                        DebugStatePurpose packedPurpose0,
                        DebugStatePurpose forwardPurpose0,
                        const HpSharkFloat<SharkFloatParams> *value1,
                        uint64_t *output1,
                        uint32_t inputBitOffset1,
                        uint32_t coefficientShift1,
                        uint32_t residualBitShift1,
                        bool negative1,
                        DebugStatePurpose packedPurpose1,
                        DebugStatePurpose forwardPurpose1)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t outputIndex = rank; outputIndex < activeN; outputIndex += gridSize) {
        output0[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value0,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset0,
                                                             coefficientShift0,
                                                             residualBitShift0,
                                                             negative0);
        output1[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value1,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset1,
                                                             coefficientShift1,
                                                             residualBitShift1,
                                                             negative1);
    }
    if constexpr (!HpShark::DebugChecksums)
        grid.sync();
    StoreReference2DebugStateBatch<SharkFloatParams>(
        debugStates, grid, block, packedPurpose0, output0, packedPurpose1, output1, activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(
        sharedData, grid, block, debugCombo, output0, output1, activeN, plan.stages, roots);
    StoreReference2DebugStateBatch<SharkFloatParams>(
        debugStates, grid, block, forwardPurpose0, output0, forwardPurpose1, output1, activeN);
}

template <class SharkFloatParams>
__device__ void
PackAlignedForwardBatch(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        uint64_t *sharedData,
                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                        DebugState<SharkFloatParams> *debugStates,
                        const SharkNTT::PlanPrime &plan,
                        SharkNTT::RootTables &roots,
                        const HpSharkFloat<SharkFloatParams> *value0,
                        uint64_t *output0,
                        uint32_t inputBitOffset0,
                        uint32_t coefficientShift0,
                        uint32_t residualBitShift0,
                        bool negative0,
                        DebugStatePurpose packedPurpose0,
                        DebugStatePurpose forwardPurpose0,
                        const HpSharkFloat<SharkFloatParams> *value1,
                        uint64_t *output1,
                        uint32_t inputBitOffset1,
                        uint32_t coefficientShift1,
                        uint32_t residualBitShift1,
                        bool negative1,
                        DebugStatePurpose packedPurpose1,
                        DebugStatePurpose forwardPurpose1,
                        const HpSharkFloat<SharkFloatParams> *value2,
                        uint64_t *output2,
                        uint32_t inputBitOffset2,
                        uint32_t coefficientShift2,
                        uint32_t residualBitShift2,
                        bool negative2,
                        DebugStatePurpose packedPurpose2,
                        DebugStatePurpose forwardPurpose2,
                        const HpSharkFloat<SharkFloatParams> *value3,
                        uint64_t *output3,
                        uint32_t inputBitOffset3,
                        uint32_t coefficientShift3,
                        uint32_t residualBitShift3,
                        bool negative3,
                        DebugStatePurpose packedPurpose3,
                        DebugStatePurpose forwardPurpose3)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t outputIndex = rank; outputIndex < activeN; outputIndex += gridSize) {
        output0[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value0,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset0,
                                                             coefficientShift0,
                                                             residualBitShift0,
                                                             negative0);
        output1[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value1,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset1,
                                                             coefficientShift1,
                                                             residualBitShift1,
                                                             negative1);
        output2[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value2,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset2,
                                                             coefficientShift2,
                                                             residualBitShift2,
                                                             negative2);
        output3[outputIndex] = PackAlignedForwardCoefficient(grid,
                                                             block,
                                                             debugCombo,
                                                             value3,
                                                             plan,
                                                             outputIndex,
                                                             inputBitOffset3,
                                                             coefficientShift3,
                                                             residualBitShift3,
                                                             negative3);
    }
    if constexpr (!HpShark::DebugChecksums)
        grid.sync();
    StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                     grid,
                                                     block,
                                                     packedPurpose0,
                                                     output0,
                                                     packedPurpose1,
                                                     output1,
                                                     packedPurpose2,
                                                     output2,
                                                     packedPurpose3,
                                                     output3,
                                                     activeN);
    NTTRadix2Batch<SharkFloatParams, false, true>(sharedData,
                                                  grid,
                                                  block,
                                                  debugCombo,
                                                  output0,
                                                  output1,
                                                  output2,
                                                  output3,
                                                  activeN,
                                                  plan.stages,
                                                  roots);
    StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                     grid,
                                                     block,
                                                     forwardPurpose0,
                                                     output0,
                                                     forwardPurpose1,
                                                     output1,
                                                     forwardPurpose2,
                                                     output2,
                                                     forwardPurpose3,
                                                     output3,
                                                     activeN);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
AccumulateAlignedOutputSpectra(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               DebugState<SharkFloatParams> *debugStates,
                               const SharkNTT::PlanPrime &plan,
                               HpSharkReference2Workspace<SharkFloatParams> &workspace,
                               bool realProductEnabled,
                               bool imagProductEnabled,
                               bool dzdcP1Enabled,
                               bool dzdcP2Enabled,
                               bool dzdcP3Enabled)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
        uint64_t real = 0ull;
        if (realProductEnabled) {
            const uint64_t sum = AddPSerial(workspace.ZReal[i], workspace.ZImag[i]);
            const uint64_t difference = SubPSerial(workspace.ZReal[i], workspace.ZImag[i]);
            real = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, sum, difference);
        }
        workspace.RealOutput[i] = real;

        uint64_t imag = 0ull;
        if (imagProductEnabled) {
            const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, workspace.ZReal[i], workspace.ZImag[i]);
            imag = AddPSerial(product, product);
        }
        workspace.ImagOutput[i] = imag;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint64_t dzdcReal = 0ull;
            uint64_t dzdcImag = 0ull;
            if (dzdcP1Enabled || dzdcP2Enabled || dzdcP3Enabled) {
                const uint64_t p1 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZReal[i], workspace.DzdcReal[i]);
                const uint64_t p2 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZImag[i], workspace.DzdcImag[i]);
                const uint64_t stateSum = AddPSerial(workspace.ZReal[i], workspace.ZImag[i]);
                const uint64_t derivativeSum = AddPSerial(workspace.DzdcReal[i], workspace.DzdcImag[i]);
                const uint64_t p3 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, stateSum, derivativeSum);
                const uint64_t realDifference = SubPSerial(p1, p2);
                const uint64_t imagDifference = SubPSerial(SubPSerial(p3, p1), p2);
                dzdcReal = AddPSerial(realDifference, realDifference);
                dzdcImag = AddPSerial(imagDifference, imagDifference);
            }
            workspace.DzdcRealOutput[i] = dzdcReal;
            workspace.DzdcImagOutput[i] = dzdcImag;
        }
    }
    if constexpr (!HpShark::DebugChecksums)
        grid.sync();

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                         grid,
                                                         block,
                                                         DebugStatePurpose::Z2_Perm1,
                                                         workspace.RealOutput,
                                                         DebugStatePurpose::Z2_Perm2,
                                                         workspace.ImagOutput,
                                                         DebugStatePurpose::Z2_PermW0,
                                                         workspace.DzdcRealOutput,
                                                         DebugStatePurpose::Z2_PermW1,
                                                         workspace.DzdcImagOutput,
                                                         activeN);
    } else {
        StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                         grid,
                                                         block,
                                                         DebugStatePurpose::Z2_Perm1,
                                                         workspace.RealOutput,
                                                         DebugStatePurpose::Z2_Perm2,
                                                         workspace.ImagOutput,
                                                         activeN);
    }
}

static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleXorUint64Width(unsigned mask, uint64_t value, int laneMask, int width)
{
    const uint32_t low = __shfl_xor_sync(mask, static_cast<uint32_t>(value), laneMask, width);
    const uint32_t high = __shfl_xor_sync(mask, static_cast<uint32_t>(value >> 32), laneMask, width);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUint64Width(unsigned mask, uint64_t value, int sourceLane, int width)
{
    const uint32_t low = __shfl_sync(mask, static_cast<uint32_t>(value), sourceLane, width);
    const uint32_t high = __shfl_sync(mask, static_cast<uint32_t>(value >> 32), sourceLane, width);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

static __device__ SharkForceInlineReleaseOnly void
RegroupWarpButterfly(uint64_t previousUpper,
                     uint64_t previousLower,
                     uint32_t shuffleDistance,
                     bool ownsLowerInput,
                     uint64_t &upper,
                     uint64_t &lower)
{
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    constexpr int SubgroupWidth = 16;
    const uint64_t ownValue = ownsLowerInput ? previousLower : previousUpper;
    const uint64_t partnerCandidate = ownsLowerInput ? previousUpper : previousLower;
    const uint64_t partner = ShuffleXorUint64Width(
        FullWarpMask, partnerCandidate, static_cast<int>(shuffleDistance), SubgroupWidth);
    upper = ownsLowerInput ? partner : ownValue;
    lower = ownsLowerInput ? ownValue : partner;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
LoadWarpStageTwiddle(const uint64_t *twiddles, uint32_t stage, uint32_t subgroupLane)
{
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    constexpr int SubgroupWidth = 16;
    const uint32_t halfSpan = 1u << (stage - 1u);
    const uint32_t j = subgroupLane & (halfSpan - 1u);
    uint64_t twiddle = 0ull;
    if (subgroupLane < halfSpan)
        twiddle = twiddles[halfSpan - 1u + subgroupLane];
    return ShuffleUint64Width(FullWarpMask, twiddle, static_cast<int>(j), SubgroupWidth);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ApplyWarpDIFButterfly(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t twiddle,
                      uint64_t &upper,
                      uint64_t &lower)
{
    const uint64_t originalUpper = upper;
    const uint64_t originalLower = lower;
    upper = SharkNTT::AddP(originalUpper, originalLower);
    lower = SharkNTT::MontgomeryMul<SharkFloatParams>(
        grid, block, debugCombo, SharkNTT::SubP(originalUpper, originalLower), twiddle);
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly void
ApplyWarpDITButterfly(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      uint64_t twiddle,
                      uint64_t &upper,
                      uint64_t &lower)
{
    const uint64_t originalUpper = upper;
    const uint64_t product =
        SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, lower, twiddle);
    upper = SharkNTT::AddP(originalUpper, product);
    lower = SharkNTT::SubP(originalUpper, product);
}

template <class SharkFloatParams, SharkNTT::Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ApplyReference2WarpPointwise(cooperative_groups::grid_group &grid,
                             cooperative_groups::thread_block &block,
                             DebugGlobalCount<SharkFloatParams> *debugCombo,
                             bool realProductEnabled,
                             bool imagProductEnabled,
                             bool dzdcP1Enabled,
                             bool dzdcP2Enabled,
                             bool dzdcP3Enabled,
                             uint64_t &zReal,
                             uint64_t &zImag,
                             uint64_t &dzdcReal,
                             uint64_t &dzdcImag)
{
    const uint64_t zRealInput = zReal;
    const uint64_t zImagInput = zImag;
    uint64_t real = 0ull;
    if (realProductEnabled) {
        const uint64_t sum = AddPSerial(zRealInput, zImagInput);
        const uint64_t difference = SubPSerial(zRealInput, zImagInput);
        real = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, sum, difference);
    }

    uint64_t imag = 0ull;
    if (imagProductEnabled) {
        const uint64_t product =
            SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, zRealInput, zImagInput);
        imag = AddPSerial(product, product);
    }

    if constexpr (Mode == SharkNTT::Multiway::FourWay) {
        const uint64_t dzdcRealInput = dzdcReal;
        const uint64_t dzdcImagInput = dzdcImag;
        uint64_t nextDzdcReal = 0ull;
        uint64_t nextDzdcImag = 0ull;
        if (dzdcP1Enabled || dzdcP2Enabled || dzdcP3Enabled) {
            const uint64_t p1 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, zRealInput, dzdcRealInput);
            const uint64_t p2 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, zImagInput, dzdcImagInput);
            const uint64_t stateSum = AddPSerial(zRealInput, zImagInput);
            const uint64_t derivativeSum = AddPSerial(dzdcRealInput, dzdcImagInput);
            const uint64_t p3 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, stateSum, derivativeSum);
            const uint64_t realDifference = SubPSerial(p1, p2);
            const uint64_t imagDifference = SubPSerial(SubPSerial(p3, p1), p2);
            nextDzdcReal = AddPSerial(realDifference, realDifference);
            nextDzdcImag = AddPSerial(imagDifference, imagDifference);
        }
        dzdcReal = nextDzdcReal;
        dzdcImag = nextDzdcImag;
    }

    zReal = real;
    zImag = imag;
}

template <class SharkFloatParams, SharkNTT::Multiway Mode>
static __device__ SharkForceInlineReleaseOnly void
ProcessReference2WarpLocalCenter(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 uint64_t *sharedDataA,
                                 uint64_t *sharedDataB,
                                 uint64_t *sharedDataC,
                                 uint64_t *sharedDataD,
                                 const uint64_t *forwardTwiddles,
                                 const uint64_t *inverseTwiddles,
                                 uint32_t len,
                                 bool realProductEnabled,
                                 bool imagProductEnabled,
                                 bool dzdcP1Enabled,
                                 bool dzdcP2Enabled,
                                 bool dzdcP3Enabled)
{
    static_assert(Mode == SharkNTT::Multiway::TwoWay || Mode == SharkNTT::Multiway::FourWay);
    constexpr uint32_t WarpSize = 32u;
    constexpr uint32_t SubgroupSize = 16u;
    constexpr uint32_t CoefficientsPerSubgroup = 32u;
    constexpr uint32_t CoefficientsPerWarp = 64u;
    constexpr uint32_t WarpLocalStages = 5u;

    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t laneIndex = threadIndex & (WarpSize - 1u);
    const uint32_t subgroupLane = laneIndex & (SubgroupSize - 1u);
    const uint32_t subgroupIndex = laneIndex / SubgroupSize;
    const uint32_t warpIndex = threadIndex / WarpSize;
    const uint32_t warpsPerBlock = block.size() / WarpSize;

    for (uint32_t warpBase = warpIndex * CoefficientsPerWarp; warpBase < len;
         warpBase += warpsPerBlock * CoefficientsPerWarp) {
        const uint32_t groupBase = warpBase + subgroupIndex * CoefficientsPerSubgroup;
        const uint32_t upperIndex = groupBase + subgroupLane;
        const uint32_t lowerIndex = upperIndex + SubgroupSize;

        uint64_t aUpper = sharedDataA[upperIndex];
        uint64_t aLower = sharedDataA[lowerIndex];
        uint64_t bUpper = sharedDataB[upperIndex];
        uint64_t bLower = sharedDataB[lowerIndex];
        uint64_t cUpper = 0ull;
        uint64_t cLower = 0ull;
        uint64_t dUpper = 0ull;
        uint64_t dLower = 0ull;
        if constexpr (Mode == SharkNTT::Multiway::FourWay) {
            cUpper = sharedDataC[upperIndex];
            cLower = sharedDataC[lowerIndex];
            dUpper = sharedDataD[upperIndex];
            dLower = sharedDataD[lowerIndex];
        }

        uint64_t twiddle = LoadWarpStageTwiddle(forwardTwiddles, WarpLocalStages, subgroupLane);
        ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, aUpper, aLower);
        ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, bUpper, bLower);
        if constexpr (Mode == SharkNTT::Multiway::FourWay) {
            ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, cUpper, cLower);
            ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, dUpper, dLower);
        }

        for (uint32_t stage = WarpLocalStages - 1u; stage > 0u; --stage) {
            const uint32_t shuffleDistance = 1u << (stage - 1u);
            const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
            uint64_t upper = 0ull;
            uint64_t lower = 0ull;

            RegroupWarpButterfly(aUpper, aLower, shuffleDistance, ownsLowerInput, upper, lower);
            aUpper = upper;
            aLower = lower;
            RegroupWarpButterfly(bUpper, bLower, shuffleDistance, ownsLowerInput, upper, lower);
            bUpper = upper;
            bLower = lower;
            if constexpr (Mode == SharkNTT::Multiway::FourWay) {
                RegroupWarpButterfly(cUpper, cLower, shuffleDistance, ownsLowerInput, upper, lower);
                cUpper = upper;
                cLower = lower;
                RegroupWarpButterfly(dUpper, dLower, shuffleDistance, ownsLowerInput, upper, lower);
                dUpper = upper;
                dLower = lower;
            }

            twiddle = LoadWarpStageTwiddle(forwardTwiddles, stage, subgroupLane);
            ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, aUpper, aLower);
            ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, bUpper, bLower);
            if constexpr (Mode == SharkNTT::Multiway::FourWay) {
                ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, cUpper, cLower);
                ApplyWarpDIFButterfly(grid, block, debugCombo, twiddle, dUpper, dLower);
            }
        }

        ApplyReference2WarpPointwise<SharkFloatParams, Mode>(grid,
                                                             block,
                                                             debugCombo,
                                                             realProductEnabled,
                                                             imagProductEnabled,
                                                             dzdcP1Enabled,
                                                             dzdcP2Enabled,
                                                             dzdcP3Enabled,
                                                             aUpper,
                                                             bUpper,
                                                             cUpper,
                                                             dUpper);
        ApplyReference2WarpPointwise<SharkFloatParams, Mode>(grid,
                                                             block,
                                                             debugCombo,
                                                             realProductEnabled,
                                                             imagProductEnabled,
                                                             dzdcP1Enabled,
                                                             dzdcP2Enabled,
                                                             dzdcP3Enabled,
                                                             aLower,
                                                             bLower,
                                                             cLower,
                                                             dLower);

        twiddle = LoadWarpStageTwiddle(inverseTwiddles, 1u, subgroupLane);
        ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, aUpper, aLower);
        ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, bUpper, bLower);
        if constexpr (Mode == SharkNTT::Multiway::FourWay) {
            ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, cUpper, cLower);
            ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, dUpper, dLower);
        }

        for (uint32_t stage = 2u; stage <= WarpLocalStages; ++stage) {
            const uint32_t shuffleDistance = 1u << (stage - 2u);
            const bool ownsLowerInput = (subgroupLane & shuffleDistance) != 0u;
            uint64_t upper = 0ull;
            uint64_t lower = 0ull;

            RegroupWarpButterfly(aUpper, aLower, shuffleDistance, ownsLowerInput, upper, lower);
            aUpper = upper;
            aLower = lower;
            RegroupWarpButterfly(bUpper, bLower, shuffleDistance, ownsLowerInput, upper, lower);
            bUpper = upper;
            bLower = lower;
            if constexpr (Mode == SharkNTT::Multiway::FourWay) {
                RegroupWarpButterfly(cUpper, cLower, shuffleDistance, ownsLowerInput, upper, lower);
                cUpper = upper;
                cLower = lower;
                RegroupWarpButterfly(dUpper, dLower, shuffleDistance, ownsLowerInput, upper, lower);
                dUpper = upper;
                dLower = lower;
            }

            twiddle = LoadWarpStageTwiddle(inverseTwiddles, stage, subgroupLane);
            ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, aUpper, aLower);
            ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, bUpper, bLower);
            if constexpr (Mode == SharkNTT::Multiway::FourWay) {
                ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, cUpper, cLower);
                ApplyWarpDITButterfly(grid, block, debugCombo, twiddle, dUpper, dLower);
            }
        }

        sharedDataA[upperIndex] = aUpper;
        sharedDataA[lowerIndex] = aLower;
        sharedDataB[upperIndex] = bUpper;
        sharedDataB[lowerIndex] = bLower;
        if constexpr (Mode == SharkNTT::Multiway::FourWay) {
            sharedDataC[upperIndex] = cUpper;
            sharedDataC[lowerIndex] = cLower;
            sharedDataD[upperIndex] = dUpper;
            sharedDataD[lowerIndex] = dLower;
        }
    }
}

template <class SharkFloatParams, SharkNTT::Multiway Mode>
__device__ SharkForceInlineReleaseOnly void
FusedAlignedPointwiseTransform(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               uint64_t *sharedData,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               const SharkNTT::PlanPrime &plan,
                               const SharkNTT::RootTables &roots,
                               uint64_t *input0,
                               uint64_t *input1,
                               uint64_t *input2,
                               uint64_t *input3,
                               uint64_t *output0,
                               uint64_t *output1,
                               uint64_t *output2,
                               uint64_t *output3,
                               bool realProductEnabled,
                               bool imagProductEnabled,
                               bool dzdcP1Enabled,
                               bool dzdcP2Enabled,
                               bool dzdcP3Enabled)
{
    static_assert(Mode == SharkNTT::Multiway::TwoWay || Mode == SharkNTT::Multiway::FourWay);
    namespace cg = cooperative_groups;

    constexpr uint32_t TileSizeLog2 = 10u;
    constexpr uint32_t TileSize = 1u << TileSizeLog2;
    constexpr uint32_t MaxCachedStages = 7u;
    constexpr size_t RequiredSharedBytes =
        (4u * TileSize + 2u * (1u << MaxCachedStages)) * sizeof(uint64_t);
    static_assert(RequiredSharedBytes <= HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>());
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t stages = static_cast<uint32_t>(plan.stages);
    const uint32_t smallStageCount = stages < TileSizeLog2 ? stages : TileSizeLog2;
    const uint32_t remainder = activeN & (TileSize - 1u);
    const uint32_t tailLength = remainder == 0u ? TileSize : remainder;
    const uint32_t tailStageCapacity = remainder == 0u ? TileSizeLog2 : CountTrailingZeros(tailLength);
    const uint32_t S1 = smallStageCount < tailStageCapacity ? smallStageCount : tailStageCapacity;
    const uint32_t cachedStages = S1 < MaxCachedStages ? S1 : MaxCachedStages;
    const uint32_t cachedTwiddleCount = cachedStages > 0u ? (1u << cachedStages) - 1u : 0u;

    auto *const sharedDataA = sharedData;
    auto *const sharedDataB = sharedDataA + TileSize;
    auto *const sharedDataC = sharedDataB + TileSize;
    auto *const sharedDataD = sharedDataC + TileSize;
    auto *const forwardTwiddles = sharedDataD + TileSize;
    auto *const inverseTwiddles = forwardTwiddles + (1u << MaxCachedStages);

    const size_t cachedBytes = static_cast<size_t>(cachedTwiddleCount) * sizeof(uint64_t);
    const size_t alignedCachedBytes = (cachedBytes + 15u) & ~static_cast<size_t>(15u);
    if (cachedBytes != 0u) {
        cg::memcpy_async(block,
                         forwardTwiddles,
                         roots.stage_twiddles_fwd,
                         cuda::aligned_size_t<16>(alignedCachedBytes));
        cg::memcpy_async(block,
                         inverseTwiddles,
                         roots.stage_twiddles_inv,
                         cuda::aligned_size_t<16>(alignedCachedBytes));
    }
    cg::wait(block);

    const uint32_t tileCount = (activeN + TileSize - 1u) / TileSize;
    for (uint32_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const bool isLastTile = tile == tileCount - 1u;
        const bool hasNextTile = tile + gridDim.x < tileCount;
        const uint32_t len = isLastTile ? tailLength : TileSize;
        constexpr uint32_t WarpLocalStages = 5u;
        constexpr uint32_t CoefficientsPerWarp = 64u;
        const bool useWarpLocalCenter = S1 > WarpLocalStages &&
                                        (len & (CoefficientsPerWarp - 1u)) == 0u &&
                                        (block.size() & 31u) == 0u;
        SharkNTT::LoadOneTilePhase1SM<SharkFloatParams, Mode>(block,
                                                              sharedDataA,
                                                              sharedDataB,
                                                              sharedDataC,
                                                              sharedDataD,
                                                              input0,
                                                              input1,
                                                              input2,
                                                              input3,
                                                              tile,
                                                              TileSize,
                                                              len);
        cg::wait(block);
        SharkNTT::ProcessLoadedTileDIFInPlace<SharkFloatParams, Mode>(
            block,
            grid,
            debugCombo,
            sharedDataA,
            sharedDataB,
            sharedDataC,
            sharedDataD,
            forwardTwiddles,
            roots.stage_twiddles_fwd,
            len,
            S1,
            cachedStages,
            useWarpLocalCenter ? WarpLocalStages : 0u);
        if (useWarpLocalCenter) {
            ProcessReference2WarpLocalCenter<SharkFloatParams, Mode>(grid,
                                                                     block,
                                                                     debugCombo,
                                                                     sharedDataA,
                                                                     sharedDataB,
                                                                     sharedDataC,
                                                                     sharedDataD,
                                                                     forwardTwiddles,
                                                                     inverseTwiddles,
                                                                     len,
                                                                     realProductEnabled,
                                                                     imagProductEnabled,
                                                                     dzdcP1Enabled,
                                                                     dzdcP2Enabled,
                                                                     dzdcP3Enabled);
            block.sync();

            if (S1 > WarpLocalStages + 1u) {
                SharkNTT::ProcessLoadedTileDITInPlace<SharkFloatParams, Mode>(block,
                                                                              grid,
                                                                              debugCombo,
                                                                              sharedDataA,
                                                                              sharedDataB,
                                                                              sharedDataC,
                                                                              sharedDataD,
                                                                              inverseTwiddles,
                                                                              roots.stage_twiddles_inv,
                                                                              len,
                                                                              WarpLocalStages + 1u,
                                                                              S1 - 1u,
                                                                              cachedStages);
            }

            SharkNTT::ProcessLoadedTileDITFinalStageToGlobal<SharkFloatParams, Mode>(
                block,
                grid,
                debugCombo,
                sharedDataA,
                sharedDataB,
                sharedDataC,
                sharedDataD,
                output0,
                output1,
                output2,
                output3,
                inverseTwiddles,
                roots.stage_twiddles_inv,
                tile * TileSize,
                len,
                S1,
                cachedStages);
        } else {
            const uint32_t rank = block.thread_index().x;
            const uint32_t blockSize = block.size();
            for (uint32_t i = rank; i < len; i += blockSize) {
                const uint64_t zReal = sharedDataA[i];
                const uint64_t zImag = sharedDataB[i];
                uint64_t real = 0ull;
                if (realProductEnabled) {
                    const uint64_t sum = AddPSerial(zReal, zImag);
                    const uint64_t difference = SubPSerial(zReal, zImag);
                    real = SharkNTT::MontgomeryMul<SharkFloatParams>(
                        grid, block, debugCombo, sum, difference);
                }
                sharedDataA[i] = real;

                uint64_t imag = 0ull;
                if (imagProductEnabled) {
                    const uint64_t product =
                        SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, zReal, zImag);
                    imag = AddPSerial(product, product);
                }
                sharedDataB[i] = imag;

                if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                    const uint64_t dzdcRealInput = sharedDataC[i];
                    const uint64_t dzdcImagInput = sharedDataD[i];
                    uint64_t dzdcReal = 0ull;
                    uint64_t dzdcImag = 0ull;
                    if (dzdcP1Enabled || dzdcP2Enabled || dzdcP3Enabled) {
                        const uint64_t p1 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                            grid, block, debugCombo, zReal, dzdcRealInput);
                        const uint64_t p2 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                            grid, block, debugCombo, zImag, dzdcImagInput);
                        const uint64_t stateSum = AddPSerial(zReal, zImag);
                        const uint64_t derivativeSum = AddPSerial(dzdcRealInput, dzdcImagInput);
                        const uint64_t p3 = SharkNTT::MontgomeryMul<SharkFloatParams>(
                            grid, block, debugCombo, stateSum, derivativeSum);
                        const uint64_t realDifference = SubPSerial(p1, p2);
                        const uint64_t imagDifference = SubPSerial(SubPSerial(p3, p1), p2);
                        dzdcReal = AddPSerial(realDifference, realDifference);
                        dzdcImag = AddPSerial(imagDifference, imagDifference);
                    }
                    sharedDataC[i] = dzdcReal;
                    sharedDataD[i] = dzdcImag;
                }
            }
            block.sync();

            SharkNTT::ProcessLoadedTileDITInPlace<SharkFloatParams, Mode>(block,
                                                                          grid,
                                                                          debugCombo,
                                                                          sharedDataA,
                                                                          sharedDataB,
                                                                          sharedDataC,
                                                                          sharedDataD,
                                                                          inverseTwiddles,
                                                                          roots.stage_twiddles_inv,
                                                                          len,
                                                                          1u,
                                                                          S1,
                                                                          cachedStages);
            for (uint32_t i = rank; i < len; i += blockSize) {
                output0[tile * TileSize + i] = sharedDataA[i];
                output1[tile * TileSize + i] = sharedDataB[i];
                if constexpr (Mode == SharkNTT::Multiway::FourWay) {
                    output2[tile * TileSize + i] = sharedDataC[i];
                    output3[tile * TileSize + i] = sharedDataD[i];
                }
            }
        }
        if (hasNextTile)
            block.sync();
    }
    grid.sync();
}

template <class IntT>
__device__ uint32_t
FunnelShiftRight(const IntT *data, int index, int count, int bitOffset)
{
    const int wordOffset = bitOffset / 32;
    const int bit = bitOffset % 32;
    const uint32_t low =
        (index + wordOffset >= count) ? 0u : static_cast<uint32_t>(data[index + wordOffset]);
    if (bit == 0)
        return low;
    const uint32_t high =
        (index + wordOffset + 1 >= count) ? 0u : static_cast<uint32_t>(data[index + wordOffset + 1]);
    return (low >> bit) | (high << (32 - bit));
}

template <class IntT>
__device__ uint32_t
FunnelShiftLeft(const IntT *data, int index, int count, int bitOffset)
{
    const int wordOffset = bitOffset / 32;
    const int bit = bitOffset % 32;
    const uint32_t low = (index - wordOffset < 0) ? 0u : static_cast<uint32_t>(data[index - wordOffset]);
    if (bit == 0)
        return low;
    const uint32_t high =
        (index - wordOffset - 1 < 0) ? 0u : static_cast<uint32_t>(data[index - wordOffset - 1]);
    return (low << bit) | (high >> (32 - bit));
}

static __device__ SharkForceInlineReleaseOnly int64_t
SignedResidueContribution(uint64_t residue,
                          uint64_t coefficientIndex,
                          uint32_t limbIndex,
                          uint32_t bitsPerCoefficient,
                          uint64_t halfPrime,
                          uint64_t outputBitOffset)
{
    if (residue == 0)
        return 0;

    const bool negative = residue > halfPrime;
    const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
    const uint64_t shiftedBits =
        outputBitOffset + coefficientIndex * static_cast<uint64_t>(bitsPerCoefficient);
    const uint32_t q = static_cast<uint32_t>(shiftedBits >> 5);
    if (q > limbIndex || limbIndex - q > 3)
        return 0;

    const int r = static_cast<int>(shiftedBits & 31);
    const uint64_t lo = r == 0 ? magnitude : magnitude << r;
    const uint64_t hi = r == 0 ? 0ull : magnitude >> (64 - r);
    uint32_t contribution = 0;
    switch (limbIndex - q) {
        case 0:
            contribution = static_cast<uint32_t>(lo);
            break;
        case 1:
            contribution = static_cast<uint32_t>(lo >> 32);
            break;
        case 2:
            contribution = static_cast<uint32_t>(hi);
            break;
        case 3:
            contribution = static_cast<uint32_t>(hi >> 32);
            break;
    }
    return negative ? -static_cast<int64_t>(contribution) : static_cast<int64_t>(contribution);
}

template <class SharkFloatParams, bool Normalize>
__device__ void
UnpackResiduesToSignedLimbsScalar(cooperative_groups::grid_group &grid,
                                  cooperative_groups::thread_block &block,
                                  DebugGlobalCount<SharkFloatParams> *debugCombo,
                                  const uint64_t *spectrum,
                                  const SharkNTT::PlanPrime &plan,
                                  const SharkNTT::RootTables &roots,
                                  uint32_t coefficientCount,
                                  int64_t *limbs,
                                  uint32_t limbCount)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = GridThreadRank(block); j < limbCount; j += gridSize) {
        const uint64_t firstBit = j >= 3 ? static_cast<uint64_t>(j - 3) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(j) + 1ull) * 32ull - 1ull;
        const uint64_t firstCoefficient = firstBit / static_cast<uint64_t>(plan.b);
        const uint64_t lastCoefficient = lastBit / static_cast<uint64_t>(plan.b);
        int64_t total = 0;
        for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
            uint64_t residue = spectrum[i];
            if constexpr (Normalize)
                residue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[i], roots.Ninv);
            total +=
                SignedResidueContribution(residue, i, j, static_cast<uint32_t>(plan.b), halfPrime, 0);
        }
        limbs[j] = total;
    }
}

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly int64_t
SignedLinearLimbContribution(const HpSharkFloat<SharkFloatParams> *value,
                             uint32_t inputBitOffset,
                             uint64_t outputBitOffset,
                             uint32_t limbIndex)
{
    if (value == nullptr)
        return 0;

    const uint64_t limbBit = static_cast<uint64_t>(limbIndex) * 32ull;
    if (limbBit + 32ull <= outputBitOffset)
        return 0;

    uint32_t contribution = 0u;
    if (limbBit < outputBitOffset) {
        const uint32_t gap = static_cast<uint32_t>(outputBitOffset - limbBit);
        if (gap < 32u) {
            const uint32_t bitCount = 32u - gap;
            const uint64_t source =
                ReadBitsSimple(*value, static_cast<int64_t>(inputBitOffset), static_cast<int>(bitCount));
            contribution = static_cast<uint32_t>(source << gap);
        }
    } else {
        const uint64_t sourceBit = limbBit - outputBitOffset;
        const uint64_t source =
            ReadBitsSimple(*value, static_cast<int64_t>(inputBitOffset + sourceBit), 32);
        contribution = static_cast<uint32_t>(source);
    }

    const int64_t signedContribution = static_cast<int64_t>(contribution);
    return value->GetNegative() ? -signedContribution : signedContribution;
}

template <class SharkFloatParams, bool Normalize>
__device__ int64_t
UnpackAlignedResidueLimbContribution(cooperative_groups::grid_group &grid,
                                     cooperative_groups::thread_block &block,
                                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                                     const uint64_t *spectrum,
                                     const SharkNTT::PlanPrime &plan,
                                     const SharkNTT::RootTables &roots,
                                     uint32_t coefficientCount,
                                     uint64_t productBitOffset,
                                     const HpSharkFloat<SharkFloatParams> *linearValue,
                                     uint32_t linearInputBitOffset,
                                     uint64_t linearBitOffset,
                                     uint32_t limbIndex)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    const uint64_t firstBit = limbIndex >= 3 ? static_cast<uint64_t>(limbIndex - 3) * 32ull : 0ull;
    const uint64_t lastBit = (static_cast<uint64_t>(limbIndex) + 1ull) * 32ull - 1ull;
    const uint64_t firstCoefficient = firstBit > productBitOffset
                                          ? (firstBit - productBitOffset) / static_cast<uint64_t>(plan.b)
                                          : 0ull;
    const uint64_t lastCoefficient = lastBit >= productBitOffset
                                         ? (lastBit - productBitOffset) / static_cast<uint64_t>(plan.b)
                                         : 0ull;
    int64_t total = 0;
    if (firstBit >= productBitOffset || productBitOffset <= lastBit) {
        for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
            uint64_t residue = spectrum[i];
            if constexpr (Normalize)
                residue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[i], roots.Ninv);
            total += SignedResidueContribution(
                residue, i, limbIndex, static_cast<uint32_t>(plan.b), halfPrime, productBitOffset);
        }
    }
    return total +
           SignedLinearLimbContribution(linearValue, linearInputBitOffset, linearBitOffset, limbIndex);
}

template <class SharkFloatParams>
__device__ void
GatherLinearToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               const HpSharkFloat<SharkFloatParams> *linearValue0,
                               uint32_t linearInputBitOffset0,
                               int64_t *limbs0,
                               const HpSharkFloat<SharkFloatParams> *linearValue1,
                               uint32_t linearInputBitOffset1,
                               int64_t *limbs1,
                               uint32_t limbCount)
{
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = rank; j < limbCount; j += gridSize) {
        limbs0[j] = SignedLinearLimbContribution(linearValue0, linearInputBitOffset0, 0, j);
        limbs1[j] = SignedLinearLimbContribution(linearValue1, linearInputBitOffset1, 0, j);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
GatherLinearToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               const HpSharkFloat<SharkFloatParams> *linearValue0,
                               uint32_t linearInputBitOffset0,
                               int64_t *limbs0,
                               const HpSharkFloat<SharkFloatParams> *linearValue1,
                               uint32_t linearInputBitOffset1,
                               int64_t *limbs1,
                               const HpSharkFloat<SharkFloatParams> *linearValue2,
                               uint32_t linearInputBitOffset2,
                               int64_t *limbs2,
                               const HpSharkFloat<SharkFloatParams> *linearValue3,
                               uint32_t linearInputBitOffset3,
                               int64_t *limbs3,
                               uint32_t limbCount)
{
    const uint32_t rank = GridThreadRank(block);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = rank; j < limbCount; j += gridSize) {
        limbs0[j] = SignedLinearLimbContribution(linearValue0, linearInputBitOffset0, 0, j);
        limbs1[j] = SignedLinearLimbContribution(linearValue1, linearInputBitOffset1, 0, j);
        limbs2[j] = SignedLinearLimbContribution(linearValue2, linearInputBitOffset2, 0, j);
        limbs3[j] = SignedLinearLimbContribution(linearValue3, linearInputBitOffset3, 0, j);
    }
    grid.sync();
}

enum class AlignedUnpackMode : uint8_t { NormalizedMontgomery, StandardResidue };

template <class SharkFloatParams, AlignedUnpackMode Mode>
__device__ void UnpackAlignedResiduesToSignedLimbsB16(cooperative_groups::grid_group &grid,
                                                      cooperative_groups::thread_block &block,
                                                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                                                      const uint64_t *spectrum,
                                                      const SharkNTT::PlanPrime &plan,
                                                      const SharkNTT::RootTables &roots,
                                                      uint32_t coefficientCount,
                                                      uint64_t productBitOffset,
                                                      const HpSharkFloat<SharkFloatParams> *linearValue,
                                                      uint32_t linearInputBitOffset,
                                                      uint64_t linearBitOffset,
                                                      int64_t *limbs,
                                                      uint32_t limbCount);

template <class SharkFloatParams, AlignedUnpackMode Mode>
__device__ void
UnpackAlignedResiduesToSignedLimbsOne(cooperative_groups::grid_group &grid,
                                      cooperative_groups::thread_block &block,
                                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                                      const uint64_t *spectrum,
                                      const SharkNTT::PlanPrime &plan,
                                      const SharkNTT::RootTables &roots,
                                      uint32_t coefficientCount,
                                      uint64_t productBitOffset,
                                      const HpSharkFloat<SharkFloatParams> *linearValue,
                                      uint32_t linearInputBitOffset,
                                      uint64_t linearBitOffset,
                                      int64_t *limbs,
                                      uint32_t limbCount)
{
    if (plan.b == 16) {
        UnpackAlignedResiduesToSignedLimbsB16<SharkFloatParams, Mode>(grid,
                                                                      block,
                                                                      debugCombo,
                                                                      spectrum,
                                                                      plan,
                                                                      roots,
                                                                      coefficientCount,
                                                                      productBitOffset,
                                                                      linearValue,
                                                                      linearInputBitOffset,
                                                                      linearBitOffset,
                                                                      limbs,
                                                                      limbCount);
        return;
    }

    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t limbIndex = GridThreadRank(block); limbIndex < limbCount; limbIndex += gridSize) {
        limbs[limbIndex] =
            UnpackAlignedResidueLimbContribution<SharkFloatParams,
                                                 Mode == AlignedUnpackMode::NormalizedMontgomery>(
                grid,
                block,
                debugCombo,
                spectrum,
                plan,
                roots,
                coefficientCount,
                productBitOffset,
                linearValue,
                linearInputBitOffset,
                linearBitOffset,
                limbIndex);
    }
}

template <class SharkFloatParams, AlignedUnpackMode Mode>
__device__ void
UnpackAlignedResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                        cooperative_groups::thread_block &block,
                                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                                        const SharkNTT::PlanPrime &plan,
                                        const SharkNTT::RootTables &roots,
                                        const uint64_t *spectrum0,
                                        uint32_t coefficientCount0,
                                        uint64_t productBitOffset0,
                                        const HpSharkFloat<SharkFloatParams> *linearValue0,
                                        uint32_t linearInputBitOffset0,
                                        uint64_t linearBitOffset0,
                                        int64_t *limbs0,
                                        const uint64_t *spectrum1,
                                        uint32_t coefficientCount1,
                                        uint64_t productBitOffset1,
                                        const HpSharkFloat<SharkFloatParams> *linearValue1,
                                        uint32_t linearInputBitOffset1,
                                        uint64_t linearBitOffset1,
                                        int64_t *limbs1,
                                        uint32_t limbCount)
{
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum0,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount0,
                                                                  productBitOffset0,
                                                                  linearValue0,
                                                                  linearInputBitOffset0,
                                                                  linearBitOffset0,
                                                                  limbs0,
                                                                  limbCount);
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum1,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount1,
                                                                  productBitOffset1,
                                                                  linearValue1,
                                                                  linearInputBitOffset1,
                                                                  linearBitOffset1,
                                                                  limbs1,
                                                                  limbCount);
}

template <class SharkFloatParams, AlignedUnpackMode Mode>
__device__ void
UnpackAlignedResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                        cooperative_groups::thread_block &block,
                                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                                        const SharkNTT::PlanPrime &plan,
                                        const SharkNTT::RootTables &roots,
                                        const uint64_t *spectrum0,
                                        uint32_t coefficientCount0,
                                        uint64_t productBitOffset0,
                                        const HpSharkFloat<SharkFloatParams> *linearValue0,
                                        uint32_t linearInputBitOffset0,
                                        uint64_t linearBitOffset0,
                                        int64_t *limbs0,
                                        const uint64_t *spectrum1,
                                        uint32_t coefficientCount1,
                                        uint64_t productBitOffset1,
                                        const HpSharkFloat<SharkFloatParams> *linearValue1,
                                        uint32_t linearInputBitOffset1,
                                        uint64_t linearBitOffset1,
                                        int64_t *limbs1,
                                        const uint64_t *spectrum2,
                                        uint32_t coefficientCount2,
                                        uint64_t productBitOffset2,
                                        const HpSharkFloat<SharkFloatParams> *linearValue2,
                                        uint32_t linearInputBitOffset2,
                                        uint64_t linearBitOffset2,
                                        int64_t *limbs2,
                                        const uint64_t *spectrum3,
                                        uint32_t coefficientCount3,
                                        uint64_t productBitOffset3,
                                        const HpSharkFloat<SharkFloatParams> *linearValue3,
                                        uint32_t linearInputBitOffset3,
                                        uint64_t linearBitOffset3,
                                        int64_t *limbs3,
                                        uint32_t limbCount)
{
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum0,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount0,
                                                                  productBitOffset0,
                                                                  linearValue0,
                                                                  linearInputBitOffset0,
                                                                  linearBitOffset0,
                                                                  limbs0,
                                                                  limbCount);
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum1,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount1,
                                                                  productBitOffset1,
                                                                  linearValue1,
                                                                  linearInputBitOffset1,
                                                                  linearBitOffset1,
                                                                  limbs1,
                                                                  limbCount);
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum2,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount2,
                                                                  productBitOffset2,
                                                                  linearValue2,
                                                                  linearInputBitOffset2,
                                                                  linearBitOffset2,
                                                                  limbs2,
                                                                  limbCount);
    UnpackAlignedResiduesToSignedLimbsOne<SharkFloatParams, Mode>(grid,
                                                                  block,
                                                                  debugCombo,
                                                                  spectrum3,
                                                                  plan,
                                                                  roots,
                                                                  coefficientCount3,
                                                                  productBitOffset3,
                                                                  linearValue3,
                                                                  linearInputBitOffset3,
                                                                  linearBitOffset3,
                                                                  limbs3,
                                                                  limbCount);
}

static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUint64(unsigned mask, uint64_t value, int sourceLane)
{
    const uint32_t low = __shfl_sync(mask, static_cast<uint32_t>(value), sourceLane);
    const uint32_t high = __shfl_sync(mask, static_cast<uint32_t>(value >> 32), sourceLane);
    return (static_cast<uint64_t>(high) << 32) | low;
}

static __device__ SharkForceInlineReleaseOnly uint64_t
ShuffleUpUint64(unsigned mask, uint64_t value, unsigned delta)
{
    const uint32_t low = __shfl_up_sync(mask, static_cast<uint32_t>(value), delta);
    const uint32_t high = __shfl_up_sync(mask, static_cast<uint32_t>(value >> 32), delta);
    return (static_cast<uint64_t>(high) << 32) | low;
}

template <class SharkFloatParams, AlignedUnpackMode Mode>
static __device__ SharkForceInlineReleaseOnly uint64_t
NormalizeB16ResidueForTile(cooperative_groups::grid_group &grid,
                           cooperative_groups::thread_block &block,
                           DebugGlobalCount<SharkFloatParams> *debugCombo,
                           const uint64_t *spectrum,
                           const SharkNTT::RootTables &roots,
                           uint32_t coefficientCount,
                           uint64_t firstCoefficient,
                           uint64_t lastCoefficient,
                           int64_t coefficientIndex)
{
    if (coefficientIndex < 0)
        return 0;
    const uint64_t index = static_cast<uint64_t>(coefficientIndex);
    if (index < firstCoefficient || index > lastCoefficient || index >= coefficientCount)
        return 0;
    if constexpr (Mode == AlignedUnpackMode::NormalizedMontgomery)
        return SharkNTT::MontgomeryMul<SharkFloatParams>(
            grid, block, debugCombo, spectrum[index], roots.Ninv);
    return spectrum[index];
}

static __device__ SharkForceInlineReleaseOnly void
SignedB16ShiftPieces(uint64_t residue,
                     uint32_t shift,
                     uint64_t halfPrime,
                     int64_t &piece0,
                     int64_t &piece1,
                     int64_t &piece2)
{
    if (residue == 0ull) {
        piece0 = 0;
        piece1 = 0;
        piece2 = 0;
        return;
    }

    const bool negative = residue > halfPrime;
    const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
    const uint32_t magnitudeHigh = static_cast<uint32_t>(magnitude >> 32u);
    const uint32_t value0 = static_cast<uint32_t>(magnitude << shift);
    const uint32_t value1 =
        shift == 0u ? magnitudeHigh : static_cast<uint32_t>(magnitude >> (32u - shift));
    const uint32_t value2 = shift == 0u ? 0u : static_cast<uint32_t>(magnitude >> (64u - shift));
    piece0 = negative ? -static_cast<int64_t>(value0) : static_cast<int64_t>(value0);
    piece1 = negative ? -static_cast<int64_t>(value1) : static_cast<int64_t>(value1);
    piece2 = negative ? -static_cast<int64_t>(value2) : static_cast<int64_t>(value2);
}

template <class SharkFloatParams, AlignedUnpackMode Mode>
static __device__ SharkForceInlineReleaseOnly int64_t
ComputeAlignedB16SignedLimb(cooperative_groups::grid_group &grid,
                            cooperative_groups::thread_block &block,
                            DebugGlobalCount<SharkFloatParams> *debugCombo,
                            const uint64_t *spectrum,
                            const SharkNTT::PlanPrime &plan,
                            const SharkNTT::RootTables &roots,
                            uint32_t coefficientCount,
                            uint64_t productBitOffset,
                            const HpSharkFloat<SharkFloatParams> *linearValue,
                            uint32_t linearInputBitOffset,
                            uint64_t linearBitOffset,
                            uint32_t limbBegin,
                            uint32_t tileLimbCount,
                            uint32_t laneIndex)
{
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    MattsCudaAssert(plan.b == 16);
    const uint32_t limbIndex = limbBegin + laneIndex;
    const bool limbIsValid = laneIndex < tileLimbCount;
    const uint32_t limbEnd = limbBegin + tileLimbCount - 1u;
    const int64_t productLimbOffset = static_cast<int64_t>(productBitOffset >> 5u);
    const uint32_t productResidualBitOffset = static_cast<uint32_t>(productBitOffset & 31ull);
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;

    const uint64_t firstBit = limbBegin >= 3u ? static_cast<uint64_t>(limbBegin - 3u) * 32ull : 0ull;
    const uint64_t lastBit = (static_cast<uint64_t>(limbEnd) + 1ull) * 32ull - 1ull;
    const bool productOverlapsTile = productBitOffset <= lastBit;
    uint64_t firstCoefficient = 0;
    uint64_t lastCoefficient = 0;
    if (productOverlapsTile) {
        firstCoefficient =
            firstBit > productBitOffset ? (firstBit - productBitOffset + 15ull) / 16ull : 0ull;
        lastCoefficient = (lastBit - productBitOffset) / 16ull;
        if (coefficientCount != 0u && lastCoefficient >= coefficientCount)
            lastCoefficient = static_cast<uint64_t>(coefficientCount - 1u);
    }
    const bool hasProduct = productOverlapsTile && coefficientCount != 0u &&
                            firstCoefficient <= lastCoefficient && firstCoefficient < coefficientCount;

    int64_t total = limbIsValid ? SignedLinearLimbContribution(
                                      linearValue, linearInputBitOffset, linearBitOffset, limbIndex)
                                : 0;
    if (!hasProduct)
        return total;

    const int64_t pairIndex = static_cast<int64_t>(limbIndex) - productLimbOffset;
    const int64_t evenCoefficientIndex = pairIndex * 2ll;
    const int64_t oddCoefficientIndex = evenCoefficientIndex + 1ll;
    const uint64_t evenResidue =
        NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                           block,
                                                           debugCombo,
                                                           spectrum,
                                                           roots,
                                                           coefficientCount,
                                                           firstCoefficient,
                                                           lastCoefficient,
                                                           evenCoefficientIndex);
    const uint64_t oddResidue = NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                                   block,
                                                                                   debugCombo,
                                                                                   spectrum,
                                                                                   roots,
                                                                                   coefficientCount,
                                                                                   firstCoefficient,
                                                                                   lastCoefficient,
                                                                                   oddCoefficientIndex);

    uint64_t haloEven1Residue = 0;
    uint64_t haloEven2Residue = 0;
    uint64_t haloOdd1Residue = 0;
    uint64_t haloOdd2Residue = 0;
    uint64_t haloOdd3Residue = 0;
    if (laneIndex == 0u) {
        const int64_t tilePairIndex = static_cast<int64_t>(limbBegin) - productLimbOffset;
        haloEven1Residue =
            NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                               block,
                                                               debugCombo,
                                                               spectrum,
                                                               roots,
                                                               coefficientCount,
                                                               firstCoefficient,
                                                               lastCoefficient,
                                                               (tilePairIndex - 1ll) * 2ll);
        haloEven2Residue =
            NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                               block,
                                                               debugCombo,
                                                               spectrum,
                                                               roots,
                                                               coefficientCount,
                                                               firstCoefficient,
                                                               lastCoefficient,
                                                               (tilePairIndex - 2ll) * 2ll);
        haloOdd1Residue =
            NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                               block,
                                                               debugCombo,
                                                               spectrum,
                                                               roots,
                                                               coefficientCount,
                                                               firstCoefficient,
                                                               lastCoefficient,
                                                               (tilePairIndex - 1ll) * 2ll + 1ll);
        haloOdd2Residue =
            NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                               block,
                                                               debugCombo,
                                                               spectrum,
                                                               roots,
                                                               coefficientCount,
                                                               firstCoefficient,
                                                               lastCoefficient,
                                                               (tilePairIndex - 2ll) * 2ll + 1ll);
        if (productResidualBitOffset >= 16u) {
            haloOdd3Residue =
                NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   spectrum,
                                                                   roots,
                                                                   coefficientCount,
                                                                   firstCoefficient,
                                                                   lastCoefficient,
                                                                   (tilePairIndex - 3ll) * 2ll + 1ll);
        }
    }

    const uint32_t oddShift =
        productResidualBitOffset < 16u ? productResidualBitOffset + 16u : productResidualBitOffset - 16u;
    int64_t evenPiece0 = 0;
    int64_t evenPiece1 = 0;
    int64_t evenPiece2 = 0;
    int64_t oddPiece0 = 0;
    int64_t oddPiece1 = 0;
    int64_t oddPiece2 = 0;
    SignedB16ShiftPieces(
        evenResidue, productResidualBitOffset, halfPrime, evenPiece0, evenPiece1, evenPiece2);
    SignedB16ShiftPieces(oddResidue, oddShift, halfPrime, oddPiece0, oddPiece1, oddPiece2);

    int64_t haloEven1Piece0 = 0;
    int64_t haloEven1Piece1 = 0;
    int64_t haloEven1Piece2 = 0;
    int64_t haloEven2Piece0 = 0;
    int64_t haloEven2Piece1 = 0;
    int64_t haloEven2Piece2 = 0;
    int64_t haloOdd1Piece0 = 0;
    int64_t haloOdd1Piece1 = 0;
    int64_t haloOdd1Piece2 = 0;
    int64_t haloOdd2Piece0 = 0;
    int64_t haloOdd2Piece1 = 0;
    int64_t haloOdd2Piece2 = 0;
    int64_t haloOdd3Piece0 = 0;
    int64_t haloOdd3Piece1 = 0;
    int64_t haloOdd3Piece2 = 0;
    if (laneIndex == 0u) {
        SignedB16ShiftPieces(haloEven1Residue,
                             productResidualBitOffset,
                             halfPrime,
                             haloEven1Piece0,
                             haloEven1Piece1,
                             haloEven1Piece2);
        SignedB16ShiftPieces(haloEven2Residue,
                             productResidualBitOffset,
                             halfPrime,
                             haloEven2Piece0,
                             haloEven2Piece1,
                             haloEven2Piece2);
        SignedB16ShiftPieces(
            haloOdd1Residue, oddShift, halfPrime, haloOdd1Piece0, haloOdd1Piece1, haloOdd1Piece2);
        SignedB16ShiftPieces(
            haloOdd2Residue, oddShift, halfPrime, haloOdd2Piece0, haloOdd2Piece1, haloOdd2Piece2);
        if (productResidualBitOffset >= 16u) {
            SignedB16ShiftPieces(
                haloOdd3Residue, oddShift, halfPrime, haloOdd3Piece0, haloOdd3Piece1, haloOdd3Piece2);
        }
    }

    haloEven1Piece1 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven1Piece1), 0));
    haloEven2Piece2 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven2Piece2), 0));
    haloEven1Piece2 =
        static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloEven1Piece2), 0));
    if (productResidualBitOffset < 16u) {
        haloOdd1Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece1), 0));
        haloOdd2Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece2), 0));
        haloOdd1Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece2), 0));
    } else {
        haloOdd1Piece0 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece0), 0));
        haloOdd2Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece1), 0));
        haloOdd3Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd3Piece2), 0));
        haloOdd1Piece1 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece1), 0));
        haloOdd2Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd2Piece2), 0));
        haloOdd1Piece2 =
            static_cast<int64_t>(ShuffleUint64(FullWarpMask, static_cast<uint64_t>(haloOdd1Piece2), 0));
    }

    int64_t evenPrevious =
        static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(evenPiece1), 1u));
    int64_t evenTwoBack =
        static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(evenPiece2), 2u));
    if (laneIndex == 0u) {
        evenPrevious = haloEven1Piece1;
        evenTwoBack = haloEven2Piece2;
    } else if (laneIndex == 1u) {
        evenTwoBack = haloEven1Piece2;
    }

    total += evenPiece0 + evenPrevious + evenTwoBack;

    if (productResidualBitOffset < 16u) {
        int64_t oddPrevious =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece1), 1u));
        int64_t oddTwoBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece2), 2u));
        if (laneIndex == 0u) {
            oddPrevious = haloOdd1Piece1;
            oddTwoBack = haloOdd2Piece2;
        } else if (laneIndex == 1u) {
            oddTwoBack = haloOdd1Piece2;
        }
        total += oddPiece0 + oddPrevious + oddTwoBack;
    } else {
        int64_t oddPrevious =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece0), 1u));
        int64_t oddTwoBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece1), 2u));
        int64_t oddThreeBack =
            static_cast<int64_t>(ShuffleUpUint64(FullWarpMask, static_cast<uint64_t>(oddPiece2), 3u));
        if (laneIndex == 0u) {
            oddPrevious = haloOdd1Piece0;
            oddTwoBack = haloOdd2Piece1;
            oddThreeBack = haloOdd3Piece2;
        } else if (laneIndex == 1u) {
            oddTwoBack = haloOdd1Piece1;
            oddThreeBack = haloOdd2Piece2;
        } else if (laneIndex == 2u) {
            oddThreeBack = haloOdd1Piece2;
        }
        total += oddPrevious + oddTwoBack + oddThreeBack;
    }

    return limbIsValid ? total : 0;
}

template <class SharkFloatParams, AlignedUnpackMode Mode>
__device__ void
UnpackAlignedResiduesToSignedLimbsB16(cooperative_groups::grid_group &grid,
                                      cooperative_groups::thread_block &block,
                                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                                      const uint64_t *spectrum,
                                      const SharkNTT::PlanPrime &plan,
                                      const SharkNTT::RootTables &roots,
                                      uint32_t coefficientCount,
                                      uint64_t productBitOffset,
                                      const HpSharkFloat<SharkFloatParams> *linearValue,
                                      uint32_t linearInputBitOffset,
                                      uint64_t linearBitOffset,
                                      int64_t *limbs,
                                      uint32_t limbCount)
{
    MattsCudaAssert(plan.b == 16);
    constexpr uint32_t WarpSize = 32u;
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t laneIndex = threadIndex & (WarpSize - 1u);
    const uint32_t warpIndexInBlock = threadIndex / WarpSize;
    const uint32_t warpsPerBlock = static_cast<uint32_t>(block.dim_threads().x / WarpSize);
    const uint32_t globalWarpIndex =
        static_cast<uint32_t>(block.group_index().x) * warpsPerBlock + warpIndexInBlock;
    const uint32_t gridWarpCount = static_cast<uint32_t>(grid.size() / WarpSize);
    const uint32_t tileCount = (limbCount + WarpSize - 1u) / WarpSize;
    const int64_t productLimbOffset = static_cast<int64_t>(productBitOffset >> 5u);
    const uint32_t productResidualBitOffset = static_cast<uint32_t>(productBitOffset & 31ull);
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;

    for (uint32_t tileIndex = globalWarpIndex; tileIndex < tileCount; tileIndex += gridWarpCount) {
        const uint32_t limbBegin = tileIndex * WarpSize;
        const uint32_t remainingLimbs = limbCount - limbBegin;
        const uint32_t tileLimbCount = remainingLimbs < WarpSize ? remainingLimbs : WarpSize;
        const uint32_t limbEnd = limbBegin + tileLimbCount - 1u;
        const uint32_t limbIndex = limbBegin + laneIndex;
        const bool limbIsValid = laneIndex < tileLimbCount;

        const uint64_t firstBit = limbBegin >= 3u ? static_cast<uint64_t>(limbBegin - 3u) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(limbEnd) + 1ull) * 32ull - 1ull;
        const bool productOverlapsTile = productBitOffset <= lastBit;
        uint64_t firstCoefficient = 0;
        uint64_t lastCoefficient = 0;
        if (productOverlapsTile) {
            firstCoefficient =
                firstBit > productBitOffset ? (firstBit - productBitOffset + 15ull) / 16ull : 0ull;
            lastCoefficient = (lastBit - productBitOffset) / 16ull;
            if (coefficientCount != 0u && lastCoefficient >= coefficientCount)
                lastCoefficient = static_cast<uint64_t>(coefficientCount - 1u);
        }
        const bool hasProduct = productOverlapsTile && coefficientCount != 0u &&
                                firstCoefficient <= lastCoefficient &&
                                firstCoefficient < coefficientCount;

        int64_t total = limbIsValid ? SignedLinearLimbContribution(
                                          linearValue, linearInputBitOffset, linearBitOffset, limbIndex)
                                    : 0;
        if (hasProduct) {
            const int64_t pairIndex = static_cast<int64_t>(limbIndex) - productLimbOffset;
            const int64_t localEvenCoefficient = pairIndex * 2ll;
            const int64_t localOddCoefficient = localEvenCoefficient + 1ll;
            const uint64_t evenResidue =
                NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   spectrum,
                                                                   roots,
                                                                   coefficientCount,
                                                                   firstCoefficient,
                                                                   lastCoefficient,
                                                                   localEvenCoefficient);
            const uint64_t oddResidue =
                NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   spectrum,
                                                                   roots,
                                                                   coefficientCount,
                                                                   firstCoefficient,
                                                                   lastCoefficient,
                                                                   localOddCoefficient);

            uint64_t haloEven1 = 0;
            uint64_t haloEven2 = 0;
            uint64_t haloEven3 = 0;
            uint64_t haloOdd1 = 0;
            uint64_t haloOdd2 = 0;
            uint64_t haloOdd3 = 0;
            uint64_t haloOdd4 = 0;
            if (laneIndex == 0u) {
                const int64_t tilePairIndex = static_cast<int64_t>(limbBegin) - productLimbOffset;
                haloEven1 =
                    NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                       block,
                                                                       debugCombo,
                                                                       spectrum,
                                                                       roots,
                                                                       coefficientCount,
                                                                       firstCoefficient,
                                                                       lastCoefficient,
                                                                       (tilePairIndex - 1ll) * 2ll);
                haloEven2 =
                    NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                       block,
                                                                       debugCombo,
                                                                       spectrum,
                                                                       roots,
                                                                       coefficientCount,
                                                                       firstCoefficient,
                                                                       lastCoefficient,
                                                                       (tilePairIndex - 2ll) * 2ll);
                haloEven3 =
                    NormalizeB16ResidueForTile<SharkFloatParams, Mode>(grid,
                                                                       block,
                                                                       debugCombo,
                                                                       spectrum,
                                                                       roots,
                                                                       coefficientCount,
                                                                       firstCoefficient,
                                                                       lastCoefficient,
                                                                       (tilePairIndex - 3ll) * 2ll);
                haloOdd1 = NormalizeB16ResidueForTile<SharkFloatParams, Mode>(
                    grid,
                    block,
                    debugCombo,
                    spectrum,
                    roots,
                    coefficientCount,
                    firstCoefficient,
                    lastCoefficient,
                    (tilePairIndex - 1ll) * 2ll + 1ll);
                haloOdd2 = NormalizeB16ResidueForTile<SharkFloatParams, Mode>(
                    grid,
                    block,
                    debugCombo,
                    spectrum,
                    roots,
                    coefficientCount,
                    firstCoefficient,
                    lastCoefficient,
                    (tilePairIndex - 2ll) * 2ll + 1ll);
                haloOdd3 = NormalizeB16ResidueForTile<SharkFloatParams, Mode>(
                    grid,
                    block,
                    debugCombo,
                    spectrum,
                    roots,
                    coefficientCount,
                    firstCoefficient,
                    lastCoefficient,
                    (tilePairIndex - 3ll) * 2ll + 1ll);
                haloOdd4 = NormalizeB16ResidueForTile<SharkFloatParams, Mode>(
                    grid,
                    block,
                    debugCombo,
                    spectrum,
                    roots,
                    coefficientCount,
                    firstCoefficient,
                    lastCoefficient,
                    (tilePairIndex - 4ll) * 2ll + 1ll);
            }

            const uint64_t broadcastHaloEven1 = ShuffleUint64(FullWarpMask, haloEven1, 0);
            const uint64_t broadcastHaloEven2 = ShuffleUint64(FullWarpMask, haloEven2, 0);
            const uint64_t broadcastHaloEven3 = ShuffleUint64(FullWarpMask, haloEven3, 0);
            const uint64_t broadcastHaloOdd1 = ShuffleUint64(FullWarpMask, haloOdd1, 0);
            const uint64_t broadcastHaloOdd2 = ShuffleUint64(FullWarpMask, haloOdd2, 0);
            const uint64_t broadcastHaloOdd3 = ShuffleUint64(FullWarpMask, haloOdd3, 0);
            const uint64_t broadcastHaloOdd4 = ShuffleUint64(FullWarpMask, haloOdd4, 0);

            for (uint32_t distance = 0; distance < 4u; ++distance) {
                const int sourceLane = static_cast<int>(laneIndex) - static_cast<int>(distance);
                const int safeSourceLane = sourceLane >= 0 ? sourceLane : 0;
                uint64_t even = ShuffleUint64(FullWarpMask, evenResidue, safeSourceLane);
                uint64_t odd = 0;
                if (sourceLane < 0) {
                    switch (distance) {
                        case 1u:
                            even = broadcastHaloEven1;
                            break;
                        case 2u:
                            even = broadcastHaloEven2;
                            break;
                        default:
                            even = broadcastHaloEven3;
                            break;
                    }
                }

                if (productResidualBitOffset < 16u) {
                    odd = ShuffleUint64(FullWarpMask, oddResidue, safeSourceLane);
                    if (sourceLane < 0) {
                        switch (distance) {
                            case 1u:
                                odd = broadcastHaloOdd1;
                                break;
                            case 2u:
                                odd = broadcastHaloOdd2;
                                break;
                            default:
                                odd = broadcastHaloOdd3;
                                break;
                        }
                    }
                } else {
                    const int oddSourceLane = sourceLane - 1;
                    const int safeOddSourceLane = oddSourceLane >= 0 ? oddSourceLane : 0;
                    odd = ShuffleUint64(FullWarpMask, oddResidue, safeOddSourceLane);
                    if (oddSourceLane < 0) {
                        switch (distance) {
                            case 0u:
                                odd = broadcastHaloOdd1;
                                break;
                            case 1u:
                                odd = broadcastHaloOdd2;
                                break;
                            case 2u:
                                odd = broadcastHaloOdd3;
                                break;
                            default:
                                odd = broadcastHaloOdd4;
                                break;
                        }
                    }
                }

                if (limbIsValid) {
                    const int64_t evenPairIndex = pairIndex - static_cast<int64_t>(distance);
                    if (evenPairIndex >= 0) {
                        const uint64_t evenCoefficientIndex =
                            static_cast<uint64_t>(evenPairIndex) * 2ull;
                        if (evenCoefficientIndex < coefficientCount)
                            total += SignedResidueContribution(
                                even, evenCoefficientIndex, limbIndex, 16u, halfPrime, productBitOffset);

                        const int64_t oddPairIndex =
                            productResidualBitOffset < 16u ? evenPairIndex : evenPairIndex - 1ll;
                        if (oddPairIndex >= 0) {
                            const uint64_t oddCoefficientIndex =
                                static_cast<uint64_t>(oddPairIndex) * 2ull + 1ull;
                            if (oddCoefficientIndex < coefficientCount)
                                total += SignedResidueContribution(odd,
                                                                   oddCoefficientIndex,
                                                                   limbIndex,
                                                                   16u,
                                                                   halfPrime,
                                                                   productBitOffset);
                        }
                    }
                }
            }
        }

        if (limbIsValid)
            limbs[limbIndex] = total;
    }
}

template <int ContributionPart>
static __device__ SharkForceInlineReleaseOnly int64_t
SignedB16Contribution(uint64_t residue, uint64_t halfPrime)
{
    static_assert(ContributionPart >= 0 && ContributionPart < 5);
    if (residue == 0)
        return 0;

    const bool negative = residue > halfPrime;
    const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
    uint32_t contribution = 0;
    if constexpr (ContributionPart == 0) {
        contribution = static_cast<uint32_t>(magnitude);
    } else if constexpr (ContributionPart == 1) {
        contribution = static_cast<uint32_t>(magnitude >> 32);
    } else if constexpr (ContributionPart == 2) {
        contribution = static_cast<uint32_t>(magnitude) << 16;
    } else if constexpr (ContributionPart == 3) {
        contribution = static_cast<uint32_t>(magnitude >> 16);
    } else {
        contribution = static_cast<uint32_t>(magnitude >> 48);
    }
    const int64_t signedContribution = static_cast<int64_t>(contribution);
    return negative ? -signedContribution : signedContribution;
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsB16(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               const uint64_t *spectrum,
                               const SharkNTT::PlanPrime &plan,
                               const SharkNTT::RootTables &roots,
                               uint32_t coefficientCount,
                               int64_t *limbs,
                               uint32_t limbCount)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    constexpr uint32_t WarpSize = 32u;
    constexpr unsigned FullWarpMask = 0xFFFF'FFFFu;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t laneIndex = threadIndex & (WarpSize - 1u);
    const uint32_t warpIndexInBlock = threadIndex / WarpSize;
    const uint32_t warpsPerBlock = static_cast<uint32_t>(blockDim.x / WarpSize);
    const uint32_t globalWarpIndex =
        static_cast<uint32_t>(block.group_index().x) * warpsPerBlock + warpIndexInBlock;
    const uint32_t gridWarpCount = static_cast<uint32_t>(grid.size() / WarpSize);
    const uint32_t tileCount = (limbCount + WarpSize - 1u) / WarpSize;

    // Each warp owns one 32-limb tile at a time. The grid-stride loop is required because the
    // number of limb tiles can exceed the number of resident warps in the cooperative grid.
    for (uint32_t tileIndex = globalWarpIndex; tileIndex < tileCount; tileIndex += gridWarpCount) {
        const uint32_t limbBegin = tileIndex * WarpSize;
        const uint32_t j = limbBegin + laneIndex;
        const bool limbIsValid = j < limbCount;
        const uint32_t coefficientIndex = 2u * j;

        uint64_t evenResidue = 0;
        uint64_t oddResidue = 0;
        if (limbIsValid) {
            if (coefficientIndex < coefficientCount) {
                evenResidue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[coefficientIndex], roots.Ninv);
            }
            if (coefficientIndex + 1u < coefficientCount) {
                oddResidue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[coefficientIndex + 1u], roots.Ninv);
            }
        }

        uint64_t haloEvenResidue = 0;
        uint64_t haloOddResidue = 0;
        uint64_t haloOddTwoBackResidue = 0;
        if (laneIndex == 0u && limbBegin != 0u) {
            const uint32_t previousCoefficient = 2u * (limbBegin - 1u);
            if (previousCoefficient < coefficientCount) {
                haloEvenResidue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[previousCoefficient], roots.Ninv);
            }
            if (previousCoefficient + 1u < coefficientCount) {
                haloOddResidue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[previousCoefficient + 1u], roots.Ninv);
            }
        }
        if (laneIndex == 1u && limbBegin > 1u) {
            const uint32_t twoBackOddCoefficient = 2u * (limbBegin - 2u) + 1u;
            if (twoBackOddCoefficient < coefficientCount) {
                haloOddTwoBackResidue = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, spectrum[twoBackOddCoefficient], roots.Ninv);
            }
        }

        const uint64_t shuffledPreviousEven = ShuffleUpUint64(FullWarpMask, evenResidue, 1);
        const uint64_t shuffledPreviousOdd = ShuffleUpUint64(FullWarpMask, oddResidue, 1);
        const uint64_t shuffledTwoBackOdd = ShuffleUpUint64(FullWarpMask, oddResidue, 2);
        const uint64_t broadcastHaloEven = ShuffleUint64(FullWarpMask, haloEvenResidue, 0);
        const uint64_t broadcastHaloOdd = ShuffleUint64(FullWarpMask, haloOddResidue, 0);
        const uint64_t broadcastHaloOddTwoBack = ShuffleUint64(FullWarpMask, haloOddTwoBackResidue, 1);

        uint64_t previousEvenResidue = shuffledPreviousEven;
        uint64_t previousOddResidue = shuffledPreviousOdd;
        uint64_t twoBackOddResidue = shuffledTwoBackOdd;
        if (laneIndex == 0u) {
            previousEvenResidue = broadcastHaloEven;
            previousOddResidue = broadcastHaloOdd;
            twoBackOddResidue = broadcastHaloOddTwoBack;
        } else if (laneIndex == 1u) {
            twoBackOddResidue = broadcastHaloOdd;
        }

        if (limbIsValid) {
            int64_t total = 0;
            total += SignedB16Contribution<0>(evenResidue, halfPrime);
            total += SignedB16Contribution<2>(oddResidue, halfPrime);
            total += SignedB16Contribution<1>(previousEvenResidue, halfPrime);
            total += SignedB16Contribution<3>(previousOddResidue, halfPrime);
            total += SignedB16Contribution<4>(twoBackOddResidue, halfPrime);
            limbs[j] = total;
        }
    }
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsOne(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               DebugGlobalCount<SharkFloatParams> *debugCombo,
                               const uint64_t *spectrum,
                               const SharkNTT::PlanPrime &plan,
                               const SharkNTT::RootTables &roots,
                               uint32_t coefficientCount,
                               int64_t *limbs,
                               uint32_t limbCount)
{
    if (plan.b != 16) {
        UnpackResiduesToSignedLimbsScalar<SharkFloatParams, true>(
            grid, block, debugCombo, spectrum, plan, roots, coefficientCount, limbs, limbCount);
    } else {
        UnpackResiduesToSignedLimbsB16(
            grid, block, debugCombo, spectrum, plan, roots, coefficientCount, limbs, limbCount);
    }
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 const SharkNTT::PlanPrime &plan,
                                 const SharkNTT::RootTables &roots,
                                 const uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 uint32_t limbCount)
{
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum0, plan, roots, coefficientCount0, limbs0, limbCount);
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 const SharkNTT::PlanPrime &plan,
                                 const SharkNTT::RootTables &roots,
                                 const uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 const uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 uint32_t limbCount)
{
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum0, plan, roots, coefficientCount0, limbs0, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum1, plan, roots, coefficientCount1, limbs1, limbCount);
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 const SharkNTT::PlanPrime &plan,
                                 const SharkNTT::RootTables &roots,
                                 const uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 const uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 const uint64_t *spectrum2,
                                 uint32_t coefficientCount2,
                                 int64_t *limbs2,
                                 uint32_t limbCount)
{
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum0, plan, roots, coefficientCount0, limbs0, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum1, plan, roots, coefficientCount1, limbs1, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum2, plan, roots, coefficientCount2, limbs2, limbCount);
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 const SharkNTT::PlanPrime &plan,
                                 const SharkNTT::RootTables &roots,
                                 const uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 const uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 const uint64_t *spectrum2,
                                 uint32_t coefficientCount2,
                                 int64_t *limbs2,
                                 const uint64_t *spectrum3,
                                 uint32_t coefficientCount3,
                                 int64_t *limbs3,
                                 uint32_t limbCount)
{
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum0, plan, roots, coefficientCount0, limbs0, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum1, plan, roots, coefficientCount1, limbs1, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum2, plan, roots, coefficientCount2, limbs2, limbCount);
    UnpackResiduesToSignedLimbsOne(
        grid, block, debugCombo, spectrum3, plan, roots, coefficientCount3, limbs3, limbCount);
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
InverseSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 uint64_t *sharedData,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 DebugState<SharkFloatParams> *debugStates,
                                 const SharkNTT::PlanPrime &plan,
                                 SharkNTT::RootTables &roots,
                                 uint32_t limbCount,
                                 uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 DebugStatePurpose limbsPurpose0)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(
        sharedData, grid, block, debugCombo, spectrum0, activeN, plan.stages, roots);
    UnpackResiduesToSignedLimbsBatch(
        grid, block, debugCombo, plan, roots, spectrum0, coefficientCount0, limbs0, limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose0, reinterpret_cast<const uint64_t *>(limbs0), limbCount);
}

template <class SharkFloatParams>
__device__ void
InverseSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 uint64_t *sharedData,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 DebugState<SharkFloatParams> *debugStates,
                                 const SharkNTT::PlanPrime &plan,
                                 SharkNTT::RootTables &roots,
                                 uint32_t limbCount,
                                 uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 DebugStatePurpose limbsPurpose0,
                                 uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 DebugStatePurpose limbsPurpose1)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(
        sharedData, grid, block, debugCombo, spectrum0, spectrum1, activeN, plan.stages, roots);
    UnpackResiduesToSignedLimbsBatch(grid,
                                     block,
                                     debugCombo,
                                     plan,
                                     roots,
                                     spectrum0,
                                     coefficientCount0,
                                     limbs0,
                                     spectrum1,
                                     coefficientCount1,
                                     limbs1,
                                     limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose0, reinterpret_cast<const uint64_t *>(limbs0), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose1, reinterpret_cast<const uint64_t *>(limbs1), limbCount);
}

template <class SharkFloatParams>
__device__ void
InverseSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 uint64_t *sharedData,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 DebugState<SharkFloatParams> *debugStates,
                                 const SharkNTT::PlanPrime &plan,
                                 SharkNTT::RootTables &roots,
                                 uint32_t limbCount,
                                 uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 DebugStatePurpose limbsPurpose0,
                                 uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 DebugStatePurpose limbsPurpose1,
                                 uint64_t *spectrum2,
                                 uint32_t coefficientCount2,
                                 int64_t *limbs2,
                                 DebugStatePurpose limbsPurpose2)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(sharedData,
                                           grid,
                                           block,
                                           debugCombo,
                                           spectrum0,
                                           spectrum1,
                                           spectrum2,
                                           activeN,
                                           plan.stages,
                                           roots);
    UnpackResiduesToSignedLimbsBatch(grid,
                                     block,
                                     debugCombo,
                                     plan,
                                     roots,
                                     spectrum0,
                                     coefficientCount0,
                                     limbs0,
                                     spectrum1,
                                     coefficientCount1,
                                     limbs1,
                                     spectrum2,
                                     coefficientCount2,
                                     limbs2,
                                     limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose0, reinterpret_cast<const uint64_t *>(limbs0), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose1, reinterpret_cast<const uint64_t *>(limbs1), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose2, reinterpret_cast<const uint64_t *>(limbs2), limbCount);
}

template <class SharkFloatParams>
__device__ void
InverseSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 uint64_t *sharedData,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 DebugState<SharkFloatParams> *debugStates,
                                 const SharkNTT::PlanPrime &plan,
                                 SharkNTT::RootTables &roots,
                                 uint32_t limbCount,
                                 uint64_t *spectrum0,
                                 uint32_t coefficientCount0,
                                 int64_t *limbs0,
                                 DebugStatePurpose limbsPurpose0,
                                 uint64_t *spectrum1,
                                 uint32_t coefficientCount1,
                                 int64_t *limbs1,
                                 DebugStatePurpose limbsPurpose1,
                                 uint64_t *spectrum2,
                                 uint32_t coefficientCount2,
                                 int64_t *limbs2,
                                 DebugStatePurpose limbsPurpose2,
                                 uint64_t *spectrum3,
                                 uint32_t coefficientCount3,
                                 int64_t *limbs3,
                                 DebugStatePurpose limbsPurpose3)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(sharedData,
                                           grid,
                                           block,
                                           debugCombo,
                                           spectrum0,
                                           spectrum1,
                                           spectrum2,
                                           spectrum3,
                                           activeN,
                                           plan.stages,
                                           roots);
    UnpackResiduesToSignedLimbsBatch(grid,
                                     block,
                                     debugCombo,
                                     plan,
                                     roots,
                                     spectrum0,
                                     coefficientCount0,
                                     limbs0,
                                     spectrum1,
                                     coefficientCount1,
                                     limbs1,
                                     spectrum2,
                                     coefficientCount2,
                                     limbs2,
                                     spectrum3,
                                     coefficientCount3,
                                     limbs3,
                                     limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose0, reinterpret_cast<const uint64_t *>(limbs0), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose1, reinterpret_cast<const uint64_t *>(limbs1), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose2, reinterpret_cast<const uint64_t *>(limbs2), limbCount);
    StoreReference2DebugState(
        debugStates, grid, block, limbsPurpose3, reinterpret_cast<const uint64_t *>(limbs3), limbCount);
}

template <class SharkFloatParams>
__device__ void
InverseAlignedSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                        cooperative_groups::thread_block &block,
                                        uint64_t *sharedData,
                                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                                        DebugState<SharkFloatParams> *debugStates,
                                        const SharkNTT::PlanPrime &plan,
                                        SharkNTT::RootTables &roots,
                                        uint32_t limbCount,
                                        uint64_t *spectrum0,
                                        uint32_t coefficientCount0,
                                        uint64_t productBitOffset0,
                                        const HpSharkFloat<SharkFloatParams> *linearValue0,
                                        uint32_t linearInputBitOffset0,
                                        uint64_t linearBitOffset0,
                                        int64_t *limbs0,
                                        DebugStatePurpose limbsPurpose0,
                                        uint64_t *spectrum1,
                                        uint32_t coefficientCount1,
                                        uint64_t productBitOffset1,
                                        const HpSharkFloat<SharkFloatParams> *linearValue1,
                                        uint32_t linearInputBitOffset1,
                                        uint64_t linearBitOffset1,
                                        int64_t *limbs1,
                                        DebugStatePurpose limbsPurpose1)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(
        sharedData, grid, block, debugCombo, spectrum0, spectrum1, activeN, plan.stages, roots);
    UnpackAlignedResiduesToSignedLimbsBatch<SharkFloatParams, AlignedUnpackMode::NormalizedMontgomery>(
        grid,
        block,
        debugCombo,
        plan,
        roots,
        spectrum0,
        coefficientCount0,
        productBitOffset0,
        linearValue0,
        linearInputBitOffset0,
        linearBitOffset0,
        limbs0,
        spectrum1,
        coefficientCount1,
        productBitOffset1,
        linearValue1,
        linearInputBitOffset1,
        linearBitOffset1,
        limbs1,
        limbCount);
    if constexpr (!HpShark::DebugChecksums)
        grid.sync();
    StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                     grid,
                                                     block,
                                                     limbsPurpose0,
                                                     reinterpret_cast<const uint64_t *>(limbs0),
                                                     limbsPurpose1,
                                                     reinterpret_cast<const uint64_t *>(limbs1),
                                                     limbCount);
}

template <class SharkFloatParams>
__device__ void
InverseAlignedSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                        cooperative_groups::thread_block &block,
                                        uint64_t *sharedData,
                                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                                        DebugState<SharkFloatParams> *debugStates,
                                        const SharkNTT::PlanPrime &plan,
                                        SharkNTT::RootTables &roots,
                                        uint32_t limbCount,
                                        uint64_t *spectrum0,
                                        uint32_t coefficientCount0,
                                        uint64_t productBitOffset0,
                                        const HpSharkFloat<SharkFloatParams> *linearValue0,
                                        uint32_t linearInputBitOffset0,
                                        uint64_t linearBitOffset0,
                                        int64_t *limbs0,
                                        DebugStatePurpose limbsPurpose0,
                                        uint64_t *spectrum1,
                                        uint32_t coefficientCount1,
                                        uint64_t productBitOffset1,
                                        const HpSharkFloat<SharkFloatParams> *linearValue1,
                                        uint32_t linearInputBitOffset1,
                                        uint64_t linearBitOffset1,
                                        int64_t *limbs1,
                                        DebugStatePurpose limbsPurpose1,
                                        uint64_t *spectrum2,
                                        uint32_t coefficientCount2,
                                        uint64_t productBitOffset2,
                                        const HpSharkFloat<SharkFloatParams> *linearValue2,
                                        uint32_t linearInputBitOffset2,
                                        uint64_t linearBitOffset2,
                                        int64_t *limbs2,
                                        DebugStatePurpose limbsPurpose2,
                                        uint64_t *spectrum3,
                                        uint32_t coefficientCount3,
                                        uint64_t productBitOffset3,
                                        const HpSharkFloat<SharkFloatParams> *linearValue3,
                                        uint32_t linearInputBitOffset3,
                                        uint64_t linearBitOffset3,
                                        int64_t *limbs3,
                                        DebugStatePurpose limbsPurpose3)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    NTTRadix2Batch<SharkFloatParams, true>(sharedData,
                                           grid,
                                           block,
                                           debugCombo,
                                           spectrum0,
                                           spectrum1,
                                           spectrum2,
                                           spectrum3,
                                           activeN,
                                           plan.stages,
                                           roots);
    UnpackAlignedResiduesToSignedLimbsBatch<SharkFloatParams, AlignedUnpackMode::NormalizedMontgomery>(
        grid,
        block,
        debugCombo,
        plan,
        roots,
        spectrum0,
        coefficientCount0,
        productBitOffset0,
        linearValue0,
        linearInputBitOffset0,
        linearBitOffset0,
        limbs0,
        spectrum1,
        coefficientCount1,
        productBitOffset1,
        linearValue1,
        linearInputBitOffset1,
        linearBitOffset1,
        limbs1,
        spectrum2,
        coefficientCount2,
        productBitOffset2,
        linearValue2,
        linearInputBitOffset2,
        linearBitOffset2,
        limbs2,
        spectrum3,
        coefficientCount3,
        productBitOffset3,
        linearValue3,
        linearInputBitOffset3,
        linearBitOffset3,
        limbs3,
        limbCount);
    if constexpr (!HpShark::DebugChecksums)
        grid.sync();
    StoreReference2DebugStateBatch<SharkFloatParams>(debugStates,
                                                     grid,
                                                     block,
                                                     limbsPurpose0,
                                                     reinterpret_cast<const uint64_t *>(limbs0),
                                                     limbsPurpose1,
                                                     reinterpret_cast<const uint64_t *>(limbs1),
                                                     limbsPurpose2,
                                                     reinterpret_cast<const uint64_t *>(limbs2),
                                                     limbsPurpose3,
                                                     reinterpret_cast<const uint64_t *>(limbs3),
                                                     limbCount);
}

static __device__ int32_t
CountLeadingZeros(uint32_t value)
{
    return __clz(value);
}

constexpr uint32_t FinalizationDigitLengthControl = 0;
constexpr uint32_t FinalizationNegativeControl = 1;
constexpr uint32_t FinalizationLowestNonZeroControl = 2;
constexpr uint32_t FinalizationHighestNonZeroControl = 3;
// Kept as an alias for the legacy non-B16 finalization helpers below.
constexpr uint32_t FinalizationNonZeroReductionControl = FinalizationLowestNonZeroControl;

constexpr int32_t CarryPrefixMin = -8;
constexpr int32_t CarryPrefixMax = 7;
constexpr uint32_t CarryPrefixStateCount = CarryPrefixMax - CarryPrefixMin + 1;
constexpr uint32_t CarryPrefixMaxWarps = 32;
constexpr uint32_t CarryPrefixWarpAggregatesOffset = 0u;
constexpr uint32_t CarryPrefixWarpPrefixesOffset = CarryPrefixWarpAggregatesOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixLookbackTransformsOffset =
    CarryPrefixWarpPrefixesOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixLookbackStatesOffset =
    CarryPrefixLookbackTransformsOffset + CarryPrefixMaxWarps;
constexpr uint32_t CarryPrefixControlSlot = 0u;

template <class SharkFloatParams>
__device__ void
FindLowestNonZero(cooperative_groups::grid_group &grid,
                  cooperative_groups::thread_block &block,
                  uint32_t *realDigits,
                  uint32_t *realControl,
                  uint32_t *imagDigits,
                  uint32_t *imagControl,
                  uint32_t *dzdcRealDigits,
                  uint32_t *dzdcRealControl,
                  uint32_t *dzdcImagDigits,
                  uint32_t *dzdcImagControl,
                  uint64_t *sharedStorage)
{
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMinimum = reinterpret_cast<uint32_t *>(sharedStorage);
    const uint32_t realCount = realControl[FinalizationDigitLengthControl];
    const uint32_t imagCount = imagControl[FinalizationDigitLengthControl];
    const bool realEnabled = realControl[FinalizationNegativeControl] != 0u;
    const bool imagEnabled = imagControl[FinalizationNegativeControl] != 0u;
    const uint32_t dzdcRealCount =
        SharkFloatParams::EnableNewtonRaphson ? dzdcRealControl[FinalizationDigitLengthControl] : 0u;
    const uint32_t dzdcImagCount =
        SharkFloatParams::EnableNewtonRaphson ? dzdcImagControl[FinalizationDigitLengthControl] : 0u;
    const bool dzdcRealEnabled =
        SharkFloatParams::EnableNewtonRaphson && dzdcRealControl[FinalizationNegativeControl] != 0u;
    const bool dzdcImagEnabled =
        SharkFloatParams::EnableNewtonRaphson && dzdcImagControl[FinalizationNegativeControl] != 0u;

    if (IsLeader<SharkFloatParams>(block)) {
        realControl[FinalizationNonZeroReductionControl] = realEnabled ? realCount : 0u;
        imagControl[FinalizationNonZeroReductionControl] = imagEnabled ? imagCount : 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            dzdcRealControl[FinalizationNonZeroReductionControl] = dzdcRealEnabled ? dzdcRealCount : 0u;
            dzdcImagControl[FinalizationNonZeroReductionControl] = dzdcImagEnabled ? dzdcImagCount : 0u;
        }
    }
    if (threadIndex == 0u) {
        blockMinimum[0] = realEnabled ? realCount : 0u;
        blockMinimum[1] = imagEnabled ? imagCount : 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            blockMinimum[2] = dzdcRealEnabled ? dzdcRealCount : 0u;
            blockMinimum[3] = dzdcImagEnabled ? dzdcImagCount : 0u;
        }
    }
    grid.sync();

    uint32_t maximumCount = realEnabled ? realCount : 0u;
    maximumCount = imagEnabled && imagCount > maximumCount ? imagCount : maximumCount;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        maximumCount = dzdcRealEnabled && dzdcRealCount > maximumCount ? dzdcRealCount : maximumCount;
        maximumCount = dzdcImagEnabled && dzdcImagCount > maximumCount ? dzdcImagCount : maximumCount;
    }

    uint32_t realLocalMinimum = realEnabled ? realCount : 0u;
    uint32_t imagLocalMinimum = imagEnabled ? imagCount : 0u;
    uint32_t dzdcRealLocalMinimum = dzdcRealEnabled ? dzdcRealCount : 0u;
    uint32_t dzdcImagLocalMinimum = dzdcImagEnabled ? dzdcImagCount : 0u;

    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
        if (realEnabled && index < realCount && realDigits[index] != 0u)
            realLocalMinimum = realLocalMinimum < index ? realLocalMinimum : index;
        if (imagEnabled && index < imagCount && imagDigits[index] != 0u)
            imagLocalMinimum = imagLocalMinimum < index ? imagLocalMinimum : index;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (dzdcRealEnabled && index < dzdcRealCount && dzdcRealDigits[index] != 0u)
                dzdcRealLocalMinimum = dzdcRealLocalMinimum < index ? dzdcRealLocalMinimum : index;
            if (dzdcImagEnabled && index < dzdcImagCount && dzdcImagDigits[index] != 0u)
                dzdcImagLocalMinimum = dzdcImagLocalMinimum < index ? dzdcImagLocalMinimum : index;
        }
    }

    if (realEnabled && realLocalMinimum != realCount)
        atomicMin(&blockMinimum[0], realLocalMinimum);
    if (imagEnabled && imagLocalMinimum != imagCount)
        atomicMin(&blockMinimum[1], imagLocalMinimum);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (dzdcRealEnabled && dzdcRealLocalMinimum != dzdcRealCount)
            atomicMin(&blockMinimum[2], dzdcRealLocalMinimum);
        if (dzdcImagEnabled && dzdcImagLocalMinimum != dzdcImagCount)
            atomicMin(&blockMinimum[3], dzdcImagLocalMinimum);
    }
    __syncthreads();

    if (threadIndex == 0u) {
        if (realEnabled && blockMinimum[0] != realCount)
            atomicMin(&realControl[FinalizationNonZeroReductionControl], blockMinimum[0]);
        if (imagEnabled && blockMinimum[1] != imagCount)
            atomicMin(&imagControl[FinalizationNonZeroReductionControl], blockMinimum[1]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (dzdcRealEnabled && blockMinimum[2] != dzdcRealCount) {
                atomicMin(&dzdcRealControl[FinalizationNonZeroReductionControl], blockMinimum[2]);
            }
            if (dzdcImagEnabled && blockMinimum[3] != dzdcImagCount) {
                atomicMin(&dzdcImagControl[FinalizationNonZeroReductionControl], blockMinimum[3]);
            }
        }
    }
    grid.sync();

    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
            const uint32_t realLowest = realControl[FinalizationNonZeroReductionControl];
            const uint32_t imagLowest = imagControl[FinalizationNonZeroReductionControl];
            if (realEnabled && index < realCount) {
                if (index < realLowest)
                    MattsCudaAssert(realDigits[index] == 0u);
                if (index == realLowest)
                    MattsCudaAssert(realDigits[index] != 0u);
            }
            if (imagEnabled && index < imagCount) {
                if (index < imagLowest)
                    MattsCudaAssert(imagDigits[index] == 0u);
                if (index == imagLowest)
                    MattsCudaAssert(imagDigits[index] != 0u);
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                const uint32_t dzdcRealLowest = dzdcRealControl[FinalizationNonZeroReductionControl];
                const uint32_t dzdcImagLowest = dzdcImagControl[FinalizationNonZeroReductionControl];
                if (dzdcRealEnabled && index < dzdcRealCount) {
                    if (index < dzdcRealLowest)
                        MattsCudaAssert(dzdcRealDigits[index] == 0u);
                    if (index == dzdcRealLowest)
                        MattsCudaAssert(dzdcRealDigits[index] != 0u);
                }
                if (dzdcImagEnabled && index < dzdcImagCount) {
                    if (index < dzdcImagLowest)
                        MattsCudaAssert(dzdcImagDigits[index] == 0u);
                    if (index == dzdcImagLowest)
                        MattsCudaAssert(dzdcImagDigits[index] != 0u);
                }
            }
        }
        grid.sync();
    }
}

enum class CarryPrefixDescriptorState : uint32_t {
    Empty = 0,
    Aggregate = 1,
    Prefix = 2,
};

enum class CarryPrefixLookbackStatus : uint32_t {
    Pending = 0,
    Ready = 1,
    Prefix = 2,
    End = 3,
};

constexpr uint32_t CarryPrefixLookbackStatusMask = 3u;

static __device__ uint32_t
MakeCarryPrefixLookbackToken(uint32_t part, uint32_t batch, uint32_t batchCount)
{
    const uint64_t token = static_cast<uint64_t>(part) * batchCount + batch;
    MattsCudaAssert(token < (1ull << 30u));
    return static_cast<uint32_t>(token);
}

static __device__ uint32_t
PackCarryPrefixLookbackStatus(uint32_t token, CarryPrefixLookbackStatus status)
{
    MattsCudaAssert(token < (1u << 30u));
    return (token << 2u) | static_cast<uint32_t>(status);
}

static __device__ int32_t
CarryOutForSignedLimb(int64_t limb, int32_t carryIn)
{
    constexpr int64_t Base = 1ll << 32;
    const int64_t sum = limb + carryIn;
    const uint32_t digit = static_cast<uint32_t>(static_cast<uint64_t>(sum));
    return static_cast<int32_t>((sum - static_cast<int64_t>(digit)) / Base);
}

static __device__ uint32_t
MakeSignedCarryPrefixByte(int64_t limb)
{
    constexpr int64_t Base = 1ll << 32;
    const int32_t carryAtMin = CarryOutForSignedLimb(limb, CarryPrefixMin);
    const int32_t carryAtMax = CarryOutForSignedLimb(limb, CarryPrefixMax);
    MattsCudaAssert(carryAtMin >= CarryPrefixMin && carryAtMin <= CarryPrefixMax);
    MattsCudaAssert(carryAtMax >= CarryPrefixMin && carryAtMax <= CarryPrefixMax);

    const uint32_t output = static_cast<uint32_t>(carryAtMin - CarryPrefixMin);
    if (carryAtMin == carryAtMax)
        return output;

    MattsCudaAssert(carryAtMax == carryAtMin + 1);
    const int64_t transitionCarry = (static_cast<int64_t>(carryAtMin) + 1) * Base - limb;
    const uint32_t threshold = static_cast<uint32_t>(transitionCarry - CarryPrefixMin);
    MattsCudaAssert(threshold >= 1u && threshold < CarryPrefixStateCount);
    return output | (threshold << 4u);
}

static __device__ uint32_t
ApplyCarryPrefixByte(uint32_t transform, int32_t carry)
{
    MattsCudaAssert(carry >= CarryPrefixMin && carry <= CarryPrefixMax);
    if (transform == 0xFFu)
        return static_cast<uint32_t>(carry - CarryPrefixMin);

    const uint32_t input = static_cast<uint32_t>(carry - CarryPrefixMin);
    const uint32_t base = transform & 0xFu;
    const uint32_t threshold = transform >> 4u;
    return base + (threshold != 0u && input >= threshold ? 1u : 0u);
}

static __device__ uint32_t
ComposeCarryPrefixBytes(uint32_t earlier, uint32_t later)
{
    if (earlier == 0xFFu)
        return later;
    if (later == 0xFFu)
        return earlier;

    const uint32_t earlierBase = earlier & 0xFu;
    const uint32_t earlierThreshold = earlier >> 4u;
    const uint32_t laterBase = later & 0xFu;
    const uint32_t laterThreshold = later >> 4u;
    if (laterThreshold == 0u)
        return laterBase;
    if (earlierThreshold == 0u)
        return ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase) + CarryPrefixMin);

    const uint32_t outputAtBase =
        ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase) + CarryPrefixMin);
    const uint32_t outputAfterStep =
        ApplyCarryPrefixByte(later, static_cast<int32_t>(earlierBase + 1u) + CarryPrefixMin);
    return outputAtBase == outputAfterStep ? outputAtBase : outputAtBase | (earlierThreshold << 4u);
}

static __device__ SharkForceInlineReleaseOnly uint32_t
ComposePackedCarryPrefixes(uint32_t earlier, uint32_t later)
{
    uint32_t combined = ComposeCarryPrefixBytes(earlier & 0xFFu, later & 0xFFu);
    combined |= ComposeCarryPrefixBytes((earlier >> 8u) & 0xFFu, (later >> 8u) & 0xFFu) << 8u;
    combined |= ComposeCarryPrefixBytes((earlier >> 16u) & 0xFFu, (later >> 16u) & 0xFFu) << 16u;
    combined |= ComposeCarryPrefixBytes(earlier >> 24u, later >> 24u) << 24u;
    return combined;
}

static __device__ uint32_t
ApplyPackedCarryPrefix(uint32_t transform, int32_t carry)
{
    uint32_t packedStates = ApplyCarryPrefixByte(transform & 0xFFu, carry);
    packedStates |= ApplyCarryPrefixByte((transform >> 8u) & 0xFFu, carry) << 8u;
    packedStates |= ApplyCarryPrefixByte((transform >> 16u) & 0xFFu, carry) << 16u;
    packedStates |= ApplyCarryPrefixByte(transform >> 24u, carry) << 24u;
    return packedStates;
}

static __device__ void
StoreSignedCarryDigit(int64_t signedLimb,
                      int32_t carryIn,
                      uint32_t index,
                      uint32_t limbCount,
                      uint32_t capacity,
                      uint32_t *digits,
                      uint32_t *control)
{
    digits[index] = static_cast<uint32_t>(static_cast<uint64_t>(signedLimb + carryIn));
    if (index + 1u != limbCount)
        return;

    int32_t finalCarry = CarryOutForSignedLimb(signedLimb, carryIn);
    uint32_t digitLength = limbCount;
    while (finalCarry != 0 && finalCarry != -1 && digitLength < capacity) {
        digits[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
        finalCarry = CarryOutForSignedLimb(finalCarry, 0);
    }
    control[FinalizationDigitLengthControl] = digitLength;
    control[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
}

static __device__ void
PublishCarryPrefixState(uint32_t *state, CarryPrefixDescriptorState value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicState(*state);
    atomicState.store(static_cast<uint32_t>(value), cuda::memory_order_release);
}

static __device__ uint32_t
LoadCarryPrefixState(uint32_t *state)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicState(*state);
    return atomicState.load(cuda::memory_order_acquire);
}

static __device__ uint32_t
LoadCarryPrefixTransform(uint32_t *transform)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicTransform(*transform);
    return atomicTransform.load(cuda::memory_order_relaxed);
}

static __device__ uint32_t
LoadCarryPrefixLookbackStatus(uint32_t *status)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atomicStatus(*status);
    return atomicStatus.load(cuda::memory_order_acquire);
}

static __device__ void
StoreCarryPrefixLookbackStatus(uint32_t *status, uint32_t value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atomicStatus(*status);
    atomicStatus.store(value, cuda::memory_order_release);
}

static __device__ bool
IsCarryPrefixLookbackComplete(uint32_t *control, uint32_t token, uint32_t lane)
{
    if (control == nullptr)
        return false;

    const uint32_t loadedControl = lane == 0u ? LoadCarryPrefixLookbackStatus(control) : 0u;
    const uint32_t controlStatus = __shfl_sync(0xFFFF'FFFFu, loadedControl, 0);
    return controlStatus == PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready);
}

static __device__ void
StoreCarryPrefixTransform(uint32_t *transform, uint32_t value)
{
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atomicTransform(*transform);
    atomicTransform.store(value, cuda::memory_order_relaxed);
}

static __device__ void
PublishCarryPrefixDescriptorAggregate(HpSharkReference2PackedCarryPrefixDescriptor &descriptor,
                                      uint32_t aggregate)
{
    StoreCarryPrefixTransform(&descriptor.AggregateTransform, aggregate);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Aggregate);
}

static __device__ void
PublishCarryPrefixDescriptorPrefix(HpSharkReference2PackedCarryPrefixDescriptor &descriptor,
                                   uint32_t prefix)
{
    StoreCarryPrefixTransform(&descriptor.PrefixTransform, prefix);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Prefix);
}

static __device__ uint32_t
ResolveCarryPrefixHistory(HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                          uint32_t part,
                          uint32_t lane)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    uint32_t exclusive = Identity;
    int32_t previousPart = static_cast<int32_t>(part) - 1;
    int spin = 0;

    while (previousPart >= 0) {
        const int32_t descriptorIndex = previousPart - static_cast<int32_t>(lane);
        const bool validDescriptor = descriptorIndex >= 0;
        CarryPrefixDescriptorState state = CarryPrefixDescriptorState::Empty;
        uint32_t descriptorCount = 0;
        uint32_t validDescriptorCount = 0;
        bool foundPrefix = false;

        do {
            if (validDescriptor && state == CarryPrefixDescriptorState::Empty) {
                state = static_cast<CarryPrefixDescriptorState>(
                    LoadCarryPrefixState(&descriptors[descriptorIndex].State));
            }

            const unsigned validMask = __ballot_sync(0xFFFF'FFFFu, validDescriptor);
            const unsigned readyMask = __ballot_sync(
                0xFFFF'FFFFu, !validDescriptor || state != CarryPrefixDescriptorState::Empty);
            const unsigned unresolvedMask = validMask & ~readyMask;
            validDescriptorCount = static_cast<uint32_t>(__popc(validMask));
            const uint32_t contiguousReadyCount = unresolvedMask == 0u
                                                      ? validDescriptorCount
                                                      : static_cast<uint32_t>(__ffs(unresolvedMask) - 1);
            const unsigned contiguousReadyMask =
                contiguousReadyCount == 32u ? 0xFFFF'FFFFu : ((1u << contiguousReadyCount) - 1u);
            const unsigned prefixMask =
                __ballot_sync(0xFFFF'FFFFu,
                              validDescriptor && state == CarryPrefixDescriptorState::Prefix) &
                contiguousReadyMask;

            if (prefixMask != 0u) {
                descriptorCount = static_cast<uint32_t>(__ffs(prefixMask));
                foundPrefix = true;
            } else if (contiguousReadyCount == validDescriptorCount) {
                descriptorCount = validDescriptorCount;
            }

            if (descriptorCount == 0u) {
                if (++spin == 64) {
                    __nanosleep(64);
                    spin = 0;
                }
            }
        } while (descriptorCount == 0u);

        MattsCudaAssert(descriptorCount <= validDescriptorCount);
        MattsCudaAssert(lane >= descriptorCount || state == CarryPrefixDescriptorState::Aggregate ||
                        state == CarryPrefixDescriptorState::Prefix);
        uint32_t transform = Identity;
        if (lane < descriptorCount) {
            transform = state == CarryPrefixDescriptorState::Prefix
                            ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                            : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
        }

        uint32_t windowTransform = transform;
#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t older = __shfl_down_sync(0xFFFF'FFFFu, windowTransform, offset);
            if (lane + offset < descriptorCount)
                windowTransform = ComposePackedCarryPrefixes(older, windowTransform);
        }

        if (lane == 0u)
            exclusive = ComposePackedCarryPrefixes(windowTransform, exclusive);
        if (foundPrefix)
            break;
        const int32_t nextPreviousPart = previousPart - static_cast<int32_t>(descriptorCount);
        MattsCudaAssert(nextPreviousPart < previousPart);
        previousPart = nextPreviousPart;
    }

    return __shfl_sync(0xFFFF'FFFFu, exclusive, 0);
}

static __device__ uint32_t
ResolveCarryPrefixWindow(HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                         uint32_t part,
                         uint32_t window,
                         uint32_t lane,
                         uint32_t controlToken,
                         uint32_t *lookbackControl,
                         uint32_t *windowStatus,
                         bool *cancelled)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const int32_t windowStart = static_cast<int32_t>(part) - 1 - static_cast<int32_t>(window * 32u);
    *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
    *cancelled = false;

    if (windowStart < 0) {
        if (IsCarryPrefixLookbackComplete(lookbackControl, controlToken, lane)) {
            *cancelled = true;
            return Identity;
        }
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::End);
        return Identity;
    }

    const int32_t descriptorIndex = windowStart - static_cast<int32_t>(lane);
    const bool validDescriptor = descriptorIndex >= 0;
    CarryPrefixDescriptorState state = CarryPrefixDescriptorState::Empty;
    uint32_t descriptorCount = 0u;
    uint32_t validDescriptorCount = 0u;
    bool foundPrefix = false;
    int spin = 0;

    do {
        if (IsCarryPrefixLookbackComplete(lookbackControl, controlToken, lane)) {
            *cancelled = true;
            return Identity;
        }

        if (validDescriptor && state == CarryPrefixDescriptorState::Empty) {
            state = static_cast<CarryPrefixDescriptorState>(
                LoadCarryPrefixState(&descriptors[descriptorIndex].State));
        }

        const unsigned validMask = __ballot_sync(0xFFFF'FFFFu, validDescriptor);
        const unsigned readyMask =
            __ballot_sync(0xFFFF'FFFFu, !validDescriptor || state != CarryPrefixDescriptorState::Empty);
        const unsigned unresolvedMask = validMask & ~readyMask;
        validDescriptorCount = static_cast<uint32_t>(__popc(validMask));
        const uint32_t contiguousReadyCount = unresolvedMask == 0u
                                                  ? validDescriptorCount
                                                  : static_cast<uint32_t>(__ffs(unresolvedMask) - 1);
        const unsigned contiguousReadyMask =
            contiguousReadyCount == 32u ? 0xFFFF'FFFFu : ((1u << contiguousReadyCount) - 1u);
        const unsigned prefixMask =
            __ballot_sync(0xFFFF'FFFFu, validDescriptor && state == CarryPrefixDescriptorState::Prefix) &
            contiguousReadyMask;

        if (prefixMask != 0u) {
            descriptorCount = static_cast<uint32_t>(__ffs(prefixMask));
            foundPrefix = true;
        } else if (contiguousReadyCount == validDescriptorCount) {
            descriptorCount = validDescriptorCount;
        }

        if (descriptorCount == 0u) {
            if (++spin == 64) {
                __nanosleep(64);
                spin = 0;
            }
        }
    } while (descriptorCount == 0u);

    MattsCudaAssert(descriptorCount <= validDescriptorCount);
    MattsCudaAssert(lane >= descriptorCount || state == CarryPrefixDescriptorState::Aggregate ||
                    state == CarryPrefixDescriptorState::Prefix);
    uint32_t transform = Identity;
    if (lane < descriptorCount) {
        transform = state == CarryPrefixDescriptorState::Prefix
                        ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                        : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
    }

    uint32_t windowTransform = transform;
#pragma unroll
    for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
        const uint32_t older = __shfl_down_sync(0xFFFF'FFFFu, windowTransform, offset);
        if (lane + offset < descriptorCount)
            windowTransform = ComposePackedCarryPrefixes(older, windowTransform);
    }

    if (foundPrefix)
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Prefix);
    else if (windowStart < 32)
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::End);
    else
        *windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Ready);

    return __shfl_sync(0xFFFF'FFFFu, windowTransform, 0);
}

static __device__ uint32_t
ResolveCarryPrefixBlockExclusive(HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                                 uint32_t part,
                                 uint32_t lane,
                                 uint32_t warp,
                                 uint32_t numWarps,
                                 uint32_t lookbackBatchCount,
                                 uint32_t *packedLookbackTransforms,
                                 uint32_t *packedLookbackStates)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const uint32_t initialToken = MakeCarryPrefixLookbackToken(part, 0u, lookbackBatchCount);

    if (warp == 0u && lane == 0u)
        StoreCarryPrefixLookbackStatus(
            &packedLookbackStates[CarryPrefixControlSlot],
            PackCarryPrefixLookbackStatus(initialToken, CarryPrefixLookbackStatus::Pending));

    if (numWarps == 1u) {
        const uint32_t packedBlockExclusive = ResolveCarryPrefixHistory(descriptors, part, lane);
        if (warp == 0u && lane == 0u) {
            packedLookbackTransforms[CarryPrefixControlSlot] = packedBlockExclusive;
            StoreCarryPrefixLookbackStatus(
                &packedLookbackStates[CarryPrefixControlSlot],
                PackCarryPrefixLookbackStatus(initialToken, CarryPrefixLookbackStatus::Ready));
        }
        return packedBlockExclusive;
    }

    if (warp == 0u) {
        // Warp zero consumes the per-warp windows in order and owns the control slot.
        uint32_t batch = 0u;
        uint32_t accumulated = Identity;
        bool done = false;
        while (!done) {
            const uint32_t token = MakeCarryPrefixLookbackToken(part, batch, lookbackBatchCount);
            uint32_t windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
            bool cancelled = false;
            const uint32_t windowTransform = ResolveCarryPrefixWindow(
                descriptors, part, batch * numWarps, lane, token, nullptr, &windowStatus, &cancelled);
            MattsCudaAssert(!cancelled);

            uint32_t lane0Done = 0u;
            uint32_t nextBatch = batch;
            if (lane == 0u) {
                uint32_t batchTransform = windowTransform;
                bool batchDone =
                    windowStatus == static_cast<uint32_t>(CarryPrefixLookbackStatus::Prefix) ||
                    windowStatus == static_cast<uint32_t>(CarryPrefixLookbackStatus::End);

                if (!batchDone) {
                    for (uint32_t windowWarp = 1u; windowWarp < numWarps; ++windowWarp) {
                        uint32_t slotStatus = 0u;
                        int spin = 0;
                        do {
                            slotStatus =
                                LoadCarryPrefixLookbackStatus(&packedLookbackStates[windowWarp]);
                            if (slotStatus == PackCarryPrefixLookbackStatus(
                                                  token, CarryPrefixLookbackStatus::Ready) ||
                                slotStatus == PackCarryPrefixLookbackStatus(
                                                  token, CarryPrefixLookbackStatus::Prefix) ||
                                slotStatus ==
                                    PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::End))
                                break;
                            if (++spin == 64) {
                                __nanosleep(64);
                                spin = 0;
                            }
                        } while (true);

                        const CarryPrefixLookbackStatus status = static_cast<CarryPrefixLookbackStatus>(
                            slotStatus & CarryPrefixLookbackStatusMask);
                        batchTransform = ComposePackedCarryPrefixes(packedLookbackTransforms[windowWarp],
                                                                    batchTransform);
                        if (status == CarryPrefixLookbackStatus::Prefix ||
                            status == CarryPrefixLookbackStatus::End) {
                            batchDone = true;
                            break;
                        }
                    }
                }

                accumulated = ComposePackedCarryPrefixes(batchTransform, accumulated);
                if (batchDone) {
                    packedLookbackTransforms[CarryPrefixControlSlot] = accumulated;
                    StoreCarryPrefixLookbackStatus(
                        &packedLookbackStates[CarryPrefixControlSlot],
                        PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready));
                    lane0Done = 1u;
                } else {
                    nextBatch = batch + 1u;
                    const uint32_t nextToken =
                        MakeCarryPrefixLookbackToken(part, nextBatch, lookbackBatchCount);
                    StoreCarryPrefixLookbackStatus(
                        &packedLookbackStates[CarryPrefixControlSlot],
                        PackCarryPrefixLookbackStatus(nextToken, CarryPrefixLookbackStatus::Pending));
                }
            }

            done = __shfl_sync(0xFFFF'FFFFu, lane0Done, 0) != 0u;
            if (!done)
                batch = __shfl_sync(0xFFFF'FFFFu, nextBatch, 0);
        }
    } else {
        // Other warps resolve one window, then follow the coordinator command.
        for (uint32_t batch = 0u; batch < lookbackBatchCount; ++batch) {
            const uint32_t token = MakeCarryPrefixLookbackToken(part, batch, lookbackBatchCount);
            uint32_t windowStatus = static_cast<uint32_t>(CarryPrefixLookbackStatus::Pending);
            bool cancelled = false;
            const uint32_t windowTransform = ResolveCarryPrefixWindow(descriptors,
                                                                      part,
                                                                      batch * numWarps + warp,
                                                                      lane,
                                                                      token,
                                                                      packedLookbackStates,
                                                                      &windowStatus,
                                                                      &cancelled);
            if (cancelled)
                break;

            if (lane == 0u) {
                packedLookbackTransforms[warp] = windowTransform;
                StoreCarryPrefixLookbackStatus(
                    &packedLookbackStates[warp],
                    PackCarryPrefixLookbackStatus(token,
                                                  static_cast<CarryPrefixLookbackStatus>(windowStatus)));
            }

            bool complete = false;
            bool advance = false;
            do {
                const uint32_t controlStatus =
                    lane == 0u
                        ? LoadCarryPrefixLookbackStatus(&packedLookbackStates[CarryPrefixControlSlot])
                        : 0u;
                const uint32_t command = __shfl_sync(0xFFFF'FFFFu, controlStatus, 0);
                const uint32_t nextToken =
                    batch + 1u < lookbackBatchCount
                        ? MakeCarryPrefixLookbackToken(part, batch + 1u, lookbackBatchCount)
                        : 0u;
                if (command == PackCarryPrefixLookbackStatus(token, CarryPrefixLookbackStatus::Ready)) {
                    complete = true;
                } else if (batch + 1u < lookbackBatchCount &&
                           command == PackCarryPrefixLookbackStatus(nextToken,
                                                                    CarryPrefixLookbackStatus::Ready)) {
                    complete = true;
                } else if (batch + 1u < lookbackBatchCount &&
                           command == PackCarryPrefixLookbackStatus(
                                          nextToken, CarryPrefixLookbackStatus::Pending)) {
                    advance = true;
                } else if (lane == 0u) {
                    __nanosleep(64);
                }
            } while (!complete && !advance);

            if (complete)
                break;
        }
    }

    return packedLookbackTransforms[CarryPrefixControlSlot];
}

template <class SharkFloatParams>
static __device__ void
EmitPackedCarryPrefixDigits(uint32_t packedInclusive,
                            uint32_t packedBlockExclusive,
                            uint32_t packedWarpPrefix,
                            uint32_t lane,
                            bool hasValue,
                            uint32_t index,
                            uint32_t count,
                            uint32_t capacity,
                            const int64_t *realLimbs,
                            uint32_t *realDigits,
                            uint32_t *realControl,
                            const int64_t *imagLimbs,
                            uint32_t *imagDigits,
                            uint32_t *imagControl,
                            const int64_t *dzdcRealLimbs,
                            uint32_t *dzdcRealDigits,
                            uint32_t *dzdcRealControl,
                            const int64_t *dzdcImagLimbs,
                            uint32_t *dzdcImagDigits,
                            uint32_t *dzdcImagControl)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, 1);
    const uint32_t packedLocalExclusive = lane == 0u ? Identity : previous;
    if (!hasValue)
        return;

    const uint32_t packedExclusive = ComposePackedCarryPrefixes(
        packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
    const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);

    StoreSignedCarryDigit(realLimbs[index],
                          static_cast<int32_t>(packedCarries & 0xFFu) + CarryPrefixMin,
                          index,
                          count,
                          capacity,
                          realDigits,
                          realControl);
    StoreSignedCarryDigit(imagLimbs[index],
                          static_cast<int32_t>((packedCarries >> 8u) & 0xFFu) + CarryPrefixMin,
                          index,
                          count,
                          capacity,
                          imagDigits,
                          imagControl);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreSignedCarryDigit(dzdcRealLimbs[index],
                              static_cast<int32_t>((packedCarries >> 16u) & 0xFFu) + CarryPrefixMin,
                              index,
                              count,
                              capacity,
                              dzdcRealDigits,
                              dzdcRealControl);
        StoreSignedCarryDigit(dzdcImagLimbs[index],
                              static_cast<int32_t>((packedCarries >> 24u) & 0xFFu) + CarryPrefixMin,
                              index,
                              count,
                              capacity,
                              dzdcImagDigits,
                              dzdcImagControl);
    }
}

template <class SharkFloatParams>
static __device__ void
EmitPackedCarryPrefixDigitValues(uint32_t packedInclusive,
                                 uint32_t packedBlockExclusive,
                                 uint32_t packedWarpPrefix,
                                 uint32_t lane,
                                 bool hasValue,
                                 uint32_t index,
                                 uint32_t count,
                                 uint32_t capacity,
                                 int64_t realLimb,
                                 uint32_t *realDigits,
                                 uint32_t *realControl,
                                 int64_t imagLimb,
                                 uint32_t *imagDigits,
                                 uint32_t *imagControl,
                                 int64_t dzdcRealLimb,
                                 uint32_t *dzdcRealDigits,
                                 uint32_t *dzdcRealControl,
                                 int64_t dzdcImagLimb,
                                 uint32_t *dzdcImagDigits,
                                 uint32_t *dzdcImagControl)
{
    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, 1);
    const uint32_t packedLocalExclusive = lane == 0u ? Identity : previous;
    if (!hasValue)
        return;

    const uint32_t packedExclusive = ComposePackedCarryPrefixes(
        packedBlockExclusive, ComposePackedCarryPrefixes(packedWarpPrefix, packedLocalExclusive));
    const uint32_t packedCarries = ApplyPackedCarryPrefix(packedExclusive, 0);

    StoreSignedCarryDigit(realLimb,
                          static_cast<int32_t>(packedCarries & 0xFFu) + CarryPrefixMin,
                          index,
                          count,
                          capacity,
                          realDigits,
                          realControl);
    StoreSignedCarryDigit(imagLimb,
                          static_cast<int32_t>((packedCarries >> 8u) & 0xFFu) + CarryPrefixMin,
                          index,
                          count,
                          capacity,
                          imagDigits,
                          imagControl);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreSignedCarryDigit(dzdcRealLimb,
                              static_cast<int32_t>((packedCarries >> 16u) & 0xFFu) + CarryPrefixMin,
                              index,
                              count,
                              capacity,
                              dzdcRealDigits,
                              dzdcRealControl);
        StoreSignedCarryDigit(dzdcImagLimb,
                              static_cast<int32_t>((packedCarries >> 24u) & 0xFFu) + CarryPrefixMin,
                              index,
                              count,
                              capacity,
                              dzdcImagDigits,
                              dzdcImagControl);
    }
}

template <class SharkFloatParams>
__device__ void
InitializeCarryPrefixTransformsDLB(cooperative_groups::grid_group &grid,
                                   cooperative_groups::thread_block &block,
                                   uint32_t count,
                                   uint32_t capacity,
                                   HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                                   uint64_t *sharedStorage)
{
    if (count == 0u)
        return;

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & 31u;
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + 31u) >> 5u;
    uint32_t *packedCarryPrefixShared = reinterpret_cast<uint32_t *>(sharedStorage);
    uint32_t *packedLookbackStates = packedCarryPrefixShared + CarryPrefixLookbackStatesOffset;

    MattsCudaAssert(blockSize >= 32u && (blockSize & 31u) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);
    MattsCudaAssert(capacity >= count);

    const uint32_t lookbackWindowsPerBatch = numWarps * 32u;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    const uint32_t processorId = block.group_index().x;
    const uint32_t firstPart = processorId < numParts ? processorId : 0u;
    const uint32_t firstToken = MakeCarryPrefixLookbackToken(firstPart, 0u, lookbackBatchCount);
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize)
        PublishCarryPrefixState(&descriptors[part].State, CarryPrefixDescriptorState::Empty);
    if (lane == 0u)
        StoreCarryPrefixLookbackStatus(
            &packedLookbackStates[warp],
            PackCarryPrefixLookbackStatus(firstToken, CarryPrefixLookbackStatus::Pending));
    // The caller must grid-sync before PrefixCarryTransformsDLB consumes these states.
}

// Build the four active stream transfers, resolve one packed DLB prefix, and emit
// the corresponding digits. Newton-Raphson uses all four packed bytes so the
// reference orbit and both derivative streams share this complete carry path.
template <class SharkFloatParams>
__device__ void
PrefixCarryTransformsDLB(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         uint32_t count,
                         uint32_t capacity,
                         const int64_t *realLimbs,
                         uint32_t *realDigits,
                         uint32_t *realControl,
                         const int64_t *imagLimbs,
                         uint32_t *imagDigits,
                         uint32_t *imagControl,
                         const int64_t *dzdcRealLimbs,
                         uint32_t *dzdcRealDigits,
                         uint32_t *dzdcRealControl,
                         const int64_t *dzdcImagLimbs,
                         uint32_t *dzdcImagDigits,
                         uint32_t *dzdcImagControl,
                         HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                         uint64_t *sharedStorage)
{
    if (count == 0u)
        return;

    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & 31u;
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + 31u) >> 5u;
    // Shared scratch is partitioned into warp aggregates, warp prefixes, lookback transforms, and
    // per-warp lookback states. Slot zero is the coordinator's control/transform slot.
    uint32_t *packedCarryPrefixShared = reinterpret_cast<uint32_t *>(sharedStorage);
    uint32_t *packedWarpAggregates = packedCarryPrefixShared + CarryPrefixWarpAggregatesOffset;
    uint32_t *packedWarpPrefixes = packedCarryPrefixShared + CarryPrefixWarpPrefixesOffset;
    uint32_t *packedLookbackTransforms = packedCarryPrefixShared + CarryPrefixLookbackTransformsOffset;
    uint32_t *packedLookbackStates = packedCarryPrefixShared + CarryPrefixLookbackStatesOffset;

    MattsCudaAssert(blockSize >= 32u && (blockSize & 31u) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);
    MattsCudaAssert(capacity >= count);
    // Descriptors and lookback states were initialized before the preceding publication barrier,
    // which publishes them before this DLB pass begins.

    const uint32_t lookbackWindowsPerBatch = numWarps * 32u;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        // Form the inclusive transform for this thread, then reduce the block to one aggregate.
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;

        uint32_t packedInclusive = Identity;
        if (hasValue) {
            packedInclusive = MakeSignedCarryPrefixByte(realLimbs[index]);
            packedInclusive |= MakeSignedCarryPrefixByte(imagLimbs[index]) << 8u;
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                packedInclusive |= MakeSignedCarryPrefixByte(dzdcRealLimbs[index]) << 16u;
                packedInclusive |= MakeSignedCarryPrefixByte(dzdcImagLimbs[index]) << 24u;
            }
        }

#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, offset);
            if (lane >= offset)
                packedInclusive = ComposePackedCarryPrefixes(previous, packedInclusive);
        }

        const uint32_t warpEnd = (warp + 1u) * 32u;
        const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
        if (threadIndex == warpLastThread)
            packedWarpAggregates[warp] = packedInclusive;
        __syncthreads();

        uint32_t packedAggregate = Identity;
        if (threadIndex < 32u) {
            uint32_t packedWarpInclusive = lane < numWarps ? packedWarpAggregates[lane] : Identity;
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, offset);
                if (lane >= offset && lane < numWarps)
                    packedWarpInclusive = ComposePackedCarryPrefixes(previous, packedWarpInclusive);
            }

            const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, 1);
            if (lane < numWarps)
                packedWarpPrefixes[lane] = lane == 0u ? Identity : previous;
            packedAggregate =
                __shfl_sync(0xFFFF'FFFFu, packedWarpInclusive, static_cast<int>(numWarps - 1u));
        }

        if (threadIndex == 0u)
            PublishCarryPrefixDescriptorAggregate(descriptors[part], packedAggregate);

        // Resolve all earlier descriptor aggregates and publish this block's exclusive prefix.
        const uint32_t resolvedBlockExclusive =
            ResolveCarryPrefixBlockExclusive(descriptors,
                                             part,
                                             lane,
                                             warp,
                                             numWarps,
                                             lookbackBatchCount,
                                             packedLookbackTransforms,
                                             packedLookbackStates);

        if (threadIndex == 0u) {
            PublishCarryPrefixDescriptorPrefix(
                descriptors[part], ComposePackedCarryPrefixes(resolvedBlockExclusive, packedAggregate));
        }
        __syncthreads();

        // Apply the block, warp, and local prefixes to each active stream.
        const uint32_t packedBlockExclusive = packedLookbackTransforms[CarryPrefixControlSlot];
        EmitPackedCarryPrefixDigits<SharkFloatParams>(packedInclusive,
                                                      packedBlockExclusive,
                                                      packedWarpPrefixes[warp],
                                                      lane,
                                                      hasValue,
                                                      index,
                                                      count,
                                                      capacity,
                                                      realLimbs,
                                                      realDigits,
                                                      realControl,
                                                      imagLimbs,
                                                      imagDigits,
                                                      imagControl,
                                                      dzdcRealLimbs,
                                                      dzdcRealDigits,
                                                      dzdcRealControl,
                                                      dzdcImagLimbs,
                                                      dzdcImagDigits,
                                                      dzdcImagControl);
        // The next partition's aggregate barrier protects shared scratch reuse.
    }
    grid.sync();
}

// B16 path: compute the signed limb contributions in the same warp that builds the carry
// transform.  This keeps the normal/release path from materializing and then rereading four
// limb arrays, while MaterializeLimbs preserves the exact Debug checksum buffers.
template <class SharkFloatParams, AlignedUnpackMode Mode, bool MaterializeLimbs>
__device__ void
PrefixAlignedB16CarryTransformsDLB(cooperative_groups::grid_group &grid,
                                   cooperative_groups::thread_block &block,
                                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                                   uint32_t count,
                                   uint32_t capacity,
                                   const SharkNTT::PlanPrime &plan,
                                   const SharkNTT::RootTables &roots,
                                   const uint64_t *realSpectrum,
                                   uint32_t realCoefficientCount,
                                   uint64_t realProductBitOffset,
                                   const HpSharkFloat<SharkFloatParams> *realLinearValue,
                                   uint32_t realLinearInputBitOffset,
                                   uint64_t realLinearBitOffset,
                                   int64_t *realLimbs,
                                   uint32_t *realDigits,
                                   uint32_t *realControl,
                                   const uint64_t *imagSpectrum,
                                   uint32_t imagCoefficientCount,
                                   uint64_t imagProductBitOffset,
                                   const HpSharkFloat<SharkFloatParams> *imagLinearValue,
                                   uint32_t imagLinearInputBitOffset,
                                   uint64_t imagLinearBitOffset,
                                   int64_t *imagLimbs,
                                   uint32_t *imagDigits,
                                   uint32_t *imagControl,
                                   const uint64_t *dzdcRealSpectrum,
                                   uint32_t dzdcRealCoefficientCount,
                                   uint64_t dzdcRealProductBitOffset,
                                   const HpSharkFloat<SharkFloatParams> *dzdcRealLinearValue,
                                   uint32_t dzdcRealLinearInputBitOffset,
                                   uint64_t dzdcRealLinearBitOffset,
                                   int64_t *dzdcRealLimbs,
                                   uint32_t *dzdcRealDigits,
                                   uint32_t *dzdcRealControl,
                                   const uint64_t *dzdcImagSpectrum,
                                   uint32_t dzdcImagCoefficientCount,
                                   uint64_t dzdcImagProductBitOffset,
                                   const HpSharkFloat<SharkFloatParams> *dzdcImagLinearValue,
                                   uint32_t dzdcImagLinearInputBitOffset,
                                   uint64_t dzdcImagLinearBitOffset,
                                   int64_t *dzdcImagLimbs,
                                   uint32_t *dzdcImagDigits,
                                   uint32_t *dzdcImagControl,
                                   HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                                   uint64_t *sharedStorage)
{
    if (count == 0u)
        return;

    constexpr uint32_t Identity = 0xFFFF'FFFFu;
    constexpr uint32_t WarpSize = 32u;
    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & (WarpSize - 1u);
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + 31u) >> 5u;
    uint32_t *packedCarryPrefixShared = reinterpret_cast<uint32_t *>(sharedStorage);
    uint32_t *packedWarpAggregates = packedCarryPrefixShared + CarryPrefixWarpAggregatesOffset;
    uint32_t *packedWarpPrefixes = packedCarryPrefixShared + CarryPrefixWarpPrefixesOffset;
    uint32_t *packedLookbackTransforms = packedCarryPrefixShared + CarryPrefixLookbackTransformsOffset;
    uint32_t *packedLookbackStates = packedCarryPrefixShared + CarryPrefixLookbackStatesOffset;

    MattsCudaAssert(plan.b == 16);
    MattsCudaAssert(blockSize >= WarpSize && (blockSize & (WarpSize - 1u)) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);
    MattsCudaAssert(capacity >= count);

    // Finalization consumes this control after the carry pass.  Initialize it before
    // the pass so the carry pass's completion barrier publishes both the digits and
    // the control state.
    if (IsLeader<SharkFloatParams>(block)) {
        realControl[FinalizationHighestNonZeroControl] = 0u;
        imagControl[FinalizationHighestNonZeroControl] = 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            dzdcRealControl[FinalizationHighestNonZeroControl] = 0u;
            dzdcImagControl[FinalizationHighestNonZeroControl] = 0u;
        }
    }

    const uint32_t lookbackWindowsPerBatch = numWarps * WarpSize;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;

    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        const uint32_t limbBegin = base + (threadIndex & ~(WarpSize - 1u));
        const uint32_t remainingLimbs = limbBegin < count ? count - limbBegin : 0u;
        const uint32_t tileLimbCount = remainingLimbs < WarpSize ? remainingLimbs : WarpSize;

        int64_t realLimb = 0;
        int64_t imagLimb = 0;
        int64_t dzdcRealLimb = 0;
        int64_t dzdcImagLimb = 0;
        if (tileLimbCount != 0u) {
            realLimb = ComputeAlignedB16SignedLimb<SharkFloatParams, Mode>(grid,
                                                                           block,
                                                                           debugCombo,
                                                                           realSpectrum,
                                                                           plan,
                                                                           roots,
                                                                           realCoefficientCount,
                                                                           realProductBitOffset,
                                                                           realLinearValue,
                                                                           realLinearInputBitOffset,
                                                                           realLinearBitOffset,
                                                                           limbBegin,
                                                                           tileLimbCount,
                                                                           lane);
            imagLimb = ComputeAlignedB16SignedLimb<SharkFloatParams, Mode>(grid,
                                                                           block,
                                                                           debugCombo,
                                                                           imagSpectrum,
                                                                           plan,
                                                                           roots,
                                                                           imagCoefficientCount,
                                                                           imagProductBitOffset,
                                                                           imagLinearValue,
                                                                           imagLinearInputBitOffset,
                                                                           imagLinearBitOffset,
                                                                           limbBegin,
                                                                           tileLimbCount,
                                                                           lane);
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                dzdcRealLimb =
                    ComputeAlignedB16SignedLimb<SharkFloatParams, Mode>(grid,
                                                                        block,
                                                                        debugCombo,
                                                                        dzdcRealSpectrum,
                                                                        plan,
                                                                        roots,
                                                                        dzdcRealCoefficientCount,
                                                                        dzdcRealProductBitOffset,
                                                                        dzdcRealLinearValue,
                                                                        dzdcRealLinearInputBitOffset,
                                                                        dzdcRealLinearBitOffset,
                                                                        limbBegin,
                                                                        tileLimbCount,
                                                                        lane);
                dzdcImagLimb =
                    ComputeAlignedB16SignedLimb<SharkFloatParams, Mode>(grid,
                                                                        block,
                                                                        debugCombo,
                                                                        dzdcImagSpectrum,
                                                                        plan,
                                                                        roots,
                                                                        dzdcImagCoefficientCount,
                                                                        dzdcImagProductBitOffset,
                                                                        dzdcImagLinearValue,
                                                                        dzdcImagLinearInputBitOffset,
                                                                        dzdcImagLinearBitOffset,
                                                                        limbBegin,
                                                                        tileLimbCount,
                                                                        lane);
            }
        }

        if constexpr (MaterializeLimbs) {
            if (hasValue) {
                realLimbs[index] = realLimb;
                imagLimbs[index] = imagLimb;
                if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                    dzdcRealLimbs[index] = dzdcRealLimb;
                    dzdcImagLimbs[index] = dzdcImagLimb;
                }
            }
        }

        uint32_t packedInclusive = Identity;
        if (hasValue) {
            packedInclusive = MakeSignedCarryPrefixByte(realLimb);
            packedInclusive |= MakeSignedCarryPrefixByte(imagLimb) << 8u;
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                packedInclusive |= MakeSignedCarryPrefixByte(dzdcRealLimb) << 16u;
                packedInclusive |= MakeSignedCarryPrefixByte(dzdcImagLimb) << 24u;
            }
        }

#pragma unroll
        for (uint32_t offset = 1u; offset < WarpSize; offset <<= 1u) {
            const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedInclusive, offset);
            if (lane >= offset)
                packedInclusive = ComposePackedCarryPrefixes(previous, packedInclusive);
        }

        const uint32_t warpEnd = (warp + 1u) * WarpSize;
        const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
        if (threadIndex == warpLastThread)
            packedWarpAggregates[warp] = packedInclusive;
        __syncthreads();

        uint32_t packedAggregate = Identity;
        if (threadIndex < WarpSize) {
            uint32_t packedWarpInclusive = lane < numWarps ? packedWarpAggregates[lane] : Identity;
#pragma unroll
            for (uint32_t offset = 1u; offset < WarpSize; offset <<= 1u) {
                const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, offset);
                if (lane >= offset && lane < numWarps)
                    packedWarpInclusive = ComposePackedCarryPrefixes(previous, packedWarpInclusive);
            }

            const uint32_t previous = __shfl_up_sync(0xFFFF'FFFFu, packedWarpInclusive, 1);
            if (lane < numWarps)
                packedWarpPrefixes[lane] = lane == 0u ? Identity : previous;
            packedAggregate =
                __shfl_sync(0xFFFF'FFFFu, packedWarpInclusive, static_cast<int>(numWarps - 1u));
        }

        if (threadIndex == 0u)
            PublishCarryPrefixDescriptorAggregate(descriptors[part], packedAggregate);

        const uint32_t resolvedBlockExclusive =
            ResolveCarryPrefixBlockExclusive(descriptors,
                                             part,
                                             lane,
                                             warp,
                                             numWarps,
                                             lookbackBatchCount,
                                             packedLookbackTransforms,
                                             packedLookbackStates);

        if (threadIndex == 0u) {
            PublishCarryPrefixDescriptorPrefix(
                descriptors[part], ComposePackedCarryPrefixes(resolvedBlockExclusive, packedAggregate));
        }
        __syncthreads();

        const uint32_t packedBlockExclusive = packedLookbackTransforms[CarryPrefixControlSlot];
        EmitPackedCarryPrefixDigitValues<SharkFloatParams>(packedInclusive,
                                                           packedBlockExclusive,
                                                           packedWarpPrefixes[warp],
                                                           lane,
                                                           hasValue,
                                                           index,
                                                           count,
                                                           capacity,
                                                           realLimb,
                                                           realDigits,
                                                           realControl,
                                                           imagLimb,
                                                           imagDigits,
                                                           imagControl,
                                                           dzdcRealLimb,
                                                           dzdcRealDigits,
                                                           dzdcRealControl,
                                                           dzdcImagLimb,
                                                           dzdcImagDigits,
                                                           dzdcImagControl);
    }
    grid.sync();
}

template <class SharkFloatParams, AlignedUnpackMode Mode, bool MaterializeLimbs>
__device__ void
PrefixAlignedB16CarryTransformsTwoWay(cooperative_groups::grid_group &grid,
                                      cooperative_groups::thread_block &block,
                                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                                      uint32_t count,
                                      uint32_t capacity,
                                      const SharkNTT::PlanPrime &plan,
                                      const SharkNTT::RootTables &roots,
                                      const uint64_t *realSpectrum,
                                      uint32_t realCoefficientCount,
                                      uint64_t realProductBitOffset,
                                      const HpSharkFloat<SharkFloatParams> *realLinearValue,
                                      uint32_t realLinearInputBitOffset,
                                      uint64_t realLinearBitOffset,
                                      int64_t *realLimbs,
                                      uint32_t *realDigits,
                                      uint32_t *realControl,
                                      const uint64_t *imagSpectrum,
                                      uint32_t imagCoefficientCount,
                                      uint64_t imagProductBitOffset,
                                      const HpSharkFloat<SharkFloatParams> *imagLinearValue,
                                      uint32_t imagLinearInputBitOffset,
                                      uint64_t imagLinearBitOffset,
                                      int64_t *imagLimbs,
                                      uint32_t *imagDigits,
                                      uint32_t *imagControl,
                                      HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                                      uint64_t *sharedStorage)
{
    PrefixAlignedB16CarryTransformsDLB<SharkFloatParams, Mode, MaterializeLimbs>(
        grid,
        block,
        debugCombo,
        count,
        capacity,
        plan,
        roots,
        realSpectrum,
        realCoefficientCount,
        realProductBitOffset,
        realLinearValue,
        realLinearInputBitOffset,
        realLinearBitOffset,
        realLimbs,
        realDigits,
        realControl,
        imagSpectrum,
        imagCoefficientCount,
        imagProductBitOffset,
        imagLinearValue,
        imagLinearInputBitOffset,
        imagLinearBitOffset,
        imagLimbs,
        imagDigits,
        imagControl,
        nullptr,
        0u,
        0ull,
        nullptr,
        0u,
        0ull,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0u,
        0ull,
        nullptr,
        0u,
        0ull,
        nullptr,
        nullptr,
        nullptr,
        descriptors,
        sharedStorage);
}

template <class SharkFloatParams, AlignedUnpackMode Mode, bool MaterializeLimbs>
__device__ void
PrefixAlignedB16CarryTransformsFourWay(cooperative_groups::grid_group &grid,
                                       cooperative_groups::thread_block &block,
                                       DebugGlobalCount<SharkFloatParams> *debugCombo,
                                       uint32_t count,
                                       uint32_t capacity,
                                       const SharkNTT::PlanPrime &plan,
                                       const SharkNTT::RootTables &roots,
                                       const uint64_t *realSpectrum,
                                       uint32_t realCoefficientCount,
                                       uint64_t realProductBitOffset,
                                       const HpSharkFloat<SharkFloatParams> *realLinearValue,
                                       uint32_t realLinearInputBitOffset,
                                       uint64_t realLinearBitOffset,
                                       int64_t *realLimbs,
                                       uint32_t *realDigits,
                                       uint32_t *realControl,
                                       const uint64_t *imagSpectrum,
                                       uint32_t imagCoefficientCount,
                                       uint64_t imagProductBitOffset,
                                       const HpSharkFloat<SharkFloatParams> *imagLinearValue,
                                       uint32_t imagLinearInputBitOffset,
                                       uint64_t imagLinearBitOffset,
                                       int64_t *imagLimbs,
                                       uint32_t *imagDigits,
                                       uint32_t *imagControl,
                                       const uint64_t *dzdcRealSpectrum,
                                       uint32_t dzdcRealCoefficientCount,
                                       uint64_t dzdcRealProductBitOffset,
                                       const HpSharkFloat<SharkFloatParams> *dzdcRealLinearValue,
                                       uint32_t dzdcRealLinearInputBitOffset,
                                       uint64_t dzdcRealLinearBitOffset,
                                       int64_t *dzdcRealLimbs,
                                       uint32_t *dzdcRealDigits,
                                       uint32_t *dzdcRealControl,
                                       const uint64_t *dzdcImagSpectrum,
                                       uint32_t dzdcImagCoefficientCount,
                                       uint64_t dzdcImagProductBitOffset,
                                       const HpSharkFloat<SharkFloatParams> *dzdcImagLinearValue,
                                       uint32_t dzdcImagLinearInputBitOffset,
                                       uint64_t dzdcImagLinearBitOffset,
                                       int64_t *dzdcImagLimbs,
                                       uint32_t *dzdcImagDigits,
                                       uint32_t *dzdcImagControl,
                                       HpSharkReference2PackedCarryPrefixDescriptor *descriptors,
                                       uint64_t *sharedStorage)
{
    PrefixAlignedB16CarryTransformsDLB<SharkFloatParams, Mode, MaterializeLimbs>(
        grid,
        block,
        debugCombo,
        count,
        capacity,
        plan,
        roots,
        realSpectrum,
        realCoefficientCount,
        realProductBitOffset,
        realLinearValue,
        realLinearInputBitOffset,
        realLinearBitOffset,
        realLimbs,
        realDigits,
        realControl,
        imagSpectrum,
        imagCoefficientCount,
        imagProductBitOffset,
        imagLinearValue,
        imagLinearInputBitOffset,
        imagLinearBitOffset,
        imagLimbs,
        imagDigits,
        imagControl,
        dzdcRealSpectrum,
        dzdcRealCoefficientCount,
        dzdcRealProductBitOffset,
        dzdcRealLinearValue,
        dzdcRealLinearInputBitOffset,
        dzdcRealLinearBitOffset,
        dzdcRealLimbs,
        dzdcRealDigits,
        dzdcRealControl,
        dzdcImagSpectrum,
        dzdcImagCoefficientCount,
        dzdcImagProductBitOffset,
        dzdcImagLinearValue,
        dzdcImagLinearInputBitOffset,
        dzdcImagLinearBitOffset,
        dzdcImagLimbs,
        dzdcImagDigits,
        dzdcImagControl,
        descriptors,
        sharedStorage);
}

template <class SharkFloatParams>
__device__ void
FinalizeSignedStream(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugState<SharkFloatParams> *debugStates,
                     uint64_t *carryPrefixShared,
                     HpSharkReference2PackedCarryPrefixDescriptor *carryPrefixDescriptors,
                     HpSharkReference2Workspace<SharkFloatParams> &workspace,
                     uint32_t limbCount,
                     int32_t realExponent,
                     int32_t imagExponent,
                     int32_t dzdcRealExponent,
                     int32_t dzdcImagExponent,
                     bool carryPrefixReady,
                     HpSharkReferenceResults<SharkFloatParams> *combo)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    using Descriptor = HpSharkReference2PackedCarryPrefixDescriptor;
    constexpr uint32_t MaxCapacity = Workspace::MaxFusedLimbs;
    constexpr uint32_t MaxDescriptorWords =
        (Workspace::MaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    constexpr uint32_t MaxControlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    static_assert((MaxCapacity * sizeof(uint64_t)) % alignof(Descriptor) == 0u);
    static_assert(MaxCapacity + MaxDescriptorWords + MaxControlWords <= Workspace::MaxFusedN);
    const uint32_t capacity = workspace.ActiveMaxFusedLimbs;
    const uint32_t descriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    const uint32_t controlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    MattsCudaAssert(capacity <= MaxCapacity);
    MattsCudaAssert(capacity + descriptorWords + controlWords <= workspace.ActiveMaxFusedN);

    // Fused B16 carries remain in the dead Z arenas; avoid copying the full workspace capacity.
    uint64_t *realOutputArena =
        carryPrefixReady ? workspace.ZReal + descriptorWords : workspace.RealOutput;
    int64_t *realLimbs = workspace.RealLimbs;
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(realOutputArena);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(realOutputArena + capacity +
                                                         (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *realOutput = &combo->Multiply.A;

    uint64_t *imagOutputArena =
        carryPrefixReady ? workspace.ZImag + descriptorWords : workspace.ImagOutput;
    int64_t *imagLimbs = workspace.ImagLimbs;
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(imagOutputArena);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(imagOutputArena + capacity +
                                                         (carryPrefixReady ? 0u : descriptorWords));
    HpSharkFloat<SharkFloatParams> *imagOutput = &combo->Multiply.B;

    int64_t *dzdcRealLimbs = nullptr;
    uint32_t *dzdcRealDigits = nullptr;
    uint32_t *dzdcRealControl = nullptr;
    HpSharkFloat<SharkFloatParams> *dzdcRealOutput = nullptr;
    int64_t *dzdcImagLimbs = nullptr;
    uint32_t *dzdcImagDigits = nullptr;
    uint32_t *dzdcImagControl = nullptr;
    HpSharkFloat<SharkFloatParams> *dzdcImagOutput = nullptr;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealLimbs = workspace.DzdcRealLimbs;
        uint64_t *dzdcRealOutputArena =
            carryPrefixReady ? workspace.DzdcReal + descriptorWords : workspace.DzdcRealOutput;
        dzdcRealDigits = reinterpret_cast<uint32_t *>(dzdcRealOutputArena);
        dzdcRealControl = reinterpret_cast<uint32_t *>(dzdcRealOutputArena + capacity +
                                                       (carryPrefixReady ? 0u : descriptorWords));
        dzdcRealOutput = &combo->Multiply.DzdcReal;
        dzdcImagLimbs = workspace.DzdcImagLimbs;
        uint64_t *dzdcImagOutputArena =
            carryPrefixReady ? workspace.DzdcImag + descriptorWords : workspace.DzdcImagOutput;
        dzdcImagDigits = reinterpret_cast<uint32_t *>(dzdcImagOutputArena);
        dzdcImagControl = reinterpret_cast<uint32_t *>(dzdcImagOutputArena + capacity +
                                                       (carryPrefixReady ? 0u : descriptorWords));
        dzdcImagOutput = &combo->Multiply.DzdcImag;
    }

    MattsCudaAssert(limbCount > 0u && limbCount <= capacity);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());

    if (!carryPrefixReady) {
        if (IsLeader<SharkFloatParams>(block)) {
            realControl[FinalizationHighestNonZeroControl] = 0u;
            imagControl[FinalizationHighestNonZeroControl] = 0u;
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                dzdcRealControl[FinalizationHighestNonZeroControl] = 0u;
                dzdcImagControl[FinalizationHighestNonZeroControl] = 0u;
            }
        }
        PrefixCarryTransformsDLB<SharkFloatParams>(grid,
                                                   block,
                                                   limbCount,
                                                   capacity,
                                                   realLimbs,
                                                   realDigits,
                                                   realControl,
                                                   imagLimbs,
                                                   imagDigits,
                                                   imagControl,
                                                   dzdcRealLimbs,
                                                   dzdcRealDigits,
                                                   dzdcRealControl,
                                                   dzdcImagLimbs,
                                                   dzdcImagDigits,
                                                   dzdcImagControl,
                                                   carryPrefixDescriptors,
                                                   carryPrefixShared);
    }

    const uint32_t realDigitLength = realControl[FinalizationDigitLengthControl];
    const uint32_t imagDigitLength = imagControl[FinalizationDigitLengthControl];
    uint32_t maximumDigitLength = realDigitLength > imagDigitLength ? realDigitLength : imagDigitLength;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const uint32_t dzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
        const uint32_t dzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
        maximumDigitLength =
            maximumDigitLength > dzdcRealDigitLength ? maximumDigitLength : dzdcRealDigitLength;
        maximumDigitLength =
            maximumDigitLength > dzdcImagDigitLength ? maximumDigitLength : dzdcImagDigitLength;
    }

    if constexpr (HpShark::DebugChecksums) {
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::SignedCarry1,
                                  realDigits,
                                  realControl[FinalizationDigitLengthControl]);
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::SignedCarry2,
                                  imagDigits,
                                  imagControl[FinalizationDigitLengthControl]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::SignedCarryDzdc1,
                                      dzdcRealDigits,
                                      dzdcRealControl[FinalizationDigitLengthControl]);
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::SignedCarryDzdc2,
                                      dzdcImagDigits,
                                      dzdcImagControl[FinalizationDigitLengthControl]);
        }
    }

    // In (~digits) + 1, the carry reaches the lowest nonzero digit and stops there.
    // Locating that digit avoids a second cross-block carry-prefix scan.
    FindLowestNonZero<SharkFloatParams>(grid,
                                        block,
                                        realDigits,
                                        realControl,
                                        imagDigits,
                                        imagControl,
                                        dzdcRealDigits,
                                        dzdcRealControl,
                                        dzdcImagDigits,
                                        dzdcImagControl,
                                        carryPrefixShared);

    uint32_t realLocalHighest = 0u;
    uint32_t imagLocalHighest = 0u;
    uint32_t dzdcRealLocalHighest = 0u;
    uint32_t dzdcImagLocalHighest = 0u;
    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
        if (realControl[FinalizationNegativeControl] != 0u && index < realDigitLength) {
            const uint32_t lowestNonZero = realControl[FinalizationLowestNonZeroControl];
            if (index < lowestNonZero)
                realDigits[index] = 0u;
            else if (index == lowestNonZero)
                realDigits[index] = 0u - realDigits[index];
            else
                realDigits[index] = ~realDigits[index];
        }
        if (imagControl[FinalizationNegativeControl] != 0u && index < imagDigitLength) {
            const uint32_t lowestNonZero = imagControl[FinalizationLowestNonZeroControl];
            if (index < lowestNonZero)
                imagDigits[index] = 0u;
            else if (index == lowestNonZero)
                imagDigits[index] = 0u - imagDigits[index];
            else
                imagDigits[index] = ~imagDigits[index];
        }
        if (index < realDigitLength && realDigits[index] != 0u)
            realLocalHighest = index + 1u;
        if (index < imagDigitLength && imagDigits[index] != 0u)
            imagLocalHighest = index + 1u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u && index < dzdcRealDigitLength) {
                const uint32_t lowestNonZero = dzdcRealControl[FinalizationLowestNonZeroControl];
                if (index < lowestNonZero)
                    dzdcRealDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcRealDigits[index] = 0u - dzdcRealDigits[index];
                else
                    dzdcRealDigits[index] = ~dzdcRealDigits[index];
            }

            const uint32_t dzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u && index < dzdcImagDigitLength) {
                const uint32_t lowestNonZero = dzdcImagControl[FinalizationLowestNonZeroControl];
                if (index < lowestNonZero)
                    dzdcImagDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcImagDigits[index] = 0u - dzdcImagDigits[index];
                else
                    dzdcImagDigits[index] = ~dzdcImagDigits[index];
            }
            if (index < dzdcRealDigitLength && dzdcRealDigits[index] != 0u)
                dzdcRealLocalHighest = index + 1u;
            if (index < dzdcImagDigitLength && dzdcImagDigits[index] != 0u)
                dzdcImagLocalHighest = index + 1u;
        }
    }

    const uint32_t threadIndex = block.thread_index().x;
    uint32_t *blockMaximum = reinterpret_cast<uint32_t *>(carryPrefixShared);
    if (threadIndex == 0u) {
        blockMaximum[0] = 0u;
        blockMaximum[1] = 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            blockMaximum[2] = 0u;
            blockMaximum[3] = 0u;
        }
    }
    __syncthreads();
    if (realLocalHighest != 0u)
        atomicMax(&blockMaximum[0], realLocalHighest);
    if (imagLocalHighest != 0u)
        atomicMax(&blockMaximum[1], imagLocalHighest);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (dzdcRealLocalHighest != 0u)
            atomicMax(&blockMaximum[2], dzdcRealLocalHighest);
        if (dzdcImagLocalHighest != 0u)
            atomicMax(&blockMaximum[3], dzdcImagLocalHighest);
    }
    __syncthreads();
    if (threadIndex == 0u) {
        atomicMax(&realControl[FinalizationHighestNonZeroControl], blockMaximum[0]);
        atomicMax(&imagControl[FinalizationHighestNonZeroControl], blockMaximum[1]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            atomicMax(&dzdcRealControl[FinalizationHighestNonZeroControl], blockMaximum[2]);
            atomicMax(&dzdcImagControl[FinalizationHighestNonZeroControl], blockMaximum[3]);
        }
    }
    grid.sync();

    if (IsLeader<SharkFloatParams>(block)) {
        uint32_t currentRealDigitLength = realControl[FinalizationDigitLengthControl];
        if (realControl[FinalizationNegativeControl] != 0u &&
            realControl[FinalizationLowestNonZeroControl] == currentRealDigitLength) {
            MattsCudaAssert(currentRealDigitLength < capacity);
            if (currentRealDigitLength < capacity)
                realDigits[currentRealDigitLength++] = 1u;
        }
        realControl[FinalizationDigitLengthControl] = currentRealDigitLength;
        if (realControl[FinalizationHighestNonZeroControl] < currentRealDigitLength &&
            currentRealDigitLength != 0u && realDigits[currentRealDigitLength - 1u] != 0u)
            realControl[FinalizationHighestNonZeroControl] = currentRealDigitLength;

        uint32_t currentImagDigitLength = imagControl[FinalizationDigitLengthControl];
        if (imagControl[FinalizationNegativeControl] != 0u &&
            imagControl[FinalizationLowestNonZeroControl] == currentImagDigitLength) {
            MattsCudaAssert(currentImagDigitLength < capacity);
            if (currentImagDigitLength < capacity)
                imagDigits[currentImagDigitLength++] = 1u;
        }
        imagControl[FinalizationDigitLengthControl] = currentImagDigitLength;
        if (imagControl[FinalizationHighestNonZeroControl] < currentImagDigitLength &&
            currentImagDigitLength != 0u && imagDigits[currentImagDigitLength - 1u] != 0u)
            imagControl[FinalizationHighestNonZeroControl] = currentImagDigitLength;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint32_t currentDzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u &&
                dzdcRealControl[FinalizationLowestNonZeroControl] == currentDzdcRealDigitLength) {
                MattsCudaAssert(currentDzdcRealDigitLength < capacity);
                if (currentDzdcRealDigitLength < capacity)
                    dzdcRealDigits[currentDzdcRealDigitLength++] = 1u;
            }
            dzdcRealControl[FinalizationDigitLengthControl] = currentDzdcRealDigitLength;
            if (dzdcRealControl[FinalizationHighestNonZeroControl] < currentDzdcRealDigitLength &&
                currentDzdcRealDigitLength != 0u &&
                dzdcRealDigits[currentDzdcRealDigitLength - 1u] != 0u)
                dzdcRealControl[FinalizationHighestNonZeroControl] = currentDzdcRealDigitLength;

            uint32_t currentDzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u &&
                dzdcImagControl[FinalizationLowestNonZeroControl] == currentDzdcImagDigitLength) {
                MattsCudaAssert(currentDzdcImagDigitLength < capacity);
                if (currentDzdcImagDigitLength < capacity)
                    dzdcImagDigits[currentDzdcImagDigitLength++] = 1u;
            }
            dzdcImagControl[FinalizationDigitLengthControl] = currentDzdcImagDigitLength;
            if (dzdcImagControl[FinalizationHighestNonZeroControl] < currentDzdcImagDigitLength &&
                currentDzdcImagDigitLength != 0u &&
                dzdcImagDigits[currentDzdcImagDigitLength - 1u] != 0u)
                dzdcImagControl[FinalizationHighestNonZeroControl] = currentDzdcImagDigitLength;
        }
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::FinalAdd1,
                                  realDigits,
                                  realControl[FinalizationHighestNonZeroControl]);
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::FinalAdd2,
                                  imagDigits,
                                  imagControl[FinalizationHighestNonZeroControl]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::FinalAddDzdc1,
                                      dzdcRealDigits,
                                      dzdcRealControl[FinalizationHighestNonZeroControl]);
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::FinalAddDzdc2,
                                      dzdcImagDigits,
                                      dzdcImagControl[FinalizationHighestNonZeroControl]);
        }
    }

    constexpr uint32_t ActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int DesiredBit = (static_cast<int>(ActualDigits) - 1) * 32 + 31;
    if (IsLeader<SharkFloatParams>(block)) {
        const uint32_t realHighestNonZeroPlusOne = realControl[FinalizationHighestNonZeroControl];
        if (realHighestNonZeroPlusOne == 0u) {
            realOutput->Exponent = -100'000'000;
            realOutput->SetNegative(false);
        } else {
            const uint32_t highestNonZero = realHighestNonZeroPlusOne - 1u;
            const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                   CountLeadingZeros(realDigits[highestNonZero]);
            realOutput->Exponent = realExponent + currentBit - DesiredBit;
            realOutput->SetNegative(realControl[FinalizationNegativeControl] != 0u);
        }

        const uint32_t imagHighestNonZeroPlusOne = imagControl[FinalizationHighestNonZeroControl];
        if (imagHighestNonZeroPlusOne == 0u) {
            imagOutput->Exponent = -100'000'000;
            imagOutput->SetNegative(false);
        } else {
            const uint32_t highestNonZero = imagHighestNonZeroPlusOne - 1u;
            const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                   CountLeadingZeros(imagDigits[highestNonZero]);
            imagOutput->Exponent = imagExponent + currentBit - DesiredBit;
            imagOutput->SetNegative(imagControl[FinalizationNegativeControl] != 0u);
        }

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealHighestNonZeroPlusOne =
                dzdcRealControl[FinalizationHighestNonZeroControl];
            if (dzdcRealHighestNonZeroPlusOne == 0u) {
                dzdcRealOutput->Exponent = -100'000'000;
                dzdcRealOutput->SetNegative(false);
            } else {
                const uint32_t highestNonZero = dzdcRealHighestNonZeroPlusOne - 1u;
                const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                       CountLeadingZeros(dzdcRealDigits[highestNonZero]);
                dzdcRealOutput->Exponent = dzdcRealExponent + currentBit - DesiredBit;
                dzdcRealOutput->SetNegative(dzdcRealControl[FinalizationNegativeControl] != 0u);
            }

            const uint32_t dzdcImagHighestNonZeroPlusOne =
                dzdcImagControl[FinalizationHighestNonZeroControl];
            if (dzdcImagHighestNonZeroPlusOne == 0u) {
                dzdcImagOutput->Exponent = -100'000'000;
                dzdcImagOutput->SetNegative(false);
            } else {
                const uint32_t highestNonZero = dzdcImagHighestNonZeroPlusOne - 1u;
                const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                       CountLeadingZeros(dzdcImagDigits[highestNonZero]);
                dzdcImagOutput->Exponent = dzdcImagExponent + currentBit - DesiredBit;
                dzdcImagOutput->SetNegative(dzdcImagControl[FinalizationNegativeControl] != 0u);
            }
        }
    }

    for (uint32_t digit = GridThreadRank(block); digit < ActualDigits; digit += gridSize) {
        const int digitIndex = static_cast<int>(digit);
        {
            const uint32_t highestNonZeroPlusOne = realControl[FinalizationHighestNonZeroControl];
            if (highestNonZeroPlusOne == 0u) {
                realOutput->Digits[digitIndex] = 0u;
            } else {
                const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                       CountLeadingZeros(realDigits[highestNonZero]);
                const int shift = currentBit - DesiredBit;
                realOutput->Digits[digitIndex] =
                    shift > 0 ? FunnelShiftRight(realDigits, digitIndex, magnitudeLength, shift)
                              : FunnelShiftLeft(realDigits, digitIndex, magnitudeLength, -shift);
            }
        }

        {
            const uint32_t highestNonZeroPlusOne = imagControl[FinalizationHighestNonZeroControl];
            if (highestNonZeroPlusOne == 0u) {
                imagOutput->Digits[digitIndex] = 0u;
            } else {
                const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                       CountLeadingZeros(imagDigits[highestNonZero]);
                const int shift = currentBit - DesiredBit;
                imagOutput->Digits[digitIndex] =
                    shift > 0 ? FunnelShiftRight(imagDigits, digitIndex, magnitudeLength, shift)
                              : FunnelShiftLeft(imagDigits, digitIndex, magnitudeLength, -shift);
            }
        }

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            {
                const uint32_t highestNonZeroPlusOne =
                    dzdcRealControl[FinalizationHighestNonZeroControl];
                if (highestNonZeroPlusOne == 0u) {
                    dzdcRealOutput->Digits[digitIndex] = 0u;
                } else {
                    const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                    const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                    const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                           CountLeadingZeros(dzdcRealDigits[highestNonZero]);
                    const int shift = currentBit - DesiredBit;
                    dzdcRealOutput->Digits[digitIndex] =
                        shift > 0 ? FunnelShiftRight(dzdcRealDigits, digitIndex, magnitudeLength, shift)
                                  : FunnelShiftLeft(dzdcRealDigits, digitIndex, magnitudeLength, -shift);
                }
            }

            {
                const uint32_t highestNonZeroPlusOne =
                    dzdcImagControl[FinalizationHighestNonZeroControl];
                if (highestNonZeroPlusOne == 0u) {
                    dzdcImagOutput->Digits[digitIndex] = 0u;
                } else {
                    const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
                    const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
                    const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                           CountLeadingZeros(dzdcImagDigits[highestNonZero]);
                    const int shift = currentBit - DesiredBit;
                    dzdcImagOutput->Digits[digitIndex] =
                        shift > 0 ? FunnelShiftRight(dzdcImagDigits, digitIndex, magnitudeLength, shift)
                                  : FunnelShiftLeft(dzdcImagDigits, digitIndex, magnitudeLength, -shift);
                }
            }
        }
    }

    if constexpr (HpShark::Debug) {
        grid.sync();
        if (IsLeader<SharkFloatParams>(block)) {
            if (realControl[FinalizationHighestNonZeroControl] != 0u) {
                MattsCudaAssert((realOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if (imagControl[FinalizationHighestNonZeroControl] != 0u) {
                MattsCudaAssert((imagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                if (dzdcRealControl[FinalizationHighestNonZeroControl] != 0u) {
                    MattsCudaAssert((dzdcRealOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
                if (dzdcImagControl[FinalizationHighestNonZeroControl] != 0u) {
                    MattsCudaAssert((dzdcImagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
            }
        }
    }
}

template <class SharkFloatParams>
__device__ void
BuildReference2IterationPlan(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Reference2Workspace;
    auto &iterationPlan = workspace.IterationPlan;
    const auto &zReal = combo->Multiply.A;
    const auto &zImag = combo->Multiply.B;
    const auto &cReal = combo->Add.C_A;
    const auto &cImag = combo->Add.E_B;
    const SharkNTT::PlanPrime basePlan = workspace.Plans[0];
    const uint32_t ignoredPrecisionBits = workspace.IgnoredPrecisionBits;
    const uint32_t bitsPerCoefficient = static_cast<uint32_t>(basePlan.b);

    iterationPlan = {};
    iterationPlan.Kind = static_cast<uint32_t>(HpSharkReference2IterationKind::Zero);
    iterationPlan.PlanSlot = 0u;
    iterationPlan.ActiveN = 0u;
    iterationPlan.LimbCount = 0u;
    iterationPlan.Flags = 0u;

    int32_t stateCommonExponent = 0;
    const bool stateBothZero = ResolveAlignedValueExponent(&stateCommonExponent, zReal, zImag);
    const bool stateRealZero = IsZero(zReal);
    const bool stateImagZero = IsZero(zImag);
    const int64_t stateProductExponent64 = static_cast<int64_t>(stateCommonExponent) * 2ll +
                                           2ll * static_cast<int64_t>(ignoredPrecisionBits);
    MattsCudaAssert(stateProductExponent64 >= INT32_MIN && stateProductExponent64 <= INT32_MAX);
    const int32_t stateProductExponent = static_cast<int32_t>(stateProductExponent64);
    const FusedTerm<SharkFloatParams> realProductTerm = MakeAlignedProductTerm<SharkFloatParams>(
        stateBothZero, stateProductExponent, SpectrumId::ZReal, SpectrumId::ZImag);
    const FusedTerm<SharkFloatParams> imagProductTerm = MakeAlignedProductTerm<SharkFloatParams>(
        stateRealZero || stateImagZero, stateProductExponent, SpectrumId::ZReal, SpectrumId::ZImag);
    const FusedTerm<SharkFloatParams> realConstantTerm =
        MakeLinearTerm(cReal, SpectrumId::CReal, false, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> imagConstantTerm =
        MakeLinearTerm(cImag, SpectrumId::CImag, false, ignoredPrecisionBits);
    int32_t realExponent = 0;
    int32_t imagExponent = 0;
    ResolveCommonExponent(&realExponent, realProductTerm, realConstantTerm);
    ResolveCommonExponent(&imagExponent, imagProductTerm, imagConstantTerm);

    int32_t derivativeCommonExponent = 0;
    bool derivativeBothZero = true;
    bool derivativeRealZero = true;
    bool derivativeImagZero = true;
    int32_t derivativeProductExponent = 0;
    FusedTerm<SharkFloatParams> dzdcP1Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZReal, SpectrumId::DzdcReal);
    FusedTerm<SharkFloatParams> dzdcP2Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZImag, SpectrumId::DzdcImag);
    FusedTerm<SharkFloatParams> dzdcP3Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZReal, SpectrumId::DzdcReal);
    FusedTerm<SharkFloatParams> dzdcOneTerm =
        MakeLinearTerm(combo->Add.One, SpectrumId::One, false, ignoredPrecisionBits);
    int32_t dzdcRealExponent = 0;
    int32_t dzdcImagExponent = 0;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeBothZero = ResolveAlignedValueExponent(
            &derivativeCommonExponent, combo->Multiply.DzdcReal, combo->Multiply.DzdcImag);
        derivativeRealZero = IsZero(combo->Multiply.DzdcReal);
        derivativeImagZero = IsZero(combo->Multiply.DzdcImag);
        const int64_t derivativeProductExponent64 = static_cast<int64_t>(stateCommonExponent) +
                                                    static_cast<int64_t>(derivativeCommonExponent) +
                                                    2ll * static_cast<int64_t>(ignoredPrecisionBits);
        MattsCudaAssert(derivativeProductExponent64 >= INT32_MIN &&
                        derivativeProductExponent64 <= INT32_MAX);
        derivativeProductExponent = static_cast<int32_t>(derivativeProductExponent64);
        dzdcP1Term = MakeAlignedProductTerm<SharkFloatParams>(stateRealZero || derivativeRealZero,
                                                              derivativeProductExponent,
                                                              SpectrumId::ZReal,
                                                              SpectrumId::DzdcReal);
        dzdcP2Term = MakeAlignedProductTerm<SharkFloatParams>(stateImagZero || derivativeImagZero,
                                                              derivativeProductExponent,
                                                              SpectrumId::ZImag,
                                                              SpectrumId::DzdcImag);
        dzdcP3Term = MakeAlignedProductTerm<SharkFloatParams>(stateBothZero || derivativeBothZero,
                                                              derivativeProductExponent,
                                                              SpectrumId::ZReal,
                                                              SpectrumId::DzdcReal);
        ResolveCommonExponent(&dzdcRealExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term, dzdcOneTerm);
        ResolveCommonExponent(&dzdcImagExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term);
    }

    const uint64_t stateRealShiftBits =
        stateRealZero ? 0ull : static_cast<uint64_t>(zReal.Exponent - stateCommonExponent);
    const uint64_t stateImagShiftBits =
        stateImagZero ? 0ull : static_cast<uint64_t>(zImag.Exponent - stateCommonExponent);
    const uint64_t stateRealCoefficientShift = stateRealShiftBits / bitsPerCoefficient;
    const uint64_t stateImagCoefficientShift = stateImagShiftBits / bitsPerCoefficient;
    const uint64_t stateRealResidualBitShift = stateRealShiftBits % bitsPerCoefficient;
    const uint64_t stateImagResidualBitShift = stateImagShiftBits % bitsPerCoefficient;
    const uint64_t stateRealInputCoefficients =
        static_cast<uint64_t>(basePlan.L) + (stateRealResidualBitShift != 0ull ? 1ull : 0ull);
    const uint64_t stateImagInputCoefficients =
        static_cast<uint64_t>(basePlan.L) + (stateImagResidualBitShift != 0ull ? 1ull : 0ull);
    const uint64_t stateRealLastCoefficient =
        stateRealCoefficientShift + stateRealInputCoefficients - 1ull;
    const uint64_t stateImagLastCoefficient =
        stateImagCoefficientShift + stateImagInputCoefficients - 1ull;
    const uint64_t stateMaxLastCoefficient = stateRealLastCoefficient > stateImagLastCoefficient
                                                 ? stateRealLastCoefficient
                                                 : stateImagLastCoefficient;
    const uint64_t realRequiredCoefficients =
        stateBothZero ? 0ull : 2ull * stateMaxLastCoefficient + 1ull;
    const uint64_t imagRequiredCoefficients =
        (stateRealZero || stateImagZero) ? 0ull
                                         : stateRealLastCoefficient + stateImagLastCoefficient + 1ull;
    uint64_t requiredCoefficients = realRequiredCoefficients > imagRequiredCoefficients
                                        ? realRequiredCoefficients
                                        : imagRequiredCoefficients;

    uint64_t derivativeRealShiftBits = 0;
    uint64_t derivativeImagShiftBits = 0;
    uint64_t derivativeRealCoefficientShift = 0;
    uint64_t derivativeImagCoefficientShift = 0;
    uint64_t derivativeRealResidualBitShift = 0;
    uint64_t derivativeImagResidualBitShift = 0;
    uint64_t derivativeRealInputCoefficients = 0;
    uint64_t derivativeImagInputCoefficients = 0;
    uint64_t derivativeRealLastCoefficient = 0;
    uint64_t derivativeImagLastCoefficient = 0;
    uint64_t derivativeMaxLastCoefficient = 0;
    uint64_t derivativeP1RequiredCoefficients = 0;
    uint64_t derivativeP2RequiredCoefficients = 0;
    uint64_t derivativeP3RequiredCoefficients = 0;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeRealShiftBits =
            derivativeRealZero
                ? 0ull
                : static_cast<uint64_t>(combo->Multiply.DzdcReal.Exponent - derivativeCommonExponent);
        derivativeImagShiftBits =
            derivativeImagZero
                ? 0ull
                : static_cast<uint64_t>(combo->Multiply.DzdcImag.Exponent - derivativeCommonExponent);
        derivativeRealCoefficientShift = derivativeRealShiftBits / bitsPerCoefficient;
        derivativeImagCoefficientShift = derivativeImagShiftBits / bitsPerCoefficient;
        derivativeRealResidualBitShift = derivativeRealShiftBits % bitsPerCoefficient;
        derivativeImagResidualBitShift = derivativeImagShiftBits % bitsPerCoefficient;
        derivativeRealInputCoefficients =
            static_cast<uint64_t>(basePlan.L) + (derivativeRealResidualBitShift != 0ull ? 1ull : 0ull);
        derivativeImagInputCoefficients =
            static_cast<uint64_t>(basePlan.L) + (derivativeImagResidualBitShift != 0ull ? 1ull : 0ull);
        derivativeRealLastCoefficient =
            derivativeRealCoefficientShift + derivativeRealInputCoefficients - 1ull;
        derivativeImagLastCoefficient =
            derivativeImagCoefficientShift + derivativeImagInputCoefficients - 1ull;
        derivativeMaxLastCoefficient = derivativeRealLastCoefficient > derivativeImagLastCoefficient
                                           ? derivativeRealLastCoefficient
                                           : derivativeImagLastCoefficient;
        derivativeP1RequiredCoefficients =
            dzdcP1Term.IsZero ? 0ull : stateRealLastCoefficient + derivativeRealLastCoefficient + 1ull;
        derivativeP2RequiredCoefficients =
            dzdcP2Term.IsZero ? 0ull : stateImagLastCoefficient + derivativeImagLastCoefficient + 1ull;
        derivativeP3RequiredCoefficients =
            dzdcP3Term.IsZero ? 0ull : stateMaxLastCoefficient + derivativeMaxLastCoefficient + 1ull;
        requiredCoefficients = requiredCoefficients > derivativeP1RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP1RequiredCoefficients;
        requiredCoefficients = requiredCoefficients > derivativeP2RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP2RequiredCoefficients;
        requiredCoefficients = requiredCoefficients > derivativeP3RequiredCoefficients
                                   ? requiredCoefficients
                                   : derivativeP3RequiredCoefficients;
    }

    uint32_t flags = 0u;
    if (!realProductTerm.IsZero)
        flags |= HpSharkReference2PlanRealProduct;
    if (!imagProductTerm.IsZero)
        flags |= HpSharkReference2PlanImagProduct;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (!dzdcP1Term.IsZero)
            flags |= HpSharkReference2PlanDzdcP1;
        if (!dzdcP2Term.IsZero)
            flags |= HpSharkReference2PlanDzdcP2;
        if (!dzdcP3Term.IsZero)
            flags |= HpSharkReference2PlanDzdcP3;
        if (!dzdcOneTerm.IsZero)
            flags |= HpSharkReference2PlanDzdcOne;
    }
    if (!realConstantTerm.IsZero)
        flags |= HpSharkReference2PlanRealLinear;
    if (!imagConstantTerm.IsZero)
        flags |= HpSharkReference2PlanImagLinear;
    iterationPlan.Flags = flags;
    iterationPlan.RealExponent = realExponent;
    iterationPlan.ImagExponent = imagExponent;
    iterationPlan.DzdcRealExponent = dzdcRealExponent;
    iterationPlan.DzdcImagExponent = dzdcImagExponent;
    iterationPlan.RealProductBitOffset =
        realProductTerm.IsZero ? 0ull : static_cast<uint64_t>(realProductTerm.Exponent - realExponent);
    iterationPlan.ImagProductBitOffset =
        imagProductTerm.IsZero ? 0ull : static_cast<uint64_t>(imagProductTerm.Exponent - imagExponent);
    iterationPlan.DzdcRealProductBitOffset =
        dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
            ? 0ull
            : static_cast<uint64_t>(derivativeProductExponent - dzdcRealExponent);
    iterationPlan.DzdcImagProductBitOffset =
        dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
            ? 0ull
            : static_cast<uint64_t>(derivativeProductExponent - dzdcImagExponent);
    iterationPlan.RealLinearBitOffset =
        realConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(realConstantTerm.Exponent - realExponent);
    iterationPlan.ImagLinearBitOffset =
        imagConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(imagConstantTerm.Exponent - imagExponent);
    iterationPlan.DzdcRealLinearBitOffset =
        dzdcOneTerm.IsZero ? 0ull : static_cast<uint64_t>(dzdcOneTerm.Exponent - dzdcRealExponent);

    const bool hasLinearTerm = !realConstantTerm.IsZero || !imagConstantTerm.IsZero ||
                               (SharkFloatParams::EnableNewtonRaphson && !dzdcOneTerm.IsZero);
    if (requiredCoefficients == 0ull) {
        iterationPlan.Kind =
            static_cast<uint32_t>(hasLinearTerm ? HpSharkReference2IterationKind::LinearOnly
                                                : HpSharkReference2IterationKind::Zero);
        iterationPlan.LimbCount = hasLinearTerm ? LinearLimbCount<SharkFloatParams>() : 0u;
        return;
    }

    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN) {
        combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }

    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const uint32_t activeN = requiredN < workspace.ActiveMinFusedN ? workspace.ActiveMinFusedN
                                                                   : static_cast<uint32_t>(requiredN);
    MattsCudaAssert(activeN >= workspace.ActiveMinFusedN);
    MattsCudaAssert(requiredCoefficients <= activeN);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Workspace::MinFusedStages;
    MattsCudaAssert(planSlot < Workspace::PlanCacheEntryCount);
    MattsCudaAssert((workspace.ValidPlanMask & (1u << planSlot)) != 0u);
    const SharkNTT::PlanPrime &plan = workspace.Plans[planSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);

    const uint64_t linearBits =
        static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull - ignoredPrecisionBits;
    uint64_t outputBits =
        iterationPlan.RealProductBitOffset + realRequiredCoefficients * bitsPerCoefficient;
    const uint64_t realLinearBits =
        iterationPlan.RealLinearBitOffset +
        ((flags & HpSharkReference2PlanRealLinear) != 0u ? linearBits : 0ull);
    outputBits = outputBits > realLinearBits ? outputBits : realLinearBits;
    const uint64_t imagProductBits =
        iterationPlan.ImagProductBitOffset + imagRequiredCoefficients * bitsPerCoefficient;
    const uint64_t imagLinearBits =
        iterationPlan.ImagLinearBitOffset +
        ((flags & HpSharkReference2PlanImagLinear) != 0u ? linearBits : 0ull);
    outputBits = outputBits > imagProductBits ? outputBits : imagProductBits;
    outputBits = outputBits > imagLinearBits ? outputBits : imagLinearBits;

    uint64_t derivativeRequiredCoefficients = 0ull;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        derivativeRequiredCoefficients =
            derivativeP1RequiredCoefficients > derivativeP2RequiredCoefficients
                ? (derivativeP1RequiredCoefficients > derivativeP3RequiredCoefficients
                       ? derivativeP1RequiredCoefficients
                       : derivativeP3RequiredCoefficients)
                : (derivativeP2RequiredCoefficients > derivativeP3RequiredCoefficients
                       ? derivativeP2RequiredCoefficients
                       : derivativeP3RequiredCoefficients);
        const uint64_t dzdcRealProductBits =
            iterationPlan.DzdcRealProductBitOffset + derivativeRequiredCoefficients * bitsPerCoefficient;
        const uint64_t dzdcRealLinearBits =
            iterationPlan.DzdcRealLinearBitOffset +
            ((flags & HpSharkReference2PlanDzdcOne) != 0u ? linearBits : 0ull);
        outputBits = outputBits > dzdcRealProductBits ? outputBits : dzdcRealProductBits;
        outputBits = outputBits > dzdcRealLinearBits ? outputBits : dzdcRealLinearBits;
        const uint64_t dzdcImagProductBits =
            iterationPlan.DzdcImagProductBitOffset + derivativeRequiredCoefficients * bitsPerCoefficient;
        outputBits = outputBits > dzdcImagProductBits ? outputBits : dzdcImagProductBits;
    }

    const uint64_t limbCount64 = (outputBits + 31ull) / 32ull + 2ull;
    MattsCudaAssert(limbCount64 <= workspace.ActiveMaxFusedLimbs);
    if (limbCount64 > workspace.ActiveMaxFusedLimbs) {
        combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }

    iterationPlan.Kind = static_cast<uint32_t>(HpSharkReference2IterationKind::Ntt);
    iterationPlan.PlanSlot = planSlot;
    iterationPlan.ActiveN = activeN;
    iterationPlan.LimbCount = static_cast<uint32_t>(limbCount64);
    iterationPlan.ZRealCoefficientShift = static_cast<uint32_t>(stateRealCoefficientShift);
    iterationPlan.ZImagCoefficientShift = static_cast<uint32_t>(stateImagCoefficientShift);
    iterationPlan.DzdcRealCoefficientShift = static_cast<uint32_t>(derivativeRealCoefficientShift);
    iterationPlan.DzdcImagCoefficientShift = static_cast<uint32_t>(derivativeImagCoefficientShift);
    iterationPlan.ZRealResidualBitShift = static_cast<uint32_t>(stateRealResidualBitShift);
    iterationPlan.ZImagResidualBitShift = static_cast<uint32_t>(stateImagResidualBitShift);
    iterationPlan.DzdcRealResidualBitShift = static_cast<uint32_t>(derivativeRealResidualBitShift);
    iterationPlan.DzdcImagResidualBitShift = static_cast<uint32_t>(derivativeImagResidualBitShift);
    MattsCudaAssert(realRequiredCoefficients <= activeN);
    MattsCudaAssert(imagRequiredCoefficients <= activeN);
    iterationPlan.RealCoefficientCount =
        realProductTerm.IsZero ? 0u : static_cast<uint32_t>(realRequiredCoefficients);
    iterationPlan.ImagCoefficientCount =
        imagProductTerm.IsZero ? 0u : static_cast<uint32_t>(imagRequiredCoefficients);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        MattsCudaAssert(derivativeRequiredCoefficients <= activeN);
        const uint32_t derivativeCoefficientCount =
            dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero
                ? 0u
                : static_cast<uint32_t>(derivativeRequiredCoefficients);
        iterationPlan.DzdcRealCoefficientCount = derivativeCoefficientCount;
        iterationPlan.DzdcImagCoefficientCount = derivativeCoefficientCount;
    }
}

template <class SharkFloatParams>
__device__ void
ExecuteReference2Iteration(cooperative_groups::grid_group &grid,
                           cooperative_groups::thread_block &block,
                           uint64_t *sharedData,
                           DebugGlobalCount<SharkFloatParams> *debugCombo,
                           DebugState<SharkFloatParams> *debugStates,
                           uint64_t *carryPrefixShared,
                           HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Reference2Workspace;
    const auto &iterationPlan = workspace.IterationPlan;
    const auto &zReal = combo->Multiply.A;
    const auto &zImag = combo->Multiply.B;
    const auto &cReal = combo->Add.C_A;
    const auto &cImag = combo->Add.E_B;
    const uint32_t flags = iterationPlan.Flags;
    const auto hasFlag = [flags](uint32_t flag) { return (flags & flag) != 0u; };
    const auto kind = static_cast<HpSharkReference2IterationKind>(iterationPlan.Kind);

    if (kind == HpSharkReference2IterationKind::Zero) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            SetZeroBatch(grid,
                         block,
                         &combo->Multiply.A,
                         &combo->Multiply.B,
                         &combo->Multiply.DzdcReal,
                         &combo->Multiply.DzdcImag);
        } else {
            SetZeroBatch(grid, block, &combo->Multiply.A, &combo->Multiply.B);
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                             grid,
                                                             block,
                                                             DebugStatePurpose::Result_Add1,
                                                             combo->Multiply.A,
                                                             DebugStatePurpose::Result_Add2,
                                                             combo->Multiply.B,
                                                             DebugStatePurpose::Result_AddDzdc1,
                                                             combo->Multiply.DzdcReal,
                                                             DebugStatePurpose::Result_AddDzdc2,
                                                             combo->Multiply.DzdcImag);
        } else {
            StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                             grid,
                                                             block,
                                                             DebugStatePurpose::Result_Add1,
                                                             combo->Multiply.A,
                                                             DebugStatePurpose::Result_Add2,
                                                             combo->Multiply.B);
        }
        return;
    }

    const uint32_t limbCount = iterationPlan.LimbCount;
    const uint32_t carryPrefixCapacity = workspace.ActiveMaxFusedLimbs;
    auto *carryPrefixDescriptors =
        reinterpret_cast<HpSharkReference2PackedCarryPrefixDescriptor *>(workspace.ZReal);

    if (kind == HpSharkReference2IterationKind::LinearOnly) {
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            GatherLinearToSignedLimbsBatch<SharkFloatParams>(
                grid,
                block,
                hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                workspace.IgnoredPrecisionBits,
                workspace.RealLimbs,
                hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                workspace.IgnoredPrecisionBits,
                workspace.ImagLimbs,
                hasFlag(HpSharkReference2PlanDzdcOne) ? &combo->Add.One : nullptr,
                workspace.IgnoredPrecisionBits,
                workspace.DzdcRealLimbs,
                nullptr,
                workspace.IgnoredPrecisionBits,
                workspace.DzdcImagLimbs,
                limbCount);
        } else {
            GatherLinearToSignedLimbsBatch(grid,
                                           block,
                                           hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                                           workspace.IgnoredPrecisionBits,
                                           workspace.RealLimbs,
                                           hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                                           workspace.IgnoredPrecisionBits,
                                           workspace.ImagLimbs,
                                           limbCount);
        }
        FinalizeSignedStream<SharkFloatParams>(grid,
                                               block,
                                               debugStates,
                                               carryPrefixShared,
                                               carryPrefixDescriptors,
                                               workspace,
                                               limbCount,
                                               iterationPlan.RealExponent,
                                               iterationPlan.ImagExponent,
                                               iterationPlan.DzdcRealExponent,
                                               iterationPlan.DzdcImagExponent,
                                               false,
                                               combo);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                             grid,
                                                             block,
                                                             DebugStatePurpose::Result_Add1,
                                                             combo->Multiply.A,
                                                             DebugStatePurpose::Result_Add2,
                                                             combo->Multiply.B,
                                                             DebugStatePurpose::Result_AddDzdc1,
                                                             combo->Multiply.DzdcReal,
                                                             DebugStatePurpose::Result_AddDzdc2,
                                                             combo->Multiply.DzdcImag);
        } else {
            StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                             grid,
                                                             block,
                                                             DebugStatePurpose::Result_Add1,
                                                             combo->Multiply.A,
                                                             DebugStatePurpose::Result_Add2,
                                                             combo->Multiply.B);
        }
        return;
    }

    const uint32_t carryPrefixDescriptorWords =
        (workspace.ActiveMaxCarryPrefixParts * sizeof(HpSharkReference2PackedCarryPrefixDescriptor) +
         sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    uint32_t *tempRealDigits =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords);
    uint32_t *tempRealControl =
        reinterpret_cast<uint32_t *>(workspace.ZReal + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempImagDigits =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords);
    uint32_t *tempImagControl =
        reinterpret_cast<uint32_t *>(workspace.ZImag + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempDzdcRealDigits =
        reinterpret_cast<uint32_t *>(workspace.DzdcReal + carryPrefixDescriptorWords);
    uint32_t *tempDzdcRealControl = reinterpret_cast<uint32_t *>(
        workspace.DzdcReal + carryPrefixDescriptorWords + carryPrefixCapacity);
    uint32_t *tempDzdcImagDigits =
        reinterpret_cast<uint32_t *>(workspace.DzdcImag + carryPrefixDescriptorWords);
    uint32_t *tempDzdcImagControl = reinterpret_cast<uint32_t *>(
        workspace.DzdcImag + carryPrefixDescriptorWords + carryPrefixCapacity);

    MattsCudaAssert(kind == HpSharkReference2IterationKind::Ntt);
    const uint32_t activeN = iterationPlan.ActiveN;
    MattsCudaAssert(iterationPlan.PlanSlot <
                    HpSharkReference2Workspace<SharkFloatParams>::PlanCacheEntryCount);
    const SharkNTT::PlanPrime &plan = workspace.Plans[iterationPlan.PlanSlot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[iterationPlan.PlanSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    const uint32_t stageCount = static_cast<uint32_t>(plan.stages);
    const uint32_t largeStageCount = stageCount > 10u ? stageCount - 10u : 0u;

    if constexpr (!HpShark::DebugChecksums) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (largeStageCount == 0u) {
                PackAlignedInputsBatch(grid,
                                       block,
                                       debugCombo,
                                       plan,
                                       roots.Reference2InputScaleR,
                                       &zReal,
                                       workspace.ZReal,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.ZRealCoefficientShift,
                                       iterationPlan.ZRealResidualBitShift,
                                       zReal.GetNegative(),
                                       &zImag,
                                       workspace.ZImag,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.ZImagCoefficientShift,
                                       iterationPlan.ZImagResidualBitShift,
                                       zImag.GetNegative(),
                                       &combo->Multiply.DzdcReal,
                                       workspace.DzdcReal,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.DzdcRealCoefficientShift,
                                       iterationPlan.DzdcRealResidualBitShift,
                                       combo->Multiply.DzdcReal.GetNegative(),
                                       &combo->Multiply.DzdcImag,
                                       workspace.DzdcImag,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.DzdcImagCoefficientShift,
                                       iterationPlan.DzdcImagResidualBitShift,
                                       combo->Multiply.DzdcImag.GetNegative());
            } else if (largeStageCount >= 2u) {
                PackAlignedForwardDIFLargeStages(grid,
                                                 block,
                                                 debugCombo,
                                                 plan,
                                                 roots,
                                                 roots.Reference2InputScaleR,
                                                 &zReal,
                                                 workspace.ZReal,
                                                 workspace.IgnoredPrecisionBits,
                                                 iterationPlan.ZRealCoefficientShift,
                                                 iterationPlan.ZRealResidualBitShift,
                                                 zReal.GetNegative(),
                                                 &zImag,
                                                 workspace.ZImag,
                                                 workspace.IgnoredPrecisionBits,
                                                 iterationPlan.ZImagCoefficientShift,
                                                 iterationPlan.ZImagResidualBitShift,
                                                 zImag.GetNegative(),
                                                 &combo->Multiply.DzdcReal,
                                                 workspace.DzdcReal,
                                                 workspace.IgnoredPrecisionBits,
                                                 iterationPlan.DzdcRealCoefficientShift,
                                                 iterationPlan.DzdcRealResidualBitShift,
                                                 combo->Multiply.DzdcReal.GetNegative(),
                                                 &combo->Multiply.DzdcImag,
                                                 workspace.DzdcImag,
                                                 workspace.IgnoredPrecisionBits,
                                                 iterationPlan.DzdcImagCoefficientShift,
                                                 iterationPlan.DzdcImagResidualBitShift,
                                                 combo->Multiply.DzdcImag.GetNegative());
            } else {
                PackAlignedForwardDIFRadix2(grid,
                                            block,
                                            debugCombo,
                                            plan,
                                            roots,
                                            roots.Reference2InputScaleR,
                                            &zReal,
                                            workspace.ZReal,
                                            workspace.IgnoredPrecisionBits,
                                            iterationPlan.ZRealCoefficientShift,
                                            iterationPlan.ZRealResidualBitShift,
                                            zReal.GetNegative(),
                                            &zImag,
                                            workspace.ZImag,
                                            workspace.IgnoredPrecisionBits,
                                            iterationPlan.ZImagCoefficientShift,
                                            iterationPlan.ZImagResidualBitShift,
                                            zImag.GetNegative(),
                                            &combo->Multiply.DzdcReal,
                                            workspace.DzdcReal,
                                            workspace.IgnoredPrecisionBits,
                                            iterationPlan.DzdcRealCoefficientShift,
                                            iterationPlan.DzdcRealResidualBitShift,
                                            combo->Multiply.DzdcReal.GetNegative(),
                                            &combo->Multiply.DzdcImag,
                                            workspace.DzdcImag,
                                            workspace.IgnoredPrecisionBits,
                                            iterationPlan.DzdcImagCoefficientShift,
                                            iterationPlan.DzdcImagResidualBitShift,
                                            combo->Multiply.DzdcImag.GetNegative());
            }
            SharkNTT::ForwardDIFLargeStages<SharkFloatParams, SharkNTT::Multiway::FourWay>(
                sharedData,
                grid,
                block,
                debugCombo,
                workspace.ZReal,
                workspace.ZImag,
                workspace.DzdcReal,
                workspace.DzdcImag,
                roots,
                largeStageCount >= 2u   ? stageCount - 2u
                : largeStageCount == 1u ? 10u
                                        : stageCount);
            FusedAlignedPointwiseTransform<SharkFloatParams, SharkNTT::Multiway::FourWay>(
                grid,
                block,
                sharedData,
                debugCombo,
                plan,
                roots,
                workspace.ZReal,
                workspace.ZImag,
                workspace.DzdcReal,
                workspace.DzdcImag,
                workspace.RealOutput,
                workspace.ImagOutput,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                hasFlag(HpSharkReference2PlanRealProduct),
                hasFlag(HpSharkReference2PlanImagProduct),
                hasFlag(HpSharkReference2PlanDzdcP1),
                hasFlag(HpSharkReference2PlanDzdcP2),
                hasFlag(HpSharkReference2PlanDzdcP3));
        } else {
            if (largeStageCount == 0u) {
                PackAlignedInputsBatch(grid,
                                       block,
                                       debugCombo,
                                       plan,
                                       roots.Reference2InputScaleR,
                                       &zReal,
                                       workspace.ZReal,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.ZRealCoefficientShift,
                                       iterationPlan.ZRealResidualBitShift,
                                       zReal.GetNegative(),
                                       &zImag,
                                       workspace.ZImag,
                                       workspace.IgnoredPrecisionBits,
                                       iterationPlan.ZImagCoefficientShift,
                                       iterationPlan.ZImagResidualBitShift,
                                       zImag.GetNegative());
            } else if (largeStageCount >= 2u) {
                PackAlignedForwardDIFLargeStages<SharkFloatParams>(grid,
                                                                   block,
                                                                   debugCombo,
                                                                   plan,
                                                                   roots,
                                                                   roots.Reference2InputScaleR,
                                                                   &zReal,
                                                                   workspace.ZReal,
                                                                   workspace.IgnoredPrecisionBits,
                                                                   iterationPlan.ZRealCoefficientShift,
                                                                   iterationPlan.ZRealResidualBitShift,
                                                                   zReal.GetNegative(),
                                                                   &zImag,
                                                                   workspace.ZImag,
                                                                   workspace.IgnoredPrecisionBits,
                                                                   iterationPlan.ZImagCoefficientShift,
                                                                   iterationPlan.ZImagResidualBitShift,
                                                                   zImag.GetNegative(),
                                                                   nullptr,
                                                                   nullptr,
                                                                   0u,
                                                                   0u,
                                                                   0u,
                                                                   false,
                                                                   nullptr,
                                                                   nullptr,
                                                                   0u,
                                                                   0u,
                                                                   0u,
                                                                   false);
            } else {
                PackAlignedForwardDIFRadix2<SharkFloatParams>(grid,
                                                              block,
                                                              debugCombo,
                                                              plan,
                                                              roots,
                                                              roots.Reference2InputScaleR,
                                                              &zReal,
                                                              workspace.ZReal,
                                                              workspace.IgnoredPrecisionBits,
                                                              iterationPlan.ZRealCoefficientShift,
                                                              iterationPlan.ZRealResidualBitShift,
                                                              zReal.GetNegative(),
                                                              &zImag,
                                                              workspace.ZImag,
                                                              workspace.IgnoredPrecisionBits,
                                                              iterationPlan.ZImagCoefficientShift,
                                                              iterationPlan.ZImagResidualBitShift,
                                                              zImag.GetNegative(),
                                                              nullptr,
                                                              nullptr,
                                                              0u,
                                                              0u,
                                                              0u,
                                                              false,
                                                              nullptr,
                                                              nullptr,
                                                              0u,
                                                              0u,
                                                              0u,
                                                              false);
            }
            SharkNTT::ForwardDIFLargeStages<SharkFloatParams, SharkNTT::Multiway::TwoWay>(
                sharedData,
                grid,
                block,
                debugCombo,
                workspace.ZReal,
                workspace.ZImag,
                nullptr,
                nullptr,
                roots,
                largeStageCount >= 2u   ? stageCount - 2u
                : largeStageCount == 1u ? 10u
                                        : stageCount);
            FusedAlignedPointwiseTransform<SharkFloatParams, SharkNTT::Multiway::TwoWay>(
                grid,
                block,
                sharedData,
                debugCombo,
                plan,
                roots,
                workspace.ZReal,
                workspace.ZImag,
                nullptr,
                nullptr,
                workspace.RealOutput,
                workspace.ImagOutput,
                nullptr,
                nullptr,
                hasFlag(HpSharkReference2PlanRealProduct),
                hasFlag(HpSharkReference2PlanImagProduct),
                false,
                false,
                false);
        }

        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            SharkNTT::InverseDITLargeStages<SharkFloatParams, SharkNTT::Multiway::FourWay>(
                sharedData,
                grid,
                block,
                debugCombo,
                workspace.RealOutput,
                workspace.ImagOutput,
                workspace.DzdcRealOutput,
                workspace.DzdcImagOutput,
                roots);
            if (plan.b == 16) {
                PrefixAlignedB16CarryTransformsFourWay<SharkFloatParams,
                                                       AlignedUnpackMode::StandardResidue,
                                                       false>(
                    grid,
                    block,
                    debugCombo,
                    limbCount,
                    carryPrefixCapacity,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    nullptr,
                    tempRealDigits,
                    tempRealControl,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    nullptr,
                    tempImagDigits,
                    tempImagControl,
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    hasFlag(HpSharkReference2PlanDzdcOne) ? &combo->Add.One : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.DzdcRealLinearBitOffset,
                    nullptr,
                    tempDzdcRealDigits,
                    tempDzdcRealControl,
                    workspace.DzdcImagOutput,
                    iterationPlan.DzdcImagCoefficientCount,
                    iterationPlan.DzdcImagProductBitOffset,
                    nullptr,
                    workspace.IgnoredPrecisionBits,
                    0u,
                    nullptr,
                    tempDzdcImagDigits,
                    tempDzdcImagControl,
                    carryPrefixDescriptors,
                    carryPrefixShared);
            } else {
                UnpackAlignedResiduesToSignedLimbsBatch<SharkFloatParams,
                                                        AlignedUnpackMode::StandardResidue>(
                    grid,
                    block,
                    debugCombo,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    hasFlag(HpSharkReference2PlanDzdcOne) ? &combo->Add.One : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.DzdcRealLinearBitOffset,
                    workspace.DzdcRealLimbs,
                    workspace.DzdcImagOutput,
                    iterationPlan.DzdcImagCoefficientCount,
                    iterationPlan.DzdcImagProductBitOffset,
                    nullptr,
                    workspace.IgnoredPrecisionBits,
                    0u,
                    workspace.DzdcImagLimbs,
                    limbCount);
            }
        } else {
            SharkNTT::InverseDITLargeStages<SharkFloatParams, SharkNTT::Multiway::TwoWay>(
                sharedData,
                grid,
                block,
                debugCombo,
                workspace.RealOutput,
                workspace.ImagOutput,
                nullptr,
                nullptr,
                roots);
            if (plan.b == 16) {
                PrefixAlignedB16CarryTransformsTwoWay<SharkFloatParams,
                                                      AlignedUnpackMode::StandardResidue,
                                                      false>(
                    grid,
                    block,
                    debugCombo,
                    limbCount,
                    carryPrefixCapacity,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    nullptr,
                    tempRealDigits,
                    tempRealControl,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    nullptr,
                    tempImagDigits,
                    tempImagControl,
                    carryPrefixDescriptors,
                    carryPrefixShared);
            } else {
                UnpackAlignedResiduesToSignedLimbsBatch<SharkFloatParams,
                                                        AlignedUnpackMode::StandardResidue>(
                    grid,
                    block,
                    debugCombo,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    limbCount);
            }
        }
        if (plan.b != 16)
            grid.sync();
    } else {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            PackAlignedForwardBatch(grid,
                                    block,
                                    sharedData,
                                    debugCombo,
                                    debugStates,
                                    plan,
                                    roots,
                                    &zReal,
                                    workspace.ZReal,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.ZRealCoefficientShift,
                                    iterationPlan.ZRealResidualBitShift,
                                    zReal.GetNegative(),
                                    DebugStatePurpose::Z0XX,
                                    DebugStatePurpose::Z2XX,
                                    &zImag,
                                    workspace.ZImag,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.ZImagCoefficientShift,
                                    iterationPlan.ZImagResidualBitShift,
                                    zImag.GetNegative(),
                                    DebugStatePurpose::Z0YY,
                                    DebugStatePurpose::Z2YY,
                                    &combo->Multiply.DzdcReal,
                                    workspace.DzdcReal,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.DzdcRealCoefficientShift,
                                    iterationPlan.DzdcRealResidualBitShift,
                                    combo->Multiply.DzdcReal.GetNegative(),
                                    DebugStatePurpose::Z0W1,
                                    DebugStatePurpose::Z2W1,
                                    &combo->Multiply.DzdcImag,
                                    workspace.DzdcImag,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.DzdcImagCoefficientShift,
                                    iterationPlan.DzdcImagResidualBitShift,
                                    combo->Multiply.DzdcImag.GetNegative(),
                                    DebugStatePurpose::Z0W2,
                                    DebugStatePurpose::Z2W2);
        } else {
            PackAlignedForwardBatch(grid,
                                    block,
                                    sharedData,
                                    debugCombo,
                                    debugStates,
                                    plan,
                                    roots,
                                    &zReal,
                                    workspace.ZReal,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.ZRealCoefficientShift,
                                    iterationPlan.ZRealResidualBitShift,
                                    zReal.GetNegative(),
                                    DebugStatePurpose::Z0XX,
                                    DebugStatePurpose::Z2XX,
                                    &zImag,
                                    workspace.ZImag,
                                    workspace.IgnoredPrecisionBits,
                                    iterationPlan.ZImagCoefficientShift,
                                    iterationPlan.ZImagResidualBitShift,
                                    zImag.GetNegative(),
                                    DebugStatePurpose::Z0YY,
                                    DebugStatePurpose::Z2YY);
        }

        AccumulateAlignedOutputSpectra(grid,
                                       block,
                                       debugCombo,
                                       debugStates,
                                       plan,
                                       workspace,
                                       hasFlag(HpSharkReference2PlanRealProduct),
                                       hasFlag(HpSharkReference2PlanImagProduct),
                                       hasFlag(HpSharkReference2PlanDzdcP1),
                                       hasFlag(HpSharkReference2PlanDzdcP2),
                                       hasFlag(HpSharkReference2PlanDzdcP3));
        InitializeCarryPrefixTransformsDLB<SharkFloatParams>(
            grid, block, limbCount, carryPrefixCapacity, carryPrefixDescriptors, carryPrefixShared);

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (plan.b == 16) {
                NTTRadix2Batch<SharkFloatParams, true>(sharedData,
                                                       grid,
                                                       block,
                                                       debugCombo,
                                                       workspace.RealOutput,
                                                       workspace.ImagOutput,
                                                       workspace.DzdcRealOutput,
                                                       workspace.DzdcImagOutput,
                                                       static_cast<uint32_t>(plan.N),
                                                       plan.stages,
                                                       roots);
                grid.sync();
                PrefixAlignedB16CarryTransformsFourWay<SharkFloatParams,
                                                       AlignedUnpackMode::NormalizedMontgomery,
                                                       true>(
                    grid,
                    block,
                    debugCombo,
                    limbCount,
                    carryPrefixCapacity,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    tempRealDigits,
                    tempRealControl,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    tempImagDigits,
                    tempImagControl,
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    hasFlag(HpSharkReference2PlanDzdcOne) ? &combo->Add.One : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.DzdcRealLinearBitOffset,
                    workspace.DzdcRealLimbs,
                    tempDzdcRealDigits,
                    tempDzdcRealControl,
                    workspace.DzdcImagOutput,
                    iterationPlan.DzdcImagCoefficientCount,
                    iterationPlan.DzdcImagProductBitOffset,
                    nullptr,
                    workspace.IgnoredPrecisionBits,
                    0u,
                    workspace.DzdcImagLimbs,
                    tempDzdcImagDigits,
                    tempDzdcImagControl,
                    carryPrefixDescriptors,
                    carryPrefixShared);
                StoreReference2DebugStateBatch<SharkFloatParams>(
                    debugStates,
                    grid,
                    block,
                    DebugStatePurpose::UnpackXX,
                    reinterpret_cast<const uint64_t *>(workspace.RealLimbs),
                    DebugStatePurpose::UnpackYY,
                    reinterpret_cast<const uint64_t *>(workspace.ImagLimbs),
                    DebugStatePurpose::UnpackW0,
                    reinterpret_cast<const uint64_t *>(workspace.DzdcRealLimbs),
                    DebugStatePurpose::UnpackW1,
                    reinterpret_cast<const uint64_t *>(workspace.DzdcImagLimbs),
                    limbCount);
            } else {
                InverseAlignedSpectraToSignedLimbsBatch<SharkFloatParams>(
                    grid,
                    block,
                    sharedData,
                    debugCombo,
                    debugStates,
                    plan,
                    roots,
                    limbCount,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    DebugStatePurpose::UnpackXX,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    DebugStatePurpose::UnpackYY,
                    workspace.DzdcRealOutput,
                    iterationPlan.DzdcRealCoefficientCount,
                    iterationPlan.DzdcRealProductBitOffset,
                    hasFlag(HpSharkReference2PlanDzdcOne) ? &combo->Add.One : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.DzdcRealLinearBitOffset,
                    workspace.DzdcRealLimbs,
                    DebugStatePurpose::UnpackW0,
                    workspace.DzdcImagOutput,
                    iterationPlan.DzdcImagCoefficientCount,
                    iterationPlan.DzdcImagProductBitOffset,
                    nullptr,
                    workspace.IgnoredPrecisionBits,
                    0u,
                    workspace.DzdcImagLimbs,
                    DebugStatePurpose::UnpackW1);
            }
        } else {
            if (plan.b == 16) {
                NTTRadix2Batch<SharkFloatParams, true>(sharedData,
                                                       grid,
                                                       block,
                                                       debugCombo,
                                                       workspace.RealOutput,
                                                       workspace.ImagOutput,
                                                       static_cast<uint32_t>(plan.N),
                                                       plan.stages,
                                                       roots);
                grid.sync();
                PrefixAlignedB16CarryTransformsTwoWay<SharkFloatParams,
                                                      AlignedUnpackMode::NormalizedMontgomery,
                                                      true>(
                    grid,
                    block,
                    debugCombo,
                    limbCount,
                    carryPrefixCapacity,
                    plan,
                    roots,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    tempRealDigits,
                    tempRealControl,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    tempImagDigits,
                    tempImagControl,
                    carryPrefixDescriptors,
                    carryPrefixShared);
                StoreReference2DebugStateBatch<SharkFloatParams>(
                    debugStates,
                    grid,
                    block,
                    DebugStatePurpose::UnpackXX,
                    reinterpret_cast<const uint64_t *>(workspace.RealLimbs),
                    DebugStatePurpose::UnpackYY,
                    reinterpret_cast<const uint64_t *>(workspace.ImagLimbs),
                    limbCount);
            } else {
                InverseAlignedSpectraToSignedLimbsBatch(
                    grid,
                    block,
                    sharedData,
                    debugCombo,
                    debugStates,
                    plan,
                    roots,
                    limbCount,
                    workspace.RealOutput,
                    iterationPlan.RealCoefficientCount,
                    iterationPlan.RealProductBitOffset,
                    hasFlag(HpSharkReference2PlanRealLinear) ? &cReal : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.RealLinearBitOffset,
                    workspace.RealLimbs,
                    DebugStatePurpose::UnpackXX,
                    workspace.ImagOutput,
                    iterationPlan.ImagCoefficientCount,
                    iterationPlan.ImagProductBitOffset,
                    hasFlag(HpSharkReference2PlanImagLinear) ? &cImag : nullptr,
                    workspace.IgnoredPrecisionBits,
                    iterationPlan.ImagLinearBitOffset,
                    workspace.ImagLimbs,
                    DebugStatePurpose::UnpackYY);
            }
        }
    }

    FinalizeSignedStream<SharkFloatParams>(grid,
                                           block,
                                           debugStates,
                                           carryPrefixShared,
                                           carryPrefixDescriptors,
                                           workspace,
                                           limbCount,
                                           iterationPlan.RealExponent,
                                           iterationPlan.ImagExponent,
                                           iterationPlan.DzdcRealExponent,
                                           iterationPlan.DzdcImagExponent,
                                           plan.b == 16,
                                           combo);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                         grid,
                                                         block,
                                                         DebugStatePurpose::Result_AddDzdc1,
                                                         combo->Multiply.DzdcReal,
                                                         DebugStatePurpose::Result_AddDzdc2,
                                                         combo->Multiply.DzdcImag,
                                                         DebugStatePurpose::Result_Add1,
                                                         combo->Multiply.A,
                                                         DebugStatePurpose::Result_Add2,
                                                         combo->Multiply.B);
    } else {
        StoreReference2DebugValueBatch<SharkFloatParams>(debugStates,
                                                         grid,
                                                         block,
                                                         DebugStatePurpose::Result_Add1,
                                                         combo->Multiply.A,
                                                         DebugStatePurpose::Result_Add2,
                                                         combo->Multiply.B);
    }
}

template <class SharkFloatParams>
__device__ void
UpdateD2(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        using Hdr = typename SharkFloatParams::Float;
        const Hdr zr = ToNormalizedHDRFloat(combo->Multiply.A);
        const Hdr zi = ToNormalizedHDRFloat(combo->Multiply.B);
        const Hdr dzr = ToNormalizedHDRFloat(combo->Multiply.DzdcReal);
        const Hdr dzi = ToNormalizedHDRFloat(combo->Multiply.DzdcImag);
        Hdr dz2r = dzr * dzr - dzi * dzi;
        HdrReduce(dz2r);
        Hdr dz2i = Hdr{2.0f} * (dzr * dzi);
        HdrReduce(dz2i);
        Hdr zd2r = zr * combo->d2Real - zi * combo->d2Imag;
        HdrReduce(zd2r);
        Hdr zd2i = zr * combo->d2Imag + zi * combo->d2Real;
        HdrReduce(zd2i);
        Hdr sumr = dz2r + zd2r;
        HdrReduce(sumr);
        Hdr sumi = dz2i + zd2i;
        HdrReduce(sumi);
        combo->d2Real = Hdr{2.0f} * sumr;
        combo->d2Imag = Hdr{2.0f} * sumi;
    }
}

template <class SharkFloatParams>
__device__ bool
CheckPeriodicity(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t iteration)
{
    if constexpr (!SharkFloatParams::EnablePeriodicity) {
        (void)combo;
        (void)iteration;
        return false;
    } else {
        using Hdr = typename SharkFloatParams::Float;
        Hdr zx = ToNormalizedHDRFloat(combo->Multiply.A);
        Hdr zy = ToNormalizedHDRFloat(combo->Multiply.B);
        if (iteration < HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters) {
            combo->OutputIters[iteration].x = zx;
            combo->OutputIters[iteration].y = zy;
        }
        HdrReduce(combo->dzdcX);
        const Hdr dxAbs = HdrAbs(combo->dzdcX);
        HdrReduce(combo->dzdcY);
        const Hdr dyAbs = HdrAbs(combo->dzdcY);
        HdrReduce(zx);
        const Hdr zxAbs = HdrAbs(zx);
        HdrReduce(zy);
        const Hdr zyAbs = HdrAbs(zy);
        const Hdr n2 = HdrMaxPositiveReduced(zxAbs, zyAbs);
        const Hdr n3 = combo->RadiusY * HdrMaxPositiveReduced(dxAbs, dyAbs) * Hdr{2.0f};
        if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
            combo->PeriodicityStatus = PeriodicityResult::PeriodFound;
            ++combo->OutputIterCount;
            return true;
        }
        const Hdr dx = combo->dzdcX;
        combo->dzdcX = Hdr{2.0f} * (zx * combo->dzdcX - zy * combo->dzdcY) + Hdr{1.0f};
        combo->dzdcY = Hdr{2.0f} * (zx * combo->dzdcY + zy * dx);
        const Hdr cx = ToNormalizedHDRFloat(combo->Add.C_A);
        const Hdr cy = ToNormalizedHDRFloat(combo->Add.E_B);
        const Hdr tx = zx + cx;
        const Hdr ty = zy + cy;
        const Hdr size = tx * tx + ty * ty;
        if (HdrCompareToBothPositiveReducedGT(size, Hdr{256.0f})) {
            combo->PeriodicityStatus = PeriodicityResult::Escaped;
            ++combo->OutputIterCount;
            return true;
        }
        (void)iteration;
        return false;
    }
}

} // namespace Reference2Detail

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReference2SetupKernel(HpSharkReference2Workspace<SharkFloatParams> *workspace,
                                 uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    DebugGlobalCount<SharkFloatParams> *debugCombo = nullptr;
    DebugState<SharkFloatParams> *debugStates = nullptr;
    if constexpr (HpShark::DebugGlobalState) {
        debugCombo = reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalGlobalSyncSpace]);
        if (Reference2Detail::IsLeader<SharkFloatParams>(block))
            debugCombo->DebugMultiplyErase();
    }
    if constexpr (HpShark::DebugChecksums) {
        debugStates = reinterpret_cast<DebugState<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalChecksumsOffset]);
        EraseAllDebugStates(debugStates, grid, block);
    }

    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    for (uint32_t stage = workspace->ActiveMinFusedStages; stage <= workspace->ActiveMaxFusedStages;
         ++stage) {
        const uint32_t activeN = 1u << stage;
        Reference2Detail::GenerateCachedPlan<SharkFloatParams>(
            grid, block, debugCombo, activeN, *workspace);
    }

    const uint32_t firstSlot = workspace->ActiveMinFusedStages - Workspace::MinFusedStages;
    const uint32_t activePlanMask = workspace->ActivePlanCacheEntryCount == 32u
                                        ? ~0u
                                        : (1u << workspace->ActivePlanCacheEntryCount) - 1u;
    const uint32_t fullPlanMask = activePlanMask << firstSlot;
    Reference2Detail::MattsCudaAssert(workspace->ValidPlanMask == fullPlanMask);
}

template <class SharkFloatParams>
__global__ void
__maxnreg__(HpShark::RegisterLimit)
    HpSharkReference2GpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ __align__(16) uint64_t sharedData[];
    __shared__ uint64_t carryPrefixShared[2u * Reference2Detail::CarryPrefixMaxWarps];
    const bool leader = Reference2Detail::IsLeader<SharkFloatParams>(block);
    DebugGlobalCount<SharkFloatParams> *debugCombo = nullptr;
    DebugState<SharkFloatParams> *debugStates = nullptr;
    if constexpr (HpShark::DebugGlobalState) {
        const auto offset = HpShark::AdditionalGlobalSyncSpace;
        debugCombo = reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempData[offset]);
        if (leader)
            debugCombo->DebugMultiplyErase();
    }
    if constexpr (HpShark::DebugChecksums) {
        debugStates = reinterpret_cast<DebugState<SharkFloatParams> *>(
            &tempData[HpShark::AdditionalChecksumsOffset]);
        EraseAllDebugStates(debugStates, grid, block);
    }

    if (leader) {
        combo->OutputIterCount = 0;
        combo->PeriodicityStatus = PeriodicityResult::Continue;
    }

    Reference2Detail::StoreReference2DebugValueBatch<SharkFloatParams>(
        debugStates,
        grid,
        block,
        DebugStatePurpose::ReferenceEntryZReal,
        combo->Multiply.A,
        DebugStatePurpose::ReferenceEntryZImag,
        combo->Multiply.B,
        DebugStatePurpose::ReferenceEntryCReal,
        combo->Add.C_A,
        DebugStatePurpose::ReferenceEntryCImag,
        combo->Add.E_B);

    for (uint64_t iteration = 0; iteration < combo->MaxRuntimeIters; ++iteration) {
        if (leader)
            Reference2Detail::CheckPeriodicity<SharkFloatParams>(combo, iteration);
        if (leader && combo->PeriodicityStatus == PeriodicityResult::Continue)
            Reference2Detail::BuildReference2IterationPlan<SharkFloatParams>(combo);
        grid.sync();
        if (combo->PeriodicityStatus != PeriodicityResult::Continue)
            break;

        if (leader)
            Reference2Detail::UpdateD2<SharkFloatParams>(combo);

        Reference2Detail::ExecuteReference2Iteration<SharkFloatParams>(
            grid, block, sharedData, debugCombo, debugStates, carryPrefixShared, combo);
        // Publish every output and PeriodicityStatus before any thread consumes them.
        grid.sync();
        if (combo->PeriodicityStatus == PeriodicityResult::Unknown)
            break;

        if (leader)
            ++combo->OutputIterCount;
    }

    Reference2Detail::StoreReference2DebugValueBatch<SharkFloatParams>(
        debugStates,
        grid,
        block,
        DebugStatePurpose::ReferenceExitZReal,
        combo->Multiply.A,
        DebugStatePurpose::ReferenceExitZImag,
        combo->Multiply.B);
}

template <class SharkFloatParams>
void
ComputeHpSharkReference2Setup(const HpShark::LaunchParams &launchParams,
                              cudaStream_t &stream,
                              void *kernelArgs[])
{
    constexpr auto SharedMemSize = 0u;
    const cudaError_t attribute = cudaFuncSetAttribute(HpSharkReference2SetupKernel<SharkFloatParams>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       SharedMemSize);
    if (attribute != cudaSuccess) {
        std::ostringstream message;
        message << "cudaFuncSetAttribute(HpSharkReference2SetupKernel) failed: "
                << cudaGetErrorString(attribute);
        throw FractalSharkSeriousException(message.str());
    }

    HpShark::LaunchParams resolved{launchParams};
    if (resolved.NumBlocks == 0) {
        HpShark::CudaLaunchConfig config;
        const cudaError_t result = config.compute(
            reinterpret_cast<const void *>(HpSharkReference2SetupKernel<SharkFloatParams>),
            SharedMemSize,
            resolved);
        if (result != cudaSuccess) {
            std::ostringstream message;
            message << "LaunchConfig.compute(HpSharkReference2SetupKernel) failed: "
                    << cudaGetErrorString(result);
            throw FractalSharkSeriousException(message.str());
        }
    }

    const cudaError_t launch = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(HpSharkReference2SetupKernel<SharkFloatParams>),
        dim3(resolved.NumBlocks),
        dim3(resolved.ThreadsPerBlock),
        kernelArgs,
        SharedMemSize,
        stream);
    if (launch != cudaSuccess) {
        std::ostringstream message;
        message << "cudaLaunchCooperativeKernel(HpSharkReference2SetupKernel) failed: "
                << cudaGetErrorString(launch) << " | blocks=" << resolved.NumBlocks
                << " threads=" << resolved.ThreadsPerBlock;
        throw FractalSharkSeriousException(message.str());
    }
    const cudaError_t immediate = cudaGetLastError();
    if (immediate != cudaSuccess) {
        std::ostringstream message;
        message << "cudaGetLastError() after HpSharkReference2SetupKernel launch failed: "
                << cudaGetErrorString(immediate);
        throw FractalSharkSeriousException(message.str());
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        std::ostringstream message;
        message << "cudaDeviceSynchronize() after HpSharkReference2SetupKernel failed: "
                << cudaGetErrorString(synchronized);
        throw FractalSharkSeriousException(message.str());
    }
}

template <class SharkFloatParams>
void
ComputeHpSharkReference2GpuLoop(const HpShark::LaunchParams &launchParams,
                                cudaStream_t &stream,
                                void *kernelArgs[])
{
    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();
    if constexpr (HpShark::CustomStream) {
        const cudaError_t attribute = cudaFuncSetAttribute(HpSharkReference2GpuLoop<SharkFloatParams>,
                                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                           SharedMemSize);
        if (attribute != cudaSuccess) {
            std::ostringstream message;
            message << "cudaFuncSetAttribute(HpSharkReference2GpuLoop) failed: "
                    << cudaGetErrorString(attribute);
            throw FractalSharkSeriousException(message.str());
        }
    }

    HpShark::LaunchParams resolved{launchParams};
    if (resolved.NumBlocks == 0) {
        HpShark::CudaLaunchConfig config;
        const cudaError_t result =
            config.compute(reinterpret_cast<const void *>(HpSharkReference2GpuLoop<SharkFloatParams>),
                           SharedMemSize,
                           resolved);
        if (result != cudaSuccess) {
            std::ostringstream message;
            message << "LaunchConfig.compute(HpSharkReference2GpuLoop) failed: "
                    << cudaGetErrorString(result);
            throw FractalSharkSeriousException(message.str());
        }
    } else {
        int device = 0;
        const cudaError_t getDevice = cudaGetDevice(&device);
        if (getDevice != cudaSuccess) {
            std::ostringstream message;
            message << "cudaGetDevice for HpSharkReference2GpuLoop failed: "
                    << cudaGetErrorString(getDevice);
            throw FractalSharkSeriousException(message.str());
        }

        int maxThreadsPerBlock = 0;
        const cudaError_t getLimit =
            cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
        if (getLimit != cudaSuccess) {
            std::ostringstream message;
            message << "cudaDeviceGetAttribute(cudaDevAttrMaxThreadsPerBlock) failed: "
                    << cudaGetErrorString(getLimit);
            throw FractalSharkSeriousException(message.str());
        }

        if (resolved.NumBlocks < 1 || resolved.ThreadsPerBlock < 32 ||
            (resolved.ThreadsPerBlock & 31) != 0 || resolved.ThreadsPerBlock > maxThreadsPerBlock) {
            std::ostringstream message;
            message << "Invalid explicit HpSharkReference2GpuLoop launch shape: blocks="
                    << resolved.NumBlocks << " threads=" << resolved.ThreadsPerBlock
                    << " (threads must be a warp multiple from 32 through " << maxThreadsPerBlock << ")";
            throw FractalSharkSeriousException(message.str());
        }
    }

    const cudaError_t launch =
        cudaLaunchCooperativeKernel(reinterpret_cast<void *>(HpSharkReference2GpuLoop<SharkFloatParams>),
                                    dim3(resolved.NumBlocks),
                                    dim3(resolved.ThreadsPerBlock),
                                    kernelArgs,
                                    SharedMemSize,
                                    stream);
    if (launch != cudaSuccess) {
        std::ostringstream message;
        message << "cudaLaunchCooperativeKernel(HpSharkReference2GpuLoop) failed: "
                << cudaGetErrorString(launch) << " | blocks=" << resolved.NumBlocks
                << " threads=" << resolved.ThreadsPerBlock;
        throw FractalSharkSeriousException(message.str());
    }
    const cudaError_t immediate = cudaGetLastError();
    if (immediate != cudaSuccess) {
        std::ostringstream message;
        message << "cudaGetLastError() after HpSharkReference2GpuLoop launch failed: "
                << cudaGetErrorString(immediate);
        throw FractalSharkSeriousException(message.str());
    }
    const cudaError_t synchronized = cudaDeviceSynchronize();
    if (synchronized != cudaSuccess) {
        std::ostringstream message;
        message << "cudaDeviceSynchronize() after HpSharkReference2GpuLoop failed: "
                << cudaGetErrorString(synchronized);
        throw FractalSharkSeriousException(message.str());
    }
}
