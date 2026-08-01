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

template <class SharkFloatParams, int OutputCount>
__device__ void
SetZeroBatch(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             HpSharkFloat<SharkFloatParams> *const (&outputs)[OutputCount])
{
    constexpr uint32_t DigitCount = SharkFloatParams::GlobalNumUint32;
    constexpr uint32_t TotalDigits = OutputCount * DigitCount;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t flatIndex = GridThreadRank(block); flatIndex < TotalDigits; flatIndex += gridSize) {
        const uint32_t outputIndex = flatIndex / DigitCount;
        const uint32_t digitIndex = flatIndex % DigitCount;
        outputs[outputIndex]->Digits[digitIndex] = 0u;
    }

    if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
        for (int outputIndex = 0; outputIndex < OutputCount; ++outputIndex) {
            outputs[outputIndex]->Exponent = -100'000'000;
            outputs[outputIndex]->SetNegative(false);
        }
    }
}

template <class SharkFloatParams>
__device__ void
SetZero(cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        HpSharkFloat<SharkFloatParams> *out)
{
    HpSharkFloat<SharkFloatParams> *outputs[1] = {out};
    SetZeroBatch<SharkFloatParams, 1>(grid, block, outputs);
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
    // Only whole base-2^b chunks move the polynomial support. The residual bit shift scales
    // coefficients in place, and its overflow is resolved by the post-inverse carry pass.
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

template <int BatchSize>
__device__ void
BitReverseInplace64Batch(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         uint64_t *const values[BatchSize],
                         uint32_t n,
                         uint32_t stages)
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    if constexpr (BatchSize == 1) {
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::OneWay>(
            grid, block, values[0], nullptr, nullptr, nullptr, n, stages);
    } else if constexpr (BatchSize == 2) {
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::TwoWay>(
            grid, block, values[0], values[1], nullptr, nullptr, n, stages);
    } else if constexpr (BatchSize == 3) {
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::ThreeWay>(
            grid, block, values[0], values[1], values[2], nullptr, n, stages);
    } else {
        SharkNTT::BitReverseInplace64_GridStride<SharkNTT::Multiway::FourWay>(
            grid, block, values[0], values[1], values[2], values[3], n, stages);
    }
}

template <class SharkFloatParams, bool Inverse, int BatchSize>
__device__ void
NTTRadix2Batch(uint64_t *sharedData,
               cooperative_groups::grid_group &grid,
               cooperative_groups::thread_block &block,
               DebugGlobalCount<SharkFloatParams> *debugCombo,
               uint64_t *const values[BatchSize],
               uint32_t n,
               uint32_t stages,
               SharkNTT::RootTables &roots)
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == n);
    MattsCudaAssert(static_cast<uint32_t>(roots.stages) == stages);
    if constexpr (BatchSize == 1) {
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::OneWay, Inverse>(
            sharedData, grid, block, debugCombo, nullptr, values[0], nullptr, nullptr, nullptr, roots);
    } else if constexpr (BatchSize == 2) {
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::TwoWay, Inverse>(
            sharedData, grid, block, debugCombo, nullptr, values[0], values[1], nullptr, nullptr, roots);
    } else if constexpr (BatchSize == 3) {
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::ThreeWay, Inverse>(
            sharedData,
            grid,
            block,
            debugCombo,
            nullptr,
            values[0],
            values[1],
            values[2],
            nullptr,
            roots);
    } else {
        SharkNTT::NTTRadix2_GridStride<SharkFloatParams, SharkNTT::Multiway::FourWay, Inverse>(
            sharedData,
            grid,
            block,
            debugCombo,
            nullptr,
            values[0],
            values[1],
            values[2],
            values[3],
            roots);
    }
}

template <class SharkFloatParams, int BatchSize>
__device__ void
PackTwistForwardBatch(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      uint64_t *sharedData,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      DebugState<SharkFloatParams> *debugStates,
                      const HpSharkFloat<SharkFloatParams> *const values[BatchSize],
                      const SharkNTT::PlanPrime &plan,
                      SharkNTT::RootTables &roots,
                      uint64_t *const outputs[BatchSize],
                      uint32_t inputBitOffset,
                      const DebugStatePurpose packedPurposes[BatchSize],
                      const DebugStatePurpose forwardPurposes[BatchSize])
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            const uint64_t coefficient = i < static_cast<uint32_t>(plan.L)
                                             ? ReadBitsSimple(*values[buffer],
                                                              static_cast<int64_t>(inputBitOffset) +
                                                                  static_cast<int64_t>(i) * plan.b,
                                                              plan.b)
                                             : 0;
            const uint64_t mont = SharkNTT::ToMontgomery<SharkFloatParams>(
                grid, block, debugCombo, coefficient % SharkNTT::MagicPrime);
            outputs[buffer][i] = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, mont, roots.psi_pows[i]);
        }
    }
    grid.sync();
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        StoreReference2DebugState(
            debugStates, grid, block, packedPurposes[buffer], outputs[buffer], activeN);
    }
    BitReverseInplace64Batch<BatchSize>(
        grid, block, outputs, activeN, static_cast<uint32_t>(plan.stages));
    grid.sync();
    NTTRadix2Batch<SharkFloatParams, false, BatchSize>(
        sharedData, grid, block, debugCombo, outputs, activeN, plan.stages, roots);
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        StoreReference2DebugState(
            debugStates, grid, block, forwardPurposes[buffer], outputs[buffer], activeN);
    }
}

template <class SharkFloatParams>
__device__ uint64_t
PsiPowerMont(cooperative_groups::grid_group &grid,
             cooperative_groups::thread_block &block,
             DebugGlobalCount<SharkFloatParams> *debugCombo,
             const SharkNTT::PlanPrime &plan,
             const SharkNTT::RootTables &roots,
             uint64_t exponent)
{
    const uint64_t reduced = exponent % (2ull * static_cast<uint64_t>(plan.N));
    if (reduced < static_cast<uint64_t>(plan.N))
        return roots.psi_pows[reduced];
    return SubPSerial(SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 0),
                      roots.psi_pows[reduced - static_cast<uint64_t>(plan.N)]);
}

struct SpectrumAlignment {
    bool IsZero;
    bool IsNegative;
    uint64_t ChunkShift;
    uint64_t BitScale;
};

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly SpectrumAlignment
MakeSpectrumAlignment(cooperative_groups::grid_group &grid,
                      cooperative_groups::thread_block &block,
                      DebugGlobalCount<SharkFloatParams> *debugCombo,
                      const SharkNTT::PlanPrime &plan,
                      int32_t commonExponent,
                      FusedTerm<SharkFloatParams> term)
{
    if (term.IsZero)
        return {true, false, 0, 0};

    const uint64_t shiftBits = static_cast<uint64_t>(term.Exponent - commonExponent);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    return {false,
            term.IsNegative,
            shiftBits / static_cast<uint64_t>(plan.b),
            SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1ull << bitShift)};
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint64_t
ScaleSpectrumCoefficient(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         DebugGlobalCount<SharkFloatParams> *debugCombo,
                         const SharkNTT::PlanPrime &plan,
                         const SharkNTT::RootTables &roots,
                         const SpectrumAlignment &alignment,
                         uint64_t oneMont,
                         uint64_t value,
                         uint32_t index)
{
    const uint64_t chunkScale =
        alignment.ChunkShift == 0
            ? oneMont
            : PsiPowerMont<SharkFloatParams>(
                  grid, block, debugCombo, plan, roots, alignment.ChunkShift * (1ull + 2ull * index));
    const uint64_t scale = SharkNTT::MontgomeryMul<SharkFloatParams>(
        grid, block, debugCombo, chunkScale, alignment.BitScale);
    return SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, value, scale);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint64_t
AccumulateSpectrumCoefficient(uint64_t accumulator, uint64_t value, bool negative)
{
    return negative ? SubPSerial(accumulator, value) : AddPSerial(accumulator, value);
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
AccumulateFixedOutputSpectra(cooperative_groups::grid_group &grid,
                             cooperative_groups::thread_block &block,
                             DebugGlobalCount<SharkFloatParams> *debugCombo,
                             DebugState<SharkFloatParams> *debugStates,
                             const SharkNTT::PlanPrime &plan,
                             const SharkNTT::RootTables &roots,
                             HpSharkReference2Workspace<SharkFloatParams> &workspace,
                             const HpSharkReference2ConstantSpectra &constantSpectra,
                             int32_t realExponent,
                             FusedTerm<SharkFloatParams> realZSquareTerm,
                             FusedTerm<SharkFloatParams> realNegativeZImagSquareTerm,
                             FusedTerm<SharkFloatParams> realConstantTerm,
                             int32_t imagExponent,
                             FusedTerm<SharkFloatParams> imagDoubleProductTerm,
                             FusedTerm<SharkFloatParams> imagConstantTerm,
                             int32_t dzdcRealExponent,
                             FusedTerm<SharkFloatParams> dzdcRealZRealTerm,
                             FusedTerm<SharkFloatParams> dzdcRealNegativeZImagTerm,
                             FusedTerm<SharkFloatParams> dzdcRealOneTerm,
                             int32_t dzdcImagExponent,
                             FusedTerm<SharkFloatParams> dzdcImagZImagTerm,
                             FusedTerm<SharkFloatParams> dzdcImagZRealTerm)
{
    const SpectrumAlignment realZSquare =
        MakeSpectrumAlignment(grid, block, debugCombo, plan, realExponent, realZSquareTerm);
    const SpectrumAlignment realNegativeZImagSquare =
        MakeSpectrumAlignment(grid, block, debugCombo, plan, realExponent, realNegativeZImagSquareTerm);
    const SpectrumAlignment realConstant =
        MakeSpectrumAlignment(grid, block, debugCombo, plan, realExponent, realConstantTerm);
    const SpectrumAlignment imagDoubleProduct =
        MakeSpectrumAlignment(grid, block, debugCombo, plan, imagExponent, imagDoubleProductTerm);
    const SpectrumAlignment imagConstant =
        MakeSpectrumAlignment(grid, block, debugCombo, plan, imagExponent, imagConstantTerm);

    SpectrumAlignment dzdcRealZReal{};
    SpectrumAlignment dzdcRealNegativeZImag{};
    SpectrumAlignment dzdcRealOne{};
    SpectrumAlignment dzdcImagZImag{};
    SpectrumAlignment dzdcImagZReal{};
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealZReal =
            MakeSpectrumAlignment(grid, block, debugCombo, plan, dzdcRealExponent, dzdcRealZRealTerm);
        dzdcRealNegativeZImag = MakeSpectrumAlignment(
            grid, block, debugCombo, plan, dzdcRealExponent, dzdcRealNegativeZImagTerm);
        dzdcRealOne =
            MakeSpectrumAlignment(grid, block, debugCombo, plan, dzdcRealExponent, dzdcRealOneTerm);
        dzdcImagZImag =
            MakeSpectrumAlignment(grid, block, debugCombo, plan, dzdcImagExponent, dzdcImagZImagTerm);
        dzdcImagZReal =
            MakeSpectrumAlignment(grid, block, debugCombo, plan, dzdcImagExponent, dzdcImagZRealTerm);
    }

    const uint64_t zeroMont = SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 0);
    const uint64_t oneMont = SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1);
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
        uint64_t real = zeroMont;
        if (!realZSquare.IsZero) {
            const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, workspace.ZReal[i], workspace.ZReal[i]);
            const uint64_t value = ScaleSpectrumCoefficient(
                grid, block, debugCombo, plan, roots, realZSquare, oneMont, product, i);
            real = AccumulateSpectrumCoefficient<SharkFloatParams>(real, value, realZSquare.IsNegative);
        }
        if (!realNegativeZImagSquare.IsZero) {
            const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, workspace.ZImag[i], workspace.ZImag[i]);
            const uint64_t value = ScaleSpectrumCoefficient(
                grid, block, debugCombo, plan, roots, realNegativeZImagSquare, oneMont, product, i);
            real = AccumulateSpectrumCoefficient<SharkFloatParams>(
                real, value, realNegativeZImagSquare.IsNegative);
        }
        if (!realConstant.IsZero) {
            const uint64_t value = ScaleSpectrumCoefficient(grid,
                                                            block,
                                                            debugCombo,
                                                            plan,
                                                            roots,
                                                            realConstant,
                                                            oneMont,
                                                            constantSpectra.CReal[i],
                                                            i);
            real = AccumulateSpectrumCoefficient<SharkFloatParams>(real, value, realConstant.IsNegative);
        }
        workspace.RealOutput[i] = real;

        uint64_t imag = zeroMont;
        if (!imagDoubleProduct.IsZero) {
            const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, workspace.ZReal[i], workspace.ZImag[i]);
            const uint64_t value = ScaleSpectrumCoefficient(
                grid, block, debugCombo, plan, roots, imagDoubleProduct, oneMont, product, i);
            imag = AccumulateSpectrumCoefficient<SharkFloatParams>(
                imag, value, imagDoubleProduct.IsNegative);
        }
        if (!imagConstant.IsZero) {
            const uint64_t value = ScaleSpectrumCoefficient(grid,
                                                            block,
                                                            debugCombo,
                                                            plan,
                                                            roots,
                                                            imagConstant,
                                                            oneMont,
                                                            constantSpectra.CImag[i],
                                                            i);
            imag = AccumulateSpectrumCoefficient<SharkFloatParams>(imag, value, imagConstant.IsNegative);
        }
        workspace.ImagOutput[i] = imag;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint64_t dzdcReal = zeroMont;
            if (!dzdcRealZReal.IsZero) {
                const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZReal[i], workspace.DzdcReal[i]);
                const uint64_t value = ScaleSpectrumCoefficient(
                    grid, block, debugCombo, plan, roots, dzdcRealZReal, oneMont, product, i);
                dzdcReal = AccumulateSpectrumCoefficient<SharkFloatParams>(
                    dzdcReal, value, dzdcRealZReal.IsNegative);
            }
            if (!dzdcRealNegativeZImag.IsZero) {
                const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZImag[i], workspace.DzdcImag[i]);
                const uint64_t value = ScaleSpectrumCoefficient(
                    grid, block, debugCombo, plan, roots, dzdcRealNegativeZImag, oneMont, product, i);
                dzdcReal = AccumulateSpectrumCoefficient<SharkFloatParams>(
                    dzdcReal, value, dzdcRealNegativeZImag.IsNegative);
            }
            if (!dzdcRealOne.IsZero) {
                const uint64_t value = ScaleSpectrumCoefficient(grid,
                                                                block,
                                                                debugCombo,
                                                                plan,
                                                                roots,
                                                                dzdcRealOne,
                                                                oneMont,
                                                                constantSpectra.One[i],
                                                                i);
                dzdcReal = AccumulateSpectrumCoefficient<SharkFloatParams>(
                    dzdcReal, value, dzdcRealOne.IsNegative);
            }
            workspace.DzdcRealOutput[i] = dzdcReal;

            uint64_t dzdcImag = zeroMont;
            if (!dzdcImagZImag.IsZero) {
                const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZImag[i], workspace.DzdcReal[i]);
                const uint64_t value = ScaleSpectrumCoefficient(
                    grid, block, debugCombo, plan, roots, dzdcImagZImag, oneMont, product, i);
                dzdcImag = AccumulateSpectrumCoefficient<SharkFloatParams>(
                    dzdcImag, value, dzdcImagZImag.IsNegative);
            }
            if (!dzdcImagZReal.IsZero) {
                const uint64_t product = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, workspace.ZReal[i], workspace.DzdcImag[i]);
                const uint64_t value = ScaleSpectrumCoefficient(
                    grid, block, debugCombo, plan, roots, dzdcImagZReal, oneMont, product, i);
                dzdcImag = AccumulateSpectrumCoefficient<SharkFloatParams>(
                    dzdcImag, value, dzdcImagZReal.IsNegative);
            }
            workspace.DzdcImagOutput[i] = dzdcImag;
        }
    }
    grid.sync();

    StoreReference2DebugState(
        debugStates, grid, block, DebugStatePurpose::Z2_Perm1, workspace.RealOutput, activeN);
    StoreReference2DebugState(
        debugStates, grid, block, DebugStatePurpose::Z2_Perm2, workspace.ImagOutput, activeN);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreReference2DebugState(
            debugStates, grid, block, DebugStatePurpose::Z2_PermW0, workspace.DzdcRealOutput, activeN);
        StoreReference2DebugState(
            debugStates, grid, block, DebugStatePurpose::Z2_PermW1, workspace.DzdcImagOutput, activeN);
    }
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

template <class SharkFloatParams, int BatchSize>
__device__ void
UnpackResiduesToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 const uint64_t *const residues[BatchSize],
                                 const SharkNTT::PlanPrime &plan,
                                 const uint32_t coefficientCounts[BatchSize],
                                 int64_t *const limbs[BatchSize],
                                 uint32_t limbCount)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t j = GridThreadRank(block); j < limbCount; j += gridSize) {
        const uint64_t firstBit = j >= 3 ? static_cast<uint64_t>(j - 3) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(j) + 1ull) * 32ull - 1ull;
        const uint64_t firstCoefficient = firstBit / static_cast<uint64_t>(plan.b);
        const uint64_t lastCoefficient = lastBit / static_cast<uint64_t>(plan.b);
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            int64_t total = 0;
            for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCounts[buffer];
                 ++i) {
                const uint64_t residue = residues[buffer][i];
                if (residue == 0)
                    continue;
                const bool negative = residue > halfPrime;
                const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
                const uint64_t shiftedBits = i * static_cast<uint64_t>(plan.b);
                const uint32_t q = static_cast<uint32_t>(shiftedBits >> 5);
                if (q > j || j - q > 3)
                    continue;
                const int r = static_cast<int>(shiftedBits & 31);
                const uint64_t lo = r == 0 ? magnitude : magnitude << r;
                const uint64_t hi = r == 0 ? 0ull : magnitude >> (64 - r);
                uint32_t contribution = 0;
                switch (j - q) {
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
                total +=
                    negative ? -static_cast<int64_t>(contribution) : static_cast<int64_t>(contribution);
            }
            limbs[buffer][j] = total;
        }
    }
    grid.sync();
}

template <class SharkFloatParams, int BatchSize>
__device__ void
InverseSpectraToSignedLimbsBatch(cooperative_groups::grid_group &grid,
                                 cooperative_groups::thread_block &block,
                                 uint64_t *sharedData,
                                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                                 DebugState<SharkFloatParams> *debugStates,
                                 const SharkNTT::PlanPrime &plan,
                                 SharkNTT::RootTables &roots,
                                 uint64_t *const spectra[BatchSize],
                                 const uint32_t coefficientCounts[BatchSize],
                                 int64_t *const limbs[BatchSize],
                                 uint32_t limbCount,
                                 const DebugStatePurpose residuesPurposes[BatchSize],
                                 const DebugStatePurpose limbsPurposes[BatchSize])
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    BitReverseInplace64Batch<BatchSize>(
        grid, block, spectra, activeN, static_cast<uint32_t>(plan.stages));
    grid.sync();
    NTTRadix2Batch<SharkFloatParams, true, BatchSize>(
        sharedData, grid, block, debugCombo, spectra, activeN, plan.stages, roots);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            uint64_t value = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, spectra[buffer][i], roots.psi_inv_pows[i]);
            value = SharkNTT::MontgomeryMul<SharkFloatParams>(
                grid, block, debugCombo, value, roots.Ninvm_mont);
            spectra[buffer][i] =
                SharkNTT::FromMontgomery<SharkFloatParams>(grid, block, debugCombo, value);
        }
    }
    grid.sync();
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        StoreReference2DebugState(
            debugStates, grid, block, residuesPurposes[buffer], spectra[buffer], activeN);
    }
    const uint64_t *residueViews[BatchSize];
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer)
        residueViews[buffer] = spectra[buffer];
    UnpackResiduesToSignedLimbsBatch<SharkFloatParams, BatchSize>(
        grid, block, residueViews, plan, coefficientCounts, limbs, limbCount);
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  limbsPurposes[buffer],
                                  reinterpret_cast<const uint64_t *>(limbs[buffer]),
                                  limbCount);
    }
}

static __device__ int32_t
CountLeadingZeros(uint32_t value)
{
    return __clz(value);
}

constexpr int32_t CarryPrefixMin = -8;
constexpr int32_t CarryPrefixMax = 7;
constexpr uint32_t CarryPrefixStateCount = CarryPrefixMax - CarryPrefixMin + 1;
constexpr uint32_t CarryPrefixMaxWarps = 32;

template <class SharkFloatParams, int BatchSize>
__device__ void
FindHighestNonZeroPlusOneBatch(cooperative_groups::grid_group &grid,
                               cooperative_groups::thread_block &block,
                               uint32_t *const values[BatchSize],
                               const uint32_t counts[BatchSize],
                               uint32_t *const results[BatchSize],
                               uint64_t *sharedStorage)
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMaximum = reinterpret_cast<uint32_t *>(sharedStorage);

    if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            *results[buffer] = 0u;
    }
    if (threadIndex == 0u) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            blockMaximum[buffer] = 0u;
    }
    grid.sync();

    uint32_t maximumCount = 0u;
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer)
        maximumCount = maximumCount > counts[buffer] ? maximumCount : counts[buffer];

    uint32_t localMaximum[BatchSize]{};
    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (index < counts[buffer] && values[buffer][index] != 0u)
                localMaximum[buffer] = index + 1u;
        }
    }

#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        if (localMaximum[buffer] != 0u)
            atomicMax(&blockMaximum[buffer], localMaximum[buffer]);
    }
    __syncthreads();

    if (threadIndex == 0u) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (blockMaximum[buffer] != 0u)
                atomicMax(results[buffer], blockMaximum[buffer]);
        }
    }
    grid.sync();

    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                if (index >= counts[buffer])
                    continue;
                const uint32_t highestNonZeroPlusOne = *results[buffer];
                if (index >= highestNonZeroPlusOne)
                    MattsCudaAssert(values[buffer][index] == 0u);
                if (index + 1u == highestNonZeroPlusOne)
                    MattsCudaAssert(values[buffer][index] != 0u);
            }
        }
        grid.sync();
    }
}

template <class SharkFloatParams, int BatchSize>
__device__ void
FindLowestNonZeroBatch(cooperative_groups::grid_group &grid,
                       cooperative_groups::thread_block &block,
                       uint32_t *const values[BatchSize],
                       const uint32_t counts[BatchSize],
                       const bool enabled[BatchSize],
                       uint32_t *const results[BatchSize],
                       uint64_t *sharedStorage)
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMinimum = reinterpret_cast<uint32_t *>(sharedStorage);

    if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            *results[buffer] = enabled[buffer] ? counts[buffer] : 0u;
    }
    if (threadIndex == 0u) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            blockMinimum[buffer] = enabled[buffer] ? counts[buffer] : 0u;
    }
    grid.sync();

    uint32_t maximumCount = 0u;
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        if (enabled[buffer])
            maximumCount = maximumCount > counts[buffer] ? maximumCount : counts[buffer];
    }

    uint32_t localMinimum[BatchSize];
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer)
        localMinimum[buffer] = enabled[buffer] ? counts[buffer] : 0u;

    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (enabled[buffer] && index < counts[buffer] && values[buffer][index] != 0u)
                localMinimum[buffer] = localMinimum[buffer] < index ? localMinimum[buffer] : index;
        }
    }

#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        if (enabled[buffer] && localMinimum[buffer] != counts[buffer])
            atomicMin(&blockMinimum[buffer], localMinimum[buffer]);
    }
    __syncthreads();

    if (threadIndex == 0u) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (enabled[buffer] && blockMinimum[buffer] != counts[buffer])
                atomicMin(results[buffer], blockMinimum[buffer]);
        }
    }
    grid.sync();

    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                if (!enabled[buffer] || index >= counts[buffer])
                    continue;
                const uint32_t lowestNonZero = *results[buffer];
                if (index < lowestNonZero)
                    MattsCudaAssert(values[buffer][index] == 0u);
                if (index == lowestNonZero)
                    MattsCudaAssert(values[buffer][index] != 0u);
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

static __device__ uint64_t
CarryPrefixIdentity()
{
    uint64_t transform = 0;
    for (uint32_t input = 0; input < CarryPrefixStateCount; ++input)
        transform |= static_cast<uint64_t>(input) << (input * 4u);
    return transform;
}

static __device__ int32_t
ApplyCarryPrefix(uint64_t transform, int32_t carry)
{
    MattsCudaAssert(carry >= CarryPrefixMin && carry <= CarryPrefixMax);
    const uint32_t input = static_cast<uint32_t>(carry - CarryPrefixMin);
    const uint32_t output = static_cast<uint32_t>((transform >> (input * 4u)) & 0xFu);
    return static_cast<int32_t>(output) + CarryPrefixMin;
}

static __device__ uint64_t
ComposeCarryPrefixes(uint64_t earlier, uint64_t later)
{
    uint64_t combined = 0;
#pragma unroll
    for (uint32_t input = 0; input < CarryPrefixStateCount; ++input) {
        const int32_t afterEarlier =
            static_cast<int32_t>((earlier >> (input * 4u)) & 0xFu) + CarryPrefixMin;
        const uint32_t afterLater = static_cast<uint32_t>(
            (later >> (static_cast<uint32_t>(afterEarlier - CarryPrefixMin) * 4u)) & 0xFu);
        combined |= static_cast<uint64_t>(afterLater) << (input * 4u);
    }
    return combined;
}

static __device__ int32_t
CarryOutForSignedLimb(int64_t limb, int32_t carryIn)
{
    constexpr int64_t Base = 1ll << 32;
    const int64_t sum = limb + carryIn;
    const uint32_t digit = static_cast<uint32_t>(static_cast<uint64_t>(sum));
    return static_cast<int32_t>((sum - static_cast<int64_t>(digit)) / Base);
}

static __device__ uint64_t
MakeSignedCarryPrefix(int64_t limb)
{
    uint64_t transform = 0;
    for (int32_t carryIn = CarryPrefixMin; carryIn <= CarryPrefixMax; ++carryIn) {
        const int32_t carryOut = CarryOutForSignedLimb(limb, carryIn);
        MattsCudaAssert(carryOut >= CarryPrefixMin && carryOut <= CarryPrefixMax);
        const uint32_t input = static_cast<uint32_t>(carryIn - CarryPrefixMin);
        const uint32_t output = static_cast<uint32_t>(carryOut - CarryPrefixMin);
        transform |= static_cast<uint64_t>(output) << (input * 4u);
    }
    return transform;
}

static __device__ uint64_t
ShuffleUpCarryPrefix(const cooperative_groups::thread_block_tile<32> &tile, uint64_t value, int offset)
{
    const uint32_t low = tile.shfl_up(static_cast<uint32_t>(value), offset);
    const uint32_t high = tile.shfl_up(static_cast<uint32_t>(value >> 32u), offset);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

static __device__ uint64_t
ShuffleDownCarryPrefix(const cooperative_groups::thread_block_tile<32> &tile, uint64_t value, int offset)
{
    const uint32_t low = tile.shfl_down(static_cast<uint32_t>(value), offset);
    const uint32_t high = tile.shfl_down(static_cast<uint32_t>(value >> 32u), offset);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
}

static __device__ uint64_t
ShuffleCarryPrefix(const cooperative_groups::thread_block_tile<32> &tile, uint64_t value, int sourceLane)
{
    const uint32_t low = tile.shfl(static_cast<uint32_t>(value), sourceLane);
    const uint32_t high = tile.shfl(static_cast<uint32_t>(value >> 32u), sourceLane);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
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

static __device__ uint64_t
LoadCarryPrefixTransform(uint64_t *transform)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomicTransform(*transform);
    return atomicTransform.load(cuda::memory_order_relaxed);
}

static __device__ void
StoreCarryPrefixTransform(uint64_t *transform, uint64_t value)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> atomicTransform(*transform);
    atomicTransform.store(value, cuda::memory_order_relaxed);
}

static __device__ void
PublishCarryPrefixDescriptorAggregate(HpSharkReference2CarryPrefixDescriptor &descriptor,
                                      uint64_t aggregate)
{
    StoreCarryPrefixTransform(&descriptor.AggregateTransform, aggregate);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Aggregate);
}

static __device__ void
PublishCarryPrefixDescriptorPrefix(HpSharkReference2CarryPrefixDescriptor &descriptor, uint64_t prefix)
{
    StoreCarryPrefixTransform(&descriptor.PrefixTransform, prefix);
    PublishCarryPrefixState(&descriptor.State, CarryPrefixDescriptorState::Prefix);
}

static __device__ uint64_t
ResolveCarryPrefixWarp(HpSharkReference2CarryPrefixDescriptor *descriptors,
                       uint32_t part,
                       const cooperative_groups::thread_block_tile<32> &tile)
{
    const uint32_t lane = tile.thread_rank();
    uint64_t exclusive = CarryPrefixIdentity();
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

            const unsigned validMask = tile.ballot(validDescriptor);
            const unsigned readyMask =
                tile.ballot(!validDescriptor || state != CarryPrefixDescriptorState::Empty);
            const unsigned unresolvedMask = validMask & ~readyMask;
            validDescriptorCount = static_cast<uint32_t>(__popc(validMask));
            const uint32_t contiguousReadyCount = unresolvedMask == 0u
                                                      ? validDescriptorCount
                                                      : static_cast<uint32_t>(__ffs(unresolvedMask) - 1);
            const unsigned contiguousReadyMask =
                contiguousReadyCount == 32u ? 0xFFFF'FFFFu : ((1u << contiguousReadyCount) - 1u);
            const unsigned prefixMask =
                tile.ballot(validDescriptor && state == CarryPrefixDescriptorState::Prefix) &
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
        uint64_t transform = CarryPrefixIdentity();
        if (lane < descriptorCount) {
            transform = state == CarryPrefixDescriptorState::Prefix
                            ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                            : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
        }

        uint64_t windowTransform = transform;
#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint64_t older = ShuffleDownCarryPrefix(tile, windowTransform, offset);
            if (lane + offset < descriptorCount)
                windowTransform = ComposeCarryPrefixes(older, windowTransform);
        }

        if (lane == 0u)
            exclusive = ComposeCarryPrefixes(windowTransform, exclusive);
        if (foundPrefix)
            break;
        const int32_t nextPreviousPart = previousPart - static_cast<int32_t>(descriptorCount);
        MattsCudaAssert(nextPreviousPart < previousPart);
        previousPart = nextPreviousPart;
    }

    return ShuffleCarryPrefix(tile, exclusive, 0);
}

template <int BatchSize>
__device__ void
PrepareSignedCarryPrefixesBatch(cooperative_groups::grid_group &grid,
                                cooperative_groups::thread_block &block,
                                int64_t *const limbs[BatchSize],
                                uint32_t limbCount,
                                uint64_t *const transforms[BatchSize],
                                HpSharkReference2CarryPrefixDescriptor *const descriptors[BatchSize])
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            transforms[buffer][index] = MakeSignedCarryPrefix(limbs[buffer][index]);
    }

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (limbCount + blockSize - 1u) / blockSize;
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            PublishCarryPrefixState(&descriptors[buffer][part].State, CarryPrefixDescriptorState::Empty);
        }
    }
    grid.sync();
}

template <int BatchSize>
__device__ void
PrefixCarryTransformsDLBBatch(cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              uint64_t *const transforms[BatchSize],
                              uint32_t count,
                              HpSharkReference2CarryPrefixDescriptor *const descriptors[BatchSize],
                              uint64_t *sharedStorage)
{
    static_assert(BatchSize >= 1 && BatchSize <= 4);
    if (count == 0u)
        return;

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (count + blockSize - 1u) / blockSize;
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t lane = threadIndex & 31u;
    const uint32_t warp = threadIndex >> 5u;
    const uint32_t numWarps = (blockSize + 31u) >> 5u;
    const cooperative_groups::thread_block_tile<32> warpTile =
        cooperative_groups::tiled_partition<32>(block);
    constexpr uint32_t SharedWordsPerStream = 2u * CarryPrefixMaxWarps + 1u;

    // Workspace descriptors are sized for the supported cooperative launch
    // minimum of one warp per block. Ref2's launch calculator selects a warp
    // multiple, which also keeps the intra-warp scan well-defined.
    MattsCudaAssert(blockSize >= 32u && (blockSize & 31u) == 0u);
    MattsCudaAssert(numWarps <= CarryPrefixMaxWarps);

    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;
    for (uint32_t part = processorId; part < numParts; part += activeProcessors) {
        const uint32_t base = part * blockSize;
        const uint32_t index = base + threadIndex;
        const bool hasValue = index < count;
        uint64_t inclusive[BatchSize];
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            inclusive[buffer] = hasValue ? transforms[buffer][index] : CarryPrefixIdentity();

#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint64_t previous =
                    ShuffleUpCarryPrefix(warpTile, inclusive[buffer], static_cast<int>(offset));
                if (lane >= offset)
                    inclusive[buffer] = ComposeCarryPrefixes(previous, inclusive[buffer]);
            }
        }

        const uint32_t warpEnd = (warp + 1u) * 32u;
        const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
        if (threadIndex == warpLastThread) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                uint64_t *warpAggregates = sharedStorage + buffer * SharedWordsPerStream;
                warpAggregates[warp] = inclusive[buffer];
            }
        }
        __syncthreads();

        uint64_t aggregates[BatchSize];
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer)
            aggregates[buffer] = CarryPrefixIdentity();

        if (warp == 0u) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                uint64_t *warpAggregates = sharedStorage + buffer * SharedWordsPerStream;
                uint64_t *warpPrefixes = warpAggregates + CarryPrefixMaxWarps;
                uint64_t warpInclusive = lane < numWarps ? warpAggregates[lane] : CarryPrefixIdentity();
#pragma unroll
                for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                    const uint64_t previous =
                        ShuffleUpCarryPrefix(warpTile, warpInclusive, static_cast<int>(offset));
                    if (lane >= offset && lane < numWarps)
                        warpInclusive = ComposeCarryPrefixes(previous, warpInclusive);
                }

                const uint64_t previous = ShuffleUpCarryPrefix(warpTile, warpInclusive, 1);
                if (lane < numWarps)
                    warpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : previous;
                aggregates[buffer] =
                    ShuffleCarryPrefix(warpTile, warpInclusive, static_cast<int>(numWarps - 1u));
            }
        }

        if (threadIndex == 0u) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                PublishCarryPrefixDescriptorAggregate(descriptors[buffer][part], aggregates[buffer]);
            }
        }

        if (warp == 0u) {
            warpTile.sync();
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                const uint64_t exclusive = ResolveCarryPrefixWarp(descriptors[buffer], part, warpTile);
                if (lane == 0u) {
                    const uint64_t prefix = ComposeCarryPrefixes(exclusive, aggregates[buffer]);
                    PublishCarryPrefixDescriptorPrefix(descriptors[buffer][part], prefix);
                    uint64_t *exclusiveStorage =
                        sharedStorage + buffer * SharedWordsPerStream + 2u * CarryPrefixMaxWarps;
                    exclusiveStorage[0] = exclusive;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            uint64_t *warpAggregates = sharedStorage + buffer * SharedWordsPerStream;
            uint64_t *warpPrefixes = warpAggregates + CarryPrefixMaxWarps;
            uint64_t *exclusiveStorage = warpPrefixes + CarryPrefixMaxWarps;
            const uint64_t exclusivePart = exclusiveStorage[0];
            const uint64_t warpExclusive = warpPrefixes[warp];
            const uint64_t previous = ShuffleUpCarryPrefix(warpTile, inclusive[buffer], 1);
            const uint64_t localExclusive = lane == 0u ? CarryPrefixIdentity() : previous;
            if (hasValue) {
                const uint64_t prefixWithinPart = ComposeCarryPrefixes(warpExclusive, localExclusive);
                transforms[buffer][index] = ComposeCarryPrefixes(exclusivePart, prefixWithinPart);
            }
        }
        // The next partition's aggregate barrier protects shared scratch reuse.
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
FinalizeSignedStream(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugState<SharkFloatParams> *debugStates,
                     uint64_t *carryPrefixShared,
                     HpSharkReference2Workspace<SharkFloatParams> &workspace,
                     uint32_t limbCount,
                     int32_t realExponent,
                     int32_t imagExponent,
                     int32_t dzdcRealExponent,
                     int32_t dzdcImagExponent,
                     HpSharkReferenceResults<SharkFloatParams> *combo)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    using Descriptor = HpSharkReference2CarryPrefixDescriptor;
    constexpr int BatchSize = SharkFloatParams::EnableNewtonRaphson ? 4 : 2;
    constexpr uint32_t Capacity = Workspace::MaxFusedLimbs;
    constexpr uint32_t DigitLengthControl = 0;
    constexpr uint32_t NegativeControl = 1;
    constexpr uint32_t NonZeroReductionControl = 2;
    constexpr uint32_t DescriptorWords =
        (Workspace::MaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    constexpr uint32_t ControlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    static_assert((Capacity * sizeof(uint64_t)) % alignof(Descriptor) == 0u);
    static_assert(Capacity + DescriptorWords + ControlWords <= Workspace::MaxFusedN);

    uint64_t *scratch[BatchSize] = {workspace.RealOutput, workspace.ImagOutput};
    int64_t *limbs[BatchSize] = {workspace.RealLimbs, workspace.ImagLimbs};
    int32_t commonExponents[BatchSize] = {realExponent, imagExponent};
    HpSharkFloat<SharkFloatParams> *outputs[BatchSize] = {&combo->Multiply.A, &combo->Multiply.B};
    DebugStatePurpose digitsPurposes[BatchSize] = {DebugStatePurpose::SignedCarry1,
                                                   DebugStatePurpose::SignedCarry2};
    DebugStatePurpose magnitudePurposes[BatchSize] = {DebugStatePurpose::FinalAdd1,
                                                      DebugStatePurpose::FinalAdd2};
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        scratch[2] = workspace.DzdcRealOutput;
        scratch[3] = workspace.DzdcImagOutput;
        limbs[2] = workspace.DzdcRealLimbs;
        limbs[3] = workspace.DzdcImagLimbs;
        commonExponents[2] = dzdcRealExponent;
        commonExponents[3] = dzdcImagExponent;
        outputs[2] = &combo->Multiply.DzdcReal;
        outputs[3] = &combo->Multiply.DzdcImag;
        digitsPurposes[2] = DebugStatePurpose::SignedCarryDzdc1;
        digitsPurposes[3] = DebugStatePurpose::SignedCarryDzdc2;
        magnitudePurposes[2] = DebugStatePurpose::FinalAddDzdc1;
        magnitudePurposes[3] = DebugStatePurpose::FinalAddDzdc2;
    }

    uint64_t *transforms[BatchSize];
    uint32_t *digits[BatchSize];
    Descriptor *descriptors[BatchSize];
    uint32_t *control[BatchSize];
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        transforms[buffer] = scratch[buffer];
        digits[buffer] = reinterpret_cast<uint32_t *>(scratch[buffer]);
        descriptors[buffer] = reinterpret_cast<Descriptor *>(scratch[buffer] + Capacity);
        control[buffer] = reinterpret_cast<uint32_t *>(scratch[buffer] + Capacity + DescriptorWords);
    }

    MattsCudaAssert(limbCount > 0u && limbCount <= Capacity);

    PrepareSignedCarryPrefixesBatch<BatchSize>(grid, block, limbs, limbCount, transforms, descriptors);
    PrefixCarryTransformsDLBBatch<BatchSize>(
        grid, block, transforms, limbCount, descriptors, carryPrefixShared);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            const int64_t signedLimb = limbs[buffer][index];
            const int32_t carryIn = ApplyCarryPrefix(transforms[buffer][index], 0);
            const uint32_t digit = static_cast<uint32_t>(static_cast<uint64_t>(signedLimb + carryIn));
            if (index + 1u == limbCount) {
                int32_t finalCarry = CarryOutForSignedLimb(signedLimb, carryIn);
                uint32_t digitLength = limbCount;
                while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
                    limbs[buffer][digitLength++] =
                        static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
                    finalCarry = CarryOutForSignedLimb(finalCarry, 0);
                }
                control[buffer][DigitLengthControl] = digitLength;
                control[buffer][NegativeControl] = finalCarry < 0 ? 1u : 0u;
            }
            limbs[buffer][index] = digit;
        }
    }
    grid.sync();

    uint32_t digitLengths[BatchSize];
    bool negative[BatchSize];
    uint32_t maximumDigitLength = 0u;
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer) {
        digitLengths[buffer] = control[buffer][DigitLengthControl];
        negative[buffer] = control[buffer][NegativeControl] != 0u;
        maximumDigitLength =
            maximumDigitLength > digitLengths[buffer] ? maximumDigitLength : digitLengths[buffer];
    }

    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (index < digitLengths[buffer])
                digits[buffer][index] = static_cast<uint32_t>(limbs[buffer][index]);
        }
    }
    grid.sync();

    if constexpr (HpShark::DebugChecksums) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            StoreReference2DebugState(
                debugStates, grid, block, digitsPurposes[buffer], digits[buffer], digitLengths[buffer]);
        }
    }

    uint32_t *nonZeroResults[BatchSize];
#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer)
        nonZeroResults[buffer] = &control[buffer][NonZeroReductionControl];

    // In (~digits) + 1, the carry reaches the lowest nonzero digit and stops there.
    // Locating that digit avoids a second cross-block carry-prefix scan.
    FindLowestNonZeroBatch<SharkFloatParams, BatchSize>(
        grid, block, digits, digitLengths, negative, nonZeroResults, carryPrefixShared);

    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (!negative[buffer] || index >= digitLengths[buffer])
                continue;
            const uint32_t lowestNonZero = *nonZeroResults[buffer];
            if (index < lowestNonZero)
                digits[buffer][index] = 0u;
            else if (index == lowestNonZero)
                digits[buffer][index] = 0u - digits[buffer][index];
            else
                digits[buffer][index] = ~digits[buffer][index];
        }
    }

    if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            if (negative[buffer] && *nonZeroResults[buffer] == digitLengths[buffer]) {
                MattsCudaAssert(digitLengths[buffer] < Capacity);
                if (digitLengths[buffer] < Capacity)
                    digits[buffer][digitLengths[buffer]++] = 1u;
            }
            control[buffer][DigitLengthControl] = digitLengths[buffer];
        }
    }
    grid.sync();

#pragma unroll
    for (int buffer = 0; buffer < BatchSize; ++buffer)
        digitLengths[buffer] = control[buffer][DigitLengthControl];

    FindHighestNonZeroPlusOneBatch<SharkFloatParams, BatchSize>(
        grid, block, digits, digitLengths, nonZeroResults, carryPrefixShared);

    if constexpr (HpShark::DebugChecksums) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      magnitudePurposes[buffer],
                                      digits[buffer],
                                      *nonZeroResults[buffer]);
        }
    }

    constexpr uint32_t ActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr uint32_t TotalOutputDigits = BatchSize * ActualDigits;
    if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            const uint32_t highestNonZeroPlusOne = *nonZeroResults[buffer];
            if (highestNonZeroPlusOne == 0u) {
                outputs[buffer]->Exponent = -100'000'000;
                outputs[buffer]->SetNegative(false);
                continue;
            }
            const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
            const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                                   CountLeadingZeros(digits[buffer][highestNonZero]);
            const int desiredBit = (static_cast<int>(ActualDigits) - 1) * 32 + 31;
            outputs[buffer]->Exponent = commonExponents[buffer] + currentBit - desiredBit;
            outputs[buffer]->SetNegative(negative[buffer]);
        }
    }

    for (uint32_t flatIndex = GridThreadRank(block); flatIndex < TotalOutputDigits;
         flatIndex += gridSize) {
        const uint32_t buffer = flatIndex / ActualDigits;
        const int digitIndex = static_cast<int>(flatIndex % ActualDigits);
        const uint32_t highestNonZeroPlusOne = *nonZeroResults[buffer];
        if (highestNonZeroPlusOne == 0u) {
            outputs[buffer]->Digits[digitIndex] = 0u;
            continue;
        }
        const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
        const int magnitudeLength = static_cast<int>(highestNonZeroPlusOne);
        const int currentBit = static_cast<int>(highestNonZero) * 32 + 31 -
                               CountLeadingZeros(digits[buffer][highestNonZero]);
        const int desiredBit = (static_cast<int>(ActualDigits) - 1) * 32 + 31;
        const int shift = currentBit - desiredBit;
        outputs[buffer]->Digits[digitIndex] =
            shift > 0 ? FunnelShiftRight(digits[buffer], digitIndex, magnitudeLength, shift)
                      : FunnelShiftLeft(digits[buffer], digitIndex, magnitudeLength, -shift);
    }
    grid.sync();

    if constexpr (HpShark::Debug) {
        if (IsLeader<SharkFloatParams>(block)) {
#pragma unroll
            for (int buffer = 0; buffer < BatchSize; ++buffer) {
                if (*nonZeroResults[buffer] != 0u) {
                    MattsCudaAssert((outputs[buffer]->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
            }
        }
    }
}

template <class SharkFloatParams>
__device__ void
FusedReferenceOrbitStep(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        uint64_t *sharedData,
                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                        DebugState<SharkFloatParams> *debugStates,
                        uint64_t *carryPrefixShared,
                        HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Reference2Workspace;
    const auto &zReal = combo->Multiply.A;
    const auto &zImag = combo->Multiply.B;
    const auto &cReal = combo->Add.C_A;
    const auto &cImag = combo->Add.E_B;
    const SharkNTT::PlanPrime basePlan = workspace.Plans[0];
    const uint32_t ignoredPrecisionBits = workspace.IgnoredPrecisionBits;

    const FusedTerm<SharkFloatParams> realZSquareTerm = MakeProductTerm(
        zReal, SpectrumId::ZReal, zReal, SpectrumId::ZReal, false, 0, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> realNegativeZImagSquareTerm = MakeProductTerm(
        zImag, SpectrumId::ZImag, zImag, SpectrumId::ZImag, true, 0, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> realConstantTerm =
        MakeLinearTerm(cReal, SpectrumId::CReal, false, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> imagDoubleProductTerm = MakeProductTerm(
        zReal, SpectrumId::ZReal, zImag, SpectrumId::ZImag, false, 1, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> imagConstantTerm =
        MakeLinearTerm(cImag, SpectrumId::CImag, false, ignoredPrecisionBits);
    int32_t realExponent;
    int32_t imagExponent;
    const bool realZero = ResolveCommonExponent(
        &realExponent, realZSquareTerm, realNegativeZImagSquareTerm, realConstantTerm);
    const bool imagZero = ResolveCommonExponent(&imagExponent, imagDoubleProductTerm, imagConstantTerm);
    uint64_t requiredCoefficients = RequiredCoefficientsForTerms(
        realExponent, basePlan, realZSquareTerm, realNegativeZImagSquareTerm, realConstantTerm);
    const uint64_t imagCoefficients =
        RequiredCoefficientsForTerms(imagExponent, basePlan, imagDoubleProductTerm, imagConstantTerm);
    requiredCoefficients =
        requiredCoefficients > imagCoefficients ? requiredCoefficients : imagCoefficients;

    FusedTerm<SharkFloatParams> dzdcRealZRealTerm{};
    FusedTerm<SharkFloatParams> dzdcRealNegativeZImagTerm{};
    FusedTerm<SharkFloatParams> dzdcRealOneTerm{};
    FusedTerm<SharkFloatParams> dzdcImagZImagTerm{};
    FusedTerm<SharkFloatParams> dzdcImagZRealTerm{};
    int32_t dzdcRealExponent = 0;
    int32_t dzdcImagExponent = 0;
    bool dzdcRealZero = true;
    bool dzdcImagZero = true;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealZRealTerm = MakeProductTerm(zReal,
                                            SpectrumId::ZReal,
                                            combo->Multiply.DzdcReal,
                                            SpectrumId::DzdcReal,
                                            false,
                                            1,
                                            ignoredPrecisionBits);
        dzdcRealNegativeZImagTerm = MakeProductTerm(zImag,
                                                    SpectrumId::ZImag,
                                                    combo->Multiply.DzdcImag,
                                                    SpectrumId::DzdcImag,
                                                    true,
                                                    1,
                                                    ignoredPrecisionBits);
        dzdcRealOneTerm = MakeLinearTerm(combo->Add.One, SpectrumId::One, false, ignoredPrecisionBits);
        dzdcImagZImagTerm = MakeProductTerm(zImag,
                                            SpectrumId::ZImag,
                                            combo->Multiply.DzdcReal,
                                            SpectrumId::DzdcReal,
                                            false,
                                            1,
                                            ignoredPrecisionBits);
        dzdcImagZRealTerm = MakeProductTerm(zReal,
                                            SpectrumId::ZReal,
                                            combo->Multiply.DzdcImag,
                                            SpectrumId::DzdcImag,
                                            false,
                                            1,
                                            ignoredPrecisionBits);
        dzdcRealZero = ResolveCommonExponent(
            &dzdcRealExponent, dzdcRealZRealTerm, dzdcRealNegativeZImagTerm, dzdcRealOneTerm);
        dzdcImagZero = ResolveCommonExponent(&dzdcImagExponent, dzdcImagZImagTerm, dzdcImagZRealTerm);
        const uint64_t dzdcRealCoefficients = RequiredCoefficientsForTerms(
            dzdcRealExponent, basePlan, dzdcRealZRealTerm, dzdcRealNegativeZImagTerm, dzdcRealOneTerm);
        const uint64_t dzdcImagCoefficients = RequiredCoefficientsForTerms(
            dzdcImagExponent, basePlan, dzdcImagZImagTerm, dzdcImagZRealTerm);
        requiredCoefficients =
            requiredCoefficients > dzdcRealCoefficients ? requiredCoefficients : dzdcRealCoefficients;
        requiredCoefficients =
            requiredCoefficients > dzdcImagCoefficients ? requiredCoefficients : dzdcImagCoefficients;
    }

    if (requiredCoefficients == 0) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            HpSharkFloat<SharkFloatParams> *outputs[4] = {&combo->Multiply.A,
                                                          &combo->Multiply.B,
                                                          &combo->Multiply.DzdcReal,
                                                          &combo->Multiply.DzdcImag};
            SetZeroBatch<SharkFloatParams, 4>(grid, block, outputs);
        } else {
            HpSharkFloat<SharkFloatParams> *outputs[2] = {&combo->Multiply.A, &combo->Multiply.B};
            SetZeroBatch<SharkFloatParams, 2>(grid, block, outputs);
        }
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_Add1, combo->Multiply.A);
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_Add2, combo->Multiply.B);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugValue(
                debugStates, grid, block, DebugStatePurpose::Result_AddDzdc1, combo->Multiply.DzdcReal);
            StoreReference2DebugValue(
                debugStates, grid, block, DebugStatePurpose::Result_AddDzdc2, combo->Multiply.DzdcImag);
        }
        return;
    }

    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN || requiredN < 2) {
        if (IsLeader<SharkFloatParams>(block))
            combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const uint32_t activeN = static_cast<uint32_t>(requiredN);
    MattsCudaAssert(activeN >= Workspace::MinFusedN);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Workspace::MinFusedStages;
    MattsCudaAssert(planSlot < Workspace::PlanCacheEntryCount);
    const SharkNTT::PlanPrime &plan = workspace.Plans[planSlot];
    SharkNTT::RootTables &roots = workspace.PlanRoots[planSlot];
    MattsCudaAssert(static_cast<uint32_t>(plan.N) == activeN);
    MattsCudaAssert(static_cast<uint32_t>(roots.N) == activeN);
    HpSharkReference2ConstantSpectra constantSpectra = workspace.ConstantSpectra[planSlot];
    const uint32_t limbCount = (activeN * static_cast<uint32_t>(plan.b) + 31u) / 32u + 2u;

    if constexpr (HpShark::DebugChecksums) {
        constantSpectra = {workspace.CRealArena, workspace.CImagArena, workspace.OneArena};
        const HpSharkFloat<SharkFloatParams> *normalForwardValues[4] = {&zReal, &zImag, &cReal, &cImag};
        uint64_t *normalForwardOutputs[4] = {
            workspace.ZReal, workspace.ZImag, constantSpectra.CReal, constantSpectra.CImag};
        const DebugStatePurpose normalPackedPurposes[4] = {DebugStatePurpose::Z0XX,
                                                           DebugStatePurpose::Z0YY,
                                                           DebugStatePurpose::Z0XY,
                                                           DebugStatePurpose::Z0W0};
        const DebugStatePurpose normalForwardPurposes[4] = {DebugStatePurpose::Z2XX,
                                                            DebugStatePurpose::Z2YY,
                                                            DebugStatePurpose::Z2XY,
                                                            DebugStatePurpose::Z2W0};
        PackTwistForwardBatch<SharkFloatParams, 4>(grid,
                                                   block,
                                                   sharedData,
                                                   debugCombo,
                                                   debugStates,
                                                   normalForwardValues,
                                                   plan,
                                                   roots,
                                                   normalForwardOutputs,
                                                   ignoredPrecisionBits,
                                                   normalPackedPurposes,
                                                   normalForwardPurposes);

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const HpSharkFloat<SharkFloatParams> *newtonRaphsonForwardValues[3] = {
                &combo->Multiply.DzdcReal, &combo->Multiply.DzdcImag, &combo->Add.One};
            uint64_t *newtonRaphsonForwardOutputs[3] = {
                workspace.DzdcReal, workspace.DzdcImag, constantSpectra.One};
            const DebugStatePurpose newtonRaphsonPackedPurposes[3] = {
                DebugStatePurpose::Z0W1, DebugStatePurpose::Z0W2, DebugStatePurpose::Z0W3};
            const DebugStatePurpose newtonRaphsonForwardPurposes[3] = {
                DebugStatePurpose::Z2W1, DebugStatePurpose::Z2W2, DebugStatePurpose::Z2W3};
            PackTwistForwardBatch<SharkFloatParams, 3>(grid,
                                                       block,
                                                       sharedData,
                                                       debugCombo,
                                                       debugStates,
                                                       newtonRaphsonForwardValues,
                                                       plan,
                                                       roots,
                                                       newtonRaphsonForwardOutputs,
                                                       ignoredPrecisionBits,
                                                       newtonRaphsonPackedPurposes,
                                                       newtonRaphsonForwardPurposes);
        }
    } else if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const HpSharkFloat<SharkFloatParams> *forwardValues[4] = {
            &zReal, &zImag, &combo->Multiply.DzdcReal, &combo->Multiply.DzdcImag};
        uint64_t *forwardOutputs[4] = {
            workspace.ZReal, workspace.ZImag, workspace.DzdcReal, workspace.DzdcImag};
        const DebugStatePurpose packedPurposes[4] = {DebugStatePurpose::Z0XX,
                                                     DebugStatePurpose::Z0YY,
                                                     DebugStatePurpose::Z0W1,
                                                     DebugStatePurpose::Z0W2};
        const DebugStatePurpose forwardPurposes[4] = {DebugStatePurpose::Z2XX,
                                                      DebugStatePurpose::Z2YY,
                                                      DebugStatePurpose::Z2W1,
                                                      DebugStatePurpose::Z2W2};
        PackTwistForwardBatch<SharkFloatParams, 4>(grid,
                                                   block,
                                                   sharedData,
                                                   debugCombo,
                                                   debugStates,
                                                   forwardValues,
                                                   plan,
                                                   roots,
                                                   forwardOutputs,
                                                   ignoredPrecisionBits,
                                                   packedPurposes,
                                                   forwardPurposes);
    } else {
        const HpSharkFloat<SharkFloatParams> *forwardValues[2] = {&zReal, &zImag};
        uint64_t *forwardOutputs[2] = {workspace.ZReal, workspace.ZImag};
        const DebugStatePurpose packedPurposes[2] = {DebugStatePurpose::Z0XX, DebugStatePurpose::Z0YY};
        const DebugStatePurpose forwardPurposes[2] = {DebugStatePurpose::Z2XX, DebugStatePurpose::Z2YY};
        PackTwistForwardBatch<SharkFloatParams, 2>(grid,
                                                   block,
                                                   sharedData,
                                                   debugCombo,
                                                   debugStates,
                                                   forwardValues,
                                                   plan,
                                                   roots,
                                                   forwardOutputs,
                                                   ignoredPrecisionBits,
                                                   packedPurposes,
                                                   forwardPurposes);
    }

    AccumulateFixedOutputSpectra(grid,
                                 block,
                                 debugCombo,
                                 debugStates,
                                 plan,
                                 roots,
                                 workspace,
                                 constantSpectra,
                                 realExponent,
                                 realZSquareTerm,
                                 realNegativeZImagSquareTerm,
                                 realConstantTerm,
                                 imagExponent,
                                 imagDoubleProductTerm,
                                 imagConstantTerm,
                                 dzdcRealExponent,
                                 dzdcRealZRealTerm,
                                 dzdcRealNegativeZImagTerm,
                                 dzdcRealOneTerm,
                                 dzdcImagExponent,
                                 dzdcImagZImagTerm,
                                 dzdcImagZRealTerm);

    constexpr int FinalizationStreamCount = SharkFloatParams::EnableNewtonRaphson ? 4 : 2;
    uint64_t *spectra[FinalizationStreamCount] = {workspace.RealOutput, workspace.ImagOutput};
    uint32_t coefficientCounts[FinalizationStreamCount] = {activeN, activeN};
    int64_t *limbs[FinalizationStreamCount] = {workspace.RealLimbs, workspace.ImagLimbs};
    DebugStatePurpose residuesPurposes[FinalizationStreamCount] = {DebugStatePurpose::Z2_Perm4,
                                                                   DebugStatePurpose::Z2_Perm5};
    DebugStatePurpose limbsPurposes[FinalizationStreamCount] = {DebugStatePurpose::UnpackXX,
                                                                DebugStatePurpose::UnpackYY};
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        spectra[2] = workspace.DzdcRealOutput;
        spectra[3] = workspace.DzdcImagOutput;
        coefficientCounts[2] = activeN;
        coefficientCounts[3] = activeN;
        limbs[2] = workspace.DzdcRealLimbs;
        limbs[3] = workspace.DzdcImagLimbs;
        residuesPurposes[2] = DebugStatePurpose::Z2_PermW0b;
        residuesPurposes[3] = DebugStatePurpose::Z2_PermW1b;
        limbsPurposes[2] = DebugStatePurpose::UnpackW0;
        limbsPurposes[3] = DebugStatePurpose::UnpackW1;
    }
    InverseSpectraToSignedLimbsBatch<SharkFloatParams, FinalizationStreamCount>(grid,
                                                                                block,
                                                                                sharedData,
                                                                                debugCombo,
                                                                                debugStates,
                                                                                plan,
                                                                                roots,
                                                                                spectra,
                                                                                coefficientCounts,
                                                                                limbs,
                                                                                limbCount,
                                                                                residuesPurposes,
                                                                                limbsPurposes);
    FinalizeSignedStream<SharkFloatParams>(grid,
                                           block,
                                           debugStates,
                                           carryPrefixShared,
                                           workspace,
                                           limbCount,
                                           realExponent,
                                           imagExponent,
                                           dzdcRealExponent,
                                           dzdcImagExponent,
                                           combo);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_AddDzdc1, combo->Multiply.DzdcReal);
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_AddDzdc2, combo->Multiply.DzdcImag);
    }
    StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::Result_Add1, combo->Multiply.A);
    StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::Result_Add2, combo->Multiply.B);
    return;
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
    HpSharkReference2GpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ __align__(16) uint64_t sharedData[];
    __shared__ uint64_t carryPrefixShared[4u * (2u * Reference2Detail::CarryPrefixMaxWarps + 1u)];
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

    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceEntryZReal, combo->Multiply.A);
    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceEntryZImag, combo->Multiply.B);
    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceEntryCReal, combo->Add.C_A);
    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceEntryCImag, combo->Add.E_B);

    for (uint64_t iteration = 0; iteration < combo->MaxRuntimeIters; ++iteration) {
        bool stop = false;
        if (leader)
            stop = Reference2Detail::CheckPeriodicity<SharkFloatParams>(combo, iteration);
        grid.sync();
        if (combo->PeriodicityStatus != PeriodicityResult::Continue)
            break;

        if (leader)
            Reference2Detail::UpdateD2<SharkFloatParams>(combo);
        grid.sync();

        Reference2Detail::FusedReferenceOrbitStep<SharkFloatParams>(
            grid, block, sharedData, debugCombo, debugStates, carryPrefixShared, combo);
        // FusedReferenceOrbitStep may return without a final barrier; this publishes every output and
        // PeriodicityStatus before any thread consumes them.
        grid.sync();
        if (combo->PeriodicityStatus == PeriodicityResult::Unknown)
            break;

        if (leader)
            ++combo->OutputIterCount;
        (void)stop;
    }

    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceExitZReal, combo->Multiply.A);
    Reference2Detail::StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::ReferenceExitZImag, combo->Multiply.B);
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
