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

    constexpr uint64_t Generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t generatorMont =
        SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, Generator);
    const uint64_t psi = MontgomeryPowSerial<SharkFloatParams>(
        grid, block, debugCombo, generatorMont, SharkNTT::PHI / (2ull * activeN));
    const uint64_t psiInverse =
        MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psi, SharkNTT::PHI - 1ull);
    const uint64_t omega = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, psi, psi);
    const uint64_t omegaInverse =
        MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, SharkNTT::PHI - 1ull);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    const uint32_t rank = GridThreadRank(block);
    if (rank < activeN) {
        uint64_t psiPower = MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psi, rank);
        uint64_t psiInversePower =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psiInverse, rank);
        const uint64_t psiStride =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psi, gridSize);
        const uint64_t psiInverseStride =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psiInverse, gridSize);
        for (uint32_t index = rank; index < activeN; index += gridSize) {
            roots.psi_pows[index] = psiPower;
            roots.psi_inv_pows[index] = psiInversePower;
            if (index + gridSize < activeN) {
                psiPower = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, psiPower, psiStride);
                psiInversePower = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, psiInversePower, psiInverseStride);
            }
        }
    }

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
        workspace.ValidPlanMask |= planBit;
    }
    grid.sync();
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

constexpr uint32_t FinalizationDigitLengthControl = 0;
constexpr uint32_t FinalizationNegativeControl = 1;
constexpr uint32_t FinalizationNonZeroReductionControl = 2;

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
FindHighestNonZeroPlusOne(cooperative_groups::grid_group &grid,
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
    uint32_t *blockMaximum = reinterpret_cast<uint32_t *>(sharedStorage);
    const uint32_t realCount = realControl[FinalizationDigitLengthControl];
    const uint32_t imagCount = imagControl[FinalizationDigitLengthControl];
    const uint32_t dzdcRealCount =
        SharkFloatParams::EnableNewtonRaphson ? dzdcRealControl[FinalizationDigitLengthControl] : 0u;
    const uint32_t dzdcImagCount =
        SharkFloatParams::EnableNewtonRaphson ? dzdcImagControl[FinalizationDigitLengthControl] : 0u;

    if (IsLeader<SharkFloatParams>(block)) {
        realControl[FinalizationNonZeroReductionControl] = 0u;
        imagControl[FinalizationNonZeroReductionControl] = 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            dzdcRealControl[FinalizationNonZeroReductionControl] = 0u;
            dzdcImagControl[FinalizationNonZeroReductionControl] = 0u;
        }
    }
    if (threadIndex == 0u) {
        blockMaximum[0] = 0u;
        blockMaximum[1] = 0u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            blockMaximum[2] = 0u;
            blockMaximum[3] = 0u;
        }
    }
    grid.sync();

    uint32_t maximumCount = realCount > imagCount ? realCount : imagCount;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        maximumCount = maximumCount > dzdcRealCount ? maximumCount : dzdcRealCount;
        maximumCount = maximumCount > dzdcImagCount ? maximumCount : dzdcImagCount;
    }

    uint32_t realLocalMaximum = 0u;
    uint32_t imagLocalMaximum = 0u;
    uint32_t dzdcRealLocalMaximum = 0u;
    uint32_t dzdcImagLocalMaximum = 0u;
    for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
        if (index < realCount && realDigits[index] != 0u)
            realLocalMaximum = index + 1u;
        if (index < imagCount && imagDigits[index] != 0u)
            imagLocalMaximum = index + 1u;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (index < dzdcRealCount && dzdcRealDigits[index] != 0u)
                dzdcRealLocalMaximum = index + 1u;
            if (index < dzdcImagCount && dzdcImagDigits[index] != 0u)
                dzdcImagLocalMaximum = index + 1u;
        }
    }

    if (realLocalMaximum != 0u)
        atomicMax(&blockMaximum[0], realLocalMaximum);
    if (imagLocalMaximum != 0u)
        atomicMax(&blockMaximum[1], imagLocalMaximum);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (dzdcRealLocalMaximum != 0u)
            atomicMax(&blockMaximum[2], dzdcRealLocalMaximum);
        if (dzdcImagLocalMaximum != 0u)
            atomicMax(&blockMaximum[3], dzdcImagLocalMaximum);
    }
    __syncthreads();

    if (threadIndex == 0u) {
        if (blockMaximum[0] != 0u)
            atomicMax(&realControl[FinalizationNonZeroReductionControl], blockMaximum[0]);
        if (blockMaximum[1] != 0u)
            atomicMax(&imagControl[FinalizationNonZeroReductionControl], blockMaximum[1]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (blockMaximum[2] != 0u)
                atomicMax(&dzdcRealControl[FinalizationNonZeroReductionControl], blockMaximum[2]);
            if (blockMaximum[3] != 0u)
                atomicMax(&dzdcImagControl[FinalizationNonZeroReductionControl], blockMaximum[3]);
        }
    }
    grid.sync();

    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < maximumCount; index += gridSize) {
            const uint32_t realHighest = realControl[FinalizationNonZeroReductionControl];
            const uint32_t imagHighest = imagControl[FinalizationNonZeroReductionControl];
            if (index < realCount) {
                if (index >= realHighest)
                    MattsCudaAssert(realDigits[index] == 0u);
                if (index + 1u == realHighest)
                    MattsCudaAssert(realDigits[index] != 0u);
            }
            if (index < imagCount) {
                if (index >= imagHighest)
                    MattsCudaAssert(imagDigits[index] == 0u);
                if (index + 1u == imagHighest)
                    MattsCudaAssert(imagDigits[index] != 0u);
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                const uint32_t dzdcRealHighest = dzdcRealControl[FinalizationNonZeroReductionControl];
                const uint32_t dzdcImagHighest = dzdcImagControl[FinalizationNonZeroReductionControl];
                if (index < dzdcRealCount) {
                    if (index >= dzdcRealHighest)
                        MattsCudaAssert(dzdcRealDigits[index] == 0u);
                    if (index + 1u == dzdcRealHighest)
                        MattsCudaAssert(dzdcRealDigits[index] != 0u);
                }
                if (index < dzdcImagCount) {
                    if (index >= dzdcImagHighest)
                        MattsCudaAssert(dzdcImagDigits[index] == 0u);
                    if (index + 1u == dzdcImagHighest)
                        MattsCudaAssert(dzdcImagDigits[index] != 0u);
                }
            }
        }
        grid.sync();
    }
}

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

    const uint32_t lookbackWindowsPerBatch = numWarps * 32u;
    const uint32_t lookbackBatchCount =
        numWarps == 1u ? 1u : (numParts + lookbackWindowsPerBatch - 1u) / lookbackWindowsPerBatch;
    MattsCudaAssert(lookbackBatchCount != 0u);
    MattsCudaAssert(numParts <= (1u << 30u) / lookbackBatchCount);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    const uint32_t processorId = block.group_index().x;
    const uint32_t activeProcessors = gridDim.x;
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize)
        PublishCarryPrefixState(&descriptors[part].State, CarryPrefixDescriptorState::Empty);
    const uint32_t firstPart = processorId < numParts ? processorId : 0u;
    const uint32_t firstToken = MakeCarryPrefixLookbackToken(firstPart, 0u, lookbackBatchCount);
    if (lane == 0u)
        StoreCarryPrefixLookbackStatus(
            &packedLookbackStates[warp],
            PackCarryPrefixLookbackStatus(firstToken, CarryPrefixLookbackStatus::Pending));
    grid.sync();

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

    uint64_t *realOutputArena = workspace.RealOutput;
    int64_t *realLimbs = workspace.RealLimbs;
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(realOutputArena);
    Descriptor *descriptors = reinterpret_cast<Descriptor *>(realOutputArena + capacity);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(realOutputArena + capacity + descriptorWords);
    HpSharkFloat<SharkFloatParams> *realOutput = &combo->Multiply.A;

    uint64_t *imagOutputArena = workspace.ImagOutput;
    int64_t *imagLimbs = workspace.ImagLimbs;
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(imagOutputArena);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(imagOutputArena + capacity + descriptorWords);
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
        dzdcRealDigits = reinterpret_cast<uint32_t *>(workspace.DzdcRealOutput);
        dzdcRealControl =
            reinterpret_cast<uint32_t *>(workspace.DzdcRealOutput + capacity + descriptorWords);
        dzdcRealOutput = &combo->Multiply.DzdcReal;
        dzdcImagLimbs = workspace.DzdcImagLimbs;
        dzdcImagDigits = reinterpret_cast<uint32_t *>(workspace.DzdcImagOutput);
        dzdcImagControl =
            reinterpret_cast<uint32_t *>(workspace.DzdcImagOutput + capacity + descriptorWords);
        dzdcImagOutput = &combo->Multiply.DzdcImag;
    }

    MattsCudaAssert(limbCount > 0u && limbCount <= capacity);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());

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
                                               descriptors,
                                               carryPrefixShared);

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

    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
        if (realControl[FinalizationNegativeControl] != 0u && index < realDigitLength) {
            const uint32_t lowestNonZero = realControl[FinalizationNonZeroReductionControl];
            if (index < lowestNonZero)
                realDigits[index] = 0u;
            else if (index == lowestNonZero)
                realDigits[index] = 0u - realDigits[index];
            else
                realDigits[index] = ~realDigits[index];
        }
        if (imagControl[FinalizationNegativeControl] != 0u && index < imagDigitLength) {
            const uint32_t lowestNonZero = imagControl[FinalizationNonZeroReductionControl];
            if (index < lowestNonZero)
                imagDigits[index] = 0u;
            else if (index == lowestNonZero)
                imagDigits[index] = 0u - imagDigits[index];
            else
                imagDigits[index] = ~imagDigits[index];
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u && index < dzdcRealDigitLength) {
                const uint32_t lowestNonZero = dzdcRealControl[FinalizationNonZeroReductionControl];
                if (index < lowestNonZero)
                    dzdcRealDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcRealDigits[index] = 0u - dzdcRealDigits[index];
                else
                    dzdcRealDigits[index] = ~dzdcRealDigits[index];
            }

            const uint32_t dzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u && index < dzdcImagDigitLength) {
                const uint32_t lowestNonZero = dzdcImagControl[FinalizationNonZeroReductionControl];
                if (index < lowestNonZero)
                    dzdcImagDigits[index] = 0u;
                else if (index == lowestNonZero)
                    dzdcImagDigits[index] = 0u - dzdcImagDigits[index];
                else
                    dzdcImagDigits[index] = ~dzdcImagDigits[index];
            }
        }
    }

    if (IsLeader<SharkFloatParams>(block)) {
        uint32_t currentRealDigitLength = realControl[FinalizationDigitLengthControl];
        if (realControl[FinalizationNegativeControl] != 0u &&
            realControl[FinalizationNonZeroReductionControl] == currentRealDigitLength) {
            MattsCudaAssert(currentRealDigitLength < capacity);
            if (currentRealDigitLength < capacity)
                realDigits[currentRealDigitLength++] = 1u;
        }
        realControl[FinalizationDigitLengthControl] = currentRealDigitLength;

        uint32_t currentImagDigitLength = imagControl[FinalizationDigitLengthControl];
        if (imagControl[FinalizationNegativeControl] != 0u &&
            imagControl[FinalizationNonZeroReductionControl] == currentImagDigitLength) {
            MattsCudaAssert(currentImagDigitLength < capacity);
            if (currentImagDigitLength < capacity)
                imagDigits[currentImagDigitLength++] = 1u;
        }
        imagControl[FinalizationDigitLengthControl] = currentImagDigitLength;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint32_t currentDzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u &&
                dzdcRealControl[FinalizationNonZeroReductionControl] == currentDzdcRealDigitLength) {
                MattsCudaAssert(currentDzdcRealDigitLength < capacity);
                if (currentDzdcRealDigitLength < capacity)
                    dzdcRealDigits[currentDzdcRealDigitLength++] = 1u;
            }
            dzdcRealControl[FinalizationDigitLengthControl] = currentDzdcRealDigitLength;

            uint32_t currentDzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u &&
                dzdcImagControl[FinalizationNonZeroReductionControl] == currentDzdcImagDigitLength) {
                MattsCudaAssert(currentDzdcImagDigitLength < capacity);
                if (currentDzdcImagDigitLength < capacity)
                    dzdcImagDigits[currentDzdcImagDigitLength++] = 1u;
            }
            dzdcImagControl[FinalizationDigitLengthControl] = currentDzdcImagDigitLength;
        }
    }
    grid.sync();

    FindHighestNonZeroPlusOne<SharkFloatParams>(grid,
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

    if constexpr (HpShark::DebugChecksums) {
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::FinalAdd1,
                                  realDigits,
                                  realControl[FinalizationNonZeroReductionControl]);
        StoreReference2DebugState(debugStates,
                                  grid,
                                  block,
                                  DebugStatePurpose::FinalAdd2,
                                  imagDigits,
                                  imagControl[FinalizationNonZeroReductionControl]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::FinalAddDzdc1,
                                      dzdcRealDigits,
                                      dzdcRealControl[FinalizationNonZeroReductionControl]);
            StoreReference2DebugState(debugStates,
                                      grid,
                                      block,
                                      DebugStatePurpose::FinalAddDzdc2,
                                      dzdcImagDigits,
                                      dzdcImagControl[FinalizationNonZeroReductionControl]);
        }
    }

    constexpr uint32_t ActualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int DesiredBit = (static_cast<int>(ActualDigits) - 1) * 32 + 31;
    if (IsLeader<SharkFloatParams>(block)) {
        const uint32_t realHighestNonZeroPlusOne = realControl[FinalizationNonZeroReductionControl];
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

        const uint32_t imagHighestNonZeroPlusOne = imagControl[FinalizationNonZeroReductionControl];
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
                dzdcRealControl[FinalizationNonZeroReductionControl];
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
                dzdcImagControl[FinalizationNonZeroReductionControl];
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
            const uint32_t highestNonZeroPlusOne = realControl[FinalizationNonZeroReductionControl];
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
            const uint32_t highestNonZeroPlusOne = imagControl[FinalizationNonZeroReductionControl];
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
                    dzdcRealControl[FinalizationNonZeroReductionControl];
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
                    dzdcImagControl[FinalizationNonZeroReductionControl];
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
    grid.sync();

    if constexpr (HpShark::Debug) {
        if (IsLeader<SharkFloatParams>(block)) {
            if (realControl[FinalizationNonZeroReductionControl] != 0u) {
                MattsCudaAssert((realOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if (imagControl[FinalizationNonZeroReductionControl] != 0u) {
                MattsCudaAssert((imagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                if (dzdcRealControl[FinalizationNonZeroReductionControl] != 0u) {
                    MattsCudaAssert((dzdcRealOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
                }
                if (dzdcImagControl[FinalizationNonZeroReductionControl] != 0u) {
                    MattsCudaAssert((dzdcImagOutput->Digits[ActualDigits - 1u] & 0x8000'0000u) != 0u);
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
    if (requiredN > HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN) {
        if (IsLeader<SharkFloatParams>(block))
            combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const uint32_t activeN = requiredN < workspace.ActiveMinFusedN ? workspace.ActiveMinFusedN
                                                                   : static_cast<uint32_t>(requiredN);
    MattsCudaAssert(activeN >= workspace.ActiveMinFusedN);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Workspace::MinFusedStages;
    MattsCudaAssert(planSlot < Workspace::PlanCacheEntryCount);
    MattsCudaAssert((workspace.ValidPlanMask & (1u << planSlot)) != 0u);
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
    HpSharkReference2SetupKernel(HpSharkReference2Workspace<SharkFloatParams> *workspace,
                                 const HpSharkFloat<SharkFloatParams> *cReal,
                                 const HpSharkFloat<SharkFloatParams> *cImag,
                                 const HpSharkFloat<SharkFloatParams> *one,
                                 uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ __align__(16) uint64_t sharedData[];
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
        const uint32_t slot = stage - Workspace::MinFusedStages;
        const uint32_t activeN = 1u << stage;
        Reference2Detail::GenerateCachedPlan<SharkFloatParams>(
            grid, block, debugCombo, activeN, *workspace);

        const SharkNTT::PlanPrime &plan = workspace->Plans[slot];
        SharkNTT::RootTables &roots = workspace->PlanRoots[slot];
        const HpSharkReference2ConstantSpectra spectra = workspace->ConstantSpectra[slot];
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const HpSharkFloat<SharkFloatParams> *values[3] = {cReal, cImag, one};
            uint64_t *outputs[3] = {spectra.CReal, spectra.CImag, spectra.One};
            const DebugStatePurpose packedPurposes[3] = {
                DebugStatePurpose::Z0XY, DebugStatePurpose::Z0W0, DebugStatePurpose::Z0W3};
            const DebugStatePurpose forwardPurposes[3] = {
                DebugStatePurpose::Z2XY, DebugStatePurpose::Z2W0, DebugStatePurpose::Z2W3};
            Reference2Detail::PackTwistForwardBatch<SharkFloatParams, 3>(grid,
                                                                         block,
                                                                         sharedData,
                                                                         debugCombo,
                                                                         debugStates,
                                                                         values,
                                                                         plan,
                                                                         roots,
                                                                         outputs,
                                                                         workspace->IgnoredPrecisionBits,
                                                                         packedPurposes,
                                                                         forwardPurposes);
        } else {
            const HpSharkFloat<SharkFloatParams> *values[2] = {cReal, cImag};
            uint64_t *outputs[2] = {spectra.CReal, spectra.CImag};
            const DebugStatePurpose packedPurposes[2] = {DebugStatePurpose::Z0XY,
                                                         DebugStatePurpose::Z0W0};
            const DebugStatePurpose forwardPurposes[2] = {DebugStatePurpose::Z2XY,
                                                          DebugStatePurpose::Z2W0};
            Reference2Detail::PackTwistForwardBatch<SharkFloatParams, 2>(grid,
                                                                         block,
                                                                         sharedData,
                                                                         debugCombo,
                                                                         debugStates,
                                                                         values,
                                                                         plan,
                                                                         roots,
                                                                         outputs,
                                                                         workspace->IgnoredPrecisionBits,
                                                                         packedPurposes,
                                                                         forwardPurposes);
        }
        grid.sync();
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
ComputeHpSharkReference2Setup(const HpShark::LaunchParams &launchParams,
                              cudaStream_t &stream,
                              void *kernelArgs[])
{
    constexpr auto SharedMemSize = HpShark::CalculateNTTSharedMemorySize<SharkFloatParams>();
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
