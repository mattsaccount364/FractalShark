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
    MattsCudaAssert(activeN >= Workspace::MinFusedN && activeN <= Workspace::MaxFusedN);
    MattsCudaAssert((activeN & (activeN - 1u)) == 0u);
    const uint32_t stages = CountTrailingZeros(activeN);
    MattsCudaAssert(stages >= Workspace::MinFusedStages);
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
ResolveCarryPrefixWarp(HpSharkReference2CarryPrefixDescriptor *descriptors, uint32_t part, uint32_t lane)
{
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
        uint64_t transform = CarryPrefixIdentity();
        if (lane < descriptorCount) {
            transform = state == CarryPrefixDescriptorState::Prefix
                            ? LoadCarryPrefixTransform(&descriptors[descriptorIndex].PrefixTransform)
                            : LoadCarryPrefixTransform(&descriptors[descriptorIndex].AggregateTransform);
        }

        uint64_t windowTransform = transform;
#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t olderLow =
                __shfl_down_sync(0xFFFF'FFFFu, static_cast<uint32_t>(windowTransform), offset);
            const uint32_t olderHigh =
                __shfl_down_sync(0xFFFF'FFFFu, static_cast<uint32_t>(windowTransform >> 32u), offset);
            const uint64_t older =
                static_cast<uint64_t>(olderLow) | (static_cast<uint64_t>(olderHigh) << 32u);
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

    const uint32_t exclusiveLow = __shfl_sync(0xFFFF'FFFFu, static_cast<uint32_t>(exclusive), 0);
    const uint32_t exclusiveHigh = __shfl_sync(0xFFFF'FFFFu, static_cast<uint32_t>(exclusive >> 32u), 0);
    return static_cast<uint64_t>(exclusiveLow) | (static_cast<uint64_t>(exclusiveHigh) << 32u);
}

template <class SharkFloatParams>
__device__ void
PrepareSignedCarryPrefixes(cooperative_groups::grid_group &grid,
                           cooperative_groups::thread_block &block,
                           uint32_t limbCount,
                           int64_t *realLimbs,
                           uint64_t *realTransforms,
                           HpSharkReference2CarryPrefixDescriptor *realDescriptors,
                           int64_t *imagLimbs,
                           uint64_t *imagTransforms,
                           HpSharkReference2CarryPrefixDescriptor *imagDescriptors,
                           int64_t *dzdcRealLimbs,
                           uint64_t *dzdcRealTransforms,
                           HpSharkReference2CarryPrefixDescriptor *dzdcRealDescriptors,
                           int64_t *dzdcImagLimbs,
                           uint64_t *dzdcImagTransforms,
                           HpSharkReference2CarryPrefixDescriptor *dzdcImagDescriptors)
{
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize) {
        realTransforms[index] = MakeSignedCarryPrefix(realLimbs[index]);
        imagTransforms[index] = MakeSignedCarryPrefix(imagLimbs[index]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            dzdcRealTransforms[index] = MakeSignedCarryPrefix(dzdcRealLimbs[index]);
            dzdcImagTransforms[index] = MakeSignedCarryPrefix(dzdcImagLimbs[index]);
        }
    }

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (limbCount + blockSize - 1u) / blockSize;
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize) {
        PublishCarryPrefixState(&realDescriptors[part].State, CarryPrefixDescriptorState::Empty);
        PublishCarryPrefixState(&imagDescriptors[part].State, CarryPrefixDescriptorState::Empty);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            PublishCarryPrefixState(&dzdcRealDescriptors[part].State, CarryPrefixDescriptorState::Empty);
            PublishCarryPrefixState(&dzdcImagDescriptors[part].State, CarryPrefixDescriptorState::Empty);
        }
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
PrefixCarryTransformsDLB(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         uint32_t count,
                         uint64_t *realTransforms,
                         HpSharkReference2CarryPrefixDescriptor *realDescriptors,
                         uint64_t *imagTransforms,
                         HpSharkReference2CarryPrefixDescriptor *imagDescriptors,
                         uint64_t *dzdcRealTransforms,
                         HpSharkReference2CarryPrefixDescriptor *dzdcRealDescriptors,
                         uint64_t *dzdcImagTransforms,
                         HpSharkReference2CarryPrefixDescriptor *dzdcImagDescriptors,
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
    constexpr uint32_t SharedWordsPerStream = 2u * CarryPrefixMaxWarps + 1u;
    uint64_t *realWarpAggregates = sharedStorage;
    uint64_t *realWarpPrefixes = realWarpAggregates + CarryPrefixMaxWarps;
    uint64_t *realExclusiveStorage = realWarpPrefixes + CarryPrefixMaxWarps;
    uint64_t *imagWarpAggregates = sharedStorage + SharedWordsPerStream;
    uint64_t *imagWarpPrefixes = imagWarpAggregates + CarryPrefixMaxWarps;
    uint64_t *imagExclusiveStorage = imagWarpPrefixes + CarryPrefixMaxWarps;
    uint64_t *dzdcRealWarpAggregates = sharedStorage + 2u * SharedWordsPerStream;
    uint64_t *dzdcRealWarpPrefixes = dzdcRealWarpAggregates + CarryPrefixMaxWarps;
    uint64_t *dzdcRealExclusiveStorage = dzdcRealWarpPrefixes + CarryPrefixMaxWarps;
    uint64_t *dzdcImagWarpAggregates = sharedStorage + 3u * SharedWordsPerStream;
    uint64_t *dzdcImagWarpPrefixes = dzdcImagWarpAggregates + CarryPrefixMaxWarps;
    uint64_t *dzdcImagExclusiveStorage = dzdcImagWarpPrefixes + CarryPrefixMaxWarps;

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
        uint64_t realInclusive = hasValue ? realTransforms[index] : CarryPrefixIdentity();
        uint64_t imagInclusive = hasValue ? imagTransforms[index] : CarryPrefixIdentity();
        uint64_t dzdcRealInclusive = SharkFloatParams::EnableNewtonRaphson && hasValue
                                         ? dzdcRealTransforms[index]
                                         : CarryPrefixIdentity();
        uint64_t dzdcImagInclusive = SharkFloatParams::EnableNewtonRaphson && hasValue
                                         ? dzdcImagTransforms[index]
                                         : CarryPrefixIdentity();

#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t previousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realInclusive), offset);
            const uint32_t previousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realInclusive >> 32u), offset);
            const uint64_t previous =
                static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
            if (lane >= offset)
                realInclusive = ComposeCarryPrefixes(previous, realInclusive);
        }
#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint32_t previousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagInclusive), offset);
            const uint32_t previousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagInclusive >> 32u), offset);
            const uint64_t previous =
                static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
            if (lane >= offset)
                imagInclusive = ComposeCarryPrefixes(previous, imagInclusive);
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint32_t previousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealInclusive), offset);
                const uint32_t previousHigh = __shfl_up_sync(
                    0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealInclusive >> 32u), offset);
                const uint64_t previous =
                    static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
                if (lane >= offset)
                    dzdcRealInclusive = ComposeCarryPrefixes(previous, dzdcRealInclusive);
            }
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint32_t previousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagInclusive), offset);
                const uint32_t previousHigh = __shfl_up_sync(
                    0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagInclusive >> 32u), offset);
                const uint64_t previous =
                    static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
                if (lane >= offset)
                    dzdcImagInclusive = ComposeCarryPrefixes(previous, dzdcImagInclusive);
            }
        }

        const uint32_t warpEnd = (warp + 1u) * 32u;
        const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
        if (threadIndex == warpLastThread) {
            realWarpAggregates[warp] = realInclusive;
            imagWarpAggregates[warp] = imagInclusive;
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                dzdcRealWarpAggregates[warp] = dzdcRealInclusive;
                dzdcImagWarpAggregates[warp] = dzdcImagInclusive;
            }
        }
        __syncthreads();

        uint64_t realAggregate = CarryPrefixIdentity();
        uint64_t imagAggregate = CarryPrefixIdentity();
        uint64_t dzdcRealAggregate = CarryPrefixIdentity();
        uint64_t dzdcImagAggregate = CarryPrefixIdentity();

        if (threadIndex < 32u) {
            uint64_t realWarpInclusive =
                lane < numWarps ? realWarpAggregates[lane] : CarryPrefixIdentity();
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint32_t previousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realWarpInclusive), offset);
                const uint32_t previousHigh = __shfl_up_sync(
                    0xFFFF'FFFFu, static_cast<uint32_t>(realWarpInclusive >> 32u), offset);
                const uint64_t previous =
                    static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
                if (lane >= offset && lane < numWarps)
                    realWarpInclusive = ComposeCarryPrefixes(previous, realWarpInclusive);
            }
            const uint32_t realPreviousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realWarpInclusive), 1);
            const uint32_t realPreviousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realWarpInclusive >> 32u), 1);
            const uint64_t realPrevious = static_cast<uint64_t>(realPreviousLow) |
                                          (static_cast<uint64_t>(realPreviousHigh) << 32u);
            if (lane < numWarps)
                realWarpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : realPrevious;
            const uint32_t realAggregateLow = __shfl_sync(
                0xFFFF'FFFFu, static_cast<uint32_t>(realWarpInclusive), static_cast<int>(numWarps - 1u));
            const uint32_t realAggregateHigh =
                __shfl_sync(0xFFFF'FFFFu,
                            static_cast<uint32_t>(realWarpInclusive >> 32u),
                            static_cast<int>(numWarps - 1u));
            realAggregate = static_cast<uint64_t>(realAggregateLow) |
                            (static_cast<uint64_t>(realAggregateHigh) << 32u);

            uint64_t imagWarpInclusive =
                lane < numWarps ? imagWarpAggregates[lane] : CarryPrefixIdentity();
#pragma unroll
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint32_t previousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagWarpInclusive), offset);
                const uint32_t previousHigh = __shfl_up_sync(
                    0xFFFF'FFFFu, static_cast<uint32_t>(imagWarpInclusive >> 32u), offset);
                const uint64_t previous =
                    static_cast<uint64_t>(previousLow) | (static_cast<uint64_t>(previousHigh) << 32u);
                if (lane >= offset && lane < numWarps)
                    imagWarpInclusive = ComposeCarryPrefixes(previous, imagWarpInclusive);
            }
            const uint32_t imagPreviousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagWarpInclusive), 1);
            const uint32_t imagPreviousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagWarpInclusive >> 32u), 1);
            const uint64_t imagPrevious = static_cast<uint64_t>(imagPreviousLow) |
                                          (static_cast<uint64_t>(imagPreviousHigh) << 32u);
            if (lane < numWarps)
                imagWarpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : imagPrevious;
            const uint32_t imagAggregateLow = __shfl_sync(
                0xFFFF'FFFFu, static_cast<uint32_t>(imagWarpInclusive), static_cast<int>(numWarps - 1u));
            const uint32_t imagAggregateHigh =
                __shfl_sync(0xFFFF'FFFFu,
                            static_cast<uint32_t>(imagWarpInclusive >> 32u),
                            static_cast<int>(numWarps - 1u));
            imagAggregate = static_cast<uint64_t>(imagAggregateLow) |
                            (static_cast<uint64_t>(imagAggregateHigh) << 32u);

            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                uint64_t dzdcRealWarpInclusive =
                    lane < numWarps ? dzdcRealWarpAggregates[lane] : CarryPrefixIdentity();
#pragma unroll
                for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                    const uint32_t previousLow = __shfl_up_sync(
                        0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealWarpInclusive), offset);
                    const uint32_t previousHigh = __shfl_up_sync(
                        0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealWarpInclusive >> 32u), offset);
                    const uint64_t previous = static_cast<uint64_t>(previousLow) |
                                              (static_cast<uint64_t>(previousHigh) << 32u);
                    if (lane >= offset && lane < numWarps) {
                        dzdcRealWarpInclusive = ComposeCarryPrefixes(previous, dzdcRealWarpInclusive);
                    }
                }
                const uint32_t dzdcRealPreviousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealWarpInclusive), 1);
                const uint32_t dzdcRealPreviousHigh =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealWarpInclusive >> 32u), 1);
                const uint64_t dzdcRealPrevious = static_cast<uint64_t>(dzdcRealPreviousLow) |
                                                  (static_cast<uint64_t>(dzdcRealPreviousHigh) << 32u);
                if (lane < numWarps) {
                    dzdcRealWarpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : dzdcRealPrevious;
                }
                const uint32_t dzdcRealAggregateLow =
                    __shfl_sync(0xFFFF'FFFFu,
                                static_cast<uint32_t>(dzdcRealWarpInclusive),
                                static_cast<int>(numWarps - 1u));
                const uint32_t dzdcRealAggregateHigh =
                    __shfl_sync(0xFFFF'FFFFu,
                                static_cast<uint32_t>(dzdcRealWarpInclusive >> 32u),
                                static_cast<int>(numWarps - 1u));
                dzdcRealAggregate = static_cast<uint64_t>(dzdcRealAggregateLow) |
                                    (static_cast<uint64_t>(dzdcRealAggregateHigh) << 32u);

                uint64_t dzdcImagWarpInclusive =
                    lane < numWarps ? dzdcImagWarpAggregates[lane] : CarryPrefixIdentity();
#pragma unroll
                for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                    const uint32_t previousLow = __shfl_up_sync(
                        0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagWarpInclusive), offset);
                    const uint32_t previousHigh = __shfl_up_sync(
                        0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagWarpInclusive >> 32u), offset);
                    const uint64_t previous = static_cast<uint64_t>(previousLow) |
                                              (static_cast<uint64_t>(previousHigh) << 32u);
                    if (lane >= offset && lane < numWarps) {
                        dzdcImagWarpInclusive = ComposeCarryPrefixes(previous, dzdcImagWarpInclusive);
                    }
                }
                const uint32_t dzdcImagPreviousLow =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagWarpInclusive), 1);
                const uint32_t dzdcImagPreviousHigh =
                    __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagWarpInclusive >> 32u), 1);
                const uint64_t dzdcImagPrevious = static_cast<uint64_t>(dzdcImagPreviousLow) |
                                                  (static_cast<uint64_t>(dzdcImagPreviousHigh) << 32u);
                if (lane < numWarps) {
                    dzdcImagWarpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : dzdcImagPrevious;
                }
                const uint32_t dzdcImagAggregateLow =
                    __shfl_sync(0xFFFF'FFFFu,
                                static_cast<uint32_t>(dzdcImagWarpInclusive),
                                static_cast<int>(numWarps - 1u));
                const uint32_t dzdcImagAggregateHigh =
                    __shfl_sync(0xFFFF'FFFFu,
                                static_cast<uint32_t>(dzdcImagWarpInclusive >> 32u),
                                static_cast<int>(numWarps - 1u));
                dzdcImagAggregate = static_cast<uint64_t>(dzdcImagAggregateLow) |
                                    (static_cast<uint64_t>(dzdcImagAggregateHigh) << 32u);
            }
        }

        if (threadIndex == 0u) {
            PublishCarryPrefixDescriptorAggregate(realDescriptors[part], realAggregate);
            PublishCarryPrefixDescriptorAggregate(imagDescriptors[part], imagAggregate);
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                PublishCarryPrefixDescriptorAggregate(dzdcRealDescriptors[part], dzdcRealAggregate);
                PublishCarryPrefixDescriptorAggregate(dzdcImagDescriptors[part], dzdcImagAggregate);
            }
        }

        if (threadIndex < 32u) {
            __syncwarp(0xFFFF'FFFFu);
            const uint64_t realExclusive = ResolveCarryPrefixWarp(realDescriptors, part, lane);
            const uint64_t imagExclusive = ResolveCarryPrefixWarp(imagDescriptors, part, lane);
            if (lane == 0u) {
                PublishCarryPrefixDescriptorPrefix(realDescriptors[part],
                                                   ComposeCarryPrefixes(realExclusive, realAggregate));
                PublishCarryPrefixDescriptorPrefix(imagDescriptors[part],
                                                   ComposeCarryPrefixes(imagExclusive, imagAggregate));
                realExclusiveStorage[0] = realExclusive;
                imagExclusiveStorage[0] = imagExclusive;
            }
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                const uint64_t dzdcRealExclusive =
                    ResolveCarryPrefixWarp(dzdcRealDescriptors, part, lane);
                const uint64_t dzdcImagExclusive =
                    ResolveCarryPrefixWarp(dzdcImagDescriptors, part, lane);
                if (lane == 0u) {
                    PublishCarryPrefixDescriptorPrefix(
                        dzdcRealDescriptors[part],
                        ComposeCarryPrefixes(dzdcRealExclusive, dzdcRealAggregate));
                    PublishCarryPrefixDescriptorPrefix(
                        dzdcImagDescriptors[part],
                        ComposeCarryPrefixes(dzdcImagExclusive, dzdcImagAggregate));
                    dzdcRealExclusiveStorage[0] = dzdcRealExclusive;
                    dzdcImagExclusiveStorage[0] = dzdcImagExclusive;
                }
            }
        }
        __syncthreads();

        const uint32_t realPreviousLow =
            __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realInclusive), 1);
        const uint32_t realPreviousHigh =
            __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(realInclusive >> 32u), 1);
        const uint64_t realPrevious =
            static_cast<uint64_t>(realPreviousLow) | (static_cast<uint64_t>(realPreviousHigh) << 32u);
        const uint64_t realLocalExclusive = lane == 0u ? CarryPrefixIdentity() : realPrevious;
        const uint32_t imagPreviousLow =
            __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagInclusive), 1);
        const uint32_t imagPreviousHigh =
            __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(imagInclusive >> 32u), 1);
        const uint64_t imagPrevious =
            static_cast<uint64_t>(imagPreviousLow) | (static_cast<uint64_t>(imagPreviousHigh) << 32u);
        const uint64_t imagLocalExclusive = lane == 0u ? CarryPrefixIdentity() : imagPrevious;
        if (hasValue) {
            realTransforms[index] =
                ComposeCarryPrefixes(realExclusiveStorage[0],
                                     ComposeCarryPrefixes(realWarpPrefixes[warp], realLocalExclusive));
            imagTransforms[index] =
                ComposeCarryPrefixes(imagExclusiveStorage[0],
                                     ComposeCarryPrefixes(imagWarpPrefixes[warp], imagLocalExclusive));
        }
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const uint32_t dzdcRealPreviousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealInclusive), 1);
            const uint32_t dzdcRealPreviousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcRealInclusive >> 32u), 1);
            const uint64_t dzdcRealPrevious = static_cast<uint64_t>(dzdcRealPreviousLow) |
                                              (static_cast<uint64_t>(dzdcRealPreviousHigh) << 32u);
            const uint64_t dzdcRealLocalExclusive =
                lane == 0u ? CarryPrefixIdentity() : dzdcRealPrevious;
            const uint32_t dzdcImagPreviousLow =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagInclusive), 1);
            const uint32_t dzdcImagPreviousHigh =
                __shfl_up_sync(0xFFFF'FFFFu, static_cast<uint32_t>(dzdcImagInclusive >> 32u), 1);
            const uint64_t dzdcImagPrevious = static_cast<uint64_t>(dzdcImagPreviousLow) |
                                              (static_cast<uint64_t>(dzdcImagPreviousHigh) << 32u);
            const uint64_t dzdcImagLocalExclusive =
                lane == 0u ? CarryPrefixIdentity() : dzdcImagPrevious;
            if (hasValue) {
                dzdcRealTransforms[index] = ComposeCarryPrefixes(
                    dzdcRealExclusiveStorage[0],
                    ComposeCarryPrefixes(dzdcRealWarpPrefixes[warp], dzdcRealLocalExclusive));
                dzdcImagTransforms[index] = ComposeCarryPrefixes(
                    dzdcImagExclusiveStorage[0],
                    ComposeCarryPrefixes(dzdcImagWarpPrefixes[warp], dzdcImagLocalExclusive));
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
    constexpr uint32_t Capacity = Workspace::MaxFusedLimbs;
    constexpr uint32_t DescriptorWords =
        (Workspace::MaxCarryPrefixParts * sizeof(Descriptor) + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    constexpr uint32_t ControlWords =
        (Workspace::CarryPrefixControlCount * sizeof(uint32_t) + sizeof(uint64_t) - 1u) /
        sizeof(uint64_t);
    static_assert((Capacity * sizeof(uint64_t)) % alignof(Descriptor) == 0u);
    static_assert(Capacity + DescriptorWords + ControlWords <= Workspace::MaxFusedN);

    uint64_t *realTransforms = workspace.RealOutput;
    int64_t *realLimbs = workspace.RealLimbs;
    uint32_t *realDigits = reinterpret_cast<uint32_t *>(realTransforms);
    Descriptor *realDescriptors = reinterpret_cast<Descriptor *>(realTransforms + Capacity);
    uint32_t *realControl = reinterpret_cast<uint32_t *>(realTransforms + Capacity + DescriptorWords);
    HpSharkFloat<SharkFloatParams> *realOutput = &combo->Multiply.A;

    uint64_t *imagTransforms = workspace.ImagOutput;
    int64_t *imagLimbs = workspace.ImagLimbs;
    uint32_t *imagDigits = reinterpret_cast<uint32_t *>(imagTransforms);
    Descriptor *imagDescriptors = reinterpret_cast<Descriptor *>(imagTransforms + Capacity);
    uint32_t *imagControl = reinterpret_cast<uint32_t *>(imagTransforms + Capacity + DescriptorWords);
    HpSharkFloat<SharkFloatParams> *imagOutput = &combo->Multiply.B;

    uint64_t *dzdcRealTransforms = nullptr;
    int64_t *dzdcRealLimbs = nullptr;
    uint32_t *dzdcRealDigits = nullptr;
    Descriptor *dzdcRealDescriptors = nullptr;
    uint32_t *dzdcRealControl = nullptr;
    HpSharkFloat<SharkFloatParams> *dzdcRealOutput = nullptr;
    uint64_t *dzdcImagTransforms = nullptr;
    int64_t *dzdcImagLimbs = nullptr;
    uint32_t *dzdcImagDigits = nullptr;
    Descriptor *dzdcImagDescriptors = nullptr;
    uint32_t *dzdcImagControl = nullptr;
    HpSharkFloat<SharkFloatParams> *dzdcImagOutput = nullptr;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealTransforms = workspace.DzdcRealOutput;
        dzdcRealLimbs = workspace.DzdcRealLimbs;
        dzdcRealDigits = reinterpret_cast<uint32_t *>(dzdcRealTransforms);
        dzdcRealDescriptors = reinterpret_cast<Descriptor *>(dzdcRealTransforms + Capacity);
        dzdcRealControl = reinterpret_cast<uint32_t *>(dzdcRealTransforms + Capacity + DescriptorWords);
        dzdcRealOutput = &combo->Multiply.DzdcReal;
        dzdcImagTransforms = workspace.DzdcImagOutput;
        dzdcImagLimbs = workspace.DzdcImagLimbs;
        dzdcImagDigits = reinterpret_cast<uint32_t *>(dzdcImagTransforms);
        dzdcImagDescriptors = reinterpret_cast<Descriptor *>(dzdcImagTransforms + Capacity);
        dzdcImagControl = reinterpret_cast<uint32_t *>(dzdcImagTransforms + Capacity + DescriptorWords);
        dzdcImagOutput = &combo->Multiply.DzdcImag;
    }

    MattsCudaAssert(limbCount > 0u && limbCount <= Capacity);

    PrepareSignedCarryPrefixes<SharkFloatParams>(grid,
                                                 block,
                                                 limbCount,
                                                 realLimbs,
                                                 realTransforms,
                                                 realDescriptors,
                                                 imagLimbs,
                                                 imagTransforms,
                                                 imagDescriptors,
                                                 dzdcRealLimbs,
                                                 dzdcRealTransforms,
                                                 dzdcRealDescriptors,
                                                 dzdcImagLimbs,
                                                 dzdcImagTransforms,
                                                 dzdcImagDescriptors);
    PrefixCarryTransformsDLB<SharkFloatParams>(grid,
                                               block,
                                               limbCount,
                                               realTransforms,
                                               realDescriptors,
                                               imagTransforms,
                                               imagDescriptors,
                                               dzdcRealTransforms,
                                               dzdcRealDescriptors,
                                               dzdcImagTransforms,
                                               dzdcImagDescriptors,
                                               carryPrefixShared);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize) {
        const int64_t realSignedLimb = realLimbs[index];
        const int32_t realCarryIn = ApplyCarryPrefix(realTransforms[index], 0);
        const uint32_t realDigit =
            static_cast<uint32_t>(static_cast<uint64_t>(realSignedLimb + realCarryIn));
        if (index + 1u == limbCount) {
            int32_t finalCarry = CarryOutForSignedLimb(realSignedLimb, realCarryIn);
            uint32_t digitLength = limbCount;
            while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
                realLimbs[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
                finalCarry = CarryOutForSignedLimb(finalCarry, 0);
            }
            realControl[FinalizationDigitLengthControl] = digitLength;
            realControl[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
        }
        realLimbs[index] = realDigit;

        const int64_t imagSignedLimb = imagLimbs[index];
        const int32_t imagCarryIn = ApplyCarryPrefix(imagTransforms[index], 0);
        const uint32_t imagDigit =
            static_cast<uint32_t>(static_cast<uint64_t>(imagSignedLimb + imagCarryIn));
        if (index + 1u == limbCount) {
            int32_t finalCarry = CarryOutForSignedLimb(imagSignedLimb, imagCarryIn);
            uint32_t digitLength = limbCount;
            while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
                imagLimbs[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
                finalCarry = CarryOutForSignedLimb(finalCarry, 0);
            }
            imagControl[FinalizationDigitLengthControl] = digitLength;
            imagControl[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
        }
        imagLimbs[index] = imagDigit;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            const int64_t dzdcRealSignedLimb = dzdcRealLimbs[index];
            const int32_t dzdcRealCarryIn = ApplyCarryPrefix(dzdcRealTransforms[index], 0);
            const uint32_t dzdcRealDigit =
                static_cast<uint32_t>(static_cast<uint64_t>(dzdcRealSignedLimb + dzdcRealCarryIn));
            if (index + 1u == limbCount) {
                int32_t finalCarry = CarryOutForSignedLimb(dzdcRealSignedLimb, dzdcRealCarryIn);
                uint32_t digitLength = limbCount;
                while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
                    dzdcRealLimbs[digitLength++] =
                        static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
                    finalCarry = CarryOutForSignedLimb(finalCarry, 0);
                }
                dzdcRealControl[FinalizationDigitLengthControl] = digitLength;
                dzdcRealControl[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
            }
            dzdcRealLimbs[index] = dzdcRealDigit;

            const int64_t dzdcImagSignedLimb = dzdcImagLimbs[index];
            const int32_t dzdcImagCarryIn = ApplyCarryPrefix(dzdcImagTransforms[index], 0);
            const uint32_t dzdcImagDigit =
                static_cast<uint32_t>(static_cast<uint64_t>(dzdcImagSignedLimb + dzdcImagCarryIn));
            if (index + 1u == limbCount) {
                int32_t finalCarry = CarryOutForSignedLimb(dzdcImagSignedLimb, dzdcImagCarryIn);
                uint32_t digitLength = limbCount;
                while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
                    dzdcImagLimbs[digitLength++] =
                        static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
                    finalCarry = CarryOutForSignedLimb(finalCarry, 0);
                }
                dzdcImagControl[FinalizationDigitLengthControl] = digitLength;
                dzdcImagControl[FinalizationNegativeControl] = finalCarry < 0 ? 1u : 0u;
            }
            dzdcImagLimbs[index] = dzdcImagDigit;
        }
    }
    grid.sync();

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

    for (uint32_t index = GridThreadRank(block); index < maximumDigitLength; index += gridSize) {
        if (index < realDigitLength)
            realDigits[index] = static_cast<uint32_t>(realLimbs[index]);
        if (index < imagDigitLength)
            imagDigits[index] = static_cast<uint32_t>(imagLimbs[index]);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            if (index < dzdcRealControl[FinalizationDigitLengthControl])
                dzdcRealDigits[index] = static_cast<uint32_t>(dzdcRealLimbs[index]);
            if (index < dzdcImagControl[FinalizationDigitLengthControl])
                dzdcImagDigits[index] = static_cast<uint32_t>(dzdcImagLimbs[index]);
        }
    }
    grid.sync();

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
            MattsCudaAssert(currentRealDigitLength < Capacity);
            if (currentRealDigitLength < Capacity)
                realDigits[currentRealDigitLength++] = 1u;
        }
        realControl[FinalizationDigitLengthControl] = currentRealDigitLength;

        uint32_t currentImagDigitLength = imagControl[FinalizationDigitLengthControl];
        if (imagControl[FinalizationNegativeControl] != 0u &&
            imagControl[FinalizationNonZeroReductionControl] == currentImagDigitLength) {
            MattsCudaAssert(currentImagDigitLength < Capacity);
            if (currentImagDigitLength < Capacity)
                imagDigits[currentImagDigitLength++] = 1u;
        }
        imagControl[FinalizationDigitLengthControl] = currentImagDigitLength;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint32_t currentDzdcRealDigitLength = dzdcRealControl[FinalizationDigitLengthControl];
            if (dzdcRealControl[FinalizationNegativeControl] != 0u &&
                dzdcRealControl[FinalizationNonZeroReductionControl] == currentDzdcRealDigitLength) {
                MattsCudaAssert(currentDzdcRealDigitLength < Capacity);
                if (currentDzdcRealDigitLength < Capacity)
                    dzdcRealDigits[currentDzdcRealDigitLength++] = 1u;
            }
            dzdcRealControl[FinalizationDigitLengthControl] = currentDzdcRealDigitLength;

            uint32_t currentDzdcImagDigitLength = dzdcImagControl[FinalizationDigitLengthControl];
            if (dzdcImagControl[FinalizationNegativeControl] != 0u &&
                dzdcImagControl[FinalizationNonZeroReductionControl] == currentDzdcImagDigitLength) {
                MattsCudaAssert(currentDzdcImagDigitLength < Capacity);
                if (currentDzdcImagDigitLength < Capacity)
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
    const uint32_t activeN =
        requiredN < Workspace::MinFusedN ? Workspace::MinFusedN : static_cast<uint32_t>(requiredN);
    MattsCudaAssert(activeN >= Workspace::MinFusedN);
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
    for (uint32_t slot = 0; slot < Workspace::PlanCacheEntryCount; ++slot) {
        const uint32_t activeN = 1u << (Workspace::MinFusedStages + slot);
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

    constexpr uint32_t FullPlanMask =
        Workspace::PlanCacheEntryCount == 32u ? ~0u : (1u << Workspace::PlanCacheEntryCount) - 1u;
    Reference2Detail::MattsCudaAssert(workspace->ValidPlanMask == FullPlanMask);
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
