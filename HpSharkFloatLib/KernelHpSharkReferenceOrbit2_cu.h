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
__device__ uint64_t
MontgomeryPowSerial(cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t aMont,
                    uint64_t exponent)
{
    uint64_t result = SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1);
    while (exponent != 0) {
        if ((exponent & 1ull) != 0)
            result = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, result, aMont);
        aMont = SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, aMont, aMont);
        exponent >>= 1;
    }
    return result;
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
                int32_t exponentOffset)
{
    if (IsZero(a) || IsZero(b))
        return {true, false, 0, TermKind::Product, aId, bId};
    return {false,
            static_cast<bool>(a.GetNegative() ^ b.GetNegative() ^ negate),
            static_cast<int32_t>(a.Exponent + b.Exponent + exponentOffset),
            TermKind::Product,
            aId,
            bId};
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly FusedTerm<SharkFloatParams>
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a, SpectrumId aId, bool negate)
{
    if (IsZero(a))
        return {true, false, 0, TermKind::Linear, aId, aId};
    return {false, static_cast<bool>(a.GetNegative() ^ negate), a.Exponent, TermKind::Linear, aId, aId};
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
RequiredBitsForTerm(int32_t commonExponent, FusedTerm<SharkFloatParams> term)
{
    constexpr uint64_t MantissaBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    if (term.IsZero)
        return 0;
    const uint64_t shift = static_cast<uint64_t>(term.Exponent - commonExponent);
    const uint64_t width = term.Kind == TermKind::Product ? 2ull * MantissaBits : MantissaBits;
    return shift + width;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
IncludeTermInRequiredBits(int32_t commonExponent, FusedTerm<SharkFloatParams> term, uint64_t &required)
{
    const uint64_t termBits = RequiredBitsForTerm(commonExponent, term);
    required = required > termBits ? required : termBits;
}

template <class SharkFloatParams, class... RemainingTerms>
__device__ SharkForceInlineReleaseOnly uint64_t
RequiredBitsForTerms(int32_t commonExponent,
                     FusedTerm<SharkFloatParams> firstTerm,
                     RemainingTerms... remainingTerms)
{
    static_assert((std::is_same_v<FusedTerm<SharkFloatParams>, RemainingTerms> && ...));
    uint64_t required = RequiredBitsForTerm(commonExponent, firstTerm);
    (IncludeTermInRequiredBits(commonExponent, remainingTerms, required), ...);
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
__device__ void
GenerateActiveRoots(cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint32_t activeN,
                    HpSharkReference2Workspace<SharkFloatParams> &workspace)
{
    if (workspace.CachedN == activeN)
        return;

    const uint32_t stages = CountTrailingZeros(activeN);
    SharkNTT::RootTables &roots = workspace.Roots;
    if (IsLeader<SharkFloatParams>(block)) {
        roots.N = static_cast<int32_t>(activeN);
        roots.stages = static_cast<int32_t>(stages);
        roots.total_twiddles = activeN - 1;
    }

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
        for (uint32_t i = rank; i < activeN; i += gridSize) {
            roots.psi_pows[i] = psiPower;
            roots.psi_inv_pows[i] = psiInversePower;
            if (i + gridSize < activeN) {
                psiPower = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, psiPower, psiStride);
                psiInversePower = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    grid, block, debugCombo, psiInversePower, psiInverseStride);
            }
        }
    }

    uint32_t offset = 0;
    for (uint32_t stage = 1; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1;
        if (IsLeader<SharkFloatParams>(block)) {
            roots.stage_omegas[stage - 1] =
                MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, activeN / width);
            roots.stage_omegas_inv[stage - 1] = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, omegaInverse, activeN / width);
        }
        grid.sync();
        if (rank < half) {
            uint64_t forwardTwiddle = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas[stage - 1], rank);
            uint64_t inverseTwiddle = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas_inv[stage - 1], rank);
            const uint64_t forwardStride = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas[stage - 1], gridSize);
            const uint64_t inverseStride = MontgomeryPowSerial<SharkFloatParams>(
                grid, block, debugCombo, roots.stage_omegas_inv[stage - 1], gridSize);
            for (uint32_t j = rank; j < half; j += gridSize) {
                roots.stage_twiddles_fwd[offset + j] = forwardTwiddle;
                roots.stage_twiddles_inv[offset + j] = inverseTwiddle;
                if (j + gridSize < half) {
                    forwardTwiddle = SharkNTT::MontgomeryMul<SharkFloatParams>(
                        grid, block, debugCombo, forwardTwiddle, forwardStride);
                    inverseTwiddle = SharkNTT::MontgomeryMul<SharkFloatParams>(
                        grid, block, debugCombo, inverseTwiddle, inverseStride);
                }
            }
        }
        offset += half;
    }

    if (IsLeader<SharkFloatParams>(block)) {
        const uint64_t inverseTwo = SharkNTT::ToMontgomery<SharkFloatParams>(
            grid, block, debugCombo, (SharkNTT::MagicPrime + 1ull) >> 1);
        roots.Ninvm_mont =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, inverseTwo, stages);
        workspace.CachedN = activeN;
    }
    // Root-table ranges are disjoint; publish the complete table once after all generators finish.
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
                      const DebugStatePurpose packedPurposes[BatchSize],
                      const DebugStatePurpose forwardPurposes[BatchSize])
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize) {
#pragma unroll
        for (int buffer = 0; buffer < BatchSize; ++buffer) {
            const uint64_t coefficient =
                i < static_cast<uint32_t>(plan.L)
                    ? ReadBitsSimple(*values[buffer], static_cast<int64_t>(i) * plan.b, plan.b)
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

template <class SharkFloatParams>
__device__ void
WriteShiftedSpectrum(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     const SharkNTT::PlanPrime &plan,
                     const SharkNTT::RootTables &roots,
                     const uint64_t *source,
                     uint64_t shiftBits,
                     bool negative,
                     uint64_t *dest)
{
    const uint64_t chunkShift = shiftBits / static_cast<uint64_t>(plan.b);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    const uint64_t bitScale =
        SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1ull << bitShift);
    const uint64_t zeroMont = SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 0);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < static_cast<uint32_t>(plan.N); i += gridSize) {
        const uint64_t chunkScale =
            chunkShift == 0 ? SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1)
                            : PsiPowerMont<SharkFloatParams>(
                                  grid, block, debugCombo, plan, roots, chunkShift * (1ull + 2ull * i));
        const uint64_t scale =
            SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, chunkScale, bitScale);
        const uint64_t shifted =
            SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, source[i], scale);
        dest[i] = negative ? SubPSerial(zeroMont, shifted) : shifted;
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ void
AddShiftedSpectrum(cooperative_groups::grid_group &grid,
                   cooperative_groups::thread_block &block,
                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                   const SharkNTT::PlanPrime &plan,
                   const SharkNTT::RootTables &roots,
                   const uint64_t *source,
                   uint64_t shiftBits,
                   bool negative,
                   uint64_t *dest)
{
    const uint64_t chunkShift = shiftBits / static_cast<uint64_t>(plan.b);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    const uint64_t bitScale =
        SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1ull << bitShift);
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t i = GridThreadRank(block); i < static_cast<uint32_t>(plan.N); i += gridSize) {
        const uint64_t chunkScale =
            chunkShift == 0 ? SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, 1)
                            : PsiPowerMont<SharkFloatParams>(
                                  grid, block, debugCombo, plan, roots, chunkShift * (1ull + 2ull * i));
        const uint64_t scale =
            SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, chunkScale, bitScale);
        const uint64_t shifted =
            SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, source[i], scale);
        dest[i] = negative ? SubPSerial(dest[i], shifted) : AddPSerial(dest[i], shifted);
    }
    grid.sync();
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly uint64_t *
GetSpectrum(HpSharkReference2Workspace<SharkFloatParams> &workspace, SpectrumId id)
{
    switch (id) {
        case SpectrumId::ZReal:
            return workspace.ZReal;
        case SpectrumId::ZImag:
            return workspace.ZImag;
        case SpectrumId::DzdcReal:
            return workspace.DzdcReal;
        case SpectrumId::DzdcImag:
            return workspace.DzdcImag;
        case SpectrumId::CReal:
            return workspace.CReal;
        case SpectrumId::CImag:
            return workspace.CImag;
        case SpectrumId::One:
            return workspace.One;
    }
    return workspace.ZReal;
}

template <class SharkFloatParams>
__device__ SharkForceInlineReleaseOnly void
AccumulateOutputTerm(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     const SharkNTT::PlanPrime &plan,
                     const SharkNTT::RootTables &roots,
                     HpSharkReference2Workspace<SharkFloatParams> &workspace,
                     int32_t commonExponent,
                     uint64_t *dest,
                     FusedTerm<SharkFloatParams> term,
                     bool &hasDestinationValue)
{
    if (term.IsZero)
        return;
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    const uint64_t shift = static_cast<uint64_t>(term.Exponent - commonExponent);
    if (term.Kind == TermKind::Product) {
        const uint64_t *a = GetSpectrum(workspace, term.A);
        const uint64_t *b = GetSpectrum(workspace, term.B);
        const uint32_t gridSize = static_cast<uint32_t>(grid.size());
        for (uint32_t i = GridThreadRank(block); i < activeN; i += gridSize)
            workspace.Product[i] =
                SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, a[i], b[i]);
        // The shifted pass has identical grid-stride ownership and supplies the publication barrier.
        if (hasDestinationValue) {
            AddShiftedSpectrum<SharkFloatParams>(
                grid, block, debugCombo, plan, roots, workspace.Product, shift, term.IsNegative, dest);
        } else {
            WriteShiftedSpectrum<SharkFloatParams>(
                grid, block, debugCombo, plan, roots, workspace.Product, shift, term.IsNegative, dest);
        }
    } else {
        const uint64_t *source = GetSpectrum(workspace, term.A);
        if (hasDestinationValue) {
            AddShiftedSpectrum<SharkFloatParams>(
                grid, block, debugCombo, plan, roots, source, shift, term.IsNegative, dest);
        } else {
            WriteShiftedSpectrum<SharkFloatParams>(
                grid, block, debugCombo, plan, roots, source, shift, term.IsNegative, dest);
        }
    }
    hasDestinationValue = true;
}

template <class SharkFloatParams, class... RemainingTerms>
__device__ SharkForceInlineReleaseOnly void
AccumulateOutputSpectrum(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         DebugGlobalCount<SharkFloatParams> *debugCombo,
                         DebugState<SharkFloatParams> *debugStates,
                         const SharkNTT::PlanPrime &plan,
                         const SharkNTT::RootTables &roots,
                         HpSharkReference2Workspace<SharkFloatParams> &workspace,
                         int32_t commonExponent,
                         uint64_t *dest,
                         DebugStatePurpose checksumPurpose,
                         FusedTerm<SharkFloatParams> firstTerm,
                         RemainingTerms... remainingTerms)
{
    static_assert((std::is_same_v<FusedTerm<SharkFloatParams>, RemainingTerms> && ...));
    bool hasDestinationValue = false;
    AccumulateOutputTerm(grid,
                         block,
                         debugCombo,
                         plan,
                         roots,
                         workspace,
                         commonExponent,
                         dest,
                         firstTerm,
                         hasDestinationValue);
    (AccumulateOutputTerm(grid,
                          block,
                          debugCombo,
                          plan,
                          roots,
                          workspace,
                          commonExponent,
                          dest,
                          remainingTerms,
                          hasDestinationValue),
     ...);
    MattsCudaAssert(hasDestinationValue);
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    StoreReference2DebugState(debugStates, grid, block, checksumPurpose, dest, activeN);
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

template <class SharkFloatParams>
__device__ uint32_t
FindHighestNonZeroPlusOne(cooperative_groups::grid_group &grid,
                          cooperative_groups::thread_block &block,
                          const uint32_t *values,
                          uint32_t count,
                          uint32_t *result,
                          uint64_t *sharedStorage)
{
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMaximum = reinterpret_cast<uint32_t *>(sharedStorage);

    if (IsLeader<SharkFloatParams>(block))
        *result = 0u;
    if (threadIndex == 0u)
        *blockMaximum = 0u;
    grid.sync();

    uint32_t localMaximum = 0u;
    for (uint32_t index = GridThreadRank(block); index < count; index += gridSize) {
        if (values[index] != 0u)
            localMaximum = index + 1u;
    }

    if (localMaximum != 0u)
        atomicMax(blockMaximum, localMaximum);
    __syncthreads();

    if (threadIndex == 0u && *blockMaximum != 0u)
        atomicMax(result, *blockMaximum);
    grid.sync();

    const uint32_t highestNonZeroPlusOne = *result;
    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < count; index += gridSize) {
            if (index >= highestNonZeroPlusOne)
                MattsCudaAssert(values[index] == 0u);
            if (index + 1u == highestNonZeroPlusOne)
                MattsCudaAssert(values[index] != 0u);
        }
        grid.sync();
    }
    return highestNonZeroPlusOne;
}

template <class SharkFloatParams>
__device__ uint32_t
FindLowestNonZero(cooperative_groups::grid_group &grid,
                  cooperative_groups::thread_block &block,
                  const uint32_t *values,
                  uint32_t count,
                  uint32_t *result,
                  uint64_t *sharedStorage)
{
    const uint32_t threadIndex = block.thread_index().x;
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    uint32_t *blockMinimum = reinterpret_cast<uint32_t *>(sharedStorage);

    if (IsLeader<SharkFloatParams>(block))
        *result = count;
    if (threadIndex == 0u)
        *blockMinimum = count;
    grid.sync();

    uint32_t localMinimum = count;
    for (uint32_t index = GridThreadRank(block); index < count; index += gridSize) {
        if (values[index] != 0u)
            localMinimum = localMinimum < index ? localMinimum : index;
    }

    if (localMinimum != count)
        atomicMin(blockMinimum, localMinimum);
    __syncthreads();

    if (threadIndex == 0u && *blockMinimum != count)
        atomicMin(result, *blockMinimum);
    grid.sync();

    const uint32_t lowestNonZero = *result;
    if constexpr (HpShark::Debug) {
        for (uint32_t index = GridThreadRank(block); index < count; index += gridSize) {
            if (index < lowestNonZero)
                MattsCudaAssert(values[index] == 0u);
            if (index == lowestNonZero)
                MattsCudaAssert(values[index] != 0u);
        }
        grid.sync();
    }
    return lowestNonZero;
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

static __device__ void
PrepareSignedCarryPrefixes(cooperative_groups::grid_group &grid,
                           cooperative_groups::thread_block &block,
                           const int64_t *limbs,
                           uint32_t limbCount,
                           uint64_t *transforms,
                           HpSharkReference2CarryPrefixDescriptor *descriptors)
{
    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize)
        transforms[index] = MakeSignedCarryPrefix(limbs[index]);

    const uint32_t blockSize = block.dim_threads().x;
    const uint32_t numParts = (limbCount + blockSize - 1u) / blockSize;
    for (uint32_t part = GridThreadRank(block); part < numParts; part += gridSize)
        PublishCarryPrefixState(&descriptors[part].State, CarryPrefixDescriptorState::Empty);
    grid.sync();
}

static __device__ void
PrefixCarryTransformsDLB(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         uint64_t *transforms,
                         uint32_t count,
                         HpSharkReference2CarryPrefixDescriptor *descriptors,
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
    const cooperative_groups::thread_block_tile<32> warpTile =
        cooperative_groups::tiled_partition<32>(block);
    uint64_t *warpAggregates = sharedStorage;
    uint64_t *warpPrefixes = sharedStorage + CarryPrefixMaxWarps;
    uint64_t *exclusiveStorage = sharedStorage + 2u * CarryPrefixMaxWarps;

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
        uint64_t inclusive = hasValue ? transforms[index] : CarryPrefixIdentity();

#pragma unroll
        for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
            const uint64_t previous =
                ShuffleUpCarryPrefix(warpTile, inclusive, static_cast<int>(offset));
            if (lane >= offset)
                inclusive = ComposeCarryPrefixes(previous, inclusive);
        }

        const uint32_t warpEnd = (warp + 1u) * 32u;
        const uint32_t warpLastThread = (warpEnd < blockSize ? warpEnd : blockSize) - 1u;
        if (threadIndex == warpLastThread)
            warpAggregates[warp] = inclusive;
        __syncthreads();

        uint64_t aggregate = CarryPrefixIdentity();
        if (warp == 0u) {
            uint64_t warpInclusive = lane < numWarps ? warpAggregates[lane] : CarryPrefixIdentity();
            for (uint32_t offset = 1u; offset < 32u; offset <<= 1u) {
                const uint64_t previous =
                    ShuffleUpCarryPrefix(warpTile, warpInclusive, static_cast<int>(offset));
                if (lane >= offset && lane < numWarps)
                    warpInclusive = ComposeCarryPrefixes(previous, warpInclusive);
            }

            const uint64_t previous = ShuffleUpCarryPrefix(warpTile, warpInclusive, 1);
            if (lane < numWarps) {
                warpPrefixes[lane] = lane == 0u ? CarryPrefixIdentity() : previous;
            }
            aggregate = ShuffleCarryPrefix(warpTile, warpInclusive, static_cast<int>(numWarps - 1u));
        }

        if (threadIndex == 0u)
            PublishCarryPrefixDescriptorAggregate(descriptors[part], aggregate);

        if (warp == 0u) {
            warpTile.sync();
            const uint64_t exclusive = ResolveCarryPrefixWarp(descriptors, part, warpTile);
            if (lane == 0u) {
                const uint64_t prefix = ComposeCarryPrefixes(exclusive, aggregate);
                PublishCarryPrefixDescriptorPrefix(descriptors[part], prefix);
                exclusiveStorage[0] = exclusive;
            }
        }
        __syncthreads();

        const uint64_t exclusivePart = exclusiveStorage[0];
        const uint64_t warpExclusive = warpPrefixes[warp];
        const uint64_t previous = ShuffleUpCarryPrefix(warpTile, inclusive, 1);
        const uint64_t localExclusive = lane == 0u ? CarryPrefixIdentity() : previous;
        if (hasValue) {
            const uint64_t prefixWithinPart = ComposeCarryPrefixes(warpExclusive, localExclusive);
            transforms[index] = ComposeCarryPrefixes(exclusivePart, prefixWithinPart);
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
                     const int64_t *limbs,
                     uint32_t limbCount,
                     int32_t commonExponent,
                     HpSharkFloat<SharkFloatParams> *out,
                     DebugStatePurpose digitsPurpose,
                     DebugStatePurpose magnitudePurpose)
{
    constexpr uint32_t Capacity = HpSharkReference2Workspace<SharkFloatParams>::MaxFusedLimbs;
    constexpr uint32_t DigitLengthControl = 0;
    constexpr uint32_t NegativeControl = 1;
    constexpr uint32_t NonZeroReductionControl = 2;
    uint32_t *digits = workspace.MagnitudeDigits;
    uint32_t *magnitude = workspace.Magnitude;
    uint64_t *transforms = workspace.CarryPrefixTransforms;
    uint32_t *control = workspace.CarryPrefixControl;
    MattsCudaAssert(limbCount > 0u && limbCount <= Capacity);

    PrepareSignedCarryPrefixes(
        grid, block, limbs, limbCount, transforms, workspace.CarryPrefixDescriptors);
    PrefixCarryTransformsDLB(
        grid, block, transforms, limbCount, workspace.CarryPrefixDescriptors, carryPrefixShared);

    const uint32_t gridSize = static_cast<uint32_t>(grid.size());
    for (uint32_t index = GridThreadRank(block); index < limbCount; index += gridSize) {
        const int32_t carryIn = ApplyCarryPrefix(transforms[index], 0);
        digits[index] = static_cast<uint32_t>(static_cast<uint64_t>(limbs[index] + carryIn));
    }
    grid.sync();

    if (IsLeader<SharkFloatParams>(block)) {
        const int32_t finalCarryIn = ApplyCarryPrefix(transforms[limbCount - 1u], 0);
        int32_t finalCarry = CarryOutForSignedLimb(limbs[limbCount - 1u], finalCarryIn);
        uint32_t digitLength = limbCount;
        while (finalCarry != 0 && finalCarry != -1 && digitLength < Capacity) {
            digits[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(finalCarry));
            finalCarry = CarryOutForSignedLimb(finalCarry, 0);
        }
        control[DigitLengthControl] = digitLength;
        control[NegativeControl] = finalCarry < 0 ? 1u : 0u;
    }
    grid.sync();

    uint32_t digitLength = control[DigitLengthControl];
    const bool negative = control[NegativeControl] != 0u;
    StoreReference2DebugState(debugStates, grid, block, digitsPurpose, digits, digitLength);
    if (negative) {
        // In (~digits) + 1, the carry reaches the lowest nonzero digit and stops there.
        // Locating that digit avoids a second cross-block carry-prefix scan.
        const uint32_t lowestNonZero = FindLowestNonZero<SharkFloatParams>(
            grid, block, digits, digitLength, &control[NonZeroReductionControl], carryPrefixShared);
        for (uint32_t index = GridThreadRank(block); index < digitLength; index += gridSize) {
            if (index < lowestNonZero)
                magnitude[index] = 0u;
            else if (index == lowestNonZero)
                magnitude[index] = 0u - digits[index];
            else
                magnitude[index] = ~digits[index];
        }

        // An appended carry cannot overlap the completed [0, digitLength) fill. The following barrier
        // publishes both the magnitude and control writes.
        if (IsLeader<SharkFloatParams>(block)) {
            if (lowestNonZero == digitLength) {
                MattsCudaAssert(digitLength < Capacity);
                if (digitLength < Capacity)
                    magnitude[digitLength++] = 1u;
            }
            control[DigitLengthControl] = digitLength;
        }
        grid.sync();
        digitLength = control[DigitLengthControl];
    } else {
        for (uint32_t index = GridThreadRank(block); index < digitLength; index += gridSize)
            magnitude[index] = digits[index];
        grid.sync();
    }

    const uint32_t highestNonZeroPlusOne = FindHighestNonZeroPlusOne<SharkFloatParams>(
        grid, block, magnitude, digitLength, &control[NonZeroReductionControl], carryPrefixShared);
    StoreReference2DebugState(
        debugStates, grid, block, magnitudePurpose, magnitude, highestNonZeroPlusOne);

    if (highestNonZeroPlusOne == 0u) {
        SetZero(grid, block, out);
    } else {
        const uint32_t highestNonZero = highestNonZeroPlusOne - 1u;
        constexpr int ActualDigits = SharkFloatParams::GlobalNumUint32;
        const int magnitudeLength = static_cast<int>(highestNonZero) + 1;
        const int currentBit =
            static_cast<int>(highestNonZero) * 32 + 31 - CountLeadingZeros(magnitude[highestNonZero]);
        const int desiredBit = (ActualDigits - 1) * 32 + 31;
        const int shift = currentBit - desiredBit;
        if (IsLeader<SharkFloatParams>(block)) {
            out->Exponent = commonExponent + shift;
            out->SetNegative(negative);
        }
        for (int i = static_cast<int>(GridThreadRank(block)); i < ActualDigits;
             i += static_cast<int>(gridSize)) {
            out->Digits[i] = shift > 0 ? FunnelShiftRight(magnitude, i, magnitudeLength, shift)
                                       : FunnelShiftLeft(magnitude, i, magnitudeLength, -shift);
        }
    }
    grid.sync();
    if constexpr (HpShark::Debug) {
        if (IsLeader<SharkFloatParams>(block) && highestNonZeroPlusOne != 0u)
            MattsCudaAssert((out->Digits[SharkFloatParams::GlobalNumUint32 - 1] & 0x8000'0000u) != 0u);
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

    const FusedTerm<SharkFloatParams> realZSquareTerm =
        MakeProductTerm(zReal, SpectrumId::ZReal, zReal, SpectrumId::ZReal, false, 0);
    const FusedTerm<SharkFloatParams> realNegativeZImagSquareTerm =
        MakeProductTerm(zImag, SpectrumId::ZImag, zImag, SpectrumId::ZImag, true, 0);
    const FusedTerm<SharkFloatParams> realConstantTerm = MakeLinearTerm(cReal, SpectrumId::CReal, false);
    const FusedTerm<SharkFloatParams> imagDoubleProductTerm =
        MakeProductTerm(zReal, SpectrumId::ZReal, zImag, SpectrumId::ZImag, false, 1);
    const FusedTerm<SharkFloatParams> imagConstantTerm = MakeLinearTerm(cImag, SpectrumId::CImag, false);
    int32_t realExponent;
    int32_t imagExponent;
    const bool realZero = ResolveCommonExponent(
        &realExponent, realZSquareTerm, realNegativeZImagSquareTerm, realConstantTerm);
    const bool imagZero = ResolveCommonExponent(&imagExponent, imagDoubleProductTerm, imagConstantTerm);
    uint64_t requiredBits = RequiredBitsForTerms(
        realExponent, realZSquareTerm, realNegativeZImagSquareTerm, realConstantTerm);
    const uint64_t imagBits =
        RequiredBitsForTerms(imagExponent, imagDoubleProductTerm, imagConstantTerm);
    requiredBits = requiredBits > imagBits ? requiredBits : imagBits;

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
        dzdcRealZRealTerm = MakeProductTerm(
            zReal, SpectrumId::ZReal, combo->Multiply.DzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcRealNegativeZImagTerm = MakeProductTerm(
            zImag, SpectrumId::ZImag, combo->Multiply.DzdcImag, SpectrumId::DzdcImag, true, 1);
        dzdcRealOneTerm = MakeLinearTerm(combo->Add.One, SpectrumId::One, false);
        dzdcImagZImagTerm = MakeProductTerm(
            zImag, SpectrumId::ZImag, combo->Multiply.DzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcImagZRealTerm = MakeProductTerm(
            zReal, SpectrumId::ZReal, combo->Multiply.DzdcImag, SpectrumId::DzdcImag, false, 1);
        dzdcRealZero = ResolveCommonExponent(
            &dzdcRealExponent, dzdcRealZRealTerm, dzdcRealNegativeZImagTerm, dzdcRealOneTerm);
        dzdcImagZero = ResolveCommonExponent(&dzdcImagExponent, dzdcImagZImagTerm, dzdcImagZRealTerm);
        const uint64_t dzdcRealBits = RequiredBitsForTerms(
            dzdcRealExponent, dzdcRealZRealTerm, dzdcRealNegativeZImagTerm, dzdcRealOneTerm);
        const uint64_t dzdcImagBits =
            RequiredBitsForTerms(dzdcImagExponent, dzdcImagZImagTerm, dzdcImagZRealTerm);
        requiredBits = requiredBits > dzdcRealBits ? requiredBits : dzdcRealBits;
        requiredBits = requiredBits > dzdcImagBits ? requiredBits : dzdcImagBits;
    }

    if (requiredBits == 0) {
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

    constexpr SharkNTT::PlanPrime basePlan = SharkFloatParams::NTTPlan2;
    const uint64_t requiredCoefficients = (requiredBits + basePlan.b - 1ull) / basePlan.b;
    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN || requiredN < 2) {
        if (IsLeader<SharkFloatParams>(block))
            combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }
    const uint32_t cachedN = workspace.CachedN;
    MattsCudaAssert(cachedN == 0 ||
                    (cachedN <= HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN &&
                     (cachedN & (cachedN - 1u)) == 0));
    const uint32_t activeN =
        requiredN > static_cast<uint64_t>(cachedN) ? static_cast<uint32_t>(requiredN) : cachedN;
    const SharkNTT::PlanPrime plan{basePlan.n32,
                                   basePlan.b,
                                   basePlan.L,
                                   static_cast<int>(activeN),
                                   static_cast<int>(CountTrailingZeros(activeN)),
                                   basePlan.ok};
    const uint32_t limbCount = (activeN * static_cast<uint32_t>(plan.b) + 31u) / 32u + 2u;
    GenerateActiveRoots<SharkFloatParams>(grid, block, debugCombo, activeN, workspace);

    const HpSharkFloat<SharkFloatParams> *normalForwardValues[4] = {&zReal, &zImag, &cReal, &cImag};
    uint64_t *normalForwardOutputs[4] = {
        workspace.ZReal, workspace.ZImag, workspace.CReal, workspace.CImag};
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
                                               workspace.Roots,
                                               normalForwardOutputs,
                                               normalPackedPurposes,
                                               normalForwardPurposes);

    if (!realZero && !imagZero) {
        AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                   block,
                                                   debugCombo,
                                                   debugStates,
                                                   plan,
                                                   workspace.Roots,
                                                   workspace,
                                                   realExponent,
                                                   workspace.RealOutput,
                                                   DebugStatePurpose::Z2_Perm1,
                                                   realZSquareTerm,
                                                   realNegativeZImagSquareTerm,
                                                   realConstantTerm);
        AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                   block,
                                                   debugCombo,
                                                   debugStates,
                                                   plan,
                                                   workspace.Roots,
                                                   workspace,
                                                   imagExponent,
                                                   workspace.ImagOutput,
                                                   DebugStatePurpose::Z2_Perm2,
                                                   imagDoubleProductTerm,
                                                   imagConstantTerm);
        uint64_t *normalSpectra[2] = {workspace.RealOutput, workspace.ImagOutput};
        const uint32_t normalCoefficientCounts[2] = {activeN, activeN};
        int64_t *normalLimbs[2] = {workspace.RealLimbs, workspace.ImagLimbs};
        const DebugStatePurpose normalResiduesPurposes[2] = {DebugStatePurpose::Z2_Perm4,
                                                             DebugStatePurpose::Z2_Perm5};
        const DebugStatePurpose normalLimbsPurposes[2] = {DebugStatePurpose::UnpackXX,
                                                          DebugStatePurpose::UnpackYY};
        InverseSpectraToSignedLimbsBatch<SharkFloatParams, 2>(grid,
                                                              block,
                                                              sharedData,
                                                              debugCombo,
                                                              debugStates,
                                                              plan,
                                                              workspace.Roots,
                                                              normalSpectra,
                                                              normalCoefficientCounts,
                                                              normalLimbs,
                                                              limbCount,
                                                              normalResiduesPurposes,
                                                              normalLimbsPurposes);
        FinalizeSignedStream<SharkFloatParams>(grid,
                                               block,
                                               debugStates,
                                               carryPrefixShared,
                                               workspace,
                                               workspace.RealLimbs,
                                               limbCount,
                                               realExponent,
                                               &combo->Multiply.A,
                                               DebugStatePurpose::SignedCarry1,
                                               DebugStatePurpose::FinalAdd1);
        FinalizeSignedStream<SharkFloatParams>(grid,
                                               block,
                                               debugStates,
                                               carryPrefixShared,
                                               workspace,
                                               workspace.ImagLimbs,
                                               limbCount,
                                               imagExponent,
                                               &combo->Multiply.B,
                                               DebugStatePurpose::SignedCarry2,
                                               DebugStatePurpose::FinalAdd2);
    } else {
        if (realZero) {
            SetZero(grid, block, &combo->Multiply.A);
        } else {
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       debugStates,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       realExponent,
                                                       workspace.RealOutput,
                                                       DebugStatePurpose::Z2_Perm1,
                                                       realZSquareTerm,
                                                       realNegativeZImagSquareTerm,
                                                       realConstantTerm);
            uint64_t *realSpectrum[1] = {workspace.RealOutput};
            const uint32_t realCoefficientCount[1] = {activeN};
            int64_t *realLimbs[1] = {workspace.RealLimbs};
            const DebugStatePurpose realResiduesPurpose[1] = {DebugStatePurpose::Z2_Perm4};
            const DebugStatePurpose realLimbsPurpose[1] = {DebugStatePurpose::UnpackXX};
            InverseSpectraToSignedLimbsBatch<SharkFloatParams, 1>(grid,
                                                                  block,
                                                                  sharedData,
                                                                  debugCombo,
                                                                  debugStates,
                                                                  plan,
                                                                  workspace.Roots,
                                                                  realSpectrum,
                                                                  realCoefficientCount,
                                                                  realLimbs,
                                                                  limbCount,
                                                                  realResiduesPurpose,
                                                                  realLimbsPurpose);
            FinalizeSignedStream<SharkFloatParams>(grid,
                                                   block,
                                                   debugStates,
                                                   carryPrefixShared,
                                                   workspace,
                                                   workspace.RealLimbs,
                                                   limbCount,
                                                   realExponent,
                                                   &combo->Multiply.A,
                                                   DebugStatePurpose::SignedCarry1,
                                                   DebugStatePurpose::FinalAdd1);
        }
        if (imagZero) {
            SetZero(grid, block, &combo->Multiply.B);
        } else {
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       debugStates,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       imagExponent,
                                                       workspace.ImagOutput,
                                                       DebugStatePurpose::Z2_Perm2,
                                                       imagDoubleProductTerm,
                                                       imagConstantTerm);
            uint64_t *imagSpectrum[1] = {workspace.ImagOutput};
            const uint32_t imagCoefficientCount[1] = {activeN};
            int64_t *imagLimbs[1] = {workspace.ImagLimbs};
            const DebugStatePurpose imagResiduesPurpose[1] = {DebugStatePurpose::Z2_Perm5};
            const DebugStatePurpose imagLimbsPurpose[1] = {DebugStatePurpose::UnpackYY};
            InverseSpectraToSignedLimbsBatch<SharkFloatParams, 1>(grid,
                                                                  block,
                                                                  sharedData,
                                                                  debugCombo,
                                                                  debugStates,
                                                                  plan,
                                                                  workspace.Roots,
                                                                  imagSpectrum,
                                                                  imagCoefficientCount,
                                                                  imagLimbs,
                                                                  limbCount,
                                                                  imagResiduesPurpose,
                                                                  imagLimbsPurpose);
            FinalizeSignedStream<SharkFloatParams>(grid,
                                                   block,
                                                   debugStates,
                                                   carryPrefixShared,
                                                   workspace,
                                                   workspace.ImagLimbs,
                                                   limbCount,
                                                   imagExponent,
                                                   &combo->Multiply.B,
                                                   DebugStatePurpose::SignedCarry2,
                                                   DebugStatePurpose::FinalAdd2);
        }
    }
    StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::Result_Add1, combo->Multiply.A);
    StoreReference2DebugValue(
        debugStates, grid, block, DebugStatePurpose::Result_Add2, combo->Multiply.B);

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const HpSharkFloat<SharkFloatParams> *newtonRaphsonForwardValues[3] = {
            &combo->Multiply.DzdcReal, &combo->Multiply.DzdcImag, &combo->Add.One};
        uint64_t *newtonRaphsonForwardOutputs[3] = {
            workspace.DzdcReal, workspace.DzdcImag, workspace.One};
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
                                                   workspace.Roots,
                                                   newtonRaphsonForwardOutputs,
                                                   newtonRaphsonPackedPurposes,
                                                   newtonRaphsonForwardPurposes);

        if (!dzdcRealZero && !dzdcImagZero) {
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       debugStates,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       dzdcRealExponent,
                                                       workspace.DzdcRealOutput,
                                                       DebugStatePurpose::Z2_PermW0,
                                                       dzdcRealZRealTerm,
                                                       dzdcRealNegativeZImagTerm,
                                                       dzdcRealOneTerm);
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       debugStates,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       dzdcImagExponent,
                                                       workspace.DzdcImagOutput,
                                                       DebugStatePurpose::Z2_PermW1,
                                                       dzdcImagZImagTerm,
                                                       dzdcImagZRealTerm);
            uint64_t *newtonRaphsonSpectra[2] = {workspace.DzdcRealOutput, workspace.DzdcImagOutput};
            const uint32_t newtonRaphsonCoefficientCounts[2] = {activeN, activeN};
            int64_t *newtonRaphsonLimbs[2] = {workspace.DzdcRealLimbs, workspace.DzdcImagLimbs};
            const DebugStatePurpose newtonRaphsonResiduesPurposes[2] = {DebugStatePurpose::Z2_PermW0b,
                                                                        DebugStatePurpose::Z2_PermW1b};
            const DebugStatePurpose newtonRaphsonLimbsPurposes[2] = {DebugStatePurpose::UnpackW0,
                                                                     DebugStatePurpose::UnpackW1};
            InverseSpectraToSignedLimbsBatch<SharkFloatParams, 2>(grid,
                                                                  block,
                                                                  sharedData,
                                                                  debugCombo,
                                                                  debugStates,
                                                                  plan,
                                                                  workspace.Roots,
                                                                  newtonRaphsonSpectra,
                                                                  newtonRaphsonCoefficientCounts,
                                                                  newtonRaphsonLimbs,
                                                                  limbCount,
                                                                  newtonRaphsonResiduesPurposes,
                                                                  newtonRaphsonLimbsPurposes);
            FinalizeSignedStream<SharkFloatParams>(grid,
                                                   block,
                                                   debugStates,
                                                   carryPrefixShared,
                                                   workspace,
                                                   workspace.DzdcRealLimbs,
                                                   limbCount,
                                                   dzdcRealExponent,
                                                   &combo->Multiply.DzdcReal,
                                                   DebugStatePurpose::SignedCarryDzdc1,
                                                   DebugStatePurpose::FinalAddDzdc1);
            FinalizeSignedStream<SharkFloatParams>(grid,
                                                   block,
                                                   debugStates,
                                                   carryPrefixShared,
                                                   workspace,
                                                   workspace.DzdcImagLimbs,
                                                   limbCount,
                                                   dzdcImagExponent,
                                                   &combo->Multiply.DzdcImag,
                                                   DebugStatePurpose::SignedCarryDzdc2,
                                                   DebugStatePurpose::FinalAddDzdc2);
        } else {
            if (dzdcRealZero) {
                SetZero(grid, block, &combo->Multiply.DzdcReal);
            } else {
                AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                           block,
                                                           debugCombo,
                                                           debugStates,
                                                           plan,
                                                           workspace.Roots,
                                                           workspace,
                                                           dzdcRealExponent,
                                                           workspace.DzdcRealOutput,
                                                           DebugStatePurpose::Z2_PermW0,
                                                           dzdcRealZRealTerm,
                                                           dzdcRealNegativeZImagTerm,
                                                           dzdcRealOneTerm);
                uint64_t *dzdcRealSpectrum[1] = {workspace.DzdcRealOutput};
                const uint32_t dzdcRealCoefficientCount[1] = {activeN};
                int64_t *dzdcRealLimbs[1] = {workspace.DzdcRealLimbs};
                const DebugStatePurpose dzdcRealResiduesPurpose[1] = {DebugStatePurpose::Z2_PermW0b};
                const DebugStatePurpose dzdcRealLimbsPurpose[1] = {DebugStatePurpose::UnpackW0};
                InverseSpectraToSignedLimbsBatch<SharkFloatParams, 1>(grid,
                                                                      block,
                                                                      sharedData,
                                                                      debugCombo,
                                                                      debugStates,
                                                                      plan,
                                                                      workspace.Roots,
                                                                      dzdcRealSpectrum,
                                                                      dzdcRealCoefficientCount,
                                                                      dzdcRealLimbs,
                                                                      limbCount,
                                                                      dzdcRealResiduesPurpose,
                                                                      dzdcRealLimbsPurpose);
                FinalizeSignedStream<SharkFloatParams>(grid,
                                                       block,
                                                       debugStates,
                                                       carryPrefixShared,
                                                       workspace,
                                                       workspace.DzdcRealLimbs,
                                                       limbCount,
                                                       dzdcRealExponent,
                                                       &combo->Multiply.DzdcReal,
                                                       DebugStatePurpose::SignedCarryDzdc1,
                                                       DebugStatePurpose::FinalAddDzdc1);
            }
            if (dzdcImagZero) {
                SetZero(grid, block, &combo->Multiply.DzdcImag);
            } else {
                AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                           block,
                                                           debugCombo,
                                                           debugStates,
                                                           plan,
                                                           workspace.Roots,
                                                           workspace,
                                                           dzdcImagExponent,
                                                           workspace.DzdcImagOutput,
                                                           DebugStatePurpose::Z2_PermW1,
                                                           dzdcImagZImagTerm,
                                                           dzdcImagZRealTerm);
                uint64_t *dzdcImagSpectrum[1] = {workspace.DzdcImagOutput};
                const uint32_t dzdcImagCoefficientCount[1] = {activeN};
                int64_t *dzdcImagLimbs[1] = {workspace.DzdcImagLimbs};
                const DebugStatePurpose dzdcImagResiduesPurpose[1] = {DebugStatePurpose::Z2_PermW1b};
                const DebugStatePurpose dzdcImagLimbsPurpose[1] = {DebugStatePurpose::UnpackW1};
                InverseSpectraToSignedLimbsBatch<SharkFloatParams, 1>(grid,
                                                                      block,
                                                                      sharedData,
                                                                      debugCombo,
                                                                      debugStates,
                                                                      plan,
                                                                      workspace.Roots,
                                                                      dzdcImagSpectrum,
                                                                      dzdcImagCoefficientCount,
                                                                      dzdcImagLimbs,
                                                                      limbCount,
                                                                      dzdcImagResiduesPurpose,
                                                                      dzdcImagLimbsPurpose);
                FinalizeSignedStream<SharkFloatParams>(grid,
                                                       block,
                                                       debugStates,
                                                       carryPrefixShared,
                                                       workspace,
                                                       workspace.DzdcImagLimbs,
                                                       limbCount,
                                                       dzdcImagExponent,
                                                       &combo->Multiply.DzdcImag,
                                                       DebugStatePurpose::SignedCarryDzdc2,
                                                       DebugStatePurpose::FinalAddDzdc2);
            }
        }
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_AddDzdc1, combo->Multiply.DzdcReal);
        StoreReference2DebugValue(
            debugStates, grid, block, DebugStatePurpose::Result_AddDzdc2, combo->Multiply.DzdcImag);
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
    HpSharkReference2GpuLoop(HpSharkReferenceResults<SharkFloatParams> *combo, uint64_t *tempData)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ __align__(16) uint64_t sharedData[];
    __shared__ uint64_t carryPrefixShared[2u * Reference2Detail::CarryPrefixMaxWarps + 1u];
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
