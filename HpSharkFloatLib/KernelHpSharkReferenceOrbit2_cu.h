#include "Exceptions.h"
#include "KernelHpSharkReferenceOrbit2.h"
#include "LaunchParamsCalculator.h"
#include "MultiplyNTT.cu"

#include <sstream>

namespace Reference2Detail {

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
ReverseBits32(uint32_t value, int bitCount)
{
    value = (value >> 16) | (value << 16);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);
    return value >> (32 - bitCount);
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
MontgomeryMulSerial(cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t a,
                    uint64_t b)
{
    return SharkNTT::MontgomeryMul<SharkFloatParams>(grid, block, debugCombo, a, b);
}

template <class SharkFloatParams>
__device__ uint64_t
ToMontgomerySerial(cooperative_groups::grid_group &grid,
                   cooperative_groups::thread_block &block,
                   DebugGlobalCount<SharkFloatParams> *debugCombo,
                   uint64_t value)
{
    return SharkNTT::ToMontgomery<SharkFloatParams>(grid, block, debugCombo, value);
}

template <class SharkFloatParams>
__device__ uint64_t
FromMontgomerySerial(cooperative_groups::grid_group &grid,
                     cooperative_groups::thread_block &block,
                     DebugGlobalCount<SharkFloatParams> *debugCombo,
                     uint64_t value)
{
    return SharkNTT::FromMontgomery<SharkFloatParams>(grid, block, debugCombo, value);
}

template <class SharkFloatParams>
__device__ uint64_t
MontgomeryPowSerial(cooperative_groups::grid_group &grid,
                    cooperative_groups::thread_block &block,
                    DebugGlobalCount<SharkFloatParams> *debugCombo,
                    uint64_t aMont,
                    uint64_t exponent)
{
    uint64_t result = ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1);
    while (exponent != 0) {
        if ((exponent & 1ull) != 0)
            result = MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, result, aMont);
        aMont = MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, aMont, aMont);
        exponent >>= 1;
    }
    return result;
}

template <class SharkFloatParams>
__device__ bool
IsZero(const HpSharkFloat<SharkFloatParams> &value)
{
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        if (value.Digits[i] != 0)
            return false;
    }
    return true;
}

template <class SharkFloatParams>
__device__ void
SetZero(HpSharkFloat<SharkFloatParams> *out)
{
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; ++i)
        out->Digits[i] = 0;
    out->Exponent = -100'000'000;
    out->SetNegative(false);
}

template <class SharkFloatParams>
__device__ FusedTerm<SharkFloatParams>
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
__device__ FusedTerm<SharkFloatParams>
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a, SpectrumId aId, bool negate)
{
    if (IsZero(a))
        return {true, false, 0, TermKind::Linear, aId, aId};
    return {false, static_cast<bool>(a.GetNegative() ^ negate), a.Exponent, TermKind::Linear, aId, aId};
}

template <class SharkFloatParams>
__device__ bool
ResolveCommonExponent(const FusedTerm<SharkFloatParams> *terms, uint32_t count, int32_t *commonExponent)
{
    bool any = false;
    int32_t common = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (terms[i].IsZero)
            continue;
        common = any && common < terms[i].Exponent ? common : terms[i].Exponent;
        any = true;
    }
    *commonExponent = any ? common : 0;
    return !any;
}

template <class SharkFloatParams>
__device__ uint64_t
RequiredBitsForTerms(int32_t commonExponent, const FusedTerm<SharkFloatParams> *terms, uint32_t count)
{
    constexpr uint64_t MantissaBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    uint64_t required = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (terms[i].IsZero)
            continue;
        const uint64_t shift = static_cast<uint64_t>(terms[i].Exponent - commonExponent);
        const uint64_t width = terms[i].Kind == TermKind::Product ? 2ull * MantissaBits : MantissaBits;
        required = required > shift + width ? required : shift + width;
    }
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

static __device__ void
BitReverseInplace64(uint64_t *values, uint32_t n, uint32_t stages)
{
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = ReverseBits32(i, static_cast<int>(stages)) & (n - 1u);
        if (j > i) {
            const uint64_t temp = values[i];
            values[i] = values[j];
            values[j] = temp;
        }
    }
}

template <class SharkFloatParams, bool Inverse>
__device__ void
NTTRadix2(cooperative_groups::grid_group &grid,
          cooperative_groups::thread_block &block,
          DebugGlobalCount<SharkFloatParams> *debugCombo,
          uint64_t *values,
          uint32_t n,
          uint32_t stages,
          SharkNTT::RootTables &roots)
{
    uint64_t *twiddles = Inverse ? roots.stage_twiddles_inv : roots.stage_twiddles_fwd;
    for (uint32_t stage = 1; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1;
        const uint32_t base = half - 1u;
        for (uint32_t k = 0; k < n; k += width) {
            for (uint32_t j = 0; j < half; ++j) {
                const uint32_t i0 = k + j;
                const uint32_t i1 = i0 + half;
                const uint64_t u = values[i0];
                const uint64_t t = MontgomeryMulSerial<SharkFloatParams>(
                    grid, block, debugCombo, values[i1], twiddles[base + j]);
                values[i0] = AddPSerial(u, t);
                values[i1] = SubPSerial(u, t);
            }
        }
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
    roots.N = static_cast<int32_t>(activeN);
    roots.stages = static_cast<int32_t>(stages);
    roots.total_twiddles = activeN - 1;

    constexpr uint64_t Generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t generatorMont =
        ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, Generator);
    const uint64_t psi = MontgomeryPowSerial<SharkFloatParams>(
        grid, block, debugCombo, generatorMont, SharkNTT::PHI / (2ull * activeN));
    const uint64_t psiInverse =
        MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, psi, SharkNTT::PHI - 1ull);
    const uint64_t omega = MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, psi, psi);
    const uint64_t omegaInverse =
        MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, SharkNTT::PHI - 1ull);
    const uint64_t one = ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1);

    roots.psi_pows[0] = one;
    roots.psi_inv_pows[0] = one;
    for (uint32_t i = 1; i < activeN; ++i) {
        roots.psi_pows[i] =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, roots.psi_pows[i - 1], psi);
        roots.psi_inv_pows[i] = MontgomeryMulSerial<SharkFloatParams>(
            grid, block, debugCombo, roots.psi_inv_pows[i - 1], psiInverse);
    }

    uint32_t offset = 0;
    for (uint32_t stage = 1; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1;
        roots.stage_omegas[stage - 1] =
            MontgomeryPowSerial<SharkFloatParams>(grid, block, debugCombo, omega, activeN / width);
        roots.stage_omegas_inv[stage - 1] = MontgomeryPowSerial<SharkFloatParams>(
            grid, block, debugCombo, omegaInverse, activeN / width);
        uint64_t forward = one;
        uint64_t inverse = one;
        for (uint32_t j = 0; j < half; ++j) {
            roots.stage_twiddles_fwd[offset + j] = forward;
            roots.stage_twiddles_inv[offset + j] = inverse;
            forward = MontgomeryMulSerial<SharkFloatParams>(
                grid, block, debugCombo, forward, roots.stage_omegas[stage - 1]);
            inverse = MontgomeryMulSerial<SharkFloatParams>(
                grid, block, debugCombo, inverse, roots.stage_omegas_inv[stage - 1]);
        }
        offset += half;
    }

    roots.Ninvm_mont = one;
    const uint64_t inverseTwo = ToMontgomerySerial<SharkFloatParams>(
        grid, block, debugCombo, (SharkNTT::MagicPrime + 1ull) >> 1);
    for (uint32_t stage = 0; stage < stages; ++stage)
        roots.Ninvm_mont =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, roots.Ninvm_mont, inverseTwo);
    workspace.CachedN = activeN;
}

template <class SharkFloatParams>
__device__ void
PackTwistForward(cooperative_groups::grid_group &grid,
                 cooperative_groups::thread_block &block,
                 DebugGlobalCount<SharkFloatParams> *debugCombo,
                 const HpSharkFloat<SharkFloatParams> &value,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint64_t *out)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    for (uint32_t i = 0; i < activeN; ++i) {
        const uint64_t coefficient =
            i < static_cast<uint32_t>(plan.L)
                ? ReadBitsSimple(value, static_cast<int64_t>(i) * plan.b, plan.b)
                : 0;
        const uint64_t mont = ToMontgomerySerial<SharkFloatParams>(
            grid, block, debugCombo, coefficient % SharkNTT::MagicPrime);
        out[i] = MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, mont, roots.psi_pows[i]);
    }
    BitReverseInplace64(out, activeN, static_cast<uint32_t>(plan.stages));
    NTTRadix2<SharkFloatParams, false>(grid, block, debugCombo, out, activeN, plan.stages, roots);
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
    return SubPSerial(ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 0),
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
        ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1ull << bitShift);
    const uint64_t zeroMont = ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 0);
    for (uint32_t i = 0; i < static_cast<uint32_t>(plan.N); ++i) {
        const uint64_t chunkScale =
            chunkShift == 0 ? ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1)
                            : PsiPowerMont<SharkFloatParams>(
                                  grid, block, debugCombo, plan, roots, chunkShift * (1ull + 2ull * i));
        const uint64_t scale =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, chunkScale, bitScale);
        const uint64_t shifted =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, source[i], scale);
        dest[i] = negative ? SubPSerial(zeroMont, shifted) : shifted;
    }
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
        ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1ull << bitShift);
    for (uint32_t i = 0; i < static_cast<uint32_t>(plan.N); ++i) {
        const uint64_t chunkScale =
            chunkShift == 0 ? ToMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, 1)
                            : PsiPowerMont<SharkFloatParams>(
                                  grid, block, debugCombo, plan, roots, chunkShift * (1ull + 2ull * i));
        const uint64_t scale =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, chunkScale, bitScale);
        const uint64_t shifted =
            MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, source[i], scale);
        dest[i] = negative ? SubPSerial(dest[i], shifted) : AddPSerial(dest[i], shifted);
    }
}

template <class SharkFloatParams>
__device__ uint64_t *
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
__device__ void
AccumulateOutputSpectrum(cooperative_groups::grid_group &grid,
                         cooperative_groups::thread_block &block,
                         DebugGlobalCount<SharkFloatParams> *debugCombo,
                         const SharkNTT::PlanPrime &plan,
                         const SharkNTT::RootTables &roots,
                         HpSharkReference2Workspace<SharkFloatParams> &workspace,
                         int32_t commonExponent,
                         uint64_t *dest,
                         const FusedTerm<SharkFloatParams> *terms,
                         uint32_t count)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    bool hasDestinationValue = false;
    for (uint32_t termIndex = 0; termIndex < count; ++termIndex) {
        const FusedTerm<SharkFloatParams> &term = terms[termIndex];
        if (term.IsZero)
            continue;
        const uint64_t shift = static_cast<uint64_t>(term.Exponent - commonExponent);
        if (term.Kind == TermKind::Product) {
            const uint64_t *a = GetSpectrum(workspace, term.A);
            const uint64_t *b = GetSpectrum(workspace, term.B);
            for (uint32_t i = 0; i < activeN; ++i)
                workspace.Product[i] =
                    MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, a[i], b[i]);
            if (hasDestinationValue) {
                AddShiftedSpectrum<SharkFloatParams>(grid,
                                                     block,
                                                     debugCombo,
                                                     plan,
                                                     roots,
                                                     workspace.Product,
                                                     shift,
                                                     term.IsNegative,
                                                     dest);
            } else {
                WriteShiftedSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       plan,
                                                       roots,
                                                       workspace.Product,
                                                       shift,
                                                       term.IsNegative,
                                                       dest);
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
    assert(hasDestinationValue);
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

template <class SharkFloatParams>
__device__ void
UnpackResiduesToSignedLimbs(const uint64_t *residues,
                            const SharkNTT::PlanPrime &plan,
                            uint32_t coefficientCount,
                            int64_t *limbs,
                            uint32_t limbCount)
{
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    for (uint32_t j = 0; j < limbCount; ++j) {
        const uint64_t firstBit = j >= 3 ? static_cast<uint64_t>(j - 3) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(j) + 1ull) * 32ull - 1ull;
        const uint64_t firstCoefficient = firstBit / static_cast<uint64_t>(plan.b);
        const uint64_t lastCoefficient = lastBit / static_cast<uint64_t>(plan.b);
        int64_t total = 0;
        for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
            const uint64_t residue = residues[i];
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
            total += negative ? -static_cast<int64_t>(contribution) : static_cast<int64_t>(contribution);
        }
        limbs[j] = total;
    }
}

template <class SharkFloatParams>
__device__ void
InverseSpectrumToSignedLimbs(cooperative_groups::grid_group &grid,
                             cooperative_groups::thread_block &block,
                             DebugGlobalCount<SharkFloatParams> *debugCombo,
                             const SharkNTT::PlanPrime &plan,
                             SharkNTT::RootTables &roots,
                             uint64_t *spectrum,
                             uint32_t coefficientCount,
                             int64_t *limbs,
                             uint32_t limbCount)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    BitReverseInplace64(spectrum, activeN, static_cast<uint32_t>(plan.stages));
    NTTRadix2<SharkFloatParams, true>(grid, block, debugCombo, spectrum, activeN, plan.stages, roots);
    for (uint32_t i = 0; i < activeN; ++i) {
        uint64_t value = MontgomeryMulSerial<SharkFloatParams>(
            grid, block, debugCombo, spectrum[i], roots.psi_inv_pows[i]);
        value = MontgomeryMulSerial<SharkFloatParams>(grid, block, debugCombo, value, roots.Ninvm_mont);
        spectrum[i] = FromMontgomerySerial<SharkFloatParams>(grid, block, debugCombo, value);
    }
    UnpackResiduesToSignedLimbs<SharkFloatParams>(spectrum, plan, coefficientCount, limbs, limbCount);
}

static __device__ int32_t
CountLeadingZeros(uint32_t value)
{
    return __clz(value);
}

template <class SharkFloatParams>
__device__ void
FinalizeSignedStream(const int64_t *limbs,
                     uint32_t limbCount,
                     int32_t commonExponent,
                     uint32_t *digits,
                     uint32_t *magnitude,
                     HpSharkFloat<SharkFloatParams> *out)
{
    constexpr int64_t Base = 1ll << 32;
    constexpr uint32_t Capacity = HpSharkReference2Workspace<SharkFloatParams>::MaxFusedLimbs;
    uint32_t digitLength = 0;
    int64_t carry = 0;
    for (uint32_t i = 0; i < limbCount; ++i) {
        const int64_t sum = limbs[i] + carry;
        digits[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(sum));
        carry = (sum - static_cast<int64_t>(digits[digitLength - 1])) / Base;
    }
    while (carry != 0 && carry != -1 && digitLength < Capacity) {
        digits[digitLength++] = static_cast<uint32_t>(static_cast<uint64_t>(carry));
        carry = (carry - static_cast<int64_t>(digits[digitLength - 1])) / Base;
    }

    bool negative = carry < 0;
    uint32_t magnitudeLength = 0;
    if (!negative) {
        while (digitLength > 0 && digits[digitLength - 1] == 0)
            --digitLength;
        for (uint32_t i = 0; i < digitLength; ++i)
            magnitude[magnitudeLength++] = digits[i];
    } else {
        uint64_t addOne = 1;
        for (uint32_t i = 0; i < digitLength; ++i) {
            const uint64_t sum = static_cast<uint32_t>(~digits[i]) + addOne;
            magnitude[magnitudeLength++] = static_cast<uint32_t>(sum);
            addOne = sum >> 32;
        }
        if (addOne != 0 && magnitudeLength < Capacity)
            magnitude[magnitudeLength++] = static_cast<uint32_t>(addOne);
        while (magnitudeLength > 0 && magnitude[magnitudeLength - 1] == 0)
            --magnitudeLength;
        if (magnitudeLength == 0)
            negative = false;
    }

    if (magnitudeLength == 0) {
        SetZero(out);
        return;
    }

    constexpr int ActualDigits = SharkFloatParams::GlobalNumUint32;
    const int mostSignificant = static_cast<int>(magnitudeLength) - 1;
    const int currentBit = mostSignificant * 32 + 31 - CountLeadingZeros(magnitude[mostSignificant]);
    const int desiredBit = (ActualDigits - 1) * 32 + 31;
    const int shift = currentBit - desiredBit;
    for (int i = 0; i < ActualDigits; ++i) {
        out->Digits[i] = shift > 0 ? FunnelShiftRight(magnitude, i, magnitudeLength, shift)
                                   : FunnelShiftLeft(magnitude, i, magnitudeLength, -shift);
    }
    out->Exponent = commonExponent + shift;
    out->SetNegative(negative);
}

template <class SharkFloatParams>
__device__ void
FusedReferenceOrbitStep(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        DebugGlobalCount<SharkFloatParams> *debugCombo,
                        HpSharkReferenceResults<SharkFloatParams> *combo)
{
    auto &workspace = *combo->Reference2Workspace;
    const auto &zReal = combo->Multiply.A;
    const auto &zImag = combo->Multiply.B;
    const auto &cReal = combo->Add.C_A;
    const auto &cImag = combo->Add.E_B;

    FusedTerm<SharkFloatParams> realTerms[3] = {
        MakeProductTerm(zReal, SpectrumId::ZReal, zReal, SpectrumId::ZReal, false, 0),
        MakeProductTerm(zImag, SpectrumId::ZImag, zImag, SpectrumId::ZImag, true, 0),
        MakeLinearTerm(cReal, SpectrumId::CReal, false)};
    FusedTerm<SharkFloatParams> imagTerms[2] = {
        MakeProductTerm(zReal, SpectrumId::ZReal, zImag, SpectrumId::ZImag, false, 1),
        MakeLinearTerm(cImag, SpectrumId::CImag, false)};
    int32_t realExponent;
    int32_t imagExponent;
    const bool realZero = ResolveCommonExponent(realTerms, 3, &realExponent);
    const bool imagZero = ResolveCommonExponent(imagTerms, 2, &imagExponent);
    uint64_t requiredBits = RequiredBitsForTerms(realExponent, realTerms, 3);
    const uint64_t imagBits = RequiredBitsForTerms(imagExponent, imagTerms, 2);
    requiredBits = requiredBits > imagBits ? requiredBits : imagBits;

    FusedTerm<SharkFloatParams> dzdcRealTerms[3]{};
    FusedTerm<SharkFloatParams> dzdcImagTerms[2]{};
    int32_t dzdcRealExponent = 0;
    int32_t dzdcImagExponent = 0;
    bool dzdcRealZero = true;
    bool dzdcImagZero = true;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealTerms[0] = MakeProductTerm(
            zReal, SpectrumId::ZReal, combo->Multiply.DzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcRealTerms[1] = MakeProductTerm(
            zImag, SpectrumId::ZImag, combo->Multiply.DzdcImag, SpectrumId::DzdcImag, true, 1);
        dzdcRealTerms[2] = MakeLinearTerm(combo->Add.One, SpectrumId::One, false);
        dzdcImagTerms[0] = MakeProductTerm(
            zImag, SpectrumId::ZImag, combo->Multiply.DzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcImagTerms[1] = MakeProductTerm(
            zReal, SpectrumId::ZReal, combo->Multiply.DzdcImag, SpectrumId::DzdcImag, false, 1);
        dzdcRealZero = ResolveCommonExponent(dzdcRealTerms, 3, &dzdcRealExponent);
        dzdcImagZero = ResolveCommonExponent(dzdcImagTerms, 2, &dzdcImagExponent);
        const uint64_t dzdcRealBits = RequiredBitsForTerms(dzdcRealExponent, dzdcRealTerms, 3);
        const uint64_t dzdcImagBits = RequiredBitsForTerms(dzdcImagExponent, dzdcImagTerms, 2);
        requiredBits = requiredBits > dzdcRealBits ? requiredBits : dzdcRealBits;
        requiredBits = requiredBits > dzdcImagBits ? requiredBits : dzdcImagBits;
    }

    if (requiredBits == 0) {
        SetZero(&combo->Multiply.A);
        SetZero(&combo->Multiply.B);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            SetZero(&combo->Multiply.DzdcReal);
            SetZero(&combo->Multiply.DzdcImag);
        }
        return;
    }

    constexpr SharkNTT::PlanPrime basePlan = SharkFloatParams::NTTPlan2;
    const uint64_t requiredCoefficients = (requiredBits + basePlan.b - 1ull) / basePlan.b;
    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > HpSharkReference2Workspace<SharkFloatParams>::MaxFusedN || requiredN < 2) {
        combo->PeriodicityStatus = PeriodicityResult::Unknown;
        return;
    }
    const uint32_t activeN = static_cast<uint32_t>(requiredN);
    const SharkNTT::PlanPrime plan{basePlan.n32,
                                   basePlan.b,
                                   basePlan.L,
                                   static_cast<int>(activeN),
                                   static_cast<int>(CountTrailingZeros(activeN)),
                                   basePlan.ok};
    const uint32_t limbCount = (activeN * static_cast<uint32_t>(plan.b) + 31u) / 32u + 2u;
    GenerateActiveRoots<SharkFloatParams>(grid, block, debugCombo, activeN, workspace);

    PackTwistForward<SharkFloatParams>(
        grid, block, debugCombo, zReal, plan, workspace.Roots, workspace.ZReal);
    PackTwistForward<SharkFloatParams>(
        grid, block, debugCombo, zImag, plan, workspace.Roots, workspace.ZImag);
    PackTwistForward<SharkFloatParams>(
        grid, block, debugCombo, cReal, plan, workspace.Roots, workspace.CReal);
    PackTwistForward<SharkFloatParams>(
        grid, block, debugCombo, cImag, plan, workspace.Roots, workspace.CImag);
    if (realZero) {
        SetZero(&combo->Multiply.A);
    } else {
        AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                   block,
                                                   debugCombo,
                                                   plan,
                                                   workspace.Roots,
                                                   workspace,
                                                   realExponent,
                                                   workspace.RealOutput,
                                                   realTerms,
                                                   3);
        InverseSpectrumToSignedLimbs<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace.RealOutput,
                                                       activeN,
                                                       workspace.RealLimbs,
                                                       limbCount);
        FinalizeSignedStream<SharkFloatParams>(workspace.RealLimbs,
                                               limbCount,
                                               realExponent,
                                               workspace.MagnitudeDigits,
                                               workspace.Magnitude,
                                               &combo->Multiply.A);
    }
    if (imagZero) {
        SetZero(&combo->Multiply.B);
    } else {
        AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                   block,
                                                   debugCombo,
                                                   plan,
                                                   workspace.Roots,
                                                   workspace,
                                                   imagExponent,
                                                   workspace.ImagOutput,
                                                   imagTerms,
                                                   2);
        InverseSpectrumToSignedLimbs<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace.ImagOutput,
                                                       activeN,
                                                       workspace.ImagLimbs,
                                                       limbCount);
        FinalizeSignedStream<SharkFloatParams>(workspace.ImagLimbs,
                                               limbCount,
                                               imagExponent,
                                               workspace.MagnitudeDigits,
                                               workspace.Magnitude,
                                               &combo->Multiply.B);
    }

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        PackTwistForward<SharkFloatParams>(grid,
                                           block,
                                           debugCombo,
                                           combo->Multiply.DzdcReal,
                                           plan,
                                           workspace.Roots,
                                           workspace.DzdcReal);
        PackTwistForward<SharkFloatParams>(grid,
                                           block,
                                           debugCombo,
                                           combo->Multiply.DzdcImag,
                                           plan,
                                           workspace.Roots,
                                           workspace.DzdcImag);
        PackTwistForward<SharkFloatParams>(
            grid, block, debugCombo, combo->Add.One, plan, workspace.Roots, workspace.One);
        if (dzdcRealZero) {
            SetZero(&combo->Multiply.DzdcReal);
        } else {
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       dzdcRealExponent,
                                                       workspace.DzdcRealOutput,
                                                       dzdcRealTerms,
                                                       3);
            InverseSpectrumToSignedLimbs<SharkFloatParams>(grid,
                                                           block,
                                                           debugCombo,
                                                           plan,
                                                           workspace.Roots,
                                                           workspace.DzdcRealOutput,
                                                           activeN,
                                                           workspace.DzdcRealLimbs,
                                                           limbCount);
            FinalizeSignedStream<SharkFloatParams>(workspace.DzdcRealLimbs,
                                                   limbCount,
                                                   dzdcRealExponent,
                                                   workspace.MagnitudeDigits,
                                                   workspace.Magnitude,
                                                   &combo->Multiply.DzdcReal);
        }
        if (dzdcImagZero) {
            SetZero(&combo->Multiply.DzdcImag);
        } else {
            AccumulateOutputSpectrum<SharkFloatParams>(grid,
                                                       block,
                                                       debugCombo,
                                                       plan,
                                                       workspace.Roots,
                                                       workspace,
                                                       dzdcImagExponent,
                                                       workspace.DzdcImagOutput,
                                                       dzdcImagTerms,
                                                       2);
            InverseSpectrumToSignedLimbs<SharkFloatParams>(grid,
                                                           block,
                                                           debugCombo,
                                                           plan,
                                                           workspace.Roots,
                                                           workspace.DzdcImagOutput,
                                                           activeN,
                                                           workspace.DzdcImagLimbs,
                                                           limbCount);
            FinalizeSignedStream<SharkFloatParams>(workspace.DzdcImagLimbs,
                                                   limbCount,
                                                   dzdcImagExponent,
                                                   workspace.MagnitudeDigits,
                                                   workspace.Magnitude,
                                                   &combo->Multiply.DzdcImag);
        }
    }
}

template <class SharkFloatParams>
__device__ void
UpdateD2(HpSharkReferenceResults<SharkFloatParams> *combo)
{
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        using Hdr = typename SharkFloatParams::Float;
        const Hdr zr = combo->Multiply.A.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
        const Hdr zi = combo->Multiply.B.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
        const Hdr dzr =
            combo->Multiply.DzdcReal.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
        const Hdr dzi =
            combo->Multiply.DzdcImag.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
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
        Hdr zx = combo->Multiply.A.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
        Hdr zy = combo->Multiply.B.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
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
        const Hdr cx = combo->Add.C_A.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
        const Hdr cy = combo->Add.E_B.template ToHDRFloat<typename SharkFloatParams::SubType>(0);
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
    const bool leader = Reference2Detail::IsLeader<SharkFloatParams>(block);
    DebugGlobalCount<SharkFloatParams> *debugCombo = nullptr;
    if constexpr (HpShark::DebugGlobalState) {
        const auto offset = HpShark::AdditionalGlobalSyncSpace;
        debugCombo = reinterpret_cast<DebugGlobalCount<SharkFloatParams> *>(&tempData[offset]);
        if (leader)
            debugCombo->DebugMultiplyErase();
    }

    grid.sync();
    if (leader) { 
        combo->OutputIterCount = 0;
        combo->PeriodicityStatus = PeriodicityResult::Continue;
    }
    grid.sync();

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

        if (leader)
            Reference2Detail::FusedReferenceOrbitStep<SharkFloatParams>(grid, block, debugCombo, combo);
        grid.sync();
        if (combo->PeriodicityStatus == PeriodicityResult::Unknown)
            break;

        if (leader)
            ++combo->OutputIterCount;
        grid.sync();
        (void)stop;
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
