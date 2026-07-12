#include "ReferenceReferenceOrbit2.h"

#include "DbgHeap.h"
#include "DebugChecksumHost.h"
#include "HDRFloat.h"
#include "HpSharkFloat.h"
#include "MultiplyNTTCudaSetup.h"
#include "NTTConstexprGenerator.h"
#include "TestVerbose.h"

#include <algorithm>
#include <assert.h>
#include <bit>
#include <cstdint>
#include <iostream>
#include <memory>

namespace {

enum class ShiftDir { Left, Right };
enum class SpectrumId { ZReal, ZImag, DzdcReal, DzdcImag, CReal, CImag, One };
enum class TermKind { Product, Linear };

template <class SharkFloatParams> struct FusedTerm {
    bool IsZero;
    bool IsNegative;
    int32_t Exponent;
    TermKind Kind;
    SpectrumId A;
    SpectrumId B;
};

template <uint32_t PlanN> struct FusedSpectraBase {
    static constexpr uint32_t Length = PlanN;

    uint64_t ZReal[PlanN];
    uint64_t ZImag[PlanN];
    uint64_t CReal[PlanN];
    uint64_t CImag[PlanN];
};

template <uint32_t PlanN, bool EnableDerivative> struct FusedSpectra : FusedSpectraBase<PlanN> {};

template <uint32_t PlanN> struct FusedSpectra<PlanN, true> : FusedSpectraBase<PlanN> {
    uint64_t DzdcReal[PlanN];
    uint64_t DzdcImag[PlanN];
    uint64_t One[PlanN];
};

template <class SharkFloatParams, uint32_t Ddigits> struct FinalizationStream {
    const int64_t *Limbs;
    int32_t CommonExp;
    HpSharkFloat<SharkFloatParams> *Out;
};

static void
PrintPlan(const SharkNTT::PlanPrime &plan)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "ReferenceOrbit2 fused PlanPrime: n32=" << plan.n32 << " b=" << plan.b
                  << " L=" << plan.L << " N=" << plan.N << " stages=" << plan.stages << " ok=" << plan.ok
                  << std::endl;
    }
}

static uint32_t
ReverseBits32(uint32_t value, int bitCount)
{
    value = (value >> 16) | (value << 16);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);
    return value >> (32 - bitCount);
}

static uint64_t
AddP(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    if (s < a || s >= SharkNTT::MagicPrime)
        s -= SharkNTT::MagicPrime;
    return s;
}

static uint64_t
SubP(uint64_t a, uint64_t b)
{
    return (a >= b) ? (a - b) : (a + SharkNTT::MagicPrime - b);
}

template <ShiftDir D, class IntT>
static uint32_t
FunnelShift32(const IntT *data, int idx, int count, int bitOffset)
{
    const int wordOff = bitOffset / 32;
    const int b = bitOffset % 32;

    auto pick = [&](int i) -> uint32_t {
        return (i < 0 || i >= count) ? 0u : static_cast<uint32_t>(data[i]);
    };

    uint32_t low;
    uint32_t high;
    if constexpr (D == ShiftDir::Right) {
        low = pick(idx + wordOff);
        high = pick(idx + wordOff + 1);
    } else {
        low = pick(idx - wordOff);
        high = pick(idx - wordOff - 1);
    }

    if (b == 0)
        return low;
    if constexpr (D == ShiftDir::Right)
        return (low >> b) | (high << (32 - b));
    else
        return (low << b) | (high >> (32 - b));
}

template <ShiftDir D, class IntT>
static void
MultiWordShift(const IntT *in, int inCount, int shiftNeeded, uint32_t *out, int outCount)
{
    for (int i = 0; i < outCount; ++i) {
        out[i] = FunnelShift32<D>(in, i, inCount, shiftNeeded);
    }
}

static int32_t
CountLeadingZeros(uint32_t x)
{
    return static_cast<int32_t>(std::countl_zero(x));
}

template <class SharkFloatParams>
static bool
IsZero(const HpSharkFloat<SharkFloatParams> &value)
{
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        if (value.Digits[i] != 0)
            return false;
    }
    return true;
}

template <class SharkFloatParams>
static void
SetZero(HpSharkFloat<SharkFloatParams> *out)
{
    std::fill_n(out->Digits, SharkFloatParams::GlobalNumUint32, uint32_t{0});
    out->Exponent = -100'000'000;
    out->SetNegative(false);
}

template <class SharkFloatParams>
static FusedTerm<SharkFloatParams>
MakeProductTerm(const HpSharkFloat<SharkFloatParams> &a,
                SpectrumId aId,
                const HpSharkFloat<SharkFloatParams> &b,
                SpectrumId bId,
                bool negate,
                int32_t exponentOffset)
{
    if (IsZero(a) || IsZero(b)) {
        return {true, false, 0, TermKind::Product, aId, bId};
    }

    const int64_t exponent = static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(b.Exponent) +
                             static_cast<int64_t>(exponentOffset);
    assert(exponent >= INT32_MIN && exponent <= INT32_MAX);

    return {false,
            static_cast<bool>(a.GetNegative() ^ b.GetNegative() ^ negate),
            static_cast<int32_t>(exponent),
            TermKind::Product,
            aId,
            bId};
}

template <class SharkFloatParams>
static FusedTerm<SharkFloatParams>
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a, SpectrumId aId, bool negate)
{
    if (IsZero(a)) {
        return {true, false, 0, TermKind::Linear, aId, aId};
    }

    return {false, static_cast<bool>(a.GetNegative() ^ negate), a.Exponent, TermKind::Linear, aId, aId};
}

template <class SharkFloatParams, class... Terms>
static bool
ResolveCommonExponent(int32_t &commonExpOut,
                      const FusedTerm<SharkFloatParams> &first,
                      const Terms &...terms)
{
    bool any = false;
    int32_t commonExp = 0;

    const auto resolveTerm = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero)
            return;
        commonExp = any ? std::min(commonExp, term.Exponent) : term.Exponent;
        any = true;
    };

    resolveTerm(first);
    (resolveTerm(terms), ...);

    commonExpOut = any ? commonExp : 0;
    return !any;
}

template <class SharkFloatParams, class... Terms>
static void
AssertFixedPlanAlignment(int32_t commonExp,
                         const FusedTerm<SharkFloatParams> &first,
                         const Terms &...terms)
{
    constexpr uint64_t MantissaBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    constexpr uint64_t MaxRelativeAlignmentBits = 16;

    bool hasProduct = false;
    const auto assertProductAlignment = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero || term.Kind != TermKind::Product)
            return;

        assert(term.Exponent >= commonExp);
        const auto shiftBits =
            static_cast<uint64_t>(static_cast<int64_t>(term.Exponent) - static_cast<int64_t>(commonExp));
        assert(shiftBits < MaxRelativeAlignmentBits);
        hasProduct = true;
    };

    assertProductAlignment(first);
    (assertProductAlignment(terms), ...);

    if (!hasProduct)
        return;

    const auto assertLinearAlignment = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero || term.Kind != TermKind::Linear)
            return;

        assert(term.Exponent >= commonExp);
        const auto shiftBits =
            static_cast<uint64_t>(static_cast<int64_t>(term.Exponent) - static_cast<int64_t>(commonExp));
        assert(shiftBits >= MantissaBits - MaxRelativeAlignmentBits);
        assert(shiftBits <= MantissaBits + MaxRelativeAlignmentBits);
    };

    assertLinearAlignment(first);
    (assertLinearAlignment(terms), ...);
}

template <class SharkFloatParams>
static uint64_t
ReadBitsSimple(const HpSharkFloat<SharkFloatParams> &x, int64_t q, int b)
{
    const int B = SharkFloatParams::GlobalNumUint32 * 32;
    if (q >= B || q < 0)
        return 0;

    uint64_t v = 0;
    int need = b;
    int outPos = 0;
    int64_t bit = q;

    while (need > 0 && bit < B) {
        const int64_t w = bit / 32;
        const int off = static_cast<int>(bit % 32);
        const uint32_t limb = (w >= 0) ? x.Digits[static_cast<int>(w)] : 0u;
        const uint32_t chunk = off ? (limb >> off) : limb;
        const int take = std::min(32 - off, need);

        v |= static_cast<uint64_t>(chunk & ((take == 32) ? 0xffffffffu : ((1u << take) - 1u))) << outPos;

        outPos += take;
        need -= take;
        bit += take;
    }
    return (b == 64) ? v : (v & ((1ull << b) - 1ull));
}

static void
BitReverseInplace64(uint64_t *a, uint32_t N, uint32_t stages)
{
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t j = ReverseBits32(i, static_cast<int>(stages)) & (N - 1u);
        if (j > i)
            std::swap(a[i], a[j]);
    }
}

template <class SharkFloatParams, bool inverse>
static void
NTTRadix2(DebugHostCombo<SharkFloatParams> &debugCombo,
          uint64_t *a,
          uint32_t N,
          uint32_t stages,
          SharkNTT::RootTables &rootTables)
{
    uint64_t *stageOmegas;
    uint64_t *stageTwiddles;

    if constexpr (inverse) {
        stageOmegas = rootTables.stage_omegas_inv;
        stageTwiddles = rootTables.stage_twiddles_inv;
    } else {
        stageOmegas = rootTables.stage_omegas;
        stageTwiddles = rootTables.stage_twiddles_fwd;
    }

    for (uint32_t s = 1; s <= stages; ++s) {
        const uint32_t m = 1u << s;
        const uint32_t half = m >> 1;
        const uint64_t wM = stageOmegas[s - 1];
        const uint32_t numTwid = 1u << (s - 1);
        const uint32_t twBase = numTwid - 1u;

        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < half; ++j) {
                const uint32_t i0 = k + j;
                const uint32_t i1 = i0 + half;
                const uint64_t u = a[i0];
                const uint64_t v = a[i1];
                const uint64_t w = stageTwiddles[twBase + j];
                const uint64_t t = SharkNTT::MontgomeryMul(debugCombo, v, w);
                a[i0] = AddP(u, t);
                a[i1] = SubP(u, t);
            }
        }

        (void)wM;
    }
}

template <class SharkFloatParams, uint32_t PlanN>
static void
PackTwistForward(DebugHostCombo<SharkFloatParams> &debugCombo,
                 const HpSharkFloat<SharkFloatParams> &x,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint64_t (&out)[PlanN])
{
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    assert(static_cast<uint32_t>(plan.N) == PlanN);
    std::fill_n(out, PlanN, zeroMont);

    for (int i = 0; i < plan.L; ++i) {
        const uint64_t coeff = ReadBitsSimple(x, static_cast<int64_t>(i) * plan.b, plan.b);
        const uint64_t coeffMont = SharkNTT::ToMontgomery(debugCombo, coeff % SharkNTT::MagicPrime);
        out[i] = SharkNTT::MontgomeryMul(debugCombo, coeffMont, roots.psi_pows[i]);
    }

    BitReverseInplace64(out, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages));
    NTTRadix2<SharkFloatParams, false>(
        debugCombo, out, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);
}

template <class SharkFloatParams>
static uint64_t
PsiPowerMont(const SharkNTT::PlanPrime &plan, const SharkNTT::RootTables &roots, uint64_t exponent)
{
    const uint64_t twoN = 2ull * static_cast<uint64_t>(plan.N);
    const uint64_t reduced = exponent % twoN;
    if (reduced < static_cast<uint64_t>(plan.N))
        return roots.psi_pows[static_cast<size_t>(reduced)];
    return SubP(SharkNTT::ToMontgomery<SharkFloatParams>(0),
                roots.psi_pows[static_cast<size_t>(reduced - static_cast<uint64_t>(plan.N))]);
}

template <class SharkFloatParams, uint32_t PlanN>
static void
AddShiftedSpectrum(DebugHostCombo<SharkFloatParams> &debugCombo,
                   const SharkNTT::PlanPrime &plan,
                   const SharkNTT::RootTables &roots,
                   const uint64_t *source,
                   uint64_t shiftBits,
                   bool negative,
                   uint64_t *dest)
{
    assert(static_cast<uint32_t>(plan.N) == PlanN);

    const uint64_t chunkShift = shiftBits / static_cast<uint64_t>(plan.b);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    const uint64_t bitScale = SharkNTT::ToMontgomery(debugCombo, 1ull << bitShift);

    for (uint32_t i = 0; i < PlanN; ++i) {
        const uint64_t psiExponent = chunkShift * (1ull + 2ull * static_cast<uint64_t>(i));
        const uint64_t chunkScale = (chunkShift == 0)
                                        ? SharkNTT::ToMontgomery<SharkFloatParams>(1)
                                        : PsiPowerMont<SharkFloatParams>(plan, roots, psiExponent);
        const uint64_t scale = SharkNTT::MontgomeryMul(debugCombo, chunkScale, bitScale);
        const uint64_t shifted = SharkNTT::MontgomeryMul(debugCombo, source[i], scale);

        if (negative)
            dest[i] = SubP(dest[i], shifted);
        else
            dest[i] = AddP(dest[i], shifted);
    }
}

template <uint32_t PlanN, bool EnableDerivative>
static uint64_t *
GetSpectrum(FusedSpectra<PlanN, EnableDerivative> &spectra, SpectrumId id)
{
    switch (id) {
        case SpectrumId::ZReal:
            return spectra.ZReal;
        case SpectrumId::ZImag:
            return spectra.ZImag;
        case SpectrumId::CReal:
            return spectra.CReal;
        case SpectrumId::CImag:
            return spectra.CImag;
        default:
            break;
    }

    if constexpr (EnableDerivative) {
        switch (id) {
            case SpectrumId::DzdcReal:
                return spectra.DzdcReal;
            case SpectrumId::DzdcImag:
                return spectra.DzdcImag;
            case SpectrumId::One:
                return spectra.One;
            default:
                break;
        }
    }

    assert(false);
    return spectra.ZReal;
}

template <uint32_t PlanN, bool EnableDerivative>
static const uint64_t *
GetSpectrum(const FusedSpectra<PlanN, EnableDerivative> &spectra, SpectrumId id)
{
    switch (id) {
        case SpectrumId::ZReal:
            return spectra.ZReal;
        case SpectrumId::ZImag:
            return spectra.ZImag;
        case SpectrumId::CReal:
            return spectra.CReal;
        case SpectrumId::CImag:
            return spectra.CImag;
        default:
            break;
    }

    if constexpr (EnableDerivative) {
        switch (id) {
            case SpectrumId::DzdcReal:
                return spectra.DzdcReal;
            case SpectrumId::DzdcImag:
                return spectra.DzdcImag;
            case SpectrumId::One:
                return spectra.One;
            default:
                break;
        }
    }

    assert(false);
    return spectra.ZReal;
}

template <class SharkFloatParams, uint32_t PlanN, bool EnableDerivative, class... Terms>
static void
AccumulateOutputSpectrum(DebugHostCombo<SharkFloatParams> &debugCombo,
                         const SharkNTT::PlanPrime &plan,
                         const SharkNTT::RootTables &roots,
                         FusedSpectra<PlanN, EnableDerivative> &spectra,
                         bool isZero,
                         int32_t commonExp,
                         uint64_t *dest,
                         const FusedTerm<SharkFloatParams> &first,
                         const Terms &...terms)
{
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    assert(static_cast<uint32_t>(plan.N) == PlanN);
    std::fill_n(dest, PlanN, zeroMont);

    if (isZero)
        return;

    uint64_t product[PlanN];
    const auto accumulateTerm = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero)
            return;

        assert(term.Exponent >= commonExp);
        const auto shiftBits =
            static_cast<uint64_t>(static_cast<int64_t>(term.Exponent) - static_cast<int64_t>(commonExp));

        if (term.Kind == TermKind::Product) {
            const uint64_t *a = GetSpectrum(spectra, term.A);
            const uint64_t *b = GetSpectrum(spectra, term.B);
            for (uint32_t i = 0; i < PlanN; ++i) {
                product[i] = SharkNTT::MontgomeryMul(debugCombo, a[i], b[i]);
            }
            AddShiftedSpectrum<SharkFloatParams, PlanN>(
                debugCombo, plan, roots, product, shiftBits, term.IsNegative, dest);
        } else {
            AddShiftedSpectrum<SharkFloatParams, PlanN>(
                debugCombo, plan, roots, GetSpectrum(spectra, term.A), shiftBits, term.IsNegative, dest);
        }
    };

    accumulateTerm(first);
    (accumulateTerm(terms), ...);
}

template <class SharkFloatParams, uint32_t PlanN, uint32_t CoefficientCount, uint32_t Ddigits>
static void
UnpackResiduesToSignedLimbs(const uint64_t *normalResidues,
                            const SharkNTT::PlanPrime &plan,
                            int64_t *limbs)
{
    assert(static_cast<uint32_t>(plan.N) == PlanN);
    std::fill_n(limbs, Ddigits, int64_t{0});

    const uint64_t half = (SharkNTT::MagicPrime - 1ull) >> 1;

    auto add32 = [&](uint32_t j, uint32_t value, bool negative) {
        if (!value)
            return;
        assert(j < Ddigits);
        const int64_t signedValue = static_cast<int64_t>(value);
        limbs[j] += negative ? -signedValue : signedValue;
    };

    for (uint32_t i = 0; i < CoefficientCount; ++i) {
        const uint64_t v = normalResidues[i];
        if (!v)
            continue;

        const bool negative = (v > half);
        const uint64_t mag64 = negative ? (SharkNTT::MagicPrime - v) : v;

        const uint64_t shiftedBits = static_cast<uint64_t>(i) * static_cast<uint64_t>(plan.b);
        const uint32_t q = static_cast<uint32_t>(shiftedBits >> 5);
        const int r = static_cast<int>(shiftedBits & 31);

        const uint64_t lo64 = r ? (mag64 << r) : mag64;
        const uint64_t hi64 = r ? (mag64 >> (64 - r)) : 0ull;

        add32(q + 0, static_cast<uint32_t>(lo64 & 0xffffffffu), negative);
        add32(q + 1, static_cast<uint32_t>((lo64 >> 32) & 0xffffffffu), negative);
        add32(q + 2, static_cast<uint32_t>(hi64 & 0xffffffffu), negative);
        add32(q + 3, static_cast<uint32_t>((hi64 >> 32) & 0xffffffffu), negative);
    }
}

template <class SharkFloatParams, uint32_t PlanN, uint32_t CoefficientCount, uint32_t Ddigits>
static void
InverseSpectrumToSignedLimbs(DebugHostCombo<SharkFloatParams> &debugCombo,
                             const SharkNTT::PlanPrime &plan,
                             SharkNTT::RootTables &roots,
                             uint64_t *spectrum,
                             int64_t *limbs)
{
    assert(static_cast<uint32_t>(plan.N) == PlanN);
    BitReverseInplace64(spectrum, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages));
    NTTRadix2<SharkFloatParams, true>(
        debugCombo, spectrum, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);

    for (uint32_t i = 0; i < PlanN; ++i) {
        uint64_t v = SharkNTT::MontgomeryMul(debugCombo, spectrum[i], roots.psi_inv_pows[i]);
        v = SharkNTT::MontgomeryMul(debugCombo, v, roots.Ninvm_mont);
        spectrum[i] = SharkNTT::FromMontgomery(debugCombo, v);
    }

    UnpackResiduesToSignedLimbs<SharkFloatParams, PlanN, CoefficientCount, Ddigits>(
        spectrum, plan, limbs);
}

template <class SharkFloatParams, uint32_t Ddigits, uint32_t MagnitudeCapacity>
static void
PropagateSignedLimbsToMagnitude(const int64_t *limbs,
                                uint32_t *magnitude,
                                uint32_t &magnitudeLength,
                                bool &negative)
{
    constexpr int64_t Base = 1ll << 32;
    uint32_t digits[MagnitudeCapacity];
    uint32_t digitLength = 0;

    const auto appendDigit = [&](uint32_t digit) {
        assert(digitLength < MagnitudeCapacity);
        digits[digitLength++] = digit;
    };

    const auto appendMagnitude = [&](uint32_t digit) {
        assert(magnitudeLength < MagnitudeCapacity);
        magnitude[magnitudeLength++] = digit;
    };

    magnitudeLength = 0;

    int64_t carry = 0;
    for (uint32_t i = 0; i < Ddigits; ++i) {
        const int64_t limb = limbs[i];
        const int64_t sum = limb + carry;
        const auto low = static_cast<uint32_t>(static_cast<uint64_t>(sum) & 0xffffffffull);
        appendDigit(low);
        carry = (sum - static_cast<int64_t>(low)) / Base;
    }

    while (carry != 0 && carry != -1) {
        const int64_t sum = carry;
        const auto low = static_cast<uint32_t>(static_cast<uint64_t>(sum) & 0xffffffffull);
        appendDigit(low);
        carry = (sum - static_cast<int64_t>(low)) / Base;
    }

    negative = (carry < 0);
    if (!negative) {
        while (digitLength > 0 && digits[digitLength - 1] == 0)
            --digitLength;
        for (uint32_t i = 0; i < digitLength; ++i) {
            appendMagnitude(digits[i]);
        }
        return;
    }

    uint64_t addOne = 1;
    for (uint32_t i = 0; i < digitLength; ++i) {
        const uint64_t sum = static_cast<uint64_t>(static_cast<uint32_t>(~digits[i])) + addOne;
        appendMagnitude(static_cast<uint32_t>(sum & 0xffffffffu));
        addOne = sum >> 32;
    }
    if (addOne)
        appendMagnitude(static_cast<uint32_t>(addOne));

    while (magnitudeLength > 0 && magnitude[magnitudeLength - 1] == 0)
        --magnitudeLength;

    if (magnitudeLength == 0)
        negative = false;
}

template <class SharkFloatParams>
static void
NormalizeMagnitudeToHpFloat(const uint32_t *magnitude,
                            uint32_t magnitudeLength,
                            int32_t commonExp,
                            bool negative,
                            HpSharkFloat<SharkFloatParams> *out)
{
    constexpr int actualDigits = SharkFloatParams::GlobalNumUint32;

    if (magnitudeLength == 0) {
        SetZero(out);
        return;
    }

    const int msd = static_cast<int>(magnitudeLength) - 1;
    const int clz = CountLeadingZeros(magnitude[msd]);
    const int currentBit = msd * 32 + (31 - clz);
    const int desiredBit = (actualDigits - 1) * 32 + 31;
    const int shiftNeeded = currentBit - desiredBit;

    if (shiftNeeded > 0) {
        MultiWordShift<ShiftDir::Right>(
            magnitude, static_cast<int>(magnitudeLength), shiftNeeded, out->Digits, actualDigits);
        out->Exponent = commonExp + shiftNeeded;
    } else if (shiftNeeded < 0) {
        const int leftShift = -shiftNeeded;
        MultiWordShift<ShiftDir::Left>(
            magnitude, static_cast<int>(magnitudeLength), leftShift, out->Digits, actualDigits);
        out->Exponent = commonExp - leftShift;
    } else {
        MultiWordShift<ShiftDir::Left>(
            magnitude, static_cast<int>(magnitudeLength), 0, out->Digits, actualDigits);
        out->Exponent = commonExp;
    }

    out->SetNegative(negative);
}

template <class SharkFloatParams, uint32_t Ddigits>
static void
FinalizeSignedStream(const FinalizationStream<SharkFloatParams, Ddigits> &stream)
{
    auto magnitude{std::make_unique<uint32_t[]>(Ddigits + 2)};
    uint32_t magnitudeLength = 0;
    bool negative = false;
    PropagateSignedLimbsToMagnitude<SharkFloatParams, Ddigits, Ddigits + 2>(
        stream.Limbs, magnitude.get(), magnitudeLength, negative);
    NormalizeMagnitudeToHpFloat<SharkFloatParams>(
        magnitude.get(), magnitudeLength, stream.CommonExp, negative, stream.Out);
}

template <class SharkFloatParams, uint32_t PlanN, bool EnableDerivative>
static void
PrepareNormalSpectra(DebugHostCombo<SharkFloatParams> &debugHostCombo,
                     const SharkNTT::PlanPrime &plan,
                     SharkNTT::RootTables &roots,
                     const HpSharkFloat<SharkFloatParams> &zReal,
                     const HpSharkFloat<SharkFloatParams> &zImag,
                     const HpSharkFloat<SharkFloatParams> &cReal,
                     const HpSharkFloat<SharkFloatParams> &cImag,
                     FusedSpectra<PlanN, EnableDerivative> &spectra)
{
    PackTwistForward(debugHostCombo, zReal, plan, roots, spectra.ZReal);
    PackTwistForward(debugHostCombo, zImag, plan, roots, spectra.ZImag);
    PackTwistForward(debugHostCombo, cReal, plan, roots, spectra.CReal);
    PackTwistForward(debugHostCombo, cImag, plan, roots, spectra.CImag);
}

template <class SharkFloatParams, uint32_t PlanN>
static void
PrepareDerivativeSpectra(DebugHostCombo<SharkFloatParams> &debugHostCombo,
                         const SharkNTT::PlanPrime &plan,
                         SharkNTT::RootTables &roots,
                         const HpSharkFloat<SharkFloatParams> &dzdcReal,
                         const HpSharkFloat<SharkFloatParams> &dzdcImag,
                         const HpSharkFloat<SharkFloatParams> &one,
                         FusedSpectra<PlanN, true> &spectra)
{
    PackTwistForward(debugHostCombo, dzdcReal, plan, roots, spectra.DzdcReal);
    PackTwistForward(debugHostCombo, dzdcImag, plan, roots, spectra.DzdcImag);
    PackTwistForward(debugHostCombo, one, plan, roots, spectra.One);
}

template <class SharkFloatParams>
static void
FusedReferenceOrbitStep(const HpSharkFloat<SharkFloatParams> &zReal,
                        const HpSharkFloat<SharkFloatParams> &zImag,
                        const HpSharkFloat<SharkFloatParams> *dzdcReal,
                        const HpSharkFloat<SharkFloatParams> *dzdcImag,
                        const HpSharkFloat<SharkFloatParams> &cReal,
                        const HpSharkFloat<SharkFloatParams> &cImag,
                        const HpSharkFloat<SharkFloatParams> *one,
                        HpSharkFloat<SharkFloatParams> *outReal,
                        HpSharkFloat<SharkFloatParams> *outImag,
                        HpSharkFloat<SharkFloatParams> *outDzdcReal,
                        HpSharkFloat<SharkFloatParams> *outDzdcImag,
                        DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    // Match the regular reference NTT plan, but use base 2^16 so the fused sum retains
    // Goldilocks-prime headroom for small relative exponent alignments.
    constexpr SharkNTT::PlanPrime plan =
        SharkNTT::BuildPlanPrime(SharkFloatParams::GlobalNumUint32, 16, 0);
    constexpr uint32_t PlanN = static_cast<uint32_t>(plan.N);
    // A product occupies coefficients [0, 2L - 2].  The c term is shifted by one
    // mantissa width and can occupy coefficient 2L - 1.
    constexpr uint32_t CoefficientCount = static_cast<uint32_t>(2 * plan.L);
    constexpr uint32_t Ddigits =
        static_cast<uint32_t>(((uint64_t)((CoefficientCount - 1) * plan.b + 64) + 31u) / 32u + 2u);

    static_assert(plan.ok, "Prime plan build failed (check b/N headroom constraints)");
    static_assert(plan.N >= CoefficientCount, "No-wrap condition violated for the fused c term");
    static_assert((SharkNTT::PHI % (2ull * static_cast<uint64_t>(plan.N))) == 0ull);

    const FusedTerm<SharkFloatParams> realZ2 =
        MakeProductTerm(zReal, SpectrumId::ZReal, zReal, SpectrumId::ZReal, false, 0);
    const FusedTerm<SharkFloatParams> realNegY2 =
        MakeProductTerm(zImag, SpectrumId::ZImag, zImag, SpectrumId::ZImag, true, 0);
    const FusedTerm<SharkFloatParams> realC = MakeLinearTerm(cReal, SpectrumId::CReal, false);

    const FusedTerm<SharkFloatParams> imagTwoZY =
        MakeProductTerm(zReal, SpectrumId::ZReal, zImag, SpectrumId::ZImag, false, 1);
    const FusedTerm<SharkFloatParams> imagC = MakeLinearTerm(cImag, SpectrumId::CImag, false);

    int32_t realCommonExp = 0;
    int32_t imagCommonExp = 0;
    const bool realIsZero = ResolveCommonExponent(realCommonExp, realZ2, realNegY2, realC);
    const bool imagIsZero = ResolveCommonExponent(imagCommonExp, imagTwoZY, imagC);
    AssertFixedPlanAlignment(realCommonExp, realZ2, realNegY2, realC);
    AssertFixedPlanAlignment(imagCommonExp, imagTwoZY, imagC);

    PrintPlan(plan);

    SharkNTT::RootTables roots{};
    SharkNTT::BuildRoots<SharkFloatParams>(
        static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);

    auto spectra{std::make_unique<FusedSpectra<PlanN, SharkFloatParams::EnableNewtonRaphson>>()};
    PrepareNormalSpectra(debugHostCombo, plan, roots, zReal, zImag, cReal, cImag, *spectra);

    auto realSpectrum{std::make_unique<uint64_t[]>(PlanN)};
    auto imagSpectrum{std::make_unique<uint64_t[]>(PlanN)};
    AccumulateOutputSpectrum(debugHostCombo,
                             plan,
                             roots,
                             *spectra,
                             realIsZero,
                             realCommonExp,
                             realSpectrum.get(),
                             realZ2,
                             realNegY2,
                             realC);
    AccumulateOutputSpectrum(debugHostCombo,
                             plan,
                             roots,
                             *spectra,
                             imagIsZero,
                             imagCommonExp,
                             imagSpectrum.get(),
                             imagTwoZY,
                             imagC);

    auto realLimbs{std::make_unique<int64_t[]>(Ddigits)};
    auto imagLimbs{std::make_unique<int64_t[]>(Ddigits)};
    if (realIsZero)
        std::fill_n(realLimbs.get(), Ddigits, int64_t{0});
    else
        InverseSpectrumToSignedLimbs<SharkFloatParams, PlanN, CoefficientCount, Ddigits>(
            debugHostCombo, plan, roots, realSpectrum.get(), realLimbs.get());

    if (imagIsZero)
        std::fill_n(imagLimbs.get(), Ddigits, int64_t{0});
    else
        InverseSpectrumToSignedLimbs<SharkFloatParams, PlanN, CoefficientCount, Ddigits>(
            debugHostCombo, plan, roots, imagSpectrum.get(), imagLimbs.get());

    const FinalizationStream<SharkFloatParams, Ddigits> realStream{
        realLimbs.get(), realCommonExp, outReal};
    const FinalizationStream<SharkFloatParams, Ddigits> imagStream{
        imagLimbs.get(), imagCommonExp, outImag};
    FinalizeSignedStream(realStream);
    FinalizeSignedStream(imagStream);

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        assert(dzdcReal != nullptr);
        assert(dzdcImag != nullptr);
        assert(one != nullptr);
        assert(outDzdcReal != nullptr);
        assert(outDzdcImag != nullptr);

        PrepareDerivativeSpectra(debugHostCombo, plan, roots, *dzdcReal, *dzdcImag, *one, *spectra);

        const FusedTerm<SharkFloatParams> dzdcRealW0 =
            MakeProductTerm(zReal, SpectrumId::ZReal, *dzdcReal, SpectrumId::DzdcReal, false, 1);
        const FusedTerm<SharkFloatParams> dzdcRealNegW1 =
            MakeProductTerm(zImag, SpectrumId::ZImag, *dzdcImag, SpectrumId::DzdcImag, true, 1);
        const FusedTerm<SharkFloatParams> dzdcRealOne = MakeLinearTerm(*one, SpectrumId::One, false);

        const FusedTerm<SharkFloatParams> dzdcImagW2 =
            MakeProductTerm(zImag, SpectrumId::ZImag, *dzdcReal, SpectrumId::DzdcReal, false, 1);
        const FusedTerm<SharkFloatParams> dzdcImagW3 =
            MakeProductTerm(zReal, SpectrumId::ZReal, *dzdcImag, SpectrumId::DzdcImag, false, 1);

        int32_t dzdcRealCommonExp = 0;
        int32_t dzdcImagCommonExp = 0;
        const bool dzdcRealIsZero =
            ResolveCommonExponent(dzdcRealCommonExp, dzdcRealW0, dzdcRealNegW1, dzdcRealOne);
        const bool dzdcImagIsZero = ResolveCommonExponent(dzdcImagCommonExp, dzdcImagW2, dzdcImagW3);

        uint64_t dzdcRealSpectrum[PlanN];
        uint64_t dzdcImagSpectrum[PlanN];
        AccumulateOutputSpectrum(debugHostCombo,
                                 plan,
                                 roots,
                                 *spectra,
                                 dzdcRealIsZero,
                                 dzdcRealCommonExp,
                                 dzdcRealSpectrum,
                                 dzdcRealW0,
                                 dzdcRealNegW1,
                                 dzdcRealOne);
        AccumulateOutputSpectrum(debugHostCombo,
                                 plan,
                                 roots,
                                 *spectra,
                                 dzdcImagIsZero,
                                 dzdcImagCommonExp,
                                 dzdcImagSpectrum,
                                 dzdcImagW2,
                                 dzdcImagW3);

        auto dzdcRealLimbs{std::make_unique<int64_t[]>(Ddigits)};
        auto dzdcImagLimbs{std::make_unique<int64_t[]>(Ddigits)};
        if (dzdcRealIsZero)
            std::fill_n(dzdcRealLimbs.get(), Ddigits, int64_t{0});
        else
            InverseSpectrumToSignedLimbs<SharkFloatParams, PlanN, CoefficientCount, Ddigits>(
                debugHostCombo, plan, roots, dzdcRealSpectrum, dzdcRealLimbs.get());

        if (dzdcImagIsZero)
            std::fill_n(dzdcImagLimbs.get(), Ddigits, int64_t{0});
        else
            InverseSpectrumToSignedLimbs<SharkFloatParams, PlanN, CoefficientCount, Ddigits>(
                debugHostCombo, plan, roots, dzdcImagSpectrum, dzdcImagLimbs.get());

        const FinalizationStream<SharkFloatParams, Ddigits> dzdcRealStream{
            dzdcRealLimbs.get(), dzdcRealCommonExp, outDzdcReal};
        const FinalizationStream<SharkFloatParams, Ddigits> dzdcImagStream{
            dzdcImagLimbs.get(), dzdcImagCommonExp, outDzdcImag};
        FinalizeSignedStream(dzdcRealStream);
        FinalizeSignedStream(dzdcImagStream);
    }

    SharkNTT::DestroyRoots<SharkFloatParams>(false, roots);
}

} // namespace

template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>
ReferenceOrbit2Helper(const HpSharkFloat<SharkFloatParams> *cReal,
                      const HpSharkFloat<SharkFloatParams> *cImag,
                      const typename SharkFloatParams::Float &radiusY,
                      uint64_t maxIters,
                      DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    auto result = std::make_unique<ReferenceOrbitResult<SharkFloatParams>>();
    result->IterationsExecuted = 0;
    result->PeriodResult = PeriodicityResult::Unknown;

    if constexpr (HpShark::DebugChecksums) {
        debugHostCombo.States.resize(static_cast<int>(DebugStatePurpose::NumPurposes));
    }

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        auto outZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        typename SharkFloatParams::Float outD2Real{};
        typename SharkFloatParams::Float outD2Imag{};

        EvaluateOrbitAndDerivative2<SharkFloatParams>(cReal,
                                                      cImag,
                                                      maxIters + 1,
                                                      outZReal.get(),
                                                      outZImag.get(),
                                                      outDzdcReal.get(),
                                                      outDzdcImag.get(),
                                                      &outD2Real,
                                                      &outD2Imag,
                                                      debugHostCombo);

        result->FinalZReal = *outZReal;
        result->FinalZImag = *outZImag;
        result->IterationsExecuted = maxIters;
        result->PeriodResult = PeriodicityResult::Continue;
        return result;
    }

    auto zReal = std::make_unique<HpSharkFloat<SharkFloatParams>>(*cReal);
    auto zImag = std::make_unique<HpSharkFloat<SharkFloatParams>>(*cImag);

    auto newZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    typename SharkFloatParams::Float dzdcX{1};
    typename SharkFloatParams::Float dzdcY{0};

    const typename SharkFloatParams::Float highTwo{2.0f};
    const typename SharkFloatParams::Float highOne{1.0f};
    const typename SharkFloatParams::Float twoFiftySix{256.0f};

    const typename SharkFloatParams::Float cxCast =
        cReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
    const typename SharkFloatParams::Float cyCast =
        cImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

    for (uint64_t i = 0; i < maxIters; ++i) {
        if constexpr (SharkFloatParams::EnablePeriodicity) {
            typename SharkFloatParams::Float doubleZx =
                zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float doubleZy =
                zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

            result->Orbit.push_back({doubleZx, doubleZy});

            HdrReduce(dzdcX);
            auto dzdcXAbs = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcYAbs = HdrAbs(dzdcY);

            HdrReduce(doubleZx);
            auto zxAbs = HdrAbs(doubleZx);

            HdrReduce(doubleZy);
            auto zyAbs = HdrAbs(doubleZy);

            typename SharkFloatParams::Float n2 = HdrMaxPositiveReduced(zxAbs, zyAbs);

            typename SharkFloatParams::Float r0 = HdrMaxPositiveReduced(dzdcXAbs, dzdcYAbs);
            auto n3 = radiusY * r0 * highTwo;
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::PeriodFound;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                return result;
            } else {
                auto dzdcXOrig = dzdcX;
                dzdcX = highTwo * (doubleZx * dzdcX - doubleZy * dzdcY) + highOne;
                dzdcY = highTwo * (doubleZx * dzdcY + doubleZy * dzdcXOrig);
            }

            typename SharkFloatParams::Float tempZX = doubleZx + cxCast;
            typename SharkFloatParams::Float tempZY = doubleZy + cyCast;
            typename SharkFloatParams::Float znSize = tempZX * tempZX + tempZY * tempZY;

            if (HdrCompareToBothPositiveReducedGT(znSize, twoFiftySix)) {
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::Escaped;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                return result;
            }
        } else {
            typename SharkFloatParams::Float doubleZx =
                zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float doubleZy =
                zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            result->Orbit.push_back({doubleZx, doubleZy});
        }

        FusedReferenceOrbitStep<SharkFloatParams>(*zReal,
                                                  *zImag,
                                                  nullptr,
                                                  nullptr,
                                                  *cReal,
                                                  *cImag,
                                                  nullptr,
                                                  newZReal.get(),
                                                  newZImag.get(),
                                                  nullptr,
                                                  nullptr,
                                                  debugHostCombo);

        *zReal = *newZReal;
        *zImag = *newZImag;

        result->IterationsExecuted = i + 1;
        result->PeriodResult = PeriodicityResult::Continue;
    }

    result->FinalZReal = *zReal;
    result->FinalZImag = *zImag;
    return result;
}

template <class SharkFloatParams>
void
EvaluateOrbitAndDerivative2(const HpSharkFloat<SharkFloatParams> *cReal,
                            const HpSharkFloat<SharkFloatParams> *cImag,
                            uint64_t period,
                            HpSharkFloat<SharkFloatParams> *outZReal,
                            HpSharkFloat<SharkFloatParams> *outZImag,
                            HpSharkFloat<SharkFloatParams> *outDzdcReal,
                            HpSharkFloat<SharkFloatParams> *outDzdcImag,
                            typename SharkFloatParams::Float *outD2Real,
                            typename SharkFloatParams::Float *outD2Imag,
                            DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    if constexpr (HpShark::DebugChecksums) {
        debugHostCombo.States.resize(static_cast<int>(DebugStatePurpose::NumPurposes));
    }

    auto zReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> dzdcReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> dzdcImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> newDzdcReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> newDzdcImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> one;
    typename SharkFloatParams::Float localD2Real{};
    typename SharkFloatParams::Float localD2Imag{};

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        dzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        newDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        newDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        one = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        one->template FromHDRFloat<typename SharkFloatParams::SubType>(
            HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});
    }

    for (uint64_t i = 0; i < period; ++i) {
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            typename SharkFloatParams::Float zr =
                zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float zi =
                zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float dzr =
                dzdcReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float dzi =
                dzdcImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

            typename SharkFloatParams::Float dz2r = dzr * dzr - dzi * dzi;
            HdrReduce(dz2r);
            typename SharkFloatParams::Float dz2i = typename SharkFloatParams::Float{2.0f} * (dzr * dzi);
            HdrReduce(dz2i);

            typename SharkFloatParams::Float zd2r = zr * localD2Real - zi * localD2Imag;
            HdrReduce(zd2r);
            typename SharkFloatParams::Float zd2i = zr * localD2Imag + zi * localD2Real;
            HdrReduce(zd2i);

            typename SharkFloatParams::Float sumr = dz2r + zd2r;
            HdrReduce(sumr);
            typename SharkFloatParams::Float sumi = dz2i + zd2i;
            HdrReduce(sumi);
            localD2Real = typename SharkFloatParams::Float{2.0f} * sumr;
            localD2Imag = typename SharkFloatParams::Float{2.0f} * sumi;
        }

        FusedReferenceOrbitStep<SharkFloatParams>(*zReal,
                                                  *zImag,
                                                  dzdcReal.get(),
                                                  dzdcImag.get(),
                                                  *cReal,
                                                  *cImag,
                                                  one.get(),
                                                  newZReal.get(),
                                                  newZImag.get(),
                                                  newDzdcReal.get(),
                                                  newDzdcImag.get(),
                                                  debugHostCombo);

        *zReal = *newZReal;
        *zImag = *newZImag;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            *dzdcReal = *newDzdcReal;
            *dzdcImag = *newDzdcImag;
        }
    }

    *outZReal = *zReal;
    *outZImag = *zImag;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        *outDzdcReal = *dzdcReal;
        *outDzdcImag = *dzdcImag;
        *outD2Real = localD2Real;
        *outD2Imag = localD2Imag;
    } else {
        SetZero(outDzdcReal);
        SetZero(outDzdcImag);
        *outD2Real = {};
        *outD2Imag = {};
    }
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>                                    \
    ReferenceOrbit2Helper<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *,                     \
                                            const HpSharkFloat<SharkFloatParams> *,                     \
                                            const typename SharkFloatParams::Float &,                   \
                                            uint64_t,                                                   \
                                            DebugHostCombo<SharkFloatParams> &);

ExplicitInstantiateAll();

#define ExplicitlyInstantiateDerivative(SharkFloatParams)                                               \
    template void EvaluateOrbitAndDerivative2<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *, \
                                                                const HpSharkFloat<SharkFloatParams> *, \
                                                                uint64_t,                               \
                                                                HpSharkFloat<SharkFloatParams> *,       \
                                                                HpSharkFloat<SharkFloatParams> *,       \
                                                                HpSharkFloat<SharkFloatParams> *,       \
                                                                HpSharkFloat<SharkFloatParams> *,       \
                                                                typename SharkFloatParams::Float *,     \
                                                                typename SharkFloatParams::Float *,     \
                                                                DebugHostCombo<SharkFloatParams> &);

#undef ExplicitlyInstantiate
#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateDerivative(SharkFloatParams)
ExplicitInstantiateAll();
