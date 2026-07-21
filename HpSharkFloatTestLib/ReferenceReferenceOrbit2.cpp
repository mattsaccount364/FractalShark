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
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

enum class ShiftDir { Left, Right };
enum class SpectrumId { ZReal, ZImag, DzdcReal, DzdcImag, CReal, CImag, One };
enum class TermKind { Product, Linear };

static bool
IsDebugTraceEnabled()
{
    return SharkVerbose == VerboseMode::Debug;
}

template <class SharkFloatParams, class ArrayType>
static void
StoreReference2DebugState(DebugHostCombo<SharkFloatParams> &debugCombo,
                          DebugStatePurpose purpose,
                          const ArrayType *arrayToChecksum,
                          size_t arraySize)
{
    if constexpr (HpShark::DebugChecksums) {
        constexpr auto CallIndex = 0;
        constexpr auto RecursionDepth = 0;
        constexpr auto UseConvolutionHere = UseConvolution::No;
        auto &debugStates = debugCombo.States;
        assert(static_cast<size_t>(purpose) < debugStates.size());
        debugStates[static_cast<size_t>(purpose)].Reset(
            arrayToChecksum, arraySize, purpose, RecursionDepth, CallIndex, UseConvolutionHere);
    }
}

template <class SharkFloatParams>
static void
StoreReference2DebugValue(DebugHostCombo<SharkFloatParams> &debugCombo,
                          DebugStatePurpose purpose,
                          const HpSharkFloat<SharkFloatParams> &value)
{
    if constexpr (HpShark::DebugChecksums) {
        constexpr auto callIndex = 0;
        constexpr auto recursionDepth = 0;
        constexpr auto useConvolution = UseConvolution::No;
        auto &debugStates = debugCombo.States;
        assert(static_cast<size_t>(purpose) < debugStates.size());
        debugStates[static_cast<size_t>(purpose)].Reset(
            value, purpose, recursionDepth, callIndex, useConvolution);
    }
}

template <class UInt>
static void
PrintHexValue(const char *label, UInt value)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "  " << label << "=0x" << std::hex << static_cast<uint64_t>(value) << std::dec;
}

static const char *
SpectrumIdName(SpectrumId id)
{
    switch (id) {
        case SpectrumId::ZReal:
            return "ZReal";
        case SpectrumId::ZImag:
            return "ZImag";
        case SpectrumId::DzdcReal:
            return "DzdcReal";
        case SpectrumId::DzdcImag:
            return "DzdcImag";
        case SpectrumId::CReal:
            return "CReal";
        case SpectrumId::CImag:
            return "CImag";
        case SpectrumId::One:
            return "One";
    }
    return "Unknown";
}

template <class SharkFloatParams>
static void
PrintHpValue(const char *label, const HpSharkFloat<SharkFloatParams> &value)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "  " << label << ": " << value.ToHexString() << '\n';
}

template <class Hdr>
static void
PrintHdrValue(const char *label, const Hdr &value)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "  " << label << ": " << value.template ToString<false>() << " ["
              << value.template ToString<true>() << "]\n";
}

template <class T>
static void
PrintArray(const char *label, const T *values, size_t count)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "  " << label << " (count=" << count << ")\n";
    for (size_t i = 0; i < count; ++i) {
        std::cout << "    [" << i << "]=0x" << std::hex << static_cast<uint64_t>(values[i]) << std::dec
                  << '\n';
    }
}

template <class SharkFloatParams> struct FusedTerm {
    bool IsZero;
    bool IsNegative;
    int32_t Exponent;
    TermKind Kind;
    SpectrumId A;
    SpectrumId B;
};

constexpr uint32_t MaxFusedN = 32u * 1024u * 1024u;
constexpr uint32_t MaxFusedStages = 25;
constexpr uint32_t MaxFusedLimbs = (MaxFusedN * 16u) / 32u + 4u;

struct FusedWorkspace {
    uint64_t *ZReal;
    uint64_t *ZImag;
    uint64_t *CReal;
    uint64_t *CImag;
    uint64_t *DzdcReal;
    uint64_t *DzdcImag;
    uint64_t *One;
    uint64_t *RealOutput;
    uint64_t *ImagOutput;
    uint64_t *DzdcRealOutput;
    uint64_t *DzdcImagOutput;
    uint64_t *Product;
    int64_t *RealLimbs;
    int64_t *ImagLimbs;
    int64_t *DzdcRealLimbs;
    int64_t *DzdcImagLimbs;
    uint32_t *MagnitudeDigits;
    uint32_t *Magnitude;
    SharkNTT::RootTables *Roots;
    uint32_t *CachedN;
};

template <class SharkFloatParams> struct GlobalFusedWorkspaceStorage {
    std::unique_ptr<uint64_t[]> ZReal;
    std::unique_ptr<uint64_t[]> ZImag;
    std::unique_ptr<uint64_t[]> CReal;
    std::unique_ptr<uint64_t[]> CImag;
    std::unique_ptr<uint64_t[]> DzdcReal;
    std::unique_ptr<uint64_t[]> DzdcImag;
    std::unique_ptr<uint64_t[]> One;
    std::unique_ptr<uint64_t[]> RealOutput;
    std::unique_ptr<uint64_t[]> ImagOutput;
    std::unique_ptr<uint64_t[]> DzdcRealOutput;
    std::unique_ptr<uint64_t[]> DzdcImagOutput;
    std::unique_ptr<uint64_t[]> Product;
    std::unique_ptr<int64_t[]> RealLimbs;
    std::unique_ptr<int64_t[]> ImagLimbs;
    std::unique_ptr<int64_t[]> DzdcRealLimbs;
    std::unique_ptr<int64_t[]> DzdcImagLimbs;
    std::unique_ptr<uint32_t[]> MagnitudeDigits;
    std::unique_ptr<uint32_t[]> Magnitude;
    std::unique_ptr<uint64_t[]> StageOmegas;
    std::unique_ptr<uint64_t[]> StageOmegasInverse;
    std::unique_ptr<uint64_t[]> PsiPowers;
    std::unique_ptr<uint64_t[]> PsiInversePowers;
    std::unique_ptr<uint64_t[]> ForwardTwiddles;
    std::unique_ptr<uint64_t[]> InverseTwiddles;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitZReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitZImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitNewZReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitNewZImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitDzdcReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitDzdcImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitNewDzdcReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitNewDzdcImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OrbitOne;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OutputZReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OutputZImag;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OutputDzdcReal;
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> OutputDzdcImag;
    SharkNTT::RootTables Roots{};
    uint32_t CachedN = 0;
};

template <class SharkFloatParams>
static GlobalFusedWorkspaceStorage<SharkFloatParams> &
GetGlobalFusedWorkspaceStorage()
{
    static GlobalFusedWorkspaceStorage<SharkFloatParams> global;
    return global;
}

template <class SharkFloatParams>
static void
EnsureGlobalFusedWorkspace()
{
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    if (!global.ZReal) {
        global.ZReal = std::make_unique<uint64_t[]>(MaxFusedN);
        global.ZImag = std::make_unique<uint64_t[]>(MaxFusedN);
        global.CReal = std::make_unique<uint64_t[]>(MaxFusedN);
        global.CImag = std::make_unique<uint64_t[]>(MaxFusedN);
        global.RealOutput = std::make_unique<uint64_t[]>(MaxFusedN);
        global.ImagOutput = std::make_unique<uint64_t[]>(MaxFusedN);
        global.DzdcRealOutput = std::make_unique<uint64_t[]>(MaxFusedN);
        global.DzdcImagOutput = std::make_unique<uint64_t[]>(MaxFusedN);
        global.Product = std::make_unique<uint64_t[]>(MaxFusedN);
        global.RealLimbs = std::make_unique<int64_t[]>(MaxFusedLimbs);
        global.ImagLimbs = std::make_unique<int64_t[]>(MaxFusedLimbs);
        global.DzdcRealLimbs = std::make_unique<int64_t[]>(MaxFusedLimbs);
        global.DzdcImagLimbs = std::make_unique<int64_t[]>(MaxFusedLimbs);
        global.MagnitudeDigits = std::make_unique<uint32_t[]>(MaxFusedLimbs);
        global.Magnitude = std::make_unique<uint32_t[]>(MaxFusedLimbs);
        global.StageOmegas = std::make_unique<uint64_t[]>(MaxFusedStages);
        global.StageOmegasInverse = std::make_unique<uint64_t[]>(MaxFusedStages);
        global.PsiPowers = std::make_unique<uint64_t[]>(MaxFusedN);
        global.PsiInversePowers = std::make_unique<uint64_t[]>(MaxFusedN);
        global.ForwardTwiddles = std::make_unique<uint64_t[]>(MaxFusedN);
        global.InverseTwiddles = std::make_unique<uint64_t[]>(MaxFusedN);
        global.Roots = {0,
                        global.StageOmegas.get(),
                        global.StageOmegasInverse.get(),
                        0,
                        global.PsiPowers.get(),
                        global.PsiInversePowers.get(),
                        0,
                        global.ForwardTwiddles.get(),
                        global.InverseTwiddles.get(),
                        0};
    }

    if (!global.OrbitZReal) {
        global.OrbitZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        global.OrbitZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        global.OrbitNewZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        global.OrbitNewZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        global.OutputZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        global.OutputZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    }

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (!global.DzdcReal) {
            global.DzdcReal = std::make_unique<uint64_t[]>(MaxFusedN);
            global.DzdcImag = std::make_unique<uint64_t[]>(MaxFusedN);
            global.One = std::make_unique<uint64_t[]>(MaxFusedN);
        }
        if (!global.OrbitDzdcReal) {
            global.OrbitDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OrbitDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OrbitNewDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OrbitNewDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OrbitOne = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OutputDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            global.OutputDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        }
    }
}

template <class SharkFloatParams>
static FusedWorkspace
GetGlobalFusedWorkspace()
{
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    return {global.ZReal.get(),
            global.ZImag.get(),
            global.CReal.get(),
            global.CImag.get(),
            global.DzdcReal.get(),
            global.DzdcImag.get(),
            global.One.get(),
            global.RealOutput.get(),
            global.ImagOutput.get(),
            global.DzdcRealOutput.get(),
            global.DzdcImagOutput.get(),
            global.Product.get(),
            global.RealLimbs.get(),
            global.ImagLimbs.get(),
            global.DzdcRealLimbs.get(),
            global.DzdcImagLimbs.get(),
            global.MagnitudeDigits.get(),
            global.Magnitude.get(),
            &global.Roots,
            &global.CachedN};
}

template <class SharkFloatParams> struct FinalizationStream {
    const int64_t *Limbs;
    uint32_t LimbCount;
    int32_t CommonExp;
    HpSharkFloat<SharkFloatParams> *Out;
};

template <class SharkFloatParams>
static void
PrintTerm(const char *label, const FusedTerm<SharkFloatParams> &term)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "  " << label << ": zero=" << term.IsZero << " negative=" << term.IsNegative
              << " kind=" << (term.Kind == TermKind::Product ? "product" : "linear")
              << " A=" << SpectrumIdName(term.A) << " B=" << SpectrumIdName(term.B)
              << " exponent=" << term.Exponent << '\n';
}

static void
PrintPlan(const SharkNTT::PlanPrime &plan)
{
    if (!IsDebugTraceEnabled())
        return;

    std::cout << "ReferenceOrbit2 fused PlanPrime:";
    PrintHexValue("n32", plan.n32);
    PrintHexValue("b", plan.b);
    PrintHexValue("L", plan.L);
    PrintHexValue("N", plan.N);
    PrintHexValue("stages", plan.stages);
    PrintHexValue("ok", plan.ok);
    std::cout << '\n';
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

static uint32_t
CountTrailingZeros(uint32_t value)
{
    assert(value != 0);
    uint32_t count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
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
    for (int i = 0; i < SharkFloatParams::GlobalNumUint32; ++i)
        out->Digits[i] = 0;
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
        if (j > i) {
            if (IsDebugTraceEnabled()) {
                std::cout << "  bit reverse swap i=" << i << " j=" << j;
                PrintHexValue("left", a[i]);
                PrintHexValue("right", a[j]);
                std::cout << '\n';
            }
            std::swap(a[i], a[j]);
        }
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
    if (IsDebugTraceEnabled()) {
        std::cout << "  " << (inverse ? "inverse" : "forward") << " NTT input\n";
        PrintArray("spectrum", a, N);
    }

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
                if (IsDebugTraceEnabled()) {
                    std::cout << "  " << (inverse ? "inverse" : "forward") << " butterfly"
                              << " stage=" << s << " k=" << k << " j=" << j << " i0=" << i0
                              << " i1=" << i1;
                    PrintHexValue("omega", wM);
                    PrintHexValue("u", u);
                    PrintHexValue("v", v);
                    PrintHexValue("twiddle", w);
                    PrintHexValue("product", t);
                    PrintHexValue("out0", a[i0]);
                    PrintHexValue("out1", a[i1]);
                    std::cout << '\n';
                }
            }
        }
        if (IsDebugTraceEnabled()) {
            std::cout << "  " << (inverse ? "inverse" : "forward") << " NTT stage " << s << " output\n";
            PrintArray("spectrum", a, N);
        }
    }
}

template <class SharkFloatParams>
static void
PackTwistForward(DebugHostCombo<SharkFloatParams> &debugCombo,
                 const HpSharkFloat<SharkFloatParams> &x,
                 const SharkNTT::PlanPrime &plan,
                 SharkNTT::RootTables &roots,
                 uint64_t *out,
                 uint32_t capacity,
                 DebugStatePurpose packedPurpose,
                 DebugStatePurpose forwardPurpose)
{
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= capacity);
    assert(zeroMont == 0);
    PrintHpValue("pack input", x);
    PrintHexValue("pack zeroMont", zeroMont);
    if (IsDebugTraceEnabled())
        std::cout << '\n';

    for (uint32_t i = 0; i < activeN; ++i) {
        const uint64_t coeff = i < static_cast<uint32_t>(plan.L)
                                   ? ReadBitsSimple(x, static_cast<int64_t>(i) * plan.b, plan.b)
                                   : 0;
        const uint64_t coeffMont = SharkNTT::ToMontgomery(debugCombo, coeff % SharkNTT::MagicPrime);
        out[i] = SharkNTT::MontgomeryMul(debugCombo, coeffMont, roots.psi_pows[i]);
        if (i < static_cast<uint32_t>(plan.L) && IsDebugTraceEnabled()) {
            std::cout << "  pack coefficient index=" << i;
            PrintHexValue("coefficient", coeff);
            PrintHexValue("coefficientMont", coeffMont);
            PrintHexValue("psi", roots.psi_pows[i]);
            PrintHexValue("twisted", out[i]);
            std::cout << '\n';
        }
    }

    PrintArray("packed/twisted spectrum", out, activeN);
    StoreReference2DebugState(debugCombo, packedPurpose, out, activeN);
    BitReverseInplace64(out, activeN, static_cast<uint32_t>(plan.stages));
    PrintArray("bit-reversed spectrum", out, activeN);
    NTTRadix2<SharkFloatParams, false>(
        debugCombo, out, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);
    StoreReference2DebugState(debugCombo, forwardPurpose, out, activeN);
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

template <class SharkFloatParams>
static void
WriteShiftedSpectrum(DebugHostCombo<SharkFloatParams> &debugCombo,
                     const SharkNTT::PlanPrime &plan,
                     const SharkNTT::RootTables &roots,
                     const uint64_t *source,
                     uint64_t shiftBits,
                     bool negative,
                     uint64_t *dest,
                     uint32_t capacity)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= capacity);

    const uint64_t chunkShift = shiftBits / static_cast<uint64_t>(plan.b);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    const uint64_t bitScale = SharkNTT::ToMontgomery(debugCombo, 1ull << bitShift);
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    for (uint32_t i = 0; i < activeN; ++i) {
        const uint64_t psiExponent = chunkShift * (1ull + 2ull * static_cast<uint64_t>(i));
        const uint64_t chunkScale = (chunkShift == 0)
                                        ? SharkNTT::ToMontgomery<SharkFloatParams>(1)
                                        : PsiPowerMont<SharkFloatParams>(plan, roots, psiExponent);
        const uint64_t scale = SharkNTT::MontgomeryMul(debugCombo, chunkScale, bitScale);
        const uint64_t shifted = SharkNTT::MontgomeryMul(debugCombo, source[i], scale);
        dest[i] = negative ? SubP(zeroMont, shifted) : shifted;
    }
}

template <class SharkFloatParams>
static void
AddShiftedSpectrum(DebugHostCombo<SharkFloatParams> &debugCombo,
                   const SharkNTT::PlanPrime &plan,
                   const SharkNTT::RootTables &roots,
                   const uint64_t *source,
                   uint64_t shiftBits,
                   bool negative,
                   uint64_t *dest,
                   uint32_t capacity)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= capacity);

    const uint64_t chunkShift = shiftBits / static_cast<uint64_t>(plan.b);
    const uint32_t bitShift = static_cast<uint32_t>(shiftBits % static_cast<uint64_t>(plan.b));
    const uint64_t bitScale = SharkNTT::ToMontgomery(debugCombo, 1ull << bitShift);
    if (IsDebugTraceEnabled()) {
        std::cout << "  AddShiftedSpectrum shiftBits=" << shiftBits << " negative=" << negative;
        PrintHexValue("chunkShift", chunkShift);
        PrintHexValue("bitShift", bitShift);
        PrintHexValue("bitScale", bitScale);
        std::cout << '\n';
    }

    for (uint32_t i = 0; i < activeN; ++i) {
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
        if (IsDebugTraceEnabled()) {
            std::cout << "  shifted spectrum index=" << i;
            PrintHexValue("source", source[i]);
            PrintHexValue("psiExponent", psiExponent);
            PrintHexValue("chunkScale", chunkScale);
            PrintHexValue("scale", scale);
            PrintHexValue("shifted", shifted);
            PrintHexValue("dest", dest[i]);
            std::cout << '\n';
        }
    }
}

static uint64_t *
GetSpectrum(FusedWorkspace &workspace, SpectrumId id)
{
    switch (id) {
        case SpectrumId::ZReal:
            return workspace.ZReal;
        case SpectrumId::ZImag:
            return workspace.ZImag;
        case SpectrumId::CReal:
            return workspace.CReal;
        case SpectrumId::CImag:
            return workspace.CImag;
        case SpectrumId::DzdcReal:
            return workspace.DzdcReal;
        case SpectrumId::DzdcImag:
            return workspace.DzdcImag;
        case SpectrumId::One:
            return workspace.One;
    }

    assert(false);
    return workspace.ZReal;
}

template <class SharkFloatParams, class... Terms>
static void
AccumulateOutputSpectrum(DebugHostCombo<SharkFloatParams> &debugCombo,
                         const SharkNTT::PlanPrime &plan,
                         const SharkNTT::RootTables &roots,
                         FusedWorkspace &workspace,
                         int32_t commonExp,
                         uint64_t *dest,
                         DebugStatePurpose checksumPurpose,
                         const FusedTerm<SharkFloatParams> &first,
                         const Terms &...terms)
{
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= MaxFusedN);
    assert(zeroMont == 0);
    if (IsDebugTraceEnabled()) {
        std::cout << "  AccumulateOutputSpectrum commonExp=" << commonExp << '\n';
        PrintHexValue("zeroMont", zeroMont);
        std::cout << '\n';
    }

    bool hasDestinationValue = false;
    const auto accumulateTerm = [&](const FusedTerm<SharkFloatParams> &term) {
        PrintTerm("accumulate term", term);
        if (term.IsZero) {
            if (IsDebugTraceEnabled())
                std::cout << "  term skipped because it is zero\n";
            return;
        }

        assert(term.Exponent >= commonExp);
        const auto shiftBits =
            static_cast<uint64_t>(static_cast<int64_t>(term.Exponent) - static_cast<int64_t>(commonExp));
        if (IsDebugTraceEnabled()) {
            std::cout << "  term alignment";
            PrintHexValue("shiftBits", shiftBits);
            std::cout << '\n';
        }

        if (term.Kind == TermKind::Product) {
            const uint64_t *a = GetSpectrum(workspace, term.A);
            const uint64_t *b = GetSpectrum(workspace, term.B);
            for (uint32_t i = 0; i < activeN; ++i) {
                workspace.Product[i] = SharkNTT::MontgomeryMul(debugCombo, a[i], b[i]);
                if (IsDebugTraceEnabled()) {
                    std::cout << "  product spectrum index=" << i;
                    PrintHexValue("a", a[i]);
                    PrintHexValue("b", b[i]);
                    PrintHexValue("product", workspace.Product[i]);
                    std::cout << '\n';
                }
            }
            PrintArray("product spectrum", workspace.Product, activeN);
            if (hasDestinationValue) {
                AddShiftedSpectrum<SharkFloatParams>(debugCombo,
                                                     plan,
                                                     roots,
                                                     workspace.Product,
                                                     shiftBits,
                                                     term.IsNegative,
                                                     dest,
                                                     MaxFusedN);
            } else {
                WriteShiftedSpectrum<SharkFloatParams>(debugCombo,
                                                       plan,
                                                       roots,
                                                       workspace.Product,
                                                       shiftBits,
                                                       term.IsNegative,
                                                       dest,
                                                       MaxFusedN);
            }
        } else {
            const uint64_t *source = GetSpectrum(workspace, term.A);
            if (hasDestinationValue) {
                AddShiftedSpectrum<SharkFloatParams>(
                    debugCombo, plan, roots, source, shiftBits, term.IsNegative, dest, MaxFusedN);
            } else {
                WriteShiftedSpectrum<SharkFloatParams>(
                    debugCombo, plan, roots, source, shiftBits, term.IsNegative, dest, MaxFusedN);
            }
        }
        hasDestinationValue = true;
        PrintArray("accumulated spectrum", dest, activeN);
    };

    accumulateTerm(first);
    (accumulateTerm(terms), ...);
    assert(hasDestinationValue);
    StoreReference2DebugState(debugCombo, checksumPurpose, dest, activeN);
}

template <class SharkFloatParams>
static void
UnpackResiduesToSignedLimbs(const uint64_t *normalResidues,
                            const SharkNTT::PlanPrime &plan,
                            uint32_t coefficientCount,
                            int64_t *limbs,
                            uint32_t limbCount)
{
    assert(coefficientCount <= static_cast<uint32_t>(plan.N));
    PrintArray("normal residues", normalResidues, coefficientCount);

    const uint64_t half = (SharkNTT::MagicPrime - 1ull) >> 1;
    for (uint32_t j = 0; j < limbCount; ++j) {
        const uint64_t firstBit = j >= 3 ? static_cast<uint64_t>(j - 3) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(j) + 1ull) * 32ull - 1ull;
        const uint64_t firstCoefficient = firstBit / static_cast<uint64_t>(plan.b);
        const uint64_t lastCoefficient = lastBit / static_cast<uint64_t>(plan.b);
        int64_t total = 0;

        for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
            const uint64_t residue = normalResidues[i];
            if (residue == 0)
                continue;

            const bool negative = residue > half;
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
    PrintArray("signed limbs", limbs, limbCount);
}

template <class SharkFloatParams>
static void
InverseSpectrumToSignedLimbs(DebugHostCombo<SharkFloatParams> &debugCombo,
                             const SharkNTT::PlanPrime &plan,
                             SharkNTT::RootTables &roots,
                             uint64_t *spectrum,
                             uint32_t coefficientCount,
                             int64_t *limbs,
                             uint32_t limbCount,
                             DebugStatePurpose residuesPurpose,
                             DebugStatePurpose limbsPurpose)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= MaxFusedN);
    PrintArray("inverse input spectrum", spectrum, activeN);
    BitReverseInplace64(spectrum, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages));
    NTTRadix2<SharkFloatParams, true>(
        debugCombo, spectrum, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);

    for (uint32_t i = 0; i < activeN; ++i) {
        uint64_t v = SharkNTT::MontgomeryMul(debugCombo, spectrum[i], roots.psi_inv_pows[i]);
        v = SharkNTT::MontgomeryMul(debugCombo, v, roots.Ninvm_mont);
        spectrum[i] = SharkNTT::FromMontgomery(debugCombo, v);
        if (IsDebugTraceEnabled()) {
            std::cout << "  inverse untwist index=" << i;
            PrintHexValue("psiInv", roots.psi_inv_pows[i]);
            PrintHexValue("normalResidue", spectrum[i]);
            std::cout << '\n';
        }
    }

    PrintArray("inverse normal residues", spectrum, activeN);
    StoreReference2DebugState(debugCombo, residuesPurpose, spectrum, activeN);

    UnpackResiduesToSignedLimbs<SharkFloatParams>(spectrum, plan, coefficientCount, limbs, limbCount);
    StoreReference2DebugState(
        debugCombo, limbsPurpose, reinterpret_cast<const uint64_t *>(limbs), limbCount);
}

template <class SharkFloatParams>
static void
PropagateSignedLimbsToMagnitude(const int64_t *limbs,
                                uint32_t limbCount,
                                uint32_t *digits,
                                uint32_t *magnitude,
                                uint32_t magnitudeCapacity,
                                uint32_t &digitLength,
                                uint32_t &magnitudeLength,
                                bool &negative)
{
    constexpr int64_t Base = 1ll << 32;
    assert(magnitudeCapacity >= limbCount + 2);
    digitLength = 0;

    const auto appendDigit = [&](uint32_t digit) {
        assert(digitLength < magnitudeCapacity);
        digits[digitLength++] = digit;
    };

    const auto appendMagnitude = [&](uint32_t digit) {
        assert(magnitudeLength < magnitudeCapacity);
        magnitude[magnitudeLength++] = digit;
    };

    magnitudeLength = 0;
    PrintArray("propagate input limbs", limbs, limbCount);

    int64_t carry = 0;
    for (uint32_t i = 0; i < limbCount; ++i) {
        const int64_t limb = limbs[i];
        const int64_t sum = limb + carry;
        const auto low = static_cast<uint32_t>(static_cast<uint64_t>(sum) & 0xffffffffull);
        appendDigit(low);
        carry = (sum - static_cast<int64_t>(low)) / Base;
        if (IsDebugTraceEnabled()) {
            std::cout << "  carry propagation index=" << i;
            PrintHexValue("limb", limb);
            PrintHexValue("sum", sum);
            PrintHexValue("low", low);
            PrintHexValue("carry", carry);
            std::cout << '\n';
        }
    }

    while (carry != 0 && carry != -1) {
        const int64_t sum = carry;
        const auto low = static_cast<uint32_t>(static_cast<uint64_t>(sum) & 0xffffffffull);
        appendDigit(low);
        carry = (sum - static_cast<int64_t>(low)) / Base;
        if (IsDebugTraceEnabled()) {
            std::cout << "  carry extension";
            PrintHexValue("sum", sum);
            PrintHexValue("low", low);
            PrintHexValue("carry", carry);
            std::cout << '\n';
        }
    }

    negative = (carry < 0);
    if (!negative) {
        uint32_t nonzeroDigitLength = digitLength;
        while (nonzeroDigitLength > 0 && digits[nonzeroDigitLength - 1] == 0)
            --nonzeroDigitLength;
        for (uint32_t i = 0; i < nonzeroDigitLength; ++i) {
            appendMagnitude(digits[i]);
        }
        PrintArray("propagated magnitude", magnitude, magnitudeLength);
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
    PrintArray("propagated magnitude", magnitude, magnitudeLength);
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

    PrintArray("normalize magnitude", magnitude, magnitudeLength);
    if (magnitudeLength == 0) {
        if (IsDebugTraceEnabled())
            std::cout << "  normalize zero magnitude\n";
        SetZero(out);
        PrintHpValue("normalized output", *out);
        return;
    }

    const int msd = static_cast<int>(magnitudeLength) - 1;
    const int clz = CountLeadingZeros(magnitude[msd]);
    const int currentBit = msd * 32 + (31 - clz);
    const int desiredBit = (actualDigits - 1) * 32 + 31;
    const int shiftNeeded = currentBit - desiredBit;
    if (IsDebugTraceEnabled()) {
        std::cout << "  normalization msd=" << msd << " clz=" << clz << " currentBit=" << currentBit
                  << " desiredBit=" << desiredBit << " shiftNeeded=" << shiftNeeded
                  << " commonExp=" << commonExp << " negative=" << negative << '\n';
    }

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
    PrintHpValue("normalized output", *out);
}

template <class SharkFloatParams>
static void
FinalizeSignedStream(const FinalizationStream<SharkFloatParams> &stream,
                     uint32_t *digits,
                     uint32_t *magnitude,
                     uint32_t magnitudeCapacity,
                     DebugHostCombo<SharkFloatParams> &debugCombo,
                     DebugStatePurpose digitsPurpose,
                     DebugStatePurpose magnitudePurpose)
{
    if (IsDebugTraceEnabled())
        std::cout << "  FinalizeSignedStream commonExp=" << stream.CommonExp << '\n';
    PrintArray("finalization limbs", stream.Limbs, stream.LimbCount);
    uint32_t digitLength = 0;
    uint32_t magnitudeLength = 0;
    bool negative = false;
    PropagateSignedLimbsToMagnitude<SharkFloatParams>(stream.Limbs,
                                                      stream.LimbCount,
                                                      digits,
                                                      magnitude,
                                                      magnitudeCapacity,
                                                      digitLength,
                                                      magnitudeLength,
                                                      negative);
    StoreReference2DebugState(debugCombo, digitsPurpose, digits, digitLength);
    StoreReference2DebugState(debugCombo, magnitudePurpose, magnitude, magnitudeLength);
    NormalizeMagnitudeToHpFloat<SharkFloatParams>(
        magnitude, magnitudeLength, stream.CommonExp, negative, stream.Out);
}

template <class SharkFloatParams>
static void
PrepareNormalSpectra(DebugHostCombo<SharkFloatParams> &debugHostCombo,
                     const SharkNTT::PlanPrime &plan,
                     SharkNTT::RootTables &roots,
                     const HpSharkFloat<SharkFloatParams> &zReal,
                     const HpSharkFloat<SharkFloatParams> &zImag,
                     const HpSharkFloat<SharkFloatParams> &cReal,
                     const HpSharkFloat<SharkFloatParams> &cImag,
                     FusedWorkspace &workspace)
{
    PackTwistForward(debugHostCombo,
                     zReal,
                     plan,
                     roots,
                     workspace.ZReal,
                     MaxFusedN,
                     DebugStatePurpose::Z0XX,
                     DebugStatePurpose::Z2XX);
    PackTwistForward(debugHostCombo,
                     zImag,
                     plan,
                     roots,
                     workspace.ZImag,
                     MaxFusedN,
                     DebugStatePurpose::Z0YY,
                     DebugStatePurpose::Z2YY);
    PackTwistForward(debugHostCombo,
                     cReal,
                     plan,
                     roots,
                     workspace.CReal,
                     MaxFusedN,
                     DebugStatePurpose::Z0XY,
                     DebugStatePurpose::Z2XY);
    PackTwistForward(debugHostCombo,
                     cImag,
                     plan,
                     roots,
                     workspace.CImag,
                     MaxFusedN,
                     DebugStatePurpose::Z0W0,
                     DebugStatePurpose::Z2W0);
}

template <class SharkFloatParams>
static void
PrepareDerivativeSpectra(DebugHostCombo<SharkFloatParams> &debugHostCombo,
                         const SharkNTT::PlanPrime &plan,
                         SharkNTT::RootTables &roots,
                         const HpSharkFloat<SharkFloatParams> &dzdcReal,
                         const HpSharkFloat<SharkFloatParams> &dzdcImag,
                         const HpSharkFloat<SharkFloatParams> &one,
                         FusedWorkspace &workspace)
{
    PackTwistForward(debugHostCombo,
                     dzdcReal,
                     plan,
                     roots,
                     workspace.DzdcReal,
                     MaxFusedN,
                     DebugStatePurpose::Z0W1,
                     DebugStatePurpose::Z2W1);
    PackTwistForward(debugHostCombo,
                     dzdcImag,
                     plan,
                     roots,
                     workspace.DzdcImag,
                     MaxFusedN,
                     DebugStatePurpose::Z0W2,
                     DebugStatePurpose::Z2W2);
    PackTwistForward(debugHostCombo,
                     one,
                     plan,
                     roots,
                     workspace.One,
                     MaxFusedN,
                     DebugStatePurpose::Z0W3,
                     DebugStatePurpose::Z2W3);
}

template <class SharkFloatParams>
static void
GenerateActiveRoots(DebugHostCombo<SharkFloatParams> &debugCombo,
                    uint32_t activeN,
                    FusedWorkspace &workspace)
{
    if (*workspace.CachedN == activeN)
        return;

    const uint32_t stages = CountTrailingZeros(activeN);
    assert(activeN <= MaxFusedN && stages <= MaxFusedStages);
    auto &roots = *workspace.Roots;
    roots.N = static_cast<int32_t>(activeN);
    roots.stages = static_cast<int32_t>(stages);
    roots.total_twiddles = activeN - 1;

    const uint64_t generator = SharkNTT::FindGeneratorConstexpr();
    const uint64_t generatorMont = SharkNTT::ToMontgomery<SharkFloatParams>(generator);
    const uint64_t exponent = SharkNTT::PHI / (2ull * activeN);
    const uint64_t psiMont = SharkNTT::MontgomeryPow<SharkFloatParams>(generatorMont, exponent);
    const uint64_t psiInverseMont =
        SharkNTT::MontgomeryPow<SharkFloatParams>(psiMont, SharkNTT::PHI - 1ull);
    const uint64_t omegaMont = SharkNTT::MontgomeryMul<SharkFloatParams>(psiMont, psiMont);
    const uint64_t omegaInverseMont =
        SharkNTT::MontgomeryPow<SharkFloatParams>(omegaMont, SharkNTT::PHI - 1ull);
    const uint64_t oneMont = SharkNTT::ToMontgomery<SharkFloatParams>(1);

    roots.psi_pows[0] = oneMont;
    roots.psi_inv_pows[0] = oneMont;
    for (uint32_t i = 1; i < activeN; ++i) {
        roots.psi_pows[i] = SharkNTT::MontgomeryMul<SharkFloatParams>(roots.psi_pows[i - 1], psiMont);
        roots.psi_inv_pows[i] =
            SharkNTT::MontgomeryMul<SharkFloatParams>(roots.psi_inv_pows[i - 1], psiInverseMont);
    }

    uint32_t offset = 0;
    for (uint32_t stage = 1; stage <= stages; ++stage) {
        const uint32_t m = 1u << stage;
        const uint32_t half = m >> 1;
        roots.stage_omegas[stage - 1] =
            SharkNTT::MontgomeryPow<SharkFloatParams>(omegaMont, activeN / m);
        roots.stage_omegas_inv[stage - 1] =
            SharkNTT::MontgomeryPow<SharkFloatParams>(omegaInverseMont, activeN / m);
        uint64_t forward = oneMont;
        uint64_t inverse = oneMont;
        for (uint32_t j = 0; j < half; ++j) {
            roots.stage_twiddles_fwd[offset + j] = forward;
            roots.stage_twiddles_inv[offset + j] = inverse;
            forward = SharkNTT::MontgomeryMul<SharkFloatParams>(forward, roots.stage_omegas[stage - 1]);
            inverse =
                SharkNTT::MontgomeryMul<SharkFloatParams>(inverse, roots.stage_omegas_inv[stage - 1]);
        }
        offset += half;
    }

    roots.Ninvm_mont = oneMont;
    const uint64_t inverseTwo =
        SharkNTT::ToMontgomery<SharkFloatParams>((SharkNTT::MagicPrime + 1) >> 1);
    for (uint32_t stage = 0; stage < stages; ++stage)
        roots.Ninvm_mont = SharkNTT::MontgomeryMul<SharkFloatParams>(roots.Ninvm_mont, inverseTwo);
    *workspace.CachedN = activeN;
    (void)debugCombo;
}

template <class SharkFloatParams, class... Terms>
static uint64_t
RequiredBitsForStream(int32_t commonExp, const FusedTerm<SharkFloatParams> &first, const Terms &...terms)
{
    constexpr uint64_t MantissaBits = static_cast<uint64_t>(SharkFloatParams::GlobalNumUint32) * 32ull;
    uint64_t requiredBits = 0;
    const auto includeTerm = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero)
            return;
        const int64_t signedShift = static_cast<int64_t>(term.Exponent) - commonExp;
        assert(signedShift >= 0);
        const uint64_t width = term.Kind == TermKind::Product ? 2ull * MantissaBits : MantissaBits;
        requiredBits = std::max(requiredBits, static_cast<uint64_t>(signedShift) + width);
    };
    includeTerm(first);
    (includeTerm(terms), ...);
    return requiredBits;
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
                        FusedWorkspace &workspace,
                        DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    if (IsDebugTraceEnabled())
        std::cout << "ReferenceOrbit2 fused step begin\n";
    PrintHpValue("zReal", zReal);
    PrintHpValue("zImag", zImag);
    PrintHpValue("cReal", cReal);
    PrintHpValue("cImag", cImag);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        PrintHpValue("dzdcReal", *dzdcReal);
        PrintHpValue("dzdcImag", *dzdcImag);
        PrintHpValue("one", *one);
    }
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

    uint64_t maxRequiredBits = std::max(RequiredBitsForStream(realCommonExp, realZ2, realNegY2, realC),
                                        RequiredBitsForStream(imagCommonExp, imagTwoZY, imagC));

    int32_t dzdcRealCommonExp = 0;
    int32_t dzdcImagCommonExp = 0;
    bool dzdcRealIsZero = true;
    bool dzdcImagIsZero = true;
    FusedTerm<SharkFloatParams> dzdcRealW0{};
    FusedTerm<SharkFloatParams> dzdcRealNegW1{};
    FusedTerm<SharkFloatParams> dzdcRealOne{};
    FusedTerm<SharkFloatParams> dzdcImagW2{};
    FusedTerm<SharkFloatParams> dzdcImagW3{};
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        dzdcRealW0 =
            MakeProductTerm(zReal, SpectrumId::ZReal, *dzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcRealNegW1 =
            MakeProductTerm(zImag, SpectrumId::ZImag, *dzdcImag, SpectrumId::DzdcImag, true, 1);
        dzdcRealOne = MakeLinearTerm(*one, SpectrumId::One, false);
        dzdcImagW2 =
            MakeProductTerm(zImag, SpectrumId::ZImag, *dzdcReal, SpectrumId::DzdcReal, false, 1);
        dzdcImagW3 =
            MakeProductTerm(zReal, SpectrumId::ZReal, *dzdcImag, SpectrumId::DzdcImag, false, 1);
        dzdcRealIsZero =
            ResolveCommonExponent(dzdcRealCommonExp, dzdcRealW0, dzdcRealNegW1, dzdcRealOne);
        dzdcImagIsZero = ResolveCommonExponent(dzdcImagCommonExp, dzdcImagW2, dzdcImagW3);
        maxRequiredBits =
            std::max(maxRequiredBits,
                     RequiredBitsForStream(dzdcRealCommonExp, dzdcRealW0, dzdcRealNegW1, dzdcRealOne));
        maxRequiredBits =
            std::max(maxRequiredBits, RequiredBitsForStream(dzdcImagCommonExp, dzdcImagW2, dzdcImagW3));
    }

    if (IsDebugTraceEnabled()) {
        std::cout << "maxRequiredBits=" << maxRequiredBits << " realCommonExp=" << realCommonExp
                  << " imagCommonExp=" << imagCommonExp << std::endl;
    }

    if (maxRequiredBits == 0) {
        SetZero(outReal);
        SetZero(outImag);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add1, *outReal);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add2, *outImag);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            SetZero(outDzdcReal);
            SetZero(outDzdcImag);
            StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc1, *outDzdcReal);
            StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc2, *outDzdcImag);
        }
        return;
    }

    static constexpr const SharkNTT::PlanPrime &basePlan = SharkFloatParams::NTTPlan2;
    assert(basePlan.ok);
    assert(basePlan.b > 0);
    const uint64_t coefficientBits = static_cast<uint64_t>(basePlan.b);
    const uint64_t requiredCoefficients = (maxRequiredBits + coefficientBits - 1ull) / coefficientBits;
    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > MaxFusedN) {
        std::cerr << "ReferenceOrbit2 fused workspace exceeded: requestedBits=" << maxRequiredBits
                  << " requiredN=" << requiredN << " capacity=" << MaxFusedN << '\n';
        assert(false);
    }
    const uint32_t cachedN = *workspace.CachedN;
    assert(cachedN == 0 || (cachedN <= MaxFusedN && (cachedN & (cachedN - 1u)) == 0));
    const uint32_t activeN =
        requiredN > static_cast<uint64_t>(cachedN) ? static_cast<uint32_t>(requiredN) : cachedN;
    assert(activeN >= 2u);
    assert((SharkNTT::PHI % (2ull * activeN)) == 0ull);
    const SharkNTT::PlanPrime plan{basePlan.n32,
                                   basePlan.b,
                                   basePlan.L,
                                   static_cast<int>(activeN),
                                   static_cast<int>(CountTrailingZeros(activeN)),
                                   basePlan.ok};
    const uint32_t coefficientCount = activeN;
    const uint32_t limbCount = (coefficientCount * static_cast<uint32_t>(plan.b) + 31u) / 32u + 2u;
    assert(limbCount <= MaxFusedLimbs);

    PrintTerm("real z^2", realZ2);
    PrintTerm("real -y^2", realNegY2);
    PrintTerm("real c", realC);
    PrintTerm("imag 2zy", imagTwoZY);
    PrintTerm("imag c", imagC);
    if (IsDebugTraceEnabled()) {
        std::cout << "  realIsZero=" << realIsZero << " realCommonExp=" << realCommonExp
                  << " imagIsZero=" << imagIsZero << " imagCommonExp=" << imagCommonExp << '\n';
    }

    PrintPlan(plan);

    GenerateActiveRoots(debugHostCombo, activeN, workspace);
    SharkNTT::RootTables &roots = *workspace.Roots;
    if (IsDebugTraceEnabled()) {
        PrintArray("roots.stage_omegas", roots.stage_omegas, roots.stages);
        PrintArray("roots.stage_omegas_inv", roots.stage_omegas_inv, roots.stages);
        PrintArray("roots.psi_pows", roots.psi_pows, roots.N);
        PrintArray("roots.psi_inv_pows", roots.psi_inv_pows, roots.N);
        PrintArray("roots.stage_twiddles_fwd", roots.stage_twiddles_fwd, roots.total_twiddles);
        PrintArray("roots.stage_twiddles_inv", roots.stage_twiddles_inv, roots.total_twiddles);
        PrintHexValue("roots.Ninvm_mont", roots.Ninvm_mont);
        std::cout << '\n';
    }

    PrepareNormalSpectra(debugHostCombo, plan, roots, zReal, zImag, cReal, cImag, workspace);

    if (realIsZero) {
        SetZero(outReal);
    } else {
        AccumulateOutputSpectrum(debugHostCombo,
                                 plan,
                                 roots,
                                 workspace,
                                 realCommonExp,
                                 workspace.RealOutput,
                                 DebugStatePurpose::Z2_Perm1,
                                 realZ2,
                                 realNegY2,
                                 realC);
        InverseSpectrumToSignedLimbs<SharkFloatParams>(debugHostCombo,
                                                       plan,
                                                       roots,
                                                       workspace.RealOutput,
                                                       coefficientCount,
                                                       workspace.RealLimbs,
                                                       limbCount,
                                                       DebugStatePurpose::Z2_Perm4,
                                                       DebugStatePurpose::UnpackXX);
        const FinalizationStream<SharkFloatParams> realStream{
            workspace.RealLimbs, limbCount, realCommonExp, outReal};
        FinalizeSignedStream(realStream,
                             workspace.MagnitudeDigits,
                             workspace.Magnitude,
                             MaxFusedLimbs,
                             debugHostCombo,
                             DebugStatePurpose::SignedCarry1,
                             DebugStatePurpose::FinalAdd1);
    }

    if (imagIsZero) {
        SetZero(outImag);
    } else {
        AccumulateOutputSpectrum(debugHostCombo,
                                 plan,
                                 roots,
                                 workspace,
                                 imagCommonExp,
                                 workspace.ImagOutput,
                                 DebugStatePurpose::Z2_Perm2,
                                 imagTwoZY,
                                 imagC);
        InverseSpectrumToSignedLimbs<SharkFloatParams>(debugHostCombo,
                                                       plan,
                                                       roots,
                                                       workspace.ImagOutput,
                                                       coefficientCount,
                                                       workspace.ImagLimbs,
                                                       limbCount,
                                                       DebugStatePurpose::Z2_Perm5,
                                                       DebugStatePurpose::UnpackYY);
        const FinalizationStream<SharkFloatParams> imagStream{
            workspace.ImagLimbs, limbCount, imagCommonExp, outImag};
        FinalizeSignedStream(imagStream,
                             workspace.MagnitudeDigits,
                             workspace.Magnitude,
                             MaxFusedLimbs,
                             debugHostCombo,
                             DebugStatePurpose::SignedCarry2,
                             DebugStatePurpose::FinalAdd2);
    }
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add1, *outReal);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add2, *outImag);
    PrintHpValue("fused outReal", *outReal);
    PrintHpValue("fused outImag", *outImag);

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        assert(dzdcReal != nullptr);
        assert(dzdcImag != nullptr);
        assert(one != nullptr);
        assert(outDzdcReal != nullptr);
        assert(outDzdcImag != nullptr);

        PrepareDerivativeSpectra(debugHostCombo, plan, roots, *dzdcReal, *dzdcImag, *one, workspace);
        PrintTerm("dzdc real w0", dzdcRealW0);
        PrintTerm("dzdc real -w1", dzdcRealNegW1);
        PrintTerm("dzdc real one", dzdcRealOne);
        PrintTerm("dzdc imag w2", dzdcImagW2);
        PrintTerm("dzdc imag w3", dzdcImagW3);
        if (IsDebugTraceEnabled()) {
            std::cout << "  dzdcRealIsZero=" << dzdcRealIsZero
                      << " dzdcRealCommonExp=" << dzdcRealCommonExp
                      << " dzdcImagIsZero=" << dzdcImagIsZero
                      << " dzdcImagCommonExp=" << dzdcImagCommonExp << '\n';
        }

        if (dzdcRealIsZero) {
            SetZero(outDzdcReal);
        } else {
            AccumulateOutputSpectrum(debugHostCombo,
                                     plan,
                                     roots,
                                     workspace,
                                     dzdcRealCommonExp,
                                     workspace.DzdcRealOutput,
                                     DebugStatePurpose::Z2_PermW0,
                                     dzdcRealW0,
                                     dzdcRealNegW1,
                                     dzdcRealOne);
            InverseSpectrumToSignedLimbs<SharkFloatParams>(debugHostCombo,
                                                           plan,
                                                           roots,
                                                           workspace.DzdcRealOutput,
                                                           coefficientCount,
                                                           workspace.DzdcRealLimbs,
                                                           limbCount,
                                                           DebugStatePurpose::Z2_PermW0b,
                                                           DebugStatePurpose::UnpackW0);
            const FinalizationStream<SharkFloatParams> dzdcRealStream{
                workspace.DzdcRealLimbs, limbCount, dzdcRealCommonExp, outDzdcReal};
            FinalizeSignedStream(dzdcRealStream,
                                 workspace.MagnitudeDigits,
                                 workspace.Magnitude,
                                 MaxFusedLimbs,
                                 debugHostCombo,
                                 DebugStatePurpose::SignedCarryDzdc1,
                                 DebugStatePurpose::FinalAddDzdc1);
        }

        if (dzdcImagIsZero) {
            SetZero(outDzdcImag);
        } else {
            AccumulateOutputSpectrum(debugHostCombo,
                                     plan,
                                     roots,
                                     workspace,
                                     dzdcImagCommonExp,
                                     workspace.DzdcImagOutput,
                                     DebugStatePurpose::Z2_PermW1,
                                     dzdcImagW2,
                                     dzdcImagW3);
            InverseSpectrumToSignedLimbs<SharkFloatParams>(debugHostCombo,
                                                           plan,
                                                           roots,
                                                           workspace.DzdcImagOutput,
                                                           coefficientCount,
                                                           workspace.DzdcImagLimbs,
                                                           limbCount,
                                                           DebugStatePurpose::Z2_PermW1b,
                                                           DebugStatePurpose::UnpackW1);
            const FinalizationStream<SharkFloatParams> dzdcImagStream{
                workspace.DzdcImagLimbs, limbCount, dzdcImagCommonExp, outDzdcImag};
            FinalizeSignedStream(dzdcImagStream,
                                 workspace.MagnitudeDigits,
                                 workspace.Magnitude,
                                 MaxFusedLimbs,
                                 debugHostCombo,
                                 DebugStatePurpose::SignedCarryDzdc2,
                                 DebugStatePurpose::FinalAddDzdc2);
        }
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc1, *outDzdcReal);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc2, *outDzdcImag);
        PrintHpValue("fused outDzdcReal", *outDzdcReal);
        PrintHpValue("fused outDzdcImag", *outDzdcImag);
    }

    if (IsDebugTraceEnabled())
        std::cout << "ReferenceOrbit2 fused step end\n";
}

} // namespace

template <class SharkFloatParams>
static void EvaluateOrbitAndDerivative2Impl(const HpSharkFloat<SharkFloatParams> *cReal,
                                            const HpSharkFloat<SharkFloatParams> *cImag,
                                            uint64_t period,
                                            HpSharkFloat<SharkFloatParams> *outZReal,
                                            HpSharkFloat<SharkFloatParams> *outZImag,
                                            HpSharkFloat<SharkFloatParams> *outDzdcReal,
                                            HpSharkFloat<SharkFloatParams> *outDzdcImag,
                                            typename SharkFloatParams::Float *outD2Real,
                                            typename SharkFloatParams::Float *outD2Imag,
                                            FusedWorkspace &workspace,
                                            DebugHostCombo<SharkFloatParams> &debugHostCombo);

template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>
ReferenceOrbit2Helper(const HpSharkFloat<SharkFloatParams> *cReal,
                      const HpSharkFloat<SharkFloatParams> *cImag,
                      const typename SharkFloatParams::Float &radiusY,
                      uint64_t maxIters,
                      DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    if (IsDebugTraceEnabled()) {
        std::cout << "ReferenceOrbit2Helper begin maxIters=" << maxIters
                  << " EnableNewtonRaphson=" << SharkFloatParams::EnableNewtonRaphson
                  << " EnablePeriodicity=" << SharkFloatParams::EnablePeriodicity << '\n';
    }
    PrintHpValue("input cReal", *cReal);
    PrintHpValue("input cImag", *cImag);
    PrintHdrValue("radiusY", radiusY);
    auto result = std::make_unique<ReferenceOrbitResult<SharkFloatParams>>();
    result->IterationsExecuted = 0;
    result->PeriodResult = PeriodicityResult::Unknown;

    EraseAllDebugStates(debugHostCombo);

    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceEntryZReal, *cReal);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceEntryZImag, *cImag);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceEntryCReal, *cReal);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceEntryCImag, *cImag);

    EnsureGlobalFusedWorkspace<SharkFloatParams>();
    FusedWorkspace workspace = GetGlobalFusedWorkspace<SharkFloatParams>();
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    global.CachedN = 0;

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        HpSharkFloat<SharkFloatParams> *outZReal = global.OutputZReal.get();
        HpSharkFloat<SharkFloatParams> *outZImag = global.OutputZImag.get();
        HpSharkFloat<SharkFloatParams> *outDzdcReal = global.OutputDzdcReal.get();
        HpSharkFloat<SharkFloatParams> *outDzdcImag = global.OutputDzdcImag.get();
        typename SharkFloatParams::Float outD2Real{};
        typename SharkFloatParams::Float outD2Imag{};

        EvaluateOrbitAndDerivative2Impl<SharkFloatParams>(cReal,
                                                          cImag,
                                                          maxIters + 1,
                                                          outZReal,
                                                          outZImag,
                                                          outDzdcReal,
                                                          outDzdcImag,
                                                          &outD2Real,
                                                          &outD2Imag,
                                                          workspace,
                                                          debugHostCombo);

        result->FinalZReal = *outZReal;
        result->FinalZImag = *outZImag;
        result->IterationsExecuted = maxIters;
        result->PeriodResult = PeriodicityResult::Continue;
        PrintHpValue("ReferenceOrbit2 NR final zReal", result->FinalZReal);
        PrintHpValue("ReferenceOrbit2 NR final zImag", result->FinalZImag);
        PrintHpValue("ReferenceOrbit2 NR final dzdcReal", *outDzdcReal);
        PrintHpValue("ReferenceOrbit2 NR final dzdcImag", *outDzdcImag);
        PrintHdrValue("ReferenceOrbit2 NR final d2Real", outD2Real);
        PrintHdrValue("ReferenceOrbit2 NR final d2Imag", outD2Imag);
        StoreReference2DebugValue(
            debugHostCombo, DebugStatePurpose::ReferenceExitZReal, result->FinalZReal);
        StoreReference2DebugValue(
            debugHostCombo, DebugStatePurpose::ReferenceExitZImag, result->FinalZImag);
        return result;
    }

    HpSharkFloat<SharkFloatParams> *zReal = global.OrbitZReal.get();
    HpSharkFloat<SharkFloatParams> *zImag = global.OrbitZImag.get();
    HpSharkFloat<SharkFloatParams> *newZReal = global.OrbitNewZReal.get();
    HpSharkFloat<SharkFloatParams> *newZImag = global.OrbitNewZImag.get();
    *zReal = *cReal;
    *zImag = *cImag;

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
        if (IsDebugTraceEnabled())
            std::cout << "ReferenceOrbit2 iteration " << i << " begin\n";
        PrintHpValue("iteration zReal", *zReal);
        PrintHpValue("iteration zImag", *zImag);
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

            PrintHdrValue("periodicity doubleZx", doubleZx);
            PrintHdrValue("periodicity doubleZy", doubleZy);
            PrintHdrValue("periodicity dzdcX", dzdcX);
            PrintHdrValue("periodicity dzdcY", dzdcY);
            PrintHdrValue("periodicity n2", n2);
            PrintHdrValue("periodicity r0", r0);
            PrintHdrValue("periodicity n3", n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if (IsDebugTraceEnabled())
                    std::cout << "ReferenceOrbit2 periodicity: period found\n";
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::PeriodFound;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                StoreReference2DebugValue(
                    debugHostCombo, DebugStatePurpose::ReferenceExitZReal, result->FinalZReal);
                StoreReference2DebugValue(
                    debugHostCombo, DebugStatePurpose::ReferenceExitZImag, result->FinalZImag);
                return result;
            } else {
                auto dzdcXOrig = dzdcX;
                dzdcX = highTwo * (doubleZx * dzdcX - doubleZy * dzdcY) + highOne;
                dzdcY = highTwo * (doubleZx * dzdcY + doubleZy * dzdcXOrig);
                PrintHdrValue("periodicity updated dzdcX", dzdcX);
                PrintHdrValue("periodicity updated dzdcY", dzdcY);
            }

            typename SharkFloatParams::Float tempZX = doubleZx + cxCast;
            typename SharkFloatParams::Float tempZY = doubleZy + cyCast;
            typename SharkFloatParams::Float znSize = tempZX * tempZX + tempZY * tempZY;
            PrintHdrValue("escape tempZX", tempZX);
            PrintHdrValue("escape tempZY", tempZY);
            PrintHdrValue("escape znSize", znSize);

            if (HdrCompareToBothPositiveReducedGT(znSize, twoFiftySix)) {
                if (IsDebugTraceEnabled())
                    std::cout << "ReferenceOrbit2 periodicity: escaped\n";
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::Escaped;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                StoreReference2DebugValue(
                    debugHostCombo, DebugStatePurpose::ReferenceExitZReal, result->FinalZReal);
                StoreReference2DebugValue(
                    debugHostCombo, DebugStatePurpose::ReferenceExitZImag, result->FinalZImag);
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
                                                  newZReal,
                                                  newZImag,
                                                  nullptr,
                                                  nullptr,
                                                  workspace,
                                                  debugHostCombo);

        *zReal = *newZReal;
        *zImag = *newZImag;
        PrintHpValue("iteration next zReal", *zReal);
        PrintHpValue("iteration next zImag", *zImag);

        result->IterationsExecuted = i + 1;
        result->PeriodResult = PeriodicityResult::Continue;
    }

    result->FinalZReal = *zReal;
    result->FinalZImag = *zImag;
    PrintHpValue("ReferenceOrbit2 final zReal", result->FinalZReal);
    PrintHpValue("ReferenceOrbit2 final zImag", result->FinalZImag);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceExitZReal, result->FinalZReal);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::ReferenceExitZImag, result->FinalZImag);
    return result;
}

template <class SharkFloatParams>
static void
EvaluateOrbitAndDerivative2Impl(const HpSharkFloat<SharkFloatParams> *cReal,
                                const HpSharkFloat<SharkFloatParams> *cImag,
                                uint64_t period,
                                HpSharkFloat<SharkFloatParams> *outZReal,
                                HpSharkFloat<SharkFloatParams> *outZImag,
                                HpSharkFloat<SharkFloatParams> *outDzdcReal,
                                HpSharkFloat<SharkFloatParams> *outDzdcImag,
                                typename SharkFloatParams::Float *outD2Real,
                                typename SharkFloatParams::Float *outD2Imag,
                                FusedWorkspace &workspace,
                                DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    if (IsDebugTraceEnabled())
        std::cout << "EvaluateOrbitAndDerivative2 begin period=" << period << '\n';
    PrintHpValue("derivative input cReal", *cReal);
    PrintHpValue("derivative input cImag", *cImag);
    if constexpr (HpShark::DebugChecksums) {
        debugHostCombo.States.resize(static_cast<int>(DebugStatePurpose::NumPurposes));
    }

    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    HpSharkFloat<SharkFloatParams> *zReal = global.OrbitZReal.get();
    HpSharkFloat<SharkFloatParams> *zImag = global.OrbitZImag.get();
    HpSharkFloat<SharkFloatParams> *newZReal = global.OrbitNewZReal.get();
    HpSharkFloat<SharkFloatParams> *newZImag = global.OrbitNewZImag.get();
    HpSharkFloat<SharkFloatParams> *dzdcReal = global.OrbitDzdcReal.get();
    HpSharkFloat<SharkFloatParams> *dzdcImag = global.OrbitDzdcImag.get();
    HpSharkFloat<SharkFloatParams> *newDzdcReal = global.OrbitNewDzdcReal.get();
    HpSharkFloat<SharkFloatParams> *newDzdcImag = global.OrbitNewDzdcImag.get();
    HpSharkFloat<SharkFloatParams> *one = global.OrbitOne.get();
    SetZero(zReal);
    SetZero(zImag);
    typename SharkFloatParams::Float localD2Real{};
    typename SharkFloatParams::Float localD2Imag{};

    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        SetZero(dzdcReal);
        SetZero(dzdcImag);
        one->template FromHDRFloat<typename SharkFloatParams::SubType>(
            HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});
        PrintHpValue("derivative one", *one);
    }

    for (uint64_t i = 0; i < period; ++i) {
        if (IsDebugTraceEnabled())
            std::cout << "EvaluateOrbitAndDerivative2 iteration " << i << " begin\n";
        PrintHpValue("derivative zReal", *zReal);
        PrintHpValue("derivative zImag", *zImag);
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

            PrintHpValue("derivative dzdcReal", *dzdcReal);
            PrintHpValue("derivative dzdcImag", *dzdcImag);
            PrintHdrValue("d2 zr", zr);
            PrintHdrValue("d2 zi", zi);
            PrintHdrValue("d2 dzr", dzr);
            PrintHdrValue("d2 dzi", dzi);
            PrintHdrValue("d2 dz2r", dz2r);
            PrintHdrValue("d2 dz2i", dz2i);
            PrintHdrValue("d2 zd2r", zd2r);
            PrintHdrValue("d2 zd2i", zd2i);
            PrintHdrValue("d2 sumr", sumr);
            PrintHdrValue("d2 sumi", sumi);
            PrintHdrValue("d2 localD2Real", localD2Real);
            PrintHdrValue("d2 localD2Imag", localD2Imag);
        }

        FusedReferenceOrbitStep<SharkFloatParams>(*zReal,
                                                  *zImag,
                                                  dzdcReal,
                                                  dzdcImag,
                                                  *cReal,
                                                  *cImag,
                                                  one,
                                                  newZReal,
                                                  newZImag,
                                                  newDzdcReal,
                                                  newDzdcImag,
                                                  workspace,
                                                  debugHostCombo);

        *zReal = *newZReal;
        *zImag = *newZImag;
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            *dzdcReal = *newDzdcReal;
            *dzdcImag = *newDzdcImag;
            PrintHpValue("derivative next dzdcReal", *dzdcReal);
            PrintHpValue("derivative next dzdcImag", *dzdcImag);
        }
        PrintHpValue("derivative next zReal", *zReal);
        PrintHpValue("derivative next zImag", *zImag);
    }

    *outZReal = *zReal;
    *outZImag = *zImag;
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        *outDzdcReal = *dzdcReal;
        *outDzdcImag = *dzdcImag;
        *outD2Real = localD2Real;
        *outD2Imag = localD2Imag;
        PrintHpValue("derivative final zReal", *outZReal);
        PrintHpValue("derivative final zImag", *outZImag);
        PrintHpValue("derivative final dzdcReal", *outDzdcReal);
        PrintHpValue("derivative final dzdcImag", *outDzdcImag);
        PrintHdrValue("derivative final d2Real", *outD2Real);
        PrintHdrValue("derivative final d2Imag", *outD2Imag);
    } else {
        SetZero(outDzdcReal);
        SetZero(outDzdcImag);
        *outD2Real = {};
        *outD2Imag = {};
    }
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
    EnsureGlobalFusedWorkspace<SharkFloatParams>();
    FusedWorkspace workspace = GetGlobalFusedWorkspace<SharkFloatParams>();
    GetGlobalFusedWorkspaceStorage<SharkFloatParams>().CachedN = 0;
    EvaluateOrbitAndDerivative2Impl<SharkFloatParams>(cReal,
                                                      cImag,
                                                      period,
                                                      outZReal,
                                                      outZImag,
                                                      outDzdcReal,
                                                      outDzdcImag,
                                                      outD2Real,
                                                      outD2Imag,
                                                      workspace,
                                                      debugHostCombo);
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
