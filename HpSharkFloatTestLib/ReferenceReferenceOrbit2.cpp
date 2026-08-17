#include "ReferenceReferenceOrbit2.h"

#include "DbgHeap.h"
#include "DebugChecksumHost.h"
#include "HDRFloat.h"
#include "HpSharkFloat.h"
#include "KernelInvokeReference2Setup.h"
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

template <class SharkFloatParams> struct FusedPlanCacheTraits {
    static constexpr uint32_t MinFusedN =
        SharkNTT::NextPow2U32(static_cast<uint32_t>(SharkFloatParams::NTTPlan2.L));
    static constexpr uint32_t MinFusedStages = SharkNTT::CeilLog2U32(MinFusedN);
    static constexpr uint32_t EntryCount = MaxFusedStages - MinFusedStages + 1u;

    static_assert(MinFusedN >= 2u && (MinFusedN & (MinFusedN - 1u)) == 0u);
    static_assert(EntryCount <= 32u);
};

struct FusedWorkspace {
    uint64_t *ZReal;
    uint64_t *ZImag;
    uint64_t *DzdcReal;
    uint64_t *DzdcImag;
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
    SharkNTT::PlanPrime *Plans;
    SharkNTT::RootTables *PlanRoots;
    uint32_t ActiveMinFusedN;
    uint32_t ActiveMaxFusedN;
    uint32_t ActiveMinFusedStages;
    uint32_t ActiveMaxFusedStages;
    uint32_t ActiveMaxFusedLimbs;
};

template <class SharkFloatParams> struct GlobalFusedWorkspaceStorage {
    using Cache = FusedPlanCacheTraits<SharkFloatParams>;

    std::unique_ptr<uint64_t[]> ZReal;
    std::unique_ptr<uint64_t[]> ZImag;
    std::unique_ptr<uint64_t[]> DzdcReal;
    std::unique_ptr<uint64_t[]> DzdcImag;
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
    SharkNTT::PlanPrime Plans[Cache::EntryCount]{};
    SharkNTT::RootTables PlanRoots[Cache::EntryCount]{};
    uint32_t ActiveMinFusedN = Cache::MinFusedN;
    uint32_t ActiveMaxFusedN = MaxFusedN;
    uint32_t ActiveMinFusedStages = Cache::MinFusedStages;
    uint32_t ActiveMaxFusedStages = MaxFusedStages;
    uint32_t ActiveMaxFusedLimbs = MaxFusedLimbs;
    uint64_t LoadedPreparedTablesId = 0;
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
    using Cache = FusedPlanCacheTraits<SharkFloatParams>;
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    if (!global.ZReal) {
        global.ZReal = std::make_unique<uint64_t[]>(MaxFusedN);
        global.ZImag = std::make_unique<uint64_t[]>(MaxFusedN);
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
        global.ForwardTwiddles = std::make_unique<uint64_t[]>(MaxFusedN);
        global.InverseTwiddles = std::make_unique<uint64_t[]>(MaxFusedN);
        for (uint32_t slot = 0; slot < Cache::EntryCount; ++slot) {
            const uint32_t stages = Cache::MinFusedStages + slot;
            const uint32_t n = 1u << stages;
            global.Plans[slot] = {SharkFloatParams::NTTPlan2.n32,
                                  SharkFloatParams::NTTPlan2.b,
                                  SharkFloatParams::NTTPlan2.L,
                                  static_cast<int>(n),
                                  static_cast<int>(stages),
                                  SharkFloatParams::NTTPlan2.ok};
            global.PlanRoots[slot] = {static_cast<int32_t>(stages),
                                      global.StageOmegas.get(),
                                      global.StageOmegasInverse.get(),
                                      static_cast<int32_t>(n),
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      0,
                                      0,
                                      global.ForwardTwiddles.get(),
                                      global.InverseTwiddles.get(),
                                      n - 1u};
        }
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
            global.DzdcReal.get(),
            global.DzdcImag.get(),
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
            global.Plans,
            global.PlanRoots,
            global.ActiveMinFusedN,
            global.ActiveMaxFusedN,
            global.ActiveMinFusedStages,
            global.ActiveMaxFusedStages,
            global.ActiveMaxFusedLimbs};
}

template <class SharkFloatParams>
static void
LoadPreparedTables(const HpShark::Reference2PreparedTables<SharkFloatParams> &preparedTables)
{
    using Cache = FusedPlanCacheTraits<SharkFloatParams>;
    using DeviceWorkspace = HpSharkReference2Workspace<SharkFloatParams>;
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();
    if (global.LoadedPreparedTablesId == preparedTables.GetId())
        return;

    DeviceWorkspace deviceWorkspace{};
    HpShark::Reference2SetupDetail::CheckCuda(cudaMemcpy(&deviceWorkspace,
                                                         preparedTables.GetDeviceDescriptor(),
                                                         sizeof(deviceWorkspace),
                                                         cudaMemcpyDeviceToHost),
                                              "cudaMemcpy(Reference2 prepared descriptor D2H)");
    const uint32_t firstSlot = deviceWorkspace.ActiveMinFusedStages - Cache::MinFusedStages;
    const uint32_t activePlanMask = deviceWorkspace.ActivePlanCacheEntryCount == 32u
                                        ? ~0u
                                        : (1u << deviceWorkspace.ActivePlanCacheEntryCount) - 1u;
    const uint32_t fullPlanMask = activePlanMask << firstSlot;
    if (deviceWorkspace.ValidPlanMask != fullPlanMask)
        throw FractalSharkSeriousException("Reference2 prepared tables are incomplete");

    const auto copy = [](void *destination, const void *source, size_t bytes, const char *operation) {
        HpShark::Reference2SetupDetail::CheckCuda(
            cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), operation);
    };
    copy(global.StageOmegas.get(),
         deviceWorkspace.StageOmegas,
         deviceWorkspace.ActiveMaxFusedStages * sizeof(uint64_t),
         "cudaMemcpy(Reference2 stage omegas D2H)");
    copy(global.StageOmegasInverse.get(),
         deviceWorkspace.StageOmegasInverse,
         deviceWorkspace.ActiveMaxFusedStages * sizeof(uint64_t),
         "cudaMemcpy(Reference2 inverse stage omegas D2H)");
    copy(global.ForwardTwiddles.get(),
         deviceWorkspace.ForwardTwiddles,
         deviceWorkspace.ActiveMaxFusedN * sizeof(uint64_t),
         "cudaMemcpy(Reference2 forward twiddles D2H)");
    copy(global.InverseTwiddles.get(),
         deviceWorkspace.InverseTwiddles,
         deviceWorkspace.ActiveMaxFusedN * sizeof(uint64_t),
         "cudaMemcpy(Reference2 inverse twiddles D2H)");
    for (uint32_t index = 0; index < deviceWorkspace.ActivePlanCacheEntryCount; ++index) {
        const uint32_t slot = firstSlot + index;
        global.Plans[slot] = deviceWorkspace.Plans[slot];
        global.PlanRoots[slot].Ninv = deviceWorkspace.PlanRoots[slot].Ninv;
    }
    global.ActiveMinFusedN = deviceWorkspace.ActiveMinFusedN;
    global.ActiveMaxFusedN = deviceWorkspace.ActiveMaxFusedN;
    global.ActiveMinFusedStages = deviceWorkspace.ActiveMinFusedStages;
    global.ActiveMaxFusedStages = deviceWorkspace.ActiveMaxFusedStages;
    global.ActiveMaxFusedLimbs = deviceWorkspace.ActiveMaxFusedLimbs;
    global.LoadedPreparedTablesId = preparedTables.GetId();
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
                int32_t exponentOffset,
                uint32_t ignoredPrecisionBits)
{
    if (IsZero(a) || IsZero(b)) {
        return {true, false, 0, TermKind::Product, aId, bId};
    }

    const int64_t exponent = static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(b.Exponent) +
                             static_cast<int64_t>(exponentOffset) +
                             2ll * static_cast<int64_t>(ignoredPrecisionBits);
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
MakeLinearTerm(const HpSharkFloat<SharkFloatParams> &a,
               SpectrumId aId,
               bool negate,
               uint32_t ignoredPrecisionBits)
{
    if (IsZero(a)) {
        return {true, false, 0, TermKind::Linear, aId, aId};
    }

    const int64_t exponent =
        static_cast<int64_t>(a.Exponent) + static_cast<int64_t>(ignoredPrecisionBits);
    assert(exponent >= INT32_MIN && exponent <= INT32_MAX);
    return {false,
            static_cast<bool>(a.GetNegative() ^ negate),
            static_cast<int32_t>(exponent),
            TermKind::Linear,
            aId,
            aId};
}

template <class SharkFloatParams>
static bool
ResolveAlignedValueExponent(int32_t &commonExponent,
                            const HpSharkFloat<SharkFloatParams> &value0,
                            const HpSharkFloat<SharkFloatParams> &value1)
{
    const bool value0Zero = IsZero(value0);
    const bool value1Zero = IsZero(value1);
    if (value0Zero && value1Zero) {
        commonExponent = 0;
        return true;
    }
    if (value0Zero) {
        commonExponent = value1.Exponent;
        return false;
    }
    if (value1Zero) {
        commonExponent = value0.Exponent;
        return false;
    }
    commonExponent = std::min(value0.Exponent, value1.Exponent);
    return false;
}

template <class SharkFloatParams>
static FusedTerm<SharkFloatParams>
MakeAlignedProductTerm(bool isZero, int32_t exponent, SpectrumId aId, SpectrumId bId)
{
    return {isZero, false, isZero ? 0 : exponent, TermKind::Product, aId, bId};
}

static uint64_t
AlignedCoefficientShift(int32_t valueExponent, int32_t commonExponent, uint32_t bitsPerCoefficient)
{
    const int64_t shiftBits = static_cast<int64_t>(valueExponent) - commonExponent;
    return shiftBits <= 0 ? 0ull : static_cast<uint64_t>(shiftBits) / bitsPerCoefficient;
}

template <class SharkFloatParams>
static uint32_t
LinearLimbCount()
{
    return (SharkFloatParams::GlobalNumUint32 * 32u + 31u) / 32u + 2u;
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

template <class SharkFloatParams>
static uint64_t
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

template <class SharkFloatParams, bool inverse, bool forwardDIF = false>
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

    if constexpr (forwardDIF) {
        static_assert(!inverse);
        for (uint32_t s = stages; s >= 1; --s) {
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
                    const uint64_t sum = AddP(u, v);
                    const uint64_t difference = SubP(u, v);
                    const uint64_t product = SharkNTT::MontgomeryMul(debugCombo, difference, w);
                    a[i0] = sum;
                    a[i1] = product;
                    if (IsDebugTraceEnabled()) {
                        std::cout << "  forward DIF butterfly"
                                  << " stage=" << s << " k=" << k << " j=" << j << " i0=" << i0
                                  << " i1=" << i1;
                        PrintHexValue("omega", wM);
                        PrintHexValue("u", u);
                        PrintHexValue("v", v);
                        PrintHexValue("twiddle", w);
                        PrintHexValue("sum", sum);
                        PrintHexValue("difference", difference);
                        PrintHexValue("product", product);
                        std::cout << '\n';
                    }
                }
            }
            if (IsDebugTraceEnabled()) {
                std::cout << "  forward DIF stage " << s << " output\n";
                PrintArray("spectrum", a, N);
            }
            if (s == 1)
                break;
        }
        return;
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
PackForward(DebugHostCombo<SharkFloatParams> &debugCombo,
            const HpSharkFloat<SharkFloatParams> &x,
            const SharkNTT::PlanPrime &plan,
            SharkNTT::RootTables &roots,
            uint64_t *out,
            uint32_t capacity,
            uint32_t inputBitOffset,
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
        const uint64_t coeff =
            i < static_cast<uint32_t>(plan.L)
                ? ReadBitsSimple(
                      x, static_cast<int64_t>(inputBitOffset) + static_cast<int64_t>(i) * plan.b, plan.b)
                : 0;
        const uint64_t coeffMont = SharkNTT::ToMontgomery(debugCombo, coeff % SharkNTT::MagicPrime);
        out[i] = coeffMont;
        if (i < static_cast<uint32_t>(plan.L) && IsDebugTraceEnabled()) {
            std::cout << "  pack coefficient index=" << i;
            PrintHexValue("coefficient", coeff);
            PrintHexValue("coefficientMont", coeffMont);
            PrintHexValue("packed", out[i]);
            std::cout << '\n';
        }
    }

    PrintArray("naturally packed spectrum", out, activeN);
    StoreReference2DebugState(debugCombo, packedPurpose, out, activeN);
    NTTRadix2<SharkFloatParams, false, true>(
        debugCombo, out, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);
    StoreReference2DebugState(debugCombo, forwardPurpose, out, activeN);
}

template <class SharkFloatParams>
static void
PackAlignedForward(DebugHostCombo<SharkFloatParams> &debugCombo,
                   const HpSharkFloat<SharkFloatParams> &value,
                   const SharkNTT::PlanPrime &plan,
                   SharkNTT::RootTables &roots,
                   uint64_t *out,
                   uint32_t capacity,
                   uint32_t inputBitOffset,
                   uint32_t coefficientShift,
                   uint32_t residualBitShift,
                   bool negative,
                   DebugStatePurpose packedPurpose,
                   DebugStatePurpose forwardPurpose)
{
    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugCombo, 0);
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= capacity);
    assert(zeroMont == 0);
    const uint32_t inputCoefficientCount =
        static_cast<uint32_t>(plan.L) + (residualBitShift != 0u ? 1u : 0u);

    for (uint32_t i = 0; i < activeN; ++i) {
        uint64_t packed = 0;
        const bool hasCoefficient =
            i >= coefficientShift && i - coefficientShift < inputCoefficientCount;
        if (hasCoefficient) {
            const uint32_t inputIndex = i - coefficientShift;
            const int64_t sourceBit = static_cast<int64_t>(inputBitOffset) +
                                      static_cast<int64_t>(inputIndex) * static_cast<int64_t>(plan.b) -
                                      static_cast<int64_t>(residualBitShift);
            const uint64_t coefficient =
                ReadAlignedBits(value, inputBitOffset, sourceBit, static_cast<int>(plan.b));
            const uint64_t coefficientMont =
                SharkNTT::ToMontgomery(debugCombo, coefficient % SharkNTT::MagicPrime);
            packed = coefficientMont;
            if (negative && coefficient != 0)
                packed = SubP(zeroMont, packed);
            if (IsDebugTraceEnabled()) {
                std::cout << "  aligned pack coefficient index=" << i;
                PrintHexValue("coefficient", coefficient);
                PrintHexValue("coefficientMont", coefficientMont);
                PrintHexValue("packed", packed);
                std::cout << '\n';
            }
        }
        out[i] = packed;
    }

    PrintArray("aligned naturally packed spectrum", out, activeN);
    StoreReference2DebugState(debugCombo, packedPurpose, out, activeN);
    NTTRadix2<SharkFloatParams, false, true>(
        debugCombo, out, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);
    StoreReference2DebugState(debugCombo, forwardPurpose, out, activeN);
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
        const uint64_t phaseIndex = ((chunkShift % activeN) * static_cast<uint64_t>(i)) % activeN;
        const uint64_t chunkScale = (chunkShift == 0) ? SharkNTT::ToMontgomery<SharkFloatParams>(1)
                                                      : roots.omega_pows[phaseIndex];
        const uint64_t scale = SharkNTT::MontgomeryMul(debugCombo, chunkScale, bitScale);
        const uint64_t shifted = SharkNTT::MontgomeryMul(debugCombo, source[i], scale);
        const uint32_t reverseIndex = ReverseBits32(i, static_cast<int>(plan.stages));
        dest[reverseIndex] = negative ? SubP(zeroMont, shifted) : shifted;
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
        const uint64_t phaseIndex = ((chunkShift % activeN) * static_cast<uint64_t>(i)) % activeN;
        const uint64_t chunkScale = (chunkShift == 0) ? SharkNTT::ToMontgomery<SharkFloatParams>(1)
                                                      : roots.omega_pows[phaseIndex];
        const uint64_t scale = SharkNTT::MontgomeryMul(debugCombo, chunkScale, bitScale);
        const uint64_t shifted = SharkNTT::MontgomeryMul(debugCombo, source[i], scale);
        const uint32_t reverseIndex = ReverseBits32(i, static_cast<int>(plan.stages));

        if (negative)
            dest[reverseIndex] = SubP(dest[reverseIndex], shifted);
        else
            dest[reverseIndex] = AddP(dest[reverseIndex], shifted);
        if (IsDebugTraceEnabled()) {
            std::cout << "  shifted spectrum index=" << i;
            PrintHexValue("source", source[i]);
            PrintHexValue("phaseIndex", phaseIndex);
            PrintHexValue("chunkScale", chunkScale);
            PrintHexValue("scale", scale);
            PrintHexValue("shifted", shifted);
            PrintHexValue("dest", dest[reverseIndex]);
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
        case SpectrumId::CImag:
        case SpectrumId::One:
            assert(false);
            return workspace.ZReal;
        case SpectrumId::DzdcReal:
            return workspace.DzdcReal;
        case SpectrumId::DzdcImag:
            return workspace.DzdcImag;
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
static int64_t
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

template <class SharkFloatParams>
static void
UnpackAlignedResiduesToSignedLimbs(DebugHostCombo<SharkFloatParams> &debugCombo,
                                   const uint64_t *normalResidues,
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
    assert(coefficientCount <= static_cast<uint32_t>(plan.N));
    const uint64_t halfPrime = (SharkNTT::MagicPrime - 1ull) >> 1;
    for (uint32_t j = 0; j < limbCount; ++j) {
        const uint64_t firstBit = j >= 3 ? static_cast<uint64_t>(j - 3) * 32ull : 0ull;
        const uint64_t lastBit = (static_cast<uint64_t>(j) + 1ull) * 32ull - 1ull;
        const uint64_t firstCoefficient =
            firstBit > productBitOffset ? (firstBit - productBitOffset) / plan.b : 0ull;
        const uint64_t lastCoefficient =
            lastBit >= productBitOffset ? (lastBit - productBitOffset) / plan.b : 0ull;
        int64_t total = 0;

        if (firstBit >= productBitOffset || productBitOffset <= lastBit) {
            for (uint64_t i = firstCoefficient; i <= lastCoefficient && i < coefficientCount; ++i) {
                const uint64_t residue =
                    SharkNTT::MontgomeryMul(debugCombo, normalResidues[i], roots.Ninv);
                if (residue == 0)
                    continue;

                const bool negative = residue > halfPrime;
                const uint64_t magnitude = negative ? SharkNTT::MagicPrime - residue : residue;
                const uint64_t shiftedBits = productBitOffset + i * static_cast<uint64_t>(plan.b);
                const uint32_t q = static_cast<uint32_t>(shiftedBits >> 5);
                if (q > j || j - q > 3)
                    continue;

                const uint32_t bitShift = static_cast<uint32_t>(shiftedBits & 31ull);
                const uint64_t low = bitShift == 0 ? magnitude : magnitude << bitShift;
                const uint64_t high = bitShift == 0 ? 0ull : magnitude >> (64u - bitShift);
                uint32_t contribution = 0u;
                switch (j - q) {
                    case 0:
                        contribution = static_cast<uint32_t>(low);
                        break;
                    case 1:
                        contribution = static_cast<uint32_t>(low >> 32);
                        break;
                    case 2:
                        contribution = static_cast<uint32_t>(high);
                        break;
                    case 3:
                        contribution = static_cast<uint32_t>(high >> 32);
                        break;
                }
                total +=
                    negative ? -static_cast<int64_t>(contribution) : static_cast<int64_t>(contribution);
            }
        }
        total += SignedLinearLimbContribution(linearValue, linearInputBitOffset, linearBitOffset, j);
        limbs[j] = total;
    }
    PrintArray("aligned signed limbs", limbs, limbCount);
}

template <class SharkFloatParams>
static void
GatherLinearToSignedLimbs(const HpSharkFloat<SharkFloatParams> *linearValue,
                          uint32_t linearInputBitOffset,
                          int64_t *limbs,
                          uint32_t limbCount)
{
    for (uint32_t j = 0; j < limbCount; ++j)
        limbs[j] = SignedLinearLimbContribution(linearValue, linearInputBitOffset, 0, j);
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
    PrintArray("inverse input bit-reversed spectrum", spectrum, activeN);
    NTTRadix2<SharkFloatParams, true>(
        debugCombo, spectrum, static_cast<uint32_t>(plan.N), static_cast<uint32_t>(plan.stages), roots);

    for (uint32_t i = 0; i < activeN; ++i) {
        spectrum[i] = SharkNTT::MontgomeryMul(debugCombo, spectrum[i], roots.Ninv);
        if (IsDebugTraceEnabled()) {
            std::cout << "  inverse normalize index=" << i;
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
InverseAlignedSpectrumToSignedLimbs(DebugHostCombo<SharkFloatParams> &debugCombo,
                                    const SharkNTT::PlanPrime &plan,
                                    SharkNTT::RootTables &roots,
                                    uint64_t *spectrum,
                                    uint32_t coefficientCount,
                                    uint64_t productBitOffset,
                                    const HpSharkFloat<SharkFloatParams> *linearValue,
                                    uint32_t linearInputBitOffset,
                                    uint64_t linearBitOffset,
                                    int64_t *limbs,
                                    uint32_t limbCount,
                                    DebugStatePurpose residuesPurpose,
                                    DebugStatePurpose limbsPurpose)
{
    const uint32_t activeN = static_cast<uint32_t>(plan.N);
    assert(activeN <= MaxFusedN);
    NTTRadix2<SharkFloatParams, true>(
        debugCombo, spectrum, activeN, static_cast<uint32_t>(plan.stages), roots);

    if (residuesPurpose != DebugStatePurpose::Invalid)
        StoreReference2DebugState(debugCombo, residuesPurpose, spectrum, activeN);
    UnpackAlignedResiduesToSignedLimbs(debugCombo,
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
                     uint32_t inputBitOffset,
                     FusedWorkspace &workspace)
{
    PackForward(debugHostCombo,
                zReal,
                plan,
                roots,
                workspace.ZReal,
                MaxFusedN,
                inputBitOffset,
                DebugStatePurpose::Z0XX,
                DebugStatePurpose::Z2XX);
    PackForward(debugHostCombo,
                zImag,
                plan,
                roots,
                workspace.ZImag,
                MaxFusedN,
                inputBitOffset,
                DebugStatePurpose::Z0YY,
                DebugStatePurpose::Z2YY);
}

template <class SharkFloatParams>
static void
PrepareDerivativeSpectra(DebugHostCombo<SharkFloatParams> &debugHostCombo,
                         const SharkNTT::PlanPrime &plan,
                         SharkNTT::RootTables &roots,
                         const HpSharkFloat<SharkFloatParams> &dzdcReal,
                         const HpSharkFloat<SharkFloatParams> &dzdcImag,
                         uint32_t inputBitOffset,
                         FusedWorkspace &workspace)
{
    PackForward(debugHostCombo,
                dzdcReal,
                plan,
                roots,
                workspace.DzdcReal,
                MaxFusedN,
                inputBitOffset,
                DebugStatePurpose::Z0W1,
                DebugStatePurpose::Z2W1);
    PackForward(debugHostCombo,
                dzdcImag,
                plan,
                roots,
                workspace.DzdcImag,
                MaxFusedN,
                inputBitOffset,
                DebugStatePurpose::Z0W2,
                DebugStatePurpose::Z2W2);
}

template <class SharkFloatParams, class... Terms>
static uint64_t
RequiredCoefficientsForStream(int32_t commonExp,
                              const SharkNTT::PlanPrime &plan,
                              const FusedTerm<SharkFloatParams> &first,
                              const Terms &...terms)
{
    assert(plan.b > 0 && plan.L > 0);
    uint64_t requiredCoefficients = 0;
    const auto includeTerm = [&](const FusedTerm<SharkFloatParams> &term) {
        if (term.IsZero)
            return;
        const int64_t signedShift = static_cast<int64_t>(term.Exponent) - commonExp;
        assert(signedShift >= 0);
        // The active aligned path below derives support from each packed operand's last coefficient,
        // including the possible high digit introduced by a residual bit shift.
        const uint64_t coefficientShift =
            static_cast<uint64_t>(signedShift) / static_cast<uint64_t>(plan.b);
        const uint64_t inputCoefficients = static_cast<uint64_t>(plan.L);
        const uint64_t termCoefficients =
            term.Kind == TermKind::Product ? 2ull * inputCoefficients - 1ull : inputCoefficients;
        requiredCoefficients = std::max(requiredCoefficients, coefficientShift + termCoefficients);
    };
    includeTerm(first);
    (includeTerm(terms), ...);
    return requiredCoefficients;
}

#if 0
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
                        uint64_t iteration,
                        uint32_t &previousActiveN,
                        uint32_t actualPrecisionLimbs,
                        FusedWorkspace &workspace,
                        DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    constexpr uint32_t storagePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    assert(actualPrecisionLimbs > storagePrecisionLimbs / 2u);
    assert(actualPrecisionLimbs <= storagePrecisionLimbs);
    const uint32_t ignoredPrecisionBits = (storagePrecisionLimbs - actualPrecisionLimbs) * 32u;
    const SharkNTT::PlanPrime basePlan =
        SharkNTT::BuildPlanPrime2(static_cast<int>(actualPrecisionLimbs));
    assert(basePlan.ok);
    assert(basePlan.b > 0);
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
    const FusedTerm<SharkFloatParams> realZ2 = MakeProductTerm(
        zReal, SpectrumId::ZReal, zReal, SpectrumId::ZReal, false, 0, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> realNegY2 = MakeProductTerm(
        zImag, SpectrumId::ZImag, zImag, SpectrumId::ZImag, true, 0, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> realC =
        MakeLinearTerm(cReal, SpectrumId::CReal, false, ignoredPrecisionBits);

    const FusedTerm<SharkFloatParams> imagTwoZY = MakeProductTerm(
        zReal, SpectrumId::ZReal, zImag, SpectrumId::ZImag, false, 1, ignoredPrecisionBits);
    const FusedTerm<SharkFloatParams> imagC =
        MakeLinearTerm(cImag, SpectrumId::CImag, false, ignoredPrecisionBits);

    int32_t realCommonExp = 0;
    int32_t imagCommonExp = 0;
    const bool realIsZero = ResolveCommonExponent(realCommonExp, realZ2, realNegY2, realC);
    const bool imagIsZero = ResolveCommonExponent(imagCommonExp, imagTwoZY, imagC);

    uint64_t maxRequiredCoefficients =
        std::max(RequiredCoefficientsForStream(realCommonExp, basePlan, realZ2, realNegY2, realC),
                 RequiredCoefficientsForStream(imagCommonExp, basePlan, imagTwoZY, imagC));

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
        dzdcRealW0 = MakeProductTerm(
            zReal, SpectrumId::ZReal, *dzdcReal, SpectrumId::DzdcReal, false, 1, ignoredPrecisionBits);
        dzdcRealNegW1 = MakeProductTerm(
            zImag, SpectrumId::ZImag, *dzdcImag, SpectrumId::DzdcImag, true, 1, ignoredPrecisionBits);
        dzdcRealOne = MakeLinearTerm(*one, SpectrumId::One, false, ignoredPrecisionBits);
        dzdcImagW2 = MakeProductTerm(
            zImag, SpectrumId::ZImag, *dzdcReal, SpectrumId::DzdcReal, false, 1, ignoredPrecisionBits);
        dzdcImagW3 = MakeProductTerm(
            zReal, SpectrumId::ZReal, *dzdcImag, SpectrumId::DzdcImag, false, 1, ignoredPrecisionBits);
        dzdcRealIsZero =
            ResolveCommonExponent(dzdcRealCommonExp, dzdcRealW0, dzdcRealNegW1, dzdcRealOne);
        dzdcImagIsZero = ResolveCommonExponent(dzdcImagCommonExp, dzdcImagW2, dzdcImagW3);
        maxRequiredCoefficients =
            std::max(maxRequiredCoefficients,
                     RequiredCoefficientsForStream(
                         dzdcRealCommonExp, basePlan, dzdcRealW0, dzdcRealNegW1, dzdcRealOne));
        maxRequiredCoefficients =
            std::max(maxRequiredCoefficients,
                     RequiredCoefficientsForStream(dzdcImagCommonExp, basePlan, dzdcImagW2, dzdcImagW3));
    }

    if (IsDebugTraceEnabled()) {
        std::cout << "maxRequiredCoefficients=" << maxRequiredCoefficients
                  << " realCommonExp=" << realCommonExp << " imagCommonExp=" << imagCommonExp
                  << std::endl;
    }

    if (maxRequiredCoefficients == 0) {
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

    const uint64_t requiredN = CeilPowerOfTwo(maxRequiredCoefficients);
    if (requiredN > workspace.ActiveMaxFusedN) {
        std::cerr << "ReferenceOrbit2 fused workspace exceeded: requestedCoefficients="
                  << maxRequiredCoefficients << " requiredN=" << requiredN
                  << " capacity=" << workspace.ActiveMaxFusedN << '\n';
        assert(false);
    }
    using Cache = FusedPlanCacheTraits<SharkFloatParams>;
    const uint32_t activeN = requiredN < workspace.ActiveMinFusedN ? workspace.ActiveMinFusedN
                                                                   : static_cast<uint32_t>(requiredN);
    assert(activeN >= workspace.ActiveMinFusedN);
    assert(maxRequiredCoefficients <= activeN);
    assert(activeN <= workspace.ActiveMaxFusedN);
    assert(activeN >= 2u);
    assert((SharkNTT::PHI % activeN) == 0ull);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Cache::MinFusedStages;
    assert(planSlot < Cache::EntryCount);
    if (activeN != previousActiveN) {
        // if (IsDebugTraceEnabled()) {
        std::cout << "ReferenceOrbit2 plan changed: iteration=" << iteration
                  << " previousN=" << previousActiveN << " activeN=" << activeN
                  << " planSlot=" << planSlot << '\n';
        //}
        previousActiveN = activeN; // Set a breakpoint here to observe every plan transition.
    }
    SharkNTT::PlanPrime plan = workspace.Plans[planSlot];
    plan.n32 = basePlan.n32;
    plan.L = basePlan.L;
    SharkNTT::RootTables &roots = workspace.PlanRoots[planSlot];
    const HpSharkReference2ConstantSpectra constantSpectra = workspace.ConstantSpectra[planSlot];
    workspace.CReal = constantSpectra.CReal;
    workspace.CImag = constantSpectra.CImag;
    workspace.One = constantSpectra.One;
    const uint32_t coefficientCount = activeN;
    const uint32_t limbCount = (coefficientCount * static_cast<uint32_t>(plan.b) + 31u) / 32u + 2u;
    assert(limbCount <= workspace.ActiveMaxFusedLimbs);

    if constexpr (HpShark::DebugChecksums) {
        // The GPU reuses the cached final constant spectra instead of repacking constants each
        // iteration. Keep both legacy constant checksum slots aligned with that cached data.
        StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z0XY, workspace.CReal, activeN);
        StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z2XY, workspace.CReal, activeN);
        StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z0W0, workspace.CImag, activeN);
        StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z2W0, workspace.CImag, activeN);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z0W3, workspace.One, activeN);
            StoreReference2DebugState(debugHostCombo, DebugStatePurpose::Z2W3, workspace.One, activeN);
        }
    }

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

    if (IsDebugTraceEnabled()) {
        PrintArray("roots.stage_omegas", roots.stage_omegas, roots.stages);
        PrintArray("roots.stage_omegas_inv", roots.stage_omegas_inv, roots.stages);
        PrintArray("roots.omega_pows", roots.omega_pows, roots.N);
        PrintArray("roots.stage_twiddles_fwd", roots.stage_twiddles_fwd, roots.total_twiddles);
        PrintArray("roots.stage_twiddles_inv", roots.stage_twiddles_inv, roots.total_twiddles);
        PrintHexValue("roots.Ninvm_mont", roots.Ninvm_mont);
        PrintHexValue("roots.Ninv", roots.Ninv);
        std::cout << '\n';
    }

    PrepareNormalSpectra(debugHostCombo, plan, roots, zReal, zImag, ignoredPrecisionBits, workspace);

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
                             workspace.ActiveMaxFusedLimbs,
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
                             workspace.ActiveMaxFusedLimbs,
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

        PrepareDerivativeSpectra(
            debugHostCombo, plan, roots, *dzdcReal, *dzdcImag, ignoredPrecisionBits, workspace);
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
                                 workspace.ActiveMaxFusedLimbs,
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
                                 workspace.ActiveMaxFusedLimbs,
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

#endif

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
                        uint64_t iteration,
                        uint32_t &previousActiveN,
                        uint32_t actualPrecisionLimbs,
                        FusedWorkspace &workspace,
                        DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    constexpr uint32_t storagePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    assert(actualPrecisionLimbs > storagePrecisionLimbs / 2u);
    assert(actualPrecisionLimbs <= storagePrecisionLimbs);
    const uint32_t ignoredPrecisionBits = (storagePrecisionLimbs - actualPrecisionLimbs) * 32u;
    const SharkNTT::PlanPrime basePlan =
        SharkNTT::BuildPlanPrime2(static_cast<int>(actualPrecisionLimbs));
    assert(basePlan.ok);
    assert(basePlan.b > 0);
    const uint32_t bitsPerCoefficient = static_cast<uint32_t>(basePlan.b);

    int32_t stateCommonExponent = 0;
    const bool stateBothZero = ResolveAlignedValueExponent(stateCommonExponent, zReal, zImag);
    const bool stateRealZero = IsZero(zReal);
    const bool stateImagZero = IsZero(zImag);
    const int64_t stateProductExponent64 = static_cast<int64_t>(stateCommonExponent) * 2ll +
                                           2ll * static_cast<int64_t>(ignoredPrecisionBits);
    assert(stateProductExponent64 >= INT32_MIN && stateProductExponent64 <= INT32_MAX);
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
    ResolveCommonExponent(realExponent, realProductTerm, realConstantTerm);
    ResolveCommonExponent(imagExponent, imagProductTerm, imagConstantTerm);

    int32_t derivativeCommonExponent = 0;
    int32_t derivativeProductExponent = 0;
    int32_t dzdcRealExponent = 0;
    int32_t dzdcImagExponent = 0;
    bool derivativeBothZero = true;
    bool derivativeRealZero = true;
    bool derivativeImagZero = true;
    FusedTerm<SharkFloatParams> dzdcP1Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZReal, SpectrumId::DzdcReal);
    FusedTerm<SharkFloatParams> dzdcP2Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZImag, SpectrumId::DzdcImag);
    FusedTerm<SharkFloatParams> dzdcP3Term =
        MakeAlignedProductTerm<SharkFloatParams>(true, 0, SpectrumId::ZReal, SpectrumId::DzdcReal);
    FusedTerm<SharkFloatParams> dzdcOneTerm{};
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        assert(dzdcReal != nullptr);
        assert(dzdcImag != nullptr);
        assert(one != nullptr);
        derivativeBothZero = ResolveAlignedValueExponent(derivativeCommonExponent, *dzdcReal, *dzdcImag);
        derivativeRealZero = IsZero(*dzdcReal);
        derivativeImagZero = IsZero(*dzdcImag);
        const int64_t derivativeProductExponent64 = static_cast<int64_t>(stateCommonExponent) +
                                                    static_cast<int64_t>(derivativeCommonExponent) +
                                                    2ll * static_cast<int64_t>(ignoredPrecisionBits);
        assert(derivativeProductExponent64 >= INT32_MIN && derivativeProductExponent64 <= INT32_MAX);
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
        dzdcOneTerm = MakeLinearTerm(*one, SpectrumId::One, false, ignoredPrecisionBits);
        ResolveCommonExponent(dzdcRealExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term, dzdcOneTerm);
        ResolveCommonExponent(dzdcImagExponent, dzdcP1Term, dzdcP2Term, dzdcP3Term);
    }

    const uint64_t stateRealShiftBits =
        stateRealZero ? 0ull : static_cast<uint64_t>(zReal.Exponent - stateCommonExponent);
    const uint64_t stateImagShiftBits =
        stateImagZero ? 0ull : static_cast<uint64_t>(zImag.Exponent - stateCommonExponent);
    const uint64_t stateRealCoefficientShift =
        AlignedCoefficientShift(zReal.Exponent, stateCommonExponent, bitsPerCoefficient);
    const uint64_t stateImagCoefficientShift =
        AlignedCoefficientShift(zImag.Exponent, stateCommonExponent, bitsPerCoefficient);
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
    const uint64_t stateMaxLastCoefficient =
        std::max(stateRealLastCoefficient, stateImagLastCoefficient);
    const uint64_t realRequiredCoefficients =
        stateBothZero ? 0ull : 2ull * stateMaxLastCoefficient + 1ull;
    const uint64_t imagRequiredCoefficients =
        (stateRealZero || stateImagZero) ? 0ull
                                         : stateRealLastCoefficient + stateImagLastCoefficient + 1ull;
    uint64_t requiredCoefficients = std::max(realRequiredCoefficients, imagRequiredCoefficients);

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
            derivativeRealZero ? 0ull
                               : static_cast<uint64_t>(dzdcReal->Exponent - derivativeCommonExponent);
        derivativeImagShiftBits =
            derivativeImagZero ? 0ull
                               : static_cast<uint64_t>(dzdcImag->Exponent - derivativeCommonExponent);
        derivativeRealCoefficientShift =
            AlignedCoefficientShift(dzdcReal->Exponent, derivativeCommonExponent, bitsPerCoefficient);
        derivativeImagCoefficientShift =
            AlignedCoefficientShift(dzdcImag->Exponent, derivativeCommonExponent, bitsPerCoefficient);
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
        derivativeMaxLastCoefficient =
            std::max(derivativeRealLastCoefficient, derivativeImagLastCoefficient);
        derivativeP1RequiredCoefficients =
            dzdcP1Term.IsZero ? 0ull : stateRealLastCoefficient + derivativeRealLastCoefficient + 1ull;
        derivativeP2RequiredCoefficients =
            dzdcP2Term.IsZero ? 0ull : stateImagLastCoefficient + derivativeImagLastCoefficient + 1ull;
        derivativeP3RequiredCoefficients =
            dzdcP3Term.IsZero ? 0ull : stateMaxLastCoefficient + derivativeMaxLastCoefficient + 1ull;
        requiredCoefficients = std::max(requiredCoefficients, derivativeP1RequiredCoefficients);
        requiredCoefficients = std::max(requiredCoefficients, derivativeP2RequiredCoefficients);
        requiredCoefficients = std::max(requiredCoefficients, derivativeP3RequiredCoefficients);
    }

    const bool hasLinearTerm = !realConstantTerm.IsZero || !imagConstantTerm.IsZero ||
                               (SharkFloatParams::EnableNewtonRaphson && !dzdcOneTerm.IsZero);
    if (requiredCoefficients == 0) {
        if (!hasLinearTerm) {
            SetZero(outReal);
            SetZero(outImag);
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                SetZero(outDzdcReal);
                SetZero(outDzdcImag);
            }
        } else {
            const uint32_t limbCount = LinearLimbCount<SharkFloatParams>();
            GatherLinearToSignedLimbs(realConstantTerm.IsZero ? nullptr : &cReal,
                                      ignoredPrecisionBits,
                                      workspace.RealLimbs,
                                      limbCount);
            GatherLinearToSignedLimbs(imagConstantTerm.IsZero ? nullptr : &cImag,
                                      ignoredPrecisionBits,
                                      workspace.ImagLimbs,
                                      limbCount);
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                GatherLinearToSignedLimbs(dzdcOneTerm.IsZero ? nullptr : one,
                                          ignoredPrecisionBits,
                                          workspace.DzdcRealLimbs,
                                          limbCount);
                GatherLinearToSignedLimbs<SharkFloatParams>(
                    nullptr, ignoredPrecisionBits, workspace.DzdcImagLimbs, limbCount);
            }
            FinalizeSignedStream({workspace.RealLimbs, limbCount, realExponent, outReal},
                                 workspace.MagnitudeDigits,
                                 workspace.Magnitude,
                                 workspace.ActiveMaxFusedLimbs,
                                 debugHostCombo,
                                 DebugStatePurpose::SignedCarry1,
                                 DebugStatePurpose::FinalAdd1);
            FinalizeSignedStream({workspace.ImagLimbs, limbCount, imagExponent, outImag},
                                 workspace.MagnitudeDigits,
                                 workspace.Magnitude,
                                 workspace.ActiveMaxFusedLimbs,
                                 debugHostCombo,
                                 DebugStatePurpose::SignedCarry2,
                                 DebugStatePurpose::FinalAdd2);
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                FinalizeSignedStream({workspace.DzdcRealLimbs, limbCount, dzdcRealExponent, outDzdcReal},
                                     workspace.MagnitudeDigits,
                                     workspace.Magnitude,
                                     workspace.ActiveMaxFusedLimbs,
                                     debugHostCombo,
                                     DebugStatePurpose::SignedCarryDzdc1,
                                     DebugStatePurpose::FinalAddDzdc1);
                FinalizeSignedStream({workspace.DzdcImagLimbs, limbCount, dzdcImagExponent, outDzdcImag},
                                     workspace.MagnitudeDigits,
                                     workspace.Magnitude,
                                     workspace.ActiveMaxFusedLimbs,
                                     debugHostCombo,
                                     DebugStatePurpose::SignedCarryDzdc2,
                                     DebugStatePurpose::FinalAddDzdc2);
            }
        }
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add1, *outReal);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add2, *outImag);
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc1, *outDzdcReal);
            StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc2, *outDzdcImag);
        }
        return;
    }

    const uint64_t requiredN = CeilPowerOfTwo(requiredCoefficients);
    if (requiredN > workspace.ActiveMaxFusedN) {
        std::cerr << "ReferenceOrbit2 fused workspace exceeded: requestedCoefficients="
                  << requiredCoefficients << " requiredN=" << requiredN
                  << " capacity=" << workspace.ActiveMaxFusedN << '\n';
        assert(false);
    }
    using Cache = FusedPlanCacheTraits<SharkFloatParams>;
    const uint32_t activeN = requiredN < workspace.ActiveMinFusedN ? workspace.ActiveMinFusedN
                                                                   : static_cast<uint32_t>(requiredN);
    assert(activeN >= workspace.ActiveMinFusedN);
    assert(requiredCoefficients <= activeN);
    assert(activeN <= workspace.ActiveMaxFusedN);
    assert(activeN >= 2u);
    assert((SharkNTT::PHI % activeN) == 0ull);
    const uint32_t planSlot = CountTrailingZeros(activeN) - Cache::MinFusedStages;
    assert(planSlot < Cache::EntryCount);
    if (activeN != previousActiveN) {
        if (IsDebugTraceEnabled())
            std::cout << "ReferenceOrbit2 plan changed: iteration=" << iteration
                      << " previousN=" << previousActiveN << " activeN=" << activeN
                      << " planSlot=" << planSlot << '\n';
        previousActiveN = activeN;
    }

    SharkNTT::PlanPrime plan = workspace.Plans[planSlot];
    plan.n32 = basePlan.n32;
    plan.L = basePlan.L;
    SharkNTT::RootTables &roots = workspace.PlanRoots[planSlot];
    const uint64_t realProductBitOffset =
        realProductTerm.IsZero ? 0ull : static_cast<uint64_t>(realProductTerm.Exponent - realExponent);
    const uint64_t imagProductBitOffset =
        imagProductTerm.IsZero ? 0ull : static_cast<uint64_t>(imagProductTerm.Exponent - imagExponent);
    const uint64_t derivativeProductBitOffset =
        (dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero)
            ? 0ull
            : static_cast<uint64_t>(derivativeProductExponent - dzdcRealExponent);
    const uint64_t realLinearBitOffset =
        realConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(realConstantTerm.Exponent - realExponent);
    const uint64_t imagLinearBitOffset =
        imagConstantTerm.IsZero ? 0ull : static_cast<uint64_t>(imagConstantTerm.Exponent - imagExponent);
    const uint64_t derivativeLinearBitOffset =
        dzdcOneTerm.IsZero ? 0ull : static_cast<uint64_t>(dzdcOneTerm.Exponent - dzdcRealExponent);
    const uint64_t linearBits =
        static_cast<uint64_t>(storagePrecisionLimbs) * 32ull - ignoredPrecisionBits;

    uint64_t outputBits = realProductBitOffset + realRequiredCoefficients * bitsPerCoefficient;
    outputBits =
        std::max(outputBits, realLinearBitOffset + (realConstantTerm.IsZero ? uint64_t{0} : linearBits));
    outputBits =
        std::max(outputBits, imagProductBitOffset + imagRequiredCoefficients * bitsPerCoefficient);
    outputBits =
        std::max(outputBits, imagLinearBitOffset + (imagConstantTerm.IsZero ? uint64_t{0} : linearBits));
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const uint64_t derivativeRequiredCoefficients =
            std::max(derivativeP1RequiredCoefficients,
                     std::max(derivativeP2RequiredCoefficients, derivativeP3RequiredCoefficients));
        outputBits =
            std::max(outputBits,
                     derivativeProductBitOffset + derivativeRequiredCoefficients * bitsPerCoefficient);
        outputBits = std::max(
            outputBits, derivativeLinearBitOffset + (dzdcOneTerm.IsZero ? uint64_t{0} : linearBits));
    }
    const uint64_t limbCount64 = (outputBits + 31ull) / 32ull + 2ull;
    assert(limbCount64 <= workspace.ActiveMaxFusedLimbs);
    const uint32_t limbCount = static_cast<uint32_t>(limbCount64);

    const uint32_t zRealResidualBitShift = static_cast<uint32_t>(stateRealResidualBitShift);
    const uint32_t zImagResidualBitShift = static_cast<uint32_t>(stateImagResidualBitShift);
    PackAlignedForward(debugHostCombo,
                       zReal,
                       plan,
                       roots,
                       workspace.ZReal,
                       MaxFusedN,
                       ignoredPrecisionBits,
                       static_cast<uint32_t>(stateRealCoefficientShift),
                       zRealResidualBitShift,
                       zReal.GetNegative(),
                       DebugStatePurpose::Z0XX,
                       DebugStatePurpose::Z2XX);
    PackAlignedForward(debugHostCombo,
                       zImag,
                       plan,
                       roots,
                       workspace.ZImag,
                       MaxFusedN,
                       ignoredPrecisionBits,
                       static_cast<uint32_t>(stateImagCoefficientShift),
                       zImagResidualBitShift,
                       zImag.GetNegative(),
                       DebugStatePurpose::Z0YY,
                       DebugStatePurpose::Z2YY);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const uint32_t dzdcRealResidualBitShift = static_cast<uint32_t>(derivativeRealResidualBitShift);
        const uint32_t dzdcImagResidualBitShift = static_cast<uint32_t>(derivativeImagResidualBitShift);
        PackAlignedForward(debugHostCombo,
                           *dzdcReal,
                           plan,
                           roots,
                           workspace.DzdcReal,
                           MaxFusedN,
                           ignoredPrecisionBits,
                           static_cast<uint32_t>(derivativeRealCoefficientShift),
                           dzdcRealResidualBitShift,
                           dzdcReal->GetNegative(),
                           DebugStatePurpose::Z0W1,
                           DebugStatePurpose::Z2W1);
        PackAlignedForward(debugHostCombo,
                           *dzdcImag,
                           plan,
                           roots,
                           workspace.DzdcImag,
                           MaxFusedN,
                           ignoredPrecisionBits,
                           static_cast<uint32_t>(derivativeImagCoefficientShift),
                           dzdcImagResidualBitShift,
                           dzdcImag->GetNegative(),
                           DebugStatePurpose::Z0W2,
                           DebugStatePurpose::Z2W2);
    }

    const uint64_t zeroMont = SharkNTT::ToMontgomery(debugHostCombo, 0);
    for (uint32_t i = 0; i < activeN; ++i) {
        uint64_t real = zeroMont;
        uint64_t imag = zeroMont;
        if (!realProductTerm.IsZero) {
            const uint64_t sum = AddP(workspace.ZReal[i], workspace.ZImag[i]);
            const uint64_t difference = SubP(workspace.ZReal[i], workspace.ZImag[i]);
            real = SharkNTT::MontgomeryMul(debugHostCombo, sum, difference);
        }
        if (!imagProductTerm.IsZero) {
            const uint64_t product =
                SharkNTT::MontgomeryMul(debugHostCombo, workspace.ZReal[i], workspace.ZImag[i]);
            imag = AddP(product, product);
        }
        workspace.RealOutput[i] = real;
        workspace.ImagOutput[i] = imag;

        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            uint64_t derivativeReal = zeroMont;
            uint64_t derivativeImag = zeroMont;
            if (!dzdcP1Term.IsZero || !dzdcP2Term.IsZero || !dzdcP3Term.IsZero) {
                const uint64_t p1 =
                    SharkNTT::MontgomeryMul(debugHostCombo, workspace.ZReal[i], workspace.DzdcReal[i]);
                const uint64_t p2 =
                    SharkNTT::MontgomeryMul(debugHostCombo, workspace.ZImag[i], workspace.DzdcImag[i]);
                const uint64_t stateSum = AddP(workspace.ZReal[i], workspace.ZImag[i]);
                const uint64_t derivativeSum = AddP(workspace.DzdcReal[i], workspace.DzdcImag[i]);
                const uint64_t p3 = SharkNTT::MontgomeryMul(debugHostCombo, stateSum, derivativeSum);
                const uint64_t realDifference = SubP(p1, p2);
                const uint64_t imagDifference = SubP(SubP(p3, p1), p2);
                derivativeReal = AddP(realDifference, realDifference);
                derivativeImag = AddP(imagDifference, imagDifference);
            }
            workspace.DzdcRealOutput[i] = derivativeReal;
            workspace.DzdcImagOutput[i] = derivativeImag;
        }
    }
    StoreReference2DebugState(
        debugHostCombo, DebugStatePurpose::Z2_Perm1, workspace.RealOutput, activeN);
    StoreReference2DebugState(
        debugHostCombo, DebugStatePurpose::Z2_Perm2, workspace.ImagOutput, activeN);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        StoreReference2DebugState(
            debugHostCombo, DebugStatePurpose::Z2_PermW0, workspace.DzdcRealOutput, activeN);
        StoreReference2DebugState(
            debugHostCombo, DebugStatePurpose::Z2_PermW1, workspace.DzdcImagOutput, activeN);
    }

    const uint32_t realCoefficientCount = realProductTerm.IsZero ? 0u : activeN;
    const uint32_t imagCoefficientCount = imagProductTerm.IsZero ? 0u : activeN;
    InverseAlignedSpectrumToSignedLimbs(debugHostCombo,
                                        plan,
                                        roots,
                                        workspace.RealOutput,
                                        realCoefficientCount,
                                        realProductBitOffset,
                                        realConstantTerm.IsZero ? nullptr : &cReal,
                                        ignoredPrecisionBits,
                                        realLinearBitOffset,
                                        workspace.RealLimbs,
                                        limbCount,
                                        DebugStatePurpose::Invalid,
                                        DebugStatePurpose::UnpackXX);
    InverseAlignedSpectrumToSignedLimbs(debugHostCombo,
                                        plan,
                                        roots,
                                        workspace.ImagOutput,
                                        imagCoefficientCount,
                                        imagProductBitOffset,
                                        imagConstantTerm.IsZero ? nullptr : &cImag,
                                        ignoredPrecisionBits,
                                        imagLinearBitOffset,
                                        workspace.ImagLimbs,
                                        limbCount,
                                        DebugStatePurpose::Invalid,
                                        DebugStatePurpose::UnpackYY);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        const uint32_t derivativeCoefficientCount =
            dzdcP1Term.IsZero && dzdcP2Term.IsZero && dzdcP3Term.IsZero ? 0u : activeN;
        InverseAlignedSpectrumToSignedLimbs(debugHostCombo,
                                            plan,
                                            roots,
                                            workspace.DzdcRealOutput,
                                            derivativeCoefficientCount,
                                            derivativeProductBitOffset,
                                            dzdcOneTerm.IsZero ? nullptr : one,
                                            ignoredPrecisionBits,
                                            derivativeLinearBitOffset,
                                            workspace.DzdcRealLimbs,
                                            limbCount,
                                            DebugStatePurpose::Invalid,
                                            DebugStatePurpose::UnpackW0);
        InverseAlignedSpectrumToSignedLimbs<SharkFloatParams>(debugHostCombo,
                                                              plan,
                                                              roots,
                                                              workspace.DzdcImagOutput,
                                                              derivativeCoefficientCount,
                                                              derivativeProductBitOffset,
                                                              nullptr,
                                                              ignoredPrecisionBits,
                                                              0,
                                                              workspace.DzdcImagLimbs,
                                                              limbCount,
                                                              DebugStatePurpose::Invalid,
                                                              DebugStatePurpose::UnpackW1);
    }

    FinalizeSignedStream({workspace.RealLimbs, limbCount, realExponent, outReal},
                         workspace.MagnitudeDigits,
                         workspace.Magnitude,
                         workspace.ActiveMaxFusedLimbs,
                         debugHostCombo,
                         DebugStatePurpose::SignedCarry1,
                         DebugStatePurpose::FinalAdd1);
    FinalizeSignedStream({workspace.ImagLimbs, limbCount, imagExponent, outImag},
                         workspace.MagnitudeDigits,
                         workspace.Magnitude,
                         workspace.ActiveMaxFusedLimbs,
                         debugHostCombo,
                         DebugStatePurpose::SignedCarry2,
                         DebugStatePurpose::FinalAdd2);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add1, *outReal);
    StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_Add2, *outImag);
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        FinalizeSignedStream({workspace.DzdcRealLimbs, limbCount, dzdcRealExponent, outDzdcReal},
                             workspace.MagnitudeDigits,
                             workspace.Magnitude,
                             workspace.ActiveMaxFusedLimbs,
                             debugHostCombo,
                             DebugStatePurpose::SignedCarryDzdc1,
                             DebugStatePurpose::FinalAddDzdc1);
        FinalizeSignedStream({workspace.DzdcImagLimbs, limbCount, dzdcImagExponent, outDzdcImag},
                             workspace.MagnitudeDigits,
                             workspace.Magnitude,
                             workspace.ActiveMaxFusedLimbs,
                             debugHostCombo,
                             DebugStatePurpose::SignedCarryDzdc2,
                             DebugStatePurpose::FinalAddDzdc2);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc1, *outDzdcReal);
        StoreReference2DebugValue(debugHostCombo, DebugStatePurpose::Result_AddDzdc2, *outDzdcImag);
    }
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
                                            uint32_t actualPrecisionLimbs,
                                            FusedWorkspace &workspace,
                                            DebugHostCombo<SharkFloatParams> &debugHostCombo);

template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>
ReferenceOrbit2Helper(const HpSharkFloat<SharkFloatParams> *cReal,
                      const HpSharkFloat<SharkFloatParams> *cImag,
                      const typename SharkFloatParams::Float &radiusY,
                      uint64_t maxIters,
                      uint32_t actualPrecisionLimbs,
                      DebugHostCombo<SharkFloatParams> &debugHostCombo,
                      HpShark::Reference2PreparedTables<SharkFloatParams> *preparedTables)
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
    std::unique_ptr<HpShark::Reference2PreparedTables<SharkFloatParams>> localPreparedTables;
    if (preparedTables == nullptr) {
        localPreparedTables = HpShark::PrepareHpSharkReference2Tables<SharkFloatParams>(
            HpShark::LaunchParams{0, 0}, *cReal, *cImag, actualPrecisionLimbs);
        preparedTables = localPreparedTables.get();
    }
    LoadPreparedTables(*preparedTables);
    FusedWorkspace workspace = GetGlobalFusedWorkspace<SharkFloatParams>();
    auto &global = GetGlobalFusedWorkspaceStorage<SharkFloatParams>();

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
                                                          actualPrecisionLimbs,
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

    uint32_t previousActiveN = 0;
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
                                                  i,
                                                  previousActiveN,
                                                  actualPrecisionLimbs,
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
                                uint32_t actualPrecisionLimbs,
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

    uint32_t previousActiveN = 0;
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
                                                  i,
                                                  previousActiveN,
                                                  actualPrecisionLimbs,
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
                            uint32_t actualPrecisionLimbs,
                            DebugHostCombo<SharkFloatParams> &debugHostCombo,
                            HpShark::Reference2PreparedTables<SharkFloatParams> *preparedTables)
{
    EnsureGlobalFusedWorkspace<SharkFloatParams>();
    std::unique_ptr<HpShark::Reference2PreparedTables<SharkFloatParams>> localPreparedTables;
    if (preparedTables == nullptr) {
        localPreparedTables = HpShark::PrepareHpSharkReference2Tables<SharkFloatParams>(
            HpShark::LaunchParams{0, 0}, *cReal, *cImag, actualPrecisionLimbs);
        preparedTables = localPreparedTables.get();
    }
    LoadPreparedTables(*preparedTables);
    FusedWorkspace workspace = GetGlobalFusedWorkspace<SharkFloatParams>();
    EvaluateOrbitAndDerivative2Impl<SharkFloatParams>(cReal,
                                                      cImag,
                                                      period,
                                                      outZReal,
                                                      outZImag,
                                                      outDzdcReal,
                                                      outDzdcImag,
                                                      outD2Real,
                                                      outD2Imag,
                                                      actualPrecisionLimbs,
                                                      workspace,
                                                      debugHostCombo);
}

#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>                                    \
    ReferenceOrbit2Helper<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> *,                     \
                                            const HpSharkFloat<SharkFloatParams> *,                     \
                                            const typename SharkFloatParams::Float &,                   \
                                            uint64_t,                                                   \
                                            uint32_t,                                                   \
                                            DebugHostCombo<SharkFloatParams> &,                         \
                                            HpShark::Reference2PreparedTables<SharkFloatParams> *);

ExplicitInstantiateAll();

ExplicitlyInstantiate(SharkParams1);
ExplicitlyInstantiate(SharkParams2);
ExplicitlyInstantiate(SharkParams3);
ExplicitlyInstantiate(SharkParams4);
ExplicitlyInstantiate(SharkParams5);
ExplicitlyInstantiate(SharkParams6);
ExplicitlyInstantiate(SharkParams8);
ExplicitlyInstantiate(SharkParams10);
ExplicitlyInstantiate(SharkParams11);
ExplicitlyInstantiate(SharkParams12);

#define ExplicitlyInstantiateDerivative(SharkFloatParams)                                               \
    template void EvaluateOrbitAndDerivative2<SharkFloatParams>(                                        \
        const HpSharkFloat<SharkFloatParams> *,                                                         \
        const HpSharkFloat<SharkFloatParams> *,                                                         \
        uint64_t,                                                                                       \
        HpSharkFloat<SharkFloatParams> *,                                                               \
        HpSharkFloat<SharkFloatParams> *,                                                               \
        HpSharkFloat<SharkFloatParams> *,                                                               \
        HpSharkFloat<SharkFloatParams> *,                                                               \
        typename SharkFloatParams::Float *,                                                             \
        typename SharkFloatParams::Float *,                                                             \
        uint32_t,                                                                                       \
        DebugHostCombo<SharkFloatParams> &,                                                             \
        HpShark::Reference2PreparedTables<SharkFloatParams> *);

#undef ExplicitlyInstantiate
#define ExplicitlyInstantiate(SharkFloatParams) ExplicitlyInstantiateDerivative(SharkFloatParams)
ExplicitInstantiateAll();
