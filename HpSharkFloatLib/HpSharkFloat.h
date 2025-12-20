#pragma once

#include "CudaCrap.h"
#include "DebugStateRaw.h"
#include "GPU_ReferenceIter.h"
#include "HDRFloat.h"
#include "MultiplyNTTCudaSetup.h"
#include "MultiplyNTTPlanBuilder.h"

#include <bit>
#include <gmp.h>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif // __CUDACC__

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
// #define SharkRestrict

enum class BasicCorrectnessMode : int {
    Correctness_P1 = 0,      // Params1 only
    PerfSub = 1,             // Individual add/multiply kernels, not the full one
    PerfSweep = 2,           // Sweep blocks/threads
    PerfSingle = 3,          // Single block/thread config (DEFAULT)
    Correctness_P1_to_P5 = 4 // Params1..5
};

namespace HpShark {

#ifdef _DEBUG
static constexpr bool Debug = true;
#else
static constexpr bool Debug = false;
#endif

#ifdef _DEBUG
#define SharkForceInlineReleaseOnly
#else
#define SharkForceInlineReleaseOnly __forceinline__
#endif

static constexpr auto NTTBHint = 32;        // 26?
static constexpr auto NTTNumBitsMargin = 0; // 2?

// Uncomment to test small warp on multiply normalize path for easier debugging
// Assumes number of threads is a power of 2 and <= 32, with one block.
// #define TEST_SMALL_NORMALIZE_WARP

static constexpr bool TestGpu = true;
// static constexpr bool TestGpu = false;

static constexpr auto TestComicalThreadCount = 13;

// Set to true to use a custom stream for the kernel launch
static constexpr auto CustomStream = true;

enum class InnerLoopOption {
    BasicGlobal,
    BasicAllInShared,
    TryVectorLoads,
    TryUnalignedLoads,
    TryUnalignedLoads2,
    TryUnalignedLoads2Shared
};

static constexpr InnerLoopOption SharkInnerLoopOption = InnerLoopOption::TryUnalignedLoads2;

static constexpr auto LoadAllInShared =
    (SharkInnerLoopOption == InnerLoopOption::BasicAllInShared) ||
    (SharkInnerLoopOption == InnerLoopOption::TryUnalignedLoads2Shared);

// Set to true to use shared memory for the incoming numbers
static constexpr auto RegisterLimit = 127;

static constexpr auto ConstantSharedRequiredBytes = 0;

static constexpr bool DebugChecksums = Debug;
// static constexpr bool DebugChecksums = true;
static constexpr bool DebugGlobalState = false; // TODO: A bit broken right now.
static constexpr bool TestCorrectness = Debug;
static constexpr bool TestInfiniteCorrectness = true;
static constexpr auto TestForceSameSign = false;
static constexpr bool TestBenchmarkAgainstHost = false;
static constexpr bool TestInitCudaMemory = true;

// True to compare against the full host-side reference implementation, false is MPIR only
// False is useful to speed up e.g. testing many cases fast but gives poor diagnostic results.
static constexpr bool TestReferenceImpl = true;

constexpr uint32_t
ceil_pow2_u32(uint32_t v)
{
    // returns 1 for v==0; otherwise rounds up to the next power of two (ceil)
    if (v <= 1)
        return 1u;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
constexpr bool
is_pow2_u32(uint32_t v)
{
    return v && ((v & (v - 1u)) == 0u);
}

struct LaunchParams {
    LaunchParams(int32_t numBlocksIn, int32_t threadsPerBlockIn)
        : NumBlocks{numBlocksIn}, ThreadsPerBlock{threadsPerBlockIn},
          TotalThreads{numBlocksIn * threadsPerBlockIn}
    {
    }

    LaunchParams()
        : NumBlocks{rand() % 128 + 1}, ThreadsPerBlock{32 * (rand() % 8 + 1)},
          TotalThreads{NumBlocks * ThreadsPerBlock}
    {
    }

    const int32_t NumBlocks;
    const int32_t ThreadsPerBlock;
    const int32_t TotalThreads;

    std::string
    ToString() const
    {
        return std::string("Blocks: ") + std::to_string(NumBlocks) + ", ThreadsPerBlock: " + std::to_string(ThreadsPerBlock) +
               ", TotalThreads: " + std::to_string(TotalThreads);
    }
};

template <int32_t pNumDigits, bool Periodicity> struct GenericSharkFloatParams {
    using Float = HDRFloat<float>; // TODO hardcoded
    using SubType = float;         // TODO hardcoded

    // TODO: this is fucked up, fix it so we don't round to next power of two
    static constexpr bool SharkUsePow2SizesOnly = false;

    // Fixed number of uint32_t values
    static constexpr int32_t Guard = 4;

    static constexpr int32_t GlobalNumUint32 =
        SharkUsePow2SizesOnly
            ? static_cast<int32_t>(HpShark::ceil_pow2_u32(static_cast<uint32_t>(pNumDigits)))
            : pNumDigits;

    constexpr static int32_t
    NumberOfBits(int32_t x)
    {
        return x < 2 ? x : 1 + NumberOfBits(x >> 1);
    }
    constexpr static auto LogNumUint32 = NumberOfBits(GlobalNumUint32);

    static constexpr int32_t HalfLimbsRoundedUp = (GlobalNumUint32 + 1) / 2;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    static constexpr auto EnablePeriodicity = Periodicity;
    static constexpr bool DisableAllAdditions = false;
    static constexpr bool DisableSubtraction = false;
    static constexpr bool DisableCarryPropagation = false;
    static constexpr bool DisableFinalConstruction = false;
    static constexpr bool ForceNoOp = false;

    static constexpr auto NumDebugStates = (3 * static_cast<int>(DebugStatePurpose::NumPurposes));

    static constexpr auto MaxBlocks = 256;
    static constexpr auto MaxThreads = 256;
    static constexpr auto NumDebugMultiplyCounts = MaxThreads * MaxBlocks;

    static std::string
    GetDescription()
    {
        return std::string("Number of digits: ") + std::to_string(GlobalNumUint32);
    }

    static constexpr SharkNTT::PlanPrime NTTPlan =
        SharkNTT::BuildPlanPrime(GlobalNumUint32, HpShark::NTTBHint, HpShark::NTTNumBitsMargin);

    using ReferenceIterT = GPUReferenceIter<Float, PerturbExtras::Disable>;
};

// This one should account for maximum call index, e.g. if we generate 500 calls
// recursively then we need this to be at 500.
static constexpr auto ScratchMemoryCopies = 256llu;

// Number of arrays of digits on each frame
static constexpr auto ScratchMemoryArraysForMultiply = 96;
static constexpr auto ScratchMemoryArraysForAdd = 64;

// Additional space per frame:
static constexpr auto AdditionalUInt64PerFrame = 256;

// Additional space up front, globally-shared:
// Units are uint64_t
static constexpr auto MaxBlocks = 256;

static constexpr auto AdditionalGlobalSyncSpace = 128 * (MaxBlocks + 1);
static constexpr auto AdditionalGlobalDebugPerThread = HpShark::DebugGlobalState ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = HpShark::DebugChecksums ? 1024 * 1024 : 0;

static constexpr auto AdditionalGlobalSyncSpaceOffset = 0;
static constexpr auto AdditionalMultipliesOffset =
    AdditionalGlobalSyncSpaceOffset + AdditionalGlobalSyncSpace;
static constexpr auto AdditionalChecksumsOffset =
    AdditionalMultipliesOffset + AdditionalGlobalDebugPerThread;

// Use the order of these three variables being added as the
// definition of how they are laid out in memory.
static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace + AdditionalGlobalDebugPerThread + AdditionalGlobalChecksumSpace;

template <class SharkFloatParams>
static constexpr auto
CalculateKaratsubaFrameSize()
{
    constexpr auto retval =
        ScratchMemoryArraysForMultiply * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}

template <class SharkFloatParams>
static constexpr auto
CalculateNTTFrameSize()
{
    constexpr auto retval =
        ScratchMemoryArraysForMultiply * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}

template <class SharkFloatParams>
constexpr int32_t
CalculateNTTSharedMemorySize()
{
    // HpShark::ConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes = 3 * 2048 * sizeof(uint64_t);
    return sharedAmountBytes;
}

template <class SharkFloatParams>
static constexpr auto
CalculateAddFrameSize()
{
    return ScratchMemoryArraysForAdd * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

static constexpr auto LowPrec = 32;
} // namespace HpShark

#include "ExplicitInstantiate.h"

// If you add a new one, search for one of the other types and copy/paste
// TODO Add tests fail with non-mult-32 sizes.
// using SharkParamsNP1 = HpShark::GenericSharkFloatParams<8>;
// using SharkParamsNP2 = HpShark::GenericSharkFloatParams<16>;
// using SharkParamsNP3 = HpShark::GenericSharkFloatParams<32>;
// using SharkParamsNP4 = HpShark::GenericSharkFloatParams<64>;
// using SharkParamsNP5 = HpShark::GenericSharkFloatParams<128>;
using SharkParamsNP1 = HpShark::GenericSharkFloatParams<64, false>;
using SharkParamsNP2 = HpShark::GenericSharkFloatParams<128, false>;
using SharkParamsNP3 = HpShark::GenericSharkFloatParams<256, false>;
using SharkParamsNP4 = HpShark::GenericSharkFloatParams<512, false>;
using SharkParamsNP5 = HpShark::GenericSharkFloatParams<1024, false>;

using SharkParamsNP6 = HpShark::GenericSharkFloatParams<8192, false>;
using SharkParamsNP7 = HpShark::GenericSharkFloatParams<16384, false>;
using SharkParamsNP8 = HpShark::GenericSharkFloatParams<32768, false>;
using SharkParamsNP9 = HpShark::GenericSharkFloatParams<65536, false>;
using SharkParamsNP10 = HpShark::GenericSharkFloatParams<131072, false>;
using SharkParamsNP11 = HpShark::GenericSharkFloatParams<262144, false>;
using SharkParamsNP12 = HpShark::GenericSharkFloatParams<524288, false>;

// Correctness test sizes
using TestCorrectnessSharkParams1 = SharkParamsNP1;
using TestCorrectnessSharkParams2 = SharkParamsNP2;
using TestCorrectnessSharkParams3 = SharkParamsNP4;
using TestCorrectnessSharkParams4 = SharkParamsNP3;
using TestCorrectnessSharkParams5 = SharkParamsNP5;

// FractalShark production sizes
using SharkParams1 = HpShark::GenericSharkFloatParams<256, true>;
using SharkParams2 = HpShark::GenericSharkFloatParams<512, true>;
using SharkParams3 = HpShark::GenericSharkFloatParams<1024, true>;
using SharkParams4 = HpShark::GenericSharkFloatParams<2048, true>;
using SharkParams5 = HpShark::GenericSharkFloatParams<4096, true>;
using SharkParams6 = HpShark::GenericSharkFloatParams<8192, true>;
using SharkParams7 = HpShark::GenericSharkFloatParams<16384, true>;
using SharkParams8 = HpShark::GenericSharkFloatParams<32768, true>;
using SharkParams9 = HpShark::GenericSharkFloatParams<65536, true>;
using SharkParams10 = HpShark::GenericSharkFloatParams<131072, true>;
using SharkParams11 = HpShark::GenericSharkFloatParams<262144, true>;
using SharkParams12 = HpShark::GenericSharkFloatParams<524288, true>;

enum class InjectNoiseInLowOrder { Disable, Enable };

// Struct to hold both integer and fractional parts of the high-precision number
template <class SharkFloatParams> struct HpSharkFloat {
    HpSharkFloat();
    // HpSharkFloat(uint32_t numDigits);
    HpSharkFloat(const uint32_t *digitsIn, int32_t expIn, bool isNegative);
    ~HpSharkFloat() = default;
    HpSharkFloat &operator=(const HpSharkFloat<SharkFloatParams> &);

    CUDA_CRAP void DeepCopySameDevice(const HpSharkFloat<SharkFloatParams> &other);

#if defined(__CUDA_ARCH__)
    CUDA_CRAP void
    DeepCopyGPU(cooperative_groups::grid_group &grid,
                cooperative_groups::thread_block &block,
                const HpSharkFloat<SharkFloatParams> &other)
    {
        // compute a global thread index
        int idx = block.group_index().x * block.dim_threads().x + block.thread_index().x;
        int stride = grid.dim_blocks().x * block.dim_threads().x;

        // copy digits in parallel
        for (int i = idx; i < SharkFloatParams::GlobalNumUint32; i += stride) {
            Digits[i] = other.Digits[i];
        }

        // let one thread handle the scalar fields
        if (idx == 0) {
            Exponent = other.Exponent;
            IsNegative = other.IsNegative;
        }
    }
#endif

    std::string ToString() const;
    std::string ToHexString() const;
    void GenerateRandomNumber();
    void GenerateRandomNumber2(bool clearLowOrder = false);
    void Negate();
    void Normalize();
    void DenormalizeLosePrecision();

    CUDA_CRAP_BOTH
    void SetNegative(bool isNegative);

    CUDA_CRAP_BOTH
    bool GetNegative() const;

    // Default precision in bits
    using DigitType = uint32_t;
    constexpr static auto NumUint32 = SharkFloatParams::GlobalNumUint32;
    constexpr static auto DefaultPrecBits = NumUint32 * sizeof(DigitType) * 8;
    constexpr static auto ConvertBitsToDecimals = 3.3219280948873623478703194294894;
    constexpr static auto DefaultPrecDigits = DefaultPrecBits / ConvertBitsToDecimals;
    constexpr static auto DefaultMpirBits = DefaultPrecBits;

    void HpGpuToMpf(mpf_t &mpf_val) const;
    void MpfToHpGpu(const mpf_t mpf_val, int prec_bits, InjectNoiseInLowOrder injectNoise);

    // Convert an HpSharkFloat<Params> to HDRFloat<SubType> (SubType = double or float).
    // - Single pass: scans from high→low once to detect zero and find MS non-zero limb.
    // - Uses top two 32-bit limbs to build a 64-bit window for accurate mantissa.
    // - Exponent is unbiased binary exponent: value = mantissa * 2^exp, mantissa ∈ [1,2) or [-2,-1].
    // - 'extraExp' lets callers add any external binary scaling they track separately.
    template <class SubType> CUDA_CRAP_BOTH HDRFloat<SubType> ToHDRFloat(int32_t extraExp = 0) const;

    template <class SubType> CUDA_CRAP_BOTH void FromHDRFloat(const HDRFloat<SubType> &h);

    // Digits in base 2^32
    DigitType Digits[NumUint32];

    // Exponent in base 2
    using ExpT = int32_t;
    ExpT Exponent;

private:
    // Sign
    bool IsNegative;

    mp_exp_t HpGpuExponentToMpfExponent(size_t numBytesToCopy) const;
};

template <class SharkFloatParams>
CUDA_CRAP_BOTH void
HpSharkFloat<SharkFloatParams>::SetNegative(bool isNegative)
{
    IsNegative = isNegative;
}

template <class SharkFloatParams>
CUDA_CRAP_BOTH bool
HpSharkFloat<SharkFloatParams>::GetNegative() const
{
    return IsNegative;
}

template <class SharkFloatParams>
template <class SubType>
CUDA_CRAP_BOTH HDRFloat<SubType>
HpSharkFloat<SharkFloatParams>::ToHDRFloat(int32_t extraExp) const
{
    static_assert(std::is_same<SubType, float>::value || std::is_same<SubType, double>::value,
                  "ToHDRFloat: SubType must be float or double");

    // CLZ helpers (host/device friendly)
    auto clz32 = [](uint32_t v) -> int {
#if defined(__CUDA_ARCH__)
        return __clz(v);
#else
        // std::countl_zero is constexpr in C++20 and returns 32 for x==0
        return static_cast<int>(std::countl_zero(v));
#endif
    };

    // ------------- Single reverse scan to find the highest non-zero limb -------------
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    int hiIdx = -1;
    for (int i = N - 1; i >= 0; --i) {
        if (Digits[i] != 0u) {
            hiIdx = i;
            break;
        }
    }

    // Zero fast-path
    if (hiIdx < 0) {
        return HDRFloat<SubType>(SubType(0));
    }

    // Form a 64-bit "window" from the top two 32-bit limbs.
    // Layout: [hi32 | lo32] -> bits 63..32 are hi32; bits 31..0 are lo32.
    const uint32_t hi32 = Digits[hiIdx];
    const uint32_t lo32 = (hiIdx > 0) ? Digits[hiIdx - 1] : 0u;
    const uint64_t window64 = (uint64_t(hi32) << 32) | uint64_t(lo32);

    // Absolute (unbiased) binary exponent p = floor(log2(|this|))
    // msb in the hi32 limb:
    const int msb32 = 31 - clz32(hi32);   // 0..31
    const int32_t p = hiIdx * 32 + msb32; // absolute MSB index in |this|

    // Within the 64-bit window, the MSB is at 32 + msb32 (since hi32 is in the top half).
    const int msbInWindow = 32 + msb32; // 32..63

    // Normalize mantissa into [1,2): m = window64 / 2^(msbInWindow)
    SubType mant = SubType(window64) / std::ldexp(SubType(1), msbInWindow);
    if (IsNegative)
        mant = -mant;

    // Combine with any external binary scaling tracked by the caller/type.
    const int32_t finalExp = p + extraExp + Exponent;

    // Build & normalize HDRFloat
    HDRFloat<SubType> out(finalExp, mant);
    HdrReduce(out); // ensure mantissa is in canonical range and exponent adjusted
    return out;
}

// Build an HpSharkFloat<Params> from HDRFloat<SubType> (SubType = double or float).
// Strategy:
//   1) Reduce HDR so mantissa ∈ [1,2); get its (unbiased) exp.
//   2) Put the MSB at the very top of the highest limb: hiIdx = N-1, msb32 = 31.
//   3) Fill the top TWO limbs with a 64-bit "window" equal to mantissa * 2^63.
//   4) Choose HpSharkFloat::Exponent so the overall value matches HDR.
//
// Representation assumed:
//   Value = (-1)^sign * (sum_i Digits[i] * 2^(32*i)) * 2^(Exponent).
//   Digits[0] is least-significant; Digits[N-1] is most-significant.
//   Assumes incoming value is reduced/normalized
template <class SharkFloatParams>
template <class SubType>
void
HpSharkFloat<SharkFloatParams>::FromHDRFloat(const HDRFloat<SubType> &h)
{
    static_assert(std::is_same<SubType, double>::value || std::is_same<SubType, float>::value,
                  "FromHDRFloat: SubType must be double or float");

    constexpr int N = HpSharkFloat<SharkFloatParams>::NumUint32;

    // Zero digits
    for (int i = 0; i < N; ++i)
        Digits[i] = HpSharkFloat<SharkFloatParams>::DigitType{0};

    // Zero fast-path
    const bool isZero = (h.mantissa == SubType(0));
    if (isZero) {
        SetNegative(false);
        Exponent = 0;
        return;
    }

    // Sign
    const bool neg = (h.mantissa < SubType(0));
    const double mantAbs = static_cast<double>(neg ? -h.mantissa : h.mantissa);
    const int32_t exp = static_cast<int32_t>(h.exp);

    // We place the absolute MSB at the very top bit of the highest 32-bit limb.
    // That bit index (absolute, counting from bit 0 of Digits[0]) is:
    //   msb_abs = (N-1)*32 + 31
    // Inside the 64-bit window that spans the top TWO limbs, that MSB is at bit 63.
    const int32_t msb_abs = (static_cast<int32_t>(N) - 1) * 32 + 31;

    // Build the 64-bit window = mantAbs * 2^63   (so the MSB of the window is bit 63).
    // Clamp to [2^63, 2^64-1] to avoid rounding overflow on (almost) 2.0 mantissas.
    double scaled = std::ldexp(mantAbs, 63);
    if (scaled < std::ldexp((double)1.0, 63)) {
        scaled = std::ldexp((double)1.0, 63);
    }
    const double max64 = std::ldexp((double)1.0, 64) - 1.0;
    if (scaled > max64)
        scaled = max64;

    const uint64_t window64 = static_cast<uint64_t>(scaled);

    // Split into top two limbs (be careful if N==1).
    const uint32_t hi32 = static_cast<uint32_t>(window64 >> 32);
    const uint32_t lo32 = static_cast<uint32_t>(window64 & 0xFFFF'FFFFu);

    // Write high-order digits
    Digits[N - 1] = static_cast<HpSharkFloat<SharkFloatParams>::DigitType>(hi32);
    if constexpr (N >= 2) {
        Digits[N - 2] = static_cast<HpSharkFloat<SharkFloatParams>::DigitType>(lo32);
    }

    // Choose Hp exponent so overall value equals mantissa * 2^exp:
    //   Value = (DigitsAsInt) * 2^(Exponent)
    // where DigitsAsInt has MSB at msb_abs.
    // Since HDR value's msb is at absolute bit index 'exp', set:
    Exponent = exp - msb_abs;

    SetNegative(neg);
}

enum class PeriodicityResult { Unknown, Continue, PeriodFound, Escaped };

#if !defined(__CUDA_ARCH__)
[[maybe_unused]] static std::string
PeriodicityStrResult(PeriodicityResult periodicityStatus)
{
    switch (periodicityStatus) {
        case PeriodicityResult::Continue:
            return "Continue";
        case PeriodicityResult::PeriodFound:
            return "PeriodFound";
        case PeriodicityResult::Escaped:
            return "Escaped";
        default:
            return "Unknown";
    }
}
#endif // !__CUDA_ARCH__

#pragma warning(push)
#pragma warning(disable : 4324)
template <class SharkFloatParams> struct alignas(16) HpSharkComboResults {
    SharkNTT::RootTables Roots;
    alignas(16) HpSharkFloat<SharkFloatParams> A;
    alignas(16) HpSharkFloat<SharkFloatParams> B;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultX2;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2XY;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultY2;
};

template <class SharkFloatParams> struct HpSharkAddComboResults {
    alignas(16) HpSharkFloat<SharkFloatParams> A_X2;
    alignas(16) HpSharkFloat<SharkFloatParams> B_Y2;
    alignas(16) HpSharkFloat<SharkFloatParams> C_A;
    alignas(16) HpSharkFloat<SharkFloatParams> D_2X;
    alignas(16) HpSharkFloat<SharkFloatParams> E_B;
    alignas(16) HpSharkFloat<SharkFloatParams> Result1_A_B_C;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2_D_E;
};

template <class SharkFloatParams> struct HpSharkReferenceResults {

    alignas(16) typename SharkFloatParams::Float RadiusY;
    alignas(16) HpSharkComboResults<SharkFloatParams> Multiply;
    alignas(16) HpSharkAddComboResults<SharkFloatParams> Add;
    alignas(16) PeriodicityResult PeriodicityStatus;
    alignas(16) typename SharkFloatParams::Float dzdcX;
    alignas(16) typename SharkFloatParams::Float dzdcY;
    alignas(16) uint64_t OutputIterCount;
    alignas(16) uint64_t MaxRuntimeIters;

    static constexpr auto MaxOutputIters = 1024;
    alignas(16) typename SharkFloatParams::ReferenceIterT OutputIters[MaxOutputIters];

    // Host only
    alignas(16) HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    alignas(16) uint64_t *d_tempProducts;
    alignas(16) uintptr_t stream; // cudaStream_t
    alignas(16) void *kernelArgs[3];
};
#pragma warning(pop)

template <class SharkFloatParams> std::string MpfToString(const mpf_t mpf_val, size_t precInBits);

template <class SharkFloatParams>
std::string Uint32ToMpf(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);

template <class IntT> std::string UintArrayToHexString(const IntT *array, size_t numElements);

template <class IntT> std::string UintToHexString(IntT val);

template <class IntT> std::string VectorUintToHexString(const std::vector<IntT> &arr);

template <class IntT> std::string VectorUintToHexString(const IntT *arr, size_t numElements);
