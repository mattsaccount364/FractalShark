#pragma once

#include "CudaCrap.h"
#include "DebugStateRaw.h"
#include "MultiplyNTTCudaSetup.h"
#include "MultiplyNTTPlanBuilder.cuh"
#include "HDRFloat.h"
#include "GPU_ReferenceIter.h"

#include <string>
#include <gmp.h>
#include <vector>
#include <bit>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#endif // __CUDACC__

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
//#define SharkRestrict

// Suggested combinations.
// For testing, define:
//  - ENABLE_ADD_KERNEL, ENABLE_MULTIPLY_NTT_KERNEL, or ENABLE_REFERENCE_KERNEL
//  - _DEBUG
//  - Ensure ENABLE_BASIC_CORRECTNESS == 0
// For profiling, define:
//  - ENABLE_ADD_KERNEL, ENABLE_MULTIPLY_NTT_KERNEL, or ENABLE_REFERENCE_KERNEL
//  - Release
//  - Ensure ENABLE_BASIC_CORRECTNESS == 2
// For E2E test:
//  - ENABLE_FULL_KERNEL
//  - Debug or release
//  - Ensure ENABLE_BASIC_CORRECTNESS == 2
//      When it runs, it'll take a minute, verify iteration count matches reference CPU kernel.
// For FractalShark, define:
//  - ENABLE_FULL_KERNEL
//  - Ensure ENABLE_BASIC_CORRECTNESS == 4
// 
// TODO:
//  - ENABLE_REFERENCE_KERNEL is slightly busted on profiling and will say there's an error
// 

// Comment out to disable specific kernels
//#define ENABLE_CONVERSION_TESTS
#define ENABLE_ADD_KERNEL
//#define ENABLE_MULTIPLY_NTT_KERNEL
//#define ENABLE_REFERENCE_KERNEL
//#define ENABLE_FULL_KERNEL

// Uncomment this to enable the HpSharkFloat test program.
// Comment for use in FractalShark
#define HP_SHARK_FLOAT_TEST

// 0 = just one correctness test, intended for fast re-compile of a specific failure
// 1 = all basic correctness tests/all basic perf tests
// 2 = setup for profiling only, one kernel
// 3 = all basic correctness tests + comical tests
// 4 = "production" kernels, include periodicity. This is what we use in FractalShark.
// See ExplicitInstantiate.h for more information

#ifdef _DEBUG
#ifdef HP_SHARK_FLOAT_TEST
// Test path - this is what we use with HpSharkFloatTest
#define ENABLE_BASIC_CORRECTNESS 0
#else
// Production path - this is what we use in FractalShark
#define ENABLE_BASIC_CORRECTNESS 4
#endif
#else // not debug
#ifdef HP_SHARK_FLOAT_TEST
#define ENABLE_BASIC_CORRECTNESS 2
#else
#define ENABLE_BASIC_CORRECTNESS 4
#endif
#endif

namespace HpShark {

    #ifdef _DEBUG
    static constexpr bool Debug = true;
    #else
    static constexpr bool Debug = false;
    #endif

    static constexpr auto BasicCorrectness = ENABLE_BASIC_CORRECTNESS;

    #ifdef ENABLE_CONVERSION_TESTS
    static constexpr auto EnableConversionTests = true;
    #else
    static constexpr auto EnableConversionTests = false;
    #endif

    #ifdef ENABLE_ADD_KERNEL
    static constexpr auto EnableAddKernel = true;
    #else
    static constexpr auto EnableAddKernel = false;
    #endif

    #ifdef ENABLE_MULTIPLY_NTT_KERNEL
    static constexpr auto EnableMultiplyNTTKernel = true;
    #else
    static constexpr auto EnableMultiplyNTTKernel = false;
    #endif

    #ifdef ENABLE_REFERENCE_KERNEL
    static constexpr auto EnableReferenceKernel = true;
    #else
    static constexpr auto EnableReferenceKernel = false;
    #endif

    #ifdef ENABLE_FULL_KERNEL
    static constexpr auto EnableFullKernel = true;
    #else
    static constexpr auto EnableFullKernel = false;
    #endif

    static constexpr auto EnablePeriodicity = EnableFullKernel;

    #ifdef _DEBUG
    #define SharkForceInlineReleaseOnly
    #else
    #define SharkForceInlineReleaseOnly __forceinline__
    #endif

    static constexpr bool TestGpu = (EnableAddKernel || EnableMultiplyNTTKernel ||
                                          EnableReferenceKernel || EnableFullKernel);
    //static constexpr bool TestGpu = false;

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

    static constexpr InnerLoopOption SharkInnerLoopOption =
        InnerLoopOption::TryUnalignedLoads2;

    static constexpr auto LoadAllInShared =
        (SharkInnerLoopOption == InnerLoopOption::BasicAllInShared) ||
        (SharkInnerLoopOption == InnerLoopOption::TryUnalignedLoads2Shared);


    // Set to true to use shared memory for the incoming numbers
    static constexpr auto RegisterLimit = 127;

    static constexpr auto ConstantSharedRequiredBytes = 0;

    // TODO we should get this shit to work
    static constexpr bool DebugChecksums = (BasicCorrectness != 2) ? Debug : false;
    //static constexpr bool DebugChecksums = false;
    static constexpr bool PrintMultiplyCounts = false; // DebugChecksums;
    static constexpr bool TestCorrectness = (BasicCorrectness == 2) ? Debug : true;
    static constexpr bool TestInfiniteCorrectness = TestCorrectness ? true : false; // Was true : false
    static constexpr auto TestForceSameSign = false;
    static constexpr bool TestBenchmarkAgainstHost = false;
    static constexpr bool TestInitCudaMemory = true;

    // True to compare against the full host-side reference implementation, false is MPIR only
    // False is useful to speed up e.g. testing many cases fast but gives poor diagnostic results.
    static constexpr bool TestReferenceImpl = false;

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
}

template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pNumDigits = pThreadsPerBlock * pNumBlocks>
struct GenericSharkFloatParams {
    using Float = HDRFloat<float>;
    using SubType = float;

    static constexpr bool SharkUsePow2SizesOnly = HpShark::EnableMultiplyNTTKernel;

    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;

    // Fixed number of uint32_t values
    static constexpr int32_t Guard = 4;

    static constexpr int32_t GlobalNumUint32 =
        SharkUsePow2SizesOnly ?
            static_cast<int32_t>(
                HpShark::ceil_pow2_u32(static_cast<uint32_t>(pNumDigits)))
            : pNumDigits;

    constexpr static int32_t
    NumberOfBits(int32_t x) {
        return x < 2 ? x : 1 + NumberOfBits(x >> 1);
    }
    constexpr static auto LogNumUint32 = NumberOfBits(GlobalNumUint32);


    static constexpr int32_t HalfLimbsRoundedUp = (GlobalNumUint32 + 1) / 2;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    static constexpr bool DisableAllAdditions = false;
    static constexpr bool DisableSubtraction = false;
    static constexpr bool DisableCarryPropagation = false;
    static constexpr bool DisableFinalConstruction = false;
    static constexpr bool ForceNoOp = false;

    static constexpr auto NumDebugStates = (3 * static_cast<int>(DebugStatePurpose::NumPurposes));
    static constexpr auto NumDebugMultiplyCounts = GlobalThreadsPerBlock * GlobalNumBlocks;

    static std::string GetDescription() {
        std::string desc = "GlobalThreadsPerBlock: " + std::to_string(GlobalThreadsPerBlock) +
            ", GlobalNumBlocks: " + std::to_string(GlobalNumBlocks);
        return desc;
    }

    using GetHalf = typename std::conditional_t<
        (GlobalThreadsPerBlock > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock / 2, GlobalNumBlocks>,
        typename std::conditional_t<(GlobalNumBlocks > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks / 2>,
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks>
        >
    >;

    static constexpr SharkNTT::PlanPrime NTTPlan = SharkNTT::BuildPlanPrime(GlobalNumUint32, 26, 2);
    static constexpr bool Periodicity = HpShark::EnablePeriodicity;

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

static constexpr auto AdditionalGlobalSyncSpace = 128 * MaxBlocks;
static constexpr auto AdditionalGlobalMultipliesPerThread = HpShark::PrintMultiplyCounts ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = HpShark::DebugChecksums ? 1024 * 1024 : 0;

static constexpr auto AdditionalGlobalSyncSpaceOffset = 0;
static constexpr auto AdditionalMultipliesOffset = AdditionalGlobalSyncSpaceOffset + AdditionalGlobalSyncSpace;
static constexpr auto AdditionalChecksumsOffset = AdditionalMultipliesOffset + AdditionalGlobalMultipliesPerThread;

// Use the order of these three variables being added as the
// definition of how they are laid out in memory.
static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace +
    AdditionalGlobalMultipliesPerThread +
    AdditionalGlobalChecksumSpace;

template<class SharkFloatParams>
static constexpr auto CalculateKaratsubaFrameSize() {
    constexpr auto retval =
        ScratchMemoryArraysForMultiply *
        SharkFloatParams::GlobalNumUint32 +
        AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}

template <class SharkFloatParams>
static constexpr auto CalculateNTTFrameSize()
{
    constexpr auto retval =
        ScratchMemoryArraysForMultiply * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
    constexpr auto alignAt16BytesConstant = (retval % 16 == 0) ? 0 : (16 - retval % 16);
    return retval + alignAt16BytesConstant;
}


template<class SharkFloatParams>
constexpr int32_t CalculateMultiplySharedMemorySize() {
    constexpr int NewN = SharkFloatParams::GlobalNumUint32;
    constexpr auto n = (NewN + 1) / 2;              // Half of NewN

    // Figure out how much shared memory to allocate if we're not loading
    // everything into shared memory and instead using a constant amount.
    constexpr auto sharedRequired = SharkFloatParams::GlobalThreadsPerBlock * sizeof(uint64_t) * 3;

    //HpShark::ConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes =
        HpShark::LoadAllInShared ?
        (2 * NewN + 2 * n) * sizeof(uint32_t) :
        sharedRequired;

    return sharedAmountBytes;
}

template <class SharkFloatParams>
constexpr int32_t
CalculateNTTSharedMemorySize()
{
    // HpShark::ConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes = 3 * 2048 * sizeof(uint64_t);
    return sharedAmountBytes;
}

template<class SharkFloatParams>
static constexpr auto CalculateAddFrameSize() {
    return ScratchMemoryArraysForAdd * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

static constexpr auto LowPrec = 32;

#include "ExplicitInstantiate.h"


// If you add a new one, search for one of the other types and copy/paste
using Test8x1SharkParams = GenericSharkFloatParams<256, 64, 16384>;
//using Test8x1SharkParams = GenericSharkFloatParams<128, 128, 8192, 9>;
//using Test8x1SharkParams = GenericSharkFloatParams<128, 108, 7776, 9>;
//using Test8x1SharkParams = GenericSharkFloatParams<8, 128, 1024, 9>;
//using Test8x1SharkParams = GenericSharkFloatParams<8, 1>; // Use for ENABLE_BASIC_CORRECTNESS==1
using Test4x36SharkParams = GenericSharkFloatParams<4, 6, 32>;
using Test4x12SharkParams = GenericSharkFloatParams<3, 18, 50>;
using Test4x9SharkParams = GenericSharkFloatParams<5, 12, 80>;
using Test4x6SharkParams = GenericSharkFloatParams<7, 9, 74>;

// Use for ENABLE_BASIC_CORRECTNESS==2

// Performance test sizes
constexpr auto StupidMult = 1;
//using TestPerSharkParams1 = GenericSharkFloatParams<64, 128>;
//using TestPerSharkParams1 = GenericSharkFloatParams<96, 81>;
//using TestPerSharkParams1 = GenericSharkFloatParams<128 * StupidMult, 108, 7776, 9>;
//using TestPerSharkParams1 = GenericSharkFloatParams<128, 108, 7776, 9>;
using TestPerSharkParams1 = GenericSharkFloatParams<32, 2, 64>;
//using TestPerSharkParams1 = GenericSharkFloatParams<256, 128, 8192>;
//using TestPerSharkParams2 = GenericSharkFloatParams<256, 128, 131072>;
using TestPerSharkParams2 = GenericSharkFloatParams<256, 64, 16384>;
//using TestPerSharkParams2 = GenericSharkFloatParams<64 * StupidMult, 108, 7776, 9>;
using TestPerSharkParams3 = GenericSharkFloatParams<32 * StupidMult, 108, 7776>;
using TestPerSharkParams4 = GenericSharkFloatParams<16 * StupidMult, 108, 7776>;

using TestPerSharkParams5 = GenericSharkFloatParams<128 * StupidMult, 108, 7776>;
using TestPerSharkParams6 = GenericSharkFloatParams<64 * StupidMult, 108, 7776>;
using TestPerSharkParams7 = GenericSharkFloatParams<32 * StupidMult,  108, 7776>;
using TestPerSharkParams8 = GenericSharkFloatParams<16 * StupidMult,  108, 7776>;

// Correctness test sizes
using TestCorrectnessSharkParams1 = Test8x1SharkParams;
using TestCorrectnessSharkParams2 = Test4x36SharkParams;
using TestCorrectnessSharkParams3 = Test4x9SharkParams;
using TestCorrectnessSharkParams4 = Test4x12SharkParams;
using TestCorrectnessSharkParams5 = Test4x6SharkParams;

// FractalShark production sizes
using ProdSharkParams1 = GenericSharkFloatParams<256, 1>; // 256
using ProdSharkParams2 = GenericSharkFloatParams<256, 2>; // 512
using ProdSharkParams3 = GenericSharkFloatParams<256, 4>; // 1024
using ProdSharkParams4 = GenericSharkFloatParams<256, 8>; // 2048
using ProdSharkParams5 = GenericSharkFloatParams<256, 16>; // 4096
using ProdSharkParams6 = GenericSharkFloatParams<256, 32>; // 8192
using ProdSharkParams7 = GenericSharkFloatParams<256, 64>; // 16384
using ProdSharkParams8 = GenericSharkFloatParams<256, 128>; // 32768
using ProdSharkParams9 = GenericSharkFloatParams<256, 128, 65536>;
using ProdSharkParams10 = GenericSharkFloatParams<256, 128, 131072>;
using ProdSharkParams11 = GenericSharkFloatParams<256, 128, 262144>;
using ProdSharkParams12 = GenericSharkFloatParams<256, 128, 524288>;

enum class InjectNoiseInLowOrder {
    Disable,
    Enable
};

// Struct to hold both integer and fractional parts of the high-precision number
template<class SharkFloatParams>
struct HpSharkFloat {
    HpSharkFloat();
    //HpSharkFloat(uint32_t numDigits);
    HpSharkFloat(const uint32_t *digitsIn, int32_t expIn, bool isNegative);
    ~HpSharkFloat() = default;
    HpSharkFloat &operator=(const HpSharkFloat<SharkFloatParams> &);

    CUDA_CRAP void DeepCopySameDevice(const HpSharkFloat<SharkFloatParams> &other);

#if defined(__CUDA_ARCH__)
    CUDA_CRAP void DeepCopyGPU(
        cooperative_groups::grid_group &grid,
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
    void GenerateRandomNumber2();
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
    template <class SubType> CUDA_CRAP_BOTH
    HDRFloat<SubType> ToHDRFloat(int32_t extraExp = 0) const;

    template <class SubType> CUDA_CRAP_BOTH
    void FromHDRFloat(const HDRFloat<SubType> &h);

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

template<class SharkFloatParams>
CUDA_CRAP_BOTH
void HpSharkFloat<SharkFloatParams>::SetNegative(bool isNegative) {
    IsNegative = isNegative;
}

template<class SharkFloatParams>
CUDA_CRAP_BOTH
bool HpSharkFloat<SharkFloatParams>::GetNegative() const {
    return IsNegative;
}

template <class SharkFloatParams>
template <class SubType>
CUDA_CRAP_BOTH
HDRFloat<SubType> HpSharkFloat<SharkFloatParams>::ToHDRFloat(int32_t extraExp) const
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

    using HpT = HpSharkFloat<SharkFloatParams>;
    using DT = typename HpT::DigitType;
    constexpr int N = HpT::NumUint32;

    // Zero digits
    for (int i = 0; i < N; ++i)
        Digits[i] = DT{0};

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
    Digits[N - 1] = static_cast<DT>(hi32);
    if constexpr (N >= 2) {
        Digits[N - 2] = static_cast<DT>(lo32);
    }

    // Choose Hp exponent so overall value equals mantissa * 2^exp:
    //   Value = (DigitsAsInt) * 2^(Exponent)
    // where DigitsAsInt has MSB at msb_abs.
    // Since HDR value's msb is at absolute bit index 'exp', set:
    Exponent = exp - msb_abs;

    SetNegative(neg);
}


#pragma warning(push)
#pragma warning(disable:4324)
template<class SharkFloatParams>
struct alignas(16) HpSharkComboResults{
    SharkNTT::RootTables Roots;
    alignas(16) HpSharkFloat<SharkFloatParams> A;
    alignas(16) HpSharkFloat<SharkFloatParams> B;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultX2;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2XY;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultY2;
};

template<class SharkFloatParams>
struct HpSharkAddComboResults {
    alignas(16) HpSharkFloat<SharkFloatParams> A_X2;
    alignas(16) HpSharkFloat<SharkFloatParams> B_Y2;
    alignas(16) HpSharkFloat<SharkFloatParams> C_A;
    alignas(16) HpSharkFloat<SharkFloatParams> D_2X;
    alignas(16) HpSharkFloat<SharkFloatParams> E_B;
    alignas(16) HpSharkFloat<SharkFloatParams> Result1_A_B_C;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2_D_E;
};

template<class SharkFloatParams>
struct HpSharkReferenceResults {
    alignas(16) HDRFloat<typename SharkFloatParams::SubType> RadiusY;
    alignas(16) HpSharkComboResults<SharkFloatParams> Multiply;
    alignas(16) HpSharkAddComboResults<SharkFloatParams> Add;
    alignas(16) uint64_t Period;
    alignas(16) uint64_t EscapedIteration;
    alignas(16) typename SharkFloatParams::ReferenceIterT *OutputIters;
};
#pragma warning(pop)

template <class SharkFloatParams> std::string MpfToString(const mpf_t mpf_val, size_t precInBits);

template<class SharkFloatParams>
std::string Uint32ToMpf(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);

void Uint64ToMpf(
    const uint64_t *array, size_t numElts, int32_t pow64Exponent, mpf_t &mpf_val, bool isNegative);

template<class IntT>
std::string
UintArrayToHexString(const IntT *array, size_t numElements);

template<class IntT>
std::string
UintToHexString(IntT val);

template<class IntT>
std::string
VectorUintToHexString(const std::vector<IntT> &arr);

template<class IntT>
std::string
VectorUintToHexString(const IntT *arr, size_t numElements);
