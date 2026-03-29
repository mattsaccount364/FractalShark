#pragma once

#include "CudaCrap.h"
#include "DebugStateRaw.h"
#include "GPU_ReferenceIter.h"
#include "HDRFloat.h"
#include "LaunchParams.h"
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
//#define SharkRestrict

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
static constexpr bool TestInitCudaMemory = true;

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

template <int32_t pNumDigits, bool Periodicity, bool NewtonRaphson = false,
          typename SubTypeT = float>
struct GenericSharkFloatParams {
    using Float = HDRFloat<SubTypeT>;
    using SubType = SubTypeT;

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
    static constexpr auto EnableNewtonRaphson = NewtonRaphson;
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

} // namespace HpShark

#include "HpSharkScratchMemory.h"

#include "ExplicitInstantiate.h"

// If you add a new one, search for one of the other types and copy/paste.
// NOTE: If you add sizes larger than 524288 limbs, also update
// HighPrecisionT::MaxPrecisionBits in HighPrecision.h.
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

// Newton-Raphson enabled params (NR derivative tracking, NO periodicity detection).
// NOTE: These all use default SubType=float. The NR kernel's d2 accumulation
// (Halley second derivative) uses SharkFloatParams::Float = HDRFloat<float>,
// giving ~24-bit d2 precision. If the user renders with HDRFloat<double> or
// HDRFloat<CudaDblflt>, the NR inner loop still computes d2 at float precision.
// This causes Halley's method to degenerate to quadratic convergence after ~24
// correct bits instead of ~53 — potentially costing one extra NR iteration.
// Adding SharkParamsNRDbl/NRDbf variants would fix this at the cost of more
// template instantiations and compile time. Left as a pending open question.
using SharkParamsNR1 = HpShark::GenericSharkFloatParams<256, false, true>;
using SharkParamsNR2 = HpShark::GenericSharkFloatParams<512, false, true>;
using SharkParamsNR3 = HpShark::GenericSharkFloatParams<1024, false, true>;
using SharkParamsNR4 = HpShark::GenericSharkFloatParams<2048, false, true>;
using SharkParamsNR5 = HpShark::GenericSharkFloatParams<4096, false, true>;
using SharkParamsNR6 = HpShark::GenericSharkFloatParams<8192, false, true>;
using SharkParamsNR7 = HpShark::GenericSharkFloatParams<16384, false, true>;
using SharkParamsNR8 = HpShark::GenericSharkFloatParams<32768, false, true>;
using SharkParamsNR9 = HpShark::GenericSharkFloatParams<65536, false, true>;
using SharkParamsNR10 = HpShark::GenericSharkFloatParams<131072, false, true>;
using SharkParamsNR11 = HpShark::GenericSharkFloatParams<262144, false, true>;
using SharkParamsNR12 = HpShark::GenericSharkFloatParams<524288, false, true>;

// Production sizes with double SubType
using SharkParamsDbl1 = HpShark::GenericSharkFloatParams<256, true, false, double>;
using SharkParamsDbl2 = HpShark::GenericSharkFloatParams<512, true, false, double>;
using SharkParamsDbl3 = HpShark::GenericSharkFloatParams<1024, true, false, double>;
using SharkParamsDbl4 = HpShark::GenericSharkFloatParams<2048, true, false, double>;
using SharkParamsDbl5 = HpShark::GenericSharkFloatParams<4096, true, false, double>;
using SharkParamsDbl6 = HpShark::GenericSharkFloatParams<8192, true, false, double>;
using SharkParamsDbl7 = HpShark::GenericSharkFloatParams<16384, true, false, double>;
using SharkParamsDbl8 = HpShark::GenericSharkFloatParams<32768, true, false, double>;
using SharkParamsDbl9 = HpShark::GenericSharkFloatParams<65536, true, false, double>;
using SharkParamsDbl10 = HpShark::GenericSharkFloatParams<131072, true, false, double>;
using SharkParamsDbl11 = HpShark::GenericSharkFloatParams<262144, true, false, double>;
using SharkParamsDbl12 = HpShark::GenericSharkFloatParams<524288, true, false, double>;

// Production sizes with CudaDblflt<dblflt> SubType
using SharkParamsDbf1 = HpShark::GenericSharkFloatParams<256, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf2 = HpShark::GenericSharkFloatParams<512, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf3 = HpShark::GenericSharkFloatParams<1024, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf4 = HpShark::GenericSharkFloatParams<2048, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf5 = HpShark::GenericSharkFloatParams<4096, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf6 = HpShark::GenericSharkFloatParams<8192, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf7 = HpShark::GenericSharkFloatParams<16384, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf8 = HpShark::GenericSharkFloatParams<32768, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf9 = HpShark::GenericSharkFloatParams<65536, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf10 = HpShark::GenericSharkFloatParams<131072, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf11 = HpShark::GenericSharkFloatParams<262144, true, false, CudaDblflt<dblflt>>;
using SharkParamsDbf12 = HpShark::GenericSharkFloatParams<524288, true, false, CudaDblflt<dblflt>>;

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

#include "HpSharkFloatConversions.h"
#include "HpSharkKernelResults.h"

template <class SharkFloatParams> std::string MpfToString(const mpf_t mpf_val, size_t precInBits);

template <class SharkFloatParams>
std::string Uint32ToMpf(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);

template <class IntT> std::string UintArrayToHexString(const IntT *array, size_t numElements);

template <class IntT> std::string UintToHexString(IntT val);

template <class IntT> std::string VectorUintToHexString(const std::vector<IntT> &arr);

template <class IntT> std::string VectorUintToHexString(const IntT *arr, size_t numElements);
