#pragma once

#include "CudaCrap.h"
#include "DebugStateRaw.h"

#include <string>
#include <gmp.h>
#include <vector>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#endif

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
// #define SharkRestrict

#ifdef _DEBUG
static constexpr bool SharkDebug = true;
#else
static constexpr bool SharkDebug = false;
#endif

// Comment out to disable specific kernels
//#define ENABLE_CONVERSION_TESTS
//#define ENABLE_ADD_KERNEL
//#define ENABLE_MULTIPLY_KARATSUBA_KERNEL
//#define ENABLE_MULTIPLY_FFT_KERNEL
#define ENABLE_MULTIPLY_FFT2_KERNEL
//#define ENABLE_REFERENCE_KERNEL

// 0 = just one correctness test, intended for fast re-compile of a specific failure
// 1 = all basic correctness tests/all basic perf tests
// 2 = setup for profiling only, one kernel
// 3 = all basic correctness tests + comical tests
// See ExplicitInstantiate.h for more information
#ifdef _DEBUG
#define ENABLE_BASIC_CORRECTNESS 0
#else
#define ENABLE_BASIC_CORRECTNESS 2
#endif

#ifdef ENABLE_CONVERSION_TESTS
static constexpr auto SharkEnableConversionTests = true;
#else
static constexpr auto SharkEnableConversionTests = false;
#endif

#ifdef ENABLE_ADD_KERNEL
static constexpr auto SharkEnableAddKernel = true;
#else
static constexpr auto SharkEnableAddKernel = false;
#endif

#ifdef ENABLE_MULTIPLY_KARATSUBA_KERNEL
static constexpr auto SharkEnableMultiplyKernel = true;
#else
static constexpr auto SharkEnableMultiplyKernel = false;
#endif

#ifdef ENABLE_MULTIPLY_FFT_KERNEL
static constexpr auto SharkEnableMultiplyFFTKernel = true;
#else
static constexpr auto SharkEnableMultiplyFFTKernel = false;
#endif

#ifdef ENABLE_MULTIPLY_FFT2_KERNEL
static constexpr auto SharkEnableMultiplyFFT2Kernel = true;
#else
static constexpr auto SharkEnableMultiplyFFT2Kernel = false;
#endif

#ifdef ENABLE_REFERENCE_KERNEL
static constexpr auto SharkEnableReferenceKernel = true;
#else
static constexpr auto SharkEnableReferenceKernel = false;
#endif

#ifdef _DEBUG
#define SharkForceInlineReleaseOnly
#else
// #define SharkForceInlineReleaseOnly __forceinline__
#define SharkForceInlineReleaseOnly __forceinline__
#endif

//static constexpr bool SharkTestGpu =
// (SharkEnableAddKernel || SharkEnableMultiplyKernel || /*SharkEnableMultiplyFFTKernel ||*/ SharkEnableMultiplyFFT2Kernel || SharkEnableReferenceKernel);
static constexpr bool SharkTestGpu = false;

static constexpr auto SharkTestComicalThreadCount = 13;

static constexpr auto SharkTestIterCount = SharkDebug ? 5 : 50000;

// Set to true to use a custom stream for the kernel launch
static constexpr auto SharkCustomStream = true;

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

static constexpr auto SharkLoadAllInShared =
    (SharkInnerLoopOption == InnerLoopOption::BasicAllInShared) ||
    (SharkInnerLoopOption == InnerLoopOption::TryUnalignedLoads2Shared);

// Set to true to use shared memory for the incoming numbers
static constexpr auto SharkRegisterLimit = 255;

static constexpr auto SharkConstantSharedRequiredBytes = 0;

static constexpr auto SharkBatchSize = SharkDebug ? 8 : 512;

static constexpr auto SharkKaratsubaBatchSize = SharkLoadAllInShared ? 1 : 4;

static constexpr bool SharkDebugChecksums = (ENABLE_BASIC_CORRECTNESS != 2) ? SharkDebug : false;

static constexpr bool SharkPrintMultiplyCounts = true; //SharkDebugChecksums; // SharkDebugChecksums;

#if ENABLE_BASIC_CORRECTNESS == 2
static constexpr bool SharkTestCorrectness = SharkDebug;
#else
static constexpr bool SharkTestCorrectness = true;
#endif

static constexpr bool SharkTestInfiniteCorrectness = SharkTestCorrectness ? true : false;
static constexpr auto SharkTestForceSameSign = false;
static constexpr bool SharkTestBenchmarkAgainstHost = false;
static constexpr bool SharkTestInitCudaMemory = true;

// ---- detail helpers (fallback for pre-C++20) ----
namespace HpShark {
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
} // namespace detail

template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pNumDigits = pThreadsPerBlock * pNumBlocks,
    int32_t pConvolutionLimit = 9>
struct GenericSharkFloatParams {
    static constexpr bool SharkUsePow2SizesOnly =
        SharkEnableMultiplyFFTKernel || SharkEnableMultiplyFFT2Kernel;

    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;
    static constexpr int32_t SharkBatchSize = ::SharkBatchSize;
    // Fixed number of uint32_t values
    static constexpr int32_t Guard = 4;

    static constexpr int32_t GlobalNumUint32 =
        SharkUsePow2SizesOnly ?
            static_cast<int32_t>(
                HpShark::ceil_pow2_u32(static_cast<uint32_t>(pNumDigits)))
            : pNumDigits;

    static constexpr int32_t HalfLimbsRoundedUp = (GlobalNumUint32 + 1) / 2;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    static constexpr bool DisableAllAdditions = false;
    static constexpr bool DisableSubtraction = false;
    static constexpr bool DisableCarryPropagation = false;
    static constexpr bool DisableFinalConstruction = false;
    static constexpr bool ForceNoOp = false;

    // 1, 3, 9, 27, 81
    static constexpr auto ConvolutionLimit = pConvolutionLimit;

    static constexpr auto NumDebugStates = ((ConvolutionLimit + 1) * 3 * static_cast<int>(DebugStatePurpose::NumPurposes));
    static constexpr auto NumDebugMultiplyCounts = GlobalThreadsPerBlock * GlobalNumBlocks;

    static std::string GetDescription() {
        std::string desc = "GlobalThreadsPerBlock: " + std::to_string(GlobalThreadsPerBlock) +
            ", GlobalNumBlocks: " + std::to_string(GlobalNumBlocks) +
            ", SharkBatchSize: " + std::to_string(SharkBatchSize);
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
};

// This one should account for maximum call index, e.g. if we generate 500 calls
// recursively then we need this to be at 500.
static constexpr auto ScratchMemoryCopies = 256;

// Number of arrays of digits on each frame
static constexpr auto ScratchMemoryArraysForMultiply = 96;
static constexpr auto ScratchMemoryArraysForAdd = 64;


// Additional space per frame:
static constexpr auto AdditionalUInt64PerFrame = 256;

// Additional space up front, globally-shared:
// Units are uint64_t
static constexpr auto MaxBlocks = 256;

static constexpr auto AdditionalGlobalSyncSpace = 128 * MaxBlocks;
static constexpr auto AdditionalGlobalMultipliesPerThread = SharkPrintMultiplyCounts ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = SharkDebugChecksums ? 1024 * 1024 : 0;

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
static constexpr auto CalculateMultiplyFrameSize() {
    constexpr auto retval =
        ScratchMemoryArraysForMultiply *
        SharkFloatParams::GlobalNumUint32 +
        AdditionalUInt64PerFrame;
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

    //SharkConstantSharedRequiredBytes
    constexpr auto sharedAmountBytes =
        SharkLoadAllInShared ?
        (2 * NewN + 2 * n) * sizeof(uint32_t) :
        sharedRequired;

    return sharedAmountBytes;
}

template<class SharkFloatParams>
static constexpr auto CalculateAddFrameSize() {
    return ScratchMemoryArraysForAdd * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

static constexpr auto LowPrec = 32;

#include "ExplicitInstantiate.h"



// If you add a new one, search for one of the other types and copy/paste
//using Test8x1SharkParams = GenericSharkFloatParams<128, 108, 7776, 9>;
using Test8x1SharkParams = GenericSharkFloatParams<8, 1>; // Use for ENABLE_BASIC_CORRECTNESS==1
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
using TestPerSharkParams1 = GenericSharkFloatParams<128, 108, 7776, 9>;
using TestPerSharkParams2 = GenericSharkFloatParams<64 * StupidMult, 108, 7776, 9>;
using TestPerSharkParams3 = GenericSharkFloatParams<32 * StupidMult, 108, 7776, 9>;
using TestPerSharkParams4 = GenericSharkFloatParams<16 * StupidMult, 108, 7776, 9>;

using TestPerSharkParams5 = GenericSharkFloatParams<128 * StupidMult, 108, 7776, 27>;
using TestPerSharkParams6 = GenericSharkFloatParams<64 * StupidMult, 108, 7776, 27>;
using TestPerSharkParams7 = GenericSharkFloatParams<32 * StupidMult,  108, 7776, 27>;
using TestPerSharkParams8 = GenericSharkFloatParams<16 * StupidMult,  108, 7776, 27>;

// Correctness test sizes
using TestCorrectnessSharkParams1 = Test8x1SharkParams;
using TestCorrectnessSharkParams2 = Test4x36SharkParams;
using TestCorrectnessSharkParams3 = Test4x9SharkParams;
using TestCorrectnessSharkParams4 = Test4x12SharkParams;
using TestCorrectnessSharkParams5 = Test4x6SharkParams;

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
    void MpfToHpGpu(const mpf_t mpf_val, int prec_bits);

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

#pragma warning(push)
#pragma warning(disable:4324)
template<class SharkFloatParams>
struct alignas(16) HpSharkComboResults{
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
    alignas(16) HpSharkComboResults<SharkFloatParams> Multiply;
    alignas(16) HpSharkAddComboResults<SharkFloatParams> Add;
};
#pragma warning(pop)

template<class SharkFloatParams>
std::string MpfToString(const mpf_t mpf_val, size_t precInBits);

std::string MpfToHexString(const mpf_t mpf_val);

template<class SharkFloatParams>
std::string Uint32ToMpf(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);

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
