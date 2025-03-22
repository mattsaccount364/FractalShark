#pragma once

#include "DebugStateRaw.h"

#include <string>
#include <gmp.h>
#include <vector>

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
// #define SharkRestrict

#ifdef _DEBUG
static constexpr bool SharkDebug = true;
#else
static constexpr bool SharkDebug = false;
#endif

// 0 = just one correctness test, intended for fast re-compile of a specific failure
// 1 = all basic correctness tests/all basic perf tests
// 2 = setup for profiling only, one kernel
// 3 = all basic correctness tests + comical tests
// See ExplicitInstantiate.h for more information
#define ENABLE_BASIC_CORRECTNESS 3
static constexpr auto SharkComicalThreadCount = 13;
static constexpr auto SharkTestIterCount = SharkDebug ? 3 : 50000;

// Set to true to use a custom stream for the kernel launch
static constexpr auto SharkCustomStream = true;

// Set to true to use shared memory for the incoming numbers
static constexpr auto SharkUseSharedMemory = true;
static constexpr auto SharkRegisterLimit = 255;

// TODO not hooked up right, leave as 0.  Idea is if we need fixed-size
// amount of shared memory we can use this but as it is we don't use it.
static constexpr auto SharkConstantSharedRequiredBytes = 0;

// Undefine to include N2, v1 etc.
// #define MULTI_KERNEL

#ifdef MULTI_KERNEL
static constexpr auto SharkMultiKernel = true;
#else
static constexpr auto SharkMultiKernel = false;
#endif

static constexpr auto SharkBatchSize = SharkDebug ? 8 : 512;
static constexpr bool SharkInfiniteCorrectnessTests = true;
static constexpr bool SharkCorrectnessTests = true;
static constexpr bool SharkDebugChecksums = SharkDebug;
static constexpr bool SharkDebugRandomDelays = false;

// Set to false to bypass all GPU tests and only do reference/host-side
static constexpr bool SharkTestGpu = true;
static constexpr bool SharkBenchmarkAgainstHost = true;


template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pNumDigits = pThreadsPerBlock * pNumBlocks,
    int32_t pConvolutionLimit = 9>
struct GenericSharkFloatParams {
    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;
    static constexpr int32_t SharkBatchSize = ::SharkBatchSize;
    // Fixed number of uint32_t values
    static constexpr int32_t GlobalNumUint32 = pNumDigits;
    static constexpr int32_t HalfLimbsRoundedUp = (GlobalNumUint32 + 1) / 2;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    static constexpr bool DisableAllAdditions = false;
    static constexpr bool DisableSubtraction = false;
    static constexpr bool DisableCarryPropagation = false;
    static constexpr bool DisableFinalConstruction = false;
    static constexpr bool ForceNoOp = false;

    // If true, the host will print out a lot of stuff
    static constexpr bool HostVerbose = false;

    // 1, 3, 9, 27, 81
    static constexpr auto ConvolutionLimit = pConvolutionLimit;

    static constexpr auto NumDebugStates = ((ConvolutionLimit + 1) * 3 * static_cast<int>(DebugStatePurpose::NumPurposes));

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
static constexpr auto ScratchMemoryArrays = 96;

// Additional space per frame:
static constexpr auto AdditionalUInt64PerFrame = 256;

// Additional space up front, globally-shared:
// Units are uint64_t
static constexpr auto MaxBlocks = 256;

static constexpr auto AdditionalGlobalSyncSpace = 128 * MaxBlocks;
static constexpr auto AdditionalGlobalRandomSpace = SharkDebugRandomDelays ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalChecksumSpace = SharkDebugChecksums ? 1024 * 1024 : 0;

// Use the order of these three variables being added as the
// definition of how they are laid out in memory.
static constexpr auto AdditionalUInt64Global =
    AdditionalGlobalSyncSpace + AdditionalGlobalRandomSpace + AdditionalGlobalChecksumSpace;

template<class SharkFloatParams>
static constexpr auto CalculateFrameSize() {
    return ScratchMemoryArrays * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

static constexpr auto LowPrec = 32;

#include "ExplicitInstantiate.h"



// If you add a new one, search for one of the other types and copy/paste
// using Test8x1SharkParams = GenericSharkFloatParams<64, 108, 7776, 9>; // Use for ENABLE_BASIC_CORRECTNESS==2
using Test8x1SharkParams = GenericSharkFloatParams<8, 1>; // Use for ENABLE_BASIC_CORRECTNESS==1
// using Test8x1SharkParams = GenericSharkFloatParams<13, 5>;
// using Test8x1SharkParams = GenericSharkFloatParams<95, 81>;
using Test4x36SharkParams = GenericSharkFloatParams<4, 6, 32>;
using Test4x12SharkParams = GenericSharkFloatParams<3, 18, 50>;
using Test4x9SharkParams = GenericSharkFloatParams<5, 12, 80>;
using Test4x6SharkParams = GenericSharkFloatParams<7, 9, 74>;

// Use for ENABLE_BASIC_CORRECTNESS==2

// Performance test sizes
//using TestPerSharkParams1 = GenericSharkFloatParams<128, 128>;
//using TestPerSharkParams1 = GenericSharkFloatParams<96, 81>;
using TestPerSharkParams1 = GenericSharkFloatParams<256, 108, 7776, 9>;
using TestPerSharkParams2 = GenericSharkFloatParams<128, 108, 7776, 9>;
using TestPerSharkParams3 = GenericSharkFloatParams<64,  108, 7776, 9>;
using TestPerSharkParams4 = GenericSharkFloatParams<32,  108, 7776, 9>;

using TestPerSharkParams5 = GenericSharkFloatParams<256, 108, 7776, 27>;
using TestPerSharkParams6 = GenericSharkFloatParams<128, 108, 7776, 27>;
using TestPerSharkParams7 = GenericSharkFloatParams<64,  108, 7776, 27>;
using TestPerSharkParams8 = GenericSharkFloatParams<32,  108, 7776, 27>;

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

    void DeepCopySameDevice(const HpSharkFloat<SharkFloatParams> &other);

    std::string ToString() const;
    std::string ToHexString() const;
    void GenerateRandomNumber();
    void GenerateRandomNumber2();
    void Negate();

    // Default precision in bits
    constexpr static auto NumUint32 = SharkFloatParams::GlobalNumUint32;
    constexpr static auto DefaultPrecBits = NumUint32 * sizeof(uint32_t) * 8;
    constexpr static auto ConvertBitsToDecimals = 3.3219280948873623478703194294894;
    constexpr static auto DefaultPrecDigits = DefaultPrecBits / ConvertBitsToDecimals;
    constexpr static auto DefaultMpirBits = DefaultPrecBits;

    // Digits in base 2^32
    uint32_t Digits[NumUint32];

    // Exponent in base 2
    using ExpT = int32_t;
    ExpT Exponent;

    // Sign
    bool IsNegative;

private:
    mp_exp_t HpGpuExponentToMpfExponent(size_t numBytesToCopy) const;
};

template<class SharkFloatParams>
struct HpSharkComboResults {
    HpSharkFloat<SharkFloatParams> A;
    HpSharkFloat<SharkFloatParams> B;
    HpSharkFloat<SharkFloatParams> ResultX2;
    HpSharkFloat<SharkFloatParams> ResultXY;
    HpSharkFloat<SharkFloatParams> ResultY2;
};

template<class SharkFloatParams>
std::string MpfToString(const mpf_t mpf_val, size_t precInBits);

std::string MpfToHexString(const mpf_t mpf_val);

template<class SharkFloatParams>
void MpfToHpGpu(const mpf_t mpf_val, HpSharkFloat<SharkFloatParams> &number, int prec_bits);

template<class SharkFloatParams>
void HpGpuToMpf(const HpSharkFloat<SharkFloatParams> &hpNum, mpf_t &mpf_val);

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
