#pragma once

#include "DebugStateRaw.h"

#include <string>
#include <gmp.h>
#include <vector>

// Assuming that SharkFloatParams::GlobalNumUint32 can be large and doesn't fit in shared memory
// We'll use the provided global memory buffers for large intermediates
#define SharkRestrict __restrict__
// #define SharkRestrict

// 0 = just one correctness test, intended for fast re-compile of a specific failure
// 1 = all basic correctness tests/all basic perf tests
// 2 = setup for profiling only, one kernel
// 3 = all basic correctness tests + comical tests
// See ExplicitInstantiate.h for more information
#define ENABLE_BASIC_CORRECTNESS 2
static constexpr auto SharkComicalThreadCount = 13;
static constexpr auto SharkTestIterCount = 5000;

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

#ifdef _DEBUG
static constexpr bool SharkDebug = true;
#else
static constexpr bool SharkDebug = false;
#endif

static constexpr auto SharkBatchSize = SharkDebug ? 8 : 512;
static constexpr bool SharkInfiniteCorrectnessTests = true;
static constexpr bool SharkCorrectnessTests = true;
static constexpr bool SharkDebugChecksums = false;
static constexpr bool SharkDebugRandomDelays = false;

// Set to false to bypass all GPU tests and only do reference/host-side
static constexpr bool SharkTestGpu = true;


template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pBatchSize,
    int32_t pNumDigits = pThreadsPerBlock * pNumBlocks>
struct GenericSharkFloatParams {
    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;
    static constexpr int32_t SharkBatchSize = pBatchSize;
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

    // 3^whatevs = ConvolutionLimit
    // static constexpr auto ConvolutionLimit = SharkDebug ? 27: 81;
    // static constexpr auto ConvolutionLimitPow = SharkDebug ? 3 : 4;

    // static constexpr auto ConvolutionLimit = 1;
    // static constexpr auto ConvolutionLimitPow = 0;

    // static constexpr auto ConvolutionLimit = 3;
    // static constexpr auto ConvolutionLimitPow = 1;

    static constexpr auto ConvolutionLimit = 9;
    static constexpr auto ConvolutionLimitPow = 2;

    // static constexpr auto ConvolutionLimit = 27;
    // static constexpr auto ConvolutionLimitPow = 3;

    // static constexpr auto ConvolutionLimit = 81;
    // static constexpr auto ConvolutionLimitPow = 4;

    static constexpr auto NumDebugStates = ((ConvolutionLimit + 1) * 3 * static_cast<int>(DebugStatePurpose::NumPurposes));

    static std::string GetDescription() {
        std::string desc = "GlobalThreadsPerBlock: " + std::to_string(GlobalThreadsPerBlock) +
            ", GlobalNumBlocks: " + std::to_string(GlobalNumBlocks) +
            ", SharkBatchSize: " + std::to_string(SharkBatchSize);
        return desc;
    }

    using GetHalf = typename std::conditional_t<
        (GlobalThreadsPerBlock > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock / 2, GlobalNumBlocks, SharkBatchSize>,
        typename std::conditional_t<(GlobalNumBlocks > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks / 2, SharkBatchSize>,
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks, SharkBatchSize>
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
using Test8x1SharkParams = GenericSharkFloatParams<8, 1, SharkBatchSize>; // Use for ENABLE_BASIC_CORRECTNESS==1
// using Test8x1SharkParams = GenericSharkFloatParams<13, 5, SharkBatchSize>;
// using Test8x1SharkParams = GenericSharkFloatParams<95, 81, SharkBatchSize>;
using Test4x36SharkParams = GenericSharkFloatParams<4, 6, SharkBatchSize>;
using Test4x12SharkParams = GenericSharkFloatParams<3, 18, SharkBatchSize>;
using Test4x9SharkParams = GenericSharkFloatParams<5, 12, SharkBatchSize>;
using Test4x6SharkParams = GenericSharkFloatParams<7, 9, SharkBatchSize>;

//using Test128x128SharkParams = GenericSharkFloatParams<128, 128, SharkBatchSize>;
//using Test128x63SharkParams = GenericSharkFloatParams<96, 81, SharkBatchSize>;
using Test128x63SharkParams = GenericSharkFloatParams<256, 126, SharkBatchSize, 7776>; // Use for ENABLE_BASIC_CORRECTNESS==2
using Test64x63SharkParams = GenericSharkFloatParams<64, 81, SharkBatchSize>;
using Test32x63SharkParams = GenericSharkFloatParams<32, 81, SharkBatchSize>;
using Test16x63SharkParams = GenericSharkFloatParams<16, 81, SharkBatchSize>;

using Test128x36SharkParams = GenericSharkFloatParams<100, 81, SharkBatchSize>;
using Test128x18SharkParams = GenericSharkFloatParams<100, 36, SharkBatchSize>;
using Test128x9SharkParams = GenericSharkFloatParams<100, 12, SharkBatchSize>;
using Test128x3SharkParams = GenericSharkFloatParams<100, 6, SharkBatchSize>;

// Performance test sizes
using TestPerSharkParams1 = Test128x63SharkParams;
using TestPerSharkParams2 = Test64x63SharkParams;
using TestPerSharkParams3 = Test32x63SharkParams;
using TestPerSharkParams4 = Test16x63SharkParams;

using TestPerSharkParams5 = Test128x36SharkParams;
using TestPerSharkParams6 = Test128x18SharkParams;
using TestPerSharkParams7 = Test128x9SharkParams;
using TestPerSharkParams8 = Test128x3SharkParams;

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
