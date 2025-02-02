#pragma once

#include "DebugStateRaw.h"

#include <string>
#include <gmp.h>
#include <vector>

static constexpr bool SharkCustomStream = true;
static constexpr bool UseSharedMemory = true;

// Undefine to include N2, v1 etc.
// #define MULTI_KERNEL

#ifdef MULTI_KERNEL
static constexpr auto MultiKernel = true;
#else
static constexpr auto MultiKernel = false;
#endif

#ifdef _DEBUG
static constexpr bool SharkDebug = true;
#else
static constexpr bool SharkDebug = false;
#endif

static constexpr auto SharkTestIterCount = 5000;
static constexpr auto SharkBatchSize = SharkDebug ? 8 : 512;
static constexpr bool SharkInfiniteCorrectnessTests = true;
static constexpr bool SharkCorrectnessTests = true;
static constexpr bool DebugChecksums = true;

template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pBatchSize,
    int32_t pTestIterCount>
struct GenericSharkFloatParams {
    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;
    static constexpr int32_t SharkBatchSize = pBatchSize;
    static constexpr int32_t SharkTestIterCount = pTestIterCount;
    // Fixed number of uint32_t values
    static constexpr int32_t GlobalNumUint32 = GlobalThreadsPerBlock * GlobalNumBlocks;
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
    static constexpr auto ConvolutionLimit = SharkDebug ? 9 : 27;
    static constexpr auto ConvolutionLimitPow = SharkDebug ? 2 : 3;

    //static constexpr auto ConvolutionLimit = 3;
    //static constexpr auto ConvolutionLimitPow = 1;

    //static constexpr auto ConvolutionLimit = 9;
    //static constexpr auto ConvolutionLimitPow = 2;

    //static constexpr auto ConvolutionLimit = 3;
    //static constexpr auto ConvolutionLimitPow = 1;

    static constexpr auto NumDebugStates = ((ConvolutionLimit + 1) * 3 * static_cast<int>(DebugStatePurpose::NumPurposes));

    static std::string GetDescription() {
        std::string desc = "GlobalThreadsPerBlock: " + std::to_string(GlobalThreadsPerBlock) +
            ", GlobalNumBlocks: " + std::to_string(GlobalNumBlocks) +
            ", SharkBatchSize: " + std::to_string(SharkBatchSize) +
            ", SharkTestIterCount: " + std::to_string(SharkTestIterCount);
        return desc;
    }

    using GetHalf = typename std::conditional_t<
        (GlobalThreadsPerBlock > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock / 2, GlobalNumBlocks, SharkBatchSize, SharkTestIterCount>,
        typename std::conditional_t<(GlobalNumBlocks > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks / 2, SharkBatchSize, SharkTestIterCount>,
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks, SharkBatchSize, SharkTestIterCount>
        >
    >;
};

// This one should account for maximum call index, e.g. if we generate 500 calls
// recursively then we need this to be at 500.
static constexpr auto ScratchMemoryCopies = 256;

// Number of arrays of digits on each frame
static constexpr auto ScratchMemoryArrays = 32;

// Additional space per frame:
static constexpr auto AdditionalUInt64PerFrame = 256;

// Additional space up front, globally-shared:
// Units are uint64_t
static constexpr auto AdditionalGlobalChecksumSpace = DebugChecksums ? 1024 * 1024 : 0;
static constexpr auto AdditionalGlobalSyncSpace = 128;
static constexpr auto AdditionalUInt64Global = AdditionalGlobalChecksumSpace + AdditionalGlobalSyncSpace;

template<class SharkFloatParams>
static constexpr auto CalculateFrameSize() {
    return ScratchMemoryArrays * SharkFloatParams::GlobalNumUint32 + AdditionalUInt64PerFrame;
}

static constexpr auto LowPrec = 32;

#include "ExplicitInstantiate.h"



// TODO: all implementations are assuming a multiple of 2 number of blocks
// When using recursion/convolution then the you need 2^n blocks where n
// is the number of recursions

// If you add a new one, search for one of the other types and copy/paste
using Test8x1SharkParams = GenericSharkFloatParams<8, 1, SharkBatchSize, SharkTestIterCount>;
//using Test8x1SharkParams = GenericSharkFloatParams<13, 3, SharkBatchSize, SharkTestIterCount>;
using Test4x36SharkParams = GenericSharkFloatParams<4, 6, SharkBatchSize, SharkTestIterCount>;
using Test4x12SharkParams = GenericSharkFloatParams<5, 6, SharkBatchSize, SharkTestIterCount>;
using Test4x9SharkParams = GenericSharkFloatParams<4, 12, SharkBatchSize, SharkTestIterCount>;
using Test4x6SharkParams = GenericSharkFloatParams<4, 36, SharkBatchSize, SharkTestIterCount>;

//using Test128x128SharkParams = GenericSharkFloatParams<128, 128, SharkBatchSize, SharkTestIterCount>;
using Test128x63SharkParams = GenericSharkFloatParams<96, 81, SharkBatchSize, SharkTestIterCount>;
using Test64x63SharkParams = GenericSharkFloatParams<64, 81, SharkBatchSize, SharkTestIterCount>;
using Test32x63SharkParams = GenericSharkFloatParams<32, 81, SharkBatchSize, SharkTestIterCount>;
using Test16x63SharkParams = GenericSharkFloatParams<16, 81, SharkBatchSize, SharkTestIterCount>;

using Test128x36SharkParams = GenericSharkFloatParams<100, 81, SharkBatchSize, SharkTestIterCount>;
using Test128x18SharkParams = GenericSharkFloatParams<100, 36, SharkBatchSize, SharkTestIterCount>;
using Test128x9SharkParams = GenericSharkFloatParams<100, 12, SharkBatchSize, SharkTestIterCount>;
using Test128x3SharkParams = GenericSharkFloatParams<100, 6, SharkBatchSize, SharkTestIterCount>;

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
    HpSharkFloat &operator=(const HpSharkFloat<SharkFloatParams> &) = delete;

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
