#pragma once

#include <cuda_runtime.h>
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
static constexpr auto SharkTestIterCount = 1000;
static constexpr auto SharkBatchSize = 8;
static constexpr bool SharkCorrectnessTests = false;
static constexpr bool SharkDebug = true;
#else
static constexpr auto SharkTestIterCount = 250;
static constexpr auto SharkBatchSize = 512;
static constexpr bool SharkCorrectnessTests = true;
static constexpr bool SharkDebug = false;
#endif

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
    static constexpr bool HostVerbose = SharkDebug;
    static constexpr bool DebugChecksums = true;

    static constexpr auto ConvolutionLimit = 81; // 3^whatevs = ConvolutionLimit
    static constexpr auto ConvolutionLimitPow = 4; // 3^whatevs = ConvolutionLimit

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

static constexpr auto ScratchMemoryCopies = 1024;
static constexpr auto ScratchMemoryArrays = 32;
// The 32 here is from DefineTempProductsOffsets
static constexpr auto Uint64ToAllocateForMultiply =
    ScratchMemoryCopies * ScratchMemoryArrays;
// Additional constants, 256-uint64 times number of "frames"
static constexpr auto AdditionalUInt64PerFrame = 256;
static constexpr auto Uint64AdditionalConstants = AdditionalUInt64PerFrame * ScratchMemoryCopies;

static constexpr int32_t LowPrec = 32;

#include "ExplicitInstantiate.h"



// TODO: all implementations are assuming a multiple of 2 number of blocks
// When using recursion/convolution then the you need 2^n blocks where n
// is the number of recursions

// If you add a new one, search for one of the other types and copy/paste
using Test8x1SharkParams = GenericSharkFloatParams<8, 1, SharkBatchSize, SharkTestIterCount>;
using Test4x36SharkParams = GenericSharkFloatParams<16, 39, SharkBatchSize, SharkTestIterCount>;
using Test4x12SharkParams = GenericSharkFloatParams<17, 36, SharkBatchSize, SharkTestIterCount>;
using Test4x9SharkParams = GenericSharkFloatParams<17, 39, SharkBatchSize, SharkTestIterCount>;
using Test4x6SharkParams = GenericSharkFloatParams<13, 48, SharkBatchSize, SharkTestIterCount>;

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
using TestCorrectnessSharkParams2 = Test4x12SharkParams;
using TestCorrectnessSharkParams3 = Test4x9SharkParams;
using TestCorrectnessSharkParams4 = Test4x36SharkParams;
using TestCorrectnessSharkParams5 = Test4x6SharkParams /*Test128x63SharkParams*/;

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
