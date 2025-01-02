#pragma once

#include <cuda_runtime.h>
#include <string>
#include <gmp.h>
#include <vector>

static constexpr bool UseCustomStream = true;
static constexpr bool UseSharedMemory = true;

#ifdef _DEBUG
static constexpr auto TestIterCount = 1000;
static constexpr auto BatchSize = 8;
static constexpr bool SkipCorrectnessTests = false;
static constexpr bool Verbose = true;
#else
static constexpr auto TestIterCount = 5000;
static constexpr auto BatchSize = 512;
static constexpr bool SkipCorrectnessTests = true;
static constexpr bool Verbose = false;
#endif

template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pBatchSize,
    int32_t pTestIterCount>
struct GenericSharkFloatParams {
    static constexpr int32_t GlobalThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t GlobalNumBlocks = pNumBlocks;
    static constexpr int32_t BatchSize = pBatchSize;
    static constexpr int32_t TestIterCount = pTestIterCount;
    // Fixed number of uint32_t values
    static constexpr int32_t GlobalNumUint32 = GlobalThreadsPerBlock * GlobalNumBlocks;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    static constexpr bool DisableAllAdditions = false;
    static constexpr bool DisableSubtraction = false;
    static constexpr bool DisableCarryPropagation = false;
    static constexpr bool DisableFinalConstruction = false;
    static constexpr bool ForceNoOp = false;
    static constexpr bool HostVerbose = Verbose;

    static std::string GetDescription() {
        std::string desc = "GlobalThreadsPerBlock: " + std::to_string(GlobalThreadsPerBlock) +
            ", GlobalNumBlocks: " + std::to_string(GlobalNumBlocks) +
            ", BatchSize: " + std::to_string(BatchSize) +
            ", TestIterCount: " + std::to_string(TestIterCount);
        return desc;
    }

    using GetHalf = typename std::conditional_t<
        (GlobalThreadsPerBlock > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock / 2, GlobalNumBlocks, BatchSize, TestIterCount>,
        typename std::conditional_t<(GlobalNumBlocks > 2),
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks / 2, BatchSize, TestIterCount>,
        GenericSharkFloatParams<GlobalThreadsPerBlock, GlobalNumBlocks, BatchSize, TestIterCount>
        >
    >;
};

static constexpr int32_t LowPrec = 32;

// If you add a new one, search for one of the other types and copy/paste
using Test4x2SharkParams = GenericSharkFloatParams<4, 2, BatchSize, TestIterCount>;
using Test4x4SharkParams = GenericSharkFloatParams<4, 4, BatchSize, TestIterCount>;
using Test8x1SharkParams = GenericSharkFloatParams<8, 1, BatchSize, TestIterCount>;
using Test8x8SharkParams = GenericSharkFloatParams<8, 8, BatchSize, TestIterCount>;
using Test16x16SharkParams = GenericSharkFloatParams<16, 4, BatchSize, TestIterCount>;

using Test128x64SharkParams = GenericSharkFloatParams<128, 64, BatchSize, TestIterCount>;
using Test64x64SharkParams = GenericSharkFloatParams<64, 64, BatchSize, TestIterCount>;
using Test32x64SharkParams = GenericSharkFloatParams<32, 64, BatchSize, TestIterCount>;
using Test16x64SharkParams = GenericSharkFloatParams<16, 64, BatchSize, TestIterCount>;

using Test128x32SharkParams = GenericSharkFloatParams<128, 32, BatchSize, TestIterCount>;
using Test128x16SharkParams = GenericSharkFloatParams<128, 16, BatchSize, TestIterCount>;
using Test128x8SharkParams = GenericSharkFloatParams<128, 8, BatchSize, TestIterCount>;
using Test128x4SharkParams = GenericSharkFloatParams<128, 4, BatchSize, TestIterCount>;


#ifdef _DEBUG
using TestSharkParams = Test4x4SharkParams;
//using TestSharkParams = Test8x1SharkParams;
#else
using TestSharkParams = Test128x64SharkParams;
#endif

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
