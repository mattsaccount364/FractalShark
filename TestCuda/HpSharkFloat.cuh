#pragma once

#include <cuda_runtime.h>
#include <string>
#include <gmp.h>

template<
    int32_t pThreadsPerBlock,
    int32_t pNumBlocks,
    int32_t pBatchSize,
    int32_t pTestIterCount>
struct GenericSharkFloatParams {
    static constexpr int32_t ThreadsPerBlock = pThreadsPerBlock;
    static constexpr int32_t NumBlocks = pNumBlocks;
    static constexpr int32_t BatchSize = pBatchSize;
    static constexpr int32_t TestIterCount = pTestIterCount;
    // Fixed number of uint32_t values
    constexpr static int32_t NumUint32 = ThreadsPerBlock * NumBlocks;

    // If these are set to false they produce wrong answers but can be useful
    // to confirm source of performance issues.
    constexpr static bool DisableCarryPropagation = false;
    constexpr static bool DisableFinalConstruction = false;
};

static constexpr int32_t LowPrec = 32;

#ifdef _DEBUG
static constexpr auto TestIterCount = 1000;
static constexpr auto BatchSize = 8;
static constexpr bool SkipCorrectnessTests = false;
constexpr bool Verbose = true;
#else
static constexpr auto TestIterCount = 200;
static constexpr auto BatchSize = 512;
static constexpr bool SkipCorrectnessTests = true;
constexpr bool Verbose = false;
#endif

// If you add a new one, search for one of the other types and copy/paste
using Test4x2SharkParams = GenericSharkFloatParams<4, 2, BatchSize, TestIterCount>;
using Test4x4SharkParams = GenericSharkFloatParams<4, 4, BatchSize, TestIterCount>;
using Test8x1SharkParams = GenericSharkFloatParams<8, 1, BatchSize, TestIterCount>;

//using Test128x64SharkParams = GenericSharkFloatParams<128, 64, BatchSize, TestIterCount>;
using Test128x64SharkParams = GenericSharkFloatParams<8, 2, BatchSize, TestIterCount>;


#ifdef _DEBUG
using TestSharkParams = Test8x1SharkParams;
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
    constexpr static auto NumUint32 = SharkFloatParams::NumUint32;
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
std::string MpfToString (const mpf_t mpf_val, size_t precInBits);

std::string MpfToHexString (const mpf_t mpf_val);

template<class SharkFloatParams>
void MpfToHpGpu (const mpf_t mpf_val, HpSharkFloat<SharkFloatParams> &number, int prec_bits);

template<class SharkFloatParams>
void HpGpuToMpf (const HpSharkFloat<SharkFloatParams> &hpNum, mpf_t &mpf_val);

template<class SharkFloatParams>
std::string Uint32ToMpf (const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);