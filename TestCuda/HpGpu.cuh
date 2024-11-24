#pragma once

#include <cuda_runtime.h>
#include <string>
#include <gmp.h>

#ifdef _DEBUG
static constexpr int32_t ThreadsPerBlock = 4;
static constexpr int32_t NumBlocks = 4;
static constexpr int32_t BatchSize = 8;
// static constexpr int32_t ThreadsPerBlock = 4;
// static constexpr int32_t NumBlocks = 4;
// static constexpr int32_t BatchSize = 4;
static constexpr int32_t LowPrec = 32;
static constexpr int32_t NUM_ITER = 1000;
static constexpr bool SkipCorrectnessTests = false;
constexpr bool Verbose = true;
#else
static constexpr int32_t ThreadsPerBlock = 64;
static constexpr int32_t NumBlocks = 64;
static constexpr int32_t BatchSize = 512;
static constexpr int32_t LowPrec = 32;
static constexpr int32_t NUM_ITER = 200;
static constexpr bool SkipCorrectnessTests = true;
constexpr bool Verbose = false;
#endif

// Struct to hold both integer and fractional parts of the high-precision number
struct HpGpu {
    HpGpu();
    HpGpu(uint32_t numDigits);
    HpGpu(const uint32_t *digitsIn, int32_t expIn, bool isNegative);
    ~HpGpu() = default;
    HpGpu &operator=(const HpGpu &) = delete;

    void CopyDeviceToHost(const HpGpu &other);
    std::string ToString() const;
    std::string ToHexString() const;
    void GenerateRandomNumber();

    // Fixed number of uint32_t values
    constexpr static int32_t NumUint32 = ThreadsPerBlock * NumBlocks;

    // Default precision in bits
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

std::string MpfToString(const mpf_t mpf_val, size_t precInBits);
std::string MpfToHexString(const mpf_t mpf_val);

void MpfToHpGpu(const mpf_t mpf_val, HpGpu &number, int prec_bits);
void HpGpuToMpf(const HpGpu &hpNum, mpf_t &mpf_val);
std::string Uint32ToMpf(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);