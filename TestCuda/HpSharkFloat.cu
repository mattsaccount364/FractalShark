
#include "HpSharkFloat.cuh"

#include <gmp.h>
#include <iostream>

#include <random>
#include <cstdint>
#include <assert.h>

//static_assert(sizeof(HpSharkFloat<SharkFloatParams>) == 4096, "HpSharkFloat<SharkFloatParams> size is not 4096 bytes");

template<class SharkFloatParams>
HpSharkFloat<SharkFloatParams>::HpSharkFloat()
    : Digits{},
    Exponent{ std::numeric_limits<ExpT>::min() },
    IsNegative{} {

    // exponent is most negative int32_t
}

//template<class SharkFloatParams>
//HpSharkFloat<SharkFloatParams>::HpSharkFloat(uint32_t numDigits)
//    : Digits{},
//    Exponent{ std::numeric_limits<ExpT>::min() },
//    IsNegative{} {
//
//    std::fill(Digits, Digits + NumUint32, 0);
//}

template<class SharkFloatParams>
HpSharkFloat<SharkFloatParams>::HpSharkFloat(
    const uint32_t *digitsIn,
    int32_t expIn,
    bool isNegative)
    : Digits{},
    Exponent{ expIn },
    IsNegative{ isNegative } {

    memcpy(Digits, digitsIn, sizeof(uint32_t) * NumUint32);
}

// Function to convert mpf_t to string
template<class SharkFloatParams>
std::string MpfToString<SharkFloatParams>(const mpf_t mpf_val, size_t precInBits) {
    char *str = NULL;

    if (precInBits == HpSharkFloat<SharkFloatParams>::DefaultPrecBits) {
        gmp_asprintf(&str, "%.Fe", mpf_val);
        std::string result(str);
        free(str);
        return result;
    } else {
        const auto decimalDigits = static_cast<uint32_t>(precInBits / HpSharkFloat<SharkFloatParams>::ConvertBitsToDecimals);
        gmp_asprintf(&str, "%.*Fe", decimalDigits, mpf_val);
        std::string result(str);
        free(str);
        return result;
    }
}

// typedef struct
// {
//   int _mp_prec;       // Max precision, in number of `mp_limb_t's.
//                       // Set by mpf_init and modified by
//                       // mpf_set_prec.  The area pointed to by the
//                       // _mp_d field contains `prec' + 1 limbs.
//   int _mp_size;       // abs(_mp_size) is the number of limbs the
//                       // last field points to.  If _mp_size is
//                       // negative this is a negative number.
//   mp_exp_t _mp_exp;   //  Exponent, in the base of `mp_limb_t'.
//   mp_limb_t *_mp_d;   //  Pointer to the limbs.
// } __mpf_struct;
// 
// typedef __mpf_struct mpf_t[1];
std::string MpfToHexString(const mpf_t mpf_val) {
    std::string result;
    mp_exp_t exponent;
    mp_limb_t *limbs = mpf_val[0]._mp_d;
    auto prec = mpf_val[0]._mp_prec;
    auto numLimbs = prec + 1;

    // First put a plus or minus depending on sign of _mp_size
    if (mpf_val[0]._mp_size < 0) {
        result += "-";
    } else {
        result += "+";
    }

    // Convert each limb to hex and append to result
    for (int i = 0; i < numLimbs; ++i) {
        char buffer[32];

        // Break lim into two 32-bit values and output individually, low then high
        uint32_t lowOrder = limbs[i] & 0xFFFFFFFF;
        uint32_t highOrder = limbs[i] >> 32;

        snprintf(buffer, sizeof(buffer), "0x%08X 0x%08X ", lowOrder, highOrder);
        result += buffer;
    }

    // Finally append exponent
    exponent = mpf_val[0]._mp_exp;
    result += "2^64^(0n" + std::to_string(exponent) + ")";

    return result;
}

template<class IntT>
std::string
UintArrayToHexString(const IntT *array, size_t numElements) {

    std::string result;

    // Append numElements
    result += "Len:" + std::to_string(numElements) + ", ";

    // Convert each 4-byte integer to hex and append to result
    for (size_t i = 0; i < numElements; ++i) {
        char buffer[64];

        if constexpr (sizeof(IntT) == 4) {
            snprintf(buffer, sizeof(buffer), "0x%08x ", static_cast<uint32_t>(array[i]));
        } else if constexpr (sizeof(IntT) == 8) {
            snprintf(buffer, sizeof(buffer), "0x%016llx ", static_cast<uint64_t>(array[i]));
        } else {
            static_assert(false, "Unsupported size");
        }

        result += buffer;
    }

    return result;
}

template<class IntT>
std::string
UintToHexString(IntT val) {

    char buffer[32];

    if constexpr (sizeof(IntT) == 4) {
        snprintf(buffer, sizeof(buffer), "0x%08X", static_cast<uint32_t>(val));
    } else if constexpr (sizeof(IntT) == 8) {
        snprintf(buffer, sizeof(buffer), "0x%016llX", static_cast<uint64_t>(val));
    } else {
        static_assert(false, "Unsupported size");
    }

    return buffer;
}

template<class IntT>
std::string
VectorUintToHexString(const std::vector<IntT> &arr) {

    return UintArrayToHexString<IntT>(arr.data(), arr.size());
}

template<class IntT>
std::string
VectorUintToHexString(const IntT *arr, size_t numElements) {

    return UintArrayToHexString<IntT>(arr, numElements);
}

// Explicitly instantiate
#define ExplicitlyInstantiateUintArrayToHexString(IntT) \
    template std::string UintArrayToHexString<IntT>(const IntT *array, size_t numElements); \
    template std::string VectorUintToHexString<IntT>(const std::vector<IntT> &arr); \
    template std::string VectorUintToHexString<IntT>(const IntT *arr, size_t numElements); \
    template std::string UintToHexString<IntT>(IntT val);

ExplicitlyInstantiateUintArrayToHexString(uint32_t);
ExplicitlyInstantiateUintArrayToHexString(uint64_t);

ExplicitlyInstantiateUintArrayToHexString(int32_t);
ExplicitlyInstantiateUintArrayToHexString(int64_t);


template<class SharkFloatParams>
static HpSharkFloat<SharkFloatParams>::ExpT
MpirExponentToHPExponent(
    const mpf_t mpf_val,
    typename HpSharkFloat<SharkFloatParams>::ExpT bytesToCopy) {

    const auto mpirExponentInPow2 = static_cast<HpSharkFloat<SharkFloatParams>::ExpT>(mpf_val[0]._mp_exp * sizeof(mp_limb_t) * 8);
    return mpirExponentInPow2 - bytesToCopy * 8;
}

template<class SharkFloatParams>
mp_exp_t
HpSharkFloat<SharkFloatParams>::HpGpuExponentToMpfExponent(
    size_t numBytesToCopy) const {

    const auto hpExponentInPow2 = static_cast<mp_exp_t>(Exponent + numBytesToCopy * 8);
    return hpExponentInPow2 / (sizeof(mp_limb_t) * 8);
}

template<class SharkFloatParams>
HpSharkFloat<SharkFloatParams> &
HpSharkFloat<SharkFloatParams>::operator= (
    const HpSharkFloat<SharkFloatParams> &other)
{
    if (this != &other) {
        memcpy(Digits, other.Digits, sizeof(uint32_t) * NumUint32);
        Exponent = other.Exponent;
        IsNegative = other.IsNegative;
    }

    return *this;
}



// Function to convert mpf_t to HpSharkFloat<SharkFloatParams>
template<class SharkFloatParams>
void
MpfToHpGpu(
    const mpf_t mpf_value,
    HpSharkFloat<SharkFloatParams> &number,
    int prec_bits) {

    // Get the absolute value of mpf_value
    mpf_t abs_val;
    mpf_init2(abs_val, HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
    mpf_abs(abs_val, mpf_value);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "abs_val: " << MpfToString<SharkFloatParams>(abs_val, prec_bits) << std::endl;
    }

    // Determine the sign
    number.IsNegative = (mpf_sgn(mpf_value) < 0);

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "prec_bits: " << std::dec << prec_bits << std::endl;
    }

    std::vector<uint32_t> data;
    const auto absMpirSize = std::abs(abs_val[0]._mp_size);
    const auto precInUint64 = std::min(mpf_value[0]._mp_prec + 1, absMpirSize);

    // Iterate over mpf_value._m_d and copy the data.
    // Put the low order uint32_t first, then the high order uint32_t
    // Keep the endian the same
    for (auto i = 0; i < precInUint64; ++i) {
        const uint32_t lowOrder = mpf_value[0]._mp_d[i] & 0xFFFFFFFF;
        data.push_back(lowOrder);
        const uint32_t highOrder = mpf_value[0]._mp_d[i] >> 32;
        data.push_back(highOrder);
    }

    // ----------------------------------------------------------------
    // Inline normalization lambdas:
    // ----------------------------------------------------------------

    int32_t N = (int32_t)data.size();

    // 1) count leading zeros in a 32-bit word
    auto countLZ32 = [&](uint32_t x) {
        int32_t c = 0;
        for (int32_t b = 31; b >= 0; --b) {
            if (x & (1u << b)) break;
            ++c;
        }
        return c;
        };

    // 2) shift one 32-bit word of the array left by L bits:
    auto shiftLeftWord = [&](int32_t L, int32_t idx) -> uint32_t {
        const int32_t shiftWords = L / 32;
        const int32_t shiftBitsMod = L % 32;
        int32_t srcIdx = idx - shiftWords;

        uint32_t lower = (srcIdx >= 0 && srcIdx < N)
            ? data[srcIdx] : 0;
        uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < N)
            ? data[srcIdx - 1] : 0;

        if (shiftBitsMod == 0) {
            return lower;
        } else {
            return (lower << shiftBitsMod)
                | (upper >> (32 - shiftBitsMod));
        }
        };

    // 3) shift the entire data[] vector left by L bits
    auto shiftLeftArray = [&](int32_t L) {
        std::vector<uint32_t> tmp(N);
        for (int32_t i = 0; i < N; ++i) {
            tmp[i] = shiftLeftWord(L, i);
        }
        data = std::move(tmp);
        };

    // 3) normalize so the top-most 1 ends up in bit (N*32-1)
    //    returns how many bits we shifted
    auto normalizeData = [&]() {
        // find highest nonzero limb
        int32_t msd = -1;
        for (int32_t i = N - 1; i >= 0; --i) {
            if (data[i] != 0) { msd = i; break; }
        }
        if (msd < 0) return 0;  // all zero --> no shift

        // count leading zeros in that limb
        int32_t lz = countLZ32(data[msd]);
        int32_t bitIndex = msd * 32 + (31 - lz);
        int32_t target = N * 32 - 1;
        int32_t shiftL = target - bitIndex;
        if (shiftL > 0) shiftLeftArray(shiftL);
        return shiftL;
        };

    // run normalization on the raw data[]
    auto dataCopy = data; // keep a copy for debugging
    int32_t shiftBits = normalizeData();

    static_assert(sizeof(mp_limb_t) == sizeof(uint64_t), "mp_limb_t is not 64 bits");

    {
        if constexpr (SharkFloatParams::HostVerbose) {
            auto originalDataStr = VectorUintToHexString(dataCopy);
            auto shiftedDataStr = VectorUintToHexString(data);
            std::cout << "Original data: " << originalDataStr << std::endl;
            std::cout << "Shifted data: " << shiftedDataStr << ", shiftBits: " << shiftBits << std::endl;
            std::cout << "Shift bits: " << shiftBits << std::endl;
        }
    }

    auto countInBytes = N * sizeof(uint32_t);

    // Copy data into digits array
    memset(number.Digits, 0, SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));

    // If count is greater than NumUint32, move forward by count - NumUint32
    // because the most significant digits are at the end of the array
    auto startOffset =
        (countInBytes > SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t)) ?
        countInBytes - SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t) :
        0;
    auto numBytesToCopy = std::min(
        countInBytes,
        SharkFloatParams::GlobalNumUint32 * sizeof(uint32_t));
    numBytesToCopy = std::min(numBytesToCopy, absMpirSize * sizeof(mp_limb_t));

    memcpy(number.Digits, reinterpret_cast<uint8_t *>(data.data()) + startOffset, numBytesToCopy);

    // Set the Exponent
    //number.Exponent = MpirExponentToHPExponent<SharkFloatParams>(
    //    mpf_value, static_cast<HpSharkFloat<SharkFloatParams>::ExpT>(numBytesToCopy));

    // set exponent from MPIR *then* subtract our normalization shift
    number.Exponent = MpirExponentToHPExponent<SharkFloatParams>(
        mpf_value,
        static_cast<typename HpSharkFloat<SharkFloatParams>::ExpT>(numBytesToCopy)
    ) - shiftBits;

    //{
    //    auto exportedFinalData = Uint32ArrayToHexString(number.Digits, SharkFloatParams::GlobalNumUint32);
    //    std::cout << "Exported Final data: " << exportedFinalData << ", exponent: " << number.Exponent << std::endl;
    //}

    //mpf_clear(scaled_val);
    mpf_clear(abs_val);
}

template<class SharkFloatParams>
std::string
HpSharkFloat<SharkFloatParams>::ToHexString() const {

    std::string result;

    // First append a sign depending on IsNegative
    if (IsNegative) {
        result += "-";
    } else {
        result += "+";
    }

    for (size_t i = 0; i < NumUint32; ++i) {
        // Convert each 4-byte integer to hex and append to result
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "0x%08X ", Digits[i]);
        result += buffer;
    }

    // Finally append exponent
    result += "2^(0n" + std::to_string(Exponent) + ")";

    return result;
}

template<class SharkFloatParams>
std::string
HpSharkFloat<SharkFloatParams>::ToString() const
{
    mpf_t mpf_value;
    mp_bitcnt_t prec = HpSharkFloat<SharkFloatParams>::DefaultMpirBits;
    mpf_init2(mpf_value, prec);

    // Import the digits into mpf_value.  Convert
    // from uint32_t to mp_limb_t, which is uint64_t
    static_assert(sizeof(mp_limb_t) == sizeof(uint64_t), "mp_limb_t is not 64 bits");
    
    // Number of 64-bit limbs = ceil(NumUint32 / 2).
    constexpr size_t limbCount = (NumUint32 + 1) / 2;

    for (size_t i = 0; i < limbCount; ++i) {
        // Store two uint32_t values in one
        auto lowPart = Digits[2 * i];
        uint64_t highPart = 0;
        if (2 * i + 1 < NumUint32) {
            highPart = Digits[2 * i + 1];
        }

        auto value = (highPart << 32) | lowPart;
        mpf_value[0]._mp_d[i] = value;
    }

    The MPIR representation requires a power of 2 exponent so we need to deal with that.

    mpf_value[0]._mp_exp = HpGpuExponentToMpfExponent(NumUint32 * sizeof(uint32_t));
    mpf_value[0]._mp_size = NumUint32 / 2;
    if (IsNegative) {
        mpf_value[0]._mp_size = -mpf_value[0]._mp_size;
    }

    // Use gmp_printf
    char *str = NULL;
    gmp_asprintf(&str, "%.Fe", mpf_value);
    std::string result(str);
    free(str);

    // Clear GMP variables
    mpf_clear(mpf_value);

    return result;
}

template<class SharkFloatParams>
void
HpSharkFloat<SharkFloatParams>::GenerateRandomNumber()
{
    // Use a random device to seed the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());  // Mersenne Twister for high-quality randomness

    std::uniform_int_distribution<uint32_t> distributionCases(0, 4);
    std::uniform_int_distribution<uint32_t> distributionSmall(0, 16);
    std::uniform_int_distribution<uint32_t> distributionRand(0, std::numeric_limits<uint32_t>::max());

    // Fill uint32_t Digits[NumUint32] with completely random numbers
    for (size_t i = 0; i < NumUint32; ++i) {
        switch (distributionCases(generator)) {
        case 0:
            Digits[i] = distributionRand(generator);
            break;
        case 1:
            Digits[i] = std::numeric_limits<uint32_t>::min();
            break;
        case 2:
            Digits[i] = std::numeric_limits<uint32_t>::max();
            break;
        case 3:
            Digits[i] = std::numeric_limits<uint32_t>::max() - distributionSmall(generator);
            break;
        case 4:
            Digits[i] = std::numeric_limits<uint32_t>::min() + distributionSmall(generator);
            break;
        default:
            i--;
            assert(false);
            continue;
        }
    }

    // Ensure the most significant bit is not set
    // Digits[NumUint32 - 1] &= 0x7FFFFFFF;

    // Generate random Exponent within range of DefaultPrecBits
    const auto PrecBitsInt = static_cast<int>(DefaultPrecBits);
    std::uniform_int_distribution<int> exp_distribution(
        -PrecBitsInt * 4, // arbitrary, not related to int size
        PrecBitsInt * 4);
    //Exponent = exp_distribution(generator);
    Exponent = 0;

    // Random boolean for IsNegative
    std::bernoulli_distribution bool_distribution(0.5);
    IsNegative = bool_distribution(generator);
}

template<class SharkFloatParams>
void
HpSharkFloat<SharkFloatParams>::GenerateRandomNumber2() {
    // Use a random device to seed the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());  // Mersenne Twister for high-quality randomness

    std::uniform_int_distribution<uint32_t> distributionCases(0, 3);
    std::uniform_int_distribution<uint32_t> distributionSmall(0, 16);
    std::uniform_int_distribution<uint32_t> distributionRand(0, std::numeric_limits<uint32_t>::max());

    mpf_t mpf_value;
    // Initialize an mpf_t variable
    mpf_init2(mpf_value, DefaultPrecBits);
    mpf_set_d(mpf_value, 1.0);
    mpf_div_ui(mpf_value, mpf_value, distributionRand(generator) + 1);
    //mpf_sqrt(mpf_value, mpf_value);

    // Convert MPF to HpSharkFloat<SharkFloatParams>
    MpfToHpGpu<SharkFloatParams>(mpf_value, *this, DefaultPrecBits);

    mpf_clear(mpf_value);

    // Random boolean for IsNegative
    std::bernoulli_distribution bool_distribution(0.5);
    IsNegative = bool_distribution(generator);
}

template<class SharkFloatParams>
void
HpSharkFloat<SharkFloatParams>::Negate()
{
    IsNegative = !IsNegative;
}

template<class SharkFloatParams>
void
HpSharkFloat<SharkFloatParams>::DeepCopySameDevice(
    const HpSharkFloat<SharkFloatParams> &other)
{
    memcpy(Digits, other.Digits, sizeof(uint32_t) * NumUint32);
    Exponent = other.Exponent;
    IsNegative = other.IsNegative;
}

// Function to convert HpSharkFloat<SharkFloatParams> to mpf_t
template<class SharkFloatParams>
void
HpGpuToMpf (
    const HpSharkFloat<SharkFloatParams> &hpNum,
    mpf_t &mpf_val)
{
    // Initialize an mpz_t integer to hold the significand
    mpz_t mpz_value;
    mpz_init(mpz_value);

    // Import the digits array into the mpz_t integer
    // Note: mpz_import expects the least significant word first
    mpz_import(mpz_value, SharkFloatParams::GlobalNumUint32, -1, sizeof(uint32_t), 0, 0, hpNum.Digits);

    // Adjust for sign
    if (hpNum.IsNegative) {
        mpz_neg(mpz_value, mpz_value);
    }

    // Set mpf_value to mpz_value
    mpf_set_z(mpf_val, mpz_value);

    // Adjust the exponent
    // Since the exponent is in base 2^32, adjust by multiplying/dividing by 2^(exponent * 32)
    int64_t total_bits = (int64_t)(hpNum.Exponent); // Exponent is already in bits
    if (total_bits > 0) {
        mpf_mul_2exp(mpf_val, mpf_val, (mp_bitcnt_t)total_bits);
    } else if (total_bits < 0) {
        mpf_div_2exp(mpf_val, mpf_val, (mp_bitcnt_t)(-total_bits));
    }
    // If exponent is zero, no adjustment needed

    // Clean up
    mpz_clear(mpz_value);
}

// Function to convert uint32_t array to mpf_t
// pow2Exponent is the exponent in base 2
template<class SharkFloatParams>
std::string
Uint32ToMpf (
    const uint32_t *array,
    int32_t pow64Exponent,
    mpf_t &mpf_val) {

    std::vector<uint64_t> data;
    for (size_t i = 0; i < SharkFloatParams::HalfLimbsRoundedUp; ++i) {
        // Store two uint32_t values in one
        auto value = static_cast<uint64_t>(array[2 * i + 1]) << 32 | array[2 * i];
        data.push_back(value);
    }

    const size_t spaceAvailable = mpf_val[0]._mp_prec + 1;
    const auto entriesToCopy = std::min(data.size(), spaceAvailable);

    // Copy the data into mpf_value[0]._mp_d
    for (size_t i = 0; i < entriesToCopy; ++i) {
        mpf_val[0]._mp_d[i] = data[i];
    }

    // Set the exponent
    mpf_val[0]._mp_exp = pow64Exponent;

    // Set the size accoring to the number of entries copied
    mpf_val[0]._mp_size = static_cast<int>(entriesToCopy);

    // Return string representation
    return MpfToString<SharkFloatParams>(mpf_val, HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
}

// Explicit instantiation
#define ExplicitlyInstantiate(SharkFloatParams) \
    template class HpSharkFloat<SharkFloatParams>; \
    template void MpfToHpGpu<SharkFloatParams>(const mpf_t mpf_val, HpSharkFloat<SharkFloatParams> &number, int prec_bits); \
    template void HpGpuToMpf<SharkFloatParams>(const HpSharkFloat<SharkFloatParams> &hpNum, mpf_t &mpf_val); \
    template std::string Uint32ToMpf<SharkFloatParams>(const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val); \
    template std::string MpfToString<SharkFloatParams>(const mpf_t mpf_val, size_t precInBits); \


ExplicitInstantiateAll();