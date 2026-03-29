#pragma once

// Template method bodies for HpSharkFloat: ToHDRFloat and FromHDRFloat.
// Included from HpSharkFloat.h after the HpSharkFloat class definition.
// Do not include this header directly — include HpSharkFloat.h instead.

template <class SharkFloatParams>
template <class SubType>
CUDA_CRAP_BOTH HDRFloat<SubType>
HpSharkFloat<SharkFloatParams>::ToHDRFloat(int32_t extraExp) const
{
    static_assert(std::is_same_v<SubType, float> ||
                  std::is_same_v<SubType, double> ||
                  std::is_same_v<SubType, CudaDblflt<dblflt>>,
                  "ToHDRFloat: SubType must be float, double, or CudaDblflt<dblflt>");

    // CLZ helpers (host/device friendly)
    auto clz32 = [](uint32_t v) -> int {
#if defined(__CUDA_ARCH__)
        return __clz(v);
#else
        // std::countl_zero is constexpr in C++20 and returns 32 for x==0
        return static_cast<int>(std::countl_zero(v));
#endif
    };

    // ------------- Single reverse scan to find the highest non-zero limb -------------
    constexpr int N = SharkFloatParams::GlobalNumUint32;

    int hiIdx = -1;
    for (int i = N - 1; i >= 0; --i) {
        if (Digits[i] != 0u) {
            hiIdx = i;
            break;
        }
    }

    // Zero fast-path
    if (hiIdx < 0) {
        return HDRFloat<SubType>{};
    }

    // Form a 64-bit "window" from the top two 32-bit limbs.
    const uint32_t hi32 = Digits[hiIdx];
    const uint32_t lo32 = (hiIdx > 0) ? Digits[hiIdx - 1] : 0u;
    const uint64_t window64 = (uint64_t(hi32) << 32) | uint64_t(lo32);

    // Absolute (unbiased) binary exponent p = floor(log2(|this|))
    const int msb32 = 31 - clz32(hi32);
    const int32_t p = hiIdx * 32 + msb32;
    const int msbInWindow = 32 + msb32;

    // Combine with any external binary scaling tracked by the caller/type.
    const int32_t finalExp = p + extraExp + Exponent;

    if constexpr (std::is_same_v<SubType, CudaDblflt<dblflt>>) {
        // Convert via double, then construct HDRFloat<CudaDblflt> from HDRFloat<double>.
        // The HDRFloat constructor handles double → CudaDblflt conversion.
        double mant_d = static_cast<double>(window64) /
                        static_cast<double>(1ull << msbInWindow);
        if (IsNegative)
            mant_d = -mant_d;
        HDRFloat<double> temp(finalExp, mant_d);
        HdrReduce(temp);
        return HDRFloat<CudaDblflt<dblflt>>(temp);
    } else {
        // Normalize mantissa into [1,2): m = window64 / 2^(msbInWindow)
        SubType mant = SubType(window64) / std::ldexp(SubType(1), msbInWindow);
        if (IsNegative)
            mant = -mant;

        HDRFloat<SubType> out(finalExp, mant);
        HdrReduce(out);
        return out;
    }
}

// Build an HpSharkFloat<Params> from HDRFloat<SubType> (SubType = double or float).
// Strategy:
//   1) Reduce HDR so mantissa ∈ [1,2); get its (unbiased) exp.
//   2) Put the MSB at the very top of the highest limb: hiIdx = N-1, msb32 = 31.
//   3) Fill the top TWO limbs with a 64-bit "window" equal to mantissa * 2^63.
//   4) Choose HpSharkFloat::Exponent so the overall value matches HDR.
//
// Representation assumed:
//   Value = (-1)^sign * (sum_i Digits[i] * 2^(32*i)) * 2^(Exponent).
//   Digits[0] is least-significant; Digits[N-1] is most-significant.
//   Assumes incoming value is reduced/normalized
template <class SharkFloatParams>
template <class SubType>
void
HpSharkFloat<SharkFloatParams>::FromHDRFloat(const HDRFloat<SubType> &h)
{
    static_assert(std::is_same<SubType, double>::value || std::is_same<SubType, float>::value,
                  "FromHDRFloat: SubType must be double or float");

    constexpr int N = HpSharkFloat<SharkFloatParams>::NumUint32;

    // Zero digits
    for (int i = 0; i < N; ++i)
        Digits[i] = HpSharkFloat<SharkFloatParams>::DigitType{0};

    // Zero fast-path
    const bool isZero = (h.mantissa == SubType(0));
    if (isZero) {
        SetNegative(false);
        Exponent = 0;
        return;
    }

    // Sign
    const bool neg = (h.mantissa < SubType(0));
    const double mantAbs = static_cast<double>(neg ? -h.mantissa : h.mantissa);
    const int32_t exp = static_cast<int32_t>(h.exp);

    // We place the absolute MSB at the very top bit of the highest 32-bit limb.
    // That bit index (absolute, counting from bit 0 of Digits[0]) is:
    //   msb_abs = (N-1)*32 + 31
    // Inside the 64-bit window that spans the top TWO limbs, that MSB is at bit 63.
    const int32_t msb_abs = (static_cast<int32_t>(N) - 1) * 32 + 31;

    // Build the 64-bit window = mantAbs * 2^63   (so the MSB of the window is bit 63).
    // Clamp to [2^63, 2^64-1] to avoid rounding overflow on (almost) 2.0 mantissas.
    double scaled = std::ldexp(mantAbs, 63);
    if (scaled < std::ldexp((double)1.0, 63)) {
        scaled = std::ldexp((double)1.0, 63);
    }
    const double max64 = std::ldexp((double)1.0, 64) - 1.0;
    if (scaled > max64)
        scaled = max64;

    const uint64_t window64 = static_cast<uint64_t>(scaled);

    // Split into top two limbs (be careful if N==1).
    const uint32_t hi32 = static_cast<uint32_t>(window64 >> 32);
    const uint32_t lo32 = static_cast<uint32_t>(window64 & 0xFFFF'FFFFu);

    // Write high-order digits
    Digits[N - 1] = static_cast<HpSharkFloat<SharkFloatParams>::DigitType>(hi32);
    if constexpr (N >= 2) {
        Digits[N - 2] = static_cast<HpSharkFloat<SharkFloatParams>::DigitType>(lo32);
    }

    // Choose Hp exponent so overall value equals mantissa * 2^exp:
    //   Value = (DigitsAsInt) * 2^(Exponent)
    // where DigitsAsInt has MSB at msb_abs.
    // Since HDR value's msb is at absolute bit index 'exp', set:
    Exponent = exp - msb_abs;

    SetNegative(neg);
}
