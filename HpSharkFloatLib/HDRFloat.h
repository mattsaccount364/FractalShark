#pragma once

#include <stdint.h>
#include <math.h>
#include <type_traits>
#include <fstream>
#include <format>
#include <algorithm>

#include "HighPrecision.h"
#include "CudaDblflt.h"

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#define CudaHostMax max
#define CudaHostMin min
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#define CudaHostMax std::max
#define CudaHostMin std::min
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

template<class T>
class HDRFloatComplex;

template<class T>
class FloatComplex;

CUDA_CRAP void InitStatics();

//#ifndef __CUDA_ARCH__
//#define MYALIGN __declspec(align(4))
//#else
//#define MYALIGN __align__(4)
//#endif

#define MYALIGN

enum class HDROrder {
    Left,
    Right
};

namespace Imagina {
    using HRReal = HDRFloat<double, HDROrder::Left, int64_t>;
    using SRReal = double;
} // namespace Imagina

template<class T, class TExp = int32_t>
class GenericHdrBase {
public:
    static CUDA_CRAP constexpr TExp MIN_BIG_EXPONENT() {
        if constexpr (
            std::is_same<TExp, int32_t>::value ||
            std::is_same<TExp, float>::value) {
            return INT32_MIN >> 3;
        } else {
            return INT16_MIN >> 3;
        }
    }
};

template<class T, class TExp = int32_t>
class LMembers : public GenericHdrBase<T, TExp> {
public:
    using GenericHdrBase = GenericHdrBase<T, TExp>;
    CUDA_CRAP LMembers() : mantissa(T{}), exp(GenericHdrBase::MIN_BIG_EXPONENT()) {}

    MYALIGN T mantissa;
    MYALIGN TExp exp;
};

template<class T, class TExp = int32_t>
class RMembers : public GenericHdrBase<T, TExp> {
public:
    using GenericHdrBase = GenericHdrBase<T, TExp>;
    CUDA_CRAP RMembers() : exp(GenericHdrBase::MIN_BIG_EXPONENT()), mantissa(T{}) {}

    MYALIGN TExp exp;
    MYALIGN T mantissa;
};

// TExp can work with float, int16_t or int64_t but all seem to offer worse perf
// float is quite close though.
//#ifndef __CUDA_ARCH__
template<class T, HDROrder Order = HDROrder::Left, class TExp = int32_t>
class HDRFloat : public std::conditional<Order == HDROrder::Left, LMembers<T, TExp>, RMembers<T, TExp>>::type {
    //#else
    //template<class T, class TExp = int32_t>
    //class __align__(8) HDRFloat {
    //#endif
public:
    typedef T TemplateSubType;
    using Base = std::conditional<Order == HDROrder::Left, LMembers<T, TExp>, RMembers<T, TExp>>::type;

    static CUDA_CRAP constexpr TExp MIN_SMALL_EXPONENT_FLOAT() {
        return -127;
    }

    static CUDA_CRAP constexpr TExp MIN_SMALL_EXPONENT_DOUBLE() {
        return -1023;
    }

    static CUDA_CRAP constexpr int32_t MIN_SMALL_EXPONENT_INT_FLOAT() {
        return -127;
    }

    static CUDA_CRAP constexpr int32_t MIN_SMALL_EXPONENT_INT_DOUBLE() {
        return -1023;
    }

public:
    using GenericHdrBase = GenericHdrBase<T, TExp>;

    static constexpr TExp EXPONENT_DIFF_IGNORED = 120;
    static constexpr TExp MINUS_EXPONENT_DIFF_IGNORED = -EXPONENT_DIFF_IGNORED;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    static constexpr int MaxFloatExponent = 127;
    static constexpr int MinFloatExponent = -126;

    template<bool IntegerOutput>
    std::string ToString() const {
        constexpr bool isDblFlt =
            std::is_same<T, CudaDblflt<dblflt>>::value ||
            std::is_same<T, CudaDblflt<MattDblflt>>::value;

        std::string result;
        if constexpr (!IntegerOutput) {
            if constexpr (!isDblFlt) {
                result += std::format("mantissa: {} exp: {}", static_cast<double>(Base::mantissa), Base::exp);
            } else {
                result += Base::mantissa.ToString<IntegerOutput>();
                result += std::format(" exp: {}", Base::exp);
            }
        } else {
            if constexpr (!isDblFlt) {
                const double res = static_cast<double>(Base::mantissa);
                const uint64_t mantissaInteger = *reinterpret_cast<const uint64_t*>(&res);
                const uint64_t localExp = Base::exp;
                const uint64_t expInteger = *reinterpret_cast<const uint64_t*>(&localExp);
                result += std::format("mantissa: 0x{:x} exp: 0x{:x}", mantissaInteger, expInteger);
            } else {
                result += Base::mantissa.ToString<IntegerOutput>();

                uint64_t tempExp = Base::exp;
                result += std::format(" exp: 0x{:x}", tempExp);
            }
        }

        return result;
    }

    template<bool IntegerOutput>
    void FromIStream(std::istream &is) {
        constexpr bool isDblFlt =
            std::is_same<T, CudaDblflt<dblflt>>::value ||
            std::is_same<T, CudaDblflt<MattDblflt>>::value;

        std::string tempstr;
        if constexpr (!IntegerOutput) {
            if constexpr (!isDblFlt) {
                is >> tempstr >> Base::mantissa >> tempstr >> Base::exp;
            } else {
                Base::mantissa.FromIStream<IntegerOutput>(is);
                is >> tempstr >> Base::exp;
            }
        } else {
            if constexpr (!isDblFlt) {
                uint64_t mantissaInteger;
                uint64_t expInteger;
                is >> tempstr >> std::hex >> mantissaInteger;
                is >> tempstr >> std::hex >> expInteger;
                Base::mantissa = static_cast<T>(*reinterpret_cast<const double *>(&mantissaInteger));
                Base::exp = static_cast<TExp>(*reinterpret_cast<const uint64_t *>(&expInteger));
            } else {
                Base::mantissa.FromIStream<IntegerOutput>(is);

                uint64_t expInteger;
                is >> tempstr >> std::hex >> expInteger;
                Base::exp = static_cast<TExp>(*reinterpret_cast<const uint64_t *>(&expInteger));
            }
        }
    }

    CUDA_CRAP constexpr HDRFloat() {
        Base::mantissa = T{};
        Base::exp = GenericHdrBase::MIN_BIG_EXPONENT();
    }

    CUDA_CRAP constexpr explicit HDRFloat(T mant) {
        Base::mantissa = mant;
        Base::exp = 0;

        HdrReduce(*this);
    }

    // Copy constructor
    CUDA_CRAP constexpr HDRFloat(const HDRFloat &other) :
        Base(other) {
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat(const HDRFloat<T, OtherOrder> &other) {
        Base::mantissa = other.mantissa;
        Base::exp = other.exp;
    }

    //template<HDROrder OtherOrder, typename OtherTExp>
    //CUDA_CRAP constexpr explicit HDRFloat(const HDRFloat<T, OtherOrder, OtherTExp> &other) {
    //    Base::mantissa = other.mantissa;
    //    Base::exp = other.exp;
    //}

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat(HDRFloat<T, OtherOrder> &&other) {
        Base::mantissa = other.mantissa;
        Base::exp = other.exp;
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &operator=(const HDRFloat<T, OtherOrder> &other) {
        Base::mantissa = other.mantissa;
        Base::exp = other.exp;
        return *this;
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &operator=(HDRFloat<T, OtherOrder> &&other) {
        Base::mantissa = other.mantissa;
        Base::exp = other.exp;
        return *this;
    }

    template<class SrcT, HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat(const HDRFloat<SrcT, OtherOrder> &other) {
        Base::mantissa = T(other.mantissa);
        Base::exp = other.exp;
    }

    template<class SrcT, HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat(HDRFloat<SrcT, OtherOrder> &&other) {
        Base::mantissa = T(other.mantissa);
        Base::exp = other.exp;
    }

    CUDA_CRAP constexpr HDRFloat(TExp ex, T mant) {
        Base::mantissa = mant;
        Base::exp = ex;
    }

    template <class To, class From, class Res = typename std::enable_if<
        (sizeof(To) == sizeof(From)) &&
        (alignof(To) == alignof(From)) &&
        std::is_trivially_copyable<From>::value &&
        std::is_trivially_copyable<To>::value,
        To>::type>
    CUDA_CRAP const Res &bit_cast(const From &src) noexcept {
        return *reinterpret_cast<const To *>(&src);
    }

#ifndef __CUDACC__
    //constexpr explicit HDRFloat(const mpf_t &other) {
    //    double mantissa;

    //    Base::exp = static_cast<TExp>(mpf_get_2exp_d(&mantissa, other));
    //    Base::mantissa = static_cast<T>(mantissa);
    //}
#endif

    template<class U, typename =
        std::enable_if_t<
        std::is_same<U, float>::value ||
        std::is_same<U, double>::value ||
        std::is_same<U, int>::value ||
        std::is_same<U, CudaDblflt<dblflt>>::value>>
        CUDA_CRAP explicit HDRFloat(const U number) { // TODO add constexpr once that compiles
        if (number == U{}) {
            Base::mantissa = T{ 0.0f };
            Base::exp = GenericHdrBase::MIN_BIG_EXPONENT();
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            // TODO use std::bit_cast once that works in CUDA
            const auto bits = bit_cast<uint64_t>((T)number);
            //constexpr uint64_t bits = __builtin_bit_cast(std::uint64_t, &number);
            const auto f_exp = (int32_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT_INT_DOUBLE();
            const auto val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            const T f_val = bit_cast<T>(val);

            Base::mantissa = (T)f_val;
            Base::exp = (TExp)f_exp;
        } else if constexpr (std::is_same<T, float>::value) {
            const auto bits = bit_cast<uint32_t>((T)number);
            const auto f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT_FLOAT();
            const auto val = (bits & 0x807F'FFFF) | 0x3F80'0000;
            const T f_val = bit_cast<T>(val);

            Base::mantissa = (T)f_val;
            Base::exp = (TExp)f_exp;
        } else if constexpr (std::is_same<T, CudaDblflt<dblflt>>::value) {
            if constexpr (std::is_same<U, CudaDblflt<dblflt>>::value) {
                Base::exp = 0;
                Base::mantissa = number;
            } else if constexpr (std::is_same<U, float>::value) {
                const auto bits = bit_cast<uint32_t>(number);
                const auto f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT_FLOAT();
                const auto val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
                const float f_val = bit_cast<float>(val);

                Base::mantissa.d.head = f_val;
                Base::mantissa.d.tail = 0;
                Base::exp = (TExp)f_exp;
            } else if constexpr (std::is_same<U, double>::value) {
                const auto bits = bit_cast<uint64_t>(number);
                const auto f_exp = (int64_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT_INT_DOUBLE();
                const auto val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
                const auto f_val = bit_cast<double>(val);

                Base::mantissa = CudaDblflt(f_val);
                Base::exp = (TExp)f_exp;
            } else if constexpr (std::is_same<U, int>::value) {
                const auto floatVal = (float)number;
                const auto bits = bit_cast<uint32_t>(floatVal);
                const auto f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT_FLOAT();
                const auto val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
                const float f_val = bit_cast<float>(val);

                Base::mantissa.d.head = f_val;
                Base::mantissa.d.tail = 0;
                Base::exp = (TExp)f_exp;
            }
        }
    }

#ifndef __CUDACC__
    explicit HDRFloat(const mpf_t number) {
        if (mpf_cmp_ui(number, 0) == 0) {
            Base::mantissa = T{};
            Base::exp = GenericHdrBase::MIN_BIG_EXPONENT();
            return;
        }

        if constexpr (std::is_same<T, CudaDblflt<dblflt>>::value) {
            long temp_exp;
            double tempMantissa;

            temp_exp = static_cast<int32_t>(mpf_get_2exp_d(&tempMantissa, number));
            Base::mantissa.d.head = (float)tempMantissa;
            Base::mantissa.d.tail = (float)((double)tempMantissa - (double)Base::mantissa.d.head);
            Base::exp = (TExp)temp_exp;
        } else {
            long temp_exp;
            double tempMantissa;
            temp_exp = static_cast<int32_t>(mpf_get_2exp_d(&tempMantissa, number));
            Base::mantissa = (T)tempMantissa;
            Base::exp = (TExp)temp_exp;
        }
    }

    explicit HDRFloat(const HighPrecisionT<HPDestructor::True> &number)
        : HDRFloat{ *number.backendRaw() } {
    }

    void GetHighPrecision(HighPrecisionT<HPDestructor::True> &number) const {
        mpf_t temp;
        mpf_init2(temp, 64);

        mpf_set_d(temp, Base::mantissa);

        if (Base::exp >= 0) {
            mpf_mul_2exp(temp, temp, Base::exp);
        } else {
            mpf_div_2exp(temp, temp, -Base::exp);
        }
        
        number = HighPrecisionT<HPDestructor::True>(temp);
        mpf_clear(temp);
    }
#endif

    template<bool GetExpAmt = false>
    CUDA_CRAP constexpr HDRFloat &Reduce(TExp *DestExp = nullptr) & {
        if constexpr (std::is_same<T, double>::value) {
            if (Base::mantissa == 0) {

                if constexpr (GetExpAmt) {
                    *DestExp = 0;
                }
                return *this;
            }

            const auto bits = bit_cast<uint64_t>(this->Base::mantissa);
            const auto f_exp =
                static_cast<TExp>(((bits & 0x7FF0'0000'0000'0000UL) >> 52UL)) + MIN_SMALL_EXPONENT_INT_DOUBLE();
            const auto val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            const auto f_val = bit_cast<const T>(val);
            Base::exp += f_exp;
            Base::mantissa = f_val;

            if constexpr (GetExpAmt) {
                *DestExp = f_exp;
            }
        } else if constexpr (std::is_same<T, float>::value) {
            if (Base::mantissa == 0) {

                if constexpr (GetExpAmt) {
                    *DestExp = 0;
                }
                return *this;
            }

            const auto bits = bit_cast<uint32_t>(this->Base::mantissa);
            const auto f_exp =
                static_cast<TExp>(((bits & 0x7F80'0000UL) >> 23UL)) + MIN_SMALL_EXPONENT_INT_FLOAT();
            const auto val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
            const auto f_val = bit_cast<T>(val);
            Base::exp += f_exp;
            Base::mantissa = f_val;

            if constexpr (GetExpAmt) {
                *DestExp = f_exp;
            }
        } else if constexpr (std::is_same<T, CudaDblflt<dblflt>>::value) {
            if (Base::mantissa.d.head == 0 && Base::mantissa.d.tail == 0) {

                if constexpr (GetExpAmt) {
                    *DestExp = 0;
                }
                return *this;
            }

            const auto bits_y = bit_cast<uint32_t>(this->Base::mantissa.d.head);
            const auto bits_x = bit_cast<uint32_t>(this->Base::mantissa.d.tail);
            const auto f_exp_y =
                static_cast<TExp>(((bits_y & 0x7F80'0000UL) >> 23UL)) + MIN_SMALL_EXPONENT_INT_FLOAT();
            const auto f_exp_x =
                static_cast<TExp>(((bits_x & 0x7F80'0000UL) >> 23UL));
            const auto val_y = (bits_y & 0x807F'FFFFU) | 0x3F80'0000U;
            const auto newexp = f_exp_x - f_exp_y;
            const auto satexp = newexp <= 0 ? 0 : newexp;
            const auto val_x = (bits_x & 0x807F'FFFFU) | (satexp << 23U);
            const auto f_val_y = bit_cast<float>(val_y);
            const auto f_val_x = bit_cast<float>(val_x);
            Base::exp += f_exp_y;
            Base::mantissa.d.head = f_val_y;
            Base::mantissa.d.tail = f_val_x;

            if constexpr (GetExpAmt) {
                *DestExp = f_exp_y;
            }
        }

        return *this;
    }

    CUDA_CRAP constexpr HDRFloat &&Reduce() && {
        Reduce();
        return std::move(*this);
    }

    static CUDA_CRAP constexpr T getMultiplier(TExp scaleFactor) {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, CudaDblflt<dblflt>>::value) {
            if (scaleFactor <= MIN_SMALL_EXPONENT_FLOAT()) {
                return T{};
            }

            if (scaleFactor >= 128) {
                return std::numeric_limits<T>::max();
            }

            return (T)twoPowExpFlt[(int)scaleFactor - MinFloatExponent];
        } else if constexpr (std::is_same<T, double>::value) {
            if (scaleFactor <= MIN_SMALL_EXPONENT_DOUBLE()) {
                return T{};
            }

            if (scaleFactor >= 1024) {
                return std::numeric_limits<T>::max();
            }

            return (T)twoPowExpDbl[(int)scaleFactor - MinDoubleExponent];
        }
    }


    template<bool IncludeCheck = true>
    static CUDA_CRAP constexpr T getMultiplierNeg(TExp scaleFactor) {
        //if constexpr (std::is_same<T, double>::value) {
        //    return scalbn(1.0, scaleFactor);
        //}
        //else {
        //    return scalbnf(1.0f, scaleFactor);
        //}

        if constexpr (std::is_same<T, double>::value) {
            if constexpr (IncludeCheck) {
                if (scaleFactor <= MIN_SMALL_EXPONENT_DOUBLE()) {
                    return T{};
                }
            }

            return twoPowExpDbl[(int)scaleFactor - MinDoubleExponent];
        } else {
            if constexpr (IncludeCheck) {
                if (scaleFactor <= MIN_SMALL_EXPONENT_FLOAT()) {
                    return T{};
                }
            }

            //return scalbnf(1.0, scaleFactor);
            return T{ twoPowExpFlt[(int)scaleFactor - MinFloatExponent] };
        }
    }

    CUDA_CRAP constexpr T toDouble() const {
        return Base::mantissa * getMultiplier(Base::exp);
    }

    CUDA_CRAP constexpr T toDoubleSub(TExp exponent) const {
        return Base::mantissa * getMultiplier(Base::exp - exponent);
    }

    CUDA_CRAP explicit constexpr operator T() const { return toDouble(); }

    template<HDROrder OtherOrder>
    CUDA_CRAP bool operator==(const HDRFloat<T, OtherOrder> &other) const {
        if (Base::exp == other.exp &&
            Base::mantissa == other.mantissa) {
            return true;
        }

        return false;
    }

    CUDA_CRAP constexpr T getMantissa() const {
        return Base::mantissa;
    }

    CUDA_CRAP constexpr TExp getExp() const {
        return Base::exp;
    }

    CUDA_CRAP constexpr void setExp(TExp localexp) {
        Base::exp = localexp;
    }

    CUDA_CRAP constexpr void setMantissa(T mant) {
        Base::mantissa = mant;
    }

    CUDA_CRAP constexpr HDRFloat reciprocal() const {
        T local_mantissa = (T)1.0 / Base::mantissa;
        TExp local_exp = -Base::exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat &reciprocal_mutable() {
        Base::mantissa = (T)1.0 / Base::mantissa;
        Base::exp = -Base::exp;

        return *this;
    }

    template<HDROrder OtherOrder, typename OtherTExp>
    CUDA_CRAP constexpr HDRFloat &divide_mutable(HDRFloat<T, OtherOrder, typename OtherTExp> factor) {
        T local_mantissa{ Base::mantissa / factor.mantissa };
        TExp local_exp = Base::exp - factor.exp;

        Base::mantissa = local_mantissa;
        Base::exp = local_exp < GenericHdrBase::MIN_BIG_EXPONENT() ? GenericHdrBase::MIN_BIG_EXPONENT() : local_exp;

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    template<HDROrder OtherOrder, typename OtherTExp>
    friend CUDA_CRAP constexpr HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat<T, OtherOrder, OtherTExp> &rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat &divide_mutable(T factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return divide_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    template<HDROrder OtherOrder, typename OtherTExp>
    friend CUDA_CRAP constexpr HDRFloat operator/(HDRFloat<T, OtherOrder, OtherTExp> lhs,        // passing lhs by value helps optimize chained a+b+c
        const T &rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    template<HDROrder OtherOrder, typename OtherTExp>
    CUDA_CRAP constexpr HDRFloat &operator/=(const HDRFloat<T, OtherOrder, OtherTExp> &other) {
        return divide_mutable(other);
    }

    template<bool plus>
    static CUDA_CRAP HDRFloat custom_perturb1(
        const HDRFloat &DeltaSubNXOrig,
        const HDRFloat &tempSum1,
        const HDRFloat &DeltaSubNYOrig,
        const HDRFloat &tempSum2,
        const HDRFloat &DeltaSub0) {

        const T local_mantissa1 = DeltaSubNXOrig.mantissa * tempSum1.mantissa;
        const TExp local_exp1 = DeltaSubNXOrig.exp + tempSum1.exp;

        const T local_mantissa2 = DeltaSubNYOrig.mantissa * tempSum2.mantissa;
        const TExp local_exp2 = DeltaSubNYOrig.exp + tempSum2.exp;
        const TExp maxexp = CudaHostMax(local_exp1, local_exp2);
        const TExp expDiff1 = local_exp1 - local_exp2;
        const T mul = getMultiplierNeg(-abs(expDiff1));

        T mantissaSum1;
        TExp expDiff2;
        TExp finalMaxexp;
        if constexpr (plus) {
            if (local_exp1 >= local_exp2) {
                // TODO constexpr double path
                mantissaSum1 = __fmaf_rn(local_mantissa2, mul, local_mantissa1);
            } else {
                mantissaSum1 = __fmaf_rn(local_mantissa1, mul, local_mantissa2);
            }
        } else {
            if (local_exp1 >= local_exp2) {
                mantissaSum1 = __fmaf_rn(-local_mantissa2, mul, local_mantissa1);
            } else {
                mantissaSum1 = __fmaf_rn(local_mantissa1, mul, -local_mantissa2);
            }
        }

        expDiff2 = maxexp - DeltaSub0.exp;
        finalMaxexp = CudaHostMax(maxexp, DeltaSub0.exp);

        const T mul2 = getMultiplierNeg(-abs(expDiff2));
        HDRFloat sum2;
        if (expDiff2 >= 0) {
            sum2 = HDRFloat(finalMaxexp, __fmaf_rn(DeltaSub0.mantissa, mul2, mantissaSum1));
        } else {
            sum2 = HDRFloat(finalMaxexp, __fmaf_rn(mantissaSum1, mul2, DeltaSub0.mantissa));
        }

        HdrReduce(sum2);
        return sum2;
    }

    static CUDA_CRAP void custom_perturb2(
        HDRFloat &DeltaSubNX,
        HDRFloat &DeltaSubNY,
        const HDRFloat &DeltaSubNXOrig,
        const HDRFloat &tempSum1,
        const HDRFloat &DeltaSubNYOrig,
        const HDRFloat &tempSum2,
        const HDRFloat &DeltaSub0X,
        const HDRFloat &DeltaSub0Y) {

        const TExp local_exp1F = DeltaSubNXOrig.exp + tempSum1.exp;
        const TExp local_exp2F = DeltaSubNYOrig.exp + tempSum2.exp;
        const TExp maxexpF = CudaHostMax(local_exp1F, local_exp2F);
        const TExp expDiff1F = local_exp1F - local_exp2F;
        const T mulF = getMultiplierNeg(-abs(expDiff1F));

        const T local_mantissa1F = DeltaSubNXOrig.mantissa * tempSum1.mantissa;
        const T local_mantissa2F = DeltaSubNYOrig.mantissa * tempSum2.mantissa;
        T mantissaSum1F;
        if (expDiff1F >= 0) {
            mantissaSum1F = __fmaf_rn(-local_mantissa2F, mulF, local_mantissa1F);
        } else {
            mantissaSum1F = __fmaf_rn(local_mantissa1F, mulF, -local_mantissa2F);
        }

        const TExp expDiff2F = maxexpF - DeltaSub0X.exp;
        const TExp finalMaxexpF = CudaHostMax(maxexpF, DeltaSub0X.exp);
        const T mul2F = getMultiplierNeg(-abs(expDiff2F));
        //const int ZeroOrOne = (int)(expDiff2F >= 0);
        //DeltaSubNX = HDRFloat(ZeroOrOne * finalMaxexpF, ZeroOrOne * __fmaf_rn(DeltaSub0X.mantissa, mul2F, mantissaSum1F)) +
        //             HDRFloat((1 - ZeroOrOne) * finalMaxexpF, (1 - ZeroOrOne) * __fmaf_rn(mantissaSum1F, mul2F, DeltaSub0X.mantissa));
        if (expDiff2F >= 0) {
            DeltaSubNX = HDRFloat(finalMaxexpF, __fmaf_rn(DeltaSub0X.mantissa, mul2F, mantissaSum1F));
        } else {
            DeltaSubNX = HDRFloat(finalMaxexpF, __fmaf_rn(mantissaSum1F, mul2F, DeltaSub0X.mantissa));
        }

        HdrReduce(DeltaSubNX);

        const TExp local_exp1T = DeltaSubNXOrig.exp + tempSum2.exp;
        const TExp local_exp2T = DeltaSubNYOrig.exp + tempSum1.exp;
        const TExp maxexpT = CudaHostMax(local_exp1T, local_exp2T);
        const TExp expDiff1T = local_exp1T - local_exp2T;
        const T mulT = getMultiplierNeg(-abs(expDiff1T));

        const T local_mantissa1T = DeltaSubNXOrig.mantissa * tempSum2.mantissa;
        const T local_mantissa2T = DeltaSubNYOrig.mantissa * tempSum1.mantissa;
        T mantissaSum1T;
        if (expDiff1T >= 0) {
            // TODO constexpr double path
            mantissaSum1T = __fmaf_rn(local_mantissa2T, mulT, local_mantissa1T);
        } else {
            mantissaSum1T = __fmaf_rn(local_mantissa1T, mulT, local_mantissa2T);
        }

        const TExp expDiff2T = maxexpT - DeltaSub0Y.exp;
        const TExp finalMaxexpT = CudaHostMax(maxexpT, DeltaSub0Y.exp);
        const T mul2T = getMultiplierNeg(-abs(expDiff2T));

        if (expDiff2T >= 0) {
            DeltaSubNY = HDRFloat(finalMaxexpT, __fmaf_rn(DeltaSub0Y.mantissa, mul2T, mantissaSum1T));
        } else {
            DeltaSubNY = HDRFloat(finalMaxexpT, __fmaf_rn(mantissaSum1T, mul2T, DeltaSub0Y.mantissa));
        }

        HdrReduce(DeltaSubNY);
    }

    static CUDA_CRAP void custom_perturb3(
        HDRFloat &DeltaSubNX,
        HDRFloat &DeltaSubNY,
        const HDRFloat &DeltaSubNXOrig,
        const HDRFloat &tempSum1,
        const HDRFloat &DeltaSubNYOrig,
        const HDRFloat &tempSum2,
        const HDRFloat &DeltaSub0X,
        const HDRFloat &DeltaSub0Y) {
        DeltaSubNX =
            DeltaSubNXOrig * tempSum1 -
            DeltaSubNYOrig * tempSum2 +
            DeltaSub0X;
        HdrReduce(DeltaSubNX);

        DeltaSubNY =
            DeltaSubNXOrig * tempSum2 +
            DeltaSubNYOrig * tempSum1 +
            DeltaSub0Y;
        HdrReduce(DeltaSubNY);

        // This kind of thing can have a minor positive effect but 
        // as a practical matter it's not worth the extra cost.
        //auto temp1 = DeltaSubNXOrig * tempSum1;
        //HdrReduce(temp1);
        //auto temp2 = DeltaSubNYOrig * tempSum2;
        //HdrReduce(temp2);
        //DeltaSubNX = temp1 - temp2 + DeltaSub0X;
        //HdrReduce(DeltaSubNX);

        //auto temp3 = DeltaSubNXOrig * tempSum2;
        //HdrReduce(temp3);
        //auto temp4 = DeltaSubNYOrig * tempSum1;
        //HdrReduce(temp4);
        //DeltaSubNY = temp3 + temp4 + DeltaSub0Y;
        //HdrReduce(DeltaSubNY);
    }

    template<HDROrder OtherOrder, typename OtherTExp>
    CUDA_CRAP constexpr HDRFloat &multiply_mutable(const HDRFloat<T, OtherOrder, OtherTExp> &factor) {
        T local_mantissa = Base::mantissa * factor.mantissa;
        TExp local_exp = Base::exp + factor.exp;

        Base::mantissa = local_mantissa;
        Base::exp = local_exp < GenericHdrBase::MIN_BIG_EXPONENT() ? GenericHdrBase::MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    template<HDROrder OtherOrder, typename OtherTExp>
    friend CUDA_CRAP constexpr HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat<T, OtherOrder, OtherTExp> &rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable<OtherOrder, OtherTExp>(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    template<HDROrder OtherOrder, typename OtherTExp>
    CUDA_CRAP constexpr HDRFloat &multiply_mutable(T factor) {
        auto factorMant = HDRFloat<T, OtherOrder, OtherTExp>(factor);
        return multiply_mutable<OtherOrder, OtherTExp>(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T &rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable<Order, TExp>(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    template<HDROrder OtherOrder, typename OtherTExp>
    CUDA_CRAP constexpr HDRFloat &operator*=(const HDRFloat<T, OtherOrder, OtherTExp> &other) {
        return multiply_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat square() const {
        T local_mantissa = Base::mantissa * Base::mantissa;
        TExp local_exp = Base::exp * 2;

        return HDRFloat(local_exp, local_mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &square_mutable() {
        T local_mantissa = Base::mantissa * Base::mantissa;
        TExp local_exp = Base::exp * 2;

        Base::mantissa = local_mantissa;
        Base::exp = local_exp < GenericHdrBase::MIN_BIG_EXPONENT() ? GenericHdrBase::MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat multiply2() const {
        return HDRFloat(Base::exp + 1, Base::mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &multiply2_mutable() {
        Base::exp++;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat multiply4() const {
        return HDRFloat(Base::exp + 2, Base::mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &multiply4_mutable() {
        Base::exp += 2;
        return *this;
    }


    CUDA_CRAP constexpr HDRFloat divide2() const {
        return HDRFloat(Base::mantissa, Base::exp - 1);
    }

    CUDA_CRAP constexpr HDRFloat &divide2_mutable() {
        Base::exp--;
        Base::exp = Base::exp < GenericHdrBase::MIN_BIG_EXPONENT() ? GenericHdrBase::MIN_BIG_EXPONENT() : Base::exp;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat divide4() const {
        return HDRFloat(Base::mantissa, Base::exp - 2);
    }

    CUDA_CRAP constexpr HDRFloat &divide4_mutable() {
        Base::exp -= 2;
        Base::exp--;
        Base::exp = Base::exp < GenericHdrBase::MIN_BIG_EXPONENT() ? GenericHdrBase::MIN_BIG_EXPONENT() : Base::exp;
        return *this;
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat add(HDRFloat<T, OtherOrder> value) const {

        TExp expDiff = Base::exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(Base::exp, Base::mantissa, true);
        } else if (expDiff >= 0) {
            T mul = getMultiplierNeg(-expDiff);
            return HDRFloat(Base::exp, Base::mantissa + value.mantissa * mul, true);
        } else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
            return HDRFloat(value.exp, Base::mantissa * mul + value.mantissa, true);
        } else {
            return HDRFloat(value.exp, value.mantissa, true);
        }
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &add_mutable(HDRFloat<T, OtherOrder> value) {

        TExp expDiff = Base::exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if (expDiff >= 0) {
            T mul = getMultiplierNeg(-expDiff);
            Base::mantissa = Base::mantissa + value.mantissa * mul;
        } else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
            Base::exp = value.exp;
            Base::mantissa = Base::mantissa * mul + value.mantissa;
        } else {
            Base::exp = value.exp;
            Base::mantissa = value.mantissa;
        }

        if (Base::mantissa == T{}) {
            Base::exp = GenericHdrBase::MIN_BIG_EXPONENT();
        }

        return *this;

    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    template<HDROrder OtherOrder>
    friend CUDA_CRAP constexpr HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat<T, OtherOrder> &rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &operator+=(const HDRFloat<T, OtherOrder> &other) {
        return add_mutable(other);
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat subtract(HDRFloat<T, OtherOrder> value) const {

        TExp expDiff = Base::exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(Base::exp, Base::mantissa, true);
        } else if (expDiff >= 0) {
            T mul = getMultiplierNeg(-expDiff);
            return HDRFloat(Base::exp, Base::mantissa - value.mantissa * mul, true);
        } else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
            return HDRFloat(value.exp, Base::mantissa * mul - value.mantissa, true);
        } else {
            return HDRFloat(value.exp, -value.mantissa, true);
        }
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &subtract_mutable(HDRFloat<T, OtherOrder> value) {

        TExp expDiff = Base::exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if (expDiff >= 0) {
            T mul = getMultiplierNeg(-expDiff);
            Base::mantissa = Base::mantissa - value.mantissa * mul;
        } else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
            Base::exp = value.exp;
            Base::mantissa = Base::mantissa * mul - value.mantissa;
        } else {
            Base::exp = value.exp;
            Base::mantissa = -value.mantissa;
        }

        if (Base::mantissa == T{}) {
            Base::exp = GenericHdrBase::MIN_BIG_EXPONENT();
        }

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    template<HDROrder OtherOrder>
    friend CUDA_CRAP constexpr HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat<T, OtherOrder> &rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr HDRFloat &operator-=(const HDRFloat<T, OtherOrder> &other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat add(T value) const {
        return add(HDRFloat(value));
    }

    CUDA_CRAP constexpr HDRFloat &add_mutable(T value) {
        return add_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP constexpr HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T &rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat subtract(T value) const {
        return subtract(HDRFloat(value));
    }

    CUDA_CRAP constexpr HDRFloat &subtract_mutable(T value) {
        return subtract_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP constexpr HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T &rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat &operator-=(const T &other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat negate() const {
        return HDRFloat(Base::exp, -Base::mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &negate_mutable() {
        Base::mantissa = -Base::mantissa;
        return *this;
    }

    template<HDROrder OtherOrder>
    friend CUDA_CRAP constexpr HDRFloat operator-(HDRFloat<T, OtherOrder> lhs) {
        return lhs.negate();
    }

    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr int compareToBothPositiveReduced(HDRFloat<T, OtherOrder> compareTo) const {
        if (Base::exp > compareTo.exp) {
            return 1;
        } else if (Base::exp < compareTo.exp) {
            return -1;
        } else {
            if (Base::mantissa > compareTo.mantissa) {
                return 1;
            } else if (Base::mantissa < compareTo.mantissa) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    template<int32_t SomeConstant = 256>
    CUDA_CRAP inline int compareToBothPositiveReducedTemplate() const {
        if (Base::exp > 1) {
            return 1;
        } else if (Base::exp < 1) {
            return -1;
        } else {
            if (Base::mantissa >= T{ SomeConstant }) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    // Matt: be sure both numbers are reduced
    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr int compareToBothPositive(HDRFloat<T, OtherOrder> compareTo) const {
        if (Base::exp > compareTo.exp) {
            return 1;
        } else if (Base::exp < compareTo.exp) {
            return -1;
        } else {
            if (Base::mantissa > compareTo.mantissa) {
                return 1;
            } else if (Base::mantissa < compareTo.mantissa) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    // Matt: be sure both numbers are reduced
    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr int compareTo(HDRFloat<T, OtherOrder> compareTo) const {
        if (Base::mantissa == 0 && compareTo.mantissa == 0) {
            return 0;
        }

        if (Base::mantissa > 0) {
            if (compareTo.mantissa <= 0) {
                return 1;
            } else if (Base::exp > compareTo.exp) {
                return 1;
            } else if (Base::exp < compareTo.exp) {
                return -1;
            } else {
                if (Base::mantissa > compareTo.mantissa) {
                    return 1;
                } else if (Base::mantissa < compareTo.mantissa) {
                    return -1;
                } else {
                    return 0;
                }
            }
        } else {
            if (compareTo.mantissa > 0) {
                return -1;
            } else if (Base::exp > compareTo.exp) {
                return -1;
            } else if (Base::exp < compareTo.exp) {
                return 1;
            } else {
                if (Base::mantissa > compareTo.mantissa) {
                    return 1;
                } else if (Base::mantissa < compareTo.mantissa) {
                    return -1;
                } else {
                    return 0;
                }
            }
        }
    }

    // Matt: be sure both numbers are reduced
    template<HDROrder OtherOrder>
    CUDA_CRAP constexpr int compareToReduced(HDRFloat<T, OtherOrder> compareToReduced) const {

        if (Base::mantissa == 0 && compareToReduced.mantissa == 0) {
            return 0;
        }

        if (Base::mantissa > 0) {
            if (compareToReduced.mantissa <= 0) {
                return 1;
            } else if (Base::exp > compareToReduced.exp) {
                return 1;
            } else if (Base::exp < compareToReduced.exp) {
                return -1;
            } else {
                if (Base::mantissa > compareToReduced.mantissa) {
                    return 1;
                } else if (Base::mantissa < compareToReduced.mantissa) {
                    return -1;
                } else {
                    return 0;
                }
            }
        } else {
            if (compareToReduced.mantissa > 0) {
                return -1;
            } else if (Base::exp > compareToReduced.exp) {
                return -1;
            } else if (Base::exp < compareToReduced.exp) {
                return 1;
            } else {
                if (Base::mantissa > compareToReduced.mantissa) {
                    return 1;
                } else if (Base::mantissa < compareToReduced.mantissa) {
                    return -1;
                } else {
                    return 0;
                }
            }
        }
    }

    //friend CUDA_CRAP constexpr bool operator<(const HDRFloat& l, const HDRFloat& r)
    //{
    //    return l.compareTo(r) < 0;
    //}

    //friend CUDA_CRAP constexpr bool operator<=(const HDRFloat& l, const HDRFloat& r)
    //{
    //    return l.compareTo(r) <= 0;
    //}

    //friend CUDA_CRAP constexpr bool operator>(const HDRFloat& l, const HDRFloat& r)
    //{
    //    return l.compareTo(r) > 0;
    //}

    //friend CUDA_CRAP constexpr bool operator>=(const HDRFloat& l, const HDRFloat& r)
    //{
    //    return l.compareTo(r) >= 0;
    //}

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat HDRMax(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareTo(b) > 0 ? a : b;
    }

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat maxBothPositive(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareToBothPositive(b) > 0 ? a : b;
    }

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat maxBothPositiveReduced(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareToBothPositiveReduced(b) > 0 ? a : b;
    }

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat minBothPositive(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareToBothPositive(b) < 0 ? a : b;
    }

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat minBothPositiveReduced(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareToBothPositiveReduced(b) < 0 ? a : b;
    }

    template<HDROrder OtherOrder>
    static CUDA_CRAP constexpr HDRFloat HDRMin(HDRFloat a, HDRFloat<T, OtherOrder> b) {
        return a.compareTo(b) < 0 ? a : b;
    }
};

template<class T>
static CUDA_CRAP T HdrSqrt(const T &incoming) {
    static_assert(std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");

    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");

    if constexpr (std::is_same<T, double>::value ||
        std::is_same<T, float>::value) {
        return sqrt((T)incoming);
    } else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        int32_t castExp = *reinterpret_cast<const int32_t *>(&incoming.exp);
        bool isOdd = (castExp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
            ::sqrt(isOdd ? 2.0 * incoming.mantissa : incoming.mantissa));
    } else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        int32_t castExp = *reinterpret_cast<const int32_t *>(&incoming.exp);
        bool isOdd = (castExp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
            ::sqrt(isOdd ? 2.0f * incoming.mantissa : incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP constexpr T HdrAbs(const T &incoming)
    requires (std::is_same<T, double>::value ||
              std::is_same<T, float>::value ||
              std::is_same<T, CudaDblflt<MattDblflt>>::value ||
              std::is_same<T, HDRFloat<double>>::value ||
              std::is_same<T, HDRFloat<float>>::value ||
              std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value) {

    if constexpr (std::is_same<T, double>::value ||
        std::is_same<T, float>::value) {
        return fabs((T)incoming);
    } else if constexpr (std::is_same<T, CudaDblflt<dblflt>>::value) {
        return incoming.abs();
    } else if constexpr (std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<double>>::value) {
        return T(incoming.exp, fabs(incoming.mantissa));
    } else {
        static_assert(std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
        return T(incoming.exp, incoming.mantissa.abs());
    }
}

template<class T>
static CUDA_CRAP constexpr void HdrReduce(T &incoming) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif

    static_assert(
        std::is_same_v<T, double> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, CudaDblflt<dblflt>> ||
        std::is_same_v<T, HDRFloat<double>> ||
        std::is_same_v<T, HDRFloat<float>> ||
        std::is_same_v<T, HDRFloat<CudaDblflt<dblflt>>> ||
        std::is_same_v<T, HDRFloat<double, HDROrder::Right>> ||
        std::is_same_v<T, HDRFloat<float, HDROrder::Right>> ||
        std::is_same_v<T, HDRFloat<CudaDblflt<dblflt>, HDROrder::Right>> ||
        std::is_same_v<T, Imagina::HRReal> ||
        std::is_same_v<T, HDRFloatComplex<double>> ||
        std::is_same_v<T, HDRFloatComplex<float>> ||
        std::is_same_v<T, FloatComplex<double>> ||
        std::is_same_v<T, FloatComplex<float>> ||
        HighPrecPossible, "No");

    if constexpr (
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value ||
        std::is_same<T, HDRFloat<double, HDROrder::Right>>::value ||
        std::is_same<T, HDRFloat<float, HDROrder::Right>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>, HDROrder::Right>>::value ||
        std::is_same<T, Imagina::HRReal>::value ||
        std::is_same<T, HDRFloatComplex<double>>::value ||
        std::is_same<T, HDRFloatComplex<float>>::value) {
        incoming.Reduce();
    }
}

template<class T>
static CUDA_CRAP constexpr T &&HdrReduce(T &&incoming) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif

    using no_const = std::remove_cv_t<T>;
    static_assert(
        std::is_same<no_const, double>::value ||
        std::is_same<no_const, float>::value ||
        std::is_same<no_const, CudaDblflt<dblflt>>::value ||
        std::is_same<no_const, HDRFloat<double>>::value ||
        std::is_same<no_const, HDRFloat<float>>::value ||
        std::is_same<no_const, HDRFloat<CudaDblflt<dblflt>>>::value ||
        std::is_same<no_const, HDRFloatComplex<double>>::value ||
        std::is_same<no_const, HDRFloatComplex<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<no_const, HDRFloat<double>>::value ||
        std::is_same<no_const, HDRFloat<float>>::value ||
        std::is_same<no_const, HDRFloat<CudaDblflt<dblflt>>>::value ||
        std::is_same<no_const, HDRFloatComplex<double>>::value ||
        std::is_same<no_const, HDRFloatComplex<float>>::value) {
        return std::move(incoming.Reduce());
    } else {
        return std::move(incoming);
    }
}

template<class T, class U>
static CUDA_CRAP constexpr T HdrMaxReduced(const T &one, const U &two) {
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");
    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");
    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        if (one.compareTo(two) > 0) {
            return one;
        }

        return two;
    } else {
        return (one > two) ? one : two;
    }
}

template<class T, class U>
static CUDA_CRAP constexpr T HdrMaxPositiveReduced(const T &one, const U &two) {
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");
    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");
    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        if (one.compareToBothPositiveReduced(two) > 0) {
            return one;
        }

        return two;
    } else {
        return (one > two) ? one : two;
    }
}

template<class T, class U>
static CUDA_CRAP constexpr T HdrMinPositiveReduced(const T &one, const U &two) {
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");
    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");
    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        if (one.compareToBothPositiveReduced(two) < 0) {
            return one;
        }

        return two;
    } else {
        return (one < two) ? one : two;
    }
}

template<class T, class U>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedLT(const T &one, const U &two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, CudaDblflt<dblflt>>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value) {
        return one.compareToBothPositiveReduced(two) < 0;
    } else {
        return one < two;
    }
}

template<class T, int32_t CompareAgainst>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedLT(const T &one) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, CudaDblflt<dblflt>>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value) {
        return one.compareToBothPositiveReducedTemplate() < 0;
    } else {
        return one < T(CompareAgainst);
    }
}

template<class T, class U>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedLE(const T &one, const U &two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif
    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) <= 0;
    } else {
        return one <= two;
    }
}

template<class T, class U>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedGT(const T &one, const U &two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif
    static_assert(!std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value, "!");
    static_assert(!std::is_same<T, CudaDblflt<dblflt>>::value, "!");
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) > 0;
    } else {
        return one > two;
    }
}

template<class T, class U>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedGE(const T &one, const U &two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::True>>::value;
#else
        false;
#endif
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, CudaDblflt<dblflt>>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value ||
        HighPrecPossible, "No");

    if constexpr (
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value) {
        return one.compareToBothPositiveReduced(two) >= 0;
    } else {
        return one >= two;
    }
}

#ifndef __CUDACC__
template<bool IntegerOutput, class T>
static CUDA_CRAP std::string HdrToString(const T &dat) {
    if constexpr (
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value) {

        if constexpr (!IntegerOutput) {
            std::stringstream ss;
            ss << std::setprecision(std::numeric_limits<double>::max_digits10);
            ss << "mantissa: " << static_cast<double>(dat) << " exp: 0";
            return ss.str();
        } else {
            // Interpret the bits as an integer and output that
            const auto doubleDat = static_cast<double>(dat);
            uint64_t bits = *reinterpret_cast<const uint64_t *>(&doubleDat);
            std::stringstream ss;
            ss << "mantissa: 0x" << std::hex << bits << " exp: 0x0";
            return ss.str();
        }
    } else if constexpr (std::is_same<T, HighPrecisionT<HPDestructor::True>>::value ||
        std::is_same<T, HighPrecisionT<HPDestructor::False>>::value) {
        return dat.str();
    } else {
        return dat.ToString<IntegerOutput>();
    }
}

template<bool IntegerInput>
static CUDA_CRAP double HdrFromIntToDbl(const std::string &mantissaStr) {
    if constexpr (!IntegerInput) {
        return std::stod(mantissaStr);
    } else {
        // Interpret bits of the integer as a double
        // Interpret the mantissaStr as hex
        uint64_t bits = std::stoull(mantissaStr, nullptr, 16);
        return *reinterpret_cast<double *>(&bits);
    }
}

template<bool IntegerInput, class T, typename SubType>
static CUDA_CRAP void HdrFromIfStream(T &out, std::istream &metafile) {
    std::string descriptor_string_junk;
    metafile >> descriptor_string_junk;

    if constexpr (
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value) {

        double tempMantissa;
        std::string mantissaStr;
        metafile >> mantissaStr;
        metafile >> mantissaStr;
        tempMantissa = HdrFromIntToDbl<IntegerInput>(mantissaStr);
        out = static_cast<SubType>(tempMantissa);

        // Read exponent
        std::string expStr;
        metafile >> expStr;
        metafile >> expStr;
    } else {
        out.FromIStream<IntegerInput>(metafile);
    }
}

#endif


// If you pass in T == HDRFloat<CudaDblflt<MattDblflt>> then ConditionalT will be HDRFloat<double>
// If you pass in SubType == CudaDblflt<MattDblflt> then ConditionalSubType will be double
// But:
// If you pass in T = anything else, then ConditionalT will be T
// If you pass in SubType = anything else, then ConditionalSubType will be SubType
template<typename T, typename SubType>
class DoubleTo2x32Converter {
public:

    DoubleTo2x32Converter() = delete;
    DoubleTo2x32Converter(const DoubleTo2x32Converter &) = delete;
    DoubleTo2x32Converter &operator=(const DoubleTo2x32Converter &) = delete;
    ~DoubleTo2x32Converter() = delete;

    // T is probably something else, like HDRFloat<float> or HDRFloat<double>
    // But if it's HDRFloat<CudaDblflt<MattDblflt>>, then we need to use HDRFloat<double> instead
    static constexpr bool ConditionalResult1 = std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    static constexpr bool ConditionalResult2 = std::is_same<T, CudaDblflt<MattDblflt>>::value;
    static constexpr bool ConditionalResult = ConditionalResult1 || ConditionalResult2;

    // SubType is probably float or double
    using ConditionalT = typename std::conditional<
        ConditionalResult1,
        HDRFloat<double>,
        typename std::conditional<
        ConditionalResult2,
        double,
        T>::type
    >::type;

    using ConditionalSubType = typename std::conditional<
        ConditionalResult1,
        double,
        typename std::conditional<
        ConditionalResult2,
        double,
        SubType>::type
    >::type;
};
