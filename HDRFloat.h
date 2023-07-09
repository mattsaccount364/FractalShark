#pragma once

#include <stdint.h>
#include <math.h>
#include "HighPrecision.h"

#ifdef __CUDA_ARCH__
#define CUDA_CRAP __device__
static __device__ double* __restrict__ twoPowExpDbl;
static __device__ float* __restrict__ twoPowExpFlt;
#else
#define CUDA_CRAP
extern double* twoPowExpDbl;
extern float* twoPowExpFlt;
#endif

CUDA_CRAP void InitStatics();

template<class T, class TExp = int32_t>
class HDRFloat {
public:
    static CUDA_CRAP constexpr TExp MIN_SMALL_EXPONENT() {
        if constexpr (std::is_same<T, double>::value) {
            return -1023;
        }
        else {
            return -127;
        }
    }

    static CUDA_CRAP constexpr int32_t MIN_SMALL_EXPONENT_INT() {
        if constexpr (std::is_same<T, double>::value) {
            return -1023;
        }
        else {
            return -127;
        }
    }

    static CUDA_CRAP constexpr TExp MIN_BIG_EXPONENT() {
        return INT32_MIN >> 3;
    }

//private:
//    static double LN2;
//    static double LN2_REC;

public:
    static constexpr TExp EXPONENT_DIFF_IGNORED = 120;
    static constexpr TExp MINUS_EXPONENT_DIFF_IGNORED = -EXPONENT_DIFF_IGNORED;
    T mantissa;
    TExp exp;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    static constexpr int MaxFloatExponent = 127;
    static constexpr int MinFloatExponent = -126;

    CUDA_CRAP constexpr HDRFloat() {
        mantissa = 0.0;
        exp = MIN_BIG_EXPONENT();
    }

    CUDA_CRAP constexpr HDRFloat(const HDRFloat<T> &other) {
        mantissa = other.mantissa;
        exp = other.exp;
    }

    template<class SrcT>
    CUDA_CRAP constexpr HDRFloat(const HDRFloat<SrcT>& other) {
        mantissa = (T)other.mantissa;
        exp = other.exp;
    }

    //CUDA_CRAP HDRFloat(T mantissa, TExp exp) {
    //    this->mantissa = mantissa;
    //    this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
    //}

    CUDA_CRAP constexpr HDRFloat(TExp exp, T mantissa) {
        this->mantissa = mantissa;
        this->exp = exp;
    }

    //CUDA_CRAP constexpr HDRFloat(TExp exp, T mantissa, bool /*check*/) {
    //    this->mantissa = mantissa;
    //    if (mantissa == 0) {
    //        this->exp = MIN_BIG_EXPONENT();
    //    }
    //    else {
    //        this->exp = exp;
    //    }
    //}

    //CUDA_CRAP constexpr int32_t internal_exponent(T x)
    //{
    //    return abs(x) >= 2 ? exponent(x / 2) + 1 :
    //        abs(x) < 1 ? exponent(x * 2) - 1 : 0;
    //}

    //CUDA_CRAP constexpr T internal_scalbn(T value, int exponent)
    //{
    //    return exponent == 0 ? value : exponent > 0 ? scalbn(value * 2, exponent - 1) :
    //        scalbn(value / 2, exponent + 1);
    //}

    //CUDA_CRAP constexpr unsigned internal_mantissa(T x)
    //{
    //    return abs(x) < std::numeric_limits<T>::infinity() ?
    //        // remove hidden 1 and bias the exponent to get integer
    //        internal_scalbn(internal_scalbn(abs(x), -internal_exponent(x)) - 1, 23) : 0;
    //}

    template <class To, class From, class Res = typename std::enable_if<
        (sizeof(To) == sizeof(From)) &&
        (alignof(To) == alignof(From)) &&
        std::is_trivially_copyable<From>::value&&
        std::is_trivially_copyable<To>::value,
        To>::type>
    CUDA_CRAP const Res& bit_cast(const From& src) noexcept {
        return *reinterpret_cast<const To*>(&src);
    }

    CUDA_CRAP explicit HDRFloat(const T number) {
        if (number == 0) {
            mantissa = 0;
            exp = MIN_BIG_EXPONENT();
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            // TODO use std::bit_cast once that works in CUDA
            const uint64_t bits = bit_cast<uint64_t>(number);
            //constexpr uint64_t bits = __builtin_bit_cast(std::uint64_t, &number);
            const int32_t f_exp = (int32_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT_INT();
            const uint64_t val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            const T f_val = bit_cast<T>(val);

            mantissa = f_val;
            exp = (TExp)f_exp;
        }
        else if constexpr (std::is_same<T, float>::value) {
            const uint32_t bits = bit_cast<uint32_t>(number);
            const int32_t f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT();
            const uint32_t val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
            const T f_val = bit_cast<T>(val);

            mantissa = f_val;
            exp = (TExp)f_exp;
        }
    }

#ifndef __CUDACC__ 
    HDRFloat(const HighPrecision &number) {

        if (number == 0) {
            mantissa = 0.0;
            exp = MIN_BIG_EXPONENT();
            return;
        }

        int temp_exp;
        mantissa = boost::multiprecision::frexp(number, &temp_exp).template convert_to<T>();
        exp = (TExp)temp_exp;
    }
#endif

    CUDA_CRAP constexpr void Reduce() {
        //if (mantissa == 0) {
        //    return;
        //}

        if constexpr (std::is_same<T, double>::value) {
            uint64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
            int32_t f_exp = (int32_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT_INT();
            uint64_t val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            T f_val = *reinterpret_cast<T*>(&val);
            exp += f_exp;
            mantissa = f_val;
        }
        else if constexpr (std::is_same<T, float>::value) {
            uint64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
            int32_t f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT();
            uint64_t val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
            T f_val = *reinterpret_cast<T*>(&val);
            exp += f_exp;
            mantissa = f_val;
        }
    }

    static CUDA_CRAP constexpr T getMultiplier(TExp scaleFactor) {
        if (scaleFactor <= MIN_SMALL_EXPONENT()) {
            return 0.0;
        }
        else if (scaleFactor >= 1024) {
            return INFINITY;
        }

        if constexpr (std::is_same<T, double>::value) {
            return (T)twoPowExpDbl[(int)scaleFactor - MinDoubleExponent];
        }
        else {
            return (T)twoPowExpFlt[(int)scaleFactor - MinFloatExponent];
        }
    }

    CUDA_CRAP constexpr T toDouble() const
    {
        return mantissa * getMultiplier(exp);
    }

    CUDA_CRAP constexpr T toDoubleSub(TExp exponent) const
    {
        return mantissa * getMultiplier(exp - exponent);
    }

    CUDA_CRAP explicit constexpr operator T() const { return toDouble(); }

    CUDA_CRAP constexpr T getMantissa() const { return  mantissa; }

    CUDA_CRAP constexpr TExp getExp() const { return exp; }

    CUDA_CRAP constexpr void setExp(TExp localexp) {
        this->exp = localexp;
    }

    CUDA_CRAP constexpr HDRFloat divide(HDRFloat factor) const {
        T local_mantissa = this->mantissa / factor.mantissa;
        TExp local_exp = this->exp - factor.exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat reciprocal() const {
        T local_mantissa = 1.0 / this->mantissa;
        TExp local_exp = -this->exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat &reciprocal_mutable() {
        mantissa = 1.0 / mantissa;
        exp = -exp;

        return *this;
    }

    CUDA_CRAP constexpr HDRFloat &divide_mutable(HDRFloat factor) {
        T local_mantissa = this->mantissa / factor.mantissa;
        TExp local_exp = this->exp - factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat divide(T factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return divide(factorMant);
    }

    CUDA_CRAP constexpr HDRFloat &divide_mutable(T factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return divide_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat& operator/=(const HDRFloat& other) {
        return divide_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat multiply(HDRFloat factor) const {
        T local_mantissa = this->mantissa * factor.mantissa;
        TExp local_exp = this->exp + factor.exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat multiply(T factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply(factorMant);
    }

    CUDA_CRAP constexpr HDRFloat &multiply_mutable(HDRFloat factor) {
        T local_mantissa = this->mantissa * factor.mantissa;
        TExp local_exp = this->exp + factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat &multiply_mutable(T factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat& operator*=(const HDRFloat& other) {
        return multiply_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat square() const {
        T local_mantissa = this->mantissa * this->mantissa;
        TExp local_exp = this->exp * 2;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat &square_mutable() {
        T local_mantissa = this->mantissa * this->mantissa;
        TExp local_exp = this->exp * 2;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat multiply2() const {
        return HDRFloat(exp + 1, mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &multiply2_mutable() {
        exp++;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat multiply4() const {
        return HDRFloat(exp + 2, mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &multiply4_mutable() {
        exp += 2;
        return *this;
    }


    CUDA_CRAP constexpr HDRFloat divide2() const {
        return HDRFloat(mantissa, exp - 1);
    }

    CUDA_CRAP constexpr HDRFloat &divide2_mutable() {
        exp--;
        this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat divide4() const {
        return HDRFloat(mantissa, exp - 2);
    }

    CUDA_CRAP constexpr HDRFloat &divide4_mutable() {
        exp -= 2;
        exp--;
        this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
        return *this;
    }

    CUDA_CRAP constexpr HDRFloat addOld(HDRFloat value) const {

        T temp_mantissa = 0;
        TExp temp_exp = exp;

        if (exp == value.exp) {
            temp_mantissa = mantissa + value.mantissa;
        }
        else if (exp > value.exp) {
            temp_mantissa = mantissa + getMultiplier(value.exp - exp) * value.mantissa;
        }
        else {
            temp_mantissa = getMultiplier(exp - value.exp) * mantissa;
            temp_exp = value.exp;
            temp_mantissa = temp_mantissa + value.mantissa;
        }

        return HDRFloat(temp_exp, temp_mantissa);

    }

    CUDA_CRAP constexpr HDRFloat add(HDRFloat value) const {

        TExp expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(exp, mantissa, true);
        }
        else if (expDiff >= 0) {
            T mul = getMultiplier(-expDiff);
            return HDRFloat(exp, mantissa + value.mantissa * mul, true);
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplier(expDiff);
            return HDRFloat(value.exp, mantissa * mul + value.mantissa, true);
        }
        else {
            return HDRFloat(value.exp, value.mantissa, true);
        }
    }

    CUDA_CRAP constexpr HDRFloat &add_mutable(HDRFloat value) {

        TExp expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        }
        else if (expDiff >= 0) {
            T mul = getMultiplier(-expDiff);
            mantissa = mantissa + value.mantissa * mul;
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplier(expDiff);
            exp = value.exp;
            mantissa = mantissa * mul + value.mantissa;
        }
        else {
            exp = value.exp;
            mantissa = value.mantissa;
        }

        if (mantissa == 0) {
            exp = MIN_BIG_EXPONENT();
        }

        return *this;

    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat& operator+=(const HDRFloat& other) {
        return add_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat subtract(HDRFloat value) const {

        TExp expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(exp, mantissa, true);
        }
        else if (expDiff >= 0) {
            T mul = getMultiplier(-expDiff);
            return HDRFloat(exp, mantissa - value.mantissa * mul, true);
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplier(expDiff);
            return HDRFloat(value.exp, mantissa * mul - value.mantissa, true);
        }
        else {
            return HDRFloat(value.exp, -value.mantissa, true);
        }
    }

    CUDA_CRAP constexpr HDRFloat &subtract_mutable(HDRFloat value) {

        TExp expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        }
        else if (expDiff >= 0) {
            T mul = getMultiplier(-expDiff);
            mantissa = mantissa - value.mantissa * mul;
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplier(expDiff);
            exp = value.exp;
            mantissa = mantissa * mul - value.mantissa;
        }
        else {
            exp = value.exp;
            mantissa = -value.mantissa;
        }

        if (mantissa == 0) {
            exp = MIN_BIG_EXPONENT();
        }

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat& operator-=(const HDRFloat& other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat add(T value) const {
        return add(HDRFloat(value));
    }

    CUDA_CRAP constexpr HDRFloat &add_mutable(T value) {
        return add_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP constexpr HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
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
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloat& operator-=(const T& other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP constexpr HDRFloat negate() const {
        return HDRFloat(exp, -mantissa);
    }

    CUDA_CRAP constexpr HDRFloat &negate_mutable() {
        mantissa = -mantissa;
        return *this;
    }

    friend CUDA_CRAP constexpr HDRFloat operator-(HDRFloat lhs)
    {
        return lhs.negate();
    }

    CUDA_CRAP constexpr int compareToBothPositiveReduced(HDRFloat compareTo) const {
        if (exp > compareTo.exp) {
            return 1;
        }
        else if (exp < compareTo.exp) {
            return -1;
        }
        else {
            if (mantissa > compareTo.mantissa) {
                return 1;
            }
            else if (mantissa < compareTo.mantissa) {
                return -1;
            }
            else {
                return 0;
            }
        }
    }

    // Matt: be sure both numbers are reduced
    CUDA_CRAP constexpr int compareToBothPositive(HDRFloat compareTo) const {
        if (exp > compareTo.exp) {
            return 1;
        }
        else if (exp < compareTo.exp) {
            return -1;
        }
        else {
            if (mantissa > compareTo.mantissa) {
                return 1;
            }
            else if (mantissa < compareTo.mantissa) {
                return -1;
            }
            else {
                return 0;
            }
        }
    }

    // Matt: be sure both numbers are reduced
    CUDA_CRAP constexpr int compareTo(HDRFloat compareTo) const {
        if (mantissa == 0 && compareTo.mantissa == 0) {
            return 0;
        }

        if (mantissa > 0) {
            if (compareTo.mantissa <= 0) {
                return 1;
            }
            else if (exp > compareTo.exp) {
                return 1;
            }
            else if (exp < compareTo.exp) {
                return -1;
            }
            else {
                if (mantissa > compareTo.mantissa) {
                    return 1;
                }
                else if (mantissa < compareTo.mantissa) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        }
        else {
            if (compareTo.mantissa > 0) {
                return -1;
            }
            else if (exp > compareTo.exp) {
                return -1;
            }
            else if (exp < compareTo.exp) {
                return 1;
            }
            else {
                if (mantissa > compareTo.mantissa) {
                    return 1;
                }
                else if (mantissa < compareTo.mantissa) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        }
    }

    // Matt: be sure both numbers are reduced
    CUDA_CRAP constexpr int compareToReduced(HDRFloat compareToReduced) const {

        if (mantissa == 0 && compareToReduced.mantissa == 0) {
            return 0;
        }

        if (mantissa > 0) {
            if (compareToReduced.mantissa <= 0) {
                return 1;
            }
            else if (exp > compareToReduced.exp) {
                return 1;
            }
            else if (exp < compareToReduced.exp) {
                return -1;
            }
            else {
                if (mantissa > compareToReduced.mantissa) {
                    return 1;
                }
                else if (mantissa < compareToReduced.mantissa) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        }
        else {
            if (compareToReduced.mantissa > 0) {
                return -1;
            }
            else if (exp > compareToReduced.exp) {
                return -1;
            }
            else if (exp < compareToReduced.exp) {
                return 1;
            }
            else {
                if (mantissa > compareToReduced.mantissa) {
                    return 1;
                }
                else if (mantissa < compareToReduced.mantissa) {
                    return -1;
                }
                else {
                    return 0;
                }
            }
        }
    }

    friend CUDA_CRAP constexpr bool operator<(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) < 0;
    }

    friend CUDA_CRAP constexpr bool operator<=(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) <= 0;
    }

    friend CUDA_CRAP constexpr bool operator>(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) > 0;
    }

    friend CUDA_CRAP constexpr bool operator>=(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) >= 0;
    }

    static CUDA_CRAP constexpr HDRFloat HDRMax(HDRFloat a, HDRFloat b) {
        return a.compareTo(b) > 0 ? a : b;
    }

    static CUDA_CRAP constexpr HDRFloat maxBothPositive(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositive(b) > 0 ? a : b;
    }

    static CUDA_CRAP constexpr HDRFloat maxBothPositiveReduced(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositiveReduced(b) > 0 ? a : b;
    }

    static CUDA_CRAP constexpr HDRFloat minBothPositive(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositive(b) < 0 ? a : b;
    }
    static CUDA_CRAP constexpr HDRFloat minBothPositiveReduced(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositiveReduced(b) < 0 ? a : b;
    }

    static CUDA_CRAP constexpr HDRFloat HDRMin(HDRFloat a, HDRFloat b) {
        return a.compareTo(b) < 0 ? a : b;
    }
};

template<class T>
static CUDA_CRAP T HdrSqrt(const T &incoming) {
    static_assert(std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");

    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, float>::value) {
        return sqrt((T)incoming);
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        int32_t castExp = *reinterpret_cast<const int32_t*>(&incoming.exp);
        bool isOdd = (castExp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
                 ::sqrt(isOdd ? 2.0 * incoming.mantissa : incoming.mantissa));
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        int32_t castExp = *reinterpret_cast<const int32_t*>(&incoming.exp);
        bool isOdd = (castExp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
            ::sqrt(isOdd ? 2.0f * incoming.mantissa : incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP constexpr T HdrAbs(const T& incoming) {
    static_assert(std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");

    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, float>::value) {
        return fabs((T)incoming);
    } else if constexpr (std::is_same<T, HDRFloat<float>>::value ||
                         std::is_same<T, HDRFloat<double>>::value) {
        return T(incoming.exp, fabs(incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP constexpr void HdrReduce(T& incoming) {
    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
                  std::is_same<T, HDRFloat<float>>::value) {
        incoming.Reduce();
    }
}
