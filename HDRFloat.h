#pragma once

#include <stdint.h>
#include <math.h>
#include "HighPrecision.h"

#ifdef __CUDA_ARCH__
#define CUDA_CRAP __device__
static __device__ double* twoPowExp;
#else
#define CUDA_CRAP
extern double* twoPowExp;
#endif

CUDA_CRAP void InitStatics();

template<class T>
class HDRFloat {
public:
    static CUDA_CRAP constexpr int32_t MIN_SMALL_EXPONENT() {
        if (std::is_same<T, double>::value) {
            return -1023;
        }
        else {
            return -127;
        }
    }

    static CUDA_CRAP constexpr int32_t MIN_BIG_EXPONENT() {
        return INT32_MIN >> 3;
    }

//private:
//    static double LN2;
//    static double LN2_REC;

public:
    static constexpr int32_t EXPONENT_DIFF_IGNORED = 120;
    static constexpr int32_t MINUS_EXPONENT_DIFF_IGNORED = -EXPONENT_DIFF_IGNORED;
    T mantissa;
    int32_t exp;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    CUDA_CRAP HDRFloat() {
        mantissa = 0.0;
        exp = MIN_BIG_EXPONENT();
    }

    CUDA_CRAP HDRFloat(const HDRFloat<T> &other) {
        mantissa = other.mantissa;
        exp = other.exp;
    }

    template<class SrcT>
    CUDA_CRAP HDRFloat(const HDRFloat<SrcT>& other) {
        mantissa = (T)other.mantissa;
        exp = other.exp;
    }

    CUDA_CRAP HDRFloat(T mantissa, int32_t exp) {
        this->mantissa = mantissa;
        this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
    }

    CUDA_CRAP HDRFloat(int32_t exp, T mantissa) {
        this->mantissa = mantissa;
        this->exp = exp;
    }

    CUDA_CRAP HDRFloat(int32_t exp, T mantissa, bool /*check*/) {
        this->mantissa = mantissa;
        if (mantissa == 0) {
            this->exp = MIN_BIG_EXPONENT();
        }
        else {
            this->exp = exp;
        }
    }

    CUDA_CRAP HDRFloat(T number) {
        if (number == 0) {
            mantissa = 0;
            exp = MIN_BIG_EXPONENT();
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            uint64_t bits = *reinterpret_cast<uint64_t*>(&number);
            int32_t f_exp = (int32_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT();
            uint64_t val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            T f_val = *reinterpret_cast<T*>(&val);

            mantissa = f_val;
            exp = f_exp;
        }
        else if constexpr (std::is_same<T, float>::value) {
            uint32_t bits = *reinterpret_cast<uint32_t*>(&number);
            int32_t f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT();
            uint64_t val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
            T f_val = *reinterpret_cast<T*>(&val);

            mantissa = f_val;
            exp = f_exp;
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
        exp = temp_exp;
    }
#endif

    CUDA_CRAP void Reduce() {
        if (mantissa == 0) {
            return;
        }

        if constexpr (std::is_same<T, double>::value) {
            uint64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
            int32_t f_exp = (int32_t)((bits & 0x7FF0'0000'0000'0000UL) >> 52UL) + MIN_SMALL_EXPONENT();
            uint64_t val = (bits & 0x800F'FFFF'FFFF'FFFFL) | 0x3FF0'0000'0000'0000L;
            T f_val = *reinterpret_cast<T*>(&val);
            exp += f_exp;
            mantissa = f_val;
        }
        else if constexpr (std::is_same<T, float>::value) {
            uint64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
            int32_t f_exp = (int32_t)((bits & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT();
            uint64_t val = (bits & 0x807F'FFFFL) | 0x3F80'0000L;
            T f_val = *reinterpret_cast<T*>(&val);
            exp += f_exp;
            mantissa = f_val;
        }
    }

    static CUDA_CRAP T getMultiplier(int32_t scaleFactor) {
        if (scaleFactor <= MIN_SMALL_EXPONENT()) {
            return 0.0;
        }
        else if (scaleFactor >= 1024) {
            return INFINITY;
        }

        return (T)twoPowExp[(int)scaleFactor - MinDoubleExponent];
    }

    CUDA_CRAP T toDouble() const
    {
        return mantissa * getMultiplier(exp);
    }

    CUDA_CRAP T toDoubleSub(int32_t exponent) const
    {
        return mantissa * getMultiplier(exp - exponent);
    }

    CUDA_CRAP explicit operator T() const { return toDouble(); }

    CUDA_CRAP T getMantissa() const { return  mantissa; }

    CUDA_CRAP int32_t getExp() const { return exp; }

    CUDA_CRAP void setExp(int32_t localexp) {
        this->exp = localexp;
    }

    CUDA_CRAP HDRFloat divide(HDRFloat factor) const {
        T local_mantissa = this->mantissa / factor.mantissa;
        int32_t local_exp = this->exp - factor.exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP HDRFloat reciprocal() const {
        T local_mantissa = 1.0 / this->mantissa;
        int32_t local_exp = -this->exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP HDRFloat &reciprocal_mutable() {
        mantissa = 1.0 / mantissa;
        exp = -exp;

        return *this;
    }

    CUDA_CRAP HDRFloat &divide_mutable(HDRFloat factor) {
        T local_mantissa = this->mantissa / factor.mantissa;
        int32_t local_exp = this->exp - factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat divide(T factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return divide(factorMant);
    }

    CUDA_CRAP HDRFloat &divide_mutable(T factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return divide_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator/=(const HDRFloat& other) {
        return divide_mutable(other);
    }

    CUDA_CRAP HDRFloat multiply(HDRFloat factor) const {
        T local_mantissa = this->mantissa * factor.mantissa;
        int32_t local_exp = this->exp + factor.exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP HDRFloat multiply(T factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply(factorMant);
    }

    CUDA_CRAP HDRFloat &multiply_mutable(HDRFloat factor) {
        T local_mantissa = this->mantissa * factor.mantissa;
        int32_t local_exp = this->exp + factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat &multiply_mutable(T factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator*=(const HDRFloat& other) {
        return multiply_mutable(other);
    }

    CUDA_CRAP HDRFloat square() const {
        T local_mantissa = this->mantissa * this->mantissa;
        int32_t local_exp = this->exp << 1;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP HDRFloat &square_mutable() {
        T local_mantissa = this->mantissa * this->mantissa;
        int32_t local_exp = this->exp << 1;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;
        return *this;
    }

    CUDA_CRAP HDRFloat multiply2() const {
        return HDRFloat(exp + 1, mantissa);
    }

    CUDA_CRAP HDRFloat &multiply2_mutable() {
        exp++;
        return *this;
    }

    CUDA_CRAP HDRFloat multiply4() const {
        return HDRFloat(exp + 2, mantissa);
    }

    CUDA_CRAP HDRFloat &multiply4_mutable() {
        exp += 2;
        return *this;
    }


    CUDA_CRAP HDRFloat divide2() const {
        return HDRFloat(mantissa, exp - 1);
    }

    CUDA_CRAP HDRFloat &divide2_mutable() {
        exp--;
        this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
        return *this;
    }

    CUDA_CRAP HDRFloat divide4() const {
        return HDRFloat(mantissa, exp - 2);
    }

    CUDA_CRAP HDRFloat &divide4_mutable() {
        exp -= 2;
        exp--;
        this->exp = exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : exp;
        return *this;
    }

    CUDA_CRAP HDRFloat addOld(HDRFloat value) const {

        T temp_mantissa = 0;
        int32_t temp_exp = exp;

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

    CUDA_CRAP HDRFloat add(HDRFloat value) const {

        int32_t expDiff = exp - value.exp;

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

    CUDA_CRAP HDRFloat &add_mutable(HDRFloat value) {

        int32_t expDiff = exp - value.exp;

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
    friend CUDA_CRAP HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator+=(const HDRFloat& other) {
        return add_mutable(other);
    }

    CUDA_CRAP HDRFloat subtract(HDRFloat value) const {

        int32_t expDiff = exp - value.exp;

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

    CUDA_CRAP HDRFloat &subtract_mutable(HDRFloat value) {

        int32_t expDiff = exp - value.exp;

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
    friend CUDA_CRAP HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator-=(const HDRFloat& other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP HDRFloat add(T value) const {
        return add(HDRFloat(value));
    }

    CUDA_CRAP HDRFloat &add_mutable(T value) {
        return add_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat subtract(T value) const {
        return subtract(HDRFloat(value));
    }

    CUDA_CRAP HDRFloat &subtract_mutable(T value) {
        return subtract_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator-=(const T& other) {
        return subtract_mutable(other);
    }

    CUDA_CRAP HDRFloat negate() const {
        return HDRFloat(exp, -mantissa);
    }

    CUDA_CRAP HDRFloat &negate_mutable() {
        mantissa = -mantissa;
        return *this;
    }

    friend CUDA_CRAP HDRFloat operator-(HDRFloat lhs)
    {
        return lhs.negate();
    }

    CUDA_CRAP int compareToBothPositiveReduced(HDRFloat compareTo) const {
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
    CUDA_CRAP int compareToBothPositive(HDRFloat compareTo) const {
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
    CUDA_CRAP int compareTo(HDRFloat compareTo) const {
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
    CUDA_CRAP int compareToReduced(HDRFloat compareToReduced) const {

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

    friend CUDA_CRAP bool operator<(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) < 0;
    }

    friend CUDA_CRAP bool operator<=(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) <= 0;
    }

    friend CUDA_CRAP bool operator>(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) > 0;
    }

    friend CUDA_CRAP bool operator>=(const HDRFloat& l, const HDRFloat& r)
    {
        return l.compareTo(r) >= 0;
    }

    static CUDA_CRAP HDRFloat HDRMax(HDRFloat a, HDRFloat b) {
        return a.compareTo(b) > 0 ? a : b;
    }

    static CUDA_CRAP HDRFloat maxBothPositive(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositive(b) > 0 ? a : b;
    }

    static CUDA_CRAP HDRFloat maxBothPositiveReduced(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositiveReduced(b) > 0 ? a : b;
    }

    static CUDA_CRAP HDRFloat minBothPositive(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositive(b) < 0 ? a : b;
    }
    static CUDA_CRAP HDRFloat minBothPositiveReduced(HDRFloat a, HDRFloat b) {
        return a.compareToBothPositiveReduced(b) < 0 ? a : b;
    }

    static CUDA_CRAP HDRFloat HDRMin(HDRFloat a, HDRFloat b) {
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
        bool isOdd = (incoming.exp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
                 ::sqrt(isOdd ? 2.0 * incoming.mantissa : incoming.mantissa));
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        bool isOdd = (incoming.exp & 1) != 0;
        return T(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
            ::sqrt(isOdd ? 2.0f * incoming.mantissa : incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP T HdrAbs(const T& incoming) {
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
static CUDA_CRAP void HdrReduce(T& incoming) {
    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
                  std::is_same<T, HDRFloat<float>>::value) {
        incoming.Reduce();
    }
}
