#pragma once

#include <stdint.h>
#include <math.h>
#include "HighPrecision.h"

#ifdef __CUDA_ARCH__
#define CUDA_CRAP __device__
#define CUDA_CRAP_BOTH __host__ __device__
static __device__ double* __restrict__ twoPowExpDbl;
static __device__ float* __restrict__ twoPowExpFlt;
#else
#define CUDA_CRAP
#define CUDA_CRAP_BOTH
extern double* twoPowExpDbl;
extern float* twoPowExpFlt;
#endif

template<class T>
class HDRFloatComplex;

CUDA_CRAP void InitStatics();

// TExp can work with float, int16_t or int64_t but all seem to offer worse perf
// float is quite close though.
//#ifndef __CUDA_ARCH__
template<class T, class TExp = int32_t>
class HDRFloat {
//#else
//template<class T, class TExp = int32_t>
//class __align__(8) HDRFloat {
//#endif
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
        if constexpr (
            std::is_same<TExp, int32_t>::value ||
            std::is_same<TExp, float>::value) {
            return INT32_MIN >> 3;
        }
        else {
            return INT16_MIN >> 3;
        }
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
        mantissa = (T)0.0;
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

    CUDA_CRAP explicit HDRFloat(const T number) { // TODO add constexpr once that compiles
        if (number == T(0)) {
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
            mantissa = (T)0.0;
            exp = MIN_BIG_EXPONENT();
            return;
        }

        int temp_exp;
        mantissa = boost::multiprecision::frexp(number, &temp_exp).template convert_to<T>();
        exp = (TExp)temp_exp;
    }
#endif

    CUDA_CRAP constexpr HDRFloat &Reduce() & {
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

        return *this;
    }

    CUDA_CRAP constexpr HDRFloat &&Reduce() && {
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

        return std::move(*this);
    }

    static CUDA_CRAP constexpr T getMultiplier(TExp scaleFactor) {
        if (scaleFactor <= MIN_SMALL_EXPONENT()) {
            return (T)0.0;
        }
        else if (scaleFactor >= 1024) {
            return (T)INFINITY;
        }

        if constexpr (std::is_same<T, double>::value) {
            return (T)twoPowExpDbl[(int)scaleFactor - MinDoubleExponent];
        }
        else {
            //return scalbnf(1.0, scaleFactor);
            return (T)twoPowExpFlt[(int)scaleFactor - MinFloatExponent];
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
                if (scaleFactor <= MIN_SMALL_EXPONENT()) {
                    return 0.0;
                }
            }

            return twoPowExpDbl[(int)scaleFactor - MinDoubleExponent];
        }
        else {
            if constexpr (IncludeCheck) {
                if (scaleFactor <= MIN_SMALL_EXPONENT()) {
                    return 0.0f;
                }
            }

            //return scalbnf(1.0, scaleFactor);
            return twoPowExpFlt[(int)scaleFactor - MinFloatExponent];
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

    CUDA_CRAP bool operator==(const HDRFloat<T>& other) const {
        if (this->exp == other.exp &&
            this->mantissa == other.mantissa) {
            return true;
        }

        return false;
    }

    CUDA_CRAP constexpr T getMantissa() const { return  mantissa; }

    CUDA_CRAP constexpr TExp getExp() const { return exp; }

    CUDA_CRAP constexpr void setExp(TExp localexp) {
        this->exp = localexp;
    }

    CUDA_CRAP constexpr HDRFloat reciprocal() const {
        T local_mantissa = (T)1.0 / this->mantissa;
        TExp local_exp = -this->exp;

        return HDRFloat(local_mantissa, local_exp);
    }

    CUDA_CRAP constexpr HDRFloat &reciprocal_mutable() {
        mantissa = (T)1.0 / mantissa;
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

    CUDA_CRAP constexpr HDRFloat &multiply_mutable(HDRFloat factor) {
        T local_mantissa = this->mantissa * factor.mantissa;
        TExp local_exp = this->exp + factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT() ? MIN_BIG_EXPONENT() : local_exp;
        return *this;
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
        const TExp maxexp = max(local_exp1, local_exp2);
        const TExp expDiff1 = local_exp1 - local_exp2;
        const T mul = getMultiplierNeg(-abs(expDiff1));

        T mantissaSum1;
        TExp expDiff2;
        TExp finalMaxexp;
        if constexpr (plus) {
            if (local_exp1 >= local_exp2) {
                // TODO constexpr double path
                mantissaSum1 = __fmaf_rn(local_mantissa2, mul, local_mantissa1);
            }
            else {
                mantissaSum1 = __fmaf_rn(local_mantissa1, mul, local_mantissa2);
            }
        }
        else {
            if (local_exp1 >= local_exp2) {
                mantissaSum1 = __fmaf_rn(-local_mantissa2, mul, local_mantissa1);
            }
            else {
                mantissaSum1 = __fmaf_rn(local_mantissa1, mul, -local_mantissa2);
            }
        }

        expDiff2 = maxexp - DeltaSub0.exp;
        finalMaxexp = max(maxexp, DeltaSub0.exp);

        const T mul2 = getMultiplierNeg(-abs(expDiff2));
        HDRFloat sum2;
        if (expDiff2 >= 0) {
            sum2 = HDRFloat(finalMaxexp, __fmaf_rn(DeltaSub0.mantissa, mul2, mantissaSum1));
        }
        else {
            sum2 = HDRFloat(finalMaxexp, __fmaf_rn(mantissaSum1, mul2, DeltaSub0.mantissa));
        }

        HdrReduce(sum2);
        return sum2;
    }

    static CUDA_CRAP HDRFloat custom_perturb2(
        HDRFloat& DeltaSubNX,
        HDRFloat& DeltaSubNY,
        const HDRFloat& DeltaSubNXOrig,
        const HDRFloat& tempSum1,
        const HDRFloat& DeltaSubNYOrig,
        const HDRFloat& tempSum2,
        const HDRFloat& DeltaSub0X,
        const HDRFloat& DeltaSub0Y) {

        const TExp local_exp1F = DeltaSubNXOrig.exp + tempSum1.exp;
        const TExp local_exp2F = DeltaSubNYOrig.exp + tempSum2.exp;
        const TExp maxexpF = max(local_exp1F, local_exp2F);
        const TExp expDiff1F = local_exp1F - local_exp2F;
        const T mulF = getMultiplierNeg(-abs(expDiff1F));

        const T local_mantissa1F = DeltaSubNXOrig.mantissa * tempSum1.mantissa;
        const T local_mantissa2F = DeltaSubNYOrig.mantissa * tempSum2.mantissa;
        T mantissaSum1F;
        if (expDiff1F >= 0) {
            mantissaSum1F = __fmaf_rn(-local_mantissa2F, mulF, local_mantissa1F);
        }
        else {
            mantissaSum1F = __fmaf_rn(local_mantissa1F, mulF, -local_mantissa2F);
        }

        const TExp expDiff2F = maxexpF - DeltaSub0X.exp;
        const TExp finalMaxexpF = max(maxexpF, DeltaSub0X.exp);
        const T mul2F = getMultiplierNeg(-abs(expDiff2F));
        //const int ZeroOrOne = (int)(expDiff2F >= 0);
        //DeltaSubNX = HDRFloat(ZeroOrOne * finalMaxexpF, ZeroOrOne * __fmaf_rn(DeltaSub0X.mantissa, mul2F, mantissaSum1F)) +
        //             HDRFloat((1 - ZeroOrOne) * finalMaxexpF, (1 - ZeroOrOne) * __fmaf_rn(mantissaSum1F, mul2F, DeltaSub0X.mantissa));
        if (expDiff2F >= 0) {
            DeltaSubNX = HDRFloat(finalMaxexpF, __fmaf_rn(DeltaSub0X.mantissa, mul2F, mantissaSum1F));
        }
        else {
            DeltaSubNX = HDRFloat(finalMaxexpF, __fmaf_rn(mantissaSum1F, mul2F, DeltaSub0X.mantissa));
        }

        HdrReduce(DeltaSubNX);

        const TExp local_exp1T = DeltaSubNXOrig.exp + tempSum2.exp;
        const TExp local_exp2T = DeltaSubNYOrig.exp + tempSum1.exp;
        const TExp maxexpT = max(local_exp1T, local_exp2T);
        const TExp expDiff1T = local_exp1T - local_exp2T;
        const T mulT = getMultiplierNeg(-abs(expDiff1T));

        const T local_mantissa1T = DeltaSubNXOrig.mantissa * tempSum2.mantissa;
        const T local_mantissa2T = DeltaSubNYOrig.mantissa * tempSum1.mantissa;
        T mantissaSum1T;
        if (expDiff1T >= 0) {
            // TODO constexpr double path
            mantissaSum1T = __fmaf_rn(local_mantissa2T, mulT, local_mantissa1T);
        }
        else {
            mantissaSum1T = __fmaf_rn(local_mantissa1T, mulT, local_mantissa2T);
        }

        const TExp expDiff2T = maxexpT - DeltaSub0Y.exp;
        const TExp finalMaxexpT = max(maxexpT, DeltaSub0Y.exp);
        const T mul2T = getMultiplierNeg(-abs(expDiff2T));

        if (expDiff2T >= 0) {
            DeltaSubNY = HDRFloat(finalMaxexpT, __fmaf_rn(DeltaSub0Y.mantissa, mul2T, mantissaSum1T));
        }
        else {
            DeltaSubNY = HDRFloat(finalMaxexpT, __fmaf_rn(mantissaSum1T, mul2T, DeltaSub0Y.mantissa));
        }

        HdrReduce(DeltaSubNY);
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

        return HDRFloat(local_exp, local_mantissa);
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

    CUDA_CRAP constexpr HDRFloat add(HDRFloat value) const {

        TExp expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(exp, mantissa, true);
        }
        else if (expDiff >= 0) {
            T mul = getMultiplierNeg(-expDiff);
            return HDRFloat(exp, mantissa + value.mantissa * mul, true);
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
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
            T mul = getMultiplierNeg(-expDiff);
            mantissa = mantissa + value.mantissa * mul;
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
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
            T mul = getMultiplierNeg(-expDiff);
            return HDRFloat(exp, mantissa - value.mantissa * mul, true);
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
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
            T mul = getMultiplierNeg(-expDiff);
            mantissa = mantissa - value.mantissa * mul;
        }
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = getMultiplierNeg(expDiff);
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
    static_assert(
        std::is_same<T, double>::value ||
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
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloatComplex<double>>::value ||
        std::is_same<T, HDRFloatComplex<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloatComplex<double>>::value ||
        std::is_same<T, HDRFloatComplex<float>>::value) {
        incoming.Reduce();
    }
}

template<class T>
static CUDA_CRAP constexpr T &&HdrReduce(T&& incoming) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    using no_const = std::remove_cv_t<T>;
    static_assert(
        std::is_same<no_const, double>::value ||
        std::is_same<no_const, float>::value ||
        std::is_same<no_const, HDRFloat<double>>::value ||
        std::is_same<no_const, HDRFloat<float>>::value ||
        std::is_same<no_const, HDRFloatComplex<double>>::value ||
        std::is_same<no_const, HDRFloatComplex<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<no_const, HDRFloat<double>>::value ||
                  std::is_same<no_const, HDRFloat<float>>::value ||
                  std::is_same<no_const, HDRFloatComplex<double>>::value ||
                  std::is_same<no_const, HDRFloatComplex<float>>::value) {
        return std::move(incoming.Reduce());
    }

    return std::move(incoming);
}

template<class T>
static CUDA_CRAP constexpr T HdrMaxPositiveReduced(const T& one, const T& two) {
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
                  std::is_same<T, HDRFloat<float>>::value) {
        if (one.compareToBothPositiveReduced(two) > 0) {
            return one;
        }

        return two;
    }
    else {
        return (one > two) ? one : two;
    }
}

template<class T>
static CUDA_CRAP constexpr T HdrMinPositiveReduced(const T& one, const T& two) {
    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
                  std::is_same<T, HDRFloat<float>>::value) {
        if (one.compareToBothPositiveReduced(two) < 0) {
            return one;
        }

        return two;
    }
    else {
        return (one < two) ? one : two;
    }
}

template<class T>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedLT(const T& one, const T& two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
                  std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) < 0;
    }
    else {
        return one < two;
    }
}

template<class T>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedLE(const T& one, const T& two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) <= 0;
    }
    else {
        return one <= two;
    }
}

template<class T>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedGT(const T& one, const T& two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) > 0;
    }
    else {
        return one > two;
    }
}

template<class T>
static CUDA_CRAP constexpr bool HdrCompareToBothPositiveReducedGE(const T& one, const T& two) {
    constexpr auto HighPrecPossible = // LOLZ I imagine there's a nicer way to do this here
#ifndef __CUDACC__
        std::is_same<T, HighPrecision>::value;
#else
        false;
#endif

    static_assert(
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        HighPrecPossible, "No");

    if constexpr (std::is_same<T, HDRFloat<double>>::value ||
        std::is_same<T, HDRFloat<float>>::value) {
        return one.compareToBothPositiveReduced(two) >= 0;
    }
    else {
        return one >= two;
    }
}
