#pragma once

#include "HDRFloat.h"
#include <algorithm>
#include <math.h>

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

template<class T>
class HDRFloatComplex {

private:
    T mantissaReal;
    T mantissaImag;
    int32_t exp;

    using HDRFloat = HDRFloat<T>;
    using TExp = int32_t;

public:
    CUDA_CRAP constexpr HDRFloatComplex() {
        mantissaReal = 0.0;
        mantissaImag = 0.0;
        exp = HDRFloat::MIN_BIG_EXPONENT();
    }

    CUDA_CRAP constexpr HDRFloatComplex(T mantissaReal, T mantissaImag, TExp exp) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
    }

    CUDA_CRAP constexpr HDRFloatComplex(TExp exp, T mantissaReal, T mantissaImag) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp;
    }

    CUDA_CRAP constexpr HDRFloatComplex(const HDRFloatComplex &other) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = other.exp;
    }

    CUDA_CRAP constexpr HDRFloatComplex(const HDRFloatComplex &other, int exp) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = exp;
    }

    CUDA_CRAP constexpr HDRFloatComplex(const HDRFloat &re, const HDRFloat &im) {
        setMantexp(re, im);
    }

    CUDA_CRAP constexpr HDRFloatComplex(T re, T im) {
        setMantexp(HDRFloat(re), HDRFloat(im));
    }

private:
    void CUDA_CRAP setMantexp(const HDRFloat &realIn, const HDRFloat &imagIn) {

        exp = max(realIn.exp, imagIn.exp);
        mantissaReal = realIn.mantissa * HDRFloat::getMultiplier(realIn.exp-exp);
        mantissaImag = imagIn.mantissa * HDRFloat::getMultiplier(imagIn.exp-exp);
    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator+(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.plus_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    friend CUDA_CRAP constexpr HDRFloatComplex operator+(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.plus_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator+=(const HDRFloatComplex& other) {
        return plus_mutable(other);
    }

private:
    HDRFloatComplex CUDA_CRAP plus_mutable(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal + value.mantissaReal * mul;
            mantissaImag = mantissaImag + value.mantissaImag * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal + value.mantissaReal;
            mantissaImag = mantissaImag + value.mantissaImag;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            exp = value.exp;
            mantissaReal = mantissaReal * mul + value.mantissaReal;
            mantissaImag =  mantissaImag * mul + value.mantissaImag;
        } else {
            exp = value.exp;
            mantissaReal = value.mantissaReal;
            mantissaImag = value.mantissaImag;
        }
        return *this;

    }


    HDRFloatComplex CUDA_CRAP times(HDRFloatComplex factor) const {
        T tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        T tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/

    }

    HDRFloatComplex CUDA_CRAP times_mutable(HDRFloatComplex factor) {
        T tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        T tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        TExp localExp = this->exp + factor.exp;

        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = localExp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : localExp;

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator*(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.times_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator*(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.times_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator*=(const HDRFloatComplex& other) {
        return times_mutable(other);
    }

private:

    HDRFloatComplex CUDA_CRAP times(T factor) const {
        return times(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times_mutable(T factor) {
        return times_mutable(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times(HDRFloat factor) const {
        T tempMantissaReal = mantissaReal * factor.mantissa;

        T tempMantissaImag = mantissaImag * factor.mantissa;

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex CUDA_CRAP times_mutable(HDRFloat factor) {
        T tempMantissaReal = mantissaReal * factor.mantissa;

        T tempMantissaImag = mantissaImag * factor.mantissa;

        TExp expLocal = this->exp + factor.exp;

        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = expLocal < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : expLocal;

        return *this;
    }

    HDRFloatComplex CUDA_CRAP plus_mutable(T real) {
        return plus_mutable(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP plus_mutable(HDRFloat real) {

        TExp expDiff = exp - real.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal + real.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal + real.mantissa;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            exp = real.exp;
            mantissaReal = mantissaReal * mul + real.mantissa;
            mantissaImag =  mantissaImag * mul;
        } else {
            exp = real.exp;
            mantissaReal = real.mantissa;
            mantissaImag = 0.0;
        }
        return *this;
    }

    HDRFloatComplex CUDA_CRAP sub_mutable(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal - value.mantissaReal * mul;
            mantissaImag = mantissaImag - value.mantissaImag * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal - value.mantissaReal;
            mantissaImag = mantissaImag - value.mantissaImag;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            exp = value.exp;
            mantissaReal = mantissaReal * mul - value.mantissaReal;
            mantissaImag =  mantissaImag * mul - value.mantissaImag;
        } else {
            exp = value.exp;
            mantissaReal = -value.mantissaReal;
            mantissaImag = -value.mantissaImag;
        }
        return *this;

    }

    HDRFloatComplex CUDA_CRAP sub_mutable(HDRFloat real) {

        TExp expDiff = exp - real.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal - real.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal - real.mantissa;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            exp = real.exp;
            mantissaReal = mantissaReal * mul - real.mantissa;
            mantissaImag =  mantissaImag * mul;
        } else {
            exp = real.exp;
            mantissaReal = -real.mantissa;
            mantissaImag = 0;
        }
        return *this;

    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator-(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator-=(const HDRFloatComplex& other) {
        return subtract_mutable(other);
    }

private:

    HDRFloatComplex CUDA_CRAP sub(T real) const {
        return sub(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP sub_mutable(T real) {
        return sub_mutable(HDRFloat(real));
    }

public:

    void CUDA_CRAP Reduce() {
        if(mantissaReal == 0 && mantissaImag == 0) {
            return;
        }

        uint32_t bitsReal = *reinterpret_cast<uint32_t*>(&mantissaReal);
        TExp f_expReal = (TExp)((bitsReal & 0x7F80'0000UL) >> 23UL);

        uint32_t bitsImag = *reinterpret_cast<uint32_t*>(&mantissaImag);
        TExp f_expImag = (TExp)((bitsImag & 0x7F80'0000UL) >> 23UL);

        TExp expDiff = max(f_expReal, f_expImag) + HDRFloat::MIN_SMALL_EXPONENT();
        TExp expCombined = exp + expDiff;
        T mul = HDRFloat::getMultiplier(-expDiff);

        mantissaReal *= mul;
        mantissaImag *= mul;
        exp = expCombined;
    }
        
private:

    HDRFloatComplex CUDA_CRAP square_mutable() {
        T temp = mantissaReal * mantissaImag;

        TExp exp = this->exp << 1;
        mantissaReal = (mantissaReal + mantissaImag) * (mantissaReal - mantissaImag);
        mantissaImag = temp + temp;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(mantissaReal);
        T absIm = Math.abs(mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

public:

    HDRFloat CUDA_CRAP norm_squared() const {
        return HDRFloat(exp << 1, mantissaReal * mantissaReal + mantissaImag * mantissaImag);
    }

    HDRFloat CUDA_CRAP norm() const {
        return HDRFloat(exp, sqrt(mantissaReal * mantissaReal + mantissaImag * mantissaImag));
    }

    HDRFloatComplex CUDA_CRAP reciprocal() const {
        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        return HDRFloatComplex(mantissaReal * temp, -mantissaImag * temp, -exp);
    }

    HDRFloatComplex CUDA_CRAP reciprocal_mutable() {
        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        mantissaReal = mantissaReal * temp;
        mantissaImag = -mantissaImag * temp;
        exp = -exp;
        return *this;
    }

private:
    HDRFloatComplex CUDA_CRAP divide_mutable(HDRFloatComplex factor) {

        T temp = 1.0 / (factor.mantissaReal * factor.mantissaReal + factor.mantissaImag * factor.mantissaImag);

        T tempMantissaReal = (mantissaReal * factor.mantissaReal + mantissaImag * factor.mantissaImag) * temp;

        T tempMantissaImag = (mantissaImag * factor.mantissaReal - mantissaReal * factor.mantissaImag)  * temp;

        TExp exp = this->exp - factor.exp;
        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    HDRFloatComplex CUDA_CRAP divide_mutable(HDRFloat real) {

        TExp exp = this->exp - real.exp;
        T temp = 1.0 / real.mantissa;
        mantissaReal = mantissaReal * temp;
        mantissaImag = mantissaImag * temp;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(mantissaReal);
        T absIm = Math.abs(mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator/(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr HDRFloatComplex operator/(HDRFloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const T& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator/=(const HDRFloatComplex& other) {
        return divide_mutable(other);
    }

private:

    HDRFloatComplex CUDA_CRAP divide(T real) const {

        return divide(HDRFloat(real));

    }

    HDRFloatComplex CUDA_CRAP divide_mutable(T real) {

        return divide_mutable(HDRFloat(real));

    }

    HDRFloatComplex CUDA_CRAP abs() const {

        return HDRFloatComplex(exp, HdrAbs(mantissaReal), HdrAbs(mantissaImag));

    }

    
    HDRFloatComplex CUDA_CRAP abs_mutable() {

        mantissaReal = HdrAbs(mantissaReal);
        mantissaImag = HdrAbs(mantissaImag);
        return *this;

    }

public:

    HDRFloat CUDA_CRAP getRe() const {
        return HDRFloat(exp, mantissaReal);
    }

    HDRFloat CUDA_CRAP getIm() const {
        return HDRFloat(exp, mantissaImag);
    }

    T CUDA_CRAP getMantissaReal() const {
        return mantissaReal;
    }

    T CUDA_CRAP getMantissaImag() const {
        return mantissaImag;
    }

    TExp CUDA_CRAP getExp() const {
        return exp;
    }

    void CUDA_CRAP setExp(TExp exp) {
        this->exp = exp;
    }
    void CUDA_CRAP addExp(TExp exp) {
        this->exp += exp;
    }

    void CUDA_CRAP subExp(TExp exp) {
        this->exp -= exp;
        this->exp = this->exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : this->exp;
    }

    /*
     *  A*X + B*Y
     */
    static HDRFloatComplex CUDA_CRAP AtXpBtY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex B, HDRFloatComplex Y) {

        return A.times(X).plus_mutable(B.times(Y));

    }

    /*
     *  A*X + B*Y
     */
    static HDRFloatComplex CUDA_CRAP AtXpBtY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex B, HDRFloat Y) {

        return A.times(X).plus_mutable(B.times(Y));

    }

    /*
     *  A*X +Y
     */
    static HDRFloatComplex CUDA_CRAP AtXpY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex Y) {

        return A.times(X).plus_mutable(Y);

    }

    /*
     *  A*X +Y
     */
    static HDRFloatComplex CUDA_CRAP AtXpY(HDRFloatComplex A, HDRFloatComplex X, HDRFloat Y) {

        return A.times(X).plus_mutable(Y);

    }

    /*
     *  A*X
     */
    static HDRFloatComplex CUDA_CRAP AtX(HDRFloatComplex A, HDRFloatComplex X) {

        return A.times(X);

    }


    bool CUDA_CRAP equals(HDRFloatComplex z2) const {

        return z2.exp == exp && z2.mantissaReal == mantissaReal && z2.mantissaImag == mantissaImag;

    }

    void CUDA_CRAP assign(HDRFloatComplex z) {
        mantissaReal = z.mantissaReal;
        mantissaImag = z.mantissaImag;
        exp = z.exp;
    }

    void CUDA_CRAP toComplex(T &re, T &img) const {
        //return new Complex(mantissaReal * MantExp.toExp(exp), mantissaImag * MantExp.toExp(exp));
        auto d = HDRFloat::getMultiplier(exp);
        //return new Complex(MantExp.toDouble(mantissaReal, exp), MantExp.toDouble(mantissaImag, exp));
        re = mantissaReal * d;
        img = mantissaImag * d;
    }
    
    HDRFloat CUDA_CRAP chebychevNorm() const {
        return HDRFloat::maxBothPositive(HdrAbs(getRe()), HdrAbs(getIm()));
    }
};
