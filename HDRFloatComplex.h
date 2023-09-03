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
    using Complex = std::complex<T>;
    using TExp = int32_t;

public:
    CUDA_CRAP HDRFloatComplex() {
        mantissaReal = 0.0;
        mantissaImag = 0.0;
        exp = HDRFloat::MIN_BIG_EXPONENT();
    }

    CUDA_CRAP HDRFloatComplex(T mantissaReal, T mantissaImag, TExp exp) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
    }

    CUDA_CRAP HDRFloatComplex(TExp exp, T mantissaReal, T mantissaImag) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp;
    }

    CUDA_CRAP HDRFloatComplex(const HDRFloatComplex &other) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = other.exp;
    }

    CUDA_CRAP HDRFloatComplex(HDRFloatComplex other, int exp) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = exp;
    }

    CUDA_CRAP HDRFloatComplex(Complex c) {
        setMantexp(HDRFloat(c.real()), HDRFloat(c.imag()));
    }

    CUDA_CRAP HDRFloatComplex(HDRFloat re, HDRFloat im) {
        setMantexp(re, im);
    }

    CUDA_CRAP HDRFloatComplex(T re, T im) {
        setMantexp(HDRFloat(re), HDRFloat(im));
    }

private:
    void CUDA_CRAP setMantexp(HDRFloat realIn, HDRFloat imagIn) {

        exp = max(realIn.exp, imagIn.exp);
        mantissaReal = realIn.mantissa * HDRFloat::getMultiplier(realIn.exp-exp);
        mantissaImag = imagIn.mantissa * HDRFloat::getMultiplier(imagIn.exp-exp);

        /*if (realIn.exp == imagIn.exp) {
            exp = realIn.exp;
            mantissaReal = realIn.mantissa;
            mantissaImag = imagIn.mantissa;
        }
        else if (realIn.exp > imagIn.exp) {

            //T temp = imagIn.mantissa / HDRFloat::toExp(realIn.exp - imagIn.exp);

            exp = realIn.exp;

            mantissaReal = realIn.mantissa;
            //mantissaImag = temp;
            //mantissaImag = HDRFloat::toDouble(imagIn.mantissa, imagIn.exp - realIn.exp);
            mantissaImag = imagIn.mantissa * HDRFloat::getMultiplier(imagIn.exp - realIn.exp);
        }
        else {
            //T temp = realIn.mantissa / HDRFloat::toExp(imagIn.exp - realIn.exp);

            exp = imagIn.exp;

            //mantissaReal = temp;
            //mantissaReal = HDRFloat::toDouble(realIn.mantissa, realIn.exp - imagIn.exp);
            mantissaReal = realIn.mantissa * HDRFloat::getMultiplier(realIn.exp - imagIn.exp);
            mantissaImag = imagIn.mantissa;
        }*/
    }

public:

    HDRFloatComplex CUDA_CRAP plus(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return HDRFloatComplex(exp, mantissaReal, mantissaImag);
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            return HDRFloatComplex(exp, mantissaReal + value.mantissaReal * mul, mantissaImag + value.mantissaImag * mul);
        }
        /*else if(expDiff == 0) {
            return HDRFloatComplex(exp, mantissaReal + value.mantissaReal, mantissaImag + value.mantissaImag);
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            return HDRFloatComplex(value.exp, mantissaReal * mul + value.mantissaReal, mantissaImag * mul + value.mantissaImag);
        } else {
            return HDRFloatComplex(value.exp, value.mantissaReal, value.mantissaImag);
        }

    }

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


    HDRFloatComplex CUDA_CRAP times(HDRFloatComplex factor) {
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

    HDRFloatComplex CUDA_CRAP times(T factor) {
        return times(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times_mutable(T factor) {
        return times_mutable(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times(HDRFloat factor) {
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

        TExp exp = this->exp + factor.exp;

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

    HDRFloatComplex CUDA_CRAP divide2() {
        return HDRFloatComplex(mantissaReal, mantissaImag, exp - 1);
    }

    HDRFloatComplex CUDA_CRAP divide2_mutable() {

        TExp exp = this->exp - 1;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP divide4() {
        return HDRFloatComplex(mantissaReal, mantissaImag, exp - 2);
    }

    HDRFloatComplex CUDA_CRAP divide4_mutable() {

        TExp exp = this->exp - 2;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
        return *this;

    }

    
    HDRFloatComplex CUDA_CRAP times2() {
        return HDRFloatComplex(exp + 1, mantissaReal, mantissaImag);
    }

    
    HDRFloatComplex CUDA_CRAP times2_mutable() {

        exp++;
        return *this;

    }

    
    HDRFloatComplex CUDA_CRAP times4() {
        return HDRFloatComplex(exp + 2, mantissaReal, mantissaImag);
    }

    HDRFloatComplex CUDA_CRAP times4_mutable() {

        exp += 2;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP times8() {
        return HDRFloatComplex(exp + 3, mantissaReal, mantissaImag);
    }

    HDRFloatComplex CUDA_CRAP times8_mutable() {

        exp += 3;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP times16() {
        return HDRFloatComplex(exp + 4, mantissaReal, mantissaImag);
    }

    HDRFloatComplex CUDA_CRAP times16_mutable() {

        exp += 4;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP times32() {
        return HDRFloatComplex(exp + 5, mantissaReal, mantissaImag);
    }

    HDRFloatComplex CUDA_CRAP times32_mutable() {

        exp += 5;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP plus(T real) {
        return plus(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP plus_mutable(T real) {
        return plus_mutable(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP plus(HDRFloat real) {

        TExp expDiff = exp - real.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return HDRFloatComplex(exp, mantissaReal, mantissaImag);
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            return HDRFloatComplex(exp, mantissaReal + real.mantissa * mul, mantissaImag);
        }
        /*else if(expDiff == 0) {
            return HDRFloatComplex(exp, mantissaReal + real.mantissa, mantissaImag);
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            return HDRFloatComplex(real.exp, mantissaReal * mul + real.mantissa, mantissaImag * mul);
        } else {
            return HDRFloatComplex(real.exp, real.mantissa, 0.0);
        }
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

    HDRFloatComplex CUDA_CRAP sub(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return HDRFloatComplex(exp, mantissaReal, mantissaImag);
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            return HDRFloatComplex(exp, mantissaReal - value.mantissaReal * mul, mantissaImag - value.mantissaImag * mul);
        }
        /*else if(expDiff == 0) {
            return HDRFloatComplex(exp, mantissaReal - value.mantissaReal, mantissaImag - value.mantissaImag);
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            return HDRFloatComplex(value.exp, mantissaReal * mul - value.mantissaReal, mantissaImag * mul - value.mantissaImag);
        } else {
            return HDRFloatComplex(value.exp, -value.mantissaReal, -value.mantissaImag);
        }

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

    HDRFloatComplex CUDA_CRAP sub(HDRFloat real) {

        TExp expDiff = exp - real.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return HDRFloatComplex(exp, mantissaReal, mantissaImag);
        } else if(expDiff >= 0) {
            T mul = HDRFloat::getMultiplier(-expDiff);
            return HDRFloatComplex(exp, mantissaReal - real.mantissa * mul, mantissaImag);
        }
        /*else if(expDiff == 0) {
            return HDRFloatComplex(exp, mantissaReal - real.mantissa, mantissaImag);
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            T mul = HDRFloat::getMultiplier(expDiff);
            return HDRFloatComplex(real.exp, mantissaReal * mul - real.mantissa, mantissaImag * mul);
        } else {
            return HDRFloatComplex(real.exp, -real.mantissa, 0.0);
        }
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

    HDRFloatComplex CUDA_CRAP sub(T real) {
        return sub(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP sub_mutable(T real) {
        return sub_mutable(HDRFloat(real));
    }

    void CUDA_CRAP Reduce() {
        if(mantissaReal == 0 && mantissaImag == 0) {
            return;
        }

        //TExp bitsRe = Double.doubleToRawLongBits(mantissaReal);
        //TExp expDiffRe = ((bitsRe & 0x7FF0000000000000L) >> 52);

        //TExp bitsIm = Double.doubleToRawLongBits(mantissaImag);
        //TExp expDiffIm = ((bitsIm & 0x7FF0000000000000L) >> 52);

        //TExp expDiff = Math.max(expDiffRe, expDiffIm) + HDRFloat::MIN_SMALL_EXPONENT;

        //TExp expCombined = exp + expDiff;
        //T mul = HDRFloat::getMultiplier(-expDiff);
        //mantissaReal *= mul;
        //mantissaImag *= mul;
        //exp = expCombined;

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

    /*void Reduce2() {

        HDRFloat mantissaRealTemp = HDRFloat(mantissaReal);

        HDRFloat mantissaImagTemp = HDRFloat(mantissaImag);

        TExp realExp = exp;
        TExp imagExp = exp;

        boolean a1 = mantissaRealTemp.mantissa == mantissaReal;
        boolean a2 =  mantissaImagTemp.mantissa == mantissaImag;

        if(a1 && a2) {
           return;
        }
        else if(a1) {
            imagExp = mantissaImagTemp.exp + exp;
        }
        else if(a2) {
            realExp = mantissaRealTemp.exp + exp;
        }
        else {
            realExp = mantissaRealTemp.exp + exp;
            imagExp = mantissaImagTemp.exp + exp;
        }

        if (realExp == imagExp) {
            exp = realExp;

            mantissaImag = mantissaImagTemp.mantissa;
            mantissaReal = mantissaRealTemp.mantissa;
        }
        else if (realExp > imagExp) {
            //T mantissa_temp = mantissaImagTemp.mantissa / HDRFloat::toExp(realExp - imagExp);

            exp = realExp;
            mantissaReal = mantissaRealTemp.mantissa;
            //mantissaImag = mantissa_temp;
            //mantissaImag = HDRFloat::toDouble(mantissaImagTemp.mantissa, imagExp - realExp);
            mantissaImag = mantissaImagTemp.mantissa * HDRFloat::getMultiplier(imagExp - realExp);
        }
        else {
            //T mantissa_temp = mantissaRealTemp.mantissa / HDRFloat::toExp(imagExp - realExp);

            exp = imagExp;
            //mantissaReal = mantissa_temp;
            //mantissaReal = HDRFloat::toDouble(mantissaRealTemp.mantissa, realExp - imagExp);
            mantissaReal = mantissaRealTemp.mantissa * HDRFloat::getMultiplier(realExp - imagExp);
            mantissaImag = mantissaImagTemp.mantissa;
        }
    }*/

    
    HDRFloatComplex CUDA_CRAP square() {
        T temp = mantissaReal * mantissaImag;
        return HDRFloatComplex((mantissaReal + mantissaImag) * (mantissaReal - mantissaImag), temp + temp, exp << 1);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    
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

    
    HDRFloatComplex CUDA_CRAP cube() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(mantissaReal * (temp - 3 * temp2), mantissaImag * (3 * temp - temp2), 3 * exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex CUDA_CRAP cube_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        TExp exp = 3 * this->exp;
        mantissaReal = mantissaReal * (temp - 3 * temp2);
        mantissaImag = mantissaImag * (3 * temp - temp2);
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(mantissaReal);
        T absIm = Math.abs(mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    
    HDRFloatComplex CUDA_CRAP fourth() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(temp * (temp - 6 * temp2) + temp2 * temp2, 4 * mantissaReal * mantissaImag * (temp - temp2), exp << 2);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex CUDA_CRAP fourth_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        TExp exp = this->exp << 2;

        T temp_re = temp * (temp - 6 * temp2) + temp2 * temp2;
        mantissaImag = 4 * mantissaReal * mantissaImag * (temp - temp2);
        mantissaReal = temp_re;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(mantissaReal);
        T absIm = Math.abs(mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    
    HDRFloatComplex CUDA_CRAP fifth() {

        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(mantissaReal * (temp * temp + temp2 * (5 * temp2 - 10 * temp)), mantissaImag * (temp2 * temp2 + temp * (5 * temp - 10 * temp2)), 5 * exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex CUDA_CRAP fifth_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        TExp exp = 5 * this->exp;
        mantissaReal = mantissaReal * (temp * temp + temp2 * (5 * temp2 - 10 * temp));
        mantissaImag = mantissaImag * (temp2 * temp2 + temp * (5 * temp - 10 * temp2));
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        /*T absRe = Math.abs(mantissaReal);
        T absIm = Math.abs(mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    HDRFloat CUDA_CRAP norm_squared() {
        return HDRFloat(exp << 1, mantissaReal * mantissaReal + mantissaImag * mantissaImag);
    }

    HDRFloat CUDA_CRAP distance_squared(HDRFloatComplex other) {
        return sub(other).norm_squared();
    }

    HDRFloat CUDA_CRAP norm() {
        return HDRFloat(exp, sqrt(mantissaReal * mantissaReal + mantissaImag * mantissaImag));
    }

    HDRFloat CUDA_CRAP distance(HDRFloatComplex other) {
        return sub(other).norm();
    }

    HDRFloatComplex CUDA_CRAP divide(HDRFloatComplex factor) {

        T temp = 1.0 / (factor.mantissaReal * factor.mantissaReal + factor.mantissaImag * factor.mantissaImag);

        T tempMantissaReal = (mantissaReal * factor.mantissaReal + mantissaImag * factor.mantissaImag) * temp;

        T tempMantissaImag = (mantissaImag * factor.mantissaReal - mantissaReal * factor.mantissaImag)  * temp;

        return HDRFloatComplex(tempMantissaReal , tempMantissaImag, exp - factor.exp);

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

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

    HDRFloatComplex CUDA_CRAP reciprocal() {

        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);

        return HDRFloatComplex(mantissaReal * temp , -mantissaImag * temp, -exp);

    }

    HDRFloatComplex CUDA_CRAP reciprocal_mutable() {

        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);

        mantissaReal = mantissaReal * temp;
        mantissaImag = -mantissaImag * temp;
        exp = -exp;

        return *this;

    }


    HDRFloatComplex CUDA_CRAP divide(HDRFloat real) {

        T temp = 1.0 / real.mantissa;
        return HDRFloatComplex(mantissaReal * temp, mantissaImag * temp, exp - real.exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
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

    HDRFloatComplex CUDA_CRAP divide(T real) {

        return divide(HDRFloat(real));

    }

    HDRFloatComplex CUDA_CRAP divide_mutable(T real) {

        return divide_mutable(HDRFloat(real));

    }

    
    HDRFloatComplex CUDA_CRAP negative() {

        return HDRFloatComplex(exp, -mantissaReal, -mantissaImag);

    }

    
    HDRFloatComplex CUDA_CRAP negative_mutable() {

        mantissaReal = -mantissaReal;
        mantissaImag = -mantissaImag;
        return *this;

    }

    HDRFloatComplex CUDA_CRAP abs() {

        return HDRFloatComplex(exp, HdrAbs(mantissaReal), HdrAbs(mantissaImag));

    }

    
    HDRFloatComplex CUDA_CRAP abs_mutable() {

        mantissaReal = HdrAbs(mantissaReal);
        mantissaImag = HdrAbs(mantissaImag);
        return *this;

    }

    HDRFloatComplex CUDA_CRAP conjugate() {

        return HDRFloatComplex(exp, mantissaReal, -mantissaImag);

    }

    
    HDRFloatComplex CUDA_CRAP conjugate_mutable() {

        mantissaImag = -mantissaImag;
        return *this;

    }

    Complex CUDA_CRAP toComplex() {
        //return new Complex(mantissaReal * HDRFloat.toExp(exp), mantissaImag * HDRFloat.toExp(exp));
        T d = HDRFloat::getMultiplier(exp);
        //return new Complex(HDRFloat.toDouble(mantissaReal, exp), HDRFloat.toDouble(mantissaImag, exp));
        return Complex(mantissaReal * d, mantissaImag * d);
    }
    
    HDRFloat CUDA_CRAP getRe() {

        return HDRFloat(exp, mantissaReal);

    }

    HDRFloat CUDA_CRAP getIm() {

        return HDRFloat(exp, mantissaImag);

    }

    T CUDA_CRAP getMantissaReal() {
        return mantissaReal;
    }

    T CUDA_CRAP getMantissaImag() {
        return mantissaImag;
    }

    TExp CUDA_CRAP getExp() {

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

    static HDRFloat CUDA_CRAP DiffAbs(HDRFloat c, HDRFloat d)
    {
        HDRFloat cd = c.add(d);
        if (c.compareTo(HDRFloat{}) >= 0.0) {
            if (cd.compareTo(HDRFloat{}) >= 0.0) {
                return d;
            }
            else {
                return d.negate().subtract_mutable(c.multiply2());
            }
        }
        else {
            if (cd.compareTo(HDRFloat{}) > 0.0) {
                return d.add(c.multiply2());
            }
            else {
                return d.negate();
            }
        }
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


    bool CUDA_CRAP equals(HDRFloatComplex z2) {

        return z2.exp == exp && z2.mantissaReal == mantissaReal && z2.mantissaImag == mantissaImag;

    }

    void CUDA_CRAP assign(HDRFloatComplex z) {
        mantissaReal = z.mantissaReal;
        mantissaImag = z.mantissaImag;
        exp = z.exp;
    }

    
    HDRFloat CUDA_CRAP chebychevNorm() {
        return HDRFloat::maxBothPositive(HdrAbs(getRe()), HdrAbs(getIm()));
    }
};

