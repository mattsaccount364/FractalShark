#pragma once

#include "HDRFloat.h"
#include <algorithm>
#include <math.h>
//package fractalzoomer.core;
//
//
//import fractalzoomer.core.mpfr.MpfrBigNum;
//import fractalzoomer.core.mpir.MpirBigNum;
//import org.apfloat.Apfloat;

template<class T>
class HDRFloatComplex {

private:
    int32_t exp;
    T mantissaReal;
    T mantissaImag;

    using HDRFloat = HDRFloat<T>;
    using Complex = std::complex<T>;

public:
    HDRFloatComplex() {
        mantissaReal = 0.0;
        mantissaImag = 0.0;
        exp = HDRFloat::MIN_BIG_EXPONENT();
    }

    HDRFloatComplex(T mantissaReal, T mantissaImag, long exp) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
    }

    HDRFloatComplex(long exp, T mantissaReal, T mantissaImag) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp;
    }

    HDRFloatComplex(const HDRFloatComplex &other) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = other.exp;
    }

    HDRFloatComplex(HDRFloatComplex other, int exp) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
        this->exp = exp;
    }

    HDRFloatComplex(Complex c) {
        setMantexp(HDRFloat(c.real()), HDRFloat(c.imag()));
    }

    HDRFloatComplex(HDRFloat re, HDRFloat im) {
        setMantexp(re, im);
    }

    HDRFloatComplex(T re, T im) {
        setMantexp(HDRFloat(re), HDRFloat(im));
    }

private:
    void setMantexp(HDRFloat realIn, HDRFloat imagIn) {

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

    HDRFloatComplex plus(HDRFloatComplex value) {

        long expDiff = exp - value.exp;

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

    HDRFloatComplex plus_mutable(HDRFloatComplex value) {

        long expDiff = exp - value.exp;

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


    HDRFloatComplex times(HDRFloatComplex factor) {
        T tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        T tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/

    }

    HDRFloatComplex times_mutable(HDRFloatComplex factor) {
        T tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        T tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        long localExp = this->exp + factor.exp;

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

    HDRFloatComplex times(T factor) {
        return times(HDRFloat(factor));
    }

    HDRFloatComplex times_mutable(T factor) {
        return times_mutable(HDRFloat(factor));
    }

    HDRFloatComplex times(HDRFloat factor) {
        T tempMantissaReal = mantissaReal * factor.mantissa;

        T tempMantissaImag = mantissaImag * factor.mantissa;

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*T absRe = Math.abs(tempMantissaReal);
        T absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex times_mutable(HDRFloat factor) {
        T tempMantissaReal = mantissaReal * factor.mantissa;

        T tempMantissaImag = mantissaImag * factor.mantissa;

        long exp = this->exp + factor.exp;

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

    HDRFloatComplex divide2() {
        return HDRFloatComplex(mantissaReal, mantissaImag, exp - 1);
    }

    HDRFloatComplex divide2_mutable() {

        long exp = this->exp - 1;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
        return *this;

    }

    HDRFloatComplex divide4() {
        return HDRFloatComplex(mantissaReal, mantissaImag, exp - 2);
    }

    HDRFloatComplex divide4_mutable() {

        long exp = this->exp - 2;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
        return *this;

    }

    
    HDRFloatComplex times2() {
        return HDRFloatComplex(exp + 1, mantissaReal, mantissaImag);
    }

    
    HDRFloatComplex times2_mutable() {

        exp++;
        return *this;

    }

    
    HDRFloatComplex times4() {
        return HDRFloatComplex(exp + 2, mantissaReal, mantissaImag);
    }

    HDRFloatComplex times4_mutable() {

        exp += 2;
        return *this;

    }

    HDRFloatComplex times8() {
        return HDRFloatComplex(exp + 3, mantissaReal, mantissaImag);
    }

    HDRFloatComplex times8_mutable() {

        exp += 3;
        return *this;

    }

    HDRFloatComplex times16() {
        return HDRFloatComplex(exp + 4, mantissaReal, mantissaImag);
    }

    HDRFloatComplex times16_mutable() {

        exp += 4;
        return *this;

    }

    HDRFloatComplex times32() {
        return HDRFloatComplex(exp + 5, mantissaReal, mantissaImag);
    }

    HDRFloatComplex times32_mutable() {

        exp += 5;
        return *this;

    }

    HDRFloatComplex plus(T real) {
        return plus(HDRFloat(real));
    }

    HDRFloatComplex plus_mutable(T real) {
        return plus_mutable(HDRFloat(real));
    }

    HDRFloatComplex plus(HDRFloat real) {

        long expDiff = exp - real.exp;

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

    HDRFloatComplex plus_mutable(HDRFloat real) {

        long expDiff = exp - real.exp;

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

    HDRFloatComplex sub(HDRFloatComplex value) {

        long expDiff = exp - value.exp;

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

    HDRFloatComplex sub_mutable(HDRFloatComplex value) {

        long expDiff = exp - value.exp;

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

    HDRFloatComplex sub(HDRFloat real) {

        long expDiff = exp - real.exp;

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

    HDRFloatComplex sub_mutable(HDRFloat real) {

        long expDiff = exp - real.exp;

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

    HDRFloatComplex sub(T real) {
        return sub(HDRFloat(real));
    }

    HDRFloatComplex sub_mutable(T real) {
        return sub_mutable(HDRFloat(real));
    }

    void Reduce() {
        if(mantissaReal == 0 && mantissaImag == 0) {
            return;
        }

        assert(false);

        //long bitsRe = Double.doubleToRawLongBits(mantissaReal);
        //long expDiffRe = ((bitsRe & 0x7FF0000000000000L) >> 52);

        //long bitsIm = Double.doubleToRawLongBits(mantissaImag);
        //long expDiffIm = ((bitsIm & 0x7FF0000000000000L) >> 52);

        //long expDiff = Math.max(expDiffRe, expDiffIm) + HDRFloat::MIN_SMALL_EXPONENT;

        //long expCombined = exp + expDiff;
        //T mul = HDRFloat::getMultiplier(-expDiff);
        //mantissaReal *= mul;
        //mantissaImag *= mul;
        //exp = expCombined;
    }


    /*void Reduce2() {

        HDRFloat mantissaRealTemp = HDRFloat(mantissaReal);

        HDRFloat mantissaImagTemp = HDRFloat(mantissaImag);

        long realExp = exp;
        long imagExp = exp;

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

    
    HDRFloatComplex square() {
        T temp = mantissaReal * mantissaImag;
        return HDRFloatComplex((mantissaReal + mantissaImag) * (mantissaReal - mantissaImag), temp + temp, exp << 1);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    
    HDRFloatComplex square_mutable() {
        T temp = mantissaReal * mantissaImag;

        long exp = this->exp << 1;
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

    
    HDRFloatComplex cube() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(mantissaReal * (temp - 3 * temp2), mantissaImag * (3 * temp - temp2), 3 * exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex cube_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        long exp = 3 * this->exp;
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

    
    HDRFloatComplex fourth() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(temp * (temp - 6 * temp2) + temp2 * temp2, 4 * mantissaReal * mantissaImag * (temp - temp2), exp << 2);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex fourth_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        long exp = this->exp << 2;

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

    
    HDRFloatComplex fifth() {

        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        return HDRFloatComplex(mantissaReal * (temp * temp + temp2 * (5 * temp2 - 10 * temp)), mantissaImag * (temp2 * temp2 + temp * (5 * temp - 10 * temp2)), 5 * exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex fifth_mutable() {
        T temp = mantissaReal * mantissaReal;
        T temp2 = mantissaImag * mantissaImag;

        long exp = 5 * this->exp;
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

    HDRFloat norm_squared() {
        return HDRFloat(exp << 1, mantissaReal * mantissaReal + mantissaImag * mantissaImag);
    }

    HDRFloat distance_squared(HDRFloatComplex other) {
        return sub(other).norm_squared();
    }

    HDRFloat norm() {
        return HDRFloat(exp, sqrt(mantissaReal * mantissaReal + mantissaImag * mantissaImag));
    }

    HDRFloat distance(HDRFloatComplex other) {
        return sub(other).norm();
    }

    HDRFloatComplex divide(HDRFloatComplex factor) {

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

    HDRFloatComplex divide_mutable(HDRFloatComplex factor) {

        T temp = 1.0 / (factor.mantissaReal * factor.mantissaReal + factor.mantissaImag * factor.mantissaImag);

        T tempMantissaReal = (mantissaReal * factor.mantissaReal + mantissaImag * factor.mantissaImag) * temp;

        T tempMantissaImag = (mantissaImag * factor.mantissaReal - mantissaReal * factor.mantissaImag)  * temp;

        long exp = this->exp - factor.exp;
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

    HDRFloatComplex reciprocal() {

        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);

        return HDRFloatComplex(mantissaReal * temp , -mantissaImag * temp, -exp);

    }

    HDRFloatComplex reciprocal_mutable() {

        T temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);

        mantissaReal = mantissaReal * temp;
        mantissaImag = -mantissaImag * temp;
        exp = -exp;

        return *this;

    }


    HDRFloatComplex divide(HDRFloat real) {

        T temp = 1.0 / real.mantissa;
        return HDRFloatComplex(mantissaReal * temp, mantissaImag * temp, exp - real.exp);

        /*T absRe = Math.abs(p.mantissaReal);
        T absIm = Math.abs(p.mantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/
    }

    HDRFloatComplex divide_mutable(HDRFloat real) {

        long exp = this->exp - real.exp;
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

    HDRFloatComplex divide(T real) {

        return divide(HDRFloat(real));

    }

    HDRFloatComplex divide_mutable(T real) {

        return divide_mutable(HDRFloat(real));

    }

    
    HDRFloatComplex negative() {

        return HDRFloatComplex(exp, -mantissaReal, -mantissaImag);

    }

    
    HDRFloatComplex negative_mutable() {

        mantissaReal = -mantissaReal;
        mantissaImag = -mantissaImag;
        return *this;

    }

    HDRFloatComplex abs() {

        return HDRFloatComplex(exp, HdrAbs(mantissaReal), HdrAbs(mantissaImag));

    }

    
    HDRFloatComplex abs_mutable() {

        mantissaReal = HdrAbs(mantissaReal);
        mantissaImag = HdrAbs(mantissaImag);
        return *this;

    }

    HDRFloatComplex conjugate() {

        return HDRFloatComplex(exp, mantissaReal, -mantissaImag);

    }

    
    HDRFloatComplex conjugate_mutable() {

        mantissaImag = -mantissaImag;
        return *this;

    }

    Complex toComplex() {
        //return new Complex(mantissaReal * HDRFloat.toExp(exp), mantissaImag * HDRFloat.toExp(exp));
        T d = HDRFloat::getMultiplier(exp);
        //return new Complex(HDRFloat.toDouble(mantissaReal, exp), HDRFloat.toDouble(mantissaImag, exp));
        return Complex(mantissaReal * d, mantissaImag * d);
    }
    
    HDRFloat getRe() {

        return HDRFloat(exp, mantissaReal);

    }

    HDRFloat getIm() {

        return HDRFloat(exp, mantissaImag);

    }

    T getMantissaReal() {
        return mantissaReal;
    }

    T getMantissaImag() {
        return mantissaImag;
    }

    long getExp() {

        return exp;

    }

    void setExp(long exp) {
        this->exp = exp;
    }
    void addExp(long exp) {
        this->exp += exp;
    }

    void subExp(long exp) {
        this->exp -= exp;
        this->exp = this->exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : this->exp;
    }

    static HDRFloat DiffAbs(HDRFloat c, HDRFloat d)
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
    static HDRFloatComplex AtXpBtY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex B, HDRFloatComplex Y) {

        return A.times(X).plus_mutable(B.times(Y));

    }

    /*
     *  A*X + B*Y
     */
    static HDRFloatComplex AtXpBtY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex B, HDRFloat Y) {

        return A.times(X).plus_mutable(B.times(Y));

    }

    /*
     *  A*X +Y
     */
    static HDRFloatComplex AtXpY(HDRFloatComplex A, HDRFloatComplex X, HDRFloatComplex Y) {

        return A.times(X).plus_mutable(Y);

    }

    /*
     *  A*X +Y
     */
    static HDRFloatComplex AtXpY(HDRFloatComplex A, HDRFloatComplex X, HDRFloat Y) {

        return A.times(X).plus_mutable(Y);

    }

    /*
     *  A*X
     */
    static HDRFloatComplex AtX(HDRFloatComplex A, HDRFloatComplex X) {

        return A.times(X);

    }


    bool equals(HDRFloatComplex z2) {

        return z2.exp == exp && z2.mantissaReal == mantissaReal && z2.mantissaImag == mantissaImag;

    }

    void assign(HDRFloatComplex z) {
        mantissaReal = z.mantissaReal;
        mantissaImag = z.mantissaImag;
        exp = z.exp;
    }

    
    HDRFloat chebychevNorm() {
        return HDRFloat::maxBothPositive(HdrAbs(getRe()), HdrAbs(getIm()));
    }

    
    HDRFloatComplex toMantExpComplex() {return *this;}
};

