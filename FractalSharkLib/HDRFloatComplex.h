#pragma once

#include "HDRFloat.h"
#include <algorithm>
#include <math.h>

template<class SubType>
class HDRFloatComplex {

private:
    SubType mantissaReal;
    SubType mantissaImag;
    int32_t exp;

    using HDRFloat = HDRFloat<SubType>;
    using TExp = int32_t;

public:
    using TemplateSubType = SubType;

    friend class HDRFloatComplex<float>;
    friend class HDRFloatComplex<double>;
    friend class HDRFloatComplex<CudaDblflt<MattDblflt>>;
    friend class HDRFloatComplex<CudaDblflt<dblflt>>;

#ifndef __CUDACC__ 
    CUDA_CRAP std::string ToString() const {
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<double>::max_digits10);
        ss << "mantissaReal: " << static_cast<double>(this->mantissaReal)
            << " mantissaImag: " << static_cast<double>(this->mantissaImag)
            << " exp: " << this->exp;
        return ss.str();
    }
#endif

    CUDA_CRAP constexpr HDRFloatComplex() {
        mantissaReal = SubType{};
        mantissaImag = SubType{};
        exp = HDRFloat::MIN_BIG_EXPONENT();
    }

    CUDA_CRAP constexpr HDRFloatComplex(SubType mantissaReal, SubType mantissaImag, TExp exp) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;
    }

    CUDA_CRAP constexpr HDRFloatComplex(TExp exp, SubType mantissaReal, SubType mantissaImag) {
        this->mantissaReal = mantissaReal;
        this->mantissaImag = mantissaImag;
        this->exp = exp;
    }

    template<class SubType2>
    CUDA_CRAP constexpr explicit HDRFloatComplex(const HDRFloatComplex<SubType2> &other) {
        this->mantissaReal = (SubType)other.mantissaReal;
        this->mantissaImag = (SubType)other.mantissaImag;
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

    CUDA_CRAP constexpr HDRFloatComplex(SubType re, SubType im) {
        setMantexp(HDRFloat(re), HDRFloat(im));
    }

private:
    void CUDA_CRAP setMantexp(const HDRFloat &realIn, const HDRFloat &imagIn) {

        exp = CudaHostMax(realIn.exp, imagIn.exp);
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
        plus_mutable(other);
        return *this;
    }

private:
    HDRFloatComplex CUDA_CRAP plus_mutable(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            SubType mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal + value.mantissaReal * mul;
            mantissaImag = mantissaImag + value.mantissaImag * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal + value.mantissaReal;
            mantissaImag = mantissaImag + value.mantissaImag;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            SubType mul = HDRFloat::getMultiplier(expDiff);
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
        SubType tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        SubType tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*SubType absRe = Math.abs(tempMantissaReal);
        SubType absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/

    }

    HDRFloatComplex CUDA_CRAP times_mutable(HDRFloatComplex factor) {
        SubType tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        SubType tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        TExp localExp = this->exp + factor.exp;

        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = localExp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : localExp;

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
        times_mutable(other);
        return *this;
    }

private:

    HDRFloatComplex CUDA_CRAP times(SubType factor) const {
        return times(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times_mutable(SubType factor) {
        return times_mutable(HDRFloat(factor));
    }

    HDRFloatComplex CUDA_CRAP times(HDRFloat factor) const {
        SubType tempMantissaReal = mantissaReal * factor.mantissa;

        SubType tempMantissaImag = mantissaImag * factor.mantissa;

        return HDRFloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);
    }

    HDRFloatComplex CUDA_CRAP times_mutable(HDRFloat factor) {
        SubType tempMantissaReal = mantissaReal * factor.mantissa;

        SubType tempMantissaImag = mantissaImag * factor.mantissa;

        TExp expLocal = this->exp + factor.exp;

        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = expLocal < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : expLocal;

        return *this;
    }

    HDRFloatComplex CUDA_CRAP plus_mutable(SubType real) {
        return plus_mutable(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP plus_mutable(HDRFloat real) {

        TExp expDiff = exp - real.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            SubType mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal + real.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal + real.mantissa;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            SubType mul = HDRFloat::getMultiplier(expDiff);
            exp = real.exp;
            mantissaReal = mantissaReal * mul + real.mantissa;
            mantissaImag =  mantissaImag * mul;
        } else {
            exp = real.exp;
            mantissaReal = real.mantissa;
            mantissaImag = SubType{};
        }
        return *this;
    }

    HDRFloatComplex CUDA_CRAP sub_mutable(HDRFloatComplex value) {

        TExp expDiff = exp - value.exp;

        if(expDiff >= HDRFloat::EXPONENT_DIFF_IGNORED) {
            return *this;
        } else if(expDiff >= 0) {
            SubType mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal - value.mantissaReal * mul;
            mantissaImag = mantissaImag - value.mantissaImag * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal - value.mantissaReal;
            mantissaImag = mantissaImag - value.mantissaImag;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            SubType mul = HDRFloat::getMultiplier(expDiff);
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
            SubType mul = HDRFloat::getMultiplier(-expDiff);
            mantissaReal = mantissaReal - real.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissaReal = mantissaReal - real.mantissa;
        }*/
        else if(expDiff > HDRFloat::MINUS_EXPONENT_DIFF_IGNORED) {
            SubType mul = HDRFloat::getMultiplier(expDiff);
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
        lhs.sub_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator-=(const HDRFloatComplex& other) {
        sub_mutable(other);
        return *this;
    }

private:

    HDRFloatComplex CUDA_CRAP sub(SubType real) const {
        return sub(HDRFloat(real));
    }

    HDRFloatComplex CUDA_CRAP sub_mutable(SubType real) {
        return sub_mutable(HDRFloat(real));
    }

public:

    void CUDA_CRAP Reduce() {
        if (mantissaReal == SubType{} && mantissaImag == SubType{}) {
            return;
        }

        TExp f_expReal;
        TExp f_expImag;

        auto helper = [&](int32_t min_small) {
            TExp expDiff = CudaHostMax(f_expReal, f_expImag) + min_small;
            TExp expCombined = exp + expDiff;
            SubType mul = HDRFloat::getMultiplier(-expDiff);

            mantissaReal *= mul;
            mantissaImag *= mul;
            exp = expCombined;
        };

        static_assert(
            std::is_same<SubType, double>::value ||
            std::is_same<SubType, float>::value ||
            std::is_same<SubType, CudaDblflt<dblflt>>::value, "!");

        if constexpr (std::is_same<SubType, double>::value) {
            uint64_t bitsReal = *reinterpret_cast<uint64_t*>(&mantissaReal);
            f_expReal = (TExp)((bitsReal & 0x7FF0'0000'0000'0000UL) >> 52UL);

            uint64_t bitsImag = *reinterpret_cast<uint64_t*>(&mantissaImag);
            f_expImag = (TExp)((bitsImag & 0x7FF0'0000'0000'0000UL) >> 52UL);
            helper(HDRFloat::MIN_SMALL_EXPONENT_DOUBLE());
        } else if constexpr (std::is_same<SubType, float>::value) {
            uint32_t bitsReal = *reinterpret_cast<uint32_t*>(&mantissaReal);
            f_expReal = (TExp)((bitsReal & 0x7F80'0000UL) >> 23UL);

            uint32_t bitsImag = *reinterpret_cast<uint32_t*>(&mantissaImag);
            f_expImag = (TExp)((bitsImag & 0x7F80'0000UL) >> 23UL);
            helper(HDRFloat::MIN_SMALL_EXPONENT_FLOAT());
        } else if constexpr (std::is_same<SubType, CudaDblflt<dblflt>>::value) {
            HDRFloat tempReal(mantissaReal);
            HDRFloat tempImag(mantissaImag);
            tempReal.Reduce();
            tempImag.Reduce();
            TExp tempExp = this->exp;
            setMantexp(tempReal, tempImag);
            this->exp += tempExp;

            //TExp f_expReal, f_expImag;
            //mantissaReal.Reduce(f_expReal);
            //mantissaImag.Reduce(f_expImag);

            //TExp expDiff = std::CudaHostMax(f_expReal, f_expImag) + HDRFloat::MIN_SMALL_EXPONENT_FLOAT();
            //TExp expCombined = exp + expDiff;
            //exp = expCombined;
        }
    }
        
private:

    HDRFloatComplex CUDA_CRAP square_mutable() {
        SubType temp = mantissaReal * mantissaImag;

        TExp exp = this->exp << 1;
        mantissaReal = (mantissaReal + mantissaImag) * (mantissaReal - mantissaImag);
        mantissaImag = temp + temp;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

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
        SubType temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        return HDRFloatComplex(mantissaReal * temp, -mantissaImag * temp, -exp);
    }

    HDRFloatComplex CUDA_CRAP reciprocal_mutable() {
        SubType temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        mantissaReal = mantissaReal * temp;
        mantissaImag = -mantissaImag * temp;
        exp = -exp;
        return *this;
    }

private:
    HDRFloatComplex CUDA_CRAP divide_mutable(HDRFloatComplex factor) {

        SubType temp = 1.0 / (factor.mantissaReal * factor.mantissaReal + factor.mantissaImag * factor.mantissaImag);

        SubType tempMantissaReal = (mantissaReal * factor.mantissaReal + mantissaImag * factor.mantissaImag) * temp;

        SubType tempMantissaImag = (mantissaImag * factor.mantissaReal - mantissaReal * factor.mantissaImag)  * temp;

        TExp exp = this->exp - factor.exp;
        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

        return *this;
    }

    HDRFloatComplex CUDA_CRAP divide_mutable(HDRFloat real) {

        TExp exp = this->exp - real.exp;
        SubType temp = 1.0 / real.mantissa;
        mantissaReal = mantissaReal * temp;
        mantissaImag = mantissaImag * temp;
        this->exp = exp < HDRFloat::MIN_BIG_EXPONENT() ? HDRFloat::MIN_BIG_EXPONENT() : exp;

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
        const SubType& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr HDRFloatComplex& operator/=(const HDRFloatComplex& other) {
        divide_mutable(other);
        return *this;
    }

private:

    HDRFloatComplex CUDA_CRAP divide_mutable(SubType real) {

        return divide_mutable(HDRFloat(real));

    }

public:

    HDRFloat CUDA_CRAP getRe() const {
        return HDRFloat(exp, mantissaReal);
    }

    HDRFloat CUDA_CRAP getIm() const {
        return HDRFloat(exp, mantissaImag);
    }

    TExp CUDA_CRAP getExp() const {
        return exp;
    }

private:
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

public:
    void CUDA_CRAP toComplex(SubType &re, SubType &img) const {
        //return new Complex(mantissaReal * MantExp.toExp(exp), mantissaImag * MantExp.toExp(exp));
        auto d = HDRFloat::getMultiplier(exp);
        //return new Complex(MantExp.toDouble(mantissaReal, exp), MantExp.toDouble(mantissaImag, exp));
        re = mantissaReal * d;
        img = mantissaImag * d;
    }

    HDRFloat CUDA_CRAP chebychevNorm() const {
        return HDRFloat::maxBothPositiveReduced(HdrAbs(getRe()), HdrAbs(getIm()));
    }
};

#include "FloatComplex.h"