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

class HDRFloat {
public:
    static constexpr int64_t MIN_SMALL_EXPONENT = -1023;
    static constexpr int64_t MIN_BIG_EXPONENT = INT64_MIN >> 3;

//private:
//    static double LN2;
//    static double LN2_REC;

public:
    static constexpr int64_t EXPONENT_DIFF_IGNORED = 120;
    static constexpr int64_t MINUS_EXPONENT_DIFF_IGNORED = -EXPONENT_DIFF_IGNORED;
    double mantissa;
    int64_t exp;

    static constexpr int MaxDoubleExponent = 1023;
    static constexpr int MinDoubleExponent = -1022;

    CUDA_CRAP HDRFloat() {
        mantissa = 0.0;
        exp = MIN_BIG_EXPONENT;
    }

    CUDA_CRAP HDRFloat(const HDRFloat &other) {
        mantissa = other.mantissa;
        exp = other.exp;
    }

    CUDA_CRAP HDRFloat(double mantissa, int64_t exp) {
        this->mantissa = mantissa;
        this->exp = exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : exp;
    }

    CUDA_CRAP HDRFloat(int64_t exp, double mantissa) {
        this->mantissa = mantissa;
        this->exp = exp;
    }

    CUDA_CRAP HDRFloat(int64_t exp, double mantissa, bool /*check*/) {
        this->mantissa = mantissa;
        if (mantissa == 0) {
            this->exp = MIN_BIG_EXPONENT;
        }
        else {
            this->exp = exp;
        }
    }

    CUDA_CRAP HDRFloat(double number) {
        if (number == 0) {
            mantissa = 0;
            exp = MIN_BIG_EXPONENT;
            return;
        }

        uint64_t bits = *reinterpret_cast<uint64_t*>(&number);
        int64_t f_exp = (int64_t)((bits & 0x7FF0000000000000UL) >> 52UL) + MIN_SMALL_EXPONENT;
        uint64_t val = (bits & 0x800FFFFFFFFFFFFFL) | 0x3FF0000000000000L;
        double f_val = *reinterpret_cast<double *>(&val);

        mantissa = f_val;
        exp = f_exp;
    }

#ifndef __CUDACC__ 
    HDRFloat(HighPrecision number) {

        if (number == 0) {
            mantissa = 0.0;
            exp = MIN_BIG_EXPONENT;
            return;
        }

        int temp_exp;
        mantissa = boost::multiprecision::frexp(number, &temp_exp).template convert_to<double>();
        exp = temp_exp;

        /*
        auto s = double{ boost::multiprecision::floor(boost::multiprecision::log(number)) + 1 };
        double double_exp = s - 1;

        int64_t long_exp = 0;

        //double double_exp = double{localExp};

        if (double_exp < 0) {
            long_exp = (int64_t)(double_exp - 0.5);
            HighPrecision twoToExp = boost::multiprecision::pow(HighPrecision(2), -long_exp);
            mantissa = double(number * twoToExp);
        }
        else {
            long_exp = (int64_t)(double_exp + 0.5);
            HighPrecision twoToExp = boost::multiprecision::pow(HighPrecision(2), long_exp);
            mantissa = double(number / twoToExp);
        }

        this->exp = long_exp;
        */

    }
#endif

    CUDA_CRAP void Reduce() {
        if (mantissa == 0) {
            return;
        }

        int64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
        int64_t f_exp = ((bits & 0x7FF0000000000000L) >> 52) + MIN_SMALL_EXPONENT;

        uint64_t val = (bits & 0x800FFFFFFFFFFFFFL) | 0x3FF0000000000000L;
        double f_val = *reinterpret_cast<double *>(&val);

        exp += f_exp;
        mantissa = f_val;
    }

    /*public void Reduce() {

       if(mantissa == 0) {
           return;
       }

       long bits = Double.doubleToRawLongBits(mantissa);
       long f_exp = ((bits & 0x7FF0000000000000L) >> 52) + MIN_SMALL_EXPONENT;
       exp += f_exp;
       mantissa = mantissa * getMultiplier(-f_exp);
   }*/

   /*public double toDouble() {
     return mantissa * toExp(exp);
   }*/

   /*public static double toExp(double exp) {

       if (exp <= MIN_SMALL_EXPONENT) {
           return 0.0;
       }
       else if (exp >= 1024) {
           return Math.pow(2, 1024);
       }
       return Math.pow(2, exp);

   }*/

    static CUDA_CRAP double getMultiplier(int64_t scaleFactor) {
        if (scaleFactor <= MIN_SMALL_EXPONENT) {
            return 0.0;
        }
        else if (scaleFactor >= 1024) {
            return INFINITY;
        }

        return twoPowExp[(int)scaleFactor - MinDoubleExponent];
    }

    //    double toDouble()
    //    {
    //        return toDouble(mantissa, exp);
    //    }

    CUDA_CRAP double toDouble() const
    {
        return mantissa * getMultiplier(exp);
    }

    CUDA_CRAP double toDoubleSub(int64_t exponent) const
    {
        return mantissa * getMultiplier(exp - exponent);
    }

    explicit operator double() const { return toDouble(); }

    CUDA_CRAP double getMantissa() const { return  mantissa; }

    CUDA_CRAP int64_t getExp() const { return exp; }

    CUDA_CRAP void setExp(int64_t localexp) {
        this->exp = localexp;
    }


    /*
    static double toDouble(double mantissa, int64_t exp)
    {
        if(mantissa == 0) {
            return 0.0;
        }

        int64_t bits = *reinterpret_cast<uint64_t*>(&mantissa);
        int64_t f_exp = ((bits & 0x7FF0000000000000L) >> 52) + MIN_SMALL_EXPONENT;

        int64_t sum_exp = exp + f_exp;

        if (sum_exp <= MIN_SMALL_EXPONENT) {
            return 0.0;
        }
        else if (sum_exp >= 1024) {
            return mantissa * Double.POSITIVE_INFINITY;
        }

        return Double.longBitsToDouble((bits & 0x800FFFFFFFFFFFFFL) | ((sum_exp - MIN_SMALL_EXPONENT) << 52));
    }*/

    CUDA_CRAP HDRFloat divide(HDRFloat factor) const {
        double local_mantissa = this->mantissa / factor.mantissa;
        int64_t local_exp = this->exp - factor.exp;

        return HDRFloat(local_mantissa, local_exp);

        /*double abs = fabs(res.mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            res.Reduce();
        }*/
    }

    CUDA_CRAP HDRFloat reciprocal() const {
        double local_mantissa = 1.0 / this->mantissa;
        int64_t local_exp = -this->exp;

        return HDRFloat(local_mantissa, local_exp);

        /*double abs = fabs(res.mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            res.Reduce();
        }*/
    }

    CUDA_CRAP HDRFloat &reciprocal_mutable() {
        mantissa = 1.0 / mantissa;
        exp = -exp;

        return *this;
    }

    CUDA_CRAP HDRFloat &divide_mutable(HDRFloat factor) {
        double local_mantissa = this->mantissa / factor.mantissa;
        int64_t local_exp = this->exp - factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : local_exp;

        /*double abs = fabs(local_mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat divide(double factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return divide(factorMant);
    }

    CUDA_CRAP HDRFloat &divide_mutable(double factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return divide_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator/(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const double& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator/=(const HDRFloat& other) {
        return divide_mutable(other);
    }

    CUDA_CRAP HDRFloat multiply(HDRFloat factor) const {
        double local_mantissa = this->mantissa * factor.mantissa;
        int64_t local_exp = this->exp + factor.exp;

        return HDRFloat(local_mantissa, local_exp);

        /*double abs = fabs(res.mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            res.Reduce();
        }*/
    }

    CUDA_CRAP HDRFloat multiply(double factor) const {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply(factorMant);
    }

    CUDA_CRAP HDRFloat &multiply_mutable(HDRFloat factor) {
        double local_mantissa = this->mantissa * factor.mantissa;
        int64_t local_exp = this->exp + factor.exp;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : local_exp;

        /*double abs = fabs(local_mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            Reduce();
        }*/

        return *this;
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const HDRFloat& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat &multiply_mutable(double factor) {
        HDRFloat factorMant = HDRFloat(factor);
        return multiply_mutable(factorMant);
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP HDRFloat operator*(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const double& rhs) // otherwise, both parameters may be const references
    {
        lhs.multiply_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator*=(const HDRFloat& other) {
        return multiply_mutable(other);
    }

    CUDA_CRAP HDRFloat square() const {
        double local_mantissa = this->mantissa * this->mantissa;
        int64_t local_exp = this->exp << 1;

        return HDRFloat(local_mantissa, local_exp);

        /*double abs = fabs(res.mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            res.Reduce();
        }*/
    }

    CUDA_CRAP HDRFloat &square_mutable() {
        double local_mantissa = this->mantissa * this->mantissa;
        int64_t local_exp = this->exp << 1;

        this->mantissa = local_mantissa;
        this->exp = local_exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : local_exp;

        /*double abs = fabs(local_mantissa);
        if (abs > 1e50 || abs < 1e-50) {
            Reduce();
        }*/

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
        this->exp = exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : exp;
        return *this;
    }

    CUDA_CRAP HDRFloat divide4() const {
        return HDRFloat(mantissa, exp - 2);
    }

    CUDA_CRAP HDRFloat &divide4_mutable() {
        exp -= 2;
        exp--;
        this->exp = exp < MIN_BIG_EXPONENT ? MIN_BIG_EXPONENT : exp;
        return *this;
    }

    CUDA_CRAP HDRFloat addOld(HDRFloat value) const {

        double temp_mantissa = 0;
        int64_t temp_exp = exp;

        if (exp == value.exp) {
            temp_mantissa = mantissa + value.mantissa;
        }
        else if (exp > value.exp) {
            //temp_mantissa = value.mantissa / toExp(exp - value.exp);
            temp_mantissa = mantissa + getMultiplier(value.exp - exp) * value.mantissa;
        }
        else {
            //temp_mantissa  = mantissa / toExp(value.exp - exp);
            temp_mantissa = getMultiplier(exp - value.exp) * mantissa;
            temp_exp = value.exp;
            temp_mantissa = temp_mantissa + value.mantissa;
        }

        return HDRFloat(temp_exp, temp_mantissa);

    }

    CUDA_CRAP HDRFloat add(HDRFloat value) const {

        int64_t expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(exp, mantissa, true);
        }
        else if (expDiff >= 0) {
            double mul = getMultiplier(-expDiff);
            return HDRFloat(exp, mantissa + value.mantissa * mul, true);
        }
        /*else if(expDiff == 0) {
            return HDRFloat(exp, mantissa + value.mantissa, true);
        }*/
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            double mul = getMultiplier(expDiff);
            return HDRFloat(value.exp, mantissa * mul + value.mantissa, true);
        }
        else {
            return HDRFloat(value.exp, value.mantissa, true);
        }

        /*double temp_mantissa = 0;
        int64_t temp_exp = exp;

        if (exp == value.exp) {
            temp_mantissa = mantissa + value.mantissa;
        }
        else if(exp > value.exp){
            //temp_mantissa = value.mantissa / toExp(exp - value.exp);
            temp_mantissa = toDouble(value.mantissa, value.exp - exp);
            temp_mantissa = mantissa + temp_mantissa;
        }
        else {
            //temp_mantissa  = mantissa / toExp(value.exp - exp);
            temp_mantissa = toDouble(mantissa, exp - value.exp);
            temp_exp = value.exp;
            temp_mantissa = temp_mantissa + value.mantissa;
        }

        return HDRFloat(temp_exp, temp_mantissa);*/

    }

    CUDA_CRAP HDRFloat &add_mutable(HDRFloat value) {

        int64_t expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        }
        else if (expDiff >= 0) {
            double mul = getMultiplier(-expDiff);
            mantissa = mantissa + value.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissa = mantissa + value.mantissa;
        }*/
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            double mul = getMultiplier(expDiff);
            exp = value.exp;
            mantissa = mantissa * mul + value.mantissa;
        }
        else {
            exp = value.exp;
            mantissa = value.mantissa;
        }

        if (mantissa == 0) {
            exp = MIN_BIG_EXPONENT;
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

        int64_t expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return HDRFloat(exp, mantissa, true);
        }
        else if (expDiff >= 0) {
            double mul = getMultiplier(-expDiff);
            return HDRFloat(exp, mantissa - value.mantissa * mul, true);
        }
        /*else if(expDiff == 0) {
            return HDRFloat(exp, mantissa - value.mantissa, true);
        }*/
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            double mul = getMultiplier(expDiff);
            return HDRFloat(value.exp, mantissa * mul - value.mantissa, true);
        }
        else {
            return HDRFloat(value.exp, -value.mantissa, true);
        }
    }

    CUDA_CRAP HDRFloat &subtract_mutable(HDRFloat value) {

        int64_t expDiff = exp - value.exp;

        if (expDiff >= EXPONENT_DIFF_IGNORED) {
            return *this;
        }
        else if (expDiff >= 0) {
            double mul = getMultiplier(-expDiff);
            mantissa = mantissa - value.mantissa * mul;
        }
        /*else if(expDiff == 0) {
            mantissa = mantissa - value.mantissa;
        }*/
        else if (expDiff > MINUS_EXPONENT_DIFF_IGNORED) {
            double mul = getMultiplier(expDiff);
            exp = value.exp;
            mantissa = mantissa * mul - value.mantissa;
        }
        else {
            exp = value.exp;
            mantissa = -value.mantissa;
        }

        if (mantissa == 0) {
            exp = MIN_BIG_EXPONENT;
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

    CUDA_CRAP HDRFloat add(double value) const {
        return add(HDRFloat(value));
    }

    CUDA_CRAP HDRFloat &add_mutable(double value) {
        return add_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP HDRFloat operator+(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const double& rhs) // otherwise, both parameters may be const references
    {
        lhs.add_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat subtract(double value) const {
        return subtract(HDRFloat(value));
    }

    CUDA_CRAP HDRFloat &subtract_mutable(double value) {
        return subtract_mutable(HDRFloat(value));
    }

    friend CUDA_CRAP HDRFloat operator-(HDRFloat lhs,        // passing lhs by value helps optimize chained a+b+c
        const double& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP HDRFloat& operator-=(const double& other) {
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

        //        if(mantissa == 0 && compareTo.mantissa == 0) {
        //            return 0;
        //        }

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
        //        if(mantissa == 0 && compareTo.mantissa == 0) {
        //            return 0;
        //        }

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

    // Matt TODO:
    //CUDA_CRAP double log2() const {
    //    return ::log(mantissa) * LN2_REC + exp;
    //}

    CUDA_CRAP int64_t log2approx() const {
        int64_t bits = *reinterpret_cast<const uint64_t*>(&mantissa);
        int64_t exponent = ((bits & 0x7FF0000000000000L) >> 52) + MIN_SMALL_EXPONENT;
        return exponent + exp;
    }

    // Matt TODO:
    //CUDA_CRAP double log() const {
    //    return ::log(mantissa) + exp * LN2;
    //}

    static CUDA_CRAP int64_t getExponent(double val) {
        int64_t bits = *reinterpret_cast<uint64_t*>(&val);
        return  ((bits & 0x7FF0000000000000L) >> 52) + MIN_SMALL_EXPONENT;
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

    CUDA_CRAP HDRFloat pow(int n) const {
        switch (n) {
        case 0: return HDRFloat(1.0);
        case 1: return HDRFloat(*this);
        case 2: return square();
        case 3: return multiply(square());
        case 4: return square().square();
        case 5: return multiply(square().square());
        case 6: return multiply(square()).square();
        case 7: return multiply(multiply(square()).square());
        case 8: return square().square().square();
        default:
        {
            if (n < 0) {
                return HDRFloat();
            }
            HDRFloat y = HDRFloat(1.0);
            HDRFloat x = HDRFloat(*this);
            while (n > 1)
            {
                if ((n & 1) != 0)
                    y.multiply_mutable(x);

                x.square_mutable();
                n >>= 1;
            }
            return x.multiply(y);
        }
        }
    }
};

template<class T>
static CUDA_CRAP T HdrSqrt(const T &incoming) {
    //double dexp = exp * 0.5;
    //return HDRFloat(sqrt(mantissa) * pow(2, dexp - (int)dexp), (int)dexp);

    static_assert(std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat>::value, "No");

    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, float>::value) {
        return sqrt((T)incoming);
    }
    else if constexpr (std::is_same<T, HDRFloat>::value) {
        bool isOdd = (incoming.exp & 1) != 0;
        return HDRFloat(isOdd ? (incoming.exp - 1) / 2 : incoming.exp / 2,
                        ::sqrt(isOdd ? 2.0 * incoming.mantissa : incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP T HdrAbs(const T& incoming) {
    static_assert(std::is_same<T, double>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, HDRFloat>::value, "No");

    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, float>::value) {
        return fabs((T)incoming);
    } else if constexpr (std::is_same<T, HDRFloat>::value) {
        return HDRFloat(incoming.exp, fabs(incoming.mantissa));
    }
}

template<class T>
static CUDA_CRAP void HdrReduce(T& incoming) {
    if constexpr (std::is_same<T, HDRFloat>::value) {
        incoming.Reduce();
    }
}
