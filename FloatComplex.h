#pragma once


template<class SubType>
class FloatComplex {

private:
    SubType mantissaReal;
    SubType mantissaImag;

public:
    using TemplateSubType = SubType;

    friend class FloatComplex<float>;
    friend class FloatComplex<double>;
    friend class FloatComplex<CudaDblflt<MattDblflt>>;
    friend class FloatComplex<CudaDblflt<dblflt>>;

    CUDA_CRAP constexpr FloatComplex() {
        mantissaReal = SubType{};
        mantissaImag = SubType{};
    }

    template<class SubType2>
    CUDA_CRAP constexpr explicit FloatComplex(const FloatComplex<SubType2>& other) {
        this->mantissaReal = (SubType)other.mantissaReal;
        this->mantissaImag = (SubType)other.mantissaImag;
    }

    CUDA_CRAP constexpr FloatComplex(const FloatComplex& other) {
        this->mantissaReal = other.mantissaReal;
        this->mantissaImag = other.mantissaImag;
    }

    CUDA_CRAP constexpr FloatComplex(const SubType re, const SubType im) {
        setMantexp(re, im);
    }


private:
    void CUDA_CRAP setMantexp(const SubType& realIn, const SubType& imagIn) {

        mantissaReal = realIn;
        mantissaImag = imagIn;
    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator+(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const FloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.plus_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    friend CUDA_CRAP constexpr FloatComplex operator+(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const SubType& rhs) // otherwise, both parameters may be const references
    {
        lhs.plus_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr FloatComplex& operator+=(const FloatComplex& other) {
        return plus_mutable(other);
    }

private:
    FloatComplex CUDA_CRAP plus_mutable(FloatComplex value) {
        mantissaReal = mantissaReal + value.mantissaReal;
        mantissaImag = mantissaImag + value.mantissaImag;
        return *this;

    }

    FloatComplex CUDA_CRAP times(FloatComplex factor) const {
        SubType tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        SubType tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        return FloatComplex(tempMantissaReal, tempMantissaImag, exp + factor.exp);

        /*SubType absRe = Math.abs(tempMantissaReal);
        SubType absIm = Math.abs(tempMantissaImag);
        if (absRe > 1e50 || absIm > 1e50 || absRe < 1e-50 || absIm < 1e-50) {
            p.Reduce();
        }*/

    }

    FloatComplex CUDA_CRAP times_mutable(FloatComplex factor) {
        mantissaReal *= factor.mantissaReal;
        mantissaImag *= factor.mantissaImag;
        return *this;
    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator*(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const FloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.times_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator*(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const SubType& rhs) // otherwise, both parameters may be const references
    {
        lhs.times_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr FloatComplex& operator*=(const FloatComplex& other) {
        return times_mutable(other);
    }

private:

    FloatComplex CUDA_CRAP times_mutable(SubType factor) {
        return times_mutable(SubType(factor));
    }

    FloatComplex CUDA_CRAP times(SubType factor) const {
        SubType tempMantissaReal = mantissaReal * factor;
        SubType tempMantissaImag = mantissaImag * factor;

        return FloatComplex(tempMantissaReal, tempMantissaImag);
    }

    FloatComplex CUDA_CRAP plus_mutable(SubType real) {
        mantissaReal += real;
        return *this;
    }

    FloatComplex CUDA_CRAP sub_mutable(FloatComplex value) {

        mantissaReal = mantissaReal - value.mantissaReal;
        mantissaImag = mantissaImag - value.mantissaImag;
        return *this;

    }

    FloatComplex CUDA_CRAP sub_mutable(SubType real) {

        mantissaReal = mantissaReal - real.mantissa;
        return *this;

    }

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator-(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const FloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.subtract_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr FloatComplex& operator-=(const FloatComplex& other) {
        return subtract_mutable(other);
    }

private:

    FloatComplex CUDA_CRAP sub(SubType real) const {
        return sub(SubType(real));
    }

public:

    void CUDA_CRAP Reduce() {
    }

private:

    //FloatComplex CUDA_CRAP square_mutable() {
    //    static_assert(false, "!");
    //    return *this;
    //}

public:

    SubType CUDA_CRAP norm_squared() const {
        return SubType(mantissaReal * mantissaReal + mantissaImag * mantissaImag);
    }

    SubType CUDA_CRAP norm() const {
        return SubType(sqrt(mantissaReal * mantissaReal + mantissaImag * mantissaImag));
    }

    FloatComplex CUDA_CRAP reciprocal() const {
        SubType temp = SubType{ 1 } / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        return FloatComplex(mantissaReal * temp, mantissaImag * temp);
    }

    FloatComplex CUDA_CRAP reciprocal_mutable() {
        SubType temp = 1.0f / (mantissaReal * mantissaReal + mantissaImag * mantissaImag);
        mantissaReal = mantissaReal * temp;
        mantissaImag = -mantissaImag * temp;
        return *this;
    }

private:
    //FloatComplex CUDA_CRAP divide_mutable(FloatComplex factor) {

    //    return *this;
    //}

public:

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator/(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const FloatComplex& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    // friends defined inside class body are inline and are hidden from non-ADL lookup
    friend CUDA_CRAP constexpr FloatComplex operator/(FloatComplex lhs,        // passing lhs by value helps optimize chained a+b+c
        const SubType& rhs) // otherwise, both parameters may be const references
    {
        lhs.divide_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr FloatComplex& operator/=(const FloatComplex& other) {
        return divide_mutable(other);
    }

private:

    FloatComplex CUDA_CRAP divide_mutable(SubType real) {

        return divide_mutable(SubType(real));

    }

public:

    SubType CUDA_CRAP getRe() const {
        return SubType(exp, mantissaReal);
    }

    SubType CUDA_CRAP getIm() const {
        return SubType(exp, mantissaImag);
    }

    SubType CUDA_CRAP getMantissaReal() const {
        return mantissaReal;
    }

    SubType CUDA_CRAP getMantissaImag() const {
        return mantissaImag;
    }

public:
    void CUDA_CRAP toComplex(SubType& re, SubType& img) const {
        //return new Complex(mantissaReal * MantExp.toExp(exp), mantissaImag * MantExp.toExp(exp));
        auto d = SubType::getMultiplier(exp);
        //return new Complex(MantExp.toDouble(mantissaReal, exp), MantExp.toDouble(mantissaImag, exp));
        re = mantissaReal * d;
        img = mantissaImag * d;
    }

    SubType CUDA_CRAP chebychevNorm() const {
        return SubType::maxBothPositiveReduced(HdrAbs(getRe()), HdrAbs(getIm()));
    }
};
