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

#ifndef __CUDACC__ 
    template<bool IntegerOutput>
    CUDA_CRAP std::string ToString() const {
        if constexpr (!IntegerOutput) {
            std::stringstream ss;
            ss << std::setprecision(std::numeric_limits<double>::max_digits10);
            ss << "mantissaReal: " << static_cast<double>(this->mantissaReal)
                << " mantissaImag: " << static_cast<double>(this->mantissaImag)
                << " exp: 0";
            return ss.str();
        }
        else {
            // Interpret the bits as a double and return the string as hex
            auto doubleMantReal = static_cast<double>(mantissaReal);
            auto doubleMantImag = static_cast<double>(mantissaImag);
            uint64_t* mantissaRealBits = reinterpret_cast<uint64_t*>(&doubleMantReal);
            uint64_t *mantissaImagBits = reinterpret_cast<uint64_t*>(&doubleMantImag);
            std::stringstream ss;
            ss << "mantissaReal: 0x" << std::hex << *mantissaRealBits
                << " mantissaImag: 0x" << std::hex << *mantissaImagBits
                << " exp: 0";
            return ss.str();
        }
    }
#endif

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
        plus_mutable(other);
        return *this;
    }

private:
    FloatComplex CUDA_CRAP plus_mutable(FloatComplex value) {
        mantissaReal = mantissaReal + value.mantissaReal;
        mantissaImag = mantissaImag + value.mantissaImag;
        return *this;

    }

    FloatComplex CUDA_CRAP times_mutable(FloatComplex factor) {
        SubType tempMantissaReal = (mantissaReal * factor.mantissaReal) - (mantissaImag * factor.mantissaImag);

        SubType tempMantissaImag = (mantissaReal * factor.mantissaImag) + (mantissaImag * factor.mantissaReal);

        mantissaReal = tempMantissaReal;
        mantissaImag = tempMantissaImag;

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
        times_mutable(other);
        return *this;
    }

private:

    FloatComplex CUDA_CRAP times_mutable(SubType factor) {
        mantissaReal *= factor;
        mantissaImag *= factor;
        return *this;
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
        lhs.sub_mutable(rhs); // reuse compound assignment
        return lhs; // return the result by value (uses move constructor)
    }

    CUDA_CRAP constexpr FloatComplex& operator-=(const FloatComplex& other) {
        sub_mutable(other);
        return *this;
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
        return FloatComplex(mantissaReal * temp, -mantissaImag * temp);
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
        divide_mutable(other);
        return *this;
    }

private:

    FloatComplex CUDA_CRAP divide_mutable(SubType real) {

        return divide_mutable(SubType(real));

    }

public:

    SubType CUDA_CRAP getRe() const {
        return mantissaReal;
    }

    SubType CUDA_CRAP getIm() const {
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
        auto absReal = HdrAbs(getRe());
        auto absImag = HdrAbs(getIm());
        return absReal > absImag ? absReal : absImag;
    }
};
