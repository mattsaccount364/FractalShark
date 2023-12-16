#pragma once

#ifdef __CUDACC__
#include "dblflt.cuh"
#endif
#include <type_traits>

struct MattDblflt;

#ifndef __CUDACC__
using dblflt = MattDblflt;
#endif

// This class implements a wrapper around the CUDA dbldbl type defined
// in dblflt.h.  It implements all the basic arithmetic operators.
// It also implements a few other functions that are useful for
// arithmetic.
#pragma pack(push, 4)
template<typename T = MattDblflt>
class CudaDblflt {
public:
    T d;
    using TemplateSubType = CudaDblflt<T>;

    template <class To, class From, class Res = typename std::enable_if<
        (sizeof(To) == sizeof(From)) &&
        (alignof(To) == alignof(From)) &&
        std::is_trivially_copyable<From>::value&&
        std::is_trivially_copyable<To>::value,
        To>::type>
    CUDA_CRAP const Res& bit_cast(const From& src) noexcept {
        return *reinterpret_cast<const To*>(&src);
    }

    // Constructs a CudaDblflt that zeros out the head and tail
    CUDA_CRAP
    constexpr
    CudaDblflt() : d{ 0.0f, 0.0f } {
    }

    CUDA_CRAP
    constexpr
    CudaDblflt(const CudaDblflt &other) :
        d{ other.d.head, other.d.tail } {
    }

#if !defined(__CUDA_ARCH__)
    // Constructs a CudaDblflt from a double
    CUDA_CRAP
    constexpr
    CudaDblflt &operator=(double other) {
        d = T{ other };
        return *this;
    }

    CUDA_CRAP
    constexpr
    explicit CudaDblflt(double ind)
        : d{ ind } {
    }
#endif

    CUDA_CRAP
    constexpr
    explicit CudaDblflt(float d)
        : d{ d } {
    }

    template<class U>
    CUDA_CRAP
    constexpr
        explicit CudaDblflt(CudaDblflt<U> other)
        : d{ other.d.head, other.d.tail } {
    }

    // Constructs a CudaDblflt from a pair of floats
    CUDA_CRAP
    constexpr
        explicit CudaDblflt(float head, float tail) :
        d{ head, tail } {
    }

    // Returns the head of the double-float
    CUDA_CRAP
    constexpr
    float head() const {
        return d.head;
    }

    // Returns the tail of the double-float
    CUDA_CRAP
    constexpr
    float tail() const {
        return d.tail;
    }

#ifndef __CUDACC__ 
    CUDA_CRAP std::string ToString() const {
        std::stringstream ss;
        //ss << "mantissa: " << (double)Base::mantissa << " exp: " << Base::exp;
        // TODO
        ss << "mantissa: TODO";
        return ss.str();
    }
#endif

#ifndef __CUDACC__
    operator double() const {
        return (double)d.head + (double)d.tail;
    }
#endif

#ifdef __CUDACC__
    //template<typename std::enable_if<std::is_same<T, dbldbl>::value, dbldbl>::type * = 0>
    __device__
    explicit operator double() const {
        return dblflt_to_double(d);
    }

    // Implements operator+ for CudaDblflt
    friend
    __device__
    CudaDblflt operator+(CudaDblflt a, const CudaDblflt& b) {
        a.d = add_dblflt(a.d, b.d);
        return a;
    }

    // Implements operator- for CudaDblflt
    friend
    __device__
    CudaDblflt operator-(CudaDblflt a, const CudaDblflt& b) {
        a.d = sub_dblflt(a.d, b.d);
        return a;
    }

    // Implements unary operator- for CudaDblflt
    friend
    __device__
    CudaDblflt operator-(CudaDblflt other) {
        return CudaDblflt{ -other.d.head, -other.d.tail };
    }

    // Implements operator* for CudaDblflt
    friend
    __device__
    CudaDblflt operator*(CudaDblflt a, const CudaDblflt& b) {
        a.d = mul_dblflt(a.d, b.d);
        return a;
    }

    __device__
    CudaDblflt& operator*=(const CudaDblflt& b) {
        this->d = mul_dblflt(this->d, b.d);
        return *this;
    }

    // Implements square() for CudaDblflt
    __device__
        CudaDblflt square() {
        CudaDblflt tmp(*this);
        tmp.d = sqr_dblflt(d);
        return tmp;
    }

    //  Implements operator/ for CudaDblflt
    friend
    __device__
    CudaDblflt operator/(CudaDblflt a, const CudaDblflt& b) {
        a.d = div_dblflt(a.d, b.d);
        return a;
    }

    // Implements operator< for CudaDblflt
    __device__
    friend bool operator<(const CudaDblflt& a, const CudaDblflt& b) {
        return a.d.head < b.d.head || (a.d.head == b.d.head && a.d.tail < b.d.tail);
    }

    // Implements operator> for CudaDblflt
    __device__
    friend bool operator>(const CudaDblflt& a, const CudaDblflt& b) {
        return !(b < a) && !(b == a);
    }

    // Implements operator<= for CudaDblflt
    __device__
    friend bool operator<=(const CudaDblflt& a, const CudaDblflt& b) {
        return !(b > a);
    }

    // Implements operator>= for CudaDblflt
    __device__
    friend bool operator>=(const CudaDblflt& a, const CudaDblflt& b) {
        return !(a < b);
    }

    // Implements operator== for CudaDblflt
    __device__
    friend bool operator==(const CudaDblflt& a, const CudaDblflt& b) {
        return a.d.head == b.d.head && a.d.tail == b.d.tail;
    }

    // Implements operator!= for CudaDblflt
    __device__
    friend bool operator!=(const CudaDblflt& a, const CudaDblflt& b) {
        return !(a == b);
    }

    __device__
    CudaDblflt abs() const {
        if (d.head < 0.0f) {
            return CudaDblflt{ -d.head, -d.tail };
        }

        return *this;
    }

    static __device__ constexpr int32_t MIN_SMALL_EXPONENT_INT() {
        return -127;
    }

    __device__
    void Reduce(int32_t &out_exp) {
        const auto bits_y = bit_cast<uint32_t>(this->d.head);
        const auto bits_x = bit_cast<uint32_t>(this->d.tail);
        const auto f_exp_y = (int32_t)((bits_y & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT();
        const auto f_exp_x = (int32_t)((bits_x & 0x7F80'0000UL) >> 23UL);
        const auto val_y = (bits_y & 0x807F'FFFF) | 0x3F80'0000;
        const auto newexp = f_exp_x - f_exp_y;
        const auto satexp = newexp <= 0 ? 0 : newexp;
        const auto val_x = (bits_x & 0x807F'FFFF) | (satexp << 23U);
        const auto f_val_y = bit_cast<const float>(val_y);
        const auto f_val_x = bit_cast<const float>(val_x);
        d.head = f_val_y;
        d.tail = f_val_x;
        out_exp = f_exp_y - MIN_SMALL_EXPONENT_INT();
    }
#endif
};

#pragma pack(pop)
