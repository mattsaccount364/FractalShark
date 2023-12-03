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
        d{ other.d.x, other.d.y } {
    }

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

    template<class T>
    CUDA_CRAP
    constexpr
        explicit CudaDblflt(T d)
        : d{ 0.0f, (float)d } {
    }

    template<class U>
    CUDA_CRAP
    constexpr
        explicit CudaDblflt(CudaDblflt<U> other)
        : d{ other.d.x, other.d.y } {
    }

    // Constructs a CudaDblflt from a pair of floats
    CUDA_CRAP
    constexpr
        explicit CudaDblflt(float head, float tail) :
        d{ tail, head } {
    }

    // Returns the head of the double-float
    CUDA_CRAP
    constexpr
    float head() const {
        return d.y;
    }

    // Returns the tail of the double-float
    CUDA_CRAP
    constexpr
    float tail() const {
        return d.x;
    }

#ifndef __CUDACC__
    operator double() const {
        return (double)d.y + (double)d.x;
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
        return CudaDblflt{ -other.d.y, -other.d.x };
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
        return a.d.y < b.d.y || (a.d.y == b.d.y && a.d.x < b.d.x);
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
        return a.d.y == b.d.y && a.d.x == b.d.x;
    }

    // Implements operator!= for CudaDblflt
    __device__
    friend bool operator!=(const CudaDblflt& a, const CudaDblflt& b) {
        return !(a == b);
    }

    __device__
    CudaDblflt abs() const {
        if (d.y < 0.0f) {
            return CudaDblflt{ -d.y, -d.x };
        }

        return *this;
    }

    static __device__ constexpr int32_t MIN_SMALL_EXPONENT_INT() {
        return -127;
    }

    __device__
    void Reduce(int32_t &out_exp) {
        const auto bits_y = bit_cast<uint32_t>(this->d.y);
        const auto bits_x = bit_cast<uint32_t>(this->d.x);
        const auto f_exp_y = (int32_t)((bits_y & 0x7F80'0000UL) >> 23UL) + MIN_SMALL_EXPONENT_INT();
        const auto f_exp_x = (int32_t)((bits_x & 0x7F80'0000UL) >> 23UL);
        const auto val_y = (bits_y & 0x807F'FFFF) | 0x3F80'0000;
        const auto newexp = f_exp_x - f_exp_y;
        const auto satexp = newexp <= 0 ? 0 : newexp;
        const auto val_x = (bits_x & 0x807F'FFFF) | (satexp << 23U);
        const auto f_val_y = bit_cast<const float>(val_y);
        const auto f_val_x = bit_cast<const float>(val_x);
        d.y = f_val_y;
        d.x = f_val_x;
        out_exp = f_exp_y - MIN_SMALL_EXPONENT_INT();
    }
#endif
};

#pragma pack(pop)

#if 0

/* Compute error-free sum of two unordered doubles. See Knuth, TAOCP vol. 2 */
__device__ __forceinline__ dblflt add_float_to_dblflt(float a, float b)
{
    float t1, t2;
    dblflt z;
    z.y = __fadd_rn(a, b);
    t1 = __fadd_rn(z.y, -a);
    t2 = __fadd_rn(z.y, -t1);
    t1 = __fadd_rn(b, -t1);
    t2 = __fadd_rn(a, -t2);
    z.x = __fadd_rn(t1, t2);
    return z;
}

/* Compute error-free product of two doubles. Take full advantage of FMA */
__device__ __forceinline__ dblflt mul_double_to_dblflt(float a, float b)
{
    dblflt z;
    z.y = __fmul_rn(a, b);
    z.x = __fmaf_rn(a, b, -z.y);
    return z;
}

/* Negate a double-float number, by separately negating head and tail */
__device__ __forceinline__ dblflt neg_dblflt(dblflt a)
{
    dblflt z;
    z.y = -a.y;
    z.x = -a.x;
    return z;
}

/* Compute high-accuracy sum of two double-float operands. In the absence of
   underflow and overflow, the maximum relative error observed with 10 billion
   test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dblflt add_dblflt(dblflt a, dblflt b)
{
    dblflt z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn(a.y, b.y);
    t2 = __fadd_rn(t1, -a.y);
    t3 = __fadd_rn(__fadd_rn(a.y, t2 - t1), __fadd_rn(b.y, -t2));
    t4 = __fadd_rn(a.x, b.x);
    t2 = __fadd_rn(t4, -a.x);
    t5 = __fadd_rn(__fadd_rn(a.x, t2 - t4), __fadd_rn(b.x, -t2));
    t3 = __fadd_rn(t3, t4);
    t4 = __fadd_rn(t1, t3);
    t3 = __fadd_rn(t1 - t4, t3);
    t3 = __fadd_rn(t3, t5);
    z.y = e = __fadd_rn(t4, t3);
    z.x = __fadd_rn(t4 - e, t3);
    return z;
}

/* Compute high-accuracy difference of two double-float operands. In the
   absence of underflow and overflow, the maximum relative error observed
   with 10 billion test cases was 3.0716194922303448e-32 (~= 2**-104.6826).
   This implementation is based on: Andrew Thall, Extended-Precision
   Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
   from http://andrewthall.org/papers/df64_qf128.pdf.
*/
__device__ __forceinline__ dblflt sub_dblflt(dblflt a, dblflt b)
{
    dblflt z;
    float t1, t2, t3, t4, t5, e;
    t1 = __fadd_rn(a.y, -b.y);
    t2 = __fadd_rn(t1, -a.y);
    t3 = __fadd_rn(__fadd_rn(a.y, t2 - t1), -__fadd_rn(b.y, t2));
    t4 = __fadd_rn(a.x, -b.x);
    t2 = __fadd_rn(t4, -a.x);
    t5 = __fadd_rn(__fadd_rn(a.x, t2 - t4), -__fadd_rn(b.x, t2));
    t3 = __fadd_rn(t3, t4);
    t4 = __fadd_rn(t1, t3);
    t3 = __fadd_rn(t1 - t4, t3);
    t3 = __fadd_rn(t3, t5);
    z.y = e = __fadd_rn(t4, t3);
    z.x = __fadd_rn(t4 - e, t3);
    return z;
}

/* Compute high-accuracy product of two double-float operands, taking full
   advantage of FMA. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was 5.238480533564479e-32
   (~= 2**-103.9125).
*/
__device__ __forceinline__ dblflt mul_dblflt(dblflt a, dblflt b)
{
    dblflt t, z;
    float e;
    t.y = __fmul_rn(a.y, b.y);
    t.x = __fmaf_rn(a.y, b.y, -t.y);
    t.x = __fmaf_rn(a.x, b.x, t.x);
    t.x = __fmaf_rn(a.y, b.x, t.x);
    t.x = __fmaf_rn(a.x, b.y, t.x);
    z.y = e = __fadd_rn(t.y, t.x);
    z.x = __fadd_rn(t.y - e, t.x);
    return z;
}

__device__ __forceinline__ dblflt mul_dblflt2x(dblflt a, dblflt b)
{
    dblflt t, z;
    float e;
    t.y = __fmul_rn(a.y, b.y);
    t.x = __fmaf_rn(a.y, b.y, -t.y);
    t.x = __fmaf_rn(a.x, b.x, t.x);
    t.x = __fmaf_rn(a.y, b.x, t.x);
    t.x = __fmaf_rn(a.x, b.y, t.x);
    z.y = e = __fadd_rn(t.y, t.x);
    z.x = __fadd_rn(t.y - e, t.x);
    z.x = __fmul_rn(z.x, 2.0f);
    z.y = __fmul_rn(z.y, 2.0f);
    return z;
}

__device__ __forceinline__ dblflt sqr_dblflt(dblflt a)
{
    dblflt t, z;
    float e;
    //t.y = __fmul_rn(a.y, a.y);
    //t.x = __fmaf_rn(a.y, a.y, -t.y);
    //t.x = __fmaf_rn(a.x, a.x, t.x);
    //t.x = __fmaf_rn(a.y, a.x, t.x);
    //t.x = __fmaf_rn(a.x, a.y, t.x);
    //z.y = e = __fadd_rn(t.y, t.x);
    //z.x = __fadd_rn(t.y - e, t.x);

    t.y = __fmul_rn(a.y, a.y);
    t.x = __fmaf_rn(a.y, a.y, -t.y);
    t.x = __fmaf_rn(a.x, a.x, t.x);
    e = __fmul_rn(a.y, a.x);
    t.x = __fmaf_rn(2.0f, e, t.x);
    z.y = e = __fadd_rn(t.y, t.x);
    z.x = __fadd_rn(t.y - e, t.x);

    return z;
}

__device__ __forceinline__ dblflt shiftleft_dblflt(dblflt a)
{
    dblflt z;
    z.x = __fmul_rn(a.x, 2.0f);
    z.y = __fmul_rn(a.y, 2.0f);
    return z;
}

/* Compute high-accuracy quotient of two double-float operands, using Newton-
   Raphson iteration. Based on: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
   Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
   In Proceedings of the 8th International Conference on Computational Science,
   ICCS '08, Part I, pp. 446-455. In the absence of underflow and overflow, the
   maximum relative error observed with 10 billion test cases was
   1.0161322480099059e-31 (~= 2**-102.9566).
*/
__device__ __forceinline__ dblflt div_dblflt(dblflt a, dblflt b)
{
    dblflt t, z;
    float e, r;
    r = 1.0 / b.y;
    t.y = __fmul_rn(a.y, r);
    e = __fmaf_rn(b.y, -t.y, a.y);
    t.y = __fmaf_rn(r, e, t.y);
    t.x = __fmaf_rn(b.y, -t.y, a.y);
    t.x = __fadd_rn(a.x, t.x);
    t.x = __fmaf_rn(b.x, -t.y, t.x);
    e = __fmul_rn(r, t.x);
    t.x = __fmaf_rn(b.y, -e, t.x);
    t.x = __fmaf_rn(r, t.x, e);
    z.y = e = __fadd_rn(t.y, t.x);
    z.x = __fadd_rn(t.y - e, t.x);
    return z;
}

/* Compute high-accuracy square root of a double-float number. Newton-Raphson
   iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
   High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
   1997, pp. 561-589. In the absence of underflow and overflow, the maximum
   relative error observed with 10 billion test cases was
   3.7564109505601846e-32 (~= 2**-104.3923).
*/
__device__ __forceinline__ dblflt sqrt_dblflt(dblflt a)
{
    dblflt t, z;
    double e, y, s, r;
    r = rsqrt(a.y);
    if (a.y == 0.0f) r = 0.0;
    y = __fmul_rn(a.y, r);
    s = __fmaf_rn(y, -y, a.y);
    r = __fmul_rn(0.5, r);
    z.y = e = __fadd_rn(s, a.x);
    z.x = __fadd_rn(s - e, a.x);
    t.y = __fmul_rn(r, z.y);
    t.x = __fmaf_rn(r, z.y, -t.y);
    t.x = __fmaf_rn(r, z.x, t.x);
    r = __fadd_rn(y, t.y);
    s = __fadd_rn(y - r, t.y);
    s = __fadd_rn(s, t.x);
    z.y = e = __fadd_rn(r, s);
    z.x = __fadd_rn(r - e, s);
    return z;
}

/* Compute high-accuracy reciprocal square root of a double-double number.
   Based on Newton-Raphson iteration. In the absence of underflow and overflow,
   the maximum relative error observed with 10 billion test cases was
   6.4937771666026349e-32 (~= 2**-103.6026)
*/
__device__ __forceinline__ dblflt rsqrt_dblflt(dblflt a)
{
    dblflt z;
    double r, s, e;
    r = rsqrt(a.y);
    e = __fmul_rn(a.y, r);
    s = __fmaf_rn(e, -r, 1.0);
    e = __fmaf_rn(a.y, r, -e);
    s = __fmaf_rn(e, -r, s);
    e = __fmul_rn(a.x, r);
    s = __fmaf_rn(e, -r, s);
    e = 0.5 * r;
    z.y = __fmul_rn(e, s);
    z.x = __fmaf_rn(e, s, -z.y);
    s = __fadd_rn(r, z.y);
    r = __fadd_rn(r, -s);
    r = __fadd_rn(r, z.y);
    r = __fadd_rn(r, z.x);
    z.y = e = __fadd_rn(s, r);
    z.x = __fadd_rn(s - e, r);
    return z;
}

__device__ __forceinline__ double dblflt_to_double(dblflt a)
{
    return (double)a.y + (double)a.x;
}

__device__ __forceinline__ dblflt double_to_dblflt(double a)
{
    // ?? 2^23 = 8388608.0
    dblflt res;
    res.y = (float)a;
    res.x = (float)(a - (double)res.y);
    return res;
}

#endif