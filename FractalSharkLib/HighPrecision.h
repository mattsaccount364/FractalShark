#pragma once


#ifndef __CUDACC__

// Include MPIR header
#include <mpir.h>
#include <string>
#include <assert.h>
#include <iostream>

class HighPrecision {
public:
    HighPrecision() {
        mpf_init(m_Data);
    }

    HighPrecision(const HighPrecision& other) {
        mpf_init2(m_Data, other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
    }

    HighPrecision(const mpf_t& data) {
        mpf_init2(m_Data, mpf_get_prec(data));
        mpf_set(m_Data, data);
    }

    HighPrecision(HighPrecision&& other) {
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};
    }

    ~HighPrecision() {
        mpf_clear(m_Data);
    }

    HighPrecision& operator=(const HighPrecision& other) {
        mpf_set_prec(m_Data, other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
        return *this;
    }

    HighPrecision& operator=(HighPrecision&& other) {
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};
        return *this;
    }

    HighPrecision(double data) {
        mpf_init(m_Data);
        mpf_set_d(m_Data, data);
    }

    HighPrecision(uint64_t data) {
        mpf_init(m_Data);
        mpf_set_ui(m_Data, data);
    }

    HighPrecision(uint32_t data) {
        mpf_init(m_Data);
        mpf_set_ui(m_Data, data);
    }

    HighPrecision(int64_t data) {
        mpf_init(m_Data);
        mpf_set_si(m_Data, data);
    }

    HighPrecision(int data) {
        mpf_init(m_Data);
        mpf_set_si(m_Data, data);
    }

    HighPrecision(long data) {
        mpf_init(m_Data);
        mpf_set_si(m_Data, data);
    }

    HighPrecision(std::string data) {
        mpf_init(m_Data);
        mpf_set_str(m_Data, data.c_str(), 10);
    }

    void precisionInBits(uint64_t prec) {
        mpf_set_prec(m_Data, prec);
    }

    uint64_t precisionInBits() const {
        return (uint32_t)mpf_get_prec(m_Data);
    }

    template<typename T>
    HighPrecision& operator=(const T &data) {
        mpf_set_d(m_Data, data);
        return *this;
    }

    HighPrecision& operator=(const char* data) {
        mpf_set_str(m_Data, data, 10);
        return *this;
    }

    HighPrecision& operator=(const std::string& data) {
        mpf_set_str(m_Data, data.c_str(), 10);
        return *this;
    }

    HighPrecision& operator+=(const HighPrecision& data) {
        mpf_add(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecision& operator-=(const HighPrecision& data) {
        mpf_sub(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecision& operator*=(const HighPrecision& data) {
        mpf_mul(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecision& operator/=(const HighPrecision& data) {
        mpf_div(m_Data, m_Data, data.m_Data);
        return *this;
    }

    friend HighPrecision operator+(const HighPrecision& lhs, const HighPrecision& rhs) {
        HighPrecision result;
        mpf_add(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend HighPrecision operator-(const HighPrecision& lhs, const HighPrecision& rhs) {
        HighPrecision result;
        mpf_sub(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    HighPrecision operator-() const {
        HighPrecision result;
        mpf_neg(result.m_Data, m_Data);
        return result;
    }

    friend HighPrecision operator*(const HighPrecision& lhs, const HighPrecision& rhs) {
        HighPrecision result;
        mpf_mul(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend HighPrecision operator/(const HighPrecision& lhs, const HighPrecision& rhs) {
        HighPrecision result;
        mpf_div(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend bool operator==(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) == 0;
    }

    friend bool operator!=(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) != 0;
    }

    friend bool operator<(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) < 0;
    }

    
    friend bool operator>(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) > 0;
    }

    friend bool operator<=(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) <= 0;
    }

    friend bool operator>=(const HighPrecision& lhs, const HighPrecision& rhs) {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) >= 0;
    }

    std::string str() const {
        mp_exp_t exponent;
        char* str = mpf_get_str(NULL, &exponent, 10, 0, m_Data);
        std::string result(str);
        free(str);
        return result;
    }

    // Provide operator<< that returns a string representation of m_Data:
    friend std::ostream& operator<<(std::ostream& os, const HighPrecision& data) {
        os << data.str();
        return os;
    }

    friend std::istream& operator>>(std::istream& is, HighPrecision& data) {
        std::string str;
        is >> str;
        data = str;
        return is;
    }

    static void defaultPrecisionInBits(uint64_t prec) {
        mpf_set_default_prec(prec);
    }

    static uint64_t defaultPrecisionInBits() {
        return mpf_get_default_prec();
    }

    // Return mantissa and exponent of m_Data in two out parameters.
    // Use double for mantissa:
    void frexp(double& mantissa, long& exponent) const {
        exponent = static_cast<int32_t>(mpf_get_2exp_d(&mantissa, m_Data));
    }

    friend HighPrecision abs(const HighPrecision& data) {
        HighPrecision result;
        mpf_abs(result.m_Data, data.m_Data);
        return result;
    }

    const __mpf_struct *backend() const {
        return &m_Data[0];
    }

    const mpf_t *backendRaw() const {
        return &m_Data;
    }

    explicit operator double() const {
        return mpf_get_d(m_Data);
    }

    explicit operator float() const {
        return static_cast<float>(mpf_get_d(m_Data));
    }

private:
    mpf_t m_Data;
};


template<class From, class To>
To Convert(From data) {
    if constexpr (std::is_same<To, double>::value || std::is_same<To, float>::value) {
        return static_cast<To>(mpf_get_d(*data.backendRaw()));
    }
    else if constexpr(std::is_same<To, int>::value || std::is_same<To, long>::value || std::is_same<To, long long>::value) {
        return static_cast<To>(mpf_get_si(*data.backendRaw()));
    }
    else {
        assert(false);
        return 0;
    }
}
#endif

template <
    typename ToCheck,
    typename ToCheck2,
    std::size_t LHS = sizeof(ToCheck),
    std::size_t RHS = sizeof(ToCheck2)>
void check_size() {
    static_assert(LHS == RHS, "Size is off!");
}

template <
    typename ToCheck,
    std::size_t RHS,
    std::size_t LHS = sizeof(ToCheck)>
void check_size() {
    static_assert(LHS == RHS, "Size is off!");
}


// Amount of extra precision before forcing a full-precision recalculation
constexpr size_t AuthoritativeMinExtraPrecisionInBits = 25 * 8;

// Amount of precision used for medium-precision reference orbit.
constexpr size_t AuthoritativeReuseExtraPrecisionInBits = 100 * 8;

// TODO move to templates
//using IterType = uint32_t;
using IterTypeFull = uint64_t;

enum class PerturbExtras {
    Disable,
    Bad,
    EnableCompression
};

// If true, choose type == float/double for primitives.
// If false, choose type == T::TemplateSubType for HdrFloat subtypes.
// This is kind of a headache.  std::conditional by itself is not adequate here.
template<bool, typename T>
class SubTypeChooser {
public:
    using type = typename T::TemplateSubType;
};

template<typename T>
class SubTypeChooser<true, T> {
public:
    using type = T;
};

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

