#pragma once

enum class HPDestructor : bool {
    False = false,
    True = true,
};

template <HPDestructor Destructor> class HighPrecisionT;

enum class HDROrder;

template <class T, HDROrder Order, class TExp> class HDRFloat;

#ifndef __CUDACC__

#include "MpirSerialization.h"

#include <assert.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

std::string MpfToHex32String(const mpf_t mpf_val);

std::string MpfToHex64StringInvertable(const mpf_t mpf_val);

void Hex64StringToMpf_Exact(const std::string &s, mpf_t out);

void MpfNormalize(mpf_t val);

template <HPDestructor Destructor> class HighPrecisionT {
public:
    // Friend the other version of HighPrecisionT so we can copy between them.
    template <HPDestructor T> friend class HighPrecisionT;

    void
    InitMpf()
    {
        mpf_init(m_Data);
    }

    void
    InitMpf2(uint64_t precisionInBits)
    {
        if (precisionInBits > 1'000'000) {
            throw std::runtime_error("Requested precision is too high.  This is probably a bug..");
        }

        mpf_init2(m_Data, precisionInBits);
    }

    HighPrecisionT() { InitMpf(); }

    HighPrecisionT(const HighPrecisionT &other)
    {
        InitMpf2(other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
    }

    template <HPDestructor T> HighPrecisionT(const HighPrecisionT<T> &other)
    {
        InitMpf2(other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
    }

    explicit HighPrecisionT(const mpf_t data)
    {
        InitMpf2(mpf_get_prec(data));
        mpf_set(m_Data, data);
    }

    HighPrecisionT(HighPrecisionT &&other) noexcept
    {
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};
    }

    template <HPDestructor T> HighPrecisionT(HighPrecisionT<T> &&other)
    {
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};
    }

    HighPrecisionT(uint64_t precisionInBits, std::istream &file)
    {
        InitMpf2(precisionInBits);
        MpirSerialization::mpf_inp_raw_stream(file, m_Data);
    }

    ~HighPrecisionT()
    {
        // When using the bump allocator, we don't want to free the memory.
        // That's part of the point of the bump allocator.
        if constexpr (Destructor == HPDestructor::True) {
            mpf_clear(m_Data);
            m_Data[0] = {};
        }
    }

    void
    SaveToStream(std::ostream &file) const
    {
        MpirSerialization::mpf_out_raw_stream(file, m_Data);
    }

    HighPrecisionT &
    operator=(const HighPrecisionT &other)
    {
        if (this == &other) {
            return *this;
        }

        mpf_set_prec(m_Data, other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
        return *this;
    }

    template <HPDestructor T>
    HighPrecisionT<T> &
    operator=(const HighPrecisionT<T> &other)
    {
        if (this == &other) {
            return *this;
        }

        mpf_set_prec(m_Data, other.precisionInBits());
        mpf_set(m_Data, other.m_Data);
        return *this;
    }

    HighPrecisionT &
    operator=(HighPrecisionT &&other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        mpf_clear(m_Data);
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};

        return *this;
    }

    template <HPDestructor T>
    HighPrecisionT<T> &
    operator=(HighPrecisionT<T> &&other)
    {
        if (this == &other) {
            return *this;
        }

        mpf_clear(m_Data);
        m_Data[0] = other.m_Data[0];
        other.m_Data[0] = {};

        return *this;
    }

    enum class SetPrecision { True };

    explicit HighPrecisionT(SetPrecision /*setPrecision*/, uint64_t precisionInBits)
    {
        InitMpf2(precisionInBits);
    }

    explicit HighPrecisionT(double data)
    {
        InitMpf();
        mpf_set_d(m_Data, data);
    }

    explicit HighPrecisionT(uint64_t data)
    {
        InitMpf();
        mpf_set_ui(m_Data, data);
    }

    explicit HighPrecisionT(uint32_t data)
    {
        InitMpf();
        mpf_set_ui(m_Data, data);
    }

    explicit HighPrecisionT(int64_t data)
    {
        InitMpf();
        mpf_set_si(m_Data, data);
    }

    explicit HighPrecisionT(int data)
    {
        InitMpf();
        mpf_set_si(m_Data, data);
    }

    explicit HighPrecisionT(long data)
    {
        InitMpf();
        mpf_set_si(m_Data, data);
    }

    explicit HighPrecisionT(std::string data)
    {
        InitMpf();

        //
        // Normalize the result of mpf_set_str -- maybe a bug in MPIR?
        //

        mpf_set_str(m_Data, data.c_str(), 10);
        MpfNormalize(m_Data);
    }

    template <typename Type, HDROrder Order, typename TExp>
    explicit HighPrecisionT(const HDRFloat<Type, Order, TExp> &other)
    {
        // Not much point in using a higher precision type given
        // that we're converting from a lower precision type.
        InitMpf2(sizeof(double) * 8);
        other.GetHighPrecision(*this);
    }

    void
    precisionInBits(uint64_t prec)
    {
        mpf_set_prec(m_Data, prec);
    }

    uint64_t
    precisionInBits() const
    {
        return (uint32_t)mpf_get_prec(m_Data);
    }

    template <typename T>
    HighPrecisionT &
    operator=(const T &data)
    requires(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        mpf_set_d(m_Data, data);
        return *this;
    }

    HighPrecisionT &
    operator+=(const HighPrecisionT &data)
    {
        mpf_add(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecisionT &
    operator-=(const HighPrecisionT &data)
    {
        mpf_sub(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecisionT &
    operator*=(const HighPrecisionT &data)
    {
        mpf_mul(m_Data, m_Data, data.m_Data);
        return *this;
    }

    HighPrecisionT &
    operator/=(const HighPrecisionT &data)
    {
        mpf_div(m_Data, m_Data, data.m_Data);
        return *this;
    }

    friend HighPrecisionT
    operator+(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        HighPrecisionT result{SetPrecision::True, lhs.precisionInBits()};
        mpf_add(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend HighPrecisionT
    operator-(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        HighPrecisionT result{SetPrecision::True, lhs.precisionInBits()};
        mpf_sub(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    HighPrecisionT
    operator-() const
    {
        HighPrecisionT result{SetPrecision::True, precisionInBits()};
        mpf_neg(result.m_Data, m_Data);
        return result;
    }

    friend HighPrecisionT
    operator*(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        HighPrecisionT result{SetPrecision::True, lhs.precisionInBits()};
        mpf_mul(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend HighPrecisionT
    operator/(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        HighPrecisionT result{SetPrecision::True, lhs.precisionInBits()};
        mpf_div(result.m_Data, lhs.m_Data, rhs.m_Data);
        return result;
    }

    friend bool
    operator==(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) == 0;
    }

    friend bool
    operator!=(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) != 0;
    }

    friend bool
    operator<(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) < 0;
    }

    friend bool
    operator>(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) > 0;
    }

    friend bool
    operator<=(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) <= 0;
    }

    friend bool
    operator>=(const HighPrecisionT &lhs, const HighPrecisionT &rhs)
    {
        return mpf_cmp(lhs.m_Data, rhs.m_Data) >= 0;
    }

    std::string
    str() const
    {
        constexpr size_t number_of_chars = 128 * 1024;
        std::vector<char> temp(number_of_chars);
        gmp_snprintf(temp.data(), number_of_chars, "%.Fe", m_Data);
        std::string result(temp.data());
        return result;
    }

    // Provide operator<< that returns a string representation of m_Data:
    friend std::ostream &
    operator<<(std::ostream &os, const HighPrecisionT &data)
    {
        os << data.str();
        return os;
    }

    friend std::istream &
    operator>>(std::istream &is, HighPrecisionT &data)
    {
        std::string str;
        is >> str;
        data = HighPrecisionT{str};
        return is;
    }

    static void
    defaultPrecisionInBits(uint64_t prec)
    {
        if (prec > 1'000'000) {
            throw std::runtime_error("Requested precision is too high.  This is probably a bug..");
        }

        mpf_set_default_prec(prec);
    }

    static uint64_t
    defaultPrecisionInBits()
    {
        return mpf_get_default_prec();
    }

    // Return mantissa and exponent of m_Data in two out parameters.
    // Use double for mantissa:
    void
    frexp(double &mantissa, long &exponent) const
    {
        exponent = static_cast<int32_t>(mpf_get_2exp_d(&mantissa, m_Data));
    }

    friend HighPrecisionT
    abs(const HighPrecisionT &data)
    {
        HighPrecisionT result{SetPrecision::True, data.precisionInBits()};
        mpf_abs(result.m_Data, data.m_Data);
        return result;
    }

    HighPrecisionT
    power(int32_t powToRaiseTo) const
    {
        HighPrecisionT result{SetPrecision::True, precisionInBits()};
        mpf_pow_ui(result.m_Data, m_Data, powToRaiseTo);
        return result;
    }

    const __mpf_struct *
    backend() const
    {
        return &m_Data[0];
    }

    const mpf_t *
    backendRaw() const
    {
        return &m_Data;
    }

    explicit
    operator double() const
    {
        return mpf_get_d(m_Data);
    }

    explicit
    operator float() const
    {
        return static_cast<float>(mpf_get_d(m_Data));
    }

private:
    mpf_t m_Data;
};

using HighPrecision = HighPrecisionT<HPDestructor::True>;

template <class From, class To>
To
Convert(From data)
{
    if constexpr (std::is_same<To, double>::value || std::is_same<To, float>::value) {
        return static_cast<To>(mpf_get_d(*data.backendRaw()));
    } else if constexpr (std::is_same<To, int>::value || std::is_same<To, long>::value ||
                         std::is_same<To, long long>::value) {
        return static_cast<To>(mpf_get_si(*data.backendRaw()));
    } else {
        assert(false);
        return 0;
    }
}

#else // __CUDACC__

//
// CUDA version of HighPrecision.  Using void
// means that we can't use the class in any way.
//

using HighPrecision = void;

#endif

template <typename ToCheck,
          typename ToCheck2,
          std::size_t LHS = sizeof(ToCheck),
          std::size_t RHS = sizeof(ToCheck2)>
void
check_size()
{
    static_assert(LHS == RHS, "Size is off!");
}

template <typename ToCheck, std::size_t RHS, std::size_t LHS = sizeof(ToCheck)>
void
check_size()
{
    static_assert(LHS == RHS, "Size is off!");
}

// Was 83: Roughly 25 digits = 83 bits = 25*3.321.  MPIR will round up anyway.
// Set to 120 instead for better compression results.  Search for DefaultCompressionExp.
constexpr size_t AuthoritativeMinExtraPrecisionInBits = 120;

// Amount of precision used for medium-precision reference orbit in bits.
// Nothing special about 800.  Just ensure it's bigger than the one above.
constexpr size_t AuthoritativeReuseExtraPrecisionInBits = 800;

// TODO move to templates
// using IterType = uint32_t;
using IterTypeFull = uint64_t;

enum class PerturbExtras { Disable, Bad, SimpleCompression, MaxCompression };

enum class CompressToDisk { Disable, SimpleCompression, MaxCompression, MaxCompressionImagina };

enum class ImaginaSettings { ConvertToCurrent, UseSaved, Max };

// If true, choose type == float/double for primitives.
// If false, choose type == T::TemplateSubType for HdrFloat subtypes.
// This is kind of a headache.  std::conditional by itself is not adequate here.
template <bool, typename T> class SubTypeChooser {
public:
    using type = typename T::TemplateSubType;
};

template <typename T> class SubTypeChooser<true, T> {
public:
    using type = T;
};

#ifdef __CUDA_ARCH__
#define CUDA_CRAP __device__
#define CUDA_CRAP_BOTH __host__ __device__
#else
#define CUDA_CRAP
#define CUDA_CRAP_BOTH
#endif
