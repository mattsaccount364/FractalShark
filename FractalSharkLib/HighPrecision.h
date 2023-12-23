#pragma once

#ifndef __CUDACC__

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

// THIS IS SO STUPID BUT I DON'T KNOW HOW TO FIX IT
// CAN WE REMOVE THIS IFDEF AND USE CONSTEXPR
// #define CONSTANT_PRECISION

// 1) Toggle the above pre-processor definition if necessary.
// 2) Set to non-zero to force specific precision
// 3) Set to zero for dynamic.
constexpr const size_t DigitPrecision = 0;

using HighPrecision = boost::multiprecision::number<
    boost::multiprecision::gmp_float<DigitPrecision>,
    boost::multiprecision::et_on>;
template<class From, class To>
To Convert(From data) {
    return data.convert_to<To>();
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
constexpr size_t AuthoritativeMinExtraPrecision = 50;

// Amount of precision used for medium-precision reference orbit.
constexpr size_t AuthoritativeReuseExtraPrecision = 100;

//using HighPrecision = boost::multiprecision::cpp_dec_float_100;
//template<class From, class To>
//To Convert(From data) {
//    return data.convert_to<To>();
//}

//using HighPrecision = double;
//template<class From, class To>
//To Convert(From data) {
//    return static_cast<To>(data);
//}

// TODO move to templates
//using IterType = uint32_t;
using IterTypeFull = uint64_t;

enum class CalcBad {
    Disable,
    Enable,
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

#ifndef __CUDACC__
struct scoped_mpfr_precision
{
    unsigned saved_digits10;
    scoped_mpfr_precision(unsigned digits10) : saved_digits10(HighPrecision::thread_default_precision())
    {
        HighPrecision::default_precision(digits10);
    }
    ~scoped_mpfr_precision()
    {
        HighPrecision::default_precision(saved_digits10);
    }
    void reset(unsigned digits10)
    {
        HighPrecision::default_precision(digits10);
    }
    void reset()
    {
        HighPrecision::default_precision(saved_digits10);
    }
};

struct scoped_mpfr_precision_options
{
    boost::multiprecision::variable_precision_options saved_options;
    scoped_mpfr_precision_options(boost::multiprecision::variable_precision_options opts) : saved_options(HighPrecision::thread_default_variable_precision_options())
    {
        HighPrecision::thread_default_variable_precision_options(opts);
    }
    ~scoped_mpfr_precision_options()
    {
        HighPrecision::thread_default_variable_precision_options(saved_options);
    }
    void reset(boost::multiprecision::variable_precision_options opts)
    {
        HighPrecision::thread_default_variable_precision_options(opts);
    }
};
#endif


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