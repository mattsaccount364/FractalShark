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
    Enable
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