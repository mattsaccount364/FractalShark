#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

#ifndef __CUDACC__

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
constexpr size_t AuthoritativeMinExtraPrecision = 20;

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
using IterType = uint64_t;
using IterTypeFull = uint64_t;
static constexpr IterType IterTypeMax = ((sizeof(IterType) == 4) ? (0xFFFF'FFFF) : (0xFFFF'FFFF'FFFF'FFFF));

// Look at that class and see that it allocates way too much
//const IterType MAXITERS = 256 * 256 * 256 * 32;
//const int MAXITERS = 256 * 32; // 256 * 256 * 256 * 32
static constexpr IterType MAXITERS = ((sizeof(IterType) == 4) ? (INT32_MAX - 1) : (INT64_MAX - 1));