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

constexpr size_t ExtraPrecision = 20;
constexpr size_t MpirEstPrecision = 15;

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
