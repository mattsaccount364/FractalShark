#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/number.hpp>

#ifndef __CUDACC__ 
constexpr const size_t DigitPrecision = 2000;

using HighPrecision = boost::multiprecision::number<
    boost::multiprecision::gmp_float<DigitPrecision>,
    boost::multiprecision::et_on>;
template<class From, class To>
To Convert(From data) {
    return data.convert_to<To>();
}
#endif

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
