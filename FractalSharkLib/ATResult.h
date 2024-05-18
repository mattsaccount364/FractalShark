#pragma once

#include "HDRFloatComplex.h"
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.MantExpComplex;

template<typename IterType, class T, class SubType>
class ATResult {
public:
    using FloatComplexT =
        std::conditional<
        std::is_same<T, ::HDRFloat<float>>::value ||
        std::is_same<T, ::HDRFloat<double>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<dblflt>>>::value,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

    CUDA_CRAP ATResult() :
        dz{},
        dzdc{},
        dzdc2{},
        bla_iterations{},
        bla_steps{} {}

    FloatComplexT dz;
    FloatComplexT dzdc;
    FloatComplexT dzdc2;
    IterType bla_iterations;
    IterType bla_steps;
};
