#pragma once

#include "HDRFloatComplex.h"
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.MantExpComplex;

template<class SubType>
class ATResult {
public:
    using HDRFloatComplex = HDRFloatComplex<SubType>;

    CUDA_CRAP_BOTH ATResult() {
    }

    HDRFloatComplex dz;
    HDRFloatComplex dzdc;
    HDRFloatComplex dzdc2;
    size_t bla_iterations;
    size_t bla_steps;
};
