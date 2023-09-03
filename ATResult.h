#pragma once

#include "HDRFloatComplex.h"
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.GenericComplex;
//import fractalzoomer.core.MantExpComplex;

class ATResult {
public:
    using Complex = HDRFloatComplex<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    CUDA_CRAP_BOTH ATResult() {
        dzdc = HDRFloatComplex();
        dzdc2 = HDRFloatComplex();
    }

    Complex dz;

    HDRFloatComplex dzdc;
    HDRFloatComplex dzdc2;
    size_t bla_iterations;
    size_t bla_steps;
};
