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

    ATResult() {
        dzdc = HDRFloatComplex();
        dzdc2 = HDRFloatComplex();
    }

    Complex dz;

    HDRFloatComplex dzdc;
    HDRFloatComplex dzdc2;
    int bla_iterations;
    int bla_steps;
};
