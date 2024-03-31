#pragma once

#include <stdint.h>
#include "GPU_Types.h"
#include "RefOrbitCalc.h"

class Fractal;

class FractalTest {
public:
    FractalTest(Fractal &fractal);
    
    void BasicTest();
    void Benchmark(RefOrbitCalc::PerturbationResultType type);

private:
    void BasicTestInternal(size_t& test_index);
    void BasicOneTest(
        size_t view_index,
        size_t test_index,
        const wchar_t* dir_name,
        const wchar_t* test_prefix,
        RenderAlgorithm alg_to_test);

    Fractal &m_Fractal;
};
