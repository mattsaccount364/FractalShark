#include "NullKernel.cuh"
#include "Conversion.cuh"
#include "HpSharkFloat.cuh"
#include "Tests.h"

#include <cuda_runtime.h> 

#include <iostream>
#include <mpir.h>
#include <conio.h>

// Function to perform the calculation on the host using MPIR
void computeNextXY_host(mpf_t x, mpf_t y, mpf_t a, mpf_t b, int num_iter) {
    mpf_t x_squared, y_squared, two_xy, temp_x, temp_y;
    mpf_init(x_squared);
    mpf_init(y_squared);
    mpf_init(two_xy);
    mpf_init(temp_x);
    mpf_init(temp_y);

    for (int iter = 0; iter < num_iter; ++iter) {
        mpf_mul(x_squared, x, x); // x^2
        mpf_mul(y_squared, y, y); // y^2
        mpf_mul(temp_y, x, y);    // xy
        mpf_mul_ui(two_xy, temp_y, 2); // 2xy

        mpf_sub(temp_x, x_squared, y_squared); // x^2 - y^2
        mpf_add(temp_x, temp_x, a);            // x^2 - y^2 + a
        mpf_add(temp_y, two_xy, b);            // 2xy + b

        mpf_set(x, temp_x);
        mpf_set(y, temp_y);
    }

    mpf_clear(x_squared);
    mpf_clear(y_squared);
    mpf_clear(two_xy);
    mpf_clear(temp_x);
    mpf_clear(temp_y);
}

char PressKey() {
    // Press any key to continue (win32)
    // on console, don't require a newline
    std::cout << "Press any key to continue...";
    // Get the character pressed and return it
    return (char)_getch();
}

constexpr auto MultiplyOperator = Operator::MultiplyKaratsubaV2;

template<typename TestSharkParams>
bool CorrectnessTests() {
    //TestNullKernel();
    //PressKey();

    int testBase = 0;

    bool res = TestConversion<TestSharkParams>(0);
    if (!res) {
        auto q = PressKey();
        if (q == 'q') {
            return false;
        }
    }

    testBase = 2000;
    res = TestAllBinaryOp<TestSharkParams, MultiplyOperator>(testBase);
    if (!res) {
        auto q = PressKey();
        if (q == 'q') {
            return false;
        }
    }

#ifdef MULTI_KERNEL
    testBase = 4000;
    res = TestAllBinaryOp<TestSharkParams, Operator::Add>(testBase);
    if (!res) {
        auto q = PressKey();
        if (q == 'q') {
            return false;
        }
    }
#endif

    return true;
}

int main(int /*argc*/, char * /*argv*/[]) {
    int testBase = 0;
    bool res = false;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.sharedMemPerMultiprocessor << " bytes of shared memory per block." << std::endl;

        // persistingL2CacheMaxSize
        std::cout << "Device " << i << ": " << prop.persistingL2CacheMaxSize << " bytes of L2 cache." << std::endl;
    }

    if constexpr (SharkCorrectnessTests) {
        do {
            if (!CorrectnessTests<TestCorrectnessSharkParams1>()) {
                return 0;
            }

            if (!CorrectnessTests<TestCorrectnessSharkParams2>()) {
                return 0;
            }

            if (!CorrectnessTests<TestCorrectnessSharkParams3>()) {
                return 0;
            }

            if (!CorrectnessTests<TestCorrectnessSharkParams4>()) {
                return 0;
            }

            if (!CorrectnessTests<TestCorrectnessSharkParams5>()) {
                return 0;
            }

            ComicalCorrectness();
        } while (SharkInfiniteCorrectnessTests);

        if (PressKey() == 'q') {
            return 0;
        }
    }

    if constexpr (MultiKernel) {
        testBase = 6000;
        res = TestBinaryOperatorPerf<Operator::Add>(testBase);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return 0;
            }
        }
    }

    testBase = 7000;
    res = TestBinaryOperatorPerf<MultiplyOperator>(testBase);
    if (!res) {
        auto q = PressKey();
        if (q == 'q') {
            return 0;
        }
    }

    if constexpr (!SharkCorrectnessTests) {
        auto q = PressKey();
        if (q == 'q') {
            return 0;
        }

        //TestNullKernel();
        //q = PressKey();
        //if (q == 'q') {
        //    return 0;
        //}

        if (!CorrectnessTests<TestCorrectnessSharkParams1>()) {
            return 0;
        }

        if (!CorrectnessTests<TestCorrectnessSharkParams2>()) {
            return 0;
        }

        if (!CorrectnessTests<TestCorrectnessSharkParams3>()) {
            return 0;
        }

        if (!CorrectnessTests<TestCorrectnessSharkParams4>()) {
            return 0;
        }

        if (!CorrectnessTests<TestCorrectnessSharkParams5>()) {
            return 0;
        }
    }

    return 0;
}
