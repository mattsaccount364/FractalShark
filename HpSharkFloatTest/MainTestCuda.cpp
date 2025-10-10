#include "Conversion.h"
#include "HpSharkFloat.cuh"
#include "NullKernel.cuh"
#include "TestVerbose.h"
#include "Tests.h"

#include <cuda_runtime.h>

#include "MainTestCuda.h"
#include <conio.h>
#include <iostream>
#include <mpir.h>
#include <stdarg.h>

#include <chrono>  // steady_clock
#include <conio.h> // _kbhit()
#include <iostream>
#include <string>

#define NOMINMAX
#include <windows.h> // Sleep()

#include "HDRFloat.h"

// Function to perform the calculation on the host using MPIR
void
computeNextXY_host(mpf_t x, mpf_t y, mpf_t a, mpf_t b, int num_iter)
{
    mpf_t x_squared, y_squared, two_xy, temp_x, temp_y;
    mpf_init(x_squared);
    mpf_init(y_squared);
    mpf_init(two_xy);
    mpf_init(temp_x);
    mpf_init(temp_y);

    for (int iter = 0; iter < num_iter; ++iter) {
        mpf_mul(x_squared, x, x);      // x^2
        mpf_mul(y_squared, y, y);      // y^2
        mpf_mul(temp_y, x, y);         // xy
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

char
PressKey()
{
    // Press any key to continue (win32)
    // on console, don't require a newline
    std::cout << "Press any key to continue...";
    // Get the character pressed and return it
    return (char)_getch();
}

template <typename TestSharkParams>
bool
CorrectnessTests()
{
    // TestNullKernel();
    // PressKey();

    int testBase = 0;

    bool res;

    if constexpr (SharkEnableConversionTests) {
        res = TestConversion<TestSharkParams>(0);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return false;
            }
        }
    }

    if constexpr (SharkEnableAddKernel) {
        testBase = 4000;
        res = TestAllBinaryOp<TestSharkParams, Operator::Add>(testBase);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return false;
            }
        }
    }

    if constexpr (SharkEnableMultiplyNTTKernel) {
        testBase = 6000;
        res = TestAllBinaryOp<TestSharkParams, Operator::MultiplyNTT>(testBase);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return false;
            }
        }
    }

    if constexpr (SharkEnableReferenceKernel) {
        testBase = 2000;
        res = TestAllBinaryOp<TestSharkParams, Operator::ReferenceOrbit>(testBase);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return false;
            }
        }
    }

    return true;
}

int
RunCorrectnessTest()
{
    std::atomic<uint64_t> testCount = 0;

#if (ENABLE_BASIC_CORRECTNESS == 0) || (ENABLE_BASIC_CORRECTNESS == 1) || (ENABLE_BASIC_CORRECTNESS == 3)
    do {
        if (!CorrectnessTests<TestCorrectnessSharkParams1>()) {
            return 0;
        }

#if (ENABLE_BASIC_CORRECTNESS == 1) || (ENABLE_BASIC_CORRECTNESS == 3)
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
#endif

        ComicalCorrectness();

        testCount++;
    } while (SharkTestInfiniteCorrectness);

    if (PressKey() == 'q') {
        return 0;
    }
#endif

    return 1;
}

/// Prompts the user with `promptText`, waits up to `timeoutSec` seconds for a line
/// on stdin, and returns the parsed integer. If the user types nothing within the
/// timeout, returns `defaultValue`.
int
PromptIntWithTimeout(const std::string &promptText,
                     int defaultValue = 1,
                     int timeoutSec = 3,
                     int sleepIntervalMs = 50)
{
    std::cout << promptText << " " << std::flush;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
    std::string line;

    while (std::chrono::steady_clock::now() < deadline) {
        if (_kbhit()) {
            // read the rest of the line
            std::getline(std::cin, line);
            break;
        }
        Sleep(sleepIntervalMs);
    }

    if (line.empty()) {
        std::cout << "\n( no input in " << timeoutSec << "s, defaulting to " << defaultValue << " )\n";
        return defaultValue;
    }

    try {
        return std::stoi(line);
    } catch (...) {
        std::cout << "( couldn’t parse “" << line << "”, defaulting to " << defaultValue << " )\n";
        return defaultValue;
    }
}

int
main(int /*argc*/, char * /*argv*/[])
{
    bool res = false;

    constexpr auto timeoutInSec = 3;
    int verboseInput =
        PromptIntWithTimeout("Verbose? Default=0. (0 = No, 1 = Yes):", /*default=*/0, timeoutInSec);
    if (verboseInput == 1) {
        SetVerboseMode(VerboseMode::Debug);
    } else {
        SetVerboseMode(VerboseMode::None);
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.sharedMemPerMultiprocessor
                  << " bytes of shared memory per block." << std::endl;

        // persistingL2CacheMaxSize
        std::cout << "Device " << i << ": " << prop.persistingL2CacheMaxSize << " bytes of L2 cache."
                  << std::endl;
    }

    // TestNullKernel();
    // q = PressKey();
    // if (q == 'q') {
    //     return 0;
    // }

    if constexpr (SharkTestCorrectness) {
        res = RunCorrectnessTest();
        if (!res) {
            return 0;
        }
    }

#if (ENABLE_BASIC_CORRECTNESS == 2) || (ENABLE_BASIC_CORRECTNESS == 1)
    int numIters = PromptIntWithTimeout("NumIters? Default 5", /*default=*/5, timeoutInSec);
    int internalTestLoopCount =
        PromptIntWithTimeout("CUDA iteration count? Default 1000", /*default=*/1000, timeoutInSec);

    int testBase = 0;
    if constexpr (SharkEnableAddKernel) {
        testBase = 10000;
        res = TestBinaryOperatorPerf<Operator::Add>(testBase, numIters, internalTestLoopCount);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return 0;
            }
        }
    }

    if constexpr (SharkEnableMultiplyNTTKernel) {
        testBase = 13000;
        res = TestBinaryOperatorPerf<Operator::MultiplyNTT>(testBase, numIters, internalTestLoopCount);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return 0;
            }
        }
    }

    if constexpr (SharkEnableReferenceKernel) {
        testBase = 14000;
        res =
            TestBinaryOperatorPerf<Operator::ReferenceOrbit>(testBase, numIters, internalTestLoopCount);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return 0;
            }
        }
    }

    if constexpr (SharkEnableFullKernel) {
        testBase = 16000;
        res = TestFullReferencePerf<Operator::ReferenceOrbit>(testBase, internalTestLoopCount);
        if (!res) {
            auto q = PressKey();
            if (q == 'q') {
                return 0;
            }
        }
    }
#endif

    if constexpr (!SharkTestCorrectness) {
        auto q = PressKey();
        if (q == 'q') {
            return 0;
        }

        res = RunCorrectnessTest();
        if (!res) {
            return 0;
        }
    }

    return 0;
}
