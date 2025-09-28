#include "TestVerbose.h"

#include <cuda_runtime.h>

#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <assert.h>


static TestTracker Tests;

template<class SharkFloatParams>
void TestConvertNumber (
    int testNum,
    const char *numberStr) {

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpf_x;
    mpf_init(mpf_x);

    auto res = mpf_set_str(mpf_x, numberStr, 10);
    if (res == -1) {
        std::cout << "Error setting mpf_x" << std::endl;
    }

    // Print the original input values
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "numberStr: " << numberStr << std::endl;
        std::cout << "X: " << MpfToString<SharkFloatParams>(
            mpf_x, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "X hex: " << MpfToHexString(mpf_x) << std::endl;
    }

    // Convert the input values to HpSharkFloat<SharkFloatParams> representations
    HpSharkFloat<SharkFloatParams> x_num{};
    x_num.MpfToHpGpu(mpf_x, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);

    if (SharkVerbose == VerboseMode::Debug) {
        // Convert the HpSharkFloat<SharkFloatParams> results to strings
        std::string gpu_str = x_num.ToString();
        std::cout << "\nHighPrecisionNumber representations:" << std::endl;
        std::cout << "X: " << gpu_str << std::endl;
        std::cout << "X hex: " << x_num.ToHexString() << std::endl;
    }

    // Convert the HpSharkFloat<SharkFloatParams> results to mpf_t for comparison
    mpf_t mpf_x_gpu_result;
    mpf_init(mpf_x_gpu_result);

    x_num.HpGpuToMpf(mpf_x_gpu_result);

    // Convert to HDRFloat using ToHDRFloat
    auto hdrFloat = x_num.ToHDRFloat<float>();
    auto hdrDouble = x_num.ToHDRFloat<double>();
    if (SharkVerbose == VerboseMode::Debug) {
        auto fstr = hdrFloat.ToString<false>();
        auto dstr = hdrDouble.ToString<false>();
        std::cout << "\nHDRFloat representations:" << std::endl;
        std::cout << "HDRFloat<float>: " << fstr << std::endl;
        std::cout << "HDRFloat<double>: " << dstr << std::endl;
    }

    HpSharkFloat<SharkFloatParams> lowPrecisionSharkFloat;
    HpSharkFloat<SharkFloatParams> lowPrecisionSharkDouble;

    lowPrecisionSharkFloat.FromHDRFloat(hdrFloat);
    lowPrecisionSharkDouble.FromHDRFloat(hdrDouble);
    if (SharkVerbose == VerboseMode::Debug) {
        auto fstr = hdrFloat.ToString<false>();
        auto dstr = hdrDouble.ToString<false>();
        std::cout << "\nLow-precision HpSharkFloat representations from HDRFloat:" << std::endl;
        std::cout << "From HDRFloat<float>: " << lowPrecisionSharkFloat.ToString() << std::endl;
        std::cout << "From HDRFloat<double>: " << lowPrecisionSharkDouble.ToString() << std::endl;
    }

    // Compute the differences between host and GPU results
    mpf_t mpf_diff;
    mpf_init(mpf_diff);

    mpf_sub(mpf_diff, mpf_x, mpf_x_gpu_result);

    // Take absolute delta:
    {
        // Compute the precision-based threshold epislon = 2^(-DefaultMpirBits)
        mpf_t mpf_diff_abs;
        mpf_init(mpf_diff_abs);
        mpf_abs(mpf_diff_abs, mpf_diff);

        mpf_t mpf_threshold;
        mpf_init2(
            mpf_threshold,
            HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
        mpf_set_ui(mpf_threshold, 1);  // mpf_threshold = 1

        // divide by 2^DefaultMpirBits
        // constexpr auto PrecisionOffset = 2 * 8 * sizeof(HpSharkFloat<SharkFloatParams>::DigitType);
        constexpr auto PrecisionOffset = 1;

        mpf_div_2exp(
            mpf_threshold,
            mpf_threshold,
            HpSharkFloat<SharkFloatParams>::DefaultMpirBits - PrecisionOffset
        );

        // for reporting: turn epsilon into a string at the same print precision you use elsewhere
        auto thresholdStr = MpfToString<SharkFloatParams>(
            mpf_threshold,
            HpSharkFloat<SharkFloatParams>::DefaultPrecBits
        );

        auto diffStr = MpfToString<SharkFloatParams>(mpf_diff_abs, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);

        // Converted GPU result
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\nConverted GPU result:" << std::endl;
            std::cout << "X: " << MpfToString<SharkFloatParams>(mpf_x_gpu_result, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
            std::cout << "X hex: " << MpfToHexString(mpf_x_gpu_result) << std::endl;

            // Print the differences
            std::cout << "\nDifference between host and GPU results:" << std::endl;
            std::cout << diffStr << std::endl;
            std::cout << "Threshold epsilon: " << thresholdStr << std::endl;
        }

        // now compare |delta| against epsilon instead of a hard-coded 1e-30
        if (mpf_cmp(mpf_diff_abs, mpf_threshold) > 0) {
            Tests.MarkFailed(testNum, "conversion", diffStr, thresholdStr);
            assert(false);
        } else {
            Tests.MarkSuccess(testNum, "conversion");
        }

        // clean up
        mpf_clear(mpf_threshold);
        mpf_clear(mpf_diff_abs);
    }

    // Clean up MPIR variables
    mpf_clear(mpf_x);
    mpf_clear(mpf_diff);
    mpf_clear(mpf_x_gpu_result);
}

template<class SharkFloatParams>
bool TestConversion(int testBase) {
    const auto set1 = testBase + 10;
    TestConvertNumber<SharkFloatParams>(set1 + 1, "0.0");
    TestConvertNumber<SharkFloatParams>(set1 + 2, "1.0");
    TestConvertNumber<SharkFloatParams>(set1 + 3, "2.0");
    TestConvertNumber<SharkFloatParams>(set1 + 4, "3.0");

    const auto set2 = testBase + 20;
    TestConvertNumber<SharkFloatParams>(set2 + 1, "0.1");
    TestConvertNumber<SharkFloatParams>(set2 + 2, "0.2");
    TestConvertNumber<SharkFloatParams>(set2 + 3, "0.3");
    TestConvertNumber<SharkFloatParams>(set2 + 4, "0.4");
    TestConvertNumber<SharkFloatParams>(set2 + 5, "0.5");
    TestConvertNumber<SharkFloatParams>(set2 + 6, "0.6");
    TestConvertNumber<SharkFloatParams>(set2 + 7, "0.7");
    TestConvertNumber<SharkFloatParams>(set2 + 8, "0.8");
    TestConvertNumber<SharkFloatParams>(set2 + 9, "0.9");

    const auto set3 = testBase + 30;
    TestConvertNumber<SharkFloatParams>(set3 + 1, "1e-50");
    TestConvertNumber<SharkFloatParams>(set3 + 2, "1e-100");
    TestConvertNumber<SharkFloatParams>(set3 + 3, "1e-150");
    TestConvertNumber<SharkFloatParams>(set3 + 4, "1e-500");
    TestConvertNumber<SharkFloatParams>(set3 + 5, "1e-1000");
    TestConvertNumber<SharkFloatParams>(set3 + 6, "-1e-50");
    TestConvertNumber<SharkFloatParams>(set3 + 7, "-1e-100");
    TestConvertNumber<SharkFloatParams>(set3 + 8, "-1e-150");
    TestConvertNumber<SharkFloatParams>(set3 + 9, "-1e-500");

    const auto set4 = testBase + 40;
    TestConvertNumber<SharkFloatParams>(set4 + 1, "-1");
    TestConvertNumber<SharkFloatParams>(set4 + 2, "-2");
    TestConvertNumber<SharkFloatParams>(set4 + 3, "-3");
    TestConvertNumber<SharkFloatParams>(set4 + 4, "-4");

    const auto set5 = testBase + 50;
    TestConvertNumber<SharkFloatParams>(set5 + 1, "-0.1");
    TestConvertNumber<SharkFloatParams>(set5 + 2, "-0.2");
    TestConvertNumber<SharkFloatParams>(set5 + 3, "-0.3");
    TestConvertNumber<SharkFloatParams>(set5 + 4, "-0.4");
    TestConvertNumber<SharkFloatParams>(set5 + 5, "-0.5");
    TestConvertNumber<SharkFloatParams>(set5 + 6, "-0.6");
    TestConvertNumber<SharkFloatParams>(set5 + 7, "-0.7");
    TestConvertNumber<SharkFloatParams>(set5 + 8, "-0.8");
    TestConvertNumber<SharkFloatParams>(set5 + 9, "-0.9");
    TestConvertNumber<SharkFloatParams>(set5 + 10, "1.999999999");
    TestConvertNumber<SharkFloatParams>(set5 + 11, "1.99999999999999999999999999999");
    TestConvertNumber<SharkFloatParams>(set5 + 12, "1.9999999999999999999999999999999999999999999999");

    const auto set6 = testBase + 70;
    TestConvertNumber<SharkFloatParams>(set6 + 1, "4294967297");
    TestConvertNumber<SharkFloatParams>(set6 + 2, "18446744073709551617");
    TestConvertNumber<SharkFloatParams>(set6 + 3, "55340232221128654849"); // 2^65 + 2^64 + 1
    TestConvertNumber<SharkFloatParams>(set6 + 4, "-4294967297");
    TestConvertNumber<SharkFloatParams>(set6 + 5, "-18446744073709551617");
    TestConvertNumber<SharkFloatParams>(set6 + 6, "-55340232221128654849");
    TestConvertNumber<SharkFloatParams>(set6 + 7, "18446744073709551615");

    const auto set7 = testBase + 80;
    TestConvertNumber<SharkFloatParams>(set7 + 1, "4294967297.0000152587890625"); // 2^32 + 1 + 1/2^16
    TestConvertNumber<SharkFloatParams>(set7 + 2, "18446744073709551617.0000152587890625");
    TestConvertNumber<SharkFloatParams>(set7 + 3, "55340232221128654849.0000152587890625");

    return Tests.CheckAllTestsPassed();
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template bool TestConversion<SharkFloatParams>(int testBase);

ExplicitInstantiateAll();