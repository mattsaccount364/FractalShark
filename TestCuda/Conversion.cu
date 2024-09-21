#include <cuda_runtime.h>

#include "HpGpu.cuh"
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

static TestTracker Tests;

void TestConvertNumber (
    bool verbose,
    int testNum,
    const char*numberStr) {

    mpf_set_default_prec(HpGpu::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpf_x;
    mpf_init(mpf_x);

    auto res = mpf_set_str(mpf_x, numberStr, 10);
    if (res == -1) {
        std::cout << "Error setting mpf_x" << std::endl;
    }

    // Print the original input values
    if (verbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "numberStr: " << numberStr << std::endl;
        std::cout << "X: " << MpfToString(mpf_x, HpGpu::DefaultPrecBits) << std::endl;
        std::cout << "X hex: " << MpfToHexString(mpf_x, HpGpu::DefaultPrecBits) << std::endl;
    }

    // Convert the input values to HpGpu representations
    HpGpu x_num{};
    MpfToHpGpu(mpf_x, x_num, HpGpu::DefaultPrecBits);

    // Convert the HpGpu results to strings
    std::string gpu_str = x_num.ToString();

    if (verbose) {
        std::cout << "\nHighPrecisionNumber representations:" << std::endl;
        std::cout << "X: " << gpu_str << std::endl;
        std::cout << "X hex: " << x_num.ToHexString() << std::endl;
    }

    // Convert the HpGpu results to mpf_t for comparison
    mpf_t mpf_x_gpu_result;
    mpf_init(mpf_x_gpu_result);

    HpGpuToMpf(x_num, mpf_x_gpu_result);

    // Compute the differences between host and GPU results
    mpf_t mpf_diff;
    mpf_init(mpf_diff);

    mpf_sub(mpf_diff, mpf_x, mpf_x_gpu_result);

    // Take absolute delta:
    mpf_t mpf_diff_abs; 
    mpf_init(mpf_diff_abs);
    mpf_abs(mpf_diff_abs, mpf_diff);

    // Converted GPU result
    if (verbose) {
        std::cout << "\nConverted GPU result:" << std::endl;
        std::cout << "X: " << MpfToString(mpf_x_gpu_result, HpGpu::DefaultPrecBits) << std::endl;
        std::cout << "X hex: " << MpfToHexString(mpf_x_gpu_result, HpGpu::DefaultPrecBits) << std::endl;
    }

    // Print the differences
    std::cout << "\nDifference between host and GPU results:" << std::endl;
    std::cout << MpfToString(mpf_diff_abs, HpGpu::DefaultPrecBits) << std::endl;

    // If absolute delta is greater than 1e-300, the test is considered failed
    if (mpf_cmp_d(mpf_diff_abs, 1e-30) > 0) {
        Tests.MarkFailed(testNum);
    }

    // Clean up MPIR variables
    mpf_clear(mpf_x);
    mpf_clear(mpf_diff);
    mpf_clear(mpf_diff_abs);
    mpf_clear(mpf_x_gpu_result);
}

void TestConversion() {
    constexpr bool verbose = true;
    const auto set1 = 0;
    TestConvertNumber(verbose, set1 + 1, "0.0");
    TestConvertNumber(verbose, set1 + 2, "1.0");
    TestConvertNumber(verbose, set1 + 3, "2.0");
    TestConvertNumber(verbose, set1 + 4, "3.0");

    const auto set2 = 20;
    TestConvertNumber(verbose, set2 + 1, "0.1");
    TestConvertNumber(verbose, set2 + 2, "0.2");
    TestConvertNumber(verbose, set2 + 3, "0.3");
    TestConvertNumber(verbose, set2 + 4, "0.4");

    const auto set3 = 30;
    TestConvertNumber(verbose, set3 + 1, "1e-50");
    TestConvertNumber(verbose, set3 + 2, "1e-100");
    TestConvertNumber(verbose, set3 + 3, "1e-150");
    TestConvertNumber(verbose, set3 + 4, "1e-500");
    TestConvertNumber(verbose, set3 + 5, "1e-1000");
    TestConvertNumber(verbose, set3 + 6, "-1e-50");
    TestConvertNumber(verbose, set3 + 7, "-1e-100");
    TestConvertNumber(verbose, set3 + 8, "-1e-150");
    TestConvertNumber(verbose, set3 + 9, "-1e-500");

    const auto set4 = 40;
    TestConvertNumber(verbose, set4 + 1, "-1");
    TestConvertNumber(verbose, set4 + 2, "-2");
    TestConvertNumber(verbose, set4 + 3, "-3");
    TestConvertNumber(verbose, set4 + 4, "-4");

    const auto set5 = 50;
    TestConvertNumber(verbose, set5 + 1, "-0.1");
    TestConvertNumber(verbose, set5 + 2, "-0.2");
    TestConvertNumber(verbose, set5 + 3, "-0.3");
    TestConvertNumber(verbose, set5 + 4, "-0.4");

    const auto set6 = 60;
    TestConvertNumber(verbose, set6 + 1, "4294967297");
    TestConvertNumber(verbose, set6 + 2, "18446744073709551617");
    TestConvertNumber(verbose, set6 + 3, "55340232221128654849"); // 2^65 + 2^64 + 1
    TestConvertNumber(verbose, set6 + 4, "-4294967297");
    TestConvertNumber(verbose, set6 + 5, "-18446744073709551617");
    TestConvertNumber(verbose, set6 + 6, "-55340232221128654849");

    const auto set7 = 70;
    TestConvertNumber(verbose, set7 + 1, "4294967297.0000152587890625"); // 2^32 + 1 + 1/2^16
    TestConvertNumber(verbose, set7 + 2, "18446744073709551617.0000152587890625");
    TestConvertNumber(verbose, set7 + 3, "55340232221128654849.0000152587890625");
    Tests.CheckAllTestsPassed();
}