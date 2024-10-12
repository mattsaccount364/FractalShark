#include <cuda_runtime.h>

#include "HpGpu.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.cuh"
#include "Add.cuh"
#include "Multiply.cuh"

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

template<Operator sharkOperator>
void DiffAgainstHost(
    int testNum,
    const mpf_t mpfHostResult,
    const HpGpu &gpuResult) {

    if (Verbose) {
        std::cout << "\nGPU result: " << std::endl;
        std::cout << gpuResult.ToString() << std::endl;
        std::cout << gpuResult.ToHexString() << std::endl;
    }

    // Convert the HpGpu results to mpf_t for comparison
    mpf_t mpfXGpuResult;
    mpf_init(mpfXGpuResult);

    HpGpuToMpf(gpuResult, mpfXGpuResult);

    // Compute the differences between host and GPU results
    mpf_t mpfDiff;
    mpf_init(mpfDiff);

    mpf_sub(mpfDiff, mpfHostResult, mpfXGpuResult);

    // Take absolute delta:
    mpf_t mpf_diff_abs;
    mpf_init(mpf_diff_abs);
    mpf_abs(mpf_diff_abs, mpfDiff);

    // Converted GPU result
    if (Verbose) {
        std::cout << "\nConverted GPU result:" << std::endl;
        std::cout << MpfToString(mpfXGpuResult, HpGpu::DefaultPrecBits) << std::endl;

        // Print the differences
        std::cout << "\nDifference between host and GPU results:" << std::endl;
        std::cout << MpfToString(mpf_diff_abs, LowPrec) << std::endl;
    }

    // If absolute delta is greater than 1e-30, the test is considered failed
    if (mpf_cmp_d(mpf_diff_abs, 1e-30) > 0) {
        Tests.MarkFailed(testNum, mpf_diff_abs);
    }

    mpf_clear(mpfDiff);
    mpf_clear(mpf_diff_abs);
    mpf_clear(mpfXGpuResult);
}

template<Operator sharkOperator>
void TestAddTwoNumbersPerf(
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t mpfX,
    const mpf_t mpfY) {

    // Print the original input values
    if (Verbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: " << MpfToString(mpfX, HpGpu::DefaultPrecBits) << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "Y: " << MpfToString(mpfY, HpGpu::DefaultPrecBits) << std::endl;
    }

    std::unique_ptr<HpGpu> xNum = std::make_unique<HpGpu>();
    std::unique_ptr<HpGpu> yNum = std::make_unique<HpGpu>();
    std::unique_ptr<HpGpu> resultNum = std::make_unique<HpGpu>();
    MpfToHpGpu(mpfX, *xNum, HpGpu::DefaultPrecBits);
    MpfToHpGpu(mpfY, *yNum, HpGpu::DefaultPrecBits);
    if (Verbose) {
        std::cout << "\nConverted HpGpu representations:" << std::endl;
        std::cout << "X: " << xNum->ToString() << std::endl;
        std::cout << "Y: " << yNum->ToString() << std::endl;
    }

    // Perform the calculation on the host using MPIR
    mpf_t mpfHostResult;
    mpf_init(mpfHostResult);

    {
        BenchmarkTimer hostTimer;
        ScopedBenchmarkStopper hostStopper{ hostTimer };

        if constexpr (sharkOperator == Operator::Add) {
            mpf_add(mpfHostResult, mpfX, mpfY);
        }
        else if constexpr (sharkOperator == Operator::Multiply) {
            mpf_mul(mpfHostResult, mpfX, mpfY);
        }

        hostTimer.StopTimer();

        std::cout << "Host iter time: " << hostTimer.GetDeltaInMs() << " ms" << std::endl;
    }

    std::unique_ptr<HpGpu> gpuResult2 = std::make_unique<HpGpu>();

    {
        // Perform the calculation on the GPU
        HpGpu *xGpu;
        cudaMalloc(&xGpu, sizeof(HpGpu));
        cudaMemcpy(xGpu, xNum.get(), sizeof(HpGpu), cudaMemcpyHostToDevice);

        HpGpu *yGpu;
        cudaMalloc(&yGpu, sizeof(HpGpu));
        cudaMemcpy(yGpu, yNum.get(), sizeof(HpGpu), cudaMemcpyHostToDevice);

        HpGpu *internalGpuResult2;
        cudaMalloc(&internalGpuResult2, sizeof(HpGpu));
        cudaMemset(internalGpuResult2, 0, sizeof(HpGpu));

        // Allocate memory for carryOuts and cumulativeCarries
        GlobalAddBlockData *globalBlockData;
        CarryInfo *d_carryOuts;
        uint32_t *d_cumulativeCarries;
        cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
        cudaMalloc(&d_carryOuts, (NumBlocks + 1) * sizeof(CarryInfo));
        cudaMalloc(&d_cumulativeCarries, (NumBlocks + 1) * sizeof(uint32_t));

        // Prepare kernel arguments
        void *kernelArgs[] = {
            (void *)&xGpu,
            (void *)&yGpu,
            (void *)&internalGpuResult2,
            (void *)&globalBlockData,
            (void *)&d_carryOuts,
            (void *)&d_cumulativeCarries
        };

        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpuTestLoop(kernelArgs);

        // Launch the cooperative kernel

        cudaMemcpy(gpuResult2.get(), internalGpuResult2, sizeof(HpGpu), cudaMemcpyDeviceToHost);

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;

        cudaFree(internalGpuResult2);
        cudaFree(globalBlockData);
        cudaFree(d_carryOuts);
        cudaFree(d_cumulativeCarries);
        cudaFree(xGpu);
    }

    DiffAgainstHost<sharkOperator>(testNum, mpfHostResult, *gpuResult2);

    // Clean up MPIR variables
    mpf_clear(mpfHostResult);
}

template<Operator sharkOperator>
void TestAddTwoNumbersPerf(
    int testNum,
    const char *num1,
    const char *num2) {

    mpf_set_default_prec(HpGpu::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_init(mpfX);
    mpf_init(mpfY);

    auto res = mpf_set_str(mpfX, num1, 10);
    if (res == -1) {
        std::cout << "Error setting mpfX" << std::endl;
    }

    res = mpf_set_str(mpfY, num2, 10);
    if (res == -1) {
        std::cout << "Error setting mpfY" << std::endl;
    }

    TestAddTwoNumbersPerf<sharkOperator>(testNum, num1, num2, mpfX, mpfY);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

template<Operator sharkOperator>
void TestBinOperatorTwoNumbers(
    int testNum,
    const HpGpu &xNum,
    const HpGpu &yNum,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    if (Verbose) {
        std::cout << "\nConverted HpGpu representations:" << std::endl;
        std::cout << "X: " << xNum.ToString() << std::endl;
        std::cout << "X hex: " << xNum.ToHexString() << std::endl;
        std::cout << "Y: " << yNum.ToString() << std::endl;
        std::cout << "Y hex: " << yNum.ToHexString() << std::endl;
    }

    // Perform the calculation on the GPU
    HpGpu *xGpu;
    HpGpu *yGpu;

    cudaMalloc(&xGpu, sizeof(HpGpu));
    cudaMalloc(&yGpu, sizeof(HpGpu));
    cudaMemcpy(xGpu, &xNum, sizeof(HpGpu), cudaMemcpyHostToDevice);
    cudaMemcpy(yGpu, &yNum, sizeof(HpGpu), cudaMemcpyHostToDevice);

    {
        // Perform the calculation on the host using MPIR
        HpGpu gpuResult{};
        mpf_t mpfHostResult;
        mpf_init(mpfHostResult);

        if constexpr (sharkOperator == Operator::Add) {
            mpf_add(mpfHostResult, mpfX, mpfY);
        } else if constexpr (sharkOperator == Operator::Multiply) {
            mpf_mul(mpfHostResult, mpfX, mpfY);
        }

        // Print host result
        if (Verbose) {
            std::cout << "\nHost result:" << std::endl;
            std::cout << "Host result: " << MpfToString(mpfHostResult, HpGpu::DefaultPrecBits) << std::endl;
            std::cout << "Host hex: " << MpfToHexString(mpfHostResult) << std::endl;
        }

        HpGpu *internalGpuResult;
        cudaMalloc(&internalGpuResult, sizeof(HpGpu));

        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };

        if constexpr (sharkOperator == Operator::Add) {
            // Allocate memory for carryOuts and cumulativeCarries
            GlobalAddBlockData *globalBlockData;
            CarryInfo *d_carryOuts;
            uint32_t *d_cumulativeCarries;
            cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
            cudaMalloc(&d_carryOuts, (NumBlocks + 1) * sizeof(CarryInfo));
            cudaMalloc(&d_cumulativeCarries, (NumBlocks + 1) * sizeof(uint32_t));

            // Prepare kernel arguments
            void *kernelArgs[] = {
                (void *)&xGpu,
                (void *)&yGpu,
                (void *)&internalGpuResult,
                (void *)&globalBlockData,
                (void *)&d_carryOuts,
                (void *)&d_cumulativeCarries
            };

            ComputeAddGpu(kernelArgs);

            cudaFree(globalBlockData);
            cudaFree(d_carryOuts);
            cudaFree(d_cumulativeCarries);
        } else if constexpr (sharkOperator == Operator::Multiply) {
            // Prepare kernel arguments
            // Allocate memory for carryOuts and cumulativeCarries
            uint64_t *d_carry1;
            uint64_t *d_carry2;
            uint64_t *d_carry3;
            cudaMalloc(&d_carry1, (NumBlocks + 1) * sizeof(uint64_t));
            cudaMalloc(&d_carry2, (NumBlocks + 1) * sizeof(uint64_t));
            cudaMalloc(&d_carry3, (NumBlocks + 1) * sizeof(uint64_t));

            void *kernelArgs[] = {
                (void *)&xGpu,
                (void *)&yGpu,
                (void *)&internalGpuResult,
                (void *)&d_carry1,
                (void *)&d_carry2,
                (void *)&d_carry3
            };

            ComputeMultiplyGpu(kernelArgs);

            cudaFree(d_carry1);
            cudaFree(d_carry2);
            cudaFree(d_carry3);
        }

        cudaMemcpy(&gpuResult, internalGpuResult, sizeof(HpGpu), cudaMemcpyDeviceToHost);

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (Verbose) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }

        cudaFree(internalGpuResult);

        DiffAgainstHost<sharkOperator>(testNum, mpfHostResult, gpuResult);

        // Clean up MPIR variables
        mpf_clear(mpfHostResult);
    }

    cudaFree(xGpu);
    cudaFree(yGpu);
}

template<Operator sharkOperator>
void TestBinOperatorTwoNumbers(
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    // Print the original input values
    if (Verbose) {
        std::cout << "Original input strings:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "MpfX: " << MpfToString(mpfX, HpGpu::DefaultPrecBits) << std::endl;
        std::cout << "MpfY: " << MpfToString(mpfY, HpGpu::DefaultPrecBits) << std::endl;
    }

    // Convert the input values to HpGpu representations
    std::unique_ptr<HpGpu> xNum = std::make_unique<HpGpu>();
    std::unique_ptr<HpGpu> yNum = std::make_unique<HpGpu>();
    MpfToHpGpu(mpfX, *xNum, HpGpu::DefaultPrecBits);
    MpfToHpGpu(mpfY, *yNum, HpGpu::DefaultPrecBits);

    TestBinOperatorTwoNumbers<sharkOperator>(testNum, *xNum, *yNum, mpfX, mpfY);
}

template<Operator sharkOperator>
void TestBinOperatorTwoNumbers(
    int testNum,
    const char *num1,
    const char *num2) {

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << testNum << std::endl;

    mpf_set_default_prec(HpGpu::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpfX, mpfY;
    mpf_init(mpfX);
    mpf_init(mpfY);

    auto res = mpf_set_str(mpfX, num1, 10);
    if (res == -1) {
        std::cout << "Error setting mpfX" << std::endl;
    }

    res = mpf_set_str(mpfY, num2, 10);
    if (res == -1) {
        std::cout << "Error setting mpfY" << std::endl;
    }

    TestBinOperatorTwoNumbers<sharkOperator>(testNum, num1, num2, mpfX, mpfY);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers(int testNum, std::vector<uint32_t> &digits1, std::vector<uint32_t> &digits2) {
    mpf_t x, y;
    mpf_init(x);
    mpf_init(y);

    auto strLargeX = Uint32ToMpf(digits1.data(), HpGpu::NumUint32 / 2, x);
    auto strLargeY = Uint32ToMpf(digits2.data(), HpGpu::NumUint32 / 2, y);
    TestBinOperatorTwoNumbers<sharkOperator>(testNum, strLargeX.c_str(), strLargeY.c_str(), x, y);

    mpf_clear(x);
    mpf_clear(y);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers(
    int testNum,
    const HpGpu &xNum,
    const HpGpu &yNum) {

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_init(mpfX);
    mpf_init(mpfY);
    HpGpuToMpf(xNum, mpfX);
    HpGpuToMpf(yNum, mpfY);

    TestBinOperatorTwoNumbers<sharkOperator>(testNum, xNum, yNum, mpfX, mpfY);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers1(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < HpGpu::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == HpGpu::NumUint32);
    testData[testData.size() - 1] = 0x80000000;

    TestAddSpecialNumbers<sharkOperator>(testNum, testData, testData);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers2(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < HpGpu::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == HpGpu::NumUint32);
    testData[testData.size() - 1] = 0xC0000000;

    TestAddSpecialNumbers<sharkOperator>(testNum, testData, testData);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers3(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < HpGpu::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == HpGpu::NumUint32);
    testData[testData.size() - 1] = 0xFFFFFFFF;

    TestAddSpecialNumbers<sharkOperator>(testNum, testData, testData);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers4(int testNum) {

    size_t i = 0;
    std::vector<uint32_t> testData1;
    testData1.resize(HpGpu::NumUint32);
    testData1[i++] = 0xF26D37FC;
    testData1[i++] = 0xA96025CE;
    testData1[i++] = 0xB03FC716;
    testData1[i++] = 0x1DF7182B;
    testData1[i++] = 0xCCBD69BD;
    testData1[i++] = 0x40C0F80C;
    testData1[i++] = 0xFAA0222E;
    testData1[i++] = 0xD1FDA456;

    i = 0;
    std::vector<uint32_t> testData2;
    testData2.resize(HpGpu::NumUint32);
    testData2[i++] = 0x8BBCDF3;
    testData2[i++] = 0x4C3E7ACB;
    testData2[i++] = 0x6691A71D;
    testData2[i++] = 0xDFE03842;
    testData2[i++] = 0x3FADCA11;
    testData2[i++] = 0x4058BC9E;
    testData2[i++] = 0xF30FD7DE;
    testData2[i++] = 0xAA6CA582;

    const bool isNegative = true;
    std::unique_ptr<HpGpu> xNum{ std::make_unique<HpGpu>(testData1.data(), 0, isNegative) };
    std::unique_ptr<HpGpu> yNum{ std::make_unique<HpGpu>(testData2.data(), 0, isNegative) };

    TestAddSpecialNumbers<sharkOperator>(testNum, *xNum, *yNum);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers5(int testNum) {
    size_t i = 0;
    std::vector<uint32_t> testData1;
    testData1.resize(HpGpu::NumUint32);
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0xEB062904;
    testData1[i++] = 0x0A85870C;

    i = 0;
    std::vector<uint32_t> testData2;
    testData2.resize(HpGpu::NumUint32);
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0xD02FF484;
    testData2[i++] = 0x76D16E0B;

    const bool isNegative = true;
    std::unique_ptr<HpGpu> xNum{ std::make_unique<HpGpu>(testData1.data(), 0, !isNegative) };
    std::unique_ptr<HpGpu> yNum{ std::make_unique<HpGpu>(testData2.data(), 0, isNegative) };

    TestAddSpecialNumbers<sharkOperator>(testNum, *xNum, *yNum);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers6(int testNum) {
    size_t i = 0;
    std::vector<uint32_t> testData1;
    testData1.resize(HpGpu::NumUint32);
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0xFFFFFFFF;
    testData1[i++] = 0xFFFFFFFF;

    i = 0;
    std::vector<uint32_t> testData2;
    testData2.resize(HpGpu::NumUint32);
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0xFFFFFFFF;
    testData2[i++] = 0xFFFFFFFF;

    const bool isNegative = true;
    std::unique_ptr<HpGpu> xNum{ std::make_unique<HpGpu>(testData1.data(), 0, !isNegative) };
    std::unique_ptr<HpGpu> yNum{ std::make_unique<HpGpu>(testData2.data(), 0, !isNegative) };

    TestAddSpecialNumbers<sharkOperator>(testNum, *xNum, *yNum);
}

template<Operator sharkOperator>
void TestAddSpecialNumbers7(int testNum) {
    size_t i = 0;
    std::vector<uint32_t> testData1;
    testData1.resize(HpGpu::NumUint32);
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0xFFFFFFFF;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;
    testData1[i++] = 0;

    i = 0;
    std::vector<uint32_t> testData2;
    testData2.resize(HpGpu::NumUint32);
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0xFFFFFFFF;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;
    testData2[i++] = 0;

    const bool isNegative = true;
    std::unique_ptr<HpGpu> xNum{ std::make_unique<HpGpu>(testData1.data(), 0, !isNegative) };
    std::unique_ptr<HpGpu> yNum{ std::make_unique<HpGpu>(testData2.data(), 0, !isNegative) };

    TestAddSpecialNumbers<sharkOperator>(testNum, *xNum, *yNum);
}


template<Operator sharkOperator>
bool TestAllBinaryOp(int testBase) {
    constexpr bool includeSet1 = true;
    constexpr bool includeSet2 = true;
    constexpr bool includeSet3 = true;
    constexpr bool includeSet4 = true;
    constexpr bool includeSet5 = true;
    constexpr bool includeSet6 = true;
    constexpr bool includeSet10 = true;

    // 200s is multiply
    // 400s is add

    if constexpr (includeSet1) {
        const auto set1 = testBase + 10;
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 1, "1", "2");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 2, "4294967295", "1");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 3, "4294967296", "1");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 4, "4294967295", "4294967296");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 5, "4294967296", "-1");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 6, "18446744073709551615", "1");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 7, "0", "0.1");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 8, "0.1", "0");
        TestBinOperatorTwoNumbers<sharkOperator>(set1 + 9, "0", "0");
    }

    if constexpr (includeSet2) {
        const auto set2 = testBase + 20;
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 1, "2", "0.1");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 2, "0.2", "0.1");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 3, "0.5", "1.2");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 4, "0.6", "1.3");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 5, "0.7", "1.4");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 6, "0.1", "1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<sharkOperator>(set2 + 7, "0.123124561464451654461", "1.2395123123127298375982735");
    }

    if constexpr (includeSet3) {
        const auto set3 = testBase + 30;
        TestBinOperatorTwoNumbers<sharkOperator>(set3 + 1, "-0.5", "1.2");
        TestBinOperatorTwoNumbers<sharkOperator>(set3 + 2, "-0.6", "1.3");
        TestBinOperatorTwoNumbers<sharkOperator>(set3 + 3, "-0.7", "1.4");
        TestBinOperatorTwoNumbers<sharkOperator>(set3 + 4, "-0.1", "1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<sharkOperator>(set3 + 5, "-0.123124561464451654461", "1.2395123123127298375982735");
    }

    if constexpr (includeSet4) {
        const auto set4 = testBase + 40;
        TestBinOperatorTwoNumbers<sharkOperator>(set4 + 1, "-0.5", "-1.2");
        TestBinOperatorTwoNumbers<sharkOperator>(set4 + 2, "-0.6", "-1.3");
        TestBinOperatorTwoNumbers<sharkOperator>(set4 + 3, "-0.7", "-1.4");
        TestBinOperatorTwoNumbers<sharkOperator>(set4 + 4, "-0.1", "-1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<sharkOperator>(set4 + 5, "-0.123124561464451654461", "-1.2395123123127298375982735");
    }

    if constexpr (includeSet5) {
        const auto set5 = testBase + 50;
        TestBinOperatorTwoNumbers<sharkOperator>(set5 + 1, "0.5265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<sharkOperator>(set5 + 2, "0.2999999999965542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<sharkOperator>(set5 + 3, "0.1265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866391730473289107302178039999999999999271839216");
        TestBinOperatorTwoNumbers<sharkOperator>(set5 + 4, "0.0265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<sharkOperator>(set5 + 5, "0.00000000000000000265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
    }

    if constexpr (includeSet6) {
        const auto set6 = testBase + 60;
        TestAddSpecialNumbers1<sharkOperator>(set6 + 1);
        TestAddSpecialNumbers2<sharkOperator>(set6 + 2);
        TestAddSpecialNumbers3<sharkOperator>(set6 + 3);
        TestAddSpecialNumbers4<sharkOperator>(set6 + 4);
        TestAddSpecialNumbers5<sharkOperator>(set6 + 5);
        TestAddSpecialNumbers6<sharkOperator>(set6 + 6);
        TestAddSpecialNumbers7<sharkOperator>(set6 + 7);
    }

    if constexpr (includeSet10) {
        const auto set10 = testBase + 100;
        for (auto i = 0; i < 100; i++) {
            std::unique_ptr<HpGpu> x = std::make_unique<HpGpu>();
            std::unique_ptr<HpGpu> y = std::make_unique<HpGpu>();

            x->GenerateRandomNumber();
            y->GenerateRandomNumber();

            if (Verbose) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->IsNegative << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->IsNegative << std::endl;
            }
            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            TestBinOperatorTwoNumbers<sharkOperator>(set10 + i, x_str.c_str(), y_str.c_str());
        }
    }

    return Tests.CheckAllTestsPassed();
}

template<Operator sharkOperator>
bool TestBinaryOperatorPerf() {
    const auto set20 = 200;
    TestAddTwoNumbersPerf<sharkOperator>(set20 + 1, ".1", ".1");
    return Tests.CheckAllTestsPassed();
}

// Explicitly instantiate TestBinaryOperatorPerf
template bool TestBinaryOperatorPerf<Operator::Add>();
template bool TestBinaryOperatorPerf<Operator::Multiply>();

// Explicitly instantiate TestAllBinaryOp
template bool TestAllBinaryOp<Operator::Add>(int testBase);
template bool TestAllBinaryOp<Operator::Multiply>(int testBase);