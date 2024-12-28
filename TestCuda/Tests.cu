#include <cuda_runtime.h>

#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.cuh"
#include "Add.cuh"
#include "Multiply.cuh"
#include "ReferenceKaratsuba.h"

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

// Returns false if the test fails, true otherwise
template<class SharkFloatParams, Operator sharkOperator>
bool DiffAgainstHost(
    int testNum,
    std::string hostCustomOrGpu,
    const mpf_t mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult) {

    if (Verbose) {
        std::cout << "\n" << hostCustomOrGpu << " result: " << std::endl;
        std::cout << gpuResult.ToString() << std::endl;
        std::cout << gpuResult.ToHexString() << std::endl;
    }

    // Convert the HpSharkFloat<SharkFloatParams> results to mpf_t for comparison
    mpf_t mpfXGpuResult;
    mpf_init(mpfXGpuResult);

    HpGpuToMpf(gpuResult, mpfXGpuResult);

    // Compute the differences between host and GPU results
    mpf_t mpfDiff;
    mpf_init(mpfDiff);

    mpf_sub(mpfDiff, mpfHostResult, mpfXGpuResult);

    // Take absolute delta:
    mpf_t mpfDiffAbs;
    mpf_init(mpfDiffAbs);
    mpf_abs(mpfDiffAbs, mpfDiff);

    // Converted GPU result
    if (Verbose) {
        std::cout << "\nConverted " << hostCustomOrGpu << " result:" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfXGpuResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;

        // Print the differences
        std::cout << "\nDifference between host and " << hostCustomOrGpu << " results:" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec) << std::endl;
    }

    // Check if the host result is zero to avoid division by zero
    mp_bitcnt_t gpuPrecBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    mp_bitcnt_t margin = sizeof(uint32_t) * 8 * 2;
    mp_bitcnt_t totalPrecBits = (gpuPrecBits > margin) ? (gpuPrecBits - margin) : 1;
    mpf_t acceptableError;

    bool testSucceeded = true;

    if (mpf_cmp_ui(mpfHostResult, 0) != 0) {
        // Host result is non-zero

        // Compute relative error
        mpf_t relativeError;
        mpf_init(relativeError);
        mpf_sub(relativeError, mpfHostResult, mpfXGpuResult);
        mpf_div(relativeError, relativeError, mpfHostResult);
        mpf_abs(relativeError, relativeError);

        // Compute machine epsilon: epsilon = 2^(-totalPrecBits)
        mpf_t epsilon;
        mpf_init2(epsilon, totalPrecBits);
        mpf_set_ui(epsilon, 1);
        mpf_div_2exp(epsilon, epsilon, totalPrecBits);

        // Compute acceptable error: acceptableError = epsilon * abs(hostResult)
        mpf_init(acceptableError);
        mpf_mul(acceptableError, epsilon, mpfHostResult);
        mpf_abs(acceptableError, acceptableError);

        // Compare absolute error with acceptable threshold
        auto relativeErrorStr = MpfToString<SharkFloatParams>(relativeError, LowPrec);
        auto epsilonStr = MpfToString<SharkFloatParams>(epsilon, LowPrec);
        if (mpf_cmp(relativeError, epsilon) <= 0) {
            if (Verbose) {
                std::cout << "\nThe relative error is within acceptable bounds." << std::endl;
                std::cout << "Relative error: " << epsilonStr << std::endl;
            }

            Tests.MarkSuccess(testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nError: The relative error exceeds acceptable bounds." << std::endl;
            std::cout << "Relative error: " << relativeErrorStr << std::endl;
            Tests.MarkFailed(testNum, hostCustomOrGpu, relativeErrorStr, epsilonStr);
            testSucceeded = false;
        }

        // Clean up
        mpf_clear(relativeError);
        mpf_clear(epsilon);
        mpf_clear(acceptableError);
    } else {
        // Host result is zero

        // For zero host result, use an absolute error threshold
        mpf_init2(acceptableError, totalPrecBits);
        mpf_set_ui(acceptableError, 1);
        mpf_div_2exp(acceptableError, acceptableError, totalPrecBits);

        auto mpfDiffAbsStr = MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec);
        auto absoluteErrorStr = MpfToString<SharkFloatParams>(acceptableError, LowPrec);

        if (mpf_cmp(mpfDiffAbs, acceptableError) <= 0) {
            if (Verbose) {
                std::cout << "\nThe absolute error is within acceptable bounds." << std::endl;
            }

            Tests.MarkSuccess(testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nError: The absolute error exceeds acceptable bounds." << std::endl;
            Tests.MarkFailed(testNum, hostCustomOrGpu, mpfDiffAbsStr, absoluteErrorStr);
            testSucceeded = false;
        }

        mpf_clear(acceptableError);
    }

    mpf_clear(mpfDiff);
    mpf_clear(mpfDiffAbs);
    mpf_clear(mpfXGpuResult);

    return testSucceeded;
}

template<class SharkFloatParams, typename KernelFunction>
void InvokeMultiplyKernel(
    KernelFunction kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    cudaMalloc(&d_tempProducts, 32 * SharkFloatParams::NumUint32 * sizeof(uint64_t));

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *yGpu;
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult2;
    cudaMalloc(&internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(internalGpuResult2, 0, sizeof(HpSharkFloat<SharkFloatParams>));

    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&d_tempProducts
    };

    cudaStream_t stream;
    cudaStreamCreate(&stream); // Create a stream

    cudaDeviceProp prop;
    int device_id = 0;

    {
        cudaGetDeviceProperties(&prop, device_id);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for persisting accesses */

        auto setAccess = [&](void *ptr, size_t num_bytes) {
            cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
            stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(ptr); // Global Memory data pointer
            stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
            stream_attribute.accessPolicyWindow.hitRatio = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
            stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
            stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

            //Set the attributes to a CUDA stream of type cudaStream_t
            cudaError_t err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
            if (err != cudaSuccess) {
                std::cerr << "CUDA error in setting stream attribute: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "Stream attribute set successfully" << std::endl;
            }
        };

        setAccess(xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
        setAccess(d_tempProducts, 32 * SharkFloatParams::NumUint32 * sizeof(uint64_t));
    }

    kernel(stream, kernelArgs);

    cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream); // Destroy the stream

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams, typename KernelFunction>
void InvokeAddKernel(
    KernelFunction kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult2) {

    // Allocate memory for carryOuts and cumulativeCarries
    GlobalAddBlockData *globalBlockData;
    CarryInfo *d_carryOuts;
    uint32_t *d_cumulativeCarries;
    cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
    cudaMalloc(&d_carryOuts, (SharkFloatParams::NumBlocks + 1) * sizeof(CarryInfo));
    cudaMalloc(&d_cumulativeCarries, (SharkFloatParams::NumBlocks + 1) * sizeof(uint32_t));

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *yGpu;
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult2;
    cudaMalloc(&internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(internalGpuResult2, 0, sizeof(HpSharkFloat<SharkFloatParams>));

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&globalBlockData,
        (void *)&d_carryOuts,
        (void *)&d_cumulativeCarries
    };

    kernel(kernelArgs);

    // Launch the cooperative kernel

    cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);

    cudaFree(globalBlockData);
    cudaFree(d_carryOuts);
    cudaFree(d_cumulativeCarries);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestPerf(
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t mpfX,
    const mpf_t mpfY) {

    // Print the original input values
    if (Verbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: " << MpfToString<SharkFloatParams>(mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "Y: " << MpfToString<SharkFloatParams>(mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
    }

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto resultNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    MpfToHpGpu(mpfX, *xNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    MpfToHpGpu(mpfY, *yNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    if (Verbose) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;
        std::cout << "X: " << xNum->ToString() << std::endl;
        std::cout << "Y: " << yNum->ToString() << std::endl;
    }

    // Perform the calculation on the host using MPIR
    mpf_t mpfHostResult;
    mpf_init(mpfHostResult);

    {
        BenchmarkTimer hostTimer;
        ScopedBenchmarkStopper hostStopper{ hostTimer };

        for (int i = 0; i < TestIterCount; ++i) {
            if constexpr (sharkOperator == Operator::Add) {
                mpf_add(mpfHostResult, mpfX, mpfY);
            } else if constexpr (
                sharkOperator == Operator::MultiplyN2 ||
                sharkOperator == Operator::MultiplyKaratsubaV1 ||
                sharkOperator == Operator::MultiplyKaratsubaV2) {

                mpf_mul(mpfHostResult, mpfX, mpfY);
            }
        }

        hostTimer.StopTimer();

        std::cout << "Host iter time: " << hostTimer.GetDeltaInMs() << " ms" << std::endl;
    }

    auto gpuResult2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    {
        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };

        if constexpr (sharkOperator == Operator::Add) {
            InvokeAddKernel<SharkFloatParams>(
                ComputeAddGpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2);
        } else if constexpr (sharkOperator == Operator::MultiplyN2) {
            InvokeMultiplyKernel<SharkFloatParams>(
                ComputeMultiplyN2GpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2);
        } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV1) {
            InvokeMultiplyKernel<SharkFloatParams>(
                ComputeMultiplyKaratsubaV1GpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2);
        } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
            InvokeMultiplyKernel<SharkFloatParams>(
                ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2);
        }

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;
    }

    bool testSucceeded = DiffAgainstHost<SharkFloatParams, sharkOperator>(
        testNum,
        "GPU",
        mpfHostResult,
        *gpuResult2);
    if (!testSucceeded) {
        std::cout << "Perf correctness test failed" << std::endl;
    } else {
        std::cout << "Perf correctness test succeeded" << std::endl;
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResult);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestPerf(
    int testNum) {

    HpSharkFloat<SharkFloatParams> xNum;
    HpSharkFloat<SharkFloatParams> yNum;

    xNum.GenerateRandomNumber();
    yNum.GenerateRandomNumber();

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_init(mpfX);
    mpf_init(mpfY);

    HpGpuToMpf(xNum, mpfX);
    HpGpuToMpf(yNum, mpfY);

    auto num1 = xNum.ToString();
    auto num2 = yNum.ToString();

    TestPerf<SharkFloatParams, sharkOperator>(testNum, num1.c_str(), num2.c_str(), mpfX, mpfY);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

template<class SharkFloatParams, Operator sharkOperator, typename KernelFunction>
void InvokeMultiplyKernelCorrectness(
    KernelFunction kernel,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult) {

    // Prepare kernel arguments
    // Allocate memory for carryOuts and cumulativeCarries
    uint64_t *d_tempProducts;
    cudaMalloc(&d_tempProducts, 32 * SharkFloatParams::NumUint32 * sizeof(uint64_t));

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *yGpu;
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult2;
    cudaMalloc(&internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemset(internalGpuResult2, 0, sizeof(HpSharkFloat<SharkFloatParams>));

    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult2,
        (void *)&d_tempProducts
    };

    kernel(kernelArgs);

    cudaMemcpy(&gpuResult, internalGpuResult2, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);

    cudaFree(internalGpuResult2);
    cudaFree(yGpu);
    cudaFree(xGpu);
    cudaFree(d_tempProducts);
}

template<class SharkFloatParams, Operator sharkOperator, typename KernelFunction>
void InvokeAddKernelCorrectness(
    KernelFunction /*kernel*/,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    HpSharkFloat<SharkFloatParams> &gpuResult) {

    // Perform the calculation on the GPU
    HpSharkFloat<SharkFloatParams> *xGpu;
    HpSharkFloat<SharkFloatParams> *yGpu;

    cudaMalloc(&xGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMalloc(&yGpu, sizeof(HpSharkFloat<SharkFloatParams>));
    cudaMemcpy(xGpu, &xNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);
    cudaMemcpy(yGpu, &yNum, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyHostToDevice);

    HpSharkFloat<SharkFloatParams> *internalGpuResult;
    cudaMalloc(&internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>));

    // Allocate memory for carryOuts and cumulativeCarries
    GlobalAddBlockData *globalBlockData;
    CarryInfo *d_carryOuts;
    uint32_t *d_cumulativeCarries;
    cudaMalloc(&globalBlockData, sizeof(GlobalAddBlockData));
    cudaMalloc(&d_carryOuts, (SharkFloatParams::NumBlocks + 1) * sizeof(CarryInfo));
    cudaMalloc(&d_cumulativeCarries, (SharkFloatParams::NumBlocks + 1) * sizeof(uint32_t));

    // Prepare kernel arguments
    void *kernelArgs[] = {
        (void *)&xGpu,
        (void *)&yGpu,
        (void *)&internalGpuResult,
        (void *)&globalBlockData,
        (void *)&d_carryOuts,
        (void *)&d_cumulativeCarries
    };

    ComputeAddGpu<SharkFloatParams>(kernelArgs);

    cudaFree(globalBlockData);
    cudaFree(d_carryOuts);
    cudaFree(d_cumulativeCarries);

    cudaMemcpy(&gpuResult, internalGpuResult, sizeof(HpSharkFloat<SharkFloatParams>), cudaMemcpyDeviceToHost);
    cudaFree(internalGpuResult);

    cudaFree(yGpu);
    cudaFree(xGpu);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbersRawNoSignChange(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    if (Verbose) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;
        std::cout << "X: " << xNum.ToString() << std::endl;
        std::cout << "X hex: " << xNum.ToHexString() << std::endl;
        std::cout << "Y: " << yNum.ToString() << std::endl;
        std::cout << "Y hex: " << yNum.ToHexString() << std::endl;
    }

    auto TestHostKaratsuba = [&](int testNum, mpf_t mpfHostResult) -> bool {

        if constexpr (sharkOperator == Operator::MultiplyKaratsubaV1 ||
            sharkOperator == Operator::MultiplyKaratsubaV2) {

            HpSharkFloat<SharkFloatParams> hostKaratsubaOutV1;
            MultiplyHelperKaratsubaV1<SharkFloatParams>(
                &xNum,
                &yNum,
                &hostKaratsubaOutV1
            );

            if (Verbose) {
                std::cout << "KaratsubaV1 result: " << hostKaratsubaOutV1.ToString() << std::endl;
                std::cout << "KaratsubaV1 hex: " << hostKaratsubaOutV1.ToHexString() << std::endl;
            }

            bool res = DiffAgainstHost<SharkFloatParams, sharkOperator>(
                testNum,
                "CustomHighPrecisionV1",
                mpfHostResult,
                hostKaratsubaOutV1);

            if (!res) {
                DebugBreak();
            };

            HpSharkFloat<SharkFloatParams> hostKaratsubaOutV2;
            MultiplyHelperKaratsubaV2<SharkFloatParams>(
                &xNum,
                &yNum,
                &hostKaratsubaOutV2
            );

            if (Verbose) {
                std::cout << "KaratsubaV2 result: " << hostKaratsubaOutV2.ToString() << std::endl;
                std::cout << "KaratsubaV2 hex: " << hostKaratsubaOutV2.ToHexString() << std::endl;
            }

            res &= DiffAgainstHost<SharkFloatParams, sharkOperator>(
                testNum,
                "CustomHighPrecisionV2",
                mpfHostResult,
                hostKaratsubaOutV2);

            if (!res) {
                DebugBreak();
            };

            return res;
        } else if constexpr (sharkOperator == Operator::Add) {
            (void)testNum;
            (void)mpfHostResult;
            return true;
        } else {
            (void)testNum;
            (void)mpfHostResult;
            return false;
        }
        };

    {
        // Perform the calculation on the host using MPIR
        HpSharkFloat<SharkFloatParams> gpuResult{};
        mpf_t mpfHostResult;
        mpf_init(mpfHostResult);

        if constexpr (sharkOperator == Operator::Add) {
            mpf_add(mpfHostResult, mpfX, mpfY);
        } else if constexpr (
            sharkOperator == Operator::MultiplyN2 ||
            sharkOperator == Operator::MultiplyKaratsubaV1 ||
            sharkOperator == Operator::MultiplyKaratsubaV2) {

            mpf_mul(mpfHostResult, mpfX, mpfY);
        }

        // Print host result
        if (Verbose) {
            std::cout << "\nHost result:" << std::endl;
            std::cout << "Host result: " << MpfToString<SharkFloatParams>(mpfHostResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
            std::cout << "Host hex: " << std::endl;
            std::cout << "" << MpfToHexString(mpfHostResult) << std::endl;
        }

        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };

        if constexpr (sharkOperator == Operator::Add) {
            InvokeAddKernelCorrectness<SharkFloatParams, Operator::Add>(
                ComputeAddGpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult);
        }
        else if constexpr (sharkOperator == Operator::MultiplyN2) {
            InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyN2>(
                ComputeMultiplyN2Gpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult);
        }
        else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV1) {
            InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyKaratsubaV1>(
                ComputeMultiplyKaratsubaV1Gpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult);
        }
        else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
            InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
                ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult);
        } else {
            assert(false);
        }

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (Verbose) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }

        bool testSucceeded = TestHostKaratsuba(testNum, mpfHostResult);
        if (!testSucceeded) {
            std::cout << "Custom High Precision failed" << std::endl;
        } else {
            std::cout << "Custom High Precision succeeded" << std::endl;
        }

        testSucceeded = DiffAgainstHost<SharkFloatParams, sharkOperator>(
            testNum,
            "GPU",
            mpfHostResult,
            gpuResult);
        if (!testSucceeded) {
            std::cout << "GPU High Precision failed" << std::endl;
            DebugBreak();
        } else {
            std::cout << "GPU High Precision succeeded" << std::endl;
        }

        // Clean up MPIR variables
        mpf_clear(mpfHostResult);
    }
}

template<class SharkFloatParams, Operator sharkOperator, bool IncludeSigns>
void TestBinOperatorTwoNumbersRaw(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    // If IncludeSigns is true, then call TestBinOperatorTwoNumbersRawNoSignChange with all four variants
    // using mpf_neg as needed

    if constexpr (IncludeSigns) {
        mpf_t mpfXCopy;
        mpf_t mpfYCopy;

        HpSharkFloat<SharkFloatParams> xNumCopy;
        HpSharkFloat<SharkFloatParams> yNumCopy;

        auto resetCopy = [&]() {
            mpf_set(mpfXCopy, mpfX);
            mpf_set(mpfYCopy, mpfY);

            xNumCopy.DeepCopySameDevice(xNum);
            yNumCopy.DeepCopySameDevice(yNum);
            };

        auto printTest = [&](int curTest) {
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << "Test " << curTest << std::endl;
            };

        auto negateMpfAndHp = [](mpf_t &mpfCopy, HpSharkFloat<SharkFloatParams> &numCopy) {
            mpf_neg(mpfCopy, mpfCopy);
            numCopy.Negate();
        };

        mpf_init(mpfXCopy);
        mpf_init(mpfYCopy);

        resetCopy();
        printTest(testNum);
        TestBinOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            testNum, xNumCopy, yNumCopy, mpfXCopy, mpfYCopy);
        testNum++;

        resetCopy();
        negateMpfAndHp(mpfXCopy, xNumCopy);
        printTest(testNum);
        TestBinOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            testNum, xNumCopy, yNumCopy, mpfXCopy, mpfYCopy);
        testNum++;

        resetCopy();
        negateMpfAndHp(mpfYCopy, yNumCopy);
        printTest(testNum);
        TestBinOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            testNum, xNumCopy, yNumCopy, mpfXCopy, mpfYCopy);
        testNum++;

        resetCopy();
        negateMpfAndHp(mpfXCopy, xNumCopy);
        negateMpfAndHp(mpfYCopy, yNumCopy);
        printTest(testNum);
        TestBinOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            testNum, xNumCopy, yNumCopy, mpfXCopy, mpfYCopy);
        testNum++;

        mpf_clear(mpfXCopy);
        mpf_clear(mpfYCopy);

    } else {
        TestBinOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(testNum, xNum, yNum, mpfX, mpfY);
    }

}

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    // Copy mpfX and mpfY
    mpf_t mpfXCopy;
    mpf_t mpfYCopy;
    mpf_init(mpfXCopy);
    mpf_init(mpfYCopy);

    auto curTest = [&]() {
        // Print the original input values
        if (Verbose) {
            std::cout << "Original input strings:" << std::endl;
            std::cout << "num1: " << num1 << std::endl;
            std::cout << "num2: " << num2 << std::endl;
            std::cout << "MpfX: " << MpfToString<SharkFloatParams>(mpfXCopy, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
            std::cout << "MpfY: " << MpfToString<SharkFloatParams>(mpfYCopy, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
            std::cout << "operator: " << OperatorToString<sharkOperator>() << std::endl;
        }

        // Convert the input values to HpSharkFloat<SharkFloatParams> representations
        std::unique_ptr<HpSharkFloat<SharkFloatParams>> xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        std::unique_ptr<HpSharkFloat<SharkFloatParams>> yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        MpfToHpGpu(mpfXCopy, *xNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
        MpfToHpGpu(mpfYCopy, *yNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);

        TestBinOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, false>(
            testNum, *xNum, *yNum, mpfXCopy, mpfYCopy);

        testNum++;
    };

    auto resetCopy = [&]() {
        mpf_set(mpfXCopy, mpfX);
        mpf_set(mpfYCopy, mpfY);
    };

    auto printTest = [&](int curTest) {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Test " << curTest << std::endl;
    };

    // All four variations of + and - tests
    {
        printTest(testNum);
        resetCopy();
        curTest();
    }

    {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfXCopy, mpfXCopy);
        curTest();
    }

    {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfYCopy, mpfYCopy);
        curTest();
    }

    {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfXCopy, mpfXCopy);
        mpf_neg(mpfYCopy, mpfYCopy);
    }

    mpf_clear(mpfXCopy);
    mpf_clear(mpfYCopy);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(
    int testNum,
    const char *num1,
    const char *num2) {

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << testNum << std::endl;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);  // Set precision for MPIR floating point

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

    TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(testNum, num1, num2, mpfX, mpfY);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers(int testNum, std::vector<uint32_t> &digits1, std::vector<uint32_t> &digits2) {
    mpf_t x, y;
    mpf_init(x);
    mpf_init(y);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << testNum << std::endl;

    auto strLargeX = Uint32ToMpf<SharkFloatParams>(digits1.data(), SharkFloatParams::NumUint32 / 2, x);
    auto strLargeY = Uint32ToMpf<SharkFloatParams>(digits2.data(), SharkFloatParams::NumUint32 / 2, y);
    TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(testNum, strLargeX.c_str(), strLargeY.c_str(), x, y);

    mpf_clear(x);
    mpf_clear(y);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum) {

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_init(mpfX);
    mpf_init(mpfY);
    HpGpuToMpf(xNum, mpfX);
    HpGpuToMpf(yNum, mpfY);

    TestBinOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, true>(
        testNum, xNum, yNum, mpfX, mpfY);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers1(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::NumUint32);
    testData[testData.size() - 1] = 0x80000000;

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, testData, testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers2(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::NumUint32);
    testData[testData.size() - 1] = 0xC0000000;

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, testData, testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers3(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::NumUint32);
    testData[testData.size() - 1] = 0xFFFFFFFF;

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, testData, testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbersHelper(
    int testNum,
    std::vector<uint32_t> testData1,
    std::vector<uint32_t> testData2) {

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << testNum << std::endl;

    std::vector<uint32_t> testData1Copy;
    testData1Copy = testData1;
    testData1Copy.resize(SharkFloatParams::NumUint32);

    std::vector<uint32_t> testData2Copy;
    testData2Copy = testData2;
    testData2Copy.resize(SharkFloatParams::NumUint32);

    std::unique_ptr<HpSharkFloat<SharkFloatParams>> xNum{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData1Copy.data(), 0, false) };
    std::unique_ptr<HpSharkFloat<SharkFloatParams>> yNum{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData2Copy.data(), 0, false) };

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, *xNum, *yNum);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers4(int testNum) {
    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xF26D37FC, 0xA96025CE, 0xB03FC716, 0x1DF7182B, 0xCCBD69BD, 0x40C0F80C, 0xFAA0222E, 0xD1FDA456 },
        std::vector<uint32_t>{ 0x8BBCDF3, 0x4C3E7ACB, 0x6691A71D, 0xDFE03842, 0x3FADCA11, 0x4058BC9E, 0xF30FD7DE, 0xAA6CA582 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers5(int testNum) {
    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers6(int testNum) {
    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers7(int testNum) {
    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers8(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers9(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x10 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers10(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0, 0x2, 0x3 },
        std::vector<uint32_t>{ 0, 0, 0, 0x5, 0x7 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers11(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0x2, 0, 0, 0, 0, 0x3 },
        std::vector<uint32_t>{ 0, 0x5, 0, 0, 0, 0, 0x7 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers12(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0xf });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers13(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x11 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers14(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x10 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers15(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000000, 0, 0xFFFFFFF1, 0x00000008, 0x00000000, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0x00000000, 0, 0x00000000, 0x0000000D, 0x00000000, 0xFFFFFFF6, 0x0000000A, 0x00000003 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers16(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x0000000C, 0xFFFFFFF0, 0x00000000, 0xFFFFFFFC, 0x00000000, 0x0000000D, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0xFFFFFFFD, 0xFFFFFFEF, 0xFFFFFFEF, 0xFFFFFFF4, 0x00000000, 0x7A6650D9, 0x00000000, 0x00000000 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers17(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF3, 0xFFFFFFF9, 0x00000004 },
        std::vector<uint32_t>{ 0x0000000E, 0x00000000, 0x00000000, 0xFFFFFFF2, 0x00000003, 0x00000000, 0xFFFFFFFF, 0x00000000 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers18(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000001, 0xFFFFFFFF, 0xFFFFFFFC, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFC, 0xE8CFC461, 0xFFFFFFF9 },
        std::vector<uint32_t>{ 0xFFFFFFF8, 0xD446522A, 0xFFFFFFFF, 0x00000010, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers19(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x685940F0, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x5008CECF, 0x2A4D4784, 0x0000000D, 0x00000006, 0x00000000, 0xFFFFFFFF, 0x00000000 });
}

template<class SharkFloatParams, Operator sharkOperator>
bool TestAllBinaryOp(int testBase) {
    constexpr bool includeSet1 = true;
    constexpr bool includeSet2 = true;
    constexpr bool includeSet3 = true;
    constexpr bool includeSet4 = true;
    constexpr bool includeSet5 = true;
    constexpr bool includeSet6 = true;
    constexpr bool includeSet10 = true;
    constexpr bool includeSet11 = false;

    // 200s is multiply
    // 400s is add
    
    if constexpr (includeSet1) {
        const auto set = testBase + 100;
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "1", "2");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "4294967295", "1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "4294967296", "1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "4294967295", "4294967296");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "4294967296", "-1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 60, "18446744073709551615", "1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 70, "0", "0.1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 80, "0.1", "0");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 90, "0", "0");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 100, "0.1", "0.1");
    }

    if constexpr (includeSet2) {
        const auto set = testBase + 300;
        TestAddSpecialNumbers1<SharkFloatParams, sharkOperator>(set + 10);
        TestAddSpecialNumbers2<SharkFloatParams, sharkOperator>(set + 20);
        TestAddSpecialNumbers3<SharkFloatParams, sharkOperator>(set + 30);
        TestAddSpecialNumbers4<SharkFloatParams, sharkOperator>(set + 40);
        TestAddSpecialNumbers5<SharkFloatParams, sharkOperator>(set + 50);
        TestAddSpecialNumbers6<SharkFloatParams, sharkOperator>(set + 60);
        TestAddSpecialNumbers7<SharkFloatParams, sharkOperator>(set + 70);
        TestAddSpecialNumbers8<SharkFloatParams, sharkOperator>(set + 80);
        TestAddSpecialNumbers9<SharkFloatParams, sharkOperator>(set + 90);
        TestAddSpecialNumbers10<SharkFloatParams, sharkOperator>(set + 100);
        TestAddSpecialNumbers11<SharkFloatParams, sharkOperator>(set + 110);
        TestAddSpecialNumbers12<SharkFloatParams, sharkOperator>(set + 120);
        TestAddSpecialNumbers13<SharkFloatParams, sharkOperator>(set + 130);
        TestAddSpecialNumbers14<SharkFloatParams, sharkOperator>(set + 140);
        TestAddSpecialNumbers15<SharkFloatParams, sharkOperator>(set + 150);
        TestAddSpecialNumbers16<SharkFloatParams, sharkOperator>(set + 160);
        TestAddSpecialNumbers17<SharkFloatParams, sharkOperator>(set + 170);
        TestAddSpecialNumbers18<SharkFloatParams, sharkOperator>(set + 180);
        TestAddSpecialNumbers19<SharkFloatParams, sharkOperator>(set + 190);
    }

    if constexpr (includeSet3) {
        const auto set = testBase + 600;
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "2", "0.1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "0.2", "0.1");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "0.5", "1.2");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "0.6", "1.3");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "0.7", "1.4");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 60, "0.1", "1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 70, "0.123124561464451654461", "1.2395123123127298375982735");
    }

    if constexpr (includeSet4) {
        const auto set = testBase + 700;
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "-0.5", "1.2");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "-0.6", "1.3");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "-0.7", "1.4");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "-0.1", "1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "-0.123124561464451654461", "1.2395123123127298375982735");
    }

    if constexpr (includeSet5) {
        const auto set = testBase + 800;
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "-0.51", "-1.29");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "-0.61", "-1.39");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "-0.71", "-1.49");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "-0.11", "-1.99999999999999999999999999999");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "-0.123124561464451654461", "-1.2395123123127298375982735");
    }

    if constexpr (includeSet6) {
        const auto set = testBase + 900;
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "0.5265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "0.2999999999965542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "0.1265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866391730473289107302178039999999999999271839216");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "0.0265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "0.00000000000000000265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
    }

    if constexpr (includeSet10) {
        const auto set10 = testBase + 1000;
        for (auto i = 0; i < 1000; i += 10) {
            std::unique_ptr<HpSharkFloat<SharkFloatParams>> x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            std::unique_ptr<HpSharkFloat<SharkFloatParams>> y = std::make_unique<HpSharkFloat<SharkFloatParams>>();

            x->GenerateRandomNumber();
            y->GenerateRandomNumber();

            if (Verbose) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->IsNegative << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->IsNegative << std::endl;
            }
            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set10 + i, x_str.c_str(), y_str.c_str());
        }
    }

    if constexpr (includeSet11) {
        for (;;) {
            std::unique_ptr<HpSharkFloat<SharkFloatParams>> x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            std::unique_ptr<HpSharkFloat<SharkFloatParams>> y = std::make_unique<HpSharkFloat<SharkFloatParams>>();

            x->GenerateRandomNumber();
            y->GenerateRandomNumber();

            if (Verbose) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->IsNegative << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->IsNegative << std::endl;
            }
            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(0, x_str.c_str(), y_str.c_str());
        }
    }

    return Tests.CheckAllTestsPassed();
}

template<class SharkFloatParams, Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase) {
    TestPerf<SharkFloatParams, sharkOperator>(testBase + 1);
    return Tests.CheckAllTestsPassed();
}

// Explicitly instantiate TestAllBinaryOp
#define ExplicitlyInstantiate(SharkFloatParams) \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::Add>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyN2>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV1>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase); \
    template bool TestBinaryOperatorPerf<SharkFloatParams, Operator::Add>(int testBase); \
    template bool TestBinaryOperatorPerf<SharkFloatParams, Operator::MultiplyN2>(int testBase); \
    template bool TestBinaryOperatorPerf<SharkFloatParams, Operator::MultiplyKaratsubaV1>(int testBase); \
    template bool TestBinaryOperatorPerf<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase);


ExplicitlyInstantiate(Test4x4SharkParams);
ExplicitlyInstantiate(Test4x2SharkParams);
ExplicitlyInstantiate(Test8x1SharkParams);
ExplicitlyInstantiate(Test8x8SharkParams);
ExplicitlyInstantiate(Test128x64SharkParams);