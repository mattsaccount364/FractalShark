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
#include <assert.h>

static TestTracker Tests;
static constexpr int NUM_ITER = 1000;

__device__ void AddHelper(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out) {

    const int idx = threadIdx.x;
    const int numDigits = HpGpu::NumUint32;

    // Shared memory for intermediate results
    __shared__ uint32_t tempSum[HpGpu::NumUint32];
    __shared__ uint32_t carryBits[HpGpu::NumUint32]; // No need for +1
    __shared__ bool isAddition;
    __shared__ bool AIsBiggerExponent;
    __shared__ bool AIsBiggerMagnitude;
    __shared__ int32_t outExponent;
    __shared__ bool resultIsZero;

    // Step 1: Exponent Alignment
    if (idx == 0) {
        int32_t expDiff = A->Exponent - B->Exponent;
        AIsBiggerExponent = expDiff >= 0;
        outExponent = AIsBiggerExponent ? A->Exponent : B->Exponent;
        isAddition = (A->IsNegative == B->IsNegative);
    }
    __syncthreads();

    // Shift digits to align exponents
    __shared__ uint32_t alignedA[HpGpu::NumUint32];
    __shared__ uint32_t alignedB[HpGpu::NumUint32];

    // Initialize aligned digits to zero
    if (idx < numDigits) {
        alignedA[idx] = 0;
        alignedB[idx] = 0;
    }
    __syncthreads();

    // Perform shifting
    int shiftBits = abs(A->Exponent - B->Exponent);
    int shiftWords = shiftBits / 32;
    int shiftBitsMod = shiftBits % 32;

    if (idx < numDigits) {
        if (AIsBiggerExponent) {
            // Shift B right
            if (idx + shiftWords < numDigits) {
                uint32_t lower = B->Digits[idx + shiftWords];
                uint32_t upper = (idx + shiftWords + 1 < numDigits) ? B->Digits[idx + shiftWords + 1] : 0;
                alignedB[idx] = (shiftBitsMod == 0) ? lower : (lower >> shiftBitsMod) | (upper << (32 - shiftBitsMod));
            }
            alignedA[idx] = A->Digits[idx];
        } else {
            // Shift A right
            if (idx + shiftWords < numDigits) {
                uint32_t lower = A->Digits[idx + shiftWords];
                uint32_t upper = (idx + shiftWords + 1 < numDigits) ? A->Digits[idx + shiftWords + 1] : 0;
                alignedA[idx] = (shiftBitsMod == 0) ? lower : (lower >> shiftBitsMod) | (upper << (32 - shiftBitsMod));
            }
            alignedB[idx] = B->Digits[idx];
        }
    }
    __syncthreads();

    // Step 2: Compare Magnitudes if necessary
    resultIsZero = false;
    if (!isAddition && idx == 0) {
        // For subtraction, compare magnitudes
        AIsBiggerMagnitude = false;
        resultIsZero = true;
        for (int i = numDigits - 1; i >= 0; --i) {
            if (alignedA[i] > alignedB[i]) {
                AIsBiggerMagnitude = true;
                resultIsZero = false;
                break;
            } else if (alignedA[i] < alignedB[i]) {
                AIsBiggerMagnitude = false;
                resultIsZero = false;
                break;
            }
            // If equal, continue to next digit
        }
        if (resultIsZero) {
            // The numbers are equal in magnitude
            for (int i = 0; i < numDigits; ++i) {
                Out->Digits[i] = 0;
            }
            Out->IsNegative = false;
            Out->Exponent = 0;
        }
    }
    __syncthreads();

    if (!resultIsZero) {
        // Step 3: Perform Addition or Subtraction
        uint64_t fullSum = 0;

        if (isAddition) {
            // Perform addition
            fullSum = (uint64_t)alignedA[idx] + (uint64_t)alignedB[idx];
            tempSum[idx] = (uint32_t)(fullSum & 0xFFFFFFFF);
            carryBits[idx] = (uint32_t)(fullSum >> 32);
        } else {
            // Perform subtraction: larger magnitude minus smaller magnitude
            uint32_t operandLarge = AIsBiggerMagnitude ? alignedA[idx] : alignedB[idx];
            uint32_t operandSmall = AIsBiggerMagnitude ? alignedB[idx] : alignedA[idx];

            if (operandLarge >= operandSmall) {
                fullSum = (uint64_t)operandLarge - (uint64_t)operandSmall;
                tempSum[idx] = (uint32_t)(fullSum & 0xFFFFFFFF);
                carryBits[idx] = 0;
            } else {
                // Borrow occurs
                fullSum = ((uint64_t)1 << 32) + (uint64_t)operandLarge - (uint64_t)operandSmall;
                tempSum[idx] = (uint32_t)(fullSum & 0xFFFFFFFF);
                carryBits[idx] = 1; // Indicate borrow
            }
        }
        __syncthreads();

        // Step 4: Carry/Borrow Propagation (Sequential in thread 0)
        __shared__ uint32_t carryIn;
        if (idx == 0) {
            carryIn = 0;

            for (int i = 0; i < numDigits; ++i) {
                if (isAddition) {
                    uint64_t sumWithCarry = (uint64_t)tempSum[i] + carryIn;
                    Out->Digits[i] = (uint32_t)(sumWithCarry & 0xFFFFFFFF);
                    carryIn = (uint32_t)(sumWithCarry >> 32) + carryBits[i];
                } else {
                    // Subtraction
                    if (carryIn) {
                        if (tempSum[i] > 0) {
                            Out->Digits[i] = tempSum[i] - 1;
                            carryIn = carryBits[i];
                        } else {
                            Out->Digits[i] = 0xFFFFFFFF;
                            carryIn = 1;
                        }
                    } else {
                        Out->Digits[i] = tempSum[i];
                        carryIn = carryBits[i];
                    }
                }
            }

            // Handle carry out for addition
            if (isAddition && carryIn > 0) {
                // Shift all digits right by 1 to accommodate the carry
                uint32_t carryOut = carryIn; // carryIn is 1 here

                for (int i = numDigits - 1; i >= 0; --i) {
                    uint64_t shifted = ((uint64_t)Out->Digits[i] >> 1) | ((uint64_t)carryOut << 31);
                    carryOut = (Out->Digits[i] & 0x1) ? 1 : 0; // Extract the bit that was shifted out
                    Out->Digits[i] = (uint32_t)(shifted & 0xFFFFFFFF);
                }

                // Increment the exponent due to normalization (since we effectively divided by 2)
                outExponent += 1;
            }

            // Step 5: Set Exponent and Sign
            Out->Exponent = outExponent;

            if (isAddition) {
                Out->IsNegative = A->IsNegative;
            } else {
                // The sign is determined by the operand with the larger magnitude
                Out->IsNegative = AIsBiggerMagnitude ? A->IsNegative : B->IsNegative;

                // Handle negative result if subtraction resulted in borrow at the highest digit
                if (carryIn > 0) {
                    // Result is negative and needs to be two's complemented
                    for (int i = 0; i < numDigits; ++i) {
                        Out->Digits[i] = ~Out->Digits[i];
                    }
                    // Add 1 to complete two's complement
                    uint32_t carry = 1;
                    for (int i = 0; i < numDigits; ++i) {
                        uint64_t sum = (uint64_t)Out->Digits[i] + carry;
                        Out->Digits[i] = (uint32_t)(sum & 0xFFFFFFFF);
                        carry = (uint32_t)(sum >> 32);
                        if (carry == 0) break;
                    }
                    // Flip the sign
                    Out->IsNegative = !Out->IsNegative;
                }
            }
        }
    }
    __syncthreads();
}

__global__ void AddKernel(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out) {

    AddHelper(A, B, Out);
}

__global__ void AddKernelTestLoop(
    HpGpu *A,
    HpGpu *Out) {

    for (int i = 0; i < NUM_ITER; ++i) {
        AddHelper(A, Out, Out);
    }
}

__global__ void multiply_high_precision_kernel(
    HpGpu *A,
    HpGpu *B,
    HpGpu *Out) {
    const int idx = threadIdx.x;
    __shared__ uint32_t tempProduct[HpGpu::NumUint32 * 2];

    // Initialize tempProduct to zero
    if (idx < HpGpu::NumUint32 * 2) {
        tempProduct[idx] = 0;
    }

    __syncthreads();

    // Each thread computes partial products
    for (int i = 0; i < HpGpu::NumUint32; i++) {
        if (idx < HpGpu::NumUint32) {
            uint64_t product = (uint64_t)A->Digits[idx] * (uint64_t)B->Digits[i];
            int pos = idx + i;

            // Atomic addition to handle concurrent writes
            atomicAdd(&tempProduct[pos], (uint32_t)(product & 0xFFFFFFFF));
            atomicAdd(&tempProduct[pos + 1], (uint32_t)(product >> 32));
        }
    }

    __syncthreads();

    // Perform carry propagation
    uint32_t carry = 0;
    if (idx < HpGpu::NumUint32 * 2) {
        uint64_t sum = (uint64_t)tempProduct[idx] + carry;
        tempProduct[idx] = (uint32_t)(sum & 0xFFFFFFFF);
        carry = (uint32_t)(sum >> 32);
    }

    __syncthreads();

    // Write the result to Out
    if (idx < HpGpu::NumUint32) {
        Out->Digits[idx] = tempProduct[idx];
    }

    // Adjust exponent and sign
    if (idx == 0) {
        Out->Exponent = A->Exponent + B->Exponent;
        Out->IsNegative = A->IsNegative ^ B->IsNegative;
    }
}

void ComputeAddGpu(
    HpGpu *x, // device
    HpGpu *a, // device
    HpGpu *temp_result // device
    ) {

    // Length of x, y, a, b are all the same.

    auto gridSize = 1;
    dim3 block(HpGpu::NumUint32, 1, 1);

    // allocate temporaries on device
    AddKernel << <gridSize, block >> > (
        x,
        a,
        temp_result);

    cudaDeviceSynchronize();
}

void ComputeAddGpuTestLoop(
    HpGpu *x, // device
    HpGpu *temp_result // device
) {

    // Length of x, y, a, b are all the same.

    auto gridSize = 1;
    dim3 block(HpGpu::NumUint32, 1, 1);

    // allocate temporaries on device
    AddKernelTestLoop << <gridSize, block >> > (
        x,
        temp_result);

    cudaDeviceSynchronize();
}

void DiffAgainstHost(
    bool verbose,
    int testNum,
    const mpf_t mpfHostResult,
    const HpGpu &gpuResult) {

    if (verbose) {
        std::cout << "\nGPU result: " << std::endl;
        std::cout << gpuResult.ToString() << std::endl;
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
    if (verbose) {
        std::cout << "\nConverted GPU result:" << std::endl;
        std::cout << MpfToString(mpfXGpuResult, HpGpu::DefaultPrecBits) << std::endl;
    }

    // Print the differences
    std::cout << "\nDifference between host and GPU results:" << std::endl;
    std::cout << MpfToString(mpf_diff_abs, HpGpu::DefaultPrecBits) << std::endl;

    // If absolute delta is greater than 1e-300, the test is considered failed
    if (mpf_cmp_d(mpf_diff_abs, 1e-30) > 0) {
        Tests.MarkFailed(testNum, mpf_diff_abs);
    }

    mpf_clear(mpfDiff);
    mpf_clear(mpf_diff_abs);
    mpf_clear(mpfXGpuResult);
}

void TestAddTwoNumbersPerf(
    bool verbose,
    int testNum,
    const char *num1,
    const mpf_t mpfX) {

    // Print the original input values
    if (verbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: " << MpfToString(mpfX, HpGpu::DefaultPrecBits) << std::endl;
    }

    HpGpu xNum{};
    MpfToHpGpu(mpfX, xNum, HpGpu::DefaultPrecBits);
    if (verbose) {
        std::cout << "\nConverted HpGpu representations:" << std::endl;
        std::cout << "X: " << xNum.ToString() << std::endl;
    }

    // Perform the calculation on the host using MPIR
    mpf_t mpfHostResult;
    mpf_init(mpfHostResult);

    {
        BenchmarkTimer hostTimer;
        ScopedBenchmarkStopper hostStopper{ hostTimer };

        for (size_t i = 0; i < NUM_ITER; ++i) {
            mpf_add(mpfHostResult, mpfHostResult, mpfX);
        }

        hostTimer.StopTimer();

        if (verbose) {
            std::cout << "Host iter time: " << hostTimer.GetDeltaInMs() << " ms" << std::endl;
        }
    }

    HpGpu gpuResult2{};
    {
        // Perform the calculation on the GPU
        HpGpu *xGpu;
        cudaMalloc(&xGpu, sizeof(HpGpu));
        cudaMemcpy(xGpu, &xNum, sizeof(HpGpu), cudaMemcpyHostToDevice);

        HpGpu *internalGpuResult2;
        cudaMalloc(&internalGpuResult2, sizeof(HpGpu));
        cudaMemset(internalGpuResult2, 0, sizeof(HpGpu));

        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpuTestLoop(
            xGpu,
            internalGpuResult2);

        cudaMemcpy(&gpuResult2, internalGpuResult2, sizeof(HpGpu), cudaMemcpyDeviceToHost);

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (verbose) {
            std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }

        cudaFree(internalGpuResult2);
        cudaFree(xGpu);
    }

    DiffAgainstHost(verbose, testNum, mpfHostResult, gpuResult2);

    // Clean up MPIR variables
    mpf_clear(mpfHostResult);
}

void TestAddTwoNumbersPerf(
    bool verbose,
    int testNum,
    const char *num1) {

    mpf_set_default_prec(HpGpu::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpfX;
    mpf_init(mpfX);

    auto res = mpf_set_str(mpfX, num1, 10);
    if (res == -1) {
        std::cout << "Error setting mpfX" << std::endl;
    }

    TestAddTwoNumbersPerf(verbose, testNum, num1, mpfX);

    mpf_clear(mpfX);
}

void TestAddTwoNumbers(
    bool verbose,
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t mpfX,
    const mpf_t mpfY) {

    // Print the original input values
    if (verbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "X: " << MpfToString(mpfX, HpGpu::DefaultPrecBits) << std::endl;
        std::cout << "Y: " << MpfToString(mpfY, HpGpu::DefaultPrecBits) << std::endl;
    }

    // Convert the input values to HpGpu representations
    HpGpu xNum{};
    HpGpu yNum{};
    MpfToHpGpu(mpfX, xNum, HpGpu::DefaultPrecBits);
    MpfToHpGpu(mpfY, yNum, HpGpu::DefaultPrecBits);

    if (verbose) {
        std::cout << "\nConverted HpGpu representations:" << std::endl;
        std::cout << "X: " << xNum.ToString() << std::endl;
        std::cout << "Y: " << yNum.ToString() << std::endl;
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
        mpf_add(mpfHostResult, mpfX, mpfY);

        // Print host result
        if (verbose) {
            std::cout << "\nHost result:" << std::endl;
            std::cout << "Host result: " << MpfToString(mpfHostResult, HpGpu::DefaultPrecBits) << std::endl;
        }

        HpGpu *internalGpuResult;
        cudaMalloc(&internalGpuResult, sizeof(HpGpu));

        BenchmarkTimer timer;
        ScopedBenchmarkStopper stopper{ timer };
        ComputeAddGpu(
            xGpu,
            yGpu,
            internalGpuResult);

        cudaMemcpy(&gpuResult, internalGpuResult, sizeof(HpGpu), cudaMemcpyDeviceToHost);

        timer.StopTimer();
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (verbose) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }

        cudaFree(internalGpuResult);

        DiffAgainstHost(verbose, testNum, mpfHostResult, gpuResult);

        // Clean up MPIR variables
        mpf_clear(mpfHostResult);
    }

    cudaFree(xGpu);
    cudaFree(yGpu);
}

void TestAddTwoNumbers(
    bool verbose,
    int testNum,
    const char *num1,
    const char *num2) {

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

    TestAddTwoNumbers(verbose, testNum, num1, num2, mpfX, mpfY);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

void TestAddSpecialNumbers(bool verbose, int testNum) {
    mpf_t x, y;
    mpf_init(x);
    mpf_init(y);

    std::vector<uint32_t> testData;
    for (size_t i = 0; i < HpGpu::NumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == HpGpu::NumUint32);
    testData[testData.size() - 1] = 0x80000000;

    auto strLargeX = Uint32ToMpf(testData.data(), HpGpu::NumUint32 / 2, x);
    auto strLargeY = Uint32ToMpf(testData.data(), HpGpu::NumUint32 / 2, y);
    TestAddTwoNumbers(verbose, testNum, strLargeX.c_str(), strLargeY.c_str(), x, y);

    mpf_clear(x);
    mpf_clear(y);
}

void TestAllAdd() {
    constexpr bool verbose = true;
    const auto set1 = 10;
    TestAddTwoNumbers(verbose, set1 + 1, "1", "2");
    TestAddTwoNumbers(verbose, set1 + 2, "0.2", "0.1");
    TestAddTwoNumbers(verbose, set1 + 3, "0.5", "1.2");
    TestAddTwoNumbers(verbose, set1 + 4, "0.6", "1.3");
    TestAddTwoNumbers(verbose, set1 + 5, "0.7", "1.4");
    TestAddTwoNumbers(verbose, set1 + 6, "0.1", "1.99999999999999999999999999999");
    TestAddTwoNumbers(verbose, set1 + 7, "0.123124561464451654461", "1.2395123123127298375982735");

    const auto set2 = 20;
    TestAddTwoNumbers(verbose, set2 + 1, "-0.5", "1.2");
    TestAddTwoNumbers(verbose, set2 + 2, "-0.6", "1.3");
    TestAddTwoNumbers(verbose, set2 + 3, "-0.7", "1.4");
    TestAddTwoNumbers(verbose, set2 + 4, "-0.1", "1.99999999999999999999999999999");
    TestAddTwoNumbers(verbose, set2 + 5, "-0.123124561464451654461", "1.2395123123127298375982735");

    const auto set3 = 30;
    TestAddTwoNumbers(verbose, set3 + 1, "-0.5", "-1.2");
    TestAddTwoNumbers(verbose, set3 + 2, "-0.6", "-1.3");
    TestAddTwoNumbers(verbose, set3 + 3, "-0.7", "-1.4");
    TestAddTwoNumbers(verbose, set3 + 4, "-0.1", "-1.99999999999999999999999999999");
    TestAddTwoNumbers(verbose, set3 + 5, "-0.123124561464451654461", "-1.2395123123127298375982735");

    const auto set4 = 40;
    TestAddTwoNumbers(verbose, set4 + 1, "0.5265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
    TestAddTwoNumbers(verbose, set4 + 2, "0.2999999999965542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
    TestAddTwoNumbers(verbose, set4 + 3, "0.1265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866391730473289107302178039999999999999271839216");
    TestAddTwoNumbers(verbose, set4 + 4, "0.0265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");
    TestAddTwoNumbers(verbose, set4 + 5, "0.00000000000000000265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216");

    const auto set5 = 50;
    TestAddSpecialNumbers(verbose, set5 + 1);

    const auto set10 = 100;
    for (auto i = 0; i < 100; i++) {
        HpGpu x, y;
        x.GenerateRandomNumber();
        y.GenerateRandomNumber();

        std::cout << "x.Exponent: " << x.Exponent << ", neg: " << x.IsNegative << std::endl;
        std::cout << "y.Exponent: " << y.Exponent << ", neg: " << y.IsNegative << std::endl;
        const std::string x_str = x.ToString();
        const std::string y_str = y.ToString();
        TestAddTwoNumbers(true, set10 + i, x_str.c_str(), y_str.c_str());
    }

    Tests.CheckAllTestsPassed();
}

void TestAddPerf() {
    constexpr bool verbose = true;
    const auto set20 = 200;
    TestAddTwoNumbersPerf(verbose, set20 + 1, ".1");
    Tests.CheckAllTestsPassed();
}