#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.h"
#include "ReferenceKaratsuba.h"
#include "DebugChecksumHost.h"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <assert.h>

#include "Add.cuh"
#include "Multiply.cuh"

#define NOMINMAX
#include <windows.h>

static TestTracker Tests;

// Returns false if the test fails, true otherwise
template<class SharkFloatParams, Operator sharkOperator>
bool DiffAgainstHost(
    int testNum,
    std::string hostCustomOrGpu,
    const mpf_t mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult) {

    if constexpr (SharkFloatParams::HostVerbose) {
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
    if constexpr (SharkFloatParams::HostVerbose) {
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

    // Init a zero mpf:
    mpf_t mpfZero;
    mpf_init(mpfZero);
    mpf_set_ui(mpfZero, 0);

    if (mpf_cmp(mpfHostResult, mpfZero) != 0) {
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
            if constexpr (SharkFloatParams::HostVerbose) {
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
            if constexpr (SharkFloatParams::HostVerbose) {
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

    mpf_clear(mpfZero);
    mpf_clear(mpfDiff);
    mpf_clear(mpfDiffAbs);
    mpf_clear(mpfXGpuResult);

    return testSucceeded;
}

template<class SharkFloatParams, Operator sharkOperator>
void TestPerf(
    int testNum,
    const char *num1,
    const char *num2,
    const mpf_t mpfX,
    const mpf_t mpfY,
    uint64_t numIters) {

    // Print the original input values
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: " << MpfToString<SharkFloatParams>(mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "Y: " << MpfToString<SharkFloatParams>(mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
    }

    auto desc = SharkFloatParams::GetDescription();
    std::cout << "\nTest " << testNum << ": " << OperatorToString<sharkOperator>() << " " << desc << std::endl;

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto resultNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    MpfToHpGpu(mpfX, *xNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    MpfToHpGpu(mpfY, *yNum, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    if constexpr (SharkFloatParams::HostVerbose) {
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

        for (int i = 0; i < numIters; ++i) {
            if constexpr (sharkOperator == Operator::Add) {
                mpf_add(mpfHostResult, mpfX, mpfY);
            } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
                mpf_mul(mpfHostResult, mpfX, mpfY);
            }
        }

        hostTimer.StopTimer();

        std::cout << "Host iter time: " << hostTimer.GetDeltaInMs() << " ms" << std::endl;
    }

    auto gpuResult2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    {
        BenchmarkTimer timer;

        if constexpr (sharkOperator == Operator::Add) {
            InvokeAddKernel<SharkFloatParams>(
                timer,
                ComputeAddGpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2,
                numIters);
        } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
            InvokeMultiplyKernel<SharkFloatParams>(
                timer,
                ComputeMultiplyKaratsubaV2GpuTestLoop<SharkFloatParams>,
                *xNum,
                *yNum,
                *gpuResult2,
                numIters);
        }

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
    int testNum,
    uint64_t numIters) {

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

    TestPerf<SharkFloatParams, sharkOperator>(testNum, num1.c_str(), num2.c_str(), mpfX, mpfY, numIters);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbersRawNoSignChange(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    const mpf_t &mpfX,
    const mpf_t &mpfY) {

    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;
        std::cout << "X: " << xNum.ToString() << std::endl;
        std::cout << "X hex: " << xNum.ToHexString() << std::endl;
        std::cout << "Y: " << yNum.ToString() << std::endl;
        std::cout << "Y hex: " << yNum.ToHexString() << std::endl;
    }

    auto TestHostKaratsuba = [&](
        int testNum,
        mpf_t mpfHostResult,
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates) -> bool {

        if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {

            HpSharkFloat<SharkFloatParams> hostKaratsubaOutV1;
            MultiplyHelperKaratsubaV1<SharkFloatParams>(
                &xNum,
                &yNum,
                &hostKaratsubaOutV1
            );

            if constexpr (SharkFloatParams::HostVerbose) {
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
                &hostKaratsubaOutV2,
                debugStates
            );

            if constexpr (SharkFloatParams::HostVerbose) {
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

    static constexpr bool TestGpu = true;

    // Perform the calculation on the host using MPIR
    HpSharkFloat<SharkFloatParams> gpuResult{};
    mpf_t mpfHostResult;
    mpf_init(mpfHostResult);

    if constexpr (sharkOperator == Operator::Add) {
        mpf_add(mpfHostResult, mpfX, mpfY);
    } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
        mpf_mul(mpfHostResult, mpfX, mpfY);
    }

    // Print host result
    if constexpr (SharkFloatParams::HostVerbose) {
        std::cout << "\nHost result:" << std::endl;
        std::cout << "Host result: " << MpfToString<SharkFloatParams>(mpfHostResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Host hex: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResult) << std::endl;
    }

    std::vector<DebugStateRaw> debugStatesCuda{};
    if constexpr (TestGpu) {
        BenchmarkTimer timer;

        if constexpr (sharkOperator == Operator::Add) {
            InvokeAddKernelCorrectness<SharkFloatParams, Operator::Add>(
                timer,
                ComputeAddGpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult);
        } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
            InvokeMultiplyKernelCorrectness<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
                timer,
                ComputeMultiplyKaratsubaV2Gpu<SharkFloatParams>,
                xNum,
                yNum,
                gpuResult,
                &debugStatesCuda);
        } else {
            assert(false);
        }

        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if constexpr (SharkFloatParams::HostVerbose) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }
    }

    std::vector<DebugStateHost<SharkFloatParams>> debugResultsHost;
    bool testSucceeded = TestHostKaratsuba(testNum, mpfHostResult, debugResultsHost);
    if (!testSucceeded) {
        std::cout << "Custom High Precision failed" << std::endl;
    } else {
        std::cout << "Custom High Precision succeeded" << std::endl;
    }

    // Compare debugResultsCuda against debugResultsHost
    bool ChecksumFailure = false;
    if constexpr (TestGpu && SharkDebugChecksums) {
        assert (debugResultsHost.size() <= debugStatesCuda.size());

        // Note that the hosts results should be exactly the right size, whereas
        // the CUDA results may be larger due to the way the kernel is written.
        for (size_t i = 0; i < debugResultsHost.size(); ++i) {
            const auto &host = debugResultsHost[i];
            const auto &cuda = debugStatesCuda[i];

            const auto maxHostArraySize = std::max(host.ArrayToChecksum32.size(), host.ArrayToChecksum64.size());

            if (host.Checksum != cuda.Checksum ||
                host.ChecksumPurpose != cuda.ChecksumPurpose ||
                host.CallIndex != cuda.CallIndex ||
                maxHostArraySize != cuda.ArraySize) {

                std::cerr << "======================================" << std::endl;
                std::cerr << "Error: Checksum mismatch" << std::endl;
                std::cerr << "GPU:" << std::endl;

                // Print all fields of cuda:
                std::cerr << "Block: " << cuda.Block << std::endl;
                std::cerr << "Thread: " << cuda.Thread << std::endl;
                std::cerr << "ArraySize: " << cuda.ArraySize << std::endl;

                std::cerr << "Checksum: 0x" << std::hex << cuda.Checksum << std::dec << std::endl;
                std::cerr << "ChecksumPurpose: " << static_cast<int>(cuda.ChecksumPurpose) << std::endl;

                std::cerr << "RecursionDepth: " << cuda.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << cuda.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(cuda.Convolution) << std::endl;

                // Print all fields of host
                std::cerr << std::endl;
                std::cerr << "Host reference implementation:" << std::endl;
                std::cerr << "ArrayToChecksum32: " << std::endl;
                for (size_t j = 0; j < host.ArrayToChecksum32.size(); ++j) {
                    std::cerr << std::hex << "0x" << host.ArrayToChecksum32[j] << std::dec << " ";
                }

                std::cerr << std::endl;
                std::cerr << "ArrayToChecksum32 length: " << host.ArrayToChecksum32.size() << std::endl;

                std::cerr << "ArrayToChecksum64: " << std::endl;
                for (size_t j = 0; j < host.ArrayToChecksum64.size(); ++j) {
                    std::cerr << std::hex << "0x" << host.ArrayToChecksum64[j] << std::dec << " ";
                }

                std::cerr << std::endl;
                std::cerr << "ArrayToChecksum64 length: " << host.ArrayToChecksum64.size() << std::endl;

                std::cerr << "Checksum: 0x" << std::hex << host.Checksum << std::dec << std::endl;
                std::cerr << "ChecksumPurpose: " << static_cast<int>(host.ChecksumPurpose) << std::endl;

                std::cerr << "RecursionDepth: " << host.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << host.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(host.Convolution) << std::endl;

                ChecksumFailure = true;

                DebugBreak();
            }
        }
    }

    if constexpr (TestGpu) {
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

            if (ChecksumFailure) {
                std::cerr << "Checksum failure (debug issue), see above results" << std::endl;
                DebugBreak();
            }
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

// Win32 clear console
void ClearConsole() {
    system("cls");
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

    // Clear the console
    ClearConsole();

    auto curTest = [&]() {
        // Print the original input values
        if constexpr (SharkFloatParams::HostVerbose) {
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

    auto strLargeX = Uint32ToMpf<SharkFloatParams>(digits1.data(), SharkFloatParams::HalfLimbsRoundedUp, x);
    auto strLargeY = Uint32ToMpf<SharkFloatParams>(digits2.data(), SharkFloatParams::HalfLimbsRoundedUp, y);
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
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0x80000000;

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, testData, testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers2(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0xC0000000;

    TestAddSpecialNumbers<SharkFloatParams, sharkOperator>(testNum, testData, testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestAddSpecialNumbers3(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
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
    testData1Copy.resize(SharkFloatParams::GlobalNumUint32);

    std::vector<uint32_t> testData2Copy;
    testData2Copy = testData2;
    testData2Copy.resize(SharkFloatParams::GlobalNumUint32);

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
void TestAddSpecialNumbers20(int testNum) {

    TestAddSpecialNumbersHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFFFFFFFF, 0x556B0E43, 0x4EECA55A, 0x0000000E, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF8, 0x9B1194D6, 0xFFFFFFFF, 0x00000000, 0x13C1799F, 0x00000000, 0xC5F37A5D, 0xFFFFFFF4, 0x6FBC0EFF, 0x00000008, 0xFFFFFFFF, 0x00000000, 0xFFFFFFEF, 0xB06FA6C3, 0x0000000F, 0xFFFFFFF4, 0x00000007, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0x0503FC0B, 0xF26CA6A5, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000007, 0x00000010, 0xE640F2D9, 0x00000000, 0xFFFFFFF5, 0xFFFFFFFF, 0xFFFFFFF0, 0xFFFFFFFF, 0x00000004, 0x379A6DBB, 0xFFFFFFFF, 0x00000008, 0x00000002, 0xFFFFFFFF, 0x00000000, 0x0000000B, 0x00000000, 0xFFFFFFEF, 0xFFFFFFFF, 0x093E223D });
}

template<class SharkFloatParams, Operator sharkOperator>
bool TestAllBinaryOp(int testBase) {
    constexpr bool includeSet1 = true;
    constexpr bool includeSet2 = true;
    constexpr bool includeSet3 = true;
    constexpr bool includeSet4 = true;
    constexpr bool includeSet5 = true;
    constexpr bool includeSet6 = true;
    constexpr bool includeSet10 = false;
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
        TestAddSpecialNumbers20<SharkFloatParams, sharkOperator>(set + 200);
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
        TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "-0.61", "-1.39"); // TODO this line bad with 13,5 config?
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
        auto x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto y = std::make_unique<HpSharkFloat<SharkFloatParams>>();

        for (auto i = 0; i < 1000; i += 10) {
            if (i % 2 == 0) {
                x->GenerateRandomNumber();
                y->GenerateRandomNumber();
            } else {
                x->GenerateRandomNumber2();
                y->GenerateRandomNumber2();
            }

            if constexpr (SharkFloatParams::HostVerbose) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->IsNegative << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->IsNegative << std::endl;
            }
            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            TestBinOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set10 + i, x_str.c_str(), y_str.c_str());
        }
    }

    if constexpr (includeSet11) {
        std::unique_ptr<HpSharkFloat<SharkFloatParams>> x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        std::unique_ptr<HpSharkFloat<SharkFloatParams>> y = std::make_unique<HpSharkFloat<SharkFloatParams>>();

        for (size_t counter = 0;; counter++) {
            if (counter % 2 == 0) {
                x->GenerateRandomNumber();
                y->GenerateRandomNumber();
            } else {
                x->GenerateRandomNumber2();
                y->GenerateRandomNumber2();
            }

            if constexpr (SharkFloatParams::HostVerbose) {
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

template<Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase) {
#if (ENABLE_BASIC_CORRECTNESS == 1) || (ENABLE_BASIC_CORRECTNESS == 3)
    TestPerf<TestPerSharkParams1, sharkOperator>(testBase + 1, SharkTestIterCount);
    TestPerf<TestPerSharkParams2, sharkOperator>(testBase + 2, SharkTestIterCount);
    TestPerf<TestPerSharkParams3, sharkOperator>(testBase + 3, SharkTestIterCount);
    TestPerf<TestPerSharkParams4, sharkOperator>(testBase + 4, SharkTestIterCount);

    TestPerf<TestPerSharkParams5, sharkOperator>(testBase + 5, SharkTestIterCount);
    TestPerf<TestPerSharkParams6, sharkOperator>(testBase + 6, SharkTestIterCount);
    TestPerf<TestPerSharkParams7, sharkOperator>(testBase + 7, SharkTestIterCount);
    TestPerf<TestPerSharkParams8, sharkOperator>(testBase + 8, SharkTestIterCount);
#elif (ENABLE_BASIC_CORRECTNESS == 2)
    TestPerf<TestPerSharkParams1, sharkOperator>(testBase + 1, SharkTestIterCount);
#endif
    return Tests.CheckAllTestsPassed();
}

// Explicitly instantiate TestAllBinaryOp
#ifdef MULTI_KERNEL
#define ExplicitlyInstantiate(SharkFloatParams) \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::Add>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase);
#else
#define ExplicitlyInstantiate(SharkFloatParams) \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase);
#endif

template bool TestBinaryOperatorPerf<Operator::Add>(int testBase);
template bool TestBinaryOperatorPerf<Operator::MultiplyKaratsubaV2>(int testBase);

ExplicitInstantiateAll();