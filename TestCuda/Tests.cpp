#include "TestVerbose.h"
#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"

#include "Tests.h"
#include "ReferenceKaratsuba.h"
#include "ReferenceAdd.h"
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

static constexpr bool EnableTestSign1 = false;
static constexpr bool EnableTestSign2 = true;
static constexpr bool EnableTestSign3 = true;
static constexpr bool EnableTestSign4 = true;
static constexpr bool EnableTestSign5 = true;
static constexpr bool EnableTestSign6 = true;
static constexpr bool EnableTestSign7 = true;
static constexpr bool EnableTestSign8 = true;

// x_(n + 1) = x_n * x_n - y_n * y_n + a
// y_(n + 1) = 2 * x_n * y_n + b

static TestTracker Tests;

struct IntSignCombo {
    IntSignCombo(bool negative, int32_t exponent, std::vector<uint32_t> digits)
        : Negative{ negative },
        Exponent{ exponent },
        Digits { std::move(digits) }
    {
    }

    IntSignCombo(std::vector<uint32_t> digits)
        : Negative{},
        Exponent{},
        Digits { std::move(digits) }
    {
    }

    bool Negative;
    int32_t Exponent;
    std::vector<uint32_t> Digits;
};

template<class SharkFloatParams, Operator sharkOperator>
bool DiffAgainstHostNonZero (
    int testNum,
    int /*numTerms*/,
    std::string hostCustomOrGpu,
    const mpf_t mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult) {
    bool testSucceeded = true;

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl << hostCustomOrGpu << " result: " << std::endl;
        std::cout << gpuResult.ToString() << std::endl;
        std::cout << gpuResult.ToHexString() << std::endl;
    }

    // Convert gpuResult --> mpfXGpuResult
    mpf_t mpfXGpuResult;
    mpf_init(mpfXGpuResult);
    gpuResult.HpGpuToMpf(mpfXGpuResult);

    // Compute absolute difference: mpfDiffAbs = |mpfHostResult - mpfXGpuResult|
    mpf_t mpfDiff, mpfDiffAbs;
    mpf_init(mpfDiff);
    mpf_init(mpfDiffAbs);
    mpf_sub(mpfDiff, mpfHostResult, mpfXGpuResult);
    mpf_abs(mpfDiffAbs, mpfDiff);

    // Converted GPU result
    if (SharkVerbose == VerboseMode::Debug) {
        // mpfHostResult:
        std::cout << "\nConverted host result (mpfHostResult):" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfHostResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;

        std::cout << "\nConverted " << hostCustomOrGpu << " result (mpfXGpuResult):" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfXGpuResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;

        // Print the differences
        std::cout << "\nDifference between host and " << hostCustomOrGpu << " results:" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec) << std::endl;
    }

    // Retrieve total precision bits:
    mp_bitcnt_t gpuPrecBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    mp_bitcnt_t margin = sizeof(uint32_t) * 8 * 3 + 2;  // as before
    mp_bitcnt_t totalPrecBits = (gpuPrecBits > margin ? gpuPrecBits - margin : 1);

    // Compute epsilon = 2^(-totalPrecBits)
    mpf_t epsilon;
    mpf_init2(epsilon, totalPrecBits);
    mpf_set_ui(epsilon, 1);
    mpf_div_2exp(epsilon, epsilon, totalPrecBits);  // epsilon = 1 / 2^totalPrecBits

    // Compute |host| into mpfAbsHost
    mpf_t mpfAbsHost;
    mpf_init(mpfAbsHost);
    mpf_abs(mpfAbsHost, mpfHostResult);

    // compute floor(log2(1/err)) in high precision
    auto BitsOfError = [&](const mpf_t err) -> int {
        mpf_t invErr;
        if (mpf_sgn(err) == 0) {
            return static_cast<int>(totalPrecBits);
        }

        mpf_init2(invErr, totalPrecBits);
        mpf_ui_div(invErr, 1, err);                  // invErr = 1/err
        
        mp_exp_t exp;
        mpf_get_d_2exp(&exp, invErr);
        mpf_clear(invErr);
        return static_cast<int>(exp) - 1;            // floor(log_2(invErr))
    };

    // CASE A: host is "effectively zero" if |host| <= epsilon.
    if (mpf_cmp(mpfAbsHost, epsilon) <= 0) {
        // Then we compare absolute error directly against epsilon:
        //
        //   If | host - gpu | <= epsilon --> PASS
        //   else                    --> FAIL
        const auto bitsErrA = BitsOfError(mpfDiffAbs);

        if (mpf_cmp(mpfDiffAbs, epsilon) <= 0) {
            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "\nPASS (|host| <= epsilon):\n"
                    << "  |host| = " << MpfToString<SharkFloatParams>(mpfAbsHost, LowPrec)
                    << "  epsilon = " << MpfToString<SharkFloatParams>(epsilon, LowPrec)
                    << "\n  |host - gpu| = "
                    << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec)
                    << "  Bits of error = " << bitsErrA
                    << std::endl;
            }
            Tests.MarkSuccess(testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nFAIL (|host| <= epsilon but absolute error > epsilon):\n"
                << "  |host| = " << MpfToString<SharkFloatParams>(mpfAbsHost, LowPrec) << std::endl
                << "  epsilon      = " << MpfToString<SharkFloatParams>(epsilon, LowPrec) << std::endl
                << "  |host - gpu| = "
                << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec) << std::endl
                << "  Bits of error = " << bitsErrA << std::endl;
            Tests.MarkFailed(
                testNum,
                hostCustomOrGpu,
                MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec),
                MpfToString<SharkFloatParams>(epsilon, LowPrec));
            testSucceeded = false;
        }
    }
    // CASE B: host is not "tiny," so do a normal relative-error check
    else {
        mpf_t relativeError;
        mpf_init(relativeError);
        {
            mpf_t tmp;
            mpf_init(tmp);
            mpf_div(tmp, mpfDiffAbs, mpfAbsHost);
            mpf_abs(relativeError, tmp);
            mpf_clear(tmp);
        }

        // Compute relativeError = | host - gpu | / | host |
        const auto bitsErrB = BitsOfError(relativeError);

        // Compare: if relativeError <= epsilon --> PASS; else FAIL
        if (mpf_cmp(relativeError, epsilon) <= 0) {
            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "\nPASS (relative-error check):\n"
                    << "  relativeError = "
                    << MpfToString<SharkFloatParams>(relativeError, LowPrec) << std::endl
                    << "  epsilon            = "
                    << MpfToString<SharkFloatParams>(epsilon, LowPrec) << std::endl
                    << "  Bits of error: " << bitsErrB << std::endl;
            }
            Tests.MarkSuccess(testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nFAIL (relative-error exceeds epsilon):\n"
                << "  relativeError = "
                << MpfToString<SharkFloatParams>(relativeError, LowPrec) << std::endl
                << "  epsilon             = "
                << MpfToString<SharkFloatParams>(epsilon, LowPrec) << std::endl
                << "  Bits of error: " << bitsErrB << std::endl;
            Tests.MarkFailed(
                testNum,
                hostCustomOrGpu,
                MpfToString<SharkFloatParams>(relativeError, LowPrec),
                MpfToString<SharkFloatParams>(epsilon, LowPrec));
            testSucceeded = false;
        }
        mpf_clear(relativeError);
    }

    // Clean up
    mpf_clear(mpfAbsHost);
    mpf_clear(epsilon);
    mpf_clear(mpfDiff);
    mpf_clear(mpfDiffAbs);
    mpf_clear(mpfXGpuResult);

    return testSucceeded;
}

template<class SharkFloatParams, Operator sharkOperator>
bool DiffAgainstHost(
    int testNum,
    int numTerms,               // 2 or 3
    std::string hostCustomOrGpu,
    const mpf_t  mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult) {
    // 1) Optional verbose print of GPU result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl << hostCustomOrGpu << " (GPU) result:\n"
            << gpuResult.ToString() << std::endl
            << gpuResult.ToHexString() << std::endl;
    }

    // 2) Convert host mpf_t --> HpSharkFloat via MpfToHpGpu
    HpSharkFloat<SharkFloatParams> hostShark;
    hostShark.MpfToHpGpu(
        mpfHostResult,
        HpSharkFloat<SharkFloatParams>::DefaultPrecBits
    );

    // 3) Build absolute-difference mpf: |host - gpu|
    mpf_t mpfXGpu;
    mpf_t mpfDiff;
    mpf_t mpfDiffAbs;

    mpf_init(mpfXGpu);
    mpf_init(mpfDiff);
    mpf_init(mpfDiffAbs);

    gpuResult.HpGpuToMpf(mpfXGpu);
    mpf_sub(mpfDiff, mpfHostResult, mpfXGpu);
    mpf_abs(mpfDiffAbs, mpfDiff);

    // 4) Quick check: is host exactly zero?
    mpf_t mpfZero;
    mpf_init(mpfZero);
    mpf_set_ui(mpfZero, 0);

    const bool hostIsZero = (mpf_cmp(mpfHostResult, mpfZero) == 0);
    mpf_clear(mpfZero);

    if (hostIsZero) {
        // ---- FALLBACK: absolute ULP-based threshold at GPU exponent ----
        mp_bitcnt_t P = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
        mpf_t eps;
        mpf_init2(eps, P);
        mpf_set_ui(eps, 1);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\nBefore fallback absolute-error threshold : "
                << MpfToString<SharkFloatParams>(eps, LowPrec) << std::endl;
            std::cout << "Absolute difference: "
                << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec) << std::endl;
        }

        // 2) compute trueExponent = expGpu + (M*32 - 1)
        const int mantBits = int(HpSharkFloat<SharkFloatParams>::NumUint32) * 32;
        int trueExp = gpuResult.Exponent + (mantBits - 1);

        // 3) shift eps to 2^trueExp
        if (trueExp >= 0) {
            mpf_mul_2exp(eps, eps, trueExp);
        } else {
            mpf_div_2exp(eps, eps, -trueExp);
        }

        // 4) scale by (numTerms-1)
        mpf_mul_ui(eps, eps, static_cast<unsigned long>(numTerms - 1));

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\nFallback absolute-error threshold : "
                << MpfToString<SharkFloatParams>(eps, LowPrec) << std::endl;
            std::cout << "Absolute difference: "
                << MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec) << std::endl;
        }

        bool ok = (mpf_cmp(mpfDiffAbs, eps) <= 0);

        if (ok) {
            Tests.MarkSuccess(testNum, hostCustomOrGpu);
        } else {
            std::string diffStr = MpfToString<SharkFloatParams>(mpfDiffAbs, LowPrec);
            std::string threshStr = MpfToString<SharkFloatParams>(eps, LowPrec);
            std::cerr << "\nError: absolute error \"" << diffStr
                << "\" > allowed \"" << threshStr << "\"\n";
            Tests.MarkFailed(testNum, hostCustomOrGpu, diffStr, threshStr);
        }

        mpf_clear(eps);
        mpf_clear(mpfXGpu);
        mpf_clear(mpfDiff);
        mpf_clear(mpfDiffAbs);
        return ok;
    }

    return DiffAgainstHostNonZero<SharkFloatParams, sharkOperator>(
        testNum,
        numTerms,
        hostCustomOrGpu,
        mpfHostResult,
        gpuResult);
}


template<class SharkFloatParams, Operator sharkOperator>
void TestPerf (
    int testNum,
    const char *num1,
    const char *num2,
    const char *num3,
    const mpf_t mpfX,
    const mpf_t mpfY,
    const mpf_t mpfZ,
    uint64_t numIters) {

    // Print the original input values
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: " << MpfToString<SharkFloatParams>(mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "Y: " << MpfToString<SharkFloatParams>(mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "num3: " << num3 << std::endl;
        std::cout << "Z: " << MpfToString<SharkFloatParams>(mpfZ, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
    }

    auto desc = SharkFloatParams::GetDescription();
    std::cout << "\nTest " << testNum << ": " << OperatorToString<sharkOperator>() << " " << desc << std::endl;

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    auto resultNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    xNum->MpfToHpGpu(mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    yNum->MpfToHpGpu(mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
    zNum->MpfToHpGpu(mpfZ, HpSharkFloat<SharkFloatParams>::DefaultPrecBits);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;
        std::cout << "X: " << xNum->ToString() << std::endl;
        std::cout << "Y: " << yNum->ToString() << std::endl;
        std::cout << "Z: " << zNum->ToString() << std::endl;
    }

    // Perform the calculation on the host using MPIR
    mpf_t mpfHostResultXX;
    mpf_t mpfHostResultXY1;
    mpf_t mpfHostResultXY2;
    mpf_t mpfHostResultYY;

    mpf_init(mpfHostResultXX);
    mpf_init(mpfHostResultXY1);
    mpf_init(mpfHostResultXY2);
    mpf_init(mpfHostResultYY);

    // Reference orbit:
    mpf_t tempX, tempY, xSquared, ySquared, twoXY;
    mpf_init(tempX);
    mpf_init(tempY);
    mpf_init(xSquared);
    mpf_init(ySquared);
    mpf_init(twoXY);

    if constexpr (SharkTestBenchmarkAgainstHost) {
        BenchmarkTimer hostTimer;
        ScopedBenchmarkStopper hostStopper{ hostTimer };

        for (int i = 0; i < numIters; ++i) {
            if constexpr (sharkOperator == Operator::Add) {
                mpf_sub(mpfHostResultXY1, mpfX, mpfY);
                mpf_add(mpfHostResultXY1, mpfHostResultXY1, mpfZ);
                mpf_add(mpfHostResultXY2, mpfX, mpfY);
            } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
                mpf_mul(mpfHostResultXX, mpfX, mpfX);
                mpf_mul(mpfHostResultXY1, mpfX, mpfY);
                mpf_mul(mpfHostResultYY, mpfY, mpfY);
            } else if constexpr (sharkOperator == Operator::ReferenceOrbit) {
                // x_(n + 1) = x_n * x_n - y_n * y_n + a
                // y_(n + 1) = 2 * x_n * y_n + b

                mpf_mul(xSquared, mpfX, mpfX); // x^2
                mpf_mul(ySquared, mpfY, mpfY); // y^2
                mpf_mul(twoXY, mpfX, mpfY);    // xy
                mpf_mul_ui(twoXY, twoXY, 2);   // 2xy
                mpf_sub(tempX, xSquared, ySquared); // x^2 - y^2
                mpf_add(mpfHostResultXX, tempX, mpfZ); // x^2 - y^2 + a
                mpf_add(mpfHostResultXY1, twoXY, mpfY); // 2xy + b
               
            } else {
                std::cerr << "Unknown operator in TestPerf" << std::endl;
                return;
            }
        }

        hostTimer.StopTimer();

        std::cout << "Host iter time: " << hostTimer.GetDeltaInMs() << " ms" << std::endl;
    }

    if constexpr (SharkTestGpu) {

        auto CheckDiff = [&](
            const int testNum,
            const int numTerms,
            const char *hostCustomOrGpu,
            const mpf_t &mpfHostResult,
            const HpSharkFloat<SharkFloatParams> &gpuResult) {

                auto testSucceeded =
                    DiffAgainstHost<SharkFloatParams, sharkOperator>(
                        testNum,
                        numTerms,
                        hostCustomOrGpu,
                        mpfHostResult,
                        gpuResult);
                if (!testSucceeded) {
                    std::cout << "Perf correctness test failed" << std::endl;
                } else {
                    std::cout << "Perf correctness test succeeded" << std::endl;
                }

                return testSucceeded;
            };

        if constexpr (sharkOperator == Operator::Add) {
            auto combo = std::make_unique<HpSharkAddComboResults<SharkFloatParams>>();

            combo->A_X2 = *xNum;
            combo->B_Y2 = *yNum;
            combo->C_A = *zNum;
            combo->D_2X = *xNum;
            combo->E_B = *yNum;

            const auto &gpuResultXY1 = combo->Result1_A_B_C;
            const auto &gpuResultXY2 = combo->Result2_D_E;

            {
                BenchmarkTimer timer;
                InvokeAddKernelPerf<SharkFloatParams>(
                    timer,
                    *combo,
                    numIters);
                Tests.AddTime(testNum, timer.GetDeltaInMs());
                std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;
            }

            if constexpr (SharkTestBenchmarkAgainstHost) {
                bool testSucceeded = true;
                constexpr auto numTerms = 3;
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultXY1, gpuResultXY1);
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultXY2, gpuResultXY2);
            }

        } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {

            auto combo = std::make_unique<HpSharkComboResults<SharkFloatParams>>();
            combo->A = *xNum;
            combo->B = *yNum;

            const auto &gpuResult2XX = combo->ResultX2;
            const auto &gpuResult2XY = combo->ResultXY;
            const auto &gpuResult2YY = combo->ResultY2;

            {
                BenchmarkTimer timer;
                InvokeMultiplyKernelPerf<SharkFloatParams>(
                    timer,
                    *combo,
                    numIters);
                Tests.AddTime(testNum, timer.GetDeltaInMs());
                std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;
            }

            if constexpr (SharkTestBenchmarkAgainstHost) {
                bool testSucceeded = true;
                constexpr auto numTerms = 2;
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultXX, gpuResult2XX);
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultXY1, gpuResult2XY);
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultYY, gpuResult2YY);
            }
        } else if constexpr (sharkOperator == Operator::ReferenceOrbit) {
            auto combo = std::make_unique<HpSharkReferenceResults<SharkFloatParams>>();
            combo->A = *xNum;
            combo->B = *yNum;

            const auto &gpuResultX = combo->ResultA;
            const auto &gpuResultY = combo->ResultB;

            {
                BenchmarkTimer timer;
                InvokeHpSharkReferenceKernelPerf<SharkFloatParams>(
                    timer,
                    *combo,
                    numIters);
                Tests.AddTime(testNum, timer.GetDeltaInMs());
                std::cout << "GPU iter time: " << timer.GetDeltaInMs() << " ms" << std::endl;
            }

            if constexpr (SharkTestBenchmarkAgainstHost) {
                bool testSucceeded = true;
                constexpr auto numTerms = 2;
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultXX, gpuResultX);
                testSucceeded &= CheckDiff(testNum, numTerms, "GPU", mpfHostResultYY, gpuResultY);
            }
        }
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResultXX);
    mpf_clear(mpfHostResultXY1);
    mpf_clear(mpfHostResultXY2);
    mpf_clear(mpfHostResultYY);

    // Clean up reference orbit variables
    mpf_clear(tempX);
    mpf_clear(tempY);
    mpf_clear(xSquared);
    mpf_clear(ySquared);
    mpf_clear(twoXY);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestPerf (
    int testNum,
    uint64_t numIters) {

    HpSharkFloat<SharkFloatParams> xNum;
    HpSharkFloat<SharkFloatParams> yNum;
    HpSharkFloat<SharkFloatParams> zNum;

    xNum.GenerateRandomNumber2();
    yNum.GenerateRandomNumber2();
    zNum.GenerateRandomNumber2();

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);  // Set precision for MPIR floating point

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_t mpfZ;

    mpf_init(mpfX);
    mpf_init(mpfY);
    mpf_init(mpfZ);

    xNum.HpGpuToMpf(mpfX);
    yNum.HpGpuToMpf(mpfY);
    zNum.HpGpuToMpf(mpfZ);

    auto num1 = xNum.ToString();
    auto num2 = yNum.ToString();
    auto num3 = zNum.ToString();

    TestPerf<SharkFloatParams, sharkOperator>(
        testNum,
        num1.c_str(),
        num2.c_str(),
        num3.c_str(),
        mpfX,
        mpfY,
        mpfZ,
        numIters);

    mpf_clear(mpfX);
    mpf_clear(mpfY);
    mpf_clear(mpfZ);
}

template<class SharkFloatParams, Operator sharkOperator>
bool CheckAgainstHost(
    int testNum,
    int numTerms,
    const char *name,
    const mpf_t mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    bool res = DiffAgainstHost<SharkFloatParams, sharkOperator>(
        testNum,
        numTerms,
        name,
        mpfHostResult,
        gpuResult);
    if (!res) {
        DebugBreak();
    };

    return res;
}

template<class SharkFloatParams>
void ChecksumsCheck (
    const std::vector<DebugStateHost<SharkFloatParams>> &debugResultsHost,
    const std::vector<DebugStateRaw> &debugStatesCuda
)
{
    // Compare debugResultsCuda against debugResultsHost
    bool ChecksumFailure = false;
    if constexpr (SharkTestGpu && SharkDebugChecksums) {
        assert(debugResultsHost.size() <= debugStatesCuda.size());

        // Note that the hosts results should be exactly the right size, whereas
        // the CUDA results may be larger due to the way the kernel is written.
        for (size_t i = 0; i < debugResultsHost.size(); ++i) {
            const auto &host = debugResultsHost[i];
            const auto &cuda = debugStatesCuda[i];

            const auto maxHostArraySize = std::max(host.ArrayToChecksum32.size(), host.ArrayToChecksum64.size());

            if (host.Initialized != (cuda.Initialized == 1) ||
                host.Checksum != cuda.Checksum ||
                host.ChecksumPurpose != cuda.ChecksumPurpose ||
                host.CallIndex != cuda.CallIndex ||
                maxHostArraySize != cuda.ArraySize) {

                std::cerr << "======================================" << std::endl;
                std::cerr << "Error: Checksum mismatch at index 0x" << std::hex << i << std::endl;
                std::cerr << "GPU:" << std::endl;

                // Print all fields of cuda:
                std::cerr << "Initialized: " << cuda.Initialized << std::endl;
                std::cerr << "Block: " << cuda.Block << std::endl;
                std::cerr << "Thread: " << cuda.Thread << std::endl;
                std::cerr << "ArraySize: " << cuda.ArraySize << std::endl;

                std::cerr << "Checksum: 0x" << std::hex << cuda.Checksum << std::dec << std::endl;
                std::cerr << "ChecksumPurpose: " << static_cast<int>(cuda.ChecksumPurpose) << std::endl;
                std::cerr << "ChecksumPurpose: " << DebugStatePurposeToString(cuda.ChecksumPurpose) << std::endl;

                std::cerr << "RecursionDepth: " << cuda.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << cuda.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(cuda.Convolution) << std::endl;

                // Print all fields of host
                std::cerr << std::endl;
                std::cerr << "Host reference implementation:" << std::endl;
                std::cerr << "Initialized: " << host.Initialized << std::endl;
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
                std::cerr << "ChecksumPurpose: " << DebugStatePurposeToString(host.ChecksumPurpose) << std::endl;

                std::cerr << "RecursionDepth: " << host.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << host.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(host.Convolution) << std::endl;

                ChecksumFailure = true;

                DebugBreak();
            }
        }
    }
}

template<class SharkFloatParams, Operator sharkOperator>
bool CheckGPUResult (
    int testNum,
    int numTerms,
    const char *name,
    const mpf_t &mpfHostResult,
    const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    auto testSucceeded = DiffAgainstHost<SharkFloatParams, sharkOperator>(
        testNum,
        numTerms,
        name,
        mpfHostResult,
        gpuResult);

    if (SharkVerbose == VerboseMode::Debug) {
        if (!testSucceeded) {
            std::cout << "GPU High Precision failed" << std::endl;
        } else {
            std::cout << "GPU High Precision succeeded" << std::endl;
        }
    }

    if (!testSucceeded) {
        // If the test failed, we should break into the debugger
        DebugBreak();
    }

    return testSucceeded;
}

template<class SharkFloatParams>
void TestCoreAdd (
    int testNum,
    const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
    const mpf_t *mpfInputX,
    size_t mpfInputLen)
{

    assert(inputX.size() == 3 || inputX.size() == 5);

    assert(inputX.size() == mpfInputLen);

    const auto &aNum = inputX[0];
    const auto &bNum = inputX[1];
    const auto &cNum = inputX[2];

    const auto &dNum = (inputX.size() == 5) ? inputX[3] : aNum;
    const auto &eNum = (inputX.size() == 5) ? inputX[4] : bNum;

    const auto &mpfA = mpfInputX[0];
    const auto &mpfB = mpfInputX[1];
    const auto &mpfC = mpfInputX[2];

    const auto &mpfD = (mpfInputLen == 5) ? mpfInputX[3] : mpfA;
    const auto &mpfE = (mpfInputLen == 5) ? mpfInputX[4] : mpfB;

    constexpr auto sharkOperator = Operator::Add;

    auto TestHostAdd = [](
        int testNum,
        const HpSharkFloat<SharkFloatParams> &aNum,
        const HpSharkFloat<SharkFloatParams> &bNum,
        const HpSharkFloat<SharkFloatParams> &cNum,
        const HpSharkFloat<SharkFloatParams> &dNum,
        const HpSharkFloat<SharkFloatParams> &eNum,
        const mpf_t &mpfHostResultXY1,
        const mpf_t &mpfHostResultXY2,
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates) -> bool {

            HpSharkFloat<SharkFloatParams> hostAddResult1;
            HpSharkFloat<SharkFloatParams> hostAddResult2;

            AddHelper<SharkFloatParams>(
                &aNum,
                &bNum,
                &cNum,
                &dNum,
                &eNum,
                &hostAddResult1,
                &hostAddResult2,
                debugStates
            );

            auto OutputAdd = [&](const char *desc, [[maybe_unused]] const HpSharkFloat<SharkFloatParams> &out) {
                if (SharkVerbose == VerboseMode::Debug) {
                    std::cout << desc << out.ToString() << std::endl;
                    std::cout << desc << " hex: " << out.ToHexString() << std::endl;
                }
                };

            OutputAdd("Add result 1: ", hostAddResult1);
            OutputAdd("Add result 2: ", hostAddResult2);

            bool res = true;
            constexpr auto numTermsPartABC = 3;
            res &= CheckAgainstHost<SharkFloatParams, sharkOperator>(
                testNum,
                numTermsPartABC,
                "CustomHighPrecisionV2XY1",
                mpfHostResultXY1,
                hostAddResult1);

            constexpr auto numTermsPartDE = 2;
            res &= CheckAgainstHost<SharkFloatParams, sharkOperator>(
                testNum,
                numTermsPartDE,
                "CustomHighPrecisionV2XY2",
                mpfHostResultXY2,
                hostAddResult2);

            return res;
        };

    // Perform the calculation on the using MPIR
    HpSharkFloat<SharkFloatParams> gpuResultXX{};
    HpSharkFloat<SharkFloatParams> gpuResultXY1{};
    HpSharkFloat<SharkFloatParams> gpuResultXY2{};
    HpSharkFloat<SharkFloatParams> gpuResultYY{};

    mpf_t mpfHostResultXX;
    mpf_t mpfHostResultXY1;
    mpf_t mpfHostResultXY2;
    mpf_t mpfHostResultYY;

    mpf_init(mpfHostResultXX);
    mpf_init(mpfHostResultXY1);
    mpf_init(mpfHostResultXY2);
    mpf_init(mpfHostResultYY);

    mpf_sub(mpfHostResultXY1, mpfA, mpfB);
    mpf_add(mpfHostResultXY1, mpfHostResultXY1, mpfC);

    mpf_add(mpfHostResultXY2, mpfD, mpfE);

    // Print host result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nCorrect MPIR result:" << std::endl;
        std::cout << "Correct MPIR result XY1: " <<
            MpfToString<SharkFloatParams>(mpfHostResultXY1, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Correct MPIR hex XY1: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultXY1) << std::endl;
        std::cout << "Correct MPIR result XY2: " <<
            MpfToString<SharkFloatParams>(mpfHostResultXY2, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Correct MPIR hex XY2: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultXY2) << std::endl;
    }

    std::vector<DebugStateRaw> debugStatesCuda{};
    if constexpr (SharkTestGpu) {
        BenchmarkTimer timer;

        HpSharkAddComboResults<SharkFloatParams> combo;
        combo.A_X2 = aNum;
        combo.B_Y2 = bNum;
        combo.C_A = cNum;
        combo.D_2X = dNum;
        combo.E_B = eNum;

        InvokeAddKernelCorrectness<SharkFloatParams>(
            timer,
            combo,
            &debugStatesCuda);

        gpuResultXY1 = combo.Result1_A_B_C;
        gpuResultXY2 = combo.Result2_D_E;

        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }
    }

    std::vector<DebugStateHost<SharkFloatParams>> debugResultsHost;

    bool testSucceeded = TestHostAdd(
        testNum,
        aNum,
        bNum,
        cNum,
        dNum,
        eNum,
        mpfHostResultXY1,
        mpfHostResultXY2,
        debugResultsHost);

    if (SharkVerbose == VerboseMode::Debug) {
        if (!testSucceeded) {
            std::cout << "Custom High Precision failed" << std::endl;
        } else {
            std::cout << "Custom High Precision succeeded" << std::endl;
        }
    }

    ChecksumsCheck<SharkFloatParams>(
        debugResultsHost,
        debugStatesCuda);

    if constexpr (SharkTestGpu) {
        testSucceeded = true;
        
        constexpr auto numTermsABC = 3;
        testSucceeded &= CheckGPUResult<SharkFloatParams, sharkOperator>(
            testNum, numTermsABC, "GPU", mpfHostResultXY1, gpuResultXY1);
        
        constexpr auto numTermsDE = 2;
        testSucceeded &= CheckGPUResult<SharkFloatParams, sharkOperator>(
            testNum, numTermsDE, "GPU", mpfHostResultXY2, gpuResultXY2);
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResultXX);
    mpf_clear(mpfHostResultXY1);
    mpf_clear(mpfHostResultXY2);
    mpf_clear(mpfHostResultYY);
}

template<class SharkFloatParams>
void TestCoreMultiply(
    int testNum,
    const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
    const mpf_t *mpfInputX,
    size_t mpfInputLen)
{
    (void)mpfInputLen; // Unused parameter, but kept for compatibility
    assert(inputX.size() == 3 || inputX.size() == 5);
    assert(mpfInputLen >= 2);

    const auto &aNum = inputX[0];
    const auto &bNum = inputX[1];

    assert(inputX.size() == mpfInputLen);
    const auto &mpfA = mpfInputX[0];
    const auto &mpfB = mpfInputX[1];

    auto TestHostKaratsuba = [](
        int testNum,
        const HpSharkFloat<SharkFloatParams> &aNum,
        const HpSharkFloat<SharkFloatParams> &bNum,
        const mpf_t &mpfHostResultXX,
        const mpf_t &mpfHostResultXY1,
        const mpf_t &mpfHostResultYY,
        std::vector<DebugStateHost<SharkFloatParams>> &debugStates) -> bool {

            HpSharkFloat<SharkFloatParams> hostKaratsubaOutXXV2;
            HpSharkFloat<SharkFloatParams> hostKaratsubaOutXYV2;
            HpSharkFloat<SharkFloatParams> hostKaratsubaOutYYV2;

            MultiplyHelperKaratsubaV2<SharkFloatParams>(
                &aNum,
                &bNum,
                &hostKaratsubaOutXXV2,
                &hostKaratsubaOutXYV2,
                &hostKaratsubaOutYYV2,
                debugStates
            );

            auto OutputV2 = [&]([[maybe_unused]] const HpSharkFloat<SharkFloatParams> &out) {
                if (SharkVerbose == VerboseMode::Debug) {
                    std::cout << "KaratsubaV2 result: " << out.ToString() << std::endl;
                    std::cout << "KaratsubaV2 hex: " << out.ToHexString() << std::endl;
                }
                };

            OutputV2(hostKaratsubaOutXXV2);
            OutputV2(hostKaratsubaOutXYV2);
            OutputV2(hostKaratsubaOutYYV2);

            bool res = true;
            constexpr auto numTerms = 2;
            res &= CheckAgainstHost<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
                testNum,
                numTerms,
                "CustomHighPrecisionV2XX",
                mpfHostResultXX,
                hostKaratsubaOutXXV2);

            res &= CheckAgainstHost<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
                testNum,
                numTerms,
                "CustomHighPrecisionV2XY",
                mpfHostResultXY1,
                hostKaratsubaOutXYV2);

            res &= CheckAgainstHost<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
                testNum,
                numTerms,
                "CustomHighPrecisionV2YY",
                mpfHostResultYY,
                hostKaratsubaOutYYV2);

            return res;
        };

    // Perform the calculation on the using MPIR
    HpSharkFloat<SharkFloatParams> gpuResultXX{};
    HpSharkFloat<SharkFloatParams> gpuResultXY1{};
    HpSharkFloat<SharkFloatParams> gpuResultXY2{};
    HpSharkFloat<SharkFloatParams> gpuResultYY{};

    mpf_t mpfHostResultXX;
    mpf_t mpfHostResultXY1;
    mpf_t mpfHostResultXY2;
    mpf_t mpfHostResultYY;

    mpf_init(mpfHostResultXX);
    mpf_init(mpfHostResultXY1);
    mpf_init(mpfHostResultXY2);
    mpf_init(mpfHostResultYY);

    mpf_mul(mpfHostResultXX, mpfA, mpfA);
    mpf_mul(mpfHostResultXY1, mpfA, mpfB);
    mpf_mul(mpfHostResultYY, mpfB, mpfB);

    // Print host result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nCorrect MPIR result:" << std::endl;
        std::cout << "Correct MPIR result XX: " <<
            MpfToString<SharkFloatParams>(mpfHostResultXX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Correct MPIR result XY: " <<
            MpfToString<SharkFloatParams>(mpfHostResultXY1, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Correct MPIR result YY: " <<
            MpfToString<SharkFloatParams>(mpfHostResultYY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;

        std::cout << "Correct MPIR hex XX: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultXX) << std::endl;
        std::cout << "Correct MPIR hex XY: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultXY1) << std::endl;
        std::cout << "Correct MPIR hex YY: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultYY) << std::endl;
    }

    std::vector<DebugStateRaw> debugStatesCuda{};
    if constexpr (SharkTestGpu) {
        BenchmarkTimer timer;

        HpSharkComboResults<SharkFloatParams> combo;
        combo.A = aNum;
        combo.B = bNum;

        InvokeMultiplyKernelCorrectness<SharkFloatParams>(
            timer,
            combo,
            &debugStatesCuda);

        gpuResultXX = combo.ResultX2;
        gpuResultXY1 = combo.ResultXY;
        gpuResultYY = combo.ResultY2;

        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }
    }

    std::vector<DebugStateHost<SharkFloatParams>> debugResultsHost;

    bool testSucceeded = false;
    testSucceeded = TestHostKaratsuba(
        testNum,
        aNum,
        bNum,
        mpfHostResultXX,
        mpfHostResultXY1,
        mpfHostResultYY,
        debugResultsHost);

    if (!testSucceeded) {
        std::cout << "Custom High Precision failed" << std::endl;
    } else {
        std::cout << "Custom High Precision succeeded" << std::endl;
    }

    ChecksumsCheck<SharkFloatParams>(
        debugResultsHost,
        debugStatesCuda);

    if constexpr (SharkTestGpu) {
        testSucceeded = true;
        
        constexpr auto numTerms = 2;

        testSucceeded &= CheckGPUResult<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
            testNum, numTerms, "GPU", mpfHostResultXX, gpuResultXX);

        testSucceeded &= CheckGPUResult<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
            testNum, numTerms, "GPU", mpfHostResultXY1, gpuResultXY1);

        testSucceeded &= CheckGPUResult<SharkFloatParams, Operator::MultiplyKaratsubaV2>(
            testNum, numTerms, "GPU", mpfHostResultYY, gpuResultYY);
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResultXX);
    mpf_clear(mpfHostResultXY1);
    mpf_clear(mpfHostResultXY2);
    mpf_clear(mpfHostResultYY);
}

template<class SharkFloatParams>
void TestCoreReferenceOrbit(
    int testNum,
    const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
    const mpf_t *mpfInputX,
    size_t mpfInputLen) {
    (void)mpfInputLen; // Unused parameter, but kept for compatibility
    assert(inputX.size() == 3 || inputX.size() == 5);
    assert(mpfInputLen >= 4);

    const auto &aNum = inputX[0];
    const auto &bNum = inputX[1];

    assert(inputX.size() == mpfInputLen);
    const auto &mpfA = mpfInputX[0];
    const auto &mpfB = mpfInputX[1];
    const auto &mpfX = mpfInputX[2];
    const auto &mpfY = mpfInputX[3];

    // Perform the calculation on the using MPIR
    HpSharkFloat<SharkFloatParams> gpuResultXX{};
    HpSharkFloat<SharkFloatParams> gpuResultXY1{};
    HpSharkFloat<SharkFloatParams> gpuResultXY2{};
    HpSharkFloat<SharkFloatParams> gpuResultYY{};

    mpf_t mpfHostResultX;
    mpf_t mpfHostResultY;

    mpf_t xSquared;
    mpf_t ySquared;
    mpf_t twoXY;
    mpf_t tempX;
    mpf_t tempY;

    mpf_init(xSquared);
    mpf_init(ySquared);
    mpf_init(twoXY);
    mpf_init(tempX);
    mpf_init(tempY);

    mpf_init(mpfHostResultX);
    mpf_init(mpfHostResultY);

    // x_(n + 1) = x_n * x_n - y_n * y_n + a
    // y_(n + 1) = 2 * x_n * y_n + b

    mpf_mul(xSquared, mpfX, mpfX); // x^2
    mpf_mul(ySquared, mpfY, mpfY); // y^2
    mpf_mul(twoXY, mpfX, mpfY);    // xy
    mpf_mul_ui(twoXY, twoXY, 2);   // 2xy
    mpf_sub(tempX, xSquared, ySquared); // x^2 - y^2
    mpf_add(mpfHostResultX, tempX, mpfA); // x^2 - y^2 + a
    mpf_add(mpfHostResultY, twoXY, mpfB); // 2xy + b

    // Print host result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nCorrect MPIR result:" << std::endl;
        std::cout << "Correct MPIR result X: " <<
            MpfToString<SharkFloatParams>(mpfHostResultX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        std::cout << "Correct MPIR result Y: " <<
            MpfToString<SharkFloatParams>(mpfHostResultY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;

        std::cout << "Correct MPIR hex X: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultX) << std::endl;
        std::cout << "Correct MPIR hex Y: " << std::endl;
        std::cout << "" << MpfToHexString(mpfHostResultY) << std::endl;
    }

    std::vector<DebugStateRaw> debugStatesCuda{};
    if constexpr (SharkTestGpu) {

        assert(false);
        BenchmarkTimer timer;

        HpSharkComboResults<SharkFloatParams> combo;
        combo.A = aNum;
        combo.B = bNum;

        InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(
            timer,
            combo,
            &debugStatesCuda);

        gpuResultXX = combo.ResultX2;
        gpuResultXY1 = combo.ResultXY;
        gpuResultYY = combo.ResultY2;

        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "GPU single time: " << timer.GetDeltaInMs() << " ms" << std::endl;
        }
    }

    std::vector<DebugStateHost<SharkFloatParams>> debugResultsHost;

    if constexpr (SharkTestGpu) {
        bool testSucceeded = true;

        constexpr auto numTerms = 2;

        testSucceeded &= CheckGPUResult<SharkFloatParams, Operator::ReferenceOrbit>(
            testNum, numTerms, "GPU", mpfHostResultX, gpuResultXX);

        testSucceeded &= CheckGPUResult<SharkFloatParams, Operator::ReferenceOrbit>(
            testNum, numTerms, "GPU", mpfHostResultY, gpuResultYY);
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResultX);
    mpf_clear(mpfHostResultY);

    mpf_clear(xSquared);
    mpf_clear(ySquared);
    mpf_clear(twoXY);
    mpf_clear(tempX);
    mpf_clear(tempY);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernaryOperatorTwoNumbersRawNoSignChange(
    int testNum,
    const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
    const mpf_t *mpfInputX,
    size_t mpfInputLen) {

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;

        for (size_t i = 0; i < inputX.size(); ++i) {
            std::cout << "X[" << i << "]: " << inputX[i].ToString() << std::endl;
            std::cout << "X[" << i << "] hex: " << inputX[i].ToHexString() << std::endl;
        }

        std::cout << "\nOriginal MPIR input values:" << std::endl;
        for (size_t i = 0; i < mpfInputLen; ++i) {
            std::cout << "X[" << i << "]: " << MpfToString<SharkFloatParams>(mpfInputX[i], HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
        }
    }

    if constexpr (sharkOperator == Operator::Add) {
        TestCoreAdd<SharkFloatParams>(testNum, inputX, mpfInputX, mpfInputLen);
    } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
        TestCoreMultiply<SharkFloatParams>(testNum, inputX, mpfInputX, mpfInputLen);
    } else if constexpr (sharkOperator == Operator::ReferenceOrbit) {
        TestCoreReferenceOrbit<SharkFloatParams>(testNum, inputX, mpfInputX, mpfInputLen);
    } else {
        static_assert(SharkTestForceSameSign, "Unsupported operator for TestTernaryOperatorTwoNumbersRawNoSignChange");
    }
}

template<class SharkFloatParams, Operator sharkOperator, bool IncludeSigns>
void TestTernaryOperatorTwoNumbersRaw (
    int testNum,
    const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
    const mpf_t *mpfInputX,
    size_t mpfInputLen) {

    std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy{};
    auto mpfXCopy = std::make_unique<mpf_t[]>(mpfInputLen);

    for (size_t i = 0; i < mpfInputLen; ++i) {
        mpf_init(mpfXCopy[i]);
        mpf_set(mpfXCopy[i], mpfInputX[i]);
    }

    // If IncludeSigns is true, then call TestTernaryOperatorTwoNumbersRawNoSignChange with all four variants
    // using mpf_neg as needed

    if constexpr (IncludeSigns) {
        assert(inputX.size() == 3 || inputX.size() == 5);
        assert(mpfInputLen == 3 || mpfInputLen == 5);
        assert(inputX.size() == mpfInputLen);

        auto resetCopy = [&]() {
            xNumCopy.clear();
            xNumCopy.resize(inputX.size());

            assert(xNumCopy.size() == mpfInputLen);

            for (size_t i = 0; i < inputX.size(); ++i) {
                xNumCopy[i].DeepCopySameDevice(inputX[i]);
                mpf_set(mpfXCopy[i], mpfInputX[i]);
            }
        };

        auto printTest = [&](int curTest) {

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << std::endl;
                std::cout << std::endl;
            }

            std::cout << "Test " << std::dec << curTest << std::endl;
            };

        auto negateMpfAndHp = [](mpf_t &mpfCopy, HpSharkFloat<SharkFloatParams> &numCopy) {
            mpf_neg(mpfCopy, mpfCopy);
            numCopy.Negate();
            };

        //
        // With three numbers, there are 8 combinations of signs
        // 

        if constexpr (EnableTestSign1) {
            resetCopy();
            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
            testNum++;
        }

        if constexpr (EnableTestSign2) {
            resetCopy();
            if constexpr (!SharkTestForceSameSign) {
                negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);

                negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);
            }

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
            testNum++;
        }

        if constexpr (EnableTestSign3) {
            resetCopy();
            if constexpr (!SharkTestForceSameSign) {
                negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);

                negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);
            }

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
            testNum++;
        }

        if constexpr (EnableTestSign4) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);
            negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);

            negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);
            negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
            testNum++;
        }

        if constexpr (EnableTestSign5) {
            resetCopy();
            if constexpr (!SharkTestForceSameSign) {
                negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);
            }

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
            testNum++;
        }

        if constexpr (EnableTestSign6) {
            resetCopy();
            if constexpr (!SharkTestForceSameSign) {
                negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);
                negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);

                negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);
            }

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
        }

        if constexpr (EnableTestSign7) {
            resetCopy();
            if constexpr (!SharkTestForceSameSign) {
                negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);
                negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);

                negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);
            }

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
        }

        if constexpr (EnableTestSign8) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);
            negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);
            negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);

            negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);
            negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                testNum, xNumCopy, mpfXCopy.get(), mpfInputLen);
        }

    } else {
        TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            testNum,
            inputX,
            mpfXCopy.get(),
            mpfInputLen);
    }

    for (size_t i = 0; i < mpfInputLen; ++i) {
        mpf_clear(mpfXCopy[i]);
    }
}

// Win32 clear console
void ClearConsole () {
    system("cls");
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernaryOperatorTwoNumbers (
    int testNum,
    const std::vector<const char *> &num,
    mpf_t *mpfIn,
    size_t mpfInLen) {

    // Copy mpfX and mpfY
    auto mpfCopy = std::make_unique<mpf_t[]>(mpfInLen);

    for (size_t i = 0; i < mpfInLen; ++i) {
        mpf_init(mpfCopy[i]);
        mpf_set(mpfCopy[i], mpfIn[i]);
    }

    ClearConsole();

    auto curTest = [&]() {
        // Print the original input values
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Original input strings:" << std::endl;

            for (size_t i = 0; i < num.size(); ++i) {
                std::cout << "num[" << i << "]: " << num[i] << std::endl;
            }

            for (size_t i = 0; i < mpfInLen; ++i) {
                std::cout << "mpfIn[" << i << "]: " << MpfToString<SharkFloatParams>(mpfIn[i], HpSharkFloat<SharkFloatParams>::DefaultPrecBits) << std::endl;
            }
            
            std::cout << "operator: " << OperatorToString<sharkOperator>() << std::endl;
        }

        // Convert the input values to HpSharkFloat<SharkFloatParams> representations
        std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy{ mpfInLen };

        assert(xNumCopy.size() == num.size());

        for (size_t i = 0; i < num.size(); ++i) {
            xNumCopy[i].MpfToHpGpu(mpfCopy[i], HpSharkFloat<SharkFloatParams>::DefaultPrecBits);
        }

        TestTernaryOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, false>(
            testNum, xNumCopy, mpfCopy.get(), mpfInLen);

        testNum++;
    };

    auto resetCopy = [&]() {
        for (size_t i = 0; i < mpfInLen; ++i) {
            mpf_init(mpfCopy[i]);
            mpf_set(mpfCopy[i], mpfIn[i]);
        }
    };

    auto printTest = [&](int curTest) {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Test " << std::dec << curTest << std::endl;
    };

    //
    // With three numbers, there are 8 combinations of signs
    // 
    
    if constexpr (EnableTestSign1) {
        printTest(testNum);
        resetCopy();
        curTest();
    }

    if constexpr (EnableTestSign2) {
        printTest(testNum);
        resetCopy();

        if constexpr (!SharkTestForceSameSign) {
            mpf_neg(mpfCopy[0], mpfCopy[0]);
        }
        curTest();
    }

    if constexpr (EnableTestSign3) {
        printTest(testNum);
        resetCopy();
        if constexpr (!SharkTestForceSameSign) {
            mpf_neg(mpfCopy[1], mpfCopy[1]);
        }
        curTest();
    }

    if constexpr (EnableTestSign4) {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfCopy[0], mpfCopy[0]);
        mpf_neg(mpfCopy[1], mpfCopy[1]);
    }

    if constexpr (EnableTestSign5) {
        printTest(testNum);
        resetCopy();
        if constexpr (!SharkTestForceSameSign) {
            mpf_neg(mpfCopy[2], mpfCopy[2]);
        }
        curTest();
    }

    if constexpr (EnableTestSign6) {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfCopy[0], mpfCopy[0]);
        mpf_neg(mpfCopy[2], mpfCopy[2]);
    }
    
    if constexpr (EnableTestSign7) {
        printTest(testNum);
        resetCopy();
        if constexpr (!SharkTestForceSameSign) {
            mpf_neg(mpfCopy[1], mpfCopy[1]);
            mpf_neg(mpfCopy[2], mpfCopy[2]);
        }
        curTest();
    }
    
    if constexpr (EnableTestSign8) {
        printTest(testNum);
        resetCopy();
        if constexpr (!SharkTestForceSameSign) {
            mpf_neg(mpfCopy[0], mpfCopy[0]);
            mpf_neg(mpfCopy[1], mpfCopy[1]);
            mpf_neg(mpfCopy[2], mpfCopy[2]);
        }
        curTest();
    }

    for (size_t i = 0; i < mpfInLen; ++i) {
        mpf_clear(mpfCopy[i]);
    }
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernaryOperatorTwoNumbers (
    int testNum,
    const char *num1,
    const char *num2,
    const char *num3) {

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << std::dec << testNum << std::endl;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);  // Set precision for MPIR floating point

    constexpr size_t NumMpfs = 3;
    mpf_t mpfs[NumMpfs];
    
    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_init(mpfs[i]);
    }

    auto res = mpf_set_str(mpfs[0], num1, 10);
    if (res == -1) {
        std::cout << "Error setting mpfX" << std::endl;
    }

    res = mpf_set_str(mpfs[1], num2, 10);
    if (res == -1) {
        std::cout << "Error setting mpfY" << std::endl;
    }

    res = mpf_set_str(mpfs[2], num3, 10);
    if (res == -1) {
        std::cout << "Error setting mpfZ" << std::endl;
    }

    {
        std::vector<const char *> strs(3);
        strs[0] = num1;
        strs[1] = num2;
        strs[2] = num3;

        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            testNum,
            strs,
            mpfs,
            NumMpfs);
    }

    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_clear(mpfs[i]);
    }
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    const HpSharkFloat<SharkFloatParams> &zNum,
    const HpSharkFloat<SharkFloatParams> &xNum2,
    const HpSharkFloat<SharkFloatParams> &yNum2) {

    static constexpr size_t NumMpfs = 5;
    std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy(NumMpfs);
    mpf_t mpfXCopy[NumMpfs];

    xNumCopy[0].DeepCopySameDevice(xNum);
    xNumCopy[1].DeepCopySameDevice(yNum);
    xNumCopy[2].DeepCopySameDevice(zNum);
    xNumCopy[3].DeepCopySameDevice(xNum2);
    xNumCopy[4].DeepCopySameDevice(yNum2);

    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_init(mpfXCopy[i]);
    }

    xNum.HpGpuToMpf(mpfXCopy[0]);
    yNum.HpGpuToMpf(mpfXCopy[1]);
    zNum.HpGpuToMpf(mpfXCopy[2]);
    xNum2.HpGpuToMpf(mpfXCopy[3]);
    yNum2.HpGpuToMpf(mpfXCopy[4]);

    TestTernaryOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, true>(
        testNum,
        xNumCopy,
        mpfXCopy,
        NumMpfs);

    // Clean up
    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_clear(mpfXCopy[i]);
    }
}


template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecialHelper(
    int testNum,
    const IntSignCombo &testData1,
    const IntSignCombo &testData2,
    const IntSignCombo &testData3,
    const IntSignCombo &testData4,
    const IntSignCombo &testData5) {
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << std::dec << testNum << std::endl;

    auto testData1Copy = testData1;
    testData1Copy.Digits.resize(SharkFloatParams::GlobalNumUint32);

    auto testData2Copy = testData2;
    testData2Copy.Digits.resize(SharkFloatParams::GlobalNumUint32);

    auto testData3Copy = testData3;
    testData3Copy.Digits.resize(SharkFloatParams::GlobalNumUint32);

    auto testData4Copy = testData4;
    testData4Copy.Digits.resize(SharkFloatParams::GlobalNumUint32);

    auto testData5Copy = testData5;
    testData5Copy.Digits.resize(SharkFloatParams::GlobalNumUint32);

    auto xNum{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData1Copy.Digits.data(),  testData1Copy.Exponent, testData1Copy.Negative) };
    auto yNum{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData2Copy.Digits.data(),  testData2Copy.Exponent, testData2Copy.Negative) };
    auto zNum{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData3Copy.Digits.data(),  testData3Copy.Exponent, testData3Copy.Negative) };
    auto xNum2{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData4Copy.Digits.data(), testData4Copy.Exponent, testData4Copy.Negative) };
    auto yNum2{ std::make_unique<HpSharkFloat<SharkFloatParams>>(testData5Copy.Digits.data(), testData5Copy.Exponent, testData5Copy.Negative) };

    TestTernarySpecial<SharkFloatParams, sharkOperator>(testNum, *xNum, *yNum, *zNum, *xNum2, *yNum2);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecialHelper(
    int testNum,
    const std::vector<uint32_t> &testData1,
    const std::vector<uint32_t> &testData2,
    const std::vector<uint32_t> &testData3
) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        testData1,
        testData2,
        testData3,
        testData1,   // Repeat
        testData2);  // Repeat
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial(
    int testNum,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    const HpSharkFloat<SharkFloatParams> &zNum) {

    TestTernarySpecial<SharkFloatParams, sharkOperator>(
        testNum,
        xNum,
        yNum,
        zNum,
        xNum,  // Repeat
        yNum); // Repeat
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial1(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0x80000000;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        testData,
        testData,
        testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial2(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0xC0000000;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        testData,
        testData,
        testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial3(int testNum) {
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0xFFFFFFFF;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        testData,
        testData,
        testData);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial4(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xF26D37FC, 0xA96025CE, 0xB03FC716, 0x1DF7182B, 0xCCBD69BD, 0x40C0F80C, 0xFAA0222E, 0xD1FDA456 },
        std::vector<uint32_t>{ 0x8BBCDF3, 0x4C3E7ACB, 0x6691A71D, 0xDFE03842, 0x3FADCA11, 0x4058BC9E, 0xF30FD7DE, 0xAA6CA582 },
        std::vector<uint32_t>{ 0xF26D37FC, 0xA96025CE, 0xB03FC716, 0x1DF7182B, 0xCCBD69BD, 0x40C0F80C, 0xFAA0222E, 0xD1FDA456 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial5(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial6(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial7(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial8(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0, 0, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial9(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x10 },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0xF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial10(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0, 0, 0x2, 0x3 },
        std::vector<uint32_t>{ 0, 0, 0, 0x5, 0x7 },
        std::vector<uint32_t>{ 0, 0, 0, 0x9, 0xb });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial11(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0, 0x2, 0, 0, 0, 0, 0x3 },
        std::vector<uint32_t>{ 0, 0x5, 0, 0, 0, 0, 0x7 },
        std::vector<uint32_t>{ 0, 0x9, 0, 0, 0, 0, 0xb });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial12(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0xf },
        std::vector<uint32_t>{ 0xFFFFFFF2, 0x10 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial13(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x11 },
        std::vector<uint32_t>{ 0xFFFFFFF2, 0x10 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial14(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFF000000, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x10 },
        std::vector<uint32_t>{ 0xFFFFFFF2, 0x10 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial15(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000000, 0, 0xFFFFFFF1, 0x00000008, 0x00000000, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0x00000000, 0, 0x00000000, 0x0000000D, 0x00000000, 0xFFFFFFF6, 0x0000000A, 0x00000003 },
        std::vector<uint32_t>{ 0x00000000, 0, 0xFFFFFFF1, 0x00000008, 0x00000000, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial16(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x0000000C, 0xFFFFFFF0, 0x00000000, 0xFFFFFFFC, 0x00000000, 0x0000000D, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0xFFFFFFFD, 0xFFFFFFEF, 0xFFFFFFEF, 0xFFFFFFF4, 0x00000000, 0x7A6650D9, 0x00000000, 0x00000000 },
        std::vector<uint32_t>{ 0x0000000C, 0xFFFFFFF0, 0x00000000, 0xFFFFFFFC, 0x00000000, 0x0000000D, 0xFFFFFFFF, 0x00000000 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial17(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF3, 0xFFFFFFF9, 0x00000004 },
        std::vector<uint32_t>{ 0x0000000E, 0x00000000, 0x00000000, 0xFFFFFFF2, 0x00000003, 0x00000000, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF3, 0xFFFFFFF9, 0x00000004 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial18(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x00000001, 0xFFFFFFFF, 0xFFFFFFFC, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFC, 0xE8CFC461, 0xFFFFFFF9 },
        std::vector<uint32_t>{ 0xFFFFFFF8, 0xD446522A, 0xFFFFFFFF, 0x00000010, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0x00000001, 0xFFFFFFFF, 0xFFFFFFFC, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFC, 0xE8CFC461, 0xFFFFFFF9 });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial19(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0x685940F0, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFF1, 0x5008CECF, 0x2A4D4784, 0x0000000D, 0x00000006, 0x00000000, 0xFFFFFFFF, 0x00000000 },
        std::vector<uint32_t>{ 0x685940F0, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial20(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFFFFFFFF, 0x556B0E43, 0x4EECA55A, 0x0000000E, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF8, 0x9B1194D6, 0xFFFFFFFF, 0x00000000, 0x13C1799F, 0x00000000, 0xC5F37A5D, 0xFFFFFFF4, 0x6FBC0EFF, 0x00000008, 0xFFFFFFFF, 0x00000000, 0xFFFFFFEF, 0xB06FA6C3, 0x0000000F, 0xFFFFFFF4, 0x00000007, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0x0503FC0B, 0xF26CA6A5, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000007, 0x00000010, 0xE640F2D9, 0x00000000, 0xFFFFFFF5, 0xFFFFFFFF, 0xFFFFFFF0, 0xFFFFFFFF, 0x00000004, 0x379A6DBB, 0xFFFFFFFF, 0x00000008, 0x00000002, 0xFFFFFFFF, 0x00000000, 0x0000000B, 0x00000000, 0xFFFFFFEF, 0xFFFFFFFF, 0x093E223D },
        std::vector<uint32_t>{ 0xFFFFFFFF, 0x556B0E43, 0x4EECA55A, 0x0000000E, 0xFFFFFFFF, 0x00000000, 0xFFFFFFF8, 0x9B1194D6, 0xFFFFFFFF, 0x00000000, 0x13C1799F, 0x00000000, 0xC5F37A5D, 0xFFFFFFF4, 0x6FBC0EFF, 0x00000008, 0xFFFFFFFF, 0x00000000, 0xFFFFFFEF, 0xB06FA6C3, 0x0000000F, 0xFFFFFFF4, 0x00000007, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial21(int testNum) {

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 0xFFFFFFFD, 0x0000000B, 0x00000000, 0xFFFFFFFE, 0x00000000, 0x88A881E4, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0x00000007, 0xD9B23983, 0x00000005, 0x00000000, 0xFFFFFFFF, 0x00000006, 0xFFFFFFFF, 0xFFFFFFFF },
        std::vector<uint32_t>{ 0xFFFFFFFD, 0x0000000B, 0x00000000, 0xFFFFFFFE, 0x00000000, 0x88A881E4, 0xFFFFFFFF, 0xFFFFFFFF });
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial21(int testNum, int exponentOverride2) {

    std::vector<uint32_t> allFs{};
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        allFs.push_back(0xFFFFFFFF);
    }
    allFs.resize(SharkFloatParams::GlobalNumUint32);

    std::vector<uint32_t> justOne{};
    justOne.push_back(1);
    justOne.resize(SharkFloatParams::GlobalNumUint32);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Test " << std::dec << testNum << ", exponentOverride2: " << exponentOverride2 << std::endl;

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>(allFs.data(), 0, false);
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>(justOne.data(), exponentOverride2, false);
    auto zNum = std::make_unique<HpSharkFloat<SharkFloatParams>>(allFs.data(), 0, false);

    TestTernarySpecial<SharkFloatParams, sharkOperator>(testNum, *xNum, *yNum, *zNum);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial22(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        IntSignCombo{ false, 0, { 5 } },
        IntSignCombo{ false, 0, { 17 } },
        IntSignCombo{ false, 0, { 0 } },
        IntSignCombo{ false, 0, { 5 } },
        IntSignCombo{ true,  0, { 17 } }
    );
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial23(int testNum) {
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        std::vector<uint32_t>{ 5 },
        std::vector<uint32_t>{ 17 },
        std::vector<uint32_t>{ 29 },
        std::vector<uint32_t>{ 57 },
        std::vector<uint32_t>{ 87 }
    );
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial24(int testNum) {

    IntSignCombo a{ true,  0, std::vector<uint32_t>{ 0x00000000, 0xFFFFFFFD, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFF, 0x00000002, 0xFFFFFFFF, 0xFFFFFFFF }};
    IntSignCombo b{ false, 0, std::vector<uint32_t>{ 0x8EB717E8, 0xFFFFFFFF, 0xA4D1162E, 0x0000000E, 0xC87AB0C2, 0x00000000, 0xFFFFFFFF, 0x00000000 }};
    IntSignCombo c{ true,  0, std::vector<uint32_t>{ 0xFFFFFFFD, 0xECE3ACF5, 0x0000000F, 0xFFFFFFFF, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF }};

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        a, b, c, a, b);
}

template<class SharkFloatParams, Operator sharkOperator>
void TestTernarySpecial25(int testNum) {

    IntSignCombo a{ true, 0, std::vector<uint32_t>{ 0x00000003, 0xfffffff3, 0x0000000e, 0xffffffff, 0x00000000, 0x00000000, 0xffffffff, 0xffffffff } };
    IntSignCombo b{ false, -59, std::vector<uint32_t>{ 0x00000000, 0x00000000, 0x00000000, 0xb0000000, 0x5fffffff, 0x78000000, 0xcfffffff, 0x87ffffff } };
    IntSignCombo c{ true, 0, std::vector<uint32_t>{ 0x00000009, 0x00000000, 0x577f96c7, 0x00000000, 0xfffffff9, 0x00000009, 0xffffffff, 0xfffffffe } };

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        testNum,
        a, b, c, a, b);
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

    // 2000s is multiply
    // 4000s is add
    //
    if constexpr (includeSet1) {
        const auto set = testBase + 100;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "7", "19", "0");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "4294967295", "1", "4294967296");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "4294967296", "1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "4294967295", "4294967296", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "4294967296", "-1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 60, "18446744073709551615", "1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 70, "0", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 80, "0.1", "0", "0.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 90, "0", "0", "0");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 100, "0.1", "0.1", "0.1");
    }

    if constexpr (includeSet2) {
        const auto set = testBase + 300;
        TestTernarySpecial1<SharkFloatParams, sharkOperator>(set + 10);
        TestTernarySpecial2<SharkFloatParams, sharkOperator>(set + 20);
        TestTernarySpecial3<SharkFloatParams, sharkOperator>(set + 30);
        TestTernarySpecial4<SharkFloatParams, sharkOperator>(set + 40);
        TestTernarySpecial5<SharkFloatParams, sharkOperator>(set + 50);
        TestTernarySpecial6<SharkFloatParams, sharkOperator>(set + 60);
        TestTernarySpecial7<SharkFloatParams, sharkOperator>(set + 70);
        TestTernarySpecial8<SharkFloatParams, sharkOperator>(set + 80);
        TestTernarySpecial9<SharkFloatParams, sharkOperator>(set + 90);
        TestTernarySpecial10<SharkFloatParams, sharkOperator>(set + 100);
        TestTernarySpecial11<SharkFloatParams, sharkOperator>(set + 110);
        TestTernarySpecial12<SharkFloatParams, sharkOperator>(set + 120);
        TestTernarySpecial13<SharkFloatParams, sharkOperator>(set + 130);
        TestTernarySpecial14<SharkFloatParams, sharkOperator>(set + 140);
        TestTernarySpecial15<SharkFloatParams, sharkOperator>(set + 150);
        TestTernarySpecial16<SharkFloatParams, sharkOperator>(set + 160);
        TestTernarySpecial17<SharkFloatParams, sharkOperator>(set + 170);
        TestTernarySpecial18<SharkFloatParams, sharkOperator>(set + 180);
        TestTernarySpecial19<SharkFloatParams, sharkOperator>(set + 190);
        TestTernarySpecial20<SharkFloatParams, sharkOperator>(set + 200);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(set + 210);
        TestTernarySpecial22<SharkFloatParams, sharkOperator>(set + 220);
        TestTernarySpecial23<SharkFloatParams, sharkOperator>(set + 230);
        TestTernarySpecial24<SharkFloatParams, sharkOperator>(set + 240);
        TestTernarySpecial25<SharkFloatParams, sharkOperator>(set + 250);
    }

    if constexpr (includeSet3) {
        const auto set = testBase + 600;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "2", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "0.2", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "0.5", "1.2", "1.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "0.6", "1.3", "1.9");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "0.7", "1.4", "2.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 60, "0.1", "1.99999999999999999999999999999", "2.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 70, "0.123124561464451654461", "1.2395123123127298375982735", "1.187236498176923871462938");
    }

    if constexpr (includeSet4) {
        const auto set = testBase + 700;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "-0.5", "1.2", "0.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "-0.6", "1.3", "0.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "-0.7", "1.4", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "-0.1", "1.99999999999999999999999999999", "0.9");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "-0.123124561464451654461", "1.2395123123127298375982735", "0.1");
    }

    if constexpr (includeSet5) {
        const auto set = testBase + 800;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "-0.51", "-1.29", "-1.49");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "-0.61", "-1.39", "-0.599");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "-0.71", "-1.49", "-0.799");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "-0.11", "-1.99999999999999999999999999999", "-0.89999999999999999999999999999");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "-0.123124561464451654461", "-1.2395123123127298375982735", "-1.1123877508482781861362735");
    }

    if constexpr (includeSet6) {
        const auto set = testBase + 900;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 10, "0.5265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216", "0.12987461239874619237469187236948716928374691827364");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 20, "0.2999999999965542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216", "1.12374861283467182367518476235481675234862e2334");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 30, "0.1265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866391730473289107302178039999999999999271839216", "1234671987263941876239487162398746e18239");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 40, "0.0265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216", "1023949123e389274");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(set + 50, "0.00000000000000000265542653452654526545625456254565446654545645649789871322131213156435546435", "-1.263468375787958774985473345435632415334245268476928454653443234164658776634854746584532186639173047328910730217803271839216", "7236.34234e5234523");
    }

    if constexpr (sharkOperator == Operator::Add && SharkFloatParams::GlobalNumUint32 == 8) {
        static constexpr auto SpecificTest1 = -129;
        static constexpr auto SpecificTest2 = -128;
        static constexpr auto SpecificTest3 = -127;
        static constexpr auto SpecificTest4 = 127;
        static constexpr auto SpecificTest5 = 255;
        static constexpr auto SpecificTest6 = 256;

        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest1);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest2);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest3);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest4);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest5);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, SpecificTest6);

        for (auto i = -512; i < 512; i++) {
            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "Exponent adjustment: " << i << std::endl;
            }
            
            TestTernarySpecial21<SharkFloatParams, sharkOperator>(0, i);
        }
    }

    if constexpr (includeSet10) {
        const auto set10 = testBase + 1000;
        auto x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto y = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto z = std::make_unique<HpSharkFloat<SharkFloatParams>>();

        for (auto i = 0; i < 1000; i += 10) {
            if (i % 2 == 0) {
                x->GenerateRandomNumber();
                y->GenerateRandomNumber();
                z->GenerateRandomNumber();
            } else {
                x->GenerateRandomNumber2();
                y->GenerateRandomNumber2();
                z->GenerateRandomNumber2();
            }

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->GetNegative() << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->GetNegative() << std::endl;
                std::cout << "z.Exponent: " << z->Exponent << ", neg: " << z->GetNegative() << std::endl;
            }

            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            const std::string z_str = z->ToString();

            TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
                set10 + i,
                x_str.c_str(),
                y_str.c_str(),
                z_str.c_str());
        }
    }

    if constexpr (includeSet11) {
        auto x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto y = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto z = std::make_unique<HpSharkFloat<SharkFloatParams>>();

        for (size_t counter = 0;; counter++) {
            if (counter % 2 == 0) {
                x->GenerateRandomNumber();
                y->GenerateRandomNumber();
                z->GenerateRandomNumber();
            } else {
                x->GenerateRandomNumber2();
                y->GenerateRandomNumber2();
                z->GenerateRandomNumber2();
            }

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "x.Exponent: " << x->Exponent << ", neg: " << x->GetNegative() << std::endl;
                std::cout << "y.Exponent: " << y->Exponent << ", neg: " << y->GetNegative() << std::endl;
                std::cout << "z.Exponent: " << z->Exponent << ", neg: " << z->GetNegative() << std::endl;
            }

            const std::string x_str = x->ToString();
            const std::string y_str = y->ToString();
            const std::string z_str = z->ToString();

            TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
                0,
                x_str.c_str(),
                y_str.c_str(),
                z_str.c_str());
        }
    }

    return Tests.CheckAllTestsPassed();
}

template<Operator sharkOperator>
bool TestBinaryOperatorPerf([[maybe_unused]] int testBase) {
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
    static constexpr auto NumIters = 5;
    for (size_t i = 0; i < NumIters; i++) {
        TestPerf<TestPerSharkParams1, sharkOperator>(testBase + 1, SharkTestIterCount);
    }
#endif
    return Tests.CheckAllTestsPassed();
}

// Explicitly instantiate TestAllBinaryOp
#ifdef MULTI_KERNEL
#define ExplicitlyInstantiate(SharkFloatParams) \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::Add>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase); \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::ReferenceOrbit>(int testBase);
#else
#define ExplicitlyInstantiate(SharkFloatParams) \
    /*template bool TestAllBinaryOp<SharkFloatParams, Operator::Add>(int testBase); */ \
    /*template bool TestAllBinaryOp<SharkFloatParams, Operator::MultiplyKaratsubaV2>(int testBase); */ \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::ReferenceOrbit>(int testBase);  
#endif

template bool TestBinaryOperatorPerf<Operator::Add>(int testBase);
template bool TestBinaryOperatorPerf<Operator::MultiplyKaratsubaV2>(int testBase);
template bool TestBinaryOperatorPerf<Operator::ReferenceOrbit>(int testBase);

ExplicitInstantiateAll();