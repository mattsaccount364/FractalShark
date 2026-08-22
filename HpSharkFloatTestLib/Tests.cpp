#include "BenchmarkTimer.h"
#include "DbgHeap.h"
#include "FractalViewPresets.h"
#include "GpuPrecisionDispatch.h"
#include "HpSharkFloat.h"
#include "HpSharkTestConfig.h"
#include "PrecisionCalculator.h"
#include "TestTracker.h"
#include "TestVerbose.h"

#include "DebugChecksumHost.h"
#include "PerfTimingResult.h"
#include "ReferenceReferenceOrbit2.h"
#include "Tests.h"

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "KernelInvoke.h"
#include "KernelInvokeReferenceCache.h"
#include "KernelInvokeReferenceSetup.h"

#include "Environment.h"

template <class SharkFloatParams>
bool ChecksumsCheck(const HpShark::LaunchParams &launchParams,
                    const DebugHostCombo<SharkFloatParams> &debugHostCombo,
                    const DebugGpuCombo &debugGpuCombo);

static constexpr bool EnableTestSign1 = true;
static constexpr bool EnableTestSign2 = true;
static constexpr bool EnableTestSign3 = true;
static constexpr bool EnableTestSign4 = true;
static constexpr bool EnableTestSign5 = true;
static constexpr bool EnableTestSign6 = true;
static constexpr bool EnableTestSign7 = true;
static constexpr bool EnableTestSign8 = true;
static constexpr auto TestOutputPrecisionBits = 32;

// x_(n + 1) = x_n * x_n - y_n * y_n + a
// y_(n + 1) = 2 * x_n * y_n + b

struct IntSignCombo {
    IntSignCombo(bool negative, int32_t exponent, std::vector<uint32_t> digits)
        : Negative{negative}, Exponent{exponent}, Digits{std::move(digits)}
    {
    }

    IntSignCombo(std::vector<uint32_t> digits) : Negative{}, Exponent{}, Digits{std::move(digits)} {}

    bool Negative;
    int32_t Exponent;
    std::vector<uint32_t> Digits;
};

template <class SharkFloatParams, Operator sharkOperator>
bool
DiffAgainstHostNonZero(const HpShark::LaunchParams &launchParams,
                       TestTracker &Tests,
                       int testNum,
                       int /*numTerms*/,
                       std::string hostCustomOrGpu,
                       const mpf_t mpfHostResult,
                       const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    bool testSucceeded = true;
    const mp_bitcnt_t comparisonPrecBits = mpf_get_prec(mpfHostResult);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl << hostCustomOrGpu << " result: " << std::endl;
        std::cout << gpuResult.ToString() << std::endl;
        std::cout << gpuResult.ToHexString() << std::endl;
    }

    // Convert gpuResult --> mpfXGpuResult
    mpf_t mpfXGpuResult;
    mpf_init2(mpfXGpuResult, comparisonPrecBits);
    gpuResult.HpGpuToMpf(mpfXGpuResult);

    // Compute absolute difference: mpfDiffAbs = |mpfHostResult - mpfXGpuResult|
    mpf_t mpfDiff, mpfDiffAbs;
    mpf_init2(mpfDiff, comparisonPrecBits);
    mpf_init2(mpfDiffAbs, comparisonPrecBits);
    mpf_sub(mpfDiff, mpfHostResult, mpfXGpuResult);
    mpf_abs(mpfDiffAbs, mpfDiff);

    // Converted GPU result
    if (SharkVerbose == VerboseMode::Debug) {
        // mpfHostResult:
        std::cout << "\nConverted host result (mpfHostResult):" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfHostResult,
                                                   HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;

        std::cout << "\nConverted " << hostCustomOrGpu << " result (mpfXGpuResult):" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfXGpuResult,
                                                   HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;

        // Print the differences
        std::cout << "\nDifference between host and " << hostCustomOrGpu << " results:" << std::endl;
        std::cout << MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits) << std::endl;
    }

    // Retrieve total precision bits:
    mp_bitcnt_t gpuPrecBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    mp_bitcnt_t margin = sizeof(uint32_t) * 8 * 3 + 2; // as before
    mp_bitcnt_t totalPrecBits = (gpuPrecBits > margin ? gpuPrecBits - margin : 1);

    // Compute epsilon = 2^(-totalPrecBits)
    mpf_t epsilon;
    mpf_init2(epsilon, totalPrecBits);
    mpf_set_ui(epsilon, 1);
    mpf_div_2exp(epsilon, epsilon, totalPrecBits); // epsilon = 1 / 2^totalPrecBits

    // Compute |host| into mpfAbsHost
    mpf_t mpfAbsHost;
    mpf_init2(mpfAbsHost, comparisonPrecBits);
    mpf_abs(mpfAbsHost, mpfHostResult);

    // compute floor(log2(1/err)) in high precision
    auto BitsOfError = [&](const mpf_t err) -> int {
        mpf_t invErr;
        if (mpf_sgn(err) == 0) {
            return static_cast<int>(totalPrecBits);
        }

        mpf_init2(invErr, comparisonPrecBits);
        mpf_ui_div(invErr, 1, err); // invErr = 1/err

        mp_exp_t exp;
        mpf_get_d_2exp(&exp, invErr);
        mpf_clear(invErr);
        return static_cast<int>(exp) - 1; // floor(log_2(invErr))
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
                          << "  |host| = "
                          << MpfToString<SharkFloatParams>(mpfAbsHost, TestOutputPrecisionBits)
                          << "  epsilon = "
                          << MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits)
                          << "\n  |host - gpu| = "
                          << MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits)
                          << "  Bits of error = " << bitsErrA << std::endl;
            }
            Tests.MarkSuccess(&launchParams, testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nFAIL (|host| <= epsilon but absolute error > epsilon):\n"
                      << "  |host| = "
                      << MpfToString<SharkFloatParams>(mpfAbsHost, TestOutputPrecisionBits) << std::endl
                      << "  epsilon      = "
                      << MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits) << std::endl
                      << "  |host - gpu| = "
                      << MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits) << std::endl
                      << "  Bits of error = " << bitsErrA << std::endl;
            Tests.MarkFailed(&launchParams,
                             testNum,
                             hostCustomOrGpu,
                             MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits),
                             MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits));
            testSucceeded = false;
        }
    }
    // CASE B: host is not "tiny," so do a normal relative-error check
    else {
        mpf_t relativeError;
        mpf_init2(relativeError, comparisonPrecBits);
        {
            mpf_t tmp;
            mpf_init2(tmp, comparisonPrecBits);
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
                          << MpfToString<SharkFloatParams>(relativeError, TestOutputPrecisionBits)
                          << std::endl
                          << "  epsilon            = "
                          << MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits) << std::endl
                          << "  Bits of error: " << bitsErrB << std::endl;
            }
            Tests.MarkSuccess(&launchParams, testNum, hostCustomOrGpu);
        } else {
            std::cerr << "\nFAIL (relative-error exceeds epsilon):\n"
                      << "  relativeError = "
                      << MpfToString<SharkFloatParams>(relativeError, TestOutputPrecisionBits)
                      << std::endl
                      << "  epsilon             = "
                      << MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits) << std::endl
                      << "  Bits of error: " << bitsErrB << std::endl;
            Tests.MarkFailed(&launchParams,
                             testNum,
                             hostCustomOrGpu,
                             MpfToString<SharkFloatParams>(relativeError, TestOutputPrecisionBits),
                             MpfToString<SharkFloatParams>(epsilon, TestOutputPrecisionBits));
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

template <class SharkFloatParams, Operator sharkOperator>
bool
DiffAgainstHost(const HpShark::LaunchParams &launchParams,
                TestTracker &Tests,
                int testNum,
                int numTerms, // 2 or 3
                std::string hostCustomOrGpu,
                const mpf_t mpfHostResult,
                const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    const mp_bitcnt_t comparisonPrecBits = mpf_get_prec(mpfHostResult);

    // 1) Optional verbose print of GPU result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl
                  << hostCustomOrGpu << " (GPU) result:\n"
                  << gpuResult.ToString() << std::endl
                  << gpuResult.ToHexString() << std::endl;
    }

    // 2) Convert host mpf_t --> HpSharkFloat via MpfToHpGpu
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Correct answer follows after converting to HpSharkFloat: " << std::endl;
    }

    auto hostShark = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    hostShark->MpfToHpGpu(
        mpfHostResult, HpSharkFloat<SharkFloatParams>::DefaultPrecBits, InjectNoiseInLowOrder::Disable);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl;
    }

    // 3) Build absolute-difference mpf: |host - gpu|
    mpf_t mpfXGpu;
    mpf_t mpfDiff;
    mpf_t mpfDiffAbs;

    mpf_init2(mpfXGpu, comparisonPrecBits);
    mpf_init2(mpfDiff, comparisonPrecBits);
    mpf_init2(mpfDiffAbs, comparisonPrecBits);

    gpuResult.HpGpuToMpf(mpfXGpu);
    mpf_sub(mpfDiff, mpfHostResult, mpfXGpu);
    mpf_abs(mpfDiffAbs, mpfDiff);

    // 4) Quick check: is host exactly zero?
    bool hostIsZero{false};
    {
        mpf_t mpfZero;
        mpf_init2(mpfZero, comparisonPrecBits);
        mpf_set_ui(mpfZero, 0);

        hostIsZero = (mpf_cmp(mpfHostResult, mpfZero) == 0);
        mpf_clear(mpfZero);
    }

    if (hostIsZero) {
        // ---- FALLBACK: absolute ULP-based threshold at GPU exponent ----
        mp_bitcnt_t P = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
        mpf_t eps;
        mpf_init2(eps, P);
        mpf_set_ui(eps, 1);

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "\nBefore fallback absolute-error threshold : "
                      << MpfToString<SharkFloatParams>(eps, TestOutputPrecisionBits) << std::endl;
            std::cout << "Absolute difference: "
                      << MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits) << std::endl;
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
                      << MpfToString<SharkFloatParams>(eps, TestOutputPrecisionBits) << std::endl;
            std::cout << "Absolute difference: "
                      << MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits) << std::endl;
        }

        bool ok = (mpf_cmp(mpfDiffAbs, eps) <= 0);

        if (ok) {
            Tests.MarkSuccess(&launchParams, testNum, hostCustomOrGpu);
        } else {
            std::string diffStr = MpfToString<SharkFloatParams>(mpfDiffAbs, TestOutputPrecisionBits);
            std::string threshStr = MpfToString<SharkFloatParams>(eps, TestOutputPrecisionBits);
            std::cerr << "\nError: absolute error \"" << diffStr << "\" > allowed \"" << threshStr
                      << "\"\n";
            Tests.MarkFailed(&launchParams, testNum, hostCustomOrGpu, diffStr, threshStr);
        }

        mpf_clear(eps);
        mpf_clear(mpfXGpu);
        mpf_clear(mpfDiff);
        mpf_clear(mpfDiffAbs);
        return ok;
    }

    mpf_clear(mpfXGpu);
    mpf_clear(mpfDiff);
    mpf_clear(mpfDiffAbs);

    return DiffAgainstHostNonZero<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, numTerms, hostCustomOrGpu, mpfHostResult, gpuResult);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestPerf(const HpShark::LaunchParams &launchParams,
         TestTracker &Tests,
         int testNum,
         const char *num1,
         const char *num2,
         const char *num3,
         const char *radiusY,
         const mpf_t mpfX,
         const mpf_t mpfY,
         const mpf_t mpfZ,
         const typename SharkFloatParams::Float &hdrRadiusY,
         uint64_t numIters,
         int64_t expectedPeriod,
         PeriodicityResult expectedResult,
         uint32_t actualPrecisionLimbs,
         bool useMT = true,
         PerfTimingResult *timingOut = nullptr,
         HpShark::ReferencePreparedTables<SharkFloatParams> *preparedTables = nullptr)
{

    // Print the original input values
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "Original input values:" << std::endl;
        std::cout << "num1: " << num1 << std::endl;
        std::cout << "X: "
                  << MpfToString<SharkFloatParams>(mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;
        std::cout << "num2: " << num2 << std::endl;
        std::cout << "Y: "
                  << MpfToString<SharkFloatParams>(mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;
        std::cout << "num3: " << num3 << std::endl;
        std::cout << "Z: "
                  << MpfToString<SharkFloatParams>(mpfZ, HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;
        std::cout << "radiusY: " << radiusY << std::endl;
    }

    auto desc = SharkFloatParams::GetDescription();
    std::cout << "\nTest " << testNum << ": " << OperatorToString<sharkOperator>() << " " << desc
              << std::endl;

    std::cout << "LaunchParams: " << launchParams.ToString() << std::endl;

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    auto resultNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    xNum->MpfToHpGpu(
        mpfX, HpSharkFloat<SharkFloatParams>::DefaultPrecBits, InjectNoiseInLowOrder::Enable);
    yNum->MpfToHpGpu(
        mpfY, HpSharkFloat<SharkFloatParams>::DefaultPrecBits, InjectNoiseInLowOrder::Enable);
    zNum->MpfToHpGpu(
        mpfZ, HpSharkFloat<SharkFloatParams>::DefaultPrecBits, InjectNoiseInLowOrder::Enable);

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

    mpf_t recurrenceX, recurrenceY;
    mpf_init(recurrenceX);
    mpf_init(recurrenceY);
    mpf_set(recurrenceX, mpfX);
    mpf_set(recurrenceY, mpfY);

    // Periodicity:
    mpf_t zx2, zy2;
    mpf_init(zx2);
    mpf_init(zy2);

    // Reference orbit:
    mpf_t tempX, tempY, xSquared, ySquared, twoXY;
    mpf_init(tempX);
    mpf_init(tempY);
    mpf_init(xSquared);
    mpf_init(ySquared);
    mpf_init(twoXY);

    BenchmarkTimer hostTimer;

    uint64_t hostIterationsExecuted = 0;
    PeriodicityResult hostPeriodicityResult = PeriodicityResult::Unknown;

    std::vector<typename SharkFloatParams::ReferenceIterT> hostReferenceOrbit;

    if constexpr (HpShark::TestMPIRImpl) {
        ScopedBenchmarkStopper hostStopper{hostTimer};

        typename SharkFloatParams::Float dzdcX{1};
        typename SharkFloatParams::Float dzdcY{0};

        const typename SharkFloatParams::Float cx_cast{mpfX};
        const typename SharkFloatParams::Float cy_cast{mpfY};

        const typename SharkFloatParams::Float HighTwo{2.0f};
        const typename SharkFloatParams::Float HighOne{1.0f};
        const typename SharkFloatParams::Float TwoFiftySix{256.0f};

        uint64_t keptIterationCounter = 0;

        // hostReferenceOrbit.push_back({typename SharkFloatParams::Float{0}, typename
        // SharkFloatParams::Float{0}}); // Initial value

        for (uint64_t i = 0; i < numIters; ++i) {
            // x_(n + 1) = x_n * x_n - y_n * y_n + a
            // y_(n + 1) = 2 * x_n * y_n + b

            typename SharkFloatParams::Float double_zx;
            typename SharkFloatParams::Float double_zy;

            if constexpr (SharkFloatParams::EnablePeriodicity) {
                double_zx = typename SharkFloatParams::Float{recurrenceX};
                double_zy = typename SharkFloatParams::Float{recurrenceY};
            }

            hostReferenceOrbit.push_back({typename SharkFloatParams::Float{recurrenceX},
                                          typename SharkFloatParams::Float{recurrenceY}});

            // Increment before periodicity
            keptIterationCounter++;

            if constexpr (SharkFloatParams::EnablePeriodicity) {
                // x^2+2*I*x*y-y^2
                // dzdc = 2.0 * z * dzdc + real(1.0);
                // dzdc = 2.0 * (zx + zy * i) * (dzdcX + dzdcY * i) + HighPrecision(1.0);
                // dzdc = 2.0 * (zx * dzdcX + zx * dzdcY * i + zy * i * dzdcX + zy * i * dzdcY * i) +
                // HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i
                // * dzdcX + 2.0 * zy * i * dzdcY * i + HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX
                // + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX - 2.0 * zy * dzdcY +
                // HighPrecision(1.0);
                //
                // dzdcX = 2.0 * zx * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0)
                // dzdcY = 2.0 * zx * dzdcY + 2.0 * zy * dzdcX

                HdrReduce(dzdcX);
                auto dzdcX1 = HdrAbs(dzdcX);

                HdrReduce(dzdcY);
                auto dzdcY1 = HdrAbs(dzdcY);

                HdrReduce(double_zx);
                auto zxCopy1 = HdrAbs(double_zx);

                HdrReduce(double_zy);
                auto zyCopy1 = HdrAbs(double_zy);

                typename SharkFloatParams::Float n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

                typename SharkFloatParams::Float r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
                auto n3 = hdrRadiusY * r0 * HighTwo;
                HdrReduce(n3);

                if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                    hostIterationsExecuted = keptIterationCounter;
                    hostPeriodicityResult = PeriodicityResult::PeriodFound;
                    break;
                } else {
                    auto dzdcXOrig = dzdcX;
                    dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                    dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
                }
            }

            mpf_add(tempX, recurrenceX, recurrenceY);
            mpf_sub(tempY, recurrenceX, recurrenceY);
            mpf_mul(xSquared, tempX, tempY);
            mpf_mul(twoXY, recurrenceX, recurrenceY);
            mpf_mul_ui(twoXY, twoXY, 2);          // 2xy
            mpf_add(recurrenceX, xSquared, mpfX); // (x + y) * (x - y) + a

            mpf_add(recurrenceY, twoXY, mpfY); // 2xy + b

            typename SharkFloatParams::Float tempZX = double_zx + cx_cast;
            typename SharkFloatParams::Float tempZY = double_zy + cy_cast;
            typename SharkFloatParams::Float zn_size = tempZX * tempZX + tempZY * tempZY;

            if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {

                //
                // Escaped
                //

                hostIterationsExecuted = keptIterationCounter;
                hostPeriodicityResult = PeriodicityResult::Escaped;
                break;
            }
        }

        if (hostPeriodicityResult == PeriodicityResult::Unknown) {
            hostIterationsExecuted = numIters;
            hostPeriodicityResult = PeriodicityResult::Continue;
        }

        hostTimer.StopTimer();

        std::cout << "Host iter timeMs: " << hostTimer.GetDeltaInMs() << std::endl;
        std::cout << "Host periodicity: " << PeriodicityStrResult(hostPeriodicityResult)
                  << ", iters=" << hostIterationsExecuted << std::endl;
        if (timingOut) {
            timingOut->hostMs = static_cast<double>(hostTimer.GetDeltaInMs());
        }
    }

    if constexpr (HpShark::TestGpu) {
        std::unique_ptr<DebugGpuCombo> debugGpuCombo;
        if constexpr (HpShark::DebugGlobalState) {
            debugGpuCombo = std::make_unique<DebugGpuCombo>();
        }

        auto CheckDiff = [&](const HpShark::LaunchParams &launchParams,
                             TestTracker &Tests,
                             const int testNum,
                             const int numTerms,
                             const char *hostCustomOrGpu,
                             const mpf_t &mpfHostResult,
                             const HpSharkFloat<SharkFloatParams> &gpuResult) {
            auto testSucceeded = DiffAgainstHost<SharkFloatParams, sharkOperator>(
                launchParams, Tests, testNum, numTerms, hostCustomOrGpu, mpfHostResult, gpuResult);
            if (!testSucceeded) {
                std::cout << "Perf correctness test failed" << std::endl;
            } else {
                std::cout << "Perf correctness test succeeded" << std::endl;
            }

            return testSucceeded;
        };

        {
            std::vector<typename SharkFloatParams::ReferenceIterT> hpSharkReferenceOrbit;
            uint64_t totalExecutedIters = 0;

            HpShark::GpuOrbitSession<SharkFloatParams> session = [&]() {
                if (preparedTables != nullptr) {
                    return HpShark::GpuOrbitSession<SharkFloatParams>(
                        launchParams, hdrRadiusY, *xNum, *yNum, *preparedTables, debugGpuCombo.get());
                }
                return HpShark::GpuOrbitSession<SharkFloatParams>(
                    launchParams, hdrRadiusY, *xNum, *yNum, actualPrecisionLimbs, debugGpuCombo.get());
            }();
            auto &combo = session.GetResults();

            {
                BenchmarkTimer timer;

                {
                    ScopedBenchmarkStopper stopper{timer};
                    for (;;) {
                        // Bound the number of iterations per kernel launch.  Do not exceed the
                        // MaxOutputIters limit.
                        constexpr auto MaxOutputIters =
                            HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters;
                        const uint64_t itersToRun = (numIters - totalExecutedIters > MaxOutputIters)
                                                        ? MaxOutputIters
                                                        : (numIters - totalExecutedIters);
                        assert(itersToRun > 0);
                        assert(itersToRun <= MaxOutputIters);

                        session.InvokeChunk(itersToRun);

                        totalExecutedIters += combo.OutputIterCount;

                        for (uint64_t i = 0; i < combo.OutputIterCount; ++i) {
                            hpSharkReferenceOrbit.push_back(combo.OutputIters[i]);
                        }

                        if (combo.OutputIterCount == 0 &&
                            combo.PeriodicityStatus == PeriodicityResult::Unknown) {
                            Tests.MarkFailed(&launchParams,
                                             testNum,
                                             "GPU reference orbit progress",
                                             "GPU reference orbit made no progress",
                                             "a nonzero iteration count or a terminal result");
                            break;
                        }

                        if constexpr (SharkFloatParams::EnablePeriodicity) {
                            if (combo.PeriodicityStatus == PeriodicityResult::PeriodFound ||
                                combo.PeriodicityStatus == PeriodicityResult::Escaped ||
                                combo.PeriodicityStatus == PeriodicityResult::Unknown || // error
                                totalExecutedIters >= numIters) {

                                // Sanity check: if periodicity is enabled, we should never exit
                                // with "Unknown"
                                if (combo.PeriodicityStatus == PeriodicityResult::Unknown) {
                                    Environment::DebugBreakpoint();
                                }
                            } else {
                                continue;
                            }

                            break;
                        } else {
                            if (totalExecutedIters >= numIters) {
                                break;
                            }
                        }
                    }
                }

                Tests.AddTime(testNum, timer.GetDeltaInMs());
                Tests.MarkSuccess(&launchParams, testNum, "GPU total time");
                std::cout << "GPU iter timeMs: " << timer.GetDeltaInMs() << std::endl;
                if (timingOut) {
                    timingOut->gpuMs = static_cast<double>(timer.GetDeltaInMs());
                }

                if (timer.GetDeltaInMs() != 0) {
                    std::cout << "Ratio: "
                              << static_cast<double>(hostTimer.GetDeltaInMs()) /
                                     static_cast<double>(timer.GetDeltaInMs())
                              << std::endl;
                }
            }

            const auto &gpuResultX = combo.ZReal;
            const auto &gpuResultY = combo.ZImag;

            // CPU HpSharkFloat-based reference orbit.
            std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>> cpuRefOrbitResult;
            if constexpr (HpShark::TestReferenceImpl) {
                DebugHostCombo<SharkFloatParams> debugHostCombo;

                BenchmarkTimer cpuRefTimer;
                {
                    ScopedBenchmarkStopper cpuRefStopper{cpuRefTimer};
                    cpuRefOrbitResult = ReferenceOrbit2Helper<SharkFloatParams>(xNum.get(),
                                                                                yNum.get(),
                                                                                hdrRadiusY,
                                                                                numIters,
                                                                                actualPrecisionLimbs,
                                                                                debugHostCombo,
                                                                                preparedTables);
                }

                std::cout << "CPU ref orbit time: " << cpuRefTimer.GetDeltaInMs()
                          << " ms, iters=" << cpuRefOrbitResult->IterationsExecuted << std::endl;
                if (timingOut) {
                    timingOut->cpuMs = static_cast<double>(cpuRefTimer.GetDeltaInMs());
                }
                std::cout << "CPU ref periodicity: "
                          << PeriodicityStrResult(cpuRefOrbitResult->PeriodResult) << std::endl;
            }

            if constexpr (IsReferenceOrbitOperator<sharkOperator>) {
                if constexpr (HpShark::TestMPIRImpl) {
                    if (hpSharkReferenceOrbit.size() != hostReferenceOrbit.size()) {
                        std::cout << "Error: Host and GPU reference orbit size mismatch: host="
                                  << hostReferenceOrbit.size() << " gpu=" << hpSharkReferenceOrbit.size()
                                  << std::endl;
                        Environment::DebugBreakpoint();
                    } else {
                        bool orbitMatch = true;
                        for (size_t i = 0; i < hostReferenceOrbit.size(); ++i) {
                            const auto &hostVal = hostReferenceOrbit[i];
                            const auto &gpuVal = hpSharkReferenceOrbit[i];

                            auto hostValX = hostVal.x;
                            auto hostValY = hostVal.y;
                            auto gpuValX = gpuVal.x;
                            auto gpuValY = gpuVal.y;

                            HdrReduce(hostValX);
                            HdrReduce(hostValY);
                            HdrReduce(gpuValX);
                            HdrReduce(gpuValY);

                            if (hostValX != gpuValX || hostValY != gpuValY) {
                                std::cout << "Error: Host and GPU reference orbit value mismatch at "
                                             "idx "
                                          << i << ": host.x=" << hostValX.template ToString<false>()
                                          << " host.y=" << hostValY.template ToString<false>()
                                          << " gpu.x=" << gpuValX.template ToString<false>()
                                          << " gpu.y=" << gpuValY.template ToString<false>()
                                          << std::endl;

                                // Show the delta
                                const auto deltaX = hostValX - gpuValX;
                                const auto deltaY = hostValY - gpuValY;

                                std::cout
                                    << "Delta: host.x - gpu.x = " << deltaX.template ToString<false>()
                                    << " host.y - gpu.y = " << deltaY.template ToString<false>()
                                    << std::endl;

                                orbitMatch = false;
                                Environment::DebugBreakpoint();
                                break;
                            }
                        }
                        if (orbitMatch) {
                            std::cout << "Host and GPU reference orbit match, length="
                                      << hostReferenceOrbit.size() << std::endl;
                        }
                    }

                    // Compare CPU HpSharkFloat-based reference orbit against MPIR host orbit
                    if constexpr (HpShark::TestReferenceImpl) {
                        if (cpuRefOrbitResult->Orbit.size() != hostReferenceOrbit.size()) {
                            std::cout << "Error: MPIR host and CPU ref orbit size mismatch: mpir="
                                      << hostReferenceOrbit.size()
                                      << " cpuRef=" << cpuRefOrbitResult->Orbit.size() << std::endl;
                            Environment::DebugBreakpoint();
                        } else {
                            bool cpuOrbitMatch = true;
                            for (size_t i = 0; i < hostReferenceOrbit.size(); ++i) {
                                auto hostValX = hostReferenceOrbit[i].x;
                                auto hostValY = hostReferenceOrbit[i].y;
                                auto cpuValX = cpuRefOrbitResult->Orbit[i].x;
                                auto cpuValY = cpuRefOrbitResult->Orbit[i].y;

                                HdrReduce(hostValX);
                                HdrReduce(hostValY);
                                HdrReduce(cpuValX);
                                HdrReduce(cpuValY);

                                if (hostValX != cpuValX || hostValY != cpuValY) {
                                    std::cout << "Error: MPIR host and CPU ref orbit mismatch at "
                                                 "idx "
                                              << i << ": mpir.x=" << hostValX.template ToString<false>()
                                              << " mpir.y=" << hostValY.template ToString<false>()
                                              << " cpu.x=" << cpuValX.template ToString<false>()
                                              << " cpu.y=" << cpuValY.template ToString<false>()
                                              << std::endl;
                                    cpuOrbitMatch = false;
                                    Environment::DebugBreakpoint();
                                    break;
                                }
                            }
                            if (cpuOrbitMatch) {
                                std::cout << "MPIR host and CPU ref orbit match, length="
                                          << hostReferenceOrbit.size() << std::endl;
                            }
                        }

                        // Compare periodicity results
                        if (cpuRefOrbitResult->PeriodResult != hostPeriodicityResult) {
                            std::cout
                                << "Error: CPU ref periodicity mismatch: mpir="
                                << PeriodicityStrResult(hostPeriodicityResult)
                                << " cpuRef=" << PeriodicityStrResult(cpuRefOrbitResult->PeriodResult)
                                << std::endl;
                            Environment::DebugBreakpoint();
                        }

                        if (cpuRefOrbitResult->IterationsExecuted != hostIterationsExecuted) {
                            std::cout << "Error: CPU ref iteration count mismatch: mpir="
                                      << hostIterationsExecuted
                                      << " cpuRef=" << cpuRefOrbitResult->IterationsExecuted
                                      << std::endl;
                            Environment::DebugBreakpoint();
                        }
                    }

                    bool testSucceeded = true;
                    constexpr auto numTerms = 2;
                    testSucceeded &= CheckDiff(
                        launchParams, Tests, testNum, numTerms, "GPU_A", mpfHostResultXX, gpuResultX);
                    testSucceeded &= CheckDiff(
                        launchParams, Tests, testNum, numTerms, "GPU_B", mpfHostResultYY, gpuResultY);

                    if ((combo.PeriodicityStatus != hostPeriodicityResult) ||
                        (totalExecutedIters != hostIterationsExecuted)) {

                        std::cout << "Periodicity status: "
                                  << PeriodicityStrResult(combo.PeriodicityStatus) << std::endl;
                        std::cout << "Escape iteration mismatch: host=" << hostIterationsExecuted
                                  << " gpu=" << totalExecutedIters << std::endl;
                        Environment::DebugBreakpoint();
                    } else {
                        std::cout << "Periodicity status: "
                                  << PeriodicityStrResult(combo.PeriodicityStatus) << std::endl;
                        std::cout << "Output iteration: " << totalExecutedIters << std::endl;
                    }
                } else {
                    std::cout << "Periodicity status: " << PeriodicityStrResult(combo.PeriodicityStatus)
                              << std::endl;
                    std::cout << "Output iteration: " << totalExecutedIters << std::endl;
                }

                // Direct CPU ref orbit vs GPU orbit comparison (independent of TestMPIRImpl)
                if constexpr (HpShark::TestReferenceImpl) {
                    if (cpuRefOrbitResult->Orbit.size() != hpSharkReferenceOrbit.size()) {
                        std::cout << "Error: CPU ref and GPU orbit size mismatch: cpuRef="
                                  << cpuRefOrbitResult->Orbit.size()
                                  << " gpu=" << hpSharkReferenceOrbit.size() << std::endl;
                        Environment::DebugBreakpoint();
                    } else {
                        bool cpuGpuOrbitMatch = true;
                        for (size_t i = 0; i < cpuRefOrbitResult->Orbit.size(); ++i) {
                            auto cpuValX = cpuRefOrbitResult->Orbit[i].x;
                            auto cpuValY = cpuRefOrbitResult->Orbit[i].y;
                            auto gpuValX = hpSharkReferenceOrbit[i].x;
                            auto gpuValY = hpSharkReferenceOrbit[i].y;

                            HdrReduce(cpuValX);
                            HdrReduce(cpuValY);
                            HdrReduce(gpuValX);
                            HdrReduce(gpuValY);

                            if (cpuValX != gpuValX || cpuValY != gpuValY) {
                                std::cout << "Error: CPU ref and GPU orbit mismatch at idx " << i
                                          << ": cpu.x=" << cpuValX.template ToString<false>()
                                          << " cpu.y=" << cpuValY.template ToString<false>()
                                          << " gpu.x=" << gpuValX.template ToString<false>()
                                          << " gpu.y=" << gpuValY.template ToString<false>()
                                          << std::endl;
                                cpuGpuOrbitMatch = false;
                                Environment::DebugBreakpoint();
                                break;
                            }
                        }
                        if (cpuGpuOrbitMatch) {
                            std::cout << "CPU ref and GPU orbit match, length="
                                      << cpuRefOrbitResult->Orbit.size() << std::endl;
                        }
                    }

                    // Direct CPU ref vs GPU periodicity/iteration comparison
                    if (cpuRefOrbitResult->PeriodResult != combo.PeriodicityStatus) {
                        std::cout << "Error: CPU ref vs GPU periodicity mismatch: cpuRef="
                                  << PeriodicityStrResult(cpuRefOrbitResult->PeriodResult)
                                  << " gpu=" << PeriodicityStrResult(combo.PeriodicityStatus)
                                  << std::endl;
                        Environment::DebugBreakpoint();
                    }

                    if (cpuRefOrbitResult->IterationsExecuted != totalExecutedIters) {
                        std::cout << "Error: CPU ref vs GPU iteration count mismatch: cpuRef="
                                  << cpuRefOrbitResult->IterationsExecuted
                                  << " gpu=" << totalExecutedIters << std::endl;
                        Environment::DebugBreakpoint();
                    }
                }

                if (expectedPeriod != -1 && static_cast<uint64_t>(expectedPeriod) <= numIters) {
                    if ((combo.PeriodicityStatus != expectedResult) ||
                        (totalExecutedIters != static_cast<uint64_t>(expectedPeriod))) {
                        std::cout << "Error: Expected result iters " << expectedPeriod << " but got "
                                  << totalExecutedIters << std::endl;
                        Environment::DebugBreakpoint();
                    }
                }

                if constexpr (HpShark::DebugGlobalState) {
                    DebugHostCombo<SharkFloatParams> debugHostCombo;
                    (void)ChecksumsCheck<SharkFloatParams>(launchParams, debugHostCombo, *debugGpuCombo);
                }
            }
        }
    }

    // Clean up MPIR variables
    mpf_clear(mpfHostResultXX);
    mpf_clear(mpfHostResultXY1);
    mpf_clear(mpfHostResultXY2);
    mpf_clear(mpfHostResultYY);

    // Clean up reference orbit variables
    mpf_clear(recurrenceX);
    mpf_clear(recurrenceY);

    mpf_clear(zx2);
    mpf_clear(zy2);

    mpf_clear(tempX);
    mpf_clear(tempY);
    mpf_clear(xSquared);
    mpf_clear(ySquared);
    mpf_clear(twoXY);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestPerfRandom(const HpShark::LaunchParams &launchParams,
               TestTracker &Tests,
               int testNum,
               uint64_t numIters)
{
    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto yNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zNum = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    xNum->GenerateRandomNumber2();
    yNum->GenerateRandomNumber2();
    zNum->GenerateRandomNumber2();

    mpf_set_default_prec(
        HpSharkFloat<SharkFloatParams>::DefaultMpirBits); // Set precision for MPIR floating point

    mpf_t mpfX;
    mpf_t mpfY;
    mpf_t mpfZ;

    mpf_init(mpfX);
    mpf_init(mpfY);
    mpf_init(mpfZ);

    xNum->HpGpuToMpf(mpfX);
    yNum->HpGpuToMpf(mpfY);
    zNum->HpGpuToMpf(mpfZ);

    auto num1 = xNum->ToString();
    auto num2 = yNum->ToString();
    auto num3 = zNum->ToString();

    const auto unknownPeriod = -1;
    const auto expectedResult = PeriodicityResult::Continue;
    auto preparedTables = HpShark::PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(
        launchParams, *xNum, *yNum, SharkFloatParams::GlobalNumUint32, testNum);

    TestPerf<SharkFloatParams, sharkOperator>(launchParams,
                                              Tests,
                                              testNum,
                                              num1.c_str(),
                                              num2.c_str(),
                                              num3.c_str(),
                                              "0.0",
                                              mpfX,
                                              mpfY,
                                              mpfZ,
                                              typename SharkFloatParams::Float{},
                                              numIters,
                                              unknownPeriod,
                                              expectedResult,
                                              SharkFloatParams::GlobalNumUint32,
                                              true,
                                              nullptr,
                                              preparedTables.get());

    mpf_clear(mpfX);
    mpf_clear(mpfY);
    mpf_clear(mpfZ);
}

template <class SharkFloatParams, Operator sharkOperator>
bool
CheckAgainstHost(const HpShark::LaunchParams &launchParams,
                 TestTracker &Tests,
                 int testNum,
                 int numTerms,
                 const char *name,
                 const mpf_t mpfHostResult,
                 const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    bool res = DiffAgainstHost<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, numTerms, name, mpfHostResult, gpuResult);
    if (!res) {
        Environment::DebugBreakpoint();
    };

    return res;
}

template <class SharkFloatParams>
bool
ChecksumsCheck(const HpShark::LaunchParams &launchParams,
               const DebugHostCombo<SharkFloatParams> &debugHostCombo,
               const DebugGpuCombo &debugGpuCombo)
{
    // Compare debugResultsCuda against debugResultsHost
    bool ChecksumFailure = false;
    if constexpr (HpShark::TestGpu && HpShark::DebugGlobalState) {
        std::map<uint64_t, int> countOfCountsMultiply;
        for (size_t i = 0; i < debugGpuCombo.MultiplyCounts.size(); ++i) {
            countOfCountsMultiply[debugGpuCombo.MultiplyCounts[i].multiplyCount]++;
        }

        uint64_t carryCount = 0;
        for (size_t i = 0; i < debugGpuCombo.MultiplyCounts.size(); ++i) {
            carryCount += debugGpuCombo.MultiplyCounts[i].carryCount;
        }

        uint64_t normalizeCount = 0;
        for (size_t i = 0; i < debugGpuCombo.MultiplyCounts.size(); ++i) {
            normalizeCount += debugGpuCombo.MultiplyCounts[i].normalizeCount;
        }

        // Print distribution of counts
        uint64_t totalMultiplyCountGpu{};
        uint64_t totalMultiplyCountResults{};
        std::cerr << "MultiplyCount distribution:" << std::endl;
        for (const auto &pair : countOfCountsMultiply) {
            std::cerr << "Count: " << pair.first << " occurred " << pair.second << " times" << std::endl;
            totalMultiplyCountGpu += pair.first * pair.second;
            totalMultiplyCountResults += pair.second;
        }

        std::cerr << "GPU total carry count: " << carryCount << std::endl;
        std::cerr << "GPU total normalize count: " << normalizeCount << std::endl;

        std::cerr << "GPU total multiply count: " << totalMultiplyCountGpu << std::endl;
        std::cerr << "GPU result count (should be total num threads): " << totalMultiplyCountResults
                  << std::endl;
        std::cerr << "Host count: " << debugHostCombo.MultiplyCounts.multiplyCount << std::endl;

        if (totalMultiplyCountGpu != debugHostCombo.MultiplyCounts.multiplyCount) {
            std::cerr << "Error: GPU total count does not match host count!" << std::endl;
            ChecksumFailure = true;
            Environment::DebugBreakpoint();
        }

        if (totalMultiplyCountResults != launchParams.TotalThreads) {
            std::cerr << "Error: Total results does not match expected number of threads!" << std::endl;
            ChecksumFailure = true;
            Environment::DebugBreakpoint();
        }

        // Print full array
        // if (ChecksumFailure) {
        //    for (size_t i = 0; i < debugGpuCombo.MultiplyCounts.size(); ++i) {
        //        std::cerr << "MultiplyCount[" << i << "]: ";
        //        std::cerr << "Block: " << debugGpuCombo.MultiplyCounts[i].blockIdx << ", ";
        //        std::cerr << "Thread: " << debugGpuCombo.MultiplyCounts[i].threadIdx << ", ";
        //        std::cerr << "Multiply count: " << debugGpuCombo.MultiplyCounts[i].multiplyCount <<
        //        std::endl; std::cerr << "Carry count: " << debugGpuCombo.MultiplyCounts[i].carryCount
        //        << std::endl;
        //    }

        //    DebugBreak();
        //}
    }

    if constexpr (HpShark::TestGpu && HpShark::DebugChecksums) {
        const auto &debugResultsHost = debugHostCombo.States;
        if (debugResultsHost.size() > debugGpuCombo.States.size()) {
            std::cerr << "Error: GPU checksum table has " << debugGpuCombo.States.size()
                      << " slots, but the host table has " << debugResultsHost.size() << std::endl;
            ChecksumFailure = true;
        }

        // Note that the hosts results should be exactly the right size, whereas
        // the CUDA results may be larger due to the way the kernel is written.
        const size_t comparableStateCount =
            std::min(debugResultsHost.size(), debugGpuCombo.States.size());
        for (size_t i = 0; i < comparableStateCount; ++i) {
            const auto &host = debugResultsHost[i];
            const auto &cuda = debugGpuCombo.States[i];

            const bool hostInitialized = host.Initialized;
            const bool cudaInitialized = cuda.Initialized == 1;
            const size_t hostArraySize = host.ArrayToChecksum32.size() + host.ArrayToChecksum64.size();
            const bool metadataMatches = host.ChecksumPurpose == cuda.ChecksumPurpose &&
                                         host.RecursionDepth == cuda.RecursionDepth &&
                                         host.CallIndex == cuda.CallIndex &&
                                         host.Convolution == cuda.Convolution;
            const bool stateMatches = hostInitialized == cudaInitialized && metadataMatches &&
                                      host.Checksum == cuda.Checksum && hostArraySize == cuda.ArraySize;

            if (!stateMatches) {

                std::cerr << "======================================" << std::endl;
                std::cerr << "Error: Checksum mismatch at index(base16) 0x" << std::hex << i
                          << std::endl;
                std::cerr << "Expected slot: "
                          << DebugStatePurposeToString(static_cast<DebugStatePurpose>(i)) << std::endl;
                std::cerr << "GPU:" << std::endl;

                // Print all fields of cuda:
                std::cerr << std::dec;
                std::cerr << "Initialized: " << cuda.Initialized << std::endl;
                std::cerr << "Block: " << cuda.Block << std::endl;
                std::cerr << "Thread: " << cuda.Thread << std::endl;
                std::cerr << "ArraySize: " << cuda.ArraySize << std::endl;

                std::cerr << "Checksum(base16): 0x" << std::hex << cuda.Checksum << std::dec
                          << std::endl;
                if (cudaInitialized) {
                    std::cerr << "ChecksumPurpose: " << static_cast<int>(cuda.ChecksumPurpose)
                              << std::endl;
                    std::cerr << "ChecksumPurpose: " << DebugStatePurposeToString(cuda.ChecksumPurpose)
                              << std::endl;
                } else {
                    std::cerr << "Checkpoint status: not written (erased)" << std::endl;
                }

                std::cerr << "RecursionDepth: " << cuda.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << cuda.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(cuda.Convolution) << std::endl;

                // Print all fields of host
                std::cerr << std::endl;
                std::cerr << "Host reference implementation:" << std::endl;
                std::cerr << "Initialized: " << host.Initialized << std::endl;
                std::cerr << "ArrayToChecksum32(base16 words): " << std::endl;
                for (size_t j = 0; j < host.ArrayToChecksum32.size(); ++j) {
                    std::cerr << std::hex << "0x" << host.ArrayToChecksum32[j] << std::dec << " ";
                }

                std::cerr << std::endl;
                std::cerr << "ArrayToChecksum32 length: " << host.ArrayToChecksum32.size() << std::endl;

                std::cerr << "ArrayToChecksum64(base16 words): " << std::endl;
                for (size_t j = 0; j < host.ArrayToChecksum64.size(); ++j) {
                    std::cerr << std::hex << "0x" << host.ArrayToChecksum64[j] << std::dec << " ";
                }

                std::cerr << std::endl;
                std::cerr << "ArrayToChecksum64 length: " << host.ArrayToChecksum64.size() << std::endl;

                std::cerr << "Checksum(base16): 0x" << std::hex << host.Checksum << std::dec
                          << std::endl;
                if (hostInitialized) {
                    std::cerr << "ChecksumPurpose: " << static_cast<int>(host.ChecksumPurpose)
                              << std::endl;
                    std::cerr << "ChecksumPurpose: " << DebugStatePurposeToString(host.ChecksumPurpose)
                              << std::endl;
                } else {
                    std::cerr << "Checkpoint status: not written (erased)" << std::endl;
                }

                std::cerr << "RecursionDepth: " << host.RecursionDepth << std::endl;
                std::cerr << "CallIndex: " << host.CallIndex << std::endl;
                std::cerr << "Convolution: " << static_cast<int>(host.Convolution) << std::endl;

                ChecksumFailure = true;

                Environment::DebugBreakpoint();
            }
        }
    }

    if (ChecksumFailure) {
        std::cerr << "Checksum test failed" << std::endl;
        Environment::DebugBreakpoint();
    } else if constexpr (HpShark::DebugChecksums) {
        std::cout << "Checksum test passed" << std::endl;
    } else {
        std::cout << "Checksum test skipped (disabled in this build)" << std::endl;
    }

    return !ChecksumFailure;
}

template <class SharkFloatParams, Operator sharkOperator>
bool
CheckGPUResult(const HpShark::LaunchParams &launchParams,
               TestTracker &Tests,
               int testNum,
               int numTerms,
               const char *name,
               const mpf_t &mpfHostResult,
               const HpSharkFloat<SharkFloatParams> &gpuResult)
{
    auto testSucceeded = DiffAgainstHost<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, numTerms, name, mpfHostResult, gpuResult);

    if (SharkVerbose == VerboseMode::Debug) {
        if (!testSucceeded) {
            std::cout << "GPU High Precision failed" << std::endl;
        } else {
            std::cout << "GPU High Precision succeeded" << std::endl;
        }
    }

    if (!testSucceeded) {
        // If the test failed, we should break into the debugger
        Environment::DebugBreakpoint();
    }

    return testSucceeded;
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestCoreReferenceOrbit(const HpShark::LaunchParams &launchParams,
                       TestTracker &Tests,
                       int testNum,
                       const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
                       const mpf_t *mpfInputX,
                       size_t mpfInputLen,
                       uint32_t reference2MinFusedStages,
                       uint32_t reference2MaxFusedStages)
{
    static_assert(IsReferenceOrbitOperator<sharkOperator>,
                  "TestCoreReferenceOrbit requires a reference-orbit operator");

    (void)mpfInputLen; // Unused parameter, but kept for compatibility
    assert(inputX.size() == 3 || inputX.size() == 5);
    assert(mpfInputLen >= 2);

    const auto &aNum = inputX[0];
    const auto &bNum = inputX[1];

    assert(inputX.size() == mpfInputLen);
    const auto &mpfA = mpfInputX[0];
    const auto &mpfB = mpfInputX[1];

    const auto &mpfX = mpfInputX[0]; // NOTE!  Same index
    const auto &mpfY = mpfInputX[1];

    // This is a reference orbit calculation.  The first iteration basically
    // just copies the constants into the results.
    // x_(n + 1) = x_n * x_n - y_n * y_n + a
    // y_(n + 1) = 2 * x_n * y_n + b
    // The second iteration is more interesting, as it uses the results of the first iteration
    // to calculate the next iteration.
    // So here, we essentially set things up with the first iteration done, and then
    // calculate the second iteration.

    // Perform the calculation on the using MPIR
    auto gpuResultXX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto gpuResultYY = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    mpf_t mpfHostResultX;
    mpf_t mpfHostResultY;

    mpf_t twoXY;
    mpf_t tempX;
    mpf_t tempY;

    constexpr mp_bitcnt_t hostCalculationPrecBits =
        2 * HpSharkFloat<SharkFloatParams>::DefaultPrecBits + 4;
    mpf_init2(twoXY, hostCalculationPrecBits);
    mpf_init2(tempX, hostCalculationPrecBits);
    mpf_init2(tempY, hostCalculationPrecBits);

    mpf_init2(mpfHostResultX, hostCalculationPrecBits);
    mpf_init2(mpfHostResultY, hostCalculationPrecBits);

    // x_(n + 1) = x_n * x_n - y_n * y_n + a
    // y_(n + 1) = 2 * x_n * y_n + b

    mpf_add(tempX, mpfX, mpfY);
    mpf_sub(tempY, mpfX, mpfY);
    mpf_mul(mpfHostResultX, tempX, tempY); // (x + y) * (x - y)
    mpf_add(mpfHostResultX, mpfHostResultX, mpfA);

    mpf_mul(twoXY, mpfX, mpfY);           // xy
    mpf_mul_ui(twoXY, twoXY, 2);          // 2xy
    mpf_add(mpfHostResultY, twoXY, mpfB); // 2xy + b

    // Print host result
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nCorrect MPIR result:" << std::endl;
        std::cout << "Correct MPIR result X: "
                  << MpfToString<SharkFloatParams>(mpfHostResultX,
                                                   HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;
        std::cout << "Correct MPIR result Y: "
                  << MpfToString<SharkFloatParams>(mpfHostResultY,
                                                   HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                  << std::endl;

        std::cout << "Correct MPIR hex X: " << std::endl;
        std::cout << "" << MpfToHex32String(mpfHostResultX) << std::endl;
        std::cout << "Correct MPIR hex Y: " << std::endl;
        std::cout << "" << MpfToHex32String(mpfHostResultY) << std::endl;
    }

    DebugHostCombo<SharkFloatParams> debugHostCombo{};
    DebugGpuCombo debugGpuCombo{};
    typename SharkFloatParams::Float emptyRadius{};
    auto preparedTables =
        HpShark::PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                                       aNum,
                                                                       bNum,
                                                                       SharkFloatParams::GlobalNumUint32,
                                                                       testNum,
                                                                       0,
                                                                       reference2MinFusedStages,
                                                                       reference2MaxFusedStages);

    // Test the HpSharkFloat CPU reference implementation against MPIR.
    if constexpr (HpShark::TestReferenceImpl) {
        auto cpuResult = ReferenceOrbit2Helper<SharkFloatParams>(&aNum,
                                                                 &bNum,
                                                                 emptyRadius,
                                                                 1,
                                                                 SharkFloatParams::GlobalNumUint32,
                                                                 debugHostCombo,
                                                                 preparedTables.get());

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "CPU ref orbit result X: " << cpuResult->FinalZReal.ToString() << std::endl;
            std::cout << "CPU ref orbit result X hex: " << cpuResult->FinalZReal.ToHexString()
                      << std::endl;
            std::cout << "CPU ref orbit result Y: " << cpuResult->FinalZImag.ToString() << std::endl;
            std::cout << "CPU ref orbit result Y hex: " << cpuResult->FinalZImag.ToHexString()
                      << std::endl;
        }

        bool testSucceeded = true;
        constexpr auto numTerms = 2;
        testSucceeded &= CheckAgainstHost<SharkFloatParams, sharkOperator>(launchParams,
                                                                           Tests,
                                                                           testNum,
                                                                           numTerms,
                                                                           "ReferenceOrbitX",
                                                                           mpfHostResultX,
                                                                           cpuResult->FinalZReal);

        testSucceeded &= CheckAgainstHost<SharkFloatParams, sharkOperator>(launchParams,
                                                                           Tests,
                                                                           testNum,
                                                                           numTerms,
                                                                           "ReferenceOrbitY",
                                                                           mpfHostResultY,
                                                                           cpuResult->FinalZImag);

        if (SharkVerbose == VerboseMode::Debug) {
            if (!testSucceeded) {
                std::cout << "Custom High Precision failed" << std::endl;
            } else {
                std::cout << "Custom High Precision succeeded" << std::endl;
            }
        }
    }

    if constexpr (HpShark::TestGpu) {
        BenchmarkTimer timer;

        auto combo = std::make_unique<HpSharkReferenceResults<SharkFloatParams>>();
        combo->CReal = aNum;
        combo->CImag = bNum;
        combo->ZReal = aNum;
        combo->ZImag = bNum;
        combo->RadiusY = {};

        HpShark::InvokeHpSharkReferenceKernelCorrectness<SharkFloatParams>(
            launchParams, timer, *combo, &debugGpuCombo, *preparedTables);

        *gpuResultXX = combo->ZReal;
        *gpuResultYY = combo->ZImag;
        Tests.AddTime(testNum, timer.GetDeltaInMs());

        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "GPU reference timeMs: " << timer.GetDeltaInMs() << std::endl;
        }
    }

    if constexpr (HpShark::TestReferenceImpl) {
        if (!ChecksumsCheck<SharkFloatParams>(launchParams, debugHostCombo, debugGpuCombo)) {
            Tests.MarkFailed(&launchParams, testNum, "Checksums", "checksum mismatch", "exact match");
        }
    }

    if constexpr (HpShark::TestGpu && IsReferenceOrbitOperator<sharkOperator>) {
        bool testSucceeded = true;

        constexpr auto numTerms = 2;

        testSucceeded &= CheckGPUResult<SharkFloatParams, sharkOperator>(
            launchParams, Tests, testNum, numTerms, "GPU", mpfHostResultX, *gpuResultXX);

        testSucceeded &= CheckGPUResult<SharkFloatParams, sharkOperator>(
            launchParams, Tests, testNum, numTerms, "GPU", mpfHostResultY, *gpuResultYY);
    }

    // Clean up MPIR variables
    mpf_clear(twoXY);
    mpf_clear(tempX);
    mpf_clear(tempY);

    mpf_clear(mpfHostResultX);
    mpf_clear(mpfHostResultY);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernaryOperatorTwoNumbersRawNoSignChange(const HpShark::LaunchParams &launchParams,
                                             TestTracker &Tests,
                                             int testNum,
                                             const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
                                             const mpf_t *mpfInputX,
                                             size_t mpfInputLen,
                                             uint32_t reference2MinFusedStages,
                                             uint32_t reference2MaxFusedStages)
{

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << "\nConverted HpSharkFloat<SharkFloatParams> representations:" << std::endl;

        for (size_t i = 0; i < inputX.size(); ++i) {
            std::cout << "X[" << i << "]: " << inputX[i].ToString() << std::endl;
            std::cout << "X[" << i << "] hex: " << inputX[i].ToHexString() << std::endl;
        }

        std::cout << "\nOriginal MPIR input values:" << std::endl;
        for (size_t i = 0; i < mpfInputLen; ++i) {
            std::cout << "X[" << i << "]: "
                      << MpfToString<SharkFloatParams>(mpfInputX[i],
                                                       HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                      << std::endl;
        }
    }

    TestCoreReferenceOrbit<SharkFloatParams, sharkOperator>(launchParams,
                                                            Tests,
                                                            testNum,
                                                            inputX,
                                                            mpfInputX,
                                                            mpfInputLen,
                                                            reference2MinFusedStages,
                                                            reference2MaxFusedStages);
}

template <class SharkFloatParams, Operator sharkOperator, bool IncludeSigns>
void
TestTernaryOperatorTwoNumbersRaw(const HpShark::LaunchParams &launchParams,
                                 TestTracker &Tests,
                                 int testNum,
                                 const std::vector<HpSharkFloat<SharkFloatParams>> &inputX,
                                 const mpf_t *mpfInputX,
                                 size_t mpfInputLen,
                                 uint32_t reference2MinFusedStages,
                                 uint32_t reference2MaxFusedStages)
{

    std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy{};
    auto mpfXCopy = std::make_unique<mpf_t[]>(mpfInputLen);

    for (size_t i = 0; i < mpfInputLen; ++i) {
        mpf_init(mpfXCopy[i]);
        mpf_set(mpfXCopy[i], mpfInputX[i]);
    }

    // If IncludeSigns is true, then call TestTernaryOperatorTwoNumbersRawNoSignChange with all four
    // variants using mpf_neg as needed

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
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
            testNum++;
        }

        if constexpr (EnableTestSign2) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);
            negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
            testNum++;
        }

        if constexpr (EnableTestSign3) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);
            negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
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
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
            testNum++;
        }

        if constexpr (EnableTestSign5) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
            testNum++;
        }

        if constexpr (EnableTestSign6) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[0], xNumCopy[0]);
            negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);
            negateMpfAndHp(mpfXCopy[3], xNumCopy[3]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
        }

        if constexpr (EnableTestSign7) {
            resetCopy();
            negateMpfAndHp(mpfXCopy[1], xNumCopy[1]);
            negateMpfAndHp(mpfXCopy[2], xNumCopy[2]);
            negateMpfAndHp(mpfXCopy[4], xNumCopy[4]);

            printTest(testNum);
            TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
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
                launchParams,
                Tests,
                testNum,
                xNumCopy,
                mpfXCopy.get(),
                mpfInputLen,
                reference2MinFusedStages,
                reference2MaxFusedStages);
        }

    } else {
        TestTernaryOperatorTwoNumbersRawNoSignChange<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            testNum,
            inputX,
            mpfXCopy.get(),
            mpfInputLen,
            reference2MinFusedStages,
            reference2MaxFusedStages);
    }

    for (size_t i = 0; i < mpfInputLen; ++i) {
        mpf_clear(mpfXCopy[i]);
    }
}

void
ClearConsole()
{
    Environment::ClearConsole();
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernaryOperatorTwoNumbers(const HpShark::LaunchParams &launchParams,
                              TestTracker &Tests,
                              int testNum,
                              const std::vector<const char *> &num,
                              mpf_t *mpfIn,
                              size_t mpfInLen)
{

    // Copy mpfX and mpfY
    auto mpfCopy = std::make_unique<mpf_t[]>(mpfInLen);
    constexpr uint32_t requiredStage = HpSharkReferenceOneShotRequiredStage<SharkFloatParams>();
    for (size_t i = 0; i < mpfInLen; ++i) {
        mpf_init(mpfCopy[i]);
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
                std::cout << "mpfIn[" << i << "]: "
                          << MpfToString<SharkFloatParams>(
                                 mpfIn[i], HpSharkFloat<SharkFloatParams>::DefaultPrecBits)
                          << std::endl;
            }

            std::cout << "operator: " << OperatorToString<sharkOperator>() << std::endl;
        }

        // Convert the input values to HpSharkFloat<SharkFloatParams> representations
        std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy{mpfInLen};

        assert(xNumCopy.size() == num.size());

        for (size_t i = 0; i < num.size(); ++i) {
            xNumCopy[i].MpfToHpGpu(mpfCopy[i],
                                   HpSharkFloat<SharkFloatParams>::DefaultPrecBits,
                                   InjectNoiseInLowOrder::Disable);
        }

        TestTernaryOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, false>(launchParams,
                                                                                 Tests,
                                                                                 testNum,
                                                                                 xNumCopy,
                                                                                 mpfCopy.get(),
                                                                                 mpfInLen,
                                                                                 requiredStage,
                                                                                 requiredStage);

        testNum++;
    };

    auto resetCopy = [&]() {
        for (size_t i = 0; i < mpfInLen; ++i) {
            mpf_clear(mpfCopy[i]);
            mpf_init(mpfCopy[i]);
            mpf_set(mpfCopy[i], mpfIn[i]);
        }
    };

    auto printTest = [&](int curTest) {
        if (SharkVerbose == VerboseMode::Debug) {
            std::cout << std::endl;
            std::cout << std::endl;
        }

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

        mpf_neg(mpfCopy[0], mpfCopy[0]);
        curTest();
    }

    if constexpr (EnableTestSign3) {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfCopy[1], mpfCopy[1]);
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
        mpf_neg(mpfCopy[2], mpfCopy[2]);
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
        mpf_neg(mpfCopy[1], mpfCopy[1]);
        mpf_neg(mpfCopy[2], mpfCopy[2]);
        curTest();
    }

    if constexpr (EnableTestSign8) {
        printTest(testNum);
        resetCopy();
        mpf_neg(mpfCopy[0], mpfCopy[0]);
        mpf_neg(mpfCopy[1], mpfCopy[1]);
        mpf_neg(mpfCopy[2], mpfCopy[2]);
        curTest();
    }

    for (size_t i = 0; i < mpfInLen; ++i) {
        mpf_clear(mpfCopy[i]);
    }
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernaryOperatorTwoNumbers(const HpShark::LaunchParams &launchParams,
                              TestTracker &Tests,
                              int testNum,
                              const char *num1,
                              const char *num2,
                              const char *num3)
{

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Test " << std::dec << testNum << std::endl;

    mpf_set_default_prec(
        HpSharkFloat<SharkFloatParams>::DefaultMpirBits); // Set precision for MPIR floating point

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

    MpfNormalize(mpfs[0]);
    MpfNormalize(mpfs[1]);
    MpfNormalize(mpfs[2]);

    {
        std::vector<const char *> strs(3);
        strs[0] = num1;
        strs[1] = num2;
        strs[2] = num3;

        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, testNum, strs, mpfs, NumMpfs);
    }

    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_clear(mpfs[i]);
    }
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial(const HpShark::LaunchParams &launchParams,
                   TestTracker &Tests,
                   int testNum,
                   const HpSharkFloat<SharkFloatParams> &xNum,
                   const HpSharkFloat<SharkFloatParams> &yNum,
                   const HpSharkFloat<SharkFloatParams> &zNum,
                   const HpSharkFloat<SharkFloatParams> &xNum2,
                   const HpSharkFloat<SharkFloatParams> &yNum2,
                   uint32_t reference2MinFusedStages,
                   uint32_t reference2MaxFusedStages)
{

    static constexpr size_t NumMpfs = 5;
    std::vector<HpSharkFloat<SharkFloatParams>> xNumCopy(NumMpfs);

    mpf_set_default_prec(
        HpSharkFloat<SharkFloatParams>::DefaultMpirBits); // Set precision for MPIR floating point
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

    TestTernaryOperatorTwoNumbersRaw<SharkFloatParams, sharkOperator, true>(launchParams,
                                                                            Tests,
                                                                            testNum,
                                                                            xNumCopy,
                                                                            mpfXCopy,
                                                                            NumMpfs,
                                                                            reference2MinFusedStages,
                                                                            reference2MaxFusedStages);

    // Clean up
    for (size_t i = 0; i < NumMpfs; ++i) {
        mpf_clear(mpfXCopy[i]);
    }
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecialHelper(const HpShark::LaunchParams &launchParams,
                         TestTracker &Tests,
                         int testNum,
                         const IntSignCombo &testData1,
                         const IntSignCombo &testData2,
                         const IntSignCombo &testData3,
                         const IntSignCombo &testData4,
                         const IntSignCombo &testData5)
{
    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl;
        std::cout << std::endl;
    }

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

    auto xNum{std::make_unique<HpSharkFloat<SharkFloatParams>>(
        testData1Copy.Digits.data(), testData1Copy.Exponent, testData1Copy.Negative)};
    auto yNum{std::make_unique<HpSharkFloat<SharkFloatParams>>(
        testData2Copy.Digits.data(), testData2Copy.Exponent, testData2Copy.Negative)};
    auto zNum{std::make_unique<HpSharkFloat<SharkFloatParams>>(
        testData3Copy.Digits.data(), testData3Copy.Exponent, testData3Copy.Negative)};
    auto xNum2{std::make_unique<HpSharkFloat<SharkFloatParams>>(
        testData4Copy.Digits.data(), testData4Copy.Exponent, testData4Copy.Negative)};
    auto yNum2{std::make_unique<HpSharkFloat<SharkFloatParams>>(
        testData5Copy.Digits.data(), testData5Copy.Exponent, testData5Copy.Negative)};

    constexpr uint32_t requiredStage = HpSharkReferenceOneShotRequiredStage<SharkFloatParams>();
    TestTernarySpecial<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, *xNum, *yNum, *zNum, *xNum2, *yNum2, requiredStage, requiredStage);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecialHelper(const HpShark::LaunchParams &launchParams,
                         TestTracker &Tests,
                         int testNum,
                         const std::vector<uint32_t> &testData1,
                         const std::vector<uint32_t> &testData2,
                         const std::vector<uint32_t> &testData3)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              testData1,
                                                              testData2,
                                                              testData3,
                                                              testData1,  // Repeat
                                                              testData2); // Repeat
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial(const HpShark::LaunchParams &launchParams,
                   TestTracker &Tests,
                   int testNum,
                   const HpSharkFloat<SharkFloatParams> &xNum,
                   const HpSharkFloat<SharkFloatParams> &yNum,
                   const HpSharkFloat<SharkFloatParams> &zNum,
                   uint32_t reference2MinFusedStages,
                   uint32_t reference2MaxFusedStages)
{

    TestTernarySpecial<SharkFloatParams, sharkOperator>(launchParams,
                                                        Tests,
                                                        testNum,
                                                        xNum,
                                                        yNum,
                                                        zNum,
                                                        xNum, // Repeat
                                                        yNum, // Repeat
                                                        reference2MinFusedStages,
                                                        reference2MaxFusedStages);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial1(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0x80000000;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, testData, testData, testData);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial2(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0xC0000000;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, testData, testData, testData);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial3(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    std::vector<uint32_t> testData;
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        testData.push_back(0);
    }

    assert(testData.size() == SharkFloatParams::GlobalNumUint32);
    testData[testData.size() - 1] = 0xFFFFFFFF;

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, testData, testData, testData);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial4(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0xF26D37FC,
                                                                                    0xA96025CE,
                                                                                    0xB03FC716,
                                                                                    0x1DF7182B,
                                                                                    0xCCBD69BD,
                                                                                    0x40C0F80C,
                                                                                    0xFAA0222E,
                                                                                    0xD1FDA456},
                                                              std::vector<uint32_t>{0x8BBCDF3,
                                                                                    0x4C3E7ACB,
                                                                                    0x6691A71D,
                                                                                    0xDFE03842,
                                                                                    0x3FADCA11,
                                                                                    0x4058BC9E,
                                                                                    0xF30FD7DE,
                                                                                    0xAA6CA582},
                                                              std::vector<uint32_t>{0xF26D37FC,
                                                                                    0xA96025CE,
                                                                                    0xB03FC716,
                                                                                    0x1DF7182B,
                                                                                    0xCCBD69BD,
                                                                                    0x40C0F80C,
                                                                                    0xFAA0222E,
                                                                                    0xD1FDA456});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial5(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0, 0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial6(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFFF, 0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial7(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0, 0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0xFFFFFFFF, 0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial8(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0, 0, 0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0, 0xFFFFFFFF, 0xFFFFFFFF},
        std::vector<uint32_t>{0, 0, 0xFFFFFFFF, 0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial9(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFF000000, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFF1, 0x10},
        std::vector<uint32_t>{0xFFFFFFF1, 0xF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial10(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0, 0, 0, 0x2, 0x3},
                                                              std::vector<uint32_t>{0, 0, 0, 0x5, 0x7},
                                                              std::vector<uint32_t>{0, 0, 0, 0x9, 0xb});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial11(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0, 0x2, 0, 0, 0, 0, 0x3},
        std::vector<uint32_t>{0, 0x5, 0, 0, 0, 0, 0x7},
        std::vector<uint32_t>{0, 0x9, 0, 0, 0, 0, 0xb});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial12(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFF000000, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFF1, 0xf},
        std::vector<uint32_t>{0xFFFFFFF2, 0x10});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial13(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFF000000, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFF1, 0x11},
        std::vector<uint32_t>{0xFFFFFFF2, 0x10});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial14(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFF000000, 0xFFFFFFFF},
        std::vector<uint32_t>{0xFFFFFFF1, 0x10},
        std::vector<uint32_t>{0xFFFFFFF2, 0x10});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial15(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{
            0x00000000, 0, 0xFFFFFFF1, 0x00000008, 0x00000000, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000},
        std::vector<uint32_t>{
            0x00000000, 0, 0x00000000, 0x0000000D, 0x00000000, 0xFFFFFFF6, 0x0000000A, 0x00000003},
        std::vector<uint32_t>{
            0x00000000, 0, 0xFFFFFFF1, 0x00000008, 0x00000000, 0xFFFFFFF8, 0xFFFFFFFF, 0x00000000});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial16(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0x0000000C,
                                                                                    0xFFFFFFF0,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFC,
                                                                                    0x00000000,
                                                                                    0x0000000D,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000},
                                                              std::vector<uint32_t>{0xFFFFFFFD,
                                                                                    0xFFFFFFEF,
                                                                                    0xFFFFFFEF,
                                                                                    0xFFFFFFF4,
                                                                                    0x00000000,
                                                                                    0x7A6650D9,
                                                                                    0x00000000,
                                                                                    0x00000000},
                                                              std::vector<uint32_t>{0x0000000C,
                                                                                    0xFFFFFFF0,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFC,
                                                                                    0x00000000,
                                                                                    0x0000000D,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial17(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFF3,
                                                                                    0xFFFFFFF9,
                                                                                    0x00000004},
                                                              std::vector<uint32_t>{0x0000000E,
                                                                                    0x00000000,
                                                                                    0x00000000,
                                                                                    0xFFFFFFF2,
                                                                                    0x00000003,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000},
                                                              std::vector<uint32_t>{0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFF3,
                                                                                    0xFFFFFFF9,
                                                                                    0x00000004});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial18(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0x00000001,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFC,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFC,
                                                                                    0xE8CFC461,
                                                                                    0xFFFFFFF9},
                                                              std::vector<uint32_t>{0xFFFFFFF8,
                                                                                    0xD446522A,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000010,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFE,
                                                                                    0xFFFFFFFF},
                                                              std::vector<uint32_t>{0x00000001,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFC,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFC,
                                                                                    0xE8CFC461,
                                                                                    0xFFFFFFF9});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial19(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0x685940F0,
                                                                                    0x00000000,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF},
                                                              std::vector<uint32_t>{0xFFFFFFF1,
                                                                                    0x5008CECF,
                                                                                    0x2A4D4784,
                                                                                    0x0000000D,
                                                                                    0x00000006,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000000},
                                                              std::vector<uint32_t>{0x685940F0,
                                                                                    0x00000000,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial20(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        std::vector<uint32_t>{0xFFFFFFFF, 0x556B0E43, 0x4EECA55A, 0x0000000E, 0xFFFFFFFF, 0x00000000,
                              0xFFFFFFF8, 0x9B1194D6, 0xFFFFFFFF, 0x00000000, 0x13C1799F, 0x00000000,
                              0xC5F37A5D, 0xFFFFFFF4, 0x6FBC0EFF, 0x00000008, 0xFFFFFFFF, 0x00000000,
                              0xFFFFFFEF, 0xB06FA6C3, 0x0000000F, 0xFFFFFFF4, 0x00000007, 0xFFFFFFFF},
        std::vector<uint32_t>{0x0503FC0B, 0xF26CA6A5, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000007, 0x00000010,
                              0xE640F2D9, 0x00000000, 0xFFFFFFF5, 0xFFFFFFFF, 0xFFFFFFF0, 0xFFFFFFFF,
                              0x00000004, 0x379A6DBB, 0xFFFFFFFF, 0x00000008, 0x00000002, 0xFFFFFFFF,
                              0x00000000, 0x0000000B, 0x00000000, 0xFFFFFFEF, 0xFFFFFFFF, 0x093E223D},
        std::vector<uint32_t>{0xFFFFFFFF, 0x556B0E43, 0x4EECA55A, 0x0000000E, 0xFFFFFFFF, 0x00000000,
                              0xFFFFFFF8, 0x9B1194D6, 0xFFFFFFFF, 0x00000000, 0x13C1799F, 0x00000000,
                              0xC5F37A5D, 0xFFFFFFF4, 0x6FBC0EFF, 0x00000008, 0xFFFFFFFF, 0x00000000,
                              0xFFFFFFEF, 0xB06FA6C3, 0x0000000F, 0xFFFFFFF4, 0x00000007, 0xFFFFFFFF});
}

static constexpr int32_t HpSharkReferenceSpecial21MinExponentOverride = -512;
static constexpr int32_t HpSharkReferenceSpecial21MaxExponentOverride = 512;

template <class SharkFloatParams>
static constexpr uint32_t
HpSharkReferenceSpecial21RequiredStage()
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;

    // TestTernarySpecial21 constructs a one in the least-significant input limb and then
    // normalizes it.  The lowest exponent in its sweep therefore gives the largest gap between
    // the squared real and imaginary terms.
    constexpr int64_t normalizedOneShift =
        static_cast<int64_t>(SharkFloatParams::GlobalNumUint32) * 32ll - 1ll;
    constexpr uint64_t maxProductExponentGap =
        2ull * static_cast<uint64_t>(normalizedOneShift - HpSharkReferenceSpecial21MinExponentOverride);
    constexpr uint64_t coefficientShift =
        maxProductExponentGap / static_cast<uint64_t>(SharkFloatParams::ReferenceNTTPlan.b);
    constexpr uint64_t productCoefficientCount =
        2ull * static_cast<uint64_t>(SharkFloatParams::ReferenceNTTPlan.L) - 1ull;
    constexpr uint32_t requiredN =
        SharkNTT::NextPow2U32(static_cast<uint32_t>(coefficientShift + productCoefficientCount));
    constexpr uint32_t requiredStage = SharkNTT::CeilLog2U32(requiredN);
    static_assert(requiredStage <= Workspace::MaxFusedStages);

    return requiredStage > Workspace::MinFusedStages ? requiredStage : Workspace::MinFusedStages;
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial21(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{0xFFFFFFFD,
                                                                                    0x0000000B,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFE,
                                                                                    0x00000000,
                                                                                    0x88A881E4,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF},
                                                              std::vector<uint32_t>{0x00000007,
                                                                                    0xD9B23983,
                                                                                    0x00000005,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFF,
                                                                                    0x00000006,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF},
                                                              std::vector<uint32_t>{0xFFFFFFFD,
                                                                                    0x0000000B,
                                                                                    0x00000000,
                                                                                    0xFFFFFFFE,
                                                                                    0x00000000,
                                                                                    0x88A881E4,
                                                                                    0xFFFFFFFF,
                                                                                    0xFFFFFFFF});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial21(const HpShark::LaunchParams &launchParams,
                     TestTracker &Tests,
                     int testNum,
                     int exponentOverride2)
{

    std::vector<uint32_t> allFs{};
    for (size_t i = 0; i < SharkFloatParams::GlobalNumUint32; ++i) {
        allFs.push_back(0xFFFFFFFF);
    }
    allFs.resize(SharkFloatParams::GlobalNumUint32);

    std::vector<uint32_t> justOne{};
    justOne.push_back(1);
    justOne.resize(SharkFloatParams::GlobalNumUint32);

    if (SharkVerbose == VerboseMode::Debug) {
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Test " << std::dec << testNum << ", exponentOverride2_exp2: " << exponentOverride2
              << std::endl;

    auto xNum = std::make_unique<HpSharkFloat<SharkFloatParams>>(allFs.data(), 0, false);
    auto yNum =
        std::make_unique<HpSharkFloat<SharkFloatParams>>(justOne.data(), exponentOverride2, false);
    auto zNum = std::make_unique<HpSharkFloat<SharkFloatParams>>(allFs.data(), 0, false);

    TestTernarySpecial<SharkFloatParams, sharkOperator>(
        launchParams,
        Tests,
        testNum,
        *xNum,
        *yNum,
        *zNum,
        HpSharkReferenceWorkspace<SharkFloatParams>::MinFusedStages,
        HpSharkReferenceSpecial21RequiredStage<SharkFloatParams>());
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial22(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              IntSignCombo{false, 0, {5}},
                                                              IntSignCombo{false, 0, {17}},
                                                              IntSignCombo{false, 0, {0}},
                                                              IntSignCombo{false, 0, {5}},
                                                              IntSignCombo{true, 0, {17}});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial23(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{
    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(launchParams,
                                                              Tests,
                                                              testNum,
                                                              std::vector<uint32_t>{5},
                                                              std::vector<uint32_t>{17},
                                                              std::vector<uint32_t>{29},
                                                              std::vector<uint32_t>{57},
                                                              std::vector<uint32_t>{87});
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial24(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    IntSignCombo a{true,
                   0,
                   std::vector<uint32_t>{0x00000000,
                                         0xFFFFFFFD,
                                         0x00000000,
                                         0xFFFFFFFE,
                                         0xFFFFFFFF,
                                         0x00000002,
                                         0xFFFFFFFF,
                                         0xFFFFFFFF}};
    IntSignCombo b{false,
                   0,
                   std::vector<uint32_t>{0x8EB717E8,
                                         0xFFFFFFFF,
                                         0xA4D1162E,
                                         0x0000000E,
                                         0xC87AB0C2,
                                         0x00000000,
                                         0xFFFFFFFF,
                                         0x00000000}};
    IntSignCombo c{true,
                   0,
                   std::vector<uint32_t>{0xFFFFFFFD,
                                         0xECE3ACF5,
                                         0x0000000F,
                                         0xFFFFFFFF,
                                         0xFFFFFFF8,
                                         0xFFFFFFFF,
                                         0x00000000,
                                         0xFFFFFFFF}};

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, a, b, c, a, b);
}

template <class SharkFloatParams, Operator sharkOperator>
void
TestTernarySpecial25(const HpShark::LaunchParams &launchParams, TestTracker &Tests, int testNum)
{

    IntSignCombo a{true,
                   0,
                   std::vector<uint32_t>{0x00000003,
                                         0xfffffff3,
                                         0x0000000e,
                                         0xffffffff,
                                         0x00000000,
                                         0x00000000,
                                         0xffffffff,
                                         0xffffffff}};
    IntSignCombo b{false,
                   -59,
                   std::vector<uint32_t>{0x00000000,
                                         0x00000000,
                                         0x00000000,
                                         0xb0000000,
                                         0x5fffffff,
                                         0x78000000,
                                         0xcfffffff,
                                         0x87ffffff}};
    IntSignCombo c{true,
                   0,
                   std::vector<uint32_t>{0x00000009,
                                         0x00000000,
                                         0x577f96c7,
                                         0x00000000,
                                         0xfffffff9,
                                         0x00000009,
                                         0xffffffff,
                                         0xfffffffe}};

    TestTernarySpecialHelper<SharkFloatParams, sharkOperator>(
        launchParams, Tests, testNum, a, b, c, a, b);
}

template <Operator sharkOperator>
bool
TestBinaryOperatorPerf(const HpShark::LaunchParams &launchParams,
                       [[maybe_unused]] int testBase,
                       [[maybe_unused]] int numIters,
                       [[maybe_unused]] int internalTestLoopCount,
                       [[maybe_unused]] BasicCorrectnessMode mode)
{
    TestTracker Tests;

    switch (mode) {
        case BasicCorrectnessMode::Correctness_P1:
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            // Not a perf mode; historically this function wouldn't run in those modes.
            break;

        case BasicCorrectnessMode::PerfSub:
            for (int i = 0; i < numIters; ++i) {
                TestPerfRandom<SharkParamsNP6, sharkOperator>(
                    launchParams, Tests, testBase + 1, internalTestLoopCount);
                TestPerfRandom<SharkParamsNP7, sharkOperator>(
                    launchParams, Tests, testBase + 2, internalTestLoopCount);
                TestPerfRandom<SharkParamsNP8, sharkOperator>(
                    launchParams, Tests, testBase + 3, internalTestLoopCount);
                TestPerfRandom<SharkParamsNP9, sharkOperator>(
                    launchParams, Tests, testBase + 4, internalTestLoopCount);

                TestPerfRandom<SharkParamsNP10, sharkOperator>(
                    launchParams, Tests, testBase + 5, internalTestLoopCount);
                TestPerfRandom<SharkParamsNP11, sharkOperator>(
                    launchParams, Tests, testBase + 6, internalTestLoopCount);
                TestPerfRandom<SharkParamsNP12, sharkOperator>(
                    launchParams, Tests, testBase + 7, internalTestLoopCount);
            }
            break;

        case BasicCorrectnessMode::PerfSingleView30:
        case BasicCorrectnessMode::PerfSingleView32:
        case BasicCorrectnessMode::PerfSingleRef:
            for (int i = 0; i < numIters; ++i) {
                TestPerfRandom<SharkParamsNP7, sharkOperator>(
                    launchParams, Tests, testBase + 1, internalTestLoopCount);
            }
            break;

        default:
            // Defensive: if enum grows and caller passes unknown value.
            break;
    }

    return Tests.CheckAllTestsPassed();
}

namespace {

enum class FullReferencePerfInputEncoding {
    Decimal,
    ExactHex,
};

struct FullReferencePerfViewOverride {
    const char *m_Label;
    const char *m_Num1;
    const char *m_Num2;
    const char *m_Num3;
    const char *m_RadiusY;
    const char *m_Num1Hex;
    const char *m_Num2Hex;
    FullReferencePerfInputEncoding m_InputEncoding;
    bool m_UseOriginalInputStrings;
    bool m_DoubleRadiusY;
    bool m_ReduceHdrRadiusY;
    uint64_t m_DefaultMaxIters;
    int64_t m_ExpectedPeriod;
    PeriodicityResult m_ExpectedResult;
    bool m_HasBenchmarkLimbSelection = false;
    FullReferencePerfLimbSelection m_BenchmarkLimbSelection;
};

std::optional<FullReferencePerfViewOverride>
GetFullReferencePerfViewOverride(size_t view)
{
    switch (view) {
        case 5:
            return FullReferencePerfViewOverride{
                "View5",
                "-5."
                "48205748070475708458212567546733029376699274622882453824444834594995999680895291"
                "29972505947379718e-01",
                "-5."
                "77570838903603842805108982201850558675551728458255317158378952895736909832155423"
                "61901805676878083e-01",
                "0",
                "0."
                "00000000000000000000000000000000000000000000401444147896341553391537310767676"
                "870110653199358192656",
                nullptr,
                nullptr,
                FullReferencePerfInputEncoding::Decimal,
                false,
                false,
                false,
                20'000,
                16'045,
                PeriodicityResult::PeriodFound,
                true,
                FullReferencePerfLimbSelection{16'384, 15'000}};
        case 30: {
#include "LargeCoords30.h"
            return FullReferencePerfViewOverride{"View30",
                                                 strX,
                                                 strY,
                                                 "0",
                                                 "1.46269686645751934186e-114514",
                                                 strXHex,
                                                 strYHex,
                                                 FullReferencePerfInputEncoding::ExactHex,
                                                 true,
                                                 true,
                                                 true,
                                                 700'000,
                                                 669'772,
                                                 PeriodicityResult::PeriodFound,
                                                 true,
                                                 FullReferencePerfLimbSelection{16'384, 15'000}};
        }
        case 32: {
#include "LargeCoords32.h"
            return FullReferencePerfViewOverride{"View32",
                                                 strX,
                                                 strY,
                                                 "0",
                                                 "1.24525e-244240",
                                                 nullptr,
                                                 nullptr,
                                                 FullReferencePerfInputEncoding::Decimal,
                                                 false,
                                                 false,
                                                 false,
                                                 30'000'000,
                                                 22'680'804,
                                                 PeriodicityResult::PeriodFound,
                                                 true,
                                                 FullReferencePerfLimbSelection{65'536, 60'000}};
        }
        default:
            return std::nullopt;
    }
}

} // namespace

uint32_t
GetMinimumFullReferencePerfEffectiveLimbs(Operator referenceOperator, uint32_t storageLimbs)
{
    (void)referenceOperator;
    return storageLimbs / 2u + 1u;
}

uint32_t
GetFullReferencePerfEffectiveLimbs(Operator referenceOperator,
                                   uint64_t requestedLimbs,
                                   uint32_t storageLimbs)
{
    (void)referenceOperator;
    return GetReferenceEffectivePrecisionLimbs(requestedLimbs, storageLimbs);
}

bool
IsValidFullReferencePerfLimbSelection(Operator referenceOperator,
                                      const FullReferencePerfLimbSelection &selection)
{
    if (!IsSupportedLimbCount(selection.m_StorageLimbs)) {
        return false;
    }

    const uint32_t minimumEffectiveLimbs =
        GetMinimumFullReferencePerfEffectiveLimbs(referenceOperator, selection.m_StorageLimbs);
    return selection.m_EffectiveLimbs >= minimumEffectiveLimbs &&
           selection.m_EffectiveLimbs <= selection.m_StorageLimbs;
}

FullReferencePerfPrecision
GetFullReferencePerfPrecision(Operator referenceOperator, size_t view)
{
    const auto preset = GetViewPreset(view,
                                      /*defaultIterations=*/0,
                                      /*defaultCompressionExpLow=*/0,
                                      /*defaultCompressionExpIntermediate=*/0);
    constexpr bool requiresReuse = false;
    const uint64_t requiredPrecisionBits = PrecisionCalculator::GetPrecision(
        preset.minX, preset.minY, preset.maxX, preset.maxY, requiresReuse);
    const uint64_t requestedPrecisionLimbs = (requiredPrecisionBits + 31u) / 32u;
    const uint32_t storagePrecisionLimbs = BitsToSupportedLimbCount(requiredPrecisionBits);
    const uint32_t effectivePrecisionLimbs = GetFullReferencePerfEffectiveLimbs(
        referenceOperator, requestedPrecisionLimbs, storagePrecisionLimbs);

    FullReferencePerfLimbSelection defaultSelection{storagePrecisionLimbs, effectivePrecisionLimbs};
    bool defaultIsBenchmarkPreset = false;
    const auto viewOverride = GetFullReferencePerfViewOverride(view);
    if (viewOverride && viewOverride->m_HasBenchmarkLimbSelection) {
        defaultSelection = viewOverride->m_BenchmarkLimbSelection;
        defaultIsBenchmarkPreset = true;
    }

    return FullReferencePerfPrecision{
        requiredPrecisionBits,
        requestedPrecisionLimbs,
        FullReferencePerfLimbSelection{storagePrecisionLimbs, effectivePrecisionLimbs},
        defaultSelection,
        defaultIsBenchmarkPreset};
}

template <Operator sharkOperator>
bool
TestFullReferencePerfView(TestTracker &Tests,
                          int numBlocks,
                          int numThreads,
                          int testBase,
                          int numIters,
                          int internalTestLoopCount,
                          bool useMT,
                          size_t view,
                          const FullReferencePerfLimbSelection &limbSelection)
{
    static_assert(IsReferenceOrbitOperator<sharkOperator>, "Reference-orbit operators only");

    const auto productionPrecision = GetFullReferencePerfPrecision(sharkOperator, view);
    const auto preset = GetViewPreset(view,
                                      /*defaultIterations=*/0,
                                      /*defaultCompressionExpLow=*/0,
                                      /*defaultCompressionExpIntermediate=*/0);
    if (!IsValidFullReferencePerfLimbSelection(sharkOperator, limbSelection)) {
        std::cout << "Invalid full-reference limb selection: storageLimbs="
                  << limbSelection.m_StorageLimbs
                  << ", effectiveLimbs=" << limbSelection.m_EffectiveLimbs << std::endl;
        return false;
    }

    const uint64_t requiredPrecisionBits = productionPrecision.m_RequiredPrecisionBits;
    const uint64_t requestedPrecisionLimbs = productionPrecision.m_RequestedPrecisionLimbs;
    const uint32_t storagePrecisionLimbs = limbSelection.m_StorageLimbs;
    const uint32_t effectivePrecisionLimbs = limbSelection.m_EffectiveLimbs;

    const auto viewOverride = GetFullReferencePerfViewOverride(view);
    const std::string genericLabel = "View" + std::to_string(view);
    std::string genericNum1;
    std::string genericNum2;
    std::string genericRadiusY;

    if (!viewOverride) {
        const HighPrecision two{2};
        const HighPrecision centerX = (preset.minX + preset.maxX) / two;
        const HighPrecision centerY = (preset.minY + preset.maxY) / two;
        const HighPrecision radiusY = (preset.maxY - preset.minY) / two;

        genericNum1 = MpfToHex64StringInvertable(*centerX.backendRaw());
        genericNum2 = MpfToHex64StringInvertable(*centerY.backendRaw());
        genericRadiusY = MpfToHex64StringInvertable(*radiusY.backendRaw());
    }

    std::cout << "View " << view << " precision: requiredBits=" << requiredPrecisionBits
              << ", requestedLimbs=" << requestedPrecisionLimbs
              << ", productionStorageLimbs=" << productionPrecision.m_ProductionSelection.m_StorageLimbs
              << ", productionEffectiveLimbs="
              << productionPrecision.m_ProductionSelection.m_EffectiveLimbs
              << ", defaultStorageLimbs=" << productionPrecision.m_DefaultSelection.m_StorageLimbs
              << ", defaultEffectiveLimbs=" << productionPrecision.m_DefaultSelection.m_EffectiveLimbs
              << ", storageLimbs=" << storagePrecisionLimbs
              << ", effectiveLimbs=" << effectivePrecisionLimbs << std::endl;

    bool result = true;
    DispatchByLimbCount<SharkParamsBaseFamily>(storagePrecisionLimbs, [&]<class SharkFloatParams>() {
        HpShark::LaunchParams launchParams{numBlocks, numThreads};
        mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

        const auto setDecimal = [](mpf_ptr target, const char *value, const char *name) {
            if (mpf_set_str(target, value, 10) == -1) {
                std::cout << "Error setting " << name << std::endl;
            }
        };

        mpf_t mpfX;
        mpf_t mpfY;
        mpf_t mpfZ;
        mpf_t mpfRadiusY;
        mpf_init(mpfZ);
        mpf_init(mpfRadiusY);

        if (viewOverride && viewOverride->m_InputEncoding == FullReferencePerfInputEncoding::ExactHex) {
            Hex64StringToMpf_Exact(viewOverride->m_Num1Hex, mpfX);
            Hex64StringToMpf_Exact(viewOverride->m_Num2Hex, mpfY);
        } else if (viewOverride) {
            mpf_init(mpfX);
            mpf_init(mpfY);
            setDecimal(mpfX, viewOverride->m_Num1, "mpfX");
            setDecimal(mpfY, viewOverride->m_Num2, "mpfY");
        } else {
            Hex64StringToMpf_Exact(genericNum1, mpfX);
            Hex64StringToMpf_Exact(genericNum2, mpfY);
        }

        if (viewOverride) {
            setDecimal(mpfZ, viewOverride->m_Num3, "mpfZ");
            setDecimal(mpfRadiusY, viewOverride->m_RadiusY, "mpfRadiusY");
        } else {
            setDecimal(mpfZ, "0", "mpfZ");
            Hex64StringToMpf_Exact(genericRadiusY, mpfRadiusY);
        }

        mpf_t mpfTwo;
        bool hasMpfTwo = false;
        if (viewOverride && viewOverride->m_DoubleRadiusY) {
            mpf_init(mpfTwo);
            hasMpfTwo = true;
            setDecimal(mpfTwo, "2", "mpfTwo");
            mpf_mul(mpfRadiusY, mpfTwo, mpfRadiusY);
        }

        if (viewOverride && viewOverride->m_InputEncoding == FullReferencePerfInputEncoding::ExactHex &&
            SharkVerbose == VerboseMode::Debug) {
            const auto mpfXConvertStr = MpfToHex64StringInvertable(mpfX);
            const auto mpfYConvertStr = MpfToHex64StringInvertable(mpfY);
            std::cout << "Correct MPIR hex X: " << std::endl << mpfXConvertStr;
            std::cout << "Correct MPIR hex Y: " << std::endl << mpfYConvertStr;
            assert(mpfXConvertStr == viewOverride->m_Num1Hex);
            assert(mpfYConvertStr == viewOverride->m_Num2Hex);
        }

        MpfNormalize(mpfX);
        MpfNormalize(mpfY);
        MpfNormalize(mpfZ);
        MpfNormalize(mpfRadiusY);

        std::string num1;
        std::string num2;
        std::string num3;
        std::string radiusYStr;

        if (viewOverride && viewOverride->m_UseOriginalInputStrings) {
            num1 = viewOverride->m_Num1;
            num2 = viewOverride->m_Num2;
            num3 = viewOverride->m_Num3;
        } else {
            num1 = MpfToString<SharkFloatParams>(mpfX, HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
            num2 = MpfToString<SharkFloatParams>(mpfY, HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
            num3 = MpfToString<SharkFloatParams>(mpfZ, HpSharkFloat<SharkFloatParams>::DefaultMpirBits);
        }
        radiusYStr = viewOverride ? viewOverride->m_RadiusY : genericRadiusY;

        typename SharkFloatParams::Float hdrRadiusY{mpfRadiusY};
        if (!viewOverride || viewOverride->m_ReduceHdrRadiusY) {
            HdrReduce(hdrRadiusY);
        }

        const uint64_t maxIters = (internalTestLoopCount != 0)
                                      ? static_cast<uint64_t>(internalTestLoopCount)
                                      : (viewOverride ? viewOverride->m_DefaultMaxIters : 20'000);
        const int64_t expectedPeriod = viewOverride ? viewOverride->m_ExpectedPeriod : -1;
        const auto expectedResult =
            viewOverride ? viewOverride->m_ExpectedResult : PeriodicityResult::Continue;

        auto preparedTables = HpShark::PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(
            launchParams, mpfX, mpfY, effectivePrecisionLimbs, testBase);

        std::vector<PerfTimingResult> timings;
        timings.reserve(numIters);
        for (int i = 0; i < numIters; ++i) {
            const int testNum = testBase + i;
            PerfTimingResult timing;
            TestPerf<SharkFloatParams, sharkOperator>(launchParams,
                                                      Tests,
                                                      testNum,
                                                      num1.c_str(),
                                                      num2.c_str(),
                                                      num3.c_str(),
                                                      radiusYStr.c_str(),
                                                      mpfX,
                                                      mpfY,
                                                      mpfZ,
                                                      hdrRadiusY,
                                                      maxIters,
                                                      expectedPeriod,
                                                      expectedResult,
                                                      effectivePrecisionLimbs,
                                                      useMT,
                                                      &timing,
                                                      preparedTables.get());
            timings.push_back(timing);
        }

        constexpr const char *cpuLabel = "CPU reference";
        if (!viewOverride) {
            std::cout << "\nGeneric view-perf for view " << view << std::endl;
        }
        const char *summaryLabel = viewOverride ? viewOverride->m_Label : genericLabel.c_str();
        PrintPerfSummaryTable(summaryLabel, useMT, timings, "MPIR", cpuLabel);

        mpf_clear(mpfX);
        mpf_clear(mpfY);
        mpf_clear(mpfZ);
        mpf_clear(mpfRadiusY);
        if (hasMpfTwo) {
            mpf_clear(mpfTwo);
        }
    });

    return result;
}
template <class SharkFloatParams, Operator sharkOperator>
bool
TestAllBinaryOp(int testBase)
{
    HpShark::LaunchParams launchParams{2, 32};
    TestTracker Tests;

    constexpr bool includeSet1 = true;
    constexpr bool includeSet2 = true;
    constexpr bool includeSet3 = true;
    constexpr bool includeSet4 = true;
    constexpr bool includeSet5 = true;
    constexpr bool includeSet6 = true;
    constexpr bool includeSet10 = true;
    constexpr bool includeSet11 = false;

    if constexpr (includeSet1) {
        const auto set = testBase + 100;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 10, "7", "19", "0");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 20, "4294967295", "1", "4294967296");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 30, "4294967296", "1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 40, "4294967295", "4294967296", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 50, "4294967296", "-1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 60, "18446744073709551615", "1", "1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 70, "0", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 80, "0.1", "0", "0.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 90, "0", "0", "0");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 100, "0.1", "0.1", "0.1");
    }

    if constexpr (includeSet2) {
        const auto set = testBase + 300;
        TestTernarySpecial1<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 10);
        TestTernarySpecial2<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 20);
        TestTernarySpecial3<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 30);
        TestTernarySpecial4<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 40);
        TestTernarySpecial5<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 50);
        TestTernarySpecial6<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 60);
        TestTernarySpecial7<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 70);
        TestTernarySpecial8<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 80);
        TestTernarySpecial9<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 90);
        TestTernarySpecial10<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 100);
        TestTernarySpecial11<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 110);
        TestTernarySpecial12<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 120);
        TestTernarySpecial13<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 130);
        TestTernarySpecial14<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 140);
        TestTernarySpecial15<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 150);
        TestTernarySpecial16<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 160);
        TestTernarySpecial17<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 170);
        TestTernarySpecial18<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 180);
        TestTernarySpecial19<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 190);
        TestTernarySpecial20<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 200);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 210);
        TestTernarySpecial22<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 220);
        TestTernarySpecial23<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 230);
        TestTernarySpecial24<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 240);
        TestTernarySpecial25<SharkFloatParams, sharkOperator>(launchParams, Tests, set + 250);
    }

    if constexpr (includeSet3) {
        const auto set = testBase + 600;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 10, "2", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 20, "0.2", "0.1", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 30, "0.5", "1.2", "1.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 40, "0.6", "1.3", "1.9");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 50, "0.7", "1.4", "2.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 60, "0.1", "1.99999999999999999999999999999", "2.1");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(launchParams,
                                                                       Tests,
                                                                       set + 70,
                                                                       "0.123124561464451654461",
                                                                       "1.2395123123127298375982735",
                                                                       "1.187236498176923871462938");
    }

    if constexpr (includeSet4) {
        const auto set = testBase + 700;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 10, "-0.5", "1.2", "0.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 20, "-0.6", "1.3", "0.7");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 30, "-0.7", "1.4", "0.3");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 40, "-0.1", "1.99999999999999999999999999999", "0.9");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(launchParams,
                                                                       Tests,
                                                                       set + 50,
                                                                       "-0.123124561464451654461",
                                                                       "1.2395123123127298375982735",
                                                                       "0.1");
    }

    if constexpr (includeSet5) {
        const auto set = testBase + 800;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 10, "-0.51", "-1.29", "-1.49");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 20, "-0.61", "-1.39", "-0.599");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams, Tests, set + 30, "-0.71", "-1.49", "-0.799");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 40,
            "-0.11",
            "-1.99999999999999999999999999999",
            "-0.89999999999999999999999999999");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(launchParams,
                                                                       Tests,
                                                                       set + 50,
                                                                       "-0.123124561464451654461",
                                                                       "-1.2395123123127298375982735",
                                                                       "-1.1123877508482781861362735");
    }

    if constexpr (includeSet6) {
        const auto set = testBase + 900;
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 10,
            "0.5265542653452654526545625456254565446654545645649789871322131213156435546435",
            "-1."
            "2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866"
            "39173047328910730217803271839216",
            "0.12987461239874619237469187236948716928374691827364");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 20,
            "0.2999999999965542653452654526545625456254565446654545645649789871322131213156435546435",
            "-1."
            "2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866"
            "39173047328910730217803271839216",
            "1.12374861283467182367518476235481675234862e2334");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 30,
            "0.1265542653452654526545625456254565446654545645649789871322131213156435546435",
            "-1."
            "2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866"
            "391730473289107302178039999999999999271839216",
            "1234671987263941876239487162398746e18239");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 40,
            "0.0265542653452654526545625456254565446654545645649789871322131213156435546435",
            "-1."
            "2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866"
            "39173047328910730217803271839216",
            "1023949123e389274");
        TestTernaryOperatorTwoNumbers<SharkFloatParams, sharkOperator>(
            launchParams,
            Tests,
            set + 50,
            "0."
            "0000000000000000026554265345265452654562545625456544665454564564978987132213121315643554643"
            "5",
            "-1."
            "2634683757879587749854733454356324153342452684769284546534432341646587766348547465845321866"
            "39173047328910730217803271839216",
            "7236.34234e5234523");
    }

    {
        static constexpr auto SpecificTest1 = -129;
        static constexpr auto SpecificTest2 = -128;
        static constexpr auto SpecificTest3 = -127;
        static constexpr auto SpecificTest4 = 127;
        static constexpr auto SpecificTest5 = 255;
        static constexpr auto SpecificTest6 = 256;

        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest1);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest2);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest3);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest4);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest5);
        TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, SpecificTest6);

        for (auto i = HpSharkReferenceSpecial21MinExponentOverride;
             i < HpSharkReferenceSpecial21MaxExponentOverride;
             i++) {
            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "Exponent adjustment: " << i << std::endl;
            }

            TestTernarySpecial21<SharkFloatParams, sharkOperator>(launchParams, Tests, 0, i);
        }
    }
    // #endif

    if constexpr (includeSet10) {
        const auto set10 = testBase + 1000;
        auto x = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto y = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto z = std::make_unique<HpSharkFloat<SharkFloatParams>>();

        for (auto i = 0; i < 1000; i += 10) {
            HpShark::LaunchParams randomizedlaunchParams{};

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
                randomizedlaunchParams, Tests, set10 + i, x_str.c_str(), y_str.c_str(), z_str.c_str());
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
                launchParams, Tests, 0, x_str.c_str(), y_str.c_str(), z_str.c_str());
        }
    }

    return Tests.CheckAllTestsPassed();
}

// Explicitly instantiate TestAllBinaryOp
#define REFERENCE_KERNEL(SharkFloatParams)                                                              \
    template bool TestAllBinaryOp<SharkFloatParams, Operator::ReferenceOrbit2>(int testBase);

// The Operator-only explicit instantiations below do not depend on
// SharkFloatParams, so instantiate each exactly once. MSVC tolerates
// duplicate explicit instantiations silently, but clang treats them as
// hard errors — keeping these out of the per-SharkFloatParams macros
// above makes the build portable.
#define OPERATOR_ONLY_INSTANTIATIONS()                                                                  \
    template bool TestBinaryOperatorPerf<Operator::ReferenceOrbit2>(const HpShark::LaunchParams &,      \
                                                                    int testBase,                       \
                                                                    int numIters,                       \
                                                                    int internalTestLoopCount,          \
                                                                    BasicCorrectnessMode mode);

#define ExplicitlyInstantiate(SharkFloatParams) REFERENCE_KERNEL(SharkFloatParams)

ExplicitInstantiateAll();

OPERATOR_ONLY_INSTANTIATIONS();

// Explicitly instantiate the generic view-perf driver (Operator-only, so once per reference orbit).
template bool TestFullReferencePerfView<Operator::ReferenceOrbit2>(
    TestTracker &,
    int numBlocks,
    int numThreads,
    int testBase,
    int numIters,
    int internalTestLoopCount,
    bool useMT,
    size_t view,
    const FullReferencePerfLimbSelection &limbSelection);
