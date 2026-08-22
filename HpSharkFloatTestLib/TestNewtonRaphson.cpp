#include "TestNewtonRaphson.h"

#include "Tests.h"

#include "BenchmarkTimer.h"
#include "DebugChecksumHost.h"
#include "HDRFloat.h"
#include "HighPrecision.h"
#include "HpSharkFloat.h"
#include "HpSharkTestConfig.h"
#include "KernelInvoke.h"
#include "KernelInvokeReferenceCache.h"
#include "KernelInvokeReferenceSetup.h"
#include "MpirOrbitEval.h"
#include "PerfTimingResult.h"
#include "ReferenceReferenceOrbit2.h"
#include "TestTracker.h"

#include <cstdint>
#include <gmp.h>
#include <iostream>
#include <memory>
#include <string>

// Wrapper around shared ST/MT orbit eval for the test benchmark.
static void
RunMpirOrbitWithD2(mpf_t cR,
                   mpf_t cI,
                   uint64_t period,
                   mpf_t outZR,
                   mpf_t outZI,
                   mpf_t outDzdcR,
                   mpf_t outDzdcI,
                   HDRFloat<double> &d2r_out,
                   HDRFloat<double> &d2i_out,
                   bool useMT)
{
    const mp_bitcnt_t prec = mpf_get_prec(cR);

    mpf_complex c_coord;
    mpf_complex_init(c_coord, prec);
    mpf_set(c_coord.re, cR);
    mpf_set(c_coord.im, cI);

    mpf_complex z_coord, dzdc_deriv;
    mpf_complex_init(z_coord, prec);
    mpf_complex_init(dzdc_deriv, prec);

    if (useMT) {
        EvaluateCriticalOrbitAndDerivsMT(
            c_coord, period, z_coord, dzdc_deriv, d2r_out, d2i_out, prec, prec);
    } else {
        EvaluateCriticalOrbitAndDerivsST(
            c_coord, period, z_coord, dzdc_deriv, d2r_out, d2i_out, prec, prec);
    }

    mpf_set(outZR, z_coord.re);
    mpf_set(outZI, z_coord.im);
    mpf_set(outDzdcR, dzdc_deriv.re);
    mpf_set(outDzdcI, dzdc_deriv.im);

    mpf_complex_clear(z_coord);
    mpf_complex_clear(dzdc_deriv);
    mpf_complex_clear(c_coord);
}

// Compute Newton step: step = z / dzdc (complex division via separate reals).
// Returns false if dzdc is zero.
static bool
ComputeNewtonStep(mpf_t zR,
                  mpf_t zI,
                  mpf_t dzdcR,
                  mpf_t dzdcI,
                  mpf_t stepR,
                  mpf_t stepI,
                  mpf_t denom,
                  mpf_t t1,
                  mpf_t t2)
{
    // denom = |dzdc|^2 = dzdcR^2 + dzdcI^2
    mpf_mul(t1, dzdcR, dzdcR);
    mpf_mul(t2, dzdcI, dzdcI);
    mpf_add(denom, t1, t2);
    if (mpf_cmp_ui(denom, 0) == 0)
        return false;

    // stepR = (zR*dzdcR + zI*dzdcI) / denom
    mpf_mul(t1, zR, dzdcR);
    mpf_mul(t2, zI, dzdcI);
    mpf_add(stepR, t1, t2);
    mpf_div(stepR, stepR, denom);

    // stepI = (zI*dzdcR - zR*dzdcI) / denom
    mpf_mul(t1, zI, dzdcR);
    mpf_mul(t2, zR, dzdcI);
    mpf_sub(stepI, t1, t2);
    mpf_div(stepI, stepI, denom);

    return true;
}

// Check Imagina convergence: err = |step|^4 * |d2|^2 / |dzdc|^2.
// Returns the exponent of err for threshold comparison.
static int
ComputeImaginaError(mpf_t stepR,
                    mpf_t stepI,
                    HDRFloat<double> d2r,
                    HDRFloat<double> d2i,
                    HDRFloat<double> dzdcNorm,
                    mpf_t normStep,
                    mpf_t t1,
                    mpf_t t2,
                    HDRFloat<double> &err_out)
{

    mpf_mul(t1, stepR, stepR);
    mpf_mul(t2, stepI, stepI);
    mpf_add(normStep, t1, t2); // |step|^2

    HDRFloat<double> normStep_hdr(normStep);
    HdrReduce(normStep_hdr);
    HDRFloat<double> normStep2 = normStep_hdr.square(); // |step|^4
    HdrReduce(normStep2);

    HDRFloat<double> d2Norm = d2r.square() + d2i.square(); // |d2|^2
    HdrReduce(d2Norm);

    err_out = (normStep2 * d2Norm) / dzdcNorm;
    HdrReduce(err_out);

    return (int)err_out.getExp();
}

template <class SharkFloatParams, Operator referenceOperator>
static bool
RunNewtonRaphsonTest(TestTracker &Tests,
                     int testBase,
                     const char *testName,
                     mpf_t mpfCReal,
                     mpf_t mpfCImag,
                     uint64_t period,
                     const HpShark::LaunchParams &launchParams,
                     uint64_t iterCountOverride = 0,
                     bool useMT = true,
                     int numRepeats = 1)
{
    static_assert(IsReferenceOrbitOperator<referenceOperator>);

    std::cout << "LaunchParams: " << launchParams.ToString()
              << ", SharedOnly: " << (SharkFloatParams::SharedOnly ? "true" : "false") << std::endl;

    // iterCountOverride > 0: perf-only mode (run exactly that many orbit iterations, no convergence).
    // iterCountOverride == 0: convergence mode (use actual period, run Newton iterations to converge).
    const bool perfOnly = (iterCountOverride > 0);
    if (perfOnly) {
        std::cout << testName << ": PERF-ONLY mode, iterationCount " << iterCountOverride << " (period "
                  << period << "), repeats " << numRepeats << std::endl;
        period = iterCountOverride;
    } else {
        std::cout << testName << ": Convergence mode, period " << period << " iterations";
        if (numRepeats > 1) {
            std::cout << " (NumIters=" << numRepeats << " ignored in convergence mode)";
        }
        std::cout << std::endl;
        numRepeats = 1; // convergence always runs once
    }

    const uint32_t maxNewtonIters = perfOnly ? static_cast<uint32_t>(numRepeats) : 32;
    const int precBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    const int targetExp = precBits * 2;

    // Shared temporaries for Newton step computation
    mpf_t stepR, stepI, denom, t1, t2, normStep;
    mpf_init(stepR);
    mpf_init(stepI);
    mpf_init(denom);
    mpf_init(t1);
    mpf_init(t2);
    mpf_init(normStep);

    // ========== Newton refinement setup ==========
    mpf_t cR, cI;
    mpf_init(cR);
    mpf_init(cI);
    mpf_set(cR, mpfCReal);
    mpf_set(cI, mpfCImag);

    mpf_t zR, zI, dzdcR, dzdcI;
    mpf_init(zR);
    mpf_init(zI);
    mpf_init(dzdcR);
    mpf_init(dzdcI);

    auto hpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpZR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpZI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    DebugHostCombo<SharkFloatParams> debugHostCombo;

    // MPIR comparison temporaries
    mpf_t mpirZR, mpirZI, mpirDzdcR, mpirDzdcI;
    mpf_init(mpirZR);
    mpf_init(mpirZI);
    mpf_init(mpirDzdcR);
    mpf_init(mpirDzdcI);

    mpf_t iterDiffZR, iterDiffZI, iterDiffDzdcR, iterDiffDzdcI;
    mpf_init(iterDiffZR);
    mpf_init(iterDiffZI);
    mpf_init(iterDiffDzdcR);
    mpf_init(iterDiffDzdcI);

    // Tolerance for per-iteration comparison
    const int singleOpMargin = 98;
    int logPeriod = 1;
    {
        uint64_t p = period;
        while (p > 1) {
            p >>= 1;
            ++logPeriod;
        }
    }
    const int toleranceBits = precBits - singleOpMargin * logPeriod;

    mpf_t tolerance, absDiff, absRef, relError;
    mpf_init(tolerance);
    mpf_init(absDiff);
    mpf_init(absRef);
    mpf_init(relError);
    mpf_set_ui(tolerance, 1);
    mpf_div_2exp(tolerance, tolerance, toleranceBits);

    bool allWithinTolerance = true;
    bool gpuWithinTolerance = true;
    bool gpuCompletedPeriod = true;
    uint32_t hpConvergedIter = maxNewtonIters;

    // Per-iteration time tracking for summary table
    std::vector<PerfTimingResult> perIterTimings;

    // GPU comparison temporaries
    mpf_t gpuZR, gpuZI, gpuDzdcR, gpuDzdcI;
    mpf_init(gpuZR);
    mpf_init(gpuZI);
    mpf_init(gpuDzdcR);
    mpf_init(gpuDzdcI);

    mpf_t gpuDiffZR, gpuDiffZI, gpuDiffDzdcR, gpuDiffDzdcI;
    mpf_init(gpuDiffZR);
    mpf_init(gpuDiffZI);
    mpf_init(gpuDiffDzdcR);
    mpf_init(gpuDiffDzdcI);

    const auto isWithinTolerance = [&](mpf_t diff, mpf_t reference) {
        mpf_abs(absDiff, diff);
        mpf_abs(absRef, reference);
        if (mpf_sgn(absRef) == 0) {
            return mpf_cmp(absDiff, tolerance) <= 0;
        }

        mpf_div(relError, absDiff, absRef);
        return mpf_cmp(relError, tolerance) <= 0;
    };

    // ========== Lockstep Newton refinement ==========
    for (uint32_t it = 0; it < maxNewtonIters; ++it) {
        std::cout << "  " << testName << " Newton iter " << it << std::endl;

        PerfTimingResult iterTiming;
        std::unique_ptr<HpShark::ReferencePreparedTables<SharkFloatParams>> preparedTables;
        if constexpr (HpShark::TestReferenceImpl || HpShark::TestGpu) {
            hpCR->MpfToHpGpu(cR, precBits, InjectNoiseInLowOrder::Disable);
            hpCI->MpfToHpGpu(cI, precBits, InjectNoiseInLowOrder::Disable);
            preparedTables = HpShark::PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(
                launchParams, *hpCR, *hpCI, SharkFloatParams::GlobalNumUint32, testBase, it);
        }

        // ---- MPIR inner loop (ground truth, gated on TestMPIRImpl) ----
        HDRFloat<double> mpirD2r{}, mpirD2i{};
        if constexpr (HpShark::TestMPIRImpl) {
            BenchmarkTimer mpirTimer;
            mpirTimer.StartTimer();
            RunMpirOrbitWithD2(
                cR, cI, period, mpirZR, mpirZI, mpirDzdcR, mpirDzdcI, mpirD2r, mpirD2i, useMT);
            mpirTimer.StopTimer();
            const double mpirMs = static_cast<double>(mpirTimer.GetDeltaInMs());
            iterTiming.hostMs = mpirMs;
            std::cout << "    MPIR inner loop timeMs: " << mpirMs << std::endl;
        }

        // ---- Selected CPU reference inner loop (if TestReferenceImpl) ----
        typename SharkFloatParams::Float hpD2r{}, hpD2i{};
        if constexpr (HpShark::TestReferenceImpl) {
            BenchmarkTimer cpuTimer;
            cpuTimer.StartTimer();
            EvaluateOrbitAndDerivative2<SharkFloatParams>(hpCR.get(),
                                                          hpCI.get(),
                                                          period,
                                                          hpZR.get(),
                                                          hpZI.get(),
                                                          hpDzdcR.get(),
                                                          hpDzdcI.get(),
                                                          &hpD2r,
                                                          &hpD2i,
                                                          SharkFloatParams::GlobalNumUint32,
                                                          debugHostCombo,
                                                          preparedTables.get());
            cpuTimer.StopTimer();
            const double cpuMs = static_cast<double>(cpuTimer.GetDeltaInMs());
            iterTiming.cpuMs = cpuMs;
            constexpr const char *cpuReferenceLabel = "CPU reference";
            std::cout << "    " << cpuReferenceLabel << " inner loop timeMs: " << cpuMs << std::endl;

            hpZR->HpGpuToMpf(zR);
            hpZI->HpGpuToMpf(zI);
            hpDzdcR->HpGpuToMpf(dzdcR);
            hpDzdcI->HpGpuToMpf(dzdcI);

            // CPU-ref vs MPIR comparison
            mpf_sub(iterDiffZR, zR, mpirZR);
            mpf_sub(iterDiffZI, zI, mpirZI);
            mpf_sub(iterDiffDzdcR, dzdcR, mpirDzdcR);
            mpf_sub(iterDiffDzdcI, dzdcI, mpirDzdcI);
            char buf[256];
            gmp_snprintf(buf, sizeof(buf), "    CPU iter %u z_real diff:    %+.6Fe", it, iterDiffZR);
            std::cout << buf << std::endl;
            gmp_snprintf(buf, sizeof(buf), "    CPU iter %u dzdc_real diff: %+.6Fe", it, iterDiffDzdcR);
            std::cout << buf << std::endl;

            if constexpr (HpShark::TestMPIRImpl) {
                allWithinTolerance &= isWithinTolerance(iterDiffZR, mpirZR);
                allWithinTolerance &= isWithinTolerance(iterDiffZI, mpirZI);
                allWithinTolerance &= isWithinTolerance(iterDiffDzdcR, mpirDzdcR);
                allWithinTolerance &= isWithinTolerance(iterDiffDzdcI, mpirDzdcI);
            }
        }

        // ---- Selected GPU reference inner loop (if TestGpu) — same c as MPIR ----
        HDRFloat<double> gpuD2r{}, gpuD2i{};
        if constexpr (HpShark::TestGpu) {
            {
                BenchmarkTimer gpuTimer;
                gpuTimer.StartTimer();
                uint64_t gpuIterationsCompleted = 0;
                gpuIterationsCompleted =
                    HpShark::EvaluateCriticalOrbitAndDerivs_GPU<SharkFloatParams>(cR,
                                                                                  cI,
                                                                                  period,
                                                                                  gpuZR,
                                                                                  gpuZI,
                                                                                  gpuDzdcR,
                                                                                  gpuDzdcI,
                                                                                  gpuD2r,
                                                                                  gpuD2i,
                                                                                  launchParams,
                                                                                  preparedTables.get(),
                                                                                  0,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  nullptr,
                                                                                  64);
                gpuTimer.StopTimer();
                const double gpuMs = static_cast<double>(gpuTimer.GetDeltaInMs());
                iterTiming.gpuMs = gpuMs;
                gpuCompletedPeriod &= gpuIterationsCompleted == period;
                constexpr const char *gpuReferenceLabel = "GPU reference";
                std::cout << "    " << gpuReferenceLabel << " inner loop timeMs: " << gpuMs
                          << ", completed " << gpuIterationsCompleted << " of " << period << std::endl;
            }

            // GPU vs MPIR comparison (only if MPIR ran)
            if constexpr (HpShark::TestMPIRImpl) {
                mpf_sub(gpuDiffZR, gpuZR, mpirZR);
                mpf_sub(gpuDiffZI, gpuZI, mpirZI);
                mpf_sub(gpuDiffDzdcR, gpuDzdcR, mpirDzdcR);
                mpf_sub(gpuDiffDzdcI, gpuDzdcI, mpirDzdcI);
                char buf[256];
                gmp_snprintf(buf, sizeof(buf), "    GPU iter %u z_real diff:    %+.6Fe", it, gpuDiffZR);
                std::cout << buf << std::endl;
                gmp_snprintf(
                    buf, sizeof(buf), "    GPU iter %u dzdc_real diff: %+.6Fe", it, gpuDiffDzdcR);
                std::cout << buf << std::endl;
                gpuWithinTolerance &= isWithinTolerance(gpuDiffZR, mpirZR);
                gpuWithinTolerance &= isWithinTolerance(gpuDiffZI, mpirZI);
                gpuWithinTolerance &= isWithinTolerance(gpuDiffDzdcR, mpirDzdcR);
                gpuWithinTolerance &= isWithinTolerance(gpuDiffDzdcI, mpirDzdcI);
            }
        }

        perIterTimings.push_back(iterTiming);

        // Newton step + convergence: use best available results (MPIR > GPU)
        // Select source of z/dzdc/d2 for Newton step
        constexpr bool hasSource = HpShark::TestMPIRImpl || HpShark::TestGpu;
        if constexpr (hasSource) {
            mpf_t &useZR = HpShark::TestMPIRImpl ? mpirZR : gpuZR;
            mpf_t &useZI = HpShark::TestMPIRImpl ? mpirZI : gpuZI;
            mpf_t &useDzdcR = HpShark::TestMPIRImpl ? mpirDzdcR : gpuDzdcR;
            mpf_t &useDzdcI = HpShark::TestMPIRImpl ? mpirDzdcI : gpuDzdcI;
            const HDRFloat<double> &useD2r = HpShark::TestMPIRImpl ? mpirD2r : gpuD2r;
            const HDRFloat<double> &useD2i = HpShark::TestMPIRImpl ? mpirD2i : gpuD2i;

            HDRFloat<double> dzr_h(useDzdcR), dzi_h(useDzdcI);
            HdrReduce(dzr_h);
            HdrReduce(dzi_h);
            HDRFloat<double> dzdcNorm = dzr_h.square() + dzi_h.square();
            HdrReduce(dzdcNorm);
            if (dzdcNorm.getMantissa() == 0.0) {
                std::cout << "    break: dzdcNorm==0" << std::endl;
                break;
            }

            if (!ComputeNewtonStep(useZR, useZI, useDzdcR, useDzdcI, stepR, stepI, denom, t1, t2)) {
                std::cout << "    break: denom==0" << std::endl;
                break;
            }

            mpf_sub(cR, cR, stepR);
            mpf_sub(cI, cI, stepI);

            HDRFloat<double> err{};
            const int e =
                ComputeImaginaError(stepR, stepI, useD2r, useD2i, dzdcNorm, normStep, t1, t2, err);

            std::cout << "    err=" << err.template ToString<false>() << " err_exp2=" << e
                      << " targetExp2=" << targetExp << std::endl;
            if (-e >= targetExp) {
                hpConvergedIter = it;
                break;
            }
        }
    }

    // Final correction pass using the selected CPU reference.
    if constexpr (HpShark::TestReferenceImpl) {
        hpCR->MpfToHpGpu(cR, precBits, InjectNoiseInLowOrder::Disable);
        hpCI->MpfToHpGpu(cI, precBits, InjectNoiseInLowOrder::Disable);
        auto preparedTables = HpShark::PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(
            launchParams, *hpCR, *hpCI, SharkFloatParams::GlobalNumUint32, testBase, maxNewtonIters);

        typename SharkFloatParams::Float finalD2r{}, finalD2i{};
        EvaluateOrbitAndDerivative2<SharkFloatParams>(hpCR.get(),
                                                      hpCI.get(),
                                                      period,
                                                      hpZR.get(),
                                                      hpZI.get(),
                                                      hpDzdcR.get(),
                                                      hpDzdcI.get(),
                                                      &finalD2r,
                                                      &finalD2i,
                                                      SharkFloatParams::GlobalNumUint32,
                                                      debugHostCombo,
                                                      preparedTables.get());

        hpZR->HpGpuToMpf(zR);
        hpZI->HpGpuToMpf(zI);
        hpDzdcR->HpGpuToMpf(dzdcR);
        hpDzdcI->HpGpuToMpf(dzdcI);

        if (ComputeNewtonStep(zR, zI, dzdcR, dzdcI, stepR, stepI, denom, t1, t2)) {
            mpf_sub(cR, cR, stepR);
            mpf_sub(cI, cI, stepI);
        }
    }

    // ========== Report results ==========
    std::cout << "\n" << testName << " RESULTS:" << std::endl;
    std::cout << testName << ": MPIR converged in iters " << hpConvergedIter << std::endl;
    constexpr const char *cpuReferenceLabel = "CPU reference";
    std::cout << testName << ": " << cpuReferenceLabel << " per-iteration z/dzdc tolerance "
              << (allWithinTolerance ? "PASS" : "FAIL") << std::endl;
    if constexpr (HpShark::TestGpu) {
        std::cout << testName << ": GPU per-iteration z/dzdc tolerance "
                  << (gpuWithinTolerance && gpuCompletedPeriod ? "PASS" : "FAIL") << std::endl;
    }

    // Always print the summary table (internal — no caller table needed)
    PrintPerfSummaryTable(testName, useMT, perIterTimings, "MPIR", cpuReferenceLabel);

    // In perf-only mode, skip convergence test (it's expected not to converge)
    if (!perfOnly) {
        std::string convName = std::string(testName) + "_Convergence";
        if (hpConvergedIter < maxNewtonIters) {
            Tests.MarkSuccess(nullptr, testBase + 0, convName);
        } else {
            Tests.MarkFailed(nullptr, testBase + 0, convName, "did not converge", "maxNewtonIters");
        }
    }

    // Check per-iteration z/dzdc tolerance
    {
        std::string tolName = std::string(testName) + "_CpuPerIterTolerance";
        char tolStr[256];
        gmp_snprintf(tolStr, sizeof(tolStr), "2^-(exponent:%d)", toleranceBits);
        if (allWithinTolerance) {
            Tests.MarkSuccess(nullptr, testBase + 1, tolName);
        } else {
            Tests.MarkFailed(
                nullptr, testBase + 1, tolName, "per-iteration diff exceeded tolerance", tolStr);
        }
    }

    if constexpr (HpShark::TestGpu) {
        std::string tolName = std::string(testName) + "_GpuPerIterTolerance";
        char tolStr[256];
        gmp_snprintf(tolStr, sizeof(tolStr), "2^-(exponent:%d)", toleranceBits);
        if (gpuCompletedPeriod && gpuWithinTolerance) {
            Tests.MarkSuccess(nullptr, testBase + 2, tolName);
        } else {
            const char *message =
                gpuCompletedPeriod
                    ? "GPU per-iteration diff exceeded tolerance"
                    : "GPU reference implementation did not complete the requested period";
            Tests.MarkFailed(nullptr, testBase + 2, tolName, message, tolStr);
        }
    }

    // Cleanup
    mpf_clear(cR);
    mpf_clear(cI);
    mpf_clear(zR);
    mpf_clear(zI);
    mpf_clear(dzdcR);
    mpf_clear(dzdcI);
    mpf_clear(stepR);
    mpf_clear(stepI);
    mpf_clear(denom);
    mpf_clear(t1);
    mpf_clear(t2);
    mpf_clear(normStep);
    mpf_clear(mpirZR);
    mpf_clear(mpirZI);
    mpf_clear(mpirDzdcR);
    mpf_clear(mpirDzdcI);
    mpf_clear(iterDiffZR);
    mpf_clear(iterDiffZI);
    mpf_clear(iterDiffDzdcR);
    mpf_clear(iterDiffDzdcI);
    mpf_clear(gpuZR);
    mpf_clear(gpuZI);
    mpf_clear(gpuDzdcR);
    mpf_clear(gpuDzdcI);
    mpf_clear(gpuDiffZR);
    mpf_clear(gpuDiffZI);
    mpf_clear(gpuDiffDzdcR);
    mpf_clear(gpuDiffDzdcI);
    mpf_clear(tolerance);
    mpf_clear(absDiff);
    mpf_clear(absRef);
    mpf_clear(relError);

    return Tests.CheckAllTestsPassed();
}

template <class SharkFloatParams, Operator referenceOperator>
bool
TestNewtonRaphsonView5(TestTracker &Tests,
                       int testBase,
                       const HpShark::LaunchParams &launchParams,
                       uint64_t iterCountOverride,
                       bool useMT,
                       int numRepeats)
{
    const char *cRealStr =
        "-5."
        "48205748070475708458212567546733029376699274622882453824444834594995999680895291"
        "29972505947379718e-01";
    const char *cImagStr =
        "-5."
        "77570838903603842805108982201850558675551728458255317158378952895736909832155423"
        "61901805676878083e-01";

    constexpr uint64_t expectedPeriod = 16045;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    mpf_t mpfCReal, mpfCImag;
    mpf_init(mpfCReal);
    mpf_init(mpfCImag);
    mpf_set_str(mpfCReal, cRealStr, 10);
    mpf_set_str(mpfCImag, cImagStr, 10);

    bool result = RunNewtonRaphsonTest<SharkFloatParams, referenceOperator>(Tests,
                                                                            testBase,
                                                                            "NR_View5",
                                                                            mpfCReal,
                                                                            mpfCImag,
                                                                            expectedPeriod,
                                                                            launchParams,
                                                                            iterCountOverride,
                                                                            useMT,
                                                                            numRepeats);

    mpf_clear(mpfCReal);
    mpf_clear(mpfCImag);
    return result;
}

template <class SharkFloatParams, Operator referenceOperator>
bool
TestNewtonRaphsonView30(TestTracker &Tests,
                        int testBase,
                        const HpShark::LaunchParams &launchParams,
                        uint64_t iterCountOverride,
                        bool useMT,
                        int numRepeats)
{
#include "LargeCoords30.h"

    constexpr uint64_t expectedPeriod = 669772;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    mpf_t mpfCReal, mpfCImag;
    Hex64StringToMpf_Exact(strXHex, mpfCReal);
    Hex64StringToMpf_Exact(strYHex, mpfCImag);

    bool result = RunNewtonRaphsonTest<SharkFloatParams, referenceOperator>(Tests,
                                                                            testBase,
                                                                            "NR_View30",
                                                                            mpfCReal,
                                                                            mpfCImag,
                                                                            expectedPeriod,
                                                                            launchParams,
                                                                            iterCountOverride,
                                                                            useMT,
                                                                            numRepeats);

    mpf_clear(mpfCReal);
    mpf_clear(mpfCImag);
    return result;
}

template <class SharkFloatParams, Operator referenceOperator>
bool
TestNewtonRaphsonView32(TestTracker &Tests,
                        int testBase,
                        const HpShark::LaunchParams &launchParams,
                        uint64_t iterCountOverride,
                        bool useMT,
                        int numRepeats)
{
#include "LargeCoords32.h"

    constexpr uint64_t expectedPeriod = 27'209'300;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    mpf_t mpfCReal, mpfCImag;
    mpf_init(mpfCReal);
    mpf_init(mpfCImag);
    mpf_set_str(mpfCReal, strX, 10);
    mpf_set_str(mpfCImag, strY, 10);

    bool result = RunNewtonRaphsonTest<SharkFloatParams, referenceOperator>(Tests,
                                                                            testBase,
                                                                            "NR_View32",
                                                                            mpfCReal,
                                                                            mpfCImag,
                                                                            expectedPeriod,
                                                                            launchParams,
                                                                            iterCountOverride,
                                                                            useMT,
                                                                            numRepeats);

    mpf_clear(mpfCReal);
    mpf_clear(mpfCImag);
    return result;
}

template bool TestNewtonRaphsonView5<SharkParamsNR7, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR1, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR2, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR3, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR4, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR5, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR6, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR8, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR9, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR10, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR11, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNR12, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNRSharedOnly256, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNRSharedOnly512, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView5<SharkParamsNRSharedOnly1024, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView30<SharkParamsNR7, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
template bool TestNewtonRaphsonView32<SharkParamsNR9, Operator::ReferenceOrbit2>(
    TestTracker &, int, const HpShark::LaunchParams &, uint64_t, bool, int);
