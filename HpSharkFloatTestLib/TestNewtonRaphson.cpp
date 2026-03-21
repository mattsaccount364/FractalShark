#include "TestNewtonRaphson.h"

#include "BenchmarkTimer.h"
#include "DebugChecksumHost.h"
#include "HDRFloat.h"
#include "HighPrecision.h"
#include "HpSharkFloat.h"
#include "KernelInvoke.h"
#include "MpirOrbitEval.h"
#include "ReferenceAdd.h"
#include "ReferenceNTT2.h"
#include "ReferenceReferenceOrbit.h"
#include "TestTracker.h"

#include <cstdint>
#include <gmp.h>
#include <iostream>
#include <memory>
#include <string>

// Wrapper around shared MT orbit eval for the test benchmark.
static void RunMpirOrbitWithD2(
    mpf_t cR, mpf_t cI, uint64_t period,
    mpf_t outZR, mpf_t outZI,
    mpf_t outDzdcR, mpf_t outDzdcI,
    HDRFloat<double> &d2r_out, HDRFloat<double> &d2i_out)
{
    const mp_bitcnt_t prec = mpf_get_prec(cR);

    mpf_complex c_coord;
    mpf_complex_init(c_coord, prec);
    mpf_set(c_coord.re, cR); mpf_set(c_coord.im, cI);

    mpf_complex z_coord, dzdc_deriv;
    mpf_complex_init(z_coord, prec);
    mpf_complex_init(dzdc_deriv, prec);

    EvaluateCriticalOrbitAndDerivsMT(
        c_coord, period, z_coord, dzdc_deriv,
        d2r_out, d2i_out, prec, prec);

    mpf_set(outZR, z_coord.re); mpf_set(outZI, z_coord.im);
    mpf_set(outDzdcR, dzdc_deriv.re); mpf_set(outDzdcI, dzdc_deriv.im);

    mpf_complex_clear(z_coord);
    mpf_complex_clear(dzdc_deriv);
    mpf_complex_clear(c_coord);
}

// Compute Newton step: step = z / dzdc (complex division via separate reals).
// Returns false if dzdc is zero.
static bool ComputeNewtonStep(
    mpf_t zR, mpf_t zI,
    mpf_t dzdcR, mpf_t dzdcI,
    mpf_t stepR, mpf_t stepI,
    mpf_t denom, mpf_t t1, mpf_t t2)
{
    // denom = |dzdc|^2 = dzdcR^2 + dzdcI^2
    mpf_mul(t1, dzdcR, dzdcR);
    mpf_mul(t2, dzdcI, dzdcI);
    mpf_add(denom, t1, t2);
    if (mpf_cmp_ui(denom, 0) == 0) return false;

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
static int ComputeImaginaError(
    mpf_t stepR, mpf_t stepI,
    HDRFloat<double> d2r, HDRFloat<double> d2i,
    HDRFloat<double> dzdcNorm,
    mpf_t normStep, mpf_t t1, mpf_t t2,
    HDRFloat<double> &err_out)
{

    mpf_mul(t1, stepR, stepR);
    mpf_mul(t2, stepI, stepI);
    mpf_add(normStep, t1, t2);  // |step|^2

    HDRFloat<double> normStep_hdr(normStep);
    HdrReduce(normStep_hdr);
    HDRFloat<double> normStep2 = normStep_hdr.square();  // |step|^4
    HdrReduce(normStep2);

    HDRFloat<double> d2Norm = d2r.square() + d2i.square();  // |d2|^2
    HdrReduce(d2Norm);

    err_out = (normStep2 * d2Norm) / dzdcNorm;
    HdrReduce(err_out);

    return (int)err_out.getExp();
}

template <class SharkFloatParams>
static bool
RunNewtonRaphsonTest(
    TestTracker &Tests,
    int testBase,
    const char *testName,
    mpf_t mpfCReal,
    mpf_t mpfCImag,
    uint64_t period)
{
    constexpr uint32_t maxNewtonIters = 32;
    const int precBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    const int targetExp = precBits * 2;

    // Shared temporaries for Newton step computation
    mpf_t stepR, stepI, denom, t1, t2, normStep;
    mpf_init(stepR); mpf_init(stepI); mpf_init(denom);
    mpf_init(t1); mpf_init(t2); mpf_init(normStep);

    // ========== Newton refinement setup ==========
    mpf_t cR, cI;
    mpf_init(cR); mpf_init(cI);
    mpf_set(cR, mpfCReal); mpf_set(cI, mpfCImag);

    mpf_t zR, zI, dzdcR, dzdcI;
    mpf_init(zR); mpf_init(zI);
    mpf_init(dzdcR); mpf_init(dzdcI);

    auto hpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpZR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpZI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    DebugHostCombo<SharkFloatParams> debugHostCombo;

    // MPIR comparison temporaries
    mpf_t mpirZR, mpirZI, mpirDzdcR, mpirDzdcI;
    mpf_init(mpirZR); mpf_init(mpirZI);
    mpf_init(mpirDzdcR); mpf_init(mpirDzdcI);

    mpf_t iterDiffZR, iterDiffZI, iterDiffDzdcR, iterDiffDzdcI;
    mpf_init(iterDiffZR); mpf_init(iterDiffZI);
    mpf_init(iterDiffDzdcR); mpf_init(iterDiffDzdcI);

    // Tolerance for per-iteration comparison
    const int singleOpMargin = 98;
    int logPeriod = 1;
    {
        uint64_t p = period;
        while (p > 1) { p >>= 1; ++logPeriod; }
    }
    const int toleranceBits = precBits - singleOpMargin * logPeriod;

    mpf_t tolerance, absDiff, absRef, relError;
    mpf_init(tolerance); mpf_init(absDiff); mpf_init(absRef); mpf_init(relError);
    mpf_set_ui(tolerance, 1);
    mpf_div_2exp(tolerance, tolerance, toleranceBits);

    bool allWithinTolerance = true;
    uint32_t hpConvergedIter = maxNewtonIters;
    double totalCpuMs = 0.0;

    // ========== Lockstep Newton refinement ==========
    for (uint32_t it = 0; it < maxNewtonIters; ++it) {
        std::cout << "  " << testName << " Newton iter " << it << std::endl;

        // CPU-ref inner loop (only if TestReferenceImpl)
        typename SharkFloatParams::Float hpD2r{}, hpD2i{};
        if constexpr (HpShark::TestReferenceImpl) {
            hpCR->MpfToHpGpu(cR, precBits, InjectNoiseInLowOrder::Disable);
            hpCI->MpfToHpGpu(cI, precBits, InjectNoiseInLowOrder::Disable);

            BenchmarkTimer cpuTimer;
            EvaluateOrbitAndDerivative<SharkFloatParams>(
                hpCR.get(), hpCI.get(), period,
                hpZR.get(), hpZI.get(), hpDzdcR.get(), hpDzdcI.get(),
                &hpD2r, &hpD2i,
                debugHostCombo);
            const double cpuMs = static_cast<double>(cpuTimer.GetDeltaInMs());
            totalCpuMs += cpuMs;
            std::cout << "    CPU inner loop: " << cpuMs << " ms" << std::endl;

            hpZR->HpGpuToMpf(zR);
            hpZI->HpGpuToMpf(zI);
            hpDzdcR->HpGpuToMpf(dzdcR);
            hpDzdcI->HpGpuToMpf(dzdcI);
        }

        // MPIR reference inner loop
        HDRFloat<double> mpirD2r{}, mpirD2i{};
        RunMpirOrbitWithD2(cR, cI, period,
            mpirZR, mpirZI, mpirDzdcR, mpirDzdcI,
            mpirD2r, mpirD2i);

        // Per-iteration CPU vs MPIR comparison (only if TestReferenceImpl)
        if constexpr (HpShark::TestReferenceImpl) {
            mpf_sub(iterDiffZR, zR, mpirZR);
            mpf_sub(iterDiffZI, zI, mpirZI);
            mpf_sub(iterDiffDzdcR, dzdcR, mpirDzdcR);
            mpf_sub(iterDiffDzdcI, dzdcI, mpirDzdcI);

            char buf[256];
            gmp_snprintf(buf, sizeof(buf), "    iter %u z_real diff:    %+.6Fe", it, iterDiffZR);
            std::cout << buf << std::endl;
            gmp_snprintf(buf, sizeof(buf), "    iter %u z_imag diff:    %+.6Fe", it, iterDiffZI);
            std::cout << buf << std::endl;
            gmp_snprintf(buf, sizeof(buf), "    iter %u dzdc_real diff: %+.6Fe", it, iterDiffDzdcR);
            std::cout << buf << std::endl;
            gmp_snprintf(buf, sizeof(buf), "    iter %u dzdc_imag diff: %+.6Fe", it, iterDiffDzdcI);
            std::cout << buf << std::endl;

            HDRFloat<double> hpD2rD = hpD2r;
            HDRFloat<double> hpD2iD = hpD2i;
            std::cout << "    iter " << it
                      << " d2r HP=" << hpD2rD.ToString<false>()
                      << " MPIR=" << mpirD2r.ToString<false>() << std::endl;
            std::cout << "    iter " << it
                      << " d2i HP=" << hpD2iD.ToString<false>()
                      << " MPIR=" << mpirD2i.ToString<false>() << std::endl;
        }

        // Check per-iteration tolerance for CPU-ref vs MPIR (only if TestReferenceImpl)
        if constexpr (HpShark::TestReferenceImpl) {
            auto checkIterTolerance = [&](mpf_t diff, mpf_t ref) {
                mpf_abs(absDiff, diff);
                mpf_abs(absRef, ref);
                if (mpf_sgn(absRef) == 0) {
                    return mpf_cmp(absDiff, tolerance) <= 0;
                } else {
                    mpf_div(relError, absDiff, absRef);
                    return mpf_cmp(relError, tolerance) <= 0;
                }
            };

            if (!checkIterTolerance(iterDiffZR, mpirZR)) allWithinTolerance = false;
            if (!checkIterTolerance(iterDiffZI, mpirZI)) allWithinTolerance = false;
            if (!checkIterTolerance(iterDiffDzdcR, mpirDzdcR)) allWithinTolerance = false;
            if (!checkIterTolerance(iterDiffDzdcI, mpirDzdcI)) allWithinTolerance = false;
        }

        // Newton step uses MPIR results (authoritative ground truth)
        HDRFloat<double> dzr_h(mpirDzdcR), dzi_h(mpirDzdcI);
        HdrReduce(dzr_h); HdrReduce(dzi_h);
        HDRFloat<double> dzdcNorm = dzr_h.square() + dzi_h.square();
        HdrReduce(dzdcNorm);
        if (dzdcNorm.getMantissa() == 0.0) {
            std::cout << "    break: dzdcNorm==0" << std::endl;
            break;
        }

        if (!ComputeNewtonStep(mpirZR, mpirZI, mpirDzdcR, mpirDzdcI, stepR, stepI, denom, t1, t2)) {
            std::cout << "    break: denom==0" << std::endl;
            break;
        }

        // c = c - step (MPIR trajectory)
        mpf_sub(cR, cR, stepR);
        mpf_sub(cI, cI, stepI);

        // Imagina convergence check (using MPIR d2)
        HDRFloat<double> err{};
        const int e = ComputeImaginaError(
            stepR, stepI, mpirD2r, mpirD2i, dzdcNorm,
            normStep, t1, t2, err);

        std::cout << "    err=" << err.ToString<false>() << " e=" << e
                  << " targetExp=" << targetExp << std::endl;
        if (-e >= targetExp) {
            hpConvergedIter = it;
            break;
        }
    }

    // Final correction pass (only if TestReferenceImpl)
    if constexpr (HpShark::TestReferenceImpl) {
        hpCR->MpfToHpGpu(cR, precBits, InjectNoiseInLowOrder::Disable);
        hpCI->MpfToHpGpu(cI, precBits, InjectNoiseInLowOrder::Disable);

        typename SharkFloatParams::Float finalD2r{}, finalD2i{};
        EvaluateOrbitAndDerivative<SharkFloatParams>(
            hpCR.get(), hpCI.get(), period,
            hpZR.get(), hpZI.get(), hpDzdcR.get(), hpDzdcI.get(),
            &finalD2r, &finalD2i,
            debugHostCombo);

        hpZR->HpGpuToMpf(zR);
        hpZI->HpGpuToMpf(zI);
        hpDzdcR->HpGpuToMpf(dzdcR);
        hpDzdcI->HpGpuToMpf(dzdcI);

        if (ComputeNewtonStep(zR, zI, dzdcR, dzdcI, stepR, stepI, denom, t1, t2)) {
            mpf_sub(cR, cR, stepR);
            mpf_sub(cI, cI, stepI);
        }
    }

    // ========== GPU Newton refinement (if TestGpu enabled) ==========
    uint32_t gpuConvergedIter = maxNewtonIters;
    mpf_t gpuCR, gpuCI;
    mpf_init(gpuCR); mpf_init(gpuCI);
    mpf_set(gpuCR, mpfCReal); mpf_set(gpuCI, mpfCImag);
    BenchmarkTimer gpuTimer, cpuRefTimer;
    bool gpuAllWithinTolerance = true;

    if constexpr (HpShark::TestGpu) {
        HpShark::LaunchParams launchParams{2, 32};
        typename SharkFloatParams::Float hdrRadiusY{1.0f};

        auto gpuInitCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto gpuInitCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        gpuInitCR->MpfToHpGpu(mpfCReal, precBits, InjectNoiseInLowOrder::Disable);
        gpuInitCI->MpfToHpGpu(mpfCImag, precBits, InjectNoiseInLowOrder::Disable);

        auto combo = HpShark::InitHpSharkReferenceKernel<SharkFloatParams>(
            launchParams, hdrRadiusY, *gpuInitCR, *gpuInitCI);

        mpf_t gpuZR, gpuZI, gpuDzdcR, gpuDzdcI;
        mpf_init(gpuZR); mpf_init(gpuZI);
        mpf_init(gpuDzdcR); mpf_init(gpuDzdcI);

        mpf_t gpuMpirZR, gpuMpirZI, gpuMpirDzdcR, gpuMpirDzdcI;
        mpf_init(gpuMpirZR); mpf_init(gpuMpirZI);
        mpf_init(gpuMpirDzdcR); mpf_init(gpuMpirDzdcI);

        mpf_t gpuIterDiffZR, gpuIterDiffZI, gpuIterDiffDzdcR, gpuIterDiffDzdcI;
        mpf_init(gpuIterDiffZR); mpf_init(gpuIterDiffZI);
        mpf_init(gpuIterDiffDzdcR); mpf_init(gpuIterDiffDzdcI);

        for (uint32_t it = 0; it < maxNewtonIters; ++it) {
            std::cout << "  " << testName << " GPU Newton iter " << it << std::endl;

            {
                ScopedBenchmarkStopper s{gpuTimer};

                // Update combo with current c, reset z/dzdc/d2
                auto tmpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
                auto tmpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
                tmpCR->MpfToHpGpu(gpuCR, precBits, InjectNoiseInLowOrder::Disable);
                tmpCI->MpfToHpGpu(gpuCI, precBits, InjectNoiseInLowOrder::Disable);

                combo->Multiply.A = HpSharkFloat<SharkFloatParams>{};  // z = 0
                combo->Multiply.B = HpSharkFloat<SharkFloatParams>{};
                combo->Add.C_A = *tmpCR;  // c_real
                combo->Add.E_B = *tmpCI;  // c_imag
                combo->Multiply.DzdcReal = HpSharkFloat<SharkFloatParams>{};
                combo->Multiply.DzdcImag = HpSharkFloat<SharkFloatParams>{};
                combo->d2Real = typename SharkFloatParams::Float{};
                combo->d2Imag = typename SharkFloatParams::Float{};
                combo->PeriodicityStatus = PeriodicityResult::Unknown;
                combo->MaxRuntimeIters = period;

                HpShark::InvokeHpSharkReferenceKernel<SharkFloatParams>(
                    launchParams, *combo, period);
            }

            // Read GPU results
            combo->Multiply.A.HpGpuToMpf(gpuZR);
            combo->Multiply.B.HpGpuToMpf(gpuZI);
            combo->Multiply.DzdcReal.HpGpuToMpf(gpuDzdcR);
            combo->Multiply.DzdcImag.HpGpuToMpf(gpuDzdcI);

            // MPIR reference for GPU comparison
            HDRFloat<double> gpuMpirD2r{}, gpuMpirD2i{};
            RunMpirOrbitWithD2(gpuCR, gpuCI, period,
                gpuMpirZR, gpuMpirZI, gpuMpirDzdcR, gpuMpirDzdcI,
                gpuMpirD2r, gpuMpirD2i);

            // GPU vs MPIR per-iteration diff
            mpf_sub(gpuIterDiffZR, gpuZR, gpuMpirZR);
            mpf_sub(gpuIterDiffZI, gpuZI, gpuMpirZI);
            mpf_sub(gpuIterDiffDzdcR, gpuDzdcR, gpuMpirDzdcR);
            mpf_sub(gpuIterDiffDzdcI, gpuDzdcI, gpuMpirDzdcI);

            {
                char dbuf[256];
                gmp_snprintf(dbuf, sizeof(dbuf), "    GPU iter %u z_real diff:    %+.6Fe", it, gpuIterDiffZR);
                std::cout << dbuf << std::endl;
                gmp_snprintf(dbuf, sizeof(dbuf), "    GPU iter %u z_imag diff:    %+.6Fe", it, gpuIterDiffZI);
                std::cout << dbuf << std::endl;
                gmp_snprintf(dbuf, sizeof(dbuf), "    GPU iter %u dzdc_real diff: %+.6Fe", it, gpuIterDiffDzdcR);
                std::cout << dbuf << std::endl;
                gmp_snprintf(dbuf, sizeof(dbuf), "    GPU iter %u dzdc_imag diff: %+.6Fe", it, gpuIterDiffDzdcI);
                std::cout << dbuf << std::endl;
            }

            auto gpuCheckIterTolerance = [&](mpf_t diff, mpf_t ref) {
                mpf_abs(absDiff, diff);
                mpf_abs(absRef, ref);
                if (mpf_sgn(absRef) == 0) {
                    return mpf_cmp(absDiff, tolerance) <= 0;
                } else {
                    mpf_div(relError, absDiff, absRef);
                    return mpf_cmp(relError, tolerance) <= 0;
                }
            };

            if (!gpuCheckIterTolerance(gpuIterDiffZR, gpuMpirZR)) gpuAllWithinTolerance = false;
            if (!gpuCheckIterTolerance(gpuIterDiffZI, gpuMpirZI)) gpuAllWithinTolerance = false;
            if (!gpuCheckIterTolerance(gpuIterDiffDzdcR, gpuMpirDzdcR)) gpuAllWithinTolerance = false;
            if (!gpuCheckIterTolerance(gpuIterDiffDzdcI, gpuMpirDzdcI)) gpuAllWithinTolerance = false;

            // Check |dzdc| != 0
            HDRFloat<double> gdzr(gpuDzdcR), gdzi(gpuDzdcI);
            HdrReduce(gdzr); HdrReduce(gdzi);
            HDRFloat<double> gdzdcNorm = gdzr.square() + gdzi.square();
            HdrReduce(gdzdcNorm);
            if (gdzdcNorm.getMantissa() == 0.0) break;

            // Newton step
            if (!ComputeNewtonStep(gpuZR, gpuZI, gpuDzdcR, gpuDzdcI,
                                   stepR, stepI, denom, t1, t2))
                break;

            mpf_sub(gpuCR, gpuCR, stepR);
            mpf_sub(gpuCI, gpuCI, stepI);

            // Convergence
            HDRFloat<double> gpuD2r = combo->d2Real;
            HDRFloat<double> gpuD2i = combo->d2Imag;
            HDRFloat<double> gpuErr{};
            const int ge = ComputeImaginaError(
                stepR, stepI, gpuD2r, gpuD2i, gdzdcNorm,
                normStep, t1, t2, gpuErr);

            std::cout << "    GPU err=" << gpuErr.ToString<false>() << " e=" << ge << std::endl;
            if (-ge >= targetExp) {
                gpuConvergedIter = it;
                break;
            }
        }

        // GPU final correction
        {
            auto tmpCR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            auto tmpCI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
            tmpCR->MpfToHpGpu(gpuCR, precBits, InjectNoiseInLowOrder::Disable);
            tmpCI->MpfToHpGpu(gpuCI, precBits, InjectNoiseInLowOrder::Disable);

            combo->Multiply.A = HpSharkFloat<SharkFloatParams>{};
            combo->Multiply.B = HpSharkFloat<SharkFloatParams>{};
            combo->Add.C_A = *tmpCR;
            combo->Add.E_B = *tmpCI;
            combo->Multiply.DzdcReal = HpSharkFloat<SharkFloatParams>{};
            combo->Multiply.DzdcImag = HpSharkFloat<SharkFloatParams>{};
            combo->d2Real = typename SharkFloatParams::Float{};
            combo->d2Imag = typename SharkFloatParams::Float{};
            combo->MaxRuntimeIters = period;

            HpShark::InvokeHpSharkReferenceKernel<SharkFloatParams>(
                launchParams, *combo, period);

            combo->Multiply.A.HpGpuToMpf(gpuZR);
            combo->Multiply.B.HpGpuToMpf(gpuZI);
            combo->Multiply.DzdcReal.HpGpuToMpf(gpuDzdcR);
            combo->Multiply.DzdcImag.HpGpuToMpf(gpuDzdcI);

            if (ComputeNewtonStep(gpuZR, gpuZI, gpuDzdcR, gpuDzdcI,
                                  stepR, stepI, denom, t1, t2)) {
                mpf_sub(gpuCR, gpuCR, stepR);
                mpf_sub(gpuCI, gpuCI, stepI);
            }
        }

        mpf_clear(gpuZR); mpf_clear(gpuZI);
        mpf_clear(gpuDzdcR); mpf_clear(gpuDzdcI);
        mpf_clear(gpuMpirZR); mpf_clear(gpuMpirZI);
        mpf_clear(gpuMpirDzdcR); mpf_clear(gpuMpirDzdcI);
        mpf_clear(gpuIterDiffZR); mpf_clear(gpuIterDiffZI);
        mpf_clear(gpuIterDiffDzdcR); mpf_clear(gpuIterDiffDzdcI);

        HpShark::ShutdownHpSharkReferenceKernel<SharkFloatParams>(
            launchParams, *combo, nullptr);
    }

    // ========== Report results ==========
    std::cout << testName << ": CPU-ref converged in " << hpConvergedIter << " iters" << std::endl;
    std::cout << testName << ": CPU-ref total inner loop time: " << totalCpuMs << " ms" << std::endl;
    if constexpr (HpShark::TestGpu) {
        std::cout << testName << ": GPU converged in " << gpuConvergedIter << " iters" << std::endl;
        const double gpuMs = static_cast<double>(gpuTimer.GetDeltaInMs());
        std::cout << testName << ": GPU total inner loop time: " << gpuMs << " ms" << std::endl;
        if (totalCpuMs > 0 && gpuMs > 0) {
            std::cout << testName << ": Speedup (CPU/GPU): " << (totalCpuMs / gpuMs) << "x" << std::endl;
        }
    }
    std::cout << testName << ": per-iteration z/dzdc tolerance "
              << (allWithinTolerance ? "PASS" : "FAIL") << std::endl;

    // Check CPU-ref convergence
    {
        std::string convName = std::string(testName) + "_Convergence";
        if (hpConvergedIter < maxNewtonIters) {
            Tests.MarkSuccess(nullptr, testBase + 0, convName);
        } else {
            Tests.MarkFailed(nullptr, testBase + 0, convName,
                             "did not converge", "maxNewtonIters");
        }
    }

    // Check per-iteration z/dzdc tolerance
    {
        std::string tolName = std::string(testName) + "_PerIterTolerance";
        char tolStr[256];
        gmp_snprintf(tolStr, sizeof(tolStr), "2^-%d", toleranceBits);
        if (allWithinTolerance) {
            Tests.MarkSuccess(nullptr, testBase + 1, tolName);
        } else {
            Tests.MarkFailed(nullptr, testBase + 1, tolName,
                             "per-iteration diff exceeded tolerance", tolStr);
        }
    }

    // Check GPU convergence (if enabled)
    if constexpr (HpShark::TestGpu) {
        std::string gpuConvName = std::string(testName) + "_GpuConvergence";
        if (gpuConvergedIter < maxNewtonIters) {
            Tests.MarkSuccess(nullptr, testBase + 2, gpuConvName);
        } else {
            Tests.MarkFailed(nullptr, testBase + 2, gpuConvName,
                             "GPU did not converge", "maxNewtonIters");
        }

        // Compare GPU refined c* vs CPU-ref refined c*
        mpf_t diffGpuCR, diffGpuCI;
        mpf_init(diffGpuCR); mpf_init(diffGpuCI);
        mpf_sub(diffGpuCR, gpuCR, cR);
        mpf_sub(diffGpuCI, gpuCI, cI);

        char buf[256];
        gmp_snprintf(buf, sizeof(buf), "  GPU vs CPU-ref c_real diff: %+.6Fe", diffGpuCR);
        std::cout << buf << std::endl;
        gmp_snprintf(buf, sizeof(buf), "  GPU vs CPU-ref c_imag diff: %+.6Fe", diffGpuCI);
        std::cout << buf << std::endl;

        // testBase+3: GPU vs MPIR per-iteration tolerance
        {
            std::string gpuMpirTolName = std::string(testName) + "_GpuVsMpirTolerance";
            char tolStr[256];
            gmp_snprintf(tolStr, sizeof(tolStr), "2^-%d", toleranceBits);
            std::cout << testName << ": GPU per-iteration z/dzdc tolerance "
                      << (gpuAllWithinTolerance ? "PASS" : "FAIL") << std::endl;
            if (gpuAllWithinTolerance) {
                Tests.MarkSuccess(nullptr, testBase + 3, gpuMpirTolName);
            } else {
                Tests.MarkFailed(nullptr, testBase + 3, gpuMpirTolName,
                                 "GPU per-iteration diff exceeded tolerance", tolStr);
            }
        }

        // testBase+4: GPU vs CPU-ref final c* tolerance
        {
            std::string gpuCTolName = std::string(testName) + "_GpuVsCpuRefC";
            bool cWithinTolerance = true;

            auto cToleranceCheck = [&](mpf_t diff, mpf_t ref) {
                mpf_abs(absDiff, diff);
                mpf_abs(absRef, ref);
                if (mpf_sgn(absRef) == 0) {
                    return mpf_cmp(absDiff, tolerance) <= 0;
                } else {
                    mpf_div(relError, absDiff, absRef);
                    return mpf_cmp(relError, tolerance) <= 0;
                }
            };

            if (!cToleranceCheck(diffGpuCR, cR)) cWithinTolerance = false;
            if (!cToleranceCheck(diffGpuCI, cI)) cWithinTolerance = false;

            char tolStr[256];
            gmp_snprintf(tolStr, sizeof(tolStr), "2^-%d", toleranceBits);
            if (cWithinTolerance) {
                Tests.MarkSuccess(nullptr, testBase + 4, gpuCTolName);
            } else {
                Tests.MarkFailed(nullptr, testBase + 4, gpuCTolName,
                                 "GPU vs CPU-ref c* diff exceeded tolerance", tolStr);
            }
        }

        mpf_clear(diffGpuCR); mpf_clear(diffGpuCI);
    }

    // Cleanup
    mpf_clear(cR); mpf_clear(cI);
    mpf_clear(gpuCR); mpf_clear(gpuCI);
    mpf_clear(zR); mpf_clear(zI);
    mpf_clear(dzdcR); mpf_clear(dzdcI);
    mpf_clear(stepR); mpf_clear(stepI); mpf_clear(denom);
    mpf_clear(t1); mpf_clear(t2); mpf_clear(normStep);
    mpf_clear(mpirZR); mpf_clear(mpirZI);
    mpf_clear(mpirDzdcR); mpf_clear(mpirDzdcI);
    mpf_clear(iterDiffZR); mpf_clear(iterDiffZI);
    mpf_clear(iterDiffDzdcR); mpf_clear(iterDiffDzdcI);
    mpf_clear(tolerance); mpf_clear(absDiff);
    mpf_clear(absRef); mpf_clear(relError);

    return Tests.CheckAllTestsPassed();
}

template <class SharkFloatParams>
bool
TestNewtonRaphsonView5(TestTracker &Tests,
                       int testBase)
{
    const char *cRealStr = "-5."
        "48205748070475708458212567546733029376699274622882453824444834594995999680895291"
        "29972505947379718e-01";
    const char *cImagStr = "-5."
        "77570838903603842805108982201850558675551728458255317158378952895736909832155423"
        "61901805676878083e-01";

    constexpr uint64_t expectedPeriod = 16045;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    mpf_t mpfCReal, mpfCImag;
    mpf_init(mpfCReal);
    mpf_init(mpfCImag);
    mpf_set_str(mpfCReal, cRealStr, 10);
    mpf_set_str(mpfCImag, cImagStr, 10);

    bool result = RunNewtonRaphsonTest<SharkFloatParams>(
        Tests, testBase, "NR_View5", mpfCReal, mpfCImag, expectedPeriod);

    mpf_clear(mpfCReal); mpf_clear(mpfCImag);
    return result;
}

template <class SharkFloatParams>
bool
TestNewtonRaphsonView30(TestTracker &Tests,
                        int testBase)
{
#include "..\FractalSharkLib\LargeCoords30.h"

    constexpr uint64_t expectedPeriod = 669772;

    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    mpf_t mpfCReal, mpfCImag;
    Hex64StringToMpf_Exact(strXHex, mpfCReal);
    Hex64StringToMpf_Exact(strYHex, mpfCImag);

    bool result = RunNewtonRaphsonTest<SharkFloatParams>(
        Tests, testBase, "NR_View30", mpfCReal, mpfCImag, expectedPeriod);

    mpf_clear(mpfCReal); mpf_clear(mpfCImag);
    return result;
}

template bool TestNewtonRaphsonView5<SharkParamsNR7>(TestTracker &, int);
template bool TestNewtonRaphsonView30<SharkParamsNR7>(TestTracker &, int);

// ========== Single-iteration NR Multiply test ==========
template <class SharkFloatParams>
bool
TestSingleNRMultiply(TestTracker &Tests, int testBase)
{
    const int precBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    const int toleranceBits = precBits - 10;
    mpf_t tolerance;
    mpf_init(tolerance);
    mpf_set_ui(tolerance, 1);
    mpf_div_2exp(tolerance, tolerance, toleranceBits);

    mpf_t gpuVal, diff, absDiff, absRef, relError;
    mpf_init(gpuVal); mpf_init(diff);
    mpf_init(absDiff); mpf_init(absRef); mpf_init(relError);

    auto checkProduct = [&](HpSharkFloat<SharkFloatParams> &gpuOutput,
                            mpf_t mpirRef,
                            int testIndex,
                            const char *name) -> bool {
        gpuOutput.HpGpuToMpf(gpuVal);
        mpf_sub(diff, gpuVal, mpirRef);
        mpf_abs(absDiff, diff);
        mpf_abs(absRef, mpirRef);

        bool pass;
        if (mpf_sgn(absRef) == 0) {
            pass = mpf_cmp(absDiff, tolerance) <= 0;
        } else {
            mpf_div(relError, absDiff, absRef);
            pass = mpf_cmp(relError, tolerance) <= 0;
        }

        std::string testName = std::string("NRMultiply_") + name;
        char buf[256];
        gmp_snprintf(buf, sizeof(buf), "diff=%+.6Fe", diff);
        char tolStr[256];
        gmp_snprintf(tolStr, sizeof(tolStr), "2^-%d", toleranceBits);

        if (pass) {
            Tests.MarkSuccess(nullptr, testIndex, testName);
        } else {
            Tests.MarkFailed(nullptr, testIndex, testName, buf, tolStr);
        }
        return pass;
    };

    // Generate random inputs
    auto hpA = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpB = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpDzdcI = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    hpA->GenerateRandomNumber2();
    hpB->GenerateRandomNumber2();
    hpDzdcR->GenerateRandomNumber2();
    hpDzdcI->GenerateRandomNumber2();

    // Get MPIR equivalents
    mpf_t mpfA, mpfB, mpfDzdcR, mpfDzdcI;
    mpf_init(mpfA); mpf_init(mpfB);
    mpf_init(mpfDzdcR); mpf_init(mpfDzdcI);
    hpA->HpGpuToMpf(mpfA);
    hpB->HpGpuToMpf(mpfB);
    hpDzdcR->HpGpuToMpf(mpfDzdcR);
    hpDzdcI->HpGpuToMpf(mpfDzdcI);

    // Output HpSharkFloats
    auto outXX = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outXY = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outYY = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outW0 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outW1 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outW2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outW3 = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    DebugHostCombo<SharkFloatParams> debugHostCombo;

    MultiplyHelperFFT2<SharkFloatParams>(
        hpA.get(), hpB.get(),
        outXX.get(), outXY.get(), outYY.get(),
        hpDzdcR.get(), hpDzdcI.get(),
        outW0.get(), outW1.get(), outW2.get(), outW3.get(),
        debugHostCombo);

    // MPIR reference products
    mpf_t mpfXX, mpf2XY, mpfYY, mpfW0, mpfW1, mpfW2, mpfW3;
    mpf_init(mpfXX); mpf_init(mpf2XY); mpf_init(mpfYY);
    mpf_init(mpfW0); mpf_init(mpfW1); mpf_init(mpfW2); mpf_init(mpfW3);

    mpf_mul(mpfXX, mpfA, mpfA);
    mpf_mul(mpf2XY, mpfA, mpfB);
    mpf_mul_ui(mpf2XY, mpf2XY, 2);
    mpf_mul(mpfYY, mpfB, mpfB);
    mpf_mul(mpfW0, mpfDzdcR, mpfA);
    mpf_mul_ui(mpfW0, mpfW0, 2);
    mpf_mul(mpfW1, mpfDzdcI, mpfB);
    mpf_mul_ui(mpfW1, mpfW1, 2);
    mpf_mul(mpfW2, mpfDzdcR, mpfB);
    mpf_mul_ui(mpfW2, mpfW2, 2);
    mpf_mul(mpfW3, mpfDzdcI, mpfA);
    mpf_mul_ui(mpfW3, mpfW3, 2);

    // Compare each output against MPIR
    checkProduct(*outXX, mpfXX, testBase + 0, "XX");
    checkProduct(*outXY, mpf2XY, testBase + 1, "2XY");
    checkProduct(*outYY, mpfYY, testBase + 2, "YY");
    checkProduct(*outW0, mpfW0, testBase + 3, "W0");
    checkProduct(*outW1, mpfW1, testBase + 4, "W1");
    checkProduct(*outW2, mpfW2, testBase + 5, "W2");
    checkProduct(*outW3, mpfW3, testBase + 6, "W3");

    // Cleanup
    mpf_clear(mpfA); mpf_clear(mpfB);
    mpf_clear(mpfDzdcR); mpf_clear(mpfDzdcI);
    mpf_clear(mpfXX); mpf_clear(mpf2XY); mpf_clear(mpfYY);
    mpf_clear(mpfW0); mpf_clear(mpfW1); mpf_clear(mpfW2); mpf_clear(mpfW3);
    mpf_clear(gpuVal); mpf_clear(diff);
    mpf_clear(absDiff); mpf_clear(absRef); mpf_clear(relError);
    mpf_clear(tolerance);

    return Tests.CheckAllTestsPassed();
}

// ========== Single-iteration NR Add test ==========
template <class SharkFloatParams>
bool
TestSingleNRAdd(TestTracker &Tests, int testBase)
{
    const int precBits = HpSharkFloat<SharkFloatParams>::DefaultPrecBits;
    mpf_set_default_prec(HpSharkFloat<SharkFloatParams>::DefaultMpirBits);

    const int toleranceBits = precBits - 10;
    mpf_t tolerance;
    mpf_init(tolerance);
    mpf_set_ui(tolerance, 1);
    mpf_div_2exp(tolerance, tolerance, toleranceBits);

    mpf_t gpuVal, diff, absDiff, absRef, relError;
    mpf_init(gpuVal); mpf_init(diff);
    mpf_init(absDiff); mpf_init(absRef); mpf_init(relError);

    auto checkResult = [&](HpSharkFloat<SharkFloatParams> &gpuOutput,
                           mpf_t mpirRef,
                           int testIndex,
                           const char *name) -> bool {
        gpuOutput.HpGpuToMpf(gpuVal);
        mpf_sub(diff, gpuVal, mpirRef);
        mpf_abs(absDiff, diff);
        mpf_abs(absRef, mpirRef);

        bool pass;
        if (mpf_sgn(absRef) == 0) {
            pass = mpf_cmp(absDiff, tolerance) <= 0;
        } else {
            mpf_div(relError, absDiff, absRef);
            pass = mpf_cmp(relError, tolerance) <= 0;
        }

        std::string testName = std::string("NRAdd_") + name;
        char buf[256];
        gmp_snprintf(buf, sizeof(buf), "diff=%+.6Fe", diff);
        char tolStr[256];
        gmp_snprintf(tolStr, sizeof(tolStr), "2^-%d", toleranceBits);

        if (pass) {
            Tests.MarkSuccess(nullptr, testIndex, testName);
        } else {
            Tests.MarkFailed(nullptr, testIndex, testName, buf, tolStr);
        }
        return pass;
    };

    // Generate 9 random inputs
    auto hpAX2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpBY2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpCA = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpD2X = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpEB = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpW0 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpW1 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpW2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto hpW3 = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    hpAX2->GenerateRandomNumber2();
    hpBY2->GenerateRandomNumber2();
    hpCA->GenerateRandomNumber2();
    hpD2X->GenerateRandomNumber2();
    hpEB->GenerateRandomNumber2();
    hpW0->GenerateRandomNumber2();
    hpW1->GenerateRandomNumber2();
    hpW2->GenerateRandomNumber2();
    hpW3->GenerateRandomNumber2();

    // Get MPIR equivalents
    mpf_t mpfAX2, mpfBY2, mpfCA, mpfD2X, mpfEB;
    mpf_t mpfW0, mpfW1, mpfW2, mpfW3;
    mpf_init(mpfAX2); mpf_init(mpfBY2); mpf_init(mpfCA);
    mpf_init(mpfD2X); mpf_init(mpfEB);
    mpf_init(mpfW0); mpf_init(mpfW1); mpf_init(mpfW2); mpf_init(mpfW3);

    hpAX2->HpGpuToMpf(mpfAX2);
    hpBY2->HpGpuToMpf(mpfBY2);
    hpCA->HpGpuToMpf(mpfCA);
    hpD2X->HpGpuToMpf(mpfD2X);
    hpEB->HpGpuToMpf(mpfEB);
    hpW0->HpGpuToMpf(mpfW0);
    hpW1->HpGpuToMpf(mpfW1);
    hpW2->HpGpuToMpf(mpfW2);
    hpW3->HpGpuToMpf(mpfW3);

    // Output HpSharkFloats
    auto outXY1 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outXY2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outDzdcR = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto outDzdcI = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    DebugHostCombo<SharkFloatParams> debugHostCombo;

    AddHelper<SharkFloatParams>(
        hpAX2.get(), hpBY2.get(), hpCA.get(),
        hpD2X.get(), hpEB.get(),
        outXY1.get(), outXY2.get(),
        hpW0.get(), hpW1.get(), hpW2.get(), hpW3.get(),
        outDzdcR.get(), outDzdcI.get(),
        debugHostCombo);

    // MPIR reference
    mpf_t mpfRefOut1, mpfRefOut2, mpfRefDzdcR, mpfRefDzdcI;
    mpf_init(mpfRefOut1); mpf_init(mpfRefOut2);
    mpf_init(mpfRefDzdcR); mpf_init(mpfRefDzdcI);

    // Out1 = A_X2 - B_Y2 + C_A
    mpf_sub(mpfRefOut1, mpfAX2, mpfBY2);
    mpf_add(mpfRefOut1, mpfRefOut1, mpfCA);

    // Out2 = D_2X + E_B
    mpf_add(mpfRefOut2, mpfD2X, mpfEB);

    // DzdcReal = W0 - W1 + 1.0
    mpf_sub(mpfRefDzdcR, mpfW0, mpfW1);
    mpf_add_ui(mpfRefDzdcR, mpfRefDzdcR, 1);

    // DzdcImag = W2 + W3
    mpf_add(mpfRefDzdcI, mpfW2, mpfW3);

    // Compare each output against MPIR
    checkResult(*outXY1, mpfRefOut1, testBase + 0, "Out1");
    checkResult(*outXY2, mpfRefOut2, testBase + 1, "Out2");
    checkResult(*outDzdcR, mpfRefDzdcR, testBase + 2, "DzdcReal");
    checkResult(*outDzdcI, mpfRefDzdcI, testBase + 3, "DzdcImag");

    // Cleanup
    mpf_clear(mpfAX2); mpf_clear(mpfBY2); mpf_clear(mpfCA);
    mpf_clear(mpfD2X); mpf_clear(mpfEB);
    mpf_clear(mpfW0); mpf_clear(mpfW1); mpf_clear(mpfW2); mpf_clear(mpfW3);
    mpf_clear(mpfRefOut1); mpf_clear(mpfRefOut2);
    mpf_clear(mpfRefDzdcR); mpf_clear(mpfRefDzdcI);
    mpf_clear(gpuVal); mpf_clear(diff);
    mpf_clear(absDiff); mpf_clear(absRef); mpf_clear(relError);
    mpf_clear(tolerance);

    return Tests.CheckAllTestsPassed();
}

template bool TestSingleNRMultiply<SharkParamsNR7>(TestTracker &, int);
template bool TestSingleNRAdd<SharkParamsNR7>(TestTracker &, int);
