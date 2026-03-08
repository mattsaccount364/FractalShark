#include "stdafx.h"

#include "AutoZoomer.h"
#include "FeatureSummary.h"
#include "PerturbationResults.h"

#include <cmath>
#include <iostream>

AutoZoomer::AutoZoomer(Fractal &fractal)
    : m_Fractal(fractal) {
}

template <Fractal::AutoZoomHeuristic h>
void
AutoZoomer::Run()
{
    const HighPrecision Two = HighPrecision{2};

    HighPrecision width = m_Fractal.GetMaxX() - m_Fractal.GetMinX();
    HighPrecision height = m_Fractal.GetMaxY() - m_Fractal.GetMinY();

    HighPrecision guessX;
    HighPrecision guessY;

    HighPrecision Divisor;

    if constexpr (h == Fractal::AutoZoomHeuristic::Default) {
        Divisor = HighPrecision{3};
    }

    if constexpr (h == Fractal::AutoZoomHeuristic::Max) {
        Divisor = HighPrecision{32};
    }

    if constexpr (h == Fractal::AutoZoomHeuristic::Feature) {
        std::cout << "Forcing GPU HDRx32 Perturbed LAv2 for AutoZoom(Feature) since it relies on "
                     "perturbation results to perform the zoom.\n";

        FeatureZoomSetup setup;

        m_Fractal.EnqueueMutation([&](Fractal &f) {
            SetupFeatureZoom(f, setup);
        }).Wait();

        if (setup.Failed) {
            return;
        }

        guessX = std::move(setup.GuessX);
        guessY = std::move(setup.GuessY);

        Divisor = HighPrecision{3};

        static constexpr size_t PipelineDepth = 4 * NumRenderers;
        std::vector<RenderJobHandle> handles(PipelineDepth);

        for (int64_t i = 0; i < setup.TotalSteps; ++i) {
            {
                MSG msg;
                PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
            }

            if (m_Fractal.GetStopCalculating())
                break;

            // Wait for the oldest in-flight item before enqueueing
            size_t slot = i % PipelineDepth;
            handles[slot].Wait();

            if (setup.ShouldInterpolateIters) {
                auto iters = setup.IterCounts[i];
                auto ptz = setup.ZoomSteps[i];
                handles[slot] = m_Fractal.EnqueueCommand(
                    [ptz = std::move(ptz), iters](Fractal &f) {
                        f.m_Ptz = ptz;
                        f.SquareCurrentView();
                        f.SetPrecision();
                        f.SetNumIterations<IterTypeFull>(iters);
                    },
                    false);
            } else {
                auto ptz = setup.ZoomSteps[i];
                handles[slot] = m_Fractal.EnqueueCommand(
                    [ptz = std::move(ptz)](Fractal &f) {
                        f.m_Ptz = ptz;
                        f.SquareCurrentView();
                        f.SetPrecision();
                    },
                    false);
            }
        }

        // Drain all remaining in-flight renders
        for (auto &h : handles) {
            h.Wait();
        }

        // Update live state to final zoom position so the view doesn't
        // snap back to the original zoom after the loop completes.
        if (!setup.ZoomSteps.empty()) {
            auto targetIters = setup.ShouldInterpolateIters
                ? setup.IterCounts.back()
                : m_Fractal.GetNumIterations<IterTypeFull>();
            m_Fractal.EnqueueCommand(
                [ptz = setup.ZoomSteps.back(), targetIters](Fractal &f) {
                    f.m_Ptz = ptz;
                    f.SquareCurrentView();
                    f.SetPrecision();
                    f.SetNumIterations<IterTypeFull>(targetIters);
                }).Wait();
        }

        return;
    } else {
        // Default/Max heuristic: serial loop that analyzes CPU-side iteration
        // data after each render. Must use CalcFractal(true) which writes to
        // m_CurIters and does GPU→CPU iter copy, NOT EnqueueCommand which
        // renders to a worker-acquired container that m_CurIters never sees.
        if (auto *pool = m_Fractal.GetRenderPool()) {
            pool->Drain();
        }

        size_t retries = 0;

        for (;;) {
            {
                MSG msg;
                PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
            }

            double geometricMeanX = 0;
            double geometricMeanSum = 0;
            double geometricMeanY = 0;

            if (retries >= 0) {
                width = m_Fractal.GetMaxX() - m_Fractal.GetMinX();
                height = m_Fractal.GetMaxY() - m_Fractal.GetMinY();
                retries = 0;
            }

            size_t numAtMax = 0;
            size_t numAtLimit = 0;
            bool shouldBreak = false;

            auto lambda = [&]([[maybe_unused]] auto **ItersArray, [[maybe_unused]] auto NumIterations) {
                // ---------------- DEFAULT ----------------
                if constexpr (h == Fractal::AutoZoomHeuristic::Default) {

                    ULONG shiftWidth = (ULONG)m_Fractal.GetScrnWidth() / 8;
                    ULONG shiftHeight = (ULONG)m_Fractal.GetScrnHeight() / 8;

                    RECT antiRect;
                    antiRect.left = shiftWidth;
                    antiRect.right = (ULONG)m_Fractal.GetScrnWidth() - shiftWidth;
                    antiRect.top = shiftHeight;
                    antiRect.bottom = (ULONG)m_Fractal.GetScrnHeight() - shiftHeight;

                    antiRect.left *= m_Fractal.GetGpuAntialiasing();
                    antiRect.right *= m_Fractal.GetGpuAntialiasing();
                    antiRect.top *= m_Fractal.GetGpuAntialiasing();
                    antiRect.bottom *= m_Fractal.GetGpuAntialiasing();

                    const auto antiRectWidthInt = antiRect.right - antiRect.left;
                    const auto antiRectHeightInt = antiRect.bottom - antiRect.top;

                    size_t maxiter = 0;
                    double totaliters = 0;

                    for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                        for (auto x = antiRect.left; x < antiRect.right; x++) {
                            auto curiter = ItersArray[y][x];
                            totaliters += curiter;
                            if (curiter > maxiter)
                                maxiter = curiter;
                        }
                    }

                    double avgiters = totaliters / ((antiRect.bottom - antiRect.top) *
                                                    (antiRect.right - antiRect.left));

                    double widthOver2 = antiRectWidthInt / 2.0;
                    double heightOver2 = antiRectHeightInt / 2.0;
                    double maxDistance = sqrt(widthOver2 * widthOver2 + heightOver2 * heightOver2);

                    for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                        for (auto x = antiRect.left; x < antiRect.right; x++) {

                            auto curiter = ItersArray[y][x];

                            if (curiter == maxiter)
                                numAtLimit++;

                            if (curiter < avgiters)
                                continue;

                            double distanceX =
                                fabs(widthOver2 - fabs(widthOver2 - fabs(x - antiRect.left)));
                            double distanceY =
                                fabs(heightOver2 - fabs(heightOver2 - fabs(y - antiRect.top)));

                            double normalizedIters = (double)curiter / (double)NumIterations;

                            if (curiter == maxiter)
                                normalizedIters *= normalizedIters;

                            double normalizedDist =
                                sqrt(distanceX * distanceX + distanceY * distanceY) / maxDistance;

                            double sq = normalizedIters * normalizedDist;

                            geometricMeanSum += sq;
                            geometricMeanX += sq * x;
                            geometricMeanY += sq * y;

                            if (curiter >= NumIterations)
                                numAtMax++;
                        }
                    }

                    if (geometricMeanSum == 0) {
                        shouldBreak = true;
                        return;
                    }

                    double meanX = geometricMeanX / geometricMeanSum;
                    double meanY = geometricMeanY / geometricMeanSum;

                    guessX = m_Fractal.XFromScreenToCalc<true>(HighPrecision{meanX});
                    guessY = m_Fractal.YFromScreenToCalc<true>(HighPrecision{meanY});

                    if (numAtLimit == antiRectWidthInt * antiRectHeightInt) {
                        std::wcerr << L"Flat screen! :(" << std::endl;
                        shouldBreak = true;
                        return;
                    }
                }

                // ---------------- MAX ----------------
                if constexpr (h == Fractal::AutoZoomHeuristic::Max) {

                    LONG targetX = -1;
                    LONG targetY = -1;
                    size_t maxiter = 0;

                    for (auto y = 0; y < m_Fractal.GetScrnHeight() * m_Fractal.GetGpuAntialiasing(); y++) {
                        for (auto x = 0; x < m_Fractal.GetScrnWidth() * m_Fractal.GetGpuAntialiasing(); x++) {
                            auto curiter = ItersArray[y][x];
                            if (curiter > maxiter)
                                maxiter = curiter;
                        }
                    }

                    for (auto y = 0; y < m_Fractal.GetScrnHeight() * m_Fractal.GetGpuAntialiasing(); y++) {
                        for (auto x = 0; x < m_Fractal.GetScrnWidth() * m_Fractal.GetGpuAntialiasing(); x++) {
                            auto curiter = ItersArray[y][x];

                            if (curiter == maxiter) {
                                numAtLimit++;
                                if (targetX == -1 && targetY == -1) {
                                    targetX = x;
                                    targetY = y;
                                }
                            }

                            if (curiter >= NumIterations)
                                numAtMax++;
                        }
                    }

                    guessX = m_Fractal.XFromScreenToCalc<true>(HighPrecision{targetX});
                    guessY = m_Fractal.YFromScreenToCalc<true>(HighPrecision{targetY});

                    if (numAtLimit ==
                        m_Fractal.GetScrnWidth() * m_Fractal.GetScrnHeight() * m_Fractal.GetGpuAntialiasing() * m_Fractal.GetGpuAntialiasing()) {
                        std::wcerr << L"Flat screen! :(" << std::endl;
                        shouldBreak = true;
                        return;
                    }
                }
            };

            if (m_Fractal.GetIterType() == IterTypeEnum::Bits32) {
                lambda(m_Fractal.m_CurIters.GetItersArray<uint32_t>(), m_Fractal.GetNumIterations<uint32_t>());
            } else {
                lambda(m_Fractal.m_CurIters.GetItersArray<uint64_t>(), m_Fractal.GetNumIterations<uint64_t>());
            }

            if (shouldBreak)
                break;

            HighPrecision newMinX = guessX - width / Divisor;
            HighPrecision newMinY = guessY - height / Divisor;
            HighPrecision newMaxX = guessX + width / Divisor;
            HighPrecision newMaxY = guessY + height / Divisor;

            PointZoomBBConverter newPtz{newMinX, newMinY, newMaxX, newMaxY, PointZoomBBConverter::TestMode::Enabled};

            m_Fractal.RecenterViewCalc(newPtz);
            m_Fractal.ForceRecalc();
            m_Fractal.CalcFractal(true);

            if (numAtMax > 500)
                break;

            retries++;

            if (m_Fractal.GetStopCalculating())
                break;
        }
    }
}

template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Default>();
template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Max>();
template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Feature>();

void
AutoZoomer::SetupFeatureZoom(Fractal &f, FeatureZoomSetup &out)
{
    [[maybe_unused]] const bool success = f.SetRenderAlgorithm(
        GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2));
    if (!success) {
        std::cerr << "Error: could not set render algorithm for AutoZoom(Feature).\n";
        out.Failed = true;
        return;
    }

    FeatureSummary *feature = f.ChooseClosestFeatureToMouse();
    if (!feature) {
        std::cout << "AutoZoom(Feature): no feature found. Use the feature finder first.\n";
        out.Failed = true;
        return;
    }

    if (!f.ZoomToFoundFeature(*feature, nullptr)) {
        std::cout << "AutoZoom(Feature): feature refinement failed.\n";
        out.Failed = true;
        return;
    }

    out.GuessX = feature->GetFoundX();
    out.GuessY = feature->GetFoundY();

    HighPrecision targetZoomFactor = feature->ComputeZoomFactor(f.GetPtz());

    std::cout << "AutoZoom(Feature): targetZoomFactor=";
    {
        double m; long e;
        targetZoomFactor.frexp(m, e);
        std::cout << m << " * 2^" << e;
    }
    std::cout << " origZoom=";
    {
        double m; long e;
        f.GetPtz().GetZoomFactor().frexp(m, e);
        std::cout << m << " * 2^" << e;
    }
    std::cout << " intrinsicRadius=";
    {
        double m; long e;
        feature->GetIntrinsicRadius().frexp(m, e);
        std::cout << m << " * 2^" << e;
    }
    std::cout << "\n";

    if (targetZoomFactor <= HighPrecision{0}) {
        std::cout << "AutoZoom(Feature): invalid target zoom factor.\n";
        out.Failed = true;
        return;
    }

    HighPrecision origZoom = f.GetPtz().GetZoomFactor();
    const IterTypeFull startIters = f.GetNumIterations<IterTypeFull>();
    const IterTypeFull targetIters = feature->GetNumIterationsAtFind();

    // Pre-compute reference orbit at the TARGET zoom depth.
    // Clear existing results and render at the destination so the ref orbit
    // has a tight enough radius for the entire zoom animation.
    {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

        PointZoomBBConverter targetPtz{
            out.GuessX, out.GuessY, targetZoomFactor, PointZoomBBConverter::TestMode::Enabled};
        f.m_Ptz = targetPtz;
        f.SquareCurrentView();
        f.SetPrecision();
        if (targetIters > 0) {
            f.SetNumIterations<IterTypeFull>(targetIters);
        }
        f.ForceRecalc();

        std::cout << "AutoZoom(Feature): pre-computing ref orbit at target zoom...\n";
        f.CalcFractal(false);
        std::cout << "AutoZoom(Feature): ref orbit ready.\n";

        // Restore starting iteration count
        f.SetNumIterations<IterTypeFull>(startIters);
    }

    // Set starting position for zoom animation
    PointZoomBBConverter startPtzCentered{
        out.GuessX, out.GuessY, origZoom, PointZoomBBConverter::TestMode::Enabled};
    // Direct assignment + SquareCurrentView instead of RecenterViewCalc.
    // RecenterViewCalc calls SetPrecision() which resets MPIR precision to
    // match the current (low) zoom depth, truncating the feature's
    // high-precision coordinates.
    f.m_Ptz = startPtzCentered;
    f.SquareCurrentView();

    const HighPrecision zoomRatio = targetZoomFactor / origZoom;
    double mantissa{};
    long exp2{};
    zoomRatio.frexp(mantissa, exp2);
    const double logRatio = std::log(std::abs(mantissa)) + exp2 * std::log(2.0);
    const double logZoomPerStep = std::log(1.1);
    out.TotalSteps =
        (zoomRatio > HighPrecision{1})
            ? static_cast<int64_t>(std::ceil(logRatio / logZoomPerStep))
            : 1;

    out.ShouldInterpolateIters = targetIters > 0 && startIters < targetIters;

    std::cout << "AutoZoom(Feature): startIters=" << startIters
              << " targetIters=" << targetIters
              << " period=" << feature->GetPeriod()
              << " totalSteps=" << out.TotalSteps
              << " interpolate=" << out.ShouldInterpolateIters << "\n";

    out.ZoomSteps.reserve(out.TotalSteps);
    {
        PointZoomBBConverter stepPtz = f.GetPtz();
        for (int64_t i = 0; i < out.TotalSteps; ++i) {
            stepPtz.ZoomInPlace(-1.0 / 22.0);
            out.ZoomSteps.push_back(stepPtz);
        }
    }

    if (out.ShouldInterpolateIters) {
        out.IterCounts.reserve(out.TotalSteps);
        for (int64_t i = 0; i < out.TotalSteps; ++i) {
            out.IterCounts.push_back(
                startIters + (targetIters - startIters) * (i + 1) / out.TotalSteps);
        }
    }

    for (size_t r = 0; r < NumRenderers; ++r) {
        auto err = f.InitializeGPUMemory(static_cast<RendererIndex>(r), false, f.m_CurIters);
        if (err) {
            f.MessageBoxCudaError(err);
            out.Failed = true;
            return;
        }
    }
}
