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

    if constexpr (h == Fractal::AutoZoomHeuristic::FilamentTip) {
        Divisor = HighPrecision{8};
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
        // Default/Max/FilamentTip heuristic: serial loop that analyzes CPU-side
        // iteration data after each render.  Each step uses EnqueueCommand +
        // Wait() so the frame is displayed by the GL consumer.  The
        // workerIters ↔ m_CurIters swap in WorkerLoop keeps m_CurIters
        // current for analysis.

        for (;;) {
            {
                MSG msg;
                PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
            }

            double geometricMeanX = 0;
            double geometricMeanSum = 0;
            double geometricMeanY = 0;

            width = m_Fractal.GetMaxX() - m_Fractal.GetMinX();
            height = m_Fractal.GetMaxY() - m_Fractal.GetMinY();

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
                        std::wcerr << L"Flat screen (geometricMeanSum=0)! :(" << std::endl;
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

                // ---------------- FILAMENT TIP ----------------
                if constexpr (h == Fractal::AutoZoomHeuristic::FilamentTip) {

                    const auto scrnW = static_cast<int>(m_Fractal.GetScrnWidth() * m_Fractal.GetGpuAntialiasing());
                    const auto scrnH = static_cast<int>(m_Fractal.GetScrnHeight() * m_Fractal.GetGpuAntialiasing());

                    // Pass 1: compute average iteration count
                    double totalIters = 0;
                    size_t pixelCount = 0;
                    for (int y = 0; y < scrnH; y++) {
                        for (int x = 0; x < scrnW; x++) {
                            totalIters += ItersArray[y][x];
                            pixelCount++;
                        }
                    }

                    if (pixelCount == 0 || totalIters == 0) {
                        shouldBreak = true;
                        return;
                    }

                    const double avgIters = totalIters / pixelCount;

                    // Candidates: any pixel above average, even barely.
                    // Faint tips may be just slightly above background.
                    const auto candidateThreshold =
                        static_cast<size_t>(avgIters + 1);

                    // Sample at multiple radii to handle fuzzy boundaries.
                    // A tip has lower iters in most directions at most radii.
                    static constexpr int dx8[] = { 0,  1, 1, 1, 0, -1, -1, -1};
                    static constexpr int dy8[] = {-1, -1, 0, 1, 1,  1,  0, -1};
                    static constexpr int radii[] = {4, 8, 16};
                    static constexpr int numRadii = 3;

                    const int margin = 18;

                    double bestScore = -1.0;
                    int bestX = scrnW / 2;
                    int bestY = scrnH / 2;

                    size_t dbgCandidates = 0;
                    size_t dbgHighCountHist[9] = {};
                    size_t dbgRunReject = 0;

                    for (int y = margin; y < scrnH - margin; y++) {
                        for (int x = margin; x < scrnW - margin; x++) {
                            auto curiter = ItersArray[y][x];

                            if (curiter < candidateThreshold)
                                continue;

                            dbgCandidates++;

                            if (curiter >= NumIterations)
                                numAtMax++;

                            // Sample 8 directions at the largest radius.
                            // Classify each direction as "high" (close to
                            // center iter count) or "low" (below center).
                            // Use center value minus 1 as the threshold:
                            // a neighbor is "high" only if it's at least
                            // as high as the center pixel (allowing for
                            // the fuzzy, gradual iteration landscape).
                            static constexpr int SampleR = 12;
                            bool isHigh[8];
                            int highCount = 0;
                            const auto highThreshold = (curiter > 0) ? curiter - 1 : curiter;
                            for (int d = 0; d < 8; d++) {
                                int sx = x + dx8[d] * SampleR;
                                int sy = y + dy8[d] * SampleR;
                                if (sx < 0 || sx >= scrnW ||
                                    sy < 0 || sy >= scrnH) {
                                    isHigh[d] = false;
                                    continue;
                                }
                                // "High" if neighbor is within 1 iteration
                                // of center.  This is very strict — only
                                // truly flat/equal neighbors count.
                                isHigh[d] = (ItersArray[sy][sx] >= highThreshold);
                                if (isHigh[d]) highCount++;
                            }

                            // A TIP has exactly 1-2 contiguous high
                            // directions (the filament body behind it)
                            // and the rest low.
                            //
                            // A BODY has 2 high directions on opposite
                            // sides (along the filament axis) and low
                            // on perpendicular sides.
                            //
                            // An INTERIOR has 5+ high directions.
                            //
                            // Count the longest contiguous run of "high"
                            // directions (wrapping around the 8-ring).
                            int maxRun = 0;
                            int curRun = 0;
                            for (int i = 0; i < 16; i++) {
                                if (isHigh[i % 8]) {
                                    curRun++;
                                    if (curRun > maxRun) maxRun = curRun;
                                } else {
                                    curRun = 0;
                                }
                            }

                            dbgHighCountHist[highCount]++;

                            // Tip criterion:
                            // - highCount 0: perfectly isolated peak (best tip)
                            // - highCount 1-2: tip with filament behind it
                            // - highCount 3: borderline, accept if contiguous
                            // - highCount 4+: body or interior, reject
                            if (highCount > 3)
                                continue;
                            // For highCount > 0, require contiguous run
                            // (rejects bodies with opposing high groups)
                            if (highCount > 0 && maxRun < highCount) {
                                dbgRunReject++;
                                continue;
                            }

                            // Tipness: fewer high neighbors = more tip-like
                            // highCount 0 → tipness 1.0 (isolated peak)
                            // highCount 3 → tipness 0.25
                            double tipness = 1.0 - static_cast<double>(highCount) / 4.0;

                            // Favor "boring" tips: low iteration count
                            // relative to screen max.  Mini-brots have high
                            // iters; plain filament tips have iters barely
                            // above background.  Invert the elevation so
                            // lower-iteration tips score higher.
                            double numerator = static_cast<double>(curiter) - avgIters;
                            double denominator = static_cast<double>(NumIterations) - avgIters;
                            double rawElevation = (denominator > 0)
                                ? log(1.0 + numerator) / log(1.0 + denominator)
                                : 0.5;
                            double elevation = 1.0 - rawElevation;

                            // Prefer tips away from center (exploring outward)
                            double ddx = static_cast<double>(x - scrnW / 2);
                            double ddy = static_cast<double>(y - scrnH / 2);
                            double maxDist = sqrt(
                                static_cast<double>(scrnW * scrnW + scrnH * scrnH)) / 2.0;
                            double distFromCenter = sqrt(ddx * ddx + ddy * ddy) / maxDist;

                            double score = tipness * elevation *
                                           (0.3 + 0.7 * distFromCenter);

                            if (score > bestScore) {
                                bestScore = score;
                                bestX = x;
                                bestY = y;
                            }
                        }
                    }

                    if (bestScore < 0) {
                        std::wcerr << L"FilamentTip: no tip found. "
                                   << L"candidates=" << dbgCandidates
                                   << L" highHist=[";
                        for (int i = 0; i <= 8; i++)
                            std::wcerr << dbgHighCountHist[i] << (i < 8 ? L"," : L"");
                        std::wcerr << L"] runReject=" << dbgRunReject
                                   << L" avgIters=" << avgIters
                                   << L" threshold=" << candidateThreshold
                                   << std::endl;
                        shouldBreak = true;
                        return;
                    }

                    std::wcerr << L"FilamentTip: target (" << bestX << L"," << bestY
                               << L") score=" << bestScore
                               << L" numAtMax=" << numAtMax << std::endl;

                    guessX = m_Fractal.XFromScreenToCalc<true>(HighPrecision{bestX});
                    guessY = m_Fractal.YFromScreenToCalc<true>(HighPrecision{bestY});

                    if (numAtMax > static_cast<size_t>(scrnW) * scrnH / 2) {
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

            m_Fractal.EnqueueCommand([newPtz = std::move(newPtz)](Fractal &f) {
                f.RecenterViewCalc(newPtz);
            }, false).Wait();

            if constexpr (h != Fractal::AutoZoomHeuristic::FilamentTip) {
                if (numAtMax > 500)
                    break;
            }

            if (m_Fractal.GetStopCalculating())
                break;
        }
    }
}

template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Default>();
template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Max>();
template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::Feature>();
template void AutoZoomer::Run<Fractal::AutoZoomHeuristic::FilamentTip>();

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
