#include "stdafx.h"

#include "AutoZoomer.h"
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

    HighPrecision width = m_Fractal.m_Ptz.GetMaxX() - m_Fractal.m_Ptz.GetMinX();
    HighPrecision height = m_Fractal.m_Ptz.GetMaxY() - m_Fractal.m_Ptz.GetMinY();

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
        const bool success = m_Fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2));
        if (!success) {
            std::cerr << "Error: could not set render algorithm for AutoZoom(Feature).\n";
            return;
        }

        Divisor = HighPrecision{3};

        // Get existing perturbation results to extract reference point
        using IterType = uint32_t;
        using T = HDRFloat<float>;
        using SubType = float;
        constexpr PerturbExtras PExtras = PerturbExtras::Disable;

        const auto *results =
            m_Fractal.m_RefOrbit.GetUsefulPerturbationResults<IterType, T, SubType, PExtras,
                                                    RefOrbitCalc::Extras::None>();
        if (!results) {
            std::cout << "AutoZoom(Feature): no perturbation results available.\n";
            return;
        }

        // Zoom target is the perturbation reference point
        guessX = results->GetHiX();
        guessY = results->GetHiY();

        // Compute target zoom factor from GetMaxRadius().
        // maxRadius represents the convergence radius of the reference orbit.
        const T maxRadius = results->GetMaxRadius();

        if (maxRadius.getMantissa() == 0.0) {
            std::cout << "AutoZoom(Feature): maxRadius is zero, cannot compute zoom depth.\n";
            return;
        }

        // Convert HDRFloat maxRadius to HighPrecision
        HighPrecision maxRadiusHP;
        maxRadius.GetHighPrecision(maxRadiusHP);

        // targetZoomFactor = PointZoomBBConverter::Factor / maxRadius
        // This sets the view so its half-width equals maxRadius.
        HighPrecision targetZoomFactor = HighPrecision{PointZoomBBConverter::Factor} / maxRadiusHP;

        HighPrecision origZoom = m_Fractal.m_Ptz.GetZoomFactor();

        // Center on the perturbation point at the original zoom
        PointZoomBBConverter startPtzCentered{guessX, guessY, origZoom, PointZoomBBConverter::TestMode::Enabled};
        m_Fractal.m_Ptz = startPtzCentered;
        m_Fractal.SquareCurrentView();

        //CalcFractal(true);

        // Compute total animation steps.
        // Each ZoomInPlace(-1/22) step multiplies the zoom factor by 1.1.
        // Decompose the ratio via frexp to avoid double overflow for
        // extreme zoom depths.
        const HighPrecision zoomRatio = targetZoomFactor / origZoom;
        double mantissa{};
        long exp2{};
        zoomRatio.frexp(mantissa, exp2);
        const double logRatio = std::log(std::abs(mantissa)) + exp2 * std::log(2.0);
        const double logZoomPerStep = std::log(1.1);
        const int64_t totalSteps =
            (zoomRatio > HighPrecision{1})
                ? static_cast<int64_t>(std::ceil(logRatio / logZoomPerStep))
                : 1;

        // Linearly interpolate iterations from current to perturbation target
        const IterTypeFull startIters = m_Fractal.GetNumIterations<IterTypeFull>();
        const IterTypeFull targetIters = results->GetMaxIterations();
        const bool shouldInterpolateIters = startIters < targetIters;

        // Pre-compute all zoom viewports
        std::vector<PointZoomBBConverter> zoomSteps;
        zoomSteps.reserve(totalSteps);
        {
            PointZoomBBConverter stepPtz = m_Fractal.m_Ptz;
            for (int64_t i = 0; i < totalSteps; ++i) {
                stepPtz.ZoomInPlace(-1.0 / 22.0);
                zoomSteps.push_back(stepPtz);
            }
        }

        // Pre-compute iteration counts
        std::vector<IterTypeFull> iterCounts;
        if (shouldInterpolateIters) {
            iterCounts.reserve(totalSteps);
            for (int64_t i = 0; i < totalSteps; ++i) {
                iterCounts.push_back(
                    startIters + (targetIters - startIters) * (i + 1) / totalSteps);
            }
        }

        // Pre-initialize all renderers' GPU memory before the zoom loop
        // so that managed memory allocations don't conflict with in-flight kernels.
        for (size_t r = 0; r < NumRenderers; ++r) {
            auto err = m_Fractal.InitializeGPUMemory(static_cast<RendererIndex>(r), false, m_Fractal.m_CurIters);
            if (err) {
                m_Fractal.MessageBoxCudaError(err);
                return;
            }
        }

        static constexpr size_t PipelineDepth = 4 * NumRenderers;
        std::vector<RenderJobHandle> handles(PipelineDepth);

        for (int64_t i = 0; i < totalSteps; ++i) {
            {
                MSG msg;
                PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
            }

            if (m_Fractal.GetStopCalculating())
                break;

            // Wait for the oldest in-flight item before enqueueing
            size_t slot = i % PipelineDepth;
            handles[slot].Wait();

            if (shouldInterpolateIters) {
                m_Fractal.SetNumIterations<IterTypeFull>(iterCounts[i]);
            }

            // Pass coordinates directly — avoids racing on m_Ptz
            handles[slot] = m_Fractal.EnqueueRender(zoomSteps[i]);
        }

        // Drain all remaining in-flight renders
        for (auto &h : handles) {
            h.Wait();
        }

        // Update live state to final zoom position so the view doesn't
        // snap back to the original zoom after the loop completes.
        if (!zoomSteps.empty()) {
            m_Fractal.m_Ptz = zoomSteps.back();
        }

        return;
    } else {
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
                width = m_Fractal.m_Ptz.GetMaxX() - m_Fractal.m_Ptz.GetMinX();
                height = m_Fractal.m_Ptz.GetMaxY() - m_Fractal.m_Ptz.GetMinY();
                retries = 0;
            }

            size_t numAtMax = 0;
            size_t numAtLimit = 0;
            bool shouldBreak = false;

            auto lambda = [&]([[maybe_unused]] auto **ItersArray, [[maybe_unused]] auto NumIterations) {
                // ---------------- DEFAULT ----------------
                if constexpr (h == Fractal::AutoZoomHeuristic::Default) {

                    ULONG shiftWidth = (ULONG)m_Fractal.m_ScrnWidth / 8;
                    ULONG shiftHeight = (ULONG)m_Fractal.m_ScrnHeight / 8;

                    RECT antiRect;
                    antiRect.left = shiftWidth;
                    antiRect.right = (ULONG)m_Fractal.m_ScrnWidth - shiftWidth;
                    antiRect.top = shiftHeight;
                    antiRect.bottom = (ULONG)m_Fractal.m_ScrnHeight - shiftHeight;

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

                    for (auto y = 0; y < m_Fractal.m_ScrnHeight * m_Fractal.GetGpuAntialiasing(); y++) {
                        for (auto x = 0; x < m_Fractal.m_ScrnWidth * m_Fractal.GetGpuAntialiasing(); x++) {
                            auto curiter = ItersArray[y][x];
                            if (curiter > maxiter)
                                maxiter = curiter;
                        }
                    }

                    for (auto y = 0; y < m_Fractal.m_ScrnHeight * m_Fractal.GetGpuAntialiasing(); y++) {
                        for (auto x = 0; x < m_Fractal.m_ScrnWidth * m_Fractal.GetGpuAntialiasing(); x++) {
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
                        m_Fractal.m_ScrnWidth * m_Fractal.m_ScrnHeight * m_Fractal.GetGpuAntialiasing() * m_Fractal.GetGpuAntialiasing()) {
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
            auto handle = m_Fractal.EnqueueRender();
            handle.Wait();

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
