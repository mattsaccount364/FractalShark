#include "stdafx.h"
#include "Fractal.h"
#include "RenderBenchmark.h"
#include "RenderThreadPool.h"
#include <fstream>
#include <vector>

void
FractalShark::BenchmarkFractal(Fractal &fractal, RefOrbitCalc::PerturbationResultType type)
{
    if (auto *pool = fractal.GetRenderPool()) {
        pool->Drain();
    }

    static constexpr size_t NumIterations = 5;

    if (fractal.GetRepaint() == false) {
        fractal.ToggleRepainting();
    }

    std::vector<size_t> overallTimes;
    std::vector<size_t> perPixelTimes;
    std::vector<size_t> refOrbitTimes;
    std::vector<size_t> laGenerationTimes;

    for (size_t i = 0; i < NumIterations; i++) {
        fractal.ClearPerturbationResults(type);
        fractal.ForceRecalc();
        fractal.CalcFractal(true);

        RefOrbitDetails details;
        fractal.GetSomeDetails(details);

        overallTimes.push_back(fractal.GetBenchmark().m_Overall.GetDeltaInMs());
        perPixelTimes.push_back(fractal.GetBenchmark().m_PerPixel.GetDeltaInMs());
        refOrbitTimes.push_back(details.OrbitMilliseconds);
        laGenerationTimes.push_back(details.LAMilliseconds);
    }

    // Write benchmarkData to the file BenchmarkResults.txt.  Truncate
    // the file if it already exists.
    std::ofstream file("BenchmarkResults.txt", std::ios::binary | std::ios::trunc);

    auto printVectorWithDescription = [&](const std::string &description,
                                          const std::vector<size_t> &vec) {
        file << description << "\r\n";
        for (size_t i = 0; i < NumIterations; i++) {
            file << vec[i] << "\r\n";
        }
        file << "\r\n";
    };

    printVectorWithDescription("Overall times (ms)", overallTimes);
    printVectorWithDescription("Per pixel times (ms)", perPixelTimes);
    printVectorWithDescription("RefOrbit times (ms)", refOrbitTimes);
    printVectorWithDescription("LA generation times (ms)", laGenerationTimes);

    file.close();
}
