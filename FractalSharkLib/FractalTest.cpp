#include "stdafx.h"
#include "FractalTest.h"

#include "Fractal.h"

FractalTest::FractalTest(Fractal& fractal) : m_Fractal(fractal) {
}

void FractalTest::BasicTestInternal(size_t& test_index) {
    constexpr bool IncludeSlow = false;
    const wchar_t* DirName = L"BasicTest";
    auto ret = CreateDirectory(DirName, nullptr);
    if (ret == 0 && GetLastError() != ERROR_ALREADY_EXISTS) {
        ::MessageBox(nullptr, L"Error creating directory!", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    // First, iterate over all the supported RenderAlgorithm entries and render the default view:
    // Skip AUTO plus all LAO-only algorithms.  They produce a black screen for the default view.
    for (size_t i = 0; i < (size_t)RenderAlgorithm::AUTO; i++) {
        auto CurAlg = static_cast<RenderAlgorithm>(i);
        if (CurAlg != RenderAlgorithm::Gpu1x32PerturbedLAv2LAO &&
            CurAlg != RenderAlgorithm::Gpu2x32PerturbedLAv2LAO &&
            CurAlg != RenderAlgorithm::Gpu1x64PerturbedLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO &&

            CurAlg != RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO &&
            CurAlg != RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO &&
            CurAlg != RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO &&
            CurAlg != RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO) {
            BasicOneTest(0, test_index, DirName, L"View0", CurAlg);
            ++test_index;
        }
    }

    // Now, iterate over all the RenderAlgorithm entries that should work with View #5.
    RenderAlgorithm View5Algs[] = {
        RenderAlgorithm::Gpu1x32PerturbedScaled,
        RenderAlgorithm::Gpu2x32PerturbedScaled,
        RenderAlgorithm::GpuHDRx32PerturbedScaled,

        RenderAlgorithm::Gpu1x64PerturbedBLA,
        RenderAlgorithm::GpuHDRx32PerturbedBLA,
        RenderAlgorithm::GpuHDRx64PerturbedBLA,

        RenderAlgorithm::Gpu1x64PerturbedLAv2,
        RenderAlgorithm::Gpu1x64PerturbedLAv2PO,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO,

        RenderAlgorithm::Gpu1x64PerturbedLAv2LAO,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
    };

    for (auto CurAlg : View5Algs) {
        BasicOneTest(5, test_index, DirName, L"View5", CurAlg);
        ++test_index;
    }

    if constexpr (IncludeSlow) {
        // This one is quite slow.  Be advised.
        RenderAlgorithm View10Algs[] = {
            RenderAlgorithm::GpuHDRx32PerturbedLAv2,
            RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
            RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
            RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
            RenderAlgorithm::GpuHDRx64PerturbedLAv2,
            RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,

            RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
            RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
            RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
        };

        for (auto CurAlg : View10Algs) {
            BasicOneTest(10, test_index, DirName, L"View10", CurAlg);
            ++test_index;
        }
    }

    // Finally, iterate over all the RenderAlgorithm entries that should work with View #11.
    RenderAlgorithm View11Algs[] = {
        RenderAlgorithm::GpuHDRx32PerturbedScaled,

        RenderAlgorithm::GpuHDRx32PerturbedBLA,
        RenderAlgorithm::GpuHDRx64PerturbedBLA,

        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO,

        RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
    };

    for (auto CurAlg : View11Algs) {
        BasicOneTest(11, test_index, DirName, L"View11", CurAlg);
        ++test_index;
    }
}

void FractalTest::BasicOneTest(
    size_t view_index,
    size_t test_index,
    const wchar_t* dir_name,
    const wchar_t* test_prefix,
    RenderAlgorithm alg_to_test) {

    size_t alg_to_test_int = static_cast<size_t>(alg_to_test);
    m_Fractal.SetRenderAlgorithm(alg_to_test);
    m_Fractal.View(view_index);
    m_Fractal.ForceRecalc();

    auto iter_type_str = IterTypeEnum::Bits32 == m_Fractal.GetIterType() ? "32" : "64";
    auto name = m_Fractal.GetRenderAlgorithmName();
    auto name_with_int_prefix =
        std::to_string(test_index) +
        " - Alg#" + std::to_string(alg_to_test_int) +
        " - Bits# " + iter_type_str +
        " - " + name;

    m_Fractal.CalcFractal(false);

    std::wstring filename_w;
    std::transform(
        name_with_int_prefix.begin(),
        name_with_int_prefix.end(),
        std::back_inserter(filename_w), [](char c) {
            return (wchar_t)c;
        });

    filename_w = std::wstring(dir_name) + L"\\" + std::wstring(test_prefix) + L" - " + filename_w;

    m_Fractal.SaveCurrentFractal(filename_w, false);
    //SaveItersAsText(filename_w);
}

void FractalTest::BasicTest() {
    m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

    size_t test_index = 0;

    auto lambda = [&]() {
        m_Fractal.SetIterType(IterTypeEnum::Bits32);
        BasicTestInternal(test_index);

        m_Fractal.SetIterType(IterTypeEnum::Bits64);
        BasicTestInternal(test_index);
        };

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);
    lambda();

    m_Fractal.SetResultsAutosave(AddPointOptions::EnableWithoutSave);
    lambda();

    m_Fractal.InitialDefaultViewAndSettings();
    m_Fractal.CalcFractal(false);
}

void FractalTest::Benchmark() {
    static constexpr size_t NumIterations = 5;

    if (m_Fractal.GetRepaint() == false) {
        m_Fractal.ToggleRepainting();
    }

    std::vector<size_t> overallTimes;
    std::vector<size_t> perPixelTimes;
    std::vector<size_t> refOrbitTimes;
    std::vector<size_t> LAGenerationTimes;

    for (size_t i = 0; i < NumIterations; i++) {
        m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        uint64_t InternalPeriodMaybeZero;
        uint64_t CompressedIters;
        uint64_t UncompressedIters;
        int32_t CompressionErrorExp;
        uint64_t OrbitMilliseconds;
        uint64_t LAMilliseconds;
        uint64_t LASize;
        std::string PerturbationAlg;
        m_Fractal.GetSomeDetails(
            InternalPeriodMaybeZero,
            CompressedIters,
            UncompressedIters,
            CompressionErrorExp,
            OrbitMilliseconds,
            LAMilliseconds,
            LASize,
            PerturbationAlg);

        overallTimes.push_back(m_Fractal.GetBenchmarkOverall().GetDeltaInMs());
        perPixelTimes.push_back(m_Fractal.GetBenchmarkPerPixel().GetDeltaInMs());
        refOrbitTimes.push_back(OrbitMilliseconds);
        LAGenerationTimes.push_back(LAMilliseconds);
    }

    // Write benchmarkData to the file BenchmarkResults.txt.  Truncate
    // the file if it already exists.
    std::ofstream file("BenchmarkResults.txt", std::ios::binary | std::ios::trunc);

    auto printVectorWithDescription = [&](const std::string& description, const std::vector<size_t>& vec) {
        file << description << "\r\n";
        for (size_t i = 0; i < NumIterations; i++) {
            file << vec[i] << "\r\n";
        }
        file << "\r\n";
    };

    printVectorWithDescription("Overall times (ms)", overallTimes);
    printVectorWithDescription("Per pixel times (ms)", perPixelTimes);
    printVectorWithDescription("RefOrbit times (ms)", refOrbitTimes);
    printVectorWithDescription("LA generation times (ms)", LAGenerationTimes);

    file.close();
}