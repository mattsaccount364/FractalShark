#include "stdafx.h"
#include "FractalTest.h"

#include "Fractal.h"

FractalTest::FractalTest(Fractal &fractal) : m_Fractal(fractal) {
}

void FractalTest::TestPreReq(const wchar_t *dirName) {
    auto ret = CreateDirectory(dirName, nullptr);
    if (ret == 0 && GetLastError() != ERROR_ALREADY_EXISTS) {
        ::MessageBox(nullptr, L"Error creating directory!", L"", MB_OK | MB_APPLMODAL);
        return;
    }
}

void FractalTest::BasicTestInternal(
    const wchar_t *dirName,
    size_t &testIndex,
    IterTypeEnum iterType) {

    constexpr bool IncludeSlow = false;

    // First, iterate over all the supported RenderAlgorithm entries and render the default view:
    // Skip AUTO plus all LAO-only algorithms.  They produce a black screen for the default view.
    for (size_t i = 0; i < (size_t)RenderAlgorithm::AUTO; i++) {
        auto curAlg = static_cast<RenderAlgorithm>(i);
        if (curAlg != RenderAlgorithm::Gpu1x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu2x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu1x64PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO &&

            curAlg != RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO) {
            BasicOneTest(0, testIndex, dirName, L"View0", curAlg, iterType);
            ++testIndex;
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

    for (auto curAlg : View5Algs) {
        BasicOneTest(5, testIndex, dirName, L"View5", curAlg, iterType);
        ++testIndex;
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

        for (auto curAlg : View10Algs) {
            BasicOneTest(10, testIndex, dirName, L"View10", curAlg, iterType);
            ++testIndex;
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

    for (auto curAlg : View11Algs) {
        BasicOneTest(11, testIndex, dirName, L"View11", curAlg, iterType);
        ++testIndex;
    }
}

std::string FractalTest::GenFilename(
    size_t testIndex,
    RenderAlgorithm algToTest,
    IterTypeEnum iterType,
    std::string baseName) {

    auto iterTypeStr =
        IterTypeEnum::Bits32 == iterType ? "32" : "64";

    auto nameWithIntPrefix =
        std::to_string(testIndex) +
        " - Alg#" + std::to_string(static_cast<size_t>(algToTest)) +
        " - Bits# " + iterTypeStr +
        " - " + baseName;
    return nameWithIntPrefix;
}

std::wstring FractalTest::GenFilenameW(
    size_t testIndex,
    RenderAlgorithm algToTest,
    IterTypeEnum iterType,
    const wchar_t *testPrefix,
    const wchar_t *dirName,
    std::string baseName) {

    auto nameWithIntPrefix = GenFilename(testIndex, algToTest, iterType, baseName);

    std::wstring filenameW;
    std::transform(
        nameWithIntPrefix.begin(),
        nameWithIntPrefix.end(),
        std::back_inserter(filenameW), [](char c) {
            return (wchar_t)c;
        });

    filenameW = std::wstring(dirName) + L"\\" + std::wstring(testPrefix) + L" - " + filenameW;

    return filenameW;
}

void FractalTest::BasicOneTest(
    size_t viewIndex,
    size_t testIndex,
    const wchar_t *dirName,
    const wchar_t *testPrefix,
    RenderAlgorithm algToTest,
    IterTypeEnum iterType) {

    m_Fractal.SetRenderAlgorithm(algToTest);
    m_Fractal.View(viewIndex);
    m_Fractal.ForceRecalc();

    auto name = m_Fractal.GetRenderAlgorithmName();
    auto nameWithIntPrefix = GenFilename(testIndex, algToTest, m_Fractal.GetIterType(), name);

    m_Fractal.CalcFractal(false);

    const auto filenameW = GenFilenameW(testIndex, algToTest, iterType, testPrefix, dirName, name);

    m_Fractal.SaveCurrentFractal(filenameW, false);
    //SaveItersAsText(filenameW);
}

void FractalTest::TestBasic() {
    const wchar_t *dirName = L"TestBasic";
    TestPreReq(dirName);

    m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

    size_t testIndex = 0;

    auto lambda = [&]() {
        IterTypeEnum curIterType = IterTypeEnum::Bits32;
        m_Fractal.SetIterType(curIterType);
        BasicTestInternal(dirName, testIndex, curIterType);

        curIterType = IterTypeEnum::Bits64;
        m_Fractal.SetIterType(curIterType);
        BasicTestInternal(dirName, testIndex, curIterType);
        };

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);
    lambda();

    m_Fractal.SetResultsAutosave(AddPointOptions::EnableWithoutSave);
    lambda();

    m_Fractal.InitialDefaultViewAndSettings();
    m_Fractal.CalcFractal(false);
}

void FractalTest::TestReferenceSave() {

    const wchar_t *dirName = L"TestReferenceSave";
    TestPreReq(dirName);

    RenderAlgorithm View5Algs[] = {
        //RenderAlgorithm::Gpu1x64PerturbedLAv2,
        //RenderAlgorithm::Gpu1x64PerturbedRCLAv2,
        //RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        //RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        //RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
    };

    auto referenceSaveLoad = [&](size_t testIndex, IterTypeEnum iterType, RenderAlgorithm algToTest) {
        m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

        m_Fractal.SetIterType(iterType);
        m_Fractal.SetRenderAlgorithm(algToTest);
        m_Fractal.View(5);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        const wchar_t *testPrefix = L"RefOrbitTest";
        const auto algStr = m_Fractal.GetRenderAlgorithmName(algToTest);

        const auto genLocalFilename = [&](const std::wstring extraPrefix) {
            std::wstring fullPrefix = testPrefix + std::wstring(L" - ") + extraPrefix;
            return GenFilenameW(
                testIndex,
                algToTest,
                iterType,
                fullPrefix.c_str(),
                dirName,
                algStr);
            };

        const auto disableFilename = genLocalFilename(L"Disable") + L".txt";
        m_Fractal.SaveRefOrbit(CompressToDisk::Disable, disableFilename);

        const auto simpleFilename = genLocalFilename(L"Simple") + L".txt";
        m_Fractal.SaveRefOrbit(CompressToDisk::SimpleCompression, simpleFilename);

        const auto maxFilename = genLocalFilename(L"Max") + L".txt";
        m_Fractal.SaveRefOrbit(CompressToDisk::MaxCompression, maxFilename);

        const auto maxImaginaFilename = genLocalFilename(L"MaxImagina") + L".im";
        m_Fractal.SaveRefOrbit(CompressToDisk::MaxCompressionImagina, maxImaginaFilename);

        const auto originalImageFilename = genLocalFilename(L"Original");
        m_Fractal.SaveCurrentFractal(originalImageFilename, false);

        m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        m_Fractal.View(0);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        // Force 64-bit on load
        m_Fractal.SetIterType(IterTypeEnum::Bits64);

        m_Fractal.LoadRefOrbit(CompressToDisk::MaxCompressionImagina, maxImaginaFilename);
        m_Fractal.CalcFractal(false);

        const auto decompressedResultFilename = genLocalFilename(L"Decompressed");
        m_Fractal.SaveCurrentFractal(decompressedResultFilename, false);
        };

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);
    size_t testIndex = 0;

    for (auto curAlg : View5Algs) {
        referenceSaveLoad(testIndex, IterTypeEnum::Bits32, curAlg);
        referenceSaveLoad(testIndex, IterTypeEnum::Bits64, curAlg);

        testIndex++;
    }
}

void FractalTest::Benchmark(RefOrbitCalc::PerturbationResultType type) {
    static constexpr size_t NumIterations = 5;

    if (m_Fractal.GetRepaint() == false) {
        m_Fractal.ToggleRepainting();
    }

    std::vector<size_t> overallTimes;
    std::vector<size_t> perPixelTimes;
    std::vector<size_t> refOrbitTimes;
    std::vector<size_t> LAGenerationTimes;

    for (size_t i = 0; i < NumIterations; i++) {
        m_Fractal.ClearPerturbationResults(type);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        RefOrbitDetails details;
        m_Fractal.GetSomeDetails(details);

        overallTimes.push_back(m_Fractal.GetBenchmarkOverall().GetDeltaInMs());
        perPixelTimes.push_back(m_Fractal.GetBenchmarkPerPixel().GetDeltaInMs());
        refOrbitTimes.push_back(details.OrbitMilliseconds);
        LAGenerationTimes.push_back(details.LAMilliseconds);
    }

    // Write benchmarkData to the file BenchmarkResults.txt.  Truncate
    // the file if it already exists.
    std::ofstream file("BenchmarkResults.txt", std::ios::binary | std::ios::trunc);

    auto printVectorWithDescription = [&](const std::string &description, const std::vector<size_t> &vec) {
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