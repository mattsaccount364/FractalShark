#include "Exceptions.h"
#include "FeatureSummary.h"
#include "Fractal.h"
#include "FractalPalette.h"
#include "FractalSaveThreadPool.h"
#include "GuiFileOperations.h"
#include "ItersMemoryContainer.h"
#include "OrbitEndpointEvaluator.h"
#include "OrbitParameterPack.h"
#include "PointZoomBBConverter.h"
#include "RecommendedSettings.h"
#include "RenderAlgorithm.h"
#include "RenderToConsole.h"
#include "RenderToPng.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ScopedMpfComplex {
    mpf_complex Value;

    explicit ScopedMpfComplex(mp_bitcnt_t precision) { mpf_complex_init(Value, precision); }
    ~ScopedMpfComplex() { mpf_complex_clear(Value); }
};

void
SetComplex(mpf_complex &value, double real, double imaginary)
{
    mpf_set_d(value.re, real);
    mpf_set_d(value.im, imaginary);
}

std::filesystem::path
FindTestMap()
{
    std::filesystem::path directory = std::filesystem::current_path();
    while (true) {
        const std::filesystem::path candidate = directory / "test.map";
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) {
            return candidate;
        }

        const std::filesystem::path parent = directory.parent_path();
        if (parent == directory) {
            return {};
        }
        directory = parent;
    }
}

} // namespace

TEST(FractalSharkLib_ItersMemoryContainerStores32BitValues)
{
    ItersMemoryContainer iters(IterTypeEnum::Bits32, 3, 2, 2);

    ASSERT_EQ(iters.m_OutputWidth, size_t{3});
    ASSERT_EQ(iters.m_OutputHeight, size_t{2});
    ASSERT_EQ(iters.m_Width, size_t{6});
    ASSERT_EQ(iters.m_Height, size_t{4});

    uint64_t expectedSum = 0;
    for (size_t y = 0; y < iters.m_Height; ++y) {
        for (size_t x = 0; x < iters.m_Width; ++x) {
            const uint32_t value = static_cast<uint32_t>(y * iters.m_Width + x + 1);
            iters.SetItersArrayValSlow(x, y, value);
            expectedSum += value;
        }
    }

    ASSERT_EQ(iters.GetItersArrayValSlow(5, 3), IterTypeFull{24});

    ReductionResults results;
    iters.GetReductionResults(results);
    ASSERT_EQ(results.Min, uint64_t{1});
    ASSERT_EQ(results.Max, uint64_t{24});
    ASSERT_EQ(results.Sum, expectedSum);
}

TEST(FractalSharkLib_ItersMemoryContainerStores64BitValues)
{
    ItersMemoryContainer iters(IterTypeEnum::Bits64, 2, 1, 1);
    iters.SetItersArrayValSlow(0, 0, uint64_t{1} << 40);
    iters.SetItersArrayValSlow(1, 0, uint64_t{1} << 41);

    ReductionResults results;
    iters.GetReductionResults(results);
    ASSERT_EQ(results.Min, uint64_t{1} << 40);
    ASSERT_EQ(results.Max, uint64_t{1} << 41);
    ASSERT_EQ(results.Sum, (uint64_t{1} << 40) + (uint64_t{1} << 41));
}

TEST(FractalSharkLib_PaletteInitializationAndStateTransitions)
{
    FractalPalette palette;
    palette.InitializeAllPalettes();

    ASSERT_EQ(palette.GetPaletteDepth(), FractalPalette::DefaultPaletteDepth);
    ASSERT_TRUE(palette.GetCurrentPalInterleaved() != nullptr);
    ASSERT_TRUE(palette.GetCurrentNumColors() > 0);
    ASSERT_EQ(palette.GetCurrentNumColors(),
              palette.GetPalIters(FractalPaletteType::Default)[palette.GetPaletteDepthIndex()]);
    ASSERT_TRUE(palette.GetPaletteGeneration() > 0);

    palette.UsePaletteType(FractalPaletteType::Patriotic);
    ASSERT_EQ(static_cast<int>(palette.GetPaletteType()),
              static_cast<int>(FractalPaletteType::Patriotic));
    palette.UsePalette(20);
    ASSERT_EQ(palette.GetPaletteDepth(), uint32_t{20});
    palette.UsePalette(999);
    ASSERT_EQ(palette.GetPaletteDepthIndex(), 0);

    palette.SetPaletteAuxDepth(0);
    palette.UseNextPaletteAuxDepth(-1);
    ASSERT_EQ(palette.GetAuxDepth(), 16);
    palette.UseNextPaletteAuxDepth(1);
    ASSERT_EQ(palette.GetAuxDepth(), 0);

    palette.ResetPaletteRotation();
    palette.RotatePalette(3, 10);
    ASSERT_EQ(palette.GetPaletteRotation(), IterTypeFull{3});
    palette.RotatePalette(7, 10);
    ASSERT_EQ(palette.GetPaletteRotation(), IterTypeFull{0});
}

TEST(FractalSharkLib_CustomPaletteLoadsRgb8MapsTransactionally)
{
    const std::filesystem::path suppliedPalettePath = FindTestMap();
    ASSERT_FALSE(suppliedPalettePath.empty());

    const std::filesystem::path palettePath =
        std::filesystem::temp_directory_path() / "fractalshark-custom-palette-test.map";
    std::error_code error;
    std::filesystem::remove(palettePath, error);
    ASSERT_FALSE(static_cast<bool>(error));

    FractalPalette palette;
    palette.InitializeAllPalettes();
    const uint64_t generationBeforeLoad = palette.GetPaletteGeneration();
    palette.LoadCustomPalette(suppliedPalettePath);

    ASSERT_TRUE(palette.HasCustomPalette());
    ASSERT_TRUE(palette.GetPaletteGeneration() > generationBeforeLoad);

    const auto *customPalettes = palette.GetPalInterleaved(FractalPaletteType::Custom);
    const auto &customCounts = palette.GetPalIters(FractalPaletteType::Custom);
    for (size_t paletteIndex = 0; paletteIndex < FractalPalette::PaletteDepths.size(); ++paletteIndex) {
        const size_t expectedCount = size_t{1} << FractalPalette::PaletteDepths[paletteIndex];
        ASSERT_EQ(customPalettes[paletteIndex].size(), expectedCount);
        ASSERT_EQ(customCounts[paletteIndex], static_cast<uint32_t>(expectedCount));
    }

    const auto &eightBitPalette = customPalettes[FractalPalette::DefaultPaletteDepthIndex];
    ASSERT_EQ(eightBitPalette.size(), size_t{256});
    ASSERT_EQ(eightBitPalette[0].r, uint16_t{65535});
    ASSERT_EQ(eightBitPalette[0].g, uint16_t{16962});
    ASSERT_EQ(eightBitPalette[0].b, uint16_t{15163});
    ASSERT_EQ(eightBitPalette[127].r, uint16_t{0});
    ASSERT_EQ(eightBitPalette[127].g, uint16_t{47802});
    ASSERT_EQ(eightBitPalette[127].b, uint16_t{50886});
    ASSERT_EQ(eightBitPalette[255].r, uint16_t{65535});
    ASSERT_EQ(eightBitPalette[255].g, uint16_t{17733});
    ASSERT_EQ(eightBitPalette[255].b, uint16_t{14649});

    palette.UsePaletteType(FractalPaletteType::Custom);
    palette.UsePalette(static_cast<int>(FractalPalette::DefaultPaletteDepth));
    ASSERT_EQ(palette.GetCurrentNumColors(), uint32_t{256});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[0].r, uint16_t{65535});

    {
        std::ofstream output(palettePath);
        ASSERT_TRUE(static_cast<bool>(output));
        output << "# comment\n";
        output << "\n";
        output << "  // another comment\n";
        output << "-1 256 42 ignored trailing components\n";
        output << "+1 2 3\n";
    }

    const uint64_t generationBeforeReplacement = palette.GetPaletteGeneration();
    palette.LoadCustomPalette(palettePath);
    ASSERT_EQ(palette.GetPaletteGeneration(), generationBeforeReplacement + 1);
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[0].r, uint16_t{0});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[0].g, uint16_t{65535});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[0].b, uint16_t{10794});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[128].r, uint16_t{257});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[128].g, uint16_t{514});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[128].b, uint16_t{771});

    {
        std::ofstream output(palettePath, std::ios::trunc);
        ASSERT_TRUE(static_cast<bool>(output));
        output << "1 2\n";
    }

    ASSERT_THROWS(palette.LoadCustomPalette(palettePath), FractalSharkSeriousException);
    ASSERT_TRUE(palette.HasCustomPalette());
    ASSERT_EQ(palette.GetPaletteGeneration(), generationBeforeReplacement + 1);
    ASSERT_EQ(palette.GetCurrentNumColors(), uint32_t{256});
    ASSERT_EQ(palette.GetCurrentPalInterleaved()[0].r, uint16_t{0});

    std::filesystem::remove(palettePath, error);
    ASSERT_FALSE(static_cast<bool>(error));
}

TEST(FractalSharkLib_RenderAlgorithmMetadataIsIndexedAndPartitioned)
{
    const size_t maxAlgorithm = static_cast<size_t>(RenderAlgorithmEnum::MAX);
    const size_t firstGpu = static_cast<size_t>(RenderAlgorithmEnum::Gpu1x32);
    const size_t autoAlgorithm = static_cast<size_t>(RenderAlgorithmEnum::AUTO);

    for (size_t i = 0; i < maxAlgorithm; ++i) {
        const auto &algorithm = RenderAlgorithms[i];
        ASSERT_EQ(static_cast<size_t>(algorithm.Algorithm), i);
        ASSERT_TRUE(algorithm.AlgorithmStr != nullptr);
        ASSERT_TRUE(algorithm.AlgorithmStr[0] != '\0');
        for (size_t j = 0; j < i; ++j) {
            ASSERT_TRUE(std::string{algorithm.AlgorithmStr} != RenderAlgorithms[j].AlgorithmStr);
        }

        if (i < firstGpu) {
            ASSERT_EQ(static_cast<int>(algorithm.Gpu), static_cast<int>(RequiresGpu::No));
        } else if (i < autoAlgorithm) {
            ASSERT_EQ(static_cast<int>(algorithm.Gpu), static_cast<int>(RequiresGpu::Yes));
        }
    }

    ASSERT_EQ(static_cast<size_t>(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64).Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::Cpu64));
    ASSERT_EQ(static_cast<size_t>(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO).Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::AUTO));
}

TEST(FractalSharkLib_FileHelpersHandleExtensionsAndOrbitDiscovery)
{
    ASSERT_EQ(FractalShark::AppendExtensionIfMissing("image", ".png"), "image.png");
    ASSERT_EQ(FractalShark::AppendExtensionIfMissing("folder.v1/image", ".png"), "folder.v1/image.png");
    ASSERT_EQ(FractalShark::AppendExtensionIfMissing("image.jpg", ".png"), "image.jpg");
    ASSERT_TRUE(FractalShark::AppendExtensionIfMissing(std::wstring{L"image"}, L".png") ==
                std::wstring{L"image.png"});

    const auto directory = std::filesystem::temp_directory_path() / "fractalshark-file-helper-test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory, error);
    ASSERT_FALSE(static_cast<bool>(error));

    std::ofstream(directory / "B.IM").put('\n');
    std::ofstream(directory / "a.im").put('\n');
    std::ofstream(directory / "ignored.txt").put('\n');
    std::filesystem::create_directories(directory / "nested.im", error);
    ASSERT_FALSE(static_cast<bool>(error));

    const auto files = FractalShark::FindReferenceOrbitFiles(directory, 1);
    ASSERT_EQ(files.size(), size_t{1});
    ASSERT_EQ(files.front().filename().string(), "a.im");

    std::filesystem::remove_all(directory, error);
}

TEST(FractalSharkLib_RecommendedSettingsAndOrbitParametersPreserveContracts)
{
    HighPrecision orbitX{0};
    HighPrecision orbitY{0};
    HighPrecision zoom{1};
    const RenderAlgorithm cpu64 = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);

    RecommendedSettings settings(
        256, orbitX, orbitY, zoom, cpu64, Fractal::GetMaxIterations<uint32_t>());
    ASSERT_EQ(static_cast<int>(settings.GetIterType()), static_cast<int>(IterTypeEnum::Bits32));
    ASSERT_EQ(static_cast<size_t>(settings.GetRenderAlgorithm().Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::Cpu64));
    ASSERT_EQ(settings.GetNumIterations(), Fractal::GetMaxIterations<uint32_t>());

    settings.OverrideIterType(IterTypeEnum::Bits64);
    ASSERT_EQ(static_cast<int>(settings.GetIterType()), static_cast<int>(IterTypeEnum::Bits64));

    RecommendedSettings empty;
    ASSERT_THROWS(empty.GetPointZoomBBConverter(), FractalSharkSeriousException);

    OrbitParameterPack parameters;
    parameters.iterationLimit = std::numeric_limits<uint64_t>::max();
    ASSERT_EQ(parameters.GetSaturatedIterationCount<uint32_t>(), uint32_t{UINT32_MAX - 1});
    ASSERT_EQ(parameters.GetSaturatedIterationCount<uint64_t>(), uint64_t{UINT64_MAX - 1});
    parameters.iterationLimit = 37;
    ASSERT_EQ(parameters.GetSaturatedIterationCount<uint32_t>(), uint32_t{37});
}

TEST(FractalSharkLib_DisabledGpuModeAvoidsRuntimeInitialization)
{
    Fractal fractal(
        32, 32, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);

    ASSERT_EQ(static_cast<int>(fractal.GetGpuMode()), static_cast<int>(GpuMode::Disabled));
    ASSERT_TRUE(fractal.GpuBypassed());
    ASSERT_EQ(static_cast<size_t>(fractal.GetRenderAlgorithm().Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::Cpu64));

    const auto gpuAlgorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32);
    ASSERT_FALSE(fractal.SetRenderAlgorithm(gpuAlgorithm));
    ASSERT_EQ(static_cast<size_t>(fractal.GetRenderAlgorithm().Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::Cpu64));

    const auto cpuAlgorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::CpuHDR32);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(cpuAlgorithm));
    ASSERT_EQ(static_cast<size_t>(fractal.GetRenderAlgorithm().Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::CpuHDR32));
}

TEST(FractalSharkLib_FeatureSummaryTracksCandidatesAndScreenCoordinates)
{
    FeatureSummary summary(
        HighPrecision{-1}, HighPrecision{-1}, HighPrecision{0.5}, FeatureFinderMode::Direct);
    summary.SetCandidate(HighPrecision{-0.75},
                         HighPrecision{0.1},
                         7,
                         HDRFloat<double>{0.001},
                         HighPrecision{0.25},
                         -3,
                         256);
    ASSERT_TRUE(summary.HasCandidate());
    ASSERT_EQ(summary.GetCandidate()->period, IterTypeFull{7});
    ASSERT_EQ(summary.GetCandidate()->mpfPrecBits, mp_bitcnt_t{256});

    summary.SetFound(
        HighPrecision{1}, HighPrecision{1}, 11, HDRFloat<double>{0.0001}, HighPrecision{0.25});
    summary.SetNumIterationsAtFind(123);
    summary.SetRefined();
    ASSERT_TRUE(summary.IsRefined());
    ASSERT_EQ(summary.GetPeriod(), IterTypeFull{11});
    ASSERT_EQ(summary.GetNumIterationsAtFind(), IterTypeFull{123});
    ASSERT_NEAR(summary.GetResidual2().toDouble(), 0.0001, 1e-15);

    PointZoomBBConverter view(
        HighPrecision{0}, HighPrecision{0}, HighPrecision{1}, PointZoomBBConverter::TestMode::Enabled);
    ASSERT_TRUE(summary.ComputeZoomFactor(view) >= view.GetZoomFactor());

    Fractal fractal(
        32, 32, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    summary.EstablishScreenCoordinates(fractal);
    int xStart = 0;
    int yStart = 0;
    int xEnd = 0;
    int yEnd = 0;
    summary.GetScreenCoordinates(xStart, yStart, xEnd, yEnd);
    ASSERT_TRUE(xStart >= 0 && xStart < static_cast<int>(fractal.GetRenderWidth()));
    ASSERT_TRUE(xEnd >= 0 && xEnd < static_cast<int>(fractal.GetRenderWidth()));
    ASSERT_TRUE(yStart >= 0 && yStart < static_cast<int>(fractal.GetRenderHeight()));
    ASSERT_TRUE(yEnd >= 0 && yEnd < static_cast<int>(fractal.GetRenderHeight()));

    summary.ClearCandidate();
    ASSERT_FALSE(summary.HasCandidate());
}

TEST(FractalSharkLib_LAFeatureFinderAgreesWithPerturbationNearPeriodTwo)
{
    for (const IterTypeEnum iterType : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        Fractal fractal(
            32, 32, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
        ASSERT_TRUE(
            fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
        fractal.SetIterType(iterType);
        fractal.SetNumIterations<uint64_t>(128);
        const PointZoomBBConverter view(HighPrecision{-1},
                                        HighPrecision{0},
                                        HighPrecision{8},
                                        PointZoomBBConverter::TestMode::Enabled);
        ASSERT_TRUE(fractal.RecenterViewCalc(view));
        fractal.CalcFractal(true);

        fractal.TryFindPeriodicPoint(16, 16, FeatureFinderMode::PT);
        const FeatureSummary *pt = fractal.ChooseClosestFeatureToScreenPoint(16, 16);
        ASSERT_TRUE(pt != nullptr);
        const IterTypeFull ptPeriod = pt->GetPeriod();
        const double ptX = static_cast<double>(pt->GetFoundX());
        const double ptY = static_cast<double>(pt->GetFoundY());

        fractal.TryFindPeriodicPoint(16, 16, FeatureFinderMode::LA);
        const FeatureSummary *la = fractal.ChooseClosestFeatureToScreenPoint(16, 16);
        ASSERT_TRUE(la != nullptr);
        ASSERT_EQ(la->GetPeriod(), ptPeriod);
        ASSERT_NEAR(static_cast<double>(la->GetFoundX()), ptX, 1e-8);
        ASSERT_NEAR(static_cast<double>(la->GetFoundY()), ptY, 1e-8);

        fractal.TryFindPeriodicPoint(20, 16, FeatureFinderMode::PT);
        ASSERT_TRUE(fractal.ChooseClosestFeatureToScreenPoint(20, 16) == nullptr);
        fractal.TryFindPeriodicPoint(20, 16, FeatureFinderMode::LA);
        ASSERT_TRUE(fractal.ChooseClosestFeatureToScreenPoint(20, 16) == nullptr);
    }
}

TEST(FractalSharkLib_LAFeatureFinderHandlesIncompleteSearch)
{
    Fractal fractal(
        32, 32, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(32);
    const PointZoomBBConverter view(HighPrecision{"-0.743643887037151"},
                                    HighPrecision{"0.13182590420533"},
                                    HighPrecision{1000},
                                    PointZoomBBConverter::TestMode::Enabled);
    ASSERT_TRUE(fractal.RecenterViewCalc(view));
    fractal.CalcFractal(true);

    fractal.TryFindPeriodicPoint(20, 16, FeatureFinderMode::PT);
    ASSERT_TRUE(fractal.ChooseClosestFeatureToScreenPoint(20, 16) == nullptr);
    fractal.TryFindPeriodicPoint(20, 16, FeatureFinderMode::LA);
    ASSERT_TRUE(fractal.ChooseClosestFeatureToScreenPoint(20, 16) == nullptr);
}

TEST(FractalSharkLib_PTFeatureFinderFindsPeriod43)
{
    Fractal fractal(
        32, 32, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(256);
    const PointZoomBBConverter view(HighPrecision{"-0.743643887037151"},
                                    HighPrecision{"0.13182590420533"},
                                    HighPrecision{1000},
                                    PointZoomBBConverter::TestMode::Enabled);
    ASSERT_TRUE(fractal.RecenterViewCalc(view));
    fractal.CalcFractal(true);

    fractal.TryFindPeriodicPoint(20, 16, FeatureFinderMode::PT);
    const FeatureSummary *feature = fractal.ChooseClosestFeatureToScreenPoint(20, 16);
    ASSERT_TRUE(feature != nullptr);
    ASSERT_EQ(feature->GetPeriod(), IterTypeFull{43});
    ASSERT_NEAR(static_cast<double>(feature->GetFoundX()), -0.7431325047313953, 1e-10);
    ASSERT_NEAR(static_cast<double>(feature->GetFoundY()), 0.1317911288697174, 1e-10);
}

TEST(FractalSharkLib_OrbitEndpointCpuBackendsAgree)
{
    constexpr mp_bitcnt_t precision = 256;
    constexpr uint64_t period = 12;
    ScopedMpfComplex c{precision};
    ScopedMpfComplex zSt{precision};
    ScopedMpfComplex dzSt{precision};
    ScopedMpfComplex zMt{precision};
    ScopedMpfComplex dzMt{precision};
    SetComplex(c.Value, -0.125, 0.2);

    HDRFloat<double> d2rSt;
    HDRFloat<double> d2iSt;
    HDRFloat<double> d2rMt;
    HDRFloat<double> d2iMt;
    const auto completedSt = EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend::CpuST,
                                                            c.Value,
                                                            period,
                                                            zSt.Value,
                                                            dzSt.Value,
                                                            d2rSt,
                                                            d2iSt,
                                                            precision,
                                                            precision);
    const auto completedMt = EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend::CpuMT,
                                                            c.Value,
                                                            period,
                                                            zMt.Value,
                                                            dzMt.Value,
                                                            d2rMt,
                                                            d2iMt,
                                                            precision,
                                                            precision);

    ASSERT_EQ(completedSt, period);
    ASSERT_EQ(completedMt, period);
    ASSERT_NEAR(mpf_get_d(zSt.Value.re), mpf_get_d(zMt.Value.re), 1e-12);
    ASSERT_NEAR(mpf_get_d(zSt.Value.im), mpf_get_d(zMt.Value.im), 1e-12);
    ASSERT_NEAR(mpf_get_d(dzSt.Value.re), mpf_get_d(dzMt.Value.re), 1e-12);
    ASSERT_NEAR(mpf_get_d(dzSt.Value.im), mpf_get_d(dzMt.Value.im), 1e-12);
    ASSERT_NEAR(d2rSt.toDouble(), d2rMt.toDouble(), 1e-12);
    ASSERT_NEAR(d2iSt.toDouble(), d2iMt.toDouble(), 1e-12);
}

TEST(FractalSharkLib_OrbitEndpointResumeMatchesFreshEvaluation)
{
    constexpr mp_bitcnt_t precision = 256;
    constexpr uint64_t checkpoint = 5;
    constexpr uint64_t period = 12;
    ScopedMpfComplex c{precision};
    ScopedMpfComplex zFresh{precision};
    ScopedMpfComplex dzFresh{precision};
    ScopedMpfComplex zResume{precision};
    ScopedMpfComplex dzResume{precision};
    SetComplex(c.Value, -0.125, 0.2);

    HDRFloat<double> d2rFresh;
    HDRFloat<double> d2iFresh;
    HDRFloat<double> d2rResume;
    HDRFloat<double> d2iResume;
    const auto completedFresh = EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend::CpuST,
                                                               c.Value,
                                                               period,
                                                               zFresh.Value,
                                                               dzFresh.Value,
                                                               d2rFresh,
                                                               d2iFresh,
                                                               precision,
                                                               precision);
    const auto completedCheckpoint = EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend::CpuST,
                                                                    c.Value,
                                                                    checkpoint,
                                                                    zResume.Value,
                                                                    dzResume.Value,
                                                                    d2rResume,
                                                                    d2iResume,
                                                                    precision,
                                                                    precision);
    const auto completedResume = EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend::CpuST,
                                                                c.Value,
                                                                period,
                                                                zResume.Value,
                                                                dzResume.Value,
                                                                d2rResume,
                                                                d2iResume,
                                                                precision,
                                                                precision,
                                                                checkpoint);

    ASSERT_EQ(completedFresh, period);
    ASSERT_EQ(completedCheckpoint, checkpoint);
    ASSERT_EQ(completedResume, period);
    ASSERT_NEAR(mpf_get_d(zFresh.Value.re), mpf_get_d(zResume.Value.re), 1e-12);
    ASSERT_NEAR(mpf_get_d(zFresh.Value.im), mpf_get_d(zResume.Value.im), 1e-12);
    ASSERT_NEAR(mpf_get_d(dzFresh.Value.re), mpf_get_d(dzResume.Value.re), 1e-12);
    ASSERT_NEAR(mpf_get_d(dzFresh.Value.im), mpf_get_d(dzResume.Value.im), 1e-12);
    ASSERT_NEAR(d2rFresh.toDouble(), d2rResume.toDouble(), 1e-12);
    ASSERT_NEAR(d2iFresh.toDouble(), d2iResume.toDouble(), 1e-12);
}

TEST(FractalSharkLib_RenderPoolSeparatesMutationFromCommand)
{
    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    const auto cpu64 = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);

    fractal.EnqueueMutation("test mutation", [](Fractal &value) { value.SetIterationPrecision(7); })
        .Wait();
    ASSERT_EQ(fractal.GetIterationPrecision(), uint32_t{7});

    fractal
        .EnqueueCommand("test command",
                        [cpu64](Fractal &value) {
                            value.SetNumIterations<uint32_t>(64);
                            (void)value.SetRenderAlgorithm(cpu64);
                        })
        .Wait();
    fractal.GetRenderPool()->Drain();

    ASSERT_EQ(fractal.GetNumIterationsRT(), IterTypeFull{64});
    ASSERT_EQ(static_cast<size_t>(fractal.GetRenderAlgorithm().Algorithm),
              static_cast<size_t>(RenderAlgorithmEnum::Cpu64));
    ASSERT_EQ(fractal.GetCurIters().m_OutputWidth, size_t{16});
    ASSERT_EQ(fractal.GetCurIters().m_OutputHeight, size_t{16});
}

TEST(FractalSharkLib_RecenterViewScreenHandlesDragPastLastRow)
{
    Fractal fractal(
        800, 800, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(
        GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLA)));

    const PointZoomBBConverter oldView = fractal.GetPtz();
    const HighPrecision expectedCenterX = oldView.XFromScreenToCalc(HighPrecision{192}, 800, 1);
    const HighPrecision expectedCenterY = oldView.YFromScreenToCalc(HighPrecision{800}, 800, 1);

    ASSERT_TRUE(fractal.RecenterViewScreen(Environment::ScreenRect{184, 790, 200, 810}));
    const HighPrecision centerX = (fractal.GetMinX() + fractal.GetMaxX()) / HighPrecision{2};
    const HighPrecision centerY = (fractal.GetMinY() + fractal.GetMaxY()) / HighPrecision{2};
    ASSERT_NEAR(static_cast<double>(centerX), static_cast<double>(expectedCenterX), 1e-12);
    ASSERT_NEAR(static_cast<double>(centerY), static_cast<double>(expectedCenterY), 1e-12);
}

TEST(FractalSharkLib_RecenterViewScreenHandlesOffscreenAndReversedDrag)
{
    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(
        GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLA)));
    fractal.ResetDimensions(16, 16, 2);

    const PointZoomBBConverter oldView = fractal.GetPtz();
    const HighPrecision reversedCenterX = oldView.XFromScreenToCalc(HighPrecision{14}, 16, 1);
    const HighPrecision reversedCenterY = oldView.YFromScreenToCalc(HighPrecision{14}, 16, 1);
    ASSERT_TRUE(fractal.RecenterViewScreen(Environment::ScreenRect{20, 20, 8, 8}));
    ASSERT_NEAR(static_cast<double>((fractal.GetMinX() + fractal.GetMaxX()) / HighPrecision{2}),
                static_cast<double>(reversedCenterX),
                1e-12);
    ASSERT_NEAR(static_cast<double>((fractal.GetMinY() + fractal.GetMaxY()) / HighPrecision{2}),
                static_cast<double>(reversedCenterY),
                1e-12);

    const PointZoomBBConverter currentView = fractal.GetPtz();
    const HighPrecision expectedCenterX = currentView.XFromScreenToCalc(HighPrecision{20}, 16, 1);
    const HighPrecision expectedCenterY = currentView.YFromScreenToCalc(HighPrecision{20}, 16, 1);
    ASSERT_TRUE(fractal.RecenterViewScreen(Environment::ScreenRect{24, 24, 16, 16}));
    const HighPrecision centerX = (fractal.GetMinX() + fractal.GetMaxX()) / HighPrecision{2};
    const HighPrecision centerY = (fractal.GetMinY() + fractal.GetMaxY()) / HighPrecision{2};
    ASSERT_NEAR(static_cast<double>(centerX), static_cast<double>(expectedCenterX), 1e-12);
    ASSERT_NEAR(static_cast<double>(centerY), static_cast<double>(expectedCenterY), 1e-12);
}

TEST(FractalSharkLib_RenderToConsoleProducesTextAndColorModes)
{
    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(64);
    fractal.CalcFractal(true);

    ConsoleRenderOptions options;
    options.ConsoleWidth = 8;
    options.ConsoleHeight = 4;
    std::ostringstream plain;
    RenderToConsole(fractal, options, plain);
    ASSERT_TRUE(!plain.str().empty());
    ASSERT_TRUE(plain.str().find('\n') != std::string::npos);

    options.Color = true;
    std::ostringstream color;
    RenderToConsole(fractal, options, color);
    ASSERT_TRUE(color.str().find("\033[") != std::string::npos ||
                color.str().find("All pixels are set-interior") != std::string::npos);
}

TEST(FractalSharkLib_SavePoolBoundsAndReusesWorkers)
{
    FractalSaveThreadPool pool(2, [] { return uint32_t{0}; });
    std::mutex mutex;
    std::condition_variable condition;
    std::set<std::thread::id> workerIds;
    size_t started = 0;
    bool release = false;

    const auto makeBlockedTask = [&] {
        return FractalSaveThreadPool::Task([&] {
            std::unique_lock lock(mutex);
            workerIds.insert(std::this_thread::get_id());
            ++started;
            condition.notify_all();
            condition.wait(lock, [&] { return release; });
        });
    };
    pool.Submit(makeBlockedTask);
    pool.Submit(makeBlockedTask);

    bool bothStarted = false;
    {
        std::unique_lock lock(mutex);
        bothStarted = condition.wait_for(lock, std::chrono::seconds(5), [&] { return started == 2; });
    }

    std::atomic<bool> attempting = false;
    std::atomic<bool> submitted = false;
    std::thread submitter([&] {
        attempting.store(true);
        attempting.notify_one();
        pool.Submit([&] {
            return FractalSaveThreadPool::Task([&] {
                std::lock_guard lock(mutex);
                workerIds.insert(std::this_thread::get_id());
            });
        });
        submitted.store(true);
    });
    attempting.wait(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool blockedAtCapacity = !submitted.load();
    const bool noEarlyCompletion = !pool.Cleanup(false);

    {
        std::lock_guard lock(mutex);
        release = true;
    }
    condition.notify_all();
    submitter.join();
    const bool hadWork = pool.Cleanup(true);

    ASSERT_TRUE(bothStarted);
    ASSERT_TRUE(blockedAtCapacity);
    ASSERT_TRUE(noEarlyCompletion);
    ASSERT_TRUE(hadWork);
    ASSERT_EQ(workerIds.size(), size_t{2});
    ASSERT_EQ(started, size_t{2});
}

TEST(FractalSharkLib_SavePoolRechecksMemoryPressure)
{
    std::atomic<uint32_t> memoryLoad = 95;
    FractalSaveThreadPool pool(2, [&] { return memoryLoad.load(); });
    std::mutex mutex;
    std::condition_variable condition;
    bool firstStarted = false;
    bool releaseFirst = false;
    bool secondSubmitted = false;

    pool.Submit([&] {
        return FractalSaveThreadPool::Task([&] {
            std::unique_lock lock(mutex);
            firstStarted = true;
            condition.notify_all();
            condition.wait(lock, [&] { return releaseFirst; });
        });
    });
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return firstStarted; });
    }

    std::thread submitter([&] {
        pool.Submit([] { return FractalSaveThreadPool::Task([] {}); });
        {
            std::lock_guard lock(mutex);
            secondSubmitted = true;
        }
        condition.notify_all();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    bool blockedUnderPressure;
    {
        std::lock_guard lock(mutex);
        blockedUnderPressure = !secondSubmitted;
    }
    memoryLoad.store(50);
    bool resumedAfterPressureDropped;
    {
        std::unique_lock lock(mutex);
        resumedAfterPressureDropped =
            condition.wait_for(lock, std::chrono::seconds(5), [&] { return secondSubmitted; });
        releaseFirst = true;
    }
    condition.notify_all();
    submitter.join();
    pool.Cleanup(true);

    ASSERT_TRUE(blockedUnderPressure);
    ASSERT_TRUE(resumedAfterPressureDropped);
}

TEST(FractalSharkLib_SavePoolReleasesFailedSubmission)
{
    FractalSaveThreadPool pool(1, [] { return uint32_t{0}; });
    ASSERT_THROWS(pool.Submit([]() -> FractalSaveThreadPool::Task {
        throw std::runtime_error("task construction failed");
    }),
                  std::runtime_error);
    ASSERT_FALSE(pool.Cleanup(false));

    std::atomic<bool> completed = false;
    pool.Submit([&] { return FractalSaveThreadPool::Task([&] { completed.store(true); }); });
    ASSERT_TRUE(pool.Cleanup(true));
    ASSERT_TRUE(completed.load());
}

TEST(FractalSharkLib_SavePoolContinuesAfterTaskFailure)
{
    FractalSaveThreadPool pool(1, [] { return uint32_t{0}; });
    pool.Submit([] {
        return FractalSaveThreadPool::Task([] { throw std::runtime_error("save task failed"); });
    });

    std::atomic<bool> completed = false;
    pool.Submit([&] { return FractalSaveThreadPool::Task([&] { completed.store(true); }); });
    ASSERT_TRUE(pool.Cleanup(true));
    ASSERT_TRUE(completed.load());
    ASSERT_FALSE(pool.Cleanup(false));
}

TEST(FractalSharkLib_SavePoolPreservesImageSnapshotsAcrossResize)
{
    const auto directory = std::filesystem::temp_directory_path();
    const auto copiedOutput = directory / "fractalshark-save-pool-copied.png";
    const auto movedOutput = directory / "fractalshark-save-pool-moved.png";
    const auto resizedOutput = directory / "fractalshark-save-pool-resized.png";
    const auto textOutput = directory / "fractalshark-save-pool-iters.txt";
    std::error_code error;
    for (const auto &output : {copiedOutput, movedOutput, resizedOutput, textOutput}) {
        std::filesystem::remove(output, error);
    }

    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(64);
    fractal.CalcFractal(true);

    ASSERT_EQ(fractal.SaveCurrentFractal(copiedOutput.wstring(), true), 0);
    ASSERT_EQ(fractal.SaveCurrentFractal(movedOutput.wstring(), false), 0);
    fractal.CalcFractal(true);
    ASSERT_EQ(fractal.SaveItersAsText(textOutput.wstring()), 0);
    fractal.ResetDimensions(24, 12);
    fractal.CalcFractal(true);
    ASSERT_EQ(fractal.SaveCurrentFractal(resizedOutput.wstring(), true), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    for (const auto &output : {copiedOutput, movedOutput, resizedOutput}) {
        ASSERT_TRUE(std::filesystem::is_regular_file(output));
        WPngImage image;
        ASSERT_TRUE(image.loadImage(output.string()) == WPngImage::kIOStatus_Ok);
        ASSERT_EQ(image.width(), output == resizedOutput ? 24 : 16);
        ASSERT_EQ(image.height(), output == resizedOutput ? 12 : 16);
        ASSERT_EQ(image.originalFileFormat(), WPngImage::kPngFileFormat_RGBA16);
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(textOutput));
    ASSERT_TRUE(std::filesystem::file_size(textOutput) > 0);

    for (const auto &output : {copiedOutput, movedOutput, resizedOutput, textOutput}) {
        std::filesystem::remove(output, error);
    }
    ASSERT_FALSE(static_cast<bool>(error));
}

TEST(FractalSharkLib_RenderToPngRejectsMissingViewSource)
{
    RenderRequest request;
    request.Width = 16;
    request.Height = 16;
    request.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);

    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    std::string error;
    ASSERT_EQ(RenderToPng(request, fractal, &error), 2);
    ASSERT_TRUE(error.find("ViewSource must be set") != std::string::npos);
}

namespace {

void
AppendPngChannel(std::vector<unsigned char> &pixels, uint16_t channel)
{
    pixels.push_back(static_cast<unsigned char>(channel >> 8));
    pixels.push_back(static_cast<unsigned char>(channel & 0xff));
}

void
VerifyPngEncodingRoundTrip(const WPngImage &image, unsigned colorType)
{
    std::vector<unsigned char> expected;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const WPngImage::Pixel16 pixel = image.get16(x, y);
            AppendPngChannel(expected, pixel.r);
            AppendPngChannel(expected, pixel.g);
            AppendPngChannel(expected, pixel.b);
            AppendPngChannel(expected, pixel.a);
        }
    }

    const WPngImage::PngEncodingOptions optimized{false, false, 32};
    std::vector<unsigned char> encoded;
    ASSERT_TRUE(image.SaveImageToRAM(encoded, WPngImage::kPngFileFormat_RGBA16, optimized) ==
                WPngImage::kIOStatus_Ok);
    ASSERT_TRUE(encoded.size() >= 33);
    ASSERT_EQ(encoded[24], 16);
    ASSERT_EQ(encoded[25], colorType);

    std::vector<unsigned char> decoded;
    unsigned width = 0;
    unsigned height = 0;
    ASSERT_EQ(lodepng::decode(decoded, width, height, encoded, LCT_RGBA, 16), 0u);
    ASSERT_EQ(width, static_cast<unsigned>(image.width()));
    ASSERT_EQ(height, static_cast<unsigned>(image.height()));
    ASSERT_TRUE(decoded == expected);

    std::vector<unsigned char> legacy;
    ASSERT_TRUE(image.saveImageToRAM(legacy, WPngImage::kPngFileFormat_RGBA16) ==
                WPngImage::kIOStatus_Ok);
    decoded.clear();
    ASSERT_EQ(lodepng::decode(decoded, width, height, legacy, LCT_RGBA, 16), 0u);
    ASSERT_TRUE(decoded == expected);

    const WPngImage::PngEncodingOptions original{true, true, 128};
    std::vector<unsigned char> explicitOriginal;
    ASSERT_TRUE(image.SaveImageToRAM(explicitOriginal, WPngImage::kPngFileFormat_RGBA16, original) ==
                WPngImage::kIOStatus_Ok);
    ASSERT_TRUE(explicitOriginal == legacy);
}

} // namespace

TEST(FractalSharkLib_PngEncodingPreserves16BitGradientsAndTransparency)
{
    WPngImage opaque(17, 9, WPngImage::Pixel16(0, 0, 0));
    WPngImage transparent(17, 9, WPngImage::Pixel16(0, 0, 0, 0));
    for (int y = 0; y < opaque.height(); ++y) {
        for (int x = 0; x < opaque.width(); ++x) {
            const auto red = static_cast<uint16_t>(0x0102 + x * 257 + y * 13);
            const auto green = static_cast<uint16_t>(0x2345 + x * 19 + y * 511);
            const auto blue = static_cast<uint16_t>(0xabcd - x * 31 - y * 127);
            const auto alpha = static_cast<uint16_t>(x * 4093 + y * 17);
            opaque.set(x, y, WPngImage::Pixel16(red, green, blue));
            transparent.set(x, y, WPngImage::Pixel16(red, green, blue, alpha));
        }
    }
    VerifyPngEncodingRoundTrip(opaque, 2);
    VerifyPngEncodingRoundTrip(transparent, 6);
}

TEST(FractalSharkLib_PngEncodingKeepsRgb16ForSimpleImages)
{
    VerifyPngEncodingRoundTrip(WPngImage(1, 1, WPngImage::Pixel16(0x0102, 0x2345, 0xabcd)), 2);
    VerifyPngEncodingRoundTrip(WPngImage(8, 5, WPngImage::Pixel16(0, 0, 0)), 2);

    WPngImage repeated(32, 8, WPngImage::Pixel16(0x1111, 0x7777, 0xeeee));
    for (int y = 0; y < repeated.height(); ++y) {
        for (int x = 0; x < repeated.width(); ++x) {
            if ((x + y) % 2 == 0) {
                repeated.set(x, y, WPngImage::Pixel16(0x3333, 0xaaaa, 0x5555));
            }
        }
    }
    VerifyPngEncodingRoundTrip(repeated, 2);

    std::vector<unsigned char> legacy;
    ASSERT_TRUE(repeated.saveImageToRAM(legacy, WPngImage::kPngFileFormat_RGBA16) ==
                WPngImage::kIOStatus_Ok);
    ASSERT_TRUE(legacy.size() >= 33);
    ASSERT_TRUE(legacy[24] <= 8);
}
