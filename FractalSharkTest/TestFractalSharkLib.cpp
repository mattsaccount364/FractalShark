#include "Exceptions.h"
#include "FeatureSummary.h"
#include "Fractal.h"
#include "FractalPalette.h"
#include "GuiFileOperations.h"
#include "ItersMemoryContainer.h"
#include "OrbitEndpointEvaluator.h"
#include "OrbitParameterPack.h"
#include "RecommendedSettings.h"
#include "RenderAlgorithm.h"
#include "RenderToConsole.h"
#include "RenderToPng.h"
#include "TestFramework.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

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

TEST(FractalSharkLib_AsyncPngSaveReclaimsWorkerBeforeCleanup)
{
    const auto output = std::filesystem::temp_directory_path() / "fractalshark-async-save-test.png";
    std::error_code error;
    std::filesystem::remove(output, error);

    Fractal fractal(
        16, 16, nullptr, false, std::numeric_limits<uint64_t>::max(), true, GpuMode::Disabled);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(64);
    fractal.CalcFractal(true);

    ASSERT_EQ(fractal.SaveCurrentFractal(output.wstring(), true), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_TRUE(std::filesystem::is_regular_file(output));

    std::filesystem::remove(output, error);
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
