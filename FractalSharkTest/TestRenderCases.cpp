#include "RenderTestCatalog.h"
#include "RenderTestSupport.h"
#include "RenderThreadPool.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace RenderTests {
namespace {
void
Configure(Fractal &fractal, const RenderCase &test, size_t view)
{
    fractal.GetRenderPool()->Drain();
    fractal.View(view, false);
    fractal.SetIterType(test.Bits);
    if (test.IterationLimit != 0) {
        fractal.SetNumIterations<uint64_t>(test.IterationLimit);
    }
    fractal.ResetDimensions(MatrixDimension(), MatrixDimension(), test.Antialiasing);
    fractal.SetIterationPrecision(test.Step);
    fractal.SetCompressionErrorExp(Fractal::CompressionError::Low, test.Compression);
    fractal.SetResultsAutosave(test.Storage);
    fractal.GetLAParameters().SetDefaults(test.LA);
    fractal.GetLAParameters().SetThreading(test.Threading);
    fractal.SetPerturbationAlg(test.Reference);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(test.Algorithm));
    ASSERT_TRUE(fractal.GetIterType() == test.Bits);
    ASSERT_EQ(fractal.GetGpuAntialiasing(), test.Antialiasing);
    ASSERT_EQ(fractal.GetIterationPrecision(), test.Step);
    ASSERT_EQ(fractal.GetCompressionErrorExp(Fractal::CompressionError::Low), test.Compression);
    if (test.Algorithm.Algorithm != RenderAlgorithmEnum::AUTO) {
        ASSERT_TRUE(fractal.GetRenderAlgorithm().Algorithm == test.Algorithm.Algorithm);
    } else {
        const auto zoom = fractal.GetZoomFactor();
        const auto expected =
            test.AutoGpu ? (zoom < HighPrecision{1e4}    ? RenderAlgorithmEnum::Gpu1x32
                            : zoom < HighPrecision{1e9}  ? RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO
                            : zoom < HighPrecision{1e34} ? RenderAlgorithmEnum::Gpu1x32PerturbedLAv2
                                                         : RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2)
                         : (zoom < HighPrecision{1e9}    ? RenderAlgorithmEnum::Cpu64
                            : zoom < HighPrecision{1e34} ? RenderAlgorithmEnum::Cpu64PerturbedBLA
                                                         : RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
        ASSERT_TRUE(fractal.GetRenderAlgorithm().Algorithm == expected);
    }
    fractal.ForceRecalc();
}

bool
Render(Fractal &fractal, std::string_view id)
{
    fractal.GetRenderPool()->Drain();
    fractal.CalcFractal(true);
    ASSERT_FALSE(fractal.GetStopCalculating());
    return SaveAndCheck(fractal, id);
}

void
ReferenceRoundtrip(Fractal &fractal, const RenderCase &test)
{
    Render(fractal, test.Name + "_Original");
    const auto expectedIterations = fractal.GetNumIterations<uint64_t>();
    const auto save = [&](CompressToDisk compression, std::string_view suffix) {
        const auto path = std::filesystem::current_path() / ("orbit-" + std::string{suffix});
        fractal.SaveRefOrbit(compression, path.wstring());
        ASSERT_TRUE(std::filesystem::is_regular_file(path));
        ASSERT_TRUE(std::filesystem::file_size(path) != 0);
        return path;
    };
    if (test.View == 5) {
        save(CompressToDisk::Disable, "plain.txt");
        if (std::string_view{test.Algorithm.AlgorithmStr}.find("2x32") == std::string_view::npos) {
            save(CompressToDisk::SimpleCompression, "simple.txt");
        }
    }
    save(CompressToDisk::MaxCompression, "max.txt");
    const auto saved = save(CompressToDisk::MaxCompressionImagina, "max.im");

    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    Configure(fractal, test, 0);
    Render(fractal, test.Name + "_Reset");
    if (test.LoadSettings == ImaginaSettings::UseSaved) {
        ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO)));
    }
    fractal.LoadRefOrbit(
        nullptr, CompressToDisk::MaxCompressionImagina, test.LoadSettings, saved.wstring());
    ASSERT_EQ(fractal.GetNumIterations<uint64_t>(), expectedIterations);
    if (test.Algorithm.Gpu == RequiresGpu::No) {
        ASSERT_TRUE(fractal.SetRenderAlgorithm(test.Algorithm));
    }
    fractal.SetIterType(test.Bits);
    fractal.ResetDimensions(MatrixDimension(), MatrixDimension(), test.Antialiasing);
    fractal.ForceRecalc();
    Render(fractal, test.Name + "_Decompressed");
}

void
Imagina(Fractal &fractal, const RenderCase &test)
{
    const auto fixture = Environment::FindEmbeddedImaginaFixture(test.Fixture);
    ASSERT_TRUE(fixture.has_value());
    const auto path = std::filesystem::current_path() / "fixture.im";
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char *>(fixture->bytes.data()), fixture->bytes.size());
        ASSERT_TRUE(output.good());
    }
    bool hasPreset = false;
    for (const auto &info : Environment::GetEmbeddedImaginaFixtureInfos()) {
        if (info.name == test.Fixture) {
            hasPreset = info.presetView.has_value();
        }
    }
    if (hasPreset) {
        Render(fractal, test.Name + "_Original");
    }
    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO)));
    fractal.LoadRefOrbit(
        nullptr, CompressToDisk::MaxCompressionImagina, ImaginaSettings::UseSaved, path.wstring());
    // Pin rendering independently of GPU availability, while checking the saved location and orbit.
    ASSERT_TRUE(fractal.SetRenderAlgorithm(test.Algorithm));
    fractal.SetIterType(test.Bits);
    fractal.ResetDimensions(MatrixDimension(), MatrixDimension(), test.Antialiasing);
    fractal.ForceRecalc();
    Render(fractal, test.Name + "_Imagina");
}
} // namespace

void
Execute(const RenderCase &test)
{
    ASSERT_TRUE(test.DisabledReason.empty());
    ScopedDirectory directory{test.Name};
    const bool useGpu = test.AutoGpu ||
                        (test.Algorithm.Algorithm != RenderAlgorithmEnum::AUTO &&
                         test.Algorithm.Gpu == RequiresGpu::Yes) ||
                        test.Reference == RefOrbitCalc::PerturbationAlg::GPU;
    Fractal fractal{static_cast<int>(MatrixDimension()),
                    static_cast<int>(MatrixDimension()),
                    nullptr,
                    false,
                    UINT64_MAX,
                    true,
                    useGpu ? GpuMode::Auto : GpuMode::Disabled};
    ASSERT_FALSE(useGpu && fractal.GpuBypassed());
    fractal.GetRenderPool()->Drain();
    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    Configure(fractal, test, test.View);
    switch (test.Kind) {
        case Scenario::ReferenceSave:
        case Scenario::Compression:
            ReferenceRoundtrip(fractal, test);
            break;
        case Scenario::Imagina:
            Imagina(fractal, test);
            break;
        case Scenario::Reuse:
            if (test.Algorithm.Algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedLAv2 ||
                test.Algorithm.Algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2) {
                // Both deep views exceed the range of these two original non-HDR selections.
                // Check the explicit renderer's range guard, then exercise reuse with its HDR variant.
                ASSERT_THROWS(fractal.CalcFractal(true), std::range_error);
                auto hdrTest = test;
                hdrTest.Algorithm = GetRenderAlgorithmTupleEntry(
                    test.Algorithm.RequiresCompression ? RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2
                                                       : RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2);
                Configure(fractal, hdrTest, test.View);
                Render(fractal, test.Name + "_HdrOriginal");
                Configure(fractal, hdrTest, 12);
                Render(fractal, test.Name + "_HdrPerturbed");
                break;
            }
            Render(fractal, test.Name + "_Original");
            Configure(fractal, test, 12);
            Render(fractal, test.Name + "_Perturbed");
            break;
        default:
            Render(fractal, test.Name);
            if (test.Kind == Scenario::ReferenceBackend &&
                test.Reference >= RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed &&
                test.Reference <= RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3) {
                RefOrbitDetails originalDetails;
                fractal.GetSomeDetails(originalDetails);
                ASSERT_TRUE(originalDetails.CompressedIntermediateIters > 0);
                const auto originalZoom = fractal.GetZoomFactor();
                fractal.PanByFraction(0.25, 0.25);
                // ZoomAtCenter uses additive edge scaling: -0.375 shrinks the box by four.
                fractal.ZoomAtCenter(-0.375);
                ASSERT_TRUE(fractal.GetZoomFactor() > originalZoom);
                // Bound this additional reuse probe; the original preset is rendered above in full.
                fractal.SetNumIterations<uint64_t>(500'000);
                fractal.ForceRecalc();
                ASSERT_TRUE(Render(fractal, test.Name + "_ReusedAfterZoom"));
                RefOrbitDetails reusedDetails;
                fractal.GetSomeDetails(reusedDetails);
                ASSERT_TRUE(reusedDetails.ExtraIntermediatePrecision > 0);
            }
            if (test.Storage == AddPointOptions::EnableWithoutSave) {
                fractal.SavePerturbationOrbits();
            }
            break;
    }
}
} // namespace RenderTests
