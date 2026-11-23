#include "ConsoleLog.h"
#include "Environment.h"
#include "Fractal.h"
#include "FractalSharkCli/MovieJob.h"
#include "FractalSharkCli/MovieRunner.h"
#include "FractalSharkCli/MovieVideo.h"
#include "MovieCommandExport.h"
#include "ScopedMpir.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <array>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

namespace {
class MovieTestDirectory {
public:
    MovieTestDirectory() : m_Previous(std::filesystem::current_path())
    {
        static std::atomic<unsigned> sequence{0};
        Path = std::filesystem::temp_directory_path() /
               ("fractalshark movie-" + std::to_string(Environment::CurrentProcessId()) + "-" +
                std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(Path);
    }
    ~MovieTestDirectory()
    {
        std::filesystem::current_path(m_Previous);
        std::error_code ignored;
        std::filesystem::remove_all(Path, ignored);
    }
    std::filesystem::path
    Config(const std::string &extra)
    {
        const auto path = Path / "input.ini";
        FractalSharkCli::WriteMovieText(
            path,
            "[movie]\ncenter-x=-0.75\ncenter-y=0.1\nzoom=16\nzoom-per-frame=2\n"
            "iterations=64\nwidth=16\nheight=16\n" +
                extra);
        return path;
    }
    std::filesystem::path Path;

private:
    std::filesystem::path m_Previous;
};
void
WriteFrame(const std::filesystem::path &path, unsigned width, unsigned height, uint64_t index)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 6);
    for (size_t i = 0; i < pixels.size() / 2; ++i) {
        // Include non-replicated RGB16 samples to exercise nearest rounding.
        const auto value = static_cast<unsigned>((i * 419 + index * 977) & 65535);
        pixels[2 * i] = static_cast<unsigned char>(value >> 8);
        pixels[2 * i + 1] = static_cast<unsigned char>(value);
    }
    std::vector<unsigned char> encoded;
    ASSERT_EQ(lodepng::encode(encoded, pixels, width, height, LCT_RGB, 16), 0u);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(encoded.data()),
                 static_cast<std::streamsize>(encoded.size()));
    output.close();
    ASSERT_TRUE(static_cast<bool>(output));
}
FractalSharkCli::MovieRenderSettings
Settings(const FractalSharkCli::MovieConfig &config)
{
    HighPrecision::defaultPrecisionInBits(1024);
    FractalSharkCli::MovieRenderSettings settings;
    settings.Request.CenterX = HighPrecision(config.Values.at("center-x"));
    settings.Request.CenterY = HighPrecision(config.Values.at("center-y"));
    settings.Request.Zoom = HighPrecision(config.Values.at("zoom"));
    settings.Request.Iterations = 64;
    settings.Request.Width = 16;
    settings.Request.Height = 16;
    settings.Request.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);
    return settings;
}
} // namespace

TEST(MovieExport_CapturesViewAndDefaultsIndependentlyOfPalette)
{
    MPIRPrecision restorePrecision{HighPrecision::defaultPrecisionInBits()};
    MovieTestDirectory directory;
    Fractal fractal(40, 24, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.EnqueueMutation("prepare movie export test", [](Fractal &current) {
        current.SetOfflineView(PointZoomBBConverter(HighPrecision("-0.75"),
                                                    HighPrecision("0.1"),
                                                    HighPrecision("16"),
                                                    PointZoomBBConverter::TestMode::Enabled));
        current.SetNumIterations<uint64_t>(12345);
        current.ResetDimensions(40, 24, 3);
        current.UsePaletteType(FractalPaletteType::Random);
        current.UsePalette(12);
        current.SetPaletteAuxDepth(3);
        current.RotateFractalPalette(19);
    });
    const auto configuration = FractalShark::CaptureMovieConfiguration(fractal);
    const auto output = FractalShark::BuildMovieCommandExport(
        directory.Path / "target.ini", "cli", FractalShark::MovieCommandShell::Posix);
    FractalShark::SaveMovieConfiguration(output, configuration);
    const auto config = FractalSharkCli::MovieConfig::Load(output.ConfigurationPath);
    ASSERT_EQ(config.Values.at("center-x"), fractal.GetPtz().GetPtX().str());
    ASSERT_EQ(config.Values.at("center-y"), fractal.GetPtz().GetPtY().str());
    ASSERT_EQ(config.Values.at("zoom"), fractal.GetPtz().GetZoomFactor().str());
    ASSERT_EQ(config.Values.at("width"), "40");
    ASSERT_EQ(config.Values.at("height"), "24");
    ASSERT_EQ(config.Values.at("antialiasing"), "3");
    ASSERT_EQ(config.Values.at("iterations"), "12345");
    ASSERT_EQ(config.Values.at("render-algorithm"), "AutoSelect");
    ASSERT_EQ(config.Values.at("perturbation-alg"), "Auto");
    ASSERT_EQ(config.Values.at("zoom-per-frame"), "1.01");
    ASSERT_EQ(config.WideIterations, 8192u);
    ASSERT_EQ(config.Video.Fps, 60u);
    ASSERT_EQ(config.Video.GopFrames, 1000u);
    ASSERT_EQ(config.SlopeRatio, 8);
    ASSERT_EQ(config.Values.at("video-profile"), "hevc-high");
    ASSERT_EQ(config.Video.Quality, 16u);
    ASSERT_TRUE(configuration.find("hevc-low") != std::string::npos);
    ASSERT_TRUE(configuration.find("palette") == std::string::npos);
    ASSERT_FALSE(std::filesystem::exists(directory.Path / "target.ini"));
    ASSERT_EQ(output.ConfigurationPath, directory.Path / "target.movie" / "movie.ini");
    ASSERT_TRUE(output.Command.find("--movie-config") == std::string::npos);
    ASSERT_TRUE(output.Command.find("--ffmpeg") == std::string::npos);
    ASSERT_THROWS(FractalShark::SaveMovieConfiguration(output, "replacement"),
                  FractalSharkSeriousException);
    ASSERT_EQ(FractalSharkCli::ReadMovieText(output.ConfigurationPath), configuration);
    ASSERT_EQ(std::distance(std::filesystem::directory_iterator(directory.Path),
                            std::filesystem::directory_iterator()),
              1);
    fractal.EnqueueMutation("change movie export test palette", [](Fractal &current) {
        current.UsePaletteType(FractalPaletteType::Summer);
        current.UsePalette(5);
        current.SetPaletteAuxDepth(0);
    });
    ASSERT_EQ(FractalShark::CaptureMovieConfiguration(fractal), configuration);
}

TEST(MovieExport_PreservesCoordinatesBeyondCommandLineLimits)
{
    MPIRPrecision restorePrecision{HighPrecision::defaultPrecisionInBits()};
    MovieTestDirectory directory;
    Fractal fractal(16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    // More than 32768 coordinate digits exercises the Windows command-line
    // boundary. Only file paths, never these digits, belong in the command.
    HighPrecision::defaultPrecisionInBits(160000);
    const HighPrecision center("0." + std::string(40000, '1'));
    const PointZoomBBConverter view(center,
                                    HighPrecision("0.125"),
                                    HighPrecision("1e40000"),
                                    PointZoomBBConverter::TestMode::Disabled);
    fractal.SetOfflineView(view);
    const auto output = FractalShark::BuildMovieCommandExport(
        directory.Path / "deep.ini", "cli", FractalShark::MovieCommandShell::PowerShell);
    const auto configuration = FractalShark::CaptureMovieConfiguration(fractal);
    FractalShark::SaveMovieConfiguration(output, configuration);
    const auto config = FractalSharkCli::MovieConfig::Load(output.ConfigurationPath);
    ASSERT_TRUE(config.Values.at("center-x").size() > 32768);
    ASSERT_EQ(config.Values.at("center-x"), fractal.GetPtz().GetPtX().str());
    ASSERT_EQ(config.Values.at("zoom"), fractal.GetPtz().GetZoomFactor().str());
    ASSERT_TRUE(output.Command.size() < 32768);
    ASSERT_TRUE(output.Command.find("--center-x") == std::string::npos);
}

TEST(MovieExport_QuotesBothShellsAndRejectsMultilinePaths)
{
    MovieTestDirectory directory;
    const auto ini = directory.Path / "named 'movie' $(literal).ini";
    const auto cli = directory.Path / "CLI with spaces";
    const auto powershell =
        FractalShark::BuildMovieCommandExport(ini, cli, FractalShark::MovieCommandShell::PowerShell);
    const auto posix =
        FractalShark::BuildMovieCommandExport(ini, cli, FractalShark::MovieCommandShell::Posix);
    ASSERT_TRUE(powershell.Command.starts_with("& '"));
    ASSERT_TRUE(powershell.Command.find("named ''movie'' $(literal).movie'") != std::string::npos);
    ASSERT_TRUE(posix.Command.starts_with("'"));
    ASSERT_TRUE(posix.Command.find("named '\\''movie'\\'' $(literal).movie'") != std::string::npos);
    const auto unicode = FractalShark::BuildMovieCommandExport(
        directory.Path / std::filesystem::path(u8"movie \u03b2.ini"),
        cli,
        FractalShark::MovieCommandShell::PowerShell);
    ASSERT_TRUE(unicode.Command.find("movie \xce\xb2.movie'") != std::string::npos);
    ASSERT_EQ(powershell.JobDirectory, directory.Path / "named 'movie' $(literal).movie");
    ASSERT_TRUE(FractalShark::DescribeMovieCommandExport(powershell, false).find(powershell.Command) !=
                std::string::npos);
    ASSERT_TRUE(FractalShark::DescribeMovieCommandExport(powershell, false).find("manually") !=
                std::string::npos);
    ASSERT_THROWS(FractalShark::BuildMovieCommandExport(
                      directory.Path / "bad\nname.ini", cli, FractalShark::MovieCommandShell::Posix),
                  FractalSharkSeriousException);
    ASSERT_THROWS(
        FractalShark::BuildMovieCommandExport({}, cli, FractalShark::MovieCommandShell::PowerShell),
        FractalSharkSeriousException);
}

TEST(MovieExport_DiscoversInstalledAndCMakeCliPaths)
{
    MovieTestDirectory directory;
    const auto gui = directory.Path / "FractalSharkGuiLinux" / "FractalSharkGuiLinux";
    std::filesystem::create_directory(gui.parent_path());
    ASSERT_FALSE(FractalShark::FindMovieCliExecutable(gui, FractalShark::MovieCommandShell::Posix));
    const auto target = directory.Path / "FractalSharkCli" / "FractalSharkCli";
    std::filesystem::create_directory(target.parent_path());
    FractalSharkCli::WriteMovieText(target, "executable fixture");
    const auto cmakeCli =
        FractalShark::FindMovieCliExecutable(gui, FractalShark::MovieCommandShell::Posix);
    ASSERT_TRUE(cmakeCli.has_value());
    ASSERT_EQ(*cmakeCli, target);
    const auto sibling = gui.parent_path() / "FractalSharkCli.exe";
    FractalSharkCli::WriteMovieText(sibling, "executable fixture");
    const auto installedCli =
        FractalShark::FindMovieCliExecutable(gui, FractalShark::MovieCommandShell::PowerShell);
    ASSERT_TRUE(installedCli.has_value());
    ASSERT_EQ(*installedCli, sibling);
    ASSERT_TRUE(Environment::ExecutablePath().is_absolute());
    ASSERT_TRUE(std::filesystem::is_regular_file(Environment::ExecutablePath()));
}

TEST(MovieExport_ValidatesTargetsAndReportsPublicationFailure)
{
    MovieTestDirectory directory;
    Fractal fractal(16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.EnqueueMutation("set invalid movie target", [](Fractal &current) {
        current.SetOfflineView(PointZoomBBConverter(HighPrecision(0),
                                                    HighPrecision(0),
                                                    HighPrecision("0.5"),
                                                    PointZoomBBConverter::TestMode::Enabled));
    });
    ASSERT_THROWS(FractalShark::CaptureMovieConfiguration(fractal), FractalSharkSeriousException);
    // A capture validation error must not terminate the render worker.
    fractal.EnqueueMutation("set valid movie target", [](Fractal &current) {
        current.SetOfflineView(PointZoomBBConverter(HighPrecision(0),
                                                    HighPrecision(0),
                                                    HighPrecision(1),
                                                    PointZoomBBConverter::TestMode::Enabled));
    });
    const auto configuration = FractalShark::CaptureMovieConfiguration(fractal);
    const auto output = FractalShark::BuildMovieCommandExport(
        directory.Path / "missing" / "target.ini", "cli", FractalShark::MovieCommandShell::Posix);
    ASSERT_THROWS(FractalShark::SaveMovieConfiguration(output, configuration),
                  FractalSharkSeriousException);
    ASSERT_FALSE(std::filesystem::exists(output.ConfigurationPath));
    ASSERT_FALSE(std::filesystem::exists(output.JobDirectory));
}

TEST(MovieExport_RendersAndResumesFromExportedConfiguration)
{
    MPIRPrecision restorePrecision{HighPrecision::defaultPrecisionInBits()};
    MovieTestDirectory directory;
    std::string configuration;
    {
        Fractal fractal(16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
        fractal.EnqueueMutation("prepare single-frame movie export", [](Fractal &current) {
            current.SetOfflineView(PointZoomBBConverter(HighPrecision("-0.75"),
                                                        HighPrecision("0.1"),
                                                        HighPrecision(1),
                                                        PointZoomBBConverter::TestMode::Enabled));
            current.SetNumIterations<uint64_t>(64);
        });
        configuration = FractalShark::CaptureMovieConfiguration(fractal);
    }
    const auto output = FractalShark::BuildMovieCommandExport(
        directory.Path / "target.ini", "cli", FractalShark::MovieCommandShell::Posix);
    FractalShark::SaveMovieConfiguration(output, configuration);
    const auto profile = configuration.find("video-profile=hevc-high");
    ASSERT_TRUE(profile != std::string::npos);
    configuration.replace(
        profile, std::string_view("video-profile=hevc-high").size(), "video-profile=png-only");
    FractalSharkCli::WriteMovieText(output.ConfigurationPath, configuration);
    ASSERT_EQ(
        FractalSharkCli::RunMovie(output.JobDirectory, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    ASSERT_EQ(
        FractalSharkCli::RunMovie(output.JobDirectory, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    FractalSharkCli::MovieJob job(output.JobDirectory, {});
    ASSERT_EQ(job.Config().Values.at("wide-iterations"), "64");
    ASSERT_EQ(job.Remaining(), 0u);
    ASSERT_TRUE(std::filesystem::exists(job.FramePath(0)));
    ASSERT_FALSE(job.Config().Values.contains("palette-map"));
}

#ifndef _WIN32
TEST(MovieExport_ClipboardFailuresPreserveManualCopyRecovery)
{
    struct ClipboardRegistration {
        bool Throw = false;
        bool Accept = false;
        std::string Text;

        static bool
        Set(void *context, std::string_view text)
        {
            auto &registration = *static_cast<ClipboardRegistration *>(context);
            if (registration.Throw) {
                throw FractalSharkSeriousException("Simulated clipboard ownership failure");
            }
            registration.Text = text;
            return registration.Accept;
        }
        ClipboardRegistration() { Environment::RegisterClipboardTextSetter(Set, this); }
        ~ClipboardRegistration() { Environment::UnregisterClipboardTextSetter(Set, this); }
    } clipboard;
    MovieTestDirectory directory;
    const auto output = FractalShark::BuildMovieCommandExport(
        directory.Path / "clipboard.ini", "cli", FractalShark::MovieCommandShell::Posix);
    const std::string configuration =
        "[movie]\ncenter-x=0\ncenter-y=0\nzoom=1\nzoom-per-frame=1.01\niterations=64\n";
    FractalShark::SaveMovieConfiguration(output, configuration);
    ASSERT_FALSE(FractalShark::CopyMovieCommandToClipboard(output));
    clipboard.Throw = true;
    ASSERT_FALSE(FractalShark::CopyMovieCommandToClipboard(output));
    ASSERT_EQ(FractalSharkCli::ReadMovieText(output.ConfigurationPath), configuration);
    clipboard.Throw = false;
    clipboard.Accept = true;
    ASSERT_TRUE(FractalShark::CopyMovieCommandToClipboard(output));
    ASSERT_EQ(clipboard.Text, output.Command);
}
#endif

TEST(MovieSchedule_EndpointsCurveAndHugeExponents)
{
    HighPrecision::defaultPrecisionInBits(1024);
    FractalSharkCli::MovieSchedule schedule(HighPrecision("16"), "2", 10, 110, 2, 4, true);
    ASSERT_EQ(schedule.FrameCount(), 5u);
    ASSERT_NEAR(static_cast<double>(schedule.ZoomAt(4)), 16, 0);
    ASSERT_NEAR(static_cast<double>(schedule.ZoomAt(0)), 1, 0);
    ASSERT_EQ(schedule.IterationsAt(4), 110u);
    ASSERT_EQ(schedule.IterationsAt(2), 30u);
    ASSERT_EQ(schedule.IterationsAt(1), 20u);
    ASSERT_EQ(schedule.IterationsAt(0), 10u);
    FractalSharkCli::MovieSchedule partial(HighPrecision("15"), "2", 10, 100, 2, 8, true);
    ASSERT_EQ(partial.FrameCount(), 5u);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(4)), 15, 0);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(3)), 8, 0);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(0)), 1, 0);
    HighPrecision huge{HighPrecision::SetPrecision::True, 256};
    mpf_set_ui(huge.backend(), 1);
    mpf_mul_2exp(huge.backend(), huge.backend(), 10'000'000);
    FractalSharkCli::MovieSchedule deep(huge, "2", 10, 1000, 30, 8, false);
    ASSERT_EQ(deep.FrameCount(), 10'000'001u);
    ASSERT_NEAR(FractalSharkCli::MovieSchedule::Log2(deep.ZoomAt(1)), 9'999'999, 1e-8L);
    ASSERT_EQ(deep.IterationsAt(10'000'000), 10u);
    ASSERT_EQ(deep.IterationsAt(0), 1000u);
    ASSERT_THROWS(FractalSharkCli::MovieSchedule(
                      HighPrecision("16"), "1.00000000000000000001", 10, 100, 2, 8, false),
                  FractalSharkSeriousException);
    FractalSharkCli::MovieSchedule single(HighPrecision("1"), "2", 10, 100, 30, 8, false);
    ASSERT_EQ(single.FrameCount(), 1u);
    ASSERT_EQ(single.IterationsAt(0), 100u);
    FractalSharkCli::MovieSchedule nearOne(
        HighPrecision("1.00000000000000000001"), "1.000000000000000000001", 10, 100, 0, 8, false);
    ASSERT_EQ(nearOne.FrameCount(), 11u);
    ASSERT_TRUE(nearOne.DepthAt(0) > 0);
    FractalSharkCli::MovieSchedule steep(
        HighPrecision("16"), "2", 10, 100, 2, std::numeric_limits<double>::max(), false);
    ASSERT_TRUE(steep.IterationsAt(1) >= steep.IterationsAt(3));
}

TEST(MovieJob_RejectsUnsupportedFrameCounts)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-gop-frames=3000\n");
    FractalSharkCli::MovieJob job(directory.Path / "job", config);
    ASSERT_THROWS(
        job.InitializeProgress(static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1, true, 0),
        FractalSharkSeriousException);
    ASSERT_FALSE(std::filesystem::exists(job.Directory() / "profile.txt"));
    job.InitializeProgress(10001, true, 0);
    ASSERT_EQ(job.Remaining(), 10001u);
    ASSERT_EQ(job.FramePath(8999).parent_path().filename().string(), "chunk_0000000002");
    ASSERT_EQ(job.FramePath(9000).parent_path().filename().string(), "chunk_0000000003");
    ASSERT_THROWS(
        FractalSharkCli::MovieSchedule(HighPrecision("16"), "1.000000001", 10, 100, 2, 8, false),
        FractalSharkSeriousException);
}

TEST(MoviePng_Rgb16RoundingAndDimensionValidation)
{
    MovieTestDirectory directory;
    const auto path = directory.Path / "frame.png";
    WriteFrame(path, 16, 16, 7);
    const auto pixels = FractalSharkCli::ReadMovieRgb8(path, 16, 16);
    for (size_t i = 0; i < pixels.size(); ++i) {
        const auto value = (i * 419 + 7 * 977) & 65535;
        ASSERT_EQ(pixels[i], (value + 128) / 257);
    }
    ASSERT_THROWS(FractalSharkCli::ReadMovieRgb8(path, 8, 32), FractalSharkSeriousException);
    FractalSharkCli::WriteMovieText(path, "broken png");
    ASSERT_THROWS(FractalSharkCli::ReadMovieRgb8(path, 16, 16), FractalSharkSeriousException);
}

TEST(MovieRunner_CreateResumeAndRecoverMissingFrame)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-gop-frames=2\nvideo-profile=png-only\n");
    const auto job = directory.Path / "job";
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    std::filesystem::path frame;
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 0);
        ASSERT_EQ(state.Remaining(), 0u);
        ASSERT_EQ(state.Config().Values.at("video-profile"), "png-only");
        frame = state.FramePath(3);
        std::filesystem::remove(frame);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    ASSERT_TRUE(std::filesystem::exists(frame));
    // Recover a malformed orphan after rewinding the newest progress record.
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 0);
        state.Rewind(3);
        FractalSharkCli::WriteMovieText(state.FramePath(3), "incomplete PNG");
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, GpuMode::Disabled, UINT64_MAX, Settings), 0);
}

TEST(MovieRunner_NativeReferenceReuseAfterRestart)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-gop-frames=2\nvideo-profile=png-only\n");
    auto contents = FractalSharkCli::ReadMovieText(config);
    contents.replace(contents.find("zoom=16"), 7, "zoom=1e40");
    contents.replace(contents.find("zoom-per-frame=2"), 16, "zoom-per-frame=1e20");
    FractalSharkCli::WriteMovieText(config, contents);
    const auto job = directory.Path / "job";
    const auto referenceSettings = [](const FractalSharkCli::MovieConfig &settings) {
        auto result = Settings(settings);
        result.Request.Algorithm =
            GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
        result.Request.Perturbation = RefOrbitCalc::PerturbationAlg::ST;
        return result;
    };
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, GpuMode::Disabled, UINT64_MAX, referenceSettings),
              0);
    const auto countMetadata = [&] {
        size_t count = 0;
        for (const auto &file : std::filesystem::directory_iterator(job)) {
            count += file.path().extension() == ".met" ? 1 : 0;
        }
        return count;
    };
    const auto count = countMetadata();
    ASSERT_TRUE(count > 0);
    std::string damagedMetadata;
    for (const auto &entry : std::filesystem::directory_iterator(job)) {
        if (entry.path().extension() == ".met") {
            damagedMetadata = FractalSharkCli::ReadMovieText(entry.path());
            break;
        }
    }
    const auto damagedBegin = damagedMetadata.find("MaxIterationsPerPixel: ");
    ASSERT_TRUE(damagedBegin != std::string::npos);
    damagedMetadata.replace(damagedBegin,
                            damagedMetadata.find('\n', damagedBegin) - damagedBegin,
                            "MaxIterationsPerPixel: incomplete");
    FractalSharkCli::WriteMovieText(job / "incomplete.met", damagedMetadata);
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(3, true, 0);
        std::filesystem::remove(state.FramePath(2));
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, GpuMode::Disabled, UINT64_MAX, referenceSettings), 0);
    ASSERT_EQ(countMetadata(), count + 1);
}

TEST(MovieRunner_OrderlyStopAndResume)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-gop-frames=2\nvideo-profile=png-only\n");
    const auto job = directory.Path / "job";
    const auto stopDuringSetup = [](const FractalSharkCli::MovieConfig &settings) {
        // RunMovie installs its signal handler before invoking the factory.
        // Request a stop synchronously, avoiding timing-dependent thread races.
        std::raise(SIGINT);
        return Settings(settings);
    };
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, GpuMode::Disabled, UINT64_MAX, stopDuringSetup),
              130);
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 4);
        ASSERT_EQ(state.Remaining(), 5u);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
}

TEST(MovieVideo_TruncatesUncommittedTailAndPublishes)
{
    MovieTestDirectory directory;
    const auto configuration = directory.Config("video-gop-frames=2\n");
    const auto jobPath = directory.Path / "job";
    const auto partial = jobPath / "movie.ts.partial";
    const auto target = jobPath / "movie.ts";
    // Fake Annex B payload. This test exercises transport and commit logic on
    // hosted runners without loading CUDA or NVENC.
    const std::array<unsigned char, 300> unit{0, 0, 1, 0x26};
    uint64_t durableBytes;
    {
        FractalSharkCli::MovieJob job(jobPath, configuration);
        job.InitializeProgress(3, true, 0);
        ASSERT_THROWS(FractalSharkCli::MovieJob(jobPath, configuration), std::system_error);
        FractalSharkCli::MovieVideo video(job);
        ASSERT_THROWS(video.Append(0, unit, false), FractalSharkSeriousException);
        video.Append(0, unit, true);
        video.Finish();
        ASSERT_EQ(job.NextFrame(), 1u);
        durableBytes = job.VideoBytes();
        ASSERT_EQ(durableBytes % 188, 0u);
        ASSERT_FALSE(std::filesystem::exists(target));
        video.Append(1, unit, false);
    }
    ASSERT_TRUE(std::filesystem::file_size(partial) > durableBytes);
    {
        FractalSharkCli::MovieJob job(jobPath, {});
        ASSERT_EQ(job.NextFrame(), 1u);
        FractalSharkCli::MovieVideo video(job);
        ASSERT_EQ(std::filesystem::file_size(partial), durableBytes);
        ASSERT_THROWS(video.Append(1, unit, false), FractalSharkSeriousException);
        video.Append(1, unit, true);
        video.Append(2, unit, true);
        video.Finish();
        ASSERT_EQ(job.Remaining(), 0u);
        ASSERT_EQ(std::filesystem::file_size(target), job.VideoBytes());
        ASSERT_FALSE(std::filesystem::exists(partial));
    }
    FractalSharkCli::MovieJob job(jobPath, {});
    FractalSharkCli::MovieVideo(job).Finish();
    ASSERT_EQ(job.NextFrame(), 3u);
    const auto transport = FractalSharkCli::ReadMovieText(target);
    unsigned accessUnits = 0;
    for (size_t offset = 0; offset < transport.size(); offset += 188) {
        const auto *packet = reinterpret_cast<const unsigned char *>(transport.data() + offset);
        ASSERT_EQ(packet[0], 0x47u);
        if ((packet[1] & 0x5f) == 0x41 && packet[2] == 0) {
            ++accessUnits;
        }
    }
    ASSERT_EQ(accessUnits, 3u);
}
