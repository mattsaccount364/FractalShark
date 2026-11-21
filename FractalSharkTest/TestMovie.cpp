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

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

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
FractalSharkCli::MovieRenderSettings
VideoSettings(const FractalSharkCli::MovieConfig &config)
{
    auto settings = Settings(config);
    settings.Request.Width = 256;
    settings.Request.Height = 256;
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
        directory.Path / "target.ini", "cli", "ffmpeg", FractalShark::MovieCommandShell::Posix);
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
    ASSERT_EQ(config.Fps, 60u);
    ASSERT_EQ(config.ChunkFrames, 1000u);
    ASSERT_EQ(config.SlopeRatio, 8);
    ASSERT_EQ(config.Values.at("video-encoder-args"),
              "-c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 16 -profile:v main -pix_fmt nv12 "
              "-colorspace bt709 -color_range tv");
    ASSERT_TRUE(configuration.find("-rc constqp -qp 22 -profile:v main -pix_fmt nv12") !=
                std::string::npos);
    ASSERT_TRUE(configuration.find("palette") == std::string::npos);
    ASSERT_FALSE(std::filesystem::exists(directory.Path / "target.ini"));
    ASSERT_EQ(output.ConfigurationPath, directory.Path / "target.movie" / "movie.ini");
    ASSERT_EQ(FractalSharkCli::ReadMovieText(output.JobDirectory / "ffmpeg.txt"),
              Environment::WideToUtf8(Environment::FromFsPath(output.FfmpegExecutable)));
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
        directory.Path / "deep.ini", "cli", "ffmpeg", FractalShark::MovieCommandShell::PowerShell);
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
    const auto ffmpeg = directory.Path / "ffmpeg & encoder";
    const auto powershell = FractalShark::BuildMovieCommandExport(
        ini, cli, ffmpeg, FractalShark::MovieCommandShell::PowerShell);
    const auto posix =
        FractalShark::BuildMovieCommandExport(ini, cli, ffmpeg, FractalShark::MovieCommandShell::Posix);
    ASSERT_TRUE(powershell.Command.starts_with("& '"));
    ASSERT_TRUE(powershell.Command.find("named ''movie'' $(literal).movie'") != std::string::npos);
    ASSERT_TRUE(posix.Command.starts_with("'"));
    ASSERT_TRUE(posix.Command.find("named '\\''movie'\\'' $(literal).movie'") != std::string::npos);
    const auto unicode = FractalShark::BuildMovieCommandExport(
        directory.Path / std::filesystem::path(u8"movie \u03b2.ini"),
        cli,
        ffmpeg,
        FractalShark::MovieCommandShell::PowerShell);
    ASSERT_TRUE(unicode.Command.find("movie \xce\xb2.movie'") != std::string::npos);
    ASSERT_EQ(powershell.JobDirectory, directory.Path / "named 'movie' $(literal).movie");
    ASSERT_TRUE(FractalShark::DescribeMovieCommandExport(powershell, false).find(powershell.Command) !=
                std::string::npos);
    ASSERT_TRUE(FractalShark::DescribeMovieCommandExport(powershell, false).find("manually") !=
                std::string::npos);
    ASSERT_THROWS(
        FractalShark::BuildMovieCommandExport(
            directory.Path / "bad\nname.ini", cli, ffmpeg, FractalShark::MovieCommandShell::Posix),
        FractalSharkSeriousException);
    ASSERT_THROWS(FractalShark::BuildMovieCommandExport(
                      {}, cli, ffmpeg, FractalShark::MovieCommandShell::PowerShell),
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
    const auto output = FractalShark::BuildMovieCommandExport(directory.Path / "missing" / "target.ini",
                                                              "cli",
                                                              "ffmpeg",
                                                              FractalShark::MovieCommandShell::Posix);
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
        directory.Path / "target.ini", "cli", "ffmpeg", FractalShark::MovieCommandShell::Posix);
    FractalShark::SaveMovieConfiguration(output, configuration);
    // This fixture exercises PNG-only rendering. Codec integration below also
    // exercises an exported job's saved encoder without any command overrides.
    std::filesystem::remove(output.JobDirectory / "ffmpeg.txt");
    ASSERT_EQ(
        FractalSharkCli::RunMovie(output.JobDirectory, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings),
        0);
    ASSERT_EQ(
        FractalSharkCli::RunMovie(output.JobDirectory, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings),
        0);
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
        directory.Path / "clipboard.ini", "cli", "ffmpeg", FractalShark::MovieCommandShell::Posix);
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

TEST(MovieJob_CheckpointsLockAndChunkDeletionOrder)
{
    MovieTestDirectory directory;
    const auto config =
        directory.Config("video-chunk-frames=2\nvideo-encoder-args=" +
                         std::string(FractalShark::LosslessMovieEncoderArguments()) + "\n");
    const auto jobPath = directory.Path / "job";
    {
        FractalSharkCli::MovieJob job(jobPath, config);
        job.InitializeProgress(5, true, 30);
        ASSERT_EQ(job.Remaining(), 5u);
        ASSERT_THROWS(FractalSharkCli::MovieJob(jobPath, config), std::system_error);
        WriteFrame(job.FramePath(0), 16, 16, 4);
        job.CompleteFrame(0);
        WriteFrame(job.FramePath(1), 16, 16, 3);
        job.CompleteFrame(1);
        ASSERT_THROWS(job.CompleteFrame(0), FractalSharkSeriousException);
        FractalSharkCli::MovieChunk chunk{0, 2, 100, 3};
        ASSERT_THROWS(job.DeleteChunkPngs(chunk), FractalSharkSeriousException);
        ASSERT_TRUE(std::filesystem::exists(job.FramePath(0)));
        FractalSharkCli::WriteMovieText(job.VideoPath(0), "mkv");
        job.CommitChunk(chunk);
        job.DeleteChunkPngs(chunk);
        job.DeleteChunkPngs(chunk);
        ASSERT_FALSE(std::filesystem::exists(job.FramePath(0).parent_path()));
        WriteFrame(job.FramePath(2), 16, 16, 2);
    }
    const auto unfinishedPng = jobPath / "frames" / ".fractalshark-save-old.tmp";
    const auto unfinishedVideo = jobPath / "chunks" / ".fractalshark-video-old.tmp";
    const auto unfinishedInputs = jobPath / "chunks" / ".fractalshark-video-inputs-old.tmp";
    FractalSharkCli::WriteMovieText(unfinishedPng, "unfinished PNG");
    FractalSharkCli::WriteMovieText(unfinishedVideo, "unfinished movie");
    std::filesystem::create_directory(unfinishedInputs);
    const auto retainedPng = jobPath / "frames" / "chunk_0000000001" / "frame_0000000002.png";
    std::filesystem::create_hard_link(retainedPng, unfinishedInputs / "frame.png");
    {
        // An external INI without an encoder setting inherits the frozen profile.
        const auto omittedEncoder = directory.Config("video-chunk-frames=2\n");
        FractalSharkCli::MovieJob job(jobPath, omittedEncoder);
        ASSERT_EQ(job.Config().PixelFormat, "gbrp");
        job.InitializeProgress(5, true, 0);
        ASSERT_FALSE(std::filesystem::exists(unfinishedPng));
        ASSERT_FALSE(std::filesystem::exists(unfinishedVideo));
        ASSERT_FALSE(std::filesystem::exists(unfinishedInputs));
        ASSERT_TRUE(std::filesystem::exists(retainedPng));
        ASSERT_EQ(std::filesystem::hard_link_count(retainedPng), 1u);
        ASSERT_EQ(job.Remaining(), 3u);
        ASSERT_NEAR(job.LaDepth(), 30, 0);
        ASSERT_TRUE(job.FindChunk(0).has_value());
    }
    const auto conflictingEncoder =
        directory.Config("video-chunk-frames=2\nvideo-encoder-args=" +
                         std::string(FractalShark::DefaultMovieEncoderArguments()) + "\n");
    ASSERT_THROWS(FractalSharkCli::MovieJob(jobPath, conflictingEncoder), FractalSharkSeriousException);
    // A torn write before the atomic replacement leaves the prior record live.
    FractalSharkCli::WriteMovieText(jobPath / "progress.txt.tmp", "torn record");
    {
        FractalSharkCli::MovieJob job(jobPath, {});
        job.InitializeProgress(5, true, 30);
        ASSERT_EQ(job.Remaining(), 3u);
        ASSERT_THROWS(job.InitializeProgress(5, false, 30), FractalSharkSeriousException);
    }
}

TEST(MovieJob_RejectsUnsupportedFrameCounts)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=3000\n");
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

TEST(MovieResume_ProductionLayoutMetadata)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=1000\n");
    const auto jobPath = directory.Path / "job";
    {
        FractalSharkCli::MovieJob job(jobPath, config);
        job.InitializeProgress(10432, true, 30);
        ASSERT_EQ(job.FramePath(9000).parent_path(), job.FramePath(9999).parent_path());
        ASSERT_EQ(job.FramePath(10000).parent_path(), job.FramePath(10431).parent_path());
        ASSERT_TRUE(job.FramePath(9999).parent_path() != job.FramePath(10000).parent_path());
        // Model the interrupted production HEVC layout using tiny metadata and
        // a three-byte movie, never thousands of rendered frames. One CRC
        // checkpoint simulates an interrupted metadata upgrade before its marker.
        auto oldSettings = FractalSharkCli::ReadMovieText(jobPath / "movie.ini");
        for (const std::string key : {"render-order=", "video-encoder-args="}) {
            const auto start = oldSettings.find("\n" + key);
            ASSERT_TRUE(start != std::string::npos);
            const auto end = oldSettings.find('\n', start + 1);
            oldSettings.erase(start, end - start);
        }
        FractalSharkCli::WriteMovieText(jobPath / "movie.ini", oldSettings);
        const std::string identity(64, 'd');
        const std::string oldHeader = std::string(64, '0') + "\n";
        FractalSharkCli::WriteMovieText(
            jobPath / "profile.txt",
            oldHeader + identity + "\nFractalSharkMovie2-hevc-nvenc-render-order\n10432 1 30\n");
        FractalSharkCli::WriteMovieText(jobPath / "progress-0.txt",
                                        oldHeader + identity + "\n1432 1432\n");
        const auto earlier = identity + "\n1431 1431\n";
        FractalSharkCli::WriteMovieText(
            jobPath / "progress-1.txt",
            FractalSharkCli::MovieMetadataChecksum(earlier) + "\n" + earlier);
        FractalSharkCli::WriteMovieText(job.VideoPath(0), "mkv");
        auto record = job.VideoPath(0);
        record += ".complete";
        FractalSharkCli::WriteMovieText(
            record, oldHeader + identity + "\n0 1000 100 3 " + std::string(64, '0') + "\n");
    }
    {
        FractalSharkCli::MovieJob resumed(jobPath, {});
        resumed.InitializeProgress(10432, true, 30);
        ASSERT_EQ(resumed.Remaining(), 9000u);
        ASSERT_EQ(resumed.FramePath(resumed.NextFrame()).filename().string(), "frame_0000001432.png");
        ASSERT_EQ(resumed.Config().ChunkFrames, 1000u);
        ASSERT_FALSE(resumed.Config().WideToDeep);
        ASSERT_EQ(resumed.PlaybackChunkStart(0), 10000u);
        ASSERT_EQ(resumed.PlaybackChunkStart(10), 0u);
        ASSERT_EQ(resumed.Config().PixelFormat, "gbrp");
        ASSERT_FALSE(resumed.Config().Values.contains("video-encoder-args"));
        ASSERT_TRUE(std::filesystem::is_empty(jobPath / "frames"));
        ASSERT_TRUE(resumed.FindChunk(0).has_value());
        ASSERT_TRUE(std::filesystem::exists(jobPath / "progress.txt"));
        ASSERT_FALSE(std::filesystem::exists(jobPath / "progress-0.txt"));
        ASSERT_FALSE(std::filesystem::exists(jobPath / "progress-1.txt"));
        ASSERT_EQ(FractalSharkCli::ReadMovieText(jobPath / "profile.txt").find('\n'), 8u);
        ASSERT_TRUE(FractalSharkCli::ReadMovieText(jobPath / "profile.txt")
                        .find("FractalSharkMovie3-hevc-nvenc-render-order") != std::string::npos);
        ASSERT_THROWS(FractalSharkCli::MovieJob(jobPath, {}), std::system_error);
    }
    const auto profile = FractalSharkCli::ReadMovieText(jobPath / "profile.txt");
    const auto identityEnd = profile.find('\n', 9);
    const auto identity = profile.substr(9, identityEnd - 9);
    const auto clock = identity + "\n100 12345\n";
    FractalSharkCli::WriteMovieText(jobPath / "time-0.txt",
                                    FractalSharkCli::MovieMetadataChecksum(clock) + "\n" + clock);
    FractalSharkCli::WriteMovieText(jobPath / "time-1.txt", "torn timer checkpoint");
    uint64_t elapsed = 0;
    {
        FractalSharkCli::MovieJob resumed(jobPath, {});
        ASSERT_TRUE(resumed.ElapsedMilliseconds() >= 12345);
        ASSERT_TRUE(resumed.FinishTiming());
        elapsed = resumed.ElapsedMilliseconds();
        ASSERT_TRUE(std::filesystem::exists(jobPath / "time.txt"));
        ASSERT_FALSE(std::filesystem::exists(jobPath / "time-0.txt"));
        ASSERT_FALSE(std::filesystem::exists(jobPath / "time-1.txt"));
    }
    FractalSharkCli::WriteMovieText(jobPath / "time.txt.tmp", "torn timer replacement");
    FractalSharkCli::MovieJob timedResume(jobPath, {});
    ASSERT_TRUE(timedResume.ElapsedMilliseconds() >= elapsed);
}

TEST(MovieVideo_ExecutionBudgets)
{
    const auto execution = FractalSharkCli::PlanMovieExecution(32);
    ASSERT_EQ(execution.size(), 4u);
    unsigned decoders = 0;
    for (unsigned slot = 0; slot < execution.size(); ++slot) {
        ASSERT_EQ(execution[slot].Slot, slot);
        ASSERT_EQ(execution[slot].DecoderThreads, slot == 3 ? 7u : 8u);
        ASSERT_EQ(execution[slot].FilterThreads, execution[slot].DecoderThreads);
        decoders += execution[slot].DecoderThreads;
    }
    ASSERT_EQ(decoders, 31u);
    for (const unsigned logical : {0u, 1u, 2u, 3u, 4u, 5u, 64u, 128u}) {
        const auto slots = FractalSharkCli::PlanMovieExecution(logical);
        const auto budget = logical > 1 ? logical - 1 : 1;
        ASSERT_EQ(slots.size(), std::min(4u, budget));
        unsigned total = 0;
        for (const auto &slot : slots) {
            total += slot.DecoderThreads;
            ASSERT_TRUE(slot.DecoderThreads >= 1);
        }
        ASSERT_EQ(total, budget);
    }
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
    const auto config = directory.Config("video-chunk-frames=2\n");
    const auto job = directory.Path / "job";
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    std::filesystem::path frame;
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 0);
        ASSERT_EQ(state.Remaining(), 0u);
        ASSERT_EQ(state.Config().Values.at("video-encoder-args"),
                  FractalShark::DefaultMovieEncoderArguments());
        frame = state.FramePath(3);
        std::filesystem::remove(frame);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    ASSERT_TRUE(std::filesystem::exists(frame));
    // Recover a malformed orphan after rewinding the newest progress record.
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 0);
        state.Rewind(3);
        FractalSharkCli::WriteMovieText(state.FramePath(3), "incomplete PNG");
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
}

TEST(MovieRunner_NativeReferenceReuseAfterRestart)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=2\n");
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
    ASSERT_EQ(
        FractalSharkCli::RunMovie(job, config, {}, GpuMode::Disabled, UINT64_MAX, referenceSettings), 0);
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
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, {}, GpuMode::Disabled, UINT64_MAX, referenceSettings),
              0);
    ASSERT_EQ(countMetadata(), count + 1);
}

TEST(MovieRunner_OrderlyStopAndResume)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=2\n");
    const auto job = directory.Path / "job";
    const auto stopDuringSetup = [](const FractalSharkCli::MovieConfig &settings) {
        // RunMovie installs its signal handler before invoking the factory.
        // Request a stop synchronously, avoiding timing-dependent thread races.
        std::raise(SIGINT);
        return Settings(settings);
    };
    ASSERT_EQ(FractalSharkCli::RunMovie(job, config, {}, GpuMode::Disabled, UINT64_MAX, stopDuringSetup),
              130);
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 4);
        ASSERT_EQ(state.Remaining(), 5u);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
}

TEST(MovieProcess_LowPriorityAndQuietCancellation)
{
    const auto executable = Environment::ExecutablePath();
    const auto priority = Environment::RunMovieProcess(executable,
                                                       {"--movie-process-test-child", "priority"},
                                                       {},
                                                       [](std::span<const unsigned char>) {},
                                                       {},
                                                       {});
    ASSERT_EQ(priority.ExitCode, 0);
    ASSERT_FALSE(priority.Cancelled);
    std::atomic<bool> ready{false};
    const auto started = std::chrono::steady_clock::now();
    const auto result = Environment::RunMovieProcess(
        executable,
        {"--movie-process-test-child", "quiet"},
        {},
        [&](std::span<const unsigned char>) { ready.store(true); },
        {},
        [&] {
            return ready.load() || std::chrono::steady_clock::now() - started > std::chrono::seconds(3);
        });
    ASSERT_TRUE(ready.load());
    ASSERT_TRUE(result.Cancelled);
    ASSERT_TRUE(std::chrono::steady_clock::now() - started < std::chrono::seconds(5));
    ASSERT_THROWS(Environment::RunMovieProcess(executable,
                                               {"--movie-process-test-child", "quiet"},
                                               {},
                                               [](std::span<const unsigned char>) {
                                                   throw FractalSharkSeriousException("consumer failed");
                                               },
                                               {},
                                               {}),
                  FractalSharkSeriousException);
}

#ifdef _WIN32
TEST(MovieProcess_ParentHardKillClosesEncoderJob)
{
    MovieTestDirectory directory;
    const auto pidFile = directory.Path / "encoder.pid";
    const auto executable = Environment::ExecutablePath();
    auto command = L"\"" + executable.wstring() + L"\" --movie-process-test-child owner \"" +
                   pidFile.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    ASSERT_TRUE(CreateProcessW(executable.c_str(),
                               command.data(),
                               nullptr,
                               nullptr,
                               FALSE,
                               CREATE_NO_WINDOW,
                               nullptr,
                               nullptr,
                               &startup,
                               &process));
    struct OwnerCleanup {
        PROCESS_INFORMATION Process;
        HANDLE Encoder = nullptr;
        ~OwnerCleanup()
        {
            TerminateProcess(Process.hProcess, 1);
            WaitForSingleObject(Process.hProcess, 5000);
            CloseHandle(Process.hThread);
            CloseHandle(Process.hProcess);
            if (Encoder) {
                CloseHandle(Encoder);
            }
        }
    } cleanup{process};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    unsigned long pid = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream input(pidFile);
        if (input >> pid) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ASSERT_TRUE(pid != 0);
    cleanup.Encoder = OpenProcess(SYNCHRONIZE, FALSE, pid);
    ASSERT_TRUE(cleanup.Encoder != nullptr);
    ASSERT_TRUE(TerminateProcess(process.hProcess, 1));
    ASSERT_EQ(WaitForSingleObject(process.hProcess, 5000), WAIT_OBJECT_0);
    ASSERT_EQ(WaitForSingleObject(cleanup.Encoder, 5000), WAIT_OBJECT_0);
}
#endif

#ifndef _WIN32
TEST(MovieProcess_BinaryPipesAndProducerFailure)
{
    std::vector<unsigned char> pixels(1024 * 1024);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<unsigned char>(i);
    }
    std::vector<unsigned char> decoded;
    const auto result = Environment::RunMovieProcess(
        "/bin/cat",
        {},
        [&](const std::function<void(std::span<const unsigned char>)> &write) { write(pixels); },
        [&](std::span<const unsigned char> bytes) {
            decoded.insert(decoded.end(), bytes.begin(), bytes.end());
        },
        {},
        {});
    ASSERT_EQ(result.ExitCode, 0);
    ASSERT_TRUE(pixels == decoded);
    ASSERT_THROWS(Environment::RunMovieProcess(
                      "/bin/cat",
                      {},
                      [](const std::function<void(std::span<const unsigned char>)> &) {
                          throw FractalSharkSeriousException("producer failed");
                      },
                      [](std::span<const unsigned char>) {},
                      {},
                      {}),
                  FractalSharkSeriousException);
}
#endif

namespace {
class MovieProxyEnvironment {
public:
    explicit MovieProxyEnvironment(const std::filesystem::path &control)
    {
        if (const auto previous = std::getenv(Name)) {
            m_Previous = previous;
        }
        const auto value = Environment::WideToUtf8(Environment::FromFsPath(control));
#ifdef _WIN32
        const auto error = _putenv_s(Name, value.c_str());
#else
        const auto error = setenv(Name, value.c_str(), 1);
#endif
        if (error != 0) {
            throw FractalSharkSeriousException("Cannot install movie test encoder environment");
        }
    }
    ~MovieProxyEnvironment()
    {
#ifdef _WIN32
        _putenv_s(Name, m_Previous ? m_Previous->c_str() : "");
#else
        if (m_Previous) {
            setenv(Name, m_Previous->c_str(), 1);
        } else {
            unsetenv(Name);
        }
#endif
    }

private:
    static constexpr const char *Name = "FRACTALSHARK_TEST_MOVIE_PROXY";
    std::optional<std::string> m_Previous;
};

enum class ResumeOutcome { Complete, Interrupted, Failed };

void
ResumeWithGatedArchive(ResumeOutcome outcome)
{
    // Fill every archive slot with a recovered range and leave one inherited
    // range queued. At most eight 256x256 frames render across BOTH attempts.
    const auto slots = FractalSharkCli::PlanMovieExecution(Environment::LogicalProcessorCount());
    const uint64_t frontier = (slots.size() + 1) * 4;
    const uint64_t frameCount = frontier + 8;
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=4\n");
    auto text = FractalSharkCli::ReadMovieText(config);
    text.replace(text.find("zoom=16"), 7, "zoom=" + std::to_string(uint64_t{1} << (frameCount - 1)));
    FractalSharkCli::WriteMovieText(config, text);
    const auto jobPath = directory.Path / "job";
    std::filesystem::path firstNewPng, collectedPng, nextBatchPng, firstRecord;
    {
        FractalSharkCli::MovieJob job(jobPath, config);
        job.InitializeProgress(frameCount, true, static_cast<long double>(frameCount - 1));
        for (uint64_t index = 0; index < frontier; ++index) {
            WriteFrame(job.FramePath(index), 256, 256, index);
            job.CompleteFrame(index);
        }
        job.Encoder(Environment::ExecutablePath());
        firstNewPng = job.FramePath(frontier);
        collectedPng = job.FramePath(frontier + 3);
        nextBatchPng = job.FramePath(frontier + 4);
        firstRecord = job.VideoPath(0);
        firstRecord += ".complete";
    }
    const auto control = directory.Path / "proxy";
    std::filesystem::create_directory(control);
    FractalSharkCli::WriteMovieText(control / "ffmpeg.txt", std::getenv("FRACTALSHARK_TEST_FFMPEG"));
    FractalSharkCli::WriteMovieText(control / "hold", "hold first archive");
    if (outcome == ResumeOutcome::Failed) {
        // Fail the youngest active slot while older slots remain blocked.
        FractalSharkCli::WriteMovieText(control / ("fail-" + std::to_string((slots.size() - 1) * 4)),
                                        "fail youngest active archive");
    }
    MovieProxyEnvironment proxy(control);
    bool overlapped = false;
    bool bounded = false;
    std::exception_ptr watchError;
    std::jthread watch([&](std::stop_token token) {
        try {
            while (!token.stop_requested()) {
                bool allStarted = true;
                for (size_t slot = 0; slot < slots.size(); ++slot) {
                    allStarted = allStarted && std::filesystem::exists(
                                                   control / ("started-" + std::to_string(slot * 4)));
                }
                if (allStarted && std::filesystem::exists(collectedPng)) {
                    // Give a broken capacity check time to leak the next
                    // batch; encoder gates keep all slots occupied meanwhile.
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    overlapped =
                        std::filesystem::exists(firstNewPng) && !std::filesystem::exists(firstRecord);
                    bounded = !std::filesystem::exists(nextBatchPng) &&
                              !std::filesystem::exists(control /
                                                       ("started-" + std::to_string(slots.size() * 4)));
                    if (outcome == ResumeOutcome::Interrupted) {
                        std::raise(SIGINT);
                    } else if (outcome == ResumeOutcome::Failed) {
                        FractalSharkCli::WriteMovieText(
                            control / ("release-" + std::to_string((slots.size() - 1) * 4)),
                            "fail this slot only");
                    } else {
                        FractalSharkCli::WriteMovieText(control / "release", "new chunk collected");
                    }
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } catch (...) {
            // A watcher failure is reported on the test thread after joining.
            watchError = std::current_exception();
        }
    });
    const auto result =
        FractalSharkCli::RunMovie(jobPath, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings);
    watch.request_stop();
    watch.join();
    if (watchError) {
        std::rethrow_exception(watchError);
    }
    ASSERT_TRUE(overlapped);
    ASSERT_TRUE(bounded);
    const auto expected = outcome == ResumeOutcome::Complete      ? 0
                          : outcome == ResumeOutcome::Interrupted ? 130
                                                                  : 1;
    ASSERT_EQ(result, expected);
    if (outcome != ResumeOutcome::Complete) {
        FractalSharkCli::MovieJob paused(jobPath, {});
        paused.InitializeProgress(frameCount, true, static_cast<long double>(frameCount - 1));
        ASSERT_EQ(paused.Remaining(), 4u);
        ASSERT_FALSE(paused.FindChunk(0).has_value());
        for (uint64_t index = 0; index < frontier + 4; ++index) {
            ASSERT_TRUE(std::filesystem::exists(paused.FramePath(index)));
        }
        for (uint64_t index = frontier + 4; index < frameCount; ++index) {
            ASSERT_FALSE(std::filesystem::exists(paused.FramePath(index)));
        }
        // On retry use the real encoder directly. The eight unique missing
        // frames across both runs are the entire rendering budget of this case.
        paused.Encoder(std::getenv("FRACTALSHARK_TEST_FFMPEG"));
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(jobPath, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings),
              0);
    FractalSharkCli::MovieJob complete(jobPath, {});
    complete.InitializeProgress(frameCount, true, static_cast<long double>(frameCount - 1));
    ASSERT_EQ(complete.Remaining(), 0u);
    const auto assembly = complete.ReadAssembly();
    ASSERT_TRUE(assembly.has_value());
    ASSERT_EQ(assembly->NextChunk, complete.ChunkCount());
    ASSERT_EQ(std::filesystem::file_size(jobPath / "movie.ts"), assembly->Bytes);
    ASSERT_TRUE(std::filesystem::is_empty(jobPath / "chunks"));
    for (uint64_t index = 0; index < frameCount; ++index) {
        ASSERT_FALSE(std::filesystem::exists(complete.FramePath(index)));
    }
}

const bool ResumeOverlapRegistered = TestFramework::RegisterCase(
    "MovieResume_OverlapsSavedChunksAndBoundsNewFrames",
    [] { ResumeWithGatedArchive(ResumeOutcome::Complete); },
    true,
    std::getenv("FRACTALSHARK_TEST_FFMPEG")
        ? ""
        : "Set FRACTALSHARK_TEST_FFMPEG to enable tiny encoder tests",
    false);
const bool ResumeInterruptRegistered = TestFramework::RegisterCase(
    "MovieResume_InterruptedArchiveDrainsPngsAndResumes",
    [] { ResumeWithGatedArchive(ResumeOutcome::Interrupted); },
    true,
    std::getenv("FRACTALSHARK_TEST_FFMPEG")
        ? ""
        : "Set FRACTALSHARK_TEST_FFMPEG to enable tiny encoder tests",
    false);
const bool ResumeFailureRegistered = TestFramework::RegisterCase(
    "MovieResume_ArchiveFailureDrainsPngsAndResumes",
    [] { ResumeWithGatedArchive(ResumeOutcome::Failed); },
    true,
    std::getenv("FRACTALSHARK_TEST_FFMPEG")
        ? ""
        : "Set FRACTALSHARK_TEST_FFMPEG to enable tiny encoder tests",
    false);

void
ArchiveCompletedPngJob()
{
    // No fractal frames render in this case. All fourteen tiny PNGs already
    // exist, so completion requires draining four recovered archive ranges.
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=4\n");
    auto text = FractalSharkCli::ReadMovieText(config);
    text.replace(text.find("zoom=16"), 7, "zoom=8192");
    FractalSharkCli::WriteMovieText(config, text);
    const auto jobPath = directory.Path / "job";
    {
        FractalSharkCli::MovieJob job(jobPath, config);
        job.InitializeProgress(14, true, 13);
        for (uint64_t index = 0; index < 14; ++index) {
            WriteFrame(job.FramePath(index), 256, 256, index);
            job.CompleteFrame(index);
        }
        job.Encoder(Environment::ExecutablePath());
    }
    const auto control = directory.Path / "proxy";
    std::filesystem::create_directory(control);
    FractalSharkCli::WriteMovieText(control / "ffmpeg.txt", std::getenv("FRACTALSHARK_TEST_FFMPEG"));
    FractalSharkCli::WriteMovieText(control / "hold", "hold each archive independently");
    MovieProxyEnvironment proxy(control);
    const auto slots = FractalSharkCli::PlanMovieExecution(Environment::LogicalProcessorCount());
    bool harvestedYoungest = false;
    std::exception_ptr watchError;
    std::jthread watch([&](std::stop_token token) {
        try {
            while (!token.stop_requested()) {
                bool allStarted = true;
                for (size_t slot = 0; slot < slots.size(); ++slot) {
                    allStarted = allStarted && std::filesystem::exists(
                                                   control / ("started-" + std::to_string(slot * 4)));
                }
                if (allStarted) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto youngest = (slots.size() - 1) * 4;
            FractalSharkCli::WriteMovieText(control / ("release-" + std::to_string(youngest)), "go");
            // Match the ascending chunk-number path contract.
            std::ostringstream padded;
            padded << std::setw(10) << std::setfill('0') << youngest / 4;
            const auto youngestRecord = jobPath / "chunks" / ("chunk_" + padded.str() + ".mkv.complete");
            while (!token.stop_requested() && !std::filesystem::exists(youngestRecord)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            harvestedYoungest =
                std::filesystem::exists(youngestRecord) &&
                (slots.size() == 1 ||
                 !std::filesystem::exists(jobPath / "chunks" / "chunk_0000000000.mkv.complete"));
            FractalSharkCli::WriteMovieText(control / "release", "release older slots");
        } catch (...) {
            watchError = std::current_exception();
        }
    });
    const auto result =
        FractalSharkCli::RunMovie(jobPath, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings);
    watch.request_stop();
    watch.join();
    if (watchError) {
        std::rethrow_exception(watchError);
    }
    ASSERT_EQ(result, 0);
    ASSERT_TRUE(harvestedYoungest);
    FractalSharkCli::MovieJob job(jobPath, {});
    job.InitializeProgress(14, true, 13);
    ASSERT_EQ(job.Remaining(), 0u);
    const auto assembly = job.ReadAssembly();
    ASSERT_TRUE(assembly.has_value());
    ASSERT_EQ(assembly->NextChunk, 4u);
    ASSERT_TRUE(std::filesystem::is_empty(jobPath / "chunks"));
    for (uint64_t index = 0; index < 14; ++index) {
        ASSERT_FALSE(std::filesystem::exists(job.FramePath(index)));
    }
    ASSERT_EQ(std::filesystem::file_size(jobPath / "movie.ts"), assembly->Bytes);
    ASSERT_FALSE(std::filesystem::exists(jobPath / "movie.ffconcat"));
}
const bool ResumeCompletedPngRegistered =
    TestFramework::RegisterCase("MovieResume_CompletedPngJobDrainsAllArchives",
                                ArchiveCompletedPngJob,
                                true,
                                std::getenv("FRACTALSHARK_TEST_FFMPEG")
                                    ? ""
                                    : "Set FRACTALSHARK_TEST_FFMPEG to enable tiny encoder tests",
                                false);

void
ChunkAssembly()
{
    // Synthetic PNGs exercise chunk assembly without rendering fractals.
    constexpr unsigned Width = 1024;
    constexpr unsigned Height = 512;
    constexpr uint64_t Frames = 4;
    MovieTestDirectory directory;
    const auto config =
        directory.Config("video-chunk-frames=2\nvideo-encoder-args=" +
                         std::string(FractalShark::LosslessMovieEncoderArguments()) + "\n");
    const auto jobPath = directory.Path / "job";
    auto job = std::make_unique<FractalSharkCli::MovieJob>(jobPath, config);
    job->InitializeProgress(Frames, true, 3);
    std::vector<unsigned char> expected;
    for (uint64_t index = 0; index < Frames; ++index) {
        WriteFrame(job->FramePath(index), Width, Height, index);
    }
    for (uint64_t index = 0; index < Frames; ++index) {
        const auto pixels = FractalSharkCli::ReadMovieRgb8(job->FramePath(index), Width, Height);
        expected.insert(expected.end(), pixels.begin(), pixels.end());
    }
    const std::filesystem::path encoder(std::getenv("FRACTALSHARK_TEST_FFMPEG"));
    auto execution = FractalSharkCli::PlanMovieExecution(32).front();
    const FractalSharkCli::MovieVideo video(
        encoder, Width, Height, job->Config(), execution, [] { return false; });
    // The neighboring chunk is deliberately unreadable while the first is
    // archived. Output frame limits alone cannot prevent image2 prefetch from
    // touching it; private input sequences must impose the actual boundary.
    FractalSharkCli::WriteMovieText(job->FramePath(2), "invalid neighboring PNG");
    job->CommitChunk(video.Archive(*job, 0, 2));
    WriteFrame(job->FramePath(2), Width, Height, 2);
    job->CommitChunk(video.Archive(*job, 2, 2));
    job->WriteConcatManifest();
    const auto joined = jobPath / "movie.ts";
    const auto partial = jobPath / "movie.ts.partial";
    const auto firstChunk = job->VideoPath(0);
    const auto secondChunk = job->VideoPath(2);
    auto firstRecord = firstChunk;
    firstRecord += ".complete";
    const auto stop = [] { return false; };
    FractalSharkCli::WriteMovieText(joined, "prior movie");
    ASSERT_THROWS(FractalSharkCli::AssembleMovie(*job, encoder, stop), FractalSharkSeriousException);
    ASSERT_TRUE(job->HasAllChunks());
    ASSERT_EQ(FractalSharkCli::ReadMovieText(joined), "prior movie");
    std::filesystem::remove(joined);

    const auto control = directory.Path / "join-proxy";
    std::filesystem::create_directory(control);
    FractalSharkCli::WriteMovieText(control / "ffmpeg.txt", std::getenv("FRACTALSHARK_TEST_FFMPEG"));
    const auto fail = control / ("fail-join-" + secondChunk.filename().string());
    const auto hold = control / ("hold-join-" + secondChunk.filename().string());
    const auto started = control / ("started-join-" + secondChunk.filename().string());
    const auto firstPayload = FractalSharkCli::ReadMovieText(firstChunk);
    const auto firstMetadata = FractalSharkCli::ReadMovieText(firstRecord);
    const auto noRender =
        [](const FractalSharkCli::MovieConfig &) -> FractalSharkCli::MovieRenderSettings {
        throw FractalSharkSeriousException("An assembly-only launch initialized the renderer");
    };
    {
        MovieProxyEnvironment proxy(control);
        FractalSharkCli::WriteMovieText(fail, "fail after writing bytes");
        ASSERT_THROWS(FractalSharkCli::AssembleMovie(*job, Environment::ExecutablePath(), stop),
                      FractalSharkSeriousException);
        const auto saved = job->ReadAssembly();
        ASSERT_TRUE(saved.has_value());
        ASSERT_EQ(saved->NextChunk, 1u);
        ASSERT_TRUE(std::filesystem::file_size(partial) > saved->Bytes);
        ASSERT_FALSE(std::filesystem::exists(firstChunk));
        ASSERT_TRUE(std::filesystem::exists(secondChunk));
        ASSERT_FALSE(std::filesystem::exists(joined));

        // Simulate a shutdown after committing the first append but before
        // unlinking its source. The retry must reclaim it without copying it.
        FractalSharkCli::WriteMovieText(firstChunk, firstPayload);
        FractalSharkCli::WriteMovieText(firstRecord, firstMetadata);
        job.reset();
        std::filesystem::remove(fail);
        std::filesystem::remove(started);
        FractalSharkCli::WriteMovieText(hold, "wait for interrupt");
        std::jthread interrupt([&](std::stop_token token) {
            std::error_code error;
            while (!token.stop_requested()) {
                if (std::filesystem::exists(started, error)) {
                    std::raise(SIGINT);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        const auto result = FractalSharkCli::RunMovie(
            jobPath, {}, Environment::ExecutablePath(), GpuMode::Disabled, UINT64_MAX, noRender);
        interrupt.request_stop();
        interrupt.join();
        ASSERT_EQ(result, 130);
        ASSERT_FALSE(std::filesystem::exists(firstChunk));
        ASSERT_TRUE(std::filesystem::exists(secondChunk));
        FractalSharkCli::MovieJob paused(jobPath, {});
        const auto pausedAssembly = paused.ReadAssembly();
        ASSERT_TRUE(pausedAssembly.has_value());
        ASSERT_EQ(pausedAssembly->NextChunk, 1u);
        ASSERT_EQ(pausedAssembly->Bytes, saved->Bytes);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(jobPath, {}, encoder, GpuMode::Disabled, UINT64_MAX, noRender),
              0);
    ASSERT_TRUE(std::filesystem::is_empty(jobPath / "chunks"));
    ASSERT_FALSE(std::filesystem::exists(partial));
    const auto movieBytes = std::filesystem::file_size(joined);
    // Both sides of final publication are retryable with the full journal and
    // no encoder executable. Moving the file back simulates the prior state.
    std::filesystem::rename(joined, partial);
    const auto missingEncoder = directory.Path / "unavailable-ffmpeg";
    ASSERT_EQ(
        FractalSharkCli::RunMovie(jobPath, {}, missingEncoder, GpuMode::Disabled, UINT64_MAX, noRender),
        0);
    ASSERT_EQ(
        FractalSharkCli::RunMovie(jobPath, {}, missingEncoder, GpuMode::Disabled, UINT64_MAX, noRender),
        0);
    ASSERT_EQ(std::filesystem::file_size(joined), movieBytes);
    const auto pathText = [](const std::filesystem::path &path) {
        return Environment::WideToUtf8(Environment::FromFsPath(path));
    };
    std::vector<unsigned char> pixels;
    const auto decoded =
        Environment::RunMovieProcess(encoder,
                                     {"-hide_banner",
                                      "-nostdin",
                                      "-v",
                                      "error",
                                      "-threads:v",
                                      "1",
                                      "-i",
                                      pathText(joined),
                                      "-filter_threads",
                                      "1",
                                      "-pix_fmt",
                                      "rgb24",
                                      "-c:v",
                                      "rawvideo",
                                      "-threads:v",
                                      "1",
                                      "-fps_mode",
                                      "passthrough",
                                      "-f",
                                      "rawvideo",
                                      "pipe:1"},
                                     {},
                                     [&](std::span<const unsigned char> bytes) {
                                         ASSERT_TRUE(pixels.size() + bytes.size() <= expected.size());
                                         pixels.insert(pixels.end(), bytes.begin(), bytes.end());
                                     },
                                     {},
                                     {});
    ASSERT_EQ(decoded.ExitCode, 0);
    ASSERT_TRUE(pixels == expected);
    const std::vector<std::string> profiles{
        std::string(FractalShark::DefaultMovieEncoderArguments()),
        "-c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 22 -profile:v main -pix_fmt nv12 "
        "-colorspace bt709 -color_range tv",
        "-c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 18 -b:v 0 -profile:v high -pix_fmt nv12",
        "-c:v h264_nvenc -preset p1 -tune hq -rc vbr -cq 30 -b:v 0 -profile:v high -pix_fmt nv12"};
    for (size_t option = 0; option < profiles.size(); ++option) {
        const auto selectedConfig =
            directory.Config("video-chunk-frames=2\nvideo-encoder-args=" + profiles[option] + "\n");
        FractalSharkCli::MovieJob selected(directory.Path / ("codec-" + std::to_string(option)),
                                           selectedConfig);
        selected.InitializeProgress(Frames, true, 3);
        FractalSharkCli::MovieVideo selectedVideo(
            encoder, Width, Height, selected.Config(), execution, [] { return false; });
        selectedVideo.Preflight(selected.Directory());
        for (uint64_t index = 0; index < Frames; ++index) {
            WriteFrame(selected.FramePath(index), Width, Height, index);
        }
        selected.CommitChunk(selectedVideo.Archive(selected, 0, 2));
        selected.CommitChunk(selectedVideo.Archive(selected, 2, 2));
        FractalSharkCli::AssembleMovie(selected, encoder, stop);
        const auto assembly = selected.ReadAssembly();
        ASSERT_TRUE(assembly.has_value());
        ASSERT_EQ(assembly->NextChunk, 2u);
        ASSERT_TRUE(std::filesystem::is_empty(selected.Directory() / "chunks"));
    }
#ifdef _WIN32
    // Reuse this GPU-gated case for a bounded serial/pipelined comparison:
    // exactly four frames per path, with varying iteration limits. Native Linux
    // CPU-only builds still exercise the independent NVENC/assembly checks above.
    std::filesystem::current_path(directory.Path);
    const auto configure = [](Fractal &current, uint64_t index) {
        current.SetOfflineView(PointZoomBBConverter(HighPrecision("-0.75"),
                                                    HighPrecision("0.1"),
                                                    HighPrecision(std::to_string(uint64_t{2} << index)),
                                                    PointZoomBBConverter::TestMode::Disabled));
        constexpr uint64_t BaseIterations = 8192;
        constexpr uint64_t IterationStep = 128;
        current.SetNumIterations<uint64_t>(BaseIterations + IterationStep * index);
    };
    constexpr unsigned PipelineWidth = 512;
    constexpr unsigned PipelineHeight = 256;
    double serialSeconds = 0;
    double parallelSeconds = 0;
    {
        Fractal serial(PipelineWidth, PipelineHeight, nullptr, false, UINT64_MAX, true, GpuMode::Auto);
        serial.GetRenderPool()->Drain();
        serial.SetIterType(IterTypeEnum::Bits64);
        ASSERT_TRUE(
            serial.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x64)));
        const auto serialStart = std::chrono::steady_clock::now();
        for (uint64_t index = 0; index < Frames; ++index) {
            configure(serial, index);
            serial.CalcFractal(true);
            ASSERT_TRUE(serial
                            .SaveCurrentFractalWithResult(
                                Environment::FromFsPath(directory.Path /
                                                        ("serial-" + std::to_string(index) + ".png")),
                                false,
                                SaveOverwritePolicy::ReplaceExisting)
                            .get()
                            .Succeeded());
        }
        serial.CleanupThreads(true);
        serialSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - serialStart).count();
    }
    {
        Fractal parallel(PipelineWidth, PipelineHeight, nullptr, false, UINT64_MAX, true, GpuMode::Auto);
        parallel.SetIterType(IterTypeEnum::Bits64);
        ASSERT_TRUE(
            parallel.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x64)));
        const auto parallelStart = std::chrono::steady_clock::now();
        std::vector<std::future<SaveResult>> saves;
        for (uint64_t index = 0; index < Frames; ++index) {
            saves.push_back(parallel.EnqueueRenderToPng(
                "bounded movie pipeline comparison",
                [configure, index](Fractal &current) { configure(current, index); },
                Environment::FromFsPath(directory.Path / ("parallel-" + std::to_string(index) + ".png")),
                SaveOverwritePolicy::ReplaceExisting,
                [] { return false; }));
        }
        for (auto &save : saves) {
            ASSERT_TRUE(save.get().Succeeded());
        }
        parallel.GetRenderPool()->Drain();
        parallel.CleanupThreads(true);
        parallelSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - parallelStart).count();
    }
    for (uint64_t index = 0; index < Frames; ++index) {
        ASSERT_TRUE(FractalSharkCli::ReadMovieRgb8(
                        directory.Path / ("serial-" + std::to_string(index) + ".png"), 512, 256) ==
                    FractalSharkCli::ReadMovieRgb8(
                        directory.Path / ("parallel-" + std::to_string(index) + ".png"), 512, 256));
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie pipeline comparison frames=" << Frames << " serial_seconds=" << serialSeconds
        << " pipelined_seconds=" << parallelSeconds;
#endif
}
const bool ChunkAssemblyRegistered =
    TestFramework::RegisterCase("MovieVideo_ChunkAssembly",
                                ChunkAssembly,
                                true,
                                std::getenv("FRACTALSHARK_TEST_FFMPEG")
                                    ? ""
                                    : "Set FRACTALSHARK_TEST_FFMPEG to enable tiny encoder tests",
                                false);

void
CodecRoundTrip()
{
    MovieTestDirectory directory;
    const auto config =
        directory.Config("video-chunk-frames=1000\nvideo-encoder-args=" +
                         std::string(FractalShark::LosslessMovieEncoderArguments()) + "\n");
    FractalSharkCli::MovieJob job(directory.Path / "job", config);
    job.InitializeProgress(1001, true, 30);
    FractalSharkCli::MovieVideo video(std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                      256,
                                      256,
                                      job.Config(),
                                      FractalSharkCli::PlanMovieExecution(32).front(),
                                      [] { return false; });
    video.Preflight(directory.Path);
    for (uint64_t i = 0; i < 1001; ++i) {
        WriteFrame(job.FramePath(i), 256, 256, i);
    }
    FractalSharkCli::MovieVideo cancelled(std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                          256,
                                          256,
                                          job.Config(),
                                          FractalSharkCli::PlanMovieExecution(32).front(),
                                          [] { return true; });
    ASSERT_THROWS(cancelled.Archive(job, 0, 1000), FractalSharkCli::MovieInterrupted);
    ASSERT_TRUE(std::filesystem::exists(job.FramePath(0)));
    // A crash before registry publication leaves PNGs authoritative. The next
    // archive safely replaces an unregistered partial/corrupt movie.
    FractalSharkCli::WriteMovieText(job.VideoPath(1000), "incomplete movie");
    for (const auto first : {uint64_t{1000}, uint64_t{0}}) {
        const auto count = first == 1000 ? 1 : 1000;
        const auto chunk = video.Archive(job, first, count);
        ASSERT_EQ(chunk.Count, static_cast<uint64_t>(count));
        ASSERT_TRUE(std::filesystem::exists(job.FramePath(first)));
        ASSERT_FALSE(job.FindChunk(first).has_value());
        job.CommitChunk(chunk);
        job.DeleteChunkPngs(chunk);
        ASSERT_FALSE(std::filesystem::exists(job.FramePath(first)));
    }
    job.WriteConcatManifest();
    ASSERT_TRUE(FractalSharkCli::ReadMovieText(job.Directory() / "movie.ffconcat").find(".mkv") !=
                std::string::npos);
    const auto pipelineConfig = directory.Config("video-chunk-frames=2\n");
    const auto pipelineJob = directory.Path / "pipeline";
    ASSERT_EQ(FractalSharkCli::RunMovie(pipelineJob,
                                        pipelineConfig,
                                        std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                        GpuMode::Disabled,
                                        UINT64_MAX,
                                        VideoSettings),
              0);
    {
        FractalSharkCli::MovieJob state(pipelineJob, {});
        state.InitializeProgress(5, true, 4);
        ASSERT_EQ(state.Remaining(), 0u);
        const auto assembly = state.ReadAssembly();
        ASSERT_TRUE(assembly.has_value());
        ASSERT_EQ(assembly->NextChunk, 3u);
        ASSERT_TRUE(std::filesystem::is_empty(pipelineJob / "chunks"));
        ASSERT_FALSE(std::filesystem::exists(state.FramePath(0)));
    }
    ASSERT_EQ(
        FractalSharkCli::RunMovie(pipelineJob, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings), 0);

    // A GUI-created folder starts and resumes with no external INI or encoder
    // arguments. Its only configuration is the file already inside that folder.
    const auto exported = FractalShark::BuildMovieCommandExport(directory.Path / "exported.ini",
                                                                Environment::ExecutablePath(),
                                                                std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                                                FractalShark::MovieCommandShell::Posix);
    auto single = FractalSharkCli::ReadMovieText(pipelineConfig);
    single.replace(single.find("zoom=16"), 7, "zoom=1");
    FractalShark::SaveMovieConfiguration(exported, single);
    ASSERT_EQ(FractalSharkCli::RunMovie(
                  exported.JobDirectory, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings),
              0);
    ASSERT_EQ(FractalSharkCli::RunMovie(
                  exported.JobDirectory, {}, {}, GpuMode::Disabled, UINT64_MAX, VideoSettings),
              0);

    // An odd stride visits every RGB16 value. Three channels repeat all 65536
    // values; the four-frame assembly case checks conversion and pixel order.
    FractalSharkCli::MovieJob exhaustive(directory.Path / std::filesystem::path(u8"100% \u03b2"),
                                         config);
    WriteFrame(exhaustive.FramePath(0), 256, 256, 17);
    FractalSharkCli::MovieVideo wide(std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                     256,
                                     256,
                                     job.Config(),
                                     FractalSharkCli::PlanMovieExecution(32).front(),
                                     [] { return false; });
    const auto wideChunk = wide.Archive(exhaustive, 0, 1);
    ASSERT_EQ(wideChunk.Count, 1u);

    FractalSharkCli::MovieJob damaged(directory.Path / "damaged", config);
    WriteFrame(damaged.FramePath(0), 256, 256, 0);
    WriteFrame(damaged.FramePath(1), 8, 32, 1);
    ASSERT_THROWS(video.Archive(damaged, 0, 2), FractalSharkSeriousException);
    ASSERT_FALSE(std::filesystem::exists(damaged.VideoPath(0)));
    WriteFrame(damaged.FramePath(1), 256, 256, 1);
    auto corrupt = FractalSharkCli::ReadMovieText(damaged.FramePath(1));
    const auto idat = corrupt.find("IDAT");
    ASSERT_TRUE(idat != std::string::npos && idat >= 4);
    uint32_t length = 0;
    for (size_t i = idat - 4; i < idat; ++i) {
        length = (length << 8) | static_cast<unsigned char>(corrupt[i]);
    }
    ASSERT_TRUE(idat + 4 + length < corrupt.size());
    corrupt[idat + 4 + length] ^= 1;
    FractalSharkCli::WriteMovieText(damaged.FramePath(1), corrupt);
    ASSERT_THROWS(video.Archive(damaged, 0, 2), FractalSharkSeriousException);
    ASSERT_TRUE(std::filesystem::exists(damaged.FramePath(0)));
    ASSERT_TRUE(std::filesystem::exists(damaged.FramePath(1)));
    FractalSharkCli::WriteMovieText(damaged.FramePath(1), corrupt.substr(0, corrupt.size() / 2));
    ASSERT_THROWS(video.Archive(damaged, 0, 2), FractalSharkSeriousException);
    ASSERT_THROWS(video.Archive(damaged, 9999, 2), FractalSharkSeriousException);

    // Interrupt an active archive while the render thread is producing the
    // following chunk. Every completed PNG must reach the ascending checkpoint
    // before exit 130, even if a worker observes cancellation first.
    const auto pauseConfig = directory.Config("video-chunk-frames=4\n");
    auto pauseText = FractalSharkCli::ReadMovieText(pauseConfig);
    pauseText.replace(pauseText.find("zoom=16"), 7, "zoom=4294967296");
    FractalSharkCli::WriteMovieText(pauseConfig, pauseText);
    const auto pauseJob = directory.Path / "pause";
    const auto pauseSettings = [](const FractalSharkCli::MovieConfig &config) {
        auto result = Settings(config);
        result.Request.Width = 256;
        result.Request.Height = 256;
        return result;
    };
    std::exception_ptr watchError;
    std::jthread interrupter([&](std::stop_token token) {
        try {
            const auto nextPng = pauseJob / "frames" / "chunk_0000000001" / "frame_0000000004.png";
            while (!token.stop_requested()) {
                if (std::filesystem::exists(nextPng)) {
                    for (const auto &entry : std::filesystem::directory_iterator(pauseJob / "chunks")) {
                        if (entry.path().filename().string().starts_with(".fractalshark-video-")) {
                            std::raise(SIGINT);
                            return;
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        } catch (...) {
            watchError = std::current_exception();
        }
    });
    const auto paused = FractalSharkCli::RunMovie(pauseJob,
                                                  pauseConfig,
                                                  std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                                  GpuMode::Disabled,
                                                  UINT64_MAX,
                                                  pauseSettings);
    interrupter.request_stop();
    interrupter.join();
    if (watchError) {
        std::rethrow_exception(watchError);
    }
    ASSERT_EQ(paused, 130);
    {
        FractalSharkCli::MovieJob state(pauseJob, {});
        state.InitializeProgress(33, true, 32);
        ASSERT_TRUE(state.Remaining() < 33 && state.Remaining() > 0);
        for (uint64_t index = 0; index < state.NextFrame(); ++index) {
            ASSERT_TRUE(state.FindChunk((index / 4) * 4).has_value() ||
                        std::filesystem::exists(state.FramePath(index)));
        }
        // No completed orphan may be left beyond the checkpoint on an orderly
        // stop. Such files would hide a skipped pending-save drain.
        for (uint64_t index = state.NextFrame(); index < 33; ++index) {
            ASSERT_FALSE(std::filesystem::exists(state.FramePath(index)));
        }
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(pauseJob, {}, {}, GpuMode::Disabled, UINT64_MAX, pauseSettings),
              0);
}
const bool CodecRegistered =
    TestFramework::RegisterCase("MovieVideo_1000FrameLosslessSingleKeyframe",
                                CodecRoundTrip,
                                true,
                                std::getenv("FRACTALSHARK_TEST_FFMPEG")
                                    ? ""
                                    : "Set FRACTALSHARK_TEST_FFMPEG to enable FFmpeg integration",
                                false);
} // namespace
