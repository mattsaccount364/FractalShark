#include "Environment.h"
#include "Fractal.h"
#include "FractalSharkCli/MovieHash.h"
#include "FractalSharkCli/MovieJob.h"
#include "FractalSharkCli/MovieRunner.h"
#include "FractalSharkCli/MovieVideo.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <limits>
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

TEST(MovieHash_StandardVectorsAndStreaming)
{
    ASSERT_EQ(FractalSharkCli::MovieHash::Digest(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ASSERT_EQ(FractalSharkCli::MovieHash::Digest("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    FractalSharkCli::MovieHash hash;
    const std::string block(1000, 'a');
    for (unsigned i = 0; i < 1000; ++i) {
        hash.Update(block);
    }
    ASSERT_EQ(hash.Finish(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    ASSERT_EQ(hash.Finish(), hash.Finish());
}

TEST(MovieSchedule_EndpointsCurveAndHugeExponents)
{
    HighPrecision::defaultPrecisionInBits(1024);
    FractalSharkCli::MovieSchedule schedule(HighPrecision("16"), "2", 10, 110, 2, 4);
    ASSERT_EQ(schedule.FrameCount(), 5u);
    ASSERT_NEAR(static_cast<double>(schedule.ZoomAt(0)), 1, 0);
    ASSERT_NEAR(static_cast<double>(schedule.ZoomAt(4)), 16, 0);
    ASSERT_EQ(schedule.IterationsAt(0), 10u);
    ASSERT_EQ(schedule.IterationsAt(2), 30u);
    ASSERT_EQ(schedule.IterationsAt(3), 70u);
    ASSERT_EQ(schedule.IterationsAt(4), 110u);
    FractalSharkCli::MovieSchedule partial(HighPrecision("15"), "2", 10, 100, 2, 8);
    ASSERT_EQ(partial.FrameCount(), 5u);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(0)), 1, 0);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(1)), 1.875, 0);
    ASSERT_NEAR(static_cast<double>(partial.ZoomAt(4)), 15, 0);
    HighPrecision huge{HighPrecision::SetPrecision::True, 256};
    mpf_set_ui(huge.backend(), 1);
    mpf_mul_2exp(huge.backend(), huge.backend(), 10'000'000);
    FractalSharkCli::MovieSchedule deep(huge, "2", 10, 1000, 30, 8);
    ASSERT_EQ(deep.FrameCount(), 10'000'001u);
    ASSERT_NEAR(FractalSharkCli::MovieSchedule::Log2(deep.ZoomAt(9'999'999)), 9'999'999, 1e-8L);
    ASSERT_EQ(deep.IterationsAt(0), 10u);
    ASSERT_EQ(deep.IterationsAt(10'000'000), 1000u);
    ASSERT_THROWS(
        FractalSharkCli::MovieSchedule(HighPrecision("16"), "1.00000000000000000001", 10, 100, 2, 8),
        FractalSharkSeriousException);
    FractalSharkCli::MovieSchedule single(HighPrecision("1"), "2", 10, 100, 30, 8);
    ASSERT_EQ(single.FrameCount(), 1u);
    ASSERT_EQ(single.IterationsAt(0), 100u);
    FractalSharkCli::MovieSchedule nearOne(
        HighPrecision("1.00000000000000000001"), "1.000000000000000000001", 10, 100, 0, 8);
    ASSERT_EQ(nearOne.FrameCount(), 11u);
    ASSERT_TRUE(nearOne.DepthAt(10) > 0);
    FractalSharkCli::MovieSchedule steep(
        HighPrecision("16"), "2", 10, 100, 2, std::numeric_limits<double>::max());
    ASSERT_TRUE(steep.IterationsAt(1) <= steep.IterationsAt(3));
}

TEST(MovieJob_CheckpointsLockAndChunkDeletionOrder)
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=2\n");
    const auto jobPath = directory.Path / "job";
    {
        FractalSharkCli::MovieJob job(jobPath, config);
        job.InitializeProgress(5, true, 30);
        ASSERT_EQ(job.Remaining(), 5u);
        ASSERT_THROWS(FractalSharkCli::MovieJob(jobPath, config), std::system_error);
        WriteFrame(job.FramePath(4), 16, 16, 4);
        job.CompleteFrame(4);
        WriteFrame(job.FramePath(3), 16, 16, 3);
        job.CompleteFrame(3);
        ASSERT_THROWS(job.CompleteFrame(1), FractalSharkSeriousException);
        FractalSharkCli::MovieChunk chunk{4, 1, 100, 3, std::string(64, '0')};
        ASSERT_THROWS(job.DeleteChunkPngs(chunk), FractalSharkSeriousException);
        ASSERT_TRUE(std::filesystem::exists(job.FramePath(4)));
        FractalSharkCli::WriteMovieText(job.VideoPath(4), "mkv");
        job.CommitChunk(chunk);
        job.DeleteChunkPngs(chunk);
        job.DeleteChunkPngs(chunk);
        ASSERT_FALSE(std::filesystem::exists(job.FramePath(4)));
    }
    const auto unfinishedPng = jobPath / "frames" / ".fractalshark-save-old.tmp";
    const auto unfinishedVideo = jobPath / "chunks" / ".fractalshark-video-old.tmp";
    FractalSharkCli::WriteMovieText(unfinishedPng, "unfinished PNG");
    FractalSharkCli::WriteMovieText(unfinishedVideo, "unfinished movie");
    {
        FractalSharkCli::MovieJob job(jobPath, {});
        job.InitializeProgress(5, true, 0);
        ASSERT_FALSE(std::filesystem::exists(unfinishedPng));
        ASSERT_FALSE(std::filesystem::exists(unfinishedVideo));
        ASSERT_EQ(job.Remaining(), 3u);
        ASSERT_NEAR(job.LaDepth(), 30, 0);
        ASSERT_TRUE(job.FindChunk(4).has_value());
    }
    // A torn newest checkpoint falls back to the previous complete one.
    FractalSharkCli::WriteMovieText(jobPath / "progress-0.txt", "torn record");
    {
        FractalSharkCli::MovieJob job(jobPath, {});
        job.InitializeProgress(5, true, 30);
        ASSERT_EQ(job.Remaining(), 4u);
        ASSERT_THROWS(job.InitializeProgress(5, false, 30), FractalSharkSeriousException);
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
        frame = state.FramePath(3);
        std::filesystem::remove(frame);
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(job, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings), 0);
    ASSERT_TRUE(std::filesystem::exists(frame));
    // Recover a malformed orphan after rewinding the newest progress record.
    {
        FractalSharkCli::MovieJob state(job, {});
        state.InitializeProgress(5, true, 0);
        state.Rewind(4);
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
                      {}),
                  FractalSharkSeriousException);
}
#endif

namespace {
void
CodecRoundTrip()
{
    MovieTestDirectory directory;
    const auto config = directory.Config("video-chunk-frames=1000\n");
    FractalSharkCli::MovieJob job(directory.Path / "job", config);
    job.InitializeProgress(1001, true, 30);
    FractalSharkCli::MovieVideo video(
        std::getenv("FRACTALSHARK_TEST_FFMPEG"), 16, 16, 60, [] { return false; });
    video.Preflight(directory.Path);
    for (uint64_t i = 0; i < 1001; ++i) {
        WriteFrame(job.FramePath(i), 16, 16, i);
    }
    FractalSharkCli::MovieVideo cancelled(
        std::getenv("FRACTALSHARK_TEST_FFMPEG"), 16, 16, 60, [] { return true; });
    ASSERT_THROWS(cancelled.Archive(job, 0, 1000), FractalSharkCli::MovieInterrupted);
    ASSERT_TRUE(std::filesystem::exists(job.FramePath(0)));
    // A crash before registry publication leaves PNGs authoritative. The next
    // archive safely replaces an unregistered partial/corrupt movie.
    FractalSharkCli::WriteMovieText(job.VideoPath(1000), "incomplete movie");
    for (const auto first : {uint64_t{1000}, uint64_t{0}}) {
        const auto count = first == 1000 ? 1 : 1000;
        const auto chunk = video.Archive(job, first, count);
        ASSERT_TRUE(std::filesystem::exists(job.FramePath(first)));
        ASSERT_FALSE(job.FindChunk(first).has_value());
        video.Verify(job.VideoPath(first), count, chunk.RgbHash);
        ASSERT_THROWS(video.Verify(job.VideoPath(first), count, std::string(64, '0')),
                      FractalSharkSeriousException);
        job.CommitChunk(chunk);
        job.DeleteChunkPngs(chunk);
        ASSERT_FALSE(std::filesystem::exists(job.FramePath(first)));
    }
    job.WriteConcatManifest(1001);
    ASSERT_TRUE(FractalSharkCli::ReadMovieText(job.Directory() / "movie.ffconcat").find(".mkv") !=
                std::string::npos);
    const auto pipelineConfig = directory.Config("video-chunk-frames=2\n");
    const auto pipelineJob = directory.Path / "pipeline";
    ASSERT_EQ(FractalSharkCli::RunMovie(pipelineJob,
                                        pipelineConfig,
                                        std::getenv("FRACTALSHARK_TEST_FFMPEG"),
                                        GpuMode::Disabled,
                                        UINT64_MAX,
                                        Settings),
              0);
    {
        FractalSharkCli::MovieJob state(pipelineJob, {});
        state.InitializeProgress(5, true, 4);
        ASSERT_EQ(state.Remaining(), 0u);
        ASSERT_EQ(state.FindChunk(0)->Count, 2u);
        ASSERT_EQ(state.FindChunk(2)->Count, 2u);
        ASSERT_EQ(state.FindChunk(4)->Count, 1u);
        ASSERT_FALSE(std::filesystem::exists(state.FramePath(0)));
    }
    ASSERT_EQ(FractalSharkCli::RunMovie(pipelineJob, {}, {}, GpuMode::Disabled, UINT64_MAX, Settings),
              0);
}
const bool CodecRegistered =
    TestFramework::RegisterCase("MovieVideo_1000FrameLosslessSingleKeyframe",
                                CodecRoundTrip,
                                false,
                                std::getenv("FRACTALSHARK_TEST_FFMPEG")
                                    ? ""
                                    : "Set FRACTALSHARK_TEST_FFMPEG to enable FFmpeg integration",
                                false);
} // namespace
