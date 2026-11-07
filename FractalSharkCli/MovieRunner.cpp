#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieRunner.h"
#include "MovieVideo.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <future>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace FractalSharkCli {
namespace {
std::atomic<bool> StopRequested{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void
StopMovie(int)
{
    StopRequested.store(true, std::memory_order_relaxed);
}
#ifdef _WIN32
BOOL WINAPI
ConsoleStop(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
        StopMovie(0);
        return TRUE;
    }
    return FALSE;
}
#endif
class MovieSignals {
public:
    MovieSignals()
    {
        StopRequested.store(false, std::memory_order_relaxed);
        m_Previous = std::signal(SIGINT, StopMovie);
#ifdef _WIN32
        if (!SetConsoleCtrlHandler(ConsoleStop, TRUE)) {
            throw FractalSharkSeriousException("Cannot install movie console stop handler");
        }
#else
        m_PreviousTerm = std::signal(SIGTERM, StopMovie);
#endif
    }
    ~MovieSignals()
    {
        std::signal(SIGINT, m_Previous);
#ifdef _WIN32
        SetConsoleCtrlHandler(ConsoleStop, FALSE);
#else
        std::signal(SIGTERM, m_PreviousTerm);
#endif
    }

private:
    decltype(&StopMovie) m_Previous;
#ifndef _WIN32
    decltype(&StopMovie) m_PreviousTerm;
#endif
};

long double
ResolveLaDepth(const RenderAlgorithm &algorithm, bool gpuBypassed, long double targetDepth)
{
    if (algorithm.Algorithm == RenderAlgorithmEnum::AUTO) {
        return std::log2(gpuBypassed ? 1e34L : 1e9L);
    }
    bool fullLa = false;
    IterateRenderAlgs([&](auto index) {
        constexpr auto candidate = std::get<index.value>(RenderAlgorithmsTuple);
        if (candidate.Algorithm == algorithm.Algorithm) {
            fullLa = candidate.LAv2 == LAv2Mode::Full || candidate.LAv2 == LAv2Mode::LAO;
        }
    });
    return fullLa ? 0 : targetDepth;
}
void
SetMovieFrame(Fractal &fractal,
              const RenderRequest &target,
              const MovieSchedule &schedule,
              uint64_t index)
{
    // Guard bits cover authoritative reuse (800 bits) plus scale rounding and
    // raster dimensions. Explicit mpf_set avoids copying full center precision
    // back into every low-precision working value through operator=.
    const auto precision = static_cast<uint64_t>(std::ceil(schedule.DepthAt(index))) +
                           AuthoritativeReuseExtraPrecisionInBits + 128;
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(precision);
    HighPrecision x{HighPrecision::SetPrecision::True, precision};
    HighPrecision y{HighPrecision::SetPrecision::True, precision};
    mpf_set(x.backend(), target.CenterX.backend());
    mpf_set(y.backend(), target.CenterY.backend());
    PointZoomBBConverter view(
        std::move(x), std::move(y), schedule.ZoomAt(index), PointZoomBBConverter::TestMode::Disabled);
    fractal.SetOfflineView(view);
    fractal.SetNumIterations<uint64_t>(schedule.IterationsAt(index));
}
} // namespace

static int
RunMovieJob(const std::filesystem::path &directory,
            const std::filesystem::path &configuration,
            const std::filesystem::path &ffmpeg,
            GpuMode gpuMode,
            uint64_t commitCapBytes,
            const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings)
{
    MovieSignals signals;
    const auto stopped = [] { return StopRequested.load(std::memory_order_relaxed); };
    MovieJob job(directory, configuration);
    const auto encoder = job.Encoder(ffmpeg);
    auto settings = buildSettings(job.Config());
    const auto &target = settings.Request;
    std::unique_ptr<MovieVideo> video;
    if (!encoder.empty()) {
        video = std::make_unique<MovieVideo>(encoder,
                                             static_cast<unsigned>(target.Width),
                                             static_cast<unsigned>(target.Height),
                                             job.Config().Fps,
                                             stopped);
        video->Preflight(job.Directory());
    }
    // Existing native autosave/load uses the working directory. Set it exactly
    // once, before constructing Fractal or starting renderer workers, with other
    // paths already absolute. Native save paths and flush behavior stay intact.
    std::filesystem::current_path(job.Directory());
    Fractal fractal(target.Width, target.Height, nullptr, false, commitCapBytes, true, gpuMode);
    fractal.SetIterType(IterTypeEnum::Bits64);
    if (target.Perturbation) {
        fractal.SetPerturbationAlg(*target.Perturbation);
    }
    if (!fractal.SetRenderAlgorithm(target.Algorithm)) {
        throw FractalSharkSeriousException("Movie render algorithm is unavailable");
    }
    if (!settings.Palette.empty()) {
        fractal.LoadCustomPalette(settings.Palette);
    }
    if (settings.PaletteDepth) {
        fractal.UsePalette(static_cast<int>(*settings.PaletteDepth));
    }
    fractal.ResetDimensions(static_cast<size_t>(target.Width),
                            static_cast<size_t>(target.Height),
                            target.Antialiasing == 0 ? UINT32_MAX : target.Antialiasing);
    const auto targetDepth = MovieSchedule::Log2(target.Zoom);
    const auto resolvedLaDepth = ResolveLaDepth(target.Algorithm, fractal.GpuBypassed(), targetDepth);
    MovieSchedule provisional(target.Zoom,
                              job.Config().Values.at("zoom-per-frame"),
                              job.Config().WideIterations,
                              target.Iterations,
                              resolvedLaDepth,
                              job.Config().SlopeRatio);
    job.InitializeProgress(provisional.FrameCount(), fractal.GpuBypassed(), resolvedLaDepth);
    MovieSchedule schedule(target.Zoom,
                           job.Config().Values.at("zoom-per-frame"),
                           job.Config().WideIterations,
                           target.Iterations,
                           job.LaDepth(),
                           job.Config().SlopeRatio);
    fractal.SetResultsAutosave(AddPointOptions::EnableWithSave);
    // A loaded native cache is an optimization, not a durability prerequisite
    // for movie progress. Normal Fractal destruction retains its old semantics.
    // Construction already loads native caches. Either explicit MT9 choice clears
    // that cache, so restore the movie's intentionally selected native resume data.
    if (target.Perturbation == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba ||
        target.Perturbation == RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision) {
        fractal.LoadPerturbationOrbits();
    }
    std::future<MovieChunk> encoding;
    const auto finishEncoding = [&] {
        if (encoding.valid()) {
            const auto chunk = encoding.get();
            job.CommitChunk(chunk);
            job.DeleteChunkPngs(chunk);
            job.WriteConcatManifest(schedule.FrameCount());
        }
    };
    const auto queueEncoding = [&](uint64_t first, uint64_t count) {
        finishEncoding();
        if (video && !stopped()) {
            encoding = std::async(std::launch::async,
                                  [&, first, count] { return video->Archive(job, first, count); });
        }
    };
    // Reconcile already completed groups before rendering more. This also
    // compacts a previously PNG-only job without staging all file paths in RAM.
    const auto chunkFrames = job.Config().ChunkFrames;
    for (uint64_t end = schedule.FrameCount(); end != 0 && !stopped();) {
        const auto first = ((end - 1) / chunkFrames) * chunkFrames;
        if (const auto chunk = job.FindChunk(first)) {
            job.DeleteChunkPngs(*chunk);
            job.SkipArchivedChunk(*chunk);
        } else {
            bool complete = true;
            const auto completedBegin = std::max(first, job.Remaining());
            for (uint64_t index = end; index-- > completedBegin;) {
                if (!std::filesystem::exists(job.FramePath(index))) {
                    job.Rewind(index + 1);
                    complete = false;
                    break;
                }
            }
            if (complete && first >= job.Remaining() && video) {
                queueEncoding(first, end - first);
            }
        }
        end = first;
    }
    struct PendingPng {
        uint64_t Index;
        std::future<SaveResult> Completion;
    };
    std::deque<PendingPng> pending;
    const auto finishPng = [&] {
        auto &frame = pending.front();
        const auto result = frame.Completion.get();
        if (!result.Succeeded()) {
            throw FractalSharkSeriousException("Movie PNG save failed: " + result.Diagnostic);
        }
        Environment::FlushMovieFile(job.FramePath(frame.Index));
        job.CompleteFrame(frame.Index);
        pending.pop_front();
    };
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie frames=" << schedule.FrameCount() << " remaining=" << job.Remaining()
        << " video=" << (video ? "AV1 RGB8 lossless" : "disabled");
    while (job.Remaining() != 0 && !stopped()) {
        const auto first = ((job.Remaining() - 1) / chunkFrames) * chunkFrames;
        const auto count = std::min(chunkFrames, schedule.FrameCount() - first);
        if (const auto chunk = job.FindChunk(first)) {
            job.SkipArchivedChunk(*chunk);
            continue;
        }
        auto next = job.Remaining();
        while (next > first && !stopped()) {
            if (encoding.valid() &&
                encoding.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                finishEncoding();
            }
            while (!pending.empty() && (pending.size() >= 4 ||
                                        pending.front().Completion.wait_for(std::chrono::seconds(0)) ==
                                            std::future_status::ready)) {
                finishPng();
            }
            const auto index = --next;
            const auto path = job.FramePath(index);
            std::filesystem::create_directories(path.parent_path());
            if (std::filesystem::exists(path)) {
                // A complete orphan can be adopted after full CRC/dimension
                // validation. Drain earlier saves to preserve the frontier.
                while (!pending.empty()) {
                    finishPng();
                }
                bool valid = false;
                try {
                    ReadMovieRgb8(
                        path, static_cast<unsigned>(target.Width), static_cast<unsigned>(target.Height));
                    valid = true;
                } catch (const FractalSharkSeriousException &exception) {
                    // An incomplete orphan is recoverable: its frame has never
                    // advanced the checkpoint, so regenerate just that output.
                    FractalSharkLog::WriteException(
                        "Regenerating incomplete movie PNG", exception, __FILE__, __LINE__);
                    std::filesystem::remove(path);
                }
                if (valid) {
                    Environment::FlushMovieFile(path);
                    job.CompleteFrame(index);
                    continue;
                }
            }
            SetMovieFrame(fractal, target, schedule, index);
            fractal.CalcFractal(true);
            pending.push_back(
                {index,
                 fractal.SaveCurrentFractalWithResult(
                     Environment::FromFsPath(path), false, SaveOverwritePolicy::SkipExisting)});
        }
        while (!pending.empty()) {
            finishPng();
        }
        if (job.Remaining() == first && video && !stopped()) {
            queueEncoding(first, count);
        }
    }
    finishEncoding();
    fractal.CleanupThreads(true);
    job.WriteConcatManifest(schedule.FrameCount());
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << (stopped() ? "Movie paused; rerun --movie to resume" : "Movie complete");
    return stopped() ? 130 : 0;
}

int
RunMovie(const std::filesystem::path &directory,
         const std::filesystem::path &configuration,
         const std::filesystem::path &ffmpeg,
         GpuMode gpuMode,
         uint64_t commitCapBytes,
         const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings)
{
    try {
        return RunMovieJob(directory, configuration, ffmpeg, gpuMode, commitCapBytes, buildSettings);
    } catch (const MovieInterrupted &) {
        FractalSharkLog::Write("Movie paused; rerun --movie to resume", __FILE__, __LINE__);
        return 130;
    } catch (const std::exception &exception) {
        // Standalone CLI boundary: all workers and file owners have unwound.
        // Report a failed job without advancing progress or discarding PNGs.
        FractalSharkLog::WriteException("Movie failed", exception, __FILE__, __LINE__);
        return 1;
    }
}
} // namespace FractalSharkCli
