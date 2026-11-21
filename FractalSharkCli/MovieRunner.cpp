#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieRunner.h"
#include "MovieVideo.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <thread>

namespace FractalSharkCli {
namespace {
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
    Environment::MovieInterruptScope signals;
    const auto stopped = [&] { return signals.StopRequested(); };
    std::atomic<bool> cancelArchive{false};
    const auto archiveStopped = [&] {
        return stopped() || cancelArchive.load(std::memory_order_relaxed);
    };
    MovieJob job(directory, configuration);
    const auto encoder = job.Encoder(ffmpeg);
    auto settings = buildSettings(job.Config());
    const auto &target = settings.Request;
    std::vector<std::unique_ptr<MovieVideo>> videos;
    if (!encoder.empty()) {
        const auto execution = PlanMovieExecution(static_cast<unsigned>(target.Width),
                                                  static_cast<unsigned>(target.Height),
                                                  Environment::LogicalProcessorCount());
        for (const auto &slot : execution) {
            videos.push_back(std::make_unique<MovieVideo>(encoder,
                                                          static_cast<unsigned>(target.Width),
                                                          static_cast<unsigned>(target.Height),
                                                          job.Config().Fps,
                                                          slot,
                                                          archiveStopped));
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie archive slot=" << slot.Slot << " decoder_budget=" << slot.DecoderThreads
                << " filter_threads=" << slot.FilterThreads << " encoder_threads=" << slot.EncoderThreads
                << " tiles=" << slot.Tiles.Columns << 'x' << slot.Tiles.Rows;
        }
        videos.front()->Preflight(job.Directory());
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
    struct ReadyChunk {
        uint64_t First;
        uint64_t Count;
    };
    // Workers read immutable job paths. Only this thread publishes completion
    // records, deletes PNGs, and updates the ordered assembly manifest.
    std::deque<ReadyChunk> readyChunks;
    std::vector<std::future<MovieChunk>> encoding(videos.size());
    struct ArchiveLifetime {
        std::atomic<bool> &Cancelled;
        ~ArchiveLifetime() { Cancelled.store(true, std::memory_order_relaxed); }
    } archiveLifetime{cancelArchive};
    // Declared after the futures so exceptional exits cancel all processes
    // before future destructors join and before the job lock closes.
    std::exception_ptr archiveFailure;
    const auto activeArchives = [&] {
        return std::count_if(
            encoding.begin(), encoding.end(), [](const auto &future) { return future.valid(); });
    };
    const auto serviceEncoding = [&] {
        for (auto &future : encoding) {
            if (!future.valid() ||
                future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                continue;
            }
            try {
                const auto chunk = future.get();
                job.CommitChunk(chunk);
                job.DeleteChunkPngs(chunk);
                job.WriteConcatManifest(schedule.FrameCount());
            } catch (const MovieInterrupted &) {
                // Cancellation must not skip the pending PNG checkpoint
                // drain. An unsolicited interruption remains an error.
                if (!archiveStopped() && !archiveFailure) {
                    archiveFailure = std::current_exception();
                    cancelArchive.store(true, std::memory_order_relaxed);
                }
            } catch (...) {
                // Preserve the first failure while joining EVERY worker and
                // checkpointing successful PNG saves at the render boundary.
                if (!archiveFailure) {
                    archiveFailure = std::current_exception();
                }
                cancelArchive.store(true, std::memory_order_relaxed);
            }
        }
        if (stopped() || archiveFailure) {
            return;
        }
        for (size_t slot = 0; slot < encoding.size() && !readyChunks.empty(); ++slot) {
            if (encoding[slot].valid()) {
                continue;
            }
            const auto range = readyChunks.front();
            // Native ownership and scheduling priority remain in Environment.
            try {
                encoding[slot] = std::async(std::launch::async, [&, slot, range] {
                    return videos[slot]->Archive(job, range.First, range.Count);
                });
            } catch (...) {
                // Thread creation can fail too; retain the same PNG-drain
                // contract as a subprocess failure rather than unwinding now.
                archiveFailure = std::current_exception();
                cancelArchive.store(true, std::memory_order_relaxed);
                return;
            }
            readyChunks.pop_front();
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie archive started slot=" << slot << " chunk=" << range.First
                << " active_workers=" << activeArchives() << " worker_limit=" << videos.size()
                << " queued_chunks=" << readyChunks.size();
        }
    };
    constexpr auto ArchivePollInterval = std::chrono::milliseconds(50);
    const auto waitForArchiveCapacity = [&] {
        const auto started = std::chrono::steady_clock::now();
        const bool waiting = !readyChunks.empty() && !stopped() && !archiveFailure;
        if (waiting) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie waiting for archive capacity active_workers=" << activeArchives()
                << " worker_limit=" << videos.size() << " queued_chunks=" << readyChunks.size()
                << " remaining=" << job.Remaining();
        }
        // Recovery never calls this wait. After one new chunk fills, wait for
        // its FIFO entry to start before collecting another. Poll all workers:
        // a slow oldest chunk must not hide a free slot or a later failure.
        while (!readyChunks.empty() && !stopped() && !archiveFailure) {
            serviceEncoding();
            if (!readyChunks.empty() && !stopped() && !archiveFailure) {
                std::this_thread::sleep_for(ArchivePollInterval);
            }
        }
        if (waiting) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie archive_wait_seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        }
    };
    // Reconcile metadata and enqueue recovered chunks without waiting for any
    // encoder. The next missing frame must not sit behind a second queue entry.
    const auto chunkFrames = job.Config().ChunkFrames;
    for (uint64_t end = schedule.FrameCount(); end != 0 && !stopped();) {
        const auto first = ((end - 1) / chunkFrames) * chunkFrames;
        if (const auto chunk = job.FindChunk(first)) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie reusing archived chunk=" << first << " frames=" << chunk->Count;
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
            if (complete && first >= job.Remaining() && !videos.empty()) {
                readyChunks.push_back({first, end - first});
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
        << "Movie frames=" << schedule.FrameCount()
        << " rendered=" << schedule.FrameCount() - job.Remaining() << " remaining=" << job.Remaining()
        << " next_frame=" << (job.Remaining() ? std::to_string(job.Remaining() - 1) : "none")
        << " ready_chunks=" << readyChunks.size() << " archive_workers=" << videos.size()
        << " video=" << (!videos.empty() ? "AV1 RGB8 lossless" : "disabled");
    serviceEncoding();
    while (job.Remaining() != 0 && !stopped() && !archiveFailure) {
        const auto first = ((job.Remaining() - 1) / chunkFrames) * chunkFrames;
        const auto count = std::min(chunkFrames, schedule.FrameCount() - first);
        if (const auto chunk = job.FindChunk(first)) {
            job.SkipArchivedChunk(*chunk);
            continue;
        }
        auto next = job.Remaining();
        while (next > first && !stopped() && !archiveFailure) {
            serviceEncoding();
            if (stopped() || archiveFailure) {
                break;
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
                    FractalSharkLog::LogLine(__FILE__, __LINE__) << "Movie reusing PNG frame=" << index;
                    Environment::FlushMovieFile(path);
                    job.CompleteFrame(index);
                    continue;
                }
            }
            SetMovieFrame(fractal, target, schedule, index);
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie rendering " << schedule.FrameCount() - index << '/' << schedule.FrameCount()
                << " frame=" << index << " iterations=" << schedule.IterationsAt(index);
            fractal.CalcFractal(true);
            pending.push_back(
                {index,
                 fractal.SaveCurrentFractalWithResult(
                     Environment::FromFsPath(path), false, SaveOverwritePolicy::SkipExisting)});
        }
        while (!pending.empty()) {
            finishPng();
        }
        if (job.Remaining() == first && !videos.empty() && !stopped() && !archiveFailure) {
            readyChunks.push_back({first, count});
            serviceEncoding();
            waitForArchiveCapacity();
        }
    }
    // A completed PNG-only job can enter with no rendering left and many
    // recovered ranges. Drain every queued archive before reporting completion.
    while (!stopped() && !archiveFailure && (activeArchives() != 0 || !readyChunks.empty())) {
        serviceEncoding();
        if (activeArchives() != 0) {
            std::this_thread::sleep_for(ArchivePollInterval);
        }
    }
    // Stop/failure prevents launches, but all owned workers must finish before
    // releasing the lock or reporting the saved frontier.
    while (activeArchives() != 0) {
        serviceEncoding();
        if (activeArchives() != 0) {
            std::this_thread::sleep_for(ArchivePollInterval);
        }
    }
    if (archiveFailure) {
        std::rethrow_exception(archiveFailure);
    }
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
