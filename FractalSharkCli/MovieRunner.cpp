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

RenderAlgorithmEnum
MovieReferenceType(const RenderAlgorithm &algorithm)
{
    // Auto's perturbation-only and full LA float modes consume the same
    // uncompressed float reference. Prepare it at the deepest of both ranges.
    if (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO) {
        return RenderAlgorithmEnum::Gpu1x32PerturbedLAv2;
    }
    return algorithm.Algorithm;
}

void
PrepareMovieReferences(Fractal &fractal,
                       const RenderRequest &target,
                       const MovieSchedule &schedule,
                       const MovieJob &job,
                       const auto &stopped)
{
    // Search in increasing zoom order even for a deep-to-wide job. The
    // effective view and selector must match rendering, including aspect ratio
    // adjustments and CPU fallback at the transition frames.
    const auto frameCount = schedule.FrameCount();
    const bool wideToDeep = job.Config().WideToDeep;
    const auto frameAtDepth = [&](uint64_t depthIndex) {
        return wideToDeep ? depthIndex : frameCount - 1 - depthIndex;
    };
    const auto firstDepth = wideToDeep ? job.NextFrame() : 0;
    auto endDepth = wideToDeep ? frameCount : frameCount - job.NextFrame();
    const auto chunkFrames = job.Config().ChunkFrames;
    while (firstDepth < endDepth && !stopped()) {
        const auto anchorFrame = frameAtDepth(endDepth - 1);
        SetMovieFrame(fractal, target, schedule, anchorFrame);
        const auto algorithm = fractal.GetRenderAlgorithm();
        if (!algorithm.RequiresReferencePoints) {
            // Auto's direct renderers occupy the remaining wider views.
            break;
        }

        auto startDepth = firstDepth;
        if (target.Algorithm == RenderAlgorithmEnum::AUTO) {
            const auto referenceType = MovieReferenceType(algorithm);
            auto right = endDepth - 1;
            while (startDepth < right && !stopped()) {
                const auto middle = startDepth + (right - startDepth) / 2;
                SetMovieFrame(fractal, target, schedule, frameAtDepth(middle));
                if (MovieReferenceType(fractal.GetRenderAlgorithm()) == referenceType) {
                    right = middle;
                } else {
                    startDepth = middle + 1;
                }
            }
        }

        // Some chunks beyond the resume frontier can already be archived.
        // Skip preparation when the entire range for this type is covered.
        const auto otherFrame = frameAtDepth(startDepth);
        const auto first = std::min(anchorFrame, otherFrame);
        const auto last = std::max(anchorFrame, otherFrame);
        for (auto chunk = (first / chunkFrames) * chunkFrames; chunk <= last && !stopped();
             chunk += chunkFrames) {
            if (!job.FindChunk(chunk)) {
                SetMovieFrame(fractal, target, schedule, anchorFrame);
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Movie preparing reference frame=" << anchorFrame
                    << " algorithm=" << algorithm.AlgorithmStr;
                fractal.PrepareReferenceForCurrentView();
                break;
            }
        }
        endDepth = startDepth;
    }
}
} // namespace

static int
RunMovieJob(MovieJob &job,
            Environment::MovieInterruptScope &signals,
            const std::filesystem::path &ffmpeg,
            GpuMode gpuMode,
            uint64_t commitCapBytes,
            const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings)
{
    const auto stopped = [&] { return signals.StopRequested(); };
    std::atomic<bool> cancelArchive{false};
    const auto archiveStopped = [&] {
        return stopped() || cancelArchive.load(std::memory_order_relaxed);
    };
    // Complete archives can be joined on a machine without the original GPU.
    // The journal also identifies consumed sources before render recovery starts.
    const auto assembly = job.ReadAssembly();
    const auto encoder = assembly && assembly->NextChunk == job.ChunkCount() ? std::filesystem::path{}
                                                                             : job.Encoder(ffmpeg);
    if (assembly || (!encoder.empty() && job.HasAllChunks())) {
        AssembleMovie(job, encoder, stopped);
        return 0;
    }
    auto settings = buildSettings(job.Config());
    const auto &target = settings.Request;
    std::vector<std::unique_ptr<MovieVideo>> videos;
    if (!encoder.empty()) {
        const auto execution = PlanMovieExecution(Environment::LogicalProcessorCount());
        for (const auto &slot : execution) {
            videos.push_back(std::make_unique<MovieVideo>(encoder,
                                                          static_cast<unsigned>(target.Width),
                                                          static_cast<unsigned>(target.Height),
                                                          job.Config(),
                                                          slot,
                                                          archiveStopped));
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie archive slot=" << slot.Slot << " decoder_budget=" << slot.DecoderThreads
                << " filter_threads=" << slot.FilterThreads << " encoder=hevc_nvenc";
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
                              job.Config().SlopeRatio,
                              job.Config().WideToDeep);
    job.InitializeProgress(provisional.FrameCount(), fractal.GpuBypassed(), resolvedLaDepth);
    MovieSchedule schedule(target.Zoom,
                           job.Config().Values.at("zoom-per-frame"),
                           job.Config().WideIterations,
                           target.Iterations,
                           job.LaDepth(),
                           job.Config().SlopeRatio,
                           job.Config().WideToDeep);
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
        job.CheckTiming();
        for (auto &future : encoding) {
            if (!future.valid() ||
                future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                continue;
            }
            try {
                const auto chunk = future.get();
                job.CommitChunk(chunk);
                job.WriteConcatManifest();
                job.DeleteChunkPngs(chunk);
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
                << "Movie archive started slot=" << slot
                << " chunk=" << range.First / job.Config().ChunkFrames << " first_frame=" << range.First
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
    for (uint64_t first = 0; first < schedule.FrameCount() && !stopped();) {
        const auto count = std::min(chunkFrames, schedule.FrameCount() - first);
        const auto end = first + count;
        if (const auto chunk = job.FindChunk(first)) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie reusing archived chunk=" << first / chunkFrames << " frames=" << chunk->Count;
            job.DeleteChunkPngs(*chunk);
            job.SkipArchivedChunk(*chunk);
        } else {
            bool complete = true;
            const auto completedEnd = std::min(end, job.NextFrame());
            for (uint64_t index = first; index < completedEnd; ++index) {
                if (!std::filesystem::exists(job.FramePath(index))) {
                    job.Rewind(index);
                    complete = false;
                    break;
                }
            }
            if (complete && end <= job.NextFrame() && !videos.empty()) {
                readyChunks.push_back({first, count});
            }
        }
        first = end;
    }
    serviceEncoding();
    if (job.Remaining() != 0 && !archiveStopped()) {
        std::packaged_task<void(Fractal &)> prepare([&](Fractal &current) {
            current.ResetStopCalculating();
            PrepareMovieReferences(current, target, schedule, job, archiveStopped);
        });
        auto prepared = prepare.get_future();
        fractal
            .EnqueueMutation("prepare movie references",
                             [&prepare](Fractal &current) { prepare(current); })
            .Wait();
        prepared.get();
    }
    struct PendingPng {
        uint64_t Index;
        std::future<SaveResult> Completion;
    };
    std::deque<PendingPng> pending;
    // Futures and view snapshots may be referenced by render/PNG workers.
    // Drain before those locals or the native reference owner can be destroyed.
    struct RenderLifetime {
        Fractal &Owner;
        std::atomic<bool> &Cancelled;
        ~RenderLifetime()
        {
            Cancelled.store(true, std::memory_order_relaxed);
            Owner.GetRenderPool()->Drain();
            Owner.CleanupThreads(true);
        }
    } renderLifetime{fractal, cancelArchive};
    constexpr size_t MaximumPendingFrames = 2 * NumRenderers;
    const auto finishPng = [&] {
        auto &frame = pending.front();
        while (frame.Completion.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            serviceEncoding();
            std::this_thread::sleep_for(ArchivePollInterval);
        }
        const auto result = frame.Completion.get();
        if (!result.Succeeded() && archiveStopped()) {
            pending.pop_front();
            return;
        }
        if (!result.Succeeded()) {
            throw FractalSharkSeriousException("Movie PNG save failed: " + result.Diagnostic);
        }
        Environment::FlushMovieFile(job.FramePath(frame.Index));
        if (frame.Index == job.NextFrame()) {
            job.CompleteFrame(frame.Index);
        }
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Movie saved frame=" << frame.Index << " pending_render_saves=" << pending.size() - 1;
        pending.pop_front();
    };
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie frames=" << schedule.FrameCount()
        << " rendered=" << schedule.FrameCount() - job.Remaining() << " remaining=" << job.Remaining()
        << " next_frame=" << (job.Remaining() ? std::to_string(job.NextFrame()) : "none")
        << " ready_chunks=" << readyChunks.size() << " archive_workers=" << videos.size()
        << " video=" << (!videos.empty() ? "configured NVENC" : "disabled");
    serviceEncoding();
    while (job.Remaining() != 0 && !stopped() && !archiveFailure) {
        const auto first = (job.NextFrame() / chunkFrames) * chunkFrames;
        const auto count = std::min(chunkFrames, schedule.FrameCount() - first);
        if (const auto chunk = job.FindChunk(first)) {
            job.SkipArchivedChunk(*chunk);
            continue;
        }
        auto next = job.NextFrame();
        while (next < first + count && !stopped() && !archiveFailure) {
            serviceEncoding();
            if (stopped() || archiveFailure) {
                break;
            }
            while (!pending.empty() && (pending.size() >= MaximumPendingFrames ||
                                        pending.front().Completion.wait_for(std::chrono::seconds(0)) ==
                                            std::future_status::ready)) {
                finishPng();
            }
            if (stopped() || archiveFailure) {
                break;
            }
            const auto index = next++;
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
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie submitted " << index + 1 << '/' << schedule.FrameCount() << " frame=" << index
                << " iterations=" << schedule.IterationsAt(index)
                << " pending_render_saves=" << pending.size() + 1 << " renderer_slots=" << NumRenderers;
            pending.push_back({index,
                               fractal.EnqueueRenderToPng(
                                   "render movie frame",
                                   [&target, &schedule, index](Fractal &current) {
                                       SetMovieFrame(current, target, schedule, index);
                                   },
                                   Environment::FromFsPath(path),
                                   SaveOverwritePolicy::SkipExisting,
                                   archiveStopped)});
        }
        while (!pending.empty()) {
            finishPng();
        }
        if (job.NextFrame() == first + count && !videos.empty() && !stopped() && !archiveFailure) {
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
    if (!stopped() && !videos.empty()) {
        AssembleMovie(job, encoder, stopped);
        return 0;
    }
    job.WriteConcatManifest();
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
    std::unique_ptr<Environment::MovieInterruptScope> signals;
    std::unique_ptr<MovieJob> job;
    try {
        signals = std::make_unique<Environment::MovieInterruptScope>();
        job = std::make_unique<MovieJob>(directory, configuration);
        const auto result = RunMovieJob(*job, *signals, ffmpeg, gpuMode, commitCapBytes, buildSettings);
        return job->FinishTiming() ? result : 1;
    } catch (const MovieInterrupted &) {
        FractalSharkLog::Write("Movie paused; rerun --movie to resume", __FILE__, __LINE__);
        return !job || job->FinishTiming() ? 130 : 1;
    } catch (const std::exception &exception) {
        // Standalone CLI boundary: all workers and file owners have unwound.
        // Report failures after all durable progress and sources have settled.
        FractalSharkLog::WriteException("Movie failed", exception, __FILE__, __LINE__);
        if (job) {
            job->FinishTiming();
        }
        return 1;
    }
}
} // namespace FractalSharkCli
