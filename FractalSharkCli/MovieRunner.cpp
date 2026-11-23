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

        if (!stopped()) {
            SetMovieFrame(fractal, target, schedule, anchorFrame);
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie preparing reference frame=" << anchorFrame
                << " algorithm=" << algorithm.AlgorithmStr;
            fractal.PrepareReferenceForCurrentView();
        }
        endDepth = startDepth;
    }
}
} // namespace

static int
RunMovieJob(MovieJob &job,
            Environment::MovieInterruptScope &signals,
            GpuMode gpuMode,
            uint64_t commitCapBytes,
            const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings)
{
    const bool pngOnly = job.Config().Video.Codec == FractalShark::MovieCodec::PngOnly;
    if (!pngOnly && job.FrameCount() != 0 && job.Remaining() == 0) {
        MovieVideo(job).Finish();
        FractalSharkLog::Write("Movie complete", __FILE__, __LINE__);
        return 0;
    }

    const auto settings = buildSettings(job.Config());
    const auto &target = settings.Request;
    // Native reference files use the process working directory. Resolve CLI
    // paths before entering the job and leave it fixed for every render worker.
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
    const auto resolvedLaDepth =
        ResolveLaDepth(target.Algorithm, fractal.GpuBypassed(), MovieSchedule::Log2(target.Zoom));
    const auto &config = job.Config();
    const MovieSchedule provisional(target.Zoom,
                                    config.Values.at("zoom-per-frame"),
                                    config.WideIterations,
                                    target.Iterations,
                                    resolvedLaDepth,
                                    config.SlopeRatio,
                                    config.WideToDeep);
    job.InitializeProgress(provisional.FrameCount(), fractal.GpuBypassed(), resolvedLaDepth);
    const MovieSchedule schedule(target.Zoom,
                                 config.Values.at("zoom-per-frame"),
                                 config.WideIterations,
                                 target.Iterations,
                                 job.LaDepth(),
                                 config.SlopeRatio,
                                 config.WideToDeep);
    fractal.SetResultsAutosave(AddPointOptions::EnableWithSave);
    // Explicit MT9 selection clears the loader's initial reference cache.
    if (target.Perturbation == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba ||
        target.Perturbation == RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision) {
        fractal.LoadPerturbationOrbits();
    }

    std::unique_ptr<MovieVideo> output;
    std::unique_ptr<FractalShark::NvencMovieEncoder> encoder;
    if (pngOnly) {
        for (uint64_t frame = 0; frame < job.NextFrame(); ++frame) {
            if (!std::filesystem::is_regular_file(job.FramePath(frame))) {
                job.Rewind(frame);
                break;
            }
        }
    } else {
        output = std::make_unique<MovieVideo>(job);
        if (!signals.StopRequested()) {
            encoder = std::make_unique<FractalShark::NvencMovieEncoder>(
                static_cast<unsigned>(target.Width), static_cast<unsigned>(target.Height), config.Video);
            if (config.Video.Codec == FractalShark::MovieCodec::Hevc) {
                const auto &rate = config.Video.Hevc;
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Movie HEVC profile=" << config.Values.at("video-profile")
                    << " fps=" << config.Video.Fps << " cq=" << config.Values.at("video-cq")
                    << " tier=" << (rate.Tier == FractalShark::MovieHevcTier::High ? "high" : "main")
                    << " level_idc=" << rate.LevelIdc << " peak_bps=" << rate.PeakBitsPerSecond
                    << " average_bps=" << rate.AverageBitsPerSecond << " vbv_bits=" << rate.BufferBits
                    << " vbv_initial_bits=" << rate.InitialBufferBits;
            }
        }
    }

    std::atomic<bool> cancelled{false};
    const auto stopped = [&] { return signals.StopRequested() || cancelled.load(); };
    // The render requests borrow both the schedule and the encoder's surfaces.
    // Drain them before either object can be destroyed, including on exceptions.
    struct RenderLifetime {
        RenderThreadPool &Pool;
        std::atomic<bool> &Cancelled;
        ~RenderLifetime()
        {
            Cancelled.store(true);
            Pool.Drain();
        }
    } renderLifetime{*fractal.GetRenderPool(), cancelled};

    if (job.Remaining() != 0 && !stopped()) {
        std::packaged_task<void(Fractal &)> prepare([&](Fractal &current) {
            current.ResetStopCalculating();
            PrepareMovieReferences(current, target, schedule, job, stopped);
        });
        auto prepared = prepare.get_future();
        fractal
            .EnqueueMutation("prepare movie references",
                             [&prepare](Fractal &current) { prepare(current); })
            .Wait();
        prepared.get();
    }

    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie frames=" << job.FrameCount() << " durable_frames=" << job.NextFrame()
        << " remaining=" << job.Remaining() << " output=" << (pngOnly ? "PNG" : "NVENC");

    // One worker owns the ordered encode/mux stream. The render pool can
    // populate the other three surfaces while the worker reads a bitstream.
    auto worker = std::async(std::launch::async, [&] {
        struct PendingFrame {
            uint64_t Index;
            std::future<SaveResult> Render;
        };
        std::deque<PendingFrame> pending;
        const auto first = job.NextFrame();
        auto next = first;
        uint64_t completed = 0;
        std::chrono::steady_clock::duration encodeTime{}, writeTime{};
        constexpr auto slots = FractalShark::NvencMovieEncoder::SurfaceCount;
        while (!pending.empty() || (next < job.FrameCount() && !stopped())) {
            while (!stopped() && next < job.FrameCount() && pending.size() < slots) {
                const auto index = next++;
                const auto prepare = [&target, &schedule, index](Fractal &current) {
                    SetMovieFrame(current, target, schedule, index);
                };
                std::future<SaveResult> rendered;
                if (encoder) {
                    rendered = fractal.GetRenderPool()->EnqueueMovieSurface(
                        prepare, *encoder, index % slots, stopped);
                } else {
                    const auto path = job.FramePath(index);
                    std::filesystem::create_directories(path.parent_path());
                    if (std::filesystem::exists(path)) {
                        // An orphan PNG is reusable only after validating it.
                        try {
                            ReadMovieRgb8(path, target.Width, target.Height);
                        } catch (const FractalSharkSeriousException &) {
                            std::filesystem::remove(path);
                        }
                    }
                    if (!std::filesystem::exists(path)) {
                        rendered = fractal.EnqueueRenderToPng("render movie PNG",
                                                              prepare,
                                                              Environment::FromFsPath(path),
                                                              SaveOverwritePolicy::SkipExisting,
                                                              stopped);
                    }
                }
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Movie submitted " << index + 1 << '/' << job.FrameCount() << " frame=" << index
                    << " iterations=" << schedule.IterationsAt(index);
                pending.push_back({index, std::move(rendered)});
            }
            if (pending.empty()) {
                break;
            }
            auto &frame = pending.front();
            if (frame.Render.valid()) {
                while (frame.Render.wait_for(std::chrono::milliseconds(50)) !=
                       std::future_status::ready) {
                    job.CheckTiming();
                    if (output) {
                        output->CheckpointIfDue();
                    }
                }
                const auto result = frame.Render.get();
                if (!result.Succeeded() && stopped()) {
                    break;
                }
                if (!result.Succeeded()) {
                    throw FractalSharkSeriousException("Movie frame failed: " + result.Diagnostic);
                }
            }
            if (encoder) {
                const bool idr = frame.Index == first || frame.Index % config.Video.GopFrames == 0;
                const auto started = std::chrono::steady_clock::now();
                const auto unit = encoder->Encode(frame.Index % slots, frame.Index, idr);
                const auto encoded = std::chrono::steady_clock::now();
                encodeTime += encoded - started;
                output->Append(frame.Index, unit, idr);
                writeTime += std::chrono::steady_clock::now() - encoded;
            } else {
                Environment::FlushMovieFile(job.FramePath(frame.Index));
                job.CompleteFrame(frame.Index);
            }
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie completed frame=" << frame.Index << " durable_frames=" << job.NextFrame();
            pending.pop_front();
            ++completed;
            job.CheckTiming();
        }
        if (output) {
            output->Finish();
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Movie NVENC session frames=" << completed
                << " encode_seconds=" << std::chrono::duration<double>(encodeTime).count()
                << " mux_seconds=" << std::chrono::duration<double>(writeTime).count();
        }
    });
    worker.get();

    FractalSharkLog::Write(
        job.Remaining() == 0 ? "Movie complete" : "Movie paused; rerun --movie to resume",
        __FILE__,
        __LINE__);
    return job.Remaining() == 0 ? 0 : 130;
}

int
RunMovie(const std::filesystem::path &directory,
         const std::filesystem::path &configuration,
         GpuMode gpuMode,
         uint64_t commitCapBytes,
         const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings)
{
    std::unique_ptr<MovieJob> job;
    try {
        Environment::MovieInterruptScope signals;
        job = std::make_unique<MovieJob>(directory, configuration);
        const auto result = RunMovieJob(*job, signals, gpuMode, commitCapBytes, buildSettings);
        return job->FinishTiming() ? result : 1;
    } catch (const std::exception &exception) {
        FractalSharkLog::WriteException("Movie failed", exception, __FILE__, __LINE__);
        if (job) {
            job->FinishTiming();
        }
        return 1;
    }
}
} // namespace FractalSharkCli
