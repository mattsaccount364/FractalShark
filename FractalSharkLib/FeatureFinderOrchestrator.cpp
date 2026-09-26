#include "stdafx.h"

#include "FeatureFinderOrchestrator.h"

#include "ConsoleLog.h"
#include "Fractal.h"

#include "BenchmarkData.h"
#include "Exceptions.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "PerturbationResults.h"
#include "PerturbationResultsHelpers.h"
#include "PointZoomBBConverter.h"
#include "RenderAlgorithm.h"
#include "WaitCursor.h"

FeatureFinderOrchestrator::FeatureFinderOrchestrator(Fractal &fractal) : m_Fractal{fractal} {}

void
FeatureFinderOrchestrator::TryFindPeriodicPoint(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    Environment::WaitCursor waitCursor;

    if (m_Fractal.GetIterType() == IterTypeEnum::Bits32) {
        TryFindPeriodicPointIterType<uint32_t>(scrnX, scrnY, mode);
    } else {
        TryFindPeriodicPointIterType<uint64_t>(scrnX, scrnY, mode);
    }
}

template <typename IterType>
void
FeatureFinderOrchestrator::TryFindPeriodicPointIterType(size_t scrnX,
                                                        size_t scrnY,
                                                        FeatureFinderMode mode)
{
    // Each group preserves the existing Direct scalar. PT and LA promote it to
    // the matching HDR type in TryFindPeriodicPointTemplate.
    switch (m_Fractal.GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::Gpu1x32:
        case RenderAlgorithmEnum::Gpu1x32PerturbedScaled:
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32>>(
                scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64:
        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
        case RenderAlgorithmEnum::Gpu1x64:
        case RenderAlgorithmEnum::Gpu1x64PerturbedBLA:
        case RenderAlgorithmEnum::Gpu2x32PerturbedScaled:
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64>>(
                scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::CpuHDR32:
        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
        case RenderAlgorithmEnum::GpuHDRx32:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedScaled:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedBLA:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32>>(
                scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
        case RenderAlgorithmEnum::CpuHDR64:
        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedBLA:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64>>(
                scrnX, scrnY, mode);
            break;
        default:
            throw FractalSharkSeriousException(
                "Current render algorithm does not support feature finding.");
    }
}

template <typename IterType, typename RenderAlg>
void
FeatureFinderOrchestrator::TryFindPeriodicPointTemplate(size_t scrnX,
                                                        size_t scrnY,
                                                        FeatureFinderMode mode)
{
    using DirectT = RenderAlg::MainType;
    using SubType = RenderAlg::SubType;
    const bool direct = mode == FeatureFinderMode::Direct || mode == FeatureFinderMode::DirectScan;
    if (direct) {
        TryFindPeriodicPointSearch<IterType, DirectT, PerturbExtras::Disable>(scrnX, scrnY, mode);
        return;
    }

    // The render type determines mantissa precision, but not the exponent range
    // needed by a search that can converge on a deeper feature.
    using SearchT = HDRFloat<SubType>;
    if (m_Fractal.GetRenderAlgorithm().RequiresCompression) {
        TryFindPeriodicPointSearch<IterType, SearchT, PerturbExtras::SimpleCompression>(
            scrnX, scrnY, mode);
    } else {
        TryFindPeriodicPointSearch<IterType, SearchT, PerturbExtras::Disable>(scrnX, scrnY, mode);
    }
}

template <typename IterType, typename T, PerturbExtras PExtras>
void
FeatureFinderOrchestrator::TryFindPeriodicPointSearch(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    ScopedBenchmarkStopper stopper(m_Fractal.m_BenchmarkData.m_FeatureFinder);
    using SubType = typename TemplateHelpers<IterType, T, PExtras>::SubType;

    auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtras>>();
    m_FeatureSummaries.clear();

    const bool scan = mode == FeatureFinderMode::DirectScan || mode == FeatureFinderMode::PTScan ||
                      mode == FeatureFinderMode::LAScan;
    FeatureFinderMode baseMode = mode;
    if (mode == FeatureFinderMode::DirectScan)
        baseMode = FeatureFinderMode::Direct;
    if (mode == FeatureFinderMode::PTScan)
        baseMode = FeatureFinderMode::PT;
    if (mode == FeatureFinderMode::LAScan)
        baseMode = FeatureFinderMode::LA;

    HighPrecision radius = m_Fractal.GetMaxY() - m_Fractal.GetMinY();
    radius /= HighPrecision{24};

    PerturbationResults<IterType, T, PExtras> *results = nullptr;
    std::unique_ptr<RuntimeDecompressor<IterType, T, PExtras>> decompressor;
    if (baseMode == FeatureFinderMode::PT || baseMode == FeatureFinderMode::LA) {
        if (baseMode == FeatureFinderMode::LA) {
            results = m_Fractal.m_RefOrbit
                          .GetAndCreateUsefulPerturbationResults<IterType,
                                                                 T,
                                                                 SubType,
                                                                 PExtras,
                                                                 RefOrbitCalc::Extras::IncludeLAv2>(
                              m_Fractal.m_Ptz);
        } else {
            results =
                m_Fractal.m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                           T,
                                                                           SubType,
                                                                           PExtras,
                                                                           RefOrbitCalc::Extras::None>(
                    m_Fractal.m_Ptz);
        }
        decompressor = std::make_unique<RuntimeDecompressor<IterType, T, PExtras>>(*results);
    }

    auto runOne = [&](size_t px, size_t py) {
        const HighPrecision cx = m_Fractal.XFromScreenToCalc(HighPrecision(px));
        const HighPrecision cy = m_Fractal.YFromScreenToCalc(HighPrecision(py));
        auto feature = std::make_unique<FeatureSummary>(cx, cy, radius, baseMode);

        bool found = false;
        if (baseMode == FeatureFinderMode::Direct) {
            found = featureFinder->FindPeriodicPoint(m_Fractal.GetNumIterations<IterType>(), *feature);
        } else if (baseMode == FeatureFinderMode::PT) {
            found = featureFinder->FindPeriodicPoint(
                m_Fractal.GetNumIterations<IterType>(), *results, *decompressor, *feature);
        } else if (results->GetLaReference() != nullptr) {
            found = featureFinder->FindPeriodicPoint(m_Fractal.GetNumIterations<IterType>(),
                                                     *results,
                                                     *decompressor,
                                                     *results->GetLaReference(),
                                                     *feature);
        } else {
            found = featureFinder->FindPeriodicPoint(
                m_Fractal.GetNumIterations<IterType>(), *results, *decompressor, *feature);
        }

        if (found) {
            feature->SetNumIterationsAtFind(m_Fractal.GetNumIterations<IterTypeFull>());
            m_FeatureSummaries.emplace_back(std::move(feature));
        }
    };

    if (!scan) {
        runOne(scrnX, scrnY);
    } else {
        constexpr size_t NX = 12;
        constexpr size_t NY = 12;
        const size_t width = m_Fractal.GetRenderWidth();
        const size_t height = m_Fractal.GetRenderHeight();
        for (size_t gy = 0; gy < NY; ++gy) {
            const size_t y = (height * (2 * gy + 1)) / (2 * NY);
            for (size_t gx = 0; gx < NX; ++gx) {
                const size_t x = (width * (2 * gx + 1)) / (2 * NX);
                runOne(x, y);
            }
        }
    }

    if (m_FeatureSummaries.empty())
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "No periodic points found.";
    else
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Found " << m_FeatureSummaries.size() << " periodic points.";
}

void
FeatureFinderOrchestrator::ClearAllFoundFeatures()
{
    m_FeatureSummaries.clear();
    m_Fractal.m_ChangedWindow = true;
}

bool
FeatureFinderOrchestrator::ZoomToFoundFeature(FeatureSummary &feature,
                                              const HighPrecision *zoomFactor,
                                              NRCheckpointSavePolicy checkpointSavePolicy)
{
    // If we only have a candidate, refine now
    if (feature.HasCandidate()) {
        ScopedBenchmarkStopper stopper(m_Fractal.m_BenchmarkData.m_FeatureFinderHP);

        using T = HDRFloat<double>;
        using IterType = uint64_t;
        constexpr PerturbExtras PExtras = PerturbExtras::Disable;

        auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtras>>();

        if (!featureFinder->RefinePeriodicPoint_HighPrecision(
                feature, m_NRInnerLoopBackend, checkpointSavePolicy)) {
            return false;
        }

        if (m_Fractal.GetStopCalculating()) {
            return false;
        }
    }

    const size_t featurePrec = feature.GetPrecision();
    if (featurePrec > m_Fractal.GetPrecision()) {
        m_Fractal.SetPrecision(featurePrec);
    }

    if (zoomFactor) {
        const HighPrecision ptX = feature.GetFoundX();
        const HighPrecision ptY = feature.GetFoundY();

        PointZoomBBConverter ptz(ptX, ptY, *zoomFactor, PointZoomBBConverter::TestMode::Enabled);
        if (ptz.Degenerate())
            return false;

        return m_Fractal.RecenterViewCalc(ptz);
    }

    return true;
}

FeatureSummary *
FeatureFinderOrchestrator::ChooseClosestFeatureToScreenPoint(int clientX, int clientY) const
{
    if (m_FeatureSummaries.empty())
        return nullptr;

    struct {
        long x;
        long y;
    } pt{};
    pt.x = clientX;
    pt.y = clientY;

    // Clamp to render bounds
    const int w = (int)m_Fractal.GetRenderWidth();
    const int h = (int)m_Fractal.GetRenderHeight();
    if (w <= 0 || h <= 0)
        return nullptr;

    if (pt.x < 0)
        pt.x = 0;
    if (pt.y < 0)
        pt.y = 0;
    if (pt.x >= w)
        pt.x = w - 1;
    if (pt.y >= h)
        pt.y = h - 1;

    // Convert mouse -> calc coords (same space as found feature coords)
    const HighPrecision mx = m_Fractal.XFromScreenToCalc(HighPrecision{(int64_t)pt.x});
    const HighPrecision my = m_Fractal.YFromScreenToCalc(HighPrecision{(int64_t)pt.y});

    FeatureSummary *best = nullptr;
    HighPrecision bestDist2{};
    bool haveBest = false;

    for (auto &fsPtr : m_FeatureSummaries) {
        if (!fsPtr)
            continue;

        FeatureSummary &fs = *fsPtr;

        const HighPrecision dx = fs.GetFoundX() - mx;
        const HighPrecision dy = fs.GetFoundY() - my;
        const HighPrecision dist2 = dx * dx + dy * dy;

        if (!haveBest || dist2 < bestDist2) {
            best = &fs;
            bestDist2 = dist2;
            haveBest = true;
        }
    }

    return best;
}

bool
FeatureFinderOrchestrator::ZoomToFoundFeature(int clientX,
                                              int clientY,
                                              NRCheckpointSavePolicy checkpointSavePolicy)
{
    Environment::WaitCursor waitCursor;

    FeatureSummary *best = ChooseClosestFeatureToScreenPoint(clientX, clientY);
    if (!best) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "No feature found to zoom to.";
        return false;
    }

    const HighPrecision z = best->ComputeZoomFactor(m_Fractal.m_Ptz);
    return ZoomToFoundFeature(*best, &z, checkpointSavePolicy);
}

bool
FeatureFinderOrchestrator::ResumeFromCheckpoint()
{
    Environment::WaitCursor waitCursor;

    NRCheckpointData ckpt;
    if (!ReadFullNRCheckpoint(ckpt)) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "No valid NR checkpoint found.";
        return false;
    }

    const bool checkpointComplete = ckpt.phase == NRCheckpointPhase::Complete;
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Resuming NR refinement: period=" << ckpt.period << " prec(bits)=" << ckpt.coord_prec
        << " iter=" << ckpt.iteration << " phase=" << NRCheckpointPhaseName(ckpt.phase)
        << " innerIter=" << ckpt.innerIteration
        << (checkpointComplete ? "; complete checkpoint uses c_* coordinates."
                               : "; feature seed uses cand_* coordinates.");

    // Keep resumed radius invariants separate: constructor/search radius is linear,
    // candidate.sqrRadius_hp is squared for Phase B, and intrinsicRadius drives final zoom.
    const HighPrecision &featureX = checkpointComplete ? ckpt.c_re : ckpt.cand_re;
    const HighPrecision &featureY = checkpointComplete ? ckpt.c_im : ckpt.cand_im;
    auto fs = std::make_unique<FeatureSummary>(
        featureX, featureY, ckpt.intrinsicRadius, FeatureFinderMode::Direct);

    HDRFloat<double> residual2{}; // not stored in checkpoint, use default

    if (!checkpointComplete) {
        fs->SetCandidate(ckpt.cand_re,
                         ckpt.cand_im,
                         (IterTypeFull)ckpt.period,
                         residual2,
                         ckpt.sqrRadius,
                         ckpt.scaleExp2,
                         ckpt.coord_prec);
    }

    // Set intrinsic radius so ComputeZoomFactor works correctly
    fs->SetFound(featureX, featureY, (IterTypeFull)ckpt.period, residual2, ckpt.intrinsicRadius);
    if (checkpointComplete) {
        fs->SetRefined();
    }
    fs->SetNumIterationsAtFind(ckpt.numIterationsAtFind);

    // Save current iteration settings so we can restore them if NR fails/aborts.
    const auto savedIterType = m_Fractal.GetIterType();
    const auto savedNumIters = m_Fractal.GetNumIterations<IterTypeFull>();

    // Restore the iteration limit that was active when the feature was found.
    // Switch to 64-bit iteration type if the saved count exceeds 32-bit max.
    if (ckpt.numIterationsAtFind > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        m_Fractal.SetIterType(IterTypeEnum::Bits64);
    }
    m_Fractal.SetNumIterations<IterTypeFull>(ckpt.numIterationsAtFind);

    m_FeatureSummaries.clear();
    m_FeatureSummaries.emplace_back(std::move(fs));

    // Non-complete checkpoints still refine through the checkpoint-aware NR path.
    // Complete checkpoints have no candidate, so ZoomToFoundFeature only recenters and renders.
    auto *feature = m_FeatureSummaries.back().get();
    const HighPrecision z = feature->ComputeZoomFactor(m_Fractal.m_Ptz);
    const bool ok = ZoomToFoundFeature(*feature, &z);

    if (!ok) {
        // NR aborted or failed — restore iteration settings to pre-resume values.
        m_Fractal.SetIterType(savedIterType);
        m_Fractal.SetNumIterations<IterTypeFull>(savedNumIters);
    }

    return ok;
}
