#include "stdafx.h"

#include "FeatureFinderOrchestrator.h"

#include "Fractal.h"

#include "BenchmarkData.h"
#include "Exceptions.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "PerturbationResults.h"
#include "PerturbationResultsHelpers.h"
#include "PointZoomBBConverter.h"
#include "RenderAlgorithm.h"

#include <iostream>

FeatureFinderOrchestrator::FeatureFinderOrchestrator(Fractal &fractal)
    : m_Fractal{fractal} {
}

void
FeatureFinderOrchestrator::TryFindPeriodicPoint(size_t scrnX, size_t scrnY, FeatureFinderMode mode) {
    if (m_Fractal.GetIterType() == IterTypeEnum::Bits32) {
        TryFindPeriodicPointIterType<uint32_t>(scrnX, scrnY, mode);
    } else {
        TryFindPeriodicPointIterType<uint64_t>(scrnX, scrnY, mode);
    }
}

template <typename IterType>
void
FeatureFinderOrchestrator::TryFindPeriodicPointIterType(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    // Note: This accounts for "Auto" being selected via the GetRenderAlgorithm call.
    switch (m_Fractal.GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::CpuHigh:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint. RenderAlgorithmEnum::CpuHigh. ");
            break;

        case RenderAlgorithmEnum::CpuHDR32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAHDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::CpuHDR64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAHDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x64:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu2x64. ");
            break;

        case RenderAlgorithmEnum::Gpu4x64:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu4x64. ");
            break;

        case RenderAlgorithmEnum::Gpu1x32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu2x32. ");
            break;

        case RenderAlgorithmEnum::Gpu4x32:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu4x32. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedScaled:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedScaled>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedScaled:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedScaled>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedScaled:
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedScaled. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

            // LAV2

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        default:
            std::cerr << "Current render algorithm does not support feature finding.\n";
            break;
    }
}

template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
void
FeatureFinderOrchestrator::TryFindPeriodicPointTemplate(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    ScopedBenchmarkStopper stopper(m_Fractal.m_BenchmarkData.m_FeatureFinder);

    using T = RenderAlg::MainType;
    using SubType = RenderAlg::SubType;
    constexpr PerturbExtras PExtrasLocal = PerturbExtras::Disable;

    auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtrasLocal>>();

    m_FeatureSummaries.clear();

    const bool scan = mode == FeatureFinderMode::DirectScan || mode == FeatureFinderMode::PTScan ||
                      mode == FeatureFinderMode::LAScan;

    // Base mode
    FeatureFinderMode baseMode = mode;
    if (mode == FeatureFinderMode::DirectScan)
        baseMode = FeatureFinderMode::Direct;
    if (mode == FeatureFinderMode::PTScan)
        baseMode = FeatureFinderMode::PT;
    if (mode == FeatureFinderMode::LAScan)
        baseMode = FeatureFinderMode::LA;

    const T radiusY{T{m_Fractal.GetMaxY() - m_Fractal.GetMinY()} / T{2.0f}};
    HighPrecision radius{radiusY};
    radius /= HighPrecision{12};

    auto RunOne = [&](size_t px, size_t py) {
        const HighPrecision cx = m_Fractal.XFromScreenToCalc(HighPrecision(px));
        const HighPrecision cy = m_Fractal.YFromScreenToCalc(HighPrecision(py));

        auto fs = std::make_unique<FeatureSummary>(cx, cy, radius, baseMode);

        bool found = false;

        if (baseMode == FeatureFinderMode::Direct) {
            found = featureFinder->FindPeriodicPoint(m_Fractal.GetNumIterations<IterType>(), *fs);
        } else if (baseMode == FeatureFinderMode::PT) {
            auto *results =
                m_Fractal.m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                 T,
                                                                 SubType,
                                                                 PExtrasLocal,
                                                                 RefOrbitCalc::Extras::None>(m_Fractal.m_Ptz);
            RuntimeDecompressor<IterType, T, PExtrasLocal> decompressor(*results);

            found = featureFinder->FindPeriodicPoint(
                m_Fractal.GetNumIterations<IterType>(), *results, decompressor, *fs);
        } else if (baseMode == FeatureFinderMode::LA) {
            auto *results =
                m_Fractal.m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                 T,
                                                                 SubType,
                                                                 PExtrasLocal,
                                                                 RefOrbitCalc::Extras::IncludeLAv2>(
                    m_Fractal.m_Ptz);
            RuntimeDecompressor<IterType, T, PExtrasLocal> decompressor(*results);

            found = featureFinder->FindPeriodicPoint(
                m_Fractal.GetNumIterations<IterType>(), *results, decompressor, *results->GetLaReference(), *fs);
        }

        if (found) {
            fs->SetNumIterationsAtFind(m_Fractal.GetNumIterations<IterTypeFull>());
            m_FeatureSummaries.emplace_back(std::move(fs));
        }
    };

    if (!scan) {
        RunOne(scrnX, scrnY);
    } else {
        constexpr size_t NX = 12;
        constexpr size_t NY = 12;

        const size_t W = m_Fractal.GetRenderWidth();
        const size_t H = m_Fractal.GetRenderHeight();

        for (size_t gy = 0; gy < NY; ++gy) {
            const size_t y = (H * (2 * gy + 1)) / (2 * NY);
            for (size_t gx = 0; gx < NX; ++gx) {
                const size_t x = (W * (2 * gx + 1)) / (2 * NX);
                RunOne(x, y);
            }
        }
    }

    if (m_FeatureSummaries.empty()) {
        std::cout << "No periodic points found.\n";
    } else {
        std::cout << "Found " << m_FeatureSummaries.size() << " periodic points.\n";
    }
}

void
FeatureFinderOrchestrator::ClearAllFoundFeatures()
{
    m_FeatureSummaries.clear();
    m_Fractal.m_ChangedWindow = true;
}

bool
FeatureFinderOrchestrator::ZoomToFoundFeature(FeatureSummary &feature, const HighPrecision *zoomFactor)
{
    // If we only have a candidate, refine now
    if (feature.HasCandidate()) {
        ScopedBenchmarkStopper stopper(m_Fractal.m_BenchmarkData.m_FeatureFinderHP);

        using T = HDRFloat<double>;
        using SubType = double;
        using IterType = uint64_t;
        constexpr PerturbExtras PExtras = PerturbExtras::Disable;

        auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtras>>();

        // Acquire PT data if possible (same logic as TryFindPeriodicPoint)
        auto extras = RefOrbitCalc::Extras::None;
        if (auto *cand = feature.GetCandidate()) {
            if (cand->modeFoundBy == FeatureFinderMode::LA)
                extras = RefOrbitCalc::Extras::IncludeLAv2;
        }

        if (!featureFinder->RefinePeriodicPoint_HighPrecision(feature)) {
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
FeatureFinderOrchestrator::ChooseClosestFeatureToMouse() const
{
    if (m_FeatureSummaries.empty())
        return nullptr;

    // Mouse in client pixels
    POINT pt{};
    if (!::GetCursorPos(&pt))
        return nullptr;

    HWND hwnd = m_Fractal.m_hWnd;
    if (!hwnd)
        return nullptr;

    if (!::ScreenToClient(hwnd, &pt))
        return nullptr;

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
FeatureFinderOrchestrator::ZoomToFoundFeature()
{
    FeatureSummary *best = ChooseClosestFeatureToMouse();
    if (!best) {
        std::cerr << "No feature found to zoom to.\n";
        return false;
    }

    const HighPrecision z = best->ComputeZoomFactor(m_Fractal.m_Ptz);
    return ZoomToFoundFeature(*best, &z);
}
