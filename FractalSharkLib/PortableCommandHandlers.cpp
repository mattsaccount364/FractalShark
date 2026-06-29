// PortableCommandHandlers.cpp
// Platform-independent command implementations shared by the native GUI shells.

#include "stdafx.h"
#include "PortableCommandHandlers.h"

#include "AlgCmds.h"
#include "CrummyTest.h"
#include "Environment.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "WaitCursor.h"

#include "LAParameters.h"

#include <climits>
#include <cstddef>
#include <iostream>

namespace FractalShark {

namespace {

const RenderAlgorithmEnum *
TryFindAlgForCommand(FractalCommand command) noexcept
{
    const auto idm = static_cast<int>(IdmFromCommand(command));
    for (const auto &entry : kAlgCmds) {
        if (entry.id == idm) {
            return &entry.alg;
        }
    }
    return nullptr;
}

} // namespace

void
PortableCommandHandlers::ExecuteCommand(FractalCommand cmd)
{
    // Algorithm-selection commands first; they all share OnSetAlgorithm.
    if (const auto *alg = TryFindAlgForCommand(cmd); alg != nullptr) {
        OnSetAlgorithm(*alg);
        return;
    }

    // Built-in view range (View1..View40) → portable index hook.
    {
        const auto raw = static_cast<uint32_t>(cmd);
        const auto v1 = static_cast<uint32_t>(FractalCommand::View1);
        const auto v40 = static_cast<uint32_t>(FractalCommand::View40);
        if (raw >= v1 && raw <= v40) {
            OnSelectBuiltInView(static_cast<size_t>(raw - v1) + 1u);
            return;
        }
    }

    switch (cmd) {
        // ---- Synthetic shortcut commands ----
        case FractalCommand::AutoZoomFeatureAtPoint:
            OnAutoZoomFeatureAtPoint();
            return;
        case FractalCommand::AutoZoomDefaultAtPoint:
            OnAutoZoomDefaultAtPoint();
            return;
        case FractalCommand::CenterViewClearPerturbation:
            OnCenterViewClearPerturbation();
            return;
        case FractalCommand::ResetCompressionDefaults:
            OnResetCompressionDefaults();
            return;
        case FractalCommand::LaThresholdScaleIncrease:
            OnLaThresholdScaleIncrease();
            return;
        case FractalCommand::LaThresholdScaleDecrease:
            OnLaThresholdScaleDecrease();
            return;
        case FractalCommand::LaPeriodDetectionIncrease:
            OnLaPeriodDetectionIncrease();
            return;
        case FractalCommand::LaPeriodDetectionDecrease:
            OnLaPeriodDetectionDecrease();
            return;
        case FractalCommand::RecalcCurrentCopyDetails:
            OnRecalcCurrentCopyDetails();
            return;
        case FractalCommand::RecalcClearMediumCopyDetails:
            OnRecalcClearMediumCopyDetails();
            return;
        case FractalCommand::RecalcClearAllCopyDetails:
            OnRecalcClearAllCopyDetails();
            return;
        case FractalCommand::RecalcClearLaCopyDetails:
            OnRecalcClearLaCopyDetails();
            return;
        case FractalCommand::IntermediateCompressionIncrease:
            OnIntermediateCompressionIncrease();
            return;
        case FractalCommand::IntermediateCompressionDecrease:
            OnIntermediateCompressionDecrease();
            return;
        case FractalCommand::LowCompressionIncrease:
            OnLowCompressionIncrease();
            return;
        case FractalCommand::LowCompressionDecrease:
            OnLowCompressionDecrease();
            return;
        case FractalCommand::PaletteAuxDepthNext:
            OnPaletteAuxDepthNext();
            return;
        case FractalCommand::PaletteAuxDepthPrevious:
            OnPaletteAuxDepthPrevious();
            return;
        case FractalCommand::PaletteDepthNext:
            OnPaletteDepthNext();
            return;
        case FractalCommand::RecalcClearAllSquareView:
            OnRecalcClearAllSquareView();
            return;

        // ---- Help / Window ----
        case FractalCommand::ShowHotkeys:
            OnShowHotkeys();
            return;
        case FractalCommand::ViewsHelp:
            OnViewsHelp();
            return;
        case FractalCommand::HelpAlg:
            OnHelpAlg();
            return;
        case FractalCommand::SquareView:
            OnSquareView();
            return;
        case FractalCommand::Repainting:
            OnRepainting();
            return;
        case FractalCommand::Windowed:
            OnWindowed();
            return;
        case FractalCommand::WindowedSq:
            OnWindowedSq();
            return;
        case FractalCommand::Minimize:
            OnMinimize();
            return;
        case FractalCommand::CurPos:
            OnCurPos();
            return;
        case FractalCommand::Exit:
            OnExit();
            return;

        // ---- Navigate ----
        case FractalCommand::Back:
            OnBack();
            return;
        case FractalCommand::CenterView:
            OnCenterView();
            return;
        case FractalCommand::ZoomIn:
            OnZoomIn();
            return;
        case FractalCommand::ZoomOut:
            OnZoomOut();
            return;
        case FractalCommand::AutoZoomDefault:
            OnAutoZoomDefault();
            return;
        case FractalCommand::AutoZoomMax:
            OnAutoZoomMax();
            return;
        case FractalCommand::AutoZoomFilament:
            OnAutoZoomFilament();
            return;
        case FractalCommand::FeatureFinderDirect:
            OnFeatureFinderDirect();
            return;
        case FractalCommand::FeatureFinderDirectScan:
            OnFeatureFinderDirectScan();
            return;
        case FractalCommand::FeatureFinderPt:
            OnFeatureFinderPt();
            return;
        case FractalCommand::FeatureFinderPtScan:
            OnFeatureFinderPtScan();
            return;
        case FractalCommand::FeatureFinderLa:
            OnFeatureFinderLa();
            return;
        case FractalCommand::FeatureFinderLaScan:
            OnFeatureFinderLaScan();
            return;
        case FractalCommand::FeatureFinderZoom:
            OnFeatureFinderZoom();
            return;
        case FractalCommand::FeatureFinderClear:
            OnFeatureFinderClear();
            return;
        case FractalCommand::FeatureFinderResume:
            OnFeatureFinderResume();
            return;
        case FractalCommand::NrInnerLoopGpu:
            OnNrInnerLoopGpu();
            return;
        case FractalCommand::NrInnerLoopCpu:
            OnNrInnerLoopCpu();
            return;
        case FractalCommand::NrInnerLoopCpuSt:
            OnNrInnerLoopCpuSt();
            return;

        // ---- Built-In Views ----
        case FractalCommand::StandardView:
            OnStandardView();
            return;

        // ---- Antialiasing ----
        case FractalCommand::GpuAntialiasing1x:
            OnGpuAntialiasing1x();
            return;
        case FractalCommand::GpuAntialiasing4x:
            OnGpuAntialiasing4x();
            return;
        case FractalCommand::GpuAntialiasing9x:
            OnGpuAntialiasing9x();
            return;
        case FractalCommand::GpuAntialiasing16x:
            OnGpuAntialiasing16x();
            return;

        // ---- Iterations ----
        case FractalCommand::ResetIterations:
            OnResetIterations();
            return;
        case FractalCommand::IncreaseIterations1p5x:
            OnIncreaseIterations1p5x();
            return;
        case FractalCommand::IncreaseIterations6x:
            OnIncreaseIterations6x();
            return;
        case FractalCommand::IncreaseIterations24x:
            OnIncreaseIterations24x();
            return;
        case FractalCommand::DecreaseIterations:
            OnDecreaseIterations();
            return;
        case FractalCommand::Iterations32Bit:
            OnIterations32Bit();
            return;
        case FractalCommand::Iterations64Bit:
            OnIterations64Bit();
            return;

        // ---- Iteration Precision ----
        case FractalCommand::IterationPrecision1x:
            OnIterationPrecision1x();
            return;
        case FractalCommand::IterationPrecision2x:
            OnIterationPrecision2x();
            return;
        case FractalCommand::IterationPrecision3x:
            OnIterationPrecision3x();
            return;
        case FractalCommand::IterationPrecision4x:
            OnIterationPrecision4x();
            return;

        // ---- Perturbation ----
        case FractalCommand::PerturbClearAll:
            OnPerturbClearAll();
            return;
        case FractalCommand::PerturbClearMed:
            OnPerturbClearMed();
            return;
        case FractalCommand::PerturbClearHigh:
            OnPerturbClearHigh();
            return;
        case FractalCommand::PerturbationAuto:
            OnPerturbationAuto();
            return;
        case FractalCommand::PerturbationSinglethread:
            OnPerturbationSinglethread();
            return;
        case FractalCommand::PerturbationMultithread:
            OnPerturbationMultithread();
            return;
        case FractalCommand::PerturbationSinglethreadPeriodicity:
            OnPerturbationSinglethreadPeriodicity();
            return;
        case FractalCommand::PerturbationMultithread2Periodicity:
            OnPerturbationMultithread2Periodicity();
            return;
        case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighStmed:
            OnPerturbationMt2PerturbMthighStmed();
            return;
        case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed1:
            OnPerturbationMt2PerturbMthighMtmed1();
            return;
        case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed2:
            OnPerturbationMt2PerturbMthighMtmed2();
            return;
        case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed3:
            OnPerturbationMt2PerturbMthighMtmed3();
            return;
        case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed4:
            OnPerturbationMt2PerturbMthighMtmed4();
            return;
        case FractalCommand::PerturbationMultithread5Periodicity:
            OnPerturbationMultithread5Periodicity();
            return;
        case FractalCommand::PerturbationGpu:
            OnPerturbationGpu();
            return;
        case FractalCommand::PerturbationLoad:
            OnPerturbationLoad();
            return;
        case FractalCommand::PerturbationSave:
            OnPerturbationSave();
            return;

        // ---- Memory / Autosave ----
        case FractalCommand::PerturbAutosaveOnDelete:
            OnPerturbAutosaveOnDelete();
            return;
        case FractalCommand::PerturbAutosaveOn:
            OnPerturbAutosaveOn();
            return;
        case FractalCommand::PerturbAutosaveOff:
            OnPerturbAutosaveOff();
            return;

        // ---- Palette ----
        case FractalCommand::PaletteType0:
            OnPaletteType0();
            return;
        case FractalCommand::PaletteType1:
            OnPaletteType1();
            return;
        case FractalCommand::PaletteType2:
            OnPaletteType2();
            return;
        case FractalCommand::PaletteType3:
            OnPaletteType3();
            return;
        case FractalCommand::PaletteType4:
            OnPaletteType4();
            return;
        case FractalCommand::CreateNewPalette:
            OnCreateNewPalette();
            return;
        case FractalCommand::Palette5:
            OnPalette5();
            return;
        case FractalCommand::Palette6:
            OnPalette6();
            return;
        case FractalCommand::Palette8:
            OnPalette8();
            return;
        case FractalCommand::Palette12:
            OnPalette12();
            return;
        case FractalCommand::Palette16:
            OnPalette16();
            return;
        case FractalCommand::Palette20:
            OnPalette20();
            return;
        case FractalCommand::PaletteRotate:
            OnPaletteRotate();
            return;

        // ---- Save / Load ----
        case FractalCommand::SaveLocation:
            OnSaveLocation();
            return;
        case FractalCommand::SaveHiResBmp:
            OnSaveHiResBmp();
            return;
        case FractalCommand::SaveItersText:
            OnSaveItersText();
            return;
        case FractalCommand::SaveBmp:
            OnSaveBmp();
            return;
        case FractalCommand::SaveRefOrbitText:
            OnSaveRefOrbitText();
            return;
        case FractalCommand::SaveRefOrbitTextSimple:
            OnSaveRefOrbitTextSimple();
            return;
        case FractalCommand::SaveRefOrbitTextMax:
            OnSaveRefOrbitTextMax();
            return;
        case FractalCommand::SaveRefOrbitImagMax:
            OnSaveRefOrbitImagMax();
            return;
        case FractalCommand::DiffRefOrbitImagMax:
            OnDiffRefOrbitImagMax();
            return;
        case FractalCommand::LoadLocation:
            OnLoadLocation();
            return;
        case FractalCommand::LoadEnterLocation:
            OnLoadEnterLocation();
            return;
        case FractalCommand::LoadRefOrbitImagMax:
            OnLoadRefOrbitImagMax();
            return;
        case FractalCommand::LoadRefOrbitImagMaxSaved:
            OnLoadRefOrbitImagMaxSaved();
            return;

        // ---- Tests / Benchmarks ----
        case FractalCommand::BasicTest:
            OnBasicTest();
            return;
        case FractalCommand::Test27:
            OnTest27();
            return;
        case FractalCommand::BenchmarkFull:
            OnBenchmarkFull();
            return;
        case FractalCommand::BenchmarkInt:
            OnBenchmarkInt();
            return;

        // ---- LA ----
        case FractalCommand::LaMultithreaded:
            OnLaMultithreaded();
            return;
        case FractalCommand::LaSinglethreaded:
            OnLaSinglethreaded();
            return;
        case FractalCommand::LaSettings1:
            OnLaSettings1();
            return;
        case FractalCommand::LaSettings2:
            OnLaSettings2();
            return;
        case FractalCommand::LaSettings3:
            OnLaSettings3();
            return;

        default:
            break;
    }

    throw FractalSharkSeriousException("Unhandled FractalCommand in ExecuteCommand");
}

// ---- Algorithm selection --------------------------------------------------
void
PortableCommandHandlers::OnSetAlgorithm(::RenderAlgorithmEnum alg)
{
    auto entry = GetRenderAlgorithmTupleEntry(alg);
    GetFractal().EnqueueMutation(
        [entry](Fractal &f) { [[maybe_unused]] const bool ok = f.SetRenderAlgorithm(entry); });
}

// ---- Synthetic shortcut command hooks -------------------------------------
void
PortableCommandHandlers::OnAutoZoomDefaultAtPoint()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.CenterAtPoint(x, y); });
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}

void
PortableCommandHandlers::OnCenterViewClearPerturbation()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.CenterAtPoint(x, y);
    });
}

void
PortableCommandHandlers::OnResetCompressionDefaults()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
    });
}

void
PortableCommandHandlers::OnLaThresholdScaleIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustLAThresholdScaleExponent(1);
        laParameters.AdjustLAThresholdCScaleExponent(1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
PortableCommandHandlers::OnLaThresholdScaleDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustLAThresholdScaleExponent(-1);
        laParameters.AdjustLAThresholdCScaleExponent(-1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
PortableCommandHandlers::OnLaPeriodDetectionIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustPeriodDetectionThreshold2Exponent(1);
        laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
PortableCommandHandlers::OnLaPeriodDetectionDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustPeriodDetectionThreshold2Exponent(-1);
        laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(-1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
PortableCommandHandlers::OnRecalcCurrentCopyDetails()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ForceRecalc(); }).Wait();
    OnCurPos();
}

void
PortableCommandHandlers::OnRecalcClearMediumCopyDetails()
{
    GetFractal()
        .EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
            f.ForceRecalc();
        })
        .Wait();
    OnCurPos();
}

void
PortableCommandHandlers::OnRecalcClearAllCopyDetails()
{
    GetFractal()
        .EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            f.ForceRecalc();
        })
        .Wait();
    OnCurPos();
}

void
PortableCommandHandlers::OnRecalcClearLaCopyDetails()
{
    GetFractal()
        .EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
            f.ForceRecalc();
        })
        .Wait();
    OnCurPos();
}

void
PortableCommandHandlers::OnIntermediateCompressionIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
PortableCommandHandlers::OnIntermediateCompressionDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
PortableCommandHandlers::OnLowCompressionIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
PortableCommandHandlers::OnLowCompressionDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
PortableCommandHandlers::OnPaletteAuxDepthNext()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(1); });
}

void
PortableCommandHandlers::OnPaletteAuxDepthPrevious()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(-1); });
}

void
PortableCommandHandlers::OnPaletteDepthNext()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteDepth(); });
}

void
PortableCommandHandlers::OnRecalcClearAllSquareView()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.SquareCurrentView();
    });
}

// ---- Help / Window --------------------------------------------------------
void
PortableCommandHandlers::OnSquareView()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.SquareCurrentView(); });
}

void
PortableCommandHandlers::OnRepainting()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ToggleRepainting(); }, false);
}

// ---- Navigate -------------------------------------------------------------
void
PortableCommandHandlers::OnBack()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.Back(); });
}

void
PortableCommandHandlers::OnCenterView()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.CenterAtPoint(x, y); });
}

void
PortableCommandHandlers::OnZoomIn()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomRecentered(x, y, -.45); });
}

void
PortableCommandHandlers::OnZoomOut()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomRecentered(x, y, 1); });
}

void
PortableCommandHandlers::OnAutoZoomDefault()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}

void
PortableCommandHandlers::OnAutoZoomMax()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Max>();
}

void
PortableCommandHandlers::OnAutoZoomFilament()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::FilamentTip>();
}

#define LINUXSHARK_DEFINE_FEATURE_FINDER(Suffix, Mode)                                                  \
    void PortableCommandHandlers::OnFeatureFinder##Suffix()                                             \
    {                                                                                                   \
        const MenuPoint pt = GetMenuMousePos();                                                         \
        const int x = pt.X;                                                                             \
        const int y = pt.Y;                                                                             \
        GetFractal().EnqueueCommand([x, y](Fractal &f) { f.TryFindPeriodicPoint(x, y, Mode); });        \
    }

LINUXSHARK_DEFINE_FEATURE_FINDER(Direct, FeatureFinderMode::Direct)
LINUXSHARK_DEFINE_FEATURE_FINDER(DirectScan, FeatureFinderMode::DirectScan)
LINUXSHARK_DEFINE_FEATURE_FINDER(Pt, FeatureFinderMode::PT)
LINUXSHARK_DEFINE_FEATURE_FINDER(PtScan, FeatureFinderMode::PTScan)
LINUXSHARK_DEFINE_FEATURE_FINDER(La, FeatureFinderMode::LA)
LINUXSHARK_DEFINE_FEATURE_FINDER(LaScan, FeatureFinderMode::LAScan)

#undef LINUXSHARK_DEFINE_FEATURE_FINDER

void
PortableCommandHandlers::OnFeatureFinderZoom()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomToFoundFeature(x, y); });
}

void
PortableCommandHandlers::OnFeatureFinderClear()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ClearAllFoundFeatures(); });
}

void
PortableCommandHandlers::OnFeatureFinderResume()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ResumeNRFromCheckpoint(); });
}

void
PortableCommandHandlers::OnNrInnerLoopGpu()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::GPU); });
}

void
PortableCommandHandlers::OnNrInnerLoopCpu()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuMT); });
}

void
PortableCommandHandlers::OnNrInnerLoopCpuSt()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuST); });
}

// ---- Built-In Views -------------------------------------------------------
void
PortableCommandHandlers::OnStandardView()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.View(0); });
}

void
PortableCommandHandlers::OnSelectBuiltInView(size_t oneBasedIndex)
{
    GetFractal().EnqueueCommand([oneBasedIndex](Fractal &f) { f.View(oneBasedIndex); });
}

// ---- Antialiasing ---------------------------------------------------------
void
PortableCommandHandlers::OnGpuAntialiasing1x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 1); });
}

void
PortableCommandHandlers::OnGpuAntialiasing4x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 2); });
}

void
PortableCommandHandlers::OnGpuAntialiasing9x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 3); });
}

void
PortableCommandHandlers::OnGpuAntialiasing16x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 4); });
}

// ---- Iterations -----------------------------------------------------------
namespace {
void
multiplyIterations(Fractal &f, double factor)
{
    if (f.GetIterType() == IterTypeEnum::Bits32) {
        uint64_t curIters = f.GetNumIterations<uint32_t>();
        curIters = static_cast<uint64_t>(static_cast<double>(curIters) * factor);
        f.SetNumIterations<uint32_t>(curIters);
    } else {
        uint64_t curIters = f.GetNumIterations<uint64_t>();
        curIters = static_cast<uint64_t>(static_cast<double>(curIters) * factor);
        f.SetNumIterations<uint64_t>(curIters);
    }
}
} // namespace

void
PortableCommandHandlers::OnResetIterations()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ResetNumIterations(); });
}

void
PortableCommandHandlers::OnIncreaseIterations1p5x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 1.5); });
}

void
PortableCommandHandlers::OnIncreaseIterations6x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 6.0); });
}

void
PortableCommandHandlers::OnIncreaseIterations24x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 24.0); });
}

void
PortableCommandHandlers::OnDecreaseIterations()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 2.0 / 3.0); });
}

void
PortableCommandHandlers::OnIterations32Bit()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits32); });
}

void
PortableCommandHandlers::OnIterations64Bit()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits64); });
}

// ---- Iteration precision --------------------------------------------------
void
PortableCommandHandlers::OnIterationPrecision1x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(1); });
}

void
PortableCommandHandlers::OnIterationPrecision2x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(4); });
}

void
PortableCommandHandlers::OnIterationPrecision3x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(8); });
}

void
PortableCommandHandlers::OnIterationPrecision4x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(16); });
}

// ---- Perturbation ---------------------------------------------------------
void
PortableCommandHandlers::OnPerturbClearAll()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All); });
}

void
PortableCommandHandlers::OnPerturbClearMed()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes); });
}

void
PortableCommandHandlers::OnPerturbClearHigh()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes); });
}

void
PortableCommandHandlers::OnPerturbationAuto()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto); });
}

void
PortableCommandHandlers::OnPerturbationSinglethread()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST); });
}

void
PortableCommandHandlers::OnPerturbationMultithread()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT); });
}

void
PortableCommandHandlers::OnPerturbationSinglethreadPeriodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity); });
}

void
PortableCommandHandlers::OnPerturbationMultithread2Periodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3); });
}

void
PortableCommandHandlers::OnPerturbationMt2PerturbMthighStmed()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
    });
}

void
PortableCommandHandlers::OnPerturbationMt2PerturbMthighMtmed1()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
    });
}

void
PortableCommandHandlers::OnPerturbationMt2PerturbMthighMtmed2()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
    });
}

void
PortableCommandHandlers::OnPerturbationMt2PerturbMthighMtmed3()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
    });
}

void
PortableCommandHandlers::OnPerturbationMt2PerturbMthighMtmed4()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
    });
}

void
PortableCommandHandlers::OnPerturbationMultithread5Periodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5); });
}

void
PortableCommandHandlers::OnPerturbationGpu()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU); });
}

void
PortableCommandHandlers::OnPerturbationLoad()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.LoadPerturbationOrbits(); });
}

void
PortableCommandHandlers::OnPerturbationSave()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SavePerturbationOrbits(); });
}

// ---- Autosave -------------------------------------------------------------
void
PortableCommandHandlers::OnPerturbAutosaveOnDelete()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithoutSave); });
}

void
PortableCommandHandlers::OnPerturbAutosaveOn()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithSave); });
}

void
PortableCommandHandlers::OnPerturbAutosaveOff()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetResultsAutosave(AddPointOptions::DontSave); });
}

// ---- Palette --------------------------------------------------------------
namespace {
void
applyPaletteType(Fractal &f, FractalPaletteType type)
{
    f.UsePaletteType(type);
    if (type == FractalPaletteType::Default) {
        f.UsePalette(8);
        f.SetPaletteAuxDepth(0);
    }
}
} // namespace

void
PortableCommandHandlers::OnPaletteType0()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Basic); });
}

void
PortableCommandHandlers::OnPaletteType1()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Default); });
}

void
PortableCommandHandlers::OnPaletteType2()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Patriotic); });
}

void
PortableCommandHandlers::OnPaletteType3()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Summer); });
}

void
PortableCommandHandlers::OnPaletteType4()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Random); });
}

void
PortableCommandHandlers::OnCreateNewPalette()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.CreateNewFractalPalette();
        f.UsePaletteType(FractalPaletteType::Random);
    });
}

void
PortableCommandHandlers::OnPalette5()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(5); });
}

void
PortableCommandHandlers::OnPalette6()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(6); });
}

void
PortableCommandHandlers::OnPalette8()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(8); });
}

void
PortableCommandHandlers::OnPalette12()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(12); });
}

void
PortableCommandHandlers::OnPalette16()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(16); });
}

void
PortableCommandHandlers::OnPalette20()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(20); });
}

void
PortableCommandHandlers::OnPaletteRotate()
{
    auto &fractal = GetFractal();
    fractal.ResetStopCalculating();
    if (auto *pool = fractal.GetRenderPool()) {
        pool->Drain();
    }

    const uint64_t presentationGroup = fractal.BeginPacedAnimation();
    while (!fractal.GetStopCalculating()) {
        auto handle = fractal.EnqueuePaletteRecolor([](Fractal &f) { f.RotateFractalPalette(10); },
                                                    false,
                                                    RenderPresentationMode::PacedAnimation,
                                                    presentationGroup);
        handle.Wait();
        Environment::PumpUIEvents();
    }

    fractal.CancelPacedAnimation(presentationGroup);
    fractal.EnqueuePaletteRecolor([](Fractal &f) { f.ResetFractalPalette(); }, false).Wait();
    fractal.ResetStopCalculating();
}

// ---- Tests / Benchmarks ---------------------------------------------------
void
PortableCommandHandlers::OnBasicTest()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.TestAll();
}

void
PortableCommandHandlers::OnTest27()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.TestReallyHardView27();
}

void
PortableCommandHandlers::OnBenchmarkFull()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::All);
}

void
PortableCommandHandlers::OnBenchmarkInt()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
}

// ---- LA -------------------------------------------------------------------
void
PortableCommandHandlers::OnLaMultithreaded()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
    });
}

void
PortableCommandHandlers::OnLaSinglethreaded()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    });
}

void
PortableCommandHandlers::OnLaSettings1()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxAccuracy); });
}

void
PortableCommandHandlers::OnLaSettings2()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxPerf); });
}

void
PortableCommandHandlers::OnLaSettings3()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MinMemory); });
}

} // namespace FractalShark
