// LinuxCommandHandlers.cpp — Linux-side default impls of platform-agnostic
// catalog command hooks.  Mirrors the bodies in
// FractalSharkGUILib/MainWindow.cpp 1:1 but lives entirely on the Linux side.
//
// LinuxMainWindow inherits from LinuxCommandHandlers and overrides only the
// Linux-specific hooks (modals, file dialogs, window mode, clipboard).

#include "LinuxCommandHandlers.h"

#include "CrummyTest.h"
#include "Fractal.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "WaitCursor.h"

#include "LAParameters.h"

#include <climits>
#include <cstddef>
#include <iostream>

namespace FractalShark::Linux {

// ---- Algorithm selection --------------------------------------------------
void
LinuxCommandHandlers::OnSetAlgorithm(::RenderAlgorithmEnum alg)
{
    auto entry = GetRenderAlgorithmTupleEntry(alg);
    GetFractal().EnqueueMutation(
        [entry](Fractal &f) { [[maybe_unused]] const bool ok = f.SetRenderAlgorithm(entry); });
}

// ---- Synthetic shortcut command hooks -------------------------------------
void
LinuxCommandHandlers::OnAutoZoomDefaultAtPoint()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.CenterAtPoint(x, y); });
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}

void
LinuxCommandHandlers::OnCenterViewClearPerturbation()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.CenterAtPoint(x, y);
    });
}

void
LinuxCommandHandlers::OnResetCompressionDefaults()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
    });
}

void
LinuxCommandHandlers::OnLaThresholdScaleIncrease()
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
LinuxCommandHandlers::OnLaThresholdScaleDecrease()
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
LinuxCommandHandlers::OnLaPeriodDetectionIncrease()
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
LinuxCommandHandlers::OnLaPeriodDetectionDecrease()
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
LinuxCommandHandlers::OnRecalcCurrentCopyDetails()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ForceRecalc(); }).Wait();
    OnCurPos();
}

void
LinuxCommandHandlers::OnRecalcClearMediumCopyDetails()
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
LinuxCommandHandlers::OnRecalcClearAllCopyDetails()
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
LinuxCommandHandlers::OnRecalcClearLaCopyDetails()
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
LinuxCommandHandlers::OnIntermediateCompressionIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
LinuxCommandHandlers::OnIntermediateCompressionDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
LinuxCommandHandlers::OnLowCompressionIncrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
LinuxCommandHandlers::OnLowCompressionDecrease()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
LinuxCommandHandlers::OnPaletteAuxDepthNext()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(1); });
}

void
LinuxCommandHandlers::OnPaletteAuxDepthPrevious()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(-1); });
}

void
LinuxCommandHandlers::OnPaletteDepthNext()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UseNextPaletteDepth(); });
}

void
LinuxCommandHandlers::OnRecalcClearAllSquareView()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.SquareCurrentView();
    });
}

// ---- Help / Window --------------------------------------------------------
void
LinuxCommandHandlers::OnSquareView()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.SquareCurrentView(); });
}

void
LinuxCommandHandlers::OnRepainting()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ToggleRepainting(); });
}

// ---- Navigate -------------------------------------------------------------
void
LinuxCommandHandlers::OnBack()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.Back(); });
}

void
LinuxCommandHandlers::OnCenterView()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.CenterAtPoint(x, y); });
}

void
LinuxCommandHandlers::OnZoomIn()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomRecentered(x, y, -.45); });
}

void
LinuxCommandHandlers::OnZoomOut()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomRecentered(x, y, 1); });
}

void
LinuxCommandHandlers::OnAutoZoomDefault()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}

void
LinuxCommandHandlers::OnAutoZoomMax()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::Max>();
}

void
LinuxCommandHandlers::OnAutoZoomFilament()
{
    GetFractal().AutoZoom<Fractal::AutoZoomHeuristic::FilamentTip>();
}

#define LINUXSHARK_DEFINE_FEATURE_FINDER(Suffix, Mode)                                                  \
    void LinuxCommandHandlers::OnFeatureFinder##Suffix()                                                \
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
LinuxCommandHandlers::OnFeatureFinderZoom()
{
    const MenuPoint pt = GetMenuMousePos();
    GetFractal().EnqueueCommand([x = pt.X, y = pt.Y](Fractal &f) { f.ZoomToFoundFeature(x, y); });
}

void
LinuxCommandHandlers::OnFeatureFinderClear()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ClearAllFoundFeatures(); });
}

void
LinuxCommandHandlers::OnFeatureFinderResume()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ResumeNRFromCheckpoint(); });
}

void
LinuxCommandHandlers::OnNrInnerLoopGpu()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::GPU); });
}

void
LinuxCommandHandlers::OnNrInnerLoopCpu()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuMT); });
}

void
LinuxCommandHandlers::OnNrInnerLoopCpuSt()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuST); });
}

// ---- Built-In Views -------------------------------------------------------
void
LinuxCommandHandlers::OnStandardView()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.View(0); });
}

void
LinuxCommandHandlers::OnSelectBuiltInView(size_t oneBasedIndex)
{
    GetFractal().EnqueueCommand([oneBasedIndex](Fractal &f) { f.View(oneBasedIndex); });
}

// ---- Antialiasing ---------------------------------------------------------
void
LinuxCommandHandlers::OnGpuAntialiasing1x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 1); });
}

void
LinuxCommandHandlers::OnGpuAntialiasing4x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 2); });
}

void
LinuxCommandHandlers::OnGpuAntialiasing9x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.ResetDimensions(SIZE_MAX, SIZE_MAX, 3); });
}

void
LinuxCommandHandlers::OnGpuAntialiasing16x()
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
LinuxCommandHandlers::OnResetIterations()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.ResetNumIterations(); });
}

void
LinuxCommandHandlers::OnIncreaseIterations1p5x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 1.5); });
}

void
LinuxCommandHandlers::OnIncreaseIterations6x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 6.0); });
}

void
LinuxCommandHandlers::OnIncreaseIterations24x()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 24.0); });
}

void
LinuxCommandHandlers::OnDecreaseIterations()
{
    GetFractal().EnqueueCommand([](Fractal &f) { multiplyIterations(f, 2.0 / 3.0); });
}

void
LinuxCommandHandlers::OnIterations32Bit()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits32); });
}

void
LinuxCommandHandlers::OnIterations64Bit()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits64); });
}

// ---- Iteration precision --------------------------------------------------
void
LinuxCommandHandlers::OnIterationPrecision1x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(1); });
}

void
LinuxCommandHandlers::OnIterationPrecision2x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(4); });
}

void
LinuxCommandHandlers::OnIterationPrecision3x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(8); });
}

void
LinuxCommandHandlers::OnIterationPrecision4x()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(16); });
}

// ---- Perturbation ---------------------------------------------------------
void
LinuxCommandHandlers::OnPerturbClearAll()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All); });
}

void
LinuxCommandHandlers::OnPerturbClearMed()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes); });
}

void
LinuxCommandHandlers::OnPerturbClearHigh()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes); });
}

void
LinuxCommandHandlers::OnPerturbationAuto()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto); });
}

void
LinuxCommandHandlers::OnPerturbationSinglethread()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST); });
}

void
LinuxCommandHandlers::OnPerturbationMultithread()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT); });
}

void
LinuxCommandHandlers::OnPerturbationSinglethreadPeriodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity); });
}

void
LinuxCommandHandlers::OnPerturbationMultithread2Periodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3); });
}

void
LinuxCommandHandlers::OnPerturbationMt2PerturbMthighStmed()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
    });
}

void
LinuxCommandHandlers::OnPerturbationMt2PerturbMthighMtmed1()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
    });
}

void
LinuxCommandHandlers::OnPerturbationMt2PerturbMthighMtmed2()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
    });
}

void
LinuxCommandHandlers::OnPerturbationMt2PerturbMthighMtmed3()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
    });
}

void
LinuxCommandHandlers::OnPerturbationMt2PerturbMthighMtmed4()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
    });
}

void
LinuxCommandHandlers::OnPerturbationMultithread5Periodicity()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5); });
}

void
LinuxCommandHandlers::OnPerturbationGpu()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU); });
}

void
LinuxCommandHandlers::OnPerturbationLoad()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.LoadPerturbationOrbits(); });
}

void
LinuxCommandHandlers::OnPerturbationSave()
{
    GetFractal().EnqueueMutation([](Fractal &f) { f.SavePerturbationOrbits(); });
}

// ---- Autosave -------------------------------------------------------------
void
LinuxCommandHandlers::OnPerturbAutosaveOnDelete()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithoutSave); });
}

void
LinuxCommandHandlers::OnPerturbAutosaveOn()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithSave); });
}

void
LinuxCommandHandlers::OnPerturbAutosaveOff()
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
LinuxCommandHandlers::OnPaletteType0()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Basic); });
}

void
LinuxCommandHandlers::OnPaletteType1()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Default); });
}

void
LinuxCommandHandlers::OnPaletteType2()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Patriotic); });
}

void
LinuxCommandHandlers::OnPaletteType3()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Summer); });
}

void
LinuxCommandHandlers::OnPaletteType4()
{
    GetFractal().EnqueueCommand([](Fractal &f) { applyPaletteType(f, FractalPaletteType::Random); });
}

void
LinuxCommandHandlers::OnCreateNewPalette()
{
    GetFractal().EnqueueCommand([](Fractal &f) {
        f.CreateNewFractalPalette();
        f.UsePaletteType(FractalPaletteType::Random);
    });
}

void
LinuxCommandHandlers::OnPalette5()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(5); });
}

void
LinuxCommandHandlers::OnPalette6()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(6); });
}

void
LinuxCommandHandlers::OnPalette8()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(8); });
}

void
LinuxCommandHandlers::OnPalette12()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(12); });
}

void
LinuxCommandHandlers::OnPalette16()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(16); });
}

void
LinuxCommandHandlers::OnPalette20()
{
    GetFractal().EnqueueCommand([](Fractal &f) { f.UsePalette(20); });
}

// ---- Tests / Benchmarks ---------------------------------------------------
void
LinuxCommandHandlers::OnBasicTest()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.TestAll();
}

void
LinuxCommandHandlers::OnTest27()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.TestReallyHardView27();
}

void
LinuxCommandHandlers::OnBenchmarkFull()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::All);
}

void
LinuxCommandHandlers::OnBenchmarkInt()
{
    Environment::WaitCursor waitCursor;
    CrummyTest t{GetFractal()};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
}

// ---- LA -------------------------------------------------------------------
void
LinuxCommandHandlers::OnLaMultithreaded()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
    });
}

void
LinuxCommandHandlers::OnLaSinglethreaded()
{
    GetFractal().EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    });
}

void
LinuxCommandHandlers::OnLaSettings1()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxAccuracy); });
}

void
LinuxCommandHandlers::OnLaSettings2()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxPerf); });
}

void
LinuxCommandHandlers::OnLaSettings3()
{
    GetFractal().EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MinMemory); });
}

} // namespace FractalShark::Linux
