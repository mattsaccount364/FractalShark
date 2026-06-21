// PortableCommandHandlers.cpp
// Platform-independent command implementations shared by the native GUI shells.

#include "stdafx.h"
#include "PortableCommandHandlers.h"

#include "CrummyTest.h"
#include "Fractal.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "WaitCursor.h"

#include "LAParameters.h"

#include <climits>
#include <cstddef>
#include <iostream>

namespace FractalShark {

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
    GetFractal().EnqueueCommand([](Fractal &f) { f.ToggleRepainting(); });
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
