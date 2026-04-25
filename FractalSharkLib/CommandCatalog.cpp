// CommandCatalog.cpp
//
// Single dispatch site for FractalCommand → ExecuteCommandHost hook calls.
// The algorithm-selection family folds into the host's OnSetAlgorithm hook
// via kAlgCmds; everything else is a 1:1 enum→method routing.  Anything
// that lacks a case here (range commands, unmigrated IDMs) falls through to
// host.DispatchByIdm() so the legacy CommandDispatcher path still handles
// it on Win32.

#include "stdafx.h"

#include "CommandCatalog.h"

#include "AlgCmds.h"
#include "RenderAlgorithm.h"

namespace FractalShark {

namespace {

// kAlgCmds maps IDM → RenderAlgorithmEnum.  Catalog FractalCommand values
// for algorithms equal their IDM_*, so we can reverse-lookup directly.
const RenderAlgorithmEnum *
TryFindAlgForCommand(FractalCommand cmd) noexcept
{
    const auto idm = static_cast<int>(IdmFromCommand(cmd));
    for (const auto &e : kAlgCmds) {
        if (e.id == idm)
            return &e.alg;
    }
    return nullptr;
}

} // namespace

void
ExecuteCommand(FractalCommand cmd, ExecuteCommandHost &host)
{
    // Algorithm-selection commands first; they all share OnSetAlgorithm.
    if (const auto *alg = TryFindAlgForCommand(cmd); alg != nullptr) {
        host.OnSetAlgorithm(*alg);
        return;
    }

    switch (cmd) {
    // ---- Help / Window ----
    case FractalCommand::ShowHotkeys:    host.OnShowHotkeys();    return;
    case FractalCommand::ViewsHelp:      host.OnViewsHelp();      return;
    case FractalCommand::HelpAlg:        host.OnHelpAlg();        return;
    case FractalCommand::SquareView:     host.OnSquareView();     return;
    case FractalCommand::Repainting:     host.OnRepainting();     return;
    case FractalCommand::Windowed:       host.OnWindowed();       return;
    case FractalCommand::WindowedSq:     host.OnWindowedSq();     return;
    case FractalCommand::Minimize:       host.OnMinimize();       return;
    case FractalCommand::CurPos:         host.OnCurPos();         return;
    case FractalCommand::Exit:           host.OnExit();           return;

    // ---- Navigate ----
    case FractalCommand::Back:                    host.OnBack();                    return;
    case FractalCommand::CenterView:              host.OnCenterView();              return;
    case FractalCommand::ZoomIn:                  host.OnZoomIn();                  return;
    case FractalCommand::ZoomOut:                 host.OnZoomOut();                 return;
    case FractalCommand::AutoZoomDefault:         host.OnAutoZoomDefault();         return;
    case FractalCommand::AutoZoomMax:             host.OnAutoZoomMax();             return;
    case FractalCommand::AutoZoomFilament:        host.OnAutoZoomFilament();        return;
    case FractalCommand::FeatureFinderDirect:     host.OnFeatureFinderDirect();     return;
    case FractalCommand::FeatureFinderDirectScan: host.OnFeatureFinderDirectScan(); return;
    case FractalCommand::FeatureFinderPt:         host.OnFeatureFinderPt();         return;
    case FractalCommand::FeatureFinderPtScan:     host.OnFeatureFinderPtScan();     return;
    case FractalCommand::FeatureFinderLa:         host.OnFeatureFinderLa();         return;
    case FractalCommand::FeatureFinderLaScan:     host.OnFeatureFinderLaScan();     return;
    case FractalCommand::FeatureFinderZoom:       host.OnFeatureFinderZoom();       return;
    case FractalCommand::FeatureFinderClear:      host.OnFeatureFinderClear();      return;
    case FractalCommand::FeatureFinderResume:     host.OnFeatureFinderResume();     return;
    case FractalCommand::NrInnerLoopGpu:          host.OnNrInnerLoopGpu();          return;
    case FractalCommand::NrInnerLoopCpu:          host.OnNrInnerLoopCpu();          return;
    case FractalCommand::NrInnerLoopCpuSt:        host.OnNrInnerLoopCpuSt();        return;

    // ---- Built-In Views ----
    case FractalCommand::StandardView:            host.OnStandardView();            return;
    // View1..View40 are payload-bearing range commands; they fall through
    // to DispatchByIdm so HandleCommandRange picks them up.

    // ---- Antialiasing ----
    case FractalCommand::GpuAntialiasing1x:       host.OnGpuAntialiasing1x();       return;
    case FractalCommand::GpuAntialiasing4x:       host.OnGpuAntialiasing4x();       return;
    case FractalCommand::GpuAntialiasing9x:       host.OnGpuAntialiasing9x();       return;
    case FractalCommand::GpuAntialiasing16x:      host.OnGpuAntialiasing16x();      return;

    // ---- Iterations ----
    case FractalCommand::ResetIterations:         host.OnResetIterations();         return;
    case FractalCommand::IncreaseIterations1p5x:  host.OnIncreaseIterations1p5x();  return;
    case FractalCommand::IncreaseIterations6x:    host.OnIncreaseIterations6x();    return;
    case FractalCommand::IncreaseIterations24x:   host.OnIncreaseIterations24x();   return;
    case FractalCommand::DecreaseIterations:      host.OnDecreaseIterations();      return;
    case FractalCommand::Iterations32Bit:         host.OnIterations32Bit();         return;
    case FractalCommand::Iterations64Bit:         host.OnIterations64Bit();         return;

    // ---- Iteration Precision ----
    case FractalCommand::IterationPrecision1x:    host.OnIterationPrecision1x();    return;
    case FractalCommand::IterationPrecision2x:    host.OnIterationPrecision2x();    return;
    case FractalCommand::IterationPrecision3x:    host.OnIterationPrecision3x();    return;
    case FractalCommand::IterationPrecision4x:    host.OnIterationPrecision4x();    return;

    // ---- Perturbation ----
    case FractalCommand::PerturbResults:                                   host.OnPerturbResults();                          return;
    case FractalCommand::PerturbClearAll:                                  host.OnPerturbClearAll();                         return;
    case FractalCommand::PerturbClearMed:                                  host.OnPerturbClearMed();                         return;
    case FractalCommand::PerturbClearHigh:                                 host.OnPerturbClearHigh();                        return;
    case FractalCommand::PerturbationAuto:                                 host.OnPerturbationAuto();                        return;
    case FractalCommand::PerturbationSinglethread:                         host.OnPerturbationSinglethread();                return;
    case FractalCommand::PerturbationMultithread:                          host.OnPerturbationMultithread();                 return;
    case FractalCommand::PerturbationSinglethreadPeriodicity:              host.OnPerturbationSinglethreadPeriodicity();     return;
    case FractalCommand::PerturbationMultithread2Periodicity:              host.OnPerturbationMultithread2Periodicity();     return;
    case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighStmed:  host.OnPerturbationMt2PerturbMthighStmed();  return;
    case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed1: host.OnPerturbationMt2PerturbMthighMtmed1(); return;
    case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed2: host.OnPerturbationMt2PerturbMthighMtmed2(); return;
    case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed3: host.OnPerturbationMt2PerturbMthighMtmed3(); return;
    case FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed4: host.OnPerturbationMt2PerturbMthighMtmed4(); return;
    case FractalCommand::PerturbationMultithread5Periodicity:              host.OnPerturbationMultithread5Periodicity();     return;
    case FractalCommand::PerturbationGpu:                                  host.OnPerturbationGpu();                         return;
    case FractalCommand::PerturbationLoad:                                 host.OnPerturbationLoad();                        return;
    case FractalCommand::PerturbationSave:                                 host.OnPerturbationSave();                        return;

    // ---- Memory / Autosave ----
    case FractalCommand::PerturbAutosaveOnDelete: host.OnPerturbAutosaveOnDelete(); return;
    case FractalCommand::PerturbAutosaveOn:       host.OnPerturbAutosaveOn();       return;
    case FractalCommand::PerturbAutosaveOff:      host.OnPerturbAutosaveOff();      return;
    case FractalCommand::MemoryLimit0:            host.OnMemoryLimit0();            return;
    case FractalCommand::MemoryLimit1:            host.OnMemoryLimit1();            return;

    // ---- Palette ----
    case FractalCommand::PaletteType0:            host.OnPaletteType0();            return;
    case FractalCommand::PaletteType1:            host.OnPaletteType1();            return;
    case FractalCommand::PaletteType2:            host.OnPaletteType2();            return;
    case FractalCommand::PaletteType3:            host.OnPaletteType3();            return;
    case FractalCommand::PaletteType4:            host.OnPaletteType4();            return;
    case FractalCommand::CreateNewPalette:        host.OnCreateNewPalette();        return;
    case FractalCommand::Palette5:                host.OnPalette5();                return;
    case FractalCommand::Palette6:                host.OnPalette6();                return;
    case FractalCommand::Palette8:                host.OnPalette8();                return;
    case FractalCommand::Palette12:               host.OnPalette12();               return;
    case FractalCommand::Palette16:               host.OnPalette16();               return;
    case FractalCommand::Palette20:               host.OnPalette20();               return;
    case FractalCommand::PaletteRotate:           host.OnPaletteRotate();           return;

    // ---- Save / Load ----
    case FractalCommand::SaveLocation:            host.OnSaveLocation();            return;
    case FractalCommand::SaveHiResBmp:            host.OnSaveHiResBmp();            return;
    case FractalCommand::SaveItersText:           host.OnSaveItersText();           return;
    case FractalCommand::SaveBmp:                 host.OnSaveBmp();                 return;
    case FractalCommand::SaveRefOrbitText:        host.OnSaveRefOrbitText();        return;
    case FractalCommand::SaveRefOrbitTextSimple:  host.OnSaveRefOrbitTextSimple();  return;
    case FractalCommand::SaveRefOrbitTextMax:     host.OnSaveRefOrbitTextMax();     return;
    case FractalCommand::SaveRefOrbitImagMax:     host.OnSaveRefOrbitImagMax();     return;
    case FractalCommand::DiffRefOrbitImagMax:     host.OnDiffRefOrbitImagMax();     return;
    case FractalCommand::LoadLocation:            host.OnLoadLocation();            return;
    case FractalCommand::LoadEnterLocation:       host.OnLoadEnterLocation();       return;
    case FractalCommand::LoadRefOrbitImagMax:     host.OnLoadRefOrbitImagMax();     return;
    case FractalCommand::LoadRefOrbitImagMaxSaved: host.OnLoadRefOrbitImagMaxSaved(); return;

    // ---- Tests / Benchmarks ----
    case FractalCommand::BasicTest:               host.OnBasicTest();               return;
    case FractalCommand::Test27:                  host.OnTest27();                  return;
    case FractalCommand::BenchmarkFull:           host.OnBenchmarkFull();           return;
    case FractalCommand::BenchmarkInt:            host.OnBenchmarkInt();            return;

    // ---- LA ----
    case FractalCommand::LaMultithreaded:         host.OnLaMultithreaded();         return;
    case FractalCommand::LaSinglethreaded:        host.OnLaSinglethreaded();        return;
    case FractalCommand::LaSettings1:             host.OnLaSettings1();             return;
    case FractalCommand::LaSettings2:             host.OnLaSettings2();             return;
    case FractalCommand::LaSettings3:             host.OnLaSettings3();             return;

    default:
        break;
    }

    // Unmigrated / range / payload-bearing: fall back to legacy.
    host.DispatchByIdm(static_cast<int>(IdmFromCommand(cmd)));
}

} // namespace FractalShark
