// PortableCommandHandlers.h
//
// Shared implementations of commands that only interact with Fractal. Native
// GUI shells provide the active Fractal and command position, and override the
// remaining hooks that require platform UI services.
#pragma once

#include "CommandCatalog.h"

class Fractal;

namespace FractalShark {

struct MenuPoint {
    int X;
    int Y;
};

class PortableCommandHandlers : public ExecuteCommandHost {
public:
    ~PortableCommandHandlers() override = default;

    // ---- Algorithm selection ---------------------------------------------
    void OnSetAlgorithm(::RenderAlgorithmEnum alg) override;

    // ---- Synthetic shortcut command hooks -------------------------------
    void OnAutoZoomDefaultAtPoint() override;
    void OnCenterViewClearPerturbation() override;
    void OnResetCompressionDefaults() override;
    void OnLaThresholdScaleIncrease() override;
    void OnLaThresholdScaleDecrease() override;
    void OnLaPeriodDetectionIncrease() override;
    void OnLaPeriodDetectionDecrease() override;
    void OnRecalcCurrentCopyDetails() override;
    void OnRecalcClearMediumCopyDetails() override;
    void OnRecalcClearAllCopyDetails() override;
    void OnRecalcClearLaCopyDetails() override;
    void OnIntermediateCompressionIncrease() override;
    void OnIntermediateCompressionDecrease() override;
    void OnLowCompressionIncrease() override;
    void OnLowCompressionDecrease() override;
    void OnPaletteAuxDepthNext() override;
    void OnPaletteAuxDepthPrevious() override;
    void OnPaletteDepthNext() override;
    void OnRecalcClearAllSquareView() override;

    // ---- Help / Window (only the platform-agnostic ones) -----------------
    void OnSquareView() override;
    void OnRepainting() override;

    // ---- Navigate --------------------------------------------------------
    void OnBack() override;
    void OnCenterView() override;
    void OnZoomIn() override;
    void OnZoomOut() override;
    void OnAutoZoomDefault() override;
    void OnAutoZoomMax() override;
    void OnAutoZoomFilament() override;
    void OnFeatureFinderDirect() override;
    void OnFeatureFinderDirectScan() override;
    void OnFeatureFinderPt() override;
    void OnFeatureFinderPtScan() override;
    void OnFeatureFinderLa() override;
    void OnFeatureFinderLaScan() override;
    void OnFeatureFinderZoom() override;
    void OnFeatureFinderClear() override;
    void OnFeatureFinderResume() override;
    void OnNrInnerLoopGpu() override;
    void OnNrInnerLoopCpu() override;
    void OnNrInnerLoopCpuSt() override;

    // ---- Built-In Views --------------------------------------------------
    void OnStandardView() override;
    void OnSelectBuiltInView(size_t oneBasedIndex) override;

    // ---- Antialiasing ----------------------------------------------------
    void OnGpuAntialiasing1x() override;
    void OnGpuAntialiasing4x() override;
    void OnGpuAntialiasing9x() override;
    void OnGpuAntialiasing16x() override;

    // ---- Iterations ------------------------------------------------------
    void OnResetIterations() override;
    void OnIncreaseIterations1p5x() override;
    void OnIncreaseIterations6x() override;
    void OnIncreaseIterations24x() override;
    void OnDecreaseIterations() override;
    void OnIterations32Bit() override;
    void OnIterations64Bit() override;

    // ---- Iteration precision ---------------------------------------------
    void OnIterationPrecision1x() override;
    void OnIterationPrecision2x() override;
    void OnIterationPrecision3x() override;
    void OnIterationPrecision4x() override;

    // ---- Perturbation ----------------------------------------------------
    void OnPerturbClearAll() override;
    void OnPerturbClearMed() override;
    void OnPerturbClearHigh() override;
    void OnPerturbationAuto() override;
    void OnPerturbationSinglethread() override;
    void OnPerturbationMultithread() override;
    void OnPerturbationSinglethreadPeriodicity() override;
    void OnPerturbationMultithread2Periodicity() override;
    void OnPerturbationMt2PerturbMthighStmed() override;
    void OnPerturbationMt2PerturbMthighMtmed1() override;
    void OnPerturbationMt2PerturbMthighMtmed2() override;
    void OnPerturbationMt2PerturbMthighMtmed3() override;
    void OnPerturbationMt2PerturbMthighMtmed4() override;
    void OnPerturbationMultithread5Periodicity() override;
    void OnPerturbationGpu() override;
    void OnPerturbationLoad() override;
    void OnPerturbationSave() override;

    // ---- Autosave --------------------------------------------------------
    void OnPerturbAutosaveOnDelete() override;
    void OnPerturbAutosaveOn() override;
    void OnPerturbAutosaveOff() override;

    // ---- Palette ---------------------------------------------------------
    void OnPaletteType0() override;
    void OnPaletteType1() override;
    void OnPaletteType2() override;
    void OnPaletteType3() override;
    void OnPaletteType4() override;
    void OnCreateNewPalette() override;
    void OnPalette5() override;
    void OnPalette6() override;
    void OnPalette8() override;
    void OnPalette12() override;
    void OnPalette16() override;
    void OnPalette20() override;

    // ---- Tests / Benchmarks ----------------------------------------------
    void OnBasicTest() override;
    void OnTest27() override;
    void OnBenchmarkFull() override;
    void OnBenchmarkInt() override;

    // ---- LA --------------------------------------------------------------
    void OnLaMultithreaded() override;
    void OnLaSinglethreaded() override;
    void OnLaSettings1() override;
    void OnLaSettings2() override;
    void OnLaSettings3() override;

protected:
    // Provided by the native main window.
    virtual Fractal &GetFractal() noexcept = 0;

    // Window-relative pixel position of the cursor at the moment the user
    // invoked the command.  Used by OnCenterView / OnZoomIn / OnZoomOut /
    // OnFeatureFinder*; ignored elsewhere.
    virtual MenuPoint GetMenuMousePos() const = 0;
};

} // namespace FractalShark
