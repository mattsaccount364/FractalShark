// PortableCommandHandlers.h
//
// Shared command dispatch and implementations for the native GUI shells.
// Native shells provide the active Fractal, command position, and operations
// that require platform UI services.
#pragma once

#include "CommandCatalog.h"

class Fractal;
enum class RenderAlgorithmEnum : uint32_t;

namespace FractalShark {

struct MenuPoint {
    int X;
    int Y;
};

class PortableCommandHandlers {
public:
    virtual ~PortableCommandHandlers() = default;

    void ExecuteCommand(FractalCommand command);

protected:
    virtual Fractal &GetFractal() noexcept = 0;
    virtual MenuPoint GetMenuMousePos() const = 0;

    virtual void OnAutoZoomFeatureAtPoint() = 0;

    virtual void OnShowHotkeys() = 0;
    virtual void OnViewsHelp() = 0;
    virtual void OnHelpAlg() = 0;
    virtual void OnWindowed() = 0;
    virtual void OnWindowedSq() = 0;
    virtual void OnMinimize() = 0;
    virtual void OnCurPos() = 0;
    virtual void OnExit() = 0;

    virtual void OnPaletteRotate() = 0;

    virtual void OnSaveLocation() = 0;
    virtual void OnSaveHiResBmp() = 0;
    virtual void OnSaveItersText() = 0;
    virtual void OnSaveBmp() = 0;
    virtual void OnSaveRefOrbitText() = 0;
    virtual void OnSaveRefOrbitTextSimple() = 0;
    virtual void OnSaveRefOrbitTextMax() = 0;
    virtual void OnSaveRefOrbitImagMax() = 0;
    virtual void OnDiffRefOrbitImagMax() = 0;
    virtual void OnLoadLocation() = 0;
    virtual void OnLoadEnterLocation() = 0;
    virtual void OnLoadRefOrbitImagMax() = 0;
    virtual void OnLoadRefOrbitImagMaxSaved() = 0;

private:
    void OnSetAlgorithm(::RenderAlgorithmEnum alg);

    void OnAutoZoomDefaultAtPoint();
    void OnCenterViewClearPerturbation();
    void OnResetCompressionDefaults();
    void OnLaThresholdScaleIncrease();
    void OnLaThresholdScaleDecrease();
    void OnLaPeriodDetectionIncrease();
    void OnLaPeriodDetectionDecrease();
    void OnRecalcCurrentCopyDetails();
    void OnRecalcClearMediumCopyDetails();
    void OnRecalcClearAllCopyDetails();
    void OnRecalcClearLaCopyDetails();
    void OnIntermediateCompressionIncrease();
    void OnIntermediateCompressionDecrease();
    void OnLowCompressionIncrease();
    void OnLowCompressionDecrease();
    void OnPaletteAuxDepthNext();
    void OnPaletteAuxDepthPrevious();
    void OnPaletteDepthNext();
    void OnRecalcClearAllSquareView();

    void OnSquareView();
    void OnRepainting();

    void OnBack();
    void OnCenterView();
    void OnZoomIn();
    void OnZoomOut();
    void OnAutoZoomDefault();
    void OnAutoZoomMax();
    void OnAutoZoomFilament();
    void OnFeatureFinderDirect();
    void OnFeatureFinderDirectScan();
    void OnFeatureFinderPt();
    void OnFeatureFinderPtScan();
    void OnFeatureFinderLa();
    void OnFeatureFinderLaScan();
    void OnFeatureFinderZoom();
    void OnFeatureFinderClear();
    void OnFeatureFinderResume();
    void OnNrInnerLoopGpu();
    void OnNrInnerLoopCpu();
    void OnNrInnerLoopCpuSt();

    void OnStandardView();
    void OnSelectBuiltInView(size_t oneBasedIndex);

    void OnGpuAntialiasing1x();
    void OnGpuAntialiasing4x();
    void OnGpuAntialiasing9x();
    void OnGpuAntialiasing16x();

    void OnResetIterations();
    void OnIncreaseIterations1p5x();
    void OnIncreaseIterations6x();
    void OnIncreaseIterations24x();
    void OnDecreaseIterations();
    void OnIterations32Bit();
    void OnIterations64Bit();

    void OnIterationPrecision1x();
    void OnIterationPrecision2x();
    void OnIterationPrecision3x();
    void OnIterationPrecision4x();

    void OnPerturbClearAll();
    void OnPerturbClearMed();
    void OnPerturbClearHigh();
    void OnPerturbationAuto();
    void OnPerturbationSinglethread();
    void OnPerturbationMultithread();
    void OnPerturbationSinglethreadPeriodicity();
    void OnPerturbationMultithread2Periodicity();
    void OnPerturbationMt2PerturbMthighStmed();
    void OnPerturbationMt2PerturbMthighMtmed1();
    void OnPerturbationMt2PerturbMthighMtmed2();
    void OnPerturbationMt2PerturbMthighMtmed3();
    void OnPerturbationMt2PerturbMthighMtmed4();
    void OnPerturbationMultithread5Periodicity();
    void OnPerturbationGpu();
    void OnPerturbationLoad();
    void OnPerturbationSave();

    void OnPerturbAutosaveOnDelete();
    void OnPerturbAutosaveOn();
    void OnPerturbAutosaveOff();

    void OnPaletteType0();
    void OnPaletteType1();
    void OnPaletteType2();
    void OnPaletteType3();
    void OnPaletteType4();
    void OnCreateNewPalette();
    void OnPalette5();
    void OnPalette6();
    void OnPalette8();
    void OnPalette12();
    void OnPalette16();
    void OnPalette20();

    void OnBasicTest();
    void OnTest27();
    void OnBenchmarkFull();
    void OnBenchmarkInt();

    void OnLaMultithreaded();
    void OnLaSinglethreaded();
    void OnLaSettings1();
    void OnLaSettings2();
    void OnLaSettings3();
};

} // namespace FractalShark
