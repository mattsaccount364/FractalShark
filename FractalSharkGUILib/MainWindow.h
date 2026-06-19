#pragma once

#include <memory>
#include <string>
#include <vector>

#include "CommandCatalog.h"
#include "CommandDispatcher.h"
#include "SplashWindow.h"
#include "UniqueHMenu.h"

class Fractal;

namespace Environment {
class JobObject;
} // namespace Environment

enum FractalPaletteType : size_t;
enum class CompressToDisk;
enum class ImaginaSettings;

namespace FractalShark::Win32 {

class MainWindowMenuState;

class MainWindow : public ExecuteCommandHost {
    friend class CommandDispatcher;
    friend class MainWindowMenuState;

public:
    struct SavedLocation;
    struct ImaginaSavedLocation;

    MainWindow(HINSTANCE hInstance, int nCmdShow);
    ~MainWindow();

    // ---- FractalShark::ExecuteCommandHost -------------------------------
    void OnSetAlgorithm(::RenderAlgorithmEnum alg) override;

    // Synthetic shortcut command hooks
    void OnAutoZoomFeatureAtPoint() override;
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

    // Help / Window
    void OnShowHotkeys() override;
    void OnViewsHelp() override;
    void OnHelpAlg() override;
    void OnSquareView() override;
    void OnRepainting() override;
    void OnWindowed() override;
    void OnWindowedSq() override;
    void OnMinimize() override;
    void OnCurPos() override;
    void OnExit() override;

    // Navigate
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

    // Built-In Views
    void OnStandardView() override;
    void OnSelectBuiltInView(size_t oneBasedIndex) override;

    // Antialiasing
    void OnGpuAntialiasing1x() override;
    void OnGpuAntialiasing4x() override;
    void OnGpuAntialiasing9x() override;
    void OnGpuAntialiasing16x() override;

    // Iterations
    void OnResetIterations() override;
    void OnIncreaseIterations1p5x() override;
    void OnIncreaseIterations6x() override;
    void OnIncreaseIterations24x() override;
    void OnDecreaseIterations() override;
    void OnIterations32Bit() override;
    void OnIterations64Bit() override;

    // Iteration precision
    void OnIterationPrecision1x() override;
    void OnIterationPrecision2x() override;
    void OnIterationPrecision3x() override;
    void OnIterationPrecision4x() override;

    // Perturbation
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

    // Memory / Autosave
    void OnPerturbAutosaveOnDelete() override;
    void OnPerturbAutosaveOn() override;
    void OnPerturbAutosaveOff() override;

    // Palette
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
    void OnPaletteRotate() override;

    // Save / Load
    void OnSaveLocation() override;
    void OnSaveHiResBmp() override;
    void OnSaveItersText() override;
    void OnSaveBmp() override;
    void OnSaveRefOrbitText() override;
    void OnSaveRefOrbitTextSimple() override;
    void OnSaveRefOrbitTextMax() override;
    void OnSaveRefOrbitImagMax() override;
    void OnDiffRefOrbitImagMax() override;
    void OnLoadLocation() override;
    void OnLoadEnterLocation() override;
    void OnLoadRefOrbitImagMax() override;
    void OnLoadRefOrbitImagMaxSaved() override;

    // Tests / Benchmarks
    void OnBasicTest() override;
    void OnTest27() override;
    void OnBenchmarkFull() override;
    void OnBenchmarkInt() override;

    // LA
    void OnLaMultithreaded() override;
    void OnLaSinglethreaded() override;
    void OnLaSettings1() override;
    void OnLaSettings2() override;
    void OnLaSettings3() override;

private:
    SplashWindow Splash;

    std::vector<SavedLocation> gSavedLocations;
    std::vector<ImaginaSavedLocation> gImaginaLocations;

    // Global Variables:
    std::unique_ptr<Environment::JobObject> gJobObj;

    HINSTANCE hInst; // current instance
    LPCWSTR szWindowClass = L"FractalWindow";
    UniqueHMenu gPopupMenu;
    bool gWindowed; // Says whether we are in windowed mode or not.
    HWND hWnd;

    HMENU LoadSubMenu;
    HMENU ImaginaMenu;

    // Fractal:
    std::unique_ptr<Fractal> gFractal;

    ATOM MyRegisterClass(HINSTANCE hInstance);
    HWND InitInstance(HINSTANCE, int);
    void ApplyBorderlessWindowedStyle();
    void ApplyBorderlessFullscreenStyle();
    void SetModeWindowed(bool windowed);
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT CALLBACK WndProc(UINT message, WPARAM wParam, LPARAM lParam);
    void UnInit();
    void HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM lParam);
    void HandleArrowAndZoomKeys(WPARAM vk);

    void DrawFractalShark();

    enum class OpenBoxType { Open, Save };

    static std::wstring OpenFileDialog(OpenBoxType type,
                                       const wchar_t *filter = nullptr,
                                       const wchar_t *defaultExtension = nullptr,
                                       unsigned long saveFlags = 0);

    void ActivateSavedOrbit(size_t index);
    void ActivateImagina(size_t index);

    // ---- Chunk A: persist menu click location as member state ----
    // Stored in CLIENT coordinates. Set by WM_CONTEXTMENU.
    POINT lastMenuPtClient_{-1, -1};

    // --- Mouse drag/zoom box state (was function-local statics in WndProc) ---
    bool lButtonDown = false;
    int dragBoxX1 = 0;
    int dragBoxY1 = 0;

    // Used for drawing the inverted rectangle properly.
    int prevX1 = -1;
    int prevY1 = -1;

    CommandDispatcher commandDispatcher;

    bool HasLastMenuPtClient() const noexcept;
    POINT GetSafeMenuPtClient() const;
    // ---------------------------------------------------------------------

    // Controlling functions
    void MenuGoBack();
    void MenuStandardView(size_t i);
    void MenuSquareView();
    void MenuCenterView(int x, int y);
    void MenuZoomIn(POINT mousePt);
    void MenuZoomOut(POINT mousePt);
    void MenuRepainting();
    void MenuWindowed(bool square);
    void MenuMultiplyIterations(double factor);
    void MenuResetIterations();
    void MenuPaletteType(FractalPaletteType type);
    void MenuPaletteDepth(int depth);
    void MenuPaletteRotation();
    void MenuCreateNewPalette();
    void MenuGetCurPos();
    void MenuSaveCurrentLocation();
    void MenuLoadCurrentLocation();
    static LRESULT EditSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    void MenuLoadEnterLocation();
    void MenuSaveBMP();
    void MenuSaveHiResBMP();
    void MenuSaveItersAsText();
    void BenchmarkMessage(size_t milliseconds);
    void MenuAlgHelp();
    void MenuViewsHelp();
    void MenuLoadImagDyn(ImaginaSettings loadSettings);
    void MenuSaveImag(CompressToDisk compression);
    void MenuDiffImag();
    void MenuLoadImag(ImaginaSettings loadSettings, CompressToDisk compression);
    void MenuShowHotkeys();

    void PaintAsNecessary();

    void ClearMenu(HMENU &menu);
    void LoadRefOrbit(CompressToDisk compressToDisk,
                      ImaginaSettings loadSettings,
                      std::wstring filename);

    bool
    IsDownControl()
    {
        return (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000;
    };
    bool
    IsDownShift()
    {
        return (GetAsyncKeyState(VK_SHIFT) & 0x8000) == 0x8000;
    };
    bool
    IsDownAlt()
    {
        return (GetAsyncKeyState(VK_MENU) & 0x8000) == 0x8000;
    };
};

} // namespace FractalShark::Win32
