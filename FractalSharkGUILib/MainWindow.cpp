#include "StdAfx.h"

#include "CommandDispatcher.h"
#include "ConsoleWindow.h"
#include "CrashHandler.h"
#include "CrummyTest.h"
#include "DynamicPopupMenu.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "MainWindowMenuState.h"
#include "MainWindowSavedLocation.h"
#include "OpenGLContext.h"
#include "PngParallelSave.h"

#include "RecommendedSettings.h"
#include "SplashWindow.h"
#include "WaitCursor.h"
#include "resource.h"
#include <Windowsx.h>

#include <commdlg.h>
#include <cstdio>
#include <random>

namespace {

constexpr bool startWindowed = true;
constexpr bool finishWindowed = false;
constexpr DWORD forceStartWidth = 0;
constexpr DWORD forceStartHeight = 0;
constexpr const wchar_t *kOrbitFileFilter = L"All\0*.*\0Imagina\0*.im\0";
constexpr const wchar_t *kPngFileFilter = L"PNG Image\0*.png\0All\0*.*\0";
constexpr const wchar_t *kTextFileFilter = L"Text File\0*.txt\0All\0*.*\0";

std::wstring
BuildHotkeysMessage()
{
    std::wstring body = L"Hotkeys\r\n\r\nCommand shortcuts\r\n";
    for (const FractalShark::Command &command : FractalShark::kCommands) {
        body += FractalShark::FormatHotKey(command.hotkey);
        body += L" - ";
        body.append(command.label.data(), command.label.size());
        body += L"\r\n";
    }

    body += L"\r\nDirect controls\r\n"
            L"Arrow keys - Pan viewport 25% of the view. Shift+Arrow: 10%, Ctrl+Arrow: 50%\r\n"
            L"Numpad + - Zoom in at center\r\n"
            L"Numpad - - Zoom out at center\r\n"
            L"Left click/drag - Zoom in\r\n"
            L"Right click - popup menu\r\n"
            L"CTRL - Press and hold to abort autozoom\r\n"
            L"ALT - Press, click/drag to move window when in windowed mode\r\n";
    return body;
}

bool
IsNumpadAddSubtractCharacter(WPARAM wParam, LPARAM lParam) noexcept
{
    constexpr UINT numpadAddScanCode = 0x4e;
    constexpr UINT numpadSubtractScanCode = 0x4a;
    const UINT scanCode = static_cast<UINT>((static_cast<DWORD_PTR>(lParam) >> 16) & 0xffu);

    return (wParam == L'+' && scanCode == numpadAddScanCode) ||
           (wParam == L'-' && scanCode == numpadSubtractScanCode);
}

void
DeleteExistingRegularFile(const std::wstring &filename)
{
    const DWORD attrs = ::GetFileAttributesW(filename.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0) {
        (void)::DeleteFileW(filename.c_str());
    }
}

} // namespace

MainWindow::MainWindow(HINSTANCE hInstance, int nCmdShow) : commandDispatcher(*this), hWnd{}
{
    gJobObj = std::make_unique<JobObject>();
    HighPrecision::defaultPrecisionInBits(256);

    CrashHandler::Install();

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // --- Splash (separate UI thread) ---
    Splash.Start(hInstance);

    auto startupBackgroundConsole = []() { AttachBackgroundConsole(true); };
    auto threadConsole = std::thread(startupBackgroundConsole);

    // Perform application initialization:
    hWnd = InitInstance(hInstance, nCmdShow);
    if (!hWnd) {
        throw FractalSharkSeriousException("Failed to create window.");
    }

    threadConsole.join();

    // Main window is ready; close splash now.
    Splash.Stop();

    ImaginaMenu = nullptr;
    LoadSubMenu = nullptr;
}

MainWindow::~MainWindow()
{
    // Cleanup
    UnInit();

    gJobObj.reset();
}

// ---- Per-menu host hook implementation -------------------------------------

void
MainWindow::OnSetAlgorithm(::RenderAlgorithmEnum alg)
{
    auto entry = GetRenderAlgorithmTupleEntry(alg);
    gFractal->EnqueueMutation(
        [entry](Fractal &f) { [[maybe_unused]] const bool ok = f.SetRenderAlgorithm(entry); });
}

// ---- Synthetic shortcut command hooks --------------------------------------
void
MainWindow::OnAutoZoomFeatureAtPoint()
{
    const POINT pt = GetSafeMenuPtClient();
    gFractal->AutoZoomFeatureAtPoint(pt.x, pt.y);
}

void
MainWindow::OnAutoZoomDefaultAtPoint()
{
    const POINT pt = GetSafeMenuPtClient();
    MenuCenterView(pt.x, pt.y);
    gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}

void
MainWindow::OnCenterViewClearPerturbation()
{
    const POINT pt = GetSafeMenuPtClient();
    gFractal->EnqueueCommand([x = pt.x, y = pt.y](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.CenterAtPoint(x, y);
    });
}

void
MainWindow::OnResetCompressionDefaults()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
        f.DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
    });
}

void
MainWindow::OnLaThresholdScaleIncrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustLAThresholdScaleExponent(1);
        laParameters.AdjustLAThresholdCScaleExponent(1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
MainWindow::OnLaThresholdScaleDecrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustLAThresholdScaleExponent(-1);
        laParameters.AdjustLAThresholdCScaleExponent(-1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
MainWindow::OnLaPeriodDetectionIncrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustPeriodDetectionThreshold2Exponent(1);
        laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
MainWindow::OnLaPeriodDetectionDecrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        auto &laParameters = f.GetLAParameters();
        laParameters.AdjustPeriodDetectionThreshold2Exponent(-1);
        laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(-1);
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
        f.ForceRecalc();
    });
}

void
MainWindow::OnRecalcCurrentCopyDetails()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.ForceRecalc(); }).Wait();
    MenuGetCurPos();
}

void
MainWindow::OnRecalcClearMediumCopyDetails()
{
    gFractal
        ->EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
            f.ForceRecalc();
        })
        .Wait();
    MenuGetCurPos();
}

void
MainWindow::OnRecalcClearAllCopyDetails()
{
    gFractal
        ->EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            f.ForceRecalc();
        })
        .Wait();
    MenuGetCurPos();
}

void
MainWindow::OnRecalcClearLaCopyDetails()
{
    gFractal
        ->EnqueueCommand([](Fractal &f) {
            f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
            f.ForceRecalc();
        })
        .Wait();
    MenuGetCurPos();
}

void
MainWindow::OnIntermediateCompressionIncrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
MainWindow::OnIntermediateCompressionDecrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Intermediate, 10);
    });
}

void
MainWindow::OnLowCompressionIncrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.IncCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
MainWindow::OnLowCompressionDecrease()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.DecCompressionError(Fractal::CompressionError::Low, 1);
    });
}

void
MainWindow::OnPaletteAuxDepthNext()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(1); });
}

void
MainWindow::OnPaletteAuxDepthPrevious()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.UseNextPaletteAuxDepth(-1); });
}

void
MainWindow::OnPaletteDepthNext()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.UseNextPaletteDepth(); });
}

void
MainWindow::OnRecalcClearAllSquareView()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        f.SquareCurrentView();
    });
}

// ---- Help / Window ---------------------------------------------------------
void
MainWindow::OnShowHotkeys()
{
    MenuShowHotkeys();
}
void
MainWindow::OnViewsHelp()
{
    MenuViewsHelp();
}
void
MainWindow::OnHelpAlg()
{
    MenuAlgHelp();
}
void
MainWindow::OnSquareView()
{
    MenuSquareView();
}
void
MainWindow::OnRepainting()
{
    MenuRepainting();
}
void
MainWindow::OnWindowed()
{
    MenuWindowed(false);
}
void
MainWindow::OnWindowedSq()
{
    MenuWindowed(true);
}
void
MainWindow::OnMinimize()
{
    ::PostMessage(hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0);
}
void
MainWindow::OnCurPos()
{
    MenuGetCurPos();
}
void
MainWindow::OnExit()
{
    ::DestroyWindow(hWnd);
}

// ---- Navigate --------------------------------------------------------------
void
MainWindow::OnBack()
{
    MenuGoBack();
}

void
MainWindow::OnCenterView()
{
    const POINT pt = GetSafeMenuPtClient();
    MenuCenterView(pt.x, pt.y);
}

void
MainWindow::OnZoomIn()
{
    const POINT pt = GetSafeMenuPtClient();
    MenuZoomIn(pt);
}

void
MainWindow::OnZoomOut()
{
    const POINT pt = GetSafeMenuPtClient();
    MenuZoomOut(pt);
}

void
MainWindow::OnAutoZoomDefault()
{
    gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
}
void
MainWindow::OnAutoZoomMax()
{
    gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>();
}
void
MainWindow::OnAutoZoomFilament()
{
    gFractal->AutoZoom<Fractal::AutoZoomHeuristic::FilamentTip>();
}

#define FRACTALSHARK_DEFINE_FEATURE_FINDER(Suffix, Mode)                                                \
    void MainWindow::OnFeatureFinder##Suffix()                                                          \
    {                                                                                                   \
        const POINT pt = GetSafeMenuPtClient();                                                         \
        auto x = pt.x;                                                                                  \
        auto y = pt.y;                                                                                  \
        gFractal->EnqueueCommand([x, y](Fractal &f) { f.TryFindPeriodicPoint(x, y, Mode); });           \
    }

FRACTALSHARK_DEFINE_FEATURE_FINDER(Direct, FeatureFinderMode::Direct)
FRACTALSHARK_DEFINE_FEATURE_FINDER(DirectScan, FeatureFinderMode::DirectScan)
FRACTALSHARK_DEFINE_FEATURE_FINDER(Pt, FeatureFinderMode::PT)
FRACTALSHARK_DEFINE_FEATURE_FINDER(PtScan, FeatureFinderMode::PTScan)
FRACTALSHARK_DEFINE_FEATURE_FINDER(La, FeatureFinderMode::LA)
FRACTALSHARK_DEFINE_FEATURE_FINDER(LaScan, FeatureFinderMode::LAScan)

#undef FRACTALSHARK_DEFINE_FEATURE_FINDER

void
MainWindow::OnFeatureFinderZoom()
{
    const POINT pt = GetSafeMenuPtClient();
    gFractal->EnqueueCommand([x = pt.x, y = pt.y](Fractal &f) { f.ZoomToFoundFeature(x, y); });
}

void
MainWindow::OnFeatureFinderClear()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.ClearAllFoundFeatures(); });
}

void
MainWindow::OnFeatureFinderResume()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.ResumeNRFromCheckpoint(); });
}

void
MainWindow::OnNrInnerLoopGpu()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::GPU); });
}

void
MainWindow::OnNrInnerLoopCpu()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuMT); });
}

void
MainWindow::OnNrInnerLoopCpuSt()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuST); });
}

// ---- Built-In Views (point entry) -----------------------------------------
void
MainWindow::OnStandardView()
{
    MenuStandardView(0);
}
void
MainWindow::OnSelectBuiltInView(size_t oneBasedIndex)
{
    MenuStandardView(oneBasedIndex);
}

// ---- Antialiasing ---------------------------------------------------------
void
MainWindow::OnGpuAntialiasing1x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1); });
}
void
MainWindow::OnGpuAntialiasing4x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2); });
}
void
MainWindow::OnGpuAntialiasing9x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.ResetDimensions(MAXSIZE_T, MAXSIZE_T, 3); });
}
void
MainWindow::OnGpuAntialiasing16x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4); });
}

// ---- Iterations -----------------------------------------------------------
void
MainWindow::OnResetIterations()
{
    MenuResetIterations();
}
void
MainWindow::OnIncreaseIterations1p5x()
{
    MenuMultiplyIterations(1.5);
}
void
MainWindow::OnIncreaseIterations6x()
{
    MenuMultiplyIterations(6.0);
}
void
MainWindow::OnIncreaseIterations24x()
{
    MenuMultiplyIterations(24.0);
}
void
MainWindow::OnDecreaseIterations()
{
    MenuMultiplyIterations(2.0 / 3.0);
}

void
MainWindow::OnIterations32Bit()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits32); });
}
void
MainWindow::OnIterations64Bit()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterType(IterTypeEnum::Bits64); });
}

// ---- Iteration precision --------------------------------------------------
void
MainWindow::OnIterationPrecision1x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(1); });
}
void
MainWindow::OnIterationPrecision2x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(4); });
}
void
MainWindow::OnIterationPrecision3x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(8); });
}
void
MainWindow::OnIterationPrecision4x()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetIterationPrecision(16); });
}

// ---- Perturbation ---------------------------------------------------------
void
MainWindow::OnPerturbResults()
{
    std::wcerr << L"TODO. By default these are shown as white pixels overlayed on the image. "
                  L"It'd be nice to have an option that shows them as white pixels against a "
                  L"black screen so their location is obvious.\n";
}

void
MainWindow::OnPerturbClearAll()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All); });
}
void
MainWindow::OnPerturbClearMed()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes); });
}
void
MainWindow::OnPerturbClearHigh()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes); });
}

void
MainWindow::OnPerturbationAuto()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto); });
}
void
MainWindow::OnPerturbationSinglethread()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST); });
}
void
MainWindow::OnPerturbationMultithread()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT); });
}
void
MainWindow::OnPerturbationSinglethreadPeriodicity()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity); });
}
void
MainWindow::OnPerturbationMultithread2Periodicity()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3); });
}
void
MainWindow::OnPerturbationMt2PerturbMthighStmed()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
    });
}
void
MainWindow::OnPerturbationMt2PerturbMthighMtmed1()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
    });
}
void
MainWindow::OnPerturbationMt2PerturbMthighMtmed2()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
    });
}
void
MainWindow::OnPerturbationMt2PerturbMthighMtmed3()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
    });
}
void
MainWindow::OnPerturbationMt2PerturbMthighMtmed4()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
    });
}
void
MainWindow::OnPerturbationMultithread5Periodicity()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5); });
}
void
MainWindow::OnPerturbationGpu()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU); });
}

void
MainWindow::OnPerturbationLoad()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.LoadPerturbationOrbits(); });
}
void
MainWindow::OnPerturbationSave()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SavePerturbationOrbits(); });
}

// ---- Memory / Autosave ---------------------------------------------------
void
MainWindow::OnPerturbAutosaveOnDelete()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithoutSave); });
}
void
MainWindow::OnPerturbAutosaveOn()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetResultsAutosave(AddPointOptions::EnableWithSave); });
}
void
MainWindow::OnPerturbAutosaveOff()
{
    gFractal->EnqueueMutation([](Fractal &f) { f.SetResultsAutosave(AddPointOptions::DontSave); });
}

void
MainWindow::OnMemoryLimit0()
{
    gJobObj = nullptr;
}
void
MainWindow::OnMemoryLimit1()
{
    gJobObj = std::make_unique<JobObject>();
}

// ---- Palette --------------------------------------------------------------
void
MainWindow::OnPaletteType0()
{
    MenuPaletteType(FractalPaletteType::Basic);
}
void
MainWindow::OnPaletteType1()
{
    MenuPaletteType(FractalPaletteType::Default);
}
void
MainWindow::OnPaletteType2()
{
    MenuPaletteType(FractalPaletteType::Patriotic);
}
void
MainWindow::OnPaletteType3()
{
    MenuPaletteType(FractalPaletteType::Summer);
}
void
MainWindow::OnPaletteType4()
{
    MenuPaletteType(FractalPaletteType::Random);
}
void
MainWindow::OnCreateNewPalette()
{
    MenuCreateNewPalette();
}
void
MainWindow::OnPalette5()
{
    MenuPaletteDepth(5);
}
void
MainWindow::OnPalette6()
{
    MenuPaletteDepth(6);
}
void
MainWindow::OnPalette8()
{
    MenuPaletteDepth(8);
}
void
MainWindow::OnPalette12()
{
    MenuPaletteDepth(12);
}
void
MainWindow::OnPalette16()
{
    MenuPaletteDepth(16);
}
void
MainWindow::OnPalette20()
{
    MenuPaletteDepth(20);
}
void
MainWindow::OnPaletteRotate()
{
    MenuPaletteRotation();
}

// ---- Save / Load ----------------------------------------------------------
void
MainWindow::OnSaveLocation()
{
    MenuSaveCurrentLocation();
}
void
MainWindow::OnSaveHiResBmp()
{
    MenuSaveHiResBMP();
}
void
MainWindow::OnSaveItersText()
{
    MenuSaveItersAsText();
}
void
MainWindow::OnSaveBmp()
{
    MenuSaveBMP();
}
void
MainWindow::OnSaveRefOrbitText()
{
    MenuSaveImag(CompressToDisk::Disable);
}
void
MainWindow::OnSaveRefOrbitTextSimple()
{
    MenuSaveImag(CompressToDisk::SimpleCompression);
}
void
MainWindow::OnSaveRefOrbitTextMax()
{
    MenuSaveImag(CompressToDisk::MaxCompression);
}
void
MainWindow::OnSaveRefOrbitImagMax()
{
    MenuSaveImag(CompressToDisk::MaxCompressionImagina);
}
void
MainWindow::OnDiffRefOrbitImagMax()
{
    MenuDiffImag();
}
void
MainWindow::OnLoadLocation()
{
    MenuLoadCurrentLocation();
}
void
MainWindow::OnLoadEnterLocation()
{
    MenuLoadEnterLocation();
}
void
MainWindow::OnLoadRefOrbitImagMax()
{
    MenuLoadImagDyn(ImaginaSettings::ConvertToCurrent);
}
void
MainWindow::OnLoadRefOrbitImagMaxSaved()
{
    MenuLoadImagDyn(ImaginaSettings::UseSaved);
}

// ---- Tests / Benchmarks ---------------------------------------------------
void
MainWindow::OnBasicTest()
{
    WaitCursor waitCursor;
    CrummyTest t{*gFractal};
    t.TestAll();
}
void
MainWindow::OnTest27()
{
    WaitCursor waitCursor;
    CrummyTest t{*gFractal};
    t.TestReallyHardView27();
}
void
MainWindow::OnBenchmarkFull()
{
    WaitCursor waitCursor;
    CrummyTest t{*gFractal};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::All);
}
void
MainWindow::OnBenchmarkInt()
{
    WaitCursor waitCursor;
    CrummyTest t{*gFractal};
    t.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
}

// ---- LA -------------------------------------------------------------------
void
MainWindow::OnLaMultithreaded()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
    });
}
void
MainWindow::OnLaSinglethreaded()
{
    gFractal->EnqueueMutation([](Fractal &f) {
        f.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    });
}
void
MainWindow::OnLaSettings1()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxAccuracy); });
}
void
MainWindow::OnLaSettings2()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxPerf); });
}
void
MainWindow::OnLaSettings3()
{
    gFractal->EnqueueMutation(
        [](Fractal &f) { f.GetLAParameters().SetDefaults(LAParameters::LADefaults::MinMemory); });
}

void
MainWindow::MainWindow::DrawFractalShark()
{
    auto glContext = std::make_unique<OpenGlContext>(static_cast<void *>(hWnd));
    if (!glContext->IsValid()) {
        return;
    }

    glContext->DrawFractalShark(static_cast<void *>(hWnd));
}

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM
MainWindow::MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(wcex);

    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)StaticWndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = sizeof(MainWindow *);
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIconW(hInstance, (LPCWSTR)IDI_FRACTALS);
    wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);

    // IMPORTANT: avoid the default white erase/paint before your first draw.
    // This makes the client area reliably black from the first frame.
    wcex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIconW(hInstance, (LPCWSTR)IDI_SMALL);

    return RegisterClassExW(&wcex);
}

HWND
MainWindow::InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    (void)nCmdShow; // ignore debugger-dependent startup show state
    hInst = hInstance;

    // Use *virtual* screen metrics (multi-monitor correct) and signed ints.
    const int vLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vTop = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int vWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    const int vHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    // Default "start windowed" geometry (you can tweak)
    const int size = std::min(vWidth / 2, vHeight / 2);
    const int startX = vLeft + (vWidth - size) / 2;
    const int startY = vTop + (vHeight - size) / 2;
    int width = size;
    int height = size;

    if constexpr (forceStartWidth)
        width = (int)forceStartWidth;
    if constexpr (forceStartHeight)
        height = (int)forceStartHeight;

    // Create MAIN WINDOW hidden (do NOT show until gFractal is ready)
    DWORD style = WS_OVERLAPPEDWINDOW; // no WS_VISIBLE
    DWORD exStyle = WS_EX_APPWINDOW;

    hWnd = CreateWindowExW(exStyle,
                           szWindowClass,
                           L"FractalShark",
                           style,
                           startX,
                           startY,
                           width,
                           height,
                           nullptr,
                           nullptr,
                           hInstance,
                           nullptr);

    if (!hWnd) {
        return nullptr;
    }

    // Store "this" in the extra bytes immediately (StaticWndProc needs it)
    SetWindowLongPtrW(hWnd, 0, (LONG_PTR)this);

    // Don't set default algorithm on menu yet; wait until gFractal exists.

    // Apply your preferred mode while still hidden
    SetModeWindowed(startWindowed);

    // If you want to end fullscreen immediately, do it here (still hidden)
    if constexpr (finishWindowed == false) {

        SetModeWindowed(false);

        // Pick monitor nearest current window
        HMONITOR mon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
        MONITORINFO mi{};
        mi.cbSize = sizeof(mi);
        GetMonitorInfoW(mon, &mi);
        RECT r = mi.rcMonitor;

        // Force "normal" placement so Windows stops restoring elsewhere
        WINDOWPLACEMENT wp{};
        wp.length = sizeof(wp);
        wp.showCmd = 0;
        wp.flags = 0;
        wp.ptMinPosition = {0, 0};
        wp.ptMaxPosition = {0, 0};
        wp.rcNormalPosition = r;
        SetWindowPlacement(hWnd, &wp);

        SetWindowPos(hWnd,
                     HWND_NOTOPMOST,
                     r.left,
                     r.top,
                     r.right - r.left,
                     r.bottom - r.top,
                     SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE);
    }

    // Create fractal using FINAL client size
    RECT rt{};
    GetClientRect(hWnd, &rt);

    gFractal = std::make_unique<Fractal>(
        rt.right, rt.bottom, static_cast<void *>(hWnd), false, gJobObj->GetCommitLimitInBytes());

    // Create menu / popup (doesn't require the window to be shown)
    MainWindowMenuState menuState(*this);
    gPopupMenu = FractalShark::DynamicPopupMenu::Create(menuState);

    // SetRenderAlgorithm: TODO kind of gross but it works, reset now that
    // gFractal exists.  If CPU-only is enforced, this will show the radio
    // button the menu properly.  Without this, the menu is out of sync until
    // the user changes algorithm manually.
    FractalShark::ExecuteCommand(FractalShark::CommandFromIdm(IDM_ALG_AUTO), *this);
    // commandDispatcher.Dispatch(IDM_ALG_GPU_HDR_64_PERTURB_LAV2);

    // Optional: force an initial black fill before first show (prevents any flash)
    {
        HDC dc = GetDC(hWnd);
        RECT rc{};
        GetClientRect(hWnd, &rc);
        FillRect(dc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));
        ReleaseDC(hWnd, dc);
    }

    // Now show exactly once
    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return hWnd;
}

void
MainWindow::ApplyBorderlessWindowedStyle()
{
    LONG_PTR style = GetWindowLongPtrW(hWnd, GWL_STYLE);
    style &= ~(WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX);
    style |= (WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX); // THICKFRAME is the resize enabler
    SetWindowLongPtrW(hWnd, GWL_STYLE, style);

    SetWindowPos(hWnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
}

void
MainWindow::ApplyBorderlessFullscreenStyle()
{
    LONG_PTR style = GetWindowLongPtrW(hWnd, GWL_STYLE);
    style &= ~(WS_THICKFRAME | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
    style |= WS_POPUP;
    SetWindowLongPtrW(hWnd, GWL_STYLE, style);

    SetWindowPos(hWnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
}

void
MainWindow::SetModeWindowed(bool enableWindowed)
{
    gWindowed = enableWindowed;
    if (enableWindowed) {
        ApplyBorderlessWindowedStyle();
        // SetWindowPos to some desired restored rect
    } else {
        ApplyBorderlessFullscreenStyle();
        // SetWindowPos to monitor rect
    }
}

//
// Performs all cleanup operations
//
void
MainWindow::UnInit()
{
    ClearMenu(LoadSubMenu);
    ClearMenu(ImaginaMenu);

    DestroyWindow(hWnd);
    UnregisterClass(szWindowClass, hInst);
    gFractal.reset();
    gJobObj.reset();
}

void
MainWindow::HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM lParam)
{
    if (IsNumpadAddSubtractCharacter(wParam, lParam)) {
        return;
    }

    POINT mousePt;
    if (::GetCursorPos(&mousePt) == 0 || ::ScreenToClient(hWnd, &mousePt) == 0) {
        return;
    }
    lastMenuPtClient_ = mousePt;

    const FractalShark::HotKey hotkey = FractalShark::HotKeyFromCharacter(
        static_cast<wchar_t>(wParam), IsDownShift(), IsDownControl(), IsDownAlt());
    if (!hotkey.HasKey()) {
        return;
    }

    if (const FractalShark::Command *command = FractalShark::FindCommandByHotKey(hotkey);
        command != nullptr) {
        FractalShark::ExecuteCommand(command->id, *this);
    }
}

void
MainWindow::HandleArrowAndZoomKeys(WPARAM vk)
{
    bool shiftDown = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    bool ctrlDown = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;

    // Pan fraction: 10% (Shift), 50% (Ctrl), 25% (default)
    double frac = 0.25;
    if (shiftDown) {
        frac = 0.10;
    } else if (ctrlDown) {
        frac = 0.50;
    }

    switch (vk) {
        case VK_LEFT:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(-frac, 0.0); });
            break;
        case VK_RIGHT:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(frac, 0.0); });
            break;
        case VK_UP:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(0.0, frac); });
            break;
        case VK_DOWN:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(0.0, -frac); });
            break;
        case VK_ADD:
            gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(-0.3); });
            break;
        case VK_SUBTRACT:
            gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); });
            break;
        default:
            break;
    }
}

LRESULT
MainWindow::StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    // Get the window class from the hWnd
    MainWindow *pThis = reinterpret_cast<MainWindow *>(GetWindowLongPtr(hWnd, 0));

    if (pThis == nullptr) {
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    auto copyToClipboard = [](std::string str) {
        if (OpenClipboard(nullptr)) {
            HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, str.size() + 1);
            if (hMem != nullptr) {
                char *pMem = (char *)GlobalLock(hMem);
                if (pMem != nullptr) {
                    memcpy(pMem, str.c_str(), str.size() + 1);
                    GlobalUnlock(hMem);
                    EmptyClipboard();
                    if (SetClipboardData(CF_TEXT, hMem) == nullptr) {
                        GlobalFree(hMem);
                    }
                } else {
                    GlobalFree(hMem);
                }
            }
            CloseClipboard();
        }
    };

    // And invoke WndProc
    if (IsDebuggerPresent()) {
        return pThis->WndProc(message, wParam, lParam);
    } else {
        try {
            return pThis->WndProc(message, wParam, lParam);
        } catch (const FractalSharkSeriousException &e) {
            const auto msg = e.GetCallstack("Message copied to clipboard.  CTRL-V to paste.");
            copyToClipboard(msg);
            std::cerr << msg.c_str() << std::endl;
            return 0;
        } catch (const std::exception &e) {
            copyToClipboard(e.what());
            std::cerr << e.what() << std::endl;
            return 0;
        }
    }
}

std::wstring
MainWindow::OpenFileDialog(OpenBoxType type,
                           const wchar_t *filter,
                           const wchar_t *defaultExtension,
                           unsigned long saveFlags)
{
    OPENFILENAME ofn;    // common dialog box structure
    wchar_t szFile[260]; // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
    ofn.lpstrFilter = filter != nullptr ? filter : kOrbitFileFilter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.lpstrDefExt = defaultExtension;

    if (type == OpenBoxType::Open) {
        // Display the Open dialog box.
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        if (GetOpenFileName(&ofn) == TRUE) {
            return std::wstring(ofn.lpstrFile);
        } else {
            return std::wstring();
        }
    } else {
        ofn.Flags = saveFlags;
        if (GetSaveFileName(&ofn) == TRUE) {
            return std::wstring(ofn.lpstrFile);
        } else {
            return std::wstring();
        }
    }
}

bool
MainWindow::HasLastMenuPtClient() const noexcept
{
    return lastMenuPtClient_.x >= 0 && lastMenuPtClient_.y >= 0;
}

POINT
MainWindow::GetSafeMenuPtClient() const
{
    // If user hasn’t opened the context menu yet, fall back to cursor pos.
    POINT pt = lastMenuPtClient_;

    if (!HasLastMenuPtClient()) {
        ::GetCursorPos(&pt);
        ::ScreenToClient(hWnd, &pt);
    }
    return pt;
}

void
MainWindow::ActivateSavedOrbit(size_t index)
{
    ClearMenu(LoadSubMenu);

    const auto ptz = gSavedLocations[index].ptz;
    const auto num_iterations = gSavedLocations[index].num_iterations;
    const auto antialiasing = gSavedLocations[index].antialiasing;

    gFractal->EnqueueCommand([ptz, num_iterations, antialiasing](Fractal &f) {
        f.RecenterViewCalc(ptz);
        f.SetNumIterations<IterTypeFull>(num_iterations);
        f.ResetDimensions(MAXSIZE_T, MAXSIZE_T, antialiasing);
    });
}

void
MainWindow::ActivateImagina(size_t index)
{
    const auto &entry = gImaginaLocations[index];

    LoadRefOrbit(CompressToDisk::MaxCompressionImagina, entry.Settings, entry.Filename);
    ClearMenu(ImaginaMenu);
}

LRESULT
MainWindow::WndProc(UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) {
        case WM_NCACTIVATE: {
            if (gWindowed) {
                // Prevent Windows from drawing the standard non-client border
                // when activating/deactivating (focus changes).
                return TRUE;
            }
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

        case WM_NCPAINT: {
            if (gWindowed) {
                // Suppress non-client painting entirely in borderless windowed mode.
                return 0;
            }
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

        case WM_NCCALCSIZE:
            if (gWindowed) {
                return 0; // Remove all standard non-client area
            }
            break;

        case WM_NCHITTEST: {
            if (!gWindowed)
                break;

            constexpr int resizeBorder = 8; // pixels
            POINT pt{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
            ScreenToClient(hWnd, &pt);

            RECT rc;
            GetClientRect(hWnd, &rc);

            const bool left = pt.x < resizeBorder;
            const bool right = pt.x >= rc.right - resizeBorder;
            const bool top = pt.y < resizeBorder;
            const bool bottom = pt.y >= rc.bottom - resizeBorder;

            if (top && left)
                return HTTOPLEFT;
            if (top && right)
                return HTTOPRIGHT;
            if (bottom && left)
                return HTBOTTOMLEFT;
            if (bottom && right)
                return HTBOTTOMRIGHT;
            if (left)
                return HTLEFT;
            if (right)
                return HTRIGHT;
            if (top)
                return HTTOP;
            if (bottom)
                return HTBOTTOM;

            return HTCLIENT;
        }

        case WM_COMMAND: {
            const int wmId = LOWORD(wParam);
            const int wmEvent = HIWORD(wParam);
            (void)wmEvent;

            // Native-only dynamic/dialog IDs sit below the portable catalog
            // range and must keep their raw command id payload.
            if (wmId < static_cast<int>(FractalShark::FractalCommand::ShowHotkeys)) {
                if (commandDispatcher.Dispatch(wmId)) {
                    return 0;
                }
            } else {
                // Static catalog commands go through ExecuteCommand.
                const auto cmd = FractalShark::CommandFromIdm(static_cast<uint32_t>(wmId));
                FractalShark::ExecuteCommand(cmd, *this);
                return 0;
            }

            wchar_t buf[256];
            swprintf_s(buf,
                       L"Unknown WM_COMMAND.\n\nwmId = %d (0x%X)\nwmEvent = %d (0x%X)\n",
                       wmId,
                       wmId,
                       wmEvent,
                       wmEvent);

            std::wcerr << buf << L" Unknown menu item" << std::endl;

            return 0;
        }

        case WM_SIZE: {
            if (gFractal) {
                auto w = LOWORD(lParam);
                auto h = HIWORD(lParam);
                gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
            }
            return 0;
        }

        case WM_CONTEXTMENU: {
            // WM_CONTEXTMENU:
            //  - If invoked by mouse: lParam contains screen coords (x,y)
            //  - If invoked by keyboard (Shift+F10 / menu key): lParam == -1
            POINT ptScreen{};

            if (lParam == static_cast<LPARAM>(-1)) {
                // Keyboard invocation: use current cursor position as the popup anchor.
                ::GetCursorPos(&ptScreen);
            } else {
                // Mouse invocation: lParam is already in SCREEN coordinates.
                ptScreen.x = GET_X_LPARAM(lParam);
                ptScreen.y = GET_Y_LPARAM(lParam);
            }

            // Rebuild the menu each time so dynamic state (radio/toggles/enabled) is fresh.
            MainWindowMenuState menuState(*this);
            gPopupMenu = FractalShark::DynamicPopupMenu::Create(menuState);

            HMENU popup = FractalShark::DynamicPopupMenu::GetPopup(gPopupMenu.get());
            if (!popup) {
                return 0;
            }

            ::TrackPopupMenu(popup,
                             TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN,
                             ptScreen.x,
                             ptScreen.y,
                             0,
                             hWnd,
                             nullptr);

            // Persist menu location as CLIENT coords on the instance (used by Center/Zoom commands).
            POINT ptClient = ptScreen;
            ::ScreenToClient(hWnd, &ptClient);
            lastMenuPtClient_ = ptClient;

            return 0;
        }

        case WM_LBUTTONDOWN: {
            if (gWindowed && IsDownAlt()) {
                ::PostMessage(hWnd, WM_NCLBUTTONDOWN, HTCAPTION, lParam);
                return 0;
            }

            if (lButtonDown)
                return 0;

            lButtonDown = true;
            dragBoxX1 = GET_X_LPARAM(lParam);
            dragBoxY1 = GET_Y_LPARAM(lParam);
            prevX1 = prevY1 = -1;

            ::SetCapture(hWnd);
            return 0;
        }

        case WM_LBUTTONUP: {
            if (!lButtonDown || IsDownAlt()) {
                if (::GetCapture() == hWnd)
                    ::ReleaseCapture();
                return 0;
            }

            // release capture early so we don’t get stuck captured on exceptions/returns
            if (::GetCapture() == hWnd)
                ::ReleaseCapture();

            lButtonDown = false;
            prevX1 = prevY1 = -1;

            // Clear the GL drag-rect overlay.
            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(false, 0, 0, 0, 0);
            }

            RECT newViewWin{};
            const bool maintainAspect = !IsDownShift();

            if (maintainAspect) {
                RECT windowRect;
                ::GetClientRect(hWnd, &windowRect);
                const double ratio = double(windowRect.right) / double(windowRect.bottom);

                newViewWin.left = dragBoxX1;
                newViewWin.top = dragBoxY1;
                newViewWin.bottom = GET_Y_LPARAM(lParam);
                newViewWin.right = LONG(double(newViewWin.left) +
                                        ratio * (double(newViewWin.bottom) - double(newViewWin.top)));
            } else {
                newViewWin.left = dragBoxX1;
                newViewWin.top = dragBoxY1;
                newViewWin.right = GET_X_LPARAM(lParam);
                newViewWin.bottom = GET_Y_LPARAM(lParam);
            }

            Environment::ScreenRect newView{
                newViewWin.left, newViewWin.top, newViewWin.right, newViewWin.bottom};

            if (gFractal) {
                gFractal->EnqueueCommand([newView, maintainAspect](Fractal &f) {
                    if (f.RecenterViewScreen(newView)) {
                        if (maintainAspect)
                            f.SquareCurrentView();
                    }
                });
            }

            return 0;
        }

        case WM_CANCELMODE:
        case WM_CAPTURECHANGED: {
            if (!lButtonDown)
                return 0;

            // Clear the GL drag-rect overlay.
            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(false, 0, 0, 0, 0);
            }

            lButtonDown = false;
            prevX1 = prevY1 = -1;

            if (::GetCapture() == hWnd)
                ::ReleaseCapture();

            return 0;
        }

        case WM_MOUSEMOVE: {
            if (lButtonDown == false) {
                return 0;
            }

            int rectX1, rectY1;

            if (IsDownShift() == false) {
                RECT windowRect;
                GetClientRect(hWnd, &windowRect);
                double ratio = (double)windowRect.right / (double)windowRect.bottom;

                rectY1 = GET_Y_LPARAM(lParam);
                rectX1 = (int)((double)dragBoxX1 + (double)ratio * (double)(rectY1 - dragBoxY1));

                prevX1 = rectX1;
                prevY1 = rectY1;
            } else {
                rectX1 = GET_X_LPARAM(lParam);
                rectY1 = GET_Y_LPARAM(lParam);

                prevX1 = rectX1;
                prevY1 = rectY1;
            }

            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(true, dragBoxX1, dragBoxY1, rectX1, rectY1);
            }

            break;
        }

        case WM_MOUSEWHEEL: {
            // Mouse wheel zoom control
            //
            // Windows convention:
            //   GET_WHEEL_DELTA_WPARAM(wParam) > 0  => wheel rotated FORWARD (away from user)
            //   GET_WHEEL_DELTA_WPARAM(wParam) < 0  => wheel rotated BACKWARD (toward user)
            //
            // UI convention (FractalShark):
            //   Wheel FORWARD  = zoom IN
            //   Wheel BACKWARD = zoom OUT

            POINT pt;
            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);

            // Convert to client coordinates
            ScreenToClient(hWnd, &pt);

            if (GET_WHEEL_DELTA_WPARAM(wParam) > 0) {
                gFractal->EnqueueCommand(
                    [x = pt.x, y = pt.y](Fractal &f) { f.ZoomTowardPoint(x, y, -0.3); });
            } else {
                gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); });
            }

            return 0;
        }

        case WM_PAINT: {
            PaintAsNecessary();

            PAINTSTRUCT ps;
            BeginPaint(hWnd, &ps);
            EndPaint(hWnd, &ps);
            return 0;
        }

        case WM_DESTROY: {
            PostQuitMessage(0);
            return 0;
        }

        case WM_CHAR: {
            HandleKeyDown(message, wParam, lParam);
            return 0;
        }

        case WM_KEYDOWN: {
            HandleArrowAndZoomKeys(wParam);
            break;
        }

        default:
            break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

void
MainWindow::MenuGoBack()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.Back(); });
}

void
MainWindow::MenuStandardView(size_t i)
{
    gFractal->EnqueueCommand([i](Fractal &f) { f.View(i); });
}

void
MainWindow::MenuSquareView()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.SquareCurrentView(); });
}

void
MainWindow::MenuCenterView(int x, int y)
{
    gFractal->EnqueueCommand([x, y](Fractal &f) { f.CenterAtPoint(x, y); });
}

void
MainWindow::MenuZoomIn(POINT mousePt)
{
    gFractal->EnqueueCommand(
        [x = mousePt.x, y = mousePt.y](Fractal &f) { f.ZoomRecentered(x, y, -.45); });
}

void
MainWindow::MenuZoomOut(POINT mousePt)
{
    gFractal->EnqueueCommand([x = mousePt.x, y = mousePt.y](Fractal &f) { f.ZoomRecentered(x, y, 1); });
}

void
MainWindow::MenuRepainting()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.ToggleRepainting(); });
}

void
MainWindow::MenuWindowed(bool square)
{
    if (gWindowed == false) {
        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true) {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SendMessage(hWnd, WM_SYSCOMMAND, SC_RESTORE, 0);

        if (temporaryChange == true) {
            gFractal->SetRepaint(true);
        }

        RECT rect;
        GetWindowRect(hWnd, &rect);

        if (square) {
            auto width = std::min((rect.right + rect.left) / 2, (rect.bottom + rect.top) / 2);
            // width /= 2;
            SetWindowPos(hWnd,
                         HWND_NOTOPMOST,
                         (rect.right + rect.left) / 2 - width / 2,
                         (rect.bottom + rect.top) / 2 - width / 2,
                         width,
                         width,
                         SWP_SHOWWINDOW);
        } else {
            SetWindowPos(hWnd,
                         HWND_NOTOPMOST,
                         (rect.right - rect.left) / 4,
                         (rect.bottom - rect.top) / 4,
                         (rect.right - rect.left) / 2,
                         (rect.bottom - rect.top) / 2,
                         SWP_SHOWWINDOW);
        }

        SetModeWindowed(true);

        if (gFractal) {
            RECT rt;
            GetClientRect(hWnd, &rt);
            auto w = rt.right;
            auto h = rt.bottom;
            gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
        }
    } else {
        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true) {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SetWindowPos(hWnd, HWND_NOTOPMOST, 0, 0, width, height, SWP_SHOWWINDOW);
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

        if (temporaryChange == true) {
            gFractal->SetRepaint(true);
        }

        SetModeWindowed(false);

        if (gFractal) {
            RECT rt;
            GetClientRect(hWnd, &rt);
            auto w = rt.right;
            auto h = rt.bottom;
            gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
        }
    }
}

void
MainWindow::MenuMultiplyIterations(double factor)
{
    gFractal->EnqueueCommand([factor](Fractal &f) {
        if (f.GetIterType() == IterTypeEnum::Bits32) {
            uint64_t curIters = f.GetNumIterations<uint32_t>();
            curIters = (uint64_t)((double)curIters * factor);
            f.SetNumIterations<uint32_t>(curIters);
        } else {
            uint64_t curIters = f.GetNumIterations<uint64_t>();
            curIters = (uint64_t)((double)curIters * factor);
            f.SetNumIterations<uint64_t>(curIters);
        }
    });
}

void
MainWindow::MenuResetIterations()
{
    gFractal->EnqueueCommand([](Fractal &f) { f.ResetNumIterations(); });
}

void
MainWindow::MenuGetCurPos()
{
    constexpr size_t numBytes = 4 * 1024 * 1024;

    BOOL ret = OpenClipboard(hWnd);
    if (ret == 0) {
        std::wcerr << L"Opening the clipboard failed.  Another program must be using it." << std::endl;
        return;
    }

    ret = EmptyClipboard();
    if (ret == 0) {
        std::wcerr << L"Emptying the clipboard of its current contents failed.  Make sure no other "
                      L"programs are using it."
                   << std::endl;
        CloseClipboard();
        return;
    }

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, numBytes);
    if (hData == nullptr) {
        std::wcerr << L"Insufficient memory." << std::endl;
        CloseClipboard();
        return;
    }

    char *mem = (char *)GlobalLock(hData);
    if (mem == nullptr) {
        std::wcerr << L"Insufficient memory 2." << std::endl;
        GlobalFree(hData);
        CloseClipboard();
        return;
    }

    std::string shortStr, longStr;
    gFractal->GetRenderDetails(shortStr, longStr);

    // Append temp2 to mem without overrunning the buffer
    // using strncat.
    mem[0] = 0;
    strncpy(mem, longStr.data(), numBytes - 1);
    mem[numBytes - 1] = '\0';

    GlobalUnlock(hData);

    //
    // This is not a memory leak - we don't "free" hData.
    //

    HANDLE clpData = SetClipboardData(CF_TEXT, hData);
    if (clpData == nullptr) {
        std::wcerr << L"Adding the data to the clipboard failed.  You are probably very low on memory.  "
                      L"Try closing other programs or restarting your computer."
                   << std::endl;
        GlobalFree(hData);
        CloseClipboard();
        return;
    }

    CloseClipboard();

    if (shortStr.length() < 5000) {
        ::MessageBoxA(hWnd, shortStr.c_str(), "Current Position", MB_OK | MB_APPLMODAL);
    } else {
        std::wcerr << L"Location copied to clipboard." << std::endl;
    }
}

void
MainWindow::MenuPaletteRotation()
{
    // TODO(palette-rotation): This is visually broken because RotateFractalPalette() updates the
    // internal palette state without forcing the displayed image to be re-rendered or recolored,
    // so the command appears to do nothing.
    POINT OrgPos, CurPos;
    GetCursorPos(&OrgPos);

    for (;;) {
        gFractal->EnqueueCommand([](Fractal &f) { f.RotateFractalPalette(10); }).Wait();
        GetCursorPos(&CurPos);
        if (abs(CurPos.x - OrgPos.x) > 5 || abs(CurPos.y - OrgPos.y) > 5) {
            break;
        }
    }

    gFractal->EnqueueCommand([](Fractal &f) { f.ResetFractalPalette(); });
}

void
MainWindow::MenuPaletteType(FractalPaletteType type)
{
    gFractal->EnqueueCommand([type](Fractal &f) {
        f.UsePaletteType(type);
        if (type == FractalPaletteType::Default) {
            f.UsePalette(8);
            f.SetPaletteAuxDepth(0);
        }
    });
}

void
MainWindow::MenuPaletteDepth(int depth)
{
    gFractal->EnqueueCommand([depth](Fractal &f) { f.UsePalette(depth); });
}

void
MainWindow::MenuCreateNewPalette()
{
    gFractal->EnqueueCommand([](Fractal &f) {
        f.CreateNewFractalPalette();
        f.UsePaletteType(FractalPaletteType::Random);
    });
}

void
MainWindow::MenuSaveCurrentLocation()
{
    int response = ::MessageBox(hWnd, L"Scale dimensions to maximum?", L"Choose!", MB_YESNO);
    char filename[256];
    SYSTEMTIME time_struct;
    GetLocalTime(&time_struct);
    snprintf(filename,
             sizeof(filename),
             "output_%d_%d_%d_%d_%d_%d.bmp",
             time_struct.wYear,
             time_struct.wMonth,
             time_struct.wDay,
             time_struct.wHour,
             time_struct.wMinute,
             time_struct.wSecond);

    size_t x, y;
    if (response == IDYES) {
        x = gFractal->GetRenderWidth();
        y = gFractal->GetRenderHeight();
        if (x > y) {
            y = (int)((double)16384.0 / (double)((double)x / (double)y));
            x = 16384;
        } else if (x < y) {
            x = (int)((double)16384.0 / (double)((double)y / (double)x));
            y = 16384;
        }
    } else {
        x = gFractal->GetRenderWidth();
        y = gFractal->GetRenderHeight();
    }

    std::stringstream ss;
    ss << x << " ";
    ss << y << " ";
    ss << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
    ss << gFractal->GetMinX() << " ";
    ss << gFractal->GetMinY() << " ";
    ss << gFractal->GetMaxX() << " ";
    ss << gFractal->GetMaxY() << " ";
    ss << gFractal->GetNumIterations<IterTypeFull>() << " ";
    ss << gFractal->GetGpuAntialiasing() << " ";
    // ss << gFractal->GetIterationPrecision() << " ";
    ss << "FractalTrayDestination";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());

    MessageBox(nullptr, ws.c_str(), L"location", MB_OK | MB_APPLMODAL);

    FILE *file = fopen("locations.txt", "at+");
    if (file != nullptr) {
        fprintf(file, "%s\r\n", s.c_str());
        fclose(file);
    }
}

void
MainWindow::MenuLoadCurrentLocation()
{
    std::ifstream infile("locations.txt");
    HMENU hSubMenu = CreatePopupMenu();

    size_t index = 0;

    gSavedLocations.clear();

    for (;;) {
        SavedLocation loc(infile);
        if (infile.rdstate() != std::ios_base::goodbit) {
            break;
        }

        // Convert loc.description to a wstring:
        std::string s = loc.description;
        const std::wstring ws(s.begin(), s.end());

        gSavedLocations.push_back(loc);
        AppendMenu(hSubMenu, MF_STRING, IDM_VIEW_DYNAMIC_ORBIT + index, ws.c_str());
        index++;

        // Limit the number of locations we show.
        if (index > 30) {
            break;
        }
    }

    POINT point;
    GetCursorPos(&point);
    TrackPopupMenu(
        hSubMenu, TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN, point.x, point.y, 0, hWnd, nullptr);

    DestroyMenu(hSubMenu);
}

// Subclass procedure for the edit controls
LRESULT
MainWindow::EditSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (uMsg == WM_KEYDOWN) {
        if ((wParam == 'A') && (GetKeyState(VK_CONTROL) & 0x8000)) {
            // CTRL+A pressed, select all text
            SendMessage(hwnd, EM_SETSEL, 0, -1);
            return 0;
        }
    }
    // Call the original window procedure for the edit control
    WNDPROC originalProc = (WNDPROC)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    return CallWindowProc(originalProc, hwnd, uMsg, wParam, lParam);
}

void
MainWindow::MenuLoadEnterLocation()
{
    // Create a window with three text boxes for entering the location.
    // The text boxes are for the real, imaginary, and zoom values.
    // Store the results from the text boxes in three strings.
    // Set the fractal to the new location and repaint.
    // Include OK and Cancel buttons.
    // Use the IDD_DIALOG_LOCATION resource

    // Define EnterLocationDialogProc:
    // This is a dialog box procedure that handles the dialog box messages.
    // It should handle WM_COMMAND, and WM_CLOSE.
    // WM_COMMAND should handle the OK and Cancel buttons.
    // If the OK button is pressed, it should store the values in the text boxes
    // in the strings, and then call EndDialog(hWnd, 0).
    // If the Cancel button is pressed, it should call EndDialog(hWnd, 1).
    // WM_CLOSE should call EndDialog(hWnd, 1).

    struct Values {
        Values() : real(""), imag(""), zoom(""), num_iterations(0) {}

        std::string real, imag, zoom;
        IterTypeFull num_iterations;

        std::string
        ItersToString() const
        {
            return std::to_string(num_iterations);
        }

        void
        StringToIters(std::string new_iters)
        {
            num_iterations = std::stoull(new_iters);
        }
    };

    Values values;

    // Store existing location in the strings.
    HighPrecision minX = gFractal->GetMinX();
    HighPrecision minY = gFractal->GetMinY();
    HighPrecision maxX = gFractal->GetMaxX();
    HighPrecision maxY = gFractal->GetMaxY();

    PointZoomBBConverter pz{minX, minY, maxX, maxY, PointZoomBBConverter::TestMode::Enabled};
    values.real = pz.GetPtX().str();
    values.imag = pz.GetPtY().str();
    values.zoom = pz.GetZoomFactor().str();
    values.num_iterations = gFractal->GetNumIterations<IterTypeFull>();

    auto EnterLocationDialogProc = [](HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) -> INT_PTR {
        // TODO: static?  This is surely not the right way to do this.
        static Values *values = nullptr;

        switch (message) {
            case WM_INITDIALOG: {
                // Get the pointer to the Values struct from lParam.
                values = (Values *)lParam;

                RECT rcDlg, rcScreen;
                int x, y;

                // Get the dimensions of the dialog box
                GetWindowRect(hDlg, &rcDlg);

                // Get the dimensions of the screen
                GetClientRect(GetDesktopWindow(), &rcScreen);

                // Calculate the position to center the dialog box
                x = (rcScreen.right - (rcDlg.right - rcDlg.left)) / 2;
                y = (rcScreen.bottom - (rcDlg.bottom - rcDlg.top)) / 2;

                // Move the dialog box to the calculated position
                SetWindowPos(hDlg, NULL, x, y, 0, 0, SWP_NOZORDER | SWP_NOSIZE);

                // Set the text in the text boxes to the values in the strings.
                SetDlgItemTextA(hDlg, IDC_EDIT_REAL, values->real.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_IMAG, values->imag.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_ZOOM, values->zoom.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_ITERATIONS, values->ItersToString().c_str());

                // Subclass the edit controls
                HWND hEditReal = GetDlgItem(hDlg, IDC_EDIT_REAL);
                HWND hEditImag = GetDlgItem(hDlg, IDC_EDIT_IMAG);
                HWND hEditZoom = GetDlgItem(hDlg, IDC_EDIT_ZOOM);
                HWND hEditIterations = GetDlgItem(hDlg, IDC_EDIT_ITERATIONS);

                auto OriginalEditProcReal =
                    (WNDPROC)SetWindowLongPtr(hEditReal, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditReal, GWLP_USERDATA, (LONG_PTR)OriginalEditProcReal);

                auto OriginalEditProcImag =
                    (WNDPROC)SetWindowLongPtr(hEditImag, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditImag, GWLP_USERDATA, (LONG_PTR)OriginalEditProcImag);

                auto OriginalEditProcZoom =
                    (WNDPROC)SetWindowLongPtr(hEditZoom, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditZoom, GWLP_USERDATA, (LONG_PTR)OriginalEditProcZoom);

                auto OriginalEditProcIterations =
                    (WNDPROC)SetWindowLongPtr(hEditIterations, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditIterations, GWLP_USERDATA, (LONG_PTR)OriginalEditProcIterations);

                break;
            }

            case WM_COMMAND: {
                if (LOWORD(wParam) == IDOK) {
                    // Get the text from the text boxes.
                    // Store the text in the strings.
                    // Call EndDialog(hDlg, 0);

                    // First, figure out how many bytes are needed
                    // to store the text in the text boxes.
                    int len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_REAL));
                    values->real.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_REAL), &values->real[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_IMAG));
                    values->imag.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_IMAG), &values->imag[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_ZOOM));
                    values->zoom.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_ZOOM), &values->zoom[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_ITERATIONS));
                    std::string new_iters;
                    new_iters.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_ITERATIONS), &new_iters[0], len + 1);
                    values->StringToIters(new_iters);

                    EndDialog(hDlg, 0);
                    return TRUE;
                } else if (LOWORD(wParam) == IDCANCEL) {
                    // Call EndDialog(hDlg, 1);
                    EndDialog(hDlg, 1);
                    return TRUE;
                }
                break;
            }
            case WM_CLOSE: {
                // Call EndDialog(hDlg, 1);
                EndDialog(hDlg, 1);
                return TRUE;
            }
        }

        return FALSE;
    };

    // (hInst, MAKEINTRESOURCE(IDD_DIALOG_LOCATION), hWnd, Dlgproc);
    LPARAM lParam = reinterpret_cast<LPARAM>(&values);
    auto OkOrCancel = DialogBoxParam(
        hInst, MAKEINTRESOURCE(IDD_DIALOG_LOCATION), hWnd, EnterLocationDialogProc, lParam);

    if (values.real.empty() || values.imag.empty() || values.zoom.empty()) {
        return;
    }

    // If OkOrCancel is 1, return.
    if (OkOrCancel == 1) {
        return;
    }

    // Convert the strings to HighPrecision and set the fractal to the new location.
    HighPrecision::defaultPrecisionInBits(Fractal::MaxPrecisionLame);
    HighPrecision realHP(values.real);
    HighPrecision imagHP(values.imag);
    HighPrecision zoomHP(values.zoom);

    PointZoomBBConverter pz2{realHP, imagHP, zoomHP, PointZoomBBConverter::TestMode::Enabled};
    auto numIters = values.num_iterations;
    gFractal->EnqueueCommand([pz2 = std::move(pz2), numIters](Fractal &f) {
        f.RecenterViewCalc(pz2);
        f.SetNumIterations<IterTypeFull>(numIters);
    });
}

void
MainWindow::MenuSaveBMP()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kPngFileFilter, L"png", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (filename.empty()) {
        return;
    }

    // Save must read m_CurIters directly, so drain the render pool first
    // to ensure no workers are active, then call synchronously.
    if (auto *pool = gFractal->GetRenderPool()) {
        pool->Drain();
    }
    DeleteExistingRegularFile(filename);
    gFractal->SaveCurrentFractal(std::move(filename), true);
}

void
MainWindow::MenuSaveHiResBMP()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kPngFileFilter, L"png", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (filename.empty()) {
        return;
    }

    if (auto *pool = gFractal->GetRenderPool()) {
        pool->Drain();
    }
    DeleteExistingRegularFile(filename);
    gFractal->SaveHiResFractal(std::move(filename));
}

void
MainWindow::MenuSaveItersAsText()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kTextFileFilter, L"txt", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (filename.empty()) {
        return;
    }

    if (auto *pool = gFractal->GetRenderPool()) {
        pool->Drain();
    }
    DeleteExistingRegularFile(filename);
    gFractal->SaveItersAsText(std::move(filename));
}

void
MainWindow::BenchmarkMessage(size_t milliseconds)
{
    std::stringstream ss;
    ss << std::string("Time taken in ms: ") << milliseconds << ".";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());
    MessageBox(hWnd, ws.c_str(), L"", MB_OK | MB_APPLMODAL);
}

void
MainWindow::ClearMenu(HMENU &menu)
{
    if (menu != nullptr) {
        DestroyMenu(menu);
        menu = nullptr;
        gImaginaLocations.clear();
    }
}

void
MainWindow::LoadRefOrbit(CompressToDisk compressToDisk,
                         ImaginaSettings loadSettings,
                         std::wstring filename)
{
    gFractal->EnqueueCommand([compressToDisk, loadSettings, filename = std::move(filename)](Fractal &f) {
        RecommendedSettings settings{};
        f.LoadRefOrbit(&settings, compressToDisk, loadSettings, filename);

        if (settings.GetRenderAlgorithm() == RenderAlgorithmEnum::AUTO) {
            const bool success = f.SetRenderAlgorithm(settings.GetRenderAlgorithm());
            if (!success) {
                std::wcerr << L"Warning: Could not set render algorithm to AUTO." << std::endl;
            }
        }
    });
}

void
MainWindow::MenuLoadImagDyn(ImaginaSettings loadSettings)
{
    ClearMenu(ImaginaMenu);

    ImaginaMenu = CreatePopupMenu();
    size_t index = 0;

    std::vector<std::wstring> imagFiles;
    // Find all files with the extension .im in current directory.
    // Add filenames to imagFiles.

    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile(L"*.im", &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            imagFiles.push_back(FindFileData.cFileName);
        } while (FindNextFile(hFind, &FindFileData) != 0);

        FindClose(hFind);
    }

    // Even if no files found, just run the following so we get
    // an empty menu.

    for (const auto &imagFile : imagFiles) {

        gImaginaLocations.push_back({imagFile, loadSettings});

        AppendMenu(ImaginaMenu, MF_STRING, IDM_VIEW_DYNAMIC_IMAG + index, imagFile.c_str());
        index++;

        if (index > 30) {
            break;
        }
    }

    if (loadSettings == ImaginaSettings::ConvertToCurrent) {
        AppendMenu(ImaginaMenu, MF_STRING, IDM_LOAD_IMAGINA_DLG, L"Load from file (Match)...");
    } else if (loadSettings == ImaginaSettings::UseSaved) {
        AppendMenu(ImaginaMenu, MF_STRING, IDM_LOAD_IMAGINA_DLG_SAVED, L"Load from file (Use Saved)...");
    } else {
    }

    POINT point;
    GetCursorPos(&point);
    TrackPopupMenu(
        ImaginaMenu, TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN, point.x, point.y, 0, hWnd, nullptr);

    DestroyMenu(ImaginaMenu);
}

void
MainWindow::MenuSaveImag(CompressToDisk compression)
{
    std::wstring filename = OpenFileDialog(OpenBoxType::Save);
    if (filename.empty()) {
        return;
    }

    gFractal->EnqueueCommand([compression, filename = std::move(filename)](Fractal &f) {
        f.SaveRefOrbit(compression, filename);
    });
}

void
MainWindow::MenuDiffImag()
{

    std::wstring outFile = OpenFileDialog(OpenBoxType::Save);
    if (outFile.empty()) {
        return;
    }

    // Open two files, both must exist
    std::wstring filename1 = OpenFileDialog(OpenBoxType::Open);
    if (filename1.empty()) {
        return;
    }

    std::wstring filename2 = OpenFileDialog(OpenBoxType::Open);
    if (filename2.empty()) {
        return;
    }

    gFractal->EnqueueCommand([outFile = std::move(outFile),
                              filename1 = std::move(filename1),
                              filename2 = std::move(filename2)](Fractal &f) {
        f.DiffRefOrbits(CompressToDisk::MaxCompressionImagina, outFile, filename1, filename2);
    });
}

void
MainWindow::MenuLoadImag(ImaginaSettings loadSettings, CompressToDisk compression)
{

    std::wstring filename = OpenFileDialog(OpenBoxType::Open);
    if (filename.empty()) {
        return;
    }

    LoadRefOrbit(compression, loadSettings, filename);
}

void
MainWindow::MenuAlgHelp()
{
    // This message box shows some help related to the algorithms.
    ::MessageBox(
        nullptr,
        L"Algorithms\r\n"
        L"\r\n"
        L"- As a general recommendation, choose AUTO.  Auto will render the fractal using "
        L"direct 32-bit evaluation at the lowest zoom depths. "
        L"From 1e4 to 1e9, it uses perturbation + 32-bit floating point. "
        L"From 1e9 to 1e34, it uses perturbation + 32-bit + linear approximation.  "
        L"Past that, it uses perturbation a 32-bit \"high dynamic range\" implementation, "
        L"which simply stores the exponent in a separate integer.\r\n"
        L"\r\n"
        L"- If you try rendering \"hard\" points, you may find that the 32-bit implementations "
        L"are not accurate enough.  In this case, you can try the 64-bit implementations.  "
        L"You may also find the 2x32 implementations to be faster than the 1x64."
        L"Generally, it's probably easiest to use the 32-bit implementations, and only "
        L"switch to the 64-bit implementations when you need to.\r\n"
        L"\r\n"
        L"Note that professional/high-end chips offer superior 64-bit performance, so if you have one "
        L"of those, you may find that the 64-bit implementations work well.  Most consumer GPUs offer"
        L"poor 64-bit performance (even RTX 4090, 5090 etc)\r\n",
        L"Algorithms",
        MB_OK);
}

void
MainWindow::MenuViewsHelp()
{
    ::MessageBox(nullptr,
                 L"Views\r\n"
                 L"\r\n"
                 L"The purpose of these is simply to make it easy to navigate to\r\n"
                 L"some interesting locations.\r\n",
                 L"Views",
                 MB_OK);
}

void
MainWindow::MenuShowHotkeys()
{
    const std::wstring body = BuildHotkeysMessage();
    ::MessageBox(nullptr, body.c_str(), L"Hotkeys", MB_OK);
}

void
MainWindow::PaintAsNecessary()
{
    RECT rt;
    GetClientRect(hWnd, &rt);

    if (rt.left == 0 && rt.right == 0 && rt.top == 0 && rt.bottom == 0) {
        return;
    }

    if (gFractal != nullptr) {
        gFractal->EnqueueCommand([](Fractal &) {});
    }
}
