// CommandDispatcher.cpp
#include "StdAfx.h"
#include "CommandDispatcher.h"

#include "CrummyTest.h"
#include "DynamicPopupMenu.h"
#include "Fractal.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "MainWindowSavedLocation.h"
#include "RecommendedSettings.h"
#include "resource.h"

#include <array>
#include <utility>

namespace FractalShark {

// ---- CommandDispatcher::Alg helpers ----

const AlgCmd *
FindAlgForCmd(int wmId) noexcept
{
    for (const auto &e : kAlgCmds) {
        if (e.id == wmId)
            return &e;
    }
    return nullptr;
}

int
FindCmdForAlg(RenderAlgorithmEnum alg) noexcept
{
    for (const auto &e : kAlgCmds) {
        if (e.alg == alg)
            return e.id;
    }
    return -1;
}

// ---- lifecycle ----

CommandDispatcher::CommandDispatcher(MainWindow &owner) : w_(owner) { BuildTable(); }

bool
CommandDispatcher::Dispatch(int wmId)
{
    if (HandleCommandRange(wmId))
        return true;
    if (HandleAlgCommand(wmId))
        return true;
    if (HandleCommandTable(wmId))
        return true;
    return false;
}

bool
CommandDispatcher::HandleCommandRange(int wmId)
{
    // ---- Views 1..40 (contiguous) ----
    if (wmId >= IDM_VIEW1 && wmId <= IDM_VIEW40) {
        static_assert(IDM_VIEW40 == IDM_VIEW1 + 39, "IDM_VIEW range must be contiguous");
        w_.MenuStandardView(static_cast<size_t>(wmId - IDM_VIEW1 + 1));
        return true;
    }

    // ---- Dynamic orbit slots (0..29) ----
    if (wmId >= IDM_VIEW_DYNAMIC_ORBIT && wmId < IDM_VIEW_DYNAMIC_ORBIT + kMaxDynamic) {
        const size_t index = static_cast<size_t>(wmId - IDM_VIEW_DYNAMIC_ORBIT);
        if (index < w_.gSavedLocations.size()) {
            w_.ActivateSavedOrbit(index);
        }
        return true;
    }

    // ---- Dynamic imag slots (0..29) ----
    if (wmId >= IDM_VIEW_DYNAMIC_IMAG && wmId < IDM_VIEW_DYNAMIC_IMAG + kMaxDynamic) {
        const size_t index = static_cast<size_t>(wmId - IDM_VIEW_DYNAMIC_IMAG);
        if (index < w_.gImaginaLocations.size()) {
            w_.ActivateImagina(index);
        }
        return true;
    }

    return false;
}

bool
CommandDispatcher::HandleAlgCommand(int wmId)
{
    const auto *e = FindAlgForCmd(wmId);
    if (!e)
        return false;

    const bool success = w_.gFractal->SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(e->alg));
    if (!success) {
        // The selection may not be accepted (e.g., GPU not working).
        const auto selectedAlg = w_.gFractal->GetRenderAlgorithm();
        const auto selectedWmid = FindCmdForAlg(selectedAlg.Algorithm);
        if (selectedWmid >= 0) {
            wmId = selectedWmid;
        }
    }

    // NEW: no global "current render algorithm id" + no post-build radio pass.
    // Your menu state provider is now the source of truth.
    // So: nothing to do here other than apply the change.

    return true;
}

bool
CommandDispatcher::HandleCommandTable(int wmId)
{
    if (auto it = table_.find(wmId); it != table_.end()) {
        it->second(w_);
        return true;
    }
    return false;
}

void
CommandDispatcher::BuildTable()
{
    // Function-pointer table: no allocations per entry, no captures.
    table_.reserve(256);

    // ---------- helpers (no captures) ----------
    auto doCenter = +[](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuCenterView(pt.x, pt.y);
    };
    auto doZoomIn = +[](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuZoomIn(pt);
    };
    auto doZoomOut = +[](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuZoomOut(pt);
    };

    auto doPeriodic = [](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.gFractal->TryFindPeriodicPoint(pt.x, pt.y);
    };

    // Navigation / view
    table_.emplace(IDM_BACK, +[](MainWindow &w) { w.MenuGoBack(); });
    table_.emplace(IDM_STANDARDVIEW, +[](MainWindow &w) { w.MenuStandardView(0); });
    table_.emplace(IDM_SQUAREVIEW, +[](MainWindow &w) { w.MenuSquareView(); });
    table_.emplace(IDM_VIEWS_HELP, +[](MainWindow &w) { w.MenuViewsHelp(); });
    table_.emplace(IDM_CENTERVIEW, doCenter);
    table_.emplace(IDM_ZOOMIN, doZoomIn);
    table_.emplace(IDM_ZOOMOUT, doZoomOut);

    table_.emplace(
        IDM_AUTOZOOM_DEFAULT,
        +[](MainWindow &w) { w.gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>(); });
    table_.emplace(
        IDM_AUTOZOOM_MAX,
        +[](MainWindow &w) { w.gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>(); });
    table_.emplace(IDM_FEATUREFINDER, doPeriodic);

    table_.emplace(IDM_REPAINTING, +[](MainWindow &w) { w.MenuRepainting(); });
    table_.emplace(IDM_WINDOWED, +[](MainWindow &w) { w.MenuWindowed(false); });
    table_.emplace(IDM_WINDOWED_SQ, +[](MainWindow &w) { w.MenuWindowed(true); });
    table_.emplace(
        IDM_MINIMIZE, +[](MainWindow &w) { ::PostMessage(w.hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0); });

    // GPU AA
    table_.emplace(
        IDM_GPUANTIALIASING_1X,
        +[](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1); });
    table_.emplace(
        IDM_GPUANTIALIASING_4X,
        +[](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2); });
    table_.emplace(
        IDM_GPUANTIALIASING_9X,
        +[](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 3); });
    table_.emplace(
        IDM_GPUANTIALIASING_16X,
        +[](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4); });

    // Iteration precision
    table_.emplace(
        IDM_ITERATIONPRECISION_1X, +[](MainWindow &w) { w.gFractal->SetIterationPrecision(1); });
    table_.emplace(
        IDM_ITERATIONPRECISION_2X, +[](MainWindow &w) { w.gFractal->SetIterationPrecision(4); });
    table_.emplace(
        IDM_ITERATIONPRECISION_3X, +[](MainWindow &w) { w.gFractal->SetIterationPrecision(8); });
    table_.emplace(
        IDM_ITERATIONPRECISION_4X, +[](MainWindow &w) { w.gFractal->SetIterationPrecision(16); });

    // Algorithm help
    table_.emplace(IDM_HELP_ALG, +[](MainWindow &w) { w.MenuAlgHelp(); });

    // LA toggles / presets
    table_.emplace(
        IDM_LA_SINGLETHREADED, +[](MainWindow &w) {
            auto &p = w.gFractal->GetLAParameters();
            p.SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
        });
    table_.emplace(
        IDM_LA_MULTITHREADED, +[](MainWindow &w) {
            auto &p = w.gFractal->GetLAParameters();
            p.SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
        });
    table_.emplace(
        IDM_LA_SETTINGS_1, +[](MainWindow &w) {
            auto &p = w.gFractal->GetLAParameters();
            p.SetDefaults(LAParameters::LADefaults::MaxAccuracy);
        });
    table_.emplace(
        IDM_LA_SETTINGS_2, +[](MainWindow &w) {
            auto &p = w.gFractal->GetLAParameters();
            p.SetDefaults(LAParameters::LADefaults::MaxPerf);
        });
    table_.emplace(
        IDM_LA_SETTINGS_3, +[](MainWindow &w) {
            auto &p = w.gFractal->GetLAParameters();
            p.SetDefaults(LAParameters::LADefaults::MinMemory);
        });

    // Tests / benchmarks
    table_.emplace(
        IDM_BASICTEST, +[](MainWindow &w) {
            CrummyTest t{*w.gFractal};
            t.TestAll();
        });
    table_.emplace(
        IDM_TEST_27, +[](MainWindow &w) {
            CrummyTest t{*w.gFractal};
            t.TestReallyHardView27();
        });
    table_.emplace(
        IDM_BENCHMARK_FULL, +[](MainWindow &w) {
            CrummyTest t{*w.gFractal};
            t.Benchmark(RefOrbitCalc::PerturbationResultType::All);
        });
    table_.emplace(
        IDM_BENCHMARK_INT, +[](MainWindow &w) {
            CrummyTest t{*w.gFractal};
            t.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
        });

    // Iterations
    table_.emplace(IDM_INCREASEITERATIONS_1P5X, +[](MainWindow &w) { w.MenuMultiplyIterations(1.5); });
    table_.emplace(IDM_INCREASEITERATIONS_6X, +[](MainWindow &w) { w.MenuMultiplyIterations(6.0); });
    table_.emplace(IDM_INCREASEITERATIONS_24X, +[](MainWindow &w) { w.MenuMultiplyIterations(24.0); });
    table_.emplace(IDM_DECREASEITERATIONS, +[](MainWindow &w) { w.MenuMultiplyIterations(2.0 / 3.0); });
    table_.emplace(IDM_RESETITERATIONS, +[](MainWindow &w) { w.MenuResetIterations(); });
    table_.emplace(
        IDM_32BIT_ITERATIONS, +[](MainWindow &w) { w.gFractal->SetIterType(IterTypeEnum::Bits32); });
    table_.emplace(
        IDM_64BIT_ITERATIONS, +[](MainWindow &w) { w.gFractal->SetIterType(IterTypeEnum::Bits64); });

    // Perturbation UI
    table_.emplace(
        IDM_PERTURB_RESULTS, +[](MainWindow &) {
            std::wcerr << L"TODO. By default these are shown as white pixels overlayed on the image. "
                          L"It'd be nice to have an option that shows them as white pixels against a "
                          L"black screen so their location is obvious.\n";
        });
    table_.emplace(
        IDM_PERTURB_CLEAR_ALL, +[](MainWindow &w) {
            w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        });
    table_.emplace(
        IDM_PERTURB_CLEAR_MED, +[](MainWindow &w) {
            w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
        });
    table_.emplace(
        IDM_PERTURB_CLEAR_HIGH, +[](MainWindow &w) {
            w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes);
        });

    table_.emplace(
        IDM_PERTURBATION_AUTO,
        +[](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto); });
    table_.emplace(
        IDM_PERTURBATION_SINGLETHREAD,
        +[](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST); });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD,
        +[](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT); });
    table_.emplace(
        IDM_PERTURBATION_SINGLETHREAD_PERIODICITY, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
        });
    table_.emplace(
        IDM_PERTURBATION_MULTITHREAD5_PERIODICITY, +[](MainWindow &w) {
            w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5);
        });
    table_.emplace(
        IDM_PERTURBATION_GPU,
        +[](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU); });

    table_.emplace(IDM_PERTURBATION_SAVE, +[](MainWindow &w) { w.gFractal->SavePerturbationOrbits(); });
    table_.emplace(IDM_PERTURBATION_LOAD, +[](MainWindow &w) { w.gFractal->LoadPerturbationOrbits(); });

    table_.emplace(
        IDM_PERTURB_AUTOSAVE_ON,
        +[](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::EnableWithSave); });
    table_.emplace(
        IDM_PERTURB_AUTOSAVE_ON_DELETE,
        +[](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::EnableWithoutSave); });
    table_.emplace(
        IDM_PERTURB_AUTOSAVE_OFF,
        +[](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::DontSave); });

    // Memory limit toggle
    table_.emplace(IDM_MEMORY_LIMIT_0, +[](MainWindow &w) { w.gJobObj = nullptr; });
    table_.emplace(
        IDM_MEMORY_LIMIT_1, +[](MainWindow &w) { w.gJobObj = std::make_unique<JobObject>(); });

    // Palettes
    table_.emplace(IDM_PALETTEROTATE, +[](MainWindow &w) { w.MenuPaletteRotation(); });
    table_.emplace(IDM_CREATENEWPALETTE, +[](MainWindow &w) { w.MenuCreateNewPalette(); });

    table_.emplace(IDM_PALETTE_TYPE_0, +[](MainWindow &w) { w.MenuPaletteType(FractalPalette::Basic); });
    table_.emplace(
        IDM_PALETTE_TYPE_1, +[](MainWindow &w) { w.MenuPaletteType(FractalPalette::Default); });
    table_.emplace(
        IDM_PALETTE_TYPE_2, +[](MainWindow &w) { w.MenuPaletteType(FractalPalette::Patriotic); });
    table_.emplace(
        IDM_PALETTE_TYPE_3, +[](MainWindow &w) { w.MenuPaletteType(FractalPalette::Summer); });
    table_.emplace(
        IDM_PALETTE_TYPE_4, +[](MainWindow &w) { w.MenuPaletteType(FractalPalette::Random); });

    table_.emplace(IDM_PALETTE_5, +[](MainWindow &w) { w.MenuPaletteDepth(5); });
    table_.emplace(IDM_PALETTE_6, +[](MainWindow &w) { w.MenuPaletteDepth(6); });
    table_.emplace(IDM_PALETTE_8, +[](MainWindow &w) { w.MenuPaletteDepth(8); });
    table_.emplace(IDM_PALETTE_12, +[](MainWindow &w) { w.MenuPaletteDepth(12); });
    table_.emplace(IDM_PALETTE_16, +[](MainWindow &w) { w.MenuPaletteDepth(16); });
    table_.emplace(IDM_PALETTE_20, +[](MainWindow &w) { w.MenuPaletteDepth(20); });

    // Location / IO
    table_.emplace(IDM_CURPOS, +[](MainWindow &w) { w.MenuGetCurPos(); });
    table_.emplace(IDM_SAVELOCATION, +[](MainWindow &w) { w.MenuSaveCurrentLocation(); });
    table_.emplace(IDM_LOADLOCATION, +[](MainWindow &w) { w.MenuLoadCurrentLocation(); });
    table_.emplace(IDM_LOAD_ENTERLOCATION, +[](MainWindow &w) { w.MenuLoadEnterLocation(); });

    table_.emplace(IDM_SAVEBMP, +[](MainWindow &w) { w.MenuSaveBMP(); });
    table_.emplace(IDM_SAVEHIRESBMP, +[](MainWindow &w) { w.MenuSaveHiResBMP(); });
    table_.emplace(IDM_SAVE_ITERS_TEXT, +[](MainWindow &w) { w.MenuSaveItersAsText(); });

    table_.emplace(
        IDM_SAVE_REFORBIT_TEXT, +[](MainWindow &w) { w.MenuSaveImag(CompressToDisk::Disable); });
    table_.emplace(
        IDM_SAVE_REFORBIT_TEXT_SIMPLE,
        +[](MainWindow &w) { w.MenuSaveImag(CompressToDisk::SimpleCompression); });
    table_.emplace(
        IDM_SAVE_REFORBIT_TEXT_MAX,
        +[](MainWindow &w) { w.MenuSaveImag(CompressToDisk::MaxCompression); });
    table_.emplace(
        IDM_SAVE_REFORBIT_IMAG_MAX,
        +[](MainWindow &w) { w.MenuSaveImag(CompressToDisk::MaxCompressionImagina); });
    table_.emplace(IDM_DIFF_REFORBIT_IMAG_MAX, +[](MainWindow &w) { w.MenuDiffImag(); });

    table_.emplace(
        IDM_LOAD_REFORBIT_IMAG_MAX,
        +[](MainWindow &w) { w.MenuLoadImagDyn(ImaginaSettings::ConvertToCurrent); });
    table_.emplace(
        IDM_LOAD_REFORBIT_IMAG_MAX_SAVED,
        +[](MainWindow &w) { w.MenuLoadImagDyn(ImaginaSettings::UseSaved); });

    // Help / exit
    table_.emplace(IDM_SHOWHOTKEYS, +[](MainWindow &w) { w.MenuShowHotkeys(); });
    table_.emplace(IDM_EXIT, +[](MainWindow &w) { ::DestroyWindow(w.hWnd); });
}

} // namespace FractalShark
