#include "StdAfx.h"

#include "MainWindowMenuState.h"

#include "Fractal.h"
#include "MainWindow.h"
#include "resource.h"

using namespace FractalShark;

MainWindowMenuState::MainWindowMenuState(const MainWindow &w) noexcept : w_(w), f_(*w.gFractal) {}

// ------------------------------------------------------------
// Enable / disable rules
// ------------------------------------------------------------

bool
MainWindowMenuState::IsEnabled(DynamicPopupMenu::Rule rule) const noexcept
{
    using Rule = DynamicPopupMenu::Rule;

    switch (rule) {
        case Rule::Always:
            return true;

        case Rule::EnableIfGpuActive:
            // Conservative: only enable GPU menus if GPU path exists & is usable
            return !f_.GpuBypassed();

        case Rule::EnableIfCpuActive:
            return true; // CPU always exists

        // The next two are really just buggy menu items that could probably be removed.
        case Rule::EnableIfPerturbationAvailable:
            return false;

        case Rule::EnableIfPaletteRotationSupported:
            return false;

        default:
            // Never brick the UI due to an unknown rule
            return true;
    }
}

// ------------------------------------------------------------
// Independent toggles
// ------------------------------------------------------------

bool
MainWindowMenuState::IsChecked(UINT commandId) const noexcept
{
    switch (commandId) {
        case IDM_REPAINTING:
            //return w_.IsRepaintingEnabled();
            return false;

        case IDM_WINDOWED:
            //return w_.IsWindowed();
            return false;

        case IDM_WINDOWED_SQ:
            //return w_.IsWindowedSquare();
            return false;

        default:
            return false;
    }
}

// ------------------------------------------------------------
// Radio groups
// ------------------------------------------------------------

UINT
MainWindowMenuState::GetRadioSelection(DynamicPopupMenu::RadioGroup group) const noexcept
{
    using RG = DynamicPopupMenu::RadioGroup;

    switch (group) {
        case RG::RenderAlgorithm: {
            if (!w_.gFractal)
                return 0;

            const auto ra = w_.gFractal->GetRenderAlgorithm();

            // Convert authoritative algorithm enum → command ID
            const int cmd = FindCmdForAlg(ra.Algorithm);
            return (cmd >= 0) ? static_cast<UINT>(cmd) : 0;
        }

        case RG::GpuAntialiasing:
            switch (f_.GetGpuAntialiasing()) {
                case 1:
                    return IDM_GPUANTIALIASING_1X;
                case 2:
                    return IDM_GPUANTIALIASING_4X;
                case 3:
                    return IDM_GPUANTIALIASING_9X;
                case 4:
                    return IDM_GPUANTIALIASING_16X;
                default:
                    return IDM_GPUANTIALIASING_1X;
            }

        case RG::IterationPrecision:
            switch (f_.GetIterationPrecision()) {
                case 1:
                    return IDM_ITERATIONPRECISION_1X;
                case 4:
                    return IDM_ITERATIONPRECISION_2X;
                case 8:
                    return IDM_ITERATIONPRECISION_3X;
                case 16:
                    return IDM_ITERATIONPRECISION_4X;
                default:
                    return IDM_ITERATIONPRECISION_1X;
            }

        case RG::LaThreading:
            return f_.GetLAParameters().GetThreading() == LAParameters::LAThreadingAlgorithm::MultiThreaded
                       ? IDM_LA_MULTITHREADED
                       : IDM_LA_SINGLETHREADED;

        case RG::PerturbationMode:
            switch (f_.GetPerturbationAlg()) {
                case RefOrbitCalc::PerturbationAlg::Auto:
                    return IDM_PERTURBATION_AUTO;
                case RefOrbitCalc::PerturbationAlg::ST:
                    return IDM_PERTURBATION_SINGLETHREAD;
                case RefOrbitCalc::PerturbationAlg::MT:
                    return IDM_PERTURBATION_MULTITHREAD;
                case RefOrbitCalc::PerturbationAlg::STPeriodicity:
                    return IDM_PERTURBATION_SINGLETHREAD_PERIODICITY;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4:
                    return IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4;
                case RefOrbitCalc::PerturbationAlg::MTPeriodicity5:
                    return IDM_PERTURBATION_MULTITHREAD5_PERIODICITY;
                case RefOrbitCalc::PerturbationAlg::GPU:
                    return IDM_PERTURBATION_GPU;
                default:
                    return IDM_PERTURBATION_AUTO;
            }

        case RG::PaletteType:
            switch (f_.GetPaletteType()) {
                case FractalPaletteType::Basic:
                    return IDM_PALETTE_TYPE_0;
                case FractalPaletteType::Default:
                    return IDM_PALETTE_TYPE_1;
                case FractalPaletteType::Patriotic:
                    return IDM_PALETTE_TYPE_2;
                case FractalPaletteType::Summer:
                    return IDM_PALETTE_TYPE_3;
                case FractalPaletteType::Random:
                    return IDM_PALETTE_TYPE_4;
            }
            return IDM_PALETTE_TYPE_1;

        case RG::PaletteBitDepth:
            switch (f_.GetPaletteDepth()) {
                case 5:
                    return IDM_PALETTE_5;
                case 6:
                    return IDM_PALETTE_6;
                case 8:
                    return IDM_PALETTE_8;
                case 12:
                    return IDM_PALETTE_12;
                case 16:
                    return IDM_PALETTE_16;
                case 20:
                    return IDM_PALETTE_20;
                default:
                    return IDM_PALETTE_8;
            }

        case RG::MemoryAutosave:
            switch (f_.GetResultsAutosave()) {
                case AddPointOptions::EnableWithSave:
                    return IDM_PERTURB_AUTOSAVE_ON;
                case AddPointOptions::EnableWithoutSave:
                    return IDM_PERTURB_AUTOSAVE_ON_DELETE;
                case AddPointOptions::DontSave:
                    return IDM_PERTURB_AUTOSAVE_OFF;
            }
            return IDM_PERTURB_AUTOSAVE_ON_DELETE;

        case RG::MemoryLimit:
            return w_.gJobObj ? IDM_MEMORY_LIMIT_1 : IDM_MEMORY_LIMIT_0;

        case RG::IterationsWidth:
            return (f_.GetIterType() == IterTypeEnum::Bits32) ? IDM_32BIT_ITERATIONS
                                                              : IDM_64BIT_ITERATIONS;

        case RG::None:
        default:
            return 0;
    }
}

// ------------------------------------------------------------
// Popup adornments
// ------------------------------------------------------------

UINT
MainWindowMenuState::GetPopupAdornmentCommandId(DynamicPopupMenu::RadioGroup group) const noexcept
{
    return GetRadioSelection(group);
}

std::wstring_view
MainWindowMenuState::GetCommandLabel(UINT /*commandId*/) const noexcept
{
    // Optional — you can fill this in later if you want text adornments.
    return {};
}
