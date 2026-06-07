// LinuxMenuState.cpp — IMenuState impl for the Linux GUI.
//
// Mirrors FractalSharkGUILib/MainWindowMenuState.cpp but uses
// std::filesystem instead of _stat and reads window/JobObject state via
// constructor refs instead of the Win32-specific MainWindow members.  No
// Win32 lib changes.

#include "LinuxMenuState.h"

#include "AlgCmds.h"
#include "Fractal.h"
#include "LAParameters.h"
#include "RefOrbitCalc.h"

#include <filesystem>

namespace FractalSharkLinux {

namespace {

int
findCmdForAlg(::RenderAlgorithmEnum alg) noexcept
{
    for (const auto &e : FractalShark::kAlgCmds) {
        if (e.alg == alg) {
            return e.id;
        }
    }
    return -1;
}

} // namespace

LinuxMenuState::LinuxMenuState(const Fractal &f, const bool &fullscreen) noexcept
    : m_Fractal(f), m_Fullscreen(fullscreen)
{
}

bool
LinuxMenuState::IsEnabled(FractalShark::Menu::Rule rule) const noexcept
{
    using Rule = FractalShark::Menu::Rule;

    switch (rule) {
        case Rule::Always:
            return true;

        case Rule::EnableIfGpuActive:
            return !m_Fractal.GpuBypassed();

        case Rule::EnableIfCpuActive:
            return true;

        case Rule::EnableIfPerturbationAvailable:
            return false;

        case Rule::EnableIfNRCheckpointExists: {
            std::error_code ec;
            return std::filesystem::exists("nr_checkpoint.txt", ec);
        }

        default:
            return true;
    }
}

bool
LinuxMenuState::IsChecked(uint32_t commandId) const noexcept
{
    switch (commandId) {
        case IDM_WINDOWED:
        case IDM_WINDOWED_SQ:
            return m_Fullscreen;

        case IDM_REPAINTING:
        default:
            return false;
    }
}

uint32_t
LinuxMenuState::GetRadioSelection(FractalShark::Menu::RadioGroup group) const noexcept
{
    using RG = FractalShark::Menu::RadioGroup;

    switch (group) {
        case RG::RenderAlgorithm: {
            const auto ra = m_Fractal.GetRenderAlgorithm();
            const int cmd = findCmdForAlg(ra.Algorithm);
            return (cmd >= 0) ? static_cast<uint32_t>(cmd) : 0;
        }

        case RG::GpuAntialiasing:
            switch (m_Fractal.GetGpuAntialiasing()) {
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
            switch (m_Fractal.GetIterationPrecision()) {
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
            return m_Fractal.GetLAParameters().GetThreading() ==
                           LAParameters::LAThreadingAlgorithm::MultiThreaded
                       ? IDM_LA_MULTITHREADED
                       : IDM_LA_SINGLETHREADED;

        case RG::PerturbationMode:
            switch (m_Fractal.GetPerturbationAlg()) {
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
            switch (m_Fractal.GetPaletteType()) {
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
            switch (m_Fractal.GetPaletteDepth()) {
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
            switch (m_Fractal.GetResultsAutosave()) {
                case AddPointOptions::EnableWithSave:
                    return IDM_PERTURB_AUTOSAVE_ON;
                case AddPointOptions::EnableWithoutSave:
                    return IDM_PERTURB_AUTOSAVE_ON_DELETE;
                case AddPointOptions::DontSave:
                    return IDM_PERTURB_AUTOSAVE_OFF;
            }
            return IDM_PERTURB_AUTOSAVE_ON_DELETE;

        case RG::MemoryLimit:
            // No Linux JobObject equivalent yet — always report "no limit".
            return IDM_MEMORY_LIMIT_0;

        case RG::IterationsWidth:
            return (m_Fractal.GetIterType() == IterTypeEnum::Bits32) ? IDM_32BIT_ITERATIONS
                                                                     : IDM_64BIT_ITERATIONS;

        case RG::NRInnerLoopBackend:
            switch (m_Fractal.GetNRInnerLoopBackend()) {
                case NRInnerLoopBackend::GPU:
                    return IDM_NR_INNERLOOP_GPU;
                case NRInnerLoopBackend::CpuMT:
                    return IDM_NR_INNERLOOP_CPU;
                case NRInnerLoopBackend::CpuST:
                    return IDM_NR_INNERLOOP_CPUST;
                default:
                    return IDM_NR_INNERLOOP_GPU;
            }

        case RG::None:
        default:
            return 0;
    }
}

} // namespace FractalSharkLinux
