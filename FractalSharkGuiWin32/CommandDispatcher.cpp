// CommandDispatcher.cpp
#include "StdAfx.h"
#include "CommandDispatcher.h"

#include "CrummyTest.h"
#include "DynamicPopupMenu.h"
#include "Fractal.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "RecommendedSettings.h"
#include "resource.h"

#include <array>
#include <utility>

namespace FractalShark::Win32 {

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

    auto alg = GetRenderAlgorithmTupleEntry(e->alg);
    w_.gFractal->EnqueueMutation(
        [alg](Fractal &f) { [[maybe_unused]] const bool success = f.SetRenderAlgorithm(alg); });

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
    table_.reserve(8);

    // Phase 0c: Imagina file-dialog entries are the only IDMs left in the
    // legacy table.  They have no FractalCommand mapping (the dialog launch
    // is GUI-shell-specific) so they remain dispatched here.  Everything else
    // routes through FractalShark::ExecuteCommand → ExecuteCommandHost on
    // MainWindow.
    table_.emplace(
        IDM_LOAD_IMAGINA_DLG, +[](MainWindow &w) {
            w.MenuLoadImag(ImaginaSettings::ConvertToCurrent, CompressToDisk::MaxCompressionImagina);
        });
    table_.emplace(
        IDM_LOAD_IMAGINA_DLG_SAVED, +[](MainWindow &w) {
            w.MenuLoadImag(ImaginaSettings::UseSaved, CompressToDisk::MaxCompressionImagina);
        });
}

} // namespace FractalShark::Win32
