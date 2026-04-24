// CommandDispatcher.h
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <unordered_map>

#include "AlgCmds.h"
#include "RenderAlgorithm.h"

// Forward declare to avoid dragging all of MainWindow.h into every TU that uses this.
class MainWindow;

namespace FractalShark {
struct AlgCmd;

const FractalShark::AlgCmd *FindAlgForCmd(int wmId) noexcept;
int FindCmdForAlg(RenderAlgorithmEnum alg) noexcept;

class CommandDispatcher final {
public:
    explicit CommandDispatcher(MainWindow &owner);

    // Returns true if wmId was handled.
    bool Dispatch(int wmId);

private:
    using Fn = void (*)(MainWindow &);

    // --- dispatch tiers ---
    bool HandleCommandRange(int wmId);
    bool HandleAlgCommand(int wmId);
    bool HandleCommandTable(int wmId);

    void BuildTable();

private:
    MainWindow &w_;
    std::unordered_map<int, Fn> table_;
};
} // namespace FractalShark
