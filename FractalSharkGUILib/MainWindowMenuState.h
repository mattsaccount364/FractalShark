#pragma once

#include "AlgCmds.h"
#include "DynamicPopupMenu.h"

// Forward declarations only
class Fractal;

// Concrete IMenuState bound to a live MainWindow.
// This is queried ONLY while the popup menu is being built.
namespace FractalShark::Win32 {

class MainWindow;

class MainWindowMenuState final : public IMenuState {
public:
    explicit MainWindowMenuState(const MainWindow &w) noexcept;

    // IMenuState
    bool IsEnabled(Rule rule) const noexcept override;
    bool IsChecked(UINT commandId) const noexcept override;
    UINT GetRadioSelection(RadioGroup group) const noexcept override;

    // Popup adornments
    UINT GetPopupAdornmentCommandId(RadioGroup group) const noexcept override;

    std::wstring_view GetCommandLabel(UINT commandId) const noexcept override;

private:
    const MainWindow &w_;
    const Fractal &f_;
};

} // namespace FractalShark::Win32
