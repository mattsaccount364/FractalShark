#pragma once

#include "DynamicPopupMenu.h"
#include "AlgCmds.h"

// Forward declarations only
class MainWindow;
class Fractal;

// Concrete IMenuState bound to a live MainWindow.
// This is queried ONLY while the popup menu is being built.
class MainWindowMenuState final : public FractalShark::DynamicPopupMenu::IMenuState {
public:
    explicit MainWindowMenuState(const MainWindow &w) noexcept;

    // IMenuState
    bool IsEnabled(FractalShark::DynamicPopupMenu::Rule rule) const noexcept override;
    bool IsChecked(UINT commandId) const noexcept override;
    UINT GetRadioSelection(FractalShark::DynamicPopupMenu::RadioGroup group) const noexcept override;

    // Popup adornments
    UINT GetPopupAdornmentCommandId(
        FractalShark::DynamicPopupMenu::RadioGroup group) const noexcept override;

    std::wstring_view GetCommandLabel(UINT commandId) const noexcept override;

private:
    const MainWindow &w_;
    const Fractal &f_;
};
