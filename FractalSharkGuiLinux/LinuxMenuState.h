// LinuxMenuState.h — IMenuState implementation for the Linux GUI.
//
// Mirrors FractalSharkGUILib/MainWindowMenuState.{h,cpp} but lives entirely
// on the Linux side.  Win32 lib is not modified.
#pragma once

#include "MenuTree.h"

class Fractal;

namespace FractalSharkLinux {

class LinuxMenuState final : public FractalShark::Menu::IMenuState {
public:
    explicit LinuxMenuState(const Fractal &f, const bool &fullscreen) noexcept;

    bool IsEnabled(FractalShark::Menu::Rule rule) const noexcept override;
    bool IsChecked(uint32_t commandId) const noexcept override;
    uint32_t GetRadioSelection(FractalShark::Menu::RadioGroup group) const override;

private:
    const Fractal &m_Fractal;
    const bool &m_Fullscreen;
};

} // namespace FractalSharkLinux
