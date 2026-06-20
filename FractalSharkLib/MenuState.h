#pragma once

#include "MenuTree.h"

class Fractal;

namespace FractalShark {

class MenuState final : public IMenuState {
public:
    explicit MenuState(const Fractal &fractal) noexcept;

    bool IsEnabled(Rule rule) const noexcept override;
    bool IsChecked(uint32_t commandId) const noexcept override;
    uint32_t GetRadioSelection(RadioGroup group) const override;

private:
    const Fractal &m_Fractal;
};

} // namespace FractalShark
