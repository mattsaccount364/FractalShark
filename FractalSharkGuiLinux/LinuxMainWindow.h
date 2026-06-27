#pragma once

#include <functional>

namespace FractalShark::Linux {

int RunMainWindow(const std::function<void()> &onReady);

} // namespace FractalShark::Linux
