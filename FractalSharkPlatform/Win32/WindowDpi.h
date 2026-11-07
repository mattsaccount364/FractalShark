#pragma once

#include "PlatformTypes.h"

#include <cstdint>

namespace Environment {

// Windows-only UI helpers. Native handles remain opaque so Windows headers and DPI constants stay
// inside the platform implementation. The GUI executable must declare PerMonitorV2 in its manifest
// before any window, including the splash window on its own thread, is created.

// Return the window's current monitor DPI, or 0 if the native window is invalid.
uint32_t GetWindowDpi(void *nativeWindow) noexcept;

// Convert a UI size measured at 96 DPI to physical client pixels, rounding to the nearest pixel.
// An invalid window leaves the logical size unchanged. Render sizes and mouse coordinates already
// use physical pixels in PerMonitorV2 mode and must never be passed through this conversion.
int32_t ScaleWindowUiPixels(void *nativeWindow, int32_t logicalPixels) noexcept;

// Apply WM_DPICHANGED's suggested physical screen bounds. Fullscreen windows instead fill the
// monitor containing the suggested bounds. A size change generates WM_SIZE through SetWindowPos;
// the GUI remains responsible for queuing the corresponding render resize. Returns false on error.
bool ApplyWindowDpiChange(void *nativeWindow,
                          const ScreenRect &suggestedBounds,
                          bool fullscreen) noexcept;

} // namespace Environment
