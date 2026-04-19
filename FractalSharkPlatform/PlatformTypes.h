#pragma once

#include <cstdint>

// Portable rectangle — replaces Win32 RECT in library interfaces.
struct ScreenRect {
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
};

// Portable 2D point — replaces Win32 POINT in library interfaces.
struct ScreenPoint {
    int32_t x;
    int32_t y;
};
