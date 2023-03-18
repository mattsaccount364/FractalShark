#include "stdafx.h"
#include <cassert>

#include "heat_lut.h"

rgba8_t heat_lut(float x)
{
    assert(0 <= x && x <= 1);
    float x0 = 1.f / 4.f;
    float x1 = 2.f / 4.f;
    float x2 = 3.f / 4.f;

    if (x < x0)
    {
        auto g = static_cast<std::uint8_t>(x / x0 * 255);
        return rgba8_t{ 0, g, 255, 255 };
    }
    else if (x < x1)
    {
        auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
        return rgba8_t{ 0, 255, b, 255 };
    }
    else if (x < x2)
    {
        auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
        return rgba8_t{ r, 255, 0, 255 };
    }
    else
    {
        auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
        return rgba8_t{ 255, b, 0, 255 };
    }
}