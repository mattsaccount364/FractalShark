#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace Png {
// Checked byte arithmetic is also used by tests of large images without allocating
// them. PNG dimensions are limited to 2^31-1; RGB16 uses six bytes per pixel.
inline bool
Rgb16ByteCount(size_t width, size_t height, size_t &bytes)
{
    constexpr size_t maximumDimension = 0x7fffffff;
    constexpr size_t pixelBytes = 3 * sizeof(uint16_t);
    if (width == 0 || height == 0 || width > maximumDimension || height > maximumDimension ||
        width > std::numeric_limits<size_t>::max() / pixelBytes ||
        height > std::numeric_limits<size_t>::max() / (width * pixelBytes)) {
        return false;
    }
    bytes = width * height * pixelBytes;
    return true;
}
} // namespace Png
