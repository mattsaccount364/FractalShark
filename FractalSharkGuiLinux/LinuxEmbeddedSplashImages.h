#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace FractalShark::Linux {

struct EmbeddedSplashImage {
    std::string_view Name;
    std::span<const std::byte> Bytes;
};

std::span<const EmbeddedSplashImage> GetEmbeddedSplashImages();

} // namespace FractalShark::Linux
