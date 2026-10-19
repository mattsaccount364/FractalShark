#pragma once

#include "MovieJob.h"
#include "RenderToPng.h"

#include <functional>

namespace FractalSharkCli {
struct MovieRenderSettings {
    RenderRequest Request;
    std::filesystem::path Palette;
    std::optional<uint32_t> PaletteDepth;
};

// The CLI supplies its existing argument parser/request builder. Keeping that
// boundary explicit avoids a second list of render/reference algorithm names.
int RunMovie(const std::filesystem::path &directory,
             const std::filesystem::path &configuration,
             const std::filesystem::path &ffmpeg,
             GpuMode gpuMode,
             uint64_t commitCapBytes,
             const std::function<MovieRenderSettings(const MovieConfig &)> &buildSettings);
} // namespace FractalSharkCli
