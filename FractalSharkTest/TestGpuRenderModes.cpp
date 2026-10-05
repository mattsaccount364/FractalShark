#include "Fractal.h"
#include "FractalPalette.h"
#include "GPU_Render.h"
#include "RenderTestSupport.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>

namespace {
template <typename IterType>
void
CheckGpuColors(uint32_t antialiasing, const std::string &caseId)
{
    RenderTests::ScopedDirectory directory{caseId};
    constexpr uint32_t width = 64, height = 48;
    constexpr IterType iterations = 1000;
    const auto sampleWidth = width * antialiasing;
    const auto sampleHeight = height * antialiasing;
    FractalPalette palette;
    palette.InitializeAllPalettes();
    palette.SetDefaults();
    GPURenderer renderer;
    ASSERT_EQ(renderer.InitializeMemory<IterType>(sampleWidth,
                                                  sampleHeight,
                                                  antialiasing,
                                                  palette.GetCurrentPalInterleaved(),
                                                  palette.GetCurrentNumColors(),
                                                  palette.GetAuxDepth(),
                                                  palette.GetPaletteRotation(),
                                                  Fractal::GetMaxIterations<IterType>(),
                                                  palette.GetPaletteGeneration(),
                                                  false),
              0u);
    std::vector<IterType> iterBuffer(static_cast<size_t>(sampleWidth) * sampleHeight);
    std::vector<Color16> colors(static_cast<size_t>(width) * height);
    for (int pass = 0; pass < 2; ++pass) {
        const auto algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32);
        ASSERT_EQ((renderer.Render<IterType, float>(
                      algorithm, -2.0f, -1.5f, 3.0f / sampleWidth, 3.0f / sampleHeight, iterations, 1)),
                  0u);
        ASSERT_EQ(renderer.SyncComputeStream(), 0u);
        ReductionResults reduction;
        ASSERT_EQ(renderer.RenderCurrent<IterType>(
                      iterations, iterBuffer.data(), colors.data(), &reduction, false),
                  0u);
        ASSERT_EQ(renderer.SyncComputeStream(), 0u);
        ASSERT_EQ(reduction.Min,
                  static_cast<uint64_t>(*std::min_element(iterBuffer.begin(), iterBuffer.end())));
        ASSERT_EQ(reduction.Max,
                  static_cast<uint64_t>(*std::max_element(iterBuffer.begin(), iterBuffer.end())));
        ASSERT_EQ(reduction.Sum, std::accumulate(iterBuffer.begin(), iterBuffer.end(), uint64_t{0}));
        std::vector<unsigned char> pixels;
        pixels.reserve(colors.size() * 8);
        for (const auto &color : colors) {
            for (const uint16_t channel : {color.r, color.g, color.b, color.a}) {
                pixels.push_back(static_cast<unsigned char>(channel >> 8));
                pixels.push_back(static_cast<unsigned char>(channel));
            }
        }
        const std::string filename = "gpu-colors-" + std::to_string(pass) + ".png";
        ASSERT_EQ(lodepng::encode(filename, pixels, width, height, LCT_RGBA, 16), 0u);
        RenderTests::CheckPng(caseId + "_Pass" + std::to_string(pass), filename, width, height);
    }
}

const bool registered = [] {
    for (const uint32_t antialiasing : {1u, 2u, 3u, 4u}) {
        for (const bool bits64 : {false, true}) {
            const auto name = "RenderGolden_GpuColors_AA" + std::to_string(antialiasing) +
                              (bits64 ? "_Bits64" : "_Bits32");
            TestFramework::RegisterCase(
                name,
                [=] {
                    if (bits64) {
                        CheckGpuColors<uint64_t>(antialiasing, name);
                    } else {
                        CheckGpuColors<uint32_t>(antialiasing, name);
                    }
                },
                true,
                "",
                true);
        }
    }
    return true;
}();
} // namespace
