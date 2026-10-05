#include "Fractal.h"
#include "FractalPalette.h"
#include "GPU_Render.h"
#include "RenderThreadPool.h"
#include "TestFramework.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace {

void
CheckColor(const Color16 &actual, const Color16 &expected)
{
    ASSERT_EQ(actual.r, expected.r);
    ASSERT_EQ(actual.g, expected.g);
    ASSERT_EQ(actual.b, expected.b);
    ASSERT_EQ(actual.a, expected.a);
}

Color16
ExpectedSample(uint64_t count,
               uint64_t iterations,
               uint64_t maxPossible,
               uint64_t rotation,
               uint32_t auxDepth,
               const Color16 *palette,
               uint32_t paletteSize,
               FractalShark::ColoringMode mode)
{
    if (count >= iterations) {
        return {0, 0, 0, 65535};
    }
    // Saturate independently of the device implementation, including rotations that would overflow.
    const uint64_t room = maxPossible - 1 - count;
    const uint64_t shifted = (count + std::min(room, rotation)) >> auxDepth;
    if (mode == FractalShark::ColoringMode::BasicGrayscale) {
        const auto gray = static_cast<uint16_t>(shifted * std::max(uint64_t{1}, 65536 / iterations));
        return {gray, gray, gray, 65535};
    }
    const auto color = palette[shifted % paletteSize];
    return {color.r, color.g, color.b, 65535};
}

template <typename ReadCount>
void
CheckColors(const Color16 *colors,
            size_t width,
            size_t height,
            uint32_t aa,
            uint64_t iterations,
            uint64_t maxPossible,
            uint64_t rotation,
            uint32_t auxDepth,
            const Color16 *palette,
            uint32_t paletteSize,
            FractalShark::ColoringMode mode,
            ReadCount readCount)
{
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint64_t red = 0, green = 0, blue = 0;
            for (size_t iy = y * aa; iy < (y + 1) * aa; ++iy) {
                for (size_t ix = x * aa; ix < (x + 1) * aa; ++ix) {
                    const auto sample = ExpectedSample(readCount(ix, iy),
                                                       iterations,
                                                       maxPossible,
                                                       rotation,
                                                       auxDepth,
                                                       palette,
                                                       paletteSize,
                                                       mode);
                    red += sample.r;
                    green += sample.g;
                    blue += sample.b;
                }
            }
            const uint32_t samples = aa * aa;
            CheckColor(colors[y * width + x],
                       {static_cast<uint16_t>(red / samples),
                        static_cast<uint16_t>(green / samples),
                        static_cast<uint16_t>(blue / samples),
                        65535});
        }
    }
}

template <typename IterType>
void
CheckHostRecolor(uint32_t aa)
{
    constexpr uint32_t width = 19, height = 11;
    constexpr uint64_t maxPossible = Fractal::GetMaxIterations<IterType>();
    const size_t stride = width * aa + 7;
    std::vector<IterType> counts(stride * height * aa, std::numeric_limits<IterType>::max());
    const Color16 sentinel{123, 456, 789, 321};
    std::vector<Color16> colors(width * height + 1, sentinel);
    FractalPalette defaultPalette;
    defaultPalette.InitializeAllPalettes();
    defaultPalette.SetDefaults();
    std::vector<Color16> customPalette{{111, 222, 333, 0},
                                       {65535, 13, 1098, 0},
                                       {4987, 3498, 12, 0},
                                       {0, 65535, 481, 0},
                                       {338, 887, 65535, 0}};
    GPURenderer renderer;
    for (const bool custom : {false, true}) {
        const auto *palette = custom ? customPalette.data() : defaultPalette.GetCurrentPalInterleaved();
        const auto paletteSize =
            custom ? static_cast<uint32_t>(customPalette.size()) : defaultPalette.GetCurrentNumColors();
        for (const uint64_t iterations : {uint64_t{97}, maxPossible}) {
            const std::vector<uint64_t> samples{
                0, 1, 7, iterations - 2, iterations - 1, iterations, iterations + 1};
            for (size_t y = 0; y < height * aa; ++y) {
                for (size_t x = 0; x < width * aa; ++x) {
                    counts[y * stride + x] =
                        static_cast<IterType>(samples[(x + 3 * y) % samples.size()]);
                }
            }
            const auto originalCounts = counts;
            for (const uint64_t rotation : {uint64_t{0}, uint64_t{19}, maxPossible - 4, UINT64_MAX}) {
                for (const uint32_t auxDepth : {0u, 3u}) {
                    ASSERT_EQ(renderer.InitializeMemory<IterType>(width * aa,
                                                                  height * aa,
                                                                  aa,
                                                                  palette,
                                                                  paletteSize,
                                                                  auxDepth,
                                                                  rotation,
                                                                  maxPossible,
                                                                  1,
                                                                  true),
                              0u);
                    for (const auto mode : {FractalShark::ColoringMode::PaletteLookup,
                                            FractalShark::ColoringMode::BasicGrayscale}) {
                        ASSERT_EQ(renderer.RecolorFromHostIterations(counts.data(),
                                                                     stride,
                                                                     static_cast<IterType>(iterations),
                                                                     mode,
                                                                     colors.data(),
                                                                     width * height),
                                  0u);
                        CheckColors(colors.data(),
                                    width,
                                    height,
                                    aa,
                                    iterations,
                                    maxPossible,
                                    rotation,
                                    auxDepth,
                                    palette,
                                    paletteSize,
                                    mode,
                                    [&](size_t x, size_t y) { return counts[y * stride + x]; });
                        CheckColor(colors.back(), sentinel);
                        ASSERT_TRUE(counts == originalCounts);
                        if (mode == FractalShark::ColoringMode::PaletteLookup) {
                            // Legacy frame extraction downloads the entire block-rounded allocation.
                            const size_t frameCapacity = ((width + GPURenderer::NB_THREADS_W_AA - 1) /
                                                          GPURenderer::NB_THREADS_W_AA) *
                                                         GPURenderer::NB_THREADS_W_AA *
                                                         ((height + GPURenderer::NB_THREADS_H_AA - 1) /
                                                          GPURenderer::NB_THREADS_H_AA) *
                                                         GPURenderer::NB_THREADS_H_AA;
                            std::vector<Color16> frameColors(frameCapacity);
                            ReductionResults reduction;
                            ASSERT_EQ(renderer.RenderCurrent<IterType>(static_cast<IterType>(iterations),
                                                                       nullptr,
                                                                       frameColors.data(),
                                                                       &reduction,
                                                                       false),
                                      0u);
                            ASSERT_EQ(renderer.SyncComputeStream(), 0u);
                            for (size_t i = 0; i < width * height; ++i) {
                                CheckColor(frameColors[i], colors[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    const auto mode = FractalShark::ColoringMode::PaletteLookup;
    ASSERT_NE(renderer.RecolorFromHostIterations(
                  counts.data(), stride, IterType{97}, mode, colors.data(), width * height - 1),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations(
                  counts.data(), width * aa - 1, IterType{97}, mode, colors.data(), colors.size()),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations<IterType>(
                  nullptr, stride, IterType{97}, mode, colors.data(), colors.size()),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations(
                  counts.data(), stride, IterType{0}, mode, colors.data(), colors.size()),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations(counts.data(),
                                                 std::numeric_limits<size_t>::max(),
                                                 IterType{97},
                                                 mode,
                                                 colors.data(),
                                                 colors.size()),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations(
                  counts.data(), stride, IterType{97}, mode, nullptr, colors.size()),
              0u);
    ASSERT_NE(renderer.RecolorFromHostIterations(counts.data(),
                                                 stride,
                                                 IterType{97},
                                                 static_cast<FractalShark::ColoringMode>(99),
                                                 colors.data(),
                                                 colors.size()),
              0u);
    // Invalid requests leave the renderer usable and do not retain host buffers in queued work.
    ASSERT_EQ(renderer.RecolorFromHostIterations(
                  counts.data(), stride, IterType{97}, mode, colors.data(), colors.size()),
              0u);
    CheckColor(colors.back(), sentinel);

    // The maximum and generation must also refresh when dimensions and palette address stay fixed.
    for (uint64_t generation = 1; generation <= 2; ++generation) {
        if (generation == 2) {
            customPalette[0] = {3999, 5999, 7999, 0};
        }
        ASSERT_EQ(renderer.InitializeMemory<IterType>(width * aa,
                                                      height * aa,
                                                      aa,
                                                      customPalette.data(),
                                                      static_cast<uint32_t>(customPalette.size()),
                                                      0,
                                                      19,
                                                      23,
                                                      generation,
                                                      true),
                  0u);
        ASSERT_EQ(renderer.RecolorFromHostIterations(
                      counts.data(), stride, IterType{97}, mode, colors.data(), colors.size()),
                  0u);
        CheckColors(colors.data(),
                    width,
                    height,
                    aa,
                    97,
                    23,
                    19,
                    0,
                    customPalette.data(),
                    static_cast<uint32_t>(customPalette.size()),
                    mode,
                    [&](size_t x, size_t y) { return counts[y * stride + x]; });
        CheckColor(colors.back(), sentinel);
    }
}

std::vector<uint64_t>
CopyCounts(const Fractal &fractal)
{
    const auto &iters = fractal.GetCurIters();
    std::vector<uint64_t> counts;
    for (size_t y = 0; y < iters.m_Height; ++y) {
        for (size_t x = 0; x < iters.m_Width; ++x) {
            counts.push_back(iters.GetItersArrayValSlow(x, y));
        }
    }
    return counts;
}

void
CheckCurrentColors(Fractal &fractal)
{
    const auto &iters = fractal.GetCurIters();
    const auto &palette = fractal.GetPalette();
    const auto mode = palette.GetPaletteType() == FractalPaletteType::Basic
                          ? FractalShark::ColoringMode::BasicGrayscale
                          : FractalShark::ColoringMode::PaletteLookup;
    CheckColors(iters.m_RoundedOutputColorMemory.get(),
                iters.m_OutputWidth,
                iters.m_OutputHeight,
                static_cast<uint32_t>(iters.m_Antialiasing),
                fractal.GetNumIterations<uint64_t>(),
                fractal.GetMaxIterationsRT(),
                palette.GetPaletteRotation(),
                palette.GetAuxDepth(),
                palette.GetCurrentPalInterleaved(),
                palette.GetCurrentNumColors(),
                mode,
                [&](size_t x, size_t y) { return iters.GetItersArrayValSlow(x, y); });
}

void
CheckQueuedRecolor(GpuMode gpuMode, RenderAlgorithmEnum algorithm)
{
    Fractal fractal{19, 13, nullptr, false, UINT64_MAX, true, gpuMode};
    auto *pool = fractal.GetRenderPool();
    pool->Drain();
    fractal.View(0, false);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(algorithm)));
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (uint32_t aa = 1; aa <= 4; ++aa) {
            pool->Drain();
            fractal.SetIterType(bits);
            fractal.ResetDimensions(19 + aa, 13 + aa, aa);
            fractal.SetNumIterations<uint64_t>(97);
            fractal.UsePaletteType(FractalPaletteType::Default);
            fractal.ResetFractalPalette();
            fractal.CalcFractal(true);
            const auto originalCounts = CopyCounts(fractal);
            for (const auto type : {FractalPaletteType::Default, FractalPaletteType::Basic}) {
                fractal
                    .EnqueuePaletteRecolor(
                        "Recolor fixture palette",
                        [=](Fractal &target) {
                            target.UsePaletteType(type);
                            target.SetPaletteAuxDepth(0);
                            target.ResetFractalPalette();
                        },
                        false,
                        RenderPresentationMode::Immediate,
                        0)
                    .Wait();
                pool->Drain();
                CheckCurrentColors(fractal);
                for (int pass = 0; pass < 3; ++pass) {
                    fractal
                        .EnqueuePaletteRecolor(
                            "Recolor fixture rotation",
                            [=](Fractal &target) {
                                target.RotateFractalPalette(19);
                                if (pass == 2) {
                                    target.SetPaletteAuxDepth(2);
                                }
                            },
                            false,
                            RenderPresentationMode::Immediate,
                            0)
                        .Wait();
                    pool->Drain();
                    CheckCurrentColors(fractal);
                    ASSERT_TRUE(CopyCounts(fractal) == originalCounts);
                }
                fractal
                    .EnqueuePaletteRecolor(
                        "Recolor fixture reset",
                        [](Fractal &target) { target.ResetFractalPalette(); },
                        false,
                        RenderPresentationMode::Immediate,
                        0)
                    .Wait();
                pool->Drain();
                CheckCurrentColors(fractal);
            }
        }
    }
    const auto group = fractal.BeginPacedAnimation();
    fractal
        .EnqueuePaletteRecolor(
            "Recolor fixture paced",
            [](Fractal &target) { target.RotateFractalPalette(10); },
            false,
            RenderPresentationMode::PacedAnimation,
            group)
        .Wait();
    std::vector<RenderJobHandle> jobs;
    for (int i = 0; i < 8; ++i) {
        jobs.push_back(fractal.EnqueuePaletteRecolor(
            "Recolor fixture cancel",
            [](Fractal &target) { target.RotateFractalPalette(10); },
            false,
            RenderPresentationMode::PacedAnimation,
            group));
    }
    fractal.CancelPacedAnimation(group);
    for (auto &job : jobs) {
        job.Wait();
    }
    pool->Drain();
    const auto originalCounts = CopyCounts(fractal);
    const auto *colors = fractal.GetCurIters().m_RoundedOutputColorMemory.get();
    const size_t colorCount = fractal.GetRenderWidth() * fractal.GetRenderHeight();
    const std::vector<Color16> originalColors(colors, colors + colorCount);
    fractal
        .EnqueuePaletteRecolor(
            "Recolor fixture disabled repaint",
            [](Fractal &target) {
                target.SetRepaint(false);
                target.RotateFractalPalette(19);
            },
            false,
            RenderPresentationMode::Immediate,
            0)
        .Wait();
    pool->Drain();
    for (size_t i = 0; i < colorCount; ++i) {
        CheckColor(colors[i], originalColors[i]);
    }
    ASSERT_TRUE(CopyCounts(fractal) == originalCounts);
    fractal
        .EnqueueCommand(
            "Recolor fixture later render",
            [](Fractal &target) {
                target.SetRepaint(true);
                target.UsePaletteType(FractalPaletteType::Default);
                target.ResetFractalPalette();
                target.ResetDimensions(21, 15, 2);
            },
            false,
            RenderPresentationMode::Immediate,
            0,
            true)
        .Wait();
    pool->Drain();
    fractal
        .EnqueuePaletteRecolor(
            "Recolor fixture after resize",
            [](Fractal &) {},
            false,
            RenderPresentationMode::Immediate,
            0)
        .Wait();
    pool->Drain();
    CheckCurrentColors(fractal);
    if (gpuMode == GpuMode::Auto) {
        ASSERT_FALSE(fractal.GpuBypassed());
    }
}

const bool registered = [] {
    for (const uint32_t aa : {1u, 2u, 3u, 4u}) {
        for (const bool bits64 : {false, true}) {
            const auto name =
                "CudaRecolor_Host_AA" + std::to_string(aa) + (bits64 ? "_Bits64" : "_Bits32");
            TestFramework::RegisterCase(
                name,
                [=] {
                    if (bits64) {
                        CheckHostRecolor<uint64_t>(aa);
                    } else {
                        CheckHostRecolor<uint32_t>(aa);
                    }
                },
                true,
                "",
                false);
        }
    }
    TestFramework::RegisterCase(
        "CudaRecolor_QueuedLifecycle",
        [] { CheckQueuedRecolor(GpuMode::Auto, RenderAlgorithmEnum::Gpu1x32); },
        true,
        "",
        false);
    return true;
}();

} // namespace

TEST(FractalSharkLib_RecolorCpuWithoutCuda)
{
    CheckQueuedRecolor(GpuMode::Disabled, RenderAlgorithmEnum::Cpu64);
}
