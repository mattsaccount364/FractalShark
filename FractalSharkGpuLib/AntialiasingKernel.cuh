#pragma once

#include "GPU_Types.h"
#include <cuda_runtime.h>

// Rendering uses padded iteration rows; saving uses packed device rows. The explicit
// stride lets both paths share identical coloring without renderer-owned state.
template <typename IterType, uint32_t Antialiasing>
__global__ void
ColorIterations(const IterType *__restrict__ outputIterMatrix,
                size_t iterationStride,
                Color16 *__restrict__ outputColors,
                const Color16 *__restrict__ palette,
                uint32_t paletteSize,
                uint32_t paletteAuxDepth,
                uint64_t paletteRotation,
                uint64_t maxPossibleIterations,
                uint32_t colorWidth,
                uint32_t colorHeight,
                IterType numIterations,
                FractalShark::ColoringMode coloringMode)
{
    const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX >= colorWidth || outputY >= colorHeight) {
        return;
    }

    const size_t colorIndex = static_cast<size_t>(colorWidth) * outputY + outputX;
    constexpr auto totalAA = Antialiasing * Antialiasing;
    const uint64_t maximum = maxPossibleIterations - 1;
    const uint64_t basicFactor = numIterations > 65536 ? 1 : 65536 / numIterations;
    size_t accR = 0;
    size_t accG = 0;
    size_t accB = 0;

    for (size_t inputX = outputX * Antialiasing; inputX < (outputX + 1) * Antialiasing; ++inputX) {
        for (size_t inputY = outputY * Antialiasing; inputY < (outputY + 1) * Antialiasing; ++inputY) {
            const uint64_t count = outputIterMatrix[inputY * iterationStride + inputX];
            // Escape status depends on the original count, before rotation and saturation.
            if (count < numIterations) {
                const uint64_t rotated = count >= maximum || paletteRotation >= maximum - count
                                             ? maximum
                                             : count + paletteRotation;
                const uint64_t shifted = rotated >> paletteAuxDepth;
                if (coloringMode == FractalShark::ColoringMode::BasicGrayscale) {
                    const auto gray = (shifted * basicFactor) & 65535;
                    accR += gray;
                    accG += gray;
                    accB += gray;
                } else {
                    const auto palIndex = shifted % paletteSize;
                    accR += palette[palIndex].r;
                    accG += palette[palIndex].g;
                    accB += palette[palIndex].b;
                }
            }
        }
    }

    outputColors[colorIndex].r = accR / totalAA;
    outputColors[colorIndex].g = accG / totalAA;
    outputColors[colorIndex].b = accB / totalAA;
    outputColors[colorIndex].a = 65535;
}

template <typename IterType, uint32_t Antialiasing>
cudaError_t
LaunchColoringTyped(const IterType *iterations,
                    size_t stride,
                    Color16 *colors,
                    const Color16 *palette,
                    uint32_t paletteSize,
                    uint32_t auxDepth,
                    uint64_t rotation,
                    uint64_t maxIterations,
                    uint32_t width,
                    uint32_t height,
                    IterType numIterations,
                    FractalShark::ColoringMode mode,
                    cudaStream_t stream)
{
    // Preserve the renderer's 16-by-8 coloring launch policy.
    constexpr uint32_t threadsWide = 16;
    constexpr uint32_t threadsHigh = 8;
    const dim3 blocks((width - 1) / threadsWide + 1, (height - 1) / threadsHigh + 1);
    const dim3 threads(threadsWide, threadsHigh);
    ColorIterations<IterType, Antialiasing><<<blocks, threads, 0, stream>>>(iterations,
                                                                            stride,
                                                                            colors,
                                                                            palette,
                                                                            paletteSize,
                                                                            auxDepth,
                                                                            rotation,
                                                                            maxIterations,
                                                                            width,
                                                                            height,
                                                                            numIterations,
                                                                            mode);
    return cudaGetLastError();
}

// Launch-only: the caller owns stream completion and any subsequent render reductions.
template <typename IterType>
cudaError_t
LaunchColoring(const IterType *iterations,
               size_t stride,
               Color16 *colors,
               const Color16 *palette,
               uint32_t paletteSize,
               uint32_t auxDepth,
               uint64_t rotation,
               uint64_t maxIterations,
               uint32_t width,
               uint32_t height,
               uint32_t antialiasing,
               IterType numIterations,
               FractalShark::ColoringMode mode,
               cudaStream_t stream)
{
    if (width == 0 || height == 0 || numIterations == 0 || antialiasing < 1 || antialiasing > 4) {
        return cudaErrorInvalidValue;
    }
    switch (antialiasing) {
        case 1:
            return LaunchColoringTyped<IterType, 1>(iterations,
                                                    stride,
                                                    colors,
                                                    palette,
                                                    paletteSize,
                                                    auxDepth,
                                                    rotation,
                                                    maxIterations,
                                                    width,
                                                    height,
                                                    numIterations,
                                                    mode,
                                                    stream);
        case 2:
            return LaunchColoringTyped<IterType, 2>(iterations,
                                                    stride,
                                                    colors,
                                                    palette,
                                                    paletteSize,
                                                    auxDepth,
                                                    rotation,
                                                    maxIterations,
                                                    width,
                                                    height,
                                                    numIterations,
                                                    mode,
                                                    stream);
        case 3:
            return LaunchColoringTyped<IterType, 3>(iterations,
                                                    stride,
                                                    colors,
                                                    palette,
                                                    paletteSize,
                                                    auxDepth,
                                                    rotation,
                                                    maxIterations,
                                                    width,
                                                    height,
                                                    numIterations,
                                                    mode,
                                                    stream);
        default:
            return LaunchColoringTyped<IterType, 4>(iterations,
                                                    stride,
                                                    colors,
                                                    palette,
                                                    paletteSize,
                                                    auxDepth,
                                                    rotation,
                                                    maxIterations,
                                                    width,
                                                    height,
                                                    numIterations,
                                                    mode,
                                                    stream);
    }
}
