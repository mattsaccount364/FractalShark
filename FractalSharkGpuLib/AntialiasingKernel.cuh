#pragma once

template <typename IterType, uint32_t Antialiasing, bool ScaledColor>
__global__ void
antialiasing_kernel(const IterType *__restrict__ outputIterMatrix,
                    uint32_t width,
                    uint32_t height,
                    AntialiasedColors outputColorMatrix,
                    Palette pals,
                    int colorWidth,
                    int colorHeight,
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
    const uint64_t maximum = pals.m_MaxPossibleIterations - 1;
    const uint64_t basicFactor = numIterations > 65536 ? 1 : 65536 / numIterations;
    size_t accR = 0;
    size_t accG = 0;
    size_t accB = 0;

    for (size_t inputX = outputX * Antialiasing; inputX < (outputX + 1) * Antialiasing; ++inputX) {
        for (size_t inputY = outputY * Antialiasing; inputY < (outputY + 1) * Antialiasing; ++inputY) {
            const uint64_t count = outputIterMatrix[ConvertLocToIndex(inputX, inputY, width)];
            // Escape status depends on the original count, before rotation and saturation.
            if (count < numIterations) {
                const uint64_t rotated = count >= maximum || pals.m_PaletteRotation >= maximum - count
                                             ? maximum
                                             : count + pals.m_PaletteRotation;
                const uint64_t shifted = rotated >> pals.palette_aux_depth;
                if (coloringMode == FractalShark::ColoringMode::BasicGrayscale) {
                    const auto gray = (shifted * basicFactor) & 65535;
                    accR += gray;
                    accG += gray;
                    accB += gray;
                } else {
                    const auto palIndex = shifted % pals.local_palIters;
                    accR += pals.local_pal[palIndex].r;
                    accG += pals.local_pal[palIndex].g;
                    accB += pals.local_pal[palIndex].b;
                }
            }
        }
    }

    outputColorMatrix.aa_colors[colorIndex].r = accR / totalAA;
    outputColorMatrix.aa_colors[colorIndex].g = accG / totalAA;
    outputColorMatrix.aa_colors[colorIndex].b = accB / totalAA;
    outputColorMatrix.aa_colors[colorIndex].a = 65535;
}
