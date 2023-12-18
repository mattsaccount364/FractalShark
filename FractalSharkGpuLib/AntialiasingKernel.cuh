template<typename IterType, uint32_t Antialiasing, bool ScaledColor>
__global__
void
antialiasing_kernel(
    const IterType* __restrict__ OutputIterMatrix,
    uint32_t Width,
    uint32_t Height,
    AntialiasedColors OutputColorMatrix,
    Palette Pals,
    int local_color_width,
    int local_color_height,
    IterType n_iterations) {
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_x >= local_color_width || output_y >= local_color_height)
        return;

    const int32_t color_idx = local_color_width * output_y + output_x; // do not use ConvertLocToIndex
    constexpr auto totalAA = Antialiasing * Antialiasing;

    // TODO reduction
    //if constexpr (ScaledColor) {
    //    IterType maxIters = 0, minIters;
    //    for (size_t input_x = output_x * Antialiasing;
    //        input_x < (output_x + 1) * Antialiasing;
    //        input_x++) {
    //        for (size_t input_y = output_y * Antialiasing;
    //            input_y < (output_y + 1) * Antialiasing;
    //            input_y++) {
    //            size_t idx = ConvertLocToIndex(input_x, input_y, Width);
    //            IterType numIters = OutputIterMatrix[idx];
    //        }
    //    }
    //}

    size_t acc_r = 0;
    size_t acc_g = 0;
    size_t acc_b = 0;

    for (size_t input_x = output_x * Antialiasing;
        input_x < (output_x + 1) * Antialiasing;
        input_x++) {
        for (size_t input_y = output_y * Antialiasing;
            input_y < (output_y + 1) * Antialiasing;
            input_y++) {

            //size_t idx = input_y * Width + input_x;
            size_t idx = ConvertLocToIndex(input_x, input_y, Width);
            IterType numIters = OutputIterMatrix[idx];

            if (numIters < n_iterations) {
                const auto palIndex = (numIters >> Pals.palette_aux_depth) % Pals.local_palIters;
                acc_r += Pals.local_pal[palIndex].r;
                acc_g += Pals.local_pal[palIndex].g;
                acc_b += Pals.local_pal[palIndex].b;
            }
        }
    }

    acc_r /= totalAA;
    acc_g /= totalAA;
    acc_b /= totalAA;

    OutputColorMatrix.aa_colors[color_idx].r = acc_r;
    OutputColorMatrix.aa_colors[color_idx].g = acc_g;
    OutputColorMatrix.aa_colors[color_idx].b = acc_b;
    OutputColorMatrix.aa_colors[color_idx].a = 65535;
}
