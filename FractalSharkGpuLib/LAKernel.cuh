#pragma once

template <typename IterType, class T, class SubType, LAv2Mode Mode, PerturbExtras PExtras>
__global__ void
//__launch_bounds__(NB_THREADS_W * NB_THREADS_H, 2)
mandel_1xHDR_float_perturb_lav2(IterType *OutputIterMatrix,
                                AntialiasedColors OutputColorMatrix,
                                GPUPerturbSingleResults<IterType, T, PExtras> Perturb,
                                GPU_LAReference<IterType, T, SubType> LaReference, // "copy"
                                int width,
                                int height,
                                int antialiasing,
                                const T cx,
                                const T cy,
                                const T dx,
                                const T dy,
                                const T centerX,
                                const T centerY,
                                IterType maxIterations)
{
    static constexpr bool IsHDR = std::is_same<T, ::HDRFloat<float>>::value ||
                                  std::is_same<T, ::HDRFloat<double>>::value ||
                                  std::is_same<T, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
                                  std::is_same<T, ::HDRFloat<CudaDblflt<dblflt>>>::value;

    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;

    if (X >= width || Y >= height)
        return;

    size_t idx = ConvertLocToIndex(X, Y, width);
    // if (OutputIterMatrix[idx] != 0) {
    //     return;
    // }

    IterType iter = 0;
    IterType RefIteration = 0;
    const T DeltaReal = dx * T(X) - centerX;
    const T DeltaImaginary = -dy * T(Y) - centerY;

    const T DeltaSub0X = DeltaReal;
    const T DeltaSub0Y = DeltaImaginary;
    T DeltaSubNX = T(0.0f);
    T DeltaSubNY = T(0.0f);

    static constexpr bool ConditionalResult = std::is_same<T, float>::value ||
                                              std::is_same<T, double>::value ||
                                              std::is_same<T, CudaDblflt<dblflt>>::value;
    using TComplex = typename std::
        conditional<ConditionalResult, FloatComplex<SubType>, HDRFloatComplex<SubType>>::type;

    ////////////
    TComplex DeltaSub0;
    TComplex DeltaSubN;

    DeltaSub0 = {DeltaReal, DeltaImaginary};
    DeltaSubN = {T(0), T(0)};

    if constexpr (Mode == LAv2Mode::Full || Mode == LAv2Mode::LAO) {
        TComplex complex0{};
        const auto readOrbit = [&](IterType index) {
            T real;
            T imaginary;
            Perturb.GetIterRandom(index, real, imaginary);
            return TComplex{real, imaginary};
        };
        FractalShark::LA::AdvancePixel(LaReference,
                                       maxIterations,
                                       DeltaSub0,
                                       Perturb.GetCountOrbitEntries() - 1,
                                       Perturb.GetPeriodMaybeZero(),
                                       readOrbit,
                                       iter,
                                       RefIteration,
                                       DeltaSubN,
                                       complex0);
    }

    if constexpr (Mode == LAv2Mode::Full || Mode == LAv2Mode::PO) {
        DeltaSubNX = DeltaSubN.getRe();
        DeltaSubNY = DeltaSubN.getIm();

        //////////////////////

        auto perturbLoop = [&](IterType maxIterations) {
            typename GPUPerturbSingleResults<IterType, T, PExtras>::SeqWorkspace workspace{Perturb,
                                                                                           RefIteration};

            for (;;) {
                const T DeltaSubNXOrig{DeltaSubNX};
                const T DeltaSubNYOrig{DeltaSubNY};

                T tempMulX2 = workspace.zx * T{2};
                T tempMulY2 = workspace.zy * T{2};

                ++RefIteration;

                const auto tempSum1{tempMulY2 + DeltaSubNYOrig};
                const auto tempSum2{tempMulX2 + DeltaSubNXOrig};

                if constexpr (std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value) {
                    T::custom_perturb3(DeltaSubNX,
                                       DeltaSubNY,
                                       DeltaSubNXOrig,
                                       tempSum2,
                                       DeltaSubNYOrig,
                                       tempSum1,
                                       DeltaSub0X,
                                       DeltaSub0Y);
                } else if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value ||
                                     std::is_same<T, CudaDblflt<dblflt>>::value) {
                    DeltaSubNX = DeltaSubNXOrig * tempSum2 - DeltaSubNYOrig * tempSum1 + DeltaSub0X;
                    HdrReduce(DeltaSubNX);

                    DeltaSubNY = DeltaSubNXOrig * tempSum1 + DeltaSubNYOrig * tempSum2 + DeltaSub0Y;
                    HdrReduce(DeltaSubNY);
                } else {
                    // DeltaSubNX =
                    //     DeltaSubNXOrig * tempSum2 -
                    //     DeltaSubNYOrig * tempSum1 +
                    //     DeltaSub0X;
                    // HdrReduce(DeltaSubNX);

                    // DeltaSubNY =
                    //     DeltaSubNXOrig * tempSum1 +
                    //     DeltaSubNYOrig * tempSum2 +
                    //     DeltaSub0Y;
                    // HdrReduce(DeltaSubNY);
                    T::custom_perturb2(DeltaSubNX,
                                       DeltaSubNY,
                                       DeltaSubNXOrig,
                                       tempSum2,
                                       DeltaSubNYOrig,
                                       tempSum1,
                                       DeltaSub0X,
                                       DeltaSub0Y);
                }

                Perturb.GetIterSeq(workspace);
                T tempVal1X = workspace.zx;
                T tempVal1Y = workspace.zy;

                const T tempZX{tempVal1X + DeltaSubNX};
                const T tempZY{tempVal1Y + DeltaSubNY};
                T normSquared;
                if constexpr (IsHDR) {
                    normSquared = {HdrReduce(tempZX.square() + tempZY.square())};
                } else {
                    normSquared = {tempZX * tempZX + tempZY * tempZY};
                }

                if (HdrCompareToBothPositiveReducedLT<T, 256>(normSquared) && iter < maxIterations) {
                    T DeltaNormSquared;
                    if constexpr (IsHDR) {
                        DeltaNormSquared = {HdrReduce(DeltaSubNX.square() + DeltaSubNY.square())};
                    } else {
                        DeltaNormSquared = {DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY};
                    }

                    if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                        RefIteration >= Perturb.GetCountOrbitEntries() - 1) {
                        DeltaSubNX = tempZX;
                        DeltaSubNY = tempZY;
                        RefIteration = 0;

                        workspace = {Perturb, RefIteration};
                    }

                    ++iter;
                } else {
                    break;
                }
            }
        };

        //    for (;;) {
        __syncthreads();

        perturbLoop(maxIterations);
    }

    // TODO
    // This code is moved from an old 1x32 float perturb-only implementation.
    // It'd be fun to hook this up again
    //
    //// Just finds the interesting Misiurewicz points.  Breaks so they're colored differently
    // if constexpr (Periodic) {
    //     auto n3 = maxRadiusSq * (dzdcX * dzdcX + dzdcY * dzdcY);
    //     HdrReduce(n3);

    //    if (HdrCompareToBothPositiveReducedGE(zn_size, n3)) {
    //        // dzdc = dzdc * 2.0 * z + ScalingFactor;
    //        // dzdc = dzdc * 2.0 * tempZ + ScalingFactor;
    //        // dzdc = (dzdcX + dzdcY * i) * 2.0 * (tempZX + tempZY * i) + ScalingFactor;
    //        // dzdc = (dzdcX * 2.0 + dzdcY * i * 2.0) * (tempZX + tempZY * i) + ScalingFactor;
    //        // dzdc = (dzdcX * 2.0) * tempZX +
    //        //        (dzdcX * 2.0) * (tempZY * i) +
    //        //        (dzdcY * i * 2.0) * tempZX +
    //        //        (dzdcY * i * 2.0) * tempZY * i
    //        //
    //        // dzdcX = (dzdcX * 2.0) * tempZX -
    //        //         (dzdcY * 2.0) * tempZY
    //        // dzdcY = (dzdcX * 2.0) * (tempZY) +
    //        //         (dzdcY * 2.0) * tempZX
    //        auto dzdcXOrig = dzdcX;
    //        dzdcX = T(2.0f) * tempZX * dzdcX - T(2.0f) * tempZY * dzdcY + scalingFactor;
    //        HdrReduce(dzdcX);

    //        dzdcY = T(2.0f) * tempZY * dzdcXOrig + T(2.0f) * tempZX * dzdcY;
    //        HdrReduce(dzdcY);
    //    }
    //    else {
    //        //iter = maxIterations;
    //        break;
    //    }
    //}

    __syncthreads();

    OutputIterMatrix[idx] = iter;
}
