
//
// Return true if we should continue iterating, false if we should stop (period found).
//

template <class SharkFloatParams>
static __device__ void
PeriodicityChecker(cg::grid_group &grid,
                   cg::thread_block &block,
                   uint64_t currentLocalIteration,
                   const typename SharkFloatParams::Float *SharkRestrict cx_cast,
                   const typename SharkFloatParams::Float *SharkRestrict cy_cast,
                   typename SharkFloatParams::Float *dzdcX,
                   typename SharkFloatParams::Float *dzdcY,
                   HpSharkReferenceResults<SharkFloatParams> *SharkRestrict reference)
{
    const auto *ConstantReal = &reference->Add.C_A;
    const auto *ConstantImaginary = &reference->Add.E_B;
    const auto *Out_A_B_C = &reference->Multiply.A;
    const auto *Out_D_E = &reference->Multiply.B;
    const auto radiusY = reference->RadiusY;

    auto *gpuReferenceIters = reference->OutputIters;
    constexpr auto MaxOutputIters = HpSharkReferenceResults<SharkFloatParams>::MaxOutputIters;

    using HdrType = typename SharkFloatParams::Float;
    using PeriodicityResult = typename HpSharkReferenceResults<SharkFloatParams>::PeriodicityResult;

    // Now lets do periodicity checking and store the results
    // Note: first iteration (current overall iteration == 0) requires Out_A_B_C
    // and Out_D_E to be initialized to 0.
    HdrType double_zx = Out_A_B_C->ToHDRFloat<SharkFloatParams::SubType>(0);
    HdrType double_zy = Out_D_E->ToHDRFloat<SharkFloatParams::SubType>(0);

    gpuReferenceIters[currentLocalIteration].x = double_zx;
    gpuReferenceIters[currentLocalIteration].y = double_zy;

    // x^2+2*I*x*y-y^2
    // dzdc = 2.0 * z * dzdc + real(1.0);
    // dzdc = 2.0 * (zx + zy * i) * (dzdcX + dzdcY * i) + HighPrecision(1.0);
    // dzdc = 2.0 * (zx * dzdcX + zx * dzdcY * i + zy * i * dzdcX + zy * i * dzdcY * i) +
    // HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX
    // + 2.0 * zy * i * dzdcY * i + HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY *
    // i + 2.0 * zy * i * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0);
    //
    // dzdcX = 2.0 * zx * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0)
    // dzdcY = 2.0 * zx * dzdcY + 2.0 * zy * dzdcX

    HdrReduce(*dzdcX);
    auto dzdcX1 = HdrAbs(*dzdcX);

    HdrReduce(*dzdcY);
    auto dzdcY1 = HdrAbs(*dzdcY);

    HdrReduce(double_zx);
    auto zxCopy1 = HdrAbs(double_zx);

    HdrReduce(double_zy);
    auto zyCopy1 = HdrAbs(double_zy);

    HdrType n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

    const HdrType HighTwo{2.0f};
    const HdrType HighOne{1.0f};
    const HdrType TwoFiftySix{256.0f};

    HdrType r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
    auto n3 = radiusY * r0 * HighTwo;
    HdrReduce(n3);

    if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
        reference->OutputIterCount = currentLocalIteration + 1;
        reference->PeriodicityStatus = PeriodicityResult::PeriodFound;
        return;
    } else {
        auto dzdcXOrig = *dzdcX;
        *dzdcX = HighTwo * (double_zx * *dzdcX - double_zy * *dzdcY) + HighOne;
        *dzdcY = HighTwo * (double_zx * *dzdcY + double_zy * dzdcXOrig);
    }

    HdrType tempZX = double_zx + *cx_cast;
    HdrType tempZY = double_zy + *cy_cast;
    HdrType zn_size = tempZX * tempZX + tempZY * tempZY;

    if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {

        //
        // Escaped
        //

        reference->PeriodicityStatus = PeriodicityResult::Escaped;
        reference->OutputIterCount = currentLocalIteration + 1;
        return;
    }

    reference->PeriodicityStatus = PeriodicityResult::Continue;
    reference->OutputIterCount = currentLocalIteration + 1;
    return;
}
