template <class SharkFloatParams>
static __device__ bool
PeriodicityChecker(cg::grid_group &grid,
    cg::thread_block &block,
    const HpSharkFloat<SharkFloatParams> *A_X2,
    const HpSharkFloat<SharkFloatParams> *B_Y2,
    const HpSharkFloat<SharkFloatParams> *C_A,
    const HpSharkFloat<SharkFloatParams> *D_2X,
    const HpSharkFloat<SharkFloatParams> *E_B,
    HpSharkFloat<SharkFloatParams> *Out_A_B_C,
    HpSharkFloat<SharkFloatParams> *Out_D_E,
    uint64_t *tempData) {

#if 0
    HdrReduce(dzdcX);
    auto dzdcX1 = HdrAbs(dzdcX);

    HdrReduce(dzdcY);
    auto dzdcY1 = HdrAbs(dzdcY);

    HdrReduce(double_zx);
    auto zxCopy1 = HdrAbs(double_zx);

    HdrReduce(double_zy);
    auto zyCopy1 = HdrAbs(double_zy);

    T n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

    T r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
    auto n3 = results->GetMaxRadius() * r0 * HighTwo; // TODO optimize HDRFloat *2.
    HdrReduce(n3);

    if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
        if constexpr (BenchmarkState == BenchmarkMode::Disable) {
            periodicity_should_break = true;
        }
    } else {
        auto dzdcXOrig = dzdcX;
        dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
        dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
    }
#endif

    return false;
}