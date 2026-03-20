#include "ReferenceReferenceOrbit.h"
#include "DbgHeap.h"
#include "DebugChecksumHost.h"
#include "HpSharkFloat.h"
#include "ReferenceAdd.h"
#include "ReferenceNTT2.h"
#include "TestVerbose.h"

#include "HDRFloat.h"

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <vector>

//
// CPU single-threaded reference orbit computation.
// Mirrors the GPU HpSharkReferenceGpuLoop kernel:
//   1. Periodicity check (if enabled)
//   2. Multiply: z^2 via NTT (MultiplyHelperFFT2)
//   3. Add:     z^2 + c      (AddHelper)
//
// z_0 is initialized to c (same as GPU: combo->Multiply.A = c_real,
// combo->Multiply.B = c_imag).
//

template <class SharkFloatParams>
std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>
ReferenceOrbitHelper(const HpSharkFloat<SharkFloatParams> *cReal,
                     const HpSharkFloat<SharkFloatParams> *cImag,
                     const typename SharkFloatParams::Float &radiusY,
                     uint64_t maxIters,
                     DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    auto result = std::make_unique<ReferenceOrbitResult<SharkFloatParams>>();
    result->IterationsExecuted = 0;
    result->PeriodResult = PeriodicityResult::Unknown;

    // For NR types, use EvaluateOrbitAndDerivative which properly computes derivatives.
    // The basic orbit loop can't handle NR because the multiply/add pipelines require
    // valid NR inputs (not zeros) when EnableNewtonRaphson is true.
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        auto outZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        auto outDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
        typename SharkFloatParams::Float outD2Real{};
        typename SharkFloatParams::Float outD2Imag{};

        EvaluateOrbitAndDerivative<SharkFloatParams>(
            cReal, cImag, maxIters + 1,
            outZReal.get(), outZImag.get(),
            outDzdcReal.get(), outDzdcImag.get(),
            &outD2Real, &outD2Imag,
            debugHostCombo);

        result->FinalZReal = *outZReal;
        result->FinalZImag = *outZImag;
        result->IterationsExecuted = maxIters;
        result->PeriodResult = PeriodicityResult::Continue;
        return result;
    }

    // Non-NR: basic orbit loop (z = z^2 + c)
    auto zReal = std::make_unique<HpSharkFloat<SharkFloatParams>>(*cReal);
    auto zImag = std::make_unique<HpSharkFloat<SharkFloatParams>>(*cImag);

    // Intermediate multiply results
    auto resultX2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();   // z_real^2
    auto result2XY = std::make_unique<HpSharkFloat<SharkFloatParams>>();  // 2 * z_real * z_imag
    auto resultY2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();   // z_imag^2

    // Intermediate add results (new z values)
    auto newZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // Periodicity tracking: dzdc derivative
    typename SharkFloatParams::Float dzdcX{1};
    typename SharkFloatParams::Float dzdcY{0};

    const typename SharkFloatParams::Float HighTwo{2.0f};
    const typename SharkFloatParams::Float HighOne{1.0f};
    const typename SharkFloatParams::Float TwoFiftySix{256.0f};

    // Convert constants to HDRFloat for periodicity/escape checks
    const typename SharkFloatParams::Float cx_cast =
        cReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
    const typename SharkFloatParams::Float cy_cast =
        cImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

    for (uint64_t i = 0; i < maxIters; ++i) {

        //
        // Step 1: Periodicity checking (mirrors GPU PeriodicityChecker)
        // This runs BEFORE the multiply/add, exactly like the GPU kernel.
        //
        if constexpr (SharkFloatParams::EnablePeriodicity) {
            typename SharkFloatParams::Float double_zx =
                zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float double_zy =
                zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

            // Store orbit point
            result->Orbit.push_back({double_zx, double_zy});

            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);

            HdrReduce(double_zx);
            auto zxCopy1 = HdrAbs(double_zx);

            HdrReduce(double_zy);
            auto zyCopy1 = HdrAbs(double_zy);

            typename SharkFloatParams::Float n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

            typename SharkFloatParams::Float r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
            auto n3 = radiusY * r0 * HighTwo;
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::PeriodFound;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                return result;
            } else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
            }

            // Escape check
            typename SharkFloatParams::Float tempZX = double_zx + cx_cast;
            typename SharkFloatParams::Float tempZY = double_zy + cy_cast;
            typename SharkFloatParams::Float zn_size = tempZX * tempZX + tempZY * tempZY;

            if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::Escaped;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                return result;
            }
        } else {
            // No periodicity — still store orbit point for comparison
            typename SharkFloatParams::Float double_zx = zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float double_zy = zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            result->Orbit.push_back({double_zx, double_zy});
        }

        //
        // Step 2: Multiply — z^2 via NTT
        // MultiplyHelperFFT2(A, B) -> (A*A, A*B [*2], B*B)
        //
        MultiplyHelperFFT2<SharkFloatParams>(zReal.get(),
                                             zImag.get(),
                                             resultX2.get(),
                                             result2XY.get(),
                                             resultY2.get(),
                                             nullptr, nullptr,
                                             nullptr, nullptr, nullptr, nullptr,
                                             debugHostCombo);

        //
        // Step 3: Add — z^2 + c
        // AddHelper(x^2, y^2, c_real, 2xy, c_imag) -> (x^2 - y^2 + a, 2xy + b)
        //
        AddHelper<SharkFloatParams>(resultX2.get(),
                                    resultY2.get(),
                                    cReal,
                                    result2XY.get(),
                                    cImag,
                                    newZReal.get(),
                                    newZImag.get(),
                                    nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr,
                                    debugHostCombo);

        // Update z for next iteration
        *zReal = *newZReal;
        *zImag = *newZImag;

        result->IterationsExecuted = i + 1;
        result->PeriodResult = PeriodicityResult::Continue;
    }

    result->FinalZReal = *zReal;
    result->FinalZImag = *zImag;
    return result;
}

//
// Newton-Raphson inner loop: iterate z = z^2 + c for `period` steps,
// tracking both z_p and dz/dc_p.
//
// dz/dc starts at 0 (not 1) because this is the *orbital* derivative
// f_p'(c), not the running per-iteration derivative used in
// periodicity checking.
//

template <class SharkFloatParams>
void
EvaluateOrbitAndDerivative(
    const HpSharkFloat<SharkFloatParams> *cReal,
    const HpSharkFloat<SharkFloatParams> *cImag,
    uint64_t period,
    HpSharkFloat<SharkFloatParams> *outZReal,
    HpSharkFloat<SharkFloatParams> *outZImag,
    HpSharkFloat<SharkFloatParams> *outDzdcReal,
    HpSharkFloat<SharkFloatParams> *outDzdcImag,
    typename SharkFloatParams::Float *outD2Real,
    typename SharkFloatParams::Float *outD2Imag,
    DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    static_assert(SharkFloatParams::EnableNewtonRaphson,
                  "EvaluateOrbitAndDerivative requires EnableNewtonRaphson = true");

    // z = 0, dz/dc = 0.
    auto zReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto dzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto dzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // d2 = 0 (HDRFloat, matches production local_d2r/local_d2i)
    typename SharkFloatParams::Float local_d2r{};
    typename SharkFloatParams::Float local_d2i{};

    // Intermediate multiply results (7 products)
    auto x2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto twoXY = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto y2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto w0 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto w1 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto w2 = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto w3 = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    // Intermediate add results
    auto newZReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newZImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newDzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto newDzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();

    for (uint64_t i = 0; i < period; ++i) {
        // d2 update BEFORE multiply/add (uses current z/dzdc, matches production order)
        {
            typename SharkFloatParams::Float zr = zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float zi = zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float dzr = dzdcReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            typename SharkFloatParams::Float dzi = dzdcImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

            // dzdc²
            typename SharkFloatParams::Float dz2r = dzr * dzr - dzi * dzi;
            HdrReduce(dz2r);
            typename SharkFloatParams::Float dz2i = typename SharkFloatParams::Float{2.0f} * (dzr * dzi);
            HdrReduce(dz2i);

            // z * d2
            typename SharkFloatParams::Float zd2r = zr * local_d2r - zi * local_d2i;
            HdrReduce(zd2r);
            typename SharkFloatParams::Float zd2i = zr * local_d2i + zi * local_d2r;
            HdrReduce(zd2i);

            // d2 = 2*(dzdc² + z*d2)
            typename SharkFloatParams::Float sumr = dz2r + zd2r;
            HdrReduce(sumr);
            typename SharkFloatParams::Float sumi = dz2i + zd2i;
            HdrReduce(sumi);
            local_d2r = typename SharkFloatParams::Float{2.0f} * sumr;
            local_d2i = typename SharkFloatParams::Float{2.0f} * sumi;
        }

        MultiplyHelperFFT2<SharkFloatParams>(
            zReal.get(), zImag.get(),
            x2.get(), twoXY.get(), y2.get(),
            dzdcReal.get(), dzdcImag.get(),
            w0.get(), w1.get(), w2.get(), w3.get(),
            debugHostCombo);

        AddHelper<SharkFloatParams>(
            x2.get(), y2.get(), cReal,
            twoXY.get(), cImag,
            newZReal.get(), newZImag.get(),
            w0.get(), w1.get(), w2.get(), w3.get(),
            newDzdcReal.get(), newDzdcImag.get(),
            debugHostCombo);

        *zReal = *newZReal;
        *zImag = *newZImag;
        *dzdcReal = *newDzdcReal;
        *dzdcImag = *newDzdcImag;
    }

    *outZReal = *zReal;
    *outZImag = *zImag;
    *outDzdcReal = *dzdcReal;
    *outDzdcImag = *dzdcImag;
    *outD2Real = local_d2r;
    *outD2Imag = local_d2i;
}

//
// Explicit instantiation
//
#define ExplicitlyInstantiate(SharkFloatParams)                                                         \
    template std::unique_ptr<ReferenceOrbitResult<SharkFloatParams>>                                    \
    ReferenceOrbitHelper<SharkFloatParams>(                                                             \
        const HpSharkFloat<SharkFloatParams> *,                                                        \
        const HpSharkFloat<SharkFloatParams> *,                                                        \
        const typename SharkFloatParams::Float &,                                                      \
        uint64_t,                                                                                      \
        DebugHostCombo<SharkFloatParams> &);

ExplicitInstantiateAll();

// NR-specific function — only instantiated for NR-enabled param types
template void EvaluateOrbitAndDerivative<SharkParamsNR7>(
    const HpSharkFloat<SharkParamsNR7> *,
    const HpSharkFloat<SharkParamsNR7> *,
    uint64_t,
    HpSharkFloat<SharkParamsNR7> *,
    HpSharkFloat<SharkParamsNR7> *,
    HpSharkFloat<SharkParamsNR7> *,
    HpSharkFloat<SharkParamsNR7> *,
    typename SharkParamsNR7::Float *,
    typename SharkParamsNR7::Float *,
    DebugHostCombo<SharkParamsNR7> &);
