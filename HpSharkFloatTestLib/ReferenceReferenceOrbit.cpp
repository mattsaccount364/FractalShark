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
    using HdrType = typename SharkFloatParams::Float;

    auto result = std::make_unique<ReferenceOrbitResult<SharkFloatParams>>();
    result->IterationsExecuted = 0;
    result->PeriodResult = PeriodicityResult::Unknown;

    // Current z value (real, imaginary) — starts at c, same as GPU init.
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
    HdrType dzdcX{1};
    HdrType dzdcY{0};

    const HdrType HighTwo{2.0f};
    const HdrType HighOne{1.0f};
    const HdrType TwoFiftySix{256.0f};

    // Convert constants to HDRFloat for periodicity/escape checks
    const HdrType cx_cast = cReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
    const HdrType cy_cast = cImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

    for (uint64_t i = 0; i < maxIters; ++i) {

        //
        // Step 1: Periodicity checking (mirrors GPU PeriodicityChecker)
        // This runs BEFORE the multiply/add, exactly like the GPU kernel.
        //
        if constexpr (SharkFloatParams::EnablePeriodicity) {
            HdrType double_zx = zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            HdrType double_zy = zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);

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

            HdrType n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

            HdrType r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
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
            HdrType tempZX = double_zx + cx_cast;
            HdrType tempZY = double_zy + cy_cast;
            HdrType zn_size = tempZX * tempZX + tempZY * tempZY;

            if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
                result->IterationsExecuted = i + 1;
                result->PeriodResult = PeriodicityResult::Escaped;
                result->FinalZReal = *zReal;
                result->FinalZImag = *zImag;
                return result;
            }
        } else {
            // No periodicity — still store orbit point for comparison
            HdrType double_zx = zReal->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
            HdrType double_zy = zImag->template ToHDRFloat<typename SharkFloatParams::SubType>(0);
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
// Each iteration:
//   z_new       = z^2 + c
//   dz/dc_new   = 2*z*dz/dc + 1
//
// The multiply helper produces 7 products:
//   x2  = zR^2,   twoXY = 2*zR*zI,   y2  = zI^2
//   w0  = dzdcR*2zR,   w1 = dzdcI*2zI,   w2 = dzdcR*2zI,   w3 = dzdcI*2zR
//
// The add helper combines them:
//   newZR     = x2 - y2 + cR
//   newZI     = twoXY + cI
//   newDzdcR  = w0 - w1 + 1
//   newDzdcI  = w2 + w3
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
    DebugHostCombo<SharkFloatParams> &debugHostCombo)
{
    // z = 0, dz/dc = 0.
    // Default constructor zeros Digits and IsNegative; Exponent is
    // set to the sentinel value (min int32_t) representing zero.
    auto zReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto zImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto dzdcReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto dzdcImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();

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
        MultiplyHelperNR<SharkFloatParams>(
            zReal.get(), zImag.get(),
            dzdcReal.get(), dzdcImag.get(),
            x2.get(), twoXY.get(), y2.get(),
            w0.get(), w1.get(), w2.get(), w3.get(),
            debugHostCombo);

        AddHelperNR<SharkFloatParams>(
            x2.get(), y2.get(), cReal,
            twoXY.get(), cImag,
            w0.get(), w1.get(), w2.get(), w3.get(),
            newZReal.get(), newZImag.get(),
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
    DebugHostCombo<SharkParamsNR7> &);
