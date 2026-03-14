#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

// x_(n + 1) = x_n * x_n - y_n * y_n + a
// y_(n + 1) = 2 * x_n * y_n + b

template <class SharkFloatParams>
void AddHelper(const HpSharkFloat<SharkFloatParams> *A_X2,
               const HpSharkFloat<SharkFloatParams> *B_Y2,
               const HpSharkFloat<SharkFloatParams> *C_A,
               const HpSharkFloat<SharkFloatParams> *D_2X,
               const HpSharkFloat<SharkFloatParams> *E_B,
               HpSharkFloat<SharkFloatParams> *OutXY1,
               HpSharkFloat<SharkFloatParams> *OutXY2,
               DebugHostCombo<SharkFloatParams> &debugHostCombo);

template <class SharkFloatParams>
void AddHelperNR(
    const HpSharkFloat<SharkFloatParams> *X2,
    const HpSharkFloat<SharkFloatParams> *Y2,
    const HpSharkFloat<SharkFloatParams> *CR,
    const HpSharkFloat<SharkFloatParams> *TwoXY,
    const HpSharkFloat<SharkFloatParams> *CI,
    const HpSharkFloat<SharkFloatParams> *W0,
    const HpSharkFloat<SharkFloatParams> *W1,
    const HpSharkFloat<SharkFloatParams> *W2,
    const HpSharkFloat<SharkFloatParams> *W3,
    HpSharkFloat<SharkFloatParams> *OutZReal,
    HpSharkFloat<SharkFloatParams> *OutZImag,
    HpSharkFloat<SharkFloatParams> *OutDzdcReal,
    HpSharkFloat<SharkFloatParams> *OutDzdcImag,
    DebugHostCombo<SharkFloatParams> &debugHostCombo);
