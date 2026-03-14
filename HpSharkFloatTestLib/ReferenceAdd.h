#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

// x_(n + 1) = x_n * x_n - y_n * y_n + a
// y_(n + 1) = 2 * x_n * y_n + b
//
// When SharkFloatParams::EnableNewtonRaphson is true, also computes:
//   OutDzdcReal = W0 - W1 + 1.0
//   OutDzdcImag = W2 + W3

template <class SharkFloatParams>
void AddHelper(const HpSharkFloat<SharkFloatParams> *A_X2,
               const HpSharkFloat<SharkFloatParams> *B_Y2,
               const HpSharkFloat<SharkFloatParams> *C_A,
               const HpSharkFloat<SharkFloatParams> *D_2X,
               const HpSharkFloat<SharkFloatParams> *E_B,
               HpSharkFloat<SharkFloatParams> *OutXY1,
               HpSharkFloat<SharkFloatParams> *OutXY2,
               const HpSharkFloat<SharkFloatParams> *W0,
               const HpSharkFloat<SharkFloatParams> *W1,
               const HpSharkFloat<SharkFloatParams> *W2,
               const HpSharkFloat<SharkFloatParams> *W3,
               HpSharkFloat<SharkFloatParams> *OutDzdcReal,
               HpSharkFloat<SharkFloatParams> *OutDzdcImag,
               DebugHostCombo<SharkFloatParams> &debugHostCombo);
