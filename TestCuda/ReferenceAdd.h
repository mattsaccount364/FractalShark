#pragma once

#include <stdint.h>
#include <vector>

template <class SharkFloatParams>
struct DebugStateHost;

template<class SharkFloatParams>
struct HpSharkFloat;

// x_(n + 1) = x_n * x_n - y_n * y_n + a
// y_(n + 1) = 2 * x_n * y_n + b

template<class SharkFloatParams>
void NewFunction(const bool sameSignDE, const int32_t &extDigits, const auto *&ext_D_2x, const int32_t &actualDigits, const auto *&ext_E_B, const int32_t &shiftD, const int32_t &shiftE, const bool &DIsBiggerMagnitude, const int32_t &diffDE, std::vector<uint64_t> &extResult_D_E, std::vector<DebugStateHost<SharkFloatParams>> &debugStates);

template<class SharkFloatParams>
void AddHelper(
    const HpSharkFloat<SharkFloatParams> *A_X2,
    const HpSharkFloat<SharkFloatParams> *B_Y2,
    const HpSharkFloat<SharkFloatParams> *C_A,
    const HpSharkFloat<SharkFloatParams> *D_2X,
    const HpSharkFloat<SharkFloatParams> *E_B,
    HpSharkFloat<SharkFloatParams> *OutXY1,
    HpSharkFloat<SharkFloatParams> *OutXY2,
    std::vector<DebugStateHost<SharkFloatParams>> &debugStates
);
