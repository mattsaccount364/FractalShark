#pragma once

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams>
void MultiplyHelperFFT2(const HpSharkFloat<SharkFloatParams> *A,
                        const HpSharkFloat<SharkFloatParams> *B,
                        HpSharkFloat<SharkFloatParams> *OutXX,
                        HpSharkFloat<SharkFloatParams> *OutXY,
                        HpSharkFloat<SharkFloatParams> *OutYY,
                        DebugHostCombo<SharkFloatParams> &debugCombo);

template <class SharkFloatParams>
void MultiplyHelperNR(
    const HpSharkFloat<SharkFloatParams> *zReal,
    const HpSharkFloat<SharkFloatParams> *zImag,
    const HpSharkFloat<SharkFloatParams> *dzdcReal,
    const HpSharkFloat<SharkFloatParams> *dzdcImag,
    HpSharkFloat<SharkFloatParams> *OutX2,
    HpSharkFloat<SharkFloatParams> *Out2XY,
    HpSharkFloat<SharkFloatParams> *OutY2,
    HpSharkFloat<SharkFloatParams> *OutW0,
    HpSharkFloat<SharkFloatParams> *OutW1,
    HpSharkFloat<SharkFloatParams> *OutW2,
    HpSharkFloat<SharkFloatParams> *OutW3,
    DebugHostCombo<SharkFloatParams> &debugCombo);
