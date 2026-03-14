#pragma once

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams>
void MultiplyHelperFFT2(const HpSharkFloat<SharkFloatParams> *A,
                        const HpSharkFloat<SharkFloatParams> *B,
                        HpSharkFloat<SharkFloatParams> *OutXX,
                        HpSharkFloat<SharkFloatParams> *OutXY,
                        HpSharkFloat<SharkFloatParams> *OutYY,
                        const HpSharkFloat<SharkFloatParams> *dzdcReal,
                        const HpSharkFloat<SharkFloatParams> *dzdcImag,
                        HpSharkFloat<SharkFloatParams> *OutW0,
                        HpSharkFloat<SharkFloatParams> *OutW1,
                        HpSharkFloat<SharkFloatParams> *OutW2,
                        HpSharkFloat<SharkFloatParams> *OutW3,
                        DebugHostCombo<SharkFloatParams> &debugCombo);
