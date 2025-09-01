#pragma once

template <class SharkFloatParams> struct DebugHostCombo;

template <class SharkFloatParams> struct HpSharkFloat;

template <class SharkFloatParams>
void MultiplyHelperFFT2(const HpSharkFloat<SharkFloatParams>* A,
                        const HpSharkFloat<SharkFloatParams>* B,
                        HpSharkFloat<SharkFloatParams>* OutXX,
                        HpSharkFloat<SharkFloatParams>* OutXY,
                        HpSharkFloat<SharkFloatParams>* OutYY,
                        DebugHostCombo<SharkFloatParams>& debugCombo);