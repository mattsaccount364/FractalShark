#pragma once

#include "HpSharkFloat.h"

enum class BasicCorrectnessMode : int {
    Error = 0,
    Correctness_P1 = 1,
    Correctness_NR = 2,
    PerfSweep = 3,
    // Non-NR perf views
    PerfSingleView5 = 4,
    PerfSingleView30 = 5,
    PerfSingleView32 = 6,
    // NR perf views
    PerfSingleNRView5 = 7,
    PerfSingleNRView30 = 8,
    PerfSingleNRView32 = 9,
    Correctness_P1_to_P5 = 10,
    PerfSingleRef = 11,
    PerfSingleViewAny = 12 // run reference-orbit perf test for any view 1..34
};

namespace HpShark {

static constexpr bool TestGpu = true;

} // namespace HpShark

// Correctness test sizes
using TestCorrectnessSharkParams1 = SharkParamsNP1;
using TestCorrectnessSharkParams2 = SharkParamsNP2;
using TestCorrectnessSharkParams3 = SharkParamsNP4;
using TestCorrectnessSharkParams4 = SharkParamsNP3;
using TestCorrectnessSharkParams5 = SharkParamsNP5;
