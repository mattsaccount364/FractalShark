#pragma once

// Test-only configuration for HpSharkFloatTest / HpSharkFloatTestLib.
// Not used by production code (FractalShark, FractalSharkLib, FractalSharkGpuLib).

#include "HpSharkFloat.h"

enum class BasicCorrectnessMode : int {
    Error = 0,
    Correctness_P1 = 1,
    Correctness_NR = 2,
    PerfSub = 3,
    PerfSweep = 4,
    // Non-NR perf views
    PerfSingleView5 = 5,
    PerfSingleView30 = 6,
    PerfSingleView32 = 7,
    // NR perf views
    PerfSingleNRView5 = 8,
    PerfSingleNRView30 = 9,
    PerfSingleNRView32 = 10,
    Correctness_P1_to_P5 = 11,
    PerfSingleRef = 12,
    PerfSingleViewAny = 13 // run reference-orbit perf test for any view 1..34
};

namespace HpShark {

// static constexpr bool TestGpu = true;
static constexpr bool TestGpu = true;
static constexpr bool TestInfiniteCorrectness = true;
static constexpr bool TestMPIRImpl = true;

// True to compare against the full host-side reference implementation, false is MPIR only
// False is useful to speed up e.g. testing many cases fast but gives poor diagnostic results.
static constexpr bool TestReferenceImpl = true;

} // namespace HpShark

// Correctness test sizes
using TestCorrectnessSharkParams1 = SharkParamsNP1;
using TestCorrectnessSharkParams2 = SharkParamsNP2;
using TestCorrectnessSharkParams3 = SharkParamsNP4;
using TestCorrectnessSharkParams4 = SharkParamsNP3;
using TestCorrectnessSharkParams5 = SharkParamsNP5;
