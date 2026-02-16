#pragma once

#include "ItersMemoryContainer.h"
#include "PointZoomBBConverter.h"

// Per-call context for CalcFractal, bundling mutable state that
// was previously swapped in/out on Fractal (m_Ptz, m_CurIters).
// Each worker thread constructs its own CalcContext so that
// multiple CalcFractal calls can execute in parallel.
struct CalcContext {
    PointZoomBBConverter Ptz;
    ItersMemoryContainer &ItersMemory;
};
