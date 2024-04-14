#pragma once

#include "HighPrecision.h"

namespace PrecisionCalculator {
    uint64_t GetPrecision(
        const HighPrecision& minX,
        const HighPrecision& minY,
        const HighPrecision& maxX,
        const HighPrecision& maxY,
        bool RequiresReuse);
} // namespace PrecisionCalculator
