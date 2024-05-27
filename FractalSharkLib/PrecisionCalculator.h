#pragma once

#include "HighPrecision.h"

namespace PrecisionCalculator {
    uint64_t GetPrecision(
        const HighPrecision &minX,
        const HighPrecision &minY,
        const HighPrecision &maxX,
        const HighPrecision &maxY,
        bool RequiresReuse);

    uint64_t GetPrecision(
        const HighPrecision &radiuxX,
        const HighPrecision &radiusY,
        bool RequiresReuse);

    template<typename T>
    uint64_t GetPrecision(
        const T &radiuxX,
        const T &radiusY,
        bool RequiresReuse);
} // namespace PrecisionCalculator
