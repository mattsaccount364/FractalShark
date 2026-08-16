#pragma once

#include "HighPrecision.h"
#include "PointZoomBBConverter.h"

#include <cstdint>

namespace PrecisionCalculator {
uint64_t GetPrecision(const PointZoomBBConverter &converter, bool requiresReuse);

uint64_t GetPrecision(const HighPrecision &minX,
                      const HighPrecision &minY,
                      const HighPrecision &maxX,
                      const HighPrecision &maxY,
                      bool requiresReuse);

uint64_t GetPrecision(const HighPrecision &radiusX, const HighPrecision &radiusY, bool requiresReuse);

template <typename T> uint64_t GetPrecision(const T &radiusX, const T &radiusY, bool requiresReuse);
} // namespace PrecisionCalculator
