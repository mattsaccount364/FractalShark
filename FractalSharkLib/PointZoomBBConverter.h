#pragma once

#include "HighPrecision.h"

struct PointZoomBBConverter {
    static constexpr auto factor = 2;

    PointZoomBBConverter(
        HighPrecision ptX,
        HighPrecision ptY,
        HighPrecision zoomFactor);

    PointZoomBBConverter(
        HighPrecision minX,
        HighPrecision minY,
        HighPrecision maxX,
        HighPrecision maxY);

    HighPrecision ptX, ptY;
    HighPrecision zoomFactor;

    HighPrecision minX, minY;
    HighPrecision maxX, maxY;
};
