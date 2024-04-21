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

    const HighPrecision &GetMinX() const;
    const HighPrecision &GetMinY() const;
    const HighPrecision &GetMaxX() const;
    const HighPrecision &GetMaxY() const;
    const HighPrecision &GetPtX() const;
    const HighPrecision &GetPtY() const;
    const HighPrecision &GetZoomFactor() const;

private:
    HighPrecision m_minX, m_minY;
    HighPrecision m_maxX, m_maxY;
    HighPrecision m_ptX, m_ptY;
    HighPrecision m_zoomFactor;
};
