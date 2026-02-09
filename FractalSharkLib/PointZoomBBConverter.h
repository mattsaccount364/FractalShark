#pragma once

#include "HighPrecision.h"

class PointZoomBBConverter {
public:
    static constexpr auto factor = 2;

    PointZoomBBConverter();
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

    void SetPrecision(uint64_t precInBits);

    bool Degenerate() const;

private:
    HighPrecision m_MinX, m_MinY;
    HighPrecision m_MaxX, m_MaxY;
    HighPrecision m_PtX, m_PtY;
    HighPrecision m_ZoomFactor;

    std::string m_MinXStr, m_MinYStr;
    std::string m_MaxXStr, m_MaxYStr;
    std::string m_PtXStr, m_PtYStr;
    std::string m_ZoomFactorStr;
    std::string m_RadiusStr;
    std::string m_DeltaYStr;

    static constexpr bool m_Test = true;
};
