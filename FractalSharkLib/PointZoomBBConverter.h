#pragma once

#include "HighPrecision.h"

class PointZoomBBConverter {
public:
    static constexpr auto Factor = 2;

    enum class TestMode { Enabled, Disabled };

    PointZoomBBConverter(TestMode testMode);
    PointZoomBBConverter(
        HighPrecision ptX,
        HighPrecision ptY,
        HighPrecision zoomFactor,
        TestMode testMode);

    PointZoomBBConverter(
        HighPrecision minX,
        HighPrecision minY,
        HighPrecision maxX,
        HighPrecision maxY,
        TestMode testMode);

    const HighPrecision &GetMinX() const;
    const HighPrecision &GetMinY() const;
    const HighPrecision &GetMaxX() const;
    const HighPrecision &GetMaxY() const;
    const HighPrecision &GetPtX() const;
    const HighPrecision &GetPtY() const;
    const HighPrecision &GetZoomFactor() const;
    const HighPrecision &GetRadius() const;

    void SetPrecision(uint64_t precInBits);

    bool Degenerate() const;
    PointZoomBBConverter ZoomedAtCenter(double scale) const;
    PointZoomBBConverter ZoomedRecentered(const HighPrecision &calcX,
                                          const HighPrecision &calcY,
                                          double scale) const;
    PointZoomBBConverter ZoomedTowardPoint(const HighPrecision &calcX,
                                           const HighPrecision &calcY,
                                           double scale) const;
    void ZoomInPlace(double scale);
    void SquareAspectRatio(size_t scrnWidth, size_t scrnHeight);

    // Coordinate conversion: screen pixels ↔ fractal complex plane
    HighPrecision XFromScreenToCalc(HighPrecision x,
                                    size_t scrnWidth,
                                    size_t antialiasing) const;
    HighPrecision YFromScreenToCalc(HighPrecision y,
                                    size_t scrnHeight,
                                    size_t antialiasing) const;
    HighPrecision XFromCalcToScreen(HighPrecision x, size_t scrnWidth) const;
    HighPrecision YFromCalcToScreen(HighPrecision y, size_t scrnHeight) const;

    // Returns a new converter centered on (calcX, calcY) with same extents
    PointZoomBBConverter Recentered(const HighPrecision &calcX,
                                    const HighPrecision &calcY) const;

    // Per-pixel stepping values for GPU rendering
    HighPrecision GetDeltaX(size_t scrnWidth, size_t antialiasing) const;
    HighPrecision GetDeltaY(size_t scrnHeight, size_t antialiasing) const;

private:
    void ZoomDivisor(double divisor);
    void SetDebugStrings(const HighPrecision* deltaY = nullptr);
    HighPrecision m_MinX, m_MinY;
    HighPrecision m_MaxX, m_MaxY;
    HighPrecision m_PtX, m_PtY;
    HighPrecision m_ZoomFactor;
    HighPrecision m_Radius;

    std::string m_MinXStr, m_MinYStr;
    std::string m_MaxXStr, m_MaxYStr;
    std::string m_PtXStr, m_PtYStr;
    std::string m_ZoomFactorStr;
    std::string m_RadiusStr;
    std::string m_DeltaYStr;

    TestMode m_Test;
};
