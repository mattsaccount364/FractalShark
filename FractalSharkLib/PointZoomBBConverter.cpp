#include "stdafx.h"
#include "PointZoomBBConverter.h"

PointZoomBBConverter::PointZoomBBConverter()
    : m_MinX{},
    m_MinY{},
    m_MaxX{},
    m_MaxY{},
    m_PtX{},
    m_PtY{},
    m_ZoomFactor{} {
}

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision ptX,
    HighPrecision ptY,
    HighPrecision zoomFactor)
    : m_PtX(ptX),
    m_PtY(ptY),
    m_ZoomFactor(zoomFactor) {

    m_MinX = ptX - (HighPrecision{ factor } / m_ZoomFactor);
    m_MinY = ptY - (HighPrecision{ factor } / m_ZoomFactor);
    m_MaxX = ptX + (HighPrecision{ factor } / m_ZoomFactor);
    m_MaxY = ptY + (HighPrecision{ factor } / m_ZoomFactor);

    auto deltaY = m_MaxY - m_MinY;
    if constexpr (m_Test) {
        if (m_MinX.precisionInBits() < 1000)
            m_MinXStr = m_MinX.str();

        if (m_MinY.precisionInBits() < 1000)
            m_MinYStr = m_MinY.str();

        if (m_MaxX.precisionInBits() < 1000)
            m_MaxXStr = m_MaxX.str();

        if (m_MaxY.precisionInBits() < 1000)
            m_MaxYStr = m_MaxY.str();

        if (m_PtX.precisionInBits() < 1000)
            m_PtXStr = m_PtX.str();
            
        if (m_PtY.precisionInBits() < 1000)
            m_PtYStr = m_PtY.str();

        if (m_ZoomFactor.precisionInBits() < 1000)
            m_ZoomFactorStr = m_ZoomFactor.str();

        HighPrecision radius = (m_MaxY - m_MinY) / HighPrecision{2};

        if (radius.precisionInBits() < 1000)
            m_RadiusStr = radius.str();

        if (deltaY.precisionInBits() < 1000)
            m_DeltaYStr = deltaY.str();
    }
}

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision minX,
    HighPrecision minY,
    HighPrecision maxX,
    HighPrecision maxY) :
    m_MinX{ minX },
    m_MinY{ minY },
    m_MaxX{ maxX },
    m_MaxY{ maxY },
    m_PtX{ (minX + maxX) / HighPrecision(2) },
    m_PtY{ (minY + maxY) / HighPrecision(2) } {

    auto deltaY = m_MaxY - m_MinY;

    if constexpr (m_Test) {
        std::string minXStr;
        std::string maxXStr;
        std::string minYStr;
        std::string maxYStr;

        if (minX.precisionInBits() < 1000)
            minXStr = minX.str();
        if (maxX.precisionInBits() < 1000)
            maxXStr = maxX.str();
        if (minY.precisionInBits() < 1000)
            minYStr = minY.str();
        if (maxY.precisionInBits() < 1000)
            maxYStr = maxY.str();

        if (m_MinX.precisionInBits() < 1000)
            m_MinXStr = m_MinX.str();
        if (m_MaxX.precisionInBits() < 1000)
            m_MaxXStr = m_MaxX.str();
        if (m_MinY.precisionInBits() < 1000)
            m_MinYStr = m_MinY.str();
        if (m_MaxY.precisionInBits() < 1000)
            m_MaxYStr = m_MaxY.str();
        if (m_PtX.precisionInBits() < 1000)
            m_PtXStr = m_PtX.str();
        if (m_PtY.precisionInBits() < 1000)
            m_PtYStr = m_PtY.str();

        HighPrecision radius = (m_MaxY - m_MinY) / HighPrecision{2};
        if (radius.precisionInBits() < 1000)
            m_RadiusStr = radius.str();
        if (deltaY.precisionInBits() < 1000)
            m_DeltaYStr = deltaY.str();
    }

    if (/*deltaX == 0 || */deltaY == HighPrecision{ 0 }) {
        m_ZoomFactor = HighPrecision{ 1 };

        if constexpr (m_Test) {
            if (m_ZoomFactor.precisionInBits() < 1000)
                m_ZoomFactorStr = m_ZoomFactor.str();
        }

        return;
    }

    //auto zf1 = HighPrecision{ factor } / deltaX * 4;
    auto zf2 = HighPrecision{ factor } / deltaY * HighPrecision{ 2 };
    //auto zf3 = HighPrecision{ factor } / (m_MaxX - m_PtX) * 4;
    //auto zf4 = HighPrecision{ factor } / (m_MaxY - m_PtY) * 4;
    //m_ZoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
    //m_ZoomFactor = std::min(zf1, zf2);
    m_ZoomFactor = zf2;

    if constexpr (m_Test) {
        if (m_ZoomFactor.precisionInBits() < 1000)
            m_ZoomFactorStr = m_ZoomFactor.str();
    }
}

const HighPrecision &PointZoomBBConverter::GetMinX() const {
    return m_MinX;
}

const HighPrecision &PointZoomBBConverter::GetMinY() const {
    return m_MinY;
}

const HighPrecision &PointZoomBBConverter::GetMaxX() const {
    return m_MaxX;
}

const HighPrecision &PointZoomBBConverter::GetMaxY() const {
    return m_MaxY;
}

const HighPrecision &PointZoomBBConverter::GetPtX() const {
    return m_PtX;
}

const HighPrecision &PointZoomBBConverter::GetPtY() const {
    return m_PtY;
}

const HighPrecision &PointZoomBBConverter::GetZoomFactor() const {
    return m_ZoomFactor;
}

void PointZoomBBConverter::SetPrecision(uint64_t precInBits) {
    HighPrecision::defaultPrecisionInBits(precInBits);

    m_MinX.precisionInBits(precInBits);
    m_MinY.precisionInBits(precInBits);
    m_MaxX.precisionInBits(precInBits);
    m_MaxY.precisionInBits(precInBits);
    m_PtX.precisionInBits(precInBits);
    m_PtY.precisionInBits(precInBits);
    m_ZoomFactor.precisionInBits(precInBits);

    if constexpr (m_Test) {
        if (m_MinX.precisionInBits() < 1000)
            m_MinXStr = m_MinX.str();

        if (m_MinY.precisionInBits() < 1000)
            m_MinYStr = m_MinY.str();

        if (m_MaxX.precisionInBits() < 1000)
            m_MaxXStr = m_MaxX.str();

        if (m_MaxY.precisionInBits() < 1000)
            m_MaxYStr = m_MaxY.str();

        if (m_PtX.precisionInBits() < 1000)
            m_PtXStr = m_PtX.str();

        if (m_PtY.precisionInBits() < 1000)
            m_PtYStr = m_PtY.str();

        if (m_ZoomFactor.precisionInBits() < 1000)
            m_ZoomFactorStr = m_ZoomFactor.str();

        HighPrecision radius = (m_MaxY - m_MinY) / HighPrecision{2};
        if (radius.precisionInBits() < 1000)
            m_RadiusStr = ((m_MaxY - m_MinY) / HighPrecision{2}).str();
    }
}

bool PointZoomBBConverter::Degenerate() const {
    if (m_MinX == m_MaxX || m_MinY == m_MaxY) {
        return true;
    }

    return false;
}