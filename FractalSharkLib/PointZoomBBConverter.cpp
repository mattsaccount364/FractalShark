#include "stdafx.h"
#include "PointZoomBBConverter.h"

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision ptX,
    HighPrecision ptY,
    HighPrecision zoomFactor)
    : m_ptX(ptX),
    m_ptY(ptY),
    m_zoomFactor(zoomFactor) {

    m_minX = ptX - (HighPrecision{ factor } / m_zoomFactor);
    m_minY = ptY - (HighPrecision{ factor } / m_zoomFactor);
    m_maxX = ptX + (HighPrecision{ factor } / m_zoomFactor);
    m_maxY = ptY + (HighPrecision{ factor } / m_zoomFactor);
}

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision minX,
    HighPrecision minY,
    HighPrecision maxX,
    HighPrecision maxY) :
    m_minX{ minX },
    m_minY{ minY },
    m_maxX{ maxX },
    m_maxY{ maxY },
    m_ptX{ (minX + maxX) / HighPrecision(2) },
    m_ptY{ (minY + maxY) / HighPrecision(2) } {

    auto deltaX = m_maxX - m_minX;
    auto deltaY = m_maxY - m_minY;

    if (deltaX == 0 || deltaY == 0) {
        m_zoomFactor = HighPrecision{ 1 };
        return;
    }

    auto zf1 = HighPrecision{ factor } / deltaX;
    auto zf2 = HighPrecision{ factor } / deltaY;
    auto zf3 = HighPrecision{ factor } / (m_maxX - m_ptX);
    auto zf4 = HighPrecision{ factor } / (m_maxY - m_ptY);
    m_zoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
}

const HighPrecision& PointZoomBBConverter::GetMinX() const {
    return m_minX;
}

const HighPrecision& PointZoomBBConverter::GetMinY() const {
    return m_minY;
}

const HighPrecision& PointZoomBBConverter::GetMaxX() const {
    return m_maxX;
}

const HighPrecision& PointZoomBBConverter::GetMaxY() const {
    return m_maxY;
}

const HighPrecision& PointZoomBBConverter::GetPtX() const {
    return m_ptX;
}

const HighPrecision& PointZoomBBConverter::GetPtY() const {
    return m_ptY;
}

const HighPrecision& PointZoomBBConverter::GetZoomFactor() const {
    return m_zoomFactor;
}