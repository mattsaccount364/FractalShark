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

    //double test_minX = (double)m_minX;
    //double test_minY = (double)m_minY;
    //double test_maxX = (double)m_maxX;
    //double test_maxY = (double)m_maxY;
    //double test_ptX = (double)m_ptX;
    //double test_ptY = (double)m_ptY;
    //double test_zoomFactor = (double)m_zoomFactor;

    // std::string all_test_nums = std::to_string(test_minX) + " " + std::to_string(test_minY) + " " + std::to_string(test_maxX) + " " + std::to_string(test_maxY) + " " + std::to_string(test_ptX) + " " + std::to_string(test_ptY) + " " + std::to_string(test_zoomFactor);
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

    //auto deltaX = m_maxX - m_minX;
    auto deltaY = m_maxY - m_minY;

    if (/*deltaX == 0 || */deltaY == 0) {
        m_zoomFactor = HighPrecision{ 1 };
        return;
    }

    //auto zf1 = HighPrecision{ factor } / deltaX * 4;
    auto zf2 = HighPrecision{ factor } / deltaY * 2;
    //auto zf3 = HighPrecision{ factor } / (m_maxX - m_ptX) * 4;
    //auto zf4 = HighPrecision{ factor } / (m_maxY - m_ptY) * 4;
    //m_zoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
    //m_zoomFactor = std::min(zf1, zf2);
    m_zoomFactor = zf2;

    //double test_zf1 = (double)zf1;
    //double test_zf2 = (double)zf2;
    //double test_zf3 = (double)zf3;
    //double test_zf4 = (double)zf4;
    //double test_zf = (double)m_zoomFactor;

    //std::string all_test_nums = std::to_string(test_zf1) + " " + std::to_string(test_zf2) + " " + std::to_string(test_zf3) + " " + std::to_string(test_zf4) + " " + std::to_string(test_zf);
}

const HighPrecision &PointZoomBBConverter::GetMinX() const {
    return m_minX;
}

const HighPrecision &PointZoomBBConverter::GetMinY() const {
    return m_minY;
}

const HighPrecision &PointZoomBBConverter::GetMaxX() const {
    return m_maxX;
}

const HighPrecision &PointZoomBBConverter::GetMaxY() const {
    return m_maxY;
}

const HighPrecision &PointZoomBBConverter::GetPtX() const {
    return m_ptX;
}

const HighPrecision &PointZoomBBConverter::GetPtY() const {
    return m_ptY;
}

const HighPrecision &PointZoomBBConverter::GetZoomFactor() const {
    return m_zoomFactor;
}