#include "stdafx.h"
#include "PointZoomBBConverter.h"

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision ptX,
    HighPrecision ptY,
    HighPrecision zoomFactor)
    : ptX(ptX),
    ptY(ptY),
    zoomFactor(zoomFactor) {
    minX = ptX - (HighPrecision{ factor } / zoomFactor);
    // minX + (HighPrecision{ factor } / zoomFactor) = ptX;
    // (HighPrecision{ factor } / zoomFactor) = ptX - minX;
    // HighPrecision{ factor } = (ptX - minX) * zoomFactor;
    // HighPrecision{ factor } / (ptX - minX) = zoomFactor;

    minY = ptY - (HighPrecision{ factor } / zoomFactor);

    maxX = ptX + (HighPrecision{ factor } / zoomFactor);
    // maxX - ptX = (HighPrecision{ factor } / zoomFactor);
    // zoomFactor * (maxX - ptX) = (HighPrecision{ factor });
    // zoomFactor = (HighPrecision{ factor }) / (maxX - ptX);

    maxY = ptY + (HighPrecision{ factor } / zoomFactor);
}

PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision minX,
    HighPrecision minY,
    HighPrecision maxX,
    HighPrecision maxY) :
    minX(minX),
    minY(minY),
    maxX(maxX),
    maxY(maxY) {
    ptX = (minX + maxX) / HighPrecision(2);
    ptY = (minY + maxY) / HighPrecision(2);

    auto zf1 = HighPrecision{ factor } / (ptX - minX);
    auto zf2 = HighPrecision{ factor } / (ptY - minY);
    auto zf3 = HighPrecision{ factor } / (maxX - ptX);
    auto zf4 = HighPrecision{ factor } / (maxY - ptY);
    zoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
}