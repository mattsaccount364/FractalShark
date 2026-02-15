#include "stdafx.h"
#include "PointZoomBBConverter.h"

PointZoomBBConverter::PointZoomBBConverter()
    : m_MinX{}, m_MinY{}, m_MaxX{}, m_MaxY{}, m_PtX{}, m_PtY{}, m_ZoomFactor{}
{
}

PointZoomBBConverter::PointZoomBBConverter(HighPrecision ptX,
                                           HighPrecision ptY,
                                           HighPrecision zoomFactor)
    : m_PtX(ptX), m_PtY(ptY), m_ZoomFactor(zoomFactor)
{

    m_MinX = ptX - (HighPrecision{Factor} / m_ZoomFactor);
    m_MinY = ptY - (HighPrecision{Factor} / m_ZoomFactor);
    m_MaxX = ptX + (HighPrecision{Factor} / m_ZoomFactor);
    m_MaxY = ptY + (HighPrecision{Factor} / m_ZoomFactor);
    m_Radius = (m_MaxY - m_MinY) / HighPrecision{2};

    auto deltaY = m_MaxY - m_MinY;
    SetDebugStrings(&deltaY);
}

PointZoomBBConverter::PointZoomBBConverter(HighPrecision minX,
                                           HighPrecision minY,
                                           HighPrecision maxX,
                                           HighPrecision maxY)
    : m_MinX{minX}, m_MinY{minY}, m_MaxX{maxX}, m_MaxY{maxY}, m_PtX{(minX + maxX) / HighPrecision(2)},
      m_PtY{(minY + maxY) / HighPrecision(2)}, m_Radius{(maxY - minY) / HighPrecision(2)}
{
    auto deltaY = m_MaxY - m_MinY;

    SetDebugStrings(&deltaY);

    if (/*deltaX == 0 || */ deltaY == HighPrecision{0}) {
        m_ZoomFactor = HighPrecision{1};

        SetDebugStrings(&deltaY);

        return;
    }

    // auto zf1 = HighPrecision{ Factor } / deltaX * 4;
    auto zf2 = HighPrecision{Factor} / deltaY * HighPrecision{2};
    // auto zf3 = HighPrecision{ Factor } / (m_MaxX - m_PtX) * 4;
    // auto zf4 = HighPrecision{ Factor } / (m_MaxY - m_PtY) * 4;
    // m_ZoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
    // m_ZoomFactor = std::min(zf1, zf2);
    m_ZoomFactor = zf2;

    SetDebugStrings(&deltaY);
}

const HighPrecision &
PointZoomBBConverter::GetMinX() const
{
    return m_MinX;
}

const HighPrecision &
PointZoomBBConverter::GetMinY() const
{
    return m_MinY;
}

const HighPrecision &
PointZoomBBConverter::GetMaxX() const
{
    return m_MaxX;
}

const HighPrecision &
PointZoomBBConverter::GetMaxY() const
{
    return m_MaxY;
}

const HighPrecision &
PointZoomBBConverter::GetPtX() const
{
    return m_PtX;
}

const HighPrecision &
PointZoomBBConverter::GetPtY() const
{
    return m_PtY;
}

const HighPrecision &
PointZoomBBConverter::GetZoomFactor() const
{
    return m_ZoomFactor;
}

const HighPrecision &
PointZoomBBConverter::GetRadius() const
{
    return m_Radius;
}

void
PointZoomBBConverter::SetPrecision(uint64_t precInBits)
{
    HighPrecision::defaultPrecisionInBits(precInBits);

    m_MinX.precisionInBits(precInBits);
    m_MinY.precisionInBits(precInBits);
    m_MaxX.precisionInBits(precInBits);
    m_MaxY.precisionInBits(precInBits);
    m_PtX.precisionInBits(precInBits);
    m_PtY.precisionInBits(precInBits);
    m_ZoomFactor.precisionInBits(precInBits);

    SetDebugStrings();
}

void
PointZoomBBConverter::SetDebugStrings(const HighPrecision *deltaY)
{
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
        if (m_Radius.precisionInBits() < 1000)
            m_RadiusStr = m_Radius.str();
        if (deltaY != nullptr && deltaY->precisionInBits() < 1000)
            m_DeltaYStr = deltaY->str();
    }
}

bool
PointZoomBBConverter::Degenerate() const
{
    if (m_MinX == m_MaxX || m_MinY == m_MaxY) {
        return true;
    }

    return false;
}

PointZoomBBConverter
PointZoomBBConverter::ZoomedAtCenter(double scale) const
{
    double divisor = 1.0 / (1.0 + 2.0 * scale);
    PointZoomBBConverter out = *this;
    out.ZoomDivisor(divisor);
    return out;
}

PointZoomBBConverter
PointZoomBBConverter::ZoomedRecentered(const HighPrecision &calcX,
                                       const HighPrecision &calcY,
                                       double scale) const
{
    // Recenter bounding box on (calcX, calcY), preserving current extents,
    // then apply ZoomedAtCenter.
    const auto prec = m_PtX.precisionInBits();

    HighPrecision width{HighPrecision::SetPrecision::True, prec};
    HighPrecision height{HighPrecision::SetPrecision::True, prec};
    HighPrecision halfW{HighPrecision::SetPrecision::True, prec};
    HighPrecision halfH{HighPrecision::SetPrecision::True, prec};

    width.subFrom(m_MaxX, m_MinX);
    height.subFrom(m_MaxY, m_MinY);

    // halfW = width / 2,  halfH = height / 2
    halfW.divFrom_ui(width, 2);
    halfH.divFrom_ui(height, 2);

    // Reuse width/height as the four bounding-box corners
    HighPrecision newMinX{HighPrecision::SetPrecision::True, prec};
    HighPrecision newMinY{HighPrecision::SetPrecision::True, prec};
    HighPrecision newMaxX{HighPrecision::SetPrecision::True, prec};
    HighPrecision newMaxY{HighPrecision::SetPrecision::True, prec};

    newMinX.subFrom(calcX, halfW);
    newMinY.subFrom(calcY, halfH);
    newMaxX.addFrom(calcX, halfW);
    newMaxY.addFrom(calcY, halfH);

    PointZoomBBConverter centered{
        std::move(newMinX), std::move(newMinY), std::move(newMaxX), std::move(newMaxY)};
    return centered.ZoomedAtCenter(scale);
}

PointZoomBBConverter
PointZoomBBConverter::ZoomedTowardPoint(const HighPrecision &calcX,
                                        const HighPrecision &calcY,
                                        double scale) const
{
    // Asymmetric weighted zoom toward (calcX, calcY) without recentering.
    // Edges expand/contract proportionally to their distance from the point.
    const auto prec = m_PtX.precisionInBits();

    HighPrecision width{HighPrecision::SetPrecision::True, prec};
    HighPrecision height{HighPrecision::SetPrecision::True, prec};
    HighPrecision tmp{HighPrecision::SetPrecision::True, prec};

    width.subFrom(m_MaxX, m_MinX);  // width = m_MaxX - m_MinX
    height.subFrom(m_MaxY, m_MinY); // height = m_MaxY - m_MinY

    // leftWeight = (calcX - m_MinX) / width
    HighPrecision leftWeight{HighPrecision::SetPrecision::True, prec};
    leftWeight.subFrom(calcX, m_MinX);
    leftWeight.divFrom(leftWeight, width);

    // rightWeight = 1 - leftWeight
    HighPrecision rightWeight{HighPrecision::SetPrecision::True, prec};
    {
        HighPrecision one{1};
        rightWeight.subFrom(one, leftWeight);
    }

    // topWeight = (calcY - m_MinY) / height
    HighPrecision topWeight{HighPrecision::SetPrecision::True, prec};
    topWeight.subFrom(calcY, m_MinY);
    topWeight.divFrom(topWeight, height);

    // bottomWeight = 1 - topWeight
    HighPrecision bottomWeight{HighPrecision::SetPrecision::True, prec};
    {
        HighPrecision one{1};
        bottomWeight.subFrom(one, topWeight);
    }

    const HighPrecision hf{scale};

    // newMinX = m_MinX - width * leftWeight * hf
    HighPrecision newMinX{HighPrecision::SetPrecision::True, prec};
    tmp.mulFrom(width, leftWeight);
    tmp.mulFrom(tmp, hf);
    newMinX.subFrom(m_MinX, tmp);

    // newMinY = m_MinY - height * topWeight * hf
    HighPrecision newMinY{HighPrecision::SetPrecision::True, prec};
    tmp.mulFrom(height, topWeight);
    tmp.mulFrom(tmp, hf);
    newMinY.subFrom(m_MinY, tmp);

    // newMaxX = m_MaxX + width * rightWeight * hf
    HighPrecision newMaxX{HighPrecision::SetPrecision::True, prec};
    tmp.mulFrom(width, rightWeight);
    tmp.mulFrom(tmp, hf);
    newMaxX.addFrom(m_MaxX, tmp);

    // newMaxY = m_MaxY + height * bottomWeight * hf
    HighPrecision newMaxY{HighPrecision::SetPrecision::True, prec};
    tmp.mulFrom(height, bottomWeight);
    tmp.mulFrom(tmp, hf);
    newMaxY.addFrom(m_MaxY, tmp);

    return PointZoomBBConverter{
        std::move(newMinX), std::move(newMinY), std::move(newMaxX), std::move(newMaxY)};
}

void
PointZoomBBConverter::SquareAspectRatio(size_t scrnWidth, size_t scrnHeight)
{
    if (scrnWidth == 0 || scrnHeight == 0) {
        return;
    }

    const auto prec = m_PtX.precisionInBits();

    HighPrecision ratio{HighPrecision::SetPrecision::True, prec};
    HighPrecision mwidth{HighPrecision::SetPrecision::True, prec};
    HighPrecision height{HighPrecision::SetPrecision::True, prec};
    HighPrecision tmp{HighPrecision::SetPrecision::True, prec};

    // ratio = scrnWidth / scrnHeight
    {
        HighPrecision w{static_cast<uint64_t>(scrnWidth)};
        HighPrecision h{static_cast<uint64_t>(scrnHeight)};
        ratio.divFrom(w, h);
    }

    // mwidth = (maxX - minX) / ratio
    mwidth.subFrom(m_MaxX, m_MinX);
    mwidth.divFrom(mwidth, ratio);

    // height = maxY - minY
    height.subFrom(m_MaxY, m_MinY);

    if (height > mwidth) {
        // diff = height - mwidth
        // adjust = ratio * diff / 2
        tmp.subFrom(height, mwidth);
        tmp.mulFrom(ratio, tmp);
        tmp.divFrom_ui(tmp, 2);
        m_MinX -= tmp;
        m_MaxX += tmp;
    } else if (height < mwidth) {
        // adjust = (mwidth - height) / 2
        tmp.subFrom(mwidth, height);
        tmp.divFrom_ui(tmp, 2);
        m_MinY -= tmp;
        m_MaxY += tmp;
    }

    // Recompute derived members from adjusted bounds
    m_PtX.addFrom(m_MinX, m_MaxX);
    m_PtX.divFrom_ui(m_PtX, 2);

    m_PtY.addFrom(m_MinY, m_MaxY);
    m_PtY.divFrom_ui(m_PtY, 2);

    m_Radius.subFrom(m_MaxY, m_MinY);
    m_Radius.divFrom_ui(m_Radius, 2);

    // Recompute zoom Factor: Factor * 2 / deltaY
    HighPrecision deltaY{HighPrecision::SetPrecision::True, prec};
    deltaY.subFrom(m_MaxY, m_MinY);

    if (deltaY == HighPrecision{0}) {
        m_ZoomFactor = HighPrecision{1};
    } else {
        m_ZoomFactor.divFrom(HighPrecision{Factor}, deltaY);
        m_ZoomFactor.mulFrom_ui(m_ZoomFactor, 2);
    }

    SetDebugStrings(&deltaY);
}

HighPrecision
PointZoomBBConverter::XFromScreenToCalc(HighPrecision x,
                                        size_t scrnWidth,
                                        size_t antialiasing) const
{
    HighPrecision aa(antialiasing);
    HighPrecision highWidth(scrnWidth);
    HighPrecision OriginX{highWidth * aa / (m_MaxX - m_MinX) * -m_MinX};
    return HighPrecision{(x - OriginX) * (m_MaxX - m_MinX) / (highWidth * aa)};
}

HighPrecision
PointZoomBBConverter::YFromScreenToCalc(HighPrecision y,
                                        size_t scrnHeight,
                                        size_t antialiasing) const
{
    HighPrecision aa(antialiasing);
    HighPrecision highHeight(scrnHeight);
    HighPrecision OriginY = highHeight * aa / (m_MaxY - m_MinY) * m_MaxY;
    return HighPrecision{-(y - OriginY) * (m_MaxY - m_MinY) / (highHeight * aa)};
}

HighPrecision
PointZoomBBConverter::XFromCalcToScreen(HighPrecision x, size_t scrnWidth) const
{
    HighPrecision highWidth(scrnWidth);
    return HighPrecision{(x - m_MinX) * (highWidth / (m_MaxX - m_MinX))};
}

HighPrecision
PointZoomBBConverter::YFromCalcToScreen(HighPrecision y, size_t scrnHeight) const
{
    HighPrecision highHeight(scrnHeight);
    return HighPrecision{highHeight - (y - m_MinY) * highHeight / (m_MaxY - m_MinY)};
}

PointZoomBBConverter
PointZoomBBConverter::Recentered(const HighPrecision &calcX,
                                 const HighPrecision &calcY) const
{
    const HighPrecision width = m_MaxX - m_MinX;
    const HighPrecision height = m_MaxY - m_MinY;
    const auto two = HighPrecision{2};

    return PointZoomBBConverter{calcX - width / two,
                                calcY - height / two,
                                calcX + width / two,
                                calcY + height / two};
}

HighPrecision
PointZoomBBConverter::GetDeltaX(size_t scrnWidth, size_t antialiasing) const
{
    return (m_MaxX - m_MinX) / (HighPrecision)(scrnWidth * antialiasing);
}

HighPrecision
PointZoomBBConverter::GetDeltaY(size_t scrnHeight, size_t antialiasing) const
{
    return (m_MaxY - m_MinY) / (HighPrecision)(scrnHeight * antialiasing);
}

void
PointZoomBBConverter::ZoomInPlace(double scale)
{
    // Old-style additive convention:
    // scale=0.3 expands each edge by 30% (zoom out)
    // scale=-0.3 shrinks each edge by 30% (zoom in)
    double divisor = 1.0 / (1.0 + 2.0 * scale);
    ZoomDivisor(divisor);
}

void
PointZoomBBConverter::ZoomDivisor(double divisor)
{
    if (!(divisor > 0.0) || Degenerate()) {
        return;
    }

    const auto prec = m_PtX.precisionInBits();
    const HighPrecision hf{divisor};

    // Scratch variables pre-allocated at working precision
    HighPrecision halfX{HighPrecision::SetPrecision::True, prec};
    HighPrecision halfY{HighPrecision::SetPrecision::True, prec};

    // halfX = (m_MaxX - m_MinX) / 2
    halfX.subFrom(m_MaxX, m_MinX);
    halfX.divFrom_ui(halfX, 2);

    // halfY = (m_MaxY - m_MinY) / 2
    halfY.subFrom(m_MaxY, m_MinY);
    halfY.divFrom_ui(halfY, 2);

    // newHalf = half / hf  (divisor > 1 => smaller box => zoom in)
    // Reuse halfX/halfY as newHalfX/newHalfY
    halfX.divFrom(halfX, hf);
    halfY.divFrom(halfY, hf);

    // Scale about current point
    m_MinX.subFrom(m_PtX, halfX);
    m_MaxX.addFrom(m_PtX, halfX);
    m_MinY.subFrom(m_PtY, halfY);
    m_MaxY.addFrom(m_PtY, halfY);

    m_Radius.setFrom(halfY);

    // zoomFactor ∝ 1/deltaY, deltaY shrinks by Factor => zoomFactor grows
    m_ZoomFactor *= hf;

    if constexpr (m_Test) {
        HighPrecision deltaY{HighPrecision::SetPrecision::True, prec};
        deltaY.subFrom(m_MaxY, m_MinY);
        SetDebugStrings(&deltaY);
    }
}
