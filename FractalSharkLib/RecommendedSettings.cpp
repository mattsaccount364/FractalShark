#include "stdafx.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "PointZoomBBConverter.h"
#include "RecommendedSettings.h"

RecommendedSettings::RecommendedSettings()
    : m_PointZoomBBConverter{}, RenderAlg{}, IterType{}, NumIterations{}
{
}

RecommendedSettings::RecommendedSettings(const RecommendedSettings &other)
    : m_PointZoomBBConverter{other.m_PointZoomBBConverter
                                 ? std::make_unique<PointZoomBBConverter>(*other.m_PointZoomBBConverter)
                                 : nullptr},
      RenderAlg{other.RenderAlg}, IterType{other.IterType}, NumIterations{other.NumIterations}
{
}

RecommendedSettings &
RecommendedSettings::operator=(const RecommendedSettings &other)
{
    if (this != &other) {
        m_PointZoomBBConverter =
            other.m_PointZoomBBConverter
                ? std::make_unique<PointZoomBBConverter>(*other.m_PointZoomBBConverter)
                : nullptr;
        RenderAlg = other.RenderAlg;
        IterType = other.IterType;
        NumIterations = other.NumIterations;
    }

    return *this;
}

RecommendedSettings::RecommendedSettings(uint64_t precisionInBits,
                                         const HighPrecision &orbitX,
                                         const HighPrecision &orbitY,
                                         const HighPrecision &zoomFactor,
                                         RenderAlgorithm renderAlg,
                                         IterTypeFull numIterations)
    : PrecisionInBits(precisionInBits),
      m_PointZoomBBConverter{std::make_unique<PointZoomBBConverter>(orbitX, orbitY, zoomFactor, PointZoomBBConverter::TestMode::Enabled)},
      RenderAlg(renderAlg), NumIterations(numIterations)
{

    if (NumIterations <= Fractal::GetMaxIterations<uint32_t>()) {
        IterType = IterTypeEnum::Bits32;
    } else {
        IterType = IterTypeEnum::Bits64;
    }
}

RecommendedSettings::RecommendedSettings(uint64_t precisionInBits,
                                         const HighPrecision &minX,
                                         const HighPrecision &minY,
                                         const HighPrecision &maxX,
                                         const HighPrecision &maxY,
                                         RenderAlgorithm renderAlg,
                                         IterTypeFull numIterations)
    : PrecisionInBits{precisionInBits},
      m_PointZoomBBConverter{std::make_unique<PointZoomBBConverter>(minX, minY, maxX, maxY, PointZoomBBConverter::TestMode::Enabled)},
      RenderAlg(renderAlg), NumIterations(numIterations)
{

    if (NumIterations <= Fractal::GetMaxIterations<uint32_t>()) {
        IterType = IterTypeEnum::Bits32;
    } else {
        IterType = IterTypeEnum::Bits64;
    }
}

uint64_t
RecommendedSettings::GetPrecisionInBits() const
{
    return PrecisionInBits;
}

const PointZoomBBConverter &
RecommendedSettings::GetPointZoomBBConverter() const
{
    if (!m_PointZoomBBConverter) {
        throw FractalSharkSeriousException("PointZoomBBConverter is not initialized");
    }
    return *m_PointZoomBBConverter;
}

RenderAlgorithm
RecommendedSettings::GetRenderAlgorithm() const
{
    return RenderAlg;
}

IterTypeEnum
RecommendedSettings::GetIterType() const
{
    return IterType;
}

IterTypeFull
RecommendedSettings::GetNumIterations() const
{
    return NumIterations;
}

void
RecommendedSettings::SetRenderAlgorithm(RenderAlgorithm renderAlg)
{
    RenderAlg = renderAlg;
}

void
RecommendedSettings::OverrideIterType(IterTypeEnum iterType)
{
    IterType = iterType;
}
