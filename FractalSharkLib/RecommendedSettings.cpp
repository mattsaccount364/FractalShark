#include "stdafx.h"
#include "RecommendedSettings.h"
#include "PointZoomBBConverter.h"
#include "Fractal.h"
#include "Exceptions.h"

RecommendedSettings::RecommendedSettings()
    : m_PointZoomBBConverter{},
    RenderAlg{},
    IterType{},
    NumIterations{} {
}

RecommendedSettings::RecommendedSettings(const RecommendedSettings &other)
    : m_PointZoomBBConverter{ other.m_PointZoomBBConverter ? std::make_unique<PointZoomBBConverter>(*other.m_PointZoomBBConverter) : nullptr },
    RenderAlg{ other.RenderAlg },
    IterType{ other.IterType },
    NumIterations{ other.NumIterations } {
}

RecommendedSettings &RecommendedSettings::operator=(const RecommendedSettings &other) {
    if (this != &other) {
        m_PointZoomBBConverter = other.m_PointZoomBBConverter ? std::make_unique<PointZoomBBConverter>(*other.m_PointZoomBBConverter) : nullptr;
        RenderAlg = other.RenderAlg;
        IterType = other.IterType;
        NumIterations = other.NumIterations;
    }

    return *this;
}

RecommendedSettings::RecommendedSettings(
    const HighPrecision &orbitX,
    const HighPrecision &orbitY,
    const HighPrecision &zoomFactor,
    RenderAlgorithm renderAlg,
    IterTypeFull numIterations) :
    m_PointZoomBBConverter{ std::make_unique<PointZoomBBConverter>(orbitX, orbitY, zoomFactor) },
    RenderAlg(renderAlg),
    NumIterations(numIterations) {

    if (NumIterations <= Fractal::GetMaxIterations<uint32_t>()) {
        IterType = IterTypeEnum::Bits32;
    } else {
        IterType = IterTypeEnum::Bits64;
    }
}

RecommendedSettings::RecommendedSettings(
    const HighPrecision &minX,
    const HighPrecision &minY,
    const HighPrecision &maxX,
    const HighPrecision &maxY,
    RenderAlgorithm renderAlg,
    IterTypeFull numIterations) :
    m_PointZoomBBConverter{ std::make_unique<PointZoomBBConverter>(minX, minY, maxX, maxY) },
    RenderAlg(renderAlg),
    NumIterations(numIterations) {

    if (NumIterations <= Fractal::GetMaxIterations<uint32_t>()) {
        IterType = IterTypeEnum::Bits32;
    } else {
        IterType = IterTypeEnum::Bits64;
    }
}

const PointZoomBBConverter &RecommendedSettings::GetPointZoomBBConverter() const {
    if (!m_PointZoomBBConverter) {
        throw FractalSharkSeriousException("PointZoomBBConverter is not initialized");
    }
    return *m_PointZoomBBConverter;
}

RenderAlgorithm RecommendedSettings::GetRenderAlgorithm() const {
    return RenderAlg;
}

IterTypeEnum RecommendedSettings::GetIterType() const {
    return IterType;
}

IterTypeFull RecommendedSettings::GetNumIterations() const {
    return NumIterations;
}

void RecommendedSettings::SetRenderAlgorithm(RenderAlgorithm renderAlg) {
    RenderAlg = renderAlg;
}

void RecommendedSettings::OverrideIterType(IterTypeEnum iterType) {
    IterType = iterType;
}
