#pragma once

class RenderAlgorithm;
enum class IterTypeEnum;

#include "HighPrecision.h"
#include "RenderAlgorithm.h"

class PointZoomBBConverter;

struct RecommendedSettings {
    // Default - initializes to empty.
    RecommendedSettings();

    // This constructor is used when the user specifies a point and zoom factor.
    RecommendedSettings(uint64_t PrecisionInBits,
                        const HighPrecision &orbitX,
                        const HighPrecision &orbitY,
                        const HighPrecision &zoomFactor,
                        RenderAlgorithm renderAlg,
                        IterTypeFull numIterations);

    // This constructor is used when the user specifies a bounding box.
    RecommendedSettings(uint64_t PrecisionInBits,
                        const HighPrecision &minX,
                        const HighPrecision &minY,
                        const HighPrecision &maxX,
                        const HighPrecision &maxY,
                        RenderAlgorithm renderAlg,
                        IterTypeFull numIterations);

    // Copy constructor and assignment operator.
    RecommendedSettings(const RecommendedSettings &);
    RecommendedSettings &operator=(const RecommendedSettings &);

    uint64_t GetPrecisionInBits() const;

    // Returns the point/zoom/bounding box.
    const PointZoomBBConverter &GetPointZoomBBConverter() const;

    // Returns the render algorithm to use.
    RenderAlgorithm GetRenderAlgorithm() const;

    // Returns the iteration type to use.
    IterTypeEnum GetIterType() const;

    // Returns the number of iterations to use.
    IterTypeFull GetNumIterations() const;

    // Use this to change the render algorithm after construction.
    void SetRenderAlgorithm(RenderAlgorithm renderAlg);

    // Use this to force the underlying iteration type to be
    // something other than the default.  The default is otherwise
    // established at construction time contingent on iteration
    // count.
    void OverrideIterType(IterTypeEnum iterType);

private:
    uint64_t PrecisionInBits;
    std::unique_ptr<PointZoomBBConverter> m_PointZoomBBConverter;

    RenderAlgorithm RenderAlg;
    IterTypeEnum IterType;
    IterTypeFull NumIterations;
};
