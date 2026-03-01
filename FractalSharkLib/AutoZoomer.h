#pragma once

#include "Fractal.h"

class FeatureSummary;

class AutoZoomer {
public:
    explicit AutoZoomer(Fractal &fractal);

    template <Fractal::AutoZoomHeuristic h>
    void Run();

private:
    struct FeatureZoomSetup {
        HighPrecision GuessX;
        HighPrecision GuessY;
        std::vector<PointZoomBBConverter> ZoomSteps;
        std::vector<IterTypeFull> IterCounts;
        int64_t TotalSteps = 0;
        bool ShouldInterpolateIters = false;
        bool Failed = false;
    };

    void SetupFeatureZoom(Fractal &f, FeatureZoomSetup &out);

    Fractal &m_Fractal;
};
