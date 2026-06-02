#pragma once

#include "Fractal.h"

class FeatureSummary;

class AutoZoomer {
public:
    explicit AutoZoomer(Fractal &fractal);

    template <Fractal::AutoZoomHeuristic h> void Run();
    void RunFeatureAtPoint(int clientX, int clientY);

private:
    struct FeatureZoomStep {
        PointZoomBBConverter Ptz;
        IterTypeFull NumIterations;
    };

    struct FeatureZoomSetup {
        std::vector<FeatureZoomStep> Steps;
        bool Failed = false;
    };

    static void ApplyFeatureZoomStep(Fractal &f, const FeatureZoomStep &step);
    void SetupFeatureZoom(Fractal &f, FeatureZoomSetup &out, int clientX, int clientY);
    bool RunFeatureZoomPipeline(const std::vector<FeatureZoomStep> &steps);
    void RestoreLastPresentedView();

    Fractal &m_Fractal;
};
