#pragma once

#include "FeatureFinderMode.h"
#include "HighPrecision.h"

#include <memory>
#include <vector>

class Fractal;
class FeatureSummary;

class FeatureFinderOrchestrator {
public:
    explicit FeatureFinderOrchestrator(Fractal &fractal);

    void TryFindPeriodicPoint(size_t scrnX, size_t scrnY, FeatureFinderMode mode);
    void ClearAllFoundFeatures();
    FeatureSummary *ChooseClosestFeatureToMouse() const;
    bool ZoomToFoundFeature(FeatureSummary &feature, const HighPrecision *zoomFactor);
    bool ZoomToFoundFeature();

    const std::vector<std::unique_ptr<FeatureSummary>> &GetFeatureSummaries() const {
        return m_FeatureSummaries;
    }

private:
    template <typename IterType>
    void TryFindPeriodicPointIterType(size_t scrnX, size_t scrnY, FeatureFinderMode mode);

    template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
    void TryFindPeriodicPointTemplate(size_t scrnX, size_t scrnY, FeatureFinderMode mode);

    Fractal &m_Fractal;
    std::vector<std::unique_ptr<FeatureSummary>> m_FeatureSummaries;
};
