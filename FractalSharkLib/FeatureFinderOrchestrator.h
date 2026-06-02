#pragma once

#include "FeatureFinderMode.h"
#include "HighPrecision.h"

#include <memory>
#include <vector>

class Fractal;
class FeatureSummary;

// Orchestrates periodic-point(feature) finding, storage, and zoom-to-feature
// operations.  The mathematical core lives in FeatureFinder<>; this class
// handles screen↔calc coordinate conversion, algorithm dispatch, perturbation
// result acquisition, scan grids, and feature lifecycle management.
class FeatureFinderOrchestrator {
public:
    explicit FeatureFinderOrchestrator(Fractal &fractal);

    // Find periodic points near screen pixel (scrnX, scrnY).
    // Clears any previously found features.  In scan mode (e.g.
    // DirectScan, PTScan, LAScan), searches a grid of screen locations.
    void TryFindPeriodicPoint(size_t scrnX, size_t scrnY, FeatureFinderMode mode);

    // Discard all found features and mark the window dirty for repaint.
    void ClearAllFoundFeatures();

    // Return the feature closest to a client-relative screen position,
    // or nullptr if no features have been found.
    FeatureSummary *ChooseClosestFeatureToScreenPoint(int clientX, int clientY) const;

    // Refine the feature (if still a candidate) and optionally zoom
    // to it at the given zoom factor.  Pass nullptr to refine only.
    bool ZoomToFoundFeature(FeatureSummary &feature, const HighPrecision *zoomFactor);

    // Choose the closest feature to a client-relative screen position and
    // zoom to it at the feature's computed intrinsic-radius zoom depth.
    bool ZoomToFoundFeature(int clientX, int clientY);

    // Resume NR refinement from a saved checkpoint file.
    // Creates a synthetic FeatureSummary and proceeds to Phase B + navigation.
    bool ResumeFromCheckpoint();

    NRInnerLoopBackend
    GetNRInnerLoopBackend() const
    {
        return m_NRInnerLoopBackend;
    }
    void
    SetNRInnerLoopBackend(NRInnerLoopBackend v)
    {
        m_NRInnerLoopBackend = v;
    }

    const std::vector<std::unique_ptr<FeatureSummary>> &
    GetFeatureSummaries() const
    {
        return m_FeatureSummaries;
    }

private:
    // Dispatch TryFindPeriodicPoint across the render-algorithm switch
    // for a given IterType (uint32_t or uint64_t).
    template <typename IterType>
    void TryFindPeriodicPointIterType(size_t scrnX, size_t scrnY, FeatureFinderMode mode);

    // Core implementation: creates a FeatureFinder, acquires perturbation
    // results / LA data as needed, and runs the search (single point or grid).
    template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
    void TryFindPeriodicPointTemplate(size_t scrnX, size_t scrnY, FeatureFinderMode mode);

    Fractal &m_Fractal;
    std::vector<std::unique_ptr<FeatureSummary>> m_FeatureSummaries;
    NRInnerLoopBackend m_NRInnerLoopBackend = NRInnerLoopBackend::CpuMT;
};
