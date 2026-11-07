#pragma once

#include <vector>

namespace FractalShark {

// Coordinates are output pixels in OpenGL's bottom-left coordinate system.
struct RenderOverlayPoint {
    int X{};
    int Y{};

    bool operator==(const RenderOverlayPoint &) const = default;
};

struct RenderOverlayLine {
    RenderOverlayPoint Start;
    RenderOverlayPoint End;

    bool operator==(const RenderOverlayLine &) const = default;
};

// A frame owns all of its overlay geometry. Presentation must never follow pointers
// back to feature summaries, reference orbits, or the mutable high-precision view.
struct RenderOverlaySnapshot {
    std::vector<RenderOverlayPoint> ReferencePoints;
    std::vector<RenderOverlayLine> FeatureLines;
    std::vector<RenderOverlayPoint> FeatureCenters;

    bool operator==(const RenderOverlaySnapshot &) const = default;
};

} // namespace FractalShark
