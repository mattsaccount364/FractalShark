#include "stdafx.h"

#include "SavedLocation.h"

#include "Fractal.h"
#include "PointZoomBBConverter.h"

#include <iomanip>
#include <limits>
#include <sstream>

namespace FractalShark {

SavedLocation
CaptureSavedLocation(const Fractal &fractal, bool scaleToMaximum)
{
    SavedLocation result;
    result.Width = fractal.GetRenderWidth();
    result.Height = fractal.GetRenderHeight();
    if (scaleToMaximum && result.Width != 0 && result.Height != 0) {
        constexpr size_t maximumDimension = 16384;
        if (result.Width > result.Height) {
            result.Height = maximumDimension * result.Height / result.Width;
            result.Width = maximumDimension;
        } else if (result.Height > result.Width) {
            result.Width = maximumDimension * result.Width / result.Height;
            result.Height = maximumDimension;
        }
    }

    result.MinX = fractal.GetMinX();
    result.MinY = fractal.GetMinY();
    result.MaxX = fractal.GetMaxX();
    result.MaxY = fractal.GetMaxY();
    result.NumIterations = fractal.GetNumIterations<IterTypeFull>();
    result.Antialiasing = fractal.GetGpuAntialiasing();
    result.Description = "FractalTrayDestination";
    return result;
}

std::string
SerializeSavedLocation(const SavedLocation &location)
{
    std::ostringstream output;
    output << location.Width << ' ' << location.Height << ' ';
    output << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
    output << location.MinX << ' ' << location.MinY << ' ' << location.MaxX << ' ' << location.MaxY
           << ' ' << location.NumIterations << ' ' << location.Antialiasing << ' '
           << location.Description;
    return output.str();
}

bool
ParseSavedLocation(std::istream &input, SavedLocation &location)
{
    SavedLocation parsed;
    if (!(input >> parsed.Width >> parsed.Height >> parsed.MinX >> parsed.MinY >> parsed.MaxX >>
          parsed.MaxY >> parsed.NumIterations >> parsed.Antialiasing)) {
        return false;
    }
    while (input.peek() == ' ' || input.peek() == '\t') {
        input.get();
    }
    std::getline(input, parsed.Description);
    if (!parsed.Description.empty() && parsed.Description.back() == '\r') {
        parsed.Description.pop_back();
    }
    location = std::move(parsed);
    return true;
}

std::vector<SavedLocation>
ReadSavedLocations(std::istream &input, size_t maximumCount)
{
    std::vector<SavedLocation> locations;
    while (locations.size() < maximumCount) {
        SavedLocation location;
        if (!ParseSavedLocation(input, location)) {
            break;
        }
        locations.push_back(std::move(location));
    }
    return locations;
}

void
EnqueueSavedLocation(Fractal &fractal, const SavedLocation &location)
{
    PointZoomBBConverter view{location.MinX,
                              location.MinY,
                              location.MaxX,
                              location.MaxY,
                              PointZoomBBConverter::TestMode::Enabled};
    const IterTypeFull numIterations = location.NumIterations;
    const uint32_t antialiasing = location.Antialiasing;
    fractal.EnqueueCommand([view = std::move(view), numIterations, antialiasing](Fractal &f) {
        f.RecenterViewCalc(view);
        f.SetNumIterations<IterTypeFull>(numIterations);
        f.ResetDimensions(SIZE_MAX, SIZE_MAX, antialiasing);
    });
}

EnteredLocation
CaptureEnteredLocation(const Fractal &fractal)
{
    PointZoomBBConverter view{fractal.GetMinX(),
                              fractal.GetMinY(),
                              fractal.GetMaxX(),
                              fractal.GetMaxY(),
                              PointZoomBBConverter::TestMode::Enabled};
    return {view.GetPtX().str(),
            view.GetPtY().str(),
            view.GetZoomFactor().str(),
            fractal.GetNumIterations<IterTypeFull>()};
}

void
EnqueueEnteredLocation(Fractal &fractal, const EnteredLocation &location)
{
    HighPrecision::defaultPrecisionInBits(Fractal::MaxPrecisionLame);
    PointZoomBBConverter view{HighPrecision(location.Real),
                              HighPrecision(location.Imaginary),
                              HighPrecision(location.Zoom),
                              PointZoomBBConverter::TestMode::Enabled};
    const IterTypeFull numIterations = location.NumIterations;
    fractal.EnqueueCommand([view = std::move(view), numIterations](Fractal &f) {
        f.RecenterViewCalc(view);
        f.SetNumIterations<IterTypeFull>(numIterations);
    });
}

} // namespace FractalShark
