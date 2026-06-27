#pragma once

#include "HighPrecision.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

class Fractal;

namespace FractalShark {

struct SavedLocation {
    size_t Width = 0;
    size_t Height = 0;
    HighPrecision MinX;
    HighPrecision MinY;
    HighPrecision MaxX;
    HighPrecision MaxY;
    IterTypeFull NumIterations = 0;
    uint32_t Antialiasing = 0;
    std::string Description;
};

struct EnteredLocation {
    std::string Real;
    std::string Imaginary;
    std::string Zoom;
    IterTypeFull NumIterations = 0;
};

inline constexpr const char *kSavedLocationsFilename = "locations.txt";

SavedLocation CaptureSavedLocation(const Fractal &fractal, bool scaleToMaximum);
std::string SerializeSavedLocation(const SavedLocation &location);
bool ParseSavedLocation(std::istream &input, SavedLocation &location);
std::vector<SavedLocation> ReadSavedLocations(std::istream &input, size_t maximumCount);
std::string AppendSavedLocation(Fractal &fractal,
                                bool scaleToMaximum,
                                const char *filename = kSavedLocationsFilename);
std::vector<SavedLocation> ReadSavedLocationsFile(const char *filename, size_t maximumCount);
std::vector<std::string> BuildSavedLocationLabels(const std::vector<SavedLocation> &locations);
void EnqueueSavedLocation(Fractal &fractal, const SavedLocation &location);
EnteredLocation CaptureEnteredLocation(const Fractal &fractal);
void EnqueueEnteredLocation(Fractal &fractal, const EnteredLocation &location);

} // namespace FractalShark
