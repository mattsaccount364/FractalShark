// RenderToPng — shared headless render helper used by FractalSharkCli and
// FractalSharkTest's golden-CRC regression tests. Wraps the Fractal
// construction + view selection + algorithm/perturbation setup +
// CalcFractal(true) + SaveCurrentFractal sequence into one call.

#pragma once

#include "Fractal.h"
#include "HighPrecision.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"

#include <cstdint>
#include <optional>
#include <string>

struct RenderRequest {
    enum class ViewSourceKind { None, Builtin, BoundingBox, Direct };

    int Width = 1024;
    int Height = 768;

    ViewSourceKind ViewSource = ViewSourceKind::None;

    // ViewSource == Builtin
    size_t BuiltinView = 0;

    // ViewSource == BoundingBox (e.g. from a saved-locations record)
    HighPrecision MinX, MinY, MaxX, MaxY;

    // ViewSource == Direct
    HighPrecision CenterX, CenterY, Zoom;

    // Optional overrides — 0 means "use whatever the view source produced".
    uint64_t Iterations = 0;
    uint32_t Antialiasing = 0;

    uint64_t CommitCapBytes = UINT64_MAX;

    RenderAlgorithm Algorithm; // mandatory
    std::optional<RefOrbitCalc::PerturbationAlg> Perturbation;

    // Filename WITHOUT the .png extension. SaveCurrentFractal appends it.
    std::wstring OutPngBasename;

    bool Quiet = true;
};

// Returns 0 on success, non-zero on failure (message appended to *err if non-null).
int RenderToPng(const RenderRequest &req, std::string *err);
