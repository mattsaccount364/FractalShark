// RenderToPng — shared headless render helper used by FractalSharkCli and
// FractalSharkTest's golden-CRC regression tests.  Wraps view selection +
// algorithm/perturbation setup + CalcFractal(true) + SaveCurrentFractal
// into one call.  The caller owns the Fractal and may reuse it for
// additional output (e.g. console rendering) after the call returns.

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
    // If empty, PNG output is skipped (useful when the caller only wants
    // the computed Fractal for other output like console rendering).
    std::wstring OutPngBasename;

    bool Quiet = true;
};

// Returns 0 on success, non-zero on failure (message appended to *err if non-null).
// Exceptions from the underlying render path propagate to the caller.
// The Fractal is fully computed on return and may be used for additional output.
int RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err);
