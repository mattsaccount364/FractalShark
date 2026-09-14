#include "stdafx.h"

#include "RenderToPng.h"

#include "PointZoomBBConverter.h"
#include "RenderThreadPool.h"

#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <utility>

namespace {

struct RenderSetup {
    RenderRequest::ViewSourceKind ViewSource = RenderRequest::ViewSourceKind::None;
    size_t BuiltinView = 0;
    std::optional<PointZoomBBConverter> View;
    int Width = 0;
    int Height = 0;
    uint64_t Iterations = 0;
    uint32_t Antialiasing = 0;
    RenderAlgorithm Algorithm;
    std::optional<RefOrbitCalc::PerturbationAlg> Perturbation;
};

std::optional<RenderSetup>
BuildRenderSetup(const RenderRequest &req, std::string &error)
{
    if (req.ViewSource == RenderRequest::ViewSourceKind::None) {
        error = "RenderToPng: ViewSource must be set";
        return std::nullopt;
    }

    try {
        RenderSetup setup;
        setup.ViewSource = req.ViewSource;
        setup.BuiltinView = req.BuiltinView;
        setup.Width = req.Width;
        setup.Height = req.Height;
        setup.Iterations = req.Iterations;
        setup.Antialiasing = req.Antialiasing;
        setup.Algorithm = req.Algorithm;
        setup.Perturbation = req.Perturbation;

        switch (req.ViewSource) {
            case RenderRequest::ViewSourceKind::Builtin:
                break;
            case RenderRequest::ViewSourceKind::BoundingBox:
                setup.View.emplace(
                    req.MinX, req.MinY, req.MaxX, req.MaxY, PointZoomBBConverter::TestMode::Enabled);
                break;
            case RenderRequest::ViewSourceKind::Direct:
                setup.View.emplace(
                    req.CenterX, req.CenterY, req.Zoom, PointZoomBBConverter::TestMode::Enabled);
                break;
            case RenderRequest::ViewSourceKind::None:
                break;
        }

        return setup;
    } catch (const std::exception &exception) {
        error = std::string{"RenderToPng: could not build view: "} + exception.what();
        return std::nullopt;
    }
}

bool
ApplyRenderSetup(RenderSetup &setup, Fractal &fractal, std::string &error)
{
    fractal.SetIterType(IterTypeEnum::Bits64);

    switch (setup.ViewSource) {
        case RenderRequest::ViewSourceKind::Builtin:
            fractal.View(setup.BuiltinView, /*includeMsgBox=*/false);
            break;
        case RenderRequest::ViewSourceKind::BoundingBox:
        case RenderRequest::ViewSourceKind::Direct:
            if (!setup.View) {
                error = "RenderToPng: view data is missing";
                return false;
            }
            fractal.RecenterViewCalc(*setup.View);
            break;
        case RenderRequest::ViewSourceKind::None:
            break; // unreachable, checked above
    }

    if (setup.Iterations != 0) {
        fractal.SetNumIterations<uint64_t>(setup.Iterations);
    }

    const uint32_t antialiasing = setup.Antialiasing == 0 ? UINT32_MAX : setup.Antialiasing;
    fractal.ResetDimensions(
        static_cast<size_t>(setup.Width), static_cast<size_t>(setup.Height), antialiasing);

    if (!fractal.SetRenderAlgorithm(setup.Algorithm)) {
        std::ostringstream ss;
        ss << "RenderToPng: SetRenderAlgorithm failed for "
           << (setup.Algorithm.AlgorithmStr ? setup.Algorithm.AlgorithmStr : "<null>");
        error = ss.str();
        return false;
    }

    if (setup.Perturbation.has_value()) {
        fractal.SetPerturbationAlg(*setup.Perturbation);
    }

    return true;
}

int
RunConfiguredRender(const RenderRequest &req,
                    Fractal &fractal,
                    std::string *err,
                    std::ostream &out,
                    PngCompletionMode completionMode)
{
    auto fail = [&](const std::string &msg, int code) -> int {
        if (err) {
            *err = msg;
        } else {
            std::cerr << msg << "\n";
        }
        return code;
    };

    if (!req.Quiet) {
        out << "Rendering " << req.Width << "x" << req.Height << " pixels with "
            << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>") << "...\n";
        out.flush();
    }

    // Direct render path (CrummyTest pattern).  Callers must drain the pool
    // before reaching this point when the Fractal is shared with it.
    fractal.GetRenderPool()->Drain();
    fractal.CalcFractal(/*drawFractal=*/true);

    // PNG output (skipped if no basename was given).
    if (!req.OutPngBasename.empty()) {
        // Move the completed buffer to the PNG worker unless the caller still
        // needs it for console output. Fractal replaces a moved buffer before
        // the next render starts.
        int rc = fractal.SaveCurrentFractal(req.OutPngBasename, req.PreserveIterationBuffer);
        if (rc != 0) {
            std::ostringstream ss;
            ss << "RenderToPng: SaveCurrentFractal returned code " << rc;
            return fail(ss.str(), rc);
        }

        if (completionMode == PngCompletionMode::Wait) {
            fractal.CleanupThreads(/*all=*/true);
        }
    }

    return 0;
}

} // namespace

int
RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err)
{
    return RenderToPng(req, fractal, err, std::cout);
}

int
RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err, std::ostream &out)
{
    return RenderToPng(req, fractal, err, out, PngCompletionMode::Wait);
}

int
RenderToPng(const RenderRequest &req,
            Fractal &fractal,
            std::string *err,
            std::ostream &out,
            PngCompletionMode completionMode)
{
    std::string setupError;
    auto setup = BuildRenderSetup(req, setupError);
    if (!setup) {
        if (err) {
            *err = setupError;
        } else {
            std::cerr << setupError << "\n";
        }
        return 2;
    }

    if (!ApplyRenderSetup(*setup, fractal, setupError)) {
        if (err) {
            *err = setupError;
        } else {
            std::cerr << setupError << "\n";
        }
        return 1;
    }

    return RunConfiguredRender(req, fractal, err, out, completionMode);
}

int
RenderToPngQueued(const RenderRequest &req, Fractal &fractal, std::string *err)
{
    return RenderToPngQueued(req, fractal, err, std::cout);
}

int
RenderToPngQueued(const RenderRequest &req, Fractal &fractal, std::string *err, std::ostream &out)
{
    return RenderToPngQueued(req, fractal, err, out, PngCompletionMode::Wait);
}

int
RenderToPngQueued(const RenderRequest &req,
                  Fractal &fractal,
                  std::string *err,
                  std::ostream &out,
                  PngCompletionMode completionMode)
{
    auto fail = [&](const std::string &msg, int code) -> int {
        if (err) {
            *err = msg;
        } else {
            std::cerr << msg << "\n";
        }
        return code;
    };

    std::string setupError;
    auto setup = BuildRenderSetup(req, setupError);
    if (!setup) {
        return fail(setupError, 2);
    }

    if (!fractal.GetRenderPool()) {
        return fail("RenderToPngQueued: Fractal has no render pool", 1);
    }

    fractal.GetRenderPool()->Drain();
    bool setupSucceeded = true;
    auto setupJob = fractal.EnqueueMutation(
        [setup = std::move(*setup), &setupSucceeded, &setupError](Fractal &f) mutable {
            setupSucceeded = ApplyRenderSetup(setup, f, setupError);
        });
    setupJob.Wait();
    fractal.GetRenderPool()->Drain();

    if (!setupSucceeded) {
        return fail(setupError, 1);
    }

    return RunConfiguredRender(req, fractal, err, out, completionMode);
}
