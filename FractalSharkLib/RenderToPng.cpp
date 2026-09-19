#include "stdafx.h"

#include "RenderToPng.h"

#include "ConsoleLog.h"
#include "PointZoomBBConverter.h"
#include "RenderThreadPool.h"

#include <iostream>
#include <ostream>
#include <sstream>

int
RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err)
{
    return RenderToPng(req, fractal, err, std::cout);
}

int
RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err, std::ostream &out)
{
    auto fail = [&](const std::string &message, int code) {
        if (err) {
            *err = message;
        } else {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << message;
        }
        return code;
    };

    if (req.ViewSource == RenderRequest::ViewSourceKind::None) {
        return fail("RenderToPng: ViewSource must be set", 2);
    }

    fractal.SetIterType(IterTypeEnum::Bits64);

    try {
        switch (req.ViewSource) {
            case RenderRequest::ViewSourceKind::Builtin:
                fractal.View(req.BuiltinView, /*includeMsgBox=*/false);
                break;
            case RenderRequest::ViewSourceKind::BoundingBox: {
                PointZoomBBConverter view(
                    req.MinX, req.MinY, req.MaxX, req.MaxY, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(view);
                break;
            }
            case RenderRequest::ViewSourceKind::Direct: {
                PointZoomBBConverter view(
                    req.CenterX, req.CenterY, req.Zoom, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(view);
                break;
            }
            case RenderRequest::ViewSourceKind::None:
                break;
        }
    } catch (const std::exception &exception) {
        return fail(std::string{"RenderToPng: could not build view: "} + exception.what(), 2);
    }

    if (req.Iterations != 0) {
        fractal.SetNumIterations<uint64_t>(req.Iterations);
    }

    const uint32_t antialiasing = req.Antialiasing == 0 ? UINT32_MAX : req.Antialiasing;
    fractal.ResetDimensions(
        static_cast<size_t>(req.Width), static_cast<size_t>(req.Height), antialiasing);

    if (!fractal.SetRenderAlgorithm(req.Algorithm)) {
        std::ostringstream message;
        message << "RenderToPng: SetRenderAlgorithm failed for "
                << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>");
        return fail(message.str(), 1);
    }

    if (req.Perturbation.has_value()) {
        fractal.SetPerturbationAlg(*req.Perturbation);
    }

    if (!req.Quiet) {
        out << "Rendering " << req.Width << "x" << req.Height << " pixels with "
            << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>") << "...\n";
        out.flush();
    }

    fractal.GetRenderPool()->Drain();
    fractal.CalcFractal(/*drawFractal=*/true);

    if (!req.OutPngBasename.empty()) {
        const int rc = fractal.SaveCurrentFractal(req.OutPngBasename, req.PreserveIterationBuffer);
        if (rc != 0) {
            std::ostringstream message;
            message << "RenderToPng: SaveCurrentFractal returned code " << rc;
            return fail(message.str(), rc);
        }
        if (req.PngCompletion == PngCompletionMode::Wait) {
            fractal.CleanupThreads(/*all=*/true);
        }
    }

    return 0;
}
