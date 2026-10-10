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

    if (req.Perturbation.has_value()) {
        // View setup chooses precision from the reference backend's reuse requirements.
        // Apply this request's backend first so a preceding render cannot affect that choice.
        fractal.SetPerturbationAlg(*req.Perturbation);
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

    if (!req.Quiet) {
        out << "Rendering " << req.Width << "x" << req.Height << " pixels with "
            << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>") << "...\n";
        out.flush();
    }

    fractal.GetRenderPool()->Drain();
    if (req.ClearReferenceOrbits) {
        // Workers must release their reference pointers before the cache is destroyed.
        // Clearing all results also discards derived LA data and resets the reference guess.
        fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        fractal.ForceRecalc();
    }
    fractal.CalcFractal(/*drawFractal=*/true);

    if (!req.OutPngBasename.empty()) {
        fractal.SaveCurrentFractal(
            req.OutPngBasename, req.PreserveIterationBuffer, SaveOverwritePolicy::SkipExisting);
        if (req.PngCompletion == PngCompletionMode::Wait) {
            fractal.CleanupThreads(/*all=*/true);
        }
    }

    return 0;
}
