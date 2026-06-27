#include "stdafx.h"

#include "RenderToPng.h"

#include "PointZoomBBConverter.h"
#include "RenderThreadPool.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

int
RenderToPng(const RenderRequest &req, Fractal &fractal, std::string *err)
{
    auto fail = [&](const std::string &msg, int code) -> int {
        if (err) {
            *err = msg;
        } else {
            std::cerr << msg << "\n";
        }
        return code;
    };

    if (req.ViewSource == RenderRequest::ViewSourceKind::None) {
        return fail("RenderToPng: ViewSource must be set", 2);
    }

    try {
        fractal.SetIterType(IterTypeEnum::Bits64);

        switch (req.ViewSource) {
            case RenderRequest::ViewSourceKind::Builtin:
                fractal.View(req.BuiltinView, /*includeMsgBox=*/false);
                break;
            case RenderRequest::ViewSourceKind::BoundingBox: {
                PointZoomBBConverter ptz(
                    req.MinX, req.MinY, req.MaxX, req.MaxY, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                break;
            }
            case RenderRequest::ViewSourceKind::Direct: {
                PointZoomBBConverter ptz(
                    req.CenterX, req.CenterY, req.Zoom, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                break;
            }
            case RenderRequest::ViewSourceKind::None:
                break; // unreachable, checked above
        }

        if (req.Iterations != 0) {
            fractal.SetNumIterations<uint64_t>(req.Iterations);
        }
        if (req.Antialiasing != 0) {
            fractal.ResetDimensions(
                static_cast<size_t>(req.Width), static_cast<size_t>(req.Height), req.Antialiasing);
        }

        if (!fractal.SetRenderAlgorithm(req.Algorithm)) {
            std::ostringstream ss;
            ss << "RenderToPng: SetRenderAlgorithm failed for "
               << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>");
            return fail(ss.str(), 1);
        }

        if (req.Perturbation.has_value()) {
            fractal.SetPerturbationAlg(*req.Perturbation);
        }

        if (!req.Quiet) {
            std::cout << "Rendering " << req.Width << "x" << req.Height << " pixels with "
                      << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>") << "...\n";
            std::cout.flush();
        }

        // Direct render path (CrummyTest pattern).
        fractal.GetRenderPool()->Drain();
        fractal.CalcFractal(/*drawFractal=*/true);

        // PNG output (skipped if no basename was given).
        if (!req.OutPngBasename.empty()) {
            int rc = fractal.SaveCurrentFractal(req.OutPngBasename, /*copy_the_iters=*/false);
            if (rc != 0) {
                std::ostringstream ss;
                ss << "RenderToPng: SaveCurrentFractal returned code " << rc;
                return fail(ss.str(), rc);
            }

            // SaveCurrentFractal spawns a background thread; wait for it so
            // the PNG file exists when this function returns.
            fractal.CleanupThreads(/*all=*/true);
        }

        return 0;
    } catch (const std::exception &e) {
        return fail(std::string("RenderToPng: ") + e.what(), 1);
    }
}
