#include "stdafx.h"

#include "RenderToPng.h"

#include "PointZoomBBConverter.h"
#include "RenderThreadPool.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

int
RenderToPng(const RenderRequest &req, std::string *err)
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
    if (req.OutPngBasename.empty()) {
        return fail("RenderToPng: OutPngBasename is required", 2);
    }

    try {
        Fractal fractal(req.Width,
                        req.Height,
                        /*nativeWindow=*/nullptr,
                        /*UseSensoCursor=*/false,
                        req.CommitCapBytes);

        fractal.SetIterType(IterTypeEnum::Bits64);

        switch (req.ViewSource) {
            case RenderRequest::ViewSourceKind::Builtin:
                fractal.View(req.BuiltinView, /*includeMsgBox=*/false);
                break;
            case RenderRequest::ViewSourceKind::BoundingBox: {
                PointZoomBBConverter ptz(req.MinX,
                                         req.MinY,
                                         req.MaxX,
                                         req.MaxY,
                                         PointZoomBBConverter::TestMode::Enabled);
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
            fractal.ResetDimensions(static_cast<size_t>(req.Width),
                                    static_cast<size_t>(req.Height),
                                    req.Antialiasing);
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
            std::cout << "Rendering " << req.Width << "x" << req.Height << " with "
                      << (req.Algorithm.AlgorithmStr ? req.Algorithm.AlgorithmStr : "<null>")
                      << "...\n";
            std::cout.flush();
        }

        // Direct render path (CrummyTest pattern).
        fractal.GetRenderPool()->Drain();
        fractal.CalcFractal(/*drawFractal=*/true);

        int rc = fractal.SaveCurrentFractal(req.OutPngBasename, /*copy_the_iters=*/false);
        if (rc != 0) {
            std::ostringstream ss;
            ss << "RenderToPng: SaveCurrentFractal returned " << rc;
            return fail(ss.str(), rc);
        }
        return 0;
    } catch (const std::exception &e) {
        return fail(std::string("RenderToPng: ") + e.what(), 1);
    }
}
