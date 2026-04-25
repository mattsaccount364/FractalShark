// FractalSharkCli: headless PNG renderer.
//
// Process-level init mirrors FractalSharkGUILib/FractalShark.cpp (WinMain +
// MainWindow ctor): Environment::RegisterHeapCleanup + GlobalCallstacks +
// CrashHandler::Install (Windows only) + FreeCallstacks at teardown.

#include "stdafx.h"

#include "Callstacks.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "Fractal.h"
#include "PointZoomBBConverter.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "RenderThreadPool.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include "heap_allocator/include/HeapCpp.h"
#endif

namespace {

enum class ViewSource { None, Builtin, LocationsFile, Direct };

struct CliArgs {
    int Width = 1024;
    int Height = 768;
    bool WidthSet = false;
    bool HeightSet = false;

    ViewSource Source = ViewSource::None;
    size_t BuiltinView = 0;
    std::string LocationsFile;
    size_t LocationIndex = SIZE_MAX; // SIZE_MAX => use last record
    std::string CenterX, CenterY, Zoom;

    uint64_t Iterations = 0;              // 0 => unspecified (parser rejects 0)
    uint32_t Antialiasing = 0;            // 0 => unspecified (parser rejects 0)
    uint64_t CommitCapBytes = UINT64_MAX; // UINT64_MAX => unlimited

    std::string RenderAlgorithm;
    std::string PerturbationAlg; // empty => unspecified

    std::string OutFile;

    bool ListRenderAlgorithms = false;
    bool Quiet = false;
    bool Help = false;
};

void
PrintUsage()
{
    std::cout << "FractalSharkCli — headless Mandelbrot renderer\n"
                 "\n"
                 "Usage:\n"
                 "  FractalSharkCli --render-algorithm NAME --out FILE.png [--width W --height H]\n"
                 "                  {--builtin-view N |\n"
                 "                   --locations FILE [--location-index N] |\n"
                 "                   --center-x X --center-y Y --zoom Z}\n"
                 "                  [--iterations N] [--antialiasing N]\n"
                 "                  [--perturbation-alg NAME] [--commit-cap-bytes N]\n"
                 "                  [--quiet]\n"
                 "\n"
                 "  FractalSharkCli --list-render-algorithms\n"
                 "  FractalSharkCli --help\n"
                 "\n"
                 "Per-pixel render algorithm names match RenderAlgorithmEnum\n"
                 "(e.g. Cpu64PerturbedBLAV2HDR, Gpu1x32PerturbedLAv2, CpuHigh).\n"
                 "Run with --list-render-algorithms for the full list.\n";
}

void
PrintRenderAlgorithms()
{
    for (const auto &alg : RenderAlgorithms) {
        if (alg.AlgorithmStr && alg.AlgorithmStr[0] != '\0') {
            std::cout << alg.AlgorithmStr << "\n";
        }
    }
}

std::optional<RenderAlgorithm>
ParseRenderAlgorithm(const std::string &name)
{
    for (const auto &alg : RenderAlgorithms) {
        if (alg.AlgorithmStr && name == alg.AlgorithmStr) {
            return alg;
        }
    }
    return std::nullopt;
}

std::optional<RefOrbitCalc::PerturbationAlg>
ParsePerturbationAlg(const std::string &name)
{
    using P = RefOrbitCalc::PerturbationAlg;
    if (name == "ST")
        return P::ST;
    if (name == "MT")
        return P::MT;
    if (name == "STPeriodicity")
        return P::STPeriodicity;
    if (name == "MTPeriodicity3")
        return P::MTPeriodicity3;
    if (name == "MTPeriodicity3PerturbMTHighSTMed")
        return P::MTPeriodicity3PerturbMTHighSTMed;
    if (name == "MTPeriodicity3PerturbMTHighMTMed1")
        return P::MTPeriodicity3PerturbMTHighMTMed1;
    if (name == "MTPeriodicity3PerturbMTHighMTMed2")
        return P::MTPeriodicity3PerturbMTHighMTMed2;
    if (name == "MTPeriodicity3PerturbMTHighMTMed3")
        return P::MTPeriodicity3PerturbMTHighMTMed3;
    if (name == "MTPeriodicity3PerturbMTHighMTMed4")
        return P::MTPeriodicity3PerturbMTHighMTMed4;
    if (name == "MTPeriodicity5")
        return P::MTPeriodicity5;
    if (name == "GPU")
        return P::GPU;
    if (name == "Auto")
        return P::Auto;
    return std::nullopt;
}

bool
ParseUint64(const char *s, uint64_t &out)
{
    if (!s || !*s)
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || *end != '\0')
        return false;
    out = static_cast<uint64_t>(v);
    return true;
}

bool
ParseSizeT(const char *s, size_t &out)
{
    uint64_t v;
    if (!ParseUint64(s, v))
        return false;
    out = static_cast<size_t>(v);
    return true;
}

bool
ParseInt(const char *s, int &out)
{
    uint64_t v;
    if (!ParseUint64(s, v))
        return false;
    if (v > static_cast<uint64_t>(INT32_MAX))
        return false;
    out = static_cast<int>(v);
    return true;
}

// Returns true on success, false on error (message already printed).
bool
ParseArgs(int argc, char *argv[], CliArgs &a)
{
    auto expectValue = [&](int &i, const char *flag) -> const char * {
        if (i + 1 >= argc) {
            std::cerr << "error: " << flag << " requires an argument\n";
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            a.Help = true;
        } else if (arg == "--list-render-algorithms") {
            a.ListRenderAlgorithms = true;
        } else if (arg == "--quiet") {
            a.Quiet = true;
        } else if (arg == "--width") {
            auto v = expectValue(i, "--width");
            if (!v || !ParseInt(v, a.Width))
                return false;
            a.WidthSet = true;
        } else if (arg == "--height") {
            auto v = expectValue(i, "--height");
            if (!v || !ParseInt(v, a.Height))
                return false;
            a.HeightSet = true;
        } else if (arg == "--render-algorithm") {
            auto v = expectValue(i, "--render-algorithm");
            if (!v)
                return false;
            a.RenderAlgorithm = v;
        } else if (arg == "--out") {
            auto v = expectValue(i, "--out");
            if (!v)
                return false;
            a.OutFile = v;
        } else if (arg == "--builtin-view") {
            auto v = expectValue(i, "--builtin-view");
            if (!v || !ParseSizeT(v, a.BuiltinView))
                return false;
            a.Source = ViewSource::Builtin;
        } else if (arg == "--locations") {
            auto v = expectValue(i, "--locations");
            if (!v)
                return false;
            a.LocationsFile = v;
            a.Source = ViewSource::LocationsFile;
        } else if (arg == "--location-index") {
            auto v = expectValue(i, "--location-index");
            size_t idx;
            if (!v || !ParseSizeT(v, idx))
                return false;
            a.LocationIndex = idx;
        } else if (arg == "--center-x") {
            auto v = expectValue(i, "--center-x");
            if (!v)
                return false;
            a.CenterX = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--center-y") {
            auto v = expectValue(i, "--center-y");
            if (!v)
                return false;
            a.CenterY = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--zoom") {
            auto v = expectValue(i, "--zoom");
            if (!v)
                return false;
            a.Zoom = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--iterations") {
            auto v = expectValue(i, "--iterations");
            uint64_t n;
            if (!v || !ParseUint64(v, n))
                return false;
            if (n == 0) {
                std::cerr << "error: --iterations must be > 0\n";
                return false;
            }
            a.Iterations = n;
        } else if (arg == "--antialiasing") {
            auto v = expectValue(i, "--antialiasing");
            uint64_t n;
            if (!v || !ParseUint64(v, n))
                return false;
            if (n == 0) {
                std::cerr << "error: --antialiasing must be >= 1\n";
                return false;
            }
            a.Antialiasing = static_cast<uint32_t>(n);
        } else if (arg == "--commit-cap-bytes") {
            auto v = expectValue(i, "--commit-cap-bytes");
            uint64_t n;
            if (!v || !ParseUint64(v, n))
                return false;
            a.CommitCapBytes = n;
        } else if (arg == "--perturbation-alg") {
            auto v = expectValue(i, "--perturbation-alg");
            if (!v)
                return false;
            a.PerturbationAlg = v;
        } else {
            std::cerr << "error: unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

// Simple saved-location record. Mirrors the format consumed by the GUI's
// MainWindow::SavedLocation parser in FractalSharkGUILib/MainWindowSavedLocation.h:
//   width height minX minY maxX maxY num_iterations antialiasing
//   <description line>
struct ParsedSavedLocation {
    size_t Width = 0;
    size_t Height = 0;
    uint64_t NumIterations = 0;
    uint32_t Antialiasing = 0;
    HighPrecision MinX, MinY, MaxX, MaxY;
    std::string Description;
};

bool
LoadLocations(const std::string &path, std::vector<ParsedSavedLocation> &out)
{
    std::ifstream in(path);
    if (!in) {
        std::cerr << "error: cannot open locations file: " << path << "\n";
        return false;
    }

    while (in.good()) {
        ParsedSavedLocation rec;
        in >> rec.Width >> rec.Height;
        in >> rec.MinX >> rec.MinY >> rec.MaxX >> rec.MaxY;
        in >> rec.NumIterations >> rec.Antialiasing;
        if (!in.good())
            break;
        in >> std::ws;
        std::getline(in, rec.Description);
        out.push_back(std::move(rec));
    }
    return !out.empty();
}

std::wstring
ToWStringUtf8(const std::string &s)
{
    // argv on Linux is UTF-8; on Windows we're limited to MultiByte (ACP)
    // since the vcxproj builds with CharacterSet=MultiByte. Plain widening
    // is adequate for ASCII filenames which is all the CLI smoke tests use.
    std::wstring w;
    w.reserve(s.size());
    for (unsigned char c : s)
        w.push_back(static_cast<wchar_t>(c));
    return w;
}

} // anonymous namespace

int
main(int argc, char *argv[])
{
    Environment::RegisterHeapCleanup();
    GlobalCallstacks->InitCallstacks();
#ifdef _WIN32
    CrashHandler::Install();
#endif

    // Ensure FreeCallstacks runs on every exit path.
    struct CallstackGuard {
        ~CallstackGuard() { GlobalCallstacks->FreeCallstacks(); }
    } callstackGuard;

    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        return 2;
    }

    if (args.Help) {
        PrintUsage();
        return 0;
    }

    if (args.ListRenderAlgorithms) {
        PrintRenderAlgorithms();
        return 0;
    }

    if (args.RenderAlgorithm.empty()) {
        std::cerr << "error: --render-algorithm is required\n";
        PrintUsage();
        return 2;
    }
    if (args.OutFile.empty()) {
        std::cerr << "error: --out is required\n";
        return 2;
    }
    if (args.Source == ViewSource::None) {
        std::cerr << "error: one of --builtin-view, --locations, or "
                     "--center-x/--center-y/--zoom is required\n";
        return 2;
    }
    if (args.Source == ViewSource::Direct &&
        (args.CenterX.empty() || args.CenterY.empty() || args.Zoom.empty())) {
        std::cerr << "error: --center-x, --center-y, and --zoom must be specified together\n";
        return 2;
    }

    auto parsedAlg = ParseRenderAlgorithm(args.RenderAlgorithm);
    if (!parsedAlg) {
        std::cerr << "error: unknown render algorithm: " << args.RenderAlgorithm
                  << "\n(run --list-render-algorithms for valid names)\n";
        return 2;
    }

    // Pre-load locations so we can honor width/height/iters/AA from the record.
    std::vector<ParsedSavedLocation> locations;
    const ParsedSavedLocation *loc = nullptr;
    if (args.Source == ViewSource::LocationsFile) {
        if (!LoadLocations(args.LocationsFile, locations)) {
            return 1;
        }
        size_t idx = (args.LocationIndex == SIZE_MAX) ? locations.size() - 1 : args.LocationIndex;
        if (idx >= locations.size()) {
            std::cerr << "error: --location-index " << idx << " out of range (file has "
                      << locations.size() << " records)\n";
            return 2;
        }
        loc = &locations[idx];
        if (!args.WidthSet)
            args.Width = static_cast<int>(loc->Width);
        if (!args.HeightSet)
            args.Height = static_cast<int>(loc->Height);
    }

    try {
        Fractal fractal(args.Width,
                        args.Height,
                        /*nativeWindow=*/nullptr,
                        /*UseSensoCursor=*/false,
                        args.CommitCapBytes);

        // 64-bit iter counter by default (max ergonomic surface for CLI users).
        fractal.SetIterType(IterTypeEnum::Bits64);

        // Apply view source.
        switch (args.Source) {
            case ViewSource::Builtin:
                fractal.View(args.BuiltinView, /*includeMsgBox=*/false);
                break;
            case ViewSource::LocationsFile: {
                PointZoomBBConverter ptz(
                    loc->MinX, loc->MinY, loc->MaxX, loc->MaxY, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                fractal.SetNumIterations<uint64_t>(loc->NumIterations);
                fractal.ResetDimensions(static_cast<size_t>(args.Width),
                                        static_cast<size_t>(args.Height),
                                        loc->Antialiasing);
                break;
            }
            case ViewSource::Direct: {
                HighPrecision cx(args.CenterX);
                HighPrecision cy(args.CenterY);
                HighPrecision zm(args.Zoom);
                PointZoomBBConverter ptz(cx, cy, zm, PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                break;
            }
            case ViewSource::None:
                break; // unreachable (checked above)
        }

        if (args.Iterations != 0) {
            fractal.SetNumIterations<uint64_t>(args.Iterations);
        }
        if (args.Antialiasing != 0) {
            fractal.ResetDimensions(
                static_cast<size_t>(args.Width), static_cast<size_t>(args.Height), args.Antialiasing);
        }

        if (!fractal.SetRenderAlgorithm(*parsedAlg)) {
            std::cerr << "error: SetRenderAlgorithm failed for " << args.RenderAlgorithm << "\n";
            return 1;
        }

        if (!args.PerturbationAlg.empty()) {
            auto p = ParsePerturbationAlg(args.PerturbationAlg);
            if (!p) {
                std::cerr << "error: unknown perturbation algorithm: " << args.PerturbationAlg << "\n";
                return 2;
            }
            fractal.SetPerturbationAlg(*p);
        }

        if (!args.Quiet) {
            std::cout << "Rendering " << args.Width << "x" << args.Height << " with "
                      << args.RenderAlgorithm << "...\n";
            std::cout.flush();
        }

        // Direct render path (CrummyTest pattern): drain the pool, render
        // synchronously into m_CurIters, then save.
        fractal.GetRenderPool()->Drain();
        fractal.CalcFractal(/*drawFractal=*/true);

        std::wstring base = ToWStringUtf8(args.OutFile);
        const std::wstring pngExt = L".png";
        if (base.size() >= pngExt.size() &&
            base.compare(base.size() - pngExt.size(), pngExt.size(), pngExt) == 0) {
            base.resize(base.size() - pngExt.size());
        }

        int rc = fractal.SaveCurrentFractal(base, /*copy_the_iters=*/false);
        if (rc != 0) {
            std::cerr << "error: SaveCurrentFractal returned " << rc << "\n";
        } else if (!args.Quiet) {
            std::cout << "Wrote " << args.OutFile << "\n";
        }

        return rc;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
