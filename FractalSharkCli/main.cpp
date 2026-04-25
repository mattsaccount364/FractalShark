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
#include "heap_allocator\include\HeapCpp.h"
#endif

namespace {

enum class ViewSource { None, Builtin, LocationsFile, Direct };

struct CliArgs {
    int width = 1024;
    int height = 768;
    bool width_set = false;
    bool height_set = false;

    ViewSource view_source = ViewSource::None;
    size_t builtin_view = 0;
    std::string locations_file;
    std::optional<size_t> location_index; // nullopt => last record
    std::string center_x, center_y, zoom;

    std::optional<uint64_t> iterations;
    std::optional<uint32_t> antialiasing;
    std::optional<uint64_t> commit_cap_bytes;

    std::string render_algorithm;
    std::optional<std::string> perturbation_alg;

    std::string out_file;

    bool list_render_algorithms = false;
    bool quiet = false;
    bool help = false;
};

void
PrintUsage()
{
    std::cout <<
        "FractalSharkCli — headless Mandelbrot renderer\n"
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
    if (name == "ST") return P::ST;
    if (name == "MT") return P::MT;
    if (name == "STPeriodicity") return P::STPeriodicity;
    if (name == "MTPeriodicity3") return P::MTPeriodicity3;
    if (name == "MTPeriodicity3PerturbMTHighSTMed") return P::MTPeriodicity3PerturbMTHighSTMed;
    if (name == "MTPeriodicity3PerturbMTHighMTMed1") return P::MTPeriodicity3PerturbMTHighMTMed1;
    if (name == "MTPeriodicity3PerturbMTHighMTMed2") return P::MTPeriodicity3PerturbMTHighMTMed2;
    if (name == "MTPeriodicity3PerturbMTHighMTMed3") return P::MTPeriodicity3PerturbMTHighMTMed3;
    if (name == "MTPeriodicity3PerturbMTHighMTMed4") return P::MTPeriodicity3PerturbMTHighMTMed4;
    if (name == "MTPeriodicity5") return P::MTPeriodicity5;
    if (name == "GPU") return P::GPU;
    if (name == "Auto") return P::Auto;
    return std::nullopt;
}

bool
ParseUint64(const char *s, uint64_t &out)
{
    if (!s || !*s) return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

bool
ParseSizeT(const char *s, size_t &out)
{
    uint64_t v;
    if (!ParseUint64(s, v)) return false;
    out = static_cast<size_t>(v);
    return true;
}

bool
ParseInt(const char *s, int &out)
{
    uint64_t v;
    if (!ParseUint64(s, v)) return false;
    if (v > static_cast<uint64_t>(INT32_MAX)) return false;
    out = static_cast<int>(v);
    return true;
}

// Returns true on success, false on error (message already printed).
bool
ParseArgs(int argc, char *argv[], CliArgs &a)
{
    auto expect_value = [&](int &i, const char *flag) -> const char * {
        if (i + 1 >= argc) {
            std::cerr << "error: " << flag << " requires an argument\n";
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--list-render-algorithms") {
            a.list_render_algorithms = true;
        } else if (arg == "--quiet") {
            a.quiet = true;
        } else if (arg == "--width") {
            auto v = expect_value(i, "--width");
            if (!v || !ParseInt(v, a.width)) return false;
            a.width_set = true;
        } else if (arg == "--height") {
            auto v = expect_value(i, "--height");
            if (!v || !ParseInt(v, a.height)) return false;
            a.height_set = true;
        } else if (arg == "--render-algorithm") {
            auto v = expect_value(i, "--render-algorithm");
            if (!v) return false;
            a.render_algorithm = v;
        } else if (arg == "--out") {
            auto v = expect_value(i, "--out");
            if (!v) return false;
            a.out_file = v;
        } else if (arg == "--builtin-view") {
            auto v = expect_value(i, "--builtin-view");
            if (!v || !ParseSizeT(v, a.builtin_view)) return false;
            a.view_source = ViewSource::Builtin;
        } else if (arg == "--locations") {
            auto v = expect_value(i, "--locations");
            if (!v) return false;
            a.locations_file = v;
            a.view_source = ViewSource::LocationsFile;
        } else if (arg == "--location-index") {
            auto v = expect_value(i, "--location-index");
            size_t idx;
            if (!v || !ParseSizeT(v, idx)) return false;
            a.location_index = idx;
        } else if (arg == "--center-x") {
            auto v = expect_value(i, "--center-x");
            if (!v) return false;
            a.center_x = v;
            a.view_source = ViewSource::Direct;
        } else if (arg == "--center-y") {
            auto v = expect_value(i, "--center-y");
            if (!v) return false;
            a.center_y = v;
            a.view_source = ViewSource::Direct;
        } else if (arg == "--zoom") {
            auto v = expect_value(i, "--zoom");
            if (!v) return false;
            a.zoom = v;
            a.view_source = ViewSource::Direct;
        } else if (arg == "--iterations") {
            auto v = expect_value(i, "--iterations");
            uint64_t n;
            if (!v || !ParseUint64(v, n)) return false;
            a.iterations = n;
        } else if (arg == "--antialiasing") {
            auto v = expect_value(i, "--antialiasing");
            uint64_t n;
            if (!v || !ParseUint64(v, n)) return false;
            a.antialiasing = static_cast<uint32_t>(n);
        } else if (arg == "--commit-cap-bytes") {
            auto v = expect_value(i, "--commit-cap-bytes");
            uint64_t n;
            if (!v || !ParseUint64(v, n)) return false;
            a.commit_cap_bytes = n;
        } else if (arg == "--perturbation-alg") {
            auto v = expect_value(i, "--perturbation-alg");
            if (!v) return false;
            a.perturbation_alg = v;
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
    size_t width = 0;
    size_t height = 0;
    uint64_t num_iterations = 0;
    uint32_t antialiasing = 0;
    HighPrecision minX, minY, maxX, maxY;
    std::string description;
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
        in >> rec.width >> rec.height;
        in >> rec.minX >> rec.minY >> rec.maxX >> rec.maxY;
        in >> rec.num_iterations >> rec.antialiasing;
        if (!in.good()) break;
        in >> std::ws;
        std::getline(in, rec.description);
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
    for (unsigned char c : s) w.push_back(static_cast<wchar_t>(c));
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

    CliArgs args;
    if (!ParseArgs(argc, argv, args)) {
        PrintUsage();
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }

    if (args.help) {
        PrintUsage();
        GlobalCallstacks->FreeCallstacks();
        return 0;
    }

    if (args.list_render_algorithms) {
        PrintRenderAlgorithms();
        GlobalCallstacks->FreeCallstacks();
        return 0;
    }

    if (args.render_algorithm.empty()) {
        std::cerr << "error: --render-algorithm is required\n";
        PrintUsage();
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }
    if (args.out_file.empty()) {
        std::cerr << "error: --out is required\n";
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }
    if (args.view_source == ViewSource::None) {
        std::cerr << "error: one of --builtin-view, --locations, or "
                     "--center-x/--center-y/--zoom is required\n";
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }
    if (args.view_source == ViewSource::Direct &&
        (args.center_x.empty() || args.center_y.empty() || args.zoom.empty())) {
        std::cerr << "error: --center-x, --center-y, and --zoom must be specified together\n";
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }

    auto parsed_alg = ParseRenderAlgorithm(args.render_algorithm);
    if (!parsed_alg) {
        std::cerr << "error: unknown render algorithm: " << args.render_algorithm
                  << "\n(run --list-render-algorithms for valid names)\n";
        GlobalCallstacks->FreeCallstacks();
        return 2;
    }

    // Pre-load locations so we can honor width/height/iters/AA from the record.
    std::vector<ParsedSavedLocation> locations;
    const ParsedSavedLocation *loc = nullptr;
    if (args.view_source == ViewSource::LocationsFile) {
        if (!LoadLocations(args.locations_file, locations)) {
            GlobalCallstacks->FreeCallstacks();
            return 1;
        }
        size_t idx = args.location_index.value_or(locations.size() - 1);
        if (idx >= locations.size()) {
            std::cerr << "error: --location-index " << idx << " out of range (file has "
                      << locations.size() << " records)\n";
            GlobalCallstacks->FreeCallstacks();
            return 2;
        }
        loc = &locations[idx];
        if (!args.width_set) args.width = static_cast<int>(loc->width);
        if (!args.height_set) args.height = static_cast<int>(loc->height);
    }

    const uint64_t commit_cap = args.commit_cap_bytes.value_or(UINT64_MAX);

    try {
        Fractal fractal(args.width, args.height,
                        /*nativeWindow=*/nullptr,
                        /*UseSensoCursor=*/false,
                        commit_cap);

        // 64-bit iter counter by default (max ergonomic surface for CLI users).
        fractal.SetIterType(IterTypeEnum::Bits64);

        // Apply view source.
        switch (args.view_source) {
            case ViewSource::Builtin:
                fractal.View(args.builtin_view, /*includeMsgBox=*/false);
                break;
            case ViewSource::LocationsFile: {
                PointZoomBBConverter ptz(loc->minX, loc->minY, loc->maxX, loc->maxY,
                                         PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                fractal.SetNumIterations<uint64_t>(loc->num_iterations);
                fractal.ResetDimensions(static_cast<size_t>(args.width),
                                        static_cast<size_t>(args.height),
                                        loc->antialiasing);
                break;
            }
            case ViewSource::Direct: {
                HighPrecision cx(args.center_x);
                HighPrecision cy(args.center_y);
                HighPrecision zm(args.zoom);
                PointZoomBBConverter ptz(cx, cy, zm,
                                         PointZoomBBConverter::TestMode::Enabled);
                fractal.RecenterViewCalc(ptz);
                break;
            }
            case ViewSource::None:
                break; // unreachable (checked above)
        }

        if (args.iterations) {
            fractal.SetNumIterations<uint64_t>(*args.iterations);
        }
        if (args.antialiasing) {
            fractal.ResetDimensions(static_cast<size_t>(args.width),
                                    static_cast<size_t>(args.height),
                                    *args.antialiasing);
        }

        if (!fractal.SetRenderAlgorithm(*parsed_alg)) {
            std::cerr << "error: SetRenderAlgorithm failed for "
                      << args.render_algorithm << "\n";
            GlobalCallstacks->FreeCallstacks();
            return 1;
        }

        if (args.perturbation_alg) {
            auto p = ParsePerturbationAlg(*args.perturbation_alg);
            if (!p) {
                std::cerr << "error: unknown perturbation algorithm: "
                          << *args.perturbation_alg << "\n";
                GlobalCallstacks->FreeCallstacks();
                return 2;
            }
            fractal.SetPerturbationAlg(*p);
        }

        if (!args.quiet) {
            std::cout << "Rendering " << args.width << "x" << args.height
                      << " with " << args.render_algorithm << "...\n";
            std::cout.flush();
        }

        // Direct render path (CrummyTest pattern): drain the pool, render
        // synchronously into m_CurIters, then save.
        fractal.GetRenderPool()->Drain();
        fractal.CalcFractal(/*drawFractal=*/true);

        std::wstring base = ToWStringUtf8(args.out_file);
        const std::wstring png_ext = L".png";
        if (base.size() >= png_ext.size() &&
            base.compare(base.size() - png_ext.size(), png_ext.size(), png_ext) == 0) {
            base.resize(base.size() - png_ext.size());
        }

        int rc = fractal.SaveCurrentFractal(base, /*copy_the_iters=*/false);
        if (rc != 0) {
            std::cerr << "error: SaveCurrentFractal returned " << rc << "\n";
        } else if (!args.quiet) {
            std::cout << "Wrote " << args.out_file << "\n";
        }

        GlobalCallstacks->FreeCallstacks();
        return rc;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        GlobalCallstacks->FreeCallstacks();
        return 1;
    }
}
